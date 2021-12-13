import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot
from numpy import dot
from scipy.optimize import minimize


def main_loop(out_of_sample_returns, in_sample_returns, sigmas, epsilons, Qbar, Q_t, **kw):
    initial_value_check(out_of_sample_returns, sigmas, epsilons, Qbar, Q_t, **kw)
    Omega_ts = []
    p = np.size(sigmas, 1)
    periods_to_run = len(out_of_sample_returns)-1

    # For t = T to M, where T is last in-sample period
    # t period values are always known
    # t+1 period values are always forecasted
    # We add last period of the in-sample returns to the out-of-sample returns, to start the process
    last_in_sample = in_sample_returns.iloc[-1]
    returns = out_of_sample_returns.append(last_in_sample)\
        .sort_index()

    # Let t+1 be first out-of-sample period
    # Then the first Omega forecast saved is Omega_{t+1|T}, where T is the last in-sample period
    # The forecast is constructed on T-measurable variables only, sigma, epsilon and r_t.

    for t, r_t in enumerate(returns.values):
        r_t = np.reshape(r_t, (p, 1))

        # Get variables from current period for all N assets
        s_t_Sq = np.reshape(sigmas[-1], (p, 1))
        e_t, e_t_Sq, epsilons = calc_epsilon(epsilons, r_t, kw['mu'])
        s_t_Sq_plus_1 = calc_sigma(e_t_Sq, s_t_Sq, e=e_t, **kw)

        Var_t, Var_t_inv = calc_Var_t(s_t_Sq)
        eta_t = dot(Var_t_inv, e_t)

        # Construct Omega_(t+1) from F_t measurable variables
        Var_t_plus_1, Var_t_plus_1_inv = calc_Var_t(s_t_Sq_plus_1)

        # 6. Calculate Q_(t+1)
        Q_t_plus_1 = calc_Q_t_plus_1(Qbar=Qbar, Q_t=Q_t, eta_t=eta_t, **kw)

        # 7. Calulate Gamma_t
        Gamma_t_plus_1 = calc_Gamma_t_plus_1(Q_t_plus_1=Q_t_plus_1)

        # 8. Calculate Omega_(t+1)
        Omega_t_plus_1 = calc_Omega_t_plus_1(Var_t_plus_1=Var_t_plus_1, Gamma_t_plus_1=Gamma_t_plus_1)
        assert np.allclose(np.diag(Omega_t_plus_1), np.ravel(s_t_Sq_plus_1), rtol=1e-3)

        # Storing Omega_t and sigmas
        Omega_ts.append(Omega_t_plus_1)
        sigmas = np.append(sigmas, np.reshape(s_t_Sq_plus_1, (1, p)), axis=0)

        # Iterate one period
        Q_t = Q_t_plus_1

        if t == periods_to_run:     # Quit one period before loop runs out. No reason to calc Omega_t+1 for last return
            break

    # Time index for Omega_ts is the same as returns out of sample
    # We define the timestamp as t+1 when Omega = Omega_{t+1|T}
    Omega_ts = [[ind, Omega] for ind, Omega in zip(out_of_sample_returns.index, Omega_ts)]

    return Omega_ts


def initial_value_check(out_of_sample, sigmas, epsilons, Qbar, Q_t, **kw):
    assert kw['omega'].shape == kw['alpha'].shape == kw['beta'].shape == kw['mu'].shape
    if kw['kappa'] is not None:
        assert kw['kappa'].shape == kw['omega'].shape
    assert np.size(kw['dcca']) == 1
    assert np.size(kw['dccb']) == 1
    assert Qbar.shape == Q_t.shape
    assert np.size(out_of_sample, axis=1) == np.size(sigmas, axis=1)
    assert np.size(epsilons, axis=0) == np.size(sigmas, axis=0)


def values_from_last_period(epsilons, sigmas):
    e_t = epsilons[-1]
    e_t_Sq = np.square(e_t)
    s_t_Sq = sigmas[-1]
    return e_t, e_t_Sq, s_t_Sq


def calc_Qbar(epsilons, sigmas, **kw):
    eta = np.empty(epsilons.shape)
    p = np.size(epsilons, 1)
    for i, (epsilon_t, sigma_t) in enumerate(zip(epsilons, sigmas)):
        epsilon_t = np.reshape(epsilon_t, (p, 1))
        Var_t, Var_t_inv = calc_Var_t(sigma_t)
        eta_dot = dot(Var_t_inv, epsilon_t).T
        eta[i] = eta_dot

    Qbar = 1 / len(epsilons) * sum([dot(np.reshape(eta, (p, 1)), np.reshape(eta, (1, p))) for eta in eta])

    #Calculating Qbar as the correlation matrix of the residuals which is theoretically identical to 1/T\sum(\eta\eta')
    #Qbar=pd.DataFrame(epsilons).corr().values

    # Regularize Qbar estimate by 50%
    regularizer = kw.get("regularizer", 0.5)
    ones = np.identity(len(Qbar))
    Qbar = ones * regularizer + Qbar * (1 - regularizer)
    assert np.size(Qbar, 1) == np.size(epsilons, 1)
    return Qbar


def calc_epsilon(epsilons, returns, mu):
    p = np.size(epsilons, 1)
    # Ensure e_t is px1
    e = np.reshape(np.array([returns - mu]).T, (p, 1))
    e_Sq = np.square(e)
    epsilons = np.append(epsilons, e.T, axis=0)
    return e, e_Sq, epsilons


def indicator_func(x):
    return 1 if x < 0 else 0


def arr_indicator_func(value_arr, indicator_arr):
    res = np.array([indicator_func(ind) * val for val, ind in zip(value_arr, indicator_arr)])
    return res


def calc_sigma(e_Sq, s_Sq, e=None, **kw):
    if kw['kappa'] is not None:  # in case of a gjrGARCH(1, 1) model
        assert (e is not None)
        next_sigma = np.array(
            [kw['omega'] + kw['alpha'] * e_Sq +
             kw['beta'] * s_Sq + kw['kappa'] * arr_indicator_func(e_Sq, e)])
    else:
        next_sigma = np.array([kw['omega'] + kw['alpha'] * e_Sq + kw['beta'] * s_Sq])
    return next_sigma


def calc_Var_t(s_t_Sq):
    Var_t = np.diag(np.ravel(np.sqrt(s_t_Sq)))
    Var_t_inv = inv(Var_t)
    return Var_t, Var_t_inv


def calc_Q_t_plus_1(Qbar, Q_t, eta_t, **kw):
    Q_t_plus_1 = np.array(Qbar * (1 - kw['dcca'] - kw['dccb']) + kw['dcca'] * dot(eta_t, eta_t.T) + kw['dccb'] * Q_t)
    return Q_t_plus_1


def calc_Gamma_t_plus_1(Q_t_plus_1):
    Q_t_plus_1_diag_inv = inv(np.sqrt(np.diag(np.diag(Q_t_plus_1))))     # Tak til Bongen
    Gamma_t_plus_1 = mdot([Q_t_plus_1_diag_inv, Q_t_plus_1, Q_t_plus_1_diag_inv])
    assert np.allclose(np.diag(Gamma_t_plus_1), np.ones(np.diag(Gamma_t_plus_1).shape))
    return Gamma_t_plus_1


def calc_Omega_t_plus_1(Var_t_plus_1, Gamma_t_plus_1):
    Omega_t_plus_1 = mdot([Var_t_plus_1, Gamma_t_plus_1, Var_t_plus_1])
    return Omega_t_plus_1


def calc_Avs(Omega_t, gamma_D, Avv_guess):
    # Setup
    global Lambda_t, rho, ones, p
    rho = 1 - np.exp(-0.02 / 250)
    Lambda_t = (gamma_D * Omega_t) / (1 - rho)
    p = Omega_t.shape[0]
    ones = np.ones((p, 1))
    Avv = calc_Avv(Omega_t, Avv_guess)
    Av1 = calc_Av1()
    return Avv, Av1


def calc_Av1():
    fraction = np.divide(dot(Lambda_t, J_t_inv), mdot([ones.T, J_t_inv, ones]))
    big_parenthesis = inv((1 - rho) ** (-1) - dot(Lambda_t, J_t_inv) + mdot([fraction, ones, ones.T, J_t_inv]))
    Av1 = dot(big_parenthesis, fraction)
    return Av1


def calc_Avv(Omega_t, Avv_guess):
    if Avv_guess is None:
        Avv_guess = np.full((p, p), 1/p)
    # Numerical solver for Avv
    options = {"maxiter": 4000, "ftol": 11e-13}
    res =minimize(fun=compare_sides,
                  args=(Omega_t),
                  method='SLSQP',
                  x0=Avv_guess,
                  options=options,
                  tol=1e-13)

    # Enforce symmetric Avv matrix
    Avv = np.reshape(res.x, (p, p))
    Avv = np.triu(Avv)
    Avv = Avv.T + Avv - np.diag(np.diag(Avv))
    return Avv


def compare_sides(Avv, Omega_t):
    global J_t_inv
    Avv = np.reshape(Avv, (p, p))
    Avv = np.triu(Avv)
    Avv = Avv.T + Avv - np.diag(np.diag(Avv))    # Enforcing symmetric Avv matrix
    J_t_inv = inv(Omega_t + Avv + Lambda_t)
    Left = LHS(Avv)
    Right = RHS()
    diff = np.trace(dot((Left-Right).T, (Left-Right)))
    return np.sum(diff)


def LHS(Avv):
    Avv = np.reshape(Avv, (p, p))
    return (1/(1-rho))*Avv


def RHS():
    return mdot([Lambda_t, J_t_inv, ones, 1/(mdot([ones.T, J_t_inv, ones])), ones.T, J_t_inv, Lambda_t])-mdot([Lambda_t, J_t_inv, Lambda_t]) - Lambda_t


if __name__ == '__main__':
    Omega_t = np.array([[1.65729471, -0.53047418,  2.07542849],
        [-0.53047418,  0.90200802, -0.6495747],
        [ 2.07542849, -0.6495747, 3.57961394]])

    gamma_Ds = np.linspace(1e-7, 100, 100)
    Avv_sum = []

    for gamma_D in gamma_Ds:
        Avv, Av1 = calc_Avs(Omega_t, gamma_D, Omega_t)
        Avv_sum.append(np.sum(Avv))

    import matplotlib.pyplot as plt
    plt.plot(pd.DataFrame(Avv_sum, index=pd.Index(gamma_Ds)))
    plt.show()
