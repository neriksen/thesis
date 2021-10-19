import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot
from numpy import dot


def main_loop(out_of_sample_returns, sigmas, epsilons, Qbar, Q_t, **kw):
    initial_value_check(out_of_sample_returns, sigmas, epsilons, Qbar, Q_t, **kw)
    if kw['kappa'] is not None:  # Model is gjrGARCH11
        print('gjrGARCH11 detected')
    Omega_ts = []
    p = np.size(sigmas, 1)

    # For t = T to M, where T is first in-sample period
    # t period values are always known
    # t+1 period values are always forecasted

    for t, r_t in enumerate(out_of_sample_returns.values):
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
        Q_t_plus_1, Q_t_plus_1_s_inv = calc_Q_t_plus_1(Qbar=Qbar, Q_t=Q_t, eta_t=eta_t, **kw)

        # 7. Calulate Gamma_t
        Gamma_t_plus_1 = calc_Gamma_t_plus_1(Q_t_plus_1_s_inv=Q_t_plus_1_s_inv, Q_t_plus_1=Q_t_plus_1)

        # 8. Calculate Omega_(t+1)
        Omega_t_plus_1 = calc_Omega_t_plus_1(Var_t_plus_1=Var_t_plus_1, Gamma_t_plus_1=Gamma_t_plus_1)

        # Storing Omega_t and sigmas
        Omega_ts.append(Omega_t_plus_1)
        sigmas = np.append(sigmas, np.reshape(s_t_Sq_plus_1, (1, p)), axis=0)

        # Iterate one period
        Q_t = Q_t_plus_1

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


def calc_Qbar(epsilons, sigmas):
    eta = np.empty(epsilons.shape)
    p = np.size(epsilons, 1)
    for i, (epsilon_t, sigma_t) in enumerate(zip(epsilons, sigmas)):
        epsilon_t = np.reshape(epsilon_t, (p, 1))
        Var_t, Var_t_inv = inv(calc_Var_t(sigma_t))
        eta_dot = dot(Var_t_inv, epsilon_t).T
        eta[i] = eta_dot

    Qbar = 1 / len(epsilons) * sum([dot(np.reshape(eta, (p, 1)), np.reshape(eta, (1, p))) for eta in eta])
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
    Var_t = np.diag(np.ravel(s_t_Sq))
    Var_t_inv = inv(Var_t)
    return Var_t, Var_t_inv


def calc_Q_t_plus_1(Qbar, Q_t, eta_t, **kw):
    Q_t_plus_1 = np.array(Qbar * (1 - kw['dcca'] - kw['dccb']) + kw['dcca'] * eta_t * eta_t.T + kw['dccb'] * Q_t)
    Q_t_plus_1_inv = inv(np.diag(np.diag(Q_t_plus_1)))
    return Q_t_plus_1, Q_t_plus_1_inv


def calc_Gamma_t_plus_1(Q_t_plus_1_s_inv, Q_t_plus_1):
    Gamma_t_plus_1 = mdot([Q_t_plus_1_s_inv, Q_t_plus_1, Q_t_plus_1_s_inv])
    return Gamma_t_plus_1


def calc_Omega_t_plus_1(Var_t_plus_1, Gamma_t_plus_1):
    Omega_t_plus_1 = mdot([Var_t_plus_1, Gamma_t_plus_1, Var_t_plus_1])
    return Omega_t_plus_1
