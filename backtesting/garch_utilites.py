import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot
from numpy import dot


def main_loop(out_of_sample, sigmas, epsilons, Qbar, Q_t, **kw):
    initial_value_check(out_of_sample, sigmas, epsilons, Qbar, Q_t, **kw)
    if kw['kappa'] is not None:  # Model is gjrGARCH11
        print('gjrGARCH11 detected')
    Omega_ts = []
    p = np.size(sigmas, 1)
    for t, r_t in enumerate(out_of_sample.values):
        r_t = np.reshape(r_t, (p, 1))
        # 0. Get current sigma^2
        s_t_Sq = np.reshape(sigmas[-1], (p, 1))

        # 1. Calculate current period epsilon
        e_t, e_t_Sq, epsilons = calc_epsilon(epsilons, r_t, kw['mu'])

        # 2. Calculate for all assets sigma_^2
        s_t_Sq_plus_1 = calc_sigma(e_t_Sq, s_t_Sq, e_t_1=e_t, **kw)

        # 3. Calculate Var_t, Var_t_inv
        Var_t, Var_t_inv = calc_Var_t(s_t_Sq)
        Var_t_plus_1, Var_t_plus_1_inv = calc_Var_t(s_t_Sq_plus_1)

        # 4. Calculate eta_t
        eta_t = dot(Var_t_inv, e_t)

        # 6. Calculate Q_(t+1)
        Q_t_plus_1, Q_t_plus_1_s_inv = calc_Q_t_plus_1(Qbar, Q_t, eta_t, **kw)

        # 7. Calulate Gamma_t
        Gamma_t_plus_1 = calc_Gamma_t_plus_1(Q_t_plus_1_s_inv, Q_t_plus_1)

        # 8. Calculate Omega_(t+1)
        Omega_t_plus_1 = calc_Omega_t_plus_1(Var_t_plus_1, Gamma_t_plus_1)

        # Storing Omega_t
        Omega_ts.append(Omega_t_plus_1)
        _sigmas = np.append(sigmas, np.reshape(s_t_Sq, (1, p)), axis=0)

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
    eta = []
    p = np.size(epsilons, 1)
    for epsilon_t, sigma_t in zip(epsilons, sigmas):
        epsilon_t = np.reshape(epsilon_t, (p, 1))
        Var_t, Var_t_inv = inv(calc_Var_t(sigma_t))
        eta.append(dot(Var_t_inv, epsilon_t))

    eta = np.array(eta)
    Qbar = 1 / len(epsilons) * sum([dot(eta, eta.T) for eta in eta])
    assert np.size(Qbar, 1) == np.size(epsilons, 1)
    return Qbar


def calc_epsilon(epsilons, r_t, mu):
    p = np.size(epsilons, 1)
    # Ensure e_t is px1
    e_t = np.reshape(np.array([r_t - mu]).T, (p, 1))
    e_t_Sq = np.square(e_t)
    epsilons = np.append(epsilons, e_t.T, axis=0)
    return e_t, e_t_Sq, epsilons


def indicator_func(x):
    return 1 if x < 0 else 0


def arr_indicator_func(value_arr, indicator_arr):
    res = np.array([indicator_func(ind) * val for val, ind in zip(value_arr, indicator_arr)])
    return res


def calc_sigma(e_t_1_Sq, s_t_1_Sq, e_t_1=None, **kw):
    if kw['kappa'] is not None:  # in case of a gjrGARCH(1, 1) model
        assert (e_t_1 is not None)
        next_sigma = np.array(
            [kw['omega'] + kw['alpha'] * e_t_1_Sq +
             kw['beta'] * s_t_1_Sq + kw['kappa'] * arr_indicator_func(e_t_1_Sq, e_t_1)])
    else:
        next_sigma = np.array([kw['omega'] + kw['alpha'] * e_t_1_Sq + kw['beta'] * s_t_1_Sq])
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
