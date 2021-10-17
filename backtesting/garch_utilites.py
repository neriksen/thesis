import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot
from numpy import dot


def main_loop(out_of_sample, sigmas, epsilons, Qbar, Q_t, dcca, dccb, o, al, be, mu, ka=None):
    initial_value_check(out_of_sample, sigmas, epsilons, Qbar, Q_t, dcca, dccb, o, al, be, mu)
    if ka is not None:  # Model is gjrGARCH11
        print('gjrGARCH11 detected')
    Omega_ts = []
    p = np.size(sigmas, 1)
    for t, r_t in enumerate(out_of_sample.values):
        r_t = np.reshape(r_t, (p, 1))
        # 0. Get current sigma^2
        s_t_squared = np.reshape(sigmas[-1], (p, 1))

        # 1. Calculate current period epsilon
        e_t, e_t_squared, epsilons = calculate_epsilon(epsilons, r_t, mu)

        # 2. Calculate for all assets sigma_^2
        s_t_squared_plus_1 = calculate_sigma(o, al, e_t_squared, be, s_t_squared, ka, e_t)

        # 3. Calculate Var_t, Var_t_inv
        Var_t, Var_t_inv = calculate_Var_t(s_t_squared)
        Var_t_plus_1, Var_t_plus_1_inv = calculate_Var_t(s_t_squared_plus_1)

        # 4. Calculate eta_t
        eta_t = dot(Var_t_inv, e_t)

        # 6. Calculate Q_t_plus_1
        Q_t_plus_1, Q_t_plus_1_s_inv = calculate_Q_t_plus_1(Qbar, dcca, dccb, eta_t, Q_t)

        # 7. Calulate Gamma_t
        Gamma_t_plus_1 = calculate_Gamma_t_plus_1(Q_t_plus_1_s_inv, Q_t_plus_1)

        # 8. Calculate Omega_t+1
        Omega_t_plus_1 = calculate_Omega_t_plus_1(Var_t_plus_1, Gamma_t_plus_1)

        # Storing Omega_t
        Omega_ts.append(Omega_t_plus_1)
        _sigmas = np.append(sigmas, np.reshape(s_t_squared, (1, p)), axis=0)

        # Iterate one period
        Q_t = Q_t_plus_1

    return Omega_ts


def initial_value_check(out_of_sample, sigmas, epsilons, Qbar, Q_t, dcca, dccb, o, al, be, mu, ka=None):
    assert o.shape == al.shape == be.shape == mu.shape
    if ka is not None:
        assert ka.shape == o.shape
    assert np.size(dcca) == 1
    assert np.size(dccb) == 1
    assert Qbar.shape == Q_t.shape
    assert np.size(out_of_sample, axis=1) == np.size(sigmas, axis=1)
    assert np.size(epsilons, axis=0) == np.size(sigmas, axis=0)


def values_from_last_period(epsilons, sigmas):
    e_t = epsilons[-1]
    e_t_squared = np.square(e_t)
    s_t_squared = sigmas[-1]
    return e_t, e_t_squared, s_t_squared


def calculate_Qbar(epsilons, sigmas):
    eta = []
    p = np.size(epsilons, 1)
    for epsilon_t, sigma_t in zip(epsilons, sigmas):
        epsilon_t = np.reshape(epsilon_t, (p, 1))
        Var_t, Var_t_inv = inv(calculate_Var_t(sigma_t))
        eta.append(dot(Var_t_inv, epsilon_t))

    eta = np.array(eta)
    Qbar = 1 / len(epsilons) * sum([dot(eta, eta.T) for eta in eta])
    assert np.size(Qbar, 1) == np.size(epsilons, 1)
    return Qbar


def calculate_epsilon(epsilons, r_t, mu):
    p = np.size(epsilons, 1)
    # Ensure e_t is px1
    e_t = np.reshape(np.array([r_t - mu]).T, (p, 1))
    e_t_squared = np.square(e_t)
    epsilons = np.append(epsilons, e_t.T, axis=0)
    return e_t, e_t_squared, epsilons


def arr_indicator_func(value_arr, indicator_arr):
    indicator_func = lambda x: 1 if x < 0 else 0
    res = np.array([indicator_func(ind) * val for val, ind in zip(value_arr, indicator_arr)])
    return res


def calculate_sigma(o, al, e_t_1_squared, be, s_t_1_squared, ka=None, e_t_1=None):
    if ka is not None:  # in case of a gjrGARCH(1, 1) model
        assert (e_t_1 is not None)
        next_sigma = np.array(
            [o + al * e_t_1_squared + be * s_t_1_squared + ka * arr_indicator_func(e_t_1_squared, e_t_1)])
    else:
        next_sigma = np.array([o + al * e_t_1_squared + be * s_t_1_squared])
    return next_sigma


def calculate_Var_t(s_t_squared):
    Var_t = np.diag(np.ravel(s_t_squared))
    Var_t_inv = inv(Var_t)
    return Var_t, Var_t_inv


def calculate_Q_t_plus_1(Qbar, dcca, dccb, eta_t, Q_t):
    Q_t_plus_1 = np.array(Qbar * (1 - dcca - dccb) + dcca * eta_t * eta_t.T + dccb * Q_t)
    Q_t_plus_1_s_inv = inv(np.diag(np.diag(Q_t_plus_1)))
    return Q_t_plus_1, Q_t_plus_1_s_inv


def calculate_Gamma_t_plus_1(Q_t_plus_1_s_inv, Q_t_plus_1):
    Gamma_t_plus_1 = mdot([Q_t_plus_1_s_inv, Q_t_plus_1, Q_t_plus_1_s_inv])
    return Gamma_t_plus_1


def calculate_Omega_t_plus_1(Var_t_plus_1, Gamma_t_plus_1):
    Omega_t_plus_1 = mdot([Var_t_plus_1, Gamma_t_plus_1, Var_t_plus_1])
    return Omega_t_plus_1
