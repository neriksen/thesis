import numpy as np
from numpy import divide
from numpy.linalg import multi_dot as mdot
from numpy.linalg import inv
from numpy import dot
import matplotlib.pyplot as plt
import pandas as pd
import yfinance
import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import IntVector
from compare_strategies import performance_table
from calibrate_trading_costs import get_gamma_D
pandas2ri.activate()
import garch_utilites as gu
from multiprocessing import Pool
from itertools import starmap


def download_return_data(tickers, start="2008-01-01", end="2021-10-02", save_to_csv=True):
    return_data = yfinance.download(tickers, start=start, end=end)['Adj Close']
    return_data = return_data/return_data.iloc[0]
    return_data = return_data.pct_change().iloc[1:]*100
    return_data = return_data[tickers]
    if save_to_csv:
        return_data.to_csv(os.path.join(os.path.dirname(__file__), '../../data/return_data.csv'), sep=";")
    return return_data


def split_sample(return_data, number_of_out_of_sample_days):
    out_of_sample = return_data.iloc[-number_of_out_of_sample_days:, ]
    in_sample = return_data.iloc[:-number_of_out_of_sample_days, ]
    return out_of_sample, in_sample


def fit_garch_model(len_out_of_sample=0, ugarch_model="sGARCH", garch_order=(1, 1)):
    """
    ugarch_model: One of "sGARCH", "gjrGARCH", not implemented: "eGARCH"
    garch_order: Default: (1, 1)
    """
    assert (ugarch_model in ("sGARCH", "gjrGARCH"))
    garch_order = IntVector(garch_order)
    # Define the R script and load the instance in Python
    r = ro.r
    r['source'](str(os.path.join(os.path.dirname(__file__), '../fitting_mgarch.R')))
    # Load the function we have defined in R.
    fit_mgarch_r = ro.globalenv['fit_mgarch']
    # Fit the MGARCH model and receive the result
    ugarch_dist_model = "std"       # t-distribution for the individual models
    coef, residuals, sigmas = fit_mgarch_r(len_out_of_sample, ugarch_model, ugarch_dist_model, garch_order)
    return coef, residuals, sigmas


def parse_garch_coef(coef, p, model_type):
    """
    Possible modeltypes: sGARCH11, sGARCH10, gjrGARCH11, not implemented: eGARCH11
    """
    assert (model_type in ("sGARCH11", "sGARCH10", "gjrGARCH11"))

    if model_type == "gjrGARCH11": coef_pr_asset = 6
    elif model_type == "sGARCH11": coef_pr_asset = 5
    else:                          coef_pr_asset = 4

    garch_params = np.hsplit(coef[:-3].reshape((p, coef_pr_asset)), coef_pr_asset)
    garch_params = map(np.ravel, garch_params)                              # Flattening to 1d array
    garch_params = map(np.reshape, garch_params, [(p, 1)] * coef_pr_asset)  # Reshaping to px1

    # Unpacking
    if model_type == "gjrGARCH11":
        mu, o, al, be, ka, shape = garch_params

    elif model_type == "sGARCH11":
        mu, o, al, be, shape = garch_params
        ka = None

    else:
        mu, o, al, shape = garch_params
        be = np.zeros(mu.shape)
        ka = None

    # Joint parameters
    dcca = coef[-3]
    dccb = coef[-2]
    joint_shape = coef[-1]

    params_dict = {
        "mu": mu,
        "omega": o,
        "alpha": al,
        "beta": be,
        "kappa": ka,
        "dcca": dcca,
        "dccb": dccb,
        "shape": shape,
        "joint_shape": joint_shape
    }

    return params_dict


def calc_weights_garch_with_trading_cost_multi_helper(Omega_t_plus_1, gamma_D=None):
    Avv_guess = Omega_t_plus_1
    Avv, Av1 = gu.calc_Avs(Omega_t=Omega_t_plus_1, gamma_D=gamma_D, Avv_guess=Avv_guess)
    print("Solved Avv")
    return Avv, Av1


def calc_weights_garch_with_trading_cost(Omega_t_plus1s, gamma_D=None):
    assert isinstance(Omega_t_plus1s, (list, np.ndarray))
    p = len(Omega_t_plus1s[0])
    ones = np.ones((p, 1))
    if gamma_D is None:
        gamma_D = get_gamma_D("median")

    # Let the first weight be similar to garch_no_trading costs:
    v_1 = divide(dot(inv(Omega_t_plus1s[0]), ones), mdot([ones.T, inv(Omega_t_plus1s[0]), ones]))
    v_t_1 = v_1
    v_ts = []

    print(f"Solving problem with trading costs. gamma_D = {gamma_D}")
    # Calculate Avv and Av1 matricies for each Omega
    multi_args = [(Omega_t, gamma_D) for Omega_t in Omega_t_plus1s]
    with Pool() as p:
        res = p.starmap(calc_weights_garch_with_trading_cost_multi_helper, multi_args)

    for t, (Omega, (Avv, Av1)) in enumerate(zip(Omega_t_plus1s, res)):
        print(t)
        if t == 0:
            v_ts.append(np.ravel(v_1))
        else:
            aim_t = mdot([inv(Avv), Av1, ones])
            aim_t = aim_t/np.sum(aim_t)

            modifier = mdot([inv(gamma_D*Omega), Avv, (v_t_1-aim_t)])
            v_t = v_t_1 + modifier

            v_ts.append(np.ravel(v_t))
            v_t_1 = v_t
    return v_ts


def calc_weights_garch_no_trading_cost(Omega_ts):
    assert isinstance(Omega_ts, (list, np.ndarray))
    p = len(Omega_ts[0])
    ones = np.ones((p, 1))
    v_t = [np.ravel(divide(dot(inv(Omega), ones), mdot([ones.T, inv(Omega), ones]))) for Omega in Omega_ts]
    return v_t


def calc_Omega_ts(out_of_sample_returns, in_sample_returns, in_sample_sigmas, in_sample_residuals, **kw):
    Qbar = gu.calc_Qbar(in_sample_residuals, in_sample_sigmas)
    Q_t = Qbar      # Qbar is the same as Q_t at the start of the out-of-sample period

    Omega_ts = gu.main_loop(out_of_sample_returns=out_of_sample_returns, in_sample_returns=in_sample_returns,
                        sigmas=in_sample_sigmas, epsilons=in_sample_residuals, Qbar=Qbar, Q_t=Q_t, **kw)
    return Omega_ts


def unconditional_weights(tickers, start="2008-01-01", end="2021-10-02", number_of_out_of_sample_days=250*4):
    return_data=download_return_data(tickers, start, end, save_to_csv=True)
    out_of_sample, _ = split_sample(return_data, number_of_out_of_sample_days)
    Omega_uncond=out_of_sample.cov() #Use only the sample that we ant to test on
    ones = np.ones((len(Omega_uncond), 1))
    weights = np.ravel(divide(dot(inv(Omega_uncond), ones), mdot([ones.T, inv(Omega_uncond), ones])))
    weights = pd.DataFrame(np.full(out_of_sample.shape, weights), columns=tickers, index=out_of_sample.index)
    return weights


def split_fit_parse(tickers, start, end, number_of_out_of_sample_days, model_type):
    print(tickers)
    garch_type = model_type[:-2]
    garch_order = IntVector((model_type[-2], model_type[-1]))

    return_data = download_return_data(tickers, start, end, True)

    # Determining dimensions
    T, p = return_data.shape
    out_of_sample, in_sample = split_sample(return_data=return_data,
                                            number_of_out_of_sample_days=number_of_out_of_sample_days)

    # Fit model
    coef, residuals, sigmas = fit_garch_model(len_out_of_sample=number_of_out_of_sample_days,
                                              ugarch_model=garch_type, garch_order=garch_order)

    # Parse variables
    params_dict = parse_garch_coef(coef=coef, p=p, model_type=model_type)
    return out_of_sample, in_sample, sigmas, residuals, params_dict


def garch_no_trading_cost(tickers, start="2008-01-01", end="2021-10-02", number_of_out_of_sample_days=250*4,
                          model_type="sGARCH11"):
    """
    tickers: ["ticker", "ticker", ..., "ticker"]
    start: "yyyy-m-d"
    end: "yyyy-m-d"
    model_type: One of "sGARCH11", "sGARCH10", "gjrGARCH11", not implemented = "eGARCH11"
    """
    assert (model_type in ("sGARCH11", "sGARCH10", "gjrGARCH11"))
    out_of_sample, in_sample, sigmas, residuals, params_dict = split_fit_parse(tickers, start, end,
                                                                               number_of_out_of_sample_days, model_type)

    Omega_ts = calc_Omega_ts(out_of_sample_returns=out_of_sample, in_sample_returns=in_sample,
                             in_sample_sigmas=sigmas, in_sample_residuals=residuals, **params_dict)
    # Generating weights
    v_t = calc_weights_garch_no_trading_cost(Omega_ts)
    v_t = pd.DataFrame(v_t, columns=tickers, index=out_of_sample.index)

    return v_t, out_of_sample, in_sample, Omega_ts


def garch_with_trading_cost(tickers, start="2008-01-01", end="2021-10-02", number_of_out_of_sample_days=250*4,
                          model_type="sGARCH11", gamma_D=None):
    """
    tickers: ["ticker", "ticker", ..., "ticker"]
    start: "yyyy-m-d"
    end: "yyyy-m-d"
    model_type: One of "sGARCH11", "sGARCH10", "gjrGARCH11", not implemented = "eGARCH11"
    """
    assert (model_type in ("sGARCH11", "sGARCH10", "gjrGARCH11"))
    out_of_sample, in_sample, sigmas, residuals, params_dict = split_fit_parse(tickers, start, end,
                                                                               number_of_out_of_sample_days, model_type)

    Omega_ts = calc_Omega_ts(out_of_sample_returns=out_of_sample, in_sample_returns=in_sample,
                             in_sample_sigmas=sigmas, in_sample_residuals=residuals, **params_dict)
    # Generating weights
    v_t = calc_weights_garch_with_trading_cost(Omega_ts, gamma_D)
    v_t = pd.DataFrame(v_t, columns=tickers, index=out_of_sample.index)

    return v_t, out_of_sample, in_sample, Omega_ts


def multiprocessing_helper(gamma_D, Omega_ts, portfolio_value, tickers, out_of_sample):
    v_t = calc_weights_garch_with_trading_cost(Omega_ts, gamma_D=gamma_D)
    v_t = pd.DataFrame(v_t, columns=tickers, index=out_of_sample.index)
    cum_returns, perf_table = performance_table(v_t, out_of_sample, Omega_ts, portfolio_value=portfolio_value)
    res = perf_table.loc[['GARCH TC', 'Equal_weight TC', 'BnH TC']][['Ann. Sharpe ratio']].values
    res = np.append(np.array([[gamma_D]]), res)
    print(res)
    return res


def test_gamma_D_params(tickers, start="2008-01-01", end="2021-10-02", number_of_out_of_sample_days=250*4,
                          model_type="sGARCH11", portfolio_value=1e9, gamma_start=1e-13, gamma_end=1e-2, gamma_num=20):

    gamma_Ds = np.geomspace(gamma_start, gamma_end, gamma_num)
    """
    tickers: ["ticker", "ticker", ..., "ticker"]
    start: "yyyy-m-d"
    end: "yyyy-m-d"
    model_type: One of "sGARCH11", "sGARCH10", "gjrGARCH11", not implemented = "eGARCH11"
    """
    assert (model_type in ("sGARCH11", "sGARCH10", "gjrGARCH11"))
    out_of_sample, in_sample, sigmas, residuals, params_dict = split_fit_parse(tickers, start, end,
                                                                               number_of_out_of_sample_days, model_type)

    Omega_ts = calc_Omega_ts(out_of_sample_returns=out_of_sample, in_sample_returns=in_sample,
                             in_sample_sigmas=sigmas, in_sample_residuals=residuals, **params_dict)
    # Generating multiprocessing job:
    multi_args = [(gamma_D, Omega_ts, portfolio_value, tickers, out_of_sample) for gamma_D in gamma_Ds]
    # Generating weights
    #with Pool() as p:
    sharpe_ratios = list(starmap(multiprocessing_helper, multi_args))

    return sharpe_ratios


if __name__ == '__main__':
    portfolio_value = 1e9
    sharpes = test_gamma_D_params(['IVV', 'TLT']
                                  , number_of_out_of_sample_days=1000, model_type="sGARCH11",
                                                                        portfolio_value=1e9,
                                  gamma_start=1e-7, gamma_end=1e-2, gamma_num=50)
    #v_t_s, out_of_sample, in_sample, Omega_ts = garch_with_trading_cost(['IVV', 'TLT', 'EEM'],
    #                                                      number_of_out_of_sample_days=1000, model_type="sGARCH11")
    #v_t_s, out_of_sample, in_sample, Omega_ts = garch_with_trading_cost(['EEM', 'IVV', 'IEV', 'IXN', 'IYR', 'IXG', 'EXI', 'GC=F', 'BZ=F', 'HYG', 'TLT'],
    #                                                                    model_type="sGARCH11")
    #
    # v_t.to_csv('v_t.csv')
    # out_of_sample.to_csv('out_of_sample.csv')
    # in_sample.to_csv('in_sample.csv')
    # v_t = pd.read_csv('v_t.csv', index_col=0)
    # out_of_sample = pd.read_csv('out_of_sample.csv', index_col=0)
    # in_sample = pd.read_csv('in_sample.csv', index_col=0)
    #print(sharpes)
    #cum_returns, perf_table = performance_table(v_t_s, out_of_sample, Omega_ts, portfolio_value=portfolio_value)
    #print(perf_table)
    for sharpe in sharpes:
        print(sharpe)
    sharpes = pd.DataFrame(sharpes, columns=['gamma_D', 'GARCH TC', 'Equal_weight TC', 'BnH TC'])
    sharpes.set_index('gamma_D', drop=True, inplace=True)
    plt.plot(sharpes)
    #plt.plot(cum_returns)
    #plt.plot(v_t_s)
    plt.xscale('log')
    plt.show()
