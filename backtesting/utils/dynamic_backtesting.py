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
from rpy2.robjects import StrVector
from compare_strategies import performance_table
from calibrate_trading_costs import get_gamma_D
pandas2ri.activate()
import garch_utilites as gu
from multiprocessing import Pool


def download_return_data(tickers, start="2008-01-01", end="2021-10-02"):
    return pd.read_csv('../../data/return_data_stable.csv', sep=";", index_col=0).loc[start:end, tickers]


def remove_Omega_timestamp(Omega_ts):
    return [Omega[1] for Omega in Omega_ts]


def split_sample(return_data, number_of_out_of_sample_days):
    out_of_sample = return_data.iloc[-number_of_out_of_sample_days:, ]
    in_sample = return_data.iloc[:-number_of_out_of_sample_days, ]
    return out_of_sample, in_sample


def fit_garch_model(tickers, len_out_of_sample=0, ugarch_model="sGARCH", garch_order=(1, 1)):
    """
    ugarch_model: One of "sGARCH", "gjrGARCH", not implemented: "eGARCH"
    garch_order: Default: (1, 1)
    """
    assert (ugarch_model in ("sGARCH", "gjrGARCH"))
    tickers = StrVector(tickers)
    garch_order = IntVector(garch_order)
    # Define the R script and load the instance in Python
    r = ro.r
    r['source'](str(os.path.join(os.path.dirname(__file__), '../fitting_mgarch.R')))
    # Load the function we have defined in R.
    fit_mgarch_r = ro.globalenv['fit_mgarch']
    # Fit the MGARCH model and receive the result
    ugarch_dist_model = "std"       # t-distribution for the individual models
    coef, residuals, sigmas = fit_mgarch_r(tickers, len_out_of_sample, ugarch_model, ugarch_dist_model, garch_order)
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


def numerical_solver_multi(Omega_t_plus_1, gamma_D):
    Avv_guess = np.multiply(Omega_t_plus_1, 1e-7)
    Avv, Av1 = gu.calc_Avs(Omega_t=Omega_t_plus_1, gamma_D=gamma_D, Avv_guess=Avv_guess)
    return Avv, Av1


def calc_weights_loop(Avv, Av1, Omega_t_plus1s, tuning_gamma_D, out_of_sample_returns_pct):
    v_ts = []
    aim_ts = []
    modifiers = []
    p = len(Omega_t_plus1s[0][1])
    ones = np.ones((p, 1))
    out_of_sample_returns = out_of_sample_returns_pct.divide(100)

    # Let the first weight in period T (last out-of-sample period) be similar to garch_no_trading costs:
    v_1 = divide(dot(inv(Omega_t_plus1s[0][1]), ones), mdot([ones.T, inv(Omega_t_plus1s[0][1]), ones]))
    v_ts.append(np.ravel(v_1))
    v_t_1 = v_1
    for t, (Omega, Avv, Av1) in enumerate(zip(Omega_t_plus1s, Avv, Av1)):
        # It is crucial that r_t is in the same period as v_t because we adjust v_{t-1} to include the return of r_t
        # The resulting list of weights begins in period T and ends in M

        Omega_value = Omega[1]
        if t != 0:
            r_t = out_of_sample_returns.iloc[t-1]
            r_t_value = np.reshape(r_t.values, (p, 1))
        else:
            # Because we don't have any return for period T since the portfolio is first established at end of day
            # on the first day the portfolio is constructed
            r_t_value = np.zeros((p, 1))

        # First weight calculated here is for period T+1, so what the investor rebalances to
        # at the end of day T+1, ie. the first day where the strategy is live
        aim_t = mdot([inv(Avv), Av1, ones])
        aim_t = aim_t/np.sum(aim_t)

        v_t_1_mod = np.divide(np.multiply(v_t_1, 1+r_t_value), 1+dot(v_t_1.T, r_t_value))
        modifier = mdot([inv(tuning_gamma_D*Omega_value), Avv, (v_t_1_mod-aim_t)])
        v_t = v_t_1_mod + modifier
        v_t = v_t/np.sum(v_t)

        v_ts.append(np.ravel(v_t))
        aim_ts.append(np.ravel(aim_t))
        modifiers.append(np.ravel(modifier))
        v_t_1 = v_t
    return v_ts, aim_ts, modifiers


def calc_weights_garch_with_trading_cost(Omega_t_plus1s, out_of_sample_returns_pct, tuning_gamma_D=None):
    # T is last period of in-sample time span
    # Omega_t_plus1s first value is T-measurable, and is a T+1 forecast
    # out_of_sample_returns_pct first value is T+1-measurable

    assert isinstance(Omega_t_plus1s, (list, np.ndarray))

    gamma_D = get_gamma_D("median")
    if tuning_gamma_D is None:
        tuning_gamma_D = gamma_D

    print(f"Solving problem with trading costs. gamma_D = {gamma_D}, tuning gamma_D = {tuning_gamma_D}")
    # Calculate Avv and Av1 matricies for each Omega
    Omega_values = remove_Omega_timestamp(Omega_t_plus1s)
    multi_args = [(Omega, gamma_D) for Omega in Omega_values]
    with Pool() as multi:
        res = multi.starmap(numerical_solver_multi, multi_args)

    Avv = [x[0] for x in res]
    Av1 = [x[1] for x in res]
    v_ts , aim_ts, modifiers = calc_weights_loop(Avv, Av1, Omega_t_plus1s, tuning_gamma_D, out_of_sample_returns_pct)

    return v_ts , aim_ts, modifiers


def calc_weights_garch_no_trading_cost(Omega_ts):
    assert isinstance(Omega_ts, (list, np.ndarray))
    Omega_values = remove_Omega_timestamp(Omega_ts)
    p = len(Omega_values[0])
    ones = np.ones((p, 1))
    v_t = [np.ravel(divide(dot(inv(Omega_values[0]), ones), mdot([ones.T, inv(Omega_values[0]), ones])))] # Add first weight twice to adhere to standard in garch with trading cost
    for Omega in Omega_values:
        v_t.append(np.ravel(divide(dot(inv(Omega), ones), mdot([ones.T, inv(Omega), ones]))))
    return v_t


def calc_Omega_ts(out_of_sample_returns, in_sample_returns, in_sample_sigmas, in_sample_residuals, **kw):
    Qbar = gu.calc_Qbar(in_sample_residuals, in_sample_sigmas)
    Q_t = Qbar      # Qbar is the same as Q_t at the start of the out-of-sample period

    Omega_ts = gu.main_loop(out_of_sample_returns=out_of_sample_returns, in_sample_returns=in_sample_returns,
                        sigmas=in_sample_sigmas, epsilons=in_sample_residuals, Qbar=Qbar, Q_t=Q_t, **kw)
    return Omega_ts


def unconditional_weights(tickers, start="2008-01-01", end="2021-10-02", number_of_out_of_sample_days=250*4):
    return_data = download_return_data(tickers, start, end)
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

    return_data = download_return_data(tickers, start, end)

    # Determining dimensions
    T, p = return_data.shape
    out_of_sample, in_sample = split_sample(return_data=return_data,
                                            number_of_out_of_sample_days=number_of_out_of_sample_days)

    # Fit model
    coef, residuals, sigmas = fit_garch_model(tickers=tickers, len_out_of_sample=number_of_out_of_sample_days,
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
    weight_index = in_sample.index[[-1]].union(out_of_sample.index)
    v_t = pd.DataFrame(v_t, columns=tickers, index=weight_index)

    return v_t, out_of_sample, in_sample, Omega_ts


def garch_with_trading_cost(tickers, start="2008-01-01", end="2021-10-02", number_of_out_of_sample_days=250*4,
                          model_type="sGARCH11", tuning_gamma_D=None):
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
    v_t,_,_ = calc_weights_garch_with_trading_cost(Omega_ts, out_of_sample, tuning_gamma_D=tuning_gamma_D)
    # Construct index for weights that start in period T (last in-sample period)
    weight_index = in_sample.index[[-1]].union(out_of_sample.index)
    v_t = pd.DataFrame(v_t, columns=tickers, index=weight_index)

    return v_t, out_of_sample, in_sample, Omega_ts


def multiprocessing_helper(gamma_D_tuning, Omega_ts, portfolio_value, tickers, out_of_sample, in_sample, Avv, Av1):
    v_t,_,_ = calc_weights_loop(Avv=Avv, Av1=Av1, Omega_t_plus1s=Omega_ts, tuning_gamma_D=gamma_D_tuning, out_of_sample_returns_pct=out_of_sample)

    weight_index = in_sample.index[[-1]].union(out_of_sample.index)
    v_t = pd.DataFrame(v_t, columns=tickers, index=weight_index)
    cum_returns, perf_table = performance_table(weights=v_t, returns_pct=out_of_sample, Omega_ts=Omega_ts,
                                                portfolio_value=portfolio_value)
    sharpe = perf_table.loc[['GARCH TC', 'Equal_weight TC', 'BnH TC']][['Ann. Sharpe ratio']].values
    sharpe = np.append(np.array([[gamma_D_tuning]]), sharpe)

    std = perf_table.loc[['GARCH TC', 'Equal_weight TC', 'BnH TC']][['Ann. standard deviation']].values
    std = np.append(np.array([[gamma_D_tuning]]), std)

    print(sharpe)
    return sharpe, std


def test_gamma_D_params(tickers, start="2008-01-01", end="2021-10-02", number_of_out_of_sample_days=250*4,
                          model_type="sGARCH11", portfolio_value=1e9, gamma_start=1e-8, gamma_end=1e-2, gamma_num=30):

    gamma_Ds_tuning = np.geomspace(gamma_start, gamma_end, gamma_num)
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
    # Generating arguments for list comprehension:
    gamma_D = get_gamma_D("median")
    Omega_t_values = remove_Omega_timestamp(Omega_ts)
    multi_args = [(Omega_t, gamma_D) for Omega_t in Omega_t_values]
    with Pool() as multi:
        res = multi.starmap(numerical_solver_multi, multi_args)

    Avv = [x[0] for x in res]
    Av1 = [x[1] for x in res]

    multi_args = [(gamma_D_tuning, Omega_ts, portfolio_value, tickers, out_of_sample, in_sample, Avv, Av1) for gamma_D_tuning in gamma_Ds_tuning]

    with Pool() as multi:
        sharpe_ratios = multi.starmap(multiprocessing_helper, multi_args)

    sharpe = [x[0] for x in sharpe_ratios]
    std = [x[1] for x in sharpe_ratios]

    return sharpe, std


if __name__ == '__main__':
    portfolio_value = 1e9
    #sharpes, std = test_gamma_D_params(['HYG','TLT']
    #                             , number_of_out_of_sample_days=1000,
    #                             gamma_start=3e-5, gamma_end=1e-2, gamma_num=100)
    #sharpes, std = test_gamma_D_params(['EEM', 'IVV', 'IEV', 'IXN', 'IYR', 'IXG', 'EXI', 'GC=F', 'BZ=F', 'HYG', 'TLT']
    # sharpes, std = test_gamma_D_params(['EEM', 'IVV', 'IEV', 'IXN', 'TLT'],
    #                                    number_of_out_of_sample_days=1000, model_type="sGARCH10",
    #                                                                    portfolio_value=1e9,
    #                              gamma_start=1e-6, gamma_end=1e-2, gamma_num=150)
    #                               #gamma_start=1e-3, gamma_end=1e10, gamma_num=300)
    #v_t_s, out_of_sample, in_sample, Omega_ts = garch_no_trading_cost(['EEM', 'IVV', 'IEV', 'IXN', 'TLT'])
    v_t_s, out_of_sample, in_sample, Omega_ts = garch_with_trading_cost(['HYG', 'TLT'], number_of_out_of_sample_days=1000, tuning_gamma_D=0.00015511818625173554)
    #v_t_s, out_of_sample, in_sample, Omega_ts = garch_with_trading_cost(['EEM', 'IVV', 'IEV', 'IXN', 'TLT'], model_type="sGARCH11")

    sharpe = False
    if sharpe == False:
        cum_returns, perf_table = performance_table(v_t_s, out_of_sample, Omega_ts, portfolio_value=portfolio_value)
        print(perf_table)
        plt.plot(cum_returns)
        #plt.plot(v_t_s)
        plt.show()
    else:
        for sharpe in sharpes:
            print(sharpe)
        sharpes = pd.DataFrame(sharpes, columns=['gamma_D', 'GARCH TC', 'Equal_weight TC', 'BnH TC'])
        std = pd.DataFrame(std, columns=['gamma_D', 'GARCH TC', 'Equal_weight TC', 'BnH TC'])
        std.set_index('gamma_D', drop=True, inplace=True)
        sharpes.set_index('gamma_D', drop=True, inplace=True)
        #plt.plot(sharpes, linestyle="--")
        plt.plot(std)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
