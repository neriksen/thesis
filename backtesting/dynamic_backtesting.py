from typing import Tuple, Any
import numpy as np
from numpy import divide
from numpy.linalg import multi_dot as mdot
from numpy.linalg import inv
from numpy import dot
import pandas as pd
import yfinance
import os
import rpy2.robjects as ro
import sys
import garch_utilites as gu
from rpy2.robjects import pandas2ri
from rpy2.robjects import IntVector
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
pandas2ri.activate()


if sys.platform == "darwin":
    os.chdir('../')
else:
    os.chdir("..\\")


def calc_turnover(weights: pd.DataFrame, returns_pct):
    returns = returns_pct/100
    #weight_delta = weights.diff(axis=1)
    #for v_t in weight_delta.iterrows():

    #TO = weight_delta * (1 + returns)


def compare_strategies(weights, returns_pct: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
    """
    Function assumes weights start in period t-1 and returns_pct start in period t
    """

    # GARCH return
    returns = returns_pct.divide(100)
    portfolio_returns = weights.multiply(returns).sum(axis=1)
    cum_portfolio_returns = portfolio_returns.add(1).cumprod().to_frame()

    # 1/N return
    cum_portfolio_returns['Equal_weight'] = returns.mean(axis=1).add(1).cumprod()

    # Buy and hold 1/N (BnH)
    # Since turnover = |v_t - v_{t-1}*(1+r_t)|, then v_{t-1} = v_t/(1+r_t) when aiming for turnover = 0.
    p = weights.shape[1]
    BnH_weights = []
    for t, (_, _return) in enumerate(returns.add(1).iterrows()):
        if t == 0:
            BnH_weights.append(weights.iloc[0].values)
        else:
            _return = np.reshape(_return.values, (1, p))
            next_weight = BnH_weights[-1]*_return
            next_weight = next_weight/np.sum(next_weight)
            BnH_weights.append(np.ravel(next_weight))

    BnH_weights = np.array(BnH_weights)
    cum_portfolio_returns['BnH'] = np.multiply(BnH_weights, returns).sum(axis=1).add(1).cumprod()
    # Formatting
    cum_portfolio_returns.columns = ["GARCH no trading costs", "Equal weight", "BnH"]

    # Normalize returns to begin at 1
    cum_portfolio_returns = cum_portfolio_returns.divide(cum_portfolio_returns.iloc[0])
    cum_portfolio_returns.index = pd.to_datetime(cum_portfolio_returns.index)

    # Calculate aggregate performance measures
    std = cum_portfolio_returns.pct_change().std()*np.sqrt(250)
    mean_return = cum_portfolio_returns.pct_change().mean() * 250
    sharpe = mean_return.divide(std)

    performance_table = pd.DataFrame([std, mean_return, sharpe]).transpose()
    performance_table.columns = ["Ann. standard deviation", "Ann. return", "Ann. Sharpe ratio"]
    return cum_portfolio_returns, performance_table


def download_return_data(tickers, start="2008-01-01", end="2021-10-02", save_to_csv=True):
    return_data = yfinance.download(tickers, start=start, end=end)['Adj Close']
    return_data = return_data/return_data.iloc[0]
    return_data = return_data.pct_change().iloc[1:]*100
    return_data = return_data[tickers]
    if save_to_csv:
        return_data.to_csv("data/return_data.csv", sep=";")
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
    r['source']('backtesting/fitting_mgarch.R')
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


def calc_weights_garch_no_trading_cost(Omega_ts):
    assert isinstance(Omega_ts, (list, np.ndarray))
    p = len(Omega_ts[0])
    ones = np.ones((p, 1))
    v_t = [np.ravel(divide(dot(inv(Omega), ones), mdot([ones.T, inv(Omega), ones]))) for Omega in Omega_ts]
    return v_t
    

def calc_Omega_ts(out_of_sample_returns, in_sample_returns, in_sample_sigmas, in_sample_residuals, **kw):
    Qbar = gu.calc_Qbar(in_sample_residuals, in_sample_sigmas)
    Q_t = Qbar      # Qbar is the same as Q_t at the start of the out-of-sample period

    Omega_ts = gu.main_loop(out_of_sample_returns=out_of_sample_returns, in_sample_returns = in_sample_returns,
                        sigmas=in_sample_sigmas, epsilons=in_sample_residuals, Qbar = Qbar, Q_t = Q_t, **kw)
    return Omega_ts


def garch_no_trading_cost(tickers, start="2008-01-01", end="2021-10-02", number_of_out_of_sample_days=250*2,
                          model_type="sGARCH11"):
    """
    tickers: ["ticker", "ticker", ..., "ticker"]
    start: "yyyy-m-d"
    end: "yyyy-m-d"
    model_type: One of "sGARCH11", "sGARCH10", "gjrGARCH11", not implemented = "eGARCH11"
    """
    assert (model_type in ("sGARCH11", "sGARCH10", "gjrGARCH11"))
    garch_type = model_type[:-2]
    garch_order = IntVector((model_type[-2], model_type[-1]))
    print(f"Calculating weights for:", *tickers)
    return_data = download_return_data(tickers, start, end, True)

    # Determining dimensions
    T, p = return_data.shape
    out_of_sample, in_sample  = split_sample(return_data=return_data,
                                             number_of_out_of_sample_days=number_of_out_of_sample_days)

    # Fit model
    coef, residuals, sigmas = fit_garch_model(len_out_of_sample=number_of_out_of_sample_days,
                                              ugarch_model=garch_type, garch_order=garch_order)

    # Parse variables
    params_dict = parse_garch_coef(coef=coef, p=p, model_type=model_type)

    Omega_ts = calc_Omega_ts(out_of_sample_returns=out_of_sample, in_sample_returns=in_sample,
                             in_sample_sigmas=sigmas, in_sample_residuals=residuals, **params_dict)
    # Generating weights
    v_t = calc_weights_garch_no_trading_cost(Omega_ts)
    v_t = pd.DataFrame(v_t, columns=tickers, index=return_data.index[-len(v_t):])

    return v_t, out_of_sample, in_sample


if __name__ == '__main__':
    v_t, out_of_sample, in_sample = garch_no_trading_cost(['EEM', 'IVV', 'IEV', 'IXN', 'IYR', 'IXG', 'EXI', 'GC=F', 'BZ=F', 'HYG', 'TLT'],
                                                          number_of_out_of_sample_days=0, model_type="sGARCH10")
    _, performance_table = compare_strategies(v_t, out_of_sample)
    print(performance_table)
