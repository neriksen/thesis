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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
pandas2ri.activate()


if sys.platform == "darwin":
    os.chdir('../')
else:
    os.chdir("..\\")


def compare_strategies(weights, returns: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
    #weights = v_t
    #returns = out_of_sample
    portfolio_returns: pd.DataFrame = pd.DataFrame((weights*returns.shift(1)).sum(axis=1)/100,
                                                   index = returns.index)
    cum_portfolio_returns = pd.DataFrame(portfolio_returns + 1).cumprod()

    # Alternative strategies
    # 1/N
    cum_portfolio_returns['Equal_weight'] = (returns.mean(axis=1)/100 + 1).cumprod()
    cum_portfolio_returns = pd.DataFrame(cum_portfolio_returns/cum_portfolio_returns.iloc[0])

    # Buy and hold 1/N
    #cum_portfolio_returns['Equal_weight_buy_n_hold'] = (returns.mean(axis=1) / 100 + 1).cumprod()
    # Formatting
    cum_portfolio_returns.columns = ["GARCH no trading costs", "Equal weight"]

    # Calculate std
    std = cum_portfolio_returns.pct_change().std()*np.sqrt(250)
    mean_return = cum_portfolio_returns.pct_change().mean() * 250
    sharpe = mean_return/std

    performance_table = pd.DataFrame([std, mean_return, sharpe]).transpose()
    performance_table.columns = ["Ann. standard deviation", "Ann. return", "Ann. Sharpe ratio"]
    return cum_portfolio_returns, performance_table


def download_return_data(tickers, start="2008-01-01", end="2021-10-02", save_to_csv=True):
    return_data = yfinance.download(tickers, start=start, end=end)['Adj Close']
    return_data = return_data/return_data.iloc[0]
    return_data = return_data.pct_change().iloc[1:]*100
    if save_to_csv:
        return_data.to_csv("data/return_data.csv", sep=";")
    return return_data


def split_sample(return_data, length_sample_period):
    out_of_sample = return_data.iloc[length_sample_period:, ]
    in_sample = return_data.iloc[:length_sample_period, ]
    return out_of_sample, in_sample


def fit_garch_model(length_sample_period, ugarch_model="sGARCH"):
    """
    ugarch_model: One of "sGARCH", "gjrGARCH", "eGARCH"
    """
    # Define the R script and load the instance in Python
    r = ro.r
    r['source']('backtesting/fitting_mgarch.R')
    # Load the function we have defined in R.
    fit_mgarch_r = ro.globalenv['fit_mgarch']
    # Fit the MGARCH model and receive the result
    ugarch_dist_model = "norm"      # FIXME: Change to "std" when parsing is correct
    coef, residuals, sigmas = fit_mgarch_r(length_sample_period, ugarch_model, ugarch_dist_model)
    return coef, residuals, sigmas


def parse_garch_coef(coef, p):
    """
    Possible parsings: sGARCH11, sGARCH10, gjrGARCH11, eGARCH11
    """
    # How elegant
    mu, o, al, be = np.hsplit(coef[:-2].reshape((p, 4)), 4)
    mu, o, al, be = map(np.ravel, (mu, o, al, be))  # Flattening to 1d array
    mu, o, al, be = map(np.reshape, (mu, o, al, be), [(p, 1)] * 4)  # Reshaping to px1
    dcca = coef[-2]
    dccb = coef[-1]
    return mu, o, al, be, dcca, dccb


def calc_weights_garch_no_trading_cost(Omega_ts):
    assert isinstance(Omega_ts, (list, np.ndarray))
    p = len(Omega_ts[0])
    ones = np.ones((p, 1))
    v_t = [np.ravel(divide(dot(inv(Omega), ones), mdot([ones.T, inv(Omega), ones]))) for Omega in Omega_ts]
    return v_t
    

def calc_Omega_ts(out_of_sample, in_sample_sigmas, in_sample_residuals, dcca, dccb, o, al, be, mu):
    Qbar = gu.calculate_Qbar(in_sample_residuals, in_sample_sigmas)
    Q_t = Qbar      # Qbar is the same as Q_t at the start of the out-of-sample period

    Omega_ts = gu.main_loop(out_of_sample, in_sample_sigmas, in_sample_residuals, Qbar, Q_t, dcca, dccb, o, al, be, mu)
    return Omega_ts


def garch_no_trading_cost(tickers, start="2008-01-01", end="2021-10-02", number_of_out_of_sample_days=250*2,
                          model_type="sGARCH11"):
    """
    tickers: ["ticker", "ticker", ..., "ticker"]
    start: "yyyy-m-d"
    end: "yyyy-m-d"
    model_type: One of "sGARCH11", not implemented = ("sGARCH10", "gjrGARCH11", "eGARCH11")
    """
    assert (model_type in ["sGARCH11", "sGARCH10", "gjrGARCH11", "eGARCH11"])
    garch_type = model_type[:-2]
    print(f"Calculating weights for:", *tickers)
    return_data = download_return_data(tickers, start, end, True)

    # Determining dimensions
    T, p = return_data.shape

    length_sample_period = T - number_of_out_of_sample_days
    out_of_sample, in_sample = split_sample(return_data, length_sample_period)

    # Fit model
    coef, residuals, sigmas = fit_garch_model(length_sample_period, garch_type)

    # Parse variables
    mu, o, al, be, dcca, dccb = parse_garch_coef(coef, p)

    Omega_ts = calc_Omega_ts(out_of_sample, sigmas, residuals, dcca, dccb, o, al, be, mu)
    # Generating weights
    v_t = calc_weights_garch_no_trading_cost(Omega_ts)
    v_t = pd.DataFrame(v_t, columns=tickers, index=return_data.index[-len(v_t):])

    return v_t, out_of_sample, in_sample


if __name__ == '__main__':
    v_t, out_of_sample, in_sample = garch_no_trading_cost(['IVV', 'HYG'], "2011-1-1", "2019-1-1", 1000)
    _, performance_table = compare_strategies(v_t, out_of_sample)
    print(performance_table)