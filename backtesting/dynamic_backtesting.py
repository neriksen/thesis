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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import garch_utilites as gu
from rpy2.robjects import pandas2ri
import sys
pandas2ri.activate()

np.set_printoptions(precision = 3, suppress = True, linewidth = 400)

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


def prepare_return_data(tickers, start, end):
    return_data = yfinance.download(tickers, start = start, end= end)['Adj Close']
    return_data = return_data/return_data.iloc[0]
    return_data = return_data.pct_change().iloc[1:]*100
    return_data.to_csv("data/return_data.csv", sep=";")
    return return_data


def split_sample(return_data, length_sample_period):
    out_of_sample = return_data.iloc[length_sample_period:, ]
    in_sample = return_data.iloc[:length_sample_period, ]
    return out_of_sample, in_sample


def fit_garch_model(length_sample_period):
    # Defining the R script and loading the instance in Python
    r = ro.r
    r['source']('backtesting/fitting_mgarch.R')
    # Loading the function we have defined in R.
    fit_mgarch_r = ro.globalenv['fit_mgarch']
    # Fitting the mgarch model and receiving the result
    ugarch_model = "sGARCH"
    ugarch_dist_model = "norm"
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


def garch_no_trading_cost(tickers: list, start, end, number_of_out_of_sample_days):
    """
    tickers: ["ticker", "ticker", ..., "ticker"]
    start: "yyyy-m-d"
    end: "yyyy-m-d"
    modeltype: One of "sGARCH11", "sGARCH10", "gjrGARCH11", "eGARCH11"
    """
    print(f"Calculating weights for:", *tickers)
    return_data = prepare_return_data(tickers, start, end)
    #Receive variables from model
    asset_names = return_data.columns.values
    p = len(asset_names)

    length_sample_period = len(return_data) - number_of_out_of_sample_days
    out_of_sample, in_sample = split_sample(return_data, length_sample_period)

    # Fit model
    coef, residuals, sigmas = fit_garch_model(length_sample_period)

    # Parse variables
    mu, o, al, be, dcca, dccb = parse_garch_coef(coef, p)

    # 5. Calculate Qbar
    Qbar = gu.calculate_Qbar(residuals, sigmas)
    Q_t = Qbar

    Omega_ts = gu.main_loop(out_of_sample, sigmas, residuals, Qbar, Q_t, dcca, dccb, o, al, be, mu)

    # Generating weights
    ones = np.ones((p, 1))
    v_t = [np.ravel(divide(dot(inv(Omega), ones), mdot([ones.T, inv(Omega), ones]))) for Omega in Omega_ts]
    v_t = pd.DataFrame(v_t, columns=asset_names, index=return_data.index[-len(v_t):])

    return v_t, out_of_sample, in_sample


if __name__ == '__main__':
    v_t, out_of_sample, in_sample = garch_no_trading_cost(['IVV', 'HYG'], "2011-1-1", "2019-1-1", 1000)
    _, performance_table = compare_strategies(v_t, out_of_sample)
    print(performance_table)