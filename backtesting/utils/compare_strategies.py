import pandas as pd
import numpy as np
import calibrate_trading_costs
from typing import Tuple, Any
from numpy.linalg import multi_dot as mdot
from numpy import dot


def Lambda(Omega, gamma_D):
    return np.multiply(Omega, gamma_D)


def calc_turnover_pct(v_t, v_t_1, r_t):
    TO = v_t - np.divide(np.multiply(v_t_1, 1+r_t), 1+dot(v_t_1.T, r_t))
    return TO


def clean_up_returns(series: pd.Series):
    tmp = series.copy()
    for t, gross_return in enumerate(series):
        if gross_return <= 0.01:
            tmp[t:] = 0
            if t < len(series):
                tmp[t+1:] = np.nan
            break
    return tmp


def calc_transaction_costs(weights: pd.DataFrame, returns, Omega_ts, portfolio_value=1e9, gamma_D=None):
    if gamma_D is None:
        gamma_D = calibrate_trading_costs.get_gamma_D("median")
    TC = np.zeros((len(weights)))
    avg_volume = calibrate_trading_costs.asset_lookup(list(returns.columns), col_lookup="Avg volume")
    avg_price = calibrate_trading_costs.asset_lookup(list(returns.columns), col_lookup="Avg price")
    for t, (v_t, v_t_1, Omega_t, r_t) in enumerate(zip(weights.iterrows(), weights.shift(1).iterrows(),
                                                       Omega_ts, returns.shift(1).iterrows())):
        v_t, v_t_1, r_t = v_t[1].values, v_t_1[1].values, r_t[1].values
        if t >= 1:   # Only measureable from period t = 2, however Python is base 0 so t >= 2 becomes t >= 1

            TO = calc_turnover_pct(v_t, v_t_1, r_t)
            amount_traded = np.reshape(portfolio_value * TO, avg_volume.shape)/avg_price
            Lambda_t = Lambda(Omega_t, gamma_D)
            dollar_transaction_costs = mdot([amount_traded.T, Lambda_t, amount_traded])
            relative_transaction_costs = dollar_transaction_costs/portfolio_value  # Divide by port value to get relative loss
            TC[t] = relative_transaction_costs
            portfolio_return = (1+dot(v_t.T, r_t)-relative_transaction_costs)
            portfolio_value *= portfolio_return

    return TC


def performance_table(weights, returns_pct: pd.DataFrame, Omega_ts, portfolio_value=1e9) -> Tuple[pd.DataFrame, Any]:
    """
    Function assumes weights start in period t-1 and returns_pct start in period t
    """
    p = weights.shape[1]

    # GARCH return
    returns = returns_pct.divide(100)
    portfolio_returns = weights.multiply(returns).sum(axis=1).to_frame()
    portfolio_returns.columns = ['GARCH']

    # Calculate returns net transaction costs
    TC_garch = calc_transaction_costs(weights, returns, Omega_ts, portfolio_value)
    portfolio_returns['GARCH TC'] = portfolio_returns['GARCH']-TC_garch

    cum_portfolio_returns = portfolio_returns.add(1).cumprod()
    TC_Equal_weight = calc_transaction_costs(pd.DataFrame(np.full(weights.shape, (1/p))), returns, Omega_ts, portfolio_value)
    # 1/N return
    cum_portfolio_returns['Equal_weight'] = returns.mean(axis=1).add(1).cumprod()
    cum_portfolio_returns['Equal_weight TC'] = returns.mean(axis=1).sub(TC_Equal_weight).add(1).cumprod()


    # Buy and hold GARCH firs (BnH)
    # Since turnover = |v_t - v_{t-1}*(1+r_t)|, then v_{t-1} = v_t/(1+r_t) when aiming for turnover = 0.

    BnH_weights = []    # FIXME: Use empirical covariance of in-sample instead of Omega_t
    for t, (_, _return) in enumerate(returns.shift(1).add(1).iterrows()):
        if t == 0:
            BnH_weights.append(weights.iloc[0].values)
        else:
            _return = np.reshape(_return.values, (p, 1))
            v_t_1 = np.reshape(BnH_weights[-1], (p, 1))
            next_weight = np.divide(np.multiply(v_t_1, (1+_return)), 1+dot(v_t_1.T, _return))
            BnH_weights.append(np.ravel(next_weight))

    BnH_weights = np.array(BnH_weights)
    BnH_returns = np.multiply(BnH_weights, returns).sum(axis=1)
    cum_portfolio_returns['BnH'] = BnH_returns.add(1).cumprod()
    cum_portfolio_returns['BnH TC'] = cum_portfolio_returns['BnH']

    # Formatting
    #cum_portfolio_returns.columns = ["GARCH", "GARCH TC", "Equal weight", "BnH"]

    # Normalize returns to begin at 1
    cum_portfolio_returns = cum_portfolio_returns.divide(cum_portfolio_returns.iloc[0])
    cum_portfolio_returns.index = pd.to_datetime(cum_portfolio_returns.index)

    # Clean up returns to ensure they end i gross returns hit 0
    cum_portfolio_returns = cum_portfolio_returns.apply(clean_up_returns, axis=0)

    # Calculate aggregate performance measures
    std = cum_portfolio_returns.pct_change().std()*np.sqrt(250)
    last_gross_return = cum_portfolio_returns.ffill(axis=0).iloc[-1, :]
    num_non_nan_periods = len(cum_portfolio_returns)-cum_portfolio_returns.isna().sum()
    ann_return = ((last_gross_return) ** (250/num_non_nan_periods))-1
    sharpe = ann_return.divide(std)

    new_order = ['GARCH', 'Equal_weight', 'BnH', 'GARCH TC', 'Equal_weight TC', 'BnH TC']
    performance_table = pd.DataFrame([std, ann_return, sharpe])[new_order].transpose()
    performance_table.columns = ["Ann. standard deviation", "Ann. return", "Ann. Sharpe ratio"]
    return cum_portfolio_returns, performance_table
