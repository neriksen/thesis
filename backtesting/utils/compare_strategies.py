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


def calc_transaction_costs(weights: pd.DataFrame, returns, Omega_ts, portfolio_value=1e9):
    assert len(weights) == len(returns)+1
    assert weights.iloc[[1]].index == returns.iloc[[0]].index
    p = weights.shape[1]
    gamma_D = calibrate_trading_costs.get_gamma_D("mean")
    TC = np.zeros((len(returns)))
    avg_price = calibrate_trading_costs.asset_lookup(list(returns.columns), col_lookup="Avg price")

    for t, (v_t, Omega_t, r_t, r_t_1) in enumerate(zip(weights.iterrows(), Omega_ts, returns.shift(1, fill_value=0).iterrows(),
                                                       returns.shift(2, fill_value=0).iterrows())):

        Omega_value = Omega_t[1]/10000

        v_t_value = np.reshape(v_t[1].values, (p, 1))
        r_t_value = np.reshape(r_t[1].values, (p, 1))
        r_t_1_value = np.reshape(r_t_1[1].values, (p, 1))

        if t >= 1:   # Only measureable from period t = 2

            TO = calc_turnover_pct(v_t_value, v_t_1, r_t_1_value)
            amount_traded = np.reshape(portfolio_value * TO, avg_price.shape)/avg_price
            Lambda_t = Lambda(Omega_value, gamma_D)
            dollar_transaction_costs = mdot([amount_traded.T, Lambda_t, amount_traded])
            relative_transaction_costs = dollar_transaction_costs/portfolio_value  # Divide by port value to get relative loss
            TC[t] = relative_transaction_costs
            portfolio_return = (1+dot(v_t_value.T, r_t_value)-relative_transaction_costs)
            portfolio_value *= portfolio_return

        v_t_1 = v_t_value

    return TC


def performance_table(weights, returns_pct: pd.DataFrame, Omega_ts, portfolio_value=1e9) -> Tuple[pd.DataFrame, Any]:
    """
    Function assumes weights start in period T (last in-sample period)
    and returns_pct start in period T+1 (first out-of-sample period)
    """
    p = weights.shape[1]

    # Assert that we have one more weight than returns
    assert len(weights) == len(returns_pct)+1
    assert weights.iloc[[1]].index == returns_pct.iloc[[0]].index

    # Delay weights to enable 1:1 multiplication with returns
    delayed_weights = weights.shift(1).iloc[1:]

    # GARCH return
    returns = returns_pct.divide(100)
    portfolio_returns = delayed_weights.multiply(returns).sum(axis=1).to_frame()
    portfolio_returns.columns = ['GARCH']

    # Calculate returns net transaction costs
    TC_garch = calc_transaction_costs(weights, returns, Omega_ts, portfolio_value)
    portfolio_returns['GARCH TC'] = portfolio_returns['GARCH']-TC_garch

    cum_portfolio_returns = portfolio_returns.add(1).cumprod()

    # 1/N return
    Equal_weights = pd.DataFrame(np.full(weights.shape, (1/p)), index=weights.index)
    TC_Equal_weight = calc_transaction_costs(weights=Equal_weights, returns=returns,
                                             Omega_ts=Omega_ts, portfolio_value=portfolio_value)
    cum_portfolio_returns['Equal_weight'] = returns.mean(axis=1).add(1).cumprod()
    cum_portfolio_returns['Equal_weight TC'] = returns.mean(axis=1).sub(TC_Equal_weight).add(1).cumprod()


    # GARCH is first weight in Buy and hold  (BnH)
    # Since turnover = |v_t - v_{t-1}*(1+r_t)|, then v_{t-1} = v_t/(1+r_t) when aiming for turnover = 0.
    BnH_weights = [weights.iloc[0].values]
    for t, (_, _return) in enumerate(returns.shift(1, fill_value=0).iterrows()):
        if t == 0:
            BnH_weights.append(weights.iloc[0].values)
        else:
            _return = np.reshape(_return.values, (p, 1))
            v_t_1 = np.reshape(BnH_weights[-1], (p, 1))
            next_weight = np.divide(np.multiply(v_t_1, (1+_return)), 1+dot(v_t_1.T, _return))
            BnH_weights.append(np.ravel(next_weight))

    # Delay weights to enable 1:1 multiplication with returns
    BnH_weights = pd.DataFrame(np.array(BnH_weights), index=weights.index)
    delayed_BnH_weights = BnH_weights.shift(1).iloc[1:]
    BnH_returns = np.multiply(delayed_BnH_weights, returns).sum(axis=1)
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
