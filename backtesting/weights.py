import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta
from numpy import dot
from numpy import divide
from numpy.linalg import multi_dot as mdot
from numpy.linalg import inv
import matplotlib.pyplot as plt
import yfinance


class Portfolio():    
    """
    start: "YYYY-M-D"
    """
    
    def __init__(self, tickers: list, r_f = 0.01, start="2010-1-1"):
        self.tickers = tickers
        self._raw_data = yfinance.download(tickers, start=start, back_adjust=True, auto_adjust=True)
        self._start_date = self._raw_data.index[0]
        self._end_date = self._raw_data.index[-1]
        self._closing_prices = self._raw_data['Close']
        self.r_f = r_f
        self.__calculate_returns()
        self._mean_return = self._returns.mean()
        self._annualized_mean_return = self._mean_return*250
        self._std = self._returns.std()
        self._annualized_std = self._std*(250**0.5)
        self._static_Sigma = self._returns.cov().values
        self._static_Sigma_inv = inv(self._static_Sigma)
        self._static_min_var_weights = self.min_var_weights(Sigma_inv = self._static_Sigma_inv)

        
    def __calculate_returns(self):
        self._returns = self._closing_prices.pct_change().iloc[1:].ffill().dropna(axis = 1, how = 'any')*100
        

    def __calculate_Sigma_inv(self, returns):
        return inv(self.__calculate_Sigma(returns))


    def __calculate_Sigma(self, returns):
        return returns.cov().values


    def __port_std(self, returns: pd.DataFrame):
        return returns.pct_change().std() * (250 ** 0.5)


    def __port_performance(self, returns: pd.DataFrame, portfolio_names: list):
        for strategy in portfolio_names:
            std = self.__port_std(returns[strategy])
            yearly_return = (returns[strategy].iloc[-1]/returns[strategy].iloc[0])**(len(returns[strategy])/250)-1
            print(f'Sharpe Ratio {strategy}: {yearly_return/std}')

    
    def min_var_weights(self, Sigma_inv):
        one = np.ones((len(Sigma_inv)))
        v_t_mvp = dot(Sigma_inv, one)/mdot([one.T, Sigma_inv, one])
        return v_t_mvp
    
    
    def efficient_weights(self, Sigma_inv, mu_tilde_star = 10, mu_tilde = None):
        v_t_mvp = self.min_var_weights(Sigma_inv)
        
        if type(mu_tilde) == int or type(mu_tilde) == float:
            if mu_tilde == 0:
                return v_t_mvp
        if mu_tilde is None:
            mu_tilde = self._annualized_mean_return

        one = np.ones((len(Sigma_inv)))
        A = mdot([one.T, Sigma_inv, one])
        B = mdot([one.T, Sigma_inv, mu_tilde])
        C = mdot([mu_tilde.T, Sigma_inv, one])
        D = mdot([mu_tilde.T, Sigma_inv, mu_tilde])

        k2 = (mu_tilde_star-divide(C, A))/(D-divide(dot(B, C), A))
        v_t_mvp = self.min_var_weights(Sigma_inv)
        v_t_eff = v_t_mvp + k2*(dot(Sigma_inv, mu_tilde)-dot(B, v_t_mvp))
        return v_t_eff
    
    
    def backtest(self, backtest_start = None):
        if backtest_start is None:
            backtest_start = self.start_date + timedelta(days=10)
        
        if isinstance(backtest_start, str):
            backtest_start = dt.datetime.strptime(backtest_start, "%Y-%m-%d")
        
        results = []
        min_var_weights = []
        eff_weights = []

        date_list = self._returns.loc[backtest_start:].index
        all_days = self._returns.index
        
        for i, date in enumerate(date_list):
            try:
                # Calculate min var weights for the coming day
                returns = self._returns.loc[all_days[i]:date]
                Sigma_inv = self.__calculate_Sigma_inv(returns)
                v_t_mvp = self.min_var_weights(Sigma_inv)

                # Calculate efficient weights for the coming day
                mu_tilde = returns.mean()
                v_t_eff = self.efficient_weights(Sigma_inv, mu_tilde=mu_tilde)
                #v_t_eff = self.efficient_weights(Sigma_inv)
                eff_weights.append([date_list[i+1], *v_t_eff.tolist()])

                # Calculate daily return
                assert date_list[i+1] != returns.index[-1]

                next_day_prices = self._returns.loc[date_list[i+1]].values
                min_var_return = dot(v_t_mvp.T, next_day_prices)
                eff_return = dot(v_t_eff.T, next_day_prices)

                results.append([date_list[i+1], min_var_return, eff_return])

            except IndexError:
                pass

        portfolios = ["Minimum variance", "Efficient portfolio"]
        results = self.__normalize_returns(results, portfolios)
        self.__port_performance(results, portfolios)

        return results, eff_weights
    
    
    def __normalize_returns(self, result_list, portfolio_names: list):
        results = pd.DataFrame(result_list, 
                               columns=["Date", *portfolio_names])
        results.set_index("Date", inplace = True)
        results = results/100+1
        results.iloc[0] = 1
        results = results.cumprod()
        return results
    

    @property
    def end_date(self):
        return self._end_date
    @property
    def start_date(self):
        return self._start_date
    @property
    def static_Sigma(self):
        return self._static_Sigma
    @property
    def static_Sigma_inv(self):
        return self._static_Sigma_inv
    @property
    def static_min_var_weights(self):
        return self._static_min_var_weights
    @property
    def mean_return(self):
        return self._mean_return
    @property
    def annualized_mean_return(self):
        return self._annualized_mean_return
    @property
    def std(self):
        return self._std
    @property
    def annualized_std(self):
        return self._annualized_std
    @property
    def returns(self):
        return self._returns
    @property
    def raw_data(self):
        return self._raw_data
    @property
    def closing_prices(self):
        return self._closing_prices

    
if __name__ == "__main__":
    spx = pd.read_csv('../data/spx.csv').stack().tolist()
    spx_small = spx[:20]
    port = Portfolio(spx_small)
    res, eff_weights = port.backtest(backtest_start = "2015-1-1")
    
    plt.style.use('seaborn')
    spx_data = yfinance.download(['SPY'], start = "2015-1-1")['Close']
    spx_data = spx_data/spx_data.iloc[0]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(spx_data, label="SPY ETF")
    ax.plot(res, label=['Minimum variance portfolio', 'Efficient portfolio'])
    ax.legend()
    
    plt.show()