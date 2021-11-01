import yfinance as yf
import pandas as pd
import numpy as np


def calc_gammaD():
    portfolio_value = 1e9
    assets = {"EEM": "Emerging Markets",
              "IVV": "S&P 500",
              "IEV": "MSCI Europe",
              "IXN": "Global Tech",
              "IYR": "US Real Estate",
              "IXG": "Global Financials",
              "EXI": "Global Industrials",
              "GC=F": "Gold Futures",
              "BZ=F": "Brent Crude Oil Futures",
              "HYG": "High-Yield Corporate Bonds",
              "TLT": "20+ Year Treasury Bonds"}
    tickers = list(assets.keys())

    data = yf.download(tickers, start="2008-01-01", end="2017-10-11")
    avgvol_std = pd.concat([data['Volume'].mean(),
                            data['Adj Close'].mean(),
                            data['Adj Close'].pct_change().std(),
                            data['Adj Close'].pct_change().std()*data['Adj Close'].mean(),
                            data['Volume'].mean()*data['Adj Close'].mean()],
                           axis=1)
    avgvol_std.columns = ["Avg volume", "Avg price", "Avg vol", "Price change vol", "Avg dollar volume"]

    # See Pedersen 2013 page 2328
    avgvol_std['gammaD'] = (avgvol_std['Avg vol']**2)*((avgvol_std['Avg volume']*0.0159*avgvol_std['Avg price'])**(-1))*0.001*portfolio_value
    print(avgvol_std['gammaD'].mean())
    print(avgvol_std['gammaD'].median())
    avgvol_std.to_csv('avg_volume.csv')


def asset_lookup(asset_names: list, col_lookup):
    """
    Possible col_lookup names: Avg volume, Avg price, Price change vol, gammaD
    """
    df = pd.read_csv('backtesting/utils/avg_volume.csv', index_col=0)
    return df[col_lookup].loc[[*asset_names]]


if __name__ == '__main__':
    #print(avg_vol_lookup(['EEM', 'EXI', 'TLT']))
    calc_gammaD()