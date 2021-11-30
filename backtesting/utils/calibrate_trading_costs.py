import yfinance as yf
import pandas as pd
import os


def calc_gammaD():
    assets = {"EEM": "Emerging Markets",
              "IVV": "S&P 500",
              "IEV": "MSCI Europe",
              "IXN": "Global Tech",
              "IYR": "US Real Estate",
              "IXG": "Global Financials",
              "EXI": "Global Industrials",
              "GCF": "Gold Futures",
              "BZF": "Brent Crude Oil Futures",
              "HYG": "High-Yield Corporate Bonds",
              "TLT": "20+ Year Treasury Bonds"}
    tickers = list(assets.keys())

    data = yf.download(tickers, start="2008-01-01", end="2017-10-11")
    data.loc[:, ('Volume', ['BZF', 'GCF'])] *= 100     # To account for contract size of 100 on futures
    avgvol_std = pd.concat([data['Volume'].mean(),
                            data['Adj Close'].mean(),
                            data['Adj Close'].pct_change().var(),
                            (data['Adj Close'].pct_change().std()*data['Adj Close'].mean())**2,
                            data['Volume'].mean()*data['Adj Close'].mean()],
                           axis=1)
    avgvol_std.columns = ["Avg volume", "Avg price", "Avg var", "Price change var", "Avg dollar volume"]

    # See Pedersen 2013 page 2328
    #  Dollar gamma_D
    avgvol_std['gammaD'] = ((avgvol_std['Avg volume']*(0.0159*avgvol_std['Price change var']))**(-1))*0.001*avgvol_std['Avg price']
    print(avgvol_std['gammaD'].mean())
    print(avgvol_std['gammaD'].median())
    avgvol_std.to_csv(os.path.join(os.path.dirname(__file__), 'avg_volume.csv'))


def asset_lookup(asset_names: list, col_lookup):
    """
    Possible col_lookup names: Avg volume, Avg price, Price change vol, gammaD
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'avg_volume.csv'), index_col=0)
    return df[col_lookup].loc[[*asset_names]]


def get_gamma_D(mean_or_median):
    gamma_D_dict = {"mean": 3.378117760647255e-05,
                    "median": 1.526988705495546e-06}
    assert (mean_or_median in gamma_D_dict.keys())
    return gamma_D_dict[mean_or_median]


if __name__ == '__main__':
    #print(avg_vol_lookup(['EEM', 'EXI', 'TLT']))
    calc_gammaD()
