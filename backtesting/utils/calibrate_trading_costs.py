import yfinance as yf
import pandas as pd


def calc_gammaD():
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
                            data['Adj Close'].pct_change().std()*data['Adj Close'].mean()], axis=1)
    avgvol_std.columns = ["Avg volume", "Avg price", "Price change vol"]

    # See Pedersen 2013 page 2328
    avgvol_std['gammaD'] = (1/(0.0159*avgvol_std['Avg volume']*(avgvol_std['Price change vol']**2)))*2*(0.001*avgvol_std['Avg price'])

    avgvol_std.to_csv('avg_volume.csv')


if __name__ == '__main__':
    calc_gammaD()