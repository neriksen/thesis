import numpy as np
from numpy import divide
from numpy.linalg import multi_dot as mdot
from numpy import dot
import yfinance
import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import garch_utilites as gu
import matplotlib.pyplot as plt
np.set_printoptions(precision = 3, suppress = True, linewidth = 400)
os.chdir('/Users/nielseriksen/thesis/')


def main():
    tickers = ['IVV', 'HYG', 'GC=F', 'IYT']
    etfs = yfinance.download(tickers, auto_adjust = True, start = "2011-9-1", end='2019-9-1')['Close']
    etfs = etfs/etfs.iloc[0]
    etfs = etfs.pct_change().iloc[1:]*100
    etfs.to_csv("data/etfs.csv", sep=";")

    # Defining the R script and loading the instance in Python
    r = ro.r
    r['source']('backtesting/fitting_mgarch.R')

    # 0. Fit model
    length_sample_period = len(etfs)-1000
    out_of_sample = etfs.iloc[length_sample_period:,]
    in_sample = etfs.iloc[:length_sample_period,]
    # Loading the function we have defined in R.
    fit_mgarch_r = ro.globalenv['fit_mgarch']
    #Fitting the mgarch model and receiving the result
    ugarch_model = "sGARCH"
    ugarch_dist_model = "norm"
    coef, residuals, sigmas = fit_mgarch_r(length_sample_period, ugarch_model, ugarch_dist_model)

    #Receive variables from model
    asset_names = etfs.columns.values
    p = len(asset_names)

    # How elegant
    mu, o, al, be = np.hsplit(coef[:-2].reshape((len(asset_names), 4)), 4)
    mu, o, al, be = map(np.ravel, (mu, o, al, be))  # Flattening to 1d array
    mu, o, al, be = map(np.reshape, (mu, o, al, be), [(p, 1)]*4)  # Reshaping to px1
    dcca = coef[-2]
    dccb = coef[-1]
    epsilons = residuals

    # 5. Calculate Qbar
    Qbar = gu.calculate_Qbar(epsilons, sigmas)
    Q_t = Qbar

    Omega_ts = gu.main_loop(out_of_sample, sigmas, epsilons, Qbar, Q_t, dcca, dccb, o, al, be, mu)

    # Generating weights
    ones = np.ones((p, 1))
    v_t = [np.ravel(divide(dot(Omega, ones), mdot([ones.T, Omega, ones]))) for Omega in Omega_ts]
    plt.plot(v_t)
    plt.show()


if __name__ == '__main__':
    main()
