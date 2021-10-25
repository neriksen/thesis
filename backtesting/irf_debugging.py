import dynamic_backtesting as db
import garch_utilites as gu
import pandas as pd
import numpy as np


def IRF_maker(GARCHTYPE,t,Asset_number,shock_size, coef, residuals, sigmas, asset_names):
    """
    Makes data for an IRF plot with a GARCH type model using the data in the csv file and simulation

    Variables:
    t (int): the length of sample
    Asset_number: The asset that receive the shock
    shock_size: the stock to the mean
    GARCHTYPE: sGARCH11, sGARCH10, gjrGARCH11
    Residuals: Past residuals
    Sigmas: Past sigmas
    """
    #unpacks paramters
    params_dict = db.parse_garch_coef(coef, 11, GARCHTYPE)
    print(params_dict)
    mu_T=params_dict["mu"].transpose()
    irf_data=pd.DataFrame(mu_T.repeat(t, axis=0), columns=asset_names)
    #Indsætter chok
    irf_data.iloc[int(t/2), Asset_number] = shock_size
    #udregner omega
    irf_omega_s=db.calc_Omega_ts(out_of_sample_returns=irf_data, in_sample_sigmas=sigmas,
                                 in_sample_residuals=residuals, **params_dict)
    #udregner vægte
    irf_weights=db.calc_weights_garch_no_trading_cost(irf_omega_s)
    return irf_weights, irf_omega_s


def GARCH_MODEL(ugarch_model="sGARCH", garch_order=(1, 1)):
    """
    Estimate af GARCH model and parse parameters, sigmas and residuals
    ugarch_model: sGARCH, gjrGARCH
    garch_order: (1, 1), (1,0)
    """
    coef, residuals, sigmas = db.fit_garch_model(0, ugarch_model=ugarch_model, garch_order=garch_order)
    return coef, residuals, sigmas


def main():
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
    asset_names = list(assets.values())
    return_data = db.download_return_data(tickers, save_to_csv=True)
    coef_ARCH, residuals_ARCH, sigmas_ARCH = GARCH_MODEL("sGARCH", (1, 0))
    irf_weights_ARCH, irf_omega_s = IRF_maker("sGARCH10", 10000, 1, 15, coef_ARCH, residuals_ARCH, sigmas_ARCH, asset_names)


if __name__ == "__main__":
    main()