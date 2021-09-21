suppressMessages(library(rmgarch))
suppressMessages(library(parallel))
suppressMessages(library(quantmod))
suppressMessages(library(tidyverse))


fit_mgarch <- function(length_sample_period){
    setwd("/Users/nielseriksen/thesis/")

    etfs <- read.csv("data/etfs.csv", sep=";")[0:length_sample_period, ] %>% select(-Date)

    xspec = ugarchspec(mean.model = list(armaOrder = c(0, 0)), variance.model = list(garchOrder = c(1,1), model = 'sGARCH'), distribution.model = 'norm')
    uspec = multispec(replicate(10, xspec))
    spec1 = dccspec(uspec = uspec, dccOrder = c(1, 1), distribution = 'mvnorm')
    cl = makePSOCKcluster(4)
    multf = multifit(uspec, etfs, cluster = cl)
    fit1 = dccfit(spec1, data = etfs, fit.control = list(eval.se = TRUE), fit = multf, cluster = cl)
    #print(fit1)           
    stopCluster(cl)

    #coef(fit1) %>% write.csv('data/coef.csv')    
    saveRDS(fit1, "backtesting/fitted_mgarch_model.rds")
    return(list(coef(fit1), residuals(fit1), sigma(fit1)))
    
}


rcov_forecast <- function(){
    fitted_model <- readRDS("backtesting/fitted_mgarch_model.rds")
    forecast <- dccforecast(fitted_model, n.ahead=1)
    return(rcov(forecast, output="matrix"))
}