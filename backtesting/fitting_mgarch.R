suppressMessages(library(rmgarch))
suppressMessages(library(parallel))
suppressMessages(library(quantmod))
suppressMessages(library(tidyverse))


fit_mgarch <- function(length_sample_period, ugarch_model, ugarch_dist_model){
    setwd("/Users/nielseriksen/thesis/")
    xspec = ugarchspec(mean.model = list(armaOrder = c(0, 0)), variance.model = list(garchOrder = c(1,1), model = ugarch_model), distribution.model = ugarch_dist_model)
    uspec = multispec(replicate(10, xspec))
    spec1 = dccspec(uspec = uspec, dccOrder = c(1, 1), distribution = 'mvnorm')
    
    in_sample <- read.csv("data/etfs.csv", sep=";")[0:length_sample_period, ] %>% select(-Date)
    
    cl = makePSOCKcluster(4)
    multf = multifit(uspec, in_sample, cluster = cl)
    fit1 = dccfit(spec1, data = in_sample, fit.control = list(eval.se = TRUE), fit = multf, cluster = cl)          
    
    stopCluster(cl)

    #saveRDS(fit1, "backtesting/fitted_mgarch_model.rds")
    return(list(coef(fit1), residuals(fit1), sigma(fit1)))
    
}


rcov_forecast <- function(){
    fitted_model <- readRDS("backtesting/fitted_mgarch_model.rds")
    forecast <- dccforecast(fitted_model, n.ahead=1)
    return(rcov(forecast, output="matrix"))
}