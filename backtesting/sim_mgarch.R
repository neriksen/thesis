suppressMessages(library(rmgarch))
suppressMessages(library(parallel))
suppressMessages(library(quantmod))
suppressMessages(library(tidyverse))
suppressMessages(library(here))


Sim_mgarch <- function(tickers, len_out_of_sample, ugarch_model, ugarch_dist_model, garchOrder){

    full_sample <- read.csv(paste(here(),"/data/return_data_stable.csv", sep=""), sep=";") %>% select(-Date) %>% select(one_of(tickers))
    len_in_sample <- nrow(full_sample) - len_out_of_sample
    in_sample <- head(full_sample, len_in_sample)
    num_assets <- ncol(in_sample)

    xspec = ugarchspec(mean.model = list(armaOrder = c(0, 0)), variance.model = list(garchOrder = garchOrder, model = ugarch_model), distribution.model = ugarch_dist_model)
    uspec = multispec(replicate(num_assets, xspec))
    spec1 = dccspec(uspec = uspec, dccOrder = c(1, 1), distribution = 'mvt')
    
    cl = makePSOCKcluster(4)
    multf = multifit(uspec, in_sample, cluster = cl)
    fit1 = dccfit(spec1, data = in_sample, fit.control = list(eval.se = TRUE), fit = multf, cluster = cl)   
    stopCluster(cl)

    dccsim(fit1, n.sim = 2000, n.start = 1000, m.sim = 1, startMethod = c("unconditional" ))
    
    return(list(coef(fit1), fitted(dccsim), sigma(dccsim)))
}