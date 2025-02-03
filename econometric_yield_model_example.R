setwd(getwd())

library(dplyr)
library(tidyr)
library(fixest)
library(ggplot2)
library(groupdata2)
library(arrow)

########################################################
# Load historical yield & climate data
########################################################
## Read and tidy county-level USDA data
usda <- read.csv('./data/usda_historical_yields.csv')
usda$fips <- sprintf("%05d", usda$fips)
usda$state <- sprintf("%02d", usda$state)
usda <- subset(usda, select=c('fips','year','yield', 'state'))

# Filter to counties with >= 30 years of data
usda <- merge(usda,  usda %>% count(fips))
usda <- filter(usda, n >= 30)

# Take log yield for model
usda$log_yield <- log(usda$yield)

## Read climate data
livneh <- read.csv('./data/livneh_historical_county_obs.csv')
livneh$fips <- sprintf("%05d", livneh$fips)
livneh$state <- substr(livneh$fips, 1, 2)

# Merge
df <- merge(livneh, usda)
df$year2 <- df$year **2

# Filter to eastern US counties to isolate rainfed (ie non-irrigated) maize
rainfed_states = c("01", "05", "09", "10", "12", "13", "17", "18", "19", "20", 
                   "21", "22", "23", "24", "25", "26", "27", "28", "29", "31", 
                   "33", "34", "36", "37", "38", "39", "40", "42", "44", "45", 
                   "46", "47", "48", "50", "51", "54", "55")

df <- df[df$state %in% rainfed_states,]


################################
# Fit the model
################################
# Schelnker & Roberts 2009
# https://www.pnas.org/doi/10.1073/pnas.0906865106
# We use county-level trends and then model the year-to-year variations
# via growing degree days (GDD), extreme degree days (EDD), season-total precip (prcp)
# and season-total precip squared (prcp2)
fx.mod_SR09 <- feols(log_yield ~ GDD + EDD + prcp + prcp2 | 
                       fips[year] + fips, df)
fx.mod_SR09 # look at coefficients

# Haqiqi et al. 2021
# https://hess.copernicus.org/articles/25/551/2021/hess-25-551-2021.pdf
# Again use county-level trends, now model the year-to-year variations
# via growing degree days (GDD), extreme degree days conditioned on soil moisture buckets,
# season-average soil moisture (SM_mean) and season-average soil squared (SM_mean2)
fx.mod_H21 <- feols(log_yield ~ GDD +
                      EDD_SM_75_below + EDD_SM_25_75_below + EDD_SM_0_25_norm + EDD_SM_25_75_above + EDD_SM_75_above +
                      SM_mean + SM_mean2 |
                      fips[year] + fips, df)
fx.mod_H21 # look at coefficients

# Compare results
etable(fx.mod_SR09, fx.mod_H21, fitstat =~ . + aic + bic + rmse)

