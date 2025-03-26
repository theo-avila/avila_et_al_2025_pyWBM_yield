
* This code generates the required data for the following paper:
* Title: "Quantifying the Impacts of Compound Extremes on Agriculture and Irrigation Water Demand"
* by: Haqiqi, I., Grogan D.S., Hertel, T.W., Schlenker, W.
* This version: 2020-06-05
* contact: ihaqiqi@purdue.edu

** Note: you need to download soilMoistureData.dta before runnig this code. 

clear all

* ---- read the input data
use "soilMoistureData.dta"

* ---- generate new variables
* square of average daily soil moisture 
gen mrsoAprMay2 = mrsoAprMay^2
gen mrsoJunJul2 = mrsoJunJul^2 
gen mrsoAugSep2 = mrsoAugSep^2

* square of average daily soil moisture fraction
gen smf2 = smf^2
label var smf2 "Squar of mean daily soil moisture fraction"

* square of average daily ET
gen et2 = et^2
label var et2 "Squar of mean daily evapotranspiration (mm)"

* square of average daily soil moisture fraction 
gen mrso_alt2 = mrso_alt^2
label var mrso_alt2 "Square of mean daily soil moisture content (mm)**, alternative interpolation"

* probability of no compound stress
gen shrDD29nl = dd29smNl  / (dd29smLo+ dd29smHi +dd29smNl)
label var shrDD29nl "Share of DD29C at normal soil moisture"

* probability of no compound or individual stress
gen shrDD10_29nl = dd10_29smNl  / (dd10_29smLo+ dd10_29smHi +dd10_29smNl)
label var shrDD10_29nl "Share of DD10-29C at normal soil moisture"

 * dummy variables
gen d10 = 0
replace d10 = 1 if shrDD29nl < 0.1

gen d90 = 0
replace d90 = 1 if shrDD29nl > 0.9
* ---- cleaning USDA data on irrigated area
replace cornAreaIrrig = cornArea if cornAreaIrrig ==. & longitude < -100
replace cornAreaNonIrrig = cornArea if cornAreaNonIrrig ==. & longitude > -100
replace cornAreaIrrig = 0 if cornAreaIrrig ==.
replace cornAreaNonIrrig = 0 if cornAreaNonIrrig ==.
gen irAreaShr = cornAreaIrrig / ( cornAreaIrrig + cornAreaNonIrrig )
label var irAreaShr "Share of irrigated corn area"

* End
