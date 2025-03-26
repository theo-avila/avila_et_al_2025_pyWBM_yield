
* This code generates the supplemental county maps  for the following paper:
* Title: "Quantifying the Impacts of Compound Extremes on Agriculture and Irrigation Water Demand"
* by: Haqiqi, I., Grogan D.S., Hertel, T.W., Schlenker, W.
* This version: 2020-06-05
* contact: ihaqiqi@purdue.edu

*---- Downlaod US shape file
copy https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip cb_2018_us_county_500k.zip
unzipfile cb_2018_us_county_500k.zip
spshape2dta cb_2018_us_county_500k.shp, saving(usacounties) replace

* --- merge with data set of averages
use _ID _CX _CY GEOID using usacounties.dta, clear
generate fips = real(GEOID)
merge 1:1 fips using meanUS
drop if _merge == 2
drop _merge

* --- focus on the continetal US
gen state = floor(fips/1000)
drop if state == 2
drop if state == 15
drop if state >= 60

* --- Plot maps and save
grmap, activate

grmap mrso , clnumber(10) clmethod(custom)  clbreaks(0 25 50 75 100 150 200 250) title("Normal soil moisture (mm)") subtitle("average Apr-Sep 1981-2015") fcolor(khaki sandb orange green navy blue*2 purple*2) osize(vthin vthin vthin vthin vthin vthin vthin vthin vthin  )
graph export "normalSoilMositure.png", as(png) replace

gen compound = 1- shrDD29nl
grmap compound , clnumber(9) clmethod(custom) clbreaks(0 0.1 .2 .3 .4 .5 .6 .7 .8 1) title("share of compound extremes in all extremes") subtitle("average Apr-Sep 1981-2015") fcolor( khaki sandb yellow green navy  blue blue*2 red red*2) osize(vthin vthin vthin vthin vthin vthin vthin vthin)
graph export "compoundExtremes.png", as(png) replace

* End
