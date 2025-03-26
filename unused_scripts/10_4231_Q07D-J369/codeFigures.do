
* This code generates some of the figures for the following paper:
* Title: "Quantifying the Impacts of Compound Extremes on Agriculture and Irrigation Water Demand"
* by: Haqiqi, I., Grogan D.S., Hertel, T.W., Schlenker, W.
* This version: 2020-06-05
* contact: ihaqiqi@purdue.edu

* Note: you need to upload the data using codeData.do before runnig this code. 

* ---- figures ----
set scheme s1mono

* scatter plot for seasonal mean soil moisture versus precipitation
twoway (scatter prec mrso, mcolor(black) msize(zero) msymbol(plus) mfcolor(black) mlcolor(black))
graph export "Fig03.pdf", as(pdf) replace
graph export "Fig03.png", as(png) replace 

* scatter plot for seasonal mean soil moisture versus ET
twoway (scatter et mrso, mcolor(black) msize(zero) msymbol(plus) mfcolor(black) mlcolor(black))
graph export "FigA02.pdf", as(pdf) replace 
graph export "FigA02.png", as(png) replace 

* scatter plot for seasonal mean soil moisture versus soil moisture fraction 
twoway (scatter smf mrso, mcolor(black) msize(zero) msymbol(plus) mfcolor(black) mlcolor(black))
graph export "FigA03.pdf", as(pdf) replace 
graph export "FigA03.png", as(png) replace 

* scatter plot for seasonal mean soil moisture versus alternative interpolations
twoway (scatter mrso_alt mrso, mcolor(black) msize(zero) msymbol(plus) mfcolor(black) mlcolor(black))
graph export "FigA01.pdf", as(pdf) replace
graph export "FigA01.png", as(png) replace 

* scatter plot for seasonal mean soil moisture versus seasonal heat index
twoway (scatter dday10C mrso, mcolor(black) msize(zero) msymbol(plus) mfcolor(black) mlcolor(black))
graph export "FigA04.pdf", as(pdf) replace 
graph export "FigA04.png", as(png) replace 

* End
