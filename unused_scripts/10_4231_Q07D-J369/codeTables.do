
* This code generates the Tables included in the following paper:
* Title: "Quantifying the Impacts of Compound Extremes on Agriculture and Irrigation Water Demand"
* by: Haqiqi, I., Grogan D.S., Hertel, T.W., Schlenker, W.
* This version: 2020-06-05
* contact: ihaqiqi@purdue.edu

* Note: you need to upload the data using codeData.do before runnig this code. 

* this requires asdoc being installed
net install asdoc, from(http://fintechprofessor.com) replace

* table 1: summary
asdoc sum dday10_29C dday29C prec mrso nd0xSM_m_gt025 nd0xSM_m_lt025 smdPos  smdNeg dd10_29smLo dd10_29smHi dd10_29smNl dd29smLo dd29smHi dd29smNl mrLo mrHi mrNl et smf mrso_alt mrsoAprMay  mrsoJunJul  mrsoAugSep, save(table01) dec(2) label replace reset

* table 2: basic models
asdoc xtreg logCornYield dday10_29C dday29C prec prec2  i.state#c.t i.state#c.t2, fe cluster(state) save(table02) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2 _cons)  stat(aic bic) label replace reset
asdoc xtreg logCornYield dday10_29C dday29C mrso mrso2  i.state#c.t i.state#c.t2, fe cluster(state) save(table02) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label
asdoc xtreg logCornYield dday10_29C dday29C nd0xSM_m_gt025 nd0xSM_m_lt025 i.state#c.t i.state#c.t2, fe cluster(state) save(table02) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label 
asdoc xtreg logCornYield dday10_29C dday29C smdPos  smdNeg   i.state#c.t i.state#c.t2, fe cluster(state) save(table02) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label 

* table 3: split heat index to individual and compound extremes
asdoc xtreg logCornYield dday10_29C gdd29smxxx_75b gdd29sm75b_25b gdd29sm25b_25a gdd29sm25a_75a gdd29sm75a_xxx mrso mrso2  i.state#c.t i.state#c.t2 , fe cluster(state) save(table03) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label replace reset

* table 4: split soil moisture  to individual and compound extremes
asdoc xtreg logCornYield dday10_29C  dday29C mrNlTvg25_50 mrHiTvg25_50 mrLoTvg25_50 mrLoTvg00_25 mrHiTvg00_25 mrNlTvg00_25 i.state#c.t i.state#c.t2 , fe cluster(state) save(table04) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label replace reset


* appendix

* table a.1: controlling for normal mrso
asdoc xtreg logCornYield c.shrDD10_29nl#c.dday10_29C dday10_29C c.shrDD29nl#c.dday29C dday29C mrNl mrHi mrLo  i.state#c.t i.state#c.t2 , fe cluster(state) save(table11) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)   stat(aic bic) label replace

* table a.2: controlling for irrigation
asdoc xtreg logCornYield dday10_29C  dday29C c.dday29C#c.irAreaShr c.dday29C#c.shrDD29nl  mrLo mrHi mrNl i.state#c.t i.state#c.t2, fe cluster(state) save(table12) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label  replace reset

* table a.3: summary stat
asdoc sum dday10_29C dday29C prec mrso nd0xSM_m_gt025 nd0xSM_m_lt025 smdPos  smdNeg dd10_29smLo dd10_29smHi dd10_29smNl dd29smLo dd29smHi dd29smNl mrLo mrHi mrNl et smf mrso_alt mrsoAprMay  mrsoJunJul  mrsoAugSep if longitude > -100,  dp(2) save(table01e) dec(2) label replace abb(.)
asdoc sum dday10_29C dday29C prec mrso nd0xSM_m_gt025 nd0xSM_m_lt025 smdPos  smdNeg dd10_29smLo dd10_29smHi dd10_29smNl dd29smLo dd29smHi dd29smNl mrLo mrHi mrNl et smf mrso_alt mrsoAprMay  mrsoJunJul  mrsoAugSep if longitude < -100,  dp(2) save(table01w) dec(2) label replace abb(.)

* table a.4: Easter US
asdoc xtreg logCornYield dday10_29C dday29C prec prec2  i.state#c.t i.state#c.t2 if longitude > -100, fe cluster(state) save(table05) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2 _cons)  stat(aic bic) label replace reset
asdoc xtreg logCornYield dday10_29C dday29C mrso mrso2  i.state#c.t i.state#c.t2 if longitude > -100, fe cluster(state) save(table05) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label
asdoc xtreg logCornYield dday10_29C dday29C mrLo mrHi mrNl   i.state#c.t i.state#c.t2 if longitude > -100, fe cluster(state) save(table05) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label
asdoc xtreg logCornYield c.shrDD10_29nl#c.dday10_29C dday10_29C c.shrDD29nl#c.dday29C dday29C mrNl mrHi mrLo  i.state#c.t i.state#c.t2 if longitude > -100, fe cluster(state) save(table05) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)   stat(aic bic) label 

* table a.5: Western US
asdoc xtreg logCornYield dday10_29C dday29C prec prec2  i.state#c.t i.state#c.t2 if longitude < -100, fe cluster(state) save(table06) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2 _cons)  stat(aic bic) label replace 
asdoc xtreg logCornYield dday10_29C dday29C mrso mrso2  i.state#c.t i.state#c.t2 if longitude < -100, fe cluster(state) save(table06) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label
asdoc xtreg logCornYield dday10_29C dday29C mrLo mrHi mrNl   i.state#c.t i.state#c.t2 if longitude < -100, fe cluster(state) save(table06) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label
asdoc xtreg logCornYield c.shrDD10_29nl#c.dday10_29C dday10_29C c.shrDD29nl#c.dday29C dday29C mrNl mrHi mrLo  i.state#c.t i.state#c.t2 if longitude < -100, fe cluster(state) save(table06) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)   stat(aic bic) label 

* table a.6: bi-monthly
asdoc xtreg logCornYield dday10_29C dday29C mrsoAprMay mrsoAprMay2 mrsoJunJul mrsoJunJul2 mrsoAugSep mrsoAugSep2 i.state#c.t i.state#c.t2, fe cluster(state) save(table07) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label replace
asdoc xtreg logCornYield dday10_29C dday29C mrsoAprMay mrsoAprMay2 mrsoJunJul mrsoJunJul2 mrsoAugSep mrsoAugSep2 i.state#c.t i.state#c.t2  if longitude < -100, fe cluster(state) save(table07) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2) stat(aic bic) label
asdoc xtreg logCornYield dday10_29C dday29C mrsoAprMay mrsoAprMay2 mrsoJunJul mrsoJunJul2 mrsoAugSep mrsoAugSep2 i.state#c.t i.state#c.t2  if longitude > -100, fe cluster(state) save(table07) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2) stat(aic bic) label

* table a.7: Other metrics of water
asdoc xtreg logCornYield dday10_29C dday29C smf smf2  i.state#c.t i.state#c.t2, fe cluster(state) save(table08) nest dec(5) fs(8) drop(i.state#c.t i.state#c.t2) stat(aic bic) label replace reset
asdoc xtreg logCornYield dday10_29C dday29C smf smf2 smf_sd  i.state#c.t i.state#c.t2, fe cluster(state) save(table08) nest dec(5) fs(8) drop(i.state#c.t i.state#c.t2) stat(aic bic) label 
asdoc xtreg logCornYield dday10_29C dday29C et et2  i.state#c.t i.state#c.t2, fe cluster(state) save(table08) nest dec(5) fs(8) drop(i.state#c.t i.state#c.t2) stat(aic bic) label
asdoc xtreg logCornYield dday10_29C dday29C et et2 et_sd  i.state#c.t i.state#c.t2, fe cluster(state) save(table08) nest dec(5) fs(8) drop(i.state#c.t i.state#c.t2) stat(aic bic) label
asdoc xtreg logCornYield dday10_29C dday29C mrso_alt mrso_alt2  i.state#c.t i.state#c.t2, fe cluster(state) save(table08) nest dec(5) fs(8) drop(i.state#c.t i.state#c.t2) stat(aic bic) label

* table a.8 split heat index
asdoc xtreg logCornYield dday10_29C gdd29smxxx_75b gdd29sm75b_25b gdd29sm25b_25a gdd29sm25a_75a gdd29sm75a_xxx mrso mrso2  i.state#c.t i.state#c.t2 , fe cluster(state) save(table09) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label replace reset
asdoc xtreg logCornYield dday10_29C gdd29smxxx_75b gdd29sm75b_25b gdd29sm25b_25a gdd29sm25a_75a gdd29sm75a_xxx mrso mrso2  i.state#c.t i.state#c.t2 if longitude < -100 , fe cluster(state) save(table09) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label 
asdoc xtreg logCornYield dday10_29C gdd29smxxx_75b gdd29sm75b_25b gdd29sm25b_25a gdd29sm25a_75a gdd29sm75a_xxx mrso mrso2  i.state#c.t i.state#c.t2 if longitude > -100 , fe cluster(state) save(table09) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label 

* table a.9: split soil moisture
asdoc xtreg logCornYield dday10_29C  dday29C mrNlTvg25_50 mrHiTvg25_50 mrLoTvg25_50 mrLoTvg00_25 mrHiTvg00_25 mrNlTvg00_25 i.state#c.t i.state#c.t2 , fe cluster(state) save(table10) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label replace reset
asdoc xtreg logCornYield dday10_29C  dday29C mrNlTvg25_50 mrHiTvg25_50 mrLoTvg25_50 mrLoTvg00_25 mrHiTvg00_25 mrNlTvg00_25 i.state#c.t i.state#c.t2 if longitude < -100 , fe cluster(state) save(table10) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label 
asdoc xtreg logCornYield dday10_29C  dday29C mrNlTvg25_50 mrHiTvg25_50 mrLoTvg25_50 mrLoTvg00_25 mrHiTvg00_25 mrNlTvg00_25 i.state#c.t i.state#c.t2 if longitude > -100 , fe cluster(state) save(table10) nest dec(7) fs(8) drop(i.state#c.t i.state#c.t2)  stat(aic bic) label 

* End
