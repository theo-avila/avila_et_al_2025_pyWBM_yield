# avila_et_al_2026_pyWBM_yield
**Compounding Uncertainity in Coupled Yield Models** <br>
Theo C. Avila<sup>1*</sup>, David Lafferty<sup>2</sup>, Tahsina Alam<sup>3</sup>, Ryan L. Sriver<sup>4</sup><br>

<sup>1</sup> Department of Physics, University of Illinois at Urbana‐Champaign, Urbana, IL, United States of America<br>
<sup>2</sup> Department of Biological and Environmental Engineering, Cornell University, Ithaca, NY, United States of America<br>
<sup>3</sup> Department of Civil and Environmental Engineering, University of Illinois at Urbana‐Champaign, Urbana, IL, United States of America<br>
<sup>4</sup> Department of Climate, Meteorology and Atmospheric Sciences, University of Illinois at Urbana‐Champaign, Urbana, IL, United States of America<br>

<sup>*</sup> corresponding author: ctavila2@illinois.edu


### Input data
| Dataset | Link | DOI | Notes |
|---------|------|-----|-------|
| NLDAS-2 forcing inputs | https://disc.gsfc.nasa.gov/datasets/NLDAS_FORA0125_H_002/summary | https://doi.org/10.5067/6J5LHHOHZHN4 | We use `Tair` and calculate daily `tmax` and `tmin` |
| NLDAS-2 model outputs | VIC: https://disc.gsfc.nasa.gov/datasets/NLDAS_VIC0125_H_002/summary <br> Noah: https://disc.gsfc.nasa.gov/datasets/NLDAS_NOAH0125_H_002/summary <br> Mosaic: https://disc.gsfc.nasa.gov/datasets/NLDAS_NOAH0125_H_002/summary | VIC: https://doi.org/10.5067/ELBDAPAKNGJ9 <br> Noah: https://doi.org/10.5067/EN4MBWTCENE5 <br> Mosaic: https://doi.org/10.5067/47Z13FNQODKV | We use `SOILM0_100cm` from VIC, and `SOILM` from Noah and Mosaic. We also subset with nasa earth dec for netcdf files and only `SOILM` | 
| LOCA2 CMIP6 | https://cirrus.ucsd.edu/~pierce/LOCA2/NAmer/GFDL-CM4/0p0625deg/r1i1p1f1/ |  | We use `tmax` and `tmin` |
| pyWBM | https://github.com/david0811/pyWBM | | We use `soilMoist`|
| USDA Crop Production | https://www.nass.usda.gov/ |  |  |
| Livneh County Shape Files |  |  |  |

### Script Order
- Some downloading & preprocessing was done offline
- Many files are not saved within the github repo so filepaths will need to be altered when neccesary
- shell scripts used when applicable
- Computations for this research were performed on the Pennsylvania State University’s Institute for Computational and Data Sciences’ Roar Collab supercomputer 

| Script | Description |
|--------|-------------|
| 1a_data_processing_gdd.py | processes nldas forcing data for daily tmax & tmin (uses sbatch 2a) |
| 2a_gdd_edd_calculation.ipynb | takes nldas forcing data and calculates growing degree days and extreme degree days |
| 3a_nldas_hr_to_day.py | averages hourly -> daily nldas data (using sbatch 3a) |
| 3b_compound_extremes_processing.ipynb | calculates mean soil moisture over historical period for nldas models |
| 4_aggregation_fips.ipynb | takes gridded data to fips level for all 2a model inputs |
| 5_future_processing.py | calculates yearly fixed effect model regressors, aggregates for LOCA2 2015-2100 and pyWBM runs |
| 6_implementation_2a.ipynb | implementation of haqiqi 2021 model 2a  |



## Recreate the environment

To recreate the environment used in this research:

```bash
mamba env create -f environment_avila26.yml
conda activate environmental_avila26
```
## Econometric Model
We are using a fixed effects model on Log(Corn Yields) (Haqiqi et al. 2021; Lafferty et al. 2021): <br>


$$logY_{i,t} = \beta_{1}GDD_{i,t} + EDD_{binned} + \beta_{2}SM + \beta_{3}SM^{2} + f_{s}(t) + g_{s}(t) + c_{i}$$

EDD_{binned} = \beta_{a}EDD^{a}_{i,t} + \beta_{b}EDD^{b}_{i,t} +\beta_{c}EDD^{c}_{i,t} + \beta_{d}EDD^{d}_{i,t} + \beta_{e}EDD^{e}_{i,t}
$$

$$EDD_{binned}$$ is Extreme Degree Days with corresponding soil moisture values.  <br>
$$a$$ soil moisture above +75mm about normal <br>
$$b$$ soil moisture +25mm to +75mm about normal <br>
$$c$$ soil moisture is between -25mm to +25mm about normal <br>
$$d$$ soil moisture is between -75mm to -25mm about normal <br>
$$e$$ soil moisture is below -75mm about normal <br>
$$GDD_{i,t}$$ is growing degree days, i denotes state (county in this case), t denotes time which is sum over year <br>
$$c_{i}$$ is county level terms for time invariant effects <br>
$$g_{s}(t)$$ linear state level trend  <br>
$$f_{s}(t)$$ quadratic state level trend  <br>

