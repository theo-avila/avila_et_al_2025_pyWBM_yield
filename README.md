# avila_et_al_2026_pyWBM_yield
**Compounding Uncertainity in Coupled Yield Models** <br>
Theo C. Avila<sup>1*</sup>, David Lafferty<sup>2</sup>, Tahsina Alam<sup>3</sup>, Ryan L. Sriver<sup>4</sup><br>

<sup>1</sup> Department of Physics, University of Illinois at Urbana‐Champaign, Urbana, IL, United States of America<br>
<sup>2</sup> Department of Biological and Environmental Engineering, Cornell University, Ithaca, NY, United States of America<br>
<sup>3</sup> Department of Civil and Environmental Engineering, University of Illinois at Urbana‐Champaign, Urbana, IL, United States of America<br>
<sup>4</sup> Department of Climate, Meteorology and Atmospheric Sciences, University of Illinois at Urbana‐Champaign, Urbana, IL, United States of America<br>

<sup>*</sup> corresponding author: ctavila2@illinois.edu


this is the repository for avila et al 2026 pyWBM yields

### Input data
| Dataset | Link | DOI | Notes |
|---------|------|-----|-------|
| NLDAS-2 forcing inputs | https://disc.gsfc.nasa.gov/datasets/NLDAS_FORA0125_H_002/summary | https://doi.org/10.5067/6J5LHHOHZHN4 | We use `Tair` and calculate daily `tmax` and `tmin` |
| NLDAS-2 model outputs | VIC: https://disc.gsfc.nasa.gov/datasets/NLDAS_VIC0125_H_002/summary <br> Noah: https://disc.gsfc.nasa.gov/datasets/NLDAS_NOAH0125_H_002/summary <br> Mosaic: https://disc.gsfc.nasa.gov/datasets/NLDAS_NOAH0125_H_002/summary | VIC: https://doi.org/10.5067/ELBDAPAKNGJ9 <br> Noah: https://doi.org/10.5067/EN4MBWTCENE5 <br> Mosaic: https://doi.org/10.5067/47Z13FNQODKV | We use `SOILM0_100cm` from VIC, and `SOILM` from Noah and Mosaic. We also subset with nasa earth dec for netcdf files and only `SOILM` | 
| LOCA2 CMIP6 | https://cirrus.ucsd.edu/~pierce/LOCA2/NAmer/GFDL-CM4/0p0625deg/r1i1p1f1/ |  | We use `tmax` and `tmin` |
| pyWBM | | | |
| USDA Crop Production |  |  |  |
| Livneh County Shape Files |  |  |  |

