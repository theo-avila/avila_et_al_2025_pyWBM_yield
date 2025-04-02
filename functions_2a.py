import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import dask
import os

def degreeDays(single_day, degree_day_method, kelvin_starting=True):
    '''
    Calculates growing degree days, using a piecewise function, ASSUMING KELVIN STARTING
    
    inputs:
    dataset with tmax & tmin as variables as single_day
    degree_day_method, which is either gdd or edd, 10.0 degrees or 29.0 degrees respectively (in degree celsius)
    outputs:
    growing degree days for dataset
    (Haqiqi et al. 2021; D’Agostino and Schlenker, 2015)
    '''
    single_day = single_day.copy(deep=True)
    
    if kelvin_starting:
        single_day = single_day - 273.15

    if degree_day_method == 'gdd':
        corn_baseline = 10.0
        # caps at 29.0 Degrees for iman 2021 purposes
        single_day['tmax'] = xr.where(single_day.tmax > 29.0, 29.0, single_day.tmax)
        
    elif degree_day_method == 'edd':
        corn_baseline = 29.0
        
    else:
        raise Exception(f'Invalid degree_day_method "{degree_day_method}", please specify either "gdd" or "edd"')
        
    gdd_ds = xr.full_like(single_day.tmax, 0.0)

    # b <= tmin
    b_lower_tmin = (single_day.tmin + single_day.tmax) / 2 - corn_baseline
    gdd_ds = xr.where(single_day.tmin >= corn_baseline, b_lower_tmin, gdd_ds)
    
    # tmin < b <= tmax
    arccos_arg = (2 * corn_baseline - single_day.tmax - single_day.tmin) / (single_day.tmax - single_day.tmin)
    arccos_arg = arccos_arg.clip(min=-1, max=1)
    
    t_bar = np.arccos(arccos_arg)
    tmin_lower_b_lower_tmax = (
        t_bar / np.pi * ((single_day.tmax + single_day.tmin) / 2 - corn_baseline)
        + (single_day.tmax - single_day.tmin) / (2 * np.pi) * np.sin(t_bar)
    )
    
    gdd_ds = xr.where((single_day.tmin < corn_baseline) & (corn_baseline <= single_day.tmax), tmin_lower_b_lower_tmax, gdd_ds)
    
    # tmax < b 
    gdd_ds = xr.where((single_day.tmax < corn_baseline), 0, gdd_ds)
    
    return gdd_ds

def yearlyCalculationSum(year, filein_base_yearly):
    '''
    calculates growing degree days, & extreme degree days from a year & file_path.
    This is for the growing season (April 1 to September 31), per Haqiqi et al. 2020
    
    inputs:
    year
    filein_base_yearly that with daily tmax & tmin values
    outputs:
    yearly summed gdd & edd as nc file, saved @ some location as single year nc file
    
    '''
    # we increase the growing season by 1 month on each side (at least to save the files so no important information is lost
    
    try:
        os.mkdir(f"{filein_base_yearly}/{year}/")
    except Exception:
        pass
        
    start_dt = datetime.datetime(year, 1, 1, 0, 0)
    end_dt = datetime.datetime(year, 12, 31, 0, 0)
    
    while start_dt <= end_dt:
        yyyymmdd = start_dt.strftime("%Y%m%d")
        filein_i = f"{filein_base_yearly}/{year}/NLDAS_FORA0125_H.A{yyyymmdd}.nc"
        
        try:
            dataset_i  = xr.open_dataset(filein_i)
            gdd = degreeDays(dataset_i, 'gdd').drop_vars('time', errors='ignore')
            edd = degreeDays(dataset_i, 'edd').drop_vars('time', errors='ignore')
            
            combined_dataset = xr.Dataset({
            "gdd": gdd,
            "edd": edd
             })
            dataset_i.close()
            combined_dataset.to_netcdf(f"{filein_base_yearly}/{year}/NLDAS_FORA0125_H.A{yyyymmdd}_dday.nc", mode='w')
            
        except Exception:
            raise Exception(f"issue with downloading or accessing underlying files at {filein_i}")
        
        start_dt += datetime.timedelta(days=1)
    
    return