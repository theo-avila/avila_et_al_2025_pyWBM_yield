import dask
import xarray as xr
import numpy as np
import pandas as pd
import datetime
import os 

def select_soilm(ds):
    '''
    preprocess each dataset to keep only the SoilM_0_100cm variable
    inputs:
    ds
    '''
    return ds[['SoilM_0_100cm']]
    
def average_hourly(start_dt, file_path_base, year_downloads_dir):
    '''
    takes base file path and averages from hourly to daily using xarray resampling. Also creates dir for files
    inputs:
    datetime as current_dt 
    outputs:
    daily average in appropriate subdirectory
    '''
    os.makedirs(year_downloads_dir, exist_ok=True)
    yyyymmdd = start_dt.strftime("%Y%m%d")
    file_pattern = f"{file_path_base}{yyyymmdd}.*.020.nc"

    try:
        ds = xr.open_mfdataset(file_pattern, combine='by_coords')
        
        daily_soilm = ds.resample(time='1D').mean(skipna=True)['SoilM_0_100cm']
        
        output_path = os.path.join(year_downloads_dir, f"NLDAS_0_100_soilm_{NLDAS_lsm}0125_H.A{yyyymmdd}.020.nc")
        
        daily_soilm.compute().to_netcdf(output_path)
        
    except OSError:
        print(f"No files to open for pattern: {file_pattern}. Skipping this date.")
    except Exception as e:
        print(f"Unexpected error processing {yyyymmdd}: {e}")
    

def day_iteration(year, file_path_base, downloads_dir):
    '''
    given some year & nldas file path, iterates through each day
    inputs:
    year
    file_path_base
    '''
    start_dt = datetime.datetime(year, 1, 1, 0, 0)
    end_dt = datetime.datetime(year, 12, 31, 0, 0)
    year_downloads_dir = (f"{downloads_dir}/{year}")
    
    while start_dt <= end_dt:
        average_hourly(start_dt, file_path_base, year_downloads_dir)
        start_dt += datetime.timedelta(days=1)


NLDAS_lsm = 'VIC'
file_path_base = f"/storage/group/pches/default/public/NLDAS/{NLDAS_lsm}/hourly/NLDAS_{NLDAS_lsm}0125_H.A"
downloads_dir = f"/storage/home/cta5244/work/pyWBM_yield_data/{NLDAS_lsm}_daily/"
start_year, end_year = 1979, 2025

from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    # account="pches",
    account="open",
    cores=1,
    memory="4GiB",
    walltime="05:00:00",
)

cluster.scale(jobs=20)

from dask.distributed import Client

client = Client(cluster)

results = []
for year in np.arange(start_year, end_year, 1):
    out = dask.delayed(day_iteration)(year=year, file_path_base=file_path_base, downloads_dir=downloads_dir)
    results.append(out)

results = dask.compute(*results)
