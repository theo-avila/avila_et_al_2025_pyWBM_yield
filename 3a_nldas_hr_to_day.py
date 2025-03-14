import dask
import xarray as xr
import numpy 
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
    
def average_hourly(start_dt, file_path_base, downloads_dir):
    '''
    takes base file path and averages from hourly to daily using xarray resampling. Also creates dir for files
    inputs:
    datetime as current_dt 
    outputs:
    daily average in appropriate subdirectory
    '''
    os.makedirs(downloads_dir, exist_ok=True)
    current_dt = start_dt
    yyyymmdd = current_dt.strftime("%Y%m%d")
    
    file_pattern = f"{file_path_base}{yyyymmdd}.*.020.nc"
    ds = xr.open_mfdataset(file_pattern, combine='by_coords', preprocess=select_soilm)
    daily_soilm = ds.resample(time='1D').mean()

    output_path = os.path.join(downloads_dir, f"NLDAS_0_100_soilm_{NLDAS_lsm}0125_H.A{yyyymmdd}.020.nc")
    daily_soilm.to_netcdf(output_path)

def day_iteration(year, file_path_base, downloads_dir):
    '''
    given some year & nldas file path, iterates through each day
    inputs:
    year
    file_path_base
    '''
    start_dt = datetime.datetime(year, 1, 1, 0, 0)
    end_dt = datetime.datetime(year, 1, 2, 0, 0)
    while start_dt <= end_dt:
        average_hourly(start_dt, file_path_base, downloads_dir)
        start_dt += datetime.timedelta(days=1)


NLDAS_lsm = 'MOS'
file_path_base = f"/storage/group/pches/default/public/NLDAS/{NLDAS_lsm}/hourly/NLDAS_{NLDAS_lsm}0125_H.A"
downloads_dir = f"/storage/home/cta5244/work/pyWBM_yield_data/{NLDAS_lsm}_daily/"


from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    # account="pches",
    account="open",
    cores=1,
    memory="8GiB",
    walltime="05:00:00",
)

cluster.scale(jobs=10) 

from dask.distributed import Client

client = Client(cluster)


results = []
for year in np.arange(start_year, end_year, 1):
    out = dask.delayed(day_iteration)(year=year, file_path_base=file_path_base, downloads_dir=downloads_dir)
    results.append(out)


results = dask.compute(*results)