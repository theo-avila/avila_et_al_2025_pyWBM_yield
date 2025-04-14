import dask
import xarray as xr
import numpy as np
import pandas as pd
import datetime
import os 
import glob

def select_soilm(ds):
    '''
    preprocess each dataset to keep only the SoilM_0_100cm variable
    inputs:
    ds
    '''
    return ds[['SoilM_0_100cm']]
    
def average_hourly(date, model, nldas_path, out_path, log_path):
    """
    averages & saves files hourly at soilm 100 cm
    
    inputs:
    - date: datetime 
    - model: Vic, mos, or noah (nldas lsms)
    - nldas_path: where hourly nldas files are stored
    - out_path: where to save files
    - log_path: where to log exceptions 
    """
    
    yymmdd = date.strftime("%Y%m%d")
    output_file = f"{out_path}/NLDAS_{model}0125_H.A{yymmdd}.nc"
    
    # Skip processing if output already exists
    if os.path.isfile(output_file):
        return None
    else:
        # get all hourly files for the day (pattern must match the file naming convention)
        files = sorted(glob.glob(f"{nldas_path}/NLDAS_{model}0125_H.A{yymmdd}*"))
        
        # Check if there are at least 23 files, otherwise log and exit
        if len(files) == 0:
            with open(f"{log_path}/NLDAS_{model}0125_H_{yymmdd}.txt", "w") as f:
                f.write(f"Only {len(files)} files found")
            return None
        
        try:
            
            ds = xr.concat([xr.open_dataset(f) for f in files], dim="time")
            ds = ds[['SoilM_0_100cm']]
            
            daily_soilm = ds['SoilM_0_100cm'].resample(time="1D").mean(skipna=True)
            daily_soilm.to_netcdf(output_file)
        
        except Exception as e:
            with open(f"{log_path}/NLDAS_{model}0125_H_{yymmdd}.txt", "w") as f:
                f.write(str(e))
            return None
    

def day_iteration(year, model, nldas_path, out_path, log_path):
    '''
    given some year & nldas file path, iterates through each day
    inputs:
    year
    file_path_base
    '''
    start_dt = datetime.datetime(year, 1, 1, 0, 0)
    end_dt = datetime.datetime(year, 12, 31, 0, 0)
    year_downloads_dir = (f"{out_path}/{year}")
    os.makedirs(year_downloads_dir, exist_ok=True)
    
    while start_dt <= end_dt:
        average_hourly(start_dt, model, nldas_path, year_downloads_dir, log_path)
        start_dt += datetime.timedelta(days=1)


NLDAS_lsm = 'MOS'
nldas_path = f"/storage/group/pches/default/public/NLDAS/{NLDAS_lsm}/hourly/"
out_path = f"/storage/home/cta5244/work/pyWBM_yield_data/{NLDAS_lsm}_daily/"
log_path = f"/storage/home/cta5244/work/pyWBM_yield_data/{NLDAS_lsm}_daily/log_path/"
start_year, end_year = 1979, 2025

os.makedirs(log_path, exist_ok=True)

from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    account="open",
    cores=1,
    memory="8GiB",
    walltime="24:00:00",
)
cluster.scale(jobs=50)

from dask.distributed import Client

client = Client(cluster)

results = []
for year in np.arange(start_year, end_year, 1):
    out = dask.delayed(day_iteration)(year=year, model=NLDAS_lsm, nldas_path=nldas_path, out_path=out_path, log_path=log_path)
    results.append(out)


results = dask.compute(*results)



