import os
import numpy as np
import datetime
import xarray as xr
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

def aggregate_daily_tmin_tmax(date, input_dir, output_dir):
    """
    function iterates through each day, and saves to output_dir given input dir
    this is giving tmax & tmin values
    """
    yyyymmdd = date.strftime("%Y%m%d")
    tmin = None
    tmax = None

    for hour in range(24):
        hour_str = f"{hour:02d}00"
        file_name = f"NLDAS_FORA0125_H.A{yyyymmdd}.{hour_str}.020.nc"
        file_path = os.path.join(input_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        try:
            ds = xr.open_dataset(file_path)
            temp = ds.Tair.isel(time=0).copy(deep=True)
            ds.close()
            
            if tmin is None:
                tmin = temp
                tmax = temp
            else:
                tmin = xr.where(temp < tmin, temp, tmin)
                tmax = xr.where(temp > tmax, temp, tmax)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if tmin is None or tmax is None:
        print(f"No data available for {yyyymmdd}")
        return None

    # attach the day's date as the time coordinate
    date_val = np.datetime64(date)
    tmin = tmin.expand_dims(time=[date_val])
    tmax = tmax.expand_dims(time=[date_val])

    # creates an xarray Dataset containing tmin and tmax.
    daily_ds = xr.Dataset({
        "tmin": tmin,
        "tmax": tmax
    })

    output_file = os.path.join(output_dir, f"NLDAS_FORA0125_H.A{yyyymmdd}.nc")
    try:
        daily_ds.to_netcdf(output_file)
        print(f"Saved daily aggregation for {yyyymmdd} to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
    
    return output_file

input_dir = "/storage/group/pches/default/public/NLDAS/FORA/hourly/"
output_dir = "/storage/home/cta5244/work/pyWBM_yield_data/NCEPNARR_NLDAS_tmax_tmin/"

start_year = 1979
end_year = 2026

cluster = SLURMCluster(
    account="open",
    cores=1,
    memory="2GiB",
    walltime="24:00:00",
)
cluster.scale(jobs=50)
client = Client(cluster)

tasks = []
for year in range(start_year, end_year):

    output_year_dir = os.path.join(output_dir, str(year))
    os.makedirs(output_year_dir, exist_ok=True)
    
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)
    current_date = start_date
    while current_date <= end_date:
        task = dask.delayed(aggregate_daily_tmin_tmax)(current_date, input_dir, output_year_dir)
        tasks.append(task)
        current_date += datetime.timedelta(days=1)

results = dask.compute(*tasks)