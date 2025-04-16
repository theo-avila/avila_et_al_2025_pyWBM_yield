import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import dask
import os
import glob as glob
import cftime 
import warnings
import geopandas as gpd
import xagg as xa
warnings.filterwarnings("ignore")
from functions_2a import degreeDays, yearlyCalculationSum

# get cmip6 model names used in loca2, the full path for reference = "/storage/group/pches/default/public/LOCA2/ACCESS-CM2/0p0625deg/r1i1p1f1/ssp245/tasmin"
base_loca_paths_for_models = "/storage/group/pches/default/public/LOCA2/"
models = sorted(glob.glob(f"{base_loca_paths_for_models}*"))
model_names = [os.path.basename(m) for m in models][:-2]

# ssp scenarios used in pyWBM are 245 and 370
ssps = ["245", "370"]

# intitalizataions, only using r1i1p1f1 for now, some runs have more than 1 init
initializations = ["r1i1p1f1"]

# loca2 is in chunks of ~30 years 
time_frames = ["2015-2044", "2045-2074", "2075-2100"]

nldas_lsm = "MOS"

# this is the soil_moiture base path
pyWBM_file_path_base = "/storage/group/pches/default/users/dcl5300/wbm_soilM_uc_2024_DATA/projections/eCONUS/out/LOCA2"

# soil moisture historical normal 
soil_moisture_normal_file_path = f"/storage/home/cta5244/work/avila_et_al_2025_pyWBM_yield/data/{nldas_lsm}_seasonal_average_alltime_average_soilmoisture.nc"

# some arbritrary pyWBM run for regridding
arbritrary_pyWBM_run = f'/storage/group/pches/default/users/dcl5300/wbm_soilM_uc_2024_DATA/projections/eCONUS/out/LOCA2/ACCESS-CM2_r1i1p1f1_ssp245_VIC_kge.nc'

# county file path for aggregation to county level
county_shp_path = "/storage/work/cta5244/avila_et_al_2025_pyWBM_yield/shape_files/counties_contig_plot.shp"

def process_year(ds_combined, year, ds_soilpyWBM_regrid, ds_soil_normal_on_wbm_grid, ds_soilpyWBM_initial, model_name_i, initialization_i, ssp_i):
    '''
    this is processing each year, input from the process model function and its usefull-ness is to allow smaller tasks for more dask workers 
    inputs
    - ds combined which is just loca2 tamx & tmin
    - year single year for parallel-ness
    - ds_soil_normal_on_wbm_grid normal soil moisture
    - ds_soilpyWBM_initial as the appropriate pyWBM scenario which is not being opened repeatedly
    - model_name_i (cmip6 model)
    - initialization_i (initlization scenario)
    - ssp_i (the ssp scenario of interest)
    outputs
    - combined_dataset_bins which is a single year of processesd values
    '''
    ds_slice = ds_combined.sel(time=slice(f"{year}-04-01", f"{year}-09-30"))
    ds_slice = ds_slice.assign_coords(lon=((ds_slice.lon + 180) % 360) - 180).sortby("lon")
    land_mask = ds_slice.tmax.isel(time=0).notnull().copy(deep=True)
    
    # calculate degree days (assuming degreeDays is defined elsewhere)
    gdd_future = degreeDays(ds_slice, 'gdd')

    # Convert year to datetime (e.g. Jan 1st of each year)
    gdd_future_mask = gdd_future.where(land_mask)
    gdd_future_regrid = gdd_future_mask.interp(
        lat=ds_soilpyWBM_regrid.lat, 
        lon=ds_soilpyWBM_regrid.lon, 
        method="linear",
        kwargs={"fill_value": np.nan}
    ).transpose('lat', 'lon', 'time')
    
    edd_future = degreeDays(ds_slice, 'edd')
    edd_future_mask = edd_future.where(land_mask)
    edd_future_regrid = edd_future_mask.interp(
        lat=ds_soilpyWBM_regrid.lat, 
        lon=ds_soilpyWBM_regrid.lon, 
        method="linear",
        kwargs={"fill_value": np.nan}
    ).transpose('lat', 'lon', 'time')

    ds_soilpyWBM_year = ds_soilpyWBM_initial.sel(time=slice(f"{year}-04-01", f"{year}-09-30")).transpose('lat', 'lon', 'time')

    ds_soil_normal_on_wbm_grid_masked = ds_soil_normal_on_wbm_grid.where(~np.isnan(ds_soilpyWBM_year))
    ds_soil_normal_on_wbm_grid_masked = ds_soil_normal_on_wbm_grid_masked.transpose('lat', 'lon', 'time')
    
    deviation_from_normal = ds_soilpyWBM_year - ds_soil_normal_on_wbm_grid_masked.soilMoist
    
    #try:
    mask = ~np.isnan(ds_soilpyWBM_year.isel(time=0).soilMoist)
    edd_future_masked = edd_future_regrid.where(mask)
    gdd_future_masked = gdd_future_regrid.where(mask)
    # Define bin conditions and compute all bins in one pass using a dictionary comprehension
    
    if isinstance(edd_future_masked, xr.Dataset):
        edd_future_masked = edd_future_masked['soilMoist']
        
    if isinstance(gdd_future_masked, xr.Dataset):
        gdd_future_masked = gdd_future_masked['soilMoist']
        
    edd_future_masked = edd_future_masked.assign_coords(time=edd_future_masked.time.dt.floor('D'))
    deviation_from_normal, edd_future_masked = xr.align(deviation_from_normal, edd_future_masked, join='right')
    
    # Create binned EDD DataArrays based on the deviation thresholds.
    ds_bin_plus75 = xr.where(deviation_from_normal.soilMoist >= 75, edd_future_masked, 0)
    ds_bin_plus25_75 = xr.where((deviation_from_normal.soilMoist < 75) & (deviation_from_normal.soilMoist > 25), edd_future_masked, 0)
    ds_bin_minus25_plus25 = xr.where((deviation_from_normal.soilMoist <= 25) & (deviation_from_normal.soilMoist >= -25), edd_future_masked, 0)
    ds_bin_minus25_75 = xr.where((deviation_from_normal.soilMoist > -75) & (deviation_from_normal.soilMoist < -25), edd_future_masked, 0)
    ds_bin_minus75 = xr.where(deviation_from_normal.soilMoist <= -75, edd_future_masked, 0)
    
    # Sum the EDD values over the time dimension for each bin.
    ds_bin_plus75 = ds_bin_plus75.sum(dim='time')
    ds_bin_plus25_75 = ds_bin_plus25_75.sum(dim='time')
    ds_bin_minus25_plus25 = ds_bin_minus25_plus25.sum(dim='time')
    ds_bin_minus25_75 = ds_bin_minus25_75.sum(dim='time')
    ds_bin_minus75 = ds_bin_minus75.sum(dim='time')
    
    # Build the final combined dataset. 
    combined_dataset_bins = xr.Dataset({
        "SoilM_0_100cm": ds_soilpyWBM_year.soilMoist.mean(dim="time"),
        "gdd": gdd_future_masked.sum(dim='time'),
        "edd_plus75": ds_bin_plus75,
        "edd_plus25_75": ds_bin_plus25_75,
        "edd_minus25_plus25": ds_bin_minus25_plus25,
        "edd_minus25_75": ds_bin_minus25_75,
        "edd_minus75": ds_bin_minus75
    })
    combined_dataset_bins = combined_dataset_bins.expand_dims(year=[year])
    
    return combined_dataset_bins
    #except Exception as e:
    #    print(f"{year} year error, pass this year:", e)
    #    return xr.Dataset()  
        
def process_model(model_name_i, initialization_i, ssp_i, ds_soilpyWBM_regrid, ds_soil_normal_on_wbm_grid, pywbm_combination_i, time_frame_i):
    '''
    input parameters that specify pyWBM run, and processes for yearly binned dday, gdd, etc. 
    This links to a process year dask function allowing for more parallel-ness
    inputs
    - model_name_i (i.e. MIROC6)
    - initialization_i (i.e. r1i1p1f1)
    - ssp_i (i.e. 370)
    - ds_soilpyWBM_regrid which is some random pyWBM run which is regridded
    - ds_soil_normal_on_wbm_grid which is soil normal for corresponding LSM
    - pywbm_combination_i which is the ith glob.glob sorted combination for the model, intitilaztion, etc. combination 
    - time_frame_i which is the ~30 year chunk that loca2 is saved in, passed in, iterated through outside loop
    outputs
    - dataset which has gdd, and binned edd 
    '''
    # this is the pyWBM combinations, eventually will need to be looped through 
    
    ds_soilpyWBM_initial = xr.open_dataset(pywbm_combination_i)
    ds_soilpyWBM_initial['time'] = ds_soilpyWBM_initial.indexes['time'].to_datetimeindex()
        
    ds_combined = None  
    
    # tmax file
    file_path_i_tmax = f"{base_loca_paths_for_models}{model_name_i}/0p0625deg/{initialization_i}/ssp{ssp_i}/tasmax"
    file_name_i_tmax = f"tasmax.{model_name_i}.ssp{ssp_i}.{initialization_i}.{time_frame_i}.LOCA_16thdeg_v20220413.nc"
    # tmin file
    file_path_i_tmin = f"{base_loca_paths_for_models}{model_name_i}/0p0625deg/{initialization_i}/ssp{ssp_i}/tasmin"
    file_name_i_tmin = f"tasmin.{model_name_i}.ssp{ssp_i}.{initialization_i}.{time_frame_i}.LOCA_16thdeg_v20220413.nc"
    
    # combing them for usage in degree day calculation
    try:
        ds_tmax = xr.open_dataset(f"{file_path_i_tmax}/{file_name_i_tmax}").rename({"tasmax": "tmax"}) # , chunks='auto'
        ds_tmin = xr.open_dataset(f"{file_path_i_tmin}/{file_name_i_tmin}").rename({"tasmin": "tmin"}) # , chunks='auto'
        ds_combined = xr.merge([ds_tmax, ds_tmin]) 
        
    except Exception:
        print(f"Issue with file location, skipping {file_path_i_tmax}/{file_name_i_tmax} or {file_path_i_tmin}/{file_name_i_tmin}")
        return

        # this inputs some big daily chunked dataset, and outputs the gdd & edd binned using pyWBM futures
        
    if ds_combined is None:
        print(f"Skipping process_model for {model_name_i}, {initialization_i}, ssp {ssp_i}: ds_combined was not assigned.")
        return
    
    try:
        start_year = int(ds_combined.time.dt.year.values[0])
        if start_year == 2015:
            start_year = 2016
        end_year = int(ds_combined.time.dt.year.values[-1])
    
        delayed_years = []
        for year in range(start_year, end_year + 1):
            # Create a delayed task for each year
            delayed_task = dask.delayed(process_year)(
                ds_combined, year, ds_soilpyWBM_regrid, ds_soil_normal_on_wbm_grid,
                ds_soilpyWBM_initial, model_name_i, initialization_i, ssp_i
            )
            delayed_years.append(delayed_task)
    
        # return a delayed concatenation of the yearly datasets
        return dask.delayed(xr.concat)(delayed_years, dim="year"), pywbm_combination_i, time_frame_i
    except Exception as e:
        print("Error in processing model:", e)
        return

from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    account="open",
    cores=2,
    memory="100GiB",
    walltime="24:00:00",
    processes=1
)

cluster.scale(jobs=8)

from dask.distributed import Client

client = Client(cluster)

us_county = gpd.read_file(county_shp_path)
us_county = us_county.to_crs("EPSG:4326")
future_pyWBM_path = f"/storage/home/cta5244/work/pyWBM_yield_data/pyWBM_dday/"
os.makedirs(future_pyWBM_path, exist_ok=True)


# each loca2 file is 10GB (large)
# first lets get some base pyWBM run, & base historical normal to use for future regridding
# by getting ds_soil_normal_on_wbm_grid we can use it without issue for any pyWBM anomaly vs the historical time frame
ds_soilpyWBM_regrid = (xr.open_dataset(arbritrary_pyWBM_run))
ds_soil_normal = xr.open_dataset(soil_moisture_normal_file_path).SoilM_0_100cm
ds_soil_normal_on_wbm_grid = ds_soil_normal.interp(
    lat=ds_soilpyWBM_regrid.lat,
    lon=ds_soilpyWBM_regrid.lon,
    method="linear"  # or "nearest" if you prefer
).persist()


for model_name_i in model_names:
    for initialization_i in initializations:
        for ssp_i in ssps:
            for time_frame_i in time_frames:
                pywbm_combinations = sorted(glob.glob(f"{pyWBM_file_path_base}/{model_name_i}_{initialization_i}_ssp{ssp_i}_{nldas_lsm}*"))
                for pywbm_combination_i in pywbm_combinations[:1]:
                    # put the delayed_tasks in here to avoid race conditions / other multiple read issues
                    delayed_tasks = []
                    
                    delayed_tasks.append(
                        dask.delayed(process_model)(
                            model_name_i, initialization_i, ssp_i,
                            ds_soilpyWBM_regrid, ds_soil_normal_on_wbm_grid,
                            pywbm_combination_i, time_frame_i
                        )
                    )
                    
                    results = dask.compute(*delayed_tasks)
                    for (dask_delayed_task_i_model, pywbm_combination_corresponding, time_frame_i) in results:
                        combined_dataset_bins = dask.compute(dask_delayed_task_i_model.load())[0]
                        
                        weightmap = xa.pixel_overlaps(combined_dataset_bins, us_county)
                        aggregated = xa.aggregate(combined_dataset_bins, weightmap)
                        
                        ds_out = aggregated.to_dataset().to_dataframe()
                        ds_out = ds_out.reset_index().set_index(['fips','year'])
                        parameter_string = (pywbm_combination_corresponding.split("/"))
                        csv_output_file = f"{future_pyWBM_path}{parameter_string[-1][:-3]}_{time_frame_i}_ddaysm.csv"
                        ds_out.to_csv(csv_output_file, index=True)
