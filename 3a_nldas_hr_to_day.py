import os
import glob
import datetime
import logging
import dask
import xarray as xr
import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def setup_logger():
    """Set up a logger to print info and error messages."""
    logger = logging.getLogger("DailyAverager")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
    return logger

logger = setup_logger()

def select_soilm(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess the dataset to select only the 'SoilM_0_100cm' variable.
    
    Parameters:
        ds (xr.Dataset): The input dataset.
        
    Returns:
        xr.Dataset: Dataset containing only the 'SoilM_0_100cm' variable.
    """
    return ds[['SoilM_0_100cm']]

def compute_daily_average(date: datetime.datetime, file_path_base: str, year_dir: str):
    """
    Compute the daily average from hourly netCDF files for a given date.
    
    This function:
      - Creates the output directory if it does not exist.
      - Uses a glob pattern to collect all hourly files for the day.
      - Opens the files with xarray (using our preprocess function to limit variables).
      - Computes the daily average using xarray's mean function over the 'time' dimension.
      - Writes the daily averaged data to a netCDF file.
      
    Parameters:
        date (datetime.datetime): The target date.
        file_path_base (str): Base path for hourly netCDF files.
        year_dir (str): Output directory for the year.
        
    Returns:
        str or None: The output file path if successful; None if no files were found or an error occurred.
    """
    os.makedirs(year_dir, exist_ok=True)
    date_str = date.strftime("%Y%m%d")
    file_pattern = f"{file_path_base}{date_str}.*.020.nc"
    matching_files = sorted(glob.glob(file_pattern))
    
    if not matching_files:
        logger.warning(f"No files found for pattern: {file_pattern}. Skipping date: {date_str}")
        return None

    # Filter out problematic files:
    valid_files = []
    for f in matching_files:
        try:
            with xr.open_dataset(f, engine='netcdf4') as ds_test:
                valid_files.append(f)
        except Exception as e:
            logger.error(f"Skipping file {f} due to error: {e}")
    if not valid_files:
        logger.warning(f"No valid files found for date {date_str}")
        return None

    try:
        # Use valid_files in place of matching_files:
        ds = xr.open_mfdataset(
            valid_files,
            combine='nested',
            concat_dim='time',
            data_vars='minimal',
            coords='minimal',
            compat='override',
            preprocess=select_soilm
        )

        # Compute the daily average over the 'time' dimension using xarray's mean function.
        daily_soilm = ds['SoilM_0_100cm'].mean(dim='time', skipna=True)

        # Add a new 'time' dimension with date_str as its coordinate.
        daily_soilm = daily_soilm.expand_dims({'time': [date_str]})

        output_filename = f"NLDAS_0_100_soilm_0125_H.A{date_str}.020.nc"
        output_path = os.path.join(year_dir, output_filename)
        daily_soilm.to_netcdf(output_path)

        logger.info(f"Processed date {date_str}, saved output to: {output_path}")

        ds.close()
        return output_path

    except Exception as e:
        logger.error(f"Error processing date {date_str}: {e}")
        return None



def process_year(year: int, file_path_base: str, downloads_dir: str):
    """
    Process an entire year of hourly netCDF files to compute daily averages.
    
    Iterates through each day of the specified year and calls compute_daily_average.
    
    Parameters:
        year (int): The year to process.
        file_path_base (str): Base path for hourly netCDF files.
        downloads_dir (str): Base output directory for daily files.
    """
    year_dir = os.path.join(downloads_dir, str(year))
    start_date = datetime.datetime(year, 1, 1)
    end_date = datetime.datetime(year, 12, 31)
    current_date = start_date
    
    while current_date <= end_date:
        compute_daily_average(current_date, file_path_base, year_dir)
        current_date += datetime.timedelta(days=1)

def main():
    # Set constants and file paths (these remain unchanged per your workflow)
    NLDAS_lsm = 'VIC'
    file_path_base = f"/storage/group/pches/default/public/NLDAS/{NLDAS_lsm}/hourly/NLDAS_{NLDAS_lsm}0125_H.A"
    downloads_dir = f"/storage/home/cta5244/work/pyWBM_yield_data/{NLDAS_lsm}_daily/"
    start_year, end_year = 1979, 2025

    # Setup Dask SLURM cluster for parallel processing
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client

    cluster = SLURMCluster(
        account="open",
        cores=1,
        memory="10GiB",
        walltime="24:00:00",
    )
    cluster.scale(jobs=2)
    client = Client(cluster)
    logger.info("Dask cluster initialized. Beginning yearly processing...")

    # Schedule processing for each year using dask.delayed.
    tasks = []
    for year in range(start_year, end_year):
        task = dask.delayed(process_year)(year, file_path_base, downloads_dir)
        tasks.append(task)
    
    # Trigger parallel computation.
    dask.compute(*tasks)
    logger.info("All processing completed.")

if __name__ == "__main__":
    main()



