import requests
import re
import os
import numpy as np
import datetime
import xarray as xr
import dask
import glob

##########################      functions              ################################################################


def downloadData(url, output_dir, session):
    '''
    Given a URL and an authenticated session, this function downloads data from
    NASA Earthdata and saves it in the specified output directory.
    
    inputs:
      url: a string representing the file URL.
      output_dir: the directory where the file should be saved.
      session: an authenticated requests.Session() object.
    
    returns: nothing
    '''
    response = session.get(url, stream=True)
    
    if response.status_code == 200:
        cd = response.headers.get("content-disposition")
        if cd:
            fname_match = re.findall('filename="?([^";]+)"?', cd)
            filename = fname_match[0] if fname_match else url.split("/")[-1]
        else:
            filename = url.split("/")[-1]
        
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        raise Exception(f"Error downloading {url}: HTTP {response.status_code}")
    return filename
    
def aggregate_day_files(date_str, output_dir):
    '''
    inputs 
    yyyymmdd as datestr
    location for files to be saved as outputdir
    '''
    pattern = os.path.join(output_dir, f"{output_dir}/NLDAS_NOAH0125_H.A{date_str}.*.grb.SUB.nc4")
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        print(f"No files found for date {date_str}")
        return None
        
    try:
        ds = xr.open_mfdataset(file_list, combine='by_coords')
        
    except Exception as e:
        print(f"Error opening files for {date_str}: {e}")
        return None
        
    ds_daily = ds.resample(time='1D').mean().sel(depth=100.0)

    daily_file = os.path.join(output_dir, f"{output_dir}/NLDAS_NOAH0125_H.A{date_str}_daily_100cm.nc")
    ds_daily.to_netcdf(daily_file)
    
    #for file_to_remove in glob.glob(pattern):
    #    os.remove(file_to_remove)
    
    
def urls_list():
    # Retrieve credentials from environment variables.
    username = os.environ.get("earthnasa_user")
    password = os.environ.get("earthnasa_pass")
    
    if not username or not password:
        raise Exception("Missing credentials. Please set earthnasa_user and earthnasa_pass in your environment.")

    url_file = "/storage/home/cta5244/work/pyWBM_yield_data/hydro_models/subset_NLDAS_NOAH0125_H_002_20250226_024826_.txt"
    output_dir = "/storage/home/cta5244/work/pyWBM_yield_data/hydro_models/NOAH/daily_soil100cm"
    
    os.makedirs(output_dir, exist_ok=True)
    
    session = requests.Session()
    session.auth = (username, password)
    
    with open(url_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
        
    for url in urls[22:25]:
        return url
        try:
            file_path = downloadData(url, output_dir, session)
            
            match = re.search(r"NLDAS_NOAH0125_H\.A(\d{8})\.(\d{4})\.002\.grb\.SUB", file_path)
            if match:
                date_str = match.group(1)  
                hour_str = match.group(2)
                
                if hour_str.startswith("23"):
                    aggregate_day_files(date_str, output_dir)
                
        except Exception as e:
            print(e)
            
urls_list()



#############              inputs            ############################################

output_dir = "/storage/home/cta5244/work/pyWBM_yield_data/hydro_models/NOAH/daily_soil100cm"
base_url = "https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_NOAH0125_H.002"
specific_model_path = "NLDAS_NOAH0125_H.A"
file_type = "grb"

start_year = 1979
end_year = 2026


#################       runs script with dask        ##################################################### 

from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    # account="pches",
    account="open",
    cores=1,
    memory="2GiB",
    walltime="24:00:00",
)

cluster.scale(jobs=50) 

from dask.distributed import Client

client = Client(cluster)

'''results = []
for year in np.arange(start_year, end_year, 1):
    out = dask.delayed(singleYearUrl)(year=year)
    results.append(out)
    
results = dask.compute(*results)'''
