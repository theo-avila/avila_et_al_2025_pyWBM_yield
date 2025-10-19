import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import dask
import os

ssps = ['ssp370', 'ssp245'] # 

sigma_threshold = True
persistence_length = 5

for ssp in ssps[:1]:
    base_path = f"/storage/group/pches/default/users/cta5244/CMIP6_tos/{ssp}_omon_tos"
    hist_path = f"/storage/group/pches/default/users/cta5244/CMIP6_tos/hist_omon_tos"
    output_path = f"/storage/group/pches/default/users/cta5244/enso4_loca2_underlying_models/{ssp}"
    model_paths = sorted(glob.glob(f"{base_path}/*"))
    yearly_paths = sorted(glob.glob(f"{model_paths[0]}/*"))
    for model_path in model_paths:
        
        try:
            yearly_paths = sorted(glob.glob(f"{model_path}/*"))
            hist_paths = sorted(glob.glob(f"{hist_path}/{model_path.split('/')[-1]}/*"))
            # output string with gr not gn 
            new_list = ['gr' if item == 'gn' else item for item in yearly_paths[0].split("/")[-1].split("_")]
            output_unique_str = "_".join(new_list)
            # opening & regridding datasets
            out_path = f"{output_path}/{output_unique_str}_enso_labels.nc"
    
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"SKIP (exists): {out_path}")
                continue
            
            else:
                ds_cmip6 = xr.open_mfdataset(
                    yearly_paths + hist_paths,
                    combine="by_coords",
                    chunks={"time": 120},          
                    parallel=True,
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    preprocess=lambda d: d[["tos"]].astype({"tos": "float32"}),
                )
                two_dim_bool = False
                if ('latitude' in ds_cmip6.coords) and (len(ds_cmip6.latitude.shape) == 1):
                    ds_cmip6 = ds_cmip6.rename({'latitude':'lat'})
                    ds_cmip6 = ds_cmip6.rename({'longitude':'lon'})
                    ds_cmip6 = ds_cmip6.assign_coords(longitude=(ds_cmip6['lon'] % 360))
                    lat = ds_cmip6['lat']
                    lon = ds_cmip6['lon']
                    lon2d, lat2d = xr.broadcast(lon, lat)
                    mask_n4 = ((lat2d >= -5) & (lat2d <= 5) & (lon2d >= 160) & (lon2d <= 210))
                elif ('lat' in ds_cmip6.coords) and (len(ds_cmip6.lat.shape) == 1):
                    ds_cmip6 = ds_cmip6.assign_coords(longitude=(ds_cmip6['lon'] % 360))
                    lat = ds_cmip6['lat']
                    lon = ds_cmip6['lon']
                    lon2d, lat2d = xr.broadcast(lon, lat)
                    mask_n4 = ((lat2d >= -5) & (lat2d <= 5) & (lon2d >= 160) & (lon2d <= 210))
                elif ('latitude' in ds_cmip6.coords) and (len(ds_cmip6.latitude.shape) == 2):
                    ds_cmip6 = ds_cmip6.assign_coords(longitude=(ds_cmip6['longitude'] % 360))
                    lon2d, lat2d = ds_cmip6.longitude, ds_cmip6.latitude
                    mask_n4 = ((lat2d >= -5) & (lat2d <= 5) & (lon2d >= 160) & (lon2d <= 210))
                    two_dim_bool = True
                elif ('lat' in ds_cmip6.coords) and (len(ds_cmip6.lat.shape) == 2):
                    ds_cmip6 = ds_cmip6.assign_coords(lon=(ds_cmip6['lon'] % 360))
                    lon2d, lat2d = ds_cmip6.lon, ds_cmip6.lat
                    mask_n4 = ((lat2d >= -5) & (lat2d <= 5) & (lon2d >= 190) & (lon2d <= 240))
                    two_dim_bool = True
                else:
                    print(ds_cmip6.coords)
                    
                tos = ds_cmip6["tos"].where(mask_n4)
                weights = xr.ufuncs.cos(np.deg2rad(lat2d)).where(mask_n4)
                weights = weights.fillna(0)
                spatial_dims = tuple(d for d in tos.dims if d != "time")
                
                num = (tos * weights).sum(spatial_dims)
                den = weights.sum(spatial_dims)
                ts = (num / den).chunk({"time": 120})   
                
                clim = ts.sel(time=slice('1991','2020')).groupby('time.month').mean('time')
                oni = ts.groupby('time.month') - clim
                
                # postprocessing of last time series 
                #nino4_3m = nino4.rolling(time=3, center=True).mean()
                #nino4_hp = nino4_3m.rolling(time=121, center=True).mean()
                #nino4_var = nino4_3m - nino4_hp   
                y = oni 
                low_pass = y.rolling(time=121, center=True, min_periods=61).mean()  
                y_highpass = (y - low_pass).rename("oni_hp")
                oni_3m_det = y_highpass.rolling(time=3, center=True, min_periods=3).mean()
                sigma = float(oni_3m_det.sel(time=slice("1991","2020")).std("time"))  # 1σ in K
                sigma_threshold = True
                if sigma_threshold:
                    thr = sigma
                else:
                    thr = .5
                mode_indx = oni_3m_det.compute()               
                
                pos_persist = (mode_indx >=  thr).rolling(time=persistence_length).sum() >= persistence_length
                neg_persist = (mode_indx <= -thr).rolling(time=persistence_length).sum() >= persistence_length
                
                labels = xr.where(pos_persist, 1, xr.where(neg_persist, -1, 0)).astype("int8")
                labels = labels.rename("enso_phase")
                labels.attrs.update({
                    "long_name": "ENSO phase label",
                    "description": "Month-by-month phase labels: +1 El Niño, -1 La Niña, 0 Neutral. "
                                   f"Computed from 3-month smoothed, detrended ONI index with threshold={thr} K "
                                   f"and {persistence_length} consecutive 5 month periods.",
                    "flag_values": np.array([-1, 0, 1], dtype="int8"),
                    "flag_meanings": "la_nina neutral el_nino",
                    "threshold_units": "K",
                    "threshold_value": float(thr),
                    "persistence_months": int(persistence_length),
                    "index_source": "ONI 3-month running mean (detrended)",
                    "unique_str": f"{output_unique_str}",
                    "sigma_threshold": f"{sigma_threshold}"
                })
                ds_out = xr.Dataset(
                    {
                        "enso_phase": labels,          
                        "oni_index": mode_indx,   
                        "threshold": thr,
                    }
                )
                
                ds_out.to_netcdf(out_path)
                print("saved:", out_path)
            
        except Exception as e:
            print(f"issue with {model_path}")
            continue