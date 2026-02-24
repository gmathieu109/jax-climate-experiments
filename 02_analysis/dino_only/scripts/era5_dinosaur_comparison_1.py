import sys, os, time
print("START", flush=True)
print("python:", sys.executable, flush=True)
print("cwd:", os.getcwd(), flush=True)
t0=time.time()

import xarray as xr
print("imported xarray", xr.__version__, "after", time.time()-t0, "s", flush=True)

import numpy as np

fc_path  = "/scratch/gmathieu/dinosaur_outputs/ds_out_small.nc"
era_path = "/scratch/gmathieu/era5_snapshot/era5_19900501T00.nc" 

print("opening ds_fc:", fc_path, flush=True)
ds_fc = xr.open_dataset(fc_path, engine="scipy", backend_kwargs={"mmap": False})
print("opened ds_fc", ds_fc.sizes, flush=True)

print("opening ds_era:", era_path, flush=True)
ds_era = xr.open_dataset(era_path, engine="scipy", backend_kwargs={"mmap": False})
# squeeze seulement si time existe
if "time" in ds_era.dims:
    ds_era = ds_era.squeeze("time")

# target grid from forecast
lon_t = ds_fc["longitude"]
lat_t = ds_fc["latitude"]

# ERA5 longitudes 0..359.75, forecast aussi normalement -> OK.
# Regrid bilinéaire (xarray interp)

fc_sp0 = ds_fc["surface_pressure"].isel(time=0) 

# 1) rendre ERA5 monotone croissant en latitude pour interp
era_sp = ds_era["surface_pressure"] 
if np.any(np.diff(ds_era["latitude"].values) < 0):
    era_sp = era_sp.sortby("latitude")

# 2) harmoniser les longitudes ERA5 sur celles du forecast (wrap 0..360 <-> -180..180)
lon_fc = ds_fc["longitude"]

era_lon = era_sp["longitude"]
# si fc est en [-180,180) et era en [0,360)
if (lon_fc.min() < 0) and (era_lon.min() >= 0):
    era_sp = era_sp.assign_coords(longitude=((era_lon + 180) % 360) - 180).sortby("longitude")
# si fc est en [0,360) et era en [-180,180)
elif (lon_fc.min() >= 0) and (era_lon.min() < 0):
    era_sp = era_sp.assign_coords(longitude=(era_lon % 360)).sortby("longitude")

# 3) regrid bilinéaire sur la grille du forecast
era_sp_rg = era_sp.interp(longitude=lon_fc, latitude=ds_fc["latitude"])

# 4) mettre même ordre de dims
era_sp_rg = era_sp_rg.transpose("longitude", "latitude")

# 5) Convertir en hPa
era_sp_rg = era_sp_rg / 100.0
fc_sp0 = fc_sp0 / 100.0

# 6) stats
diff = fc_sp0 - era_sp_rg
print("fc lon range:", float(lon_fc.min()), float(lon_fc.max()), flush=True)
print("era lon range after:", float(era_sp_rg.longitude.min()), float(era_sp_rg.longitude.max()), flush=True)
print("SP mean:", float(diff.mean()), flush=True)
print("SP rmse:", float(np.sqrt((diff**2).mean())), flush=True)
print("diff min/max:", float(diff.min()), float(diff.max()), flush=True)


# Attention: dims order
era_sp_rg = era_sp_rg.transpose("longitude", "latitude")

fc_sp0 = ds_fc["surface_pressure"].isel(time=0)

diff = fc_sp0 - era_sp_rg
print("SP mean:", float(diff.mean()))
print("SP rmse:", float(np.sqrt((diff**2).mean())))
print("diff min/max:", float(diff.min()), float(diff.max()))

print("opened ds_era", ds_era.sizes, flush=True)

# era_sp_rg: ERA5 regriddé (lon,lat), en Pa
# fc_sp: forecast (time,lon,lat), en Pa
fc_sp = ds_fc["surface_pressure"] / 100.0  # convertir en hPa

for ti in [0, 1, 5, 20, 50, 100, 192]:
    ti = min(ti, fc_sp.sizes["time"]-1)
    d = fc_sp.isel(time=ti) - era_sp_rg
    rmse = float(np.sqrt((d**2).mean()))
    mean = float(d.mean())
    print(ti, "rmse(Pa)=", rmse, "mean(Pa)=", mean)
