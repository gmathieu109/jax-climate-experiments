import xarray as xr
import numpy as np

p = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/forcing_humid_clouds_rad_sfc_10d/run_nc/jcm_forcing_humid_clouds_rad_sfc_30d.nc"
ds = xr.open_dataset(p)

t = ds["temperature"]

print("shape:", t.shape)
print()

for i in range(t.sizes["time"]):
    arr = t.isel(time=i)
    n_nan = int(arr.isnull().sum())
    total = arr.size
    tmin = float(arr.min(skipna=True)) if n_nan < total else np.nan
    tmax = float(arr.max(skipna=True)) if n_nan < total else np.nan
    print(f"time[{i}] = {ds.time.values[i]} | NaN = {n_nan}/{total} | Tmin = {tmin} | Tmax = {tmax}")
