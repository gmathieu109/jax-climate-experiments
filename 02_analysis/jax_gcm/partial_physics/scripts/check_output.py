import xarray as xr

p = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/forcing_humid_clouds_rad_10d/run_nc/jcm_forcing_humid_clouds_rad_10d.nc"
ds = xr.open_dataset(p)

print(ds)
print("\nVars:", list(ds.data_vars)[:40])

if "temperature" in ds:
    print("T min/max:", float(ds.temperature.min()), float(ds.temperature.max()))
    print("NaN T:", int(ds.temperature.isnull().sum()))
