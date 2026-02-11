import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

fc_path = "/scratch/gmathieu/dinosaur_outputs/ds_out_small.nc"
ds_fc = xr.open_dataset(fc_path, engine="scipy", backend_kwargs={"mmap": False})



t_model = ds_fc["time"].values  # float
# devine si c’est en secondes: très souvent oui
# si c’est en heures, tu ajusteras juste le facteur

start = np.datetime64("1990-05-01T00:00:00")
times_dt64 = start + (t_model * 3600).astype("timedelta64[s]")
ds_fc_dt = ds_fc.assign_coords(time=times_dt64)
fc_sp = ds_fc_dt["surface_pressure"]     # maintenant time est datetime64



# Target grid + times
lon_t = ds_fc["longitude"]
lat_t = ds_fc["latitude"]
times = ds_fc["time"].values

# ERA5 Zarr (internet)
ds_era = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options=dict(token="anon"),
)


era_sp_all = ds_era["surface_pressure"].sel(time=slice(times_dt64.min(), times_dt64.max()))
era_sp = era_sp_all.reindex(time=times_dt64, method="nearest", tolerance=np.timedelta64(30, "m"))

# make latitude increasing for interp
if np.any(np.diff(era_sp["latitude"].values) < 0):
    era_sp = era_sp.sortby("latitude")

# regrid to forecast grid (Pa)
era_sp_rg = era_sp.interp(longitude=lon_t, latitude=lat_t).transpose("time","longitude","latitude")

fc_sp = ds_fc["surface_pressure"]  # (time, lon, lat)

# Force forecast time to datetime64 (overwrite the coordinate)
fc_sp = ds_fc["surface_pressure"].copy()
fc_sp = fc_sp.assign_coords(time=times_dt64)

# Ensure ERA5 has the same time coordinate objects
era_sp_rg = era_sp_rg.assign_coords(time=times_dt64)



# RMSE(t) and bias(t) in hPa
diff = (fc_sp - era_sp_rg) / 100.0
rmse = np.sqrt((diff**2).mean(dim=("longitude","latitude"))).values
bias = diff.mean(dim=("longitude","latitude")).values


# Save CSV
out_csv = "/scratch/gmathieu/dinosaur_outputs/sp_rmse_vs_time_hpa.csv"
df = pd.DataFrame({"time": ds_fc["time"].values, "rmse_hpa": rmse, "bias_hpa": bias})
df.to_csv(out_csv, index=False)
print("WROTE:", out_csv)


# Plot
out_png = "/scratch/gmathieu/dinosaur_outputs/sp_rmse_vs_time_hpa.png"
plt.figure()
plt.plot(ds_fc["time"].values, rmse)
plt.xlabel("time (h)")
plt.ylabel("Surface pressure (hPa)")
plt.title("Surface pressure RMSE vs ERA5 (time-matched)")
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print("WROTE:", out_png)

plt.figure()
plt.plot(ds_fc["time"].values, fc_sp.mean(dim=("longitude","latitude")), label="Forecast")
plt.plot(ds_fc["time"].values, era_sp_rg.mean(dim=("longitude","latitude")), label="ERA5")
plt.legend()
plt.xlabel("time (h)")
plt.ylabel("Surface pressure (hPa)")
plt.title("Surface pressure fc vs ERA5 (time-matched)")
out_png_sp = "/scratch/gmathieu/dinosaur_outputs/surface_pressure.png"
plt.savefig(out_png_sp, dpi=150)
print("WROTE:", out_png_sp)

plt.figure()
plt.plot(ds_fc["time"].values, abs(diff.mean(dim=("longitude","latitude"))), label="Forecast - ERA5")
plt.legend()
plt.xlabel("time (h)")
plt.ylabel("Surface pressure difference (hPa)")
plt.title("Surface pressure bias vs ERA5 (time-matched)")
out_png_bias = "/scratch/gmathieu/dinosaur_outputs/surface_pressure_bias.png"
plt.savefig(out_png_bias, dpi=150)
print("WROTE:", out_png_bias)