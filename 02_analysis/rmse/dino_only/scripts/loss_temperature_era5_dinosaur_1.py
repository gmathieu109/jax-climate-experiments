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
fc_temp = ds_fc_dt["temperature"]     # maintenant time est datetime64



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


# ERA5 low-level T (K) : prendre un niveau proche surface
era_temp_all = ds_era["temperature"].isel(level=-1).sel(
    time=slice(times_dt64.min(), times_dt64.max())
)

# Aligner les temps (nearest)
era_temp = era_temp_all.reindex(
    time=times_dt64,
    method="nearest",
    tolerance=np.timedelta64(60, "m"),
)

# Latitude croissante pour interp
if np.any(np.diff(era_temp["latitude"].values) < 0):
    era_temp = era_temp.sortby("latitude")

# Regrid horizontal -> grille forecast
era_temp_rg = era_temp.interp(longitude=lon_t, latitude=lat_t).transpose("time","longitude","latitude")



# Force forecast time to datetime64 (overwrite the coordinate)
fc_temp = ds_fc["temperature"].isel(sigma = -1)  # surface level
fc_temp = fc_temp.assign_coords(time=times_dt64)
# Ensure ERA5 has the same time coordinate objects
era_temp_rg = era_temp_rg.assign_coords(time=times_dt64)



# RMSE(t) and bias(t) in K
diff = (fc_temp.values - era_temp_rg.values)
rmse = np.sqrt(np.mean(diff**2, axis=(1,2)))
bias = np.mean(diff, axis=(1,2))


# Save CSV
out_csv = "/scratch/gmathieu/dinosaur_outputs/temp_rmse_vs_time_k.csv"
df = pd.DataFrame({"time": ds_fc["time"].values, "rmse_k": rmse, "bias_k": bias})
df.to_csv(out_csv, index=False)
print("WROTE:", out_csv)

# Plot
out_png = "/scratch/gmathieu/dinosaur_outputs/temp_rmse_vs_time_k.png"
plt.figure()
plt.plot(ds_fc["time"].values, rmse)
plt.xlabel("time (h)")
plt.ylabel("Temperature (K)")
plt.title("Temperature RMSE vs ERA5 (time-matched)")
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print("WROTE:", out_png)

plt.figure()
plt.plot(ds_fc["time"].values, fc_temp.mean(dim=("longitude","latitude")), label="Forecast")
plt.plot(ds_fc["time"].values, era_temp_rg.mean(dim=("longitude","latitude")), label="ERA5")
plt.legend()
plt.xlabel("time (h)")
plt.ylabel("Temperature (K)")
plt.title("Temperature fc vs ERA5 (time-matched)")
out_png_sp = "/scratch/gmathieu/dinosaur_outputs/temperature.png"
plt.savefig(out_png_sp, dpi=150)
print("WROTE:", out_png_sp)

plt.figure()
plt.plot(ds_fc["time"].values, abs(bias), label="Forecast - ERA5")
plt.legend()
plt.xlabel("time (h)")
plt.ylabel("Temperature difference (K)")
plt.title("Temperature bias vs ERA5 (time-matched)")
out_png_bias = "/scratch/gmathieu/dinosaur_outputs/temperature_bias.png"
plt.savefig(out_png_bias, dpi=150)
print("WROTE:", out_png_bias)