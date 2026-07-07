#!/usr/bin/env python3

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ERA5_FIELD = "/cuyo/gmathieu/eval_cache_2021/era5_T850_daily_on_jcm_grid.nc"

BASELINE_NC = "/cuyo/gmathieu/baseline_2021/full_1y_no_delta/run_nc/jcm_full_2021_1y_time2021.nc"
OPT_NC = "/cuyo/gmathieu/optimized_10y/sst_masked_jan15d_sm0p02_2021_1y/run_nc/jcm_sst_masked_jan15d_sm0p02_2021_1y_time2021.nc"

OUTDIR = Path("/cuyo/gmathieu/optimized_10y/sst_masked_jan15d_sm0p02_2021_1y/comparison_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

JCM_SIGMA_850 = 0.835


def get_t850_jcm(path):
    ds = xr.open_dataset(path)
    level_idx = int(np.argmin(np.abs(ds.level.values - JCM_SIGMA_850)))
    da = ds["temperature"].isel(level=level_idx)

    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")

    return da


def cos_weighted_global_mean(da):
    weights = np.cos(np.deg2rad(da.lat))
    return da.weighted(weights).mean(dim=("lat", "lon"))


era5 = xr.open_dataarray(ERA5_FIELD).sel(time=slice("2021-01-01", "2021-12-31"))

baseline = get_t850_jcm(BASELINE_NC)
opt = get_t850_jcm(OPT_NC)

era5 = era5.sel(time=baseline.time)

# ---- Time series ----
era5_ts = cos_weighted_global_mean(era5)
base_ts = cos_weighted_global_mean(baseline)
opt_ts = cos_weighted_global_mean(opt)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(era5_ts.time, era5_ts, color="black", lw=2, label="ERA5")
ax.plot(base_ts.time, base_ts, color="firebrick", lw=1.8, label="JCM baseline")
ax.plot(opt_ts.time, opt_ts, color="royalblue", lw=1.8, label="JCM + ΔSST")

ax.set_title("Global mean T850 over one year")
ax.set_ylabel("T850 (K)")
ax.set_xlabel("Time")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(OUTDIR / "timeseries_global_mean_T850_baseline_vs_delta.png", dpi=200)
plt.close(fig)

# ---- Mean bias maps ----
era5_mean = era5.mean(dim="time")
base_bias = baseline.mean(dim="time") - era5_mean
opt_bias = opt.mean(dim="time") - era5_mean
bias_change = opt_bias - base_bias

def global_mean_timeseries_rmse(da_model, da_ref):
    weights = np.cos(np.deg2rad(da_ref.lat))
    model_ts = da_model.weighted(weights).mean(dim=("lat", "lon"))
    ref_ts = da_ref.weighted(weights).mean(dim=("lat", "lon"))
    return float(np.sqrt(((model_ts - ref_ts) ** 2).mean(dim="time")))
def area_mean_bias(da_bias):
    weights = np.cos(np.deg2rad(da_bias.lat))
    return float(da_bias.weighted(weights).mean(dim=("lat", "lon")))

base_rmse = global_mean_timeseries_rmse(baseline, era5)
opt_rmse = global_mean_timeseries_rmse(opt, era5)
improvement = (base_rmse - opt_rmse) / base_rmse * 100

base_mean_bias = area_mean_bias(base_bias)
opt_mean_bias = area_mean_bias(opt_bias)

import matplotlib.colors as colors
import cartopy.crs as ccrs

levels = np.arange(-6, 7, 1)
cmap = plt.get_cmap("RdBu_r")
norm = colors.BoundaryNorm(levels, ncolors=cmap.N)

plots = [
    (
        base_bias,
        f"Baseline bias: JCM − ERA5\nMean bias = {base_mean_bias:.2f} K | RMSE = {base_rmse:.2f} K",
    ),
    (
        opt_bias,
        f"Optimized bias: JCM + ΔSST − ERA5\nMean bias = {opt_mean_bias:.2f} K | RMSE = {opt_rmse:.2f} K",
    ),
    (
        bias_change,
        f"Change in T850 bias: optimized − baseline\nGlobal RMSE improvement = {improvement:.1f}%",
    ),
]

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 12),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

for ax, (da, title) in zip(axes, plots):
    field = da.transpose("lat", "lon")

    # Convertir les longitudes 0–360 vers -180–180
    lon_new = ((field.lon + 180) % 360) - 180
    field = field.assign_coords(lon=lon_new).sortby("lon")

    # Remplacer la colonne NaN par interpolation longitudinale
    field = field.interpolate_na(dim="lon", method="linear", fill_value="extrapolate")

    im = ax.imshow(
        field.values,
        origin="lower",
        extent=[-180, 180, -90, 90],
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
    )

    ax.coastlines(linewidth=0.4, color="black")
    import cartopy.feature as cfeature

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.25,
        edgecolor="0.3",
    )
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.patch.set_edgecolor("none")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Latitude")

axes[-1].set_xlabel("Longitude")

fig.subplots_adjust(right=0.86, hspace=0.28)

print(field.lon.values[:5])
print(field.lon.values[-5:])

cax = fig.add_axes([0.75, 0.18, 0.025, 0.64])
cbar = fig.colorbar(im, cax=cax, ticks=levels, boundaries=levels)
cbar.set_label("Temperature bias (K)")

fig.savefig(OUTDIR / "mean_bias_maps_baseline_vs_delta_2021.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(OUTDIR / "timeseries_global_mean_T850_baseline_vs_delta_2021.png")
print(OUTDIR / "mean_bias_maps_baseline_vs_delta_2021.png")
