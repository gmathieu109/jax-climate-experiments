#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CACHE_DIR = Path("/cuyo/gmathieu/eval_cache_2021")

ERA5 = CACHE_DIR / "era5_regional_T850_daily.nc"
BASE = CACHE_DIR / "jcm_regional_T850_daily_jcm_full_2021_1y_time2021.nc"
OPT  = CACHE_DIR / "jcm_regional_T850_daily_jcm_sst_masked_jan15d_sm0p02_2021_1y_time2021.nc"

OUTDIR = Path("/cuyo/gmathieu/optimized_10y/sst_masked_jan15d_sm0p02_2021_1y/comparison_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTDIR / "regional_T850_timeseries_era5_baseline_optimized_uniform_y.png"

ds_era5 = xr.open_dataset(ERA5)
ds_base = xr.open_dataset(BASE)
ds_opt  = xr.open_dataset(OPT)

regions = list(ds_base.data_vars)

# Same 1-year period
time = ds_base.time
ds_era5 = ds_era5.sel(time=time)
ds_opt = ds_opt.sel(time=time)

# Uniform y-axis across all panels
all_vals = []
for r in regions:
    all_vals.extend([
        ds_era5[r].values,
        ds_base[r].values,
        ds_opt[r].values,
    ])

ymin = min(np.nanmin(v) for v in all_vals) - 1.0
ymax = max(np.nanmax(v) for v in all_vals) + 1.0

n = len(regions)
fig, axes = plt.subplots(n, 1, figsize=(12, 3.0 * n), sharex=True, sharey=False)

YLIMS = {
    "Global": (277, 284),
    "Tropics": (286.5, 291),
    "NH Midlats": (266, 291),
    "SH Midlats": (271, 281),
    "Arctic": (250, 280),
    "Antarctic": (248, 266),
    "N America": (264, 292),
}

if n == 1:
    axes = [axes]

for ax, r in zip(axes, regions):
    ax.plot(time.values, ds_era5[r].values, color="black", lw=1.5, label="ERA5")
    ax.plot(time.values, ds_base[r].values, color="firebrick", lw=1.4, label="Baseline")
    ax.plot(time.values, ds_opt[r].values, color="royalblue", lw=1.4, label="JCM + ΔSST")

    ax.set_title(r, fontsize=11, fontweight="bold")
    ax.set_ylabel("T850 (K)")
    if r in YLIMS:
        ax.set_ylim(*YLIMS[r])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[-1].set_xlabel("Time")

fig.suptitle(
    "Regional mean T850 over one year: ERA5 vs baseline vs optimized SST",
    fontsize=14,
    fontweight="bold",
    y=1.005,
)

fig.tight_layout()
fig.savefig(OUTFILE, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUTFILE)
