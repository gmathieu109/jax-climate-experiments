#!/usr/bin/env python3
"""
Compare JCM simulation temperature to ERA5 reanalysis.

ERA5 is regridded onto the JCM lat/lon grid before comparison so that
regional means, biases, and spatial maps are computed on identical
grid points.

The first 365 days of JCM output are discarded as spin-up.  If the
remaining data spans more than one year, a day-of-year climatology
(multi-year daily mean) is computed before comparison.

Cached intermediate files (in ./eval_cache/):
  - era5_T850_daily_on_jcm_grid.nc : ERA5 daily T850 regridded to JCM
  - era5_regional_T850_daily.nc    : ERA5 regional means (from regridded)
  - jcm_regional_T850_daily.nc     : JCM regional means

Usage:
    python eval.py --jcm jcm_full_run.nc
    python eval.py --jcm test_30day.nc --spinup_days 0
    python eval.py --recompute # Use this if you add/change region
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

CACHE_DIR = Path("eval_cache") # pth to save era5 data

# !! Note we are using low res ERA5 data !!
ERA5_ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)
ERA5_YEARS = slice("2000", "2009")
ERA5_LEVEL_HPA = 850

JCM_SIGMA_850 = 0.835

REGIONS = {
    "Global":         {"lat": (-90, 90)},
    "Tropics":        {"lat": (-30, 30)},
    "NH Midlats":     {"lat": (30, 60)},
    "SH Midlats":     {"lat": (-60, -30)},
    "Arctic":         {"lat": (60, 90)},
    "Antarctic":      {"lat": (-90, -60)},
    "N America":      {"lat": (25, 55), "lon": (230, 300)},
}


def cos_weighted_mean(da, lat_name="lat"):
    """Area-weighted (cos-lat) spatial mean."""
    weights = np.cos(np.deg2rad(da[lat_name]))
    weights = weights / weights.mean()
    return da.weighted(weights).mean(dim=[lat_name, "lon"])


def regional_means(da, regions, lat_name="lat"):
    """Compute cos-lat-weighted mean for each region → Dataset.
    
    Regions can specify lat bounds only, or lat + lon bounds.
    Longitude is expected in [0, 360) degrees.
    """
    ds = xr.Dataset()
    for name, bounds in regions.items():
        lat_s, lat_n = bounds["lat"]
        subset = da.sel(**{lat_name: slice(lat_s, lat_n)})

        if "lon" in bounds:
            lon_s, lon_n = bounds["lon"]
            if lon_s < lon_n:
                subset = subset.sel(lon=slice(lon_s, lon_n))
            else:
                # wrap-around (e.g. 350 → 10)
                subset = subset.where(
                    (subset.lon >= lon_s) | (subset.lon <= lon_n), drop=True
                )

        ds[name] = cos_weighted_mean(subset, lat_name)
    return ds

def get_jcm_grid(jcm_file):
    """Extract lat/lon coordinate arrays from a JCM output file."""
    ds = xr.open_dataset(jcm_file)
    lat = ds.lat.values
    lon = ds.lon.values
    ds.close()
    # Ensure lat ascending
    lat = np.sort(lat)
    return lat, lon


def discard_spinup_and_climatologize(ds, spinup_days=365):
    """Discard spin-up, then compute DOY climatology if >1 year remains."""
    n_total = len(ds.time)
    if n_total <= spinup_days:
        print(f"  WARNING: only {n_total} days, need >{spinup_days} for spin-up removal.")
        print(f"           Returning all data without spin-up removal.")
        return ds, 0, False

    ds_post = ds.isel(time=slice(spinup_days, None))
    n_post = len(ds_post.time)
    t0 = ds_post.time.values[0]
    t1 = ds_post.time.values[-1]
    span_days = (np.datetime64(t1, "D") - np.datetime64(t0, "D")).astype(int)
    n_years = span_days / 365.25

    print(f"  Discarded first {spinup_days} days of spin-up.")
    print(f"  Post-spinup: {n_post} days, {t0} → {t1} ({n_years:.1f} years)")

    if n_years > 1.0:
        doy = ds_post.time.dt.dayofyear
        ds_clim = ds_post.groupby(doy).mean(dim="time")
        print(f"  Multi-year data → computed DOY climatology ({len(ds_clim.dayofyear)} days)")
        return ds_clim, n_years, True
    else:
        print(f"  Single post-spinup year → using raw daily data")
        return ds_post, n_years, False


# ERA5 processing (regridded)

def process_era5_regridded(cache_path, jcm_lat, jcm_lon):
    """Load ERA5 850 hPa T, regrid to JCM grid, daily mean, save full field."""
    print("Opening ERA5 zarr (lazy) …")
    era5 = xr.open_zarr(
        ERA5_ZARR, consolidated=True,
        storage_options={"token": "anon"},
    )
    era5 = era5.rename({"latitude": "lat", "longitude": "lon"})

    T850 = era5["temperature"].sel(level=ERA5_LEVEL_HPA, time=ERA5_YEARS)
    print(f"  ERA5 T850 shape (lazy): {dict(T850.sizes)}")

    # Sort ERA5 lat ascending to match JCM
    if T850.lat.values[0] > T850.lat.values[-1]:
        T850 = T850.sortby("lat")

    print(f"  ERA5 grid: lat {T850.lat.values[[0,-1]]}, lon {T850.lon.values[[0,-1]]}")
    print(f"  JCM  grid: lat [{jcm_lat[0]:.2f}, {jcm_lat[-1]:.2f}], lon [{jcm_lon[0]:.2f}, {jcm_lon[-1]:.2f}]")

    # Regrid ERA5 → JCM grid via bilinear interpolation
    print("  Regridding ERA5 → JCM grid …")
    T850_regridded = T850.interp(lat=jcm_lat, lon=jcm_lon, method="linear")

    # Daily mean from 6-hourly
    print("  Computing daily means (this may take a while) …")
    T850_daily = T850_regridded.resample(time="1D").mean()

    print(f"  Saving regridded field → {cache_path}")
    T850_daily.compute().to_netcdf(cache_path)
    print("  Done.")
    return xr.open_dataarray(cache_path)


def process_era5_regional(da_era5, cache_path):
    """Compute regional means from regridded ERA5 field."""
    print("  Computing ERA5 regional means …")
    ds = regional_means(da_era5, REGIONS)
    ds.attrs["description"] = f"ERA5 {ERA5_LEVEL_HPA} hPa daily-mean T, regridded to JCM grid"
    ds.attrs["source"] = ERA5_ZARR
    ds.attrs["years"] = f"{ERA5_YEARS.start}–{ERA5_YEARS.stop}"
    ds.to_netcdf(cache_path)
    print(f"  Saved → {cache_path}")
    return xr.open_dataset(cache_path)


# JCM processing 

def process_jcm(jcm_file, cache_path):
    """Load JCM output, pick 850 hPa level, compute regional means."""
    print(f"Loading JCM output: {jcm_file}")
    ds_jcm = xr.open_dataset(jcm_file)

    print(f"  JCM dims: {dict(ds_jcm.sizes)}")
    print(f"  JCM level values: {ds_jcm.level.values}")
    print(f"  JCM time: {ds_jcm.time.values[0]} → {ds_jcm.time.values[-1]}")

    level_idx = int(np.argmin(np.abs(ds_jcm.level.values - JCM_SIGMA_850)))
    actual_sigma = float(ds_jcm.level.values[level_idx])
    print(f"  Selected level {level_idx}, sigma={actual_sigma:.4f} (target {JCM_SIGMA_850})")

    T_jcm = ds_jcm["temperature"].isel(level=level_idx)
    print(f"  T range: [{float(T_jcm.min()):.1f}, {float(T_jcm.max()):.1f}] K")

    if T_jcm.lat.values[0] > T_jcm.lat.values[-1]:
        T_jcm = T_jcm.sortby("lat")

    print("  Computing regional means …")
    ds = regional_means(T_jcm, REGIONS)
    ds.attrs["description"] = f"JCM sigma={actual_sigma:.3f} (~{int(actual_sigma*1013.25)} hPa) daily T"
    ds.attrs["source"] = str(jcm_file)

    ds.to_netcdf(cache_path)
    print(f"  Saved → {cache_path}")
    return xr.open_dataset(cache_path)


# ERA5 climatology

def era5_doy_climatology(ds_era5):
    """Day-of-year mean ± std from 10 yr of daily regional means."""
    doy = ds_era5.time.dt.dayofyear
    return ds_era5.groupby(doy).mean(dim="time"), ds_era5.groupby(doy).std(dim="time")


# Plotting 

def plot_comparison(ds_era5, ds_jcm, jcm_is_clim, out_path="eval_era5_vs_jcm.png"):
    """Plot JCM vs ERA5. Adapts to climatology or raw daily JCM data."""
    clim_mean, clim_std = era5_doy_climatology(ds_era5)

    region_names = list(REGIONS.keys())
    n = len(region_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    if jcm_is_clim:
        doys = clim_mean.dayofyear.values
        jcm_doys = ds_jcm.dayofyear.values

        for ax, region in zip(axes, region_names):
            cm = clim_mean[region].values
            cs = clim_std[region].values

            ax.fill_between(doys, cm - cs, cm + cs,
                            color="grey", alpha=0.25, label="ERA5 clim ±1σ")
            ax.plot(doys, cm, color="steelblue", lw=1.2, label="ERA5 clim mean")
            ax.plot(jcm_doys, ds_jcm[region].values,
                    color="firebrick", lw=1.8, label="JCM clim mean")

            ax.set_ylabel("T (K)")
            ax.set_title(region, fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Day of year")
        fig.suptitle(f"JCM vs ERA5 climatology — {ERA5_LEVEL_HPA} hPa Temperature",
                     fontsize=13, fontweight="bold", y=1.01)

    else:
        jcm_times = ds_jcm.time.values
        jcm_start = np.datetime64(jcm_times[0], "D")
        jcm_end   = np.datetime64(jcm_times[-1], "D")
        era5_overlap = ds_era5.sel(time=slice(jcm_start, jcm_end))

        doys_for_plot = clim_mean.dayofyear.values
        jcm_year = int(np.datetime_as_string(jcm_start, unit="Y"))
        clim_dates = np.array([
            np.datetime64(f"{jcm_year}-01-01") + np.timedelta64(int(d) - 1, "D")
            for d in doys_for_plot
        ], dtype="datetime64[ns]")

        for ax, region in zip(axes, region_names):
            cm = clim_mean[region].values
            cs = clim_std[region].values

            ax.fill_between(clim_dates, cm - cs, cm + cs,
                            color="grey", alpha=0.25, label="ERA5 clim ±1σ")
            ax.plot(clim_dates, cm, color="grey", lw=0.8, alpha=0.6)

            if len(era5_overlap.time) > 0:
                ax.plot(era5_overlap.time.values, era5_overlap[region].values,
                        color="steelblue", lw=1.3, label=f"ERA5 {jcm_year}")

            ax.plot(jcm_times, ds_jcm[region].values,
                    color="firebrick", lw=1.5, label="JCM")

            ax.set_ylabel("T (K)")
            ax.set_title(region, fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

            margin = np.timedelta64(3, "D")
            ax.set_xlim(jcm_start - margin, jcm_end + margin)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        fig.suptitle(f"JCM vs ERA5 — {ERA5_LEVEL_HPA} hPa Temperature",
                     fontsize=13, fontweight="bold", y=1.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out_path}")
    plt.close(fig)



# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare JCM to ERA5 temperature")
    parser.add_argument("--jcm", default="test_30day.nc",
                        help="Path to JCM output netCDF (default: test_30day.nc)")
    parser.add_argument("--spinup_days", type=int, default=365,
                        help="Days to discard as spin-up (default: 365)")
    parser.add_argument("--recompute", action="store_true",
                        help="Force recompute even if cache exists")
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    era5_field_cache   = CACHE_DIR / "era5_T850_daily_on_jcm_grid.nc"
    era5_regional_cache = CACHE_DIR / "era5_regional_T850_daily.nc"
    jcm_cache = CACHE_DIR / f"jcm_regional_T850_daily_{Path(args.jcm).stem}.nc"

    # ── Read JCM grid ──
    jcm_lat, jcm_lon = get_jcm_grid(args.jcm)
    print(f"JCM grid: {len(jcm_lon)} lon × {len(jcm_lat)} lat")

    # ── ERA5 (regridded to JCM grid) ──
    if era5_field_cache.exists() and not args.recompute:
        print(f"Loading cached regridded ERA5 → {era5_field_cache}")
        da_era5 = xr.open_dataarray(era5_field_cache)
    else:
        da_era5 = process_era5_regridded(era5_field_cache, jcm_lat, jcm_lon)

    if era5_regional_cache.exists() and not args.recompute:
        print(f"Loading cached ERA5 regional → {era5_regional_cache}")
        ds_era5 = xr.open_dataset(era5_regional_cache)
    else:
        ds_era5 = process_era5_regional(da_era5, era5_regional_cache)

    print(f"  ERA5: {len(ds_era5.time)} days, {ds_era5.time.values[0]} → {ds_era5.time.values[-1]}")

    # ── JCM ──
    if jcm_cache.exists() and not args.recompute:
        print(f"Loading cached JCM → {jcm_cache}")
        ds_jcm_raw = xr.open_dataset(jcm_cache)
    else:
        ds_jcm_raw = process_jcm(args.jcm, jcm_cache)

    print(f"  JCM raw: {len(ds_jcm_raw.time)} days, {ds_jcm_raw.time.values[0]} → {ds_jcm_raw.time.values[-1]}")

    # ── Spin-up removal + optional climatology ──
    print("\nProcessing JCM spin-up …")
    ds_jcm, n_years, jcm_is_clim = discard_spinup_and_climatologize(ds_jcm_raw, args.spinup_days)

    # ── Compare ──
    plot_comparison(ds_era5, ds_jcm, jcm_is_clim)


if __name__ == "__main__":
    main()
