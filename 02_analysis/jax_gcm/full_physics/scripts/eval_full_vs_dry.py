#!/usr/bin/env python3
"""
Compare JAX-GCM Full and Dry simulations against ERA5.

ERA5 is regridded onto the JCM lat/lon grid before comparison so that
regional means, biases, and spatial maps are computed on identical
grid points.

This script compares:
  - Full vs ERA5
  - Dry  vs ERA5
  - Full vs Dry (implicitly, through the main comparison)

The first `spinup_days` days of Full and Dry outputs are discarded as spin-up.
If the remaining data spans more than one year, a day-of-year climatology
(multi-year daily mean) is computed before comparison.

Cached intermediate files (in ./eval_cache/):
  - era5_T850_daily_on_jcm_grid.nc
  - era5_regional_T850_daily.nc
  - full_regional_T850_daily_<stem>.nc
  - dry_regional_T850_daily_<stem>.nc

Usage:
    python eval_full_vs_dry.py --full jcm_2year.nc --dry jcm_2year_dry.nc
    python eval_full_vs_dry.py --full jcm_2year.nc --dry jcm_2year_dry.nc --spinup_days 365
    python eval_full_vs_dry.py --full jcm_2year.nc --dry jcm_2year_dry.nc --outdir ./comparison_outputs
    python eval_full_vs_dry.py --full jcm_2year.nc --dry jcm_2year_dry.nc --recompute
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

CACHE_DIR = Path("eval_cache")

# Low-resolution ERA5 from WeatherBench2
ERA5_ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)
ERA5_YEARS = slice("2000", "2009")
ERA5_LEVEL_HPA = 850

JCM_SIGMA_850 = 0.835

REGIONS = {
    "Global":     {"lat": (-90, 90)},
    "Tropics":    {"lat": (-30, 30)},
    "NH Midlats": {"lat": (30, 60)},
    "SH Midlats": {"lat": (-60, -30)},
    "Arctic":     {"lat": (60, 90)},
    "Antarctic":  {"lat": (-90, -60)},
    "N America":  {"lat": (25, 55), "lon": (230, 300)},
}


def cos_weighted_mean(da, lat_name="lat"):
    """Area-weighted (cos-lat) spatial mean."""
    weights = np.cos(np.deg2rad(da[lat_name]))
    weights = weights / weights.mean()
    return da.weighted(weights).mean(dim=[lat_name, "lon"])


def regional_means(da, regions, lat_name="lat"):
    """Compute cos-lat-weighted mean for each region → Dataset."""
    ds = xr.Dataset()
    for name, bounds in regions.items():
        lat_s, lat_n = bounds["lat"]
        subset = da.sel(**{lat_name: slice(lat_s, lat_n)})

        if "lon" in bounds:
            lon_s, lon_n = bounds["lon"]
            if lon_s < lon_n:
                subset = subset.sel(lon=slice(lon_s, lon_n))
            else:
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
    lat = np.sort(lat)
    return lat, lon


def discard_spinup_and_climatologize(ds, spinup_days=365):
    """Discard spin-up, then compute DOY climatology if >1 year remains."""
    n_total = len(ds.time)
    if n_total <= spinup_days:
        print(f"  WARNING: only {n_total} days, need >{spinup_days} for spin-up removal.")
        print("           Returning all data without spin-up removal.")
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
        print("  Single post-spinup year → using raw daily data")
        return ds_post, n_years, False


# ERA5 processing

def process_era5_regridded(cache_path, jcm_lat, jcm_lon):
    """Load ERA5 850 hPa T, regrid to JCM grid, daily mean, save full field."""
    print("Opening ERA5 zarr (lazy) …")
    era5 = xr.open_zarr(
        ERA5_ZARR,
        consolidated=True,
        storage_options={"token": "anon"},
    )
    era5 = era5.rename({"latitude": "lat", "longitude": "lon"})

    T850 = era5["temperature"].sel(level=ERA5_LEVEL_HPA, time=ERA5_YEARS)
    print(f"  ERA5 T850 shape (lazy): {dict(T850.sizes)}")

    if T850.lat.values[0] > T850.lat.values[-1]:
        T850 = T850.sortby("lat")

    print(f"  ERA5 grid: lat {T850.lat.values[[0, -1]]}, lon {T850.lon.values[[0, -1]]}")
    print(f"  JCM  grid: lat [{jcm_lat[0]:.2f}, {jcm_lat[-1]:.2f}], lon [{jcm_lon[0]:.2f}, {jcm_lon[-1]:.2f}]")

    print("  Regridding ERA5 → JCM grid …")
    T850_regridded = T850.interp(lat=jcm_lat, lon=jcm_lon, method="linear")

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

def process_jcm(jcm_file, cache_path, label):
    """Load JCM output, pick ~850 hPa level, compute regional means."""
    print(f"Loading {label} output: {jcm_file}")
    ds_jcm = xr.open_dataset(jcm_file)

    print(f"  {label} dims: {dict(ds_jcm.sizes)}")
    print(f"  {label} level values: {ds_jcm.level.values}")
    print(f"  {label} time: {ds_jcm.time.values[0]} → {ds_jcm.time.values[-1]}")

    level_idx = int(np.argmin(np.abs(ds_jcm.level.values - JCM_SIGMA_850)))
    actual_sigma = float(ds_jcm.level.values[level_idx])
    print(f"  Selected level {level_idx}, sigma={actual_sigma:.4f} (target {JCM_SIGMA_850})")

    T_jcm = ds_jcm["temperature"].isel(level=level_idx)
    print(f"  T range: [{float(T_jcm.min()):.1f}, {float(T_jcm.max()):.1f}] K")

    if T_jcm.lat.values[0] > T_jcm.lat.values[-1]:
        T_jcm = T_jcm.sortby("lat")

    print(f"  Computing {label} regional means …")
    ds = regional_means(T_jcm, REGIONS)
    ds.attrs["description"] = (
        f"{label} JCM sigma={actual_sigma:.3f} (~{int(actual_sigma * 1013.25)} hPa) daily T"
    )
    ds.attrs["source"] = str(jcm_file)

    ds.to_netcdf(cache_path)
    print(f"  Saved → {cache_path}")
    return xr.open_dataset(cache_path)


def era5_doy_climatology(ds_era5):
    """Day-of-year mean ± std from daily ERA5 regional means."""
    doy = ds_era5.time.dt.dayofyear
    return ds_era5.groupby(doy).mean(dim="time"), ds_era5.groupby(doy).std(dim="time")


# Plotting

def plot_comparison(ds_era5, ds_full_raw, ds_dry_raw, ds_full, ds_dry, jcm_is_clim, out_path):
    """Plot Full and Dry against ERA5, with std bands in climatology mode."""
    clim_mean, clim_std = era5_doy_climatology(ds_era5)

    region_names = list(REGIONS.keys())
    n = len(region_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    if jcm_is_clim:
        full_clim_mean, full_clim_std = jcm_doy_climatology(ds_full_raw)
        dry_clim_mean, dry_clim_std = jcm_doy_climatology(ds_dry_raw)

        doys = clim_mean.dayofyear.values
        full_doys = full_clim_mean.dayofyear.values
        dry_doys = dry_clim_mean.dayofyear.values

        for ax, region in zip(axes, region_names):
            cm = clim_mean[region].values
            cs = clim_std[region].values

            fm = full_clim_mean[region].values
            fs = full_clim_std[region].values

            dm = dry_clim_mean[region].values
            ds = dry_clim_std[region].values

            ax.fill_between(doys, cm - cs, cm + cs,
                            color="grey", alpha=0.25, label="ERA5 clim ±1σ")
            ax.plot(doys, cm, color="black", lw=1.2, label="ERA5 clim mean")

            ax.fill_between(full_doys, fm - fs, fm + fs,
                            color="firebrick", alpha=0.20, label="Full clim ±1σ")
            ax.plot(full_doys, fm,
                    color="firebrick", lw=1.8, label="Full clim mean")

            ax.fill_between(dry_doys, dm - ds, dm + ds,
                            color="steelblue", alpha=0.20, label="Dry clim ±1σ")
            ax.plot(dry_doys, dm,
                    color="steelblue", lw=1.8, label="Dry clim mean")

            ax.set_ylabel("T (K)")
            ax.set_title(region, fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Day of year")
        fig.suptitle(
            f"Full and Dry vs ERA5 climatology — {ERA5_LEVEL_HPA} hPa Temperature",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

    else:
        full_times = ds_full.time.values
        dry_times = ds_dry.time.values

        t_start = min(np.datetime64(full_times[0], "D"), np.datetime64(dry_times[0], "D"))
        t_end = max(np.datetime64(full_times[-1], "D"), np.datetime64(dry_times[-1], "D"))
        era5_overlap = ds_era5.sel(time=slice(t_start, t_end))

        doys_for_plot = clim_mean.dayofyear.values
        ref_year = int(np.datetime_as_string(t_start, unit="Y"))
        clim_dates = np.array([
            np.datetime64(f"{ref_year}-01-01") + np.timedelta64(int(d) - 1, "D")
            for d in doys_for_plot
        ], dtype="datetime64[ns]")

        for ax, region in zip(axes, region_names):
            cm = clim_mean[region].values
            cs = clim_std[region].values

            ax.fill_between(clim_dates, cm - cs, cm + cs,
                            color="grey", alpha=0.25, label="ERA5 clim ±1σ")
            ax.plot(clim_dates, cm, color="grey", lw=0.8, alpha=0.7)

            if len(era5_overlap.time) > 0:
                ax.plot(era5_overlap.time.values, era5_overlap[region].values,
                        color="black", lw=1.2, label=f"ERA5 {ref_year}")

            ax.plot(full_times, ds_full[region].values,
                    color="firebrick", lw=1.5, label="Full")
            ax.plot(dry_times, ds_dry[region].values,
                    color="steelblue", lw=1.5, label="Dry")

            ax.set_ylabel("T (K)")
            ax.set_title(region, fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

            margin = np.timedelta64(3, "D")
            ax.set_xlim(t_start - margin, t_end + margin)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        fig.suptitle(
            f"Full and Dry vs ERA5 — {ERA5_LEVEL_HPA} hPa Temperature",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out_path}")
    plt.close(fig)


def plot_bias_both(ds_era5, ds_full, ds_dry, jcm_is_clim, out_path):
    """Plot Full−ERA5 and Dry−ERA5 on the same regional figure."""
    regions = list(REGIONS.keys())
    n = len(regions)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        if jcm_is_clim:
            full = ds_full[region]
            dry = ds_dry[region]

            era5_ref = ds_era5.groupby(ds_era5.time.dt.dayofyear).mean(dim="time")[region]
            era5_ref = era5_ref.sel(dayofyear=full.dayofyear)

            x = full.dayofyear.values
            full_bias = full - era5_ref
            dry_bias = dry - era5_ref
        else:
            full = ds_full[region]
            dry = ds_dry[region]

            era5_full = ds_era5.sel(time=full.time)[region]
            era5_dry = ds_era5.sel(time=dry.time)[region]

            x = full.time.values
            full_bias = full - era5_full
            dry_bias = dry - era5_dry

        ax.axhline(0, color="black", lw=0.8)
        ax.plot(x, full_bias.values, color="firebrick", lw=1.8, label="Full − ERA5")
        ax.plot(x, dry_bias.values, color="steelblue", lw=1.8, label="Dry − ERA5")

        ax.set_ylabel("Bias (K)")
        ax.set_title(region, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        if not jcm_is_clim:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axes[-1].set_xlabel("Day of year" if jcm_is_clim else "Time")
    fig.suptitle("Temperature Bias Relative to ERA5 (~850 hPa)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved bias plot → {out_path}")
    plt.close(fig)


def plot_bias_map(da_era5_field, da_jcm_field, out_path, title):
    """Plot global lat-lon map of JCM − ERA5 bias (time mean at ~850 hPa)."""

    era_start = np.datetime64(da_era5_field.time.values[0], "D")
    era_end   = np.datetime64(da_era5_field.time.values[-1], "D")
    jcm_start = np.datetime64(da_jcm_field.time.values[0], "D")
    jcm_end   = np.datetime64(da_jcm_field.time.values[-1], "D")

    t0 = max(era_start, jcm_start)
    t1 = min(era_end, jcm_end)

    if t1 < t0:
        raise ValueError("No overlapping time range between JCM and ERA5 for bias map.")

    da_jcm_overlap = da_jcm_field.sel(time=slice(t0, t1))
    da_era5_overlap = da_era5_field.sel(time=slice(t0, t1))

    da_era5_overlap = da_era5_overlap.sel(
        time=da_jcm_overlap.time,
        method="nearest",
    )

    era_mean = da_era5_overlap.mean(dim="time")
    jcm_mean = da_jcm_overlap.mean(dim="time")

    bias = jcm_mean - era_mean
    bias = bias.transpose("lat", "lon")

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(
        bias.lon,
        bias.lat,
        bias,
        cmap="RdBu_r",
        vmin=-10,
        vmax=10,
        shading="auto",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Bias (K)")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved bias map → {out_path}")
    plt.close(fig)

def jcm_doy_climatology(ds_jcm):
    """Day-of-year mean ± std from multi-year JCM daily regional means."""
    doy = ds_jcm.time.dt.dayofyear
    return ds_jcm.groupby(doy).mean(dim="time"), ds_jcm.groupby(doy).std(dim="time")

def plot_bias_map_seasonal(da_era5_field, da_jcm_field, season, out_path, title):
    """Plot seasonal lat-lon map of JCM − ERA5 bias (~850 hPa)."""

    season_months = {
        "DJF": [12, 1, 2],
        "JJA": [6, 7, 8],
    }
    if season not in season_months:
        raise ValueError(f"Unsupported season: {season}")

    era_start = np.datetime64(da_era5_field.time.values[0], "D")
    era_end   = np.datetime64(da_era5_field.time.values[-1], "D")
    jcm_start = np.datetime64(da_jcm_field.time.values[0], "D")
    jcm_end   = np.datetime64(da_jcm_field.time.values[-1], "D")

    t0 = max(era_start, jcm_start)
    t1 = min(era_end, jcm_end)

    if t1 < t0:
        raise ValueError("No overlapping time range between JCM and ERA5 for seasonal bias map.")

    da_jcm_overlap = da_jcm_field.sel(time=slice(t0, t1))
    da_era5_overlap = da_era5_field.sel(time=slice(t0, t1))

    da_era5_overlap = da_era5_overlap.sel(
        time=da_jcm_overlap.time,
        method="nearest",
    )

    months = season_months[season]
    da_jcm_season = da_jcm_overlap.where(da_jcm_overlap.time.dt.month.isin(months), drop=True)
    da_era5_season = da_era5_overlap.where(da_era5_overlap.time.dt.month.isin(months), drop=True)

    era_mean = da_era5_season.mean(dim="time")
    jcm_mean = da_jcm_season.mean(dim="time")

    bias = jcm_mean - era_mean
    bias = bias.transpose("lat", "lon")

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(
        bias.lon,
        bias.lat,
        bias,
        cmap="RdBu_r",
        vmin=-10,
        vmax=10,
        shading="auto",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Bias (K)")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved seasonal bias map → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare JAX-GCM Full and Dry simulations against ERA5 at ~850 hPa"
    )
    parser.add_argument(
        "--full",
        required=True,
        help="Path to Full JAX-GCM output netCDF",
    )
    parser.add_argument(
        "--dry",
        required=True,
        help="Path to Dry JAX-GCM output netCDF",
    )
    parser.add_argument(
        "--spinup_days",
        type=int,
        default=365,
        help="Days to discard as spin-up (default: 365)",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where output figures will be saved",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute even if cache exists",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    outdir = Path(args.outdir)
    graphs_dir = outdir / "graphs_png"

    graphs_dir.mkdir(parents=True, exist_ok=True)

    era5_field_cache = CACHE_DIR / "era5_T850_daily_on_jcm_grid.nc"
    era5_regional_cache = CACHE_DIR / "era5_regional_T850_daily.nc"
    full_cache = CACHE_DIR / f"full_regional_T850_daily_{Path(args.full).stem}.nc"
    dry_cache = CACHE_DIR / f"dry_regional_T850_daily_{Path(args.dry).stem}.nc"

    # Read JCM grid
    full_lat, full_lon = get_jcm_grid(args.full)
    dry_lat, dry_lon = get_jcm_grid(args.dry)

    if not (np.allclose(full_lat, dry_lat) and np.allclose(full_lon, dry_lon)):
        raise ValueError("Full and Dry grids do not match.")

    print(f"Common JCM grid: {len(full_lon)} lon × {len(full_lat)} lat")

    # ERA5
    if era5_field_cache.exists() and not args.recompute:
        print(f"Loading cached regridded ERA5 → {era5_field_cache}")
        da_era5 = xr.open_dataarray(era5_field_cache)
    else:
        da_era5 = process_era5_regridded(era5_field_cache, full_lat, full_lon)

    if era5_regional_cache.exists() and not args.recompute:
        print(f"Loading cached ERA5 regional → {era5_regional_cache}")
        ds_era5 = xr.open_dataset(era5_regional_cache)
    else:
        ds_era5 = process_era5_regional(da_era5, era5_regional_cache)

    print(f"  ERA5: {len(ds_era5.time)} days, {ds_era5.time.values[0]} → {ds_era5.time.values[-1]}")

    # Full
    if full_cache.exists() and not args.recompute:
        print(f"Loading cached Full regional → {full_cache}")
        ds_full_raw = xr.open_dataset(full_cache)
    else:
        ds_full_raw = process_jcm(args.full, full_cache, label="Full")

    # Dry
    if dry_cache.exists() and not args.recompute:
        print(f"Loading cached Dry regional → {dry_cache}")
        ds_dry_raw = xr.open_dataset(dry_cache)
    else:
        ds_dry_raw = process_jcm(args.dry, dry_cache, label="Dry")

    print(f"  Full raw: {len(ds_full_raw.time)} days, {ds_full_raw.time.values[0]} → {ds_full_raw.time.values[-1]}")
    print(f"  Dry  raw: {len(ds_dry_raw.time)} days, {ds_dry_raw.time.values[0]} → {ds_dry_raw.time.values[-1]}")

    # Full field for map
    ds_full_field = xr.open_dataset(args.full)
    level_idx_full = int(np.argmin(np.abs(ds_full_field.level.values - JCM_SIGMA_850)))
    da_full_field = ds_full_field["temperature"].isel(level=level_idx_full)
    if da_full_field.lat.values[0] > da_full_field.lat.values[-1]:
        da_full_field = da_full_field.sortby("lat")

    # Dry field for map
    ds_dry_field = xr.open_dataset(args.dry)
    level_idx_dry = int(np.argmin(np.abs(ds_dry_field.level.values - JCM_SIGMA_850)))
    da_dry_field = ds_dry_field["temperature"].isel(level=level_idx_dry)
    if da_dry_field.lat.values[0] > da_dry_field.lat.values[-1]:
        da_dry_field = da_dry_field.sortby("lat")

    # Spin-up removal + optional climatology
    print("\nProcessing Full spin-up …")
    ds_full, _, full_is_clim = discard_spinup_and_climatologize(ds_full_raw, args.spinup_days)

    print("\nProcessing Dry spin-up …")
    ds_dry, _, dry_is_clim = discard_spinup_and_climatologize(ds_dry_raw, args.spinup_days)

    if full_is_clim != dry_is_clim:
        raise ValueError("Full and Dry do not produce the same comparison mode (climatology vs raw).")

    # Post-spinup raw fields for maps (keep time dimension)
    if len(da_full_field.time) <= args.spinup_days:
        print("WARNING: not enough Full data for spin-up removal in map field; using full field.")
        da_full_field_post = da_full_field
    else:
        da_full_field_post = da_full_field.isel(time=slice(args.spinup_days, None))

    if len(da_dry_field.time) <= args.spinup_days:
        print("WARNING: not enough Dry data for spin-up removal in map field; using full field.")
        da_dry_field_post = da_dry_field
    else:
        da_dry_field_post = da_dry_field.isel(time=slice(args.spinup_days, None))

    # Plots
    plot_comparison(
        ds_era5,
        ds_full_raw,
        ds_dry_raw,
        ds_full,
        ds_dry,
        full_is_clim,
        out_path=graphs_dir / "eval_era5_full_dry.png",
    )

    plot_bias_both(
        ds_era5,
        ds_full,
        ds_dry,
        full_is_clim,
        out_path=graphs_dir / "bias_full_and_dry_minus_era5.png",
    )

    if not full_is_clim:
        plot_bias_map(
            da_era5,
            da_full_field_post,
            out_path=graphs_dir / "bias_map_full_minus_era5.png",
            title="Full − ERA5 Time-Mean Temperature Bias (850 hPa)",
        )

        plot_bias_map(
            da_era5,
            da_dry_field_post,
            out_path=graphs_dir / "bias_map_dry_minus_era5.png",
            title="Dry − ERA5 Time-Mean Temperature Bias (850 hPa)",
        )
    else:
        print("Skipping bias maps for climatology mode (current map plot expects a time dimension).")

    plot_bias_map_seasonal(
        da_era5,
        da_full_field_post,
        season="DJF",
        out_path=graphs_dir / "bias_map_full_minus_era5_DJF.png",
        title="Full − ERA5 Temperature Bias (850 hPa, DJF)",
    )

    plot_bias_map_seasonal(
        da_era5,
        da_full_field_post,
        season="JJA",
        out_path=graphs_dir / "bias_map_full_minus_era5_JJA.png",
        title="Full − ERA5 Temperature Bias (850 hPa, JJA)",
    )

    plot_bias_map_seasonal(
        da_era5,
        da_dry_field_post,
        season="DJF",
        out_path=graphs_dir / "bias_map_dry_minus_era5_DJF.png",
        title="Dry − ERA5 Temperature Bias (850 hPa, DJF)",
    )

    plot_bias_map_seasonal(
        da_era5,
        da_dry_field_post,
        season="JJA",
        out_path=graphs_dir / "bias_map_dry_minus_era5_JJA.png",
        title="Dry − ERA5 Temperature Bias (850 hPa, JJA)",
    )

    #Write README
    readme_path = outdir / "README.txt"
    with open(readme_path, "w") as f:
        f.write(f"""Experiment: Full vs Dry (physics=None)
    Full file: {args.full}
    Dry file: {args.dry}

    Spin-up removed: {args.spinup_days} days

    Outputs:
    - run_nc/: processed datasets
    - graphs_png/: figures

    Notes:
    - Dry = no physics
    - Strong cold drift expected
    """)

if __name__ == "__main__":
    main()