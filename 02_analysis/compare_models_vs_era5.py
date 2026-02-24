#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare DINO-only and JAX-GCM full-physics vs ERA5 snapshots (2 days, 12h step).
Outputs:
  - metrics.csv (bias + RMSE for each lead time and each model)
  - 3 figures: PS_hPa.png, T_K.png, Vmag_ms.png

Memory-stable strategy:
  - open model datasets lazily with small chunks
  - loop over times, open ONE ERA5 snapshot at a time
  - regrid ERA5 -> model grid with xarray.interp (bilinear)
  - compute metrics streaming (no giant arrays kept)
"""

import os
import re
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# -----------------------------
# User paths defaults (edit if you want)
# -----------------------------
DEFAULT_FULL = "/home/gmathieu/links/scratch/jaxgcm_outputs/pred_ds_jax_gcm_full_physics_2days_12h.nc"
DEFAULT_DINO = "/home/gmathieu/links/scratch/dinosaur_outputs/ds_out_dino_2days_12h.nc"
DEFAULT_ERA5_LIST = [
    "/home/gmathieu/links/scratch/era5_snapshot/era5_light2d_19900501T00.nc",
    "/home/gmathieu/links/scratch/era5_snapshot/era5_light2d_19900501T12.nc",
    "/home/gmathieu/links/scratch/era5_snapshot/era5_light2d_19900502T00.nc",
    "/home/gmathieu/links/scratch/era5_snapshot/era5_light2d_19900502T12.nc",
    "/home/gmathieu/links/scratch/era5_snapshot/era5_light2d_19900503T00.nc",
]


# -----------------------------
# Helpers: variable + coord detection
# -----------------------------
def _first_existing(ds: xr.Dataset, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    return None


def find_lat_lon_names(ds: xr.Dataset):
    lat = _first_existing(ds, ["lat", "latitude", "Latitude", "nav_lat", "y"])
    lon = _first_existing(ds, ["lon", "longitude", "Longitude", "nav_lon", "x"])
    if lat is None or lon is None:
        raise KeyError(
            f"Could not find lat/lon in dataset. Found vars: {list(ds.variables)[:50]}"
        )
    return lat, lon


def normalize_lon(da_lon: xr.DataArray, target_range="[-180,180)"):
    """
    Normalize longitude coordinate to match common conventions.
    - If target_range is [-180,180): map lon to [-180,180)
    - If target_range is [0,360): map lon to [0,360)
    """
    lon = da_lon.copy()
    if target_range == "[-180,180)":
        lon = ((lon + 180) % 360) - 180
    elif target_range == "[0,360)":
        lon = lon % 360
    else:
        raise ValueError("target_range must be '[-180,180)' or '[0,360)'.")
    return lon


def maybe_sort_lon(ds: xr.Dataset, lon_name: str):
    # xarray.interp likes monotonic coords
    lon = ds[lon_name]
    if lon.ndim != 1:
        return ds  # grid might be curvilinear; interp won't work in that case
    if not np.all(np.diff(lon.values) > 0):
        return ds.sortby(lon_name)
    return ds


def ensure_1d_lat_lon(ds: xr.Dataset, lat_name: str, lon_name: str):
    if ds[lat_name].ndim != 1 or ds[lon_name].ndim != 1:
        raise ValueError(
            f"Lat/Lon are not 1D in this dataset (lat.ndim={ds[lat_name].ndim}, lon.ndim={ds[lon_name].ndim}). "
            "This script uses xarray.interp (works for rectilinear 1D lat/lon). "
            "If your grid is curvilinear, we should switch to xESMF."
        )


def get_time_name(ds: xr.Dataset):
    t = _first_existing(ds, ["time", "Time", "valid_time", "forecast_time"])
    if t is None:
        raise KeyError("Could not find time coordinate (tried: time, valid_time, forecast_time).")
    return t


def find_sp_name(ds: xr.Dataset, prefer_normalized=False):
    if prefer_normalized:
        # FULL case
        return _first_existing(ds, ["normalized_surface_pressure", "normalized_ps", "ps_norm"])
    return _first_existing(ds, ["surface_pressure", "sp", "ps"])


def find_t_name(ds: xr.Dataset):
    return _first_existing(ds, ["temperature", "t2m", "t", "T"])


def find_u_name(ds: xr.Dataset):
    return _first_existing(ds, ["u_component_of_wind", "u10", "u", "U"])


def find_v_name(ds: xr.Dataset):
    return _first_existing(ds, ["v_component_of_wind", "v10", "v", "V"])


# -----------------------------
# Physics conversions
# -----------------------------
def sp_to_hpa(sp_da: xr.DataArray) -> xr.DataArray:
    # If units in attrs say Pa, convert; else infer by magnitude.
    units = str(sp_da.attrs.get("units", "")).lower()
    if "pa" in units:
        return sp_da / 100.0
    # heuristic: typical ps ~ 100000 Pa or 1000 hPa
    m = float(sp_da.mean().values)
    if m > 2000.0:   # likely Pa
        return sp_da / 100.0
    return sp_da     # already hPa


def reconstruct_full_ps_hpa(ds_full: xr.Dataset, sp_norm_name: str) -> xr.DataArray:
    """
    FULL uses normalized_surface_pressure. We reconstruct p_s.

    Strategy:
      - If dataset has a scalar variable 'p0' or 'reference_surface_pressure', use it (Pa)
      - Else, default p0 = 1e5 Pa (standard)
    """
    # Find p0-like scalar
    p0_var = _first_existing(ds_full, ["p0", "reference_surface_pressure", "p_ref", "surface_pressure_reference"])
    if p0_var is not None:
        p0 = ds_full[p0_var]
        # if p0 has dims, try to reduce to scalar
        if p0.size > 1:
            p0 = p0.isel({d: 0 for d in p0.dims})
        p0_val = float(p0.values)
    else:
        p0_val = float(ds_full.attrs.get("p0", 1e5))

    ps_pa = ds_full[sp_norm_name] * p0_val
    ps_pa.attrs["units"] = "Pa"
    return sp_to_hpa(ps_pa)


def vmag(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(
        lambda a, b: np.sqrt(a * a + b * b),
        u, v,
        dask="allowed",
        output_dtypes=[np.float32],
    )


# -----------------------------
# Regridding (ERA -> model grid)
# -----------------------------
def regrid_era_to_model(era_da: xr.DataArray, era_lat: str, era_lon: str,
                        model_lat_vals: xr.DataArray, model_lon_vals: xr.DataArray,
                        target_lon_range: str):
    """
    Bilinear regrid using xarray.interp for rectilinear 1D lat/lon.
    - normalizes ERA lon to match model lon convention
    - sorts lon to be monotonic
    """
    # Normalize ERA longitude to match model convention
    era_da = era_da.assign_coords({era_lon: normalize_lon(era_da[era_lon], target_lon_range)})

    # Ensure monotonic lon for interp
    if era_da[era_lon].ndim == 1 and not np.all(np.diff(era_da[era_lon].values) > 0):
        era_da = era_da.sortby(era_lon)

    # Interp onto model grid
    out = era_da.interp(
        {era_lat: model_lat_vals, era_lon: model_lon_vals},
        method="linear",
        kwargs={"fill_value": np.nan},
    )
    return out


def detect_model_lon_range(model_lon: xr.DataArray) -> str:
    """
    Decide whether model lon looks like [-180,180) or [0,360).
    """
    mn = float(model_lon.min().values)
    mx = float(model_lon.max().values)
    # If max > 180 -> probably [0,360)
    if mx > 180.0:
        return "[0,360)"
    return "[-180,180)"


# -----------------------------
# Metrics (optionally cosine-weighted)
# -----------------------------
def area_weights_coslat(lat_1d: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat_1d))
    w = xr.DataArray(w, coords={lat_1d.name: lat_1d}, dims=(lat_1d.name,))
    return w


def weighted_mean_2d(da2d: xr.DataArray, lat_name: str, lon_name: str, weights_lat: xr.DataArray):
    # weights only depend on latitude
    w2 = weights_lat / weights_lat.mean()
    # broadcast to 2D
    w_b = w2.broadcast_like(da2d)
    num = (da2d * w_b).sum(dim=(lat_name, lon_name), skipna=True)
    den = w_b.where(np.isfinite(da2d)).sum(dim=(lat_name, lon_name), skipna=True)
    return num / den


def bias_rmse(model: xr.DataArray, truth: xr.DataArray, lat_name: str, lon_name: str,
              weights_lat: xr.DataArray | None):
    diff = model - truth
    if weights_lat is None:
        bias = diff.mean(dim=(lat_name, lon_name), skipna=True)
        rmse = np.sqrt((diff ** 2).mean(dim=(lat_name, lon_name), skipna=True))
    else:
        bias = weighted_mean_2d(diff, lat_name, lon_name, weights_lat)
        rmse = np.sqrt(weighted_mean_2d(diff ** 2, lat_name, lon_name, weights_lat))
    return float(bias.values), float(rmse.values)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=DEFAULT_FULL, help="FULL model NetCDF path")
    ap.add_argument("--dino", default=DEFAULT_DINO, help="DINO model NetCDF path")
    ap.add_argument("--era5", nargs="+", default=DEFAULT_ERA5_LIST, help="List of ERA5 snapshot files (ordered in time)")
    ap.add_argument("--outdir", default="./compare_out", help="Output directory")
    ap.add_argument("--no_weights", action="store_true", help="Disable cosine(latitude) weighting")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Open model datasets lazily (keep memory low)
    # chunk time=1 to avoid loading whole cubes
    ds_full = xr.open_dataset(args.full, chunks={"time": 1})
    ds_dino = xr.open_dataset(args.dino, chunks={"time": 1})

    # ---- Detect coords/time
    t_full = get_time_name(ds_full)
    t_dino = get_time_name(ds_dino)
    lat_full, lon_full = find_lat_lon_names(ds_full)
    lat_dino, lon_dino = find_lat_lon_names(ds_dino)

    ensure_1d_lat_lon(ds_full, lat_full, lon_full)
    ensure_1d_lat_lon(ds_dino, lat_dino, lon_dino)

    ds_full = maybe_sort_lon(ds_full, lon_full)
    ds_dino = maybe_sort_lon(ds_dino, lon_dino)

    lon_range_full = detect_model_lon_range(ds_full[lon_full])
    lon_range_dino = detect_model_lon_range(ds_dino[lon_dino])

    # ---- Detect vars (models)
    sp_full_norm = find_sp_name(ds_full, prefer_normalized=True)
    if sp_full_norm is None:
        raise KeyError("FULL: could not find normalized surface pressure (expected 'normalized_surface_pressure' or similar).")
    t_full_name = find_t_name(ds_full)
    u_full_name = find_u_name(ds_full)
    v_full_name = find_v_name(ds_full)

    sp_dino_name = find_sp_name(ds_dino, prefer_normalized=False)
    t_dino_name = find_t_name(ds_dino)
    u_dino_name = find_u_name(ds_dino)
    v_dino_name = find_v_name(ds_dino)

    for nm, who in [
        (t_full_name, "FULL temperature"),
        (u_full_name, "FULL u"),
        (v_full_name, "FULL v"),
        (sp_dino_name, "DINO surface pressure"),
        (t_dino_name, "DINO temperature"),
        (u_dino_name, "DINO u"),
        (v_dino_name, "DINO v"),
    ]:
        if nm is None:
            raise KeyError(f"Missing variable for {who}. Please adjust name detection at top.")

    # ---- Load ERA5 coord names from first snapshot
    ds_era0 = xr.open_dataset(args.era5[0])
    t_era = get_time_name(ds_era0)
    lat_era, lon_era = find_lat_lon_names(ds_era0)
    ensure_1d_lat_lon(ds_era0, lat_era, lon_era)

    sp_era_name = find_sp_name(ds_era0, prefer_normalized=False)
    t_era_name = find_t_name(ds_era0)
    u_era_name = find_u_name(ds_era0)
    v_era_name = find_v_name(ds_era0)
    if sp_era_name is None or t_era_name is None or u_era_name is None or v_era_name is None:
        raise KeyError(
            "ERA5: could not detect required variables among "
            f"sp={sp_era_name}, T={t_era_name}, u={u_era_name}, v={v_era_name}. "
            "Adjust find_*_name() candidates."
        )
    ds_era0.close()

    # ---- Determine lead times from ERA5 file list (assume ordered)
    # We'll parse times from each ERA snapshot file content (robust).
    era_times = []
    for f in args.era5:
        ds_e = xr.open_dataset(f)
        # If time length > 1, take first (snapshots should be length 1)
        tt = ds_e[t_era]
        if tt.size > 1:
            tt = tt.isel({t_era: 0})
        era_times.append(pd.Timestamp(tt.values))
        ds_e.close()

    t0 = era_times[0]
    lead_hours = [int((t - t0) / pd.Timedelta(hours=1)) for t in era_times]
    # Expect [0,12,24,36,48]
    # Still proceed even if slightly different.

    # ---- (Optional) cosine-lat weights per model grid
    w_full = None if args.no_weights else area_weights_coslat(ds_full[lat_full])
    w_dino = None if args.no_weights else area_weights_coslat(ds_dino[lat_dino])

    rows = []

    # ---- Main loop over lead times: open ERA snapshot, take model slice, regrid ERA->model, compute metrics
    for k, (f_era, valid_time, lead) in enumerate(zip(args.era5, era_times, lead_hours)):
        print(f"[{k+1}/{len(args.era5)}] lead={lead:>3}h  ERA={Path(f_era).name}")

        ds_e = xr.open_dataset(f_era)
        # extract ERA vars (drop time dim if present)
        era_sp = ds_e[sp_era_name]
        era_t = ds_e[t_era_name]
        era_u = ds_e[u_era_name]
        era_v = ds_e[v_era_name]

        # If ERA has time dim, select first
        if t_era in era_sp.dims:
            era_sp = era_sp.isel({t_era: 0})
            era_t = era_t.isel({t_era: 0})
            era_u = era_u.isel({t_era: 0})
            era_v = era_v.isel({t_era: 0})

        era_sp_hpa = sp_to_hpa(era_sp)
        era_vmag = vmag(era_u, era_v)

        # ---- FULL at this lead time:
        # select by index (assume same number of times as era snapshots)
        full_slice = ds_full.isel({t_full: k})

        full_ps_hpa = reconstruct_full_ps_hpa(full_slice, sp_full_norm)
        full_t = full_slice[t_full_name]
        full_vmag = vmag(full_slice[u_full_name], full_slice[v_full_name])

        # Regrid ERA -> FULL grid
        era_ps_on_full = regrid_era_to_model(
            era_sp_hpa, lat_era, lon_era,
            ds_full[lat_full], ds_full[lon_full],
            lon_range_full
        )
        era_t_on_full = regrid_era_to_model(
            era_t, lat_era, lon_era,
            ds_full[lat_full], ds_full[lon_full],
            lon_range_full
        )
        era_v_on_full = regrid_era_to_model(
            era_vmag, lat_era, lon_era,
            ds_full[lat_full], ds_full[lon_full],
            lon_range_full
        )

        # Compute metrics (trigger minimal load)
        b_ps, r_ps = bias_rmse(full_ps_hpa, era_ps_on_full, lat_full, lon_full, w_full)
        b_t,  r_t  = bias_rmse(full_t,     era_t_on_full,  lat_full, lon_full, w_full)
        b_v,  r_v  = bias_rmse(full_vmag,  era_v_on_full,  lat_full, lon_full, w_full)

        rows += [
            dict(model="FULL", var="PS", units="hPa", lead_h=lead, valid_time=str(valid_time), bias=b_ps, rmse=r_ps),
            dict(model="FULL", var="T",  units="K",   lead_h=lead, valid_time=str(valid_time), bias=b_t,  rmse=r_t),
            dict(model="FULL", var="V",  units="m/s", lead_h=lead, valid_time=str(valid_time), bias=b_v,  rmse=r_v),
        ]

        # ---- DINO at this lead time:
        dino_slice = ds_dino.isel({t_dino: k})

        dino_sp_hpa = sp_to_hpa(dino_slice[sp_dino_name])
        dino_t = dino_slice[t_dino_name]
        dino_vmag = vmag(dino_slice[u_dino_name], dino_slice[v_dino_name])

        # Regrid ERA -> DINO grid
        era_ps_on_dino = regrid_era_to_model(
            era_sp_hpa, lat_era, lon_era,
            ds_dino[lat_dino], ds_dino[lon_dino],
            lon_range_dino
        )
        era_t_on_dino = regrid_era_to_model(
            era_t, lat_era, lon_era,
            ds_dino[lat_dino], ds_dino[lon_dino],
            lon_range_dino
        )
        era_v_on_dino = regrid_era_to_model(
            era_vmag, lat_era, lon_era,
            ds_dino[lat_dino], ds_dino[lon_dino],
            lon_range_dino
        )

        b_ps, r_ps = bias_rmse(dino_sp_hpa, era_ps_on_dino, lat_dino, lon_dino, w_dino)
        b_t,  r_t  = bias_rmse(dino_t,      era_t_on_dino,  lat_dino, lon_dino, w_dino)
        b_v,  r_v  = bias_rmse(dino_vmag,   era_v_on_dino,  lat_dino, lon_dino, w_dino)

        rows += [
            dict(model="DINO", var="PS", units="hPa", lead_h=lead, valid_time=str(valid_time), bias=b_ps, rmse=r_ps),
            dict(model="DINO", var="T",  units="K",   lead_h=lead, valid_time=str(valid_time), bias=b_t,  rmse=r_t),
            dict(model="DINO", var="V",  units="m/s", lead_h=lead, valid_time=str(valid_time), bias=b_v,  rmse=r_v),
        ]

        ds_e.close()

    # ---- Save CSV
    df = pd.DataFrame(rows).sort_values(["var", "model", "lead_h"])
    csv_path = outdir / "metrics_bias_rmse.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")

    # ---- Plotting
    def plot_var(var, ylabel, fname):
        d = df[df["var"] == var].copy()
        fig = plt.figure(figsize=(9, 5), dpi=200)
        ax = plt.gca()

        for model in ["DINO", "FULL"]:
            dd = d[d["model"] == model].sort_values("lead_h")
            ax.plot(dd["lead_h"], dd["rmse"], marker="o", label=f"{model} RMSE")

        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(f"RMSE ({ylabel})")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        for model in ["DINO", "FULL"]:
            dd = d[d["model"] == model].sort_values("lead_h")
            ax2.plot(dd["lead_h"], dd["bias"], marker="s", linestyle="--", label=f"{model} bias")

        ax2.set_ylabel(f"Bias ({ylabel})")

        # Combine legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", frameon=True)

        fig.tight_layout()
        out = outdir / fname
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote: {out}")

    plot_var("PS", "hPa", "PS_hPa.png")
    plot_var("T", "K", "T_K.png")
    plot_var("V", "m/s", "Vmag_ms.png")

    print("\nDone.")
    if not args.no_weights:
        print("Note: metrics are cosine(latitude)-weighted global means. Use --no_weights to disable.")


if __name__ == "__main__":
    main()
