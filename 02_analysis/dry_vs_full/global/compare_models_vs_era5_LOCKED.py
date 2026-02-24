#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

FULL="/home/gmathieu/links/scratch/jaxgcm_outputs/pred_ds_jax_gcm_full_physics_2days_12h.nc"
DINO="/home/gmathieu/links/scratch/dinosaur_outputs/ds_out_dino_2days_12h.nc"

def sp_to_hpa(sp):
    units = str(sp.attrs.get("units","")).lower()
    if "pa" in units:
        return sp/100.0
    m = float(sp.mean().values)
    return sp/100.0 if m > 2000.0 else sp

def reconstruct_full_ps_hpa(ds_full_slice):
    sp_norm = ds_full_slice["normalized_surface_pressure"]
    p0 = float(ds_full_slice.attrs.get("p0", 1e5))
    ps_pa = sp_norm * p0
    ps_pa.attrs["units"] = "Pa"
    return sp_to_hpa(ps_pa)

def vmag(u,v):
    return xr.apply_ufunc(lambda a,b: np.sqrt(a*a+b*b), u, v, dask="allowed", output_dtypes=[np.float32])

def normalize_lon(lon, target_range):
    if target_range == "[-180,180)":
        return ((lon + 180) % 360) - 180
    if target_range == "[0,360)":
        return lon % 360
    raise ValueError

def detect_lon_range(lon):
    mx = float(lon.max().values)
    return "[0,360)" if mx > 180.0 else "[-180,180)"

def maybe_sort_lon(ds, lon_name):
    lon = ds[lon_name]
    if lon.ndim == 1:
        vals = lon.values
        if len(vals) > 2 and not np.all(np.diff(vals) > 0):
            return ds.sortby(lon_name)
    return ds

def area_weights_coslat(lat_1d):
    w = np.cos(np.deg2rad(lat_1d))
    return xr.DataArray(w, coords={lat_1d.name: lat_1d}, dims=(lat_1d.name,))

def weighted_mean_2d(da2d, lat, lon, wlat):
    w2 = wlat / wlat.mean()
    wb = w2.broadcast_like(da2d)
    num = (da2d*wb).sum(dim=(lat,lon), skipna=True)
    den = wb.where(np.isfinite(da2d)).sum(dim=(lat,lon), skipna=True)
    return num/den

def to_2d(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """
    Ensure da is 2D on (lat, lon) by dropping/reducing any extra dims safely.
    - If extra dims remain (e.g., level/sigma), we squeeze if size==1 else take last index.
      (Change policy here if you prefer mean over vertical.)
    """
    # If lat/lon are not dims but coords, xarray.interp usually makes them dims.
    # We handle both orders.
    if lat_name not in da.dims or lon_name not in da.dims:
        # Sometimes dims are (lon, lat) still fine; check and proceed
        pass

    # Reduce any remaining dims other than lat/lon
    extra_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
    for d in extra_dims:
        if da.sizes[d] == 1:
            da = da.isel({d: 0})
        else:
            # choose last level/sigma by default (near-surface convention in your script)
            da = da.isel({d: 0})

    # Now squeeze any leftover length-1 dims
    da = da.squeeze(drop=True)

    # Finally, make sure we have lat/lon dims; reorder if needed
    if lat_name in da.dims and lon_name in da.dims:
        da = da.transpose(lat_name, lon_name)
    return da


def bias_rmse(model: xr.DataArray, truth: xr.DataArray, lat: str, lon: str, wlat=None):
    model2 = to_2d(model, lat, lon)
    truth2 = to_2d(truth, lat, lon)

    diff = model2 - truth2

    if wlat is None:
        bias = diff.mean(dim=(lat, lon), skipna=True)
        rmse = np.sqrt((diff**2).mean(dim=(lat, lon), skipna=True))
    else:
        # weights depend on lat only
        w2 = wlat / wlat.mean()
        wb = w2.broadcast_like(diff)
        num = (diff * wb).sum(dim=(lat, lon), skipna=True)
        den = wb.where(np.isfinite(diff)).sum(dim=(lat, lon), skipna=True)
        bias = num / den

        num2 = ((diff**2) * wb).sum(dim=(lat, lon), skipna=True)
        den2 = wb.where(np.isfinite(diff)).sum(dim=(lat, lon), skipna=True)
        rmse = np.sqrt(num2 / den2)

    # bias/rmse should now be scalar DataArrays
    return float(bias.values), float(rmse.values)

def plot_rmse_only(df, var, ylabel, outpath):
    d = df[df["var"] == var].copy()
    fig = plt.figure(figsize=(9,5), dpi=200)
    ax = plt.gca()
    ax.set_title(f"{var} – RMSE vs ERA5 (2 days, 12h step)", fontsize=14, pad=12)
    for model in ["DINO","FULL"]:
        dd = d[d["model"]==model].sort_values("lead_h")
        ax.plot(dd["lead_h"], dd["rmse"], marker="o", label=model)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"RMSE ({ylabel})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_bias_only(df, var, ylabel, outpath):
    d = df[df["var"] == var].copy()
    fig = plt.figure(figsize=(9,5), dpi=200)
    ax = plt.gca()
    ax.set_title(f"{var} – Bias vs ERA5 (2 days, 12h step)", fontsize=14, pad=12)
    for model in ["DINO","FULL"]:
        dd = d[d["model"]==model].sort_values("lead_h")
        ax.plot(dd["lead_h"], dd["bias"], marker="o", label=model)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"Bias ({ylabel})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_means(dfm, var, ylabel, outpath, era_variant="ERA_on_FULL"):
    d = dfm[dfm["var"] == var].copy()
    fig = plt.figure(figsize=(9,5), dpi=200)
    ax = plt.gca()
    ax.set_title(
        f"{var} – Global Mean Evolution\nERA5 vs FULL vs DINO (0–48h, 12h step)",
        fontsize=14,
        pad=12
    )

    for model in [era_variant, "FULL", "DINO"]:
        dd = d[d["model"]==model].sort_values("lead_h")
        ax.plot(dd["lead_h"], dd["mean"], marker="o", label=model)

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"Global mean ({ylabel})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def regrid_era_to_model(era_da, era_lat, era_lon, model_lat, model_lon, target_lon_range):
    era_da = era_da.assign_coords({era_lon: normalize_lon(era_da[era_lon], target_lon_range)})
    if era_da[era_lon].ndim == 1 and not np.all(np.diff(era_da[era_lon].values) > 0):
        era_da = era_da.sortby(era_lon)
    return era_da.interp({era_lat: model_lat, era_lon: model_lon}, method="linear", kwargs={"fill_value": np.nan})

def plot_var(df, var, ylabel, outpath):
    d = df[df["var"] == var].copy()
    fig = plt.figure(figsize=(9,5), dpi=200)
    ax = plt.gca()
    for model in ["DINO","FULL"]:
        dd = d[d["model"]==model].sort_values("lead_h")
        ax.plot(dd["lead_h"], dd["rmse"], marker="o", label=f"{model} RMSE")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"RMSE ({ylabel})")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    for model in ["DINO","FULL"]:
        dd = d[d["model"]==model].sort_values("lead_h")
        ax2.plot(dd["lead_h"], dd["bias"], marker="s", linestyle="--", label=f"{model} bias")
    ax2.set_ylabel(f"Bias ({ylabel})")

    h1,l1 = ax.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=FULL)
    ap.add_argument("--dino", default=DINO)
    ap.add_argument("--era5", nargs="+", required=True)
    ap.add_argument("--outdir", default="./compare_out")
    ap.add_argument("--no_weights", action="store_true")
    ap.add_argument("--era_sp", default=None)
    ap.add_argument("--era_t",  default=None)
    ap.add_argument("--era_u",  default=None)
    ap.add_argument("--era_v",  default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ds_full = xr.open_dataset(args.full, chunks={"time": 1})
    ds_dino = xr.open_dataset(args.dino, chunks={"time": 1})

    # model grids / coords (known from your probe)
    lat_full, lon_full = "lat","lon"
    lat_dino, lon_dino = "latitude","longitude"
    ds_full = maybe_sort_lon(ds_full, lon_full)
    ds_dino = maybe_sort_lon(ds_dino, lon_dino)

    lon_range_full = detect_lon_range(ds_full[lon_full])
    lon_range_dino = detect_lon_range(ds_dino[lon_dino])

    w_full = None if args.no_weights else area_weights_coslat(ds_full[lat_full])
    w_dino = None if args.no_weights else area_weights_coslat(ds_dino[lat_dino])

    # ERA5: open first snapshot to detect coord + var names
    ds_e0 = xr.open_dataset(args.era5[0])
    # coords
    era_lat = "latitude" if "latitude" in ds_e0.coords else "lat"
    era_lon = "longitude" if "longitude" in ds_e0.coords else "lon"
    era_time = "time" if "time" in ds_e0.coords else list(ds_e0.coords)[0]

    # vars: allow overrides for big ERA snapshots
    sp_era = args.era_sp or ("surface_pressure" if "surface_pressure" in ds_e0.data_vars else ("sp" if "sp" in ds_e0.data_vars else ("ps" if "ps" in ds_e0.data_vars else None)))
    t_era  = args.era_t or (
    "t2m" if "t2m" in ds_e0.data_vars else
    ("2m_temperature" if "2m_temperature" in ds_e0.data_vars else
     ("t" if "t" in ds_e0.data_vars else
      ("temperature" if "temperature" in ds_e0.data_vars else None)))
    )

    u_era  = args.era_u or (
        "u10" if "u10" in ds_e0.data_vars else
        ("10m_u_component_of_wind" if "10m_u_component_of_wind" in ds_e0.data_vars else
         ("u" if "u" in ds_e0.data_vars else
          ("u_component_of_wind" if "u_component_of_wind" in ds_e0.data_vars else None)))
    )

    v_era  = args.era_v or (
        "v10" if "v10" in ds_e0.data_vars else
        ("10m_v_component_of_wind" if "10m_v_component_of_wind" in ds_e0.data_vars else
         ("v" if "v" in ds_e0.data_vars else
          ("v_component_of_wind" if "v_component_of_wind" in ds_e0.data_vars else None)))
    )



    if None in [sp_era, t_era, u_era, v_era]:
        print("ERA5 data_vars head:", sorted(ds_e0.data_vars)[:80])
        raise KeyError(f"Could not detect ERA vars: sp={sp_era}, t={t_era}, u={u_era}, v={v_era}. Use --era_sp/--era_t/--era_u/--era_v.")
    ds_e0.close()

    # times/leads from snapshots
    era_times = []
    for f in args.era5:
        ds = xr.open_dataset(f)
        tt = ds[era_time]
        if tt.size > 1:
            tt = tt.isel({era_time: 0})
        era_times.append(pd.Timestamp(tt.values))
        ds.close()
    t0 = era_times[0]
    lead_hours = [int((t - t0)/pd.Timedelta(hours=1)) for t in era_times]

    rows = []
    rows_means = []


    for k, (f_era, valid_time, lead) in enumerate(zip(args.era5, era_times, lead_hours)):
        print(f"[{k+1}/{len(args.era5)}] lead={lead:>3}h ERA={Path(f_era).name}")

        ds_e = xr.open_dataset(f_era)
        era_sp_da = ds_e[sp_era]; era_t_da = ds_e[t_era]; era_u_da = ds_e[u_era]; era_v_da = ds_e[v_era]


        if k == 0:
            print("ERA chosen vars:", sp_era, t_era, u_era, v_era)
            print("ERA T dims:", era_t_da.dims)
            print("ERA sizes:", {d: era_t_da.sizes[d] for d in era_t_da.dims})
        if era_time in era_sp_da.dims:
            era_sp_da = era_sp_da.isel({era_time:0}); era_t_da = era_t_da.isel({era_time:0}); era_u_da = era_u_da.isel({era_time:0}); era_v_da = era_v_da.isel({era_time:0})
        # ERA model-level fields -> pick lowest (near-sfc) hybrid level
        era_t_da = era_t_da.isel(hybrid=-1)
        era_u_da = era_u_da.isel(hybrid=-1)
        era_v_da = era_v_da.isel(hybrid=-1)

        
        era_sp_hpa = sp_to_hpa(era_sp_da)
        era_vmag = vmag(era_u_da, era_v_da)

        # FULL slice (known names from probe)
        full = ds_full.isel(time=k)
        full_ps_hpa = reconstruct_full_ps_hpa(full)
        full_t = full["temperature"].isel(level=0)
        full_vmag = vmag(full["u_wind"].isel(level=0), full["v_wind"].isel(level=0))


        era_ps_on_full = regrid_era_to_model(era_sp_hpa, era_lat, era_lon, ds_full[lat_full], ds_full[lon_full], lon_range_full)
        era_t_on_full  = regrid_era_to_model(era_t_da,   era_lat, era_lon, ds_full[lat_full], ds_full[lon_full], lon_range_full)
        era_v_on_full  = regrid_era_to_model(era_vmag,   era_lat, era_lon, ds_full[lat_full], ds_full[lon_full], lon_range_full)

        b_ps, r_ps = bias_rmse(full_ps_hpa, era_ps_on_full, lat_full, lon_full, w_full)
        b_t,  r_t  = bias_rmse(full_t,      era_t_on_full,  lat_full, lon_full, w_full)
        b_v,  r_v  = bias_rmse(full_vmag,   era_v_on_full,  lat_full, lon_full, w_full)

        rows += [
            dict(model="FULL", var="PS", units="hPa", lead_h=lead, valid_time=str(valid_time), bias=b_ps, rmse=r_ps),
            dict(model="FULL", var="T",  units="K",   lead_h=lead, valid_time=str(valid_time), bias=b_t,  rmse=r_t),
            dict(model="FULL", var="V",  units="m/s", lead_h=lead, valid_time=str(valid_time), bias=b_v,  rmse=r_v),
        ]

        # DINO slice
        dino = ds_dino.isel(time=k)
        dino_sp_hpa = sp_to_hpa(dino["surface_pressure"])
        dino_t = dino["temperature"].isel(sigma=-1)
        dino_vmag = vmag(dino["u_component_of_wind"].isel(sigma=-1), dino["v_component_of_wind"].isel(sigma=-1))


        era_ps_on_dino = regrid_era_to_model(era_sp_hpa, era_lat, era_lon, ds_dino[lat_dino], ds_dino[lon_dino], lon_range_dino)
        era_t_on_dino  = regrid_era_to_model(era_t_da,   era_lat, era_lon, ds_dino[lat_dino], ds_dino[lon_dino], lon_range_dino)
        era_v_on_dino  = regrid_era_to_model(era_vmag,   era_lat, era_lon, ds_dino[lat_dino], ds_dino[lon_dino], lon_range_dino)

                # ---- also store global means (same weighting as metrics)
        def global_mean(da2d, lat, lon, wlat):
            da2d = to_2d(da2d, lat, lon)
            if wlat is None:
                return float(da2d.mean(dim=(lat, lon), skipna=True).values)
            w2 = wlat / wlat.mean()
            wb = w2.broadcast_like(da2d)
            num = (da2d * wb).sum(dim=(lat, lon), skipna=True)
            den = wb.where(da2d.notnull()).sum(dim=(lat, lon), skipna=True)
            return float((num / den).values)

        # ERA means on each model grid (so means are comparable to what metrics saw)
        mean_era_ps_full = global_mean(era_ps_on_full, lat_full, lon_full, w_full)
        mean_era_t_full  = global_mean(era_t_on_full,  lat_full, lon_full, w_full)
        mean_era_v_full  = global_mean(era_v_on_full,  lat_full, lon_full, w_full)

        mean_full_ps = global_mean(full_ps_hpa, lat_full, lon_full, w_full)
        mean_full_t  = global_mean(full_t,      lat_full, lon_full, w_full)
        mean_full_v  = global_mean(full_vmag,   lat_full, lon_full, w_full)

        mean_era_ps_dino = global_mean(era_ps_on_dino, lat_dino, lon_dino, w_dino)
        mean_era_t_dino  = global_mean(era_t_on_dino,  lat_dino, lon_dino, w_dino)
        mean_era_v_dino  = global_mean(era_v_on_dino,  lat_dino, lon_dino, w_dino)

        mean_dino_ps = global_mean(dino_sp_hpa, lat_dino, lon_dino, w_dino)
        mean_dino_t  = global_mean(dino_t,      lat_dino, lon_dino, w_dino)
        mean_dino_v  = global_mean(dino_vmag,   lat_dino, lon_dino, w_dino)

        # Save "direct" means. We keep ERA on both grids (FULL-grid and DINO-grid) for consistency.
        # You can choose later which ERA series you want to show.
        rows_means += [
            dict(model="ERA_on_FULL", var="PS", units="hPa", lead_h=lead, valid_time=str(valid_time), mean=mean_era_ps_full),
            dict(model="ERA_on_FULL", var="T",  units="K",   lead_h=lead, valid_time=str(valid_time), mean=mean_era_t_full),
            dict(model="ERA_on_FULL", var="V",  units="m/s", lead_h=lead, valid_time=str(valid_time), mean=mean_era_v_full),

            dict(model="FULL", var="PS", units="hPa", lead_h=lead, valid_time=str(valid_time), mean=mean_full_ps),
            dict(model="FULL", var="T",  units="K",   lead_h=lead, valid_time=str(valid_time), mean=mean_full_t),
            dict(model="FULL", var="V",  units="m/s", lead_h=lead, valid_time=str(valid_time), mean=mean_full_v),

            dict(model="ERA_on_DINO", var="PS", units="hPa", lead_h=lead, valid_time=str(valid_time), mean=mean_era_ps_dino),
            dict(model="ERA_on_DINO", var="T",  units="K",   lead_h=lead, valid_time=str(valid_time), mean=mean_era_t_dino),
            dict(model="ERA_on_DINO", var="V",  units="m/s", lead_h=lead, valid_time=str(valid_time), mean=mean_era_v_dino),

            dict(model="DINO", var="PS", units="hPa", lead_h=lead, valid_time=str(valid_time), mean=mean_dino_ps),
            dict(model="DINO", var="T",  units="K",   lead_h=lead, valid_time=str(valid_time), mean=mean_dino_t),
            dict(model="DINO", var="V",  units="m/s", lead_h=lead, valid_time=str(valid_time), mean=mean_dino_v),
        ]


        b_ps, r_ps = bias_rmse(dino_sp_hpa, era_ps_on_dino, lat_dino, lon_dino, w_dino)
        b_t,  r_t  = bias_rmse(dino_t,      era_t_on_dino,  lat_dino, lon_dino, w_dino)
        b_v,  r_v  = bias_rmse(dino_vmag,   era_v_on_dino,  lat_dino, lon_dino, w_dino)

        rows += [
            dict(model="DINO", var="PS", units="hPa", lead_h=lead, valid_time=str(valid_time), bias=b_ps, rmse=r_ps),
            dict(model="DINO", var="T",  units="K",   lead_h=lead, valid_time=str(valid_time), bias=b_t,  rmse=r_t),
            dict(model="DINO", var="V",  units="m/s", lead_h=lead, valid_time=str(valid_time), bias=b_v,  rmse=r_v),
        ]

        ds_e.close()

    df = pd.DataFrame(rows).sort_values(["var","model","lead_h"])
    csv_path = outdir/"metrics_bias_rmse.csv"
    df.to_csv(csv_path, index=False)
    print("Wrote:", csv_path)

    dfm = pd.DataFrame(rows_means).sort_values(["var","model","lead_h"])
    means_path = outdir/"means_global.csv"
    dfm.to_csv(means_path, index=False)
    print("Wrote:", means_path)

    # RMSE-only figures
    plot_rmse_only(df, "PS", "hPa", outdir/"PS_RMSE.png")
    plot_rmse_only(df, "T",  "K",   outdir/"T_RMSE.png")
    plot_rmse_only(df, "V",  "m/s", outdir/"Vmag_RMSE.png")

    # Bias-only figures
    plot_bias_only(df, "PS", "hPa", outdir/"PS_BIAS.png")
    plot_bias_only(df, "T",  "K",   outdir/"T_BIAS.png")
    plot_bias_only(df, "V",  "m/s", outdir/"Vmag_BIAS.png")

    # Direct global-mean value figures (choose ERA on FULL grid by default)
    plot_means(dfm, "PS", "hPa", outdir/"PS_MEAN.png", era_variant="ERA_on_FULL")
    plot_means(dfm, "T",  "K",   outdir/"T_MEAN.png",  era_variant="ERA_on_FULL")
    plot_means(dfm, "V",  "m/s", outdir/"Vmag_MEAN.png", era_variant="ERA_on_FULL")


    plot_var(df, "PS", "hPa", outdir/"PS_hPa.png")
    plot_var(df, "T",  "K",   outdir/"T_K.png")
    plot_var(df, "V",  "m/s", outdir/"Vmag_ms.png")
    print("Wrote figures in:", outdir)

if __name__ == "__main__":
    main()
