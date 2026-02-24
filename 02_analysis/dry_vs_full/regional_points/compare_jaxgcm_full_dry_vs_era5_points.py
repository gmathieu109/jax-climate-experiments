import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

COLORS = {
    "ERA5": "black",
    "dino-dry": "green",
    "dino-full": "orange",
}

UNITS = {
    "T": "K",
    "U": "m/s",
    "V": "m/s",
    "PS": "hPa",
}


FULL="/scratch/gmathieu/pred_dino_full_48h_6h.nc"
DRY ="/scratch/gmathieu/pred_dino_only_48h_6h.nc"   # rename if needed
ERA_DIR="/scratch/gmathieu/era5_snapshots_48h_6h_1990"  # pass list via --era5

def nearest_idx(arr, val):
    arr = np.asarray(arr)
    return int(np.argmin(np.abs(arr - val)))

def normalize_lon_0360(lon_deg):
    return lon_deg % 360.0

def model_ps_hpa(ds_slice):
    # ds_slice: one time slice Dataset from model output
    sp_norm = ds_slice["normalized_surface_pressure"]
    p0 = float(ds_slice.attrs.get("p0", 1e5))
    ps_pa = sp_norm * p0
    return (ps_pa / 100.0)  # hPa

def pick_era_vars(ds):
    # From your IC snapshots
    # coords: latitude, longitude, hybrid (and scalar time)
    sp = "surface_pressure" if "surface_pressure" in ds.data_vars else None
    t  = "temperature" if "temperature" in ds.data_vars else None
    u  = "u_component_of_wind" if "u_component_of_wind" in ds.data_vars else None
    v  = "v_component_of_wind" if "v_component_of_wind" in ds.data_vars else None
    if None in (sp,t,u,v):
        raise KeyError(f"ERA vars missing. have={list(ds.data_vars)[:50]}")
    return sp,t,u,v

def load_era_file(f):
    ds = xr.open_dataset(f)
    # ensure scalar time
    if "time" in ds.dims:
        ds = ds.isel(time=0)
    return ds

def box_slices(i, n, r):
    """Return slice indices [i-r, i+r] clipped to [0, n-1]."""
    i0 = max(0, i - r)
    i1 = min(n - 1, i + r)
    return slice(i0, i1 + 1)

import numpy as np
import xarray as xr

G = 9.80665

def standardize_era_lon(ds_e):
    """Force ERA longitude to [0,360) and strictly increasing for interp."""
    if "longitude" not in ds_e.coords:
        return ds_e
    lon = ds_e["longitude"]
    lonv = lon.values
    # if any negative lon, convert
    if np.nanmin(lonv) < 0:
        ds_e = ds_e.assign_coords(longitude=((lon + 360) % 360))
    # ensure sorted increasing
    if lon.ndim == 1:
        v = ds_e["longitude"].values
        if len(v) > 2 and not np.all(np.diff(v) > 0):
            ds_e = ds_e.sortby("longitude")
    return ds_e

def detect_land_ocean_coast(ds_e, lat2, lon2):
    """
    Try to infer land/ocean/coast from an orography-like variable.
    Returns (label, details_str). If unknown, returns ("UNKNOWN", "...").
    """
    # common candidates in ERA-ish files
    candidates = [
        "geopotential_at_surface",  # JAX-GCM/ARCO style
        "z",                        # ECMWF geopotential
        "orography",                # sometimes already in meters
        "surface_geopotential",
        "geopotential",
    ]
    var = None
    for c in candidates:
        if c in ds_e:
            var = c
            break
    if var is None:
        return "UNKNOWN", "no orography var found"

    oro = ds_e[var]

    # If it has time dim, take first
    for td in ["time", "valid_time"]:
        if td in oro.dims and oro.sizes.get(td, 1) > 1:
            oro = oro.isel({td: 0})
        elif td in oro.dims and oro.sizes.get(td, 1) == 1:
            oro = oro.isel({td: 0})

    # Interp on the same target 2D points
    oro2 = oro.interp(latitude=lat2, longitude=lon2)

    vals = oro2.values
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "UNKNOWN", f"{var} interpolated all-NaN"

    # Convert to meters if geopotential-looking
    units = str(oro.attrs.get("units", "")).lower()
    # heuristic: if values look huge (order 1e4+), likely m^2/s^2
    if ("m2" in units and "s-2" in units) or (np.nanmedian(np.abs(vals)) > 1e3):
        h = vals / G
    else:
        h = vals

    # classify by "near sea-level" threshold
    thr = 50.0  # meters
    frac_ocean = float(np.mean(h < thr))
    hmin = float(np.min(h))
    hmax = float(np.max(h))

    if frac_ocean > 0.8:
        return "OCEAN", f"{var}: frac_ocean={frac_ocean:.2f}, h[min,max]=[{hmin:.1f},{hmax:.1f}] m"
    if frac_ocean < 0.2:
        return "LAND", f"{var}: frac_ocean={frac_ocean:.2f}, h[min,max]=[{hmin:.1f},{hmax:.1f}] m"
    return "COAST/MIXED", f"{var}: frac_ocean={frac_ocean:.2f}, h[min,max]=[{hmin:.1f},{hmax:.1f}] m"

def compute_point_series(ds_model, ds_era_files, point, level0=0, era_hybrid=-10, remove_initial_bias=True, box_r=0):
    """
    Point-only: interpolate ERA5 only at the target point (lat,lon), not full grid.
    """
    # Model nearest gridpoint
    lats_m = ds_model["lat"].values
    lons_m = ds_model["lon"].values
    lon0360 = normalize_lon_0360( point["lon"])
    ilat = nearest_idx(lats_m, point["lat"])
    ilon = nearest_idx(lons_m, lon0360)
    
    # box around nearest gridpoint (model grid)
    slat = box_slices(ilat, len(lats_m), box_r)
    slon = box_slices(ilon, len(lons_m), box_r)

    # box grid coords (1D)
    box_lats = ds_model["lat"].isel(lat=slat).values
    box_lons = ds_model["lon"].isel(lon=slon).values

    # For reporting: central gridpoint coords (as before)
    plat = float(lats_m[ilat]); plon = float(lons_m[ilon])

    # model time axis
    mt = pd.to_datetime(ds_model["time"].values)
    t0_model = mt[0]

    # open first ERA to get varnames
    ds_e0 = load_era_file(ds_era_files[0])
    sp_era, t_era, u_era, v_era = pick_era_vars(ds_e0)
    ds_e0.close()

    rows = []
        # Build 2D target grid once (lat_box x lon_box)
    targ_lat_1d = xr.DataArray(box_lats, dims=("lat_box",))
    targ_lon_1d = xr.DataArray(box_lons, dims=("lon_box",))
    lat2, lon2 = xr.broadcast(targ_lat_1d, targ_lon_1d)

    printed_loc = False

    for k, f in enumerate(ds_era_files):
        ds_e = load_era_file(f).sortby("latitude")
        ds_e = standardize_era_lon(ds_e)

        if (not printed_loc) and k == 0:
            label, details = detect_land_ocean_coast(ds_e, lat2, lon2)
            boxN = 2*box_r + 1
            print(f"[{point['name']}] grid center lat={plat:.3f} lon={plon:.3f} (0-360), box={boxN}x{boxN} -> {label} ({details})")
            printed_loc = True

        # ---- ERA5: interpolate onto MODEL box points, then average ----
        Te_box  = ds_e[t_era].isel(hybrid=era_hybrid).interp(latitude=lat2, longitude=lon2)
        Ue_box  = ds_e[u_era].isel(hybrid=era_hybrid).interp(latitude=lat2, longitude=lon2)
        Ve_box  = ds_e[v_era].isel(hybrid=era_hybrid).interp(latitude=lat2, longitude=lon2)
        PSe_box = ds_e["surface_pressure"].interp(latitude=lat2, longitude=lon2) / 100.0  # hPa

        Te  = float(Te_box.mean(dim=("lat_box","lon_box"), skipna=True).values)
        Ue  = float(Ue_box.mean(dim=("lat_box","lon_box"), skipna=True).values)
        Ve  = float(Ve_box.mean(dim=("lat_box","lon_box"), skipna=True).values)
        PSe = float(PSe_box.mean(dim=("lat_box","lon_box"), skipna=True).values)

        # ---- MODEL: average over the same box on model grid ----
        Tm  = float(ds_model["temperature"].isel(time=k, level=level0, lat=slat, lon=slon).mean().values)
        Um  = float(ds_model["u_wind"].isel(time=k, level=level0, lat=slat, lon=slon).mean().values)
        Vm  = float(ds_model["v_wind"].isel(time=k, level=level0, lat=slat, lon=slon).mean().values)
        PSm = float(model_ps_hpa(ds_model.isel(time=k)).isel(lat=slat, lon=slon).mean().values)

        tt = mt[k]
        lead_h = float((tt - t0_model) / pd.Timedelta(hours=1))

        rows += [
            dict(point=point["name"], grid_lat=plat, grid_lon=plon, time=str(tt), lead_h=lead_h, var="T",  model=Tm,  era=Te),
            dict(point=point["name"], grid_lat=plat, grid_lon=plon, time=str(tt), lead_h=lead_h, var="U",  model=Um,  era=Ue),
            dict(point=point["name"], grid_lat=plat, grid_lon=plon, time=str(tt), lead_h=lead_h, var="V",  model=Vm,  era=Ve),
            dict(point=point["name"], grid_lat=plat, grid_lon=plon, time=str(tt), lead_h=lead_h, var="PS", model=PSm, era=PSe),
        ]

        ds_e.close()

    df = pd.DataFrame(rows).sort_values(["var","lead_h"]).reset_index(drop=True)
    df["err"] = df["model"] - df["era"]

    if remove_initial_bias:
        df["err0"] = df.groupby(["point","var"])["err"].transform("first")
        df["err_anom"] = df["err"] - df["err0"]
    else:
        df["err_anom"] = df["err"]

    return df


def plot_timeseries(df, var, outpath, title, ylabel):
    d = df[df["var"]==var].sort_values("lead_h")
    fig = plt.figure(figsize=(9,5), dpi=200)
    ax = plt.gca()
    ax.set_title(title, fontsize=14, pad=12)
    ax.plot(d["lead_h"], d["era"], marker="o", label="ERA5")
    ax.plot(d["lead_h"], d["model_full"], marker="o", label="FULL")
    ax.plot(d["lead_h"], d["model_dry"], marker="o", label="DRY")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_bias(df, var, outpath, title):
    d = df[df["var"]==var].sort_values("lead_h")
    yunit = UNITS.get(var, "")
    fig = plt.figure(figsize=(9,5), dpi=200)
    ax = plt.gca()
    ax.set_title(title, fontsize=14, pad=12)

    ax.plot(d["lead_h"], d["bias_dry"],  color=COLORS["dino-dry"], marker="o", label="dino-dry bias (t0 removed)")
    ax.plot(d["lead_h"], d["bias_full"], color=COLORS["dino-full"], marker="o", label="dino-full bias (t0 removed)")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"Bias ({yunit})" if yunit else "Bias")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_rmse(df, var, outpath, title):
    d = df[df["var"]==var].sort_values("lead_h")
    yunit = UNITS.get(var, "")
    fig = plt.figure(figsize=(9,5), dpi=200)
    ax = plt.gca()
    ax.set_title(title, fontsize=14, pad=12)

    ax.plot(d["lead_h"], d["rmse_dry"],  color=COLORS["dino-dry"], marker="o", label="dino-dry RMSE (point)")
    ax.plot(d["lead_h"], d["rmse_full"], color=COLORS["dino-full"], marker="o", label="dino-full RMSE (point)")

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"RMSE ({yunit})" if yunit else "RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_error(df, var, outpath, title, ylabel):
    d = df[df["var"]==var].sort_values("lead_h")
    fig = plt.figure(figsize=(9,5), dpi=200)
    ax = plt.gca()
    ax.set_title(title, fontsize=14, pad=12)
    ax.plot(d["lead_h"], d["err_full"], marker="o", label="FULL error (bias removed)" )
    ax.plot(d["lead_h"], d["err_dry"],  marker="o", label="DRY error (bias removed)" )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_values(df, var, outpath, title):
    d = df[df["var"]==var].sort_values("lead_h")
    yunit = UNITS.get(var, "")
    fig = plt.figure(figsize=(9,5), dpi=200)
    ax = plt.gca()
    ax.set_title(title, fontsize=14, pad=12)

    ax.plot(d["lead_h"], d["val_era"], color=COLORS["ERA5"], marker="o", label="ERA5")
    ax.plot(d["lead_h"], d["val_dry"], color=COLORS["dino-dry"], marker="o", label="dino-dry")
    ax.plot(d["lead_h"], d["val_full"], color=COLORS["dino-full"], marker="o", label="dino-full")

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"{var} ({yunit})" if yunit else var)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=FULL)
    ap.add_argument("--dry",  default=DRY)
    ap.add_argument("--era5", nargs="+", required=True)
    ap.add_argument("--outdir", default="./compare_out_points")
    ap.add_argument("--point", default="MTL", choices=["MTL","TROP"])
    ap.add_argument("--remove_initial_bias", action="store_true")
    ap.add_argument("--box", type=int, default=0,
                help="half-size of box mean (0=point, 1=3x3, 2=5x5, ...)")
    ap.add_argument("--era_hybrid", type=int, default=-10,
                help="ERA5 hybrid index to use for T/U/V (e.g., -10)")
    ap.add_argument("--level0", type=int, default=0,
                help="Model level index to use (e.g., 0 or 1)")
    args = ap.parse_args()

    box_r = args.box
    boxN = 2*box_r + 1
    box_tag   = f"box{boxN}x{boxN}"
    map_tag = f"L{args.level0}_H{args.era_hybrid}"
    box_title = f" ({boxN}x{boxN} box mean)" if box_r > 0 else " (point)"

    print(f"Using box_r={box_r} -> {box_tag}")

    # points
    points = {
        "MTL":  dict(name="MTL",  lat=45.5, lon=-73.6),
        "TROP": dict(name="TROP", lat=15.0, lon=-45.0),
    }
    pt = points[args.point]

    
    outdir = Path(args.outdir) / f"{pt['name']}_{box_tag}_{map_tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Using mapping: level0={args.level0}, era_hybrid={args.era_hybrid}")
    
    # open models
    ds_full = xr.open_dataset(args.full)
    ds_dry  = xr.open_dataset(args.dry)

    # compute series (bias removed optional)
    df_full = compute_point_series(
        ds_full, args.era5, pt,
        level0=args.level0,
        era_hybrid=args.era_hybrid,
        remove_initial_bias=args.remove_initial_bias,
        box_r=box_r
    )

    df_dry = compute_point_series(
        ds_dry, args.era5, pt,
        level0=args.level0,
        era_hybrid=args.era_hybrid,
        remove_initial_bias=args.remove_initial_bias,
        box_r=box_r
    )
    # --- Merge FULL / DRY / ERA values ---

    df = df_full.rename(columns={
        "model": "val_full",
        "err_anom": "bias_full"
    })[["point","grid_lat","grid_lon","time","lead_h","var","era","val_full","bias_full"]]

    df = df.merge(
        df_dry.rename(columns={
            "model": "val_dry",
            "err_anom": "bias_dry"
        })[["time","lead_h","var","val_dry","bias_dry"]],
        on=["time","lead_h","var"],
        how="inner"
    )

    # rename ERA values
    df = df.rename(columns={"era":"val_era"})

    # point "RMSE" = magnitude of bias (bias already initial-bias removed)
    df["rmse_full"] = np.abs(df["bias_full"])
    df["rmse_dry"]  = np.abs(df["bias_dry"])

    # write CSV
    df["box_r"] = box_r
    df["box_tag"] = box_tag
    csv_path = outdir / f"point_{pt['name']}_series_{box_tag}.csv"
    df.to_csv(csv_path, index=False)
    print("Wrote:", csv_path)

    # 9 figures: (T,U,V) × (timeseries, error, abs_error)
    tag = "bias_removed" if args.remove_initial_bias else "raw"

    for var in ["PS","T","U","V"]:
        plot_values(df, var, outdir/f"{pt['name']}_{var}_VALUES_{tag}_{box_tag}_{map_tag}.png",
                    title=f"{pt['name']} {var} – values (ERA5 vs dino-dry vs dino-full)")
        plot_bias(df, var, outdir/f"{pt['name']}_{var}_BIAS_{tag}_{box_tag}_{map_tag}.png",
                title=f"{pt['name']} {var} – bias vs ERA5 (t0 removed)" if args.remove_initial_bias else f"{pt['name']} {var} – bias vs ERA5")
        plot_rmse(df, var, outdir/f"{pt['name']}_{var}_RMSE_{tag}_{box_tag}_{map_tag}.png",
                title=f"{pt['name']} {var} – RMSE (point) vs ERA5")


    print("Wrote figures in:", outdir)

if __name__ == "__main__":
    main()
