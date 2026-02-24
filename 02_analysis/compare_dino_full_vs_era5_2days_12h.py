import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import gc


# ----------------------------
# PATHS (tes chemins)
# ----------------------------
FULL_PATH = "/home/gmathieu/links/scratch/jaxgcm_outputs/pred_ds_jax_gcm_full_physics_2days_12h.nc"
DINO_PATH = "/home/gmathieu/links/scratch/dinosaur_outputs/ds_out_dino_2days_12h.nc"
ERA_DIR   = Path("/home/gmathieu/links/scratch/era5_snapshot")

# ERA5 vérité: 2 jours, pas 12h
ERA_T0_STR = "19900501T00"
ERA_TIMES = ["19900501T00", "19900501T12", "19900502T00", "19900502T12", "19900503T00"]
LEAD_HOURS = [0, 12, 24, 36, 48]  # doit matcher l’ordre ci-dessus

OUTDIR = Path("/home/gmathieu/links/scratch/results_compare")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTCSV = OUTDIR / "compare_dino_full_vs_era5_2days_12h.csv"

# ----------------------------
# Helpers
# ----------------------------
def open_any(path: str) -> xr.Dataset:
    for eng in ["h5netcdf", "netcdf4", "scipy"]:
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception:
            pass
    raise RuntimeError(f"Could not open {path} with h5netcdf/netcdf4/scipy")

def pick_lowest(da: xr.DataArray) -> xr.DataArray:
    for lev in ["hybrid", "level", "model_level", "lev", "sigma"]:
        if lev in da.dims:
            return da.isel({lev: -1})
    return da

def to_hpa(ps_pa: xr.DataArray) -> xr.DataArray:
    return ps_pa / 100.0

def find_latlon_names(obj) -> tuple[str, str]:
    # returns (lat_name, lon_name) as they appear in coords/dims
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]
    lat = next((n for n in lat_candidates if (n in obj.coords or n in obj.dims)), None)
    lon = next((n for n in lon_candidates if (n in obj.coords or n in obj.dims)), None)
    if lat is None or lon is None:
        raise ValueError(f"Could not find lat/lon names. dims={obj.dims} coords={list(obj.coords)}")
    return lat, lon

def get_grid_1d(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, str, str]:
    latn, lonn = find_latlon_names(ds)
    lat = np.asarray(ds[latn].values)
    lon = np.asarray(ds[lonn].values)

    # If lon in 0..360 and we want -180..180, keep target convention as-is.
    # We'll adapt ERA lon to match target lon convention later.
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Curvilinear grid (2D lat/lon) not supported by this simple interp().")

    # Ensure lat increasing
    if lat[0] > lat[-1]:
        lat = lat[::-1]

    return lon, lat, lonn, latn

def regrid2d_era_to_target(da: xr.DataArray, target_lon: np.ndarray, target_lat: np.ndarray) -> xr.DataArray:
    # normalize names to lon/lat for interp
    lat_name, lon_name = find_latlon_names(da)
    da = da.drop_vars(["lon", "lat"], errors="ignore")

    rename = {}
    if lon_name == "longitude":
        rename["longitude"] = "lon"
    elif lon_name == "lon":
        pass
    else:
        rename[lon_name] = "lon"

    if lat_name == "latitude":
        rename["latitude"] = "lat"
    elif lat_name == "lat":
        pass
    else:
        rename[lat_name] = "lat"

    da = da.rename(rename)

    # lat increasing
    if da["lat"][0].values > da["lat"][-1].values:
        da = da.sortby("lat")

    # Match lon convention to target
    tgt_lon = np.asarray(target_lon)
    target_is_180 = float(np.nanmin(tgt_lon)) < 0.0

    src_lon = da["lon"].values
    if target_is_180:
        new_lon = ((src_lon + 180.0) % 360.0) - 180.0
    else:
        new_lon = src_lon % 360.0

    if not np.allclose(new_lon, src_lon):
        da = da.assign_coords(lon=new_lon).sortby("lon")

    out = da.interp(
        lon=xr.DataArray(tgt_lon, dims=("lon",), name="lon"),
        lat=xr.DataArray(np.asarray(target_lat), dims=("lat",), name="lat"),
        method="linear",
    )
    return out

def rmse_bias(p: xr.DataArray, e: xr.DataArray) -> tuple[float, float]:
    p, e = xr.align(p, e, join="exact")
    d = p - e
    return float(np.sqrt((d**2).mean())), float(d.mean())

def first_existing(ds: xr.Dataset, names):
    for n in names:
        if n in ds:
            return ds[n]
    return None

def get_pred_vars(pred_t: xr.Dataset):
    # T
    T = first_existing(pred_t, ["temperature", "T"])
    if T is None:
        raise KeyError(f"Missing temperature. Vars={list(pred_t.data_vars)}")
    T = pick_lowest(T)

    # u/v
    u = first_existing(pred_t, ["u_wind", "u", "u_component_of_wind"])
    v = first_existing(pred_t, ["v_wind", "v", "v_component_of_wind"])
    if u is None or v is None:
        raise KeyError(f"Missing u/v. Vars={list(pred_t.data_vars)}")
    u = pick_lowest(u)
    v = pick_lowest(v)

    return T, u, v


def get_ps_hpa(pred_t: xr.Dataset, era_ps_hpa_on_same_grid: xr.DataArray, p0_pa_fixed: float | None):
    # Prefer direct surface_pressure if present
    if "surface_pressure" in pred_t:
        return to_hpa(pred_t["surface_pressure"]), p0_pa_fixed

    # Else normalized_surface_pressure (dimensionless)
    if "normalized_surface_pressure" in pred_t:
        psn = pred_t["normalized_surface_pressure"]

        if p0_pa_fixed is None:
            # Estimate p0 from t0 mean-match (once)
            p0_pa_fixed = float((era_ps_hpa_on_same_grid.mean() * 100.0) / psn.mean())

        ps_hpa = psn * (p0_pa_fixed / 100.0)
        return ps_hpa, p0_pa_fixed

    raise KeyError(f"No surface pressure found. Vars={list(pred_t.data_vars)}")

import numpy as np

def wind_speed(u, v):
    return np.sqrt(u**2 + v**2)

# ----------------------------
# Load datasets
# ----------------------------
full = open_any(FULL_PATH)
dino = open_any(DINO_PATH)

# grids (we regrid ERA separately to each)
full_lon, full_lat, full_lonn, full_latn = get_grid_1d(full)
dino_lon, dino_lat, dino_lonn, dino_latn = get_grid_1d(dino)

# We will index by lead position (0..4)
def get_time_slice(ds: xr.Dataset, i: int) -> xr.Dataset:
    if "time" in ds.dims:
        return ds.isel(time=i)
    return ds

# For normalized_surface_pressure, keep p0 estimated at t0 for each model
p0_full_pa = None
p0_dino_pa = None

rows = []

for i, (tstr, lead_h) in enumerate(zip(ERA_TIMES, LEAD_HOURS)):
    era_path = ERA_DIR / f"era5_{tstr}.nc"
    if not era_path.exists():
        raise FileNotFoundError(f"Missing ERA snapshot: {era_path}")

    era = xr.open_dataset(era_path, engine="scipy")

    # ERA vars (native grid)
    era_ps_native = era["surface_pressure"]                    # Pa
    era_T_native  = pick_lowest(era["temperature"])           # K
    era_u_native  = pick_lowest(era["u_component_of_wind"])   # m/s
    era_v_native  = pick_lowest(era["v_component_of_wind"])   # m/s

    # --- FULL: regrid ERA -> FULL grid ---
    era_ps_full = regrid2d_era_to_target(era_ps_native, full_lon, full_lat)
    era_T_full  = regrid2d_era_to_target(era_T_native,  full_lon, full_lat)
    era_u_full  = regrid2d_era_to_target(era_u_native,  full_lon, full_lat)
    era_v_full  = regrid2d_era_to_target(era_v_native,  full_lon, full_lat)

    # --- DINO: regrid ERA -> DINO grid ---
    era_ps_dino = regrid2d_era_to_target(era_ps_native, dino_lon, dino_lat)
    era_T_dino  = regrid2d_era_to_target(era_T_native,  dino_lon, dino_lat)
    era_u_dino  = regrid2d_era_to_target(era_u_native,  dino_lon, dino_lat)
    era_v_dino  = regrid2d_era_to_target(era_v_native,  dino_lon, dino_lat)

    # Convert ERA ps to hPa (on each grid)
    era_ps_full_hpa = to_hpa(era_ps_full)
    era_ps_dino_hpa = to_hpa(era_ps_dino)


    # model slices
    full_t = get_time_slice(full, i)
    dino_t = get_time_slice(dino, i)


    # DINO pred vars
    T_d, u_d, v_d = get_pred_vars(dino_t)
    ps_d_hpa, p0_dino_pa = get_ps_hpa(dino_t, era_ps_dino_hpa, p0_dino_pa)

    # FULL pred vars
    T_f, u_f, v_f = get_pred_vars(full_t)
    ps_f_hpa, p0_full_pa = get_ps_hpa(full_t, era_ps_full_hpa, p0_full_pa)


    # Align coords naming for exact align: ensure ps grids share same coord names as ERA grids
    # FULL: rename model coords to lon/lat if needed
    def normalize_model_latlon(da: xr.DataArray, latn: str, lonn: str) -> xr.DataArray:
        rename = {}
        if lonn in da.dims and lonn != "lon":
            rename[lonn] = "lon"
        if latn in da.dims and latn != "lat":
            rename[latn] = "lat"
        if rename:
            da = da.rename(rename)
        return da


    # FULL normalize
    ps_f_hpa_n  = normalize_model_latlon(ps_f_hpa, full_latn, full_lonn)
    T_f_n       = normalize_model_latlon(T_f,      full_latn, full_lonn)
    u_f_n       = normalize_model_latlon(u_f,      full_latn, full_lonn)
    v_f_n       = normalize_model_latlon(v_f,      full_latn, full_lonn)

    era_ps_full_hpa_n = era_ps_full_hpa
    era_T_full_n = era_T_full
    era_u_full_n = era_u_full
    era_v_full_n = era_v_full

    # DINO normalize
    ps_d_hpa_n  = normalize_model_latlon(ps_d_hpa, dino_latn, dino_lonn)
    T_d_n       = normalize_model_latlon(T_d,      dino_latn, dino_lonn)
    u_d_n       = normalize_model_latlon(u_d,      dino_latn, dino_lonn)
    v_d_n       = normalize_model_latlon(v_d,      dino_latn, dino_lonn)

    era_ps_dino_hpa_n = era_ps_dino_hpa
    era_T_dino_n = era_T_dino
    era_u_dino_n = era_u_dino
    era_v_dino_n = era_v_dino

        # ---- FULL wind speed ----
    V_full = wind_speed(u_f, v_f)
    V_era_full = wind_speed(era_u_full_n, era_v_full_n)
    
    rmse_V_full, bias_V_full = rmse_bias(V_full, V_era_full)
    
    # ---- DINO wind speed ----
    V_dino = wind_speed(u_d, v_d)
    V_era_dino = wind_speed(era_u_dino_n, era_v_dino_n)
    
    rmse_V_dino, bias_V_dino = rmse_bias(V_dino, V_era_dino)

    # Metrics FULL
    f_ps_rmse, f_ps_bias = rmse_bias(ps_f_hpa_n, era_ps_full_hpa_n)
    f_T_rmse,  f_T_bias  = rmse_bias(T_f_n,      era_T_full_n)
    f_u_rmse,  f_u_bias  = rmse_bias(u_f_n,      era_u_full_n)
    f_v_rmse,  f_v_bias  = rmse_bias(v_f_n,      era_v_full_n)

    # Metrics DINO
    d_ps_rmse, d_ps_bias = rmse_bias(ps_d_hpa_n, era_ps_dino_hpa_n)
    d_T_rmse,  d_T_bias  = rmse_bias(T_d_n,      era_T_dino_n)
    d_u_rmse,  d_u_bias  = rmse_bias(u_d_n,      era_u_dino_n)
    d_v_rmse,  d_v_bias  = rmse_bias(v_d_n,      era_v_dino_n)

    
    # ======================
    # Wind speed |V|
    # ======================
    
    # FULL
    V_full = wind_speed(u_f, v_f)
    V_era_full = wind_speed(era_u_full_n, era_v_full_n)
    rmse_V_full, bias_V_full = rmse_bias(V_full, V_era_full)
    
    # DINO
    V_dino = wind_speed(u_d, v_d)
    V_era_dino = wind_speed(era_u_dino_n, era_v_dino_n)
    rmse_V_dino, bias_V_dino = rmse_bias(V_dino, V_era_dino)

    rows.append({
    
        "era_time": tstr,
        "lead_hours": lead_h,
    
        # ======================
        # FULL
        # ======================
        "full_ps_rmse_hPa": f_ps_rmse, "full_ps_bias_hPa": f_ps_bias,
        "full_T_rmse_K": f_T_rmse, "full_T_bias_K": f_T_bias,
        "full_u_rmse_ms": f_u_rmse, "full_u_bias_ms": f_u_bias,
        "full_v_rmse_ms": f_v_rmse, "full_v_bias_ms": f_v_bias,
    
        # NEW: wind magnitude FULL
        "full_V_rmse_ms": f_V_rmse,
        "full_V_bias_ms": f_V_bias,
    
        # ======================
        # DINO
        # ======================
        "dino_ps_rmse_hPa": d_ps_rmse, "dino_ps_bias_hPa": d_ps_bias,
        "dino_T_rmse_K": d_T_rmse, "dino_T_bias_K": d_T_bias,
        "dino_u_rmse_ms": d_u_rmse, "dino_u_bias_ms": d_u_bias,
        "dino_v_rmse_ms": d_v_rmse, "dino_v_bias_ms": d_v_bias,
    
        # NEW: wind magnitude DINO
        "dino_V_rmse_ms": d_V_rmse,
        "dino_V_bias_ms": d_V_bias,
    
        # ======================
        # p0 info
        # ======================
        "p0_full_hPa_used": (p0_full_pa / 100.0) if p0_full_pa is not None else np.nan,
        "p0_dino_hPa_used": (p0_dino_pa / 100.0) if p0_dino_pa is not None else np.nan,
    })

    # free memory aggressively
    del era, era_ps_native, era_T_native, era_u_native, era_v_native
    del era_ps_full, era_T_full, era_u_full, era_v_full
    del era_ps_dino, era_T_dino, era_u_dino, era_v_dino
    del era_ps_full_hpa, era_ps_dino_hpa
    del full_t, dino_t
    del T_f, u_f, v_f, T_d, u_d, v_d
    del ps_f_hpa, ps_d_hpa
    try:
        del u_f_n, v_f_n, u_d_n, v_d_n
        del era_u_full_n, era_v_full_n, era_u_dino_n, era_v_dino_n
    except NameError:
        pass
    gc.collect()

df = pd.DataFrame(rows)
df.to_csv(OUTCSV, index=False)
print("WROTE", OUTCSV)
print(df)
