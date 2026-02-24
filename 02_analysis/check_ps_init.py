import numpy as np
import xarray as xr

FULL="/home/gmathieu/src/jax-gcm/notebooks/pred_ds_full_48h_12h_19900501T00.nc"
DINO="/home/gmathieu/links/scratch/dinosaur_outputs/ds_out_dino_2days_12h.nc"
ERA0="/home/gmathieu/links/scratch/era5_snapshot/era5_19900501T00.nc"

# Si tu as un p0 "corrigé" (hPa) déjà établi, mets-le ici.
# Sinon on laisse 977.0 hPa comme tu l'avais mentionné.
P0_HPA = 977.0

def area_weights(lat_deg):
    lat = np.deg2rad(lat_deg)
    w = np.cos(lat)
    return w / w.mean()

def wmean_2d(da, lat_name, lon_name):
    # da dims: (lon, lat) ou (lat, lon)
    lat = da[lat_name]
    w = xr.DataArray(area_weights(lat), dims=(lat_name,), coords={lat_name: lat})
    out = (da * w).mean(dim=(lat_name, lon_name), skipna=True)
    return float(out.values)

def detect_latlon(ds):
    for lat in ["lat","latitude","y"]:
        if lat in ds.coords or lat in ds.variables:
            lat_name = lat
            break
    else:
        raise KeyError("No lat coord found")
    for lon in ["lon","longitude","x"]:
        if lon in ds.coords or lon in ds.variables:
            lon_name = lon
            break
    else:
        raise KeyError("No lon coord found")
    return lat_name, lon_name

def pick_surface_pressure_full(ds_full):
    # FULL: either normalized_surface_pressure (unitless) OR surface_pressure (Pa/hPa)
    if "surface_pressure" in ds_full:
        ps = ds_full["surface_pressure"]
        # convert to hPa if it looks like Pa
        if ps.max() > 2000:
            ps = ps / 100.0
        return ps
    if "normalized_surface_pressure" in ds_full:
        return ds_full["normalized_surface_pressure"] * P0_HPA
    raise KeyError("FULL: no surface_pressure or normalized_surface_pressure")

def pick_surface_pressure_era(ds_era):
    # Try common ERA5 names
    for name in ["surface_pressure","sp","mean_sea_level_pressure","msl"]:
        if name in ds_era:
            ps = ds_era[name]
            # Pa -> hPa
            if ps.max() > 2000:
                ps = ps / 100.0
            return ps
    raise KeyError(f"ERA5: can't find surface pressure var in {list(ds_era.data_vars)}")

def pick_surface_pressure_dino(ds_dino):
    if "surface_pressure" in ds_dino:
        ps = ds_dino["surface_pressure"]
        # Pa -> hPa if needed
        if ps.max() > 2000:
            ps = ps / 100.0
        return ps
    raise KeyError("DINO: no surface_pressure")

def regrid_bilinear(src, src_lat, src_lon, tgt_lat, tgt_lon):
    # xarray interp: expects 1D coords
    return src.interp({src_lat: tgt_lat, src_lon: tgt_lon}, method="linear")

def main():
    ds_full = xr.open_dataset(FULL, chunks={"time": 1})
    ds_dino = xr.open_dataset(DINO, chunks={"time": 1})
    ds_era  = xr.open_dataset(ERA0)

    latF, lonF = detect_latlon(ds_full)
    latD, lonD = detect_latlon(ds_dino)
    latE, lonE = detect_latlon(ds_era)

    ps_full0 = pick_surface_pressure_full(ds_full).isel(time=0)
    ps_dino0 = pick_surface_pressure_dino(ds_dino).isel(time=0)
    ps_era0  = pick_surface_pressure_era(ds_era).squeeze()

    # ERA regriddé sur FULL / DINO
    era_on_full = regrid_bilinear(ps_era0, latE, lonE, ds_full[latF], ds_full[lonF])
    era_on_dino = regrid_bilinear(ps_era0, latE, lonE, ds_dino[latD], ds_dino[lonD])

    # means
    m_full = wmean_2d(ps_full0, latF, lonF)
    m_dino = wmean_2d(ps_dino0, latD, lonD)
    m_eF   = wmean_2d(era_on_full, latF, lonF)
    m_eD   = wmean_2d(era_on_dino, latD, lonD)

    print("\n=== Global mean surface pressure at t=0 (hPa) ===")
    print(f"FULL (native):     {m_full:8.3f}")
    print(f"ERA  -> FULL grid: {m_eF:8.3f}   (FULL - ERA_on_FULL = {m_full-m_eF:+.3f})")
    print(f"DINO (native):     {m_dino:8.3f}")
    print(f"ERA  -> DINO grid: {m_eD:8.3f}   (DINO - ERA_on_DINO = {m_dino-m_eD:+.3f})")

    # quick sanity
    print("\n=== Sanity ranges (min/max, hPa) ===")
    def rng(x): return (float(x.min().values), float(x.max().values))
    print("FULL ps0:", rng(ps_full0))
    print("DINO ps0:", rng(ps_dino0))
    print("ERA0    :", rng(ps_era0/100.0 if ps_era0.max()>2000 else ps_era0))

if __name__ == "__main__":
    main()
