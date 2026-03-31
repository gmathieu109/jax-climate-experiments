#!/usr/bin/env python
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================
forcing_file = "/home/gmathieu/code_uqam/src/jax-gcm/jcm/data/bc/t30/clim/forcing.nc"
era5_zarr = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
outdir = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/forcing_analysis/triptychs"
os.makedirs(outdir, exist_ok=True)



# ============================================================
# CONFIG
# ============================================================
months_to_plot = {
    1: "January",
    7: "July",
}

# IMPORTANT:
# mode = "raw"         -> comparaison directe
# mode = "standardized"-> utile quand les unités ne sont pas comparables
#
# Recommandation :
# - SST      : raw
# - soilw_am : raw si tu sais que c'est comparable; sinon standardized
# - snowc    : standardized fortement recommandé
var_configs = {
    "sst": {
        "era_name": "sea_surface_temperature",
        "title": "Sea surface temperature",
        "units": "K",   # ou "°C" si tu convertis les deux
        "cmap": "coolwarm",
        "diff_cmap": "RdBu_r",
        "mode": "raw",
        "jcm_transform": lambda da: da,
        "era_transform": lambda da: da,
    },
    "soilw_am": {
        "era_name": "volumetric_soil_water_layer_1",
        "title": "Soil moisture",
        "units": "model / ERA5 units",
        "cmap": "YlGnBu",
        "diff_cmap": "RdBu_r",
        "mode": "raw",  # mettre "standardized" si tu veux comparaison de pattern seulement
        "jcm_transform": lambda da: da,
        "era_transform": lambda da: da,
    },
    "snowc": {
        "era_name": "snow_depth",
        "title": "Snow: JCM snowc vs ERA5 snow_depth",
        "units": "pattern-comparison units",
        "cmap": "Blues",
        "diff_cmap": "RdBu_r",
        "mode": "standardized",
        "jcm_transform": lambda da: da,
        "era_transform": lambda da: xr.where((da / 0.06) > 1, 1, da / 0.06),
    },
}

# ============================================================
# HELPERS
# ============================================================
def rename_latlon(ds):
    """Rename common ERA/JCM coordinate names to lat/lon."""
    rename_dict = {}
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds


def lon_to_0360(ds):
    """Convert longitude to [0, 360) if needed, then sort."""
    if "lon" not in ds.coords:
        return ds
    lon = ds["lon"]
    if float(lon.min()) < 0:
        ds = ds.assign_coords(lon=((lon + 360) % 360))
    ds = ds.sortby("lon")
    return ds


def sort_latlon(ds):
    """Sort coordinates for safe interpolation / plotting."""
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    return ds


def monthly_climatology(da):
    """Return monthly climatology from ERA5 time series."""
    return da.groupby("time.month").mean("time")


def select_jcm_month(da, month):
    """
    Select month from JCM forcing.
    Assumes monthly climatology stored on time dimension of length 12.
    month = 1..12
    """
    if "time" not in da.dims:
        return da
    return da.isel(time=month - 1)


def regrid_era_to_jcm(era_da, jcm_template):
    """
    Bilinear interpolation of ERA5 onto JCM lat/lon grid using xarray.interp.
    Assumes regular lat/lon grids.
    """
    return era_da.interp(
        lat=jcm_template["lat"],
        lon=jcm_template["lon"],
        method="linear"
    )


def robust_minmax(a, b, q_low=0.02, q_high=0.98):
    """
    Shared robust min/max from combined finite values of a and b.
    Avoids one outlier ruining the color scale.
    """
    vals = np.concatenate([
        np.ravel(a.values[np.isfinite(a.values)]),
        np.ravel(b.values[np.isfinite(b.values)]),
    ])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = np.quantile(vals, q_low)
    vmax = np.quantile(vals, q_high)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def robust_symmetric_limit(diff, q=0.98):
    """Symmetric limit for difference panel."""
    vals = np.ravel(diff.values[np.isfinite(diff.values)])
    if vals.size == 0:
        return 1.0
    lim = np.quantile(np.abs(vals), q)
    if np.isclose(lim, 0):
        lim = 1e-6
    return float(lim)


def standardize_2d(da):
    """
    Standardize spatial field:
    (x - mean) / std
    Useful for pattern comparison when units differ.
    """
    mean = da.mean(dim=("lat", "lon"), skipna=True)
    std = da.std(dim=("lat", "lon"), skipna=True)
    std = xr.where(std == 0, np.nan, std)
    return (da - mean) / std


def apply_common_mask(a, b):
    """Keep only points where both fields are finite."""
    mask = np.isfinite(a) & np.isfinite(b)
    return a.where(mask), b.where(mask)


def plot_triptych(jcm_da, era_da, title, units, outpath,
                  cmap="coolwarm", diff_cmap="RdBu_r",
                  mode="raw", dpi=300):
    """
    Plot [JCM] [ERA5] [JCM - ERA5]
    """
    # Ensure lat/lon ordering
    jcm_da = jcm_da.transpose("lat", "lon")
    era_da = era_da.transpose("lat", "lon")

    # Common mask
    jcm_da, era_da = apply_common_mask(jcm_da, era_da)

    if mode == "standardized":
        jcm_plot = standardize_2d(jcm_da)
        era_plot = standardize_2d(era_da)
        panel_units = "standardized spatial units"
    else:
        jcm_plot = jcm_da
        era_plot = era_da
        panel_units = units

    diff = jcm_plot - era_plot

    # Shared scale for JCM and ERA5
    vmin, vmax = robust_minmax(jcm_plot, era_plot)
    dlim = robust_symmetric_limit(diff)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # Panel 1: JCM
    im0 = axes[0].pcolormesh(
        jcm_plot["lon"], jcm_plot["lat"], jcm_plot,
        shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axes[0].set_title("JCM")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # Panel 2: ERA5
    im1 = axes[1].pcolormesh(
        era_plot["lon"], era_plot["lat"], era_plot,
        shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axes[1].set_title("ERA5 (regridded to JCM grid)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("")

    # Panel 3: Difference
    im2 = axes[2].pcolormesh(
        diff["lon"], diff["lat"], diff,
        shading="auto", cmap=diff_cmap, vmin=-dlim, vmax=dlim
    )
    axes[2].set_title("JCM - ERA5")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("")

    # Shared colorbar for first two panels
    cbar_main = fig.colorbar(im1, ax=axes[:2], shrink=0.90, pad=0.03)
    cbar_main.set_label(panel_units)

    # Difference colorbar
    cbar_diff = fig.colorbar(im2, ax=axes[2], shrink=0.90, pad=0.03)
    cbar_diff.set_label(f"Difference ({panel_units})")

    fig.suptitle(title, fontsize=14)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# LOAD DATA
# ============================================================
print("Loading JCM forcing...")
ds_jcm = xr.open_dataset(forcing_file)
ds_jcm = rename_latlon(ds_jcm)
ds_jcm = lon_to_0360(ds_jcm)
ds_jcm = sort_latlon(ds_jcm)

print("Loading ERA5 Zarr...")
ds_era = xr.open_zarr(era5_zarr)
ds_era = rename_latlon(ds_era)
ds_era = lon_to_0360(ds_era)
ds_era = sort_latlon(ds_era)

# ============================================================
# MAIN LOOP
# ============================================================
for jcm_name, cfg in var_configs.items():
    era_name = cfg["era_name"]

    print(f"\nProcessing {jcm_name} vs {era_name}...")

    if jcm_name not in ds_jcm:
        print(f"  -> Skipping: {jcm_name} not found in JCM file.")
        continue
    if era_name not in ds_era:
        print(f"  -> Skipping: {era_name} not found in ERA5 dataset.")
        continue

    da_jcm = ds_jcm[jcm_name]
    da_era = ds_era[era_name]

    # Apply variable transforms if needed
    da_jcm = cfg["jcm_transform"](da_jcm)
    da_era = cfg["era_transform"](da_era)

    # Monthly climatology for ERA5
    if "time" in da_era.dims:
        da_era_monthly = monthly_climatology(da_era)
    else:
        raise ValueError(f"ERA5 variable {era_name} has no time dimension.")

    for month_num, month_label in months_to_plot.items():
        print(f"  -> {month_label}")

        # JCM monthly forcing
        jcm_m = select_jcm_month(da_jcm, month_num)

        # ERA5 climatological month
        era_m = da_era_monthly.sel(month=month_num)

        # Regrid ERA5 to JCM grid
        era_m_on_jcm = regrid_era_to_jcm(era_m, ds_jcm)

        # Some variables can contain extra dims (e.g., depth, level)
        # Here we expect 2D lat/lon fields only.
        if set(jcm_m.dims) != {"lat", "lon"}:
            raise ValueError(
                f"JCM variable {jcm_name} for month {month_num} has dims {jcm_m.dims}, "
                "expected exactly ('lat', 'lon')."
            )
        if set(era_m_on_jcm.dims) != {"lat", "lon"}:
            raise ValueError(
                f"ERA5 variable {era_name} for month {month_num} has dims {era_m_on_jcm.dims}, "
                "expected exactly ('lat', 'lon') after selection/regridding."
            )

        outname = f"{jcm_name}_vs_{era_name}_{month_label.lower()}.png"
        outpath = os.path.join(outdir, outname)

        title = f"{cfg['title']} — {month_label}"

        plot_triptych(
            jcm_da=jcm_m,
            era_da=era_m_on_jcm,
            title=title,
            units=cfg["units"],
            outpath=outpath,
            cmap=cfg["cmap"],
            diff_cmap=cfg["diff_cmap"],
            mode=cfg["mode"],
            dpi=300,
        )

print(f"\n✅ Triptychs saved in: {outdir}")