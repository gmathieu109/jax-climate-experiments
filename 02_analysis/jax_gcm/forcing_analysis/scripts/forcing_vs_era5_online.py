import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# PATHS
# =========================
forcing_file = "/home/gmathieu/code_uqam/src/jax-gcm/jcm/data/bc/t30/clim/forcing.nc"
outdir = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/forcing_analysis/era5_online"
os.makedirs(outdir, exist_ok=True)

# =========================
# LOAD JCM
# =========================
jcm = xr.open_dataset(forcing_file)

# =========================
# LOAD ERA5 (ONLINE)
# =========================
print("Opening ERA5 Zarr...")

era = xr.open_zarr(
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
    consolidated=True,
    storage_options={"token": "anon"},
)

era = era.rename({
    "latitude": "lat",
    "longitude": "lon",
})

# =========================
# SELECT VARIABLES
# =========================
ERA_VARS = {
    "sst": "sea_surface_temperature",
    "icec": "sea_ice_cover",
    "snowc": "snow_depth",
}

era = era[list(ERA_VARS.values())]

# =========================
# CLIMATOLOGY (KEY STEP)
# =========================
print("Computing climatology...")
era = era.sel(time=slice("1990", "2010"))
era_clim = era.groupby("time.month").mean("time")

# =========================
# FIX COORDS
# =========================
era_clim = era_clim.assign_coords(
    lon=((era_clim.lon + 360) % 360)
).sortby("lon")
era_clim = era_clim.sortby("lat")

# =========================
# REGRID ERA → JCM
# =========================
def regrid(da):
    return da.interp(
        lat=jcm["lat"],
        lon=jcm["lon"],
        method="linear"
    )

# =========================
# PLOTTING
# =========================
def plot_triptych(j, e, diff, title, fname, cmap):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    j.plot(ax=axes[0], cmap=cmap, robust=True)
    axes[0].set_title("JCM")

    e.plot(ax=axes[1], cmap=cmap, robust=True)
    axes[1].set_title("ERA5")

    diff.plot(ax=axes[2], cmap="RdBu_r", robust=True)
    axes[2].set_title("JCM - ERA5")

    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(title)
    fig.savefig(os.path.join(outdir, fname), dpi=160)
    plt.close()

# =========================
# COMPARISON LOOP
# =========================
months = {
    "jan": 1,
    "jul": 7,
}

for key, era_name in ERA_VARS.items():
    print(f"Processing {key}...")

    jcm_da = jcm[key]
    era_da = era_clim[era_name]

    era_rg = regrid(era_da)

    for label, m in months.items():
        j = jcm_da.isel(time=m - 1).transpose("lat", "lon")
        e = era_rg.sel(month=m).transpose("lat", "lon")

        if key == "snowc":
            # Comparaison en présence/absence de neige
            j_plot = (j > 1.0).astype(float)
            e_plot = (e > 0.01).astype(float)   # 1 cm de neige
            diff = j_plot - e_plot

            fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

            j_plot.plot(ax=axes[0], cmap="Blues", vmin=0, vmax=1)
            axes[0].set_title("JCM")

            e_plot.plot(ax=axes[1], cmap="Blues", vmin=0, vmax=1)
            axes[1].set_title("ERA5")

            diff.plot(ax=axes[2], cmap="RdBu_r", vmin=-1, vmax=1)
            axes[2].set_title("JCM - ERA5")

            for ax in axes:
                ax.set_xlabel("")
                ax.set_ylabel("")

            fig.suptitle(f"SNOW PRESENCE - {label.upper()}")
            fig.savefig(os.path.join(outdir, f"snowmask_{label}.png"), dpi=160)
            plt.close(fig)

        else:
            diff = j - e
            cmap = "Blues" if key == "icec" else "coolwarm"

            plot_triptych(
                j, e, diff,
                title=f"{key.upper()} - {label.upper()}",
                fname=f"{key}_{label}.png",
                cmap=cmap,
            )

        print(f"  done {label}")

print("DONE")

