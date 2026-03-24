#!/usr/bin/env python

import os
import xarray as xr
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
forcing_file = "/home/gmathieu/code_uqam/src/jax-gcm/jcm/data/bc/t30/clim/forcing.nc"
outdir = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/forcing_analysis/panels"
os.makedirs(outdir, exist_ok=True)

# =========================
# LOAD DATA
# =========================
ds = xr.open_dataset(forcing_file)

# =========================
# CONFIG
# =========================
months = [("Jan", 0), ("Apr", 3), ("Jul", 6), ("Oct", 9)]

var_configs = {
    "alb":      {"cmap": "viridis",  "time_dependent": False, "title": "Surface albedo"},
    "snowc":    {"cmap": "Blues",    "time_dependent": True,  "title": "Snow cover"},
    "sst":      {"cmap": "coolwarm", "time_dependent": True,  "title": "Sea surface temperature"},
    "icec":     {"cmap": "Blues",    "time_dependent": True,  "title": "Sea ice concentration"},
    "stl":      {"cmap": "coolwarm", "time_dependent": True,  "title": "Land surface temperature"},
    "soilw_am": {"cmap": "YlGnBu",   "time_dependent": True,  "title": "Soil moisture"},
}

# =========================
# PLOTTING FUNCTION
# =========================
def save_2x2_panels(varname, cfg):
    da = ds[varname]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()

    # Color scale globale (important pour comparaison)
    vmin = float(da.min())
    vmax = float(da.max())

    if cfg["time_dependent"]:
        for ax, (label, idx) in zip(axes, months):
            da_plot = da.isel(time=idx).transpose("lat", "lon")

            da_plot.plot(
                ax=ax,
                cmap=cfg["cmap"],
                vmin=vmin,
                vmax=vmax,
                add_colorbar=True,
            )

            ax.set_title(label)
            ax.set_xlabel("")
            ax.set_ylabel("")

    else:
        da_plot = da.transpose("lat", "lon")

        for ax, (label, _) in zip(axes, months):
            da_plot.plot(
                ax=ax,
                cmap=cfg["cmap"],
                vmin=vmin,
                vmax=vmax,
                add_colorbar=True,
            )

            ax.set_title(label)
            ax.set_xlabel("")
            ax.set_ylabel("")

    fig.suptitle(f"JCM forcing - {cfg['title']}", fontsize=14)
    fig.savefig(os.path.join(outdir, f"{varname}_panel.png"), dpi=160)
    plt.close(fig)

# =========================
# RUN
# =========================
for varname, cfg in var_configs.items():
    save_2x2_panels(varname, cfg)

print(f"✅ Figures saved in: {outdir}")