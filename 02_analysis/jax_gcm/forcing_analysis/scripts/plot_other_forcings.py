import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

forcing_file = "/home/gmathieu/code_uqam/src/jax-gcm/jcm/data/bc/t30/clim/forcing.nc"
outdir = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/forcing_analysis/other_forcings"
os.makedirs(outdir, exist_ok=True)

ds = xr.open_dataset(forcing_file)

months = [("Jan", 0), ("Apr", 3), ("Jul", 6), ("Oct", 9)]

var_configs = {
    "alb": {
        "title": "Surface albedo",
        "time_dependent": False,
        "cmap": "viridis",
    },
    "icec": {
        "title": "Sea ice concentration",
        "time_dependent": True,
        "cmap": "Blues",
    },
    "sst": {
        "title": "Sea surface temperature",
        "time_dependent": True,
        "cmap": "coolwarm",
    },
    "stl": {
        "title": "Land surface temperature",
        "time_dependent": True,
        "cmap": "coolwarm",
    },
    "soilw_am": {
        "title": "Soil moisture",
        "time_dependent": True,
        "cmap": "YlGnBu",
    },
}

def area_weighted_mean(da):
    lat = da["lat"]
    weights = np.cos(np.deg2rad(lat))
    return da.weighted(weights).mean(dim=("lat", "lon"))

def plot_2x2_panels(varname, cfg):
    da = ds[varname]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()

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

def plot_monthly_mean(varname, cfg):
    if not cfg["time_dependent"]:
        return

    da = ds[varname]
    monthly = area_weighted_mean(da)

    plt.figure(figsize=(8, 4))
    monthly.plot(marker="o")
    plt.title(f"JCM forcing - monthly global mean - {cfg['title']}")
    plt.xlabel("Month")
    plt.ylabel(varname)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{varname}_monthly_mean.png"), dpi=160)
    plt.close()

for varname, cfg in var_configs.items():
    print(f"Processing {varname}...")
    plot_2x2_panels(varname, cfg)
    plot_monthly_mean(varname, cfg)

print(f"Done. Figures saved in: {outdir}")
