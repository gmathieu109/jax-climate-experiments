#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/cuyo/gmathieu/outputs_uqam/jcm_experiments/optimisation_sst_masked")

RUNS = {
    "Jan": "delta_sst_masked_init2000-01-01_15d_steps15_lr0p5_l2_0p0001_sm_0p02_global",
    "Apr": "delta_sst_masked_init2000-04-01_15d_steps15_lr0p5_l2_0p0001_sm_0p02_global",
    "Jul": "delta_sst_masked_init2000-07-01_15d_steps15_lr0p5_l2_0p0001_sm_0p02_global",
    "Oct": "delta_sst_masked_init2000-10-01_15d_steps15_lr0p5_l2_0p0001_sm_0p02_global",
}

OUTDIR = BASE / "comparison_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
axes = axes.ravel()

vmin, vmax = -4, 4

for ax, (label, run) in zip(axes, RUNS.items()):
    run_dir = BASE / run
    delta = np.load(run_dir / "delta_sst_final.npy")
    ocean_mask = np.load(run_dir / "ocean_mask.npy")

    # Mask land as white
    delta_plot = np.where(ocean_mask > 0.5, delta, np.nan)

    im = ax.imshow(
        delta_plot.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    mean = np.nanmean(delta_plot)

    rmse_gain = {
        "Jan": 11.7,
        "Apr": 11.3,
        "Jul": 11.2,
        "Oct": 11.4,
    }[label]

    ax.set_title(
        f"{label}\nRMSE ↓ {rmse_gain:.1f}%   |   mean ΔSST = {mean:.2f} K",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude index")
    ax.set_ylabel("Latitude index")

cbar = fig.colorbar(im, ax=axes, shrink=0.9)
cbar.set_label("Optimized ΔSST (K)", fontsize=12)

fig.suptitle(
    "Optimized ΔSST fields for 15-day windows\n(similar spatial structures across seasons)",
    fontsize=16,
    fontweight="bold",
)

outfile = OUTDIR / "delta_sst_15d_seasonal_maps.png"
fig.savefig(outfile, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved:", outfile)

