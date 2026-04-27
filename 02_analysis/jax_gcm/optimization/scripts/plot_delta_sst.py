#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def savefig(fig_dir: str, name: str):
    path = os.path.join(fig_dir, name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)


def symmetric_vmax(arr: np.ndarray, floor: float = 1e-8) -> float:
    vmax = float(np.max(np.abs(arr)))
    return max(vmax, floor)


def load_metrics(metrics_path: str):
    m = np.load(metrics_path, allow_pickle=True).item()
    return np.array(m["iter"]), np.array(m["rmse"])


def load_checkpoints(run_dir: str):
    files = sorted(glob.glob(os.path.join(run_dir, "delta_sst_step_*.npy")))
    out = []
    for f in files:
        m = re.search(r"delta_sst_step_(\d+)\.npy", os.path.basename(f))
        if m:
            out.append((int(m.group(1)), f))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    metrics_path = os.path.join(run_dir, "metrics.npy")
    final_path = os.path.join(run_dir, "delta_sst_final.npy")

    # -------------------------
    # Figure 1: RMSE vs iteration
    # -------------------------
    iters, rmses = load_metrics(metrics_path)

    plt.figure(figsize=(7, 4.5))
    plt.plot(iters, rmses, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("RMSE evolution during SST optimization")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig_dir, "rmse_vs_iteration.png")
    plt.close()

    print("RMSE summary:")
    print("  init :", float(rmses[0]))
    print("  best :", float(np.min(rmses)))
    print("  final:", float(rmses[-1]))

    # -------------------------
    # Figure 2: final delta_sst
    # -------------------------
    delta = np.load(final_path)

    print("delta_sst_final shape:", delta.shape)
    print("delta_sst_final mean/min/max:",
          float(delta.mean()), float(delta.min()), float(delta.max()))

    vmax = symmetric_vmax(delta)

    plt.figure(figsize=(10, 4.5))
    im = plt.imshow(
        delta.T,
        origin="lower",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        cmap="RdBu_r",
    )
    plt.colorbar(im, label=r"$\Delta SST$ [K]")
    plt.xlabel("Longitude index")
    plt.ylabel("Latitude index")
    plt.title(r"Optimized SST perturbation ($\Delta SST$)")
    plt.tight_layout()
    savefig(fig_dir, "delta_sst_final.png")
    plt.close()

    # -------------------------
    # Figure 3: checkpoint evolution
    # -------------------------
    ckpts = load_checkpoints(run_dir)

    if len(ckpts) > 0:
        if len(ckpts) <= 4:
            chosen = ckpts
        else:
            idxs = np.linspace(0, len(ckpts) - 1, 4).astype(int)
            chosen = [ckpts[i] for i in idxs]

        deltas = []
        labels = []
        for it, f in chosen:
            d = np.load(f)
            deltas.append(d)
            labels.append(f"iter {it}")

        vmax_ckpt = max(symmetric_vmax(d) for d in deltas)

        fig, axes = plt.subplots(1, len(deltas), figsize=(4 * len(deltas), 4), constrained_layout=True)
        if len(deltas) == 1:
            axes = [axes]

        for ax, d, lab in zip(axes, deltas, labels):
            im = ax.imshow(
                d.T,
                origin="lower",
                aspect="auto",
                vmin=-vmax_ckpt,
                vmax=vmax_ckpt,
                cmap="RdBu_r",
            )
            ax.set_title(lab)
            ax.set_xlabel("Lon index")
            ax.set_ylabel("Lat index")

        cbar = fig.colorbar(im, ax=axes, shrink=0.85)
        cbar.set_label(r"$\Delta SST$ [K]")
        savefig(fig_dir, "delta_sst_checkpoint_evolution.png")
        plt.close(fig)
    else:
        print("No checkpoints found, skipping checkpoint figure.")

    print("Done.")


if __name__ == "__main__":
    main()
