#!/usr/bin/env python3
"""
Build heatmap of SST optimization RMSE improvement
as a function of initialization month and optimization window length.

Expected run names:
delta_sst_masked_init2000-01-01_5d_steps15_lr0p5_l2_0p0001_sm_0p02_global
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OPTIM_DIR = "/cuyo/gmathieu/outputs_uqam/jcm_experiments/optimisation_sst_masked"

SEASON_LABELS = {
    "2000-01-01": "Jan",
    "2000-04-01": "Apr",
    "2000-07-01": "Jul",
    "2000-10-01": "Oct",
}

SEASON_ORDER = ["Jan", "Apr", "Jul", "Oct"]
DURATION_ORDER = [5, 10, 15]


def parse_run_name(dirname: str) -> dict | None:
    pattern = (
        r"delta_sst_masked_"
        r"init(?P<init_date>\d{4}-\d{2}-\d{2})_"
        r"(?P<days>\d+)d_"
        r"steps(?P<n_steps>\d+)_"
        r"lr(?P<lr>[0-9p]+)_"
        r"l2_(?P<lambda_l2>[0-9p]+)_"
        r"sm_(?P<lambda_smooth>[0-9p]+)_"
        r"(?P<scope>global|nh30)"
    )

    m = re.match(pattern, dirname)
    if m is None:
        return None

    d = m.groupdict()

    init_date = d["init_date"]

    return {
        "dirname": dirname,
        "init_date": init_date,
        "season": SEASON_LABELS.get(init_date, init_date),
        "days": int(d["days"]),
        "n_steps": int(d["n_steps"]),
        "lr": float(d["lr"].replace("p", ".")),
        "lambda_l2": float(d["lambda_l2"].replace("p", ".")),
        "lambda_smooth": float(d["lambda_smooth"].replace("p", ".")),
        "scope": d["scope"],
    }


def load_results() -> pd.DataFrame:
    results = []

    if not os.path.exists(OPTIM_DIR):
        raise FileNotFoundError(f"OPTIM_DIR not found: {OPTIM_DIR}")

    for dirname in sorted(os.listdir(OPTIM_DIR)):
        if not dirname.startswith("delta_sst_masked_"):
            continue

        config = parse_run_name(dirname)
        if config is None:
            print(f"Skipping unrecognized run name: {dirname}")
            continue

        metrics_path = os.path.join(OPTIM_DIR, dirname, "metrics.npy")
        if not os.path.exists(metrics_path):
            print(f"Skipping incomplete run, no metrics.npy: {dirname}")
            continue

        try:
            metrics = np.load(metrics_path, allow_pickle=True).item()
            rmse_values = np.asarray(metrics["rmse"], dtype=float)

            if len(rmse_values) == 0:
                print(f"Skipping run with empty RMSE: {dirname}")
                continue

            rmse_initial = float(rmse_values[0])
            rmse_final = float(rmse_values[-1])
            improvement = (rmse_initial - rmse_final) / rmse_initial * 100.0

            row = {
                **config,
                "rmse_initial": rmse_initial,
                "rmse_final": rmse_final,
                "rmse_improvement_%": improvement,
            }
            results.append(row)

        except Exception as e:
            print(f"Error reading {dirname}: {e}")

    return pd.DataFrame(results)


def plot_heatmap(df: pd.DataFrame):
    sub = df[
        (df["season"].isin(SEASON_ORDER)) &
        (df["days"].isin(DURATION_ORDER)) &
        (df["scope"] == "global") &
        (df["n_steps"] == 15) &
        (np.isclose(df["lr"], 0.5)) &
        (np.isclose(df["lambda_l2"], 1e-4)) &
        (np.isclose(df["lambda_smooth"], 0.02))
    ].copy()

    if sub.empty:
        print("No matching runs found for season-duration heatmap.")
        return

    print("\nRuns used in heatmap:")
    print(
        sub[[
            "dirname",
            "season",
            "days",
            "n_steps",
            "lr",
            "lambda_l2",
            "lambda_smooth",
            "rmse_improvement_%",
        ]].sort_values(["season", "days"]).to_string(index=False)
    )

    pivot = sub.pivot_table(
        index="season",
        columns="days",
        values="rmse_improvement_%",
        aggfunc="max",
    )

    pivot = pivot.reindex(index=SEASON_ORDER, columns=DURATION_ORDER)

    fig, ax = plt.subplots(figsize=(7, 5))

    vmax = max(8.0, np.nanmax(pivot.values))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(DURATION_ORDER)))
    ax.set_xticklabels([f"{d}d" for d in DURATION_ORDER])

    ax.set_yticks(np.arange(len(SEASON_ORDER)))
    ax.set_yticklabels(SEASON_ORDER)

    ax.set_xlabel("Optimization window length")
    ax.set_ylabel("Initialization month")
    ax.set_title("RMSE improvement (%)\nSST optimization by season and window length")

    for i in range(len(SEASON_ORDER)):
        for j in range(len(DURATION_ORDER)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="gray")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE reduction [%]")

    outdir = os.path.join(OPTIM_DIR, "comparison_plots")
    os.makedirs(outdir, exist_ok=True)

    outfile = os.path.join(outdir, "heatmap_season_vs_duration.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"\nSaved heatmap to: {outfile}")

    csvfile = os.path.join(outdir, "season_duration_results.csv")
    sub.to_csv(csvfile, index=False)
    print(f"Saved CSV to: {csvfile}")

    plt.close(fig)


def main():
    df = load_results()

    if df.empty:
        print("No SST masked optimization results found.")
        return

    print(f"Loaded {len(df)} runs.")
    plot_heatmap(df)


if __name__ == "__main__":
    main()
