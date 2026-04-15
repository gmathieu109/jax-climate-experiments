#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def maybe_load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path)


def infer_rmse_series(metrics: np.ndarray) -> np.ndarray:
    """
    Tries to robustly extract a 1D RMSE/loss series from metrics.npy.
    Adapt this if your metrics.npy has a known structure.
    """
    metrics = np.asarray(metrics)

    if metrics.ndim == 1:
        return metrics

    if metrics.ndim == 2:
        # Common cases:
        # shape = (n_steps, n_metrics) -> take first column
        # or shape = (n_metrics, n_steps) -> take first row if smaller first dim
        if metrics.shape[0] > metrics.shape[1]:
            return metrics[:, 0]
        return metrics[0, :]

    raise ValueError(f"Unsupported metrics shape: {metrics.shape}")


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window > len(x):
        return x.copy()
    kernel = np.ones(window) / window
    y = np.convolve(x, kernel, mode="valid")
    pad_left = window - 1
    return np.concatenate([np.full(pad_left, np.nan), y])


def area_weighted_mean(field: np.ndarray, lat_deg: np.ndarray) -> float:
    """
    field shape assumed (lon, lat) or (lat, lon).
    Tries to detect layout from lat size.
    """
    lat_deg = np.asarray(lat_deg)
    weights = np.cos(np.deg2rad(lat_deg))
    weights = weights / weights.mean()

    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got shape {field.shape}")

    if field.shape[1] == lat_deg.size:
        # (lon, lat)
        return np.nanmean(field * weights[None, :])
    elif field.shape[0] == lat_deg.size:
        # (lat, lon)
        return np.nanmean(field * weights[:, None])
    else:
        raise ValueError(
            f"Could not align field shape {field.shape} with lat size {lat_deg.size}"
        )


def compute_time_mean_bias(
    pred: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Returns time-mean bias = mean(pred - target, time)
    Supports shape (time, lon, lat) or (time, lat, lon)
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
    return np.mean(pred - target, axis=0)


def add_common_map_format(ax, lon_deg, lat_deg, title):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    if lon_deg is not None and len(lon_deg) > 1:
        ax.set_xticks(np.linspace(0, len(lon_deg) - 1, 5))
        ax.set_xticklabels([f"{lon_deg[int(i)]:.0f}" for i in np.linspace(0, len(lon_deg) - 1, 5)])

    if lat_deg is not None and len(lat_deg) > 1:
        ax.set_yticks(np.linspace(0, len(lat_deg) - 1, 5))
        ax.set_yticklabels([f"{lat_deg[int(i)]:.0f}" for i in np.linspace(0, len(lat_deg) - 1, 5)])


def plot_rmse_panel(
    runs: List[Dict],
    out_path: Path,
    smooth_window: int = 1,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    for run in runs:
        metrics = load_npy(run["train_dir"] / "metrics.npy")
        series = infer_rmse_series(metrics)
        smoothed = moving_average(series, smooth_window)
        ax.plot(series, alpha=0.35, label=f'{run["label"]} raw')
        if smooth_window > 1:
            ax.plot(smoothed, linewidth=2, label=f'{run["label"]} smooth')
        else:
            ax.plot(series, linewidth=2, label=run["label"])

    ax.set_title("RMSE / loss during albedo optimization")
    ax.set_xlabel("iteration")
    ax.set_ylabel("RMSE or loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_alpha_panel(
    runs: List[Dict],
    out_path: Path,
):
    n = len(runs)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.8 * nrows), squeeze=False)

    all_fields = []
    lon_deg = None
    lat_deg = None

    for run in runs:
        alpha = load_npy(run["train_dir"] / "alpha_alb_final.npy")
        field = alpha - 1.0
        all_fields.append(field)

        eval_dir = run.get("eval_dir")
        if eval_dir is not None:
            lon_deg = maybe_load_npy(eval_dir / "lon_deg.npy") if lon_deg is None else lon_deg
            lat_deg = maybe_load_npy(eval_dir / "lat_deg.npy") if lat_deg is None else lat_deg

    vmax = max(np.nanmax(np.abs(f)) for f in all_fields)
    vmax = max(vmax, 1e-8)

    for i, run in enumerate(runs):
        ax = axes[i // ncols][i % ncols]
        alpha = load_npy(run["train_dir"] / "alpha_alb_final.npy")
        field = alpha - 1.0

        # transpose if needed so lat is vertical
        if lat_deg is not None and field.shape[1] == lat_deg.size:
            img = field.T
        else:
            img = field

        im = ax.imshow(img, origin="lower", aspect="auto", vmin=-vmax, vmax=vmax)
        mean_abs = np.nanmean(np.abs(field))
        add_common_map_format(ax, lon_deg, lat_deg, f'{run["label"]} | mean|δα|={mean_abs:.3e}')

    # hide empty axes
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("alpha_alb_final - 1")
    fig.suptitle("Final optimized albedo perturbations", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bias_panel(
    runs: List[Dict],
    out_path: Path,
):
    valid_runs = []
    for run in runs:
        eval_dir = run.get("eval_dir")
        if eval_dir is None:
            continue
        needed = [
            eval_dir / "t850_init.npy",
            eval_dir / "t850_final.npy",
            eval_dir / "t850_target.npy",
            eval_dir / "lat_deg.npy",
            eval_dir / "lon_deg.npy",
        ]
        if all(p.exists() for p in needed):
            valid_runs.append(run)

    if not valid_runs:
        print("No complete eval dirs found for bias panel. Skipping.")
        return

    n = len(valid_runs)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.8 * n), squeeze=False)

    global_abs_max = 0.0
    prepared = []

    for run in valid_runs:
        eval_dir = run["eval_dir"]
        t850_init = load_npy(eval_dir / "t850_init.npy")
        t850_final = load_npy(eval_dir / "t850_final.npy")
        t850_target = load_npy(eval_dir / "t850_target.npy")
        lat_deg = load_npy(eval_dir / "lat_deg.npy")
        lon_deg = load_npy(eval_dir / "lon_deg.npy")

        bias_init = compute_time_mean_bias(t850_init, t850_target)
        bias_final = compute_time_mean_bias(t850_final, t850_target)

        global_abs_max = max(
            global_abs_max,
            np.nanmax(np.abs(bias_init)),
            np.nanmax(np.abs(bias_final)),
        )

        prepared.append((run, bias_init, bias_final, lon_deg, lat_deg))

    global_abs_max = max(global_abs_max, 1e-8)

    for i, (run, bias_init, bias_final, lon_deg, lat_deg) in enumerate(prepared):
        for j, (field, title_suffix) in enumerate(
            [(bias_init, "initial bias"), (bias_final, "final bias")]
        ):
            ax = axes[i][j]
            if field.shape[1] == lat_deg.size:
                img = field.T
            else:
                img = field

            im = ax.imshow(
                img,
                origin="lower",
                aspect="auto",
                vmin=-global_abs_max,
                vmax=global_abs_max,
            )
            add_common_map_format(ax, lon_deg, lat_deg, f'{run["label"]} | {title_suffix}')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("T850 bias (model - ERA5)")
    fig.suptitle("Time-mean T850 bias maps", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_run_spec(spec: str) -> Dict:
    """
    Format:
      label|train_dir|eval_dir
    eval_dir can be empty:
      label|train_dir|
    """
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(
            f"Each --run must be 'label|train_dir|eval_dir', got: {spec}"
        )
    label, train_dir, eval_dir = parts
    return {
        "label": label,
        "train_dir": Path(train_dir),
        "eval_dir": Path(eval_dir) if eval_dir else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: label|train_dir|eval_dir",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--smooth_window", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = [parse_run_spec(spec) for spec in args.run]

    plot_rmse_panel(
        runs=runs,
        out_path=args.out_dir / "panel_rmse_multi_lr.png",
        smooth_window=args.smooth_window,
    )
    plot_alpha_panel(
        runs=runs,
        out_path=args.out_dir / "panel_alpha_alb_minus1_multi_lr.png",
    )
    plot_bias_panel(
        runs=runs,
        out_path=args.out_dir / "panel_t850_bias_init_final_multi_lr.png",
    )

    summary = {
        "runs": [
            {
                "label": r["label"],
                "train_dir": str(r["train_dir"]),
                "eval_dir": str(r["eval_dir"]) if r["eval_dir"] is not None else "",
            }
            for r in runs
        ]
    }
    with open(args.out_dir / "panel_config.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved panels to: {args.out_dir}")


if __name__ == "__main__":
    main()
