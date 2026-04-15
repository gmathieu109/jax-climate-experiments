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
    return np.load(path, allow_pickle=True)


def maybe_load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def infer_rmse_series(metrics: np.ndarray) -> np.ndarray:
    # handle dict case
    if isinstance(metrics, np.ndarray) and metrics.dtype == object:
        metrics = metrics.item()

    if isinstance(metrics, dict):
        return np.array(metrics["rmse"])

    metrics = np.asarray(metrics)

    if metrics.ndim == 1:
        return metrics

    if metrics.ndim == 2:
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


def summarize_run(run: Dict):
    metrics = load_npy(run["train_dir"] / "metrics.npy")
    series = infer_rmse_series(metrics)
    alpha = load_npy(run["train_dir"] / "alpha_alb_final.npy")
    delta = alpha - 1.0

    init_loss = float(series[0])
    best_loss = float(np.nanmin(series))
    final_loss = float(series[-1])
    rel_gain = 100.0 * (init_loss - best_loss) / init_loss if init_loss != 0 else np.nan
    mean_abs_delta = float(np.nanmean(np.abs(delta)))
    max_abs_delta = float(np.nanmax(np.abs(delta)))

    print(
        f'{run["label"]:>10s} | init={init_loss:.4f} | best={best_loss:.4f} | '
        f'final={final_loss:.4f} | gain={rel_gain:.2f}% | '
        f'mean|da|={mean_abs_delta:.3e} | max|da|={max_abs_delta:.3e}'
    )


def plot_rmse_panel(runs: List[Dict], out_path: Path, smooth_window: int = 1):
    fig, ax = plt.subplots(figsize=(9, 5))

    for run in runs:
        metrics = load_npy(run["train_dir"] / "metrics.npy")
        series = infer_rmse_series(metrics)

        if smooth_window > 1:
            ax.plot(series, alpha=0.3, label=f'{run["label"]} raw')
            ax.plot(moving_average(series, smooth_window), linewidth=2, label=f'{run["label"]} smooth')
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


def plot_alpha_panel(runs: List[Dict], out_path: Path):
    n = len(runs)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    fields = []
    for run in runs:
        alpha = load_npy(run["train_dir"] / "alpha_alb_final.npy")
        fields.append(alpha - 1.0)

    vmax = max(np.nanmax(np.abs(f)) for f in fields)
    vmax = max(vmax, 1e-8)

    for i, run in enumerate(runs):
        ax = axes[i // ncols][i % ncols]
        alpha = load_npy(run["train_dir"] / "alpha_alb_final.npy")
        field = alpha - 1.0

        im = ax.imshow(field.T, origin="lower", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(run["label"])
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("alpha_alb_final - 1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_run_spec(spec: str) -> Dict:
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(f"Each --run must be 'label|train_dir|eval_dir', got: {spec}")
    label, train_dir, eval_dir = parts
    return {
        "label": label,
        "train_dir": Path(train_dir),
        "eval_dir": Path(eval_dir) if eval_dir else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="Format: label|train_dir|eval_dir")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--smooth_window", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = [parse_run_spec(spec) for spec in args.run]

    print("\n=== RUN SUMMARY ===")
    for run in runs:
        summarize_run(run)

    plot_rmse_panel(
        runs=runs,
        out_path=args.out_dir / "panel_rmse_multi_lr.png",
        smooth_window=args.smooth_window,
    )
    plot_alpha_panel(
        runs=runs,
        out_path=args.out_dir / "panel_alpha_alb_minus1_multi_lr.png",
    )

    with open(args.out_dir / "panel_config.json", "w") as f:
        json.dump(
            {
                "runs": [
                    {
                        "label": r["label"],
                        "train_dir": str(r["train_dir"]),
                        "eval_dir": str(r["eval_dir"]) if r["eval_dir"] else "",
                    }
                    for r in runs
                ]
            },
            f,
            indent=2,
        )

    print(f"\nSaved panels to: {args.out_dir}")


if __name__ == "__main__":
    main()