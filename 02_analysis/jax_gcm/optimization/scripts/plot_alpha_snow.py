import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =========================
# Paths
# =========================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True)
args = parser.parse_args()

data_dir = args.run_dir
fig_dir = os.path.join(data_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

alpha_final_path = os.path.join(data_dir, "alpha_snow_final.npy")
metrics_path = os.path.join(data_dir, "metrics.npy")

t850_init_path = os.path.join(data_dir, "t850_init.npy")
t850_final_path = os.path.join(data_dir, "t850_final.npy")
t850_target_path = os.path.join(data_dir, "t850_target.npy")
lat_path = os.path.join(data_dir, "lat_deg.npy")
lon_path = os.path.join(data_dir, "lon_deg.npy")



# =========================
# Helpers
# =========================
def symmetric_vmax(arr: np.ndarray, floor: float = 1e-6) -> float:
    vmax = float(np.max(np.abs(arr)))
    return max(vmax, floor)

def savefig(name: str):
    path = os.path.join(fig_dir, name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)

def load_checkpoints():
    files = sorted(glob.glob(os.path.join(data_dir, "alpha_step_*.npy")))
    out = []
    for f in files:
        m = re.search(r"alpha_step_(\d+)\.npy", os.path.basename(f))
        if m:
            it = int(m.group(1))
            out.append((it, f))
    return out

def parse_log_for_rmse(log_file: str):
    """
    Parse lines like:
    iter 013 | loss=... | rmse=... | ...
    """
    if not os.path.exists(log_file):
        print("No log file found at:", log_file)
        return None, None

    iters = []
    rmses = []

    pattern = re.compile(r"iter\s+(\d+)\s+\|\s+loss=.*?\|\s+rmse=([0-9eE\+\-\.]+)")
    with open(log_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                rmses.append(float(m.group(2)))

    if len(iters) == 0:
        print("No RMSE entries found in log.")
        return None, None

    return np.array(iters), np.array(rmses)

def load_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        print("No metrics file found at:", metrics_path)
        return None, None

    m = np.load(metrics_path, allow_pickle=True).item()
    return np.array(m["iter"]), np.array(m["rmse"])

# =========================
# Figure 1: final alpha anomaly
# =========================
alpha = np.load(alpha_final_path)
delta = alpha - 1.0

print("alpha shape:", alpha.shape)
print("alpha min/max/mean:", float(alpha.min()), float(alpha.max()), float(alpha.mean()))
print("delta min/max:", float(delta.min()), float(delta.max()))

vmax = symmetric_vmax(delta)

plt.figure(figsize=(10, 4.5))
im = plt.imshow(
    delta.T,
    origin="lower",
    aspect="auto",
    vmin=-vmax,
    vmax=vmax,
)
plt.colorbar(im, label=r"$\alpha_{snow} - 1$")
plt.xlabel("Longitude index")
plt.ylabel("Latitude index")
plt.title(r"Optimized snow-cover scaling anomaly ($\alpha_{snow} - 1$)")
plt.tight_layout()
savefig("alpha_snow_final_anomaly.png")
plt.close()

# =========================
# Figure 2: checkpoint evolution maps
# =========================
ckpts = load_checkpoints()

if len(ckpts) > 0:
    # choose up to 4 representative checkpoints
    chosen = []
    if len(ckpts) <= 4:
        chosen = ckpts
    else:
        idxs = np.linspace(0, len(ckpts) - 1, 4).astype(int)
        chosen = [ckpts[i] for i in idxs]

    deltas = []
    labels = []
    for it, f in chosen:
        a = np.load(f)
        deltas.append(a - 1.0)
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
        )
        ax.set_title(lab)
        ax.set_xlabel("Lon index")
        ax.set_ylabel("Lat index")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(r"$\alpha_{snow} - 1$")
    savefig("alpha_snow_checkpoint_evolution.png")
    plt.close(fig)
else:
    print("No checkpoints found, skipping checkpoint figure.")

# =========================
# Figure 3: RMSE vs iteration
# =========================
iters, rmses = load_metrics(metrics_path)

if iters is not None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(iters, rmses, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("RMSE evolution during optimization")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("rmse_vs_iteration.png")
    plt.close()
else:
    print("Skipping RMSE plot because no parsable log was found.")

# =========================
# Figure 4: mean bias maps
# =========================
if all(os.path.exists(p) for p in [t850_init_path, t850_final_path, t850_target_path]):
    t850_init = np.load(t850_init_path)
    t850_final = np.load(t850_final_path)
    t850_target = np.load(t850_target_path)

    print("Loaded T850 arrays:")
    print("t850_init shape  :", t850_init.shape)
    print("t850_final shape :", t850_final.shape)
    print("t850_target shape:", t850_target.shape)

    if t850_init.shape != t850_target.shape or t850_final.shape != t850_target.shape:
        raise ValueError("Shape mismatch between init/final/target T850 arrays.")

    # expected shape: (time, lon, lat)
    bias_init = t850_init - t850_target
    bias_final = t850_final - t850_target

    mean_bias_init = bias_init.mean(axis=0)    # (lon, lat)
    mean_bias_final = bias_final.mean(axis=0)  # (lon, lat)
    improvement = np.mean(np.abs(bias_init) - np.abs(bias_final), axis=0)

    np.save(os.path.join(data_dir, "mean_bias_init.npy"), mean_bias_init)
    np.save(os.path.join(data_dir, "mean_bias_final.npy"), mean_bias_final)
    np.save(os.path.join(data_dir, "bias_improvement.npy"), improvement)

    both = np.concatenate([mean_bias_init.ravel(), mean_bias_final.ravel()])
    vmax_bias = np.percentile(np.abs(both), 99.0)
    vmax_bias = max(float(vmax_bias), 1e-6)
    norm_bias = TwoSlopeNorm(vmin=-vmax_bias, vcenter=0.0, vmax=vmax_bias)

    vmax_improve = np.percentile(np.abs(improvement), 99.0)
    vmax_improve = max(float(vmax_improve), 1e-6)
    norm_improve = TwoSlopeNorm(vmin=-vmax_improve, vcenter=0.0, vmax=vmax_improve)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        mean_bias_init.T,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        norm=norm_bias,
    )
    axes[0].set_title("Mean initial bias (T850 init - ERA5)")
    axes[0].set_xlabel("Longitude index")
    axes[0].set_ylabel("Latitude index")

    im1 = axes[1].imshow(
        mean_bias_final.T,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        norm=norm_bias,
    )
    axes[1].set_title("Mean final bias (T850 final - ERA5)")
    axes[1].set_xlabel("Longitude index")
    axes[1].set_ylabel("Latitude index")

    im2 = axes[2].imshow(
        improvement.T,
        origin="lower",
        aspect="auto",
        cmap="BrBG",
        norm=norm_improve,
    )
    axes[2].set_title(r"Improvement in abs. bias ($|b_{init}| - |b_{final}|$)")
    axes[2].set_xlabel("Longitude index")
    axes[2].set_ylabel("Latitude index")

    cbar01 = fig.colorbar(im0, ax=axes[:2], shrink=0.9)
    cbar01.set_label("Bias [K]")

    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.9)
    cbar2.set_label("Improvement [K]")

    savefig("t850_bias_maps.png")
    plt.close(fig)

    print("Bias diagnostics:")
    print("mean_bias_init min/max :", float(np.min(mean_bias_init)), float(np.max(mean_bias_init)))
    print("mean_bias_final min/max:", float(np.min(mean_bias_final)), float(np.max(mean_bias_final)))
    print("improvement min/max    :", float(np.min(improvement)), float(np.max(improvement)))
else:
    print("Missing T850 .npy files, skipping bias maps.")

# =========================
# Figure 5: zonal-mean bias
# =========================
if all(os.path.exists(p) for p in [t850_init_path, t850_final_path, t850_target_path, lat_path]):
    t850_init = np.load(t850_init_path)
    t850_final = np.load(t850_final_path)
    t850_target = np.load(t850_target_path)
    lat_deg = np.load(lat_path)

    if t850_init.shape != t850_target.shape or t850_final.shape != t850_target.shape:
        raise ValueError("Shape mismatch between init/final/target T850 arrays.")

    # (time, lon, lat)
    bias_init = t850_init - t850_target
    bias_final = t850_final - t850_target

    # zonal mean = mean over longitude axis
    bias_init_zonal = bias_init.mean(axis=1)   # (time, lat)
    bias_final_zonal = bias_final.mean(axis=1) # (time, lat)

    # time mean
    mean_bias_init_zonal = bias_init_zonal.mean(axis=0)   # (lat,)
    mean_bias_final_zonal = bias_final_zonal.mean(axis=0) # (lat,)

    # zonal improvement in absolute bias
    abs_bias_init_zonal = np.abs(bias_init).mean(axis=1)    # (time, lat)
    abs_bias_final_zonal = np.abs(bias_final).mean(axis=1)  # (time, lat)
    mean_abs_improve_zonal = (abs_bias_init_zonal - abs_bias_final_zonal).mean(axis=0)

    np.save(os.path.join(data_dir, "mean_bias_init_zonal.npy"), mean_bias_init_zonal)
    np.save(os.path.join(data_dir, "mean_bias_final_zonal.npy"), mean_bias_final_zonal)
    np.save(os.path.join(data_dir, "mean_abs_improve_zonal.npy"), mean_abs_improve_zonal)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)

    # Left panel: zonal mean signed bias
    axes[0].plot(mean_bias_init_zonal, lat_deg, label="Initial", linewidth=2)
    axes[0].plot(mean_bias_final_zonal, lat_deg, label="Final", linewidth=2)
    axes[0].axvline(0.0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Bias [K]")
    axes[0].set_ylabel("Latitude [deg]")
    axes[0].set_title("Zonal-mean T850 bias")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right panel: zonal improvement in absolute bias
    axes[1].plot(mean_abs_improve_zonal, lat_deg, linewidth=2)
    axes[1].axvline(0.0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel(r"Improvement in |bias| [K]")
    axes[1].set_ylabel("Latitude [deg]")
    axes[1].set_title(r"Zonal improvement ($|b_{init}| - |b_{final}|$)")
    axes[1].grid(True, alpha=0.3)

    savefig("t850_zonal_bias.png")
    plt.close(fig)

    print("Saved zonal-mean bias figure.")
else:
    print("Missing files for zonal-mean bias figure, skipping.")

print("Done.")