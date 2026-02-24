import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# Paths
# ======================
CSV_PATH = "/home/gmathieu/links/scratch/results_compare/compare_dino_full_vs_era5_2days_12h.csv"
OUTDIR = Path("/home/gmathieu/links/scratch/results_compare/ppt_outputs")
OUTDIR.mkdir(exist_ok=True)

# ======================
# Load data
# ======================
df = pd.read_csv(CSV_PATH)

# ======================
# Save clean table for PPT
# ======================
cols_keep = [
    "lead_hours",

    "full_ps_rmse_hPa","full_ps_bias_hPa",
    "dino_ps_rmse_hPa","dino_ps_bias_hPa",

    "full_T_rmse_K","full_T_bias_K",
    "dino_T_rmse_K","dino_T_bias_K",

    "full_u_rmse_ms","full_u_bias_ms",
    "dino_u_rmse_ms","dino_u_bias_ms",

    "full_v_rmse_ms","full_v_bias_ms",
    "dino_v_rmse_ms","dino_v_bias_ms",
]

df_clean = df[cols_keep]
df_clean.to_csv(OUTDIR / "table_full_vs_dino_metrics.csv", index=False)

print("WROTE:", OUTDIR / "table_full_vs_dino_metrics.csv")

# ======================
# Helper plotting function
# ======================
def plot_metric(var, unit):
    plt.figure(figsize=(7,5))
    
    plt.plot(df["lead_hours"], df[f"full_{var}_rmse_{unit}"], marker="o", label="FULL RMSE")
    plt.plot(df["lead_hours"], df[f"dino_{var}_rmse_{unit}"], marker="o", label="DINO RMSE")

    plt.xlabel("Lead time (hours)")
    plt.ylabel(f"RMSE ({unit})")
    plt.title(f"{var.upper()} RMSE vs Lead Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{var}_rmse.png", dpi=200)
    plt.close()

    # Bias plot
    plt.figure(figsize=(7,5))
    plt.plot(df["lead_hours"], df[f"full_{var}_bias_{unit}"], marker="o", label="FULL Bias")
    plt.plot(df["lead_hours"], df[f"dino_{var}_bias_{unit}"], marker="o", label="DINO Bias")

    plt.xlabel("Lead time (hours)")
    plt.ylabel(f"Bias ({unit})")
    plt.title(f"{var.upper()} Bias vs Lead Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{var}_bias.png", dpi=200)
    plt.close()

# ======================
# Generate plots
# ======================
plot_metric("ps", "hPa")
plot_metric("T", "K")
plot_metric("u", "ms")
plot_metric("v", "ms")

print("ALL FIGURES WRITTEN TO:", OUTDIR)
