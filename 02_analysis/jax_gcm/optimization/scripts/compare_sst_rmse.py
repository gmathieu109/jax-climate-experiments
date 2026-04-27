#!/usr/bin/env python3
"""
Script to load all SST optimization runs and compare their final RMSE values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Directory containing optimization runs
OPTIM_DIR = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/optimisation"


def extract_config_from_dirname(dirname: str) -> dict:
    """Parse directory name to extract configuration parameters."""
    config = {}
    
    # Expected format: delta_sst_{days}d_steps{n_steps}_lr{lr}_{reg}_{scope}
    # Example: delta_sst_5d_steps150_lr0p0001_noreg_global
    
    parts = dirname.split("_")
    
    # Extract days
    for part in parts:
        if part.endswith("d"):
            config["days"] = int(part[:-1])
        elif part.startswith("steps"):
            config["n_steps"] = int(part[5:])
        elif part.startswith("lr"):
            lr_str = part[2:].replace("p", ".")
            config["lr"] = float(lr_str)
        elif part == "reg":
            config["regularization"] = True
        elif part == "noreg":
            config["regularization"] = False
        elif part in ["global", "nh30"]:
            config["scope"] = part
    
    return config


def load_sst_results() -> pd.DataFrame:
    """
    Load metrics from all SST optimization directories.
    
    Returns:
        DataFrame with columns: dirname, days, n_steps, lr, regularization, 
                               scope, rmse_initial, rmse_final, rmse_improvement
    """
    results = []
    
    # Find all delta_sst directories
    if not os.path.exists(OPTIM_DIR):
        print(f"Directory not found: {OPTIM_DIR}")
        return pd.DataFrame()
    
    sst_dirs = [d for d in os.listdir(OPTIM_DIR) if d.startswith("delta_sst_")]
    
    print(f"Found {len(sst_dirs)} SST optimization runs")
    
    for dirname in sorted(sst_dirs):
        dirpath = os.path.join(OPTIM_DIR, dirname)
        metrics_file = os.path.join(dirpath, "metrics.npy")
        
        if not os.path.exists(metrics_file):
            print(f"  ⚠ No metrics.npy in {dirname}")
            continue
        
        try:
            # Load metrics dictionary
            metrics = np.load(metrics_file, allow_pickle=True).item()
            
            # Extract RMSE values
            rmse_values = metrics.get("rmse", [])
            if not rmse_values:
                print(f"  ⚠ No RMSE data in {dirname}")
                continue
            
            rmse_initial = rmse_values[0]
            rmse_final = rmse_values[-1]
            rmse_improvement = ((rmse_initial - rmse_final) / rmse_initial) * 100
            
            # Parse configuration from directory name
            config = extract_config_from_dirname(dirname)
            
            result = {
                "dirname": dirname,
                "rmse_initial": rmse_initial,
                "rmse_final": rmse_final,
                "rmse_improvement_%": rmse_improvement,
                **config
            }
            results.append(result)
            
        except Exception as e:
            print(f"  ✗ Error loading {dirname}: {e}")
    
    df = pd.DataFrame(results)
    return df


def print_comparison_table(df: pd.DataFrame):
    """Print a nicely formatted comparison table."""
    if df.empty:
        print("No results to display")
        return
    
    print("\n" + "="*120)
    print("SST OPTIMIZATION RUN COMPARISON - RMSE")
    print("="*120)
    
    # Sort by final RMSE (best first)
    df_sorted = df.sort_values("rmse_final")
    
    display_df = df_sorted[[
        "dirname", 
        "days", 
        "n_steps", 
        "lr", 
        "regularization", 
        "scope",
        "rmse_initial",
        "rmse_final",
        "rmse_improvement_%"
    ]].copy()
    
    # Format float columns
    display_df["rmse_initial"] = display_df["rmse_initial"].apply(lambda x: f"{x:.4f}")
    display_df["rmse_final"] = display_df["rmse_final"].apply(lambda x: f"{x:.4f}")
    display_df["rmse_improvement_%"] = display_df["rmse_improvement_%"].apply(lambda x: f"{x:.1f}%")
    
    print(display_df.to_string(index=False))
    print("="*120 + "\n")


def create_run_label(row) -> str:
    """Create a descriptive label for a run."""
    lr_str = f"{row['lr']:.0e}" if row['lr'] < 0.001 else f"{row['lr']:.4f}"
    reg_str = "reg" if row.get('regularization', False) else "noreg"
    scope_str = row.get('scope', 'global')
    return f"lr={lr_str}, {row['days']}d, {row['n_steps']}s"


def plot_rmse_comparison(df: pd.DataFrame, output_file: str = None):
    """Create a visualization comparing RMSE across runs."""
    if df.empty:
        print("No data to plot")
        return
    
    df_sorted = df.sort_values("rmse_final")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Create labels for the sorted runs
    run_labels = [create_run_label(row) for _, row in df_sorted.iterrows()]
    
    # Plot 1: RMSE comparison (initial vs final)
    ax = axes[0, 0]
    x = np.arange(len(df_sorted))
    width = 0.35
    ax.bar(x - width/2, df_sorted["rmse_initial"], width, label="Initial RMSE", alpha=0.8)
    ax.bar(x + width/2, df_sorted["rmse_final"], width, label="Final RMSE", alpha=0.8)
    ax.set_xlabel("Run (sorted by final RMSE)")
    ax.set_ylabel("RMSE [K]")
    ax.set_title("Initial vs Final RMSE")
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=45, ha='right', fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    
    # Plot 2: RMSE improvement percentage
    ax = axes[0, 1]
    colors = ["green" if x > 0 else "red" for x in df_sorted["rmse_improvement_%"]]
    ax.bar(range(len(df_sorted)), df_sorted["rmse_improvement_%"], color=colors, alpha=0.7)
    ax.set_xlabel("Run (sorted by final RMSE)")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("RMSE Improvement %")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(run_labels, rotation=45, ha='right', fontsize=7)
    
    # Plot 3: Final RMSE vs Learning Rate
    ax = axes[1, 0]
    for scope in df_sorted["scope"].unique():
        mask = df_sorted["scope"] == scope
        subset = df_sorted[mask]
        labels = [create_run_label(row) for _, row in subset.iterrows()]
        scatter = ax.scatter(subset["lr"], subset["rmse_final"], 
                           label=f"Scope: {scope}", s=100, alpha=0.7)
        
        # Add text labels for each point
        for i, label in enumerate(labels):
            ax.annotate(label, 
                       (subset["lr"].iloc[i], subset["rmse_final"].iloc[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=6, alpha=0.8, rotation=30)
    
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Final RMSE [K]")
    ax.set_title("Final RMSE vs Learning Rate")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Final RMSE vs Number of Steps
    ax = axes[1, 1]
    for days in df_sorted["days"].unique():
        mask = df_sorted["days"] == days
        subset = df_sorted[mask]
        labels = [create_run_label(row) for _, row in subset.iterrows()]
        scatter = ax.scatter(subset["n_steps"], subset["rmse_final"], 
                           label=f"Days: {days}", s=100, alpha=0.7)
        
        # Add text labels for each point
        for i, label in enumerate(labels):
            ax.annotate(label, 
                       (subset["n_steps"].iloc[i], subset["rmse_final"].iloc[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=6, alpha=0.8, rotation=30)
    
    ax.set_xlabel("Number of Optimization Steps")
    ax.set_ylabel("Final RMSE [K]")
    ax.set_title("Final RMSE vs Optimization Steps")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_file}")
    
    plt.show()


def plot_convergence_curves(df: pd.DataFrame, n_top: int = 5, output_file: str = None):
    """Plot convergence curves for the top N runs."""
    if df.empty:
        print("No data to plot")
        return
    
    # Sort by final RMSE and take top N
    df_top = df.sort_values("rmse_final").head(n_top)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for _, row in df_top.iterrows():
        dirpath = os.path.join(OPTIM_DIR, row["dirname"])
        metrics_file = os.path.join(dirpath, "metrics.npy")
        
        try:
            metrics = np.load(metrics_file, allow_pickle=True).item()
            rmse_values = metrics.get("rmse", [])
            
            label = create_run_label(row)
            ax.plot(rmse_values, marker="o", label=label, linewidth=2, markersize=4)
        except Exception as e:
            print(f"Error plotting {row['dirname']}: {e}")
    
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("RMSE [K]")
    ax.set_title(f"Convergence Curves - Top {n_top} Runs (Best Final RMSE)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved convergence plot to: {output_file}")
    
    plt.show()


def print_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    if df.empty:
        return
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total runs: {len(df)}")
    print(f"\nFinal RMSE:")
    print(f"  Min:  {df['rmse_final'].min():.4f} K")
    print(f"  Max:  {df['rmse_final'].max():.4f} K")
    print(f"  Mean: {df['rmse_final'].mean():.4f} K")
    print(f"  Std:  {df['rmse_final'].std():.4f} K")
    
    print(f"\nImprovement %:")
    print(f"  Min:  {df['rmse_improvement_%'].min():.1f} %")
    print(f"  Max:  {df['rmse_improvement_%'].max():.1f} %")
    print(f"  Mean: {df['rmse_improvement_%'].mean():.1f} %")
    
    # Group by different parameters
    print(f"\n\nRMSE by Window Length:")
    for days in sorted(df["days"].unique()):
        subset = df[df["days"] == days]["rmse_final"]
        print(f"  {days} days: mean={subset.mean():.4f}, min={subset.min():.4f}, max={subset.max():.4f}")
    
    print(f"\nRMSE by Learning Rate:")
    for lr in sorted(df["lr"].unique()):
        subset = df[df["lr"] == lr]["rmse_final"]
        print(f"  lr={lr:.6f}: mean={subset.mean():.4f}, min={subset.min():.4f}, max={subset.max():.4f}")
    
    print("="*60 + "\n")


def main():
    print("Loading SST optimization results...")
    df = load_sst_results()
    
    if df.empty:
        print("No results found!")
        return
    
    print_comparison_table(df)
    print_statistics(df)
    
    # Create output directory for plots if needed
    plot_dir = os.path.join(OPTIM_DIR, "comparison_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate plots
    plot_rmse_comparison(df, output_file=os.path.join(plot_dir, "rmse_comparison.png"))
    plot_convergence_curves(df, n_top=5, output_file=os.path.join(plot_dir, "convergence_curves.png"))
    
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
