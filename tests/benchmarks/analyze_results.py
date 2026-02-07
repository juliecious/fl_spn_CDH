import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
from datetime import datetime
import numpy as np


def extract_common_params(df):
    """Extracts common arg_ columns that have the same value across all rows."""
    arg_cols = [c for c in df.columns if c.startswith("arg_")]
    params = []
    for col in arg_cols:
        unique_vals = df[col].unique()
        if len(unique_vals) == 1:
            params.append(f"{col.replace('arg_', '')}={unique_vals[0]}")
    return ", ".join(params)


def aggregate_and_plot(metrics_dir, output_dir_base):
    """
    Reads all metrics.csv files, aggregates, and produces a 3-panel summary plot.
    """
    # 1. Load Data
    search_path = os.path.join(metrics_dir, "**", "raw_metrics.csv")
    print(f"Searching for metrics in: {search_path}")
    all_files = glob.glob(search_path, recursive=True)

    if not all_files:
        print("No raw_metrics.csv files found.")
        return

    print(f"Found {len(all_files)} result files.")
    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    if "config" not in full_df.columns:
        full_df["config"] = "custom"

    # 2. Setup Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_base, f"summary_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving analysis to: {output_dir}")

    # 3. Define Metric Groups
    structure_metrics = ["f1_skeleton", "precision_skeleton", "recall_skeleton", "f1"]
    shd_metrics = ["shd_skeleton", "shd"]
    efficiency_metrics = ["time_train", "time_cd", "comm_cost"]

    all_metrics = structure_metrics + shd_metrics + efficiency_metrics
    available_metrics = [m for m in all_metrics if m in full_df.columns]

    # 4. Aggregation
    agg_df = (
        full_df.groupby("config")[available_metrics].agg(["mean", "std"]).reset_index()
    )
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    print(agg_df)
    agg_df.to_csv(os.path.join(output_dir, "aggregated_results.csv"))

    # 5. Visualization
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Helper to plot group
    def plot_group(metrics, ax, title, ylabel):
        valid = [m for m in metrics if m in full_df.columns]
        if not valid:
            return
        plot_df = full_df.melt(
            id_vars=["config"], value_vars=valid, var_name="Metric", value_name="Score"
        )

        # Clean labels
        plot_df["Metric"] = plot_df["Metric"].str.replace("_skeleton", " (Skel)")
        plot_df["Metric"] = plot_df["Metric"].str.replace("time_", "Time ")
        plot_df["Metric"] = plot_df["Metric"].str.replace("comm_cost", "Cost (KB)")

        sns.barplot(
            data=plot_df,
            x="config",
            y="Score",
            hue="Metric",
            errorbar="sd",
            capsize=0.1,
            ax=ax,
            palette="muted",
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title=None)

    # Plot 1: Structure Quality
    plot_group(
        structure_metrics,
        axes[0],
        "Structure Quality (Higher is Better)",
        "Score (0-1)",
    )
    axes[0].set_ylim(0, 1.05)

    # Plot 2: SHD (Error)
    plot_group(
        shd_metrics,
        axes[1],
        "Structural Hamming Distance (Lower is Better)",
        "Error Count",
    )

    # Plot 3: Efficiency
    # Use Log Scale for efficiency if variance is huge
    plot_group(efficiency_metrics, axes[2], "Efficiency & Cost", "Value")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Log Scale (s or KB)")

    # Global Title
    params_str = extract_common_params(full_df)
    plt.suptitle(
        f"FedCDH Benchmark Results\nParameters: {params_str}", fontsize=16, y=1.02
    )

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "combined_benchmark_results.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_dir",
        default="tests/experiments",
        help="Root directory to search for metrics",
    )
    parser.add_argument(
        "--output_dir",
        default="tests/experiments",
        help="Base directory for summary output",
    )
    args = parser.parse_args()

    aggregate_and_plot(args.metrics_dir, args.output_dir)
