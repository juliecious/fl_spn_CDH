import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import argparse
from datetime import datetime


def extract_common_params(df):
    """Extracts common arg_ columns that have the same value across all rows."""
    arg_cols = [c for c in df.columns if c.startswith("arg_")]
    params = []
    for col in arg_cols:
        try:
            unique_vals = df[col].unique()
            if len(unique_vals) == 1:
                params.append(f"{col.replace('arg_', '')}={unique_vals[0]}")
        except:
            continue
    return ", ".join(params)


def aggregate_and_plot_final(metrics_dir, output_path):
    """
    Produces a 4-panel comparison plot across methods with legend at the bottom.
    """
    # 1. Load Data
    search_path = os.path.join(metrics_dir, "**", "raw_metrics.csv")
    print(f"Searching for metrics in: {search_path}")
    all_files = glob.glob(search_path, recursive=True)

    if not all_files:
        print("No raw_metrics.csv files found.")
        return

    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            df_list.append(temp_df)
        except:
            continue

    full_df = pd.concat(df_list, ignore_index=True)

    # Map long names to cleaner display names
    name_map = {
        "fedspn_horizontal_synthetic": "FedSPN (H)",
        "fedspn_vertical_synthetic": "FedSPN (V)",
        "fedspn_hybrid_synthetic": "FedSPN (Hy)",
        "kci_synthetic": "Oracle (FisherZ)",
        "voting_synthetic": "Voting (Baseline)",
    }
    full_df["Method"] = full_df["config"].map(lambda x: name_map.get(x, x))

    # 2. Setup Plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(24, 8))

    # Subplot 1: Skeleton Scores
    skel_vars = ["f1_skeleton", "precision_skeleton", "recall_skeleton"]
    df_skel = full_df.melt(
        id_vars=["Method"], value_vars=skel_vars, var_name="Metric", value_name="Score"
    )
    df_skel["Metric"] = df_skel["Metric"].str.replace("_skeleton", "").str.capitalize()
    sns.barplot(
        data=df_skel, x="Metric", y="Score", hue="Method", errorbar="sd", ax=axes[0]
    )
    axes[0].set_title("Skeleton Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].get_legend().remove()

    # Subplot 2: DAG Scores
    dag_vars = ["f1", "precision", "recall"]
    df_dag = full_df.melt(
        id_vars=["Method"], value_vars=dag_vars, var_name="Metric", value_name="Score"
    )
    df_dag["Metric"] = df_dag["Metric"].str.capitalize()
    sns.barplot(
        data=df_dag, x="Metric", y="Score", hue="Method", errorbar="sd", ax=axes[1]
    )
    axes[1].set_title("DAG Accuracy (Orientation)", fontsize=14, fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].get_legend().remove()

    # Subplot 3: SHD
    shd_vars = ["shd_skeleton", "shd"]
    df_shd = full_df.melt(
        id_vars=["Method"],
        value_vars=shd_vars,
        var_name="Metric",
        value_name="Distance",
    )
    df_shd["Metric"] = df_shd["Metric"].replace({"shd_skeleton": "Skel", "shd": "DAG"})
    sns.barplot(
        data=df_shd, x="Metric", y="Distance", hue="Method", errorbar="sd", ax=axes[2]
    )
    axes[2].set_title("SHD (Lower is Better)", fontsize=14, fontweight="bold")
    axes[2].get_legend().remove()

    # Subplot 4: Efficiency
    eff_vars = ["time_cd", "comm_cost"]
    df_eff = full_df.melt(
        id_vars=["Method"], value_vars=eff_vars, var_name="Metric", value_name="Value"
    )
    df_eff["Metric"] = df_eff["Metric"].replace(
        {"time_cd": "Time (s)", "comm_cost": "Cost (KB)"}
    )
    sns.barplot(
        data=df_eff, x="Metric", y="Value", hue="Method", errorbar="sd", ax=axes[3]
    )
    axes[3].set_title("Efficiency & Overhead", fontsize=14, fontweight="bold")
    axes[3].set_yscale("log")
    axes[3].get_legend().remove()

    # Global Legend at Bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=12,
    )

    # Title
    params = extract_common_params(full_df)
    plt.suptitle(
        f"Multi-Method Benchmark: Linear Gaussian Case\nParams: {params}",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.85)  # Room for legend and title
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Final aggregated plot saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_dir",
        default="tests/experiments",
        help="Root directory to search for raw_metrics.csv",
    )
    parser.add_argument(
        "--output_dir", default="tests/experiments", help="Where to save the plot"
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern to filter folders (e.g. '*sachs_N10*')",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_path = os.path.join(args.metrics_dir, args.pattern)
    output_file = os.path.join(args.output_dir, f"aggregated_report_{timestamp}.png")

    aggregate_and_plot_final(search_path, output_file)
