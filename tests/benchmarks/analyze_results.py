import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
from datetime import datetime


def aggregate_and_plot(metrics_dir, output_dir_base):
    """
    Reads all metrics.csv files in metrics_dir (recursively), aggregates by 'config', and plots results.
    """
    # 1. Load Data
    search_path = os.path.join(metrics_dir, "**", "metrics.csv")
    print(f"Searching for metrics in: {search_path}")
    all_files = glob.glob(search_path, recursive=True)

    if not all_files:
        print("No metrics.csv files found.")
        return

    print(f"Found {len(all_files)} result files.")
    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)

    # 2. Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_base, f"summary_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving analysis to: {output_dir}")

    # 3. Aggregate stats
    if "config" not in full_df.columns:
        full_df["config"] = "custom"

    metrics_of_interest = [
        "f1_skeleton",
        "f1",
        "precision",
        "recall",
        "comm_cost",
        "time_train",
        "time_cd",
    ]

    metrics_of_interest = [m for m in metrics_of_interest if m in full_df.columns]

    agg_df = (
        full_df.groupby("config")[metrics_of_interest]
        .agg(["mean", "std"])
        .reset_index()
    )

    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    print(agg_df)

    agg_df.to_csv(os.path.join(output_dir, "aggregated_results.csv"))

    # 4. Plot
    sns.set_theme(style="whitegrid")

    # Filter for plotting F1
    plot_metrics = [m for m in ["f1_skeleton", "f1"] if m in full_df.columns]

    if plot_metrics:
        plot_df = full_df.melt(
            id_vars=["config"],
            value_vars=plot_metrics,
            var_name="Metric",
            value_name="Score",
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=plot_df,
            x="config",
            y="Score",
            hue="Metric",
            errorbar="sd",
            capsize=0.1,
        )
        plt.title("Benchmark Results: F1 Scores")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "f1_scores.png"))
        print(f"Plot saved to {os.path.join(output_dir, 'f1_scores.png')}")


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
