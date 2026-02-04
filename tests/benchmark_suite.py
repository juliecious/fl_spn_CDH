import sys
import os

# Fix OpenMP and Threading issues before importing any heavy libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Prioritize local project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the test function from the local tests directory
from tests.TestFedCDH import test_fedCDH


class Args:
    # ... (Args implementation remains same)
    def __init__(self, **kwargs):
        self.N = 1
        self.d = 5
        self.K = 2
        self.n = 200
        self.model_type = "general"
        self.ci_method = "spn"
        self.scenario = "horizontal"
        self.__dict__.update(kwargs)


def run_benchmarks():
    # ... (run_benchmarks setup remains same)
    configs = [
        {
            "name": "KCI (Horizontal)",
            "args": {"ci_method": "kci", "scenario": "horizontal"},
        },
        {
            "name": "FedSPN (Horizontal)",
            "args": {"ci_method": "spn", "scenario": "horizontal"},
        },
        {
            "name": "FedSPN (Vertical)",
            "args": {"ci_method": "spn", "scenario": "vertical"},
        },
        {"name": "FedSPN (Hybrid)", "args": {"ci_method": "spn", "scenario": "hybrid"}},
    ]

    metrics_order = [
        "f1_skeleton",
        "precision_skeleton",
        "recall_skeleton",
        "shd_skeleton",
        "f1",
        "precision",
        "recall",
        "shd",
        "time_train",
        "time_cd",
        "comm_cost",
    ]

    results = []

    print(f"{'Method':<20} | Processing...")

    for config in configs:
        args = Args(**config["args"])
        print(f"Running {config['name']}...")

        try:
            # Run instance 0
            res = test_fedCDH(0, args)

            row = {"Method": config["name"]}
            for m in metrics_order:
                val = res.get(m, 0.0)
                # Handle None or non-float types safely
                try:
                    row[m] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    row[m] = 0.0

            results.append(row)
        except Exception as e:
            print(f"Failed {config['name']}: {e}")

    if not results:
        print("No results collected.")
        return

    # Create DataFrame for nice printing
    df = pd.DataFrame(results)

    # Print Table
    print("\n\n" + "=" * 120)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 120)

    header = (
        f"| {'Method':<20} | " + " | ".join([f"{m:<18}" for m in metrics_order]) + " |"
    )
    print(header)
    print("|" + "-" * 22 + "|" + "|".join(["-" * 20 for _ in metrics_order]) + "|")

    for _, row in df.iterrows():
        # Safely format each value, ensuring it's a float
        values = []
        for m in metrics_order:
            val = row[m]
            val_str = f"{float(val):<18.4f}" if val is not None else f"{0.0:<18.4f}"
            values.append(val_str)
        print(f"| {row['Method']:<20} | " + " | ".join(values) + " |")

    print("=" * 120)

    # Generate Plot
    plot_results(df)


def plot_results(df):
    metrics_f1 = ["f1_skeleton", "f1"]
    labels_f1 = ["Skel F1", "DAG F1"]
    metrics_shd = ["shd_skeleton", "shd"]
    labels_shd = ["Skel SHD", "DAG SHD"]
    metrics_cost = ["comm_cost"]
    labels_cost = ["Comm Cost (KB)"]

    methods = df["Method"].tolist()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    width = 0.2

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [2, 2, 1]}
    )

    # --- Plot 1: F1 Scores ---
    x_f1 = np.arange(len(metrics_f1))
    for i, method in enumerate(methods):
        vals = [df[df["Method"] == method][m].values[0] for m in metrics_f1]
        offset = (i - (len(methods) - 1) / 2) * width
        rects = ax1.bar(x_f1 + offset, vals, width, label=method, color=colors[i])
        for rect in rects:
            h = rect.get_height()
            ax1.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax1.set_ylabel("Score")
    ax1.set_title("Discovery Accuracy (F1)")
    ax1.set_xticks(x_f1)
    ax1.set_xticklabels(labels_f1)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.legend(loc="upper left", fontsize="x-small")

    # --- Plot 2: SHD ---
    x_shd = np.arange(len(metrics_shd))
    for i, method in enumerate(methods):
        vals = [df[df["Method"] == method][m].values[0] for m in metrics_shd]
        offset = (i - (len(methods) - 1) / 2) * width
        rects = ax2.bar(x_shd + offset, vals, width, label=method, color=colors[i])
        for rect in rects:
            h = rect.get_height()
            ax2.annotate(
                f"{int(h)}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax2.set_ylabel("Distance (Lower is Better)")
    ax2.set_title("Structural Hamming Distance")
    ax2.set_xticks(x_shd)
    ax2.set_xticklabels(labels_shd)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # --- Plot 3: Communication Cost ---
    x_cost = np.arange(len(metrics_cost))
    for i, method in enumerate(methods):
        vals = [df[df["Method"] == method][m].values[0] for m in metrics_cost]
        offset = (i - (len(methods) - 1) / 2) * width
        rects = ax3.bar(x_cost + offset, vals, width, label=method, color=colors[i])
        for rect in rects:
            h = rect.get_height()
            ax3.annotate(
                f"{h:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax3.set_ylabel("KB")
    ax3.set_title("Efficiency (Comm Cost)")
    ax3.set_xticks(x_cost)
    ax3.set_xticklabels(labels_cost)
    ax3.grid(axis="y", linestyle="--", alpha=0.7)

    output_dir = "tests/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_plot.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    run_benchmarks()
