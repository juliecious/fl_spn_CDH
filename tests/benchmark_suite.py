import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the test function
from tests.TestFedCDH import test_fedCDH


class Args:
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
                row[m] = float(val) if val is not None else 0.0

            results.append(row)
        except Exception as e:
            print(f"Failed {config['name']}: {e}")

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
        values = [f"{row[m]:<18.4f}" for m in metrics_order]
        print(f"| {row['Method']:<20} | " + " | ".join(values) + " |")

    print("=" * 120)

    # Generate Plot
    plot_results(df)


def plot_results(df):
    metrics_to_plot = ["f1_skeleton", "f1", "shd_skeleton", "shd"]
    labels = ["Skel F1", "DAG F1", "Skel SHD", "DAG SHD"]

    x = np.arange(len(metrics_to_plot))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    methods = df["Method"].tolist()
    # Colors: KCI=Blue, FedSPN(H)=Orange, FedSPN(V)=Green, FedSPN(Hyb)=Red
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, method in enumerate(methods):
        vals = []
        for m in metrics_to_plot:
            vals.append(df[df["Method"] == method][m].values[0])

        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=method, color=colors[i])

        # Label bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Score / Value")
    ax.set_title("Federated Causal Discovery Performance (FedSPN vs KCI)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    output_dir = "tests/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_plot.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    run_benchmarks()
