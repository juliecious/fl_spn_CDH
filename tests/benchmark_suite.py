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
import seaborn as sns
import numpy as np

# Set Seaborn theme
sns.set_theme(style="whitegrid")

# Import the test function from the local tests directory
from tests.TestFedCDH import test_fedCDH


class Args:
    # ... (Args implementation remains same)
    def __init__(self, **kwargs):
        self.N = 1
        self.d = 11
        self.K = 2
        self.n = 30
        self.model_type = "sachs"
        self.ci_method = "spn"
        self.scenario = "horizontal"
        self.__dict__.update(kwargs)


def run_benchmarks():
    # ... (run_benchmarks setup remains same)
    configs = [
        {
            "name": "KCI (Real Sachs)",
            "args": {
                "ci_method": "kci",
                "scenario": "horizontal",
                "model_type": "sachs_real",
            },
        },
        {
            "name": "Voting-FedPC (Real Sachs)",
            "args": {
                "ci_method": "voting_pc",
                "scenario": "horizontal",
                "model_type": "sachs_real",
                "n": 853,
                "d": 11,
            },
        },
        {
            "name": "FedSPN (Real Sachs - Horizontal)",
            "args": {
                "ci_method": "spn",
                "scenario": "horizontal",
                "model_type": "sachs_real",
                "ablation_orientation": "mi_only",
                "n": 853,
                "d": 11,
            },
        },
        {
            "name": "FedSPN (Real Sachs - Vertical)",
            "args": {
                "ci_method": "spn",
                "scenario": "vertical",
                "model_type": "sachs_real",
                "ablation_orientation": "mi_only",
                "n": 853,
                "d": 11,
            },
        },
        {
            "name": "FedSPN (Real Sachs - Hybrid)",
            "args": {
                "ci_method": "spn",
                "scenario": "hybrid",
                "model_type": "sachs_real",
                "ablation_orientation": "mi_only",
                "n": 853,
                "d": 11,
            },
        },
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

    num_seeds = 5
    print(f"{'Method':<20} | Processing {num_seeds} seeds...")

    for config in configs:
        method_name = config["name"]
        print(f"Running {method_name}...")

        seed_results = []
        for seed in range(num_seeds):
            args = Args(**config["args"])
            try:
                # Run instance 'seed'
                # print(f"  Seed {seed}...", end="\r")
                res = test_fedCDH(seed, args)
                seed_results.append(res)
            except Exception as e:
                print(f"  Failed Seed {seed}: {e}")

        if not seed_results:
            continue

        # Aggregate results
        agg_row = {"Method": method_name}
        for m in metrics_order:
            values = [float(r.get(m, 0.0)) for r in seed_results]
            agg_row[m] = np.mean(values)
            agg_row[f"{m}_std"] = np.std(values)

        results.append(agg_row)

    if not results:
        print("No results collected.")
        return

    # Create DataFrame for nice printing
    df = pd.DataFrame(results)

    # Print Table
    print("\n\n" + "=" * 140)
    print("FINAL BENCHMARK RESULTS (Mean ± Std over 5 seeds)")
    print("=" * 140)

    header = (
        f"| {'Method':<20} | " + " | ".join([f"{m:<22}" for m in metrics_order]) + " |"
    )
    print(header)
    print("|" + "-" * 22 + "|" + "|".join(["-" * 24 for _ in metrics_order]) + "|")

    for _, row in df.iterrows():
        values = []
        for m in metrics_order:
            mean = row[m]
            std = row[f"{m}_std"]
            val_str = f"{mean:.2f} ± {std:.2f}"
            values.append(f"{val_str:<22}")
        print(f"| {row['Method']:<20} | " + " | ".join(values) + " |")

    print("=" * 140)

    # Generate Plot
    params = {
        "n": Args().n,
        "d": Args().d,
        "K": Args().K,
        "model_type": Args().model_type,
        "seeds": num_seeds,
    }
    plot_results(df, params)


def plot_results(df, params):
    metrics_f1 = ["f1_skeleton", "f1"]
    labels_f1 = ["Skel F1", "DAG F1"]

    metrics_pr = ["precision_skeleton", "recall_skeleton", "precision", "recall"]
    labels_pr = ["Skel Prec", "Skel Rec", "DAG Prec", "DAG Rec"]

    metrics_cost = ["comm_cost"]
    labels_cost = ["Comm Cost (KB)"]

    methods = df["Method"].tolist()
    # Professional color palette
    palette = sns.color_palette("muted", len(methods))
    width = 0.2

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(20, 8),
        gridspec_kw={"width_ratios": [2, 3, 1]},
    )

    # Add Suptitle with Parameters
    param_str = f"n={params['n']}, d={params['d']}, K={params['K']}, Model={params['model_type']}, Seeds={params['seeds']}"
    fig.suptitle(
        f"Federated Causal Discovery Performance Analysis\n{param_str}",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    # --- Plot 1: F1 Scores ---
    x_f1 = np.arange(len(metrics_f1))
    for i, method in enumerate(methods):
        row = df[df["Method"] == method]
        vals = [row[m].values[0] for m in metrics_f1]
        errs = [row[f"{m}_std"].values[0] for m in metrics_f1]

        offset = (i - (len(methods) - 1) / 2) * width
        rects = ax1.bar(
            x_f1 + offset,
            vals,
            width,
            yerr=errs,
            capsize=5,
            label=method,
            color=palette[i],
            edgecolor="black",
            alpha=0.8,
            error_kw={"linestyle": "--"},
        )

        for rect, v in zip(rects, vals):
            h = rect.get_height()
            ax1.annotate(
                f"{v:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h + (0.02 if h > 0 else 0)),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax1.set_ylabel("F1 Score", fontsize=14, fontweight="bold")
    ax1.set_title("Discovery Accuracy", fontsize=16, fontweight="bold")
    ax1.set_xticks(x_f1)
    ax1.set_xticklabels(labels_f1, fontsize=12)
    ax1.set_ylim(0, 1.1)

    # --- Plot 2: Precision & Recall ---
    x_pr = np.arange(len(metrics_pr))
    for i, method in enumerate(methods):
        row = df[df["Method"] == method]
        vals = [row[m].values[0] for m in metrics_pr]
        errs = [row[f"{m}_std"].values[0] for m in metrics_pr]

        offset = (i - (len(methods) - 1) / 2) * width
        rects = ax2.bar(
            x_pr + offset,
            vals,
            width,
            yerr=errs,
            capsize=5,
            label=method,
            color=palette[i],
            edgecolor="black",
            alpha=0.8,
            error_kw={"linestyle": "--"},
        )

        for rect, v in zip(rects, vals):
            h = rect.get_height()
            ax2.annotate(
                f"{v:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h + 0.02),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax2.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax2.set_title("Precision & Recall", fontsize=16, fontweight="bold")
    ax2.set_xticks(x_pr)
    ax2.set_xticklabels(labels_pr, fontsize=12)
    ax2.set_ylim(0, 1.1)

    # --- Plot 3: Communication Cost ---
    x_cost = np.arange(len(metrics_cost))
    for i, method in enumerate(methods):
        row = df[df["Method"] == method]
        vals = [row[m].values[0] for m in metrics_cost]
        errs = [row[f"{m}_std"].values[0] for m in metrics_cost]

        offset = (i - (len(methods) - 1) / 2) * width
        rects = ax3.bar(
            x_cost + offset,
            vals,
            width,
            yerr=errs,
            capsize=5,
            label=method,
            color=palette[i],
            edgecolor="black",
            alpha=0.8,
            error_kw={"linestyle": "--"},
        )

        for rect, v in zip(rects, vals):
            h = rect.get_height()
            ax3.annotate(
                f"{v:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax3.set_ylabel("Communication Cost (KB)", fontsize=14, fontweight="bold")
    ax3.set_title("Efficiency", fontsize=16, fontweight="bold")
    ax3.set_xticks(x_cost)
    ax3.set_xticklabels(labels_cost, fontsize=12)

    # --- Figure-Level Legend at Bottom ---
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(methods),
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
    )

    output_dir = "tests/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_plot.png")

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(output_path, dpi=300)
    print(f"\nBeautified plot saved to {output_path}")


if __name__ == "__main__":
    run_benchmarks()
