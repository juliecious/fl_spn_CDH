import os
import sys
import json
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
        self.ablation_structure = True
        self.ablation_em = True
        self.ablation_orientation = "hybrid"
        self.__dict__.update(kwargs)


def run_ablation_study():
    # Setup directories
    base_dir = "tests/results/ablations"
    os.makedirs(base_dir, exist_ok=True)

    log_file = os.path.join(base_dir, "ablation_results.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.File_Watcher(log_file), logging.StreamHandler()]
        if hasattr(logging, "File_Watcher")
        else [logging.FileHandler(log_file), logging.StreamHandler()],
    )

    results = []

    # --- STUDY 1: Orientation Logic ---
    logging.info("=" * 60)
    logging.info("STUDY 1: Orientation Logic (Hybrid vs SPN vs HSIC)")
    logging.info("=" * 60)

    orient_configs = [
        {"name": "Hybrid Ensemble", "ablation_orientation": "hybrid"},
        {"name": "SPN-Only (Invariance)", "ablation_orientation": "spn"},
        {"name": "HSIC-Only (Baseline)", "ablation_orientation": "hsic"},
    ]

    for conf in orient_configs:
        args = Args(scenario="horizontal", **conf)
        res = test_fedCDH(0, args)
        res["Study"] = "Orientation"
        res["Config"] = conf["name"]
        results.append(res)

    # --- STUDY 2: Structure Learning ---
    logging.info("\n" + "=" * 60)
    logging.info("STUDY 2: Structure Learning (Spectral vs Random)")
    logging.info("=" * 60)

    struct_configs = [
        {"name": "Spectral Ordering", "ablation_structure": True},
        {"name": "Random Ordering", "ablation_structure": False},
    ]

    for conf in struct_configs:
        args = Args(scenario="horizontal", **conf)
        res = test_fedCDH(0, args)
        res["Study"] = "Structure"
        res["Config"] = conf["name"]
        results.append(res)

    # --- STUDY 3: EM Refinement (Vertical FL) ---
    logging.info("\n" + "=" * 60)
    logging.info("STUDY 3: EM weight refinement (Vertical FL)")
    logging.info("=" * 60)

    em_configs = [
        {"name": "With EM Refinement", "ablation_em": True},
        {"name": "Without EM Refinement", "ablation_em": False},
    ]

    for conf in em_configs:
        args = Args(scenario="vertical", **conf)
        res = test_fedCDH(0, args)
        res["Study"] = "EM_Refinement"
        res["Config"] = conf["name"]
        results.append(res)

    # --- Finalize Results ---
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(base_dir, "ablation_metrics.csv"), index=False)

    # --- Generate Plots ---
    generate_ablation_plots(df, base_dir)
    logging.info(f"\nAblation studies complete. Results saved to {base_dir}")


def generate_ablation_plots(df, output_dir):
    studies = df["Study"].unique()

    for study in studies:
        sub_df = df[df["Study"] == study]

        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = ["f1_skeleton", "f1", "precision", "recall"]
        labels = ["Skel F1", "DAG F1", "Precision", "Recall"]

        x = np.arange(len(metrics))
        width = 0.25

        configs = sub_df["Config"].tolist()
        for i, config in enumerate(configs):
            row = sub_df[sub_df["Config"] == config]
            vals = [row[m].values[0] for m in metrics]
            offset = (i - (len(configs) - 1) / 2) * width
            rects = ax.bar(x + offset, vals, width, label=config)

            # Label bars
            for rect in rects:
                h = rect.get_height()
                ax.annotate(
                    f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_ylabel("Score")
        ax.set_title(f"Ablation Study: {study}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ablation_{study.lower()}.png"))
        plt.close()


if __name__ == "__main__":
    run_ablation_study()
