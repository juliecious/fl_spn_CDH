import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    set_random_seed,
    simulate_dag,
    my_simulate_linear_gaussian,
    my_simulate_general_hetero,
)
from tests.utils.benchmark_loaders import (
    load_standard_graph,
    simulate_heterogeneous_data,
)
from tests.utils.sachs_loader import load_sachs_federated
from tests.benchmarks.configs import PRODUCTION_CONFIGS, SMOKE_CONFIG


class ExperimentArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def setup_descriptive_dir(
    cfg, config_name, num_seeds, start_seed, base_dir="tests/experiments"
):
    """Creates a directory named according to the user's requested template."""
    dataset = cfg.get("model_type", "unknown")
    n = cfg.get("n", "?")
    d = cfg.get("d", "?")
    k = cfg.get("K", "?")

    # Suffix: SeedsN if Monte Carlo, else SeedN
    suffix = f"Seeds{num_seeds}" if num_seeds > 1 else f"Seed{start_seed}"

    # Build folder name: dataset_NN_Dd_KK_suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{config_name}_{dataset}_N{n}_D{d}_K{k}_{suffix}_{timestamp}"

    exp_dir = os.path.join(base_dir, folder_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Setup Logging
    log_file = os.path.join(exp_dir, "batch.log")
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Descriptive Experiment Directory Created: {exp_dir}")
    return exp_dir


def load_data(args):
    """Loads or generates data based on args."""
    logging.info(f"Loading data for model_type={args.model_type}...")

    if args.model_type == "sachs_real":
        X_splits, true_DAG_bin, c_indx = load_sachs_federated(
            args.K, n_samples_limit=None
        )
        if not isinstance(X_splits, list):
            X_splits = np.array_split(X_splits, args.K)
        return X_splits, c_indx, true_DAG_bin

    elif args.model_type in ["sachs", "asia", "alarm"]:
        true_DAG_bin = load_standard_graph(args.model_type)
        args.d = true_DAG_bin.shape[0]
        X_global, c_indx = simulate_heterogeneous_data(
            true_DAG_bin, args.K, args.n, mode="general"
        )
        c_indx = c_indx.astype(int)
        X_splits = np.array_split(X_global, args.K)
        return X_splits, c_indx, true_DAG_bin

    else:
        true_DAG_bin = simulate_dag(args.d, args.d, "ER")
        total_samples = args.n * args.K
        if args.model_type == "linear":
            X_global, c_indx = my_simulate_linear_gaussian(
                true_DAG_bin, args.K, total_samples, "gauss"
            )
        else:
            X_global, c_indx = my_simulate_general_hetero(
                true_DAG_bin, args.K, total_samples, "gauss"
            )
        c_indx = np.repeat(np.arange(args.K), args.n).reshape(-1, 1)
        X_global = (X_global - X_global.mean(0)) / (X_global.std(0) + 1e-6)
        X_splits = np.array_split(X_global, args.K)
        return X_splits, c_indx, true_DAG_bin


def run_single_experiment(config_name, seed, custom_args=None):
    """Runs a single seed of a specific configuration."""

    # 1. Resolve Configuration
    if config_name == "smoke":
        cfg_dict = SMOKE_CONFIG.copy()
    elif config_name in PRODUCTION_CONFIGS:
        cfg_dict = PRODUCTION_CONFIGS[config_name].copy()
    else:
        if custom_args:
            cfg_dict = vars(custom_args).copy()
        else:
            raise ValueError(f"Unknown config: {config_name}")

    # Override with any custom args provided
    if custom_args:
        for k, v in vars(custom_args).items():
            if v is not None:
                cfg_dict[k] = v

    args = ExperimentArgs(**cfg_dict)
    logging.info(f"Starting Seed: {seed} | Config: {config_name}")
    # logging.info(f"Params: {cfg_dict}") # Reduce log noise in batch mode

    set_random_seed(seed)

    # 2. Load Data
    X_splits, c_indx, true_DAG_bin = load_data(args)

    # 3. Run Pipeline
    runner = FedCDH(args)
    try:
        result = runner.fit(X_splits, c_indx, true_DAG_bin)
    except Exception as e:
        logging.error(f"Seed {seed} Failed: {e}", exc_info=True)
        return None

    # 4. Add Metadata
    result["config"] = config_name
    result["seed"] = seed
    result["timestamp"] = datetime.now().isoformat()
    result["n"] = args.n
    result["d"] = args.d
    result["K"] = args.K

    # Add args for context
    for k, v in cfg_dict.items():
        result[f"arg_{k}"] = v

    return result


def plot_batch_results(df, output_dir, config_name):
    """Generates a detailed 3-panel summary plot for the batch."""
    sns.set_theme(style="whitegrid")

    # metrics groups
    structure_metrics = [
        "f1_skeleton",
        "precision_skeleton",
        "recall_skeleton",
        "f1",
        "precision",
        "recall",
    ]
    shd_metrics = ["shd_skeleton", "shd"]
    efficiency_metrics = ["time_train", "time_cd", "comm_cost"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Helper to plot group
    def plot_group(metrics, ax, title, ylabel):
        valid = [m for m in metrics if m in df.columns]
        if not valid:
            return
        plot_df = df.melt(
            id_vars=["config"], value_vars=valid, var_name="Metric", value_name="Score"
        )

        sns.barplot(
            data=plot_df,
            x="Metric",
            y="Score",
            hue="Metric",
            errorbar="sd",
            capsize=0.1,
            ax=ax,
            palette="muted",
            legend=False,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    # Plot 1: Structure Quality
    plot_group(structure_metrics, axes[0], "Structure Quality", "Score (0-1)")
    axes[0].set_ylim(0, 1.05)

    # Plot 2: SHD (Error)
    plot_group(
        shd_metrics,
        axes[1],
        "Structural Hamming Distance (Lower is Better)",
        "Error Count",
    )

    # Plot 3: Efficiency
    plot_group(efficiency_metrics, axes[2], "Efficiency & Cost", "Value")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Log Scale")

    # Global Title
    # Extract params from first row (use direct columns if available)
    try:
        n_val = df.iloc[0]["n"]
        d_val = df.iloc[0]["d"]
        k_val = df.iloc[0]["K"]
        dataset_name = df.iloc[0].get(
            "arg_model_type", df.iloc[0].get("model_type", "unknown")
        )
    except KeyError:
        # Fallback to arg_ columns
        n_val = df.iloc[0].get("arg_n", "?")
        d_val = df.iloc[0].get("arg_d", "?")
        k_val = df.iloc[0].get("arg_K", "?")
        dataset_name = df.iloc[0].get("arg_model_type", "?")

    plt.suptitle(
        f"Experiment Results: {config_name} (Dataset: {dataset_name})\n(N={n_val} samples/client, D={d_val} nodes, K={k_val} clients)",
        fontsize=16,
        y=1.02,
    )

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "performance_plot.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    logging.info(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a batch of FedCDH experiments.")
    parser.add_argument("--config", type=str, help="Name of the configuration to run")
    parser.add_argument("--seed", type=int, default=0, help="Starting random seed")
    parser.add_argument(
        "--num_seeds", type=int, default=1, help="Number of seeds to run (Monte Carlo)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="tests/experiments",
        help="Base directory for outputs",
    )

    # Allow overrides
    parser.add_argument("--ci_method", type=str)
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--n", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--epochs", type=int, help="Training epochs for SPN")
    parser.add_argument("--alpha", type=float, help="Significance level for CI tests")
    parser.add_argument(
        "--num_sums", type=int, help="Number of sum nodes per scope in SPN"
    )
    parser.add_argument(
        "--num_leaves", type=int, help="Number of leaf nodes per feature in SPN"
    )
    parser.add_argument("--num_repetitions", type=int, help="Number of SPN repetitions")

    args = parser.parse_args()
    config_name = args.config if args.config else "custom"

    # 1. Resolve Configuration parameters for naming
    if config_name == "smoke":
        cfg = SMOKE_CONFIG.copy()
    elif config_name in PRODUCTION_CONFIGS:
        cfg = PRODUCTION_CONFIGS[config_name].copy()
    else:
        cfg = {}

    # Apply overrides
    for k in ["model_type", "n", "d", "K"]:
        val = getattr(args, k, None)
        if val is not None:
            cfg[k] = val

    # 2. Setup Batch Directory with descriptive name
    exp_dir = setup_descriptive_dir(
        cfg, config_name, args.num_seeds, args.seed, args.base_dir
    )

    # 3. Filter overrides for the execution engine
    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ["config", "num_seeds", "base_dir", "seed"]
    }
    override_args = ExperimentArgs(**overrides) if overrides else None

    # 4. Run Loop
    all_results = []
    seeds = list(range(args.seed, args.seed + args.num_seeds))

    logging.info(f"Starting Batch execution for {len(seeds)} seeds...")

    for seed in seeds:
        res = run_single_experiment(config_name, seed, override_args)
        if res:
            all_results.append(res)
            logging.info(
                f"Seed {seed} Complete: Skel F1={res.get('f1_skeleton'):.2f}, Dir F1={res.get('f1'):.2f}"
            )
        else:
            logging.warning(f"Seed {seed} Failed.")

    if not all_results:
        logging.error("All seeds failed.")
        sys.exit(1)

    # 5. Aggregate & Save
    df = pd.DataFrame(all_results)

    # Save Raw
    raw_path = os.path.join(exp_dir, "raw_metrics.csv")
    df.to_csv(raw_path, index=False)

    # Save Summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].agg(["mean", "std"])
    summary_path = os.path.join(exp_dir, "summary_metrics.csv")
    summary.to_csv(summary_path)

    logging.info("\n" + "=" * 50)
    logging.info("BATCH SUMMARY")
    logging.info("=" * 50)
    logging.info(f"\n{summary.transpose()}")

    # 6. Plot (Local)
    plot_batch_results(df, exp_dir, config_name)

    logging.info(
        f"Experiment Batch Complete. Results in {exp_dir}"
    )  # ... [rest of file] ...

    for seed in seeds:
        res = run_single_experiment(config_name, seed, override_args)
        if res:
            all_results.append(res)
            logging.info(
                f"Seed {seed} Complete: Skel F1={res.get('f1_skeleton'):.2f}, Dir F1={res.get('f1'):.2f}"
            )
        else:
            logging.warning(f"Seed {seed} Failed.")

    if not all_results:
        logging.error("All seeds failed.")
        sys.exit(1)

    # 4. Aggregate & Save
    df = pd.DataFrame(all_results)

    # Save Raw
    raw_path = os.path.join(exp_dir, "raw_metrics.csv")
    df.to_csv(raw_path, index=False)

    # Save Summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].agg(["mean", "std"])
    summary_path = os.path.join(exp_dir, "summary_metrics.csv")
    summary.to_csv(summary_path)

    logging.info("\n" + "=" * 50)
    logging.info("BATCH SUMMARY")
    logging.info("=" * 50)
    logging.info(f"\n{summary.transpose()}")

    # 5. Plot (Local)
    plot_batch_results(df, exp_dir, config_name)

    logging.info(f"Experiment Batch Complete. Results in {exp_dir}")
