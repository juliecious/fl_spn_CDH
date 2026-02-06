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


def setup_batch_dir(config_name, base_dir="tests/experiments"):
    """Creates a unique directory for the batch experiment and sets up logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{config_name}_batch_{timestamp}"
    exp_dir = os.path.join(base_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Setup Logging
    log_file = os.path.join(exp_dir, "batch.log")

    # Clear any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Batch Experiment Directory Created: {exp_dir}")
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
        cfg_dict = SMOKE_CONFIG
    elif config_name in PRODUCTION_CONFIGS:
        cfg_dict = PRODUCTION_CONFIGS[config_name]
    else:
        if custom_args:
            cfg_dict = vars(custom_args)
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

    return result


def plot_batch_results(df, output_dir, config_name):
    """Generates a summary plot for the batch."""
    sns.set_theme(style="whitegrid")

    metrics = ["f1_skeleton", "f1", "precision", "recall"]
    existing_metrics = [m for m in metrics if m in df.columns]

    if not existing_metrics:
        return

    # Melt for Seaborn
    plot_df = df.melt(
        id_vars=["config"],
        value_vars=existing_metrics,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df, x="Metric", y="Score", errorbar="sd", capsize=0.1, palette="muted"
    )
    plt.title(f"Performance Distribution: {config_name}\n(N={len(df)} seeds)")
    plt.ylim(0, 1.05)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "performance_plot.png")
    plt.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a batch of FedCDH experiments.")
    parser.add_argument("--config", type=str, help="Name of the configuration to run")
    parser.add_argument(
        "--num_seeds", type=int, default=5, help="Number of seeds to run (Monte Carlo)"
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

    args = parser.parse_args()

    config_name = args.config if args.config else "custom"

    # 1. Setup Batch Directory
    exp_dir = setup_batch_dir(config_name, args.base_dir)

    # 2. Filter overrides
    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ["config", "num_seeds", "base_dir"]
    }
    override_args = ExperimentArgs(**overrides) if overrides else None

    # 3. Run Loop
    all_results = []
    seeds = list(range(args.num_seeds))

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

    # 5. Plot
    plot_batch_results(df, exp_dir, config_name)

    logging.info(f"Experiment Batch Complete. Results in {exp_dir}")
