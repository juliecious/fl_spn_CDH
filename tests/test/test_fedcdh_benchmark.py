"""
FedCDH Benchmark Suite - Scenario Comparison.

Compares 3 SPN aggregation scenarios (Horizontal/Vertical/Hybrid) with
production-quality hyperparameters to evaluate real performance of FedCDH.

Scenarios:
- Horizontal: Mixture-of-experts (sample partitioning)
- Vertical: Product-of-experts (feature partitioning)
- Hybrid: Product-then-Mixture (feature groups + sample partitioning)

Hardware: Requires CUDA-capable GPU (recommended)
Runtime: ~15-30 minutes per configuration
"""

import logging
import os
import sys
import time
from argparse import Namespace

import numpy as np
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from causallearn.search.FCMBased.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    my_simulate_general_hetero,
    my_simulate_linear_gaussian,
    set_random_seed,
    simulate_dag,
    simulate_parameter,
)

# ============================================================
# Configuration
# ============================================================

# Experiment configurations
BENCHMARK_CONFIGS = {
    "small": {
        "d": 8,
        "K": 3,
        "n": 600,
        "epochs": 50,
        "description": "Small-scale: 8 vars, 3 clients, 600 samples",
    },
    "medium": {
        "d": 10,
        "K": 3,
        "n": 1200,
        "epochs": 100,
        "description": "Medium-scale: 10 vars, 3 clients, 1200 samples (fixed: n=120/dim, adaptive perms)",
    },
    "large": {
        "d": 11,
        "K": 5,
        "n": 1000,
        "epochs": 150,
        "description": "Large-scale: 11 vars, 5 clients, 1000 samples (Sachs-like)",
    },
}

# Seeds for statistical robustness
SEEDS = [42, 123, 456, 789, 2024]  # 5 runs per configuration

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Data Generation
# ============================================================


def create_benchmark_data(d, K, n, seed, data_type="linear", sem_type="gauss"):
    """
    Create benchmark data with heterogeneity across K clients.

    Args:
        d: Number of variables
        K: Number of clients
        n: Total number of samples
        seed: Random seed
        data_type: 'linear' or 'nonlinear'
        sem_type: SEM type ('gauss', etc.)

    Returns:
        W, B, X, c_indx, choice
    """
    set_random_seed(seed)

    # Generate random DAG
    s0 = d  # Expected number of edges
    B = simulate_dag(d, s0, graph_type="ER")
    W = simulate_parameter(B)

    # Generate heterogeneous data
    if data_type == "linear":
        X, choice = my_simulate_linear_gaussian(W, K=K, n=n, sem_type=sem_type)
    else:
        X, choice = my_simulate_general_hetero(W, K=K, n=n, sem_type=sem_type)

    # Create context indices
    c_indx = np.repeat(np.arange(K), n // K).reshape(-1, 1)

    return W, B, X, c_indx, choice


def partition_data(X, c_indx, K, scenario):
    """Partition data according to federated scenario."""
    n, d = X.shape

    if scenario == "horizontal":
        samples_per_client = n // K
        X_splits = [
            X[k * samples_per_client : (k + 1) * samples_per_client, :]
            for k in range(K)
        ]
    elif scenario == "vertical":
        features_per_client = d // K
        X_splits = [
            X[:, k * features_per_client : (k + 1) * features_per_client]
            for k in range(K)
        ]
        if d % K != 0:
            X_splits[-1] = X[:, (K - 1) * features_per_client :]
    elif scenario == "hybrid":
        samples_per_client = n // K
        X_splits = []
        for k in range(K):
            X_k = X[k * samples_per_client : (k + 1) * samples_per_client, :]
            X_splits.append(X_k)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return X_splits


# ============================================================
# Benchmark Execution
# ============================================================


def run_single_experiment(
    config_name, config, scenario, data_type, seed, device="cuda"
):
    """
    Run a single benchmark experiment.

    Returns:
        Dictionary with results
    """
    d = config["d"]
    K = config["K"]
    n = config["n"]
    epochs = config["epochs"]

    logging.info(
        f"Running: {config_name} | {scenario} | {data_type} | seed={seed} | device={device}"
    )

    start_time = time.time()

    # Generate data
    W, B, X, c_indx, choice = create_benchmark_data(
        d=d, K=K, n=n, seed=seed, data_type=data_type, sem_type="gauss"
    )

    # Partition data
    X_splits = partition_data(X, c_indx, K, scenario)

    # Setup FedCDH with production parameters
    args = Namespace(
        K=K,
        d=d,
        n=n // K,
        scenario=scenario,
        model_type="synthetic",
        ci_method="spn",
        alpha=0.05,
        epochs=epochs,
        device=device,
        skip_bic=False,  # Use BIC for optimal cluster selection
    )

    fedcdh = FedCDH(args)

    # Fit model
    train_start = time.time()
    results = fedcdh.fit(X_splits, c_indx, B)
    train_time = time.time() - train_start
    total_time = time.time() - start_time

    # Extract metrics
    return {
        "config": config_name,
        "scenario": scenario,
        "data_type": data_type,
        "seed": seed,
        "d": d,
        "K": K,
        "n": n,
        "epochs": epochs,
        "device": device,
        "skeleton_f1": results.get("f1_skeleton", 0.0),
        "skeleton_precision": results.get("precision_skeleton", 0.0),
        "skeleton_recall": results.get("recall_skeleton", 0.0),
        "skeleton_shd": results.get("shd_skeleton", 0),
        "dag_f1": results.get("f1", 0.0),
        "dag_precision": results.get("precision", 0.0),
        "dag_recall": results.get("recall", 0.0),
        "dag_shd": results.get("shd", 0),
        "train_time": train_time,
        "total_time": total_time,
        "comm_cost": results.get("comm_cost", 0.0),
        "eval_dir": getattr(fedcdh, "spn_eval_dir", None),
    }


# ============================================================
# Experiment Suites
# ============================================================


def run_scenario_comparison(config_name="medium", data_type="linear", seeds=None):
    """
    Benchmark: Compare 3 SPN scenarios (H/V/Hy) on specified data type.

    Tests which aggregation strategy performs best.

    Args:
        config_name: Configuration to use (small/medium/large)
        data_type: Type of data generation (linear/nonlinear)
        seeds: Random seeds for multiple runs
    """
    if seeds is None:
        seeds = SEEDS

    config = BENCHMARK_CONFIGS[config_name]
    scenarios = ["horizontal", "vertical", "hybrid"]

    logging.info("\n" + "=" * 80)
    logging.info(f"BENCHMARK: SCENARIO COMPARISON ({config_name})")
    logging.info(config["description"])
    logging.info(f"Data type: {data_type}")
    logging.info(f"Seeds: {seeds}")
    logging.info("=" * 80)

    results = []
    run_counter = 1
    total_runs = len(scenarios) * len(seeds)

    for scenario in scenarios:
        for seed in seeds:
            logging.info(
                f"\n[Run {run_counter}/{total_runs}] Starting: {scenario} (seed={seed})"
            )

            result = run_single_experiment(
                config_name=config_name,
                config=config,
                scenario=scenario,
                data_type=data_type,
                seed=seed,
                device=DEVICE,
            )
            results.append(result)

            # Log result with eval_dir
            eval_dir = result.get("eval_dir", "N/A")
            logging.info(
                f"  {scenario:12s} seed={seed} | "
                f"Skeleton F1={result['skeleton_f1']:.3f} | "
                f"DAG F1={result['dag_f1']:.3f} | "
                f"Time={result['train_time']:.1f}s"
            )
            if eval_dir != "N/A":
                logging.info(f"  Experiment directory: {eval_dir}")
                logging.info(f"  UMAP images: {eval_dir}/umap_*.png")

            run_counter += 1

    return results


# ============================================================
# Results Analysis
# ============================================================


def create_experiment_manifest(results, output_dir="benchmark_results"):
    """
    Create a manifest file documenting all experiment directories.

    Args:
        results: List of result dictionaries
        output_dir: Output directory for manifest
    """
    manifest_path = os.path.join(output_dir, "experiment_manifest.txt")

    with open(manifest_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT MANIFEST\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Experiment {i}/{len(results)}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Configuration:  {result['config']}\n")
            f.write(f"Scenario:       {result['scenario']}\n")
            f.write(f"Data Type:      {result['data_type']}\n")
            f.write(f"Seed:           {result['seed']}\n")
            f.write(
                f"Dimensions:     d={result['d']}, K={result['K']}, n={result['n']}\n"
            )
            f.write(f"Epochs:         {result['epochs']}\n")
            f.write(f"Device:         {result['device']}\n")
            f.write(f"\nPerformance:\n")
            f.write(f"  Skeleton F1:  {result['skeleton_f1']:.3f}\n")
            f.write(f"  Skeleton SHD: {result['skeleton_shd']}\n")
            f.write(f"  DAG F1:       {result['dag_f1']:.3f}\n")
            f.write(f"  Train Time:   {result['train_time']:.1f}s\n")

            eval_dir = result.get("eval_dir", None)
            if eval_dir:
                f.write(f"\nExperiment Directory:\n")
                f.write(f"  {eval_dir}\n")
                f.write(f"\nGenerated Files:\n")
                f.write(f"  - run.log (detailed training log)\n")
                f.write(
                    f"  - umap_local_client_*.png (UMAP visualizations for each local SPN)\n"
                )
                f.write(
                    f"  - umap_global_spn.png (UMAP visualization for global SPN)\n"
                )
            else:
                f.write(f"\nExperiment Directory: Not available\n")

        f.write(f"\n{'=' * 80}\n")
        f.write("END OF MANIFEST\n")
        f.write(f"{'=' * 80}\n")

    logging.info(f"\nExperiment manifest saved to: {manifest_path}")
    return manifest_path


def analyze_results(results, output_dir="benchmark_results"):
    """
    Analyze benchmark results and generate summary statistics.
    """
    df = pd.DataFrame(results)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create experiment manifest (tracks all individual experiment directories)
    create_experiment_manifest(results, output_dir)

    # Save raw results
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"\nRaw results saved to: {csv_path}")

    # Compute summary statistics
    summary = (
        df.groupby(["config", "scenario", "data_type"])
        .agg(
            {
                "skeleton_f1": ["mean", "std"],
                "dag_f1": ["mean", "std"],
                "skeleton_shd": ["mean", "std"],
                "dag_shd": ["mean", "std"],
                "train_time": ["mean", "std"],
            }
        )
        .round(3)
    )

    # Save summary
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_path)
    logging.info(f"Summary statistics saved to: {summary_path}")

    # Print summary
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY STATISTICS")
    logging.info("=" * 80)
    logging.info(f"\n{summary}")

    # List all experiment directories
    eval_dirs = [r.get("eval_dir") for r in results if r.get("eval_dir")]
    if eval_dirs:
        logging.info("\n" + "=" * 80)
        logging.info(f"EXPERIMENT DIRECTORIES ({len(eval_dirs)} total)")
        logging.info("=" * 80)
        logging.info("Each directory contains:")
        logging.info("  - run.log: Detailed training and evaluation logs")
        logging.info("  - umap_local_client_*.png: UMAP visualizations for local SPNs")
        logging.info("  - umap_global_spn.png: UMAP visualization for global SPN")
        logging.info("\nSee experiment_manifest.txt for complete directory listing")

    return df, summary


# ============================================================
# Main Benchmark Runner
# ============================================================


def main(config_name, data_type="linear"):
    """
    Run scenario comparison benchmark.

    Args:
        config_name: Configuration to use (small/medium/large)
        data_type: Type of data generation (linear/nonlinear)
    """
    # Create output directory
    os.makedirs("benchmark_results", exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("benchmark_results/benchmark.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info("\n" + "=" * 80)
    logging.info("FEDCDH BENCHMARK SUITE - SCENARIO COMPARISON")
    logging.info("=" * 80)
    logging.info(f"Configuration: {config_name}")
    logging.info(f"Data type: {data_type}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logging.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    logging.info(f"Seeds: {SEEDS}")
    logging.info("=" * 80)

    overall_start = time.time()

    # Run scenario comparison benchmark
    all_results = run_scenario_comparison(
        config_name=config_name, data_type=data_type, seeds=SEEDS
    )

    overall_time = time.time() - overall_start

    # Analyze results
    df, summary = analyze_results(all_results, output_dir="benchmark_results")

    # Final summary
    logging.info("\n" + "=" * 80)
    logging.info("BENCHMARK COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Total experiments: {len(all_results)}")
    logging.info(f"Total time: {overall_time / 60:.1f} minutes")
    logging.info(f"\nResults saved to: benchmark_results/")
    logging.info(f"  - benchmark_results.csv (raw metrics for all runs)")
    logging.info(f"  - summary_statistics.csv (aggregated statistics)")
    logging.info(f"  - experiment_manifest.txt (directory listing with parameters)")
    logging.info(f"  - benchmark.log (complete execution log)")
    logging.info(f"\nIndividual experiment directories:")
    logging.info(f"  Each experiment has its own eval/ directory containing:")
    logging.info(f"    - run.log (detailed training/evaluation log)")
    logging.info(f"    - umap_local_client_*.png (local SPN visualizations)")
    logging.info(f"    - umap_global_spn.png (global SPN visualization)")
    logging.info(f"\n  See experiment_manifest.txt for full directory paths")
    logging.info("=" * 80)

    return df, summary


if __name__ == "__main__":
    df, summary = main(config_name="small", data_type="linear")
