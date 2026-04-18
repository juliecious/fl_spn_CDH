"""
FedCDH Benchmark Suite - Scenario Comparison.

Compares 3 SPN aggregation scenarios (Horizontal/Vertical/Hybrid) with
production-quality hyperparameters to evaluate real performance of FedCDH.

Scenarios:
- Horizontal: Mixture-of-experts (sample partitioning)
- Vertical: Product-of-experts (feature partitioning)
- Hybrid: Mixture-then-Product ✅ (Algorithm 1 automatic feature grouping)

Hardware: GPU-enabled (CUDA/MPS) recommended for faster training
Runtime: ~10-20 minutes per configuration (GPU), ~30-60 minutes (CPU)

Data:
- quick/small/medium/large configs: Synthetic data (linear or nonlinear)
- sachs config: Real Sachs protein signaling dataset (loaded via sachs_loader.py)

Updated: April 18, 2026 - Fixed Sachs config to load real dataset
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
    "quick": {
        "d": 5,
        "K": 2,
        "n": 200,
        "epochs": 20,
        "description": "Quick smoke test: 5 vars, 2 clients, 200 samples",
    },
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
        "description": "Medium-scale: 10 vars, 3 clients, 1200 samples",
    },
    "large": {
        "d": 11,
        "K": 5,
        "n": 1650,
        "epochs": 150,
        "description": "Large-scale: 11 vars, 5 clients, 1650 samples (Sachs-like: 330/client)",
    },
    "sachs": {
        "d": 11,
        "K": 3,
        "n": 853,
        "epochs": 150,
        "description": "Real Sachs protein signaling dataset: 11 vars, 3 clients, 853 samples (interventional heterogeneity)",
    },
}

# Seeds for statistical robustness
SEEDS = [42, 123, 456, 789, 2024]  # 5 runs per configuration

# Device configuration (supports CUDA and MPS for Mac)
def get_device():
    """Detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


DEVICE = get_device()


# ============================================================
# GPU Utilities
# ============================================================


def warmup_gpu(device="cuda"):
    """Warmup GPU with a small tensor operation."""
    if device in ["cuda", "mps"]:
        try:
            x = torch.randn(1000, 1000, device=device)
            y = torch.matmul(x, x)
            del x, y
            if device == "cuda":
                torch.cuda.synchronize()
            logging.info(f"GPU warmup completed on {device}")
        except Exception as e:
            logging.warning(f"GPU warmup failed: {e}")


def get_gpu_memory_info(device="cuda"):
    """Get GPU memory usage info."""
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - reserved,
        }
    return None


def log_gpu_memory(prefix="", device="cuda"):
    """Log GPU memory usage."""
    if device == "cuda" and torch.cuda.is_available():
        mem_info = get_gpu_memory_info(device)
        if mem_info:
            logging.info(
                f"{prefix}GPU Memory: "
                f"Allocated={mem_info['allocated_gb']:.2f}GB, "
                f"Reserved={mem_info['reserved_gb']:.2f}GB, "
                f"Free={mem_info['free_gb']:.2f}GB"
            )


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

    # Log GPU memory before experiment
    log_gpu_memory(prefix="[Pre-experiment] ", device=device)

    start_time = time.time()

    # Load data: Use real Sachs dataset if config is "sachs", otherwise generate synthetic
    if config_name == "sachs":
        logging.info("Loading real Sachs dataset...")
        from tests.utils.sachs_loader import load_sachs_federated

        # Load Sachs data partitioned by interventional conditions (horizontal-like)
        X_splits_raw, B, c_indx_raw = load_sachs_federated(
            n_clients=K, n_samples_limit=n
        )

        # Reconstruct global data
        X = np.vstack(X_splits_raw)

        # Re-partition based on scenario
        # Create c_indx matching the global data shape
        c_indx = np.repeat(np.arange(K), n // K).reshape(-1, 1)
        X_splits = partition_data(X, c_indx, K, scenario)

        W = B  # Use ground truth DAG as W (no need for separate weights)
        choice = None  # No heterogeneity choice for real data

        logging.info(
            f"  Sachs data: {X.shape[0]} samples, {X.shape[1]} features, "
            f"partitioned for {scenario} scenario"
        )
    else:
        # Generate synthetic data
        W, B, X, c_indx, choice = create_benchmark_data(
            d=d, K=K, n=n, seed=seed, data_type=data_type, sem_type="gauss"
        )

        # Partition data
        X_splits = partition_data(X, c_indx, K, scenario)

    # Setup FedCDH with production parameters
    model_type = "real" if config_name == "sachs" else "synthetic"
    args = Namespace(
        K=K,
        d=d,
        n=n // K,
        scenario=scenario,
        model_type=model_type,
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

    # Log GPU memory after experiment
    log_gpu_memory(prefix="[Post-experiment] ", device=device)

    # Clear GPU cache if using CUDA
    if device == "cuda":
        torch.cuda.empty_cache()

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


def run_scenario_comparison(
    config_name="medium", data_type="linear", seeds=None, device=None
):
    """
    Benchmark: Compare 3 SPN scenarios (H/V/Hy) on specified data type.

    Tests which aggregation strategy performs best.

    Args:
        config_name: Configuration to use (quick/small/medium/large/sachs)
        data_type: Type of data generation (linear/nonlinear)
        seeds: Random seeds for multiple runs
        device: Device to use (cuda/mps/cpu) or None for global DEVICE
    """
    if seeds is None:
        seeds = SEEDS

    active_device = device if device is not None else DEVICE

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
                device=active_device,
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


def main(config_name, data_type="linear", device=None, seeds=None):
    """
    Run scenario comparison benchmark.

    Args:
        config_name: Configuration to use (quick/small/medium/large/sachs)
        data_type: Type of data generation (linear/nonlinear)
        device: Device to use (cuda/mps/cpu) or None for auto-detect
        seeds: List of random seeds or None for default
    """
    # Use provided device or global DEVICE
    active_device = device if device is not None else DEVICE
    active_seeds = seeds if seeds is not None else SEEDS
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
    logging.info(f"Device: {active_device}")

    # Device information
    if active_device == "cuda":
        logging.info(f"CUDA Available: Yes")
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logging.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        logging.info(f"CUDA Version: {torch.version.cuda}")
    elif active_device == "mps":
        logging.info(f"MPS (Apple Silicon GPU) Available: Yes")
        logging.info(f"Running on Apple Silicon GPU")
    else:
        logging.info(f"Running on CPU (GPU not available)")

    logging.info(f"PyTorch Version: {torch.__version__}")
    logging.info(f"Seeds: {active_seeds}")
    logging.info("=" * 80)

    # GPU warmup
    if active_device in ["cuda", "mps"]:
        logging.info("\nWarming up GPU...")
        warmup_gpu(device=active_device)

    overall_start = time.time()

    # Run scenario comparison benchmark
    all_results = run_scenario_comparison(
        config_name=config_name,
        data_type=data_type,
        seeds=active_seeds,
        device=active_device,
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
    import argparse

    parser = argparse.ArgumentParser(description="FedCDH Benchmark Suite")
    parser.add_argument(
        "--config",
        type=str,
        default="quick",
        choices=["quick", "small", "medium", "large", "sachs"],
        help="Benchmark configuration (default: quick)",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="linear",
        choices=["linear", "nonlinear"],
        help="Data generation type (default: linear)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Override device selection (default: auto-detect)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Custom seed list (default: [42, 123, 456, 789, 2024])",
    )

    args = parser.parse_args()

    # Pass device and seeds to main()
    df, summary = main(
        config_name=args.config,
        data_type=args.data_type,
        device=args.device,
        seeds=args.seeds,
    )
