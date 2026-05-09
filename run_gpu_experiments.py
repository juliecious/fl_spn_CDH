"""
GPU Experiment Runner for V3 Aggregation Strategies.

Runs full-scale experiments on GPU to validate V3 fixes:
1. Horizontal aggregation comparison (structure_voting, ll_weighted, mixture)
2. Hybrid mode with GlobalSumOfProducts
3. Vertical mode baseline

Usage:
    # Run all experiments
    python run_gpu_experiments.py --device cuda

    # Run specific scenario
    python run_gpu_experiments.py --scenario horizontal --device cuda

    # Run with specific config
    python run_gpu_experiments.py --config MEDIUM --device cuda
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from argparse import Namespace

from causallearn.search.FCMBased.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    simulate_dag,
    simulate_parameter,
    my_simulate_linear_gaussian,
    set_random_seed,
)
from causallearn.utils.metrics import compute_structural_metrics


def get_config(config_name: str) -> dict:
    """Get experiment configuration."""
    configs = {
        "SMALL": {"d": 8, "K": 3, "n_total": 900, "epochs": 50},
        "MEDIUM": {"d": 12, "K": 3, "n_total": 1800, "epochs": 100},
        "LARGE": {"d": 20, "K": 5, "n_total": 3600, "epochs": 150},
    }
    return configs[config_name]


def run_horizontal_experiment(config: dict, strategy: str, device: str, seed: int):
    """Run horizontal mode experiment with specified aggregation strategy."""
    print("\n" + "=" * 80)
    print(f"HORIZONTAL MODE: {strategy.upper()}")
    print(f"Config: d={config['d']}, K={config['K']}, n={config['n_total']}")
    print(f"Device: {device}, Seed: {seed}")
    print("=" * 80)

    set_random_seed(seed)

    # Generate data
    d, K, n_total = config["d"], config["K"], config["n_total"]
    G_bin = simulate_dag(d, d, "ER")
    B = simulate_parameter(G_bin)
    X_samples, _ = my_simulate_linear_gaussian(B, K, n_total, "gauss")

    # Split horizontally
    c_indx = np.zeros((n_total, 1))
    n_per = n_total // K
    X_splits = [
        X_samples[k * n_per : (k + 1) * n_per if k < K - 1 else n_total, :]
        for k in range(K)
    ]

    # Configure
    args = Namespace(
        K=K,
        d=d,
        scenario="horizontal",
        model_type="SPN",
        ci_method="spn",
        n=n_per,
        epochs=config["epochs"],
        alpha=0.05,
        data_type="linear",
        device=device,
        horizontal_aggregation=strategy,
        structure_vote_threshold=0.5,
        skip_spn_eval=False,  # Full evaluation
    )

    # Run experiment
    start_time = time.time()
    fedcdh = FedCDH(args)
    fedcdh.fit(X_splits, c_indx, G_bin)
    runtime = time.time() - start_time

    # Compute metrics
    if hasattr(fedcdh, "discovered_graph") and fedcdh.discovered_graph is not None:
        metrics = compute_structural_metrics(G_bin, fedcdh.discovered_graph)
    else:
        print("Warning: No discovered_graph found")
        metrics = {}

    # Add runtime
    metrics["runtime_seconds"] = runtime
    metrics["strategy"] = strategy
    metrics["true_edges"] = int(np.sum(G_bin))

    # Add strategy-specific info
    if strategy == "structure_voting" and hasattr(fedcdh, "edge_confidence"):
        metrics["avg_edge_confidence"] = float(
            np.mean(list(fedcdh.edge_confidence.values()))
            if fedcdh.edge_confidence
            else 0.0
        )
        metrics["num_consensus_edges"] = len(fedcdh.edge_confidence)

    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def run_hybrid_experiment(config: dict, device: str, seed: int):
    """Run hybrid mode experiment with GlobalSumOfProducts."""
    print("\n" + "=" * 80)
    print("HYBRID MODE: GlobalSumOfProducts")
    print(f"Config: d={config['d']}, K={config['K']}, n={config['n_total']}")
    print(f"Device: {device}, Seed: {seed}")
    print("=" * 80)

    set_random_seed(seed)

    # Generate data
    d, K, n_total = config["d"], config["K"], config["n_total"]
    G_bin = simulate_dag(d, d, "ER")
    B = simulate_parameter(G_bin)
    X_samples, _ = my_simulate_linear_gaussian(B, K, n_total, "gauss")
    c_indx = np.zeros((n_total, 1))

    # Configure
    args = Namespace(
        K=K,
        d=d,
        scenario="hybrid",
        model_type="SPN",
        ci_method="spn",
        n=n_total // K,
        epochs=config["epochs"],
        alpha=0.05,
        data_type="linear",
        device=device,
        num_cluster_samples=20,  # Full clustering
        skip_spn_eval=False,
    )

    # Run experiment
    start_time = time.time()
    fedcdh = FedCDH(args)
    fedcdh.fit(X_samples, c_indx, G_bin)
    runtime = time.time() - start_time

    # Compute metrics
    if hasattr(fedcdh, "discovered_graph") and fedcdh.discovered_graph is not None:
        metrics = compute_structural_metrics(G_bin, fedcdh.discovered_graph)
    else:
        print("Warning: No discovered_graph found")
        metrics = {}

    metrics["runtime_seconds"] = runtime
    metrics["true_edges"] = int(np.sum(G_bin))

    # Check if using GlobalSumOfProducts
    from causallearn.utils.FedPC import GlobalSumOfProducts

    if hasattr(fedcdh, "fed_spn_model") and fedcdh.fed_spn_model is not None:
        inner_model = (
            fedcdh.fed_spn_model.spn
            if hasattr(fedcdh.fed_spn_model, "spn")
            else fedcdh.fed_spn_model
        )
        metrics["using_global_sum_of_products"] = isinstance(
            inner_model, GlobalSumOfProducts
        )
        if isinstance(inner_model, GlobalSumOfProducts):
            metrics["num_cluster_combinations"] = len(inner_model.products)

    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def run_vertical_experiment(config: dict, device: str, seed: int):
    """Run vertical mode experiment."""
    print("\n" + "=" * 80)
    print("VERTICAL MODE: ProductOverGroups")
    print(f"Config: d={config['d']}, K={config['K']}, n={config['n_total']}")
    print(f"Device: {device}, Seed: {seed}")
    print("=" * 80)

    set_random_seed(seed)

    # Generate data
    d, K, n_total = config["d"], config["K"], config["n_total"]
    G_bin = simulate_dag(d, d, "ER")
    B = simulate_parameter(G_bin)
    X_samples, _ = my_simulate_linear_gaussian(B, K, n_total, "gauss")
    c_indx = np.zeros((n_total, 1))

    # Configure
    args = Namespace(
        K=K,
        d=d,
        scenario="vertical",
        model_type="SPN",
        ci_method="spn",
        n=n_total,
        epochs=config["epochs"],
        alpha=0.05,
        data_type="linear",
        device=device,
        skip_spn_eval=False,
    )

    # Run experiment
    start_time = time.time()
    fedcdh = FedCDH(args)
    fedcdh.fit(X_samples, c_indx, G_bin)
    runtime = time.time() - start_time

    # Compute metrics
    if hasattr(fedcdh, "discovered_graph") and fedcdh.discovered_graph is not None:
        metrics = compute_structural_metrics(G_bin, fedcdh.discovered_graph)
    else:
        print("Warning: No discovered_graph found")
        metrics = {}

    metrics["runtime_seconds"] = runtime
    metrics["true_edges"] = int(np.sum(G_bin))

    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run GPU experiments for V3")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "horizontal", "hybrid", "vertical"],
        help="Which scenario to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="MEDIUM",
        choices=["SMALL", "MEDIUM", "LARGE"],
        help="Experiment configuration",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5, help="Number of random seeds"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gpu_experiment_results",
        help="Output directory",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get config
    config = get_config(args.config)

    # Run experiments
    all_results = []

    for seed in range(args.num_seeds):
        print(f"\n\n{'='*80}")
        print(f"SEED {seed + 1}/{args.num_seeds}")
        print(f"{'='*80}")

        if args.scenario in ["all", "horizontal"]:
            # Test all 3 horizontal strategies
            for strategy in ["structure_voting", "ll_weighted", "mixture"]:
                try:
                    result = run_horizontal_experiment(
                        config, strategy, args.device, seed
                    )
                    result["scenario"] = "horizontal"
                    result["config"] = args.config
                    result["seed"] = seed
                    result["timestamp"] = datetime.now().isoformat()
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in horizontal {strategy}: {e}")
                    import traceback

                    traceback.print_exc()

        if args.scenario in ["all", "hybrid"]:
            try:
                result = run_hybrid_experiment(config, args.device, seed)
                result["scenario"] = "hybrid"
                result["config"] = args.config
                result["seed"] = seed
                result["timestamp"] = datetime.now().isoformat()
                all_results.append(result)
            except Exception as e:
                print(f"Error in hybrid: {e}")
                import traceback

                traceback.print_exc()

        if args.scenario in ["all", "vertical"]:
            try:
                result = run_vertical_experiment(config, args.device, seed)
                result["scenario"] = "vertical"
                result["config"] = args.config
                result["seed"] = seed
                result["timestamp"] = datetime.now().isoformat()
                all_results.append(result)
            except Exception as e:
                print(f"Error in vertical: {e}")
                import traceback

                traceback.print_exc()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{args.scenario}_{args.config}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Total experiments: {len(all_results)}")

    # Print summary statistics
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        # Group by scenario/strategy
        from collections import defaultdict

        grouped = defaultdict(list)
        for r in all_results:
            key = f"{r['scenario']}"
            if r["scenario"] == "horizontal":
                key += f"_{r.get('strategy', 'unknown')}"
            grouped[key].append(r)

        for key, results in grouped.items():
            print(f"\n{key}:")
            if "f1_skeleton" in results[0]:
                f1_values = [r["f1_skeleton"] for r in results]
                print(
                    f"  F1 Skeleton: {np.mean(f1_values):.3f} ± {np.std(f1_values):.3f}"
                )
            if "shd" in results[0]:
                shd_values = [r["shd"] for r in results]
                print(f"  SHD: {np.mean(shd_values):.1f} ± {np.std(shd_values):.1f}")
            if "runtime_seconds" in results[0]:
                time_values = [r["runtime_seconds"] for r in results]
                print(
                    f"  Runtime: {np.mean(time_values):.1f}s ± {np.std(time_values):.1f}s"
                )


if __name__ == "__main__":
    main()
