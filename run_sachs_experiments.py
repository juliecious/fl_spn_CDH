"""
Sachs Protein Network Experiments for V3 Validation.

Real-world validation on the Sachs protein signaling dataset:
- 11 nodes (proteins)
- 7,466 samples (interventional data)
- 17 known edges (ground truth)

Tests all 3 scenarios (horizontal, vertical, hybrid) with V3 fixes.

Usage:
    # Run all scenarios
    python run_sachs_experiments.py --device cuda

    # Run specific scenario
    python run_sachs_experiments.py --scenario horizontal --device cuda

    # Run with specific aggregation strategy (horizontal only)
    python run_sachs_experiments.py --scenario horizontal --strategy structure_voting --device cuda
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from argparse import Namespace

from causallearn.search.FCMBased.FedCDH import FedCDH
from causallearn.utils.metrics import compute_structural_metrics


def load_sachs_data(data_path: str = "tests/data/sachs.interventional.txt.gz"):
    """Load Sachs protein network dataset."""
    print(f"Loading Sachs dataset from: {data_path}")

    # Decompress if needed
    if data_path.endswith(".gz"):
        with gzip.open(data_path, "rt") as f:
            df = pd.read_csv(f, sep="\t")
    else:
        df = pd.read_csv(data_path, sep="\t")

    print(f"Loaded: {df.shape[0]} samples, {df.shape[1]} variables")
    print(f"Variables: {list(df.columns)}")

    return df.values


def get_sachs_ground_truth():
    """
    Get Sachs ground truth network (17 edges).

    Based on Sachs et al. (2005) Science paper.
    Node order: Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk

    Returns adjacency matrix (11x11) with 1 indicating edge X→Y.
    """
    # Node names (for reference)
    nodes = [
        "Raf",
        "Mek",
        "Plcg",
        "PIP2",
        "PIP3",
        "Erk",
        "Akt",
        "PKA",
        "PKC",
        "P38",
        "Jnk",
    ]

    # Initialize empty adjacency matrix
    G = np.zeros((11, 11))

    # Ground truth edges (from Sachs et al. 2005)
    # Format: (from, to)
    edges = [
        (7, 0),  # PKA → Raf
        (7, 1),  # PKA → Mek
        (7, 5),  # PKA → Erk
        (7, 6),  # PKA → Akt
        (7, 9),  # PKA → P38
        (7, 10),  # PKA → Jnk
        (8, 0),  # PKC → Raf
        (8, 1),  # PKC → Mek
        (8, 8),  # PKC → PKC (self-loop, sometimes excluded)
        (8, 9),  # PKC → P38
        (8, 10),  # PKC → Jnk
        (2, 3),  # Plcg → PIP2
        (3, 4),  # PIP2 → PIP3
        (4, 2),  # PIP3 → Plcg
        (0, 1),  # Raf → Mek
        (1, 5),  # Mek → Erk
        (5, 6),  # Erk → Akt
    ]

    for i, j in edges:
        G[i, j] = 1

    print(f"Ground truth: {len(edges)} edges")
    return G


def run_sachs_horizontal(strategy: str, device: str, K: int = 3, epochs: int = 100):
    """Run Sachs experiment in horizontal mode."""
    print("\n" + "=" * 80)
    print(f"SACHS: HORIZONTAL MODE - {strategy.upper()}")
    print(f"K={K} clients, epochs={epochs}, device={device}")
    print("=" * 80)

    # Load data
    X = load_sachs_data()
    G_true = get_sachs_ground_truth()
    n, d = X.shape

    # Split horizontally (same features, different samples)
    n_per = n // K
    X_splits = [X[k * n_per : (k + 1) * n_per if k < K - 1 else n, :] for k in range(K)]

    print(f"\nData splits:")
    for k, X_k in enumerate(X_splits):
        print(f"  Client {k}: {X_k.shape[0]} samples × {X_k.shape[1]} features")

    c_indx = np.zeros((n, 1))

    # Configure
    args = Namespace(
        K=K,
        d=d,
        scenario="horizontal",
        model_type="SPN",
        ci_method="spn",
        n=n_per,
        epochs=epochs,
        alpha=0.05,
        data_type="nonlinear",  # Sachs is nonlinear
        device=device,
        horizontal_aggregation=strategy,
        structure_vote_threshold=0.5,
        skip_spn_eval=False,
    )

    # Run experiment
    import time

    start_time = time.time()
    fedcdh = FedCDH(args)
    fedcdh.fit(X_splits, c_indx, G_true)
    runtime = time.time() - start_time

    # Compute metrics
    if hasattr(fedcdh, "discovered_graph") and fedcdh.discovered_graph is not None:
        metrics = compute_structural_metrics(G_true, fedcdh.discovered_graph)
    else:
        print("Warning: No discovered_graph found")
        metrics = {}

    metrics["runtime_seconds"] = runtime
    metrics["strategy"] = strategy
    metrics["true_edges"] = 17
    metrics["num_samples"] = n
    metrics["num_features"] = d

    # Strategy-specific metrics
    if strategy == "structure_voting" and hasattr(fedcdh, "edge_confidence"):
        if fedcdh.edge_confidence:
            metrics["avg_edge_confidence"] = float(
                np.mean(list(fedcdh.edge_confidence.values()))
            )
            metrics["num_consensus_edges"] = len(fedcdh.edge_confidence)

    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return metrics


def run_sachs_hybrid(device: str, K: int = 3, epochs: int = 100):
    """Run Sachs experiment in hybrid mode."""
    print("\n" + "=" * 80)
    print("SACHS: HYBRID MODE - GlobalSumOfProducts")
    print(f"K={K} clients, epochs={epochs}, device={device}")
    print("=" * 80)

    # Load data
    X = load_sachs_data()
    G_true = get_sachs_ground_truth()
    n, d = X.shape

    c_indx = np.zeros((n, 1))

    # Configure
    args = Namespace(
        K=K,
        d=d,
        scenario="hybrid",
        model_type="SPN",
        ci_method="spn",
        n=n // K,
        epochs=epochs,
        alpha=0.05,
        data_type="nonlinear",
        device=device,
        num_cluster_samples=30,
        skip_spn_eval=False,
    )

    # Run experiment
    import time

    start_time = time.time()
    fedcdh = FedCDH(args)
    fedcdh.fit(X, c_indx, G_true)
    runtime = time.time() - start_time

    # Compute metrics
    if hasattr(fedcdh, "discovered_graph") and fedcdh.discovered_graph is not None:
        metrics = compute_structural_metrics(G_true, fedcdh.discovered_graph)
    else:
        print("Warning: No discovered_graph found")
        metrics = {}

    metrics["runtime_seconds"] = runtime
    metrics["true_edges"] = 17
    metrics["num_samples"] = n
    metrics["num_features"] = d

    # Check GlobalSumOfProducts usage
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
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return metrics


def run_sachs_vertical(device: str, K: int = 3, epochs: int = 100):
    """Run Sachs experiment in vertical mode."""
    print("\n" + "=" * 80)
    print("SACHS: VERTICAL MODE - ProductOverGroups")
    print(f"K={K} clients, epochs={epochs}, device={device}")
    print("=" * 80)

    # Load data
    X = load_sachs_data()
    G_true = get_sachs_ground_truth()
    n, d = X.shape

    c_indx = np.zeros((n, 1))

    # Configure
    args = Namespace(
        K=K,
        d=d,
        scenario="vertical",
        model_type="SPN",
        ci_method="spn",
        n=n,
        epochs=epochs,
        alpha=0.05,
        data_type="nonlinear",
        device=device,
        skip_spn_eval=False,
    )

    # Run experiment
    import time

    start_time = time.time()
    fedcdh = FedCDH(args)
    fedcdh.fit(X, c_indx, G_true)
    runtime = time.time() - start_time

    # Compute metrics
    if hasattr(fedcdh, "discovered_graph") and fedcdh.discovered_graph is not None:
        metrics = compute_structural_metrics(G_true, fedcdh.discovered_graph)
    else:
        print("Warning: No discovered_graph found")
        metrics = {}

    metrics["runtime_seconds"] = runtime
    metrics["true_edges"] = 17
    metrics["num_samples"] = n
    metrics["num_features"] = d

    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Sachs experiments for V3")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "horizontal", "hybrid", "vertical"],
        help="Which scenario to run",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="structure_voting",
        choices=["structure_voting", "ll_weighted", "mixture"],
        help="Horizontal aggregation strategy (horizontal only)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda, mps, cpu)"
    )
    parser.add_argument("--K", type=int, default=3, help="Number of clients")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument(
        "--num-runs", type=int, default=5, help="Number of independent runs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sachs_results",
        help="Output directory",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run experiments
    all_results = []

    for run_idx in range(args.num_runs):
        print(f"\n\n{'='*80}")
        print(f"RUN {run_idx + 1}/{args.num_runs}")
        print(f"{'='*80}")

        if args.scenario in ["all", "horizontal"]:
            # Test specified strategy (or all 3 if scenario=all)
            strategies = (
                ["structure_voting", "ll_weighted", "mixture"]
                if args.scenario == "all"
                else [args.strategy]
            )

            for strategy in strategies:
                try:
                    result = run_sachs_horizontal(
                        strategy, args.device, args.K, args.epochs
                    )
                    result["scenario"] = "horizontal"
                    result["run"] = run_idx
                    result["timestamp"] = datetime.now().isoformat()
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in horizontal {strategy}: {e}")
                    import traceback

                    traceback.print_exc()

        if args.scenario in ["all", "hybrid"]:
            try:
                result = run_sachs_hybrid(args.device, args.K, args.epochs)
                result["scenario"] = "hybrid"
                result["run"] = run_idx
                result["timestamp"] = datetime.now().isoformat()
                all_results.append(result)
            except Exception as e:
                print(f"Error in hybrid: {e}")
                import traceback

                traceback.print_exc()

        if args.scenario in ["all", "vertical"]:
            try:
                result = run_sachs_vertical(args.device, args.K, args.epochs)
                result["scenario"] = "vertical"
                result["run"] = run_idx
                result["timestamp"] = datetime.now().isoformat()
                all_results.append(result)
            except Exception as e:
                print(f"Error in vertical: {e}")
                import traceback

                traceback.print_exc()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sachs_results_{args.scenario}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("SACHS EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Total experiments: {len(all_results)}")

    # Print summary statistics
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

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
            if "precision" in results[0]:
                prec_values = [r["precision"] for r in results]
                print(
                    f"  Precision: {np.mean(prec_values):.3f} ± {np.std(prec_values):.3f}"
                )
            if "recall" in results[0]:
                rec_values = [r["recall"] for r in results]
                print(f"  Recall: {np.mean(rec_values):.3f} ± {np.std(rec_values):.3f}")
            if "runtime_seconds" in results[0]:
                time_values = [r["runtime_seconds"] for r in results]
                print(
                    f"  Runtime: {np.mean(time_values):.1f}s ± {np.std(time_values):.1f}s"
                )


if __name__ == "__main__":
    main()
