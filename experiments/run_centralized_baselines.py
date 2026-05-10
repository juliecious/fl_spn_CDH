#!/usr/bin/env python
"""
Centralized Baseline Evaluation for Causal Discovery.

Runs standard causal discovery algorithms (PC, GES, FCI) on pooled data
to establish upper-bound performance (no privacy constraints).

Datasets:
- Sachs: Protein signaling network (d=11, n=7466, 17 edges)
- Law School: Admissions fairness (d=5, n=21000, 7 edges)
- Synthetic: Controllable ground truth datasets

Algorithms:
- PC (Peter-Clark): Constraint-based
- GES (Greedy Equivalence Search): Score-based
- FCI (Fast Causal Inference): Handles latent confounders

Usage:
    python experiments/run_centralized_baselines.py --datasets sachs,law_school --methods pc,ges,fci --seeds 42,43,44
"""

import argparse
import logging
import time
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import causal discovery algorithms
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci

# Import dataset loaders
from tests.utils.law_school_loader import load_law_school_federated
from tests.utils.benchmark_loaders import (
    simulate_heterogeneous_data,
    load_standard_graph,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_sachs_data():
    """Load Sachs protein network dataset."""
    import gzip

    logging.info("Loading Sachs protein network dataset...")

    data_path = "tests/data/sachs.interventional.txt.gz"
    with gzip.open(data_path, "rt") as f:
        df = pd.read_csv(f, sep=" ")

    # Remove INT column if present
    if "INT" in df.columns:
        df = df.drop("INT", axis=1)

    X = df.values

    # Ground truth adjacency matrix (17 edges from Sachs et al. 2005)
    B = np.zeros((11, 11))
    edges = [
        (7, 0),
        (7, 1),
        (7, 5),
        (7, 6),
        (7, 9),
        (7, 10),  # PKA
        (8, 0),
        (8, 1),
        (8, 8),
        (8, 9),
        (8, 10),  # PKC
        (2, 3),
        (3, 4),
        (4, 2),  # Plcg-PIP2-PIP3 cycle
        (0, 1),
        (1, 5),
        (5, 6),  # Raf-Mek-Erk-Akt pathway
    ]
    for i, j in edges:
        B[i, j] = 1

    feature_names = list(df.columns)

    logging.info(
        f"  Sachs: {X.shape[0]} samples, {X.shape[1]} features, {int(np.sum(B))} edges"
    )

    return X, B, feature_names


def load_law_school_data():
    """Load Law School admissions dataset."""
    logging.info("Loading Law School admissions dataset...")

    X, B, feature_names = load_law_school_federated(n_clients=1, n_samples_limit=None)

    logging.info(
        f"  Law School: {X.shape[0]} samples, {X.shape[1]} features, {int(np.sum(B))} edges"
    )

    return X, B, feature_names


def load_synthetic_data(d=10, n=1000, model_type="linear", seed=42):
    """Generate synthetic data with known ground truth."""
    logging.info(f"Generating synthetic data (d={d}, n={n}, {model_type})...")

    # Generate synthetic DAG
    np.random.seed(seed)

    # Create sparse random DAG
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < 0.3:  # 30% sparsity
                B[i, j] = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])

    # Generate data using simulate_heterogeneous_data
    X, c_indx = simulate_heterogeneous_data(
        true_dag=B,
        n_clients=1,
        n_samples=n,
        data_type=model_type,
    )

    feature_names = [f"X{i}" for i in range(d)]

    # Binarize B for evaluation (0/1 edges)
    B_binary = (np.abs(B) > 0).astype(int)

    logging.info(
        f"  Synthetic ({model_type}): {X.shape[0]} samples, {X.shape[1]} features, {int(np.sum(B_binary))} edges"
    )

    return X, B_binary, feature_names


def compute_skeleton_metrics(pred_graph: np.ndarray, true_graph: np.ndarray) -> Dict:
    """
    Compute skeleton-based evaluation metrics.

    Args:
        pred_graph: Predicted adjacency matrix (may be CPDAG with -1/0/1 encoding)
        true_graph: Ground truth adjacency matrix (DAG with 0/1 encoding)

    Returns:
        Dictionary with F1, Precision, Recall, SHD
    """
    # Convert to skeleton (undirected)
    # Use abs() to handle CPDAG encoding (-1=tail, 1=head, 0=no edge)
    pred_skeleton = (np.abs(pred_graph) + np.abs(pred_graph.T)) > 0
    true_skeleton = (np.abs(true_graph) + np.abs(true_graph.T)) > 0

    # Remove self-loops
    np.fill_diagonal(pred_skeleton, 0)
    np.fill_diagonal(true_skeleton, 0)

    # Since skeleton is symmetric, only count upper triangle
    pred_edges = pred_skeleton[np.triu_indices_from(pred_skeleton, k=1)]
    true_edges = true_skeleton[np.triu_indices_from(true_skeleton, k=1)]

    # Confusion matrix
    tp = np.sum((pred_edges == 1) & (true_edges == 1))
    fp = np.sum((pred_edges == 1) & (true_edges == 0))
    fn = np.sum((pred_edges == 0) & (true_edges == 1))
    tn = np.sum((pred_edges == 0) & (true_edges == 0))

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Structural Hamming Distance (skeleton)
    shd = fp + fn

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "shd": shd,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_pc_algorithm(
    X: np.ndarray, alpha: float = 0.05, ci_test: str = "fisherz"
) -> Tuple[np.ndarray, float]:
    """Run PC algorithm."""
    logging.info(f"  Running PC (alpha={alpha}, ci_test={ci_test})...")

    start_time = time.time()

    try:
        # PC expects string name of test, not function
        cg = pc(X, alpha=alpha, indep_test=ci_test, verbose=False, show_progress=False)
        runtime = time.time() - start_time

        # Extract graph (may be CPDAG)
        G = cg.G.graph

        return G, runtime

    except Exception as e:
        logging.error(f"  PC failed: {e}")
        import traceback

        traceback.print_exc()
        runtime = time.time() - start_time
        return np.zeros((X.shape[1], X.shape[1])), runtime


def run_ges_algorithm(
    X: np.ndarray, score_func: str = "local_score_BIC"
) -> Tuple[np.ndarray, float]:
    """Run GES algorithm."""
    logging.info(f"  Running GES (score={score_func})...")

    start_time = time.time()

    try:
        # GES returns a Record object
        record = ges(X, score_func=score_func, maxP=None, parameters=None)
        runtime = time.time() - start_time

        # Extract graph (CPDAG)
        G = record["G"].graph

        return G, runtime

    except Exception as e:
        logging.error(f"  GES failed: {e}")
        runtime = time.time() - start_time
        return np.zeros((X.shape[1], X.shape[1])), runtime


def run_fci_algorithm(
    X: np.ndarray, alpha: float = 0.05, ci_test: str = "fisherz"
) -> Tuple[np.ndarray, float]:
    """Run FCI algorithm (handles latent confounders)."""
    logging.info(f"  Running FCI (alpha={alpha}, ci_test={ci_test})...")

    start_time = time.time()

    try:
        # FCI expects string name of test
        G, edges = fci(X, ci_test, alpha=alpha, verbose=False, show_progress=False)
        runtime = time.time() - start_time

        # Extract adjacency matrix from graph object
        graph_matrix = G.graph

        return graph_matrix, runtime

    except Exception as e:
        logging.error(f"  FCI failed: {e}")
        import traceback

        traceback.print_exc()
        runtime = time.time() - start_time
        return np.zeros((X.shape[1], X.shape[1])), runtime


def run_single_experiment(
    dataset_name: str,
    X: np.ndarray,
    B: np.ndarray,
    method: str,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Run a single centralized baseline experiment."""

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Run algorithm
    if method == "pc":
        G, runtime = run_pc_algorithm(X, alpha=alpha, ci_test="fisherz")
    elif method == "ges":
        G, runtime = run_ges_algorithm(X, score_func="local_score_BIC")
    elif method == "fci":
        G, runtime = run_fci_algorithm(X, alpha=alpha, ci_test="fisherz")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute metrics
    metrics = compute_skeleton_metrics(G, B)

    # Compile results
    result = {
        "dataset": dataset_name,
        "method": method,
        "alpha": alpha,
        "seed": seed,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_edges_true": int(np.sum(B)),
        "n_edges_pred": int(np.sum((G + G.T) > 0) / 2),  # Skeleton edges
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "shd": metrics["shd"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "runtime_sec": runtime,
    }

    logging.info(
        f"  {method.upper()}: F1={result['f1']:.3f}, SHD={result['shd']}, "
        f"Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, "
        f"Time={result['runtime_sec']:.2f}s"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run centralized baseline experiments")
    parser.add_argument(
        "--datasets",
        type=str,
        default="sachs,law_school",
        help="Comma-separated list of datasets (sachs,law_school,synthetic_linear,synthetic_nonlinear)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="ges",
        help="Comma-separated list of methods (pc,ges,fci). Note: PC has API issues in current causal-learn version.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.05],
        help="Alpha values for CI tests (space-separated)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated list of random seeds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/centralized_baselines.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--synthetic-d",
        type=int,
        default=10,
        help="Number of features for synthetic data",
    )
    parser.add_argument(
        "--synthetic-n",
        type=int,
        default=1000,
        help="Number of samples for synthetic data",
    )

    args = parser.parse_args()

    # Parse arguments
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = args.alpha

    logging.info("=" * 80)
    logging.info("CENTRALIZED BASELINE EVALUATION")
    logging.info("=" * 80)
    logging.info(f"Datasets: {datasets}")
    logging.info(f"Methods: {methods}")
    logging.info(f"Alpha values: {alphas}")
    logging.info(f"Seeds: {seeds}")
    logging.info("=" * 80)

    # Load datasets
    dataset_dict = {}

    for dataset_name in datasets:
        if dataset_name == "sachs":
            X, B, feature_names = load_sachs_data()
            dataset_dict["sachs"] = (X, B, feature_names)

        elif dataset_name == "law_school":
            X, B, feature_names = load_law_school_data()
            dataset_dict["law_school"] = (X, B, feature_names)

        elif dataset_name == "synthetic_linear":
            X, B, feature_names = load_synthetic_data(
                d=args.synthetic_d,
                n=args.synthetic_n,
                model_type="linear",
                seed=seeds[0],
            )
            dataset_dict["synthetic_linear"] = (X, B, feature_names)

        elif dataset_name == "synthetic_nonlinear":
            X, B, feature_names = load_synthetic_data(
                d=args.synthetic_d,
                n=args.synthetic_n,
                model_type="nonlinear",
                seed=seeds[0],
            )
            dataset_dict["synthetic_nonlinear"] = (X, B, feature_names)

        else:
            logging.warning(f"Unknown dataset: {dataset_name}, skipping")

    # Run experiments
    all_results = []

    for dataset_name, (X, B, feature_names) in dataset_dict.items():
        logging.info(f"\n{'='*80}")
        logging.info(f"Dataset: {dataset_name.upper()}")
        logging.info(f"{'='*80}")

        for method in methods:
            logging.info(f"\nMethod: {method.upper()}")

            # Determine alpha values to use
            if method in ["pc", "fci"]:
                alpha_values = alphas
            else:
                alpha_values = [None]  # GES doesn't use alpha

            for alpha in alpha_values:
                for seed in seeds:
                    logging.info(f"\n  Seed: {seed}, Alpha: {alpha}")

                    result = run_single_experiment(
                        dataset_name=dataset_name,
                        X=X,
                        B=B,
                        method=method,
                        alpha=alpha if alpha is not None else 0.05,
                        seed=seed,
                    )

                    all_results.append(result)

    # Save results
    df_results = pd.DataFrame(all_results)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(output_path, index=False)
    logging.info(f"\n{'='*80}")
    logging.info(f"Results saved to: {output_path}")
    logging.info(f"{'='*80}")

    # Print summary statistics
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY STATISTICS")
    logging.info("=" * 80)

    summary = (
        df_results.groupby(["dataset", "method"])
        .agg(
            {
                "f1": ["mean", "std"],
                "shd": ["mean", "std"],
                "precision": ["mean", "std"],
                "recall": ["mean", "std"],
                "runtime_sec": ["mean", "std"],
            }
        )
        .round(3)
    )

    print(summary)

    # Find best methods per dataset
    logging.info("\n" + "=" * 80)
    logging.info("BEST METHODS PER DATASET")
    logging.info("=" * 80)

    for dataset_name in df_results["dataset"].unique():
        dataset_results = df_results[df_results["dataset"] == dataset_name]
        best_f1 = dataset_results.groupby("method")["f1"].mean().idxmax()
        best_f1_score = dataset_results.groupby("method")["f1"].mean().max()

        logging.info(f"\n{dataset_name.upper()}:")
        logging.info(f"  Best method: {best_f1.upper()} (F1={best_f1_score:.3f})")

        # Show all methods
        for method in dataset_results["method"].unique():
            method_results = dataset_results[dataset_results["method"] == method]
            mean_f1 = method_results["f1"].mean()
            std_f1 = method_results["f1"].std()
            mean_shd = method_results["shd"].mean()
            logging.info(
                f"    {method.upper()}: F1={mean_f1:.3f}±{std_f1:.3f}, SHD={mean_shd:.1f}"
            )

    logging.info("\n" + "=" * 80)
    logging.info("CENTRALIZED BASELINE EVALUATION COMPLETE")
    logging.info("=" * 80)

    return df_results


if __name__ == "__main__":
    main()
