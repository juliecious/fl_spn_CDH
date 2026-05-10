#!/usr/bin/env python
"""
Original FedCDH (CD-NOD) Baseline Evaluation.

Runs the original FedCDH algorithm (Li et al. 2024) using CD-NOD
(Causal Discovery with Nodewise distributional Heterogeneity) on
horizontal federated scenarios.

Reference: TestFedCDH.py from original implementation

Usage:
    python experiments/run_fedcdh_baseline.py --datasets sachs,law_school --seeds 42,43,44
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

# Import FedCDH algorithm (CD-NOD)
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci
from causallearn.utils.data_utils import (
    get_cpdag_from_cdnod,
    get_dag_from_pdag,
    count_skeleton_accuracy,
    count_dag_accuracy,
)

# Import dataset loaders
from tests.utils.law_school_loader import load_law_school_federated

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

    # Ground truth adjacency matrix (17 edges)
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


def partition_horizontal(X: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partition data horizontally for federated learning.

    Args:
        X: Data matrix (n, d)
        K: Number of clients

    Returns:
        X_partitioned: Same data with samples split
        c_indx: Client index array (n, 1)
    """
    n = X.shape[0]
    samples_per_client = n // K

    # Create client index array
    c_indx = np.repeat(np.arange(K), samples_per_client)

    # Handle remainder
    remainder = n % K
    if remainder > 0:
        c_indx = np.concatenate([c_indx, np.arange(remainder)])

    c_indx = c_indx[:n].reshape(-1, 1)

    return X, c_indx


def run_fedcdh_experiment(
    dataset_name: str,
    X: np.ndarray,
    B: np.ndarray,
    K: int,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """
    Run original FedCDH (CD-NOD) experiment.

    Args:
        dataset_name: Name of dataset
        X: Data matrix (n, d)
        B: Ground truth adjacency matrix
        K: Number of clients
        alpha: Significance level for CI tests
        seed: Random seed

    Returns:
        Dictionary with results
    """
    np.random.seed(seed)

    d = X.shape[1]
    n_total = X.shape[0]

    logging.info(f"  Running FedCDH (CD-NOD) with K={K} clients, alpha={alpha}")

    # Partition data horizontally
    X_partitioned, c_indx = partition_horizontal(X, K)

    # Run CD-NOD (original FedCDH)
    start_time = time.time()

    try:
        cg = cdnod(
            X_partitioned,
            c_indx,
            K,
            alpha,
            kci,  # KCI test (as in original paper)
            True,  # background_knowledge
            0,  # uc_rule
            -1,  # uc_priority
        )
        runtime = time.time() - start_time

        # Extract graph
        est_graph = cg.G.graph[0:d, 0:d]

        # Convert to CPDAG
        est_cpdag = get_cpdag_from_cdnod(est_graph)

        # Convert to DAG for orientation metrics
        est_dag = get_dag_from_pdag(est_cpdag)

        # Compute skeleton metrics (undirected)
        skeleton_metrics = count_skeleton_accuracy(B, est_cpdag)

        # Compute DAG metrics (directed)
        dag_metrics = count_dag_accuracy(B, est_dag)

        # Combine metrics
        result = {
            "dataset": dataset_name,
            "method": "fedcdh",
            "K": K,
            "alpha": alpha,
            "seed": seed,
            "n_samples": n_total,
            "n_features": d,
            "n_edges_true": int(np.sum(B)),
            # Skeleton metrics (undirected)
            "skeleton_f1": skeleton_metrics.get("f1", 0.0),
            "skeleton_precision": skeleton_metrics.get("precision", 0.0),
            "skeleton_recall": skeleton_metrics.get("recall", 0.0),
            "skeleton_shd": skeleton_metrics.get("shd", 0),
            # DAG metrics (directed, includes orientation)
            "dag_f1": dag_metrics.get("f1", 0.0),
            "dag_precision": dag_metrics.get("precision", 0.0),
            "dag_recall": dag_metrics.get("recall", 0.0),
            "dag_shd": dag_metrics.get("shd", 0),
            "runtime_sec": runtime,
        }

        logging.info(
            f"  FedCDH: Skeleton F1={result['skeleton_f1']:.3f}, "
            f"DAG F1={result['dag_f1']:.3f}, "
            f"SHD={result['skeleton_shd']}, Time={result['runtime_sec']:.2f}s"
        )

        return result

    except Exception as e:
        logging.error(f"  FedCDH failed: {e}")
        import traceback

        traceback.print_exc()

        # Return empty result
        return {
            "dataset": dataset_name,
            "method": "fedcdh",
            "K": K,
            "alpha": alpha,
            "seed": seed,
            "n_samples": n_total,
            "n_features": d,
            "n_edges_true": int(np.sum(B)),
            "skeleton_f1": 0.0,
            "skeleton_precision": 0.0,
            "skeleton_recall": 0.0,
            "skeleton_shd": 0,
            "dag_f1": 0.0,
            "dag_precision": 0.0,
            "dag_recall": 0.0,
            "dag_shd": 0,
            "runtime_sec": time.time() - start_time,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run FedCDH (CD-NOD) baseline experiments"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="sachs,law_school",
        help="Comma-separated list of datasets",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="Number of federated clients",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CI tests",
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
        default="benchmark_results/fedcdh_baselines.csv",
        help="Output CSV file for results",
    )

    args = parser.parse_args()

    # Parse arguments
    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    logging.info("=" * 80)
    logging.info("FEDCDH (CD-NOD) BASELINE EVALUATION")
    logging.info("=" * 80)
    logging.info(f"Datasets: {datasets}")
    logging.info(f"Number of clients: {args.K}")
    logging.info(f"Alpha: {args.alpha}")
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

        else:
            logging.warning(f"Unknown dataset: {dataset_name}, skipping")

    # Run experiments
    all_results = []

    for dataset_name, (X, B, feature_names) in dataset_dict.items():
        logging.info(f"\n{'='*80}")
        logging.info(f"Dataset: {dataset_name.upper()}")
        logging.info(f"{'='*80}")

        for seed in seeds:
            logging.info(f"\nSeed: {seed}")

            result = run_fedcdh_experiment(
                dataset_name=dataset_name,
                X=X,
                B=B,
                K=args.K,
                alpha=args.alpha,
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
        df_results.groupby("dataset")
        .agg(
            {
                "skeleton_f1": ["mean", "std"],
                "skeleton_shd": ["mean", "std"],
                "dag_f1": ["mean", "std"],
                "dag_shd": ["mean", "std"],
                "runtime_sec": ["mean", "std"],
            }
        )
        .round(3)
    )

    print(summary)

    # Print per-dataset summary
    logging.info("\n" + "=" * 80)
    logging.info("PER-DATASET RESULTS")
    logging.info("=" * 80)

    for dataset_name in df_results["dataset"].unique():
        dataset_results = df_results[df_results["dataset"] == dataset_name]

        mean_skel_f1 = dataset_results["skeleton_f1"].mean()
        std_skel_f1 = dataset_results["skeleton_f1"].std()
        mean_dag_f1 = dataset_results["dag_f1"].mean()
        std_dag_f1 = dataset_results["dag_f1"].std()
        mean_shd = dataset_results["skeleton_shd"].mean()
        mean_time = dataset_results["runtime_sec"].mean()

        logging.info(f"\n{dataset_name.upper()}:")
        logging.info(f"  Skeleton F1: {mean_skel_f1:.3f}±{std_skel_f1:.3f}")
        logging.info(f"  DAG F1: {mean_dag_f1:.3f}±{std_dag_f1:.3f}")
        logging.info(f"  SHD: {mean_shd:.1f}")
        logging.info(f"  Runtime: {mean_time:.2f}s")

    logging.info("\n" + "=" * 80)
    logging.info("FEDCDH BASELINE EVALUATION COMPLETE")
    logging.info("=" * 80)

    return df_results


if __name__ == "__main__":
    main()
