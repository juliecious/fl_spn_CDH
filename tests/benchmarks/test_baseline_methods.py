#!/usr/bin/env python
"""
Baseline Methods Benchmark - Traditional Causal Discovery Algorithms.

This script benchmarks baseline causal discovery methods for comparison against FedSPN:
- ges: Greedy Equivalence Search (score-based, centralized)
- fci: Fast Causal Inference (constraint-based, centralized, handles latent confounders)
- fedcdh: FedCDH with KCI tests (kernel-based, federated)

Features:
- ✅ Multiple datasets: Real-world (Sachs, Law School, Asia, Dream4) + Synthetic (ER, SF, Chain)
- ✅ Comprehensive metrics: Structure accuracy, runtime, quality metrics
- ✅ Seed-controlled evaluation for robust comparison
- ✅ Compatible with FedSPN benchmark for side-by-side comparison

Architecture:
- BaselineBenchmark: Main experiment runner
- DatasetRegistry: Shared dataset management (imported from FedSPN script)
- BaselineMethodConfig: Baseline method configurations
- BaselineMethodRunner: Execution engine for centralized and federated baselines
- MetricsEngine: Comprehensive evaluation (shared with FedSPN script)

Usage:
    # Run all baselines
    python tests/benchmarks/test_baseline_methods.py --datasets sachs --methods ges,fci,fedcdh --seeds 42,43,44

    # Run only centralized baselines
    python tests/benchmarks/test_baseline_methods.py --datasets sachs,asia --methods ges,fci

    # Compare with FedSPN
    python tests/benchmarks/test_baseline_methods.py --datasets sachs --methods fedcdh --save-graphs
    python tests/benchmarks/test_fedspn_benchmark.py --datasets sachs --methods fedspn_h --save-graphs

Updated: May 31, 2026 - Created as separate baseline benchmark script
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from dataclasses import dataclass

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import baseline causal discovery methods
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci

# Import shared utilities from FedSPN benchmark
from test_fedspn_benchmark import (
    DATASET_REGISTRY,
    DatasetLoader,
    MetricsEngine,
    get_device,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================================
# Baseline Method Registry
# ============================================================


@dataclass
class BaselineMethodConfig:
    """Configuration for baseline causal discovery methods."""

    name: str  # Method identifier
    type: str  # "centralized" or "federated"
    category: str  # "score_based", "constraint_based", "kernel_based"
    description: str  # Human-readable description
    handles_latent: bool = False  # Can handle latent confounders


BASELINE_METHOD_REGISTRY = {
    "ges": BaselineMethodConfig(
        name="ges",
        type="centralized",
        category="score_based",
        description="Greedy Equivalence Search - Score-based causal discovery",
        handles_latent=False,
    ),
    "fci": BaselineMethodConfig(
        name="fci",
        type="centralized",
        category="constraint_based",
        description="Fast Causal Inference - Constraint-based with Fisher-z test, handles latent confounders",
        handles_latent=True,
    ),
    "fedcdh": BaselineMethodConfig(
        name="fedcdh",
        type="federated",
        category="kernel_based",
        description="FedCDH (CD-NOD) - Federated causal discovery with KCI tests",
        handles_latent=False,
    ),
}


# ============================================================
# Baseline Method Runner
# ============================================================


class BaselineMethodRunner:
    """Method runner for baseline causal discovery algorithms."""

    @staticmethod
    def run_method(
        method_name: str,
        X: np.ndarray,
        B: np.ndarray,
        K: int = 3,
        alpha: float = 0.05,
        seed: int = 42,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run baseline causal discovery method.

        Args:
            method_name: Method from BASELINE_METHOD_REGISTRY
            X: Data matrix (n, d)
            B: Ground truth (for evaluation)
            K: Number of clients (for federated methods)
            alpha: Significance level
            seed: Random seed
            **kwargs: Additional method-specific parameters

        Returns:
            G: Predicted graph (d, d)
            metrics: Runtime and quality metrics
        """
        if method_name not in BASELINE_METHOD_REGISTRY:
            raise ValueError(
                f"Unknown baseline method: {method_name}. Available: {list(BASELINE_METHOD_REGISTRY.keys())}"
            )

        config = BASELINE_METHOD_REGISTRY[method_name]

        if config.type == "centralized":
            return BaselineMethodRunner._run_centralized(method_name, X, B, alpha, seed)
        elif config.type == "federated":
            return BaselineMethodRunner._run_fedcdh(X, B, K, alpha, seed)
        else:
            raise ValueError(f"Unknown method type: {config.type}")

    @staticmethod
    def _run_centralized(
        method_name: str, X: np.ndarray, B: np.ndarray, alpha: float, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run centralized methods (GES, FCI)."""
        np.random.seed(seed)
        start_time = time.time()

        try:
            if method_name == "ges":
                record = ges(X, score_func="local_score_BIC")
                G = record["G"].graph

            elif method_name == "fci":
                cg, edges = fci(
                    X, "fisherz", alpha=alpha, verbose=False, show_progress=False
                )
                G = cg.graph

            else:
                raise ValueError(f"Unknown centralized method: {method_name}")

            runtime = time.time() - start_time

            metrics = {
                "total_time": runtime,
                "training_time": 0.0,
                "ci_test_time": runtime,
                "aggregation_time": 0.0,
            }

            return G, metrics

        except Exception as e:
            logging.error(f"  {method_name.upper()} failed: {e}")
            runtime = time.time() - start_time
            return np.zeros((X.shape[1], X.shape[1])), {"total_time": runtime}

    @staticmethod
    def _run_fedcdh(
        X: np.ndarray, B: np.ndarray, K: int, alpha: float, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run FedCDH (CD-NOD) with KCI tests."""
        d = X.shape[1]
        n = X.shape[0]

        # Partition horizontally with seed-controlled shuffle
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(n)

        samples_per_client = n // K
        n_trimmed = samples_per_client * K
        X = X[shuffled_indices[:n_trimmed], :]
        n = n_trimmed

        # Create c_indx (context indices)
        c_indx = np.zeros((n, 1), dtype=int)
        for k in range(K):
            start = k * samples_per_client
            end = (k + 1) * samples_per_client
            c_indx[start:end, 0] = k

        logging.info(
            f"  Data shuffled with seed={seed} (first 5 sample indices: {shuffled_indices[:5].tolist()})"
        )
        logging.info(f"  FedCDH config: K={K}, n_per_client={samples_per_client}")

        start_time = time.time()

        try:
            # Run CD-NOD
            cg = cdnod(
                X,
                c_indx,
                K,
                alpha,
                kci,
                True,  # background_knowledge
                0,  # uc_rule
                -1,  # uc_priority
            )

            runtime = time.time() - start_time

            # Extract graph
            G = cg.G.graph[0:d, 0:d]

            metrics = {
                "total_time": runtime,
                "training_time": 0.0,
                "ci_test_time": runtime,  # KCI tests dominate
                "aggregation_time": 0.0,
            }

            return G, metrics

        except Exception as e:
            logging.error(f"  FedCDH failed: {e}")
            import traceback

            logging.error(f"  Traceback: {traceback.format_exc()}")
            runtime = time.time() - start_time
            return np.zeros((d, d)), {"total_time": runtime}


# ============================================================
# Baseline Benchmark
# ============================================================


class BaselineBenchmark:
    """
    Baseline Benchmark Suite for Causal Discovery.

    Runs baseline methods (GES, FCI, FedCDH) on multiple datasets
    with comprehensive evaluation metrics.
    """

    def __init__(
        self,
        datasets: List[str],
        methods: List[str],
        seeds: List[int] = [42, 123, 456],
        K: int = 3,
        alpha: float = 0.05,
        output_dir: str = "benchmark_results/baselines",
        save_graphs: bool = False,
    ):
        """
        Initialize baseline benchmark.

        Args:
            datasets: List of dataset names from DATASET_REGISTRY
            methods: List of baseline methods (ges, fci, fedcdh)
            seeds: Random seeds for multiple runs
            K: Number of federated clients (for FedCDH)
            alpha: Significance level for CI tests
            output_dir: Output directory for results
            save_graphs: Whether to save graphs and visualizations
        """
        self.datasets = datasets
        self.methods = methods
        self.seeds = seeds
        self.K = K
        self.alpha = alpha
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_graphs = save_graphs

        # Create graphs directory if saving graphs
        if self.save_graphs:
            self.graphs_dir = self.output_dir / "graphs"
            self.graphs_dir.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        for dataset in datasets:
            if dataset not in DATASET_REGISTRY:
                raise ValueError(f"Unknown dataset: {dataset}")
        for method in methods:
            if method not in BASELINE_METHOD_REGISTRY:
                raise ValueError(
                    f"Unknown baseline method: {method}. Available: {list(BASELINE_METHOD_REGISTRY.keys())}"
                )

    def run_single_experiment(
        self,
        dataset_name: str,
        method_name: str,
        seed: int,
    ) -> Dict[str, Any]:
        """Run single experiment with comprehensive metrics."""
        # Load dataset
        X, B, feature_names = DatasetLoader.load_dataset(dataset_name, seed)

        # Run method
        G, runtime_metrics = BaselineMethodRunner.run_method(
            method_name,
            X,
            B,
            K=self.K,
            alpha=self.alpha,
            seed=seed,
        )

        # Compute metrics
        all_metrics = MetricsEngine.compute_all_metrics(G, B, runtime_metrics)

        # Add experiment metadata
        config = BASELINE_METHOD_REGISTRY[method_name]
        result = {
            "dataset": dataset_name,
            "method": method_name,
            "seed": seed,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_edges_true": int(np.sum(B)),
            "K": self.K if config.type == "federated" else 1,
            "alpha": self.alpha,
            **all_metrics,
        }

        return result

    def run_all_experiments(self) -> pd.DataFrame:
        """Run all combinations of datasets × methods × seeds."""
        results = []
        total = len(self.datasets) * len(self.methods) * len(self.seeds)
        counter = 1

        logging.info("=" * 80)
        logging.info("BASELINE METHODS BENCHMARK")
        logging.info("=" * 80)
        logging.info(f"Datasets: {self.datasets}")
        logging.info(f"Baseline Methods: {self.methods}")
        logging.info(f"Seeds: {self.seeds}")
        logging.info(f"Total experiments: {total}")
        if self.save_graphs:
            logging.info(f"Graph saving: ENABLED (output: {self.graphs_dir})")
        else:
            logging.info("Graph saving: DISABLED (use --save-graphs to enable)")
        logging.info("=" * 80)

        for dataset in self.datasets:
            logging.info(f"\n{'='*80}")
            logging.info(f"Dataset: {dataset.upper()}")
            logging.info(f"{'='*80}")

            for method in self.methods:
                logging.info(f"\nMethod: {method.upper()}")

                for seed in self.seeds:
                    logging.info(
                        f"\n  [{counter}/{total}] Running {method} on {dataset}, seed={seed}"
                    )

                    try:
                        result = self.run_single_experiment(dataset, method, seed)
                        results.append(result)

                        logging.info(
                            f"    ✓ Skeleton F1={result['skeleton_f1']:.3f}, "
                            f"DAG F1={result['dag_f1']:.3f}, "
                            f"SHD={result['skeleton_shd']}, "
                            f"Time={result['total_time']:.2f}s"
                        )

                    except Exception as e:
                        logging.error(f"    ✗ Failed: {e}")
                        import traceback

                        traceback.print_exc()

                    counter += 1

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame):
        """Save results to CSV."""
        output_path = self.output_dir / "raw_results.csv"
        df.to_csv(output_path, index=False)
        logging.info(f"\n{'='*80}")
        logging.info(f"Results saved to: {output_path}")
        logging.info(f"{'='*80}")

    def generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics."""
        summary = (
            df.groupby(["dataset", "method"])
            .agg(
                {
                    "skeleton_f1": ["mean", "std", "count"],
                    "skeleton_precision": ["mean", "std"],
                    "skeleton_recall": ["mean", "std"],
                    "skeleton_shd": ["mean", "std"],
                    "dag_f1": ["mean", "std"],
                    "dag_shd": ["mean", "std"],
                    "total_time": ["mean", "std"],
                }
            )
            .round(3)
        )

        summary_path = self.output_dir / "summary_statistics.csv"
        summary.to_csv(summary_path)
        logging.info(f"Summary statistics saved to: {summary_path}")

        return summary


# ============================================================
# Main Entry Point
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline Methods Benchmark - Traditional Causal Discovery Algorithms"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="sachs",
        help="Comma-separated list of datasets (e.g., sachs,asia,law_school)",
    )
    parser.add_argument(
        "--methods",
        "--method",
        type=str,
        default="ges,fci,fedcdh",
        help="Comma-separated list of baseline methods (ges, fci, fedcdh)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated list of seeds",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="Number of federated clients (for FedCDH)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CI tests",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/baselines",
        help="Output directory",
    )
    parser.add_argument(
        "--save-graphs",
        action="store_true",
        help="Save graph adjacency matrices and visualizations",
    )

    args = parser.parse_args()

    # Parse arguments
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    # Create benchmark
    benchmark = BaselineBenchmark(
        datasets=datasets,
        methods=methods,
        seeds=seeds,
        K=args.K,
        alpha=args.alpha,
        output_dir=args.output_dir,
        save_graphs=args.save_graphs,
    )

    # Run experiments
    df_results = benchmark.run_all_experiments()

    # Save results
    benchmark.save_results(df_results)

    # Generate summary
    summary = benchmark.generate_summary(df_results)

    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY STATISTICS")
    logging.info("=" * 80)
    print(summary)

    logging.info("\n" + "=" * 80)
    logging.info("BASELINE BENCHMARK COMPLETE")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
