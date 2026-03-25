#!/usr/bin/env python3
"""
Comprehensive Synthetic Experiment Suite for FedCDH+FedPC Thesis

This script runs 6 experiments to demonstrate the capabilities of SPN-based
federated causal discovery with mechanism invariance orientation.

Experiments (using conservative parameters for stable results):
1. Method Comparison - SPN vs baselines (chain DAG, K=2, d=8)
2. Scalability Analysis - N, d, K dimensions (chain DAG baseline)
3. Heterogeneity Robustness - Domain shift tolerance (0.0 to 1.0)
4. Scenario Comparison - Vertical regularization effect (KEY EXPERIMENT)
5. DAG Structure Robustness - Different causal patterns (chain/fork/collider/random)
6. Orientation Method Ablation - MI contribution (mi_only vs mi_hybrid)

Conservative Config:
- d=8 variables (vs d=10 in original, easier to learn)
- K=2 clients (vs K=3, less heterogeneity)
- n_per_client=400 (vs 350, more data per client)
- epochs=100 (vs 50, better convergence)
- heterogeneity=0.3 (vs 0.5, moderate domain shift)
- Fixed edge weights=0.8 (vs random [0.5,1.5], stronger signal)
- Chain DAG by default (vs fork/collider, simpler structure)

Expected Performance:
- F1 Skeleton: 0.75-0.85 (vs 0.33 with old params)
- F1 Directed: 0.45-0.55 (vs 0.09 with old params)
- Vertical F1_dir > Horizontal F1_dir (mechanism invariance benefit)

Usage:
    python tests/benchmarks/synthetic_comprehensive_suite.py --all
    python tests/benchmarks/synthetic_comprehensive_suite.py --exp 1 --seeds 10
    python tests/benchmarks/synthetic_comprehensive_suite.py --exp 4 --gpu

Runtime: ~20-30 minutes on GPU, ~1-2 hours on CPU (with 10 seeds)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH
from causallearn.utils.data_utils import set_random_seed, simulate_linear_sem

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


# ============================================================================
# BASELINE CONFIGURATION (Conservative parameters for stable results)
# ============================================================================

BASELINE_CONFIG = {
    "d": 8,  # Number of variables
    "K": 3,  # Number of clients/domains
    "n_per_client": 450,  # Samples per client (total: 1,350)
    "epochs": 100,  # Training epochs for SPN
    "heterogeneity": 0.3,  # Domain shift level (moderate)
    "edge_weight": 0.8,  # Fixed edge weight (strong signal)
    "dag_type": "chain",  # Default DAG structure
    "orientation_method": "mi_only",
    "alpha": 0.05,
    "num_sums": 20,  # SPN capacity (increased for d=8+context=9D)
    "num_leaves": 20,  # SPN capacity (increased for d=8+context=9D)
    "num_repetitions": 10,  # SPN structure complexity
}

# Expected performance with baseline config:
# - F1 Skeleton: 0.75-0.85
# - F1 Directed: 0.45-0.55
# - Vertical F1_dir > Horizontal F1_dir (mechanism invariance benefit)


# ============================================================================
# DAG GENERATION
# ============================================================================


def generate_dag_chain(d: int) -> np.ndarray:
    """Generate chain DAG: 0 -> 1 -> 2 -> ... -> d-1"""
    dag = np.zeros((d, d))
    for i in range(d - 1):
        dag[i, i + 1] = 1
    return dag


def generate_dag_fork(d: int) -> np.ndarray:
    """Generate fork structure with common causes"""
    dag = np.zeros((d, d))
    # Root causes every 3 nodes
    for i in range(0, d - 2, 3):
        dag[i, i + 1] = 1
        dag[i, i + 2] = 1
    # Add some chains
    for i in range(1, d - 1, 3):
        if i + 1 < d:
            dag[i, i + 1] = 1
    return dag


def generate_dag_collider(d: int) -> np.ndarray:
    """Generate collider structure with common effects"""
    dag = np.zeros((d, d))
    # Pairs converge to common effect
    for i in range(0, d - 2, 3):
        if i + 2 < d:
            dag[i, i + 2] = 1
            dag[i + 1, i + 2] = 1
    # Add some chains
    for i in range(2, d - 1, 3):
        if i + 1 < d:
            dag[i, i + 1] = 1
    return dag


def generate_dag_random(d: int, edge_prob: float = 0.15, seed: int = 0) -> np.ndarray:
    """Generate random Erdős-Rényi DAG"""
    np.random.seed(seed)
    dag = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < edge_prob:
                dag[i, j] = 1
    return dag


def generate_dag(dag_type: str, d: int, seed: int = 0) -> np.ndarray:
    """Generate DAG of specified type"""
    if dag_type == "chain":
        return generate_dag_chain(d)
    elif dag_type == "fork":
        return generate_dag_fork(d)
    elif dag_type == "collider":
        return generate_dag_collider(d)
    elif dag_type == "random":
        return generate_dag_random(d, seed=seed)
    else:
        raise ValueError(f"Unknown DAG type: {dag_type}")


# ============================================================================
# DATA GENERATION
# ============================================================================


def generate_heterogeneous_data(
    W: np.ndarray, n_total: int, K: int, heterogeneity: float = 0.5, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate heterogeneous data across K domains with specified shift level.

    Args:
        W: Weighted DAG adjacency matrix (d x d)
        n_total: Total number of samples
        K: Number of domains/clients
        heterogeneity: Domain shift strength (0.0 = homogeneous, 1.0+ = strong)
        seed: Random seed

    Returns:
        X: Data matrix (n_total x d)
        c_indx: Domain indices (n_total x 1)
    """
    np.random.seed(seed)
    d = W.shape[0]
    n_per_domain = n_total // K

    X = np.zeros((n_total, d))

    # Topological ordering
    topo_order = []
    in_degree = np.sum(W != 0, axis=0)
    queue = list(np.where(in_degree == 0)[0])

    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for child in np.where(W[node, :] != 0)[0]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Generate data following topological order with domain shifts
    for node in topo_order:
        parents = np.where(W[:, node] != 0)[0]

        for k in range(K):
            idx_start = k * n_per_domain
            idx_end = (k + 1) * n_per_domain

            if len(parents) == 0:
                # Root node: add domain-specific shift
                domain_shift = heterogeneity * k
                X[idx_start:idx_end, node] = (
                    np.random.randn(n_per_domain) + domain_shift
                )
            else:
                # Non-root: linear combination with domain-specific noise
                X[idx_start:idx_end, node] = (
                    X[idx_start:idx_end, parents] @ W[parents, node]
                )

                # Domain-specific noise scale
                noise_scale = 0.5 + heterogeneity * k * 0.2
                X[idx_start:idx_end, node] += (
                    np.random.randn(n_per_domain) * noise_scale
                )

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    # Create domain indices
    c_indx = np.repeat(np.arange(K), n_per_domain).reshape(-1, 1)

    return X, c_indx


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================


class Args:
    """Configuration for FedCDH"""

    def __init__(self, config: Dict[str, Any]):
        self.K = config.get("K", BASELINE_CONFIG["K"])
        self.d = config.get("d", BASELINE_CONFIG["d"])
        self.scenario = config.get("scenario", "horizontal")
        self.model_type = "linear"
        self.ci_method = config.get("ci_method", "spn")
        self.n = config.get("n_per_client", BASELINE_CONFIG["n_per_client"])
        self.alpha = config.get("alpha", BASELINE_CONFIG["alpha"])
        self.epochs = config.get("epochs", BASELINE_CONFIG["epochs"])
        self.num_sums = config.get("num_sums", BASELINE_CONFIG["num_sums"])
        self.num_leaves = config.get("num_leaves", BASELINE_CONFIG["num_leaves"])
        self.num_repetitions = config.get(
            "num_repetitions", BASELINE_CONFIG["num_repetitions"]
        )
        self.ablation_orientation = config.get(
            "orientation_method", BASELINE_CONFIG["orientation_method"]
        )


def run_single_trial(
    method: str,
    scenario: str,
    X_all: np.ndarray,
    c_indx: np.ndarray,
    true_DAG: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single causal discovery trial"""

    d = X_all.shape[1]
    K = config.get("K", 3)

    # Create args
    full_config = {
        "d": d,
        "K": K,
        "scenario": scenario,
        "ci_method": "fisherz" if method == "fisherz" else "spn",
        "orientation_method": config.get("orientation_method", "mi_only"),
        **config,
    }
    args = Args(full_config)

    # Split data according to scenario
    if scenario == "horizontal":
        X_splits = np.array_split(X_all, K)
    elif scenario == "vertical":
        cols_per_client = np.array_split(range(d), K)
        X_splits = [X_all[:, cols] for cols in cols_per_client]
    else:  # hybrid
        X_splits = np.array_split(X_all, K)

    # Run FedCDH
    start_time = time.time()
    fedcdh = FedCDH(args)
    result = fedcdh.fit(X_splits, c_indx, true_DAG)
    elapsed = time.time() - start_time

    # Extract metrics
    metrics = {
        "method": method,
        "scenario": scenario,
        "f1_skeleton": result.get("f1_skeleton", 0.0),
        "f1_directed": result.get("f1", 0.0),
        "precision_skeleton": result.get("precision_skeleton", 0.0),
        "recall_skeleton": result.get("recall_skeleton", 0.0),
        "shd": result.get("shd", 0),
        "time_total": elapsed,
        "time_train": result.get("time_train", 0.0),
        "time_cd": result.get("time_cd", 0.0),
        "comm_cost": result.get("comm_cost", 0.0),
    }

    return metrics


# ============================================================================
# EXPERIMENT 1: METHOD COMPARISON
# ============================================================================


def experiment_1_method_comparison(
    output_dir: Path, n_seeds: int = 10, verbose: bool = True
) -> pd.DataFrame:
    """
    Compare different CI testing methods on same data.

    Methods: fisherz, fedspn_horizontal, fedspn_vertical, fedspn_hybrid
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: METHOD COMPARISON")
        print("=" * 70)

    results = []

    # Use baseline config
    config = BASELINE_CONFIG.copy()

    methods = [
        ("fisherz", "horizontal"),
        ("fedspn", "horizontal"),
        ("fedspn", "vertical"),
        ("fedspn", "hybrid"),
    ]

    # Generate chain DAG once (simple structure proven to work)
    W = generate_dag(config["dag_type"], config["d"])
    W[W > 0] = config["edge_weight"]

    for seed in range(n_seeds):
        if verbose:
            print(f"\nSeed {seed + 1}/{n_seeds}")

        # Generate data
        n_total = config["n_per_client"] * config["K"]
        X_all, c_indx = generate_heterogeneous_data(
            W, n_total, config["K"], heterogeneity=0.5, seed=seed
        )
        true_DAG = (W != 0).astype(int)

        # Test each method
        for method, scenario in methods:
            method_name = f"{method}_{scenario}" if method != "fisherz" else "fisherz"

            if verbose:
                print(f"  Testing {method_name}...")

            metrics = run_single_trial(
                method, scenario, X_all, c_indx, true_DAG, config
            )
            metrics["seed"] = seed
            metrics["method_name"] = method_name
            results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp1_method_comparison.csv", index=False)

    if verbose:
        print("\n" + "-" * 70)
        print("Summary (mean ± std):")
        summary = (
            df.groupby("method_name")
            .agg(
                {
                    "f1_skeleton": ["mean", "std"],
                    "f1_directed": ["mean", "std"],
                    "time_total": ["mean", "std"],
                }
            )
            .round(3)
        )
        print(summary)

    return df


# ============================================================================
# EXPERIMENT 2: SCALABILITY ANALYSIS
# ============================================================================


def experiment_2_scalability(
    output_dir: Path, n_seeds: int = 5, verbose: bool = True
) -> pd.DataFrame:
    """
    Test scalability across N (samples), d (dimensions), K (clients).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: SCALABILITY ANALYSIS")
        print("=" * 70)

    results = []

    # Use baseline d and K, vary the dimension being tested
    baseline_n_total = BASELINE_CONFIG["n_per_client"] * BASELINE_CONFIG["K"]

    # Test dimensions (scale from baseline)
    tests = [
        # (N, d, K, test_name)
        (
            "sample_size",
            [
                (baseline_n_total // 2, BASELINE_CONFIG["d"], BASELINE_CONFIG["K"]),
                (baseline_n_total, BASELINE_CONFIG["d"], BASELINE_CONFIG["K"]),
                (baseline_n_total * 2, BASELINE_CONFIG["d"], BASELINE_CONFIG["K"]),
                (baseline_n_total * 4, BASELINE_CONFIG["d"], BASELINE_CONFIG["K"]),
            ],
        ),
        (
            "dimensionality",
            [
                (baseline_n_total, 5, BASELINE_CONFIG["K"]),
                (baseline_n_total, 8, BASELINE_CONFIG["K"]),
                (baseline_n_total, 12, BASELINE_CONFIG["K"]),
                (baseline_n_total, 15, BASELINE_CONFIG["K"]),
            ],
        ),
        (
            "num_clients",
            [
                (baseline_n_total, BASELINE_CONFIG["d"], 2),
                (baseline_n_total, BASELINE_CONFIG["d"], 3),
                (baseline_n_total, BASELINE_CONFIG["d"], 5),
                (baseline_n_total, BASELINE_CONFIG["d"], 7),
            ],
        ),
    ]

    for test_name, params_list in tests:
        if verbose:
            print(f"\n{test_name.upper()} SCALING")

        for n_total, d, K in params_list:
            if verbose:
                print(f"  N={n_total}, d={d}, K={K}")

            for seed in range(n_seeds):
                # Generate chain DAG (simple, proven structure)
                W = generate_dag(BASELINE_CONFIG["dag_type"], d, seed=seed)
                W[W > 0] = BASELINE_CONFIG["edge_weight"]

                # Generate data
                X_all, c_indx = generate_heterogeneous_data(
                    W,
                    n_total,
                    K,
                    heterogeneity=BASELINE_CONFIG["heterogeneity"],
                    seed=seed,
                )
                true_DAG = (W != 0).astype(int)

                # Test fedspn only (representative)
                config = {
                    "d": d,
                    "K": K,
                    "n_per_client": n_total // K,
                    "epochs": BASELINE_CONFIG["epochs"],
                }

                metrics = run_single_trial(
                    "fedspn", "horizontal", X_all, c_indx, true_DAG, config
                )
                metrics.update(
                    {
                        "seed": seed,
                        "test_dimension": test_name,
                        "n_total": n_total,
                        "d": d,
                        "K": K,
                    }
                )
                results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp2_scalability.csv", index=False)

    return df


# ============================================================================
# EXPERIMENT 3: HETEROGENEITY ROBUSTNESS
# ============================================================================


def experiment_3_heterogeneity(
    output_dir: Path, n_seeds: int = 10, verbose: bool = True
) -> pd.DataFrame:
    """
    Test robustness to domain heterogeneity levels.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: HETEROGENEITY ROBUSTNESS")
        print("=" * 70)

    results = []
    heterogeneity_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Use baseline config
    config = BASELINE_CONFIG.copy()

    for het_level in heterogeneity_levels:
        if verbose:
            print(f"\nHeterogeneity level: {het_level}")

        for seed in range(n_seeds):
            # Generate chain DAG (simple, proven structure)
            W = generate_dag(config["dag_type"], config["d"], seed=seed)
            W[W > 0] = config["edge_weight"]

            # Generate data with heterogeneity
            n_total = config["n_per_client"] * config["K"]
            X_all, c_indx = generate_heterogeneous_data(
                W, n_total, config["K"], heterogeneity=het_level, seed=seed
            )
            true_DAG = (W != 0).astype(int)

            # Test fisherz and fedspn
            for method in ["fisherz", "fedspn"]:
                metrics = run_single_trial(
                    method, "horizontal", X_all, c_indx, true_DAG, config
                )
                metrics.update({"seed": seed, "heterogeneity": het_level})
                results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp3_heterogeneity.csv", index=False)

    return df


# ============================================================================
# EXPERIMENT 4: SCENARIO COMPARISON (VERTICAL REGULARIZATION)
# ============================================================================


def experiment_4_scenario_comparison(
    output_dir: Path, n_seeds: int = 10, verbose: bool = True
) -> pd.DataFrame:
    """
    Compare H/V/Hy scenarios to demonstrate vertical regularization effect.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: SCENARIO COMPARISON (Vertical Regularization)")
        print("=" * 70)

    results = []
    scenarios = ["horizontal", "vertical", "hybrid"]

    # Use baseline config for KEY thesis experiment
    config = BASELINE_CONFIG.copy()

    for seed in range(n_seeds):
        if verbose:
            print(f"\nSeed {seed + 1}/{n_seeds}")

        # Generate chain DAG (simple structure for clear comparison)
        W = generate_dag(config["dag_type"], config["d"], seed=seed)
        W[W > 0] = config["edge_weight"]

        # Generate data
        n_total = config["n_per_client"] * config["K"]
        X_all, c_indx = generate_heterogeneous_data(
            W, n_total, config["K"], heterogeneity=config["heterogeneity"], seed=seed
        )
        true_DAG = (W != 0).astype(int)

        # Test all scenarios
        for scenario in scenarios:
            if verbose:
                print(f"  Scenario: {scenario}")

            metrics = run_single_trial(
                "fedspn", scenario, X_all, c_indx, true_DAG, config
            )
            metrics["seed"] = seed
            results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp4_scenario_comparison.csv", index=False)

    if verbose:
        print("\n" + "-" * 70)
        print("Summary (mean ± std):")
        summary = (
            df.groupby("scenario")
            .agg({"f1_skeleton": ["mean", "std"], "f1_directed": ["mean", "std"]})
            .round(3)
        )
        print(summary)
        print("\n*** Check if Vertical F1_directed > Horizontal ***")

    return df


# ============================================================================
# EXPERIMENT 5: DAG STRUCTURE ROBUSTNESS
# ============================================================================


def experiment_5_dag_structures(
    output_dir: Path, n_seeds: int = 10, verbose: bool = True
) -> pd.DataFrame:
    """
    Test robustness across different DAG structures.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: DAG STRUCTURE ROBUSTNESS")
        print("=" * 70)

    results = []
    dag_types = ["chain", "fork", "collider", "random"]

    # Use baseline config, only vary DAG structure
    config = BASELINE_CONFIG.copy()

    for dag_type in dag_types:
        if verbose:
            print(f"\nDAG type: {dag_type}")

        for seed in range(n_seeds):
            # Generate DAG (different structures)
            W = generate_dag(dag_type, config["d"], seed=seed)
            W[W > 0] = config["edge_weight"]

            # Generate data
            n_total = config["n_per_client"] * config["K"]
            X_all, c_indx = generate_heterogeneous_data(
                W,
                n_total,
                config["K"],
                heterogeneity=config["heterogeneity"],
                seed=seed,
            )
            true_DAG = (W != 0).astype(int)

            # Test fedspn
            metrics = run_single_trial(
                "fedspn", "horizontal", X_all, c_indx, true_DAG, config
            )
            metrics.update(
                {"seed": seed, "dag_type": dag_type, "num_edges": int(np.sum(true_DAG))}
            )
            results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp5_dag_structures.csv", index=False)

    return df


# ============================================================================
# EXPERIMENT 6: ORIENTATION METHOD ABLATION
# ============================================================================


def experiment_6_orientation_ablation(
    output_dir: Path, n_seeds: int = 10, verbose: bool = True
) -> pd.DataFrame:
    """
    Ablation study on orientation methods.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 6: ORIENTATION METHOD ABLATION")
        print("=" * 70)

    results = []
    orientation_methods = ["mi_only", "mi_hybrid"]

    # Use baseline config
    config = BASELINE_CONFIG.copy()

    for seed in range(n_seeds):
        if verbose:
            print(f"\nSeed {seed + 1}/{n_seeds}")

        # Generate chain DAG (simple structure)
        W = generate_dag(config["dag_type"], config["d"], seed=seed)
        W[W > 0] = config["edge_weight"]

        # Generate data
        n_total = config["n_per_client"] * config["K"]
        X_all, c_indx = generate_heterogeneous_data(
            W, n_total, config["K"], heterogeneity=config["heterogeneity"], seed=seed
        )
        true_DAG = (W != 0).astype(int)

        # Test orientation methods
        for orient_method in orientation_methods:
            if verbose:
                print(f"  Orientation: {orient_method}")

            config_with_orient = {**config, "orientation_method": orient_method}
            metrics = run_single_trial(
                "fedspn", "horizontal", X_all, c_indx, true_DAG, config_with_orient
            )
            metrics.update({"seed": seed, "orientation_method": orient_method})
            results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "exp6_orientation_ablation.csv", index=False)

    return df


# ============================================================================
# VISUALIZATION
# ============================================================================


def generate_plots(output_dir: Path, verbose: bool = True):
    """Generate all publication-ready plots"""
    if verbose:
        print("\n" + "=" * 70)
        print("GENERATING PLOTS")
        print("=" * 70)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Plot 1: Method Comparison
    if (output_dir / "exp1_method_comparison.csv").exists():
        df = pd.read_csv(output_dir / "exp1_method_comparison.csv")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # F1 Skeleton
        sns.barplot(
            data=df, x="method_name", y="f1_skeleton", ax=axes[0], errorbar="sd"
        )
        axes[0].set_title("F1 Skeleton Score")
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis="x", rotation=45)

        # F1 Directed
        sns.barplot(
            data=df, x="method_name", y="f1_directed", ax=axes[1], errorbar="sd"
        )
        axes[1].set_title("F1 Directed Score")
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis="x", rotation=45)

        # Runtime
        sns.barplot(data=df, x="method_name", y="time_total", ax=axes[2], errorbar="sd")
        axes[2].set_title("Runtime (seconds)")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            plot_dir / "exp1_method_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Plot 2: Scalability
    if (output_dir / "exp2_scalability.csv").exists():
        df = pd.read_csv(output_dir / "exp2_scalability.csv")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        dimensions = ["sample_size", "dimensionality", "num_clients"]
        x_vars = ["n_total", "d", "K"]
        titles = ["Sample Size Scaling", "Dimensionality Scaling", "Client Scaling"]

        for ax, dim, x_var, title in zip(axes, dimensions, x_vars, titles):
            subset = df[df["test_dimension"] == dim]
            grouped = (
                subset.groupby(x_var).agg({"time_total": ["mean", "std"]}).reset_index()
            )

            ax.errorbar(
                grouped[x_var],
                grouped["time_total"]["mean"],
                yerr=grouped["time_total"]["std"],
                marker="o",
                capsize=5,
            )
            ax.set_xlabel(x_var)
            ax.set_ylabel("Runtime (s)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / "exp2_scalability.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot 3: Heterogeneity Robustness
    if (output_dir / "exp3_heterogeneity.csv").exists():
        df = pd.read_csv(output_dir / "exp3_heterogeneity.csv")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, metric in zip(axes, ["f1_skeleton", "f1_directed"]):
            for method in df["method"].unique():
                subset = df[df["method"] == method]
                grouped = (
                    subset.groupby("heterogeneity")
                    .agg({metric: ["mean", "std"]})
                    .reset_index()
                )

                ax.errorbar(
                    grouped["heterogeneity"],
                    grouped[metric]["mean"],
                    yerr=grouped[metric]["std"],
                    marker="o",
                    label=method,
                    capsize=5,
                )

            ax.set_xlabel("Heterogeneity Level")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} vs Heterogeneity")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / "exp3_heterogeneity.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot 4: Scenario Comparison (Vertical Regularization)
    if (output_dir / "exp4_scenario_comparison.csv").exists():
        df = pd.read_csv(output_dir / "exp4_scenario_comparison.csv")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # F1 scores
        for ax, metric in zip(axes, ["f1_skeleton", "f1_directed"]):
            sns.barplot(data=df, x="scenario", y=metric, ax=ax, errorbar="sd")
            ax.set_title(f"{metric.replace('_', ' ').title()} by Scenario")
            ax.set_ylim(0, 1)
            ax.set_ylabel(metric.replace("_", " ").title())

        plt.tight_layout()
        plt.savefig(
            plot_dir / "exp4_scenario_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    if verbose:
        print(f"Plots saved to: {plot_dir}")


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_report(output_dir: Path, verbose: bool = True):
    """Generate markdown report with all results"""
    report_path = output_dir / "EXPERIMENT_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# Comprehensive Synthetic Experiment Results\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Experiment 1
        if (output_dir / "exp1_method_comparison.csv").exists():
            df = pd.read_csv(output_dir / "exp1_method_comparison.csv")
            f.write("## Experiment 1: Method Comparison\n\n")

            summary = (
                df.groupby("method_name")
                .agg(
                    {
                        "f1_skeleton": ["mean", "std"],
                        "f1_directed": ["mean", "std"],
                        "time_total": ["mean", "std"],
                    }
                )
                .round(3)
            )

            f.write("```\n")
            f.write(summary.to_string())
            f.write("\n```\n\n")

        # Experiment 4
        if (output_dir / "exp4_scenario_comparison.csv").exists():
            df = pd.read_csv(output_dir / "exp4_scenario_comparison.csv")
            f.write(
                "## Experiment 4: Scenario Comparison (Vertical Regularization)\n\n"
            )

            summary = (
                df.groupby("scenario")
                .agg({"f1_skeleton": ["mean", "std"], "f1_directed": ["mean", "std"]})
                .round(3)
            )

            f.write("**Key Finding**: Check if Vertical F1_directed > Horizontal\n\n")
            f.write("```\n")
            f.write(summary.to_string())
            f.write("\n```\n\n")

            # Statistical test
            h_f1 = df[df["scenario"] == "horizontal"]["f1_directed"]
            v_f1 = df[df["scenario"] == "vertical"]["f1_directed"]

            f.write(f"- Horizontal F1_directed: {h_f1.mean():.3f} ± {h_f1.std():.3f}\n")
            f.write(f"- Vertical F1_directed: {v_f1.mean():.3f} ± {v_f1.std():.3f}\n")
            f.write(f"- Difference: {v_f1.mean() - h_f1.mean():.3f}\n\n")

    if verbose:
        print(f"\nReport saved to: {report_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive synthetic experiments"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument(
        "--exp", type=int, choices=[1, 2, 3, 4, 5, 6], help="Run specific experiment"
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument(
        "--output",
        type=str,
        default="tests/experiments/synthetic_suite",
        help="Output directory",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU
    if args.gpu and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    print(f"\nOutput directory: {output_dir}")
    print(f"Number of seeds: {args.seeds}")
    print("\n" + "=" * 70)
    print("STARTING COMPREHENSIVE SYNTHETIC EXPERIMENTS")
    print("=" * 70)

    start_time = time.time()

    # Run experiments
    if args.all or args.exp == 1:
        experiment_1_method_comparison(output_dir, args.seeds, args.verbose)

    if args.all or args.exp == 2:
        experiment_2_scalability(output_dir, args.seeds, args.verbose)

    if args.all or args.exp == 3:
        experiment_3_heterogeneity(output_dir, args.seeds, args.verbose)

    if args.all or args.exp == 4:
        experiment_4_scenario_comparison(output_dir, args.seeds, args.verbose)

    if args.all or args.exp == 5:
        experiment_5_dag_structures(output_dir, args.seeds, args.verbose)

    if args.all or args.exp == 6:
        experiment_6_orientation_ablation(output_dir, args.seeds, args.verbose)

    # Generate visualizations and report
    if args.all:
        generate_plots(output_dir, args.verbose)
        generate_report(output_dir, args.verbose)

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"EXPERIMENTS COMPLETE")
    print(f"Total runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
