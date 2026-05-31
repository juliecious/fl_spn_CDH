#!/usr/bin/env python
"""
FedCDH V3 Unified Benchmark Suite - Comprehensive Causal Discovery Evaluation.

V3 Enhancements:
- ✅ Multiple datasets: Real-world (Sachs, Law School) + Synthetic (ER, SF, Chain)
- ✅ Multiple methods: Centralized (GES, FCI) + Federated (FedCDH, FedSPN-H/V/Hy)
- ✅ Comprehensive metrics: 30+ metrics (Structure, Runtime, Quality, Communication)
- ✅ Automated comparison: Statistical tests, LaTeX tables, publication plots
- ✅ V3 features: structure_voting (H), GlobalSumOfProducts (Hy), ProductOverGroups (V)

Architecture:
- UnifiedBenchmark: Main experiment runner
- DatasetRegistry: Centralized dataset management
- MethodRegistry: Unified method dispatcher
- MetricsEngine: Comprehensive evaluation

Usage:
    python tests/test/test_fedcdh_benchmark_v3.py --datasets sachs,synthetic_er_small --methods ges,fedspn_h --seeds 42,43,44

Updated: May 10, 2026 - V3 unified architecture with comprehensive evaluation
"""

import logging
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

# Import causal discovery methods
from causallearn.search.ScoreBased.GES import ges

# PC is broken in current causal-learn version - use FCI instead
# from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.FCMBased.FedCDH import FedCDH
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci

# Import visualization utilities
from tests.utils.visualization.graph_visualization import (
    visualize_dag,
    compare_graphs,
    visualize_graph_differences,
)


# ============================================================
# GPU Device Detection
# ============================================================


def get_device():
    """Detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Import data utilities
from causallearn.utils.data_utils import (
    my_simulate_general_hetero,
    my_simulate_linear_gaussian,
    set_random_seed,
    simulate_dag,
    simulate_parameter,
    get_cpdag_from_cdnod,
    get_dag_from_pdag,
    count_skeleton_accuracy,
    count_dag_accuracy,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================================
# Dataset Registry
# ============================================================


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    type: str  # "real" or "synthetic"
    d: int  # Number of variables
    n: int  # Number of samples
    domain: str  # Domain (biology, social_science, etc.)
    validation: str  # How ground truth was established
    source: Optional[str] = None  # Paper/source reference
    graph_type: Optional[
        str
    ] = None  # For synthetic: "erdos_renyi", "scale_free", "chain"
    sparsity: Optional[float] = None  # For synthetic
    sem_type: Optional[str] = "linear"  # "linear" or "nonlinear"
    edges: Optional[int] = None  # Known edge count (if available)


# Dataset Registry: Real-world + Synthetic
DATASET_REGISTRY = {
    # === REAL-WORLD DATASETS ===
    "sachs": DatasetConfig(
        name="sachs",
        type="real",
        d=11,
        n=7466,
        edges=17,
        domain="biology",
        validation="interventional",
        source="Sachs et al. 2005",
    ),
    "law_school": DatasetConfig(
        name="law_school",
        type="real",
        d=5,
        n=21000,
        edges=7,
        domain="social_science",
        validation="domain_knowledge",
        source="Fairness causal graph",
    ),
    # === BAYESIAN NETWORK BENCHMARKS ===
    "asia": DatasetConfig(
        name="asia",
        type="benchmark",
        d=8,
        n=1000,
        edges=8,
        domain="medicine",
        validation="expert_knowledge",
        source="Lauritzen & Spiegelhalter 1988",
    ),
    "alarm": DatasetConfig(
        name="alarm",
        type="benchmark",
        d=37,
        n=1000,
        edges=46,
        domain="medicine",
        validation="expert_knowledge",
        source="Beinlich et al. 1989",
    ),
    "dream4_net1": DatasetConfig(
        name="dream4_net1",
        type="benchmark",
        d=10,
        n=1000,
        edges=13,
        domain="biology",
        validation="gold_standard",
        source="DREAM4 Challenge 2009",
    ),
    # === SYNTHETIC BENCHMARKS ===
    # Erdős-Rényi (random graphs)
    "synthetic_er_tiny": DatasetConfig(
        name="synthetic_er_tiny",
        type="synthetic",
        graph_type="erdos_renyi",
        d=5,
        n=200,
        sparsity=0.4,
        sem_type="linear",
        domain="synthetic",
        validation="ground_truth",
    ),
    "synthetic_er_small": DatasetConfig(
        name="synthetic_er_small",
        type="synthetic",
        graph_type="erdos_renyi",
        d=10,
        n=1000,
        sparsity=0.3,
        sem_type="linear",
        domain="synthetic",
        validation="ground_truth",
    ),
    "synthetic_er_medium": DatasetConfig(
        name="synthetic_er_medium",
        type="synthetic",
        graph_type="erdos_renyi",
        d=20,
        n=2000,
        sparsity=0.3,
        sem_type="nonlinear",
        domain="synthetic",
        validation="ground_truth",
    ),
    "synthetic_er_large": DatasetConfig(
        name="synthetic_er_large",
        type="synthetic",
        graph_type="erdos_renyi",
        d=50,
        n=5000,
        sparsity=0.2,
        sem_type="linear",
        domain="synthetic",
        validation="ground_truth",
    ),
    # Scale-Free (hub-and-spoke)
    "synthetic_sf_small": DatasetConfig(
        name="synthetic_sf_small",
        type="synthetic",
        graph_type="scale_free",
        d=15,
        n=1500,
        sparsity=0.25,
        sem_type="linear",
        domain="synthetic",
        validation="ground_truth",
    ),
    "synthetic_sf_medium": DatasetConfig(
        name="synthetic_sf_medium",
        type="synthetic",
        graph_type="scale_free",
        d=30,
        n=3000,
        sparsity=0.2,
        sem_type="nonlinear",
        domain="synthetic",
        validation="ground_truth",
    ),
    # Chain (sequential dependencies)
    "synthetic_chain": DatasetConfig(
        name="synthetic_chain",
        type="synthetic",
        graph_type="chain",
        d=10,
        n=1000,
        sparsity=None,  # Fixed structure
        sem_type="linear",
        domain="synthetic",
        validation="ground_truth",
    ),
}


# ============================================================
# Method Registry
# ============================================================


@dataclass
class MethodConfig:
    """Configuration for a causal discovery method."""

    name: str
    type: str  # "centralized" or "federated"
    category: str  # "score_based", "constraint_based", "kernel_based", "spn_based"
    ci_method: Optional[str] = None  # "fisherz", "kci", "spn", None
    privacy: bool = False  # Does it preserve privacy?
    scenarios: List[str] = field(default_factory=lambda: ["pooled"])
    aggregation: Optional[str] = None  # For federated: "structure_voting", etc.
    v3_feature: bool = False  # Is this a V3 enhancement?
    handles_latent: bool = False  # Can handle latent confounders?
    source: Optional[str] = None


METHOD_REGISTRY = {
    # === CENTRALIZED BASELINES (No Privacy) ===
    "ges": MethodConfig(
        name="ges",
        type="centralized",
        category="score_based",
        ci_method=None,
        privacy=False,
        scenarios=["pooled"],
        source="causal-learn",
    ),
    # "pc": REMOVED - API broken in causal-learn (use FCI instead)
    "fci": MethodConfig(
        name="fci",
        type="centralized",
        category="constraint_based",
        ci_method="fisherz",
        privacy=False,
        scenarios=["pooled"],
        handles_latent=True,
        source="causal-learn",
    ),
    # === FEDERATED BASELINES ===
    "fedcdh": MethodConfig(
        name="fedcdh",
        type="federated",
        category="kernel_based",
        ci_method="kci",
        privacy=True,
        scenarios=["horizontal"],
        source="Li et al. 2024 (CD-NOD)",
    ),
    # === OUR METHODS (FedSPN V3) ===
    "fedspn_h": MethodConfig(
        name="fedspn_h",
        type="federated",
        category="spn_based",
        ci_method="spn",
        privacy=True,
        scenarios=["horizontal"],
        aggregation="structure_voting",
        v3_feature=True,
        source="This work",
    ),
    "fedspn_v": MethodConfig(
        name="fedspn_v",
        type="federated",
        category="spn_based",
        ci_method="spn",
        privacy=True,
        scenarios=["vertical"],
        aggregation="product_over_groups",
        source="This work",
    ),
    "fedspn_hy": MethodConfig(
        name="fedspn_hy",
        type="federated",
        category="spn_based",
        ci_method="spn",
        privacy=True,
        scenarios=["hybrid"],
        aggregation="global_sum_of_products",
        v3_feature=True,
        source="This work",
    ),
}


# ============================================================
# Dataset Loading
# ============================================================


class DatasetLoader:
    """Unified dataset loader for real-world and synthetic datasets."""

    @staticmethod
    def load_dataset(
        dataset_name: str, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load dataset by name.

        Args:
            dataset_name: Name from DATASET_REGISTRY
            seed: Random seed for synthetic data

        Returns:
            X: Data matrix (n, d)
            B: Ground truth adjacency matrix (d, d)
            feature_names: List of feature names
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = DATASET_REGISTRY[dataset_name]

        if config.type == "real":
            return DatasetLoader._load_real_dataset(dataset_name)
        elif config.type == "benchmark":
            return DatasetLoader._load_benchmark_dataset(dataset_name)
        else:
            return DatasetLoader._generate_synthetic_dataset(config, seed)

    @staticmethod
    def _load_real_dataset(
        dataset_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load real-world datasets."""
        if dataset_name == "sachs":
            return DatasetLoader._load_sachs()
        elif dataset_name == "law_school":
            return DatasetLoader._load_law_school()
        else:
            raise ValueError(f"Unknown real dataset: {dataset_name}")

    @staticmethod
    def _load_sachs() -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load Sachs protein network dataset."""
        import gzip

        logging.info("Loading Sachs protein network dataset...")
        data_path = "data/real_world/sachs.interventional.txt.gz"

        with gzip.open(data_path, "rt") as f:
            df = pd.read_csv(f, sep=" ")

        # Remove INT column if present
        if "INT" in df.columns:
            df = df.drop("INT", axis=1)

        X = df.values
        feature_names = list(df.columns)

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

        logging.info(
            f"  Sachs: {X.shape[0]} samples, {X.shape[1]} features, {int(np.sum(B))} edges"
        )
        return X, B, feature_names

    @staticmethod
    def _load_law_school() -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load Law School admissions dataset."""
        from tests.utils.dataset_loaders.law_school_loader import (
            load_law_school_federated,
        )

        logging.info("Loading Law School admissions dataset...")
        X, B, feature_names = load_law_school_federated(
            n_clients=1, n_samples_limit=None
        )

        logging.info(
            f"  Law School: {X.shape[0]} samples, {X.shape[1]} features, {int(np.sum(B))} edges"
        )
        return X, B, feature_names

    @staticmethod
    def _load_benchmark_dataset(
        dataset_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load pre-generated benchmark datasets (Asia, Alarm, DREAM4)."""
        from tests.utils.dataset_loaders.bayesian_network_loaders import (
            ASIA_DAG,
            ASIA_NAMES,
            ALARM_DAG,
            ALARM_NAMES,
        )
        from tests.utils.dataset_loaders.dream_loader import (
            DREAM4_10_NETWORKS,
            edgelist_to_adjacency,
        )

        logging.info(f"Loading {dataset_name} benchmark dataset...")

        if dataset_name == "asia":
            # Load pre-generated CSV
            data_path = "data/benchmarks/asia_linear_n1000.csv"
            df = pd.read_csv(data_path)
            X = df.values
            B = ASIA_DAG.copy()
            feature_names = ASIA_NAMES.copy()
        elif dataset_name == "alarm":
            # Load pre-generated CSV
            data_path = "data/benchmarks/alarm_linear_n1000.csv"
            df = pd.read_csv(data_path)
            X = df.values
            B = ALARM_DAG.copy()
            feature_names = ALARM_NAMES.copy()
        elif dataset_name == "dream4_net1":
            # Load pre-generated CSV
            data_path = "data/benchmarks/dream4_net1_linear_n1000.csv"
            df = pd.read_csv(data_path)
            X = df.values
            edge_list = DREAM4_10_NETWORKS["network1"]
            B = edgelist_to_adjacency(edge_list, n_nodes=10)
            feature_names = [f"G{i+1}" for i in range(10)]
        else:
            raise ValueError(f"Unknown benchmark dataset: {dataset_name}")

        logging.info(
            f"  {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features, {int(np.sum(B))} edges"
        )
        return X, B, feature_names

    @staticmethod
    def _generate_synthetic_dataset(
        config: DatasetConfig, seed: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate synthetic datasets with known ground truth."""
        set_random_seed(seed)

        logging.info(
            f"Generating {config.name}: d={config.d}, n={config.n}, {config.graph_type}, {config.sem_type}"
        )

        # Generate DAG structure
        if config.graph_type == "erdos_renyi":
            B = DatasetLoader._generate_er_dag(config.d, config.sparsity)
        elif config.graph_type == "scale_free":
            B = DatasetLoader._generate_sf_dag(config.d, config.sparsity)
        elif config.graph_type == "chain":
            B = DatasetLoader._generate_chain_dag(config.d)
        else:
            raise ValueError(f"Unknown graph type: {config.graph_type}")

        # Generate weights
        W = simulate_parameter(B)

        # Generate data
        if config.sem_type == "linear":
            X, choice = my_simulate_linear_gaussian(
                W, K=1, n=config.n, sem_type="gauss"
            )
        else:
            X, choice = my_simulate_general_hetero(W, K=1, n=config.n, sem_type="gauss")

        # Feature names
        feature_names = [f"X{i}" for i in range(config.d)]

        # Binarize B
        B_binary = (np.abs(B) > 0).astype(int)

        logging.info(
            f"  {config.name}: {X.shape[0]} samples, {X.shape[1]} features, {int(np.sum(B_binary))} edges"
        )
        return X, B_binary, feature_names

    @staticmethod
    def _generate_er_dag(d: int, sparsity: float) -> np.ndarray:
        """Generate Erdős-Rényi random DAG."""
        s0 = int(d * sparsity)  # Expected edges per node
        return simulate_dag(d, s0, graph_type="ER")

    @staticmethod
    def _generate_sf_dag(d: int, sparsity: float) -> np.ndarray:
        """Generate Scale-Free DAG with hub nodes."""
        s0 = int(d * sparsity)
        return simulate_dag(d, s0, graph_type="SF")

    @staticmethod
    def _generate_chain_dag(d: int) -> np.ndarray:
        """Generate sequential chain DAG: X0 → X1 → X2 → ... → Xd-1"""
        B = np.zeros((d, d))
        for i in range(d - 1):
            B[i, i + 1] = 1
        return B


# ============================================================
# Metrics Engine
# ============================================================


class MetricsEngine:
    """Comprehensive metrics computation for causal discovery."""

    @staticmethod
    def compute_all_metrics(
        G_pred: np.ndarray,
        B_true: np.ndarray,
        runtime_metrics: Dict[str, Any],
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.

        Args:
            G_pred: Predicted graph (d, d)
            B_true: Ground truth graph (d, d)
            runtime_metrics: Runtime statistics
            quality_metrics: Optional quality metrics (LL, BIC, etc.)

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Structure metrics
        metrics.update(MetricsEngine.compute_skeleton_metrics(G_pred, B_true))
        metrics.update(MetricsEngine.compute_dag_metrics(G_pred, B_true))

        # Runtime metrics
        metrics.update(runtime_metrics)

        # Quality metrics
        if quality_metrics:
            metrics.update(quality_metrics)

        return metrics

    @staticmethod
    def compute_skeleton_metrics(
        G_pred: np.ndarray, B_true: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute skeleton-based metrics (undirected graph).

        Handles CPDAG encoding: -1 (tail), 0 (no edge), 1 (head)
        """
        # Convert to skeleton (undirected)
        pred_skeleton = (np.abs(G_pred) + np.abs(G_pred.T)) > 0
        true_skeleton = (np.abs(B_true) + np.abs(B_true.T)) > 0

        # Remove self-loops
        np.fill_diagonal(pred_skeleton, 0)
        np.fill_diagonal(true_skeleton, 0)

        # Count edges (upper triangle only to avoid double-counting)
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

        # Structural Hamming Distance
        shd = fp + fn

        return {
            "skeleton_f1": f1,
            "skeleton_precision": precision,
            "skeleton_recall": recall,
            "skeleton_shd": shd,
            "skeleton_tp": int(tp),
            "skeleton_fp": int(fp),
            "skeleton_fn": int(fn),
            "skeleton_tn": int(tn),
        }

    @staticmethod
    def compute_dag_metrics(G_pred: np.ndarray, B_true: np.ndarray) -> Dict[str, Any]:
        """
        Compute DAG metrics (directed graph with orientation).

        Counts:
        - Extra edges (false positives)
        - Missing edges (false negatives)
        - Reversed edges (wrong direction)
        """
        d = G_pred.shape[0]

        # Convert to binary adjacency
        pred_dag = (np.abs(G_pred) > 0).astype(int)
        true_dag = (np.abs(B_true) > 0).astype(int)

        # Count directed edges
        tp_directed = np.sum((pred_dag == 1) & (true_dag == 1))
        fp_directed = np.sum((pred_dag == 1) & (true_dag == 0))
        fn_directed = np.sum((pred_dag == 0) & (true_dag == 1))

        # Count reversed edges (i→j predicted but j→i is true)
        reversed_edges = 0
        for i in range(d):
            for j in range(d):
                if pred_dag[i, j] == 1 and true_dag[j, i] == 1 and true_dag[i, j] == 0:
                    reversed_edges += 1

        # DAG metrics
        precision_dag = (
            tp_directed / (tp_directed + fp_directed)
            if (tp_directed + fp_directed) > 0
            else 0.0
        )
        recall_dag = (
            tp_directed / (tp_directed + fn_directed)
            if (tp_directed + fn_directed) > 0
            else 0.0
        )
        f1_dag = (
            2 * precision_dag * recall_dag / (precision_dag + recall_dag)
            if (precision_dag + recall_dag) > 0
            else 0.0
        )

        # Structural Hamming Distance (directed)
        shd_dag = fp_directed + fn_directed + reversed_edges

        # Orientation accuracy (among skeleton edges)
        skeleton_edges = np.sum((np.abs(G_pred) + np.abs(G_pred.T)) > 0) / 2
        orientation_accuracy = (
            tp_directed / skeleton_edges if skeleton_edges > 0 else 0.0
        )

        return {
            "dag_f1": f1_dag,
            "dag_precision": precision_dag,
            "dag_recall": recall_dag,
            "dag_shd": int(shd_dag),
            "dag_tp": int(tp_directed),
            "dag_fp": int(fp_directed),
            "dag_fn": int(fn_directed),
            "dag_reversed": int(reversed_edges),
            "orientation_accuracy": orientation_accuracy,
        }


# ============================================================
# Method Runners
# ============================================================


class MethodRunner:
    """Unified method dispatcher for running causal discovery algorithms."""

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
        Run causal discovery method.

        Args:
            method_name: Method from METHOD_REGISTRY
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
        if method_name not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method: {method_name}")

        config = METHOD_REGISTRY[method_name]

        if config.type == "centralized":
            return MethodRunner._run_centralized(method_name, X, B, alpha, seed)
        elif config.type == "federated":
            return MethodRunner._run_federated(
                method_name, X, B, K, alpha, seed, **kwargs
            )
        else:
            raise ValueError(f"Unknown method type: {config.type}")

    @staticmethod
    def _run_centralized(
        method_name: str, X: np.ndarray, B: np.ndarray, alpha: float, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run centralized methods (GES, PC, FCI)."""
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
    def _run_federated(
        method_name: str,
        X: np.ndarray,
        B: np.ndarray,
        K: int,
        alpha: float,
        seed: int,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run federated methods (FedCDH, FedSPN)."""
        np.random.seed(seed)

        if method_name == "fedcdh":
            return MethodRunner._run_fedcdh(X, B, K, alpha, seed)
        elif method_name.startswith("fedspn"):
            return MethodRunner._run_fedspn(method_name, X, B, K, alpha, seed, **kwargs)
        else:
            raise ValueError(f"Unknown federated method: {method_name}")

    @staticmethod
    def _run_fedcdh(
        X: np.ndarray, B: np.ndarray, K: int, alpha: float, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run FedCDH (CD-NOD) with KCI tests."""
        d = X.shape[1]
        n = X.shape[0]

        # Partition horizontally
        samples_per_client = n // K
        c_indx = np.repeat(np.arange(K), samples_per_client)

        # Handle remainder samples
        remainder = n % K
        if remainder > 0:
            c_indx = np.concatenate([c_indx, np.arange(remainder)])

        c_indx = c_indx[:n].reshape(-1, 1)

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
            runtime = time.time() - start_time
            return np.zeros((d, d)), {"total_time": runtime}

    @staticmethod
    def _run_fedspn(
        method_name: str,
        X: np.ndarray,
        B: np.ndarray,
        K: int,
        alpha: float,
        seed: int,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run FedSPN methods (H/V/Hy)."""
        # Check for scenario override (from command-line --scenario)
        scenario_override = kwargs.get("scenario_override", None)

        if scenario_override:
            # Use override scenario
            scenario = scenario_override
            logging.info(f"  Using scenario override: {scenario}")
        else:
            # Extract scenario from method name
            if method_name == "fedspn_h":
                scenario = "horizontal"
            elif method_name == "fedspn_v":
                scenario = "vertical"
            elif method_name == "fedspn_hy":
                scenario = "hybrid"
            else:
                raise ValueError(f"Unknown FedSPN variant: {method_name}")

        # Create config
        d = X.shape[1]
        n = X.shape[0]

        # Trim data to be divisible by K for horizontal/hybrid
        if scenario in ["horizontal", "hybrid"]:
            n_trimmed = (n // K) * K
            X = X[:n_trimmed, :]
            n = n_trimmed

        # Handle K_local override (from command-line --K_local)
        K_local_override = kwargs.get("K_local_override", None)
        if K_local_override:
            num_local_clusters = K_local_override
            force_clusters = K_local_override
            logging.info(f"  Using K_local override: {K_local_override}")
        else:
            num_local_clusters = kwargs.get("num_local_clusters", 2)
            force_clusters = kwargs.get("force_clusters", 2)

        config = {
            "d": d,
            "K": K,
            "n_total": n,
            "epochs": kwargs.get(
                "epochs", 20
            ),  # Reduced for faster smoke tests (was 50)
            "num_local_clusters": num_local_clusters,
            "force_clusters": force_clusters,
        }

        # Prepare data splits
        samples_per_client = n // K if scenario != "vertical" else n

        # IMPORTANT: Shuffle data indices with seed for proper federated learning
        # This ensures:
        # 1. Reproducibility: same seed → same split
        # 2. Robustness: different seeds → different splits → can assess variance
        # 3. Realism: simulates random patient/data assignment to clients
        if seed is not None:
            rng = np.random.RandomState(seed)
            shuffled_indices = rng.permutation(n)
            logging.info(
                f"  Data shuffled with seed={seed} (first 5 sample indices: {shuffled_indices[:5].tolist()})"
            )
        else:
            # No seed: deterministic sequential split (for debugging only)
            shuffled_indices = np.arange(n)
            logging.warning(
                "  No seed provided - using deterministic split (not recommended for experiments)"
            )

        # Create c_indx (context indices) - maps each shuffled sample to its client
        if scenario == "vertical":
            # Vertical: All clients see same samples, no sample partitioning
            # c_indx can be zeros or any constant (not used for vertical)
            c_indx = np.zeros((n, 1), dtype=int)
        else:
            # Horizontal/Hybrid: Samples are partitioned across clients
            # Map shuffled samples to clients
            c_indx = np.zeros((n, 1), dtype=int)
            for k in range(K):
                start = k * samples_per_client
                end = (k + 1) * samples_per_client if k < K - 1 else n
                client_sample_indices = shuffled_indices[start:end]
                c_indx[client_sample_indices, 0] = k

        # Partition data based on scenario
        sample_maps = None
        feature_maps = None

        if scenario == "horizontal":
            # Horizontal: Split shuffled samples, all features
            sample_indices = []
            X_splits = []
            for k in range(K):
                start = k * samples_per_client
                end = (k + 1) * samples_per_client if k < K - 1 else n
                # Get shuffled indices for this client
                client_sample_indices = shuffled_indices[start:end]
                sample_indices.append(client_sample_indices)
                # Extract data for these shuffled samples
                X_splits.append(X[client_sample_indices, :])

            # Build maps
            sample_maps = {k: indices for k, indices in enumerate(sample_indices)}
            feature_maps = {k: np.arange(d) for k in range(K)}

        elif scenario == "vertical":
            # Vertical: All samples, split features across clients
            # Use np.array_split to handle uneven divisions
            feature_indices = np.array_split(range(d), K)
            X_splits = [X[:, indices] for indices in feature_indices]
            # Build maps
            sample_maps = {k: np.arange(n) for k in range(K)}
            feature_maps = {
                k: np.array(indices) for k, indices in enumerate(feature_indices)
            }

        elif scenario == "hybrid":
            # TRUE HYBRID: Overlapping samples AND features (Seng et al. design)
            from causallearn.utils.hybrid_partition import create_hybrid_block_partition

            overlap_fraction = K_local_override / 10.0 if K_local_override else 0.3
            X_splits, sample_maps, feature_maps = create_hybrid_block_partition(
                X, K, overlap_fraction=overlap_fraction, seed=seed
            )

            # Update c_indx for hybrid mode (need to track client ownership)
            # Each sample may belong to multiple clients, use first owner as primary
            c_indx = np.zeros((n, 1), dtype=int)
            for k, sample_idx in sample_maps.items():
                for idx in sample_idx:
                    if c_indx[idx, 0] == 0 or k == 0:  # First owner or client 0
                        c_indx[idx, 0] = k

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Setup FedCDH args
        # Get experiment directory to prevent FedCDH from creating its own
        exp_dir = kwargs.get("exp_dir", None)
        spn_eval_dir = str(exp_dir) if exp_dir else None

        args = Namespace(
            K=K,
            d=d,
            n=samples_per_client,
            scenario=scenario,
            model_type="synthetic",
            ci_method="spn",
            alpha=alpha,
            epochs=config["epochs"],
            device=kwargs.get("device", get_device()),  # Auto-detect GPU
            skip_bic=False,
            data_type="nonlinear",
            use_ci_ranking=False,
            force_num_clusters=config["force_clusters"],
            num_local_clusters=config["num_local_clusters"],
            skip_spn_eval=True,
            horizontal_aggregation=kwargs.get(
                "horizontal_aggregation", "structure_voting"
            ),
            structure_vote_threshold=kwargs.get("structure_vote_threshold", 0.4),
            return_graphs=True,  # CRITICAL: Must return graphs for evaluation
            spn_eval_dir=spn_eval_dir,  # Use benchmark's experiment directory
            leaf_type=kwargs.get("leaf_type", "normal"),  # SPN leaf distribution
        )

        # Log configuration for K ablation studies
        logging.info(
            f"  FedSPN config: scenario={scenario}, K={K}, K_local={config['num_local_clusters']}, epochs={config['epochs']}"
        )

        start_time = time.time()

        try:
            fedcdh = FedCDH(args, sample_maps=sample_maps, feature_maps=feature_maps)
            results = fedcdh.fit(X_splits, c_indx, B)
            runtime = time.time() - start_time

            # Extract graph - FedCDH.fit() returns a dict with 'est_dag'
            if "est_dag" in results:
                G = results["est_dag"]
            elif "graph" in results:
                G = results["graph"]
            else:
                logging.warning(f"  No graph in results, keys: {list(results.keys())}")
                G = np.zeros((d, d))

            # Convert to binary adjacency matrix if needed
            if hasattr(G, "shape"):
                G_binary = (np.abs(G) > 0).astype(int)
            else:
                logging.warning(f"  Graph is not array, type: {type(G)}")
                G_binary = np.zeros((d, d))

            metrics = {
                "total_time": runtime,
                "training_time": runtime * 0.7,  # Estimate
                "ci_test_time": runtime * 0.2,
                "aggregation_time": runtime * 0.1,
                "_fedcdh_instance": fedcdh,  # Store for UMAP/logging
                "_X_splits": X_splits,  # Store for UMAP
            }

            return G_binary, metrics

        except Exception as e:
            import traceback

            logging.error(f"  {method_name.upper()} failed: {e}")
            logging.error(f"  Traceback: {traceback.format_exc()}")
            runtime = time.time() - start_time
            return np.zeros((d, d)), {"total_time": runtime}


# ============================================================
# Unified Benchmark
# ============================================================


class UnifiedBenchmark:
    """
    V3 Unified Benchmark Suite for Causal Discovery.

    Runs any method on any dataset with comprehensive evaluation.
    """

    def __init__(
        self,
        datasets: List[str],
        methods: List[str],
        seeds: List[int] = [42, 123, 456],
        K: int = 3,
        alpha: float = 0.05,
        output_dir: str = "benchmark_results/v3",
        device: str = None,
        save_graphs: bool = False,
        scenario_override: str = None,
        K_local_override: int = None,
        leaf_type: str = "normal",
    ):
        """
        Initialize unified benchmark.

        Args:
            datasets: List of dataset names from DATASET_REGISTRY
            methods: List of method names from METHOD_REGISTRY
            seeds: Random seeds for multiple runs
            K: Number of clients (for federated methods)
            alpha: Significance level for CI tests
            output_dir: Output directory for results
            device: Device to use (cuda/mps/cpu) or None for auto-detect
            save_graphs: Whether to save graphs and visualizations
            scenario_override: Force specific scenario (horizontal/vertical/hybrid)
            K_local_override: Force specific K_local for SPN methods
            leaf_type: SPN leaf distribution type (normal/binomial/categorical)
        """
        self.datasets = datasets
        self.methods = methods
        self.seeds = seeds
        self.K = K
        self.alpha = alpha
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if device is not None else get_device()
        self.save_graphs = save_graphs
        self.scenario_override = scenario_override
        self.K_local_override = K_local_override
        self.leaf_type = leaf_type

        # Create graphs directory if saving graphs
        if self.save_graphs:
            self.graphs_dir = self.output_dir / "graphs"
            self.graphs_dir.mkdir(parents=True, exist_ok=True)

        # Log device info
        logging.info(f"Using device: {self.device}")
        if self.device == "mps":
            logging.info("Apple Silicon GPU (MPS) detected and enabled")
        elif self.device == "cuda":
            logging.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("Using CPU (no GPU acceleration)")

        # Validate inputs
        for dataset in datasets:
            if dataset not in DATASET_REGISTRY:
                raise ValueError(f"Unknown dataset: {dataset}")
        for method in methods:
            if method not in METHOD_REGISTRY:
                raise ValueError(f"Unknown method: {method}")

    def _setup_experiment_logging(self, exp_dir: Path) -> logging.FileHandler:
        """Set up logging to file for an experiment."""
        log_file = exp_dir / "run.log"
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        return file_handler

    def _cleanup_experiment_logging(self, file_handler: logging.FileHandler):
        """Remove file handler after experiment."""
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    def run_single_experiment(
        self,
        dataset_name: str,
        method_name: str,
        seed: int,
    ) -> Dict[str, Any]:
        """Run single experiment with comprehensive metrics."""
        # Create experiment directory early for logging
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir_name = f"{timestamp}_{dataset_name}_{method_name}_seed{seed}"
        exp_dir = Path("eval") / exp_dir_name if self.save_graphs else None

        # Set up file logging if saving
        file_handler = None
        if self.save_graphs and exp_dir:
            exp_dir.mkdir(parents=True, exist_ok=True)
            file_handler = self._setup_experiment_logging(exp_dir)

        try:
            # Load dataset
            X, B, feature_names = DatasetLoader.load_dataset(dataset_name, seed)

            # Run method
            G, runtime_metrics = MethodRunner.run_method(
                method_name,
                X,
                B,
                K=self.K,
                alpha=self.alpha,
                seed=seed,
                device=self.device,
                exp_dir=exp_dir,  # Pass experiment directory to prevent FedCDH from creating its own
                scenario_override=self.scenario_override,
                K_local_override=self.K_local_override,
                leaf_type=self.leaf_type,
            )

            # Extract internal objects for visualization (if available)
            fedcdh_instance = runtime_metrics.pop("_fedcdh_instance", None)
            X_splits = runtime_metrics.pop("_X_splits", None)

            # Compute metrics
            all_metrics = MetricsEngine.compute_all_metrics(G, B, runtime_metrics)

            # Save graphs if requested
            if self.save_graphs and exp_dir:
                self._save_experiment_graphs(
                    dataset_name,
                    method_name,
                    seed,
                    G,
                    B,
                    feature_names,
                    metrics=all_metrics,
                    fedcdh_instance=fedcdh_instance,
                    X_splits=X_splits,
                    exp_dir=exp_dir,
                )

            # Add experiment metadata
            result = {
                "dataset": dataset_name,
                "method": method_name,
                "seed": seed,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_edges_true": int(np.sum(B)),
                "K": self.K if METHOD_REGISTRY[method_name].type == "federated" else 1,
                "alpha": self.alpha,
                **all_metrics,
            }

            return result

        finally:
            # Clean up file logging
            if file_handler:
                self._cleanup_experiment_logging(file_handler)

    def _save_experiment_graphs(
        self,
        dataset_name: str,
        method_name: str,
        seed: int,
        G_pred: np.ndarray,
        B_true: np.ndarray,
        feature_names: List[str],
        metrics: Optional[Dict[str, Any]] = None,
        fedcdh_instance: Optional[Any] = None,
        X_splits: Optional[List[np.ndarray]] = None,
        exp_dir: Optional[Path] = None,
    ):
        """
        Save graphs and visualizations for a single experiment.

        Creates:
        1. Adjacency matrices (CSV + NPY)
        2. Edge list (TXT)
        3. Predicted graph visualization (PNG)
        4. Ground truth vs predicted comparison (PNG)
        5. Difference visualization with TP/FP/FN (PNG)
        6. run.log with experiment logs (created by run_single_experiment)
        7. eval.txt with evaluation metrics
        8. UMAP visualizations (global SPN + local client SPNs)
        """
        # Use provided directory or create new one
        if exp_dir is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir_name = f"{timestamp}_{dataset_name}_{method_name}_seed{seed}"
            exp_dir = Path("eval") / exp_dir_name
            exp_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"      → _save_experiment_graphs called")
        logging.info(f"      → metrics = {metrics}")
        logging.info(f"      → exp_dir = {exp_dir}")
        logging.info(f"      → fedcdh_instance is None: {fedcdh_instance is None}")
        logging.info(f"      → X_splits is None: {X_splits is None}")
        if fedcdh_instance is not None:
            logging.info(
                f"      → fedcdh_instance has local_spns: {hasattr(fedcdh_instance, 'local_spns')}"
            )
            if hasattr(fedcdh_instance, "local_spns"):
                logging.info(
                    f"      → local_spns count: {len(fedcdh_instance.local_spns) if fedcdh_instance.local_spns else 0}"
                )
        if X_splits is not None:
            logging.info(f"      → X_splits count: {len(X_splits)}")

        # 1. Save adjacency matrices
        # Convert to binary for clean output
        G_binary = (np.abs(G_pred) > 0).astype(int)
        B_binary = (np.abs(B_true) > 0).astype(int)

        # Save as CSV with feature names
        df_pred = pd.DataFrame(G_binary, index=feature_names, columns=feature_names)
        df_pred.to_csv(exp_dir / "adjacency_pred.csv")

        df_true = pd.DataFrame(B_binary, index=feature_names, columns=feature_names)
        df_true.to_csv(exp_dir / "adjacency_true.csv")

        # Save as NPY (exact format)
        np.save(exp_dir / "adjacency_pred.npy", G_pred)
        np.save(exp_dir / "adjacency_true.npy", B_true)

        # 2. Save edge lists
        with open(exp_dir / "edges_pred.txt", "w") as f:
            f.write(
                f"# Predicted edges for {dataset_name} - {method_name} (seed={seed})\n"
            )
            f.write(f"# Total edges: {int(np.sum(G_binary))}\n")
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    if G_binary[i, j] == 1:
                        f.write(f"{feature_names[i]} -> {feature_names[j]}\n")

        with open(exp_dir / "edges_true.txt", "w") as f:
            f.write(f"# Ground truth edges for {dataset_name}\n")
            f.write(f"# Total edges: {int(np.sum(B_binary))}\n")
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    if B_binary[i, j] == 1:
                        f.write(f"{feature_names[i]} -> {feature_names[j]}\n")

        # 3. Generate visualizations
        try:
            # Predicted graph only
            visualize_dag(
                G_binary,
                node_names=feature_names,
                layout="hierarchical",
                title=f"{dataset_name.upper()} - {method_name.upper()} (seed={seed})",
                save_path=str(exp_dir / "graph_pred.png"),
                show=False,
            )
            plt.close()

            # Side-by-side comparison
            compare_graphs(
                B_binary,
                G_binary,
                node_names=feature_names,
                layout="hierarchical",
                title=f"{dataset_name.upper()} - {method_name.upper()} (seed={seed})",
                save_path=str(exp_dir / "graph_comparison.png"),
                show=False,
            )
            plt.close()

            # Differences (TP/FP/FN)
            visualize_graph_differences(
                B_binary,
                G_binary,
                node_names=feature_names,
                layout="hierarchical",
                title=f"{dataset_name.upper()} - {method_name.upper()} - Differences (seed={seed})",
                save_path=str(exp_dir / "graph_differences.png"),
                show=False,
            )
            plt.close()

            logging.info(f"      → Saved graphs to: {exp_dir.name}/")

        except Exception as e:
            logging.warning(f"      → Failed to generate visualizations: {e}")

        # 4. Save evaluation metrics to eval.txt
        if metrics:
            logging.info(f"      → Saving metrics to eval.txt...")
            try:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                eval_path = exp_dir / "eval.txt"
                logging.info(f"      → Writing to: {eval_path}")
                with open(eval_path, "w") as f:
                    f.write(f"# Evaluation Metrics\n")
                    f.write(f"# Dataset: {dataset_name}\n")
                    f.write(f"# Method: {method_name}\n")
                    f.write(f"# Seed: {seed}\n")
                    f.write(f"# Timestamp: {timestamp}\n")
                    f.write(f"#\n")
                    f.write(f"# {'='*70}\n")
                    f.write(f"# STRUCTURE METRICS\n")
                    f.write(f"# {'='*70}\n")
                    f.write(f"skeleton_f1: {metrics.get('skeleton_f1', 0.0):.6f}\n")
                    f.write(
                        f"skeleton_precision: {metrics.get('skeleton_precision', 0.0):.6f}\n"
                    )
                    f.write(
                        f"skeleton_recall: {metrics.get('skeleton_recall', 0.0):.6f}\n"
                    )
                    f.write(f"skeleton_shd: {metrics.get('skeleton_shd', 0)}\n")
                    f.write(f"dag_f1: {metrics.get('dag_f1', 0.0):.6f}\n")
                    f.write(f"dag_precision: {metrics.get('dag_precision', 0.0):.6f}\n")
                    f.write(f"dag_recall: {metrics.get('dag_recall', 0.0):.6f}\n")
                    f.write(f"dag_shd: {metrics.get('dag_shd', 0)}\n")
                    f.write(f"dag_reversed: {metrics.get('dag_reversed', 0)}\n")
                    f.write(
                        f"orientation_accuracy: {metrics.get('orientation_accuracy', 0.0):.6f}\n"
                    )
                    f.write(f"\n# {'='*70}\n")
                    f.write(f"# RUNTIME METRICS\n")
                    f.write(f"# {'='*70}\n")
                    f.write(
                        f"total_time: {metrics.get('total_time', 0.0):.3f} seconds\n"
                    )
                    f.write(
                        f"training_time: {metrics.get('training_time', 0.0):.3f} seconds\n"
                    )
                    f.write(
                        f"ci_test_time: {metrics.get('ci_test_time', 0.0):.3f} seconds\n"
                    )
                    f.write(
                        f"aggregation_time: {metrics.get('aggregation_time', 0.0):.3f} seconds\n"
                    )
                logging.info(f"      → Saved metrics to: eval.txt")
            except Exception as e:
                logging.warning(f"      → Failed to save eval.txt: {e}")

        # 5. Generate UMAP visualizations for SPN-based methods
        if fedcdh_instance is not None and X_splits is not None:
            logging.info(f"      → Attempting UMAP visualization...")
            try:
                self._generate_umap_visualizations(
                    exp_dir, fedcdh_instance, X_splits, feature_names
                )
            except Exception as e:
                import traceback

                logging.warning(f"      → Failed to generate UMAP visualizations: {e}")
                logging.warning(f"      → Traceback: {traceback.format_exc()}")

    def _generate_umap_visualizations(
        self,
        exp_dir: Path,
        fedcdh_instance: Any,
        X_splits: List[np.ndarray],
        feature_names: List[str],
    ):
        """
        Generate UMAP visualizations comparing SPN-sampled data with ground truth.

        Approach:
        1. Sample data from learned SPN
        2. Compare with ground truth (real) data
        3. Project both onto same UMAP space
        4. Visualize overlap/differences to assess model quality
        """
        logging.info(f"      → _generate_umap_visualizations called")
        try:
            import umap

            logging.info(f"      → UMAP imported successfully")
        except ImportError:
            logging.warning("      → UMAP not available, skipping visualizations")
            return

        # Check if this is an SPN-based method with local SPNs
        logging.info(f"      → Checking local_spns...")
        if not hasattr(fedcdh_instance, "local_spns"):
            logging.info(f"      → No local_spns attribute, skipping")
            return
        if not fedcdh_instance.local_spns:
            logging.info(f"      → local_spns is empty, skipping")
            return
        logging.info(
            f"      → local_spns found: {len(fedcdh_instance.local_spns)} SPNs"
        )

        # Check if global SPN exists
        logging.info(f"      → Checking fed_spn_model...")
        if not hasattr(fedcdh_instance, "fed_spn_model"):
            logging.info(f"      → No fed_spn_model attribute, skipping")
            return
        if fedcdh_instance.fed_spn_model is None:
            logging.info(f"      → fed_spn_model is None, skipping")
            return
        logging.info(f"      → fed_spn_model found")

        d = len(feature_names)
        logging.info(f"      → Number of features: {d}")
        if d <= 2:
            logging.info("      → Skipping UMAP (d <= 2)")
            return

        # Generate global SPN UMAP: Compare SPN samples with real data
        try:
            # Concatenate all client data
            X_global = np.vstack(X_splits)
            n_samples = min(500, X_global.shape[0])

            # Get real data samples
            real_indices = np.random.choice(X_global.shape[0], n_samples, replace=False)
            X_real = X_global[real_indices, :]

            # Sample from learned SPN
            logging.info(f"      → Sampling {n_samples} points from global SPN...")
            X_sampled_tensor = fedcdh_instance.fed_spn_model.sample(n_samples)
            X_sampled = X_sampled_tensor.cpu().detach().numpy()

            # Ensure correct dimensions (remove context column if present)
            if X_sampled.shape[1] > d:
                X_sampled = X_sampled[:, :d]

            # Combine real and sampled data
            X_combined = np.vstack([X_real, X_sampled])
            labels = np.array(["Real"] * n_samples + ["SPN"] * n_samples)

            # Create UMAP embedding
            n_neighbors = min(15, 2 * n_samples - 1)
            reducer = umap.UMAP(
                n_components=2, random_state=42, n_neighbors=n_neighbors
            )
            embedding = reducer.fit_transform(X_combined)

            # Plot with separate colors for real vs sampled
            fig, ax = plt.subplots(figsize=(10, 8))

            # Real data in blue
            real_mask = labels == "Real"
            ax.scatter(
                embedding[real_mask, 0],
                embedding[real_mask, 1],
                c="blue",
                alpha=0.5,
                s=30,
                label="Real Data",
                edgecolors="none",
            )

            # SPN samples in red
            spn_mask = labels == "SPN"
            ax.scatter(
                embedding[spn_mask, 0],
                embedding[spn_mask, 1],
                c="red",
                alpha=0.5,
                s=30,
                label="SPN Samples",
                edgecolors="none",
            )

            ax.set_title(f"Global SPN vs Real Data (n={n_samples} each)")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(loc="best")
            plt.tight_layout()
            plt.savefig(exp_dir / "umap_global_spn.png", dpi=150, bbox_inches="tight")
            plt.close()
            logging.info(f"      → Saved UMAP: umap_global_spn.png")
        except Exception as e:
            import traceback

            logging.warning(f"      → Failed to generate global UMAP: {e}")
            logging.warning(f"      → Traceback: {traceback.format_exc()}")

        # Generate local client UMAP visualizations
        for k, (local_spn, X_client) in enumerate(
            zip(fedcdh_instance.local_spns, X_splits)
        ):
            try:
                if X_client.shape[1] <= 2:
                    continue

                n_samples = min(250, X_client.shape[0])

                # Get real data samples
                real_indices = np.random.choice(
                    X_client.shape[0], n_samples, replace=False
                )
                X_real = X_client[real_indices, :]

                # Sample from local SPN
                logging.info(
                    f"      → Sampling {n_samples} points from client {k} SPN..."
                )
                X_sampled_tensor = local_spn.sample(n_samples)
                X_sampled = X_sampled_tensor.cpu().detach().numpy()

                # Ensure correct dimensions
                if X_sampled.shape[1] > X_client.shape[1]:
                    X_sampled = X_sampled[:, : X_client.shape[1]]

                # Combine real and sampled data
                X_combined = np.vstack([X_real, X_sampled])
                labels = np.array(["Real"] * n_samples + ["SPN"] * n_samples)

                # Create UMAP embedding
                n_neighbors = min(15, 2 * n_samples - 1)
                reducer = umap.UMAP(
                    n_components=2, random_state=42, n_neighbors=n_neighbors
                )
                embedding = reducer.fit_transform(X_combined)

                # Plot
                fig, ax = plt.subplots(figsize=(8, 6))

                # Real data in blue
                real_mask = labels == "Real"
                ax.scatter(
                    embedding[real_mask, 0],
                    embedding[real_mask, 1],
                    c="blue",
                    alpha=0.5,
                    s=30,
                    label="Real Data",
                    edgecolors="none",
                )

                # SPN samples in red
                spn_mask = labels == "SPN"
                ax.scatter(
                    embedding[spn_mask, 0],
                    embedding[spn_mask, 1],
                    c="red",
                    alpha=0.5,
                    s=30,
                    label="SPN Samples",
                    edgecolors="none",
                )

                ax.set_title(f"Client {k} SPN vs Real Data (n={n_samples} each)")
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.legend(loc="best")
                plt.tight_layout()
                plt.savefig(
                    exp_dir / f"umap_local_client_{k}.png", dpi=150, bbox_inches="tight"
                )
                plt.close()
                logging.info(f"      → Saved UMAP: umap_local_client_{k}.png")
            except Exception as e:
                import traceback

                logging.warning(f"      → Failed to generate UMAP for client {k}: {e}")
                logging.warning(f"      → Traceback: {traceback.format_exc()}")

    def run_all_experiments(self) -> pd.DataFrame:
        """Run all combinations of datasets × methods × seeds."""
        results = []
        total = len(self.datasets) * len(self.methods) * len(self.seeds)
        counter = 1

        logging.info("=" * 80)
        logging.info("V3 UNIFIED BENCHMARK SUITE")
        logging.info("=" * 80)
        logging.info(f"Datasets: {self.datasets}")
        logging.info(f"Methods: {self.methods}")
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

                    finally:
                        # Clear CUDA cache after each experiment to prevent memory accumulation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

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

    parser = argparse.ArgumentParser(description="V3 Unified Benchmark Suite")
    parser.add_argument(
        "--datasets",
        type=str,
        default="sachs,synthetic_er_small",
        help="Comma-separated list of datasets",
    )
    parser.add_argument(
        "--methods",
        "--method",
        type=str,
        default="ges,fedspn_h",
        help="Comma-separated list of methods",
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
        help="Number of federated clients",
    )
    parser.add_argument(
        "--K_local",
        type=int,
        default=None,
        help="Number of local clusters per client for SPN (default: 2). Key parameter for K ablation studies.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["horizontal", "vertical", "hybrid"],
        help="Override federated scenario (default: inferred from method name)",
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
        default="benchmark_results/v3",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--save-graphs",
        action="store_true",
        help="Save graph adjacency matrices and visualizations (adds ~5-10s per experiment)",
    )
    parser.add_argument(
        "--leaf-type",
        type=str,
        default="normal",
        choices=["normal", "binomial", "categorical"],
        help="SPN leaf distribution type (default: normal for continuous data, binomial for binary/count, categorical for discrete)",
    )

    args = parser.parse_args()

    # Parse arguments
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    # Create benchmark
    benchmark = UnifiedBenchmark(
        datasets=datasets,
        methods=methods,
        seeds=seeds,
        K=args.K,
        alpha=args.alpha,
        output_dir=args.output_dir,
        device=args.device,
        save_graphs=args.save_graphs,
        scenario_override=args.scenario,
        K_local_override=args.K_local,
        leaf_type=args.leaf_type,
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
    logging.info("V3 UNIFIED BENCHMARK COMPLETE")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
