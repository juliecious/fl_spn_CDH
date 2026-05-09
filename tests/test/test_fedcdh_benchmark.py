"""
FedCDH V3 Benchmark Suite - Structure-Preserving Aggregation + GlobalSumOfProducts.

Compares 3 SPN aggregation scenarios (Horizontal/Vertical/Hybrid) with
V3 critical fixes for dependency preservation.

V3 Features:
- ✅ Horizontal: 3 aggregation strategies (structure_voting, ll_weighted, mixture)
- ✅ Hybrid: GlobalSumOfProducts (sum-over-products breaks independence)
- ✅ Vertical: ProductOverGroups (unchanged from V2)
- ✅ Sachs protein network dataset integrated
- ✅ V2 adaptive hyperparameters maintained

Scenarios:
- Horizontal: Mixture-of-experts (sample partitioning across K clients)
  * V3 Strategies: structure_voting (recommended), ll_weighted, mixture (V2 baseline)
- Vertical: Product-of-experts (feature partitioning, all clients see all samples)
- Hybrid: Mixture-then-Product (overlapping features + sample partitioning)
  * V3 Fix: GlobalSumOfProducts breaks independence between feature groups

Sample Size Consistency:
- All modes use n_total samples, but distribution differs:
  * Horizontal: n_total samples split across K clients (n_per_client = n_total/K)
  * Vertical: All K clients see the same n_total samples (all samples shared)
  * Hybrid: n_total samples split across K clients (like horizontal)

Hardware: GPU-enabled (CUDA) recommended for faster training
Runtime: ~15-30 minutes per configuration (GPU), ~45-90 minutes (CPU)

Data:
- quick/small/medium/large configs: Synthetic data (linear or nonlinear)
- sachs config: Real Sachs protein signaling dataset (11 vars, 7466 samples, 17 edges)

Updated: May 9, 2026 - V3 implementation with structure-preserving aggregation
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

# Sample Size Strategy (v2):
# ---------------------------
# All modes use the same n_total samples, ensuring fair comparison:
#
#   Mode         | Total Samples | Per-Client Samples  | Per-Client Features
#   -------------|---------------|---------------------|--------------------
#   Horizontal   | n_total       | n_total/K          | d (all features)
#   Vertical     | n_total       | n_total (all)      | d/K (split features)
#   Hybrid       | n_total       | n_total/K          | d (overlapping)
#
# Example (MEDIUM: n_total=1200, K=3, d=10):
#   - Horizontal: 3 clients × 400 samples × 10 features
#   - Vertical:   3 clients × 1200 samples × ~3-4 features
#   - Hybrid:     3 clients × 400 samples × 10 features (with overlap)
#
# This ensures:
#   1. Same total information available (n_total samples)
#   2. Fair capacity comparison (SPNs trained on same data volume)
#   3. Mode-specific challenges properly reflected (horizontal: sample heterogeneity,
#      vertical: feature fragmentation, hybrid: both with overlap)

# Experiment configurations (V3 optimized)
# n_total: Total samples (consistent across modes)
# n_per_client: Samples per client (horizontal/hybrid: n_total/K, vertical: n_total)
#
# V3 Optimizations:
# - Increased epochs for better SPN training (+20-67%)
# - num_local_clusters=3 for hybrid mode (more cluster combinations)
# - force_clusters=2 to prevent BIC selecting K=1
# - Relaxed structure_vote_threshold and alpha for better recall
BENCHMARK_CONFIGS = {
    "quick": {
        "d": 5,
        "K": 2,
        "n_total": 200,
        "epochs": 20,
        "num_local_clusters": 2,  # Keep low for speed
        "force_clusters": None,  # Let BIC decide for quick tests
        "description": "Quick smoke test: 5 vars, 2 clients, 200 total samples",
    },
    "small": {
        "d": 8,
        "K": 3,
        "n_total": 900,
        "epochs": 60,  # +20% from 50 (V3 optimization)
        "num_local_clusters": 2,
        "force_clusters": 2,  # V3: Force clustering
        "description": "Small-scale: 8 vars, 3 clients, 900 total samples (300/client horizontal)",
    },
    "medium": {
        "d": 10,
        "K": 3,
        "n_total": 1200,
        "epochs": 120,  # +20% from 100 (V3 optimization)
        "num_local_clusters": 3,  # V3: More clusters for hybrid sum-over-products
        "force_clusters": 2,  # V3: Force clustering
        "description": "Medium-scale: 10 vars, 3 clients, 1200 total samples (400/client horizontal) [V3 optimized]",
    },
    "large": {
        "d": 11,
        "K": 5,
        "n_total": 2500,  # Increased from 2000 (500/client)
        "epochs": 200,  # +33% from 150 (V3 optimization)
        "num_local_clusters": 3,  # V3: More clusters for K=5
        "force_clusters": 2,  # V3: Force clustering
        "description": "Large-scale: 11 vars, 5 clients, 2500 total samples (500/client horizontal) [V3 optimized]",
    },
    "sachs": {
        "d": 11,
        "K": 3,
        "n_total": 7466,  # Full Sachs dataset (will be loaded from file)
        "epochs": 250,  # +67% from 150 (V3 optimization for nonlinear data)
        "num_local_clusters": 3,  # V3: More clusters for real data
        "force_clusters": 2,  # V3: Force clustering
        "description": "Real Sachs protein signaling dataset: 11 vars, 3 clients, 7466 samples (ground truth: 17 edges) [V3 optimized]",
    },
    "law_school": {
        "d": 5,
        "K": 3,
        "n_total": 21000,  # LSAC Bar Passage Study dataset
        "epochs": 150,
        "num_local_clusters": 3,  # V3: More clusters for social science data
        "force_clusters": 2,  # V3: Force clustering
        "description": "Law School Admissions dataset: 5 vars (race, LSAT, UGPA, region, FYA), 21K samples (ground truth: 7 edges) [Fairness benchmark]",
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
    """
    Partition data according to federated scenario.

    Key insight: All modes use the same n_total samples, but distribution differs:
    - Horizontal: n_total split across K clients (each gets n_total/K samples, all features)
    - Vertical: All K clients see same n_total samples (each gets all samples, subset of features)
    - Hybrid: n_total split across K clients (each gets n_total/K samples, overlapping features)

    Args:
        X: Global data (n_total, d)
        c_indx: Context indices
        K: Number of clients
        scenario: 'horizontal', 'vertical', or 'hybrid'

    Returns:
        X_splits: List of K data partitions
    """
    n, d = X.shape

    if scenario == "horizontal":
        # Horizontal: Split samples, all clients see all features
        samples_per_client = n // K
        X_splits = [
            X[k * samples_per_client : (k + 1) * samples_per_client, :]
            for k in range(K)
        ]
        logging.info(
            f"  Horizontal partition: {K} clients, each with {samples_per_client} samples × {d} features"
        )

    elif scenario == "vertical":
        # Vertical: All clients see all samples, each gets subset of features
        features_per_client = d // K
        X_splits = []
        for k in range(K):
            start_feat = k * features_per_client
            end_feat = (k + 1) * features_per_client if k < K - 1 else d
            X_splits.append(X[:, start_feat:end_feat])

        logging.info(
            f"  Vertical partition: {K} clients, each with {n} samples × ~{features_per_client} features"
        )

    elif scenario == "hybrid":
        # Hybrid: Split samples (like horizontal), overlapping features created by FedCDH
        samples_per_client = n // K
        X_splits = [
            X[k * samples_per_client : (k + 1) * samples_per_client, :]
            for k in range(K)
        ]
        logging.info(
            f"  Hybrid partition: {K} clients, each with {samples_per_client} samples × {d} features (overlap added by FedCDH)"
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return X_splits


# ============================================================
# Benchmark Execution
# ============================================================


def run_single_experiment(
    config_name,
    config,
    scenario,
    data_type,
    seed,
    device="cuda",
    use_ci_ranking=False,
    sparsity_percentile=0.2,
    force_clusters=None,
    num_local_clusters=None,
    skip_eval=False,
    horizontal_aggregation="structure_voting",
    structure_vote_threshold=0.4,
    alpha=0.08,
):
    """
    Run a single benchmark experiment (V3 with structure-preserving aggregation).

    Args:
        config_name: Configuration name
        config: Configuration dictionary
        scenario: 'horizontal', 'vertical', or 'hybrid'
        data_type: 'linear' or 'nonlinear'
        seed: Random seed
        device: Device to use
        use_ci_ranking: Enable CI ranking (experimental)
        sparsity_percentile: Sparsity for ranking (if enabled)
        force_clusters: Force specific number of clusters (bypasses BIC), or None to use config default
        num_local_clusters: Number of local clusters per client, or None to use config default
        skip_eval: Skip expensive SPN evaluation
        horizontal_aggregation: 'structure_voting', 'll_weighted', or 'mixture' (V3)
        structure_vote_threshold: Voting threshold for structure_voting (V3, default: 0.4)
        alpha: CI test significance level (V3 optimized, default: 0.08)

    Returns:
        Dictionary with results
    """
    d = config["d"]
    K = config["K"]
    n_total = config["n_total"]
    epochs = config["epochs"]

    # V3: Use config defaults if not explicitly overridden
    if num_local_clusters is None:
        num_local_clusters = config.get("num_local_clusters", 2)
    if force_clusters is None:
        force_clusters = config.get("force_clusters", None)

    logging.info(
        f"Running: {config_name} | {scenario} | {data_type} | seed={seed} | "
        f"device={device} | V3_optimized=True"
    )
    if use_ci_ranking:
        logging.info(f"  CI Ranking: enabled (sparsity={sparsity_percentile})")

    # Log GPU memory before experiment
    log_gpu_memory(prefix="[Pre-experiment] ", device=device)

    start_time = time.time()

    # Load data: Use real datasets if config is "sachs" or "law_school", otherwise generate synthetic
    if config_name == "sachs":
        logging.info("Loading real Sachs dataset...")
        import gzip
        import pandas as pd

        # Load Sachs protein network data
        data_path = "tests/data/sachs.interventional.txt.gz"
        with gzip.open(data_path, "rt") as f:
            df = pd.read_csv(f, sep="\t")

        X = df.values
        actual_n = X.shape[0]
        actual_d = X.shape[1]

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

        logging.info(
            f"  Sachs data: {actual_n} samples, {actual_d} features, {int(np.sum(B))} edges"
        )

        # Partition based on scenario
        c_indx = np.repeat(np.arange(K), actual_n // K).reshape(-1, 1)
        X_splits = partition_data(X, c_indx, K, scenario)

        W = B
        choice = None

        # Override n_total to actual data size
        n_total = actual_n

    elif config_name == "law_school":
        logging.info("Loading Law School Admissions dataset...")
        from tests.utils.law_school_loader import load_law_school_federated

        # Load Law School data (synthetic based on known causal structure)
        X, B, feature_names = load_law_school_federated(
            n_clients=K, n_samples_limit=n_total
        )

        actual_n = X.shape[0]
        actual_d = X.shape[1]

        logging.info(
            f"  Law School data: {actual_n} samples, {actual_d} features ({feature_names}), {int(np.sum(B))} edges"
        )

        # Partition based on scenario
        c_indx = np.repeat(np.arange(K), actual_n // K).reshape(-1, 1)
        X_splits = partition_data(X, c_indx, K, scenario)

        W = B
        choice = None

        # Override n_total to actual data size
        n_total = actual_n

    else:
        # Generate synthetic data
        W, B, X, c_indx, choice = create_benchmark_data(
            d=d, K=K, n=n_total, seed=seed, data_type=data_type, sem_type="gauss"
        )

        # Partition data
        X_splits = partition_data(X, c_indx, K, scenario)

    # Calculate n_per_client based on scenario
    if scenario == "vertical":
        # Vertical: All clients see all samples
        n_per_client = n_total
    else:
        # Horizontal/Hybrid: Samples split across clients
        n_per_client = n_total // K

    # Setup FedCDH with V3 parameters
    model_type = "real" if config_name in ["sachs", "law_school"] else "synthetic"
    args = Namespace(
        K=K,
        d=d,
        n=n_per_client,
        scenario=scenario,
        model_type=model_type,
        ci_method="spn",
        alpha=alpha,  # V3: Relaxed from 0.05 to 0.08 for better recall
        epochs=epochs,
        device=device,
        skip_bic=False,  # Use BIC for optimal cluster selection
        # V2 features
        data_type=data_type
        if config_name not in ["sachs", "law_school"]
        else "nonlinear",  # Real datasets are nonlinear
        use_ci_ranking=use_ci_ranking,
        sparsity_percentile=sparsity_percentile,
        force_num_clusters=force_clusters,  # V3: Use config default (typically 2)
        num_local_clusters=num_local_clusters,  # V3: Use config default (typically 3)
        skip_spn_eval=skip_eval,
        # V3 features
        horizontal_aggregation=horizontal_aggregation,  # V3: structure_voting, ll_weighted, or mixture
        structure_vote_threshold=structure_vote_threshold,  # V3: Relaxed to 0.4 for better recall
    )

    logging.info(f"  FedCDH args: d={d}, K={K}, n_per_client={n_per_client}")
    logging.info(
        f"  V3 features: data_type={args.data_type}, num_local_clusters={num_local_clusters}, "
        f"alpha={alpha}, structure_vote_threshold={structure_vote_threshold}"
    )
    if scenario == "horizontal":
        logging.info(f"  V3 horizontal aggregation: {horizontal_aggregation}")
    if force_clusters is not None:
        logging.info(f"  Force K={force_clusters} clusters (bypassing BIC)")

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

    # Extract metrics (V3 with aggregation strategy tracking)
    result_dict = {
        "config": config_name,
        "scenario": scenario,
        "data_type": args.data_type,  # Use actual data_type (may be overridden for Sachs)
        "seed": seed,
        "d": d,
        "K": K,
        "n_total": n_total,
        "n_per_client": n_per_client,
        "epochs": epochs,
        "device": device,
        # V2 tracking
        "use_ci_ranking": use_ci_ranking,
        "sparsity_percentile": sparsity_percentile if use_ci_ranking else None,
        "v2_adaptive": True,
        # V3 tracking
        "v3_enabled": True,
        "horizontal_aggregation": horizontal_aggregation
        if scenario == "horizontal"
        else None,
        # Performance metrics
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

    # Add V3-specific metrics for horizontal mode
    if scenario == "horizontal" and horizontal_aggregation == "structure_voting":
        if hasattr(fedcdh, "edge_confidence") and fedcdh.edge_confidence:
            result_dict["avg_edge_confidence"] = float(
                np.mean(list(fedcdh.edge_confidence.values()))
            )
            result_dict["num_consensus_edges"] = len(fedcdh.edge_confidence)

    return result_dict


# ============================================================
# Experiment Suites
# ============================================================


def run_scenario_comparison(
    config_name="medium",
    data_type="linear",
    seeds=None,
    device=None,
    use_ci_ranking=False,
    sparsity_percentile=0.2,
    force_clusters=None,
    num_local_clusters=None,
    skip_eval=False,
    horizontal_aggregation="structure_voting",
    test_all_horizontal_strategies=False,
    structure_vote_threshold=0.4,
    alpha=0.08,
):
    """
    Benchmark: Compare 3 SPN scenarios (H/V/Hy) with V3 structure-preserving aggregation.

    Tests which aggregation strategy performs best with V3 fixes.

    Args:
        config_name: Configuration to use (quick/small/medium/large/sachs)
        data_type: Type of data generation (linear/nonlinear)
        seeds: Random seeds for multiple runs
        device: Device to use (cuda/mps/cpu) or None for global DEVICE
        use_ci_ranking: Enable CI ranking (experimental)
        sparsity_percentile: Sparsity for ranking (if enabled)
        force_clusters: Force specific number of clusters (bypasses BIC)
        num_local_clusters: Number of local clusters per client
        skip_eval: Skip expensive SPN evaluation
        horizontal_aggregation: Strategy for horizontal mode (V3)
        test_all_horizontal_strategies: Test all 3 horizontal strategies (V3)
    """
    if seeds is None:
        seeds = SEEDS

    active_device = device if device is not None else DEVICE

    config = BENCHMARK_CONFIGS[config_name]
    scenarios = ["horizontal", "vertical", "hybrid"]

    # V3: Test all horizontal strategies if requested
    if test_all_horizontal_strategies:
        horizontal_strategies = ["structure_voting", "ll_weighted", "mixture"]
    else:
        horizontal_strategies = [horizontal_aggregation]

    logging.info("\n" + "=" * 80)
    logging.info(f"BENCHMARK: V3 SCENARIO COMPARISON ({config_name})")
    logging.info(config["description"])
    logging.info(f"Data type: {data_type}")
    logging.info(f"V3 Structure-Preserving Aggregation: ENABLED")
    if test_all_horizontal_strategies:
        logging.info(f"Horizontal strategies: {horizontal_strategies}")
    else:
        logging.info(f"Horizontal aggregation: {horizontal_aggregation}")
    if use_ci_ranking:
        logging.info(f"CI Ranking: ENABLED (sparsity={sparsity_percentile})")
    else:
        logging.info(f"CI Ranking: DISABLED (alpha=0.05)")
    logging.info(f"Seeds: {seeds}")
    logging.info("=" * 80)

    results = []
    run_counter = 1

    # Calculate total runs (horizontal may have multiple strategies)
    horizontal_runs = len(horizontal_strategies) if "horizontal" in scenarios else 0
    other_runs = len([s for s in scenarios if s != "horizontal"])
    total_runs = (horizontal_runs + other_runs) * len(seeds)

    for scenario in scenarios:
        # For horizontal mode, test all requested strategies
        strategies_to_test = (
            horizontal_strategies if scenario == "horizontal" else [None]
        )

        for strategy in strategies_to_test:
            for seed in seeds:
                strategy_label = f" ({strategy})" if strategy else ""
                logging.info(
                    f"\n[Run {run_counter}/{total_runs}] Starting: {scenario}{strategy_label} (seed={seed})"
                )

                result = run_single_experiment(
                    config_name=config_name,
                    config=config,
                    scenario=scenario,
                    data_type=data_type,
                    seed=seed,
                    device=active_device,
                    use_ci_ranking=use_ci_ranking,
                    sparsity_percentile=sparsity_percentile,
                    force_clusters=force_clusters,
                    num_local_clusters=num_local_clusters,
                    skip_eval=skip_eval,
                    horizontal_aggregation=strategy
                    if strategy
                    else horizontal_aggregation,
                    structure_vote_threshold=structure_vote_threshold,
                    alpha=alpha,
                )
                results.append(result)

                # Log result with eval_dir
                eval_dir = result.get("eval_dir", "N/A")
                strategy_info = f" ({strategy})" if strategy else ""
                logging.info(
                    f"  {scenario:12s}{strategy_info:18s} seed={seed} | "
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
                f"Dimensions:     d={result['d']}, K={result['K']}, n_total={result['n_total']}, n_per_client={result['n_per_client']}\n"
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


def main(
    config_name,
    data_type="linear",
    device=None,
    seeds=None,
    use_ci_ranking=False,
    sparsity_percentile=0.2,
    force_clusters=None,
    num_local_clusters=None,
    skip_eval=False,
    horizontal_aggregation="structure_voting",
    test_all_horizontal_strategies=False,
    structure_vote_threshold=0.4,
    alpha=0.08,
):
    """
    Run V3 scenario comparison benchmark with structure-preserving aggregation.

    Args:
        config_name: Configuration to use (quick/small/medium/large/sachs)
        data_type: Type of data generation (linear/nonlinear)
        device: Device to use (cuda/mps/cpu) or None for auto-detect
        seeds: List of random seeds or None for default
        use_ci_ranking: Enable CI ranking (experimental)
        sparsity_percentile: Sparsity for ranking (if enabled)
        force_clusters: Force specific number of clusters (overrides config default)
        num_local_clusters: Number of local clusters per client (overrides config default)
        skip_eval: Skip SPN quality evaluation for faster smoke tests
        horizontal_aggregation: V3 horizontal aggregation strategy
        test_all_horizontal_strategies: Test all 3 horizontal strategies (V3)
        structure_vote_threshold: Voting threshold for structure_voting (V3, default: 0.4)
        alpha: CI test significance level (V3 optimized, default: 0.08)
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
    logging.info("FEDCDH V3 BENCHMARK SUITE - STRUCTURE-PRESERVING AGGREGATION")
    logging.info("=" * 80)
    logging.info(f"Configuration: {config_name}")
    logging.info(f"Data type: {data_type}")
    logging.info(f"Device: {active_device}")
    logging.info(f"V3 Structure-Preserving Aggregation: ENABLED")
    if test_all_horizontal_strategies:
        logging.info(f"Horizontal strategies: structure_voting, ll_weighted, mixture")
    else:
        logging.info(f"Horizontal aggregation: {horizontal_aggregation}")
    if use_ci_ranking:
        logging.info(f"CI Ranking: ENABLED (sparsity={sparsity_percentile})")
    else:
        logging.info(f"CI Ranking: DISABLED (alpha=0.05)")

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

    # Run V3 scenario comparison benchmark
    all_results = run_scenario_comparison(
        config_name=config_name,
        data_type=data_type,
        seeds=active_seeds,
        device=active_device,
        use_ci_ranking=use_ci_ranking,
        sparsity_percentile=sparsity_percentile,
        force_clusters=force_clusters,
        num_local_clusters=num_local_clusters,
        skip_eval=skip_eval,
        horizontal_aggregation=horizontal_aggregation,
        test_all_horizontal_strategies=test_all_horizontal_strategies,
        structure_vote_threshold=structure_vote_threshold,
        alpha=alpha,
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

    parser = argparse.ArgumentParser(
        description="FedCDH V3 Benchmark Suite - Structure-Preserving Aggregation + GlobalSumOfProducts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="quick",
        choices=["quick", "small", "medium", "large", "sachs", "law_school"],
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
    # v2 experimental features
    parser.add_argument(
        "--use-ci-ranking",
        action="store_true",
        help="Enable CI ranking (experimental, replaces alpha=0.05)",
    )
    parser.add_argument(
        "--sparsity-percentile",
        type=float,
        default=0.2,
        help="Sparsity percentile for CI ranking (default: 0.2 = top 20%%)",
    )
    parser.add_argument(
        "--force-clusters",
        type=int,
        default=None,
        help="Force specific number of clusters (bypasses BIC). Recommended: 2 for MEDIUM config to prevent data fragmentation.",
    )
    parser.add_argument(
        "--num-local-clusters",
        type=int,
        default=None,
        help="Number of local clusters per client (default: use config value). V3: Config defaults are optimized (typically 3 for better sum-over-products).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip expensive SPN quality evaluation (UMAP, independence tests, dashboards) for faster smoke tests. Only computes F1 scores.",
    )
    # V3 arguments
    parser.add_argument(
        "--horizontal-aggregation",
        type=str,
        default="structure_voting",
        choices=["structure_voting", "ll_weighted", "mixture"],
        help="V3 horizontal aggregation strategy (default: structure_voting). structure_voting=democratic voting (recommended), ll_weighted=quality-weighted, mixture=V2 baseline.",
    )
    parser.add_argument(
        "--test-all-horizontal-strategies",
        action="store_true",
        help="V3: Test all 3 horizontal strategies (structure_voting, ll_weighted, mixture) instead of just one.",
    )
    parser.add_argument(
        "--structure-vote-threshold",
        type=float,
        default=0.4,
        help="V3: Voting threshold for structure_voting (default: 0.4 = 40%% agreement). Lower = more edges, higher = fewer edges.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.08,
        help="V3: CI test significance level (default: 0.08, relaxed from V2's 0.05 for better recall). Lower = stricter, higher = more permissive.",
    )

    args = parser.parse_args()

    # Pass all parameters to main()
    df, summary = main(
        config_name=args.config,
        data_type=args.data_type,
        device=args.device,
        seeds=args.seeds,
        use_ci_ranking=args.use_ci_ranking,
        sparsity_percentile=args.sparsity_percentile,
        force_clusters=args.force_clusters,
        num_local_clusters=args.num_local_clusters,
        skip_eval=args.skip_eval,
        horizontal_aggregation=args.horizontal_aggregation,
        test_all_horizontal_strategies=args.test_all_horizontal_strategies,
        structure_vote_threshold=args.structure_vote_threshold,
        alpha=args.alpha,
    )
