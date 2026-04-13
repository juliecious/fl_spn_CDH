"""
Quick smoke test for simplified hybrid implementation.

Simplified Hybrid: Product-then-Mixture
- Each client partitions features into 2 groups
- Each client trains one SPN per feature group
- Product combines feature groups per client
- Mixture combines clients

Test: d=5, K=2, n=200, 20 epochs
"""

import sys
import os
import time
import logging
import numpy as np
from argparse import Namespace

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from causallearn.utils.data_utils import (
    my_simulate_linear_gaussian,
    set_random_seed,
    simulate_dag,
    simulate_parameter,
)
from causallearn.search.FCMBased.FedCDH import FedCDH

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def create_test_data(d=5, K=2, n=200, seed=42):
    """Create test data."""
    set_random_seed(seed)
    s0 = d
    B = simulate_dag(d, s0, graph_type="ER")
    W = simulate_parameter(B)
    X, choice = my_simulate_linear_gaussian(W, K=K, n=n, sem_type="gauss")
    c_indx = np.repeat(np.arange(K), n // K).reshape(-1, 1)
    return W, B, X, c_indx, choice


def partition_data_hybrid(X, c_indx, K, num_feature_groups=2):
    """
    Hybrid partitioning: samples + feature groups.

    Returns:
        X_splits: List of data splits (samples partitioned)
        feature_groups: List of feature group definitions
    """
    n, d = X.shape

    # Partition samples across clients (like horizontal)
    samples_per_client = n // K
    X_splits = [
        X[k * samples_per_client : (k + 1) * samples_per_client, :] for k in range(K)
    ]

    # Define feature groups (split features into groups)
    features_per_group = d // num_feature_groups
    feature_groups = []
    for g in range(num_feature_groups):
        if g == num_feature_groups - 1:
            # Last group gets remaining features
            group = list(range(g * features_per_group, d))
        else:
            group = list(range(g * features_per_group, (g + 1) * features_per_group))
        feature_groups.append(group)

    logging.info(
        f"Hybrid partitioning: {K} clients, {num_feature_groups} feature groups"
    )
    logging.info(f"  Feature groups: {feature_groups}")

    return X_splits, feature_groups


def test_hybrid_simplified():
    """Test simplified hybrid implementation."""
    logging.info("\n" + "=" * 70)
    logging.info("SIMPLIFIED HYBRID IMPLEMENTATION - SMOKE TEST")
    logging.info("=" * 70)

    # Parameters
    d = 5
    K = 2
    n = 200
    epochs = 20
    seed = 42
    num_feature_groups = 2

    logging.info(f"\nParameters:")
    logging.info(f"  Variables (d): {d}")
    logging.info(f"  Clients (K): {K}")
    logging.info(f"  Samples (n): {n}")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Feature groups: {num_feature_groups}")

    start_time = time.time()

    # Generate data
    logging.info("\nGenerating test data...")
    W, B, X, c_indx, choice = create_test_data(d=d, K=K, n=n, seed=seed)

    # Partition data for hybrid
    logging.info("\nPartitioning data for hybrid scenario...")
    X_splits, feature_groups = partition_data_hybrid(X, c_indx, K, num_feature_groups)

    # Setup FedCDH
    logging.info("\nInitializing FedCDH with hybrid scenario...")
    args = Namespace(
        K=K,
        d=d,
        n=n // K,
        scenario="hybrid",
        model_type="synthetic",
        ci_method="spn",
        alpha=0.05,
        epochs=epochs,
        device="cpu",
        skip_bic=True,
        num_feature_groups=num_feature_groups,  # New parameter
    )

    fedcdh = FedCDH(args)

    # Store feature groups for hybrid processing
    fedcdh.feature_groups = feature_groups

    # Fit model
    logging.info("\nTraining FedCDH with simplified hybrid approach...")
    try:
        results = fedcdh.fit(X_splits, c_indx, B)
        total_time = time.time() - start_time

        # Extract metrics
        skeleton_f1 = results.get("f1_skeleton", 0.0)
        dag_f1 = results.get("f1", 0.0)
        skeleton_shd = results.get("shd_skeleton", 0)

        logging.info("\n" + "=" * 70)
        logging.info("RESULTS")
        logging.info("=" * 70)
        logging.info(f"Total time: {total_time:.2f}s")
        logging.info(f"Skeleton F1: {skeleton_f1:.3f}")
        logging.info(f"Skeleton SHD: {skeleton_shd}")
        logging.info(f"DAG F1: {dag_f1:.3f}" if dag_f1 is not None else "DAG F1: None")

        # Check success criteria
        passed = True
        if total_time > 120:
            passed = False
            logging.warning(f"Runtime too long: {total_time:.1f}s > 120s")

        if skeleton_f1 is None or skeleton_f1 == 0:
            passed = False
            logging.warning("No edges discovered (F1=0 or None)")

        if fedcdh.fed_spn_model is None:
            passed = False
            logging.warning("SPN model not created")

        status = "✓ PASS" if passed else "✗ FAIL"
        logging.info(f"\nTest Status: {status}")
        logging.info("=" * 70)

        return passed, results

    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False, {}


if __name__ == "__main__":
    passed, results = test_hybrid_simplified()
    sys.exit(0 if passed else 1)
