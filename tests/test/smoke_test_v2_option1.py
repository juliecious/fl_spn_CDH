#!/usr/bin/env python3
"""
Smoke Test for V2 Option 1: FedCDH Baseline (no k-means)

Tests that the implementation works with num_clusters = K_clients
Quick test on CPU to validate basic functionality before GPU runs.

Usage:
    python tests/test/smoke_test_v2_option1.py
"""

import sys
import os
import logging
import time
import numpy as np
from argparse import Namespace

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    simulate_dag,
    simulate_parameter,
    my_simulate_linear_gaussian,
    set_random_seed,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_smoke_test():
    """Run minimal smoke test for Option 1."""
    logging.info("=" * 80)
    logging.info("V2 OPTION 1 SMOKE TEST - FedCDH Baseline (CPU)")
    logging.info("=" * 80)

    # Minimal config for fast testing
    seed = 42
    d = 5  # Small number of features
    K = 2  # Two clients
    n_per_client = 50  # Minimal samples
    device = "cpu"

    set_random_seed(seed)

    logging.info(f"Config: d={d}, K={K}, n_per_client={n_per_client}, device={device}")

    # Generate synthetic data
    logging.info("Generating synthetic DAG and data...")
    s0 = d  # Expected edges
    B = simulate_dag(d, s0, graph_type="ER")
    W = simulate_parameter(B)

    # Generate linear data
    n_total = K * n_per_client
    X, choice = my_simulate_linear_gaussian(W, K=K, n=n_total, sem_type="gauss")

    # Horizontal split
    X_splits = []
    for k in range(K):
        start = k * n_per_client
        end = (k + 1) * n_per_client
        X_splits.append(X[start:end, :])

    B_true = B

    logging.info(f"Data splits: {[x.shape for x in X_splits]}")

    # Create FedCDH with Option 1 settings
    args = Namespace(
        K=K,
        d=d,
        n=n_per_client,
        scenario="horizontal",
        model_type="synthetic",
        ci_method="spn",
        alpha=0.05,
        epochs=20,  # Very few epochs for smoke test
        device=device,
        # V2 Option 1 settings
        data_type="linear",
        use_ci_ranking=False,
        use_kmeans_clustering=False,  # KEY: Disable k-means
        force_num_clusters=None,  # No override
    )

    logging.info("\nInitializing FedCDH with Option 1 (no k-means)...")
    fedcdh = FedCDH(args)

    # Feature maps (horizontal: all features on all clients)
    c_indx = [list(range(d)) for _ in range(K)]

    # Create client index vector for augmentation
    c_indx_vector = np.repeat(np.arange(K), n_per_client).reshape(-1, 1)

    # Run causal discovery
    logging.info("\nRunning FedCDH.fit()...")
    logging.info(f"c_indx (feature maps): {c_indx}")
    logging.info(f"c_indx_vector shape: {c_indx_vector.shape}")
    start_time = time.time()

    try:
        # Note: FedCDH.fit() expects (X_splits, feature_maps, B_true)
        # But internally uses the client index from data concatenation
        results = fedcdh.fit(X_splits, c_indx, B_true)
        elapsed = time.time() - start_time

        logging.info(f"\n✓ FedCDH.fit() completed in {elapsed:.1f}s")

        # Check results
        if results is not None:
            logging.info("\n" + "=" * 80)
            logging.info("SMOKE TEST RESULTS")
            logging.info("=" * 80)
            logging.info(f"Graph shape: {results['G'].graph.shape}")
            logging.info(f"Training time: {elapsed:.2f}s")

            # Check that we used K clusters (not BIC selection)
            if hasattr(fedcdh, "_num_clusters_used"):
                logging.info(f"Number of clusters used: {fedcdh._num_clusters_used}")
                assert (
                    fedcdh._num_clusters_used == K
                ), f"Expected {K} clusters, got {fedcdh._num_clusters_used}"

            logging.info("\n✅ SMOKE TEST PASSED")
            logging.info("Option 1 implementation working correctly!")
            logging.info("Ready for GPU testing.")
            return True

        else:
            logging.error("❌ FedCDH.fit() returned None")
            return False

    except Exception as e:
        logging.error(f"❌ SMOKE TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
