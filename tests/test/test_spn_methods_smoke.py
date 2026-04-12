"""
Smoke test for 3 SPN methods (Horizontal, Vertical, Hybrid).
Tests basic functionality with small parameters to ensure it completes within 2 minutes on CPU.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

from causallearn.utils.data_utils import (
    my_simulate_general_hetero,
    my_simulate_linear_gaussian,
    set_random_seed,
    simulate_dag,
    simulate_parameter,
    count_skeleton_accuracy,
    get_cpdag_from_cdnod,
    get_dag_from_pdag,
    count_dag_accuracy,
)
from causallearn.search.FCMBased.FedCDH import FedCDH


def create_test_data(d=5, K=2, n=200, seed=42, data_type="linear", sem_type="gauss"):
    """
    Create test data with heterogeneity across K clients.

    Args:
        d: Number of variables
        K: Number of clients
        n: Total number of samples (will be split across clients)
        seed: Random seed
        data_type: 'linear' or 'nonlinear'
        sem_type: SEM type ('gauss', 'general', etc.)

    Returns:
        W: Ground truth DAG weights
        X: Generated data
        c_indx: Context indices
        choice: Heterogeneous variables
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

    # Create context indices (which client each sample belongs to)
    c_indx = np.repeat(np.arange(K), n // K).reshape(-1, 1)

    logging.info(f"Generated DAG with {d} nodes, {np.sum(B)} edges")
    logging.info(f"Heterogeneous variables: {choice}")

    return W, B, X, c_indx, choice


def partition_data(X, c_indx, K, scenario):
    """
    Partition data according to federated scenario.

    Args:
        X: Full dataset [n, d]
        c_indx: Context indices [n, 1]
        K: Number of clients
        scenario: 'horizontal', 'vertical', or 'hybrid'

    Returns:
        X_splits: List of data partitions per client
    """
    n, d = X.shape

    if scenario == "horizontal":
        # Horizontal: Each client has all features but different samples
        samples_per_client = n // K
        X_splits = [
            X[k * samples_per_client : (k + 1) * samples_per_client, :]
            for k in range(K)
        ]

    elif scenario == "vertical":
        # Vertical: Each client has different features but all samples
        features_per_client = d // K
        X_splits = [
            X[:, k * features_per_client : (k + 1) * features_per_client]
            for k in range(K)
        ]
        # Handle remainder features (give to last client)
        if d % K != 0:
            X_splits[-1] = X[:, (K - 1) * features_per_client :]

    elif scenario == "hybrid":
        # Hybrid: Both sample and feature partitioning
        # Each client has different samples but OVERLAPPING features
        # This is different from pure vertical where features don't overlap
        samples_per_client = n // K
        features_per_client = d // K
        X_splits = []
        for k in range(K):
            # Each client gets its own sample range
            # But gets a subset of features that may overlap with other clients
            # For simplicity in smoke test: give each client all features
            # (in production, this would be more sophisticated)
            X_k = X[k * samples_per_client : (k + 1) * samples_per_client, :]
            X_splits.append(X_k)
        # Note: In true hybrid scenario, feature maps would specify which features each client sees
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return X_splits


def run_smoke_test(
    scenario="horizontal",
    d=5,
    K=2,
    n=200,
    epochs=20,
    seed=42,
    data_type="linear",
    sem_type="gauss",
):
    """
    Run smoke test for a single scenario.

    Args:
        scenario: 'horizontal', 'vertical', or 'hybrid'
        d: Number of variables
        K: Number of clients
        n: Total number of samples
        epochs: Training epochs (small for smoke test)
        seed: Random seed
        data_type: 'linear' or 'nonlinear'
        sem_type: SEM type ('gauss', 'general', etc.)
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Testing {scenario.upper()} scenario")
    logging.info(f"Parameters: d={d}, K={K}, n={n}, epochs={epochs}, seed={seed}")
    logging.info(f"Data type: {data_type}, SEM type: {sem_type}")
    logging.info(f"{'='*60}\n")

    start_time = time.time()

    # Step 1: Generate test data
    logging.info("Step 1: Generating test data...")
    W, B, X, c_indx, choice = create_test_data(
        d=d, K=K, n=n, seed=seed, data_type=data_type, sem_type=sem_type
    )

    # Step 2: Partition data
    logging.info(f"Step 2: Partitioning data for {scenario} scenario...")
    X_splits = partition_data(X, c_indx, K, scenario)
    logging.info(f"  Created {len(X_splits)} partitions")
    for i, split in enumerate(X_splits):
        logging.info(f"  Client {i}: shape {split.shape}")

    # Step 3: Setup FedCDH
    logging.info("Step 3: Initializing FedCDH...")
    args = Namespace(
        K=K,
        d=d,
        n=n // K,
        scenario=scenario,
        model_type="synthetic",
        ci_method="spn",
        alpha=0.05,
        epochs=epochs,
        device="cpu",
        skip_bic=True,  # Skip BIC for faster smoke test
    )

    fedcdh = FedCDH(args)

    # Step 4: Fit model and get results
    logging.info("Step 4: Training FedCDH (this may take a minute)...")
    train_start = time.time()
    results = fedcdh.fit(X_splits, c_indx, B)
    train_time = time.time() - train_start
    logging.info(f"  Training completed in {train_time:.2f}s")

    # Step 5: Extract metrics from results
    # FedCDH.fit() returns a dictionary with metrics already computed
    logging.info("Step 5: Extracting metrics from results...")

    # Get skeleton metrics
    skeleton_f1 = results.get("f1_skeleton", 0.0)
    skeleton_precision = results.get("precision_skeleton", 0.0)
    skeleton_recall = results.get("recall_skeleton", 0.0)
    skeleton_shd = results.get("shd_skeleton", 0)

    logging.info(f"  Skeleton F1:        {skeleton_f1:.3f}")
    logging.info(f"  Skeleton Precision: {skeleton_precision:.3f}")
    logging.info(f"  Skeleton Recall:    {skeleton_recall:.3f}")
    logging.info(f"  Skeleton SHD:       {skeleton_shd}")

    # Get DAG metrics
    dag_f1 = results.get("f1", 0.0)
    dag_precision = results.get("precision", 0.0)
    dag_recall = results.get("recall", 0.0)
    dag_shd = results.get("shd", 0)

    logging.info(
        f"  DAG F1:             {dag_f1:.3f}"
        if dag_f1 is not None
        else "  DAG F1:             None"
    )
    logging.info(
        f"  DAG Precision:      {dag_precision:.3f}"
        if dag_precision is not None
        else "  DAG Precision:      None"
    )
    logging.info(
        f"  DAG Recall:         {dag_recall:.3f}"
        if dag_recall is not None
        else "  DAG Recall:         None"
    )
    logging.info(f"  DAG SHD:            {dag_shd}")

    # Step 6: Report statistics
    total_time = time.time() - start_time
    cd_time = results.get("time_cd", 0.0)
    comm_cost = results.get("comm_cost", 0.0)

    logging.info(f"\n{'='*60}")
    logging.info(f"SUMMARY - {scenario.upper()}")
    logging.info(f"{'='*60}")
    logging.info(f"Total runtime:      {total_time:.2f}s")
    logging.info(f"Training time:      {train_time:.2f}s")
    logging.info(f"CD time:            {cd_time:.2f}s")
    logging.info(f"Skeleton F1:        {skeleton_f1:.3f}")
    logging.info(
        f"DAG F1:             {dag_f1:.3f}"
        if dag_f1 is not None
        else "DAG F1:             None"
    )
    logging.info(f"Comm cost (bytes):  {comm_cost:.0f}")
    logging.info(f"{'='*60}\n")

    # Check if test passed (basic sanity checks)
    passed = True
    reasons = []

    # Check 1: Runtime should be reasonable
    if total_time > 120:  # 2 minutes
        passed = False
        reasons.append(f"Runtime too long: {total_time:.1f}s > 120s")

    # Check 2: Should discover at least some edges
    if skeleton_f1 is None or skeleton_f1 == 0:
        passed = False
        reasons.append("No edges discovered (F1=0 or None)")

    # Check 3: Model should be trained
    if fedcdh.fed_spn_model is None:
        passed = False
        reasons.append("SPN model not created")

    return {
        "scenario": scenario,
        "passed": passed,
        "reasons": reasons,
        "total_time": total_time,
        "skeleton_f1": skeleton_f1,
        "dag_f1": dag_f1,
        "shd": dag_shd,
    }


def main():
    """Run smoke tests for all three scenarios."""
    logging.info("\n" + "=" * 70)
    logging.info("FEDCDH SPN METHODS SMOKE TEST")
    logging.info("Testing 3 scenarios: Horizontal, Vertical, Hybrid")
    logging.info("=" * 70 + "\n")

    # Test parameters (small for fast smoke test)
    test_params = {
        "d": 5,  # Small number of variables
        "K": 2,  # Minimal number of clients
        "n": 200,  # Small sample size
        "epochs": 20,  # Reduced epochs for speed
        "seed": 42,
        "data_type": "linear",  # Linear is faster than nonlinear
    }

    scenarios = ["horizontal", "vertical", "hybrid"]
    results = []

    overall_start = time.time()

    # Run test for each scenario
    for scenario in scenarios:
        result = run_smoke_test(scenario=scenario, **test_params)
        results.append(result)

    overall_time = time.time() - overall_start

    # Print final summary
    logging.info("\n" + "=" * 70)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 70)
    logging.info(f"Total test time: {overall_time:.2f}s")
    logging.info("")

    all_passed = True
    for result in results:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        logging.info(
            f"{result['scenario'].upper():12s} {status:8s}  "
            f"Time: {result['total_time']:5.1f}s  "
            f"Skeleton F1: {result['skeleton_f1']:.3f}  "
            f"DAG F1: {result['dag_f1']:.3f}"
            if result["dag_f1"] is not None
            else f"DAG F1: None"
        )

        if not result["passed"]:
            all_passed = False
            for reason in result["reasons"]:
                logging.info(f"  └─ {reason}")

    logging.info("=" * 70)

    if all_passed:
        logging.info("\n✓ All tests PASSED!")
        return 0
    else:
        logging.info("\n✗ Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
