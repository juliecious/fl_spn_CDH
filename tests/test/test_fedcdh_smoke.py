"""
Consolidated FedCDH Smoke Test Suite.

This unified test script combines all smoke tests for FedCDH with SPN-based causal discovery:
1. Basic functionality test (3 SPN scenarios: Horizontal, Vertical, Hybrid)
2. Data generation scenarios (Linear vs Nonlinear)
3. Integrated evaluation (quality metrics + independence structure)

All tests complete in < 5 minutes on CPU.
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
    my_simulate_general_hetero,
    my_simulate_linear_gaussian,
    set_random_seed,
    simulate_dag,
    simulate_parameter,
)
from causallearn.search.FCMBased.FedCDH import FedCDH


# ============================================================
# Data Generation
# ============================================================


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
        B: Ground truth binary adjacency matrix
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
        # Hybrid: Each client gets different samples with all features
        samples_per_client = n // K
        X_splits = []
        for k in range(K):
            X_k = X[k * samples_per_client : (k + 1) * samples_per_client, :]
            X_splits.append(X_k)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return X_splits


# ============================================================
# Test Execution
# ============================================================


def run_single_test(
    scenario, d, K, n, epochs, seed, data_type, sem_type, ci_method="spn"
):
    """
    Run a single FedCDH test.

    Args:
        scenario: 'horizontal', 'vertical', or 'hybrid'
        d: Number of variables
        K: Number of clients
        n: Total samples
        epochs: Training epochs
        seed: Random seed
        data_type: 'linear' or 'nonlinear'
        sem_type: SEM type ('gauss', etc.)
        ci_method: CI test method ('spn', 'fisherz', etc.)

    Returns:
        Dictionary with test results
    """
    start_time = time.time()

    # Step 1: Generate test data
    W, B, X, c_indx, choice = create_test_data(
        d=d, K=K, n=n, seed=seed, data_type=data_type, sem_type=sem_type
    )

    # Step 2: Partition data
    X_splits = partition_data(X, c_indx, K, scenario)

    # Step 3: Setup FedCDH
    args = Namespace(
        K=K,
        d=d,
        n=n // K,
        scenario=scenario,
        model_type="synthetic",
        ci_method=ci_method,
        alpha=0.05,
        epochs=epochs,
        device="cpu",
        skip_bic=True,  # Skip BIC for faster tests
    )

    fedcdh = FedCDH(args)

    # Step 4: Fit model
    results = fedcdh.fit(X_splits, c_indx, B)

    # Step 5: Extract metrics
    total_time = time.time() - start_time
    skeleton_f1 = results.get("f1_skeleton", 0.0)
    dag_f1 = results.get("f1", 0.0)
    shd = results.get("shd", 0)

    # Check if test passed (basic sanity checks)
    passed = True
    reasons = []

    if total_time > 120:  # 2 minutes per test
        passed = False
        reasons.append(f"Runtime too long: {total_time:.1f}s > 120s")

    if skeleton_f1 is None or skeleton_f1 == 0:
        passed = False
        reasons.append("No edges discovered (F1=0 or None)")

    if fedcdh.fed_spn_model is None and ci_method == "spn":
        passed = False
        reasons.append("SPN model not created")

    return {
        "scenario": scenario,
        "data_type": data_type,
        "sem_type": sem_type,
        "passed": passed,
        "reasons": reasons,
        "total_time": total_time,
        "skeleton_f1": skeleton_f1,
        "dag_f1": dag_f1,
        "shd": shd,
        "eval_dir": getattr(fedcdh, "spn_eval_dir", None),
    }


# ============================================================
# Test Suites
# ============================================================


def test_basic_functionality(verbose=True):
    """
    Test 1: Basic Functionality
    Tests all 3 SPN scenarios with linear Gaussian data.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("TEST 1: BASIC FUNCTIONALITY")
        print("Testing 3 SPN scenarios: Horizontal, Vertical, Hybrid")
        print("Data: Linear Gaussian (my_simulate_linear_gaussian)")
        print("=" * 80)

    test_params = {
        "d": 5,
        "K": 2,
        "n": 200,
        "epochs": 20,
        "seed": 42,
        "data_type": "linear",
        "sem_type": "gauss",
    }

    scenarios = ["horizontal", "vertical", "hybrid"]
    results = []

    for scenario in scenarios:
        if verbose:
            print(f"\nRunning {scenario.upper()}...")
        result = run_single_test(scenario=scenario, **test_params)
        results.append(result)

        if verbose:
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(
                f"  {status}  Time: {result['total_time']:5.1f}s  "
                f"Skeleton F1: {result['skeleton_f1']:.3f}  "
                f"DAG F1: {result['dag_f1']:.3f}"
            )
            if result["eval_dir"]:
                print(f"  Eval dir: {result['eval_dir']}")

    all_passed = all(r["passed"] for r in results)
    return all_passed, results


def test_data_generation_scenarios(verbose=True):
    """
    Test 2: Data Generation Scenarios
    Tests linear vs nonlinear data generation with Gaussian noise.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("TEST 2: DATA GENERATION SCENARIOS")
        print("Scenario A: Linear Gaussian (my_simulate_linear_gaussian)")
        print("Scenario B: Nonlinear Gaussian (my_simulate_general_hetero)")
        print("=" * 80)

    test_params = {
        "d": 5,
        "K": 2,
        "n": 200,
        "epochs": 20,
        "seed": 42,
    }

    scenarios = ["horizontal", "vertical", "hybrid"]
    data_configs = [
        {"data_type": "linear", "sem_type": "gauss", "name": "Linear Gaussian"},
        {"data_type": "nonlinear", "sem_type": "gauss", "name": "Nonlinear Gaussian"},
    ]

    results = []

    for data_config in data_configs:
        if verbose:
            print(f"\n{data_config['name']}:")

        for scenario in scenarios:
            result = run_single_test(
                scenario=scenario,
                data_type=data_config["data_type"],
                sem_type=data_config["sem_type"],
                **test_params,
            )
            result["data_config"] = data_config["name"]
            results.append(result)

            if verbose:
                status = "✓ PASS" if result["passed"] else "✗ FAIL"
                print(
                    f"  {scenario.upper():12s} {status}  "
                    f"Time: {result['total_time']:5.1f}s  "
                    f"F1: {result['skeleton_f1']:.3f}"
                )

    all_passed = all(r["passed"] for r in results)
    return all_passed, results


def test_integrated_evaluation(verbose=True):
    """
    Test 3: Integrated Evaluation
    Tests the integrated SPN quality + independence structure evaluation.
    Uses ci_method='spn' which triggers automatic evaluation.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("TEST 3: INTEGRATED EVALUATION")
        print("Testing automatic SPN quality + independence structure evaluation")
        print("=" * 80)

    test_params = {
        "d": 4,
        "K": 2,
        "n": 100,
        "epochs": 10,
        "seed": 42,
        "data_type": "linear",
        "sem_type": "gauss",
        "ci_method": "spn",
    }

    # Test horizontal only (fastest)
    result = run_single_test(scenario="horizontal", **test_params)

    if verbose:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"\nHORIZONTAL {status}")
        print(f"  Time: {result['total_time']:.1f}s")
        print(f"  Skeleton F1: {result['skeleton_f1']:.3f}")
        if result["eval_dir"]:
            print(f"  Evaluation artifacts saved to: {result['eval_dir']}")
            # Check if artifacts exist
            import os

            if os.path.exists(result["eval_dir"]):
                files = os.listdir(result["eval_dir"])
                print(f"  Generated {len(files)} files: {', '.join(files)}")

    return result["passed"], [result]


# ============================================================
# Main Test Runner
# ============================================================


def main():
    """Run all smoke tests."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("\n" + "=" * 80)
    print("FEDCDH CONSOLIDATED SMOKE TEST SUITE")
    print("=" * 80)
    print("\nThis test suite validates:")
    print("  1. Basic functionality (3 SPN scenarios)")
    print("  2. Data generation (linear vs nonlinear)")
    print("  3. Integrated evaluation (quality + independence)")
    print("\nEstimated runtime: 3-5 minutes on CPU")
    print("=" * 80)

    overall_start = time.time()
    all_results = []

    # Test 1: Basic Functionality
    passed_1, results_1 = test_basic_functionality(verbose=True)
    all_results.extend(results_1)

    # Test 2: Data Generation Scenarios
    passed_2, results_2 = test_data_generation_scenarios(verbose=True)
    all_results.extend(results_2)

    # Test 3: Integrated Evaluation
    passed_3, results_3 = test_integrated_evaluation(verbose=True)
    all_results.extend(results_3)

    overall_time = time.time() - overall_start

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total tests run: {len(all_results)}")
    print(f"Total time: {overall_time:.1f}s")
    print("")

    # Count results by test suite
    passed_count = sum(1 for r in all_results if r["passed"])
    failed_count = len(all_results) - passed_count

    print(f"Results:")
    print(f"  Test 1 (Basic Functionality):     {'✓ PASS' if passed_1 else '✗ FAIL'}")
    print(f"  Test 2 (Data Generation):          {'✓ PASS' if passed_2 else '✗ FAIL'}")
    print(f"  Test 3 (Integrated Evaluation):    {'✓ PASS' if passed_3 else '✗ FAIL'}")
    print("")
    print(f"Overall: {passed_count}/{len(all_results)} tests passed")

    if failed_count > 0:
        print(f"\nFailed tests:")
        for r in all_results:
            if not r["passed"]:
                print(f"  - {r.get('data_config', r['data_type'])} / {r['scenario']}")
                for reason in r["reasons"]:
                    print(f"    └─ {reason}")

    print("=" * 80)

    if passed_1 and passed_2 and passed_3:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
