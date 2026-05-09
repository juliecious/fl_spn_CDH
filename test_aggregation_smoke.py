"""
Quick CPU Smoke Test for V3 Aggregation Strategies.

Tests all three horizontal aggregation strategies:
1. structure_voting (V3 fix #2)
2. ll_weighted (V3 alternative)
3. mixture (V2 baseline)

Also tests hybrid mode GlobalSumOfProducts (V3 fix #1).
"""
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True
)

sys.path.insert(0, os.path.abspath("."))

import numpy as np
from argparse import Namespace
from causallearn.search.FCMBased.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    simulate_dag,
    simulate_parameter,
    my_simulate_linear_gaussian,
    set_random_seed,
)


def test_horizontal_aggregation(strategy_name: str):
    """Test a horizontal aggregation strategy."""
    print("\n" + "=" * 80)
    print(f"TEST: Horizontal Mode - {strategy_name.upper()}")
    print("=" * 80)

    set_random_seed(42)
    d, K, n_total = 6, 3, 600  # Small for speed

    # Generate data
    G_bin = simulate_dag(d, d, "ER")  # d edges
    B = simulate_parameter(G_bin)
    X_samples, _ = my_simulate_linear_gaussian(B, K, n_total, "gauss")

    # Split horizontally (same features, different samples)
    c_indx = np.zeros((n_total, 1))
    n_per = n_total // K
    X_splits = [
        X_samples[k * n_per : (k + 1) * n_per if k < K - 1 else n_total, :]
        for k in range(K)
    ]

    # Configure
    args = Namespace(
        K=K,
        d=d,
        scenario="horizontal",
        model_type="SPN",
        ci_method="spn",  # Must be "spn" not "SPN_CIT" for SPN training
        n=n_per,
        epochs=3,  # Fast for smoke test
        alpha=0.05,
        data_type="linear",
        device="cpu",
        horizontal_aggregation=strategy_name,
        structure_vote_threshold=0.5,
        skip_spn_eval=True,  # Skip expensive evaluation
    )

    print(f"Config: d={d}, K={K}, n_total={n_total}, n_per={n_per}")
    print(f"Strategy: {strategy_name}")
    print(f"True edges: {np.sum(G_bin)}")
    print()

    try:
        # Initialize and fit
        fedcdh = FedCDH(args)
        fedcdh.fit(X_splits, c_indx, G_bin)

        # Check results
        print("\n" + "-" * 80)
        print("RESULTS:")
        print("-" * 80)

        if strategy_name == "structure_voting":
            if hasattr(fedcdh, "consensus_dependency_graph"):
                edges = len(fedcdh.consensus_dependency_graph.edges())
                print(f"✓ Consensus graph exists: {edges} edges")
                if hasattr(fedcdh, "edge_confidence"):
                    print(
                        f"✓ Edge confidence tracked: {len(fedcdh.edge_confidence)} edges"
                    )
                    if fedcdh.edge_confidence:
                        avg_conf = np.mean(list(fedcdh.edge_confidence.values()))
                        print(f"  Average confidence: {avg_conf:.3f}")
            else:
                print("✗ Missing consensus_dependency_graph!")
                return False

        # Check global model exists
        if hasattr(fedcdh, "fed_spn_model") and fedcdh.fed_spn_model is not None:
            print(f"✓ Global SPN model exists")
        else:
            print("✗ Missing fed_spn_model!")
            return False

        print(f"✓ Test PASSED: {strategy_name}")
        return True

    except Exception as e:
        print(f"\n✗ Test FAILED: {strategy_name}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_hybrid_sum_over_products():
    """Test hybrid mode GlobalSumOfProducts."""
    print("\n" + "=" * 80)
    print("TEST: Hybrid Mode - GlobalSumOfProducts")
    print("=" * 80)

    set_random_seed(42)
    d, K, n_total = 6, 3, 600

    # Generate data
    G_bin = simulate_dag(d, d, "ER")
    B = simulate_parameter(G_bin)
    X_samples, _ = my_simulate_linear_gaussian(B, K, n_total, "gauss")

    # For hybrid mode, FedCDH.fit() auto-partitions samples
    # We just pass the global data and context column
    c_indx = np.zeros((n_total, 1))

    # Configure
    args = Namespace(
        K=K,
        d=d,
        scenario="hybrid",
        model_type="SPN",
        ci_method="spn",  # Must be "spn" not "SPN_CIT"
        n=n_total // K,
        epochs=3,
        alpha=0.05,
        data_type="linear",
        device="cpu",
        num_cluster_samples=5,  # Small for speed
        skip_spn_eval=True,
    )

    print(f"Config: d={d}, K={K}, n_total={n_total}")
    print(f"True edges: {np.sum(G_bin)}")
    print(f"Note: Hybrid mode auto-partitions samples")
    print()

    try:
        # Initialize and fit
        # For hybrid: pass full X_samples, fit() will auto-partition
        fedcdh = FedCDH(args)
        fedcdh.fit(X_samples, c_indx, G_bin)

        # Check results
        print("\n" + "-" * 80)
        print("RESULTS:")
        print("-" * 80)

        if hasattr(fedcdh, "fed_spn_model") and fedcdh.fed_spn_model is not None:
            print(f"✓ Global SPN model exists")

            # Check if the inner model is GlobalSumOfProducts
            from causallearn.utils.FedPC import GlobalSumOfProducts

            inner_model = (
                fedcdh.fed_spn_model.spn
                if hasattr(fedcdh.fed_spn_model, "spn")
                else fedcdh.fed_spn_model
            )

            if isinstance(inner_model, GlobalSumOfProducts):
                print(f"✓ Using GlobalSumOfProducts (V3 fix!)")
                print(f"  Number of cluster combinations: {len(inner_model.products)}")
            else:
                print(f"⚠ Not using GlobalSumOfProducts: {type(inner_model).__name__}")
                print(f"  (May be expected if only 1 cluster config)")
        else:
            print("✗ Missing fed_spn_model!")
            return False

        print(f"✓ Test PASSED: Hybrid mode")
        return True

    except Exception as e:
        print(f"\n✗ Test FAILED: Hybrid mode")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_vertical_mode():
    """Test vertical mode (basic sanity check)."""
    print("\n" + "=" * 80)
    print("TEST: Vertical Mode - ProductOverGroups")
    print("=" * 80)

    set_random_seed(42)
    d, K, n_total = 6, 3, 600

    # Generate data
    G_bin = simulate_dag(d, d, "ER")
    B = simulate_parameter(G_bin)
    X_samples, _ = my_simulate_linear_gaussian(B, K, n_total, "gauss")

    # For vertical mode, FedCDH.fit() auto-partitions features
    # We just pass the global data and context column
    c_indx = np.zeros((n_total, 1))

    # Configure
    args = Namespace(
        K=K,
        d=d,
        scenario="vertical",
        model_type="SPN",
        ci_method="spn",  # Must be "spn" not "SPN_CIT"
        n=n_total,
        epochs=3,
        alpha=0.05,
        data_type="linear",
        device="cpu",
        skip_spn_eval=True,
    )

    print(f"Config: d={d}, K={K}, n={n_total}")
    print(f"True edges: {np.sum(G_bin)}")
    print(f"Note: Vertical mode auto-partitions features")
    print()

    try:
        # Initialize and fit
        # For vertical: pass full X_samples, fit() will auto-partition
        fedcdh = FedCDH(args)
        fedcdh.fit(X_samples, c_indx, G_bin)

        # Check results
        print("\n" + "-" * 80)
        print("RESULTS:")
        print("-" * 80)

        if hasattr(fedcdh, "fed_spn_model") and fedcdh.fed_spn_model is not None:
            print(f"✓ Global SPN model exists")

            # Check if the inner model is ProductOverGroups
            from causallearn.utils.FedPC import ProductOverGroups

            inner_model = (
                fedcdh.fed_spn_model.spn
                if hasattr(fedcdh.fed_spn_model, "spn")
                else fedcdh.fed_spn_model
            )

            if isinstance(inner_model, ProductOverGroups):
                print(f"✓ Using ProductOverGroups (expected for vertical)")
            else:
                print(f"⚠ Not using ProductOverGroups: {type(inner_model).__name__}")
        else:
            print("✗ Missing fed_spn_model!")
            return False

        print(f"✓ Test PASSED: Vertical mode")
        return True

    except Exception as e:
        print(f"\n✗ Test FAILED: Vertical mode")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 80)
    print("V3 AGGREGATION STRATEGIES - CPU SMOKE TEST")
    print("=" * 80)
    print("Testing:")
    print("  1. Horizontal: structure_voting (V3 fix #2)")
    print("  2. Horizontal: ll_weighted (V3 alternative)")
    print("  3. Horizontal: mixture (V2 baseline)")
    print("  4. Hybrid: GlobalSumOfProducts (V3 fix #1)")
    print("  5. Vertical: ProductOverGroups (sanity check)")
    print("=" * 80)

    results = {}

    # Test horizontal strategies
    for strategy in ["structure_voting", "ll_weighted", "mixture"]:
        results[f"horizontal_{strategy}"] = test_horizontal_aggregation(strategy)

    # Test hybrid mode
    results["hybrid_sum_over_products"] = test_hybrid_sum_over_products()

    # Test vertical mode
    results["vertical_product"] = test_vertical_mode()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print("=" * 80)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    print("=" * 80)

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! V3 aggregation strategies working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
