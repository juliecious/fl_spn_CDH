"""
Smoke Test for v2 Implementation (Top-N% Ranking + Adaptive Hyperparameters)

Verifies that:
1. Adaptive hyperparameters are computed correctly
2. CI ranking tracker can be initialized and used
3. FedCDH can use adaptive hyperparameters
4. Basic functionality works end-to-end

This is a minimal smoke test, not a full experiment.
"""

import sys
import os

# Ensure we import from the local codebase
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import logging
from types import SimpleNamespace

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def test_adaptive_hyperparameters():
    """Test that adaptive hyperparameters module works."""
    from causallearn.utils.FedPC import compute_adaptive_hyperparameters

    print("\n" + "=" * 60)
    print("TEST 1: Adaptive Hyperparameters")
    print("=" * 60)

    # Test horizontal mode (known failure case: F1=0.255)
    params = compute_adaptive_hyperparameters(
        mode="horizontal", num_features=10, num_samples=400, data_type="linear"
    )

    print(f"✓ Horizontal mode (d=10, n=400):")
    print(
        f"  Architecture: {params['num_sums']} sums, {params['num_leaves']} leaves, depth={params['depth']}"
    )
    print(f"  Training: {params['epochs']} epochs, dropout={params['dropout']:.2f}")
    print(f"  Regularization: weight_decay={params['weight_decay']:.1e}")

    # Verify all keys present
    required_keys = [
        "num_sums",
        "num_leaves",
        "depth",
        "epochs",
        "dropout",
        "weight_decay",
    ]
    for key in required_keys:
        assert key in params, f"Missing key: {key}"

    # Verify positive values
    assert params["num_sums"] > 0
    assert params["num_leaves"] > 0
    assert params["depth"] > 0
    assert params["epochs"] > 0

    print("✅ Adaptive hyperparameters test passed!")
    return True


def test_ci_ranking_tracker():
    """Test that CI ranking tracker works."""
    from causallearn.utils.ci_ranking import CIRankingTracker

    print("\n" + "=" * 60)
    print("TEST 2: CI Ranking Tracker")
    print("=" * 60)

    tracker = CIRankingTracker(sparsity_percentile=0.2)
    print(f"✓ Tracker initialized with sparsity={tracker.sparsity_percentile}")

    # Add synthetic CI test results
    for i in range(20):
        tracker.add_result(
            x=0,
            y=i + 1,
            S=(),
            cmi_score=np.random.rand(),
            p_value=0.05,
            stat_obs=np.random.rand() * 10,
            depth=0,
            test_index=i,
        )

    print(f"✓ Added {len(tracker.results)} test results")

    # Compute threshold
    threshold = tracker.compute_threshold()
    print(f"✓ Threshold computed: {threshold:.4f}")

    # Get statistics
    stats = tracker.get_statistics()
    print(
        f"✓ Statistics: {stats['num_dependent']} dependent, {stats['num_independent']} independent"
    )

    # Verify correct split
    expected_dependent = int(20 * 0.2)
    assert (
        stats["num_dependent"] == expected_dependent
    ), f"Expected {expected_dependent} dependent, got {stats['num_dependent']}"

    print("✅ CI ranking tracker test passed!")
    return True


def test_fedcdh_with_adaptive_params():
    """Test that FedCDH can use adaptive hyperparameters."""
    from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH

    print("\n" + "=" * 60)
    print("TEST 3: FedCDH with Adaptive Parameters")
    print("=" * 60)

    # Create minimal synthetic data
    np.random.seed(42)
    n_samples = 300
    d_features = 5
    K_clients = 2

    # Generate random data
    data = np.random.randn(n_samples, d_features)

    # Split into clients (horizontal)
    samples_per_client = n_samples // K_clients
    X_splits = [
        data[k * samples_per_client : (k + 1) * samples_per_client, :]
        for k in range(K_clients)
    ]

    print(
        f"✓ Synthetic data: {n_samples} samples, {d_features} features, {K_clients} clients"
    )

    # Create FedCDH args
    args = SimpleNamespace(
        K=K_clients,
        d=d_features,
        n=samples_per_client,
        scenario="horizontal",
        model_type="spn",
        ci_method="spn",
        data_type="linear",  # NEW: for adaptive hyperparameters
        device="cpu",
    )

    print(
        f"✓ FedCDH args created (scenario={args.scenario}, data_type={args.data_type})"
    )

    # Initialize FedCDH
    try:
        fedcdh = FedCDH(args)
        print(f"✓ FedCDH initialized successfully")
        print(f"  Device: {fedcdh.device}")
        print(f"  Scenario: {fedcdh.scenario}")
        print(f"  Data type: {fedcdh.data_type}")
    except Exception as e:
        print(f"❌ FedCDH initialization failed: {e}")
        raise

    print("✅ FedCDH integration test passed!")
    return True


def test_spn_cit_with_ranking():
    """Test that SPN_CIT can use ranking mode (simplified test)."""
    from causallearn.utils.ci_ranking import CIRankingTracker

    print("\n" + "=" * 60)
    print("TEST 4: SPN_CIT with Ranking Mode")
    print("=" * 60)

    # For this smoke test, we just verify that the ranking tracker
    # can be integrated with SPN_CIT's parameters.
    # Full integration testing requires a complete FedCDH run.

    # Create ranking tracker
    tracker = CIRankingTracker(sparsity_percentile=0.3)
    print("✓ Ranking tracker created")

    # Simulate CI test results (as would be collected by SPN_CIT)
    for i in range(10):
        tracker.add_result(
            x=0,
            y=i + 1,
            S=(),
            cmi_score=np.random.rand(),
            p_value=np.random.rand(),
            stat_obs=np.random.rand() * 10,
            depth=0,
            test_index=i,
        )

    print(f"✓ Simulated {len(tracker.results)} CI test results")

    # Compute threshold (as would happen in skeleton discovery)
    threshold = tracker.compute_threshold()
    stats = tracker.get_statistics()

    print(f"✓ Threshold: {threshold:.4f}")
    print(
        f"✓ Dependent: {stats['num_dependent']}, Independent: {stats['num_independent']}"
    )

    # Verify the split is correct
    expected_dependent = int(10 * 0.3)
    assert (
        stats["num_dependent"] == expected_dependent
    ), f"Expected {expected_dependent} dependent, got {stats['num_dependent']}"

    print("✅ SPN_CIT ranking test passed!")
    print("   (Note: Full SPN_CIT integration requires complete FedCDH run)")
    return True


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("v2 IMPLEMENTATION SMOKE TESTS")
    print("=" * 60)
    print("Testing: Top-N% Ranking + Adaptive Hyperparameters")

    results = []

    try:
        results.append(("Adaptive Hyperparameters", test_adaptive_hyperparameters()))
    except Exception as e:
        print(f"❌ Adaptive Hyperparameters test failed: {e}")
        results.append(("Adaptive Hyperparameters", False))

    try:
        results.append(("CI Ranking Tracker", test_ci_ranking_tracker()))
    except Exception as e:
        print(f"❌ CI Ranking Tracker test failed: {e}")
        results.append(("CI Ranking Tracker", False))

    try:
        results.append(("FedCDH Integration", test_fedcdh_with_adaptive_params()))
    except Exception as e:
        print(f"❌ FedCDH Integration test failed: {e}")
        results.append(("FedCDH Integration", False))

    try:
        results.append(("SPN_CIT Ranking", test_spn_cit_with_ranking()))
    except Exception as e:
        print(f"❌ SPN_CIT Ranking test failed: {e}")
        results.append(("SPN_CIT Ranking", False))

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All smoke tests passed! v2 implementation ready for experiments.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
