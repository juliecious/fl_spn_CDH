"""
Test suite for automatic feature grouping (Algorithm 1).

Tests build_feature_indicator_matrix() and group_features_by_client_set()
functions implementing Seng et al. (2025) Algorithm 1.
"""
import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from causallearn.utils.FedPC import (
    build_feature_indicator_matrix,
    group_features_by_client_set,
)


# ============================================================
# Test 1: build_feature_indicator_matrix
# ============================================================


def test_indicator_matrix_horizontal():
    """Test 1.1: Horizontal scenario - all clients have all features."""
    # Horizontal: All clients have same features
    X_splits = [
        np.random.randn(50, 5),  # Client 0: 50 samples, 5 features
        np.random.randn(50, 5),  # Client 1: 50 samples, 5 features
    ]

    M, feature_names = build_feature_indicator_matrix(X_splits, "horizontal")

    # Expected: All 1s (both clients have all features)
    expected_M = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

    assert M.shape == (2, 5), "Shape mismatch"
    assert np.array_equal(M, expected_M), (
        f"Horizontal: All clients should have all features\n"
        f"Got:\n{M}\n"
        f"Expected:\n{expected_M}"
    )
    assert feature_names == [0, 1, 2, 3, 4], "Feature names mismatch"

    print("✓ Test 1.1 passed: Horizontal indicator matrix")


def test_indicator_matrix_vertical():
    """Test 1.2: Vertical scenario - disjoint features."""
    # Vertical: Different features per client (same samples)
    X_splits = [
        np.random.randn(100, 3),  # Client 0: 100 samples, features [0,1,2]
        np.random.randn(100, 2),  # Client 1: 100 samples, features [3,4]
    ]

    M, feature_names = build_feature_indicator_matrix(
        X_splits, "vertical", d_features=5
    )

    # Expected: Disjoint features
    expected_M = np.array(
        [
            [1, 1, 1, 0, 0],  # Client 0 has first 3 features
            [0, 0, 0, 1, 1],  # Client 1 has last 2 features
        ]
    )

    assert M.shape == (2, 5), "Shape mismatch"
    assert np.array_equal(M, expected_M), (
        f"Vertical: Features should be disjoint\n"
        f"Got:\n{M}\n"
        f"Expected:\n{expected_M}"
    )

    print("✓ Test 1.2 passed: Vertical indicator matrix")


def test_indicator_matrix_hybrid():
    """Test 1.3: Hybrid scenario - uses equal split by default."""
    # Hybrid: Uses equal feature split (simplification)
    X_splits = [
        np.random.randn(50, 3),  # Client 0
        np.random.randn(60, 2),  # Client 1
    ]

    M, feature_names = build_feature_indicator_matrix(X_splits, "hybrid", d_features=5)

    # Expected: Equal split (conservative approach)
    expected_M = np.array(
        [[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]  # Client 0: [0,1,2]  # Client 1: [3,4]
    )

    assert M.shape == (2, 5), "Shape mismatch"
    assert np.array_equal(M, expected_M), (
        f"Hybrid: Should use equal split\n" f"Got:\n{M}"
    )

    print("✓ Test 1.3 passed: Hybrid indicator matrix")


def test_indicator_matrix_validation():
    """Test 1.4: Validation - d_features required for vertical/hybrid."""
    X_splits = [np.random.randn(100, 3)]

    # Should raise ValueError without d_features
    try:
        M, _ = build_feature_indicator_matrix(X_splits, "vertical")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "d_features required" in str(e)

    print("✓ Test 1.4 passed: Validation checks")


# ============================================================
# Test 2: group_features_by_client_set (Vertical - Disjoint)
# ============================================================


def test_grouping_vertical_disjoint():
    """Test 2.1: Vertical scenario produces disjoint groups."""
    # Vertical: Features split between clients
    M = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])

    groups = group_features_by_client_set(M, list(range(5)))

    # Expected: Two disjoint groups
    expected = {(0,): [0, 1, 2], (1,): [3, 4]}  # Client 0 only  # Client 1 only

    assert groups == expected, (
        f"Vertical grouping incorrect\n" f"Got: {groups}\n" f"Expected: {expected}"
    )

    # Verify disjoint: each feature appears exactly once
    all_features = [f for features in groups.values() for f in features]
    assert len(all_features) == len(set(all_features)), "Features overlap!"
    assert set(all_features) == set(range(5)), "Missing features!"

    print("✓ Test 2.1 passed: Vertical grouping (disjoint)")


# ============================================================
# Test 3: group_features_by_client_set (Horizontal - All shared)
# ============================================================


def test_grouping_horizontal_all_shared():
    """Test 3.1: Horizontal scenario - all features on all clients."""
    # Horizontal: All clients have all features
    M = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

    groups = group_features_by_client_set(M, list(range(5)))

    # Expected: Single group with all clients, all features
    expected = {(0, 1): [0, 1, 2, 3, 4]}  # Both clients have all features

    assert groups == expected, (
        f"Horizontal grouping incorrect\n" f"Got: {groups}\n" f"Expected: {expected}"
    )

    print("✓ Test 3.1 passed: Horizontal grouping (all shared)")


# ============================================================
# Test 4: group_features_by_client_set (Hybrid - Overlapping)
# ============================================================


def test_grouping_hybrid_overlapping():
    """
    Test 4.1: Hybrid scenario with overlapping features (Algorithm 1 pattern).

    Scenario (mimicking paper example):
        Client 0 has features [0, 1, 2]
        Client 1 has features [1, 2, 3]

        Indicator matrix:
            M[0, :] = [1, 1, 1, 0]
            M[1, :] = [0, 1, 1, 1]

        Expected grouping (Algorithm 1):
            Group A: [0] (only client 0)       → (0,): [0]
            Group B: [1, 2] (both clients)     → (0,1): [1, 2]
            Group C: [3] (only client 1)       → (1,): [3]

        This is EXACTLY the pattern from Seng et al. (2025) Algorithm 1.
    """
    # Hybrid with overlaps
    M = np.array(
        [[1, 1, 1, 0], [0, 1, 1, 1]]  # Client 0: [0, 1, 2]  # Client 1: [1, 2, 3]
    )

    groups = group_features_by_client_set(M, list(range(4)))

    # Expected: 3 groups
    expected = {
        (0,): [0],  # Feature 0 only on client 0
        (0, 1): [1, 2],  # Features 1,2 on BOTH clients (overlap resolved!)
        (1,): [3],  # Feature 3 only on client 1
    }

    assert groups == expected, (
        f"Hybrid grouping incorrect\n" f"Got: {groups}\n" f"Expected: {expected}"
    )

    # Verify each feature appears exactly once (no double-counting)
    all_features = [f for features in groups.values() for f in features]
    assert len(all_features) == len(
        set(all_features)
    ), "Features appear multiple times! Algorithm 1 should prevent this."
    assert set(all_features) == {0, 1, 2, 3}, "Missing features!"

    print("✓ Test 4.1 passed: Hybrid grouping (overlapping, Algorithm 1)")


def test_grouping_hybrid_3_clients():
    """Test 4.2: Hybrid with 3 clients and various overlaps."""
    # 3 clients with complex overlaps
    M = np.array(
        [
            [1, 1, 0, 0, 1],  # Client 0: [0, 1, 4]
            [0, 1, 1, 0, 1],  # Client 1: [1, 2, 4]
            [0, 0, 1, 1, 0],  # Client 2: [2, 3]
        ]
    )

    groups = group_features_by_client_set(M, list(range(5)))

    # Expected grouping by column pattern:
    # Feature 0: (1,0,0) → client_set=(0,)
    # Feature 1: (1,1,0) → client_set=(0,1)
    # Feature 2: (0,1,1) → client_set=(1,2)
    # Feature 3: (0,0,1) → client_set=(2,)
    # Feature 4: (1,1,0) → client_set=(0,1)
    expected = {
        (0,): [0],
        (0, 1): [1, 4],  # Features 1 and 4 share same client set
        (1, 2): [2],
        (2,): [3],
    }

    assert groups == expected, (
        f"3-client grouping incorrect\n" f"Got: {groups}\n" f"Expected: {expected}"
    )

    # Verify disjoint
    all_features = [f for features in groups.values() for f in features]
    assert len(all_features) == 5, "Missing features"
    assert len(all_features) == len(set(all_features)), "Features overlap!"

    print("✓ Test 4.2 passed: Hybrid with 3 clients")


# ============================================================
# Test 5: Integration - Indicator Matrix → Grouping Pipeline
# ============================================================


def test_integration_vertical_pipeline():
    """Test 5.1: Full pipeline - indicator matrix → grouping (vertical)."""
    # Generate vertical data
    X_splits = [
        np.random.randn(100, 3),  # Client 0: features [0,1,2]
        np.random.randn(100, 2),  # Client 1: features [3,4]
    ]

    # Step 1: Build indicator matrix
    M, feature_names = build_feature_indicator_matrix(
        X_splits, "vertical", d_features=5
    )

    # Step 2: Group features
    groups = group_features_by_client_set(M, feature_names)

    # Expected result
    expected_groups = {(0,): [0, 1, 2], (1,): [3, 4]}

    assert groups == expected_groups, "Vertical pipeline failed"
    print("✓ Test 5.1 passed: Vertical pipeline")


def test_integration_horizontal_pipeline():
    """Test 5.2: Full pipeline - indicator matrix → grouping (horizontal)."""
    # Generate horizontal data
    X_splits = [
        np.random.randn(50, 5),  # Client 0: all 5 features
        np.random.randn(60, 5),  # Client 1: all 5 features
    ]

    # Step 1: Build indicator matrix
    M, feature_names = build_feature_indicator_matrix(X_splits, "horizontal")

    # Step 2: Group features
    groups = group_features_by_client_set(M, feature_names)

    # Expected result
    expected_groups = {(0, 1): [0, 1, 2, 3, 4]}  # All features shared

    assert groups == expected_groups, "Horizontal pipeline failed"
    print("✓ Test 5.2 passed: Horizontal pipeline")


# ============================================================
# Run all tests
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print("AUTOMATIC FEATURE GROUPING TEST SUITE (Day 6-7)")
    print("=" * 70)

    # Test 1: Indicator Matrix Construction
    print("\n[Test 1] Indicator Matrix Construction")
    test_indicator_matrix_horizontal()
    test_indicator_matrix_vertical()
    test_indicator_matrix_hybrid()
    test_indicator_matrix_validation()

    # Test 2: Vertical Grouping (Disjoint)
    print("\n[Test 2] Vertical Grouping (Disjoint)")
    test_grouping_vertical_disjoint()

    # Test 3: Horizontal Grouping (All Shared)
    print("\n[Test 3] Horizontal Grouping (All Shared)")
    test_grouping_horizontal_all_shared()

    # Test 4: Hybrid Grouping (Overlapping)
    print("\n[Test 4] Hybrid Grouping (Overlapping)")
    test_grouping_hybrid_overlapping()
    test_grouping_hybrid_3_clients()

    # Test 5: Integration Tests
    print("\n[Test 5] Integration Tests")
    test_integration_vertical_pipeline()
    test_integration_horizontal_pipeline()

    print("\n" + "=" * 70)
    print("✓ ALL AUTOMATIC FEATURE GROUPING TESTS PASSED!")
    print("=" * 70)
    print("\nKey Achievement:")
    print("  ✓ Algorithm 1 from Seng et al. (2025) implemented")
    print("  ✓ Indicator matrix M construction working")
    print("  ✓ Feature grouping by client set working")
    print("  ✓ Handles horizontal, vertical, hybrid scenarios")
    print("  ✓ Overlapping features resolved correctly")
    print("=" * 70)
