#!/usr/bin/env python
"""
Tiny smoke test to verify Vertical and Hybrid architecture details.

Tests:
1. How features are split across product nodes
2. How they're aggregated in sum nodes
3. Differences between Vertical and Hybrid modes
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from argparse import Namespace
import torch

print("=" * 80)
print("TINY ARCHITECTURE TEST: Vertical vs Hybrid")
print("=" * 80)
print()

# ============================================================================
# Setup: Minimal dataset
# ============================================================================
np.random.seed(42)
n_samples = 150  # Divisible by 3 for hybrid
n_features = 12  # 4 per client (d/K=4 > 3, allows K_local=2)
n_clients = 3

X = np.random.randn(n_samples, n_features)

print(f"Dataset: {n_samples} samples × {n_features} features")
print(f"Clients: {n_clients}")
print(
    f"Features per client: {n_features // n_clients} (d/K={n_features/n_clients:.1f})"
)
print()


def test_mode(scenario_name, X_splits, c_indx):
    """Test a specific mode and print architecture details."""

    print("=" * 80)
    print(f"TESTING: {scenario_name.upper()} MODE")
    print("=" * 80)
    print()

    # Show data splits
    print(f"Data splits:")
    for k, split in enumerate(X_splits):
        print(f"  Client {k}: shape {split.shape}")
    print()

    # Setup FedCDH
    from causallearn.search.FCMBased.FedCDH import FedCDH

    args = Namespace(
        K=n_clients,
        d=n_features,
        n=n_samples,
        scenario=scenario_name,
        model_type="synthetic",
        ci_method="spn",
        alpha=0.05,
        epochs=5,  # Very fast
        device="cpu",
        skip_bic=False,
        data_type="nonlinear",
        use_ci_ranking=False,
        force_num_clusters=2,
        num_local_clusters=2,  # K_local=2 → 2^3=8 products
        skip_spn_eval=True,
        horizontal_aggregation="mixture",
        return_graphs=False,  # Skip CD
        spn_eval_dir=None,
    )

    fedcdh = FedCDH(args)

    # Train (will build architecture)
    B = np.zeros((n_features, n_features))
    results = fedcdh.fit(X_splits, c_indx, B)

    # Inspect architecture
    fed_spn_wrapper = fedcdh.fed_spn_model
    fed_spn = fed_spn_wrapper.spn

    from causallearn.utils.FedPC import (
        GlobalSumOfProducts,
        ProductOverGroupsWithOverlap,
        GroupMixture,
    )

    print("\n" + "=" * 80)
    print("ARCHITECTURE DETAILS")
    print("=" * 80)
    print()

    if not isinstance(fed_spn, GlobalSumOfProducts):
        print(f"✗ ERROR: Expected GlobalSumOfProducts, got {type(fed_spn).__name__}")
        return

    print(f"✓ GlobalSumOfProducts: {len(fed_spn.products)} products")
    print(f"  Product weights: {fed_spn.weights}")
    print()

    # Inspect first few products in detail
    num_to_inspect = min(3, len(fed_spn.products))

    for p_idx in range(num_to_inspect):
        product = fed_spn.products[p_idx]
        weight = fed_spn.weights[p_idx]

        print(f"Product {p_idx + 1} (weight={weight:.4f}):")
        print(f"  Type: {type(product).__name__}")
        print(f"  Number of groups: {len(product.group_mixtures)}")
        print()

        for g_idx, group in enumerate(product.group_mixtures):
            if isinstance(group, GroupMixture):
                features = group.feature_indices
                num_spns = len(group.client_spns)
                weights = (
                    group.weights.cpu().numpy()
                    if hasattr(group.weights, "cpu")
                    else group.weights
                )

                print(f"    Group {g_idx}:")
                print(f"      Features: {features}")
                print(f"      Num SPNs: {num_spns}")
                print(f"      SPN weights: {weights}")
                print(f"      Use NaN masking: {group.use_nan_masking}")
                print(f"      Full_d: {group.full_d}")

                if num_spns > 1:
                    print(
                        f"      → MIXTURE: Averages {num_spns} client SPNs (horizontal aggregation)"
                    )
                else:
                    print(f"      → SINGLE SPN: From one client")
                print()

        print()

    if len(fed_spn.products) > num_to_inspect:
        print(f"... ({len(fed_spn.products) - num_to_inspect} more products)")
        print()

    # Test inference
    print("=" * 80)
    print("INFERENCE TEST")
    print("=" * 80)
    print()

    X_test = torch.tensor(X[:5], dtype=torch.float32)
    with torch.no_grad():
        log_probs = fed_spn.log_prob(X_test)

    print(f"✓ Inference successful: log_probs shape = {log_probs.shape}")
    print(f"  Log probabilities: {log_probs.squeeze().cpu().numpy()}")
    print()


# ============================================================================
# Test 1: VERTICAL MODE
# ============================================================================

# Vertical: Features split, all samples shared
X_splits_vertical = [
    X[:, [0, 1, 2, 3]],  # Client 0: features 0-3
    X[:, [4, 5, 6, 7]],  # Client 1: features 4-7
    X[:, [8, 9, 10, 11]],  # Client 2: features 8-11
]

c_indx_vertical = np.zeros((n_samples, 1), dtype=int)

test_mode("vertical", X_splits_vertical, c_indx_vertical)

# ============================================================================
# Test 2: HYBRID MODE
# ============================================================================

# Hybrid: Samples split, features can overlap
samples_per_client = n_samples // n_clients

X_splits_hybrid = [
    X[
        0 * samples_per_client : 1 * samples_per_client, :
    ],  # Client 0: samples 0-33, all features
    X[
        1 * samples_per_client : 2 * samples_per_client, :
    ],  # Client 1: samples 34-66, all features
    X[
        2 * samples_per_client : 3 * samples_per_client, :
    ],  # Client 2: samples 67-99, all features
]

c_indx_hybrid = np.concatenate(
    [
        np.full((samples_per_client, 1), 0, dtype=int),
        np.full((samples_per_client, 1), 1, dtype=int),
        np.full((samples_per_client, 1), 2, dtype=int),
    ]
)

test_mode("hybrid", X_splits_hybrid, c_indx_hybrid)

# ============================================================================
# Summary Comparison
# ============================================================================

print("=" * 80)
print("SUMMARY: Vertical vs Hybrid")
print("=" * 80)
print()
print("VERTICAL MODE:")
print("  - Features split across clients: [0-3], [4-7], [8-11]")
print("  - All clients see ALL samples")
print("  - Each group has 1 SPN (no horizontal mixing)")
print("  - Use NaN masking: False (feature extraction)")
print()
print("HYBRID MODE:")
print("  - All features on all clients")
print("  - Samples split across clients")
print("  - Groups can have multiple SPNs (horizontal mixing)")
print("  - Use NaN masking: True (for marginalization)")
print()
print("UNIFIED ARCHITECTURE:")
print("  - Both use GlobalSumOfProducts")
print("  - Both use ProductOverGroupsWithOverlap")
print("  - Both use GroupMixture for feature groups")
print("  - Vertical is just hybrid with no overlapping features!")
print()
