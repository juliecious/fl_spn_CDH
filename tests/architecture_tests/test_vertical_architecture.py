#!/usr/bin/env python
"""
Smoke test for Seng's Vertical Mode Mixture-of-Products Architecture.

Tests that:
1. Each product node contains features from ALL clients
2. Multiple products are created (mixture-of-products)
3. Architecture matches Seng et al. (2025) Algorithm 1
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import logging
from argparse import Namespace
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

print("=" * 80)
print("SMOKE TEST: Seng's Vertical Mode Architecture")
print("=" * 80)
print()

# ============================================================================
# Step 1: Create synthetic dataset
# ============================================================================
print("[Step 1] Creating synthetic dataset...")

np.random.seed(42)
n_samples = 300
n_features = 12  # 12 features to split across 3 clients (4 per client, d/K=4 > 3)

# Create data with known dependencies
# Features: [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11]
# Client 0: [X0, X1, X2, X3]
# Client 1: [X4, X5, X6, X7]
# Client 2: [X8, X9, X10, X11]
# Cross-client dependencies: X0 -> X4, X1 -> X8, X4 -> X8

X = np.random.randn(n_samples, n_features)
# Add cross-client dependencies
X[:, 4] += 0.8 * X[:, 0]  # X0 (client 0) -> X4 (client 1)
X[:, 8] += (
    0.7 * X[:, 1] + 0.5 * X[:, 4]
)  # X1 (client 0) -> X8 (client 2), X4 (client 1) -> X8 (client 2)
# Add intra-client dependencies
X[:, 2] += 0.6 * X[:, 0]  # X0 -> X2 (both client 0)
X[:, 6] += 0.5 * X[:, 4]  # X4 -> X6 (both client 1)
X[:, 10] += 0.6 * X[:, 8]  # X8 -> X10 (both client 2)

# Ground truth adjacency matrix
B = np.zeros((n_features, n_features))
B[0, 4] = 1  # X0 (client 0) -> X4 (client 1) - CROSS-CLIENT
B[1, 8] = 1  # X1 (client 0) -> X8 (client 2) - CROSS-CLIENT
B[4, 8] = 1  # X4 (client 1) -> X8 (client 2) - CROSS-CLIENT
B[0, 2] = 1  # X0 -> X2 (within client 0)
B[4, 6] = 1  # X4 -> X6 (within client 1)
B[8, 10] = 1  # X8 -> X10 (within client 2)

print(f"  Dataset: {n_samples} samples × {n_features} features")
print(f"  Ground truth edges: {int(B.sum())}")
print(f"  CROSS-CLIENT dependencies:")
print(f"    - X0 (client 0) -> X4 (client 1)")
print(f"    - X1 (client 0) -> X8 (client 2)")
print(f"    - X4 (client 1) -> X8 (client 2)")
print(f"  INTRA-CLIENT dependencies:")
print(f"    - X0 -> X2 (within client 0)")
print(f"    - X4 -> X6 (within client 1)")
print(f"    - X8 -> X10 (within client 2)")
print()

# ============================================================================
# Step 2: Partition data vertically across 3 clients
# ============================================================================
print("[Step 2] Partitioning data vertically...")

K_clients = 3
feature_partition = [
    [0, 1, 2, 3],  # Client 0: X0, X1, X2, X3
    [4, 5, 6, 7],  # Client 1: X4, X5, X6, X7
    [8, 9, 10, 11],  # Client 2: X8, X9, X10, X11
]

X_splits = [X[:, features] for features in feature_partition]
feature_maps = feature_partition

print(f"  Clients: {K_clients}")
for k, features in enumerate(feature_partition):
    print(f"    Client {k}: features {features} (shape: {X_splits[k].shape})")
print()

# Create c_indx (context indices) - all zeros for vertical mode
c_indx = np.zeros((n_samples, 1), dtype=int)

# ============================================================================
# Step 3: Set up FedCDH for vertical mode
# ============================================================================
print("[Step 3] Setting up FedCDH (Vertical Mode)...")

from causallearn.search.FCMBased.FedCDH import FedCDH

args = Namespace(
    K=K_clients,
    d=n_features,
    n=n_samples,
    scenario="vertical",
    model_type="synthetic",
    ci_method="spn",
    alpha=0.05,
    epochs=10,  # Reduced for smoke test
    device="cpu",  # CPU for speed
    skip_bic=False,
    data_type="nonlinear",
    use_ci_ranking=False,
    force_num_clusters=2,
    num_local_clusters=2,  # K_local = 2 → 2^3 = 8 products
    skip_spn_eval=True,
    horizontal_aggregation="mixture",
    structure_vote_threshold=0.4,
    return_graphs=True,
    spn_eval_dir=None,
)

print(f"  Scenario: {args.scenario}")
print(f"  K_local: {args.num_local_clusters}")
print(f"  Expected products: {args.num_local_clusters ** K_clients} (K_local^K)")
print(f"  Device: {args.device}")
print()

# ============================================================================
# Step 4: Train FedSPN
# ============================================================================
print("[Step 4] Training FedSPN...")
print()

fedcdh = FedCDH(args)
results = fedcdh.fit(X_splits, c_indx, B)

print()
print("  ✓ Training completed")
print()

# ============================================================================
# Step 5: Inspect Architecture
# ============================================================================
print("=" * 80)
print("[Step 5] ARCHITECTURE INSPECTION")
print("=" * 80)
print()

fed_spn_wrapper = fedcdh.fed_spn_model
fed_spn = fed_spn_wrapper.spn  # Unwrap to get actual GlobalFedSPN

# Check type
print(f"Global SPN wrapper type: {type(fed_spn_wrapper).__name__}")
print(f"Actual SPN type: {type(fed_spn).__name__}")
print()

# Verify it's a GlobalSumOfProducts
from causallearn.utils.FedPC import (
    GlobalSumOfProducts,
    ProductOverGroups,
    ProductOverGroupsWithOverlap,
    GroupMixture,
)

if isinstance(fed_spn, GlobalSumOfProducts):
    print("✓ CORRECT: Global SPN is GlobalSumOfProducts (Seng's architecture)")
    print()

    # Inspect products
    num_products = len(fed_spn.products)
    print(f"Number of products: {num_products}")
    print(f"Product weights: {fed_spn.weights}")
    print()

    # Inspect each product
    print("Product Architecture:")
    print("-" * 80)

    for p_idx, product in enumerate(fed_spn.products):
        print(
            f"\nProduct {p_idx + 1}/{num_products} (weight={fed_spn.weights[p_idx]:.4f}):"
        )

        if isinstance(product, (ProductOverGroups, ProductOverGroupsWithOverlap)):
            product_type = (
                "ProductOverGroupsWithOverlap"
                if isinstance(product, ProductOverGroupsWithOverlap)
                else "ProductOverGroups"
            )
            print(f"  Type: {product_type} ✓")
            print(f"  Number of groups: {len(product.group_mixtures)}")

            # Check each group (should be one per client)
            for g_idx, group in enumerate(product.group_mixtures):
                if isinstance(group, GroupMixture):
                    features = group.feature_indices
                    print(f"    Group {g_idx}: features {features} (client {g_idx})")

                    # Verify features belong to the correct client
                    expected_features = feature_partition[g_idx]
                    if features == expected_features:
                        print(f"      ✓ Correct features for client {g_idx}")
                    else:
                        print(
                            f"      ✗ ERROR: Expected {expected_features}, got {features}"
                        )

            # Check that product spans all clients
            all_features = []
            for group in product.group_mixtures:
                all_features.extend(group.feature_indices)
            all_features = sorted(all_features)

            if all_features == list(range(n_features)):
                print(f"  ✓ Product spans all features: {all_features}")
            else:
                print(
                    f"  ✗ ERROR: Product missing features. Has {all_features}, expected {list(range(n_features))}"
                )
        else:
            print(
                f"  ✗ ERROR: Product is {type(product).__name__}, not ProductOverGroups"
            )

    print()
    print("-" * 80)
    print()

    # Summary check
    print("VERIFICATION SUMMARY:")
    print()

    checks = []

    # Check 1: Is GlobalSumOfProducts?
    checks.append(("GlobalSumOfProducts architecture", True))

    # Check 2: Correct number of products?
    expected_products = args.num_local_clusters**K_clients
    actual_products = len(fed_spn.products)
    checks.append(
        (
            f"Number of products ({actual_products} == {expected_products})",
            actual_products == expected_products,
        )
    )

    # Check 3: Each product has all clients?
    all_products_valid = True
    for product in fed_spn.products:
        if not isinstance(product, (ProductOverGroups, ProductOverGroupsWithOverlap)):
            all_products_valid = False
            break
        if len(product.group_mixtures) != K_clients:
            all_products_valid = False
            break
        # Check features span full space
        all_features = []
        for group in product.group_mixtures:
            all_features.extend(group.feature_indices)
        if sorted(all_features) != list(range(n_features)):
            all_products_valid = False
            break

    checks.append(("Each product spans all clients/features", all_products_valid))

    # Print summary
    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("=" * 80)
        print("✓✓✓ ALL CHECKS PASSED - Seng's architecture correctly implemented!")
        print("=" * 80)
    else:
        print("=" * 80)
        print("✗✗✗ SOME CHECKS FAILED - Architecture needs fixing")
        print("=" * 80)

else:
    print(f"✗ ERROR: Global SPN is {type(fed_spn).__name__}, not GlobalSumOfProducts")
    print(f"  This means the old architecture is still being used!")
    print(f"  Expected: Mixture of products (Seng's design)")
    print(f"  Got: Single product (forced independence)")

print()

# ============================================================================
# Step 6: Test Inference
# ============================================================================
print("=" * 80)
print("[Step 6] INFERENCE TEST")
print("=" * 80)
print()

# Test that inference works
print("Testing inference on a few samples...")
test_samples = 5
X_test = X[:test_samples]

try:
    # Convert to torch (use actual data, not NaN-masked)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Get log probabilities
    with torch.no_grad():
        log_probs = fed_spn.log_prob(X_test_tensor)

    print(f"  Log probabilities shape: {log_probs.shape}")
    print(f"  Log probabilities: {log_probs.cpu().numpy()}")
    print()
    print("  ✓ Inference successful!")

except Exception as e:
    print(f"  ✗ Inference failed: {e}")
    import traceback

    traceback.print_exc()

print()

# ============================================================================
# Step 7: Test CI Tests
# ============================================================================
print("=" * 80)
print("[Step 7] CONDITIONAL INDEPENDENCE TEST")
print("=" * 80)
print()

print("Testing cross-client edge detection...")
print()

# Test edge X0 -> X4 (client 0 -> client 1)
print("Test: X0 (client 0) ⊥? X4 (client 1) | {}")
print("  Ground truth: DEPENDENT (X0 -> X4)")
print()

from causallearn.utils.cit import CIT

ci_test = CIT(
    data=X,
    method="spn",
    global_model=fed_spn_wrapper,
    feature_map=fedcdh.vertical_feature_map
    if hasattr(fedcdh, "vertical_feature_map")
    else None,
    device="cpu",
    num_permutations=0,
)

# Test X0 ⊥? X4 | {}
try:
    p_value = ci_test(0, 4, [])
    print(f"  P-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  Decision: DEPENDENT (reject independence)")
        print(f"  ✓ CORRECT: Cross-client dependency detected!")
    else:
        print(f"  Decision: INDEPENDENT (fail to reject)")
        if p_value == 1.0:
            print(
                f"  ✗ ERROR: P-value = 1.0 suggests forced independence (old architecture)"
            )
        else:
            print(
                f"  ⚠ WARNING: Failed to detect dependency (may need more data/epochs)"
            )
except Exception as e:
    print(f"  ✗ CI test failed: {e}")
    import traceback

    traceback.print_exc()
    print()

print()
print("=" * 80)
print("SMOKE TEST COMPLETE")
print("=" * 80)
