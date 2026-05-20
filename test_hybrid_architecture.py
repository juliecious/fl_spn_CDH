#!/usr/bin/env python
"""
Smoke test for Hybrid Mode Architecture.

Tests that:
1. Multiple products are created (mixture-of-products)
2. Each product contains feature groups based on client overlap patterns
3. Architecture uses GlobalSumOfProducts (not forced independence)
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
print("SMOKE TEST: Hybrid Mode Architecture")
print("=" * 80)
print()

# ============================================================================
# Step 1: Create synthetic dataset
# ============================================================================
print("[Step 1] Creating synthetic dataset...")

np.random.seed(42)
n_samples = 300
n_features = 12  # Use 12 features (d/K=4 > 3 to avoid K_local=1 forcing)
n_clients = 3

# Create data with dependencies
X = np.random.randn(n_samples, n_features)
# Add some dependencies
X[:, 3] += 0.8 * X[:, 0]
X[:, 5] += 0.7 * X[:, 2]
X[:, 8] += 0.6 * X[:, 1]

print(f"  Dataset: {n_samples} samples × {n_features} features")
print(f"  Hybrid scenario: Each client has ALL features but different samples")
print()

# ============================================================================
# Step 2: Partition data in hybrid fashion
# ============================================================================
print("[Step 2] Partitioning data for hybrid mode...")

# For hybrid mode: split samples, not features
# Each client gets all features but different samples
samples_per_client = n_samples // n_clients
X_splits = [
    X[i * samples_per_client : (i + 1) * samples_per_client, :]
    for i in range(n_clients)
]

print(f"  Clients: {n_clients}")
print(f"  Samples per client: {samples_per_client}")
for k in range(n_clients):
    print(f"    Client {k}: {X_splits[k].shape} (all features, subset of samples)")
print()

# Ground truth adjacency matrix
B = np.zeros((n_features, n_features))
B[0, 3] = 1  # X0 -> X3
B[2, 5] = 1  # X2 -> X5
B[1, 8] = 1  # X1 -> X8

# Create c_indx (context indices) - one per client's samples
c_indx_splits = [
    np.full((samples_per_client, 1), k, dtype=int) for k in range(n_clients)
]
c_indx = np.concatenate(c_indx_splits, axis=0)

# ============================================================================
# Step 3: Set up FedCDH for hybrid mode
# ============================================================================
print("[Step 3] Setting up FedCDH (Hybrid Mode)...")

from causallearn.search.FCMBased.FedCDH import FedCDH

args = Namespace(
    K=n_clients,
    d=n_features,
    n=n_samples,
    scenario="hybrid",
    model_type="synthetic",
    ci_method="spn",
    alpha=0.05,
    epochs=10,
    device="cpu",
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
print(f"  Expected products: {args.num_local_clusters ** n_clients} (K_local^K)")
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
fed_spn = fed_spn_wrapper.spn  # Unwrap to get actual GlobalSumOfProducts

# Check type
print(f"Global SPN wrapper type: {type(fed_spn_wrapper).__name__}")
print(f"Actual SPN type: {type(fed_spn).__name__}")
print()

# Verify it's a GlobalSumOfProducts
from causallearn.utils.FedPC import (
    GlobalSumOfProducts,
    ProductOverGroupsWithOverlap,
    GroupMixture,
)

if isinstance(fed_spn, GlobalSumOfProducts):
    print("✓ CORRECT: Global SPN is GlobalSumOfProducts (mixture architecture)")
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

        if isinstance(product, ProductOverGroupsWithOverlap):
            print(f"  Type: ProductOverGroupsWithOverlap ✓")
            print(f"  Number of groups: {len(product.group_mixtures)}")

            # Check each group
            all_features_in_product = []
            for g_idx, group in enumerate(product.group_mixtures):
                if isinstance(group, GroupMixture):
                    features = group.feature_indices
                    print(f"    Group {g_idx}: features {features}")
                    all_features_in_product.extend(features)

            # Check coverage
            unique_features = sorted(set(all_features_in_product))
            print(f"  Features covered: {unique_features}")

            if unique_features == list(range(n_features)):
                print(f"  ✓ Product covers all features")
            else:
                print(f"  ⚠ Product covers subset: {unique_features}")
        else:
            print(
                f"  ✗ ERROR: Product is {type(product).__name__}, not ProductOverGroupsWithOverlap"
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
    expected_products = args.num_local_clusters**n_clients
    actual_products = len(fed_spn.products)
    checks.append(
        (
            f"Number of products ({actual_products} == {expected_products})",
            actual_products == expected_products,
        )
    )

    # Check 3: Each product uses ProductOverGroupsWithOverlap?
    all_products_valid = True
    for product in fed_spn.products:
        if not isinstance(product, ProductOverGroupsWithOverlap):
            all_products_valid = False
            break

    checks.append(("Each product is ProductOverGroupsWithOverlap", all_products_valid))

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
        print("✓✓✓ ALL CHECKS PASSED - Hybrid architecture correctly implemented!")
        print("=" * 80)
    else:
        print("=" * 80)
        print("✗✗✗ SOME CHECKS FAILED - Architecture needs fixing")
        print("=" * 80)

else:
    print(f"✗ ERROR: Global SPN is {type(fed_spn).__name__}, not GlobalSumOfProducts")
    print(f"  This means the mixture architecture is not being used!")

print()

# ============================================================================
# Step 6: Test Inference
# ============================================================================
print("=" * 80)
print("[Step 6] INFERENCE TEST")
print("=" * 80)
print()

print("Testing inference on a few samples...")
test_samples = 5
X_test = X[:test_samples]

try:
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

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
print("=" * 80)
print("SMOKE TEST COMPLETE")
print("=" * 80)
