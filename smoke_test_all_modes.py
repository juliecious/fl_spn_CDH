#!/usr/bin/env python
"""
Tiny smoke test for all FedCDH modes (Horizontal, Vertical, Hybrid).
Tests both old and new implementations (with Gap 3 & 4).
"""

import sys
import os
import time
import numpy as np
from argparse import Namespace

# Add project root
sys.path.insert(0, os.path.dirname(__file__))

from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    simulate_dag,
    simulate_parameter,
    count_skeleton_accuracy,
    count_dag_accuracy,
)

print("=" * 70)
print("FedCDH All Modes Smoke Test (Tiny Scale)")
print("=" * 70)

# Tiny test parameters
d = 5  # 5 nodes
n_total = 150  # 150 samples
K = 3  # 3 clients
seed = 42

# Generate tiny DAG
np.random.seed(seed)
B_true = simulate_dag(d=d, s0=4, graph_type="ER")
W = simulate_parameter(B_true)
X_all = np.random.randn(n_total, d) @ W.T

print(f"\nDataset: {d} nodes, {n_total} samples, {int(B_true.sum())} edges")


def run_test(name, args, X_splits=None, feature_maps=None):
    """Run a single FedCDH test."""
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")

    start = time.time()
    try:
        # Initialize
        model = FedCDH(args, feature_maps=feature_maps)
        print(f"  ✓ Initialized ({args.scenario} mode)")

        # Fit
        if X_splits is None:
            # Horizontal: split rows
            n_per_client = n_total // K
            X_splits = [
                X_all[i * n_per_client : (i + 1) * n_per_client] for i in range(K)
            ]

        result = model.fit(X_splits, B_true)

        # Metrics
        skel_metrics = count_skeleton_accuracy(B_true, result["skeleton"])
        dag_metrics = count_dag_accuracy(B_true, result["graph"])

        elapsed = time.time() - start

        print(f"  ✓ Completed in {elapsed:.1f}s")
        print(f"  Skeleton F1: {skel_metrics['f1']:.3f}")
        print(f"  DAG F1: {dag_metrics['f1']:.3f}")
        print(f"  Status: {'✓ PASS' if skel_metrics['f1'] > 0.3 else '⚠ LOW'}")

        return True

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ FAILED after {elapsed:.1f}s")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return False


# Test 1: Horizontal mode (baseline)
args_h = Namespace(
    K=K,
    d=d,
    n=n_total // K,
    scenario="horizontal",
    model_type="linear",
    ci_method="spn",
    device="cpu",
    verbose=False,
    auto_structure=False,
    use_cluster_conditional=False,
)
pass_h = run_test("Test 1: Horizontal Mode (Baseline)", args_h)

# Test 2: Vertical mode (baseline)
print("\nPreparing vertical partitioning...")
feature_maps_v = {
    0: [0, 1],  # Client 0: features 0,1
    1: [2, 3],  # Client 1: features 2,3
    2: [4],  # Client 2: feature 4
}
X_splits_v = [X_all[:, feature_maps_v[k]] for k in range(K)]

args_v = Namespace(
    K=K,
    d=d,
    n=n_total,
    scenario="vertical",
    model_type="linear",
    ci_method="spn",
    device="cpu",
    verbose=False,
    auto_structure=False,
    use_cluster_conditional=False,
)
pass_v = run_test(
    "Test 2: Vertical Mode (Baseline)", args_v, X_splits_v, feature_maps_v
)

# Test 3: Vertical with Gap 4 (cluster-conditional)
args_v_gap4 = Namespace(
    K=K,
    d=d,
    n=n_total,
    scenario="vertical",
    model_type="linear",
    ci_method="spn",
    device="cpu",
    verbose=False,
    auto_structure=False,
    use_cluster_conditional=True,  # Gap 4: ON
)
pass_v_gap4 = run_test(
    "Test 3: Vertical Mode + Gap 4 (Cluster-Conditional)",
    args_v_gap4,
    X_splits_v,
    feature_maps_v,
)

# Test 4: Hybrid mode
print("\nPreparing hybrid partitioning...")
# Hybrid: different samples AND features
n1, n2, n3 = 50, 50, 50
X_splits_hy = [
    X_all[:n1, :3],  # Client 0: samples 0-50, features 0-2
    X_all[n1 : n1 + n2, 1:4],  # Client 1: samples 50-100, features 1-3 (overlap!)
    X_all[n1 + n2 :, 3:],  # Client 2: samples 100-150, features 3-4
]
feature_maps_hy = {
    0: [0, 1, 2],
    1: [1, 2, 3],  # Overlap with client 0
    2: [3, 4],
}

args_hy = Namespace(
    K=K,
    d=d,
    n=n1,  # Average per client
    scenario="hybrid",
    model_type="linear",
    ci_method="spn",
    device="cpu",
    verbose=False,
    auto_structure=False,
    use_cluster_conditional=False,
)
pass_hy = run_test(
    "Test 4: Hybrid Mode (Baseline)", args_hy, X_splits_hy, feature_maps_hy
)

# Test 5: Auto-structure detection (Gap 3)
args_auto = Namespace(
    K=K,
    d=d,
    n=n_total,
    scenario="vertical",  # Will be auto-detected
    model_type="linear",
    ci_method="spn",
    device="cpu",
    verbose=True,
    auto_structure=True,  # Gap 3: ON
    use_cluster_conditional=False,
)
print(f"\n{'=' * 70}")
print(f"Test 5: Auto-Structure Detection (Gap 3)")
print(f"{'=' * 70}")
print("NOTE: This test only validates initialization (full fit is slow)")
try:
    model_auto = FedCDH(args_auto, feature_maps=feature_maps_v)
    print(f"  ✓ Initialized with auto_structure=True")
    print(f"  Status: ✓ PASS (initialization only)")
    pass_auto = True
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    pass_auto = False

# Summary
print(f"\n{'=' * 70}")
print(f"Summary")
print(f"{'=' * 70}")
results = {
    "Horizontal": pass_h,
    "Vertical": pass_v,
    "Vertical + Gap 4": pass_v_gap4,
    "Hybrid": pass_hy,
    "Auto-Structure (Gap 3)": pass_auto,
}

for name, passed in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status:8} - {name}")

all_pass = all(results.values())
print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
print("\nNote: Low F1 scores expected on tiny dataset (150 samples, 5 nodes)")
print("      Integration test - validates code runs without errors")

sys.exit(0 if all_pass else 1)
