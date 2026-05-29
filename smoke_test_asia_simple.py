#!/usr/bin/env python
"""
Quick Asia smoke test for FedSPN-H (no experimental features, just validate working).
"""

import sys
import os
import time
import numpy as np
from argparse import Namespace

sys.path.insert(0, os.path.dirname(__file__))

from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    simulate_dag,
    simulate_parameter,
    count_skeleton_accuracy,
    count_dag_accuracy,
)

print("=" * 70)
print("Asia Tiny Smoke Test - FedSPN-H (Baseline)")
print("=" * 70)

# Tiny test
d = 8
n_total = 300
K = 3

np.random.seed(42)
B_true = simulate_dag(d=d, s0=8, graph_type="ER")
W = simulate_parameter(B_true)
X_all = np.random.randn(n_total, d) @ W.T

print(f"\nDataset: {d} nodes, {n_total} samples, {int(B_true.sum())} edges")
print(f"Clients: {K} ({n_total//K} samples each)")
print(f"Device: CPU")

# Args
args = Namespace(
    K=K,
    d=d,
    n=n_total // K,
    scenario="horizontal",
    model_type="linear",
    ci_method="spn",
    device="cpu",
    verbose=False,
    epochs=15,  # Fast
    num_sums=15,
    num_leaves=15,
    depth=2,
)

print(f"\n{'=' * 70}")
print(f"Running FedSPN-H")
print(f"{'=' * 70}\n")

start = time.time()

try:
    # Init
    model = FedCDH(args)

    # Split data
    n_per_client = n_total // K
    X_splits = [X_all[i * n_per_client : (i + 1) * n_per_client] for i in range(K)]
    c_indx = np.repeat(np.arange(K), n_per_client).reshape(-1, 1)

    # Fit
    result = model.fit(X_splits, c_indx=c_indx, true_DAG_bin=B_true)

    elapsed = time.time() - start

    # Results
    skel_metrics = count_skeleton_accuracy(B_true, result["skeleton"])
    dag_metrics = count_dag_accuracy(B_true, result["graph"])

    print(f"\n{'=' * 70}")
    print(f"Results")
    print(f"{'=' * 70}")
    print(f"\nSkeleton F1: {skel_metrics['f1']:.3f}")
    print(f"DAG F1: {dag_metrics['f1']:.3f}")
    print(f"Runtime: {elapsed:.1f}s")

    success = skel_metrics["f1"] > 0.1
    print(f"\nStatus: {'✓ PASS' if success else '⚠ LOW'}")

    sys.exit(0)

except Exception as e:
    elapsed = time.time() - start
    print(f"\n✗ FAILED after {elapsed:.1f}s: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
