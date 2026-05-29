#!/usr/bin/env python
"""
Minimal smoke test - 100 samples, 5 nodes, 2 clients, fast settings.
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

print("=" * 60)
print("MINIMAL SMOKE TEST - FedSPN-V (Vertical)")
print("=" * 60)

# Minimal parameters
d = 5
n_total = 100
K = 2

np.random.seed(42)
B_true = simulate_dag(d=d, s0=4, graph_type="ER")
W = simulate_parameter(B_true)
X_all = np.random.randn(n_total, d) @ W.T

print(f"\nDataset: {d} nodes, {n_total} samples, {int(B_true.sum())} edges")
print(f"Clients: {K} (vertical partitioning - different features)")

# Vertical mode: partition features across clients
feature_maps_v = {
    0: [0, 1],  # Client 0: features 0, 1
    1: [2, 3, 4],  # Client 1: features 2, 3, 4
}
print(f"Feature distribution:")
for client_id, features in feature_maps_v.items():
    print(f"  Client {client_id}: features {features}")

# Minimal args - fast settings
args = Namespace(
    K=K,
    d=d,
    n=n_total,  # Vertical: all clients see all samples
    scenario="vertical",
    model_type="linear",
    ci_method="spn",
    device="cpu",
    verbose=True,  # Enable verbose for debugging
    # FAST settings
    epochs=10,
    num_sums=10,
    num_leaves=10,
    depth=2,
    alpha=0.1,  # More lenient
    # Gap 4: Try cluster-conditional vertical
    use_cluster_conditional=True,
)

print(f"\n{'=' * 60}")
print(f"Running FedSPN-V (FAST mode)")
print(f"{'=' * 60}")

start = time.time()

try:
    model = FedCDH(args, feature_maps=feature_maps_v)

    # Vertical: split columns (features), all clients see all samples
    X_splits = [X_all[:, feature_maps_v[k]] for k in range(K)]

    # c_indx for vertical: all samples belong to all clients (or None)
    # Use client index 0 for all samples since it's vertical partitioning
    c_indx = np.zeros((n_total, 1), dtype=int)

    print(f"\n✓ Initialized")
    print(f"✓ Data split: {[x.shape for x in X_splits]}")
    print(f"\nRunning fit()...")

    result = model.fit(X_splits, c_indx=c_indx, true_DAG_bin=B_true)

    elapsed = time.time() - start

    # Metrics are already computed in result
    skel_f1 = result.get("f1_skeleton", 0.0) or 0.0
    dag_f1 = result.get("f1", 0.0) or 0.0
    time_train = result.get("time_train", 0.0) or 0.0
    time_cd = result.get("time_cd", 0.0) or 0.0

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Skeleton F1: {skel_f1:.3f}")
    print(f"DAG F1: {dag_f1:.3f}")
    print(f"Total Runtime: {elapsed:.1f}s")
    print(f"  - Training: {time_train:.1f}s")
    print(f"  - Causal Discovery: {time_cd:.1f}s")
    print(f"\n{'=' * 60}")
    print(f"✓ TEST PASSED - System Working!")
    print(f"{'=' * 60}")

    sys.exit(0)

except Exception as e:
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"✗ FAILED after {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
