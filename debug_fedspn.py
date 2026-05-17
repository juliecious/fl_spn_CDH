#!/usr/bin/env python
"""
Debug script to understand why FedSPN methods return empty graphs.
"""

import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.insert(0, ".")

from causallearn.utils.data_utils import simulate_dag, simulate_parameter
from causallearn.utils.data_utils import my_simulate_linear_gaussian, set_random_seed
from causallearn.search.FCMBased.FedCDH import FedCDH

# Generate tiny synthetic dataset
set_random_seed(42)
d = 5
n = 200
s0 = 2

B = simulate_dag(d, s0, graph_type="ER")
W = simulate_parameter(B)
X, _ = my_simulate_linear_gaussian(W, K=1, n=n, sem_type="gauss")

print("=" * 80)
print("GROUND TRUTH DAG")
print("=" * 80)
print(B)
print(f"Edges: {int(np.sum(B))}")
print()

# Split into 3 clients (horizontal mode)
K = 3
n_per_client = n // K
X_splits = [X[i * n_per_client : (i + 1) * n_per_client, :] for i in range(K)]
c_indx = np.repeat(np.arange(K), n_per_client).reshape(-1, 1)

print("=" * 80)
print("TESTING FEDSPN-H")
print("=" * 80)

# Create FedCDH with SPN CI test
from argparse import Namespace

args = Namespace(
    K=K,
    d=d,
    n=n_per_client,
    scenario="horizontal",
    model_type="einet",
    ci_method="spn",
    data_type="linear",
    horizontal_aggregation="structure_voting",
    num_permutations=0,  # Use parametric test
)

fedcdh = FedCDH(args)

# Fit model
print("\n1. Training SPNs...")
G = fedcdh.fit(X_splits, c_indx, B)

print("\n" + "=" * 80)
print("PREDICTED DAG")
print("=" * 80)
# G is returned as a dict from FedCDH.fit()
print(f"Type of G: {type(G)}")
print(f"Keys: {list(G.keys()) if isinstance(G, dict) else 'N/A'}")

if isinstance(G, dict):
    G_matrix = G["G"]  # Extract the graph matrix
else:
    if hasattr(G, "G"):
        G_matrix = G.G.graph
    elif hasattr(G, "graph"):
        G_matrix = G.graph
    else:
        G_matrix = G

print(f"G_matrix shape: {G_matrix.shape}")
print("First d x d block:")
print(G_matrix[:d, :d])
print(f"Edges: {int(np.sum(np.abs(G_matrix[:d, :d]) > 0))}")

# Check skeleton accuracy
from causallearn.utils.data_utils import count_skeleton_accuracy

G_binary = (np.abs(G_matrix[:d, :d]) > 0).astype(int)
skeleton_stats = count_skeleton_accuracy(B, G_binary)
print(f"\nSkeleton F1: {skeleton_stats['f1']:.3f}")
print(f"Skeleton Precision: {skeleton_stats['precision']:.3f}")
print(f"Skeleton Recall: {skeleton_stats['recall']:.3f}")

# Also print full augmented graph to see context variable
print("\nFull Augmented Graph (including context var at index d):")
print(G_matrix)
print(f"Context variable index: {d}")

print("\n" + "=" * 80)
print("DIAGNOSTICS")
print("=" * 80)

# Check if SPNs are trained
print(f"Fed SPN model exists: {fedcdh.fed_spn_model is not None}")
print(f"Local SPNs count: {len(fedcdh.local_spns)}")

# Test a few CI queries manually
if fedcdh.fed_spn_model is not None:
    from causallearn.utils.cit import SPN_CIT

    # Augment data with context
    X_global = np.vstack(X_splits)
    c_indx_flat = c_indx.flatten()
    context = c_indx_flat.reshape(-1, 1)
    X_aug = np.hstack([X_global, context])

    cit = SPN_CIT(X_aug, global_model=fedcdh.fed_spn_model, num_permutations=0)

    print("\nManual CI tests:")
    print("-" * 80)

    # Test some pairs that SHOULD be independent
    test_cases = [
        ([0], [1], []),  # X0 vs X1 (marginal)
        ([0], [2], []),  # X0 vs X2 (marginal)
        ([0], [1], [2]),  # X0 vs X1 | X2 (conditional)
    ]

    for X, Y, Z in test_cases:
        p_val = cit(X, Y, Z)
        independent = p_val > 0.05
        print(f"X={X}, Y={Y}, Z={Z} -> p={p_val:.4f}, Independent={independent}")
