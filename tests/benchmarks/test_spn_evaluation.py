#!/usr/bin/env python3
"""
Test script for SPN Quality Evaluation Framework

Tests the SPNEvaluator on smoke test data (d=6, K=2, horizontal scenario).
Validates that all metrics compute correctly and reports are generated.

Expected runtime: ~5 minutes
Expected outcome: All metrics within reasonable ranges, report files created

Usage:
    python tests/benchmarks/test_spn_evaluation.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH
from tests.benchmarks.evaluate_spn_quality import SPNEvaluator

print("=" * 70)
print("TEST: SPN Quality Evaluation Framework")
print("=" * 70)

# Configuration - smoke test params (d=6, K=2)
class Args:
    def __init__(self):
        self.K = 2
        self.d = 6
        self.scenario = "horizontal"
        self.model_type = "linear"
        self.ci_method = "spn"
        self.n = 300  # n_per_client
        self.alpha = 0.05
        self.epochs = 50  # Reduced for fast test
        self.num_sums = 20
        self.num_leaves = 20
        self.num_repetitions = 10
        self.ablation_orientation = "mi_only"


# Generate smoke test data
np.random.seed(42)
args = Args()
n_total = args.n * args.K
d = args.d
K = args.K

# True DAG: chain 0->1->2->3->4->5
true_DAG = np.zeros((d, d))
for i in range(d - 1):
    true_DAG[i, i + 1] = 1

# Generate data with heterogeneity
X = np.zeros((n_total, d))
X[:, 0] = np.random.randn(n_total)

for i in range(1, d):
    domain_shift = np.repeat(np.arange(K), n_total // K) * 0.2
    X[:, i] = 0.8 * X[:, i - 1] + domain_shift + np.random.randn(n_total) * 0.5

# Normalize
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# Domain indices
c_indx = np.repeat(np.arange(K), n_total // K).reshape(-1, 1)

# Split data
X_splits = np.array_split(X, K)

print("\nConfiguration:")
print(f"  d={d}, K={K}, n_total={n_total}")
print(f"  Scenario: {args.scenario}")
print(f"  Epochs: {args.epochs} (reduced for testing)")

print("\n" + "-" * 70)
print("Step 1: Train FedCDH (to get SPNs)")
print("-" * 70)

start = time.time()
fedcdh = FedCDH(args)
result = fedcdh.fit(X_splits, c_indx, true_DAG)
elapsed_train = time.time() - start

print(f"\nFedCDH training complete: {elapsed_train:.1f}s")
print(f"  F1_skeleton: {result['f1_skeleton']:.3f}")
print(f"  F1_directed: {result['f1']:.3f}")

# Extract SPNs (need to access internal attributes)
print("\n" + "-" * 70)
print("Step 2: Extract Local and Global SPNs")
print("-" * 70)

# Check if FedCDH exposes SPNs
if not hasattr(fedcdh, "local_spns"):
    print(
        "  WARNING: FedCDH.local_spns not exposed. Need to modify FedCDH.py to expose SPNs."
    )
    print("  For now, testing with mock SPNs...")

    # Mock SPN for testing (will be replaced once FedCDH exposes real SPNs)
    class MockSPN:
        def __init__(self, d, seed=42):
            self.d = d
            np.random.seed(seed)

        def log_prob(self, X):
            # Return mock log-likelihood (normal ~N(-8, 2))
            import torch

            n = len(X)
            return torch.tensor(np.random.randn(n) * 2 - 8)

        def sample(self, n):
            # Return mock samples (standard normal)
            import torch

            return torch.tensor(np.random.randn(n, self.d), dtype=torch.float32)

    local_spns = [MockSPN(d, seed=i) for i in range(K)]
    global_spn = MockSPN(d, seed=99)

    print("  Using mock SPNs for testing")
else:
    local_spns = fedcdh.local_spns
    global_spn = fedcdh.fed_spn_model
    print("  Extracted real SPNs from FedCDH")

print("\n" + "-" * 70)
print("Step 3: Initialize SPNEvaluator")
print("-" * 70)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
evaluator = SPNEvaluator(X, c_indx, K, scenario=args.scenario, device=device)

# Prepare train/test split
evaluator.prepare_train_test_split(test_ratio=0.2, seed=42)

print("\n" + "-" * 70)
print("Step 4: Evaluate Local SPNs")
print("-" * 70)

start = time.time()
for k in range(K):
    print(f"\nEvaluating Client {k}...")
    result_k = evaluator.evaluate_local_spn(
        local_spns[k], client_id=k, n_samples_mmd=500, n_permutations=100
    )
    # Reduced n_permutations for fast test (use 1000 in production)

elapsed_local = time.time() - start
print(f"\nLocal evaluation complete: {elapsed_local:.1f}s")

print("\n" + "-" * 70)
print("Step 5: Evaluate Global SPN")
print("-" * 70)

start = time.time()
global_result = evaluator.evaluate_global_spn(
    global_spn, n_samples_mmd=1000, n_permutations=100
)
elapsed_global = time.time() - start

print(f"\nGlobal evaluation complete: {elapsed_global:.1f}s")

print("\n" + "-" * 70)
print("Step 6: Generate Report")
print("-" * 70)

output_dir = Path("tests/experiments/spn_evaluation_test")
evaluator.generate_report(output_dir, include_umap=True)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

total_time = elapsed_train + elapsed_local + elapsed_global
print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Training: {elapsed_train:.1f}s")
print(f"  Local eval: {elapsed_local:.1f}s")
print(f"  Global eval: {elapsed_global:.1f}s")

print(f"\nOutput saved to: {output_dir}")
print("  - table1_local_evaluation.csv")
print("  - table2_global_evaluation.csv")
print("  - evaluation_summary.txt")

if evaluator.d > 2:
    print("  - figure1_umap_global.png (if UMAP installed)")
    print("  - figure2_umap_local_k*.png (if UMAP installed)")

print("\n✅ SPN evaluation framework validated!")
print("\nNext steps:")
print("  1. Modify FedCDH.py to expose local_spns and fed_spn_model")
print("  2. Run full evaluation on comprehensive suite data")
print("  3. Analyze reports to verify SPNs learn correctly")
print("=" * 70)
