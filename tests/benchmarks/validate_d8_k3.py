#!/usr/bin/env python3
"""
Fast validation: Test d=8, K=3 comprehensive suite params work.

Single run: horizontal scenario, 1 seed
Expected runtime: ~10 minutes
Target: F1_skeleton > 0.7

If this passes → Full suite ready to run
If this fails → Need further debugging

Usage:
    python tests/benchmarks/validate_d8_k3.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH

print("=" * 70)
print("FAST VALIDATION: d=8, K=3 Comprehensive Suite Params")
print("=" * 70)

# Comprehensive suite BASELINE_CONFIG
class Args:
    def __init__(self):
        self.K = 3
        self.d = 8
        self.scenario = "horizontal"
        self.model_type = "linear"
        self.ci_method = "spn"
        self.n = 450  # n_per_client
        self.alpha = 0.05
        self.epochs = 100
        self.num_sums = 20
        self.num_leaves = 20
        self.num_repetitions = 10
        self.ablation_orientation = "mi_only"


# Generate chain DAG with fixed edge weights (0.8)
np.random.seed(42)
args = Args()
n_total = args.n * args.K
d = args.d
K = args.K

# True DAG: chain 0->1->2->...->d-1
true_DAG = np.zeros((d, d))
for i in range(d - 1):
    true_DAG[i, i + 1] = 1

# Generate data
X = np.zeros((n_total, d))
X[:, 0] = np.random.randn(n_total)

for i in range(1, d):
    domain_shift = np.repeat(np.arange(K), n_total // K) * 0.3  # heterogeneity=0.3
    X[:, i] = (
        0.8 * X[:, i - 1] + domain_shift + np.random.randn(n_total) * 0.5
    )  # edge_weight=0.8

# Normalize
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# Create domain indices
c_indx = np.repeat(np.arange(K), n_total // K).reshape(-1, 1)

# Split data (horizontal: row-split)
X_splits = np.array_split(X, K)

print("\nConfiguration:")
print(f"  d={d}, K={K}, n_total={n_total} (n_per_client={args.n})")
print(f"  True edges: {int(np.sum(true_DAG))} (chain)")
print(f"  Heterogeneity: 0.3")
print(f"  Edge weight: 0.8 (fixed)")
print("\nSPN params:")
print(f"  epochs={args.epochs}")
print(f"  num_sums={args.num_sums}")
print(f"  num_leaves={args.num_leaves}")
print(f"  num_repetitions={args.num_repetitions}")

print("\n" + "-" * 70)
print("Testing SPN with num_permutations=50...")
print("-" * 70)

start = time.time()
fedcdh_spn = FedCDH(args)
result_spn = fedcdh_spn.fit(X_splits, c_indx, true_DAG)
elapsed = time.time() - start

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\n  F1_skeleton:  {result_spn['f1_skeleton']:.3f}")
print(f"  F1_directed:  {result_spn['f1']:.3f}")
print(f"  SHD:          {result_spn['shd']:.1f}")
print(f"  Runtime:      {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Validation
threshold_skel = 0.7
threshold_dir = 0.4

skel_pass = result_spn["f1_skeleton"] >= threshold_skel
dir_pass = result_spn["f1"] >= threshold_dir

print("\n" + "-" * 70)
print("VALIDATION")
print("-" * 70)
print(f"  F1_skeleton >= {threshold_skel}: {'✅ PASS' if skel_pass else '❌ FAIL'}")
print(f"  F1_directed >= {threshold_dir}: {'✅ PASS' if dir_pass else '❌ FAIL'}")

print("\n" + "=" * 70)
if skel_pass and dir_pass:
    print("✅ VALIDATION PASSED!")
    print("=" * 70)
    print("\nComprehensive suite params (d=8, K=3) work correctly!")
    print("\n🚀 READY FOR FULL THESIS EXPERIMENTS")
    print("\nNext step: Run full suite")
    print(
        "  python tests/benchmarks/synthetic_comprehensive_suite.py --all --seeds 10 --gpu"
    )
    print("\nExpected runtime: 2-3 hours on GPU")
    print("Output: 6 experiments × 10 seeds = 60 runs")
else:
    print("❌ VALIDATION FAILED")
    print("=" * 70)
    print("\nResults below expected thresholds.")
    print("\nPossible issues:")
    print("  1. Check num_permutations=50 is set in FedCDH.py:467")
    print("  2. Verify BASELINE_CONFIG in synthetic_comprehensive_suite.py")
    print("  3. Try increasing epochs to 150")
    print("\nDebug suggestions:")
    print("  - Check SPN training logs (should converge)")
    print("  - Verify chain DAG generation (not fork/random)")
    print("  - Check heterogeneity=0.3, edge_weight=0.8")
print("=" * 70)
