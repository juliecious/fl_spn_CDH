#!/usr/bin/env python
"""Quick test to verify the permutation test fix for Asia dataset."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from tests.benchmarks.test_fedcdh_benchmark_v3 import UnifiedBenchmark

print("=" * 80)
print("ASIA PERMUTATION TEST FIX VERIFICATION")
print("=" * 80)
print()
print("Testing: Asia dataset with num_permutations=50 (default)")
print("Expected: Non-zero skeleton F1 (should find some edges)")
print()

benchmark = UnifiedBenchmark(
    datasets=["asia"],
    methods=["fedspn_h"],  # Test horizontal only for speed
    seeds=[42],
    K=3,
    alpha=0.05,
    output_dir="eval/asia_fix_test",
    device="cpu",  # CPU for quick test
    save_graphs=False,  # Skip graphs for speed
)

df_results = benchmark.run_all_experiments()

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(
    df_results[
        [
            "dataset",
            "method",
            "skeleton_f1",
            "skeleton_precision",
            "skeleton_recall",
            "skeleton_shd",
        ]
    ].to_string()
)
print()

if df_results["skeleton_f1"].iloc[0] > 0.0:
    print(f"✅ SUCCESS: Skeleton F1 = {df_results['skeleton_f1'].iloc[0]:.3f}")
    print("   The permutation test fix resolved the Asia 0-edge bug!")
else:
    print(f"❌ FAILED: Skeleton F1 still 0.000")
    print("   Additional debugging needed.")

print()
