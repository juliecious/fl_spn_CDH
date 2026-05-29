#!/usr/bin/env python
"""Test depth_limit fix for Asia dataset."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from tests.benchmarks.test_fedcdh_benchmark_v3 import UnifiedBenchmark

print("=" * 80)
print("ASIA DEPTH_LIMIT FIX TEST")
print("=" * 80)
print()
print("Testing: Asia dataset with depth_limit (adaptive)")
print("Expected: Non-zero skeleton F1 (stops at reasonable depth)")
print()

benchmark = UnifiedBenchmark(
    datasets=["asia"],
    methods=["fedspn_h"],
    seeds=[42],
    K=3,
    alpha=0.05,
    output_dir="eval/asia_depth_fix",
    device="cpu",
    save_graphs=False,
)

df_results = benchmark.run_all_experiments()

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(
    df_results[
        ["dataset", "method", "skeleton_f1", "skeleton_shd", "total_time"]
    ].to_string()
)
print()

if df_results["skeleton_f1"].iloc[0] > 0.0:
    print(f"✅ SUCCESS: Skeleton F1 = {df_results['skeleton_f1'].iloc[0]:.3f}")
    print("   The depth_limit fix resolved the Asia issue!")
else:
    print(f"❌ FAILED: Skeleton F1 still 0.000")
    print("   Need additional debugging.")

print()
