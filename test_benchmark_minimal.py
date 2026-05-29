#!/usr/bin/env python
"""
Minimal benchmark test to debug the 0.000 metrics issue.

Tests only horizontal mode on tiny synthetic dataset.
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import after path setup
from tests.benchmarks.test_fedcdh_benchmark_v3 import UnifiedBenchmark

print("=" * 80)
print("MINIMAL BENCHMARK TEST - Debug 0.000 Metrics")
print("=" * 80)
print()

# Configuration
print("Configuration:")
print("  Dataset: synthetic_er_tiny (5 nodes, 200 samples)")
print("  Method: fedspn_h (horizontal only)")
print("  Seeds: [42]")
print("  K: 3 clients")
print("  Device: cpu")
print("  Save graphs: True (for debugging)")
print()

# Create benchmark
benchmark = UnifiedBenchmark(
    datasets=["synthetic_er_tiny"],  # Tiny: 5 nodes, 200 samples
    methods=["fedspn_h"],  # Horizontal only
    seeds=[42],  # Single seed
    K=3,
    alpha=0.05,
    output_dir="eval/debug_minimal",
    device="cpu",  # CPU for quick test
    save_graphs=True,  # Enable for debugging
)

print("=" * 80)
print("Running benchmark...")
print("=" * 80)
print()

# Run
df_results = benchmark.run_all_experiments()

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(df_results.to_string())
print()

# Check for failures
if df_results["skeleton_f1"].iloc[0] == 0.0:
    print("❌ FAILED: Skeleton F1 = 0.000")
    print()
    print("Checking logs in eval/debug_minimal/...")

    # Find the experiment folder
    import glob

    exp_folders = glob.glob("eval/debug_minimal/*/")
    if exp_folders:
        log_file = Path(exp_folders[0]) / "run.log"
        if log_file.exists():
            print(f"\nLast 50 lines of {log_file}:")
            print("-" * 80)
            with open(log_file, "r") as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line.rstrip())
else:
    print(f"✅ SUCCESS: Skeleton F1 = {df_results['skeleton_f1'].iloc[0]:.3f}")

print()
print("=" * 80)
print("Test complete")
print("=" * 80)
