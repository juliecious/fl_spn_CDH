#!/usr/bin/env python3
"""
Simplified Smoke Test for V2 Option 1

Minimal test directly using benchmark infrastructure.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Just import and run the benchmark with minimal config
from test_fedcdh_benchmark import run_single_experiment
from argparse import Namespace

if __name__ == "__main__":
    print("=" * 80)
    print("V2 OPTION 1 SMOKE TEST (CPU)")
    print("=" * 80)

    # Minimal args
    args = Namespace(
        config="quick",
        data_type="linear",
        device="cpu",
        seeds=[42],  # Just one seed
        use_ci_ranking=False,
        sparsity_percentile=0.2,
        force_clusters=None,  # Test default behavior
    )

    # Import config
    from test_fedcdh_benchmark import BENCHMARK_CONFIGS

    config = BENCHMARK_CONFIGS["quick"]

    print(f"\nConfig: {config}")
    print(f"Device: cpu")
    print(f"Seeds: [42]")
    print("\nRunning horizontal mode only...")

    # Run one experiment
    result = run_single_experiment(
        scenario="horizontal",
        seed=42,
        config_name="quick",
        data_type="linear",
        device="cpu",
        use_ci_ranking=False,
        sparsity_percentile=0.2,
        force_clusters=None,
    )

    if result is not None:
        print("\n" + "=" * 80)
        print("✅ SMOKE TEST PASSED")
        print("=" * 80)
        print(f"Skeleton F1: {result.get('skeleton_f1', 'N/A')}")
        print(f"DAG F1: {result.get('dag_f1', 'N/A')}")
        print(f"Time: {result.get('total_time', 'N/A'):.1f}s")
        print("\nOption 1 working! Ready for GPU testing.")
        sys.exit(0)
    else:
        print("\n❌ SMOKE TEST FAILED")
        sys.exit(1)
