#!/usr/bin/env python
"""
Smoke test for vertical mode orientation fix.
Tests if passing feature_maps to cdnod resolves orientation failure.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Run tiny benchmark with vertical mode
if __name__ == "__main__":
    print("=" * 80)
    print("VERTICAL MODE ORIENTATION FIX - SMOKE TEST")
    print("=" * 80)
    print()
    print("Testing: Does passing feature_maps to cdnod fix orientation?")
    print("Dataset: synthetic_er_tiny (d=5, n=200)")
    print("Method: fedspn_v (vertical mode)")
    print("Expected: orientation_accuracy > 0.0 (was 0.0 before fix)")
    print()

    # Run benchmark
    os.system(
        "python tests/benchmarks/test_fedcdh_benchmark_v3.py "
        "--datasets synthetic_er_tiny "
        "--methods fedspn_v "
        "--seeds 42 "
        "--K 3 "
        "--save-graphs 2>&1 | tee /tmp/vertical_fix_test.log"
    )

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Parse results from latest eval.txt
    import glob

    eval_files = glob.glob("eval/*/eval.txt")
    if eval_files:
        latest_eval = max(eval_files, key=os.path.getctime)
        print(f"\nReading: {latest_eval}")
        print()
        with open(latest_eval) as f:
            content = f.read()
            print(content)

            # Check if orientation improved
            import re

            orientation_match = re.search(r"orientation_accuracy: ([\d.]+)", content)
            dag_f1_match = re.search(r"dag_f1: ([\d.]+)", content)

            if orientation_match and dag_f1_match:
                orientation = float(orientation_match.group(1))
                dag_f1 = float(dag_f1_match.group(1))

                print()
                print("=" * 80)
                print("VERDICT")
                print("=" * 80)
                if orientation > 0.0 and dag_f1 > 0.0:
                    print("✓ FIX SUCCESSFUL!")
                    print(f"  orientation_accuracy: {orientation:.3f} (was 0.000)")
                    print(f"  dag_f1: {dag_f1:.3f} (was 0.000)")
                else:
                    print("✗ FIX FAILED")
                    print(f"  orientation_accuracy: {orientation:.3f} (still zero)")
                    print(f"  dag_f1: {dag_f1:.3f} (still zero)")
    else:
        print("No eval.txt found. Check logs above for errors.")
