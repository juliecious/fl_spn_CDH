#!/usr/bin/env python
"""
Test vertical mode orientation fix V2.
Now cdnod properly handles feature_maps for vertical partitioning.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Run law_school benchmark with vertical mode
if __name__ == "__main__":
    print("=" * 80)
    print("VERTICAL MODE ORIENTATION FIX V2 - LAW SCHOOL TEST")
    print("=" * 80)
    print()
    print("Testing: Does cdnod properly use feature_maps for vertical partitioning?")
    print("Dataset: law_school (d=5, n=21000)")
    print("Method: fedspn_v (vertical mode)")
    print()
    print("Expected:")
    print("  - BEFORE: orientation_accuracy: 0.000")
    print("  - AFTER:  orientation_accuracy: > 0.000")
    print()

    # Run benchmark
    os.system(
        "python tests/benchmarks/test_fedcdh_benchmark_v3.py "
        "--datasets law_school "
        "--methods fedspn_v "
        "--seeds 42 "
        "--K 3 "
        "--save-graphs 2>&1 | tail -100"
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
                    print("✓ FIX V2 SUCCESSFUL!")
                    print(f"  orientation_accuracy: {orientation:.3f} (was 0.000)")
                    print(f"  dag_f1: {dag_f1:.3f} (was 0.000)")
                else:
                    print("✗ FIX V2 FAILED")
                    print(f"  orientation_accuracy: {orientation:.3f} (still zero)")
                    print(f"  dag_f1: {dag_f1:.3f} (still zero)")
    else:
        print("No eval.txt found. Check logs above for errors.")
