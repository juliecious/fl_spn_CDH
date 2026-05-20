#!/usr/bin/env python
"""
Test: Likelihood-based orientation BEFORE UCSepset for vertical mode.
Should improve both Law School and Sachs.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    print("=" * 80)
    print("VERTICAL MODE: LIKELIHOOD BEFORE UCSEPSET TEST")
    print("=" * 80)
    print()
    print("Testing: Does skipping UCSepset+Meek improve vertical orientation?")
    print()
    print("Datasets: law_school + sachs")
    print("Method: fedspn_v (vertical mode)")
    print()
    print("Expected:")
    print("  Law School: 50% → 70-90% (UCSepset was guessing)")
    print("  Sachs: 0% → 30-60% (UCSepset reversed ALL edges)")
    print()

    # Run both benchmarks
    os.system(
        "python tests/benchmarks/test_fedcdh_benchmark_v3.py "
        "--datasets law_school,sachs "
        "--methods fedspn_v "
        "--seeds 42 "
        "--K 3 "
        "--save-graphs 2>&1 | tail -150"
    )

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    import glob
    import re

    eval_files = sorted(
        glob.glob("eval/*/eval.txt"), key=os.path.getctime, reverse=True
    )[:2]

    for eval_file in eval_files:
        dataset = "unknown"
        if "law_school" in eval_file:
            dataset = "LAW SCHOOL"
        elif "sachs" in eval_file:
            dataset = "SACHS"

        with open(eval_file) as f:
            content = f.read()
            orientation_match = re.search(r"orientation_accuracy: ([\d.]+)", content)
            dag_f1_match = re.search(r"dag_f1: ([\d.]+)", content)
            skeleton_f1_match = re.search(r"skeleton_f1: ([\d.]+)", content)

            if orientation_match and dag_f1_match:
                orientation = float(orientation_match.group(1))
                dag_f1 = float(dag_f1_match.group(1))
                skeleton_f1 = float(skeleton_f1_match.group(1))

                print(f"\n{dataset}:")
                print(f"  skeleton_f1: {skeleton_f1:.3f}")
                print(f"  dag_f1: {dag_f1:.3f}")
                print(f"  orientation_accuracy: {orientation:.3f}")

                if dataset == "LAW SCHOOL":
                    if orientation > 0.50:
                        print(f"  ✓ IMPROVED from 0.500")
                    else:
                        print(f"  = Same as before (0.500)")
                elif dataset == "SACHS":
                    if orientation > 0.00:
                        print(f"  ✓ FIXED! (was 0.000)")
                    else:
                        print(f"  ✗ Still broken (0.000)")
