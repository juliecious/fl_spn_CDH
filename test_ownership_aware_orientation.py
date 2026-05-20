#!/usr/bin/env python
"""
Test ownership-aware orientation for vertical mode.

This test implements the Seng-inspired approach:
- Within-client edges: use local SPN
- Cross-client edges: use global product SPN
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    print("=" * 80)
    print("OWNERSHIP-AWARE ORIENTATION TEST (SENG-INSPIRED)")
    print("=" * 80)
    print()
    print("Implementation:")
    print(
        "  - Within-client edges (e.g., Age → Blood_Pressure): Use local Hospital SPN"
    )
    print("  - Cross-client edges (e.g., Age → Glucose): Use global product SPN")
    print()
    print("Testing on: law_school + sachs (vertical mode)")
    print()
    print("Expected improvements:")
    print("  Law School: 50% → 70-90% (local SPNs more accurate for within-client)")
    print("  Sachs: 0% → 40-70% (anything > 0% is huge improvement!)")
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

    results = {}
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
            dag_reversed_match = re.search(r"dag_reversed: (\d+)", content)

            if orientation_match and dag_f1_match:
                results[dataset] = {
                    "orientation": float(orientation_match.group(1)),
                    "dag_f1": float(dag_f1_match.group(1)),
                    "skeleton_f1": float(skeleton_f1_match.group(1)),
                    "dag_reversed": int(dag_reversed_match.group(1))
                    if dag_reversed_match
                    else 0,
                }

    # Print comparison
    for dataset in ["LAW SCHOOL", "SACHS"]:
        if dataset in results:
            r = results[dataset]
            print(f"\n{dataset}:")
            print(f"  skeleton_f1: {r['skeleton_f1']:.3f}")
            print(f"  dag_f1: {r['dag_f1']:.3f}")
            print(f"  orientation_accuracy: {r['orientation']:.3f}")
            print(f"  dag_reversed: {r['dag_reversed']}")

            if dataset == "LAW SCHOOL":
                if r["orientation"] > 0.50:
                    print(f"  ✓ IMPROVED from 0.500")
                elif r["orientation"] == 0.50:
                    print(f"  = Same as before (0.500)")
                else:
                    print(f"  ✗ WORSE than before (0.500)")
            elif dataset == "SACHS":
                if r["orientation"] > 0.00:
                    print(f"  ✓✓ FIXED! Was completely broken (0.000)")
                    print(
                        f"  ✓✓ No longer reversing all edges (reversed={r['dag_reversed']}/7)"
                    )
                else:
                    print(f"  ✗ Still broken (0.000)")

    print()
    print("=" * 80)
    print("OWNERSHIP-AWARE ORIENTATION")
    print("=" * 80)
    print("This approach leverages Seng's ProductOverGroups design:")
    print("  - Feature ownership encoded in model structure")
    print("  - Local SPNs for within-client edges (more accurate)")
    print("  - Global product SPN for cross-client dependencies")
    print("=" * 80)
