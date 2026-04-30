#!/usr/bin/env python3
"""
Quick smoke test for hybrid mode only to verify context column bug fix.
Target runtime: <5 minutes.
"""

import sys
import time

sys.path.insert(0, "/Users/M279402/PycharmProjects/fl_spn_CDH")

from tests.test.test_fedcdh_benchmark import run_single_experiment


def main():
    print("\n" + "=" * 80)
    print("SMOKE TEST: Hybrid Mode Context Column Bug Fix")
    print("=" * 80)
    print("Testing V2 hybrid mode with local clustering and sum-over-products\n")

    # Minimal configuration for speed
    config = {
        "K": 3,
        "d": 8,
        "n_total": 300,
        "graph_type": "ER",
        "degree": 2,
        "sem_type": "gauss",
        "epochs": 30,
    }

    print(f"Configuration:")
    print(f"  Clients: {config['K']}")
    print(f"  Features: {config['d']}")
    print(f"  Total samples: {config['n_total']}")
    print(f"  Epochs: {config['epochs']} (minimal for speed)")
    print()

    start_time = time.time()

    try:
        print("Running hybrid mode with V2 local clustering...")
        result = run_single_experiment(
            config_name="smoke_hybrid_bugfix",
            config=config,
            scenario="hybrid",
            data_type="linear",
            seed=42,
            device="cpu",
            use_ci_ranking=False,
            num_local_clusters=2,  # V2: local clustering
            skip_eval=True,  # Skip quality eval for speed
        )

        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print("✅ HYBRID MODE TEST PASSED")
        print("=" * 80)
        print(f"Runtime: {elapsed:.1f}s")
        print(f"Skeleton F1: {result.get('skeleton_f1', 0):.3f}")
        print(f"Skeleton SHD: {result.get('skeleton_shd', 0)}")
        print(f"Training time: {result.get('train_time', 0):.1f}s")
        print(f"CD time: {result.get('cd_time', 0):.1f}s")
        print()

        # Check if sum-over-products is working
        if result.get("skeleton_f1", 0) > 0:
            print("✓ Sum-over-products working: F1 > 0")
            print("✓ Context column bug fixed: No dimension errors")
        else:
            print("⚠ F1=0 (may need more data/epochs, but no errors)")

        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("❌ HYBRID MODE TEST FAILED")
        print("=" * 80)
        print(f"Runtime before failure: {elapsed:.1f}s")
        print(f"Error: {e}\n")
        import traceback

        traceback.print_exc()
        print("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
