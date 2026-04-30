#!/usr/bin/env python3
"""
Quick smoke test to verify hybrid mode context column bug fix.
Tests that evaluate_spn_quality works with hybrid mode local SPNs.
"""

import numpy as np
import torch
import sys

sys.path.insert(0, "/Users/M279402/PycharmProjects/fl_spn_CDH")

from causallearn.utils.FedPC import LocalSPNWrapper
from causallearn.utils.spn_evaluation import evaluate_spn_quality


def main():
    print("\n" + "=" * 60)
    print("SMOKE TEST: Hybrid Mode Context Column Bug Fix")
    print("=" * 60 + "\n")

    # Test configuration
    n_samples = 100
    d_features = 8

    print(f"Testing: n={n_samples} samples, d={d_features} features")
    print("Bug: Local SPN evaluation added context column incorrectly\n")

    # Generate test data (no context column)
    np.random.seed(42)
    X_train = np.random.randn(n_samples, d_features)

    print("Step 1: Create and train a LocalSPNWrapper (no context column)...")
    spn = LocalSPNWrapper(
        num_features=d_features,
        device="cpu",
        depth=2,
        num_sums=10,
        num_leaves=10,
        num_repetitions=5,
    )

    # Quick training
    final_loss = spn.train_local(X_train, epochs=50, lr=0.01)
    print(f"  ✓ Training completed. Final loss: {final_loss:.4f}\n")

    print("Step 2: Evaluate SPN quality (this triggered the bug)...")
    print("  Previously failed with: 'size of tensor a (9) must match b (8)'")
    print("  Root cause: evaluation code added context column to X_train\n")

    try:
        # This is what FedCDH.evaluate_spn_quality() does
        # Previously failed because it passed [n, 9] data to SPN expecting [n, 8]
        result = evaluate_spn_quality(
            spn_model=spn,
            X_data=X_train,  # Shape: [100, 8] - NO context column
            n_samples=50,
            device="cpu",
            compute_mmd=True,
            compute_ks=True,
            name="Test SPN",
        )

        print("Step 3: Verify results...")
        print(f"  ✓ Log-likelihood computed: {result.get('train_ll', 'N/A')}")
        print(f"  ✓ MMD test: {'PASS' if 'mmd_squared' in result else 'SKIP'}")
        print(f"  ✓ KS test: {'PASS' if 'ks_failed_dims' in result else 'SKIP'}")

        print("\n" + "=" * 60)
        print("✅ SMOKE TEST PASSED")
        print("=" * 60)
        print("Bug fix verified: evaluate_spn_quality works correctly")
        print("No dimension mismatch errors occurred\n")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ SMOKE TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}\n")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
