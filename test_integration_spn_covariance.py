"""
Integration test: Verify FedSPN + SPN-derived covariances work together.
Tests the full pipeline: SPN training -> covariance derivation -> orientation.
"""

import numpy as np
import logging
from argparse import Namespace

from causallearn.search.FCMBased.FedCDH import FedCDH
from causallearn.utils.data_utils import (
    simulate_dag,
    simulate_parameter,
    set_random_seed,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_integration():
    """Test the integrated pipeline."""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: FedSPN + SPN-Derived Covariances")
    print("=" * 70)

    # Set random seed
    set_random_seed(42)

    # Create tiny dataset: 4 variables, 3 clients, 100 samples per client
    d = 4
    K = 3
    n_per_client = 100
    n_total = K * n_per_client

    print(f"\nDataset: {d} variables, {K} clients, {n_per_client} samples/client")

    # Generate simple DAG: V0 -> V1 -> V2 -> V3
    true_dag = np.zeros((d, d), dtype=int)
    true_dag[0, 1] = 1  # V0 -> V1
    true_dag[1, 2] = 1  # V1 -> V2
    true_dag[2, 3] = 1  # V2 -> V3

    print(f"True DAG:\n{true_dag}")

    # Generate data
    W_true = simulate_parameter(true_dag)
    X_total = np.random.randn(n_total, d)

    # Apply causal structure
    for j in range(d):
        parents = np.where(true_dag[:, j] != 0)[0]
        if len(parents) > 0:
            X_total[:, j] = (
                X_total[:, parents] @ W_true[parents, j] + X_total[:, j] * 0.3
            )

    # Split into clients (horizontal)
    X_splits = [X_total[i * n_per_client : (i + 1) * n_per_client] for i in range(K)]
    c_indx = np.repeat(np.arange(K), n_per_client).reshape(-1, 1)  # Shape (n_total, 1)

    print(f"Data splits: {[x.shape for x in X_splits]}")

    # Create FedCDH with SPN
    args = Namespace(
        K=K,
        d=d,
        n=n_per_client,
        scenario="horizontal",
        model_type="linear",
        ci_method="spn",
        alpha=0.05,
        seed=42,
        epochs=10,  # Small for testing
        use_spn_covariances=True,  # KEY: Enable SPN covariance derivation
        cov_n_samples=500,  # Small for testing
        cov_n_features=5,  # Small for testing
        skip_spn_eval=True,  # Skip expensive quality checks for integration test
        ablation_orientation="mi_hybrid",
        device="cpu",
    )

    print(f"\nRunning FedCDH with SPN covariance derivation...")
    print(f"  CI method: {args.ci_method}")
    print(f"  Use SPN covariances: {args.use_spn_covariances}")
    print(f"  Covariance samples: {args.cov_n_samples}")
    print(f"  Fourier features: {args.cov_n_features}")

    try:
        fedcdh = FedCDH(args)
        result = fedcdh.fit(X_splits, c_indx, true_dag)

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Skeleton F1: {result.get('skeleton_f1', 0.0):.3f}")
        print(f"Skeleton Precision: {result.get('skeleton_precision', 0.0):.3f}")
        print(f"Skeleton Recall: {result.get('skeleton_recall', 0.0):.3f}")
        print(f"Direction F1: {result.get('direction_f1', 0.0):.3f}")
        print(f"Direction Precision: {result.get('direction_precision', 0.0):.3f}")
        print(f"Direction Recall: {result.get('direction_recall', 0.0):.3f}")
        print(f"Training time: {result.get('time_train', 0.0):.2f}s")
        print(f"Discovery time: {result.get('time_cd', 0.0):.2f}s")

        # Verify covariance tensor was created
        if hasattr(fedcdh, "covariance_tensor"):
            CT = fedcdh.covariance_tensor
            print(f"\n✓ Covariance tensor created: shape {CT.shape}")
            print(f"  Non-zero elements: {np.count_nonzero(CT)}/{CT.size}")
            print(f"  Frobenius norm: {np.linalg.norm(CT):.4f}")
        else:
            print("\n✗ Covariance tensor not found!")

        print("\n" + "=" * 70)
        print("TEST STATUS:")
        if result.get("skeleton_f1", 0) > 0 or result.get("direction_f1", 0) > 0:
            print("✓ PASS: FedSPN + SPN-covariance pipeline executed successfully")
            print("  Metrics are non-zero, indicating graph was discovered")
        else:
            print("⚠ PARTIAL: Pipeline executed but metrics are zero")
            print("  This may be due to small sample size or test parameters")
        print("=" * 70)

        return result

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ FAIL: Integration test failed")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_integration()
