#!/usr/bin/env python
"""
Quick smoke test for Law School and Sachs datasets.
Tests data loading and SPN depth constraint with minimal training.
"""
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_dataset(config_name):
    """Test a single dataset configuration."""
    logging.info(f"\n{'='*70}")
    logging.info(f"TESTING: {config_name.upper()}")
    logging.info(f"{'='*70}")

    try:
        if config_name == "law_school":
            from tests.utils.law_school_loader import load_law_school_federated

            X, B, feature_names = load_law_school_federated(
                n_clients=3, n_samples_limit=900
            )
            logging.info(
                f"✅ Law School loaded: {X.shape[0]} samples, {X.shape[1]} features"
            )
            logging.info(f"   Features: {feature_names}")
            logging.info(f"   Ground truth edges: {int(np.sum(B))}")

        elif config_name == "sachs":
            import gzip
            import pandas as pd

            data_path = "tests/data/sachs.interventional.txt.gz"
            with gzip.open(data_path, "rt") as f:
                df = pd.read_csv(f, sep=" ")

            # Remove INT column if present
            if "INT" in df.columns:
                df = df.drop("INT", axis=1)

            X = df.values[:900]  # Limit to 900 samples for quick test

            # Ground truth (17 edges)
            B = np.zeros((11, 11))
            edges = [
                (7, 0),
                (7, 1),
                (7, 5),
                (7, 6),
                (7, 9),
                (7, 10),  # PKA
                (8, 0),
                (8, 1),
                (8, 8),
                (8, 9),
                (8, 10),  # PKC
                (2, 3),
                (3, 4),
                (4, 2),  # Plcg-PIP2-PIP3
                (0, 1),
                (1, 5),
                (5, 6),  # Raf-Mek-Erk-Akt
            ]
            for i, j in edges:
                B[i, j] = 1

            logging.info(f"✅ Sachs loaded: {X.shape[0]} samples, {X.shape[1]} features")
            logging.info(f"   Ground truth edges: {int(np.sum(B))}")

        # Test SPN depth calculation
        from causallearn.utils.FedPC import compute_adaptive_hyperparameters

        d_with_context = X.shape[1] + 1  # Add context column
        params = compute_adaptive_hyperparameters(
            mode="horizontal",
            num_features=d_with_context,
            num_samples=X.shape[0] // 3,
            data_type="nonlinear",
        )

        depth_valid = 2 ** params["depth"] <= d_with_context
        logging.info(
            f"   SPN depth: {params['depth']}, valid: {'✅' if depth_valid else '❌'}"
        )

        if depth_valid:
            logging.info(f"✅ {config_name.upper()} SMOKE TEST PASSED")
            return True
        else:
            logging.error(
                f"❌ {config_name.upper()} SMOKE TEST FAILED: Depth constraint violated"
            )
            return False

    except Exception as e:
        logging.error(f"❌ {config_name.upper()} SMOKE TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = {}

    # Test both datasets
    for dataset in ["law_school", "sachs"]:
        results[dataset] = test_dataset(dataset)

    # Summary
    logging.info(f"\n{'='*70}")
    logging.info("SMOKE TEST SUMMARY")
    logging.info(f"{'='*70}")
    for dataset, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logging.info(f"  {dataset.ljust(15)}: {status}")

    # Exit with appropriate code
    if all(results.values()):
        logging.info(f"\n🎉 ALL SMOKE TESTS PASSED")
        sys.exit(0)
    else:
        logging.error(f"\n❌ SOME SMOKE TESTS FAILED")
        sys.exit(1)
