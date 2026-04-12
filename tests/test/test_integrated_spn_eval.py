"""
Smoke test demonstrating integrated SPN quality evaluation.
Shows MMD, KS metrics logged during training and UMAP visualizations.
"""

import sys
import os
import logging
import numpy as np
from argparse import Namespace

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging to see all output
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

from causallearn.utils.data_utils import (
    my_simulate_linear_gaussian,
    set_random_seed,
    simulate_dag,
    simulate_parameter,
)
from causallearn.search.FCMBased.FedCDH import FedCDH


def run_integrated_test(scenario="horizontal", d=6, K=2, n=200, epochs=30, seed=42):
    """
    Run FedCDH with integrated SPN quality evaluation.

    The workflow now automatically:
    - Trains SPNs
    - Evaluates quality with MMD and KS tests
    - Generates UMAP visualizations
    - Logs everything during execution
    """
    print("\n" + "=" * 70)
    print(f"INTEGRATED SPN EVALUATION TEST - {scenario.upper()}")
    print(f"d={d}, K={K}, n={n}, epochs={epochs}")
    print("=" * 70 + "\n")

    # Generate data
    print("Generating synthetic data...")
    set_random_seed(seed)
    B = simulate_dag(d, s0=d, graph_type="ER")
    W = simulate_parameter(B)
    X, choice = my_simulate_linear_gaussian(W, K=K, n=n, sem_type="gauss")
    c_indx = np.repeat(np.arange(K), n // K).reshape(-1, 1)

    print(f"Data: {n} samples, {d} variables, {K} clients")
    print(f"Heterogeneous variables: {choice}\n")

    # Partition data
    print(f"Partitioning data ({scenario})...")
    if scenario == "horizontal":
        samples_per_client = n // K
        X_splits = [
            X[k * samples_per_client : (k + 1) * samples_per_client, :]
            for k in range(K)
        ]
    elif scenario == "vertical":
        features_per_client = d // K
        X_splits = [
            X[:, k * features_per_client : (k + 1) * features_per_client]
            for k in range(K)
        ]
        if d % K != 0:
            X_splits[-1] = X[:, (K - 1) * features_per_client :]
    else:  # hybrid
        samples_per_client = n // K
        X_splits = [
            X[k * samples_per_client : (k + 1) * samples_per_client, :]
            for k in range(K)
        ]

    # Setup FedCDH - output directory will be auto-generated
    args = Namespace(
        K=K,
        d=d,
        n=n // K,
        scenario=scenario,
        model_type="synthetic",
        ci_method="spn",  # Important: use 'spn' to trigger evaluation
        alpha=0.05,
        epochs=epochs,
        device="cpu",
        skip_bic=True,
        # spn_eval_dir not specified - will auto-generate in eval/ with unique ID
    )

    print("\nRunning FedCDH with integrated SPN evaluation...")
    print("Watch for automatic MMD, KS, and UMAP generation!\n")

    # Run FedCDH - this will now automatically evaluate SPNs
    fedcdh = FedCDH(args)
    results = fedcdh.fit(X_splits, c_indx, B)

    # Get the auto-generated output directory
    output_dir = getattr(fedcdh, "spn_eval_dir", None)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Training time: {results.get('time_train', 0):.2f}s")
    print(f"Skeleton F1: {results.get('f1_skeleton', 0):.3f}")
    print(f"DAG F1: {results.get('f1', 0):.3f}")
    if output_dir:
        print(f"\nUMAP visualizations saved to: {output_dir}/")
    print("=" * 70 + "\n")

    # List generated files
    if output_dir and os.path.exists(output_dir):
        print("Generated files:")
        files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
        for f in files:
            print(f"  - {f}")
    else:
        print("No UMAP files generated")

    return results


def main():
    """Run smoke test for integrated SPN evaluation."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: Integrated SPN Quality Evaluation")
    print("=" * 70)
    print("\nThis test demonstrates the NEW integrated workflow:")
    print("1. SPNs are trained as usual")
    print("2. Automatically evaluates each local SPN (MMD, KS tests)")
    print("3. Automatically evaluates global federated SPN")
    print("4. Automatically generates UMAP visualizations")
    print("5. All metrics logged during training")
    print("\nNo manual evaluation needed - it's all automatic!")
    print("=" * 70)

    # Test parameters
    test_params = {
        "d": 6,  # Multivariate for UMAP
        "K": 2,  # 2 clients
        "n": 200,  # Small for speed
        "epochs": 30,  # Enough for some convergence
        "seed": 42,
    }

    # Test horizontal scenario (vertical has dimension complexities)
    scenarios_to_test = ["horizontal", "hybrid"]

    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    for scenario in scenarios_to_test:
        result = run_integrated_test(scenario=scenario, **test_params)
        print("\n")  # Spacing between tests

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)
    print("\nEach run created a unique folder under eval/ with:")
    print("  - Timestamp + scenario + configuration in folder name")
    print("  - All UMAP visualizations inside")
    print("\nThe workflow now automatically evaluates SPN quality!")
    print("Check the eval/ directory for all results.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
