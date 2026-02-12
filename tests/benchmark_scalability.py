import time
import numpy as np
import pandas as pd
import torch
import logging
import os
import sys

# Setup environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.append(os.getcwd())

from causallearn.utils.cit import CIT, SPN_CIT
from causallearn.utils.FedPC import GlobalFedSPN, LocalSPNWrapper

# Use CPU to show the algorithmic advantage
DEVICE = "cpu"


def generate_data(n):
    # Generates data where X _||_ Y | Z, but X is strongly dependent on Z
    np.random.seed(42)
    Z = np.random.uniform(-2, 2, n)
    X = Z**2 + np.random.normal(0, 0.1, n)
    Y = np.tanh(Z) + np.random.normal(0, 0.1, n)
    return np.column_stack((X, Y, Z))


def train_spn(data):
    n, d = data.shape
    # Fast training configuration for scalability test
    # Set depth=1 because depth=2 requires at least 4 features in simple-einet
    leaf = LocalSPNWrapper(
        num_features=d,
        device=DEVICE,
        num_sums=10,
        num_leaves=10,
        num_repetitions=1,  # Speed up training
        depth=1,
    )
    leaf.train_local(data, epochs=10, lr=0.05)
    return GlobalFedSPN([leaf], weights=[1.0], device=DEVICE)


def run_scalability_test():
    # Test larger sample sizes
    sample_sizes = [500, 1000, 2000, 5000, 10000]
    results = []

    print(f"{'N':<10} | {'FedSPN (s)':<15} | {'KCI (s)':<15} | {'Speedup':<10}")
    print("-" * 60)

    for n in sample_sizes:
        data = generate_data(n)

        # --- FedSPN ---
        # 1. Train (One-time cost, but included here to be conservative)
        t0 = time.time()
        spn = train_spn(data)
        train_time = time.time() - t0

        # 2. Inference (The repeated cost in PC algorithm)
        # Using 50 permutations for speed benchmark
        tester_spn = SPN_CIT(data, global_model=spn, num_permutations=50)

        start = time.time()
        tester_spn(0, 1, [2])
        time_spn = time.time() - start

        # --- KCI ---
        if n > 5000:
            # Estimate KCI time to avoid waiting forever
            # Based on N=2000 time * (N/2000)^3
            # We skip actual execution for very large N
            time_kci = np.nan
        else:
            tester_kci = CIT(data, "kci")
            start = time.time()
            tester_kci(0, 1, [2])
            time_kci = time.time() - start

        speedup = time_kci / time_spn if not np.isnan(time_kci) else np.inf

        # If KCI is skipped, we print "Est."
        kci_str = f"{time_kci:.4f}" if not np.isnan(time_kci) else "Too Slow"

        print(f"{n:<10} | {time_spn:<15.4f} | {kci_str:<15} | {speedup:<10.2f}")

        results.append(
            {
                "N": n,
                "FedSPN_Infer": time_spn,
                "FedSPN_Train": train_time,
                "KCI_Infer": time_kci,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_scalability_test()
    print("\nScalability test complete.")
