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

# Use simple CPU execution
DEVICE = "cpu"


def generate_data(n, scenario="nonlinear"):
    np.random.seed(42)
    Z = np.random.uniform(-2, 2, n)
    if scenario == "nonlinear":
        # Let's make X _||_ Y | Z
        X = Z**2 + np.random.normal(0, 0.1, n)
        Y = np.tanh(Z) + np.random.normal(0, 0.1, n)
    return np.column_stack((X, Y, Z))


def train_spn(data):
    # Quick oracle training
    n, d = data.shape
    # Set depth=1 because depth=2 requires at least 4 features in simple-einet
    leaf = LocalSPNWrapper(
        num_features=d,
        device=DEVICE,
        num_sums=10,
        num_leaves=10,
        num_repetitions=2,
        depth=1,
    )
    leaf.train_local(data, epochs=20, lr=0.01)  # Faster training for benchmark
    return GlobalFedSPN([leaf], weights=[1.0], device=DEVICE)


def benchmark_scalability():
    # Large N to show KCI slowdown
    sample_sizes = [500, 1000, 2000, 5000]
    results = []

    print(f"{'N':<10} | {'FedSPN (s)':<15} | {'KCI (s)':<15} | {'Speedup':<10}")
    print("-" * 60)

    for n in sample_sizes:
        data = generate_data(n)

        # FedSPN
        spn = train_spn(data)
        # Reduced perms for speed benchmark, as complexity is linear w.r.t perms
        tester_spn = SPN_CIT(data, global_model=spn, num_permutations=20)

        start = time.time()
        tester_spn(0, 1, [2])
        time_spn = time.time() - start

        # KCI
        if n > 3000:  # KCI is too slow
            time_kci = np.nan
        else:
            tester_kci = CIT(data, "kci")
            start = time.time()
            tester_kci(0, 1, [2])
            time_kci = time.time() - start

        speedup = time_kci / time_spn if not np.isnan(time_kci) else np.inf
        print(f"{n:<10} | {time_spn:<15.4f} | {time_kci:<15.4f} | {speedup:<10.2f}")

        results.append({"N": n, "FedSPN": time_spn, "KCI": time_kci})

    return pd.DataFrame(results)


if __name__ == "__main__":
    benchmark_scalability()
