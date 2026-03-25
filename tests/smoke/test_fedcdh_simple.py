import numpy as np
import torch
import pandas as pd
from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH
from causallearn.utils.data_utils import set_random_seed

# Mock args class
class Args:
    def __init__(self):
        self.K = 2  # 2 clients
        self.d = 3  # 3 features (X, Y, Z)
        self.scenario = "horizontal"
        self.model_type = "synthetic"
        self.ci_method = "spn"
        self.n = 200  # samples per client
        self.alpha = 0.05
        self.epochs = 5
        self.num_sums = 5
        self.num_leaves = 5
        self.num_repetitions = 2
        self.ablation_orientation = (
            "mi_only"  # Use mechanism invariance instead of hybrid
        )


def generate_synthetic_data(n_samples):
    # X -> Y -> Z
    # X ~ N(0, 1)
    # Y = X + N(0, 0.5)
    # Z = Y + N(0, 0.5)
    np.random.seed(42)
    X = np.random.normal(0, 1, n_samples)
    Y = X + np.random.normal(0, 0.5, n_samples)
    Z = Y + np.random.normal(0, 0.5, n_samples)

    data = np.vstack([X, Y, Z]).T
    # Normalize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data


def main():
    try:
        import simple_einet

        print("simple_einet is available.")
    except ImportError:
        print("simple_einet is NOT available. Simulation might fail.")
        # Mocking simple_einet if not present just to show the logic flow?
        # No, better to fail and report if it's missing.

    set_random_seed(42)
    args = Args()

    # Generate data
    data = generate_synthetic_data(args.n * args.K)

    # Split data for horizontal federated scenario
    # Client 1: first half, Client 2: second half
    # We also need a context/domain index.
    # For horizontal with same distribution, context can be 0 for all or random.
    # FedCDH expects c_indx.

    c_indx = np.zeros((len(data), 1))  # Single domain

    X_splits = np.array_split(data, args.K)

    # True DAG: 0->1, 1->2 (X->Y->Z)
    true_DAG = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

    fedcdh = FedCDH(args)
    result = fedcdh.fit(X_splits, c_indx, true_DAG)

    print("Result:", result)

    # Check if we found the edges
    # The result contains F1 scores etc, but not the graph itself returned by fit.
    # I should check FedCDH.py to see if I can get the graph.
    # fit returns 'safe_result' dictionary.

    # To verify "finding causal relationship", we look at the F1 scores in result.
    if result["f1"] > 0.3:  # Directed graph F1
        print(f"Success: Causal relationships detected (F1={result['f1']:.3f}).")
    else:
        print(f"Warning: F1 score is low (F1={result['f1']:.3f}).")


if __name__ == "__main__":
    main()
