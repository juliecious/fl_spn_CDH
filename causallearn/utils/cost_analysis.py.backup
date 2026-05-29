"""
Theoretical Communication Cost Analysis for Federated Causal Discovery.
Calculates and compares the communication overhead of FedCDH (SPN-based)
vs. Federated KCI/HSIC (Kernel-based).
"""

import numpy as np


def estimate_kci_comm_cost(n_samples, n_clients, n_queries, m_components=100):
    """
    Estimates the communication cost for a distributed KCI test using Nystroem approximation.
    For each query, clients must share partial kernel embeddings.

    Formula: Queries * Clients * (N_samples * M_components * 4 bytes)
    """
    # 4 bytes per float32
    cost_per_query = n_clients * (n_samples * m_components * 4)
    total_cost_bytes = n_queries * cost_per_query
    return total_cost_bytes / 1024.0  # KB


def estimate_fedcdh_comm_cost(n_features, n_clients, n_clusters, model_size_params):
    """
    Estimates the communication cost for FedCDH.
    Overhead is front-loaded (Clustering + Model Exchange). Discovery is local (0 cost).

    Formula: (Clustering_Iter * K * Clusters * D * 4) + (K * Model_Size * 4)
    """
    # 1. Clustering (approx 10 iterations)
    clustering_cost = 10 * n_clients * (n_clusters * n_features * 4)

    # 2. Model Exchange
    model_exchange_cost = n_clients * (model_size_params * 4)

    total_cost_bytes = clustering_cost + model_exchange_cost
    return total_cost_bytes / 1024.0  # KB


def analyze_cost_tradeoff(n_samples, n_features, n_clients, n_queries):
    """
    Prints a comparison table for the paper.
    """
    # Assume a typical SPN size for d=11: ~5000 parameters
    spn_params = 5000
    # Nystroem components: 100
    m_comp = 100

    kci_cost = estimate_kci_comm_cost(n_samples, n_clients, n_queries, m_comp)
    fed_cost = estimate_fedcdh_comm_cost(n_features, n_clients, 3, spn_params)

    print("\n" + "=" * 50)
    print(
        f"THEORETICAL COMMUNICATION COST (N={n_samples}, D={n_features}, Q={n_queries})"
    )
    print("-" * 50)
    print(f"{'Method':<20} | {'Comm Cost (KB)':<15} | {'Scalability'}")
    print("-" * 50)
    print(f"{'FedKCI (Nystroem)':<20} | {kci_cost:>15.2f} | O(Q * N * K)")
    print(f"{'FedCDH (SPN)':<20} | {fed_cost:>15.2f} | O(K * S)")
    print("-" * 50)
    print(f"Improvement: {kci_cost/fed_cost:.1f}x reduction")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Example: Sachs-like scale
    analyze_cost_tradeoff(n_samples=1000, n_features=11, n_clients=3, n_queries=150)
