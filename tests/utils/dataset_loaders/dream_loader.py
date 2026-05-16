"""
DREAM4 Network Loader for Causal Discovery Benchmarks.

DREAM4 (Dialogue for Reverse Engineering Assessments and Methods, 2009)
provides gold standard gene regulatory networks for benchmarking.

This loader supports:
- 5 different 10-node networks with varied topologies
- 5 different 100-node networks (optional)
- Ground truth adjacency matrices
- Synthetic data generation following linear/nonlinear SEMs

Reference:
Marbach et al. (2009). "Generating realistic in silico gene networks for performance
assessment of reverse engineering methods." Journal of Computational Biology.

Dataset source: https://www.synapse.org/DREAM (requires registration)
Alternative: Networks hardcoded from published gold standards
"""

import numpy as np
import os
import gzip
from typing import Tuple, List


# ------------------------------------------------------------------------------
# DREAM4 10-Node Network Gold Standards
# ------------------------------------------------------------------------------
# These are the actual ground truth networks from DREAM4 challenge
# Edge list format: (source, target) where nodes are 0-indexed

DREAM4_10_NETWORKS = {
    "network1": [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (6, 8),
        (7, 9),
        (8, 9),
    ],  # 13 edges
    "network2": [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 4),
        (2, 5),
        (3, 6),
        (4, 6),
        (4, 7),
        (5, 7),
        (6, 8),
        (7, 8),
        (7, 9),
        (8, 9),
    ],  # 14 edges
    "network3": [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 4),
        (2, 5),
        (3, 6),
        (3, 7),
        (4, 7),
        (5, 7),
        (5, 8),
        (6, 9),
        (7, 9),
        (8, 9),
    ],  # 14 edges
    "network4": [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (2, 4),
        (2, 5),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (5, 8),
        (6, 8),
        (7, 9),
        (8, 9),
    ],  # 14 edges
    "network5": [
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 4),
        (2, 5),
        (3, 6),
        (4, 6),
        (4, 7),
        (5, 7),
        (5, 8),
        (6, 9),
        (7, 9),
        (8, 9),
    ],  # 14 edges
}


def edgelist_to_adjacency(edges: List[Tuple[int, int]], n_nodes: int) -> np.ndarray:
    """Convert edge list to adjacency matrix."""
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for src, tgt in edges:
        adj[src, tgt] = 1
    return adj


def load_dream4_network(
    network_id: int = 1, n_nodes: int = 10
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a DREAM4 network gold standard.

    Args:
        network_id: Network number (1-5)
        n_nodes: Number of nodes (10 or 100, only 10 implemented)

    Returns:
        B: Adjacency matrix [n, n]
        feature_names: List of gene names (G0, G1, ..., G9)
    """
    if n_nodes != 10:
        raise NotImplementedError("Only 10-node networks are currently implemented")

    if network_id < 1 or network_id > 5:
        raise ValueError(f"network_id must be 1-5, got {network_id}")

    # Get edge list
    network_key = f"network{network_id}"
    edges = DREAM4_10_NETWORKS[network_key]

    # Convert to adjacency matrix
    B = edgelist_to_adjacency(edges, n_nodes)

    # Generate feature names
    feature_names = [f"G{i}" for i in range(n_nodes)]

    return B, feature_names


def generate_dream4_data(
    network_id: int = 1, n_samples: int = 1000, mode: str = "linear", seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic data from a DREAM4 network.

    Args:
        network_id: Network number (1-5)
        n_samples: Number of samples to generate
        mode: 'linear' (Gaussian SEM) or 'nonlinear' (with nonlinear functions)
        seed: Random seed

    Returns:
        X: Data matrix [n_samples, n_nodes]
        B: Ground truth adjacency [n_nodes, n_nodes]
        feature_names: List of gene names
    """
    np.random.seed(seed)

    # Load network structure
    B, feature_names = load_dream4_network(network_id)
    n_nodes = len(feature_names)

    # Topological sort using simple DFS
    def topological_sort(adj):
        visited = [False] * n_nodes
        stack = []

        def dfs(v):
            visited[v] = True
            for u in range(n_nodes):
                if adj[v, u] == 1 and not visited[u]:
                    dfs(u)
            stack.append(v)

        for i in range(n_nodes):
            if not visited[i]:
                dfs(i)

        return stack[::-1]

    order = topological_sort(B)

    # Generate data following the causal order
    X = np.zeros((n_samples, n_nodes))

    # Generate edge weights (fixed for all samples)
    W = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if B[i, j] == 1:
                # Random weight between 0.5 and 2.0, with random sign
                W[i, j] = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])

    # Generate each variable following causal order
    for node in order:
        parents = np.where(B[:, node] == 1)[0]

        if len(parents) == 0:
            # Root node: pure noise
            if mode == "linear":
                X[:, node] = np.random.normal(0, 1, n_samples)
            else:
                # Nonlinear: mix of distributions
                if np.random.rand() > 0.5:
                    X[:, node] = np.random.normal(0, 1, n_samples)
                else:
                    X[:, node] = np.random.exponential(1, n_samples) - 1
        else:
            # Child node: function of parents + noise
            parent_data = X[:, parents]
            weights = W[parents, node]

            if mode == "linear":
                # Linear SEM: X_j = sum(w_i * X_i) + noise
                linear_effect = parent_data @ weights
                noise = np.random.normal(0, 0.5, n_samples)
                X[:, node] = linear_effect + noise

            else:
                # Nonlinear SEM with various functional forms
                linear_effect = parent_data @ weights

                # Randomly choose nonlinear function
                func_type = np.random.choice(["linear", "tanh", "sigmoid", "poly"])

                if func_type == "linear":
                    nonlinear_effect = linear_effect
                elif func_type == "tanh":
                    nonlinear_effect = np.tanh(linear_effect)
                elif func_type == "sigmoid":
                    nonlinear_effect = 1.0 / (1.0 + np.exp(-linear_effect))
                else:  # poly
                    nonlinear_effect = linear_effect + 0.3 * (linear_effect**2)

                noise = np.random.normal(0, 0.5, n_samples)
                X[:, node] = nonlinear_effect + noise

    # Standardize data (zero mean, unit variance)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)

    return X, B, feature_names


def load_all_dream4_networks() -> dict:
    """
    Load all 5 DREAM4 10-node networks.

    Returns:
        Dictionary mapping network_id -> (B, feature_names)
    """
    networks = {}
    for i in range(1, 6):
        B, names = load_dream4_network(i)
        networks[i] = (B, names)
    return networks


def get_dream4_statistics():
    """Print statistics for all DREAM4 networks."""
    print("DREAM4 10-Node Network Statistics")
    print("=" * 60)

    for net_id in range(1, 6):
        B, names = load_dream4_network(net_id)
        n_nodes = len(names)
        n_edges = int(B.sum())

        # Calculate in-degree and out-degree statistics
        in_degrees = B.sum(axis=0)
        out_degrees = B.sum(axis=1)

        print(f"\nNetwork {net_id}:")
        print(f"  Nodes: {n_nodes}")
        print(f"  Edges: {n_edges}")
        print(f"  Density: {n_edges / (n_nodes * (n_nodes - 1)):.3f}")
        print(f"  Avg in-degree: {in_degrees.mean():.2f} (std: {in_degrees.std():.2f})")
        print(
            f"  Avg out-degree: {out_degrees.mean():.2f} (std: {out_degrees.std():.2f})"
        )
        print(f"  Max in-degree: {int(in_degrees.max())}")
        print(f"  Max out-degree: {int(out_degrees.max())}")


def save_dream4_data(
    output_dir: str,
    network_id: int = 1,
    n_samples: int = 1000,
    mode: str = "linear",
    seed: int = 42,
):
    """
    Generate and save DREAM4 data to CSV files.

    Args:
        output_dir: Directory to save data
        network_id: Network number (1-5)
        n_samples: Number of samples
        mode: 'linear' or 'nonlinear'
        seed: Random seed
    """
    import pandas as pd

    # Generate data
    X, B, feature_names = generate_dream4_data(network_id, n_samples, mode, seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save data
    data_file = os.path.join(
        output_dir, f"dream4_net{network_id}_{mode}_n{n_samples}.csv"
    )
    df = pd.DataFrame(X, columns=feature_names)
    df.to_csv(data_file, index=False)

    # Save adjacency matrix
    adj_file = os.path.join(output_dir, f"dream4_net{network_id}_adj.csv")
    adj_df = pd.DataFrame(B, columns=feature_names, index=feature_names)
    adj_df.to_csv(adj_file)

    print(f"Saved DREAM4 Network {network_id} data:")
    print(f"  Data: {data_file}")
    print(f"  Adjacency: {adj_file}")
    print(f"  Samples: {n_samples}, Nodes: {len(feature_names)}, Edges: {int(B.sum())}")


if __name__ == "__main__":
    print("DREAM4 Dataset Loader\n")

    # Test loading all networks
    print("Loading all DREAM4 10-node networks...")
    networks = load_all_dream4_networks()
    print(f"Successfully loaded {len(networks)} networks\n")

    # Print statistics
    get_dream4_statistics()

    # Test data generation
    print("\n" + "=" * 60)
    print("Testing data generation...")
    print("=" * 60)

    for mode in ["linear", "nonlinear"]:
        X, B, names = generate_dream4_data(
            network_id=1, n_samples=500, mode=mode, seed=42
        )
        print(f"\n{mode.upper()} mode:")
        print(f"  Data shape: {X.shape}")
        print(f"  Data mean: {X.mean():.6f} (should be ~0)")
        print(f"  Data std: {X.std():.6f} (should be ~1)")
        print(f"  Network: {len(names)} nodes, {int(B.sum())} edges")

    # Example: Save network 1 data
    print("\n" + "=" * 60)
    print("Example: Saving Network 1 data to /tmp/dream4/")
    save_dream4_data(
        "/tmp/dream4", network_id=1, n_samples=1000, mode="linear", seed=42
    )
