"""
Benchmark Loaders for Standard Bayesian Networks (Sachs, Asia, Alarm).
Provides ground truth DAGs and heterogeneous data generation for Federated Causal Discovery.
"""

import numpy as np
import igraph as ig
import torch

# ------------------------------------------------------------------------------
# 1. Standard Graph Structures (Adjacency Matrices)
# ------------------------------------------------------------------------------

# Asia (Lung Cancer) Network - 8 Nodes
# A, S, T, L, B, E, X, D
ASIA_DAG = np.array(
    [
        [0, 0, 1, 0, 0, 0, 0, 0],  # A -> T
        [0, 0, 0, 1, 1, 0, 0, 0],  # S -> L, B
        [0, 0, 0, 0, 0, 1, 0, 0],  # T -> E
        [0, 0, 0, 0, 0, 1, 0, 0],  # L -> E
        [0, 0, 0, 0, 0, 0, 0, 1],  # B -> D
        [0, 0, 0, 0, 0, 0, 1, 1],  # E -> X, D
        [0, 0, 0, 0, 0, 0, 0, 0],  # X
        [0, 0, 0, 0, 0, 0, 0, 0],  # D
    ]
)

# Sachs (Protein Signaling) Network - 11 Nodes
# Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk
SACHS_DAG = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Raf -> Mek
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Mek -> Erk
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Plcg -> PIP2, PIP3
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # PIP2 -> PKC
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # PIP3 -> PIP2
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Erk -> Akt
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Akt
        [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],  # PKA -> Raf, Mek, Erk, Akt, P38, Jnk
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # PKC -> Raf, Mek, PKA, P38, Jnk
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # P38
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Jnk
    ]
)

# ------------------------------------------------------------------------------
# 2. Loader Utilities
# ------------------------------------------------------------------------------


def load_standard_graph(name: str) -> np.ndarray:
    """
    Load the binary adjacency matrix for a standard benchmark graph.

    Args:
        name: 'asia', 'sachs', or 'alarm' (placeholder)

    Returns:
        Adjacency matrix [d, d]
    """
    name = name.lower()
    if name == "asia":
        return ASIA_DAG
    elif name == "sachs":
        return SACHS_DAG
    elif name == "alarm":
        # Placeholder for Alarm (37 nodes) - usually requires BIF parsing
        # For now, return a random DAG of size 37 if requested, or raise error
        # Returning generic DAG for now to prevent crash if selected
        from causallearn.utils.data_utils import simulate_dag

        return simulate_dag(37, 37 * 1.5, "SF")
    else:
        raise ValueError(f"Unknown graph name: {name}")


# ------------------------------------------------------------------------------
# 3. Heterogeneous Data Generation
# ------------------------------------------------------------------------------


def simulate_heterogeneous_data(
    dag: np.ndarray, n_clients: int, n_samples: int, mode: str = "general"
):
    """
    Generate heterogeneous data for FedCDH benchmarks.

    Simulates mechanism shifts where P(Y|Parents) changes across clients.

    Args:
        dag: Binary adjacency matrix [d, d]
        n_clients: Number of clients (domains)
        n_samples: Samples per client
        mode: 'linear' (Gaussian) or 'general' (Non-linear/Non-Gaussian)

    Returns:
        X_global: Combined dataset [N_total, d]
        c_indx: Domain indicator [N_total, 1]
    """
    d = dag.shape[0]
    total_samples = n_clients * n_samples

    # Topological Sort to ensure correct generation order
    g = ig.Graph.Adjacency(dag.tolist())
    if not g.is_dag():
        raise ValueError("Input matrix is not a DAG!")
    order = g.topological_sorting()

    X = np.zeros((total_samples, d))
    c_indx = np.repeat(np.arange(n_clients), n_samples).reshape(-1, 1)

    # Randomly select mechanism-changing variables (Heterogeneity)
    # We select 20-30% of nodes to have changing mechanisms
    num_changing = max(1, int(d * 0.3))
    changing_nodes = np.random.choice(d, num_changing, replace=False)

    # Base weights (Global Skeleton)
    # W = DAG * Uniform(0.5, 2.0) * Sign
    W_base = np.random.uniform(0.5, 2.0, (d, d)) * dag
    W_base *= np.random.choice([-1, 1], size=(d, d))

    for j in order:
        parents = g.neighbors(j, mode=ig.IN)

        # Generate noise
        # Base noise
        if mode == "linear":
            noise = np.random.normal(0, 1, total_samples)
        else:
            # Mix of Uniform and Gaussian for General
            if np.random.rand() > 0.5:
                noise = np.random.uniform(-0.5, 0.5, total_samples)
            else:
                noise = np.random.normal(0, 1, total_samples)

        # Apply Heterogeneity (Mechanism Shift)
        if j in changing_nodes:
            # Different clients get different noise scales or functional parameters
            shifts = np.random.uniform(0.5, 3.0, n_clients)  # Scale shifts
            client_noise_scales = np.repeat(shifts, n_samples)
            noise *= client_noise_scales

            # For linear, we can also shift weights slightly
            if mode == "linear":
                # Slight weight perturbation per client?
                # FedCDH assumes Skeleton is invariant, but strength can change.
                pass

        # Generate Data
        if len(parents) == 0:
            X[:, j] = noise
        else:
            parent_data = X[:, parents]

            if mode == "linear":
                # Linear: Y = X @ W + Noise
                weights = W_base[parents, j]
                X[:, j] = parent_data @ weights + noise
            else:
                # General: Non-linear functions
                # Y = f(X) + Noise
                # We use simple non-linearities: sin, tanh, square

                # Global function shape, but maybe local parameters?
                # Keeping it simple: fixed function structure, changing noise is enough for "Mechanism Change"
                # effectively P(Y|X, U) != P(Y|X)

                # Random non-linearity per node
                func_type = np.random.choice(["sin", "tanh", "poly", "linear"])

                pre_act = parent_data @ W_base[parents, j]

                if func_type == "sin":
                    X[:, j] = np.sin(pre_act) + noise
                elif func_type == "tanh":
                    X[:, j] = np.tanh(pre_act) + noise
                elif func_type == "poly":
                    X[:, j] = (pre_act**2) / 5.0 + noise
                else:
                    X[:, j] = pre_act + noise

    # Normalize Global Data (Important for SPN/Neural training)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    return X, c_indx
