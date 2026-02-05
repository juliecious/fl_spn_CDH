"""
Mechanism Invariance-Based Edge Orientation for Federated Causal Discovery

This module implements edge orientation based on the principle that the correct
causal direction exhibits more invariant mechanisms across heterogeneous domains.

Based on:
- Li et al. (2024) "Federated Causal Discovery from Heterogeneous Data" (ICLR)
- Peters et al. (2016) "Causal inference using invariant prediction"

Key Idea:
For edge X -- Y, if X → Y is correct, then P(Y|X) should be similar across clients,
while P(X|Y) may vary. We use SPNs to efficiently compute conditional distributions.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from itertools import combinations


def compute_conditional_log_likelihood(
    spn_model,
    data: np.ndarray,
    target_idx: int,
    parent_indices: List[int],
    num_samples: int = 200,
) -> float:
    """
    Compute average log P(target | parents) using SPN.

    Args:
        spn_model: LocalSPNWrapper or similar SPN model
        data: [n, d] data matrix (may include context column at the end)
        target_idx: Index of target variable (in feature space, 0 to d-1)
        parent_indices: List of parent variable indices (in feature space)
        num_samples: Number of samples to use for estimation

    Returns:
        Average log conditional likelihood
    """
    if len(data) == 0:
        return 0.0

    # Sample subset if data is large
    if len(data) > num_samples:
        indices = np.random.choice(len(data), size=num_samples, replace=False)
        data_subset = data[indices]
    else:
        data_subset = data

    # Create masked data for marginalization
    n = len(data_subset)
    d = data.shape[1]  # Full dimensionality (including context if present)

    # Compute P(target, parents) via marginalization
    joint_indices = [target_idx] + parent_indices
    masked_joint = np.full((n, d), np.nan)
    masked_joint[:, joint_indices] = data_subset[:, joint_indices]

    with torch.no_grad():
        masked_joint_t = torch.from_numpy(masked_joint).float()
        if hasattr(spn_model, "device"):
            masked_joint_t = masked_joint_t.to(spn_model.device)
        log_prob_joint = spn_model.log_prob(masked_joint_t)
        ll_joint = log_prob_joint.cpu().numpy().flatten()

    # Compute P(parents) via marginalization
    if len(parent_indices) > 0:
        masked_parents = np.full((n, d), np.nan)
        masked_parents[:, parent_indices] = data_subset[:, parent_indices]

        with torch.no_grad():
            masked_parents_t = torch.from_numpy(masked_parents).float()
            if hasattr(spn_model, "device"):
                masked_parents_t = masked_parents_t.to(spn_model.device)
            log_prob_parents = spn_model.log_prob(masked_parents_t)
            ll_parents = log_prob_parents.cpu().numpy().flatten()
    else:
        ll_parents = 0.0

    # P(target | parents) = P(target, parents) / P(parents)
    log_cond = ll_joint - ll_parents

    # Return mean, handling any infinities/NaNs
    log_cond = log_cond[np.isfinite(log_cond)]
    if len(log_cond) == 0:
        return 0.0

    return np.mean(log_cond)


def compute_mechanism_variance(
    fed_spn_model,
    X_splits: List[np.ndarray],
    target_idx: int,
    parent_indices: List[int],
    use_local_models: bool = True,
) -> float:
    """
    Compute variance of mechanism P(target | parents) across clients.

    Lower variance indicates more invariant mechanism (likely correct direction).

    Args:
        fed_spn_model: FedCDH_SPN_Wrapper containing GlobalFedSPN
        X_splits: List of client data matrices (without context column)
        target_idx: Index of target variable
        parent_indices: List of parent variable indices
        use_local_models: If True, use client-specific local SPNs

    Returns:
        Variance of mechanism scores across clients
    """
    K = len(X_splits)
    mechanism_scores = []

    for k in range(K):
        if len(X_splits[k]) == 0:
            continue

        # Get the appropriate SPN model for this client
        if use_local_models and hasattr(fed_spn_model.spn, "components"):
            # Use client-specific local SPN
            local_spn = fed_spn_model.spn.components[k]
        else:
            # Use global SPN
            local_spn = fed_spn_model.spn

        # Compute log P(target | parents) on client k's data
        score = compute_conditional_log_likelihood(
            local_spn, X_splits[k], target_idx, parent_indices, num_samples=200
        )
        mechanism_scores.append(score)

    if len(mechanism_scores) < 2:
        return float("inf")  # Not enough data to compute variance

    # Return variance of mechanism scores
    return np.var(mechanism_scores)


def orient_edge_mechanism_invariance(
    i: int,
    j: int,
    fed_spn_model,
    X_splits: List[np.ndarray],
    method: str = "variance",
) -> int:
    """
    Orient edge i -- j using mechanism invariance principle.

    Tests both directions and returns the one with more invariant mechanism.

    Args:
        i: Index of first variable
        j: Index of second variable
        fed_spn_model: FedCDH_SPN_Wrapper
        X_splits: List of client data matrices [X_0, X_1, ..., X_K]
        method: "variance" or "score" (how to measure invariance)

    Returns:
        1 if i → j (i is parent of j)
        2 if j → i (j is parent of i)
    """
    # Test direction i → j: variance of P(j | i) across clients
    var_i_to_j = compute_mechanism_variance(
        fed_spn_model, X_splits, target_idx=j, parent_indices=[i]
    )

    # Test direction j → i: variance of P(i | j) across clients
    var_j_to_i = compute_mechanism_variance(
        fed_spn_model, X_splits, target_idx=i, parent_indices=[j]
    )

    # Orient toward more invariant (lower variance) direction
    if var_i_to_j < var_j_to_i:
        return 1  # i → j is more invariant
    else:
        return 2  # j → i is more invariant


def orient_skeleton_mechanism_invariance(
    skeleton_graph: np.ndarray,
    fed_spn_model,
    X_splits: List[np.ndarray],
    orientation_method: str = "mi_only",
    verbose: bool = False,
) -> np.ndarray:
    """
    Orient all undirected edges in skeleton using mechanism invariance.

    Args:
        skeleton_graph: [d, d] adjacency matrix (CPDAG format)
                       -1 = endpoint, 1 = tail
                       Undirected edge: graph[i,j] = graph[j,i] = -1
        fed_spn_model: FedCDH_SPN_Wrapper
        X_splits: List of client data matrices
        orientation_method: "mi_only", "mi_hybrid", "mi_score"
        verbose: Print orientation decisions

    Returns:
        Oriented graph (PDAG format)
    """
    d = skeleton_graph.shape[0]
    oriented_graph = skeleton_graph.copy()

    # Find all undirected edges
    undirected_edges = []
    for i in range(d):
        for j in range(i + 1, d):
            # Undirected edge: both endpoints are circles (-1, -1)
            if skeleton_graph[i, j] == -1 and skeleton_graph[j, i] == -1:
                undirected_edges.append((i, j))

    if verbose:
        print(f"\n[Mechanism Invariance Orientation]")
        print(f"Found {len(undirected_edges)} undirected edges to orient")

    # Orient each edge
    oriented_count = 0
    for i, j in undirected_edges:
        direction = orient_edge_mechanism_invariance(
            i, j, fed_spn_model, X_splits, method=orientation_method
        )

        if direction == 1:  # i → j
            oriented_graph[i, j] = 1  # Tail at i
            oriented_graph[j, i] = -1  # Arrow at j
            oriented_count += 1
            if verbose:
                print(f"  {i} → {j} (P(j|i) more invariant)")
        else:  # j → i
            oriented_graph[j, i] = 1  # Tail at j
            oriented_graph[i, j] = -1  # Arrow at i
            oriented_count += 1
            if verbose:
                print(f"  {j} → {i} (P(i|j) more invariant)")

    if verbose:
        print(f"Oriented {oriented_count} edges using mechanism invariance\n")

    return oriented_graph


def compute_hybrid_orientation_score(
    i: int,
    j: int,
    fed_spn_model,
    X_splits: List[np.ndarray],
    data_aug: np.ndarray,
    c_idx: int,
    alpha: float = 0.5,
) -> int:
    """
    Hybrid orientation combining mechanism invariance (SPN) and HSIC.

    Args:
        i, j: Variable indices
        fed_spn_model: FedCDH_SPN_Wrapper
        X_splits: List of client data matrices
        data_aug: Full augmented data (with context)
        c_idx: Index of context variable
        alpha: Weight for SPN score (1-alpha for HSIC)

    Returns:
        1 if i → j, 2 if j → i
    """
    # 1. SPN-based mechanism invariance score
    var_i_to_j = compute_mechanism_variance(
        fed_spn_model, X_splits, target_idx=j, parent_indices=[i]
    )
    var_j_to_i = compute_mechanism_variance(
        fed_spn_model, X_splits, target_idx=i, parent_indices=[j]
    )

    # Normalize to [0, 1] range
    total_var = var_i_to_j + var_j_to_i + 1e-9
    spn_score_i_j = var_j_to_i / total_var  # Higher if i→j is better
    spn_score_j_i = var_i_to_j / total_var  # Higher if j→i is better

    # 2. HSIC-based score (traditional method)
    from sklearn.kernel_approximation import Nystroem

    try:
        # Compute HSIC scores
        c_data = data_aug[:, c_idx].reshape(-1, 1)
        feature_map = Nystroem(gamma=0.2, n_components=5, random_state=1)
        C_f = feature_map.fit_transform(c_data)

        X_i = data_aug[:, i].reshape(-1, 1)
        X_j = data_aug[:, j].reshape(-1, 1)
        X_i_f = feature_map.fit_transform(X_i)
        X_j_f = feature_map.fit_transform(X_j)

        from causallearn.search.ConstraintBased.CDNOD import my_cov

        Ccc = my_cov(C_f, C_f)
        iCcc = np.linalg.inv(Ccc + np.eye(5) * 1e-10)

        Mu_i = my_cov(X_i_f, C_f) @ iCcc @ C_f.T
        Mu_j = my_cov(X_j_f, C_f) @ iCcc @ C_f.T

        XY = np.concatenate((X_i, X_j), axis=1)
        XY_f = feature_map.fit_transform(XY)
        Mu_xy = my_cov(XY_f, C_f) @ iCcc @ C_f.T

        h_i = np.sum(my_cov(Mu_i, Mu_xy) ** 2) / (np.trace(my_cov(Mu_i, Mu_i)) + 1e-9)
        h_j = np.sum(my_cov(Mu_j, Mu_xy) ** 2) / (np.trace(my_cov(Mu_j, Mu_j)) + 1e-9)

        # Normalize
        total_h = h_i + h_j + 1e-9
        hsic_score_i_j = h_j / total_h  # Higher if i→j is better (h_i < h_j)
        hsic_score_j_i = h_i / total_h

    except Exception as e:
        # Fallback to pure SPN if HSIC fails
        hsic_score_i_j = 0.5
        hsic_score_j_i = 0.5

    # 3. Combine scores
    combined_i_j = alpha * spn_score_i_j + (1 - alpha) * hsic_score_i_j
    combined_j_i = alpha * spn_score_j_i + (1 - alpha) * hsic_score_j_i

    if combined_i_j > combined_j_i:
        return 1  # i → j
    else:
        return 2  # j → i
