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
    model,
    data: np.ndarray,
    target_idx: int,
    parent_indices: List[int],
    num_samples: int = 200,
) -> float:
    """
    Compute average log P(target | parents) using the model (SPN Wrapper).
    Data should include the context variable U if the model expects it.
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
    d = data.shape[1]

    # Compute P(target, parents, [U]) via marginalization
    # Note: If U is present in data, we keep it to condition on the domain.
    # In CDNOD, data is augmented [X, U].
    joint_indices = [target_idx] + parent_indices
    # If the last column is U, we always keep it
    if d > max(joint_indices):
        u_idx = d - 1
        if u_idx not in joint_indices:
            joint_indices.append(u_idx)

    masked_joint = np.full((n, d), np.nan)
    masked_joint[:, joint_indices] = data_subset[:, joint_indices]

    with torch.no_grad():
        masked_joint_t = torch.from_numpy(masked_joint).float()
        if hasattr(model, "device"):
            masked_joint_t = masked_joint_t.to(model.device)
        log_prob_joint = model.log_prob(masked_joint_t)
        ll_joint = log_prob_joint.cpu().numpy().flatten()

    # Compute P(parents, [U]) via marginalization
    parent_u_indices = parent_indices.copy()
    if d > max(parent_u_indices if parent_u_indices else [0]):
        u_idx = d - 1
        if u_idx not in parent_u_indices:
            parent_u_indices.append(u_idx)

    masked_parents = np.full((n, d), np.nan)
    masked_parents[:, parent_u_indices] = data_subset[:, parent_u_indices]

    with torch.no_grad():
        masked_parents_t = torch.from_numpy(masked_parents).float()
        if hasattr(model, "device"):
            masked_parents_t = masked_parents_t.to(model.device)
        log_prob_parents = model.log_prob(masked_parents_t)
        ll_parents = log_prob_parents.cpu().numpy().flatten()

    # P(target | parents, U) = P(target, parents, U) / P(parents, U)
    log_cond = ll_joint - ll_parents

    # Return mean
    log_cond = log_cond[np.isfinite(log_cond)]
    if len(log_cond) == 0:
        return 0.0

    return np.mean(log_cond)


def compute_mechanism_variance(
    fed_spn_model,
    X_aug_splits: List[np.ndarray],
    target_idx: int,
    parent_indices: List[int],
) -> float:
    """
    Compute variance of mechanism P(target | parents, U=k) across splits k.
    """
    mechanism_scores = []

    for k in range(len(X_aug_splits)):
        if len(X_aug_splits[k]) == 0:
            continue

        # Evaluate the conditional LL for this domain k using the Global Wrapper
        score = compute_conditional_log_likelihood(
            fed_spn_model, X_aug_splits[k], target_idx, parent_indices, num_samples=200
        )
        mechanism_scores.append(score)

    if len(mechanism_scores) < 2:
        return 0.0  # No variance if only one domain

    return np.var(mechanism_scores)


def orient_edge_mechanism_invariance(
    i: int,
    j: int,
    fed_spn_model,
    X_aug_splits: List[np.ndarray],
    method: str = "mi_only",
    data_aug: Optional[np.ndarray] = None,
    c_idx: int = -1,
) -> int:
    """
    Orient edge i -- j using mechanism invariance principle.

    Theoretical Foundation:
        Based on Invariant Causal Prediction (Peters et al., 2016). For the correct
        causal direction X → Y, the conditional mechanism P(Y|X) should remain invariant
        across heterogeneous environments, while P(X|Y) may vary.

    Key Assumptions:
        1. Structural Causal Model: Y = f(X, ε_Y) where ε_Y ⊥ X (noise is independent)
        2. Invariance Principle: P(ε_Y|U=k) invariant across domains k ⟺ X → Y is correct
        3. Sufficient Heterogeneity: Domains exhibit enough distributional shift to detect
           mechanism variance differences between causal directions

    Failure Modes:
        - Weak instruments: Low signal-to-noise ratio makes variance differences undetectable
        - Context-dependent confounders: Hidden variables H that affect both X and Y differently
          across domains, violating the independence assumption
        - Adaptive mechanisms: When causal mechanisms themselves change across domains
          (non-stationary environments), violating the invariance principle

    Args:
        i: Index of first variable
        j: Index of second variable
        fed_spn_model: Trained federated SPN model for density estimation
        X_aug_splits: List of domain-specific data splits [X, U]
        method: 'mi_only' (variance-based) or 'mi_hybrid' (variance + HSIC)
        data_aug: Global augmented data [X, U] (required for mi_hybrid)
        c_idx: Context variable index in data (default: -1)

    Returns:
        int: 1 if i → j is more invariant, 2 if j → i is more invariant

    References:
        - Peters, J., Bühlmann, P., & Meinshausen, N. (2016). Causal inference using
          invariant prediction: identification and confidence intervals. JRSS-B.
        - Arjovsky, M., et al. (2019). Invariant Risk Minimization. arXiv:1907.02893.
        - Li, B., et al. (2024). Federated Causal Discovery from Heterogeneous Data. ICLR.
    """
    if method == "mi_hybrid":
        if data_aug is None:
            raise ValueError("mi_hybrid requires data_aug")
        return compute_hybrid_orientation_score(
            i, j, fed_spn_model, X_aug_splits, data_aug, c_idx
        )

    # Default: "mi_only" or "variance"
    # Test direction i → j: variance of P(j | i, U) across splits
    var_i_to_j = compute_mechanism_variance(
        fed_spn_model, X_aug_splits, target_idx=j, parent_indices=[i]
    )

    # Test direction j → i: variance of P(i | j, U) across splits
    var_j_to_i = compute_mechanism_variance(
        fed_spn_model, X_aug_splits, target_idx=i, parent_indices=[j]
    )

    if var_i_to_j < var_j_to_i:
        return 1  # i → j is more invariant
    else:
        return 2  # j → i is more invariant


def orient_skeleton_mechanism_invariance(
    skeleton_graph: np.ndarray,
    fed_spn_model,
    X_aug_splits: List[np.ndarray],
    orientation_method: str = "mi_only",
    verbose: bool = False,
    data_aug: Optional[np.ndarray] = None,
    c_idx: int = -1,
    feature_maps: Optional[dict] = None,
    local_spns: Optional[List] = None,
) -> np.ndarray:
    """
    Orient all undirected edges in skeleton using mechanism invariance.

    For vertical mode (feature_maps provided), uses ownership-aware orientation
    that leverages both local SPNs (within-client) and global product SPN (cross-client).
    """
    d = skeleton_graph.shape[0]
    oriented_graph = skeleton_graph.copy()

    # Find all undirected edges
    undirected_edges = []
    for i in range(d):
        for j in range(i + 1, d):
            if skeleton_graph[i, j] == -1 and skeleton_graph[j, i] == -1:
                undirected_edges.append((i, j))

    # Detect vertical mode: feature_maps provided
    is_vertical_mode = feature_maps is not None

    if is_vertical_mode and verbose:
        print(f"\n[Vertical Mode: Ownership-Aware Orientation]")
        print(f"Found {len(undirected_edges)} undirected edges to orient")
        print(
            f"Using ownership-aware strategy (local SPN for within-client, global for cross-client)"
        )
    elif verbose:
        print(f"\n[Mechanism Invariance Orientation]")
        print(f"Found {len(undirected_edges)} undirected edges to orient")

    # Orient each edge
    oriented_count = 0
    for i, j in undirected_edges:
        if is_vertical_mode:
            # Vertical mode: Use ownership-aware orientation
            if local_spns is not None:
                direction = orient_edge_vertical_with_ownership(
                    i,
                    j,
                    fed_spn_model,
                    local_spns,
                    feature_maps,
                    data_aug if data_aug is not None else X_aug_splits[0],
                    verbose=verbose,
                )
            else:
                # Fallback to likelihood-based if no local SPNs provided
                direction = orient_edge_likelihood_based(
                    i,
                    j,
                    fed_spn_model,
                    data_aug if data_aug is not None else X_aug_splits[0],
                    verbose=False,
                )
        else:
            # Horizontal/Hybrid: Use mechanism invariance
            direction = orient_edge_mechanism_invariance(
                i,
                j,
                fed_spn_model,
                X_aug_splits,
                method=orientation_method,
                data_aug=data_aug,
                c_idx=c_idx,
            )

        if direction == 1:  # i → j
            oriented_graph[i, j] = 1  # Tail at i
            oriented_graph[j, i] = -1  # Arrow at j
            oriented_count += 1
            if verbose:
                if is_vertical_mode:
                    print(f"  {i} → {j} (better likelihood fit)")
                else:
                    print(f"  {i} → {j} (P(j|i) more invariant)")
        else:  # j → i
            oriented_graph[j, i] = 1  # Tail at j
            oriented_graph[i, j] = -1  # Arrow at i
            oriented_count += 1
            if verbose:
                if is_vertical_mode:
                    print(f"  {j} → {i} (better likelihood fit)")
                else:
                    print(f"  {j} → {i} (P(i|j) more invariant)")

    if verbose:
        print(f"Oriented {oriented_count} edges using mechanism invariance\n")

    return oriented_graph


def compute_hybrid_orientation_score(
    i: int,
    j: int,
    fed_spn_model,
    X_aug_splits: List[np.ndarray],
    data_aug: np.ndarray,
    c_idx: int,
    alpha: float = 0.5,
) -> int:
    """
    Hybrid orientation combining mechanism invariance (SPN) and HSIC.
    """
    # 1. SPN-based mechanism invariance score
    var_i_to_j = compute_mechanism_variance(
        fed_spn_model, X_aug_splits, target_idx=j, parent_indices=[i]
    )
    var_j_to_i = compute_mechanism_variance(
        fed_spn_model, X_aug_splits, target_idx=i, parent_indices=[j]
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


def orient_edge_likelihood_based(
    i: int,
    j: int,
    fed_spn_model,
    data_aug: np.ndarray,
    num_samples: int = 500,
    verbose: bool = False,
) -> int:
    """
    Orient edge i--j using likelihood comparison (for vertical mode).

    Since vertical mode has no heterogeneity, we cant use MI. Instead,
    we compare the likelihood fit of X→Y vs Y→X using the trained SPN.

    Algorithm:
        1. Compute log P(Y|X) ≈ log P(X,Y) - log P(X) using SPN
        2. Compute log P(X|Y) ≈ log P(X,Y) - log P(Y) using SPN
        3. Return direction with higher average conditional likelihood

    Args:
        i, j: Feature indices
        fed_spn_model: Trained global SPN
        data_aug: Full augmented data [X, context]
        num_samples: Number of samples to use for evaluation
        verbose: Print debug info

    Returns:
        1 if i→j, -1 if j→i
    """
    import torch

    n = data_aug.shape[0]
    d = data_aug.shape[1]

    # Sample subset for efficiency
    if n > num_samples:
        indices = np.random.choice(n, size=num_samples, replace=False)
        data_subset = data_aug[indices]
    else:
        data_subset = data_aug

    # Compute log P(i, j) - joint probability
    masked_joint = np.full((len(data_subset), d), np.nan)
    masked_joint[:, [i, j]] = data_subset[:, [i, j]]
    # Keep context column if present
    if d > max(i, j):
        context_idx = d - 1
        masked_joint[:, context_idx] = data_subset[:, context_idx]

    with torch.no_grad():
        masked_joint_t = torch.from_numpy(masked_joint).float()
        if hasattr(fed_spn_model, "device"):
            masked_joint_t = masked_joint_t.to(fed_spn_model.device)
        log_prob_joint = fed_spn_model.log_prob(masked_joint_t)
        ll_joint = log_prob_joint.cpu().numpy().flatten()

    # Compute log P(i) - marginal of i
    masked_i = np.full((len(data_subset), d), np.nan)
    masked_i[:, i] = data_subset[:, i]
    if d > max(i, j):
        masked_i[:, context_idx] = data_subset[:, context_idx]

    with torch.no_grad():
        masked_i_t = torch.from_numpy(masked_i).float()
        if hasattr(fed_spn_model, "device"):
            masked_i_t = masked_i_t.to(fed_spn_model.device)
        log_prob_i = fed_spn_model.log_prob(masked_i_t)
        ll_i = log_prob_i.cpu().numpy().flatten()

    # Compute log P(j) - marginal of j
    masked_j = np.full((len(data_subset), d), np.nan)
    masked_j[:, j] = data_subset[:, j]
    if d > max(i, j):
        masked_j[:, context_idx] = data_subset[:, context_idx]

    with torch.no_grad():
        masked_j_t = torch.from_numpy(masked_j).float()
        if hasattr(fed_spn_model, "device"):
            masked_j_t = masked_j_t.to(fed_spn_model.device)
        log_prob_j = fed_spn_model.log_prob(masked_j_t)
        ll_j = log_prob_j.cpu().numpy().flatten()

    # Compute conditional likelihoods
    # log P(j|i) = log P(i,j) - log P(i)
    log_cond_j_given_i = ll_joint - ll_i
    # log P(i|j) = log P(i,j) - log P(j)
    log_cond_i_given_j = ll_joint - ll_j

    # Average over samples (filter out infinities/NaNs)
    log_cond_j_given_i = log_cond_j_given_i[np.isfinite(log_cond_j_given_i)]
    log_cond_i_given_j = log_cond_i_given_j[np.isfinite(log_cond_i_given_j)]

    if len(log_cond_j_given_i) == 0 or len(log_cond_i_given_j) == 0:
        # Fallback: random orientation if we cant compute
        return 1 if np.random.rand() < 0.5 else -1

    score_i_to_j = np.mean(log_cond_j_given_i)
    score_j_to_i = np.mean(log_cond_i_given_j)

    if verbose:
        print(f"Edge {i}--{j}:")
        print(f"  log P(j|i) = {score_i_to_j:.3f}")
        print(f"  log P(i|j) = {score_j_to_i:.3f}")

    # Orient towards higher conditional likelihood
    # Higher log P(j|i) suggests i→j is better fit
    if score_i_to_j > score_j_to_i:
        return 1  # i → j
    else:
        return -1  # j → i


def orient_edge_vertical_with_ownership(
    i: int,
    j: int,
    fed_spn_model,
    local_spns: list,
    feature_maps: dict,
    data_aug: np.ndarray,
    num_samples: int = 500,
    verbose: bool = False,
) -> int:
    """
    Orient edge i--j using ownership-aware strategy for vertical mode.

    Inspired by Seng's ProductOverGroups design:
    - Within-client edges: Use local SPN (more accurate)
    - Cross-client edges: Use global product SPN (captures dependencies)

    Args:
        i, j: Feature indices
        fed_spn_model: Global product SPN
        local_spns: List of local SPNs per client
        feature_maps: {client_id: [feature_indices]} ownership mapping
        data_aug: Full augmented data [X, context]
        num_samples: Number of samples for evaluation
        verbose: Print debug info

    Returns:
        1 if i→j, -1 if j→i
    """
    import torch

    # Determine ownership
    client_i = None
    client_j = None

    for client_id, features in feature_maps.items():
        if i in features:
            client_i = client_id
        if j in features:
            client_j = client_id

    if client_i is None or client_j is None:
        # Fallback to global if ownership unclear
        if verbose:
            print(f"  Warning: Could not determine ownership for edge {i}--{j}")
        return orient_edge_likelihood_based(
            i, j, fed_spn_model, data_aug, num_samples, verbose=False
        )

    if client_i == client_j:
        # Within-client edge: use local SPN
        if verbose:
            print(f"  Edge {i}--{j}: within-client (client {client_i})")

        local_spn = local_spns[client_i]

        # Get client's features for proper indexing
        client_features = feature_maps[client_i]

        # Map global indices to local indices
        local_i = client_features.index(i)
        local_j = client_features.index(j)

        # Use local SPN to compare conditionals
        score_i_to_j = compute_conditional_local(
            local_j, [local_i], local_spn, data_aug, client_features, num_samples
        )
        score_j_to_i = compute_conditional_local(
            local_i, [local_j], local_spn, data_aug, client_features, num_samples
        )

        if verbose:
            print(
                f"    Local SPN: log P(j|i)={score_i_to_j:.3f}, log P(i|j)={score_j_to_i:.3f}"
            )

    else:
        # Cross-client edge: Cannot use ProductOverGroups with conditional masking in vertical mode
        #
        # Problem:
        # - ProductOverGroups uses feature extraction (use_nan_masking=False)
        # - Conditional via masking [val, NaN, ...] doesn't work with feature extraction
        # - Each client SPN extracts its features, sees incorrect partially-masked data
        #
        # Solution: Use ANM-based (Additive Noise Model) residual scoring
        # - Fit i→j: compute residual var(j - f(i))
        # - Fit j→i: compute residual var(i - f(j))
        # - Lower residual variance suggests correct causal direction
        if verbose:
            print(
                f"  Edge {i}--{j}: cross-client (client {client_i} → client {client_j})"
            )

        # Use ANM residual scoring for cross-client orientation
        score_i_to_j = compute_anm_score(j, [i], data_aug, num_samples)
        score_j_to_i = compute_anm_score(i, [j], data_aug, num_samples)

        if verbose:
            print(f"    ANM residuals: j~i={score_i_to_j:.3f}, i~j={score_j_to_i:.3f}")

    # Orient towards higher conditional likelihood
    if score_i_to_j > score_j_to_i:
        return 1  # i → j
    else:
        return -1  # j → i


def compute_conditional_local(
    target_local_idx: int,
    parent_local_indices: list,
    local_spn,
    data_aug: np.ndarray,
    client_features: list,
    num_samples: int = 500,
) -> float:
    """
    Compute log P(target | parents) using LOCAL SPN.

    Args:
        target_local_idx: Target feature index in LOCAL space (0 to d_client-1)
        parent_local_indices: Parent indices in LOCAL space
        local_spn: Client's local SPN (LocalClusterMixture)
        data_aug: Full augmented data [X, context]
        client_features: List of global feature indices owned by this client
        num_samples: Number of samples to use

    Returns:
        Average log P(target | parents)
    """
    import torch

    n = data_aug.shape[0]

    # Sample subset
    if n > num_samples:
        indices = np.random.choice(n, size=num_samples, replace=False)
        data_subset = data_aug[indices]
    else:
        data_subset = data_aug

    # Extract client's features from global data
    X_client = data_subset[:, client_features]  # Shape: (num_samples, d_client)

    d_client = len(client_features)

    # Create masked data for marginalization
    # Compute P(target, parents)
    joint_indices = [target_local_idx] + parent_local_indices
    masked_joint = np.full((len(X_client), d_client), np.nan)
    masked_joint[:, joint_indices] = X_client[:, joint_indices]

    with torch.no_grad():
        masked_joint_t = torch.from_numpy(masked_joint).float()
        if hasattr(local_spn, "device"):
            masked_joint_t = masked_joint_t.to(local_spn.device)
        log_prob_joint = local_spn.log_prob(masked_joint_t)
        ll_joint = log_prob_joint.cpu().numpy().flatten()

    # Compute P(parents)
    if len(parent_local_indices) > 0:
        masked_parents = np.full((len(X_client), d_client), np.nan)
        masked_parents[:, parent_local_indices] = X_client[:, parent_local_indices]

        with torch.no_grad():
            masked_parents_t = torch.from_numpy(masked_parents).float()
            if hasattr(local_spn, "device"):
                masked_parents_t = masked_parents_t.to(local_spn.device)
            log_prob_parents = local_spn.log_prob(masked_parents_t)
            ll_parents = log_prob_parents.cpu().numpy().flatten()
    else:
        ll_parents = np.zeros_like(ll_joint)

    # log P(target | parents) = log P(target, parents) - log P(parents)
    log_cond = ll_joint - ll_parents
    log_cond = log_cond[np.isfinite(log_cond)]

    if len(log_cond) == 0:
        return 0.0

    return np.mean(log_cond)


def compute_conditional_global(
    target_idx: int,
    parent_indices: list,
    fed_spn_model,
    data_aug: np.ndarray,
    num_samples: int = 500,
) -> float:
    """
    Compute log P(target | parents) using GLOBAL product SPN.

    WARNING: This function uses NaN masking which is INCOMPATIBLE with
    ProductOverGroups in vertical mode (use_nan_masking=False).

    For vertical mode cross-client edges, use compute_conditional_global_via_marginals instead.

    Similar to compute_conditional_local but uses global indices.
    """
    import torch

    n = data_aug.shape[0]
    d = data_aug.shape[1]

    # Sample subset
    if n > num_samples:
        indices = np.random.choice(n, size=num_samples, replace=False)
        data_subset = data_aug[indices]
    else:
        data_subset = data_aug

    # Compute P(target, parents) - joint probability
    joint_indices = [target_idx] + parent_indices
    masked_joint = np.full((len(data_subset), d), np.nan)
    masked_joint[:, joint_indices] = data_subset[:, joint_indices]

    # Keep context column if present
    if d > max(joint_indices):
        context_idx = d - 1
        masked_joint[:, context_idx] = data_subset[:, context_idx]

    with torch.no_grad():
        masked_joint_t = torch.from_numpy(masked_joint).float()
        if hasattr(fed_spn_model, "device"):
            masked_joint_t = masked_joint_t.to(fed_spn_model.device)
        log_prob_joint = fed_spn_model.log_prob(masked_joint_t)
        ll_joint = log_prob_joint.cpu().numpy().flatten()

    # Compute P(parents) - marginal
    if len(parent_indices) > 0:
        masked_parents = np.full((len(data_subset), d), np.nan)
        masked_parents[:, parent_indices] = data_subset[:, parent_indices]

        if d > max(parent_indices):
            masked_parents[:, context_idx] = data_subset[:, context_idx]

        with torch.no_grad():
            masked_parents_t = torch.from_numpy(masked_parents).float()
            if hasattr(fed_spn_model, "device"):
                masked_parents_t = masked_parents_t.to(fed_spn_model.device)
            log_prob_parents = fed_spn_model.log_prob(masked_parents_t)
            ll_parents = log_prob_parents.cpu().numpy().flatten()
    else:
        ll_parents = np.zeros_like(ll_joint)

    # log P(target | parents) = log P(target, parents) - log P(parents)
    log_cond = ll_joint - ll_parents
    log_cond = log_cond[np.isfinite(log_cond)]

    if len(log_cond) == 0:
        return 0.0

    return np.mean(log_cond)


def compute_anm_score(
    target_idx: int,
    parent_indices: list,
    data_aug: np.ndarray,
    num_samples: int = 500,
) -> float:
    """
    Compute ANM (Additive Noise Model) score for causal direction.

    ANM Assumption: If X → Y is correct, then Y = f(X) + ε where ε ⊥ X

    Key Insight:
        - Correct direction: residuals should be INDEPENDENT of the cause
        - Wrong direction: residuals will be DEPENDENT on the effect

    We test independence using HSIC (Hilbert-Schmidt Independence Criterion):
        score = -HSIC(residuals, X)
        Higher score (closer to 0) = more independent = likely correct direction

    Args:
        target_idx: Target variable index
        parent_indices: Parent variable indices (typically single parent)
        data_aug: Full data including context column
        num_samples: Number of samples to use

    Returns:
        Negative HSIC score (higher = more independent residuals = better)
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform

    n = data_aug.shape[0]

    # Sample subset
    if n > num_samples:
        indices = np.random.choice(n, size=num_samples, replace=False)
        data_subset = data_aug[indices]
    else:
        data_subset = data_aug

    # Extract features (exclude context column if present)
    if len(parent_indices) == 0:
        return 0.0

    parent_idx = parent_indices[0]

    # Get X and Y
    X = data_subset[:, parent_idx].reshape(-1, 1)
    Y = data_subset[:, target_idx].reshape(-1, 1)

    # Standardize for numerical stability
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y)

    try:
        # Fit non-linear model: Y = f(X) + ε
        model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8,
        )
        model.fit(X_scaled, Y_scaled.ravel())

        # Compute residuals
        Y_pred = model.predict(X_scaled)
        residuals = (Y_scaled.ravel() - Y_pred).reshape(-1, 1)

        # Test independence: HSIC(residuals, X)
        # Use RBF kernel for both
        def rbf_kernel(X, gamma=1.0):
            """Compute RBF kernel matrix."""
            pairwise_sq_dists = squareform(pdist(X, "sqeuclidean"))
            return np.exp(-gamma * pairwise_sq_dists)

        # Compute kernel matrices
        K_res = rbf_kernel(residuals, gamma=1.0)
        K_x = rbf_kernel(X_scaled, gamma=1.0)

        # Center the kernel matrices
        n_samples = K_res.shape[0]
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        K_res_c = H @ K_res @ H
        K_x_c = H @ K_x @ H

        # HSIC = (1/(n-1)²) * trace(K_res_c @ K_x_c)
        hsic = np.trace(K_res_c @ K_x_c) / ((n_samples - 1) ** 2)

        # Return negative HSIC (higher score = more independent = better)
        return -hsic

    except Exception as e:
        # Fallback: use correlation-based independence test
        corr = np.corrcoef(X.ravel(), Y.ravel())[0, 1]
        # Return squared correlation as rough independence measure
        return -abs(corr) ** 2
