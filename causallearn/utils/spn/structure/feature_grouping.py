"""
Feature Grouping Utilities for Federated SPNs.

This module provides utilities for building feature indicator matrices
and grouping features by ownership patterns.
"""

import numpy as np
from typing import Dict, List, Tuple


def build_feature_indicator_matrix(X_splits, scenario, d_features=None):
    """
    Build indicator matrix M where M[k,j] = 1 if client k has feature j.

    Reference: Seng et al. (2025), Algorithm 1, Line 1 (implicit)

    Design Rationale:
        - Indicator matrix is the foundation for automatic feature grouping
        - Column j represents which clients have feature j
        - Used by group_features_by_client_set() to create feature subspaces

    Args:
        X_splits (List[np.ndarray]): Data splits per client
            - Horizontal: [n_k, d] - all have same features
            - Vertical: [n, d_k] - different features per client
            - Hybrid: varies
        scenario (str): 'horizontal', 'vertical', or 'hybrid'
        d_features (int): Total number of features (required for vertical/hybrid)

    Returns:
        M (np.ndarray): [K, d] binary indicator matrix
        feature_names (List[int]): Feature indices [0, 1, ..., d-1]

    Example (Vertical):
        >>> X_splits = [
        ...     np.random.randn(100, 3),  # Client 0: features [0,1,2]
        ...     np.random.randn(100, 2)   # Client 1: features [3,4]
        ... ]
        >>> M, names = build_feature_indicator_matrix(X_splits, 'vertical', d_features=5)
        >>> M
        array([[1, 1, 1, 0, 0],
               [0, 0, 0, 1, 1]])
    """
    K = len(X_splits)

    # Justification: Need total dimensionality to allocate M
    # For vertical/hybrid, must be provided explicitly
    if scenario == "horizontal":
        # Horizontal: all clients have same features
        d = X_splits[0].shape[1]
    else:
        # Vertical/Hybrid: must provide d_features
        if d_features is None:
            raise ValueError(
                f"d_features required for scenario='{scenario}'. "
                f"Provide total feature count."
            )
        d = d_features

    # Initialize indicator matrix
    # Justification: Binary matrix, start with all zeros
    M = np.zeros((K, d), dtype=int)

    if scenario == "horizontal":
        # Horizontal: All clients have all features
        # Justification: Same features across clients
        M[:, :] = 1  # All entries are 1

    elif scenario == "vertical":
        # Vertical: Auto-split features equally across clients
        # Justification: Standard vertical FL assumes equal split
        cols_per_client = np.array_split(range(d), K)
        for k in range(K):
            feature_indices = cols_per_client[k].tolist()
            M[k, feature_indices] = 1

    elif scenario == "hybrid":
        # Hybrid: Create overlapping feature splits
        # Justification: Hybrid FL has overlapping features across clients
        # Strategy: Each client gets base features + overlap with neighbors
        # This validates ProductOverGroupsWithOverlap architecture

        base_size = d // K
        overlap_size = max(1, d // (2 * K))  # ~15-20% overlap

        feature_sets = []
        for k in range(K):
            start = k * base_size
            end = min((k + 1) * base_size + overlap_size, d)
            features = list(range(start, end))
            feature_sets.append(features)

        # Ensure all features are covered
        all_features = set()
        for features in feature_sets:
            all_features.update(features)

        missing = set(range(d)) - all_features
        if missing:
            # Add missing features to last client
            feature_sets[-1].extend(sorted(missing))

        # Build indicator matrix
        for k in range(K):
            M[k, feature_sets[k]] = 1

        # Log overlap statistics
        total_refs = sum(len(fs) for fs in feature_sets)
        overlap_count = total_refs - d
        logging.info(
            f"Hybrid mode: Created overlapping feature splits with {overlap_count} overlaps"
        )
        for k, features in enumerate(feature_sets):
            logging.info(
                f"  Client {k}: features {features[:5]}...{features[-2:]} (size={len(features)})"
            )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    feature_names = list(range(d))
    return M, feature_names


def group_features_by_client_set(M, feature_names):
    """
    Group features by which clients have them (Algorithm 1 from Seng et al. 2025).

    Reference: Seng et al. (2025), Algorithm 1, Lines 3-6

    Algorithm:
        1. For each feature j: extract column M[:, j] (which clients have it)
        2. Group features with identical column patterns
        3. Convert column pattern to client set tuple
        4. Return mapping: client_set → [features]

    Mathematical Insight:
        Features with same column pattern M[:, j] share the same client set.
        These features should be modeled by a single GroupMixture over those clients.

    Design Rationale:
        - Automatic: No manual specification needed
        - Disjoint: Each feature appears in exactly one group (by construction)
        - Handles overlaps: If feature j appears in multiple clients,
          it gets grouped with other features having the same client set

    Args:
        M (np.ndarray): [K, d] indicator matrix
        feature_names (List[int]): Feature indices

    Returns:
        feature_subspaces (Dict): {
            (client_tuple): [feature_indices]
        }

    Example 1 (Vertical - Disjoint):
        >>> M = np.array([
        ...     [1, 1, 1, 0, 0],
        ...     [0, 0, 0, 1, 1]
        ... ])
        >>> group_features_by_client_set(M, list(range(5)))
        {
            (0,): [0, 1, 2],  # Features 0,1,2 only on client 0
            (1,): [3, 4]      # Features 3,4 only on client 1
        }

    Example 2 (Hybrid - Overlapping):
        >>> M = np.array([
        ...     [1, 1, 1, 0],
        ...     [0, 1, 1, 1]
        ... ])
        >>> group_features_by_client_set(M, list(range(4)))
        {
            (0,): [0],        # Feature 0 only on client 0
            (0, 1): [1, 2],   # Features 1,2 on BOTH clients (overlap!)
            (1,): [3]         # Feature 3 only on client 1
        }

        Result: GroupMixture for (0,1) combines both clients for features [1,2]
    """
    K, d = M.shape

    # Step 1: Find distinct column patterns
    # Justification: Features with same column pattern share same client set
    feature_to_clients = {}
    for j in range(d):
        # Extract column j (which clients have feature j)
        col = tuple(M[:, j])  # e.g., (1, 0, 1) means clients 0 and 2 have it

        if col not in feature_to_clients:
            feature_to_clients[col] = []
        feature_to_clients[col].append(feature_names[j])

    # Step 2: Convert column patterns to client sets
    # Justification: Client set is more interpretable than binary pattern
    feature_subspaces = {}
    for col_pattern, features in feature_to_clients.items():
        # col_pattern is (1, 0, 1, ...) → client_set is (0, 2, ...)
        # Justification: Extract indices where col_pattern[k] == 1
        client_set = tuple(k for k in range(K) if col_pattern[k] == 1)

        # Skip invalid patterns (no clients have these features)
        # Justification: Features must belong to at least one client
        if len(client_set) == 0:
            logging.warning(f"Features {features} have no clients! Skipping.")
            continue

        feature_subspaces[client_set] = features

    # Log results for debugging
    logging.info(
        f"[FedPC] Automatic feature grouping: {len(feature_subspaces)} subspaces"
    )
    for clients, features in feature_subspaces.items():
        logging.info(f"  Clients {clients} share features {features}")

    return feature_subspaces
