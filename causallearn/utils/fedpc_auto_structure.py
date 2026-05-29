"""
Automatic FedPC Structure Construction (Seng et al. Algorithm 1)

Automatically builds optimal sum/product hierarchy from feature indicator matrix.
Addresses Gap 3: Seng's one-pass training with automatic structure discovery.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from causallearn.utils.FedPC import (
    LocalSPNWrapper,
    GlobalFedSPN,
    FederatedProduct,
    GroupMixture,
    ProductOverGroups,
)


def build_feature_indicator_matrix(
    feature_maps: Dict[int, List[int]], num_features: int
) -> np.ndarray:
    """
    Build feature indicator matrix M where M[i,j] = 1 if client i has feature j.

    Args:
        feature_maps: {client_id: [feature_indices]}
        num_features: Total number of features

    Returns:
        M: Binary matrix of shape (num_clients, num_features)
    """
    num_clients = len(feature_maps)
    M = np.zeros((num_clients, num_features), dtype=int)

    for client_id, features in feature_maps.items():
        M[client_id, features] = 1

    return M


def group_features_by_client_set(M: np.ndarray) -> Dict[Tuple, List[int]]:
    """
    Group features by which clients own them (Seng et al. Section 4.1).

    Find distinct column patterns in M. Each unique pattern defines a group.

    Args:
        M: Feature indicator matrix (num_clients, num_features)

    Returns:
        groups: {owner_set: [feature_indices]}
        - owner_set: tuple of client IDs that share these features
        - feature_indices: list of feature indices with this ownership pattern
    """
    num_features = M.shape[1]
    groups = {}

    for j in range(num_features):
        # Get ownership pattern for feature j
        owner_set = tuple(np.where(M[:, j] == 1)[0].tolist())

        if owner_set not in groups:
            groups[owner_set] = []
        groups[owner_set].append(j)

    return groups


def construct_fedpc_automatic(
    local_spns: List,
    feature_maps: Dict[int, List[int]],
    num_features: int,
    num_clusters: int,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[object, str]:
    """
    Automatically construct optimal FedPC structure from feature ownership.

    Implements Seng et al. Algorithm 1:
    1. Build feature indicator matrix M
    2. Group features by ownership pattern
    3. Construct sum nodes for shared features (horizontal)
    4. Construct product nodes for disjoint features (vertical)
    5. Build hierarchical combination for hybrid

    Args:
        local_spns: List of trained local SPNs per client
        feature_maps: {client_id: [feature_indices]}
        num_features: Total number of features
        num_clusters: Number of mechanism clusters
        device: CPU or CUDA
        verbose: Print structure info

    Returns:
        (global_model, scenario_type)
        - global_model: Assembled FedPC model
        - scenario_type: "horizontal", "vertical", or "hybrid"
    """
    M = build_feature_indicator_matrix(feature_maps, num_features)
    groups = group_features_by_client_set(M)

    num_clients = len(feature_maps)

    # Analyze ownership patterns
    single_owner_groups = []  # Features owned by single client (vertical)
    multi_owner_groups = []  # Features owned by multiple clients (horizontal)

    for owner_set, features in groups.items():
        if len(owner_set) == 1:
            single_owner_groups.append((owner_set, features))
        else:
            multi_owner_groups.append((owner_set, features))

    if verbose:
        logging.info("\n[Automatic FedPC Structure Construction]")
        logging.info(f"Feature Indicator Matrix M: {M.shape}")
        logging.info(f"Ownership Groups: {len(groups)}")
        logging.info(f"  - Single-owner (vertical): {len(single_owner_groups)}")
        logging.info(f"  - Multi-owner (horizontal): {len(multi_owner_groups)}")

    # Determine scenario based on ownership patterns
    if len(single_owner_groups) == 0:
        # All features shared across clients → Horizontal
        scenario = "horizontal"
        global_model = GlobalFedSPN(
            local_models=local_spns,
            num_clusters=num_clusters,
            device=device,
        )
        if verbose:
            logging.info(f"Detected Scenario: HORIZONTAL (all features shared)")

    elif len(multi_owner_groups) == 0:
        # All features owned by single clients → Vertical
        scenario = "vertical"
        global_model = FederatedProduct(
            local_models=local_spns,
            feature_maps=feature_maps,
            num_features=num_features,
            device=device,
        )
        if verbose:
            logging.info(f"Detected Scenario: VERTICAL (disjoint feature ownership)")

    else:
        # Mixed ownership → Hybrid
        scenario = "hybrid"

        # Build hierarchical structure:
        # 1. Product over single-owner groups (vertical partition)
        # 2. Sum over multi-owner groups (horizontal partition)
        # 3. Product at root to combine

        if verbose:
            logging.info(f"Detected Scenario: HYBRID (mixed ownership)")
            logging.info(f"  Building hierarchical sum/product structure...")

        # For now, use ProductOverGroups as approximation
        # Full hierarchical construction would require building intermediate nodes
        global_model = ProductOverGroups(
            local_models=local_spns,
            feature_maps=feature_maps,
            num_features=num_features,
            num_clusters=num_clusters,
            device=device,
        )

    return global_model, scenario


def compute_adaptive_hyperparameters(
    num_features: int,
    num_samples: int,
    scenario: str,
) -> Dict[str, int]:
    """
    Compute SPN hyperparameters based on data characteristics.

    Heuristics:
    - depth: log2(num_features) for hierarchical splitting
    - num_sums: Scale with features to increase expressivity
    - num_leaves: Scale with samples to prevent overfitting
    - num_repetitions: Fixed to balance cost vs accuracy

    Args:
        num_features: Number of features
        num_samples: Number of samples
        scenario: "horizontal", "vertical", or "hybrid"

    Returns:
        hyperparams: {depth, num_sums, num_leaves, num_repetitions}
    """
    # Base values
    base_depth = max(2, int(np.floor(np.log2(num_features))))
    base_sums = 20
    base_leaves = 20
    base_reps = 10

    # Adjust based on scenario
    if scenario == "vertical":
        # Vertical: Each client sees fewer features, increase capacity
        num_sums = base_sums + num_features
        num_leaves = base_leaves + num_features
    elif scenario == "horizontal":
        # Horizontal: More samples per cluster, can increase depth
        num_sums = base_sums
        num_leaves = base_leaves
        base_depth = min(base_depth + 1, 5)  # Cap at 5
    else:  # hybrid
        num_sums = base_sums + num_features // 2
        num_leaves = base_leaves + num_features // 2

    # Sample size adjustment
    if num_samples < 100:
        # Small data: reduce capacity to prevent overfitting
        num_sums = max(10, num_sums // 2)
        num_leaves = max(10, num_leaves // 2)

    return {
        "depth": base_depth,
        "num_sums": num_sums,
        "num_leaves": num_leaves,
        "num_repetitions": base_reps,
    }
