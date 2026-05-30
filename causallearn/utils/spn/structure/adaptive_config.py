"""
Adaptive Hyperparameter Configuration for Federated SPNs.

This module provides utilities for computing adaptive SPN hyperparameters
based on data characteristics.
"""

import logging
import numpy as np
from typing import Dict, Any


def sample_cluster_combinations(K_clients, K_local, num_samples=10, seed=42):
    """
    Sample cluster combinations for sum-over-products in hybrid mode.

    Following Seng's guidance: "combine these clusters 'randomly' (since pairing
    each cluster from client i with each cluster from client j is too demanding)"

    Strategy:
        - If K_local^K_clients ≤ 20: Enumerate all combinations (exact)
        - Otherwise: Randomly sample num_samples combinations (approximate)

    Mathematical Context:
        Each combination represents a different "factorization" of the data:
        - Combination (0,0,0): All clients use cluster 0
        - Combination (0,1,1): Client 0 uses cluster 0, others use cluster 1
        - etc.

        The sum over these combinations breaks independence between feature groups!

    Args:
        K_clients (int): Number of clients
        K_local (int): Number of local clusters per client
        num_samples (int): How many combinations to sample if not enumerating all
        seed (int): Random seed for reproducibility

    Returns:
        combinations (List[Tuple[int]]): List of cluster configurations
            Each tuple has K_clients elements, specifying which cluster
            each client contributes to this product
        weights (np.ndarray): Uniform weights for each combination (sum to 1)

    Example:
        >>> # For K=3 clients, K_local=2 clusters per client
        >>> combos, weights = sample_cluster_combinations(3, 2)
        >>> # Output: 8 combinations (enumerate all since 2^3 = 8 ≤ 20)
        >>> combos
        [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
        >>> weights
        array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

    Reference:
        Seng feedback (April 29, 2026): "combine these clusters 'randomly'"
    """
    np.random.seed(seed)

    max_combinations = K_local**K_clients

    # Decision: Enumerate if small enough, sample if too large
    if max_combinations <= 20:
        # Enumerate all combinations
        import itertools

        combinations = list(itertools.product(range(K_local), repeat=K_clients))
        logging.info(
            f"[ClusterCombinations] Enumerating all {len(combinations)} combinations "
            f"(K_clients={K_clients}, K_local={K_local})"
        )
    else:
        # Random sampling without replacement
        # If num_samples not specified, use default of min(20, max_combinations)
        if num_samples is None:
            num_samples = min(20, max_combinations)
        else:
            num_samples = min(num_samples, max_combinations)
        combinations = set()

        # Sample unique combinations
        while len(combinations) < num_samples:
            config = tuple(np.random.randint(0, K_local) for _ in range(K_clients))
            combinations.add(config)

        combinations = list(combinations)
        logging.info(
            f"[ClusterCombinations] Sampled {len(combinations)}/{max_combinations} combinations "
            f"(K_clients={K_clients}, K_local={K_local})"
        )

    # Uniform weights (all combinations equally likely)
    # Justification: Seng mentions "randomly", no principled weighting yet
    weights = np.ones(len(combinations)) / len(combinations)

    return combinations, weights


def compute_adaptive_hyperparameters(
    mode: str,
    num_features: int,
    num_samples: int,
    data_type: str,
    base_num_sums: int = 20,
    base_num_leaves: int = 20,
    base_epochs: int = 100,
    base_depth: int = None,
) -> Dict[str, Any]:
    """
    Compute adaptive hyperparameters following 5-criterion system for FedCDH SPNs.

    This implements the documented adaptive capacity scaling system that addresses
    horizontal mode underperformance (F1=0.133-0.255 → target 0.5+).

    The 5 criteria are:
    1. Mode-Specific Base Capacity: Different architectures for horizontal/vertical/hybrid
    2. Sample-to-Feature Ratio Scaling: Adjust capacity based on data availability
    3. Data Type Differentiation: Linear vs nonlinear complexity adjustments
    4. Quality-Aware Epoch Scheduling: Mode and feature-dependent training duration
    5. Mode-Aware Regularization: Dropout and weight decay tuned per scenario

    Args:
        mode: Federated scenario mode ("horizontal", "vertical", "hybrid")
        num_features: Number of features in local data (d_k)
        num_samples: Number of samples in local data (n_k)
        data_type: Data generation type ("linear" or "nonlinear")
        base_num_sums: Base number of sum nodes (default 20)
        base_num_leaves: Base number of leaf nodes (default 20)
        base_epochs: Base training epochs (default 100)
        base_depth: Base tree depth (default None, auto-computed from features)

    Returns:
        Dictionary with keys:
            - num_sums: Adapted number of sum nodes
            - num_leaves: Adapted number of leaf nodes
            - depth: Adapted tree depth
            - epochs: Adapted training epochs
            - dropout: Dropout rate for regularization
            - weight_decay: L2 regularization weight

    Example:
        >>> params = compute_adaptive_hyperparameters(
        ...     mode="horizontal",
        ...     num_features=10,
        ...     num_samples=400,
        ...     data_type="linear"
        ... )
        >>> print(params['num_sums'])  # Expected: 40+ (4×10)
        >>> print(params['dropout'])   # Expected: 0.1 (ratio 40 < 100)

    References:
        - agents/working_state.md: "5-Criterion Adaptive Hyperparameter System"
        - v2 experiment proposal: Horizontal mode performance fix
    """

    # Criterion 1: Mode-Specific Base Capacity
    # Horizontal: Many features need broader representation (4×d)
    # Vertical: Few features need depth (8×d for d>3, otherwise small)
    # Hybrid: Intermediate complexity (6×d)
    if mode == "horizontal":
        base_sums = max(32, 4 * num_features)
        base_leaves = max(16, 2 * num_features)
    elif mode == "vertical":
        if num_features <= 3:
            base_sums = 8
            base_leaves = 8
        else:
            base_sums = 8 * num_features
            base_leaves = 4 * num_features
    else:  # hybrid
        base_sums = 6 * num_features
        base_leaves = 3 * num_features

    # Criterion 2: Sample-to-Feature Ratio Scaling
    # Low ratio (< 50): Risk of overfitting, reduce capacity
    # Medium ratio (50-200): Standard capacity
    # High ratio (> 200): Can afford more capacity for complex patterns
    ratio = num_samples / max(1, num_features)
    if ratio < 50:
        capacity_scale = 0.5
    elif ratio < 100:
        capacity_scale = 0.75
    elif ratio < 200:
        capacity_scale = 1.0
    else:
        capacity_scale = min(1.5, 1.0 + (ratio - 200) / 400)

    final_num_sums = int(base_sums * capacity_scale)
    final_num_leaves = int(base_leaves * capacity_scale)

    # Criterion 3: Data Type Differentiation
    # Nonlinear data needs deeper trees and more training
    # Linear data can use shallower structures
    if base_depth is None:
        base_depth = int(np.floor(np.log2(max(1, num_features))))

    if data_type == "nonlinear":
        depth_bonus = 1
        epoch_multiplier = 1.3
    else:  # linear
        depth_bonus = 0
        epoch_multiplier = 1.0

    final_depth = max(1, base_depth + depth_bonus)

    # Safety constraint: Ensure 2**depth doesn't exceed num_features
    # (simple_einet requires 2**depth <= num_features for tree construction)
    max_safe_depth = int(np.floor(np.log2(max(1, num_features))))
    final_depth = min(final_depth, max_safe_depth)

    # Criterion 4: Quality-Aware Epoch Scheduling
    # More features need more training, especially in horizontal mode
    # Base scaling: d^1.5 (superlinear growth)
    epoch_base_scale = (num_features / 5.0) ** 1.5

    # Mode-specific multipliers
    if mode == "horizontal":
        # Horizontal needs more epochs due to broader feature space
        mode_multiplier = 1.0 + num_features / 30
    elif mode == "vertical":
        # Vertical can train faster (fewer features)
        mode_multiplier = 0.8
    else:  # hybrid
        mode_multiplier = 0.9

    final_epochs = int(
        base_epochs * epoch_base_scale * epoch_multiplier * mode_multiplier
    )
    final_epochs = max(100, min(final_epochs, 500))  # Clamp to [100, 500]

    # Criterion 5: Mode-Aware Regularization
    # Horizontal: Moderate regularization (many features, risk of spurious correlations)
    # Vertical: Higher regularization (few features, risk of overfitting to noise)
    # Hybrid: Light regularization (balanced scenario)
    if mode == "horizontal":
        weight_decay = 1e-4
        dropout = 0.1 if ratio < 100 else 0.0
    elif mode == "vertical":
        weight_decay = 1e-3  # Higher for vertical (fewer features)
        dropout = 0.0  # Vertical typically has enough regularization from structure
    else:  # hybrid
        weight_decay = 5e-5
        dropout = 0.05 if ratio < 100 else 0.0

    return {
        "num_sums": final_num_sums,
        "num_leaves": final_num_leaves,
        "depth": final_depth,
        "epochs": final_epochs,
        "dropout": dropout,
        "weight_decay": weight_decay,
    }
