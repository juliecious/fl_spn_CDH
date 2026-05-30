"""
Federated Structure Learning for SPNs.

This module provides structure learning capabilities for federated SPNs.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


class FederatedStructureLearner:
    """
    Learns a global variable dependency structure from federated metadata.
    Goal: Find an ordering of variables that places dependent variables close together,
    improving the effectiveness of the SPN's internal splitting rules.
    """

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.local_correlations = []

    def add_local_metadata(self, data: np.ndarray):
        """Clients call this to share their local correlation matrix."""
        # Use Spearman Rank Correlation to capture non-linear (monotonic) dependencies
        # This is more robust for SPN structure learning than Pearson correlation
        corr, _ = spearmanr(data, axis=0)

        # Take absolute value as we only care about dependency strength
        corr = np.abs(corr)

        # Handle NaNs (e.g. constant features)
        corr = np.nan_to_num(corr, nan=0.0)
        self.local_correlations.append(corr)

    def get_causal_order(self) -> np.ndarray:
        """
        Aggregates correlations and returns a dependency-aware variable ordering.
        Uses Hierarchical Clustering to group dependent variables.
        """
        if not self.local_correlations:
            return np.arange(self.num_features)

        # 1. Aggregate: Global Mean Dependency Matrix
        global_corr = np.mean(self.local_correlations, axis=0)

        # 2. Convert to Distance Matrix (1 - corr)
        # We want highly correlated variables to have low distance
        dist_matrix = 1.0 - global_corr
        np.fill_diagonal(dist_matrix, 0)

        # 3. Hierarchical Clustering (Ward's Method)
        # squareform converts the matrix to the compressed distance vector expected by linkage
        try:
            Z = linkage(squareform(dist_matrix, checks=False), method="ward")
            # 4. Extract Optimal Leaf Ordering
            order = leaves_list(Z)
            return order
        except Exception:
            # Fallback to identity order if clustering fails (e.g. too few features)
            return np.arange(self.num_features)
