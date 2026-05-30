"""
Structure learning utilities for federated SPNs.

This package provides utilities for automatic structure discovery,
feature grouping, and adaptive hyperparameter configuration.
"""

from .feature_grouping import (
    build_feature_indicator_matrix,
    group_features_by_client_set,
)
from .adaptive_config import (
    compute_adaptive_hyperparameters,
    sample_cluster_combinations,
)
from .auto_detection import construct_fedpc_automatic
from .auto_tuning import auto_tune_spn_config

__all__ = [
    "build_feature_indicator_matrix",
    "group_features_by_client_set",
    "compute_adaptive_hyperparameters",
    "sample_cluster_combinations",
    "construct_fedpc_automatic",
    "auto_tune_spn_config",
]
