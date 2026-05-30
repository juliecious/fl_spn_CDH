"""
Backwards compatibility shim for FedPC.py

This module re-exports all classes from the new causallearn.utils.spn structure.
Direct imports from FedPC.py are deprecated and will be removed in a future version.

DEPRECATED: Import from causallearn.utils.spn instead.

Example:
    # Old (deprecated):
    from causallearn.utils.FedPC import LocalSPNWrapper, GlobalFedSPN

    # New (recommended):
    from causallearn.utils.spn import LocalSPNWrapper, GlobalFedSPN
"""
import warnings

warnings.warn(
    "Importing from causallearn.utils.FedPC is deprecated. "
    "Use 'from causallearn.utils.spn import ...' instead. "
    "The FedPC module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from new structure for backwards compatibility
from causallearn.utils.spn import *  # noqa: F401, F403

# Explicitly re-export all classes to maintain compatibility
from causallearn.utils.spn import (
    # Core
    LocalSPNWrapper,
    LocalClusterMixture,
    UnivariateSPNWrapper,
    EnsembleSPNWrapper,
    FederatedStructureLearner,
    # Federated
    GlobalFedSPN,
    FederatedProduct,
    ProductOverGroups,
    GroupMixture,
    ProductOverGroupsWithOverlap,
    GlobalSumOfProducts,
    FederatedProductWithClusters,
    # Structure
    build_feature_indicator_matrix,
    group_features_by_client_set,
    compute_adaptive_hyperparameters,
    sample_cluster_combinations,
    construct_fedpc_automatic,
    auto_tune_spn_config,
)

__all__ = [
    # Core
    "LocalSPNWrapper",
    "LocalClusterMixture",
    "UnivariateSPNWrapper",
    "EnsembleSPNWrapper",
    "FederatedStructureLearner",
    # Federated
    "GlobalFedSPN",
    "FederatedProduct",
    "ProductOverGroups",
    "GroupMixture",
    "ProductOverGroupsWithOverlap",
    "GlobalSumOfProducts",
    "FederatedProductWithClusters",
    # Structure
    "build_feature_indicator_matrix",
    "group_features_by_client_set",
    "compute_adaptive_hyperparameters",
    "sample_cluster_combinations",
    "construct_fedpc_automatic",
    "auto_tune_spn_config",
]
