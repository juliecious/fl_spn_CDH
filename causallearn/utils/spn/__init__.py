"""
Sum-Product Network (SPN) Library for Federated Causal Discovery.

This package provides a modular SPN implementation supporting:
- Local SPN training (single client)
- Federated SPN aggregation (horizontal, vertical, hybrid)
- Automatic structure learning (Gap 3)
- Quality evaluation and visualization

Public API:
    Core SPNs:
        - LocalSPNWrapper: Single-client Gaussian SPN
        - LocalClusterMixture: Mixture of local SPNs
        - UnivariateSPNWrapper: 1D Gaussian mixture
        - EnsembleSPNWrapper: Ensemble of SPNs

    Federated SPNs:
        - GlobalFedSPN: Horizontal federation (mixture)
        - FederatedProduct: Vertical federation (product)
        - ProductOverGroups: Vertical with feature groups
        - GroupMixture: Mixture for overlapping features
        - GlobalSumOfProducts: Hybrid federation (sum-of-products)
        - ProductOverGroupsWithOverlap: Hybrid with overlaps
        - FederatedProductWithClusters: Gap 4 cluster-conditional

    Structure Learning:
        - construct_fedpc_automatic: Gap 3 automatic structure
        - build_feature_indicator_matrix: Feature ownership matrix
        - group_features_by_client_set: Group features by ownership
        - compute_adaptive_hyperparameters: Adaptive SPN config

    Evaluation:
        - compute_mmd_squared: Maximum Mean Discrepancy
        - compute_ks_tests: Kolmogorov-Smirnov tests
        - generate_umap_visualization: UMAP quality plots
        - generate_spn_dashboard: HTML quality dashboard

Example:
    >>> from causallearn.utils.spn import LocalSPNWrapper, GlobalFedSPN
    >>> # Train local SPNs
    >>> local_spns = [LocalSPNWrapper(num_features=d) for _ in range(K)]
    >>> for spn, X_client in zip(local_spns, X_splits):
    ...     spn.fit(X_client)
    >>> # Aggregate horizontally
    >>> global_spn = GlobalFedSPN(local_spns, weights=[1/K]*K)
    >>> # Query density
    >>> log_prob = global_spn.log_prob(X_test)
"""

# Core SPNs (will be populated after refactoring)
from causallearn.utils.FedPC import (
    LocalSPNWrapper,
    LocalClusterMixture,
    UnivariateSPNWrapper,
    EnsembleSPNWrapper,
    FederatedStructureLearner,
)

# Federated SPNs
from causallearn.utils.FedPC import (
    GlobalFedSPN,
    FederatedProduct,
    ProductOverGroups,
    GroupMixture,
    ProductOverGroupsWithOverlap,
    GlobalSumOfProducts,
    FederatedProductWithClusters,
)

# Structure learning
from causallearn.utils.FedPC import (
    build_feature_indicator_matrix,
    group_features_by_client_set,
    sample_cluster_combinations,
    compute_adaptive_hyperparameters,
)
from causallearn.utils.fedpc_auto_structure import construct_fedpc_automatic

# Evaluation
from causallearn.utils.spn_evaluation import (
    compute_mmd_squared,
    compute_ks_tests,
    compute_train_log_likelihood,
    evaluate_spn_quality,
)
from causallearn.utils.spn_umap_visualization import generate_umap_visualization
from causallearn.utils.spn_dashboard import generate_spn_dashboard

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
    "construct_fedpc_automatic",
    "build_feature_indicator_matrix",
    "group_features_by_client_set",
    "sample_cluster_combinations",
    "compute_adaptive_hyperparameters",
    # Evaluation
    "compute_mmd_squared",
    "compute_ks_tests",
    "compute_train_log_likelihood",
    "evaluate_spn_quality",
    "generate_umap_visualization",
    "generate_spn_dashboard",
]
