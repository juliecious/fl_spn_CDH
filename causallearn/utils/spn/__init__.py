"""
Federated Sum-Product Networks (SPNs) for Causal Discovery.

This package provides modular implementations of federated SPNs for
horizontal, vertical, and hybrid federated learning scenarios.

Public API:
-----------

Core SPNs:
    - LocalSPNWrapper: Basic SPN wrapper for local training
    - LocalClusterMixture: Mixture of cluster-specific SPNs
    - UnivariateSPNWrapper: 1D Gaussian mixture model
    - EnsembleSPNWrapper: Ensemble of SPNs for variance reduction
    - FederatedStructureLearner: Dependency-aware structure learning

Federated SPNs:
    - GlobalFedSPN: Horizontal federated SPN
    - FederatedProduct: Vertical federated SPN (product over clients)
    - ProductOverGroups: Vertical SPN with feature grouping
    - GroupMixture: Mixture model for feature groups
    - ProductOverGroupsWithOverlap: Hybrid SPN with overlapping features
    - GlobalSumOfProducts: Hybrid SPN with sum-of-products structure
    - FederatedProductWithClusters: Cluster-conditional vertical SPN (Gap 4)

Structure Learning:
    - build_feature_indicator_matrix: Build feature ownership matrix
    - group_features_by_client_set: Group features by ownership
    - compute_adaptive_hyperparameters: Adaptive SPN configuration
    - sample_cluster_combinations: Sample cluster assignments
    - construct_fedpc_automatic: Automatic structure construction
    - auto_tune_spn_config: Optuna-based hyperparameter tuning

Evaluation:
    - compute_mmd_squared: Maximum Mean Discrepancy test
    - mmd_permutation_test: MMD with permutation testing
    - ks_test_dimensions: Kolmogorov-Smirnov test per dimension
    - evaluate_spn_quality: Comprehensive quality evaluation
    - evaluate_spn_independence_structure: CI structure evaluation
    - create_simple_umap_plot: Basic UMAP visualization
    - create_global_umap_visualization: Multi-model UMAP comparison
    - create_umap_comparison_report: Complete UMAP analysis report
    - compute_summary_statistics: Aggregate metrics across SPNs
    - create_summary_dashboard: Visual quality dashboard
    - generate_html_report: HTML report generation

Usage:
------
    from causallearn.utils.spn import LocalSPNWrapper, GlobalFedSPN
    from causallearn.utils.spn import evaluate_spn_quality

    # Train local SPN
    local_spn = LocalSPNWrapper(num_features=8, device='cpu')
    local_spn.train_local(data, epochs=50)

    # Evaluate quality
    results = evaluate_spn_quality(local_spn, data, n_samples=200)
"""

# Core SPNs
from .core import (
    LocalSPNWrapper,
    LocalClusterMixture,
    UnivariateSPNWrapper,
    EnsembleSPNWrapper,
    FederatedStructureLearner,
)

# Federated SPNs
from .federated import (
    GlobalFedSPN,
    FederatedProduct,
    ProductOverGroups,
    GroupMixture,
    ProductOverGroupsWithOverlap,
    GlobalSumOfProducts,
    FederatedProductWithClusters,
)

# Structure Learning
from .structure import (
    build_feature_indicator_matrix,
    group_features_by_client_set,
    compute_adaptive_hyperparameters,
    sample_cluster_combinations,
    construct_fedpc_automatic,
    auto_tune_spn_config,
)

# Evaluation
from .evaluation import (
    compute_mmd_squared,
    mmd_permutation_test,
    ks_test_dimensions,
    evaluate_spn_quality,
    evaluate_spn_independence_structure,
    log_spn_quality,
    log_independence_structure_results,
    extract_spn_samples,
    create_simple_umap_plot,
    create_global_umap_visualization,
    create_comparison_umap_overlay,
    create_cross_client_dependency_umap,
    create_umap_comparison_report,
    assign_ownership_labels,
    check_dependencies,
    compute_summary_statistics,
    create_summary_dashboard,
    generate_html_report,
    get_quality_rating,
    THRESHOLDS,
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
    # Evaluation
    "compute_mmd_squared",
    "mmd_permutation_test",
    "ks_test_dimensions",
    "evaluate_spn_quality",
    "evaluate_spn_independence_structure",
    "log_spn_quality",
    "log_independence_structure_results",
    "extract_spn_samples",
    "create_simple_umap_plot",
    "create_global_umap_visualization",
    "create_comparison_umap_overlay",
    "create_cross_client_dependency_umap",
    "create_umap_comparison_report",
    "assign_ownership_labels",
    "check_dependencies",
    "compute_summary_statistics",
    "create_summary_dashboard",
    "generate_html_report",
    "get_quality_rating",
    "THRESHOLDS",
]
