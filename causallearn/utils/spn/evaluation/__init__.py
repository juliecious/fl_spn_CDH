"""
SPN quality evaluation and visualization utilities.

This package provides metrics, visualization, and dashboard generation
for evaluating SPN quality.
"""

from .metrics import (
    compute_mmd_squared,
    mmd_permutation_test,
    ks_test_dimensions,
    evaluate_spn_quality,
    evaluate_spn_independence_structure,
    log_spn_quality,
    log_independence_structure_results,
    extract_spn_samples,
    create_simple_umap_plot,
)
from .visualization import (
    create_global_umap_visualization,
    create_comparison_umap_overlay,
    create_cross_client_dependency_umap,
    create_umap_comparison_report,
    assign_ownership_labels,
    check_dependencies,
)
from .dashboard import (
    compute_summary_statistics,
    create_summary_dashboard,
    generate_html_report,
    get_quality_rating,
    THRESHOLDS,
)

__all__ = [
    # Metrics
    "compute_mmd_squared",
    "mmd_permutation_test",
    "ks_test_dimensions",
    "evaluate_spn_quality",
    "evaluate_spn_independence_structure",
    "log_spn_quality",
    "log_independence_structure_results",
    "extract_spn_samples",
    # Visualization
    "create_simple_umap_plot",
    "create_global_umap_visualization",
    "create_comparison_umap_overlay",
    "create_cross_client_dependency_umap",
    "create_umap_comparison_report",
    "assign_ownership_labels",
    "check_dependencies",
    # Dashboard
    "compute_summary_statistics",
    "create_summary_dashboard",
    "generate_html_report",
    "get_quality_rating",
    "THRESHOLDS",
]
