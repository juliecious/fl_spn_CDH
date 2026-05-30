"""
UMAP Visualization for Vertical Federated SPNs.

This module provides comprehensive UMAP visualizations for analyzing federated SPNs.
"""

"""
Global UMAP Visualization for Vertical Federated SPNs.

This module provides comprehensive UMAP visualizations to analyze:
1. How well vertical SPNs capture cross-client dependencies
2. Whether the GlobalSumOfProducts architecture preserves data structure
3. Comparison of vertical vs horizontal vs centralized SPNs
4. Feature ownership patterns in the embedding space

Rationale (from Seng et al. 2025):
- Vertical FL: Features split across clients, same samples
- GlobalSumOfProducts: Mixture of products to capture dependencies
- Goal: Verify that P(X1, X2) = Σ_l q(L=l) × P(X1|L=l) × P(X2|L=l)
  captures dependencies marginally (not just forcing independence)

Author: Vertical SPN Analysis Module
Date: 2026-05-21
"""

import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import torch

# Conditional imports
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available.")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def check_dependencies():
    """Check if required dependencies are available."""
    if not UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is required. Install with: pip install umap-learn"
        )
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required. Install with: pip install matplotlib"
        )
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        )


def extract_spn_samples(
    spn_model,
    n_samples: int = 500,
    device: str = "cpu",
    has_context: bool = True,
) -> np.ndarray:
    """
    Extract samples from an SPN model.

    Args:
        spn_model: Trained SPN (LocalSPNWrapper or GlobalFedSPN)
        n_samples: Number of samples to generate
        device: torch device
        has_context: Whether to remove context column

    Returns:
        samples: [n_samples, d] numpy array
    """
    try:
        with torch.no_grad():
            samples = spn_model.sample(n_samples).cpu().numpy()

        # Remove context column if present
        if has_context and samples.shape[1] > 1:
            samples = samples[:, :-1]

        return samples
    except Exception as e:
        logging.error(f"Failed to sample from SPN: {e}")
        return None


def create_global_umap_visualization(
    X_real: np.ndarray,
    X_vertical_spn: Optional[np.ndarray] = None,
    X_horizontal_spn: Optional[np.ndarray] = None,
    X_centralized_spn: Optional[np.ndarray] = None,
    feature_maps: Optional[Dict[int, List[int]]] = None,
    save_path: Optional[str] = None,
    title: str = "Global UMAP: Vertical SPN Analysis",
    show_feature_ownership: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Create comprehensive UMAP visualization comparing vertical SPN to baselines.

    This is the MAIN function for analyzing vertical SPNs per Seng et al. (2025).

    Args:
        X_real: Real data [n, d]
        X_vertical_spn: Samples from vertical SPN [m, d]
        X_horizontal_spn: Samples from horizontal SPN [m, d] (optional baseline)
        X_centralized_spn: Samples from centralized SPN [m, d] (optional baseline)
        feature_maps: Dict mapping client_id -> [feature_indices] (for coloring)
        save_path: Path to save figure
        title: Main title
        show_feature_ownership: Color points by feature ownership pattern

    Returns:
        embeddings: Dict of UMAP embeddings for each data source
    """
    check_dependencies()

    if X_real.shape[1] < 2:
        logging.warning(f"UMAP requires d>=2, got d={X_real.shape[1]}")
        return None

    # Collect all data sources
    data_sources = {"Real Data": X_real}
    if X_vertical_spn is not None:
        data_sources["Vertical SPN"] = X_vertical_spn
    if X_horizontal_spn is not None:
        data_sources["Horizontal SPN"] = X_horizontal_spn
    if X_centralized_spn is not None:
        data_sources["Centralized SPN"] = X_centralized_spn

    # Determine grid layout
    n_plots = len(data_sources) if len(data_sources) > 1 else 2

    # Create figure
    fig = plt.figure(figsize=(16, 5 * ((n_plots + 1) // 2)))

    if n_plots == 2:
        gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
    elif n_plots == 3:
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    else:
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")

    embeddings = {}

    # Fit UMAP on real data
    logging.info("  Computing UMAP projection...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=min(15, len(X_real) - 1),
        min_dist=0.1,
        verbose=False,
    )

    real_embedding = reducer.fit_transform(X_real)
    embeddings["Real Data"] = real_embedding

    # Plot each data source
    colors = {
        "Real Data": "blue",
        "Vertical SPN": "red",
        "Horizontal SPN": "green",
        "Centralized SPN": "purple",
    }

    plot_idx = 0
    for name, data in data_sources.items():
        if plot_idx < 4:
            ax = fig.add_subplot(
                gs[plot_idx // 2, plot_idx % 2] if n_plots > 2 else gs[0, plot_idx]
            )
        else:
            break

        if name == "Real Data":
            embedding = real_embedding
        else:
            # Transform generated samples using fitted UMAP
            embedding = reducer.transform(data)
            embeddings[name] = embedding

        # Plot
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors[name],
            alpha=0.5,
            s=20,
            label=name,
        )

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plot_idx += 1

    # Feature ownership visualization (if feature_maps provided)
    if show_feature_ownership and feature_maps is not None and plot_idx < 4:
        ax_ownership = fig.add_subplot(
            gs[plot_idx // 2, plot_idx % 2] if n_plots > 2 else gs[0, 1]
        )

        # Assign ownership label to each sample based on max variance feature
        ownership_labels = assign_ownership_labels(X_real, feature_maps)

        # Plot with ownership colors
        client_colors = plt.cm.tab10(np.linspace(0, 1, len(feature_maps)))
        for client_id, color in zip(sorted(feature_maps.keys()), client_colors):
            mask = ownership_labels == client_id
            ax_ownership.scatter(
                real_embedding[mask, 0],
                real_embedding[mask, 1],
                c=[color],
                alpha=0.6,
                s=20,
                label=f"Client {client_id}",
            )

        ax_ownership.set_xlabel("UMAP 1", fontsize=12)
        ax_ownership.set_ylabel("UMAP 2", fontsize=12)
        ax_ownership.set_title(
            "Feature Ownership Pattern", fontsize=14, fontweight="bold"
        )
        ax_ownership.grid(True, alpha=0.3)
        ax_ownership.legend()

    # Save
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"  ✓ Saved UMAP visualization: {save_path}")

    plt.close()
    return embeddings


def create_comparison_umap_overlay(
    X_real: np.ndarray,
    X_vertical_spn: np.ndarray,
    X_centralized_spn: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "UMAP Overlay: Real vs Generated",
) -> np.ndarray:
    """
    Create overlay UMAP plot showing real and generated data together.

    This visualization helps assess if vertical SPN captures the data manifold.

    Args:
        X_real: Real data
        X_vertical_spn: Vertical SPN samples
        X_centralized_spn: Centralized baseline (optional)
        save_path: Save path
        title: Plot title

    Returns:
        combined_embedding: UMAP embedding of all data
    """
    check_dependencies()

    # Combine data
    n_real = len(X_real)
    n_vertical = len(X_vertical_spn)

    if X_centralized_spn is not None:
        n_central = len(X_centralized_spn)
        combined = np.vstack([X_real, X_vertical_spn, X_centralized_spn])
        labels = (
            ["Real"] * n_real
            + ["Vertical SPN"] * n_vertical
            + ["Centralized"] * n_central
        )
    else:
        combined = np.vstack([X_real, X_vertical_spn])
        labels = ["Real"] * n_real + ["Vertical SPN"] * n_vertical

    # UMAP projection
    logging.info("  Computing UMAP overlay projection...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=min(15, len(combined) - 1),
        min_dist=0.1,
        verbose=False,
    )
    embedding = reducer.fit_transform(combined)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Left: All data overlayed
    ax = axes[0]
    for label, color in zip(
        ["Real", "Vertical SPN", "Centralized"], ["blue", "red", "purple"]
    ):
        if label not in labels:
            continue
        mask = np.array(labels) == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            alpha=0.4,
            s=15,
            label=label,
        )
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Overlay (All Data)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Density comparison
    ax = axes[1]
    from scipy.stats import gaussian_kde

    # Real data density
    try:
        real_mask = np.array(labels) == "Real"
        kde_real = gaussian_kde(embedding[real_mask].T)

        # Create grid
        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        # Evaluate
        density = kde_real(positions).reshape(xx.shape)

        # Plot
        ax.contourf(xx, yy, density, levels=10, cmap="Blues", alpha=0.6)
        ax.scatter(
            embedding[real_mask, 0],
            embedding[real_mask, 1],
            c="blue",
            alpha=0.3,
            s=5,
            label="Real Data",
        )

        # Vertical SPN overlay
        vertical_mask = np.array(labels) == "Vertical SPN"
        ax.scatter(
            embedding[vertical_mask, 0],
            embedding[vertical_mask, 1],
            c="red",
            alpha=0.4,
            s=10,
            label="Vertical SPN",
            marker="x",
        )

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title("Density Comparison", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logging.warning(f"Density plot failed: {e}")
        ax.text(0.5, 0.5, "Density plot unavailable", ha="center", va="center")

    # Save
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"  ✓ Saved UMAP overlay: {save_path}")

    plt.close()
    return embedding


def create_cross_client_dependency_umap(
    X_real: np.ndarray,
    X_vertical_spn: np.ndarray,
    feature_maps: Dict[int, List[int]],
    save_path: Optional[str] = None,
    title: str = "Cross-Client Dependency Analysis (UMAP)",
) -> None:
    """
    Visualize cross-client dependencies in vertical SPN.

    This plots feature importance from each client in UMAP space to show
    whether the vertical SPN captures dependencies across feature partitions.

    Key question: Does GlobalSumOfProducts capture correlations between
    features from different clients?

    Args:
        X_real: Real data [n, d]
        X_vertical_spn: Vertical SPN samples [m, d]
        feature_maps: Dict[client_id -> feature_indices]
        save_path: Path to save figure
        title: Plot title
    """
    check_dependencies()

    # Combine data
    combined = np.vstack([X_real, X_vertical_spn])
    labels = ["Real"] * len(X_real) + ["Generated"] * len(X_vertical_spn)

    # UMAP projection
    logging.info("  Computing cross-client dependency UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=min(15, len(combined) - 1),
        min_dist=0.1,
        verbose=False,
    )
    embedding = reducer.fit_transform(combined)

    # Create subplots: one per client
    K_clients = len(feature_maps)
    fig, axes = plt.subplots(1, K_clients, figsize=(6 * K_clients, 5))
    if K_clients == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=16, fontweight="bold")

    # For each client, color by their feature magnitude
    for client_id, ax in enumerate(axes):
        feature_indices = feature_maps[client_id]

        # Compute feature magnitude for this client's features
        real_magnitudes = np.linalg.norm(X_real[:, feature_indices], axis=1)
        gen_magnitudes = np.linalg.norm(X_vertical_spn[:, feature_indices], axis=1)
        all_magnitudes = np.concatenate([real_magnitudes, gen_magnitudes])

        # Normalize
        all_magnitudes = (all_magnitudes - all_magnitudes.min()) / (
            all_magnitudes.max() - all_magnitudes.min() + 1e-8
        )

        # Plot
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=all_magnitudes,
            cmap="viridis",
            alpha=0.6,
            s=15,
        )

        # Overlay real vs generated
        real_mask = np.array(labels) == "Real"
        ax.scatter(
            embedding[real_mask, 0],
            embedding[real_mask, 1],
            c="none",
            edgecolors="blue",
            linewidths=1,
            s=30,
            alpha=0.3,
            label="Real",
        )

        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.set_title(f"Client {client_id}\nFeatures: {feature_indices}", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Feature Magnitude", fontsize=10)

    # Save
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"  ✓ Saved cross-client dependency UMAP: {save_path}")

    plt.close()


def assign_ownership_labels(
    X: np.ndarray, feature_maps: Dict[int, List[int]]
) -> np.ndarray:
    """
    Assign each sample to a client based on which client's features have max variance.

    Args:
        X: Data [n, d]
        feature_maps: Dict[client_id -> feature_indices]

    Returns:
        labels: [n] array of client IDs
    """
    n = len(X)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        max_var = -1
        max_client = 0

        for client_id, features in feature_maps.items():
            var = np.var(X[i, features])
            if var > max_var:
                max_var = var
                max_client = client_id

        labels[i] = max_client

    return labels


def create_umap_comparison_report(
    X_real: np.ndarray,
    spn_models: Dict[str, any],
    feature_maps: Optional[Dict[int, List[int]]] = None,
    output_dir: str = "./umap_analysis",
    dataset_name: str = "experiment",
    n_samples: int = 500,
    device: str = "cpu",
) -> str:
    """
    Generate a complete UMAP analysis report for vertical SPN.

    This is the HIGH-LEVEL function to call for comprehensive analysis.

    Args:
        X_real: Real data [n, d]
        spn_models: Dict of models {"vertical": model, "horizontal": model, "centralized": model}
        feature_maps: Dict[client_id -> feature_indices] (required for vertical analysis)
        output_dir: Directory to save all visualizations
        dataset_name: Name for output files
        n_samples: Number of samples to generate from each SPN
        device: torch device

    Returns:
        output_dir: Path to directory with all visualizations
    """
    check_dependencies()

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"\n{'='*60}")
    logging.info(f"UMAP Analysis Report: {dataset_name.upper()}")
    logging.info(f"{'='*60}")

    # Extract samples from each model
    X_vertical = None
    X_horizontal = None
    X_centralized = None

    if "vertical" in spn_models and spn_models["vertical"] is not None:
        logging.info("  Sampling from vertical SPN...")
        X_vertical = extract_spn_samples(
            spn_models["vertical"], n_samples, device, has_context=False
        )

    if "horizontal" in spn_models and spn_models["horizontal"] is not None:
        logging.info("  Sampling from horizontal SPN...")
        X_horizontal = extract_spn_samples(
            spn_models["horizontal"], n_samples, device, has_context=True
        )

    if "centralized" in spn_models and spn_models["centralized"] is not None:
        logging.info("  Sampling from centralized SPN...")
        X_centralized = extract_spn_samples(
            spn_models["centralized"], n_samples, device, has_context=True
        )

    # 1. Global comparison visualization
    logging.info("\n  Creating global UMAP comparison...")
    create_global_umap_visualization(
        X_real=X_real,
        X_vertical_spn=X_vertical,
        X_horizontal_spn=X_horizontal,
        X_centralized_spn=X_centralized,
        feature_maps=feature_maps,
        save_path=os.path.join(output_dir, f"{dataset_name}_umap_global.png"),
        title=f"{dataset_name.upper()} - Global UMAP Comparison",
        show_feature_ownership=True,
    )

    # 2. Overlay visualization
    if X_vertical is not None:
        logging.info("  Creating UMAP overlay...")
        create_comparison_umap_overlay(
            X_real=X_real,
            X_vertical_spn=X_vertical,
            X_centralized_spn=X_centralized,
            save_path=os.path.join(output_dir, f"{dataset_name}_umap_overlay.png"),
            title=f"{dataset_name.upper()} - UMAP Overlay",
        )

    # 3. Cross-client dependency analysis
    if X_vertical is not None and feature_maps is not None:
        logging.info("  Creating cross-client dependency analysis...")
        create_cross_client_dependency_umap(
            X_real=X_real,
            X_vertical_spn=X_vertical,
            feature_maps=feature_maps,
            save_path=os.path.join(output_dir, f"{dataset_name}_umap_cross_client.png"),
            title=f"{dataset_name.upper()} - Cross-Client Dependencies",
        )

    logging.info(f"\n{'='*60}")
    logging.info(f"✅ UMAP Analysis Complete!")
    logging.info(f"   Output: {output_dir}/")
    logging.info(f"{'='*60}\n")

    return output_dir


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    """Demo: UMAP visualization for vertical SPN."""

    print("=" * 60)
    print("Demo: Vertical SPN UMAP Analysis")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n, d = 500, 6

    # Create correlated features
    mean = np.zeros(d)
    cov = np.eye(d) + 0.5  # Add correlations
    X_real = np.random.multivariate_normal(mean, cov, n)

    # Simulate vertical SPN samples (with slight distribution shift)
    X_vertical = np.random.multivariate_normal(mean, cov * 1.1, n)

    # Simulate centralized baseline
    X_centralized = np.random.multivariate_normal(mean, cov * 0.95, n)

    # Feature maps (3 clients, 2 features each)
    feature_maps = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
    }

    # Create visualizations
    output_dir = "/tmp/umap_vertical_spn_demo"

    # 1. Global comparison
    create_global_umap_visualization(
        X_real=X_real,
        X_vertical_spn=X_vertical,
        X_centralized_spn=X_centralized,
        feature_maps=feature_maps,
        save_path=f"{output_dir}/demo_global.png",
        title="Demo: Global UMAP Comparison",
    )

    # 2. Overlay
    create_comparison_umap_overlay(
        X_real=X_real,
        X_vertical_spn=X_vertical,
        X_centralized_spn=X_centralized,
        save_path=f"{output_dir}/demo_overlay.png",
        title="Demo: UMAP Overlay",
    )

    # 3. Cross-client dependencies
    create_cross_client_dependency_umap(
        X_real=X_real,
        X_vertical_spn=X_vertical,
        feature_maps=feature_maps,
        save_path=f"{output_dir}/demo_cross_client.png",
        title="Demo: Cross-Client Dependencies",
    )

    print(f"\n✅ Demo complete! Check {output_dir}/")
