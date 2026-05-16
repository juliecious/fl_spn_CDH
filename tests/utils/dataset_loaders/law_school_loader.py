"""
Law School Admissions Dataset Loader.

Based on LSAC Bar Passage Study (Wightman, 1998).
Causal structure from Kusner et al. (2017) "Counterfactual Fairness".

Variables:
- race: Binary protected attribute (0=non-minority, 1=minority)
- LSAT: Law School Admission Test score (continuous, normalized)
- UGPA: Undergraduate GPA (continuous, normalized)
- region_first: Region of first-choice law school (categorical → continuous)
- ZFYA: First-year law school average (continuous, normalized, outcome)

Causal Structure (Ground Truth):
    race → LSAT, UGPA, region_first, ZFYA
    LSAT → ZFYA
    UGPA → ZFYA
    region_first → ZFYA

Total edges: 7
(1 node with 4 outgoing, 3 nodes with 1 outgoing each)

Reference:
- Wightman, L. F. (1998). LSAC national longitudinal bar passage study.
- Kusner et al. (2017). Counterfactual fairness. NeurIPS.
"""

import numpy as np
import pandas as pd
import gzip
import os


def generate_law_school_data(n_samples=21000, seed=42):
    """
    Generate synthetic Law School data based on known causal structure.

    Uses realistic parameters from Kusner et al. (2017) and Wightman (1998).

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        X: (n_samples, 5) array of features
        B: (5, 5) adjacency matrix (ground truth)
        feature_names: List of feature names
    """
    np.random.seed(seed)

    # Feature names
    feature_names = ["race", "LSAT", "UGPA", "region_first", "ZFYA"]
    d = len(feature_names)

    # Generate data following causal structure

    # 1. race: Exogenous binary variable (minority = 1)
    # Approximately 20% minority (based on LSAC data)
    race = np.random.binomial(1, 0.20, n_samples).astype(float)

    # 2. LSAT: Affected by race (structural bias)
    # Non-minority mean ~0.5, minority mean ~0.3 (normalized scale)
    # Standard deviation ~0.2
    lsat_base = np.random.normal(0.5, 0.2, n_samples)
    lsat_race_effect = -0.2 * race  # Minority penalty
    LSAT = lsat_base + lsat_race_effect + np.random.normal(0, 0.1, n_samples)
    LSAT = np.clip(LSAT, 0, 1)  # Normalize to [0, 1]

    # 3. UGPA: Affected by race (structural bias)
    # Non-minority mean ~0.6, minority mean ~0.45
    ugpa_base = np.random.normal(0.6, 0.15, n_samples)
    ugpa_race_effect = -0.15 * race  # Minority penalty
    UGPA = ugpa_base + ugpa_race_effect + np.random.normal(0, 0.1, n_samples)
    UGPA = np.clip(UGPA, 0, 1)

    # 4. region_first: Region of first-choice school, affected by race
    # Continuous encoding: 0-1 (e.g., 0=West, 0.5=Midwest, 1=East)
    # Minority students tend toward certain regions (0.4 baseline vs 0.6)
    region_base = np.random.uniform(0, 1, n_samples)
    region_race_effect = -0.2 * race
    region_first = (
        region_base + region_race_effect + np.random.normal(0, 0.15, n_samples)
    )
    region_first = np.clip(region_first, 0, 1)

    # 5. ZFYA: First-year average (outcome)
    # Affected by: race (direct discrimination), LSAT, UGPA, region_first
    zfya_base = np.random.normal(0.5, 0.1, n_samples)
    zfya_race_effect = -0.1 * race  # Direct discrimination
    zfya_lsat_effect = 0.3 * LSAT
    zfya_ugpa_effect = 0.25 * UGPA
    zfya_region_effect = 0.15 * region_first

    ZFYA = (
        zfya_base
        + zfya_race_effect
        + zfya_lsat_effect
        + zfya_ugpa_effect
        + zfya_region_effect
        + np.random.normal(0, 0.1, n_samples)
    )
    ZFYA = np.clip(ZFYA, 0, 1)

    # Combine into data matrix
    X = np.column_stack([race, LSAT, UGPA, region_first, ZFYA])

    # Ground truth adjacency matrix
    # Columns: race, LSAT, UGPA, region_first, ZFYA (indices 0-4)
    B = np.zeros((d, d))

    # race → LSAT, UGPA, region_first, ZFYA
    B[0, 1] = 1  # race → LSAT
    B[0, 2] = 1  # race → UGPA
    B[0, 3] = 1  # race → region_first
    B[0, 4] = 1  # race → ZFYA

    # LSAT → ZFYA
    B[1, 4] = 1

    # UGPA → ZFYA
    B[2, 4] = 1

    # region_first → ZFYA
    B[3, 4] = 1

    return X, B, feature_names


def save_law_school_data(output_path, n_samples=21000, seed=42):
    """
    Generate and save Law School data to CSV (optionally gzipped).

    Args:
        output_path: Output file path (.csv or .csv.gz)
        n_samples: Number of samples
        seed: Random seed
    """
    X, B, feature_names = generate_law_school_data(n_samples, seed)

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    # Save
    if output_path.endswith(".gz"):
        with gzip.open(output_path, "wt") as f:
            df.to_csv(f, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Saved Law School data to: {output_path}")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Edges: {int(np.sum(B))}")

    return output_path


def load_law_school_data(data_path=None, n_samples_limit=None):
    """
    Load Law School Admissions dataset.

    If data_path doesn't exist, generates synthetic data based on
    known causal structure from Kusner et al. (2017).

    Args:
        data_path: Path to CSV file (None = generate synthetic)
        n_samples_limit: Limit number of samples (None = all)

    Returns:
        X: (n, 5) array of features
        B: (5, 5) adjacency matrix (ground truth)
        feature_names: List of feature names
    """
    # If no path or path doesn't exist, generate synthetic data
    if data_path is None or not os.path.exists(data_path):
        if data_path is not None:
            print(f"Data file not found: {data_path}")
        print("Generating synthetic Law School data based on known causal structure...")

        n_samples = n_samples_limit if n_samples_limit else 21000
        X, B, feature_names = generate_law_school_data(n_samples)

        return X, B, feature_names

    # Load from file
    if data_path.endswith(".gz"):
        with gzip.open(data_path, "rt") as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(data_path)

    feature_names = list(df.columns)
    X = df.values

    if n_samples_limit:
        X = X[:n_samples_limit]

    # Ground truth adjacency (same structure regardless of source)
    d = X.shape[1]
    B = np.zeros((d, d))

    # Assume standard order: race, LSAT, UGPA, region_first, ZFYA
    if d >= 5:
        B[0, 1:5] = 1  # race → all others
        B[1, 4] = 1  # LSAT → ZFYA
        B[2, 4] = 1  # UGPA → ZFYA
        B[3, 4] = 1  # region_first → ZFYA

    return X, B, feature_names


def load_law_school_federated(n_clients=3, n_samples_limit=None, data_path=None):
    """
    Load Law School data and prepare for federated learning.

    Args:
        n_clients: Number of federated clients
        n_samples_limit: Total samples to use (None = 21000)
        data_path: Path to data file (None = generate synthetic)

    Returns:
        X: Full dataset (n, d)
        B: Ground truth adjacency (d, d)
        feature_names: List of feature names
    """
    X, B, feature_names = load_law_school_data(data_path, n_samples_limit)

    print(f"\nLaw School Admissions Dataset:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]} - {feature_names}")
    print(f"  Ground truth edges: {int(np.sum(B))}")
    print(f"  Causal structure:")
    print(f"    race → LSAT, UGPA, region_first, ZFYA (4 edges)")
    print(f"    LSAT → ZFYA (1 edge)")
    print(f"    UGPA → ZFYA (1 edge)")
    print(f"    region_first → ZFYA (1 edge)")
    print(f"  Total: 7 edges")

    return X, B, feature_names


if __name__ == "__main__":
    # Test data generation
    print("Testing Law School data generation...")

    # Generate small test dataset
    X, B, names = generate_law_school_data(n_samples=1000, seed=42)

    print(f"\nGenerated data shape: {X.shape}")
    print(f"Feature names: {names}")
    print(f"Ground truth edges: {int(np.sum(B))}")
    print(f"\nAdjacency matrix B:")
    print(B.astype(int))

    print(f"\nSample statistics:")
    for i, name in enumerate(names):
        print(f"  {name}: mean={X[:, i].mean():.3f}, std={X[:, i].std():.3f}")

    # Test save/load
    test_path = "/tmp/law_school_test.csv.gz"
    save_law_school_data(test_path, n_samples=1000)

    X_loaded, B_loaded, names_loaded = load_law_school_data(test_path)
    print(f"\nLoaded data shape: {X_loaded.shape}")
    print(f"Data matches: {np.allclose(X, X_loaded)}")
