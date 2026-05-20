# Code Examples for Proposed Improvements

**Note:** These are illustrative examples showing the proposed logic. NOT to be implemented yet.

---

## Example 1: Dataset Complexity Profiling

### Proposed New File: `causallearn/utils/data_profiling.py`

```python
"""
Dataset complexity profiling for adaptive hyperparameter tuning.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import Dict

def profile_dataset_complexity(
    X: np.ndarray,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Analyze dataset characteristics to guide hyperparameter selection.

    Args:
        X: Data matrix [n, d]
        verbose: If True, print profiling results

    Returns:
        Dictionary with complexity metrics:
        - correlation_strength: Mean absolute Spearman correlation [0,1]
        - effective_dimension: Ratio of explained variance (PCA)
        - noise_estimate: Estimated noise level [0,1]
        - feature_heterogeneity: CV of feature variances
        - expected_sparsity: Proportion of weak correlations [0,1]
    """
    n, d = X.shape
    profile = {}

    # 1. Correlation Strength Analysis
    # Measures: How strongly are features related?
    corr_matrix, _ = spearmanr(X, axis=0)
    corr_matrix = np.abs(corr_matrix)
    np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation

    # Mean absolute correlation (excluding diagonal)
    profile['correlation_strength'] = np.mean(corr_matrix)

    # Expected sparsity: proportion of weak correlations (<0.2)
    weak_corr_count = np.sum(corr_matrix < 0.2)
    total_pairs = d * (d - 1) / 2
    profile['expected_sparsity'] = weak_corr_count / (2 * total_pairs)

    # 2. Effective Dimensionality (PCA)
    # Measures: How many dimensions are truly informative?
    if d > 1:
        pca = PCA(n_components=min(d, n-1))
        pca.fit(X)

        # Ratio of dimensions needed to explain 90% variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_90 = np.searchsorted(cumsum_var, 0.9) + 1
        profile['effective_dimension'] = n_components_90 / d
    else:
        profile['effective_dimension'] = 1.0

    # 3. Noise Estimation (via linear fit residuals)
    # Measures: How much unexplained variance?
    if d >= 2:
        # Fit linear model: X[:, 0] ~ X[:, 1:]
        lr = LinearRegression()
        lr.fit(X[:, 1:], X[:, 0])
        y_pred = lr.predict(X[:, 1:])
        residuals = X[:, 0] - y_pred

        # Normalized residual variance
        signal_var = np.var(X[:, 0])
        noise_var = np.var(residuals)
        profile['noise_estimate'] = noise_var / (signal_var + 1e-8)
        profile['noise_estimate'] = min(profile['noise_estimate'], 1.0)
    else:
        profile['noise_estimate'] = 0.5  # Moderate default

    # 4. Feature Heterogeneity
    # Measures: How variable are feature scales?
    feature_vars = np.var(X, axis=0)
    if np.mean(feature_vars) > 0:
        # Coefficient of variation
        profile['feature_heterogeneity'] = np.std(feature_vars) / np.mean(feature_vars)
    else:
        profile['feature_heterogeneity'] = 0.0

    if verbose:
        print("\n=== Dataset Complexity Profile ===")
        print(f"Correlation strength: {profile['correlation_strength']:.3f}")
        print(f"  → {'Strong' if profile['correlation_strength'] > 0.5 else 'Weak'} dependencies")
        print(f"Effective dimension: {profile['effective_dimension']:.3f}")
        print(f"  → {int(profile['effective_dimension'] * d)}/{d} dimensions informative")
        print(f"Noise estimate: {profile['noise_estimate']:.3f}")
        print(f"  → {'High' if profile['noise_estimate'] > 0.3 else 'Low'} noise level")
        print(f"Expected sparsity: {profile['expected_sparsity']:.3f}")
        print(f"  → Graph likely {'sparse' if profile['expected_sparsity'] > 0.5 else 'dense'}")
        print(f"Feature heterogeneity: {profile['feature_heterogeneity']:.3f}")
        print("="*35 + "\n")

    return profile


def recommend_hyperparameters_from_profile(
    profile: Dict[str, float],
    mode: str = "horizontal"
) -> Dict[str, str]:
    """
    Generate human-readable recommendations based on complexity profile.

    Returns:
        Dictionary with recommendations for capacity, regularization, epochs
    """
    recommendations = {}

    # Capacity recommendation
    corr = profile['correlation_strength']
    if corr > 0.5:
        recommendations['capacity'] = "HIGH (1.3× base) - Strong dependencies detected"
    elif corr > 0.3:
        recommendations['capacity'] = "MEDIUM (1.0× base) - Moderate dependencies"
    else:
        recommendations['capacity'] = "LOW (0.8× base) - Weak dependencies"

    # Regularization recommendation
    noise = profile['noise_estimate']
    if noise > 0.3:
        recommendations['regularization'] = "HIGH (2× weight_decay, +0.05 dropout) - Noisy data"
    elif noise > 0.15:
        recommendations['regularization'] = "MEDIUM (1× weight_decay) - Moderate noise"
    else:
        recommendations['regularization'] = "LOW (0.5× weight_decay) - Clean data"

    # Epochs recommendation
    eff_dim = profile['effective_dimension']
    if eff_dim > 0.8:
        recommendations['epochs'] = "HIGH (1.2× base) - High dimensionality"
    elif eff_dim > 0.5:
        recommendations['epochs'] = "MEDIUM (1.0× base) - Moderate dimensionality"
    else:
        recommendations['epochs'] = "LOW (0.8× base) - Low dimensionality"

    return recommendations
```

**Usage Example:**

```python
# In FedCDH.fit() before training
from causallearn.utils.data_profiling import profile_dataset_complexity

# Combine all client data for profiling (only for horizontal mode)
X_combined = np.vstack(X_splits)
complexity_profile = profile_dataset_complexity(X_combined, verbose=True)

# Pass to hyperparameter computation
hyperparams = compute_adaptive_hyperparameters(
    mode=self.scenario,
    num_features=d,
    num_samples=n_k,
    data_type=self.data_type,
    complexity_profile=complexity_profile,  # NEW
)
```

---

## Example 2: Enhanced Hyperparameter Function

### Modified: `causallearn/utils/FedPC.py`

```python
def compute_adaptive_hyperparameters(
    mode: str,
    num_features: int,
    num_samples: int,
    data_type: str,
    complexity_profile: Optional[Dict[str, float]] = None,  # NEW PARAMETER
    base_num_sums: int = 20,
    base_num_leaves: int = 20,
    base_epochs: int = 100,
    base_depth: int = None,
) -> Dict[str, Any]:
    """
    Compute adaptive hyperparameters with optional complexity-based tuning.

    NEW in v3.1: Criterion 6 - Data Complexity Adaptation
    """

    # === Existing Criteria 1-5 (unchanged) ===
    # [Lines 2432-2520 remain the same]
    # ...

    # === NEW: Criterion 6 - Data Complexity Adaptation ===
    if complexity_profile is not None:
        logging.info("  [Criterion 6] Applying data complexity adjustments...")

        # 6A. Correlation Strength Adjustment
        corr_strength = complexity_profile.get('correlation_strength', 0.3)

        if corr_strength > 0.5:
            # Strong correlations need more capacity to model
            capacity_boost = 1.3
            logging.info(f"    → Strong correlations ({corr_strength:.2f}) → capacity +30%")
        elif corr_strength > 0.3:
            capacity_boost = 1.0
            logging.info(f"    → Moderate correlations ({corr_strength:.2f}) → capacity baseline")
        else:
            # Weak correlations → simpler model sufficient
            capacity_boost = 0.8
            logging.info(f"    → Weak correlations ({corr_strength:.2f}) → capacity -20%")

        final_num_sums = int(final_num_sums * capacity_boost)
        final_num_leaves = int(final_num_leaves * capacity_boost)

        # 6B. Noise Level Adjustment
        noise = complexity_profile.get('noise_estimate', 0.2)

        if noise > 0.3:
            # High noise → increase regularization
            weight_decay *= 2.0
            dropout = min(dropout + 0.05, 0.3)  # Add dropout, cap at 0.3
            logging.info(f"    → High noise ({noise:.2f}) → 2× regularization, +0.05 dropout")
        elif noise < 0.1:
            # Low noise → can reduce regularization
            weight_decay *= 0.5
            logging.info(f"    → Low noise ({noise:.2f}) → 0.5× regularization")

        # 6C. Effective Dimension Adjustment
        eff_dim = complexity_profile.get('effective_dimension', 0.7)

        if eff_dim > 0.8:
            # Most dimensions are informative → more training needed
            epoch_multiplier = 1.2
            logging.info(f"    → High eff. dim ({eff_dim:.2f}) → +20% epochs")
        elif eff_dim < 0.4:
            # Many redundant dimensions → less training needed
            epoch_multiplier = 0.8
            logging.info(f"    → Low eff. dim ({eff_dim:.2f}) → -20% epochs")
        else:
            epoch_multiplier = 1.0

        final_epochs = int(final_epochs * epoch_multiplier)

        # Log final adjustments
        logging.info(f"    Final: sums={final_num_sums}, leaves={final_num_leaves}, "
                    f"epochs={final_epochs}, dropout={dropout:.2f}, wd={weight_decay:.1e}")

    # === Return (updated) ===
    return {
        "num_sums": final_num_sums,
        "num_leaves": final_num_leaves,
        "depth": final_depth,
        "epochs": final_epochs,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "complexity_profile": complexity_profile,  # Store for logging
    }
```

**Expected Log Output:**

```
[Criterion 1] Mode-Specific Base: horizontal → sums=20, leaves=10
[Criterion 2] Sample-to-Feature Ratio: 4200 → capacity_scale=1.5
[Criterion 3] Data Type: linear → depth=2, epoch_mult=1.0
[Criterion 4] Epoch Scheduling: mode=horizontal → 250 epochs
[Criterion 5] Regularization: horizontal → wd=1e-4, dropout=0.0
[Criterion 6] Applying data complexity adjustments...
  → Weak correlations (0.18) → capacity -20%
  → Moderate noise (0.25) → regularization baseline
  → High eff. dim (0.85) → +20% epochs
  Final: sums=24, leaves=12, epochs=300, dropout=0.00, wd=1.0e-04
```

---

## Example 3: Adaptive Threshold Calibration

### Proposed New Functions: `causallearn/utils/structure_aggregation.py`

```python
def calibrate_structure_threshold(
    K: int,
    num_features: int,
    expected_edges: Optional[int] = None,
    client_heterogeneity: float = 0.5,
    target_agreement: str = "majority",
) -> float:
    """
    Compute dataset-adaptive structure voting threshold.

    Args:
        K: Number of clients
        num_features: Number of features (d)
        expected_edges: Prior belief about edge count (from dataset metadata)
        client_heterogeneity: Estimated data heterogeneity [0,1]
            0 = IID (homogeneous clients)
            1 = highly heterogeneous (non-IID)
        target_agreement: "majority" | "supermajority" | "consensus"

    Returns:
        Calibrated threshold in [0.3, 0.8]

    Example:
        >>> # Law school: 5 features, 7 edges, moderate heterogeneity
        >>> thresh = calibrate_structure_threshold(
        ...     K=3, num_features=5, expected_edges=7,
        ...     client_heterogeneity=0.3, target_agreement="majority"
        ... )
        >>> print(f"Threshold: {thresh:.3f}")  # ~0.485
    """

    # Step 1: Base threshold from target agreement
    if target_agreement == "majority":
        base = 0.5
    elif target_agreement == "supermajority":
        base = 0.67
    elif target_agreement == "consensus":
        base = 0.9
    else:
        base = 0.5

    # Step 2: Adjust for client heterogeneity
    # More heterogeneous → lower threshold (clients disagree more)
    # Less heterogeneous → higher threshold (clients agree more)
    heterogeneity_adjustment = -0.2 * client_heterogeneity

    # Step 3: Adjust for expected graph sparsity
    if expected_edges is not None:
        max_edges = num_features * (num_features - 1) / 2
        sparsity = 1.0 - (expected_edges / max_edges)

        # Sparse graphs → higher threshold (be conservative, avoid false positives)
        # Dense graphs → lower threshold (be liberal, avoid missing edges)
        sparsity_adjustment = 0.15 * sparsity
    else:
        sparsity_adjustment = 0.0

    # Step 4: Adjust for client count (avoid ceiling effects)
    # Small K → need integer-aware adjustment
    if K <= 5:
        # For small K, ensure threshold maps to reasonable vote counts
        required_votes = max(2, int(np.ceil(base * K)))
        effective_threshold = required_votes / K

        # If ceiling effect changes threshold significantly, adjust base
        if abs(effective_threshold - base) > 0.1:
            base = effective_threshold
            logging.info(f"  [Threshold] Ceiling adjustment for K={K}: {base:.3f}")

    # Combine adjustments
    threshold = base + heterogeneity_adjustment + sparsity_adjustment

    # Clamp to reasonable range
    threshold = np.clip(threshold, 0.3, 0.8)

    return threshold


def estimate_client_heterogeneity(
    X_splits: List[np.ndarray],
    method: str = "correlation_variance"
) -> float:
    """
    Estimate heterogeneity across clients.

    Args:
        X_splits: List of client data arrays
        method: "correlation_variance" | "mean_difference"

    Returns:
        Heterogeneity estimate in [0, 1]
        0 = perfectly homogeneous (IID)
        1 = maximally heterogeneous (non-IID)

    Example:
        >>> X_splits = [client0_data, client1_data, client2_data]
        >>> het = estimate_client_heterogeneity(X_splits)
        >>> print(f"Heterogeneity: {het:.2f}")  # 0.35 → moderate
    """
    K = len(X_splits)
    d = X_splits[0].shape[1]

    if method == "correlation_variance":
        # Compute correlation matrix for each client
        local_corrs = []
        for X_k in X_splits:
            corr_k, _ = spearmanr(X_k, axis=0)
            corr_k = np.abs(corr_k)
            np.fill_diagonal(corr_k, 0)
            local_corrs.append(corr_k)

        # Variance of correlations across clients
        corr_stack = np.stack(local_corrs, axis=0)  # [K, d, d]
        corr_variance = np.var(corr_stack, axis=0)  # [d, d]

        # Average variance (excluding diagonal)
        mask = ~np.eye(d, dtype=bool)
        avg_variance = np.mean(corr_variance[mask])

        # Normalize to [0, 1] (heuristic: variance > 0.1 is "high")
        heterogeneity = min(avg_variance / 0.1, 1.0)

    elif method == "mean_difference":
        # Compare feature means across clients
        feature_means = np.array([np.mean(X_k, axis=0) for X_k in X_splits])  # [K, d]

        # Coefficient of variation for each feature
        cv_per_feature = np.std(feature_means, axis=0) / (np.mean(feature_means, axis=0) + 1e-8)

        # Average CV (normalized)
        avg_cv = np.mean(cv_per_feature)
        heterogeneity = min(avg_cv, 1.0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return heterogeneity
```

**Usage Example:**

```python
# In FedCDH.fit() before structure voting

# Estimate heterogeneity from data
from causallearn.utils.structure_aggregation import (
    estimate_client_heterogeneity,
    calibrate_structure_threshold
)

heterogeneity = estimate_client_heterogeneity(X_splits)
logging.info(f"Estimated client heterogeneity: {heterogeneity:.3f}")

# Calibrate threshold
adaptive_threshold = calibrate_structure_threshold(
    K=self.K_clients,
    num_features=self.d_features,
    expected_edges=int(np.sum(B)) if B is not None else None,
    client_heterogeneity=heterogeneity,
    target_agreement="majority"
)

logging.info(f"Adaptive threshold: {adaptive_threshold:.3f} (vs default {self.structure_vote_threshold:.3f})")

# Use adaptive threshold
consensus_graph, edge_confidence = aggregate_structures_by_voting(
    local_graphs=local_graphs,
    threshold=adaptive_threshold,  # Was: self.structure_vote_threshold
)
```

**Expected Log Output:**

```
Estimated client heterogeneity: 0.342
  [Threshold] K=3, d=5, expected_edges=7
  [Threshold] Base (majority): 0.500
  [Threshold] Heterogeneity adjustment: -0.068
  [Threshold] Sparsity adjustment (0.30): +0.045
  [Threshold] Ceiling adjustment for K=3: 0.667
Adaptive threshold: 0.485 (vs default 0.400)
```

---

## Example 4: Confidence-Weighted Voting

### Modified: `causallearn/utils/structure_aggregation.py`

```python
def aggregate_structures_weighted_voting(
    local_graphs: List[nx.Graph],
    threshold: float = 0.5,
    use_pvalue_weights: bool = True,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], float]]:
    """
    Aggregate structures with confidence-weighted voting.

    Instead of binary votes (0 or 1), each client contributes:
        vote_weight = (1 - p_value) if p_value available, else 1.0

    Edge is included if: sum(vote_weights) / K >= threshold

    Args:
        local_graphs: List of NetworkX graphs from each client
        threshold: Proportion of weighted votes required
        use_pvalue_weights: If True, weight by (1-p_value); if False, binary votes

    Returns:
        consensus_graph: NetworkX Graph with weighted consensus edges
        edge_weighted_confidence: Dict mapping (i,j) -> confidence score

    Example:
        # Client 0: edge (1,2) with p=0.001 → weight=0.999
        # Client 1: edge (1,2) with p=0.030 → weight=0.970
        # Client 2: no edge (1,2) → weight=0.000
        # Average: (0.999 + 0.970 + 0.0) / 3 = 0.656
        # If threshold=0.5 → INCLUDE (strong evidence from 2/3 clients)
    """
    K = len(local_graphs)

    if K == 0:
        return nx.Graph(), {}

    # Get max node index
    d = max(max(g.nodes()) + 1 if len(g.nodes()) > 0 else 0 for g in local_graphs)

    # Collect weighted votes
    edge_weighted_votes = defaultdict(float)
    edge_pvalues = defaultdict(list)

    for graph in local_graphs:
        for i, j in graph.edges():
            edge = (min(i, j), max(i, j))

            if use_pvalue_weights and 'p_value' in graph[i][j]:
                # Confidence weight from p-value
                p_val = graph[i][j]['p_value']
                weight = 1.0 - p_val  # Lower p-value → higher weight

                # Store p-value for median computation
                edge_pvalues[edge].append(p_val)
            else:
                # Binary vote
                weight = 1.0

            edge_weighted_votes[edge] += weight

    # Normalize by K to get average confidence
    edge_confidence = {}
    for edge in edge_weighted_votes:
        edge_confidence[edge] = edge_weighted_votes[edge] / K

    # Build consensus graph
    consensus_graph = nx.Graph()
    consensus_graph.add_nodes_from(range(d))

    for edge, confidence in edge_confidence.items():
        if confidence >= threshold:
            i, j = edge

            # Add edge with confidence and optional median p-value
            edge_attrs = {
                'confidence': confidence,
                'weighted_votes': edge_weighted_votes[edge],
            }

            if edge in edge_pvalues:
                edge_attrs['median_pvalue'] = np.median(edge_pvalues[edge])

            consensus_graph.add_edge(i, j, **edge_attrs)

    logging.info(f"  Weighted voting: {len(consensus_graph.edges())} edges "
                f"(threshold={threshold:.2f}, K={K})")

    return consensus_graph, edge_confidence


# Comparison function for ablation studies
def compare_voting_methods(
    local_graphs: List[nx.Graph],
    true_graph: np.ndarray,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
) -> pd.DataFrame:
    """
    Compare binary vs weighted voting across threshold range.

    Returns:
        DataFrame with columns: [threshold, method, f1, precision, recall]
    """
    results = []

    for thresh in thresholds:
        # Binary voting
        consensus_bin, _ = aggregate_structures_by_voting(
            local_graphs, threshold=thresh
        )
        metrics_bin = compute_skeleton_metrics(consensus_bin, true_graph)
        results.append({
            'threshold': thresh,
            'method': 'binary',
            **metrics_bin
        })

        # Weighted voting
        consensus_wt, _ = aggregate_structures_weighted_voting(
            local_graphs, threshold=thresh, use_pvalue_weights=True
        )
        metrics_wt = compute_skeleton_metrics(consensus_wt, true_graph)
        results.append({
            'threshold': thresh,
            'method': 'weighted',
            **metrics_wt
        })

    return pd.DataFrame(results)
```

**Usage Example:**

```python
# In benchmark evaluation (test_fedcdh_benchmark_v3.py)

# After running FedSPN, compare voting methods
comparison_df = compare_voting_methods(
    local_graphs=local_graphs,
    true_graph=B,
    thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]
)

print(comparison_df)
#    threshold    method     f1  precision  recall
# 0        0.3    binary  0.45       0.35    0.65
# 1        0.3  weighted  0.52       0.42    0.68
# 2        0.4    binary  0.60       0.55    0.67
# 3        0.4  weighted  0.68       0.63    0.74
# ...

# Save for analysis
comparison_df.to_csv(exp_dir / "voting_comparison.csv", index=False)
```

---

## Example 5: Two-Phase Training

### Modified: `causallearn/utils/FedPC.py` (LocalSPNWrapper)

```python
class LocalSPNWrapper(nn.Module):
    # ... existing __init__ and other methods ...

    def train_local_twophase(
        self,
        data: np.ndarray,
        total_epochs: int = 100,
        phase1_ratio: float = 0.2,
        validation_split: float = 0.1,
        **kwargs
    ):
        """
        Two-phase adaptive training:

        Phase 1 (20% epochs): Exploration
          - Use higher capacity (1.5× base)
          - Lower regularization
          - Collect diagnostics (train/val LL)

        Phase 2 (80% epochs): Exploitation
          - Adjust based on Phase 1 overfitting/underfitting
          - If overfitting: reduce capacity, increase regularization
          - If underfitting: increase capacity, reduce regularization

        Args:
            data: Training data [n, d]
            total_epochs: Total training epochs
            phase1_ratio: Fraction of epochs for Phase 1 (default 0.2)
            validation_split: Fraction of data for validation (default 0.1)
            **kwargs: Additional hyperparameters (lr, dropout, etc.)
        """
        n = len(data)
        n_val = int(n * validation_split)
        n_train = n - n_val

        # Split data
        indices = np.random.permutation(n)
        train_data = data[indices[:n_train]]
        val_data = data[indices[n_train:]]

        # === Phase 1: Exploration ===
        phase1_epochs = int(total_epochs * phase1_ratio)

        logging.info(f"  [Phase 1] Exploration: {phase1_epochs} epochs")

        # Boost capacity for exploration
        self.train_local(
            train_data,
            epochs=phase1_epochs,
            **kwargs
        )

        # Collect diagnostics
        train_ll = self.compute_log_likelihood(train_data)
        val_ll = self.compute_log_likelihood(val_data)
        ll_gap = train_ll - val_ll

        logging.info(f"    Train LL: {train_ll:.2f}, Val LL: {val_ll:.2f}, Gap: {ll_gap:.2f}")

        # === Detect overfitting/underfitting ===
        if ll_gap > 2.0:
            # Overfitting: train much better than val
            adjustment = "reduce_capacity"
            capacity_factor = 0.7
            dropout_bonus = 0.1
            logging.info(f"    → Overfitting detected (gap={ll_gap:.2f})")

        elif train_ll < -5.0 and val_ll < -5.0:
            # Underfitting: both poor
            adjustment = "increase_capacity"
            capacity_factor = 1.3
            dropout_bonus = -0.05
            logging.info(f"    → Underfitting detected (train_ll={train_ll:.2f})")

        else:
            # Balanced
            adjustment = "maintain"
            capacity_factor = 1.0
            dropout_bonus = 0.0
            logging.info(f"    → Balanced fit")

        # === Phase 2: Exploitation ===
        phase2_epochs = total_epochs - phase1_epochs

        logging.info(f"  [Phase 2] Exploitation: {phase2_epochs} epochs ({adjustment})")

        # Adjust hyperparameters
        adjusted_kwargs = kwargs.copy()
        adjusted_kwargs['dropout'] = max(0.0, kwargs.get('dropout', 0.0) + dropout_bonus)
        adjusted_kwargs['l2_weight'] = kwargs.get('l2_weight', 1e-5) * (2.0 if ll_gap > 2.0 else 1.0)

        # If capacity adjustment needed, would require re-initializing model
        # (not shown here - would need to create new model with adjusted num_sums/leaves)

        # Continue training with adjusted params
        self.train_local(
            train_data,
            epochs=phase2_epochs,
            **adjusted_kwargs
        )

        # Final diagnostics
        final_train_ll = self.compute_log_likelihood(train_data)
        final_val_ll = self.compute_log_likelihood(val_data)
        logging.info(f"  [Final] Train LL: {final_train_ll:.2f}, Val LL: {final_val_ll:.2f}")

        return {
            'phase1_train_ll': train_ll,
            'phase1_val_ll': val_ll,
            'phase2_train_ll': final_train_ll,
            'phase2_val_ll': final_val_ll,
            'adjustment': adjustment,
        }

    def compute_log_likelihood(self, data: np.ndarray) -> float:
        """Compute average log-likelihood on data."""
        with torch.no_grad():
            X_tensor = torch.from_numpy(data).float().to(self.device)
            ll = self.log_prob(X_tensor).mean().item()
        return ll
```

**Expected Log Output:**

```
[Phase 1] Exploration: 20 epochs
  Train LL: -2.34, Val LL: -4.56, Gap: 2.22
  → Overfitting detected (gap=2.22)
[Phase 2] Exploitation: 80 epochs (reduce_capacity)
  Adjusted: dropout=0.15 (+0.10), l2_weight=2.0e-4 (2×)
[Final] Train LL: -3.12, Val LL: -3.45
  Improvement: Gap reduced 2.22 → 0.33
```

---

## Summary of Code Examples

### Files That Would Be Created/Modified:

**New Files:**
1. `causallearn/utils/data_profiling.py` (~150 lines)
   - `profile_dataset_complexity()`
   - `recommend_hyperparameters_from_profile()`

**Modified Files:**
1. `causallearn/utils/FedPC.py` (~50 lines added)
   - `compute_adaptive_hyperparameters()` - add complexity_profile parameter
   - `LocalSPNWrapper.train_local_twophase()` - new method

2. `causallearn/utils/structure_aggregation.py` (~100 lines added)
   - `calibrate_structure_threshold()`
   - `estimate_client_heterogeneity()`
   - `aggregate_structures_weighted_voting()`
   - `compare_voting_methods()`

3. `causallearn/search/FCMBased/FedCDH/FedCDH.py` (~30 lines modified)
   - Call `profile_dataset_complexity()` before training
   - Call `calibrate_structure_threshold()` before voting
   - Pass complexity profile to hyperparameter function

4. `tests/benchmarks/test_fedcdh_benchmark_v3.py` (~20 lines added)
   - Add complexity profiling step
   - Add threshold sweep evaluation
   - Save profiling results

**Total Estimated Changes:**
- New code: ~300 lines
- Modified code: ~100 lines
- Total: ~400 lines (manageable scope)

---

**Note:** These are illustrative examples only. Actual implementation should:
1. Follow existing code style
2. Add comprehensive tests
3. Include error handling
4. Add documentation
5. Consider edge cases (d=1, K=1, etc.)
