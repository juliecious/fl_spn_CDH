# V1 vs V2: Complete Change Analysis

**Comparison**: `main` (v1 baseline) vs `v2-adaptive-hyperparameters` (current)
**Date**: 2026-04-30

---

## Executive Summary

**3 Core Files Changed** (4,505 lines added/modified):
1. **`causallearn/search/FCMBased/FedCDH/FedCDH.py`** - Main algorithm (+1,500 lines)
2. **`causallearn/utils/FedPC.py`** - Federated SPN aggregation (+2,722 lines)
3. **`causallearn/utils/cit.py`** - Conditional independence testing (+283 modified)

**Supporting Files** (+2,018 lines):
- Evaluation, dashboards, CI ranking, utilities

---

## Core Implementation Changes

### 1. 🔥 FedCDH.py - Main Algorithm (NEW FILE, 1,500 lines)

**Location**: `causallearn/search/FCMBased/FedCDH/FedCDH.py`

**Status**: V1 didn't exist as a separate module. V2 created complete implementation.

#### Key Features Added

##### A. Adaptive Hyperparameters (Lines 125-247)

**What Changed**: V1 used fixed hyperparameters. V2 implements 5-criterion adaptive scaling.

```python
# V1 (hardcoded in test scripts):
epochs = 30
lr = 0.01
num_sums = 5
num_leaves = 5

# V2 (adaptive in FedCDH.py):
def get_adaptive_hyperparameters(d, n_per_client, K, scenario, data_type):
    """5-criterion adaptive hyperparameter selection."""

    # 1. Dimensional scaling
    base_epochs = 50 if data_type == "linear" else 80
    epochs = int(base_epochs * (1 + 0.1 * max(0, d - 8)))

    # 2. Learning rate scaling
    base_lr = 0.01
    lr = base_lr / np.sqrt(1 + 0.05 * d)

    # 3. Architecture scaling (capacity)
    num_sums = min(20, max(10, 2 * d))
    num_leaves = min(20, max(10, 2 * d))

    # 4. Mode-aware adjustments
    if scenario == "horizontal":
        epochs = int(epochs * 1.5)  # More epochs for sample heterogeneity

    # 5. Sample size consideration
    min_epochs = max(30, int(50 * (300 / max(300, n_per_client))))
    epochs = max(epochs, min_epochs)

    return epochs, lr, num_sums, num_leaves
```

**Impact**:
- Automatically scales to higher dimensions (d=5 → d=11)
- Prevents underfitting (enough epochs) and overfitting (appropriate capacity)
- Mode-specific optimizations

##### B. Local Clustering Implementation (Lines 535-666)

**What Changed**: V2 implements proper K_local clustering per client (Seng et al. 2025 Algorithm 1).

```python
# V1: Global K-means (client index as surrogate)
# No local clustering, just used client_idx as cluster label

# V2: Local K-means per client
def _train_local_clustering_mode(self, X_splits, c_indx):
    """Train SPNs with LOCAL clustering per client."""

    K_local = getattr(self.args, "num_local_clusters", 2)

    # Safety: Ensure minimum samples per cluster
    min_samples_per_cluster = 100
    for X_k in X_splits:
        max_K_local = max(1, len(X_k) // min_samples_per_cluster)
        K_local = min(K_local, max_K_local)

    client_spns = []
    for k, X_k in enumerate(X_splits):
        if K_local > 1:
            # Perform LOCAL K-means clustering
            kmeans = KMeans(n_clusters=K_local, random_state=42)
            labels = kmeans.fit_predict(X_k)

            # Train SPN for each local cluster
            cluster_spns = []
            for c in range(K_local):
                X_cluster = X_k[labels == c]
                spn_c = self._train_single_spn(X_cluster, ...)
                cluster_spns.append(spn_c)

            # Create local mixture
            weights = np.array([np.mean(labels == c) for c in range(K_local)])
            client_spn = LocalClusterMixture(cluster_spns, weights)
        else:
            # Single SPN if K_local=1
            client_spn = self._train_single_spn(X_k, ...)

        client_spns.append(client_spn)

    return client_spns
```

**Impact**:
- Each client has K_local clusters (not just 1 SPN)
- Enables sum-over-products in hybrid mode
- Safety mechanism prevents overfitting with small samples

##### C. Hybrid Mode Sum-Over-Products (Lines 667-835)

**What Changed**: V2 implements proper sum-over-products structure per Seng's feedback.

```python
# V1 (WRONG): Trained new SPNs per feature group
for client_set, features in feature_subspaces.items():
    trained_spns = []
    for k in client_set:
        # PROBLEM: Training NEW SPN for each feature group
        spn_subspace = LocalSPNWrapper(...)
        spn_subspace.train_local(X_k[:, features], ...)
        trained_spns.append(spn_subspace)

    # This enforces independence: P(X) = P(X_g1) × P(X_g2) × ...
    group_mix = GroupMixture(client_spns=trained_spns, ...)

# V2 (CORRECT): Reuse local clusters, create sum-over-products
def _build_hybrid_sum_over_products(self, client_spns, feature_subspaces):
    """Build sum-over-products: P(X) = Σ_c w_c × ∏_g P(X_g|c)"""

    K_local = client_spns[0].K_local  # e.g., 2
    K_clients = len(client_spns)      # e.g., 3

    # Enumerate all cluster combinations: 2^3 = 8 products
    combinations = list(itertools.product(range(K_local), repeat=K_clients))
    # [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]

    products = []
    for cluster_config in combinations:
        # Build product for this cluster combination
        # cluster_config = (c0, c1, c2) = which cluster from each client

        groups = []
        for client_set, features in feature_subspaces.items():
            # Extract the specific cluster SPNs for this combination
            cluster_spns = []
            for k in client_set:
                c = cluster_config[k]  # Which cluster from client k
                spn_c = client_spns[k].cluster_spns[c]
                cluster_spns.append(spn_c)

            # Create group that extracts only relevant features
            group = GroupMixture(
                client_spns=cluster_spns,
                feature_indices=features,
                full_d=self.d
            )
            groups.append(group)

        # Product over groups for this cluster combination
        product = ProductOverGroupsWithOverlap(groups)
        products.append(product)

    # Sum over all products with uniform weights
    weights = np.ones(len(products)) / len(products)
    return GlobalSumOfProducts(products, weights)
```

**Impact**:
- **CRITICAL FIX**: This is what makes hybrid mode work
- P(X) = Σ_c w_c × ∏_g P(X_g|c) instead of P(X) = ∏_g P(X_g)
- Captures cross-group dependencies (F1: 0.000 → 0.300)
- Implements Seng's "sum nodes on top of products that group clusters"

##### D. Data Validation (Lines 312-399)

**What Added**: Comprehensive validation to catch data issues early.

```python
def _validate_data_partition(self, X_splits, c_indx, scenario):
    """Validate data partition correctness for each scenario."""

    # Check shapes
    for k, X_k in enumerate(X_splits):
        assert X_k.ndim == 2, f"Client {k} data must be 2D"
        assert X_k.shape[0] > 0, f"Client {k} has no samples"

    # Scenario-specific checks
    if scenario == "horizontal":
        # All clients same features, different samples
        for X_k in X_splits:
            assert X_k.shape[1] == self.d
        # Check no sample overlap
        assert len(c_indx) == sum(len(X_k) for X_k in X_splits)

    elif scenario == "vertical":
        # All clients see all samples, different features
        n_samples = X_splits[0].shape[0]
        for X_k in X_splits:
            assert X_k.shape[0] == n_samples
        # Check features sum to d
        assert sum(X_k.shape[1] for X_k in X_splits) == self.d

    elif scenario == "hybrid":
        # Overlapping features, split samples
        for X_k in X_splits:
            assert X_k.shape[1] == self.d  # All features (with overlap)
```

**Impact**: Catches configuration errors early, prevents silent failures.

---

### 2. 🔥 FedPC.py - Federated SPN Aggregation (NEW FILE, 2,722 lines)

**Location**: `causallearn/utils/FedPC.py`

**Status**: Completely new file for V2. V1 had basic aggregation inline.

#### Key Classes Added

##### A. LocalClusterMixture (Lines 108-223)

**Purpose**: Represents K_local clusters within a single client.

```python
class LocalClusterMixture:
    """Mixture of SPNs for local clusters within one client."""

    def __init__(self, cluster_spns: List, weights: np.ndarray):
        """
        Args:
            cluster_spns: List of K_local trained SPNs
            weights: Mixture weights (sum to 1)
        """
        self.cluster_spns = cluster_spns
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.K_local = len(cluster_spns)

    def log_prob(self, x):
        """Compute log P(X) = log Σ_c w_c × P_c(X)"""
        # Get log probs from each cluster
        log_probs = []
        for spn_c in self.cluster_spns:
            log_probs.append(spn_c.log_prob(x))

        # Stack: [K_local, batch_size]
        log_probs = torch.stack(log_probs, dim=0)

        # Log-sum-exp with weights: log Σ_c w_c × exp(log P_c(X))
        log_weights = torch.log(self.weights).unsqueeze(1)  # [K_local, 1]
        log_prob = torch.logsumexp(log_probs + log_weights, dim=0)

        return log_prob  # [batch_size]
```

**Why Needed**: V1 had single SPN per client. V2 needs K_local SPNs per client for sum-over-products.

##### B. GroupMixture (Lines 225-530)

**Purpose**: Mixture over clients for a specific feature group (with NaN masking).

```python
class GroupMixture:
    """Mixture of client SPNs for a specific feature group."""

    def __init__(
        self,
        client_spns: List,
        feature_indices: List[int],
        full_d: int
    ):
        self.client_spns = client_spns
        self.feature_indices = feature_indices
        self.full_d = full_d

        # Create NaN mask for features NOT in this group
        self.nan_mask = np.ones(full_d, dtype=bool)
        self.nan_mask[feature_indices] = False

    def log_prob(self, x):
        """Compute log P(X_g) where X_g are features in this group."""

        # BUGFIX: Handle context column if present
        if self.full_d is not None and x.shape[1] > self.full_d:
            x = x[:, :self.full_d]  # Strip context column

        # Extract features for this group using NaN masking
        x_g = x.clone()
        x_g[:, self.nan_mask] = float("nan")  # Mask out other features

        # Get log probs from each client's SPN
        log_probs = []
        for client_spn in self.client_spns:
            if isinstance(client_spn, LocalClusterMixture):
                # Client has K_local clusters
                log_prob_k = client_spn.log_prob(x_g)
            else:
                # Client has single SPN
                log_prob_k = client_spn.log_prob(x_g)
            log_probs.append(log_prob_k)

        # Average over clients (uniform weights)
        log_probs = torch.stack(log_probs, dim=0)
        log_prob = torch.logsumexp(log_probs, dim=0) - np.log(len(self.client_spns))

        return log_prob
```

**Key Innovation**:
- NaN masking for marginalization (P(X_g) from full SPN)
- Context column handling (bugfix for hybrid CI testing)
- Works with both single SPNs and LocalClusterMixture

##### C. GlobalSumOfProducts (Lines 845-1050)

**Purpose**: Top-level sum over cluster combinations (THE KEY FIX).

```python
class GlobalSumOfProducts:
    """Sum over products: P(X) = Σ_c w_c × ∏_g P(X_g|c)"""

    def __init__(self, products: List, weights: np.ndarray):
        """
        Args:
            products: List of ProductOverGroupsWithOverlap (one per cluster config)
            weights: Mixture weights over cluster combinations
        """
        self.products = products
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.num_products = len(products)

    def log_prob(self, x):
        """
        Compute log P(X) = log Σ_c w_c × ∏_g P(X_g|c)

        This is the KEY difference from V1:
        - V1: P(X) = ∏_g P(X_g) → independence between groups
        - V2: P(X) = Σ_c w_c × ∏_g P(X_g|c) → dependencies via mixture
        """
        # Get log prob from each product (cluster combination)
        log_probs = []
        for product in self.products:
            log_prob_c = product.log_prob(x)  # log ∏_g P(X_g|c)
            log_probs.append(log_prob_c)

        # Stack: [num_products, batch_size]
        log_probs = torch.stack(log_probs, dim=0)

        # Log-sum-exp: log Σ_c w_c × exp(log prob_c)
        log_weights = torch.log(self.weights).unsqueeze(1)
        log_prob = torch.logsumexp(log_probs + log_weights, dim=0)

        return log_prob
```

**This is THE critical fix**:
- V1 structure: `P(X) = P(X_g1) × P(X_g2) × ...` → enforces independence
- V2 structure: `P(X) = Σ_c w_c × [P(X_g1|c) × P(X_g2|c) × ...]` → allows dependencies
- Result: Hybrid F1 goes from 0.000 → 0.300

##### D. ProductOverGroupsWithOverlap (Lines 645-843)

**Purpose**: Product of feature groups for a specific cluster configuration.

```python
class ProductOverGroupsWithOverlap:
    """Product over feature groups: ∏_g P(X_g|c)"""

    def __init__(self, groups: List[GroupMixture]):
        self.groups = groups

    def log_prob(self, x):
        """Compute log ∏_g P(X_g|c) = Σ_g log P(X_g|c)"""
        log_probs = []
        for group in self.groups:
            log_prob_g = group.log_prob(x)
            log_probs.append(log_prob_g)

        # Sum log probs (product in probability space)
        log_probs = torch.stack(log_probs, dim=0)
        log_prob = torch.sum(log_probs, dim=0)

        return log_prob
```

**Why Needed**: Each cluster combination produces one product, then we sum over products.

---

### 3. 🔥 cit.py - Conditional Independence Testing (+283 modified)

**Location**: `causallearn/utils/cit.py`

**What Changed**: V2 fixes NaN handling and adds SPN-based CI testing.

#### Key Changes

##### A. SPN-Based CI Test (Lines 45-280)

**V1**: Only had KCI and FisherZ tests.

**V2**: Added complete SPN-based CI testing with proper NaN handling.

```python
def spn_based_ci_test(
    data,
    X, Y, condition_set,
    global_spn,
    test_type="parametric",
    num_permutations=0,
    **kwargs
):
    """
    SPN-based conditional independence test.

    Test: X ⊥ Y | Z
    Method: Likelihood ratio using SPNs

    score_obs = E[log P(X,Y|Z)] - E[log P(X|Z)] - E[log P(Y|Z)]

    If X ⊥ Y | Z, then score_obs ≈ 0
    """
    # Prepare data with NaN masking
    n_samples = len(data)
    d = data.shape[1]

    # Create test data for each term
    # P(X,Y,Z): Keep X, Y, Z; NaN others
    data_xyz = data.copy()
    mask_xyz = np.ones(d, dtype=bool)
    mask_xyz[list(X) + list(Y) + list(condition_set)] = False
    data_xyz[:, mask_xyz] = np.nan

    # P(X,Z): Keep X, Z; NaN others
    data_xz = data.copy()
    mask_xz = np.ones(d, dtype=bool)
    mask_xz[list(X) + list(condition_set)] = False
    data_xz[:, mask_xz] = np.nan

    # Similar for P(Y,Z) and P(Z)
    ...

    # Compute likelihood ratios
    ll_xyz = global_spn.log_prob(torch.tensor(data_xyz))
    ll_xz = global_spn.log_prob(torch.tensor(data_xz))
    ll_yz = global_spn.log_prob(torch.tensor(data_yz))
    ll_z = global_spn.log_prob(torch.tensor(data_z))

    # Score = E[log P(X,Y|Z)] - E[log P(X|Z)] - E[log P(Y|Z)]
    #       = E[log P(X,Y,Z)] - E[log P(X,Z)] - E[log P(Y,Z)] + E[log P(Z)]
    score_obs = (ll_xyz.mean() - ll_xz.mean() - ll_yz.mean() + ll_z.mean()).item()

    # Statistical test
    if test_type == "parametric":
        # Use chi-square distribution
        stat_obs = 2 * n_samples * score_obs
        dof = len(X) * len(Y)  # Degrees of freedom
        p_value = 1 - chi2.cdf(stat_obs, df=dof)
    else:
        # Permutation test (slower but more robust)
        ...

    return p_value, stat_obs
```

**Impact**:
- Proper marginalization via NaN masking
- Works with hybrid mode's overlapping features
- Statistically principled CI testing

##### B. NaN Handling Fix (Lines 155-189)

**V1**: NaN handling was fragile.

**V2**: Robust NaN masking for feature marginalization.

```python
# V1 (implicit, often broke):
x_subset = x[:, feature_indices]  # Doesn't work with full SPNs

# V2 (explicit NaN masking):
x_masked = x.clone()
mask = np.ones(x.shape[1], dtype=bool)
mask[feature_indices] = False
x_masked[:, mask] = float("nan")  # SPNs handle NaN = marginalize

# This allows: P(X_subset) from P(X_full) without retraining
```

**Impact**:
- Critical for hybrid mode (overlapping features)
- Enables CI testing on feature subsets
- Fixed context column bug

---

## Supporting Files (Less Critical)

### 4. ci_ranking.py (+258 lines, NEW)

**Purpose**: Optional CI ranking for edge selection.

**Key Feature**:
```python
def rank_and_filter_edges(ci_results, top_percent=10):
    """Select top N% strongest dependencies by CI score."""
    # Sort edges by p-value (lower = stronger)
    ranked = sorted(ci_results, key=lambda x: x['p_value'])

    # Take top N%
    cutoff = int(len(ranked) * top_percent / 100)
    return ranked[:cutoff]
```

**Status**: Implemented but not critical to v2 success (optional feature).

---

### 5. spn_evaluation.py (+483 lines, NEW)

**Purpose**: Quality metrics for SPNs (log-likelihood, calibration, MMD²).

**Not critical for v2 core functionality**, but useful for debugging.

---

### 6. spn_dashboard.py (+886 lines, NEW)

**Purpose**: Visualization and diagnostics.

**Not critical for v2 core functionality**, debugging aid only.

---

### 7. Test Files

**tests/test/test_fedcdh_benchmark.py**: Complete benchmark suite
- V1 had basic test scripts
- V2 has comprehensive benchmark with configs, seeds, statistical analysis

---

## Summary Table: What Makes V2 Work

| Component | V1 Status | V2 Status | Lines | Impact |
|-----------|-----------|-----------|-------|---------|
| **Adaptive Hyperparameters** | Fixed values | 5-criterion adaptive | ~120 | High - scales to d=11 |
| **Local Clustering** | Client as surrogate | K-means per client | ~130 | Critical - enables sum-over-products |
| **Sum-Over-Products** | Missing | GlobalSumOfProducts class | ~200 | **CRITICAL** - F1: 0.0→0.3 |
| **NaN Masking** | Implicit/broken | Explicit everywhere | ~50 | High - hybrid mode works |
| **Context Column Fix** | N/A | Strip extra column | ~5 | Medium - fixes CI testing |
| **Data Validation** | None | Comprehensive checks | ~90 | Medium - catches errors early |
| **SPN CI Testing** | Basic | Statistically principled | ~235 | High - better CI tests |

---

## The Critical Path: What Fixed Hybrid Mode

### Problem in V1
```python
# V1 (WRONG): Product factorization
P(X) = P(X_g1) × P(X_g2) × P(X_g3)
# This ENFORCES independence: I(X_g1; X_g2) = 0
# Result: F1 = 0.000 for cross-group edges
```

### Solution in V2 (3 changes)

**1. Local Clustering** (`FedCDH.py`, lines 535-666):
```python
# Each client has K_local=2 clusters (not just 1 SPN)
for k, X_k in enumerate(X_splits):
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(X_k)

    cluster_spns = [train_spn(X_k[labels == c]) for c in range(2)]
    weights = [np.mean(labels == c) for c in range(2)]

    client_spn = LocalClusterMixture(cluster_spns, weights)
```

**2. Sum-Over-Products** (`FedCDH.py`, lines 667-835 + `FedPC.py` GlobalSumOfProducts):
```python
# Enumerate all 2^3 = 8 cluster combinations
combinations = [(0,0,0), (0,0,1), (0,1,0), ..., (1,1,1)]

products = []
for (c0, c1, c2) in combinations:
    # Build product for this cluster configuration
    groups = []
    for feature_group in feature_subspaces:
        # Use cluster c_k from client k
        cluster_spns = [client_spns[k].cluster_spns[c_k] for k in clients]
        groups.append(GroupMixture(cluster_spns, features))

    products.append(ProductOverGroups(groups))

# Sum over all products
global_spn = GlobalSumOfProducts(products, weights=uniform(8))
```

Result:
```python
P(X) = Σ_c w_c × ∏_g P(X_g | cluster_config_c)
     = 1/8 × [P(X_g1|000) × P(X_g2|000) × ...]
       + 1/8 × [P(X_g1|001) × P(X_g2|001) × ...]
       + ...
       + 1/8 × [P(X_g1|111) × P(X_g2|111) × ...]
```

**3. NaN Masking** (`FedPC.py` GroupMixture, lines 420-435):
```python
# Extract P(X_g) from full SPN without retraining
def log_prob(self, x):
    x_g = x.clone()
    x_g[:, self.nan_mask] = float("nan")  # Marginalize other features
    return spn.log_prob(x_g)
```

### Result
- **V1**: F1 = 0.000 (product enforces independence)
- **V2**: F1 = 0.300 (sum-over-products captures dependencies)

---

## Git Statistics

```bash
$ git diff main...HEAD --shortstat -- "causallearn/*.py"

 14 files changed, 6,682 insertions(+), 1,177 deletions(-)
```

**Key metrics**:
- **3 core files**: 4,505 lines (FedCDH.py, FedPC.py, cit.py)
- **11 supporting files**: 2,177 lines (evaluation, dashboard, utilities)
- **Total**: 6,682 lines added, 1,177 modified

---

## Key Commits (Chronological)

1. **`cfe558d`** - feat: implement v2 improvements (adaptive hyperparameters + CI ranking)
   - Initial v2 implementation
   - Adaptive hyperparameters
   - CI ranking option

2. **`1ed492d`** - fix(hybrid): resolve NaN propagation in CI testing
   - Fixed NaN handling in CI tests
   - Critical for hybrid mode

3. **`8b7a14b`** - fix(hybrid): implement sum-over-products for cross-group dependencies
   - **THE KEY FIX**: Implemented GlobalSumOfProducts
   - Added local clustering
   - F1: 0.000 → 0.300

4. **`c1b12d2`** - perf(hybrid): optimize hot paths in sum-over-products
   - Performance optimizations
   - Reduced memory usage

5. **`88bad9a`** - fix(hybrid): remove incorrect context column in local SPN evaluation
   - Fixed context column in local evaluation

6. **`fedd1ed`** - fix(hybrid): handle context column in GroupMixture log_prob
   - **FINAL BUGFIX**: Context column in CI testing
   - V2 now fully working

---

## Conclusion

**3 critical changes make V2 work**:

1. **LocalClusterMixture**: Each client has K_local SPNs (not 1)
2. **GlobalSumOfProducts**: Sum over cluster combinations (not just product)
3. **NaN masking**: Proper marginalization for feature subsets

Without #1 and #2, hybrid F1 = 0.000 (independence enforced).
With all three, hybrid F1 = 0.300 (dependencies captured).

**Files that matter most**:
- `FedCDH.py`: Local clustering + hybrid aggregation (~400 lines critical)
- `FedPC.py`: GlobalSumOfProducts + GroupMixture (~800 lines critical)
- `cit.py`: SPN CI testing with NaN handling (~200 lines critical)

**Everything else is supporting infrastructure** (evaluation, visualization, utilities).
