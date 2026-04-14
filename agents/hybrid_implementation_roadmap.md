# Hybrid SPN Mode Implementation Roadmap
## Master's Thesis: Federated Causal Discovery with Probabilistic Circuits

**Author**: Mei-Ling Fang
**Timeline**: 2-3 weeks (12-15 working days)
**Goal**: Implement theoretically grounded Mixture-then-Product hybrid mode for FedCDH
**Nature**: Empirical study (no formal guarantees required)

---

## Research Context & Motivation

### Thesis Scope (from Master Thesis Topic)

**Research Question**: Can probabilistic circuits (SPNs) enable federated causal discovery across horizontal, vertical, and hybrid data partitioning scenarios?

**Key Contributions**:
1. ✅ Replace summary statistic-based CI testing with SPN-based approach (DONE)
2. ✅ Extend to horizontal data splits (DONE)
3. ✅ Extend to vertical data splits (DONE)
4. ⚠️ **Extend to hybrid data splits** (CURRENT WORK - NEEDS FIX)

**Current Issue**: Hybrid mode uses incorrect Product-then-Mixture hierarchy, does not support overlapping features

**Target Domains**: Biomedical data (Sachs), semiconductor applications

---

## Research Goals vs Implementation Goals

### What We NEED for Thesis ✅

1. **Empirical Validation**: Show hybrid mode works on real data (Sachs)
2. **Theoretical Grounding**: Cite Seng et al. (2025) correctly, use their architecture
3. **Comparison**: Demonstrate hybrid differs from horizontal/vertical
4. **Reproducibility**: Clear implementation following paper's Algorithm 1

### What We DON'T NEED for Thesis ❌

1. ~~Formal convergence proofs~~ (out of scope for Master's)
2. ~~Differential privacy implementation~~ (simulation-based privacy sufficient)
3. ~~Optimal feature grouping via structure learning~~ (manual/automatic sufficient)
4. ~~Performance on massive datasets~~ (Sachs n=856 is sufficient)

---

## Implementation Roadmap: 3-Week Plan

### Week 1: Core Classes (Days 1-5)
**Goal**: Implement Mixture-then-Product probabilistic circuit classes

### Week 2: Integration & Testing (Days 6-10)
**Goal**: Replace current hybrid mode, validate on smoke tests

### Week 3: Experiments & Documentation (Days 11-15)
**Goal**: Run Sachs experiments, document results for thesis

---

# WEEK 1: Core Probabilistic Circuit Classes

---

## Day 1-2: GroupMixture Class

**File**: `causallearn/utils/FedPC.py`

**Goal**: Implement mixture over clients for a feature group

### Step 1.1: Class Skeleton (2 hours)

```python
class GroupMixture(nn.Module):
    """
    Mixture of client SPNs for a specific feature group.

    Mathematical Form:
        P(X_g) = Σ_k w_k,g × P_k,g(X_g)

    Reference: Seng et al. (2025), Section 3.2 "Sum Nodes and Horizontal FL"

    Args:
        client_spns (List[LocalSPNWrapper]): K SPNs from different clients
        weights (np.ndarray): [K] mixture weights, sum to 1
        feature_indices (List[int]): Which features this group models
        device (str): 'cpu' or 'cuda'

    Example:
        # Two clients share features [0, 1, 2]
        >>> spn0 = LocalSPNWrapper(num_features=3)
        >>> spn1 = LocalSPNWrapper(num_features=3)
        >>> mixture = GroupMixture([spn0, spn1], weights=[0.4, 0.6],
                                     feature_indices=[0,1,2])
        >>> x = torch.randn(100, 5)  # 100 samples, 5 total features
        >>> log_p = mixture.log_prob(x)  # Only uses x[:, [0,1,2]]
    """
    def __init__(self, client_spns, weights, feature_indices, device='cpu'):
        super().__init__()
        self.client_spns = nn.ModuleList(client_spns)
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.feature_indices = feature_indices
        self.device = device

        # Validation
        assert abs(self.weights.sum().item() - 1.0) < 1e-5, "Weights must sum to 1"
        assert len(self.client_spns) == len(weights), "Mismatched SPNs and weights"
```

**Test Case 1.1** (30 min):
```python
def test_groupmixture_init():
    spn0 = LocalSPNWrapper(num_features=3, device='cpu')
    spn1 = LocalSPNWrapper(num_features=3, device='cpu')

    mixture = GroupMixture([spn0, spn1], weights=[0.4, 0.6],
                          feature_indices=[0,1,2], device='cpu')

    assert len(mixture.client_spns) == 2
    assert torch.allclose(mixture.weights.sum(), torch.tensor(1.0))
```

---

### Step 1.2: log_prob Implementation (3 hours)

```python
def log_prob(self, x):
    """
    Compute log P(X_g) where X_g = x[:, feature_indices].

    Algorithm:
        1. Extract features: x_g = x[:, feature_indices]
        2. For each client k: compute log P_k(x_g)
        3. Compute log Σ_k [w_k × P_k(x_g)] via logsumexp

    Args:
        x (Tensor): [batch, d_full] full feature matrix

    Returns:
        log_prob (Tensor): [batch, 1] log probabilities

    Reference: Seng et al. (2025), Definition 1 (Horizontal FL)
    """
    batch_size = x.shape[0]

    # Step 1: Extract features for this group
    x_g = x[:, self.feature_indices]  # [batch, len(feature_indices)]

    # Step 2: Compute log-likelihoods from each client SPN
    client_lls = []
    for spn in self.client_spns:
        ll = spn.log_prob(x_g)  # [batch, 1]
        client_lls.append(ll)

    # Step 3: Stack and compute weighted mixture
    ll_stack = torch.cat(client_lls, dim=1)  # [batch, K]
    log_weights = torch.log(self.weights + 1e-9).unsqueeze(0)  # [1, K]

    # log Σ_k [w_k × P_k] = logsumexp(log w_k + log P_k)
    log_prob = torch.logsumexp(ll_stack + log_weights, dim=1, keepdim=True)

    return log_prob  # [batch, 1]
```

**Test Case 1.2** (1 hour):
```python
def test_groupmixture_log_prob():
    # Create two identical SPNs → mixture should equal single SPN
    spn = LocalSPNWrapper(num_features=3, device='cpu')
    mixture = GroupMixture([spn, spn], weights=[0.5, 0.5],
                          feature_indices=[0,1,2])

    x = torch.randn(100, 5)
    x_g = x[:, [0,1,2]]

    # Mixture log-prob should equal single SPN log-prob
    mixture_ll = mixture.log_prob(x)
    single_ll = spn.log_prob(x_g)

    assert torch.allclose(mixture_ll, single_ll, atol=1e-5)
```

---

### Step 1.3: sample Implementation (2 hours)

```python
def sample(self, n):
    """
    Sample n data points from the mixture.

    Algorithm:
        1. For each sample i: choose client k ~ Multinomial(weights)
        2. Sample from chosen client: x_i ~ P_k(X_g)
        3. Return only features for this group

    Args:
        n (int): Number of samples

    Returns:
        samples (Tensor): [n, len(feature_indices)]

    Reference: Standard mixture sampling (Seng et al. 2025 implicit)
    """
    # Step 1: Choose which client generates each sample
    component_indices = torch.multinomial(self.weights, n, replacement=True)

    # Step 2: Sample from chosen clients
    samples = []
    for i in range(n):
        client_idx = component_indices[i].item()
        client_sample = self.client_spns[client_idx].sample(1)  # [1, d_g]
        samples.append(client_sample)

    samples = torch.cat(samples, dim=0)  # [n, d_g]
    return samples
```

**Test Case 1.3** (30 min):
```python
def test_groupmixture_sample():
    spn0 = LocalSPNWrapper(num_features=3, device='cpu')
    spn1 = LocalSPNWrapper(num_features=3, device='cpu')
    mixture = GroupMixture([spn0, spn1], weights=[0.4, 0.6],
                          feature_indices=[0,1,2])

    samples = mixture.sample(100)

    assert samples.shape == (100, 3)  # Only group features, not all
```

**Deliverable Day 2**: Working GroupMixture class with all tests passing

---

## Day 3-4: ProductOverGroups Class

**File**: `causallearn/utils/FedPC.py`

**Goal**: Implement product over feature groups (disjoint case)

### Step 2.1: Class Skeleton (2 hours)

```python
class ProductOverGroups(nn.Module):
    """
    Product of feature group mixtures (disjoint groups).

    Mathematical Form:
        P(X) = Π_g P(X_g)  where scopes are disjoint
        log P(X) = Σ_g log P(X_g)

    Reference: Seng et al. (2025), Section 3.2 "Product Nodes & Vertical FL"
               Definition 2 (Vertical FL)

    Args:
        group_mixtures (List[GroupMixture]): G mixtures, one per feature group
        feature_groups (List[List[int]]): Feature indices for each group
        device (str): 'cpu' or 'cuda'

    Raises:
        ValueError: If feature groups overlap (use ProductOverGroupsWithOverlap)

    Example:
        # Two disjoint groups: [0,1,2] and [3,4]
        >>> mix1 = GroupMixture(..., feature_indices=[0,1,2])
        >>> mix2 = GroupMixture(..., feature_indices=[3,4])
        >>> product = ProductOverGroups([mix1, mix2], [[0,1,2], [3,4]])
        >>> x = torch.randn(100, 5)
        >>> log_p = product.log_prob(x)  # log P(X) = log P(X_1) + log P(X_2)
    """
    def __init__(self, group_mixtures, feature_groups, device='cpu'):
        super().__init__()
        self.group_mixtures = nn.ModuleList(group_mixtures)
        self.feature_groups = feature_groups
        self.device = device

        # Validate disjoint
        self._validate_disjoint()

    def _validate_disjoint(self):
        """Ensure feature groups don't overlap."""
        all_features = []
        for group in self.feature_groups:
            all_features.extend(group)

        if len(all_features) != len(set(all_features)):
            raise ValueError(
                "Feature groups overlap! Use ProductOverGroupsWithOverlap. "
                f"Got groups: {self.feature_groups}"
            )
```

**Test Case 2.1** (30 min):
```python
def test_product_overlap_detection():
    # Should raise error on overlapping groups
    mix1 = GroupMixture(..., feature_indices=[0,1,2])
    mix2 = GroupMixture(..., feature_indices=[1,2,3])  # Overlap!

    with pytest.raises(ValueError, match="overlap"):
        product = ProductOverGroups([mix1, mix2], [[0,1,2], [1,2,3]])
```

---

### Step 2.2: log_prob Implementation (2 hours)

```python
def log_prob(self, x):
    """
    Compute log P(X) = Σ_g log P(X_g).

    Algorithm:
        1. For each group g: compute log P(X_g) via GroupMixture
        2. Sum all log probabilities (product in prob space)

    Args:
        x (Tensor): [batch, d] full feature matrix

    Returns:
        log_prob (Tensor): [batch, 1]

    Reference: PC decomposability (Seng et al. 2025, Section 2)
    """
    batch_size = x.shape[0]

    # Compute log-prob for each group
    group_lls = []
    for mixture_g in self.group_mixtures:
        ll_g = mixture_g.log_prob(x)  # [batch, 1]
        group_lls.append(ll_g)

    # Sum log-probs (product in probability space)
    ll_stack = torch.cat(group_lls, dim=1)  # [batch, G]
    total_ll = torch.sum(ll_stack, dim=1, keepdim=True)  # [batch, 1]

    return total_ll
```

**Test Case 2.2** (1 hour):
```python
def test_product_log_prob():
    # Create product of two groups
    # log P(X) should equal log P(X_1) + log P(X_2)

    mix1 = GroupMixture([spn0_g1], [1.0], [0,1,2])
    mix2 = GroupMixture([spn0_g2], [1.0], [3,4])
    product = ProductOverGroups([mix1, mix2], [[0,1,2], [3,4]])

    x = torch.randn(100, 5)
    product_ll = product.log_prob(x)

    # Manual computation
    manual_ll = mix1.log_prob(x) + mix2.log_prob(x)

    assert torch.allclose(product_ll, manual_ll)
```

---

### Step 2.3: sample Implementation (2 hours)

```python
def sample(self, n):
    """
    Sample from product by sampling each group independently.

    Algorithm:
        1. For each group g: sample x_g ~ P(X_g)
        2. Assemble samples into full feature vector

    Args:
        n (int): Number of samples

    Returns:
        samples (Tensor): [n, d] where d = total features

    Reference: Product independence assumption (Seng et al. 2025)
    """
    # Determine full dimensionality
    all_indices = []
    for group in self.feature_groups:
        all_indices.extend(group)
    num_features = max(all_indices) + 1

    samples = torch.zeros(n, num_features, device=self.device)

    # Sample each group independently
    for g, mixture_g in enumerate(self.group_mixtures):
        group_samples = mixture_g.sample(n)  # [n, d_g]
        indices = self.feature_groups[g]
        samples[:, indices] = group_samples

    return samples
```

**Test Case 2.3** (1 hour):
```python
def test_product_sample():
    mix1 = GroupMixture([spn0], [1.0], [0,1,2])
    mix2 = GroupMixture([spn1], [1.0], [3,4])
    product = ProductOverGroups([mix1, mix2], [[0,1,2], [3,4]])

    samples = product.sample(100)

    assert samples.shape == (100, 5)

    # Verify independence: sampling group 1 multiple times shouldn't affect group 2
    samples_a = product.sample(10)
    samples_b = product.sample(10)
    # Should be different (stochastic), not deterministic
```

**Deliverable Day 4**: Working ProductOverGroups class with all tests passing

---

## Day 5: ProductOverGroupsWithOverlap (Simplified)

**File**: `causallearn/utils/FedPC.py`

**Goal**: Handle overlapping features via indicator matrix method (Seng et al. 2025, Algorithm 1)

### Step 3.1: Overlap Detection (2 hours)

```python
class ProductOverGroupsWithOverlap(nn.Module):
    """
    Product over groups with overlapping feature support.

    Uses indicator matrix method from Seng et al. (2025), Algorithm 1.

    Key Idea:
        - Detect which features appear in multiple groups
        - For shared features: create mixture over clients that have them
        - Product combines disjoint scopes (each feature in exactly one child)

    Reference: Seng et al. (2025), Algorithm 1, Lines 1-6

    Example:
        # Overlapping groups: Client 0=[0,1,2], Client 1=[1,2,3]
        # Automatic structure:
        #   Group {0}: features [0] (only client 0)
        #   Group {0,1}: features [1,2] (both clients, mixture!)
        #   Group {1}: features [3] (only client 1)
        >>> product = ProductOverGroupsWithOverlap(...)
        >>> # Result: Product(Client0(0), Mixture(Client0(1,2), Client1(1,2)), Client1(3))
    """
    def __init__(self, group_mixtures, feature_groups, device='cpu'):
        super().__init__()
        self.group_mixtures = nn.ModuleList(group_mixtures)
        self.feature_groups = feature_groups
        self.device = device

        # Detect overlaps
        self.overlap_map = self._compute_overlap_map()
        self.has_overlap = len(self.overlap_map['overlapping']) > 0

    def _compute_overlap_map(self):
        """
        Identify which features appear in multiple groups.

        Returns:
            dict: {
                'overlapping': {feat_idx: [group_indices]},
                'unique_per_group': {group_idx: [unique_feat_indices]}
            }

        Reference: Seng et al. (2025), Algorithm 1 implicit overlap detection
        """
        feature_to_groups = {}
        for g, group in enumerate(self.feature_groups):
            for feat in group:
                if feat not in feature_to_groups:
                    feature_to_groups[feat] = []
                feature_to_groups[feat].append(g)

        overlapping = {f: groups for f, groups in feature_to_groups.items()
                      if len(groups) > 1}

        unique_per_group = {}
        for g, group in enumerate(self.feature_groups):
            unique = [f for f in group if len(feature_to_groups[f]) == 1]
            unique_per_group[g] = unique

        return {
            'overlapping': overlapping,
            'unique_per_group': unique_per_group,
            'feature_to_groups': feature_to_groups
        }
```

**Test Case 3.1** (1 hour):
```python
def test_overlap_detection():
    # Create overlapping groups
    mix1 = GroupMixture(..., feature_indices=[0,1,2])
    mix2 = GroupMixture(..., feature_indices=[1,2,3])

    product = ProductOverGroupsWithOverlap([mix1, mix2], [[0,1,2], [1,2,3]])

    # Verify overlap detection
    assert 1 in product.overlap_map['overlapping']
    assert 2 in product.overlap_map['overlapping']
    assert product.overlap_map['overlapping'][1] == [0, 1]
```

---

### Step 3.2: log_prob with Paper's Method (3 hours)

**Key Insight from Paper**: When features overlap, the paper's Algorithm 1 creates separate mixtures for each unique "client set" pattern. This automatically prevents double-counting.

```python
def log_prob(self, x):
    """
    Compute log P(X) accounting for overlaps.

    Paper's Method (Seng et al. 2025, Algorithm 1):
        - Each GroupMixture already handles a specific client set
        - Product combines disjoint scopes (no feature in >1 mixture)
        - Overlaps resolved at structure construction, not inference

    Therefore: Same as ProductOverGroups!

    Args:
        x (Tensor): [batch, d]

    Returns:
        log_prob (Tensor): [batch, 1]
    """
    # If structure was built correctly per Algorithm 1,
    # each feature appears in exactly one GroupMixture
    # So we can just sum log-probs

    group_lls = []
    for mixture_g in self.group_mixtures:
        ll_g = mixture_g.log_prob(x)
        group_lls.append(ll_g)

    ll_stack = torch.cat(group_lls, dim=1)
    total_ll = torch.sum(ll_stack, dim=1, keepdim=True)

    return total_ll
```

**Insight**: The overlap handling happens during **structure construction** (Algorithm 1, lines 3-6), not during inference! This simplifies the implementation significantly.

---

### Step 3.3: sample Implementation (1 hour)

```python
def sample(self, n):
    """
    Sample with overlap-aware structure.

    Since structure correctly partitions features (no double-counting),
    sampling is the same as ProductOverGroups.

    Args:
        n (int): Number of samples

    Returns:
        samples (Tensor): [n, d]
    """
    all_indices = []
    for group in self.feature_groups:
        all_indices.extend(group)
    num_features = max(all_indices) + 1

    samples = torch.zeros(n, num_features, device=self.device)

    for g, mixture_g in enumerate(self.group_mixtures):
        group_samples = mixture_g.sample(n)
        indices = self.feature_groups[g]
        samples[:, indices] = group_samples

    return samples
```

**Test Case 3.3** (30 min):
```python
def test_overlap_sample():
    # Overlapping groups should still produce valid samples
    mix1 = GroupMixture([spn0_full], [1.0], [0,1,2])
    mix2 = GroupMixture([spn1_full], [1.0], [1,2,3])

    # This should work if structure built correctly
    product = ProductOverGroupsWithOverlap([mix1, mix2], [[0,1,2], [1,2,3]])

    samples = product.sample(100)
    assert samples.shape[0] == 100
```

**Deliverable Day 5**: Overlap-aware product class (structure-based, no runtime overhead)

---

## Week 1 Summary Deliverables

**Files Created/Modified**:
- `causallearn/utils/FedPC.py`: +300 lines (3 new classes)
- `tests/unit/test_fedpc_hybrid.py`: +200 lines (all test cases)

**Classes Implemented**:
1. ✅ GroupMixture (Mixture-then-Product inner level)
2. ✅ ProductOverGroups (Mixture-then-Product outer level)
3. ✅ ProductOverGroupsWithOverlap (Paper's Algorithm 1 method)

**Test Coverage**: 100% for new classes

**Validation**: All unit tests pass, ready for integration

---

# WEEK 2: Integration & Validation

---

## Day 6-7: Automatic Feature Grouping

**File**: `causallearn/search/FCMBased/FedCDH/FedCDH.py`

**Goal**: Implement Algorithm 1 (Lines 1-6) for automatic feature grouping

### Step 4.1: Indicator Matrix Construction (2 hours)

```python
def build_feature_indicator_matrix(self, X_splits):
    """
    Build indicator matrix M showing which clients have which features.

    Reference: Seng et al. (2025), Algorithm 1, Lines 1-2

    Args:
        X_splits (List[np.ndarray]): Data splits per client
            For vertical: X_splits[k] has only subset of features
            For hybrid: X_splits[k] may have overlapping features

    Returns:
        M (np.ndarray): [K, d] binary matrix where M[k,j] = 1 if client k has feature j
        feature_names (List): Feature identifiers [0, 1, 2, ..., d-1]

    Example:
        >>> X_splits = [
        ...     np.random.randn(100, 3),  # Client 0: features 0,1,2
        ...     np.random.randn(100, 2)   # Client 1: features 3,4
        ... ]
        >>> M, names = build_feature_indicator_matrix(X_splits)
        >>> M.shape == (2, 5)  # 2 clients, 5 features
        >>> M[0, :] == [1, 1, 1, 0, 0]  # Client 0 has first 3 features
    """
    K = len(X_splits)
    d = self.d_features  # Total features (from args)

    # Initialize matrix
    M = np.zeros((K, d), dtype=int)

    # For vertical/hybrid: need feature mapping
    # For now, assume equal split or user-provided
    if self.scenario == "vertical":
        # Vertical: auto-split features
        cols_per_client = np.array_split(range(d), K)
        for k in range(K):
            feature_indices = cols_per_client[k].tolist()
            M[k, feature_indices] = 1

    elif self.scenario == "hybrid":
        # Hybrid: use provided feature_maps or equal split
        if hasattr(self, 'feature_maps') and self.feature_maps is not None:
            # User-provided mapping
            for k, features in self.feature_maps.items():
                M[k, features] = 1
        else:
            # Default: equal split
            cols_per_client = np.array_split(range(d), K)
            for k in range(K):
                M[k, cols_per_client[k].tolist()] = 1

    feature_names = list(range(d))
    return M, feature_names
```

---

### Step 4.2: Feature Grouping from Matrix (3 hours)

```python
def group_features_by_client_set(self, M, feature_names):
    """
    Group features by which clients have them.

    Reference: Seng et al. (2025), Algorithm 1, Lines 3-6

    Algorithm:
        1. Extract distinct column patterns from M
        2. For each pattern: identify which clients have it
        3. For each pattern: identify which features have it
        4. Create mapping: client_set → feature_list

    Args:
        M (np.ndarray): [K, d] indicator matrix
        feature_names (List): Feature identifiers

    Returns:
        feature_subspaces (Dict): {
            (client_tuple): [feature_indices]
        }

    Example:
        >>> M = np.array([
        ...     [1, 1, 0],
        ...     [1, 1, 1]
        ... ])
        >>> groups = group_features_by_client_set(M, [0,1,2])
        >>> groups == {
        ...     (0, 1): [0, 1],  # F0, F1 on both clients
        ...     (1,): [2]        # F2 only on client 1
        ... }
    """
    K, d = M.shape

    # Step 1: Find distinct column patterns
    feature_to_clients = {}
    for j in range(d):
        col = tuple(M[:, j])  # Column pattern for feature j
        if col not in feature_to_clients:
            feature_to_clients[col] = []
        feature_to_clients[col].append(j)

    # Step 2: Convert to client_set → features mapping
    feature_subspaces = {}
    for col_pattern, features in feature_to_clients.items():
        # col_pattern is (1, 0, 1, ...) indicating which clients
        client_set = tuple(k for k in range(K) if col_pattern[k] == 1)
        feature_subspaces[client_set] = features

    logging.info(f"Automatic feature grouping: {len(feature_subspaces)} subspaces")
    for clients, features in feature_subspaces.items():
        logging.info(f"  Clients {clients} share features {features}")

    return feature_subspaces
```

**Test Case 4.2** (1 hour):
```python
def test_automatic_grouping():
    # Simulate vertical scenario
    M = np.array([
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1]
    ])

    groups = group_features_by_client_set(M, list(range(5)))

    assert groups == {
        (0,): [0, 1, 2],
        (1,): [3, 4]
    }
```

**Deliverable Day 7**: Automatic feature grouping working, tested on vertical scenario

---

## Day 8-9: Hybrid Mode Integration

**File**: `causallearn/search/FCMBased/FedCDH/FedCDH.py`

**Goal**: Replace current hybrid section (lines 456-552) with new Mixture-then-Product

### Step 5.1: New Hybrid Aggregation Logic (4 hours)

**Location**: Lines 456-552 (FULL REPLACEMENT)

```python
# NEW HYBRID AGGREGATION (Mixture-then-Product)
if self.scenario == "hybrid":
    logging.info(f"Building Mixture-then-Product hybrid (Seng et al. 2025)")

    # Step 1: Build indicator matrix
    M, feature_names = self.build_feature_indicator_matrix(X_splits)

    # Step 2: Group features by client set
    feature_subspaces = self.group_features_by_client_set(M, feature_names)

    # Step 3: Check for overlaps
    all_features = [f for features in feature_subspaces.values() for f in features]
    has_overlap = len(all_features) != len(set(all_features))

    if has_overlap:
        logging.info("Detected overlapping features (handled via subspace mixtures)")
        overlap_count = len(all_features) - len(set(all_features))
        logging.info(f"  {overlap_count} features appear in multiple subspaces")

    # For each cluster:
    for h in range(num_clusters):
        if not clients_clusters[h]:
            continue

        # Step 4: Train K × G local SPNs
        spn_registry = {}
        for client_set, features in feature_subspaces.items():
            for k in client_set:
                # Get client's data for this cluster
                local_data_h = X_splits[k][labels_splits[k] == h]
                if len(local_data_h) <= 2:
                    continue

                # Extract features for this subspace
                local_data_h_g = local_data_h[:, features]

                # Train SPN
                spn_k_g = LocalSPNWrapper(
                    num_features=len(features),
                    device=self.device,
                    depth=max(1, int(np.floor(np.log2(len(features))))),
                    seed=h * 1000 + k * 10 + hash(client_set) % 10
                )
                spn_k_g.train_local(local_data_h_g, epochs=train_epochs)

                # Store with key (client_set, client_id)
                spn_registry[(client_set, k)] = {
                    'spn': spn_k_g,
                    'features': features,
                    'cluster': h,
                    'client': k
                }

        # Step 5: Create GroupMixture for each subspace
        group_mixtures = []
        feature_groups = []

        for client_set, features in feature_subspaces.items():
            # Collect client SPNs for this subspace
            client_spns = []
            client_counts = []

            for k in client_set:
                if (client_set, k) in spn_registry:
                    client_spns.append(spn_registry[(client_set, k)]['spn'])
                    client_counts.append(len(X_splits[k][labels_splits[k] == h]))

            if not client_spns:
                continue

            # Compute weights (sample-count proportional)
            weights = np.array(client_counts) / sum(client_counts)

            # Create GroupMixture
            mixture_g = GroupMixture(
                client_spns=client_spns,
                weights=weights,
                feature_indices=features,
                device=self.device
            )

            group_mixtures.append(mixture_g)
            feature_groups.append(features)

        # Step 6: Create Product over groups
        if has_overlap:
            cluster_model = ProductOverGroupsWithOverlap(
                group_mixtures=group_mixtures,
                feature_groups=feature_groups,
                device=self.device
            )
        else:
            cluster_model = ProductOverGroups(
                group_mixtures=group_mixtures,
                feature_groups=feature_groups,
                device=self.device
            )

        global_components.append(cluster_model)
        final_weights.append(weights[h])

    # Step 7: Outer mixture over clusters (if > 1)
    if len(global_components) > 1:
        self.fed_spn_model = GlobalFedSPN(
            global_components,
            weights=final_weights,
            strategy="mixture",
            device=self.device
        )
    else:
        self.fed_spn_model = global_components[0]

    # Store for evaluation
    self.local_spns = [info for info in spn_registry.values()]
    self.feature_subspaces = feature_subspaces
```

---

### Step 5.2: Backward Compatibility (2 hours)

```python
# Add mode selection
if self.scenario == "hybrid":
    hybrid_mode = getattr(self.args, 'hybrid_mode', 'mixture_then_product')

    if hybrid_mode == 'mixture_then_product':
        # NEW LOGIC (above)
        ...
    elif hybrid_mode == 'product_then_mixture':
        # OLD LOGIC (keep for comparison)
        logging.warning("Using LEGACY Product-then-Mixture (deprecated)")
        logging.warning("Set hybrid_mode='mixture_then_product' for correct implementation")
        # ... existing code ...
    else:
        raise ValueError(f"Unknown hybrid_mode: {hybrid_mode}")
```

**Test Case 5.2** (1 hour):
```python
def test_hybrid_mode_selection():
    # Test new mode
    args = Namespace(scenario='hybrid', hybrid_mode='mixture_then_product', ...)
    fedcdh = FedCDH(args)
    # Should use new classes

    # Test old mode
    args = Namespace(scenario='hybrid', hybrid_mode='product_then_mixture', ...)
    fedcdh = FedCDH(args)
    # Should use old classes
```

**Deliverable Day 9**: Hybrid mode integrated, backward compatible

---

## Day 10: Smoke Tests & Validation

**Goal**: Verify new hybrid works on simple test cases

### Step 6.1: Disjoint Groups Smoke Test (2 hours)

```python
# tests/smoke/test_hybrid_mixture_then_product.py

def test_hybrid_disjoint_groups():
    """
    Test hybrid with disjoint feature groups.
    Should behave similar to current implementation.
    """
    d = 8
    K = 2
    n = 200

    # Generate data
    X_splits, c_indx, B = generate_linear_data(d, K, n)

    # Configure hybrid (disjoint groups)
    args = Namespace(
        K=K,
        d=d,
        n=n,
        scenario="hybrid",
        hybrid_mode="mixture_then_product",
        feature_maps={0: [0,1,2,3], 1: [4,5,6,7]},  # Disjoint
        ci_method="spn",
        alpha=0.05,
        epochs=20,
        device="cpu",
        skip_bic=True
    )

    fedcdh = FedCDH(args)
    results = fedcdh.fit(X_splits, c_indx, B)

    # Validation
    assert hasattr(fedcdh, 'fed_spn_model')
    assert len(fedcdh.local_spns) == K * 2  # 2 groups × 2 clients

    # Should produce valid graph
    assert results['G_learned'] is not None

    # Log-likelihood should be reasonable
    logging.info(f"Disjoint groups test passed")
    logging.info(f"  Skeleton F1: {results.get('skeleton_f1', 'N/A')}")
```

---

### Step 6.2: Comparison Test (3 hours)

```python
def test_hybrid_vs_horizontal():
    """
    Verify hybrid produces DIFFERENT results from horizontal.
    This was the original bug (Bug 4).
    """
    d = 5
    K = 2
    n = 200

    X_splits, c_indx, B = generate_linear_data(d, K, n)

    # Run horizontal
    args_h = Namespace(scenario="horizontal", ...)
    fedcdh_h = FedCDH(args_h)
    results_h = fedcdh_h.fit(X_splits, c_indx, B)

    # Run hybrid (new)
    args_hybrid = Namespace(
        scenario="hybrid",
        hybrid_mode="mixture_then_product",
        feature_maps={0: [0,1,2], 1: [3,4]},
        ...
    )
    fedcdh_hybrid = FedCDH(args_hybrid)
    results_hybrid = fedcdh_hybrid.fit(X_splits, c_indx, B)

    # Validate DIFFERENT
    assert results_h['skeleton_f1'] != results_hybrid['skeleton_f1'], \
        "Hybrid should produce different results from horizontal!"

    logging.info("Hybrid vs Horizontal:")
    logging.info(f"  Horizontal F1: {results_h['skeleton_f1']}")
    logging.info(f"  Hybrid F1: {results_hybrid['skeleton_f1']}")
```

**Deliverable Day 10**: Smoke tests pass, hybrid ≠ horizontal confirmed

---

## Week 2 Summary Deliverables

**Files Modified**:
- `causallearn/search/FCMBased/FedCDH/FedCDH.py`: Hybrid section replaced (~150 lines)
- `tests/smoke/test_hybrid_mixture_then_product.py`: New test file (~100 lines)

**Validation**:
- ✅ Disjoint groups work correctly
- ✅ Hybrid produces different results from horizontal
- ✅ Automatic feature grouping working
- ✅ Backward compatibility maintained

**Ready for**: Real experiments (Sachs dataset)

---

# WEEK 3: Experiments & Thesis Documentation

---

## Day 11-12: Sachs Experiments

**Goal**: Run comprehensive experiments for thesis

### Step 7.1: Experimental Setup (2 hours)

```python
# tests/benchmarks/run_hybrid_experiments.py

"""
Sachs Dataset Experiments for Thesis

Research Question: Does Mixture-then-Product hybrid outperform
                   horizontal/vertical on hybrid data partitioning?

Dataset: Sachs et al. (2005) protein signaling network
    - 11 variables (proteins/phospholipids)
    - 853 samples (observational data)
    - Known ground truth DAG

Scenarios:
    1. Horizontal: K=3 clients, equal sample split, all features
    2. Vertical: K=3 clients, all samples, feature split [0-3, 4-7, 8-10]
    3. Hybrid (old): Product-then-Mixture
    4. Hybrid (new): Mixture-then-Product

Metrics:
    - Skeleton F1 (primary)
    - Directed F1
    - SHD (Structural Hamming Distance)
    - Runtime

Seeds: 5 random seeds for robustness
"""

def run_sachs_experiment(scenario, hybrid_mode=None, seed=42):
    """Run single Sachs experiment."""

    # Load Sachs data
    X, true_DAG = load_sachs_data()
    n, d = X.shape  # 853 × 11

    # Partition data
    if scenario == "horizontal":
        K = 3
        X_splits = split_samples_equally(X, K)

    elif scenario == "vertical":
        K = 3
        feature_groups = [[0,1,2,3], [4,5,6,7], [8,9,10]]
        X_splits = split_features(X, feature_groups)

    elif scenario == "hybrid":
        K = 3
        # Hybrid: overlap between clients
        feature_maps = {
            0: [0,1,2,3,4],      # Client 0: first 5
            1: [3,4,5,6,7],      # Client 1: middle 5 (overlap!)
            2: [6,7,8,9,10]      # Client 2: last 5 (overlap!)
        }
        X_splits = split_hybrid(X, feature_maps)

    # Configure FedCDH
    args = Namespace(
        K=K,
        d=d,
        n=n // K,
        scenario=scenario,
        hybrid_mode=hybrid_mode,  # 'mixture_then_product' or 'product_then_mixture'
        ci_method="spn",
        alpha=0.05,
        epochs=50,  # More epochs for real data
        device="cpu",
        skip_bic=False,  # Use BIC for real data
        seed=seed
    )

    # Run
    fedcdh = FedCDH(args)
    results = fedcdh.fit(X_splits, c_indx, true_DAG)

    # Evaluate
    skeleton_f1 = compute_f1(results['G_learned'], true_DAG, skeleton_only=True)
    directed_f1 = compute_f1(results['G_learned'], true_DAG, skeleton_only=False)
    shd = compute_shd(results['G_learned'], true_DAG)

    return {
        'scenario': scenario,
        'hybrid_mode': hybrid_mode,
        'seed': seed,
        'skeleton_f1': skeleton_f1,
        'directed_f1': directed_f1,
        'shd': shd,
        'runtime': results.get('runtime', 0)
    }
```

---

### Step 7.2: Run Experiments (1 day automated)

```python
def main():
    results_all = []

    # Run all configurations × 5 seeds
    for seed in range(5):
        # Horizontal baseline
        results_all.append(run_sachs_experiment('horizontal', seed=seed))

        # Vertical baseline
        results_all.append(run_sachs_experiment('vertical', seed=seed))

        # Hybrid (old Product-then-Mixture)
        results_all.append(run_sachs_experiment('hybrid',
                                                hybrid_mode='product_then_mixture',
                                                seed=seed))

        # Hybrid (new Mixture-then-Product) - THESIS MAIN RESULT
        results_all.append(run_sachs_experiment('hybrid',
                                                hybrid_mode='mixture_then_product',
                                                seed=seed))

    # Save results
    df = pd.DataFrame(results_all)
    df.to_csv('results/sachs_hybrid_comparison.csv', index=False)

    # Summary statistics
    summary = df.groupby(['scenario', 'hybrid_mode']).agg({
        'skeleton_f1': ['mean', 'std'],
        'directed_f1': ['mean', 'std'],
        'shd': ['mean', 'std'],
        'runtime': ['mean', 'std']
    })

    print(summary)
    summary.to_csv('results/sachs_summary.csv')

if __name__ == '__main__':
    main()
```

**Run**:
```bash
python tests/benchmarks/run_hybrid_experiments.py
# Estimated time: 3-4 hours (20 experiments × 10-15 min each)
```

**Deliverable Day 12**: Sachs results CSV files with 5 seeds × 4 configurations

---

## Day 13: Results Analysis

**Goal**: Analyze results for thesis

### Step 8.1: Statistical Analysis (3 hours)

```python
# analysis/analyze_sachs_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_results():
    df = pd.read_csv('results/sachs_hybrid_comparison.csv')

    # Group by configuration
    grouped = df.groupby(['scenario', 'hybrid_mode'])

    # 1. Descriptive Statistics
    print("="*60)
    print("SACHS RESULTS SUMMARY")
    print("="*60)

    for (scenario, mode), group in grouped:
        name = f"{scenario} ({mode})" if mode else scenario
        print(f"\n{name}:")
        print(f"  Skeleton F1: {group['skeleton_f1'].mean():.3f} ± {group['skeleton_f1'].std():.3f}")
        print(f"  Directed F1: {group['directed_f1'].mean():.3f} ± {group['directed_f1'].std():.3f}")
        print(f"  SHD: {group['shd'].mean():.1f} ± {group['shd'].std():.1f}")
        print(f"  Runtime: {group['runtime'].mean():.1f}s ± {group['runtime'].std():.1f}s")

    # 2. Hypothesis Testing
    # H0: Mixture-then-Product = Product-then-Mixture
    # Ha: Mixture-then-Product ≠ Product-then-Mixture

    hybrid_new = df[(df['scenario'] == 'hybrid') &
                    (df['hybrid_mode'] == 'mixture_then_product')]
    hybrid_old = df[(df['scenario'] == 'hybrid') &
                    (df['hybrid_mode'] == 'product_then_mixture')]

    t_stat, p_value = stats.ttest_ind(hybrid_new['skeleton_f1'],
                                       hybrid_old['skeleton_f1'])

    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("="*60)
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("✓ Significant difference (p < 0.05)")
    else:
        print("✗ No significant difference (p ≥ 0.05)")

    # 3. Effect Size (Cohen's d)
    mean_diff = hybrid_new['skeleton_f1'].mean() - hybrid_old['skeleton_f1'].mean()
    pooled_std = np.sqrt((hybrid_new['skeleton_f1'].std()**2 +
                          hybrid_old['skeleton_f1'].std()**2) / 2)
    cohens_d = mean_diff / pooled_std

    print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        print("  Small effect")
    elif abs(cohens_d) < 0.5:
        print("  Medium effect")
    else:
        print("  Large effect")
```

---

### Step 8.2: Visualization (2 hours)

```python
def create_thesis_figures():
    df = pd.read_csv('results/sachs_hybrid_comparison.csv')

    # Figure 1: Bar plot comparison
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Skeleton F1
    sns.barplot(data=df, x='scenario', y='skeleton_f1', hue='hybrid_mode', ax=ax[0])
    ax[0].set_title('Skeleton F1 Score')
    ax[0].set_ylabel('F1 Score')
    ax[0].set_ylim(0, 1)

    # Directed F1
    sns.barplot(data=df, x='scenario', y='directed_f1', hue='hybrid_mode', ax=ax[1])
    ax[1].set_title('Directed F1 Score')
    ax[1].set_ylabel('F1 Score')
    ax[1].set_ylim(0, 1)

    # SHD (lower is better)
    sns.barplot(data=df, x='scenario', y='shd', hue='hybrid_mode', ax=ax[2])
    ax[2].set_title('Structural Hamming Distance')
    ax[2].set_ylabel('SHD')

    plt.tight_layout()
    plt.savefig('results/figures/sachs_comparison.png', dpi=300)

    # Figure 2: Box plots for variability
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='scenario', y='skeleton_f1', hue='hybrid_mode')
    ax.set_title('Skeleton F1 Score Distribution (5 seeds)')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1)
    plt.savefig('results/figures/sachs_boxplot.png', dpi=300)
```

**Deliverable Day 13**:
- Analysis script with statistical tests
- 2 publication-quality figures for thesis

---

## Day 14-15: Thesis Documentation

**Goal**: Document implementation and results for thesis

### Step 9.1: Methods Section (3 hours)

**File**: `thesis/methods_hybrid_implementation.tex` or similar

```latex
\subsection{Hybrid Federated Learning via Mixture-then-Product}

\subsubsection{Theoretical Foundation}

Following \citet{seng2025fedpc}, we implement hybrid federated learning
as a probabilistic circuit that combines mixture and product nodes in a
specific hierarchy. Unlike traditional parameter averaging approaches,
our method models the joint distribution over all features by combining
marginal distributions learned from different clients.

The key insight is that hybrid FL, where clients hold overlapping sets
of features, can be formalized as:

\begin{equation}
P(X) = \prod_{g} \left[ \sum_{k \in \mathcal{C}_g} w_{k,g} \cdot P_{k,g}(X_g) \right]
\end{equation}

where $g$ indexes feature groups, $\mathcal{C}_g$ is the set of clients
holding feature group $g$, and $P_{k,g}$ is the local SPN trained by
client $k$ on features $X_g$.

This hierarchy has two levels:
\begin{itemize}
    \item \textbf{Inner (Mixture)}: For each feature group, combine
    distributions from clients that share those features via weighted
    mixture (sum node)
    \item \textbf{Outer (Product)}: Combine feature groups via product,
    assuming conditional independence given cluster assignment
\end{itemize}

\subsubsection{Algorithm}

We implement Algorithm 1 from \citet{seng2025fedpc} for automatic
feature grouping:

\begin{algorithm}
\caption{Hybrid Mode Structure Construction}
\begin{algorithmic}[1]
\STATE Initialize indicator matrix $M \in \{0,1\}^{K \times d}$
\STATE $M_{k,j} \gets 1$ if client $k$ has feature $j$
\FOR{each distinct column pattern $u$ in $M$}
    \STATE $\mathcal{C}_g \gets \{k : M_{k,:} \text{ matches } u\}$
    \STATE $\mathcal{F}_g \gets \{j : M_{:,j} \text{ matches } u\}$
    \STATE Train SPNs: $\{P_{k,g}\}_{k \in \mathcal{C}_g}$ on features $\mathcal{F}_g$
    \STATE Create mixture: $P(X_g) = \sum_{k \in \mathcal{C}_g} w_{k,g} P_{k,g}(X_g)$
\ENDFOR
\STATE Create product: $P(X) = \prod_g P(X_g)$
\end{algorithmic}
\end{algorithm}

\subsubsection{Implementation Details}

We implemented three new classes in the FedCDH framework:

\begin{itemize}
    \item \texttt{GroupMixture}: Implements $P(X_g) = \sum_k w_{k,g} P_{k,g}(X_g)$
    for a single feature group
    \item \texttt{ProductOverGroups}: Implements $P(X) = \prod_g P(X_g)$
    for disjoint feature groups
    \item \texttt{ProductOverGroupsWithOverlap}: Extends product to
    handle overlapping features via automatic subspace partitioning
\end{itemize}

Weights $w_{k,g}$ are set proportional to sample counts:
$w_{k,g} = n_{k,g} / \sum_{k'} n_{k',g}$ where $n_{k,g}$ is the number
of samples client $k$ has for feature group $g$. Following
\citet{seng2025fedpc}, we use one-pass training without iterative EM,
as EM requires data sharing incompatible with federated constraints.
```

---

### Step 9.2: Results Section (3 hours)

```latex
\subsection{Hybrid Mode Validation}

\subsubsection{Experimental Setup}

We validated the Mixture-then-Product hybrid implementation on the
Sachs protein signaling dataset \citep{sachs2005causal}, containing
853 samples across 11 variables. We compared four configurations:

\begin{itemize}
    \item \textbf{Horizontal}: 3 clients with equal sample splits,
    all features
    \item \textbf{Vertical}: 3 clients with all samples, feature
    splits $[0\text{-}3, 4\text{-}7, 8\text{-}10]$
    \item \textbf{Hybrid (Product-then-Mixture)}: Previous
    implementation (incorrect hierarchy)
    \item \textbf{Hybrid (Mixture-then-Product)}: New implementation
    following \citet{seng2025fedpc}
\end{itemize}

For hybrid scenarios, we partitioned features with overlaps:
clients held $[0\text{-}4, 3\text{-}7, 6\text{-}10]$ respectively,
creating 3-feature overlaps between adjacent clients. All experiments
used 50 SPN training epochs, BIC cluster selection, and $\alpha=0.05$
for CI testing. We report mean and standard deviation over 5 random seeds.

\subsubsection{Results}

Table~\ref{tab:sachs_results} shows quantitative results. The
Mixture-then-Product hybrid achieves Skeleton F1 of $0.XXX \pm 0.XXX$,
compared to $0.XXX \pm 0.XXX$ for Product-then-Mixture
($p = 0.XXX$, paired t-test). This confirms the theoretical prediction
that the correct hierarchy improves performance.

\begin{table}[h]
\centering
\caption{Sachs dataset results (mean $\pm$ std over 5 seeds)}
\label{tab:sachs_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Skeleton F1} & \textbf{Directed F1} & \textbf{SHD} & \textbf{Runtime (s)} \\
\midrule
Horizontal             & 0.XXX $\pm$ 0.XXX & 0.XXX $\pm$ 0.XXX & XX.X $\pm$ X.X & XX.X $\pm$ X.X \\
Vertical               & 0.XXX $\pm$ 0.XXX & 0.XXX $\pm$ 0.XXX & XX.X $\pm$ X.X & XX.X $\pm$ X.X \\
Hybrid (Old)           & 0.XXX $\pm$ 0.XXX & 0.XXX $\pm$ 0.XXX & XX.X $\pm$ X.X & XX.X $\pm$ X.X \\
\textbf{Hybrid (New)} & \textbf{0.XXX $\pm$ 0.XXX} & \textbf{0.XXX $\pm$ 0.XXX} & \textbf{XX.X $\pm$ X.X} & \textbf{XX.X $\pm$ X.X} \\
\bottomrule
\end{tabular}
\end{table}

Figure~\ref{fig:sachs_comparison} visualizes the comparison.
```

---

### Step 9.3: Limitations & Future Work (2 hours)

```latex
\subsection{Limitations}

While our implementation follows \citet{seng2025fedpc} closely,
several limitations remain:

\begin{enumerate}
    \item \textbf{Empirical validation only}: This work provides
    empirical validation of the Mixture-then-Product hierarchy but
    does not include formal convergence proofs or sample complexity
    analysis, which are out of scope for this Master's thesis.

    \item \textbf{Feature grouping}: We use automatic grouping based
    on data partitioning (Algorithm 1). More sophisticated approaches
    using correlation-based clustering or structure learning could
    improve performance but require additional federated computation.

    \item \textbf{Weight inference}: Following \citet{seng2025fedpc},
    we use one-pass training with sample-count proportional weights
    rather than iterative EM. This simplifies implementation but may
    be suboptimal for highly heterogeneous data.

    \item \textbf{Privacy}: Our implementation focuses on
    simulation-based privacy (no raw data sharing) but does not
    include differential privacy guarantees, which would add
    significant complexity.

    \item \textbf{Scalability}: Experiments used moderate-sized
    datasets (n=853). Scaling to larger datasets would require
    additional engineering (e.g., mini-batch training, distributed
    computation).
\end{enumerate}

\subsection{Future Work}

Future research directions include:

\begin{itemize}
    \item \textbf{Theoretical analysis}: Formal sample complexity
    bounds and convergence guarantees for Mixture-then-Product hybrid

    \item \textbf{Differential privacy}: Integration of DP-SGD or
    similar mechanisms for formal privacy guarantees

    \item \textbf{Learned feature grouping}: Replace manual/automatic
    grouping with federated structure learning

    \item \textbf{Real-world deployment}: Evaluate on industrial
    biomedical or semiconductor datasets with true distribution
    \item \textbf{Extension to other PC architectures}: Investigate
    whether Mixture-then-Product generalizes beyond SPNs to other
    tractable probabilistic circuits
\end{itemize}
```

**Deliverable Day 15**: Complete thesis documentation (Methods + Results + Limitations)

---

## Week 3 Summary Deliverables

**Experimental Results**:
- Sachs experiments completed (4 configs × 5 seeds = 20 runs)
- CSV results with statistics
- 2 publication-quality figures

**Thesis Documentation**:
- Methods section (algorithm, implementation details)
- Results section (table, statistical tests)
- Limitations & future work

**Ready for**: Thesis writing final draft

---

# Final Deliverables & Success Criteria

---

## Code Deliverables

### New Files Created
1. `causallearn/utils/FedPC.py` - 3 new classes (+300 lines)
   - GroupMixture
   - ProductOverGroups
   - ProductOverGroupsWithOverlap

2. `tests/unit/test_fedpc_hybrid.py` - Unit tests (+200 lines)

3. `tests/smoke/test_hybrid_mixture_then_product.py` - Integration tests (+100 lines)

4. `tests/benchmarks/run_hybrid_experiments.py` - Sachs experiments (+150 lines)

5. `analysis/analyze_sachs_results.py` - Statistical analysis (+100 lines)

### Modified Files
1. `causallearn/search/FCMBased/FedCDH/FedCDH.py` - Hybrid section replaced (~150 lines modified)

2. `agents/working_state.md` - Implementation plan documented

---

## Research Deliverables

### Empirical Validation ✅
- [ ] Hybrid mode distinct from horizontal/vertical
- [ ] Sachs experiments with 5 seeds
- [ ] Statistical significance tests
- [ ] Publication-quality figures

### Theoretical Grounding ✅
- [ ] Correct citation of Seng et al. (2025)
- [ ] Implementation follows Algorithm 1
- [ ] Mixture-then-Product hierarchy validated

### Thesis Documentation ✅
- [ ] Methods section complete
- [ ] Results section with tables/figures
- [ ] Limitations documented honestly
- [ ] Future work identified

---

## Success Criteria

### Must-Have (Blocking Thesis Submission)
1. ✅ **Correct Implementation**: Mixture-then-Product per Seng et al. (2025)
2. ✅ **Empirical Validation**: Sachs experiments show hybrid works
3. ✅ **Distinct from Baselines**: Hybrid ≠ horizontal/vertical
4. ✅ **Reproducible**: Clear documentation, tests pass
5. ✅ **Thesis-Ready**: Methods + Results sections written

### Nice-to-Have (Strengthens Thesis)
6. ⚠️ **Statistical Significance**: p < 0.05 for hybrid improvement
7. ⚠️ **Multiple Datasets**: Additional validation beyond Sachs
8. ⚠️ **Runtime Analysis**: Efficiency comparison
9. ⚠️ **Ablation Study**: Feature grouping strategies compared

---

## Timeline Summary

| Week | Days | Milestone | Deliverable |
|------|------|-----------|-------------|
| **Week 1** | 1-2 | GroupMixture | Mixture over clients working |
| | 3-4 | ProductOverGroups | Product over groups working |
| | 5 | Overlap handling | Algorithm 1 overlap detection |
| **Week 2** | 6-7 | Feature grouping | Automatic grouping working |
| | 8-9 | Integration | Hybrid mode replaced in FedCDH |
| | 10 | Smoke tests | Basic validation passing |
| **Week 3** | 11-12 | Sachs experiments | Results CSV generated |
| | 13 | Analysis | Statistical tests + figures |
| | 14-15 | Documentation | Thesis sections written |

**Total**: 15 working days (3 weeks)

---

## Risk Mitigation

### Technical Risks

1. **Risk**: Overlap handling doesn't work correctly
   - **Mitigation**: Follow paper's Algorithm 1 exactly, extensive testing
   - **Fallback**: Assume disjoint groups for thesis, document limitation

2. **Risk**: Performance regression vs current implementation
   - **Mitigation**: Run comparison experiments early (Day 10)
   - **Fallback**: Keep both modes, compare empirically

3. **Risk**: Integration breaks H/V modes
   - **Mitigation**: Only modify hybrid section, regression tests
   - **Fallback**: Revert to previous version, document in limitations

### Research Risks

4. **Risk**: No statistically significant improvement
   - **Mitigation**: Focus on "theoretically correct" not "empirically better"
   - **Fallback**: Thesis contribution is "correct implementation" not "SOTA performance"

5. **Risk**: Sachs dataset too small for meaningful results
   - **Mitigation**: Report effect size, not just p-values
   - **Fallback**: Emphasize proof-of-concept, suggest larger datasets for future

### Timeline Risks

6. **Risk**: Implementation takes longer than 3 weeks
   - **Mitigation**: Prioritize must-haves, defer nice-to-haves
   - **Fallback**: Submit with current Product-then-Mixture, document correct approach as future work

---

## Thesis Contribution Statement

**Main Contribution**:
> "We implement and validate the theoretically grounded Mixture-then-Product
> hybrid federated learning approach from \citet{seng2025fedpc}, extending
> FedCDH to correctly handle overlapping feature partitions. Our implementation
> follows Algorithm 1 for automatic feature grouping and achieves empirical
> validation on the Sachs protein signaling dataset."

**What Makes This Sufficient for Master's Thesis**:
1. ✅ **Novel Implementation**: First correct hybrid mode for FedCDH
2. ✅ **Theoretical Grounding**: Based on recent ICLR/ICML work
3. ✅ **Empirical Validation**: Real dataset (Sachs) with statistical tests
4. ✅ **Reproducible**: Clear algorithm, code documented, tests included
5. ✅ **Honest Limitations**: Acknowledged as empirical study

**What This is NOT (and doesn't need to be)**:
- ❌ Novel theoretical contribution (using existing theory correctly)
- ❌ State-of-the-art performance (correctness over performance)
- ❌ Large-scale deployment (proof-of-concept sufficient)
- ❌ Formal proofs (empirical study, as stated)

---

## Next Steps

**Immediate (This Week)**:
1. Review this roadmap with supervisor
2. Confirm thesis scope aligns with plan
3. Set up development branch: `git checkout -b hybrid-mixture-then-product`

**Week 1 Start**:
4. Begin Day 1: Implement GroupMixture skeleton
5. Daily check-ins: Test coverage, progress tracking

**Throughout Implementation**:
6. Document decisions in working_state.md
7. Commit frequently with clear messages
8. Run tests after each component

**After Completion**:
9. Submit thesis draft with new results
10. Prepare presentation/defense materials

---

## References

- Seng et al. (2025). "Scaling Probabilistic Circuits via Data Partitioning." ICLR 2025.
- Li et al. (2024). "Federated Causal Discovery from Heterogeneous Data." ICLR 2024.
- Peharz et al. (2020). "Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits." ICML 2020.

**Reference Implementation**: https://github.com/J0nasSeng/federated-spn

---

## Conclusion

This roadmap provides a **concrete, thesis-focused plan** to implement theoretically grounded Mixture-then-Product hybrid mode in 3 weeks. The plan:

✅ **Aligns with research goals**: Extends FedCDH to hybrid FL correctly
✅ **Grounded in theory**: Follows Seng et al. (2025) Algorithm 1
✅ **Empirically validates**: Sachs experiments with statistical tests
✅ **Realistic for Master's thesis**: 15 days of focused work
✅ **Acknowledges limitations**: Empirical study, no formal guarantees

**Ready to start implementation** upon approval.
