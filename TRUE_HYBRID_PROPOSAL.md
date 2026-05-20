# True Hybrid Mode: Feature Maps with Overlap

## Problem

Current "hybrid" mode is NOT true hybrid:
- Only partitions samples (like horizontal)
- All clients have ALL features
- Just uses different SPN aggregation (ProductOverGroups)

**Real hybrid** should support:
- Sample partitioning (different patients per hospital)
- Feature overlap (some features shared, some unique)

## Real-World Example

**Hospital Network:**

```
Hospital A (Urban):
  - Patients: 0-300
  - Features: Demographics, Vitals, Lab_tests
  - Total: 10 features

Hospital B (Rural):
  - Patients: 300-600
  - Features: Demographics, Vitals, Imaging
  - Total: 8 features (some overlap with A)

Hospital C (Specialty):
  - Patients: 600-900
  - Features: Demographics, Lab_tests, Imaging, Genetic
  - Total: 12 features (overlaps both A and B)
```

**Overlap structure:**
- Demographics: Available at ALL hospitals
- Vitals: Available at A, B
- Lab_tests: Available at A, C
- Imaging: Available at B, C
- Genetic: Available at C only

## Context Encoding for True Hybrid

### Current Encoding (Insufficient)

```python
# Only sample-level context
c_indx = [0,0,0,...,1,1,1,...,2,2,2,...]  # Which hospital owns each sample
```

### Proposed Encoding

```python
# 1. Sample-level context (keep as is)
c_indx = [0,0,0,...,1,1,1,...,2,2,2,...]

# 2. Feature-level ownership (NEW - with overlap!)
feature_maps = {
    0: [0,1,2,3,4,5,6,7,8,9],        # Hospital A: 10 features
    1: [0,1,2,3,10,11,12,13],        # Hospital B: 8 features (overlap on 0-3)
    2: [0,1,4,5,10,11,14,15,16,17],  # Hospital C: 10 features (overlaps both)
}

# 3. Feature availability matrix (derived from feature_maps)
feature_availability = compute_availability_matrix(feature_maps, d_total=18)
# Shape: (K, d_total)
# [client, feature] = 1 if client has that feature
```

## Orientation Strategy for True Hybrid

### Edge Classification

For edge i--j, classify based on feature ownership:

```python
def classify_edge(i, j, feature_maps):
    """Classify edge based on feature availability."""
    clients_with_i = get_owners(i, feature_maps)
    clients_with_j = get_owners(j, feature_maps)

    if len(clients_with_i) == 1 and clients_with_i == clients_with_j:
        return "WITHIN_CLIENT"  # Both at same single client

    elif set(clients_with_i).isdisjoint(set(clients_with_j)):
        return "CROSS_CLIENT_DISJOINT"  # No overlap

    else:
        return "CROSS_CLIENT_OVERLAP"  # Some clients have both (TRUE HYBRID!)
```

### Orientation Logic

```python
def orient_edge_true_hybrid(i, j, feature_maps, c_indx, local_spns, global_spn):
    """
    Orientation for true hybrid mode with overlapping features.
    """
    edge_type = classify_edge(i, j, feature_maps)

    if edge_type == "WITHIN_CLIENT":
        # Both features at single client only
        client = get_owners(i, feature_maps)[0]

        # Use local SPN with that client's samples
        client_mask = (c_indx.flatten() == client)
        local_data = data_aug[client_mask]

        return orient_using_local_spn(i, j, local_spns[client], local_data)

    elif edge_type == "CROSS_CLIENT_DISJOINT":
        # Features at different clients with no overlap
        # Example: Hospital A's Vitals vs Hospital C's Genetic

        # Use global product SPN with ALL samples
        return orient_using_global_spn(i, j, global_spn, data_aug)

    elif edge_type == "CROSS_CLIENT_OVERLAP":
        # TRUE HYBRID: Some clients have both features
        # Example: Demographics (at A,B,C) vs Lab_tests (at A,C)

        # Strategy: Use samples from overlapping clients
        common_clients = set(get_owners(i, feature_maps)) & set(get_owners(j, feature_maps))
        overlap_mask = np.isin(c_indx.flatten(), list(common_clients))
        overlap_data = data_aug[overlap_mask]

        # Use global SPN but only on relevant samples
        return orient_using_global_spn(i, j, global_spn, overlap_data)
```

## Implementation Steps

### Step 1: Update Test to Create True Hybrid Data

```python
# In test_fedcdh_benchmark_v3.py

elif scenario == "hybrid":
    # TRUE HYBRID: Both sample AND feature partitioning with overlap

    # Sample partitioning
    samples_per_client = n // K
    X_splits_samples = [
        X[i * samples_per_client : (i + 1) * samples_per_client, :]
        for i in range(K)
    ]

    # Feature overlap: Create overlapping feature sets
    # Example: 50% overlap between consecutive clients
    feature_sets = create_overlapping_feature_sets(d, K, overlap_ratio=0.5)
    # Returns: {0: [0,1,2,3,4,5], 1: [3,4,5,6,7,8], 2: [6,7,8,9,10]}

    # Extract features for each client
    X_splits = [
        X_splits_samples[k][:, feature_sets[k]]
        for k in range(K)
    ]

    # Create feature maps (with overlap!)
    feature_maps = feature_sets

    # c_indx remains sample-based
    c_indx = np.repeat(np.arange(K), samples_per_client).reshape(-1, 1)
```

### Step 2: Update FedCDH to Handle Overlapping Features

```python
# In FedCDH.py

if self.scenario == "hybrid":
    # Build feature_maps from X_splits (detect overlap)
    feature_maps = {}
    for k in range(self.K_clients):
        # Track which global features this client has
        # (More complex than vertical - need overlap detection)
        feature_maps[k] = detect_features(X_splits[k], X_global)

    self.feature_maps = feature_maps
```

### Step 3: Update Orientation for True Hybrid

```python
# In mechanism_invariance.py

def orient_edge_true_hybrid_with_overlap(
    i, j,
    feature_maps,  # Now can have overlapping features
    c_indx,        # Sample-level context
    local_spns,
    global_spn,
    data_aug
):
    """Handle true hybrid with overlapping features."""

    clients_with_i = [k for k, feats in feature_maps.items() if i in feats]
    clients_with_j = [k for k, feats in feature_maps.items() if j in feats]

    # ... (as described above)
```

## Benefits of True Hybrid Support

1. **Real-world applicability:** Matches actual federated scenarios
   - Hospital networks with partial equipment overlap
   - Multi-site studies with varying measurement capabilities

2. **Better orientation:** Uses appropriate samples/SPNs based on overlap
   - Overlapping features: Use samples from relevant clients only
   - Disjoint features: Use global aggregation

3. **Principled approach:** Extends feature_maps concept consistently
   - Vertical: Disjoint features (special case)
   - Hybrid: Overlapping features (general case)

## Current vs Proposed

**Current "Hybrid":**
```
Sample-level: Partitioned
Feature-level: ALL features at ALL clients
Context: c_indx only
Aggregation: ProductOverGroups
```
→ Just horizontal with different aggregation

**Proposed True Hybrid:**
```
Sample-level: Partitioned
Feature-level: Overlapping subsets
Context: c_indx + feature_maps (with overlap)
Aggregation: ProductOverGroups
Orientation: Overlap-aware
```
→ True hybrid supporting real-world scenarios

## Open Questions

1. **How to create overlapping feature sets in benchmarks?**
   - Random overlap?
   - Structured overlap (common features + unique features)?
   - Domain-specific overlap patterns?

2. **Should we keep current "hybrid" as separate mode?**
   - Rename current to "hybrid_simple" (horizontal with product aggregation)
   - New "hybrid_true" (true overlapping features)

3. **How to handle orientation with partial overlap?**
   - Use only overlapping samples?
   - Ensemble over multiple local SPNs?
   - Weight by overlap degree?

## Next Steps

1. Analyze if current hybrid results are actually good or just lucky
2. Design true hybrid data partitioning scheme
3. Implement overlap detection and classification
4. Extend orientation logic for overlapping cases
5. Test on synthetic true hybrid scenarios
