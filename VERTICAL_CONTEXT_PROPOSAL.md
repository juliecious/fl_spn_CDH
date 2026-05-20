# Vertical Mode Context Encoding Proposal
## Inspired by Jonas Seng's Vertical Design

### Problem
Current approach: `c_indx = [0,0,0,...]` (all zeros) provides no information for orientation.

### Real-World Scenario
Healthcare vertical federation:
- Hospital: Age, Gender, Blood_Pressure (features 0-2, client 0)
- Lab: Glucose, Cholesterol (features 3-4, client 1)
- Pharmacy: DrugA, DrugB (features 5-6, client 2)

All clients see the SAME 1000 patients, but different features.

### Key Insight from Seng
In Seng's ProductOverGroups, feature ownership is encoded in the **model structure**:
```
P(all features) = P(Hospital features) × P(Lab features) × P(Pharmacy features)
```

No explicit "context variable" needed - the product structure IS the ownership encoding!

### Proposed Solution

#### Option A: Feature-Level Context Matrix (Explicit Encoding)

```python
# Create (n, d) context matrix indicating feature ownership
c_indx_vertical = np.zeros((n, d), dtype=int)

for client_id, feature_indices in feature_maps.items():
    c_indx_vertical[:, feature_indices] = client_id

# Example: c_indx_vertical[:, 0] = 0 (feature 0 belongs to client 0)
#          c_indx_vertical[:, 3] = 1 (feature 3 belongs to client 1)
```

Augmented data: `X_aug = [X, c_indx_vertical]` with shape (n, 2d)

**Pros:**
- Explicit encoding in data
- cdnod can condition on feature ownership
- Similar to horizontal's approach

**Cons:**
- Doubles dimensionality
- Still doesn't capture that ALL clients see ALL samples

#### Option B: Ownership-Aware Orientation (Structural Encoding)

Keep `feature_maps` as metadata, modify orientation logic:

```python
def orient_edge_vertical_aware(i, j, fed_spn_model, feature_maps, local_spns):
    """
    Orient edge i--j using ownership information.

    Within-client edges: Use local SPN (more accurate, trained on those features)
    Cross-client edges: Use global product SPN (captures dependencies)
    """
    client_i = get_owner(i, feature_maps)
    client_j = get_owner(j, feature_maps)

    if client_i == client_j:
        # Within-client: X and Y both from same client
        # Use LOCAL SPN for this client
        local_spn = local_spns[client_i]
        return compare_conditionals_local(i, j, local_spn)
    else:
        # Cross-client: X from client_i, Y from client_j
        # Use GLOBAL product SPN (has learned cross-client dependencies)
        return compare_conditionals_global(i, j, fed_spn_model)
```

**Pros:**
- ✅ Leverages Seng's ProductOverGroups structure
- ✅ Uses local SPNs where appropriate (within-client)
- ✅ Uses global SPN for cross-client dependencies
- ✅ No artificial context variable
- ✅ Doesn't double dimensionality

**Cons:**
- Requires modification to orientation logic
- Can't reuse horizontal's UCSepset+Meek

#### Option C: Hybrid Approach

Use lightweight feature ownership encoding + ownership-aware orientation:

```python
# Create K binary columns (one per client) indicating which clients are "active"
# For vertical: all clients are active for all samples
c_indx_vertical = np.ones((n, K), dtype=int)  # Shape: (n, K)

# BUT also pass feature_maps to enable ownership-aware queries
# When testing X ⊥ Y | Z, check:
# - Which clients own X, Y, Z?
# - Use appropriate conditional test based on ownership
```

### Recommendation: Option B (Ownership-Aware)

This is closest to Seng's design philosophy:
1. Recognize that vertical mode has fundamentally different structure
2. Use ProductOverGroups architecture (already implemented)
3. Leverage local SPNs for within-client edges (more accurate)
4. Use global product SPN for cross-client edges (captures dependencies)
5. Don't force sample-level context where it doesn't belong

### Implementation Sketch

```python
# In mechanism_invariance.py

def orient_edge_vertical_with_ownership(
    i: int,
    j: int,
    fed_spn_model,  # Global ProductOverGroups
    local_spns: List,  # List of local SPNs per client
    feature_maps: dict,  # Feature ownership
) -> int:
    """Orient edge using ownership-aware strategy."""

    # Determine ownership
    client_i = None
    client_j = None
    for cid, features in feature_maps.items():
        if i in features:
            client_i = cid
        if j in features:
            client_j = cid

    if client_i == client_j:
        # Within-client edge: use local SPN
        # This is more accurate since local SPN was trained on exactly these features
        local_spn = local_spns[client_i]
        score_i_to_j = compute_conditional_local(j, [i], local_spn)
        score_j_to_i = compute_conditional_local(i, [j], local_spn)
    else:
        # Cross-client edge: use global product SPN
        # Global SPN learned cross-client dependencies via product structure
        score_i_to_j = compute_conditional_global(j, [i], fed_spn_model)
        score_j_to_i = compute_conditional_global(i, [j], fed_spn_model)

    return 1 if score_i_to_j > score_j_to_i else -1
```

### Why This Works

1. **Within-client edges** (e.g., Age → Blood_Pressure):
   - Both features from Hospital
   - Hospital's local SPN learned P(Age, Gender, BP)
   - Can accurately compare P(BP|Age) vs P(Age|BP)

2. **Cross-client edges** (e.g., Age → Glucose):
   - Age from Hospital, Glucose from Lab
   - Global SPN learned P(Age, Gender, BP) × P(Glucose, Chol)
   - Can evaluate P(Glucose | Age) using product structure

3. **No artificial context needed:**
   - Ownership is in the model structure (ProductOverGroups)
   - Orientation logic is ownership-aware
   - More principled than forcing (n,1) context

### Next Steps

1. Implement `orient_edge_vertical_with_ownership()`
2. Modify `orient_skeleton_mechanism_invariance()` to call it for vertical mode
3. Pass `local_spns` to orientation function (currently only pass `fed_spn_model`)
4. Test on law_school and sachs
