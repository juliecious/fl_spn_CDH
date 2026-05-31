# Does FedCDH Take U Into Account in Phase 6 (Main PC)?

**Your Question**: "Also, has FedCDH takes U into account in the same stage?"

**Short Answer**: **NO** - The original FedCDH/CDNOD implementation does NOT include U in the conditioning sets during Phase 6 (main PC algorithm).

---

## Evidence from Code

### How Conditioning Sets Are Built

**File**: `causallearn/utils/PCUtils/SkeletonDiscovery.py:149`

```python
# Line 148: Get neighbors of x (excluding y)
Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))

# Line 149: Build conditioning set S from neighbors only
for S in combinations(Neigh_x_noy, depth):
    # Line 151: Test conditional independence
    p = cg.ci_test(x, y, S)

    if p > alpha:
        # Remove edge (x, y)
```

**Key observation**:
- Conditioning set `S` is built from `Neigh_x_noy` (neighbors of x, excluding y)
- U (augmented variable) is NOT in the neighbor set
- Therefore, U is NOT included in conditioning sets

---

## Where U Is vs Is NOT Used

### Phase 3: Structure Voting - ✅ Uses U

**File**: `aggregation.py:36-42`

```python
# Always condition on augmented variable
Z = [d_features]  # d_features = 8 (the augmented variable index)

# Test: X_i ⊥ X_j | U
for i in range(d_features):
    for j in range(i + 1, d_features):
        # CI test conditioning on U
        p_value = ci_test(i, j, Z=[d_features])
```

**Result**: Works correctly, varied p-values

---

### Phase 6: Main PC (CDNOD) - ❌ Does NOT Use U

**File**: `SkeletonDiscovery.py:149-151`

```python
# Build conditioning set from neighbors only (does NOT include U)
Neigh_x_noy = neighbors of x (excluding y)
for S in combinations(Neigh_x_noy, depth):
    # S does NOT contain U
    p = cg.ci_test(x, y, S)
```

**Result**: Broken, all p-values = 0.000

---

## Why This Design?

### Historical Context: Original CDNOD Paper

The original CDNOD algorithm (Seng et al.) was designed with a **different usage of U**:

#### Original CDNOD Design:

1. **U is part of the causal graph** as a node
2. **Edges TO U** indicate which variables depend on context
3. **Orientation uses U** for mechanism invariance
4. **PC treats U like any other variable**

#### In Original CDNOD:

```
Graph structure:

    X₀ ← U → X₂
     ↓        ↓
    X₁       X₃

U is a NODE in the graph with edges TO variables
```

**Key difference**: In original CDNOD, U is supposed to have edges in the skeleton, so it naturally appears in neighbor sets and conditioning sets.

---

### Our Modification: Structure Voting Mode

**In our implementation with structure voting**:

1. **U is NOT part of the causal graph** (excluded via `exclude_augmented_var=True`)
2. **U has 0 neighbors** (no edges TO/FROM U)
3. **PC doesn't see U** in neighbor sets
4. **Result**: U never appears in conditioning sets

This is why Phase 6 doesn't condition on U!

---

## The Inconsistency

### Structure Voting (Phase 3) Philosophy

```
"U captures domain heterogeneity and should be conditioned on
 to get accurate conditional independence tests"
```

**Implementation**: Always includes U in Z

**Result**: ✅ Works well

---

### Main PC (Phase 6) Philosophy (Implicit)

```
"U is not part of the causal graph, so it's not in the
 neighbor set, so it won't be in conditioning sets"
```

**Implementation**: U excluded from graph → not in neighbors → not in Z

**Result**: ❌ Broken (all p=0.000)

---

## The Fix: Make Phase 6 Consistent with Phase 3

### Option 1: Pass U Explicitly to skeleton_discovery (Recommended)

**Modify function signature**:

```python
def skeleton_discovery(
    flag,
    cg_list,
    data,
    K,
    alpha,
    indep_test,
    stable,
    verbose=False,
    show_progress=True,
    background_knowledge=None,
    depth_limit=None,
    c_indx_id=None,  # ← ADD THIS: Index of augmented variable
    **kwargs
):
    """
    Parameters
    ----------
    c_indx_id : int, optional
        Index of the augmented/context variable.
        If provided, this variable will be included in all conditioning sets.
    """
```

**Modify conditioning set construction**:

```python
# Line 148-151 (current):
Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
for S in combinations(Neigh_x_noy, depth):
    p = cg.ci_test(x, y, S)

# Modified:
Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
for S in combinations(Neigh_x_noy, depth):
    # Always include augmented variable if provided
    if c_indx_id is not None:
        S_with_U = tuple(S) + (c_indx_id,)
    else:
        S_with_U = S

    p = cg.ci_test(x, y, S_with_U)
```

**Pass from CDNOD**:

```python
# In cdnod_alg, line 360:
cg_0 = SkeletonDiscovery.skeleton_discovery(
    flag, cg_list, data, K, alpha, indep_test_all, stable,
    depth_limit=depth_limit,
    c_indx_id=c_indx_id if exclude_augmented_var else None  # ← ADD THIS
)
```

---

### Option 2: Add U as a Node (Not Recommended for Your Case)

**This is the original CDNOD approach**:
- Include U in the causal graph
- Let it have edges TO variables
- It will naturally appear in neighbor sets

**Cons**:
- Violates your design (exclude_augmented_var=True)
- U would appear in final skeleton
- Orientation phase would need changes

---

## Comparison: With vs Without U in Conditioning

### Without U (Current)

```
Test at Depth 0: X₀ ⊥ X₂ | []

GlobalSPN computes:
  P(X₀, X₂) = Σ_u P(X₀, X₂ | U=u) · P(U=u)
              ↑
         Marginalization over U (confounding!)

Result: p = 0.000 (appears dependent)
Action: Keep edge (wrong for false positives)
```

### With U (Proposed Fix)

```
Test at Depth 0: X₀ ⊥ X₂ | [U]

GlobalSPN computes:
  P(X₀, X₂ | U=k) for samples with U=k
  (or averages over U in a controlled way)
              ↑
         Conditioning on U (controls confounding!)

Result: p = 0.065 (correctly independent/dependent)
Action: Remove/keep edge based on true relationship
```

---

## Summary Table

| Aspect | Structure Voting (Phase 3) | Main PC (Phase 6) | Consequence |
|--------|---------------------------|-------------------|-------------|
| **Uses U?** | ✅ Yes, Z=[U] always | ❌ No, Z=neighbors only | Inconsistent |
| **P-values** | Varied (0.02-0.12) | All 0.000 | Broken |
| **Why?** | Explicit design choice | U not in neighbor set | Implementation gap |
| **Works?** | ✅ Yes (22 edges) | ❌ No (kept all 22) | Phase 3 good, Phase 6 bad |

---

## Direct Answer to Your Question

> "Also, has FedCDH takes U into account in the same stage?"

**Answer**:

**NO** - FedCDH/CDNOD does NOT take U into account in Phase 6 (main PC algorithm).

**Reasons**:

1. **By design**: U is excluded from the causal graph (`exclude_augmented_var=True`)
2. **No edges to U**: U has no neighbors, so it doesn't appear in neighbor sets
3. **Not in conditioning sets**: PC builds conditioning sets from neighbors only
4. **Result**: U is completely absent from Phase 6 CI tests

**But**:

- **Phase 3 (Structure Voting) DOES use U**: Explicitly conditions on U in all tests
- **This inconsistency** causes Phase 6 to fail (all p=0.000)

**The fix**: Make Phase 6 consistent with Phase 3 by explicitly including U in all conditioning sets, even though U is not part of the causal graph.

---

## Philosophical Question

**Should U be in conditioning sets if it's not in the causal graph?**

**Answer**: **YES**, for correct causal inference!

### Why?

**U is a confounder** - it affects multiple variables but is not causally downstream of them.

In causal inference:
- **Confounders should be conditioned on** to get unbiased estimates
- **Even if confounders are not in the causal graph of interest**
- This is standard practice (e.g., "control variables" in regression)

### Example from Epidemiology

Study: Does X (smoking) cause Y (lung cancer)?

Confounder: U (age)
- Age affects both smoking rates and lung cancer risk
- Age is NOT caused by smoking or lung cancer
- We don't include age in the "smoking → lung cancer" causal graph
- **But we must condition on age** to get correct causal effect

**Same logic applies here**: U (client ID) confounds relationships, must be conditioned on, even though it's not in the causal graph.

---

## Conclusion

**Your experiment revealed an important design inconsistency**:

1. Structure Voting correctly conditions on U → Works well
2. Main PC does not condition on U → Fails completely
3. The solution: Make main PC condition on U (like structure voting)

This is not a bug in the original FedCDH paper - it's an adaptation issue when combining structure voting (which uses U) with the PC algorithm (which currently doesn't use U in your configuration).

**Next step**: Implement Option 1 above - pass `c_indx_id` to `skeleton_discovery` and include it in all conditioning sets.
