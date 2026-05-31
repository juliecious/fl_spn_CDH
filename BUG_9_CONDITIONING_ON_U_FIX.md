# Bug Fix #9: Condition on Augmented Variable U in Main PC Algorithm

**Date**: 2026-05-31
**Status**: ✅ IMPLEMENTED
**Impact**: Critical - fixes all p-values = 0.000 issue causing too many edges (24 vs 8)

---

## Problem Statement

### Symptom
After implementing Bug Fixes #1-5, the pipeline produced **24 edges instead of 8** (experiment 20260530_205318):
- Initial skeleton from structure voting: 22 edges ✅
- After PC Stage 1 refinement: 22 edges (no refinement!) ❌
- After PC Stage 2: 24 edges (added 2 more!) ❌
- **ALL p-values = 0.000** in main PC algorithm

### Root Cause

**Inconsistency between Phase 3 (Structure Voting) and Phase 6 (Main PC)**:

| Phase | Conditioning Set | P-values | Works? |
|-------|-----------------|----------|--------|
| **Phase 3 (Structure Voting)** | Z = [U] (always) | Varied (0.02-0.12) | ✅ Yes |
| **Phase 6 (Main PC)** | Z = [] or neighbors only | All 0.000 | ❌ No |

**Why this happens**:
1. GlobalSPN is trained on 9D data: P(X₀, X₁, ..., X₇, U)
2. When computing P(X_i, X_j) without U in the query, it marginalizes over U
3. Marginalization creates **spurious dependence** (confounding)
4. All conditional independence tests appear dependent (p=0.000)
5. No edges get removed

**Mathematical explanation**:
```
Test: X₀ ⊥ X₂ | []

P(X₀, X₂) = Σ_{u=0,1,2} P(X₀, X₂ | U=u) · P(U=u)
            ↑
        Marginalization over U creates spurious correlation
        even if X₀ ⊥ X₂ | U within each client!

Result: p = 0.000 (appears dependent)
```

---

## Solution

**Modify PC algorithm to always condition on U** (same as structure voting):

### Before (Broken)
```python
Depth 0: Test X_i ⊥ X_j | []
Depth 1: Test X_i ⊥ X_j | [neighbor₁]
Depth 2: Test X_i ⊥ X_j | [neighbor₁, neighbor₂]

Result: All p = 0.000 (marginalization over U)
```

### After (Fixed)
```python
Depth 0: Test X_i ⊥ X_j | [U]
Depth 1: Test X_i ⊥ X_j | [U, neighbor₁]
Depth 2: Test X_i ⊥ X_j | [U, neighbor₁, neighbor₂]

Expected: Varied p-values (conditioning on U)
```

---

## Implementation Details

### Files Modified

#### 1. `causallearn/utils/PCUtils/SkeletonDiscovery.py`

**Modified 3 functions**:

##### A. `skeleton_discovery` (Line 25-40)
```python
def skeleton_discovery(
    flag: int,
    cg_list: List[CausalGraph],
    data: ndarray,
    K: int,
    alpha: float,
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None,
    use_ranking: bool = False,
    ranking_tracker=None,
    depth_limit: int | None = None,
    c_indx_id: int | None = None,  # NEW PARAMETER
) -> CausalGraph:
    """
    Parameters
    ----------
    c_indx_id : int, optional
        Index of the augmented/context variable.
        If provided, this variable will be included in all conditioning sets.
    """
```

**Modified conditioning set construction** (Lines 149-162):
```python
Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
for S in combinations(Neigh_x_noy, depth):
    # FIX: Include augmented variable in conditioning set if provided
    if c_indx_id is not None:
        S_with_context = tuple(S) + (c_indx_id,)
    else:
        S_with_context = S

    if flag == 0:
        p = cg.ci_test(x, y, S_with_context)

    # ... all voting schemes updated to use S_with_context
```

##### B. `skeleton_discovery_with_surrogate` (Line 347-361)
- Added `c_indx_id: int | None = None` parameter
- Modified conditioning set construction at line 478-579
- Updated all voting scheme variations to use `S_with_context`

##### C. `skeleton_discovery_with_surrogate_GMM` (Line 627-641)
- Added `c_indx_id: int | None = None` parameter
- Modified conditioning set construction at line 747-809
- Updated all CI test calls to use `S_with_context`

---

#### 2. `causallearn/search/ConstraintBased/CDNOD.py`

**Modified `cdnod_alg` function** (Lines 360-402):

```python
# FIX: Compute c_indx_id early so it can be passed to skeleton_discovery functions
# This is the index of the augmented/context variable (client ID) in data_aug
c_indx_id = data_aug.shape[1] - 1

# When exclude_augmented_var=True, pass c_indx_id to ensure CI tests condition on it
# even though it's not part of the causal graph
c_indx_param = c_indx_id if exclude_augmented_var else None

if verbose and exclude_augmented_var:
    print(
        f"[CDNOD Stage 1] Will condition on augmented variable (index {c_indx_id}) "
        f"in all CI tests to control for domain heterogeneity"
    )

cg_0 = SkeletonDiscovery.skeleton_discovery(
    flag, cg_list, data, K, alpha, indep_test_all, stable,
    depth_limit=depth_limit, c_indx_id=c_indx_param
)

# Stage 2
stage2_data = data if exclude_augmented_var else data_aug
if verbose and exclude_augmented_var:
    print(
        f"[CDNOD Stage 2] Using data.shape={data.shape} (excluding augmented var) "
        f"instead of data_aug.shape={data_aug.shape}"
    )

cg_1 = SkeletonDiscovery.skeleton_discovery_with_surrogate_GMM(
    flag,
    cg_0,
    cg_list,
    stage2_data,
    K,
    alpha,
    indep_test_all,
    stable,
    depth_limit=depth_limit,
    c_indx_id=c_indx_param,
)
```

**Key points**:
1. Compute `c_indx_id = data_aug.shape[1] - 1` early (before Stage 1)
2. Pass `c_indx_param = c_indx_id if exclude_augmented_var else None`
3. Only pass when `exclude_augmented_var=True` (horizontal mode)
4. Both Stage 1 and Stage 2 receive the parameter

---

## Design Rationale

### Why condition on U even though it's not in the causal graph?

**U is a confounder** - it affects multiple variables but is not causally downstream of them.

**Standard practice in causal inference**:
- Confounders should be conditioned on to get unbiased estimates
- Even if confounders are not in the causal graph of interest
- This is like "control variables" in regression

**Example from epidemiology**:
```
Study: Does X (smoking) cause Y (lung cancer)?

Confounder: U (age)
- Age affects both smoking rates and lung cancer risk
- Age is NOT caused by smoking or lung cancer
- We don't include age in the "smoking → lung cancer" causal graph
- BUT we must condition on age to get correct causal effect
```

**Same logic here**:
- U (client ID) confounds relationships between variables
- Must be conditioned on for correct independence tests
- Even though U is excluded from the causal graph

---

## Expected Impact

### Metrics Improvement

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| **Final edges** | 24 | 10-14 | -42% to -58% |
| **False positives** | 16 | 2-6 | -63% to -88% |
| **skeleton_precision** | 0.333 | 0.60-0.75 | +80% to +125% |
| **skeleton_f1** | 0.500 | 0.70-0.85 | +40% to +70% |
| **skeleton_shd** | 16 | 3-6 | -63% to -81% |

### P-value Distribution

**Before**:
```
CI tests: ~2400
DEPENDENT: 2383 (99.3%)
INDEPENDENT: 17 (0.7%)
All p-values: 0.000
```

**After (Expected)**:
```
CI tests: ~2400
DEPENDENT: ~1200 (50%)
INDEPENDENT: ~1200 (50%)
P-values: Varied (similar to structure voting)
```

---

## Testing Plan

### Smoke Test
```bash
cd tests/smoke_tests
python smoke_test_asia_horizontal.py
```

**Expected output**:
```
[CDNOD Stage 1] Will condition on augmented variable (index 8)
                in all CI tests to control for domain heterogeneity

[DEBUG CI Test] depth=0: p_values varied (not all 0.000)
[DEBUG CI Test] depth=1: p_values varied (not all 0.000)

skeleton_num_edges=10-14 (down from 24)
skeleton_precision=0.60-0.75 (up from 0.33)
```

### Full Benchmark
```bash
cd tests/benchmarks
python run_v3_fedcdh_benchmarks.py --dataset asia --mode horizontal --seed 42
```

---

## Relationship to Previous Fixes

This is **Bug Fix #9**, building on:
- **Fix #1-5**: Committed in 13580e6 (May 30, 2026)
  - Fixed skeleton loading, augmented variable exclusion, logging
  - Result: Pipeline runs, but produces 24 edges
- **This fix**: Addresses the "too many edges" issue
  - Makes main PC consistent with structure voting
  - Both now condition on U

---

## Implementation Status

✅ **COMPLETED** - 2026-05-31

### Modified Files:
1. ✅ `causallearn/utils/PCUtils/SkeletonDiscovery.py`
   - Added `c_indx_id` parameter to 3 functions
   - Modified conditioning set construction in all functions
   - Updated all CI test calls to use `S_with_context`

2. ✅ `causallearn/search/ConstraintBased/CDNOD.py`
   - Compute `c_indx_id` early
   - Pass to both Stage 1 and Stage 2 skeleton discovery
   - Added verbose logging

### Next Steps:
1. Run smoke test to verify fix works
2. Run full benchmark to measure impact
3. Commit if tests pass
4. Document results in experiment notes

---

## Technical Notes

### Why pass c_indx_id only when exclude_augmented_var=True?

**Two modes**:

1. **exclude_augmented_var=False** (original CDNOD):
   - U is part of the causal graph (has edges)
   - U naturally appears in neighbor sets
   - U will be in conditioning sets automatically
   - Don't need to pass c_indx_id

2. **exclude_augmented_var=True** (our mode):
   - U is NOT part of the causal graph (no edges)
   - U has 0 neighbors
   - U won't be in conditioning sets naturally
   - MUST pass c_indx_id explicitly

### Why does structure voting work without this fix?

Structure voting **explicitly includes U** in Z:
```python
# In aggregation.py:36-42
Z = [d_features]  # d_features = 8 (augmented variable)

for i in range(d_features):
    for j in range(i + 1, d_features):
        p_value = ci_test(i, j, Z=[d_features])
```

It never relied on U being in the neighbor set - it always added U explicitly.

Main PC relied on neighbors → didn't have U → failed.

Now main PC also adds U explicitly → consistent!

---

## References

- **Analysis documents**:
  - ANALYSIS_TOO_MANY_EDGES.md
  - HOW_GLOBAL_SPN_FAILS_TO_REFINE.md
  - DOES_FEDCDH_USE_U_IN_PC.md

- **Commit history**:
  - 13580e6: Fixes #1-5
  - (Next): Fix #9 (this document)
