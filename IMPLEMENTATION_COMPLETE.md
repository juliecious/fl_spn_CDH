# Bug Fix #9 Implementation Complete

**Date**: 2026-05-31
**Status**: ✅ IMPLEMENTED - Ready for Testing

---

## Summary

Successfully implemented the fix for **all p-values = 0.000** issue that caused the PC algorithm to produce 24 edges instead of ~10-14.

### Root Cause
The main PC algorithm was not conditioning on the augmented variable U (client ID), causing spurious dependencies due to marginalization over U.

### Solution
Modified the PC algorithm to always include U in conditioning sets when `exclude_augmented_var=True` (horizontal mode).

---

## Files Modified

### 1. `causallearn/utils/PCUtils/SkeletonDiscovery.py`

#### Modified 3 Functions:

**A. `skeleton_discovery` (Line 25)**
- Added parameter: `c_indx_id: int | None = None`
- Modified conditioning set construction (lines 149-203)
- Now includes U in all CI tests: `S_with_context = tuple(S) + (c_indx_id,)`

**B. `skeleton_discovery_with_surrogate` (Line 347)**
- Added parameter: `c_indx_id: int | None = None`
- Modified conditioning set construction (lines 478-579)
- Updated all voting scheme variations to use `S_with_context`

**C. `skeleton_discovery_with_surrogate_GMM` (Line 627)**
- Added parameter: `c_indx_id: int | None = None`
- Modified conditioning set construction (lines 747-809)
- Updated all CI test calls to use `S_with_context`

**Key change pattern**:
```python
# Before:
for S in combinations(Neigh_x_noy, depth):
    p = cg.ci_test(x, y, S)

# After:
for S in combinations(Neigh_x_noy, depth):
    if c_indx_id is not None:
        S_with_context = tuple(S) + (c_indx_id,)
    else:
        S_with_context = S
    p = cg.ci_test(x, y, S_with_context)
```

---

### 2. `causallearn/search/ConstraintBased/CDNOD.py`

#### Modified `cdnod_alg` Function (Lines 360-402):

**Before Stage 1**:
```python
# Compute c_indx_id early
c_indx_id = data_aug.shape[1] - 1

# Pass when exclude_augmented_var=True
c_indx_param = c_indx_id if exclude_augmented_var else None

if verbose and exclude_augmented_var:
    print(
        f"[CDNOD Stage 1] Will condition on augmented variable (index {c_indx_id}) "
        f"in all CI tests to control for domain heterogeneity"
    )
```

**Stage 1 Call**:
```python
cg_0 = SkeletonDiscovery.skeleton_discovery(
    flag, cg_list, data, K, alpha, indep_test_all, stable,
    depth_limit=depth_limit, c_indx_id=c_indx_param  # NEW
)
```

**Stage 2 Call**:
```python
cg_1 = SkeletonDiscovery.skeleton_discovery_with_surrogate_GMM(
    flag, cg_0, cg_list, stage2_data, K, alpha, indep_test_all, stable,
    depth_limit=depth_limit,
    c_indx_id=c_indx_param,  # NEW
)
```

---

## How It Works

### Conditioning Set Construction

#### Before Fix (Broken):
```
Depth 0: Test X_i ⊥ X_j | []
Depth 1: Test X_i ⊥ X_j | [neighbor₁]
Depth 2: Test X_i ⊥ X_j | [neighbor₁, neighbor₂]

Query to GlobalSPN: P(X_i, X_j) - marginalizes over U
Result: All p = 0.000 (spurious dependence)
```

#### After Fix (Working):
```
Depth 0: Test X_i ⊥ X_j | [U]
Depth 1: Test X_i ⊥ X_j | [U, neighbor₁]
Depth 2: Test X_i ⊥ X_j | [U, neighbor₁, neighbor₂]

Query to GlobalSPN: P(X_i, X_j | U) - conditions on U
Result: Varied p-values (correct independence tests)
```

---

## Expected Impact

### Before Fix (Experiment 20260530_205318):
```
Initial skeleton: 22 edges (from structure voting)
After PC Stage 1: 22 edges (NO refinement, all p=0.000)
After PC Stage 2: 24 edges (ADDED 2 more!)
Metrics:
  skeleton_precision: 0.333
  skeleton_f1: 0.500
  skeleton_shd: 16
```

### After Fix (Expected):
```
Initial skeleton: 22 edges (from structure voting)
After PC Stage 1: 12-16 edges (refined, varied p-values)
After PC Stage 2: 10-14 edges (further refined)
Metrics:
  skeleton_precision: 0.60-0.75 (+80% to +125%)
  skeleton_f1: 0.70-0.85 (+40% to +70%)
  skeleton_shd: 3-6 (-63% to -81%)
```

---

## Testing

### Quick Test
```bash
python test_conditioning_fix.py
```

**Look for**:
1. Log message: `[CDNOD Stage 1] Will condition on augmented variable (index 8)`
2. Final edges: 10-14 (not 24)
3. No errors during execution

### Full Benchmark
```bash
cd experiments
python run_benchmark.py --dataset asia --mode horizontal --seed 42
```

**Expected results**:
- Check `test.log` for varied p-values (not all 0.000)
- skeleton_num_edges: 10-14
- skeleton_precision: 0.60+
- skeleton_f1: 0.70+

---

## Verification Checklist

- [x] Modified all 3 skeleton_discovery functions in SkeletonDiscovery.py
- [x] Added c_indx_id parameter to function signatures
- [x] Modified conditioning set construction to include U
- [x] Updated all CI test calls to use S_with_context
- [x] Modified CDNOD.py to compute c_indx_id early
- [x] Pass c_indx_id to both Stage 1 and Stage 2
- [x] Added verbose logging to show when conditioning on U
- [x] Created test script (test_conditioning_fix.py)
- [x] Created comprehensive documentation (BUG_9_CONDITIONING_ON_U_FIX.md)
- [ ] Run smoke test (pending)
- [ ] Run full benchmark (pending)
- [ ] Commit changes (next step)

---

## Next Steps

1. **Test the fix**:
   ```bash
   python test_conditioning_fix.py
   ```

2. **If test passes, commit**:
   ```bash
   git add causallearn/utils/PCUtils/SkeletonDiscovery.py
   git add causallearn/search/ConstraintBased/CDNOD.py
   git commit -m "fix: condition on augmented variable U in main PC algorithm

   - Fixes Bug #9: all p-values = 0.000 causing too many edges (24 vs 8)
   - Modified skeleton_discovery functions to accept c_indx_id parameter
   - When exclude_augmented_var=True, always include U in conditioning sets
   - Prevents spurious dependence from marginalizing over client ID
   - Expected: final skeleton ~10-14 edges with precision 0.60-0.75

   Modified files:
   - causallearn/utils/PCUtils/SkeletonDiscovery.py: Added c_indx_id param to 3 functions
   - causallearn/search/ConstraintBased/CDNOD.py: Pass c_indx_id to skeleton_discovery calls

   Related: DOES_FEDCDH_USE_U_IN_PC.md, HOW_GLOBAL_SPN_FAILS_TO_REFINE.md"
   ```

3. **Run full benchmark**:
   ```bash
   cd experiments
   python run_benchmark.py --dataset asia --mode horizontal --seed 42
   ```

4. **Analyze results**:
   - Check final skeleton edges (should be 10-14)
   - Check precision (should be 0.60+)
   - Check test.log for p-value distribution
   - Compare with experiment 20260530_205318

---

## Design Notes

### Why pass c_indx_id only when exclude_augmented_var=True?

**Two operational modes**:

1. **exclude_augmented_var=False** (original CDNOD):
   - U is part of the causal graph (has edges TO variables)
   - U naturally appears in neighbor sets
   - U will be in conditioning sets automatically via neighbors
   - Don't need to pass c_indx_id explicitly

2. **exclude_augmented_var=True** (our horizontal mode):
   - U is NOT part of the causal graph (excluded to get 8-node graph)
   - U has 0 neighbors (no edges TO/FROM U)
   - U won't be in conditioning sets via neighbors
   - MUST pass c_indx_id explicitly to include it

### Why didn't structure voting have this problem?

Structure voting **explicitly hardcoded U** into Z:
```python
# In aggregation.py:36-42
Z = [d_features]  # d_features = 8 (augmented variable index)

for i in range(d_features):
    for j in range(i + 1, d_features):
        p_value = ci_test(i, j, Z=[d_features])
```

It never relied on U being in the neighbor set - it always added U directly in the Z parameter.

Main PC relied on building Z from neighbors → U wasn't in neighbors → U wasn't in Z → failed.

Now main PC also adds U explicitly (via c_indx_id parameter) → consistent with structure voting!

---

## Technical Details

### What happens during a CI test with c_indx_id?

**Without c_indx_id** (Before):
```python
# Test: X₀ ⊥ X₂ | []
S = []
p = cg.ci_test(0, 2, [])

# GlobalSPN computes:
P(X₀, X₂) = Σ_u P(X₀, X₂ | U=u) · P(U=u)
# Marginalization over U creates spurious correlation
Result: p = 0.000 (appears dependent)
```

**With c_indx_id** (After):
```python
# Test: X₀ ⊥ X₂ | [U]
S = []
S_with_context = tuple(S) + (8,)  # c_indx_id = 8
p = cg.ci_test(0, 2, [8])

# GlobalSPN computes:
P(X₀, X₂ | U) - properly controls for client membership
# Conditioning on U removes confounding
Result: p = 0.065 or other varied value (correct test)
```

### Data flow

```
FedCDH.fit()
  ↓
cdnod_alg()
  ↓
  # Compute c_indx_id = data_aug.shape[1] - 1 = 8
  # c_indx_param = 8 (when exclude_augmented_var=True)
  ↓
skeleton_discovery(..., c_indx_id=8)
  ↓
  for S in combinations(neighbors, depth):
      S_with_context = tuple(S) + (8,)  # Add U to every conditioning set
      p = cg.ci_test(x, y, S_with_context)
  ↓
CausalGraph.ci_test(x, y, [8])
  ↓
FCIT test with GlobalSPN
  ↓
P(x, y | U=8) - conditioning on U controls confounding
  ↓
Varied p-values (not all 0.000)
  ↓
Proper edge removal
  ↓
Final skeleton: 10-14 edges (not 24)
```

---

## Related Documentation

- **BUG_9_CONDITIONING_ON_U_FIX.md**: Detailed explanation of the fix
- **DOES_FEDCDH_USE_U_IN_PC.md**: Analysis of why U wasn't used before
- **HOW_GLOBAL_SPN_FAILS_TO_REFINE.md**: How GlobalSPN participates but fails
- **ANALYSIS_TOO_MANY_EDGES.md**: Why 24 edges instead of 8
- **WORKFLOW_HORIZONTAL_FEDSPN.md**: Complete pipeline explanation

---

## Implementation Complete

All code changes have been made. The fix is ready for testing and commit.

**Files modified**:
1. `causallearn/utils/PCUtils/SkeletonDiscovery.py` - 3 functions updated
2. `causallearn/search/ConstraintBased/CDNOD.py` - cdnod_alg function updated

**Test created**:
- `test_conditioning_fix.py` - Quick validation script

**Documentation created**:
- `BUG_9_CONDITIONING_ON_U_FIX.md` - Comprehensive fix documentation
- `IMPLEMENTATION_COMPLETE.md` - This file

**Ready for**: Testing → Commit → Benchmark
