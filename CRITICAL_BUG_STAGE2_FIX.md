# CRITICAL BUG DISCOVERED: Stage 2 Re-introduces Augmented Variable

## Problem Analysis from Test Run (20260530_141220)

### Symptoms
- Metrics still 0 despite FIX #2 and FIX #3 being active
- Initial skeleton from structure voting (22 edges) was passed correctly
- Logs showed "Converted consensus graph to initial skeleton: 22 edges"
- But final skeleton had 0 edges

### Root Cause Discovery

By analyzing test.log, I discovered TWO skeleton discovery phases:

**Phase 1 (Stage 1): CORRECT**
- Progress bar: `0/8` (processing 8 nodes)
- Operates on nodes 0-7 only
- Uses `data` (shape: 999, 8)
- Correctly excludes augmented variable

**Phase 2 (Stage 2): BUG!**
- Progress bar: `0/9` (processing 9 nodes!)
- Tests include variable 8: `[8] ⊥̸ [0]`, `[8] ⊥̸ [1]`, etc.
- Uses `data_aug` (shape: 999, 9)
- **Re-introduces augmented variable into skeleton discovery**

### Code Location

**File**: `causallearn/search/ConstraintBased/CDNOD.py`

**Lines 329-347** (before fix):
```python
# Stage 1
cg_0 = SkeletonDiscovery.skeleton_discovery(
    flag, cg_list, data, K, alpha, indep_test_all, stable, depth_limit=depth_limit
)  # Uses data (8D) ✓

# Stage 2
cg_1 = SkeletonDiscovery.skeleton_discovery_with_surrogate_GMM(
    flag,
    cg_0,
    cg_list,
    data_aug,  # ← BUG! Uses data_aug (9D) even though cg_0 has 8 nodes
    K,
    alpha,
    indep_test_all,
    stable,
    depth_limit=depth_limit,
)
```

### Why This Breaks Everything

**Stage 2 (`skeleton_discovery_with_surrogate_GMM`) behavior**:
1. Takes input graph `cg` from Stage 1 (8 nodes)
2. Takes input `data` parameter
3. Adds a **surrogate node** to the graph (line 656-657)
4. Assumes the **last variable in `data`** is the context variable (line 660)

**When `data_aug` (9D) is passed**:
1. Stage 2 receives `cg_0` with 8 nodes from Stage 1
2. Stage 2 adds surrogate node → 9 nodes total
3. Stage 2 assumes variable 8 (last in `data_aug`) is context
4. Stage 2 adds edges from surrogate to all other nodes (line 662-664)
5. **But variable 8 is NOT the context - it's the augmented variable we wanted to exclude!**
6. Skeleton discovery in Stage 2 now tests variable 8 against all other variables
7. Variable 8 (client ID) shows strong dependence with everything
8. All edges removed during conditioning

**Result**: Empty skeleton despite correct Stage 1

### The Fix

**Change**: Pass `data` (not `data_aug`) to Stage 2 when `exclude_augmented_var=True`

```python
# Stage 2
# FIX #2 (CRITICAL): Use same data dimensionality as Stage 1
stage2_data = data if exclude_augmented_var else data_aug

if verbose and exclude_augmented_var:
    print(f"[CDNOD Stage 2] Using data.shape={data.shape} (excluding augmented var) "
          f"instead of data_aug.shape={data_aug.shape}")

cg_1 = SkeletonDiscovery.skeleton_discovery_with_surrogate_GMM(
    flag,
    cg_0,
    cg_list,
    stage2_data,  # ← FIXED: Use data (8D) when excluding augmented var
    K,
    alpha,
    indep_test_all,
    stable,
    depth_limit=depth_limit,
)
```

### Impact

**Before fix**:
- Stage 1: Correct (8 nodes)
- Stage 2: Wrong (tests 9 nodes including augmented variable)
- Result: Empty skeleton

**After fix**:
- Stage 1: Correct (8 nodes)
- Stage 2: Correct (8 nodes, adds surrogate for context)
- Result: Should preserve edges from Stage 1 and initial skeleton

### Why This Was Hard to Catch

1. **Stage 1 looked correct** in logs (8 nodes)
2. **FIX #2 parameter was being passed** correctly
3. **Initial skeleton was being set** correctly
4. **Only Stage 2 logs revealed the issue**: progress bar showing 9 nodes instead of 8
5. **The bug was in data dimensionality**, not graph initialization

### Lesson Learned

**Multi-stage algorithms need consistent data dimensionality across all stages.**

When adding `exclude_augmented_var` flag, we only fixed:
- CausalGraph initialization (line 301)
- But forgot to propagate the dimensionality change to Stage 2

This is a classic example of:
- **Partial fix**: Fixed one component but missed downstream dependencies
- **Hidden coupling**: Stage 2 assumes data matches graph dimensions
- **Implicit contracts**: skeleton_discovery_with_surrogate_GMM expects last variable to be context

### Complete Fix Checklist

✅ **FIX #2.1**: Set `n_causal_vars` based on `exclude_augmented_var`
✅ **FIX #2.2**: Initialize CausalGraph with `n_causal_vars`
✅ **FIX #2.3**: Pass `stage2_data` to Stage 2 based on `exclude_augmented_var`

All three components are required for the fix to work!

### Testing

**Next run should show**:
- Stage 1: `0/8` progress bar ✓
- Stage 2: `0/8` progress bar ✓ (NOT `0/9`)
- No CI tests involving `[8]` in main PC algorithm
- Skeleton edges > 0
- SHD < 8

### Files Modified

1. **`causallearn/search/ConstraintBased/CDNOD.py`**
   - Added `stage2_data` conditional variable selection
   - Total lines added: ~8 lines

**This completes the fix for FIX #2.**
