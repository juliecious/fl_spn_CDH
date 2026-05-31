# CRITICAL BUG: Initial Skeleton Never Actually Loaded

**Date**: 2026-05-30
**Status**: ✅ Fixed

---

## Problem

Despite implementing FIX #3 to use structure voting results as initial skeleton, experiments still showed:
- ❌ Final skeleton: 0 edges
- ❌ All metrics: 0.000
- ❌ SHD: 8 (no improvement)

---

## Root Cause Analysis

### Discovery Process

1. **First Issue**: Method name bug (`.num_edges` → `.get_num_edges()`) - Fixed ✓
2. **Experiment still failed**: Metrics unchanged after fix
3. **Hypothesis**: Initial skeleton not actually being used

### The Smoking Gun

Testing revealed that `CausalGraph` initialization **starts fully connected**:

```python
cg = CausalGraph(8)
print(cg.G.get_num_edges())  # Output: 28 (fully connected!)
```

### What Was Happening

In `CDNOD.py` lines 304-318 (original code):

```python
fed_cg = CausalGraph(no_of_var=8)  # Creates graph with 28 edges
# Now "load" initial skeleton with 22 edges
for node_i in range(8):
    for node_j in range(node_i + 1, 8):
        if initial_skeleton[node_i, node_j] == 1:
            fed_cg.G.graph[node_i, node_j] = -1  # Set to -1 (was already -1!)
            fed_cg.G.graph[node_j, node_i] = -1  # Set to -1 (was already -1!)

print(fed_cg.G.get_num_edges())  # Still 28 edges!
```

**The Problem**:
- CausalGraph starts with ALL 28 edges present (fully connected)
- Setting some edges to `-1` doesn't change anything (they're already `-1`)
- Edges NOT in the initial skeleton remain at `-1` (present)
- Result: Graph still has 28 edges, not 22

### Why This Broke Everything

1. ✅ Structure voting found 22 high-quality edges
2. ❌ CDNOD "loaded" them but graph still had 28 edges (fully connected)
3. ❌ `skeleton_discovery()` condition checked `get_num_edges() > 0` → always TRUE
4. ❌ PC algorithm started with 28 edges (fully connected), not 22
5. ❌ All 28 edges tested at Depth=0 with Z=[] → all show DEPENDENT (p=0.000)
6. ❌ PC removes all edges → final skeleton: 0 edges

---

## The Fix

**File**: `causallearn/search/ConstraintBased/CDNOD.py` lines 304-327

### Key Change: Clear Graph Before Loading Skeleton

```python
fed_cg = CausalGraph(no_of_var=n_causal_vars, node_names=None)

if initial_skeleton is not None:
    if verbose:
        print(f"[CDNOD] Initializing CausalGraph with skeleton from structure voting: "
              f"{initial_skeleton.sum() // 2} edges")

    # CRITICAL: CausalGraph starts fully connected by default!
    # We need to CLEAR it first, then add only the skeleton edges
    fed_cg.G.graph[:] = 0  # ← THE FIX: Remove all edges first

    # Now set ONLY the edges from initial skeleton
    for node_i in range(min(n_causal_vars, initial_skeleton.shape[0])):
        for node_j in range(node_i + 1, min(n_causal_vars, initial_skeleton.shape[1])):
            if initial_skeleton[node_i, node_j] == 1:
                fed_cg.G.graph[node_i, node_j] = -1  # Add edge
                fed_cg.G.graph[node_j, node_i] = -1  # (undirected)

    if verbose:
        actual_edges = fed_cg.G.get_num_edges()
        print(f"[CDNOD] After loading skeleton: {actual_edges} edges in fed_cg.G")
```

---

## Impact

### Before Fix
```python
# Graph state after "loading" 22-edge skeleton
cg.G.get_num_edges() = 28  # Still fully connected!
```

### After Fix
```python
# Graph state after loading 22-edge skeleton
cg.G.get_num_edges() = 22  # Correct!
```

---

## Verification

**Test Script**: `test_skeleton_loading_v2.py`

```
--- OLD WAY (without clearing) ---
Fresh CausalGraph: 28 edges (fully connected)
After loading skeleton (OLD): 28 edges
❌ WRONG: Still fully connected (28 edges), not 22 edges

--- NEW WAY (with clearing first) ---
Fresh CausalGraph: 28 edges (fully connected)
After clearing: 0 edges
After loading skeleton (NEW): 22 edges
✅ CORRECT: 22 edges matches 22 expected

TEST PASSED ✓
```

---

## Additional Fixes

Also added verbose logging to track edge counts:

1. **FedCDH.py** line 2253: Added `"verbose": True` to cdnod_kwargs
2. **CDNOD.py** lines 333-336: Log edge count before passing to skeleton_discovery
3. **SkeletonDiscovery.py** lines 82-87: Log whether initial skeleton is used

---

## Expected Results (Next Run)

With this fix, the pipeline should:

1. ✅ Structure voting finds 22 edges
2. ✅ CDNOD clears graph and loads exactly 22 edges
3. ✅ Verbose log: `[CDNOD] After loading skeleton: 22 edges in fed_cg.G`
4. ✅ `skeleton_discovery()` starts with 22 edges (not 28)
5. ✅ PC refines these 22 edges → final skeleton ~8-12 edges
6. ✅ Metrics: SHD < 8, F1 > 0.6, Precision/Recall > 0.5

---

## Related Files Modified

1. `causallearn/search/ConstraintBased/CDNOD.py` (lines 304-327)
2. `causallearn/search/FCMBased/FedCDH/FedCDH.py` (line 2253)
3. `causallearn/utils/PCUtils/SkeletonDiscovery.py` (lines 82-91)

---

## Conclusion

This was a subtle but critical bug. The code **appeared** to load the initial skeleton, but because `CausalGraph` starts fully connected, the skeleton was never actually applied. The fix is simple: clear the graph first with `fed_cg.G.graph[:] = 0` before loading the skeleton edges.

**All experiments prior to this fix were running PC on fully-connected graphs (28 edges), not the initial skeleton (22 edges).**
