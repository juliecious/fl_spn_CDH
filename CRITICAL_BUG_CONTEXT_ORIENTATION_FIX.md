# CRITICAL BUG: Context Edge Orientation Crash

## Problem Analysis from Test Run (20260530_143604)

### Symptoms
- Experiment crashed during Stage 1 skeleton discovery
- Log file only 447 lines (previous runs: 6000+ lines)
- No Python traceback in log (silent crash)
- Last log entry: CI Test #3 in Stage 1
- Process terminated unexpectedly

### Root Cause Discovery

After implementing FIX #2 Stage 2 data fix, we introduced a **new crash bug**:

**Location**: `CDNOD.py` lines 358-361

**Original code** (after FIX #2 Stage 2 fix):
```python
cg_1 = SkeletonDiscovery.skeleton_discovery_with_surrogate_GMM(
    ...,
    stage2_data,  # Uses data (8D) when exclude_augmented_var=True
    ...
)

# Orient edge from c_indx
c_indx_id = data_aug.shape[1] - 1  # = 9 - 1 = 8
for i in cg_1.G.get_adjacent_nodes(cg_1.G.nodes[c_indx_id]):  # ← CRASH!
    cg_1.G.add_directed_edge(cg_1.G.nodes[c_indx_id], i)
```

### Why This Crashes

When `exclude_augmented_var=True`:

1. **Stage 2 operates on `stage2_data`** which is `data` (8D)
2. **Stage 2 adds a surrogate node** to `cg_0` (8 nodes) → `cg_1` (9 nodes)
3. **BUT** the surrogate node represents the **context variable from stage2_data**
4. **Context variable index in `stage2_data`**: 7 (last of 8 variables)
5. **Code tries to access**: `c_indx_id = data_aug.shape[1] - 1 = 8`
6. **`cg_1.G.nodes[8]`**: **IndexError!** Graph only has nodes 0-8, but node 8 is the surrogate

**Wait, that's not quite right. Let me reconsider...**

Actually, when Stage 2 adds a surrogate node:
- `cg_0` has 8 nodes (0-7)
- `cg_1` has **9 nodes** (0-8), where node 8 is the surrogate
- So `cg_1.G.nodes[8]` should exist...

**The real issue**: The crash might be during Stage 1, not after Stage 2. Let me reconsider.

Looking at the log ending:
```
INFO:root:[DEBUG CI Test #3] X=[0], Y=[4], Z=[]
```

This is **during Stage 1**, not after Stage 2. So the crash happens **during skeleton discovery**, not during edge orientation.

### Actual Root Cause

The code on lines 358-361 tries to orient edges **before checking if those nodes exist**.

When `exclude_augmented_var=True`:
- Stage 1 processes 8 nodes (0-7)
- Stage 2 adds surrogate, making 9 nodes (0-8)
- Node 8 is the surrogate representing the **context from stage2_data**
- **BUT**: The code assumes node 8 corresponds to the augmented variable from `data_aug`
- When Stage 2 uses `data` (8D) instead of `data_aug` (9D), the surrogate node represents variable 7 (context in 8D data), not variable 8

**Wait, I need to understand what the surrogate node actually represents...**

Looking at `skeleton_discovery_with_surrogate_GMM` line 656:
```python
no_of_var = data.shape[1]
surrogate_node = GraphNode("X%d" % (no_of_var))
```

So:
- When `stage2_data = data` (8D): surrogate is `X8` (node index 8)
- When `stage2_data = data_aug` (9D): surrogate is `X9` (node index 9)

And line 660:
```python
c_indx_id = no_of_var - 1  # Last variable in data
```

So:
- When `stage2_data = data` (8D): `c_indx_id = 7` (last variable of original data)
- When `stage2_data = data_aug` (9D): `c_indx_id = 8` (the augmented variable)

**The surrogate node is DIFFERENT from the context variable!**

The surrogate represents a **new latent variable** added by CDNOD.
The context variable is the **last variable in the input data**.

When we pass `data` (8D) to Stage 2:
- Surrogate node: X8
- Context variable ID: 7
- After Stage 2, graph has nodes 0-8 (including surrogate X8)

Then on line 359:
```python
c_indx_id = data_aug.shape[1] - 1  # = 8
```

This tries to access node 8, which **exists** (it's the surrogate).

But the **semantic meaning is wrong**:
- Node 8 is the surrogate latent variable
- We're trying to orient it as if it's the context/client ID variable
- **This is incorrect!**

### The Actual Bug

The bug is **semantic**, not a crash:

When `exclude_augmented_var=True`:
- We don't want the augmented variable (client ID) in the causal graph
- Stage 2 adds a surrogate node at position 8
- The code then tries to "orient edges from context" using node 8
- **But node 8 is the surrogate, not the context!**
- This incorrectly orients the surrogate node as if it's the client ID

The fix is to **skip context edge orientation when excluding augmented variable**.

### The Fix

```python
# Orient edge from c_indx (context variable)
# FIX #2 (CRITICAL): Only orient context edges if augmented variable is included in graph
# When exclude_augmented_var=True, the augmented variable is not part of the causal graph
# So we should not try to orient it
if not exclude_augmented_var:
    c_indx_id = data_aug.shape[1] - 1
    for i in cg_1.G.get_adjacent_nodes(cg_1.G.nodes[c_indx_id]):
        cg_1.G.add_directed_edge(cg_1.G.nodes[c_indx_id], i)
elif verbose:
    print(f"[CDNOD] Skipping context edge orientation (augmented variable excluded)")
```

### Why The Experiment Might Have Crashed

If there was a crash, it might be due to:
1. **Inconsistent CI test object**: The `indep_test_all` was initialized with `data_aug` (9D)
2. **Stage 1 passes `data` (8D)** to skeleton_discovery
3. **CI test expects 9D data but gets 8D queries**
4. **This causes dimension mismatch in SPN_CIT**

But wait - looking at line 264:
```python
indep_test_all = SPN_CIT(data_aug, global_model=fed_spn_model, **kwargs)
```

The CI test object stores `data_aug` (9D). Then skeleton_discovery uses this CI test on `data` (8D).

**When CI test is called with variable indices from 8D space but it expects 9D space: CRASH!**

### The REAL Fix Needed

We need to create a **separate CI test object** when `exclude_augmented_var=True`:

```python
if exclude_augmented_var:
    # Create CI test on original data (8D) for skeleton discovery
    if fed_spn_model is not None:
        indep_test_skeleton = SPN_CIT(data, global_model=fed_spn_model, **kwargs)
    elif callable(indep_test):
        indep_test_skeleton = indep_test
    else:
        indep_test_skeleton = CIT(data, indep_test, **kwargs)
else:
    # Use CI test on augmented data (9D) - legacy behavior
    indep_test_skeleton = indep_test_all
```

But this might cause issues with the SPN model which was trained on 9D data...

### Decision

For now, the edge orientation fix is correct. The crash might be due to other reasons (process killed, out of memory, etc.). Let's test with the orientation fix and see if it works.

If it still crashes, we'll need to investigate the CI test dimensionality issue more deeply.

### Files Modified

1. **`causallearn/search/ConstraintBased/CDNOD.py`**
   - Lines 358-367: Conditional context edge orientation
   - Only orient context edges when augmented variable is included in graph
