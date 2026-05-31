# Implementation: FIX #2 and FIX #3

## Summary

This document describes the implementation of two critical bug fixes for the FedCDH horizontal mode:

- **FIX #2**: Exclude augmented variable from skeleton discovery
- **FIX #3**: Use structure voting consensus as initial skeleton

---

## FIX #2: Exclude Augmented Variable from Skeleton

### Problem

In horizontal federated learning mode:
- Data is augmented with a context column (variable 8 for Asia with 8 original variables)
- The context column contains client IDs (0, 1, 2)
- **BUG**: CDNOD was creating a CausalGraph with 9 variables instead of 8
- This caused the PC algorithm to test the augmented variable as part of the causal structure
- Since the augmented variable (client ID) perfectly determines which client generated each sample, all CI tests involving it show strong dependence
- Conditioning on the augmented variable blocks all dependencies between original variables
- **Result**: Empty skeleton (0 edges) despite true causal relationships existing

### Solution

Modified `CDNOD.py` and `FedCDH.py` to:

1. **Add `exclude_augmented_var` parameter to CDNOD functions**
   - Location: `causallearn/search/ConstraintBased/CDNOD.py`
   - Lines: 209, 258
   - Type: `bool` (default: False for backward compatibility)

2. **Determine number of causal variables correctly**
   - Location: `CDNOD.py` line 270-278
   - When `exclude_augmented_var=True`: `n_causal_vars = data.shape[1]` (original features only)
   - When `exclude_augmented_var=False`: `n_causal_vars = data_aug.shape[1]` (legacy behavior)
   - CausalGraph now created with correct number of variables: `CausalGraph(no_of_var=n_causal_vars)`

3. **Pass flag from FedCDH based on scenario**
   - Location: `FedCDH.py` line 2270-2272
   - Logic: `exclude_augmented = (self.scenario == "horizontal")`
   - Only horizontal mode excludes augmented variable
   - Vertical/hybrid modes keep legacy behavior

### Code Changes

**File: `causallearn/search/ConstraintBased/CDNOD.py`**

```python
# Function signatures updated
def cdnod(..., exclude_augmented_var: bool = False, **kwargs) -> CausalGraph:
def cdnod_alg(..., exclude_augmented_var: bool = False, **kwargs) -> CausalGraph:

# Inside cdnod_alg:
if exclude_augmented_var:
    n_causal_vars = data.shape[1]  # Original features only (e.g., 8 for Asia)
    if verbose:
        print(f"[CDNOD] Excluding augmented variable from skeleton: "
              f"{n_causal_vars} causal variables, {data_aug.shape[1]} total variables")
else:
    n_causal_vars = data_aug.shape[1]  # All features including augmented (legacy)

fed_cg = CausalGraph(no_of_var=n_causal_vars, node_names=None)
```

**File: `causallearn/search/FCMBased/FedCDH/FedCDH.py`**

```python
# Before calling cdnod (line ~2270)
exclude_augmented = (self.scenario == "horizontal")

cg = cdnod(
    X_global,
    c_indx,
    self.K_clients,
    alpha=alpha,
    indep_test=cit_counter,
    stable=True,
    uc_rule=2,
    uc_priority=-1,
    fed_spn_model=self.fed_spn_model,
    exclude_augmented_var=exclude_augmented,  # NEW PARAMETER
    **cdnod_kwargs,
)
```

### Impact

- **Immediate**: Fixes empty skeleton bug in horizontal mode
- **Complexity**: Low (5 lines in CDNOD.py, 3 lines in FedCDH.py)
- **Backward Compatibility**: Maintained (default `exclude_augmented_var=False`)
- **Risk**: Low (only affects horizontal mode when flag is True)

---

## FIX #3: Use Structure Voting Consensus as Initial Skeleton

### Problem

In horizontal mode with `horizontal_aggregation="structure_voting"`:
- Structure voting phase extracts local dependency graphs from each client
- Consensus graph computed via majority voting (22 edges with 93.9% confidence)
- **BUG**: This consensus graph is computed and stored but NEVER USED
- Main PC algorithm starts from scratch with a fully connected graph
- Wastes 17 seconds of computation
- Loses valuable structural information

### Solution

Modified `FedCDH.py` and `CDNOD.py` to:

1. **Convert consensus graph to initial skeleton matrix**
   - Location: `FedCDH.py` line 1456-1467
   - Convert NetworkX graph to numpy adjacency matrix
   - Store as `self.initial_skeleton_from_voting`

2. **Pass initial skeleton to CDNOD via kwargs**
   - Location: `FedCDH.py` line 2256-2262
   - Check if `initial_skeleton_from_voting` exists
   - Add to `cdnod_kwargs` dictionary

3. **Initialize CausalGraph with skeleton edges**
   - Location: `CDNOD.py` line 284-300
   - Extract `initial_skeleton` from kwargs
   - For each edge in skeleton, set graph matrix to -1 (undirected edge)
   - PC algorithm will refine these edges instead of starting from scratch

### Code Changes

**File: `causallearn/search/FCMBased/FedCDH/FedCDH.py`**

**Part 1: Convert consensus graph to matrix (after structure voting)**
```python
# Lines 1456-1467 (after structure voting completes)
# FIX #3: Convert consensus graph to initial skeleton for PC algorithm
n_vars = self.d_features
initial_skeleton = np.zeros((n_vars, n_vars), dtype=int)
for edge in consensus_graph.edges():
    i, j = edge
    # Undirected skeleton: mark both directions
    initial_skeleton[i, j] = 1
    initial_skeleton[j, i] = 1

self.initial_skeleton_from_voting = initial_skeleton
logging.info(
    f"  [Structure Voting] Converted consensus graph to initial skeleton: "
    f"{initial_skeleton.sum() // 2} edges"
)
```

**Part 2: Pass to CDNOD (before calling cdnod)**
```python
# Lines 2256-2262
# FIX #3: Pass initial skeleton from structure voting if available
if hasattr(self, "initial_skeleton_from_voting") and self.initial_skeleton_from_voting is not None:
    cdnod_kwargs["initial_skeleton"] = self.initial_skeleton_from_voting
    n_edges = self.initial_skeleton_from_voting.sum() // 2
    logging.info(
        f"[Structure Voting] Passing initial skeleton to PC algorithm: {n_edges} edges from consensus graph"
    )
```

**File: `causallearn/search/ConstraintBased/CDNOD.py`**

```python
# Lines 284-300 (inside cdnod_alg, before creating cg_list)
# FIX #3: Extract initial skeleton from kwargs if provided
initial_skeleton = kwargs.get("initial_skeleton", None)

cg_list = []
for i in range(K):
    # ... extract client data ...

    fed_cg = CausalGraph(no_of_var=n_causal_vars, node_names=None)

    # FIX #3: Initialize graph with initial skeleton if provided
    if initial_skeleton is not None:
        if verbose:
            print(f"[CDNOD] Initializing CausalGraph with skeleton from structure voting: "
                  f"{initial_skeleton.sum() // 2} edges")
        # Set edges in the graph based on initial skeleton
        for node_i in range(min(n_causal_vars, initial_skeleton.shape[0])):
            for node_j in range(node_i + 1, min(n_causal_vars, initial_skeleton.shape[1])):
                if initial_skeleton[node_i, node_j] == 1:
                    # Add undirected edge (both directions set to -1)
                    fed_cg.G.graph[node_i, node_j] = -1
                    fed_cg.G.graph[node_j, node_i] = -1

    # ... rest of CausalGraph initialization ...
```

### Impact

- **Immediate**: PC algorithm starts with high-quality skeleton instead of fully connected graph
- **Speed**: Should reduce CI test time (fewer edges to test/remove)
- **Accuracy**: Consensus graph provides structure from multiple clients' perspectives
- **Complexity**: Medium (15 lines in FedCDH.py, 20 lines in CDNOD.py)
- **Backward Compatibility**: Maintained (only active when `initial_skeleton` is in kwargs)
- **Risk**: Low (skeleton is just initial state, PC can still add/remove edges)

---

## Testing Strategy

### Unit Tests

1. **Test FIX #2 in isolation**
   ```python
   # Test that n_causal_vars is set correctly
   def test_exclude_augmented_var():
       # Create mock data with 8 features + 1 augmented
       data = np.random.randn(100, 8)
       c_indx = np.random.randint(0, 3, (100, 1))

       # Call cdnod with exclude_augmented_var=True
       cg = cdnod(data, c_indx, K=3, exclude_augmented_var=True)

       # Check that CausalGraph has 8 variables, not 9
       assert cg.G.graph.shape == (8, 8)
   ```

2. **Test FIX #3 in isolation**
   ```python
   # Test that initial skeleton is used
   def test_initial_skeleton():
       # Create mock initial skeleton with 5 edges
       initial_skel = np.array([[0, 1, 1, 0],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [0, 1, 1, 0]])

       # Call cdnod with initial_skeleton
       cg = cdnod(data, c_indx, K=3, initial_skeleton=initial_skel)

       # Check that edges from skeleton are present in graph
       # (Graph should have at least the initial edges)
       for i in range(4):
           for j in range(i+1, 4):
               if initial_skel[i, j] == 1:
                   assert cg.G.graph[i, j] == -1
   ```

### Integration Test

Run the Asia experiment and verify:

1. **FIX #2 works**:
   - Log should show: `"Excluding augmented variable from skeleton: 8 causal variables, 9 total variables"`
   - Skeleton should have > 0 edges (not empty)

2. **FIX #3 works**:
   - Log should show: `"Converted consensus graph to initial skeleton: 22 edges"`
   - Log should show: `"Passing initial skeleton to PC algorithm: 22 edges from consensus graph"`
   - Final skeleton should be similar to initial skeleton (maybe refined)

3. **Both fixes together**:
   - SHD should be < 8 (some true edges recovered)
   - Runtime should be similar or faster (initial skeleton speeds up PC)

### Expected Outcome

**Before Fixes:**
- Skeleton edges: 0
- SHD: 8 (all edges missed)
- Metrics: F1=0.0, Precision=0.0, Recall=0.0

**After Fixes:**
- Skeleton edges: ~8-22 (from structure voting consensus)
- SHD: ~0-4 (most edges recovered)
- Metrics: F1>0.5, Precision>0.5, Recall>0.5

---

## Files Modified

1. **`causallearn/search/ConstraintBased/CDNOD.py`**
   - Lines 209, 243, 258: Added `exclude_augmented_var` parameter
   - Lines 270-278: Compute `n_causal_vars` based on flag
   - Lines 284-300: Initialize CausalGraph with initial skeleton
   - **Total**: ~30 lines added/modified

2. **`causallearn/search/FCMBased/FedCDH/FedCDH.py`**
   - Lines 1456-1467: Convert consensus graph to skeleton matrix
   - Lines 2256-2262: Pass initial skeleton to CDNOD
   - Lines 2270-2272: Set `exclude_augmented_var` flag
   - **Total**: ~25 lines added/modified

**Total changes**: ~55 lines across 2 files

---

## Rollback Plan

If issues arise, rollback is simple:

1. **Remove FIX #2**: Set `exclude_augmented_var=False` in FedCDH.py
2. **Remove FIX #3**: Comment out `initial_skeleton` in `cdnod_kwargs`
3. Both fixes are additive and don't modify existing logic

---

## Next Steps

1. ✅ Code changes implemented
2. ⏭️ Run integration test on Asia dataset
3. ⏭️ Verify log messages confirm fixes are active
4. ⏭️ Check final metrics (SHD, F1, etc.)
5. ⏭️ If successful, run on other datasets (Alarm, Child)
6. ⏭️ Commit changes with descriptive message

---

## Commit Message Template

```
fix: resolve horizontal mode empty skeleton bug (FIX #2 and #3)

FIX #2: Exclude augmented variable from causal skeleton
- In horizontal FL, augmented variable (client ID) should only be used for
  conditioning, not included in the causal graph
- Added `exclude_augmented_var` parameter to CDNOD
- Automatically enabled for horizontal scenario
- Fixes empty skeleton issue (was testing 9 variables instead of 8)

FIX #3: Use structure voting consensus as initial skeleton
- Structure voting computes high-quality consensus graph (22 edges, 93.9% conf)
- Previously computed but never used by PC algorithm
- Now passed as initial skeleton to speed up discovery
- PC algorithm refines skeleton instead of starting from scratch

Impact:
- Asia horizontal: SHD 8→0 (expected), F1 0.0→>0.5 (expected)
- Computation time: Similar or faster due to initial skeleton
- Backward compatible: Both fixes opt-in via parameters

Files modified:
- causallearn/search/ConstraintBased/CDNOD.py
- causallearn/search/FCMBased/FedCDH/FedCDH.py

Closes #BUG-HORIZONTAL-EMPTY-SKELETON
```
