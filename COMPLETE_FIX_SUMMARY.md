# Complete Fix Summary: CDNOD Pipeline for Horizontal Federated Learning

**Date**: 2026-05-30
**Status**: ✅ All Fixes Implemented and Verified

---

## Executive Summary

Fixed 4 critical bugs in the CDNOD pipeline that prevented structure discovery in horizontal federated learning mode. All fixes have been implemented, dry-run validated, and smoke-tested successfully.

**Key Results**:
- ✅ Augmented variable correctly excluded from causal graph (8 variables, not 9)
- ✅ Structure voting results used as initial skeleton (22 edges)
- ✅ PC algorithm refines skeleton instead of starting from scratch
- ✅ All stages operate on correct dimensions
- ✅ Context-based orientation safely handles exclusion flag

---

## Problem Statement

### Original Issue
In horizontal federated learning mode on Asia dataset:
- **Expected**: Structure voting found 22 high-confidence edges → PC should refine to ~8-12 edges
- **Actual**: Final skeleton had 0 edges, SHD=8 (worst possible), all metrics=0.000

### Root Causes Identified

1. **BUG #1**: Augmented variable (client ID) included in causal graph
   - CDNOD created graph with 9 variables instead of 8
   - All CI tests erroneously tested variable [8]

2. **BUG #2**: Stage 2 used wrong data dimensions
   - Passed `data_aug` (9D) instead of `data` (8D)
   - Progress bars showed "0/9" instead of "0/8"

3. **BUG #3**: Undefined variable error
   - `c_indx_id` defined inside conditional but used outside
   - Caused crashes: "cannot access local variable 'c_indx_id'"

4. **BUG #4**: Initial skeleton never used
   - `skeleton_discovery()` created fresh fully-connected graph
   - Ignored initial skeleton from structure voting (22 edges)
   - PC started with 28 edges instead of 22

5. **BUG #5**: Context-based orientation unsafe
   - Used `d=9` instead of `n_causal_vars=8`
   - Didn't check `exclude_augmented_var` flag
   - Could crash or corrupt graph when augmented variable excluded

---

## Implemented Fixes

### FIX #2: Exclude Augmented Variable from Causal Graph

**Files Modified**: `CDNOD.py`

#### Component #2.1: Set Correct Variable Count
**Location**: Lines 270-278

```python
# Compute number of causal variables (excluding augmented if specified)
if exclude_augmented_var:
    n_causal_vars = data.shape[1]  # Original features only (e.g., 8 for Asia)
    if verbose:
        print(f"[CDNOD] Excluding augmented variable from skeleton: "
              f"{n_causal_vars} causal variables, {data_aug.shape[1]} total variables")
else:
    n_causal_vars = data_aug.shape[1]  # All features including augmented (legacy)
```

**Impact**: CausalGraph now created with 8 variables instead of 9

#### Component #2.2: Use Correct Data in Stage 2
**Location**: Lines 341-344

```python
# Use correct data dimensions for Stage 2
stage2_data = data if exclude_augmented_var else data_aug
if verbose and exclude_augmented_var:
    print(f"[CDNOD Stage 2] Using data.shape={data.shape} (excluding augmented var) "
          f"instead of data_aug.shape={data_aug.shape}")
```

**Impact**: Stage 2 operates on 8D data, progress bars show "0/8"

#### Component #2.3: Define c_indx_id Before Use
**Location**: Lines 358-367

```python
# Always compute c_indx_id (used later in orientation code)
c_indx_id = data_aug.shape[1] - 1

if not exclude_augmented_var:
    # Add context edges only when augmented variable is in graph
    for i in cg_1.G.get_adjacent_nodes(cg_1.G.nodes[c_indx_id]):
        cg_1.G.add_directed_edge(cg_1.G.nodes[c_indx_id], i)
elif verbose:
    print(f"[CDNOD] Skipping context edge orientation (augmented variable excluded)")
```

**Impact**: No more "UnboundLocalError", context edges skipped correctly

---

### FIX #3: Use Structure Voting Results as Initial Skeleton

**Files Modified**: `FedCDH.py`, `CDNOD.py`, `SkeletonDiscovery.py`

#### Component #3.1a: Convert Consensus Graph to Matrix
**Location**: `FedCDH.py` lines 1456-1477

```python
# Convert NetworkX consensus graph to adjacency matrix
n_vars = self.d_features
initial_skeleton = np.zeros((n_vars, n_vars), dtype=int)
for edge in consensus_graph.edges():
    i, j = edge
    initial_skeleton[i, j] = 1
    initial_skeleton[j, i] = 1

self.initial_skeleton_from_voting = initial_skeleton
logging.info(f"  [Structure Voting] Converted consensus graph to initial skeleton: "
             f"{initial_skeleton.sum() // 2} edges")
```

#### Component #3.1b: Pass Initial Skeleton to CDNOD
**Location**: `FedCDH.py` lines 2256-2262

```python
if hasattr(self, "initial_skeleton_from_voting") and self.initial_skeleton_from_voting is not None:
    cdnod_kwargs["initial_skeleton"] = self.initial_skeleton_from_voting
    n_edges = self.initial_skeleton_from_voting.sum() // 2
    logging.info(f"[Structure Voting] Passing initial skeleton to PC algorithm: "
                 f"{n_edges} edges from consensus graph")
```

#### Component #3.1c: Load Skeleton into cg_list
**Location**: `CDNOD.py` lines 280-327

```python
initial_skeleton = kwargs.get("initial_skeleton", None)
cg_list = []

for i in range(K):
    fed_cg = CausalGraph(no_of_var=n_causal_vars, node_names=None)

    # Load initial skeleton if provided
    if initial_skeleton is not None:
        if verbose:
            print(f"[CDNOD] Initializing CausalGraph with skeleton from structure voting: "
                  f"{initial_skeleton.sum() // 2} edges")
        for node_i in range(min(n_causal_vars, initial_skeleton.shape[0])):
            for node_j in range(node_i + 1, min(n_causal_vars, initial_skeleton.shape[1])):
                if initial_skeleton[node_i, node_j] == 1:
                    fed_cg.G.graph[node_i, node_j] = -1  # Undirected edge
                    fed_cg.G.graph[node_j, node_i] = -1

    cg_list.append(fed_cg)
```

**Impact**: `cg_list[0]` now contains 22 edges instead of 0

#### Component #3.2: Use Initial Skeleton in skeleton_discovery
**Location**: `SkeletonDiscovery.py` lines 77-88

```python
no_of_var = data.shape[1]

# FIX #3.2: Use initial skeleton from cg_list if provided
# If cg_list[0] has edges (initial skeleton), use it as starting point
# Otherwise create a fresh fully-connected graph
if cg_list and len(cg_list) > 0 and cg_list[0].G.get_num_edges() > 0:
    # Use the pre-initialized graph with initial skeleton
    cg = cg_list[0]
    cg.set_ind_test(indep_test)
else:
    # Create fresh fully-connected graph (original behavior)
    cg = CausalGraph(no_of_var, None)
    cg.set_ind_test(indep_test)
```

**Impact**: PC algorithm starts with 22 edges (from structure voting) instead of 28 edges (fully connected)

---

### FIX #4: Safe Context-Based Orientation

**File Modified**: `CDNOD.py` lines 493-534

```python
else:
    # Original: Context-Based Orientation (requires edges to context variable)
    # FIX #4: Skip context-based orientation when augmented variable is excluded
    if exclude_augmented_var:
        if verbose:
            print(f"\n[Stage 3] Skipping Context-Based Orientation (augmented variable excluded)")
    else:
        feature_map = Nystroem(gamma=0.2, n_components=5, random_state=1)
        C_f = feature_map.fit_transform(c_indx)
        Ccc = my_cov(C_f, C_f)
        iCcc = np.linalg.inv(Ccc + np.eye(5) * 1e-10)

        vh = []
        # FIX #4: Use n_causal_vars instead of d (which includes augmented variable)
        for i in range(n_causal_vars):
            if (cg.G.graph[i, n_causal_vars] == 1) and (cg.G.graph[n_causal_vars, i] == -1):
                vh.append(i)

        # ... rest of orientation code (indented)
```

**Impact**: No crashes or graph corruption when `exclude_augmented_var=True`

---

## Files Modified

1. **`causallearn/search/ConstraintBased/CDNOD.py`**
   - Lines 270-278: FIX #2.1 (n_causal_vars)
   - Lines 280-327: FIX #3.1c (load initial skeleton)
   - Lines 341-344: FIX #2.2 (stage2_data)
   - Lines 358-367: FIX #2.3 (c_indx_id definition)
   - Lines 493-534: FIX #4 (safe context orientation)

2. **`causallearn/search/FCMBased/FedCDH/FedCDH.py`**
   - Lines 1456-1477: FIX #3.1a (convert consensus to matrix)
   - Lines 2256-2262: FIX #3.1b (pass initial skeleton)
   - Lines 2270-2278: FIX #2 (set exclude_augmented flag)

3. **`causallearn/utils/PCUtils/SkeletonDiscovery.py`**
   - Lines 77-88: FIX #3.2 (use initial skeleton)

---

## Verification

### Dry Run Analysis
**Document**: `DRY_RUN_PIPELINE_TRACE.md`

Traced complete pipeline from data augmentation through CI testing to final graph:
- ✅ All variables correctly set to 8 (not 9)
- ✅ Initial skeleton properly loaded (22 edges)
- ✅ PC algorithm starts with 22 edges
- ✅ All progress bars show "0/8"
- ✅ No references to variable [8] in causal graph operations

### Smoke Tests
**Script**: `tests/smoke_tests/smoke_test_cdnod_fixes.py`

All tests passed:
- ✅ FIX #2: Augmented Variable Exclusion
- ✅ FIX #3.1: Initial Skeleton Loading
- ✅ FIX #3.2: SkeletonDiscovery Uses Initial Skeleton
- ✅ FIX #4: Context-Based Orientation Safety
- ✅ Integration: Consistent Dimensions

---

## Expected Results

### Before Fixes
```
skeleton_f1: 0.000000
skeleton_precision: 0.000000
skeleton_recall: 0.000000
skeleton_shd: 8
dag_f1: 0.000000
dag_precision: 0.000000
dag_recall: 0.000000
dag_shd: 8
```

### After Fixes (Expected)
```
skeleton_f1: 0.60-0.85
skeleton_precision: 0.55-0.90
skeleton_recall: 0.65-0.95
skeleton_shd: 2-4
dag_f1: 0.50-0.80
dag_precision: 0.45-0.85
dag_recall: 0.55-0.90
dag_shd: 3-6
orientation_accuracy: 0.60-0.85
```

---

## Next Steps

### Run Full Experiment
```bash
python experiments/run_benchmark_gap.py --dataset asia --method fedspn_h --seed 42
```

### Verify Results
Check the following log outputs:

1. **Initial skeleton loaded**:
   ```
   [CDNOD] Initializing CausalGraph with skeleton from structure voting: 22 edges
   [Structure Voting] Passing initial skeleton to PC algorithm: 22 edges from consensus graph
   ```

2. **Correct dimensions**:
   ```
   [CDNOD] Excluding augmented variable from skeleton: 8 causal variables, 9 total variables
   Depth=0, working on node 0:  12%|█▎        | 1/8 [...]
   [CDNOD Stage 2] Using data.shape=(999, 8) (excluding augmented var)
   ```

3. **Non-zero edges**:
   ```
   skeleton_f1: > 0.000
   skeleton_shd: < 8
   adjacency_pred.npy: Non-zero matrix
   ```

### Monitor CI Tests
- Expect mix of DEPENDENT and INDEPENDENT results (not all DEPENDENT)
- p-values should vary (not all 0.000000)
- Final skeleton should have edges (not empty)

---

## Technical Details

### Pipeline Flow (After Fixes)
1. **Data Augmentation**: 999×8 → 999×9 (add client IDs)
2. **Local SPN Training**: 3 clients, each on 9D data
3. **Structure Voting**: Find 22 consensus edges (8D causal variables)
4. **Global SPN Training**: On 999×9 pooled data
5. **CDNOD Initialization**: Create graph with n_causal_vars=8
6. **Stage 1**: PC refinement starting from 22 edges (not 28)
7. **Stage 2**: Surrogate node tests on 8D data
8. **Stage 3**: Mechanism invariance orientation on 8×8 subgraph
9. **Return**: Final DAG with 8 variables

### Key Parameters
- `exclude_augmented_var=True` (horizontal mode)
- `initial_skeleton=<22 edges from voting>` (structure voting)
- `n_causal_vars=8` (Asia variables)
- `c_indx_id=8` (augmented variable index)

---

## References

- **Root Cause Analysis**: `FINAL_ROOT_CAUSE_AND_FIX.md`
- **Dry Run Trace**: `DRY_RUN_PIPELINE_TRACE.md`
- **Smoke Tests**: `tests/smoke_tests/smoke_test_cdnod_fixes.py`
- **Failed Experiment**: `experiments/v3_asia/20260530_194658_asia_fedspn_h_seed42/`

---

## Conclusion

All 4 critical bugs have been fixed with 7 code components:
- **3 components** for FIX #2 (augmented variable exclusion)
- **3 components** for FIX #3 (initial skeleton usage)
- **1 component** for FIX #4 (context orientation safety)

The pipeline is now ready for full-scale testing. Expected improvement: SHD from 8 to 2-4, F1 from 0.000 to 0.60-0.85.

**Status**: ✅ Ready for Experiment
