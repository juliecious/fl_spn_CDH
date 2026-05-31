# Analysis: Why 24 Edges Instead of 8?

**Experiment**: 20260530_205318_asia_fedspn_h_seed42
**Problem**: Too many edges (24 predicted vs 8 true)
**Status**: Pipeline working, but CI test is too conservative

---

## Summary of the Problem

| Metric | Value | Assessment |
|--------|-------|------------|
| Initial skeleton | 22 edges | ✅ From structure voting |
| Final skeleton | 24 edges | ❌ Should be ~8-12 |
| True edges | 8 edges | Ground truth |
| False positives | 16 edges | Too many! |
| True positives | 8 edges | ✅ Perfect recall |

**The pipeline ADDED 2 edges** during Stage 1 and Stage 2, instead of removing false positives.

---

## Detailed Analysis

### Step 1: Initial Skeleton (Structure Voting)

```
Structure voting found: 22 edges
  - 8 true positive edges (correct)
  - 14 false positive edges (noise)
  - Confidence: ~94%
```

**This is the starting point for PC algorithm.**

### Step 2: Stage 1 - Skeleton Discovery (PC Algorithm)

**Goal**: Refine the 22-edge skeleton by removing false positives

**What Actually Happened**:

```
Depth 0: Test all 22 edges with Z=[]
  Result: ALL p-values = 0.000 → ALL DEPENDENT → NO edges removed

Depth 1: Test all 22 edges with Z=[single variable]
  Result: ALL p-values = 0.000 → ALL DEPENDENT → NO edges removed

Depth 2: Test all 22 edges with Z=[two variables]
  Result: ALL p-values = 0.000 → ALL DEPENDENT → NO edges removed

Depth 3: Test all 22 edges with Z=[three variables]
  Result: ALL p-values = 0.000 → ALL DEPENDENT → NO edges removed
  Reached depth_limit=3, stopping
```

**CI Test Statistics**:
- Total tests: ~2400
- DEPENDENT: 2383 (99.3%)
- INDEPENDENT: 17 (0.7%)

**Result**: PC algorithm kept all 22 edges (no refinement happened)

### Step 3: Stage 2 - Surrogate Node

```
[CDNOD Stage 2] Using data.shape=(999, 8) (excluding augmented var)

Stage 2 also tests with Z=[]
  Result: ALL p-values = 0.000 → ALL DEPENDENT
```

**Stage 2 can ADD edges** if it finds new dependencies via the surrogate node mechanism.

**Result**: Added 2 edges, bringing total to 24

---

## Why All P-Values = 0.000?

### The Root Cause

**The SPN-based CI test computes marginals that include influence from the augmented variable (client ID).**

#### What's Happening:

1. **GlobalSPN trained on 9D data**: P(X₀, X₁, ..., X₇, U)
   - U = client ID (which client owns the sample)
   - Variables correlated within each client

2. **CI test at Depth 0 with Z=[]**: Test X_i ⊥ X_j | []
   ```python
   # Compute marginal P(X_i, X_j) by marginalizing over U
   P(X_i, X_j) = Σ_k P(X_i, X_j | U=k) · P(U=k)
   ```

3. **Problem**: Marginal includes U's confounding effect
   - Even if X_i ⊥ X_j | U (conditionally independent given client)
   - Marginal X_i ⊥̸ X_j (marginally dependent through client membership)

#### Example:

```
Suppose in reality:
  - Within Client 0: X₀ ⊥ X₁ | U=0 (independent)
  - Within Client 1: X₀ ⊥ X₁ | U=1 (independent)
  - Within Client 2: X₀ ⊥ X₁ | U=2 (independent)

But marginally (averaging over U):
  - X₀ ⊥̸ X₁ (appear dependent because client membership creates correlation)

Result: CI test with Z=[] returns p=0.000 (DEPENDENT)
```

### Why Structure Voting Worked

**Structure voting explicitly conditions on U**:

```python
# Structure voting tests: X_i ⊥ X_j | U
Z = [8]  # Always include augmented variable

# This properly controls for client membership
P(X_i, X_j | U=k) = GlobalSPN([x_i, x_j, k]) / GlobalSPN([k])
```

**Result**: Accurate p-values (varied: 0.02, 0.05, 0.10, 0.12, etc.)

---

## Comparison: Structure Voting vs Main PC

### Structure Voting CI Tests (Z=[8])

```
Sample results from test.log:

[DEBUG CI Test #0] X=[0], Y=[1], Z=[8]
  p_value=0.098039 → INDEPENDENT ✓

[DEBUG CI Test #3] X=[0], Y=[4], Z=[8]
  p_value=0.117647 → INDEPENDENT ✓

[DEBUG CI Test #1] X=[0], Y=[2], Z=[8]
  p_value=0.039216 → DEPENDENT ✓

[DEBUG CI Test #2] X=[0], Y=[3], Z=[8]
  p_value=0.019608 → DEPENDENT ✓
```

**Observations**:
- P-values are varied (not all 0.000)
- Some INDEPENDENT, some DEPENDENT
- Tests are working correctly
- Result: 22 edges with reasonable quality

---

### Main PC CI Tests (Z=[])

```
Sample results from test.log:

Depth=0:
[DEBUG CI Test #0] X=[0], Y=[2], Z=[]
  p_value=0.000000 → DEPENDENT ✗

[DEBUG CI Test #1] X=[0], Y=[3], Z=[]
  p_value=0.000000 → DEPENDENT ✗

[DEBUG CI Test #2] X=[0], Y=[4], Z=[]
  p_value=0.000000 → DEPENDENT ✗

... (all tests show p=0.000)
```

**Observations**:
- ALL p-values = 0.000
- Everything appears DEPENDENT
- Tests are NOT working correctly
- Result: No edges removed, kept all 22 (even added 2 more)

---

## The False Positives

### 16 False Positive Edges:

```
(0, 3), (0, 4), (0, 5), (0, 6), (0, 7)  # 5 edges from node 0
(1, 5), (1, 6), (1, 7)                  # 3 edges from node 1
(2, 3), (2, 7)                          # 2 edges from node 2
(3, 4), (3, 6), (3, 7)                  # 3 edges from node 3
(4, 5), (4, 6)                          # 2 edges from node 4
(6, 7)                                  # 1 edge from node 6
```

**Pattern**: Node 3 appears as a hub (many false edges connected to it)

### Why These Should Have Been Removed

If the CI test was working correctly:
- Some of these pairs should test as INDEPENDENT with proper conditioning
- Example: (0, 3) has no causal path in truth → should be removable
- With Z=[] (empty), they all appear DEPENDENT due to U confounding
- With Z containing U, they would likely test as INDEPENDENT

---

## Quantitative Evidence

### CI Test Effectiveness

| Conditioning | P-values | INDEPENDENT Rate | Edge Removal | Quality |
|--------------|----------|------------------|--------------|---------|
| **Structure Voting** (Z=[8]) | Varied | ~30% | Good selection | ✅ High |
| **Main PC** (Z=[]) | All 0.000 | 0.7% | None | ❌ Low |

### Expected vs Actual

| Stage | Expected | Actual | Problem |
|-------|----------|--------|---------|
| Structure Voting | ~20-25 edges | 22 edges | ✅ Good |
| PC Stage 1 | Refine to ~10-15 | Kept 22 | ❌ No refinement |
| PC Stage 2 | Refine to ~8-12 | Added to 24 | ❌ Made worse |
| Final | 8-12 edges | 24 edges | ❌ 3× too many |

---

## The Solution

### Recommended Fix: Condition on U in Main PC

**Modify the PC algorithm to always include augmented variable in conditioning sets** (just like structure voting does).

#### Current Approach (Broken):
```python
# Depth 0: Z = []
# Depth 1: Z = [single neighbor]
# Depth 2: Z = [two neighbors]
# ...

# Problem: Marginals over U are confounded
```

#### Proposed Approach (Fixed):
```python
# Depth 0: Z = [U]  (condition on augmented variable)
# Depth 1: Z = [U, single neighbor]
# Depth 2: Z = [U, two neighbors]
# ...

# Solution: Always control for client membership
```

### Implementation Location

**File**: `causallearn/utils/PCUtils/SkeletonDiscovery.py`

**Modify**: Line ~100-150 where conditioning sets are constructed

**Change**:
```python
# Current:
for S in combinations(neighbors, depth):
    ci_test(i, j, list(S))

# Proposed:
for S in combinations(neighbors, depth):
    S_with_U = list(S) + [c_indx_id]  # Always include U
    ci_test(i, j, S_with_U)
```

### Expected Impact

With this fix:

| Metric | Current | After Fix | Expected Improvement |
|--------|---------|-----------|---------------------|
| skeleton_precision | 0.333 | 0.60-0.75 | ✅ +80-125% |
| skeleton_f1 | 0.500 | 0.70-0.85 | ✅ +40-70% |
| skeleton_shd | 16 | 3-6 | ✅ -63-81% |
| Final edges | 24 | 10-14 | ✅ -42-58% |
| False positives | 16 | 2-6 | ✅ -63-88% |

**P-value distribution**: Will be varied (not all 0.000), similar to structure voting

---

## Alternative Solutions

### Option 2: Train Separate 8D Global SPN

**Approach**: Train global SPN on 8D data (without U) for main PC

**Pros**:
- Marginals won't be confounded by U
- More principled separation

**Cons**:
- Requires significant refactoring
- Need two global SPNs (9D for structure voting, 8D for main PC)
- More computational cost

### Option 3: Use Structure Voting Result Directly

**Approach**: Skip PC refinement, use structure voting's 22 edges directly

**Pros**:
- Simple, no code changes
- Already working reasonably well

**Cons**:
- Still has 14 false positives
- Missing the benefit of PC refinement
- Not utilizing the full pipeline

---

## Conclusion

**Root Cause**: CI test computes marginals that marginalize over U (client ID), making all variables appear dependent through confounding.

**Evidence**:
- Structure voting (Z=[8]): Works correctly, varied p-values
- Main PC (Z=[]): Broken, all p-values = 0.000
- Result: No edges removed, even added 2 more

**Solution**: Condition on U in main PC (same as structure voting)

**Expected Outcome**:
- Varied p-values
- Proper edge removal
- Final skeleton: ~10-14 edges (closer to 8 true)
- Precision: 0.33 → 0.65+
- F1: 0.50 → 0.75+

**Next Step**: Implement conditioning on U in `SkeletonDiscovery.py`
