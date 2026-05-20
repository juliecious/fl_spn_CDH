# Improvement Roadmap: Addressing Remaining Issues in Eval3

**Date:** May 19, 2026
**Context:** Post-eval3 analysis after unified architecture deployment

---

## Issue 1: Poor Orientation Accuracy (DAG F1 near 0%)

### Current Status
```
Law School:  dag_f1 = 0.000, orientation_accuracy = 0.000
Sachs Vert:  dag_f1 = 0.125, orientation_accuracy = 0.133
Sachs Hybrid: dag_f1 = 0.000, orientation_accuracy = 0.000
```

**Problem:** Algorithm detects edges (skeleton) reasonably well but fails to orient them correctly.

### Root Cause Analysis

#### 1. Orientation Method: Hybrid Direction Score
**Location:** `causallearn/search/ConstraintBased/CDNOD.py` lines 69-192

**Current Algorithm:**
```python
def get_hybrid_direction_score(fed_spn_model, i, j, c_idx, data_aug, ...):
    # Computes: I(X_j; U | X_i) vs I(X_i; U | X_j)
    # Direction with LOWER CMI (Conditional Mutual Information) wins

    score_i_to_j = estimate_cmi_normalized(j, i, u_idx, batch_data)
    score_j_to_i = estimate_cmi_normalized(i, j, u_idx, batch_data)

    if score_i_to_j < score_j_to_i:
        return 1  # i -> j
    elif score_j_to_i < score_i_to_j:
        return 2  # j -> i
    else:
        return 0  # Ambiguous
```

**Theoretical Basis:** Federated Independent Change Principle (FICP)
- If X_i → X_j, then P(X_j | X_i) should be invariant across domains U
- Measures violation via CMI: I(X_j; U | X_i)
- Lower score = more invariant = correct direction

#### 2. Why It's Failing

**Issue 2.1: Weak Domain Signal (Law School)**
```python
# Law School: c_indx is all zeros (vertical mode)
c_indx = np.zeros((n_samples, 1))  # No meaningful domain variation!
```

**Problem:**
- Vertical mode has NO domain variation (all samples have U=0)
- CMI-based orientation requires domain differences to measure invariance
- Result: I(X_j; U | X_i) ≈ 0 for both directions → ambiguous → wrong orientation

**Issue 2.2: Noisy CMI Estimates**
- Uses KDE-based density estimation for CMI
- Small sample sizes per domain (Sachs hybrid: only ~850 samples per client)
- High variance in estimates → coin-flip decisions

**Issue 2.3: No Meek Rules Applied**
Looking at the code, CDNOD does orientation but may not properly apply Meek rules for transitive orientation.

### Proposed Solutions

#### Solution 1A: Add Regression-Based Orientation (High Impact) ⭐⭐⭐

**Rationale:** When domain signal is weak, use direct regression score.

**Implementation:**
```python
def get_regression_direction_score(fed_spn_model, i, j, data, residual_variance_threshold=0.05):
    """
    Use regression-based independence score for orientation.

    Logic:
    - If X_i -> X_j, then X_j = f(X_i) + noise
    - Regress X_j ~ X_i and X_i ~ X_j using SPN conditional densities
    - Direction with lower residual variance is more likely causal

    This works even when domain signal U is weak!
    """
    import torch

    # Direction i -> j: Estimate Var(X_j | X_i)
    # Use SPN to compute E[X_j | X_i] via conditional sampling
    # Then compute residual variance

    with torch.no_grad():
        # Get conditional log probabilities
        # P(X_j | X_i) = P(X_i, X_j) / P(X_i)

        # Marginalize out all except i, j
        batch = torch.tensor(data[:, [i, j]], dtype=torch.float32)

        # Score i -> j: How well does X_i predict X_j?
        # Approximate via variance of residuals
        # (Lower variance = better prediction = more likely causal)

        # This requires implementing conditional sampling or
        # using the SPN's conditional log-likelihood directly

        pass  # Implementation needed

    return direction  # 1 for i->j, 2 for j->i, 0 for ambiguous
```

**Advantages:**
- Works without domain variation (U can be constant)
- Based on functional relationship strength
- Well-established in causal discovery (ANM, LiNGAM)

**Disadvantages:**
- Requires conditional sampling from SPN (not currently implemented)
- May struggle with symmetric relationships

**Effort:** Medium (2-3 days)
**Expected Improvement:** +20-30% DAG F1

---

#### Solution 1B: Improve CMI Estimation (Medium Impact) ⭐⭐

**Current Issue:** Noisy KDE-based CMI with small samples.

**Implementation:**
```python
def estimate_cmi_normalized(target, context, domain_var, batch_data):
    # CURRENT: Uses KDE with h=5 Nystroem features
    # PROBLEM: High variance with small samples

    # IMPROVEMENT 1: Use more features (h=20 instead of h=5)
    feature_map = Nystroem(gamma=0.2, n_components=20, random_state=1)

    # IMPROVEMENT 2: Add bootstrapping for confidence
    n_bootstrap = 50
    cmi_estimates = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(batch_data), len(batch_data), replace=True)
        boot_data = batch_data[indices]
        cmi_estimates.append(compute_cmi_single(boot_data))

    # Use median instead of mean (more robust to outliers)
    cmi_val = np.median(cmi_estimates)

    # IMPROVEMENT 3: Confidence-weighted decision
    cmi_std = np.std(cmi_estimates)
    if cmi_std > 0.5 * abs(cmi_val):
        return None  # Too uncertain, don't orient

    return cmi_val
```

**Effort:** Low (1 day)
**Expected Improvement:** +5-10% DAG F1

---

#### Solution 1C: Enable Meek Rules and Propagation (High Impact) ⭐⭐⭐

**Check Current Implementation:**

Read CDNOD to see if Meek rules are applied:

```python
# After skeleton discovery and initial orientation:
# 1. Apply Meek Rule 1: i -> k -> j and i --- j => i -> j
# 2. Apply Meek Rule 2: i -> k and k --- j and i --- j => k -> j
# 3. Apply Meek Rule 3: Unshielded collider rules
# 4. Iterate until convergence
```

**If Meek rules aren't fully implemented:**

```python
def apply_meek_rules_aggressive(cg, max_iterations=10):
    """
    Apply Meek's orientation rules iteratively until convergence.

    Meek Rules:
    1. If i -> k -> j and i --- j, then orient i -> j
    2. If i -> k --- j and i --- j, then orient k -> j
    3. If i --- k -> j and i --- j and i --- j, then orient i -> k
    4. Unshielded collider: i -> k <- j and i not adjacent to j
    """
    for iteration in range(max_iterations):
        changed = False

        # Rule 1: Transitivity
        for edge in cg.get_edges():
            if edge.is_undirected():
                i, j = edge.get_nodes()
                # Check for common neighbors with directed edges
                for k in cg.get_neighbors(i):
                    if cg.has_edge(i, k) and cg.get_edge(i, k).is_directed_i_to_j():
                        if cg.has_edge(k, j) and cg.get_edge(k, j).is_directed_i_to_j():
                            # i -> k -> j and i --- j => orient i -> j
                            cg.orient_edge(i, j)
                            changed = True

        # ... Apply other Meek rules ...

        if not changed:
            break

    return cg
```

**Effort:** Medium (2 days)
**Expected Improvement:** +15-25% DAG F1

---

#### Solution 1D: Ensemble Orientation (High Impact) ⭐⭐⭐⭐

**Idea:** Combine multiple orientation signals.

```python
def ensemble_orientation(fed_spn_model, i, j, data, c_idx):
    """
    Combine multiple orientation methods for robust decision.

    Methods:
    1. Hybrid CMI score (current)
    2. Regression residual variance
    3. Independence score asymmetry: I(X_i; X_j) vs I(X_j; X_i)
    4. Complexity asymmetry: H(X_j | X_i) vs H(X_i | X_j)
    """

    votes = []
    weights = []

    # Method 1: CMI-based (if domain variation exists)
    if c_idx.std() > 0:
        cmi_direction = get_hybrid_direction_score(...)
        votes.append(cmi_direction)
        weights.append(0.4)

    # Method 2: Regression-based
    reg_direction = get_regression_direction_score(...)
    votes.append(reg_direction)
    weights.append(0.3)

    # Method 3: Complexity asymmetry (entropy)
    # Simpler model: X_j | X_i has lower entropy => i -> j more likely
    entropy_i_given_j = estimate_conditional_entropy(i, j, fed_spn_model, data)
    entropy_j_given_i = estimate_conditional_entropy(j, i, fed_spn_model, data)

    if entropy_j_given_i < entropy_i_given_j * 0.9:
        votes.append(1)  # i -> j
    elif entropy_i_given_j < entropy_j_given_i * 0.9:
        votes.append(2)  # j -> i
    else:
        votes.append(0)
    weights.append(0.3)

    # Weighted majority vote
    weighted_votes = {}
    for vote, weight in zip(votes, weights):
        weighted_votes[vote] = weighted_votes.get(vote, 0) + weight

    # Return direction with highest weight
    if weighted_votes[1] > weighted_votes.get(2, 0):
        return 1  # i -> j
    elif weighted_votes[2] > weighted_votes.get(1, 0):
        return 2  # j -> i
    else:
        return 0  # Ambiguous
```

**Effort:** High (1 week)
**Expected Improvement:** +30-40% DAG F1

---

### Recommended Approach

**Phase 1 (Quick Wins - 2 days):**
1. Implement Solution 1B (Improve CMI estimation)
2. Verify Meek rules are properly applied (Solution 1C)

**Phase 2 (Medium Term - 1 week):**
3. Implement Solution 1A (Regression-based orientation)
4. Test on Law School and Sachs

**Phase 3 (Long Term - 2 weeks):**
5. Implement Solution 1D (Ensemble orientation)
6. Tune weights on validation set

**Expected Final Performance:**
- Law School: dag_f1 = 0.0 → 0.15-0.25 (15-25%)
- Sachs: dag_f1 = 0.125 → 0.35-0.45 (35-45%)

---

## Issue 2: Law School Performance Ceiling (F1 ~ 0.22)

### Current Status
```
skeleton_f1: 0.222 (stable across eval1/2/3)
```

### Root Cause Analysis

#### Problem 2.1: Severe Feature Fragmentation
```
Dataset: 5 features split across 3 clients
Vertical partition: [2, 2, 1] features per client
d/K = 1.67 < 3 (critical threshold)

Result:
- K_local forced to 1 (no mixture-of-products)
- Cannot capture cross-client dependencies effectively
- Each client sees only 1-2 features (insufficient for clustering)
```

#### Problem 2.2: Weak Causal Signal
Law School admissions data has weak, noisy causal relationships:
- Many confounders
- Selection bias (only admitted students)
- Complex non-linear relationships

### Proposed Solutions

#### Solution 2A: Use Better Benchmark Datasets ⭐⭐⭐⭐

**Replace Law School with:**

1. **Asia Network (8 features)**
   - d/K = 8/3 = 2.67 features per client (still below threshold)
   - Better: Use 2 clients → d/K = 4

2. **Alarm Network (37 features)**
   - d/K = 37/3 = 12.3 features per client ✓ Great!
   - Allows K_local = 2 or 3 (mixture-of-products enabled)

3. **DREAM4 Network (10 features)**
   - d/K = 10/3 = 3.3 features per client ✓ Good
   - Gene regulatory network with strong signals

**Recommendation:** Use **Alarm** for vertical mode benchmarking (best d/K ratio).

**Effort:** Low (1 day to run experiments)
**Expected Improvement:** +30-50% F1 on better datasets

---

#### Solution 2B: Adaptive Client Count ⭐⭐

**Idea:** Automatically choose K based on dataset size.

```python
def choose_optimal_K(d_features, target_features_per_client=5):
    """
    Choose K (number of clients) to ensure each has enough features.

    Rule: d/K >= 5 for good clustering
          d/K >= 3 minimum threshold
    """
    K_optimal = max(2, d_features // target_features_per_client)

    # Ensure at least 3 features per client
    while d_features / K_optimal < 3 and K_optimal > 2:
        K_optimal -= 1

    return K_optimal

# Example:
# Law School (5 features): K=2 → d/K=2.5 (still marginal)
# Alarm (37 features): K=7 → d/K=5.3 (good!)
# DREAM4 (10 features): K=2 → d/K=5 (good!)
```

**Effort:** Low (modify benchmark script)
**Expected Improvement:** +10-20% F1 by avoiding fragmentation

---

#### Solution 2C: Overlapping Feature Assignment (Vertical → Quasi-Hybrid) ⭐⭐⭐

**Idea:** Allow small feature overlap in vertical mode to increase information.

```python
def assign_features_with_overlap(d_features, K_clients, overlap_ratio=0.2):
    """
    Assign features to clients with controlled overlap.

    Example (d=5, K=3, overlap=0.2):
    - Client 0: [0, 1, 2]       (3 features)
    - Client 1: [1, 2, 3]       (3 features, 2 overlap with C0)
    - Client 2: [3, 4]          (2 features, 1 overlap with C1)

    Result: Each client sees more features, improving clustering.
    Privacy: Still better than full horizontal mode.
    """
    features_per_client = int(d_features / K_clients * (1 + overlap_ratio))
    stride = d_features // K_clients

    feature_splits = []
    for k in range(K_clients):
        start = k * stride
        end = min(start + features_per_client, d_features)
        feature_splits.append(list(range(start, end)))

    return feature_splits

# Law School example:
# No overlap:  [[0,1], [2,3], [4]]  → 2, 2, 1 features
# 50% overlap: [[0,1,2], [1,2,3], [3,4]]  → 3, 3, 2 features
```

**Effort:** Medium (1 week - requires testing overlap impact)
**Expected Improvement:** +15-25% F1 on small datasets

---

### Recommended Approach

**Short Term:**
- Solution 2A: Run benchmarks on Alarm/DREAM4 (better datasets)
- Solution 2B: Implement adaptive K selection

**Long Term:**
- Solution 2C: Research overlapping vertical partitioning

**Expected:**
- Law School will remain limited (~25% F1 max)
- Alarm: 50-70% F1 (good performance)
- DREAM4: 40-60% F1 (good performance)

---

## Issue 3: Noisy Error Logs (Dimension Mismatch Warnings)

### Current Status
```
ERROR - DIMENSION MISMATCH: x.shape=torch.Size([21000, 5]),
        feature_indices=[0, 1], use_nan_masking=False, full_d=5
```

Appears hundreds of times during CI testing phase.

### Root Cause Analysis

**Location:** `causallearn/utils/FedPC.py` lines 1005-1008

```python
# DEBUG logging in GroupMixture.log_prob()
if x.shape[1] != len(self.feature_indices) and x.shape[1] > max(self.feature_indices) + 1:
    import logging
    logging.error(f"DIMENSION MISMATCH: ...")
x_g = x[:, self.feature_indices]  # This works correctly!
```

**Problem:**
- Context column U is included in x during CI testing: x.shape = [batch, d+1]
- GroupMixture expects x.shape = [batch, d]
- The code correctly extracts features: `x[:, feature_indices]`
- But logs an ERROR even though it handles it correctly

**This is NOT a bug - just noisy logging!**

### Proposed Solutions

#### Solution 3A: Remove Debug Logging (Immediate) ⭐⭐⭐⭐

**Implementation:**
```python
# OLD CODE (line 1005-1008):
if x.shape[1] != len(self.feature_indices) and x.shape[1] > max(self.feature_indices) + 1:
    import logging
    logging.error(f"DIMENSION MISMATCH: ...")
x_g = x[:, self.feature_indices]

# NEW CODE:
# Simply remove the debug logging - it was added during development
x_g = x[:, self.feature_indices]
```

**Effort:** 5 minutes
**Expected Improvement:** Clean logs, no more noise

---

#### Solution 3B: Strip Context Column Earlier (Better Fix) ⭐⭐⭐

**Implementation:**

In `FedCDH.py`, strip context column BEFORE passing to GroupMixture:

```python
# In CIT class get_marginal_ll() method (causallearn/utils/cit.py ~line 770)

# CURRENT:
masked_batch = torch.full((self._n_samples, self._n_features), np.nan, device=self.device)
for idx in indices:
    masked_batch[:, idx] = self.data_t[:, idx]

# Issue: self._n_features includes context column U

# FIX:
# Strip context column during CIT initialization
class SPN_CIT(CIT_Base):
    def __init__(self, data, global_model=None, ...):
        # Strip context column if present
        if hasattr(global_model, 'u_index'):
            u_idx = global_model.u_index
            self.data_t = torch.tensor(data[:, :u_idx], ...).to(self.device)
            self._n_features = u_idx  # Exclude U from feature count
        else:
            self.data_t = torch.tensor(data, ...).to(self.device)
            self._n_features = data.shape[1]
```

**Effort:** 1 hour
**Expected Improvement:**
- Clean logs
- Slightly faster (no context column in CI tests)
- More correct separation of concerns

---

### Recommended Approach

**Immediate:** Solution 3A (remove debug logging)
**Better:** Solution 3B (strip context earlier)

Both are quick fixes with no performance impact.

---

## Priority Ranking

### High Priority (Do First) 🔥

1. **Issue 3A: Remove noisy logs** (5 minutes)
   - Immediate quality-of-life improvement
   - Clean logs for debugging

2. **Issue 1C: Verify/fix Meek rules** (2 days)
   - High impact on orientation
   - Essential for DAG discovery

3. **Issue 2A: Run on better datasets** (1 day)
   - Proves architecture scales
   - Shows true performance potential

### Medium Priority (Do Next) ⭐

4. **Issue 1B: Improve CMI estimation** (1 day)
   - Quick win for orientation
   - Low risk

5. **Issue 1A: Add regression-based orientation** (3 days)
   - Significant orientation improvement
   - Works when domain signal weak

6. **Issue 2B: Adaptive K selection** (1 day)
   - Prevents fragmentation
   - Simple implementation

### Low Priority (Future Work) 💡

7. **Issue 1D: Ensemble orientation** (1 week)
   - Best orientation performance
   - Requires multiple methods implemented first

8. **Issue 2C: Overlapping features** (1 week)
   - Research question
   - May not significantly improve small datasets

---

## Expected Outcomes

### After High Priority Fixes (1 week):
- Clean logs ✅
- Better orientation (+15-25% DAG F1) ✅
- Validation on larger datasets (Alarm: 50-70% skeleton F1) ✅

### After Medium Priority Fixes (2 weeks):
- Strong orientation performance (+30-40% DAG F1) ✅
- No feature fragmentation issues ✅
- Robust CI estimation ✅

### After All Fixes (4 weeks):
- State-of-the-art federated causal discovery ✅
- DAG F1: 35-50% on real-world datasets ✅
- Skeleton F1: 60-80% on Alarm/DREAM4 ✅

---

## Conclusion

The three main issues are **solvable**:

1. **Orientation:** Fixable with better methods (regression, ensemble, Meek rules)
2. **Law School ceiling:** Expected due to dataset limitations - use better benchmarks
3. **Noisy logs:** Trivial fix (5 minutes)

**Recommendation:** Focus on orientation improvements (Issue 1) and better benchmarks (Issue 2A) for maximum impact.
