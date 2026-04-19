# V2 Experiment Proposal: Adaptive Architecture + Verification

**Date**: April 19, 2026
**Branch**: fedpc
**Purpose**: Address v1 baseline failures through adaptive hyperparameters and systematic verification

---

## Executive Summary

**V1 Critical Failures**:
- MEDIUM/LARGE Horizontal: F1=0.0 (predicts zero edges)
- Root cause: Fixed architecture (20/20) insufficient for d×K > 30
- Paradox: High accuracy (35-61%) but F1=0 due to "predict nothing" strategy

**V2 Objectives**:
1. Fix catastrophic horizontal mode failures
2. Verify SPN quality drives causal discovery performance
3. Validate adaptive hyperparameter system
4. Establish baseline for realistic problem sizes

---

## Root Cause Analysis Summary

### The "Predict Nothing" Paradox

**MEDIUM Horizontal** (F1=0.0):
```
Confusion: TP=0, FP=0, FN=37, TN=58
Overall Accuracy: 58/95 = 0.611 (61%!)
F1: 0.000 (complete failure)
```

**Explanation**:
- Model predicts ZERO edges exist
- Gets credit for all true non-edges (TN=58)
- In sparse graphs (40% edges), predicting "no edges" gives 60% accuracy
- But F1=0 because precision and recall are both 0

**Root Cause Chain**:
```
Fixed num_sums=20, num_leaves=20
→ Poor density estimation (MMD p=0.000)
→ Unreliable conditional independence tests
→ Conservative p-values (fail to reject independence)
→ Predict "everything independent"
→ Zero edges detected
```

### Why SMALL Works
- Complexity d×K=24 within capacity of 20/20
- Better SPN quality → reliable CI tests
- Can actually detect dependencies

### Why Vertical Survives
- Low dimensionality per client (2-4 features)
- Large sample size (full dataset)
- Simpler modeling problem

---

## V2 Experiment Design

### Phase 1: Verification Experiments (Week 1)

**Goal**: Validate root cause hypotheses before full v2 run

#### Experiment 1.1: Oracle Density Test
**Purpose**: Verify SPN quality is the bottleneck

**Setup**:
- Use MEDIUM config (d=10, K=3, n=1200)
- Generate data from known Gaussian SEM
- Run TWO tests:
  1. **Learned SPN** (current): Fixed 20/20 architecture
  2. **Oracle**: Use true covariance matrix for CI tests

**Implementation**:
```python
def oracle_ci_test(data, i, j, cond_set, true_dag, true_noise_scales):
    """CI test using ground truth Gaussian distribution."""
    # Use known B (graph weights) and noise scales
    # Compute partial correlation ρ_ij | cond_set analytically
    # Test if ρ = 0
    from causallearn.utils.cit import fisherz
    return fisherz(data, i, j, cond_set)
```

**Expected Results**:
| Method | F1 | Interpretation |
|--------|----|----|
| Learned SPN (20/20) | 0.000 | Current failure |
| Oracle (true dist) | > 0.7 | Confirms SPN quality is bottleneck |

**Decision Criterion**: If oracle F1 > 0.7, proceed with capacity increase. If oracle F1 < 0.5, investigate CI test logic.

---

#### Experiment 1.2: Capacity Sweep
**Purpose**: Find minimum architecture for MEDIUM/LARGE success

**Setup**:
- MEDIUM config (d=10, K=3)
- Horizontal mode (most challenging)
- Sweep num_sums/num_leaves: [20, 40, 60, 80, 100]
- Fix other params: epochs=150, num_reps=10

**Configurations**:
```python
configs = [
    {'num_sums': 20, 'num_leaves': 20},   # Baseline (F1=0)
    {'num_sums': 40, 'num_leaves': 40},   # 2× capacity
    {'num_sums': 60, 'num_leaves': 60},   # 3× capacity (predicted min)
    {'num_sums': 80, 'num_leaves': 80},   # 4× capacity
    {'num_sums': 100, 'num_leaves': 100}, # 5× capacity
]
```

**Metrics to Track**:
- Overall F1 (primary)
- MMD p-value (SPN quality proxy)
- KS fail rate
- Train LL
- Confusion matrix (TP, FP, FN, TN)

**Expected Results**:
```
num_sums=20:  F1=0.000, MMD p=0.000 (current)
num_sums=40:  F1=0.2-0.3, MMD p~0.01
num_sums=60:  F1=0.5+, MMD p>0.05 (hypothesis: sufficient)
num_sums=80+: F1 plateaus or slight improvement
```

**Decision Criterion**: Identify minimum num_sums where:
1. F1 > 0.5
2. MMD p-value > 0.05
3. No TP=0 failures

---

#### Experiment 1.3: Alpha Sensitivity Test
**Purpose**: Check if threshold adjustment helps

**Setup**:
- MEDIUM Horizontal (worst case)
- Fixed architecture: num_sums=20 (baseline)
- Vary alpha: [0.01, 0.05, 0.10, 0.15]

**Expected Results**:
- Alpha=0.01: F1~0 (more conservative, worse)
- Alpha=0.05: F1=0 (current)
- Alpha=0.10: F1 might improve slightly but with FP increase
- Alpha=0.15: More edges but likely more false positives

**Hypothesis**: Alpha adjustment alone won't fix F1=0 if SPN quality is poor.

---

#### Experiment 1.4: Permutation Count Test
**Purpose**: Check if num_permutations=50 is sufficient

**Setup**:
- MEDIUM Horizontal
- Fixed 20/20 architecture
- Vary num_permutations: [50, 100, 200, 500]

**Expected Results**: Minimal change (hypothesis: SPN quality dominates)

---

### Phase 2: Full V2 Adaptive Experiments (Week 2-3)

**Goal**: Run complete experimental suite with adaptive hyperparameters

#### Configuration Matrix

Same as v1: 3 configs × 3 modes × 2 data types = 18 experiments

**Configs**:
- SMALL: d=8, K=3, n=600
- MEDIUM: d=10, K=3, n=1200
- LARGE: d=11, K=5, n=1650

**Modes**: horizontal, vertical, hybrid
**Data Types**: linear, nonlinear

#### Adaptive Hyperparameter Rules

**Horizontal Mode**:
```python
if d <= 8:
    num_sums = 32
    num_leaves = 16
elif d <= 10:
    num_sums = 60
    num_leaves = 30
else:  # d >= 11
    num_sums = 80
    num_leaves = 40

epochs = int(50 * (1.0 + d / 30))  # Scale with dimensionality
```

**Vertical Mode**:
```python
local_features = d // K

if local_features <= 3:
    num_sums = 10  # Reduce overfitting
    num_leaves = 10
else:
    num_sums = 8 * local_features
    num_leaves = 4 * local_features

epochs = int(50 * 0.75)  # Fewer epochs for simpler problem
```

**Hybrid Mode**:
```python
num_sums = 6 * d
num_leaves = 3 * d
epochs = int(50 * 1.2)
```

**Additional Settings**:
```python
gradient_clip = 1.0  # Prevent explosions
num_permutations = 100  # Double from v1
alpha = 0.05  # Keep standard
```

#### Expected V2 Results

| Config | Mode | V1 F1 | V2 F1 (Target) | Change |
|--------|------|-------|---------------|--------|
| SMALL | Horiz | 0.303 | 0.5+ | +66% |
| SMALL | Vert | 0.857 | 0.8+ | Maintain |
| SMALL | Hybrid | 0.776 | 0.8+ | +3% |
| MEDIUM | Horiz | 0.000 | **0.5+** | Fix! |
| MEDIUM | Vert | 0.261 | 0.4+ | +53% |
| MEDIUM | Hybrid | 0.585 | 0.7+ | +20% |
| LARGE | Horiz | 0.000 | **0.4+** | Fix! |
| LARGE | Vert | 0.532 | 0.5+ | Maintain |
| LARGE | Hybrid | 0.727 | 0.8+ | +10% |

**Success Criteria**:
1. ✅ No F1=0.0 failures (eliminate "predict nothing")
2. ✅ MMD p-value > 0.05 in >50% of experiments (vs 5% in v1)
3. ✅ Horizontal mode functional for all configs
4. ✅ F1 improvement for SMALL/MEDIUM (establish proper baseline)

---

### Phase 3: Advanced Improvements (Week 4)

#### 3.1: Early Stopping Based on Validation LL
```python
patience = 20
best_val_ll = -inf
no_improvement_count = 0

for epoch in range(max_epochs):
    train_ll = train_spn(...)
    val_ll = evaluate_spn(val_data)

    if val_ll > best_val_ll:
        best_val_ll = val_ll
        no_improvement_count = 0
        save_checkpoint()
    else:
        no_improvement_count += 1

    if no_improvement_count >= patience:
        break  # Stop early
```

#### 3.2: SPN Quality Rejection Threshold
```python
# After training SPN, check quality before using for CI
if mmd_pvalue < 0.01 or ks_fail_rate > 0.8:
    # Retrain with increased capacity
    num_sums *= 1.5
    num_leaves *= 1.5
    retrain_spn()
```

#### 3.3: Ensemble SPNs
```python
# Train multiple SPNs with different initializations
spns = [train_spn(seed=i) for i in range(5)]

# Use ensemble for CI test
def ensemble_ci_test(X, i, j, cond_set):
    p_values = [spn_ci_test(spn, X, i, j, cond_set) for spn in spns]
    return np.median(p_values)  # More robust
```

---

## Implementation Plan

### Week 1: Verification (Experiment 1.1-1.4)
**Priority**: Critical path to validate hypotheses

**Day 1-2**: Oracle test implementation
- Create oracle CI test using true covariance
- Run MEDIUM horizontal with oracle vs learned SPN
- Compare F1 scores

**Day 3-4**: Capacity sweep
- Run 5 experiments (num_sums: 20, 40, 60, 80, 100)
- Plot F1 vs capacity, MMD vs capacity

**Day 5**: Alpha and permutation tests
- Quick sweep to rule out these factors

**Day 6-7**: Analysis and decision
- Determine minimum viable architecture
- Finalize v2 adaptive rules

**Deliverable**: Verification report confirming root causes

---

### Week 2-3: Full V2 Experiments
**Priority**: Establish new baseline

**Day 8-10**: Implement adaptive hyperparameter system
- Add adaptive logic to FedCDH.py
- Add gradient clipping
- Add early stopping (optional)

**Day 11-16**: Run all 18 experiments
- 3 configs × 3 modes × 2 data types
- Save results to experiments/v2_adaptive/

**Day 17-18**: Analysis
- Generate HTML report (same format as v1)
- Compare v1 vs v2 side-by-side
- Document improvements

**Deliverable**: V2 experiment results with comparison

---

### Week 4: Advanced Improvements (Optional)
**Priority**: Medium (only if Week 1-3 successful)

**Day 19-21**: SPN quality rejection + retraining
**Day 22-24**: Ensemble SPNs
**Day 25-28**: Final analysis and thesis integration

---

## Code Changes Required

### 1. FedCDH.py: Add Adaptive Architecture

**Location**: `causallearn/search/FCMBased/FedCDH/FedCDH.py` ~line 530

**Current**:
```python
# Fixed architecture
num_sums = 20
num_leaves = 20
num_repetitions = 10
depth = max(1, int(np.floor(np.log2(local_d))))
```

**New**:
```python
def get_adaptive_architecture(mode, num_features, num_clients, num_samples,
                              data_type='linear'):
    """Adaptive hyperparameter selection based on problem characteristics."""

    # Determine local problem size
    if mode == 'horizontal':
        local_features = num_features
        local_samples = num_samples // num_clients
    elif mode == 'vertical':
        local_features = num_features // num_clients
        local_samples = num_samples
    else:  # hybrid
        local_features = num_features
        local_samples = num_samples // num_clients

    local_d = local_features + 1  # +1 for context

    # Mode-specific base capacity
    if mode == 'horizontal':
        base_sums = max(32, 4 * local_features)
        base_leaves = max(16, 2 * local_features)
    elif mode == 'vertical':
        if local_features <= 3:
            base_sums = 10
            base_leaves = 10
        else:
            base_sums = 8 * local_features
            base_leaves = 4 * local_features
    else:  # hybrid
        base_sums = 6 * local_features
        base_leaves = 3 * local_features

    # Sample-to-feature ratio scaling
    ratio = local_samples / local_features
    if ratio < 50:
        scale = 0.5
    elif ratio < 100:
        scale = 0.75
    elif ratio < 200:
        scale = 1.0
    else:
        scale = min(1.5, 1.0 + (ratio - 200) / 400)

    num_sums = int(base_sums * scale)
    num_leaves = int(base_leaves * scale)

    # Data type adjustment
    if data_type == 'nonlinear':
        num_sums = int(num_sums * 1.5)
        num_leaves = int(num_leaves * 2.0)
        depth_bonus = 1
    else:
        depth_bonus = 0

    # Calculate depth
    base_depth = max(1, int(np.floor(np.log2(local_d))))
    depth = base_depth + depth_bonus

    # Epoch scaling
    if data_type == 'nonlinear':
        epoch_multiplier = 1.5
    else:
        epoch_multiplier = 1.0

    if mode == 'horizontal':
        epoch_multiplier *= (1.0 + local_features / 30)
    elif mode == 'vertical':
        epoch_multiplier *= 0.75

    # Enforce bounds
    num_sums = max(8, min(num_sums, 128))
    num_leaves = max(8, min(num_leaves, 256))
    depth = max(1, min(depth, 6))

    return {
        'num_sums': num_sums,
        'num_leaves': num_leaves,
        'depth': depth,
        'num_repetitions': 10,
        'epoch_multiplier': epoch_multiplier
    }
```

**Integration**:
```python
# In FedCDH.__init__() or train_local_spns()
arch = get_adaptive_architecture(
    mode=self.scenario,
    num_features=self.d,
    num_clients=self.K,
    num_samples=len(X),
    data_type='linear'  # or pass as parameter
)

# Use adaptive values
spn_wrapper = LocalSPNWrapper(
    num_sums=arch['num_sums'],
    num_leaves=arch['num_leaves'],
    depth=arch['depth'],
    num_repetitions=arch['num_repetitions']
)
```

---

### 2. FedPC.py: Add Gradient Clipping

**Location**: `causallearn/utils/FedPC.py` in `LocalSPNWrapper.fit()`

**Add**:
```python
# In training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    ll = einet(data_tensor)
    loss = -ll.mean()
    loss.backward()

    # Gradient clipping (NEW)
    torch.nn.utils.clip_grad_norm_(einet.parameters(), max_norm=1.0)

    optimizer.step()
```

---

### 3. cit.py: Increase num_permutations

**Location**: `causallearn/utils/cit.py` ~line 930

**Current**:
```python
num_permutations = 50
```

**New**:
```python
num_permutations = 100  # More robust permutation test
```

---

### 4. Oracle CI Test (For Verification)

**New File**: `causallearn/utils/oracle_cit.py`

```python
import numpy as np
from scipy.stats import norm

def oracle_gaussian_ci_test(data, i, j, cond_set, true_B, true_noise_scales, alpha=0.05):
    """
    Conditional independence test using known Gaussian SEM parameters.

    Args:
        data: n×d data matrix
        i, j: Variable indices to test
        cond_set: Conditioning set
        true_B: d×d true causal graph weight matrix
        true_noise_scales: d-vector of noise standard deviations
        alpha: Significance level

    Returns:
        p_value: Test p-value
    """

    # Compute true covariance matrix from SEM parameters
    d = len(true_noise_scales)
    I = np.eye(d)
    Sigma_noise = np.diag(true_noise_scales ** 2)

    # Cov(X) = (I - B)^{-1} Sigma_noise (I - B)^{-T}
    IB_inv = np.linalg.inv(I - true_B)
    Sigma = IB_inv @ Sigma_noise @ IB_inv.T

    # Compute partial correlation
    if len(cond_set) == 0:
        # Marginal test
        rho = Sigma[i, j] / np.sqrt(Sigma[i, i] * Sigma[j, j])
    else:
        # Partial correlation using Schur complement
        vars_idx = [i, j] + list(cond_set)
        Sigma_sub = Sigma[np.ix_(vars_idx, vars_idx)]

        # Precision matrix
        Omega = np.linalg.inv(Sigma_sub)

        # Partial correlation
        rho = -Omega[0, 1] / np.sqrt(Omega[0, 0] * Omega[1, 1])

    # Fisher's z-transform
    n = len(data)
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    z_stat = z * np.sqrt(n - len(cond_set) - 3)

    # Two-tailed test
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return p_value
```

---

## Success Metrics

### Primary Metrics
1. **No F1=0 failures** in any experiment
2. **MEDIUM Horizontal F1 > 0.5** (currently 0.0)
3. **LARGE Horizontal F1 > 0.4** (currently 0.0)

### Secondary Metrics
4. **MMD p-value > 0.05** in >50% of experiments (currently ~5%)
5. **Average F1 improvement** across all 18 experiments: +0.2-0.3
6. **Horizontal mode competitive** with vertical/hybrid (currently loses)

### Verification Metrics (Phase 1)
7. **Oracle F1 > 0.7** on MEDIUM config
8. **Capacity threshold identified**: Minimum num_sums for F1 > 0.5
9. **Correlation confirmed**: SPN quality (MMD) vs F1 score

---

## Risk Mitigation

### Risk 1: Adaptive rules still insufficient
**Mitigation**: Phase 1 capacity sweep establishes empirical minimums

### Risk 2: Computational cost explosion
**Mitigation**:
- Cap num_sums at 128, num_leaves at 256
- Use early stopping to prevent wasted epochs
- Run LARGE config last (most expensive)

### Risk 3: Oracle test doesn't work
**Mitigation**: If oracle also fails, investigate:
- CI test implementation bugs
- Data generation issues
- PC algorithm logic

### Risk 4: Time constraints
**Mitigation**:
- Phase 1 (verification) is highest priority
- Phase 2 critical for thesis
- Phase 3 optional enhancements

---

## Deliverables

### Week 1: Verification Report
- Root cause confirmation
- Oracle vs learned SPN comparison
- Capacity sweep results (F1 vs num_sums plot)
- Minimum viable architecture identified

### Week 2-3: V2 Experimental Results
- `experiments/v2_adaptive/` directory
- HTML report with all 18 experiments
- V1 vs V2 comparison table
- Updated working_state.md

### Week 4: Thesis Integration
- Methodology section: Adaptive hyperparameter system
- Results section: V1 (baseline) vs V2 (adaptive)
- Discussion: Capacity limitations and solutions

---

## Open Questions

1. **Should we use data_type='linear' or 'nonlinear' in adaptive rules?**
   - Requires knowing data type a priori
   - Alternative: Detect automatically from data (how?)

2. **Should we retrain poor-quality SPNs automatically?**
   - If MMD p < 0.01, increase capacity and retrain
   - Risk: Infinite loop if fundamentally insufficient samples

3. **How to handle hybrid mode feature grouping?**
   - Current: Equal split (simplified)
   - Proper: Use overlapping features (requires user input)

4. **Should we ensemble SPNs for robustness?**
   - Multiple seeds, take median p-value
   - Computational cost: 5× increase

---

## Timeline Summary

| Week | Phase | Key Deliverables | Priority |
|------|-------|-----------------|----------|
| **1** | Verification | Oracle test, capacity sweep, root cause confirmation | CRITICAL |
| **2-3** | V2 Experiments | 18 adaptive experiments, HTML report, v1 vs v2 comparison | HIGH |
| **4** | Advanced | Early stopping, quality rejection, ensembles | MEDIUM |

**Total Duration**: 4 weeks
**Critical Path**: Week 1 → Week 2-3 (Week 4 optional)

---

## Conclusion

The v1 baseline revealed a critical capacity limitation: fixed architecture (20/20) completely fails for d×K > 30. The "predict nothing" paradox (high accuracy but F1=0) occurs because poor SPN quality makes CI tests too conservative.

**V2 addresses this through**:
1. Adaptive architecture scaling with problem size
2. Systematic verification (oracle test, capacity sweep)
3. Improved training (gradient clipping, more permutations)

**Expected impact**: Fix all F1=0 failures, establish proper baseline for realistic problem sizes, validate that FedCDH can work with sufficient SPN capacity.
