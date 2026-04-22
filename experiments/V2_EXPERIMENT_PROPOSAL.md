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
def oracle_ci_test(data, i, j, cond_set, true_B, true_noise_scales, alpha=0.05):
    """
    True oracle CI test using known SEM parameters (not sample data).
    Computes exact partial correlation from ground truth covariance.
    """
    import numpy as np
    from scipy.stats import norm

    # Compute true covariance: Σ = (I-B)^{-1} Σ_noise (I-B)^{-T}
    d = len(true_noise_scales)
    I = np.eye(d)
    Sigma_noise = np.diag(true_noise_scales ** 2)
    IB_inv = np.linalg.inv(I - true_B)
    Sigma = IB_inv @ Sigma_noise @ IB_inv.T

    # Compute partial correlation
    if len(cond_set) == 0:
        rho = Sigma[i, j] / np.sqrt(Sigma[i, i] * Sigma[j, j])
    else:
        vars_idx = [i, j] + list(cond_set)
        Sigma_sub = Sigma[np.ix_(vars_idx, vars_idx)]
        Omega = np.linalg.inv(Sigma_sub)
        rho = -Omega[0, 1] / np.sqrt(Omega[0, 0] * Omega[1, 1])

    # Fisher's z-transform
    n = len(data)
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    z_stat = z * np.sqrt(n - len(cond_set) - 3)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return p_value
```

**Ground Truth Extraction** (required for oracle test):
```python
# In benchmark data generation script
import numpy as np

# Generate SEM
dag, B_true = create_synthetic_dag(d=10, density=0.3)
noise_scales = np.random.uniform(0.5, 2.0, d)

# Save ground truth parameters
np.save(f'{exp_dir}/true_B.npy', B_true)
np.save(f'{exp_dir}/true_noise_scales.npy', noise_scales)

# Generate data
X = generate_linear_sem_data(B_true, noise_scales, n=1200)
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

### Phase 3: Advanced Improvements (FUTURE WORK - Not in 4-Week Timeline)

**Note**: Phase 3 features require significant additional computational resources (15+ days GPU time). These are deferred to future work or separate experimental runs.

#### 3.1: Early Stopping Based on Validation LL
**Time cost**: Implementation only (no extra runtime)
**Benefit**: Prevent overfitting, potentially reduce epochs needed

```python
# Requires validation split (reduces training data by 20%)
patience = 20
best_val_ll = -inf
no_improvement_count = 0

for epoch in range(max_epochs):
    train_ll = train_spn(X_train)
    val_ll = evaluate_spn(X_val)  # Need to create X_val split

    if val_ll > best_val_ll:
        best_val_ll = val_ll
        no_improvement_count = 0
        save_checkpoint()
    else:
        no_improvement_count += 1

    if no_improvement_count >= patience:
        break  # Stop early

# Only use if client has >300 samples (SMALL config has 200/client)
```

#### 3.2: SPN Quality Rejection Threshold
**Time cost**: +20% runtime (some SPNs need retraining)
**Benefit**: Automatic quality control

```python
# After training SPN, check quality before using for CI
if mmd_pvalue < 0.01 or ks_fail_rate > 0.8:
    print(f"⚠️ Poor SPN quality (MMD p={mmd_pvalue:.3f}), retraining with increased capacity")
    num_sums = int(num_sums * 1.5)
    num_leaves = int(num_leaves * 1.5)
    retrain_spn()

# Risk: Infinite loop if samples insufficient. Need max retries.
max_retries = 2
```

#### 3.3: Ensemble SPNs
**Time cost**: 5× training time (360 hours for 18 experiments)
**Benefit**: More robust CI tests, reduced variance

```python
# Train multiple SPNs with different initializations
spns = [train_spn(seed=i) for i in range(5)]

# Use ensemble for CI test
def ensemble_ci_test(X, i, j, cond_set):
    p_values = [spn_ci_test(spn, X, i, j, cond_set) for spn in spns]
    return np.median(p_values)  # More robust

# Alternative: Test on SMALL config only (3 experiments) to assess benefit
```

**Recommendation**: Only implement Phase 3 if:
1. Phase 2 shows MMD p-values still low (<0.05) despite adaptive architecture
2. Additional computational budget available (15+ GPU days)
3. Needed for thesis contribution beyond fixing baseline failures

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

### Week 4: Documentation & Future Work Planning
**Priority**: HIGH (essential for thesis)

**Day 19-21**: Analysis & visualization
- Generate v1 vs v2 comparison visualizations
- Statistical significance tests (paired t-test on F1 scores)
- Document capacity thresholds found in Phase 1

**Day 22-24**: Thesis integration
- Methodology section: Adaptive hyperparameter system
- Results section: V2 experimental results
- Discussion: Capacity limitations and solutions

**Day 25-28**: Phase 3 planning (optional future work)
- Design ensemble SPN experiment (if time permits)
- Document early stopping implementation
- Create roadmap for Phase 3 experiments

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

**Exact location**: Look for the training loop in `LocalSPNWrapper.fit()` method (~line 150-200)

**Current code pattern**:
```python
for epoch in range(num_epochs):
    optimizer.zero_grad()
    ll = einet(data_tensor)
    loss = -ll.mean()
    loss.backward()
    optimizer.step()  # ← ADD CLIPPING BEFORE THIS LINE
```

**Add gradient clipping AFTER backward() and BEFORE optimizer.step()**:
```python
# In training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    ll = einet(data_tensor)
    loss = -ll.mean()
    loss.backward()

    # Gradient clipping (NEW) - prevents gradient explosions
    torch.nn.utils.clip_grad_norm_(einet.parameters(), max_norm=1.0)

    optimizer.step()
```

**Verification**: After implementation, add logging to confirm clipping is active:
```python
grad_norm = torch.nn.utils.clip_grad_norm_(einet.parameters(), max_norm=1.0)
if epoch % 10 == 0:
    print(f"Epoch {epoch}: Grad norm = {grad_norm:.4f}")
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

## Resource Requirements & Computational Budget

### Hardware Assumptions
- **GPU**: 1× NVIDIA GPU (CUDA 12.4 compatible)
- **RAM**: 32GB minimum
- **Storage**: 50GB for experiment results

### Time Estimates

**Phase 1 (Verification)**:
| Experiment | Description | Est. Time per Run | Total |
|------------|-------------|-------------------|-------|
| 1.1 Oracle Test | 1 MEDIUM horizontal | 2 hours | 2 hours |
| 1.2 Capacity Sweep | 5 configs (20/40/60/80/100 sums) | 2-6 hours each | 20 hours |
| 1.3 Alpha Sweep | 4 alpha values | 2 hours each | 8 hours |
| 1.4 Permutation Sweep | 4 permutation counts | 2 hours each | 8 hours |
| **Phase 1 Total** | | | **38 hours** (1.6 days) |

**Phase 2 (Full V2 Adaptive)**:
- 18 experiments (3 configs × 3 modes × 2 data types)
- With adaptive architecture (60-100 sums/leaves for horizontal):
  - SMALL: ~2 hours per experiment → 6 experiments × 2h = 12 hours
  - MEDIUM: ~4 hours per experiment → 6 experiments × 4h = 24 hours
  - LARGE: ~6 hours per experiment → 6 experiments × 6h = 36 hours
- **Phase 2 Total**: **72 hours** (3 days)

**Phase 3 (Advanced Features) - OPTIONAL**:
- Early stopping implementation: 1 day (no extra runtime)
- SPN quality rejection: Adds ~20% to Phase 2 time → 14 hours
- Ensemble SPNs (5 seeds): 5× training time → 360 hours (15 days)
- **Phase 3 Total**: **360+ hours** (15 days) - **NOT FEASIBLE IN 4-WEEK TIMELINE**

### Total Timeline

**Realistic 4-Week Plan** (Phase 1-2 only):
- Week 1: Phase 1 verification (2 days compute + 3 days analysis)
- Week 2-3: Phase 2 full v2 (3 days compute + 4 days analysis/comparison)
- Week 4: Documentation, thesis integration, prepare for Phase 3 (future work)

**With Multiple GPUs**:
- 2 GPUs: Cut Phase 1-2 time in half (2.5 days total compute)
- 4 GPUs: Phase 1-2 completes in ~1.5 days compute

### Cost-Benefit Analysis

| Feature | Time Cost | Expected Benefit | Priority |
|---------|-----------|------------------|----------|
| Adaptive architecture | 72 hours | Fix F1=0 failures | **CRITICAL** |
| Gradient clipping | 0 hours | Prevent instabilities | **HIGH** |
| Increased permutations | +20% time | More robust CI tests | **MEDIUM** |
| Early stopping | 0 hours | Prevent overfitting | **MEDIUM** |
| Ensemble SPNs | 5× time | Marginal MMD improvement | **LOW** |

**Recommendation**: Focus on Phase 1-2, defer Phase 3 to separate experimental run or thesis future work.

---

## Contingency Plans

### If Phase 1 Results Are Unexpected

#### Scenario A: Oracle F1 < 0.5 (CI test logic issue)
**Symptom**: Even with true covariance, causal discovery fails

**Actions**:
1. Verify oracle implementation computes partial correlations correctly
2. Check PC algorithm implementation for bugs
3. Test with trivial DAGs (chain, fork, collider)
4. Compare with causal-learn's fisherz on same ground truth

**Mitigation**: Do NOT proceed to Phase 2 capacity increase. Fix CI test first.

#### Scenario B: Capacity sweep plateaus early (F1 stops improving at num_sums=40)
**Symptom**: F1 doesn't reach >0.5 even with high capacity

**Actions**:
1. Check local SPN quality per client (are individual SPNs good?)
2. Verify global aggregation (mixture-of-experts weights correct?)
3. Test with oracle CI on individual clients
4. Inspect CI test p-value distributions (are they all >0.5? → too conservative)

**Mitigation**: Investigate aggregation logic before increasing capacity further.

#### Scenario C: Alpha=0.15 shows F1 > 0.5 with acceptable FP rate
**Symptom**: Threshold adjustment helps more than capacity increase

**Actions**:
1. Implement adaptive alpha based on SPN quality:
   ```python
   if mmd_pvalue < 0.01:
       alpha = 0.10  # More lenient for poor SPNs
   else:
       alpha = 0.05  # Standard for good SPNs
   ```
2. Compare adaptive alpha vs fixed capacity increase

**Mitigation**: Consider alpha adjustment as complement to capacity increase, not replacement.

#### Scenario D: Phase 2 v2 still fails on LARGE config
**Symptom**: Even with adaptive architecture, LARGE F1 < 0.2

**Actions**:
1. Check if LARGE complexity (d×K=55) exceeds SPN limits
2. Compare with vertical mode on LARGE (if vertical works, horizontal can too)
3. Increase capacity bounds: max_sums from 128 to 200
4. Test with even more samples (n=3000 instead of 1650)

**Mitigation**: Accept that some problem sizes require infeasible computational resources. Document limitations.

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
- `experiments/v2_adaptive/` directory structure:
  ```
  v2_adaptive/
  ├── eval_linear/
  ├── eval_nonlinear/
  ├── experiment_analysis_report.html
  ├── v1_vs_v2_comparison.html
  └── README.md
  ```
- HTML report with all 18 experiments (same format as v1)
- V1 vs V2 side-by-side comparison with statistical tests
- Updated working_state.md with v2 results

### Week 4: Thesis Integration & Documentation
- Methodology section: Adaptive hyperparameter system design
- Results section: V2 experimental results with v1 comparison
- Discussion: Capacity limitations, solutions, and remaining challenges
- Future work section: Phase 3 advanced features (ensemble, early stopping)

---

## Design Decisions & Justifications

### 1. Data Type Detection

**Decision**: For v2 synthetic experiments, data type is KNOWN (use it explicitly)

**Rationale**:
- Synthetic benchmark generates linear or nonlinear data by design
- Pass `data_type='linear'` or `data_type='nonlinear'` to adaptive architecture
- For real-world data (Sachs), default to `data_type='nonlinear'` (conservative approach)

**Future work**: Automatic linearity detection via R² heuristic or hypothesis test

### 2. Hybrid Mode Feature Grouping

**Decision**: Use simplified hybrid (equal feature split) for v2

**Rationale**:
- Proper overlapping features require user-defined grouping (not available in benchmark)
- Simplified hybrid still provides 2-level hierarchy benefit
- Adding proper hybrid would delay v2 by 2-3 days

**Future work**: Implement overlapping feature groups with user configuration

### 3. SPN Quality Rejection

**Decision**: Do NOT implement automatic retraining in v2

**Rationale**:
- Risk of infinite loops if samples fundamentally insufficient
- Adaptive architecture should prevent poor quality upfront
- If Phase 2 still shows low MMD, can manually investigate

**Phase 3**: Implement with max_retries=2 safety limit

### 4. Ensemble SPNs

**Decision**: DEFER to Phase 3 (future work)

**Rationale**:
- 5× computational cost (360 hours)
- Benefit unclear (may only marginally improve already-good SPNs)
- Not critical for fixing F1=0 failures

**Alternative**: Test on SMALL config only (3 experiments) to assess potential benefit

### 5. Validation Splits for Early Stopping

**Decision**: Do NOT use validation splits in v2

**Rationale**:
- Reduces training data by 20% (SMALL has only 200 samples/client → 160 train)
- Fixed epoch counts (50-75) are reasonable for current problem sizes
- Early stopping adds complexity without proven benefit

**Phase 3**: Implement only for clients with >300 samples

---

## Timeline Summary

| Week | Phase | Key Deliverables | Compute Time | Priority |
|------|-------|-----------------|--------------|----------|
| **1** | Verification | Oracle test, capacity sweep, root cause confirmation | 38 hours | CRITICAL |
| **2-3** | V2 Experiments | 18 adaptive experiments, HTML report, v1 vs v2 comparison | 72 hours | HIGH |
| **4** | Documentation | Thesis integration, analysis, visualizations | 0 hours | HIGH |
| **Future** | Phase 3 | Early stopping, quality rejection, ensembles | 360+ hours | OPTIONAL |

**Total Duration**: 4 weeks for Phase 1-2 + documentation
**Compute Time**: 110 hours (4.6 days with 1 GPU, 2.3 days with 2 GPUs)
**Critical Path**: Week 1 (verification) → Week 2-3 (full v2) → Week 4 (analysis)
**Phase 3**: Deferred to future work (requires 15+ additional GPU days)

---

## V1 vs V2 Comparison Methodology

### Metrics to Compare

**SPN Quality**:
- Train log-likelihood (higher is better)
- MMD p-value (>0.05 indicates good quality)
- KS test failure rate (<0.5 acceptable)

**Causal Discovery**:
- Overall F1 (primary metric)
- Precision, Recall
- Skeleton Accuracy, Overall Accuracy
- Confusion matrix (TP, FP, FN, TN)

### Statistical Tests

**Paired comparison** (same config, v1 vs v2):
```python
# Compare F1 scores across 18 experiments
v1_f1 = [0.303, 0.857, ..., 0.727]  # 18 values
v2_f1 = [0.5+, 0.8+, ..., 0.8+]      # 18 values

# Paired t-test
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(v2_f1, v1_f1)
print(f"V2 improvement significant: p={p_value:.4f}")

# Effect size (Cohen's d)
mean_diff = np.mean(v2_f1) - np.mean(v1_f1)
pooled_std = np.sqrt((np.var(v1_f1) + np.var(v2_f1)) / 2)
cohens_d = mean_diff / pooled_std
```

### Visualizations

1. **Side-by-side bar chart**: F1 scores for all 18 experiments
2. **Scatter plot**: V1 F1 vs V2 F1 (points above diagonal = improvement)
3. **Heatmap**: MMD p-values (v1 vs v2) per config/mode
4. **Box plots**: F1 distribution by mode (horizontal, vertical, hybrid)

### Success Criteria

**Primary**: No F1=0 failures in v2
**Secondary**:
- Mean F1 improvement ≥ +0.2
- ≥50% experiments with MMD p > 0.05
- Horizontal mode competitive with vertical/hybrid

---

## Implementation Validation

### Before Running Phase 1

**Unit tests for adaptive architecture**:
```python
def test_adaptive_architecture():
    # Test horizontal mode
    arch = get_adaptive_architecture('horizontal', num_features=10,
                                     num_clients=3, num_samples=1200)
    assert arch['num_sums'] >= 60, "Horizontal needs high capacity"

    # Test vertical mode with few features
    arch = get_adaptive_architecture('vertical', num_features=6,
                                     num_clients=3, num_samples=1200)
    assert arch['num_sums'] == 10, "Vertical with 2 features should use 10 sums"

    # Test capacity bounds
    arch = get_adaptive_architecture('horizontal', num_features=50,
                                     num_clients=3, num_samples=5000)
    assert arch['num_sums'] <= 128, "Should enforce max bounds"
```

**Smoke test on toy data**:
```bash
# Run SMALL config with d=5, K=2, n=200 (should complete in 10 minutes)
python tests/smoke/test_v2_adaptive.py --config tiny
```

**Gradient clipping verification**:
```python
# After implementation, check logs for gradient norms
# Should see lines like: "Epoch 10: Grad norm = 0.8342"
grep "Grad norm" experiments/v2_adaptive/*/run.log
```

**Oracle CI test validation**:
```python
# Test with known covariance matrix
true_B = np.array([[0, 0.5], [0, 0]])  # X→Y
noise_scales = np.array([1.0, 1.0])

# Generate data
X = generate_linear_sem_data(true_B, noise_scales, n=1000)

# Oracle test should reject independence for X⊥Y (edge exists)
p_value = oracle_ci_test(X, 0, 1, [], true_B, noise_scales)
assert p_value < 0.05, "Should detect X→Y edge"

# Should accept independence for X⊥Y|Y (blocked by conditioning)
p_value = oracle_ci_test(X, 0, 1, [1], true_B, noise_scales)
# (Note: This is a degenerate case, just for testing)
```

---

## Conclusion

The v1 baseline revealed a critical capacity limitation: fixed architecture (20/20) completely fails for d×K > 30. The "predict nothing" paradox (high accuracy but F1=0) occurs because poor SPN quality makes CI tests too conservative.

**V2 addresses this through**:
1. **Adaptive architecture** scaling with problem size (horizontal: 60-80 sums, vertical: 10 sums for ≤3 features)
2. **Systematic verification** (oracle test validates hypothesis, capacity sweep finds thresholds)
3. **Improved training** (gradient clipping prevents instabilities, increased permutations for robustness)
4. **Clear contingency plans** (if Phase 1 fails, investigate CI test logic before increasing capacity)

**Expected impact**:
- Fix all F1=0 failures (MEDIUM/LARGE horizontal)
- Establish proper baseline for realistic problem sizes (d×K up to 55)
- Validate that FedCDH works with sufficient SPN capacity
- Provide empirical evidence that SPN quality drives causal discovery performance

**Phase 3 (future work)**: Advanced features like ensemble SPNs and early stopping require 15+ additional GPU days, deferred to separate experimental run or thesis future work section.
