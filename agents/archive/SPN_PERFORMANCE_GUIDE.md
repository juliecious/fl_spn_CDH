# How to Ensure SPNs Are Performant

**Last Updated**: 2026-04-02

---

## Overview

SPN performance is critical for FedCDH success - if SPNs don't learn data distributions correctly, conditional independence tests will be unreliable. This guide covers **what's implemented** and **best practices**.

---

## 1. Training-Time Monitoring (Implemented ✅)

### Convergence Detection

**Location**: `causallearn/utils/FedPC.py:172-220`

**What's Monitored**:

```python
# During training
loss_history = []
for epoch in range(epochs):
    loss = compute_loss(...)
    loss_history.append(loss.item())

    # Log every 10 epochs
    if (epoch + 1) % 10 == 0:
        logging.debug(f"Epoch {epoch + 1}/{epochs}: Loss={loss:.4f}")

# After training
logging.info(f"Final Loss={loss_history[-1]:.4f}")

# Convergence warning
if len(loss_history) >= 20:
    last_20_losses = loss_history[-20:]
    if min(last_20_losses) >= loss_history[-20]:
        logging.warning(
            "SPN may not have converged: loss not decreasing "
            f"in last 20 epochs (started at {loss_history[-20]:.4f}, "
            f"ended at {loss_history[-1]:.4f})"
        )
```

**Interpretation**:
- ✅ Loss decreasing → SPN learning
- ⚠️ Loss plateau early → Increase epochs or learning rate
- ❌ Loss not decreasing in last 20 epochs → Undertraining

**Action**: Check logs for convergence warnings before running experiments.

---

## 2. Post-Training Quality Metrics (Implemented ✅)

### Framework: `tests/benchmarks/evaluate_spn.py`

**4 Core Metrics**:

#### Metric 1: Train/Test Log-Likelihood

```python
# What's computed
train_ll = spn.log_prob(X_train).mean()
test_ll = spn.log_prob(X_test).mean()
overfitting_gap = abs(train_ll - test_ll) / abs(train_ll)

# Thresholds
✅ gap < 0.20: Excellent generalization
⚠️ gap 0.20-0.50: Mild overfitting (acceptable for thesis)
❌ gap > 0.50: Severe overfitting (reduce complexity or get more data)
```

**Why This Matters**:
- SPNs can memorize training data without learning true distribution
- Test LL detects this overfitting
- Causal discovery needs **generalization**, not memorization

**Example Output**:
```
Local SPN (Client 0):
  Train LL: -7.653
  Test LL: -8.171
  Overfitting gap: 0.068 ✅ (threshold: <0.20)
```

---

#### Metric 2: MMD² (Maximum Mean Discrepancy)

```python
# What's computed
generated_samples = spn.sample(n=1000)
mmd_squared = compute_mmd(real_data, generated_samples)
mmd_pvalue = permutation_test(mmd_squared, n_permutations=1000)

# Thresholds
✅ p > 0.05: Generated data statistically matches real data
⚠️ p 0.01-0.05: Marginal match
❌ p < 0.01: Distributions clearly different
```

**Why This Matters**:
- Gold standard for distribution comparison (Gretton et al. 2012, JMLR)
- Used in GAN/VAE evaluation
- Catches subtle distribution mismatches that LL might miss

**Implementation Details**:
```python
# Unbiased MMD² estimator with RBF kernel
def _compute_mmd_squared(X, Y):
    # Median bandwidth heuristic (standard in literature)
    median_pairwise = np.median(pairwise_distances(X))
    gamma = 1.0 / (2 * median_pairwise**2)

    # Kernel matrices
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)

    # Unbiased estimator (removes diagonal)
    n = len(X)
    mmd² = (Kxx.sum() - np.trace(Kxx)) / (n * (n-1))
        + (Kyy.sum() - np.trace(Kyy)) / (n * (n-1))
        - 2 * Kxy.mean()

    return mmd²
```

**Example Output**:
```
Local SPN (Client 0):
  MMD² (×10⁻³): 0.15
  MMD p-value: 0.693 ✅ (threshold: >0.05)
  → Generated samples match real distribution
```

---

#### Metric 3: Kolmogorov-Smirnov Test (Per Dimension)

```python
# What's computed
for dim in range(d):
    real_marginal = real_data[:, dim]
    gen_marginal = generated_samples[:, dim]
    ks_stat, ks_pvalue = ks_2samp(real_marginal, gen_marginal)

    # Bonferroni correction for multiple testing
    if ks_pvalue < 0.05 / d:
        failed_dims.append(dim)

# Thresholds
✅ failed < 30%: Most marginals correct
⚠️ failed 30-50%: Some issues but acceptable
❌ failed > 50%: Poor marginal matching
```

**Why This Matters**:
- Checks if SPNs learn correct **marginal distributions**
- Complements MMD (which checks joint distribution)
- Easier to interpret (per-variable diagnostic)

**Example Output**:
```
Local SPN (Client 0):
  KS test: 2/6 dimensions failed (33%)
  Failed dimensions: [1, 5]
  → Marginal distributions mostly correct
```

---

#### Metric 4: Convergence Analysis (If Training Losses Provided)

```python
# What's computed
last_10_losses = loss_history[-10:]
loss_std = np.std(last_10_losses)
loss_trend = (last_10_losses[-1] - last_10_losses[0]) / 10

# Thresholds
✅ std < 0.05 AND |trend| < 0.1: Converged
⚠️ std 0.05-0.10 OR |trend| 0.1-0.2: Marginal convergence
❌ std > 0.10 OR |trend| > 0.2: Not converged
```

**Why This Matters**:
- Detects if training stopped too early
- Identifies oscillating optimization (need lower learning rate)

**Example Output**:
```
Convergence Analysis:
  Last 10 epochs std: 0.023 ✅
  Last 10 epochs trend: -0.015 ✅
  → Training converged
```

---

## 3. Quality Criteria Summary

### Minimum Acceptable for Thesis

**Critical (Must Pass)**:
- ✅ **Global SPN MMD p-value > 0.05** (most important!)

**Acceptable (At Least One)**:
- ⚠️ Local SPNs overfitting gap < 0.50 (mild overfitting OK)
- OR ⚠️ Local SPNs MMD p-value > 0.01 (marginal match OK)

**Rationale**: Global SPN is what's used for CI testing. Local SPNs can underperform if global compensates via aggregation.

### Ideal Criteria (Stretch Goal)

- ✅ All SPNs: MMD p-value > 0.05
- ✅ All SPNs: Overfitting gap < 0.20
- ✅ All SPNs: KS failed dims < 30%
- ✅ Training: Loss converged (trend ≈ 0, std < 0.05)

---

## 4. Hyperparameter Tuning (What Affects Performance)

### Epochs

**Current Default**: 50 (GPU), 10 (CPU)

**Guideline**:
```
d=5:  30-50 epochs sufficient
d=8:  50-100 epochs recommended
d=10: 100-150 epochs needed
d>15: 150-200 epochs
```

**How to Check**: Look for convergence warnings in logs. If loss still decreasing at end, increase epochs.

---

### SPN Architecture

**Current Settings**:
```python
num_sums = 20       # Number of sum nodes per layer
num_leaves = 20     # Number of leaf distributions
num_repetitions = 10  # Structural redundancy
depth = 3           # Network depth
```

**Trade-offs**:

| Parameter | ↑ Increase | ↓ Decrease |
|-----------|-----------|-----------|
| `num_sums` | More expressive, slower | Faster, may underfit |
| `num_leaves` | Better marginals | Faster training |
| `depth` | Captures complex dependencies | Avoids overfitting |
| `num_repetitions` | Robustness to bad initializations | Faster, less memory |

**Recommendation for d=8**:
```python
# Conservative (fast, less overfitting risk)
num_sums=15, num_leaves=15, depth=3

# Recommended (balanced)
num_sums=20, num_leaves=20, depth=3

# Aggressive (high capacity, needs more data)
num_sums=30, num_leaves=30, depth=4
```

**Rule of Thumb**: Capacity should scale with `√d × √N`

---

### Learning Rate

**Current Default**: 0.005 (LocalSPN), 0.01 (UnivariateSPN)

**Guideline**:
```python
# If loss oscillates → Decrease LR
lr = 0.001  # Conservative

# If loss plateaus early → Increase LR
lr = 0.01   # Aggressive

# Adaptive option (recommended)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

---

### L1 Regularization

**Current Default**: `l1_weight=1e-4`

**Effect**: Sparsifies SPN structure (prunes weak edges)

**Guideline**:
```python
# Small data (N < 500) → More regularization
l1_weight = 1e-3

# Medium data (N = 500-2000) → Moderate
l1_weight = 1e-4

# Large data (N > 2000) → Less regularization
l1_weight = 1e-5
```

---

## 5. Practical Workflow

### Before Running Experiments

**Step 1: Quick Convergence Check** (5 minutes)
```bash
# Run smoke test with verbose logging
python tests/benchmarks/smoke_test_nonlinear.py

# Check for convergence warnings in output
grep "WARNING" output.log
grep "Final Loss" output.log
```

**Red Flags**:
- "SPN may not have converged" warnings
- Final loss still decreasing rapidly (> 0.1 per 10 epochs)

**Fix**: Increase epochs (30 → 50 → 100)

---

### During Development

**Step 2: Full SPN Evaluation** (20 minutes)
```python
from tests.benchmarks.evaluate_spn import evaluate_fedcdh_spns

# After training FedCDH
results = evaluate_fedcdh_spns(
    fedcdh_model=model,
    X=X_global,
    c_indx=c_indx,
    K=3,
    scenario="horizontal",
    output_dir="results/spn_eval/"
)

# Check global SPN quality
if results['global']['mmd_pvalue'] < 0.05:
    print("⚠️ WARNING: Global SPN quality poor!")
    print("→ Increase epochs or adjust architecture")
```

**Output Files**:
- `table1_local_evaluation.csv` - Per-client metrics
- `table2_global_evaluation.csv` - Global SPN metrics
- `evaluation_summary.txt` - Pass/fail assessment

---

### Before Thesis Experiments

**Step 3: Validate on Representative Data** (1 hour)
```bash
# Test on d=8, K=3, N=900 (similar to Sachs)
python tests/benchmarks/comprehensive_benchmark.py \
    --d 8 --K 3 --n 900 --epochs 100 --device cuda

# Evaluate SPN quality
python tests/benchmarks/evaluate_spn.py \
    --model_path results/fedcdh_model.pkl \
    --output_dir results/spn_quality/
```

**Acceptance Criteria**:
- Global SPN MMD p-value > 0.05 ✅
- No "not converged" warnings ✅
- Runtime acceptable (< 5 min on GPU) ✅

**If Criteria Not Met**:
1. Increase epochs: 100 → 150 → 200
2. Increase architecture: num_sums 20 → 25 → 30
3. Check data quality (standardize, remove outliers)

---

## 6. Common Issues & Solutions

### Issue #1: Global SPN MMD p-value < 0.05

**Symptom**:
```
Global SPN:
  MMD² (×10⁻³): 12.45
  MMD p-value: 0.002 ❌ (threshold: >0.05)
```

**Possible Causes**:
1. **Insufficient training epochs** → Increase to 150
2. **FedCDH clustering bypassed** → Don't use manual SPN training (use `FedCDH.fit()`)
3. **Wrong aggregation strategy** → Verify vertical=Product, horizontal=Mixture
4. **Context column handling** → Check U is appended correctly

**Debugging**:
```python
# Check local SPNs first
for k, local_spn in enumerate(fedcdh_model.local_spns):
    result = evaluator.evaluate_local_spn(local_spn, client_id=k)
    print(f"Client {k}: MMD p={result['mmd_pvalue']:.3f}")

# If locals are good but global is bad → Aggregation issue
# If locals are bad → Training issue
```

---

### Issue #2: High Overfitting Gap (> 0.50)

**Symptom**:
```
Local SPN (Client 0):
  Train LL: -5.231
  Test LL: -8.945
  Overfitting gap: 0.710 ❌
```

**Possible Causes**:
1. **Too complex architecture** for small data
2. **Insufficient regularization**
3. **Too many epochs** (overfitting)

**Solutions**:
```python
# Option 1: Reduce architecture
num_sums = 15  # Was 20
num_leaves = 15  # Was 20

# Option 2: Increase L1 regularization
l1_weight = 5e-4  # Was 1e-4

# Option 3: Early stopping based on validation LL
# (Not implemented, would require code change)
```

---

### Issue #3: KS Test Fails > 50% Dimensions

**Symptom**:
```
Local SPN (Client 0):
  KS test: 4/6 dimensions failed (67%) ❌
  Failed dimensions: [0, 1, 4, 5]
```

**Possible Causes**:
1. **Leaf distributions wrong** (Gaussian when data is non-Gaussian)
2. **Standardization issues** (SPNs expect ~N(0,1) data)
3. **Architecture too small** (can't model complex marginals)

**Solutions**:
```python
# Check data preprocessing
print(f"Data mean: {X_train.mean(axis=0)}")  # Should be ~0
print(f"Data std: {X_train.std(axis=0)}")    # Should be ~1

# Ensure standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Increase num_leaves (controls marginal quality)
num_leaves = 30  # Was 20
```

---

## 7. Performance Checklist

### Before Running Full Thesis Experiments

- [ ] **Convergence**: No warnings in smoke test logs
- [ ] **Global SPN**: MMD p-value > 0.05 on d=8 test
- [ ] **Runtime**: < 5 min per run on GPU (acceptable for 170 runs)
- [ ] **Overfitting**: Gap < 0.50 for local SPNs
- [ ] **Architecture**: Validated on d=8 (thesis target)

### During Experiments

- [ ] **Monitor logs**: Check for convergence warnings in real-time
- [ ] **Spot check**: Evaluate SPN quality on first 3 seeds
- [ ] **Early abort**: If MMD p < 0.05 consistently, stop and retune

### After Experiments

- [ ] **Quality report**: Run `evaluate_spn.py` on final models
- [ ] **Thesis appendix**: Include MMD/KS results as validation
- [ ] **Ablation**: Compare SPN quality across scenarios (H/V/Hy)

---

## 8. What's Missing (Future Work)

### Not Implemented

❌ **Validation-based early stopping**
- Would stop training when validation LL plateaus
- Prevents overfitting automatically
- Requires splitting data into train/val/test (currently only train/test)

❌ **Adaptive architecture selection**
- Would tune num_sums/num_leaves based on data
- Could use cross-validation or BIC
- Currently uses fixed hyperparameters

❌ **Cross-scenario SPN comparison**
- Would compare SPN quality: horizontal vs vertical vs hybrid
- Useful for "vertical regularization" hypothesis validation
- Framework supports this, but not automated

---

## Summary: Ensuring SPN Performance

**Three-Level Guarantee**:

1. **Training-time** (FedPC.py):
   - ✅ Loss convergence monitoring (last 20 epochs check)
   - ✅ Automatic warnings if not converged

2. **Post-training** (evaluate_spn.py):
   - ✅ MMD p-value (gold standard, must pass)
   - ✅ Overfitting gap (< 0.50 acceptable)
   - ✅ KS test (marginal validation)
   - ✅ Convergence analysis (if losses provided)

3. **Experiment-time** (your workflow):
   - ✅ Smoke test before full runs
   - ✅ Spot checks during experiments
   - ✅ Quality report after experiments

**Bottom Line**: Global SPN MMD p-value > 0.05 is the **minimum bar**. If this passes, causal discovery can proceed. Local SPNs can underperform as long as global aggregation compensates.

**Current Status**: Framework implemented and validated. Ready for thesis experiments once hyperparameters are tuned on d=8 representative data.
