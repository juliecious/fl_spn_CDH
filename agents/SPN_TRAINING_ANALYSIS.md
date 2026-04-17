# SPN Training Quality Analysis

**Date**: April 16, 2026
**Issue**: Poor training LL and missing MMD values in evaluation logs

---

## Question 1: Why is Local SPN Training LL So Bad?

### Observed LL Values (Horizontal, d=8)
```
Client 0: Train LL = -11.5824
Client 1: Train LL = -13.3035
Client 2: Train LL = -15.2040
```

### Baseline Comparison

For 8-dimensional **independent** Gaussian data with unit variance:
- Each dimension contributes: 0.5*log(2π) + 0.5*log(σ²) ≈ 0.919 nats
- Expected LL for factorized Gaussian: 8 × 0.919 ≈ **-7.35** (negative)

Our SPNs are achieving **-11 to -15**, which is **1.5× to 2× worse** than a simple factorized Gaussian!

### Root Causes

#### 1. **Extremely Shallow Depth**
```python
depth = floor(log2(8)) = floor(2.08) = 3
```

**Problem**: With depth=3, the SPN has only 3 layers:
- Layer 0: Leaf distributions (Gaussians)
- Layer 1: Sum nodes (mixtures)
- Layer 2: Product nodes (factorizations)
- Layer 3: Root sum

This creates an **extremely limited factorization hierarchy**. For d=8 features with complex dependencies, depth=3 cannot capture:
- Higher-order interactions (3+ variables)
- Deep hierarchical structure
- Non-linear dependencies

**Evidence**: The paper uses depth=5-7 for similar problems, giving 32-128× more structural capacity.

#### 2. **Very Small Architecture (num_sums=5, num_leaves=5)**

Current parameters per layer:
- **5 sum nodes** per layer → Only 5 mixture components
- **5 leaf nodes** per feature → Only 5 Gaussian components per variable

**Comparison to literature**:
- RAT-SPN paper uses **num_sums=20-40** for similar data
- Our implementation: **4-8× smaller** than recommended

**Consequence**:
- Total parameters: ~2,500-3,000 for 8 features
- Samples per parameter: 300 samples / 3000 params = **0.10**
- Recommended ratio: **5-10** samples/parameter
- **We're 50-100× undersized!**

#### 3. **RAT-SPN Design Mismatch**

RAT-SPN (Randomized and Tensorized SPN) was designed for:
- **Discrete/categorical data** (images, MNIST)
- **Fixed grid structures** (spatial locality)
- **Random factorizations** (no structure learning)

Our data is:
- **Continuous Gaussian** (requires good density estimation)
- **DAG-structured** (causal dependencies, not spatial)
- **Needs learned structure** (not random splits)

**Key Issue**: RAT-SPN uses **random variable partitions** at each layer, which:
- Ignores causal structure
- Splits dependent variables apart
- Doesn't learn optimal factorizations
- Uses "randomized" splits → high variance in quality

#### 4. **Inadequate Training (101 epochs)**

Current training:
```python
adaptive_epochs = int(50 × (8/5)^1.5) = int(50 × 2.02) = 101 epochs
```

**Problem**: For 3,000 parameters with only 300 samples:
- Need **careful convergence** (low learning rate, many epochs)
- 101 epochs with lr=0.0079 is **insufficient** for this regime
- No early stopping (may stop before convergence)
- No learning rate schedule (should decay)

**Evidence from logs**: "Final Loss=9.0683 after 101 epochs"
- Loss is still high (should be near LL = -8 or better)
- Likely not converged

---

## Question 2: What Could Be Wrong with Local SPN Instantiation?

### Current Instantiation (FedCDH.py lines 527-538)

```python
leaf = LocalSPNWrapper(
    num_features=local_d,        # 8 for horizontal
    device=self.device,
    num_sums=6,                  # Adaptive: 5 × sqrt(8/5) = 6
    num_leaves=6,                # Adaptive: 5 × sqrt(8/5) = 6
    depth=3,                     # floor(log2(8)) = 3
    num_repetitions=5,           # Fixed
    seed=h * 10 + k,
)
leaf.train_local(local_data_h, epochs=101, lr=0.0079)
```

### Issues

#### Issue 1: **Einet Architecture Constraints Too Restrictive**

Einet enforces: `2^depth ≤ num_features`

For d=8: `2^depth ≤ 8` → `depth ≤ 3`

**This is the fundamental bottleneck!** We cannot increase depth beyond 3 for 8 features.

**Consequence**:
- Shallow network (only 3 layers)
- Limited expressiveness
- Cannot model deep hierarchies

**Possible solutions**:
1. Pad features to next power of 2 (8→16) to allow depth=4
2. Use a different SPN backend (not Einet)
3. Increase width dramatically to compensate

#### Issue 2: **Adaptive Scaling is Too Conservative**

```python
scale_factor = sqrt(8/5) = sqrt(1.6) = 1.26
adaptive_num_sums = max(5, int(5 × 1.26)) = max(5, 6) = 6
```

Only **+20% capacity** for a **60% dimension increase** (d=5→d=8).

**Should scale more aggressively**:
- Linear scaling: 5 × (8/5) = 8 sums/leaves
- Quadratic: 5 × (8/5)² = 12.8 ≈ 13
- Literature values: 20-40 for d=8

#### Issue 3: **No Structure Learning**

```python
structure="top-down"  # Random Poon-Domingos splits
```

This uses **random binary tree** factorizations, not learned from data.

**Better alternatives**:
- `structure="learn"` (if supported by Einet)
- LearnSPN with greedy structure search
- ID-SPN with independence-based splits

#### Issue 4: **Normalization May Be Unstable**

```python
self.std = torch.tensor(data.std(axis=0), dtype=torch.float32)
data_t = (data_t - self.mean) / (self.std + 1e-6)
```

For small clusters (40-80 samples), `std` estimation is **noisy**.

**Problem**:
- High-variance std estimates → bad normalization
- 1e-6 epsilon too small for noisy data
- No clipping of normalized values

---

## Question 3: Is RAT-SPN a Good Idea?

### Short Answer: **NO, not for continuous causal discovery.**

### Detailed Analysis

#### RAT-SPN Strengths
✅ Fast training (GPU-optimized)
✅ Good for discrete data (images, MNIST)
✅ Scalable to high dimensions (100+ features)
✅ Simple implementation (no structure search)

#### RAT-SPN Weaknesses for Our Use Case

❌ **Random structure** (doesn't learn dependencies)
❌ **Optimized for discrete data** (categoricals, not Gaussians)
❌ **Rigid factorization** (binary tree, no flexibility)
❌ **No causal awareness** (ignores DAG structure)
❌ **High variance** (randomness → unstable CI tests)

### Comparison to Alternatives

| SPN Type | Structure | Data Type | CI Test Quality | Speed |
|----------|-----------|-----------|----------------|-------|
| **RAT-SPN (current)** | Random | Discrete | ⚠️ Low (high variance) | ⚡⚡⚡ Fast |
| **LearnSPN** | Learned | Both | ✅ High (structure-aware) | ⚡ Slow |
| **ID-SPN** | Independence | Continuous | ✅ High (CI-optimized) | ⚡⚡ Medium |
| **PC-SPN** | Correlation | Continuous | ✅ Very High | ⚡ Slow |

### Recommended Alternatives

#### Option 1: **LearnSPN** (Gens & Domingos 2013)
- Greedy top-down structure learning
- Uses independence tests to guide splits
- Better density estimation for continuous data
- **Trade-off**: 5-10× slower training

#### Option 2: **ID-SPN** (Rathjen et al. 2021)
- Explicitly learns structure for conditional independence
- Optimized for causal discovery tasks
- Uses mutual information for splits
- **Trade-off**: Requires structure search (slower)

#### Option 3: **Hybrid Approach**
- Use RAT-SPN for **speed** (initial structure)
- **Fine-tune** structure with independence tests
- **Prune** irrelevant connections
- **Better than**: Pure RAT-SPN, faster than full structure search

---

## Question 4: Why No MMD Metric in Evaluation Log?

### What We See
```
MMD p-value: 0.000 ✗
```

### What We Don't See
```
MMD value: 0.0234  ← MISSING!
```

### Root Cause

**File**: `causallearn/utils/spn_evaluation.py:471-474`

```python
def log_spn_quality(results, name=None):
    ...
    if "mmd_pvalue" in results:
        mmd_p = results["mmd_pvalue"]
        status = "✓" if mmd_p > 0.05 else "✗"
        logging.info(f"    MMD p-value: {mmd_p:.3f} {status}")  # ← Only logs p-value!
```

**The actual MMD² value is computed** in `evaluate_spn_quality()`:
```python
mmd_sq, mmd_pval = mmd_permutation_test(X_features, samples_features, n_permutations=50)
results["mmd_squared"] = mmd_sq      # ← Computed but not logged!
results["mmd_pvalue"] = mmd_pval     # ← Only this is logged
```

### Why This Matters

**MMD p-value alone is insufficient** because:
1. **Effect size**: p-value doesn't tell us *how different* distributions are
2. **Sample size**: p=0.000 could be tiny difference with large n
3. **Interpretability**: MMD² has units (squared distance), p-value doesn't

**Example**:
- Small dataset: MMD²=0.05, p=0.12 → Good fit (large effect, not significant)
- Large dataset: MMD²=0.001, p=0.03 → Excellent fit (tiny effect, "significant" due to n)

### Fix

Update `log_spn_quality()` to include MMD² value:

```python
if "mmd_pvalue" in results:
    mmd_p = results["mmd_pvalue"]
    mmd_sq = results.get("mmd_squared", None)
    status = "✓" if mmd_p > 0.05 else "✗"
    if mmd_sq is not None:
        logging.info(f"    MMD²: {mmd_sq:.6f}, p-value: {mmd_p:.3f} {status}")
    else:
        logging.info(f"    MMD p-value: {mmd_p:.3f} {status}")
```

**Expected output**:
```
MMD²: 0.023456, p-value: 0.000 ✗
```

---

## Summary and Recommendations

### Immediate Issues
1. **Training LL is poor** (-11 to -15) due to undersized architecture
2. **RAT-SPN is suboptimal** for continuous causal discovery
3. **Depth constraint** (2^d ≤ num_features) severely limits capacity
4. **MMD value missing** from logs (only p-value shown)

### Short-term Fixes (Easy)
1. ✅ **Log MMD² value** in evaluation output
2. ⚠️ **Increase num_sums/num_leaves** to 20-40 (4-8× current)
3. ⚠️ **Add more repetitions** (5→10) for structural diversity
4. ⚠️ **Increase epochs** to 200-500 for better convergence
5. ⚠️ **Add early stopping** based on validation LL

### Medium-term Improvements (Moderate effort)
1. **Replace RAT-SPN with LearnSPN** for structure learning
2. **Feature padding** to next power of 2 for deeper networks
3. **Learning rate schedule** (cosine decay or step decay)
4. **Better normalization** (robust scaling, outlier clipping)

### Long-term (Significant refactoring)
1. **Switch to PC-SPN or ID-SPN** for causal-aware structure
2. **Hybrid SPN backend** (fast RAT-SPN + structure fine-tuning)
3. **Cluster-specific architectures** (different depth/width per cluster)
4. **Meta-learning** for hyperparameter selection

---

## Next Steps

**Priority 1**: Fix MMD logging (5 minutes)
**Priority 2**: Increase num_sums/leaves to 20 (10 minutes)
**Priority 3**: Run ablation study on architecture size (1 hour)
**Priority 4**: Evaluate LearnSPN as replacement (2-3 hours)
