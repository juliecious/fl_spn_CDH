# Increased SPN Training Epochs

**Date**: 2026-05-31
**Purpose**: Improve density estimator accuracy for better conditional independence tests

---

## Changes Made

### 1. **Increased Default Base Epochs**

**Previous Defaults**:
```python
train_epochs = 50 if device == "cuda" else 10
```

**New Defaults**:
```python
train_epochs = 80 if device == "cuda" else 20
```

**Rationale**:
- Better SPN quality → More accurate likelihood estimates
- More accurate CI tests → Better causal discovery
- GPU can handle the extra training time
- CPU also gets 2× increase (10 → 20)

---

### 2. **Improved Logging**

Added logging to show:
1. Base epochs being used
2. Source (user-specified vs auto-detected)
3. Device type

**Example log output**:
```
Base SPN training epochs: 80 (auto-detected default, device=cuda)
Adaptive epochs: base=80, d=8 (complexity=2.30×), n_per_client=333 (data_scale=0.82×)
  → epochs=150 (1.9×)
```

---

## Expected Impact

### For Asia Dataset (d=8, n_per_client=333)

**With User-Specified epochs=20** (previous experiments):
```
Base: 20
Adaptive: 20 × 2.30 × 0.82 ≈ 38 epochs
```

**With New Auto-Detected Default**:
```
Base: 80
Adaptive: 80 × 2.30 × 0.82 ≈ 150 epochs
```

**Increase**: 38 → 150 epochs (~4× more training)

---

### Benefits

1. **Better Density Estimation**
   - More accurate P(X | Z) estimates
   - Better conditional independence detection

2. **More Reliable P-values**
   - Reduced variance in permutation tests
   - More stable independence decisions

3. **Fewer False Positives**
   - Current: 8 false positives (16 edges vs 8 true)
   - Expected: 4-6 false positives (10-14 edges)
   - Target: 0-2 false positives (8-10 edges)

4. **Better Precision**
   - Current: 37.5% precision
   - Expected: 50-60% precision
   - Target: 60-75% precision

---

### Trade-offs

**Computational Cost**:
- Training time: ~105s → ~400s (4× increase)
- Total runtime: ~150s → ~450s (3× increase)

**Assessment**: Acceptable trade-off
- Still reasonable runtime (~7-8 minutes total)
- Accuracy improvement worth the cost
- One-time training cost for potentially much better results

---

## Adaptive Scaling Remains

The adaptive scaling formula still applies:
```python
complexity_scale = (d/5)^1.5
data_scale = sqrt(n_per_client / 500)
adaptive_epochs = base × complexity_scale × data_scale
```

**This is on top of the increased base**, so:
- Small datasets (n=200, d=5): 80 × 1.0 × 0.63 ≈ 50 epochs
- Asia (n=333, d=8): 80 × 2.30 × 0.82 ≈ 150 epochs
- Large datasets (n=1000, d=8): 80 × 2.30 × 1.41 ≈ 260 epochs

---

## How to Use

### Option 1: Use New Default (Recommended)
```python
# Don't specify epochs - will use 80 (GPU) or 20 (CPU)
fedcdh = FedCDH(num_clients=3, mode='horizontal', alpha=0.05)
fedcdh.fit(data)
```

### Option 2: Override Manually
```python
# Specify custom epochs
fedcdh = FedCDH(
    num_clients=3,
    mode='horizontal',
    alpha=0.05,
    epochs=100  # Custom value
)
fedcdh.fit(data)
```

### Option 3: Use Even More Training (For Critical Applications)
```python
fedcdh = FedCDH(
    num_clients=3,
    mode='horizontal',
    alpha=0.05,
    epochs=200  # Very thorough training
)
fedcdh.fit(data)
```

---

## Expected Results

### Next Experiment (with 80 base epochs)

**Predictions**:

| Metric | Current (20 epochs) | Expected (80 epochs) | Improvement |
|--------|---------------------|----------------------|-------------|
| **Edges** | 16 | 10-12 | -25% to -38% |
| **Precision** | 37.5% | 55-65% | +47% to +73% |
| **SHD** | 12 | 6-8 | -33% to -50% |
| **Runtime** | 150s | 400-450s | +167% to +200% |

**Assessment**: Worth the runtime cost for accuracy improvement

---

## Verification

To verify the changes are working, check the logs for:

1. **Base epochs logging**:
   ```
   Base SPN training epochs: 80 (auto-detected default, device=cuda)
   ```

2. **Adaptive scaling logging**:
   ```
   Adaptive epochs: base=80, d=8 (complexity=2.30×),
     n_per_client=333 (data_scale=0.82×) → epochs=150 (1.9×)
   ```

3. **Training progress**:
   - Should take ~4× longer than before
   - More epochs in training logs

---

## Comparison with Previous Experiments

### Experiment History

| Experiment | Base Epochs | Adaptive Epochs | Final Edges | Precision | Notes |
|------------|-------------|-----------------|-------------|-----------|-------|
| 20260530_205318 | 20 | ~38 | 26 | 0.333 | Before Bug #9 |
| 20260531_110932 | 20 | ~38 | 19 | 0.353 | After Bug #9 |
| 20260531_141416 | 20 | ~38 | 16 | 0.375 | After Adaptive |
| (Next) | **80** | **~150** | **10-12?** | **0.55-0.65?** | 🎯 More training |

---

## Technical Details

### Why More Epochs Help

1. **SPN Convergence**
   - 20-40 epochs: Basic structure learned
   - 40-80 epochs: Fine-tuning of distributions
   - 80-150 epochs: Accurate conditional distributions

2. **Complex Dependencies**
   - Asia network has 8 variables with complex interactions
   - Horizontal partitioning adds noise
   - Need more training to capture subtle patterns

3. **Conditional Independence Testing**
   - CI tests use P(X, Y | Z) from SPN
   - Accuracy of P(X, Y | Z) directly affects p-values
   - Better SPN → Better p-values → Better edge decisions

---

## Alternative: Try Both

**Fast Mode** (epochs=20, ~150s):
- Good for rapid prototyping
- Reasonable results (precision ~37%)
- Use for initial exploration

**Accurate Mode** (epochs=80, ~450s):
- Better for final results
- Higher precision (expected ~60%)
- Use for production/publication

**Thorough Mode** (epochs=200, ~1000s):
- Best possible accuracy
- Use only when accuracy is critical
- Diminishing returns after 100-150 epochs

---

## Summary

**Change**: Increased default SPN training epochs from 50 → 80 (GPU)

**Expected Benefit**:
- Better density estimation
- More accurate CI tests
- Fewer false positives (16 → 10-12 edges)
- Higher precision (37.5% → 55-65%)

**Trade-off**:
- Longer runtime (150s → 450s)
- Worth it for accuracy improvement

**Next Step**: Run experiment and verify improvement

---

**Modified File**: `causallearn/search/FCMBased/FedCDH/FedCDH.py`
- Lines 556-570: Increased default epochs and improved logging
- Adaptive scaling (lines 827-858) remains unchanged
