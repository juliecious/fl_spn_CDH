# Quick Summary: Hyperparameter Tuning & Threshold Calibration Proposal

## 🎯 Core Problem

FedSPN methods fail on law_school dataset while GES/FCI achieve perfect performance:
- **GES/FCI**: F1 = 1.0
- **FedSPN_h**: F1 = 0.0 (CUDA OOM)
- **FedSPN_v**: F1 = 0.22
- **FedSPN_hy**: F1 = 0.0

## 🔍 Root Causes Identified

### 1. Hyperparameter System Issues
- ✗ Ignores dataset sparsity (law school has only 7/10 possible edges)
- ✗ Doesn't consider feature correlation strength
- ✗ Fixed breakpoints (ratio 50, 100, 200) not optimal for all datasets
- ✗ No validation-based tuning

### 2. Threshold System Issues
- ✗ Fixed 0.4 threshold regardless of:
  - Client count (K=3 vs K=10 have different voting dynamics)
  - Data heterogeneity (IID vs non-IID clients)
  - Graph sparsity (sparse graphs need different thresholds)
- ✗ Binary voting (ignores p-value strength)
- ✗ Ceiling effect: threshold=0.4 with K=3 → actual threshold=0.667

## 💡 Proposed Solutions

### Solution 1: Dataset Complexity Profiling
Add pre-training analysis to measure:
- Correlation strength (weak vs strong dependencies)
- Effective dimensionality (redundant vs independent features)
- Noise level (clean vs noisy data)
- Expected sparsity (few vs many edges)

**Impact:** Adapt capacity & regularization to data characteristics

### Solution 2: Enhanced Hyperparameters
Add **Criterion 6** to existing 5-criterion system:
```
If high correlation strength (>0.5):
  → Increase capacity by 1.3×

If high noise (>0.3):
  → Double regularization
  → Add 0.05 dropout
```

**Impact:** Better fit for complex/simple datasets

### Solution 3: Adaptive Threshold Calibration
Replace fixed 0.4 with formula:
```
threshold = 0.5
          - 0.2 × heterogeneity        # Lower for non-IID
          + 0.15 × sparsity             # Higher for sparse graphs

Clamped to [0.3, 0.8]
```

**Example:**
- Law school (sparse, moderate heterogeneity): threshold = 0.485
- Sachs (sparse, high heterogeneity): threshold = 0.504

**Impact:** Dataset-specific precision/recall tradeoff

### Solution 4: Confidence-Weighted Voting
Instead of binary votes:
```
vote_weight = 1.0 - p_value

Edge included if: mean(vote_weights) ≥ threshold
```

**Impact:** Strong evidence (p=0.001) counts more than weak (p=0.049)

### Solution 5: Two-Phase Training
**Phase 1 (20% epochs):** Explore with high capacity, collect diagnostics
**Phase 2 (80% epochs):** Adjust based on overfitting/underfitting

**Impact:** Adapt to actual training dynamics

## 📊 Expected Improvements

### Conservative Estimates
| Dataset | Current F1 | Target F1 | Gain |
|---------|-----------|-----------|------|
| law_school (H) | 0.00 | 0.45-0.60 | +45-60 pts |
| law_school (V) | 0.22 | 0.50-0.65 | +28-43 pts |
| Average | ~0.15 | ~0.45 | +30 pts |

### Optimistic Estimates
- Law school: F1 = 0.7-0.8 (near GES/FCI)
- Average across datasets: F1 > 0.6
- Zero failures (currently 33% failure rate)

## 🗓️ Implementation Plan

### Phase 1: Profiling (Week 1)
- Implement `profile_dataset_complexity()`
- Collect baselines for all 12 datasets
- **Deliverable:** Complexity profiles

### Phase 2: Hyperparameters (Week 2)
- Add Criterion 6 to `compute_adaptive_hyperparameters()`
- Validate on 3 datasets
- **Deliverable:** Improved F1 on law_school

### Phase 3: Adaptive Thresholds (Week 3)
- Implement `calibrate_structure_threshold()`
- Add confidence-weighted voting
- **Deliverable:** Threshold sensitivity analysis

### Phase 4: Validation (Week 4)
- Full benchmark: 12 datasets × 6 methods × 3 seeds
- Statistical significance testing
- **Deliverable:** Publication-ready results

**Total:** 4 weeks, 90 hours

## ✅ Success Criteria

### Minimum (Must Have)
- ✓ Law school F1 > 0.4 for all FedSPN variants
- ✓ No CUDA OOM failures
- ✓ Improvement on ≥8/12 datasets

### Target (Should Have)
- ✓ Average F1 improvement: +20 points
- ✓ Law school F1 > 0.6
- ✓ Reduced variance across datasets

### Stretch (Could Have)
- ✓ Law school F1 > 0.8 (match GES/FCI)
- ✓ Outperform FedCDH baseline on ≥5 datasets
- ✓ Publication-quality results

## 🚀 Recommendation

**Proceed with Phases 1-2 immediately:**
1. Profiling enables all other improvements
2. Hyperparameter tuning has biggest expected impact
3. Low risk, high reward

**Defer Phase 3 if resources limited:**
- Threshold tuning is medium impact
- Can be added later as enhancement

## 📁 Key Files

**Proposal Details:** `PROPOSAL_HYPERPARAMETER_TUNING.md` (full specification)
**This Summary:** `PROPOSAL_SUMMARY.md` (you are here)
**CUDA Fix:** `FIXES_SUMMARY.md` (already implemented)

## 🔗 Next Actions

1. ☐ Review proposal with team
2. ☐ Approve Phase 1 budget (20 hours)
3. ☐ Run GPU benchmark on law_school to confirm current baseline
4. ☐ Start Phase 1 implementation

---

**Status:** 📝 Proposal (Not Implemented)
**Priority:** 🔴 High (addresses critical performance gap)
**Estimated Impact:** 📈 +30 percentage points F1
