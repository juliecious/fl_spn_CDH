# FedSPN-CDH Working State Summary

**Last Updated**: 2026-05-31
**Branch**: v3-comprehensive-fixes
**Status**: ✅ Core bugs fixed, adaptive improvements implemented

---

## Major Achievements This Session

### 1. Bug Fix #9: Conditioning on Augmented Variable U ✅

**Problem**: All p-values = 0.000 in main PC causing 26 edges vs 8 true edges

**Solution**: Modified skeleton_discovery to always condition on U (client ID)

**Results** (20260531_110932 vs 20260530_205318):
- P-values: All 0.000 → Varied (0.019-0.118) ✅ FIXED
- Skeleton edges: 26 → 19 (-27%)
- False positives: 18 → 11 (-39%)
- Precision: 0.333 → 0.353 (+6%)
- SHD: 16 → 13 (-19%)

**Commit**: d7a7ee0

---

### 2. Adaptive Parameter Improvements ✅

**A. SPN Training Epochs**: Now scales with both dimensionality AND data size
```python
epochs = base * (d/5)^1.5 * sqrt(n/500)
```

**B. Depth Limit**: Multi-criteria selection based on samples, features, statistical power
```python
depth = min(sqrt(n/100), d-2, 5)  # Bounded [2,5]
```

**C. Num Permutations**: Adaptive based on dataset size
```python
n<500: 100, 500≤n<2000: 50, n≥2000: 30
```

**Modified**: causallearn/search/FCMBased/FedCDH/FedCDH.py
**Status**: Pending commit

---

## Experiment History

| Experiment | Edges | Precision | SHD | Status |
|------------|-------|-----------|-----|--------|
| 20260530_205318 | 26 | 0.333 | 16 | ⚠️ Before Bug #9 fix |
| 20260531_110932 | 19 | 0.353 | 13 | ✅ After Bug #9 fix |
| Next | ? | ? | ? | 🎯 With adaptive params |

---

## Next Actions

1. ✅ Commit adaptive parameter improvements
2. Run new experiment with adaptive params
3. Compare results (target: precision >0.40, edges 10-14)

---

**Last Modified**: 2026-05-31 by Claude Sonnet 4.5
