# FedSPN-CDH Working State Summary

**Last Updated**: 2026-05-31
**Branch**: v3-comprehensive-fixes
**Status**: ✅ Core bugs fixed, adaptive improvements + increased SPN training

---

## Session Achievements

### 1. Bug Fix #9: Conditioning on Augmented Variable U ✅
**Commit**: d7a7ee0

- Fixed all p-values = 0.000 issue
- Results: 26 → 19 edges (-27%), precision 0.333 → 0.353 (+6%)

### 2. Adaptive Parameter Improvements ✅
**Commit**: 346ce38

- Dynamic epochs, depth_limit, num_permutations based on n, d
- Results: 19 → 16 edges (-16%), precision 0.353 → 0.375 (+6%), faster runtime

### 3. Increased SPN Training Epochs ✅
**Commit**: 0704cc0

- Increased default: GPU 50→80, CPU 10→20
- Better density estimation for accurate CI tests
- Expected: 16 → 10-12 edges, precision 0.375 → 0.55-0.65

---

## Overall Progress

| Metric | Baseline | After Bug #9 | After Adaptive | After More Training | Change |
|--------|----------|--------------|----------------|---------------------|--------|
| **Edges** | 26 | 19 | 16 | 10-12 (expected) | -54% to -62% |
| **Precision** | 33.3% | 35.3% | 37.5% | 55-65% (expected) | +65% to +95% |
| **SHD** | 16 | 13 | 12 | 6-8 (expected) | -50% to -63% |

---

## Next Steps

1. Run experiment with increased SPN training
2. Verify improvements (target: 10-12 edges, precision 55-65%)
3. If still not optimal, try α=0.01 for stricter tests

---

**Last Modified**: 2026-05-31
