# V3 Quick Reference

**Status**: ✅ CRITICAL FIXES IMPLEMENTED - Testing Phase
**Date**: 2026-05-03

---

## TL;DR

🎉 **Good News**: V3 critical fixes are already implemented!
⏰ **Time Saved**: 6 hours
🎯 **Next**: Test to verify improvements

---

## What's Implemented ✅

### 1. GlobalSumOfProducts (Hybrid Fix)
- **Location**: `causallearn/utils/FedPC.py:1621-1801`
- **What**: Sum-over-products breaks independence between feature groups
- **Expected**: Cross-group F1: 0.000 → 0.3-0.7
- **Test**: `tests/run_hybrid_ci_ranking_test.py`

### 2. Horizontal Aggregation Strategies
- **Location**: `causallearn/search/FCMBased/FedCDH/FedCDH.py:779-880`
- **Strategies**:
  1. `structure_voting` (RECOMMENDED) - Democratic edge voting
  2. `ll_weighted` - Quality-based mixing
  3. `mixture` (baseline) - Simple averaging
- **Expected**: Global F1: 0.000 → 0.3+
- **Test**: Create comparison script

### 3. Vertical Validation (Partial)
- **Status**: ⚠️ Needs 1-2 hours to add validation
- **What**: Ensure min 4 features per client
- **Priority**: LOW

---

## Quick Start: Testing

```bash
cd /Users/M279402/PycharmProjects/fl_spn_CDH

# 1. Test hybrid mode
python tests/run_hybrid_ci_ranking_test.py

# 2. Test horizontal aggregation
# (create test script comparing 3 strategies)

# 3. Git commit with results
git add -A
git commit -m "docs(v3): verify critical fixes + test results"
```

---

## Configuration

```python
# Hybrid mode (sum-over-products active by default)
args.scenario = "hybrid"

# Horizontal mode with structure voting
args.scenario = "horizontal"
args.horizontal_aggregation = "structure_voting"
args.structure_vote_threshold = 0.5

# Vertical mode (add validation later)
args.scenario = "vertical"
# args.min_features_per_client = 4  # TODO: implement
```

---

## File Locations

| Component | File | Lines |
|-----------|------|-------|
| GlobalSumOfProducts | `FedPC.py` | 1621-1801 |
| Hybrid integration | `FedCDH.py` | 924-1054 |
| Horizontal aggregation | `FedCDH.py` | 779-880 |
| Structure voting utils | `structure_aggregation.py` | Full file |

---

## Timeline

| Phase | Status | Hours | Next |
|-------|--------|-------|------|
| Phase 1: Testing | 🎯 NOW | 4-6 | Verify fixes work |
| Phase 2: Baselines | ⬜ | 12-16 | PC/GES/FCI comparison |
| Phase 3: Real-World | ⬜ | 12-16 | Sachs, Law School |
| Phase 4: Ablations | ⬜ | 24-30 | n, d, K studies |
| Phase 5: Report | ⬜ | 14-18 | Final thesis materials |

**Total Remaining**: 66-88 hours (~3 weeks)

---

## Expected Results

### Before V3 (V2)
- Hybrid cross-group: F1 = 0.000 ❌
- Horizontal global: F1 = 0.000 ❌
- Vertical: High variance ⚠️

### After V3 (Expected)
- Hybrid cross-group: F1 > 0.3 ✅
- Horizontal global: F1 > 0.3 ✅
- Vertical: Stable metrics ✅

---

## Next Steps (Priority Order)

1. **TODAY**: Run hybrid test → Verify cross-group F1 > 0.3
2. **THIS WEEK**: Test horizontal strategies → Select best
3. **NEXT WEEK**: Sachs dataset → Real-world validation
4. **NEXT 2 WEEKS**: Ablations → Comprehensive evaluation

---

**Questions?** See `V3_IMPLEMENTATION_STATUS.md` for detailed documentation
