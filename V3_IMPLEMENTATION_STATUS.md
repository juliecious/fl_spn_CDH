# V3 Implementation Status Report

**Date**: 2026-05-03
**Branch**: `v3-comprehensive-fixes`
**Status**: ✅ CRITICAL FIXES IMPLEMENTED - Ready for Testing

---

## Executive Summary

**Key Discovery**: The critical fixes from the V3 roadmap were **already implemented**! This verification saves 6 hours of implementation work.

### What's Done ✅

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| GlobalSumOfProducts | ✅ Complete | `FedPC.py:1621-1801` | Hybrid mode integrated |
| Horizontal Aggregation | ✅ Complete | `FedCDH.py:779-880` | 3 strategies ready |
| Structure Aggregation Utils | ✅ Complete | `structure_aggregation.py` | Voting + LL-weighting |
| Hybrid Sum-over-Products | ✅ Complete | `FedCDH.py:924-1054` | Cluster combinations |

### What Needs Work ⚠️

1. **Testing** (4-6 hours) - IMMEDIATE PRIORITY
   - Verify hybrid cross-group F1 > 0.3 (from 0.000)
   - Compare horizontal aggregation strategies
   - Measure performance improvements

2. **Vertical Validation** (1-2 hours) - LOW PRIORITY
   - Add `min_features_per_client` check
   - Ensure ≥4 features per client

---

## Implementation Details

### 1. GlobalSumOfProducts (Fix #1) ✅

**Problem Solved**: Hybrid mode had F1=0.000 for cross-group dependencies because it used pure product factorization, which enforces independence.

**Solution Implemented**:
```python
# OLD (V2): Product only → enforces independence
P(X) = P(X_g1) × P(X_g2) × P(X_g3)
→ I(X_g1; X_g2) = 0 always ❌

# NEW (V3): Sum-over-products → can model dependencies
P(X) = Σ_c w_c × ∏_g P(X_g | cluster_config_c)
→ I(X_g1; X_g2) ≠ 0 via coupling ✅
```

**Implementation**:
- Class: `GlobalSumOfProducts` (FedPC.py:1621-1801)
- Features:
  - Logsumexp for numerical stability
  - Pre-computed log weights
  - Sampling via mixture component selection
  - Proper PyTorch module registration
- Integration: Hybrid mode (FedCDH.py:924-1054)
  - Samples cluster combinations
  - Builds products per combination
  - Wraps in GlobalSumOfProducts

**Expected Impact**: Cross-group F1: 0.000 → 0.3-0.7

**Testing Required**:
```bash
# Run hybrid CI test
python tests/run_hybrid_ci_ranking_test.py

# Expected: CI tests return p < 1.0 (not always 1.0)
# Expected: Cross-group F1 > 0.3
```

---

### 2. Horizontal Aggregation Strategies (Fix #2) ✅

**Problem Solved**: Horizontal mode had global F1=0.000 despite local F1=0.26-0.57 because simple mixture averaging dilutes causal structure.

**Solutions Implemented**:

#### Strategy 1: Structure Voting (RECOMMENDED)
```python
args.horizontal_aggregation = "structure_voting"
args.structure_vote_threshold = 0.5  # 50% majority
```

**How it Works**:
1. Extract local dependency graphs from each client's SPN
2. Vote on edges: Include if ≥threshold clients detect it
3. Build consensus graph with confidence scores
4. Use standard mixture for CI inference

**Justification**:
- ✅ Preserves causal structure (democratic, not averaging)
- ✅ Robust to outliers (single bad client can't destroy structure)
- ✅ Confidence tracking (vote proportions)

**Implementation**:
- Main logic: FedCDH.py:796-848
- Utils: `structure_aggregation.py`
  - `extract_local_dependency_graph()` - CI tests on local SPNs
  - `aggregate_structures_by_voting()` - Majority voting
  - `log_structure_aggregation_summary()` - Logging

#### Strategy 2: LL-Weighted Mixing
```python
args.horizontal_aggregation = "ll_weighted"
```

**How it Works**:
1. Compute train log-likelihood for each client
2. Convert to quality weights: `exp(ll/10)`
3. Weight mixture by quality (better SPNs → higher influence)

**Justification**:
- ✅ Quality-aware (rewards better models)
- ✅ Automatic (no threshold tuning)
- ⚠️ Risk: Could amplify overfitting

**Implementation**: FedCDH.py:850-867

#### Strategy 3: Default Mixture (V2 Baseline)
```python
args.horizontal_aggregation = "mixture"  # default
```

Simple sample-weighted averaging (original V2 behavior).

**Expected Impact**: Global F1: 0.000 → 0.3+

**Testing Required**:
```bash
# Test all 3 strategies on Linear SMALL Horizontal
python tests/test_horizontal_aggregation.py

# Compare:
# - structure_voting (should be best)
# - ll_weighted
# - mixture (baseline)

# Measure: Global F1, edge preservation, CI accuracy
```

---

### 3. Vertical Feature Validation (Fix #3) ⚠️

**Problem**: Some clients have only 1-2 features → only 1 pairwise test → high variance

**Current Status**:
- ✅ Data partition validation exists (FedCDH.py:379-403)
- ❌ No explicit min_features_per_client check

**Required Implementation** (1-2 hours):
```python
# Add to FedCDH.__init__
self.min_features_per_client = getattr(args, "min_features_per_client", 4)

# Add validation in fit() for vertical mode
if self.scenario == "vertical":
    for k, f_indices in feature_maps.items():
        num_features = len([idx for idx in f_indices if idx < self.d_features])
        if num_features < self.min_features_per_client:
            raise ValueError(
                f"Client {k} has only {num_features} features, "
                f"need at least {self.min_features_per_client} for reliable CI tests"
            )
        logging.info(f"Client {k}: {num_features} features → {num_features*(num_features-1)//2} pairwise tests")
```

**Expected Impact**: Minimum tests/client: 1-2 → 6+

---

## Next Steps

### Phase 1: Testing (4-6 hours) 🎯 IMMEDIATE

**Day 1: Hybrid Mode Testing**
```bash
# 1. Run hybrid CI ranking test
cd /Users/M279402/PycharmProjects/fl_spn_CDH
python tests/run_hybrid_ci_ranking_test.py

# Expected results:
# - Cross-group F1 > 0.3 (from 0.000) ✓
# - CI tests return p < 1.0 (not always 1.0) ✓
# - Dense-local F1 ~ 1.0 maintained ✓

# 2. Document results in working_state.md
```

**Day 2: Horizontal Aggregation Comparison**
```bash
# Test all 3 strategies on Linear SMALL
# Create test script if doesn't exist:
# tests/test_horizontal_aggregation.py

# Compare:
# 1. structure_voting (recommended)
# 2. ll_weighted
# 3. mixture (baseline)

# Metrics: Global F1, edge preservation, consensus quality

# Select best method for thesis
```

**Day 3: Vertical Validation (Optional)**
```bash
# Add min_features_per_client validation
# Test on Vertical LARGE (d=11, K=5)
# Verify ≥6 tests/client
```

**Day 4: V2 vs V3 Comparison**
```bash
# Run SMALL config with both versions
# Generate comparison table
# Document improvements
```

**Day 5: Git Commit & Documentation**
```bash
# Commit all changes
# Update working_state.md with test results
# Create V3_TEST_RESULTS.md
# Tag: v3-fixes-verified
```

---

### Phase 2: Baseline Evaluation (12-16 hours)

1. Centralized methods (PC, GES, FCI) on Sachs
2. Compare federated vs centralized performance gap
3. Run original FedCDH comparison

---

### Phase 3: Real-World Datasets (12-16 hours)

1. Sachs protein network (priority)
2. Law School admissions dataset
3. HyperPC synthetic benchmarks

---

### Phase 4: Ablation Studies (24-30 hours)

1. Sample size ablation (n: 300-3600)
2. Dimensionality ablation (d: 5-30)
3. Number of clients ablation (K: 2-10)

---

### Phase 5: Final Report (14-18 hours)

1. V3 comprehensive HTML report
2. Publication-ready figures
3. Thesis tables and summaries

---

## Timeline Update

**Original Estimate**: 78-100 hours (4 weeks)
**Revised Estimate**: 72-94 hours (3 weeks)
**Time Saved**: 6 hours (implementation already done!)

**Critical Path** (minimum viable):
- Testing (4 hrs) + Sachs (6 hrs) + Sample ablation (10 hrs) + Report (6 hrs) = **26 hours** (~4 days)

---

## Configuration Reference

### To Use V3 Fixes

```python
# In your experiment script:
import argparse

args = argparse.Namespace()

# For Hybrid mode: Use sum-over-products (automatically active)
args.scenario = "hybrid"
# No config needed - GlobalSumOfProducts is default

# For Horizontal mode: Use structure voting (recommended)
args.scenario = "horizontal"
args.horizontal_aggregation = "structure_voting"  # Options: "structure_voting", "ll_weighted", "mixture"
args.structure_vote_threshold = 0.5  # 50% of clients must agree

# For Vertical mode: Add validation (after implementing)
args.scenario = "vertical"
args.min_features_per_client = 4  # Ensure ≥6 pairwise tests
```

---

## Files Modified

| File | Lines | Status | Changes |
|------|-------|--------|---------|
| `causallearn/utils/FedPC.py` | 1621-1801 | ✅ Complete | GlobalSumOfProducts class |
| `causallearn/utils/FedPC.py` | Various | ✅ Complete | sample_cluster_combinations helper |
| `causallearn/search/FCMBased/FedCDH/FedCDH.py` | 924-1054 | ✅ Complete | Hybrid mode integration |
| `causallearn/search/FCMBased/FedCDH/FedCDH.py` | 779-880 | ✅ Complete | Horizontal aggregation |
| `causallearn/utils/structure_aggregation.py` | Full file | ✅ Complete | Structure voting utils |
| `causallearn/search/FCMBased/FedCDH/FedCDH.py` | TBD | ⚠️ Needed | Vertical validation |

---

## Questions?

1. **How do I test the fixes?**
   - Run `tests/run_hybrid_ci_ranking_test.py` for hybrid mode
   - Create test script for horizontal aggregation comparison
   - Check metrics: F1, Precision, Recall, SHD

2. **Which horizontal strategy should I use?**
   - **Recommended**: `structure_voting` (preserves structure)
   - Test all 3 and compare results
   - Select based on global F1 performance

3. **Do I need to implement vertical validation?**
   - Priority: LOW (medium importance, but not blocking)
   - Impact: Reduces metric variance, ensures ≥6 tests/client
   - Time: 1-2 hours

4. **What's the fastest path to results?**
   - Test hybrid + horizontal (2 days)
   - Run Sachs dataset (1 day)
   - Generate comparison report (1 day)
   - **Total**: 4 days minimum viable results

---

**Next Action**: Run `tests/run_hybrid_ci_ranking_test.py` to verify hybrid mode improvements!
