# V3 Comprehensive Fixes and Enhancements

**Branch**: `v3-comprehensive-fixes`
**Created**: 2026-05-02
**Status**: 🚀 Planning Phase

---

## Overview

V3 addresses critical issues identified in V2 experiments while adding real-world dataset validation. This plan consolidates findings from V2 analysis report and working_state.md backlogs.

---

## Part 1: Critical Bug Fixes (from V2 Analysis)

### 1.1 🚨 CRITICAL: Horizontal Mode F1 Dilution

**Issue**: Global SPN achieves F1=0.000 despite local SPNs detecting dependencies (F1=0.26-0.57)

**Root Cause**: When aggregating local SPNs with GroupMixture, the weighted sum over clients may be diluting the conditional independence structure.

**Evidence**:
- Linear SMALL Horizontal:
  - Local Client 0: F1=0.571, Acc=0.845
  - Local Client 1: F1=0.381, Acc=0.776
  - Local Client 2: F1=0.261, Acc=0.707
  - **Global**: F1=0.000, Acc=0.731 ❌

**Proposed Fixes**:
1. **Structure-Preserving Aggregation**: Use majority voting or intersection of local dependency graphs before SPN aggregation
2. **Conservative Weights**: Weight local SPNs by their log-likelihood quality (better SPNs get higher weight)
3. **Cluster-Aware Mixing**: Instead of uniform weights, use cluster-based weights that preserve local structures

**Testing**:
- Target: Global F1 > 0.3 for Horizontal mode
- Verify: Local F1 remains high (>0.5 average)

**Priority**: 🔴 CRITICAL (blocking thesis claims)

---

### 1.2 🟡 Vertical Mode Insufficient Edges

**Issue**: Some clients have only 1-2 test edges due to strict feature subset constraints

**Evidence**:
- Vertical LARGE: Some local SPNs test only 1-2 feature pairs
- Results in high variance metrics (one edge can swing F1 from 0 to 1)

**Root Cause**: When d=11 and K=5, some feature groups are very small (d/K ~ 2-3 variables)
- With 2 features: Only 1 pairwise test (0-1)
- With 3 features: Only 3 pairwise tests (0-1, 0-2, 1-2)

**Proposed Fixes**:
1. **Minimum Feature Constraint**: Ensure each client gets at least 4-5 features
2. **Overlapping Vertical Partitions**: Allow 10-20% feature overlap between clients
3. **Feature Overlap Validation**: Add warning if any client has <3 features

**Testing**:
- Verify: Each local SPN has ≥6 pairwise tests (minimum from 4 features)
- Track: Confusion matrix total count (TP+FP+FN+TN ≥ 10)

**Priority**: 🟡 MEDIUM

---

### 1.3 🟡 LARGE Config Confusion Matrix Dilution

**Issue**: Despite having 5 clients, some local SPNs show very low test counts (TP+FP+FN+TN < 10)

**Evidence**:
- Expected: d=11 → 55 pairwise tests total
- Actual: Some clients show only 5-8 total tests

**Root Cause**: Similar to 1.2 - feature partition creates small subsets

**Proposed Fixes**:
1. **Adaptive K**: Use K=3 for d<12, K=5 only for d≥12
2. **Confidence Interval Reporting**: Add ±CI for F1/Accuracy when test count is low (<15)
3. **Stratified Metrics**: Report separately for "well-sampled" vs "under-sampled" clients

**Testing**:
- Track: Minimum confusion matrix count across all experiments
- Alert: If any client has <10 total tests

**Priority**: 🟡 MEDIUM

---

## Part 2: Missing Feature Implementation (from working_state.md)

### 2.1 🚨 CRITICAL: Hybrid Mode Sum-over-Products

**Issue**: Hybrid mode enforces independence between feature groups due to missing top-level sum

**Current (WRONG)**:
```
P(X) = P(X_g1) × P(X_g2) × P(X_g3)  → I(X_g1; X_g2) = 0 ✗
```

**Correct (Seng's Feedback)**:
```
P(X) = Σ_c w_c × P(X_g1|c) × P(X_g2|c) × P(X_g3|c)  → Can model dependencies ✓
```

**Solution**: Implement `GlobalSumOfProducts` class that creates sum over cluster combinations

**Expected Impact**:
- Cross-group F1: 0.000 → 0.3-0.7
- Dense-local F1: 1.000 → 0.8-1.0 (maintain)

**Implementation**:
1. Create `GlobalSumOfProducts` in `causallearn/utils/FedPC.py`
2. Create `sample_cluster_combinations` helper
3. Modify hybrid mode in `causallearn/search/FCMBased/FedCDH/FedCDH.py`
4. Reuse local clusters instead of training new SPNs

**Files Modified**:
- `causallearn/utils/FedPC.py` (add GlobalSumOfProducts)
- `causallearn/search/FCMBased/FedCDH/FedCDH.py` (rewrite hybrid mode)

**Testing**:
- Cross-group CI test: Target F1 > 0.3
- Dense-local test: Maintain F1 ~ 1.0
- Verify CI tests return p<1.0 for cross-group pairs

**Priority**: 🔴 CRITICAL (author-identified bug)

---

### 2.2 🟢 SPN Architecture Optimizations (Optional)

**Status**: Backlog from working_state.md (investigated but not implemented)

**Options**:
1. **Adaptive Scaling**: `num_sums = 20 + d * 2`
   - Already implemented in V2 ✓
2. **Ensemble**: Use 5 models with different random seeds
   - Expected: Train LL -9→-7.5, CI Accuracy +10-15%
   - Cost: 5× training time (10-15 min vs 2 min)

**Decision**: Skip for V3, revisit if V3 results show F1 < 0.70

**Priority**: 🟢 LOW (optional enhancement)

---

## Part 3: Real-World Data Testing

### 3.1 Dataset Selection

**Available**:
- ✅ Sachs (11 nodes, protein signaling network)
  - Location: `tests/data/sachs.interventional.txt.gz`
  - Ground truth: Known causal structure

**To Add**:
1. **ALARM Network** (37 nodes, medical diagnosis)
   - Source: bnlearn R package
   - Ground truth available
   - Test scalability to larger graphs

2. **CHILD Network** (20 nodes, medical diagnosis)
   - Source: bnlearn R package
   - Ground truth available
   - Medium-scale benchmark

3. **Insurance Network** (27 nodes)
   - Source: bnlearn R package
   - Ground truth available

4. **Asia Network** (8 nodes, small benchmark)
   - Source: bnlearn R package
   - Quick validation test

**Priority**:
- 🔴 CRITICAL: Sachs (already available)
- 🟡 MEDIUM: CHILD (medium scale)
- 🟡 MEDIUM: ALARM (large scale)
- 🟢 LOW: Insurance, Asia

---

### 3.2 Real-World Experiment Design

**Test Scenarios**:

1. **Sachs Network (d=11, n=7,466 samples)**
   - Horizontal: K=3 clients, split samples evenly (~2,500 each)
   - Vertical: K=3 clients, split features (3-4 variables each)
   - Hybrid: K=3 clients, both partitions
   - Ground Truth: Compare against known protein network

2. **CHILD Network (d=20, n=5,000 samples)**
   - Horizontal: K=3 clients
   - Vertical: K=4 clients (5 variables each)
   - Hybrid: K=4 clients
   - Ground Truth: Medical diagnosis network

3. **ALARM Network (d=37, n=10,000 samples)**
   - Horizontal: K=5 clients (scalability test)
   - Vertical: Skip (too many variables for vertical)
   - Hybrid: K=5 clients
   - Ground Truth: Large medical network

**Metrics to Report**:
- Structural Hamming Distance (SHD)
- Skeleton F1 Score
- Precision/Recall for edges
- CI Test Accuracy
- Train/Test Log-Likelihood

---

### 3.3 Data Preprocessing

**Requirements**:
1. Download BN datasets from bnlearn
2. Convert to pandas DataFrame format
3. Add discretization for continuous variables (if any)
4. Create federated splits (horizontal/vertical/hybrid)
5. Store ground truth adjacency matrices

**Scripts to Create**:
- `data/download_real_datasets.py` - Download from bnlearn
- `data/preprocess_real_data.py` - Convert and discretize
- `data/create_federated_splits.py` - Generate H/V/Hy partitions

**Location**: `/experiments/v3_real_world_data/`

---

## Part 4: Experimental Design

### 4.1 V3 Experiment Structure

```
experiments/v3_comprehensive_fixes/
├── synthetic/           # V2 configs + fixes
│   ├── linear/
│   │   ├── small/      # d=8, K=3, n=900 (baseline)
│   │   ├── medium/     # d=10, K=3, n=1200
│   │   └── large/      # d=11, K=5, n=2000
│   └── nonlinear/
│       └── (same structure)
├── real_world/          # NEW: Real datasets
│   ├── sachs/          # d=11, K=3, n=7466
│   ├── child/          # d=20, K=4, n=5000
│   └── alarm/          # d=37, K=5, n=10000
└── ablation/            # NEW: Ablation studies
    ├── horizontal_aggregation/  # Test different aggregation methods
    ├── vertical_overlap/        # Test feature overlap strategies
    └── hybrid_sum_products/     # Compare old vs new hybrid
```

---

### 4.2 Ablation Studies

**Purpose**: Validate that fixes actually improve performance

**Experiments**:

1. **Horizontal Aggregation Methods** (Fix 1.1)
   - Baseline: Uniform weights (V2)
   - Method A: Majority voting on structure
   - Method B: LL-weighted aggregation
   - Method C: Cluster-aware mixing
   - Metric: Global F1 improvement

2. **Vertical Feature Overlap** (Fix 1.2)
   - Baseline: No overlap (V2)
   - Overlap 10%: Each client shares 10% features with neighbors
   - Overlap 20%: Each client shares 20% features with neighbors
   - Metric: Minimum tests per client

3. **Hybrid Sum-over-Products** (Fix 2.1)
   - Baseline: Current product-only (V2)
   - New: Sum-over-products (V3)
   - Metric: Cross-group F1 score

**Config**: Use SMALL (d=8, K=3) for fast iteration

---

## Part 5: Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
- [x] Create v3 branch
- [ ] **Fix 2.1**: Implement GlobalSumOfProducts (~4-6 hours)
  - [ ] Add GlobalSumOfProducts class to FedPC.py
  - [ ] Add sample_cluster_combinations helper
  - [ ] Rewrite hybrid mode in FedCDH.py
  - [ ] Test: run_hybrid_ci_ranking_test.py
- [ ] **Fix 1.1**: Implement structure-preserving aggregation (~3-4 hours)
  - [ ] Add LL-weighted mixing to GroupMixture
  - [ ] Add majority-vote structure extraction
  - [ ] Test: Compare global F1 before/after
- [ ] **Fix 1.2**: Add minimum feature constraint (~2 hours)
  - [ ] Add validation in vertical partitioning
  - [ ] Add warning for <4 features per client
  - [ ] Test: Verify minimum test count

**Deliverable**: V3 with all critical fixes applied

---

### Phase 2: Real-World Data (Week 2)
- [ ] Download and preprocess datasets (~4 hours)
  - [ ] Download Sachs, CHILD, ALARM from bnlearn
  - [ ] Convert to DataFrame format
  - [ ] Create federated splits (H/V/Hy)
  - [ ] Store ground truth matrices
- [ ] Run Sachs experiments (~6 hours)
  - [ ] Horizontal mode (K=3)
  - [ ] Vertical mode (K=3)
  - [ ] Hybrid mode (K=3)
  - [ ] Compare against ground truth
- [ ] Run CHILD experiments (~8 hours)
  - [ ] Horizontal mode (K=4)
  - [ ] Vertical mode (K=4)
  - [ ] Hybrid mode (K=4)
- [ ] (Optional) Run ALARM experiments (~10 hours)
  - [ ] Horizontal mode (K=5)
  - [ ] Hybrid mode (K=5)

**Deliverable**: Real-world validation results

---

### Phase 3: Ablation Studies (Week 3)
- [ ] Horizontal aggregation ablation (~6 hours)
  - [ ] Baseline (V2)
  - [ ] Majority voting
  - [ ] LL-weighted
  - [ ] Cluster-aware
  - [ ] Compare F1 scores
- [ ] Vertical overlap ablation (~4 hours)
  - [ ] 0% overlap (V2)
  - [ ] 10% overlap
  - [ ] 20% overlap
  - [ ] Compare minimum test counts
- [ ] Hybrid sum-products ablation (~4 hours)
  - [ ] Product-only (V2)
  - [ ] Sum-over-products (V3)
  - [ ] Compare cross-group F1

**Deliverable**: Ablation study report

---

### Phase 4: Analysis and Reporting (Week 4)
- [ ] Generate V3 comprehensive report (~4 hours)
  - [ ] Extend V2 report generator
  - [ ] Add real-world results section
  - [ ] Add ablation study visualizations
  - [ ] Add fix effectiveness analysis
- [ ] Write V3 summary document (~3 hours)
  - [ ] Compare V2 vs V3 performance
  - [ ] Quantify improvement from each fix
  - [ ] Real-world vs synthetic comparison
  - [ ] Lessons learned
- [ ] Create publication-ready figures (~3 hours)
  - [ ] F1 score improvements
  - [ ] Real-world SHD comparison
  - [ ] Ablation study results

**Deliverable**: V3 final report and thesis-ready figures

---

## Part 6: Success Criteria

### Must Have (Critical)
- ✅ Horizontal Global F1 > 0.3 (from 0.000)
- ✅ Hybrid Cross-group F1 > 0.3 (from 0.000)
- ✅ Sachs dataset results reported
- ✅ All local SPNs have ≥10 confusion matrix tests

### Should Have (Important)
- ✅ CHILD dataset results reported
- ✅ Ablation studies completed for all 3 fixes
- ✅ V3 comprehensive HTML report generated
- ✅ Minimum 6 pairwise tests per vertical client

### Nice to Have (Optional)
- ⚪ ALARM dataset results
- ⚪ Asia/Insurance datasets
- ⚪ Ensemble SPN implementation
- ⚪ LearnSPN integration

---

## Part 7: Risk Assessment

### High Risk
1. **GlobalSumOfProducts complexity**
   - Risk: Implementation may be more complex than expected
   - Mitigation: Follow Seng's guidance closely, test incrementally
   - Fallback: Document limitation for thesis

2. **Real-world data quality**
   - Risk: Discretization may lose information
   - Mitigation: Use established bnlearn preprocessing
   - Fallback: Focus on Sachs (already discrete)

### Medium Risk
1. **Horizontal F1 fix effectiveness**
   - Risk: Aggregation changes may not improve F1
   - Mitigation: Test multiple aggregation strategies
   - Fallback: Document as limitation, focus on hybrid fix

2. **Computational time**
   - Risk: Real-world experiments may take too long
   - Mitigation: Start with Sachs (smallest), scale up
   - Fallback: Skip ALARM if time constrained

### Low Risk
1. **Vertical feature constraint**
   - Risk: May reduce vertical partition flexibility
   - Mitigation: Make configurable (min_features parameter)
   - Fallback: Revert to V2 behavior

---

## Part 8: File Inventory

### New Files to Create
```
experiments/v3_comprehensive_fixes/
├── V3_COMPREHENSIVE_PLAN.md (this file)
├── data/
│   ├── download_real_datasets.py
│   ├── preprocess_real_data.py
│   ├── create_federated_splits.py
│   └── datasets/
│       ├── sachs/
│       ├── child/
│       └── alarm/
├── scripts/
│   ├── run_v3_synthetic.py
│   ├── run_v3_real_world.py
│   └── run_v3_ablation.py
├── analysis/
│   ├── generate_v3_report.py
│   ├── compare_v2_v3.py
│   └── ablation_analysis.py
└── results/
    ├── synthetic/
    ├── real_world/
    └── ablation/
```

### Files to Modify
```
causallearn/
├── utils/FedPC.py
│   └── + GlobalSumOfProducts class (~150 lines)
└── search/FCMBased/FedCDH/
    └── FedCDH.py
        ├── hybrid mode rewrite (~100 lines changed)
        ├── + vertical feature validation (~20 lines)
        └── + horizontal aggregation options (~50 lines)
```

---

## Part 9: Estimated Time Investment

| Phase | Tasks | Hours | Priority |
|-------|-------|-------|----------|
| Phase 1 | Critical Fixes | 10-12 | 🔴 CRITICAL |
| Phase 2 | Real-World Data | 18-28 | 🔴 CRITICAL (Sachs) |
| Phase 3 | Ablation Studies | 14-18 | 🟡 MEDIUM |
| Phase 4 | Analysis/Reporting | 10-12 | 🟡 MEDIUM |
| **Total** | | **52-70 hours** | **~2-3 weeks** |

**Critical Path** (minimum viable):
- Phase 1 (10 hrs) + Sachs only (6 hrs) + Basic report (4 hrs) = **20 hours** (~3 days)

---

## Part 10: References

### From V2 Analysis Report
- Horizontal F1 Dilution (Analysis Tab, Critical Issues #1)
- Vertical Insufficient Edges (Analysis Tab, Critical Issues #2)
- LARGE Config Dilution (Analysis Tab, Critical Issues #3)

### From working_state.md
- GlobalSumOfProducts Implementation Plan (Lines 1-200)
- Adaptive Scaling + Ensemble Backlog (Lines 5740-5839)
- LearnSPN Investigation (archived as non-viable)

### External
- Seng's Feedback (April 29, 2026) on sum-over-products
- bnlearn package for benchmark networks
- Sachs et al. (2005) protein signaling data

---

## Part 11: Next Steps

**Immediate Actions** (this week):
1. ✅ Create v3 branch
2. ✅ Write V3_COMPREHENSIVE_PLAN.md (this document)
3. ⬜ Review plan with advisor/team
4. ⬜ Start Phase 1: Implement GlobalSumOfProducts
5. ⬜ Test hybrid mode fix on SMALL config

**User Decision Points**:
- Should we implement all fixes or prioritize specific ones?
- Which real-world datasets are most important? (Sachs vs CHILD vs ALARM)
- Should we run full ablation studies or focus on critical fixes?
- Is 2-3 weeks timeline acceptable?

---

**Status**: 📋 Ready for review and approval to proceed
