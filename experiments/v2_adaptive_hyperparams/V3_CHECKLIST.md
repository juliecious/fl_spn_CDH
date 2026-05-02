# V3 Implementation Checklist

**Branch**: `v3-comprehensive-fixes`
**Target**: Fix critical V2 issues + real-world validation
**Estimated Time**: 52-70 hours (2-3 weeks) | **Minimum Viable**: 20 hours (3 days)

---

## 🔴 CRITICAL FIXES (Must Complete)

### Fix #1: Hybrid Mode Sum-over-Products
- [ ] Read implementation plan in working_state.md (lines 1-400)
- [ ] Create `GlobalSumOfProducts` class in `causallearn/utils/FedPC.py`
  - [ ] Accept list of cluster SPNs and feature groups
  - [ ] Sample/enumerate cluster combinations
  - [ ] Create sum over product SPNs
  - [ ] Implement log_prob with NaN masking
  - [ ] Implement sample method
- [ ] Create `sample_cluster_combinations` helper function
  - [ ] Enumerate all if ≤20 combinations
  - [ ] Random sample if >20 combinations
  - [ ] Return list of cluster configs
- [ ] Rewrite hybrid mode in `causallearn/search/FCMBased/FedCDH/FedCDH.py`
  - [ ] Reuse local cluster SPNs (don't train new ones)
  - [ ] Extract feature groups with NaN masking
  - [ ] Build GlobalSumOfProducts instead of ProductOverGroupsWithOverlap
- [ ] Test hybrid cross-group dependencies
  - [ ] Run `tests/run_hybrid_ci_ranking_test.py`
  - [ ] Verify cross-group F1 > 0.3 (from 0.000)
  - [ ] Verify CI tests return p < 1.0
- [ ] Test hybrid dense-local dependencies
  - [ ] Verify F1 maintains ~0.8-1.0

**Expected Impact**: Cross-group F1: 0.000 → 0.3-0.7 ✅
**Time Estimate**: 4-6 hours
**Priority**: 🔴 BLOCKING (author-identified bug)

---

### Fix #2: Horizontal Mode F1 Dilution
- [ ] Investigate aggregation mechanism in GroupMixture
  - [ ] Check how local structures are combined
  - [ ] Identify why F1=0.000 despite local F1>0.3
- [ ] Implement structure-preserving aggregation options:
  - [ ] **Option A**: Majority voting on edges before aggregation
    - [ ] Extract local dependency graphs
    - [ ] Vote on each edge (include if >50% clients detect it)
    - [ ] Build global SPN with voted structure
  - [ ] **Option B**: Log-likelihood weighted mixing
    - [ ] Weight clients by their train LL quality
    - [ ] Better SPNs get higher influence
  - [ ] **Option C**: Cluster-aware mixing
    - [ ] Use cluster assignments to group similar clients
    - [ ] Preserve within-cluster structures
- [ ] Add `aggregation_mode` parameter to horizontal mode
  - [ ] Default: "uniform" (V2 behavior)
  - [ ] Options: "majority_vote", "ll_weighted", "cluster_aware"
- [ ] Test on Linear SMALL Horizontal
  - [ ] Baseline (V2): F1=0.000
  - [ ] Target (V3): F1 > 0.3
  - [ ] Compare all aggregation options
- [ ] Verify local SPNs maintain F1>0.3

**Expected Impact**: Global F1: 0.000 → 0.3+ ✅
**Time Estimate**: 3-4 hours
**Priority**: 🔴 CRITICAL (thesis blocker)

---

### Fix #3: Vertical Feature Constraints
- [ ] Add minimum feature validation
  - [ ] Check each client has ≥4 features
  - [ ] Warn if any client has <4 features
  - [ ] Calculate minimum tests: (d choose 2)
- [ ] Implement feature overlap option (optional)
  - [ ] Add `feature_overlap_pct` parameter (default: 0%)
  - [ ] Allow 10-20% overlap between adjacent clients
  - [ ] Ensure coverage of all feature pairs
- [ ] Add `min_features_per_client` parameter
  - [ ] Default: 4
  - [ ] Adjust K if needed to satisfy constraint
- [ ] Test on Vertical LARGE (d=11, K=5)
  - [ ] Verify each client has ≥10 pairwise tests
  - [ ] Check confusion matrix totals
- [ ] Add logging for feature distribution
  - [ ] Log: "Client 0: 4 features, 6 tests"
  - [ ] Warn: "Client 3: Only 2 features (below minimum)"

**Expected Impact**: Minimum tests/client: 1-2 → 6+ ✅
**Time Estimate**: 2 hours
**Priority**: 🟡 MEDIUM (quality improvement)

---

## 🌍 REAL-WORLD DATA (Critical: Sachs Only)

### Sachs Dataset (Protein Signaling Network)
- [ ] Load existing Sachs data
  - [ ] Path: `tests/data/sachs.interventional.txt.gz`
  - [ ] Decompress and load
  - [ ] Verify: d=11, n=7,466 samples
  - [ ] Load ground truth network
- [ ] Create federated splits
  - [ ] **Horizontal**: K=3, ~2,500 samples each
  - [ ] **Vertical**: K=3, 3-4 features each (ensure ≥4)
  - [ ] **Hybrid**: K=3, both partitions
- [ ] Run experiments (SMALL config: epochs=50, lr scaled)
  - [ ] Horizontal mode
  - [ ] Vertical mode
  - [ ] Hybrid mode (with new sum-over-products)
- [ ] Compare against ground truth
  - [ ] Calculate SHD (Structural Hamming Distance)
  - [ ] Calculate Skeleton F1
  - [ ] Calculate Precision/Recall
  - [ ] Calculate CI Test Accuracy
- [ ] Generate Sachs-specific report
  - [ ] Visualize predicted vs true network
  - [ ] Highlight correctly/incorrectly detected edges
  - [ ] Compare H/V/Hy performance

**Expected Impact**: Real-world validation of fixes ✅
**Time Estimate**: 6 hours
**Priority**: 🔴 CRITICAL (thesis validation)

---

## 📊 OPTIONAL ENHANCEMENTS (If Time Permits)

### Additional Real-World Datasets
- [ ] **CHILD Network** (d=20, K=4, n=5,000)
  - [ ] Download from bnlearn R package
  - [ ] Preprocess and discretize
  - [ ] Create federated splits
  - [ ] Run H/V/Hy experiments
- [ ] **ALARM Network** (d=37, K=5, n=10,000)
  - [ ] Download from bnlearn
  - [ ] Run scalability test
  - [ ] Horizontal and Hybrid only (skip Vertical)

**Time Estimate**: 12-18 hours
**Priority**: 🟡 MEDIUM

---

### Ablation Studies
- [ ] **Horizontal Aggregation Ablation**
  - [ ] Baseline (V2 uniform)
  - [ ] Majority voting
  - [ ] LL-weighted
  - [ ] Cluster-aware
  - [ ] Compare F1 improvements
- [ ] **Vertical Overlap Ablation**
  - [ ] 0% overlap (V2)
  - [ ] 10% overlap
  - [ ] 20% overlap
  - [ ] Compare minimum test counts
- [ ] **Hybrid Sum-over-Products Ablation**
  - [ ] Product-only (V2)
  - [ ] Sum-over-products (V3)
  - [ ] Compare cross-group F1

**Time Estimate**: 14-18 hours
**Priority**: 🟡 MEDIUM

---

### V3 Report Generation
- [ ] Extend `generate_v2_report_enhanced.py` for V3
  - [ ] Add real-world results section
  - [ ] Add ablation study visualizations
  - [ ] Add before/after fix comparisons
- [ ] Generate V3 comprehensive HTML report
  - [ ] Tab 1: Synthetic Results (V3)
  - [ ] Tab 2: Real-World Results (Sachs, CHILD, ALARM)
  - [ ] Tab 3: Ablation Studies
  - [ ] Tab 4: V2 vs V3 Comparison
  - [ ] Tab 5: Architecture & Analysis
- [ ] Create publication figures
  - [ ] F1 improvement chart (V2 → V3)
  - [ ] Real-world SHD comparison
  - [ ] Ablation study results

**Time Estimate**: 10-12 hours
**Priority**: 🟡 MEDIUM

---

## 🎯 MINIMUM VIABLE PATH (20 hours / 3 days)

**Critical path for thesis defense**:

### Day 1: Core Fixes (8-10 hours)
- [x] Create v3 branch
- [x] Write V3 plan document
- [ ] Implement GlobalSumOfProducts (4-6 hrs)
- [ ] Quick test on SMALL hybrid (1 hr)

### Day 2: Horizontal Fix + Sachs (8-10 hours)
- [ ] Implement horizontal aggregation fix (3-4 hrs)
- [ ] Test on SMALL horizontal (1 hr)
- [ ] Run Sachs experiments (H/V/Hy) (4-5 hrs)

### Day 3: Validation + Report (4-6 hours)
- [ ] Verify all fixes work (1-2 hrs)
- [ ] Generate basic V3 report (2-3 hrs)
- [ ] Write V3 summary document (1 hr)

**Deliverable**: V3 with critical fixes + Sachs validation

---

## 📋 PROGRESS TRACKING

### Week 1: Critical Fixes
- [ ] GlobalSumOfProducts implemented
- [ ] Horizontal aggregation fixed
- [ ] Vertical constraints added
- [ ] All fixes tested on synthetic SMALL

### Week 2: Real-World Validation
- [ ] Sachs experiments complete
- [ ] CHILD experiments (optional)
- [ ] Results compared to ground truth

### Week 3: Polish and Analysis
- [ ] Ablation studies complete
- [ ] V3 comprehensive report generated
- [ ] V2 vs V3 comparison documented
- [ ] Publication figures created

---

## 🚀 START HERE

**Immediate next steps**:
1. Review V3_COMPREHENSIVE_PLAN.md
2. Decide: Full plan (70 hrs) or Minimum viable (20 hrs)?
3. Read working_state.md GlobalSumOfProducts section (lines 1-400)
4. Start implementing Fix #1 (Hybrid mode)

**Command to begin**:
```bash
cd /Users/M279402/PycharmProjects/fl_spn_CDH
git checkout v3-comprehensive-fixes
# Start with GlobalSumOfProducts implementation
```

---

**Questions for User**:
1. Should we do full plan (2-3 weeks) or minimum viable (3 days)?
2. Which real-world datasets are priority? (Sachs only vs Sachs+CHILD+ALARM)
3. Should we include ablation studies or focus on fixes only?
4. Any specific metrics/visualizations needed for thesis defense?
