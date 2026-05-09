# V3 Thesis-Critical Roadmap

**Date**: 2026-05-03
**Research Goal**: "Federated Causal Discovery with Probabilistic Circuits"
**Branch**: `v3-comprehensive-fixes`

---

## Research Objective

**Thesis Title**: "Federated Causal Discovery with Probabilistic Circuits"

**Core Research Question**: Can Sum-Product Networks (SPNs) replace traditional kernel-based CI tests in federated causal discovery while maintaining accuracy and improving scalability?

**Key Contributions**:
1. **Novel CI Testing**: SPN-based conditional independence testing for federated causal discovery
2. **Three Federated Scenarios**: Horizontal, Vertical, Hybrid data partitioning with SPN architectures
3. **Structure-Preserving Aggregation**: Democratic voting to prevent dependency dilution (NEW V3 finding)
4. **Sum-over-Products Architecture**: Breaking independence in hybrid scenarios (NEW V3 finding)

**Success Criteria**:
- Sachs F1 ≥ 0.80 (vs centralized baseline)
- FedSPN time < 0.5 × KCI time for d ≥ 20
- Communication cost < 0.1 × raw data size

---

## V2 Results Summary (What We Know)

### V2 Performance (Before V3 Fixes)

| Config | Mode | F1 Skeleton | Issues |
|--------|------|-------------|--------|
| Linear SMALL | Horizontal | 0.000 | ❌ Mixture averaging destroys dependencies |
| Linear SMALL | Vertical | 0.133 | ✅ Works but low sample variance |
| Linear SMALL | Hybrid | 0.000 | ❌ Product enforces independence |
| Linear MEDIUM | Horizontal | 0.000 | ❌ Same dilution issue |
| Linear MEDIUM | Hybrid | 0.563 | ⚠️ Partial success |
| **Linear LARGE** | Hybrid | **0.778** | ✅ Best V2 result |
| **Nonlinear LARGE** | Hybrid | **0.814** | ✅ **Best overall** |

**Key Findings**:
1. ❌ **Horizontal F1 collapse**: Local F1=0.26-0.57 → Global F1=0.000 (mixture averaging)
2. ❌ **Hybrid independence bug**: Cross-group F1=0.000 (missing sum-over-products)
3. ✅ **Sample-size pattern**: Nonlinear > Linear only when n > 1500
4. ✅ **Architecture validation**: SPNs CAN work (F1=0.814 on LARGE nonlinear)

---

## V3 Critical Fixes (IMPLEMENTED)

### Fix #1: GlobalSumOfProducts (Hybrid Mode) ✅
- **Status**: IMPLEMENTED (FedPC.py:1621-1801)
- **Problem**: Product factorization enforces independence → Cross-group F1=0.000
- **Solution**: Sum-over-products breaks independence via cluster coupling
- **Expected**: Cross-group F1: 0.000 → 0.3-0.7

### Fix #2: Structure-Preserving Aggregation (Horizontal Mode) ✅
- **Status**: IMPLEMENTED (structure_aggregation.py + FedCDH.py:779-880)
- **Problem**: Mixture averaging dilutes dependencies → Global F1=0.000
- **Solution**: Democratic voting on dependency graphs (structure_voting mode)
- **Expected**: Global F1: 0.000 → 0.3+

### Fix #3: Vertical Feature Validation ⚠️
- **Status**: PARTIAL (needs validation code)
- **Problem**: Clients with <4 features → unreliable metrics (high variance)
- **Solution**: min_features_per_client constraint
- **Priority**: MEDIUM (not thesis blocker)

---

## Thesis-Critical Roadmap

### PHASE 1: Verify V3 Fixes (IMMEDIATE - 1 week)

**Priority**: 🔴 **CRITICAL** - Prove fixes work before proceeding

#### Week 1: Testing & Verification (4-6 hours)

**Goal**: Verify V3 fixes achieve expected improvements

**Tasks**:
1. **Test Hybrid Mode (2 hours)**
   ```bash
   python tests/run_hybrid_ci_ranking_test.py
   ```
   - ✅ Success: Cross-group F1 > 0.3 (from 0.000)
   - ✅ Success: CI tests return p < 1.0 (not always 1.0)
   - ✅ Success: Dense-local F1 ~ 1.0 maintained

2. **Test Horizontal Aggregation Strategies (2-3 hours)**
   ```bash
   python test_horizontal_comparison.py  # Create this
   ```
   - Compare: structure_voting vs ll_weighted vs mixture
   - Measure: Global F1, consensus quality, edge preservation
   - ✅ Success: structure_voting F1 > 0.3 (from 0.000)

3. **V2 vs V3 Comparison (1 hour)**
   - Run SMALL config with both versions
   - Generate comparison table
   - Document improvements

4. **Git Commit (30 mins)**
   - Commit verification results
   - Tag: `v3-fixes-verified`

**Deliverables**:
- Test results showing F1 improvements
- V2 vs V3 comparison table
- Verified implementation ready for experiments

**Thesis Impact**: ✅ Validates core technical contributions (#3 and #4)

---

### PHASE 2: Baseline Comparison (CRITICAL - 1 week)

**Priority**: 🔴 **CRITICAL** - Thesis requires baseline comparison

#### Week 2: Establish Performance Benchmarks (12-16 hours)

**Goal**: Compare FedSPN against centralized and federated baselines

**Tasks**:

1. **Centralized Baselines (6-8 hours)**
   - **What**: PC, GES, FCI on pooled Sachs data
   - **Why**: Upper bound performance (no privacy constraints)
   - **Implementation**:
     ```python
     # Already available in causal-learn
     from causallearn.search.ConstraintBased.PC import pc
     from causallearn.search.ScoreBased.GES import ges
     from causallearn.search.ConstraintBased.FCI import fci

     # Run on pooled Sachs data (n=7,466, d=11)
     # Measure: F1, SHD, Precision, Recall, Time
     ```
   - **Expected**: F1 ~ 0.85-0.95 (centralized upper bound)
   - **Thesis Impact**: Shows privacy-accuracy tradeoff

2. **Original FedCDH Baseline (4-6 hours)**
   - **What**: Li et al. (2024) ICLR horizontal-only implementation
   - **Why**: Direct comparison to published method
   - **Source**: `fedcdh_code.zip` (if available) or re-implement horizontal-only
   - **Expected**: F1 ~ 0.60-0.80 on Sachs horizontal
   - **Thesis Impact**: Shows our extension to V/Hy scenarios

3. **Comparison Table Generation (2 hours)**
   - Compile all results
   - Generate publication-ready table
   - Statistical significance tests (t-test, Wilcoxon)

**Deliverables**:
- Centralized baseline results (PC, GES, FCI)
- FedCDH original baseline results
- Comparison table: Centralized vs FedCDH vs FedSPN (V3)

**Thesis Impact**: ✅ Validates RQ1 (accuracy) and establishes competitive performance

---

### PHASE 3: Real-World Validation (CRITICAL - 1 week)

**Priority**: 🔴 **CRITICAL** - Thesis requires real-world dataset

#### Week 3: Sachs Protein Network Evaluation (12-16 hours)

**Goal**: Validate FedSPN on real-world causal discovery benchmark

**Why Sachs is Critical**:
- Standard benchmark in causal discovery literature
- Known ground truth (17 edges, 11 nodes)
- Real biological data (protein signaling)
- Interventional study (validates causal claims)
- **Required** for thesis credibility

**Tasks**:

1. **Data Preparation (4-6 hours)**
   ```bash
   # Decompress Sachs data
   gunzip tests/data/sachs.interventional.txt.gz

   # Load ground truth network
   # Create federated splits (H/V/Hy)
   python scripts/prepare_sachs_splits.py
   ```
   - Horizontal: K=3, ~2,500 samples each
   - Vertical: K=3, 3-4 features each
   - Hybrid: K=3, both partitions

2. **Run FedSPN Experiments (6-8 hours)**
   ```bash
   # Horizontal with structure_voting
   python run_sachs_experiment.py --scenario horizontal --aggregation structure_voting

   # Vertical
   python run_sachs_experiment.py --scenario vertical

   # Hybrid with sum-over-products
   python run_sachs_experiment.py --scenario hybrid
   ```
   - Run each scenario 5-10 times (different seeds)
   - Measure: F1, SHD, Precision, Recall, CI accuracy, Time, LL

3. **Analysis & Visualization (2 hours)**
   - Compare against ground truth
   - Generate network diagrams
   - Plot performance across scenarios

**Deliverables**:
- Sachs results for all 3 scenarios (H/V/Hy)
- Comparison against ground truth
- Network visualization showing learned vs true graph

**Thesis Impact**: ✅ Validates RQ1 (accuracy) on real-world data - **REQUIRED for publication**

---

### PHASE 4: Ablation Studies (IMPORTANT - 1-2 weeks)

**Priority**: 🟡 **IMPORTANT** - Strengthens thesis but not strictly required

#### Week 4-5: Systematic Analysis (24-30 hours)

**Goal**: Understand how key factors affect performance

**Tasks**:

1. **Sample Size Ablation (10-12 hours)** - **HIGHEST PRIORITY**
   - **Research Question**: When does FedSPN have enough data?
   - **Design**:
     ```python
     n ∈ [300, 600, 900, 1200, 1800, 2400, 3600]  # 7 configs
     modes = ['horizontal', 'vertical', 'hybrid']  # 3 modes
     data_types = ['linear', 'nonlinear']          # 2 types
     Total: 7 × 3 × 2 = 42 experiments
     ```
   - **Expected Finding**: Identify minimum n for reliable CI tests
   - **Thesis Impact**: Practical guidance for deployment

2. **Dimensionality Ablation (8-10 hours)** - **MEDIUM PRIORITY**
   - **Research Question**: Does FedSPN scale to high dimensions?
   - **Design**:
     ```python
     d ∈ [5, 8, 10, 15, 20, 30]  # 6 configs
     modes = ['horizontal', 'vertical', 'hybrid']
     Total: 6 × 3 = 18 experiments
     ```
   - **Expected Finding**: Validate RQ2 (scalability)
   - **Thesis Impact**: Shows competitive advantage over KCI

3. **Number of Clients Ablation (6-8 hours)** - **LOW PRIORITY**
   - **Research Question**: How does federation granularity affect performance?
   - **Design**:
     ```python
     K ∈ [2, 3, 5, 7, 10]  # 5 configs
     modes = ['horizontal', 'vertical', 'hybrid']
     Total: 5 × 3 = 15 experiments
     ```
   - **Expected Finding**: Communication-accuracy tradeoff
   - **Thesis Impact**: Nice-to-have, not critical

**Deliverables**:
- Ablation study results (line plots)
- Analysis: Optimal n, d, K for each scenario
- Insights: When to use H/V/Hy

**Thesis Impact**: ✅ Validates RQ2 (scalability) and provides practical insights

**Time-Saving Strategy**:
- **If time-constrained**: Only do sample size ablation (most important)
- **Skip**: Number of clients ablation (least important)

---

### PHASE 5: Thesis Writing & Reporting (CRITICAL - 1-2 weeks)

**Priority**: 🔴 **CRITICAL** - Final deliverable

#### Week 5-6: Documentation & Writing (14-18 hours)

**Goal**: Complete thesis with all results

**Tasks**:

1. **Results Chapter (6-8 hours)**
   - Table 1: Baseline comparison (Centralized vs FedCDH vs FedSPN)
   - Table 2: Sachs results (3 scenarios × metrics)
   - Table 3: V2 vs V3 improvements
   - Table 4: Sample size ablation summary
   - Figure 1: Sachs learned vs ground truth networks
   - Figure 2: Sample size vs F1 line plots
   - Figure 3: V2 vs V3 bar chart comparison

2. **Methods Chapter (4-6 hours)**
   - Algorithm 1: FedSPN training
   - Algorithm 2: SPN_CIT (CI testing)
   - Algorithm 3: GlobalSumOfProducts (hybrid)
   - Algorithm 4: Structure voting (horizontal)
   - Figures: Architecture diagrams (H/V/Hy)

3. **Discussion Chapter (2-3 hours)**
   - **Novel Findings**:
     - Structure-preserving aggregation necessity
     - Sum-over-products for hybrid scenarios
     - Sample-size dependent linear/nonlinear crossover
   - **Limitations**:
     - Hybrid F1 still lower than centralized
     - Requires minimum samples per client
     - SPN hyperparameter sensitivity
   - **Future Work**:
     - Theoretical analysis of vertical regularization
     - Differential privacy guarantees
     - Adaptive scenario selection

4. **Final Polish (2 hours)**
   - Abstract
   - Introduction
   - Conclusion
   - References
   - Proofread

**Deliverables**:
- Complete thesis draft
- Publication-ready figures
- Supplementary materials

**Thesis Impact**: ✅ GRADUATION!

---

## Minimum Viable Thesis (Time-Constrained Path)

**Total Time**: 26-32 hours (~4-5 days intensive work)

### Critical Path Only

1. **Phase 1: Verify V3 Fixes** (4-6 hrs) - MUST DO
   - Test hybrid and horizontal improvements
   - Document V2 vs V3 comparison

2. **Phase 2: Baseline Comparison** (8-10 hrs) - MUST DO
   - Centralized baselines (PC, GES) on Sachs
   - FedSPN on Sachs (3 scenarios)
   - Skip: Original FedCDH comparison

3. **Phase 3: Sachs Only** (6-8 hrs) - MUST DO
   - Run Sachs experiments (H/V/Hy)
   - Compare against ground truth
   - Skip: Law School, HyperPC

4. **Phase 4: Sample Size Ablation Only** (8-10 hrs) - OPTIONAL BUT RECOMMENDED
   - Run n ablation (7 × 3 × 2 = 42 experiments)
   - Skip: Dimensionality and K ablations

5. **Phase 5: Thesis Writing** (14-18 hrs) - MUST DO
   - Focus on results and methods
   - Minimal discussion
   - Basic figures

**Total**: 40-52 hours (~1-1.5 weeks intensive)

---

## Full Comprehensive Thesis (Time Available)

**Total Time**: 72-94 hours (~3-4 weeks)

### All Phases

1. **Phase 1: Verify V3 Fixes** (4-6 hrs)
2. **Phase 2: Baseline Comparison** (12-16 hrs) - Include original FedCDH
3. **Phase 3: Real-World Datasets** (12-16 hrs) - Sachs + Law School
4. **Phase 4: All Ablation Studies** (24-30 hrs) - n, d, K
5. **Phase 5: Comprehensive Thesis** (14-18 hrs) - Full writing

**Total**: 72-94 hours (~3-4 weeks)

---

## Research Questions Mapping

| Research Question | Addressed By | Critical? |
|-------------------|--------------|-----------|
| **RQ1**: Can FedSPN match KCI accuracy? | Phase 2 (Baselines) + Phase 3 (Sachs) | ✅ CRITICAL |
| **RQ2**: Does FedSPN scale better? | Phase 4 (Dimensionality ablation) | ✅ CRITICAL |
| **RQ3**: Does mechanism invariance help? | Phase 3 (Sachs orientation) | 🟡 IMPORTANT |
| **RQ4**: Privacy-accuracy tradeoff? | Phase 4 (Sample size ablation) | 🟡 IMPORTANT |
| **RQ5**: Why vertical outperforms? | V2 analysis (already documented) | 🟢 NICE-TO-HAVE |

---

## Key Thesis Contributions (What Makes This Novel)

### Contribution #1: SPN-based Federated CI Testing
- **What**: First use of probabilistic circuits for federated causal discovery
- **Evidence**: Phase 2 (baseline comparison) + Phase 3 (Sachs)
- **Novelty**: Replaces kernel methods with tractable density estimation

### Contribution #2: Three Federated Scenarios (H/V/Hy)
- **What**: Extension to vertical and hybrid data partitioning
- **Evidence**: Phase 3 (Sachs on all 3 scenarios)
- **Novelty**: Original FedCDH only handles horizontal

### Contribution #3: Structure-Preserving Aggregation (NEW)
- **What**: Democratic voting prevents dependency dilution
- **Evidence**: Phase 1 (V2 vs V3 comparison) - Global F1: 0.000 → 0.3+
- **Novelty**: First to identify mixture averaging problem for causal discovery

### Contribution #4: Sum-over-Products for Hybrid (NEW)
- **What**: GlobalSumOfProducts breaks independence in hybrid scenarios
- **Evidence**: Phase 1 (V2 vs V3 comparison) - Cross-group F1: 0.000 → 0.3+
- **Novelty**: Implements Seng's architecture correctly for causal discovery

### Contribution #5: Sample-Size Dependent Performance Pattern
- **What**: Nonlinear > Linear only when n > 1500
- **Evidence**: V2 results (already documented)
- **Novelty**: Practical insight for deployment

---

## Success Metrics (Final Thesis)

### Must Achieve (CRITICAL)
- [x] V3 fixes implemented ✅
- [ ] V3 fixes verified (F1 improvements documented)
- [ ] Sachs F1 ≥ 0.60 (federated, any scenario)
- [ ] Sachs F1 ≥ 0.80 × Centralized F1 (privacy-accuracy tradeoff)
- [ ] Baseline comparison complete (PC, GES vs FedSPN)
- [ ] Thesis draft complete

### Should Achieve (IMPORTANT)
- [ ] Sachs F1 ≥ 0.70 (competitive)
- [ ] Sample size ablation complete
- [ ] Original FedCDH comparison
- [ ] Dimensionality ablation complete

### Nice to Have (OPTIONAL)
- [ ] Law School dataset results
- [ ] Number of clients ablation
- [ ] HyperPC benchmarks
- [ ] Interactive visualizations

---

## Task Priority Ranking

| Priority | Task | Thesis Impact | Time |
|----------|------|---------------|------|
| 🔴 **P0** | Verify V3 fixes | Validates contributions #3, #4 | 4-6 hrs |
| 🔴 **P0** | Sachs experiments | Required for credibility | 6-8 hrs |
| 🔴 **P0** | Centralized baselines | Establishes upper bound | 6-8 hrs |
| 🔴 **P0** | Thesis writing | Final deliverable | 14-18 hrs |
| 🟡 **P1** | Sample size ablation | Shows practical limits | 10-12 hrs |
| 🟡 **P1** | Original FedCDH baseline | Direct comparison | 4-6 hrs |
| 🟡 **P1** | V2 vs V3 comparison | Shows improvements | 2-3 hrs |
| 🟢 **P2** | Dimensionality ablation | Scalability claim | 8-10 hrs |
| 🟢 **P2** | Law School dataset | 2nd real-world validation | 6-8 hrs |
| 🟢 **P3** | Number of clients ablation | Least important | 6-8 hrs |
| 🟢 **P3** | HyperPC benchmarks | Nice-to-have | 6-8 hrs |

---

## Recommended Schedule

### Week 1: Foundation (Critical)
- **Mon-Tue**: Verify V3 fixes (4-6 hrs)
- **Wed-Thu**: Centralized baselines (6-8 hrs)
- **Fri**: V2 vs V3 comparison (2-3 hrs)
- **Total**: 12-17 hrs

### Week 2: Real-World Validation (Critical)
- **Mon**: Prepare Sachs data (4 hrs)
- **Tue-Wed**: Run Sachs experiments (6-8 hrs)
- **Thu**: Original FedCDH baseline (4-6 hrs)
- **Fri**: Analysis & visualization (2 hrs)
- **Total**: 16-20 hrs

### Week 3: Ablation Studies (Important)
- **Mon-Wed**: Sample size ablation (10-12 hrs)
- **Thu-Fri**: Dimensionality ablation (8-10 hrs)
- **Total**: 18-22 hrs

### Week 4: Writing & Polish (Critical)
- **Mon-Tue**: Results chapter (6-8 hrs)
- **Wed**: Methods chapter (4-6 hrs)
- **Thu**: Discussion chapter (2-3 hrs)
- **Fri**: Final polish (2 hrs)
- **Total**: 14-19 hrs

**Grand Total**: 60-78 hours (~2.5-3.5 weeks)

---

## Current Status (2026-05-03)

**Phase Status**:
- Phase 0 (Setup): 80% ✅
- Phase 1 (Fixes): 85% ✅ (implementation done, testing needed)
- Phase 2 (Baselines): 0% ⬜
- Phase 3 (Real-World): 0% ⬜
- Phase 4 (Ablations): 0% ⬜
- Phase 5 (Writing): 0% ⬜

**Next Immediate Actions**:
1. Test hybrid mode (verify cross-group F1 > 0.3)
2. Test horizontal aggregation strategies
3. Run Sachs experiments

**Timeline to Completion**: 3-4 weeks (if starting today)

---

## Risk Mitigation

### High Risk: Sachs F1 Too Low
- **Mitigation**: Tune SPN hyperparameters (epochs, num_sums, learning_rate)
- **Fallback**: Lower success threshold to F1 ≥ 0.50 (still publishable if we explain)
- **Plan B**: Focus on scalability (RQ2) instead of accuracy (RQ1)

### Medium Risk: Time Overrun
- **Mitigation**: Follow minimum viable path (26-32 hrs)
- **Fallback**: Skip ablation studies, focus on Sachs only
- **Plan B**: Defer Law School and HyperPC to future work

### Low Risk: Implementation Bugs
- **Mitigation**: Comprehensive testing in Phase 1
- **Fallback**: Revert to V2 if V3 fixes fail
- **Plan B**: Document negative results (still valid research)

---

## Conclusion

**Thesis is Achievable**: V3 fixes are implemented, only testing and experiments remain.

**Critical Path**: 26-32 hours minimum (Verify + Sachs + Baselines + Writing)

**Recommended Path**: 60-78 hours comprehensive (includes ablations)

**Key Success Factors**:
1. Verify V3 fixes work (Phase 1)
2. Sachs real-world validation (Phase 3)
3. Baseline comparison (Phase 2)
4. Clear thesis writing (Phase 5)

**Start Immediately**: Test hybrid mode to verify cross-group F1 improvement!
