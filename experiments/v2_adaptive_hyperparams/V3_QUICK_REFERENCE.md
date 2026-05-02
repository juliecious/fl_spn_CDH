# V3 Quick Reference

**Branch**: `v3-comprehensive-fixes`
**Status**: Ready to implement
**Documents**: V3_UPDATED_PLAN.md (full details), V3_CHECKLIST.md (tasks)

---

## 🎯 Three-Pronged Approach

### 1. **Critical Fixes** (from V2 bugs)
- Hybrid sum-over-products (F1: 0.000 → 0.3+)
- Horizontal F1 dilution (F1: 0.000 → 0.3+)
- Vertical feature constraints (min 4 features/client)

### 2. **Datasets & Baselines**
- **Datasets**: Sachs, Law School, HyperPC, synthetic
- **Baselines**: Centralized PC/GIES, Original FedCDH, Other federated methods
- **Comparison**: Our H/V/Hy vs baselines

### 3. **Ablation Studies**
- **Sample Size** (n): 300 to 3600
- **Dimensionality** (d): 5 to 30
- **Num Clients** (K): 2 to 10

---

## 📊 Datasets Overview

| Dataset | Type | d | n | Source | Priority | Status |
|---------|------|---|---|--------|----------|--------|
| **Sachs** | Real | 11 | 7,466 | tests/data/ | 🔴 CRITICAL | ✅ Available |
| **Law School** | Real | ~10 | ? | arXiv:2506.06039 | 🟡 MEDIUM | ⬜ Need download |
| **HyperPC** | Synthetic | Var | Var | v3_data/hyperpc-main | 🟡 MEDIUM | ✅ Extracted |
| **V2 Synthetic** | Synthetic | 8-11 | 900-2000 | Existing | 🔴 CRITICAL | ✅ Available |
| **ALARM** | Real | 37 | ? | bnlearn | 🟢 LOW | ⬜ Optional |
| **CHILD** | Real | 20 | ? | bnlearn | 🟢 LOW | ⬜ Optional |

---

## 🏆 Baselines to Compare

### **Tier 1: Must Have (Critical)**
1. **Centralized PC** - Upper bound (pooled data)
2. **Original FedCDH (Horizontal)** - Direct comparison
3. **Our V3 Methods** - FedCDH-SPN (H/V/Hy)

### **Tier 2: Should Have (Important)**
4. **Centralized GIES** - Alternative algorithm
5. **Other Federated Methods** - Recent papers (2023-2026)

### **Tier 3: Nice to Have (Optional)**
6. **Privacy-preserving methods** - DP-based causal discovery
7. **Vertical/Hybrid baselines** - If available in literature

---

## 📈 Ablation Studies Detail

### **Ablation 1: Sample Size → Performance**
```
Fixed: d=10, K=3
Vary:  n ∈ {300, 600, 900, 1200, 1800, 2400, 3600}
Modes: H, V, Hy × Linear, Nonlinear
Total: 7 × 3 × 2 = 42 experiments
Time:  ~7-10 hours runtime
```

**Expected**: F1 increases with n, plateaus at different points for H/V/Hy

---

### **Ablation 2: Dimensionality → Performance**
```
Fixed: n=1200, K=3
Vary:  d ∈ {5, 8, 10, 12, 15, 20, 25, 30}
Modes: H, V*, Hy × Linear, Nonlinear
       (*V skipped if d < 12 due to min feature constraint)
Total: 8 × ~2.5 × 2 = ~40 experiments
Time:  ~7-10 hours runtime
```

**Expected**: F1 decreases with d, V more affected than H

---

### **Ablation 3: Number of Clients → Performance**
```
Fixed: d=12, n=1200
Vary:  K ∈ {2, 3, 4, 5, 7, 10}
Modes: H, V*, Hy × Linear, Nonlinear
       (*V skipped if d/K < 4)
Total: 6 × ~2.5 × 2 = ~30 experiments
Time:  ~3-5 hours runtime
```

**Expected**: F1 decreases with K (less data/features per client)

---

## ⏱️ Time Investment Summary

| Scope | Tasks | Hours | Timeline |
|-------|-------|-------|----------|
| **Minimum Viable** | Fixes + Sachs + 1 ablation | 26 | 4 days |
| **Recommended** | + Baselines + All ablations | 66-92 | 3-4 weeks |
| **Full** | + All datasets + Extra baselines | 100+ | 5-6 weeks |

---

## 🚀 Recommended Path (3-4 weeks)

### **Week 1: Core Implementation**
- Days 1-2: Implement 3 critical fixes (10-12 hrs)
- Days 3-4: Setup baselines + centralized comparison (8 hrs)
- Day 5: Sachs experiments (4 hrs)

### **Week 2: Ablations**
- Days 1-2: Sample size ablation (8-10 hrs)
- Days 3-4: Dimensionality ablation (8-10 hrs)
- Day 5: Number of clients ablation (4-6 hrs)

### **Week 3: Additional Datasets**
- Days 1-2: Law School dataset (6-8 hrs)
- Days 3-4: HyperPC benchmarks (6-8 hrs)
- Day 5: Additional baselines (4-6 hrs)

### **Week 4: Analysis & Reporting**
- Days 1-2: Generate comprehensive report (6-8 hrs)
- Days 3-4: Publication figures (4-6 hrs)
- Day 5: Final documentation (2-4 hrs)

**Total**: ~66-92 hours over 3-4 weeks

---

## 📋 Files Created So Far

```
experiments/v2_adaptive_hyperparams/
├── V3_COMPREHENSIVE_PLAN.md        # Original detailed plan
├── V3_UPDATED_PLAN.md              # Updated with datasets/baselines/ablations
├── V3_CHECKLIST.md                 # Task checklist
├── V3_QUICK_REFERENCE.md           # This file
└── v3_data/
    └── hyperpc-main/                # Extracted HyperPC code
```

---

## 🔍 Key Metrics to Report

**For Every Experiment**:
- Skeleton F1 Score ⭐
- Structural Hamming Distance (SHD) ⭐
- Precision / Recall (edges)
- CI Test Accuracy ⭐
- Train/Test Log-Likelihood
- Runtime (seconds)

**For Comparisons**:
- V2 vs V3 improvement (%)
- Federated vs Centralized gap (%)
- Mode comparison (H vs V vs Hy)

---

## 📊 Key Figures to Generate

1. **Fix Effectiveness**: V2 vs V3 bar chart (F1 scores)
2. **Baseline Comparison**: Bar chart (all methods on Sachs)
3. **Ablation Plots**: 3 line plots (n, d, K vs F1)
4. **Mode Comparison**: Heatmap (dataset × mode)
5. **Real-World Results**: Network diagrams (predicted vs true)

---

## ❓ Decision Points

**User needs to decide**:

1. **Scope**: Minimum (26 hrs) vs Recommended (66-92 hrs)?
2. **Datasets**: Which are most important? (Sachs only vs multiple)
3. **Baselines**: How many to implement? (Centralized only vs multiple federated)
4. **Ablations**: All three or subset? (n+d+K vs just n)
5. **Timeline**: Thesis defense date? Available hours/week?

---

## 🎬 Next Steps

**Immediate** (this session):
1. User decides on scope (minimum vs full)
2. User prioritizes datasets
3. User sets timeline

**Then** (next session):
1. Start Phase 1: Implement GlobalSumOfProducts
2. Test hybrid mode fix
3. Continue with horizontal aggregation fix

**Commands**:
```bash
# Already done
cd /Users/M279402/PycharmProjects/fl_spn_CDH
git checkout v3-comprehensive-fixes

# Ready to start
# See V3_CHECKLIST.md for detailed tasks
# See V3_UPDATED_PLAN.md for full implementation plan
```

---

## 📚 Reference Documents

- **V3_UPDATED_PLAN.md**: Full detailed plan (66-92 hours, all details)
- **V3_CHECKLIST.md**: Actionable task list with checkboxes
- **V3_QUICK_REFERENCE.md**: This file (quick overview)
- **working_state.md**: Technical details on GlobalSumOfProducts
- **V2 Analysis Report**: Evidence for bugs/fixes needed

---

**Status**: 📋 Ready for user to decide on scope and start implementation
