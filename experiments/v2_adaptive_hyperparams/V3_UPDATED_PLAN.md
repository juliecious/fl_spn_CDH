# V3 Updated Plan: Datasets, Baselines & Ablations

**Branch**: `v3-comprehensive-fixes`
**Updated**: 2026-05-02
**Status**: 🚀 Ready to implement

---

## Overview

V3 addresses critical bugs from V2 while adding comprehensive evaluation:
1. **Critical Fixes** (Hybrid sum-over-products, Horizontal F1 dilution, Vertical constraints)
2. **New Datasets** (Law School, CATE synthetic, HyperPC benchmarks)
3. **Baseline Comparisons** (FedCDH-H, other federated methods, vertical/hybrid methods)
4. **Ablation Studies** (Dataset size, Dimensionality, Number of clients)

---

## Part 1: Datasets for Evaluation

### 1.1 Real-World Datasets

#### **Law School Admissions** (from arXiv:2506.06039v1)
- **Source**: 1998 LSAC National Longitudinal Bar Passage Study
- **Variables**: Race (protected attribute), first-year-average (FYA), other admission factors
- **Use Case**: Causal fairness, interventional prediction
- **Ground Truth**: Established causal graph from Kusner et al. (2017)
- **Task**: Predict interventional distributions and CATE
- **Priority**: 🟡 MEDIUM
- **Implementation**:
  - [ ] Download from LSAC or DoWhy examples
  - [ ] Preprocess: Extract causal variables
  - [ ] Create federated splits (H/V/Hy)
  - [ ] Establish ground truth DAG

#### **Sachs Protein Network** (already available)
- **Source**: `tests/data/sachs.interventional.txt.gz`
- **Variables**: d=11 protein expression levels
- **Samples**: n=7,466
- **Ground Truth**: Known protein signaling network
- **Priority**: 🔴 CRITICAL (baseline benchmark)
- **Implementation**:
  - [x] Already downloaded
  - [ ] Create federated splits
  - [ ] Load ground truth adjacency matrix

#### **HyperPC Benchmarks** (from /Users/M279402/Downloads/hyperpc-main.zip)
- **Source**: Extracted to `v3_data/hyperpc-main/`
- **Type**: Synthetic data generator using priors
- **Features**:
  - Transformer-based hypernetwork
  - Tractable probabilistic circuits
  - Interventional inference
- **Priority**: 🟡 MEDIUM
- **Implementation**:
  - [x] Extracted zip file
  - [ ] Explore `src/hyperpc/prior/` for data generation
  - [ ] Generate synthetic datasets with varying d, n, K
  - [ ] Use as ablation study baseline

---

### 1.2 Synthetic Datasets (CATE Estimation - arXiv:2506.10914)

#### **Appendix E Synthetic Data Generator**
- **Source**: Need to access arXiv:2506.10914 Appendix E text
- **Use Case**: Standard CATE estimation setting
- **Priority**: 🟡 MEDIUM
- **Implementation**:
  - [ ] Download paper PDF and extract Appendix E
  - [ ] Implement synthetic data generator
  - [ ] Vary: Linear/Nonlinear, d, n, edge density
  - [ ] Generate datasets for ablation studies

**Alternative**: Use existing V2 synthetic generator with extensions:
```python
# Extend V2 generator with:
- Configurable edge density (sparse, medium, dense)
- Configurable causal strength (weak, medium, strong)
- Multiple DAG structures (chain, fork, collider, complex)
```

---

### 1.3 Benchmark Datasets from bnlearn (Optional)

**Standard Bayesian Network Benchmarks**:
- ALARM (d=37, medical diagnosis)
- CHILD (d=20, medical diagnosis)
- Insurance (d=27)
- Asia (d=8)

**Priority**: 🟢 LOW (use if time permits)

---

## Part 2: Baselines for Comparison

### 2.1 Our Methods (FedSPN Variants)

**FedCDH with SPNs**:
1. **Horizontal Mode** (FedCDH-H)
   - Same features, different samples per client
   - Baseline: Current V2 implementation
   - V3 Fix: Structure-preserving aggregation

2. **Vertical Mode** (FedCDH-V)
   - Different features, same samples per client
   - Baseline: Current V2 implementation
   - V3 Fix: Minimum feature constraint

3. **Hybrid Mode** (FedCDH-Hy)
   - Both feature and sample partitioning
   - Baseline: Current V2 implementation (product-only)
   - V3 Fix: Sum-over-products

**Key Comparison**: V2 vs V3 for each mode

---

### 2.2 Federated Causal Discovery Baselines

#### **2.2.1 FedCDH (Original - Horizontal Only)**
- **Source**: Original FedCDH paper implementation
- **Mode**: Horizontal federated learning only
- **Algorithm**: PC algorithm with federated CI tests
- **SPN**: Uses basic SPN without clustering
- **Use as**: Baseline for FedCDH-H comparison
- **Priority**: 🔴 CRITICAL

**Implementation**:
- [ ] Extract from `fedcdh_code.zip` in Downloads
- [ ] Run on same datasets as V3
- [ ] Compare: F1, Accuracy, SHD, CI accuracy
- [ ] Report: Table comparing original vs V3 horizontal

---

#### **2.2.2 Other Federated Causal Methods**

**Research needed - potential candidates**:

1. **Federated PC (if exists separately)**
   - Standard PC with federated communication
   - No SPN, uses empirical CI tests

2. **Federated GES/GIES**
   - Score-based methods in federated setting
   - If available in literature

3. **Federated Constraint-Based Methods**
   - FCI, RFCI variants
   - Handle latent confounders

**Priority**: 🟡 MEDIUM (research paper search needed)

**Implementation Plan**:
- [ ] Search recent papers (2023-2026): "federated causal discovery"
- [ ] Identify 2-3 comparable methods
- [ ] Check if code is available
- [ ] Implement adapters if needed
- [ ] Run on same test suite

---

### 2.3 Vertical/Hybrid Federated Methods

**Research needed - potential candidates**:

1. **Vertical Federated Learning with Causal Discovery**
   - Methods that handle feature partitioning
   - Example: SplitNN + causal discovery

2. **Hybrid Federated Approaches**
   - Methods combining horizontal and vertical FL
   - Check: FedML library, FedBCD

3. **Privacy-Preserving Causal Discovery**
   - Differential privacy + causal discovery
   - Secure multi-party computation variants

**Priority**: 🟡 MEDIUM

**Implementation Plan**:
- [ ] Literature review: vertical FL + causality
- [ ] Identify 1-2 comparable methods
- [ ] Implement or adapt existing code
- [ ] Compare on same datasets

---

### 2.4 Non-Federated Baselines (Centralized)

**Use as upper bound on performance**:

1. **Centralized PC** (from causal-learn)
   - Run on pooled data (all clients combined)
   - Use as "oracle" performance

2. **Centralized GIES** (score-based)
   - Alternative algorithm on pooled data

3. **Centralized FCI** (handles latent confounders)
   - More robust baseline

**Priority**: 🔴 CRITICAL (need upper bound)

**Implementation**:
- [ ] Use existing causal-learn implementations
- [ ] Run on pooled datasets
- [ ] Compare: How much does federation hurt performance?

---

## Part 3: Ablation Studies

### 3.1 Dataset Size → Performance

**Research Question**: How does sample size affect causal discovery accuracy?

**Experimental Design**:
- **Fixed**: d=10, K=3 clients
- **Vary**: n ∈ {300, 600, 900, 1200, 1800, 2400, 3600}
- **Per client**: n_local = n / K
- **Modes**: Horizontal, Vertical, Hybrid
- **Data**: Linear and Nonlinear

**Metrics to Track**:
- Skeleton F1 Score
- Structural Hamming Distance (SHD)
- CI Test Accuracy
- Train Log-Likelihood
- Runtime (seconds)

**Expected Results**:
- F1 should increase with n (more data = better CI tests)
- Horizontal should plateau earlier (duplicated features)
- Vertical should benefit more from larger n (more tests per feature pair)

**Priority**: 🔴 CRITICAL (core ablation)

**Implementation**:
```python
# ablation_sample_size.py
for n in [300, 600, 900, 1200, 1800, 2400, 3600]:
    for mode in ['horizontal', 'vertical', 'hybrid']:
        for data_type in ['linear', 'nonlinear']:
            run_experiment(d=10, K=3, n=n, mode=mode, data_type=data_type)
            record_metrics()
```

**Visualization**:
- Line plot: n (x-axis) vs F1 (y-axis)
- Separate lines for H/V/Hy
- Separate plots for linear/nonlinear

---

### 3.2 Dimensionality → Performance

**Research Question**: How does number of variables affect causal discovery?

**Experimental Design**:
- **Fixed**: n=1200, K=3 clients
- **Vary**: d ∈ {5, 8, 10, 12, 15, 20, 25, 30}
- **Modes**: Horizontal, Vertical, Hybrid
- **Data**: Linear and Nonlinear

**Key Considerations**:
- **Vertical mode**: Ensure min 4 features per client
  - d=5, K=3: Skip (too few features)
  - d=8, K=3: 2-3 features each (marginal)
  - d=12+: Feasible
- **Adaptive epochs**: Use V2 scaling (50/100/150)
- **Adaptive LR**: 0.01 × (8/d)^0.5

**Metrics to Track**:
- Same as 3.1 plus:
- Number of CI tests performed
- Tests per client (for vertical)
- SPN training time

**Expected Results**:
- F1 should decrease with d (more tests = more errors)
- Horizontal less affected (samples distributed)
- Vertical more challenged (feature splits)
- Hybrid should be middle ground

**Priority**: 🔴 CRITICAL (core ablation)

**Implementation**:
```python
# ablation_dimensionality.py
for d in [5, 8, 10, 12, 15, 20, 25, 30]:
    K = 3
    # Skip vertical if d < 3*K (too few features per client)
    modes = ['horizontal', 'vertical', 'hybrid'] if d >= 12 else ['horizontal', 'hybrid']

    for mode in modes:
        for data_type in ['linear', 'nonlinear']:
            epochs, lr = adaptive_hyperparams(d, K, n=1200)
            run_experiment(d=d, K=K, n=1200, mode=mode,
                          data_type=data_type, epochs=epochs, lr=lr)
```

**Visualization**:
- Line plot: d (x-axis) vs F1 (y-axis)
- Separate lines for H/V/Hy
- Log scale for x-axis if needed

---

### 3.3 Number of Clients → Performance

**Research Question**: How does federation granularity affect performance?

**Experimental Design**:
- **Fixed**: d=12, n=1200
- **Vary**: K ∈ {2, 3, 4, 5, 7, 10}
- **Modes**: Horizontal, Vertical, Hybrid
- **Data**: Linear and Nonlinear

**Key Considerations**:
- **Horizontal**: More clients = fewer samples per client
  - n_local = n / K
  - K=10 → only 120 samples/client (challenging)
- **Vertical**: More clients = fewer features per client
  - d_local ≈ d / K
  - K=10 → only 1-2 features/client (may skip)
- **Sample allocation**:
  - Option A: Fixed total n=1200, divide by K
  - Option B: Fixed per-client n=400, total = 400×K (varies)
  - Choose: Option A (more realistic)

**Metrics to Track**:
- Same as 3.1 plus:
- Communication rounds (if applicable)
- Aggregation overhead
- Per-client data stats

**Expected Results**:
- **Horizontal**: F1 decreases with K (less data per client)
- **Vertical**: F1 decreases with K (fewer features per client)
- **Hybrid**: More robust? (distributes burden)
- **Optimal K**: Likely K=3-5 for d=12, n=1200

**Priority**: 🟡 MEDIUM (useful insight)

**Implementation**:
```python
# ablation_num_clients.py
d, n = 12, 1200
for K in [2, 3, 4, 5, 7, 10]:
    # Skip vertical if d/K < 4
    modes = ['horizontal', 'vertical', 'hybrid'] if d/K >= 4 else ['horizontal', 'hybrid']

    for mode in modes:
        for data_type in ['linear', 'nonlinear']:
            n_per_client = n // K
            if mode == 'horizontal' and n_per_client < 50:
                print(f"Warning: Only {n_per_client} samples/client for K={K}")

            run_experiment(d=d, K=K, n=n, mode=mode, data_type=data_type)
```

**Visualization**:
- Line plot: K (x-axis) vs F1 (y-axis)
- Separate lines for H/V/Hy
- Annotate: samples/client for horizontal, features/client for vertical

---

## Part 4: Experimental Setup

### 4.1 Directory Structure

```
experiments/v3_comprehensive_fixes/
├── data/
│   ├── synthetic/
│   │   ├── linear/
│   │   └── nonlinear/
│   ├── real_world/
│   │   ├── sachs/
│   │   ├── law_school/
│   │   └── hyperpc_benchmarks/
│   └── preprocessing/
│       ├── download_datasets.py
│       ├── preprocess_real_data.py
│       └── create_federated_splits.py
├── baselines/
│   ├── fedcdh_original/       # Original FedCDH-H
│   ├── centralized/           # PC, GIES, FCI on pooled data
│   ├── federated_methods/     # Other fed causal methods
│   └── vertical_hybrid/       # Vertical/Hybrid baselines
├── ablations/
│   ├── sample_size/           # n ablation results
│   ├── dimensionality/        # d ablation results
│   └── num_clients/           # K ablation results
├── scripts/
│   ├── run_v3_fixes.py              # Critical fixes
│   ├── run_v3_baselines.py          # Baseline comparisons
│   ├── run_ablation_sample_size.py  # n ablation
│   ├── run_ablation_dimensionality.py  # d ablation
│   └── run_ablation_num_clients.py  # K ablation
├── analysis/
│   ├── generate_v3_report.py
│   ├── compare_baselines.py
│   ├── ablation_analysis.py
│   └── publication_figures.py
└── results/
    ├── fixes/                 # V2 vs V3 comparison
    ├── baselines/            # Baseline comparisons
    └── ablations/            # Ablation study results
```

---

### 4.2 Metrics to Report (Standardized)

**For All Experiments**:

1. **Causal Discovery Metrics**:
   - Skeleton F1 Score
   - Structural Hamming Distance (SHD)
   - Precision (edges)
   - Recall (edges)
   - Accuracy (overall)

2. **CI Test Metrics**:
   - CI Test Accuracy (vs ground truth)
   - Number of CI tests performed
   - Average p-value distribution

3. **SPN Quality Metrics**:
   - Train Log-Likelihood (LL)
   - Test Log-Likelihood
   - MMD² (distribution match)
   - KS Test pass rate

4. **Computational Metrics**:
   - Training time (seconds)
   - Memory usage (MB)
   - Number of parameters

5. **Federated Metrics** (if applicable):
   - Communication rounds
   - Data transferred (MB)
   - Privacy guarantees (if any)

---

### 4.3 Comparison Tables

#### **Table 1: Baseline Comparison (Sachs Dataset)**

| Method | Mode | F1 | SHD | Precision | Recall | CI Acc | Time (s) |
|--------|------|-----|-----|-----------|--------|---------|----------|
| Centralized PC | - | - | - | - | - | - | - |
| Centralized GIES | - | - | - | - | - | - | - |
| FedCDH (Original) | H | - | - | - | - | - | - |
| FedCDH-SPN (V2) | H | - | - | - | - | - | - |
| FedCDH-SPN (V3) | H | - | - | - | - | - | - |
| FedCDH-SPN (V3) | V | - | - | - | - | - | - |
| FedCDH-SPN (V3) | Hy | - | - | - | - | - | - |
| Other Method 1 | - | - | - | - | - | - | - |
| Other Method 2 | - | - | - | - | - | - | - |

---

#### **Table 2: Ablation - Sample Size (d=10, K=3, Linear)**

| n | Mode | F1 | SHD | CI Acc | Train LL | Time (s) |
|---|------|-----|-----|---------|----------|----------|
| 300 | H | - | - | - | - | - |
| 300 | V | - | - | - | - | - |
| 300 | Hy | - | - | - | - | - |
| 600 | H | - | - | - | - | - |
| ... | ... | ... | ... | ... | ... | ... |
| 3600 | Hy | - | - | - | - | - |

---

#### **Table 3: Ablation - Dimensionality (n=1200, K=3, Linear)**

| d | Mode | F1 | SHD | Num Tests | Tests/Client (V) | Time (s) |
|---|------|-----|-----|-----------|------------------|----------|
| 5 | H | - | - | - | - | - |
| 5 | Hy | - | - | - | - | - |
| 8 | H | - | - | - | - | - |
| 8 | V | - | - | - | 3 | - |
| ... | ... | ... | ... | ... | ... | ... |
| 30 | Hy | - | - | - | - | - |

---

#### **Table 4: Ablation - Number of Clients (d=12, n=1200, Linear)**

| K | Mode | F1 | SHD | n/client (H) | d/client (V) | Time (s) |
|---|------|-----|-----|--------------|--------------|----------|
| 2 | H | - | - | 600 | - | - |
| 2 | V | - | - | - | 6 | - |
| 2 | Hy | - | - | 600 | 6 | - |
| 3 | H | - | - | 400 | - | - |
| ... | ... | ... | ... | ... | ... | ... |
| 10 | Hy | - | - | 120 | 1.2 | - |

---

## Part 5: Implementation Roadmap (Updated)

### **Phase 0: Setup (Week 0) - 4-6 hours**
- [x] Create v3 branch
- [x] Extract HyperPC data
- [ ] Research federated causal baselines (2 hrs)
  - Search papers: "federated causal discovery" 2023-2026
  - Identify 2-3 comparable methods
  - Check code availability
- [ ] Download Law School dataset (1 hr)
- [ ] Setup experiment directory structure (1 hr)

---

### **Phase 1: Critical Fixes (Week 1) - 10-12 hours**
- [ ] Fix #1: GlobalSumOfProducts (4-6 hrs)
- [ ] Fix #2: Horizontal aggregation (3-4 hrs)
- [ ] Fix #3: Vertical constraints (2 hrs)
- [ ] Test all fixes on SMALL config (1 hr)

**Deliverable**: V3 with all critical fixes

---

### **Phase 2: Baselines (Week 2) - 12-16 hours**
- [ ] Implement centralized baselines (2-3 hrs)
  - PC, GIES, FCI on pooled data
- [ ] Run original FedCDH (2-3 hrs)
  - Extract from fedcdh_code.zip
  - Run on Sachs + synthetic
- [ ] Implement/adapt other federated methods (6-8 hrs)
  - Research-dependent
  - 2-3 methods
- [ ] Compare on Sachs dataset (2 hrs)
  - Run all methods
  - Generate comparison table

**Deliverable**: Baseline comparison results

---

### **Phase 3: Real-World Data (Week 2-3) - 8-12 hours**
- [ ] Sachs experiments (4 hrs)
  - H/V/Hy modes
  - V2 vs V3 comparison
- [ ] Law School experiments (4 hrs)
  - Preprocess + federated splits
  - H/V/Hy modes
- [ ] (Optional) HyperPC benchmarks (4 hrs)

**Deliverable**: Real-world validation results

---

### **Phase 4: Ablation Studies (Week 3-4) - 20-30 hours**
- [ ] Sample size ablation (8-10 hrs)
  - 7 values of n × 3 modes × 2 data types = 42 experiments
  - ~10-15 min each = 7-10 hours runtime
- [ ] Dimensionality ablation (8-10 hrs)
  - 8 values of d × 2-3 modes × 2 data types = ~40 experiments
  - ~10-15 min each = 7-10 hours runtime
- [ ] Number of clients ablation (4-6 hrs)
  - 6 values of K × 2-3 modes × 2 data types = ~30 experiments
  - ~5-10 min each = 3-5 hours runtime
- [ ] Analysis and visualization (4 hrs)

**Deliverable**: Complete ablation study results

---

### **Phase 5: Analysis & Reporting (Week 4) - 12-16 hours**
- [ ] Generate V3 comprehensive report (6 hrs)
  - Extend V2 report with new tabs
  - Add baseline comparisons
  - Add ablation visualizations
- [ ] Create publication figures (4 hrs)
  - V2 vs V3 improvements
  - Baseline comparison bar charts
  - Ablation line plots
- [ ] Write V3 summary document (2-3 hrs)
- [ ] Prepare thesis-ready materials (2-3 hrs)

**Deliverable**: V3 final report + publication figures

---

## Part 6: Time Estimates (Updated)

| Phase | Description | Hours | Priority |
|-------|-------------|-------|----------|
| 0 | Setup + Research | 4-6 | 🔴 CRITICAL |
| 1 | Critical Fixes | 10-12 | 🔴 CRITICAL |
| 2 | Baselines | 12-16 | 🔴 CRITICAL |
| 3 | Real-World Data | 8-12 | 🔴 CRITICAL |
| 4 | Ablation Studies | 20-30 | 🟡 MEDIUM |
| 5 | Analysis & Reporting | 12-16 | 🟡 MEDIUM |
| **Total** | | **66-92 hours** | **3-4 weeks** |

**Critical Path** (minimum viable):
- Phase 1 (10 hrs) + Sachs baseline (4 hrs) + Sample size ablation (8 hrs) + Basic report (4 hrs) = **26 hours** (~4 days)

---

## Part 7: Questions for User

1. **Scope Decision**:
   - Full plan (66-92 hours, 3-4 weeks)?
   - Or minimum viable (26 hours, 4 days)?

2. **Baseline Priority**:
   - Which baselines are most important?
   - Should we focus on centralized comparison only?
   - Or also implement other federated methods?

3. **Ablation Priority**:
   - All three ablations (n, d, K)?
   - Or focus on most important (sample size)?

4. **Dataset Priority**:
   - Sachs only or also Law School + HyperPC?
   - Real-world vs synthetic emphasis?

5. **Timeline**:
   - Thesis defense date?
   - How much time available for V3?

---

## Part 8: Next Immediate Steps

**This Week**:
1. ✅ Create v3 branch
2. ✅ Extract HyperPC data
3. ✅ Write updated V3 plan
4. ⬜ User decision on scope
5. ⬜ Start Phase 1: Implement GlobalSumOfProducts

**Command to Start**:
```bash
cd /Users/M279402/PycharmProjects/fl_spn_CDH
git checkout v3-comprehensive-fixes

# Phase 1: Critical fixes
# Start with most critical: GlobalSumOfProducts for hybrid mode
```

---

**Status**: 📋 Awaiting user decision on scope and priorities
