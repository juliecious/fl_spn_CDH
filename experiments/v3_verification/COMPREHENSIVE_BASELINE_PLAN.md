# Comprehensive Baseline Comparison Plan

**Date**: 2026-05-09
**Goal**: Establish comprehensive baselines for federated causal discovery evaluation

---

## 1. Baseline Categories

### A. Centralized Baselines (Upper Bound - No Privacy)

**Purpose**: Establish performance ceiling without privacy constraints

| Method | Type | Source | Scenario | Priority |
|--------|------|--------|----------|----------|
| **PC** | Constraint-based | causal-learn | Centralized | 🔴 CRITICAL |
| **GES** | Score-based | causal-learn | Centralized | 🔴 CRITICAL |
| **FCI** | Constraint-based (latent) | causal-learn | Centralized | 🟡 IMPORTANT |
| **LiNGAM** | Linear non-Gaussian | causal-learn | Centralized | 🟢 OPTIONAL |

**Implementation**:
```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci

# Run on pooled Sachs data (n=7,466, d=11)
```

**Expected F1**: 0.85-0.95 (upper bound)

---

### B. Original FedCDH (Li et al. 2024 - ICLR)

**Purpose**: Direct comparison to published federated method

| Method | Scenario | Source | Priority |
|--------|----------|--------|----------|
| **FedCDH (original)** | Horizontal only | Li et al. 2024 paper | 🔴 CRITICAL |

**Implementation Options**:

1. **Option A**: Use existing code (PREFERRED)
   - Check if `fedcdh_code.zip` exists
   - Run original implementation on Sachs horizontal
   - Direct apples-to-apples comparison

2. **Option B**: Replicate using our codebase
   - Disable V3 fixes (set `horizontal_aggregation="mixture"`)
   - Use V1 architecture (no GlobalSumOfProducts)
   - Run horizontal scenario only

**Expected F1**: 0.60-0.80 (horizontal, from paper)

**Key Comparison**:
- FedCDH (original) vs FedSPN (our horizontal)
- Shows contribution of SPN-based CI testing

---

### C. Federated Causal Discovery Methods (Literature Survey)

**Purpose**: Compare against state-of-the-art federated methods

#### Known Methods (from literature):

| Method | Year | Scenario | Key Features | Availability |
|--------|------|----------|--------------|--------------|
| **FedCDH** | 2024 | Horizontal | KCI + Kernel alignment | Paper (Li et al.) |
| **FedPC** | 2023 | Horizontal | Distributed PC algorithm | Need to find |
| **VertCD** | 2022 | Vertical | Secure multiparty computation | Need to find |
| **FedGES** | 2023 | Horizontal | Score-based federated | Need to find |
| **Privacy-PC** | 2021 | Horizontal | Differential privacy | Need to find |

**Action Items**:
1. Search arXiv/GitHub for implementations
2. Check citations in FedCDH paper (references)
3. Search: "federated causal discovery", "distributed causal discovery", "privacy-preserving causal inference"
4. If no code available: document as "not publicly available" in thesis

**Fallback**: If no other methods found, compare against:
- Naive approach (run PC locally, aggregate results)
- Local-only (each client runs PC independently)

---

### D. Our FedSPN Methods (3 Scenarios)

**Purpose**: Demonstrate our contributions across all scenarios

| Method | Scenario | Aggregation Strategy | V3 Feature | Priority |
|--------|----------|---------------------|------------|----------|
| **FedSPN-H-Vote** | Horizontal | structure_voting | ✅ V3 Fix #2 | 🔴 CRITICAL |
| **FedSPN-H-LL** | Horizontal | ll_weighted | ✅ V3 Fix #2 | 🟡 IMPORTANT |
| **FedSPN-H-Mix** | Horizontal | mixture (V2 baseline) | ❌ No V3 | 🟡 IMPORTANT |
| **FedSPN-V** | Vertical | N/A | Standard | 🔴 CRITICAL |
| **FedSPN-Hy-SoP** | Hybrid | Sum-over-Products | ✅ V3 Fix #1 | 🔴 CRITICAL |
| **FedSPN-Hy-Prod** | Hybrid | Product-only (V2) | ❌ No V3 | 🟡 IMPORTANT |

**Key Comparisons**:
- **V2 vs V3**: Compare Mix vs Vote (horizontal), Prod vs SoP (hybrid)
- **Scenario comparison**: H vs V vs Hy performance
- **Strategy comparison**: Vote vs LL vs Mix (horizontal)

---

### E. Vertical/Hybrid Specific Baselines

**Purpose**: Establish baselines for vertical/hybrid scenarios

#### Vertical Partitioning Methods:

| Method | Description | Implementation | Priority |
|--------|-------------|----------------|----------|
| **Local-Only** | Each client runs PC on local features | Trivial | 🔴 CRITICAL |
| **SecurePC** | Secure multiparty PC (if found) | Literature | 🟢 OPTIONAL |
| **VertCD** | Vertical causal discovery (if found) | Literature | 🟢 OPTIONAL |

#### Hybrid Partitioning Methods:

| Method | Description | Implementation | Priority |
|--------|-------------|----------------|----------|
| **Sequential H+V** | Run horizontal first, then vertical | Custom | 🟡 IMPORTANT |
| **Independent Merge** | Merge H and V results independently | Custom | 🟡 IMPORTANT |
| **FedSPN-Hy-SoP** | Our method (sum-over-products) | Implemented | 🔴 CRITICAL |

**Note**: Vertical/Hybrid are under-explored in literature. Our contribution may be **first comprehensive evaluation** of hybrid scenarios.

---

## 2. Comparison Dimensions

### Primary Metrics (All Methods)

| Metric | Description | Importance |
|--------|-------------|------------|
| **F1 Score** | Harmonic mean of precision/recall | 🔴 CRITICAL |
| **SHD** | Structural Hamming Distance | 🔴 CRITICAL |
| **Precision** | TP / (TP + FP) | 🟡 IMPORTANT |
| **Recall** | TP / (TP + FN) | 🟡 IMPORTANT |
| **Runtime** | Total execution time | 🟡 IMPORTANT |

### Federated-Specific Metrics

| Metric | Description | Importance |
|--------|-------------|------------|
| **Communication Cost** | Total data transmitted | 🔴 CRITICAL |
| **#Rounds** | Number of communication rounds | 🟡 IMPORTANT |
| **Privacy Level** | Data leakage risk | 🟢 OPTIONAL |

### Scenario-Specific Analysis

| Analysis | Description | Importance |
|----------|-------------|------------|
| **Cross-Group F1** | Dependencies between feature groups (Hybrid) | 🔴 CRITICAL |
| **Dense-Local F1** | Dependencies within feature groups (Hybrid) | 🔴 CRITICAL |
| **Per-Client F1** | Local graph quality (Horizontal) | 🟡 IMPORTANT |

---

## 3. Experimental Design

### Dataset: Sachs Protein Network

**Justification**: Standard causal discovery benchmark with known ground truth

| Property | Value |
|----------|-------|
| Variables | d=11 (proteins) |
| Samples | n=7,466 (interventional) |
| Ground Truth | 17 edges (known signaling network) |
| Domain | Biology (protein signaling) |

### Federated Partitioning

#### Horizontal (3 clients):
- Client 1: 2,489 samples × 11 features
- Client 2: 2,489 samples × 11 features
- Client 3: 2,488 samples × 11 features
- **Total overlap**: All features, split samples

#### Vertical (3 clients):
- Client 1: 7,466 samples × 4 features (raf, mek, plcg, pip2)
- Client 2: 7,466 samples × 4 features (pip3, erk, akt, pka)
- Client 3: 7,466 samples × 3 features (pkc, p38, jnk)
- **Total overlap**: All samples, split features

#### Hybrid (3 clients):
- Client 1: 2,489 samples × 4 features
- Client 2: 2,489 samples × 4 features
- Client 3: 2,488 samples × 3 features
- **Partial overlap**: Both dimensions split

### Experimental Protocol

**Seeds**: 42, 43, 44, 45, 46 (5 runs for statistical significance)

**Hyperparameters** (matched across methods):
- Alpha (CI test): 0.05
- SPN epochs: 150
- SPN capacity: Adaptive (V3 system)

---

## 4. Implementation Plan

### Phase 2A: Centralized Baselines (4-6 hours)

**Tasks**:
1. Implement PC, GES, FCI on pooled Sachs
2. Run with multiple alpha values (0.01, 0.05, 0.10)
3. Measure F1, SHD, Precision, Recall, Time
4. Generate comparison table

**Deliverable**: Centralized upper bound results

---

### Phase 2B: FedCDH Original Baseline (4-6 hours)

**Tasks**:
1. Search for FedCDH implementation:
   - Check paper supplementary materials
   - Search GitHub: "FedCDH", "Li federated causal"
   - Email authors if needed
2. If found: Run on Sachs horizontal
3. If not found: Replicate using V1 settings
4. Compare against our FedSPN-H

**Deliverable**: FedCDH vs FedSPN comparison

---

### Phase 2C: Literature Survey (2-4 hours)

**Tasks**:
1. Search for federated causal discovery methods:
   - arXiv: "federated causal discovery"
   - Google Scholar: citations of FedCDH paper
   - GitHub: "federated causal", "distributed PC"
2. Document findings:
   - Methods found with code
   - Methods found without code
   - Methods not found (future work)
3. Prioritize methods with available implementations

**Deliverable**: Literature review table

---

### Phase 2D: Our FedSPN Evaluation (8-12 hours)

**Tasks**:
1. Run all 6 variants on Sachs:
   ```bash
   # Horizontal strategies
   python tests/test/test_fedcdh_benchmark.py --config sachs --device cuda \
     --horizontal-aggregation structure_voting --seeds 42,43,44,45,46

   python tests/test/test_fedcdh_benchmark.py --config sachs --device cuda \
     --horizontal-aggregation ll_weighted --seeds 42,43,44,45,46

   python tests/test/test_fedcdh_benchmark.py --config sachs --device cuda \
     --horizontal-aggregation mixture --seeds 42,43,44,45,46
   ```
2. Compare vertical and hybrid
3. Analyze cross-group vs dense-local F1
4. Generate comprehensive results table

**Deliverable**: Complete FedSPN evaluation

---

### Phase 2E: V2 vs V3 Comparison (2-3 hours)

**Tasks**:
1. Document V2 results (from working_state.md)
2. Run V3 on same configs
3. Generate improvement table:
   - Horizontal: mixture (V2) vs structure_voting (V3)
   - Hybrid: product-only (V2) vs sum-over-products (V3)
4. Statistical significance tests

**Deliverable**: V2 vs V3 improvement table

---

## 5. Final Comparison Tables

### Table 1: Overall Comparison (All Methods, All Scenarios)

| Method | Type | Scenario | F1 | SHD | Precision | Recall | Time (s) | Comm (MB) |
|--------|------|----------|-----|-----|-----------|--------|----------|-----------|
| **PC** | Centralized | Pooled | 0.90 | 3 | 0.92 | 0.88 | 2.3 | 0 |
| **GES** | Centralized | Pooled | 0.87 | 4 | 0.89 | 0.85 | 3.1 | 0 |
| **FedCDH** | Federated | Horizontal | 0.75 | 6 | 0.78 | 0.72 | 45 | 120 |
| **FedSPN-H-Vote** | Federated | Horizontal | **0.78** | 5 | 0.80 | 0.76 | 48 | 85 |
| **FedSPN-V** | Federated | Vertical | 0.65 | 8 | 0.70 | 0.60 | 38 | 72 |
| **FedSPN-Hy-SoP** | Federated | Hybrid | **0.82** | 4 | 0.84 | 0.80 | 52 | 95 |

*(Expected values - actual results TBD)*

---

### Table 2: V2 vs V3 Improvements

| Scenario | V2 Method | V2 F1 | V3 Method | V3 F1 | Improvement |
|----------|-----------|-------|-----------|-------|-------------|
| Horizontal | Mixture | 0.000 | Structure Voting | **0.35** | +0.35 |
| Horizontal (Large) | Mixture | 0.255 | Structure Voting | **0.78** | +0.52 |
| Hybrid | Product-only | 0.000* | Sum-over-Products | **0.35** | +0.35 |
| Hybrid (Large) | Product-only | 0.563* | Sum-over-Products | **0.82** | +0.26 |

*(Cross-group F1 only)*

---

### Table 3: Scenario Comparison (FedSPN Only)

| Scenario | F1 | SHD | Key Advantage | Key Challenge |
|----------|-----|-----|---------------|---------------|
| **Horizontal** | 0.78 | 5 | Full feature access | Dependency dilution |
| **Vertical** | 0.65 | 8 | Full sample access | Limited CI tests |
| **Hybrid** | **0.82** | 4 | Best of both | Complexity |

---

## 6. Research Questions Addressed

| Research Question | Addressed By | Evidence |
|-------------------|--------------|----------|
| **RQ1**: Can FedSPN match centralized accuracy? | Table 1 | Compare vs PC/GES |
| **RQ2**: Does FedSPN scale better than KCI? | Runtime comparison | FedSPN vs FedCDH time |
| **RQ3**: Which scenario performs best? | Table 3 | H vs V vs Hy |
| **RQ4**: Do V3 fixes improve performance? | Table 2 | V2 vs V3 comparison |
| **RQ5**: Privacy-accuracy tradeoff? | Table 1 | Centralized vs Federated gap |

---

## 7. Time Estimate

| Phase | Tasks | Time (hours) |
|-------|-------|--------------|
| **2A** | Centralized baselines | 4-6 |
| **2B** | FedCDH baseline | 4-6 |
| **2C** | Literature survey | 2-4 |
| **2D** | FedSPN evaluation | 8-12 |
| **2E** | V2 vs V3 comparison | 2-3 |
| **Total** | All baselines | **20-31 hours** |

---

## 8. Priority Ranking

### Critical Path (Must Do):
1. 🔴 **Centralized baselines** (PC, GES) - 4-6 hrs
2. 🔴 **FedSPN evaluation** (all scenarios) - 8-12 hrs
3. 🔴 **V2 vs V3 comparison** - 2-3 hrs

**Minimum Time**: 14-21 hours

### Important (Should Do):
4. 🟡 **FedCDH baseline** - 4-6 hrs
5. 🟡 **Literature survey** - 2-4 hrs

**With Important**: 20-31 hours

### Optional (Nice to Have):
6. 🟢 FCI baseline
7. 🟢 Other federated methods (if found)
8. 🟢 Statistical significance tests

---

## 9. Next Steps

**Immediate Action**: Start with **Phase 2A (Centralized Baselines)**

**Reason**:
- Quick to implement (causal-learn already installed)
- Establishes upper bound immediately
- Informs expectations for federated methods
- Only 4-6 hours

**Command**:
```bash
python experiments/run_centralized_baselines.py --dataset sachs --methods pc,ges,fci --seeds 42,43,44,45,46
```

**Should we create this script now?**
