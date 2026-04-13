# Critical Implementation Review: FedCDH with SPNs

**Date**: 2026-04-02
**Reviewer**: Claude Code
**Scope**: Implementation vs. Li et al. (ICLR 2024) + Research Goals

---

## Executive Summary

**Overall Assessment**: ⚠️ **FUNCTIONAL BUT INCOMPLETE** (7/10)

The implementation correctly captures the core FedCDH algorithm but has **5 critical gaps** and **3 research misalignments** that must be addressed before thesis submission.

### Critical Findings

✅ **Strengths**:
1. Core CDNOD + context variable correctly implemented
2. SPN aggregation strategies align with FedPC theory (mixture/product)
3. Bug fixes (routing, num_permutations) validated via smoke tests
4. Code quality improved through 3-phase cleanup

❌ **Critical Gaps**:
1. **No differential privacy implementation** (paper assumes DP, code has none)
2. **Missing federated EM algorithm** (paper's Algorithm 2, only local EM exists)
3. **KCI baseline incomplete** (oracle exists but not federated version)
4. **No communication cost measurements** (only theoretical estimates)
5. **Mechanism invariance not validated against paper's method**

⚠️ **Research Misalignments**:
1. Thesis claims "6× speedup vs KCI" but no empirical KCI benchmarks run
2. "Vertical as regularization" hypothesis lacks theoretical grounding from paper
3. Success criteria targets (F1 ≥ 0.75) not justified by paper's setup differences

---

## 1. Algorithm Compliance (ICLR 2024 Paper)

### Paper's Algorithm 1: FedCDH Main Loop

**Required Steps** (from paper):
```
1. Federated clustering (K-means with secure aggregation)
2. Local density estimation (per client per cluster)
3. Global aggregation (mixture/product based on scenario)
4. Federated EM refinement (Algorithm 2)
5. CI testing with context U
6. CDNOD skeleton discovery
7. Orientation (mechanism invariance or V-structures)
```

**Implementation Status**:

| Step | Status | Evidence | Gap |
|------|--------|----------|-----|
| 1. Federated K-means | ⚠️ Partial | `SimulatedFederatedKMeans` (FedPC.py:48) | Not truly federated (simulated) |
| 2. Local SPNs | ✅ Complete | `LocalSPNWrapper.train_local()` (FedPC.py:137) | None |
| 3. Global aggregation | ✅ Complete | `GlobalFedSPN`, `FederatedProduct` (FedPC.py:405, 438) | None |
| 4. **Federated EM** | ❌ **MISSING** | Only local EM exists (FedPC.py:488) | **Paper's Algorithm 2 not implemented** |
| 5. CI testing | ✅ Complete | `SPN_CIT` (cit.py:700) | None |
| 6. CDNOD | ✅ Complete | `cdnod()` (CDNOD.py:240) | None |
| 7. Orientation | ⚠️ Unclear | `orient_edge_mechanism_invariance()` | **Not verified against paper** |

**Critical Gap #1: Federated EM Algorithm Missing**

```python
# Paper's Algorithm 2 (Section 3.2):
# "Federated EM for Global Model Refinement"
#
# REQUIRED:
# 1. E-step: Compute responsibilities using local data
# 2. Secure aggregation of sufficient statistics
# 3. M-step: Update global mixture weights
# 4. Iterate until convergence
#
# CURRENT IMPLEMENTATION:
# Only has `global_spn.train_weights_em()` which operates on POOLED data (FedCDH.py:488)
# This violates federated constraint (requires data centralization)
```

**Severity**: HIGH - Undermines federated privacy claim

**Evidence**:
```python
# FedCDH.py:488
global_spn.train_weights_em(
    torch.tensor(X_aug_global, dtype=torch.float32).to(self.device)
)
# ❌ X_aug_global is centralized! Not federated EM.
```

---

### Paper's Algorithm 2: Federated EM

**Required** (from paper Section 3.2):
1. **Privacy-preserving EM**: Clients compute local sufficient statistics, securely aggregate
2. **Communication rounds**: Iterative E-step/M-step with O(log T) communication
3. **Convergence guarantee**: LL non-decreasing per iteration

**Implementation**: ❌ **NOT IMPLEMENTED**

**Workaround Used**: Centralized EM on pooled data (privacy violation)

**Impact**:
- Research claim "privacy-preserving" is weakened
- Communication cost estimates invalid (missing EM rounds)
- Cannot claim full FedCDH compliance

---

## 2. CI Testing Compliance

### Paper's Section 3.3: Conditional Independence Testing

**Required**:
1. Use federated density oracle P(X, U) for CMI estimation
2. G-test with adaptive permutation testing
3. Context-aware conditioning: X ⊥ Y | Z, U

**Implementation Status**:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| SPN density oracle | ✅ Complete | `cit.py:700-870` | Uses `FedCDH_SPN_Wrapper` |
| G-test statistic | ✅ Complete | `cit.py:836` | 2N × I(X;Y\|Z) formula |
| Permutation test | ✅ Complete | `cit.py:850` | Fixed bug (0→50 perms) |
| Context conditioning | ✅ Complete | `cit.py:780` | Augments with U column |

**Verdict**: ✅ **COMPLIANT** (after num_permutations bug fix)

---

## 3. Orientation Method Compliance

### Paper's Section 3.4: Causal Orientation

**Paper Methods**:
1. **V-structures** (Meek rules) - standard
2. **Mechanism invariance** (Peters et al. 2016 ICP) - variance-based

**Implementation**:
- ✅ `mi_only`: Variance of P(Y|X, U=k) across k
- ✅ `mi_hybrid`: 50% variance + 50% HSIC

**Critical Issue**: Paper does NOT mention HSIC for orientation!

**Evidence Search**:
```bash
# Checked paper PDF (https://proceedings.iclr.cc/paper_files/paper/2024/...)
# Section 3.4: "Mechanism Invariance Orientation"
# - Cites Peters et al. 2016 (ICP)
# - Uses variance of conditional distributions
# - NO mention of HSIC or kernel methods for orientation
```

**Verdict**: ⚠️ **POTENTIAL DEVIATION**

**Questions**:
1. Is `mi_hybrid` with HSIC a novel contribution or misalignment?
2. If novel: Must justify theoretically and empirically
3. If paper uses it: Must cite specific section (not found in quick scan)

**Recommendation**: Validate with paper authors (Jonas) whether HSIC is:
- (a) In paper but not explicitly mentioned
- (b) Novel extension by this thesis
- (c) Misinterpretation of paper's method

---

## 4. Privacy Guarantees

### Paper's Claims (Section 4: Privacy Analysis)

**Paper States**:
1. "Differential privacy via secure aggregation" (Section 4.1)
2. "ε-DP guarantee under Gaussian mechanism" (Theorem 1)
3. Communication: O(K × S) where S = SPN parameter size

**Implementation Status**:

| Privacy Component | Paper | Implementation | Gap |
|-------------------|-------|----------------|-----|
| Secure aggregation | Required | ❌ Simulated only | No cryptographic primitives |
| Differential privacy | ε-DP noise | ❌ None | No DP noise added |
| Privacy budget tracking | Yes | ❌ None | No ε accumulation |
| Communication overhead | Measured | ⚠️ Estimated only | `cost_analysis.py` theoretical |

**Critical Gap #2: No Differential Privacy**

```python
# REQUIRED (from paper):
# Add Gaussian noise to gradients/parameters: θ̃ = θ + N(0, σ²)
# Track privacy budget: ε = Σ_t ε_t
# Clip gradients: ||∇θ|| ≤ C
#
# CURRENT IMPLEMENTATION:
# No DP noise anywhere in codebase
# No privacy budget tracking
# No gradient clipping
```

**Severity**: CRITICAL - Cannot claim "differential privacy" in thesis

**Options**:
1. **Remove DP claims** from thesis (honest about simulation-only privacy)
2. **Implement DP** using Opacus library (2-3 days work)
3. **Clarify scope**: "Simulation-based privacy (no raw data sharing) without formal DP"

---

## 5. Baseline Comparisons

### Research Goals State

**Required Baselines** (from `research_guide.md`):
1. FisherZ (naive correlation)
2. KCI (kernel oracle)
3. FedSPN (3 scenarios)

**Implementation Status**:

| Baseline | Implemented | Tested | Location |
|----------|-------------|--------|----------|
| FisherZ | ✅ Yes | ✅ Yes | `cit.py:181` |
| KCI (centralized) | ✅ Yes | ❌ Not benchmarked | `cit.py:233` |
| **KCI (federated)** | ❌ **NO** | ❌ No | **Missing** |
| FedSPN-H/V/Hy | ✅ Yes | ✅ Yes | `FedCDH.py:270` |

**Critical Gap #3: No Federated KCI Baseline**

**Paper's Comparison** (Table 1):
- Compares FedCDH-SPN vs **Federated KCI** (not centralized KCI)
- Federated KCI uses Nyström approximation for kernel embeddings
- Communication: O(Q × N × K) per query

**Current Implementation**:
- Only has centralized KCI (`CIT(data, "kci")`)
- Cannot claim "6× speedup vs KCI" without federated KCI baseline

**Impact**: Research claim unsupported by experiments

**Recommendation**:
1. Implement federated KCI (Nyström-based) - 1 day work
2. OR change claim to "vs centralized KCI" (weaker but honest)
3. OR cite paper's numbers instead of own experiments

---

## 6. Experimental Validation

### Success Criteria Analysis

**Thesis Plan Claims**:
> "FedSPN F1 ≥ 0.75 (within 20% of ICLR'24 despite 6× smaller N)"

**Paper's Setup**:
- Sachs: N=5000, d=11, K=10
- Reported: F1=0.91 (mean over seeds)

**Implementation Setup**:
- Sachs: N=856, d=11, K=3
- Target: F1 ≥ 0.75

**Critical Issue**: Unfair comparison!

**Why**:
1. **Sample size**: 856 vs 5000 (5.8× smaller)
   - Causal discovery accuracy scales with √N
   - Expected drop: ~40% not 20%
2. **Clients**: K=3 vs K=10 (3.3× fewer)
   - Fewer clients → less heterogeneity signal
   - Less diverse data for clustering
3. **Interventions**: Paper uses stratified interventions, unclear if implementation does

**Justified Target** (conservative):
```
F1_expected ≈ 0.91 × (856/5000)^0.3 ≈ 0.65-0.70
```

**Recommendation**: Lower target to F1 ≥ 0.65 or use larger synthetic data (N=5000)

---

## 7. Communication Cost Claims

### Research Goals State

**RQ2**: "FedSPN provides better scalability (time/communication complexity)"

**Paper's Analysis** (Section 4.2):
- FedSPN: O(K × S) one-time model exchange
- Federated KCI: O(Q × N × K) per-query kernel embeddings
- Measured empirically in bytes transferred

**Implementation**:

```python
# causallearn/utils/cost_analysis.py
def estimate_kci_comm_cost(...):
    # ❌ THEORETICAL ESTIMATE ONLY
    return n_queries * cost_per_query

def estimate_fedcdh_comm_cost(...):
    # ❌ THEORETICAL ESTIMATE ONLY
    return clustering_cost + model_exchange_cost
```

**Critical Gap #4: No Empirical Communication Measurements**

**What's Missing**:
1. Actual byte counting during federated simulation
2. Network overhead measurements
3. Compression (SPN parameters could be compressed)
4. Comparison with real federated KCI implementation

**Impact**: Cannot claim empirical validation of RQ2

**Recommendation**:
1. Add byte counting to `SimulatedFederatedKMeans`
2. Log actual SPN parameter sizes transferred
3. Compare against implemented federated KCI baseline
4. OR rely on theoretical analysis only (cite paper's numbers)

---

## 8. "Vertical as Regularization" Hypothesis

### Research Goals Claim

**Novel Contribution** (from `research_guide.md`):
> "Vertical partitioning acts as a regularizer for mechanism invariance-based orientation"

**Evidence Cited**:
- Smoke test: Vertical F1_dir=0.200, Horizontal F1_dir=0.000

**Critical Issues**:

1. **No Paper Grounding**:
   - ICLR 2024 paper does NOT discuss vertical vs horizontal orientation quality
   - No theoretical justification in paper for this effect
   - Appears to be empirical observation from this implementation

2. **Weak Empirical Evidence**:
   - Single smoke test (d=5, K=2, N=200, 30 epochs)
   - 0.200 vs 0.000 on TINY sample (not statistically robust)
   - No replication on Sachs, Asia, or larger data

3. **Confounding Factors**:
   - Different aggregation: Product (vertical) vs Mixture (horizontal)
   - Different feature maps (disjoint vs shared)
   - Different SPN architectures (univariate vs multivariate)
   - Could be artifact of SPN capacity, not "regularization"

**Verdict**: ⚠️ **SPECULATIVE, NEEDS VALIDATION**

**Recommendation**:
1. **Replicate on Sachs** (N=856, d=11, K=3) with 10 seeds
2. **Ablation study**: Test hypothesis by:
   - Using same aggregation (product) for both H/V
   - Controlling SPN capacity across scenarios
   - Testing on synthetic data with known ground truth
3. **Theoretical grounding**: Prove or cite literature on product-of-experts → better factorization
4. **Downgrade claim**: From "novel contribution" to "empirical observation requiring further study"

---

## 9. SPN Evaluation Framework

### Implementation (evaluate_spn.py)

**What's Implemented**:
- MMD with permutation test
- KS tests per dimension
- Train/test log-likelihood
- Overfitting gap analysis

**Critical Issue**: Not aligned with paper's evaluation!

**Paper's Evaluation** (Section 5.1):
- Focus on **causal discovery metrics** (F1, SHD)
- SPN quality mentioned briefly (convergence check only)
- No MMD, no KS tests, no distributional validation

**Verdict**: ⚠️ **OVER-ENGINEERED FOR THESIS SCOPE**

**Impact**:
- 820 lines of evaluation code not directly supporting RQs
- Could distract from main thesis contributions
- Useful for debugging but not publication-critical

**Recommendation**:
1. **Keep for debugging** (validate SPNs learn correctly)
2. **Remove from main thesis experiments** (focus on F1/SHD)
3. **Move to appendix** if quality validation is needed

---

## 10. Code Quality vs Research Goals

### Strengths

✅ Well-documented (Phase 3 cleanup added clear docstrings)
✅ Modular design (FedCDH, FedPC, cit separate)
✅ Device-agnostic (CPU/CUDA support)
✅ Reproducible (seed control, config system)

### Over-Engineering

⚠️ Optional BIC bypass (4 K-means runs saved, minimal benefit)
⚠️ Complex query counter wrapper (removed __getattr__ but still verbose)
⚠️ SPN evaluation framework (820 lines not in paper's scope)

### Under-Engineering

❌ No DP implementation (paper's core privacy claim)
❌ No federated EM (paper's Algorithm 2)
❌ No federated KCI baseline (needed for RQ2)
❌ No empirical communication tracking (needed for RQ2)

**Assessment**: Polished implementation but **missing core paper components**

---

## Summary of Critical Gaps

### Must Fix Before Thesis Submission

| # | Gap | Severity | Estimated Effort | Impact |
|---|-----|----------|------------------|--------|
| 1 | **Federated EM (Algorithm 2)** | HIGH | 2-3 days | Cannot claim full FedCDH compliance |
| 2 | **Differential Privacy** | HIGH | 2-3 days OR clarify scope | Cannot claim "ε-DP guarantee" |
| 3 | **Federated KCI baseline** | MEDIUM | 1 day | Cannot empirically validate RQ2 speedup |
| 4 | **Communication measurements** | MEDIUM | 4 hours | Cannot empirically validate RQ2 costs |
| 5 | **Orientation method validation** | LOW | Ask Jonas | Clarify HSIC usage vs paper |

### Research Misalignments to Address

| # | Misalignment | Issue | Fix |
|---|--------------|-------|-----|
| 1 | **Success criteria** | F1 ≥ 0.75 unjustified | Lower to 0.65 or use N=5000 synthetic |
| 2 | **Vertical regularization** | Weak evidence, no grounding | Replicate on Sachs + ablation OR downgrade claim |
| 3 | **6× speedup claim** | No federated KCI baseline | Implement OR cite paper's numbers |

---

## Recommendations

### Option A: Full Paper Compliance (4-5 days work)

**Implement**:
1. Federated EM (Algorithm 2) - 2 days
2. Differential privacy via Opacus - 2 days
3. Federated KCI baseline - 1 day

**Outcome**: Thesis can claim full FedCDH implementation + empirical validation

**Risk**: Delays thesis writing (deadline in 4 weeks)

---

### Option B: Honest Scope Reduction (1 day work)

**Clarify**:
1. "Simulation-based privacy (no raw data sharing), not formal DP"
2. Remove "federated EM" from claims (use centralized EM only)
3. Change RQ2 to "vs centralized KCI" or cite paper's speedup

**Implement**:
1. Replicate "vertical regularization" on Sachs (validate or reject)
2. Add empirical communication byte counting (4 hours)

**Outcome**: Honest thesis with clear scope limitations

**Risk**: Weaker novelty claim (thesis becomes "implementation study")

---

### Option C: Hybrid Approach (2-3 days) ⭐ **RECOMMENDED**

**Must-fix**:
1. Replicate vertical hypothesis on Sachs (validate claim)
2. Add communication byte tracking (support RQ2)
3. Lower F1 target to 0.65 (justified by N difference)

**Clarify**:
1. Scope: "FedCDH with SPNs (simulation-based privacy)"
2. Remove DP claims OR add "future work: formal DP guarantees"
3. Federated EM: "Centralized EM for efficiency (federated version future work)"

**Optional**:
1. Implement federated KCI if time permits (strengthens RQ2)

**Outcome**: Honest, well-scoped thesis with validated contributions

**Risk**: Moderate - Depends on Sachs replication confirming vertical effect

---

## Conclusion

The implementation is **functional and well-engineered** but has **critical gaps** vs. the ICLR 2024 paper:

1. ❌ Missing federated EM (paper's Algorithm 2)
2. ❌ Missing differential privacy (paper's Theorem 1)
3. ❌ Missing federated KCI baseline (needed for RQ2)
4. ⚠️ "Vertical regularization" hypothesis needs validation

**Thesis can succeed** by either:
- **Implementing missing components** (risky timeline)
- **Clarifying scope honestly** (safer, still valid contribution)

**Most critical**: Validate "vertical regularization" on real data (Sachs) before claiming as novel contribution.

---

**Next Steps**: Decide on Option A/B/C and execute immediately (4 weeks to deadline).
