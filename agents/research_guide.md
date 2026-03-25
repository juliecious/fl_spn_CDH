# Research Guide

## High-Level Goals

### Thesis Objective
**"Federated Causal Discovery with Probabilistic Circuits"**

Replace traditional summary-statistic conditional independence tests in FedCDH with probabilistic circuits (FPC/RAT-SPN) as the summary mechanism, enabling:
1. Privacy-preserving causal discovery from distributed heterogeneous data
2. Improved scalability compared to kernel-based methods (KCI)
3. Support for horizontal, vertical, and hybrid data partitioning scenarios

**Critical Update (March 24, 2026)**: Bug fix validated (num_permutations=0→50), achieving F1_skeleton=1.000 on smoke tests. Implementation ready for full thesis experiments.

### Research Questions
1. **RQ1**: Can FedPC-based CI testing match or exceed the accuracy of kernel-based oracle methods (KCI)?
2. **RQ2**: Does FedPC provide better scalability (time/communication complexity) in high-dimensional settings?
3. **RQ3**: How does mechanism invariance orientation improve causal direction disambiguation in federated settings?
4. **RQ4**: What is the privacy-accuracy tradeoff when varying SPN model complexity?
5. **RQ5** (NEW): Does vertical partitioning improve orientation quality compared to horizontal/hybrid? If so, why?

## Design Decisions

### ✅ Confirmed Decisions

1. **Probabilistic Circuit Backend**: Use Sum-Product Networks (SPNs) via simple-einet library
   - Justification: Tractable exact inference, differentiable, supports efficient marginalization
   - Alternative considered: Flow-based models (rejected due to sampling inefficiency)

2. **Federated Learning Strategy**:
   - **Horizontal**: Mixture-of-Experts (GlobalFedSPN with sum node)
   - **Vertical**: Product-of-Experts (FederatedProduct with product node)
   - **Hybrid**: Combination of both
   - Justification: Aligns with probabilistic semantics of SPN sum/product nodes

3. **Clustering**: Simulated Federated K-Means for mechanism discovery
   - Privacy-preserving via secure aggregation simulation
   - BIC-based automatic cluster selection (h ∈ {2, 3, 4, 5, 6})

4. **CI Test Implementation**: SPN_CIT with G-test + permutation testing
   - G-test statistic: 2N × I(X;Y|Z) approximated from SPN log-likelihoods
   - Conditional permutation via local binning (bucket size ~20)
   - Fallback to chi-squared(df=1) approximation for speed

5. **Orientation Strategy**: Novel mechanism invariance approach
   - **Pure (mi_only)**: Variance of P(Y|X,U=k) across clients k
   - **Hybrid (mi_hybrid)**: Weighted combination with HSIC score
   - Principle: Correct causal direction exhibits more invariant conditional distributions

6. **Regularization**: L1 sparsity on SPN sum weights
   - Prevents overfitting in low-data regimes
   - Improves interpretability by pruning redundant paths

### 🤔 Open Design Questions

1. **Should we implement differential privacy guarantees?**
   - Current: Simulation-based privacy (no raw data sharing)
   - Alternative: Add DP noise to gradients/model parameters
   - Tradeoff: Accuracy loss vs. formal privacy bounds

2. **How to handle missing data in federated setting?**
   - Current: Assume complete data
   - Extension needed: Marginalization-friendly imputation strategies

3. **Optimal SPN architecture search?**
   - Current: Fixed hyperparameters (num_sums=20, num_leaves=20, num_repetitions=10)
   - Future: Federated NAS or meta-learning for structure optimization

4. **Extend to time-series causal discovery?**
   - Current: i.i.d. cross-sectional data
   - Extension: Temporal SPNs or recurrent probabilistic circuits

## Key Implementation Insights

### Critical Integration Points
1. **FedCDH.fit() → GlobalFedSPN**: Trains federated density oracle (causallearn/search/FCMBased/FedCDH/FedCDH.py:378)
2. **CDNOD → SPN_CIT**: Passes SPN to skeleton discovery algorithm (causallearn/search/ConstraintBased/CDNOD.py:240)
3. **Mechanism Invariance → FedCDH_SPN_Wrapper**: Queries conditional likelihoods for orientation (causallearn/utils/mechanism_invariance.py:60-82)

### Data Flow Summary
```
Raw Data (K clients)
  ↓ [Federated K-Means]
Mechanism Clusters
  ↓ [Local SPN Training]
Per-Client SPNs
  ↓ [Global Assembly]
GlobalFedSPN / FederatedProduct
  ↓ [EM Weight Refinement]
Calibrated Global Model
  ↓ [SPN_CIT Oracle]
Conditional Independence Tests
  ↓ [CDNOD Skeleton Discovery]
Undirected Graph
  ↓ [Mechanism Invariance Orientation]
Causal DAG
```

## Evaluation Strategy

### Minimal Experiment Suite (MVP)

#### Datasets
1. **Sachs (Real-World)**: N=856, D=11, K=3 (interventional partitioning)
   - Ground truth: 17 edges, 11 nodes (protein signaling network)
   - Priority: HIGH (validates real-world applicability)

2. **Asia (Benchmark)**: N=500, D=8, K=3 (synthetic with heterogeneity)
   - Ground truth: Known Bayesian network structure
   - Priority: HIGH (standard benchmark)

3. **Alarm (Scalability)**: N=1000, D=37, K=5
   - Ground truth: Large-scale Bayesian network
   - Priority: MEDIUM (tests scalability limits)

#### Baselines
1. **FisherZ**: Fast correlation-based (naive baseline)
2. **KCI**: Kernel-based oracle (accuracy reference)
3. **FedSPN (Ours)**: Three scenarios (horizontal, vertical, hybrid)

#### Metrics
**Primary**:
- F1 Score (harmonic mean of precision/recall)
- Structural Hamming Distance (SHD) - lower is better
- Time (training + discovery)
- Communication Cost (bytes transferred)

**Secondary**:
- Precision (1 - false positives / total predicted)
- Recall (true positives / total ground truth)
- F1/SHD on skeleton (undirected graph)

#### Experiment Grid (Scalability Study)
- N ∈ {100, 500, 1000}
- d ∈ {5, 10, 20}
- K ∈ {2, 3, 5}
- Monte Carlo seeds: 5-10 per configuration

### Success Criteria

**Comparison to ICLR 2024 Baseline**:
- Paper reports: Sachs F1 = 0.91 (N=5000, K=10)
- Our target: Sachs F1 ≥ 0.80 (N=856, K=3, more realistic setup)

**Research Questions**:
- **RQ1 (Accuracy)**: FedSPN F1 ≥ 0.9 × KCI F1 on Sachs
- **RQ2 (Scalability)**: FedSPN time < 0.5 × KCI time for d ≥ 20
- **RQ3 (Orientation)**: Mechanism invariance improves SHD by ≥ 10% vs. context-based
- **RQ4 (Privacy)**: Communication cost < 0.1 × raw data size

## Key Resources

### Core Implementation
- `causallearn/search/FCMBased/FedCDH/FedCDH.py` - Main pipeline orchestration
- `causallearn/utils/FedPC.py` - SPN wrappers and federated assembly
- `causallearn/utils/cit.py` - SPN_CIT implementation
- `causallearn/utils/mechanism_invariance.py` - Novel orientation strategy

### Experiment Infrastructure
- `tests/benchmarks/run_experiment.py` - Execution engine
- `tests/benchmarks/configs.py` - Configuration registry
- `tests/benchmarks/analyze_results.py` - Result aggregation
- `tests/benchmarks/benchmark_scalability.py` - Grid search runner

### Data & Baselines
- `tests/utils/benchmark_loaders.py` - Standard graph loaders
- `tests/utils/sachs_loader.py` - Real-world Sachs dataset

## Thesis Structure (Planned)

1. **Introduction**: Motivation, research questions (RQ1-RQ5), contributions
   - Novel contribution: Vertical partitioning as regularization for causal discovery

2. **Background**: Causal discovery, federated learning, probabilistic circuits
   - Add: Factorization in probabilistic models (product vs mixture)

3. **Method**: FedCDH algorithm, FedPC integration, mechanism invariance
   - Detail: Three partitioning scenarios (H/V/Hy) and their SPN architectures

4. **Experiments**: Datasets, baselines, metrics, results
   - Main results: Sachs benchmark (5 methods × 3 scenarios)
   - Ablation: Orientation quality across scenarios (addresses RQ5)
   - Analysis: Communication-accuracy tradeoff

5. **Discussion**: Insights, limitations, privacy analysis
   - **Key Insight**: Why vertical outperforms (3-point hypothesis)
   - Product-of-experts vs mixture-of-experts for causal structure
   - Limitations: When vertical may not help (dense graphs, collinear features)

6. **Conclusion**: Summary, future work
   - Future: Theoretical analysis of vertical regularization effect
   - Future: Adaptive scenario selection based on graph structure

## Novel Research Contribution Identified (March 6, 2026)

### 🎁 Vertical Partitioning as Regularization for Causal Discovery

**Empirical Finding**: During validation testing, the **vertical scenario achieved directed F1 = 0.200**, while horizontal and hybrid scenarios achieved F1 = 0.000 (with identical skeleton F1 = 0.500).

**Hypothesis**: Feature partitioning acts as a regularizer for mechanism invariance-based orientation through:

1. **Reduced Spurious Correlations**
   - Vertical partitioning separates features across clients
   - Each client models P(X_subset) independently
   - Global model combines via product: P(X) = ∏_k P(X_k)
   - Spurious feature correlations are broken by factorization
   - Result: Cleaner mechanism variance signal for orientation

2. **Enhanced Mechanism Variance Detection**
   - In horizontal: All features observed together, confounding can mask true mechanisms
   - In vertical: Features separated, mechanism shifts more detectable
   - Example: If X→Y with heterogeneity, separating X and Y reveals P(Y|X,U) variance better
   - Mechanism invariance relies on variance of P(target|parents,U) across domains
   - Feature independence helps isolate this variance

3. **Causal Structure Preservation via Factorization**
   - Product-of-experts model: P(X) = ∏_k P_k(X_k)
   - Maintains conditional independence structure better than mixture models
   - Horizontal uses mixture-of-experts: P(X) = Σ_k w_k P_k(X)
   - Mixture can blur causal boundaries, product preserves them
   - Aligns with causal factorization: P(X) = ∏_i P(X_i|Pa(X_i))

**Implications for Thesis**:
- Add dedicated section in Discussion chapter
- Compare orientation quality across scenarios as ablation study
- Investigate with additional datasets (Asia, Alarm)
- Potential contribution: "Vertical Federated Learning as Implicit Regularization for Causal Discovery"

**Follow-up Experiments**:
- **EXP-4**: Ablation study on orientation method × scenario
- **EXP-5**: Vary number of clients K in vertical to test factorization hypothesis
- **EXP-6**: Compare vertical FedSPN vs centralized on orientation quality

**Theoretical Questions**:
- **THEORY-4**: Can we prove that product-of-experts preserves more causal structure than mixture-of-experts?
- **THEORY-5**: Under what conditions does feature partitioning improve mechanism invariance detection?

---

## Critical Questions for Meeting (March 24, 2026)

### For Jonas (FedCDH ICLR 2024 Author)
1. **SPN Performance**: What F1 scores did FedCDH achieve with KCI on Sachs (N=5000)?
2. **Vertical Effect**: Did you observe vertical → better orientation in your experiments?
3. **Novelty Check**: Is "vertical as regularization" documented in your supplementary materials?
4. **Realistic Targets**: What F1 should we expect for N=856 Sachs (vs your N=5000)?
5. **Implementation Alignment**: Does our MI orientation match your paper's method?

### For Prof. Devendra Singh Dhami (Causality Expert)
1. **Statistical Validity**: Is 50 permutations standard for α=0.05 CI testing?
2. **Theoretical Grounding**: Is "vertical as regularization" causally plausible?
3. **Evaluation Metrics**: Are our metrics (F1, SHD, comm cost) sufficient for thesis?
4. **Privacy Level**: What privacy does model parameter sharing provide (vs differential privacy)?
5. **Thesis Scope**: Is 6 experiments + Sachs validation enough for Master's thesis?

## Open Questions & Future Work

### Theoretical
- **THEORY-1**: Under what conditions does variance-based mechanism invariance provably recover causal direction?
- **THEORY-2**: Can we derive finite-sample guarantees for SPN_CIT permutation test?
- **THEORY-3**: What is the statistical power of G-test approximation vs. exact chi-squared?
- **THEORY-4**: Can we prove that product-of-experts preserves more causal structure than mixture-of-experts? (NEW)
- **THEORY-5**: Under what conditions does feature partitioning improve mechanism invariance detection? (NEW)

### Practical
- **IMPL-1**: How to handle clients with very small sample sizes (N_k < 50)?
- **IMPL-2**: Can we use adaptive permutation testing (early stopping) to reduce computation?
- **IMPL-3**: Should we implement cross-validation for SPN hyperparameter tuning in federated setting?

### Practical
- **IMPL-1**: How to handle clients with very small sample sizes (N_k < 50)?
- **IMPL-2**: Can we use adaptive permutation testing (early stopping) to reduce computation?
- **IMPL-3**: Should we implement cross-validation for SPN hyperparameter tuning in federated setting?

### Extensions
- **EXT-1**: Extend to continuous optimization (gradient-based causal discovery)
- **EXT-2**: Multi-modal data (images, text, tabular) with heterogeneous SPNs
- **EXT-3**: Dynamic federated causal discovery (clients join/leave over time)
- **EXT-4**: Integration with differential privacy frameworks (e.g., Opacus)

### New Experiments (Following Vertical Discovery)
- **EXP-4**: Ablation study - orientation method (mi_only, mi_hybrid, context) × scenario (H, V, Hy)
- **EXP-5**: Vertical scalability - vary K ∈ {2, 3, 5, 10} to test factorization hypothesis
- **EXP-6**: Centralized comparison - vertical FedSPN vs centralized SPN on orientation quality
- **EXP-7**: Communication-accuracy frontier - plot directed F1 vs comm cost for all scenarios

## Next Research Milestones

**Thesis Deadline**: End of April 2026 (~8 weeks remaining)

### Week 1-2 (March 6-20): Core Results
- [x] Complete horizontal scenario bug fix and validation
- [ ] Run full benchmark suite on Sachs (5 methods × 10 seeds on Colab GPU)
- [ ] Generate first draft of results section with plots
- [ ] Prepare Google Colab notebook for GPU experiments

### Week 3-4 (March 21 - April 3): Scalability & Extensions
- [ ] Implement Asia benchmark experiments (8 nodes)
- [ ] Run scalability parameter grid (N/d/K variations)
- [ ] Draft method section with algorithm pseudocode
- [ ] Complete Alarm benchmark (37 nodes) if time permits

### Week 5-6 (April 4-17): Analysis & Writing
- [ ] Privacy-accuracy tradeoff experiments (vary SPN complexity)
- [ ] Ablation studies (orientation strategies, clustering methods)
- [ ] Write full thesis draft (intro, background, method, experiments)
- [ ] Generate all publication-quality figures

### Week 7-8 (April 18-30): Polish & Submission
- [ ] Revise thesis based on advisor feedback
- [ ] Finalize discussion and related work sections
- [ ] Proofread and format for submission
- [ ] **Final submission by April 30, 2026**

---
*Created on 2026-03-06 | Updated by Claude Code*
