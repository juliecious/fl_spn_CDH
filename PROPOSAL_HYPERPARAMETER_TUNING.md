# Proposal: Enhanced Hyperparameter Tuning and Dataset-Adaptive Threshold Calibration

**Date:** 2026-05-18
**Status:** Proposal (Not Implemented)
**Target:** FedCDH v3 Performance Optimization

---

## Executive Summary

This proposal outlines improvements to FedCDH's hyperparameter tuning and threshold calibration systems to address performance variability across datasets and improve causal discovery accuracy. Current results show:

- **GES/FCI baselines**: Skeleton F1 = 1.0 on law_school
- **FedSPN methods**: Skeleton F1 = 0.0-0.222 on law_school (significant gap)

The root causes are:
1. **Fixed thresholds** don't adapt to dataset characteristics
2. **Hyperparameters** don't account for data complexity/density
3. **Structure voting** uses static 0.4 threshold regardless of client agreement patterns

---

## Part 1: Current State Analysis

### 1.1 Current Hyperparameter System

**File:** `causallearn/utils/FedPC.py:2375-2530`

The `compute_adaptive_hyperparameters()` function implements a **5-criterion system**:

1. **Mode-Specific Base Capacity**:
   - Horizontal: 4×d sums, 2×d leaves
   - Vertical: 8×d sums (if d>3), 4×d leaves
   - Hybrid: 6×d sums, 3×d leaves

2. **Sample-to-Feature Ratio Scaling**:
   - ratio < 50: capacity_scale = 0.5
   - ratio 50-100: capacity_scale = 0.75
   - ratio 100-200: capacity_scale = 1.0
   - ratio > 200: capacity_scale = min(1.5, 1.0 + (ratio-200)/400)

3. **Data Type Differentiation**:
   - Nonlinear: +1 depth, 1.3× epochs
   - Linear: base depth, 1.0× epochs

4. **Quality-Aware Epoch Scheduling**:
   - Base: (d/5)^1.5 × 100 epochs
   - Mode multipliers: H=1.0+d/30, V=0.8, Hy=0.9
   - Clamped to [100, 500]

5. **Mode-Aware Regularization**:
   - Horizontal: weight_decay=1e-4, dropout=0.1 (if ratio<100)
   - Vertical: weight_decay=1e-3, dropout=0.0
   - Hybrid: weight_decay=5e-5, dropout=0.05 (if ratio<100)

**Strengths:**
✓ Considers multiple dimensions (mode, ratio, data type)
✓ Prevents extreme values (clamping)
✓ Mode-specific tuning

**Weaknesses:**
✗ Ignores dataset-specific properties (edge density, graph sparsity)
✗ No calibration based on validation performance
✗ Fixed breakpoints (50, 100, 200) may not be optimal for all datasets
✗ Doesn't consider feature correlation strength
✗ No adaptation based on client heterogeneity

### 1.2 Current Threshold System

**File:** `causallearn/search/FCMBased/FedCDH/FedCDH.py:284`

Default `structure_vote_threshold = 0.5` (50% of clients must agree)

**File:** `tests/benchmarks/test_fedcdh_benchmark_v3.py:991`

Benchmark uses `structure_vote_threshold = 0.4` (40% agreement)

**File:** `causallearn/utils/structure_aggregation.py:67-132`

`aggregate_structures_by_voting()` function:
- Counts votes for each edge across K clients
- Includes edge if `votes >= ceil(threshold × K)`
- Returns consensus graph + edge confidence scores

**Strengths:**
✓ Simple majority voting
✓ Provides confidence scores
✓ Handles variable client counts

**Weaknesses:**
✗ **Fixed threshold** doesn't adapt to:
  - Dataset characteristics (sparse vs dense graphs)
  - Client agreement patterns (high vs low heterogeneity)
  - Number of clients (3 vs 10 clients have different voting dynamics)
✗ No consideration of edge strength (p-values)
✗ Doesn't account for ground truth sparsity when known
✗ May be too conservative (miss true edges) or too liberal (include false positives)

---

## Part 2: Identified Problems

### 2.1 Problem 1: Dataset Characteristic Blindness

**Current Issue:**
The hyperparameter system doesn't consider:

1. **Graph Sparsity**:
   - Law school: 7 edges / 10 possible = 70% sparse
   - Sachs: 17 edges / 55 possible = 69% sparse
   - Dense graphs need different capacity than sparse graphs

2. **Feature Correlation Strength**:
   - Strongly correlated features → need more capacity to model complex dependencies
   - Weakly correlated → simpler models sufficient

3. **Sample Quality**:
   - High noise → need regularization
   - Clean data → can use more capacity

**Evidence from Results:**
```
Law School (d=5, n=21000, edges=7, sparse graph):
- GES: skeleton_f1=1.0, skeleton_shd=0
- FCI: skeleton_f1=1.0, skeleton_shd=0
- FedSPN_h: skeleton_f1=0.0, skeleton_shd=7 (FAILED - CUDA OOM)
- FedSPN_v: skeleton_f1=0.222, skeleton_shd=7 (POOR)
- FedSPN_hy: skeleton_f1=0.0, skeleton_shd=7 (FAILED)
```

The FedSPN methods completely failed despite having sufficient data (21000 samples, 5 features).

### 2.2 Problem 2: Threshold Rigidity

**Current Issue:**
Fixed 0.4 or 0.5 threshold doesn't adapt to:

1. **Client Count (K)**:
   - K=3: threshold=0.4 → need 2/3 clients (67% agreement required)
   - K=10: threshold=0.4 → need 4/10 clients (40% agreement required)
   - Same threshold value has different semantic meaning

2. **Data Heterogeneity**:
   - IID clients: High agreement expected → can use higher threshold (0.6-0.7)
   - Non-IID clients: Low agreement expected → need lower threshold (0.3-0.4)

3. **Graph Sparsity**:
   - Sparse graphs (few edges): Conservative threshold to avoid false positives
   - Dense graphs (many edges): Liberal threshold to capture more dependencies

**Mathematical Analysis:**

For K=3 clients with threshold=0.4:
- Required votes: ceil(0.4 × 3) = ceil(1.2) = 2
- Actual threshold: 2/3 = **66.7%** (not 40%!)

For K=10 clients with threshold=0.4:
- Required votes: ceil(0.4 × 10) = 4
- Actual threshold: 4/10 = **40%** (as intended)

**Impact:** Small client counts amplify threshold effects unpredictably.

### 2.3 Problem 3: No Validation-Based Calibration

**Current Issue:**
No mechanism to:
1. Tune hyperparameters based on held-out validation performance
2. Adjust thresholds based on skeleton recovery quality
3. Learn dataset-specific optimal settings

**Consequence:**
One-size-fits-all parameters perform well on some datasets, poorly on others.

### 2.4 Problem 4: Ignoring Edge Strength

**Current Issue:**
Structure voting treats all edges equally:
- Strong edge (p-value=0.001) = 1 vote
- Weak edge (p-value=0.049) = 1 vote

**Better Approach:**
Weight votes by confidence or p-value strength.

---

## Part 3: Proposed Solutions

### Solution 1: Dataset-Adaptive Hyperparameter Calibration

#### 3.1.1 Add Data Complexity Profiling

**Proposal:** Before training, analyze dataset characteristics:

```python
def profile_dataset_complexity(X: np.ndarray) -> Dict[str, float]:
    """
    Profile dataset to guide hyperparameter selection.

    Returns:
        - correlation_strength: Mean absolute correlation [0,1]
        - feature_variance_ratio: CV of feature variances (heterogeneity)
        - effective_dimension: Intrinsic dimensionality estimate
        - noise_estimate: Estimated noise level
        - sparsity_hint: Expected graph sparsity based on correlations
    """
```

**Metrics to Compute:**

1. **Correlation Strength** (mean |correlation|):
   - High (>0.5): Complex dependencies → need more capacity
   - Low (<0.2): Weak dependencies → simpler models

2. **Effective Dimensionality** (via PCA):
   - Low (<<d): Redundant features → reduce capacity
   - High (≈d): Independent features → increase capacity

3. **Noise Estimate** (via residual analysis):
   - High noise → increase regularization
   - Low noise → reduce regularization

4. **Expected Sparsity** (via partial correlation matrix):
   - Sparse (<20% edges): Conservative thresholds
   - Dense (>50% edges): Liberal thresholds

**Integration Point:**
Call before `compute_adaptive_hyperparameters()` and pass complexity profile as input.

#### 3.1.2 Enhanced Hyperparameter Function

**Proposal:** Extend `compute_adaptive_hyperparameters()` signature:

```python
def compute_adaptive_hyperparameters(
    mode: str,
    num_features: int,
    num_samples: int,
    data_type: str,
    complexity_profile: Optional[Dict[str, float]] = None,  # NEW
    base_num_sums: int = 20,
    base_num_leaves: int = 20,
    base_epochs: int = 100,
    base_depth: int = None,
) -> Dict[str, Any]:
```

**New Logic:**

```python
# Criterion 6: Data Complexity Adaptation
if complexity_profile:
    corr_strength = complexity_profile['correlation_strength']

    # Strong correlations need more capacity
    if corr_strength > 0.5:
        capacity_boost = 1.3
    elif corr_strength > 0.3:
        capacity_boost = 1.0
    else:
        capacity_boost = 0.8

    final_num_sums *= capacity_boost
    final_num_leaves *= capacity_boost

    # High noise needs more regularization
    noise = complexity_profile['noise_estimate']
    if noise > 0.3:
        weight_decay *= 2.0
        dropout += 0.05
```

**Expected Impact:**
- Law school (5 features, sparse): Lower capacity, higher regularization
- Sachs (11 features, dense): Higher capacity, balanced regularization
- Synthetic (varies): Adaptive to generated complexity

### Solution 2: Adaptive Structure Voting Thresholds

#### 3.2.1 Client-Count Aware Thresholds

**Proposal:** Adjust threshold based on K to maintain consistent semantics:

```python
def get_adaptive_vote_threshold(
    K: int,
    base_threshold: float = 0.5,
    target_agreement: str = "majority"
) -> float:
    """
    Compute threshold that maintains semantic consistency across K.

    Args:
        K: Number of clients
        base_threshold: Desired agreement proportion
        target_agreement: "majority" | "supermajority" | "consensus"

    Returns:
        Adjusted threshold that accounts for ceiling effects
    """
    if target_agreement == "majority":
        # Ensure at least 50% actual votes
        required_votes = max(2, int(np.ceil(K / 2)))
    elif target_agreement == "supermajority":
        # 2/3 majority
        required_votes = max(2, int(np.ceil(2 * K / 3)))
    else:  # consensus
        # All but one
        required_votes = max(K - 1, 2)

    # Return effective threshold
    return required_votes / K
```

**Example:**
- K=3: majority → 2/3 = 0.667 (vs current 0.4→0.667)
- K=10: majority → 5/10 = 0.5 (vs current 0.4→0.4)

**Benefit:** Consistent agreement semantics regardless of K.

#### 3.2.2 Data-Driven Threshold Calibration

**Proposal:** Estimate optimal threshold from data characteristics:

```python
def calibrate_structure_threshold(
    dataset_name: str,
    num_features: int,
    expected_edges: Optional[int] = None,
    client_heterogeneity: float = 0.5,
) -> float:
    """
    Calibrate structure voting threshold based on dataset properties.

    Args:
        dataset_name: Dataset identifier
        num_features: Number of features (d)
        expected_edges: Prior belief about edge count (if available)
        client_heterogeneity: Estimated data heterogeneity [0,1]
            0 = IID (homogeneous), 1 = highly heterogeneous

    Returns:
        Calibrated threshold in [0.3, 0.8]
    """
    # Base threshold
    base = 0.5

    # Adjust for heterogeneity
    # More heterogeneous → lower threshold (clients disagree more)
    heterogeneity_adjustment = -0.2 * client_heterogeneity

    # Adjust for sparsity
    if expected_edges:
        max_edges = num_features * (num_features - 1) / 2
        sparsity = 1.0 - (expected_edges / max_edges)

        # Sparse graphs → higher threshold (be conservative)
        sparsity_adjustment = 0.15 * sparsity
    else:
        sparsity_adjustment = 0.0

    threshold = base + heterogeneity_adjustment + sparsity_adjustment

    # Clamp to reasonable range
    return np.clip(threshold, 0.3, 0.8)
```

**Example Calibrations:**

| Dataset | d | edges | sparsity | heterogeneity | threshold |
|---------|---|-------|----------|---------------|-----------|
| law_school | 5 | 7 | 0.3 | 0.3 (moderate) | 0.5 - 0.06 + 0.045 = **0.485** |
| sachs | 11 | 17 | 0.69 | 0.5 (high) | 0.5 - 0.1 + 0.104 = **0.504** |
| synthetic_er_small | 10 | 13 | 0.71 | 0.2 (low) | 0.5 - 0.04 + 0.107 = **0.567** |

**Benefit:** Dataset-specific thresholds improve precision/recall tradeoff.

#### 3.2.3 Confidence-Weighted Voting

**Proposal:** Weight votes by edge confidence (p-value strength):

```python
def aggregate_structures_weighted_voting(
    local_graphs: List[nx.Graph],
    threshold: float = 0.5,
    use_pvalue_weights: bool = True,
) -> Tuple[nx.Graph, Dict]:
    """
    Aggregate with confidence-weighted voting.

    Instead of binary votes, each client contributes:
        vote_weight = (1 - p_value) if p_value available, else 1.0

    Edge included if: sum(vote_weights) / K >= threshold
    """
    K = len(local_graphs)
    edge_weighted_votes = defaultdict(float)

    for graph in local_graphs:
        for i, j in graph.edges():
            edge = (min(i, j), max(i, j))

            if use_pvalue_weights and 'p_value' in graph[i][j]:
                # Stronger evidence (lower p-value) = higher weight
                p_val = graph[i][j]['p_value']
                weight = 1.0 - p_val  # p=0.001 → weight=0.999
            else:
                weight = 1.0

            edge_weighted_votes[edge] += weight

    # Normalize by K to get average confidence
    for edge in edge_weighted_votes:
        edge_weighted_votes[edge] /= K

    # Include edge if average confidence >= threshold
    consensus_edges = {
        edge for edge, conf in edge_weighted_votes.items()
        if conf >= threshold
    }

    return consensus_edges, edge_weighted_votes
```

**Benefit:** Strong evidence (p=0.001) contributes more than weak evidence (p=0.049).

### Solution 3: Multi-Stage Adaptive Calibration

#### 3.3.1 Two-Phase Training

**Proposal:** Split training into exploration + exploitation:

**Phase 1: Exploration (20% epochs)**
- Use broader capacity (1.5× base)
- Lower regularization
- Collect diagnostics (LL, gradient norms, loss curves)

**Phase 2: Exploitation (80% epochs)**
- Adjust capacity based on Phase 1 diagnostics:
  - If overfitting (train LL >> val LL): Reduce capacity 30%, increase regularization
  - If underfitting (both low): Increase capacity 20%, reduce regularization
  - If balanced: Continue with Phase 1 settings

**Implementation Sketch:**
```python
# Phase 1: Exploration
hyperparams_explore = compute_adaptive_hyperparameters(...)
hyperparams_explore['num_sums'] = int(hyperparams_explore['num_sums'] * 1.5)
hyperparams_explore['epochs'] = int(hyperparams_explore['epochs'] * 0.2)

spn.train_local(data, **hyperparams_explore)
diagnostics = spn.get_training_diagnostics()

# Phase 2: Exploitation
if diagnostics['overfitting_detected']:
    hyperparams_exploit = adjust_for_overfitting(hyperparams_explore)
else:
    hyperparams_exploit = hyperparams_explore

hyperparams_exploit['epochs'] = int(total_epochs * 0.8)
spn.train_local(data, **hyperparams_exploit)
```

**Benefit:** Adapts to actual training dynamics, not just static data properties.

#### 3.3.2 Threshold Grid Search (Optional)

**Proposal:** For benchmark evaluation, test multiple thresholds:

```python
def evaluate_with_threshold_sweep(
    local_graphs: List[nx.Graph],
    true_graph: np.ndarray,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate structure voting across threshold range.

    Returns:
        {threshold: {'f1': ..., 'precision': ..., 'recall': ...}}
    """
    results = {}

    for thresh in thresholds:
        consensus, _ = aggregate_structures_by_voting(local_graphs, thresh)
        metrics = compute_skeleton_metrics(consensus, true_graph)
        results[thresh] = metrics

    # Find optimal threshold
    best_thresh = max(results.items(), key=lambda x: x[1]['f1'])[0]

    return results, best_thresh
```

**Use Case:**
- Report optimal threshold for each dataset
- Understand sensitivity to threshold choice
- Guide default threshold selection

**Benefit:** Empirical evidence for threshold tuning.

---

## Part 4: Implementation Strategy

### Phase 1: Foundation (Week 1)

**Goals:**
1. Implement `profile_dataset_complexity()` utility
2. Add complexity profiling to benchmark pipeline
3. Collect baseline complexity statistics for all datasets

**Files to Modify:**
- `causallearn/utils/data_profiling.py` (NEW)
- `tests/benchmarks/test_fedcdh_benchmark_v3.py` (add profiling step)

**Deliverables:**
- Complexity profiles for all 12 datasets (real + synthetic)
- Analysis report: correlation strength vs F1 performance

### Phase 2: Hyperparameter Enhancement (Week 2)

**Goals:**
1. Extend `compute_adaptive_hyperparameters()` with complexity input
2. Add Criterion 6 (Data Complexity Adaptation)
3. Validate on 3 datasets (law_school, sachs, synthetic_er_small)

**Files to Modify:**
- `causallearn/utils/FedPC.py` (extend function)
- `causallearn/search/FCMBased/FedCDH/FedCDH.py` (pass complexity profile)

**Deliverables:**
- Improved F1 scores on law_school (target: 0.0 → 0.5+)
- Hyperparameter sensitivity analysis

### Phase 3: Adaptive Thresholds (Week 3)

**Goals:**
1. Implement `calibrate_structure_threshold()`
2. Add `aggregate_structures_weighted_voting()`
3. A/B test: fixed vs adaptive thresholds

**Files to Modify:**
- `causallearn/utils/structure_aggregation.py` (new functions)
- `causallearn/search/FCMBased/FedCDH/FedCDH.py` (use adaptive threshold)
- `tests/benchmarks/test_fedcdh_benchmark_v3.py` (add threshold sweep)

**Deliverables:**
- Threshold sensitivity curves for each dataset
- Recommendation table: dataset → optimal threshold

### Phase 4: Validation & Benchmarking (Week 4)

**Goals:**
1. Run full benchmark suite with all improvements
2. Compare: baseline vs enhanced system
3. Statistical significance testing

**Experiments:**
- All 12 datasets × 6 methods × 3 seeds = 216 runs
- Compare: fixed hyperparams vs adaptive
- Analyze: which datasets benefit most

**Success Metrics:**
- Average F1 improvement: +15% or more
- Reduced variance across datasets: -20% std dev
- Law school: F1 > 0.5 (currently 0.0-0.222)

---

## Part 5: Expected Impact

### 5.1 Performance Gains

**Conservative Estimates:**

| Dataset | Current F1 | Expected F1 | Improvement |
|---------|-----------|-------------|-------------|
| law_school (FedSPN_h) | 0.00 | 0.45-0.60 | +45-60 pts |
| law_school (FedSPN_v) | 0.22 | 0.50-0.65 | +28-43 pts |
| sachs (FedSPN_h) | 0.13* | 0.30-0.45 | +17-32 pts |
| synthetic_er_small | 0.25* | 0.50-0.70 | +25-45 pts |

*estimated from similar datasets

**Optimistic Estimates:**
- Law school could reach F1=0.7-0.8 (near GES/FCI baseline)
- Average improvement: 30-40 percentage points

### 5.2 Robustness Improvements

**Reduced Failure Modes:**
1. CUDA OOM → CPU fallback (already implemented)
2. Poor hyperparams → adaptive tuning (proposed)
3. Wrong threshold → dataset calibration (proposed)

**Expected:**
- 0% failure rate (currently ~33% on law_school)
- Consistent performance across dataset types

### 5.3 Generalization

**Better Performance on:**
- **New datasets**: Profiling adapts automatically
- **Different K**: Client-aware thresholds
- **Variable sparsity**: Sparsity-adjusted thresholds
- **Different noise levels**: Noise-adaptive regularization

---

## Part 6: Risks & Mitigations

### Risk 1: Overfitting to Benchmark Datasets

**Risk:** Hyperparameters optimized for 12 datasets may not generalize.

**Mitigation:**
- Use principled heuristics (complexity profiling)
- Validate on held-out datasets not in benchmark
- Keep defaults conservative

### Risk 2: Computational Overhead

**Risk:** Profiling + calibration adds latency.

**Mitigation:**
- Profiling is O(d²) for correlation matrix (cheap for d<100)
- Calibration is O(1) formula evaluation
- Expected overhead: <5% of total runtime

### Risk 3: Hyperparameter Sensitivity

**Risk:** More complex tuning → more hyperparameters to set.

**Mitigation:**
- Keep defaults that work well
- Document sensitivity ranges
- Provide simple presets (conservative/balanced/aggressive)

### Risk 4: Debugging Complexity

**Risk:** Adaptive systems harder to debug when they fail.

**Mitigation:**
- Log all calibration decisions
- Save complexity profiles with results
- Provide diagnostic tools

---

## Part 7: Alternative Approaches Considered

### Alternative 1: Meta-Learning

**Idea:** Learn hyperparameter mappings from past experiments.

**Pros:** Could discover non-obvious patterns

**Cons:**
- Requires large dataset history
- Complex implementation
- Less interpretable

**Decision:** Defer to future work (Phase 5)

### Alternative 2: Bayesian Optimization

**Idea:** Use BO to tune hyperparameters per dataset.

**Pros:** Theoretically optimal tuning

**Cons:**
- Expensive (100+ SPN training runs per dataset)
- Impractical for federated setting (privacy concerns)
- Overfitting risk

**Decision:** Not feasible for production use

### Alternative 3: Ensemble Thresholds

**Idea:** Include edges at multiple thresholds, weight by confidence.

**Pros:** Doesn't require picking "the right" threshold

**Cons:**
- More complex aggregation
- Harder to interpret
- May include many weak edges

**Decision:** Could combine with weighted voting (Solution 2.3)

---

## Part 8: Success Criteria

### Minimum Viable Success (Must Have)

✓ Law school FedSPN_h: F1 > 0.4 (vs current 0.0)
✓ Law school FedSPN_v: F1 > 0.4 (vs current 0.22)
✓ No CUDA OOM failures (already fixed)
✓ Consistent improvement across ≥8/12 datasets

### Target Success (Should Have)

✓ Average F1 improvement: +20 percentage points
✓ Reduced variance: std(F1) < 0.15 across datasets
✓ Law school: F1 > 0.6 (approaching GES/FCI baseline)
✓ Sachs: F1 > 0.4

### Stretch Success (Could Have)

✓ Law school: F1 > 0.8 (matching GES/FCI)
✓ Average F1 > 0.6 across all datasets
✓ Outperform FedCDH baseline on ≥5 datasets
✓ Publication-quality improvement (statistical significance)

---

## Part 9: Timeline & Resources

### Timeline

| Phase | Duration | Effort | Priority |
|-------|----------|--------|----------|
| Phase 1: Profiling | 1 week | 20 hours | HIGH |
| Phase 2: Hyperparams | 1 week | 30 hours | HIGH |
| Phase 3: Thresholds | 1 week | 25 hours | MEDIUM |
| Phase 4: Validation | 1 week | 15 hours | HIGH |
| **Total** | **4 weeks** | **90 hours** | - |

### Resources Needed

**Computational:**
- GPU access for benchmark runs (8-16 hours per full sweep)
- CPU cluster for parallel experiments (optional)

**Human:**
- 1 researcher/engineer (90 hours)
- 1 advisor for review (5 hours)

**Dependencies:**
- Current codebase (FedCDH v3)
- Benchmark suite (12 datasets)
- CUDA OOM fix (already implemented)

---

## Part 10: Conclusion

### Summary

This proposal addresses critical performance gaps in FedCDH by:

1. **Adding dataset-aware hyperparameter tuning** (complexity profiling)
2. **Implementing adaptive threshold calibration** (sparsity/heterogeneity aware)
3. **Enabling confidence-weighted voting** (p-value strength)
4. **Introducing two-phase training** (explore → exploit)

### Expected Outcomes

- **30-40 point F1 improvement** on law_school dataset
- **20 point average improvement** across all datasets
- **Zero failure rate** (vs current 33% on law_school)
- **Publishable results** with statistical significance

### Recommendation

**Proceed with implementation in priority order:**
1. ✅ Phase 1: Profiling (enables all other phases)
2. ✅ Phase 2: Hyperparameters (biggest expected impact)
3. ⚠️ Phase 3: Thresholds (medium impact, good for ablation)
4. ✅ Phase 4: Validation (required for publication)

### Next Steps

1. **Review this proposal** with team (1 hour meeting)
2. **Prioritize phases** based on available resources
3. **Implement Phase 1** (profiling) as proof of concept
4. **Evaluate on law_school** before proceeding to Phase 2

---

## Appendices

### Appendix A: Current Performance Baseline

From `experiments/v3_realworld/law_school/`:

```
Method          | skeleton_f1 | skeleton_shd | runtime
----------------|-------------|--------------|--------
GES             | 1.000       | 0            | 0.66s
FCI             | 1.000       | 0            | 0.08s
FedSPN_h        | 0.000       | 7            | 5.19s (FAILED - CUDA OOM)
FedSPN_v        | 0.222       | 7            | 34.14s
FedSPN_hy       | 0.000       | 7            | 5.71s
```

### Appendix B: Hyperparameter Sensitivity Analysis (Needed)

Planned experiment: Vary hyperparameters ±30% on law_school

| Parameter | -30% | Base | +30% | Impact |
|-----------|------|------|------|--------|
| num_sums | ? | ? | ? | TBD |
| num_leaves | ? | ? | ? | TBD |
| epochs | ? | ? | ? | TBD |
| dropout | ? | ? | ? | TBD |

### Appendix C: Threshold Sensitivity Curve (Needed)

Planned plot: F1 vs threshold for law_school

```
Expected pattern:
F1
^
|     /‾‾‾‾\
|    /      \___
|   /           \
|__/             \___
   0.2 0.4 0.6 0.8
      threshold
```

Optimal threshold estimated: 0.5-0.6 for sparse graphs

### Appendix D: References

1. Seng et al. (2025): FedCDH Algorithm 1 (local clustering)
2. Li et al. (2024): CD-NOD baseline
3. Poon & Domingos (2011): SPNs for density estimation
4. Simple-einet documentation: Hyperparameter guidelines

---

**Document Version:** 1.0
**Last Updated:** 2026-05-18
**Status:** Awaiting Review & Approval
