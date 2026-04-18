# LearnSPN Investigation for Federated Causal Discovery

**Date**: April 18, 2026
**Goal**: Evaluate if LearnSPN is better than RAT-SPN for federated causal discovery
**Status**: Investigation in progress

---

## Research Context

**Objective**: Discover causal graphs from federated heterogeneous data using SPNs for CI testing

**Current Implementation**: RAT-SPN (via simple-einet)
- Random structure (no learning)
- Fixed architecture: num_sums=20, num_leaves=20, depth=2-3
- Works but needs large capacity (50-100× oversized)

**Research Questions**:
1. Does LearnSPN provide better density estimation for CI tests?
2. Is learned structure better than random for causal discovery?
3. What's the speed/accuracy trade-off?
4. Can LearnSPN work in federated setting (H/V/Hybrid)?

---

## LearnSPN Algorithm (Gens & Domingos 2013)

### Core Idea
Learn SPN structure AND parameters from data using greedy top-down approach.

### Algorithm Pseudocode
```python
def LearnSPN(data):
    """
    Greedy top-down structure learning.

    1. If single variable → return Leaf
    2. Test independence:
       - Independent → Product node (split features)
       - Dependent → Sum node (cluster instances)
    3. Recurse on splits/clusters
    """
    if is_univariate(data):
        return fit_leaf_distribution(data)

    # Test feature independence
    if are_features_independent(data):
        # Product node: P(X) = P(X1) * P(X2) * ...
        splits = partition_features(data)
        children = [LearnSPN(split) for split in splits]
        return ProductNode(children)
    else:
        # Sum node: P(X) = Σ w_k * P_k(X)
        clusters = cluster_instances(data)
        children = [LearnSPN(cluster) for cluster in clusters]
        weights = compute_cluster_weights(clusters)
        return SumNode(children, weights)

def are_features_independent(data):
    """Test pairwise independence using G-test or χ²."""
    # For continuous: discretize or use correlation
    # Return True if most pairs are independent
    pass

def cluster_instances(data):
    """Cluster data into K groups (e.g., K-means)."""
    # Determines how many mixture components
    pass
```

### Key Parameters
- **Independence threshold** (α): For feature independence test
- **Min instances**: Minimum samples to split further
- **Max depth**: Stop recursion depth
- **Discretization bins**: For continuous data independence tests

---

## Theoretical Fit for Federated Causal Discovery

### ✅ Advantages for Causal Discovery

1. **Structure Matches Conditional Independence**
   - Product nodes encode independence
   - Sum nodes encode mixtures (heterogeneity)
   - Should improve CI test accuracy

2. **Adaptive to Data**
   - Learns which features are independent
   - Creates structure matching causal relationships
   - Less capacity waste than random structure

3. **Interpretable**
   - Product nodes → features are conditionally independent
   - Sum nodes → multiple regimes/contexts
   - Matches federated clustering + heterogeneity

### ⚠️ Challenges for Federated Learning

1. **Structure Learning Requires Full Data Access**
   - Independence tests need joint distribution
   - Vertical FL: Clients have different features (problem!)
   - Solution: Learn structure on server after clustering?

2. **Slower Training**
   - Structure search: 5-10× slower than fixed architecture
   - Recursion depth can be large
   - May not scale to high dimensions (d>20)

3. **Continuous Data Handling**
   - Original LearnSPN: Designed for discrete data
   - Continuous: Need discretization or correlation tests
   - Gaussians: Can use mutual information instead

---

## Available Implementations

### Option 1: SPFlow (Most Complete)
```bash
pip install spflow
```
- ✅ Implements LearnSPN algorithm
- ✅ Supports continuous data (Gaussians)
- ✅ Well-tested, active development
- ⚠️ Different API than simple-einet
- ⚠️ Integration effort: 2-3 hours

### Option 2: simple-einet Extensions
- ❌ simple-einet only has RAT-SPN
- ✅ Could implement LearnSPN on top
- ⚠️ Implementation from scratch: 6-8 hours

### Option 3: PyTorch Implementation
```bash
pip install torch-spn  # If available
```
- May have LearnSPN
- Check compatibility with FedCDH

---

## Experimental Design

### Comparison Metrics

| Metric | RAT-SPN (Baseline) | LearnSPN (Test) | Better If |
|--------|-------------------|-----------------|-----------|
| **Train LL** | -9 to -11 | ? | Higher (less negative) |
| **Test LL** | ? | ? | Higher |
| **CI Test Accuracy** | 60-70% | ? | Higher |
| **Skeleton F1** | 0.6-0.7 | ? | Higher |
| **SHD** | 15-20 | ? | Lower |
| **Training Time** | 1-2 min | ? | Ideally <10 min |
| **Parameters** | 10,000+ | ? | Fewer |

### Test Configurations

**Quick Test (d=5, K=2, n=200)**:
- Fast iteration
- Validate integration works
- Check basic metrics

**Small Test (d=8, K=3, n=600)**:
- Production-like
- Compare with existing benchmarks
- Check federated scenarios (H/V/Hybrid)

**Medium Test (d=10, K=3, n=1200)**:
- Scalability test
- Check if LearnSPN overfits
- Compare training times

### Scenarios to Test

1. **Horizontal**: Both should work (full data per client)
2. **Vertical**: LearnSPN may struggle (feature partitioning)
3. **Hybrid**: Most realistic test case

---

## Implementation Plan

### Phase 1: Research & Setup (1 hour)
- [x] Document research context
- [ ] Install SPFlow
- [ ] Test basic LearnSPN usage
- [ ] Verify continuous Gaussian support

### Phase 2: Integration (2-3 hours)
- [ ] Create LearnSPNWrapper (similar to LocalSPNWrapper)
- [ ] Integrate with FedCDH.fit()
- [ ] Add `--spn-type` flag: "rat" or "learn"
- [ ] Test horizontal mode first (simplest)

### Phase 3: Experiments (2-4 hours)
- [ ] Run quick test (d=5) for both RAT-SPN and LearnSPN
- [ ] Run small test (d=8) for comparison
- [ ] Run vertical/hybrid if time permits
- [ ] Collect metrics: LL, CI accuracy, F1, SHD, time

### Phase 4: Analysis (1 hour)
- [ ] Create comparison table
- [ ] Identify trade-offs
- [ ] Recommend which to use when
- [ ] Document findings

**Total Estimated Time**: 6-9 hours

---

## Success Criteria

**LearnSPN is better if**:
1. ✅ Train LL improves by >10% (e.g., -9 → -8)
2. ✅ CI test accuracy improves by >5% (e.g., 65% → 70%)
3. ✅ Skeleton F1 improves by >0.05 (e.g., 0.65 → 0.70)
4. ✅ Training time <5× slower (e.g., 2 min → <10 min)

**LearnSPN is acceptable if**:
1. ⚠️ Modest LL improvement (+5%)
2. ⚠️ Similar CI accuracy
3. ⚠️ Better interpretability (structure matches data)
4. ⚠️ Training time <10× slower

**LearnSPN is not worth it if**:
1. ❌ No improvement in any metric
2. ❌ Much slower (>10× training time)
3. ❌ Doesn't work with vertical/hybrid FL

---

## Next Steps

1. Install SPFlow and test basic usage
2. Create LearnSPNWrapper class
3. Run quick comparison experiment
4. Analyze results and decide

**Decision Point**: After Phase 3, decide if LearnSPN should replace RAT-SPN in production.

---

## References

- Gens & Domingos (2013): "Learning the Structure of Sum-Product Networks"
- SPFlow: https://github.com/SPFlow/SPFlow
- Seng et al. (2025): "Scaling Probabilistic Circuits via Data Partitioning"
