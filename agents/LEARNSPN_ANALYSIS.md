# LearnSPN Analysis: Integration Challenges and Recommendations

**Date**: April 18, 2026
**Status**: Investigation paused - significant integration challenges identified
**Recommendation**: Focus on RAT-SPN optimizations instead

---

## Summary

After attempting to integrate SPFlow's LearnSPN implementation, I've identified **significant integration challenges** that make it impractical for immediate integration into FedCDH.

**Conclusion**: RAT-SPN with the current 4× architecture increase (num_sums=20, num_leaves=20) is **sufficient for thesis scope**. LearnSPN integration would require 20-40 hours of work with uncertain benefits.

---

## Integration Challenges Identified

### 1. API Incompatibility ❌

**SPFlow vs simple-einet**:
```python
# Current (simple-einet):
from simple_einet.einet import Einet, EinetConfig
config = EinetConfig(num_features=d, num_sums=20, ...)
spn = Einet(config)
spn.fit(X_train, epochs=50)
ll = spn.ll(X_test)

# SPFlow LearnSPN:
from spflow.learn import learn_spn
from spflow.modules.leaves import Normal
leaf = Normal(scope=???)  # Scope parameter unclear
spn = learn_spn(X_train, leaf_modules=leaf, ...)
# Different inference API entirely
```

**Issues**:
- Completely different object models
- Different tensor handling (SPFlow has complex scoping)
- No drop-in replacement possible
- Would require rewriting LocalSPNWrapper entirely

### 2. SPFlow Complexity ⚠️

**Scope Management**:
- SPFlow requires explicit "scope" (which features a node covers)
- simple-einet handles this automatically
- Adding scopes for federated scenarios (V/H/Hybrid) is non-trivial

**Leaf Modules**:
```python
# Need to specify scope for EACH feature:
leaves = [Normal(scope=[i]) for i in range(d)]
# Then learn_spn needs to understand this
```

**Problem**: Vertical FL has clients with different features - how to manage scopes across clients?

### 3. Federated Learning Incompatibility ⚠️

**Structure Learning Needs Full Data**:
```python
# LearnSPN algorithm:
1. Test feature independence → needs ALL features
2. Partition features → needs full data distribution
3. Cluster instances → needs all samples
```

**Federated scenarios**:
- **Vertical**: Clients have DIFFERENT features → Can't test independence locally
- **Horizontal**: Clients have DIFFERENT samples → Could work but needs aggregation
- **Hybrid**: Both problems

**Implication**: LearnSPN designed for centralized learning, not federated

### 4. Time Investment vs Benefit 📊

**Integration effort estimated**:
- API adaptation: 8-10 hours
- Testing & debugging: 4-6 hours
- Federated adaptation: 8-12 hours
- **Total: 20-28 hours minimum**

**Uncertain benefits**:
- May not improve CI test accuracy (structure learned on wrong objective)
- May be slower in federated setting
- May not handle vertical FL well

**Known benefits of current RAT-SPN**:
- ✅ Already integrated
- ✅ Works with H/V/Hybrid
- ✅ Performance acceptable after 4× increase
- ✅ Fast training (1-2 minutes)

---

## Theoretical Analysis: Why LearnSPN May Not Help

### 1. Causal Discovery ≠ Density Estimation

**LearnSPN optimizes**: Log-likelihood P(X)
```
max LL(θ) = Σ log P(X | θ)
```

**Causal discovery needs**: Conditional independence P(X|Y,Z)
```
X ⊥ Y | Z  ⟺  P(X|Y,Z) = P(X|Z)
```

**Problem**: Structure that maximizes LL may NOT align with CI structure

**Example**:
```python
# True causal model:
#   A → B → C
#   P(A,B,C) = P(A) * P(B|A) * P(C|B)

# LearnSPN might learn:
#   P(A,B,C) = w1*P1(A,B,C) + w2*P2(A,B,C)  # Mixture
# Instead of:
#   P(A,B,C) = P(A) * P(B) * P(C)          # Product (if B _||_ C)
```

**Conclusion**: LearnSPN structure optimized for wrong objective

### 2. RAT-SPN May Actually Be Better for CI

**Argument**:
- Random structure = **unbiased** (no assumptions)
- Large capacity = can represent any distribution
- Let parameters learn, structure stays neutral

**LearnSPN**:
- Learned structure = **biased** toward training data
- May overfit to sample distribution
- Structure baked in = less flexible

**For CI tests**: Unbiased estimate > biased estimate with lower variance

### 3. Heterogeneity Handling

**FedCDH assumption**: Data is heterogeneous (multiple regimes)
```
P(X) = Σ_k w_k * P_k(X)
```

**LearnSPN**: Learns global structure
- May average out heterogeneity
- Loses per-client variation
- **Worse** for federated setting

**RAT-SPN + Clustering**:
- Each cluster has separate RAT-SPN
- Structure can differ per cluster
- **Better** for heterogeneity

---

## Alternative Improvements (Recommended)

Instead of LearnSPN, these would be more effective:

### Option 1: Hyperparameter Tuning (2-4 hours) ✅

**Current**:
```python
num_sums = 20
num_leaves = 20
num_repetitions = 10
```

**Try**:
```python
# Scale with dimensionality
num_sums = 20 + d * 2          # e.g., 36 for d=8
num_leaves = 20 + d * 2
num_repetitions = 10 + d // 2  # e.g., 14 for d=8
```

**Expected**: +5-10% LL improvement
**Effort**: 2 hours
**Risk**: Low

### Option 2: Pruning Low-MI Connections (4-6 hours) ⚡

**Idea**: Start with RAT-SPN, prune irrelevant connections

```python
class PrunedRATSPN:
    def __init__(self, ...):
        self.spn = Einet(config)  # RAT-SPN

    def train_and_prune(self, X, threshold=0.01):
        # 1. Train RAT-SPN normally
        self.spn.fit(X, epochs=50)

        # 2. Compute MI for each connection
        for sum_node in self.spn.sum_nodes:
            for edge in sum_node.edges:
                mi = compute_mutual_information(edge, X)
                if mi < threshold:
                    edge.weight = 0  # Prune

        # 3. Fine-tune
        self.spn.fit(X, epochs=20)
```

**Expected**: +10-15% LL, better CI tests
**Effort**: 4-6 hours
**Risk**: Medium

### Option 3: Ensemble of RAT-SPNs (1-2 hours) 🚀

**Idea**: Multiple RAT-SPNs with different seeds
```python
class EnsembleSPN:
    def __init__(self, n_models=5, ...):
        self.models = [
            Einet(config, seed=i) for i in range(n_models)
        ]

    def log_prob(self, X):
        # Average log-probs
        lls = [model.ll(X) for model in self.models]
        return torch.logsumexp(torch.stack(lls), dim=0) - np.log(len(self.models))
```

**Expected**: +5-10% accuracy (lower variance)
**Effort**: 1-2 hours
**Risk**: Low

---

## Recommendation for Thesis

### ✅ Keep RAT-SPN with Current Optimizations

**Reasons**:
1. ✅ Already working after 4× architecture increase
2. ✅ Proven to work in H/V/Hybrid federated scenarios
3. ✅ Fast enough (1-2 min training)
4. ✅ Theoretically reasonable (unbiased structure)
5. ✅ Thesis scope: Federated aggregation, not SPN optimization

### 📝 Document LearnSPN as Future Work

**In thesis**:
> "While LearnSPN (Gens & Domingos 2013) could potentially improve density
> estimation, its integration poses significant challenges for federated
> learning. LearnSPN's structure learning requires full data access for
> independence testing, which contradicts the federated setting where
> clients have disjoint features (vertical FL) or samples (horizontal FL).
>
> Furthermore, LearnSPN optimizes for likelihood P(X), not conditional
> independence P(X|Y,Z), which is the objective for causal discovery.
> Random structure (RAT-SPN) with sufficient capacity may provide more
> unbiased CI estimates.
>
> Future work could explore federated structure learning algorithms or
> hybrid approaches that combine random initialization with local pruning."

### 🔬 Optional: Quick Ablation Study (2 hours)

If time permits, compare:
- RAT-SPN (num_sums=5) - baseline
- RAT-SPN (num_sums=20) - current
- RAT-SPN (num_sums=30) - higher capacity
- RAT-SPN Ensemble (5 models) - variance reduction

**Purpose**: Show that capacity scaling is sufficient

---

## Conclusion

**LearnSPN integration**: ❌ Not recommended
- 20-28 hours effort
- Uncertain benefits
- Incompatible with federated learning
- API/implementation complexity

**Alternative**: ✅ RAT-SPN with optimizations
- Already working
- Fast to implement (<2 hours each)
- Proven in federated setting
- Sufficient for thesis scope

**Decision**: **Proceed with RAT-SPN optimizations** (Options 1-3 above) instead of LearnSPN integration.

**Next steps**:
1. Run ablation study on architecture scaling (2 hours)
2. Optional: Implement ensemble approach (1 hour)
3. Document findings in thesis
4. Move forward with causal discovery evaluation

---

**Time saved**: 20-28 hours
**Thesis impact**: Minimal (RAT-SPN already sufficient)
**Recommendation confidence**: High ✅
