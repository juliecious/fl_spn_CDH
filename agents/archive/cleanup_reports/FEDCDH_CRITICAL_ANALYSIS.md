# Critical Analysis: FedCDH.py Implementation

## Executive Summary

After thorough review with expertise in causality and federated learning, **the implementation has significant over-engineering and redundancy issues**. While the core causal discovery logic is sound, there are several areas that violate KISS principles.

---

## 🔴 CRITICAL ISSUES

### 1. **Redundant Epoch/Alpha Extraction (Lines 239-243, 348-352)**

**Problem**: Same code appears twice in the `fit()` method

```python
# Lines 239-243 (FIRST TIME)
if hasattr(self.args, "epochs"):
    train_epochs = self.args.epochs
else:
    train_epochs = 50 if self.device.type in ["cuda", "gpu"] else 10
alpha = self.args.alpha if hasattr(self.args, "alpha") else 0.05

# Lines 348-352 (DUPLICATE - SECOND TIME)
if hasattr(self.args, "epochs"):
    train_epochs = self.args.epochs
else:
    train_epochs = 50 if self.device.type in ["cuda", "gpu"] else 10
alpha = self.args.alpha if hasattr(self.args, "alpha") else 0.05
```

**Impact**: Code duplication, maintenance burden
**Fix**: Extract once at beginning of `fit()`, remove second occurrence
**Severity**: Medium (maintainability issue, not correctness)

---

### 2. **Over-Complex Feature Map Handling**

**Problem**: Feature maps are created but then mostly redundant

Lines 267-293 create `feature_maps` for all scenarios:
- **Horizontal**: `{k: list(range(d_aug_total)) for k}` - ALL clients see ALL features (redundant)
- **Vertical**: Actually needed (disjoint features)
- **Hybrid**: `{k: list(range(d_aug_total)) for k}` - Same as horizontal (redundant)

**Analysis**:
- Feature maps are ONLY truly needed for vertical scenario (disjoint features)
- For horizontal/hybrid, all clients see all features → feature_maps are just identity mappings
- This adds complexity without benefit

**Fix**:
```python
if self.scenario == "vertical":
    # Only create feature maps for vertical (actually disjoint)
    feature_maps = {...}
else:
    # Horizontal/hybrid: no need for feature maps
    feature_maps = None
```

**Severity**: Medium (unnecessary complexity, confuses readers)

---

### 3. **Unnecessary Data Reconstruction (Lines 246-256)**

**Problem**: Global data is reconstructed even when already available

```python
# Reconstruct Global for KCI/Oracle baselines
if isinstance(X_splits, list):
    if self.scenario == "vertical":
        X_global = np.concatenate(X_splits, axis=1)
    else:
        X_global = np.concatenate(X_splits, axis=0)
else:
    X_global = X_splits
```

**Analysis**:
- This reconstruction happens EVERY time
- For non-SPN methods (fisherz, kci), we could pass `X_global` directly
- Only SPNs need client-specific splits for federated training
- Concatenating then splitting is wasteful

**Causal Discovery Perspective**:
- Centralized CI tests (FisherZ, KCI) don't need splits at all
- Only federated SPN training needs splits
- Current design forces unnecessary work for baselines

**Fix**: Only reconstruct if needed by the CI method
**Severity**: Low-Medium (performance, not correctness)

---

### 4. **Confusing Routing Logic in FedCDH_SPN_Wrapper**

**Problem**: Lines 156-181 have confusing conditional logic

```python
def log_prob(self, x):
    if not self.routing:
        return self.spn.log_prob(x)  # Simple path

    # Complex routing logic (28 lines)
    if self.u_index == -1 or self.u_index == x.shape[1] - 1:
        x_feat = x[:, :-1]
        u_col = x[:, -1]
    else:
        x_feat = torch.cat([x[:, : self.u_index], x[:, self.u_index + 1 :]], dim=1)
        u_col = x[:, self.u_index]

    u_is_observed = not torch.isnan(u_col[0]).item()
    if u_is_observed:
        # Conditioning logic...
    else:
        return self.spn.log_prob(x_feat)
```

**Analysis**:
- The u_index slicing logic (lines 159-164) is overly general
- In practice, context U is ALWAYS the last column (u_index = d)
- The "else" case (lines 162-164) never executes in current usage
- Dead code that confuses readers

**From Causal Discovery**:
- Context variable U is always appended (standard practice)
- No use case where U would be in the middle

**Fix**: Simplify to assume U is always last column
**Severity**: Medium (confusing, but functionally correct)

---

### 5. **Unused Local SPNs Storage (Lines 433-448)**

**Problem**: Local SPNs stored but retrieval logic is convoluted

```python
# Lines 436-448: Complex logic to find "representative" SPN per client
self.local_spns = []
for k in range(self.K_clients):
    # Find the first SPN trained on client k's data across all clusters
    for h in range(num_clusters):
        if clients_clusters[h] and len(clients_clusters[h]) > k:
            self.local_spns.append(clients_clusters[h][k])
            break
    else:
        # Fallback: if no SPN found, use first available
        for h in range(num_clusters):
            if clients_clusters[h]:
                self.local_spns.append(clients_clusters[h][0])
                break
```

**Analysis**:
- This logic tries to get one SPN per client, but clusters may have different numbers of SPNs
- The "fallback" (lines 444-448) uses first available SPN (not client-specific)
- `local_spns` is only used for evaluation (external), not for causal discovery
- **This code doesn't affect discovery results** - it's purely for post-hoc analysis

**From Causal Discovery**:
- Causal discovery only uses `fed_spn_model` (the global model)
- Local SPNs are auxiliary information
- This complexity doesn't improve discovery accuracy

**Fix**: Either simplify or remove if not needed for paper/evaluation
**Severity**: Low (doesn't affect core algorithm)

---

### 6. **Voting PC Method is Dead Code (Lines 455-496)**

**Problem**: Lines 455-496 implement "voting_pc" which is never used

```python
if self.ci_method == "voting_pc":
    # 41 lines of code...
    # Run local PC on each client, then vote
```

**Analysis**:
- `ci_method` options in practice: "spn", "fisherz", "kci"
- "voting_pc" is never mentioned in args, configs, or tests
- Dead code that adds 41 lines and complexity
- From paper (Li et al. ICLR 2024): FedCDH uses centralized CI testing, not voting

**From Federated Causal Discovery Literature**:
- Voting-based approaches exist (e.g., FedPC-Vote) but are different algorithms
- FedCDH specifically uses global CI testing with federated SPNs
- Mixing paradigms confuses the implementation

**Fix**: Remove entire voting_pc branch
**Severity**: Medium (dead code, confuses intent)

---

### 7. **Over-Engineered BIC Cluster Selection (Lines 321-335)**

**Problem**: BIC search over [2, 3, 4, 5] clusters may be overkill

```python
best_h = 2
min_bic = float("inf")
best_model = None
for h_candidate in range(2, 6):  # Tests 2, 3, 4, 5 clusters
    fed_km = SimulatedFederatedKMeans(...)
    fed_km.fit(...)
    bic = fed_km.inertia_ + h_candidate * np.log(total_samples) * d_aug_total
    if bic < min_bic:
        min_bic = bic
        best_h = h_candidate
        best_model = fed_km
```

**Analysis**:
- Runs K-means 4 times (for h=2,3,4,5)
- Adds significant computational cost
- From Li et al. (ICLR 2024): Paper doesn't emphasize cluster count as critical
- Many papers use **fixed K=3** or K=5 for heterogeneity modeling

**Causal Discovery Perspective**:
- Cluster count affects SPN mixture quality, not causal structure directly
- Over-optimization here has diminishing returns
- Could use fixed K=3 (matches most experiments in paper)

**Fix**: Make cluster count a hyperparameter (default=3), optionally enable BIC search
**Severity**: Low-Medium (performance vs accuracy tradeoff)

---

## 🟡 MODERATE ISSUES

### 8. **Communication Cost Tracking is Vestigial**

**Problem**: Lines 260-261, 496-538 track `comm_cost` but it's not used for decisions

```python
comm_cost = 0.0
clustering_cost = 0.0
# ...
comm_cost = estimate_fedcdh_comm_cost(...)
```

**Analysis**:
- Communication cost is estimated but only returned in results
- Not used for any algorithmic decisions (e.g., early stopping, pruning)
- Purely for benchmarking/reporting
- **Not part of causal discovery algorithm**

**From Paper**:
- Communication cost is discussed as a theoretical advantage
- But actual algorithm doesn't adapt based on cost
- This is fine, but makes tracking somewhat redundant in core code

**Fix**: Move to separate analysis/benchmarking module
**Severity**: Low (doesn't affect algorithm, just cleanliness)

---

### 9. **Query Counter Wrapper is Over-Designed**

**Problem**: Lines 197-213 implement `QueryCounterCIT` with complex `__getattr__`

```python
class QueryCounterCIT:
    def __init__(self, cit_instance):
        self.cit = cit_instance
        self.query_count = 0
        self.method = getattr(cit_instance, "method", "unknown")

    def __call__(self, *args, **kwargs):
        self.query_count += 1
        if self.method in ["spn", "kci"]:
            return self.cit(*args, **kwargs)
        else:
            return self.cit(*args[:3])  # Why different signatures?

    def __getattr__(self, name):  # Magic method delegation
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.cit, name)
```

**Analysis**:
- Uses Python magic methods (`__getattr__`) for delegation
- Different call signatures for different methods (fragile)
- Query counting is simple - doesn't need this complexity
- Could be a simple wrapper incrementing counter

**Fix**: Simple counter class without magic methods
**Severity**: Low (works but overly clever)

---

## 🟢 MINOR ISSUES

### 10. **Device Selection Logic is Hardcoded (Lines 220-226)**

```python
if torch.cuda.is_available():
    self.device = torch.device("cuda")
else:
    self.device = torch.device("cpu")
```

**Analysis**:
- MPS (Apple Silicon GPU) is explicitly disabled - fine, documented reason
- But device is hardcoded, not configurable via args
- User can't force CPU even if CUDA available (e.g., for debugging)

**Fix**: Add `args.device` parameter (default="auto")
**Severity**: Low (convenience, not critical)

---

## ✅ WHAT'S ACTUALLY GOOD

### Core Causal Discovery Logic (Lines 507-524)
```python
cg = cdnod(
    X_global,
    c_indx,
    self.K_clients,
    alpha=alpha,
    indep_test=cit_counter,
    ...
)
```

**Analysis**: This is clean and correct!
- Uses CDNOD (constraint-based discovery) correctly
- Passes federated SPN as CI test
- Handles context variable properly
- **No issues here**

### SPN Aggregation Strategy (Lines 389-416)

**Analysis**: This logic is sophisticated but **necessary**
- Correctly chooses FederatedProduct (vertical) vs GlobalFedSPN (horizontal)
- Handles mixed scenarios
- Based on sound federated learning principles
- **Not over-engineered** - this complexity is inherent to federated SPNs

---

## 📊 SUMMARY TABLE

| Issue | Severity | Lines | Impact | Fix Effort |
|-------|----------|-------|--------|------------|
| Duplicate epoch/alpha extraction | Medium | 239-243, 348-352 | Maintainability | 5 min |
| Over-complex feature maps | Medium | 267-293 | Clarity | 15 min |
| Unnecessary data reconstruction | Low-Med | 246-256 | Performance | 10 min |
| Confusing routing logic | Medium | 156-181 | Clarity | 20 min |
| Unused local SPNs storage | Low | 433-448 | Clarity | 5 min (keep or simplify) |
| **Dead code: voting_pc** | **Medium** | **455-496** | **Clarity** | **5 min (delete)** |
| Over-engineered BIC search | Low-Med | 321-335 | Performance | 10 min (make optional) |
| Communication cost tracking | Low | Multiple | Clarity | 30 min (move to utils) |
| Query counter wrapper | Low | 197-213 | Clarity | 15 min |
| Device selection hardcoded | Low | 220-226 | UX | 5 min |

**Total Cleanup Time**: ~2 hours
**Impact on Correctness**: NONE (all issues are about clarity/performance, not bugs)

---

## 🎯 RECOMMENDED ACTIONS

### Priority 1 (Do Now):
1. **Remove voting_pc dead code** (lines 455-496) - 5 min
2. **Fix duplicate epoch/alpha extraction** - 5 min

### Priority 2 (Before Paper Submission):
3. **Simplify feature maps** for horizontal/hybrid - 15 min
4. **Clarify routing logic** in FedCDH_SPN_Wrapper - 20 min
5. **Add device selection parameter** - 5 min

### Priority 3 (Nice to Have):
6. **Make BIC search optional** (default to K=3) - 10 min
7. **Simplify query counter** - 15 min
8. **Move comm cost to utils** - 30 min

---

## ⚖️ FINAL VERDICT

**Over-Engineered?** Yes, moderately.

**Does it work correctly?** Yes, the causal discovery logic is sound.

**Is it publishable as-is?** Yes, but cleanup would improve clarity.

**Biggest Win**: Remove voting_pc (41 lines of dead code)

**Biggest Conceptual Issue**: Feature maps are over-designed for horizontal scenario

**Core Algorithm Quality**: ✅ The actual causal discovery (CDNOD + SPN CI) is well-implemented

---

## 📚 REFERENCES

- Li et al. (2024). "FedCDH: Federated Causal Discovery from Heterogeneous Data." ICLR.
- Peters et al. (2016). "Causal inference using invariant prediction." JRSS-B.
- Seng et al. (2025). "Federated Probabilistic Circuits." Under review.

---

*Analysis Date: 2026-03-31*
*Reviewer: Claude Code (Causal Discovery Expertise)*
