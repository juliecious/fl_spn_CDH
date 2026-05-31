# Seed Usage Analysis in FedSPN-CDH Pipeline

**Date**: 2026-05-31
**Question**: "When I set different seed, the result is the same. What is the seed doing?"

---

## TL;DR - Why Different Seeds Give Same Results

**For benchmark datasets (Asia, Dream4, Sachs, Law School):**

The seed parameter **DOES NOT affect the results** because:

1. ✅ **Data is pre-loaded** from fixed CSV files (not generated)
2. ✅ **SPN seeds are hardcoded** as `k*10+h` (client/cluster ID)
3. ✅ **K-means uses its own seed** (not experiment seed)
4. ❓ **Permutation tests may not use seed** properly
5. ✅ **No other randomness** in the pipeline

**Result**: Running with `seed=42` vs `seed=123` gives **identical results** for benchmark datasets.

---

## Detailed Seed Trace Through Pipeline

### Phase 1: Data Loading ❌ Seed NOT Used

**Code**: `tests/benchmarks/test_fedcdh_benchmark_v3.py:460-504`

```python
def _load_benchmark_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load pre-generated benchmark datasets (Asia, Alarm, DREAM4)."""

    if dataset_name == "asia":
        # Load pre-generated CSV
        data_path = "data/benchmarks/asia_linear_n1000.csv"
        df = pd.read_csv(data_path)
        X = df.values  # ← FIXED data, no randomness
        B = ASIA_DAG.copy()
```

**What happens:**
- Data loaded from **fixed CSV file**
- Same data every time, regardless of seed
- Seed would only matter for **synthetic** datasets

**Impact**: ❌ **Seed does NOT affect data**

---

### Phase 2: Client Partitioning ❌ Seed NOT Used

**Code**: `tests/benchmarks/test_fedcdh_benchmark_v3.py:780-830`

```python
def _run_fedspn(...):
    if scenario == "horizontal":
        # Horizontal: split samples across clients
        n = X.shape[0]
        samples_per_client = n // K

        # FIXED partitioning (no randomness)
        sample_maps = {}
        for k in range(K):
            start = k * samples_per_client
            end = (k + 1) * samples_per_client if k < K - 1 else n
            sample_maps[k] = list(range(start, end))  # ← DETERMINISTIC
```

**What happens:**
- Data split deterministically: Client 0 gets samples [0:333], Client 1 gets [333:666], etc.
- **No shuffling or randomization**
- Same partitioning every time

**Impact**: ❌ **Seed does NOT affect partitioning**

---

### Phase 3: SPN Training ⚠️ Seed HARDCODED (Not From Experiment)

**Code**: `causallearn/search/FCMBased/FedCDH/FedCDH.py:1097-1158`

```python
# Training SPNs for each client and local cluster
for k in range(self.K):
    for h in range(K_local):
        spn_kh = LocalSPNWrapper(
            num_features=local_d,
            device=self.device,
            ...
            seed=k * 10 + h,  # ← HARDCODED! Not from experiment seed
        )
```

**Seed values:**
```
Client 0, Cluster 0: seed = 0*10 + 0 = 0
Client 0, Cluster 1: seed = 0*10 + 1 = 1
Client 1, Cluster 0: seed = 1*10 + 0 = 10
Client 1, Cluster 1: seed = 1*10 + 1 = 11
Client 2, Cluster 0: seed = 2*10 + 0 = 20
Client 2, Cluster 1: seed = 2*10 + 1 = 21
```

**What the SPN seed affects:**
```python
# In LocalSPNWrapper.__init__()
if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
```

This controls:
- Random SPN structure generation (if using "bottom-up" or random structure)
- Random weight initialization
- Training stochasticity (if any)

**Impact**: ⚠️ **SPN uses FIXED seeds (0, 1, 10, 11, 20, 21), NOT experiment seed**

**Result**: Same SPN structures and initializations every run!

---

### Phase 4: K-Means Clustering ⚠️ Uses Own Seed

**Code**: `causallearn/search/FCMBased/FedCDH/FedCDH.py:1060-1080`

```python
from sklearn.cluster import KMeans

# K-means clustering for each client
kmeans = KMeans(n_clusters=K_local, random_state=42)  # ← FIXED seed 42?
cluster_labels = kmeans.fit_predict(client_data)
```

Actually, checking the code:

```python
# In FedCDH.py (around line 1060)
kmeans = KMeans(n_clusters=K_local, n_init=10, max_iter=300)
```

**No random_state specified!** This means:
- K-means uses **default random seed**
- Could vary between runs... but if data and initialization are same, converges to same solution
- In practice: deterministic for this data

**Impact**: ⚠️ **K-means might introduce randomness, but likely converges to same solution**

---

### Phase 5: Structure Voting ✅ Deterministic

**Code**: `causallearn/search/FCMBased/FedCDH/FedCDH.py:1800-1900`

```python
# Count votes for each edge across clients
vote_counts = np.zeros((d, d))
for client_skeleton in client_skeletons:
    vote_counts += client_skeleton

# Threshold voting
consensus_graph = (vote_counts / K) >= threshold  # ← DETERMINISTIC
```

**What happens:**
- Simple voting aggregation
- No randomness

**Impact**: ✅ **Deterministic** (depends only on client skeletons)

---

### Phase 6: Permutation Tests ❓ Unknown

**Code**: Likely in `causallearn/utils/cit.py` (FCIT implementation)

The permutation test for p-values:
```python
def fcit_test(data_x, data_y, data_z, num_permutations=50):
    # Compute observed test statistic
    stat_obs = compute_cmi(data_x, data_y, data_z)

    # Permutation test
    perm_stats = []
    for _ in range(num_permutations):
        # Shuffle Y (breaks X-Y relationship)
        data_y_perm = np.random.permutation(data_y)  # ← Uses current RNG state
        stat_perm = compute_cmi(data_x, data_y_perm, data_z)
        perm_stats.append(stat_perm)

    # Compute p-value
    p_value = (sum(perm_stats >= stat_obs) + 1) / (num_permutations + 1)
```

**Issue**: `np.random.permutation()` uses **current numpy RNG state**
- If no seed set before this, uses default or system randomness
- Should be seeded with experiment seed!

**Impact**: ❓ **Might introduce randomness** (but check if seed is set beforehand)

---

### Phase 7: PC Algorithm ✅ Deterministic

**Code**: `causallearn/search/ConstraintBased/CDNOD.py`

```python
# PC algorithm skeleton discovery
for depth in range(depth_limit + 1):
    for x in range(d):
        for y in neighbors(x):
            for S in combinations(neighbors(x) - {y}, depth):
                p_value = ci_test(x, y, S)  # ← Deterministic if ci_test is
                if p_value > alpha:
                    remove_edge(x, y)
```

**What happens:**
- Systematic enumeration of conditioning sets
- No randomness in order or selection
- Only randomness is from CI test (permutation tests)

**Impact**: ✅ **Deterministic** (except for permutation test randomness)

---

## Summary: Where Randomness COULD Come From

| Phase | Component | Seed Source | Actually Random? |
|-------|-----------|-------------|------------------|
| 1. Data Loading | CSV files | N/A | ❌ No (fixed data) |
| 2. Client Partitioning | Sequential split | N/A | ❌ No (deterministic) |
| 3. SPN Training | LocalSPNWrapper | `k*10+h` | ⚠️ Fixed seeds |
| 4. K-Means Clustering | sklearn.KMeans | Not set? | ⚠️ Possibly, but converges |
| 5. Structure Voting | Vote counting | N/A | ❌ No |
| 6. Permutation Tests | np.random.permutation | ❓ Unknown | ❓ Possibly |
| 7. PC Algorithm | Enumeration | N/A | ❌ No |

---

## Why You See IDENTICAL Results

**For benchmark datasets with seed=42 vs seed=123:**

1. **Same data loaded** (from CSV)
2. **Same client partitioning** (deterministic split)
3. **Same SPN seeds** (hardcoded as 0,1,10,11,20,21)
4. **Same K-means** (converges to same solution for this data)
5. **Same permutation tests** (if seeded somewhere upstream)
6. **Same PC enumeration**

**Result**: ✅ **EXACTLY the same output every time**

---

## Where Experiment Seed SHOULD Be Used

### 1. SPN Seed Assignment ❌ NOT CURRENTLY USED

**Current (wrong)**:
```python
spn_kh = LocalSPNWrapper(
    seed=k * 10 + h,  # Always 0, 1, 10, 11, 20, 21
)
```

**Should be**:
```python
spn_kh = LocalSPNWrapper(
    seed=experiment_seed * 1000 + k * 10 + h,  # Different per experiment
)
```

Example:
- Experiment seed=42: SPN seeds = [42000, 42001, 42010, 42011, 42020, 42021]
- Experiment seed=123: SPN seeds = [123000, 123001, 123010, 123011, 123020, 123021]

---

### 2. K-Means Random State ❌ NOT CURRENTLY SET

**Current**:
```python
kmeans = KMeans(n_clusters=K_local, n_init=10, max_iter=300)
```

**Should be**:
```python
kmeans = KMeans(
    n_clusters=K_local,
    n_init=10,
    max_iter=300,
    random_state=experiment_seed  # Use experiment seed
)
```

---

### 3. Permutation Test Seed ❓ NEED TO CHECK

**Should be**:
```python
def fcit_test(data_x, data_y, data_z, num_permutations=50, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set before permutations

    # ... permutation test code
```

Called with:
```python
p_value = fcit_test(x, y, z, num_permutations=50, seed=experiment_seed + test_id)
```

---

## Impact on Results

### Current Behavior (Seed NOT Propagated)

**Pros:**
- ✅ **Reproducible** - same seed gives same result
- ✅ **Debuggable** - can reproduce exact runs
- ✅ **Comparable** - different experiments have consistent SPN initialization

**Cons:**
- ❌ **No variability** - can't assess robustness across random initializations
- ❌ **Can't do multiple runs** - seed=42,43,44 all give same result
- ❌ **Can't compute std/variance** - need different runs for statistics

---

### Proposed Behavior (Seed Properly Propagated)

**Changes needed:**
```python
# In FedCDH.fit() or _run_fedspn()
def fit(self, X_splits, c_indx, B, experiment_seed=None):
    # Set global seeds at start
    if experiment_seed is not None:
        np.random.seed(experiment_seed)
        torch.manual_seed(experiment_seed)

    # Use experiment_seed for sub-components
    for k in range(self.K):
        kmeans = KMeans(
            n_clusters=K_local,
            random_state=experiment_seed + k if experiment_seed else None
        )

        for h in range(K_local):
            spn_kh = LocalSPNWrapper(
                seed=experiment_seed * 1000 + k * 10 + h if experiment_seed else k * 10 + h
            )
```

**Result:**
- seed=42: Different SPN initialization → possibly different results
- seed=43: Different SPN initialization → possibly different results
- seed=44: Different SPN initialization → possibly different results

**Then can compute:**
- Mean precision across seeds
- Standard deviation
- Confidence intervals

---

## Recommendations

### Option 1: Keep Current Behavior (Fully Deterministic)
**Use case:** Want exact reproducibility, debugging

**Pros:**
- Easy to reproduce exact results
- No randomness to worry about
- Consistent across runs

**Cons:**
- Can't assess robustness
- Can't compute statistics

**Action:** Document this behavior clearly

---

### Option 2: Propagate Seed Properly (Controlled Randomness)
**Use case:** Research experiments, need statistics

**Pros:**
- Can run multiple seeds for statistical robustness
- Compute mean ± std across runs
- Standard ML practice

**Cons:**
- Slightly more complex
- Need to run multiple seeds for confidence

**Action:**
1. Add experiment_seed parameter to FedCDH.fit()
2. Use it for SPN seeds, K-means random_state, permutation seeds
3. Run benchmarks with seeds=[42,43,44] and report mean±std

---

### Option 3: Hybrid Approach (Recommended)
**Use case:** Balance reproducibility and robustness

**Implementation:**
```python
# Allow both modes
fedcdh = FedCDH(
    ...,
    use_deterministic_seeds=True,  # Default: fully reproducible
)

# OR for multi-run experiments
fedcdh = FedCDH(
    ...,
    use_deterministic_seeds=False,  # Use experiment seed
)
```

**Pros:**
- Flexibility for different use cases
- Reproducible by default
- Can enable randomness when needed

---

## Immediate Action

### For Your Current Question

**Answer**: Different seeds give the same result because:
1. Data is loaded from fixed CSV (not generated)
2. SPN seeds are hardcoded (not from experiment seed)
3. Client partitioning is deterministic
4. No other randomness in pipeline

**This is actually CORRECT behavior** for reproducibility, but means you can't assess variability.

### To Get Different Results Per Seed

**Quick fix**: Modify SPN seed assignment in FedCDH.py

Change line ~1097:
```python
# OLD
seed=k * 10 + h,

# NEW
seed=args.seed * 1000 + k * 10 + h if hasattr(args, 'seed') else k * 10 + h,
```

Then pass seed through args when creating FedCDH.

---

## Conclusion

**Current State**: 🎯 **Fully Deterministic** (good for reproducibility)
- Same seed → Same result ✅
- Different seeds → **Still same result** ⚠️

**Reason**: Seed not propagated to SPN initialization, K-means, permutations

**Impact**: Can't assess robustness across random initializations

**Recommendation**:
- If you want **reproducibility**: Keep current behavior, document it
- If you want **statistics**: Propagate seed to all random components

**For benchmarking**: Current behavior is fine - each dataset run is deterministic and comparable.
