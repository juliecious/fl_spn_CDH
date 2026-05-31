# Data Split Randomization - Important Issue

**Date**: 2026-05-31
**Issue**: Data split should be randomized with seed, not deterministic

---

## Current Behavior (WRONG)

**Code**: `tests/benchmarks/test_fedcdh_benchmark_v3.py:985-1007`

```python
if scenario == "horizontal":
    # Horizontal: split samples across clients
    n = X.shape[0]
    samples_per_client = n // K

    sample_maps = {}
    for k in range(K):
        start = k * samples_per_client
        end = (k + 1) * samples_per_client if k < K - 1 else n
        sample_maps[k] = list(range(start, end))  # DETERMINISTIC!
```

**Result:**
- Client 0: always gets samples [0, 1, 2, ..., 332]
- Client 1: always gets samples [333, 334, ..., 665]
- Client 2: always gets samples [666, 667, ..., 999]

**Same partition every time, regardless of seed!**

---

## Why This is Wrong

### 1. No Robustness Testing ❌
Can't assess whether results depend on **which samples** each client gets.

**Example**: What if Client 0 happens to get all the "easy" samples?
- Current: Can't test this - same split every time
- Should: Run with different seeds → different splits → assess robustness

### 2. Not Realistic for Federated Learning ❌
In real federated learning:
- Hospitals/institutions get "random" patients
- Data distribution to clients is not deterministic
- Each run could have different client data distributions

**Current setup doesn't simulate this randomness!**

### 3. Can't Compute Statistics ❌
To report mean ± std across multiple runs:
- Need **different data splits** per seed
- Current: Same split every time → can't compute variance

### 4. Potential Ordering Bias ⚠️
If CSV data has any ordering (even subtle):
- Time series data (sorted by date)
- Sorted by a variable
- Grouped by category

Then deterministic split introduces **systematic bias**:
- Client 0 gets data from one distribution
- Client 1 gets data from another distribution
- Client 2 gets data from yet another distribution

**This is NOT horizontal federation!** Should be i.i.d. distribution.

---

## Evidence: Data is Already Shuffled

I checked the benchmark datasets:

### Asia
```
Correlation between row index and features:
  Asia: -0.0196
  Smoke: -0.0084
  Tub: 0.0215
```
✅ Low correlation → data appears shuffled

### Dream4
```
Correlation between row index and features:
  G0: 0.0053
  G1: -0.0132
  G2: 0.0036
```
✅ Low correlation → data appears shuffled

**BUT**: This doesn't mean we should rely on it!
- Data might have been shuffled once with a specific seed
- Can't test different splits
- Not generalizable to new datasets

---

## What Should Happen (CORRECT)

### Randomized Split with Seed Control

```python
if scenario == "horizontal":
    n = X.shape[0]
    samples_per_client = n // K

    # IMPORTANT: Shuffle indices with seed
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # Shuffle sample indices
    shuffled_indices = rng.permutation(n)

    # Split shuffled indices across clients
    sample_maps = {}
    for k in range(K):
        start = k * samples_per_client
        end = (k + 1) * samples_per_client if k < K - 1 else n
        sample_maps[k] = shuffled_indices[start:end].tolist()
```

**Result:**
- Seed=42: Client 0 gets shuffled samples [234, 891, 456, ...]
- Seed=43: Client 0 gets different shuffled samples [567, 123, 789, ...]
- Seed=42 (again): Client 0 gets same as first run → reproducible!

---

## Benefits of Randomized Split

### 1. Robustness Testing ✅
```
Run 1 (seed=42): Client 0 gets samples A → Precision=64.3%
Run 2 (seed=43): Client 0 gets samples B → Precision=62.1%
Run 3 (seed=44): Client 0 gets samples C → Precision=65.8%

Average: 64.1% ± 1.9%
```

Can assess how sensitive results are to data partitioning!

### 2. Realistic Federated Learning ✅
Simulates real-world scenario where client data is "randomly" assigned.

### 3. Proper Statistics ✅
```python
# Run multiple seeds
results = []
for seed in [42, 43, 44, 45, 46]:
    precision = run_experiment(seed)
    results.append(precision)

# Report with confidence
print(f"Precision: {np.mean(results):.1f}% ± {np.std(results):.1f}%")
```

### 4. Fair Comparison ✅
Different methods should be tested on **same random splits**:
```python
for seed in seeds:
    # Same data split for all methods with this seed
    split = create_split(data, seed)

    result_ges = run_ges(split)
    result_fedspn = run_fedspn(split)
    # Both see same client data distributions
```

---

## Impact on Current Results

### Are Current Results Invalid? 🤔

**No, but...**:
1. ✅ Results are **valid** for the specific deterministic split used
2. ⚠️ We don't know if results are **robust** to different splits
3. ⚠️ Can't compute **variance** or confidence intervals
4. ⚠️ Not testing **realistic** federated scenarios

### Should We Re-Run Everything? 🤔

**Not necessarily**:
1. Current results are reproducible and comparable
2. Data appears already shuffled (low ordering bias)
3. Main findings (Sachs 64%, Asia 37%) likely robust

**But**:
- For publication/paper: Should use randomized splits with multiple seeds
- For robustness claims: Need to show results hold across splits
- For fair comparison: All methods should use same random splits

---

## Recommended Implementation

### Option 1: Minimal Change (Quick Fix)

```python
# In _run_fedspn() around line 985
if scenario == "horizontal":
    n = X.shape[0]
    samples_per_client = n // K

    # Shuffle with seed
    rng = np.random.RandomState(seed) if seed is not None else np.random
    shuffled_indices = rng.permutation(n)

    sample_maps = {}
    for k in range(K):
        start = k * samples_per_client
        end = (k + 1) * samples_per_client if k < K - 1 else n
        sample_maps[k] = shuffled_indices[start:end].tolist()
```

**Impact**: Different seeds → different data splits → results may vary

---

### Option 2: Controlled Mode (Recommended)

```python
# Add parameter to control behavior
parser.add_argument(
    "--shuffle-data",
    action="store_true",
    default=True,  # Default: shuffle (realistic)
    help="Shuffle data before splitting to clients (default: True for robustness testing)"
)

# In _run_fedspn()
if shuffle_data:
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(n)
else:
    shuffled_indices = np.arange(n)  # Deterministic

sample_maps = {}
for k in range(K):
    start = k * samples_per_client
    end = (k + 1) * samples_per_client if k < K - 1 else n
    sample_maps[k] = shuffled_indices[start:end].tolist()
```

**Benefits**:
- Default: Shuffle (realistic, proper testing)
- Can disable for debugging (deterministic, reproducible)
- Explicit control over behavior

---

## Expected Changes in Results

### With Randomized Split

**Same seed:**
- ✅ Reproducible (seed=42 always gives same split)

**Different seeds:**
- ⚠️ Results will vary (different client data distributions)
- Example: Precision might be 64.3%, 62.1%, 65.8% across seeds

**Statistical reporting:**
```
Sachs: 64.1% ± 1.9% (n=5 seeds)
Asia:  37.5% ± 2.3% (n=5 seeds)
```

### Will Dream4 Results Change?

**Probably NOT significantly**, because:
1. Dream4 is already failing completely (0 edges)
2. Root cause is SPN architecture mismatch, not data split
3. Even with different splits, SPN can't model gene networks

**But**: Should still use randomized split for proper testing!

---

## Immediate Action

### Quick Test: Does Split Matter?

Let's manually test if different splits affect results:

```python
# Test on Asia with 3 different manual splits
splits_to_test = [
    # Split 1: Sequential (current)
    {0: list(range(0, 333)), 1: list(range(333, 666)), 2: list(range(666, 999))},

    # Split 2: Interleaved
    {0: list(range(0, 999, 3)), 1: list(range(1, 999, 3)), 2: list(range(2, 999, 3))},

    # Split 3: Random (seed=42)
    # ... random split ...
]

for split_id, sample_maps in enumerate(splits_to_test):
    result = run_fedspn(data, sample_maps)
    print(f"Split {split_id}: Precision={result['precision']}")
```

If results are very similar → not sensitive to split
If results vary a lot → sensitive to split → need randomization!

---

## Conclusion

### ✅ You Are Correct!

Data split **SHOULD be randomized** with seed control for:
1. Proper robustness testing
2. Realistic federated learning simulation
3. Computing statistics across runs
4. Fair comparison across methods

### Current Limitation

- Deterministic split: Can't assess robustness
- Same results every seed: Can't compute variance
- Not testing realistic scenarios

### Recommendation

**Implement randomized split with seed control** (Option 2):
```python
# Shuffle data indices with seed before splitting
rng = np.random.RandomState(seed)
shuffled_indices = rng.permutation(n)

# Then split shuffled indices to clients
sample_maps = {k: shuffled_indices[start:end] for k in range(K)}
```

**This is standard practice in federated learning research!**

---

**Should I implement this change?** It's a simple but important fix.
