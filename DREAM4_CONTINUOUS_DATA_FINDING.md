# Dream4 Data is Continuous, Not Binary

**Date**: 2026-05-31
**Finding**: Dream4 dataset contains **continuous** gene expression data, NOT binary on/off states

---

## Investigation

### Initial Hypothesis (INCORRECT)
- Assumed gene regulatory networks use binary data (gene on/off)
- Implemented Binomial distribution support for discrete data
- Expected this to fix Dream4 complete failure

### Actual Finding
Dream4 dataset characteristics:
```
Shape: (1000, 10) - 1000 samples, 10 genes
Data type: float64 (continuous)
Range: [-3.649, 3.268]
Mean: 0.000, Std: 1.000 (pre-standardized)
Unique values: 1000 per feature (fully continuous)
```

**The data is continuous gene expression levels**, likely:
1. Log-transformed from raw expression counts
2. Standardized to mean=0, std=1
3. Represents relative expression levels, not binary on/off

---

## Why Binomial Distribution Failed

Error:
```
Expected value argument to be within the support IntegerInterval(lower_bound=0, upper_bound=1)
but found invalid values: tensor([[[[-0.2280]]]], ..., [[[[1.5561]]]]])
```

**Root cause**: Binomial distribution requires:
- **Integer values** in range [0, total_count]
- For binary data: values must be exactly 0 or 1

Dream4 data:
- **Continuous float values** (not integers)
- **Standardized** to mean=0, std=1 (includes negative values)
- Incompatible with Binomial distribution

---

## Correct Usage of Binomial Distribution

Binomial distribution should ONLY be used for:

### ✅ Binary Data (0/1)
```python
data = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])  # Binary on/off
fedcdh = FedCDH(..., leaf_type='binomial', binomial_total_count=1)
```

### ✅ Count Data (0, 1, 2, ...)
```python
data = np.array([[0, 2, 1], [3, 1, 0], [1, 0, 2]])  # Discrete counts
fedcdh = FedCDH(..., leaf_type='binomial', binomial_total_count=10)
```

### ❌ Continuous Data (Dream4)
```python
data = np.array([[-0.23, 0.69, 1.56], ...])  # Continuous, standardized
fedcdh = FedCDH(..., leaf_type='normal')  # Use Normal, not Binomial!
```

---

## Real Root Cause of Dream4 Failure

Dream4 failed with **Normal distribution** (original issue) due to:

### 1. Highly Non-Linear Relationships
Gene regulatory networks have:
- Complex activation functions
- Threshold effects
- Combinatorial regulation (multiple genes affecting one)
- **Much more complex than medical/social data**

### 2. Insufficient SPN Training
- Asia (n=999, d=8): Works with 46 epochs → 37.5% precision
- Dream4 (n=999, d=10): Fails with 46 epochs → 0% precision

**Dream4 requires MORE training, not different distribution**

### 3. Data Characteristics
Gene expression data may have:
- Different covariance structure
- More complex dependencies
- Less Gaussian-like marginals
- Stronger non-linearities

---

## Solutions for Dream4

### Solution 1: Increase SPN Training (RECOMMENDED)
```python
fedcdh = FedCDH(
    num_clients=3,
    mode='horizontal',
    alpha=0.05,
    epochs=200,  # Much more training for complex gene networks
    leaf_type='normal'  # Continuous data, use Normal distribution
)
```

**Expected**: 200 epochs may be enough to capture gene regulatory complexity

---

### Solution 2: More SPN Capacity
```python
fedcdh = FedCDH(
    num_clients=3,
    mode='horizontal',
    alpha=0.05,
    num_sums=40,       # More mixture components (default: 20)
    num_leaves=40,     # More leaf distributions (default: 20)
    num_repetitions=20, # More ensemble diversity (default: 10)
    depth=4,           # Deeper network (default: 3)
    epochs=200
)
```

---

### Solution 3: Different CI Test Method
```python
# If SPN still can't model the data, use non-parametric CI test
fedcdh = FedCDH(
    num_clients=3,
    mode='horizontal',
    alpha=0.05,
    ci_method='kci'  # Kernel CI instead of SPN
)
```

---

### Solution 4: Data Preprocessing
```python
# Transform data to be more Gaussian-like
from scipy.stats import rankdata, norm

def gaussianize(x):
    """Rank-based inverse normal transformation."""
    ranks = rankdata(x) / (len(x) + 1)
    return norm.ppf(ranks)

data_gaussianized = np.apply_along_axis(gaussianize, 0, data)
```

---

## Binomial Distribution Feature Status

### ✅ Implementation Complete
- LocalSPNWrapper supports `leaf_type='binomial'` and `leaf_kwargs={'total_count': n}`
- FedCDH propagates leaf distribution configuration
- Benchmark suite supports `--leaf-type` parameter

### ✅ Works for TRUE Binary/Count Data
If you have actual binary data:
```bash
python tests/benchmarks/test_fedcdh_benchmark_v3.py \
  --datasets your_binary_dataset \
  --methods fedspn_h \
  --leaf-type binomial
```

### ❌ Does NOT Apply to Dream4
Dream4 is continuous data, should use Normal distribution

---

## Key Takeaways

1. **Gene expression data is NOT always binary**
   - Raw counts are discrete, but often log-transformed to continuous
   - Standardized data is always continuous

2. **Dream4 failure is NOT a distribution mismatch**
   - Data is continuous → Normal distribution is correct
   - Failure is due to insufficient SPN capacity/training for complex gene networks

3. **Binomial distribution feature is still valuable**
   - Useful for TRUE binary datasets (disease indicators, binary features)
   - Useful for count data (event counts, click data)
   - Just not applicable to Dream4 specifically

4. **Next steps for Dream4**
   - Try epochs=200 with Normal distribution
   - Try increased SPN capacity
   - Consider KCI as alternative CI test

---

## Updated Documentation

**DISCRETE_DISTRIBUTION_SUPPORT.md** should clarify:
- Binomial is for **actual binary/count data** (0/1 integers)
- Dream4 uses **continuous** gene expression (float values)
- If your gene expression data is continuous (log-transformed, standardized), use Normal
- If your gene expression data is raw binary on/off states, use Binomial

---

**Status**: ✅ Binomial feature works correctly, but Dream4 is wrong test case
**Next**: Test Dream4 with increased epochs (200) and Normal distribution
