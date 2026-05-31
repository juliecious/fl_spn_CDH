# Discrete Distribution Support for SPNs

**Date**: 2026-05-31
**Purpose**: Enable SPNs to handle discrete/binary data (gene regulatory networks, categorical data)
**Solution**: Support multiple leaf distributions (Normal, Binomial, Categorical)

---

## Problem

Dream4 gene regulatory network dataset **completely failed** with continuous (Normal) SPNs:
- **Predicted edges**: 0 (should be 13)
- **All p-values**: 0.000
- **Test statistics**: 10-30× larger than normal
- **Root cause**: Continuous SPN cannot model discrete/binary gene expression data

---

## Solution

Added support for **multiple leaf distributions** in SPNs:

### Available Distributions

| Distribution | Use Case | Data Type | Example |
|--------------|----------|-----------|---------|
| **Normal** (default) | Continuous data | Real-valued | Medical, social, biological protein data |
| **Binomial** | Binary/count data | 0/1, counts | Gene expression (on/off), binary features |
| **Categorical** | Discrete classes | Multi-class | Categorical variables, discrete states |

---

## Implementation

### 1. Modified `LocalSPNWrapper` (local.py)

**Added imports**:
```python
from simple_einet.layers.distributions.normal import Normal
from simple_einet.layers.distributions.binomial import Binomial
from simple_einet.layers.distributions.categorical import Categorical
```

**Added parameter**:
```python
def __init__(
    self,
    num_features,
    device="cpu",
    ...
    leaf_type="normal",  # NEW: 'normal', 'binomial', or 'categorical'
):
```

**Added distribution selection logic**:
```python
if isinstance(leaf_type, str):
    leaf_type_lower = leaf_type.lower()
    if leaf_type_lower == "normal":
        leaf_dist = Normal
    elif leaf_type_lower == "binomial":
        leaf_dist = Binomial
    elif leaf_type_lower == "categorical":
        leaf_dist = Categorical
    else:
        raise ValueError(...)
else:
    # Already a distribution class
    leaf_dist = leaf_type

self.config = EinetConfig(
    ...,
    leaf_type=leaf_dist,  # Use selected distribution
)
```

---

### 2. Modified `FedCDH` Class (FedCDH.py)

**Added attribute**:
```python
# NEW: SPN leaf distribution type
self.leaf_type = getattr(args, "leaf_type", "normal")
if self.leaf_type != "normal":
    logging.info(
        f"Using {self.leaf_type} leaf distribution "
        f"(suitable for {'binary/count' if self.leaf_type == 'binomial' else 'discrete'} data)"
    )
```

**Propagated to all LocalSPNWrapper calls** (4 locations):
```python
spn = LocalSPNWrapper(
    num_features=local_d,
    device=self.device,
    ...
    leaf_type=self.leaf_type,  # NEW: Pass distribution type
)
```

---

## Usage

### For Continuous Data (Default)

```python
# No change needed - uses Normal distribution by default
fedcdh = FedCDH(
    num_clients=3,
    mode='horizontal',
    alpha=0.05
)
fedcdh.fit(data)
```

---

### For Binary/Gene Expression Data (Dream4)

```python
# Use Binomial distribution for binary gene expression
fedcdh = FedCDH(
    num_clients=3,
    mode='horizontal',
    alpha=0.05,
    leaf_type='binomial'  # NEW: For binary/count data
)
fedcdh.fit(data)
```

**Expected improvement for Dream4**:
- Current (Normal): 0 edges (complete failure)
- Expected (Binomial): 8-13 edges (working)
- More accurate likelihood estimates for discrete data

---

### For Categorical/Discrete Data

```python
# Use Categorical distribution for discrete multi-class data
fedcdh = FedCDH(
    num_clients=3,
    mode='horizontal',
    alpha=0.05,
    leaf_type='categorical'  # NEW: For discrete classes
)
fedcdh.fit(data)
```

---

## When to Use Each Distribution

### Normal (Continuous)
**Use for**:
- Medical data (blood pressure, BMI, etc.)
- Social science data (income, education level)
- Biological measurements (protein levels)
- Any real-valued continuous data

**Examples**:
- ✅ Asia (medical diagnosis)
- ✅ Sachs (protein signaling)
- ✅ Law School (test scores, GPA)

---

### Binomial (Binary/Count)
**Use for**:
- Gene expression (on/off, expressed/not expressed)
- Binary features (true/false, yes/no)
- Count data (number of events)
- Presence/absence data

**Examples**:
- 🎯 Dream4 (gene regulatory networks)
- Binary disease indicators
- Click/no-click (web analytics)
- Success/failure counts

---

### Categorical (Discrete)
**Use for**:
- Discrete states (low/medium/high)
- Multi-class labels (type A/B/C)
- Ordinal categories
- Discrete multinomial data

**Examples**:
- Disease severity (mild/moderate/severe)
- Gene expression levels (low/medium/high)
- Categorical survey responses

---

## Expected Results for Dream4

### Before (Normal Distribution)
```
Dataset: Dream4 Net1 (gene regulatory)
Distribution: Normal (continuous)
Result: 0 edges predicted (complete failure)
Precision: 0.0%
Test statistics: 1000-1400 (10-30× too high)
```

### After (Binomial Distribution)
```
Dataset: Dream4 Net1 (gene regulatory)
Distribution: Binomial (binary)
Expected: 8-13 edges (working)
Expected Precision: 50-70%
Test statistics: 50-200 (normal range)
```

---

## Technical Details

### Why Binomial for Gene Expression?

Gene expression data is often:
1. **Binary**: Gene is "on" (expressed) or "off" (not expressed)
2. **Count-based**: Number of transcripts (discrete counts)
3. **Non-negative**: Can't have negative expression
4. **Bounded**: Limited dynamic range

**Normal distribution** (continuous, unbounded, symmetric):
- Assumes values can be any real number (-∞ to +∞)
- Poor fit for binary on/off states
- Likelihood estimates are inaccurate

**Binomial distribution** (discrete, bounded, 0/1):
- Models binary outcomes naturally
- Correct statistical properties
- Accurate likelihood estimates

---

### Distribution Comparison

| Aspect | Normal | Binomial | Categorical |
|--------|--------|----------|-------------|
| **Support** | (-∞, +∞) | {0, 1} or counts | {0, 1, ..., K-1} |
| **Parameters** | μ (mean), σ (std) | n (trials), p (prob) | π (probabilities) |
| **Best for** | Continuous | Binary/count | Discrete classes |
| **Example** | Height, weight | On/off, clicks | Low/med/high |

---

## Validation

### Test on Dream4 with Binomial

```bash
# Run Dream4 with binomial distribution
python run_benchmark.py --dataset dream4_net1 --mode horizontal \
  --seed 42 --leaf_type binomial
```

**Expected improvements**:
1. ✅ Non-zero edges predicted (8-13 vs 0)
2. ✅ Varied p-values (not all 0.000)
3. ✅ Normal test statistics (50-200 vs 1000-1400)
4. ✅ Reasonable precision (50-70% vs 0%)

---

### Test Other Datasets (Verify No Regression)

```bash
# Verify Normal still works for continuous data
python run_benchmark.py --dataset asia --mode horizontal --seed 42
python run_benchmark.py --dataset sachs --mode horizontal --seed 42
python run_benchmark.py --dataset law_school --mode horizontal --seed 42
```

**Expected**: No regression (should work as before with Normal)

---

## Alternative Solution: Data Preprocessing

If distribution change doesn't help, consider:

### 1. Log-Transform Gene Expression
```python
# For non-negative expression data
data_transformed = np.log1p(data)  # log(1 + x)
# Then use Normal distribution
```

### 2. Standardize to Gaussian
```python
# Rank-based inverse normal transformation
from scipy.stats import rankdata, norm
def to_gaussian(x):
    ranks = rankdata(x) / (len(x) + 1)
    return norm.ppf(ranks)

data_gaussian = np.apply_along_axis(to_gaussian, 0, data)
```

### 3. Discretize Expression Levels
```python
# Convert to categorical (low/medium/high)
def discretize_expression(x, n_bins=3):
    return np.digitize(x, np.percentile(x, np.linspace(0, 100, n_bins+1)))

data_categorical = discretize_expression(data)
# Then use Categorical distribution
```

---

## Summary

**Problem**: Continuous Normal SPNs failed on discrete gene regulatory data

**Solution**: Added support for Binomial and Categorical distributions

**Usage**: Simply specify `leaf_type='binomial'` for binary/gene data

**Impact**:
- ✅ Dream4 expected to work (0 → 8-13 edges)
- ✅ No regression on other datasets (still use Normal)
- ✅ Flexible for future discrete datasets

**Next Step**: Test Dream4 with Binomial distribution to verify improvement

---

## Modified Files

1. **causallearn/utils/spn/core/local.py**:
   - Added Binomial, Categorical imports
   - Added leaf_type parameter
   - Added distribution selection logic

2. **causallearn/search/FCMBased/FedCDH/FedCDH.py**:
   - Added leaf_type attribute to FedCDH class
   - Propagated to all LocalSPNWrapper instantiations (4 locations)
   - Added logging for non-default distributions

---

**Status**: ✅ Ready for testing
**Next**: Run Dream4 experiment with `leaf_type='binomial'`
