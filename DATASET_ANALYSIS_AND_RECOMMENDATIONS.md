# Dataset Analysis & SPN Distribution Recommendations

**Date**: 2026-05-31
**Analysis**: Real-world datasets for FedSPN-CDH experiments

---

## Summary Table

| Dataset | n | d | Data Type | Current Dist | **Recommended Dist** | Adaptive Epochs |
|---------|---|---|-----------|--------------|---------------------|-----------------|
| **Asia** | 1,000 | 8 | Continuous | Normal | ✅ **Normal** | 33 |
| **Sachs** | 5,400 | 11 | Discrete (3 levels) | Normal | ⚠️ **Categorical** | 495 |
| **Law School** | 21,000 | 5 | Mixed | Normal | ✅ **Normal** | 299 |
| **Dream4** | 1,000 | 10 | Continuous | Normal | ✅ **Normal** | 46 |

---

## Detailed Analysis

### 1. Asia (Medical Diagnosis) ✅ CORRECT

**Dataset Characteristics:**
```
Shape: n=1,000 samples, d=8 features
Features: Asia, Smoke, Tub, Lung, Bronc, Either, Xray, Dysp
Data Range: [-2.984, 8.624]
Mean: 0.000, Std: 1.000

Per-feature analysis:
- All 8 features: 1000 unique values (100% of n)
- All values: Continuous (non-integer)
- Data type: Pre-standardized continuous
```

**Distribution Assessment:**
- ✅ **Continuous data** - each feature has unique values for every sample
- ✅ **Pre-standardized** - mean=0, std=1
- ❌ Not binary - no features with only 2 values
- ❌ Not categorical - not discrete levels

**Recommended Distribution:** ✅ **Normal (current)**
- Continuous Gaussian distribution is appropriate
- Data is already standardized for Normal SPN

**SPN Training:**
- Base epochs: 20 (CPU device)
- Adaptive epochs: **33 epochs**
- Complexity: 2.024× (d=8)
- Data scale: 0.816× (n_per_client=333)
- Total scaling: 1.65×

---

### 2. Sachs (Protein Signaling) ⚠️ **SHOULD USE CATEGORICAL**

**Dataset Characteristics:**
```
Shape: n=5,400 samples, d=11 features
Features: Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk
Data Range: [1.000, 3.000]
Mean: 1.631, Std: 0.718

Per-feature analysis:
- All 11 features: EXACTLY 3 unique values each
- All values: Integer (1, 2, 3)
- Values represent: Low (1), Medium (2), High (3) protein expression levels
```

**Distribution Assessment:**
- ❌ **NOT continuous** - only 3 discrete levels per feature
- ✅ **Categorical/Ordinal data** - Low/Medium/High expression
- ✅ **All integer values** - discrete measurements
- ❌ Not binary - has 3 levels, not 2

**Current vs Recommended:**
- ❌ **Current: Normal** - Inappropriate for discrete 3-level data
- ✅ **Recommended: Categorical** - Proper for discrete classes

**Why Categorical is Better:**
```python
# Data structure (all features like this):
Raf: [1, 1, 2, 3, 1, 2, 3, 1, ...]  # Only values: 1, 2, 3
Mek: [2, 1, 1, 3, 2, 1, 3, 2, ...]  # Only values: 1, 2, 3
...

# Normal distribution assumes:
- Continuous values between 1.0 and 3.0
- Gaussian probability density
→ WRONG: Data has NO values like 1.5, 2.7, etc.!

# Categorical distribution assumes:
- Discrete classes: {1, 2, 3}
- Multinomial probability: P(class=1), P(class=2), P(class=3)
→ CORRECT: Matches actual data structure!
```

**Impact of Using Wrong Distribution:**
- Normal SPN treats discrete jumps (1→2→3) as continuous
- Likelihood estimates may be inaccurate
- Could affect CI test quality
- **However**: Current results are good (64.3% precision) despite wrong distribution!

**Recommendation:**
```python
# Change from:
fedcdh = FedCDH(..., leaf_type='normal')

# To:
fedcdh = FedCDH(..., leaf_type='categorical')
```

**SPN Training:**
- Base epochs: 80 (GPU device)
- Adaptive epochs: **495 epochs** 🔥
- Complexity: 3.263× (d=11)
- Data scale: 1.897× (n_per_client=1800)
- Total scaling: 6.19× (highest of all datasets!)

---

### 3. Law School (Social Science) ✅ CORRECT

**Dataset Characteristics:**
```
Shape: n=21,000 samples, d=5 features
Features: race, LSAT, UGPA, region_first, ZFYA
Data Range: [0.000, 1.000]
Mean: 0.503, Std: 0.336

Per-feature analysis:
- race: 2 unique (binary: 0 or 1)
- LSAT: 20,196 unique (continuous, normalized)
- UGPA: 20,744 unique (continuous, normalized)
- region_first: 18,123 unique (continuous, normalized)
- ZFYA: 17,047 unique (continuous, normalized)
```

**Distribution Assessment:**
- ✅ **Mixed data** - 1 binary + 4 continuous
- ✅ **Mostly continuous** - 4/5 features have 70%+ unique values
- ⚠️ **One binary feature** (race) - but minority
- ✅ **Normalized** to [0, 1] range

**Recommended Distribution:** ✅ **Normal (current)**
- Majority (4/5) features are continuous
- Normal distribution can handle the one binary feature
- Alternative: Could use **mixed distribution** (future enhancement)

**Why Normal is OK:**
```python
# Feature types:
race:         [0, 1, 0, 1, 0, ...]  # Binary (2 values)
LSAT:         [0.456, 0.789, 0.234, ...]  # Continuous
UGPA:         [0.567, 0.890, 0.345, ...]  # Continuous
region_first: [0.123, 0.678, 0.912, ...]  # Continuous
ZFYA:         [0.234, 0.789, 0.456, ...]  # Continuous

# Normal distribution:
- Works well for 4/5 continuous features
- Can approximate binary (0,1) as extreme-valued continuous
- Reasonable compromise for mixed data
→ ACCEPTABLE
```

**SPN Training:**
- Base epochs: 80 (GPU device)
- Adaptive epochs: **299 epochs**
- Complexity: 1.000× (d=5, simplest)
- Data scale: 3.742× (n_per_client=7000, largest)
- Total scaling: 3.74×

---

### 4. Dream4 (Gene Regulatory) ✅ CORRECT

**Dataset Characteristics:**
```
Shape: n=1,000 samples, d=10 features
Features: G0, G1, G2, G3, G4, G5, G6, G7, G8, G9
Data Range: [-3.649, 3.268]
Mean: 0.000, Std: 1.000

Per-feature analysis:
- All 10 features: 1000 unique values (100% of n)
- All values: Continuous (non-integer)
- Data type: Pre-standardized gene expression
```

**Distribution Assessment:**
- ✅ **Continuous data** - NOT binary as initially thought
- ✅ **Pre-standardized** - mean=0, std=1
- ❌ Not binary - data is continuous float values
- ✅ **Log-transformed** gene expression (likely)

**Recommended Distribution:** ✅ **Normal (current)**
- Data is continuous, not binary on/off gene expression
- Pre-standardized for Normal distribution
- Binomial would be WRONG (requires integer 0/1 values)

**Why Previous Binomial Attempt Failed:**
```python
# Actual Dream4 data:
G0: [-2.397, 0.554, -0.275, ...]  # Continuous floats!

# Binomial distribution requires:
G0: [0, 1, 0, 1, 1, 0, ...]  # Integer 0/1 only

# Error when using Binomial:
"Expected integer values in [0, 1] but found continuous floats"
→ Data is CONTINUOUS, not binary!
```

**Note:** Dream4 still fails (0 edges) because:
- Root cause: SPN cannot model complex gene regulatory dynamics
- NOT because of wrong distribution
- Solution: Use KCI instead of SPN-based CI test

**SPN Training:**
- Base epochs: 20 (CPU device)
- Adaptive epochs: **46 epochs**
- Complexity: 2.828× (d=10)
- Data scale: 0.816× (n_per_client=333)
- Total scaling: 2.30×

---

## Distribution Usage Guide

### When to Use Each Distribution

#### Normal (Continuous)
**Use for:**
- Continuous measurements (temperature, pressure, scores)
- Pre-standardized data (mean=0, std=1)
- High cardinality (many unique values)
- Real-valued features

**Datasets:**
- ✅ Asia (medical measurements)
- ✅ Law School (test scores, GPA)
- ✅ Dream4 (log-transformed gene expression)

**Implementation:**
```python
fedcdh = FedCDH(..., leaf_type='normal')  # Default
```

---

#### Categorical (Discrete Classes)
**Use for:**
- Discrete levels/categories (low/medium/high)
- Ordinal data (1st/2nd/3rd)
- Low cardinality (few unique values: 3-10)
- Integer-valued classes

**Datasets:**
- ⚠️ **Sachs** (protein levels: 1=low, 2=med, 3=high)

**Implementation:**
```python
fedcdh = FedCDH(..., leaf_type='categorical')
```

**Why Sachs Should Use Categorical:**
- Data has EXACTLY 3 discrete levels per feature
- Normal distribution assumes continuity between levels
- Categorical properly models discrete jumps
- More accurate likelihood estimates

---

#### Binomial (Binary)
**Use for:**
- Binary features (0/1, yes/no, on/off)
- Count data with known total_count
- Integer values only
- TRUE binary data (not continuous)

**Datasets:**
- ❌ None of our current datasets
- Dream4 is NOT binary (continuous floats)

**Implementation:**
```python
fedcdh = FedCDH(..., leaf_type='binomial', binomial_total_count=1)
```

**When You'd Use This:**
- Raw binary gene expression (not log-transformed)
- Disease indicators (has_disease: 0 or 1)
- Click data (clicked: 0 or 1)
- Binary survey responses

---

## Adaptive Epochs Summary

### Epoch Calculation

**Formula:**
```python
complexity_factor = (d / 5) ** 1.5
data_scale_factor = sqrt(n_per_client / 500)
adaptive_epochs = base_epochs × complexity_factor × data_scale_factor
```

**Base Epochs:**
- GPU: 80 epochs
- CPU: 20 epochs

### Results by Dataset

| Dataset | Base | Adaptive | Scaling | Why |
|---------|------|----------|---------|-----|
| **Law School** | 80 | **299** | 3.74× | Large n_per_client (7000) |
| **Sachs** | 80 | **495** | 6.19× | High d (11) + large n (1800) |
| **Dream4** | 20 | **46** | 2.30× | Moderate d (10), small n (333) |
| **Asia** | 20 | **33** | 1.65× | Moderate d (8), small n (333) |

### Key Insights

1. **Sachs gets MOST training** (495 epochs)
   - Highest dimensionality (d=11)
   - Large sample size (n=5400)
   - Most complex to learn

2. **Law School gets 2nd most** (299 epochs)
   - Lowest dimensionality (d=5) BUT
   - Largest dataset (n=21000)
   - Data scale dominates

3. **Dream4 gets moderate** (46 epochs)
   - Still not enough (fails with 0 edges)
   - Would need 200+ epochs or KCI

4. **Asia gets least** (33 epochs)
   - Small dataset (n=1000)
   - Moderate dimensionality (d=8)
   - CPU device (base=20)

---

## Recommendations

### Immediate Actions

1. **Change Sachs to Categorical** ⚠️ HIGH PRIORITY
   ```bash
   python tests/benchmarks/test_fedcdh_benchmark_v3.py \
     --datasets sachs \
     --methods fedspn_h \
     --seeds 42,43,44 \
     --leaf-type categorical
   ```

   **Expected impact:**
   - More accurate likelihood estimates
   - Potentially better precision (already 64.3%, could improve)
   - Proper modeling of discrete protein levels

2. **Keep Others as Normal** ✅
   - Asia: Continuous → Normal is correct
   - Law School: Mixed but mostly continuous → Normal is OK
   - Dream4: Continuous → Normal is correct (problem is complexity, not distribution)

### For Dream4 Failure

**NOT a distribution issue** - data is continuous, Normal is correct.

**Real solutions:**
1. **Use KCI instead of SPN** (recommended)
   ```python
   fedcdh = FedCDH(..., ci_method='kci')
   ```

2. **Increase epochs dramatically** (if sticking with SPN)
   ```python
   fedcdh = FedCDH(..., epochs=200)  # Force override
   ```

3. **Different method** - Gene networks may need specialized approach

---

## Expected Results After Changes

### If Sachs Uses Categorical

**Current (Normal):**
```
Precision: 64.3%
Recall: 47.4%
SPN training: 495 epochs with Normal distribution
```

**Expected (Categorical):**
```
Precision: 65-70% (slight improvement)
Recall: 48-52% (slight improvement)
SPN training: 495 epochs with Categorical distribution
Benefit: More principled, accurate discrete modeling
```

**Note:** Results may not change dramatically because:
- 495 epochs is already extensive training
- SPN was working reasonably well despite wrong distribution
- But conceptually correct and more principled

---

## Conclusion

### Distribution Assignments

| Dataset | Current | Recommended | Action |
|---------|---------|-------------|--------|
| Asia | Normal | Normal | ✅ Keep as-is |
| **Sachs** | Normal | **Categorical** | ⚠️ **CHANGE** |
| Law School | Normal | Normal | ✅ Keep as-is |
| Dream4 | Normal | Normal | ✅ Keep as-is |

### Epochs Summary

- **Sachs**: 495 epochs (most training)
- **Law School**: 299 epochs
- **Dream4**: 46 epochs (insufficient for complexity)
- **Asia**: 33 epochs (adequate for size)

### Key Insight

**Sachs is the ONLY dataset that should use a different distribution!**
- All features are discrete 3-level ordinal (1, 2, 3)
- Categorical distribution is proper choice
- Simple one-line change to fix

**All others should stay with Normal distribution:**
- Truly continuous data
- Pre-standardized
- High cardinality
