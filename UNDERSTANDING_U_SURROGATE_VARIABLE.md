# Understanding U: The Surrogate/Context Variable in FedCDH

**Question**: How is U derived in FedCDH, and with SPN how could U (surrogate variable) be obtained?

---

## What is U?

**U is the augmented/context/surrogate variable** that represents which **client (domain)** each sample belongs to.

- In horizontal mode: U = client ID (which client owns the sample)
- In vertical mode: U is typically constant (all clients see same samples)
- In hybrid mode: U = client ID for sample-partitioned data

**Purpose**: U captures domain heterogeneity and enables:
1. Causal discovery across federated domains
2. Mechanism invariance-based orientation (FICP)
3. Conditioning for more accurate CI tests

---

## How U is Created (Data Augmentation)

### Step 1: Original Data

```python
# Asia dataset example
X_train.shape = (999, 8)  # 999 samples, 8 features
# Variables: [asia, smoke, tub, lung, bronc, either, xray, dysp]
```

### Step 2: Create Context Indices (c_indx)

**Code** (from `test_fedcdh_benchmark_v3.py`):

```python
# For horizontal partitioning with K=3 clients
K = 3
n = 999
samples_per_client = n // K  # 333 samples per client

# Create c_indx: [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2]
c_indx = np.repeat(np.arange(K), samples_per_client)  # [0]*333 + [1]*333 + [2]*333

# Handle remainder samples (999 % 3 = 0, no remainder for Asia)
remainder = n % K
if remainder > 0:
    c_indx = np.concatenate([c_indx, np.arange(remainder)])

c_indx = c_indx[:n].reshape(-1, 1)  # Shape: (999, 1)
```

**Result**:
```python
c_indx = [[0],      # Sample 0 belongs to client 0
          [0],      # Sample 1 belongs to client 0
          ...       # ...
          [0],      # Sample 332 belongs to client 0
          [1],      # Sample 333 belongs to client 1
          [1],      # Sample 334 belongs to client 1
          ...       # ...
          [2],      # Sample 666 belongs to client 2
          ...       # ...
          [2]]      # Sample 998 belongs to client 2
```

### Step 3: Augment Data with U

**Code** (from `FedCDH.py:601`):

```python
# Original data
X_global.shape = (999, 8)

# Concatenate context column
X_aug_global = np.concatenate([X_global, c_indx], axis=1)

# Augmented data
X_aug_global.shape = (999, 9)  # 8 features + 1 context variable
```

**Result**:
```python
X_aug_global = [
    [asia_0, smoke_0, tub_0, lung_0, bronc_0, either_0, xray_0, dysp_0, 0],  # Client 0
    [asia_1, smoke_1, tub_1, lung_1, bronc_1, either_1, xray_1, dysp_1, 0],  # Client 0
    ...
    [asia_333, smoke_333, ..., 1],  # Client 1
    ...
    [asia_666, smoke_666, ..., 2],  # Client 2
    ...
]
```

**Column 8 (index 8) = U (context variable)**

---

## How U is Used in SPN

### 1. SPN Training on Augmented Data

**Local SPN Training** (per client):

```python
# Client 0 trains on its samples (with U=0)
X_client_0 = X_aug_global[0:333, :]  # Shape: (333, 9)
# Trains LocalSPN_0 on P(X_0, X_1, ..., X_7, U | U=0)

# Client 1 trains on its samples (with U=1)
X_client_1 = X_aug_global[333:666, :]  # Shape: (333, 9)
# Trains LocalSPN_1 on P(X_0, X_1, ..., X_7, U | U=1)

# Client 2 trains on its samples (with U=2)
X_client_2 = X_aug_global[666:999, :]  # Shape: (333, 9)
# Trains LocalSPN_2 on P(X_0, X_1, ..., X_7, U | U=2)
```

**Global SPN** (mixture):

```python
# GlobalSPN = weighted mixture of local SPNs
GlobalSPN(x_0, ..., x_7, u) = (1/3) * LocalSPN_0(x_0, ..., x_7, u) +
                               (1/3) * LocalSPN_1(x_0, ..., x_7, u) +
                               (1/3) * LocalSPN_2(x_0, ..., x_7, u)

# This learns P(X_0, ..., X_7, U) over all data
```

---

## How U is Obtained from SPN for CI Tests

### Method 1: Direct Conditioning (Structure Voting Approach)

**Conditioning on U explicitly in CI tests**:

```python
# Test: X_i ⊥ X_j | U
# This tests if X_i and X_j are conditionally independent given the client ID

def ci_test_with_U(X_i, X_j, U, GlobalSPN):
    """
    Test conditional independence: X_i ⊥ X_j | U
    """
    # Compute log-likelihoods using SPN
    ll_ijk = log P(X_i, X_j, U)     # Joint
    ll_ik  = log P(X_i, U)          # Marginal of X_i and U
    ll_jk  = log P(X_j, U)          # Marginal of X_j and U
    ll_k   = log P(U)               # Marginal of U

    # Conditional Mutual Information (CMI)
    CMI = ll_ijk - ll_ik - ll_jk + ll_k

    # If CMI ≈ 0 → X_i ⊥ X_j | U (conditionally independent)
    # If CMI >> 0 → X_i ⊥̸ X_j | U (conditionally dependent)

    # Use permutation test to get p-value
    return p_value
```

**How SPN computes these**:

```python
# Given data point: [x_0, x_1, ..., x_7, u]
# SPN has learned P(X_0, ..., X_7, U)

# 1. Joint P(X_i, X_j, U)
data_ijk = [x_i, x_j, u]
ll_ijk = GlobalSPN.log_likelihood(data_ijk,
                                   marginalize=[0,1,...,7] except i,j)

# 2. Marginal P(X_i, U)
ll_ik = GlobalSPN.log_likelihood([x_i, u],
                                  marginalize=[0,1,...,7] except i)

# 3. Marginal P(X_j, U)
ll_jk = GlobalSPN.log_likelihood([x_j, u],
                                  marginalize=[0,1,...,7] except j)

# 4. Marginal P(U)
ll_k = GlobalSPN.log_likelihood([u],
                                 marginalize=[0,1,...,7])
```

**Key Point**: SPN can marginalize over variables to compute any joint/marginal distribution needed.

---

### Method 2: Implicit in Mixture (Current Main PC)

**Without explicit conditioning on U**:

```python
# Test: X_i ⊥ X_j | Z  (where Z doesn't include U)
# This computes marginals over U implicitly

# Problem: When Z=[], computing P(X_i, X_j) marginalizes over U
# Result: Marginal dependence inflated by U's influence
# All pairs appear dependent (p=0.000)
```

**Why this is problematic**:

```python
# The SPN learned: P(X_0, ..., X_7 | U)
# where variables are correlated within each client/domain

# When computing P(X_i, X_j) = ∫ P(X_i, X_j | U) P(U) dU
# The marginal includes dependence through U

# Example:
# - Within Client 0: X_i and X_j might be independent given U=0
# - Within Client 1: X_i and X_j might be independent given U=1
# - But marginally (averaging over U): X_i and X_j appear dependent!
```

---

## Obtaining U from SPN: Practical Methods

### Approach 1: Direct Extraction (Current Implementation)

**U is directly available** - it's column 8 in the augmented data:

```python
# During CI test
X_aug_sample = [x_0, x_1, ..., x_7, u]  # Shape: (9,)
u_value = X_aug_sample[8]  # Extract U directly

# Pass to SPN for likelihood computation
ll = GlobalSPN.log_likelihood(X_aug_sample)
```

**Advantage**: Simple, direct, no inference needed

**Usage**: This is what structure voting does

---

### Approach 2: Posterior Inference (Alternative)

**Infer U from observed features**:

```python
# Given only X = [x_0, ..., x_7], infer U
# Compute P(U | X) using Bayes rule

def infer_U(X, GlobalSPN):
    """
    Infer most likely client for given observation.
    """
    # Compute P(U=k | X) for each client k
    posteriors = []
    for k in range(K):
        # P(U=k | X) ∝ P(X | U=k) * P(U=k)
        # where P(U=k) = 1/K (uniform prior)

        X_with_u = np.concatenate([X, [k]])
        log_prob = GlobalSPN.log_likelihood(X_with_u)
        posteriors.append(log_prob)

    # Return most likely U
    u_inferred = np.argmax(posteriors)
    return u_inferred
```

**Advantage**: Can infer U even when not directly observed

**Usage**: Useful for deployment on new data without known client IDs

---

### Approach 3: Sampling from SPN (For Missing U)

**Sample U from conditional distribution**:

```python
def sample_U(X, GlobalSPN, num_samples=100):
    """
    Sample U values conditioned on observed X.
    """
    u_samples = []
    for _ in range(num_samples):
        # Sample from P(U | X) using SPN
        u_sample = GlobalSPN.sample_conditional(
            evidence={0: x_0, 1: x_1, ..., 7: x_7},
            query_vars=[8]
        )
        u_samples.append(u_sample[8])

    return u_samples

# Use samples for CI test with uncertainty
```

**Advantage**: Captures uncertainty in U

**Usage**: Research scenarios with latent domains

---

## Current Usage in FedCDH

### Structure Voting Phase

```python
# From aggregation.py:36-42
# Always condition on augmented variable (column 8)

Z = [d_features]  # d_features = 8 (the augmented variable index)

# Test: X_i ⊥ X_j | U
ll_xyz = spn.log_likelihood([X_i, X_j, U])
ll_xz  = spn.log_likelihood([X_i, U])
ll_yz  = spn.log_likelihood([X_j, U])
ll_z   = spn.log_likelihood([U])

# Compute CMI and p-value
# Result: Accurate independence detection (22 edges with ~94% confidence)
```

**This works well!** ✅

---

### Main PC Algorithm (Current Issue)

```python
# From CDNOD.py
# Does NOT condition on U by default

# Depth 0: Z = []
# Test: X_i ⊥ X_j | []  (no conditioning)

ll_xyz = spn.log_likelihood([X_i, X_j])  # Marginalizes over U
ll_xz  = spn.log_likelihood([X_i])       # Marginalizes over U
ll_yz  = spn.log_likelihood([X_j])       # Marginalizes over U
ll_z   = spn.log_likelihood([])          # Constant

# Problem: Marginals include U's influence
# Result: All p-values = 0.000 (everything appears dependent)
```

**This doesn't work well** ❌

---

## Recommended Solution

**Make Main PC use the same approach as Structure Voting**:

```python
# In skeleton_discovery, modify conditioning sets

def skeleton_discovery_with_U(data, alpha, indep_test,
                               c_indx_id=None, ...):
    """
    Modified PC algorithm that conditions on U.
    """

    for depth in range(max_depth):
        for (i, j) in edges:
            # Original: Z = subset of neighbors, size = depth
            Z_original = get_conditioning_set(i, j, depth)

            # Modified: Always include U in conditioning set
            if c_indx_id is not None:
                Z_augmented = Z_original + [c_indx_id]
            else:
                Z_augmented = Z_original

            # Test: X_i ⊥ X_j | (Z ∪ {U})
            p_value = ci_test(i, j, Z_augmented)

            if p_value > alpha:
                remove_edge(i, j)
```

**Expected improvement**:
- P-values will be varied (not all 0.000)
- More accurate independence detection
- Precision: 0.33 → 0.6-0.7
- F1: 0.5 → 0.7-0.8

---

## Summary

### How U is Derived

1. **Creation**: `c_indx = np.repeat(np.arange(K), samples_per_client)`
   - Simple assignment of client IDs to samples
   - Shape: (n, 1)

2. **Augmentation**: `X_aug = np.concatenate([X, c_indx], axis=1)`
   - Append as column 8
   - Shape: (n, d+1)

3. **SPN Training**: SPNs learn `P(X_0, ..., X_7, U)`
   - U is treated as a regular variable
   - SPN learns dependencies involving U

### How U is Obtained from SPN

1. **Direct**: U is column 8 in augmented data (current)
2. **Inference**: Compute `P(U | X)` using Bayes rule (alternative)
3. **Sampling**: Sample from `P(U | X)` for uncertainty (advanced)

### Key Insight

**U is not a "surrogate" in the sense of being inferred** - it's a **known augmentation** (client ID) that:
- Captures domain heterogeneity
- Enables conditional independence testing across domains
- Is directly accessible from the data

The SPN simply learns the joint distribution `P(X, U)` which enables:
- Conditioning on U for CI tests
- Marginalization over U when needed
- Any joint/marginal computation involving U

**The fix**: Make main PC algorithm condition on U (like structure voting does) instead of marginalizing over it.
