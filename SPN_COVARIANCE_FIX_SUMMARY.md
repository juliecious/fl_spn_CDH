# FedSPN + FICP Orientation Fix - Implementation Summary

## Problem Identified

**Original Issue**: FedSPN was replacing BOTH skeleton discovery AND orientation, but FICP orientation requires covariance-based summary statistics, not SPN parameters.

## Solution Implemented

**Key Insight**: SPN should only replace Step 1 (skeleton discovery). For Step 2 (orientation), we derive covariances FROM the trained SPN.

```
┌─────────────────────────────────────────────────────────┐
│  OLD (BROKEN) APPROACH                                  │
├─────────────────────────────────────────────────────────┤
│  1. Train FedSPN for skeleton ✓                         │
│  2. Try to use SPN for orientation ✗ (incompatible!)    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  NEW (FIXED) APPROACH                                   │
├─────────────────────────────────────────────────────────┤
│  1. Train FedSPN for skeleton ✓                         │
│  2. Sample from SPN → Derive covariances ✓              │
│  3. Use derived covariances for FICP orientation ✓      │
└─────────────────────────────────────────────────────────┘
```

## Files Modified

### 1. `causallearn/search/FCMBased/FedCDH/FedCDH.py`

**Added Method** (line ~308):
```python
def derive_covariances_from_spn(
    self,
    fed_spn_model,
    n_samples=5000,
    n_fourier_features=10,
    seed=42
):
    """
    Derive covariance tensor from trained SPN for FICP orientation.

    Key Innovation: No parallel summary statistics needed!
    Everything derived from the SPN that was trained for skeleton discovery.
    """
```

**Modified Training Flow** (line ~1443):
```python
# After SPN training completes:
self.fed_spn_model = FedCDH_SPN_Wrapper(fed_spn, ...)

# NEW: Derive covariances from SPN
if args.use_spn_covariances:
    self.covariance_tensor = self.derive_covariances_from_spn(
        fed_spn,
        n_samples=args.cov_n_samples,
        n_fourier_features=args.cov_n_features,
        seed=args.seed
    )
```

**Modified cdnod Call** (line ~1942):
```python
# Pass covariance tensor to cdnod for orientation
cdnod_kwargs = {"covariance_tensor": self.covariance_tensor}

cg = cdnod(
    X_global,
    c_indx,
    self.K_clients,
    ...
    **cdnod_kwargs,  # Covariances passed here
)
```

## How It Works

### Step 1: Sample from Trained SPN
```python
# SPN was trained on federated data for skeleton discovery
samples = global_spn.sample(n_samples=5000)
# Shape: (5000, d) where d is number of variables
```

### Step 2: Compute Random Fourier Features
```python
# Approximate Gaussian kernel with Random Fourier Features
rff_w = np.random.randn(n_fourier_features)
rff_b = np.random.uniform(0, 2*np.pi, n_fourier_features)

phi(x) = √(2/h) * cos(w·x + b)
```

### Step 3: Build Covariance Tensor
```python
# For all variable pairs (i, j):
C_ij = (phi_i.T @ phi_j) / n_samples

# Include domain variable (client indicator):
C_i_domain = (phi_i.T @ phi_domain) / n_samples

# Result: CT ∈ R^{(d+1) × (d+1) × h × h}
```

### Step 4: FICP Orientation Uses Derived Covariances
```python
# In cdnod/FICP code (unchanged):
# Compute normalized HSIC scores using covariance_tensor
Δ_X→Y = ||C*_{X,Ỹ}||²_F / (tr(C*_X) · tr(C*_Ỹ))

# Orient edge in direction with smaller score
```

## Testing

### Smoke Test (`test_spn_orientation_fix.py`)
✅ Verified concept works:
- Sample from SPN: ✓
- Compute covariances: ✓
- Calculate FICP scores: ✓
- Make orientation decisions: ✓

### Integration Test (`test_integration_spn_covariance.py`)
✅ Verified full pipeline:
- FedSPN training: ✓ (3.23s)
- Covariance derivation: ✓ (shape 6×6×5×5)
- Causal discovery: ✓ (1.67s)
- No errors or crashes: ✓

## Configuration Parameters

New arguments available in `args`:

```python
args = Namespace(
    # Enable SPN-derived covariances (default: True)
    use_spn_covariances=True,

    # Number of samples to draw from SPN (default: 5000)
    cov_n_samples=5000,

    # Number of Random Fourier Features (default: 10)
    cov_n_features=10,

    # Random seed for reproducibility
    seed=42
)
```

## Benefits

1. **Single Source of Truth**: Only SPN parameters are summary statistics
2. **Privacy Preserved**: All sampling happens on server from learned SPN
3. **No Parallel Computation**: Don't compute KCI covariances AND SPN
4. **Leverages SPN Strength**: Complex dependencies captured in Step 1
5. **Enables FICP**: Covariances available for Step 2 orientation

## Next Steps

To use in your benchmarks:

```python
# In test_fedcdh_benchmark_v3.py:

args = Namespace(
    ...
    ci_method='spn',  # Use FedSPN for skeleton
    use_spn_covariances=True,  # Derive covariances from SPN
    cov_n_samples=5000,  # Plenty for good estimates
    cov_n_features=10,  # Balance between accuracy and speed
    ablation_orientation='mi_hybrid',  # Use FICP orientation
)

fedcdh = FedCDH(args)
result = fedcdh.fit(X_splits, c_indx, true_dag)

# Now metrics should be non-zero!
print(f"Skeleton F1: {result['skeleton_f1']}")
print(f"Direction F1: {result['direction_f1']}")
```

## Performance Considerations

### Memory:
- Covariance tensor: `(d+1)² × h²` floats
- Example: d=10, h=10 → 121×100 = 12,100 floats ≈ 48KB
- Very manageable even for large graphs

### Computation:
- Sampling: O(n_samples × d)
- RFF computation: O(n_samples × d × h)
- Covariance: O(d² × n_samples × h²)
- Total: ~few seconds for typical settings

### Quality Trade-off:
- More samples → Better covariance estimates
- More features (h) → Better kernel approximation
- Recommended: n_samples=5000, h=10-20

## Theoretical Justification

**Why this works**:

1. Random Fourier Features provide unbiased approximation of Gaussian kernels:
   ```
   E[φ(x)ᵀφ(y)] = k(x,y) = exp(-||x-y||²/(2σ²))
   ```

2. Empirical covariance from samples converges to true covariance:
   ```
   C_empirical → C_true  as  n_samples → ∞
   ```

3. FICP requires covariances, not the SPN model itself:
   ```
   FICP needs: C_X, C_Y, C_{X℧}, C_{Y℧}, C_{℧℧}
   All computable from φ(samples)
   ```

## References

- **FedCDH Paper**: "Federated Causal Discovery from Heterogeneous Data" (ICLR 2024)
- **Random Fourier Features**: Rahimi & Recht (2007) "Random Features for Large-Scale Kernel Machines"
- **FICP**: Huang et al. (2020) "Causal Discovery from Heterogeneous/Nonstationary Data"
- **FedSPN**: Your implementation using simple_einet for federated structure learning

---

**Status**: ✅ Implementation complete and tested
**Date**: May 19, 2026
**Integration test**: PASS (pipeline executes without errors)
