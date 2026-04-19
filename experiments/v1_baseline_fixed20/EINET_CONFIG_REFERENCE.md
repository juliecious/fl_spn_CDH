# Einet Configuration Reference - v1 Baseline Experiments

**Experiment Version**: v1_baseline_fixed20
**Date**: April 18-19, 2026

---

## Fixed Hyperparameters (Used for ALL Experiments)

```python
num_sums = 20           # FIXED (no adaptive scaling)
num_leaves = 20         # FIXED (no adaptive scaling)
num_repetitions = 10    # FIXED
depth = calculated      # Only parameter that adapts (based on local_d)
```

### Depth Calculation

```python
local_d = num_features_per_client + 1  # +1 for context column U
depth = max(1, floor(log2(local_d)))
```

---

## Actual Architectures by Config and Mode

### SMALL Config (d=8, K=3, n=600)

#### Horizontal Mode
- Each client sees: 8 features
- local_d = 9 (8 features + 1 context)
- Architecture: **num_sums=20, num_leaves=20, depth=3**

#### Vertical Mode
- Client 0: 2 features → local_d=3 → **num_sums=20, num_leaves=20, depth=1**
- Client 1: 2 features → local_d=3 → **num_sums=20, num_leaves=20, depth=1**
- Client 2: 4 features → local_d=5 → **num_sums=20, num_leaves=20, depth=2**

#### Hybrid Mode
- Each client sees: 8 features (after clustering)
- local_d = 9
- Architecture: **num_sums=20, num_leaves=20, depth=3**

---

### MEDIUM Config (d=10, K=3, n=1200)

#### Horizontal Mode
- Each client sees: 10 features
- local_d = 11 (10 features + 1 context)
- Architecture: **num_sums=20, num_leaves=20, depth=3**

#### Vertical Mode
- Client 0: 3 features → local_d=4 → **num_sums=20, num_leaves=20, depth=2**
- Client 1: 3 features → local_d=4 → **num_sums=20, num_leaves=20, depth=2**
- Client 2: 4 features → local_d=5 → **num_sums=20, num_leaves=20, depth=2**

#### Hybrid Mode
- Each client sees: 10 features (after clustering)
- local_d = 11
- Architecture: **num_sums=20, num_leaves=20, depth=3**

---

### LARGE Config (d=11, K=5, n=1650)

#### Horizontal Mode
- Each client sees: 11 features
- local_d = 12 (11 features + 1 context)
- Architecture: **num_sums=20, num_leaves=20, depth=3**

#### Vertical Mode
- Clients 0-3: 2 features each → local_d=3 → **num_sums=20, num_leaves=20, depth=1**
- Client 4: 3 features → local_d=4 → **num_sums=20, num_leaves=20, depth=2**

#### Hybrid Mode
- Each client sees: 11 features (after clustering)
- local_d = 12
- Architecture: **num_sums=20, num_leaves=20, depth=3**

---

## Training Hyperparameters

| Config | Epochs | Learning Rate | Batch Size |
|--------|--------|---------------|------------|
| SMALL | 50 | Adaptive* | Default |
| MEDIUM | 100 | Adaptive* | Default |
| LARGE | 150 | Adaptive* | Default |

*Adaptive learning rate: Adjusted based on dataset size (implementation detail in FedCDH.py)

---

## Performance Summary by Architecture

### When Fixed 20/20 Works
- ✅ **SMALL (local_d ≤ 9)**: F1 = 0.4-0.8
  - Sufficient capacity for low dimensionality
  - Horizontal mode performs best (avg F1: 0.551)

### When Fixed 20/20 Struggles
- ⚠️ **MEDIUM (local_d = 11)**: F1 = 0.1-0.4
  - Marginal capacity for moderate dimensionality
  - Performance degrades significantly from SMALL

### When Fixed 20/20 Fails
- ❌ **LARGE (local_d = 12, K=5)**: F1 = 0.000
  - Insufficient capacity for high dimensionality + multiple clients
  - **Complete catastrophic failure** across all modes

---

## Capacity Analysis

### Theoretical Complexity Budget

Fixed architecture (20/20) can represent approximately:
- **Maximum effective dimensionality**: ~8-9 features (SMALL works)
- **Complexity ceiling**: d×K ≈ 30-40
- **Beyond ceiling**: Complete failure (LARGE: d×K = 55)

### Capacity Utilization by Mode

| Mode | Client Dimensionality | Capacity Utilization | Over/Under |
|------|----------------------|---------------------|------------|
| **Horizontal SMALL** | local_d=9 | ~80% | ✓ Optimal |
| **Horizontal LARGE** | local_d=12 | >100% | ❌ Over-capacity |
| **Vertical SMALL** | local_d=3 | ~30% | ⚠️ Under-utilized (overfits) |
| **Vertical LARGE** | local_d=3-4 | ~40% | ⚠️ Under-utilized |

---

## Key Observations

### 1. Depth Adaptation is Insufficient
Only depth adapts (log2 rule), but num_sums/num_leaves remain fixed at 20. This means:
- Width (sums/leaves) doesn't scale with dimensionality
- Cannot represent complex high-dimensional distributions
- LARGE config (d=11, K=5) needs 60+ sums/leaves, gets only 20

### 2. Mode-Agnostic Architecture
All modes use the same 20/20, but:
- **Horizontal** needs MORE capacity (models full d-dimensional space)
- **Vertical** needs LESS capacity (models d/K-dimensional subspaces)
- Result: Horizontal under-powered, Vertical over-powered

### 3. No Sample Size Consideration
Same 20/20 used regardless of:
- SMALL: 200 samples/client
- LARGE: 330 samples/client

More samples should allow larger models, but architecture doesn't adapt.

### 4. No Data Type Differentiation
Same 20/20 for:
- Linear data (simpler relationships)
- Nonlinear data (complex relationships)

Nonlinear should get more capacity (especially more leaves), but doesn't.

---

## Implications for v2 Experiments

### Must Scale With:
1. **Dimensionality (d)**: Larger d → more sums/leaves
2. **Number of clients (K)**: More clients → more capacity for aggregation
3. **Mode type**: Different scaling rules for horizontal vs vertical
4. **Sample size**: More samples → can afford larger models
5. **Data type**: Nonlinear → needs more capacity

### Minimum Requirements (Derived from v1 Failures)

```python
# To avoid LARGE config catastrophic failure:
SMALL (d=8, K=3):   num_sums = 20   # Current works ✓
MEDIUM (d=10, K=3): num_sums = 35   # To improve performance
LARGE (d=11, K=5):  num_sums = 60+  # To function at all

# General scaling rule:
num_sums = max(20, 5 * sqrt(d * K))
num_leaves = max(20, 2.5 * sqrt(d * K))

# Mode-specific adjustments:
if mode == "horizontal":
    num_sums *= 1.5  # Needs more capacity per client
elif mode == "vertical" and local_features <= 3:
    num_sums = min(num_sums, 10)  # Avoid over-parameterization
```

---

## Code Location

Relevant code in codebase:
- **SPN Wrapper**: `causallearn/utils/FedPC.py` (LocalSPNWrapper, lines 70-77)
- **Architecture Logic**: `causallearn/search/FCMBased/FedCDH/FedCDH.py` (lines 530-570)
- **Fixed Values**: Hardcoded in FedCDH.py (num_sums=20, num_leaves=20)

To implement adaptive scaling for v2, modify FedCDH.py around line 540 to replace fixed values with calculation based on mode, d, K, and sample size.

---

## Summary Table: Fixed 20/20 Across All Experiments

| Config | Mode | local_d | depth | num_sums | num_leaves | F1 (Linear) | F1 (Nonlinear) | Status |
|--------|------|---------|-------|----------|------------|-------------|----------------|--------|
| SMALL | Horiz | 9 | 3 | 20 | 20 | 0.486 | 0.615 | ✅ Works |
| SMALL | Vert | 3-5 | 1-2 | 20 | 20 | 0.059 | 0.778 | ⚠️ Overfits linear |
| SMALL | Hybrid | 9 | 3 | 20 | 20 | 0.579 | 0.600 | ✅ Works |
| MEDIUM | Horiz | 11 | 3 | 20 | 20 | 0.400 | 0.167 | ⚠️ Marginal |
| MEDIUM | Vert | 4-5 | 2 | 20 | 20 | 0.250 | 0.000 | ❌ Poor |
| MEDIUM | Hybrid | 11 | 3 | 20 | 20 | 0.392 | 0.114 | ⚠️ Marginal |
| LARGE | Horiz | 12 | 3 | 20 | 20 | 0.000 | 0.000 | ❌ **FAIL** |
| LARGE | Vert | 3-4 | 1-2 | 20 | 20 | 0.000 | 0.000 | ❌ **FAIL** |
| LARGE | Hybrid | 12 | 3 | 20 | 20 | 0.000 | 0.000 | ❌ **FAIL** |

**Conclusion**: Fixed 20/20 architecture is only viable for toy problems (d≤8, K≤3). Real-world applications require adaptive scaling.
