# v2 Experiment Quickstart Guide

This guide explains how to run v2 experiments with the new adaptive hyperparameters and CI ranking features.

## Prerequisites

1. Ensure all v2 implementation files are present:
   ```bash
   ls causallearn/utils/ci_ranking.py              # CI ranking tracker
   ls causallearn/utils/FedPC.py | grep compute    # Adaptive hyperparameters
   ```

2. Run smoke tests to verify installation:
   ```bash
   python tests/test_v2_integration_smoke.py
   # Expected: 4/4 tests pass
   ```

## Configuration Options

### 1. Adaptive Hyperparameters (ENABLED BY DEFAULT)

Adaptive hyperparameters are automatically used in v2. No configuration needed!

The system will:
- Detect scenario mode (horizontal/vertical/hybrid)
- Count features and samples per client
- Apply 5-criterion scaling automatically

To specify data type (optional):
```python
args.data_type = "linear"    # or "nonlinear" (default)
```

**Example Log Output:**
```
[Client 0, Cluster 0] Adaptive hyperparameters: d=10, n=400, mode=horizontal, type=linear
  Architecture: sums=20 (base=20), leaves=10 (base=20), depth=3
  Training: epochs=377 (base=100), dropout=0.100, weight_decay=1.0e-04
```

### 2. CI Ranking (OPTIONAL - EXPERIMENTAL)

To enable percentile-based edge selection instead of alpha=0.05:

```python
# In your experiment script
args.use_ci_ranking = True
args.sparsity_percentile = 0.2  # Keep top 20% of edges
```

**Note:** Full CI ranking integration is pending. For v2.0, focus on adaptive hyperparameters.

## Experiment Templates

### Template 1: Quick Validation (MEDIUM Horizontal)

**Goal:** Verify that horizontal F1 improves from 0.255 → 0.5+

```bash
# Run MEDIUM config (d=10, K=3, n=1200 total)
python -c "
import numpy as np
from types import SimpleNamespace
from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH

# Config
args = SimpleNamespace(
    K=3,
    d=10,
    n=400,  # per client
    scenario='horizontal',
    model_type='spn',
    ci_method='spn',
    data_type='linear',
    device='cpu'
)

# Generate synthetic data
np.random.seed(42)
data = np.random.randn(1200, 10)
X_splits = [data[k*400:(k+1)*400, :] for k in range(3)]

# Run FedCDH
fedcdh = FedCDH(args)
result = fedcdh.run_v2(X_splits, true_dag=None)

print(f'F1 Score: {result[\"F1\"]:.3f}')
print(f'Expected: > 0.5 (v1 baseline was 0.255)')
"
```

### Template 2: Full Capacity Sweep

**Goal:** Compare v2 adaptive vs v1 baseline across all scenarios

```bash
# Run all 3 configs × 3 modes × 2 data types = 18 experiments
python experiments/run_capacity_sweep.py \
  --configs SMALL,MEDIUM,LARGE \
  --modes horizontal,vertical,hybrid \
  --data_types linear,nonlinear \
  --output_dir experiments/v2_capacity_sweep/
```

### Template 3: Sparsity Sweep (Ranking Exploration)

**Goal:** Find optimal sparsity_percentile for each scenario

```bash
# Sweep sparsity values
python experiments/run_sparsity_sweep.py \
  --config SMALL \
  --mode horizontal \
  --sparsity_values 0.1,0.2,0.3,0.4,0.5 \
  --output_dir experiments/v2_sparsity_sweep/
```

**Note:** Requires full CI ranking integration (pending).

## Interpreting Results

### Expected Improvements (v2 vs v1)

| Scenario | v1 Baseline F1 | v2 Target F1 | Expected Gain |
|----------|----------------|--------------|---------------|
| MEDIUM Horizontal | 0.255 | 0.5+ | 2× improvement |
| SMALL Horizontal | 0.133 | 0.4+ | 3× improvement |
| MEDIUM Vertical | 0.45 | 0.45-0.5 | Maintain/slight gain |
| MEDIUM Hybrid | 0.35 | 0.4+ | ~15% improvement |

### Key Metrics to Track

1. **F1 Score** (primary)
   - Harmonic mean of precision and recall
   - Target: ≥0.5 for horizontal mode

2. **SHD (Structural Hamming Distance)**
   - Lower is better
   - Measures edge-level errors

3. **Training Time**
   - Expected: 2-3× longer (due to more epochs)
   - Acceptable trade-off for 2-4× F1 gain

4. **Edge Count**
   - With ranking: Explicitly controlled by sparsity_percentile
   - With alpha: Implicit (depends on data)

### Logging Analysis

Look for these log lines to verify adaptive scaling:

```
[Client X, Cluster Y] Adaptive hyperparameters: d=10, n=400, mode=horizontal, type=linear
  Architecture: sums=20, leaves=10, depth=3
  Training: epochs=377, dropout=0.100, weight_decay=1.0e-04
```

**Red Flags:**
- If architecture matches old sqrt scaling (e.g., sums=28 for d=10):
  → Adaptive hyperparameters not being used
- If epochs=100 for horizontal mode:
  → Adaptive epochs not being applied

## Troubleshooting

### Issue 1: No improvement in F1

**Check:**
1. Verify adaptive hyperparameters are logged
2. Check that `data_type` matches actual data (linear vs nonlinear)
3. Increase epochs manually if needed: `args.base_epochs = 200`

### Issue 2: NaN/Inf during training

**Solutions:**
1. Verify gradient clipping is active (should be by default)
2. Reduce learning rate: `args.lr = 0.001`
3. Check data normalization

### Issue 3: Memory errors on large configs

**Solutions:**
1. Use smaller batch size: `args.batch_size = 32`
2. Reduce num_sums/num_leaves: Override with `args.num_sums = 10`
3. Switch to CPU if GPU OOM: `args.device = 'cpu'`

## Next Steps After Validation

1. **If F1 ≥ 0.5 on MEDIUM horizontal:**
   - ✅ Success! Run full capacity sweep
   - Document results in thesis

2. **If F1 < 0.5 but > 0.255:**
   - Partial success - investigate hyperparameter tuning
   - Try increasing epochs or capacity manually
   - Run ablation studies

3. **If F1 ≤ 0.255:**
   - Debug: Check logs for adaptive hyperparameter application
   - Verify code changes are active (smoke tests pass)
   - Review data generation for issues

## References

- **Implementation Summary:** `V2_IMPLEMENTATION_SUMMARY.md`
- **Working State:** `agents/working_state.md`
- **v1 Baseline Results:** `experiments/v1_final_analysis/`
- **Smoke Tests:** `tests/test_v2_integration_smoke.py`

## Quick Commands

```bash
# Verify v2 implementation
python tests/test_v2_integration_smoke.py

# Run single MEDIUM horizontal test
python tests/capacity_validation_test.py

# Check git status for v2 changes
git status

# View recent commits
git log --oneline -5
```

---

**Last Updated:** 2026-04-22
**Status:** ✅ Ready for experiments
