# v2 Benchmark Guide - Adaptive Hyperparameters

## Overview

The updated `test_fedcdh_benchmark.py` script now supports v2 experiments with:
- ✅ **5-Criterion Adaptive Hyperparameters** (automatic mode-aware capacity scaling)
- ✅ **Consistent Sample Sizes** (fair comparison across horizontal/vertical/hybrid)
- ✅ **Optional CI Ranking** (experimental percentile-based edge selection)
- ✅ **GPU Optimization** (CUDA support with memory tracking)

## Key Changes from v1

### 1. Sample Size Consistency

**Problem:** v1 used inconsistent sample sizes across modes, making comparison unfair.

**Solution:** All modes now use `n_total` samples, but distributed differently:

| Mode       | Total Samples | Per-Client Samples | Per-Client Features | Training Data per Client |
|------------|---------------|-------------------|---------------------|--------------------------|
| Horizontal | n_total       | n_total/K         | d (all)            | (n_total/K) × d          |
| Vertical   | n_total       | n_total (all)     | d/K (split)        | n_total × (d/K)          |
| Hybrid     | n_total       | n_total/K         | d (overlapping)    | (n_total/K) × d          |

**Example (MEDIUM config: n_total=1200, K=3, d=10):**
- **Horizontal:** 3 clients × 400 samples × 10 features = 4000 data points each
- **Vertical:** 3 clients × 1200 samples × 3-4 features = 3600-4800 data points each
- **Hybrid:** 3 clients × 400 samples × 10 features = 4000 data points each

### 2. Adaptive Hyperparameters (Automatic)

All experiments now automatically use the 5-criterion adaptive system:

**Criterion 1: Mode-Specific Base Capacity**
- Horizontal: 4×d sums, 2×d leaves (broad feature space)
- Vertical: 8×d sums, 4×d leaves (depth-focused)
- Hybrid: 6×d sums, 3×d leaves (intermediate)

**Criterion 2: Sample-to-Feature Ratio Scaling**
- Automatically adjusts capacity based on data availability
- Prevents overfitting on low-ratio scenarios

**Criterion 3: Data Type Differentiation**
- Linear: Standard depth and epochs
- Nonlinear: +1 depth, 1.3× more epochs

**Criterion 4: Quality-Aware Epoch Scheduling**
- Horizontal: More epochs (broader learning task)
- Vertical: Fewer epochs (focused learning)

**Criterion 5: Mode-Aware Regularization**
- Dropout and weight_decay tuned per scenario
- Prevents overfitting while preserving capacity

### 3. Updated Configurations

Sample sizes adjusted for better sample-to-feature ratios:

```python
BENCHMARK_CONFIGS = {
    "quick": {
        "d": 5,
        "K": 2,
        "n_total": 200,         # Ratio: 40 (200/5)
        "epochs": 20,
    },
    "small": {
        "d": 8,
        "K": 3,
        "n_total": 900,         # Ratio: 112.5 (900/8) ✅ Better than 75
        "epochs": 50,
    },
    "medium": {
        "d": 10,
        "K": 3,
        "n_total": 1200,        # Ratio: 120 (1200/10)
        "epochs": 100,
    },
    "large": {
        "d": 11,
        "K": 5,
        "n_total": 2000,        # Ratio: 182 (2000/11) ✅ Better than 150
        "epochs": 150,
    },
}
```

**Rationale for changes:**
- SMALL: 600→900 (ratio 75→112.5, crosses into capacity_scale=0.75)
- LARGE: 1650→2000 (ratio 150→182, cleaner division by K=5)

## Usage

### Basic Usage (v2 with Adaptive Hyperparameters)

```bash
# Quick test on GPU
python tests/test/test_fedcdh_benchmark.py --config quick --data-type linear --device cuda

# MEDIUM config on GPU (recommended for v2 validation)
python tests/test/test_fedcdh_benchmark.py --config medium --data-type linear --device cuda

# LARGE config (full capacity test)
python tests/test/test_fedcdh_benchmark.py --config large --data-type nonlinear --device cuda
```

### Advanced: CI Ranking (Experimental)

```bash
# Enable CI ranking with default sparsity (top 20%)
python tests/test/test_fedcdh_benchmark.py \
  --config medium \
  --data-type linear \
  --device cuda \
  --use-ci-ranking \
  --sparsity-percentile 0.2

# Sparsity sweep
for sparsity in 0.1 0.2 0.3 0.4 0.5; do
  python tests/test/test_fedcdh_benchmark.py \
    --config small \
    --data-type linear \
    --device cuda \
    --use-ci-ranking \
    --sparsity-percentile $sparsity
done
```

### Custom Seeds

```bash
# Single run with specific seed
python tests/test/test_fedcdh_benchmark.py --config medium --seeds 42

# Three runs for faster testing
python tests/test/test_fedcdh_benchmark.py --config medium --seeds 42 123 456
```

## Expected Results

### Horizontal Mode Improvement (MEDIUM config)

**v1 Baseline (sqrt scaling):**
- Architecture: ~28 sums (20 × sqrt(10/5))
- Epochs: 100
- F1: 0.133-0.255

**v2 Adaptive:**
- Architecture: 20 sums (4×10 × 0.5 ratio scaling)
- Epochs: 377 (adaptive for horizontal)
- Dropout: 0.1 (ratio 120 > 100, but horizontal still benefits)
- **Expected F1: 0.5+ (2× improvement)**

### Vertical/Hybrid Modes

- Should maintain or slightly improve performance
- Better capacity utilization prevents waste
- No regression expected

## Output Files

The benchmark creates several output files:

```
benchmark_results/
├── benchmark.log                  # Complete execution log
├── benchmark_results.csv          # Raw metrics for all runs
├── summary_statistics.csv         # Aggregated statistics
└── experiment_manifest.txt        # Directory listing with all parameters

eval/
└── [timestamp]_[scenario]_[config]/
    ├── run.log                    # Detailed training/evaluation log
    ├── umap_local_client_0.png    # UMAP visualization (client 0)
    ├── umap_local_client_1.png    # UMAP visualization (client 1)
    ├── umap_local_client_2.png    # UMAP visualization (client 2)
    └── umap_global_spn.png        # Global SPN UMAP visualization
```

## Interpreting Results

### Key Metrics

1. **Skeleton F1** (primary metric for v2)
   - Measures edge detection accuracy (ignoring orientation)
   - Target: ≥0.5 for horizontal mode
   - Baseline: 0.133-0.255 (v1)

2. **DAG F1** (includes orientation)
   - Harder metric (orientation is challenging)
   - Useful for comparing scenarios

3. **SHD (Structural Hamming Distance)**
   - Lower is better
   - Counts total edge errors

4. **Train Time**
   - Expected: 2-3× longer with v2 (more epochs)
   - Trade-off: Acceptable for 2-4× F1 gain

### Statistical Significance

The benchmark runs 5 seeds by default (42, 123, 456, 789, 2024) to provide:
- Mean and std for each metric
- Wilcoxon signed-rank test can compare v1 vs v2

### Red Flags

If results don't improve:
1. Check logs for "Adaptive hyperparameters" messages
2. Verify `data_type` matches data generation (linear/nonlinear)
3. Look for NaN/Inf in training (gradient issues)
4. Check GPU memory (may need smaller batch size)

## Recommended Experiments

### 1. Quick Validation (30 minutes)

```bash
# Verify v2 works on all modes
python tests/test/test_fedcdh_benchmark.py --config quick --data-type linear --device cuda
```

### 2. Horizontal Fix Validation (2-3 hours)

```bash
# Focus on horizontal mode improvement
python tests/test/test_fedcdh_benchmark.py --config medium --data-type linear --device cuda
# Expected: Horizontal F1 from 0.255 → 0.5+
```

### 3. Full Capacity Sweep (6-8 hours)

```bash
# Test all 3 configs × 2 data types = 6 experiments
for config in small medium large; do
  for dtype in linear nonlinear; do
    python tests/test/test_fedcdh_benchmark.py \
      --config $config \
      --data-type $dtype \
      --device cuda
  done
done
```

### 4. Sparsity Exploration (4-6 hours)

```bash
# CI ranking experiments (SMALL config for speed)
for sparsity in 0.1 0.2 0.3 0.4 0.5; do
  python tests/test/test_fedcdh_benchmark.py \
    --config small \
    --data-type linear \
    --device cuda \
    --use-ci-ranking \
    --sparsity-percentile $sparsity
done
```

## Troubleshooting

### GPU Out of Memory

```bash
# Use smaller config or CPU
python tests/test/test_fedcdh_benchmark.py --config small --device cpu

# Or reduce batch size (requires code modification in FedCDH.py)
```

### Slow Training

```bash
# Use fewer seeds
python tests/test/test_fedcdh_benchmark.py --config medium --seeds 42

# Or use smaller config
python tests/test/test_fedcdh_benchmark.py --config small --device cuda
```

### No Improvement in F1

1. **Check adaptive hyperparameters are active:**
   ```bash
   grep "Adaptive hyperparameters" benchmark_results/benchmark.log
   ```

2. **Verify correct data_type:**
   - Linear data needs `--data-type linear`
   - Nonlinear data needs `--data-type nonlinear`

3. **Inspect individual experiment logs:**
   ```bash
   # Find experiment directory from manifest
   cat benchmark_results/experiment_manifest.txt
   # Check detailed log
   cat eval/[experiment_dir]/run.log
   ```

## References

- **Implementation:** `V2_IMPLEMENTATION_SUMMARY.md`
- **Quickstart:** `EXPERIMENT_QUICKSTART.md`
- **Working State:** `agents/working_state.md`

---

**Last Updated:** 2026-04-22
**Status:** ✅ Ready for v2 experiments on GPU
