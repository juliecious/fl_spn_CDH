# V3 Experiment Quick Start

**One unified script**: `tests/test/test_fedcdh_benchmark.py`

---

## ✅ Status

- CPU Verification: ✅ COMPLETE (5/5 tests passing)
- V3 Integration: ✅ COMPLETE (unified benchmark script)
- Sachs Dataset: ✅ INTEGRATED (full 7,466 samples)
- Ready for GPU: ✅ YES

---

## Quick Commands

### Smoke Test (2-3 minutes)
```bash
python tests/test/test_fedcdh_benchmark.py --config quick --device cuda --skip-eval
```

### Sachs Real-World (30-45 minutes)
```bash
python tests/test/test_fedcdh_benchmark.py --config sachs --device cuda --seeds 42
```

### Compare All 3 Horizontal Strategies (1-2 hours)
```bash
python tests/test/test_fedcdh_benchmark.py --config medium --test-all-horizontal-strategies
```

### Full Benchmark (4-6 hours)
```bash
python tests/test/test_fedcdh_benchmark.py --config medium --device cuda
```

---

## V3 Features

### Horizontal Aggregation (3 strategies)

**Default: structure_voting** (recommended V3 fix)
```bash
# Uses structure_voting by default
python tests/test/test_fedcdh_benchmark.py --config medium
```

**Test specific strategy:**
```bash
# V3 alternative (quality-weighted)
python tests/test/test_fedcdh_benchmark.py --config medium --horizontal-aggregation ll_weighted

# V2 baseline (for comparison)
python tests/test/test_fedcdh_benchmark.py --config medium --horizontal-aggregation mixture
```

**Test all 3 strategies:**
```bash
python tests/test/test_fedcdh_benchmark.py --config medium --test-all-horizontal-strategies
```

### Hybrid Mode
- GlobalSumOfProducts enabled automatically (V3 Fix #1)
- No additional flags needed

### Vertical Mode
- ProductOverGroups (unchanged from V2)
- No additional flags needed

---

## Configurations

| Config | d | K | n_total | Epochs | Runtime (GPU) | Use Case |
|--------|---|---|---------|--------|---------------|----------|
| `quick` | 5 | 2 | 200 | 20 | ~2 min | Smoke test |
| `small` | 8 | 3 | 900 | 50 | ~10 min | Quick validation |
| `medium` | 10 | 3 | 1200 | 100 | ~30 min | Standard |
| `large` | 11 | 5 | 2000 | 150 | ~90 min | Comprehensive |
| `sachs` | 11 | 3 | 7466 | 150 | ~45 min | Real-world |

---

## Common Options

```bash
--config {quick,small,medium,large,sachs}     # Configuration size
--data-type {linear,nonlinear}                # Synthetic data type
--device {cuda,mps,cpu}                       # Hardware device
--seeds 42 123 456                            # Multiple random seeds
--skip-eval                                   # Skip UMAP/dashboards (faster)
--horizontal-aggregation {structure_voting,ll_weighted,mixture}  # V3
--test-all-horizontal-strategies              # V3: test all 3
```

---

## Output

Results saved to `benchmark_results/`:
- `benchmark_results.csv` - All runs with metrics
- `summary_statistics.csv` - Aggregated statistics
- `experiment_manifest.txt` - Directory listing
- `benchmark.log` - Complete execution log

Each experiment also creates:
- `eval/YYYYMMDD_HHMMSS_*/run.log` - Detailed logs
- `eval/YYYYMMDD_HHMMSS_*/umap_*.png` - Visualizations (unless --skip-eval)

---

## Expected Results (V2 vs V3)

### Horizontal Mode

| Strategy | V2 Global F1 | V3 Global F1 | Improvement |
|----------|--------------|--------------|-------------|
| mixture (baseline) | 0.000 | 0.000 | No change ❌ |
| ll_weighted | N/A | >0.25 | New ✅ |
| structure_voting | N/A | >0.30 | **Best** ✅ |

### Hybrid Mode

| Metric | V2 | V3 (GlobalSumOfProducts) | Improvement |
|--------|----|--------------------------|------------|
| Cross-group F1 | 0.000 | >0.30 | Fixed ✅ |
| Dense-local F1 | ~1.0 | ~1.0 | Maintained ✅ |

### Sachs Target

| Scenario | Strategy | Target F1 |
|----------|----------|-----------|
| Horizontal | structure_voting | ≥ 0.60 |
| Hybrid | GlobalSumOfProducts | ≥ 0.50 |
| Vertical | ProductOverGroups | ≥ 0.55 |

---

## Help

```bash
python tests/test/test_fedcdh_benchmark.py --help
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller config
python tests/test/test_fedcdh_benchmark.py --config small --device cuda
```

### Mac M-Series
```bash
python tests/test/test_fedcdh_benchmark.py --config small --device mps
```

### Fast Smoke Test
```bash
python tests/test/test_fedcdh_benchmark.py --config quick --skip-eval --seeds 42
```

---

## V2 vs V3 Comparison

To compare V2 (mixture) vs V3 (structure_voting):

```bash
# V2 baseline
python tests/test/test_fedcdh_benchmark.py --config medium \
    --horizontal-aggregation mixture

# V3 fix
python tests/test/test_fedcdh_benchmark.py --config medium \
    --horizontal-aggregation structure_voting

# Or test both at once
python tests/test/test_fedcdh_benchmark.py --config medium \
    --test-all-horizontal-strategies
```

---

**Next**: Run your experiments, analyze results, write thesis! 🎓
