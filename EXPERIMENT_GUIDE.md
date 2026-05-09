# V3 Experiment Guide

**Status**: ✅ CPU verification complete, ready for GPU experiments

**Date**: 2026-05-09

---

## Quick Start

### 1. CPU Smoke Test (Already Verified ✅)

```bash
# Verify all V3 strategies work
python test_aggregation_smoke.py
# Expected: 5/5 tests PASSED
```

### 2. GPU Experiments (Synthetic Data)

```bash
# Run all scenarios with MEDIUM config on GPU
python run_gpu_experiments.py --config MEDIUM --device cuda --num-seeds 5

# Run only horizontal mode (test all 3 strategies)
python run_gpu_experiments.py --scenario horizontal --config MEDIUM --device cuda

# Run only hybrid mode (GlobalSumOfProducts)
python run_gpu_experiments.py --scenario hybrid --config LARGE --device cuda

# For Mac M-series
python run_gpu_experiments.py --device mps --config SMALL
```

**Configurations:**
- `SMALL`: d=8, K=3, n=900, epochs=50 (fast, ~5 min/experiment)
- `MEDIUM`: d=12, K=3, n=1800, epochs=100 (standard, ~15 min/experiment)
- `LARGE`: d=20, K=5, n=3600, epochs=150 (comprehensive, ~45 min/experiment)

### 3. Sachs Real-World Validation

```bash
# Run all scenarios on Sachs protein network
python run_sachs_experiments.py --device cuda --num-runs 5

# Run only horizontal with structure_voting (recommended)
python run_sachs_experiments.py --scenario horizontal --strategy structure_voting --device cuda

# Run with more training epochs for better convergence
python run_sachs_experiments.py --epochs 150 --device cuda

# Quick test on CPU
python run_sachs_experiments.py --device cpu --num-runs 1 --epochs 50
```

**Sachs Dataset:**
- 11 nodes (proteins): Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk
- 7,466 samples (interventional data)
- 17 known edges (ground truth)
- Nonlinear relationships

---

## Experiment Matrix

### Phase 1: V3 Fix Verification ✅ COMPLETE

| Test | Status | Result |
|------|--------|--------|
| Horizontal: structure_voting | ✅ PASS | CPU verified |
| Horizontal: ll_weighted | ✅ PASS | CPU verified |
| Horizontal: mixture | ✅ PASS | CPU verified |
| Hybrid: GlobalSumOfProducts | ✅ PASS | CPU verified |
| Vertical: ProductOverGroups | ✅ PASS | CPU verified |

### Phase 2: GPU Synthetic Experiments (TODO)

**Goal**: Compare V3 strategies on larger data

| Scenario | Strategies | Config | Expected Result |
|----------|-----------|--------|-----------------|
| Horizontal | structure_voting vs ll_weighted vs mixture | MEDIUM | structure_voting F1 > mixture |
| Hybrid | GlobalSumOfProducts | MEDIUM | Cross-group F1 > 0.3 |
| Vertical | ProductOverGroups | MEDIUM | Baseline performance |

**Command:**
```bash
python run_gpu_experiments.py --config MEDIUM --device cuda --num-seeds 5
```

**Expected Runtime**: ~2-3 hours for all scenarios with 5 seeds

### Phase 3: Sachs Real-World Validation (TODO)

**Goal**: Validate on real protein signaling network

| Scenario | Strategy | Target F1 | Critical? |
|----------|----------|-----------|-----------|
| Horizontal | structure_voting | ≥ 0.60 | ✅ CRITICAL |
| Horizontal | ll_weighted | ≥ 0.55 | 🟡 IMPORTANT |
| Horizontal | mixture (baseline) | ~ 0.40 | 🟢 REFERENCE |
| Hybrid | GlobalSumOfProducts | ≥ 0.50 | ✅ CRITICAL |
| Vertical | ProductOverGroups | ≥ 0.55 | 🟡 IMPORTANT |

**Command:**
```bash
python run_sachs_experiments.py --device cuda --num-runs 5
```

**Expected Runtime**: ~4-6 hours for all scenarios with 5 runs

---

## Expected Improvements (V2 vs V3)

### Horizontal Mode
- **V2 (mixture)**: Global F1 = 0.000 ❌
- **V3 (structure_voting)**: Global F1 > 0.3 ✅
- **V3 (ll_weighted)**: Global F1 > 0.25 ✅

**Key Fix**: Democratic voting prevents dependency dilution

### Hybrid Mode
- **V2 (product only)**: Cross-group F1 = 0.000 ❌
- **V3 (sum-over-products)**: Cross-group F1 > 0.3 ✅

**Key Fix**: Sum breaks independence between feature groups

### Vertical Mode
- **V2**: High variance with few features ⚠️
- **V3**: Stable (no changes needed) ✅

---

## Output Files

### GPU Experiments
```
gpu_experiment_results/
  results_all_MEDIUM_20260509_123456.json
  results_horizontal_MEDIUM_20260509_123456.json
  results_hybrid_LARGE_20260509_123456.json
```

### Sachs Experiments
```
sachs_results/
  sachs_results_all_20260509_123456.json
  sachs_results_horizontal_20260509_123456.json
```

### Result Format (JSON)
```json
{
  "scenario": "horizontal",
  "strategy": "structure_voting",
  "f1_skeleton": 0.67,
  "shd": 12,
  "precision": 0.72,
  "recall": 0.63,
  "runtime_seconds": 45.3,
  "true_edges": 17,
  "config": "MEDIUM",
  "seed": 0,
  "timestamp": "2026-05-09T12:34:56"
}
```

---

## Analysis Scripts (TODO - Create if needed)

```bash
# Analyze GPU experiment results
python scripts/analyze_gpu_results.py gpu_experiment_results/

# Compare V2 vs V3 performance
python scripts/compare_v2_v3.py --v2-results v2_results/ --v3-results gpu_experiment_results/

# Generate publication figures
python scripts/generate_figures.py sachs_results/
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller config
python run_gpu_experiments.py --config SMALL --device cuda

# Or reduce batch size in FedCDH.py (if exposed)
```

### Slow Training
```bash
# Reduce epochs for quick testing
python run_sachs_experiments.py --epochs 50 --num-runs 1

# Use CPU for debugging
python run_sachs_experiments.py --device cpu --epochs 20
```

### Import Errors
```bash
# Ensure proper Python environment
conda activate fedcdh_env  # or your environment name

# Verify installation
python -c "from causallearn.search.FCMBased.FedCDH import FedCDH; print('OK')"
```

---

## Next Steps After Experiments

1. **Analyze Results**
   - Compare strategies within each scenario
   - V2 vs V3 comparison tables
   - Identify best strategy per scenario

2. **Generate Thesis Figures**
   - Table 1: Baseline comparison (Centralized vs FedSPN)
   - Table 2: Sachs results (3 scenarios × metrics)
   - Table 3: V2 vs V3 improvements
   - Figure 1: Sachs learned vs ground truth networks
   - Figure 2: Strategy comparison bar charts

3. **Write Results Chapter**
   - Document F1 improvements
   - Explain why structure_voting works
   - Discuss GlobalSumOfProducts effectiveness
   - Report Sachs real-world performance

4. **Run Ablation Studies** (Optional)
   - Sample size ablation (n ∈ [300, 600, 900, 1200, 1800, 2400, 3600])
   - Dimensionality ablation (d ∈ [5, 8, 10, 15, 20, 30])
   - Number of clients ablation (K ∈ [2, 3, 5, 7, 10])

---

## Thesis Roadmap Status

- [x] **Phase 0**: Setup & Implementation (COMPLETE)
- [x] **Phase 1**: V3 Fixes Verification (COMPLETE - CPU smoke tests ✅)
- [ ] **Phase 2**: GPU Experiments (IN PROGRESS - scripts ready)
- [ ] **Phase 3**: Sachs Validation (IN PROGRESS - scripts ready)
- [ ] **Phase 4**: Ablation Studies (PENDING)
- [ ] **Phase 5**: Thesis Writing (PENDING)

**Estimated Time Remaining**: 3-4 weeks

---

## Contact & References

**Documentation**:
- `V3_IMPLEMENTATION_STATUS.md` - Detailed implementation docs
- `V3_QUICK_REFERENCE.md` - Quick reference guide
- `V3_THESIS_CRITICAL_ROADMAP.md` - Complete thesis plan

**Key Files**:
- `test_aggregation_smoke.py` - CPU verification tests
- `run_gpu_experiments.py` - GPU synthetic experiments
- `run_sachs_experiments.py` - Sachs real-world validation
- `causallearn/utils/structure_aggregation.py` - Voting utilities
- `causallearn/utils/FedPC.py` - GlobalSumOfProducts implementation

**Questions?** Check the roadmap or run with `--help` flag.
