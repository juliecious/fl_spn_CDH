# Tests Directory - Minimal Structure

**Last Updated**: March 25, 2026
**Status**: ✅ Cleaned up and ready for thesis experiments

---

## Quick Start

### Smoke Test (30 seconds)
```bash
cd smoke && ./run_all_scenarios.sh
```

### Key Experiment - Vertical Regularization (15 minutes)
```bash
python benchmarks/synthetic_comprehensive_suite.py --exp 4 --seeds 10
```

### Full Thesis Suite (2-3 hours on GPU)
```bash
python benchmarks/synthetic_comprehensive_suite.py --all --seeds 10
```

---

## Directory Structure

```
tests/
├── benchmarks/              Core experiment infrastructure
│   ├── synthetic_comprehensive_suite.py  Main suite (6 experiments, 800 lines)
│   ├── run_experiment.py                 Individual experiment execution
│   ├── configs.py                        Configuration registry
│   └── analyze_results.py                Result aggregation & plotting
│
├── smoke/                   Quick validation
│   ├── test_fedcdh_simple.py            Minimal smoke test (~5s)
│   └── run_all_scenarios.sh             Test H/V/Hy scenarios (~30s)
│
├── utils/                   Data loaders
│   ├── sachs_loader.py                  Real-world Sachs dataset
│   └── benchmark_loaders.py             Synthetic graph generators
│
├── experiments/             Output directory (experiment results)
├── results/                 Documentation (result summaries)
└── data/                    Data files (Sachs and other datasets)
```

---

## Essential Files (8 total)

### 1. synthetic_comprehensive_suite.py (Main Experiment Suite)
**Purpose**: Complete thesis experiment infrastructure
**6 Experiments**:
1. Method Comparison (SPN vs fisherz)
2. Scalability Analysis (N, d, K dimensions)
3. Heterogeneity Robustness (domain shift tolerance)
4. **Scenario Comparison** ⭐ Vertical regularization (THESIS NOVELTY)
5. DAG Structure Robustness (chain/fork/collider/random)
6. Orientation Method Ablation (mi_only vs mi_hybrid)

**Usage**:
```bash
# Single experiment
python benchmarks/synthetic_comprehensive_suite.py --exp 4 --seeds 10

# All experiments
python benchmarks/synthetic_comprehensive_suite.py --all --seeds 10
```

### 2. run_experiment.py (Execution Engine)
**Purpose**: Run individual experiments with specific configs
**Usage**:
```bash
python benchmarks/run_experiment.py \
  --config fedspn_vertical \
  --seed 0 \
  --epochs 50
```

### 3. configs.py (Configuration Registry)
**Purpose**: Centralized experiment configurations
**Key Config**:
```python
BASELINE_CONFIG = {
    "d": 8, "K": 3, "n_per_client": 450,
    "epochs": 100, "num_sums": 20, "num_leaves": 20
}
```

### 4. analyze_results.py (Result Aggregation)
**Purpose**: Aggregate and visualize experiment results
**Usage**:
```bash
python benchmarks/analyze_results.py experiments/ --output results.csv
```

### 5-8. Other Essential Files
- **test_fedcdh_simple.py**: Quick 3-node smoke test
- **run_all_scenarios.sh**: Test H/V/Hy scenarios
- **sachs_loader.py**: Load Sachs protein network (N=856, d=11)
- **benchmark_loaders.py**: Generate synthetic benchmark graphs

---

## Cleanup Summary (March 25, 2026)

**Removed**: 23 files (74% reduction)
- 5 debugging scripts (bugs fixed)
- 6 redundant test scripts
- 5 old/superseded scripts
- 2 unit tests
- 1 old notebook
- 4 directories (unit/, notebooks/, benchmarks/tests/, ...)

**Result**: Clean, minimal test infrastructure focused on thesis experiments

---

## For Thesis Experiments

### Week 3-4: Validation
```bash
# After bug fix, validate it works
./smoke/run_all_scenarios.sh

# Run key novelty experiment
python benchmarks/synthetic_comprehensive_suite.py --exp 4 --seeds 10
```

### Week 4-5: Full Experiments
```bash
# Run complete synthetic suite
python benchmarks/synthetic_comprehensive_suite.py --all --seeds 10

# Real-world validation on Sachs (10 seeds)
for seed in {0..9}; do
  python benchmarks/run_experiment.py --config fedspn_vertical --seed $seed
done
```

### Week 6: Analysis
```bash
# Aggregate all results
python benchmarks/analyze_results.py experiments/ --output thesis_results.csv

# Plots automatically generated in experiments/plots/
```

---

## Preserved Data

- **experiments/**: All historical experiment results (for comparison)
- **results/**: Test result documentation (TEST_RESULTS_FINAL_20260306.md, etc.)
- **data/**: Dataset files (Sachs and other benchmark datasets)

---

## Notes

- All debugging scripts removed (bugs fixed on March 24)
- Unit tests removed (not needed for thesis experiments)
- Old notebooks removed (superseded by current suite)
- FedCDH_Colab_Template.ipynb available at project root for GPU deployment

---

*Minimal, thesis-focused test infrastructure*
*See agents/working_state.md for full cleanup history*
