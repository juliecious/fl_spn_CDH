# FedCDH Benchmark Suite

Production-quality benchmark for comparing Horizontal, Vertical, and Hybrid federated learning scenarios.

**Updated**: April 14, 2026 - Reflects Week 2 Mixture-then-Product hybrid implementation

---

## Overview

Compares three SPN aggregation strategies:
- **Horizontal**: Mixture-of-experts (sample partitioning)
- **Vertical**: Product-of-experts (feature partitioning)
- **Hybrid**: Mixture-then-Product ✅ (Algorithm 1 automatic feature grouping)

---

## Quick Start

```bash
# Quick smoke test (5 vars, 2 clients, 200 samples, ~2-3 minutes)
python test_fedcdh_benchmark.py --config quick

# Small-scale benchmark (8 vars, 3 clients, 600 samples, ~5-10 minutes)
python test_fedcdh_benchmark.py --config small

# Medium-scale benchmark (10 vars, 3 clients, 1200 samples, ~15-20 minutes)
python test_fedcdh_benchmark.py --config medium

# Large-scale benchmark (11 vars, 5 clients, 1650 samples, ~30-40 minutes)
python test_fedcdh_benchmark.py --config large

# Sachs-like dimensions (11 vars, 3 clients, 853 samples, ~20-30 minutes)
python test_fedcdh_benchmark.py --config sachs
```

---

## Command-Line Options

### Configuration
```bash
--config {quick,small,medium,large,sachs}
```

| Config | Variables | Clients | Samples | Epochs | Runtime (GPU) | Use Case |
|--------|-----------|---------|---------|--------|---------------|----------|
| `quick` | 5 | 2 | 200 | 20 | ~2-3 min | Smoke test |
| `small` | 8 | 3 | 600 | 50 | ~5-10 min | Development |
| `medium` | 10 | 3 | 1200 | 100 | ~15-20 min | Standard benchmark |
| `large` | 11 | 5 | 1650 | 150 | ~30-40 min | Large-scale evaluation |
| `sachs` | 11 | 3 | 853 | 150 | ~20-30 min | Sachs dataset dimensions |

### Data Type
```bash
--data-type {linear,nonlinear}
```
- `linear`: Linear Gaussian SEM (default)
- `nonlinear`: Nonlinear SEM with heterogeneity

### Device
```bash
--device {cuda,mps,cpu}
```
- Auto-detection by default (CUDA > MPS > CPU)
- `cuda`: NVIDIA GPU (fastest)
- `mps`: Apple Silicon GPU (M1/M2/M3)
- `cpu`: CPU fallback (slowest)

### Seeds
```bash
--seeds 42 123 456
```
- Default: `[42, 123, 456, 789, 2024]` (5 runs)
- Specify custom seeds for reproducibility

---

## Examples

### Basic Usage
```bash
# Quick test with auto-detected device
python test_fedcdh_benchmark.py --config quick

# Medium benchmark with linear data
python test_fedcdh_benchmark.py --config medium --data-type linear

# Large benchmark with nonlinear data
python test_fedcdh_benchmark.py --config large --data-type nonlinear
```

### Device Selection
```bash
# Force CUDA GPU
python test_fedcdh_benchmark.py --config small --device cuda

# Force MPS (Apple Silicon)
python test_fedcdh_benchmark.py --config small --device mps

# Force CPU (testing)
python test_fedcdh_benchmark.py --config quick --device cpu
```

### Custom Seeds
```bash
# Single run with seed 42
python test_fedcdh_benchmark.py --config quick --seeds 42

# 10 runs with custom seeds
python test_fedcdh_benchmark.py --config small --seeds 1 2 3 4 5 6 7 8 9 10
```

---

## Output Structure

### Results Directory: `benchmark_results/`

```
benchmark_results/
├── benchmark_results.csv          # Raw metrics for all runs
├── summary_statistics.csv         # Aggregated statistics (mean ± std)
├── experiment_manifest.txt        # Detailed experiment directory listing
└── benchmark.log                  # Complete execution log
```

### Individual Experiment Directories: `eval/YYYYMMDD_HHMMSS_*/`

Each experiment generates its own timestamped directory in `eval/`:

```
eval/20260414_183045_hybrid_3clients_10vars_1200samples/
├── run.log                        # Detailed training/evaluation log
├── umap_local_client_0.png        # UMAP visualization for local SPN 0
├── umap_local_client_1.png        # UMAP visualization for local SPN 1
├── umap_local_client_2.png        # UMAP visualization for local SPN 2
└── umap_global_spn.png            # UMAP visualization for global SPN
```

**Important**: Use `experiment_manifest.txt` to map results to specific experiment directories.

---

## Metrics Reported

### Skeleton Metrics
- **Skeleton F1**: F1 score for undirected skeleton (edges without orientation)
- **Skeleton Precision**: Precision for skeleton edges
- **Skeleton Recall**: Recall for skeleton edges
- **Skeleton SHD**: Structural Hamming Distance for skeleton

### DAG Metrics
- **DAG F1**: F1 score for fully oriented DAG
- **DAG Precision**: Precision for oriented edges
- **DAG Recall**: Recall for oriented edges
- **DAG SHD**: Structural Hamming Distance for oriented DAG

### Performance Metrics
- **Train Time**: Training time (seconds)
- **Total Time**: Total experiment time including data generation (seconds)
- **Communication Cost**: Federated communication overhead (bytes)

---

## GPU Configuration

### CUDA (NVIDIA GPUs)
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run benchmark on CUDA
python test_fedcdh_benchmark.py --config small --device cuda
```

**Requirements**:
- CUDA-capable GPU
- CUDA toolkit installed
- PyTorch with CUDA support: `torch.cuda.is_available() == True`

### MPS (Apple Silicon)
```bash
# Check MPS availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Run benchmark on MPS
python test_fedcdh_benchmark.py --config small --device mps
```

**Requirements**:
- M1/M2/M3 Mac
- macOS 12.3+ (Monterey or later)
- PyTorch 1.12+ with MPS support

### CPU Fallback
```bash
# Run on CPU (no GPU required)
python test_fedcdh_benchmark.py --config quick --device cpu
```

**Note**: CPU is ~5-10x slower than GPU for SPN training.

---

## Interpreting Results

### Scenario Comparison

**Expected Patterns**:
1. **Horizontal**: Best when data is homogeneous across clients
2. **Vertical**: Best when features are truly independent
3. **Hybrid**: Best when features overlap and data is heterogeneous ✅

**Week 2 Update**: Hybrid mode now uses **Mixture-then-Product** architecture:
- Algorithm 1 automatically groups features by client set
- Mixture captures heterogeneity within feature groups
- Product captures independence across feature groups
- Matches Seng et al. (2025) formulation exactly

### Quality Indicators

**Good SPN Quality**:
- MMD p-value > 0.05 (distribution match)
- KS test < 50% failed (marginal match)
- Train LL > 0 (positive log-likelihood)

**Poor SPN Quality** (insufficient training):
- MMD p-value < 0.01
- KS test > 80% failed
- Train LL < 0 or very negative

**Solution**: Increase `--epochs` or `--config` to larger scale.

---

## Performance Optimization

### For Speed
1. Use `--config quick` for rapid iteration
2. Reduce seeds: `--seeds 42 123` (2 runs instead of 5)
3. Use GPU: `--device cuda` or `--device mps`
4. Use linear data: `--data-type linear` (faster than nonlinear)

### For Accuracy
1. Use `--config large` or `--config sachs`
2. Increase seeds: `--seeds 1 2 3 4 5 6 7 8 9 10` (10 runs)
3. Use nonlinear data: `--data-type nonlinear` (more realistic)

### GPU Memory Management

**If GPU Out-of-Memory**:
1. Reduce configuration: Use `--config small` instead of `large`
2. Monitor GPU memory in logs: Check `[Pre-experiment] GPU Memory` lines
3. Clear cache between runs: Script automatically calls `torch.cuda.empty_cache()`

---

## Troubleshooting

### GPU Not Detected
```bash
# Check PyTorch configuration
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

**Solutions**:
- CUDA: Reinstall PyTorch with CUDA support
- MPS: Update to PyTorch 1.12+ and macOS 12.3+
- CPU: Use `--device cpu` as fallback

### Slow Performance
- GPU not being used: Check device in logs
- Small GPU: Reduce `--config` size
- CPU mode: Expected ~5-10x slower

### Poor Results
- Insufficient epochs: Try `--config large` (150 epochs)
- Insufficient samples: Need ~200+ samples per client
- Wrong scenario: Verify data partitioning matches scenario

---

## Citation

If using this benchmark in research:

```bibtex
@software{fedcdh_benchmark_2026,
  title={FedCDH Benchmark Suite},
  author={FedCDH Team},
  year={2026},
  note={Week 2 Mixture-then-Product Implementation}
}
```

**References**:
- Seng et al. (2025): "Scaling Probabilistic Circuits via Data Partitioning"
- Working State Documentation: `agents/working_state.md`

---

## Recent Updates

**April 14, 2026 - Week 2 Implementation**:
- ✅ Updated hybrid mode to Mixture-then-Product (correct hierarchy)
- ✅ Algorithm 1 automatic feature grouping
- ✅ Support for MPS (Apple Silicon GPU)
- ✅ GPU memory monitoring and warmup
- ✅ Command-line interface with flexible options
- ✅ Experiment manifest tracking all eval directories
- ✅ Comprehensive output structure documentation
