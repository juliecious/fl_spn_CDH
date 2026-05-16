# Benchmark Datasets for Causal Discovery

This document describes all available benchmark datasets for testing FedCDH and other causal discovery algorithms.

**Last Updated**: May 13, 2026

---

## 📊 Available Benchmarks

| Dataset | Nodes | Edges | Density | Type | Difficulty | Status |
|---------|-------|-------|---------|------|------------|--------|
| **Sachs** | 11 | 17 | 0.155 | Real biological | Hard | ✅ Available |
| **Law School** | 5 | 7 | 0.350 | Synthetic fairness | Easy | ✅ Available |
| **DREAM4 Net1-5** | 10 | 13-14 | 0.144-0.156 | Gene regulatory | Medium | ✅ Available |
| **Asia** | 8 | 8 | 0.143 | Bayesian network | Easy-Medium | ✅ Available |
| **Alarm** | 37 | 35 | 0.026 | Medical monitoring | Medium-Hard | ✅ Available |

---

## 🧬 DREAM4 Networks (Gene Regulatory Networks)

### Overview
- **Source**: DREAM4 Challenge (2009)
- **Networks**: 5 different 10-node networks with varied topologies
- **Type**: In silico gene regulatory networks
- **Reference**: Marbach et al. (2009), *Journal of Computational Biology*

### Network Characteristics

| Network | Nodes | Edges | Avg Degree | Max In-Degree | Max Out-Degree |
|---------|-------|-------|------------|---------------|----------------|
| Net 1 | 10 | 13 | 1.30 | 2 | 2 |
| Net 2 | 10 | 14 | 1.40 | 2 | 2 |
| Net 3 | 10 | 14 | 1.40 | 3 | 2 |
| Net 4 | 10 | 14 | 1.40 | 2 | 3 |
| Net 5 | 10 | 14 | 1.40 | 3 | 2 |

### Usage

```python
from tests.utils.dream_loader import generate_dream4_data, load_dream4_network

# Load network structure only
B, feature_names = load_dream4_network(network_id=1)
print(f"Network has {len(feature_names)} nodes and {int(B.sum())} edges")

# Generate synthetic data
X, B, feature_names = generate_dream4_data(
    network_id=1,      # Choose network 1-5
    n_samples=1000,    # Number of samples
    mode="linear",     # "linear" or "nonlinear"
    seed=42
)
```

### Why Use DREAM4?

✅ **Multiple topologies**: 5 different networks test robustness
✅ **Standard benchmark**: Widely cited, easy comparison with literature
✅ **Medium complexity**: 10 nodes is similar to Sachs (11 nodes)
✅ **Varied structures**: Different in/out-degree distributions

---

## 🏥 Asia Network (Medical Diagnosis)

### Overview
- **Source**: Lauritzen & Spiegelhalter (1988)
- **Nodes**: 8 (medical conditions and symptoms)
- **Edges**: 8
- **Type**: Classic Bayesian network
- **Domain**: Tuberculosis and lung cancer diagnosis

### Variables
1. **Asia**: Visit to Asia (risk factor for TB)
2. **Smoke**: Smoking status
3. **Tub**: Tuberculosis diagnosis
4. **Lung**: Lung cancer diagnosis
5. **Bronc**: Bronchitis diagnosis
6. **Either**: TB or Lung cancer (logical OR)
7. **Xray**: X-ray result
8. **Dysp**: Dyspnoea (shortness of breath)

### Structure
```
     Asia → Tub ↘
                  Either → Xray
     Smoke → Lung ↗    ↓
        ↓              Dysp
        Bronc ────────→ ↑
```

### Usage

```python
from tests.utils.bayesian_network_loaders import generate_asia_data

# Generate data
X, B, feature_names = generate_asia_data(
    n_samples=1000,
    mode="linear",  # "linear" or "nonlinear"
    seed=42
)

print(f"Variables: {feature_names}")
# Output: ['Asia', 'Smoke', 'Tub', 'Lung', 'Bronc', 'Either', 'Xray', 'Dysp']
```

### Why Use Asia?

✅ **Simple structure**: 8 nodes, easy to visualize
✅ **Classic benchmark**: Used for decades in Bayesian network literature
✅ **Interpretable**: Medical domain with clear semantics
✅ **Logical node**: "Either" variable tests handling of deterministic relationships

---

## 🚨 Alarm Network (Medical Monitoring)

### Overview
- **Source**: Beinlich et al. (1989)
- **Nodes**: 37 (physiological measurements, diseases, equipment)
- **Edges**: 35
- **Type**: Large Bayesian network
- **Domain**: Anesthesia monitoring system

### Network Characteristics
- **Size**: 37 variables
- **Edges**: 35 directed edges
- **Density**: 0.026 (sparse)
- **Complexity**: Medium-Hard (large scale, but sparse)

### Usage

```python
from tests.utils.bayesian_network_loaders import generate_alarm_data

# Generate data
X, B, feature_names = generate_alarm_data(
    n_samples=1000,
    mode="linear",
    seed=42
)

print(f"Network size: {X.shape[1]} nodes, {int(B.sum())} edges")
# Output: Network size: 37 nodes, 35 edges
```

### Why Use Alarm?

✅ **Larger scale**: 37 nodes tests scalability
✅ **Sparse network**: Low density (2.6%) like real-world networks
✅ **Standard benchmark**: Widely used in Bayesian network research
✅ **Medical domain**: Realistic application scenario

---

## 🧪 Sachs Protein Network (Real Biological Data)

### Overview
- **Source**: Sachs et al. (2005), *Science*
- **Nodes**: 11 proteins
- **Edges**: 17 known causal relationships
- **Type**: Real experimental data (flow cytometry)
- **Domain**: Cell signaling pathways

### Usage

```python
from tests.utils.sachs_loader import load_sachs_data

# Load real data
X, B, feature_names = load_sachs_data()
print(f"Sachs: {X.shape[0]} samples, {X.shape[1]} proteins, {int(B.sum())} edges")
```

### Why Use Sachs?

✅ **Real biological data**: Gold standard in causal discovery
✅ **Nonlinear relationships**: Protein interactions are complex
✅ **Widely cited**: Hundreds of papers use this benchmark
✅ **Challenging**: F1 ≈ 0.44 for centralized methods (vs 1.0 on Law School)

**Note**: This is your main benchmark. Focus thesis results on Sachs.

---

## ⚖️ Law School Admissions (Fairness Dataset)

### Overview
- **Source**: Kusner et al. (2017), "Counterfactual Fairness"
- **Nodes**: 5 (race, test scores, admissions outcome)
- **Edges**: 7
- **Type**: Synthetic data with known causal structure
- **Domain**: Educational fairness

### Usage

```python
from tests.utils.law_school_loader import load_law_school_federated

# Generate data
X, B, feature_names = load_law_school_federated(
    n_clients=3,
    n_samples_limit=21000
)
```

### Why Use Law School?

✅ **Sanity check**: Perfect centralized performance (F1=1.0) validates implementation
✅ **Large sample**: 21,000 samples provides high statistical power
✅ **Privacy analysis**: Any F1 drop is purely due to federation (not data insufficiency)

**Warning**: Too easy for main results. Use as validation only.

---

## 📋 Comparison Summary

### By Difficulty (for Causal Discovery)

| Dataset | Difficulty | Why |
|---------|-----------|-----|
| **Law School** | Easy | 21K samples, 5 vars, linear, simple structure |
| **Asia** | Easy-Medium | 8 nodes, simple topology, pedagogical |
| **DREAM4** | Medium | 10 nodes, varied topologies, gene regulation |
| **Alarm** | Medium-Hard | 37 nodes, sparse, large scale |
| **Sachs** | Hard | Real data, nonlinear, complex biology, only 5.4K samples |

### By Sample Size

| Dataset | Typical Samples | Samples per Variable |
|---------|----------------|----------------------|
| **Law School** | 21,000 | 4,200 |
| **DREAM4** | 500-1,000 | 50-100 |
| **Sachs** | 5,400 | 491 |
| **Asia** | 500-1,000 | 62-125 |
| **Alarm** | 1,000-5,000 | 27-135 |

### By Domain

- **Biological**: Sachs, DREAM4
- **Medical**: Asia, Alarm
- **Social Science**: Law School

---

## 🎯 Recommendations for Your Thesis

### Minimum Viable Set (3 datasets)
1. ✅ **Sachs** - Main result (real, challenging)
2. ✅ **Law School** - Sanity check
3. ⭐ **DREAM4** - Robustness (5 networks)

### Strong Set (5+ datasets)
1. ✅ **Sachs** - Main biological benchmark
2. ✅ **Law School** - Upper bound / sanity check
3. ⭐ **DREAM4 (all 5)** - Robustness across topologies
4. ⭐ **Asia** - Simple classical benchmark
5. ⭐ **Alarm** - Large-scale benchmark

### Excellent Set (7+ datasets)
- All of the above +
- Synthetic linear/nonlinear (you already have)
- Additional real dataset (e.g., download DREAM5 E. coli)

---

## 🚀 Quick Start

### 1. Test All Benchmarks

```bash
cd /Users/M279402/PycharmProjects/fl_spn_CDH
python tests/test_benchmark_datasets.py
```

This will:
- Load all 5 DREAM4 networks
- Generate Asia and Alarm data
- Print comprehensive statistics
- Create example CSV files in `data/benchmarks/`

### 2. Use in Your Experiments

```python
# Example: Run FedCDH on DREAM4 Network 3
from tests.utils.dream_loader import generate_dream4_data

# Generate data
X, B_true, names = generate_dream4_data(
    network_id=3,
    n_samples=1000,
    mode="linear",
    seed=42
)

# Run your causal discovery method
from causallearn.search.FCMBased.FedCDH.FedCDH import FedCDH

# ... your experiment code ...
```

### 3. Generate Pre-made Datasets

Example datasets are automatically created at:
```
data/benchmarks/
├── dream4_net1_linear_n1000.csv
├── asia_linear_n1000.csv
└── alarm_linear_n1000.csv
```

---

## 📚 References

### Papers to Cite

**DREAM4**:
```
Marbach, D., Prill, R. J., Schaffter, T., Mattiussi, C., Floreano, D., & Stolovitzky, G. (2009).
Generating realistic in silico gene networks for performance assessment of reverse engineering methods.
Journal of Computational Biology, 16(2), 229-239.
```

**Asia**:
```
Lauritzen, S. L., & Spiegelhalter, D. J. (1988).
Local computations with probabilities on graphical structures and their application to expert systems.
Journal of the Royal Statistical Society: Series B, 50(2), 157-194.
```

**Alarm**:
```
Beinlich, I. A., Suermondt, H. J., Chavez, R. M., & Cooper, G. F. (1989).
The ALARM monitoring system: A case study with two probabilistic inference techniques for belief networks.
In AIME 89 (pp. 247-256). Springer, Berlin, Heidelberg.
```

**Sachs**:
```
Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005).
Causal protein-signaling networks derived from multiparameter single-cell data.
Science, 308(5721), 523-529.
```

**Law School**:
```
Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017).
Counterfactual fairness. In NeurIPS (pp. 4066-4076).
```

---

## 🔧 Implementation Files

| File | Purpose |
|------|---------|
| `tests/utils/dream_loader.py` | DREAM4 networks (5 networks × 10 nodes) |
| `tests/utils/bayesian_network_loaders.py` | Asia (8 nodes) and Alarm (37 nodes) |
| `tests/utils/sachs_loader.py` | Sachs protein network (11 nodes) |
| `tests/utils/law_school_loader.py` | Law School admissions (5 nodes) |
| `tests/utils/benchmark_loaders.py` | Unified interface + heterogeneous data generation |
| `tests/test_benchmark_datasets.py` | Test suite for all benchmarks |

---

## ✅ Verification Status

All benchmarks have been tested and verified:

- ✅ DREAM4 (5 networks): Linear & nonlinear data generation
- ✅ Asia: Data generation with proper causal structure
- ✅ Alarm: 37-node network with topological ordering
- ✅ All data: Standardized (mean ≈ 0, std ≈ 1)
- ✅ All networks: DAG property verified (no cycles)
- ✅ Example datasets: Generated and saved to `data/benchmarks/`

**Status**: Production-ready ✅

---

**For questions or issues, see the test output from `test_benchmark_datasets.py`**
