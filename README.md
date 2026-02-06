# FedCDH with Federated Circuits (FedPC)

A high-performance implementation of **Federated Causal Discovery from Heterogeneous Data (FedCDH)** (Li et al., ICLR 2024), powered by **Federated Probabilistic Circuits (FedPC)** (Seng et al., 2025) as the privacy-preserving density oracle.

## 🚀 Overview

This library solves the problem of discovering causal graphs from heterogeneous data distributed across multiple clients (Horizontal, Vertical, or Hybrid partitions) **without sharing raw data**.

Instead of slow, kernel-based conditional independence tests (like KCI), this implementation uses **Sum-Product Networks (SPNs)** trained in a federated "Mixture of Experts" architecture to estimate global densities efficiently.

### Key Features
*   **Privacy-First:** Clients share only model parameters (SPN circuits) or cluster summaries, never raw data.
*   **Universal Federation:** Supports **Horizontal**, **Vertical** (via Feature Mapping), and **Hybrid** data splitting.
*   **Speed:** Efficient analytic Conditional Mutual Information (CMI) calculation via SPN inference.
*   **Robust Orientation:** Implements **Hybrid Orientation** combining Mechanism Invariance (variance across clients) with Information Theoretic directionality (SPN entropy).

## 📦 Architecture

### 1. Federated Data Partitioning (Layer 1)
We implement a **Simulated Federated K-Means** protocol to align heterogeneous clients into global "mechanism clusters" (e.g., Condition A vs. Condition B) without sharing data.
*   **Horizontal:** Aggregates centroids via secure summation.
*   **Vertical:** Aggregates partial distances via secure summation.

### 2. The "Castle" (Layer 3: Aggregation)
To handle structural heterogeneity, we aggregate Local SPNs into a **Global Joint Density** $P(X, U)$:
*   **Horizontal/Hybrid:** Uses **Mixture of Experts** (Sum) to prevent density sharpening.
    $$P_{global}(X) = \sum_{k=1}^{K} w_k P_k(X)$$
*   **Vertical:** Uses **Product of Experts** (Factorization) to stitch disjoint feature sets.
    $$P(X) = \prod_{k=1}^{K} P(X_k)$$

### 3. SPN-CIT Oracle (Layer 4: Discovery)
We replace the standard `fisherz` or `kci` test with a rigorous G-test based on CMI:
*   **Statistic:** $2N \cdot I(X;Y|Z)$ (calculated analytically from SPN).
*   **Test:** Approximated as $\chi^2(df=1)$ to yield a valid p-value for the PC algorithm.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/fl_spn_CDH.git
cd fl_spn_CDH

# Install dependencies
pip install -r requirements.txt
# Requires: torch, numpy, scipy, networkx, simple-einet, pandas, matplotlib, seaborn
```

## 📊 Benchmarking Workflow

We provide a modular, professional benchmarking suite to compare FedCDH against Oracles (KCI) and Naive Baselines (Voting).

### 1. Run Experiments
Run specific configurations in batches (Monte Carlo simulation with seeds). Results are saved to distinct timestamped folders.

```bash
# Run Voting-FedPC Baseline (Fast)
python tests/benchmarks/run_experiment.py --config voting_synthetic --num_seeds 5

# Run FedSPN (Horizontal)
python tests/benchmarks/run_experiment.py --config fedspn_horizontal_synthetic --num_seeds 5

# Run Centralized KCI (Oracle - Slow!)
python tests/benchmarks/run_experiment.py --config kci_synthetic --num_seeds 5
```

### 2. Analyze & Visualize
Aggregate all run metrics into a summary table and generate publication-ready plots.

```bash
python tests/benchmarks/analyze_results.py
```
Outputs are saved to `tests/experiments/summary_{TIMESTAMP}/`.

## 📈 Performance (Preliminary)

Recent **Smoke Test ($N=100$)** results on synthetic data:

| Method | Scenario | Skel F1 | DAG F1 | Cost (KB) |
| :--- | :--- | :--- | :--- | :--- |
| **FedSPN** | **Horizontal** | **0.75** | **0.25** | **43.2** |
| **Voting-FedPC** | Horizontal | 0.00 | 0.00 | N/A |

*Note: Voting fails completely on small heterogeneous samples due to Simpson's Paradox. FedSPN successfully recovers structure.*

## 🗺️ Project Roadmap

### Completed
- [x] **Refactoring:** Modular library structure (`causallearn.search.FCMBased.FedCDH`).
- [x] **Baselines:** Voting-FedPC and Centralized KCI.
- [x] **Real Data:** Robust Sachs dataset loader with interventional partitioning.
- [x] **Theory:** G-test p-values for SPN-CIT.
- [x] **Orientation:** Hybrid Score (Invariance + Entropy).

### In Progress
- [ ] **Full Benchmarking:** Running $N=500$ suite for final paper tables.
- [ ] **Real-World Validation:** Scaling to full Sachs dataset ($N=853$).

## 📚 References
1.  **FedCDH:** Li, L., et al. "Federated Causal Discovery from Heterogeneous Data." *ICLR 2024*.
2.  **Federated Circuits:** Seng, J., et al. "Federated Probabilistic Circuits." *AISTATS 2025 (Preprint)*.
