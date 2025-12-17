# Federated Causal Discovery via Probabilistic Circuits

This repository implements **Federated Causal Discovery** using **Sum-Product Networks (SPNs)** as a privacy-preserving density estimator. By replacing traditional summary statistics or raw data exchange with a generative global SPN, this framework supports **Horizontal, Vertical, and Hybrid** data splits.

The core innovation is a two-phase pipeline that decouples *Federated Density Estimation* (learning the joint probability $P(V)$) from *Causal Structure Learning* (discovering the graph $G$). This approach aligns with the Thesis goals of integrating Einsum Networks into the FedCDH pipeline.

## 🚀 Key Features

* **Privacy-Preserving Oracle:** Uses a Federated RAT-SPN (Random & Tensorized SPN) to answer Conditional Independence Tests (CITs) without sharing raw data or covariance matrices.
* **Hybrid & Vertical Split Support:** Handles missing features across clients using **Masked Generative Learning**. Clients only update parameters for variables they observe, enabling seamless aggregation of heterogeneous data.
* **Robust Statistics:** Replaces brittle hardcoded thresholds with Monte Carlo-based Conditional Mutual Information (CMI) estimation and permutation testing.
* **Efficiency:** Decouples training from inference. The SPN is trained once (Phase 1), allowing the PC algorithm (Phase 2) to query the model instantly without repeated communication rounds.

## 📂 Project Structure

This project is organized to separate the federated learning mechanics from the causal discovery logic.

```text
fl_spn_CDH/
├── fed_spn/                  # Core package for SPN logic
│   ├── __init__.py
│   ├── structure.py          # RAT-SPN model wrapper (simple-einet integration)
│   ├── privacy.py            # Federated aggregation & Gradient masking logic
│   └── oracle.py             # SPN-based CIT estimator (The "Oracle" for FedCDH)
│
├── data/                     # Data management
│   ├── __init__.py
│   └── loader.py             # Generates Horizontal/Vertical/Hybrid splits
│
├── experiments/              # Execution scripts
│   └── driver.py             # Main entry point (Training + Discovery)
│
├── causallearn/              # (Submodule) Standard Causal Discovery Library
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

# 🛠️ Installation
1. Clone the repository:

```Bash
git clone [https://github.com/yourusername/fl-spn-cdh.git](https://github.com/yourusername/fl-spn-cdh.git)
cd fl-spn-cdh
```

2. Install dependencies: This project relies on simple-einet for the probabilistic backend and torch.

```Bash
pip install torch numpy pandas
pip install simple-einet  # [https://github.com/juliecious/simple-einet](https://github.com/juliecious/simple-einet)
pip install -r requirements.txt
```

# 🏃 Usage
The `driver.py` script handles the end-to-end pipeline: Phase 1 (Federated Density Estimation) $\to$ Phase 2 (Causal Discovery).
1. Run a Standard Experiment (Hybrid Split)To simulate 3 clients with heterogeneous data (hybrid split) and run the full discovery pipeline:

To simulate 3 clients with heterogeneous data (hybrid split) and run the full discovery pipeline:
```bash
python experiments/driver.py \
    --clients 3 \
    --d 10 \
    --split_type hybrid \
    --rounds 20 \
    --local_epochs 5 \
    --alpha 0.01
```

2. Run with Vertical SplittingSimulates a scenario where clients hold disjoint feature sets (e.g., Client A has $X_1 \dots X_5$, Client B has $X_6 \dots X_{10}$). The Global SPN learns the joint distribution by aggregating partial gradients.

```bash
python experiments/driver.py \
    --split_type vertical \
    --d 20 \
    --clients 2 \
    --rounds 50
```

3. Skip Training (Use Pre-trained Model)
If you have already trained the Global SPN and want to debug the Causal Discovery phase (PC Algorithm) independently:
```bash
python experiments/driver.py \
    --load_model global_spn_hybrid.pth \
    --skip_training
```


# 🧠 Methodology
Phase 1: Federated Density Estimation
Clients collaboratively train a Global RAT-SPN1.
- Horizontal Split: Standard FedAvg on SPN weights.
- Vertical/Hybrid Split: Clients apply Gradient Masking during local training. They only update leaf parameters for the variables they observe, preventing corruption of the global model's unobserved features.

Phase 2: Causal Discovery
The FedCDH algorithm runs centrally using the trained Global SPN as an oracle3.
- Instead of sending data requests to clients, the server queries the SPN to estimate $CMI(X; Y | Z)$.
- Metric: We use Conditional Mutual Information (CMI) derived from the SPN's log-likelihoods:$$CMI(X;Y|Z) = \mathbb{E}_{Q} \left[ \log \frac{Q(x,y|z)}{Q(x|z)Q(y|z)} \right]$$This significantly reduces communication overhead and improves privacy4.

# 📜 References
- FedCDH: Federated Causal Discovery from Heterogeneous Data (Li et al., 2024).
- RAT-SPN: Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning (Peharz et al., 2019).
- Einsum Networks: Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits (Peharz et al., 2020).
