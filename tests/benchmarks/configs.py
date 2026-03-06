"""
Configuration definitions for FedCDH Benchmarks.
"""

# Default Production Configurations (CPU-Optimized)
PRODUCTION_CONFIGS = {
    "fisherz_baseline": {
        "ci_method": "fisherz",
        "scenario": "horizontal",
        "model_type": "sachs",
        "n": 500,
        "d": 11,
        "K": 3,
        "alpha": 0.05,
    },
    "kci_oracle": {
        "ci_method": "kci",
        "scenario": "horizontal",
        "model_type": "sachs",
        "n": 500,
        "d": 11,
        "K": 3,
        "alpha": 0.05,
    },
    "fedspn_horizontal": {
        "ci_method": "spn",
        "scenario": "horizontal",
        "model_type": "sachs",
        "ablation_orientation": "mi_hybrid",
        "n": 500,
        "d": 11,
        "K": 3,
        "epochs": 50,
        "num_sums": 10,
        "num_leaves": 10,
        "alpha": 0.05,
    },
    "fedspn_vertical": {
        "ci_method": "spn",
        "scenario": "vertical",
        "model_type": "sachs",
        "ablation_orientation": "mi_hybrid",
        "n": 500,
        "d": 11,
        "K": 3,
        "epochs": 50,
        "num_sums": 10,
        "num_leaves": 10,
        "alpha": 0.05,
    },
    "fedspn_hybrid": {
        "ci_method": "spn",
        "scenario": "hybrid",
        "model_type": "sachs",
        "ablation_orientation": "mi_hybrid",
        "n": 1000,
        "d": 11,
        "K": 3,
        "epochs": 50,
        "num_sums": 10,
        "num_leaves": 10,
        "alpha": 0.05,
    },
}

# Smoke Test Config
SMOKE_CONFIG = {
    "ci_method": "spn",
    "scenario": "horizontal",
    "model_type": "sachs",
    "ablation_orientation": "mi_hybrid",
    "n": 100,
    "d": 11,
    "K": 2,
    "epochs": 2,
}

# Suite Definitions
BENCHMARK_SUITES = {
    "sachs_cpu": [
        "fisherz_baseline",
        "kci_oracle",
        "fedspn_horizontal",
        "fedspn_vertical",
        "fedspn_hybrid",
    ],
    "sachs_smoke": [
        "fisherz_baseline",
        "fedspn_horizontal",
    ],
}
