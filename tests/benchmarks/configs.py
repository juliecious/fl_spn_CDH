"""
Configuration definitions for FedCDH Benchmarks.
"""


class BenchmarkConfig:
    def __init__(self, name, args):
        self.name = name
        self.args = args


# Default Production Configurations (CPU-Optimized)
PRODUCTION_CONFIGS = {
    "kci_synthetic": {
        "ci_method": "kci",
        "scenario": "horizontal",
        "model_type": "sachs",
        "n": 200,
        "d": 11,
        "K": 3,
    },
    "voting_synthetic": {
        "ci_method": "voting_pc",
        "scenario": "horizontal",
        "model_type": "sachs",
        "n": 200,
        "d": 11,
        "K": 3,
    },
    "fedspn_horizontal_synthetic": {
        "ci_method": "spn",
        "scenario": "horizontal",
        "model_type": "sachs",
        "ablation_orientation": "mi_hybrid",
        "n": 200,
        "d": 11,
        "K": 3,
    },
    "fedspn_vertical_synthetic": {
        "ci_method": "spn",
        "scenario": "vertical",
        "model_type": "sachs",
        "ablation_orientation": "mi_hybrid",
        "n": 200,
        "d": 11,
        "K": 3,
    },
    "fedspn_hybrid_synthetic": {
        "ci_method": "spn",
        "scenario": "hybrid",
        "model_type": "sachs",
        "ablation_orientation": "mi_hybrid",
        "n": 200,
        "d": 11,
        "K": 3,
    },
    # --- Fast Linear (Speed Check) ---
    "fast_linear": {
        "ci_method": "spn",
        "scenario": "horizontal",
        "model_type": "linear",
        "ablation_orientation": "mi_hybrid",
        "n": 100,
        "d": 5,
        "K": 2,
        "epochs": 2,
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
}
