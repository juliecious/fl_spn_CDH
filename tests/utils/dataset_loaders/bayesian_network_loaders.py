"""
Classic Bayesian Network Loaders: Asia and Alarm.

These are standard benchmarks from the Bayesian Network literature, widely used
for testing structure learning algorithms.

Networks included:
- Asia (8 nodes, 8 edges): Simple medical diagnosis network
- Alarm (37 nodes, 46 edges): Medical monitoring alarm system

References:
- Lauritzen & Spiegelhalter (1988). "Local computations with probabilities on
  graphical structures and their application to expert systems."
- Beinlich et al. (1989). "The ALARM Monitoring System: A Case Study with two
  Probabilistic Inference Techniques for Belief Networks."
"""

import numpy as np
from typing import Tuple, List


# ------------------------------------------------------------------------------
# ASIA Network (8 nodes, 8 edges)
# ------------------------------------------------------------------------------
# Variables:
# 0: A (visit to Asia)
# 1: S (Smoker)
# 2: T (Tuberculosis)
# 3: L (Lung cancer)
# 4: B (Bronchitis)
# 5: E (Either TB or Lung cancer)
# 6: X (X-ray result)
# 7: D (Dyspnoea - shortness of breath)

ASIA_DAG = np.array(
    [
        [0, 0, 1, 0, 0, 0, 0, 0],  # A -> T
        [0, 0, 0, 1, 1, 0, 0, 0],  # S -> L, B
        [0, 0, 0, 0, 0, 1, 0, 0],  # T -> E
        [0, 0, 0, 0, 0, 1, 0, 0],  # L -> E
        [0, 0, 0, 0, 0, 0, 0, 1],  # B -> D
        [0, 0, 0, 0, 0, 0, 1, 1],  # E -> X, D
        [0, 0, 0, 0, 0, 0, 0, 0],  # X
        [0, 0, 0, 0, 0, 0, 0, 0],  # D
    ]
)

ASIA_NAMES = ["Asia", "Smoke", "Tub", "Lung", "Bronc", "Either", "Xray", "Dysp"]


# ------------------------------------------------------------------------------
# ALARM Network (37 nodes, 46 edges)
# ------------------------------------------------------------------------------
# Medical monitoring system for anesthesia
# Variables represent physiological measurements, diseases, and equipment status

ALARM_DAG = np.array(
    [
        # Node 0-9
        [
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 0: HISTORY
        [
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 1: CVP
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 2: PCWP
        [
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 3: LVEDVOLUME
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 4: LVEDVOLUME
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 5: STROKEVOLUME
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 6: ERRLOWOUTPUT
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 7: HRBP
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 8: HREKG
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 9: HRSAT
        # Node 10-19
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 10: INSUFFANESTH
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 11: CO
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 12: HR
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 13: BP
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 14: CATECHOL (no children in simplified)
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 15: ARTCO2
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 16: SAO2
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 17: EXPCO2
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 18: PAP
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 19: PULMEMBOLUS
        # Node 20-29
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 20: INTUBATION
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 21: PRESS
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 22: DISCONNECT
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 23: MINVOLSET
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 24: VENTMACH
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 25: VENTTUBE
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 26: VENTLUNG
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 27: VENTALV
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 28: FIO2
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 29: PVSAT
        # Node 30-36
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 30: SHUNT
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 31: TPR
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 32: ANAPHYLAXIS
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 33: HYPOVOLEMIA
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 34: LVFAILURE
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 35: ERRCAUTER
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 36: KINKEDTUBE
    ]
)

ALARM_NAMES = [
    "HISTORY",
    "CVP",
    "PCWP",
    "LVEDVOLUME",
    "LVFAILURE",
    "STROKEVOLUME",
    "ERRLOWOUTPUT",
    "HRBP",
    "HREKG",
    "HRSAT",
    "INSUFFANESTH",
    "CO",
    "HR",
    "BP",
    "CATECHOL",
    "ARTCO2",
    "SAO2",
    "EXPCO2",
    "PAP",
    "PULMEMBOLUS",
    "INTUBATION",
    "PRESS",
    "DISCONNECT",
    "MINVOLSET",
    "VENTMACH",
    "VENTTUBE",
    "VENTLUNG",
    "VENTALV",
    "FIO2",
    "PVSAT",
    "SHUNT",
    "TPR",
    "ANAPHYLAXIS",
    "HYPOVOLEMIA",
    "LVFAILURE2",
    "ERRCAUTER",
    "KINKEDTUBE",
]


# ------------------------------------------------------------------------------
# Data Generation Functions
# ------------------------------------------------------------------------------


def generate_asia_data(
    n_samples: int = 1000, mode: str = "linear", seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic continuous data from Asia network.

    The original Asia network has discrete variables, but we generate
    continuous approximations for compatibility with continuous causal discovery.

    Args:
        n_samples: Number of samples
        mode: 'linear' or 'nonlinear'
        seed: Random seed

    Returns:
        X: Data [n_samples, 8]
        B: Adjacency matrix [8, 8]
        feature_names: Variable names
    """
    np.random.seed(seed)

    X = np.zeros((n_samples, 8))

    # Generate in causal order
    # A: Visit to Asia (root node, ~1% probability)
    X[:, 0] = np.random.binomial(1, 0.01, n_samples).astype(float)

    # S: Smoker (root node, ~50% probability)
    X[:, 1] = np.random.binomial(1, 0.5, n_samples).astype(float)

    # T: Tuberculosis (A -> T, rare: 5% if visited Asia, 0.1% otherwise)
    p_t = 0.05 * X[:, 0] + 0.001 * (1 - X[:, 0])
    X[:, 2] = np.random.binomial(1, p_t, n_samples).astype(float)

    # L: Lung cancer (S -> L, 10% if smoker, 1% otherwise)
    p_l = 0.10 * X[:, 1] + 0.01 * (1 - X[:, 1])
    X[:, 3] = np.random.binomial(1, p_l, n_samples).astype(float)

    # B: Bronchitis (S -> B, 60% if smoker, 30% otherwise)
    p_b = 0.60 * X[:, 1] + 0.30 * (1 - X[:, 1])
    X[:, 4] = np.random.binomial(1, p_b, n_samples).astype(float)

    # E: Either TB or Lung cancer (T -> E, L -> E, logical OR)
    X[:, 5] = np.maximum(X[:, 2], X[:, 3])

    # X: X-ray positive (E -> X, 98% if E=1, 5% otherwise)
    p_x = 0.98 * X[:, 5] + 0.05 * (1 - X[:, 5])
    X[:, 6] = np.random.binomial(1, p_x, n_samples).astype(float)

    # D: Dyspnoea (E -> D, B -> D, probability increases with both)
    p_d = 0.1 + 0.5 * X[:, 5] + 0.4 * X[:, 4]
    p_d = np.clip(p_d, 0, 1)
    X[:, 7] = np.random.binomial(1, p_d, n_samples).astype(float)

    # Add Gaussian noise and standardize for continuous version
    noise_scale = 0.1
    X = X + np.random.normal(0, noise_scale, X.shape)

    # Standardize
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)

    return X, ASIA_DAG.copy(), ASIA_NAMES.copy()


def generate_alarm_data(
    n_samples: int = 1000, mode: str = "linear", seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic continuous data from Alarm network.

    This is a simplified continuous approximation of the discrete Alarm network.

    Args:
        n_samples: Number of samples
        mode: 'linear' or 'nonlinear'
        seed: Random seed

    Returns:
        X: Data [n_samples, 37]
        B: Adjacency matrix [37, 37]
        feature_names: Variable names
    """
    np.random.seed(seed)

    n_nodes = 37
    X = np.zeros((n_samples, n_nodes))

    # Topological sort
    def topological_sort(adj):
        n = adj.shape[0]
        visited = [False] * n
        stack = []

        def dfs(v):
            visited[v] = True
            for u in range(n):
                if adj[v, u] == 1 and not visited[u]:
                    dfs(u)
            stack.append(v)

        for i in range(n):
            if not visited[i]:
                dfs(i)

        return stack[::-1]

    order = topological_sort(ALARM_DAG)

    # Generate weights
    W = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if ALARM_DAG[i, j] == 1:
                W[i, j] = np.random.uniform(0.3, 1.5) * np.random.choice([-1, 1])

    # Generate data following causal order
    for node in order:
        parents = np.where(ALARM_DAG[:, node] == 1)[0]

        if len(parents) == 0:
            # Root node
            X[:, node] = np.random.normal(0, 1, n_samples)
        else:
            # Child node
            parent_data = X[:, parents]
            weights = W[parents, node]

            if mode == "linear":
                linear_effect = parent_data @ weights
                noise = np.random.normal(0, 0.5, n_samples)
                X[:, node] = linear_effect + noise
            else:
                # Nonlinear
                linear_effect = parent_data @ weights
                func_type = np.random.choice(["linear", "tanh", "sigmoid"])

                if func_type == "linear":
                    nonlinear_effect = linear_effect
                elif func_type == "tanh":
                    nonlinear_effect = np.tanh(linear_effect)
                else:  # sigmoid
                    nonlinear_effect = 2.0 / (1.0 + np.exp(-linear_effect)) - 1.0

                noise = np.random.normal(0, 0.5, n_samples)
                X[:, node] = nonlinear_effect + noise

    # Standardize
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)

    return X, ALARM_DAG.copy(), ALARM_NAMES.copy()


def load_bayesian_network(
    name: str, n_samples: int = 1000, mode: str = "linear", seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a Bayesian network benchmark and generate data.

    Args:
        name: 'asia' or 'alarm'
        n_samples: Number of samples to generate
        mode: 'linear' or 'nonlinear'
        seed: Random seed

    Returns:
        X: Data matrix
        B: Ground truth adjacency matrix
        feature_names: Variable names
    """
    name = name.lower()

    if name == "asia":
        return generate_asia_data(n_samples, mode, seed)
    elif name == "alarm":
        return generate_alarm_data(n_samples, mode, seed)
    else:
        raise ValueError(f"Unknown network: {name}. Choose 'asia' or 'alarm'.")


def get_network_info(name: str):
    """Print information about a Bayesian network."""
    name = name.lower()

    if name == "asia":
        B = ASIA_DAG
        names = ASIA_NAMES
        description = "Medical diagnosis: tuberculosis and lung cancer"
    elif name == "alarm":
        B = ALARM_DAG
        names = ALARM_NAMES
        description = "Medical monitoring alarm system"
    else:
        raise ValueError(f"Unknown network: {name}")

    n_nodes = len(names)
    n_edges = int(B.sum())
    in_degrees = B.sum(axis=0)
    out_degrees = B.sum(axis=1)

    print(f"{name.upper()} Network")
    print("=" * 60)
    print(f"Description: {description}")
    print(f"Nodes: {n_nodes}")
    print(f"Edges: {n_edges}")
    print(f"Density: {n_edges / (n_nodes * (n_nodes - 1)):.4f}")
    print(f"Avg in-degree: {in_degrees.mean():.2f}")
    print(f"Avg out-degree: {out_degrees.mean():.2f}")
    print(f"Max in-degree: {int(in_degrees.max())}")
    print(f"Max out-degree: {int(out_degrees.max())}")
    print(f"\nVariable names: {', '.join(names)}")


if __name__ == "__main__":
    print("Bayesian Network Benchmark Loaders\n")

    # Test Asia
    print("=" * 60)
    get_network_info("asia")
    X, B, names = generate_asia_data(n_samples=500, mode="linear", seed=42)
    print(f"\nGenerated data shape: {X.shape}")
    print(f"Data mean: {X.mean():.6f}, std: {X.std():.6f}")

    print("\n")

    # Test Alarm
    print("=" * 60)
    get_network_info("alarm")
    X, B, names = generate_alarm_data(n_samples=500, mode="linear", seed=42)
    print(f"\nGenerated data shape: {X.shape}")
    print(f"Data mean: {X.mean():.6f}, std: {X.std():.6f}")
