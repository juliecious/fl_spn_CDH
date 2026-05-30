"""
Core SPN implementations for federated learning.

This package contains the fundamental SPN building blocks.
"""

from .local import LocalSPNWrapper, LocalClusterMixture
from .univariate import UnivariateSPNWrapper
from .ensemble import EnsembleSPNWrapper
from .structure_learner import FederatedStructureLearner

__all__ = [
    "LocalSPNWrapper",
    "LocalClusterMixture",
    "UnivariateSPNWrapper",
    "EnsembleSPNWrapper",
    "FederatedStructureLearner",
]
