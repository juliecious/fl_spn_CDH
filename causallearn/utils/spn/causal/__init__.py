"""
Causal discovery using tractable SPN inference.

This module provides CI testing and DAG search using exact probabilistic queries.
"""

from .ci_testing import (
    compute_circuit_ci_discrepancy,
    greedy_dag_search_via_circuit_ci,
    pc_algorithm_with_circuit_ci,
)

__all__ = [
    "compute_circuit_ci_discrepancy",
    "greedy_dag_search_via_circuit_ci",
    "pc_algorithm_with_circuit_ci",
]
