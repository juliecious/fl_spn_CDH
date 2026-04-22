"""
CI Test Result Ranking System for Adaptive Edge Selection

This module implements percentile-based conditional independence testing
as an alternative to fixed alpha thresholds. Instead of using a statistical
significance threshold (e.g., alpha=0.05), this approach:

1. Collects all CI test results with their CMI scores
2. Ranks tests by CMI magnitude (higher CMI = stronger dependence)
3. Accepts the top-N% as dependent edges (rejects independence)
4. Makes graph sparsity an explicit hyperparameter

This decouples edge selection from statistical convention and provides
explicit control over graph density.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class CITestResult:
    """
    Container for a single conditional independence test result.

    Attributes:
        x: Index of first variable
        y: Index of second variable
        S: Tuple of conditioning set indices
        cmi_score: Conditional mutual information score (higher = more dependent)
        p_value: Statistical p-value from permutation test
        stat_obs: Observed test statistic
        depth: Size of conditioning set |S|
        test_index: Sequential index of this test in the discovery process
    """

    x: int
    y: int
    S: tuple
    cmi_score: float
    p_value: float
    stat_obs: float
    depth: int
    test_index: int


class CIRankingTracker:
    """
    Collects and ranks CI test results for percentile-based edge selection.

    This tracker enables a two-stage skeleton discovery process:
    1. Collection phase: Store all CI test results without making decisions
    2. Ranking phase: Compute CMI threshold at desired percentile and prune edges

    The sparsity_percentile parameter controls graph density:
    - sparsity_percentile=0.1 → Keep top 10% strongest dependencies (sparse graph)
    - sparsity_percentile=0.2 → Keep top 20% (medium sparsity)
    - sparsity_percentile=0.5 → Keep top 50% (dense graph)

    Example:
        >>> tracker = CIRankingTracker(sparsity_percentile=0.2)
        >>> # Collection phase
        >>> tracker.add_result(x=0, y=1, S=(), cmi_score=0.85, p_value=0.001,
        ...                    stat_obs=12.3, depth=0, test_index=0)
        >>> tracker.add_result(x=0, y=2, S=(), cmi_score=0.12, p_value=0.45,
        ...                    stat_obs=2.1, depth=0, test_index=1)
        >>> # Ranking phase
        >>> threshold = tracker.compute_threshold()
        >>> # Decision: Keep edges with CMI > threshold
    """

    def __init__(self, sparsity_percentile: float = 0.2):
        """
        Initialize ranking tracker.

        Args:
            sparsity_percentile: Fraction of tests to accept as dependent (0.0-1.0).
                                Higher values = denser graphs.
                                Default 0.2 (top 20%) provides moderate sparsity.

        Raises:
            ValueError: If sparsity_percentile not in (0.0, 1.0]
        """
        if not 0.0 < sparsity_percentile <= 1.0:
            raise ValueError(
                f"sparsity_percentile must be in (0.0, 1.0], got {sparsity_percentile}"
            )

        self.results: List[CITestResult] = []
        self.sparsity_percentile = sparsity_percentile
        self._threshold_cache = None
        self._is_sorted = False

    def add_result(
        self,
        x: int,
        y: int,
        S: Tuple[int, ...],
        cmi_score: float,
        p_value: float,
        stat_obs: float,
        depth: int,
        test_index: int,
    ):
        """
        Store a CI test result for later ranking.

        Args:
            x: Index of first variable
            y: Index of second variable
            S: Tuple of conditioning set indices
            cmi_score: CMI magnitude (higher = stronger dependence)
            p_value: Statistical p-value from test
            stat_obs: Observed test statistic
            depth: Size of conditioning set
            test_index: Sequential test number
        """
        result = CITestResult(
            x=x,
            y=y,
            S=tuple(S) if not isinstance(S, tuple) else S,
            cmi_score=cmi_score,
            p_value=p_value,
            stat_obs=stat_obs,
            depth=depth,
            test_index=test_index,
        )
        self.results.append(result)
        self._is_sorted = False
        self._threshold_cache = None

    def compute_threshold(self) -> float:
        """
        Compute CMI threshold at the specified sparsity percentile.

        The threshold is computed such that (1 - sparsity_percentile) × 100%
        of tests fall below it. Tests with CMI > threshold are considered
        dependent (reject independence hypothesis).

        Returns:
            CMI threshold value. Edges with cmi_score > threshold are kept.
            Returns 0.0 if no results collected (accept all edges).

        Example:
            If sparsity_percentile=0.2 and we have 100 tests:
            - Sort by CMI descending
            - Threshold is at position 20 (80th percentile)
            - Top 20 tests (highest CMI) are marked dependent
        """
        if not self.results:
            return 0.0

        # Use cached threshold if available
        if self._threshold_cache is not None:
            return self._threshold_cache

        # Sort by CMI score descending (strongest dependencies first)
        if not self._is_sorted:
            self.results.sort(key=lambda r: r.cmi_score, reverse=True)
            self._is_sorted = True

        # Compute percentile threshold
        # percentile = (1 - sparsity_percentile) to get "top N%"
        # Example: sparsity=0.2 → percentile=0.8 → top 20%
        cmi_scores = [r.cmi_score for r in self.results]
        percentile_rank = (1.0 - self.sparsity_percentile) * 100

        threshold = np.percentile(cmi_scores, percentile_rank)
        self._threshold_cache = threshold

        return threshold

    def should_reject_independence(
        self, cmi_score: float, threshold: float = None
    ) -> bool:
        """
        Decision rule: Should we reject independence (i.e., keep the edge)?

        Args:
            cmi_score: CMI score for the test
            threshold: Optional explicit threshold. If None, uses compute_threshold().

        Returns:
            True if cmi_score > threshold (reject independence, keep edge)
            False if cmi_score <= threshold (accept independence, remove edge)
        """
        if threshold is None:
            threshold = self.compute_threshold()

        return cmi_score > threshold

    def get_dependent_pairs(self) -> List[Tuple[int, int, tuple]]:
        """
        Get all variable pairs marked as dependent based on ranking.

        Returns:
            List of (x, y, S) tuples for tests above the threshold
        """
        threshold = self.compute_threshold()
        return [(r.x, r.y, r.S) for r in self.results if r.cmi_score > threshold]

    def get_independent_pairs(self) -> List[Tuple[int, int, tuple]]:
        """
        Get all variable pairs marked as independent based on ranking.

        Returns:
            List of (x, y, S) tuples for tests at or below the threshold
        """
        threshold = self.compute_threshold()
        return [(r.x, r.y, r.S) for r in self.results if r.cmi_score <= threshold]

    def get_statistics(self) -> dict:
        """
        Compute summary statistics for collected results.

        Returns:
            Dictionary with keys:
                - num_tests: Total number of tests
                - threshold: CMI threshold value
                - num_dependent: Number of tests above threshold
                - num_independent: Number of tests at/below threshold
                - mean_cmi: Mean CMI across all tests
                - median_cmi: Median CMI
                - max_cmi: Maximum CMI
                - min_cmi: Minimum CMI
        """
        if not self.results:
            return {
                "num_tests": 0,
                "threshold": 0.0,
                "num_dependent": 0,
                "num_independent": 0,
                "mean_cmi": 0.0,
                "median_cmi": 0.0,
                "max_cmi": 0.0,
                "min_cmi": 0.0,
            }

        threshold = self.compute_threshold()
        cmi_scores = [r.cmi_score for r in self.results]

        return {
            "num_tests": len(self.results),
            "threshold": threshold,
            "num_dependent": sum(1 for r in self.results if r.cmi_score > threshold),
            "num_independent": sum(1 for r in self.results if r.cmi_score <= threshold),
            "mean_cmi": np.mean(cmi_scores),
            "median_cmi": np.median(cmi_scores),
            "max_cmi": np.max(cmi_scores),
            "min_cmi": np.min(cmi_scores),
        }

    def reset(self):
        """Clear all collected results and cached values."""
        self.results.clear()
        self._threshold_cache = None
        self._is_sorted = False
