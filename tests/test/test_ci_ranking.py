"""
Unit Tests for CI Test Result Ranking System

Tests the CIRankingTracker class that implements percentile-based
conditional independence testing as an alternative to fixed alpha thresholds.
"""

import pytest
import numpy as np
from causallearn.utils.ci_ranking import CIRankingTracker, CITestResult


class TestCIRankingTracker:
    """Test suite for CI ranking tracker."""

    def test_initialization(self):
        """Test that tracker initializes correctly."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)
        assert tracker.sparsity_percentile == 0.2
        assert len(tracker.results) == 0
        assert tracker._threshold_cache is None

    def test_invalid_sparsity_percentile(self):
        """Test that invalid percentiles raise errors."""
        with pytest.raises(ValueError):
            CIRankingTracker(sparsity_percentile=0.0)

        with pytest.raises(ValueError):
            CIRankingTracker(sparsity_percentile=1.5)

        with pytest.raises(ValueError):
            CIRankingTracker(sparsity_percentile=-0.1)

    def test_add_result(self):
        """Test adding individual results."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        tracker.add_result(
            x=0,
            y=1,
            S=(),
            cmi_score=0.85,
            p_value=0.001,
            stat_obs=12.3,
            depth=0,
            test_index=0,
        )

        assert len(tracker.results) == 1
        result = tracker.results[0]
        assert result.x == 0
        assert result.y == 1
        assert result.S == ()
        assert result.cmi_score == 0.85

    def test_add_multiple_results(self):
        """Test adding multiple results."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        for i in range(10):
            tracker.add_result(
                x=0,
                y=i + 1,
                S=(),
                cmi_score=i * 0.1,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        assert len(tracker.results) == 10

    def test_compute_threshold_empty(self):
        """Test threshold computation with no results."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)
        threshold = tracker.compute_threshold()
        assert threshold == 0.0

    def test_compute_threshold_percentile(self):
        """Test that threshold is computed at correct percentile."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        # Add 100 results with CMI scores 0.00, 0.01, ..., 0.99
        for i in range(100):
            tracker.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=i * 0.01,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        threshold = tracker.compute_threshold()

        # sparsity_percentile=0.2 means top 20%
        # percentile_rank = (1 - 0.2) * 100 = 80
        # At 80th percentile with scores [0.00, 0.01, ..., 0.99]
        # Expected threshold ≈ 0.80
        assert 0.79 <= threshold <= 0.81, f"Expected ~0.80, got {threshold}"

    def test_should_reject_independence(self):
        """Test decision rule for rejecting independence."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        # Add results
        for i in range(100):
            tracker.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=i * 0.01,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        threshold = tracker.compute_threshold()

        # CMI > threshold → reject independence (keep edge)
        assert tracker.should_reject_independence(0.90, threshold) == True
        assert tracker.should_reject_independence(0.85, threshold) == True

        # CMI <= threshold → accept independence (remove edge)
        assert tracker.should_reject_independence(0.79, threshold) == False
        assert tracker.should_reject_independence(0.50, threshold) == False

    def test_get_dependent_pairs(self):
        """Test retrieval of dependent pairs."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        # Add 10 results with varying CMI scores
        for i in range(10):
            tracker.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=i * 0.1,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        dependent = tracker.get_dependent_pairs()

        # Top 20% of 10 = 2 pairs
        # Highest CMI scores: 0.9 (y=9), 0.8 (y=8)
        assert len(dependent) == 2
        assert (0, 9, ()) in dependent
        assert (0, 8, ()) in dependent

    def test_get_independent_pairs(self):
        """Test retrieval of independent pairs."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        # Add 10 results
        for i in range(10):
            tracker.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=i * 0.1,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        independent = tracker.get_independent_pairs()

        # Bottom 80% of 10 = 8 pairs
        assert len(independent) == 8

    def test_get_statistics(self):
        """Test summary statistics computation."""
        tracker = CIRankingTracker(sparsity_percentile=0.3)

        # Add 10 results
        for i in range(10):
            tracker.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=i * 0.1,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        stats = tracker.get_statistics()

        assert stats["num_tests"] == 10
        assert stats["num_dependent"] == 3  # Top 30% of 10 = 3
        assert stats["num_independent"] == 7
        assert stats["mean_cmi"] == pytest.approx(0.45)  # Mean of 0.0-0.9
        assert stats["median_cmi"] == pytest.approx(0.45)
        assert stats["max_cmi"] == 0.9
        assert stats["min_cmi"] == 0.0

    def test_statistics_empty(self):
        """Test statistics on empty tracker."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)
        stats = tracker.get_statistics()

        assert stats["num_tests"] == 0
        assert stats["num_dependent"] == 0
        assert stats["num_independent"] == 0
        assert stats["threshold"] == 0.0

    def test_threshold_caching(self):
        """Test that threshold is cached correctly."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        for i in range(10):
            tracker.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=i * 0.1,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        threshold1 = tracker.compute_threshold()
        threshold2 = tracker.compute_threshold()

        # Should return cached value
        assert threshold1 == threshold2

    def test_reset(self):
        """Test resetting the tracker."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        for i in range(10):
            tracker.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=i * 0.1,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        tracker.reset()

        assert len(tracker.results) == 0
        assert tracker._threshold_cache is None

    def test_conditioning_set_storage(self):
        """Test that conditioning sets are stored correctly."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        tracker.add_result(
            x=0,
            y=1,
            S=(2, 3, 4),
            cmi_score=0.5,
            p_value=0.05,
            stat_obs=5.0,
            depth=3,
            test_index=0,
        )

        result = tracker.results[0]
        assert result.S == (2, 3, 4)
        assert result.depth == 3

    def test_sparsity_percentile_extremes(self):
        """Test behavior at extreme sparsity values."""
        # Very sparse (top 5%)
        tracker_sparse = CIRankingTracker(sparsity_percentile=0.05)
        for i in range(100):
            tracker_sparse.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=i * 0.01,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        dependent_sparse = tracker_sparse.get_dependent_pairs()
        assert len(dependent_sparse) == 5  # Top 5%

        # Very dense (top 90%)
        tracker_dense = CIRankingTracker(sparsity_percentile=0.9)
        for i in range(100):
            tracker_dense.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=i * 0.01,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        dependent_dense = tracker_dense.get_dependent_pairs()
        assert len(dependent_dense) == 90  # Top 90%

    def test_sorting_stability(self):
        """Test that results are sorted correctly by CMI."""
        tracker = CIRankingTracker(sparsity_percentile=0.2)

        # Add results in random order
        cmi_scores = [0.3, 0.7, 0.1, 0.9, 0.5]
        for i, cmi in enumerate(cmi_scores):
            tracker.add_result(
                x=0,
                y=i,
                S=(),
                cmi_score=cmi,
                p_value=0.05,
                stat_obs=i,
                depth=0,
                test_index=i,
            )

        # Compute threshold (triggers sorting)
        tracker.compute_threshold()

        # Check that results are sorted by CMI descending
        assert tracker._is_sorted == True
        sorted_cmis = [r.cmi_score for r in tracker.results]
        assert sorted_cmis == sorted(sorted_cmis, reverse=True)

    def test_real_world_scenario(self):
        """
        Simulate a realistic CI testing scenario.

        Scenario: Testing 10 variable pairs, some strongly dependent, some independent.
        """
        tracker = CIRankingTracker(sparsity_percentile=0.3)

        # Simulate CI test results
        # Pairs (0,1), (0,2), (0,3) are strongly dependent (high CMI)
        # Pairs (0,4)-(0,9) are weakly/not dependent (low CMI)
        test_results = [
            (0, 1, (), 0.92, 0.001),  # Strong dependence
            (0, 2, (), 0.87, 0.002),  # Strong dependence
            (0, 3, (), 0.81, 0.003),  # Strong dependence
            (0, 4, (), 0.15, 0.45),  # Weak/no dependence
            (0, 5, (), 0.12, 0.52),
            (0, 6, (), 0.08, 0.63),
            (0, 7, (), 0.05, 0.71),
            (0, 8, (), 0.03, 0.82),
            (0, 9, (), 0.01, 0.93),
        ]

        for i, (x, y, S, cmi, pval) in enumerate(test_results):
            tracker.add_result(
                x=x,
                y=y,
                S=S,
                cmi_score=cmi,
                p_value=pval,
                stat_obs=cmi * 100,
                depth=0,
                test_index=i,
            )

        # With sparsity_percentile=0.3, we keep top 30% = ~3 edges
        dependent = tracker.get_dependent_pairs()
        stats = tracker.get_statistics()

        # Should identify the 3 strongly dependent pairs
        assert len(dependent) >= 2  # At least the top 2-3
        assert (0, 1, ()) in dependent
        assert (0, 2, ()) in dependent

        # Most pairs should be marked independent
        assert stats["num_independent"] >= 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
