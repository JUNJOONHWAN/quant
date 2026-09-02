from __future__ import annotations

import unittest

import numpy as np

from training.quant_flow_graph_v11_r2.phase_b_cluster import (
    CLUSTER_ANCHORS,
    cross_sectional_metrics,
    lag_matrix_by_cluster,
    shuffle_within_cluster,
)


class PhaseBClusterTests(unittest.TestCase):
    def test_unclassified_is_not_silently_mapped(self) -> None:
        self.assertNotIn("UNCLASSIFIED", CLUSTER_ANCHORS)
        self.assertEqual(CLUSTER_ANCHORS["FINANCIALS"], "XLF")
        self.assertEqual(CLUSTER_ANCHORS["SEMICONDUCTOR_MEMORY"], "SMH")

    def test_cluster_lag_stays_inside_cluster(self) -> None:
        values = np.asarray([[1.0], [10.0], [2.0], [20.0], [3.0], [30.0]])
        clusters = ("A", "B", "A", "B", "A", "B")
        lagged = lag_matrix_by_cluster(values, clusters, 1)
        self.assertTrue(np.isnan(lagged[0, 0]))
        self.assertTrue(np.isnan(lagged[1, 0]))
        self.assertEqual(lagged[2, 0], 1.0)
        self.assertEqual(lagged[3, 0], 10.0)

    def test_cluster_shuffle_is_deterministic(self) -> None:
        values = np.arange(80, dtype=float).reshape(40, 2)
        clusters = tuple("A" if index % 2 == 0 else "B" for index in range(40))
        first = shuffle_within_cluster(values, clusters, seed=7)
        second = shuffle_within_cluster(values, clusters, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_cross_sectional_rank_metrics_reward_correct_order(self) -> None:
        dates = tuple(["2026-01-02"] * 10 + ["2026-01-05"] * 10)
        target = np.tile(np.arange(10, dtype=float), 2)
        metrics = cross_sectional_metrics(dates, target, target)
        self.assertAlmostEqual(metrics["mean_daily_rank_ic"], 1.0)
        self.assertGreater(metrics["mean_top_minus_bottom_spread"], 0)


if __name__ == "__main__":
    unittest.main()
