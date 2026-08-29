from __future__ import annotations

import unittest

import numpy as np

from training.quant_flow_graph_v11_r2.phase_b_stock import (
    DIRECT_FLOW_FIELDS,
    DIRECT_MASK_FIELDS,
    INDIRECT_BASE_FIELDS,
    aggregate_snapshot_features,
    lag_flow_by_symbol,
    price_capacity_controls,
    topology_shuffle,
)


class PhaseBStockTests(unittest.TestCase):
    def test_family_breadth_deduplicates_clone_etfs(self) -> None:
        cluster_state = np.zeros((1, len(INDIRECT_BASE_FIELDS)), dtype=float)
        cluster_state[0, 0] = 3.0
        direct, indirect, audit = aggregate_snapshot_features(
            stock_count=1,
            edge_stock=np.asarray([0, 0]),
            edge_etf=np.asarray([0, 1]),
            edge_weight=np.asarray([0.2, 0.1]),
            edge_age=np.asarray([0.0, 0.0]),
            family_code=np.asarray([0, 0]),
            cluster_code=np.asarray([0, 0]),
            clean_observed=np.asarray([True, True]),
            special_observed=np.asarray([False, False]),
            flow_rate=np.asarray([2.0, -1.0]),
            fund_flow=np.asarray([10_000_000.0, -1_000_000.0]),
            effective_sign=np.ones(2),
            target_multiple=np.ones(2),
            cluster_states=cluster_state,
            drift_rate=1.0,
        )
        names = DIRECT_MASK_FIELDS + DIRECT_FLOW_FIELDS
        breadth = direct[0, names.index("direct_family_breadth_net")]
        self.assertEqual(breadth, 1.0)
        self.assertEqual(audit["clean_observed_edge_count"], 2)
        self.assertAlmostEqual(indirect[0, 0], 3.0)

    def test_indirect_state_uses_full_cluster_not_only_direct_etf_rate(self) -> None:
        cluster_state = np.zeros((2, len(INDIRECT_BASE_FIELDS)), dtype=float)
        cluster_state[:, 0] = [8.0, -4.0]
        _, indirect, _ = aggregate_snapshot_features(
            stock_count=1,
            edge_stock=np.asarray([0, 0]),
            edge_etf=np.asarray([0, 1]),
            edge_weight=np.asarray([0.75, 0.25]),
            edge_age=np.asarray([0.0, 0.0]),
            family_code=np.asarray([0, 1]),
            cluster_code=np.asarray([0, 1]),
            clean_observed=np.asarray([True, True]),
            special_observed=np.asarray([False, False]),
            flow_rate=np.asarray([0.1, 0.1]),
            fund_flow=np.asarray([1.0, 1.0]),
            effective_sign=np.ones(2),
            target_multiple=np.ones(2),
            cluster_states=cluster_state,
            drift_rate=1.0,
        )
        self.assertAlmostEqual(indirect[0, 0], 5.0)

    def test_price_capacity_width_is_exact(self) -> None:
        price = np.arange(30, dtype=float).reshape(5, 6)
        controls = price_capacity_controls(price, 11)
        self.assertEqual(controls.shape, (5, 11))
        self.assertTrue(np.all(np.isfinite(controls)))

    def test_lag_is_exact_signal_session_and_symbol(self) -> None:
        flow = np.arange(12, dtype=float).reshape(6, 2)
        dates = np.asarray([0, 0, 1, 1, 2, 2])
        symbols = np.asarray([0, 1, 0, 1, 0, 1])
        lagged = lag_flow_by_symbol(flow, dates, symbols, 3, 2, 1)
        self.assertTrue(np.all(np.isnan(lagged[:2])))
        np.testing.assert_array_equal(lagged[2:4], flow[:2])
        np.testing.assert_array_equal(lagged[4:6], flow[2:4])

    def test_topology_shuffle_preserves_global_date_state(self) -> None:
        flow = np.asarray(
            [
                [1.0, 2.0, 10.0],
                [1.0, 2.0, 20.0],
                [3.0, 4.0, 30.0],
                [3.0, 4.0, 40.0],
            ]
        )
        dates = np.asarray([0, 0, 1, 1])
        shuffled = topology_shuffle(flow, dates, 2, seed=7)
        np.testing.assert_array_equal(shuffled[:, :2], flow[:, :2])
        self.assertEqual(sorted(shuffled[:2, 2]), [10.0, 20.0])
        self.assertEqual(sorted(shuffled[2:, 2]), [30.0, 40.0])


if __name__ == "__main__":
    unittest.main()
