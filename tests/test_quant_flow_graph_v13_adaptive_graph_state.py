import unittest

import numpy as np

from training.quant_flow_graph_v13.adaptive_graph_state import (
    FIXED_GAIN,
    GLOBAL_STATE_FIELDS,
    STATE_FIELDS,
    STOCK_GRAPH_STATE_FIELDS,
    build_graph_state_features,
    causal_regime_schedule,
    preregistration,
    topology_shuffle_state_inputs,
)


class AdaptiveGraphStateTests(unittest.TestCase):
    def _fixture(self):
        date_codes = np.repeat(np.arange(4, dtype=np.int32), 2)
        symbol_codes = np.tile(np.arange(2, dtype=np.int32), 4)
        names = (
            "drift_rate_pct",
            "independent_breadth_net",
            "diffusion_coverage",
            "stale_ratio",
            *STOCK_GRAPH_STATE_FIELDS,
        )
        global_values = np.asarray(
            [
                [1.0, 0.2, 0.8, 0.0],
                [1.2, 0.3, 0.8, 0.0],
                [-2.0, -0.5, 0.7, 0.0],
                [-1.5, -0.3, 0.7, 0.8],
            ],
            dtype=np.float32,
        )
        flow = np.repeat(global_values, 2, axis=0)
        stock = np.column_stack(
            [np.arange(8, dtype=np.float32) + offset for offset in range(7)]
        )
        selected = np.column_stack([flow, stock]).astype(np.float32)
        return selected, names, date_codes, symbol_codes

    def test_regime_sign_flip_increases_gain_and_stale_reduces_it(self):
        selected, names, date_codes, _ = self._fixture()
        regime = causal_regime_schedule(
            flow=selected,
            flow_names=names,
            date_codes=date_codes,
            date_count=4,
        )
        self.assertGreater(regime["gain"][2], regime["gain"][1])
        self.assertEqual(regime["sign_flip"][2], 1.0)
        self.assertLess(regime["stale_trust"][3], regime["stale_trust"][2])
        self.assertLess(regime["gain"][3], regime["gain"][2])

    def test_state_is_causal_when_future_inputs_change(self):
        selected, names, date_codes, symbol_codes = self._fixture()
        regime = causal_regime_schedule(
            flow=selected,
            flow_names=names,
            date_codes=date_codes,
            date_count=4,
        )
        before, feature_names = build_graph_state_features(
            selected=selected,
            selected_names=names,
            date_codes=date_codes,
            symbol_codes=symbol_codes,
            date_count=4,
            symbol_count=2,
            regime=regime,
            adaptive=True,
        )
        changed = selected.copy()
        changed[date_codes == 3] += 1000.0
        changed_regime = causal_regime_schedule(
            flow=changed,
            flow_names=names,
            date_codes=date_codes,
            date_count=4,
        )
        after, changed_names = build_graph_state_features(
            selected=changed,
            selected_names=names,
            date_codes=date_codes,
            symbol_codes=symbol_codes,
            date_count=4,
            symbol_count=2,
            regime=changed_regime,
            adaptive=True,
        )
        self.assertEqual(feature_names, changed_names)
        np.testing.assert_allclose(before[date_codes < 3], after[date_codes < 3])

    def test_topology_shuffle_preserves_global_and_stock_multiset(self):
        selected, names, date_codes, _ = self._fixture()
        shuffled = topology_shuffle_state_inputs(
            selected=selected,
            selected_names=names,
            date_codes=date_codes,
            seed=123,
        )
        global_count = len(GLOBAL_STATE_FIELDS)
        np.testing.assert_array_equal(
            selected[:, :global_count], shuffled[:, :global_count]
        )
        for date_code in np.unique(date_codes):
            rows = date_codes == date_code
            np.testing.assert_array_equal(
                np.sort(selected[rows, global_count:], axis=0),
                np.sort(shuffled[rows, global_count:], axis=0),
            )

    def test_fixed_state_contract_and_gain_are_frozen(self):
        selected, names, date_codes, symbol_codes = self._fixture()
        regime = causal_regime_schedule(
            flow=selected,
            flow_names=names,
            date_codes=date_codes,
            date_count=4,
        )
        state, feature_names = build_graph_state_features(
            selected=selected,
            selected_names=names,
            date_codes=date_codes,
            symbol_codes=symbol_codes,
            date_count=4,
            symbol_count=2,
            regime=regime,
            adaptive=False,
        )
        self.assertEqual(tuple(names), STATE_FIELDS)
        self.assertEqual(state.shape, (8, len(STATE_FIELDS) * 2 + 4))
        self.assertEqual(len(feature_names), state.shape[1])
        self.assertAlmostEqual(FIXED_GAIN, 2.0 / 21.0)

    def test_preregistration_forbids_leakage_and_deployment(self):
        contract = preregistration()
        self.assertTrue(contract["frozen_before_results"])
        self.assertFalse(contract["architecture"]["transformer_used"])
        self.assertTrue(contract["prohibitions"]["historical_holdings_imputation"])
        self.assertTrue(contract["prohibitions"]["current_holdings_used_as_historical"])
        self.assertTrue(contract["interpretation"]["future_forward_window_required_before_bf16_or_nvfp4"])


if __name__ == "__main__":
    unittest.main()
