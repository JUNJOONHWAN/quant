from __future__ import annotations

import unittest

import numpy as np

from training.quant_flow_graph_v12.residual_canary import (
    RESIDUAL_SHRINKAGE,
    capped_residual_prediction,
    date_balanced_weights,
    preregistration,
    residual_caps,
)


class V12ResidualCanaryTests(unittest.TestCase):
    def test_date_balancing_equalizes_total_date_weight(self) -> None:
        dates = np.asarray([0, 0, 0, 1, 2, 2], dtype=np.int32)
        indices = np.arange(len(dates))
        weights = date_balanced_weights(dates, indices)
        totals = [float(np.sum(weights[dates == code])) for code in (0, 1, 2)]
        self.assertAlmostEqual(totals[0], totals[1], places=6)
        self.assertAlmostEqual(totals[1], totals[2], places=6)
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=6)

    def test_residual_adapter_is_zero_when_flow_adds_nothing(self) -> None:
        price = np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
        caps = np.asarray([0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(
            capped_residual_prediction(price, price, caps), price
        )

    def test_residual_adapter_obeys_shrinkage_and_caps(self) -> None:
        price = np.zeros((2, 2), dtype=np.float32)
        enriched = np.asarray([[1.0, -1.0], [100.0, -100.0]], dtype=np.float32)
        caps = np.asarray([0.5, 0.5], dtype=np.float32)
        result = capped_residual_prediction(price, enriched, caps)
        self.assertAlmostEqual(float(result[0, 0]), RESIDUAL_SHRINKAGE)
        self.assertAlmostEqual(float(result[0, 1]), -RESIDUAL_SHRINKAGE)
        self.assertAlmostEqual(float(result[1, 0]), 0.5)
        self.assertAlmostEqual(float(result[1, 1]), -0.5)

    def test_caps_use_training_targets_only_and_are_positive(self) -> None:
        targets = np.arange(60, dtype=np.float32).reshape(5, 12)
        caps = residual_caps(targets)
        self.assertEqual(caps.shape, (12,))
        self.assertTrue(np.all(caps > 0))

    def test_preregistration_separates_later_model_families(self) -> None:
        payload = preregistration()
        self.assertTrue(payload["frozen_before_results"])
        self.assertTrue(payload["scope"]["no_row_or_symbol_sampling"])
        interpretation = payload["interpretation"]
        self.assertTrue(interpretation["new_future_lockbox_required"])
        self.assertTrue(
            interpretation["graph_diffusion_and_state_space_models_not_part_of_this_canary"]
        )
        self.assertTrue(
            interpretation["generative_denoising_diffusion_not_part_of_this_canary"]
        )


if __name__ == "__main__":
    unittest.main()
