from __future__ import annotations

import unittest

import numpy as np

from training.quant_flow_graph_v11_r2.phase_b_market import (
    PriceSeries,
    block_shuffle,
    future_targets,
    lag_matrix,
    ridge_fit,
    ridge_predict,
    summarize_gate,
)


class PhaseBMarketTests(unittest.TestCase):
    def test_future_target_starts_after_price_date(self) -> None:
        dates = tuple(f"2026-01-{day:02d}" for day in range(1, 8))
        series = PriceSeries(
            dates=dates,
            close=np.asarray([100, 101, 102, 103, 104, 105, 106], dtype=float),
            high=np.asarray([100, 102, 103, 104, 105, 106, 107], dtype=float),
            low=np.asarray([100, 99, 98, 97, 96, 95, 94], dtype=float),
            index={date: index for index, date in enumerate(dates)},
        )
        future_return, loss = future_targets(series, dates[0], 5)
        self.assertAlmostEqual(future_return, 5.0)
        self.assertAlmostEqual(loss, 5.0)

    def test_lag_never_uses_future_row(self) -> None:
        values = np.arange(12, dtype=float).reshape(6, 2)
        lagged = lag_matrix(values, 2)
        self.assertTrue(np.isnan(lagged[:2]).all())
        np.testing.assert_array_equal(lagged[2:], values[:-2])

    def test_block_shuffle_is_deterministic_and_shape_preserving(self) -> None:
        values = np.arange(60, dtype=float).reshape(30, 2)
        first = block_shuffle(values, seed=11, block_size=5)
        second = block_shuffle(values, seed=11, block_size=5)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, values.shape)

    def test_ridge_fits_simple_relation(self) -> None:
        matrix = np.arange(100, dtype=float).reshape(-1, 1)
        target = matrix[:, 0] * 2.0 + 3.0
        model = ridge_fit(matrix, target, 0.01)
        prediction = ridge_predict(matrix, model)
        self.assertLess(float(np.mean(np.abs(prediction - target))), 0.01)

    def test_gate_requires_all_fixed_checks(self) -> None:
        targets = {}
        for index in range(12):
            name = f"T{index}"
            models = {
                model: {
                    "mae": (
                        1.0
                        if model == "price_only"
                        else 0.9
                        if model == "drift_plus_diffusion"
                        else 0.95
                    )
                }
                for model in (
                    "price_only",
                    "drift_plus_diffusion",
                    "date_block_shuffle",
                    "lagged_5",
                    "raw_flow",
                    "drift_only",
                )
            }
            models["drift_plus_diffusion"]["relative_mae_improvement_vs_price_pct"] = 10.0
            targets[name] = {
                "pooled": models,
                "folds": [
                    {
                        "models": {
                            "price_only": {"mae": 1.0},
                            "drift_plus_diffusion": {"mae": 0.9},
                        }
                    }
                ],
            }
        self.assertEqual(summarize_gate(targets)["status"], "PHASE_B_MARKET_SURVIVOR")


if __name__ == "__main__":
    unittest.main()
