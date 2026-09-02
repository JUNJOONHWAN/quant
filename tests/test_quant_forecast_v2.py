from __future__ import annotations

import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from training.quant_forecast_v2.contracts import TimingRow, model_feature_columns
from training.quant_forecast_v2.features import compute_price_features, price_frame
from training.quant_forecast_v2.finalize import (
    _prediction_interval_bounds,
    select_point_variants,
)
from training.quant_forecast_v2.flow import (
    FLOW_CACHE_SCHEMA,
    FlowCache,
    _available_session,
    aggregate_symbol_flow,
    benchmark_flow_features,
)
from training.quant_forecast_v2.index_membership import reconstruct_memberships
from training.quant_forecast_v2.io_utils import write_json_atomic
from training.quant_forecast_v2.panel import make_timing_rows


class ForecastV2ContractTests(unittest.TestCase):
    def test_operational_timing_is_t_minus_one_price_t_minus_two_flow(self) -> None:
        sessions = ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"]
        rows = make_timing_rows(sessions, "2026-08-27", "2026-08-27")
        self.assertEqual(
            rows,
            [TimingRow("2026-08-27", "2026-08-26", "2026-08-25", 2)],
        )

    def test_flow_provider_date_becomes_available_on_second_later_session(self) -> None:
        sessions = ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"]
        self.assertEqual(
            _available_session(sessions, "2026-08-25", "2026-08-25"),
            "2026-08-27",
        )

    def test_benchmark_flow_uses_signal_cutoff_for_t2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
                connection.execute(
                    "INSERT INTO metadata VALUES('schema_version',?)", (FLOW_CACHE_SCHEMA,)
                )
                connection.execute(
                    "CREATE TABLE flow(ticker TEXT,effective_date TEXT,processed_date TEXT,"
                    "available_session TEXT,flow_rate_pct REAL,fund_flow REAL,nav REAL,"
                    "shares_outstanding REAL,PRIMARY KEY(ticker,effective_date)) WITHOUT ROWID"
                )
                for ticker, rate in (("SPY", 1.25), ("QQQ", -2.5)):
                    connection.execute(
                        "INSERT INTO flow VALUES(?,?,?,?,?,?,?,?)",
                        (ticker, "2026-08-25", "2026-08-25", "2026-08-27", rate, rate, 1, 1),
                    )
            timing = TimingRow("2026-08-27", "2026-08-26", "2026-08-25", 2)
            cache = FlowCache(path)
            try:
                values = benchmark_flow_features(
                    cache,
                    [timing],
                    ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"],
                )["2026-08-26"]
            finally:
                cache.close()
        self.assertEqual(values["spy_flow_rate_t2"], 1.25)
        self.assertEqual(values["qqq_flow_rate_t2"], -2.5)

    def test_full_etf_net_uses_every_observed_holding_not_top_three(self) -> None:
        exposures = {f"ETF{index}": 10.0 for index in range(1, 6)}
        flows = {"ETF1": 5.0, "ETF2": 4.0, "ETF3": 3.0, "ETF4": -8.0, "ETF5": -7.0}
        result = aggregate_symbol_flow(exposures, flows)
        expected = sum(flows[key] * 0.10 for key in exposures)
        self.assertTrue(math.isclose(result["all_etf_flow_net"], expected))
        self.assertEqual(result["all_etf_flow_observed_count"], 5)
        self.assertLess(result["all_etf_flow_top3_abs_share"], 1.0)

    def test_price_targets_use_t_minus_one_close(self) -> None:
        sessions = [f"2026-01-{day:02d}" for day in range(1, 31)]
        rows = []
        for index, session in enumerate(sessions):
            close = 100.0 + index
            rows.append(
                {
                    "trade_date": session,
                    "open": close,
                    "high": close + 2,
                    "low": close - 3,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1_000_000,
                }
            )
        frame = price_frame(rows, sessions)
        features = compute_price_features(frame, frame, frame, frame)
        reference = sessions[20]
        self.assertTrue(math.isclose(features.at[reference, "return_5d_pct"], ((125.0 / 120.0) - 1.0) * 100.0))
        self.assertTrue(math.isclose(features.at[reference, "upside_5d_pct"], ((127.0 / 120.0) - 1.0) * 100.0))
        self.assertTrue(math.isclose(features.at[reference, "loss_5d_pct"], (1.0 - 118.0 / 120.0) * 100.0))

    def test_membership_reconstruction_reverses_change_event(self) -> None:
        sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]
        payload = {
            "endpoints": {
                "indexes_s_and_p_500_index": {"data": [{"symbol": "NEW"}]},
                "indexes_historical_s_and_p_500": {
                    "data": [{"symbol": "NEW", "removedTicker": "OLD", "dateAdded": "January 05, 2026"}]
                },
                "indexes_nasdaq_index": {"data": [{"symbol": "Q"}]},
                "indexes_historical_nasdaq": {
                    "data": [{"symbol": "Q", "dateAdded": "January 02, 2020"}]
                },
            }
        }
        memberships, _ = reconstruct_memberships(payload, sessions)
        self.assertEqual(memberships["SPY"]["2026-01-02"], frozenset({"OLD"}))
        self.assertEqual(memberships["SPY"]["2026-01-05"], frozenset({"NEW"}))

    def test_t3_is_an_explicit_ablation_not_a_fallback(self) -> None:
        operational = model_feature_columns("price_benchmark_flow")
        legacy = model_feature_columns("price_benchmark_flow_t3")
        self.assertIn("spy_flow_rate_t2", operational)
        self.assertNotIn("spy_flow_rate_t3", operational)
        self.assertIn("spy_flow_rate_t3", legacy)
        self.assertNotIn("spy_flow_rate_t2", legacy)

    def test_atomic_json_accepts_numpy_scalars(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_json_atomic(path, {"year": np.int64(2026), "score": np.float64(1.5)})
            self.assertEqual(path.read_text(encoding="utf-8").strip(), '{\n  "score": 1.5,\n  "year": 2026\n}')

    def test_point_selection_uses_numeric_error_not_basket_score(self) -> None:
        variants = (
            "price",
            "price_benchmark_flow_t3",
            "price_benchmark_flow",
            "price_all_etf_flow",
            "full",
        )
        payload = {"ablation_results": {variant: {"5": {"aggregate": {"targets": {}}}} for variant in variants}}
        for kind in ("return", "upside", "loss"):
            for index, variant in enumerate(variants):
                payload["ablation_results"][variant]["5"]["aggregate"]["targets"][kind] = {
                    "mae_pct": 1.0 + index,
                    "rmse_pct": 2.0 + index,
                    "daily_spearman_ic": {"mean": 0.9 if variant == "full" else 0.1},
                }
        selected = select_point_variants(payload, 5)
        self.assertEqual({item["variant"] for item in selected.values()}, {"price"})

    def test_nonnegative_target_interval_respects_zero_support(self) -> None:
        import numpy as np

        p10, p90 = _prediction_interval_bounds(
            np.array([1.0, 3.0]),
            {"q10": -4.0, "q90": 2.0},
            nonnegative=True,
        )
        np.testing.assert_array_equal(p10, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(p90, np.array([3.0, 5.0]))


if __name__ == "__main__":
    unittest.main()
