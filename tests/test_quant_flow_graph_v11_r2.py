from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from training.quant_flow_graph_v11_r2.contracts import (
    FORBIDDEN_DIFFUSION_INPUTS,
    MIN_ASSETS_USD,
)
from training.quant_flow_graph_v11_r2.hypotheses import build_registry
from training.quant_flow_graph_v11_r2.phase_a import (
    build_event_cube,
    classify_exposure,
    eligibility_decision,
    infer_issuer,
    weighted_jaccard,
)


class EligibilityTests(unittest.TestCase):
    def test_small_etf_is_removed_not_downweighted(self) -> None:
        decision = eligibility_decision(
            ticker="TINY",
            metadata={
                "type": "etf",
                "currency": "USD",
                "clean_rotation_eligible": True,
                "instrument_class": "CLEAN_ROTATION_ELIGIBLE",
            },
            assets=MIN_ASSETS_USD - 1,
            price=10.0,
            dollar_volume=2_000_000.0,
        )
        self.assertEqual(decision["strict_eligible"], 0)
        self.assertIn("ASSETS_BELOW_MIN", decision["reasons"])

    def test_missing_price_is_not_silently_admitted(self) -> None:
        decision = eligibility_decision(
            ticker="ETF",
            metadata={"type": "etf", "currency": "USD"},
            assets=100_000_000.0,
            price=None,
            dollar_volume=None,
        )
        self.assertEqual(decision["strict_eligible"], 0)
        self.assertIn("PRICE_T1_MISSING", decision["reasons"])
        self.assertIn("DOLLAR_VOLUME_T1_MISSING", decision["reasons"])

    def test_inverse_and_defensive_signs_are_typed(self) -> None:
        inverse = classify_exposure(
            {
                "name": "ProShares UltraShort QQQ -2x",
                "instrument_class": "INVERSE_HEDGE",
                "inverse_flag": True,
            }
        )
        defensive = classify_exposure(
            {
                "name": "Treasury ETF",
                "instrument_class": "BOND_CASH_DEFENSIVE",
                "bond_cash_flag": True,
            }
        )
        self.assertEqual(inverse["effective_sign"], -1.0)
        self.assertEqual(abs(inverse["target_multiple"]), 2.0)
        self.assertEqual(defensive["effective_sign"], -1.0)
        self.assertEqual(inverse["observation_channel"], "TYPED_SPECIAL")

    def test_family_helpers_are_deterministic(self) -> None:
        self.assertEqual(infer_issuer("iShares Core S&P 500 ETF"), "ISHARES")
        similarity, shared = weighted_jaccard(
            {"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}
        )
        self.assertAlmostEqual(similarity, 0.8181818181818181)
        self.assertEqual(shared, 2)


class HypothesisRegistryTests(unittest.TestCase):
    def test_registry_is_frozen_and_hash_is_timestamp_independent(self) -> None:
        first = build_registry(generated_at_utc="2026-01-01T00:00:00+00:00")
        second = build_registry(generated_at_utc="2026-02-01T00:00:00+00:00")
        self.assertEqual(first["specification_sha256"], second["specification_sha256"])
        self.assertEqual(first["status"], "FROZEN_BEFORE_PHASE_B")
        self.assertIn(
            "48_MASSIVE_ACCUM_CLUSTER.flow_breadth", FORBIDDEN_DIFFUSION_INPUTS
        )


class EventCubeIntegrationTests(unittest.TestCase):
    def _source(self, path: Path) -> None:
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE daily_observations(
              source TEXT,symbol TEXT,trade_date TEXT,adjusted_close REAL,
              close REAL,volume REAL
            );
            CREATE TABLE etf_flow_observations(
              ticker TEXT,effective_date TEXT,available_at_date TEXT,
              processed_date TEXT,fund_flow REAL,nav REAL,shares_outstanding REAL
            );
            """
        )
        sessions = [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
        ]
        connection.executemany(
            "INSERT INTO daily_observations VALUES('fmp','SPY',?,?,?,?)",
            [(date, 100.0, 100.0, 1_000_000.0) for date in sessions],
        )
        for ticker in ("A", "B", "C"):
            connection.executemany(
                "INSERT INTO daily_observations VALUES('fmp',?,?,10,10,200000)",
                [(ticker, date) for date in sessions],
            )
        connection.executemany(
            "INSERT INTO etf_flow_observations VALUES(?,?,?,?,?,?,?)",
            [
                ("A", "2026-01-02", "2026-01-02", "2026-01-02", 1_000_000, 10, 10_000_000),
                ("B", "2026-01-02", "2026-01-02", "2026-01-02", 0, 10, 1_000_000),
                ("A", "2026-01-05", "2026-01-05", "2026-01-05", 0, 10, 10_000_000),
                ("C", "2026-01-06", "2026-01-06", "2026-01-06", 5_000, 10, 10_000_000),
            ],
        )
        connection.commit()
        connection.close()

    def _family(self, path: Path) -> None:
        connection = sqlite3.connect(path)
        connection.execute(
            """
            CREATE TABLE etf_identity(
              ticker TEXT PRIMARY KEY,issuer_family TEXT,benchmark_family TEXT,
              cluster_family TEXT,independent_family_id TEXT,
              clean_rotation_eligible INTEGER,instrument_class TEXT,
              effective_sign REAL,target_multiple REAL,observation_channel TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO etf_identity VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (ticker, "ISSUER", "", "TEST", f"FAM_{ticker}", 1, "CLEAN_ROTATION_ELIGIBLE", 1, 1, "CLEAN_INDEPENDENT")
                for ticker in ("A", "B", "C")
            ],
        )
        connection.commit()
        connection.close()

    def test_zero_missing_and_timing_stay_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.sqlite3"
            family_path = root / "family.sqlite3"
            event_path = root / "events.sqlite3"
            self._source(source_path)
            self._family(family_path)
            source = sqlite3.connect(source_path)
            source.row_factory = sqlite3.Row
            metadata = {
                ticker: {
                    "type": "etf",
                    "currency": "USD",
                    "clean_rotation_eligible": True,
                    "instrument_class": "CLEAN_ROTATION_ELIGIBLE",
                }
                for ticker in ("A", "B", "C")
            }
            receipt = build_event_cube(
                source=source,
                metadata=metadata,
                family_registry_path=family_path,
                output_path=event_path,
                start_date=None,
                end_date=None,
            )
            source.close()
            self.assertEqual(receipt["timing_violation_count"], 0)
            output = sqlite3.connect(event_path)
            zero = output.execute(
                "SELECT true_zero,missing_exact_t2,fund_flow FROM etf_flow_events WHERE signal_date='2026-01-07' AND ticker='A'"
            ).fetchone()
            missing = output.execute(
                "SELECT true_zero,missing_exact_t2,fund_flow FROM etf_flow_events WHERE signal_date='2026-01-08' AND ticker='A'"
            ).fetchone()
            tiny = output.execute(
                "SELECT strict_eligible,exclusion_reasons FROM etf_flow_events WHERE signal_date='2026-01-06' AND ticker='B'"
            ).fetchone()
            output.close()
            self.assertEqual(zero, (1, 0, 0.0))
            self.assertEqual(missing, (0, 1, None))
            self.assertEqual(tiny[0], 0)
            self.assertIn("ASSETS_BELOW_MIN", tiny[1])


if __name__ == "__main__":
    unittest.main()
