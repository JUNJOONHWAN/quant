import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from workflows.market_structure_oracle import incremental_store


class _Pipeline:
    class _Database:
        def __init__(self, db_path):
            self.db_path = db_path

    def __init__(self, db_path, outcomes):
        self.database = self._Database(db_path)
        self.calls = []
        self.outcomes = outcomes

    def capture_daily(self, session, symbols, **kwargs):
        self.calls.append((session, symbols, kwargs))
        job_id = "daily:test"
        with sqlite3.connect(self.database.db_path) as connection:
            for symbol in symbols:
                status, count, error = self.outcomes[symbol]
                connection.execute(
                    """
                    INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        job_id,
                        "fmp",
                        f"{symbol}:{session}:{session}",
                        status,
                        1,
                        "{}",
                        None,
                        count,
                        error,
                    ),
                )
        return {"ok": True, "symbols": symbols, "job_id": job_id}


class ProtectedSymbolGateTests(unittest.TestCase):
    def test_loads_unique_uppercase_favorites(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "favorites.json"
            path.write_text(
                json.dumps(["stx", "CRDO", "STX"]),
                encoding="utf-8",
            )
            self.assertEqual(
                incremental_store._load_protected_symbols(path),
                ["STX", "CRDO"],
            )

    def test_active_master_warning_blocks_release(self):
        pipeline = mock.Mock()
        pipeline.capture_fmp_active_universe.return_value = {
            "warnings": ["screener_hit_max_pages_NASDAQ"]
        }
        with self.assertRaises(incremental_store.IncrementalStoreError):
            incremental_store._daily_universe(pipeline, "2026-07-30")

    def test_missing_symbols_trigger_per_symbol_fmp_repair(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = Path(root) / "coverage.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE checkpoints(
                        job_id TEXT,source TEXT,item_key TEXT,status TEXT,
                        attempts INTEGER,scope_json TEXT,raw_artifact_id INTEGER,
                        observation_count INTEGER,last_error TEXT
                    )
                    """
                )
            pipeline = _Pipeline(
                db_path,
                {
                    "STX": ("done", 1, None),
                    "CRDO": ("done", 0, None),
                },
            )
            ledger_path = Path(root) / "ledger.jsonl"
            with (
                mock.patch.object(
                    incremental_store,
                    "_observed_market_symbols",
                    side_effect=[
                        {"NVDA"},
                        {"NVDA", "STX"},
                    ],
                ),
                mock.patch.object(
                    incremental_store,
                    "_invalid_market_symbols",
                    side_effect=[set(), set()],
                ),
            ):
                result = incremental_store._repair_required_symbols(
                    pipeline=pipeline,
                    session="2026-07-30",
                    required_symbols=["NVDA", "STX", "CRDO"],
                    membership_basis_by_symbol={
                        "NVDA": "active",
                        "STX": "active",
                        "CRDO": "active",
                    },
                    ledger_path=ledger_path,
                )
            ledger = [
                json.loads(line)
                for line in ledger_path.read_text().splitlines()
            ]
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["bar_count"], 2)
            self.assertEqual(result["no_bar_count"], 1)
            self.assertEqual(result["error_count"], 0)
            self.assertEqual(
                {row["symbol"]: row["outcome"] for row in ledger},
                {"CRDO": "NO_BAR", "NVDA": "BAR", "STX": "BAR"},
            )
            self.assertEqual(
                pipeline.calls,
                [
                    (
                        "2026-07-30",
                        ["CRDO", "STX"],
                        {"source": "fmp", "continue_on_error": True},
                    )
                ],
            )

    def test_gate_records_incomplete_when_repair_does_not_fill_symbol(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = Path(root) / "coverage.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE checkpoints(
                        job_id TEXT,source TEXT,item_key TEXT,status TEXT,
                        attempts INTEGER,scope_json TEXT,raw_artifact_id INTEGER,
                        observation_count INTEGER,last_error TEXT
                    )
                    """
                )
            pipeline = _Pipeline(
                db_path,
                {
                    "STX": ("done", 1, None),
                    "CRDO": ("failed", 0, "HTTP 500"),
                },
            )
            with (
                mock.patch.object(
                    incremental_store,
                    "_observed_market_symbols",
                    side_effect=[set(), {"STX"}],
                ),
                mock.patch.object(
                    incremental_store,
                    "_invalid_market_symbols",
                    side_effect=[set(), set()],
                ),
            ):
                result = incremental_store._repair_required_symbols(
                    pipeline=pipeline,
                    session="2026-07-30",
                    required_symbols=["STX", "CRDO"],
                    membership_basis_by_symbol={
                        "STX": "active",
                        "CRDO": "active",
                    },
                    ledger_path=Path(root) / "ledger.jsonl",
                )
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["missing_after"], ["CRDO"])
            self.assertEqual(result["error_count"], 1)

    def test_invalid_bulk_bar_is_requeried_before_bar_outcome(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = Path(root) / "coverage.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE checkpoints(
                        job_id TEXT,source TEXT,item_key TEXT,status TEXT,
                        attempts INTEGER,scope_json TEXT,raw_artifact_id INTEGER,
                        observation_count INTEGER,last_error TEXT
                    )
                    """
                )
            pipeline = _Pipeline(
                db_path, {"STX": ("done", 1, None)}
            )
            with (
                mock.patch.object(
                    incremental_store,
                    "_observed_market_symbols",
                    side_effect=[{"STX"}, {"STX"}],
                ),
                mock.patch.object(
                    incremental_store,
                    "_invalid_market_symbols",
                    side_effect=[{"STX"}, set()],
                ),
                mock.patch.object(
                    incremental_store,
                    "_reconcile_invalid_exact_rows",
                    return_value={},
                ),
            ):
                result = incremental_store._repair_required_symbols(
                    pipeline=pipeline,
                    session="2026-07-30",
                    required_symbols=["STX"],
                    membership_basis_by_symbol={"STX": "active"},
                    ledger_path=Path(root) / "ledger.jsonl",
                )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["invalid_before_count"], 1)
            self.assertEqual(result["bar_count"], 1)
            self.assertEqual(pipeline.calls[0][1], ["STX"])

    def test_invalid_bulk_projection_becomes_no_bar_after_exact_empty(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = Path(root) / "projection.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE daily_observations(
                        source TEXT,symbol TEXT,trade_date TEXT,extra_json TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE quality_checks(
                        symbol TEXT,trade_date TEXT,status TEXT
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO daily_observations VALUES(?,?,?,?)",
                    (
                        "fmp",
                        "BAD",
                        "2026-07-30",
                        json.dumps(
                            {
                                "endpoint_contract":
                                "fmp_v4_legacy_batch_historical_eod"
                            }
                        ),
                    ),
                )
                connection.execute(
                    "INSERT INTO quality_checks VALUES(?,?,?)",
                    ("BAD", "2026-07-30", "invalid"),
                )
            pipeline = mock.Mock()
            pipeline.database.db_path = db_path
            outcomes = (
                incremental_store._reconcile_invalid_exact_rows(
                    pipeline=pipeline,
                    session="2026-07-30",
                    invalid_symbols={"BAD"},
                    checkpoint_rows={
                        "BAD": {
                            "status": "done",
                            "observation_count": 0,
                        }
                    },
                )
            )
            with sqlite3.connect(db_path) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM daily_observations"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM quality_checks"
                    ).fetchone()[0],
                    0,
                )
            self.assertEqual(outcomes, {"BAD": "NO_BAR"})

    def test_quarantined_invalid_bar_is_resolved_but_not_a_bar(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = Path(root) / "coverage.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE checkpoints(
                        job_id TEXT,source TEXT,item_key TEXT,status TEXT,
                        attempts INTEGER,scope_json TEXT,raw_artifact_id INTEGER,
                        observation_count INTEGER,last_error TEXT
                    )
                    """
                )
            pipeline = _Pipeline(
                db_path, {"BAD": ("done", 1, None)}
            )
            with (
                mock.patch.object(
                    incremental_store,
                    "_observed_market_symbols",
                    side_effect=[{"BAD"}, set()],
                ),
                mock.patch.object(
                    incremental_store,
                    "_invalid_market_symbols",
                    side_effect=[{"BAD"}, set()],
                ),
                mock.patch.object(
                    incremental_store,
                    "_reconcile_invalid_exact_rows",
                    return_value={"BAD": "QUARANTINED_INVALID_BAR"},
                ),
            ):
                result = incremental_store._repair_required_symbols(
                    pipeline=pipeline,
                    session="2026-07-30",
                    required_symbols=["BAD"],
                    membership_basis_by_symbol={"BAD": "active"},
                    ledger_path=Path(root) / "ledger.jsonl",
                )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["bar_count"], 0)
            self.assertEqual(result["quarantined_invalid_bar_count"], 1)
            self.assertEqual(result["error_count"], 0)

    def test_daily_price_capture_uses_ultimate_bulk_never_massive(self):
        pipeline = mock.Mock()
        pipeline.database.db_path = Path("/tmp/fake.sqlite3")
        with (
            mock.patch.object(
                incremental_store,
                    "_capture_fmp_stable_eod_bulk",
                return_value={"ok": True},
            ) as legacy,
            mock.patch.object(
                incremental_store,
                "_market_row_count",
                return_value=10_500,
            ),
        ):
            result = incremental_store._capture_current_session(
                pipeline=pipeline,
                session="2026-07-30",
                allowed_reference_symbols={"AAPL", "NVDA"},
            )
            self.assertEqual(result["mode"], "fmp_stable_eod_bulk")
        legacy.assert_called_once()
        pipeline.capture_daily.assert_not_called()

    def test_symbol_change_lineage_is_pit_visible(self):
        with tempfile.TemporaryDirectory() as root:
            db_path = Path(root) / "lineage.sqlite3"
            incremental_store._initialize_oracle_tables(db_path)
            path = Path(root) / "changes.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "old_symbol": "ABC",
                        "new_symbol": "XYZ",
                        "event_date": "2026-07-20",
                        "available_date": "2026-07-31",
                        "company_name": "Example",
                        "raw_artifact_id": 1,
                        "capture_event_id": 2,
                        "source_row_index": 0,
                        "captured_at_utc": "2026-07-31T00:00:00+00:00",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            before = incremental_store._ingest_symbol_changes(
                db_path, path, "2026-07-30"
            )
            after = incremental_store._ingest_symbol_changes(
                db_path, path, "2026-07-31"
            )
            self.assertEqual(before["visible_as_of_count"], 0)
            self.assertEqual(after["visible_as_of_count"], 1)


if __name__ == "__main__":
    unittest.main()
