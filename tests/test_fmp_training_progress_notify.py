import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "fmp_training_progress_notify.py"
)
SPEC = importlib.util.spec_from_file_location("fmp_training_progress_notify", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FmpTrainingProgressNotifyTest(unittest.TestCase):
    def test_expected_items_match_dimension_variant_window_expansion(self):
        plan = {
            "endpoints": [
                {
                    "id": "global",
                    "action": "backfill",
                    "collection": {"mode": "global"},
                },
                {
                    "id": "symbols",
                    "action": "backfill",
                    "collection": {
                        "mode": "per_symbol",
                        "variants": [{"period": "a"}, {"period": "q"}],
                        "date_windows": "year",
                    },
                },
                {
                    "id": "etfs",
                    "action": "snapshot",
                    "collection": {"mode": "per_etf"},
                },
                {
                    "id": "values",
                    "action": "backfill",
                    "collection": {"mode": "per_value", "values": [1, 2, 3]},
                },
                {
                    "id": "discovered",
                    "action": "backfill",
                    "collection": {
                        "mode": "per_discovered",
                        "source_endpoint_id": "source",
                        "source_keys": ["symbol"],
                    },
                },
                {"id": "denied", "action": "not_entitled"},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            connection = sqlite3.connect(str(Path(temporary) / "test.sqlite3"))
            connection.execute(
                "CREATE TABLE fmp_training_facts (endpoint_id TEXT, row_json TEXT)"
            )
            for symbol in ("A", "B", "A"):
                connection.execute(
                    "INSERT INTO fmp_training_facts VALUES (?, ?)",
                    ("source", json.dumps({"symbol": symbol})),
                )
            result = MODULE.estimate_expected_work_items(
                plan, 2, 1, "2025-01-01", "2026-07-14", connection
            )
            connection.close()

        # global 1 + symbols (2*2*2) + ETF 1 + values 3 + discovered 2
        self.assertEqual(result["total"], 15)
        self.assertEqual(result["by_endpoint"]["symbols"], 8)
        self.assertEqual(result["provisional_dimensions"], [])

    def test_percent_never_reaches_100_before_hard_gate(self):
        self.assertEqual(MODULE.calculate_percent(10, 10, False), 99.99)
        self.assertEqual(MODULE.calculate_percent(10, 10, True), 100.0)
        self.assertEqual(MODULE.calculate_percent(0, 0, False), 0.0)

    def test_completion_message_is_explicit(self):
        progress = {
            "percent": 100.0,
            "final_complete": True,
            "checked_at_kst": "2026-07-15T09:00:00+09:00",
            "work_items": {
                "done": 10,
                "expected": 10,
                "not_entitled": 2,
                "failed": 0,
                "processed_percent": 100.0,
            },
            "endpoints": {"completed": 2, "collection_total": 2, "touched": 2},
            "facts": {"rows": 100},
            "service": {"state": "inactive"},
            "disk": {"used_percent": 50.0, "free_bytes": 100 * 1024**3},
            "checkpoint_health": {"failure_groups": []},
            "alert": {"severity": "ok", "issues": []},
            "eta": {"available": False},
        }
        message = MODULE.format_message(progress)
        self.assertIn("다운로드 완료", message)
        self.assertIn("100.00%", message)
        self.assertIn("권한종결 2", message)

    def test_warning_message_names_http_failure_and_disk_risk(self):
        progress = {
            "percent": 6.25,
            "final_complete": False,
            "checked_at_kst": "2026-07-15T23:15:00+09:00",
            "work_items": {
                "done": 168329,
                "expected": 2694803,
                "failed": 7589,
                "processed_percent": 6.53,
            },
            "endpoints": {"completed": 18, "collection_total": 158, "touched": 19},
            "facts": {"rows": 25481062},
            "service": {"state": "active"},
            "disk": {"used_percent": 88.0, "free_bytes": 106 * 1024**3},
            "checkpoint_health": {
                "failure_groups": [
                    {
                        "endpoint_id": "company_information_executive_compensation",
                        "http_status": "402",
                        "count": 7589,
                    }
                ]
            },
            "alert": {
                "severity": "warning",
                "issues": ["checkpoint_failures", "disk_usage_high"],
            },
            "provisional_dimension_count": 0,
            "eta": {
                "available": True,
                "collection_loop_eta_kst": "2026-07-24T18:00+09:00",
                "final_gate_eta_kst": None,
                "final_gate_blockers": ["failed_checkpoints"],
            },
        }
        message = MODULE.format_message(progress)
        self.assertIn("다운로드 경고", message)
        self.assertIn("HTTP402", message)
        self.assertIn("디스크 88.0%", message)
        self.assertIn("checkpoint_failures", message)
        self.assertIn("수집루프 ETA", message)
        self.assertIn("100% ETA 없음", message)

    def test_alert_state_marks_stalled_active_service_critical(self):
        progress = {
            "final_complete": False,
            "work_items": {"failed": 0},
            "service": {"state": "active"},
            "disk": {"used_percent": 50.0, "free_bytes": 100 * 1024**3},
            "checkpoint_health": {"last_checkpoint_age_seconds": 1800},
        }
        alert = MODULE._alert_state(progress)
        self.assertEqual(alert["severity"], "critical")
        self.assertIn("checkpoint_stalled_30m", alert["issues"])

    def test_df_style_disk_percent_excludes_reserved_blocks(self):
        # df reports Use% as used / (used + available), excluding reserved blocks.
        self.assertAlmostEqual(MODULE._df_used_percent(820, 112), 87.98, places=2)

    def test_eta_separates_collection_loop_from_blocked_final_gate(self):
        checked = "2026-07-15T10:00:00+09:00"
        progress = {
            "checked_at_kst": checked,
            "fmp_run": {"started_at_utc": "2026-07-15T00:00:00+00:00"},
            "work_items": {"processed": 3600, "expected": 7200, "failed": 10},
            "provisional_dimension_count": 1,
        }
        eta = MODULE._eta_projection(progress)
        self.assertTrue(eta["available"])
        self.assertAlmostEqual(eta["rate_items_per_second"], 1.0)
        self.assertEqual(eta["remaining_seconds"], 3600.0)
        self.assertEqual(eta["collection_loop_eta_kst"], "2026-07-15T11:00+09:00")
        self.assertIsNone(eta["final_gate_eta_kst"])
        self.assertEqual(
            eta["final_gate_blockers"],
            ["failed_checkpoints", "dynamic_dimensions_pending"],
        )


if __name__ == "__main__":
    unittest.main()
