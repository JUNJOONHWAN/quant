from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workflows.quant_ai_radar.run_quant_ai_radar import (
    _archive_changed_source_run,
)
from workflows.quant_ai_radar.run_queue import RadarQueue


class QuantAiRadarSourceRevisionTest(unittest.TestCase):
    def test_changed_source_archives_old_queue_without_deleting_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "runs" / "2026-07-30"
            run_dir.mkdir(parents=True)
            queue = RadarQueue(run_dir / "selected_run_queue.sqlite3")
            queue.bind_metadata(
                {
                    "dataset_source_fingerprint_sha256": "a" * 64,
                    "as_of_date": "2026-07-30",
                }
            )
            report_path = run_dir / "market_report.json"
            report_path.write_text(
                json.dumps({"as_of_date": "2026-07-30"}),
                encoding="utf-8",
            )

            result = _archive_changed_source_run(
                run_dir,
                source_fingerprint_sha256="b" * 64,
            )

            archive_dir = Path(result["archive_dir"])
            self.assertEqual(result["status"], "archived")
            self.assertTrue(
                (archive_dir / "selected_run_queue.sqlite3").is_file()
            )
            self.assertTrue(
                (archive_dir / "revision_manifest.json").is_file()
            )
            self.assertFalse(
                (run_dir / "selected_run_queue.sqlite3").exists()
            )
            self.assertTrue(report_path.is_file())

    def test_unchanged_source_reuses_existing_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            queue = RadarQueue(run_dir / "selected_run_queue.sqlite3")
            queue.bind_metadata(
                {
                    "dataset_source_fingerprint_sha256": "c" * 64,
                }
            )
            result = _archive_changed_source_run(
                run_dir,
                source_fingerprint_sha256="c" * 64,
            )
            self.assertEqual(result["status"], "reused")
            self.assertTrue(
                (run_dir / "selected_run_queue.sqlite3").is_file()
            )


if __name__ == "__main__":
    unittest.main()
