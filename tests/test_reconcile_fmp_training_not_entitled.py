from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from quant_dataset.config import CredentialSet
from quant_dataset.pipeline import DatasetPipeline


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reconcile_fmp_training_not_entitled.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reconcile_fmp_training_not_entitled", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeResponse:
    status_code = 402
    headers = {"content-type": "application/json"}
    content = json.dumps({"error": "subscription"}).encode("utf-8")


class Session:
    def get(self, url, params=None, headers=None, timeout=None):
        return FakeResponse()


class ReconcileNotEntitledTest(unittest.TestCase):
    def test_dry_run_and_apply_require_matching_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            symbols = Path(temporary) / "symbols.txt"
            symbols.write_text("AAPL\n", encoding="utf-8")
            universe = Path(temporary) / "universe.jsonl"
            universe.write_text("", encoding="utf-8")
            plan = Path(temporary) / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "endpoints": [
                            {
                                "id": "feature",
                                "path": "/stable/feature",
                                "action": "backfill",
                                "collection": {
                                    "mode": "per_symbol",
                                    "dimension_param": "symbol",
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pipeline = DatasetPipeline(
                root,
                CredentialSet("FMP_TEST", "MASSIVE_TEST", "test", "test"),
                session=Session(),
                retries=0,
                rate_limiters={},
            )
            result = pipeline.backfill_fmp_training(
                plan, symbols, universe, "2017-01-01", "2026-07-14"
            )
            job_id = result["job_id"]
            with pipeline.database.connect() as connection:
                connection.execute(
                    """
                    UPDATE checkpoints
                    SET status='failed', raw_artifact_id=NULL, observation_count=NULL
                    WHERE job_id=?
                    """,
                    (job_id,),
                )
            dry_run = MODULE.reconcile(pipeline.database.db_path, job_id, apply=False)
            self.assertEqual(dry_run["eligible"], 1)
            self.assertEqual(dry_run["invalid"], 0)
            self.assertEqual(dry_run["updated"], 0)
            applied = MODULE.reconcile(pipeline.database.db_path, job_id, apply=True)
            self.assertEqual(applied["updated"], 1)
            with pipeline.database.connect() as connection:
                row = connection.execute(
                    "SELECT status, raw_artifact_id, observation_count FROM checkpoints"
                ).fetchone()
            self.assertEqual(row["status"], "not_entitled")
            self.assertIsNotNone(row["raw_artifact_id"])
            self.assertEqual(row["observation_count"], 0)


if __name__ == "__main__":
    unittest.main()
