import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from workflows.market_structure_oracle import incremental_store


class _Http:
    def __init__(self):
        self.calls = []

    def get_json(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["dataset"] == "stable_delisted_companies":
            rows = [{
                "symbol": "OLD", "companyName": "Old Co", "exchange": "NASDAQ",
                "ipoDate": "2010-01-01", "delistedDate": "2026-07-20",
            }]
            artifact_id = 11
        else:
            rows = [{
                "symbol": "BUY", "companyName": "Buyer", "targetedSymbol": "TGT",
                "targetedCompanyName": "Target", "transactionDate": "2026-07-21",
                "acceptedDate": "2026-07-22 09:00:00", "link": "https://www.sec.gov/example",
            }]
            artifact_id = 12
        return SimpleNamespace(
            document=rows,
            artifact=SimpleNamespace(
                artifact_id=artifact_id,
                capture_event_id=artifact_id + 100,
                captured_at_utc="2026-07-23T00:00:00+00:00",
            ),
        )


class OracleUltimateLifecycleRegistryTests(unittest.TestCase):
    def test_registers_delisting_and_merger_as_pit_versions(self):
        with tempfile.TemporaryDirectory() as root:
            database = Path(root) / "oracle.sqlite3"
            incremental_store._initialize_oracle_tables(database)
            pipeline = SimpleNamespace(
                credentials=SimpleNamespace(fmp_api_key="test-key"),
                database=SimpleNamespace(db_path=database),
                http=_Http(),
            )
            result = incremental_store._capture_fmp_lifecycle_events(
                pipeline=pipeline, target="2026-07-31"
            )
            self.assertEqual(result["normalized_record_count"], 2)
            self.assertEqual(result["visible_as_of_count"], 2)
            with sqlite3.connect(database) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT event_type,symbol,related_symbol FROM oracle_lifecycle_events "
                        "ORDER BY event_type"
                    ).fetchall(),
                    [("delisted", "OLD", None), ("merger_acquisition", "BUY", "TGT")],
                )


if __name__ == "__main__":
    unittest.main()
