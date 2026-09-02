from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quant_dataset.corporate_actions import (
    CorporateActionStore,
    _base_record,
    corporate_action_summary,
    normalize_fmp_rows,
    normalize_massive_rows,
)
from quant_dataset.storage import Database, RawStore
from workflows.quant_ai_radar.corporate_actions import (
    load_oracle_corporate_actions,
)


class _Artifact:
    artifact_id = 1
    capture_event_id = 2
    captured_at_utc = "2026-07-30T01:00:00+00:00"


class _OracleRows:
    def __init__(self, rows):
        self.rows = rows

    def corporate_action_rows(self, as_of_date):
        return list(self.rows)


class OracleCorporateActionsTest(unittest.TestCase):
    def test_massive_and_fmp_normalize_to_same_reverse_split(self):
        massive, invalid_massive = normalize_massive_rows(
            [
                {
                    "id": "massive-tza",
                    "ticker": "TZA",
                    "execution_date": "2026-07-15",
                    "split_from": 10,
                    "split_to": 1,
                    "adjustment_type": "reverse_split",
                }
            ],
            artifact=_Artifact(),
            start_date="2026-07-15",
            end_date="2026-07-29",
        )
        fmp, invalid_fmp = normalize_fmp_rows(
            [
                {
                    "symbol": "TZA",
                    "date": "2026-07-15",
                    "numerator": 1,
                    "denominator": 10,
                    "splitType": "stock-split",
                }
            ],
            artifact=_Artifact(),
            symbol="TZA",
            start_date="2026-07-15",
            end_date="2026-07-29",
        )
        self.assertFalse(invalid_massive)
        self.assertFalse(invalid_fmp)
        self.assertEqual(
            (
                massive[0]["symbol"],
                massive[0]["effective_date"],
                massive[0]["old_shares"],
                massive[0]["new_shares"],
            ),
            (
                fmp[0]["symbol"],
                fmp[0]["effective_date"],
                fmp[0]["old_shares"],
                fmp[0]["new_shares"],
            ),
        )
        self.assertEqual(massive[0]["price_factor_for_prior_rows"], 10)
        self.assertEqual(massive[0]["volume_factor_for_prior_rows"], 0.1)
        self.assertEqual(
            massive[0]["available_date"],
            "2026-07-30",
        )

    def test_projection_preserves_earliest_observed_available_date(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = Database(root)
            raw = RawStore(root, database)
            first = raw.store(
                "massive",
                "stock_splits",
                "2026-07",
                b'{"results":[1]}',
                {"method": "GET", "url": "https://api.massive.com/stocks/v1/splits"},
                {"status_code": 200},
            )
            second = raw.store(
                "massive",
                "stock_splits",
                "2026-07",
                b'{"results":[2]}',
                {"method": "GET", "url": "https://api.massive.com/stocks/v1/splits"},
                {"status_code": 200},
            )
            store = CorporateActionStore(database)
            common = {
                "provider": "massive",
                "endpoint_id": "stocks_splits_2",
                "provider_event_id": "tza-2026",
                "symbol": "TZA",
                "effective_date": "2026-07-15",
                "old_shares": 10,
                "new_shares": 1,
                "source_row_index": 0,
                "availability_basis": "first_observed_provider_capture_date",
                "pit_confidence": "capture_date_only",
                "source_type": "structured_provider",
                "source_name": "Massive",
                "source_url": "https://massive.com/docs/rest/stocks/corporate-actions/splits",
            }
            later = _base_record(
                **common,
                available_date="2026-07-30",
                raw_artifact_id=first.artifact_id,
                capture_event_id=first.capture_event_id,
                captured_at_utc=first.captured_at_utc,
            )
            earlier = _base_record(
                **common,
                available_date="2026-07-29",
                raw_artifact_id=second.artifact_id,
                capture_event_id=second.capture_event_id,
                captured_at_utc=second.captured_at_utc,
            )
            store.ingest([later])
            store.ingest([earlier])
            with database.connect() as connection:
                row = connection.execute(
                    "SELECT available_date FROM corporate_actions"
                ).fetchone()
            hidden = corporate_action_summary(database.db_path, "2026-07-28")
            visible = corporate_action_summary(database.db_path, "2026-07-29")
        self.assertEqual(row["available_date"], "2026-07-29")
        self.assertEqual(hidden["visible_record_count"], 0)
        self.assertEqual(visible["visible_record_count"], 1)
        self.assertEqual(visible["version_count"], 2)

    def test_oracle_loader_accepts_official_or_two_provider_evidence_only(self):
        base = {
            "symbol": "TZA",
            "action_type": "reverse_split",
            "effective_date": "2026-07-15",
            "available_date": "2026-07-15",
            "announcement_date": None,
            "old_shares": 10,
            "new_shares": 1,
            "source_name": "",
            "source_url": "https://example.test/splits",
            "payload_sha256": "a" * 64,
        }
        cross = [
            {
                **base,
                "provider": provider,
                "source_type": "structured_provider",
            }
            for provider in ("massive", "fmp")
        ]
        single = {
            **base,
            "symbol": "SOLO",
            "provider": "massive",
            "source_type": "structured_provider",
        }
        official = {
            **base,
            "symbol": "OFF",
            "provider": "official_issuer",
            "source_type": "official_issuer",
            "available_date": "2026-06-01",
        }
        ledger = load_oracle_corporate_actions(
            _OracleRows([*cross, single, official]),
            as_of_date="2026-07-29",
        )
        self.assertEqual(
            [row["symbol"] for row in ledger["events"]],
            ["OFF", "TZA"],
        )
        self.assertEqual(
            ledger["events_by_symbol"]["TZA"][0]["verification_status"],
            "cross_provider",
        )
        self.assertEqual(
            ledger["events_by_symbol"]["OFF"][0]["verification_status"],
            "official",
        )


if __name__ == "__main__":
    unittest.main()
