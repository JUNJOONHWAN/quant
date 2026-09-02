from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quant_dataset.config import CredentialSet
from quant_dataset.fmp_training import FmpWorkItem
from quant_dataset.pipeline import DatasetPipeline


STATUS_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fmp_training_status.py"
STATUS_SPEC = importlib.util.spec_from_file_location("fmp_training_status", STATUS_SCRIPT)
STATUS_MODULE = importlib.util.module_from_spec(STATUS_SPEC)
assert STATUS_SPEC.loader is not None
STATUS_SPEC.loader.exec_module(STATUS_MODULE)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.content = json.dumps(payload).encode("utf-8")
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}


class FeatureSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params, dict(headers or {})))
        if url.endswith("/stable/financial-statement-symbol-list"):
            return FakeResponse([{"symbol": "AAPL"}, {"symbol": "MSFT"}])
        if url.endswith("/stable/income-statement"):
            symbol = params["symbol"]
            return FakeResponse(
                [
                    {
                        "symbol": symbol,
                        "date": "2024-09-30",
                        "acceptedDate": "2024-10-30 16:01:02",
                        "period": params.get("period", "FY"),
                        "revenue": 100,
                    }
                ]
            )
        raise AssertionError(url)


class FmpTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "data"
        self.symbols = Path(self.temp.name) / "symbols.txt"
        self.symbols.write_text("AAPL\nMSFT\n", encoding="utf-8")
        self.universe = Path(self.temp.name) / "universe.jsonl"
        self.universe.write_text(
            json.dumps({"symbol": "SPY", "is_etf": True}) + "\n",
            encoding="utf-8",
        )
        self.plan = Path(self.temp.name) / "plan.json"
        self.plan.write_text(
            json.dumps(
                {
                    "endpoints": [
                        {
                            "id": "stock_directory_financial_statement_symbols_list",
                            "path": "/stable/financial-statement-symbol-list",
                            "action": "backfill",
                            "collection": {"mode": "global"},
                        },
                        {
                            "id": "statements_income_statement",
                            "path": "/stable/income-statement",
                            "action": "backfill",
                            "collection": {
                                "mode": "per_discovered",
                                "dimension_param": "symbol",
                                "source_endpoint_id": "stock_directory_financial_statement_symbols_list",
                                "source_keys": ["symbol"],
                                "variants": [
                                    {"period": "annual"},
                                    {"period": "quarter"},
                                ],
                            },
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_backfill_discovers_dimensions_preserves_dates_and_resumes(self):
        session = FeatureSession()
        pipeline = DatasetPipeline(
            self.root,
            CredentialSet("FMP_TEST", "MASSIVE_TEST", "test", "test"),
            session=session,
            retries=0,
            rate_limiters={},
        )
        first = pipeline.backfill_fmp_training(
            self.plan,
            self.symbols,
            self.universe,
            "2017-01-01",
            "2026-07-14",
        )
        call_count = len(session.calls)
        second = pipeline.backfill_fmp_training(
            self.plan,
            self.symbols,
            self.universe,
            "2017-01-01",
            "2026-07-14",
        )
        self.assertTrue(first["ok"])
        self.assertEqual(first["done"], 5)
        self.assertEqual(second["skipped"], 5)
        self.assertEqual(len(session.calls), call_count)
        with pipeline.database.connect() as connection:
            facts = connection.execute(
                "SELECT * FROM fmp_training_facts "
                "WHERE endpoint_id='statements_income_statement'"
            ).fetchall()
        self.assertEqual(len(facts), 4)
        self.assertEqual({row["available_date"] for row in facts}, {"2024-10-30"})
        self.assertEqual({row["event_date"] for row in facts}, {"2024-09-30"})
        self.assertTrue(pipeline.fmp_training.verify()["ok"])
        for _, _, headers in session.calls:
            self.assertEqual(headers["apikey"], "FMP_TEST")
        for path in self.root.glob("raw/**/*.metadata.json"):
            self.assertNotIn("FMP_TEST", path.read_text(encoding="utf-8"))

    def test_pagination_resumes_from_first_uncaptured_page(self):
        class PagedSession:
            def __init__(self):
                self.pages = []

            def get(self, url, params=None, headers=None, timeout=None):
                page = int((params or {}).get("page", 0))
                self.pages.append(page)
                return FakeResponse([{"cik": str(page)}] if page < 2 else [])

        session = PagedSession()
        pipeline = DatasetPipeline(
            self.root,
            CredentialSet("FMP_TEST", "MASSIVE_TEST", "test", "test"),
            session=session,
            retries=0,
            rate_limiters={},
        )
        first = FmpWorkItem(
            endpoint_id="stock_directory_cik_list",
            path="/stable/cik-list",
            entity_key="scope=global",
            params={},
            pagination=True,
            page_size=1,
            max_pages=1,
        )
        pipeline.fmp_training._capture_item(first)
        second = FmpWorkItem(
            endpoint_id="stock_directory_cik_list",
            path="/stable/cik-list",
            entity_key="scope=global",
            params={},
            pagination=True,
            page_size=1,
            max_pages=3,
        )
        _, count = pipeline.fmp_training._capture_item(second)
        self.assertEqual(session.pages, [0, 1, 2])
        self.assertEqual(count, 2)

    def test_live_402_is_source_preserved_terminal_and_resumes_without_recall(self):
        class EntitlementSession:
            def __init__(self):
                self.calls = []

            def get(self, url, params=None, headers=None, timeout=None):
                symbol = str((params or {}).get("symbol"))
                self.calls.append(symbol)
                if symbol == "MSFT":
                    return FakeResponse(
                        {"Error Message": "value is not available under current subscription"},
                        status_code=402,
                    )
                return FakeResponse([{"symbol": symbol, "date": "2024-01-01"}])

        self.plan.write_text(
            json.dumps(
                {
                    "endpoint_count": 1,
                    "action_counts": {"backfill": 1},
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
        session = EntitlementSession()
        pipeline = DatasetPipeline(
            self.root,
            CredentialSet("FMP_TEST", "MASSIVE_TEST", "test", "test"),
            session=session,
            retries=0,
            rate_limiters={},
        )
        first = pipeline.backfill_fmp_training(
            self.plan, self.symbols, self.universe, "2017-01-01", "2026-07-14"
        )
        call_count = len(session.calls)
        second = pipeline.backfill_fmp_training(
            self.plan, self.symbols, self.universe, "2017-01-01", "2026-07-14"
        )
        self.assertTrue(first["ok"])
        self.assertEqual(first["done"], 1)
        self.assertEqual(first["not_entitled"], 1)
        self.assertEqual(first["failed"], 0)
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(second["skipped_not_entitled"], 1)
        self.assertEqual(len(session.calls), call_count)
        with pipeline.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.status, c.raw_artifact_id, r.response_status
                FROM checkpoints c
                LEFT JOIN raw_artifacts r ON r.id=c.raw_artifact_id
                WHERE c.source='fmp_training' ORDER BY c.status
                """
            ).fetchall()
        self.assertEqual(
            [(row["status"], row["response_status"]) for row in rows],
            [("done", 200), ("not_entitled", 402)],
        )
        with mock.patch.object(STATUS_MODULE, "_service_state", return_value="inactive"):
            status = STATUS_MODULE.build_status(self.root, self.plan, "dummy.service")
        self.assertTrue(status["overall_complete"])
        self.assertEqual(status["not_entitled_evidence"]["valid"], 1)
        self.assertEqual(status["not_entitled_evidence"]["invalid"], 0)
        with pipeline.database.connect() as connection:
            connection.execute(
                "UPDATE raw_artifacts SET response_status=200 WHERE response_status=402"
            )
        with mock.patch.object(STATUS_MODULE, "_service_state", return_value="inactive"):
            invalid = STATUS_MODULE.build_status(self.root, self.plan, "dummy.service")
        self.assertFalse(invalid["overall_complete"])
        self.assertEqual(invalid["not_entitled_evidence"]["invalid"], 1)

    def test_non_entitlement_http_error_remains_failed(self):
        class ErrorSession:
            def get(self, url, params=None, headers=None, timeout=None):
                return FakeResponse({"error": "bad request"}, status_code=400)

        self.plan.write_text(
            json.dumps(
                {
                    "endpoints": [
                        {
                            "id": "feature",
                            "path": "/stable/feature",
                            "action": "backfill",
                            "collection": {"mode": "global"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        pipeline = DatasetPipeline(
            self.root,
            CredentialSet("FMP_TEST", "MASSIVE_TEST", "test", "test"),
            session=ErrorSession(),
            retries=0,
            rate_limiters={},
        )
        result = pipeline.backfill_fmp_training(
            self.plan, self.symbols, self.universe, "2017-01-01", "2026-07-14"
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["not_entitled"], 0)
        self.assertEqual(result["failed"], 1)


if __name__ == "__main__":
    unittest.main()
