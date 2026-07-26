from __future__ import annotations

import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quant_dataset.config import CredentialSet, load_credentials
from quant_dataset.pipeline import DatasetPipeline
from quant_dataset.rate_limit import FileWindowRateLimiter, RateLimitSpec
from quant_dataset.storage import Database, RawStore, redacted_request_metadata


class FakeResponse:
    def __init__(self, document, status_code=200, headers=None):
        self.content = json.dumps(document, separators=(",", ":")).encode("utf-8")
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}


class FakeSession:
    def __init__(self, fmp_rows=None, massive_by_date=None):
        self.fmp_rows = list(fmp_rows or [])
        self.massive_by_date = dict(massive_by_date or {})
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        call = {
            "url": url,
            "params": dict(params or {}),
            "headers": dict(headers or {}),
            "timeout": timeout,
        }
        self.calls.append(call)
        if "financialmodelingprep" in url:
            symbol = call["params"]["symbol"]
            rows = [row for row in self.fmp_rows if row.get("symbol", symbol) == symbol]
            return FakeResponse(rows)
        trade_date = url.rsplit("/", 1)[-1]
        return FakeResponse(
            {
                "status": "OK",
                "resultsCount": len(self.massive_by_date.get(trade_date, [])),
                "results": self.massive_by_date.get(trade_date, []),
            }
        )


def fmp_row(trade_date="2024-01-02", close=105.0):
    return {
        "symbol": "AAPL",
        "date": trade_date,
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": close,
        "adjClose": close,
        "volume": 1000,
        "vwap": 103.0,
    }


def massive_rows(close=105.1):
    return [
        {"T": "AAPL", "o": 100.01, "h": max(110.01, close + 1.0), "l": 95.01, "c": close, "v": 1050, "vw": 103.2, "n": 42, "t": 1},
        {"T": "MSFT", "o": 200.0, "h": 205.0, "l": 198.0, "c": 203.0, "v": 500, "vw": 202.0, "n": 20, "t": 2},
    ]


class QuantDatasetPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "QUANT_DATASET"
        self.credentials = CredentialSet("FMP_TEST_KEY", "MASSIVE_TEST_KEY", "test", "test")

    def tearDown(self):
        self.temp.cleanup()

    def pipeline(self, session):
        return DatasetPipeline(
            self.root,
            self.credentials,
            session=session,
            retries=0,
            sleep=lambda _: None,
            rate_limiters={},
        )

    def test_credentials_environment_precedes_secrets_file(self):
        secrets = Path(self.temp.name) / "secrets.env"
        secrets.write_text("FMP_API_KEY=file-fmp\nMASSIVE_API_KEY=file-massive\n", encoding="utf-8")
        credentials = load_credentials(
            {"FMP_API_KEY": "env-fmp", "MASSIVE_API_KEY": ""}, secrets
        )
        self.assertEqual(credentials.fmp_api_key, "env-fmp")
        self.assertEqual(credentials.massive_api_key, "file-massive")
        self.assertNotIn("env-fmp", repr(credentials))
        self.assertNotIn("file-massive", repr(credentials))

    def test_raw_store_is_immutable_redacted_and_records_each_capture_event(self):
        database = Database(self.root)
        store = RawStore(self.root, database)
        request = redacted_request_metadata(
            "GET",
            "https://example.test/data?apikey=secret-in-url",
            {"apiKey": "secret-in-param", "symbol": "AAPL"},
            {"symbol": "AAPL"},
        )
        first = store.store("test", "daily", "2024-01-02", b'{"ok":true}', request, {"status_code": 200})
        second = store.store("test", "daily", "2024-01-02", b'{"ok":true}', request, {"status_code": 200})
        self.assertEqual(first.artifact_id, second.artifact_id)
        self.assertNotEqual(first.capture_event_id, second.capture_event_id)
        self.assertEqual(gzip.decompress(first.raw_path.read_bytes()), b'{"ok":true}')
        metadata_text = first.metadata_path.read_text(encoding="utf-8")
        self.assertNotIn("secret-in-url", metadata_text)
        self.assertNotIn("secret-in-param", metadata_text)
        self.assertEqual(database.counts()["raw_artifacts"], 1)
        self.assertEqual(database.counts()["capture_events"], 2)
        self.assertTrue(store.verify_all()["ok"])

    def test_daily_capture_keeps_sources_separate_and_passes_qc(self):
        session = FakeSession(
            fmp_rows=[fmp_row()], massive_by_date={"2024-01-02": massive_rows()}
        )
        pipeline = self.pipeline(session)
        result = pipeline.capture_daily("2024-01-02", ["AAPL"])
        self.assertTrue(result["ok"])
        counts = pipeline.database.counts()
        self.assertEqual(counts["daily_observations"], 3)
        self.assertEqual(counts["daily_observation_versions"], 3)
        self.assertEqual(counts["observations_by_source"], {"fmp": 1, "massive": 2})
        self.assertEqual(
            pipeline.database.quality_for_pair("AAPL", "2024-01-02")["status"], "pass"
        )
        self.assertEqual(
            pipeline.database.quality_for_pair("MSFT", "2024-01-02")["status"],
            "single_source",
        )
        fmp_call = next(call for call in session.calls if "financialmodelingprep" in call["url"])
        massive_call = next(call for call in session.calls if "api.massive.com" in call["url"])
        self.assertNotIn("apikey", {key.lower() for key in fmp_call["params"]})
        self.assertEqual(fmp_call["headers"]["apikey"], "FMP_TEST_KEY")
        self.assertNotIn("apiKey", massive_call["params"])
        self.assertEqual(massive_call["headers"]["Authorization"], "Bearer MASSIVE_TEST_KEY")
        for metadata in self.root.glob("raw/**/*.metadata.json"):
            text = metadata.read_text(encoding="utf-8")
            self.assertNotIn("FMP_TEST_KEY", text)
            self.assertNotIn("MASSIVE_TEST_KEY", text)

    def test_massive_provider_preserves_case_sensitive_symbols(self):
        case_rows = [
            {"T": "TPC", "o": 74.82, "h": 76.5, "l": 73.5, "c": 75.89, "v": 342584, "vw": 75.91, "n": 8562, "t": 1},
            {"T": "TpC", "o": 17.56, "h": 17.59, "l": 17.43, "c": 17.43, "v": 144862, "vw": 17.47, "n": 1265, "t": 2},
            {"T": "BCPC", "o": 164.19, "h": 165.46, "l": 161.91, "c": 162.15, "v": 249120, "vw": 163.25, "n": 7975, "t": 3},
            {"T": "BCpC", "o": 24.03, "h": 24.09, "l": 23.99, "c": 24.06, "v": 5360, "vw": 24.04, "n": 61, "t": 4},
        ]
        pipeline = self.pipeline(
            FakeSession(massive_by_date={"2024-01-02": case_rows})
        )
        result = pipeline.capture_daily(
            "2024-01-02", [], source="massive"
        )
        self.assertTrue(result["ok"])
        with pipeline.database.connect() as connection:
            symbols = {
                row["symbol"]
                for row in connection.execute(
                    "SELECT symbol FROM daily_observations ORDER BY symbol"
                )
            }
        self.assertEqual(symbols, {"TPC", "TpC", "BCPC", "BCpC"})
        self.assertEqual(pipeline.database.counts()["daily_observations"], 4)

    def test_hard_cross_source_mismatch_fails_verify(self):
        session = FakeSession(
            fmp_rows=[fmp_row(close=105.0)],
            massive_by_date={"2024-01-02": massive_rows(close=160.0)},
        )
        pipeline = self.pipeline(session)
        pipeline.capture_daily("2024-01-02", ["AAPL"])
        self.assertEqual(
            pipeline.database.quality_for_pair("AAPL", "2024-01-02")["status"], "fail"
        )
        report = pipeline.verify("2024-01-02", "2024-01-02", ["AAPL"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["quality_recomputed"]["fail"], 1)

    def test_backfill_checkpoints_skip_completed_work(self):
        session = FakeSession(
            fmp_rows=[fmp_row("2024-01-02"), fmp_row("2024-01-03")],
            massive_by_date={
                "2024-01-02": massive_rows(),
                "2024-01-03": massive_rows(),
            },
        )
        pipeline = self.pipeline(session)
        first = pipeline.backfill("2024-01-02", "2024-01-03", ["AAPL"])
        call_count = len(session.calls)
        second = pipeline.backfill("2024-01-02", "2024-01-03", ["AAPL"])
        self.assertTrue(first["ok"])
        self.assertEqual(call_count, 3)  # one FMP range, two Massive dates
        self.assertEqual(len(session.calls), call_count)
        self.assertEqual(second["skipped"], 3)

    def test_fmp_empty_range_is_a_completed_empty_checkpoint(self):
        session = FakeSession(fmp_rows=[])
        pipeline = self.pipeline(session)
        first = pipeline.backfill(
            "2017-01-01", "2026-07-14", ["EMPTY"], source="fmp"
        )
        call_count = len(session.calls)
        second = pipeline.backfill(
            "2017-01-01", "2026-07-14", ["EMPTY"], source="fmp"
        )
        self.assertTrue(first["ok"])
        self.assertEqual(first["empty"], 1)
        self.assertEqual(first["failed"], 0)
        self.assertEqual(second["skipped"], 1)
        self.assertEqual(len(session.calls), call_count)

    def test_fmp_completed_item_is_reused_across_universe_contracts(self):
        session = FakeSession(fmp_rows=[fmp_row("2024-01-02")])
        pipeline = self.pipeline(session)
        first = pipeline.backfill(
            "2024-01-01",
            "2024-01-31",
            ["AAPL"],
            source="fmp",
            symbol_universe={"sha256": "first"},
        )
        call_count = len(session.calls)
        second = pipeline.backfill(
            "2024-01-01",
            "2024-01-31",
            ["AAPL"],
            source="fmp",
            symbol_universe={"sha256": "second"},
        )
        self.assertNotEqual(first["job_id"], second["job_id"])
        self.assertEqual(second["skipped"], 1)
        self.assertEqual(len(session.calls), call_count)

    def test_export_is_deterministic_unlabeled_and_asof_bounded(self):
        session = FakeSession(
            fmp_rows=[fmp_row("2024-01-02"), fmp_row("2024-01-03")],
            massive_by_date={
                "2024-01-02": massive_rows(),
                "2024-01-03": massive_rows(),
            },
        )
        pipeline = self.pipeline(session)
        pipeline.backfill("2024-01-02", "2024-01-03", ["AAPL"])
        first = self.root / "training_packets" / "first.jsonl"
        second = self.root / "training_packets" / "second.jsonl"
        result_one = pipeline.export_packets(
            "2024-01-02", "2024-01-03", first, ["AAPL"], lookback_days=20
        )
        result_two = pipeline.export_packets(
            "2024-01-02", "2024-01-03", second, ["AAPL"], lookback_days=20
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(result_one["sha256"], result_two["sha256"])
        packets = [json.loads(line) for line in first.read_text().splitlines()]
        self.assertEqual(len(packets), 2)
        forbidden = ("expert_answer", "answer", "label", "future_return", "recommendation")
        rendered = json.dumps(packets, sort_keys=True).lower()
        for key in forbidden:
            self.assertNotIn('"{}"'.format(key), rendered)
        for packet in packets:
            self.assertTrue(all(row["trade_date"] <= packet["as_of_date"] for row in packet["history"]))
            self.assertFalse(packet["provenance"]["historical_backfill_is_true_point_in_time"])

    def test_preflight_surfaces_layer_one_and_fmp_bulk_402(self):
        pipeline = self.pipeline(FakeSession())
        report = pipeline.preflight()
        self.assertTrue(report["ok"])
        bulk = report["source_capability_policy"]["fmp_eod_bulk"]
        self.assertFalse(bulk["enabled"])
        self.assertEqual(bulk["latest_observed_http_status"], 402)
        manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["implemented_scope"],
            "daily_backbone_plus_etf_relations_and_fmp_training_features",
        )
        self.assertEqual(report["network_requests_performed"], 0)

    def test_file_rate_limiter_shares_a_rolling_window(self):
        now = [100.0]
        sleeps = []

        def clock():
            return now[0]

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        limiter = FileWindowRateLimiter(
            Path(self.temp.name) / "limit.json",
            RateLimitSpec("test", 2, 1.0, "unit_test"),
            sleep=sleep,
            clock=clock,
        )
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        self.assertTrue(sleeps)
        self.assertGreaterEqual(now[0], 101.0)

    def test_fmp_429_honors_long_retry_after(self):
        class RetrySession:
            def __init__(self):
                self.responses = [
                    FakeResponse([], status_code=429, headers={"retry-after": "60"}),
                    FakeResponse([fmp_row()]),
                ]

            def get(self, url, params=None, headers=None, timeout=None):
                return self.responses.pop(0)

        sleeps = []
        pipeline = DatasetPipeline(
            self.root,
            self.credentials,
            session=RetrySession(),
            retries=1,
            sleep=sleeps.append,
            rate_limiters={},
        )
        result = pipeline.capture_daily(
            "2024-01-02", ["AAPL"], source="fmp"
        )
        self.assertTrue(result["ok"])
        self.assertEqual(sleeps, [60.0])


if __name__ == "__main__":
    unittest.main()
