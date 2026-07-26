from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from quant_dataset.config import CredentialSet
from quant_dataset.pipeline import DatasetPipeline
from quant_dataset.point_in_time import ETF_FLOW_PIT_FILTER, ETF_FLOW_POLICY_ID


class FakeResponse:
    def __init__(self, document, status_code=200, headers=None):
        self.content = json.dumps(document, separators=(",", ":")).encode("utf-8")
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}


class ScriptedSession:
    def __init__(self, flow_responses=None, massive_by_date=None):
        self.flow_responses = list(flow_responses or [])
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
        if "/etf-global/v1/fund-flows" in url:
            if not self.flow_responses:
                raise AssertionError("unexpected ETF flow request")
            response = self.flow_responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return FakeResponse(response)
        if "/v2/aggs/grouped/locale/us/market/stocks/" in url:
            trade_date = url.rsplit("/", 1)[-1]
            rows = self.massive_by_date.get(trade_date, [])
            return FakeResponse(
                {"status": "OK", "resultsCount": len(rows), "results": rows}
            )
        raise AssertionError("unexpected URL: {}".format(url))


def flow_row(
    ticker="QQQ",
    effective_date="2024-01-02",
    processed_date="2024-01-03",
    fund_flow=100.0,
):
    return {
        "composite_ticker": ticker,
        "effective_date": effective_date,
        "processed_date": processed_date,
        "fund_flow": fund_flow,
        "nav": 400.0,
        "shares_outstanding": 1_000_000.0,
    }


def grouped_row(ticker="QQQ", close=400.0):
    return {
        "T": ticker,
        "o": close - 1.0,
        "h": close + 2.0,
        "l": close - 2.0,
        "c": close,
        "v": 1_000_000,
        "vw": close - 0.25,
        "n": 1000,
        "t": 1,
    }


class EtfFlowLayerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "QUANT_DATASET"
        self.credentials = CredentialSet(None, "MASSIVE_TEST_KEY", "missing", "test")

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

    def test_pagination_header_auth_versions_and_latest_projection(self):
        session = ScriptedSession(
            flow_responses=[
                {
                    "results": [flow_row(processed_date="2024-01-02", fund_flow=100.0)],
                    "next_url": "https://api.massive.com/etf-global/v1/fund-flows?cursor=page2",
                },
                {
                    "results": [
                        flow_row(processed_date="2024-01-03", fund_flow=125.0),
                        flow_row("SPY", "2024-01-02", "2024-01-03", 75.0),
                    ]
                },
            ]
        )
        pipeline = self.pipeline(session)
        result = pipeline.capture_etf_flows(
            "2024-01-03", lookback_days=3, tickers=["QQQ", "SPY"]
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["page_count"], 2)
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["freshness_status"], "fresh")
        self.assertEqual(len(session.calls), 2)
        for call in session.calls:
            self.assertEqual(
                call["headers"]["Authorization"], "Bearer MASSIVE_TEST_KEY"
            )
            self.assertNotIn("apikey", {key.lower() for key in call["params"]})
            self.assertNotIn("MASSIVE_TEST_KEY", call["url"])
        first = session.calls[0]
        self.assertEqual(first["params"]["processed_date.gte"], "2024-01-01")
        self.assertEqual(first["params"]["processed_date.lte"], "2024-01-03")
        self.assertEqual(first["params"]["composite_ticker.any_of"], "QQQ,SPY")
        self.assertEqual(parse_qs(urlsplit(session.calls[1]["url"]).query)["cursor"], ["page2"])

        with pipeline.database.connect() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM etf_flow_versions"
                ).fetchone()["n"],
                3,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM etf_flow_observations"
                ).fetchone()["n"],
                2,
            )
            qqq = connection.execute(
                "SELECT * FROM etf_flow_observations WHERE ticker='QQQ'"
            ).fetchone()
            self.assertEqual(qqq["processed_date"], "2024-01-03")
            self.assertEqual(qqq["fund_flow"], 125.0)
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM etf_flow_latest"
                ).fetchone()["n"],
                2,
            )
        self.assertEqual(pipeline.database.counts()["capture_events"], 2)
        for metadata in self.root.glob("raw/**/*.metadata.json"):
            self.assertNotIn(
                "MASSIVE_TEST_KEY", metadata.read_text(encoding="utf-8")
            )

    def test_failed_second_page_resumes_without_repeating_first_page(self):
        session = ScriptedSession(
            flow_responses=[
                {
                    "results": [flow_row(processed_date="2024-01-02")],
                    "next_url": "https://api.massive.com/etf-global/v1/fund-flows?cursor=resume",
                },
                RuntimeError("temporary transport failure"),
                {"results": [flow_row("SPY", processed_date="2024-01-03")]},
            ]
        )
        pipeline = self.pipeline(session)
        with self.assertRaises(Exception):
            pipeline.capture_etf_flows("2024-01-03", lookback_days=3)

        with pipeline.database.connect() as connection:
            failed = connection.execute(
                "SELECT * FROM etf_flow_runs"
            ).fetchone()
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["page_count"], 1)
            run_id = failed["run_id"]

        resumed = pipeline.capture_etf_flows("2024-01-03", lookback_days=3)
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["run_id"], run_id)
        self.assertEqual(resumed["page_count"], 2)
        self.assertEqual(len(session.calls), 3)
        self.assertNotIn("cursor=resume", session.calls[0]["url"])
        self.assertIn("cursor=resume", session.calls[1]["url"])
        self.assertIn("cursor=resume", session.calls[2]["url"])

    def test_stale_normalized_hash_is_detected_across_capture_dates(self):
        first_document = {
            "request_id": "one",
            "results": [flow_row(processed_date="2024-01-01")],
        }
        second_document = {
            "request_id": "two",
            "results": [flow_row(processed_date="2024-01-01")],
        }
        pipeline = self.pipeline(
            ScriptedSession(flow_responses=[first_document, second_document])
        )
        first = pipeline.capture_etf_flows(
            "2024-01-10", lookback_days=30, max_lag_days=2
        )
        second = pipeline.capture_etf_flows(
            "2024-01-11", lookback_days=30, max_lag_days=2
        )

        self.assertEqual(first["freshness_status"], "stale_source_date")
        self.assertEqual(second["freshness_status"], "stale_repeated_hash")
        self.assertFalse(second["repeated_payload_hash"])
        self.assertTrue(second["repeated_normalized_hash"])
        audit = pipeline.verify()
        self.assertFalse(audit["ok"])
        self.assertEqual(
            audit["etf_flow"]["errors"][0]["error"], "etf_flow_freshness_gate"
        )

    def test_export_packet_applies_two_session_effective_date_gate(self):
        session = ScriptedSession(
            flow_responses=[
                {
                    "results": [
                        flow_row(
                            "QQQ",
                            effective_date="2024-01-02",
                            processed_date="2024-01-02",
                        )
                    ]
                }
            ],
            massive_by_date={
                "2024-01-02": [grouped_row(close=398.0)],
                "2024-01-03": [grouped_row(close=400.0)],
                "2024-01-04": [grouped_row(close=402.0)],
            },
        )
        pipeline = self.pipeline(session)
        pipeline.capture_etf_flows("2024-01-02", lookback_days=5)
        pipeline.capture_daily("2024-01-02", [], source="massive")
        pipeline.capture_daily("2024-01-03", [], source="massive")
        pipeline.capture_daily("2024-01-04", [], source="massive")
        output = self.root / "training_packets" / "qqq.jsonl"
        pipeline.export_packets(
            "2024-01-02", "2024-01-04", output, symbols=["QQQ"]
        )
        packets = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(packets), 3)
        self.assertIsNone(packets[0]["etf_flow"]["latest"])
        self.assertIsNone(packets[1]["etf_flow"]["latest"])
        self.assertEqual(
            packets[2]["etf_flow"]["latest"]["processed_date"], "2024-01-02"
        )
        self.assertEqual(
            packets[2]["etf_flow"]["latest"]["training_available_session_date"],
            "2024-01-04",
        )
        self.assertEqual(packets[2]["etf_flow"]["pit_filter"], ETF_FLOW_PIT_FILTER)
        self.assertEqual(
            packets[2]["etf_flow"]["availability_policy"]["policy_id"],
            ETF_FLOW_POLICY_ID,
        )
        self.assertEqual(packets[2]["schema_version"], "quant.analysis_packet.v3")
        flow_artifacts = [
            item
            for item in packets[2]["provenance"]["raw_artifacts"]
            if item.get("dataset") == "etf_fund_flows"
        ]
        self.assertEqual(len(flow_artifacts), 1)

    def test_processed_date_later_than_effective_gate_delays_one_more_session(self):
        session = ScriptedSession(
            flow_responses=[
                {
                    "results": [
                        flow_row(
                            "QQQ",
                            effective_date="2024-01-02",
                            processed_date="2024-01-04",
                        )
                    ]
                }
            ],
            massive_by_date={
                day: [grouped_row(close=400.0)]
                for day in ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
            },
        )
        pipeline = self.pipeline(session)
        pipeline.capture_etf_flows("2024-01-04", lookback_days=5)
        for day in ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"):
            pipeline.capture_daily(day, [], source="massive")
        output = self.root / "training_packets" / "delayed.jsonl"
        pipeline.export_packets("2024-01-04", "2024-01-05", output, symbols=["QQQ"])
        packets = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertIsNone(packets[0]["etf_flow"]["latest"])
        self.assertEqual(
            packets[1]["etf_flow"]["latest"]["training_available_session_date"],
            "2024-01-05",
        )

    def test_missing_future_trading_calendar_fails_closed(self):
        session = ScriptedSession(
            flow_responses=[{"results": [flow_row(processed_date="2024-01-02")]}],
            massive_by_date={"2024-01-02": [grouped_row(close=400.0)]},
        )
        pipeline = self.pipeline(session)
        pipeline.capture_etf_flows("2024-01-02", lookback_days=5)
        pipeline.capture_daily("2024-01-02", [], source="massive")
        output = self.root / "training_packets" / "fail_closed.jsonl"
        pipeline.export_packets("2024-01-02", "2024-01-02", output, symbols=["QQQ"])
        packet = json.loads(output.read_text().strip())
        self.assertIsNone(packet["etf_flow"]["latest"])
        self.assertEqual(packet["etf_flow"]["observations"], [])

    def test_isolated_weekend_junk_row_does_not_advance_session_lag(self):
        session = ScriptedSession(
            flow_responses=[
                {
                    "results": [
                        flow_row(
                            effective_date="2024-01-05",
                            processed_date="2024-01-05",
                        )
                    ]
                }
            ],
            massive_by_date={
                "2024-01-05": [grouped_row("QQQ", 400.0)],
                "2024-01-07": [grouped_row("JUNK", 1.0)],
                "2024-01-08": [grouped_row("QQQ", 401.0)],
                "2024-01-09": [grouped_row("QQQ", 402.0)],
            },
        )
        pipeline = self.pipeline(session)
        pipeline.capture_etf_flows("2024-01-05", lookback_days=5)
        for day in ("2024-01-05", "2024-01-07", "2024-01-08", "2024-01-09"):
            pipeline.capture_daily(day, [], source="massive")
        output = self.root / "training_packets" / "weekend_junk.jsonl"
        pipeline.export_packets("2024-01-08", "2024-01-09", output, symbols=["QQQ"])
        packets = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertIsNone(packets[0]["etf_flow"]["latest"])
        self.assertEqual(
            packets[1]["etf_flow"]["latest"]["training_available_session_date"],
            "2024-01-09",
        )

    def test_backfill_uses_documented_historical_filters(self):
        session = ScriptedSession(
            flow_responses=[
                {
                    "results": [
                        flow_row(
                            effective_date="2020-01-02",
                            processed_date="2020-01-03",
                        )
                    ]
                }
            ]
        )
        pipeline = self.pipeline(session)
        result = pipeline.backfill_etf_flows(
            "2020-01-01", "2020-01-31", tickers=["QQQ"]
        )
        self.assertTrue(result["historical_filters_supported"])
        self.assertEqual(result["freshness_status"], "historical_window_captured")
        call = session.calls[0]
        self.assertEqual(call["params"]["processed_date.gte"], "2020-01-01")
        self.assertEqual(call["params"]["processed_date.lte"], "2020-01-31")
        self.assertEqual(call["params"]["limit"], 5000)
        self.assertNotIn("apiKey", call["params"])

    def test_same_contract_recapture_keeps_capture_event_and_version(self):
        document = {
            "results": [
                flow_row(
                    effective_date="2020-01-02",
                    processed_date="2020-01-03",
                )
            ]
        }
        pipeline = self.pipeline(
            ScriptedSession(flow_responses=[document, document])
        )
        first = pipeline.backfill_etf_flows("2020-01-01", "2020-01-31")
        second = pipeline.backfill_etf_flows("2020-01-01", "2020-01-31")
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertTrue(second["repeated_payload_hash"])
        self.assertTrue(second["repeated_normalized_hash"])
        with pipeline.database.connect() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM raw_artifacts"
                ).fetchone()["n"],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM capture_events"
                ).fetchone()["n"],
                2,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM etf_flow_versions"
                ).fetchone()["n"],
                2,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM etf_flow_observations"
                ).fetchone()["n"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
