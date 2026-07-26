from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_dataset.config import CredentialSet
from quant_dataset.fmp_etf_constituents import (
    read_etf_symbols_from_universe,
    shard_symbols,
)
from quant_dataset.pipeline import DatasetPipeline


class FakeResponse:
    def __init__(self, document, status_code=200):
        self.content = json.dumps(document, separators=(",", ":")).encode("utf-8")
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}


class ConstituentSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        call = {
            "url": url,
            "params": dict(params or {}),
            "headers": dict(headers or {}),
            "timeout": timeout,
        }
        self.calls.append(call)
        if url.endswith("/portfolio-date"):
            return FakeResponse([{"date": "2024-03-31"}])
        if url.endswith("/api/v4/etf-holdings"):
            return FakeResponse(
                [
                    {
                        "acceptanceTime": "2024-05-15 17:00:00",
                        "date": "2024-03-31",
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "cusip": "037833100",
                        "pctVal": 10.5,
                        "valUsd": 1000000,
                        "balance": 5000,
                    },
                    {
                        "acceptanceTime": "2024-05-15 17:00:00",
                        "date": "2024-03-31",
                        "symbol": "AAPL",
                        "name": "Apple Inc. second position",
                        "cusip": "037833100",
                        "pctVal": 1.5,
                        "valUsd": 100000,
                        "balance": 500,
                    }
                ]
            )
        trade_date = url.rsplit("/", 1)[-1]
        close = 400.0 if trade_date == "2024-05-14" else 402.0
        return FakeResponse(
            {
                "status": "OK",
                "resultsCount": 2,
                "results": [
                    {
                        "T": "QQQ",
                        "o": close - 1,
                        "h": close + 1,
                        "l": close - 2,
                        "c": close,
                        "v": 1000,
                        "vw": close,
                        "n": 10,
                        "t": 1,
                    },
                    {
                        "T": "AAPL",
                        "o": 190.0,
                        "h": 192.0,
                        "l": 189.0,
                        "c": 191.0,
                        "v": 2000,
                        "vw": 190.5,
                        "n": 20,
                        "t": 2,
                    },
                ],
            }
        )


class FmpEtfConstituentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "QUANT_DATASET"
        self.session = ConstituentSession()
        self.pipeline = DatasetPipeline(
            self.root,
            CredentialSet("FMP_TEST_KEY", "MASSIVE_TEST_KEY", "test", "test"),
            session=self.session,
            retries=0,
            sleep=lambda _: None,
            rate_limiters={},
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_universe_reader_and_deterministic_shards(self):
        path = Path(self.temp.name) / "universe.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"symbol": "SPY", "is_etf": True}),
                    json.dumps({"symbol": "AAPL", "is_etf": False}),
                    json.dumps({"symbol": "QQQ", "is_etf": True}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        symbols = read_etf_symbols_from_universe(path)
        self.assertEqual(symbols, ["QQQ", "SPY"])
        self.assertEqual(shard_symbols(symbols, 2, 0), ["QQQ"])
        self.assertEqual(shard_symbols(symbols, 2, 1), ["SPY"])

    def test_backfill_and_training_packet_apply_acceptance_date_gate(self):
        result = self.pipeline.backfill_fmp_etf_constituents(
            "2024-01-01", "2024-12-31", ["QQQ"]
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["done"], 1)
        self.assertEqual(result["records"], 2)

        for trade_date in ("2024-05-14", "2024-05-15", "2024-05-16"):
            self.pipeline.capture_daily(trade_date, [], source="massive")
        output = self.root / "training_packets" / "constituents.jsonl"
        self.pipeline.export_packets(
            "2024-05-14",
            "2024-05-16",
            output,
            symbols=["AAPL", "QQQ"],
        )
        packets = {
            (row["symbol"], row["as_of_date"]): row
            for row in map(json.loads, output.read_text(encoding="utf-8").splitlines())
        }
        self.assertEqual(
            packets[("QQQ", "2024-05-14")]["etf_constituents"]["constituents"],
            [],
        )
        self.assertEqual(
            packets[("AAPL", "2024-05-14")]["etf_constituents"]["etf_memberships"],
            [],
        )
        self.assertEqual(
            packets[("QQQ", "2024-05-15")]["etf_constituents"]["constituents"],
            [],
        )
        qqq = packets[("QQQ", "2024-05-16")]["etf_constituents"]
        apple = packets[("AAPL", "2024-05-16")]["etf_constituents"]
        self.assertEqual(qqq["constituent_snapshot_date"], "2024-03-31")
        self.assertEqual(len(qqq["constituents"]), 2)
        self.assertTrue(
            all(row["constituent_ticker"] == "AAPL" for row in qqq["constituents"])
        )
        self.assertEqual(len(apple["etf_memberships"]), 2)
        self.assertTrue(
            all(row["etf_ticker"] == "QQQ" for row in apple["etf_memberships"])
        )
        self.assertEqual(
            qqq["constituents"][0]["training_available_session_date"],
            "2024-05-16",
        )
        self.assertTrue(self.pipeline.etf_constituents.verify()["ok"])


if __name__ == "__main__":
    unittest.main()
