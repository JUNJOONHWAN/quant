from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant_dataset.config import CredentialSet
from quant_dataset.fmp_universe import read_symbol_file, symbol_file_contract
from quant_dataset.pipeline import DatasetPipeline


class FakeResponse:
    def __init__(self, document):
        self.content = json.dumps(document, separators=(",", ":")).encode("utf-8")
        self.status_code = 200
        self.headers = {"content-type": "application/json"}


class FakeUniverseSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        call = {
            "url": url,
            "params": dict(params or {}),
            "headers": dict(headers or {}),
        }
        self.calls.append(call)
        if url.endswith("company-screener"):
            if call["params"].get("isEtf") == "true":
                return FakeResponse(
                    [
                        {
                            "symbol": "SPY",
                            "companyName": "SPDR S&P 500 ETF Trust",
                            "country": "US",
                            "exchangeShortName": "AMEX",
                            "isEtf": True,
                            "isFund": False,
                            "isActivelyTrading": True,
                        }
                    ]
                )
            return FakeResponse(
                [
                    {
                        "symbol": "AAPL",
                        "companyName": "Apple Inc.",
                        "country": "US",
                        "exchangeShortName": "NASDAQ",
                        "isEtf": False,
                        "isFund": False,
                        "isActivelyTrading": True,
                    }
                ]
            )
        if url.endswith("stock-list"):
            return FakeResponse([{"symbol": "AAPL", "companyName": "Apple Inc."}])
        if url.endswith("etf-list"):
            return FakeResponse(
                [
                    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust"},
                    {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
                ]
            )
        if url.endswith("actively-trading-list"):
            return FakeResponse([{"symbol": "AAPL", "name": "Apple Inc."}])
        if url.endswith("delisted-companies"):
            return FakeResponse(
                [
                    {
                        "symbol": "DEAD",
                        "companyName": "Delisted Corp",
                        "exchange": "NASDAQ",
                        "ipoDate": "2001-01-01",
                        "delistedDate": "2020-01-02",
                    },
                    {
                        "symbol": "FOREIGN.T",
                        "companyName": "Foreign Delisted Corp",
                        "exchange": "JPX",
                        "ipoDate": "2001-01-01",
                        "delistedDate": "2020-01-02",
                    }
                ]
            )
        if url.endswith("symbol-change"):
            return FakeResponse(
                [
                    {
                        "oldSymbol": "OLD",
                        "newSymbol": "NEW",
                        "companyName": "Renamed Corp",
                        "date": "2022-05-01",
                    }
                ]
            )
        raise AssertionError(url)


class FmpUniverseTest(unittest.TestCase):
    def test_snapshot_is_raw_first_and_symbols_file_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "QUANT_DATASET"
            session = FakeUniverseSession()
            pipeline = DatasetPipeline(
                root,
                CredentialSet("FMP_TEST_KEY", "MASSIVE_TEST_KEY", "test", "test"),
                session=session,
                retries=0,
                sleep=lambda _: None,
                rate_limiters={},
            )
            result = pipeline.capture_fmp_universe("2026-07-14")
            self.assertTrue(result["ok"])
            symbols_path = Path(result["symbols_path"])
            symbols = read_symbol_file(symbols_path)
            self.assertEqual(symbols, ["AAPL", "DEAD", "NEW", "OLD", "QQQ", "SPY"])
            contract = symbol_file_contract(symbols_path)
            self.assertEqual(contract["sha256"], result["symbols_sha256"])
            self.assertEqual(contract["symbol_count"], 6)
            manifest = json.loads(Path(result["manifest_path"]).read_text())
            self.assertGreaterEqual(len(manifest["raw_artifacts"]), 8)
            self.assertEqual(manifest["delisted_count"], 1)
            self.assertEqual(manifest["excluded_non_us_delisted_count"], 1)
            for metadata in root.glob("raw/**/*.metadata.json"):
                self.assertNotIn("FMP_TEST_KEY", metadata.read_text())


if __name__ == "__main__":
    unittest.main()
