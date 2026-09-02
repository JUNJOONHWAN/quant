import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from quant_dataset.fmp_active_universe import FmpActiveUniverseCollector
from quant_dataset.providers import PayloadValidationError


class _Http:
    def __init__(self):
        self.calls = []
        self.next_artifact = 1

    def get_json(self, **kwargs):
        self.calls.append(kwargs)
        dataset = kwargs["dataset"]
        params = kwargs["params"]
        if dataset == "active_company_screener_nasdaq":
            document = (
                [
                    {
                        "symbol": "AAPL",
                        "exchangeShortName": "NASDAQ",
                        "isEtf": False,
                        "isFund": False,
                    },
                    {
                        "symbol": "MUTFX",
                        "exchangeShortName": "NASDAQ",
                        "isEtf": False,
                        "isFund": True,
                    },
                ]
                if params["page"] == 0
                else []
            )
        elif dataset == "active_company_screener_nyse":
            document = []
        elif dataset == "active_company_screener_amex":
            document = [
                {
                    "symbol": "AMETF",
                    "exchangeShortName": "AMEX",
                    "isEtf": True,
                    "isFund": True,
                }
            ]
        elif dataset == "active_company_screener_cboe":
            document = [
                {
                    "symbol": "CBOEX",
                    "exchangeShortName": "CBOE",
                    "isEtf": False,
                    "isFund": False,
                }
            ]
        elif dataset == "stable_actively_trading_list":
            document = [{"symbol": "AAPL"}, {"symbol": "AMETF"}]
        elif dataset == "stable_stock_list":
            document = [
                {
                    "symbol": "AAPL",
                    "exchangeShortName": "NASDAQ",
                    "type": "stock",
                }
            ]
        elif dataset == "stable_etf_list":
            document = [
                {
                    "symbol": "AMETF",
                    "exchangeShortName": "AMEX",
                    "type": "etf",
                }
            ]
        elif dataset == "legacy_available_traded_list":
            document = [
                {
                    "symbol": "PREFP",
                    "exchangeShortName": "NASDAQ",
                    "type": "stock",
                },
                {
                    "symbol": "BATSX",
                    "exchangeShortName": "BATS",
                    "type": "stock",
                },
                {
                    "symbol": "OTCX",
                    "exchangeShortName": "OTC",
                    "type": "stock",
                },
            ]
        elif dataset == "legacy_stock_list":
            document = [
                {
                    "symbol": "OLDX",
                    "exchangeShortName": "NYSE",
                    "type": "stock",
                }
            ]
        elif dataset == "stable_symbol_change":
            document = [
                {
                    "oldSymbol": "OLDX",
                    "newSymbol": "NEWX",
                    "date": "2026-07-20",
                    "companyName": "Renamed Inc.",
                },
                {
                    "oldSymbol": "NOOP",
                    "newSymbol": "NOOP",
                    "date": "1969-12-31",
                    "companyName": "Provider sentinel",
                },
            ]
        else:
            raise AssertionError(dataset)
        artifact = SimpleNamespace(
            artifact_id=self.next_artifact,
            capture_event_id=self.next_artifact + 100,
            captured_at_utc="2026-07-31T01:00:00+00:00",
            payload_sha256="{:064x}".format(self.next_artifact),
        )
        self.next_artifact += 1
        return SimpleNamespace(document=document, artifact=artifact)


class FmpActiveUniverseTests(unittest.TestCase):
    def test_builds_four_venue_active_and_reference_masters(self):
        with tempfile.TemporaryDirectory() as root:
            http = _Http()
            collector = FmpActiveUniverseCollector(
                Path(root), http, "test-key"
            )
            with mock.patch(
                "quant_dataset.fmp_active_universe."
                "FMP_ACTIVE_SCREENER_PAGE_SIZE",
                2,
            ):
                result = collector.capture("2026-07-30")

            active = Path(result["symbols_path"]).read_text().splitlines()
            reference = Path(
                result["reference_symbols_path"]
            ).read_text().splitlines()
            manifest = json.loads(
                Path(result["manifest_path"]).read_text()
            )
            self.assertEqual(
                active,
                ["AAPL", "AMETF", "BATSX", "CBOEX", "PREFP"],
            )
            self.assertNotIn("MUTFX", active)
            self.assertNotIn("OTCX", active)
            self.assertIn("OLDX", reference)
            self.assertEqual(result["core_symbol_count"], 3)
            self.assertEqual(manifest["exchange_scope"], [
                "NASDAQ",
                "NYSE",
                "AMEX",
                "CBOE",
            ])
            self.assertFalse(manifest["country_filter_used"])
            self.assertEqual(result["symbol_change_event_count"], 1)
            self.assertEqual(result["symbol_change_exclusion_count"], 1)
            self.assertEqual(
                manifest["symbol_change_exclusions"][0]["reason"],
                "provider_no_op_same_symbol",
            )
            symbol_changes = [
                json.loads(line)
                for line in Path(result["symbol_changes_path"])
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                (symbol_changes[0]["old_symbol"], symbol_changes[0]["new_symbol"]),
                ("OLDX", "NEWX"),
            )
            screener_calls = [
                call
                for call in http.calls
                if call["dataset"].startswith("active_company_screener_")
            ]
            self.assertGreaterEqual(len(screener_calls), 5)
            self.assertTrue(
                all("country" not in call["params"] for call in screener_calls)
            )

    def test_symbol_change_limit_is_release_blocking(self):
        with tempfile.TemporaryDirectory() as root:
            collector = FmpActiveUniverseCollector(
                Path(root), _Http(), "test-key"
            )
            with (
                mock.patch(
                    "quant_dataset.fmp_active_universe."
                    "FMP_ACTIVE_SCREENER_PAGE_SIZE",
                    2,
                ),
                mock.patch(
                    "quant_dataset.fmp_active_universe."
                    "FMP_SYMBOL_CHANGE_LIMIT",
                    1,
                ),
            ):
                with self.assertRaises(PayloadValidationError):
                    collector.capture("2026-07-30")


if __name__ == "__main__":
    unittest.main()
