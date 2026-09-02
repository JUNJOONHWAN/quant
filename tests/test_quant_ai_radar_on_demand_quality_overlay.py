import unittest
from unittest import mock

from workflows.quant_ai_radar import analyze_on_demand


class _Quality:
    def evaluate(self, symbol, trade_date, rows):
        return {
            "status": "pass",
            "sources": ["fmp", "massive"],
            "metrics": {},
            "reasons": [],
            "tolerances": {},
        }


class _Database:
    def observation_rows(self, start, end, symbols):
        return [{"symbol": symbols[0], "trade_date": start, "source": "fmp"}]

    def quality_for_pair(self, symbol, trade_date):
        return None


class _SharedDatabase:
    def history_payload_rows(self, symbol, as_of_date, lookback_days):
        return [
            {
                "symbol": symbol,
                "trade_date": as_of_date,
                "source": "fmp",
            }
        ]

    def quality_for_pair(self, symbol, trade_date):
        return None


class _StaleSharedDatabase(_SharedDatabase):
    def history_payload_rows(self, symbol, as_of_date, lookback_days):
        return [
            {
                "symbol": symbol,
                "trade_date": "2026-07-28",
                "source": "fmp",
            }
        ]


class _Pipeline:
    def __init__(self):
        self.database = _Database()
        self.quality = _Quality()
        self.calls = 0

    def analysis_packet_for_pair(self, symbol, as_of_date, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ValueError(f"quality row missing for {symbol} {as_of_date}")
        row = self.database.quality_for_pair(symbol, as_of_date)
        return {"quality": {"status": row["status"]}, "symbol": symbol}


class OnDemandQualityOverlayTests(unittest.TestCase):
    def test_missing_persisted_quality_is_evaluated_without_database_write(self):
        pipeline = _Pipeline()
        original_database = pipeline.database
        with mock.patch.object(
            analyze_on_demand,
            "adjust_packet_for_verified_corporate_actions",
            side_effect=lambda packet, actions: packet,
        ):
            packet = analyze_on_demand._analysis_packet(
                pipeline,
                symbol="STX",
                as_of_date="2026-07-30",
                corporate_actions={},
            )
        self.assertIs(pipeline.database, original_database)
        self.assertEqual(packet["quality"]["status"], "pass")
        self.assertEqual(
            packet["quality"]["evaluation_mode"],
            "ephemeral_read_only_overlay",
        )

    def test_unrelated_packet_error_is_not_hidden(self):
        pipeline = _Pipeline()
        pipeline.analysis_packet_for_pair = mock.Mock(
            side_effect=ValueError("different failure")
        )
        with self.assertRaisesRegex(ValueError, "different failure"):
            analyze_on_demand._analysis_packet(
                pipeline,
                symbol="STX",
                as_of_date="2026-07-30",
                corporate_actions={},
            )

    def test_shared_read_only_database_uses_history_payload_rows(self):
        pipeline = _Pipeline()
        pipeline.database = _SharedDatabase()
        original_database = pipeline.database
        with mock.patch.object(
            analyze_on_demand,
            "adjust_packet_for_verified_corporate_actions",
            side_effect=lambda packet, actions: packet,
        ):
            packet = analyze_on_demand._analysis_packet(
                pipeline,
                symbol="CRDO",
                as_of_date="2026-07-30",
                corporate_actions={},
            )
        self.assertIs(pipeline.database, original_database)
        self.assertEqual(packet["quality"]["status"], "pass")

    def test_latest_prior_sealed_observation_is_explicitly_marked_stale(self):
        pipeline = _Pipeline()
        pipeline.database = _StaleSharedDatabase()
        original_database = pipeline.database
        with mock.patch.object(
            analyze_on_demand,
            "adjust_packet_for_verified_corporate_actions",
            side_effect=lambda packet, actions: packet,
        ):
            packet = analyze_on_demand._analysis_packet(
                pipeline,
                symbol="STX",
                as_of_date="2026-07-30",
                corporate_actions={},
            )
        self.assertIs(pipeline.database, original_database)
        self.assertEqual(packet["freshness"]["analysis_as_of_date"], "2026-07-28")
        self.assertEqual(packet["freshness"]["stale_sessions_or_calendar_days"], 2)


if __name__ == "__main__":
    unittest.main()
