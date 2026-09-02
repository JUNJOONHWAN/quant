from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.quant_llm.build_sft_dataset import (
    _preferred_price_rows,
    packet_eligibility,
)
from workflows.quant_ai_radar.corporate_actions import (
    CorporateActionError,
    adjust_packet_for_verified_corporate_actions,
    load_verified_corporate_actions,
)
from workflows.quant_ai_radar.run_queue import RadarQueue
from workflows.quant_ai_radar.universe import Candidate


def _event(source_type: str = "official_issuer") -> dict:
    return {
        "symbol": "TZA",
        "action_type": "reverse_split",
        "effective_date": "2026-07-15",
        "available_date": "2026-06-10",
        "announcement_date": "2026-06-10",
        "old_shares": 10,
        "new_shares": 1,
        "source_type": source_type,
        "source_name": "Direxion",
        "source_url": (
            "https://www.direxion.com/press-release/"
            "direxion-to-split-nine-etfs"
        ),
    }


def _packet() -> dict:
    history = []
    for index in range(10):
        before = index < 5
        history.append(
            {
                "trade_date": f"2026-07-{10 + index:02d}",
                "sources": [
                    {
                        "source": "fmp" if before else "massive",
                        "open": 4.0 if before else 40.0,
                        "high": 4.1 if before else 41.0,
                        "low": 3.9 if before else 39.0,
                        "close": 4.02 if before else 40.2,
                        "volume": 10_000_000 if before else 1_000_000,
                    }
                ],
            }
        )
    return {
        "schema_version": "quant.analysis_packet.v3",
        "symbol": "TZA",
        "as_of_date": "2026-07-19",
        "history": history,
        "etf_flow": {"observations": [{"fund_flow": 1.0}]},
        "etf_constituents": {},
        "provenance": {},
        "packet_id": "old",
    }


class QuantAiRadarCorporateActionsTest(unittest.TestCase):
    def test_verified_reverse_split_adjusts_prior_rows_and_eligibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "actions.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "quant.verified_corporate_actions.v1"
                        ),
                        "events": [_event()],
                    }
                ),
                encoding="utf-8",
            )
            ledger = load_verified_corporate_actions(
                path, as_of_date="2026-07-19"
            )
            adjusted = adjust_packet_for_verified_corporate_actions(
                _packet(), ledger
            )
        prior = adjusted["history"][0]["sources"][0]
        current = adjusted["history"][-1]["sources"][0]
        self.assertAlmostEqual(prior["close"], 40.2)
        self.assertAlmostEqual(prior["volume"], 1_000_000)
        self.assertAlmostEqual(current["close"], 40.2)
        self.assertNotEqual(adjusted["packet_id"], "old")
        self.assertEqual(
            _preferred_price_rows(adjusted)[0]["price_basis"],
            "verified_pit_corporate_action_adjusted",
        )
        self.assertNotIn(
            "raw_price_discontinuity_ge_45pct_without_pit_corporate_action",
            packet_eligibility(
                adjusted,
                min_etf_observed_sessions=5,
                min_etf_median_dollar_volume=1,
            )["reasons"],
        )

    def test_future_or_unavailable_action_is_not_applied(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "actions.json"
            future = _event()
            future["available_date"] = "2026-07-20"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "quant.verified_corporate_actions.v1"
                        ),
                        "events": [future],
                    }
                ),
                encoding="utf-8",
            )
            ledger = load_verified_corporate_actions(
                path, as_of_date="2026-07-19"
            )
        self.assertEqual(ledger["events"], [])

    def test_multiple_forward_and_reverse_splits_use_cumulative_basis(self):
        packet = _packet()
        packet["symbol"] = "PAIR"
        ledger = {
            "schema_version": "quant.oracle_corporate_actions.v1",
            "sha256": "a" * 64,
            "events_by_symbol": {
                "PAIR": [
                    {
                        **_event(),
                        "symbol": "PAIR",
                        "effective_date": "2026-07-15",
                        "old_shares": 10,
                        "new_shares": 1,
                        "price_factor_for_prior_rows": 10,
                        "volume_factor_for_prior_rows": 0.1,
                    },
                    {
                        **_event(),
                        "symbol": "PAIR",
                        "action_type": "forward_split",
                        "effective_date": "2026-07-18",
                        "old_shares": 1,
                        "new_shares": 2,
                        "price_factor_for_prior_rows": 0.5,
                        "volume_factor_for_prior_rows": 2,
                    },
                ]
            },
        }
        adjusted = adjust_packet_for_verified_corporate_actions(packet, ledger)
        before_both = adjusted["history"][0]["sources"][0]
        between = adjusted["history"][6]["sources"][0]
        after_both = adjusted["history"][-1]["sources"][0]
        self.assertAlmostEqual(before_both["close"], 4.02 * 10 * 0.5)
        self.assertAlmostEqual(before_both["volume"], 10_000_000 * 0.1 * 2)
        self.assertAlmostEqual(between["close"], 40.2 * 0.5)
        self.assertAlmostEqual(between["volume"], 1_000_000 * 2)
        self.assertAlmostEqual(after_both["close"], 40.2)

    def test_non_official_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "actions.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "quant.verified_corporate_actions.v1"
                        ),
                        "events": [_event("community")],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(CorporateActionError):
                load_verified_corporate_actions(
                    path, as_of_date="2026-07-19"
                )

    def test_queue_retries_only_matching_discontinuity_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = RadarQueue(Path(temporary) / "queue.sqlite3")
            queue.seed(
                [
                    Candidate(
                        symbol="TZA",
                        proxy_task_type="etf_own_flow_analysis",
                        quality_status="pass",
                        relation_types=("own_flow",),
                    )
                ]
            )
            queue.mark_excluded(
                "TZA",
                {"eligible": False},
                "raw_price_discontinuity_ge_45pct_without_pit_corporate_action",
            )
            changed = queue.requeue_verified_corporate_action_exclusions(
                ["TZA"]
            )
            pending = queue.pending()
        self.assertEqual(changed, 1)
        self.assertEqual([row["symbol"] for row in pending], ["TZA"])


if __name__ == "__main__":
    unittest.main()
