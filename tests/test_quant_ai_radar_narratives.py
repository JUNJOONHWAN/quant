from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workflows.quant_ai_radar.email_delivery import _email_html
from workflows.quant_ai_radar.model_runtime import ResponseContractError
from workflows.quant_ai_radar.presentation import (
    confidence_pct,
    label_regime,
    label_rotation_state,
    whole,
)
from workflows.quant_ai_radar.report_narratives import (
    _clean_prose,
    build_multistage_narratives,
)
from workflows.quant_ai_radar.report_renderer import render_reports


class FakeNarrativeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, response_schema, **kwargs):
        self.calls += 1
        properties = response_schema["properties"]
        if "items" not in properties:
            prompt = str(kwargs.get("user") or "")
            supported_text = prompt.split(
                "SUPPORTED_GROUPED_SYMBOLS=", 1
            )[-1].split("\n", 1)[0]
            supported = json.loads(supported_text) if supported_text else []
            response = {
                "headline": "시장의 주도 흐름과 경계 신호입니다.",
                "executive_summary": "시장 전체에서는 섹터별 자금 확산이 엇갈려 선별적 해석이 필요합니다.",
                "rotation_summary": "강한 섹터와 약한 섹터가 분리되며 회전의 폭을 함께 확인해야 합니다.",
                "selection_summary": (
                    f"주요 종목은 {' '.join(supported[:3])}입니다. "
                    "소속 섹터 흐름과 ETF 전달 경로가 함께 확인됩니다. "
                    "반대 근거도 함께 점검해야 합니다."
                ),
                "risk_summary": "가격과 자금 방향이 어긋나는 후보는 지속성 확인 전까지 반대 근거가 큽니다.",
            }
        else:
            prompt = str(kwargs.get("user") or "")
            if not prompt and kwargs.get("messages"):
                prompt = "\n".join(
                    str(message.get("content") or "")
                    for message in kwargs["messages"]
                )
            mention_text = prompt.split(
                "REQUIRED_MENTIONS=", 1
            )[-1].split("\n", 1)[0]
            mentions = json.loads(mention_text) if mention_text else {}
            all_mention_text = prompt.split(
                "REQUIRED_ALL_MENTIONS=", 1
            )[-1].split("\n", 1)[0]
            all_mentions = (
                json.loads(all_mention_text) if all_mention_text else {}
            )
            item_schema = properties["items"]["items"]
            id_key = next(
                key
                for key, value in item_schema["properties"].items()
                if "enum" in value
            )
            identifiers = item_schema["properties"][id_key]["enum"]
            response = {"items": []}
            for identifier in identifiers:
                row = {id_key: identifier}
                for key in item_schema["required"]:
                    if key != id_key:
                        required = (
                            mentions.get(identifier, {}).get(key) or []
                        )
                        required_all = (
                            all_mentions.get(identifier, {}).get(key) or []
                        )
                        named = " " + " ".join(
                            [required[0] if required else "", *required_all]
                        ).strip()
                        row[key] = (
                            "전체 시장 흐름과 소속 집단의 자금 방향을 함께 해석한 "
                            f"구체적인 한국어 설명입니다{named}."
                        )
                response["items"].append(row)
        return response, {
            "request_sha256": f"request-{self.calls}",
            "response_sha256": f"response-{self.calls}",
        }

    def complete_messages(self, **kwargs):
        return self.complete(**kwargs)


class ReorderedNarrativeClient(FakeNarrativeClient):
    def complete_messages(self, **kwargs):
        response, trace = self.complete(**kwargs)
        if isinstance(response.get("items"), list):
            response["items"].reverse()
        return response, trace


def _result(symbol: str, regime: str, price: str, flow: str) -> dict:
    return {
        "symbol": symbol,
        "task_type": "stock_constituent_flow_analysis",
        "trace": {
            "request_sha256": f"request-{symbol}",
            "response_sha256": f"response-{symbol}",
        },
        "judgement": {
            "facts": {
                "symbol": symbol,
                "as_of_date": "2026-07-29",
                "quality_status": "pass",
                "price": {
                    "return_1_session_pct": 1,
                    "return_5_session_pct": 2,
                    "return_20_session_pct": 3,
                    "observed_sessions": 21,
                },
                "liquidity": {},
                "etf_flow": {},
                "etf_flow_to_constituent": {
                    "net_weighted_flow_rate_contribution_pct": 1,
                    "eligible_etf_count": 2,
                    "excluded_etf_count": 0,
                    "positive_etf_count": 2,
                    "negative_etf_count": 0,
                    "top_contributing_etfs": [],
                },
                "etf_relations": {"membership_count": 2},
            },
            "interpretation": {
                "task_type": "stock_constituent_flow_analysis",
                "price_signal": price,
                "etf_flow_signal": flow,
                "relationship": regime,
            },
            "regime": regime,
            "confidence": 0.7,
            "conclusion": "근거가 확인되지만 지속성은 추가 확인이 필요합니다.",
            "counter_evidence": [],
            "unknowns": [],
        },
    }


class QuantAiRadarNarrativeTest(unittest.TestCase):
    def test_clean_prose_normalizes_missing_terminal_punctuation(self):
        cleaned, changed = _clean_prose(
            "기술 섹터 안에서 AVGO의 가격과 ETF 흐름이 함께 약화"
        )
        self.assertTrue(changed)
        self.assertEqual(
            cleaned,
            "기술 섹터 안에서 AVGO의 가격과 ETF 흐름이 함께 약화.",
        )

    def test_multistage_calls_cover_all_sectors_and_three_stock_lanes(self):
        positive = "price_flow_positive_confirmation"
        negative = "price_flow_negative_confirmation"
        divergence = "price_up_flow_out_divergence"
        rows = [
            _result("AAA", positive, "positive", "positive"),
            _result("BBB", negative, "negative", "negative"),
            _result("CCC", divergence, "positive", "negative"),
        ]
        rankings = {
            "stocks_by_regime": {
                positive: [{"symbol": "AAA", "regime": positive}],
                negative: [{"symbol": "BBB", "regime": negative}],
                divergence: [{"symbol": "CCC", "regime": divergence}],
            },
            "etfs_by_regime": {},
        }
        aggregate = {
            "analyzed_security_count": 3,
            "price_signal_counts": {"positive": 2, "negative": 1},
            "etf_flow_signal_counts": {"positive": 1, "negative": 2},
            "regime_counts": {positive: 1, negative: 1, divergence: 1},
            "candidate_rankings": rankings,
            "etf_leaders": [],
            "stock_leaders": [
                {"symbol": "AAA", "regime": positive},
                {"symbol": "BBB", "regime": negative},
                {"symbol": "CCC", "regime": divergence},
            ],
        }
        radar = {
            "integrated_rotation_clusters": [
                {
                    "integrated_cluster": cluster,
                    "integrated_state": state,
                    "breadth_score": 50,
                    "top_related_stocks": [f"{symbol}:10%"],
                }
                for cluster, state, symbol in (
                    ("Technology", "rotation_in", "AAA"),
                    ("Healthcare", "rotation_out", "BBB"),
                    ("Utilities", "mixed", "CCC"),
                )
            ],
            "accumulation_clusters": [],
        }
        client = FakeNarrativeClient()
        narratives, trace = build_multistage_narratives(
            client=client,
            aggregate=aggregate,
            radar=radar,
            market_judgement={
                "market_state": "rotation",
                "summary": "시장 회전",
            },
            results=rows,
        )
        self.assertEqual(narratives["sector_count"], 3)
        self.assertEqual(narratives["security_count"], 3)
        self.assertEqual(
            [row["symbol"] for row in narratives["security_explanations"]],
            ["AAA", "BBB", "CCC"],
        )
        self.assertEqual(narratives["model_call_count"], 3)
        self.assertEqual(trace["model_call_count"], 3)
        self.assertEqual(client.calls, 3)

    def test_model_item_order_is_normalized_without_replacing_model_prose(self):
        positive = "price_flow_positive_confirmation"
        rows = [
            _result("AAA", positive, "positive", "positive"),
            _result("BBB", positive, "positive", "positive"),
        ]
        aggregate = {
            "analyzed_security_count": 2,
            "price_signal_counts": {"positive": 2},
            "etf_flow_signal_counts": {"positive": 2},
            "regime_counts": {positive: 2},
            "candidate_rankings": {
                "stocks_by_regime": {
                    positive: [
                        {"symbol": "AAA", "regime": positive},
                        {"symbol": "BBB", "regime": positive},
                    ]
                },
                "etfs_by_regime": {},
            },
            "etf_leaders": [],
            "stock_leaders": [],
        }
        radar = {
            "integrated_rotation_clusters": [
                {
                    "integrated_cluster": "Technology",
                    "integrated_state": "rotation_in",
                    "breadth_score": 50,
                    "top_related_stocks": ["AAA:10%", "BBB:9%"],
                }
            ],
            "accumulation_clusters": [],
        }
        narratives, _ = build_multistage_narratives(
            client=ReorderedNarrativeClient(),
            aggregate=aggregate,
            radar=radar,
            market_judgement={"market_state": "rotation", "summary": "시장 회전"},
            results=rows,
        )
        self.assertEqual(
            [row["symbol"] for row in narratives["security_explanations"]],
            ["AAA", "BBB"],
        )

    def test_completed_model_stages_are_reused_from_exact_checkpoints(self):
        positive = "price_flow_positive_confirmation"
        rows = [_result("AAA", positive, "positive", "positive")]
        aggregate = {
            "analyzed_security_count": 1,
            "price_signal_counts": {"positive": 1},
            "etf_flow_signal_counts": {"positive": 1},
            "regime_counts": {positive: 1},
            "candidate_rankings": {
                "stocks_by_regime": {
                    positive: [{"symbol": "AAA", "regime": positive}]
                },
                "etfs_by_regime": {},
            },
            "etf_leaders": [],
            "stock_leaders": [],
        }
        radar = {
            "integrated_rotation_clusters": [
                {
                    "integrated_cluster": "Technology",
                    "integrated_state": "rotation_in",
                    "breadth_score": 50,
                    "top_related_stocks": ["AAA:10%"],
                }
            ],
            "accumulation_clusters": [],
        }
        client = FakeNarrativeClient()
        with tempfile.TemporaryDirectory() as temporary:
            kwargs = {
                "client": client,
                "aggregate": aggregate,
                "radar": radar,
                "market_judgement": {
                    "market_state": "rotation",
                    "summary": "시장 회전",
                },
                "results": rows,
                "checkpoint_dir": Path(temporary),
            }
            build_multistage_narratives(**kwargs)
            calls_after_first_run = client.calls
            _, trace = build_multistage_narratives(**kwargs)
        self.assertEqual(client.calls, calls_after_first_run)
        self.assertTrue(all(stage.get("cache_hit") for stage in trace["stages"]))

    def test_v2_renderer_and_email_fail_when_ai_narratives_are_missing(self):
        report = {"schema_version": "quant.ai_radar_report.v2"}
        with self.assertRaises(ResponseContractError):
            _email_html(report)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ResponseContractError):
                render_reports(
                    run_dir=Path(temporary),
                    report=report,
                    results=[],
                    coverage_ledger=[],
                )

    def test_presentation_translates_codes_and_rounds_display_only(self):
        self.assertEqual(
            label_rotation_state("rotation_in"),
            "자금 유입과 가격 확산",
        )
        self.assertEqual(
            label_regime("price_flow_negative_confirmation"),
            "가격·ETF 자금 동반 약세",
        )
        self.assertEqual(confidence_pct(0.846), "85%")
        self.assertEqual(whole(12.67, signed=True, suffix="%"), "+13%")


if __name__ == "__main__":
    unittest.main()
