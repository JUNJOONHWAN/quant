from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from workflows.quant_ai_radar.app_cli import build_analyze_command
from workflows.quant_ai_radar.forecast_synthesis import ForecastSynthesisClient
from workflows.quant_ai_radar.historical_analog import (
    aggregate_outcomes,
    feature_distance,
    feature_row_from_judgement,
    realised_outcomes,
)
from workflows.quant_ai_radar.model_runtime import ResponseContractError
from workflows.quant_ai_radar.report_narratives import (
    NARRATIVE_SCHEMA_VERSION,
    validate_report_narratives,
)
from workflows.quant_ai_radar.training_native import (
    TRAINING_NATIVE_PROMPT_CONTRACT,
    complete_training_native_judgement,
)


def judgement(symbol: str = "AAA") -> dict:
    return {
        "facts": {
            "symbol": symbol,
            "as_of_date": "2020-04-01",
            "price": {
                "return_1_session_pct": 1.0,
                "return_5_session_pct": 2.0,
                "return_20_session_pct": 3.0,
                "annualized_realized_volatility_pct": 20.0,
                "max_drawdown_in_packet_pct": -5.0,
            },
            "etf_flow": {
                "latest_robust_zscore": 1.5,
                "latest_flow_to_assets_pct": 0.5,
            },
            "etf_flow_to_constituent": {
                "net_weighted_flow_rate_contribution_pct": 0.3,
                "eligible_etf_count": 4,
                "positive_etf_count": 3,
                "negative_etf_count": 1,
            },
            "liquidity": {"median_dollar_volume": 100_000_000},
            "quality_status": "complete",
        },
        "interpretation": {
            "task_type": "stock_constituent_flow_analysis",
            "price_signal": "positive",
            "etf_flow_signal": "positive",
            "etf_flow_signal_source": "constituent_etf_flow_exposure",
            "relationship": "price_flow_positive_confirmation",
            "scope": "data_interpretation_not_trade_execution",
        },
        "regime": "price_flow_positive_confirmation",
        "confidence": 0.8,
        "conclusion": "현재 가격과 ETF 자금 구조가 함께 확인됩니다.",
        "counter_evidence": ["단기 변동성"],
        "unknowns": ["향후 뉴스"],
    }


class HistoricalAnalogTests(unittest.TestCase):
    def test_feature_extraction_and_distance_are_deterministic(self) -> None:
        row = feature_row_from_judgement(judgement())
        self.assertEqual(row["flow_breadth"], 0.5)
        self.assertEqual(row["task_type"], "stock_constituent_flow_analysis")
        self.assertEqual(feature_distance(row, row), 0.0)

    def test_realised_outcomes_never_cross_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            connection.execute(
                "CREATE TABLE daily_observations ("
                "source TEXT, symbol TEXT, trade_date TEXT, close REAL, "
                "adjusted_close REAL)"
            )
            start = date(2020, 1, 1)
            for offset in range(66):
                trade_date = (start + timedelta(days=offset)).isoformat()
                connection.execute(
                    "INSERT INTO daily_observations VALUES (?,?,?,?,?)",
                    ("fmp", "AAA", trade_date, 100 + offset, 100 + offset),
                )
                connection.execute(
                    "INSERT INTO daily_observations VALUES (?,?,?,?,?)",
                    ("fmp", "SPY", trade_date, 200 + offset, 200 + offset),
                )
            connection.commit()
            cutoff = (start + timedelta(days=20)).isoformat()
            outcomes = realised_outcomes(
                connection,
                symbol="AAA",
                analog_date=start.isoformat(),
                cutoff_date=cutoff,
            )
            self.assertEqual(set(outcomes), {5, 20})
            self.assertLessEqual(outcomes[20]["outcome_end_date"], cutoff)
            self.assertNotIn(60, outcomes)
            connection.close()

    def test_aggregate_preserves_distribution_not_action(self) -> None:
        summary = aggregate_outcomes(
            [
                {
                    "return_pct": -2.0,
                    "spy_excess_return_pct": -1.0,
                    "maximum_favorable_excursion_pct": 1.0,
                    "maximum_adverse_excursion_pct": -3.0,
                },
                {
                    "return_pct": 4.0,
                    "spy_excess_return_pct": 2.0,
                    "maximum_favorable_excursion_pct": 5.0,
                    "maximum_adverse_excursion_pct": -1.0,
                },
            ]
        )
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["positive_probability_pct"], 50.0)
        self.assertNotIn("action_view", summary)


class TrainingNativePromptTests(unittest.TestCase):
    def test_inference_uses_exact_sft_context_and_instruction(self) -> None:
        captured = {}

        class Client:
            def complete_validated(self, **kwargs):
                captured.update(kwargs)
                return {"ok": True}, {"request_sha256": "a" * 64}

        example = {
            "context": "exact training context",
            "instruction": "exact training instruction /no_think",
            "response": json.dumps({"facts": {"symbol": "AAA"}}),
            "metadata": {"input_packet_schema": "quant.analysis_packet.v3"},
        }
        result, _ = complete_training_native_judgement(
            client=Client(),
            example=example,
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(captured["system"], example["context"])
        self.assertEqual(captured["user"], example["instruction"])
        self.assertEqual(
            captured["expected_response"],
            {"facts": {"symbol": "AAA"}},
        )
        self.assertEqual(
            TRAINING_NATIVE_PROMPT_CONTRACT,
            "quant.analysis_packet.v3.build_example.context_instruction.v1",
        )


class ForecastSynthesisTests(unittest.TestCase):
    def test_27b_request_is_model_bound_and_validated(self) -> None:
        model = "test-27b"
        call_count = 0

        def transport(payload, _headers, _timeout):
            nonlocal call_count
            call_count += 1
            self.assertEqual(payload["model"], model)
            content = {
                "symbol": "AAA",
                "forecast_view": "매수 검토",
                "primary_horizon_sessions": 20,
                "thesis": "학습 패턴과 유사사례 분포가 중기 우위를 함께 지지합니다.",
                "learned_pattern_use": "팔비 모델이 확인한 동반 강세 구조를 출발점으로 사용했습니다.",
                "historical_evidence": "완결된 유사사례의 분포가 상방 쪽으로 기울어 있습니다.",
                "market_context_effect": "현재 시장 맥락은 이 판단을 보강하지만 단독 근거는 아닙니다.",
                "supporting_evidence": ["가격과 ETF 자금 방향이 함께 확인됩니다."],
                "counter_evidence": ["단기 변동성이 확대되면 경로가 달라질 수 있습니다."],
                "invalidation_conditions": ["가격과 ETF 자금의 동반 구조가 해제되는지 확인합니다."],
                "confidence": 0.72,
            }
            if call_count == 1:
                content["thesis"] = "입력에 없는 300달러 기준을 새로 만든 잘못된 전망입니다."
            return {
                "model": model,
                "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 100},
            }

        client = ForecastSynthesisClient(
            endpoint="http://127.0.0.1:1/v1/chat/completions",
            model=model,
            transport=transport,
        )
        result, trace = client.synthesize(
            symbol="AAA",
            judgement=judgement(),
            analog_forecast={"horizon_statistics": {}, "sha256": "a" * 64},
            market_context={"status": "not_available"},
        )
        self.assertEqual(result["forecast_view"], "매수 검토")
        self.assertEqual(trace["endpoint_model"], model)
        self.assertEqual(len(trace["request_sha256"]), 64)
        self.assertEqual(trace["contract_attempts"], 2)
        self.assertTrue(trace["contract_repair_applied"])


class DailyContractTests(unittest.TestCase):
    @staticmethod
    def report(security_row: dict) -> dict:
        return {
            "schema_version": "quant.ai_radar_report.v2",
            "market_dashboard": {
                "rotation_clusters": [],
                "candidate_lanes": {
                    "positive_confirmation_stocks": [{"symbol": "AAA"}],
                    "negative_confirmation_stocks": [],
                    "divergence_stocks": [],
                },
            },
            "multistage_narratives": {
                "schema_version": NARRATIVE_SCHEMA_VERSION,
                "sector_explanations": [],
                "security_explanations": [security_row],
                "editorial": {
                    "headline": "현재 시장 구조를 학습 패턴에 따라 요약합니다.",
                    "executive_summary": "현재 가격과 자금 구조의 동행 여부를 설명합니다.",
                    "rotation_summary": "현재 섹터 회전과 확산 정도를 설명합니다.",
                    "selection_summary": "현재 주요 종목의 구조적 위치를 설명합니다.",
                    "risk_summary": "반대 근거와 미확인 사항을 함께 설명합니다.",
                },
            },
        }

    def test_daily_security_requires_current_pattern_not_action(self) -> None:
        row = {
            "symbol": "AAA",
            "headline": "AAA의 현재 가격과 자금 구조를 설명합니다.",
            "group_context": "AAA는 현재 확인된 시장 회전 영역과 연결됩니다.",
            "etf_transmission": "ETF 구성 관계에서 현재 자금 전달 구조가 확인됩니다.",
            "counterpoint": "가격과 자금 확산이 약해질 가능성도 함께 봅니다.",
            "watch_condition": "다음 자료에서 동행 구조가 유지되는지 확인합니다.",
            "learned_pattern": "학습된 동반 강세 국면과 가까운 현재 구조입니다.",
            "pattern_evidence": "가격과 구성 ETF 자금 방향이 함께 정렬됩니다.",
            "pattern_risk": "확산 범위가 좁아지면 현재 패턴의 신뢰가 낮아집니다.",
        }
        validate_report_narratives(self.report(row))
        with self.assertRaises(ResponseContractError):
            validate_report_narratives(self.report({**row, "action_view": "매수 검토"}))
        for invalid_row in (
            {**row, "pattern_risk": "향후 조정 가능성이 있습니다."},
            {**row, "etf_transmission": "ETF 흐름이 3개 경로에서 확인됩니다."},
            {**row, "pattern_evidence": "ETF 흐름이 주가에 영향을 미칩니다."},
            {**row, "learned_pattern": "투자자 기대가 강해진 현재 패턴입니다."},
        ):
            with self.assertRaises(ResponseContractError):
                validate_report_narratives(self.report(invalid_row))


class CliWiringTests(unittest.TestCase):
    def test_individual_cli_wires_both_models_and_analog_sources(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "QUANT_AI_MODEL_ENDPOINT": "http://127.0.0.1:8018/v1/chat/completions",
                "QUANT_AI_FORECAST_MODEL_ENDPOINT": "http://127.0.0.1:8004/v1/chat/completions",
                "QUANT_AI_FORECAST_MODEL_NAME": "test-27b",
            },
            clear=False,
        ):
            command = build_analyze_command(["AAA"])
        self.assertIn("--model-endpoint", command)
        self.assertIn("--forecast-model-endpoint", command)
        self.assertIn("--analog-example-database", command)
        self.assertIn("--analog-price-database", command)
        self.assertIn("--analog-index-database", command)


if __name__ == "__main__":
    unittest.main()
