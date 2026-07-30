import copy
import tempfile
import unittest
from pathlib import Path

from workflows.quant_ai_radar.decision_support import (
    audit_report_quality,
    build_market_dashboard,
    build_security_brief,
    market_semantic_issues,
)
from workflows.quant_ai_radar.market_report import validate_market_synthesis
from workflows.quant_ai_radar.model_runtime import (
    ResponseContractError,
    validate_symbol_judgement,
)
from workflows.quant_ai_radar.report_renderer import render_reports


def judgement(symbol: str = "AAPL") -> dict:
    return {
        "facts": {
            "symbol": symbol,
            "as_of_date": "2026-07-28",
            "price": {
                "latest_close": 340.08,
                "return_1_session_pct": 0.940904,
                "return_5_session_pct": 3.76518,
                "return_20_session_pct": 20.707035,
                "annualized_realized_volatility_pct": 29.039729,
                "max_drawdown_in_packet_pct": -3.619584,
                "observed_sessions": 21,
            },
            "liquidity": {"median_dollar_volume": 15_516_427_337.14},
            "etf_flow": {
                "latest_effective_date": None,
                "latest_training_available_session_date": None,
                "latest_fund_flow": None,
                "latest_robust_zscore": None,
                "sum_last_5_visible_flows": None,
                "sum_last_20_visible_flows": None,
                "visible_observations": 0,
            },
            "etf_flow_to_constituent": {
                "eligible_etf_count": 396,
                "excluded_etf_count": 310,
                "positive_etf_count": 61,
                "negative_etf_count": 32,
                "net_weighted_flow_rate_contribution_pct": -17.951426,
                "top_contributing_etfs": [
                    {
                        "etf_ticker": "SPY",
                        "weighted_flow_rate_contribution_pct": 0.01410532,
                        "membership_weight_percent": 6.665214315,
                        "flow_effective_date": "2026-07-24",
                        "flow_training_available_session_date": "2026-07-28",
                    }
                ],
            },
            "etf_relations": {
                "constituent_count": 0,
                "membership_count": 1401,
            },
            "quality_status": "single_source",
        },
        "interpretation": {
            "price_signal": "positive",
            "etf_flow_signal": "negative",
            "etf_flow_signal_source": "constituent_etf_flow_exposure",
            "relationship": "price_up_flow_out_divergence",
            "scope": "data_interpretation_not_trade_execution",
            "task_type": "stock_constituent_flow_analysis",
        },
        "counter_evidence": [
            "price_and_etf_flow_signals_diverge",
            "price_quality_status_single_source",
        ],
        "unknowns": ["historical_backfill_not_true_as_observed_point_in_time"],
        "regime": "price_up_flow_out_divergence",
        "confidence": 0.5,
        "conclusion": "legacy template",
    }


def aggregate() -> dict:
    base = {
        "regime": "price_up_flow_out_divergence",
        "confidence": 0.5,
        "price_signal": "positive",
        "etf_flow_signal": "negative",
    }
    return {
        "analyzed_security_count": 10,
        "task_type_counts": {"stock_constituent_flow_analysis": 10},
        "regime_counts": {
            "price_flow_positive_confirmation": 4,
            "price_flow_negative_confirmation": 1,
            "price_up_flow_out_divergence": 3,
            "price_down_flow_in_divergence": 2,
        },
        "price_signal_counts": {"positive": 7, "negative": 3},
        "etf_flow_signal_counts": {"positive": 6, "negative": 4},
        "mean_model_confidence": 0.62,
        "etf_leaders": [
            {
                **base,
                "symbol": "AAA",
                "latest_robust_zscore": 2.1,
                "latest_effective_date": "2026-07-24",
            }
        ],
        "stock_leaders": [
            {
                **base,
                "symbol": "AAPL",
                "net_weighted_flow_rate_contribution_pct": -1.2,
                "eligible_etf_count": 10,
            }
        ],
    }


def radar() -> dict:
    return {
        "integrated_rotation_clusters": [
            {
                "integrated_cluster": "Software/Cyber",
                "integrated_state": "LEADING",
                "integrated_score": 82,
                "breadth_score": 74,
                "quality_score": 90,
                "median_fmp_ret_1d": 1.1,
                "median_fmp_ret_5d": 3.2,
                "median_fmp_ret_21d": 8.4,
                "representative_tickers": ["AAA"],
            }
        ],
        "accumulation_clusters": [
            {
                "accum_cluster": "AI Broad",
                "selection_state": "ACCUMULATING",
                "cluster_score": 78,
                "flow_anomaly_score": 2.2,
                "flow_5d_to_assets": 1.0,
                "flow_21d_to_assets": 2.0,
                "positive_flow_count": 4,
                "confirmed_flow_count": 3,
                "top_related_stocks": ["AAPL"],
            }
        ],
    }


class QuantAiRadarQualityTest(unittest.TestCase):
    def test_security_brief_uses_exact_weighted_contribution(self):
        brief = build_security_brief(judgement())
        self.assertIn("+20.71%", brief["price"]["summary"])
        self.assertIn("-17.951426%", brief["flow"]["summary"])
        self.assertIn("SPY +0.014105%", brief["flow"]["summary"])
        self.assertNotIn("0.21162591", brief["flow"]["summary"])
        self.assertTrue(brief["relationship"]["is_divergence"])

    def test_market_semantic_gate_rejects_reversed_count_order(self):
        catalog = {
            "aggregate.price_signal_counts": {
                "negative": 40,
                "positive": 77,
            }
        }
        market = {
            "confirmations": [],
            "contradictions": [
                {
                    "evidence_id": "aggregate.price_signal_counts",
                    "interpretation": "부정 40개가 긍정 77개보다 많습니다.",
                }
            ],
        }
        self.assertIn(
            "reversed_price_signal_order:negative=40:positive=77",
            market_semantic_issues(market, catalog),
        )

    def test_symbol_contract_rejects_signal_regime_mismatch(self):
        expected = judgement()
        observed = copy.deepcopy(expected)
        observed["interpretation"]["etf_flow_signal"] = "positive"
        with self.assertRaises(ResponseContractError):
            validate_symbol_judgement(observed, expected)

    def test_market_validator_requires_diverse_grounded_evidence(self):
        catalog = {
            "aggregate.regime_counts": {"positive": 6, "negative": 4},
            "aggregate.price_signal_counts": {"positive": 7, "negative": 3},
            "oracle.rotation_cluster.Software": {
                "integrated_score": 82
            },
            "etf.AAA": {"latest_robust_zscore": 2.1},
            "stock.AAPL": {
                "net_weighted_flow_rate_contribution_pct": -1.2
            },
        }
        value = {
            "market_state": "rotation",
            "confidence": 0.7,
            "summary": "가격 폭과 ETF Flow, 회전 cluster, 구성종목 전달이 서로 다른 강도로 나타나 회전 상태로 해석합니다.",
            "confirmations": [
                {
                    "evidence_id": "aggregate.regime_counts",
                    "interpretation": "확인 국면 분포가 존재합니다.",
                },
                {
                    "evidence_id": "oracle.rotation_cluster.Software",
                    "interpretation": "Software 회전 강도가 상위권입니다.",
                },
                {
                    "evidence_id": "etf.AAA",
                    "interpretation": "AAA의 Flow 이상 강도가 확인됩니다.",
                },
            ],
            "contradictions": [
                {
                    "evidence_id": "aggregate.price_signal_counts",
                    "interpretation": "가격 신호의 폭이 한 방향으로 완전히 수렴하지 않았습니다.",
                },
                {
                    "evidence_id": "stock.AAPL",
                    "interpretation": "AAPL의 가격과 가중 Flow 방향이 엇갈립니다.",
                },
            ],
            "unknowns": ["동일 방향의 지속성은 미확인입니다."],
            "leading_etfs": ["AAA"],
            "affected_stocks": ["AAPL"],
            "scope": "market_and_security_analysis_not_trade_execution",
        }
        self.assertEqual(
            validate_market_synthesis(
                value, as_of_date="2026-07-28", catalog=catalog
            )["market_state"],
            "rotation",
        )
        broken = dict(value)
        broken["contradictions"] = [
            {
                "evidence_id": "aggregate.price_signal_counts",
                "interpretation": "가격 음수 3개가 양수 7개보다 많습니다.",
            },
            value["contradictions"][1],
        ]
        with self.assertRaises(ResponseContractError):
            validate_market_synthesis(
                broken, as_of_date="2026-07-28", catalog=catalog
            )

    def test_dashboard_and_renderer_surface_decision_grade_sections(self):
        result = {
            "symbol": "AAPL",
            "task_type": "stock_constituent_flow_analysis",
            "judgement": judgement(),
            "trace": {
                "request_sha256": "a" * 64,
                "response_sha256": "b" * 64,
            },
        }
        market = {
            "market_state": "rotation",
            "confidence": 0.7,
            "summary": "가격과 Flow의 확인 및 괴리가 공존합니다.",
            "confirmations": [{"evidence_id": "x", "interpretation": "확인"}],
            "contradictions": [{"evidence_id": "y", "interpretation": "반대"}],
            "unknowns": ["지속성"],
            "leading_etfs": ["AAA"],
            "affected_stocks": ["AAPL"],
            "scope": "market_and_security_analysis_not_trade_execution",
        }
        report = {
            "as_of_date": "2026-07-28",
            "generated_at_kst": "2026-07-30T06:00:00+09:00",
            "market_judgement": market,
            "aggregate": aggregate(),
            "oracle_market": radar(),
            "selection": {
                "full_candidate_count": 10,
                "selected_count": 1,
            },
            "market_dashboard": build_market_dashboard(aggregate(), radar()),
            "full_universe_quantitative_scan_complete": True,
            "selected_model_scope_complete": True,
            "source_status": {
                "quant_dataset": {"status": "confirmed"},
                "oracle_market_features": {"status": "confirmed"},
            },
            "market_evidence_catalog": {"x": 1, "y": 2},
        }
        report["quality_audit"] = audit_report_quality(
            report=report, results=[result]
        )
        with tempfile.TemporaryDirectory() as temporary:
            rendered = render_reports(
                run_dir=Path(temporary),
                report=report,
                results=[result],
                coverage_ledger=[
                    {
                        "symbol": "AAPL",
                        "priority_score": 10,
                        "selection_reasons": ["test"],
                    }
                ],
            )
            market_html = Path(rendered["market_report_html"]).read_text(
                encoding="utf-8"
            )
            security_html = (
                Path(temporary) / "security_reports" / "AAPL.html"
            ).read_text(encoding="utf-8")
        self.assertIn("정확 집계 기반 시장 구조", market_html)
        self.assertIn("섹터·테마 회전", market_html)
        self.assertIn("가격 구조", security_html)
        self.assertIn("ETF Flow 전달", security_html)
        self.assertIn("상위 ETF→종목 기여 경로", security_html)


if __name__ == "__main__":
    unittest.main()
