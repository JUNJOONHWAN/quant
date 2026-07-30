"""Grounded decision-support views and quality gates for Quant AI Radar.

The trained model remains responsible for the regime, confidence, evidence
selection, and bounded conclusion.  This module turns the immutable facts into
exactly rendered evidence cards, checks model prose against those facts, and
scores whether a report is useful enough to publish as a reference artifact.
It never creates a trade instruction or changes model-supplied facts.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from statistics import fmean
from typing import Any, Mapping, Sequence

from quant_dataset.etf_flow_exposure import (
    ETF_CONSTITUENT_FLOW_POLICY_ID,
    MAX_ABSOLUTE_FLOW_TO_NET_ASSETS_PCT,
    MAX_FLOW_OBSERVATION_AGE_CALENDAR_DAYS,
    flow_age_calendar_days,
)

QUALITY_SCHEMA_VERSION = "quant.ai_radar_quality_audit.v2"
SECURITY_BRIEF_SCHEMA_VERSION = "quant.ai_radar_security_brief.v1"
MARKET_DASHBOARD_SCHEMA_VERSION = "quant.ai_radar_market_dashboard.v2"
MIN_QUALITY_SCORE = 8.0
MIN_MARKET_SUMMARY_KOREAN_CHARS = 40
MIN_MARKET_UNKNOWN_KOREAN_CHARS = 8
MARKET_META_LANGUAGE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"근거를\s*제시하지\s*않",
        r"한국어\s*(?:제한|로\s*(?:작성|해석|요약))",
        r"(?:요청|지시|프롬프트|스키마|JSON).*(?:따르|작성|출력)",
        r"(?:시장|데이터).*(?:해석하며|설명하며).*(?:근거|한국어)",
    )
)
MARKET_CONCEPT_GROUPS = (
    re.compile(r"가격|시장\s*폭|강세|약세"),
    re.compile(r"ETF|Flow|플로우|자금"),
    re.compile(r"회전|로테이션|섹터|테마"),
    re.compile(r"괴리|분화|반대|위험|불확실"),
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 2) -> str:
    number = _number(value)
    if number is None:
        return "확인 불가"
    rendered = f"{number:,.{digits}f}"
    return rendered.rstrip("0").rstrip(".")


def _pct(value: Any, digits: int = 2) -> str:
    number = _number(value)
    if number is None:
        return "확인 불가"
    prefix = "+" if number > 0 else ""
    return f"{prefix}{_fmt(number, digits)}%"


def _direction(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "미확인"
    if number > 0:
        return "양수"
    if number < 0:
        return "음수"
    return "중립"


def _regime_label(regime: str) -> str:
    return {
        "price_flow_positive_confirmation": "가격·ETF Flow 동반 강세",
        "price_flow_negative_confirmation": "가격·ETF Flow 동반 약세",
        "price_up_flow_out_divergence": "가격 상승·ETF Flow 유출 괴리",
        "price_down_flow_in_divergence": "가격 하락·ETF Flow 유입 괴리",
        "mixed_or_flat": "가격·ETF Flow 혼조",
        "insufficient_joint_evidence": "공동 증거 부족",
    }.get(regime, regime or "미확인")


def _evidence(
    evidence_id: str,
    label: str,
    value: Any,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "label": label,
        "value": value,
        "interpretation": interpretation,
    }


def build_security_brief(judgement: Mapping[str, Any]) -> dict[str, Any]:
    """Build an exact, readable fact view without replacing model judgement."""

    facts = judgement.get("facts") or {}
    interpretation = judgement.get("interpretation") or {}
    price = facts.get("price") or {}
    liquidity = facts.get("liquidity") or {}
    flow = facts.get("etf_flow") or {}
    exposure = facts.get("etf_flow_to_constituent") or {}
    relations = facts.get("etf_relations") or {}
    symbol = str(facts.get("symbol") or "")
    as_of = str(facts.get("as_of_date") or "")
    regime = str(judgement.get("regime") or interpretation.get("relationship") or "")
    task = str(interpretation.get("task_type") or "")

    ret_1 = _number(price.get("return_1_session_pct"))
    ret_5 = _number(price.get("return_5_session_pct"))
    ret_20 = _number(price.get("return_20_session_pct"))
    volatility = _number(price.get("annualized_realized_volatility_pct"))
    drawdown = _number(price.get("max_drawdown_in_packet_pct"))
    price_summary = (
        f"가격은 1일 {_pct(ret_1)}, 5일 {_pct(ret_5)}, 20일 {_pct(ret_20)}입니다. "
        f"연환산 실현변동성은 {_pct(volatility)}이고 관측 구간 최대 낙폭은 "
        f"{_pct(drawdown)}입니다."
    )

    top_contributors = []
    for row in (exposure.get("top_contributing_etfs") or [])[:5]:
        top_contributors.append(
            {
                "etf_ticker": row.get("etf_ticker"),
                "weighted_flow_rate_contribution_pct": row.get(
                    "weighted_flow_rate_contribution_pct"
                ),
                "membership_weight_percent": row.get("membership_weight_percent"),
                "flow_effective_date": row.get("flow_effective_date"),
                "flow_training_available_session_date": row.get(
                    "flow_training_available_session_date"
                ),
            }
        )

    confirmations: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    if ret_20 is not None:
        bucket = confirmations if (
            (ret_20 > 0 and interpretation.get("price_signal") == "positive")
            or (ret_20 < 0 and interpretation.get("price_signal") == "negative")
        ) else contradictions
        bucket.append(
            _evidence(
                "price.return_20_session_pct",
                "20일 가격 변화",
                ret_20,
                f"20일 가격 방향은 {_direction(ret_20)}입니다.",
            )
        )

    if task == "etf_own_flow_analysis":
        latest_flow = _number(flow.get("latest_fund_flow"))
        robust_z = _number(flow.get("latest_robust_zscore"))
        sum_5 = _number(flow.get("sum_last_5_visible_flows"))
        sum_20 = _number(flow.get("sum_last_20_visible_flows"))
        flow_summary = (
            "ETF 자체 Flow는 최신 가시 유효일 "
            f"{flow.get('latest_effective_date') or '확인 불가'}, 최신 provider-reported "
            f"flow {_fmt(latest_flow)}, 5개 관측 합계 {_fmt(sum_5)}, "
            f"20개 관측 합계 {_fmt(sum_20)}, robust z-score {_fmt(robust_z)}입니다. "
            "통화가 명시되지 않은 원시 금액은 USD로 해석하지 않습니다."
        )
        flow_metric = robust_z if robust_z is not None else latest_flow
        if flow_metric is not None:
            bucket = confirmations if (
                (flow_metric > 0 and interpretation.get("etf_flow_signal") == "positive")
                or (flow_metric < 0 and interpretation.get("etf_flow_signal") == "negative")
            ) else contradictions
            bucket.append(
                _evidence(
                    "etf_flow.latest_robust_zscore",
                    "최신 ETF Flow robust z-score",
                    robust_z,
                    f"ETF 자체 Flow 방향은 {_direction(flow_metric)}입니다.",
                )
            )
        flow_mode = "own_etf_flow"
    else:
        net = _number(exposure.get("net_weighted_flow_rate_contribution_pct"))
        eligible = int(_number(exposure.get("eligible_etf_count")) or 0)
        excluded = int(_number(exposure.get("excluded_etf_count")) or 0)
        positive = int(_number(exposure.get("positive_etf_count")) or 0)
        negative = int(_number(exposure.get("negative_etf_count")) or 0)
        leaders = ", ".join(
            f"{row.get('etf_ticker')} {_pct(row.get('weighted_flow_rate_contribution_pct'), 6)}"
            for row in top_contributors
            if row.get("etf_ticker")
        ) or "확인 불가"
        flow_summary = (
            f"구성종목 전달 Flow 기여도는 {_pct(net, 6)}입니다. "
            f"가시성·품질 조건을 통과한 ETF {eligible}개 중 양수 {positive}개, "
            f"음수 {negative}개이며 제외 ETF는 {excluded}개입니다. "
            f"상위 기여 ETF는 {leaders}입니다."
        )
        if net is not None:
            bucket = confirmations if (
                (net > 0 and interpretation.get("etf_flow_signal") == "positive")
                or (net < 0 and interpretation.get("etf_flow_signal") == "negative")
            ) else contradictions
            bucket.append(
                _evidence(
                    "etf_flow_to_constituent.net_weighted_flow_rate_contribution_pct",
                    "ETF→종목 가중 Flow 기여도",
                    net,
                    f"구성종목 전달 Flow 방향은 {_direction(net)}입니다.",
                )
            )
        flow_mode = "constituent_etf_flow_exposure"

    is_divergence = "divergence" in regime
    relationship_summary = (
        f"{_regime_label(regime)}로 분류됩니다. "
        + (
            "가격과 ETF Flow 방향이 엇갈리므로 한쪽 신호만으로 시장 원인을 "
            "확정할 수 없습니다."
            if is_divergence
            else "가격과 ETF Flow 방향이 일치하지만 동일 방향성만으로 인과관계를 "
            "확정할 수는 없습니다."
        )
    )
    if is_divergence:
        contradictions.append(
            _evidence(
                "relationship.price_vs_etf_flow",
                "가격·ETF Flow 관계",
                regime,
                "두 신호의 방향이 서로 다릅니다.",
            )
        )
    else:
        confirmations.append(
            _evidence(
                "relationship.price_vs_etf_flow",
                "가격·ETF Flow 관계",
                regime,
                "두 신호의 방향이 일치합니다.",
            )
        )

    quality_status = str(facts.get("quality_status") or "unknown")
    observed_sessions = int(_number(price.get("observed_sessions")) or 0)
    visible_flow_observations = int(_number(flow.get("visible_observations")) or 0)
    has_flow_basis = visible_flow_observations > 0 or int(
        _number(exposure.get("eligible_etf_count")) or 0
    ) > 0
    strength = 40.0
    strength += min(observed_sessions, 21) / 21.0 * 25.0
    strength += 20.0 if has_flow_basis else 0.0
    strength += 15.0 if quality_status == "pass" else 7.5 if quality_status == "single_source" else 0.0
    strength = round(min(strength, 100.0), 1)

    unknowns = list(judgement.get("unknowns") or [])
    if quality_status != "pass" and "price_quality_not_cross_source_confirmed" not in unknowns:
        unknowns.append("price_quality_not_cross_source_confirmed")
    if facts.get("quality_status") == "single_source":
        quality_note = "가격 품질은 단일 소스 상태여서 교차 소스 확정이 아닙니다."
    else:
        quality_note = f"가격 품질 상태는 {quality_status}입니다."

    headline = f"{symbol} · {_regime_label(regime)}"
    conclusion = (
        f"{symbol}은 기준일 {as_of}에 {_regime_label(regime)}입니다. "
        f"가격 20일 {_pct(ret_20)}와 ETF Flow 신호를 함께 보면 "
        f"{'방향 불일치의 원인을 더 확인해야 합니다.' if is_divergence else '방향 확인은 있으나 인과와 지속성은 미확인입니다.'}"
    )
    return {
        "schema_version": SECURITY_BRIEF_SCHEMA_VERSION,
        "symbol": symbol,
        "as_of_date": as_of,
        "task_type": task,
        "headline": headline,
        "price": {
            "summary": price_summary,
            "return_1_session_pct": ret_1,
            "return_5_session_pct": ret_5,
            "return_20_session_pct": ret_20,
            "annualized_realized_volatility_pct": volatility,
            "max_drawdown_in_packet_pct": drawdown,
            "latest_close": price.get("latest_close"),
            "observed_sessions": observed_sessions,
            "median_dollar_volume": liquidity.get("median_dollar_volume"),
        },
        "flow": {
            "mode": flow_mode,
            "summary": flow_summary,
            "latest_effective_date": flow.get("latest_effective_date"),
            "latest_available_session_date": flow.get(
                "latest_training_available_session_date"
            ),
            "latest_fund_flow_provider_units": flow.get("latest_fund_flow"),
            "latest_robust_zscore": flow.get("latest_robust_zscore"),
            "sum_last_5_visible_flows_provider_units": flow.get(
                "sum_last_5_visible_flows"
            ),
            "sum_last_20_visible_flows_provider_units": flow.get(
                "sum_last_20_visible_flows"
            ),
            "net_weighted_flow_rate_contribution_pct": exposure.get(
                "net_weighted_flow_rate_contribution_pct"
            ),
            "eligible_etf_count": exposure.get("eligible_etf_count"),
            "excluded_etf_count": exposure.get("excluded_etf_count"),
            "positive_etf_count": exposure.get("positive_etf_count"),
            "negative_etf_count": exposure.get("negative_etf_count"),
            "top_contributing_etfs": top_contributors,
        },
        "relationship": {
            "regime": regime,
            "label": _regime_label(regime),
            "summary": relationship_summary,
            "is_divergence": is_divergence,
            "constituent_count": relations.get("constituent_count"),
            "membership_count": relations.get("membership_count"),
        },
        "confirmations": confirmations,
        "contradictions": contradictions,
        "unknowns": unknowns,
        "data_quality": {
            "status": quality_status,
            "note": quality_note,
            "evidence_strength_score": strength,
        },
        "confirmation_conditions": [
            "다음 가시 세션에서 가격과 ETF Flow 방향이 유지되는지 확인",
            "상위 기여 ETF의 집중도가 넓은 ETF군으로 확산되는지 확인",
            "단일 소스 가격 상태가 교차 소스로 확인되는지 확인",
        ],
        "conclusion": conclusion,
        "scope": "data_interpretation_not_trade_execution",
    }


def model_analysis_target(judgement: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact learned-writing target embedded in interpretation."""

    brief = build_security_brief(judgement)
    evidence_ids = [
        str(item["evidence_id"])
        for item in (*brief["confirmations"], *brief["contradictions"])
    ]
    return {
        "schema_version": "quant.learned_security_analysis.v1",
        "headline": brief["headline"],
        "price_context": brief["price"]["summary"],
        "flow_context": brief["flow"]["summary"],
        "relationship_context": brief["relationship"]["summary"],
        "evidence_ids": evidence_ids,
        "confidence_context": brief["data_quality"]["note"],
        "conclusion": brief["conclusion"],
    }


def market_semantic_issues(
    market: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> list[str]:
    """Detect numeric and directional contradictions in market prose."""

    issues: list[str] = []
    summary = str(market.get("summary") or "").strip()
    korean_character_count = len(re.findall(r"[가-힣]", summary))
    if korean_character_count < MIN_MARKET_SUMMARY_KOREAN_CHARS:
        issues.append(
            f"market_summary_too_shallow:korean_chars={korean_character_count}"
        )
    if any(pattern.search(summary) for pattern in MARKET_META_LANGUAGE_PATTERNS):
        issues.append("market_summary_contains_meta_language")
    concept_count = sum(
        1 for pattern in MARKET_CONCEPT_GROUPS if pattern.search(summary)
    )
    if concept_count < 3:
        issues.append(f"market_summary_missing_core_concepts:{concept_count}")
    for item in market.get("unknowns") or []:
        text = str(item).strip()
        korean_count = len(re.findall(r"[가-힣]", text))
        if korean_count < MIN_MARKET_UNKNOWN_KOREAN_CHARS:
            issues.append(
                f"market_unknown_too_generic:korean_chars={korean_count}"
            )
        if any(pattern.search(text) for pattern in MARKET_META_LANGUAGE_PATTERNS):
            issues.append("market_unknown_contains_meta_language")

    rows = [
        *list(market.get("confirmations") or []),
        *list(market.get("contradictions") or []),
    ]
    for row in rows:
        evidence_id = str(row.get("evidence_id") or "")
        text = str(row.get("interpretation") or "")
        evidence = catalog.get(evidence_id)
        if isinstance(evidence, Mapping):
            allowed_numbers = {
                float(value)
                for value in evidence.values()
                if _number(value) is not None
            }
            mentioned = [
                float(value.replace(",", ""))
                for value in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", text)
            ]
            for number in mentioned:
                if number not in allowed_numbers:
                    issues.append(
                        f"uncited_numeric_claim:{evidence_id}:{number:g}"
                    )

        if evidence_id == "aggregate.price_signal_counts" and isinstance(
            evidence, Mapping
        ):
            positive = int(_number(evidence.get("positive")) or 0)
            negative = int(_number(evidence.get("negative")) or 0)
            compact = re.sub(r"\s+", "", text)
            negative_claimed_more = bool(
                re.search(r"(?:부정|음수).*?(?:긍정|양수).*(?:보다)?많", compact)
            )
            positive_claimed_more = bool(
                re.search(r"(?:긍정|양수).*?(?:부정|음수).*(?:보다)?많", compact)
            )
            if negative_claimed_more and not negative > positive:
                issues.append(
                    "reversed_price_signal_order:"
                    f"negative={negative}:positive={positive}"
                )
            if positive_claimed_more and not positive > negative:
                issues.append(
                    "reversed_price_signal_order:"
                    f"positive={positive}:negative={negative}"
                )
    return sorted(set(issues))


def build_market_dashboard(
    aggregate: Mapping[str, Any],
    radar: Mapping[str, Any],
) -> dict[str, Any]:
    total = int(_number(aggregate.get("analyzed_security_count")) or 0)
    prices = aggregate.get("price_signal_counts") or {}
    flows = aggregate.get("etf_flow_signal_counts") or {}
    regimes = aggregate.get("regime_counts") or {}
    positive_price = int(_number(prices.get("positive")) or 0)
    negative_price = int(_number(prices.get("negative")) or 0)
    positive_flow = int(_number(flows.get("positive")) or 0)
    negative_flow = int(_number(flows.get("negative")) or 0)
    positive_confirmation = int(
        _number(regimes.get("price_flow_positive_confirmation")) or 0
    )
    negative_confirmation = int(
        _number(regimes.get("price_flow_negative_confirmation")) or 0
    )
    divergence = int(
        _number(regimes.get("price_up_flow_out_divergence")) or 0
    ) + int(_number(regimes.get("price_down_flow_in_divergence")) or 0)
    confirmation = positive_confirmation + negative_confirmation

    rotation = []
    for row in (radar.get("integrated_rotation_clusters") or [])[:10]:
        rotation.append(
            {
                "cluster": row.get("integrated_cluster"),
                "state": row.get("integrated_state"),
                "score": row.get("integrated_score"),
                "breadth_score": row.get("breadth_score"),
                "quality_score": row.get("quality_score"),
                "median_return_1d_pct": row.get("median_fmp_ret_1d"),
                "median_return_5d_pct": row.get("median_fmp_ret_5d"),
                "median_return_21d_pct": row.get("median_fmp_ret_21d"),
                "representative_tickers": row.get("representative_tickers") or [],
            }
        )
    accumulation = []
    for row in (radar.get("accumulation_clusters") or [])[:10]:
        accumulation.append(
            {
                "cluster": row.get("accum_cluster"),
                "state": row.get("selection_state"),
                "score": row.get("cluster_score"),
                "flow_anomaly_score": row.get("flow_anomaly_score"),
                "flow_5d_to_assets": row.get("flow_5d_to_assets"),
                "flow_21d_to_assets": row.get("flow_21d_to_assets"),
                "positive_flow_count": row.get("positive_flow_count"),
                "confirmed_flow_count": row.get("confirmed_flow_count"),
                "top_related_stocks": row.get("top_related_stocks") or [],
            }
        )
    def lane(
        rows: Sequence[Mapping[str, Any]],
        regimes: set[str],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        selected = []
        for row in rows:
            if str(row.get("regime") or "") not in regimes:
                continue
            selected.append(
                {
                    key: row.get(key)
                    for key in (
                        "symbol",
                        "regime",
                        "confidence",
                        "price_signal",
                        "etf_flow_signal",
                        "latest_effective_date",
                        "latest_robust_zscore",
                        "net_weighted_flow_rate_contribution_pct",
                        "eligible_etf_count",
                    )
                    if row.get(key) is not None
                }
            )
            if len(selected) >= limit:
                break
        return selected

    etf_leaders = list(aggregate.get("etf_leaders") or [])
    stock_leaders = list(aggregate.get("stock_leaders") or [])
    candidate_rankings = aggregate.get("candidate_rankings") or {}
    etfs_by_regime = candidate_rankings.get("etfs_by_regime") or {}
    stocks_by_regime = candidate_rankings.get("stocks_by_regime") or {}
    positive = {"price_flow_positive_confirmation"}
    negative = {"price_flow_negative_confirmation"}
    divergence_regimes = {
        "price_up_flow_out_divergence",
        "price_down_flow_in_divergence",
    }
    candidate_lanes = {
        "positive_confirmation_etfs": lane(
            etfs_by_regime.get("price_flow_positive_confirmation", etf_leaders),
            positive,
        ),
        "positive_confirmation_stocks": lane(
            stocks_by_regime.get(
                "price_flow_positive_confirmation", stock_leaders
            ),
            positive,
        ),
        "negative_confirmation_etfs": lane(
            etfs_by_regime.get("price_flow_negative_confirmation", etf_leaders),
            negative,
        ),
        "negative_confirmation_stocks": lane(
            stocks_by_regime.get(
                "price_flow_negative_confirmation", stock_leaders
            ),
            negative,
        ),
        "divergence_etfs": lane(
            [
                *etfs_by_regime.get("price_up_flow_out_divergence", []),
                *etfs_by_regime.get("price_down_flow_in_divergence", []),
            ]
            or etf_leaders,
            divergence_regimes,
        ),
        "divergence_stocks": lane(
            [
                *stocks_by_regime.get("price_up_flow_out_divergence", []),
                *stocks_by_regime.get("price_down_flow_in_divergence", []),
            ]
            or stock_leaders,
            divergence_regimes,
        ),
        "policy": (
            "confirmation lanes are reference watchlists, not buy/sell orders; "
            "positive requires price and ETF Flow positive, negative requires both "
            "negative, and divergence remains a separate verification lane"
        ),
    }
    return {
        "schema_version": MARKET_DASHBOARD_SCHEMA_VERSION,
        "analyzed_security_count": total,
        "breadth": {
            "price_positive_count": positive_price,
            "price_negative_count": negative_price,
            "price_positive_pct": round(positive_price / total * 100, 2)
            if total
            else None,
            "etf_flow_positive_count": positive_flow,
            "etf_flow_negative_count": negative_flow,
            "etf_flow_positive_pct": round(positive_flow / total * 100, 2)
            if total
            else None,
            "confirmation_count": confirmation,
            "divergence_count": divergence,
            "confirmation_pct": round(confirmation / total * 100, 2)
            if total
            else None,
            "divergence_pct": round(divergence / total * 100, 2)
            if total
            else None,
        },
        "regime_counts": dict(regimes),
        "rotation_clusters": rotation,
        "accumulation_clusters": accumulation,
        "leading_etfs": list(aggregate.get("etf_leaders") or [])[:12],
        "affected_stocks": list(aggregate.get("stock_leaders") or [])[:12],
        "candidate_lanes": candidate_lanes,
        "mean_model_confidence": aggregate.get("mean_model_confidence"),
        "interpretation": (
            f"가격 양수 비중 {_pct(positive_price / total * 100 if total else None)}와 "
            f"ETF Flow 양수 비중 {_pct(positive_flow / total * 100 if total else None)}를 "
            f"함께 봅니다. 확인 국면 {confirmation}건과 괴리 국면 {divergence}건이 "
            "비슷하면 단일 방향보다 회전·분화를 우선 해석합니다."
        ),
    }


def audit_report_quality(
    *,
    report: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score explicit publish gates; every category must be at least 8/10."""

    sources = report.get("source_status") or {}
    source_states = [
        str((sources.get(name) or {}).get("status") or "")
        for name in ("quant_dataset", "oracle_market_features")
    ]
    data_integrity = 10.0 if (
        report.get("full_universe_quantitative_scan_complete") is True
        and report.get("selected_model_scope_complete") is True
        and all(state == "confirmed" for state in source_states)
    ) else 5.0

    catalog = report.get("market_evidence_catalog") or {}
    semantic_issues = market_semantic_issues(
        report.get("market_judgement") or {},
        catalog,
    )
    numeric_faithfulness = 10.0 if not semantic_issues else 0.0

    as_of_date = str(report.get("as_of_date") or "")
    flow_quality_issues: list[str] = []
    for item in results:
        judgement = item.get("judgement") or {}
        facts = judgement.get("facts") or {}
        symbol = str(item.get("symbol") or facts.get("symbol") or "")
        own_flow = facts.get("etf_flow") or {}
        own_effective = own_flow.get("latest_effective_date")
        if own_effective:
            age = flow_age_calendar_days(as_of_date, own_effective)
            if (
                age is None
                or age < 0
                or age > MAX_FLOW_OBSERVATION_AGE_CALENDAR_DAYS
            ):
                flow_quality_issues.append(
                    f"{symbol}:stale_or_invalid_own_etf_flow:{own_effective}"
                )
            own_rate = _number(own_flow.get("latest_flow_to_assets_pct"))
            if (
                own_rate is not None
                and abs(own_rate) > MAX_ABSOLUTE_FLOW_TO_NET_ASSETS_PCT
            ):
                flow_quality_issues.append(
                    f"{symbol}:own_etf_flow_rate_outside_gate:{own_rate:g}"
                )
        exposure = facts.get("etf_flow_to_constituent") or {}
        if int(_number(exposure.get("eligible_etf_count")) or 0) > 0 and (
            exposure.get("policy_id") != ETF_CONSTITUENT_FLOW_POLICY_ID
        ):
            flow_quality_issues.append(
                f"{symbol}:constituent_flow_policy_not_current"
            )
        for row in exposure.get("top_contributing_etfs") or []:
            effective = row.get("flow_effective_date")
            age = flow_age_calendar_days(as_of_date, effective)
            if (
                age is None
                or age < 0
                or age > MAX_FLOW_OBSERVATION_AGE_CALENDAR_DAYS
            ):
                flow_quality_issues.append(
                    f"{symbol}:{row.get('etf_ticker')}:stale_or_invalid_flow:"
                    f"{effective}"
                )
            rate = _number(row.get("flow_to_estimated_net_assets_pct"))
            if (
                rate is not None
                and abs(rate) > MAX_ABSOLUTE_FLOW_TO_NET_ASSETS_PCT
            ):
                flow_quality_issues.append(
                    f"{symbol}:{row.get('etf_ticker')}:flow_rate_outside_gate:"
                    f"{rate:g}"
                )
    flow_evidence_quality = 10.0 if not flow_quality_issues else 0.0

    briefs = [build_security_brief(item.get("judgement") or {}) for item in results]
    complete_briefs = [
        item
        for item in briefs
        if item.get("headline")
        and item.get("price", {}).get("summary")
        and item.get("flow", {}).get("summary")
        and item.get("relationship", {}).get("summary")
        and item.get("conclusion")
    ]
    security_analysis = round(
        min(10.0, len(complete_briefs) / max(len(results), 1) * 10.0),
        2,
    )

    radar = report.get("oracle_market") or {}
    aggregate = report.get("aggregate") or {}
    dashboard = build_market_dashboard(aggregate, radar)
    market_structure_checks = (
        bool(dashboard["breadth"]),
        bool(dashboard["rotation_clusters"]),
        bool(dashboard["accumulation_clusters"]),
        bool(dashboard["leading_etfs"]),
        bool(dashboard["affected_stocks"]),
        bool(dashboard["candidate_lanes"]),
    )
    market_structure = round(
        sum(market_structure_checks) / len(market_structure_checks) * 10.0,
        2,
    )

    conclusions = [
        str((item.get("judgement") or {}).get("conclusion") or "")
        for item in results
    ]
    expected_regime = {
        ("positive", "positive"): "price_flow_positive_confirmation",
        ("negative", "negative"): "price_flow_negative_confirmation",
        ("positive", "negative"): "price_up_flow_out_divergence",
        ("negative", "positive"): "price_down_flow_in_divergence",
    }
    model_consistent = 0
    for item in results:
        judgement = item.get("judgement") or {}
        interpretation = judgement.get("interpretation") or {}
        regime = str(judgement.get("regime") or "")
        pair = (
            str(interpretation.get("price_signal") or ""),
            str(interpretation.get("etf_flow_signal") or ""),
        )
        confidence = _number(judgement.get("confidence"))
        trace = item.get("trace") or {}
        if (
            regime == interpretation.get("relationship")
            and expected_regime.get(pair, regime) == regime
            and confidence is not None
            and 0 <= confidence <= 1
            and isinstance(judgement.get("facts"), Mapping)
            and bool(trace.get("request_sha256"))
            and bool(trace.get("response_sha256"))
        ):
            model_consistent += 1
    model_judgement_integration = round(
        model_consistent / max(len(results), 1) * 10.0,
        2,
    )

    narratives = report.get("multistage_narratives") or {}
    expected_sector_count = len(radar.get("integrated_rotation_clusters") or [])
    sector_explanations = narratives.get("sector_explanations") or []
    security_explanations = narratives.get("security_explanations") or []
    editorial = narratives.get("editorial") or {}
    requires_narratives = report.get("schema_version") == "quant.ai_radar_report.v2"
    narrative_checks = (
        not requires_narratives
        or narratives.get("schema_version")
        == "quant.ai_radar_multistage_narratives.v1",
        not requires_narratives
        or (
            len(sector_explanations) == expected_sector_count
            and expected_sector_count > 0
        ),
        not requires_narratives or bool(security_explanations),
        not requires_narratives
        or all(
            bool(editorial.get(key))
            for key in (
                "headline",
                "executive_summary",
                "rotation_summary",
                "selection_summary",
                "risk_summary",
            )
        ),
        not requires_narratives
        or int(narratives.get("model_call_count") or 0) >= 3,
    )
    multistage_explanation = round(
        sum(narrative_checks) / len(narrative_checks) * 10.0,
        2,
    )

    report_usability_checks = (
        bool(report.get("market_dashboard")),
        bool(report.get("rendered_reports")),
        bool(results),
        bool(report.get("market_judgement", {}).get("summary")),
        bool(report.get("market_judgement", {}).get("confirmations")),
        not requires_narratives or bool(narratives),
    )
    report_usability = round(
        sum(report_usability_checks) / len(report_usability_checks) * 10.0,
        2,
    )

    scores = {
        "data_integrity": data_integrity,
        "numeric_faithfulness": numeric_faithfulness,
        "flow_evidence_quality": flow_evidence_quality,
        "security_analysis": security_analysis,
        "market_structure": market_structure,
        "model_judgement_integration": model_judgement_integration,
        "multistage_explanation": multistage_explanation,
        "report_usability": report_usability,
    }
    failed = sorted(
        name for name, score in scores.items() if score < MIN_QUALITY_SCORE
    )
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "green" if not failed else "red",
        "minimum_required_score": MIN_QUALITY_SCORE,
        "scores": scores,
        "failed_categories": failed,
        "semantic_issues": semantic_issues,
        "flow_quality_issues": sorted(set(flow_quality_issues)),
        "security_report_count": len(results),
        "complete_security_brief_count": len(complete_briefs),
        "model_consistent_judgement_count": model_consistent,
        "multistage_narrative_checks": {
            "schema": narrative_checks[0],
            "all_rotation_clusters": narrative_checks[1],
            "material_securities": narrative_checks[2],
            "editorial": narrative_checks[3],
            "minimum_model_calls": narrative_checks[4],
        },
        "unique_model_conclusion_count": len(set(conclusions)),
        "mean_evidence_strength_score": round(
            fmean(
                item["data_quality"]["evidence_strength_score"]
                for item in briefs
            ),
            2,
        )
        if briefs
        else None,
        "model_conclusion_frequency": dict(
            Counter(conclusions).most_common(20)
        ),
        "publishable_reference_report": not failed,
        "scope": "analysis_quality_not_trade_execution",
    }


def canonical_quality_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
