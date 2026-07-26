"""Deterministic full-run aggregation plus trained-model market synthesis."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from statistics import fmean
from typing import Any, Mapping, Sequence

from .model_runtime import (
    ResponseContractError,
    TRADE_DIRECTIVE_PATTERN,
    TrainedQuantClient,
    canonical_json,
)


MARKET_STATES = {"risk_on", "risk_off", "rotation", "mixed", "insufficient_evidence"}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_judgements(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regimes: Counter[str] = Counter()
    price_signals: Counter[str] = Counter()
    flow_signals: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    confidences = []
    etfs = []
    stocks = []
    for item in results:
        symbol = str(item["symbol"])
        task = str(item["task_type"])
        judgement = item["judgement"]
        interpretation = judgement.get("interpretation") or {}
        facts = judgement.get("facts") or {}
        task_counts[task] += 1
        regimes[str(judgement.get("regime"))] += 1
        price_signals[str(interpretation.get("price_signal"))] += 1
        flow_signals[str(interpretation.get("etf_flow_signal"))] += 1
        confidence = _number(judgement.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)
        base = {
            "symbol": symbol,
            "regime": judgement.get("regime"),
            "confidence": confidence,
            "price_signal": interpretation.get("price_signal"),
            "etf_flow_signal": interpretation.get("etf_flow_signal"),
        }
        if task == "etf_own_flow_analysis":
            flow = facts.get("etf_flow") or {}
            base.update(
                {
                    "latest_effective_date": flow.get("latest_effective_date"),
                    "latest_flow_to_assets_pct": flow.get("latest_flow_to_assets_pct"),
                    "latest_robust_zscore": flow.get("latest_robust_zscore"),
                    "sum_last_20_visible_flows": flow.get("sum_last_20_visible_flows"),
                }
            )
            score = _number(flow.get("latest_robust_zscore"))
            if score is None:
                score = _number(flow.get("latest_flow_to_assets_pct"))
            base["ranking_magnitude"] = abs(score or 0.0)
            etfs.append(base)
        elif task == "stock_constituent_flow_analysis":
            exposure = facts.get("etf_flow_to_constituent") or {}
            base.update(
                {
                    "net_weighted_flow_rate_contribution_pct": exposure.get(
                        "net_weighted_flow_rate_contribution_pct"
                    ),
                    "eligible_etf_count": exposure.get("eligible_etf_count"),
                    "top_contributing_etfs": exposure.get("top_contributing_etfs") or [],
                }
            )
            score = _number(exposure.get("net_weighted_flow_rate_contribution_pct"))
            base["ranking_magnitude"] = abs(score or 0.0)
            stocks.append(base)
    etfs.sort(key=lambda row: (-float(row["ranking_magnitude"]), row["symbol"]))
    stocks.sort(key=lambda row: (-float(row["ranking_magnitude"]), row["symbol"]))
    return {
        "schema_version": "quant.ai_radar_aggregate.v1",
        "analyzed_security_count": len(results),
        "task_type_counts": dict(sorted(task_counts.items())),
        "regime_counts": dict(sorted(regimes.items())),
        "price_signal_counts": dict(sorted(price_signals.items())),
        "etf_flow_signal_counts": dict(sorted(flow_signals.items())),
        "mean_model_confidence": round(fmean(confidences), 6) if confidences else None,
        "presentation_policy": (
            "all eligible securities were model-analyzed; leader arrays are display-only rankings"
        ),
        "etf_leaders": etfs[:25],
        "stock_leaders": stocks[:25],
    }


def etfradar_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    tables = evidence["tables"]
    master = tables["02_ETF_MASTER"]
    eligibility = Counter(str(row.get("eligibility_tier")) for row in master)
    flow_status = Counter(str(row.get("flow_universe_status")) for row in master)

    def ranked(name: str, limit: int) -> list[dict[str, Any]]:
        rows = list(tables[name])
        rows.sort(
            key=lambda row: (
                int(_number(row.get("rank") or row.get("prompt_rank")) or 10**9),
                str(row.get("ticker") or ""),
            )
        )
        return rows[:limit]

    return {
        "release_binding": evidence["binding"],
        "full_table_row_counts": {key: len(value) for key, value in sorted(tables.items())},
        "master_eligibility_counts": dict(sorted(eligibility.items())),
        "master_flow_status_counts": dict(sorted(flow_status.items())),
        "massive_flow_fusion_top": ranked("36_MASSIVE_FLOW_FUSION", 12),
        "early_accumulation_top": ranked("47_EARLY_ACCUMULATION_RADAR", 12),
        "accumulation_clusters": ranked("48_MASSIVE_ACCUM_CLUSTER", 15),
        "integrated_rotation_clusters": ranked("49_INTEGRATED_ROTATION_RADAR", 15),
    }


def _evidence_catalog(aggregate: Mapping[str, Any], radar: Mapping[str, Any]) -> dict[str, Any]:
    catalog: dict[str, Any] = {
        "aggregate.analyzed_security_count": aggregate["analyzed_security_count"],
        "aggregate.task_type_counts": aggregate["task_type_counts"],
        "aggregate.regime_counts": aggregate["regime_counts"],
        "aggregate.price_signal_counts": aggregate["price_signal_counts"],
        "aggregate.etf_flow_signal_counts": aggregate["etf_flow_signal_counts"],
        "aggregate.mean_model_confidence": aggregate["mean_model_confidence"],
        "etfradar.release_id": radar["release_binding"]["release_id"],
        "etfradar.trade_date_us": radar["release_binding"]["trade_date_us"],
        "etfradar.master_eligibility_counts": radar["master_eligibility_counts"],
        "etfradar.master_flow_status_counts": radar["master_flow_status_counts"],
    }
    for row in aggregate["etf_leaders"][:12]:
        catalog[f"etf.{row['symbol']}"] = row
    for row in aggregate["stock_leaders"][:12]:
        catalog[f"stock.{row['symbol']}"] = row
    for row in radar["accumulation_clusters"][:10]:
        key = str(row.get("accum_cluster") or row.get("rank"))
        catalog[f"etfradar.accumulation_cluster.{key}"] = row
    for row in radar["integrated_rotation_clusters"][:10]:
        key = str(row.get("integrated_cluster") or row.get("rank"))
        catalog[f"etfradar.rotation_cluster.{key}"] = row
    return catalog


def validate_market_synthesis(
    value: Mapping[str, Any], *, as_of_date: str, catalog: Mapping[str, Any]
) -> dict[str, Any]:
    required = (
        "market_state",
        "confidence",
        "summary",
        "confirmations",
        "contradictions",
        "unknowns",
        "leading_etfs",
        "affected_stocks",
        "scope",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ResponseContractError(f"market synthesis is missing fields: {missing}")
    if value.get("market_state") not in MARKET_STATES:
        raise ResponseContractError("market synthesis has an invalid market_state")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
        raise ResponseContractError("market synthesis confidence is outside [0,1]")
    if value.get("scope") != "market_and_security_analysis_not_trade_execution":
        raise ResponseContractError("market synthesis escaped the analysis-only scope")
    for key in ("confirmations", "contradictions"):
        rows = value.get(key)
        if not isinstance(rows, list) or not rows:
            raise ResponseContractError(f"market synthesis {key} must be a nonempty array")
        for row in rows:
            if not isinstance(row, dict) or row.get("evidence_id") not in catalog:
                raise ResponseContractError(f"market synthesis {key} cited unknown evidence")
    if not isinstance(value.get("unknowns"), list):
        raise ResponseContractError("market synthesis unknowns must be an array")
    allowed_etfs = {
        key.removeprefix("etf.") for key in catalog if key.startswith("etf.")
    }
    allowed_stocks = {
        key.removeprefix("stock.") for key in catalog if key.startswith("stock.")
    }
    if not set(value.get("leading_etfs") or []).issubset(allowed_etfs):
        raise ResponseContractError("market synthesis introduced an unranked ETF")
    if not set(value.get("affected_stocks") or []).issubset(allowed_stocks):
        raise ResponseContractError("market synthesis introduced an unranked stock")
    text = canonical_json(value)
    for raw_date in __import__("re").findall(r"\b20\d{2}-\d{2}-\d{2}\b", text):
        if date.fromisoformat(raw_date) > date.fromisoformat(as_of_date):
            raise ResponseContractError("market synthesis contains a post-as-of date")
    if TRADE_DIRECTIVE_PATTERN.search(text):
        raise ResponseContractError("market synthesis contains a trade directive")
    return dict(value)


def synthesize_market(
    *,
    client: TrainedQuantClient,
    as_of_date: str,
    aggregate: Mapping[str, Any],
    radar: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    catalog = _evidence_catalog(aggregate, radar)
    schema = {
        "market_state": "risk_on|risk_off|rotation|mixed|insufficient_evidence",
        "confidence": "0..1",
        "summary": "Korean market interpretation without uncited numbers",
        "confirmations": [{"evidence_id": "catalog id", "interpretation": "Korean"}],
        "contradictions": [{"evidence_id": "catalog id", "interpretation": "Korean"}],
        "unknowns": ["Korean limitation"],
        "leading_etfs": ["symbols drawn only from etf.* catalog entries"],
        "affected_stocks": ["symbols drawn only from stock.* catalog entries"],
        "scope": "market_and_security_analysis_not_trade_execution",
    }
    system = (
        "You are the released quant-analysis LoRA. Interpret only the supplied as-of "
        "catalog produced from the complete eligible-security inference run and the "
        "hash-verified ETF RADAR release. Separate confirmation, contradiction, and "
        "unknowns. Never issue a trade instruction or invent a number. /no_think"
    )
    user = (
        f"AS_OF_DATE={as_of_date}\nEVIDENCE_CATALOG={canonical_json(catalog)}\n"
        f"Return only one JSON object with this schema: {canonical_json(schema)}"
    )
    response, trace = client.complete(system=system, user=user, max_tokens=1600)
    return validate_market_synthesis(
        response, as_of_date=as_of_date, catalog=catalog
    ), trace, catalog
