"""Deterministic full-run aggregation plus trained-model market synthesis."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from statistics import fmean
from typing import Any, Mapping, Sequence

from .model_runtime import (
    ModelResponseParseError,
    ResponseContractError,
    TRADE_DIRECTIVE_PATTERN,
    TrainedQuantClient,
    canonical_json,
)
from .decision_support import market_semantic_issues


MARKET_STATES = {"risk_on", "risk_off", "rotation", "mixed", "insufficient_evidence"}
MAX_MARKET_CATALOG_CHARS = 18000
MARKET_SYNTHESIS_MAX_TOKENS = 1200
MARKET_REPAIR_MAX_TOKENS = 1000


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_market_synthesis_confidence(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Normalize only the model-owned confidence scale; never alter evidence."""

    normalized = dict(value)
    confidence = _number(value.get("confidence"))
    if confidence is None:
        return normalized, False
    if 1 < confidence <= 100:
        normalized["confidence"] = round(confidence / 100.0, 6)
        return normalized, True
    if value.get("confidence") != confidence:
        normalized["confidence"] = confidence
        return normalized, True
    return normalized, False


def strip_renderer_owned_numbers(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Keep model interpretation, but remove exact values owned by the renderer."""

    normalized = dict(value)
    changed = False

    def clean(text: Any) -> str:
        nonlocal changed
        original = str(text)
        cleaned = __import__("re").sub(
            r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|퍼센트|개|건|일|점)?",
            "",
            original,
        )
        cleaned = __import__("re").sub(r"\s+", " ", cleaned).strip()
        cleaned = __import__("re").sub(r"\s+([.,;:!?])", r"\1", cleaned)
        if cleaned != original:
            changed = True
        return cleaned

    normalized["summary"] = clean(value.get("summary") or "")
    for key in ("confirmations", "contradictions"):
        rows = []
        for row in value.get(key) or []:
            normalized_row = (
                {"evidence_id": row.get("evidence_id")}
                if isinstance(row, Mapping)
                else row
            )
            if isinstance(row, Mapping) and set(row) != {"evidence_id"}:
                changed = True
            rows.append(normalized_row)
        normalized[key] = rows
    normalized["unknowns"] = [
        clean(item) for item in value.get("unknowns") or []
    ]
    return normalized, changed


def market_guided_json_schema(
    catalog: Mapping[str, Any],
    *,
    minimum_confirmations: int,
    minimum_contradictions: int,
) -> dict[str, Any]:
    evidence_ids = sorted(catalog)
    etfs = sorted(
        key.removeprefix("etf.") for key in catalog if key.startswith("etf.")
    )
    stocks = sorted(
        key.removeprefix("stock.") for key in catalog if key.startswith("stock.")
    )
    evidence_row = {
        "type": "object",
        "properties": {
            "evidence_id": {"type": "string", "enum": evidence_ids},
        },
        "required": ["evidence_id"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "market_state": {"type": "string", "enum": sorted(MARKET_STATES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 100},
            "summary": {"type": "string"},
            "confirmations": {
                "type": "array",
                "items": evidence_row,
                "minItems": minimum_confirmations,
                "maxItems": 5,
            },
            "contradictions": {
                "type": "array",
                "items": evidence_row,
                "minItems": minimum_contradictions,
                "maxItems": 5,
            },
            "unknowns": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
            "leading_etfs": {
                "type": "array",
                "items": (
                    {"type": "string", "enum": etfs}
                    if etfs
                    else {"type": "string"}
                ),
                "minItems": 1 if etfs else 0,
                "maxItems": min(5, len(etfs)) if etfs else 0,
            },
            "affected_stocks": {
                "type": "array",
                "items": (
                    {"type": "string", "enum": stocks}
                    if stocks
                    else {"type": "string"}
                ),
                "minItems": 1 if stocks else 0,
                "maxItems": min(5, len(stocks)) if stocks else 0,
            },
            "scope": {
                "type": "string",
                "const": "market_and_security_analysis_not_trade_execution",
            },
        },
        "required": [
            "market_state",
            "confidence",
            "summary",
            "confirmations",
            "contradictions",
            "unknowns",
            "leading_etfs",
            "affected_stocks",
            "scope",
        ],
        "additionalProperties": False,
    }


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
            "all ETF-related securities were quantitatively scanned; this aggregate "
            "contains the dynamically selected material-evidence model scope"
        ),
        "etf_leaders": etfs[:25],
        "stock_leaders": stocks[:25],
    }


def oracle_market_summary(features: Mapping[str, Any]) -> dict[str, Any]:
    """Bound the sealed Oracle-derived evidence used for market synthesis."""
    return {
        "release_binding": dict(features["binding"]),
        "snapshot_sha256": features["snapshot_sha256"],
        "full_table_row_counts": {
            "oracle_etfs": len(features.get("etfs") or []),
            "oracle_stocks": len(features.get("stocks") or []),
            "oracle_rotation_clusters": len(
                features.get("integrated_rotation_clusters") or []
            ),
        },
        "master_eligibility_counts": dict(
            features.get("master_eligibility_counts") or {}
        ),
        "master_flow_status_counts": dict(
            features.get("master_flow_status_counts") or {}
        ),
        "massive_flow_fusion_top": list(features.get("etfs") or [])[:12],
        "early_accumulation_top": list(features.get("etfs") or [])[:12],
        "accumulation_clusters": list(
            features.get("accumulation_clusters") or []
        )[:15],
        "integrated_rotation_clusters": list(
            features.get("integrated_rotation_clusters") or []
        )[:15],
    }


def _pick(
    row: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    return {field: row.get(field) for field in fields if row.get(field) is not None}


def _compact_leader(row: Mapping[str, Any]) -> dict[str, Any]:
    value = _pick(
        row,
        (
            "symbol",
            "regime",
            "confidence",
            "price_signal",
            "etf_flow_signal",
            "latest_effective_date",
            "latest_flow_to_assets_pct",
            "latest_robust_zscore",
            "sum_last_20_visible_flows",
            "net_weighted_flow_rate_contribution_pct",
            "eligible_etf_count",
        ),
    )
    contributors = []
    for item in (row.get("top_contributing_etfs") or [])[:3]:
        contributors.append(
            _pick(
                item,
                (
                    "etf_ticker",
                    "weighted_flow_rate_contribution_pct",
                    "flow_training_available_session_date",
                ),
            )
        )
    if contributors:
        value["top_contributing_etfs"] = contributors
    return value


def _compact_cluster(
    row: Mapping[str, Any],
    *,
    integrated: bool,
) -> dict[str, Any]:
    common = (
        "rank",
        "cluster_role",
        "ticker_count",
        "top_ticker",
        "representative_tickers",
        "breadth_score",
        "quality_score",
    )
    if integrated:
        fields = (
            "integrated_cluster",
            "integrated_state",
            "integrated_score",
            "median_fmp_ret_1d",
            "median_fmp_ret_5d",
            "median_fmp_ret_21d",
            "median_nav_ret_5d",
            "median_nav_ret_21d",
        )
    else:
        fields = (
            "accum_cluster",
            "selection_state",
            "cluster_score",
            "flow_anomaly_score",
            "flow_5d_to_assets",
            "flow_21d_to_assets",
            "positive_flow_count",
            "confirmed_flow_count",
            "top_related_stocks",
        )
    return _pick(row, (*common, *fields))


def _evidence_catalog(
    aggregate: Mapping[str, Any],
    radar: Mapping[str, Any],
) -> dict[str, Any]:
    catalog: dict[str, Any] = {
        "aggregate.analyzed_security_count": aggregate["analyzed_security_count"],
        "aggregate.task_type_counts": aggregate["task_type_counts"],
        "aggregate.regime_counts": aggregate["regime_counts"],
        "aggregate.price_signal_counts": aggregate["price_signal_counts"],
        "aggregate.etf_flow_signal_counts": aggregate["etf_flow_signal_counts"],
        "aggregate.mean_model_confidence": aggregate["mean_model_confidence"],
        "oracle.release_id": radar["release_binding"]["release_id"],
        "oracle.trade_date_us": radar["release_binding"]["trade_date_us"],
        "oracle.master_eligibility_counts": radar["master_eligibility_counts"],
        "oracle.master_flow_status_counts": radar["master_flow_status_counts"],
    }
    for row in aggregate["etf_leaders"][:8]:
        catalog[f"etf.{row['symbol']}"] = _compact_leader(row)
    for row in aggregate["stock_leaders"][:8]:
        catalog[f"stock.{row['symbol']}"] = _compact_leader(row)
    for row in radar["accumulation_clusters"][:6]:
        key = str(row.get("accum_cluster") or row.get("rank"))
        catalog[f"oracle.accumulation_cluster.{key}"] = _compact_cluster(
            row,
            integrated=False,
        )
    for row in radar["integrated_rotation_clusters"][:6]:
        key = str(row.get("integrated_cluster") or row.get("rank"))
        catalog[f"oracle.rotation_cluster.{key}"] = _compact_cluster(
            row,
            integrated=True,
        )
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
    if not __import__("re").search(r"[가-힣]", str(value.get("summary") or "")):
        raise ResponseContractError("market synthesis summary must be Korean")
    natural_language = [str(value.get("summary") or "")]
    has_cluster_evidence = any(
        evidence_id.startswith(
            (
                "oracle.rotation_cluster.",
                "oracle.accumulation_cluster.",
            )
        )
        for evidence_id in catalog
    )
    has_security_evidence = any(
        evidence_id.startswith(("etf.", "stock.")) for evidence_id in catalog
    )
    minimum_rows = (
        {"confirmations": 3, "contradictions": 2}
        if has_cluster_evidence and has_security_evidence
        else {"confirmations": 1, "contradictions": 1}
    )
    cited_ids: list[str] = []
    for key in ("confirmations", "contradictions"):
        rows = value.get(key)
        if not isinstance(rows, list) or len(rows) < minimum_rows[key]:
            raise ResponseContractError(
                f"market synthesis {key} must contain at least "
                f"{minimum_rows[key]} grounded rows"
            )
        for row in rows:
            if not isinstance(row, dict) or row.get("evidence_id") not in catalog:
                raise ResponseContractError(f"market synthesis {key} cited unknown evidence")
            cited_ids.append(str(row["evidence_id"]))
    if len(cited_ids) != len(set(cited_ids)):
        raise ResponseContractError("market synthesis repeated an evidence citation")
    if has_cluster_evidence and not any(
        evidence_id.startswith(
            (
                "oracle.rotation_cluster.",
                "oracle.accumulation_cluster.",
            )
        )
        for evidence_id in cited_ids
    ):
        raise ResponseContractError("market synthesis omitted Oracle cluster evidence")
    if has_security_evidence and not any(
        evidence_id.startswith(("etf.", "stock.")) for evidence_id in cited_ids
    ):
        raise ResponseContractError("market synthesis omitted security-level evidence")
    if not isinstance(value.get("unknowns"), list):
        raise ResponseContractError("market synthesis unknowns must be an array")
    if any(
        not __import__("re").search(r"[가-힣]", str(item))
        for item in value.get("unknowns") or []
    ):
        raise ResponseContractError("market synthesis unknowns must be Korean")
    natural_language.extend(str(item) for item in value.get("unknowns") or [])
    if any(__import__("re").search(r"\d", text) for text in natural_language):
        raise ResponseContractError(
            "market synthesis natural language must not repeat numeric facts; "
            "the renderer owns exact values"
        )
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
    if (allowed_etfs and not value.get("leading_etfs")) or (
        allowed_stocks and not value.get("affected_stocks")
    ):
        raise ResponseContractError("market synthesis omitted ranked ETF or stock leaders")
    text = canonical_json(value)
    for raw_date in __import__("re").findall(r"\b20\d{2}-\d{2}-\d{2}\b", text):
        if date.fromisoformat(raw_date) > date.fromisoformat(as_of_date):
            raise ResponseContractError("market synthesis contains a post-as-of date")
    if TRADE_DIRECTIVE_PATTERN.search(text):
        raise ResponseContractError("market synthesis contains a trade directive")
    semantic_issues = market_semantic_issues(value, catalog)
    if semantic_issues:
        raise ResponseContractError(
            "market synthesis contradicts cited numeric evidence: "
            + ",".join(semantic_issues)
        )
    return dict(value)


def market_contract_repair_instruction(
    *,
    contract_error: str,
    catalog: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> str:
    allowed_evidence_ids = sorted(catalog)
    if not allowed_evidence_ids:
        raise ResponseContractError(
            "market synthesis repair has no allowed evidence"
        )
    confirmation_id = (
        "aggregate.regime_counts"
        if "aggregate.regime_counts" in catalog
        else allowed_evidence_ids[0]
    )
    contradiction_id = (
        "aggregate.price_signal_counts"
        if "aggregate.price_signal_counts" in catalog
        else next(
            (
                evidence_id
                for evidence_id in allowed_evidence_ids
                if evidence_id != confirmation_id
            ),
            confirmation_id,
        )
    )
    allowed_etfs = sorted(
        key.removeprefix("etf.")
        for key in catalog
        if key.startswith("etf.")
    )
    allowed_stocks = sorted(
        key.removeprefix("stock.")
        for key in catalog
        if key.startswith("stock.")
    )
    rotation_ids = [
        evidence_id
        for evidence_id in allowed_evidence_ids
        if evidence_id.startswith("oracle.rotation_cluster.")
    ]
    security_ids = [
        evidence_id
        for evidence_id in allowed_evidence_ids
        if evidence_id.startswith(("etf.", "stock."))
    ]
    mandatory_confirmations = [
        confirmation_id,
        *(rotation_ids[:1]),
        *([security_ids[0]] if security_ids else []),
    ]
    mandatory_confirmations = list(dict.fromkeys(mandatory_confirmations))
    remaining_ids = [
        evidence_id
        for evidence_id in allowed_evidence_ids
        if evidence_id not in mandatory_confirmations
    ]
    mandatory_contradictions = []
    if contradiction_id not in mandatory_confirmations:
        mandatory_contradictions.append(contradiction_id)
    if security_ids:
        candidate = security_ids[-1]
        if (
            candidate not in mandatory_confirmations
            and candidate not in mandatory_contradictions
        ):
            mandatory_contradictions.append(candidate)
    for evidence_id in remaining_ids:
        if len(mandatory_contradictions) >= 2:
            break
        if evidence_id not in mandatory_contradictions:
            mandatory_contradictions.append(evidence_id)
    return (
        "이전 시장 synthesis는 계약 위반이다. 새로운 사실이나 evidence ID를 "
        "만들지 말고 같은 입력만 다시 해석하라. confirmations와 "
        "contradictions의 evidence_id는 ALLOWED_EVIDENCE_IDS_JSON에서만 "
        "고르고 evidence ID를 중복하지 마라. confirmations는 최소 3개, "
        "contradictions는 최소 2개를 만들며, MANDATORY_EVIDENCE_IDS_JSON의 "
        "ID를 정확히 포함하라. 각 행에는 evidence_id만 쓰고 해석문이나 숫자를 "
        "넣지 마라. 숫자·퍼센트·날짜와 근거 설명은 renderer가 담당하므로 "
        "summary·unknowns 자연어에는 숫자를 하나도 쓰지 마라. "
        "leading_etfs와 "
        "affected_stocks도 각 허용 목록 안에서만 고르라. 매매 지시와 입력 "
        "시점 이후 날짜를 포함하지 말고, summary·unknowns의 "
        "모든 자연어는 한국어로 작성하며 JSON 객체만 출력하라.\n"
        f"CONTRACT_ERROR={contract_error}\n"
        f"ALLOWED_EVIDENCE_IDS_JSON={canonical_json(allowed_evidence_ids)}\n"
        "MANDATORY_EVIDENCE_IDS_JSON="
        f"{canonical_json({'confirmations': mandatory_confirmations, 'contradictions': mandatory_contradictions})}\n"
        f"ALLOWED_LEADING_ETFS_JSON={canonical_json(allowed_etfs)}\n"
        f"ALLOWED_AFFECTED_STOCKS_JSON={canonical_json(allowed_stocks)}\n"
        f"REQUIRED_SCHEMA_JSON={canonical_json(schema)}\n"
        "/no_think"
    )


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
        "summary": "Korean breadth, flow, rotation, and divergence interpretation without uncited numbers",
        "confirmations": [
            {
                "evidence_id": "unique catalog id; at least 3 rows total",
            }
        ],
        "contradictions": [
            {
                "evidence_id": "unique catalog id; at least 2 rows total",
            }
        ],
        "unknowns": ["Korean limitation"],
        "leading_etfs": ["symbols drawn only from etf.* catalog entries"],
        "affected_stocks": ["symbols drawn only from stock.* catalog entries"],
        "scope": "market_and_security_analysis_not_trade_execution",
    }
    system = (
        "You are the released quant-analysis LoRA. Interpret only the supplied as-of "
        "catalog produced from the full quantitative scan, dynamically selected "
        "material-evidence inference scope, and the "
        "sealed Oracle market feature snapshot. Separate confirmation, contradiction, and "
        "unknowns. Cover aggregate breadth, one Oracle rotation cluster, one "
        "ETF leader, and one affected stock. Cite at least three confirmations and "
        "two contradictions with unique catalog IDs. Every number and comparison "
        "is rendered from the cited catalog by the program. Return only evidence_id "
        "inside confirmation and contradiction rows. Do not write digits, percentages, "
        "or dates in summary or unknowns. Never issue a trade "
        "instruction or invent a number. /no_think"
    )
    catalog_json = canonical_json(catalog)
    if len(catalog_json) > MAX_MARKET_CATALOG_CHARS:
        raise ResponseContractError(
            "market evidence catalog exceeds the bounded synthesis contract: "
            f"{len(catalog_json)}>{MAX_MARKET_CATALOG_CHARS}"
        )
    user = (
        f"AS_OF_DATE={as_of_date}\nEVIDENCE_CATALOG={catalog_json}\n"
        f"Return only one JSON object with this schema: {canonical_json(schema)}"
    )
    has_cluster_evidence = any(
        key.startswith(
            ("oracle.rotation_cluster.", "oracle.accumulation_cluster.")
        )
        for key in catalog
    )
    has_security_evidence = any(
        key.startswith(("etf.", "stock.")) for key in catalog
    )
    guided_schema = market_guided_json_schema(
        catalog,
        minimum_confirmations=(
            3 if has_cluster_evidence and has_security_evidence else 1
        ),
        minimum_contradictions=(
            2 if has_cluster_evidence and has_security_evidence else 1
        ),
    )
    program_normalizations: list[str] = []
    try:
        response, trace = client.complete(
            system=system,
            user=user,
            max_tokens=MARKET_SYNTHESIS_MAX_TOKENS,
            response_schema=guided_schema,
        )
        response, stripped = strip_renderer_owned_numbers(response)
        if stripped:
            program_normalizations.append("initial_renderer_number_strip")
        response, normalized = normalize_market_synthesis_confidence(response)
        if normalized:
            program_normalizations.append("initial_confidence_scale")
        validated = validate_market_synthesis(
            response,
            as_of_date=as_of_date,
            catalog=catalog,
        )
    except (ModelResponseParseError, ResponseContractError) as exc:
        if isinstance(exc, ModelResponseParseError):
            trace = exc.trace
        repair = market_contract_repair_instruction(
            contract_error=str(exc),
            catalog=catalog,
            schema=schema,
        )
        repaired, repaired_trace = client.complete_messages(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "user", "content": repair},
            ],
            max_tokens=MARKET_REPAIR_MAX_TOKENS,
            response_schema=guided_schema,
        )
        repaired, stripped = strip_renderer_owned_numbers(repaired)
        if stripped:
            program_normalizations.append("repair_renderer_number_strip")
        repaired, normalized = normalize_market_synthesis_confidence(repaired)
        if normalized:
            program_normalizations.append("repair_confidence_scale")
        try:
            validated = validate_market_synthesis(
                repaired,
                as_of_date=as_of_date,
                catalog=catalog,
            )
            final_trace = repaired_trace
            attempts = 2
            second_error = None
        except (ModelResponseParseError, ResponseContractError) as second_exc:
            if isinstance(second_exc, ModelResponseParseError):
                repaired_trace = second_exc.trace
            final_repair = market_contract_repair_instruction(
                contract_error=str(second_exc),
                catalog=catalog,
                schema=schema,
            )
            final, final_trace = client.complete_messages(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "user", "content": final_repair},
                ],
                max_tokens=MARKET_REPAIR_MAX_TOKENS,
                response_schema=guided_schema,
            )
            final, stripped = strip_renderer_owned_numbers(final)
            if stripped:
                program_normalizations.append("final_renderer_number_strip")
            final, normalized = normalize_market_synthesis_confidence(final)
            if normalized:
                program_normalizations.append("final_confidence_scale")
            validated = validate_market_synthesis(
                final,
                as_of_date=as_of_date,
                catalog=catalog,
            )
            attempts = 3
            second_error = f"{type(second_exc).__name__}: {second_exc}"
        result_trace = {
            **final_trace,
            "contract_attempts": attempts,
            "contract_repair_applied": True,
            "initial_request_sha256": trace["request_sha256"],
            "initial_response_sha256": trace["response_sha256"],
            "initial_contract_error": f"{type(exc).__name__}: {exc}",
            "program_normalizations": program_normalizations,
        }
        if second_error:
            result_trace.update(
                {
                    "second_request_sha256": repaired_trace["request_sha256"],
                    "second_response_sha256": repaired_trace["response_sha256"],
                    "second_contract_error": second_error,
                }
            )
        return validated, result_trace, catalog
    return validated, {
        **trace,
        "contract_attempts": 1,
        "contract_repair_applied": False,
        "program_normalizations": program_normalizations,
    }, catalog
