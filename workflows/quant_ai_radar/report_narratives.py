"""Bounded multi-call Qwen explanations for the deterministic Radar report."""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_support import build_market_dashboard, build_security_brief
from .model_runtime import (
    ModelResponseParseError,
    ResponseContractError,
    TRADE_DIRECTIVE_PATTERN,
    TrainedQuantClient,
    canonical_json,
)
from .presentation import label_rotation_state


NARRATIVE_SCHEMA_VERSION = "quant.ai_radar_multistage_narratives.v1"
NARRATIVE_CONTRACT_VERSION = "quant.ai_radar_narrative_contract.v4"
EDITORIAL_CONTRACT_VERSION = "quant.ai_radar_editorial_contract.v2"
SECTOR_BATCH_SIZE = 3
SECURITY_BATCH_SIZE = 3
SECURITY_PER_LANE = 4
PROSE_FIELDS = {
    "headline",
    "explanation",
    "counterpoint",
    "stock_context",
    "group_context",
    "etf_transmission",
    "watch_condition",
    "executive_summary",
    "rotation_summary",
    "selection_summary",
    "risk_summary",
}
HANGUL = re.compile(r"[가-힣]")
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|퍼센트|개|건|일|점)?")
RAW_CODE = re.compile(
    r"\b(?:rotation_in|rotation_out|price_flow_[a-z_]+|mixed_or_flat|"
    r"insufficient_joint_evidence)\b"
)
CAUSAL_OVERCLAIM = re.compile(
    r"(?:ETF|자금|섹터|회전)[^.]{0,80}(?:가격|수익률)[^.]{0,50}"
    r"(?:영향을\s*미치|유발하|원인이\s*(?:되|다))|"
    r"(?:영향을\s*미치|유발하|원인이\s*(?:되|다))[^.]{0,50}"
    r"(?:가격|수익률)"
)
KOREAN_SENTENCE_ENDING = re.compile(
    r"(?:습니다|입니다|있다|없다|한다|된다|이다|필요하다|보인다|"
    r"나타난다|시사한다|반영한다|요구된다)$"
)
CLUSTER_TERMS = (
    "Healthcare",
    "Industrials",
    "Financial Services",
    "Technology",
    "Unclassified",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Utilities",
    "건강케어",
    "헬스케어",
    "산업 섹터",
    "금융",
    "기술 섹터",
    "비분류",
    "경기소비재",
    "방어소비재",
    "소비재 섹터",
    "에너지 섹터",
    "유틸리티",
    "전력 섹터",
)
CLUSTER_LABELS = {
    "Healthcare": ("Healthcare", "건강케어", "헬스케어"),
    "Industrials": ("Industrials", "산업"),
    "Financial Services": ("Financial Services", "금융"),
    "Technology": ("Technology", "기술"),
    "Unclassified": ("Unclassified", "비분류"),
    "Consumer Cyclical": ("Consumer Cyclical", "경기소비재"),
    "Consumer Defensive": ("Consumer Defensive", "방어소비재"),
    "Energy": ("Energy", "에너지"),
    "Utilities": ("Utilities", "유틸리티", "전력"),
}
NO_DIRECT_CLUSTER_LINK_TERMS = (
    "제공된 직접 섹터 연결 근거 없음",
    "직접 섹터 연결 근거가 없다",
    "직접 섹터 연결 근거가 없습니다",
    "직접 섹터 연결 근거가 제공되지",
    "직접적인 섹터 연결 근거가 없다",
    "직접적인 섹터 연결 근거가 없습니다",
    "직접적인 섹터 연결 근거가 제공되지",
)


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _clean_prose(value: Any) -> tuple[str, bool]:
    original = str(value or "").strip()
    cleaned = NUMBER.sub("", original)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    if cleaned and not _is_complete_sentence(cleaned):
        cleaned = f"{cleaned}."
    return cleaned, cleaned != original


def _normalize_item(item: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized = dict(item)
    changed = False
    for key in PROSE_FIELDS:
        if key in normalized:
            normalized[key], stripped = _clean_prose(normalized[key])
            changed = changed or stripped
    return normalized, changed


def _has_no_direct_cluster_link(value: Any) -> bool:
    text = str(value or "")
    return any(term in text for term in NO_DIRECT_CLUSTER_LINK_TERMS)


def _is_complete_sentence(value: Any) -> bool:
    text = str(value or "").strip()
    return text.endswith((".", "!", "?")) or bool(
        KOREAN_SENTENCE_ENDING.search(text)
    )


def _validate_items(
    response: Mapping[str, Any],
    *,
    id_key: str,
    expected_ids: Sequence[str],
    text_fields: Sequence[str],
    required_mentions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    required_all_mentions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    forbidden_mentions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> list[dict[str, Any]]:
    rows = response.get("items")
    if not isinstance(rows, list):
        raise ResponseContractError("narrative response items must be a list")
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ResponseContractError("narrative item must be an object")
        normalized, _ = _normalize_item(row)
        identifier = str(normalized.get(id_key) or "")
        if not identifier:
            raise ResponseContractError(f"narrative item is missing {id_key}")
        for field in text_fields:
            text = str(normalized.get(field) or "").strip()
            minimum_hangul = (
                0
                if field == "stock_context"
                else 3
                if field == "headline"
                else 6
            )
            if len(HANGUL.findall(text)) < minimum_hangul:
                raise ResponseContractError(
                    f"{identifier} narrative {field} is not substantive Korean"
                )
            if TRADE_DIRECTIVE_PATTERN.search(text):
                raise ResponseContractError(
                    f"{identifier} narrative contains a trade directive"
                )
            if RAW_CODE.search(text):
                raise ResponseContractError(
                    f"{identifier} narrative contains a raw state code"
                )
            if CAUSAL_OVERCLAIM.search(text):
                raise ResponseContractError(
                    f"{identifier} narrative {field} overstates causality"
                )
            if field not in ("headline", "stock_context") and not (
                _is_complete_sentence(text)
            ):
                raise ResponseContractError(
                    f"{identifier} narrative {field} is not a complete sentence"
                )
        for field, allowed in (required_mentions or {}).get(
            identifier, {}
        ).items():
            text = str(normalized.get(field) or "")
            if allowed and not any(token in text for token in allowed):
                raise ResponseContractError(
                    f"{identifier} narrative {field} must name one of {list(allowed)}"
                )
        for field, required in (required_all_mentions or {}).get(
            identifier, {}
        ).items():
            text = str(normalized.get(field) or "")
            missing = [
                token
                for token in required
                if token not in text
                and not (
                    field == "group_context"
                    and token == identifier
                    and token in str(normalized.get("headline") or "")
                )
                and not (
                    field == "group_context"
                    and token == "제공된 직접 섹터 연결 근거 없음"
                    and _has_no_direct_cluster_link(text)
                )
                and not (
                    field == "group_context"
                    and token == "제공된 직접 섹터 연결 근거 없음"
                    and not any(
                        forbidden in text
                        for forbidden in (forbidden_mentions or {})
                        .get(identifier, {})
                        .get(field, ())
                    )
                )
            ]
            if missing:
                raise ResponseContractError(
                    f"{identifier} narrative {field} is missing {missing}"
                )
        for field, forbidden in (forbidden_mentions or {}).get(
            identifier, {}
        ).items():
            text = str(normalized.get(field) or "")
            found = [token for token in forbidden if token in text]
            if found:
                raise ResponseContractError(
                    f"{identifier} narrative {field} inferred unsupported group terms {found}"
                )
        normalized_rows.append(normalized)
    actual = [str(row.get(id_key) or "") for row in normalized_rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected_ids):
        raise ResponseContractError(
            f"narrative coverage mismatch: expected={list(expected_ids)} actual={actual}"
        )
    by_id = {str(row[id_key]): row for row in normalized_rows}
    return [by_id[identifier] for identifier in expected_ids]


def _items_schema(
    *,
    id_key: str,
    expected_ids: Sequence[str],
    text_fields: Sequence[str],
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        id_key: {"type": "string", "enum": list(expected_ids)}
    }
    for field in text_fields:
        if field == "stock_context":
            limits = (1, 140)
        elif field == "headline":
            limits = (24, 200)
        else:
            limits = (40, 440)
        properties[field] = {
            "type": "string",
            "minLength": limits[0],
            "maxLength": limits[1],
        }
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": len(expected_ids),
                "maxItems": len(expected_ids),
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": [id_key, *text_fields],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def _call_items(
    *,
    client: TrainedQuantClient,
    stage: str,
    id_key: str,
    expected_ids: Sequence[str],
    text_fields: Sequence[str],
    global_context: Mapping[str, Any],
    batch_context: Sequence[Mapping[str, Any]],
    required_mentions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    required_all_mentions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    forbidden_mentions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schema = _items_schema(
        id_key=id_key,
        expected_ids=expected_ids,
        text_fields=text_fields,
    )
    system = (
        "You are the released Quant AI Radar LoRA explanation layer. Read the complete "
        "deterministic market context first, then explain each requested subset in that "
        "context. Use only supplied point-in-time facts. Connect market breadth, sector "
        "rotation, ETF capital transmission, the item's own price/flow relationship, "
        "counter-evidence, and a non-trading confirmation condition. Write concrete Korean "
        "report prose. Do not write digits, percentages, dates, raw code labels, instructions, "
        "or buy/sell orders; the renderer owns every exact number. Preserve the requested "
        "item order. End every prose field except stock_context as a complete sentence. "
        "Never infer a security's sector when sector_memberships is empty; use the exact "
        "phrase '제공된 직접 섹터 연결 근거 없음'. Describe price and ETF flow as "
        "aligned, divergent, or associated; never claim that one causes or affects the "
        "other. Return JSON only. /no_think"
    )
    user = (
        f"STAGE={stage}\n"
        f"EXPECTED_IDS={canonical_json(list(expected_ids))}\n"
        f"COMPLETE_MARKET_CONTEXT={canonical_json(global_context)}\n"
        f"REQUESTED_SUBSET={canonical_json(list(batch_context))}\n"
        f"REQUIRED_MENTIONS={canonical_json(required_mentions or {})}\n"
        f"REQUIRED_ALL_MENTIONS={canonical_json(required_all_mentions or {})}\n"
        f"FORBIDDEN_MENTIONS={canonical_json(forbidden_mentions or {})}"
    )
    traces: list[dict[str, Any]] = []
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    contract_error = ""
    for attempt in range(1, 4):
        messages = list(base_messages)
        if contract_error:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"CONTRACT_ERROR={contract_error}\n"
                        "Return every expected item exactly once and in the exact "
                        "requested order. Every prose field must be a complete Korean "
                        "report sentence with no digits, raw code labels, or trade "
                        "directives. End every prose field except stock_context as a "
                        "complete sentence. Obey REQUIRED_MENTIONS and "
                        "REQUIRED_ALL_MENTIONS exactly and do not use any "
                        "FORBIDDEN_MENTIONS. Describe association without causal claims."
                    ),
                }
            )
        try:
            response, trace = client.complete_messages(
                messages=messages,
                max_tokens=2600,
                response_schema=schema,
            )
            traces.append(trace)
            items = _validate_items(
                response,
                id_key=id_key,
                expected_ids=expected_ids,
                text_fields=text_fields,
                required_mentions=required_mentions,
                required_all_mentions=required_all_mentions,
                forbidden_mentions=forbidden_mentions,
            )
            break
        except (ModelResponseParseError, ResponseContractError) as exc:
            if isinstance(exc, ModelResponseParseError) and exc.trace:
                traces.append(exc.trace)
            contract_error = f"{type(exc).__name__}: {exc}"
            if attempt == 3:
                raise
    else:
        raise ResponseContractError(f"{stage} exhausted narrative attempts")
    return items, {
        "stage": stage,
        "contract_attempts": len(traces),
        "contract_repair_applied": len(traces) > 1,
        "calls": traces,
    }


def _cached_call_items(
    *,
    checkpoint_dir: Path | None,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if checkpoint_dir is None:
        return _call_items(**kwargs)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage = str(kwargs["stage"])
    fingerprint_value = {
        key: kwargs.get(key)
        for key in (
            "stage",
            "id_key",
            "expected_ids",
            "text_fields",
            "global_context",
            "batch_context",
            "required_mentions",
            "required_all_mentions",
            "forbidden_mentions",
        )
    }
    fingerprint_value["contract_version"] = NARRATIVE_CONTRACT_VERSION
    input_sha = hashlib.sha256(
        canonical_json(fingerprint_value).encode("utf-8")
    ).hexdigest()
    path = checkpoint_dir / f"{stage}.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_sha:
            items = _validate_items(
                {"items": cached.get("items")},
                id_key=str(kwargs["id_key"]),
                expected_ids=kwargs["expected_ids"],
                text_fields=kwargs["text_fields"],
                required_mentions=kwargs.get("required_mentions"),
                required_all_mentions=kwargs.get("required_all_mentions"),
                forbidden_mentions=kwargs.get("forbidden_mentions"),
            )
            trace = dict(cached.get("trace") or {})
            trace["cache_hit"] = True
            return items, trace
    items, trace = _call_items(**kwargs)
    value = {
        "schema_version": "quant.ai_radar_narrative_stage.v1",
        "input_sha256": input_sha,
        "items": items,
        "trace": trace,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return items, trace


def _parse_related_stock(value: Any) -> str:
    return str(value or "").split(":", 1)[0].strip()


def _global_context(
    aggregate: Mapping[str, Any],
    radar: Mapping[str, Any],
    dashboard: Mapping[str, Any],
    market_judgement: Mapping[str, Any],
) -> dict[str, Any]:
    lanes = dashboard.get("candidate_lanes") or {}
    return {
        "as_of_scope": "point_in_time_close_aware",
        "analyzed_security_count": aggregate.get("analyzed_security_count"),
        "price_signal_counts": aggregate.get("price_signal_counts") or {},
        "etf_flow_signal_counts": aggregate.get("etf_flow_signal_counts") or {},
        "regime_counts": aggregate.get("regime_counts") or {},
        "breadth": dashboard.get("breadth") or {},
        "market_judgement": {
            key: market_judgement.get(key)
            for key in (
                "market_state",
                "confidence",
                "summary",
                "unknowns",
                "leading_etfs",
                "affected_stocks",
            )
        },
        "rotation_clusters": [
            {
                key: row.get(key)
                for key in (
                    "rank",
                    "integrated_cluster",
                    "integrated_state",
                    "breadth_score",
                    "flow_5d_to_assets",
                    "flow_21d_to_assets",
                    "median_fmp_ret_5d",
                    "median_fmp_ret_21d",
                    "representative_tickers",
                    "top_related_stocks",
                )
            }
            for row in radar.get("integrated_rotation_clusters") or []
        ],
        "candidate_lane_symbols": {
            key: [row.get("symbol") for row in value]
            for key, value in lanes.items()
            if isinstance(value, list)
        },
        "leading_etfs": [
            {
                key: row.get(key)
                for key in (
                    "symbol",
                    "regime",
                    "confidence",
                    "latest_robust_zscore",
                    "latest_effective_date",
                )
            }
            for row in dashboard.get("leading_etfs") or []
        ],
        "affected_stocks": [
            {
                key: row.get(key)
                for key in (
                    "symbol",
                    "regime",
                    "confidence",
                    "net_weighted_flow_rate_contribution_pct",
                    "eligible_etf_count",
                )
            }
            for row in dashboard.get("affected_stocks") or []
        ],
    }


def _material_symbols(dashboard: Mapping[str, Any]) -> list[str]:
    lanes = dashboard.get("candidate_lanes") or {}
    ordered: list[str] = []
    for key in (
        "positive_confirmation_stocks",
        "negative_confirmation_stocks",
        "divergence_stocks",
    ):
        for row in list(lanes.get(key) or [])[:SECURITY_PER_LANE]:
            symbol = str(row.get("symbol") or "")
            if symbol and symbol not in ordered:
                ordered.append(symbol)
    return ordered


def _security_contexts(
    symbols: Sequence[str],
    results: Sequence[Mapping[str, Any]],
    radar: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_symbol = {str(row.get("symbol") or ""): row for row in results}
    memberships: dict[str, list[dict[str, Any]]] = {}
    for cluster in radar.get("integrated_rotation_clusters") or []:
        for raw in cluster.get("top_related_stocks") or []:
            symbol = _parse_related_stock(raw)
            if symbol:
                memberships.setdefault(symbol, []).append(
                    {
                        "cluster": cluster.get("integrated_cluster"),
                        "cluster_state": cluster.get("integrated_state"),
                        "cluster_breadth_score": cluster.get("breadth_score"),
                        "cluster_return_5d_pct": cluster.get("median_fmp_ret_5d"),
                        "relation": raw,
                    }
                )
    contexts = []
    for symbol in symbols:
        result = by_symbol.get(symbol)
        if result is None:
            raise ResponseContractError(
                f"material narrative symbol has no completed judgement: {symbol}"
            )
        judgement = result.get("judgement") or {}
        brief = build_security_brief(judgement)
        contexts.append(
            {
                "symbol": symbol,
                "task_type": result.get("task_type"),
                "regime": judgement.get("regime"),
                "confidence": judgement.get("confidence"),
                "brief": {
                    "price": {
                        key: brief.get("price", {}).get(key)
                        for key in (
                            "return_1_session_pct",
                            "return_5_session_pct",
                            "return_20_session_pct",
                            "annualized_realized_volatility_pct",
                            "max_drawdown_in_packet_pct",
                        )
                    },
                    "flow": {
                        key: brief.get("flow", {}).get(key)
                        for key in (
                            "mode",
                            "latest_effective_date",
                            "latest_robust_zscore",
                            "net_weighted_flow_rate_contribution_pct",
                            "eligible_etf_count",
                            "positive_etf_count",
                            "negative_etf_count",
                        )
                    },
                    "top_contributing_etfs": list(
                        brief.get("flow", {}).get("top_contributing_etfs") or []
                    )[:3],
                    "relationship": brief.get("relationship") or {},
                    "data_quality": brief.get("data_quality") or {},
                    "unknowns": brief.get("unknowns") or [],
                },
                "sector_memberships": memberships.get(symbol, []),
            }
        )
    return contexts


def _editorial_schema() -> dict[str, Any]:
    fields = (
        "headline",
        "executive_summary",
        "rotation_summary",
        "selection_summary",
        "risk_summary",
    )
    properties = {
        field: {
            "type": "string",
            "minLength": 24,
            "maxLength": (
                220
                if field == "headline"
                else 520
                if field == "selection_summary"
                else 700
            ),
        }
        for field in fields
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(fields),
        "additionalProperties": False,
    }


def _validate_editorial(
    response: Mapping[str, Any],
    security_items: Sequence[Mapping[str, Any]],
    unsupported_symbols: Sequence[str],
) -> tuple[dict[str, Any], bool]:
    normalized, stripped = _normalize_item(response)
    unsupported_symbols = [str(symbol) for symbol in unsupported_symbols]
    supported_symbols = [
        str(row.get("symbol") or "")
        for row in security_items
        if str(row.get("symbol") or "") not in unsupported_symbols
    ]
    for field in _editorial_schema()["required"]:
        text = str(normalized.get(field) or "")
        if len(HANGUL.findall(text)) < 10:
            raise ResponseContractError(
                f"editorial {field} is not substantive Korean"
            )
        if TRADE_DIRECTIVE_PATTERN.search(text):
            raise ResponseContractError(
                f"editorial {field} contains a trade directive"
            )
        if RAW_CODE.search(text):
            raise ResponseContractError(
                f"editorial {field} contains a raw state code"
            )
        if CAUSAL_OVERCLAIM.search(text):
            raise ResponseContractError(
                f"editorial {field} overstates causality"
            )
        if field != "headline" and not _is_complete_sentence(text):
            raise ResponseContractError(
                f"editorial {field} is not a complete sentence"
            )
        mentioned_unsupported = [
            symbol for symbol in unsupported_symbols if symbol in text
        ]
        if mentioned_unsupported:
            raise ResponseContractError(
                "editorial mentioned securities reserved for ungrouped cards: "
                f"symbols={mentioned_unsupported}"
            )
    selection = str(normalized.get("selection_summary") or "")
    named_supported = [
        symbol for symbol in supported_symbols if symbol in selection
    ]
    minimum_named = min(3, len(supported_symbols))
    if len(named_supported) < minimum_named:
        raise ResponseContractError(
            "editorial selection_summary lacks supported material securities: "
            f"required={minimum_named} actual={named_supported}"
        )
    selection_sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", selection)
        if sentence.strip()
    ]
    if not 2 <= len(selection_sentences) <= 9:
        raise ResponseContractError(
            "editorial selection_summary must contain two to nine concise sentences: "
            f"actual={len(selection_sentences)}"
        )
    if len(selection_sentences) != len(set(selection_sentences)):
        raise ResponseContractError(
            "editorial selection_summary repeats a sentence"
        )
    return normalized, stripped


def _editorial_call(
    *,
    client: TrainedQuantClient,
    global_context: Mapping[str, Any],
    sector_items: Sequence[Mapping[str, Any]],
    security_items: Sequence[Mapping[str, Any]],
    unsupported_symbols: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    unsupported_symbols = [str(symbol) for symbol in unsupported_symbols]
    supported_symbols = [
        str(row.get("symbol") or "")
        for row in security_items
        if str(row.get("symbol") or "") not in unsupported_symbols
    ]
    system = (
        "You are the final Korean editor of Quant AI Radar. Reconcile the complete "
        "deterministic market context with every sector and material-security explanation. "
        "Explain the dominant structure, where rotation is broad or narrow, why the selected "
        "stocks matter inside their groups, and the strongest counter-evidence. Use only the "
        "supplied facts. Write polished report prose without digits, dates, raw code labels, "
        "meta instructions, or trade directives. Securities without supplied direct sector "
        "links are explained in separate cards and must be omitted entirely from every "
        f"editorial field. OMIT_SECURITY_SYMBOLS={canonical_json(unsupported_symbols)}. "
        "The selection_summary must name at least three securities from "
        f"SUPPORTED_GROUPED_SYMBOLS={canonical_json(supported_symbols)}. Describe association "
        "without saying that ETF flow or sector rotation causes or affects price. Write "
        "selection_summary as three to nine short, non-repeating sentences, each ending "
        "with a period and covering selected names, their evidence relationship, and "
        "counter-evidence. "
        "End every field as a complete sentence. Return JSON only. /no_think"
    )
    user = (
        f"COMPLETE_MARKET_CONTEXT={canonical_json(global_context)}\n"
        f"ALL_SECTOR_EXPLANATIONS={canonical_json(list(sector_items))}\n"
        f"ALL_SECURITY_EXPLANATIONS={canonical_json(list(security_items))}\n"
        "UNSUPPORTED_SECTOR_ASSIGNMENT_SYMBOLS="
        f"{canonical_json(unsupported_symbols)}\n"
        f"SUPPORTED_GROUPED_SYMBOLS={canonical_json(supported_symbols)}"
    )
    traces: list[dict[str, Any]] = []
    contract_error = ""
    for attempt in range(1, 4):
        repair = (
            ""
            if not contract_error
            else (
                f"\nCONTRACT_ERROR={contract_error}\n"
                "Rewrite every required field as substantive Korean report prose. "
                "Do not omit fields, use digits, expose raw codes, or give trade directives. "
                "Do not mention any UNSUPPORTED_SECTOR_ASSIGNMENT_SYMBOLS ticker anywhere "
                "in the editorial; its separate security card owns that explanation. "
                "Name at least three SUPPORTED_GROUPED_SYMBOLS in selection_summary and "
                "describe association without causal language. Rewrite selection_summary "
                "as three to nine short, non-repeating sentences ending with periods."
            )
        )
        try:
            response, trace = client.complete(
                system=system,
                user=user + repair,
                max_tokens=2400,
                response_schema=_editorial_schema(),
            )
            traces.append(trace)
            normalized, stripped = _validate_editorial(
                response, security_items, unsupported_symbols
            )
            return normalized, {
                "stage": "editorial",
                "contract_attempts": len(traces),
                "contract_repair_applied": len(traces) > 1,
                "program_number_strip_applied": stripped,
                "calls": traces,
            }
        except (ModelResponseParseError, ResponseContractError) as exc:
            if isinstance(exc, ModelResponseParseError) and exc.trace:
                traces.append(exc.trace)
            contract_error = f"{type(exc).__name__}: {exc}"
            if attempt == 3:
                raise
    raise ResponseContractError("editorial exhausted narrative attempts")


def _cached_editorial_call(
    *,
    checkpoint_dir: Path | None,
    client: TrainedQuantClient,
    global_context: Mapping[str, Any],
    sector_items: Sequence[Mapping[str, Any]],
    security_items: Sequence[Mapping[str, Any]],
    unsupported_symbols: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if checkpoint_dir is None:
        return _editorial_call(
            client=client,
            global_context=global_context,
            sector_items=sector_items,
            security_items=security_items,
            unsupported_symbols=unsupported_symbols,
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "contract_version": EDITORIAL_CONTRACT_VERSION,
        "stage": "editorial",
        "global_context": global_context,
        "sector_items": list(sector_items),
        "security_items": list(security_items),
        "unsupported_symbols": list(unsupported_symbols),
    }
    input_sha = hashlib.sha256(
        canonical_json(fingerprint).encode("utf-8")
    ).hexdigest()
    path = checkpoint_dir / "editorial.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_sha:
            editorial = cached.get("editorial")
            if not isinstance(editorial, Mapping):
                raise ResponseContractError("cached editorial is not an object")
            normalized, _ = _validate_editorial(
                editorial, security_items, unsupported_symbols
            )
            trace = dict(cached.get("trace") or {})
            trace["cache_hit"] = True
            return normalized, trace
    editorial, trace = _editorial_call(
        client=client,
        global_context=global_context,
        sector_items=sector_items,
        security_items=security_items,
        unsupported_symbols=unsupported_symbols,
    )
    payload = {
        "schema_version": "quant.ai_radar_narrative_stage.v1",
        "input_sha256": input_sha,
        "editorial": editorial,
        "trace": trace,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return editorial, trace


def validate_report_narratives(report: Mapping[str, Any]) -> None:
    """Fail closed when a v2 report lacks trained-model explanation coverage."""

    if report.get("schema_version") != "quant.ai_radar_report.v2":
        return
    narratives = report.get("multistage_narratives")
    if not isinstance(narratives, Mapping):
        raise ResponseContractError("v2 report requires multistage_narratives")
    if narratives.get("schema_version") != NARRATIVE_SCHEMA_VERSION:
        raise ResponseContractError("v2 report has an unaccepted narrative schema")
    editorial = narratives.get("editorial")
    if not isinstance(editorial, Mapping):
        raise ResponseContractError("v2 report requires trained-model editorial")
    for field in _editorial_schema()["required"]:
        if not str(editorial.get(field) or "").strip():
            raise ResponseContractError(f"v2 editorial is missing {field}")

    dashboard = report.get("market_dashboard")
    dashboard = dashboard if isinstance(dashboard, Mapping) else {}
    expected_clusters = [
        str(row.get("cluster") or "")
        for row in dashboard.get("rotation_clusters") or []
        if isinstance(row, Mapping) and row.get("cluster")
    ]
    sector_rows = narratives.get("sector_explanations")
    if not isinstance(sector_rows, list):
        raise ResponseContractError("v2 report requires sector explanations")
    actual_clusters = [
        str(row.get("cluster") or "")
        for row in sector_rows
        if isinstance(row, Mapping)
    ]
    if actual_clusters != expected_clusters:
        raise ResponseContractError(
            "v2 sector narrative coverage does not match the deterministic clusters"
        )

    expected_symbols = _material_symbols(dashboard)
    security_rows = narratives.get("security_explanations")
    if not isinstance(security_rows, list):
        raise ResponseContractError("v2 report requires security explanations")
    actual_symbols = [
        str(row.get("symbol") or "")
        for row in security_rows
        if isinstance(row, Mapping)
    ]
    if actual_symbols != expected_symbols:
        raise ResponseContractError(
            "v2 security narrative coverage does not match the deterministic lanes"
        )


def build_multistage_narratives(
    *,
    client: TrainedQuantClient,
    aggregate: Mapping[str, Any],
    radar: Mapping[str, Any],
    market_judgement: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all-cluster, material-security, then final editorial explanation."""

    dashboard = build_market_dashboard(aggregate, radar)
    global_context = _global_context(
        aggregate,
        radar,
        dashboard,
        market_judgement,
    )
    clusters = list(radar.get("integrated_rotation_clusters") or [])
    sector_items: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    sector_batches = _chunks(clusters, SECTOR_BATCH_SIZE)

    def sector_call(
        payload: tuple[int, list[Mapping[str, Any]]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        batch_number, batch = payload
        prepared_batch = [
            {
                **dict(row),
                "required_state_phrase": label_rotation_state(
                    row.get("integrated_state")
                ),
            }
            for row in batch
        ]
        expected = [
            str(row.get("integrated_cluster") or "") for row in prepared_batch
        ]
        return _cached_call_items(
            checkpoint_dir=checkpoint_dir,
            client=client,
            stage=f"sector_batch_{batch_number}",
            id_key="cluster",
            expected_ids=expected,
            text_fields=(
                "headline",
                "explanation",
                "counterpoint",
                "stock_context",
            ),
            global_context=global_context,
            batch_context=prepared_batch,
            required_mentions={
                str(row.get("integrated_cluster") or ""): {
                    "stock_context": [
                        _parse_related_stock(value)
                        for value in (row.get("top_related_stocks") or [])[:4]
                        if _parse_related_stock(value)
                    ]
                }
                for row in prepared_batch
            },
            required_all_mentions={
                str(row.get("integrated_cluster") or ""): {
                    "headline": [str(row["required_state_phrase"])]
                }
                for row in prepared_batch
            },
        )

    with ThreadPoolExecutor(max_workers=min(3, len(sector_batches) or 1)) as pool:
        sector_outputs = list(
            pool.map(sector_call, enumerate(sector_batches, 1))
        )
    for items, trace in sector_outputs:
        sector_items.extend(items)
        traces.append(trace)

    material_symbols = _material_symbols(dashboard)
    contexts = _security_contexts(material_symbols, results, radar)
    security_items: list[dict[str, Any]] = []
    security_batches = _chunks(contexts, SECURITY_BATCH_SIZE)

    def security_call(
        payload: tuple[int, list[Mapping[str, Any]]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        batch_number, batch = payload
        expected = [str(row["symbol"]) for row in batch]
        return _cached_call_items(
            checkpoint_dir=checkpoint_dir,
            client=client,
            stage=f"security_batch_{batch_number}",
            id_key="symbol",
            expected_ids=expected,
            text_fields=(
                "headline",
                "group_context",
                "etf_transmission",
                "counterpoint",
                "watch_condition",
            ),
            global_context=global_context,
            batch_context=batch,
            required_mentions={
                str(row["symbol"]): {
                    "headline": [str(row["symbol"])],
                    "etf_transmission": [
                        str(etf.get("etf_ticker") or "")
                        for etf in (
                            row.get("brief", {}).get("top_contributing_etfs")
                            or []
                        )
                        if etf.get("etf_ticker")
                    ],
                    **(
                        {
                            "group_context": list(
                                CLUSTER_LABELS.get(
                                    str(
                                        row.get(
                                            "sector_memberships", [{}]
                                        )[0].get("cluster")
                                    ),
                                    (
                                        str(
                                            row.get(
                                                "sector_memberships", [{}]
                                            )[0].get("cluster")
                                        ),
                                    ),
                                )
                            )
                        }
                        if row.get("sector_memberships")
                        else {}
                    ),
                }
                for row in batch
            },
            required_all_mentions={
                str(row["symbol"]): {
                    "group_context": [
                        str(row["symbol"]),
                        *(
                            []
                            if row.get("sector_memberships")
                            else ["제공된 직접 섹터 연결 근거 없음"]
                        ),
                    ]
                }
                for row in batch
            },
            forbidden_mentions={
                str(row["symbol"]): {
                    "group_context": list(CLUSTER_TERMS)
                }
                for row in batch
                if not row.get("sector_memberships")
            },
        )

    with ThreadPoolExecutor(
        max_workers=min(4, len(security_batches) or 1)
    ) as pool:
        security_outputs = list(
            pool.map(security_call, enumerate(security_batches, 1))
        )
    for items, trace in security_outputs:
        security_items.extend(items)
        traces.append(trace)

    editorial, editorial_trace = _cached_editorial_call(
        checkpoint_dir=checkpoint_dir,
        client=client,
        global_context=global_context,
        sector_items=sector_items,
        security_items=security_items,
        unsupported_symbols=[
            str(row["symbol"])
            for row in contexts
            if not row.get("sector_memberships")
        ],
    )
    traces.append(editorial_trace)
    return {
        "schema_version": NARRATIVE_SCHEMA_VERSION,
        "coverage_policy": (
            "all deterministic rotation clusters; four leading stocks from each "
            "positive, negative, and divergence lane; final editorial reconciliation"
        ),
        "sector_explanations": sector_items,
        "security_explanations": security_items,
        "editorial": editorial,
        "sector_count": len(sector_items),
        "security_count": len(security_items),
        "model_call_count": len(traces),
    }, {
        "schema_version": "quant.ai_radar_multistage_narrative_trace.v1",
        "stages": traces,
        "model_call_count": len(traces),
    }
