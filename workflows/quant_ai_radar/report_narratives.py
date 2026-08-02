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
    TrainedQuantClient,
    canonical_json,
)
from .presentation import label_regime, label_rotation_state, label_signal
NARRATIVE_SCHEMA_VERSION = "quant.ai_radar_multistage_narratives.v2"
NARRATIVE_CONTRACT_VERSION = "quant.ai_radar_narrative_contract.v11"
EDITORIAL_CONTRACT_VERSION = "quant.ai_radar_editorial_contract.v3"
LEARNED_PATTERN_PROMPT_CONTRACT = "quant.ai_radar_daily_learned_pattern.v1"
SECTOR_BATCH_SIZE = 3
SECURITY_BATCH_SIZE = 3
SECURITY_PER_LANE = 4
NARRATIVE_MAX_ATTEMPTS = 2
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
    "learned_pattern",
    "pattern_evidence",
    "pattern_risk",
}
HANGUL = re.compile(r"[가-힣]")
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|퍼센트|개|건|일|점)?")
RAW_CODE = re.compile(
    r"\b(?:rotation_in|rotation_out|price_flow_[a-z_]+|mixed_or_flat|"
    r"insufficient_joint_evidence)\b"
)
CAUSAL_OVERCLAIM = re.compile(
    r"(?:ETF|자금|섹터|회전)[^.]{0,80}(?:가격|주가|수익률)[^.]{0,50}"
    r"(?:영향|유발|원인)|"
    r"(?:가격|주가|수익률)[^.]{0,80}(?:ETF|자금|섹터|회전)[^.]{0,50}"
    r"(?:영향|유발|원인)"
)
DAILY_FUTURE_OR_ACTION = re.compile(
    r"(?:향후|미래|앞으로|과매수|목표가|매수|매도|비중\s*축소|"
    r"(?:상승|하락|조정|반등|회복)\s*가능성)"
)
UNSUPPORTED_MOTIVE = re.compile(
    r"(?:투자자|기관)[^.]{0,40}(?:심리|기대|관심|선호|의도)"
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
    "금융 서비스",
    "금융서비스",
    "금융주",
    "은행",
    "보험",
    "기술 섹터",
    "기술주",
    "IT",
    "정보기술",
    "비분류",
    "경기소비재",
    "방어소비재",
    "소비재",
    "소비재 방어",
    "소비재 순환",
    "소비재 섹터",
    "산업군",
    "에너지",
    "헬스케어",
    "건강케어",
    "유틸리티",
    "전력",
    "산업",
    "산업재",
    "산업주",
    "방산",
    "항공",
    "기술",
    "에너지 섹터",
    "유틸리티",
    "전력 섹터",
    "소재",
    "원자재",
    "부동산",
    "리츠",
    "통신",
    "통신서비스",
    "커뮤니케이션",
    "미디어",
    "반도체",
    "소프트웨어",
    "바이오",
    "의약",
    "제약",
    "성장주",
    "가치주",
    "필수소비재",
    "경기소비재",
    "소비자",
)
CLUSTER_LABELS = {
    "Healthcare": ("Healthcare", "건강케어", "헬스케어"),
    "Industrials": ("Industrials", "산업"),
    "Financial Services": ("Financial Services", "금융"),
    "Technology": ("Technology", "기술"),
    "Unclassified": ("Unclassified", "비분류"),
    "Consumer Cyclical": ("Consumer Cyclical", "경기소비재"),
    "Consumer Defensive": (
        "Consumer Defensive",
        "방어소비재",
        "필수소비재",
        "소비재 방어",
        "소비재 섹터",
        "소비재",
    ),
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


def _unsupported_cluster_terms(cluster: str) -> list[str]:
    """Reject other groups without rejecting substrings of the verified label."""

    allowed = set(CLUSTER_LABELS.get(cluster, (cluster,)))
    return [
        term
        for term in CLUSTER_TERMS
        if term not in allowed
        and not any(term in allowed_term for allowed_term in allowed)
    ]


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _clean_prose(value: Any) -> tuple[str, bool]:
    original = str(value or "").strip()
    cleaned = re.sub(r"\s+", " ", original).strip()
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


def _adapt_native_narrative_items(
    response: Mapping[str, Any],
    *,
    id_key: str,
    expected_ids: Sequence[str],
    text_fields: Sequence[str],
    enum_fields: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], bool]:
    rows = response.get("items")
    if not isinstance(rows, list):
        return dict(response), False
    adapted_rows: list[dict[str, Any]] = []
    changed = False
    for identifier in expected_ids:
        source = next(
            (
                row
                for row in rows
                if isinstance(row, Mapping)
                and str(row.get(id_key) or "") == identifier
            ),
            None,
        )
        if source is None:
            continue
        adapted: dict[str, Any] = {id_key: identifier}
        for field in text_fields:
            value = source.get(field)
            if field == "explanation" and not isinstance(value, str):
                value = source.get("supporting_analysis")
                changed = changed or isinstance(value, str)
            if isinstance(value, list):
                strings = [str(item).strip() for item in value if str(item).strip()]
                value = ", ".join(strings) if field == "stock_context" else (
                    strings[0] if strings else ""
                )
                changed = True
            adapted[field] = value
        for field in (enum_fields or {}):
            adapted[field] = source.get(field)
        if set(source) != set(adapted):
            changed = True
        adapted_rows.append(adapted)
    if len(adapted_rows) != len(rows):
        changed = True
    return {"items": adapted_rows}, changed


def _validate_items(
    response: Mapping[str, Any],
    *,
    id_key: str,
    expected_ids: Sequence[str],
    text_fields: Sequence[str],
    required_mentions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    required_all_mentions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    forbidden_mentions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    enum_fields: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    rows = response.get("items")
    if not isinstance(rows, list):
        raise ResponseContractError("narrative response items must be a list")
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ResponseContractError("narrative item must be an object")
        for field in text_fields:
            if not isinstance(row.get(field), str):
                raise ResponseContractError(
                    f"narrative item {field} must be a Korean prose string"
                )
        normalized, _ = _normalize_item(row)
        identifier = str(normalized.get(id_key) or "")
        if not identifier:
            raise ResponseContractError(f"narrative item is missing {id_key}")
        # Group membership is a source-of-truth relationship, not an AI claim.
        # Canonicalize it from the deterministic contract so the model cannot
        # copy unrelated clusters from the full market context into one symbol.
        if id_key == "symbol" and "group_context" in text_fields:
            allowed_groups = list(
                (required_mentions or {})
                .get(identifier, {})
                .get("group_context", ())
            )
            if allowed_groups:
                label = next(
                    (
                        str(value)
                        for value in allowed_groups
                        if any("가" <= char <= "힣" for char in str(value))
                    ),
                    str(allowed_groups[0]),
                )
                normalized["group_context"] = (
                    f"{identifier}는 {label} 연결이 제공된 데이터에서 확인됩니다."
                )
            else:
                normalized["group_context"] = (
                    f"{identifier}는 제공된 직접 섹터 연결 근거 없음으로 분류됩니다."
                )
            # The model may repeat another cluster from COMPLETE_MARKET_CONTEXT
            # in the explanatory prose. Keep the AI judgement, but neutralize
            # unsupported cluster labels before any report or email renderer sees it.
            allowed_group_set = set(allowed_groups)
            for field in text_fields:
                if field == "group_context":
                    continue
                text = str(normalized.get(field) or "")
                for term in sorted(CLUSTER_TERMS, key=len, reverse=True):
                    if term in allowed_group_set:
                        continue
                    if re.fullmatch(r"[A-Za-z][A-Za-z ]*", term):
                        text = re.sub(
                            rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])",
                            "시장 회전 영역",
                            text,
                        )
                    else:
                        text = re.sub(
                            rf"{re.escape(term)}\s*섹터",
                            "시장 회전 영역",
                            text,
                        )
                        text = text.replace(term, "시장 회전 영역")
                text = text.replace("시장 회전 영역 섹터", "시장 회전 영역")
                text = text.replace("시장 회전 영역는", "시장 회전 영역은")
                text = text.replace("시장 회전 영역가", "시장 회전 영역이")
                normalized[field] = text
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
            if RAW_CODE.search(text):
                raise ResponseContractError(
                    f"{identifier} narrative contains a raw state code"
                )
            if NUMBER.search(text):
                raise ResponseContractError(
                    f"{identifier} narrative {field} contains renderer-owned numbers"
                )
            if CAUSAL_OVERCLAIM.search(text):
                raise ResponseContractError(
                    f"{identifier} narrative {field} overstates causality"
                )
            if DAILY_FUTURE_OR_ACTION.search(text):
                raise ResponseContractError(
                    f"{identifier} narrative {field} forecasts beyond the daily as-of scope"
                )
            if UNSUPPORTED_MOTIVE.search(text):
                raise ResponseContractError(
                    f"{identifier} narrative {field} invents an investor motive"
                )
            if field not in ("headline", "stock_context") and not (
                _is_complete_sentence(text)
            ):
                raise ResponseContractError(
                    f"{identifier} narrative {field} is not a complete sentence"
                )
        for field, allowed in (enum_fields or {}).items():
            value = str(normalized.get(field) or "")
            if value not in set(allowed):
                raise ResponseContractError(
                    f"{identifier} narrative {field} must be one of {list(allowed)}"
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
    enum_fields: Mapping[str, Sequence[str]] | None = None,
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
            "pattern": r"^[^0-9]*$",
        }
    for field, allowed in (enum_fields or {}).items():
        properties[field] = {"type": "string", "enum": list(allowed)}
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
                    "required": [id_key, *text_fields, *(enum_fields or {})],
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
    enum_fields: Mapping[str, Sequence[str]] | None = None,
    max_attempts: int = NARRATIVE_MAX_ATTEMPTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_attempts <= 0:
        raise ResponseContractError(
            f"{stage} is configured for single-item generation"
        )
    schema = _items_schema(
        id_key=id_key,
        expected_ids=expected_ids,
        text_fields=text_fields,
        enum_fields=enum_fields,
    )
    system = (
        "You are the released Quant AI Radar LoRA explanation layer. Read the complete "
        "deterministic market context first, then explain each requested subset in that "
        "context. Use only supplied point-in-time facts. Connect market breadth, sector "
        "rotation, ETF capital transmission, the item's own price/flow relationship, "
        "counter-evidence, and learned historical-pattern interpretation. Explain only the "
        "current as-of structure; do not forecast a future return or classify a buy/sell action. "
        "Write concrete Korean report prose. Do not write digits, percentages, dates, "
        "raw code labels; the renderer owns every exact number. This report cannot execute "
        "orders. Do not infer investor psychology, expectations, interest, preference, or "
        "intent from price and ETF Flow. In Korean prose, do not use 투자자, 기관, "
        "심리, 기대, 관심, 선호, or 의도 to explain observed price/flow structure. "
        "Preserve the requested item order. "
        "End every prose field except stock_context as a complete sentence. "
        "Never infer a security's sector when sector_memberships is empty. In that case, "
        "group_context must contain the exact phrase '제공된 직접 섹터 연결 근거 없음' "
        "and must not name any sector, cluster, industry, or consumer category. "
        "Every headline must include its requested symbol and a substantive Korean "
        "sentence; never return a ticker alone. "
        "Describe price and ETF flow as "
        "aligned, divergent, or associated; never claim that one causes or affects the "
        "other. Return JSON only. /no_think"
    )
    unsupported_group_symbols = sorted((forbidden_mentions or {}).keys())
    if unsupported_group_symbols:
        system += (
            " UNSUPPORTED_GROUP_SYMBOLS="
            f"{canonical_json(unsupported_group_symbols)}. For every one of these "
            "symbols, group_context must be exactly the symbol followed by the phrase "
            "'제공된 직접 섹터 연결 근거 없음' and a neutral statement that no direct "
            "group link was supplied. Do not copy any sector, industry, consumer, or "
            "cluster label from COMPLETE_MARKET_CONTEXT or another item into those "
            "group_context fields."
        )
    user = (
        f"STAGE={stage}\n"
        f"EXPECTED_IDS={canonical_json(list(expected_ids))}\n"
        f"COMPLETE_MARKET_CONTEXT={canonical_json(global_context)}\n"
        f"REQUESTED_SUBSET={canonical_json(list(batch_context))}\n"
        f"REQUIRED_MENTIONS={canonical_json(required_mentions or {})}\n"
        f"REQUIRED_ALL_MENTIONS={canonical_json(required_all_mentions or {})}\n"
        f"FORBIDDEN_MENTIONS={canonical_json(forbidden_mentions or {})}\n"
        f"UNSUPPORTED_GROUP_SYMBOLS={canonical_json(unsupported_group_symbols)}\n"
        "UNSUPPORTED_GROUP_RULE=If sector_memberships is empty, write the exact Korean "
        "phrase '제공된 직접 섹터 연결 근거 없음' in group_context for every listed "
        "UNSUPPORTED_GROUP_SYMBOLS ticker, and do not write any sector or category label, "
        "including generic labels."
    )
    traces: list[dict[str, Any]] = []
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    contract_error = ""
    for attempt in range(1, max_attempts + 1):
        messages = list(base_messages)
        if contract_error:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"CONTRACT_ERROR={contract_error}\n"
                        "Return every expected item exactly once and in the exact "
                        "requested order. Every prose field must be a complete Korean "
                        "report sentence with no digits or raw code labels. Preserve direct "
                        "Do not write investor motives, future possibilities, or causal claims. "
                        "Do not use 투자자, 기관, 심리, 기대, 관심, 선호, or 의도; "
                        "rewrite those claims as observable price/flow association. "
                        "End every prose field except stock_context as a "
                        "complete sentence. Obey REQUIRED_MENTIONS and "
                        "REQUIRED_ALL_MENTIONS exactly and do not use any "
                        "FORBIDDEN_MENTIONS. If sector_memberships is empty, replace "
                        "group_context for each UNSUPPORTED_GROUP_SYMBOLS ticker with the "
                        "exact phrase '제공된 직접 섹터 연결 근거 없음' and remove every "
                        "sector/category label. Ensure every headline includes its symbol "
                        "and a substantive Korean sentence, never a ticker alone. Describe association "
                        "without causal claims."
                    ),
                }
            )
        try:
            response, trace = client.complete_messages(
                messages=messages,
                max_tokens=min(3600, 2400 * len(expected_ids)),
                response_schema=schema,
            )
            traces.append(trace)
            response, native_adapter = _adapt_native_narrative_items(
                response,
                id_key=id_key,
                expected_ids=expected_ids,
                text_fields=text_fields,
                enum_fields=enum_fields,
            )
            items = _validate_items(
                response,
                id_key=id_key,
                expected_ids=expected_ids,
                text_fields=text_fields,
                required_mentions=required_mentions,
                required_all_mentions=required_all_mentions,
                forbidden_mentions=forbidden_mentions,
                enum_fields=enum_fields,
            )
            trace["native_contract_adapter_applied"] = native_adapter
            break
        except (ModelResponseParseError, ResponseContractError) as exc:
            if isinstance(exc, ModelResponseParseError) and exc.trace:
                traces.append(exc.trace)
            contract_error = f"{type(exc).__name__}: {exc}"
            if attempt == max_attempts:
                if not isinstance(exc, ModelResponseParseError):
                    setattr(exc, "trace", traces[-1] if traces else None)
                    if "response" in locals():
                        setattr(exc, "raw_content", canonical_json(response))
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
    split_on_failure = bool(kwargs.pop("split_on_failure", False))
    if checkpoint_dir is None:
        try:
            return _call_items(**kwargs)
        except (ModelResponseParseError, ResponseContractError) as exc:
            if split_on_failure:
                return _split_failed_batch(kwargs=kwargs, error=exc)
            raise
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
            "enum_fields",
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
                enum_fields=kwargs.get("enum_fields"),
            )
            trace = dict(cached.get("trace") or {})
            trace["cache_hit"] = True
            return items, trace
    try:
        items, trace = _call_items(**kwargs)
    except (ModelResponseParseError, ResponseContractError) as exc:
        if not split_on_failure:
            _write_failed_narrative_checkpoint(
                checkpoint_dir=checkpoint_dir,
                stage=stage,
                error=exc,
            )
            raise
        items, trace = _split_failed_batch(
            kwargs={**kwargs, "checkpoint_dir": checkpoint_dir},
            error=exc,
        )
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


def _write_failed_narrative_checkpoint(
    *,
    checkpoint_dir: Path,
    stage: str,
    error: Exception,
) -> None:
    value = {
        "schema_version": "quant.ai_radar_narrative_failure.v1",
        "stage": stage,
        "error": f"{type(error).__name__}: {error}",
        "trace": getattr(error, "trace", None),
        "raw_content": getattr(error, "raw_content", None),
    }
    path = checkpoint_dir / f"{stage}.failure.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _split_failed_batch(
    *,
    kwargs: Mapping[str, Any],
    error: Exception,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_ids = list(kwargs.get("expected_ids") or [])
    batch_context = list(kwargs.get("batch_context") or [])
    if len(expected_ids) <= 1 or len(batch_context) != len(expected_ids):
        raise error
    base_stage = str(kwargs.get("stage") or "narrative_batch")
    items: list[dict[str, Any]] = []
    sub_stages: list[dict[str, Any]] = []
    for index, identifier in enumerate(expected_ids, 1):
        single = dict(kwargs)
        single["stage"] = f"{base_stage}_item_{index}"
        single["expected_ids"] = [identifier]
        single["batch_context"] = [batch_context[index - 1]]
        single["max_attempts"] = NARRATIVE_MAX_ATTEMPTS
        for key in (
            "required_mentions",
            "required_all_mentions",
            "forbidden_mentions",
        ):
            mapping = kwargs.get(key)
            single[key] = (
                {identifier: dict(mapping.get(identifier) or {})}
                if isinstance(mapping, Mapping)
                else None
            )
        checkpoint_dir = single.pop("checkpoint_dir", None)
        single_items, single_trace = _cached_call_items(
            checkpoint_dir=checkpoint_dir,
            split_on_failure=False,
            **single,
        )
        items.extend(single_items)
        sub_stages.append(single_trace)
    return items, {
        "stage": base_stage,
        "batch_fallback": "single_item_after_contract_failure",
        "batch_error": f"{type(error).__name__}: {error}",
        "contract_attempts": sum(
            int(stage.get("contract_attempts") or 0) for stage in sub_stages
        ),
        "contract_repair_applied": True,
        "calls": [
            call
            for stage in sub_stages
            for call in (stage.get("calls") or [])
        ],
        "sub_stages": sub_stages,
    }


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


def _native_limitation_text(values: Sequence[Any]) -> str:
    labels = {
        "price_and_etf_flow_signals_diverge": "가격과 ETF 자금 신호가 엇갈립니다.",
        "historical_backfill_not_true_as_observed_point_in_time": (
            "과거 원장은 당시 관측 화면과 동일하지 않다는 제한이 있습니다."
        ),
        "no_etf_flow_visible_under_session_lag_policy": (
            "가시성 지연 정책을 통과한 ETF 자금 근거가 없습니다."
        ),
        "insufficient_price_history_for_short_horizon_statistics": (
            "가격 관측 이력이 짧아 단기 통계 근거가 제한됩니다."
        ),
        "mixed_flow_currencies_prevent_aggregation": (
            "자금 통화가 섞여 합산 해석이 제한됩니다."
        ),
    }
    rendered = [labels.get(str(value), "추가 확인이 필요한 제한 근거가 있습니다.") for value in values]
    return rendered[0] if rendered else "네이티브 판단에 별도 제한 근거가 기록되지 않았습니다."


def build_native_judgement_narratives(
    *,
    aggregate: Mapping[str, Any],
    radar: Mapping[str, Any],
    market_judgement: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render only the model's training-native judgement contract."""

    dashboard = build_market_dashboard(aggregate, radar)
    sector_items: list[dict[str, Any]] = []
    for row in radar.get("integrated_rotation_clusters") or []:
        cluster = str(row.get("integrated_cluster") or "비분류")
        state_label = label_rotation_state(row.get("integrated_state"))
        tickers = [
            _parse_related_stock(value)
            for value in (row.get("top_related_stocks") or [])[:4]
            if _parse_related_stock(value)
        ]
        sector_items.append(
            {
                "cluster": cluster,
                "headline": f"{cluster} · {state_label}",
                "explanation": (
                    f"{cluster}은 공통 DB 가격·ETF 자금 집계에서 {state_label} 상태로 분류됩니다."
                ),
                "counterpoint": (
                    "구성 종목의 학습 모델 판단에는 같은 방향과 괴리 방향이 함께 포함될 수 있습니다."
                ),
                "stock_context": ", ".join(tickers),
            }
        )

    material_symbols = _material_symbols(dashboard)
    contexts = _security_contexts(material_symbols, results, radar)
    by_symbol = {str(row.get("symbol") or ""): row for row in results}
    security_items: list[dict[str, Any]] = []
    for context in contexts:
        symbol = str(context["symbol"])
        judgement = (by_symbol[symbol].get("judgement") or {})
        interpretation = judgement.get("interpretation") or {}
        regime_label = label_regime(judgement.get("regime"))
        price_label = label_signal(interpretation.get("price_signal"))
        flow_label = label_signal(interpretation.get("etf_flow_signal"))
        memberships = context.get("sector_memberships") or []
        if memberships:
            cluster = str(memberships[0].get("cluster") or "비분류")
            group_context = f"{symbol}는 {cluster} 연결이 제공된 데이터에서 확인됩니다."
        else:
            group_context = f"{symbol}는 제공된 직접 섹터 연결 근거 없음으로 분류됩니다."
        contributors = context.get("brief", {}).get("top_contributing_etfs") or []
        etf = str((contributors[0] if contributors else {}).get("etf_ticker") or "")
        etf_transmission = (
            f"{etf}가 제공된 구성종목 ETF 자금 전달 근거에 포함됩니다."
            if etf
            else "가시성 조건을 통과한 직접 ETF 자금 전달 근거가 없습니다."
        )
        counter_values = list(judgement.get("counter_evidence") or [])
        unknown_values = list(judgement.get("unknowns") or [])
        security_items.append(
            {
                "symbol": symbol,
                "headline": f"{symbol} · {regime_label}",
                "group_context": group_context,
                "etf_transmission": etf_transmission,
                "counterpoint": _native_limitation_text(counter_values),
                "watch_condition": _native_limitation_text(unknown_values),
                "learned_pattern": (
                    f"학습 모델은 현재 가격과 ETF 자금의 결합을 {regime_label} 패턴으로 분류했습니다."
                ),
                "pattern_evidence": (
                    f"학습 입력에서 가격 신호는 {price_label}, ETF 자금 신호는 {flow_label}로 해석됐습니다."
                ),
                "pattern_risk": _native_limitation_text(
                    [*counter_values, *unknown_values]
                ),
            }
        )

    named = material_symbols[:3]
    selection_summary = (
        f"주요 종목은 {', '.join(named)}입니다. "
        "각 종목의 학습 모델 네이티브 판단과 ETF 전달 근거를 함께 확인합니다."
        if named
        else "학습 모델의 적격 주요 종목이 확인되지 않았습니다. 현재 원장의 품질 조건을 먼저 확인합니다."
    )
    editorial = {
        "headline": "학습 모델 네이티브 판단으로 본 오늘의 시장 구조",
        "executive_summary": str(market_judgement.get("summary") or "학습 모델 판단을 집계했습니다."),
        "rotation_summary": "공통 DB 섹터 회전 집계와 종목별 학습 모델 패턴을 같은 기준일에서 대조합니다.",
        "selection_summary": selection_summary,
        "risk_summary": "가격과 ETF 자금이 엇갈리거나 가시성 제한이 기록된 항목을 반대 근거로 함께 봅니다.",
    }
    calls = [dict(row.get("trace") or {}) for row in results]
    narratives = {
        "schema_version": NARRATIVE_SCHEMA_VERSION,
        "coverage_policy": "training-native judgements aggregated without extra model prompts",
        "generation_source": "quant.analysis_packet.v3 training-native outputs",
        "sector_explanations": sector_items,
        "security_explanations": security_items,
        "editorial": editorial,
        "sector_count": len(sector_items),
        "security_count": len(security_items),
        "model_call_count": len(calls),
        "learned_pattern_prompt_contract": "quant.analysis_packet.v3",
    }
    trace = {
        "schema_version": "quant.ai_radar_native_judgement_trace.v1",
        "stages": [
            {
                "stage": "training_native_judgement_aggregation",
                "contract_attempts": 0,
                "contract_repair_applied": False,
                "calls": calls,
            }
        ],
        "model_call_count": len(calls),
        "learned_pattern_prompt_contract": "quant.analysis_packet.v3",
    }
    return narratives, trace


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
        if RAW_CODE.search(text):
            raise ResponseContractError(
                f"editorial {field} contains a raw state code"
            )
        if NUMBER.search(text):
            raise ResponseContractError(
                f"editorial {field} contains renderer-owned numbers"
            )
        if CAUSAL_OVERCLAIM.search(text):
            raise ResponseContractError(
                f"editorial {field} overstates causality"
            )
        if DAILY_FUTURE_OR_ACTION.search(text):
            raise ResponseContractError(
                f"editorial {field} forecasts beyond the daily as-of scope"
            )
        if UNSUPPORTED_MOTIVE.search(text):
            raise ResponseContractError(
                f"editorial {field} invents an investor motive"
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
        "or meta instructions. Explain only the current as-of market structure and do not "
        "forecast future returns, write buy/sell classifications, or infer investor motives. "
        "Do not use 투자자, 기관, 심리, 기대, 관심, 선호, or 의도; describe only "
        "observable price/flow association. "
        "Securities without supplied direct sector "
        "links are explained in separate cards and must be omitted entirely from every "
        f"editorial field. OMIT_SECURITY_SYMBOLS={canonical_json(unsupported_symbols)}. "
        "The selection_summary must name at least three securities from "
        f"SUPPORTED_GROUPED_SYMBOLS={canonical_json(supported_symbols)}. Describe association "
        "without saying that ETF flow or sector rotation causes or affects price. Write "
        "selection_summary as exactly three short, distinct sentences, each ending "
        "with a period. Start each sentence with a different symbol from "
        "SUPPORTED_GROUPED_SYMBOLS; cover the selected name, its evidence relationship, "
        "and counter-evidence without repeating any sentence. "
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
    for attempt in range(1, NARRATIVE_MAX_ATTEMPTS + 1):
        repair = (
            ""
            if not contract_error
            else (
                f"\nCONTRACT_ERROR={contract_error}\n"
                "Rewrite every required field as substantive Korean report prose. "
                "Do not omit fields, use digits, expose raw codes, forecast returns, add buy/sell classifications, or infer investor motives. "
                "Do not use 투자자, 기관, 심리, 기대, 관심, 선호, or 의도; "
                "rewrite those claims as observable price/flow association. "
                "Do not mention any UNSUPPORTED_SECTOR_ASSIGNMENT_SYMBOLS ticker anywhere "
                "in the editorial; its separate security card owns that explanation. "
                "Name at least three different SUPPORTED_GROUPED_SYMBOLS in selection_summary "
                "and describe association without causal language. Rewrite selection_summary "
                "as exactly three distinct sentences: begin each sentence with a different "
                "listed symbol, end each with a period, and do not repeat a sentence."
            )
        )
        try:
            response, trace = client.complete(
                system=system,
                user=user + repair,
                # The editorial prompt contains every sector and selected-security
                # explanation. Keep the completion budget below the FLOW 16K context
                # ceiling even when the input reaches the bounded 14K-token edge.
                # Keep the final editor inside the FLOW 16K context after the
                # explicit unsupported-symbol guard is included in the prompt.
                max_tokens=1200,
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
                "program_prose_normalization_applied": stripped,
                "calls": traces,
            }
        except (ModelResponseParseError, ResponseContractError) as exc:
            if isinstance(exc, ModelResponseParseError) and exc.trace:
                traces.append(exc.trace)
            contract_error = f"{type(exc).__name__}: {exc}"
            if attempt == NARRATIVE_MAX_ATTEMPTS:
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
    required_security_fields = (
        "headline",
        "group_context",
        "etf_transmission",
        "counterpoint",
        "watch_condition",
        "learned_pattern",
        "pattern_evidence",
        "pattern_risk",
    )
    for row in security_rows:
        if not isinstance(row, Mapping):
            raise ResponseContractError("v2 security explanation must be an object")
        symbol = str(row.get("symbol") or "")
        if "action_view" in row:
            raise ResponseContractError(
                f"{symbol} daily security explanation must not contain action_view"
            )
        for field in required_security_fields:
            text = str(row.get(field) or "").strip()
            if not text:
                raise ResponseContractError(
                    f"{symbol} security explanation is missing {field}"
                )
            if NUMBER.search(text):
                raise ResponseContractError(
                    f"{symbol} security explanation {field} contains renderer-owned numbers"
                )
            if RAW_CODE.search(text):
                raise ResponseContractError(
                    f"{symbol} security explanation {field} contains a raw state code"
                )
            if CAUSAL_OVERCLAIM.search(text):
                raise ResponseContractError(
                    f"{symbol} security explanation {field} overstates causality"
                )
            if DAILY_FUTURE_OR_ACTION.search(text):
                raise ResponseContractError(
                    f"{symbol} security explanation {field} forecasts beyond the daily as-of scope"
                )
            if UNSUPPORTED_MOTIVE.search(text):
                raise ResponseContractError(
                    f"{symbol} security explanation {field} invents an investor motive"
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
        prepared_batch = [dict(row) for row in batch]
        expected = [
            str(row.get("integrated_cluster") or "") for row in prepared_batch
        ]
        return _cached_call_items(
            checkpoint_dir=checkpoint_dir,
            split_on_failure=len(prepared_batch) > 1,
            max_attempts=(
                0 if len(prepared_batch) > 1 else NARRATIVE_MAX_ATTEMPTS
            ),
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
            split_on_failure=len(batch) > 1,
            max_attempts=(0 if len(batch) > 1 else NARRATIVE_MAX_ATTEMPTS),
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
                "learned_pattern",
                "pattern_evidence",
                "pattern_risk",
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
                    "group_context": (
                        list(CLUSTER_TERMS)
                        if not row.get("sector_memberships")
                        else [
                            *_unsupported_cluster_terms(
                                str(
                                    row.get(
                                        "sector_memberships", [{}]
                                    )[0].get("cluster")
                                )
                            ),
                            *NO_DIRECT_CLUSTER_LINK_TERMS,
                        ]
                    )
                }
                for row in batch
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
        "learned_pattern_prompt_contract": LEARNED_PATTERN_PROMPT_CONTRACT,
    }, {
        "schema_version": "quant.ai_radar_multistage_narrative_trace.v1",
        "stages": traces,
        "model_call_count": len(traces),
        "learned_pattern_prompt_contract": LEARNED_PATTERN_PROMPT_CONTRACT,
    }
