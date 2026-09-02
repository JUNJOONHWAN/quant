"""LoRA-backed decision-support assessment for an explicitly requested security."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .decision_support import build_security_brief
from .model_runtime import (
    ModelResponseParseError,
    ResponseContractError,
    TrainedQuantClient,
    canonical_json,
)


ACTION_SCHEMA_VERSION = "quant.ai_radar_action_assessment.v3"
ACTION_PROMPT_CONTRACT = "quant.ai_radar_two_stage_candidate_judgement.v1"
ACTION_VIEWS = (
    "매수 검토",
    "보유 관찰",
    "관망",
    "비중 축소 검토",
    "회피",
)
HORIZONS = ("단기", "중기", "장기", "확인 불가")
RAW_CODE = re.compile(
    r"\b(?:rotation_in|rotation_out|price_flow_[a-z_]+|mixed_or_flat|"
    r"insufficient_joint_evidence)\b"
)
DATE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


def _schema(symbol: str) -> dict[str, Any]:
    prose = {"type": "string", "minLength": 24, "maxLength": 600}
    return {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "enum": [symbol]},
            "action_view": {"type": "string", "enum": list(ACTION_VIEWS)},
            "horizon": {"type": "string", "enum": list(HORIZONS)},
            "historical_pattern": prose,
            "reason": prose,
            "supporting_evidence": prose,
            "counter_evidence": prose,
            "invalidation_condition": prose,
        },
        "required": [
            "symbol", "action_view", "horizon", "historical_pattern",
            "reason", "supporting_evidence", "counter_evidence",
            "invalidation_condition",
        ],
        "additionalProperties": False,
    }


def _validate(
    value: Mapping[str, Any], *, symbol: str, as_of_date: str
) -> dict[str, Any]:
    expected = {
        "symbol", "action_view", "horizon", "historical_pattern", "reason",
        "supporting_evidence", "counter_evidence", "invalidation_condition",
    }
    if set(value) != expected:
        raise ResponseContractError(
            f"action assessment fields mismatch: expected={sorted(expected)} "
            f"actual={sorted(value)}"
        )
    if str(value.get("symbol")) != symbol:
        raise ResponseContractError("action assessment symbol mismatch")
    if value.get("action_view") not in ACTION_VIEWS:
        raise ResponseContractError("action assessment has invalid action_view")
    if value.get("horizon") not in HORIZONS:
        raise ResponseContractError("action assessment has invalid horizon")
    normalized = dict(value)
    for field in expected - {"symbol", "action_view", "horizon"}:
        text = str(normalized.get(field) or "").strip()
        if len(re.findall(r"[가-힣]", text)) < 10:
            raise ResponseContractError(
                f"action assessment {field} is not substantive Korean"
            )
        if RAW_CODE.search(text):
            raise ResponseContractError(
                f"action assessment {field} contains a raw state code"
            )
        if DATE.search(text):
            raise ResponseContractError(
                f"action assessment {field} contains a raw date"
            )
        normalized[field] = text
    normalized["schema_version"] = ACTION_SCHEMA_VERSION
    normalized["prompt_contract"] = ACTION_PROMPT_CONTRACT
    normalized["decision_scope"] = "candidate_assessment_not_trade_execution"
    normalized["as_of_date"] = as_of_date
    return normalized


def build_action_assessment(
    *, client: TrainedQuantClient, result: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask the released LoRA for a bounded candidate assessment with reasons."""

    symbol = str(result.get("symbol") or "")
    judgement = result.get("judgement") or {}
    facts = judgement.get("facts") or {}
    as_of_date = str(facts.get("as_of_date") or "")
    brief = build_security_brief(judgement)
    decision_input = {
        "symbol": symbol,
        "as_of_date": as_of_date,
        "interpretation": judgement.get("interpretation") or {},
        "counter_evidence": judgement.get("counter_evidence") or [],
        "unknowns": judgement.get("unknowns") or [],
        "regime": judgement.get("regime"),
        "confidence": judgement.get("confidence"),
        "conclusion": judgement.get("conclusion"),
    }
    system = (
        "You are the released Quant AI Radar LoRA decision-support layer. "
        "The model was trained on historical FMP and ETF relationship data. "
        "Use the learned historical pattern associations to classify the current "
        "security, but use only the supplied point-in-time facts. Choose one "
        "candidate view: 매수 검토, 보유 관찰, 관망, 비중 축소 검토, 회피. "
        "This response is a research opinion and cannot execute an order. You may "
        "state direct Korean buy, sell, hold, reduce, or avoid wording that matches "
        "the selected view. Explain why the learned pattern does or does "
        "not transfer to the current facts, include supporting and counter evidence, "
        "and state what would invalidate the assessment. Do not invent news, prices, "
        "dates, sectors, or future outcomes. Return JSON only. /no_think"
    )
    user = (
        f"SYMBOL={symbol}\n"
        f"AS_OF_DATE={as_of_date}\n"
        f"CURRENT_JUDGEMENT={canonical_json(decision_input)}\n"
        f"EXACT_SECURITY_BRIEF={canonical_json(brief)}\n"
        "The renderer owns exact numeric display; write Korean prose explaining "
        "the pattern and evidence. Do not use raw dates, internal regime codes, "
        "or imperative trading language."
    )
    schema = _schema(symbol)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    traces: list[dict[str, Any]] = []
    contract_error = ""
    for attempt in range(1, 4):
        attempt_messages = list(messages)
        if contract_error:
            attempt_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"CONTRACT_ERROR={contract_error}\n"
                        "Repair the JSON only. Keep the same symbol, choose exactly "
                        "one allowed candidate view, and make every explanation a "
                        "substantive Korean sentence without dates, raw codes, or "
                        "missing evidence. Preserve the model's candidate opinion."
                    ),
                }
            )
        try:
            response, trace = client.complete_messages(
                messages=attempt_messages,
                max_tokens=1800,
                response_schema=schema,
            )
            traces.append(trace)
            assessment = _validate(
                response, symbol=symbol, as_of_date=as_of_date
            )
            return assessment, {
                "stage": "security_action_assessment",
                "contract_attempts": attempt,
                "contract_repair_applied": attempt > 1,
                "calls": traces,
            }
        except (ModelResponseParseError, ResponseContractError) as exc:
            if isinstance(exc, ModelResponseParseError) and exc.trace:
                traces.append(exc.trace)
            contract_error = f"{type(exc).__name__}: {exc}"
            if attempt == 3:
                raise
    raise ResponseContractError("security action assessment exhausted attempts")
