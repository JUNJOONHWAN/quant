"""Qwen 27B synthesis over 8B learned patterns and computed analogues."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .model_runtime import InferenceError, ResponseContractError, canonical_json
from .model_runtime import parse_json_object


FORECAST_SCHEMA_VERSION = "quant.qwen27_individual_forecast.v1"
FORECAST_VIEWS = (
    "매수 검토",
    "보유 관찰",
    "관망",
    "비중 축소 검토",
    "회피",
)
FORECAST_HORIZONS = (5, 20, 60)
DEFAULT_FORECAST_ENDPOINT = "http://127.0.0.1:8004/v1/chat/completions"
DEFAULT_FORECAST_MODEL = "rdtand/Qwen3.6-27B-PrismaAURA-5.5bit-vllm"
DIGIT = re.compile(r"\d")
RAW_SOURCE_ID = re.compile(
    r"\b(?:aggregate|oracle|stock|etf|evidence|regime_counts|price_signal_counts)\."
    r"[A-Za-z0-9_.-]+\b|\b(?:price_flow_[a-z_]+|rotation_[a-z_]+)\b",
    re.IGNORECASE,
)
HANGUL = re.compile(r"[가-힣]")
ENGLISH_STATE_LABEL = re.compile(
    r"\b(?:price|flow|positive|negative|confirmation|divergence|mixed|rotation)\b",
    re.IGNORECASE,
)


Transport = Callable[[dict[str, Any], Mapping[str, str], int], dict[str, Any]]


def _schema(symbol: str) -> dict[str, Any]:
    prose = {"type": "string", "minLength": 12, "maxLength": 900}
    prose_list = {
        "type": "array",
        "minItems": 1,
        "maxItems": 8,
        "items": {"type": "string", "minLength": 6, "maxLength": 500},
    }
    return {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "enum": [symbol]},
            "forecast_view": {"type": "string", "enum": list(FORECAST_VIEWS)},
            "primary_horizon_sessions": {
                "type": "integer",
                "enum": list(FORECAST_HORIZONS),
            },
            "thesis": prose,
            "learned_pattern_use": prose,
            "historical_evidence": prose,
            "market_context_effect": prose,
            "supporting_evidence": prose_list,
            "counter_evidence": prose_list,
            "invalidation_conditions": prose_list,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "symbol",
            "forecast_view",
            "primary_horizon_sessions",
            "thesis",
            "learned_pattern_use",
            "historical_evidence",
            "market_context_effect",
            "supporting_evidence",
            "counter_evidence",
            "invalidation_conditions",
            "confidence",
        ],
        "additionalProperties": False,
    }


def _validate(value: Mapping[str, Any], symbol: str) -> dict[str, Any]:
    expected = set(_schema(symbol)["required"])
    if set(value) != expected:
        raise ResponseContractError(
            f"27B forecast fields mismatch: expected={sorted(expected)} "
            f"actual={sorted(value)}"
        )
    if value.get("symbol") != symbol:
        raise ResponseContractError("27B forecast changed the requested symbol")
    if value.get("forecast_view") not in FORECAST_VIEWS:
        raise ResponseContractError("27B forecast returned an invalid view")
    if value.get("primary_horizon_sessions") not in FORECAST_HORIZONS:
        raise ResponseContractError("27B forecast returned an invalid horizon")
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
    ):
        raise ResponseContractError("27B forecast confidence is invalid")
    for field in (
        "thesis",
        "learned_pattern_use",
        "historical_evidence",
        "market_context_effect",
    ):
        text = str(value.get(field) or "").strip()
        if len(text) < 12 or len(HANGUL.findall(text)) < 6:
            raise ResponseContractError(f"27B forecast {field} is not substantive")
        if (
            DIGIT.search(text)
            or RAW_SOURCE_ID.search(text)
            or ENGLISH_STATE_LABEL.search(text)
        ):
            raise ResponseContractError(
                f"27B forecast {field} contains a new number or raw source id"
            )
    for field in (
        "supporting_evidence",
        "counter_evidence",
        "invalidation_conditions",
    ):
        rows = value.get(field)
        if not isinstance(rows, list) or not rows:
            raise ResponseContractError(f"27B forecast {field} must be a non-empty list")
        for row in rows:
            text = str(row).strip()
            if len(text) < 6 or len(HANGUL.findall(text)) < 3:
                raise ResponseContractError(f"27B forecast {field} has an empty item")
            if (
                DIGIT.search(text)
                or RAW_SOURCE_ID.search(text)
                or ENGLISH_STATE_LABEL.search(text)
            ):
                raise ResponseContractError(
                    f"27B forecast {field} contains a new number or raw source id"
                )
    return dict(value)


def _compact_judgement(judgement: Mapping[str, Any]) -> dict[str, Any]:
    facts = judgement.get("facts")
    facts = facts if isinstance(facts, Mapping) else {}
    return {
        "interpretation": judgement.get("interpretation") or {},
        "regime": judgement.get("regime"),
        "confidence": judgement.get("confidence"),
        "conclusion": judgement.get("conclusion"),
        "counter_evidence": judgement.get("counter_evidence") or [],
        "unknowns": judgement.get("unknowns") or [],
        "facts": {
            key: facts.get(key)
            for key in (
                "symbol",
                "as_of_date",
                "price",
                "etf_flow",
                "etf_flow_to_constituent",
                "etf_relations",
                "liquidity",
                "quality_status",
            )
        },
    }


@dataclass
class ForecastSynthesisClient:
    endpoint: str = DEFAULT_FORECAST_ENDPOINT
    model: str = DEFAULT_FORECAST_MODEL
    token: str | None = None
    timeout: int = 240
    transport: Transport | None = None

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise InferenceError("27B endpoint must be an absolute HTTP URL")

    def _http_transport(
        self, payload: dict[str, Any], headers: Mapping[str, str], timeout: int
    ) -> dict[str, Any]:
        raw_body = canonical_json(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=raw_body,
            method="POST",
            headers={**dict(headers), "content-length": str(len(raw_body))},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise InferenceError(
                f"27B endpoint request failed: HTTP {exc.code}: {detail[:2000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise InferenceError(f"27B endpoint request failed: {exc}") from exc
        if status != 200:
            raise InferenceError(f"27B endpoint returned HTTP {status}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InferenceError("27B endpoint returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise InferenceError("27B endpoint returned a non-object response")
        return value

    def synthesize(
        self,
        *,
        symbol: str,
        judgement: Mapping[str, Any],
        analog_forecast: Mapping[str, Any],
        market_context: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = _schema(symbol)
        system = (
            "You are the 27B forecast synthesis layer of Quant AI Radar. The released "
            "8B LoRA has identified the current learned price--ETF Flow pattern. A "
            "deterministic engine has searched the historical point-in-time corpus and "
            "calculated realised forward distributions. Reconcile those two sources with "
            "the supplied current market context and produce a probabilistic Korean "
            "individual-security outlook. Historical similarity is evidence, not a claim "
            "that history repeats. Python owns every number, probability, date, sample "
            "count, and return. Do not write any digit, date, price level, percentage, raw "
            "state code, evidence identifier, or database field name in prose; the renderer "
            "shows source numbers separately. Explain their direction and meaning in Korean. "
            "Give supporting evidence, counter-evidence, and observable qualitative "
            "invalidation conditions. The forecast is research only and cannot place an "
            "order. Return exactly one JSON object. /no_think"
        )
        evidence = {
            "symbol": symbol,
            "qwen8_learned_pattern": _compact_judgement(judgement),
            "historical_analog_forecast": analog_forecast,
            "current_market_context": market_context,
        }
        base_messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "SOURCE_BACKED_EVIDENCE_JSON=" + canonical_json(evidence),
            },
        ]
        headers = {
            "content-type": "application/json",
            "user-agent": "quant-ai-radar/2.0",
        }
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        transport = self.transport or self._http_transport
        calls: list[dict[str, Any]] = []
        contract_error = ""
        for attempt in range(1, 4):
            messages = list(base_messages)
            if contract_error:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"CONTRACT_ERROR={contract_error}\n"
                            "Keep the same evidence-based forecast judgement, but rewrite every "
                            "prose field in natural Korean without any digit, date, price level, "
                            "percentage, English state label, raw evidence id, database field, or "
                            "ticker-based source reference. Use qualitative observable invalidation "
                            "conditions only. Return every schema field exactly once. /no_think"
                        ),
                    }
                )
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "seed": 1111,
                "max_tokens": 1800,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "quant_ai_radar_individual_forecast",
                        "schema": schema,
                    },
                },
                "chat_template_kwargs": {"enable_thinking": False},
            }
            body = transport(payload, headers, self.timeout)
            returned_model = str(body.get("model") or "")
            if returned_model != self.model:
                raise InferenceError(
                    f"27B endpoint served a different model: expected={self.model!r} "
                    f"got={returned_model!r}"
                )
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise InferenceError(
                    "27B endpoint response has no assistant content"
                ) from exc
            if not isinstance(content, str):
                raise InferenceError("27B assistant content is not text")
            call_trace = {
                "endpoint_model": returned_model,
                "request_sha256": hashlib.sha256(
                    canonical_json(payload).encode("utf-8")
                ).hexdigest(),
                "response_sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                "finish_reason": (body.get("choices") or [{}])[0].get(
                    "finish_reason"
                ),
                "usage": body.get("usage"),
            }
            calls.append(call_trace)
            try:
                forecast = _validate(parse_json_object(content), symbol)
                break
            except (ResponseContractError, InferenceError) as exc:
                contract_error = f"{type(exc).__name__}: {exc}"
                if attempt == 3:
                    raise
        else:
            raise ResponseContractError("27B forecast exhausted contract attempts")
        forecast["schema_version"] = FORECAST_SCHEMA_VERSION
        final_call = calls[-1]
        trace = {
            **final_call,
            "source_evidence_sha256": hashlib.sha256(
                canonical_json(evidence).encode("utf-8")
            ).hexdigest(),
            "contract_attempts": len(calls),
            "contract_repair_applied": len(calls) > 1,
            "calls": calls,
        }
        return forecast, trace
