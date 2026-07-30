"""Strict release and inference contracts for the trained quant LoRA.

There is deliberately no alternate writer backend in this module. A release
that has not passed the frozen evaluation gate, an endpoint serving a different
model, or a response that changes deterministic facts is a hard failure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RELEASE_SCHEMA = "quant.trained_model_release.v1"
REQUIRED_RESPONSE_KEYS = (
    "facts",
    "interpretation",
    "counter_evidence",
    "unknowns",
    "regime",
    "confidence",
    "conclusion",
)
REGIMES = {
    "insufficient_joint_evidence",
    "price_flow_positive_confirmation",
    "price_flow_negative_confirmation",
    "price_up_flow_out_divergence",
    "price_down_flow_in_divergence",
    "mixed_or_flat",
}
SIGNALS = {"positive", "negative", "flat", "unknown"}
FLOW_SIGNAL_SOURCES = {
    "own_etf_flow",
    "constituent_etf_flow_exposure",
    "none",
}
TASK_TYPES = {
    "etf_own_flow_analysis",
    "stock_constituent_flow_analysis",
    "all_stock_control_analysis",
}
DETERMINISTIC_FACT_PRECISION = {
    "etf_flow_to_constituent.net_weighted_flow_rate_contribution_pct": 6,
}
DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
TRADE_DIRECTIVE_PATTERN = re.compile(
    r"(?:매수|매도|진입|청산|주문)\s*(?:하|해|해야|추천|신호)|"
    r"\b(?:buy|sell|enter|exit)\s+(?:now|signal|the position)\b",
    flags=re.IGNORECASE,
)


class ModelGateError(RuntimeError):
    """Raised before inference when the released adapter is not trustworthy."""


class InferenceError(RuntimeError):
    """Raised when the one authorized model endpoint cannot answer."""


class ModelResponseParseError(InferenceError):
    """Raised when the endpoint answered but its assistant content is not JSON."""


class ResponseContractError(RuntimeError):
    """Raised when a model response mutates facts or violates analysis scope."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contract_repair_instruction(
    expected_response: Mapping[str, Any], contract_error: str
) -> str:
    """Build the one allowed repair turn from deterministic, already-known facts."""

    facts = expected_response.get("facts")
    if not isinstance(facts, Mapping):
        raise ResponseContractError(
            "expected response has no deterministic facts for contract repair"
        )
    interpretation = expected_response.get("interpretation")
    if not isinstance(interpretation, Mapping):
        raise ResponseContractError(
            "expected response has no interpretation contract for repair"
        )
    required_interpretation = {
        "scope": interpretation.get("scope"),
        "task_type": interpretation.get("task_type"),
        "price_signal": interpretation.get("price_signal"),
        "etf_flow_signal": interpretation.get("etf_flow_signal"),
        "etf_flow_signal_source": interpretation.get(
            "etf_flow_signal_source"
        ),
        "relationship": interpretation.get("relationship"),
        "regime": expected_response.get("regime"),
        "allowed_price_signal_values": sorted(SIGNALS),
        "allowed_etf_flow_signal_values": sorted(SIGNALS),
        "allowed_regime_values": sorted(REGIMES),
    }
    if (
        required_interpretation["scope"]
        != "data_interpretation_not_trade_execution"
        or required_interpretation["task_type"] not in TASK_TYPES
        or required_interpretation["price_signal"] not in SIGNALS
        or required_interpretation["etf_flow_signal"] not in SIGNALS
        or required_interpretation["etf_flow_signal_source"]
        not in FLOW_SIGNAL_SOURCES
        or required_interpretation["relationship"] not in REGIMES
        or required_interpretation["regime"] not in REGIMES
    ):
        raise ResponseContractError(
            "expected response has invalid deterministic interpretation fields"
        )
    interpretation_clause = (
        "아래 REQUIRED_INTERPRETATION_CONTRACT_JSON의 scope, task_type, "
        "price_signal, etf_flow_signal, etf_flow_signal_source, relationship, "
        "regime은 정확히 그대로 사용하라. allowed 목록은 형식 검증용이며 "
        "다른 허용값으로 바꾸면 안 된다.\n"
        "REQUIRED_INTERPRETATION_CONTRACT_JSON="
        f"{canonical_json(required_interpretation)}\n"
    )
    if "fields do not match the contract" not in contract_error:
        return (
            "이전 응답은 계약 위반이다. 입력 시점 이후 날짜를 만들거나 결정론적 "
            "facts를 변경하면 안 된다. 아래 DETERMINISTIC_FACTS_JSON을 facts "
            "값으로 정확히 사용하고, 이전 응답과 동일한 7개 최상위 필드 구조로 "
            "JSON 객체만 다시 출력하라. 해석은 입력 시점에 이용 가능한 증거만 "
            "사용하고 매매 지시를 포함하지 마라.\n"
            f"DETERMINISTIC_FACTS_JSON={canonical_json(dict(facts))}\n"
            f"{interpretation_clause}"
            "/no_think"
        )
    return (
        "이전 응답은 계약 위반이다. 입력 시점 이후 날짜를 만들거나 결정론적 "
        "facts를 변경하면 안 된다. 아래 DETERMINISTIC_FACTS_JSON을 facts 값으로 "
        "정확히 사용하라. 최상위 필드는 ALLOWED_TOP_LEVEL_KEYS_JSON의 7개를 "
        "각각 정확히 한 번만 사용하고 그 외 필드는 절대 출력하지 마라. JSON "
        "객체만 다시 출력하고, 해석은 입력 시점에 이용 가능한 증거만 사용하며 "
        "매매 지시를 포함하지 마라.\n"
        f"ALLOWED_TOP_LEVEL_KEYS_JSON={canonical_json(list(REQUIRED_RESPONSE_KEYS))}\n"
        f"DETERMINISTIC_FACTS_JSON={canonical_json(dict(facts))}\n"
        f"{interpretation_clause}"
        "/no_think"
    )


def canonicalize_deterministic_facts(value: Any) -> Any:
    """Apply only the numeric precision declared by the deterministic contract."""

    if not isinstance(value, Mapping):
        return value
    normalized = copy.deepcopy(dict(value))
    exposure = normalized.get("etf_flow_to_constituent")
    if not isinstance(exposure, Mapping):
        return normalized
    exposure = dict(exposure)
    normalized["etf_flow_to_constituent"] = exposure
    field = "net_weighted_flow_rate_contribution_pct"
    raw = exposure.get(field)
    if (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and math.isfinite(float(raw))
    ):
        exposure[field] = round(
            float(raw),
            DETERMINISTIC_FACT_PRECISION[
                "etf_flow_to_constituent."
                "net_weighted_flow_rate_contribution_pct"
            ],
        )
    return normalized


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ModelGateError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelGateError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ModelGateError(f"{label} must be one JSON object: {path}")
    return value


def _resolve_release_path(release_path: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ModelGateError(f"{label} path is missing from the release manifest")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = release_path.parent / candidate
    return candidate.resolve()


def _verify_bound_file(
    release_path: Path, value: Mapping[str, Any], label: str
) -> tuple[Path, str]:
    path = _resolve_release_path(release_path, value.get("path"), label)
    expected = str(value.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ModelGateError(f"{label} SHA256 is missing or invalid")
    if not path.is_file():
        raise ModelGateError(f"{label} file is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ModelGateError(
            f"{label} SHA256 mismatch: expected={expected} observed={observed} path={path}"
        )
    return path, observed


@dataclass(frozen=True)
class ModelRelease:
    manifest_path: Path
    manifest_sha256: str
    model_id: str
    endpoint_model: str
    base_model: str
    adapter_root: Path
    adapter_set_sha256: str
    dataset_manifest_sha256: str
    evaluation_sha256: str
    raw: dict[str, Any]

    def public_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": RELEASE_SCHEMA,
            "status": "accepted",
            "model_id": self.model_id,
            "endpoint_model": self.endpoint_model,
            "base_model": self.base_model,
            "adapter_set_sha256": self.adapter_set_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "release_manifest_sha256": self.manifest_sha256,
        }


def load_model_release(path: Path) -> ModelRelease:
    """Verify the frozen dataset, evaluation, and every adapter artifact."""

    release_path = Path(path).expanduser().resolve()
    value = _read_object(release_path, "trained-model release manifest")
    if value.get("schema_version") != RELEASE_SCHEMA:
        raise ModelGateError(
            f"unsupported release schema: {value.get('schema_version')!r}"
        )
    if value.get("status") != "accepted":
        raise ModelGateError(
            f"trained-model release is not accepted: {value.get('status')!r}"
        )
    model_id = str(value.get("model_id") or "").strip()
    endpoint_model = str(value.get("endpoint_model") or "").strip()
    base_model = str(value.get("base_model") or "").strip()
    if not model_id or not endpoint_model or not base_model:
        raise ModelGateError("release model_id, endpoint_model, and base_model are required")

    adapter_root = _resolve_release_path(
        release_path, value.get("adapter_root"), "adapter_root"
    )
    if not adapter_root.is_dir():
        raise ModelGateError(f"released adapter directory is missing: {adapter_root}")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ModelGateError("release manifest has no adapter artifacts")
    verified_artifacts: list[tuple[str, str]] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise ModelGateError(f"adapter artifact {index} is not an object")
        artifact_path, digest = _verify_bound_file(
            release_path, item, f"adapter artifact {index}"
        )
        try:
            relative = artifact_path.relative_to(adapter_root)
        except ValueError as exc:
            raise ModelGateError(
                f"adapter artifact escapes adapter_root: {artifact_path}"
            ) from exc
        verified_artifacts.append((relative.as_posix(), digest))

    dataset = value.get("dataset_manifest")
    evaluation = value.get("evaluation")
    if not isinstance(dataset, dict) or not isinstance(evaluation, dict):
        raise ModelGateError("release must bind dataset_manifest and evaluation")
    _, dataset_sha = _verify_bound_file(release_path, dataset, "dataset manifest")
    evaluation_path, evaluation_sha = _verify_bound_file(
        release_path, evaluation, "evaluation report"
    )
    evaluation_value = _read_object(evaluation_path, "evaluation report")
    if evaluation_value.get("status") != "green":
        raise ModelGateError(
            f"evaluation report is not green: {evaluation_value.get('status')!r}"
        )
    if int(evaluation_value.get("prohibited_violation_count", -1)) != 0:
        raise ModelGateError("evaluation report contains prohibited violations")
    required_gates = evaluation_value.get("required_gates")
    if not isinstance(required_gates, dict) or not required_gates:
        raise ModelGateError("evaluation report has no required_gates")
    failed_gates = sorted(key for key, passed in required_gates.items() if passed is not True)
    if failed_gates:
        raise ModelGateError(f"evaluation gates are not green: {failed_gates}")
    evaluation_inputs = value.get("evaluation_inputs")
    if not isinstance(evaluation_inputs, dict):
        raise ModelGateError("release has no bound evaluation_inputs")
    for key in ("frozen_test", "predictions"):
        item = evaluation_inputs.get(key)
        if not isinstance(item, dict):
            raise ModelGateError(f"release has no bound evaluation input: {key}")
        _, bound_sha = _verify_bound_file(
            release_path, item, f"evaluation input {key}"
        )
        evaluated_item = evaluation_value.get(key) or {}
        if evaluated_item.get("sha256") != bound_sha:
            raise ModelGateError(f"evaluation input {key} does not match its report")

    adapter_set_sha = sha256_bytes(
        canonical_json(sorted(verified_artifacts)).encode("utf-8")
    )
    declared_adapter_set_sha = str(value.get("adapter_set_sha256") or "")
    if declared_adapter_set_sha != adapter_set_sha:
        raise ModelGateError(
            "adapter_set_sha256 does not match the verified released artifacts"
        )
    return ModelRelease(
        manifest_path=release_path,
        manifest_sha256=sha256_file(release_path),
        model_id=model_id,
        endpoint_model=endpoint_model,
        base_model=base_model,
        adapter_root=adapter_root,
        adapter_set_sha256=adapter_set_sha,
        dataset_manifest_sha256=dataset_sha,
        evaluation_sha256=evaluation_sha,
        raw=value,
    )


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ModelResponseParseError(
        "trained model response did not contain one valid JSON object"
    )


Transport = Callable[[dict[str, Any], Mapping[str, str], int], dict[str, Any]]


class TrainedQuantClient:
    """OpenAI-compatible client bound to exactly one accepted LoRA release."""

    def __init__(
        self,
        *,
        endpoint: str,
        release: ModelRelease,
        token: str | None = None,
        timeout: int = 180,
        transport: Transport | None = None,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ModelGateError("trained-model endpoint must be an absolute HTTP URL")
        self.endpoint = endpoint
        self.release = release
        self.token = token
        self.timeout = timeout
        self.transport = transport or self._http_transport

    def _http_transport(
        self, payload: dict[str, Any], headers: Mapping[str, str], timeout: int
    ) -> dict[str, Any]:
        body = canonical_json(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={**dict(headers), "content-length": str(len(body))},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            suffix = f": {detail[:2000]}" if detail else ""
            raise InferenceError(
                f"trained-model endpoint request failed: HTTP {exc.code}{suffix}"
            ) from exc
        except urllib.error.URLError as exc:
            raise InferenceError(f"trained-model endpoint request failed: {exc}") from exc
        if status != 200:
            raise InferenceError(f"trained-model endpoint returned HTTP {status}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InferenceError("trained-model endpoint returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise InferenceError("trained-model endpoint returned a non-object response")
        return value

    def _complete_messages_raw(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int = 1400,
        response_schema: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.release.endpoint_model,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
            "seed": 1111,
            "max_tokens": max_tokens,
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "quant_ai_radar_response",
                        "schema": dict(response_schema),
                    },
                }
                if response_schema is not None
                else {"type": "json_object"}
            ),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {
            "content-type": "application/json",
            "user-agent": "quant-ai-radar/1.0",
        }
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        body = self.transport(payload, headers, self.timeout)
        returned_model = str(body.get("model") or "")
        if returned_model != self.release.endpoint_model:
            raise InferenceError(
                "endpoint served a different model: "
                f"expected={self.release.endpoint_model!r} got={returned_model!r}"
            )
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError("trained-model endpoint response has no assistant content") from exc
        if not isinstance(content, str):
            raise InferenceError("trained-model assistant content is not text")
        trace = {
            "endpoint_model": returned_model,
            "request_sha256": sha256_bytes(canonical_json(payload).encode("utf-8")),
            "response_sha256": sha256_bytes(content.encode("utf-8")),
            "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
            "usage": body.get("usage"),
        }
        return content, trace

    def complete_messages(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int = 1400,
        response_schema: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        content, trace = self._complete_messages_raw(
            messages=messages,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )
        try:
            return parse_json_object(content), trace
        except ModelResponseParseError as exc:
            exc.trace = trace
            exc.raw_content = content
            raise

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1400,
        response_schema: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.complete_messages(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

    def complete_validated(
        self,
        *,
        system: str,
        user: str,
        expected_response: Mapping[str, Any],
        max_tokens: int = 1400,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the trained prompt, then allow one explicit contract-repair turn."""

        initial_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        content, initial_trace = self._complete_messages_raw(
            messages=initial_messages, max_tokens=max_tokens
        )
        initial: dict[str, Any] | None = None
        try:
            initial = parse_json_object(content)
            validated = validate_symbol_judgement(initial, expected_response)
        except (InferenceError, ResponseContractError) as exc:
            repair = contract_repair_instruction(expected_response, str(exc))
            repaired_content, final_trace = self._complete_messages_raw(
                messages=[
                    *initial_messages,
                    {
                        "role": "assistant",
                        "content": (
                            canonical_json(initial)
                            if initial is not None
                            else content
                        ),
                    },
                    {"role": "user", "content": repair},
                ],
                max_tokens=max_tokens,
            )
            repaired = parse_json_object(repaired_content)
            validated = validate_symbol_judgement(repaired, expected_response)
            return validated, {
                **final_trace,
                "contract_attempts": 2,
                "contract_repair_applied": True,
                "initial_request_sha256": initial_trace["request_sha256"],
                "initial_response_sha256": initial_trace["response_sha256"],
                "initial_contract_error": f"{type(exc).__name__}: {exc}",
            }
        return validated, {
            **initial_trace,
            "contract_attempts": 1,
            "contract_repair_applied": False,
        }


def _future_dates(value: Any, as_of_date: str) -> list[str]:
    found = set(DATE_PATTERN.findall(canonical_json(value)))
    as_of = date.fromisoformat(as_of_date)
    return sorted(item for item in found if date.fromisoformat(item) > as_of)


def judgement_prohibited_violations(
    value: Mapping[str, Any], as_of_date: str
) -> list[str]:
    """Return the release-blocking lookahead/scope violations in a judgement."""

    violations = [f"post_as_of_date:{item}" for item in _future_dates(value, as_of_date)]
    if TRADE_DIRECTIVE_PATTERN.search(canonical_json(value)):
        violations.append("trade_directive")
    return violations


def validate_symbol_judgement(
    value: Mapping[str, Any], expected_response: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep model interpretation flexible while binding every supplied fact."""

    missing = [key for key in REQUIRED_RESPONSE_KEYS if key not in value]
    unexpected = [key for key in value if key not in REQUIRED_RESPONSE_KEYS]
    if missing or unexpected:
        raise ResponseContractError(
            "symbol judgement fields do not match the contract: "
            f"missing={missing} unexpected={unexpected}"
        )
    observed_facts = canonicalize_deterministic_facts(value.get("facts"))
    expected_facts = canonicalize_deterministic_facts(
        expected_response.get("facts")
    )
    if observed_facts != expected_facts:
        raise ResponseContractError("trained model changed deterministic facts")
    interpretation = value.get("interpretation")
    expected_interpretation = expected_response.get("interpretation") or {}
    if not isinstance(interpretation, dict):
        raise ResponseContractError("symbol interpretation must be an object")
    if interpretation.get("scope") != "data_interpretation_not_trade_execution":
        raise ResponseContractError("symbol judgement escaped the analysis-only scope")
    if interpretation.get("task_type") != expected_interpretation.get("task_type"):
        raise ResponseContractError("symbol judgement changed the deterministic task type")
    if interpretation.get("price_signal") != expected_interpretation.get(
        "price_signal"
    ):
        raise ResponseContractError(
            "symbol judgement changed the deterministic price signal"
        )
    if interpretation.get("etf_flow_signal") != expected_interpretation.get(
        "etf_flow_signal"
    ):
        raise ResponseContractError(
            "symbol judgement changed the deterministic ETF-flow signal"
        )
    if interpretation.get("etf_flow_signal_source") != expected_interpretation.get(
        "etf_flow_signal_source"
    ):
        raise ResponseContractError(
            "symbol judgement changed the deterministic ETF-flow source"
        )
    if interpretation.get("price_signal") not in SIGNALS:
        raise ResponseContractError("symbol judgement has an invalid price signal")
    if interpretation.get("etf_flow_signal") not in SIGNALS:
        raise ResponseContractError("symbol judgement has an invalid ETF-flow signal")
    if interpretation.get("etf_flow_signal_source") not in FLOW_SIGNAL_SOURCES:
        raise ResponseContractError("symbol judgement has an invalid ETF-flow source")
    if interpretation.get("task_type") not in TASK_TYPES:
        raise ResponseContractError("symbol judgement has an invalid task type")
    if value.get("regime") not in REGIMES:
        raise ResponseContractError("symbol judgement has an invalid regime")
    if value.get("regime") != expected_response.get("regime"):
        raise ResponseContractError(
            "symbol judgement changed the deterministic regime"
        )
    if interpretation.get("relationship") != value.get("regime"):
        raise ResponseContractError(
            "symbol interpretation relationship does not match regime"
        )
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ResponseContractError("symbol judgement confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise ResponseContractError("symbol judgement confidence is outside [0,1]")
    if not isinstance(value.get("counter_evidence"), list) or not all(
        isinstance(item, str) for item in value["counter_evidence"]
    ):
        raise ResponseContractError("counter_evidence must be a string array")
    if not isinstance(value.get("unknowns"), list) or not all(
        isinstance(item, str) for item in value["unknowns"]
    ):
        raise ResponseContractError("unknowns must be a string array")
    if not isinstance(value.get("conclusion"), str) or not value["conclusion"].strip():
        raise ResponseContractError("symbol judgement conclusion is empty")
    as_of_date = str((expected_response.get("facts") or {}).get("as_of_date") or "")
    violations = judgement_prohibited_violations(value, as_of_date)
    future_dates = [item.split(":", 1)[1] for item in violations if item.startswith("post_as_of_date:")]
    if future_dates:
        raise ResponseContractError(
            f"symbol judgement contains post-as-of dates: {future_dates}"
        )
    if "trade_directive" in violations:
        raise ResponseContractError("symbol judgement contains a trade directive")
    normalized = dict(value)
    normalized["facts"] = expected_facts
    return normalized
