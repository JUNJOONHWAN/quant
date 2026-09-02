#!/usr/bin/env python3
"""Standalone CLI and Hermes independent-app entrypoint for Quant AI Radar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from quant_dataset.shared_market import (  # noqa: E402
    DEFAULT_BASE_DATABASE,
    DEFAULT_INCREMENTAL_DATABASE,
    DEFAULT_ORACLE_STATUS,
    load_shared_market_binding,
)
from workflows.quant_ai_radar.model_runtime import load_model_release  # noqa: E402
from workflows.quant_ai_radar.forecast_synthesis import (  # noqa: E402
    DEFAULT_FORECAST_ENDPOINT,
    DEFAULT_FORECAST_MODEL,
)
from workflows.quant_ai_radar.historical_analog import (  # noqa: E402
    DEFAULT_ANALOG_INDEX,
    DEFAULT_EXAMPLE_DATABASE,
    DEFAULT_PRICE_DATABASE,
)
from workflows.quant_ai_radar.decision_support import (  # noqa: E402
    QUALITY_SCHEMA_VERSION,
)
from workflows.quant_ai_radar.email_delivery import (  # noqa: E402
    deliver_daily_report,
    email_delivery_status,
    email_transport_status,
)
from workflows.quant_ai_radar.relation_index import (  # noqa: E402
    DEFAULT_RELATION_INDEX,
)
from workflows.quant_ai_radar.run_daily_cycle import build_stage_commands  # noqa: E402
from workflows.quant_ai_radar.universe import write_json  # noqa: E402
from workflows.market_structure_oracle.incremental_store import (  # noqa: E402
    latest_closed_nyse_session,
)


KST = ZoneInfo("Asia/Seoul")
APP_ID = "quant-ai-radar"
DEFAULT_ENV_FILE = Path("/home/zooh/.config/quant/quant-ai-radar.env")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR"
)
DEFAULT_RELEASE_MANIFEST = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/"
    "Qwen3-8B-FLOW/release_manifest.json"
)
APP_STATE_PATH = DEFAULT_OUTPUT_ROOT / "status" / "app_cli.json"
REQUEST_FIELDS = frozenset(
    {
        "action",
        "question",
        "query",
        "symbols",
        "symbols_file",
        "shadow",
        "workers",
        "max_ai_etfs",
        "max_ai_stocks",
        "smoke_max_items",
    }
)
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")
QUESTION_TICKER_PATTERN = re.compile(
    r"(?<![A-Z0-9.\-])([A-Z][A-Z0-9.\-]{0,14})(?![A-Z0-9.\-])"
)
QUESTION_TICKER_STOPWORDS = frozenset(
    {"AI", "ETF", "FLOW", "RADAR", "US", "KST", "FMP", "MASSIVE", "QWEN"}
)


class AppCliError(RuntimeError):
    """The standalone application cannot safely complete the request."""


def _now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _generated_today_kst(report: dict[str, Any]) -> bool:
    generated_at = str(report.get("generated_at_kst") or "")
    return generated_at[:10] == datetime.now(KST).date().isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def load_env_file(path: Path, *, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Load simple KEY=VALUE settings without executing shell content."""
    target = os.environ if environ is None else environ
    if not path.expanduser().is_file():
        raise AppCliError(f"Quant AI Radar environment file is missing: {path}")
    loaded: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.expanduser().read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise AppCliError(
                f"invalid environment line {line_number} in {path}"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", key):
            raise AppCliError(
                f"invalid environment key {key!r} at {path}:{line_number}"
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key not in target:
            target[key] = value
        loaded[key] = target[key]
    return loaded


def load_sealed_request(path: Path | None = None) -> dict[str, Any] | None:
    """Load and verify the immutable App Manager request, when present."""
    request_path = path
    if request_path is None:
        raw = os.environ.get("OPERATIONS_APP_INPUT_FILE", "").strip()
        request_path = Path(raw) if raw else None
    if request_path is None:
        return None
    if not request_path.is_file():
        raise AppCliError(f"application request file is missing: {request_path}")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AppCliError(f"invalid application request JSON: {exc}") from exc
    if not isinstance(request, dict):
        raise AppCliError("application request must be one JSON object")
    unknown = sorted(set(request) - REQUEST_FIELDS)
    if unknown:
        raise AppCliError(f"unsupported application request fields: {unknown}")
    expected = os.environ.get("OPERATIONS_APP_INPUT_SHA256", "").strip()
    if expected:
        actual = hashlib.sha256(_canonical_bytes(request)).hexdigest()
        if actual != expected:
            raise AppCliError("application request SHA-256 mismatch")
    return request


def normalize_symbols(values: Any) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raise AppCliError("symbols must be a string or list of strings")
    symbols: list[str] = []
    for value in raw_values:
        if not isinstance(value, str):
            raise AppCliError("every symbol must be a string")
        for piece in value.split(","):
            symbol = piece.strip().upper()
            if not SYMBOL_PATTERN.fullmatch(symbol):
                raise AppCliError(f"invalid symbol: {piece!r}")
            if symbol not in symbols:
                symbols.append(symbol)
    if not symbols:
        raise AppCliError("at least one symbol is required")
    return symbols


def load_symbols_file(value: Any) -> tuple[list[str], dict[str, Any]]:
    """Load one complete symbol universe and seal its source provenance."""
    raw_path = str(value or "").strip()
    if not raw_path:
        raise AppCliError("symbols_file must be a non-empty path")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise AppCliError(f"symbols_file is missing: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AppCliError(f"unable to read symbols_file {path}: {exc}") from exc
    if len(raw) > 1024 * 1024:
        raise AppCliError("symbols_file exceeds the 1 MiB safety limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AppCliError(f"invalid symbols_file JSON: {path}: {exc}") from exc
    values = payload.get("symbols") if isinstance(payload, dict) else payload
    symbols = normalize_symbols(values)
    return symbols, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "symbol_count": len(symbols),
    }


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise AppCliError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AppCliError(f"{field} must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        raise AppCliError(f"{field} must be >= {minimum}")
    return parsed


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise AppCliError(f"{field} must be a JSON boolean")
    return value


def _provided_or_default(
    source: dict[str, Any], field: str, default: Any
) -> Any:
    value = source.get(field)
    return default if value is None else value


def _token_file() -> str:
    return os.environ.get("QUANT_AI_MODEL_TOKEN_FILE", "").strip()


def _release_manifest() -> str:
    return os.environ.get(
        "QUANT_AI_RELEASE_MANIFEST", str(DEFAULT_RELEASE_MANIFEST)
    ).strip()


def _model_endpoint() -> str:
    endpoint = os.environ.get("QUANT_AI_MODEL_ENDPOINT", "").strip()
    if not endpoint:
        raise AppCliError("QUANT_AI_MODEL_ENDPOINT is missing")
    return endpoint


def _forecast_model_endpoint() -> str:
    return os.environ.get(
        "QUANT_AI_FORECAST_MODEL_ENDPOINT", DEFAULT_FORECAST_ENDPOINT
    ).strip()


def _forecast_model_name() -> str:
    return os.environ.get(
        "QUANT_AI_FORECAST_MODEL_NAME", DEFAULT_FORECAST_MODEL
    ).strip()


def build_daily_commands(
    *,
    shadow: bool,
    workers: int,
    max_ai_etfs: int,
    max_ai_stocks: int,
    smoke_max_items: int = 0,
) -> list[list[str]]:
    commands = build_stage_commands(
        model_endpoint=_model_endpoint(),
        release_manifest=_release_manifest(),
        workers=str(workers),
        timeout_seconds=os.environ.get(
            "QUANT_AI_MODEL_TIMEOUT_SECONDS", "360"
        ),
        token_file=_token_file(),
        max_constituent_available_lag_days=os.environ.get(
            "QUANT_AI_MAX_CONSTITUENT_AVAILABLE_LAG_DAYS", "45"
        ),
        max_ai_etfs=str(max_ai_etfs),
        max_ai_stocks=str(max_ai_stocks),
    )
    if shadow:
        commands[1].append("--shadow")
    if smoke_max_items:
        commands[1].extend(["--smoke-max-items", str(smoke_max_items)])
    return commands


def build_analyze_command(symbols: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "workflows.quant_ai_radar.analyze_on_demand",
        *symbols,
        "--release-manifest",
        _release_manifest(),
        "--model-endpoint",
        _model_endpoint(),
        "--forecast-model-endpoint",
        _forecast_model_endpoint(),
        "--forecast-model-name",
        _forecast_model_name(),
        "--forecast-timeout",
        os.environ.get("QUANT_AI_FORECAST_MODEL_TIMEOUT_SECONDS", "240"),
        "--analog-example-database",
        os.environ.get(
            "QUANT_AI_ANALOG_EXAMPLE_DATABASE", str(DEFAULT_EXAMPLE_DATABASE)
        ),
        "--analog-price-database",
        os.environ.get(
            "QUANT_AI_ANALOG_PRICE_DATABASE", str(DEFAULT_PRICE_DATABASE)
        ),
        "--analog-index-database",
        os.environ.get(
            "QUANT_AI_ANALOG_INDEX_DATABASE", str(DEFAULT_ANALOG_INDEX)
        ),
        "--analog-neighbor-limit",
        os.environ.get("QUANT_AI_ANALOG_NEIGHBOR_LIMIT", "80"),
        "--analog-per-symbol-limit",
        os.environ.get("QUANT_AI_ANALOG_PER_SYMBOL_LIMIT", "2"),
        "--max-constituent-available-lag-days",
        os.environ.get(
            "QUANT_AI_MAX_CONSTITUENT_AVAILABLE_LAG_DAYS", "45"
        ),
    ]
    token_file = _token_file()
    if token_file:
        command.extend(["--model-token-file", token_file])
    forecast_token_file = os.environ.get(
        "QUANT_AI_FORECAST_MODEL_TOKEN_FILE", ""
    ).strip()
    if forecast_token_file:
        command.extend(
            ["--forecast-model-token-file", forecast_token_file]
        )
    return command


def _last_json(stdout: str, *, command: list[str]) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AppCliError(
        f"stage returned no JSON object: {' '.join(command)}"
    )


def run_json_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=QUANT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = _last_json(completed.stdout, command=command)
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[-1000:]
        raise AppCliError(
            f"stage failed rc={completed.returncode}: {' '.join(command)}; "
            f"result={json.dumps(payload, ensure_ascii=False, sort_keys=True)}; "
            f"stderr={detail}"
        )
    return payload


def daily_completion_status(
    *,
    expected_source_fingerprint_sha256: str,
) -> dict[str, Any] | None:
    """Return only a report and email bound to the current Oracle snapshot."""

    oracle = _read_json(DEFAULT_ORACLE_STATUS) or {}
    target = str(oracle.get("target_as_of_date") or "")[:10]
    latest_path = DEFAULT_OUTPUT_ROOT / "status" / "latest.json"
    run_dir = DEFAULT_OUTPUT_ROOT / "runs" / target
    report_path = run_dir / "market_report.json"
    report = _read_json(report_path) or {}
    latest = _read_json(latest_path) or {}
    source_status = (
        report.get("source_status")
        if isinstance(report.get("source_status"), dict)
        else {}
    )
    shared_oracle = (
        source_status.get("shared_oracle_store")
        if isinstance(source_status.get("shared_oracle_store"), dict)
        else {}
    )
    report_source_fingerprint = str(
        shared_oracle.get("source_fingerprint_sha256") or ""
    )
    quality = (
        report.get("quality_audit")
        if isinstance(report.get("quality_audit"), dict)
        else {}
    )
    scores = (
        quality.get("scores")
        if isinstance(quality.get("scores"), dict)
        else {}
    )
    model = (
        report.get("model_release")
        if isinstance(report.get("model_release"), dict)
        else {}
    )
    merged = (
        model.get("merged_model")
        if isinstance(model.get("merged_model"), dict)
        else {}
    )
    report_sha256 = (
        hashlib.sha256(report_path.read_bytes()).hexdigest()
        if report_path.is_file()
        else ""
    )
    latest_sha256 = (
        hashlib.sha256(latest_path.read_bytes()).hexdigest()
        if latest_path.is_file()
        else ""
    )
    email = (
        email_delivery_status(
            target,
            report_sha256=report_sha256,
            source_fingerprint_sha256=expected_source_fingerprint_sha256,
        )
        if target and report_sha256
        else {}
    )
    complete = bool(
        target
        and str(report.get("as_of_date") or "") == target
        and str(latest.get("as_of_date") or "") == target
        and report_sha256
        and latest_sha256 == report_sha256
        and report_source_fingerprint == expected_source_fingerprint_sha256
        and _generated_today_kst(report)
        and report.get("schema_version") == "quant.ai_radar_report.v2"
        and report.get("deployment_mode") == "reference_publish"
        and report.get("selected_model_scope_complete") is True
        and quality.get("schema_version") == QUALITY_SCHEMA_VERSION
        and quality.get("status") == "green"
        and quality.get("publishable_reference_report") is True
        and scores
        and all(
            isinstance(value, (int, float)) and float(value) >= 8.0
            for value in scores.values()
        )
        and model.get("status") == "accepted"
        and model.get("model_id") == "Qwen3-8B-FLOW"
        and model.get("endpoint_model") == "Qwen3-8B-FLOW"
        and merged.get("precision") == "bfloat16"
        and str(merged.get("manifest_sha256") or "")
        and str(merged.get("content_sha256") or "")
        and email.get("complete") is True
    )
    if not complete:
        return None
    if not all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            report_path,
            run_dir / "market_report.html",
            run_dir / "market_report_email_420.html",
        )
    ):
        return None
    return {
        "as_of_date": target,
        "run_dir": str(run_dir),
        "report": str(report_path),
        "source_fingerprint_sha256": report_source_fingerprint,
        "queue_counts": report.get("queue_counts"),
        "email_delivery": email,
    }


def run_daily(
    *,
    shadow: bool,
    workers: int,
    max_ai_etfs: int,
    max_ai_stocks: int,
    smoke_max_items: int = 0,
) -> dict[str, Any]:
    production_run = not shadow and not smoke_max_items
    commands = build_daily_commands(
        shadow=shadow,
        workers=workers,
        max_ai_etfs=max_ai_etfs,
        max_ai_stocks=max_ai_stocks,
        smoke_max_items=smoke_max_items,
    )
    state = {
        "schema_version": "quant.ai_radar_app_run.v1",
        "status": "running_shared_oracle_store_prepare",
        "app_id": APP_ID,
        "action": "daily",
        "shadow": shadow,
        "smoke_max_items": smoke_max_items,
        "operations_app_run_id": os.environ.get("OPERATIONS_APP_RUN_ID"),
        "started_at_kst": _now_kst(),
    }
    write_json(APP_STATE_PATH, state)
    prepare = run_json_command(commands[0])
    prepared_target = str(prepare.get("target_as_of_date") or "")[:10]
    if production_run:
        prepare_binding = (
            prepare.get("binding")
            if isinstance(prepare.get("binding"), dict)
            else {}
        )
        expected_source_fingerprint = str(
            prepare_binding.get("source_fingerprint_sha256") or ""
        )
        if not re.fullmatch(r"[0-9a-f]{64}", expected_source_fingerprint):
            raise AppCliError(
                "Oracle prepare did not return a valid source fingerprint"
            )
        closed_target = latest_closed_nyse_session(
            publish_grace_hour_et=16
        )
        if prepared_target != closed_target:
            raise AppCliError(
                "Oracle target is not yet vendor-publishable for the latest "
                f"closed NYSE session: oracle={prepared_target or 'missing'} "
                f"closed_session={closed_target}"
            )
        completed = daily_completion_status(
            expected_source_fingerprint_sha256=(
                expected_source_fingerprint
            )
        )
        if completed is not None:
            result = {
                **state,
                "status": "PASS",
                "prepare": prepare,
                "engine_status": "already_complete",
                "generation_skipped": True,
                "skip_reason": (
                    "current_oracle_target_generated_today_green_and_"
                    "email_v3_complete"
                ),
                "as_of_date": completed["as_of_date"],
                "run_dir": completed["run_dir"],
                "report": completed["report"],
                "production_latest_published": True,
                "queue_counts": completed.get("queue_counts"),
                "email_delivery": completed["email_delivery"],
                "completed_at_kst": _now_kst(),
            }
            write_json(APP_STATE_PATH, result)
            return result
    state["status"] = "running_full_scan_prioritized_inference"
    state["prepare"] = prepare
    write_json(APP_STATE_PATH, state)
    radar = run_json_command(commands[1])
    success = {
        "complete",
        "shadow_complete_not_published",
        "smoke_complete_not_publishable",
    }
    if str(radar.get("status")) not in success:
        raise AppCliError(
            "Radar engine did not reach a complete state: "
            f"{radar.get('status')}"
        )
    if production_run:
        if (
            radar.get("status") != "complete"
            or not radar.get("production_latest_published")
            or not radar.get("report")
        ):
            raise AppCliError(
                "production Radar did not publish an accepted report"
            )
        email_delivery = deliver_daily_report(Path(str(radar["report"])))
        if not email_delivery.get("complete"):
            raise AppCliError(
                "production email final gate did not complete: "
                f"{email_delivery.get('status')}"
            )
    else:
        email_delivery = {
            "status": "NOT_REQUIRED",
            "complete": True,
            "reason": "shadow_or_smoke_run",
        }
    result = {
        **state,
        "status": "PASS",
        "engine_status": radar["status"],
        "as_of_date": radar.get("as_of_date"),
        "run_dir": radar.get("run_dir"),
        "report": radar.get("report"),
        "production_latest_published": radar.get(
            "production_latest_published", False
        ),
        "queue_counts": radar.get("queue_counts"),
        "email_delivery": email_delivery,
        "completed_at_kst": _now_kst(),
    }
    write_json(APP_STATE_PATH, result)
    return result


def run_analysis(
    symbols: list[str],
    *,
    symbols_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested_symbols = normalize_symbols(symbols)
    state = {
        "schema_version": "quant.ai_radar_app_run.v1",
        "status": "running_on_demand_analysis",
        "app_id": APP_ID,
        "action": "analyze",
        "symbols": requested_symbols,
        "requested_symbols": requested_symbols,
        "requested_symbol_count": len(requested_symbols),
        "symbols_source": symbols_source,
        "operations_app_run_id": os.environ.get("OPERATIONS_APP_RUN_ID"),
        "started_at_kst": _now_kst(),
    }
    write_json(APP_STATE_PATH, state)
    receipt = run_json_command(build_analyze_command(requested_symbols))
    if receipt.get("status") != "complete":
        raise AppCliError(
            f"on-demand analysis did not fully complete: {receipt.get('status')}"
        )
    engine_requested = receipt.get("requested_symbols")
    if not isinstance(engine_requested, list):
        raise AppCliError("on-demand receipt has no requested_symbols coverage")
    engine_requested = normalize_symbols(engine_requested)
    if engine_requested != requested_symbols:
        raise AppCliError(
            "on-demand requested_symbols coverage mismatch: "
            f"expected={requested_symbols}, actual={engine_requested}"
        )
    rows = receipt.get("results")
    if not isinstance(rows, list):
        raise AppCliError("on-demand receipt results must be a list")
    completed_symbols: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "complete":
            raise AppCliError("on-demand receipt contains an incomplete result")
        symbol = str(row.get("symbol") or "").strip().upper()
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise AppCliError("on-demand receipt contains an invalid result symbol")
        if symbol in completed_symbols:
            raise AppCliError(f"on-demand receipt duplicates symbol: {symbol}")
        completed_symbols.append(symbol)
    if completed_symbols != requested_symbols:
        raise AppCliError(
            "on-demand completed_symbols coverage mismatch: "
            f"expected={requested_symbols}, actual={completed_symbols}"
        )
    result = {
        **state,
        "status": "PASS",
        "engine_status": receipt["status"],
        "as_of_date": receipt.get("as_of_date"),
        "requested_symbols": requested_symbols,
        "completed_symbols": completed_symbols,
        "requested_symbol_count": len(requested_symbols),
        "completed_symbol_count": len(completed_symbols),
        "coverage_complete": True,
        "symbols_source": symbols_source,
        "results": rows,
        "completed_at_kst": _now_kst(),
    }
    write_json(APP_STATE_PATH, result)
    return result


def _question_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    if not isinstance(value, str) or not value.strip():
        raise AppCliError("question must be a non-empty string")
    question = re.sub(r"\s+", " ", value).strip()
    if len(question.encode("utf-8")) > 4096:
        raise AppCliError("question exceeds the 4096-byte limit")
    return question


def _question_symbols(question: str) -> list[str]:
    symbols = []
    for match in QUESTION_TICKER_PATTERN.finditer(question.upper()):
        symbol = match.group(1)
        if symbol in QUESTION_TICKER_STOPWORDS:
            continue
        if SYMBOL_PATTERN.fullmatch(symbol) and symbol not in symbols:
            symbols.append(symbol)
    if len(symbols) > 8:
        raise AppCliError("one natural-language request may analyze at most 8 symbols")
    return symbols


def _latest_accepted_report() -> dict[str, Any]:
    report = _read_json(DEFAULT_OUTPUT_ROOT / "status" / "latest.json")
    if report is None:
        raise AppCliError(
            "accepted AI Radar daily report is missing; run `ai-radar daily` first"
        )
    quality = report.get("quality_audit") or {}
    if (
        quality.get("schema_version") != QUALITY_SCHEMA_VERSION
        or quality.get("status") != "green"
        or not quality.get("publishable_reference_report")
        or float((quality.get("scores") or {}).get("flow_evidence_quality", 0))
        < 8.0
    ):
        raise AppCliError("latest AI Radar report did not pass all quality gates")
    oracle = _read_json(DEFAULT_ORACLE_STATUS) or {}
    oracle_date = str(oracle.get("target_as_of_date") or "")
    report_date = str(report.get("as_of_date") or "")
    if oracle_date and report_date != oracle_date:
        raise AppCliError(
            "latest AI Radar report is stale relative to Oracle: "
            f"report={report_date} oracle={oracle_date}; run `ai-radar daily`"
        )
    return report


def _question_intent(question: str) -> str:
    compact = question.lower()
    if any(token in compact for token in ("상태", "status", "준비", "작동")):
        return "status"
    if any(
        token in compact
        for token in (
            "후보",
            "강세",
            "약세",
            "위험 종목",
            "관찰 종목",
            "어느 종목",
        )
    ):
        return "candidates"
    if any(
        token in compact
        for token in ("회전", "로테이션", "rotation", "섹터", "테마")
    ):
        return "rotation"
    return "market"


def run_question(question: str) -> dict[str, Any]:
    """Answer Oracle-style requests from accepted AI Radar artifacts."""

    normalized = _question_text(question)
    symbols = _question_symbols(normalized)
    if symbols:
        result = run_analysis(symbols)
        return {
            **result,
            "action": "ask",
            "intent": "explicit_symbol_analysis",
            "question": normalized,
            "answer_basis": "fresh_on_demand_trained_model_inference",
        }
    intent = _question_intent(normalized)
    if intent == "status":
        return {
            **status(),
            "action": "ask",
            "intent": intent,
            "question": normalized,
            "answer_basis": "live_runtime_status",
        }
    report = _latest_accepted_report()
    market = report.get("market_judgement") or {}
    dashboard = report.get("market_dashboard") or {}
    answer: dict[str, Any] = {
        "schema_version": "quant.ai_radar_natural_answer.v1",
        "status": "PASS",
        "app_id": APP_ID,
        "action": "ask",
        "intent": intent,
        "question": normalized,
        "as_of_date": report.get("as_of_date"),
        "answer_basis": "latest_accepted_trained_model_report",
        "market_state": market.get("market_state"),
        "confidence": market.get("confidence"),
        "summary": market.get("summary"),
        "source_status": report.get("source_status"),
        "quality_audit": report.get("quality_audit"),
        "market_report_html": str(
            DEFAULT_OUTPUT_ROOT
            / "runs"
            / str(report.get("as_of_date"))
            / "market_report.html"
        ),
        "completed_at_kst": _now_kst(),
    }
    if intent == "candidates":
        answer["candidate_lanes"] = dashboard.get("candidate_lanes") or {}
    elif intent == "rotation":
        answer["rotation_clusters"] = dashboard.get("rotation_clusters") or []
        answer["accumulation_clusters"] = (
            dashboard.get("accumulation_clusters") or []
        )
    else:
        answer["breadth"] = dashboard.get("breadth") or {}
        answer["candidate_lanes"] = dashboard.get("candidate_lanes") or {}
        answer["rotation_clusters"] = dashboard.get("rotation_clusters") or []
        answer["confirmations"] = market.get("confirmations") or []
        answer["contradictions"] = market.get("contradictions") or []
        answer["unknowns"] = market.get("unknowns") or []
    write_json(APP_STATE_PATH, answer)
    return answer


def _models_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    path = parsed.path
    marker = "/v1/"
    if marker not in path:
        raise AppCliError(
            "model endpoint must use an OpenAI-compatible /v1/ path"
        )
    prefix = path.split(marker, 1)[0]
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"{prefix}/v1/models", "", "")
    )


def endpoint_status(*, timeout: int = 5) -> dict[str, Any]:
    endpoint = _model_endpoint()
    headers = {"Accept": "application/json"}
    token_file = _token_file()
    if token_file:
        token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
        if not token:
            raise AppCliError(f"model token file is empty: {token_file}")
        headers["Authorization"] = f"Bearer {token}"
    models_url = _models_url(endpoint)
    try:
        with urlopen(Request(models_url, headers=headers), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "models_url": models_url,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    model_ids = sorted(
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    )
    return {
        "status": "confirmed",
        "models_url": models_url,
        "model_ids": model_ids,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _oracle_summary(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    flow = value.get("etf_flow")
    flow = flow if isinstance(flow, dict) else {}
    return {
        "status": value.get("status"),
        "target_as_of_date": value.get("target_as_of_date"),
        "base_history_end": value.get("base_history_end"),
        "database": value.get("database"),
        "missing_session_count": len(value.get("missing_sessions") or []),
        "latest_flow_effective_date": flow.get("latest_effective_date"),
        "latest_flow_processed_date": flow.get("latest_processed_date"),
        "flow_record_count": flow.get("record_count"),
        "flow_ticker_count": flow.get("ticker_count"),
    }


def preflight() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    release_path = Path(_release_manifest()).expanduser().resolve()
    release = load_model_release(release_path)
    checks["model_release"] = {
        "status": "confirmed",
        **release.public_metadata(),
    }
    binding = load_shared_market_binding(
        base_database=DEFAULT_BASE_DATABASE,
        incremental_database=DEFAULT_INCREMENTAL_DATABASE,
        oracle_status_path=DEFAULT_ORACLE_STATUS,
        max_constituent_available_lag_days=_positive_int(
            os.environ.get(
                "QUANT_AI_MAX_CONSTITUENT_AVAILABLE_LAG_DAYS", "45"
            ),
            "QUANT_AI_MAX_CONSTITUENT_AVAILABLE_LAG_DAYS",
            allow_zero=True,
        ),
    )
    checks["data_backend"] = {
        "source_owner": "market-structure-oracle",
        **binding.public_metadata(),
    }
    checks["relation_index"] = {
        "status": (
            "confirmed" if DEFAULT_RELATION_INDEX.is_file() else "failed"
        ),
        "path": str(DEFAULT_RELATION_INDEX),
    }
    checks["model_endpoint"] = endpoint_status()
    checks["email_transport"] = email_transport_status()
    if (
        checks["model_endpoint"].get("status") == "confirmed"
        and release.endpoint_model
        not in checks["model_endpoint"].get("model_ids", [])
    ):
        checks["model_endpoint"] = {
            **checks["model_endpoint"],
            "status": "failed",
            "error": (
                "accepted endpoint model is not served: "
                f"{release.endpoint_model}"
            ),
        }
    failed = [
        name
        for name, value in checks.items()
        if isinstance(value, dict) and value.get("status") == "failed"
    ]
    if failed:
        raise AppCliError(f"preflight checks failed: {failed}")
    return {
        "schema_version": "quant.ai_radar_app_preflight.v1",
        "status": "PREFLIGHT_PASS",
        "app_id": APP_ID,
        "checked_at_kst": _now_kst(),
        "checks": checks,
        "fmp_historical_backfill": "PAUSED",
        "timer_activation_changed": False,
        "trade_execution": "NOT_APPLICABLE",
    }


def status() -> dict[str, Any]:
    endpoint = endpoint_status()
    latest = _read_json(DEFAULT_OUTPUT_ROOT / "status" / "latest.json")
    app_state = _read_json(APP_STATE_PATH)
    daily_cycle = _read_json(DEFAULT_OUTPUT_ROOT / "status" / "daily_cycle.json")
    oracle = _read_json(DEFAULT_ORACLE_STATUS)
    latest_as_of = str((latest or {}).get("as_of_date") or "")
    prepare = (
        (app_state or {}).get("prepare")
        if isinstance((app_state or {}).get("prepare"), dict)
        else {}
    )
    prepared_binding = (
        prepare.get("binding")
        if isinstance(prepare.get("binding"), dict)
        else {}
    )
    current_source_fingerprint = str(
        prepared_binding.get("source_fingerprint_sha256") or ""
    )
    binding_error = (
        ""
        if re.fullmatch(r"[0-9a-f]{64}", current_source_fingerprint)
        else "current prepared Oracle source fingerprint is unavailable"
    )
    latest_source = (
        (latest or {}).get("source_status")
        if isinstance((latest or {}).get("source_status"), dict)
        else {}
    )
    latest_oracle = (
        latest_source.get("shared_oracle_store")
        if isinstance(latest_source.get("shared_oracle_store"), dict)
        else {}
    )
    latest_source_fingerprint = str(
        latest_oracle.get("source_fingerprint_sha256") or ""
    )
    report_path = (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / latest_as_of
        / "market_report.json"
    )
    report_sha256 = (
        hashlib.sha256(report_path.read_bytes()).hexdigest()
        if report_path.is_file()
        else ""
    )
    publication_source_current = bool(
        current_source_fingerprint
        and latest_source_fingerprint == current_source_fingerprint
    )
    email = (
        email_delivery_status(
            latest_as_of,
            report_sha256=report_sha256,
            source_fingerprint_sha256=current_source_fingerprint,
        )
        if latest_as_of and report_sha256 and current_source_fingerprint
        else None
    )
    publication_current = bool(
        publication_source_current
        and email
        and email.get("complete") is True
    )
    running = str((app_state or {}).get("status") or "").startswith("running_")
    if endpoint["status"] != "confirmed" or binding_error:
        overall_status = "partial"
    elif running:
        overall_status = "running"
    elif publication_current:
        overall_status = "confirmed"
    else:
        overall_status = "waiting"
    return {
        "schema_version": "quant.ai_radar_app_status.v1",
        "status": overall_status,
        "app_id": APP_ID,
        "checked_at_kst": _now_kst(),
        "model_endpoint": endpoint,
        "latest_published": (
            {
                "as_of_date": latest.get("as_of_date"),
                "deployment_mode": latest.get("deployment_mode"),
                "selected_model_scope_complete": latest.get(
                    "selected_model_scope_complete"
                ),
                "queue_counts": latest.get("queue_counts"),
                "market_report_json": str(report_path),
                "report_sha256": report_sha256,
                "source_fingerprint_sha256": latest_source_fingerprint,
                "current_oracle_source_fingerprint_sha256": (
                    current_source_fingerprint
                ),
                "source_matches_current_oracle": publication_source_current,
                "publication_current": publication_current,
                "publication_status": (
                    "CURRENT"
                    if publication_current
                    else (
                        "WAITING_EMAIL"
                        if publication_source_current
                        else "STALE_ORACLE_SOURCE"
                    )
                ),
            }
            if latest
            else None
        ),
        "app_state": app_state,
        "legacy_daily_cycle_state": daily_cycle,
        "data_backend_status": {
            "source_owner": "market-structure-oracle",
            **(_oracle_summary(oracle) or {}),
            "source_fingerprint_sha256": current_source_fingerprint,
            "binding_error": binding_error or None,
        },
        "email_delivery": email,
        "fmp_historical_backfill": "PAUSED",
        "timer_activation_changed": False,
        "trade_execution": "NOT_APPLICABLE",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-radar",
        description="Run the standalone Quant AI Radar application.",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    sub = parser.add_subparsers(dest="action")

    daily = sub.add_parser("daily", help="Run the full daily Radar workflow")
    daily.add_argument("--shadow", action="store_true")
    daily.add_argument("--workers", type=int)
    daily.add_argument("--max-ai-etfs", type=int)
    daily.add_argument("--max-ai-stocks", type=int)
    daily.add_argument("--smoke-max-items", type=int, default=0)

    analyze = sub.add_parser(
        "analyze", help="Analyze explicitly requested symbols"
    )
    analyze.add_argument("symbols", nargs="*")
    analyze.add_argument("--symbols-file", type=Path)

    ask = sub.add_parser(
        "ask", help="Ask for market, rotation, candidates, or symbol analysis"
    )
    ask.add_argument("question", nargs="+")

    sub.add_parser("status", help="Show model, Oracle, and output status")
    sub.add_parser("preflight", help="Validate the app without running inference")
    return parser


def _request_action(
    request: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    action = str(request.get("action") or "daily").strip().lower()
    if action not in {"daily", "analyze", "ask", "status", "preflight"}:
        raise AppCliError(f"unsupported action: {action!r}")
    return action, request


def _daily_values(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "shadow": _boolean(
            _provided_or_default(source, "shadow", False), "shadow"
        ),
        "workers": _positive_int(
            _provided_or_default(
                source, "workers", os.environ.get("QUANT_AI_WORKERS", "4")
            ),
            "workers",
        ),
        "max_ai_etfs": _positive_int(
            _provided_or_default(
                source,
                "max_ai_etfs",
                os.environ.get("QUANT_AI_MAX_ETFS", "64"),
            ),
            "max_ai_etfs",
        ),
        "max_ai_stocks": _positive_int(
            _provided_or_default(
                source,
                "max_ai_stocks",
                os.environ.get("QUANT_AI_MAX_STOCKS", "192"),
            ),
            "max_ai_stocks",
        ),
        "smoke_max_items": _positive_int(
            _provided_or_default(source, "smoke_max_items", 0),
            "smoke_max_items",
            allow_zero=True,
        ),
    }


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    request = load_sealed_request()
    if request is not None and args.action is not None:
        raise AppCliError(
            "cannot combine direct CLI arguments with a sealed App Manager request"
        )
    if os.environ.get("QUANT_AI_RADAR_APP_PREFLIGHT_ONLY") == "1":
        return preflight()
    if request is not None:
        action, source = _request_action(request)
    else:
        action = args.action or "daily"
        source = vars(args)
    if action == "daily":
        values = _daily_values(source)
        return run_daily(**values)
    if action == "analyze":
        if source.get("symbols") and source.get("symbols_file"):
            raise AppCliError("cannot combine symbols with symbols_file")
        if source.get("symbols_file"):
            symbols, symbols_source = load_symbols_file(source.get("symbols_file"))
            return run_analysis(symbols, symbols_source=symbols_source)
        return run_analysis(normalize_symbols(source.get("symbols")))
    if action == "ask":
        return run_question(source.get("question") or source.get("query"))
    if action == "status":
        return status()
    if action == "preflight":
        return preflight()
    raise AppCliError(f"unsupported action: {action!r}")


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        load_env_file(args.env_file)
        result = dispatch(args)
    except Exception as exc:
        failure = {
            "schema_version": "quant.ai_radar_app_result.v1",
            "status": "FAIL",
            "app_id": APP_ID,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at_kst": _now_kst(),
        }
        try:
            write_json(APP_STATE_PATH, failure)
        except OSError:
            pass
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
