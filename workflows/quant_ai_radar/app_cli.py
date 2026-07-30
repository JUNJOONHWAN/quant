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
from workflows.quant_ai_radar.relation_index import (  # noqa: E402
    DEFAULT_RELATION_INDEX,
)
from workflows.quant_ai_radar.run_daily_cycle import build_stage_commands  # noqa: E402
from workflows.quant_ai_radar.universe import write_json  # noqa: E402


KST = ZoneInfo("Asia/Seoul")
APP_ID = "quant-ai-radar"
DEFAULT_ENV_FILE = Path("/home/zooh/.config/quant/quant-ai-radar.env")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR"
)
DEFAULT_RELEASE_MANIFEST = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/"
    "qwen3_8b_quant_lora_v1/release_manifest.json"
)
APP_STATE_PATH = DEFAULT_OUTPUT_ROOT / "status" / "app_cli.json"
REQUEST_FIELDS = frozenset(
    {
        "action",
        "symbols",
        "shadow",
        "workers",
        "max_ai_etfs",
        "max_ai_stocks",
        "smoke_max_items",
    }
)
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")


class AppCliError(RuntimeError):
    """The standalone application cannot safely complete the request."""


def _now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


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
        "--max-constituent-available-lag-days",
        os.environ.get(
            "QUANT_AI_MAX_CONSTITUENT_AVAILABLE_LAG_DAYS", "45"
        ),
    ]
    token_file = _token_file()
    if token_file:
        command.extend(["--model-token-file", token_file])
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


def run_daily(
    *,
    shadow: bool,
    workers: int,
    max_ai_etfs: int,
    max_ai_stocks: int,
    smoke_max_items: int = 0,
) -> dict[str, Any]:
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
        "completed_at_kst": _now_kst(),
    }
    write_json(APP_STATE_PATH, result)
    return result


def run_analysis(symbols: list[str]) -> dict[str, Any]:
    state = {
        "schema_version": "quant.ai_radar_app_run.v1",
        "status": "running_on_demand_analysis",
        "app_id": APP_ID,
        "action": "analyze",
        "symbols": symbols,
        "operations_app_run_id": os.environ.get("OPERATIONS_APP_RUN_ID"),
        "started_at_kst": _now_kst(),
    }
    write_json(APP_STATE_PATH, state)
    receipt = run_json_command(build_analyze_command(symbols))
    if receipt.get("status") != "complete":
        raise AppCliError(
            f"on-demand analysis did not fully complete: {receipt.get('status')}"
        )
    result = {
        **state,
        "status": "PASS",
        "engine_status": receipt["status"],
        "as_of_date": receipt.get("as_of_date"),
        "results": receipt.get("results", []),
        "completed_at_kst": _now_kst(),
    }
    write_json(APP_STATE_PATH, result)
    return result


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
    return {
        "schema_version": "quant.ai_radar_app_status.v1",
        "status": "confirmed" if endpoint["status"] == "confirmed" else "partial",
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
                "market_report_json": str(
                    DEFAULT_OUTPUT_ROOT
                    / "runs"
                    / str(latest.get("as_of_date"))
                    / "market_report.json"
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
        },
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
    analyze.add_argument("symbols", nargs="+")

    sub.add_parser("status", help="Show model, Oracle, and output status")
    sub.add_parser("preflight", help="Validate the app without running inference")
    return parser


def _request_action(
    request: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    action = str(request.get("action") or "daily").strip().lower()
    if action not in {"daily", "analyze", "status", "preflight"}:
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
        return run_analysis(normalize_symbols(source.get("symbols")))
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
