#!/usr/bin/env python3
"""Analyze explicitly requested symbols with the accepted Quant LoRA."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from quant_dataset.config import load_credentials  # noqa: E402
from quant_dataset.pipeline import DatasetPipeline  # noqa: E402
from quant_dataset.shared_market import (  # noqa: E402
    DEFAULT_INCREMENTAL_DATABASE,
    DEFAULT_ORACLE_STATUS,
    SharedReadOnlyDatabase,
    load_shared_market_binding,
)
from training.quant_llm.build_sft_dataset import (  # noqa: E402
    build_example,
    packet_eligibility,
)
from workflows.quant_ai_radar.model_runtime import (  # noqa: E402
    TrainedQuantClient,
    load_model_release,
)
from workflows.quant_ai_radar.corporate_actions import (  # noqa: E402
    adjust_packet_for_verified_corporate_actions,
    load_oracle_corporate_actions,
)
from workflows.quant_ai_radar.forecast_synthesis import (  # noqa: E402
    DEFAULT_FORECAST_ENDPOINT,
    DEFAULT_FORECAST_MODEL,
    ForecastSynthesisClient,
)
from workflows.quant_ai_radar.historical_analog import (  # noqa: E402
    DEFAULT_ANALOG_INDEX,
    DEFAULT_EXAMPLE_DATABASE,
    DEFAULT_PRICE_DATABASE,
    HistoricalAnalogEngine,
)
from workflows.quant_ai_radar.universe import resolve_as_of_date, write_json  # noqa: E402
from workflows.quant_ai_radar.report_renderer import (  # noqa: E402
    render_single_security_html,
)
from workflows.quant_ai_radar.training_native import (  # noqa: E402
    TRAINING_NATIVE_PROMPT_CONTRACT,
    complete_training_native_judgement,
)


KST = ZoneInfo("Asia/Seoul")
DEFAULT_DATA_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR"
)
DEFAULT_RELEASE_MANIFEST = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/"
    "Qwen3-8B-FLOW/release_manifest.json"
)
DEFAULT_SECRETS = Path("/home/zooh/.dgx-secrets/secrets.env")
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")


class OnDemandError(RuntimeError):
    """A requested analysis cannot safely be produced."""


def _token(path: Path | None) -> str | None:
    if path is None:
        return None
    value = path.expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise OnDemandError(f"model token file is empty: {path}")
    return value


def _symbols(raw: list[str]) -> list[str]:
    result = []
    for value in raw:
        for piece in value.split(","):
            symbol = piece.strip().upper()
            if not SYMBOL_PATTERN.fullmatch(symbol):
                raise OnDemandError(f"invalid symbol: {piece!r}")
            if symbol not in result:
                result.append(symbol)
    if not result:
        raise OnDemandError("at least one symbol is required")
    return result


def _daily_market_context(
    output_root: Path, analysis_as_of_date: str
) -> dict[str, Any]:
    """Load a compact, same-date daily context without requiring it to exist."""

    report_path = output_root / "runs" / analysis_as_of_date / "market_report.json"
    if not report_path.is_file():
        return {
            "status": "not_available",
            "as_of_date": analysis_as_of_date,
            "reason": "same_date_daily_report_not_found",
        }
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OnDemandError(f"same-date daily report is invalid JSON: {report_path}") from exc
    if not isinstance(report, dict) or str(report.get("as_of_date") or "") != analysis_as_of_date:
        raise OnDemandError("same-date daily report has an incompatible as-of date")
    dashboard = report.get("market_dashboard")
    dashboard = dashboard if isinstance(dashboard, dict) else {}
    market_judgement = report.get("market_judgement")
    market_judgement = market_judgement if isinstance(market_judgement, dict) else {}
    return {
        "status": "available",
        "as_of_date": analysis_as_of_date,
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
        "breadth": dashboard.get("breadth") or {},
        "rotation_clusters": list(dashboard.get("rotation_clusters") or [])[:12],
        "source_report": str(report_path),
    }


def _analysis_packet(
    pipeline: DatasetPipeline,
    *,
    symbol: str,
    as_of_date: str,
    corporate_actions: dict[str, Any],
) -> dict[str, Any]:
    try:
        packet = pipeline.analysis_packet_for_pair(
            symbol,
            as_of_date,
            lookback_days=21,
            recompute_quality=False,
        )
    except ValueError as exc:
        expected = f"quality row missing for {symbol} {as_of_date}"
        if str(exc) != expected:
            raise

        # The sealed Oracle overlay is intentionally read-only. A symbol that
        # was outside the daily capacity selection can still have complete
        # source observations but no persisted quality row. Evaluate that
        # exact as-of pair in memory and expose it only while the packet is
        # assembled; no Oracle/base database row is mutated.
        packet_as_of_date = as_of_date
        if hasattr(pipeline.database, "observation_rows"):
            observations = pipeline.database.observation_rows(
                as_of_date,
                as_of_date,
                [symbol],
            )
        else:
            visible_observations = pipeline.database.history_payload_rows(
                symbol,
                as_of_date,
                1,
            )
            observations = [
                row
                for row in visible_observations
                if str(row.get("trade_date")) == as_of_date
            ]
            if not observations and visible_observations:
                packet_as_of_date = max(
                    str(row.get("trade_date"))
                    for row in visible_observations
                    if row.get("trade_date")
                )
                observations = [
                    row
                    for row in visible_observations
                    if str(row.get("trade_date")) == packet_as_of_date
                ]
        quality = pipeline.quality.evaluate(
            symbol,
            packet_as_of_date,
            observations,
        )
        if not observations:
            raise OnDemandError(
                f"no sealed Oracle observations for {symbol} {as_of_date}"
            ) from exc

        class _QualityOverlay:
            def __init__(self, delegate: Any, row: dict[str, Any]):
                self._delegate = delegate
                self._row = row

            def quality_for_pair(self, requested_symbol: str, trade_date: str):
                if requested_symbol == symbol and trade_date == packet_as_of_date:
                    return {
                        "status": self._row["status"],
                        "sources_json": json.dumps(self._row["sources"]),
                        "metrics_json": json.dumps(self._row["metrics"]),
                        "reasons_json": json.dumps(self._row["reasons"]),
                        "tolerances_json": json.dumps(self._row["tolerances"]),
                    }
                return self._delegate.quality_for_pair(
                    requested_symbol,
                    trade_date,
                )

            def __getattr__(self, name: str):
                return getattr(self._delegate, name)

        original_database = pipeline.database
        pipeline.database = _QualityOverlay(original_database, quality)
        try:
            packet = pipeline.analysis_packet_for_pair(
                symbol,
                packet_as_of_date,
                lookback_days=21,
                recompute_quality=False,
            )
        finally:
            pipeline.database = original_database
        packet["quality"]["evaluation_mode"] = "ephemeral_read_only_overlay"
        packet["freshness"] = {
            "requested_as_of_date": as_of_date,
            "analysis_as_of_date": packet_as_of_date,
            "stale_sessions_or_calendar_days": (
                0
                if packet_as_of_date == as_of_date
                else (
                    datetime.fromisoformat(as_of_date)
                    - datetime.fromisoformat(packet_as_of_date)
                ).days
            ),
            "fallback_policy": "latest_sealed_observation_not_after_requested_as_of",
        }
    return adjust_packet_for_verified_corporate_actions(
        packet,
        corporate_actions,
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    symbols = _symbols(args.symbols)
    if args.analog_neighbor_limit < 1 or args.analog_per_symbol_limit < 1:
        raise OnDemandError("historical analogue limits must be positive")
    data_root = args.data_root.expanduser().resolve()
    binding = load_shared_market_binding(
        base_database=data_root / "normalized" / "daily_observations.sqlite3",
        incremental_database=args.oracle_incremental_database,
        oracle_status_path=args.oracle_status,
        max_constituent_available_lag_days=(
            args.max_constituent_available_lag_days
        ),
    )
    database = SharedReadOnlyDatabase(binding)
    as_of = resolve_as_of_date(database)
    if as_of != binding.target_as_of_date:
        raise OnDemandError(
            f"quality/Oracle as-of mismatch: {as_of}!={binding.target_as_of_date}"
        )
    release = load_model_release(args.release_manifest)
    client = TrainedQuantClient(
        endpoint=args.model_endpoint,
        release=release,
        token=_token(args.model_token_file),
        timeout=args.timeout,
    )
    analog_engine = HistoricalAnalogEngine(
        example_database=args.analog_example_database,
        price_database=args.analog_price_database,
        index_path=args.analog_index_database,
        neighbor_limit=args.analog_neighbor_limit,
        per_symbol_limit=args.analog_per_symbol_limit,
    )
    forecast_client = ForecastSynthesisClient(
        endpoint=args.forecast_model_endpoint,
        model=args.forecast_model_name,
        token=_token(args.forecast_model_token_file),
        timeout=args.forecast_timeout,
    )
    pipeline = DatasetPipeline(
        data_root=data_root,
        credentials=load_credentials(secrets_path=args.secrets_file),
        timeout_seconds=args.timeout,
        retries=1,
        database=database,
        read_only=True,
    )
    corporate_actions = load_oracle_corporate_actions(
        database,
        as_of_date=as_of,
    )
    output_dir = args.output_root.expanduser().resolve() / "on_demand" / as_of
    results = []
    for symbol in symbols:
        output_path = output_dir / f"{symbol}.json"
        try:
            packet = _analysis_packet(
                pipeline,
                symbol=symbol,
                as_of_date=as_of,
                corporate_actions=corporate_actions,
            )
            eligibility = packet_eligibility(packet)
            if not eligibility["eligible"]:
                value = {
                    "schema_version": "quant.ai_radar_on_demand.v2",
                    "status": "not_analyzed_fail_closed",
                    "symbol": symbol,
                    "as_of_date": as_of,
                    "analysis_as_of_date": packet.get("as_of_date", as_of),
                    "freshness": packet.get("freshness") or {
                        "requested_as_of_date": as_of,
                        "analysis_as_of_date": packet.get("as_of_date", as_of),
                        "stale_sessions_or_calendar_days": 0,
                        "fallback_policy": "exact_requested_as_of",
                    },
                    "eligibility": eligibility,
                    "reason": "|".join(eligibility["reasons"]),
                    "shared_market_store": binding.public_metadata(),
                }
            else:
                example = build_example(packet)
                judgement, trace = complete_training_native_judgement(
                    client=client,
                    example=example,
                    max_tokens=900,
                )
                analysis_as_of = str(packet.get("as_of_date") or as_of)
                analog_forecast = analog_engine.forecast(
                    judgement=judgement,
                    analysis_as_of_date=analysis_as_of,
                )
                market_context = _daily_market_context(
                    args.output_root.expanduser().resolve(),
                    analysis_as_of,
                )
                qwen27_forecast, qwen27_trace = forecast_client.synthesize(
                    symbol=symbol,
                    judgement=judgement,
                    analog_forecast=analog_forecast,
                    market_context=market_context,
                )
                value = {
                    "schema_version": "quant.ai_radar_on_demand.v2",
                    "status": "complete",
                    "symbol": symbol,
                    "as_of_date": as_of,
                    "task_type": str(example["metadata"]["task_type"]),
                    "packet_id": str(packet["packet_id"]),
                    "eligibility": eligibility,
                    "model_release": release.public_metadata(),
                    "shared_market_store": binding.public_metadata(),
                    "oracle_corporate_actions": {
                        "schema_version": corporate_actions["schema_version"],
                        "sha256": corporate_actions["sha256"],
                        "accepted_event_count": len(
                            corporate_actions["events"]
                        ),
                    },
                    "judgement": judgement,
                    "trace": trace,
                    "historical_analog_forecast": analog_forecast,
                    "qwen27_forecast": qwen27_forecast,
                    "qwen27_forecast_trace": qwen27_trace,
                    "market_context": market_context,
                    "forecast_architecture": (
                        "qwen8_learned_pattern_to_historical_analog_to_qwen27"
                    ),
                    "training_native_prompt_contract": (
                        TRAINING_NATIVE_PROMPT_CONTRACT
                    ),
                    "scope": "data_interpretation_not_trade_execution",
                }
        except Exception as exc:
            value = {
                "schema_version": "quant.ai_radar_on_demand.v2",
                "status": "error",
                "symbol": symbol,
                "as_of_date": as_of,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scope": "data_interpretation_not_trade_execution",
            }
        value["generated_at_kst"] = datetime.now(KST).isoformat(timespec="seconds")
        if value["status"] == "complete":
            value["rendered_html"] = render_single_security_html(
                output_dir / f"{symbol}.html",
                result={
                    "symbol": symbol,
                    "task_type": value["task_type"],
                    "judgement": value["judgement"],
                    "historical_analog_forecast": value.get(
                        "historical_analog_forecast"
                    ),
                    "qwen27_forecast": value.get("qwen27_forecast"),
                },
                as_of_date=as_of,
            )
        write_json(output_path, value)
        result_item = {
            "symbol": symbol,
            "status": value["status"],
            "path": str(output_path),
        }
        if value.get("rendered_html"):
            result_item["html_path"] = value["rendered_html"]["path"]
        results.append(result_item)
    receipt = {
        "schema_version": "quant.ai_radar_on_demand_receipt.v2",
        "status": (
            "complete"
            if all(item["status"] == "complete" for item in results)
            else "partial_or_failed"
        ),
        "as_of_date": as_of,
        "requested_symbols": symbols,
        "oracle_corporate_actions": {
            "schema_version": corporate_actions["schema_version"],
            "sha256": corporate_actions["sha256"],
            "accepted_event_count": len(corporate_actions["events"]),
        },
        "results": results,
        "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
    }
    write_json(output_dir / "latest_request.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--oracle-incremental-database",
        type=Path,
        default=DEFAULT_INCREMENTAL_DATABASE,
    )
    parser.add_argument("--oracle-status", type=Path, default=DEFAULT_ORACLE_STATUS)
    parser.add_argument(
        "--max-constituent-available-lag-days", type=int, default=45
    )
    parser.add_argument(
        "--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST
    )
    parser.add_argument(
        "--model-endpoint",
        default="http://127.0.0.1:8018/v1/chat/completions",
    )
    parser.add_argument("--model-token-file", type=Path)
    parser.add_argument(
        "--forecast-model-endpoint",
        default=DEFAULT_FORECAST_ENDPOINT,
    )
    parser.add_argument(
        "--forecast-model-name",
        default=DEFAULT_FORECAST_MODEL,
    )
    parser.add_argument("--forecast-model-token-file", type=Path)
    parser.add_argument("--forecast-timeout", type=int, default=240)
    parser.add_argument(
        "--analog-example-database",
        type=Path,
        default=DEFAULT_EXAMPLE_DATABASE,
    )
    parser.add_argument(
        "--analog-price-database",
        type=Path,
        default=DEFAULT_PRICE_DATABASE,
    )
    parser.add_argument(
        "--analog-index-database",
        type=Path,
        default=DEFAULT_ANALOG_INDEX,
    )
    parser.add_argument("--analog-neighbor-limit", type=int, default=80)
    parser.add_argument("--analog-per-symbol-limit", type=int, default=2)
    parser.add_argument("--secrets-file", type=Path, default=DEFAULT_SECRETS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout", type=int, default=180)
    return parser


def main() -> int:
    try:
        result = analyze(build_parser().parse_args())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
