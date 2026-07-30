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
from workflows.quant_ai_radar.universe import resolve_as_of_date, write_json  # noqa: E402
from workflows.quant_ai_radar.report_renderer import (  # noqa: E402
    render_single_security_html,
)


KST = ZoneInfo("Asia/Seoul")
DEFAULT_DATA_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR"
)
DEFAULT_RELEASE_MANIFEST = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/"
    "qwen3_8b_quant_lora_v1/release_manifest.json"
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


def _analysis_packet(
    pipeline: DatasetPipeline,
    *,
    symbol: str,
    as_of_date: str,
    corporate_actions: dict[str, Any],
) -> dict[str, Any]:
    packet = pipeline.analysis_packet_for_pair(
        symbol,
        as_of_date,
        lookback_days=21,
        recompute_quality=False,
    )
    return adjust_packet_for_verified_corporate_actions(
        packet,
        corporate_actions,
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    symbols = _symbols(args.symbols)
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
                    "schema_version": "quant.ai_radar_on_demand.v1",
                    "status": "not_analyzed_fail_closed",
                    "symbol": symbol,
                    "as_of_date": as_of,
                    "eligibility": eligibility,
                    "reason": "|".join(eligibility["reasons"]),
                    "shared_market_store": binding.public_metadata(),
                }
            else:
                example = build_example(packet)
                expected = json.loads(str(example["response"]))
                judgement, trace = client.complete_validated(
                    system=str(example["context"]),
                    user=str(example["instruction"]),
                    expected_response=expected,
                    max_tokens=1400,
                )
                value = {
                    "schema_version": "quant.ai_radar_on_demand.v1",
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
                    "scope": "data_interpretation_not_trade_execution",
                }
        except Exception as exc:
            value = {
                "schema_version": "quant.ai_radar_on_demand.v1",
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
        "schema_version": "quant.ai_radar_on_demand_receipt.v1",
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
