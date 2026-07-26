#!/usr/bin/env python3
"""Prepare or run the full-universe trained Quant AI Radar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from quant_dataset.config import DEFAULT_SECRETS_PATH, load_credentials  # noqa: E402
from quant_dataset.pipeline import DatasetPipeline  # noqa: E402
from training.quant_llm.build_sft_dataset import (  # noqa: E402
    build_example,
    packet_eligibility,
)
from workflows.quant_ai_radar.etfradar_release import (  # noqa: E402
    EtfRadarReleaseError,
    load_release_evidence,
)
from workflows.quant_ai_radar.market_report import (  # noqa: E402
    aggregate_judgements,
    etfradar_summary,
    synthesize_market,
)
from workflows.quant_ai_radar.model_runtime import (  # noqa: E402
    InferenceError,
    ModelGateError,
    ResponseContractError,
    TrainedQuantClient,
    load_model_release,
)
from workflows.quant_ai_radar.run_queue import RadarQueue  # noqa: E402
from workflows.quant_ai_radar.universe import (  # noqa: E402
    UniverseError,
    dataset_source_fingerprint,
    resolve_as_of_date,
    scan_universe,
    write_candidates,
    write_json,
)


KST = ZoneInfo("Asia/Seoul")
DEFAULT_DATA_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET")
DEFAULT_ETFRADAR_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/ETFRADAR")
DEFAULT_OUTPUT_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR")
DEFAULT_RELEASE_MANIFEST = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/"
    "qwen3_8b_quant_lora_v1/release_manifest.json"
)


class RadarRunError(RuntimeError):
    """Raised when a complete production radar cannot be produced."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _token(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise ModelGateError(f"model endpoint token file is missing: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ModelGateError(f"model endpoint token file is empty: {path}")
    return value


def _response_work(
    client: TrainedQuantClient, example: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = json.loads(str(example["response"]))
    return client.complete_validated(
        system=str(example["context"]),
        user=str(example["instruction"]),
        expected_response=expected,
        max_tokens=1400,
    )


def _status_document(
    *, status: str, as_of_date: str, run_dir: Path, queue: RadarQueue, **extra: Any
) -> dict[str, Any]:
    return {
        "schema_version": "quant.ai_radar_run_status.v1",
        "status": status,
        "as_of_date": as_of_date,
        "run_dir": str(run_dir),
        "queue_counts": queue.counts(),
        "updated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        **extra,
    }


def _handle_future(
    *, future: Future, context: dict[str, Any], queue: RadarQueue
) -> None:
    symbol = context["symbol"]
    try:
        judgement, trace = future.result()
        queue.mark_done(
            symbol=symbol,
            actual_task_type=context["actual_task_type"],
            packet_id=context["packet_id"],
            eligibility=context["eligibility"],
            prompt_sha256=trace["request_sha256"],
            response_sha256=trace["response_sha256"],
            result=judgement,
        )
    except Exception as exc:
        queue.mark_error(symbol, f"{type(exc).__name__}: {exc}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.expanduser().resolve()
    database = data_root / "normalized" / "daily_observations.sqlite3"
    as_of = resolve_as_of_date(database, args.as_of_date)
    run_dir = args.output_root.expanduser().resolve() / "runs" / as_of
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"

    etfradar = load_release_evidence(args.etfradar_data_root, as_of)
    candidates, universe_manifest = scan_universe(database, as_of)
    etfradar_tickers = {
        str(row.get("ticker") or "").upper()
        for row in etfradar["tables"]["02_ETF_MASTER"]
        if row.get("ticker")
    }
    quant_candidate_symbols = {item.symbol for item in candidates}
    universe_manifest["etfradar_release_id"] = etfradar["binding"]["release_id"]
    universe_manifest["etfradar_master_ticker_count"] = len(etfradar_tickers)
    universe_manifest["etfradar_tickers_without_quant_candidate_count"] = len(
        etfradar_tickers - quant_candidate_symbols
    )
    universe_manifest["etfradar_tickers_without_quant_candidate_sample"] = sorted(
        etfradar_tickers - quant_candidate_symbols
    )[:100]
    write_json(run_dir / "universe_manifest.json", universe_manifest)
    write_candidates(run_dir / "candidates.jsonl", candidates)
    write_json(run_dir / "etfradar_release_binding.json", etfradar["binding"])

    dataset_manifest = data_root / "state" / "dataset_manifest.json"
    if not dataset_manifest.is_file():
        raise RadarRunError(f"quant dataset manifest is missing: {dataset_manifest}")
    queue = RadarQueue(run_dir / "run_queue.sqlite3")
    source_fingerprint = dataset_source_fingerprint(database, as_of)
    queue.bind_metadata(
        {
            "as_of_date": as_of,
            "dataset_source_fingerprint_sha256": source_fingerprint["sha256"],
            "etfradar_release_manifest_sha256": etfradar["binding"][
                "release_manifest_sha256"
            ],
            "candidate_count": len(candidates),
        }
    )
    queue.seed(candidates)

    if args.prepare_only:
        state = _status_document(
            status="prepared_waiting_for_accepted_model",
            as_of_date=as_of,
            run_dir=run_dir,
            queue=queue,
            inference_started=False,
            training_started=False,
            model_release_required=str(args.release_manifest),
            universe_manifest=str(run_dir / "universe_manifest.json"),
            etfradar_release_binding=str(run_dir / "etfradar_release_binding.json"),
        )
        write_json(state_path, state)
        return state

    release = load_model_release(args.release_manifest)
    queue.bind_metadata(
        {
            "model_release_manifest_sha256": release.manifest_sha256,
            "model_id": release.model_id,
        }
    )
    if not args.model_endpoint:
        raise ModelGateError("--model-endpoint is required outside --prepare-only")
    client = TrainedQuantClient(
        endpoint=args.model_endpoint,
        release=release,
        token=_token(args.model_token_file),
        timeout=args.timeout,
    )
    credentials = load_credentials(secrets_path=args.secrets_file)
    pipeline = DatasetPipeline(
        data_root=data_root,
        credentials=credentials,
        timeout_seconds=args.timeout,
        retries=1,
    )

    pending = queue.pending()
    smoke_limited = args.smoke_max_items > 0
    if smoke_limited:
        pending = pending[: args.smoke_max_items]
    inflight: dict[Future, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for item in pending:
            symbol = str(item["symbol"])
            queue.mark_running(symbol)
            try:
                packet = pipeline.analysis_packet_for_pair(
                    symbol, as_of, lookback_days=21, recompute_quality=False
                )
                eligibility = packet_eligibility(packet)
                if not eligibility["eligible"]:
                    queue.mark_excluded(
                        symbol,
                        eligibility,
                        "|".join(eligibility["reasons"]) or "packet_ineligible",
                    )
                    continue
                example = build_example(packet)
                actual_task_type = str(example["metadata"]["task_type"])
                if actual_task_type == "all_stock_control_analysis":
                    queue.mark_excluded(
                        symbol,
                        eligibility,
                        "no_visible_etf_relation_under_pit_packet",
                    )
                    continue
                future = executor.submit(_response_work, client, example)
                inflight[future] = {
                    "symbol": symbol,
                    "actual_task_type": actual_task_type,
                    "packet_id": str(packet["packet_id"]),
                    "eligibility": eligibility,
                }
                if len(inflight) >= max(args.workers * 2, 1):
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for completed in done:
                        context = inflight.pop(completed)
                        _handle_future(future=completed, context=context, queue=queue)
            except Exception as exc:
                queue.mark_error(symbol, f"{type(exc).__name__}: {exc}")
        for completed in list(inflight):
            _handle_future(
                future=completed, context=inflight[completed], queue=queue
            )

    counts = queue.counts()
    if smoke_limited:
        state = _status_document(
            status="smoke_complete_not_publishable",
            as_of_date=as_of,
            run_dir=run_dir,
            queue=queue,
            inference_started=True,
            production_scope_complete=False,
            smoke_max_items=args.smoke_max_items,
        )
        write_json(state_path, state)
        return state
    if counts.get("pending", 0) or counts.get("running", 0) or counts.get("error", 0):
        state = _status_document(
            status="incomplete_no_market_judgement",
            as_of_date=as_of,
            run_dir=run_dir,
            queue=queue,
            inference_started=True,
            production_scope_complete=False,
            exclusion_counts=queue.exclusions(),
        )
        write_json(state_path, state)
        return state

    results = queue.done_results()
    _write_jsonl(run_dir / "security_judgements.jsonl", results)
    aggregate = aggregate_judgements(results)
    radar = etfradar_summary(etfradar)
    synthesis, synthesis_trace, catalog = synthesize_market(
        client=client,
        as_of_date=as_of,
        aggregate=aggregate,
        radar=radar,
    )
    report = {
        "schema_version": "quant.ai_radar_report.v1",
        "as_of_date": as_of,
        "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "scope": "market_and_security_analysis_not_trade_execution",
        "full_universe_complete": True,
        "model_release": release.public_metadata(),
        "source_status": {
            "quant_dataset": {
                "status": "confirmed",
                "manifest_sha256": _sha256(dataset_manifest),
                "source_fingerprint": source_fingerprint,
            },
            "etfradar_release": {
                "status": "confirmed",
                **etfradar["binding"],
            },
        },
        "universe": universe_manifest,
        "queue_counts": counts,
        "exclusion_counts": queue.exclusions(),
        "aggregate": aggregate,
        "etfradar": radar,
        "market_judgement": synthesis,
        "market_judgement_trace": synthesis_trace,
        "market_evidence_catalog": catalog,
        "security_judgements_path": str(run_dir / "security_judgements.jsonl"),
    }
    report_path = run_dir / "market_report.json"
    write_json(report_path, report)
    latest_path = args.output_root.expanduser().resolve() / "status" / "latest.json"
    write_json(latest_path, report)
    state = _status_document(
        status="complete",
        as_of_date=as_of,
        run_dir=run_dir,
        queue=queue,
        inference_started=True,
        production_scope_complete=True,
        report=str(report_path),
    )
    write_json(state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--etfradar-data-root", type=Path, default=DEFAULT_ETFRADAR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--as-of-date")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    parser.add_argument("--model-endpoint")
    parser.add_argument("--model-token-file", type=Path)
    parser.add_argument("--secrets-file", type=Path, default=DEFAULT_SECRETS_PATH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--smoke-max-items",
        type=int,
        default=0,
        help="explicit non-publishable smoke cap; 0 means the complete eligible universe",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    try:
        result = run(args)
    except (
        EtfRadarReleaseError,
        InferenceError,
        ModelGateError,
        RadarRunError,
        ResponseContractError,
        UniverseError,
        OSError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] in {
        "complete",
        "prepared_waiting_for_accepted_model",
        "smoke_complete_not_publishable",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
