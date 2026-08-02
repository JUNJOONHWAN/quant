#!/usr/bin/env python3
"""Prepare or run the full-universe trained Quant AI Radar."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import date, datetime
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
    SharedMarketStoreError,
    SharedReadOnlyDatabase,
    load_shared_market_binding,
)
from training.quant_llm.build_sft_dataset import (  # noqa: E402
    build_example,
    packet_eligibility,
)
from workflows.quant_ai_radar.market_report import (  # noqa: E402
    aggregate_judgements,
    oracle_market_summary,
    synthesize_market_from_native_judgements,
)
from workflows.quant_ai_radar.oracle_features import (  # noqa: E402
    build_oracle_market_features,
)
from workflows.quant_ai_radar.corporate_actions import (  # noqa: E402
    adjust_packet_for_verified_corporate_actions,
    load_oracle_corporate_actions,
)
from workflows.quant_ai_radar.model_runtime import (  # noqa: E402
    InferenceError,
    ModelGateError,
    ResponseContractError,
    TrainedQuantClient,
    load_model_release,
)
from workflows.quant_ai_radar.run_queue import RadarQueue  # noqa: E402
from workflows.quant_ai_radar.relation_index import (  # noqa: E402
    DEFAULT_RELATION_INDEX,
    RelationIndexError,
    load_verified_relation_index,
)
from workflows.quant_ai_radar.universe import (  # noqa: E402
    UniverseError,
    dataset_source_fingerprint,
    resolve_as_of_date,
    scan_universe,
    write_candidates,
    write_json,
)
from workflows.quant_ai_radar.selection import (  # noqa: E402
    select_daily_inference,
)
from workflows.quant_ai_radar.report_renderer import render_reports  # noqa: E402
from workflows.quant_ai_radar.report_narratives import (  # noqa: E402
    build_native_judgement_narratives,
)
from workflows.quant_ai_radar.decision_support import (  # noqa: E402
    audit_report_quality,
    build_market_dashboard,
)
from workflows.quant_ai_radar.training_native import (  # noqa: E402
    TRAINING_NATIVE_PROMPT_CONTRACT,
    complete_training_native_judgement,
)


KST = ZoneInfo("Asia/Seoul")
DEFAULT_DATA_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET")
DEFAULT_OUTPUT_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR")
DEFAULT_RELEASE_MANIFEST = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/"
    "Qwen3-8B-FLOW/release_manifest.json"
)
DEFAULT_SECRETS_PATH = Path("/home/zooh/.dgx-secrets/secrets.env")
SUCCESSFUL_RUN_STATUSES = frozenset(
    {
        "complete",
        "prepared_waiting_for_accepted_model",
        "shadow_complete_not_published",
        "smoke_complete_not_publishable",
    }
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


def _archive_changed_source_run(
    run_dir: Path,
    *,
    source_fingerprint_sha256: str,
) -> dict[str, Any]:
    """Preserve and rotate a same-date queue when the sealed source changed."""

    queue_path = run_dir / "selected_run_queue.sqlite3"
    if not queue_path.is_file():
        return {
            "status": "not_required",
            "reason": "no_existing_queue",
        }
    try:
        with sqlite3.connect(
            f"file:{queue_path}?mode=ro",
            uri=True,
        ) as connection:
            metadata = {
                str(key): json.loads(str(value))
                for key, value in connection.execute(
                    "SELECT key,value_json FROM run_metadata"
                )
            }
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise RadarRunError(
            "existing run queue metadata is unreadable; refusing to "
            f"overwrite it: {type(exc).__name__}: {exc}"
        ) from exc
    previous = str(
        metadata.get("dataset_source_fingerprint_sha256") or ""
    )
    if previous == source_fingerprint_sha256:
        return {
            "status": "reused",
            "source_fingerprint_sha256": previous,
        }
    if not previous:
        raise RadarRunError(
            "existing run queue has no dataset source fingerprint"
        )

    stamp = datetime.now(KST).strftime("%Y%m%dT%H%M%SKST")
    archive_dir = (
        run_dir
        / "revisions"
        / (
            f"{previous[:12]}_to_"
            f"{source_fingerprint_sha256[:12]}_{stamp}"
        )
    )
    archive_dir.mkdir(parents=True, exist_ok=False)
    for child in list(run_dir.iterdir()):
        if child.name == "revisions":
            continue
        destination = archive_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{queue_path}{suffix}")
        if candidate.exists():
            candidate.unlink()
    manifest = {
        "schema_version": "quant.ai_radar_source_revision_archive.v1",
        "status": "archived",
        "reason": "sealed_oracle_source_fingerprint_changed",
        "previous_source_fingerprint_sha256": previous,
        "next_source_fingerprint_sha256": source_fingerprint_sha256,
        "archived_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "archive_dir": str(archive_dir),
    }
    write_json(archive_dir / "revision_manifest.json", manifest)
    return manifest


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
    return complete_training_native_judgement(
        client=client,
        example=example,
        max_tokens=900,
    )


def _inference_failure_trace(exc: Exception) -> dict[str, Any]:
    trace = getattr(exc, "trace", None)
    result = dict(trace) if isinstance(trace, dict) else {}
    raw_content = getattr(exc, "raw_content", None)
    if isinstance(raw_content, str):
        result["failed_response_sha256"] = hashlib.sha256(
            raw_content.encode("utf-8")
        ).hexdigest()
        result["failed_response_text"] = raw_content
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)
    return result


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
            trace=trace,
            result=judgement,
        )
    except Exception as exc:
        queue.mark_error(
            symbol,
            f"{type(exc).__name__}: {exc}",
            trace=_inference_failure_trace(exc),
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.expanduser().resolve()
    database = data_root / "normalized" / "daily_observations.sqlite3"
    shared_binding = load_shared_market_binding(
        base_database=database,
        incremental_database=args.oracle_incremental_database,
        oracle_status_path=args.oracle_status,
        max_constituent_available_lag_days=(
            args.max_constituent_available_lag_days
        ),
    )
    shared_database = SharedReadOnlyDatabase(shared_binding)
    requested_as_of = args.as_of_date or shared_binding.target_as_of_date
    if requested_as_of != shared_binding.target_as_of_date:
        raise RadarRunError(
            "as-of must match the sealed Oracle snapshot: "
            f"requested={requested_as_of} "
            f"oracle={shared_binding.target_as_of_date}"
        )
    as_of = resolve_as_of_date(shared_database, None)
    if as_of != shared_binding.target_as_of_date:
        raise RadarRunError(
            "latest quality-eligible market date does not match Oracle snapshot: "
            f"quality={as_of} oracle={shared_binding.target_as_of_date}"
        )
    run_dir = args.output_root.expanduser().resolve() / "runs" / as_of
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"
    source_fingerprint = dataset_source_fingerprint(shared_database, as_of)
    source_revision = _archive_changed_source_run(
        run_dir,
        source_fingerprint_sha256=source_fingerprint["sha256"],
    )

    relation_index = load_verified_relation_index(
        shared_binding, args.relation_index
    )
    candidates, universe_manifest = scan_universe(
        shared_database,
        as_of,
        relation_index_path=args.relation_index,
    )
    universe_manifest["shared_market_store"] = shared_binding.public_metadata()
    universe_manifest["relation_index"] = relation_index
    universe_manifest["source_revision"] = source_revision
    oracle_features = build_oracle_market_features(
        shared_database, candidates, as_of
    )
    universe_manifest["oracle_feature_snapshot_sha256"] = oracle_features[
        "snapshot_sha256"
    ]
    universe_manifest["oracle_feature_etf_count"] = len(
        oracle_features["etfs"]
    )
    universe_manifest["oracle_feature_stock_count"] = len(
        oracle_features["stocks"]
    )
    write_json(run_dir / "universe_manifest.json", universe_manifest)
    write_candidates(run_dir / "candidates.jsonl", candidates)
    write_json(run_dir / "oracle_market_features.json", oracle_features)
    selection = select_daily_inference(
        candidates,
        oracle_features,
        max_etfs=args.max_ai_etfs,
        max_stocks=args.max_ai_stocks,
    )
    if not selection.selected:
        raise RadarRunError("dynamic daily evidence selection produced zero candidates")
    write_json(run_dir / "selection_manifest.json", selection.manifest)
    write_candidates(run_dir / "selected_candidates.jsonl", selection.selected)
    _write_jsonl(
        run_dir / "coverage_ledger.jsonl",
        list(selection.coverage_ledger),
    )

    dataset_manifest = data_root / "state" / "dataset_manifest.json"
    if not dataset_manifest.is_file():
        raise RadarRunError(f"quant dataset manifest is missing: {dataset_manifest}")
    queue = RadarQueue(run_dir / "selected_run_queue.sqlite3")
    corporate_actions = load_oracle_corporate_actions(
        shared_database,
        as_of_date=as_of,
    )
    queue.bind_metadata(
        {
            "as_of_date": as_of,
            "dataset_source_fingerprint_sha256": source_fingerprint["sha256"],
            "oracle_status_sha256": shared_binding.source_fingerprint[
                "oracle_status_sha256"
            ],
            "oracle_feature_snapshot_sha256": oracle_features[
                "snapshot_sha256"
            ],
            "full_candidate_count": len(candidates),
            "selected_candidate_count": len(selection.selected),
            "selection_schema_version": selection.manifest["schema_version"],
            "selection_max_ai_etfs": args.max_ai_etfs,
            "selection_max_ai_stocks": args.max_ai_stocks,
            "oracle_corporate_actions_sha256": corporate_actions["sha256"],
        }
    )
    queue.seed(selection.selected)
    requeued_corporate_actions = (
        queue.requeue_verified_corporate_action_exclusions(
            corporate_actions["events_by_symbol"]
        )
    )

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
            oracle_market_features=str(run_dir / "oracle_market_features.json"),
            selection_manifest=str(run_dir / "selection_manifest.json"),
            coverage_ledger=str(run_dir / "coverage_ledger.jsonl"),
            shared_market_store=shared_binding.public_metadata(),
        )
        write_json(state_path, state)
        return state

    release = load_model_release(args.release_manifest)
    queue.bind_metadata(
        {
            "model_release_manifest_sha256": release.manifest_sha256,
            "model_id": release.model_id,
            "training_native_prompt_contract": TRAINING_NATIVE_PROMPT_CONTRACT,
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
        database=shared_database,
        read_only=True,
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
                packet = adjust_packet_for_verified_corporate_actions(
                    packet,
                    corporate_actions,
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
                queue.mark_error(
                    symbol,
                    f"{type(exc).__name__}: {exc}",
                    trace=_inference_failure_trace(exc),
                )
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
    radar = oracle_market_summary(oracle_features)
    synthesis, synthesis_trace, catalog = synthesize_market_from_native_judgements(
        as_of_date=as_of,
        aggregate=aggregate,
        radar=radar,
        results=results,
    )
    report = {
        "schema_version": "quant.ai_radar_report.v2",
        "as_of_date": as_of,
        "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "scope": "market_and_security_analysis_not_trade_execution",
        "deployment_mode": "shadow" if args.shadow else "reference_publish",
        "full_universe_quantitative_scan_complete": True,
        "full_universe_model_inference_requested": False,
        "selected_model_scope_complete": True,
        "model_release": release.public_metadata(),
        "source_status": {
            "quant_dataset": {
                "status": "confirmed",
                "manifest_sha256": _sha256(dataset_manifest),
                "source_fingerprint": source_fingerprint,
            },
            "shared_oracle_store": shared_binding.public_metadata(),
            "oracle_market_features": {
                "status": "confirmed",
                **oracle_features["binding"],
                "snapshot_sha256": oracle_features["snapshot_sha256"],
            },
            "oracle_corporate_actions": {
                "status": (
                    "confirmed"
                    if corporate_actions["events"]
                    else "no_verified_visible_events"
                ),
                "schema_version": corporate_actions["schema_version"],
                "ledger_sha256": corporate_actions["sha256"],
                "source_row_count": corporate_actions["source_row_count"],
                "event_count": len(corporate_actions["events"]),
                "requeued_exclusion_count": requeued_corporate_actions,
                "events": corporate_actions["events"],
            },
        },
        "universe": universe_manifest,
        "selection": selection.manifest,
        "coverage_ledger_path": str(run_dir / "coverage_ledger.jsonl"),
        "queue_counts": counts,
        "exclusion_counts": queue.exclusions(),
        "aggregate": aggregate,
        "oracle_market": radar,
        "market_judgement": synthesis,
        "market_judgement_trace": synthesis_trace,
        "market_evidence_catalog": catalog,
        "security_judgements_path": str(run_dir / "security_judgements.jsonl"),
    }
    report["market_dashboard"] = build_market_dashboard(aggregate, radar)
    narratives, narrative_trace = build_native_judgement_narratives(
        aggregate=aggregate,
        radar=radar,
        market_judgement=synthesis,
        results=results,
    )
    report["multistage_narratives"] = narratives
    report["multistage_narrative_trace"] = narrative_trace
    report["quality_audit"] = audit_report_quality(
        report=report,
        results=results,
    )
    report_path = run_dir / "market_report.json"
    write_json(report_path, report)
    rendered = render_reports(
        run_dir=run_dir,
        report=report,
        results=results,
        coverage_ledger=selection.coverage_ledger,
    )
    report["rendered_reports"] = rendered
    report["quality_audit"] = audit_report_quality(
        report=report,
        results=results,
    )
    write_json(run_dir / "quality_audit.json", report["quality_audit"])
    rendered = render_reports(
        run_dir=run_dir,
        report=report,
        results=results,
        coverage_ledger=selection.coverage_ledger,
    )
    report["rendered_reports"] = rendered
    write_json(report_path, report)
    latest_path = args.output_root.expanduser().resolve() / "status" / "latest.json"
    quality_green = report["quality_audit"]["status"] == "green"
    if not args.shadow and quality_green:
        write_json(latest_path, report)
    state = _status_document(
        status=(
            "shadow_complete_not_published"
            if args.shadow and quality_green
            else "shadow_quality_failed_not_published"
            if args.shadow
            else "complete"
            if quality_green
            else "quality_failed_not_published"
        ),
        as_of_date=as_of,
        run_dir=run_dir,
        queue=queue,
        inference_started=True,
        production_scope_complete=quality_green,
        production_latest_published=not args.shadow and quality_green,
        report=str(report_path),
        quality_audit=str(run_dir / "quality_audit.json"),
        quality_scores=report["quality_audit"]["scores"],
        quality_failed_categories=report["quality_audit"][
            "failed_categories"
        ],
    )
    write_json(state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--oracle-incremental-database",
        type=Path,
        default=DEFAULT_INCREMENTAL_DATABASE,
    )
    parser.add_argument(
        "--oracle-status",
        type=Path,
        default=DEFAULT_ORACLE_STATUS,
    )
    parser.add_argument(
        "--relation-index", type=Path, default=DEFAULT_RELATION_INDEX
    )
    parser.add_argument(
        "--max-constituent-available-lag-days",
        type=int,
        default=45,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--as-of-date")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="complete and render the selected queue without updating status/latest.json",
    )
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    parser.add_argument("--model-endpoint")
    parser.add_argument("--model-token-file", type=Path)
    parser.add_argument("--secrets-file", type=Path, default=DEFAULT_SECRETS_PATH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-ai-etfs", type=int, default=64)
    parser.add_argument("--max-ai-stocks", type=int, default=192)
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
    if args.max_constituent_available_lag_days < 0:
        parser.error("--max-constituent-available-lag-days must be >= 0")
    if args.max_ai_etfs < 1 or args.max_ai_stocks < 1:
        parser.error("--max-ai-etfs and --max-ai-stocks must be >= 1")
    try:
        result = run(args)
    except (
        InferenceError,
        ModelGateError,
        RadarRunError,
        ResponseContractError,
        RelationIndexError,
        SharedMarketStoreError,
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
    return 0 if result["status"] in SUCCESSFUL_RUN_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
