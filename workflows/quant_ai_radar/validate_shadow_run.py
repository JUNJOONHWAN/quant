#!/usr/bin/env python3
"""Fail-closed acceptance audit for a completed Quant AI Radar shadow run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from workflows.quant_ai_radar.model_runtime import (
    judgement_prohibited_violations,
)
from workflows.quant_ai_radar.universe import write_json


KST = ZoneInfo("Asia/Seoul")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR"
)


class ShadowGateError(RuntimeError):
    """The shadow output does not satisfy an activation prerequisite."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ShadowGateError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _queue_evidence(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        counts = {
            str(row["status"]): int(row["count"])
            for row in connection.execute(
                "SELECT status,COUNT(*) count FROM items GROUP BY status"
            )
        }
        done_hash_failures = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM items
                WHERE status='done' AND (
                    prompt_sha256 IS NULL OR response_sha256 IS NULL OR
                    length(prompt_sha256) != 64 OR length(response_sha256) != 64
                )
                """
            ).fetchone()[0]
        )
        metadata = {
            str(row["key"]): json.loads(str(row["value_json"]))
            for row in connection.execute("SELECT key,value_json FROM run_metadata")
        }
    return {
        "counts": counts,
        "done_hash_failures": done_hash_failures,
        "metadata": metadata,
    }


def _render_evidence(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "rendered_reports_manifest.json"
    manifest = _read_json(manifest_path)
    stored_content_hash = str(manifest.get("content_sha256") or "")
    core = dict(manifest)
    core.pop("content_sha256", None)
    content_hash_valid = (
        SHA256_PATTERN.fullmatch(stored_content_hash) is not None
        and _canonical_hash(core) == stored_content_hash
    )
    artifact_failures = []
    for row in manifest.get("artifacts") or []:
        relative = Path(str(row.get("path") or ""))
        resolved = (run_dir / relative).resolve()
        try:
            resolved.relative_to(run_dir.resolve())
        except ValueError:
            artifact_failures.append(
                {"path": str(relative), "reason": "path_escape"}
            )
            continue
        if not resolved.is_file():
            artifact_failures.append(
                {"path": str(relative), "reason": "missing"}
            )
            continue
        if resolved.stat().st_size != int(row.get("bytes") or -1):
            artifact_failures.append(
                {"path": str(relative), "reason": "byte_count_mismatch"}
            )
        if _sha256(resolved) != str(row.get("sha256") or ""):
            artifact_failures.append(
                {"path": str(relative), "reason": "sha256_mismatch"}
            )
    return {
        "manifest": manifest,
        "content_hash_valid": content_hash_valid,
        "artifact_failures": artifact_failures,
    }


def validate_shadow_run(
    *,
    run_dir: Path,
    latest_path: Path,
    elapsed_seconds: float,
    operating_window_seconds: float,
    expected_latest_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    state = _read_json(root / "run_state.json")
    report = _read_json(root / "market_report.json")
    queue = _queue_evidence(root / "selected_run_queue.sqlite3")
    rendered = _render_evidence(root)
    runtime_path = root / "runtime_readiness_audit.json"
    runtime = _read_json(runtime_path) if runtime_path.is_file() else None
    as_of_date = str(report.get("as_of_date") or "")
    selected_count = int((report.get("selection") or {}).get("selected_count") or 0)
    counts = queue["counts"]
    terminal_count = counts.get("done", 0) + counts.get("excluded", 0)
    blocking_queue_count = sum(
        counts.get(status, 0) for status in ("pending", "running", "error")
    )

    judgement_count = 0
    prohibited = []
    judgement_path = root / "security_judgements.jsonl"
    with judgement_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            judgement_count += 1
            judgement = row.get("judgement") or {}
            facts_as_of = str((judgement.get("facts") or {}).get("as_of_date") or "")
            if facts_as_of != as_of_date:
                prohibited.append(
                    f"line_{line_number}:facts_as_of_mismatch:{facts_as_of}"
                )
            for violation in judgement_prohibited_violations(
                judgement,
                as_of_date,
            ):
                prohibited.append(f"line_{line_number}:{violation}")
    for violation in judgement_prohibited_violations(
        report.get("market_judgement") or {},
        as_of_date,
    ):
        prohibited.append(f"market_judgement:{violation}")

    latest = Path(latest_path).expanduser().resolve()
    if latest.is_file():
        latest_sha256 = _sha256(latest)
        latest_unchanged = (
            expected_latest_sha256 is not None
            and latest_sha256 == expected_latest_sha256
        )
    else:
        latest_sha256 = None
        latest_unchanged = expected_latest_sha256 is None

    source_status = report.get("source_status") or {}
    shared_oracle = source_status.get("shared_oracle_store") or {}
    source_hashes = [
        (source_status.get("quant_dataset") or {}).get("manifest_sha256"),
        ((source_status.get("quant_dataset") or {}).get("source_fingerprint") or {}).get(
            "sha256"
        ),
        shared_oracle.get("source_fingerprint_sha256"),
        (source_status.get("oracle_market_features") or {}).get(
            "snapshot_sha256"
        ),
    ]
    source_hashes_valid = all(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value)
        for value in source_hashes
    )

    gates = {
        "shadow_state": (
            state.get("status") == "shadow_complete_not_published"
            and state.get("production_scope_complete") is True
            and state.get("production_latest_published") is False
        ),
        "shadow_report_mode": (
            report.get("deployment_mode") == "shadow"
            and report.get("full_universe_quantitative_scan_complete") is True
            and report.get("selected_model_scope_complete") is True
        ),
        "queue_complete": (
            blocking_queue_count == 0
            and terminal_count == selected_count
            and queue["done_hash_failures"] == 0
        ),
        "judgements_complete": judgement_count == counts.get("done", 0),
        "no_lookahead": not prohibited,
        "source_hashes_valid": source_hashes_valid,
        "render_manifest_valid": (
            rendered["content_hash_valid"]
            and not rendered["artifact_failures"]
            and int(
                rendered["manifest"].get("security_report_count") or -1
            )
            == counts.get("done", 0)
        ),
        "production_latest_unchanged": latest_unchanged,
        "capacity_inside_operating_window": (
            elapsed_seconds > 0
            and operating_window_seconds > 0
            and elapsed_seconds <= operating_window_seconds
        ),
    }
    status = "pass" if all(gates.values()) else "fail"
    runtime_manual_ready = bool(
        runtime and runtime.get("manual_reference_ready") is True
    )
    runtime_timer_eligible = bool(
        runtime and runtime.get("timer_activation_eligible") is True
    )
    audit = {
        "schema_version": "quant.ai_radar_shadow_gate.v1",
        "status": status,
        "as_of_date": as_of_date,
        "run_dir": str(root),
        "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "gates": gates,
        "queue": queue,
        "judgement_count": judgement_count,
        "prohibited_violations": prohibited,
        "rendered_reports": {
            "content_hash_valid": rendered["content_hash_valid"],
            "artifact_failures": rendered["artifact_failures"],
            "security_report_count": rendered["manifest"].get(
                "security_report_count"
            ),
        },
        "production_latest": {
            "path": str(latest),
            "exists": latest.is_file(),
            "sha256": latest_sha256,
            "expected_sha256": expected_latest_sha256,
            "unchanged": latest_unchanged,
        },
        "capacity": {
            "elapsed_seconds": elapsed_seconds,
            "operating_window_seconds": operating_window_seconds,
            "inside_window": gates["capacity_inside_operating_window"],
        },
        "runtime_readiness": {
            "path": str(runtime_path),
            "exists": runtime is not None,
            "status": runtime.get("status") if runtime else "missing",
            "manual_reference_ready": runtime_manual_ready,
            "timer_activation_eligible": runtime_timer_eligible,
            "watchpoints": runtime.get("watchpoints") if runtime else [],
        },
        "activation_policy": {
            "this_shadow_counts_toward_required_consecutive_runs": (
                status == "pass" and runtime_timer_eligible
            ),
            "timer_activation_still_requires_five_consecutive_daily_shadows": True,
            "timer_activation_still_requires_explicit_user_approval": True,
        },
    }
    write_json(root / "shadow_gate_audit.json", audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--latest-path",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "status" / "latest.json",
    )
    parser.add_argument("--expected-latest-sha256")
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument(
        "--operating-window-seconds",
        type=float,
        default=21600.0,
        help="default six-hour reference-analysis operating window",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        audit = validate_shadow_run(
            run_dir=args.run_dir,
            latest_path=args.latest_path,
            expected_latest_sha256=args.expected_latest_sha256,
            elapsed_seconds=args.elapsed_seconds,
            operating_window_seconds=args.operating_window_seconds,
        )
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
        return 2
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
