"""Report resumable selected-pair materialization progress and KST ETA."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence
from zoneinfo import ZoneInfo


DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/"
    "qwen3_8b_candidate_v2"
)
KST = ZoneInfo("Asia/Seoul")


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _iso(value: Optional[datetime], zone: timezone = timezone.utc) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(zone).isoformat()


def inspect_status(output_root: Path) -> dict:
    output = Path(output_root).expanduser().resolve()
    state_path = output / "materialization_state.sqlite3"
    manifest_path = output / "manifest.json"
    now = datetime.now(timezone.utc)
    result = {
        "schema_version": "quant.materialization_status.v1",
        "output_root": str(output),
        "checked_at_utc": _iso(now),
        "checked_at_kst": _iso(now, KST),
        "state_exists": state_path.is_file(),
        "manifest_exists": manifest_path.is_file(),
        "complete": manifest_path.is_file(),
    }
    if not state_path.is_file():
        result.update({"run_status": "not_started", "errors": ["state database missing"]})
        return result

    connection = sqlite3.connect("file:{}?mode=ro".format(state_path), uri=True)
    try:
        metadata = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
        processed = int(connection.execute("SELECT COUNT(*) FROM pair_results").fetchone()[0])
        examples = int(connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0])
        status_counts = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT status,COUNT(*) FROM pair_results GROUP BY status"
            )
        }
        split_task_counts = defaultdict(dict)
        for split, task, count in connection.execute(
            "SELECT split,task_type,COUNT(*) FROM examples GROUP BY split,task_type"
        ):
            split_task_counts[str(split)][str(task)] = int(count)
    finally:
        connection.close()

    expected = None
    pair_manifest = None
    contract = metadata.get("contract_sha256")
    if manifest_path.is_file():
        completed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pair_manifest = ((completed_manifest.get("input_pair_selection") or {}).get("manifest"))
        expected = ((completed_manifest.get("input_pair_selection") or {}).get("selected_pairs"))
    else:
        # The state intentionally stores only the contract digest.  Locate the pair
        # manifest through the output's sibling selection roots only when a caller
        # supplies it via metadata in newer runs.
        pair_manifest = metadata.get("pair_manifest")
        expected_text = metadata.get("expected_pairs")
        expected = int(expected_text) if expected_text else None

    started = _parse_time(metadata.get("started_at_utc"))
    last_progress = _parse_time(metadata.get("last_progress_at_utc"))
    seed_pair_count = int(metadata.get("seed_pair_count") or 0)
    extension_processed = max(0, processed - seed_pair_count)
    extension_expected = (
        max(0, int(expected) - seed_pair_count) if expected is not None else None
    )
    rate = None
    eta = None
    remaining = None
    progress_percent = None
    if expected is not None:
        expected = int(expected)
        remaining = max(0, expected - processed)
        progress_percent = round((processed / expected * 100.0) if expected else 100.0, 4)
    rate_numerator = extension_processed if seed_pair_count else processed
    if started and rate_numerator > 0:
        elapsed_hours = max((now - started).total_seconds() / 3600.0, 1.0 / 3600.0)
        rate = rate_numerator / elapsed_hours
        if remaining is not None and rate > 0:
            eta = now + timedelta(hours=remaining / rate)

    run_status = metadata.get("run_status")
    if manifest_path.is_file():
        run_status = "complete"
    elif status_counts.get("error"):
        run_status = "error"
    elif not run_status:
        run_status = "running_or_interrupted"
    stale_seconds = None
    if last_progress:
        stale_seconds = max(0, int((now - last_progress).total_seconds()))

    result.update(
        {
            "run_status": run_status,
            "contract_sha256": contract,
            "pair_manifest": pair_manifest,
            "expected_pairs": expected,
            "processed_pairs": processed,
            "remaining_pairs": remaining,
            "progress_percent": progress_percent,
            "seed_pair_count": seed_pair_count,
            "extension_expected_pairs": extension_expected,
            "extension_processed_pairs": extension_processed,
            "extension_progress_percent": (
                round(extension_processed / extension_expected * 100.0, 4)
                if extension_expected
                else None
            ),
            "pair_status_counts": status_counts,
            "materialized_examples": examples,
            "actual_task_type_counts": dict(split_task_counts),
            "started_at_utc": _iso(started),
            "started_at_kst": _iso(started, KST),
            "last_progress_at_utc": _iso(last_progress),
            "last_progress_at_kst": _iso(last_progress, KST),
            "seconds_since_progress": stale_seconds,
            "throughput_pairs_per_hour": round(rate, 2) if rate is not None else None,
            "eta_utc": _iso(eta),
            "eta_kst": _iso(eta, KST),
            "last_as_of_date": metadata.get("last_as_of_date"),
            "last_error": metadata.get("last_error"),
            "errors": [] if not status_counts.get("error") else ["pair materialization error"],
        }
    )
    return result


def _write_atomic(path: Path, document: dict) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".materialization-status-", dir=str(target.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_jsonl(path: Path, document: dict) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--append-jsonl", type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args(argv)
    result = inspect_status(args.output_root)
    if args.output:
        _write_atomic(args.output, result)
    if args.append_jsonl:
        _append_jsonl(args.append_jsonl, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 1 if args.fail_on_error and result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
