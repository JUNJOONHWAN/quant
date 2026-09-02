#!/usr/bin/env python3
"""Report FMP dataset progress and pause its Hermes cron at the hard 100% gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")
DEFAULT_QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")
DEFAULT_DATA_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET"
)
DEFAULT_PLAN = DEFAULT_DATA_ROOT / "state" / "fmp_training_plan_20260715.json"
DEFAULT_SYMBOLS = (
    DEFAULT_DATA_ROOT
    / "state"
    / "universe"
    / "fmp_us_equity_etf_20260714.symbols.txt"
)
DEFAULT_UNIVERSE = (
    DEFAULT_DATA_ROOT
    / "state"
    / "universe"
    / "fmp_us_all_20260714.jsonl"
)
DEFAULT_STATE = DEFAULT_DATA_ROOT / "state" / "fmp_training_progress_notify.json"
DEFAULT_HERMES = Path("/home/zooh/.local/bin/hermes")
DEFAULT_CRON_JOB_NAME = "quant-fmp-training-download-progress"
COLLECTION_ACTIONS = {"backfill", "snapshot"}


def _read_symbols(path: Path) -> set[str]:
    return {
        line.strip().upper()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }


def _read_etfs(path: Path) -> set[str]:
    values = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("is_etf") is True:
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol:
                    values.add(symbol)
    return values


def _discovered_dimension_count(
    connection: sqlite3.Connection,
    endpoint_id: str,
    keys: Sequence[str],
) -> int:
    values = set()
    for item in connection.execute(
        "SELECT row_json FROM fmp_training_facts WHERE endpoint_id=?",
        (endpoint_id,),
    ):
        try:
            row = json.loads(item[0])
        except (TypeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                values.add(str(value).strip())
                break
    return len({value for value in values if value})


def estimate_expected_work_items(
    plan: Mapping[str, Any],
    symbol_count: int,
    etf_count: int,
    start_date: str,
    end_date: str,
    connection: sqlite3.Connection,
) -> dict:
    """Count the same logical dimension x variant x window items as backfill."""

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be <= end_date")
    discovered_cache: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    by_endpoint: Dict[str, int] = {}
    provisional_dimensions = []

    for endpoint in plan.get("endpoints", []):
        if endpoint.get("action") not in COLLECTION_ACTIONS:
            continue
        endpoint_id = str(endpoint["id"])
        collection = dict(endpoint.get("collection") or {})
        mode = str(collection.get("mode") or "global")
        if mode == "global":
            dimension_count = 1
        elif mode == "per_symbol":
            dimension_count = symbol_count
        elif mode == "per_etf":
            dimension_count = etf_count
        elif mode == "per_value":
            dimension_count = len(collection.get("values") or [])
        elif mode == "per_discovered":
            source = str(collection["source_endpoint_id"])
            keys = tuple(
                str(key) for key in collection.get("source_keys", ["symbol"])
            )
            cache_key = (source, keys)
            if cache_key not in discovered_cache:
                discovered_cache[cache_key] = _discovered_dimension_count(
                    connection, source, keys
                )
            dimension_count = discovered_cache[cache_key]
            if dimension_count == 0:
                provisional_dimensions.append(
                    {
                        "endpoint_id": endpoint_id,
                        "source_endpoint_id": source,
                        "source_keys": list(keys),
                    }
                )
        else:
            raise ValueError(
                "unsupported FMP training collection mode: {}".format(mode)
            )

        variant_count = len(collection.get("variants") or []) or 1
        window_count = (
            end.year - start.year + 1
            if collection.get("date_windows") == "year"
            else 1
        )
        by_endpoint[endpoint_id] = dimension_count * variant_count * window_count

    return {
        "total": sum(by_endpoint.values()),
        "by_endpoint": by_endpoint,
        "provisional_dimensions": provisional_dimensions,
        "symbol_count": symbol_count,
        "etf_count": etf_count,
    }


def _run_json(command: Sequence[str]) -> dict:
    result = subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "status command failed rc={} stderr={}".format(
                result.returncode, result.stderr.strip()[-500:]
            )
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("status command returned non-object JSON")
    return payload


def _checkpoint_endpoint_counts(
    connection: sqlite3.Connection, job_id: str | None
) -> Dict[str, Counter[str]]:
    result: Dict[str, Counter[str]] = defaultdict(Counter)
    if not job_id:
        return result
    for row in connection.execute(
        "SELECT status, scope_json FROM checkpoints WHERE job_id=?", (job_id,)
    ):
        try:
            scope = json.loads(row[1])
            endpoint_id = str(scope.get("endpoint_id") or "unknown")
        except (TypeError, ValueError):
            endpoint_id = "unknown"
        result[endpoint_id][str(row[0])] += 1
    return result


def _checkpoint_health(
    connection: sqlite3.Connection, job_id: str | None, now: datetime
) -> dict:
    failure_groups: Counter[Tuple[str, str]] = Counter()
    active = None
    last_checkpoint_utc = None
    if job_id:
        for row in connection.execute(
            """
            SELECT json_extract(scope_json, '$.endpoint_id') endpoint_id,
                   last_error
            FROM checkpoints WHERE job_id=? AND status='failed'
            """,
            (job_id,),
        ):
            endpoint_id = str(row[0] or "unknown")
            match = re.search(r"HTTP\s+(\d{3})", str(row[1] or ""))
            failure_groups[(endpoint_id, match.group(1) if match else "other")] += 1
        row = connection.execute(
            """
            SELECT json_extract(scope_json, '$.endpoint_id') endpoint_id,
                   item_key, updated_at_utc
            FROM checkpoints
            WHERE job_id=? AND status='running'
            ORDER BY updated_at_utc DESC LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        if row:
            active = {
                "endpoint_id": str(row[0] or "unknown"),
                "item_key": str(row[1]),
                "updated_at_utc": str(row[2]),
            }
        latest = connection.execute(
            "SELECT MAX(updated_at_utc) FROM checkpoints WHERE job_id=?", (job_id,)
        ).fetchone()
        last_checkpoint_utc = str(latest[0]) if latest and latest[0] else None

    age_seconds = None
    if last_checkpoint_utc:
        parsed = datetime.fromisoformat(last_checkpoint_utc.replace("Z", "+00:00"))
        age_seconds = max(0, int((now - parsed.astimezone(KST)).total_seconds()))
    groups = [
        {"endpoint_id": endpoint, "http_status": status, "count": count}
        for (endpoint, status), count in failure_groups.most_common()
    ]
    return {
        "failure_groups": groups,
        "active_checkpoint": active,
        "last_checkpoint_utc": last_checkpoint_utc,
        "last_checkpoint_age_seconds": age_seconds,
    }


def _df_used_percent(used_bytes: int, free_bytes: int) -> float:
    available_total = used_bytes + free_bytes
    return used_bytes * 100.0 / available_total if available_total else 0.0


def _disk_health(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    percent = _df_used_percent(usage.used, usage.free)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_percent": percent,
    }


def _alert_state(progress: Mapping[str, Any]) -> dict:
    issues = []
    severity = "ok"
    failed = int(progress["work_items"].get("failed", 0))
    disk = progress.get("disk") or {}
    service = progress.get("service") or {}
    checkpoint = progress.get("checkpoint_health") or {}
    age = checkpoint.get("last_checkpoint_age_seconds")
    if failed:
        issues.append("checkpoint_failures")
        severity = "warning"
    if float(disk.get("used_percent", 0.0)) >= 85.0:
        issues.append("disk_usage_high")
        severity = "warning"
    if int(disk.get("free_bytes", 0)) < 50 * 1024**3:
        issues.append("disk_free_critical")
        severity = "critical"
    if not progress.get("final_complete") and service.get("state") != "active":
        issues.append("download_service_inactive")
        severity = "critical"
    if age is not None and int(age) >= 1800 and service.get("state") == "active":
        issues.append("checkpoint_stalled_30m")
        severity = "critical"
    return {"severity": severity, "issues": issues}


def _eta_projection(progress: Mapping[str, Any]) -> dict:
    """Project logical-loop completion from the run's observed lifetime rate."""

    run = progress.get("fmp_run") or {}
    started_text = run.get("started_at_utc")
    checked_text = progress.get("checked_at_kst")
    work = progress.get("work_items") or {}
    processed = int(work.get("processed", 0))
    expected = int(work.get("expected", 0))
    if not started_text or not checked_text or processed <= 0 or expected <= 0:
        return {
            "available": False,
            "basis": "logical_work_items_lifetime_rate",
            "reason": "insufficient_progress_history",
        }
    started = datetime.fromisoformat(str(started_text).replace("Z", "+00:00"))
    checked = datetime.fromisoformat(str(checked_text).replace("Z", "+00:00"))
    elapsed_seconds = max(0.0, (checked - started.astimezone(KST)).total_seconds())
    if elapsed_seconds <= 0:
        return {
            "available": False,
            "basis": "logical_work_items_lifetime_rate",
            "reason": "non_positive_elapsed_time",
        }
    rate = processed / elapsed_seconds
    remaining = max(0, expected - processed)
    remaining_seconds = remaining / rate if rate > 0 else None
    projected = checked + timedelta(seconds=remaining_seconds or 0)
    failed = int(work.get("failed", 0))
    provisional = int(progress.get("provisional_dimension_count", 0))
    blockers = []
    if failed:
        blockers.append("failed_checkpoints")
    if provisional:
        blockers.append("dynamic_dimensions_pending")
    return {
        "available": True,
        "basis": "logical_work_items_lifetime_rate",
        "rate_items_per_second": rate,
        "processed_items": processed,
        "remaining_items": remaining,
        "remaining_seconds": remaining_seconds,
        "collection_loop_eta_kst": projected.isoformat(timespec="minutes"),
        "final_gate_eta_kst": projected.isoformat(timespec="minutes") if not blockers else None,
        "final_gate_blockers": blockers,
        "confidence": "low" if blockers else "medium",
    }


def calculate_percent(done_items: int, expected_items: int, complete: bool) -> float:
    if complete:
        return 100.0
    if expected_items <= 0:
        return 0.0
    return min(99.99, max(0.0, done_items * 100.0 / expected_items))


def build_progress(args: argparse.Namespace) -> dict:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    symbols = _read_symbols(args.symbols_file)
    etfs = _read_etfs(args.universe_jsonl)
    database_path = args.data_root / "normalized" / "daily_observations.sqlite3"

    fmp_status = _run_json(
        [
            str(args.python),
            str(args.quant_root / "scripts" / "fmp_training_status.py"),
            "--data-root",
            str(args.data_root),
            "--plan",
            str(args.plan),
            "--service",
            args.service,
        ]
    )
    base_status = _run_json(
        [
            str(args.python),
            str(args.quant_root / "scripts" / "quant_dataset_completion_status.py"),
            "--data-root",
            str(args.data_root),
        ]
    )

    now = datetime.now(KST)
    connection = sqlite3.connect(str(database_path), timeout=60)
    connection.execute("PRAGMA busy_timeout=60000")
    try:
        # Keep checkpoint totals and failure groups on one WAL read snapshot while
        # the backfill process continues writing several records per second.
        connection.execute("BEGIN")
        estimate = estimate_expected_work_items(
            plan,
            len(symbols),
            len(etfs),
            args.start_date,
            args.end_date,
            connection,
        )
        run = fmp_status.get("run") or {}
        endpoint_counts = _checkpoint_endpoint_counts(
            connection, str(run.get("job_id")) if run.get("job_id") else None
        )
        checkpoint_health = _checkpoint_health(
            connection,
            str(run.get("job_id")) if run.get("job_id") else None,
            now,
        )
    finally:
        connection.close()

    checkpoint_counts: Counter[str] = Counter()
    for counts in endpoint_counts.values():
        checkpoint_counts.update(counts)
    done_items = int(checkpoint_counts.get("done", 0))
    not_entitled_items = int(checkpoint_counts.get("not_entitled", 0))
    failed_items = int(checkpoint_counts.get("failed", 0))
    expected_items = int(estimate["total"])
    fmp_complete = bool(fmp_status.get("overall_complete"))
    base_complete = bool(base_status.get("overall_complete"))
    final_complete = fmp_complete and base_complete
    completed_endpoints = sum(
        1
        for endpoint_id, expected in estimate["by_endpoint"].items()
        if expected > 0
        and (
            endpoint_counts.get(endpoint_id, {}).get("done", 0)
            + endpoint_counts.get(endpoint_id, {}).get("not_entitled", 0)
        ) >= expected
        and not endpoint_counts.get(endpoint_id, {}).get("failed", 0)
        and not endpoint_counts.get(endpoint_id, {}).get("running", 0)
        and not endpoint_counts.get(endpoint_id, {}).get("pending", 0)
    )
    collection_ids = set(estimate["by_endpoint"])
    touched_ids = set(endpoint_counts)
    active_ids = {
        endpoint_id
        for endpoint_id, counts in endpoint_counts.items()
        if counts.get("running", 0) or counts.get("pending", 0)
    }
    facts = fmp_status.get("facts") or {}
    progress = {
        "checked_at_kst": now.isoformat(timespec="seconds"),
        "percent": calculate_percent(done_items, expected_items, final_complete),
        "percent_basis": "logical_work_items_dynamic_estimate",
        "final_complete": final_complete,
        "base_dataset_complete": base_complete,
        "fmp_training_complete": fmp_complete,
        "work_items": {
            "done": done_items,
            "expected": expected_items,
            "not_entitled": not_entitled_items,
            "failed": failed_items,
            "processed": done_items + not_entitled_items + failed_items,
            "processed_percent": calculate_percent(
                done_items + not_entitled_items + failed_items,
                expected_items,
                final_complete,
            ),
        },
        "endpoints": {
            "completed": completed_endpoints,
            "touched": len(touched_ids),
            "collection_total": len(estimate["by_endpoint"]),
            "missing": len(collection_ids - touched_ids),
            "active": len(active_ids),
        },
        "facts": {
            "rows": int(facts.get("rows", 0)),
            "endpoints_with_rows": int(facts.get("endpoints_with_rows", 0)),
        },
        "universe": {
            "symbols": len(symbols),
            "etfs": len(etfs),
            "from": args.start_date,
            "to": args.end_date,
        },
        "provisional_dimension_count": len(estimate["provisional_dimensions"]),
        "provisional_dimensions": estimate["provisional_dimensions"],
        "service": fmp_status.get("service") or {},
        "fmp_run": fmp_status.get("run"),
        "checkpoint_health": checkpoint_health,
        "disk": _disk_health(args.data_root),
    }
    progress["eta"] = _eta_projection(progress)
    progress["alert"] = _alert_state(progress)
    return progress


def format_message(progress: Mapping[str, Any]) -> str:
    work = progress["work_items"]
    endpoints = progress["endpoints"]
    facts = progress["facts"]
    service = progress.get("service") or {}
    percent = float(progress["percent"])
    alert = progress.get("alert") or {}
    disk = progress.get("disk") or {}
    checkpoint = progress.get("checkpoint_health") or {}
    failure_groups = checkpoint.get("failure_groups") or []
    eta = progress.get("eta") or {}
    failure_text = ""
    if failure_groups:
        first = failure_groups[0]
        failure_text = " {} HTTP{} {:,}건".format(
            first["endpoint_id"], first["http_status"], int(first["count"])
        )
    common = (
        "성공 {:.2f}% ({:,}/{:,}) | 처리 {:.2f}% | endpoint 완료 {}/{} (접촉 {}) | "
        "facts {:,}행 | 권한종결 {:,} | 실패 {:,}{} | 디스크 {:.1f}%·여유 {:.1f}GB | 서비스 {} | {}"
    ).format(
        percent,
        int(work["done"]),
        int(work["expected"]),
        float(work.get("processed_percent", percent)),
        int(endpoints["completed"]),
        int(endpoints["collection_total"]),
        int(endpoints["touched"]),
        int(facts["rows"]),
        int(work.get("not_entitled", 0)),
        int(work["failed"]),
        failure_text,
        float(disk.get("used_percent", 0.0)),
        int(disk.get("free_bytes", 0)) / 1024**3,
        service.get("state", "unknown"),
        progress["checked_at_kst"],
    )
    if progress["final_complete"]:
        return "✅ [FMP 학습데이터 다운로드 완료] " + common
    provisional = int(progress.get("provisional_dimension_count", 0))
    suffix = " | 동적 분모 발견 중 {}개".format(provisional) if provisional else ""
    prefix = {
        "critical": "🚨 [FMP 학습데이터 다운로드 장애] ",
        "warning": "⚠️ [FMP 학습데이터 다운로드 경고] ",
    }.get(str(alert.get("severity")), "[FMP 학습데이터 다운로드] ")
    issues = alert.get("issues") or []
    issue_suffix = " | 감시 {}".format(",".join(issues)) if issues else ""
    eta_suffix = ""
    if eta.get("available"):
        eta_text = str(eta.get("collection_loop_eta_kst") or "unknown")
        eta_suffix = " | 수집루프 ETA {}".format(eta_text)
        if not eta.get("final_gate_eta_kst"):
            eta_suffix += "·100% ETA 없음({})".format(
                ",".join(eta.get("final_gate_blockers") or ["blocked"])
            )
    return prefix + common + suffix + eta_suffix + issue_suffix


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _pause_cron(hermes: Path, job_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(hermes), "cron", "pause", job_name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-root", type=Path, default=DEFAULT_QUANT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--universe-jsonl", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--python", type=Path, default=Path("/usr/bin/python3"))
    parser.add_argument("--hermes", type=Path, default=DEFAULT_HERMES)
    parser.add_argument("--cron-job-name", default=DEFAULT_CRON_JOB_NAME)
    parser.add_argument("--service", default="quant-fmp-training-backfill.service")
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default="2026-07-14")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        progress = build_progress(args)
    except Exception as error:
        print(
            "⚠️ [FMP 학습데이터 진행률 실패] {}: {}".format(
                type(error).__name__, str(error)
            )
        )
        return 2

    if args.json:
        print(json.dumps(progress, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    state = dict(progress)
    if progress["final_complete"] and not args.dry_run:
        pause = _pause_cron(args.hermes, args.cron_job_name)
        if pause.returncode != 0:
            state["cron_pause"] = {
                "ok": False,
                "returncode": pause.returncode,
                "stderr": pause.stderr.strip()[-500:],
            }
            _atomic_write_json(args.state_file, state)
            print(
                format_message(progress)
                + " | ⚠️ Hermes 크론 정지 실패: {}".format(
                    pause.stderr.strip()[-300:] or "unknown error"
                )
            )
            return 2
        state["cron_pause"] = {"ok": True, "paused_at_kst": progress["checked_at_kst"]}
        state["completion_notified_at_kst"] = progress["checked_at_kst"]
        _atomic_write_json(args.state_file, state)
        print(format_message(progress) + " | Hermes 크론 자동 정지")
        return 0

    if not args.dry_run:
        _atomic_write_json(args.state_file, state)
    print(format_message(progress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
