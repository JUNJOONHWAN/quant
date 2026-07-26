#!/usr/bin/env python3
"""Reclassify source-proven FMP 402/403 failures as terminal not_entitled."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


ERROR_PATTERN = re.compile(r"HTTP\s+(402|403)\s+\(raw artifact id=(\d+)\)")


def _audit(connection: sqlite3.Connection, job_id: str) -> Tuple[dict, List[tuple]]:
    connection.row_factory = sqlite3.Row
    failed = connection.execute(
        """
        SELECT item_key, scope_json, last_error
        FROM checkpoints
        WHERE job_id=? AND source='fmp_training' AND status='failed'
        ORDER BY item_key
        """,
        (job_id,),
    ).fetchall()
    candidates: List[tuple] = []
    invalid: Counter[str] = Counter()
    invalid_samples: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    endpoint_counts: Counter[str] = Counter()
    for row in failed:
        item_key = str(row["item_key"])
        match = ERROR_PATTERN.search(str(row["last_error"] or ""))
        if not match:
            invalid["error_not_terminal_http"] += 1
            if len(invalid_samples) < 20:
                invalid_samples.append({"item_key": item_key, "reason": "error_not_terminal_http"})
            continue
        expected_status = int(match.group(1))
        artifact_id = int(match.group(2))
        try:
            scope = json.loads(row["scope_json"])
        except (TypeError, ValueError):
            invalid["invalid_scope_json"] += 1
            if len(invalid_samples) < 20:
                invalid_samples.append({"item_key": item_key, "reason": "invalid_scope_json"})
            continue
        artifact = connection.execute(
            """
            SELECT source, dataset, response_status, request_json
            FROM raw_artifacts WHERE id=?
            """,
            (artifact_id,),
        ).fetchone()
        reason = None
        request = None
        if artifact is None:
            reason = "raw_artifact_missing"
        else:
            try:
                request = json.loads(artifact["request_json"])
            except (TypeError, ValueError):
                reason = "invalid_request_json"
        endpoint_id = str(scope.get("endpoint_id") or "")
        entity_key = str(scope.get("entity_key") or "")
        logical = (request or {}).get("logical_request") or {}
        if reason is None and (
            str(artifact["source"]) != "fmp"
            or int(artifact["response_status"]) != expected_status
            or expected_status not in {402, 403}
            or str(artifact["dataset"]) != endpoint_id
        ):
            reason = "artifact_response_mismatch"
        if reason is None and (
            str(logical.get("endpoint_id") or "") != endpoint_id
            or str(logical.get("entity_key") or "") != entity_key
        ):
            reason = "logical_request_mismatch"
        if reason is not None:
            invalid[reason] += 1
            if len(invalid_samples) < 20:
                invalid_samples.append({"item_key": item_key, "reason": reason})
            continue
        candidates.append((artifact_id, item_key))
        status_counts[str(expected_status)] += 1
        endpoint_counts[endpoint_id] += 1
    return (
        {
            "job_id": job_id,
            "failed_rows": len(failed),
            "eligible": len(candidates),
            "invalid": sum(invalid.values()),
            "invalid_reasons": dict(sorted(invalid.items())),
            "invalid_samples": invalid_samples,
            "http_status_counts": dict(sorted(status_counts.items())),
            "endpoint_counts": dict(sorted(endpoint_counts.items())),
        },
        candidates,
    )


def reconcile(database_path: Path, job_id: str, apply: bool = False) -> dict:
    connection = sqlite3.connect(str(database_path), timeout=60)
    connection.execute("PRAGMA busy_timeout=60000")
    try:
        audit, candidates = _audit(connection, job_id)
        audit["mode"] = "apply" if apply else "dry_run"
        audit["updated"] = 0
        if apply:
            if audit["invalid"]:
                audit["apply_refused"] = "invalid_failed_checkpoints_present"
                return audit
            now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            connection.execute("BEGIN IMMEDIATE")
            updated = 0
            for artifact_id, item_key in candidates:
                cursor = connection.execute(
                    """
                    UPDATE checkpoints
                    SET status='not_entitled', raw_artifact_id=?, observation_count=0,
                        updated_at_utc=?
                    WHERE job_id=? AND source='fmp_training' AND item_key=?
                      AND status='failed'
                    """,
                    (artifact_id, now, job_id, item_key),
                )
                updated += cursor.rowcount
            connection.commit()
            audit["updated"] = updated
            audit["apply_refused"] = None
        return audit
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = reconcile(args.database.expanduser(), args.job_id, args.apply)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if result.get("invalid") or result.get("updated", 0) != (
        result.get("eligible", 0) if args.apply else 0
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
