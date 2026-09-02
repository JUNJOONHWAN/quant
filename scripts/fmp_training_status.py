#!/usr/bin/env python3
"""Report the hard completion gate for the full classified FMP training plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict


def _service_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def build_status(data_root: Path, plan_path: Path, service: str) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    collection_ids = {
        str(item["id"])
        for item in plan["endpoints"]
        if item.get("action") in {"backfill", "snapshot"}
    }
    db_path = data_root / "normalized" / "daily_observations.sqlite3"
    connection = sqlite3.connect(str(db_path), timeout=60)
    connection.row_factory = sqlite3.Row
    run = connection.execute(
        """
        SELECT * FROM fmp_training_runs
        WHERE plan_sha256=? ORDER BY started_at_utc DESC LIMIT 1
        """,
        (plan_sha,),
    ).fetchone()
    checkpoint_counts: Counter[str] = Counter()
    endpoint_statuses: Dict[str, Counter[str]] = defaultdict(Counter)
    job_id = str(run["job_id"]) if run else None
    if job_id:
        for row in connection.execute(
            "SELECT status, scope_json FROM checkpoints WHERE job_id=?",
            (job_id,),
        ):
            status = str(row["status"])
            checkpoint_counts[status] += 1
            try:
                scope = json.loads(row["scope_json"])
                endpoint_id = str(scope.get("endpoint_id") or "unknown")
            except (TypeError, ValueError):
                endpoint_id = "unknown"
            endpoint_statuses[endpoint_id][status] += 1
    not_entitled_evidence = {"valid": 0, "invalid": 0, "invalid_item_keys": []}
    if job_id:
        rows = connection.execute(
            """
            SELECT c.item_key,
                   CASE WHEN r.id IS NOT NULL
                             AND r.source='fmp'
                             AND r.response_status IN (402, 403)
                             AND r.dataset=json_extract(c.scope_json, '$.endpoint_id')
                             AND json_extract(r.request_json, '$.logical_request.endpoint_id')
                                 =json_extract(c.scope_json, '$.endpoint_id')
                             AND json_extract(r.request_json, '$.logical_request.entity_key')
                                 =json_extract(c.scope_json, '$.entity_key')
                        THEN 1 ELSE 0 END evidence_valid
            FROM checkpoints c
            LEFT JOIN raw_artifacts r ON r.id=c.raw_artifact_id
            WHERE c.job_id=? AND c.status='not_entitled'
            """,
            (job_id,),
        ).fetchall()
        invalid_keys = [str(row["item_key"]) for row in rows if not row["evidence_valid"]]
        not_entitled_evidence = {
            "valid": len(rows) - len(invalid_keys),
            "invalid": len(invalid_keys),
            "invalid_item_keys": invalid_keys[:100],
        }
    facts = {
        str(row["endpoint_id"]): int(row["count"])
        for row in connection.execute(
            "SELECT endpoint_id, COUNT(*) count FROM fmp_training_facts GROUP BY endpoint_id"
        )
    }
    connection.close()
    touched_ids = set(endpoint_statuses)
    failed_ids = {
        endpoint_id
        for endpoint_id, counts in endpoint_statuses.items()
        if counts.get("failed", 0)
    }
    active_ids = {
        endpoint_id
        for endpoint_id, counts in endpoint_statuses.items()
        if counts.get("pending", 0) or counts.get("running", 0)
    }
    missing_ids = collection_ids - touched_ids
    run_status = str(run["status"]) if run else "not_started"
    complete = (
        run_status == "complete"
        and not failed_ids
        and not active_ids
        and not missing_ids
        and set(checkpoint_counts).issubset({"done", "not_entitled"})
        and not not_entitled_evidence["invalid"]
    )
    return {
        "overall_complete": complete,
        "plan": {
            "path": str(plan_path),
            "sha256": plan_sha,
            "catalog_endpoints": int(plan["endpoint_count"]),
            "action_counts": plan["action_counts"],
            "collection_endpoints": len(collection_ids),
        },
        "run": dict(run) if run else None,
        "checkpoint_counts": dict(checkpoint_counts),
        "not_entitled_evidence": not_entitled_evidence,
        "endpoint_coverage": {
            "touched": len(touched_ids),
            "missing": len(missing_ids),
            "failed": len(failed_ids),
            "active": len(active_ids),
            "missing_ids": sorted(missing_ids),
            "failed_ids": sorted(failed_ids),
            "active_ids": sorted(active_ids),
        },
        "facts": {
            "rows": sum(facts.values()),
            "endpoints_with_rows": len(facts),
            "by_endpoint": facts,
        },
        "service": {"unit": service, "state": _service_state(service)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--service", default="quant-fmp-training-backfill.service")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    status = build_status(
        args.data_root.expanduser(), args.plan.expanduser(), args.service
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["overall_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
