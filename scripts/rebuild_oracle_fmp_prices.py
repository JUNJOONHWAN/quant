#!/usr/bin/env python3
"""Rebuild post-training Oracle stock/ETF daily prices from FMP only."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_dataset.shared_market import (
    DEFAULT_BASE_DATABASE,
    DEFAULT_INCREMENTAL_DATABASE,
)
from workflows.market_structure_oracle.incremental_store import (
    STATUS_FILE,
    ensure_oracle_snapshot,
)


DEFAULT_INCREMENTAL_ROOT = DEFAULT_INCREMENTAL_DATABASE.parent.parent


def _backup_database(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_connection:
        with sqlite3.connect(target) as target_connection:
            source_connection.backup(target_connection)


def _capture_daily_job_ids(
    connection: sqlite3.Connection, start_date: str
) -> list[str]:
    result: list[str] = []
    for job_id, contract_json in connection.execute(
        "SELECT job_id,contract_json FROM jobs WHERE job_type='capture_daily'"
    ):
        try:
            contract = json.loads(str(contract_json))
        except json.JSONDecodeError:
            continue
        trade_date = str(contract.get("date") or "")
        if trade_date >= start_date:
            result.append(str(job_id))
    return result


def _rows_by_source(
    connection: sqlite3.Connection, start_date: str
) -> dict[str, int]:
    return {
        str(source): int(count)
        for source, count in connection.execute(
            """
            SELECT source,COUNT(*) FROM daily_observations
            WHERE trade_date>=? GROUP BY source ORDER BY source
            """,
            (start_date,),
        )
    }


def reset_price_projection(
    *,
    database_path: Path,
    start_date: str,
    backup_root: Path,
) -> dict[str, Any]:
    backup_database = backup_root / database_path.name
    _backup_database(database_path, backup_database)
    with sqlite3.connect(database_path) as connection:
        before = _rows_by_source(connection, start_date)
        job_ids = _capture_daily_job_ids(connection, start_date)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "DELETE FROM quality_checks WHERE trade_date>=?", (start_date,)
        )
        connection.execute(
            """
            DELETE FROM daily_observation_versions
            WHERE trade_date>=? AND source IN ('fmp','massive')
            """,
            (start_date,),
        )
        connection.execute(
            """
            DELETE FROM daily_observations
            WHERE trade_date>=? AND source IN ('fmp','massive')
            """,
            (start_date,),
        )
        for job_id in job_ids:
            connection.execute(
                "DELETE FROM checkpoints WHERE job_id=?", (job_id,)
            )
            connection.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        connection.execute(
            "DELETE FROM oracle_snapshot_seals WHERE target_as_of_date>=?",
            (start_date,),
        )
        connection.execute(
            "DELETE FROM oracle_incremental_runs WHERE target_as_of_date>=?",
            (start_date,),
        )
        connection.commit()
        after = _rows_by_source(connection, start_date)
        raw_count = int(
            connection.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone()[0]
        )
    return {
        "backup_database": str(backup_database),
        "price_rows_before_by_source": before,
        "price_rows_after_by_source": after,
        "capture_daily_jobs_removed": len(job_ids),
        "raw_artifacts_preserved": raw_count,
    }


def _backup_derived_state(
    incremental_root: Path, backup_root: Path, start_date: str
) -> list[str]:
    del start_date
    copied: list[str] = []
    candidates = [incremental_root / STATUS_FILE]
    for directory in (
        incremental_root / "state" / "daily_coverage",
        incremental_root / "state" / "active_universe",
    ):
        if directory.is_dir():
            candidates.extend(path for path in directory.iterdir() if path.is_file())
    for path in sorted(set(candidates)):
        if not path.is_file():
            continue
        relative = path.relative_to(incremental_root)
        target = backup_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(path))
        if path != incremental_root / STATUS_FILE:
            path.unlink()
    return copied


def rebuild(args: argparse.Namespace) -> dict[str, Any]:
    incremental_root = args.incremental_root.expanduser().resolve()
    database_path = incremental_root / "normalized" / "daily_observations.sqlite3"
    if database_path != args.incremental_database.expanduser().resolve():
        raise RuntimeError(
            "--incremental-database must belong to --incremental-root"
        )
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (
        incremental_root / "state" / "rebuild_backups" / stamp
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    state_backup = _backup_derived_state(
        incremental_root, backup_root, args.start_date
    )
    reset = reset_price_projection(
        database_path=database_path,
        start_date=args.start_date,
        backup_root=backup_root,
    )
    ensured = ensure_oracle_snapshot(
        base_database=args.base_database.expanduser().resolve(),
        incremental_root=incremental_root,
        target_as_of_date=args.target_as_of_date,
        force_repair=True,
        constituent_refresh_max_etfs=args.constituent_refresh_max_etfs,
    )
    return {
        "schema": "quant.oracle_fmp_price_rebuild.v1",
        "status": "complete",
        "start_date": args.start_date,
        "target_as_of_date": ensured["target_as_of_date"],
        "backup_root": str(backup_root),
        "derived_state_backed_up": state_backup,
        "reset": reset,
        "oracle_status": ensured,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-07-15")
    parser.add_argument("--target-as-of-date", required=True)
    parser.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    parser.add_argument(
        "--incremental-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT
    )
    parser.add_argument(
        "--incremental-database",
        type=Path,
        default=DEFAULT_INCREMENTAL_DATABASE,
    )
    parser.add_argument("--constituent-refresh-max-etfs", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = rebuild(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": "quant.oracle_fmp_price_rebuild.v1",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
