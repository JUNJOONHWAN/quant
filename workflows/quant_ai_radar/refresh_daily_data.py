#!/usr/bin/env python3
"""Compatibility CLI for the Oracle-owned shared market refresh.

This command no longer downloads an independent AI Radar dataset.  It enters
the same interprocess-locked Oracle single-writer path used by the daily cycle.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_dataset.shared_market import DEFAULT_BASE_DATABASE
from workflows.market_structure_oracle.incremental_store import (
    ensure_oracle_snapshot,
    latest_closed_nyse_session,
)
from workflows.quant_ai_radar.universe import write_json


KST = ZoneInfo("Asia/Seoul")
DEFAULT_INCREMENTAL_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/incremental"
)
DEFAULT_STATE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/status/"
    "daily_data_refresh.json"
)


class DailyRefreshError(RuntimeError):
    """Raised when the Oracle shared refresh cannot be completed."""


def latest_completed_us_weekday(now: datetime | None = None) -> str:
    """Backward-compatible name for the close-aware NYSE session policy."""

    return latest_closed_nyse_session(now, publish_grace_hour_et=18)


def run(args: argparse.Namespace) -> dict:
    status = ensure_oracle_snapshot(
        base_database=args.base_database.expanduser().resolve(),
        incremental_root=args.incremental_root.expanduser().resolve(),
        target_as_of_date=args.market_date,
        force_repair=args.force_repair,
        publish_grace_hour_et=args.publish_grace_hour_et,
        constituent_stale_days=args.constituent_stale_days,
        constituent_refresh_max_etfs=args.constituent_refresh_max_etfs,
    )
    result = {
        "schema_version": "quant.ai_radar_daily_refresh.v2",
        "status": "complete",
        "market_date": status["target_as_of_date"],
        "completed_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "mode": status["ensure_mode"],
        "source_owner": "market_structure_oracle_single_writer",
        "duplicate_fmp_massive_collection": False,
        "etf_radar_runtime_dependency": False,
        "oracle_status": status,
    }
    write_json(args.state_file, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-date")
    parser.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    parser.add_argument(
        "--incremental-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT
    )
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--publish-grace-hour-et", type=int, default=18)
    parser.add_argument("--constituent-stale-days", type=int, default=45)
    parser.add_argument("--constituent-refresh-max-etfs", type=int, default=50)
    parser.add_argument("--force-repair", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except (DailyRefreshError, OSError, ValueError) as exc:
        error = {
            "schema_version": "quant.ai_radar_daily_refresh.v2",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        }
        write_json(args.state_file, error)
        print(json.dumps(error, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
