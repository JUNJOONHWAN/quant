#!/usr/bin/env python3
"""Prepare or reuse the Oracle-owned source snapshot for Quant AI Radar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_dataset.shared_market import (
    DEFAULT_BASE_DATABASE,
    DEFAULT_INCREMENTAL_DATABASE,
    DEFAULT_ORACLE_STATUS,
    load_shared_market_binding,
)
from workflows.market_structure_oracle.incremental_store import (
    ensure_oracle_snapshot,
)
from workflows.quant_ai_radar.relation_index import (
    DEFAULT_RELATION_INDEX,
    refresh_relation_index,
)


DEFAULT_INCREMENTAL_ROOT = DEFAULT_INCREMENTAL_DATABASE.parent.parent


def prepare(args: argparse.Namespace) -> dict:
    status_path = args.oracle_status.expanduser().resolve()
    ensured = ensure_oracle_snapshot(
        base_database=args.base_database.expanduser().resolve(),
        incremental_root=args.incremental_root.expanduser().resolve(),
        target_as_of_date=args.target_as_of_date,
        force_repair=args.force_repair,
        publish_grace_hour_et=args.publish_grace_hour_et,
        constituent_stale_days=args.constituent_stale_days,
        constituent_refresh_max_etfs=args.constituent_refresh_max_etfs,
    )
    binding = load_shared_market_binding(
        base_database=args.base_database,
        incremental_database=args.incremental_database,
        oracle_status_path=status_path,
        max_constituent_available_lag_days=(
            args.max_constituent_available_lag_days
        ),
    )
    if binding.target_as_of_date != ensured["target_as_of_date"]:
        raise RuntimeError("shared market binding does not match Oracle snapshot seal")
    relation_index = refresh_relation_index(binding, args.relation_index)
    return {
        "schema_version": "quant.shared_market_prepare.v2",
        "status": "complete",
        "mode": ensured["ensure_mode"],
        "target_as_of_date": binding.target_as_of_date,
        "source_contract": binding.source_fingerprint["incremental"][
            "snapshot_seal"
        ]["source_contract"],
        "binding": binding.public_metadata(),
        "relation_index": relation_index,
        "fmp_historical_backfill_resumed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    parser.add_argument(
        "--incremental-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT
    )
    parser.add_argument(
        "--incremental-database",
        type=Path,
        default=DEFAULT_INCREMENTAL_DATABASE,
    )
    parser.add_argument("--oracle-status", type=Path, default=DEFAULT_ORACLE_STATUS)
    parser.add_argument(
        "--relation-index", type=Path, default=DEFAULT_RELATION_INDEX
    )
    parser.add_argument("--target-as-of-date")
    parser.add_argument("--publish-grace-hour-et", type=int, default=18)
    parser.add_argument("--constituent-stale-days", type=int, default=45)
    parser.add_argument("--constituent-refresh-max-etfs", type=int, default=50)
    parser.add_argument(
        "--max-constituent-available-lag-days", type=int, default=45
    )
    parser.add_argument("--force-repair", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_constituent_available_lag_days < 0:
        raise SystemExit("--max-constituent-available-lag-days must be >= 0")
    if args.constituent_stale_days < 1:
        raise SystemExit("--constituent-stale-days must be >= 1")
    if args.constituent_refresh_max_etfs < 0:
        raise SystemExit("--constituent-refresh-max-etfs must be >= 0")
    try:
        result = prepare(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": "quant.shared_market_prepare.v2",
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
