#!/usr/bin/env python3
"""Run a larger weekly Oracle constituent refresh and rebuild relations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
from workflows.quant_ai_radar.universe import write_json


KST = ZoneInfo("Asia/Seoul")
DEFAULT_INCREMENTAL_ROOT = DEFAULT_INCREMENTAL_DATABASE.parent.parent
DEFAULT_STATE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/status/"
    "weekly_relations.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-date")
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
    parser.add_argument("--constituent-refresh-max-etfs", type=int, default=300)
    parser.add_argument("--constituent-stale-days", type=int, default=45)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    state = {
        "schema_version": "quant.ai_radar_weekly_relations.v2",
        "status": "running",
        "started_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "source_owner": "market_structure_oracle_single_writer",
    }
    write_json(args.state_file, state)
    try:
        status = ensure_oracle_snapshot(
            base_database=args.base_database.expanduser().resolve(),
            incremental_root=args.incremental_root.expanduser().resolve(),
            target_as_of_date=args.market_date,
            force_repair=True,
            constituent_stale_days=args.constituent_stale_days,
            constituent_refresh_max_etfs=args.constituent_refresh_max_etfs,
        )
        binding = load_shared_market_binding(
            base_database=args.base_database,
            incremental_database=args.incremental_database,
            oracle_status_path=args.oracle_status,
        )
        relation = refresh_relation_index(binding, args.relation_index)
        state.update(
            {
                "status": "complete",
                "completed_at_kst": datetime.now(KST).isoformat(
                    timespec="seconds"
                ),
                "target_date": status["target_as_of_date"],
                "oracle_ensure_mode": status["ensure_mode"],
                "constituent_refresh": status["etf_constituents"],
                "relation_index": relation,
                "duplicate_collection": False,
                "etf_radar_runtime_dependency": False,
            }
        )
    except Exception as exc:
        state.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_kst": datetime.now(KST).isoformat(
                    timespec="seconds"
                ),
            }
        )
        write_json(args.state_file, state)
        print(json.dumps(state, ensure_ascii=False))
        return 1
    write_json(args.state_file, state)
    print(json.dumps(state, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
