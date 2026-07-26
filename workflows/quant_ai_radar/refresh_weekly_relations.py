#!/usr/bin/env python3
"""Refresh the live FMP universe and PIT ETF constituent relationship layer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from workflows.quant_ai_radar.refresh_daily_data import (
    DailyRefreshError,
    latest_completed_us_weekday,
)
from workflows.quant_ai_radar.universe import write_json


KST = ZoneInfo("Asia/Seoul")
QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")
DEFAULT_DATA_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET")
DEFAULT_STATE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/status/weekly_relations.json"
)


def _run_json(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=QUANT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DailyRefreshError(
            f"weekly relation command failed ({completed.returncode}): {detail[-4000:]}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict) or value.get("ok") is False:
        raise DailyRefreshError(f"weekly relation command reported failure: {value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-date")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--secrets-file",
        type=Path,
        default=Path("/home/zooh/Documents/GitHub/STOCK/secrets.env"),
    )
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    target = date.fromisoformat(
        args.market_date or latest_completed_us_weekday()
    )
    start = target - timedelta(days=args.lookback_days)
    root = args.data_root.expanduser().resolve()
    common = [
        sys.executable,
        "-m",
        "quant_dataset",
        "--data-root",
        str(root),
        "--secrets-file",
        str(args.secrets_file),
    ]
    state = {
        "schema_version": "quant.ai_radar_weekly_relations.v1",
        "status": "running",
        "target_date": target.isoformat(),
        "started_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
    }
    write_json(args.state_file, state)
    try:
        universe = _run_json(
            common + ["capture-fmp-universe", "--date", target.isoformat()]
        )
        universe_jsonl = Path(str(universe["jsonl_path"]))
        filtered_symbols = root / "state" / "universe" / (
            f"fmp_us_equity_etf_{target.strftime('%Y%m%d')}.symbols.txt"
        )
        symbol_result = _run_json(
            [
                sys.executable,
                str(QUANT_ROOT / "scripts" / "build_fmp_us_equity_etf_symbols.py"),
                "--input-jsonl",
                str(universe_jsonl),
                "--output-symbols",
                str(filtered_symbols),
            ]
        )
        constituents = _run_json(
            common
            + [
                "backfill-fmp-etf-constituents",
                "--from",
                start.isoformat(),
                "--to",
                target.isoformat(),
                "--universe-jsonl",
                str(universe_jsonl),
            ]
        )
        state.update(
            {
                "status": "complete",
                "completed_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
                "universe": universe,
                "filtered_symbols": symbol_result,
                "constituents": constituents,
                "historical_rows_preserved": True,
                "present_day_active_list_used_for_historical_filter": False,
            }
        )
    except Exception as exc:
        state.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
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
