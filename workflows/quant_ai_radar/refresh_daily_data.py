#!/usr/bin/env python3
"""Daily source refresh preceding trained Quant AI inference.

FMP full-symbol EOD requests use quant_dataset's shared 240/min limiter, leaving
headroom under the user's 300/min account ceiling.  ETF RADAR evidence is reused
from a hash-verified immutable release; it is never recollected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from workflows.quant_ai_radar.etfradar_release import (  # noqa: E402
    discover_release,
    verify_release,
)
from workflows.quant_ai_radar.universe import write_json  # noqa: E402


KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")
DEFAULT_DATA_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET")
DEFAULT_ETFRADAR_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/ETFRADAR")
DEFAULT_STATE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/status/daily_data_refresh.json"
)


class DailyRefreshError(RuntimeError):
    """Raised when any required source refresh stage fails."""


def latest_completed_us_weekday(now: datetime | None = None) -> str:
    current = (now or datetime.now(ET)).astimezone(ET)
    candidate = current.date()
    if current.time() < time(20, 0):
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate.isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_symbols_file(data_root: Path) -> Path:
    files = sorted(
        data_root.expanduser().resolve().glob(
            "state/universe/fmp_us_equity_etf_*.symbols.txt"
        )
    )
    if not files:
        raise DailyRefreshError("no filtered FMP US equity/ETF symbol file exists")
    path = files[-1]
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise DailyRefreshError(f"symbol manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_symbols_sha256") != _sha256(path):
        raise DailyRefreshError("symbol file SHA256 does not match its manifest")
    if int(manifest.get("symbol_count", 0)) != sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ):
        raise DailyRefreshError("symbol file row count does not match its manifest")
    return path


def _run_json(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DailyRefreshError(
            f"command failed ({completed.returncode}): {' '.join(command)}: {detail[-4000:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DailyRefreshError(
            f"command returned invalid JSON: {' '.join(command)}"
        ) from exc
    if not isinstance(value, dict) or value.get("ok") is False:
        raise DailyRefreshError(f"command reported failure: {' '.join(command)}: {value}")
    return value


def _summary(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ok",
        "job_id",
        "done",
        "empty",
        "skipped",
        "failed",
        "observations",
        "checkpoint_summary",
        "quality",
        "run_id",
        "status",
        "pages",
        "rows",
        "latest_processed_date",
        "freshness",
    )
    return {key: value.get(key) for key in keys if key in value}


def run(args: argparse.Namespace) -> dict[str, Any]:
    market_date = date.fromisoformat(
        args.market_date or latest_completed_us_weekday()
    ).isoformat()
    data_root = args.data_root.expanduser().resolve()
    symbol_path = (
        args.symbols_file.expanduser().resolve()
        if args.symbols_file
        else latest_symbols_file(data_root)
    )
    if args.symbols_file and not symbol_path.is_file():
        raise DailyRefreshError(f"symbols file is missing: {symbol_path}")

    # This proves the already-downloaded ETF RADAR source package before any
    # duplicate FMP/Massive ETF collection could be considered.
    etfradar_release = discover_release(args.etfradar_data_root, market_date)
    etfradar_binding = verify_release(etfradar_release)
    if str(etfradar_binding["trade_date_us"]) != market_date:
        raise DailyRefreshError(
            "latest complete ETF RADAR release is stale: "
            f"target={market_date} release={etfradar_binding['trade_date_us']}"
        )

    common = [
        sys.executable,
        "-m",
        "quant_dataset",
        "--data-root",
        str(data_root),
        "--secrets-file",
        str(args.secrets_file),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
    ]
    stages: dict[str, Any] = {
        "etfradar_release": {
            "status": "confirmed_reused_no_download",
            **etfradar_binding,
        }
    }
    fmp = _run_json(
        common
        + [
            "capture-daily",
            "--date",
            market_date,
            "--source",
            "fmp",
            "--symbols-file",
            str(symbol_path),
        ],
        QUANT_ROOT,
    )
    stages["fmp_full_symbol_daily"] = _summary(fmp)
    massive = _run_json(
        common
        + [
            "capture-daily",
            "--date",
            market_date,
            "--source",
            "massive",
        ],
        QUANT_ROOT,
    )
    stages["massive_grouped_daily"] = _summary(massive)
    flow = _run_json(
        common
        + [
            "capture-etf-flows",
            "--date",
            market_date,
            "--lookback-days",
            "7",
            "--max-lag-days",
            str(args.max_flow_lag_days),
            "--strict-freshness",
        ],
        QUANT_ROOT,
    )
    stages["massive_etf_global_flow"] = _summary(flow)
    result = {
        "schema_version": "quant.ai_radar_daily_refresh.v1",
        "status": "complete",
        "market_date": market_date,
        "completed_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "symbols_file": str(symbol_path),
        "symbols_file_sha256": _sha256(symbol_path),
        "fmp_rate_limit_policy": "240 requests/min shared limiter under 300/min account limit",
        "stages": stages,
        "next_stage": "run_quant_ai_radar.py using the accepted trained-model release",
    }
    write_json(args.state_file, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-date")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--etfradar-data-root", type=Path, default=DEFAULT_ETFRADAR_ROOT)
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument(
        "--secrets-file",
        type=Path,
        default=Path("/home/zooh/Documents/GitHub/STOCK/secrets.env"),
    )
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--max-flow-lag-days", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except (DailyRefreshError, OSError, ValueError) as exc:
        error = {
            "schema_version": "quant.ai_radar_daily_refresh.v1",
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
