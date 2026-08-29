"""Forecast RADAR command-line interface for batch jobs and Vectorman."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .contracts import DEFAULT_LIVE_ROOT
from .model_bundle import main as train_main
from .pipeline import query_latest, run_daily


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-final")
    train.add_argument("args", nargs=argparse.REMAINDER)
    run = commands.add_parser("run-daily")
    run.add_argument("--signal-date")
    run.add_argument("--if-needed", action="store_true")
    scheduled = commands.add_parser("scheduled-daily")
    scheduled.add_argument("--signal-date")
    scheduled.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    query = commands.add_parser("query")
    query.add_argument("--symbol")
    query.add_argument("--sector")
    query.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    status = commands.add_parser("status")
    status.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "train-final":
        return train_main(args.args)
    if args.command == "run-daily":
        value = run_daily(signal_date=args.signal_date, if_needed=args.if_needed)
    elif args.command == "scheduled-daily":
        batch = run_daily(
            signal_date=args.signal_date,
            if_needed=True,
            live_root=args.live_root,
        )
        snapshot = query_latest(live_root=args.live_root)
        value = {
            "batch": batch,
            "latest": snapshot["latest"],
            "probability_resolution": snapshot["probability_resolution"],
        }
    elif args.command == "query":
        value = query_latest(
            live_root=args.live_root,
            symbol=args.symbol,
            sector=args.sector,
        )
    else:
        latest_path = args.live_root / "latest.json"
        value = (
            json.loads(latest_path.read_text(encoding="utf-8"))
            if latest_path.is_file()
            else {"status": "NO_COMPLETED_RUN"}
        )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
