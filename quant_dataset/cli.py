"""Command-line interface for the daily dataset pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import DEFAULT_SECRETS_PATH, load_credentials, resolve_data_root
from .fmp_etf_constituents import (
    read_etf_symbols_from_universe,
    shard_symbols,
    universe_jsonl_contract,
)
from .fmp_universe import read_symbol_file, symbol_file_contract
from .pipeline import DatasetPipeline
from .providers import DatasetError


def _symbols(value: Optional[str]) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _statuses(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m quant_dataset")
    parser.add_argument("--data-root", help="default: QUANT_DATASET_ROOT or DGX STOCKDATA path")
    parser.add_argument("--secrets-file", default=str(DEFAULT_SECRETS_PATH))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="check paths, credentials, and source policy")
    preflight.add_argument("--allow-missing-keys", action="store_true")

    daily = commands.add_parser("capture-daily", help="capture one market date")
    daily.add_argument("--date", required=True)
    daily.add_argument("--symbols", help="comma-separated FMP universe")
    daily.add_argument(
        "--symbols-file", help="UTF-8 file with one FMP symbol per line"
    )
    daily.add_argument("--source", choices=("both", "fmp", "massive"), default="both")
    daily.add_argument("--include-otc", action="store_true")
    daily.add_argument("--unadjusted", action="store_true")
    daily.add_argument("--fail-fast", action="store_true")

    backfill = commands.add_parser("backfill", help="resume a historical date range")
    backfill.add_argument("--from", dest="start_date", required=True)
    backfill.add_argument("--to", dest="end_date", required=True)
    backfill.add_argument("--symbols", help="comma-separated FMP universe")
    backfill.add_argument(
        "--symbols-file", help="UTF-8 file with one FMP symbol per line"
    )
    backfill.add_argument("--source", choices=("both", "fmp", "massive"), default="both")
    backfill.add_argument("--include-otc", action="store_true")
    backfill.add_argument("--unadjusted", action="store_true")
    backfill.add_argument("--fail-fast", action="store_true")

    flow = commands.add_parser(
        "capture-etf-flows",
        help="capture a recent Massive ETF Global processed-date window",
    )
    flow.add_argument("--date", required=True, help="freshness as-of date, YYYY-MM-DD")
    flow.add_argument("--lookback-days", type=int, default=7)
    flow.add_argument("--tickers", help="optional comma-separated composite tickers")
    flow.add_argument("--limit", type=int, default=5000, help="page size, maximum 5000")
    flow.add_argument("--max-lag-days", type=int, default=4)
    flow.add_argument("--no-resume", action="store_true")
    flow.add_argument("--strict-freshness", action="store_true")

    flow_backfill = commands.add_parser(
        "backfill-etf-flows",
        help="backfill Massive ETF flows by historical processed_date filters",
    )
    flow_backfill.add_argument("--from", dest="start_date", required=True)
    flow_backfill.add_argument("--to", dest="end_date", required=True)
    flow_backfill.add_argument("--tickers", help="optional comma-separated composite tickers")
    flow_backfill.add_argument(
        "--limit", type=int, default=5000, help="page size, maximum 5000"
    )
    flow_backfill.add_argument("--no-resume", action="store_true")

    universe = commands.add_parser(
        "capture-fmp-universe",
        help="capture FMP current, ETF, delisted, and symbol-change universe",
    )
    universe.add_argument("--date", required=True, help="snapshot date, YYYY-MM-DD")

    constituents = commands.add_parser(
        "backfill-fmp-etf-constituents",
        help="backfill historical FMP ETF constituents with PIT availability gates",
    )
    constituents.add_argument("--from", dest="start_date", required=True)
    constituents.add_argument("--to", dest="end_date", required=True)
    constituents.add_argument("--tickers", help="optional comma-separated ETF tickers")
    constituents.add_argument(
        "--universe-jsonl", help="FMP universe JSONL used to select all ETFs"
    )
    constituents.add_argument("--shard-count", type=int, default=1)
    constituents.add_argument("--shard-index", type=int, default=0)
    constituents.add_argument("--fail-fast", action="store_true")

    fmp_training = commands.add_parser(
        "backfill-fmp-training",
        help="backfill the live-classified FMP training endpoint catalog",
    )
    fmp_training.add_argument("--plan", required=True)
    fmp_training.add_argument("--symbols-file", required=True)
    fmp_training.add_argument("--universe-jsonl", required=True)
    fmp_training.add_argument("--from", dest="start_date", required=True)
    fmp_training.add_argument("--to", dest="end_date", required=True)
    fmp_training.add_argument(
        "--endpoint-ids", help="optional comma-separated endpoint ids"
    )
    fmp_training.add_argument("--fail-fast", action="store_true")

    verify = commands.add_parser("verify", help="verify checksums, normalized rows, and QC")
    verify.add_argument("--from", dest="start_date")
    verify.add_argument("--to", dest="end_date")
    verify.add_argument("--symbols")
    verify.add_argument("--strict-warnings", action="store_true")

    export = commands.add_parser("export-packets", help="export deterministic unlabeled JSONL")
    export.add_argument("--from", dest="start_date", required=True)
    export.add_argument("--to", dest="end_date", required=True)
    export.add_argument("--output")
    export.add_argument("--symbols")
    export.add_argument(
        "--lookback-days",
        type=int,
        default=21,
        help="price sessions per packet; ETF Flow observations are capped at 20",
    )
    export.add_argument(
        "--quality-statuses", default="pass,warn,single_source", help="comma-separated"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        data_root = resolve_data_root(args.data_root)
        credentials = load_credentials(secrets_path=Path(args.secrets_file))
        pipeline = DatasetPipeline(
            data_root=data_root,
            credentials=credentials,
            timeout_seconds=args.timeout,
            retries=args.retries,
        )
        if args.command == "preflight":
            result = pipeline.preflight(require_keys=not args.allow_missing_keys)
        elif args.command == "capture-daily":
            symbols = _symbols(args.symbols)
            if args.symbols_file:
                symbols.extend(read_symbol_file(Path(args.symbols_file).expanduser()))
            result = pipeline.capture_daily(
                args.date,
                symbols,
                source=args.source,
                adjusted=not args.unadjusted,
                include_otc=args.include_otc,
                continue_on_error=not args.fail_fast,
            )
        elif args.command == "backfill":
            symbols = _symbols(args.symbols)
            universe_contract = None
            if args.symbols_file:
                symbol_path = Path(args.symbols_file).expanduser()
                symbols.extend(read_symbol_file(symbol_path))
                universe_contract = symbol_file_contract(symbol_path)
            result = pipeline.backfill(
                args.start_date,
                args.end_date,
                symbols,
                source=args.source,
                adjusted=not args.unadjusted,
                include_otc=args.include_otc,
                continue_on_error=not args.fail_fast,
                symbol_universe=universe_contract,
            )
        elif args.command == "capture-fmp-universe":
            result = pipeline.capture_fmp_universe(args.date)
        elif args.command == "backfill-fmp-etf-constituents":
            tickers = _symbols(args.tickers)
            universe_contract = None
            if args.universe_jsonl:
                universe_path = Path(args.universe_jsonl).expanduser()
                tickers.extend(read_etf_symbols_from_universe(universe_path))
                universe_contract = universe_jsonl_contract(universe_path)
            tickers = shard_symbols(tickers, args.shard_count, args.shard_index)
            if universe_contract is not None:
                universe_contract = {
                    **universe_contract,
                    "shard_count": args.shard_count,
                    "shard_index": args.shard_index,
                    "shard_symbol_count": len(tickers),
                }
            result = pipeline.backfill_fmp_etf_constituents(
                args.start_date,
                args.end_date,
                tickers,
                universe_contract=universe_contract,
                continue_on_error=not args.fail_fast,
            )
        elif args.command == "backfill-fmp-training":
            result = pipeline.backfill_fmp_training(
                Path(args.plan).expanduser(),
                Path(args.symbols_file).expanduser(),
                Path(args.universe_jsonl).expanduser(),
                args.start_date,
                args.end_date,
                endpoint_ids=_symbols(args.endpoint_ids),
                continue_on_error=not args.fail_fast,
            )
        elif args.command == "capture-etf-flows":
            result = pipeline.capture_etf_flows(
                args.date,
                lookback_days=args.lookback_days,
                tickers=_symbols(args.tickers),
                limit=args.limit,
                max_lag_days=args.max_lag_days,
                resume=not args.no_resume,
                strict_freshness=args.strict_freshness,
            )
        elif args.command == "backfill-etf-flows":
            result = pipeline.backfill_etf_flows(
                args.start_date,
                args.end_date,
                tickers=_symbols(args.tickers),
                limit=args.limit,
                resume=not args.no_resume,
            )
        elif args.command == "verify":
            result = pipeline.verify(
                args.start_date, args.end_date, _symbols(args.symbols) or None
            )
            if args.strict_warnings and result["quality_totals"].get("warn", 0):
                result["ok"] = False
                result["errors"].append(
                    {
                        "error": "strict_warning_gate",
                        "count": result["quality_totals"]["warn"],
                    }
                )
        else:
            output = args.output or str(
                data_root
                / "training_packets"
                / "analysis_packets_{}_{}.jsonl".format(args.start_date, args.end_date)
            )
            result = pipeline.export_packets(
                args.start_date,
                args.end_date,
                Path(output),
                symbols=_symbols(args.symbols),
                lookback_days=args.lookback_days,
                quality_statuses=_statuses(args.quality_statuses),
            )
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if result.get("ok", True) else 1
    except (DatasetError, ValueError, OSError) as error:
        print(
            json.dumps(
                {"ok": False, "error_type": type(error).__name__, "error": str(error)},
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
