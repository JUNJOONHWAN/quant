#!/usr/bin/env python3
"""Backfill helper for market probability report history.

This script replays historical SoT payloads (derived from AutoTrade2's regime engine)
and invokes `market_analysis.market_report.generate_market_report` for each trading day
between the requested start/end dates. Each invocation appends a record to
`ml_cache/market_prob_history.jsonl`, which can then be used to train or backtest the
Gaussian NB + Platt model.

Example:
    python3 scripts/backfill_market_prob_history.py --start 2020-01-02 --benchmark QQQ
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from market_analysis.market_report import generate_market_report
from regime_service import at2_get_payload_close_raw

DEFAULT_START = dt.date(2020, 1, 2)


def _parse_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc


def _coerce_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value)
    if "T" in text:
        text = text.split("T", 1)[0]
    return dt.date.fromisoformat(text)


def _load_existing_dates(path: Path, base_symbol: str, horizon: int) -> Set[str]:
    if not path.exists():
        return set()
    hits: Set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("base_symbol", "")).upper() != base_symbol.upper():
                    continue
                try:
                    if int(row.get("horizon_days", -1)) != int(horizon):
                        continue
                except (TypeError, ValueError):
                    continue
                asof = row.get("asof_date")
                if asof:
                    hits.add(str(asof))
    except FileNotFoundError:
        return set()
    return hits


def _slice_series_map(series_map: Any, end_idx: int) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    if not isinstance(series_map, dict):
        return out
    for key, value in series_map.items():
        if isinstance(value, list):
            out[key] = value[: end_idx + 1]
    return out


def _slice_nested(obj: Any, end_idx: int) -> Any:
    if isinstance(obj, list):
        return obj[: end_idx + 1]
    if isinstance(obj, dict):
        return {k: _slice_nested(v, end_idx) for k, v in obj.items()}
    return obj


def _build_snapshot(payload: Dict[str, Any], end_idx: int, date_str: str) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "dates": _slice_nested(payload.get("dates", []), end_idx),
        "series": _slice_series_map(payload.get("series"), end_idx),
    }
    if payload.get("series_open"):
        snapshot["series_open"] = _slice_series_map(payload.get("series_open"), end_idx)
    for key in (
        "fusion",
        "ffl_stab",
        "classic",
        "stability",
        "smoothed",
        "delta",
        "sub",
        "backtest",
    ):
        if key in payload:
            snapshot[key] = _slice_nested(payload[key], end_idx)
    snapshot["window"] = payload.get("window")
    preset = payload.get("fusion_preset")
    manifest = dict(payload.get("manifest") or {})
    manifest.update(
        {
            "origin": "autotrade2.backfill",
            "mode": "close",
            "use_realtime": False,
            "intraday_base": False,
            "anchor": manifest.get("anchor", "ET_16:00"),
            "tz": manifest.get("tz", "America/New_York"),
            "as_of_ts": f"{date_str}T23:59:00+00:00",
            "backfill_end": date_str,
        }
    )
    if preset and "preset" not in manifest:
        manifest["preset"] = preset
    snapshot["manifest"] = manifest
    asof = dict(payload.get("asof") or {})
    asof.update(
        {
            "today_utc": date_str,
            "fusion_last_date": date_str,
            "now_utc": f"{date_str} 23:59:00 UTC",
            "backfill": True,
        }
    )
    snapshot["asof"] = asof
    return snapshot


def _iter_targets(
    dates: Iterable[Any],
    start_date: dt.date,
    end_date: dt.date,
) -> List[Tuple[int, dt.date, str]]:
    targets: List[Tuple[int, dt.date, str]] = []
    for idx, raw in enumerate(dates):
        try:
            day = _coerce_date(raw)
        except Exception:
            continue
        if day < start_date or day > end_date:
            continue
        targets.append((idx, day, day.isoformat()))
    return targets


def _validate_env() -> None:
    if not os.getenv("FMP_API_KEY"):
        raise RuntimeError(
            "FMP_API_KEY is required for regime payload computation. "
            "Set it in your environment or .env file."
        )


def run(args: argparse.Namespace) -> None:
    _validate_env()
    start_date = args.start or DEFAULT_START
    end_date = args.end or dt.date.today()
    if end_date < start_date:
        raise ValueError("End date must be on/after start date.")

    history_path = Path(args.history or "ml_cache/market_prob_history.jsonl")
    existing = set()
    if not args.overwrite:
        existing = _load_existing_dates(history_path, args.benchmark.upper(), args.horizon)

    payload = at2_get_payload_close_raw(window=args.window, preset=args.preset)
    dates = payload.get("dates", [])
    if not isinstance(dates, list) or not dates:
        raise RuntimeError("Regime payload did not return any dates; cannot proceed.")

    targets = _iter_targets(dates, start_date, end_date)
    if not targets:
        print("No trading days within the requested window; nothing to backfill.")
        return
    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    to_run = [
        (idx, day, date_str)
        for idx, day, date_str in targets
        if args.overwrite or date_str not in existing
    ]

    print(
        "Total trading days in range: "
        f"{len(targets)} (covering {targets[0][1]} → {targets[-1][1]})"
    )
    print(
        "Pending backfill runs: "
        f"{len(to_run)} (skip duplicates={'no' if args.overwrite else 'yes'})"
    )
    if args.dry_run:
        for _, day, _ in to_run[:10]:
            print(f"- would backfill {day.isoformat()}")
        if len(to_run) > 10:
            print(f"... and {len(to_run) - 10} more")
        return

    success = 0
    failures: List[Tuple[str, str]] = []
    for idx, _, date_str in to_run:
        snapshot = _build_snapshot(payload, idx, date_str)
        try:
            result = generate_market_report(
                horizon_days=args.horizon,
                lookback_days=args.lookback,
                benchmark=args.benchmark,
                sot_payload=snapshot,
                use_cache=not args.no_cache,
                auto_calibrate=not args.no_calib,
            )
            success += 1
            prob = (result.get("prob") or {}).get("p_up")
            prob_txt = f"{float(prob):.2%}" if isinstance(prob, (int, float)) else "N/A"
            print(f"[{success}/{len(to_run)}] {date_str} · P(Up)={prob_txt} · history append OK")
        except Exception as exc:  # pragma: no cover - runtime failure path
            failures.append((date_str, str(exc)))
            print(f"[!] {date_str} failed: {exc}")
            if args.stop_on_error:
                break
        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Backfill complete → success {success}, failed {len(failures)}")
    if failures:
        print("Failures:")
        for day, err in failures:
            print(f"- {day}: {err}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill market probability history via SoT payload replay."
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="Start date (YYYY-MM-DD, default 2020-01-02).",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="End date (YYYY-MM-DD, default today).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward horizon days (match training).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        choices=[20, 30, 60],
        help="Regime window (20/30/60).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="QQQ",
        help="Benchmark/base symbol.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=1260,
        help="Model lookback (doc-only).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Regime preset (optional).",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="ml_cache/market_prob_history.jsonl",
        help="History file path (append-only).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N days (0 = all).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Throttle between runs in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report planned runs without executing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if entries already exist (appends new rows).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort on first failure.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cached model parameters (cold inference).",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="Disable Platt scaling during inference.",
    )
    return parser


def main() -> None:  # pragma: no cover - CLI glue
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()
