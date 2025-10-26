#!/usr/bin/env python3
"""Download long-horizon price history with yfinance.

The script fetches daily close prices for the dashboard's asset universe,
covering 2017-01-01 through today (inclusive by default), and stores the
result as JSON under ``static_site/data``.  Later steps in the pipeline can
merge this historical snapshot with the Alpha Vantage pulls that focus on
recent data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

try:
    import yfinance as yf
except ImportError as error:  # pragma: no cover - handled at runtime
    print(
        "yfinance is required. Install it via 'pip install yfinance' before running this script.",
        file=sys.stderr,
    )
    raise


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "static_site" / "data"
DEFAULT_OUTPUT = DATA_DIR / "historical_prices.json"
DEFAULT_START = date(2017, 1, 1)


@dataclass(frozen=True)
class Asset:
    symbol: str
    label: str
    category: str


ASSETS: Sequence[Asset] = (
    Asset("QQQ", "QQQ (NASDAQ 100 ETF)", "stock"),
    Asset("IWM", "IWM (Russell 2000 ETF)", "stock"),
    Asset("SPY", "SPY (S&P 500 ETF)", "stock"),
    Asset("TLT", "TLT (US Long Treasury)", "bond"),
    Asset("GLD", "GLD (Gold ETF)", "gold"),
    Asset("BTC-USD", "BTC-USD (Bitcoin)", "crypto"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default=DEFAULT_START.isoformat(),
        help="ISO start date (default: 2017-01-01)",
    )
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="ISO end date (default: today)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON path (default: static_site/data/historical_prices.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file even if it already exists.",
    )
    return parser.parse_args()


def parse_iso_date(value: str, *, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise SystemExit(f"Invalid {field} date '{value}': {exc}") from exc


def ensure_valid_range(start: date, end: date) -> None:
    if end < start:
        raise SystemExit(f"End date {end} must be on/after start date {start}.")


def fetch_symbol_history(symbol: str, start: date, end: date) -> List[Tuple[str, float]]:
    """Return sorted (date, close) tuples between start and end inclusive."""

    # yfinance's ``end`` parameter is exclusive; push it one day forward so
    # the closing price for ``end`` itself is included when available.
    end_plus_one = end + timedelta(days=1)
    frame = yf.download(
        symbol,
        start=start.isoformat(),
        end=end_plus_one.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    frame = _flatten_download_columns(frame, symbol)

    if frame.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}.")

    column = _select_close_column(frame)
    series = frame[column].dropna()

    rows: List[Tuple[str, float]] = []
    for idx, value in series.items():
        # yfinance can emit timezone-aware timestamps; convert to date.
        if hasattr(idx, "to_pydatetime"):
            idx_datetime = idx.to_pydatetime()
        else:
            idx_datetime = datetime.combine(idx, datetime.min.time())
        date_key = idx_datetime.date().isoformat()
        rows.append((date_key, float(value)))

    rows.sort(key=lambda pair: pair[0])
    # Deduplicate (yfinance can emit multiple entries for dividends/actions).
    deduped: Dict[str, float] = {}
    for date_key, price in rows:
        deduped[date_key] = price

    return sorted(deduped.items(), key=lambda pair: pair[0])


def _flatten_download_columns(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    columns = frame.columns
    if not isinstance(columns, pd.MultiIndex):
        return frame

    for level in range(columns.nlevels):
        level_values = columns.get_level_values(level)
        if all(value == symbol for value in level_values):
            return frame.droplevel(level, axis=1)

    flattened = [
        "_".join(str(part) for part in column if part and part != symbol)
        for column in columns
    ]
    frame = frame.copy()
    frame.columns = flattened
    return frame


def _select_close_column(frame: pd.DataFrame) -> str:
    candidates = ["Adj Close", "Close", "adjclose", "close"]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    # fallback: match columns that contain the keyword (e.g., "Adj Close_QQQ")
    lowered = [col.lower() for col in frame.columns]
    for keyword in ("adj close", "close"):
        for idx, name in enumerate(lowered):
            if keyword in name:
                return frame.columns[idx]

    raise KeyError("Close")


def build_payload(start: date, end: date, history: Dict[str, List[Tuple[str, float]]]) -> Dict[str, object]:
    assets_payload = []
    for asset in ASSETS:
        rows = history.get(asset.symbol, [])
        if not rows:
            raise RuntimeError(f"Missing history for {asset.symbol}.")
        dates, prices = zip(*rows)
        assets_payload.append(
            {
                "symbol": asset.symbol,
                "label": asset.label,
                "category": asset.category,
                "dates": list(dates),
                "prices": [round(float(price), 6) for price in prices],
            }
        )

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "source": "yfinance",
        "assets": assets_payload,
    }


def write_payload(path: Path, payload: Dict[str, object], *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file {path}. Use --force to override.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = parse_iso_date(args.start, field="start")
    end = parse_iso_date(args.end, field="end")
    ensure_valid_range(start, end)

    history: Dict[str, List[Tuple[str, float]]] = {}
    for asset in ASSETS:
        print(f"[yfinance] downloading {asset.symbol} from {start} to {end}")
        rows = fetch_symbol_history(asset.symbol, start, end)
        history[asset.symbol] = rows
        print(f"[yfinance] {asset.symbol}: {len(rows)} rows")

    payload = build_payload(start, end, history)
    write_payload(args.out, payload, force=args.force)
    try:
        display_path = args.out.relative_to(ROOT)
    except ValueError:
        display_path = args.out
    print(f"Wrote {display_path}")


if __name__ == "__main__":
    main()
