#!/usr/bin/env python3
"""Download long-horizon price history from Financial Modeling Prep.

The script fetches daily close prices for the dashboard's asset universe,
covering 2017-01-01 through today (inclusive by default), and stores the
result as JSON under ``static_site/data``.  Later steps in the pipeline can
reuse this historical snapshot when building the precomputed dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "static_site" / "data"
DEFAULT_OUTPUT = DATA_DIR / "historical_prices.json"
DEFAULT_START = date(2017, 1, 1)
API_KEY = os.environ.get("FMP_API_KEY", "").strip()
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3/historical-price-full"
USER_AGENT = "market-stability-dashboard/1.0 (+https://github.com/)"


@dataclass(frozen=True)
class Asset:
    symbol: str
    label: str
    category: str
    fmp_symbol: str


ASSETS: Sequence[Asset] = (
    Asset("QQQ", "QQQ (NASDAQ 100 ETF)", "stock", "QQQ"),
    Asset("IWM", "IWM (Russell 2000 ETF)", "stock", "IWM"),
    Asset("SPY", "SPY (S&P 500 ETF)", "stock", "SPY"),
    Asset("TLT", "TLT (US Long Treasury)", "bond", "TLT"),
    Asset("GLD", "GLD (Gold ETF)", "gold", "GLD"),
    Asset("BTC-USD", "BTC-USD (Bitcoin)", "crypto", "BTCUSD"),
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


def fetch_symbol_history(symbol: str, start: date, end: date) -> List[Tuple[str, float, Optional[float]]]:
    """Return sorted (date, adj_close, adj_open) tuples between start and end inclusive."""

    asset = next((candidate for candidate in ASSETS if candidate.symbol == symbol), None)
    if asset is None:
        raise RuntimeError(f"Unknown asset symbol {symbol}.")

    params = {
        "apikey": API_KEY,
        "from": start.isoformat(),
        "to": end.isoformat(),
    }
    url = f"{FMP_BASE_URL}/{asset.fmp_symbol}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"FMP request for {symbol} failed with HTTP {response.status}.")
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"FMP request for {symbol} returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"FMP request for {symbol} failed: {exc}") from exc

    rows = payload.get("historical") or []
    if not rows:
        message = payload.get("Error Message") or payload.get("Note")
        if message:
            raise RuntimeError(f"FMP returned no data for {symbol}: {message}")
        raise RuntimeError(f"FMP returned no data for {symbol}.")

    start_key = start.isoformat()
    end_key = end.isoformat()
    dedup: Dict[str, Tuple[float, Optional[float]]] = {}
    for row in rows:
        date_key = row.get("date")
        if not isinstance(date_key, str):
            continue
        if date_key < start_key or date_key > end_key:
            continue
        adj_close = row.get("adjClose")
        close = row.get("close")
        open_raw = row.get("open")
        price = None
        try:
            if isinstance(adj_close, (int, float)):
                price = float(adj_close)
            elif isinstance(close, (int, float)):
                price = float(close)
        except (TypeError, ValueError):
            price = None
        if price is None:
            continue
        adj_open = None
        if isinstance(open_raw, (int, float)):
            if isinstance(adj_close, (int, float)) and isinstance(close, (int, float)) and close != 0:
                factor = float(adj_close) / float(close)
                adj_open = float(open_raw) * factor
            else:
                adj_open = float(open_raw)
        dedup[date_key] = (price, adj_open)

    if not dedup:
        raise RuntimeError(f"FMP returned no usable closing prices for {symbol}.")

    ordered: List[Tuple[str, float, Optional[float]]] = []
    for date_key in sorted(dedup.keys()):
        price, adj_open = dedup[date_key]
        ordered.append((date_key, price, adj_open))
    return ordered


def build_payload(start: date, end: date, history: Dict[str, List[Tuple[str, float, Optional[float]]]]) -> Dict[str, object]:
    assets_payload = []
    for asset in ASSETS:
        rows = history.get(asset.symbol, [])
        if not rows:
            raise RuntimeError(f"Missing history for {asset.symbol}.")
        dates = [row[0] for row in rows]
        closes = [row[1] for row in rows]
        opens = [row[2] if row[2] is not None else row[1] for row in rows]
        assets_payload.append(
            {
                "symbol": asset.symbol,
                "label": asset.label,
                "category": asset.category,
                "dates": list(dates),
                "prices": [round(float(price), 6) for price in closes],
                "opens": [round(float(opn), 6) if opn is not None else None for opn in opens],
            }
        )

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "source": "financialmodelingprep",
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

    if not API_KEY:
        raise SystemExit("Environment variable FMP_API_KEY is not set. Please export it before running.")

    history: Dict[str, List[Tuple[str, float]]] = {}
    for asset in ASSETS:
        print(f"[fmp] downloading {asset.symbol} from {start} to {end}")
        rows = fetch_symbol_history(asset.symbol, start, end)
        history[asset.symbol] = rows
        print(f"[fmp] {asset.symbol}: {len(rows)} rows")

    payload = build_payload(start, end, history)
    write_payload(args.out, payload, force=args.force)
    try:
        display_path = args.out.relative_to(ROOT)
    except ValueError:
        display_path = args.out
    print(f"Wrote {display_path}")


if __name__ == "__main__":
    main()
