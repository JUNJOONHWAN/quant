"""Shared FMP download and realtime patch helpers for regime engines."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


def _map_symbol_for_fmp(sym: str) -> str:
    if sym == "BTC-USD":
        return "BTCUSD"
    return sym


def fetch_daily_history_fmp(symbols: List[str], from_date: str, api_key: str) -> Dict[str, pd.DataFrame]:
    """Download adjusted close/open history for each symbol from FMP."""
    base = "https://financialmodelingprep.com/api/v3/historical-price-full/"
    out: Dict[str, pd.DataFrame] = {}
    sess = requests.Session()

    def _fetch(url: str) -> List[Dict[str, Any]]:
        r = sess.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("historical") or []

    for sym in symbols:
        mapped = _map_symbol_for_fmp(sym)
        url_primary = f"{base}{mapped}?from={from_date}&apikey={api_key}"
        hist = _fetch(url_primary)
        if not hist:
            url_alt = f"{base}{mapped}?from={from_date}&serietype=line&apikey={api_key}"
            hist = _fetch(url_alt)
        if not hist:
            raise RuntimeError(f"FMP 응답에 {sym} 데이터가 없습니다. API 키 권한을 확인하세요.")
        rows = []
        for row in hist:
            d = row.get("date")
            ac = row.get("adjClose")
            close = ac if isinstance(ac, (int, float)) else row.get("close")
            open_raw = row.get("open")
            if not d or not isinstance(close, (int, float)):
                continue
            if isinstance(ac, (int, float)) and isinstance(row.get("close"), (int, float)) and row.get("close"):
                factor = float(ac) / float(row.get("close"))
                adj_open = float(open_raw) * factor if isinstance(open_raw, (int, float)) else None
            else:
                adj_open = float(open_raw) if isinstance(open_raw, (int, float)) else None
            rows.append(
                {
                    "date": d,
                    "adj_close": float(close),
                    "adj_open": float(adj_open) if adj_open is not None else None,
                }
            )
        if not rows:
            continue
        df = pd.DataFrame(rows)
        dt = pd.to_datetime(df["date"], utc=True, errors="coerce")
        dt = dt.dt.tz_convert(None).dt.normalize()
        df["date"] = dt
        df = df.set_index("date").sort_index()
        out[sym] = df
    return out


def fetch_realtime_quotes_fmp(
    symbols: List[str],
    api_key: str,
    *,
    retries: int = 2,
    timeout: int = 12,
    must_have: Optional[List[str]] = None,
) -> Dict[str, Tuple[pd.Timestamp, float, Optional[float]]]:
    """Fetch realtime quotes with retries and fallbacks."""

    base = "https://financialmodelingprep.com/api/v3/quote/"
    out: Dict[str, Tuple[pd.Timestamp, float, Optional[float]]] = {}
    if not symbols:
        return out
    sess = requests.Session()

    def _parse_rows(rows: List[dict]):
        by_symbol = {row.get("symbol"): row for row in rows if isinstance(row, dict)}
        for sym in symbols:
            key = _map_symbol_for_fmp(sym)
            row = by_symbol.get(key)
            if not row:
                continue
            price_raw = row.get("price")
            prev_close_raw = row.get("previousClose")
            price: Optional[float] = None
            if isinstance(price_raw, (int, float)) and price_raw > 0:
                price = float(price_raw)
            elif isinstance(prev_close_raw, (int, float)) and prev_close_raw > 0:
                price = float(prev_close_raw)
            else:
                continue
            ts = row.get("timestamp") or row.get("lastUpdated")
            try:
                t = (
                    pd.to_datetime(ts, unit="s", utc=True)
                    if isinstance(ts, (int, float))
                    else pd.to_datetime(ts, utc=True)
                )
            except Exception:
                t = pd.Timestamp.utcnow()
            prev_close = float(prev_close_raw) if isinstance(prev_close_raw, (int, float)) else None
            out[sym] = (t, price, prev_close)

    def _try_batch(batch_syms: List[str]) -> None:
        if not batch_syms:
            return
        q_syms = ",".join([_map_symbol_for_fmp(s) for s in batch_syms])
        url = f"{base}{q_syms}?apikey={api_key}"
        r = sess.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        rows = data if isinstance(data, list) else []
        _parse_rows(rows)

    for k in range(max(1, retries)):
        try:
            _try_batch(symbols)
            if out:
                return out
        except Exception:
            time.sleep(min(1.0 * (k + 1), 2.0))

    chunk = 5
    for i in range(0, len(symbols), chunk):
        batch = symbols[i : i + chunk]
        for k in range(max(1, retries)):
            try:
                _try_batch(batch)
                break
            except Exception:
                time.sleep(min(1.0 * (k + 1), 2.0))

    must_list = []
    for s in (must_have or []):
        if s and (s in symbols) and (s not in must_list):
            must_list.append(s)
    for s in must_list:
        if s in out:
            continue
        try:
            _try_batch([s])
        except Exception:
            continue

    return out


def patch_with_realtime_last_price(
    series_map: Dict[str, pd.Series],
    quotes: Dict[str, Tuple[pd.Timestamp, float, Optional[float]]],
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for sym, s in series_map.items():
        q = quotes.get(sym)
        if not q:
            out[sym] = s.copy()
            continue
        ts, price, _ = q
        day = pd.Timestamp(year=ts.year, month=ts.month, day=ts.day, tz="UTC").tz_convert(None).normalize()
        base = s.copy()
        if day in base.index:
            base.loc[day] = float(price)
        else:
            append = pd.Series([float(price)], index=pd.to_datetime([day]))
            base = pd.concat([base, append]).sort_index()
        out[sym] = base
    return out


def ensure_day_stub(series_map: Dict[str, pd.Series], symbols: List[str], day: pd.Timestamp) -> Dict[str, pd.Series]:
    """Ensure each requested symbol has a row for the given day (copy last close)."""

    out: Dict[str, pd.Series] = {}
    any_has_day = any((s is not None and not s.empty and day in s.index) for s in series_map.values())
    if not any_has_day:
        return series_map
    for sym in symbols:
        base = series_map.get(sym)
        if base is None or base.empty:
            continue
        if day in base.index:
            out[sym] = base
            continue
        last_val = float(base.iloc[-1])
        stub = pd.Series([last_val], index=pd.to_datetime([day]))
        out[sym] = pd.concat([base, stub]).sort_index()
    for sym, s in series_map.items():
        if sym not in out:
            out[sym] = s
    return out


def fetch_yahoo_quotes(symbols: List[str], *, timeout: int = 5) -> Dict[str, Dict[str, Any]]:
    """Fetch quote metadata (pre/post/regular) from Yahoo Finance.

    Yahoo Finance does not provide an official API; this helper leverages the
    public quote endpoint strictly for fallback purposes when FMP lacks
    pre/post-market prices. Source: https://query1.finance.yahoo.com/v7/finance/quote
    """

    if not symbols:
        return {}
    url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + ",".join(symbols)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}
    results = data.get("quoteResponse", {}).get("result", []) if isinstance(data, dict) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        sym = row.get("symbol")
        if not sym:
            continue
        info = {
            "preMarketPrice": row.get("preMarketPrice"),
            "preMarketTime": row.get("preMarketTime"),
            "postMarketPrice": row.get("postMarketPrice"),
            "postMarketTime": row.get("postMarketTime"),
            "regularMarketPrice": row.get("regularMarketPrice"),
        }
        out[sym.upper()] = info
    return out
