"""Execution backtest for the AutoTrade2 ETF RADAR Mac engine.

The live engine uses QQQ 1-hour KIS bars near the VWAP window by default. This
module keeps the same CASH/QQQ/TQQQ target mapping and can simulate it from
historical FMP intraday bars, with an explicit daily-OHLC proxy retained for
comparison.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from market_analysis.etf_radar import (
    DEFAULT_TUNE_HORIZONS,
    DEFAULT_UNIVERSE,
    FlowPoint,
    MarketDataClient,
    PricePoint,
    auto_tune_gostop,
    build_ddm_signal,
    build_nowcast_overlay,
    build_qqq_decision,
    build_radar_rows,
    build_tactical_overlay,
    classify_gostop_decision,
    classify_market_state,
    _historical_quote_map,
    _parse_date,
    _slice_flow_map,
    _slice_price_map,
)


MANAGED_SYMBOLS = ("QQQ", "TQQQ")
DEFAULT_OUTPUT_DIR = Path("sweet_spot_reports") / "backtests"
DEFAULT_INTRADAY_CACHE_DIR = Path("sweet_spot_reports") / "intraday_cache"


@dataclass(frozen=True)
class MinuteBar:
    """One exchange-time intraday bar used by the execution simulator."""

    symbol: str
    date: str
    time: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _price_at(point: Optional[PricePoint], field: str) -> Optional[float]:
    if point is None:
        return None
    value = getattr(point, field, None)
    if value is None:
        value = point.close
    return _safe_float(value, 0.0) or None


def _pct_change(current: float, prior: float) -> Optional[float]:
    if not prior:
        return None
    return (current / prior - 1.0) * 100.0


def _mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _std(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    avg = sum(clean) / len(clean)
    variance = sum((value - avg) ** 2 for value in clean) / (len(clean) - 1)
    return math.sqrt(variance)


def _date_index(points: Sequence[PricePoint]) -> Dict[str, PricePoint]:
    return {point.date[:10]: point for point in points if point.date}


def _minute_date_index(points: Sequence[MinuteBar]) -> Dict[str, List[MinuteBar]]:
    grouped: Dict[str, List[MinuteBar]] = {}
    for point in points:
        grouped.setdefault(point.date, []).append(point)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.timestamp)
    return grouped


def _ordered_dates(points: Sequence[PricePoint]) -> List[str]:
    return sorted({point.date[:10] for point in points if point.date})


def _next_trading_date(dates: Sequence[str], asof_date: str) -> Optional[str]:
    asof = _parse_date(asof_date)
    if not asof:
        return None
    for value in dates:
        parsed = _parse_date(value)
        if parsed and parsed > asof:
            return value
    return None


def _hhmm_to_minutes(value: str) -> int:
    text = str(value or "10:30").strip()
    if ":" in text:
        hour_text, minute_text = text.split(":", 1)
    else:
        compact = "".join(ch for ch in text if ch.isdigit()).zfill(4)
        hour_text, minute_text = compact[:2], compact[2:4]
    return int(hour_text) * 60 + int(minute_text)


def _minute_bar_minutes(bar: MinuteBar) -> int:
    compact = "".join(ch for ch in bar.time if ch.isdigit()).zfill(6)
    return int(compact[:2]) * 60 + int(compact[2:4])


def _minute_bar_completed_minutes(bar: MinuteBar, interval_minutes: int) -> int:
    # FMP intraday timestamps are treated as bar start times; only completed
    # bars are visible to the backtest at a decision timestamp.
    return _minute_bar_minutes(bar) + max(1, int(interval_minutes))


def _minute_to_hhmm(value: int) -> str:
    hour = int(value) // 60
    minute = int(value) % 60
    return f"{hour:02d}:{minute:02d}"


def _minute_price_at(
    grouped: Mapping[str, Sequence[MinuteBar]],
    date: str,
    minute: int,
    *,
    field: str = "close",
    interval_minutes: int = 1,
) -> Optional[float]:
    rows = grouped.get(date) or []
    chosen: Optional[MinuteBar] = None
    for row in rows:
        if _minute_bar_completed_minutes(row, interval_minutes) <= minute:
            chosen = row
        else:
            break
    if chosen is None:
        return None
    return _safe_float(getattr(chosen, field, None), 0.0) or None


def _minute_close_price(
    grouped: Mapping[str, Sequence[MinuteBar]],
    date: str,
) -> Optional[float]:
    rows = grouped.get(date) or []
    if not rows:
        return None
    return rows[-1].close


def _decision_minutes(
    qqq_bars: Sequence[MinuteBar],
    *,
    start_minute: int,
    end_minute: int,
    interval_minutes: int = 1,
) -> List[int]:
    """Return available bar timestamps to test during the VWAP wait window."""

    available = sorted(
        {_minute_bar_completed_minutes(bar, interval_minutes) for bar in qqq_bars}
    )
    minutes: List[int] = []
    minutes.append(start_minute)
    minutes.extend(
        minute for minute in available if start_minute < minute <= end_minute
    )
    return sorted(set(minutes))


def build_trade_signal_from_report(
    report: Mapping[str, Any],
    *,
    buy_hold_symbol: str = "QQQ",
    boost_symbol: str = "TQQQ",
    managed_symbols: Sequence[str] = MANAGED_SYMBOLS,
    tqqq_mode: str = "full",
) -> Dict[str, Any]:
    """Mirror AutoTrade2's ETF RADAR report -> target allocation mapping."""

    qqq_decision = report.get("qqq_decision") or {}
    boost = report.get("tqqq_boost") or {}
    gostop = report.get("gostop") or {}
    state = report.get("market_state") or {}
    qqq_exposure = _safe_float(qqq_decision.get("recommended_exposure_pct"))
    boost_pct = _safe_float(boost.get("tqqq_boost_pct"))
    qqq_alloc = _safe_float(
        boost.get("qqq_alloc_pct"),
        max(0.0, qqq_exposure - boost_pct),
    )
    effective_beta = _safe_float(
        boost.get("effective_beta"),
        (qqq_alloc + boost_pct * 3.0) / 100.0,
    )

    allocations: Dict[str, float] = {}
    if qqq_exposure <= 0:
        mode = "CASH"
        target_symbol = None
        target_pct = 0.0
        reason = "ETF RADAR exposure is 0%; move to cash."
    elif boost_pct > 0:
        mode = "TQQQ"
        target_symbol = boost_symbol
        if tqqq_mode == "sleeve":
            if qqq_alloc > 0:
                allocations[buy_hold_symbol] = min(1.0, qqq_alloc / 100.0)
            allocations[boost_symbol] = min(1.0, boost_pct / 100.0)
            target_pct = sum(allocations.values())
            reason = "ETF RADAR boost sleeve executed as QQQ+TQQQ weights."
        elif tqqq_mode == "beta_match":
            target_pct = min(1.0, max(0.0, effective_beta / 3.0))
            allocations[boost_symbol] = target_pct
            reason = "ETF RADAR effective beta converted to TQQQ notional."
        else:
            target_pct = min(1.0, qqq_exposure / 100.0)
            allocations[boost_symbol] = target_pct
            reason = "ETF RADAR TQQQ boost converts buy-and-hold sleeve to TQQQ."
    else:
        mode = "QQQ"
        target_symbol = buy_hold_symbol
        target_pct = min(1.0, qqq_exposure / 100.0)
        allocations[buy_hold_symbol] = target_pct
        reason = "ETF RADAR keeps or enters QQQ buy-and-hold exposure."

    return {
        "schema": "autotrade2_etf_radar_signal.v1",
        "source": "ohlc_proxy_backtest",
        "source_generated_at_utc": report.get("generated_at_utc"),
        "data_window": report.get("data_window") or {},
        "mode": mode,
        "target_symbol": target_symbol,
        "target_pct": round(target_pct, 4),
        "target_allocations": {
            symbol: round(pct, 4) for symbol, pct in allocations.items()
        },
        "managed_symbols": list(managed_symbols),
        "qqq_decision": qqq_decision,
        "tqqq_boost": boost,
        "gostop": gostop,
        "market_state": state,
        "reason": reason,
        "report_warnings": list(report.get("warnings") or []),
    }


def build_daily_technical_proxy(
    qqq_bar: Optional[PricePoint],
    *,
    reclaim_vwap_bps: float = 10.0,
    breakdown_bps: float = 20.0,
) -> Dict[str, Any]:
    """Approximate the live QQQ VWAP check from one daily OHLC bar.

    This intentionally marks the result as a proxy because daily OHLC uses the
    full session profile, unlike the live KIS intraday VWAP window.
    """

    if qqq_bar is None:
        return {
            "enabled": True,
            "available": False,
            "symbol": "QQQ",
            "reason": "QQQ daily OHLC bar is missing",
            "uses_full_day_ohlc": True,
        }

    close = _price_at(qqq_bar, "close")
    open_price = _price_at(qqq_bar, "open") or close
    high = _price_at(qqq_bar, "high") or close
    low = _price_at(qqq_bar, "low") or close
    if not close or not open_price or not high or not low:
        return {
            "enabled": True,
            "available": False,
            "symbol": "QQQ",
            "date": qqq_bar.date,
            "reason": "QQQ daily OHLC values are incomplete",
            "uses_full_day_ohlc": True,
        }

    vwap_proxy = (high + low + close) / 3.0
    slope_pct = _pct_change(close, open_price) or 0.0
    from_open_pct = slope_pct
    vwap_gap_pct = _pct_change(close, vwap_proxy) or 0.0
    reclaim_pct = float(reclaim_vwap_bps) / 100.0
    breakdown_pct = -float(breakdown_bps) / 100.0
    bullish = vwap_gap_pct >= reclaim_pct and slope_pct >= 0
    weak = vwap_gap_pct <= -reclaim_pct and slope_pct <= 0
    hard_bearish = vwap_gap_pct <= breakdown_pct and slope_pct < 0 and from_open_pct < 0
    return {
        "enabled": True,
        "available": True,
        "symbol": "QQQ",
        "date": qqq_bar.date,
        "open": open_price,
        "high": high,
        "low": low,
        "last": close,
        "close": close,
        "vwap": vwap_proxy,
        "vwap_proxy": vwap_proxy,
        "from_open_pct": from_open_pct,
        "vwap_gap_pct": vwap_gap_pct,
        "slope_pct": slope_pct,
        "bullish": bullish,
        "weak": weak,
        "hard_bearish": hard_bearish,
        "uses_full_day_ohlc": True,
    }


def build_minute_technical_context(
    qqq_bars: Sequence[MinuteBar],
    *,
    current_minute: int,
    bar_interval_minutes: int = 1,
    min_bars: int = 3,
    lookback_bars: int = 60,
    momentum_bars: int = 3,
    reclaim_vwap_bps: float = 10.0,
    breakdown_bps: float = 20.0,
) -> Dict[str, Any]:
    """Compute the live VWAP/slope gate from completed intraday bars."""

    usable = [
        bar
        for bar in qqq_bars
        if _minute_bar_completed_minutes(bar, bar_interval_minutes) <= current_minute
    ]
    if lookback_bars > 0:
        usable = usable[-int(lookback_bars) :]
    if len(usable) < int(min_bars):
        return {
            "enabled": True,
            "available": False,
            "symbol": "QQQ",
            "bars": len(usable),
            "minute": current_minute,
            "reason": (
                f"QQQ {int(bar_interval_minutes)}분봉 부족"
                f"({len(usable)}/{int(min_bars)})"
            ),
            "uses_intraday_bars": True,
            "bar_interval_minutes": int(bar_interval_minutes),
        }

    first = usable[0]
    last = usable[-1]
    typical_weighted = 0.0
    total_volume = 0.0
    for bar in usable:
        volume = max(0.0, float(bar.volume or 0.0))
        typical = (bar.high + bar.low + bar.close) / 3.0
        if volume > 0:
            typical_weighted += typical * volume
            total_volume += volume
    if total_volume > 0:
        vwap = typical_weighted / total_volume
    else:
        vwap = sum(bar.close for bar in usable) / len(usable)

    mom = min(max(1, int(momentum_bars)), len(usable) - 1)
    reference = usable[-1 - mom].close
    slope_pct = _pct_change(last.close, reference) or 0.0
    from_open_pct = _pct_change(last.close, first.open) or 0.0
    vwap_gap_pct = _pct_change(last.close, vwap) or 0.0
    reclaim_pct = float(reclaim_vwap_bps) / 100.0
    breakdown_pct = -float(breakdown_bps) / 100.0
    bullish = vwap_gap_pct >= reclaim_pct and slope_pct >= 0
    weak = vwap_gap_pct <= -reclaim_pct and slope_pct <= 0
    hard_bearish = vwap_gap_pct <= breakdown_pct and slope_pct < 0 and from_open_pct < 0
    return {
        "enabled": True,
        "available": True,
        "symbol": "QQQ",
        "bars": len(usable),
        "date": last.date,
        "time": last.time,
        "timestamp": last.timestamp,
        "completed_through_time": _minute_to_hhmm(
            _minute_bar_completed_minutes(last, bar_interval_minutes)
        ),
        "open": first.open,
        "last": last.close,
        "vwap": vwap,
        "from_open_pct": from_open_pct,
        "vwap_gap_pct": vwap_gap_pct,
        "slope_pct": slope_pct,
        "bullish": bullish,
        "weak": weak,
        "hard_bearish": hard_bearish,
        "uses_intraday_bars": True,
        "bar_interval_minutes": int(bar_interval_minutes),
    }


def execution_permission_from_daily_proxy(
    signal: Mapping[str, Any],
    technical: Mapping[str, Any],
    *,
    buy_hold_symbol: str = "QQQ",
    tech_enabled: bool = True,
    require_for_risk_add: bool = True,
) -> Dict[str, Any]:
    """Apply the AutoTrade2 technical-entry rules to a technical context."""

    mode = str(signal.get("mode") or "CASH").upper()
    qqq_decision = signal.get("qqq_decision") or {}
    boost = signal.get("tqqq_boost") or {}
    qqq_exposure = _safe_float(qqq_decision.get("recommended_exposure_pct"))
    boost_pct = _safe_float(boost.get("tqqq_boost_pct"))

    if not tech_enabled:
        return {
            "allowed": True,
            "allow_buys": True,
            "preserve_symbols": [],
            "mode": mode,
            "reason": "technical proxy disabled",
            "technical": {"enabled": False},
        }

    if mode == "CASH" or qqq_exposure <= 0:
        return {
            "allowed": True,
            "allow_buys": False,
            "preserve_symbols": [],
            "mode": mode,
            "reason": "ETF RADAR avoid/cash signal reduces risk immediately.",
            "technical": dict(technical),
        }

    if not technical.get("available"):
        allow = not require_for_risk_add
        return {
            "allowed": allow,
            "allow_buys": allow,
            "preserve_symbols": [buy_hold_symbol] if mode == "TQQQ" else [],
            "mode": mode,
            "reason": technical.get("reason") or "QQQ technical proxy unavailable",
            "technical": dict(technical),
        }

    if mode == "TQQQ":
        if technical.get("hard_bearish"):
            return {
                "allowed": False,
                "allow_buys": False,
                "preserve_symbols": [buy_hold_symbol],
                "mode": mode,
                "reason": (
                    "ETF RADAR boost is open, but QQQ technical context is hard "
                    "bearish; delay the TQQQ flip and preserve QQQ."
                ),
                "technical": dict(technical),
            }
        return {
            "allowed": True,
            "allow_buys": True,
            "preserve_symbols": [],
            "mode": mode,
            "reason": (
                "ETF RADAR TQQQ boost and QQQ technical context is not hard bearish; "
                "enter aggressively."
            ),
            "technical": dict(technical),
            "boost_pct": boost_pct,
        }

    if mode == "QQQ":
        if technical.get("weak"):
            return {
                "allowed": False,
                "allow_buys": False,
                "preserve_symbols": [],
                "mode": mode,
                "reason": "QQQ technical context is weak; delay new QQQ buys.",
                "technical": dict(technical),
            }
        return {
            "allowed": True,
            "allow_buys": True,
            "preserve_symbols": [],
            "mode": mode,
            "reason": "ETF RADAR QQQ exposure and QQQ technical context is not weak.",
            "technical": dict(technical),
        }

    return {
        "allowed": True,
        "allow_buys": True,
        "preserve_symbols": [],
        "mode": mode,
        "reason": "unknown mode; follow ETF RADAR target allocation.",
        "technical": dict(technical),
    }


def build_point_in_time_report(
    *,
    symbols: Sequence[str],
    price_map: Mapping[str, Sequence[PricePoint]],
    flow_map: Mapping[str, Sequence[FlowPoint]],
    asof_date: str,
    auto_tune: bool = True,
    tune_history_days: int = 180,
    tune_horizons: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build the report slice that AutoTrade2 would map into a trade signal."""

    sliced_prices = _slice_price_map(price_map, asof_date)
    sliced_flows = _slice_flow_map(flow_map, asof_date)
    rows = build_radar_rows(
        symbols=symbols,
        price_map=sliced_prices,
        flow_map=sliced_flows,
        asof_date=asof_date,
    )
    market_state = classify_market_state(rows)
    warnings: List[str] = []
    missing_flow = [row.symbol for row in rows if not sliced_flows.get(row.symbol)]
    missing_price = [row.symbol for row in rows if not sliced_prices.get(row.symbol)]
    if missing_flow:
        warnings.append("Fund flow missing: " + ", ".join(missing_flow))
    if missing_price:
        warnings.append("Price missing: " + ", ".join(missing_price))

    gostop = classify_gostop_decision(rows, market_state, warnings=warnings)
    quote_map = _historical_quote_map(price_map, asof_date)
    nowcast = build_nowcast_overlay(
        symbols=symbols,
        price_map=sliced_prices,
        quote_map=quote_map,
        gostop=asdict(gostop),
    )
    tactical = build_tactical_overlay(gostop=asdict(gostop), nowcast=nowcast)
    ddm_signal = build_ddm_signal(
        symbols=symbols,
        price_map=sliced_prices,
        rows=rows,
        nowcast=nowcast,
        asof_date=asof_date,
    )

    payload: Dict[str, Any] = {
        "generated_at_utc": f"{asof_date}T21:10:00Z",
        "data_window": {"to": asof_date},
        "universe": list(symbols),
        "market_state": market_state,
        "gostop": asdict(gostop),
        "rows": [asdict(row) for row in rows],
        "warnings": warnings,
        "nowcast": nowcast,
        "tactical_overlay": tactical,
        "ddm_signal": ddm_signal,
    }
    if auto_tune:
        payload["auto_tune"] = auto_tune_gostop(
            symbols=symbols,
            price_map=sliced_prices,
            flow_map=sliced_flows,
            rows=rows,
            current_decision=payload["gostop"],
            end_date=asof_date,
            current_context=payload,
            history_days=int(tune_history_days),
            horizons=list(tune_horizons or DEFAULT_TUNE_HORIZONS),
        )
    else:
        payload["auto_tune"] = {"enabled": False, "reason": "disabled_by_backtest"}

    payload["qqq_decision"] = build_qqq_decision(
        gostop=payload["gostop"],
        tactical=tactical,
        nowcast=nowcast,
        auto_tune=payload.get("auto_tune") or {},
        ddm_signal=ddm_signal,
    )
    payload["tqqq_boost"] = (
        ((payload.get("auto_tune") or {}).get("selected") or {}).get("tqqq_boost_decision")
        or (payload.get("auto_tune") or {}).get("tqqq_boost_decision")
        or {"enabled": False, "source": "no_auto_tune", "tqqq_boost_pct": 0}
    )
    return payload


def build_signal_records(
    *,
    symbols: Sequence[str],
    price_map: Mapping[str, Sequence[PricePoint]],
    flow_map: Mapping[str, Sequence[FlowPoint]],
    start_date: str,
    end_date: str,
    auto_tune: bool = True,
    tune_history_days: int = 180,
    tune_horizons: Optional[Sequence[int]] = None,
    tqqq_mode: str = "full",
    limit_signals: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build one point-in-time signal per QQQ trading day."""

    qqq_dates = _ordered_dates(price_map.get("QQQ") or [])
    start = _parse_date(start_date) or dt.date.min
    end = _parse_date(end_date) or dt.date.max
    candidate_dates = [
        value
        for value in qqq_dates
        if start <= (_parse_date(value) or dt.date.min) <= end
        and _next_trading_date(qqq_dates, value) is not None
    ]
    if limit_signals is not None and limit_signals > 0:
        candidate_dates = candidate_dates[-int(limit_signals) :]

    records: List[Dict[str, Any]] = []
    for asof_date in candidate_dates:
        report = build_point_in_time_report(
            symbols=symbols,
            price_map=price_map,
            flow_map=flow_map,
            asof_date=asof_date,
            auto_tune=auto_tune,
            tune_history_days=tune_history_days,
            tune_horizons=tune_horizons,
        )
        signal = build_trade_signal_from_report(report, tqqq_mode=tqqq_mode)
        records.append(
            {
                "asof_date": asof_date,
                "execution_date": _next_trading_date(qqq_dates, asof_date),
                "signal": signal,
                "report": {
                    "gostop": report.get("gostop"),
                    "market_state": report.get("market_state"),
                    "qqq_decision": report.get("qqq_decision"),
                    "tqqq_boost": report.get("tqqq_boost"),
                    "warnings": report.get("warnings") or [],
                },
            }
        )
    return records


def _portfolio_value(
    *,
    cash: float,
    shares: Mapping[str, float],
    price_indexes: Mapping[str, Mapping[str, PricePoint]],
    date: str,
    field: str,
) -> float:
    value = float(cash)
    for symbol, qty in shares.items():
        if not qty:
            continue
        price = _price_at(price_indexes.get(symbol, {}).get(date), field)
        if price is not None:
            value += float(qty) * price
    return value


def simulate_autotrade2_ohlc_execution(
    signal_records: Sequence[Mapping[str, Any]],
    *,
    price_map: Mapping[str, Sequence[PricePoint]],
    initial_cash: float = 100_000.0,
    buy_hold_symbol: str = "QQQ",
    boost_symbol: str = "TQQQ",
    managed_symbols: Sequence[str] = MANAGED_SYMBOLS,
    tech_enabled: bool = True,
    require_for_risk_add: bool = True,
    reclaim_vwap_bps: float = 10.0,
    breakdown_bps: float = 20.0,
) -> Dict[str, Any]:
    """Simulate next-session open execution using daily OHLC technical proxy."""

    price_indexes = {
        symbol: _date_index(points) for symbol, points in price_map.items()
    }
    qqq_index = price_indexes.get(buy_hold_symbol, {})
    cash = float(initial_cash)
    shares: Dict[str, float] = {symbol: 0.0 for symbol in managed_symbols}
    equity_curve: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    prev_close_equity = float(initial_cash)
    skipped_buy_count = 0
    delayed_count = 0
    preserve_count = 0
    turnover_values: List[float] = []

    sorted_records = sorted(
        signal_records,
        key=lambda item: str(item.get("execution_date") or item.get("asof_date") or ""),
    )

    for record in sorted_records:
        asof_date = str(record.get("asof_date") or "")
        execution_date = str(record.get("execution_date") or "")
        if not asof_date or not execution_date:
            continue
        qqq_bar = qqq_index.get(execution_date)
        if qqq_bar is None:
            continue
        signal = record.get("signal") or {}
        allocations = {
            str(symbol).upper(): float(pct)
            for symbol, pct in (signal.get("target_allocations") or {}).items()
            if _safe_float(pct) > 0
        }
        technical = build_daily_technical_proxy(
            qqq_bar,
            reclaim_vwap_bps=reclaim_vwap_bps,
            breakdown_bps=breakdown_bps,
        )
        execution = execution_permission_from_daily_proxy(
            signal,
            technical,
            buy_hold_symbol=buy_hold_symbol,
            tech_enabled=tech_enabled,
            require_for_risk_add=require_for_risk_add,
        )
        allow_buys = bool(execution.get("allow_buys"))
        preserve_symbols = set(execution.get("preserve_symbols") or [])
        if not allow_buys and allocations:
            delayed_count += 1

        open_value = _portfolio_value(
            cash=cash,
            shares=shares,
            price_indexes=price_indexes,
            date=execution_date,
            field="open",
        )
        target_shares: Dict[str, float] = {}
        for symbol, pct in allocations.items():
            point = price_indexes.get(symbol, {}).get(execution_date)
            price = _price_at(point, "open")
            if price:
                target_shares[symbol] = (open_value * float(pct)) / price

        day_trades: List[Dict[str, Any]] = []
        day_turnover = 0.0
        # Sell risk first, mirroring the live engine's flip protection flow.
        for symbol in managed_symbols:
            current_qty = float(shares.get(symbol, 0.0) or 0.0)
            desired_qty = float(target_shares.get(symbol, 0.0) or 0.0)
            if symbol in preserve_symbols and current_qty > desired_qty:
                preserve_count += 1
                skipped = {
                    "date": execution_date,
                    "asof_date": asof_date,
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": current_qty - desired_qty,
                    "price": _price_at(price_indexes.get(symbol, {}).get(execution_date), "open"),
                    "skipped": True,
                    "reason": execution.get("reason"),
                }
                trades.append(skipped)
                day_trades.append(skipped)
                continue
            if current_qty > desired_qty:
                price = _price_at(price_indexes.get(symbol, {}).get(execution_date), "open")
                if not price:
                    continue
                qty = current_qty - desired_qty
                cash += qty * price
                shares[symbol] = desired_qty
                day_turnover += qty * price
                trade = {
                    "date": execution_date,
                    "asof_date": asof_date,
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": qty,
                    "price": price,
                    "notional": qty * price,
                    "skipped": False,
                    "reason": execution.get("reason"),
                }
                trades.append(trade)
                day_trades.append(trade)

        for symbol, desired_qty in target_shares.items():
            current_qty = float(shares.get(symbol, 0.0) or 0.0)
            if desired_qty <= current_qty:
                continue
            price = _price_at(price_indexes.get(symbol, {}).get(execution_date), "open")
            if not price:
                continue
            if not allow_buys:
                skipped_buy_count += 1
                skipped = {
                    "date": execution_date,
                    "asof_date": asof_date,
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": desired_qty - current_qty,
                    "price": price,
                    "skipped": True,
                    "reason": execution.get("reason"),
                }
                trades.append(skipped)
                day_trades.append(skipped)
                continue
            qty = min(desired_qty - current_qty, cash / price if price else 0.0)
            if qty <= 0:
                continue
            cash -= qty * price
            shares[symbol] = current_qty + qty
            day_turnover += qty * price
            trade = {
                "date": execution_date,
                "asof_date": asof_date,
                "symbol": symbol,
                "side": "BUY",
                "quantity": qty,
                "price": price,
                "notional": qty * price,
                "skipped": False,
                "reason": execution.get("reason"),
            }
            trades.append(trade)
            day_trades.append(trade)

        close_value = _portfolio_value(
            cash=cash,
            shares=shares,
            price_indexes=price_indexes,
            date=execution_date,
            field="close",
        )
        daily_return = close_value / prev_close_equity - 1.0 if prev_close_equity else 0.0
        prev_close_equity = close_value
        turnover_pct = day_turnover / open_value * 100.0 if open_value else 0.0
        turnover_values.append(turnover_pct)
        close_positions: Dict[str, float] = {}
        for symbol in managed_symbols:
            price = _price_at(price_indexes.get(symbol, {}).get(execution_date), "close")
            close_positions[symbol] = float(shares.get(symbol, 0.0) or 0.0) * (price or 0.0)
        primary_symbol = "CASH"
        if close_positions:
            primary_symbol = max(close_positions, key=lambda key: close_positions[key])
            if close_positions.get(primary_symbol, 0.0) <= close_value * 0.05:
                primary_symbol = "CASH"
        equity_curve.append(
            {
                "date": execution_date,
                "asof_date": asof_date,
                "equity": close_value,
                "daily_return_pct": daily_return * 100.0,
                "cash": cash,
                "cash_pct": cash / close_value * 100.0 if close_value else 0.0,
                "mode": signal.get("mode"),
                "held_symbol": primary_symbol,
                "target_allocations": allocations,
                "allow_buys": allow_buys,
                "delayed": (not allow_buys and bool(allocations)),
                "technical": technical,
                "execution_reason": execution.get("reason"),
                "turnover_pct": turnover_pct,
                "shares": {symbol: shares.get(symbol, 0.0) for symbol in managed_symbols},
                "positions_value": close_positions,
                "orders": day_trades,
            }
        )

    return _build_backtest_summary(
        equity_curve=equity_curve,
        trades=trades,
        price_map=price_map,
        initial_cash=initial_cash,
        buy_hold_symbol=buy_hold_symbol,
        boost_symbol=boost_symbol,
        final_cash=cash,
        final_shares=shares,
        delayed_count=delayed_count,
        skipped_buy_count=skipped_buy_count,
        preserve_count=preserve_count,
        turnover_values=turnover_values,
    )


def simulate_autotrade2_intraday_execution(
    signal_records: Sequence[Mapping[str, Any]],
    *,
    price_map: Mapping[str, Sequence[PricePoint]],
    intraday_map: Mapping[str, Sequence[MinuteBar]],
    initial_cash: float = 100_000.0,
    buy_hold_symbol: str = "QQQ",
    boost_symbol: str = "TQQQ",
    managed_symbols: Sequence[str] = MANAGED_SYMBOLS,
    execution_time_et: str = "10:30",
    max_wait_minutes: int = 20,
    bar_interval_minutes: int = 60,
    tech_enabled: bool = True,
    require_for_risk_add: bool = True,
    min_bars: int = 1,
    lookback_bars: int = 60,
    momentum_bars: int = 3,
    reclaim_vwap_bps: float = 10.0,
    breakdown_bps: float = 20.0,
) -> Dict[str, Any]:
    """Simulate the live VWAP-window engine with historical intraday bars."""

    intraday_indexes = {
        symbol: _minute_date_index(points) for symbol, points in intraday_map.items()
    }
    qqq_by_date = intraday_indexes.get(buy_hold_symbol, {})
    cash = float(initial_cash)
    shares: Dict[str, float] = {symbol: 0.0 for symbol in managed_symbols}
    equity_curve: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    missing_intraday_dates: List[str] = []
    prev_close_equity = float(initial_cash)
    skipped_buy_count = 0
    delayed_count = 0
    preserve_count = 0
    timeout_count = 0
    turnover_values: List[float] = []
    start_minute = _hhmm_to_minutes(execution_time_et)
    end_minute = start_minute + max(0, int(max_wait_minutes))

    sorted_records = sorted(
        signal_records,
        key=lambda item: str(item.get("execution_date") or item.get("asof_date") or ""),
    )

    for record in sorted_records:
        asof_date = str(record.get("asof_date") or "")
        execution_date = str(record.get("execution_date") or "")
        if not asof_date or not execution_date:
            continue
        qqq_session = qqq_by_date.get(execution_date) or []
        if not qqq_session:
            missing_intraday_dates.append(execution_date)
            continue
        signal = record.get("signal") or {}
        allocations = {
            str(symbol).upper(): float(pct)
            for symbol, pct in (signal.get("target_allocations") or {}).items()
            if _safe_float(pct) > 0
        }

        execution: Dict[str, Any] = {}
        decision_minute = start_minute
        delayed_today = False
        minute_candidates = _decision_minutes(
            qqq_session,
            start_minute=start_minute,
            end_minute=end_minute,
            interval_minutes=bar_interval_minutes,
        )
        for minute in minute_candidates:
            technical = build_minute_technical_context(
                qqq_session,
                current_minute=minute,
                bar_interval_minutes=bar_interval_minutes,
                min_bars=min_bars,
                lookback_bars=lookback_bars,
                momentum_bars=momentum_bars,
                reclaim_vwap_bps=reclaim_vwap_bps,
                breakdown_bps=breakdown_bps,
            )
            execution = execution_permission_from_daily_proxy(
                signal,
                technical,
                buy_hold_symbol=buy_hold_symbol,
                tech_enabled=tech_enabled,
                require_for_risk_add=require_for_risk_add,
            )
            decision_minute = minute
            if execution.get("allow_buys") or not allocations:
                break
            delayed_today = True
        if delayed_today:
            delayed_count += 1
        if allocations and not execution.get("allow_buys") and decision_minute >= end_minute:
            timeout_count += 1
            execution = dict(execution)
            execution["timed_out"] = True
            execution["reason"] = (
                f"{execution.get('reason')} · {int(bar_interval_minutes)}분봉 대기시간 초과"
            )

        allow_buys = bool(execution.get("allow_buys"))
        preserve_symbols = set(execution.get("preserve_symbols") or [])
        trade_prices: Dict[str, float] = {}
        for symbol in managed_symbols:
            price = _minute_price_at(
                intraday_indexes.get(symbol, {}),
                execution_date,
                decision_minute,
                field="close",
                interval_minutes=bar_interval_minutes,
            )
            if price is not None:
                trade_prices[symbol] = price
        if buy_hold_symbol not in trade_prices:
            missing_intraday_dates.append(execution_date)
            continue

        trade_value = cash + sum(
            float(shares.get(symbol, 0.0) or 0.0) * trade_prices.get(symbol, 0.0)
            for symbol in managed_symbols
        )
        target_shares: Dict[str, float] = {}
        for symbol, pct in allocations.items():
            price = trade_prices.get(symbol)
            if price:
                target_shares[symbol] = (trade_value * float(pct)) / price

        day_trades: List[Dict[str, Any]] = []
        day_turnover = 0.0
        for symbol in managed_symbols:
            current_qty = float(shares.get(symbol, 0.0) or 0.0)
            desired_qty = float(target_shares.get(symbol, 0.0) or 0.0)
            price = trade_prices.get(symbol)
            if not price:
                continue
            if symbol in preserve_symbols and current_qty > desired_qty:
                preserve_count += 1
                skipped = {
                    "date": execution_date,
                    "time": _minute_to_hhmm(decision_minute),
                    "asof_date": asof_date,
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": current_qty - desired_qty,
                    "price": price,
                    "skipped": True,
                    "reason": execution.get("reason"),
                }
                trades.append(skipped)
                day_trades.append(skipped)
                continue
            if current_qty > desired_qty:
                qty = current_qty - desired_qty
                cash += qty * price
                shares[symbol] = desired_qty
                day_turnover += qty * price
                trade = {
                    "date": execution_date,
                    "time": _minute_to_hhmm(decision_minute),
                    "asof_date": asof_date,
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": qty,
                    "price": price,
                    "notional": qty * price,
                    "skipped": False,
                    "reason": execution.get("reason"),
                }
                trades.append(trade)
                day_trades.append(trade)

        for symbol, desired_qty in target_shares.items():
            current_qty = float(shares.get(symbol, 0.0) or 0.0)
            if desired_qty <= current_qty:
                continue
            price = trade_prices.get(symbol)
            if not price:
                continue
            if not allow_buys:
                skipped_buy_count += 1
                skipped = {
                    "date": execution_date,
                    "time": _minute_to_hhmm(decision_minute),
                    "asof_date": asof_date,
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": desired_qty - current_qty,
                    "price": price,
                    "skipped": True,
                    "reason": execution.get("reason"),
                }
                trades.append(skipped)
                day_trades.append(skipped)
                continue
            qty = min(desired_qty - current_qty, cash / price if price else 0.0)
            if qty <= 0:
                continue
            cash -= qty * price
            shares[symbol] = current_qty + qty
            day_turnover += qty * price
            trade = {
                "date": execution_date,
                "time": _minute_to_hhmm(decision_minute),
                "asof_date": asof_date,
                "symbol": symbol,
                "side": "BUY",
                "quantity": qty,
                "price": price,
                "notional": qty * price,
                "skipped": False,
                "reason": execution.get("reason"),
            }
            trades.append(trade)
            day_trades.append(trade)

        close_value = cash + sum(
            float(shares.get(symbol, 0.0) or 0.0)
            * (_minute_close_price(intraday_indexes.get(symbol, {}), execution_date) or trade_prices.get(symbol, 0.0))
            for symbol in managed_symbols
        )
        daily_return = close_value / prev_close_equity - 1.0 if prev_close_equity else 0.0
        prev_close_equity = close_value
        turnover_pct = day_turnover / trade_value * 100.0 if trade_value else 0.0
        turnover_values.append(turnover_pct)
        close_positions: Dict[str, float] = {}
        for symbol in managed_symbols:
            close_price = _minute_close_price(intraday_indexes.get(symbol, {}), execution_date)
            close_positions[symbol] = float(shares.get(symbol, 0.0) or 0.0) * (close_price or 0.0)
        primary_symbol = max(close_positions, key=lambda key: close_positions[key]) if close_positions else "CASH"
        if close_positions.get(primary_symbol, 0.0) <= close_value * 0.05:
            primary_symbol = "CASH"
        equity_curve.append(
            {
                "date": execution_date,
                "time": _minute_to_hhmm(decision_minute),
                "asof_date": asof_date,
                "equity": close_value,
                "daily_return_pct": daily_return * 100.0,
                "cash": cash,
                "cash_pct": cash / close_value * 100.0 if close_value else 0.0,
                "mode": signal.get("mode"),
                "held_symbol": primary_symbol,
                "target_allocations": allocations,
                "allow_buys": allow_buys,
                "delayed": delayed_today,
                "technical": execution.get("technical") or {},
                "execution_reason": execution.get("reason"),
                "turnover_pct": turnover_pct,
                "shares": {symbol: shares.get(symbol, 0.0) for symbol in managed_symbols},
                "positions_value": close_positions,
                "orders": day_trades,
            }
        )

    result = _build_backtest_summary(
        equity_curve=equity_curve,
        trades=trades,
        price_map=price_map,
        initial_cash=initial_cash,
        buy_hold_symbol=buy_hold_symbol,
        boost_symbol=boost_symbol,
        final_cash=cash,
        final_shares=shares,
        delayed_count=delayed_count,
        skipped_buy_count=skipped_buy_count,
        preserve_count=preserve_count,
        turnover_values=turnover_values,
    )
    result["schema"] = "autotrade2_etf_radar_intraday_backtest.v1"
    result["assumption"] = (
        "ETF RADAR close signal is executed during the next session VWAP window; "
        "technical entry delay is computed from completed historical QQQ intraday bars."
    )
    result["execution_time_et"] = execution_time_et
    result["max_wait_minutes"] = int(max_wait_minutes)
    result["bar_interval_minutes"] = int(bar_interval_minutes)
    result["intraday_source"] = "intraday_bars"
    result["missing_intraday_dates"] = sorted(set(missing_intraday_dates))
    result["timeout_days"] = timeout_count
    return result


def _build_backtest_summary(
    *,
    equity_curve: Sequence[Mapping[str, Any]],
    trades: Sequence[Mapping[str, Any]],
    price_map: Mapping[str, Sequence[PricePoint]],
    initial_cash: float,
    buy_hold_symbol: str,
    boost_symbol: str,
    final_cash: float,
    final_shares: Mapping[str, float],
    delayed_count: int,
    skipped_buy_count: int,
    preserve_count: int,
    turnover_values: Sequence[float],
) -> Dict[str, Any]:
    if not equity_curve:
        return {
            "evaluated_days": 0,
            "initial_cash": initial_cash,
            "note": "No executable signals were available.",
        }

    first_date = str(equity_curve[0]["date"])
    last_date = str(equity_curve[-1]["date"])
    final_equity = float(equity_curve[-1]["equity"])
    total_return_pct = (final_equity / initial_cash - 1.0) * 100.0 if initial_cash else 0.0
    parsed_first = _parse_date(first_date)
    parsed_last = _parse_date(last_date)
    calendar_days = max(1, (parsed_last - parsed_first).days if parsed_first and parsed_last else len(equity_curve))
    cagr_pct = ((final_equity / initial_cash) ** (365.0 / calendar_days) - 1.0) * 100.0 if initial_cash and final_equity > 0 else None

    peak = -math.inf
    max_drawdown_pct = 0.0
    for point in equity_curve:
        equity = float(point.get("equity") or 0.0)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown_pct = min(max_drawdown_pct, (equity / peak - 1.0) * 100.0)

    returns = [float(point.get("daily_return_pct") or 0.0) / 100.0 for point in equity_curve]
    avg_return = _mean(returns) or 0.0
    std_return = _std(returns)
    sharpe = (avg_return / std_return * math.sqrt(252.0)) if std_return else None
    mode_counts: Dict[str, int] = {}
    held_counts: Dict[str, int] = {}
    for point in equity_curve:
        mode_counts[str(point.get("mode") or "UNKNOWN")] = mode_counts.get(str(point.get("mode") or "UNKNOWN"), 0) + 1
        held_counts[str(point.get("held_symbol") or "UNKNOWN")] = held_counts.get(str(point.get("held_symbol") or "UNKNOWN"), 0) + 1

    benchmark = _benchmark_return(price_map, buy_hold_symbol, first_date, last_date)
    boost_benchmark = _benchmark_return(price_map, boost_symbol, first_date, last_date)
    executed_trades = [trade for trade in trades if not trade.get("skipped")]
    skipped_trades = [trade for trade in trades if trade.get("skipped")]
    return {
        "schema": "autotrade2_etf_radar_ohlc_backtest.v1",
        "assumption": (
            "ETF RADAR close signal is executed at the next session open; "
            "QQQ technical entry delay is approximated from the full daily OHLC bar."
        ),
        "initial_cash": initial_cash,
        "first_execution_date": first_date,
        "last_execution_date": last_date,
        "evaluated_days": len(equity_curve),
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_approx": sharpe,
        "benchmark_symbol": buy_hold_symbol,
        "benchmark_return_pct": benchmark,
        "excess_vs_benchmark_pct": total_return_pct - benchmark if benchmark is not None else None,
        "boost_benchmark_symbol": boost_symbol,
        "boost_benchmark_return_pct": boost_benchmark,
        "executed_trade_count": len(executed_trades),
        "skipped_trade_count": len(skipped_trades),
        "delayed_days": delayed_count,
        "skipped_buy_count": skipped_buy_count,
        "preserve_count": preserve_count,
        "avg_turnover_pct": _mean(list(turnover_values)),
        "mode_counts": mode_counts,
        "held_counts": held_counts,
        "final_cash": final_cash,
        "final_shares": dict(final_shares),
        "recent": list(equity_curve[-10:]),
        "equity_curve": list(equity_curve),
        "trades": list(trades),
    }


def _benchmark_return(
    price_map: Mapping[str, Sequence[PricePoint]],
    symbol: str,
    first_date: str,
    last_date: str,
) -> Optional[float]:
    index = _date_index(price_map.get(symbol) or [])
    first = index.get(first_date)
    last = index.get(last_date)
    first_open = _price_at(first, "open")
    last_close = _price_at(last, "close")
    if not first_open or not last_close:
        return None
    return (last_close / first_open - 1.0) * 100.0


def run_autotrade2_ohlc_backtest(
    *,
    symbols: Sequence[str],
    price_map: Mapping[str, Sequence[PricePoint]],
    flow_map: Mapping[str, Sequence[FlowPoint]],
    end_date: str,
    history_days: int = 180,
    initial_cash: float = 100_000.0,
    auto_tune: bool = True,
    tune_history_days: int = 180,
    tune_horizons: Optional[Sequence[int]] = None,
    tqqq_mode: str = "full",
    tech_enabled: bool = True,
    require_for_risk_add: bool = True,
    reclaim_vwap_bps: float = 10.0,
    breakdown_bps: float = 20.0,
    limit_signals: Optional[int] = None,
) -> Dict[str, Any]:
    end = _parse_date(end_date) or dt.date.today()
    start = end - dt.timedelta(days=max(1, int(history_days)))
    signal_records = build_signal_records(
        symbols=symbols,
        price_map=price_map,
        flow_map=flow_map,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        auto_tune=auto_tune,
        tune_history_days=tune_history_days,
        tune_horizons=tune_horizons,
        tqqq_mode=tqqq_mode,
        limit_signals=limit_signals,
    )
    result = simulate_autotrade2_ohlc_execution(
        signal_records,
        price_map=price_map,
        initial_cash=initial_cash,
        tech_enabled=tech_enabled,
        require_for_risk_add=require_for_risk_add,
        reclaim_vwap_bps=reclaim_vwap_bps,
        breakdown_bps=breakdown_bps,
    )
    result["requested_history_days"] = int(history_days)
    result["signal_count"] = len(signal_records)
    result["auto_tune"] = bool(auto_tune)
    result["tune_history_days"] = int(tune_history_days)
    result["tune_horizons"] = list(tune_horizons or DEFAULT_TUNE_HORIZONS)
    result["tqqq_mode"] = tqqq_mode
    result["tech_enabled"] = bool(tech_enabled)
    result["require_for_risk_add"] = bool(require_for_risk_add)
    return result


def run_autotrade2_intraday_backtest(
    *,
    symbols: Sequence[str],
    price_map: Mapping[str, Sequence[PricePoint]],
    flow_map: Mapping[str, Sequence[FlowPoint]],
    intraday_map: Mapping[str, Sequence[MinuteBar]],
    end_date: str,
    history_days: int = 180,
    initial_cash: float = 100_000.0,
    auto_tune: bool = True,
    tune_history_days: int = 180,
    tune_horizons: Optional[Sequence[int]] = None,
    tqqq_mode: str = "full",
    execution_time_et: str = "10:30",
    max_wait_minutes: int = 20,
    bar_interval_minutes: int = 60,
    tech_enabled: bool = True,
    require_for_risk_add: bool = True,
    min_bars: int = 1,
    lookback_bars: int = 60,
    momentum_bars: int = 3,
    reclaim_vwap_bps: float = 10.0,
    breakdown_bps: float = 20.0,
    limit_signals: Optional[int] = None,
) -> Dict[str, Any]:
    end = _parse_date(end_date) or dt.date.today()
    start = end - dt.timedelta(days=max(1, int(history_days)))
    signal_records = build_signal_records(
        symbols=symbols,
        price_map=price_map,
        flow_map=flow_map,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        auto_tune=auto_tune,
        tune_history_days=tune_history_days,
        tune_horizons=tune_horizons,
        tqqq_mode=tqqq_mode,
        limit_signals=limit_signals,
    )
    result = simulate_autotrade2_intraday_execution(
        signal_records,
        price_map=price_map,
        intraday_map=intraday_map,
        initial_cash=initial_cash,
        execution_time_et=execution_time_et,
        max_wait_minutes=max_wait_minutes,
        bar_interval_minutes=bar_interval_minutes,
        tech_enabled=tech_enabled,
        require_for_risk_add=require_for_risk_add,
        min_bars=min_bars,
        lookback_bars=lookback_bars,
        momentum_bars=momentum_bars,
        reclaim_vwap_bps=reclaim_vwap_bps,
        breakdown_bps=breakdown_bps,
    )
    result["requested_history_days"] = int(history_days)
    result["signal_count"] = len(signal_records)
    result["auto_tune"] = bool(auto_tune)
    result["tune_history_days"] = int(tune_history_days)
    result["tune_horizons"] = list(tune_horizons or DEFAULT_TUNE_HORIZONS)
    result["tqqq_mode"] = tqqq_mode
    result["tech_enabled"] = bool(tech_enabled)
    result["require_for_risk_add"] = bool(require_for_risk_add)
    result["intraday_min_bars"] = int(min_bars)
    result["intraday_lookback_bars"] = int(lookback_bars)
    result["intraday_momentum_bars"] = int(momentum_bars)
    return result


def fetch_backtest_market_data(
    *,
    symbols: Sequence[str],
    end_date: str,
    history_days: int,
    lookback_days: int = 260,
    auto_tune: bool = True,
    tune_history_days: int = 180,
    tune_horizons: Optional[Sequence[int]] = None,
    client: Optional[MarketDataClient] = None,
) -> Tuple[Dict[str, List[PricePoint]], Dict[str, List[FlowPoint]], Dict[str, Any]]:
    """Fetch the historical daily inputs needed by the OHLC proxy backtest."""

    end = _parse_date(end_date) or dt.date.today()
    max_horizon = max([1] + [int(value) for value in (tune_horizons or DEFAULT_TUNE_HORIZONS)])
    warmup_days = max(
        int(lookback_days),
        int(history_days) + 45,
        int(history_days) + int(tune_history_days) + max_horizon + 60 if auto_tune else 0,
    )
    start = end - dt.timedelta(days=warmup_days)
    data_client = client or MarketDataClient()
    universe = [symbol.upper() for symbol in symbols]

    price_map: Dict[str, List[PricePoint]] = {}
    for symbol in universe:
        try:
            price_map[symbol] = data_client.fmp_historical_prices(
                symbol,
                from_date=start.isoformat(),
                to_date=end.isoformat(),
            )
        except Exception:
            price_map[symbol] = []

    flow_points = data_client.massive_fund_flows(
        universe,
        processed_date_gte=start.isoformat(),
        limit=min(max(len(universe) * (warmup_days + 20), 1000), 5000),
    )
    flow_map: Dict[str, List[FlowPoint]] = {symbol: [] for symbol in universe}
    for point in flow_points:
        if point.composite_ticker in flow_map:
            flow_map[point.composite_ticker].append(point)
    for points in flow_map.values():
        points.sort(key=lambda item: (item.effective_date, item.processed_date))

    meta = {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "warmup_days": warmup_days,
        "price_symbols": [symbol for symbol, points in price_map.items() if points],
        "flow_symbols": [symbol for symbol, points in flow_map.items() if points],
    }
    return price_map, flow_map, meta


def fetch_fmp_intraday_1m(
    *,
    symbols: Sequence[str],
    end_date: str,
    interval: str = "1hour",
    from_date: Optional[str] = None,
    cache_dir: Path | str = DEFAULT_INTRADAY_CACHE_DIR,
    refresh: bool = False,
    client: Optional[MarketDataClient] = None,
) -> Tuple[Dict[str, List[MinuteBar]], Dict[str, Any]]:
    """Fetch recent FMP api/v3 intraday bars and cache the raw response.

    FMP's legacy api/v3 endpoint is currently usable on this account while the
    stable 1min endpoint is restricted. The endpoint may still return only the
    provider-retained recent sessions, so the simulator evaluates only dates
    present in the returned bars.
    """

    clean_interval = str(interval or "1hour").strip().lower()
    if clean_interval not in {"1min", "5min", "15min", "30min", "1hour"}:
        raise ValueError(f"Unsupported FMP intraday interval: {interval}")
    data_client = client or MarketDataClient()
    if not data_client.fmp_key:
        return {}, {"source": f"fmp_api_v3_{clean_interval}", "error": "missing FMP_API_KEY"}
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    session = data_client.session or requests.Session()
    out: Dict[str, List[MinuteBar]] = {}
    errors: Dict[str, str] = {}
    meta_symbols: Dict[str, Any] = {}
    end = (_parse_date(end_date) or dt.date.today()).isoformat()
    start = from_date

    for symbol in [item.upper() for item in symbols if item]:
        cache_file = cache_path / f"fmp_{clean_interval}_{symbol}_{end}.json"
        rows: Any = None
        if cache_file.exists() and not refresh:
            try:
                rows = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                rows = None
        if rows is None:
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/{clean_interval}/{symbol}"
            params = {"apikey": data_client.fmp_key}
            if start:
                params.update({"from": start, "to": end})
            try:
                response = session.get(url, params=params, timeout=data_client.timeout)
                response.raise_for_status()
                rows = response.json()
                cache_file.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception as exc:
                errors[symbol] = str(exc)[:500]
                rows = []
        bars = _parse_fmp_minute_rows(symbol, rows if isinstance(rows, list) else [])
        out[symbol] = bars
        dates = sorted({bar.date for bar in bars})
        meta_symbols[symbol] = {
            "bars": len(bars),
            "dates": dates,
            "first": bars[0].timestamp if bars else None,
            "last": bars[-1].timestamp if bars else None,
            "cache": str(cache_file),
        }

    return out, {
        "source": f"fmp_api_v3_historical_chart_{clean_interval}",
        "interval": clean_interval,
        "requested_from": start,
        "requested_to": end if start else None,
        "symbols": meta_symbols,
        "errors": errors,
        "note": "FMP may cap api/v3 intraday history to provider-retained sessions.",
    }


def _parse_fmp_minute_rows(symbol: str, rows: Sequence[Mapping[str, Any]]) -> List[MinuteBar]:
    bars: List[MinuteBar] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw_date = str(row.get("date") or "")
        if " " not in raw_date:
            continue
        date_text, time_text = raw_date.split(" ", 1)
        open_price = _safe_float(row.get("open"))
        high_price = _safe_float(row.get("high"))
        low_price = _safe_float(row.get("low"))
        close_price = _safe_float(row.get("close"))
        if close_price <= 0:
            continue
        if open_price <= 0:
            open_price = close_price
        if high_price <= 0:
            high_price = max(open_price, close_price)
        if low_price <= 0:
            low_price = min(open_price, close_price)
        compact_time = "".join(ch for ch in time_text if ch.isdigit()).zfill(6)
        timestamp = f"{date_text} {compact_time[:2]}:{compact_time[2:4]}:{compact_time[4:6]}"
        bars.append(
            MinuteBar(
                symbol=symbol.upper(),
                date=date_text[:10],
                time=f"{compact_time[:2]}:{compact_time[2:4]}:{compact_time[4:6]}",
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=_safe_float(row.get("volume")),
            )
        )
    bars.sort(key=lambda item: item.timestamp)
    return bars


def save_backtest_outputs(
    result: Mapping[str, Any],
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    prefix: str = "autotrade2_etf_radar_ohlc",
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = output_path / f"{prefix}_{stamp}"
    json_path = base.with_suffix(".json")
    equity_path = output_path / f"{base.name}_equity.csv"
    trades_path = output_path / f"{base.name}_trades.csv"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_csv(equity_path, result.get("equity_curve") or [])
    _write_csv(trades_path, result.get("trades") or [])
    return {
        "json": str(json_path),
        "equity_csv": str(equity_path),
        "trades_csv": str(trades_path),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    clean_rows = list(rows)
    if not clean_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in clean_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in clean_rows:
            flat = {
                key: json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(flat)


def default_symbols(extra: Optional[Sequence[str]] = None) -> List[str]:
    symbols = {symbol.upper() for symbol in DEFAULT_UNIVERSE}
    symbols.update(MANAGED_SYMBOLS)
    for symbol in extra or []:
        if symbol:
            symbols.add(symbol.upper())
    return sorted(symbols)
