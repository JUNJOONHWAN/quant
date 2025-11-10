"""Backtest helpers for regime signals (T+1 open execution)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from regime_core.calculations import backtest_from_state


ASSET_DEFAULT = (os.getenv("REGIME_STRATEGY_SYMBOL") or "TQQQ").upper()
BENCH_DEFAULT = (os.getenv("REGIME_BENCH_SYMBOL") or "QQQ").upper()
NEUTRAL_WEIGHT_DEFAULT = float(os.getenv("REGIME_NEUTRAL_BENCH_WEIGHT", "0.33"))
RISK_ON_WEIGHT_DEFAULT = float(os.getenv("REGIME_RISK_ON_BENCH_WEIGHT", "1.0"))
LEVERAGE_DEFAULT = float(os.getenv("REGIME_STRATEGY_LEVERAGE", "3.0"))


def _idx_range(dates: List[str], start_val: Optional[str], end_val: Optional[str]) -> Tuple[int, int]:
    if not dates:
        return (0, -1)
    start_idx = 0
    end_idx = len(dates) - 1
    if start_val:
        for idx, day in enumerate(dates):
            if day >= start_val:
                start_idx = idx
                break
    if end_val:
        for idx in range(len(dates) - 1, -1, -1):
            if dates[idx] <= end_val:
                end_idx = idx
                break
    return (max(0, start_idx), max(start_idx, end_idx))


def _slice(series: Optional[List[Any]], start_idx: int, end_idx: int) -> Optional[List[Any]]:
    if not isinstance(series, list):
        return None
    if end_idx < start_idx:
        return []
    return series[start_idx : end_idx + 1]


def run_tomorrow_open_backtest(
    payload: Dict[str, Any],
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    asset_symbol: Optional[str] = None,
    bench_symbol: Optional[str] = None,
    leverage: Optional[float] = None,
    neutral_weight: Optional[float] = None,
    risk_on_weight: Optional[float] = None,
    execution_delay: int = 1,
) -> Dict[str, Any]:
    """Backtest FFL-STAB regime assuming 전일 신호 → T+1 시초가 집행."""

    asset = (asset_symbol or ASSET_DEFAULT or "TQQQ").upper()
    bench = (bench_symbol or BENCH_DEFAULT or "QQQ").upper()
    lev = leverage if leverage is not None else LEVERAGE_DEFAULT
    neutral = neutral_weight if neutral_weight is not None else NEUTRAL_WEIGHT_DEFAULT
    risk_on = risk_on_weight if risk_on_weight is not None else RISK_ON_WEIGHT_DEFAULT

    dates: List[str] = payload.get("dates", []) or []
    states: List[int] = (payload.get("ffl_stab", {}) or {}).get("state", []) or []
    if not dates or not states:
        raise RuntimeError("payload에 dates/state가 없습니다")

    series = (payload.get("series", {}) or {})
    series_open = (payload.get("series_open", {}) or {})

    bench_close = series.get(bench)
    bench_open = series_open.get(bench)
    asset_close = series.get(asset)
    asset_open = series_open.get(asset)

    if not isinstance(bench_close, list) or not bench_close:
        raise RuntimeError(f"벤치마크({bench}) 시계열이 없습니다")

    start_idx, end_idx = _idx_range(dates, start_date, end_date)
    sliced_dates = dates[start_idx : end_idx + 1]
    sliced_states = states[start_idx : end_idx + 1]
    bench_close_slice = _slice(bench_close, start_idx, end_idx) or []
    bench_open_slice = _slice(bench_open, start_idx, end_idx)
    asset_close_slice = _slice(asset_close, start_idx, end_idx)
    asset_open_slice = _slice(asset_open, start_idx, end_idx)

    bt = backtest_from_state(
        prices_close=bench_close_slice,
        dates=sliced_dates,
        state=sliced_states,
        leverage=int(round(lev)) if lev is not None else 3,
        delay_days=max(1, execution_delay),
        price_mode="open",
        prices_open=bench_open_slice,
        strategy_close=asset_close_slice,
        strategy_open=asset_open_slice,
        benchmark_close=bench_close_slice,
        benchmark_open=bench_open_slice,
        neutral_bench_weight=neutral,
        risk_on_bench_weight=risk_on,
    )

    manifest = {
        "origin": "regime_backtest.tomorrow_open",
        "execution_basis": "T+1_open",
        "price_mode": "open",
        "delay_days": max(1, execution_delay),
        "asset_symbol": asset,
        "bench_symbol": bench,
        "leverage": lev,
        "use_realtime": False,
        "start_date_et": start_date or (sliced_dates[0] if sliced_dates else None),
        "end_date_et": end_date or (sliced_dates[-1] if sliced_dates else None),
        "tz": "America/New_York",
    }

    return {
        "dates": sliced_dates,
        "regime_ffl_stab": sliced_states,
        "equity_strategy": bt.get("equity_strategy", []),
        "equity_benchmark": bt.get("equity_bh", []),
        "equity_asset": bt.get("equity_asset", []),
        "prices_benchmark": bt.get("prices_benchmark", []),
        "prices_strategy": bt.get("prices_strategy", []),
        "manifest": manifest,
    }


__all__ = ["run_tomorrow_open_backtest"]

