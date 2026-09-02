"""Leakage-safe price, target, and FMP fundamental feature engineering."""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import HORIZONS
from .source import parse_json_row


FUNDAMENTAL_ENDPOINTS = (
    "company_information_historical_market_cap",
    "statements_income_statement",
    "statements_balance_sheet_statement",
    "statements_cash_flow_statement",
)


def _finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _safe_divide(numerator: object, denominator: object) -> float:
    num = _finite(numerator)
    den = _finite(denominator)
    if not math.isfinite(num) or not math.isfinite(den) or den == 0:
        return math.nan
    return num / den


def next_session_strictly_after(
    sessions: Sequence[str], reference_date: object, count: int = 1
) -> str | None:
    try:
        normalized = date.fromisoformat(str(reference_date)[:10]).isoformat()
    except (TypeError, ValueError):
        return None
    position = bisect.bisect_right(sessions, normalized) + count - 1
    return sessions[position] if position < len(sessions) else None


def price_frame(rows: Iterable[Mapping], sessions: Sequence[str]) -> pd.DataFrame:
    """Create split-adjusted OHLCV on the canonical SPY session calendar."""

    records = []
    for row in rows:
        close = _finite(row["close"])
        adjusted = _finite(row["adjusted_close"])
        factor = adjusted / close if close > 0 and adjusted > 0 else 1.0
        if not math.isfinite(factor) or factor <= 0:
            factor = 1.0
        records.append(
            {
                "trade_date": str(row["trade_date"]),
                "open": _finite(row["open"]) * factor,
                "high": _finite(row["high"]) * factor,
                "low": _finite(row["low"]) * factor,
                "close": close * factor,
                "volume": _finite(row["volume"]),
            }
        )
    if records:
        frame = pd.DataFrame.from_records(records).set_index("trade_date")
    else:
        frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return frame.reindex(pd.Index(sessions, name="trade_date")).astype(float)


def _rolling_beta(stock: pd.Series, market: pd.Series, window: int) -> pd.Series:
    covariance = stock.rolling(window, min_periods=window).cov(market)
    variance = market.rolling(window, min_periods=window).var()
    return covariance / variance.replace(0.0, np.nan)


def compute_price_features(
    frame: pd.DataFrame,
    benchmark: pd.DataFrame,
    spy: pd.DataFrame,
    qqq: pd.DataFrame,
) -> pd.DataFrame:
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    returns = close.pct_change(fill_method=None)
    log_returns = np.log(close / close.shift(1))
    result = pd.DataFrame(index=frame.index)
    for horizon in (1, 2, 5, 10, 20, 60, 120):
        result[f"ret_{horizon}d"] = (close / close.shift(horizon) - 1.0) * 100.0
    for window in (5, 20, 60):
        result[f"realized_vol_{window}d"] = (
            log_returns.rolling(window, min_periods=window).std() * np.sqrt(252.0) * 100.0
        )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    result["atr_14d_pct"] = (
        true_range.rolling(14, min_periods=14).mean() / close * 100.0
    )
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    relative_strength = gain / loss.replace(0.0, np.nan)
    result["rsi_14d"] = 100.0 - 100.0 / (1.0 + relative_strength)
    result.loc[(loss == 0) & (gain > 0), "rsi_14d"] = 100.0
    for window in (20, 50, 200):
        average = close.rolling(window, min_periods=window).mean()
        result[f"close_to_sma{window}_pct"] = (close / average - 1.0) * 100.0
    for window in (20, 60, 252):
        peak = close.rolling(window, min_periods=min(window, 60)).max()
        result[f"drawdown_{window}d_pct"] = (close / peak - 1.0) * 100.0
    dollar_volume = close * volume
    result["log_dollar_volume_20d"] = np.log1p(
        dollar_volume.rolling(20, min_periods=10).median()
    )
    result["volume_ratio_20d"] = volume / volume.rolling(20, min_periods=10).mean()

    benchmark_returns = benchmark["close"].pct_change(fill_method=None)
    for horizon in (5, 20, 60):
        benchmark_return = (
            benchmark["close"] / benchmark["close"].shift(horizon) - 1.0
        ) * 100.0
        result[f"benchmark_ret_{horizon}d"] = benchmark_return
        result[f"relative_ret_{horizon}d"] = result[f"ret_{horizon}d"] - benchmark_return
    for window in (20, 60):
        result[f"beta_{window}d"] = _rolling_beta(returns, benchmark_returns, window)
        result[f"corr_{window}d"] = returns.rolling(window, min_periods=window).corr(
            benchmark_returns
        )
    result["spy_ret_5d"] = (spy["close"] / spy["close"].shift(5) - 1.0) * 100.0
    result["qqq_ret_5d"] = (qqq["close"] / qqq["close"].shift(5) - 1.0) * 100.0
    result["spy_vol_20d"] = (
        np.log(spy["close"] / spy["close"].shift(1))
        .rolling(20, min_periods=20)
        .std()
        * np.sqrt(252.0)
        * 100.0
    )
    result["qqq_vol_20d"] = (
        np.log(qqq["close"] / qqq["close"].shift(1))
        .rolling(20, min_periods=20)
        .std()
        * np.sqrt(252.0)
        * 100.0
    )
    timestamps = pd.to_datetime(result.index)
    result["month_sin"] = np.sin(2.0 * np.pi * timestamps.month / 12.0)
    result["month_cos"] = np.cos(2.0 * np.pi * timestamps.month / 12.0)
    result["weekday_sin"] = np.sin(2.0 * np.pi * timestamps.dayofweek / 5.0)
    result["weekday_cos"] = np.cos(2.0 * np.pi * timestamps.dayofweek / 5.0)

    for horizon in HORIZONS:
        future_closes = close.shift(-horizon)
        future_high = pd.concat(
            [high.shift(-offset) for offset in range(1, horizon + 1)], axis=1
        ).max(axis=1, skipna=False)
        future_low = pd.concat(
            [low.shift(-offset) for offset in range(1, horizon + 1)], axis=1
        ).min(axis=1, skipna=False)
        result[f"return_{horizon}d_pct"] = (future_closes / close - 1.0) * 100.0
        result[f"upside_{horizon}d_pct"] = (
            (future_high / close - 1.0).clip(lower=0.0) * 100.0
        )
        result[f"loss_{horizon}d_pct"] = (
            (1.0 - future_low / close).clip(lower=0.0) * 100.0
        )
    return result.replace([np.inf, -np.inf], np.nan)


def _statement_features(
    income: Mapping, balance: Mapping, cash_flow: Mapping, prior_income: Mapping
) -> dict[str, float]:
    revenue = income.get("revenue")
    prior_revenue = prior_income.get("revenue")
    total_assets = balance.get("totalAssets")
    current_liabilities = balance.get("totalCurrentLiabilities")
    return {
        "revenue_yoy": (
            (_safe_divide(revenue, prior_revenue) - 1.0) * 100.0
            if math.isfinite(_safe_divide(revenue, prior_revenue))
            else math.nan
        ),
        "net_margin": _safe_divide(income.get("netIncome"), revenue) * 100.0,
        "operating_margin": _safe_divide(income.get("operatingIncome"), revenue)
        * 100.0,
        "gross_margin": _safe_divide(income.get("grossProfit"), revenue) * 100.0,
        "free_cash_flow_margin": _safe_divide(cash_flow.get("freeCashFlow"), revenue)
        * 100.0,
        "operating_cash_flow_margin": _safe_divide(
            cash_flow.get("operatingCashFlow"), revenue
        )
        * 100.0,
        "debt_to_assets": _safe_divide(balance.get("totalDebt"), total_assets),
        "cash_to_assets": _safe_divide(
            balance.get("cashAndShortTermInvestments"), total_assets
        ),
        "current_ratio": _safe_divide(
            balance.get("totalCurrentAssets"), current_liabilities
        ),
    }


def compute_fundamental_features(
    fact_rows: Iterable[Mapping], sessions: Sequence[str]
) -> pd.DataFrame:
    """Project only facts whose conservative availability session has elapsed."""

    market_caps: dict[str, float] = {}
    statement_events: dict[str, dict[str, list[tuple[str, dict]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in fact_rows:
        endpoint = str(row["endpoint_id"])
        payload = parse_json_row(row["row_json"])
        if endpoint == "company_information_historical_market_cap":
            event_date = str(row["event_date"] or payload.get("date") or "")
            value = _finite(payload.get("marketCap"))
            if event_date and math.isfinite(value) and value > 0:
                market_caps[event_date] = value
            continue
        period = str(payload.get("period") or "")
        if period == "FY":
            continue
        available = next_session_strictly_after(sessions, row["available_date"], 1)
        event_date = str(row["event_date"] or payload.get("date") or "")
        if available and event_date:
            statement_events[available][endpoint].append((event_date, payload))

    index = pd.Index(sessions, name="trade_date")
    frame = pd.DataFrame(index=index)
    cap = pd.Series(market_caps, dtype=float).reindex(index)
    frame["log_market_cap"] = np.log(cap.where(cap > 0))
    frame["market_cap_to_20d_avg"] = cap / cap.rolling(20, min_periods=5).mean()

    latest: dict[str, tuple[str, dict]] = {}
    income_history: list[tuple[str, dict]] = []
    output_rows = []
    for session in sessions:
        for endpoint, candidates in statement_events.get(session, {}).items():
            latest[endpoint] = max(candidates, key=lambda item: item[0])
            if endpoint == "statements_income_statement":
                income_history.extend(candidates)
                income_history.sort(key=lambda item: item[0])
        income_date, income = latest.get("statements_income_statement", ("", {}))
        _, balance = latest.get("statements_balance_sheet_statement", ("", {}))
        _, cash_flow = latest.get("statements_cash_flow_statement", ("", {}))
        prior_income = {}
        if income_date:
            cutoff = (date.fromisoformat(income_date) - timedelta(days=365)).isoformat()
            prior_candidates = [item for item in income_history if item[0] <= cutoff]
            if prior_candidates:
                prior_income = prior_candidates[-1][1]
        values = _statement_features(income, balance, cash_flow, prior_income)
        if income_date:
            values["financial_statement_age_days"] = (
                date.fromisoformat(session) - date.fromisoformat(income_date)
            ).days
        else:
            values["financial_statement_age_days"] = math.nan
        required = (
            "net_margin",
            "operating_margin",
            "debt_to_assets",
            "current_ratio",
        )
        values["fundamental_coverage"] = sum(
            math.isfinite(values[key]) for key in required
        ) / len(required)
        output_rows.append(values)
    statement_frame = pd.DataFrame(output_rows, index=index)
    for column in statement_frame:
        frame[column] = statement_frame[column]
    return frame.replace([np.inf, -np.inf], np.nan)
