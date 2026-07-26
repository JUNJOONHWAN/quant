#!/usr/bin/env python3
"""Hermes Worker application for point-in-time market-structure analysis."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sqlite3
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from quant_dataset.point_in_time import (  # noqa: E402
    ETF_FLOW_POLICY_ID,
    derive_etf_flow_available_session,
)


DEFAULT_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/"
    "normalized/daily_observations.sqlite3"
)
DEFAULT_UNIVERSE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/state/universe/"
    "fmp_us_equity_etf_20260714.symbols.txt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle"
)
SOURCE = "fmp"
NORMALIZATION_SESSIONS = 504
ANALOG_COUNT = 30
ANALOG_EMBARGO_SESSIONS = 10
STATE_FEATURES = (
    "breadth_up_1d",
    "breadth_up_5d",
    "breadth_up_20d",
    "breadth_up_63d",
    "median_ret_20d",
    "median_ret_63d",
    "dispersion_20d",
    "left_tail_20d",
    "median_drawdown",
    "deep_damage_frac",
    "liquidity_weighted_ret_20d",
    "leadership_gap_20d",
    "liquidity_hhi",
    "common_factor_20d",
    "flow_net_5d_usd",
    "flow_net_20d_usd",
    "flow_balance_5d",
    "flow_balance_20d",
    "qqq_spy_rel_20d",
    "iwm_spy_rel_20d",
    "rsp_spy_rel_20d",
    "hyg_tlt_rel_20d",
    "xlk_xlu_rel_20d",
    "dbc_tlt_rel_20d",
)


class OracleError(RuntimeError):
    """Raised when the deterministic application contract cannot be satisfied."""


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_ratio(mask: np.ndarray, finite: np.ndarray) -> np.ndarray:
    numerator = np.sum(mask & finite, axis=1)
    denominator = np.sum(finite, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _horizon_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    result = np.full(close.shape, np.nan, dtype=np.float32)
    base = close[:-horizon].astype(np.float64)
    future = close[horizon:].astype(np.float64)
    valid = np.isfinite(base) & np.isfinite(future) & (base > 0)
    ratio = np.divide(
        future,
        base,
        out=np.full(base.shape, np.nan, dtype=np.float64),
        where=valid,
    )
    computed = ratio - 1.0
    computed[(computed < -0.999999) | (computed > 1000.0)] = np.nan
    result[horizon:] = computed.astype(np.float32)
    return result


def _winsorize_rows(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower = np.nanpercentile(values, 1.0, axis=1)
        upper = np.nanpercentile(values, 99.0, axis=1)
    return np.clip(values, lower[:, None], upper[:, None])


def _robust_causal_z(features: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=features.index, columns=features.columns, dtype=float)
    minimum = 126
    for column in features.columns:
        values = features[column].astype(float)
        median = values.rolling(
            NORMALIZATION_SESSIONS, min_periods=minimum
        ).median()
        deviation = (values - median).abs()
        mad = deviation.rolling(
            NORMALIZATION_SESSIONS, min_periods=minimum
        ).median()
        scale = (1.4826 * mad).replace(0.0, np.nan)
        zscore = (values - median) / scale
        fallback = values.expanding(min_periods=minimum).std()
        zscore = zscore.where(scale.notna(), (values - median) / fallback)
        output[column] = zscore.clip(-6.0, 6.0)
    return output


def _select_nonoverlapping(
    candidates: np.ndarray,
    distances: np.ndarray,
    count: int,
    embargo: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    selected_distances: list[float] = []
    for order_index in np.argsort(distances):
        candidate = int(candidates[order_index])
        if all(abs(candidate - prior) > embargo for prior in selected):
            selected.append(candidate)
            selected_distances.append(float(distances[order_index]))
            if len(selected) >= count:
                break
    return np.asarray(selected, dtype=int), np.asarray(selected_distances, dtype=float)


def _weighted_distribution(
    values: np.ndarray, distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values) & np.isfinite(distances)
    clean_values = values[finite]
    clean_distances = distances[finite]
    if clean_values.size == 0:
        return clean_values, clean_distances
    positive = clean_distances[clean_distances > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    weights = np.exp(-clean_distances / max(scale, 1e-9))
    weights /= weights.sum()
    return clean_values, weights


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> list[float]:
    if values.size == 0:
        return [math.nan for _ in quantiles]
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return [
        float(np.interp(quantile, cumulative, values[order]))
        for quantile in quantiles
    ]


def _forward_path(
    benchmark: np.ndarray, start_index: int, horizon: int
) -> tuple[float, float]:
    if start_index + horizon >= benchmark.size:
        return math.nan, math.nan
    base = benchmark[start_index]
    path = benchmark[start_index + 1 : start_index + horizon + 1]
    if not np.isfinite(base) or base <= 0 or not np.all(np.isfinite(path)):
        return math.nan, math.nan
    returns = path / base - 1.0
    return float(returns[-1]), float(np.min(returns))


def _pct(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value * 100:+.2f}%"


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    if not database_path.is_file():
        raise OracleError(f"database missing: {database_path}")
    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro", uri=True, timeout=60
    )
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def _preflight(
    database_path: Path, universe_path: Path, as_of: str | None
) -> dict[str, Any]:
    if not universe_path.is_file():
        raise OracleError(f"universe missing: {universe_path}")
    connection = _connect_read_only(database_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"daily_observations", "etf_flow_observations", "quality_checks"}
        missing = sorted(required - tables)
        if missing:
            raise OracleError(f"required tables missing: {missing}")
        latest_daily = connection.execute(
            "SELECT MAX(trade_date) FROM daily_observations WHERE source=?",
            (SOURCE,),
        ).fetchone()[0]
        latest_flow = connection.execute(
            "SELECT MAX(effective_date) FROM etf_flow_observations"
        ).fetchone()[0]
    finally:
        connection.close()
    if as_of and (not latest_daily or as_of > latest_daily):
        raise OracleError(
            f"requested as-of {as_of} is after latest daily observation {latest_daily}"
        )
    symbols = {
        line.strip().upper()
        for line in universe_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    anchors = {"SPY", "QQQ", "IWM", "RSP", "HYG", "TLT", "XLK", "XLU", "DBC"}
    missing_anchors = sorted(anchors - symbols)
    if missing_anchors:
        raise OracleError(f"universe anchors missing: {missing_anchors}")
    return {
        "status": "PREFLIGHT_PASS",
        "app_id": os.environ.get(
            "OPERATIONS_APP_ID", "market-structure-oracle"
        ),
        "database": str(database_path),
        "database_size": database_path.stat().st_size,
        "universe": str(universe_path),
        "universe_symbols": len(symbols),
        "latest_daily_date": latest_daily,
        "latest_flow_effective_date": latest_flow,
        "requested_as_of": as_of,
        "flow_policy_id": ETF_FLOW_POLICY_ID,
        "operations_app_run_id": os.environ.get("OPERATIONS_APP_RUN_ID"),
    }


def _load_calendar(
    connection: sqlite3.Connection, as_of: str | None
) -> list[str]:
    upper_clause = " AND trade_date<=?" if as_of else ""
    parameters: tuple[Any, ...] = (SOURCE, as_of) if as_of else (SOURCE,)
    rows = connection.execute(
        f"""
        SELECT trade_date
        FROM daily_observations
        WHERE source=? AND symbol IN ('SPY','QQQ')
          AND volume>0 AND COALESCE(adjusted_close,close)>0
          {upper_clause}
        GROUP BY trade_date
        HAVING COUNT(DISTINCT symbol)=2
        ORDER BY trade_date
        """,
        parameters,
    ).fetchall()
    dates = [str(row[0]) for row in rows]
    if len(dates) <= NORMALIZATION_SESSIONS + 63:
        raise OracleError(
            f"insufficient shared SPY/QQQ sessions: {len(dates)}"
        )
    return dates


def _load_daily_matrices(
    connection: sqlite3.Connection,
    dates: list[str],
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    date_to_index = {date: index for index, date in enumerate(dates)}
    symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
    close = np.full((len(dates), len(symbols)), np.nan, dtype=np.float32)
    volume = np.full((len(dates), len(symbols)), np.nan, dtype=np.float32)
    query = connection.execute(
        """
        SELECT d.symbol, d.trade_date,
               COALESCE(d.adjusted_close,d.close), d.volume
        FROM daily_observations d
        LEFT JOIN quality_checks q
          ON q.symbol=d.symbol AND q.trade_date=d.trade_date
        WHERE d.source=? AND d.trade_date<=?
          AND (q.status IS NULL OR q.status!='invalid')
        ORDER BY d.trade_date,d.symbol
        """,
        (SOURCE, dates[-1]),
    )
    scanned = 0
    while True:
        batch = query.fetchmany(250_000)
        if not batch:
            break
        for symbol, trade_date, price, shares in batch:
            date_index = date_to_index.get(str(trade_date))
            symbol_index = symbol_to_index.get(str(symbol).upper())
            if date_index is None or symbol_index is None:
                continue
            if price is not None and float(price) > 0:
                close[date_index, symbol_index] = float(price)
            if shares is not None and float(shares) >= 0:
                volume[date_index, symbol_index] = float(shares)
        scanned += len(batch)
        if scanned % 5_000_000 < 250_000:
            print(
                json.dumps(
                    {"progress": "daily_rows", "scanned": scanned},
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
    return close, volume, scanned


def _flow_features(
    connection: sqlite3.Connection,
    dates: list[str],
    date_index: pd.Index,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_groups = connection.execute(
        """
        SELECT effective_date,processed_date,
               SUM(COALESCE(fund_flow,0.0)),
               SUM(CASE WHEN fund_flow>0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN fund_flow<0 THEN 1 ELSE 0 END),
               COUNT(fund_flow),COUNT(DISTINCT ticker),COUNT(*)
        FROM etf_flow_observations
        WHERE effective_date<=?
        GROUP BY effective_date,processed_date
        ORDER BY effective_date,processed_date
        """,
        (dates[-1],),
    ).fetchall()
    visible: list[dict[str, Any]] = []
    for row in source_groups:
        available = derive_etf_flow_available_session(row[0], row[1], dates)
        if available is None or available > dates[-1]:
            continue
        visible.append(
            {
                "date": pd.Timestamp(available),
                "effective_date": str(row[0]),
                "net_flow": float(row[2] or 0.0),
                "positive_count": int(row[3] or 0),
                "negative_count": int(row[4] or 0),
                "flow_count": int(row[5] or 0),
                "ticker_count": int(row[6] or 0),
                "record_count": int(row[7] or 0),
            }
        )
    if not visible:
        raise OracleError("no ETF Flow rows are visible under the D+2 policy")
    visible_frame = pd.DataFrame(visible)
    grouped = visible_frame.groupby("date").agg(
        net_flow=("net_flow", "sum"),
        positive_count=("positive_count", "sum"),
        negative_count=("negative_count", "sum"),
        flow_count=("flow_count", "sum"),
        ticker_count=("ticker_count", "sum"),
    )
    grouped = grouped.reindex(date_index)
    numeric_columns = [
        "net_flow",
        "positive_count",
        "negative_count",
        "flow_count",
        "ticker_count",
    ]
    grouped[numeric_columns] = grouped[numeric_columns].fillna(0.0)
    balance = (
        (grouped["positive_count"] - grouped["negative_count"])
        / grouped["flow_count"].replace(0.0, np.nan)
    )
    features = pd.DataFrame(index=date_index)
    features["flow_net_5d_usd"] = grouped["net_flow"].rolling(
        5, min_periods=1
    ).sum()
    features["flow_net_20d_usd"] = grouped["net_flow"].rolling(
        20, min_periods=1
    ).sum()
    features["flow_balance_5d"] = balance.rolling(5, min_periods=1).mean()
    features["flow_balance_20d"] = balance.rolling(20, min_periods=1).mean()
    coverage = {
        "policy_id": ETF_FLOW_POLICY_ID,
        "visible_records": int(visible_frame["record_count"].sum()),
        "latest_available_session": visible_frame["date"].max().strftime(
            "%Y-%m-%d"
        ),
        "latest_effective_date_visible": str(
            visible_frame["effective_date"].max()
        ),
    }
    return features, coverage


def _build_features(
    connection: sqlite3.Connection,
    dates: list[str],
    symbols: list[str],
    close: np.ndarray,
    volume: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray, dict[str, int]]:
    symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
    date_index = pd.Index(pd.to_datetime(dates), name="date")
    features = pd.DataFrame(index=date_index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        one_day = _horizon_returns(close, 1)
        finite_one = np.isfinite(one_day)
        features["breadth_up_1d"] = _finite_ratio(one_day > 0, finite_one)
        robust_one_day = _winsorize_rows(one_day)
        equal_weight_return = np.nanmean(robust_one_day, axis=1)
        mean_return_squared = np.nanmean(np.square(robust_one_day), axis=1)

        horizon_cache: dict[int, np.ndarray] = {}
        for horizon in (5, 20, 63):
            returns = _horizon_returns(close, horizon)
            horizon_cache[horizon] = returns
            finite = np.isfinite(returns)
            features[f"breadth_up_{horizon}d"] = _finite_ratio(
                returns > 0, finite
            )
            features[f"median_ret_{horizon}d"] = np.nanmedian(
                returns, axis=1
            )
            if horizon in (20, 63):
                percentile_10 = np.nanpercentile(returns, 10, axis=1)
                percentile_90 = np.nanpercentile(returns, 90, axis=1)
                features[f"dispersion_{horizon}d"] = (
                    percentile_90 - percentile_10
                ) / 2.563
                features[f"left_tail_{horizon}d"] = percentile_10

        running_high = np.maximum.accumulate(
            np.where(np.isfinite(close), close, -np.inf), axis=0
        )
        drawdown = np.divide(
            close,
            running_high,
            out=np.full(close.shape, np.nan, dtype=np.float32),
            where=np.isfinite(close)
            & np.isfinite(running_high)
            & (running_high > 0),
        )
        drawdown -= 1.0
        finite_drawdown = np.isfinite(drawdown)
        features["median_drawdown"] = np.nanmedian(drawdown, axis=1)
        features["deep_damage_frac"] = _finite_ratio(
            drawdown <= -0.20, finite_drawdown
        )

        dollar_volume = close * volume
        robust_return_20 = _winsorize_rows(horizon_cache[20])
        finite_liquidity = (
            np.isfinite(robust_return_20)
            & np.isfinite(dollar_volume)
            & (dollar_volume > 0)
        )
        liquidity_total = np.sum(
            np.where(finite_liquidity, dollar_volume, 0.0), axis=1
        )
        numerator = np.sum(
            np.where(
                finite_liquidity, robust_return_20 * dollar_volume, 0.0
            ),
            axis=1,
        )
        liquidity_return = np.divide(
            numerator,
            liquidity_total,
            out=np.full(len(dates), np.nan),
            where=liquidity_total > 0,
        )
        features["liquidity_weighted_ret_20d"] = liquidity_return
        features["leadership_gap_20d"] = (
            liquidity_return - features["median_ret_20d"].to_numpy()
        )
        squared = np.sum(
            np.where(finite_liquidity, np.square(dollar_volume), 0.0), axis=1
        )
        features["liquidity_hhi"] = np.divide(
            squared,
            np.square(liquidity_total),
            out=np.full(len(dates), np.nan),
            where=liquidity_total > 0,
        )

    market_series = pd.Series(equal_weight_return, index=date_index)
    mean_squared_series = pd.Series(mean_return_squared, index=date_index)
    market_variance = market_series.rolling(20, min_periods=15).var()
    individual_variance = (
        mean_squared_series.rolling(20, min_periods=15).mean()
        - np.square(market_series.rolling(20, min_periods=15).mean())
    )
    features["common_factor_20d"] = (
        market_variance / individual_variance.replace(0.0, np.nan)
    ).clip(0.0, 1.0)

    flow_frame, flow_coverage = _flow_features(
        connection, dates, date_index
    )
    for column in flow_frame:
        features[column] = flow_frame[column]

    def series(symbol: str) -> np.ndarray:
        if symbol not in symbol_to_index:
            raise OracleError(f"anchor missing from universe: {symbol}")
        return close[:, symbol_to_index[symbol]].astype(np.float64)

    pairs = {
        "qqq_spy_rel_20d": ("QQQ", "SPY"),
        "iwm_spy_rel_20d": ("IWM", "SPY"),
        "rsp_spy_rel_20d": ("RSP", "SPY"),
        "hyg_tlt_rel_20d": ("HYG", "TLT"),
        "xlk_xlu_rel_20d": ("XLK", "XLU"),
        "dbc_tlt_rel_20d": ("DBC", "TLT"),
    }
    for name, (numerator_symbol, denominator_symbol) in pairs.items():
        numerator = series(numerator_symbol)
        denominator = series(denominator_symbol)
        ratio = np.divide(
            numerator,
            denominator,
            out=np.full(len(dates), np.nan),
            where=np.isfinite(numerator)
            & np.isfinite(denominator)
            & (denominator > 0),
        )
        relative = np.full(len(dates), np.nan)
        valid = (
            np.isfinite(ratio[20:])
            & np.isfinite(ratio[:-20])
            & (ratio[:-20] > 0)
        )
        relative[20:][valid] = (
            ratio[20:][valid] / ratio[:-20][valid] - 1.0
        )
        features[name] = relative

    state = features[list(STATE_FEATURES)].replace(
        [np.inf, -np.inf], np.nan
    )
    zscore = _robust_causal_z(state)
    active_counts = np.sum(np.isfinite(close), axis=1)
    coverage = {
        "universe_symbols": len(symbols),
        "symbols_with_observations": int(
            np.sum(np.any(np.isfinite(close), axis=0))
        ),
        "active_symbols_as_of": int(active_counts[-1]),
        **flow_coverage,
    }
    return state, zscore, coverage, series("QQQ"), symbol_to_index


def _current_topology(one_day: np.ndarray) -> dict[str, Any]:
    window = one_day[-63:]
    sufficient = np.sum(np.isfinite(window), axis=0) >= 55
    matrix = window[:, sufficient].astype(np.float64)
    means = np.nanmean(matrix, axis=0)
    row_indices, column_indices = np.where(~np.isfinite(matrix))
    matrix[row_indices, column_indices] = means[column_indices]
    matrix -= np.mean(matrix, axis=0)
    standard_deviation = np.std(matrix, axis=0, ddof=1)
    matrix = matrix[:, standard_deviation > 0] / standard_deviation[
        standard_deviation > 0
    ]
    count = matrix.shape[1]
    row_sum = np.sum(matrix, axis=1)
    pair_sum = (
        np.sum(np.square(row_sum)) - np.sum(np.square(matrix))
    ) / 2
    average_correlation = float(
        pair_sum / ((63 - 1) * count * (count - 1) / 2)
    )
    gram = matrix @ matrix.T / (63 - 1)
    eigenvalues = np.linalg.eigvalsh(gram)
    return {
        "symbols": int(count),
        "average_pairwise_correlation": average_correlation,
        "top_eigen_share": float(eigenvalues[-1] / np.sum(eigenvalues)),
    }


def _classify_current(
    raw: dict[str, float], zscore: dict[str, float]
) -> dict[str, Any]:
    components = {
        "breadth_20d": zscore.get("breadth_up_20d", 0.0),
        "breadth_63d": zscore.get("breadth_up_63d", 0.0),
        "median_ret_20d": zscore.get("median_ret_20d", 0.0),
        "median_ret_63d": zscore.get("median_ret_63d", 0.0),
        "flow_5d": zscore.get("flow_net_5d_usd", 0.0),
        "flow_20d": zscore.get("flow_net_20d_usd", 0.0),
        "equal_weight_relative": zscore.get("rsp_spy_rel_20d", 0.0),
        "small_cap_relative": zscore.get("iwm_spy_rel_20d", 0.0),
        "credit_relative": zscore.get("hyg_tlt_rel_20d", 0.0),
        "damage_inverse": -zscore.get("deep_damage_frac", 0.0),
        "concentration_inverse": -zscore.get("leadership_gap_20d", 0.0),
    }
    structural_score = float(np.mean(list(components.values())))
    breadth_score = float(
        np.mean(
            [
                zscore.get("breadth_up_20d", 0.0),
                zscore.get("breadth_up_63d", 0.0),
                zscore.get("rsp_spy_rel_20d", 0.0),
                zscore.get("iwm_spy_rel_20d", 0.0),
            ]
        )
    )
    flow_score = float(
        np.mean(
            [
                zscore.get("flow_net_5d_usd", 0.0),
                zscore.get("flow_net_20d_usd", 0.0),
                zscore.get("flow_balance_5d", 0.0),
                zscore.get("flow_balance_20d", 0.0),
            ]
        )
    )
    concentration_score = float(
        np.mean(
            [
                zscore.get("leadership_gap_20d", 0.0),
                zscore.get("liquidity_hhi", 0.0),
                -zscore.get("rsp_spy_rel_20d", 0.0),
            ]
        )
    )
    if structural_score >= 0.75 and breadth_score >= 0.25:
        regime = "broad_risk_on"
    elif structural_score >= 0.25 and concentration_score >= 0.75:
        regime = "narrow_risk_on"
    elif structural_score >= 0.25:
        regime = "fragile_risk_on"
    elif structural_score <= -0.75:
        regime = "structural_risk_off"
    elif structural_score <= -0.25:
        regime = "risk_off_transition"
    else:
        regime = "mixed_transition"
    return {
        "regime": regime,
        "structural_score": structural_score,
        "breadth_score_z": breadth_score,
        "flow_score_z": flow_score,
        "concentration_score_z": concentration_score,
        "risk_score_components": components,
        "raw_features": raw,
        "causal_z_features": zscore,
    }


def _forecast_and_validate(
    dates: list[str],
    state_z: pd.DataFrame,
    benchmark: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    matrix = state_z.to_numpy(dtype=float)
    current = matrix[-1]
    valid_current = np.isfinite(current)
    minimum_features = max(16, int(0.75 * len(STATE_FEATURES)))
    indices = np.arange(len(dates))
    candidate_mask = (
        (indices >= NORMALIZATION_SESSIONS)
        & (indices + 63 < len(dates))
    )
    comparable = np.sum(np.isfinite(matrix) & valid_current, axis=1)
    candidate_mask &= comparable >= minimum_features
    candidates = np.flatnonzero(candidate_mask)
    differences = matrix[candidates] - current
    finite = np.isfinite(differences) & valid_current
    distances = np.sqrt(
        np.nansum(np.square(differences), axis=1)
        / np.maximum(np.sum(finite, axis=1), 1)
    )
    analog_indices, analog_distances = _select_nonoverlapping(
        candidates, distances, ANALOG_COUNT, ANALOG_EMBARGO_SESSIONS
    )

    horizon_definitions = {
        5: {"strong_up": 0.03, "correction": -0.03, "crash": -0.07},
        20: {"strong_up": 0.07, "correction": -0.07, "crash": -0.12},
        63: {"strong_up": 0.12, "correction": -0.12, "crash": -0.20},
    }
    metrics = {
        int(index): {
            horizon: _forward_path(benchmark, int(index), horizon)
            for horizon in horizon_definitions
        }
        for index in analog_indices
    }
    analogs = [
        {
            "rank": rank,
            "state_date": dates[int(index)],
            "distance": float(distance),
            "forward": {
                str(horizon): {
                    "return": metrics[int(index)][horizon][0],
                    "max_drawdown": metrics[int(index)][horizon][1],
                }
                for horizon in horizon_definitions
            },
        }
        for rank, (index, distance) in enumerate(
            zip(analog_indices, analog_distances), start=1
        )
    ]
    forecast: dict[str, Any] = {}
    for horizon, thresholds in horizon_definitions.items():
        outcomes = np.asarray(
            [metrics[int(index)][horizon][0] for index in analog_indices]
        )
        drawdowns = np.asarray(
            [metrics[int(index)][horizon][1] for index in analog_indices]
        )
        clean, weights = _weighted_distribution(outcomes, analog_distances)
        clean_drawdowns = drawdowns[np.isfinite(outcomes)]
        quantiles = _weighted_quantile(
            clean, weights, (0.1, 0.25, 0.5, 0.75, 0.9)
        )
        drawdown_quantiles = _weighted_quantile(
            clean_drawdowns, weights, (0.1, 0.25, 0.5)
        )
        prior_indices = np.arange(
            NORMALIZATION_SESSIONS, len(dates) - horizon
        )
        prior = np.asarray(
            [
                _forward_path(benchmark, int(index), horizon)[0]
                for index in prior_indices
            ]
        )
        prior = prior[np.isfinite(prior)]
        unconditional = {
            "sample_size": int(prior.size),
            "median_return": float(np.median(prior)),
            "probability_positive": float(np.mean(prior > 0)),
            "p10": float(np.quantile(prior, 0.1)),
            "p90": float(np.quantile(prior, 0.9)),
        }
        forecast[str(horizon)] = {
            "sample_size": int(clean.size),
            "return_quantiles": {
                "p10": quantiles[0],
                "p25": quantiles[1],
                "p50": quantiles[2],
                "p75": quantiles[3],
                "p90": quantiles[4],
            },
            "max_drawdown_quantiles": {
                "p10": drawdown_quantiles[0],
                "p25": drawdown_quantiles[1],
                "p50": drawdown_quantiles[2],
            },
            "probability_positive": float(np.sum(weights[clean > 0])),
            "probability_strong_up": float(
                np.sum(weights[clean >= thresholds["strong_up"]])
            ),
            "probability_correction": float(
                np.sum(weights[clean <= thresholds["correction"]])
            ),
            "probability_crash": float(
                np.sum(weights[clean <= thresholds["crash"]])
            ),
            "thresholds": thresholds,
            "unconditional_baseline": unconditional,
            "relative_to_unconditional": {
                "median_return_delta": float(
                    quantiles[2] - unconditional["median_return"]
                ),
                "probability_positive_delta": float(
                    np.sum(weights[clean > 0])
                    - unconditional["probability_positive"]
                ),
            },
        }

    actuals: list[float] = []
    predictions: list[float] = []
    probabilities: list[float] = []
    baseline_probabilities: list[float] = []
    validation_dates: list[str] = []
    for forecast_index in range(
        NORMALIZATION_SESSIONS + 126, len(dates) - 20, 5
    ):
        vector = matrix[forecast_index]
        valid_vector = np.isfinite(vector)
        historical_candidates = np.arange(
            NORMALIZATION_SESSIONS, forecast_index - 20
        )
        historical_matrix = matrix[historical_candidates]
        eligible = (
            np.sum(np.isfinite(historical_matrix) & valid_vector, axis=1)
            >= minimum_features
        )
        historical_candidates = historical_candidates[eligible]
        historical_matrix = historical_matrix[eligible]
        if historical_candidates.size < 30:
            continue
        difference = historical_matrix - vector
        finite_difference = np.isfinite(difference) & valid_vector
        candidate_distances = np.sqrt(
            np.nansum(np.square(difference), axis=1)
            / np.maximum(np.sum(finite_difference, axis=1), 1)
        )
        selected, selected_distances = _select_nonoverlapping(
            historical_candidates,
            candidate_distances,
            20,
            ANALOG_EMBARGO_SESSIONS,
        )
        selected_outcomes = np.asarray(
            [
                _forward_path(benchmark, int(index), 20)[0]
                for index in selected
            ]
        )
        clean, weights = _weighted_distribution(
            selected_outcomes, selected_distances
        )
        actual, _ = _forward_path(benchmark, forecast_index, 20)
        if clean.size < 10 or not np.isfinite(actual):
            continue
        prior_indices = np.arange(
            NORMALIZATION_SESSIONS, forecast_index - 20
        )
        prior = np.asarray(
            [
                _forward_path(benchmark, int(index), 20)[0]
                for index in prior_indices
            ]
        )
        prior = prior[np.isfinite(prior)]
        actuals.append(actual)
        predictions.append(float(np.sum(clean * weights)))
        probabilities.append(float(np.sum(weights[clean > 0])))
        baseline_probabilities.append(float(np.mean(prior > 0)))
        validation_dates.append(dates[forecast_index])

    actual_array = np.asarray(actuals)
    prediction_array = np.asarray(predictions)
    probability_array = np.asarray(probabilities)
    baseline_probability_array = np.asarray(baseline_probabilities)
    actual_up = (actual_array > 0).astype(float)
    brier = float(np.mean(np.square(probability_array - actual_up)))
    baseline_brier = float(
        np.mean(np.square(baseline_probability_array - actual_up))
    )
    validation = {
        "horizon_sessions": 20,
        "forecast_step_sessions": 5,
        "sample_size": int(actual_array.size),
        "start_date": validation_dates[0] if validation_dates else None,
        "end_date": validation_dates[-1] if validation_dates else None,
        "directional_hit_rate": float(
            np.mean((prediction_array > 0) == (actual_array > 0))
        ),
        "always_up_hit_rate": float(np.mean(actual_array > 0)),
        "mean_absolute_error": float(
            np.mean(np.abs(prediction_array - actual_array))
        ),
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "brier_skill_vs_expanding_base": float(1.0 - brier / baseline_brier),
        "prediction_actual_correlation": float(
            np.corrcoef(prediction_array, actual_array)[0, 1]
        ),
        "forecast_confidence": (
            "validated"
            if brier < baseline_brier
            else "scenario_only_no_incremental_probability_skill"
        ),
    }
    return analogs, forecast, validation


def _render_html(payload: dict[str, Any]) -> str:
    current = payload["current_structure"]
    forecast = payload["forecast"]
    validation = payload["walk_forward_validation"]
    forecast_rows = "".join(
        "<tr>"
        f"<td>{horizon} sessions</td>"
        f"<td>{_pct(item['return_quantiles']['p10'])}</td>"
        f"<td>{_pct(item['return_quantiles']['p50'])}</td>"
        f"<td>{_pct(item['return_quantiles']['p90'])}</td>"
        f"<td>{item['probability_positive'] * 100:.1f}%</td>"
        f"<td>{_pct(item['unconditional_baseline']['median_return'])}</td>"
        f"<td>{item['unconditional_baseline']['probability_positive'] * 100:.1f}%</td>"
        "</tr>"
        for horizon, item in forecast.items()
    )
    analog_rows = "".join(
        "<tr>"
        f"<td>{row['rank']}</td><td>{row['state_date']}</td>"
        f"<td>{row['distance']:.3f}</td>"
        f"<td>{_pct(row['forward']['5']['return'])}</td>"
        f"<td>{_pct(row['forward']['20']['return'])}</td>"
        f"<td>{_pct(row['forward']['63']['return'])}</td>"
        "</tr>"
        for row in payload["analogs"][:15]
    )
    strongest = sorted(
        current["causal_z_features"].items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:14]
    feature_rows = "".join(
        f"<tr><td>{html.escape(name)}</td>"
        f"<td>{current['raw_features'].get(name, math.nan):.6f}</td>"
        f"<td>{value:+.2f}</td></tr>"
        for name, value in strongest
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Structure Oracle — {payload['as_of_date']}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0b1020;color:#e8ecf4;margin:0}}
main{{max-width:1080px;margin:auto;padding:32px 20px 72px}}.card{{background:#131a2c;border:1px solid #26314b;border-radius:16px;padding:20px;margin:14px 0}}
h1,h2{{margin:0 0 12px}}.muted{{color:#9aa8c2}}.metric{{font-size:32px;font-weight:750}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #26314b;text-align:right}}
th:first-child,td:first-child{{text-align:left}}code{{color:#93c5fd}}.warn{{color:#fbbf24}}
</style></head><body><main>
<p class="muted">Hermes Worker App · Frozen SOT · ETF Flow D+2 · no external overlay</p>
<h1>Market Structure Oracle</h1><p class="muted">As of {payload['as_of_date']}</p>
<section class="grid">
<div class="card"><div class="muted">구조 레짐</div><div class="metric">{current['regime']}</div></div>
<div class="card"><div class="muted">구조 점수</div><div class="metric">{current['structural_score']:+.2f}z</div></div>
<div class="card"><div class="muted">관측 유니버스</div><div class="metric">{payload['coverage']['symbols_with_observations']:,}</div></div>
<div class="card"><div class="muted">Flow D+2 최신 effective</div><div class="metric">{payload['coverage']['latest_effective_date_visible']}</div></div>
</section>
<section class="card"><h2>조건부 경로와 평상시 기준</h2><table><thead><tr>
<th>기간</th><th>P10</th><th>중앙값</th><th>P90</th><th>상승</th><th>평상시 중앙</th><th>평상시 상승</th>
</tr></thead><tbody>{forecast_rows}</tbody></table></section>
<section class="card"><h2>가장 가까운 과거 상태</h2><table><thead><tr>
<th>#</th><th>상태일</th><th>거리</th><th>5D</th><th>20D</th><th>63D</th>
</tr></thead><tbody>{analog_rows}</tbody></table></section>
<section class="card"><h2>현재를 구분하는 구조 변수</h2><table><thead><tr>
<th>변수</th><th>원값</th><th>과거 대비 z</th></tr></thead><tbody>{feature_rows}</tbody></table></section>
<section class="card"><h2>워크포워드 검증</h2>
<p>20-session · {validation['sample_size']} forecasts · 방향 적중률
<b>{validation['directional_hit_rate']:.1%}</b> · 항상 상승 기준
<b>{validation['always_up_hit_rate']:.1%}</b> · Brier skill
<b>{validation['brier_skill_vs_expanding_base']:+.1%}</b></p>
<p class="warn">{validation['forecast_confidence']}</p></section>
<section class="card"><h2>실행 계약</h2><p>Hermes Worker가 앱을 실행했다.
원천 DB는 read-only이고 특징은 각 날짜까지의 정보만 사용한다.
ETF Flow는 <code>{ETF_FLOW_POLICY_ID}</code>로 노출하며 미래 수익률은 결과 라벨에만 사용한다.</p></section>
</main></body></html>"""


def run_oracle(
    *,
    database_path: Path,
    universe_path: Path,
    output_root: Path,
    as_of: str | None,
) -> dict[str, Any]:
    started = time.monotonic()
    before = database_path.stat()
    symbols = sorted(
        {
            line.strip().upper()
            for line in universe_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    connection = _connect_read_only(database_path)
    try:
        dates = _load_calendar(connection, as_of)
        close, volume, scanned = _load_daily_matrices(
            connection, dates, symbols
        )
        state, state_z, coverage, benchmark, _ = _build_features(
            connection, dates, symbols, close, volume
        )
        invalid_count = connection.execute(
            "SELECT COUNT(*) FROM quality_checks WHERE status='invalid'"
        ).fetchone()[0]
    finally:
        connection.close()

    current_raw = {
        name: float(state.iloc[-1][name])
        for name in STATE_FEATURES
        if np.isfinite(state.iloc[-1][name])
    }
    current_z = {
        name: float(state_z.iloc[-1][name])
        for name in STATE_FEATURES
        if np.isfinite(state_z.iloc[-1][name])
    }
    current = _classify_current(current_raw, current_z)
    current["topology_63d"] = _current_topology(_horizon_returns(close, 1))
    analogs, forecast, validation = _forecast_and_validate(
        dates, state_z, benchmark
    )
    run_id = os.environ.get("OPERATIONS_APP_RUN_ID") or time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime()
    )
    if not re.fullmatch(r"[A-Za-z0-9._-]{8,128}", run_id):
        raise OracleError(f"invalid operations run id: {run_id!r}")
    after = database_path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise OracleError("source database changed during read-only analysis")

    payload: dict[str, Any] = {
        "schema": "quant.market_structure_oracle.v1",
        "app_id": os.environ.get(
            "OPERATIONS_APP_ID", "market-structure-oracle"
        ),
        "operations_app_run_id": run_id,
        "operations_app_trigger": os.environ.get("OPERATIONS_APP_TRIGGER"),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "as_of_date": dates[-1],
        "source_contract": {
            "database": str(database_path),
            "universe": str(universe_path),
            "price_source": SOURCE,
            "flow_policy_id": ETF_FLOW_POLICY_ID,
            "external_overlay": False,
            "lookahead_in_features": False,
            "future_returns_role": "labels_only",
            "database_open_mode": "read_only_query_only",
        },
        "coverage": {
            "calendar_start": dates[0],
            "calendar_end": dates[-1],
            "sessions": len(dates),
            "daily_rows_scanned": scanned,
            "quality_invalid_rows_excluded": int(invalid_count),
            **coverage,
        },
        "current_structure": current,
        "analog_method": {
            "feature_count": len(STATE_FEATURES),
            "features": list(STATE_FEATURES),
            "normalization": (
                f"trailing {NORMALIZATION_SESSIONS}-session median/MAD"
            ),
            "analog_count": len(analogs),
            "analog_embargo_sessions": ANALOG_EMBARGO_SESSIONS,
            "future_label_leakage": False,
        },
        "analogs": analogs,
        "forecast": forecast,
        "walk_forward_validation": validation,
        "source_database_integrity": {
            "before_size": before.st_size,
            "after_size": after.st_size,
            "before_mtime_ns": before.st_mtime_ns,
            "after_mtime_ns": after.st_mtime_ns,
            "unchanged": True,
        },
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    run_directory = output_root / dates[-1] / "runs" / run_id
    json_path = run_directory / "market_structure_oracle.json"
    html_path = run_directory / "market_structure_oracle.html"
    latest_json = output_root / dates[-1] / "latest.json"
    latest_html = output_root / dates[-1] / "latest.html"
    html_document = _render_html(payload)
    _atomic_write(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
    )
    _atomic_write(html_path, html_document)
    payload["outputs"] = {
        "json": str(json_path),
        "html": str(html_path),
        "latest_json": str(latest_json),
        "latest_html": str(latest_html),
        "html_sha256": _sha256(html_path),
    }
    _atomic_write(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
    )
    _atomic_write(
        latest_json,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
    )
    _atomic_write(latest_html, html_document)
    return {
        "status": "PASS",
        "app_id": payload["app_id"],
        "operations_app_run_id": run_id,
        "as_of_date": dates[-1],
        "regime": current["regime"],
        "structural_score": current["structural_score"],
        "forecast_confidence": validation["forecast_confidence"],
        "source_database_unchanged": True,
        "outputs": payload["outputs"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Hermes-managed market-structure Oracle."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--as-of")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    preflight_only = args.preflight or os.environ.get(
        "MARKET_STRUCTURE_ORACLE_PREFLIGHT_ONLY"
    ) == "1"
    try:
        if preflight_only:
            result = _preflight(args.database, args.universe, args.as_of)
        else:
            result = run_oracle(
                database_path=args.database,
                universe_path=args.universe,
                output_root=args.output_root,
                as_of=args.as_of,
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "app_id": os.environ.get(
                        "OPERATIONS_APP_ID", "market-structure-oracle"
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
