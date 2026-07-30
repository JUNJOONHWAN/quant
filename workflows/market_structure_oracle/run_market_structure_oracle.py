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
from workflows.market_structure_oracle.incremental_store import (  # noqa: E402
    IncrementalStoreError,
    ensure_oracle_snapshot,
    expected_nyse_sessions,
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
DEFAULT_INCREMENTAL_ROOT = DEFAULT_OUTPUT_ROOT / "incremental"
SOURCE = "fmp"
NORMALIZATION_SESSIONS = 504
ANALOG_COUNT = 30
ANALOG_EMBARGO_SESSIONS = 10
STATE_CUBE_SCHEMA = "quant.market_structure_state_cube.v2"
OUTPUT_SCHEMA = "quant.market_structure_oracle.v3"
SCOPE_REGISTRY = {
    "technology": {
        "label_ko": "기술",
        "etfs": ["XLK"],
        "aliases": ["technology", "tech", "기술", "테크"],
    },
    "semiconductors": {
        "label_ko": "반도체",
        "etfs": ["SMH", "SOXX"],
        "aliases": ["semiconductor", "semiconductors", "semis", "반도체"],
    },
    "communication_services": {
        "label_ko": "커뮤니케이션 서비스",
        "etfs": ["XLC"],
        "aliases": ["communication", "communications", "커뮤니케이션", "통신"],
    },
    "consumer_discretionary": {
        "label_ko": "경기소비재",
        "etfs": ["XLY"],
        "aliases": ["consumer discretionary", "경기소비재", "임의소비재"],
    },
    "consumer_staples": {
        "label_ko": "필수소비재",
        "etfs": ["XLP"],
        "aliases": ["consumer staples", "staples", "필수소비재"],
    },
    "energy": {
        "label_ko": "에너지",
        "etfs": ["XLE"],
        "aliases": ["energy", "에너지"],
    },
    "financials": {
        "label_ko": "금융",
        "etfs": ["XLF"],
        "aliases": ["financial", "financials", "finance", "금융"],
    },
    "healthcare": {
        "label_ko": "헬스케어",
        "etfs": ["XLV"],
        "aliases": ["healthcare", "health care", "헬스케어", "건강관리"],
    },
    "industrials": {
        "label_ko": "산업재",
        "etfs": ["XLI"],
        "aliases": ["industrial", "industrials", "산업재"],
    },
    "materials": {
        "label_ko": "소재",
        "etfs": ["XLB"],
        "aliases": ["material", "materials", "소재"],
    },
    "real_estate": {
        "label_ko": "부동산",
        "etfs": ["XLRE"],
        "aliases": ["real estate", "reit", "reits", "부동산", "리츠"],
    },
    "utilities": {
        "label_ko": "유틸리티",
        "etfs": ["XLU"],
        "aliases": ["utility", "utilities", "유틸리티"],
    },
    "biotechnology": {
        "label_ko": "바이오테크",
        "etfs": ["XBI", "IBB"],
        "aliases": ["biotech", "biotechnology", "바이오", "바이오테크"],
    },
    "cybersecurity": {
        "label_ko": "사이버보안",
        "etfs": ["CIBR", "HACK"],
        "aliases": ["cybersecurity", "cyber security", "사이버보안", "보안"],
    },
    "clean_energy": {
        "label_ko": "청정에너지",
        "etfs": ["ICLN", "TAN"],
        "aliases": ["clean energy", "solar", "청정에너지", "태양광"],
    },
    "defense": {
        "label_ko": "방산",
        "etfs": ["ITA", "XAR"],
        "aliases": ["defense", "aerospace", "방산", "항공우주"],
    },
    "regional_banks": {
        "label_ko": "지역은행",
        "etfs": ["KRE"],
        "aliases": ["regional banks", "regional bank", "지역은행"],
    },
    "homebuilders": {
        "label_ko": "주택건설",
        "etfs": ["XHB", "ITB"],
        "aliases": ["homebuilders", "home builders", "housing", "주택건설"],
    },
    "gold_miners": {
        "label_ko": "금광주",
        "etfs": ["GDX", "GDXJ"],
        "aliases": ["gold miners", "gold mining", "금광", "금광주"],
    },
    "uranium": {
        "label_ko": "우라늄",
        "etfs": ["URA"],
        "aliases": ["uranium", "우라늄"],
    },
}
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


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
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


def _pct(value: float | None) -> str:
    return (
        "N/A"
        if value is None or not np.isfinite(value)
        else f"{value * 100:+.2f}%"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    if not database_path.is_file():
        raise OracleError(f"database missing: {database_path}")
    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro", uri=True, timeout=60
    )
    # ``mode=ro`` makes the attached source immutable while still allowing
    # TEMP views that combine the immutable history and Oracle delta.
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def _connect_oracle_view(
    database_path: Path, incremental_database_path: Path
) -> sqlite3.Connection:
    """Expose immutable FMP history plus Oracle-owned Massive delta as temp views."""
    if not incremental_database_path.is_file():
        raise OracleError(f"incremental database missing: {incremental_database_path}")
    connection = _connect_read_only(database_path)
    connection.execute(
        "ATTACH DATABASE ? AS oracle_incremental",
        (f"file:{incremental_database_path}?mode=ro",),
    )
    connection.executescript(
        """
        CREATE TEMP VIEW oracle_daily_observations AS
          SELECT * FROM main.daily_observations WHERE source='fmp'
          UNION ALL
          SELECT * FROM oracle_incremental.daily_observations
          WHERE source IN ('massive','fmp');
        CREATE TEMP VIEW oracle_quality_checks AS
          SELECT * FROM main.quality_checks
          UNION ALL
          SELECT * FROM oracle_incremental.quality_checks;
        CREATE TEMP VIEW oracle_etf_flow_observations AS
          SELECT * FROM main.etf_flow_observations
          UNION ALL
          SELECT * FROM oracle_incremental.etf_flow_observations;
        """
    )
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
    parameters: tuple[Any, ...] = (as_of,) if as_of else ()
    rows = connection.execute(
        f"""
        SELECT trade_date
        FROM oracle_daily_observations
        WHERE symbol IN ('SPY','QQQ')
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
        FROM oracle_daily_observations d
        LEFT JOIN oracle_quality_checks q
          ON q.symbol=d.symbol AND q.trade_date=d.trade_date
        WHERE d.trade_date<=?
          AND (q.status IS NULL OR q.status!='invalid')
        ORDER BY d.trade_date,d.symbol
        """,
        (dates[-1],),
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
    tickers: list[str] | None = None,
    required: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ticker_clause = ""
    parameters: list[Any] = [dates[-1]]
    if tickers:
        placeholders = ",".join("?" for _ in tickers)
        ticker_clause = f" AND ticker IN ({placeholders})"
        parameters.extend(tickers)
    source_groups = connection.execute(
        f"""
        SELECT effective_date,processed_date,
               SUM(COALESCE(fund_flow,0.0)),
               SUM(CASE WHEN fund_flow>0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN fund_flow<0 THEN 1 ELSE 0 END),
               COUNT(fund_flow),COUNT(DISTINCT ticker),COUNT(*)
        FROM oracle_etf_flow_observations
        WHERE effective_date<=?
          {ticker_clause}
        GROUP BY effective_date,processed_date
        ORDER BY effective_date,processed_date
        """,
        tuple(parameters),
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
        if required:
            raise OracleError(
                "no ETF Flow rows are visible under the D+2 policy"
            )
        empty = pd.DataFrame(index=date_index)
        for column in (
            "flow_net_5d_usd",
            "flow_net_20d_usd",
            "flow_balance_5d",
            "flow_balance_20d",
        ):
            empty[column] = np.nan
        return empty, {
            "policy_id": ETF_FLOW_POLICY_ID,
            "visible_records": 0,
            "latest_available_session": None,
            "latest_effective_date_visible": None,
            "tickers": list(tickers or []),
        }
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
        "tickers": list(tickers or []),
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


def _state_cube_directory(output_root: Path, as_of_date: str) -> Path:
    return output_root / "state_cubes" / as_of_date


def _incremental_data_fingerprint(incremental_database_path: Path) -> dict[str, Any]:
    """Hash data-bearing tables only; audit-run rows must not evict a valid cube."""
    connection = sqlite3.connect(
        f"file:{incremental_database_path}?mode=ro", uri=True
    )
    try:
        daily = connection.execute(
            """SELECT source,trade_date,COUNT(*),MIN(raw_artifact_id),MAX(raw_artifact_id)
               FROM daily_observations GROUP BY source,trade_date ORDER BY source,trade_date"""
        ).fetchall()
        flows = connection.execute(
            """SELECT MAX(effective_date),MAX(processed_date),COUNT(*),
                      COUNT(DISTINCT ticker),MAX(record_hash)
               FROM etf_flow_observations"""
        ).fetchone()
        seals = connection.execute(
            """SELECT target_as_of_date,schema_version,source_contract,receipt_sha256
               FROM oracle_snapshot_seals ORDER BY target_as_of_date"""
        ).fetchall()
    finally:
        connection.close()
    content = json.dumps(
        {"daily": daily, "flows": flows, "snapshot_seals": seals},
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "path": str(incremental_database_path),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "daily_session_count": len(daily),
        "snapshot_seal_count": len(seals),
    }


def _state_cube_fingerprint(
    database_path: Path,
    incremental_database_path: Path,
    universe_path: Path,
    as_of_date: str,
) -> dict[str, Any]:
    stat = database_path.stat()
    return {
        "database": str(database_path),
        "database_size": stat.st_size,
        "database_mtime_ns": stat.st_mtime_ns,
        "incremental_data": _incremental_data_fingerprint(
            incremental_database_path
        ),
        "universe": str(universe_path),
        "universe_sha256": _sha256(universe_path),
        "as_of_date": as_of_date,
        "price_source": "fmp_baseline_plus_massive_incremental",
        "flow_policy_id": ETF_FLOW_POLICY_ID,
    }


def _load_state_cube(
    cube_directory: Path,
    expected_fingerprint: dict[str, Any],
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
] | None:
    metadata_path = cube_directory / "metadata.json"
    required = (
        metadata_path,
        cube_directory / "dates.json",
        cube_directory / "symbols.json",
        cube_directory / "close.npy",
        cube_directory / "volume.npy",
        cube_directory / "global_state.csv",
        cube_directory / "global_state_z.csv",
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != STATE_CUBE_SCHEMA:
            return None
        if metadata.get("source_fingerprint") != expected_fingerprint:
            return None
        dates = json.loads(
            (cube_directory / "dates.json").read_text(encoding="utf-8")
        )
        symbols = json.loads(
            (cube_directory / "symbols.json").read_text(encoding="utf-8")
        )
        close = np.load(
            cube_directory / "close.npy", mmap_mode="r", allow_pickle=False
        )
        volume = np.load(
            cube_directory / "volume.npy", mmap_mode="r", allow_pickle=False
        )
        expected_shape = (len(dates), len(symbols))
        if close.shape != expected_shape or volume.shape != expected_shape:
            return None
        state = pd.read_csv(
            cube_directory / "global_state.csv",
            index_col=0,
            parse_dates=True,
        )
        state_z = pd.read_csv(
            cube_directory / "global_state_z.csv",
            index_col=0,
            parse_dates=True,
        )
        state.index.name = "date"
        state_z.index.name = "date"
        return dates, symbols, close, volume, state, state_z, metadata
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _save_state_cube(
    cube_directory: Path,
    *,
    fingerprint: dict[str, Any],
    dates: list[str],
    symbols: list[str],
    close: np.ndarray,
    volume: np.ndarray,
    state: pd.DataFrame,
    state_z: pd.DataFrame,
    coverage: dict[str, Any],
    daily_rows_scanned: int,
) -> dict[str, Any]:
    cube_directory.mkdir(parents=True, exist_ok=True)
    _atomic_save_array(cube_directory / "close.npy", close)
    _atomic_save_array(cube_directory / "volume.npy", volume)
    _atomic_write(
        cube_directory / "dates.json",
        json.dumps(dates, ensure_ascii=False, separators=(",", ":")),
    )
    _atomic_write(
        cube_directory / "symbols.json",
        json.dumps(symbols, ensure_ascii=False, separators=(",", ":")),
    )
    _atomic_write(cube_directory / "global_state.csv", state.to_csv())
    _atomic_write(cube_directory / "global_state_z.csv", state_z.to_csv())
    metadata = {
        "schema": STATE_CUBE_SCHEMA,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_fingerprint": fingerprint,
        "sessions": len(dates),
        "symbols": len(symbols),
        "matrix_shape": [len(dates), len(symbols)],
        "matrix_dtype": "float32",
        "daily_rows_scanned": int(daily_rows_scanned),
        "coverage": coverage,
        "artifacts": {
            "close": "close.npy",
            "volume": "volume.npy",
            "dates": "dates.json",
            "symbols": "symbols.json",
            "global_state": "global_state.csv",
            "global_state_z": "global_state_z.csv",
        },
    }
    _atomic_write(
        cube_directory / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
    )
    return metadata


def _load_or_build_state_cube(
    connection: sqlite3.Connection,
    *,
    database_path: Path,
    incremental_database_path: Path,
    universe_path: Path,
    output_root: Path,
    as_of: str | None,
    symbols: list[str],
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    bool,
]:
    dates = _load_calendar(connection, as_of)
    cube_directory = _state_cube_directory(output_root, dates[-1])
    fingerprint = _state_cube_fingerprint(
        database_path, incremental_database_path, universe_path, dates[-1]
    )
    cached = _load_state_cube(cube_directory, fingerprint)
    if cached is not None:
        (
            cached_dates,
            cached_symbols,
            close,
            volume,
            state,
            state_z,
            metadata,
        ) = cached
        return (
            cached_dates,
            cached_symbols,
            close,
            volume,
            state,
            state_z,
            metadata,
            True,
        )

    close, volume, scanned = _load_daily_matrices(
        connection, dates, symbols
    )
    state, state_z, coverage, _, _ = _build_features(
        connection, dates, symbols, close, volume
    )
    metadata = _save_state_cube(
        cube_directory,
        fingerprint=fingerprint,
        dates=dates,
        symbols=symbols,
        close=close,
        volume=volume,
        state=state,
        state_z=state_z,
        coverage=coverage,
        daily_rows_scanned=scanned,
    )
    return dates, symbols, close, volume, state, state_z, metadata, False


def _load_request(request_file: Path | None) -> dict[str, Any] | None:
    path = request_file
    if path is None:
        raw = os.environ.get("OPERATIONS_APP_INPUT_FILE", "").strip()
        path = Path(raw) if raw else None
    if path is None:
        return None
    if not path.is_file():
        raise OracleError(f"application request file missing: {path}")
    encoded = path.read_bytes()
    expected_sha = os.environ.get("OPERATIONS_APP_INPUT_SHA256", "").strip()
    if expected_sha:
        canonical = json.dumps(
            json.loads(encoded),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        actual_sha = hashlib.sha256(canonical).hexdigest()
        if actual_sha != expected_sha:
            raise OracleError("application request SHA-256 mismatch")
    try:
        request = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise OracleError(f"invalid application request JSON: {exc}") from exc
    if not isinstance(request, dict):
        raise OracleError("application request must be one JSON object")
    unknown = sorted(set(request) - {"query", "scope", "etfs"})
    if unknown:
        raise OracleError(f"unsupported application request fields: {unknown}")
    return request


def _resolve_scope(request: dict[str, Any] | None) -> dict[str, Any] | None:
    if not request:
        return None
    query = str(request.get("query") or "").strip()
    requested_scope = str(request.get("scope") or "").strip().lower()
    explicit_etfs = request.get("etfs")
    full_market_aliases = (
        "full_market",
        "full market",
        "market structure",
        "전체 시장",
        "전체시장",
        "시장 구조",
    )
    if (
        explicit_etfs is None
        and not requested_scope
        and (not query or any(alias in query.lower() for alias in full_market_aliases))
    ):
        return None
    if explicit_etfs is not None:
        if (
            not isinstance(explicit_etfs, list)
            or not explicit_etfs
            or len(explicit_etfs) > 12
        ):
            raise OracleError("request.etfs must contain 1 to 12 ETF tickers")
        etfs = []
        for value in explicit_etfs:
            ticker = str(value).strip().upper()
            if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", ticker):
                raise OracleError(f"invalid ETF ticker: {value!r}")
            if ticker not in etfs:
                etfs.append(ticker)
        scope_id = requested_scope or "custom_etf_basket"
        registry = SCOPE_REGISTRY.get(scope_id, {})
        return {
            "scope_id": scope_id,
            "label_ko": registry.get("label_ko", scope_id),
            "etfs": etfs,
            "query": query,
            "resolution": "explicit_etfs",
        }
    if requested_scope in {"", "market", "full_market", "전체", "시장전체"}:
        if requested_scope:
            return None
    if requested_scope in SCOPE_REGISTRY:
        registry = SCOPE_REGISTRY[requested_scope]
        return {
            "scope_id": requested_scope,
            "label_ko": registry["label_ko"],
            "etfs": list(registry["etfs"]),
            "query": query,
            "resolution": "canonical_scope",
        }
    haystack = f"{requested_scope} {query}".lower()
    matches: list[tuple[int, str]] = []
    for scope_id, registry in SCOPE_REGISTRY.items():
        for alias in registry["aliases"]:
            if alias.lower() in haystack:
                matches.append((len(alias), scope_id))
    if not matches:
        raise OracleError(
            "scope could not be resolved; provide a canonical scope or etfs"
        )
    scope_id = max(matches)[1]
    registry = SCOPE_REGISTRY[scope_id]
    return {
        "scope_id": scope_id,
        "label_ko": registry["label_ko"],
        "etfs": list(registry["etfs"]),
        "query": query,
        "resolution": "alias",
    }


def _load_pit_scope_memberships(
    connection: sqlite3.Connection,
    *,
    dates: list[str],
    symbols: list[str],
    etfs: list[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray] | None], dict[str, Any]]:
    placeholders = ",".join("?" for _ in etfs)
    rows = connection.execute(
        f"""
        SELECT etf_ticker,constituent_ticker,effective_date,available_date,
               weight_percent,pit_confidence
        FROM etf_constituent_observations
        WHERE etf_ticker IN ({placeholders})
          AND effective_date<=? AND available_date<=?
          AND constituent_ticker IS NOT NULL
        ORDER BY etf_ticker,available_date,effective_date,constituent_ticker
        """,
        tuple(etfs) + (dates[-1], dates[-1]),
    ).fetchall()
    if not rows:
        raise OracleError(
            f"no point-in-time constituent observations for ETFs: {etfs}"
        )
    symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
    snapshots: dict[str, dict[tuple[str, str], list[tuple[int, float]]]] = {
        etf: {} for etf in etfs
    }
    confidences: set[str] = set()
    excluded_symbols: set[str] = set()
    for (
        etf,
        constituent,
        effective_date,
        available_date,
        weight_percent,
        confidence,
    ) in rows:
        ticker = str(constituent).strip().upper()
        symbol_index = symbol_to_index.get(ticker)
        if symbol_index is None:
            excluded_symbols.add(ticker)
            continue
        key = (str(available_date), str(effective_date))
        weight = float(weight_percent or 0.0)
        snapshots.setdefault(str(etf), {}).setdefault(key, []).append(
            (symbol_index, max(weight, 0.0))
        )
        confidences.add(str(confidence))

    timelines: dict[str, list[tuple[str, str, np.ndarray, np.ndarray]]] = {}
    for etf in etfs:
        timeline = []
        for (available_date, effective_date), members in sorted(
            snapshots.get(etf, {}).items()
        ):
            if not members:
                continue
            indices = np.asarray([member[0] for member in members], dtype=int)
            weights = np.asarray([member[1] for member in members], dtype=float)
            positive_total = float(np.sum(weights))
            if positive_total <= 0:
                weights = np.full(len(members), 1.0 / len(members))
            else:
                weights /= positive_total
            timeline.append(
                (available_date, effective_date, indices, weights)
            )
        if not timeline:
            raise OracleError(
                f"no usable point-in-time constituents for ETF: {etf}"
            )
        timelines[etf] = timeline

    pointers = {etf: -1 for etf in etfs}
    memberships: list[tuple[np.ndarray, np.ndarray] | None] = []
    latest_snapshot: dict[str, dict[str, Any]] = {}
    dates_with_membership = 0
    for date in dates:
        combined: dict[int, float] = {}
        active_etfs = 0
        for etf in etfs:
            timeline = timelines[etf]
            pointer = pointers[etf]
            while pointer + 1 < len(timeline):
                candidate = timeline[pointer + 1]
                if candidate[0] > date or candidate[1] > date:
                    break
                pointer += 1
            pointers[etf] = pointer
            if pointer < 0:
                continue
            available_date, effective_date, indices, weights = timeline[pointer]
            active_etfs += 1
            latest_snapshot[etf] = {
                "available_date": available_date,
                "effective_date": effective_date,
                "members_in_universe": int(indices.size),
            }
            for index, weight in zip(indices, weights):
                combined[int(index)] = combined.get(int(index), 0.0) + float(
                    weight
                )
        if not combined or active_etfs == 0:
            memberships.append(None)
            continue
        indices = np.asarray(sorted(combined), dtype=int)
        weights = np.asarray([combined[int(i)] for i in indices], dtype=float)
        weights /= float(np.sum(weights))
        memberships.append((indices, weights))
        dates_with_membership += 1
    if memberships[-1] is None:
        raise OracleError("scope has no visible point-in-time membership as-of")
    current_indices, current_weights = memberships[-1]
    return memberships, {
        "etfs": etfs,
        "snapshot_count": int(
            sum(len(timeline) for timeline in timelines.values())
        ),
        "history_start": next(
            dates[index]
            for index, membership in enumerate(memberships)
            if membership is not None
        ),
        "dates_with_membership": dates_with_membership,
        "current_member_count": int(current_indices.size),
        "current_weight_hhi": float(np.sum(np.square(current_weights))),
        "latest_snapshot_by_etf": latest_snapshot,
        "pit_confidence_values": sorted(confidences),
        "constituents_excluded_outside_universe": len(excluded_symbols),
    }


def _synthetic_scope_index(
    close: np.ndarray,
    symbols: list[str],
    etfs: list[str],
) -> tuple[np.ndarray, list[str]]:
    symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
    usable = [ticker for ticker in etfs if ticker in symbol_to_index]
    if not usable:
        raise OracleError(f"scope ETFs are absent from universe: {etfs}")
    prices = close[:, [symbol_to_index[ticker] for ticker in usable]].astype(
        np.float64
    )
    daily = _horizon_returns(prices, 1).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        equal_weight_daily = np.nanmean(daily, axis=1)
    synthetic = np.full(prices.shape[0], np.nan, dtype=np.float64)
    first = int(
        np.flatnonzero(np.any(np.isfinite(prices) & (prices > 0), axis=1))[0]
    )
    synthetic[first] = 100.0
    for index in range(first + 1, len(synthetic)):
        value = equal_weight_daily[index]
        if np.isfinite(value) and np.isfinite(synthetic[index - 1]):
            synthetic[index] = synthetic[index - 1] * (1.0 + value)
    return synthetic, usable


def _build_scope_features(
    connection: sqlite3.Connection,
    *,
    dates: list[str],
    symbols: list[str],
    close: np.ndarray,
    volume: np.ndarray,
    etfs: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    memberships, membership_coverage = _load_pit_scope_memberships(
        connection,
        dates=dates,
        symbols=symbols,
        etfs=etfs,
    )
    date_index = pd.Index(pd.to_datetime(dates), name="date")
    feature_names = (
        "scope_breadth_up_5d",
        "scope_breadth_up_20d",
        "scope_breadth_up_63d",
        "scope_median_ret_20d",
        "scope_median_ret_63d",
        "scope_dispersion_20d",
        "scope_left_tail_20d",
        "scope_median_drawdown",
        "scope_deep_damage_frac",
        "scope_liquidity_weighted_ret_20d",
        "scope_constituent_weighted_ret_20d",
        "scope_leadership_gap_20d",
        "scope_liquidity_hhi",
        "scope_etf_ret_20d",
        "scope_etf_ret_63d",
        "scope_vs_qqq_20d",
        "scope_vs_qqq_63d",
    )
    values = {
        name: np.full(len(dates), np.nan, dtype=float)
        for name in feature_names
    }
    running_high = np.full(len(symbols), np.nan, dtype=np.float64)
    for date_position, membership in enumerate(memberships):
        current_prices = close[date_position].astype(np.float64)
        finite_prices = np.isfinite(current_prices) & (current_prices > 0)
        running_high[finite_prices] = np.fmax(
            running_high[finite_prices], current_prices[finite_prices]
        )
        if membership is None:
            continue
        indices, membership_weights = membership
        for horizon in (5, 20, 63):
            if date_position < horizon:
                continue
            base = close[date_position - horizon, indices].astype(float)
            future = close[date_position, indices].astype(float)
            valid = (
                np.isfinite(base)
                & np.isfinite(future)
                & (base > 0)
                & (future > 0)
            )
            returns = np.divide(
                future,
                base,
                out=np.full(indices.size, np.nan),
                where=valid,
            ) - 1.0
            finite = np.isfinite(returns)
            if not np.any(finite):
                continue
            values[f"scope_breadth_up_{horizon}d"][date_position] = float(
                np.mean(returns[finite] > 0)
            )
            if horizon in (20, 63):
                values[f"scope_median_ret_{horizon}d"][
                    date_position
                ] = float(np.median(returns[finite]))
            if horizon == 20:
                p10, p90 = np.quantile(returns[finite], [0.1, 0.9])
                values["scope_dispersion_20d"][date_position] = float(
                    (p90 - p10) / 2.563
                )
                values["scope_left_tail_20d"][date_position] = float(p10)
                usable_weights = membership_weights[finite]
                usable_weights /= float(np.sum(usable_weights))
                weighted_return = float(
                    np.sum(returns[finite] * usable_weights)
                )
                values["scope_constituent_weighted_ret_20d"][
                    date_position
                ] = weighted_return
                dollar_volume = (
                    close[date_position, indices].astype(float)
                    * volume[date_position, indices].astype(float)
                )
                liquid = finite & np.isfinite(dollar_volume) & (
                    dollar_volume > 0
                )
                if np.any(liquid):
                    liquid_weights = dollar_volume[liquid]
                    liquid_weights /= float(np.sum(liquid_weights))
                    liquid_return = float(
                        np.sum(returns[liquid] * liquid_weights)
                    )
                    values["scope_liquidity_weighted_ret_20d"][
                        date_position
                    ] = liquid_return
                    values["scope_leadership_gap_20d"][
                        date_position
                    ] = liquid_return - float(np.median(returns[finite]))
                    values["scope_liquidity_hhi"][date_position] = float(
                        np.sum(np.square(liquid_weights))
                    )
        highs = running_high[indices]
        member_prices = close[date_position, indices].astype(float)
        valid_drawdown = (
            np.isfinite(highs)
            & np.isfinite(member_prices)
            & (highs > 0)
        )
        if np.any(valid_drawdown):
            drawdowns = (
                member_prices[valid_drawdown] / highs[valid_drawdown] - 1.0
            )
            values["scope_median_drawdown"][date_position] = float(
                np.median(drawdowns)
            )
            values["scope_deep_damage_frac"][date_position] = float(
                np.mean(drawdowns <= -0.20)
            )

    scope_index, usable_etfs = _synthetic_scope_index(close, symbols, etfs)
    symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
    qqq = close[:, symbol_to_index["QQQ"]].astype(np.float64)
    relative_index = np.divide(
        scope_index,
        qqq,
        out=np.full(len(dates), np.nan),
        where=np.isfinite(scope_index) & np.isfinite(qqq) & (qqq > 0),
    )
    for horizon in (20, 63):
        values[f"scope_etf_ret_{horizon}d"] = _horizon_returns(
            scope_index[:, None], horizon
        )[:, 0]
        values[f"scope_vs_qqq_{horizon}d"] = _horizon_returns(
            relative_index[:, None], horizon
        )[:, 0]

    features = pd.DataFrame(values, index=date_index)
    flow_frame, flow_coverage = _flow_features(
        connection,
        dates,
        date_index,
        tickers=usable_etfs,
        required=False,
    )
    for column in flow_frame:
        features[f"scope_{column}"] = flow_frame[column]
    state = features.replace([np.inf, -np.inf], np.nan)
    state_z = _robust_causal_z(state)
    return (
        state,
        state_z,
        scope_index,
        relative_index,
        {
            "membership": membership_coverage,
            "flow": flow_coverage,
            "usable_etfs": usable_etfs,
        },
    )


def _classify_scope(
    raw: dict[str, float],
    zscore: dict[str, float],
) -> dict[str, Any]:
    components = {
        "internal_breadth": float(
            np.mean(
                [
                    zscore.get("scope_breadth_up_20d", 0.0),
                    zscore.get("scope_breadth_up_63d", 0.0),
                ]
            )
        ),
        "absolute_momentum": float(
            np.mean(
                [
                    zscore.get("scope_etf_ret_20d", 0.0),
                    zscore.get("scope_etf_ret_63d", 0.0),
                ]
            )
        ),
        "relative_to_qqq": float(
            np.mean(
                [
                    zscore.get("scope_vs_qqq_20d", 0.0),
                    zscore.get("scope_vs_qqq_63d", 0.0),
                ]
            )
        ),
        "d2_flow": float(
            np.mean(
                [
                    zscore.get("scope_flow_net_5d_usd", 0.0),
                    zscore.get("scope_flow_net_20d_usd", 0.0),
                ]
            )
        ),
        "damage_inverse": -zscore.get("scope_deep_damage_frac", 0.0),
        "concentration_inverse": -zscore.get(
            "scope_leadership_gap_20d", 0.0
        ),
    }
    scope_score = float(np.mean(list(components.values())))
    breadth = components["internal_breadth"]
    relative = components["relative_to_qqq"]
    if scope_score >= 0.6 and breadth >= 0.25 and relative >= 0:
        regime = "scope_leading_broad"
    elif scope_score >= 0.25 and relative >= 0:
        regime = "scope_leading_narrow"
    elif scope_score <= -0.6:
        regime = "scope_structural_risk_off"
    elif scope_score <= -0.25 or relative <= -0.5:
        regime = "scope_lagging"
    else:
        regime = "scope_mixed_transition"
    return {
        "regime": regime,
        "structural_score": scope_score,
        "risk_score_components": components,
        "raw_features": raw,
        "causal_z_features": zscore,
    }


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
    minimum_features = max(8, int(0.75 * matrix.shape[1]))
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
    if actual_array.size:
        brier = float(np.mean(np.square(probability_array - actual_up)))
        baseline_brier = float(
            np.mean(np.square(baseline_probability_array - actual_up))
        )
        directional_hit_rate = float(
            np.mean((prediction_array > 0) == (actual_array > 0))
        )
        always_up_hit_rate = float(np.mean(actual_array > 0))
        mean_absolute_error = float(
            np.mean(np.abs(prediction_array - actual_array))
        )
        correlation = (
            float(np.corrcoef(prediction_array, actual_array)[0, 1])
            if actual_array.size >= 3
            else math.nan
        )
        brier_skill = (
            float(1.0 - brier / baseline_brier)
            if baseline_brier > 0
            else math.nan
        )
    else:
        brier = math.nan
        baseline_brier = math.nan
        directional_hit_rate = math.nan
        always_up_hit_rate = math.nan
        mean_absolute_error = math.nan
        correlation = math.nan
        brier_skill = math.nan
    validation = {
        "horizon_sessions": 20,
        "forecast_step_sessions": 5,
        "sample_size": int(actual_array.size),
        "start_date": validation_dates[0] if validation_dates else None,
        "end_date": validation_dates[-1] if validation_dates else None,
        "directional_hit_rate": directional_hit_rate,
        "always_up_hit_rate": always_up_hit_rate,
        "mean_absolute_error": mean_absolute_error,
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "brier_skill_vs_expanding_base": brier_skill,
        "prediction_actual_correlation": correlation,
        "forecast_confidence": (
            "validated"
            if np.isfinite(brier)
            and np.isfinite(baseline_brier)
            and brier < baseline_brier
            else "scenario_only_no_incremental_probability_skill"
        ),
    }
    return analogs, forecast, validation


def _render_html(payload: dict[str, Any]) -> str:
    current = payload["current_structure"]
    forecast = payload["forecast"]
    validation = payload["walk_forward_validation"]
    scope = payload.get("scope_analysis")
    incremental = payload["incremental_market_data"]
    source_rows = "".join(
        (
            "<tr><td>Massive grouped daily</td><td>Massive</td>"
            f"<td>{html.escape(str(session))}</td><td>Oracle single writer</td>"
            f"<td>{int(rows):,}</td></tr>"
        )
        for session, rows in (
            incremental["market_row_gate"]["rows_by_session"].items()
        )
    )
    source_rows += (
        "<tr><td>ETF fund flow</td><td>Massive ETF Global</td>"
        f"<td>{html.escape(str(incremental['etf_flow']['latest_effective_date']))}</td>"
        "<td>D+2 PIT gate</td>"
        f"<td>{int(incremental['etf_flow']['record_count']):,}</td></tr>"
    )
    incremental_section = f"""<section class=\"card\"><h2>현재 시장 증분 DB — 완결</h2>
<p>기준 원본 FMP 장기 이력 종료 <b>{incremental['base_history_end']}</b> →
현재 기준일 <b>{incremental['target_as_of_date']}</b>. 누락 거래일 없이
{len(incremental['expected_sessions'])}개 NYSE 세션을 Massive 전시장 일봉으로 누적했습니다.</p>
<p class=\"muted\">가격 전환: {html.escape(incremental['base_price_source'])} +
{html.escape(incremental['incremental_price_source'])}. ETF Flow는 D+2 기준
effective <b>{incremental['etf_flow']['latest_effective_date']}</b>까지 반영됩니다.</p>
<table><thead><tr><th>보관 테이블</th><th>출처</th><th>기준일</th><th>수집시각 UTC</th><th>행 수</th></tr></thead><tbody>{source_rows}</tbody></table></section>"""
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
    scope_section = ""
    if scope:
        scope_current = scope["current_structure"]
        scope_absolute_rows = "".join(
            "<tr>"
            f"<td>{horizon} sessions</td>"
            f"<td>{_pct(item['return_quantiles']['p10'])}</td>"
            f"<td>{_pct(item['return_quantiles']['p50'])}</td>"
            f"<td>{_pct(item['return_quantiles']['p90'])}</td>"
            f"<td>{item['probability_positive'] * 100:.1f}%</td>"
            f"<td>{_pct(item['relative_to_unconditional']['median_return_delta'])}</td>"
            "</tr>"
            for horizon, item in scope["forecast_absolute"].items()
        )
        scope_relative_rows = "".join(
            "<tr>"
            f"<td>{horizon} sessions</td>"
            f"<td>{_pct(item['return_quantiles']['p10'])}</td>"
            f"<td>{_pct(item['return_quantiles']['p50'])}</td>"
            f"<td>{_pct(item['return_quantiles']['p90'])}</td>"
            f"<td>{item['probability_positive'] * 100:.1f}%</td>"
            f"<td>{_pct(item['relative_to_unconditional']['median_return_delta'])}</td>"
            "</tr>"
            for horizon, item in scope["forecast_relative_to_qqq"].items()
        )
        scope_section = f"""
<section class="card"><h2>조건부 범위: {html.escape(scope['label_ko'])}</h2>
<p>전체 시장 레짐 <b>{current['regime']}</b> 안에서
<b>{scope_current['regime']}</b> · 범위 구조 점수
<b>{scope_current['structural_score']:+.2f}z</b></p>
<p class="muted">PIT 바스켓 {', '.join(scope['etfs'])} · 현재 구성종목
{scope['coverage']['membership']['current_member_count']:,} · D+2 Flow</p>
<h3>범위 절대 경로</h3><table><thead><tr>
<th>기간</th><th>P10</th><th>중앙값</th><th>P90</th><th>상승</th><th>평상시 대비 중앙</th>
</tr></thead><tbody>{scope_absolute_rows}</tbody></table>
<h3>QQQ 대비 상대 경로</h3><table><thead><tr>
<th>기간</th><th>P10</th><th>중앙값</th><th>P90</th><th>초과성과</th><th>평상시 대비 중앙</th>
</tr></thead><tbody>{scope_relative_rows}</tbody></table>
<p class="warn">절대 경로: {scope['walk_forward_validation_absolute']['forecast_confidence']}
· 상대 경로: {scope['walk_forward_validation_relative']['forecast_confidence']}</p>
</section>"""
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
<p class="muted">Hermes Worker App · frozen PIT history + audited current ETF overlay</p>
<h1>Market Structure Oracle</h1><p class="muted">장기 데이터 기준일 {payload['as_of_date']} · 실행 UTC {payload['created_at_utc']}</p>
<section class="grid">
<div class="card"><div class="muted">구조 레짐</div><div class="metric">{current['regime']}</div></div>
<div class="card"><div class="muted">구조 점수</div><div class="metric">{current['structural_score']:+.2f}z</div></div>
<div class="card"><div class="muted">관측 유니버스</div><div class="metric">{payload['coverage']['symbols_with_observations']:,}</div></div>
<div class="card"><div class="muted">Flow D+2 최신 effective</div><div class="metric">{payload['coverage']['latest_effective_date_visible']}</div></div>
</section>
{incremental_section}
{scope_section}
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
<section class="card"><h2>출처·기간·대상 계약</h2><p>장기 PIT 상태 큐브: FMP 기준 이력 + Oracle 증분 Massive 전시장 일봉 {payload['coverage']['calendar_start']} → {payload['coverage']['calendar_end']} · {payload['coverage']['sessions']:,} 세션 · {payload['coverage']['symbols_with_observations']:,} 심볼. ETF Flow는 {payload['coverage']['latest_effective_date_visible']} effective까지 D+2 정책으로 반영합니다.</p><p>Oracle이 Massive 일봉·ETF Flow와 FMP ETF 구성종목을 단일 writer로 갱신하고 봉인합니다. ETF RADAR는 별도 앱이며 이 리포트의 데이터·릴리스 게이트가 아닙니다.</p></section>
</main></body></html>"""


def run_oracle(
    *,
    database_path: Path,
    universe_path: Path,
    output_root: Path,
    as_of: str | None,
    incremental_root: Path = DEFAULT_INCREMENTAL_ROOT,
    request_file: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        incremental = ensure_oracle_snapshot(
            base_database=database_path,
            incremental_root=incremental_root,
            target_as_of_date=as_of,
        )
    except IncrementalStoreError as exc:
        raise OracleError(f"incremental market-data gate failed: {exc}") from exc
    target_as_of = str(incremental["target_as_of_date"])
    if as_of is not None and as_of != target_as_of:
        raise OracleError(
            f"as-of must equal completed incremental target {target_as_of}, got {as_of}"
        )
    as_of = target_as_of
    incremental_database_path = Path(str(incremental["database"]))
    before = database_path.stat()
    incremental_before = incremental_database_path.stat()
    request = _load_request(request_file)
    scope_definition = _resolve_scope(request)
    symbols = sorted(
        {
            line.strip().upper()
            for line in universe_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    connection = _connect_oracle_view(database_path, incremental_database_path)
    try:
        (
            dates,
            symbols,
            close,
            volume,
            state,
            state_z,
            cube_metadata,
            cube_cache_hit,
        ) = _load_or_build_state_cube(
            connection,
            database_path=database_path,
            incremental_database_path=incremental_database_path,
            universe_path=universe_path,
            output_root=output_root,
            as_of=as_of,
            symbols=symbols,
        )
        coverage = dict(cube_metadata["coverage"])
        scanned = int(cube_metadata["daily_rows_scanned"])
        symbol_to_index = {
            symbol: index for index, symbol in enumerate(symbols)
        }
        benchmark = close[:, symbol_to_index["QQQ"]].astype(np.float64)
        scope_data = None
        if scope_definition is not None:
            (
                scope_state,
                scope_state_z,
                scope_benchmark,
                relative_benchmark,
                scope_coverage,
            ) = _build_scope_features(
                connection,
                dates=dates,
                symbols=symbols,
                close=close,
                volume=volume,
                etfs=scope_definition["etfs"],
            )
            scope_data = {
                "state": scope_state,
                "state_z": scope_state_z,
                "benchmark": scope_benchmark,
                "relative_benchmark": relative_benchmark,
                "coverage": scope_coverage,
            }
        invalid_count = connection.execute(
            "SELECT COUNT(*) FROM oracle_quality_checks WHERE status='invalid'"
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
    scope_analysis = None
    if scope_definition is not None and scope_data is not None:
        scope_state = scope_data["state"]
        scope_state_z = scope_data["state_z"]
        scope_raw = {
            name: float(scope_state.iloc[-1][name])
            for name in scope_state.columns
            if np.isfinite(scope_state.iloc[-1][name])
        }
        scope_z = {
            name: float(scope_state_z.iloc[-1][name])
            for name in scope_state_z.columns
            if np.isfinite(scope_state_z.iloc[-1][name])
        }
        scope_current = _classify_scope(scope_raw, scope_z)
        combined_state = pd.concat(
            [
                state_z.add_prefix("global__"),
                scope_state_z.add_prefix("scope__"),
            ],
            axis=1,
        )
        (
            scope_analogs_absolute,
            scope_forecast_absolute,
            scope_validation_absolute,
        ) = _forecast_and_validate(
            dates,
            combined_state,
            scope_data["benchmark"],
        )
        (
            scope_analogs_relative,
            scope_forecast_relative,
            scope_validation_relative,
        ) = _forecast_and_validate(
            dates,
            combined_state,
            scope_data["relative_benchmark"],
        )
        scope_analysis = {
            **scope_definition,
            "formula": (
                "scope future | full market state + scope internal structure "
                "+ relative position + ETF Flow D+2"
            ),
            "global_context_preserved": True,
            "current_structure": scope_current,
            "coverage": scope_data["coverage"],
            "combined_state_feature_count": int(combined_state.shape[1]),
            "analogs_absolute": scope_analogs_absolute,
            "analogs_relative_to_qqq": scope_analogs_relative,
            "forecast_absolute": scope_forecast_absolute,
            "forecast_relative_to_qqq": scope_forecast_relative,
            "walk_forward_validation_absolute": scope_validation_absolute,
            "walk_forward_validation_relative": scope_validation_relative,
        }
    run_id = os.environ.get("OPERATIONS_APP_RUN_ID") or time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime()
    )
    if not re.fullmatch(r"[A-Za-z0-9._-]{8,128}", run_id):
        raise OracleError(f"invalid operations run id: {run_id!r}")
    after = database_path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise OracleError("source database changed during read-only analysis")
    incremental_after = incremental_database_path.stat()
    if (
        incremental_before.st_size != incremental_after.st_size
        or incremental_before.st_mtime_ns != incremental_after.st_mtime_ns
    ):
        raise OracleError("incremental database changed during read-only analysis")

    payload: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "app_id": os.environ.get(
            "OPERATIONS_APP_ID", "market-structure-oracle"
        ),
        "operations_app_run_id": run_id,
        "operations_app_trigger": os.environ.get("OPERATIONS_APP_TRIGGER"),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "as_of_date": dates[-1],
        "query_mode": (
            "conditional_scope" if scope_analysis else "full_market"
        ),
        "request": request,
        "source_contract": {
            "database": str(database_path),
            "incremental_database": str(incremental_database_path),
            "universe": str(universe_path),
            "price_source": "FMP immutable baseline + Massive full-market incremental sessions",
            "flow_policy_id": ETF_FLOW_POLICY_ID,
            "incremental_repair_required_before_report": True,
            "source_owner": "market_structure_oracle_single_writer",
            "etf_radar_runtime_dependency": False,
            "lookahead_in_features": False,
            "future_returns_role": "labels_only",
            "database_open_mode": "sqlite_uri_mode_ro + temp_union_views",
        },
        "coverage": {
            "calendar_start": dates[0],
            "calendar_end": dates[-1],
            "sessions": len(dates),
            "daily_rows_scanned": scanned,
            "quality_invalid_rows_excluded": int(invalid_count),
            **coverage,
        },
        "state_cube": {
            "schema": STATE_CUBE_SCHEMA,
            "directory": str(
                _state_cube_directory(output_root, dates[-1])
            ),
            "cache_hit": cube_cache_hit,
            "full_market_computed_once_reused_for_scope": True,
            "matrix_shape": cube_metadata["matrix_shape"],
            "source_fingerprint": cube_metadata["source_fingerprint"],
        },
        "incremental_market_data": incremental,
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
        "scope_analysis": scope_analysis,
        "source_database_integrity": {
            "before_size": before.st_size,
            "after_size": after.st_size,
            "before_mtime_ns": before.st_mtime_ns,
            "after_mtime_ns": after.st_mtime_ns,
            "unchanged": True,
            "incremental_before_size": incremental_before.st_size,
            "incremental_after_size": incremental_after.st_size,
            "incremental_before_mtime_ns": incremental_before.st_mtime_ns,
            "incremental_after_mtime_ns": incremental_after.st_mtime_ns,
            "incremental_unchanged_during_analysis": True,
        },
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    run_directory = output_root / dates[-1] / "runs" / run_id
    if scope_analysis:
        scope_id = scope_analysis["scope_id"]
        json_path = run_directory / f"scope-{scope_id}.json"
        html_path = run_directory / f"scope-{scope_id}.html"
        latest_json = (
            output_root / dates[-1] / "latest-scopes" / f"{scope_id}.json"
        )
        latest_html = (
            output_root / dates[-1] / "latest-scopes" / f"{scope_id}.html"
        )
    else:
        json_path = run_directory / "market_structure_oracle.json"
        html_path = run_directory / "market_structure_oracle.html"
        latest_json = output_root / dates[-1] / "latest.json"
        latest_html = output_root / dates[-1] / "latest.html"
    html_document = _render_html(payload)
    payload = _json_safe(payload)
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
        "query_mode": payload["query_mode"],
        "global_regime": current["regime"],
        "global_structural_score": current["structural_score"],
        "scope_id": (
            scope_analysis["scope_id"] if scope_analysis else None
        ),
        "scope_regime": (
            scope_analysis["current_structure"]["regime"]
            if scope_analysis
            else None
        ),
        "forecast_confidence": (
            scope_analysis["walk_forward_validation_absolute"][
                "forecast_confidence"
            ]
            if scope_analysis
            else validation["forecast_confidence"]
        ),
        "state_cube_cache_hit": cube_cache_hit,
        "source_database_unchanged": True,
        "incremental_market_data_status": incremental["status"],
        "outputs": payload["outputs"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Hermes-managed market-structure Oracle."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--incremental-root", type=Path, default=DEFAULT_INCREMENTAL_ROOT)
    parser.add_argument("--as-of")
    parser.add_argument("--request-file", type=Path)
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
                incremental_root=args.incremental_root,
                request_file=args.request_file,
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
