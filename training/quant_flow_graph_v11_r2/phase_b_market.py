"""Interpretable Phase B market Drift-Diffusion walk-forward tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    ANCHOR_TICKERS,
    DEFAULT_SOURCE_DATABASE,
    TIMING_CONTRACT,
)
from .hypotheses import specification
from .phase_a import json_sha256, readonly_connection, sha256_file, utc_now, write_json_atomic


PHASE_B_MARKET_SCHEMA_VERSION = "quant.etf_flow_v11_r2.phase_b_market.v1"
DEFAULT_PHASE_A_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/r2_phase_a"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/r2_phase_b_market"
)
TARGET_ANCHORS = ("SPY", "QQQ", "RSP")
HORIZONS = (5, 20)
OUTER_YEARS = tuple(range(2021, 2027))
PURGE_SESSIONS = 20
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def signed_log_dollars(value: object) -> float:
    number = _finite(value)
    if not math.isfinite(number):
        return math.nan
    return math.copysign(math.log1p(abs(number) / 1_000_000.0), number)


@dataclass(frozen=True)
class PriceSeries:
    dates: tuple[str, ...]
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    index: Mapping[str, int]


def load_price_series(
    source: sqlite3.Connection, symbol: str
) -> PriceSeries:
    rows = list(
        source.execute(
            """
            SELECT trade_date,COALESCE(adjusted_close,close),high,low
            FROM daily_observations
            WHERE source='fmp' AND symbol=?
            ORDER BY trade_date
            """,
            (symbol,),
        )
    )
    dates = tuple(str(row[0]) for row in rows)
    close = np.asarray([_finite(row[1]) for row in rows], dtype=np.float64)
    high = np.asarray([_finite(row[2]) for row in rows], dtype=np.float64)
    low = np.asarray([_finite(row[3]) for row in rows], dtype=np.float64)
    return PriceSeries(
        dates=dates,
        close=close,
        high=high,
        low=low,
        index={date: index for index, date in enumerate(dates)},
    )


def past_return(series: PriceSeries, position: int, lookback: int) -> float:
    if position < lookback:
        return math.nan
    start = series.close[position - lookback]
    end = series.close[position]
    return (end / start - 1.0) * 100.0 if start > 0 and end > 0 else math.nan


def realized_volatility(series: PriceSeries, position: int, lookback: int) -> float:
    if position < lookback:
        return math.nan
    prices = series.close[position - lookback : position + 1]
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        return math.nan
    returns = np.diff(np.log(prices))
    return float(np.std(returns, ddof=1) * math.sqrt(252.0) * 100.0)


def trailing_drawdown(series: PriceSeries, position: int, lookback: int) -> float:
    if position < lookback:
        return math.nan
    window = series.close[position - lookback : position + 1]
    current = series.close[position]
    peak = np.nanmax(window)
    return (current / peak - 1.0) * 100.0 if peak > 0 and current > 0 else math.nan


def future_targets(
    series: PriceSeries, price_date: str, horizon: int
) -> tuple[float, float]:
    position = series.index.get(price_date)
    if position is None or position + horizon >= len(series.dates):
        return math.nan, math.nan
    reference = series.close[position]
    end = series.close[position + horizon]
    future_lows = series.low[position + 1 : position + horizon + 1]
    if reference <= 0 or end <= 0 or not np.isfinite(future_lows).any():
        return math.nan, math.nan
    future_return = (end / reference - 1.0) * 100.0
    loss = max(0.0, (reference - float(np.nanmin(future_lows))) / reference * 100.0)
    return future_return, loss


def rolling_mean(values: np.ndarray, lookback: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - lookback + 1)
        window = values[start : index + 1]
        finite_values = window[np.isfinite(window)]
        if len(finite_values):
            result[index] = float(np.mean(finite_values))
    return result


def rolling_z(values: np.ndarray, lookback: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - lookback + 1)
        window = values[start : index + 1]
        finite_values = window[np.isfinite(window)]
        if len(finite_values) >= max(10, lookback // 3):
            std = float(np.std(finite_values))
            if std > 1e-12:
                result[index] = (values[index] - float(np.mean(finite_values))) / std
    return result


def lag_matrix(matrix: np.ndarray, sessions: int) -> np.ndarray:
    result = np.full_like(matrix, np.nan, dtype=np.float64)
    if sessions < len(matrix):
        result[sessions:] = matrix[:-sessions]
    return result


def future_matrix(matrix: np.ndarray, sessions: int) -> np.ndarray:
    result = np.full_like(matrix, np.nan, dtype=np.float64)
    if sessions < len(matrix):
        result[:-sessions] = matrix[sessions:]
    return result


def block_shuffle(matrix: np.ndarray, seed: int, block_size: int = 20) -> np.ndarray:
    if not len(matrix):
        return matrix.copy()
    blocks = [matrix[index : index + block_size] for index in range(0, len(matrix), block_size)]
    order = np.arange(len(blocks))
    np.random.default_rng(seed).shuffle(order)
    return np.concatenate([blocks[index] for index in order], axis=0)[: len(matrix)]


def raw_breadth_by_date(
    event: sqlite3.Connection,
) -> dict[str, dict[str, float]]:
    rows = event.execute(
        """
        SELECT signal_date,
          SUM(CASE WHEN clean_eligible=1 AND observed_exact_t2=1 AND fund_flow>0 THEN 1 ELSE 0 END),
          SUM(CASE WHEN clean_eligible=1 AND observed_exact_t2=1 AND fund_flow<0 THEN 1 ELSE 0 END),
          SUM(CASE WHEN clean_eligible=1 AND observed_exact_t2=1 AND fund_flow=0 THEN 1 ELSE 0 END),
          SUM(CASE WHEN clean_eligible=1 AND observed_exact_t2=1 THEN 1 ELSE 0 END)
        FROM etf_flow_events GROUP BY signal_date ORDER BY signal_date
        """
    )
    result = {}
    for date, positive, negative, zero, observed in rows:
        observed = int(observed or 0)
        result[str(date)] = {
            "raw_positive_etf_count": float(positive or 0),
            "raw_negative_etf_count": float(negative or 0),
            "raw_zero_etf_count": float(zero or 0),
            "raw_observed_etf_count": float(observed),
            "raw_etf_breadth_net": (
                (float(positive or 0) - float(negative or 0)) / observed
                if observed
                else math.nan
            ),
        }
    return result


def anchor_flow_by_date(
    event: sqlite3.Connection,
) -> dict[str, dict[str, float]]:
    placeholders = ",".join("?" for _ in ANCHOR_TICKERS)
    rows = event.execute(
        f"""
        SELECT signal_date,ticker,flow_rate_pct,observed_exact_t2
        FROM etf_flow_events
        WHERE ticker IN ({placeholders})
        ORDER BY signal_date,ticker
        """,
        ANCHOR_TICKERS,
    )
    result: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for date, ticker, rate, observed in rows:
        result[str(date)][f"anchor_{str(ticker).lower()}_flow_rate"] = (
            _finite(rate) if int(observed) else math.nan
        )
    return dict(result)


def build_market_matrix(
    *, event: sqlite3.Connection, source: sqlite3.Connection
) -> dict[str, Any]:
    daily_rows = [dict(row) for row in event.execute("SELECT * FROM daily_flow_state ORDER BY signal_date")]
    dates = tuple(str(row["signal_date"]) for row in daily_rows)
    price_dates = tuple(str(row["price_date"]) for row in daily_rows)
    raw_breadth = raw_breadth_by_date(event)
    anchor_flow = anchor_flow_by_date(event)
    price_series = {symbol: load_price_series(source, symbol) for symbol in TARGET_ANCHORS}

    price_names: list[str] = []
    price_columns: list[list[float]] = []
    for symbol in TARGET_ANCHORS:
        series = price_series[symbol]
        for suffix, function in (
            ("ret_1d", lambda p, s=series: past_return(s, p, 1)),
            ("ret_5d", lambda p, s=series: past_return(s, p, 5)),
            ("ret_20d", lambda p, s=series: past_return(s, p, 20)),
            ("vol_20d", lambda p, s=series: realized_volatility(s, p, 20)),
            ("drawdown_20d", lambda p, s=series: trailing_drawdown(s, p, 20)),
        ):
            price_names.append(f"{symbol.lower()}_{suffix}")
            price_columns.append(
                [
                    function(series.index[date]) if date in series.index else math.nan
                    for date in price_dates
                ]
            )
    price_matrix = np.asarray(price_columns, dtype=np.float64).T
    qqq_ret5 = price_matrix[:, price_names.index("qqq_ret_5d")]
    spy_ret5 = price_matrix[:, price_names.index("spy_ret_5d")]
    rsp_ret5 = price_matrix[:, price_names.index("rsp_ret_5d")]
    price_matrix = np.column_stack(
        [price_matrix, qqq_ret5 - spy_ret5, rsp_ret5 - spy_ret5]
    )
    price_names.extend(("qqq_minus_spy_ret_5d", "rsp_minus_spy_ret_5d"))

    base_flow_names = [
        "raw_signed_flow_log",
        "eligible_signed_flow_log",
        "eligible_absolute_flow_log",
        "clean_signed_flow_log",
        "special_effective_signed_flow_log",
        "drift_signed_flow_log",
        "drift_rate_pct",
        "independent_breadth_net",
        "diffusion_coverage",
        "observed_ratio",
        "missing_ratio",
        "stale_ratio",
        "true_zero_observed_ratio",
        "positive_family_share",
        "negative_family_share",
        "raw_etf_breadth_net",
        "anchor_spy_flow_rate",
        "anchor_qqq_flow_rate",
        "anchor_vti_flow_rate",
        "anchor_rsp_flow_rate",
        "anchor_iwm_flow_rate",
        "anchor_dia_flow_rate",
    ]
    base_flow = np.full((len(daily_rows), len(base_flow_names)), np.nan, dtype=np.float64)
    for index, row in enumerate(daily_rows):
        strict = float(row["strict_eligible_etf_count"] or 0)
        observed = float(row["observed_eligible_count"] or 0)
        families = float(row["observed_independent_family_count"] or 0)
        date = dates[index]
        values = {
            "raw_signed_flow_log": signed_log_dollars(row["raw_signed_flow_usd"]),
            "eligible_signed_flow_log": signed_log_dollars(row["eligible_signed_flow_usd"]),
            "eligible_absolute_flow_log": math.log1p(float(row["eligible_absolute_flow_usd"] or 0) / 1_000_000.0),
            "clean_signed_flow_log": signed_log_dollars(row["clean_signed_flow_usd"]),
            "special_effective_signed_flow_log": signed_log_dollars(row["special_effective_signed_flow_usd"]),
            "drift_signed_flow_log": signed_log_dollars(row["drift_signed_flow_usd"]),
            "drift_rate_pct": _finite(row["drift_rate_pct"]),
            "independent_breadth_net": _finite(row["independent_breadth_net"]),
            "diffusion_coverage": _finite(row["diffusion_coverage"]),
            "observed_ratio": observed / strict if strict else math.nan,
            "missing_ratio": float(row["missing_eligible_count"] or 0) / strict if strict else math.nan,
            "stale_ratio": float(row["stale_eligible_count"] or 0) / strict if strict else math.nan,
            "true_zero_observed_ratio": float(row["true_zero_eligible_count"] or 0) / observed if observed else math.nan,
            "positive_family_share": float(row["positive_independent_family_count"] or 0) / families if families else math.nan,
            "negative_family_share": float(row["negative_independent_family_count"] or 0) / families if families else math.nan,
            "raw_etf_breadth_net": raw_breadth.get(date, {}).get("raw_etf_breadth_net", math.nan),
        }
        values.update(anchor_flow.get(date, {}))
        base_flow[index] = [values.get(name, math.nan) for name in base_flow_names]

    derived_names: list[str] = []
    derived_columns: list[np.ndarray] = []
    for name in (
        "drift_rate_pct",
        "independent_breadth_net",
        "diffusion_coverage",
    ):
        values = base_flow[:, base_flow_names.index(name)]
        for window in (5, 20):
            derived_names.append(f"{name}_mean_{window}")
            derived_columns.append(rolling_mean(values, window))
        derived_names.append(f"{name}_z60")
        derived_columns.append(rolling_z(values, 60))
        derived_names.append(f"{name}_change_5")
        change = np.full(len(values), np.nan)
        change[5:] = values[5:] - values[:-5]
        derived_columns.append(change)
    flow_names = base_flow_names + derived_names
    flow_matrix = np.column_stack([base_flow, *derived_columns])

    target_names = []
    target_columns = []
    for symbol in TARGET_ANCHORS:
        series = price_series[symbol]
        for horizon in HORIZONS:
            returns = []
            losses = []
            for price_date in price_dates:
                future_return, loss = future_targets(series, price_date, horizon)
                returns.append(future_return)
                losses.append(loss)
            target_names.extend(
                (f"{symbol}_return_{horizon}d_pct", f"{symbol}_loss_{horizon}d_pct")
            )
            target_columns.extend((returns, losses))
    targets = np.asarray(target_columns, dtype=np.float64).T
    return {
        "dates": dates,
        "price_dates": price_dates,
        "price_names": tuple(price_names),
        "price_matrix": price_matrix,
        "flow_names": tuple(flow_names),
        "flow_matrix": flow_matrix,
        "target_names": tuple(target_names),
        "targets": targets,
    }


def impute_and_scale_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(matrix, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    filled = np.where(np.isfinite(matrix), matrix, medians)
    means = np.mean(filled, axis=0)
    scales = np.std(filled, axis=0)
    scales[scales < 1e-9] = 1.0
    return (filled - means) / scales, medians, np.vstack([means, scales])


def impute_and_scale_apply(
    matrix: np.ndarray, medians: np.ndarray, moments: np.ndarray
) -> np.ndarray:
    filled = np.where(np.isfinite(matrix), matrix, medians)
    return (filled - moments[0]) / moments[1]


def ridge_fit(matrix: np.ndarray, target: np.ndarray, alpha: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    scaled, medians, moments = impute_and_scale_fit(matrix)
    target_mean = float(np.mean(target))
    centered = target - target_mean
    penalty = np.eye(scaled.shape[1], dtype=np.float64) * alpha
    beta = np.linalg.solve(scaled.T @ scaled + penalty, scaled.T @ centered)
    return beta, target_mean, medians, moments


def ridge_predict(
    matrix: np.ndarray,
    model: tuple[np.ndarray, float, np.ndarray, np.ndarray],
) -> np.ndarray:
    beta, target_mean, medians, moments = model
    scaled = impute_and_scale_apply(matrix, medians, moments)
    return scaled @ beta + target_mean


def metric(target: np.ndarray, prediction: np.ndarray, direction: bool) -> dict[str, float]:
    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    if len(target) >= 2 and np.std(target) > 1e-12 and np.std(prediction) > 1e-12:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    else:
        correlation = math.nan
    result = {"mae": mae, "rmse": rmse, "correlation": correlation}
    if direction:
        result["direction_accuracy"] = float(np.mean((target >= 0) == (prediction >= 0)))
    return result


def tune_alpha(
    matrix: np.ndarray,
    target: np.ndarray,
    dates: Sequence[str],
    train_indices: np.ndarray,
) -> float:
    years = sorted({int(dates[index][:4]) for index in train_indices})
    if len(years) < 3:
        return 1.0
    validation_year = years[-1]
    validation = train_indices[
        np.asarray([int(dates[index][:4]) == validation_year for index in train_indices])
    ]
    if not len(validation):
        return 1.0
    first_validation = int(validation[0])
    inner_train = train_indices[train_indices < first_validation - PURGE_SESSIONS]
    if len(inner_train) < 100:
        return 1.0
    valid_target = np.isfinite(target)
    inner_train = inner_train[valid_target[inner_train]]
    validation = validation[valid_target[validation]]
    if len(inner_train) < 100 or len(validation) < 20:
        return 1.0
    scores = []
    for alpha in ALPHA_GRID:
        model = ridge_fit(matrix[inner_train], target[inner_train], alpha)
        prediction = ridge_predict(matrix[validation], model)
        scores.append((float(np.mean(np.abs(prediction - target[validation]))), alpha))
    return min(scores)[1]


def model_feature_sets(
    price_names: Sequence[str], flow_names: Sequence[str]
) -> dict[str, tuple[int, ...]]:
    price_count = len(price_names)
    flow_index = {name: price_count + index for index, name in enumerate(flow_names)}
    price_indices = tuple(range(price_count))
    mask_fields = (
        "observed_ratio",
        "missing_ratio",
        "stale_ratio",
        "true_zero_observed_ratio",
        "diffusion_coverage",
    )
    raw_fields = ("raw_signed_flow_log", "raw_etf_breadth_net")
    drift_fields = tuple(
        name
        for name in flow_names
        if name.startswith("drift_")
        or name.startswith("eligible_signed")
        or name.startswith("clean_signed")
        or name.startswith("special_effective")
        or name.startswith("anchor_")
    )
    diffusion_fields = tuple(
        name
        for name in flow_names
        if "breadth" in name
        or "diffusion" in name
        or name in ("positive_family_share", "negative_family_share")
    )
    def indices(fields: Sequence[str]) -> tuple[int, ...]:
        return price_indices + tuple(flow_index[name] for name in fields)

    return {
        "price_only": price_indices,
        "mask_only": indices(mask_fields),
        "raw_flow": indices(raw_fields),
        "drift_only": indices(drift_fields),
        "diffusion_only": indices(diffusion_fields),
        "drift_plus_diffusion": indices(tuple(dict.fromkeys(drift_fields + diffusion_fields + mask_fields))),
        "special_channel_off": indices(
            tuple(name for name in drift_fields + diffusion_fields + mask_fields if not name.startswith("special_effective"))
        ),
        "duplicate_adjustment_off": indices(
            tuple(
                "raw_etf_breadth_net" if name == "independent_breadth_net" else name
                for name in dict.fromkeys(drift_fields + diffusion_fields + mask_fields)
                if not name.startswith("independent_breadth_net_")
            )
        ),
    }


def evaluate_walk_forward(matrix: Mapping[str, Any]) -> dict[str, Any]:
    dates = matrix["dates"]
    price = matrix["price_matrix"]
    flow = matrix["flow_matrix"]
    combined = np.column_stack([price, flow])
    groups = model_feature_sets(matrix["price_names"], matrix["flow_names"])
    price_count = price.shape[1]
    flow_count = flow.shape[1]
    lagged5 = np.column_stack([price, lag_matrix(flow, 5)])
    lagged20 = np.column_stack([price, lag_matrix(flow, 20)])
    future5 = np.column_stack([price, future_matrix(flow, 5)])
    control_matrices = {
        "lagged_5": lagged5,
        "lagged_20": lagged20,
        "future_5_negative_control": future5,
    }
    dd_indices = groups["drift_plus_diffusion"]
    for name in control_matrices:
        groups[name] = dd_indices

    result_targets: dict[str, Any] = {}
    for target_index, target_name in enumerate(matrix["target_names"]):
        target = matrix["targets"][:, target_index]
        direction = "_return_" in target_name
        predictions: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        actuals: list[np.ndarray] = []
        fold_receipts = []
        for outer_year in OUTER_YEARS:
            test_indices = np.asarray(
                [index for index, date in enumerate(dates) if int(date[:4]) == outer_year],
                dtype=np.int64,
            )
            test_indices = test_indices[np.isfinite(target[test_indices])]
            if not len(test_indices):
                continue
            first_test = int(test_indices[0])
            train_indices = np.arange(0, max(0, first_test - PURGE_SESSIONS), dtype=np.int64)
            train_indices = train_indices[np.isfinite(target[train_indices])]
            if len(train_indices) < 250:
                continue
            actuals.append(target[test_indices])
            fold_models = {}
            for model_name, feature_indices in groups.items():
                source_matrix = control_matrices.get(model_name, combined)
                train_matrix = source_matrix[train_indices][:, feature_indices]
                test_matrix = source_matrix[test_indices][:, feature_indices]
                if model_name == "date_block_shuffle":
                    raise AssertionError("date_block_shuffle is handled separately")
                alpha = tune_alpha(source_matrix[:, feature_indices], target, dates, train_indices)
                model = ridge_fit(train_matrix, target[train_indices], alpha)
                prediction = ridge_predict(test_matrix, model)
                predictions[model_name].append(prediction)
                fold_models[model_name] = {
                    "alpha": alpha,
                    **metric(target[test_indices], prediction, direction),
                }
            train_flow = block_shuffle(flow[train_indices], seed=outer_year * 1009 + target_index)
            test_flow = block_shuffle(flow[test_indices], seed=outer_year * 1013 + target_index)
            train_shuffled = np.column_stack([price[train_indices], train_flow])[:, dd_indices]
            test_shuffled = np.column_stack([price[test_indices], test_flow])[:, dd_indices]
            shuffled_alpha = tune_alpha(combined[:, dd_indices], target, dates, train_indices)
            shuffled_model = ridge_fit(train_shuffled, target[train_indices], shuffled_alpha)
            shuffled_prediction = ridge_predict(test_shuffled, shuffled_model)
            predictions["date_block_shuffle"].append(shuffled_prediction)
            fold_models["date_block_shuffle"] = {
                "alpha": shuffled_alpha,
                **metric(target[test_indices], shuffled_prediction, direction),
            }
            fold_receipts.append(
                {
                    "outer_year": outer_year,
                    "train_end_signal_date": dates[int(train_indices[-1])],
                    "test_start_signal_date": dates[int(test_indices[0])],
                    "test_end_signal_date": dates[int(test_indices[-1])],
                    "train_rows": len(train_indices),
                    "test_rows": len(test_indices),
                    "purge_sessions": PURGE_SESSIONS,
                    "models": fold_models,
                }
            )
        pooled_target = np.concatenate(actuals) if actuals else np.asarray([])
        pooled = {
            model_name: metric(
                pooled_target,
                np.concatenate(model_predictions),
                direction,
            )
            for model_name, model_predictions in predictions.items()
        }
        if pooled:
            price_mae = pooled["price_only"]["mae"]
            dd_mae = pooled["drift_plus_diffusion"]["mae"]
            pooled["drift_plus_diffusion"]["relative_mae_improvement_vs_price_pct"] = (
                (price_mae - dd_mae) / price_mae * 100.0 if price_mae else math.nan
            )
            for control in (
                "raw_flow",
                "drift_only",
                "diffusion_only",
                "lagged_5",
                "lagged_20",
                "date_block_shuffle",
                "mask_only",
            ):
                pooled["drift_plus_diffusion"][f"{control}_minus_dd_mae"] = (
                    pooled[control]["mae"] - dd_mae
                )
        result_targets[target_name] = {
            "rows": int(len(pooled_target)),
            "direction_target": direction,
            "pooled": pooled,
            "folds": fold_receipts,
        }
    return result_targets


def summarize_gate(targets: Mapping[str, Any]) -> dict[str, Any]:
    names = sorted(targets)
    dd_beats_price = 0
    dd_beats_shuffle = 0
    dd_beats_lag5 = 0
    dd_beats_raw = 0
    dd_beats_drift = 0
    improvements = []
    positive_fold_count = 0
    total_fold_count = 0
    for name in names:
        pooled = targets[name]["pooled"]
        if not pooled:
            continue
        dd = pooled["drift_plus_diffusion"]["mae"]
        dd_beats_price += dd < pooled["price_only"]["mae"]
        dd_beats_shuffle += dd < pooled["date_block_shuffle"]["mae"]
        dd_beats_lag5 += dd < pooled["lagged_5"]["mae"]
        dd_beats_raw += dd < pooled["raw_flow"]["mae"]
        dd_beats_drift += dd < pooled["drift_only"]["mae"]
        improvements.append(
            pooled["drift_plus_diffusion"]["relative_mae_improvement_vs_price_pct"]
        )
        for fold in targets[name]["folds"]:
            total_fold_count += 1
            positive_fold_count += (
                fold["models"]["drift_plus_diffusion"]["mae"]
                < fold["models"]["price_only"]["mae"]
            )
    counters = {
        "target_count": len(names),
        "dd_beats_price_count": int(dd_beats_price),
        "dd_beats_date_block_shuffle_count": int(dd_beats_shuffle),
        "dd_beats_lagged_5_count": int(dd_beats_lag5),
        "dd_beats_raw_flow_count": int(dd_beats_raw),
        "dd_beats_drift_only_count": int(dd_beats_drift),
        "mean_relative_mae_improvement_vs_price_pct": float(np.mean(improvements)),
        "positive_outer_fold_target_count": int(positive_fold_count),
        "outer_fold_target_count": int(total_fold_count),
    }
    checks = {
        "dd_beats_price_7_of_12": dd_beats_price >= 7,
        "mean_relative_mae_improvement_positive": bool(np.mean(improvements) > 0),
        "actual_beats_shuffle_7_of_12": dd_beats_shuffle >= 7,
        "actual_beats_lag5_7_of_12": dd_beats_lag5 >= 7,
        "dd_beats_raw_4_of_12": dd_beats_raw >= 4,
        "diffusion_adds_beyond_drift_4_of_12": dd_beats_drift >= 4,
        "positive_in_at_least_half_outer_fold_targets": positive_fold_count
        >= math.ceil(total_fold_count / 2),
    }
    return {
        "status": "PHASE_B_MARKET_SURVIVOR" if all(checks.values()) else "PHASE_B_MARKET_FAIL",
        "checks": checks,
        "counters": counters,
        "fixed_before_results": True,
        "interpretation": (
            "market survivor only activates cluster/stock interpretable tests; it does not prove alpha"
        ),
    }


def run(
    *,
    source_database: Path,
    phase_a_root: Path,
    output_root: Path,
    replace: bool,
) -> dict[str, Any]:
    if output_root.exists():
        if not replace:
            raise FileExistsError(output_root)
        import shutil

        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    started_at = utc_now()
    hypothesis_path = phase_a_root / "v11_r2_drift_diffusion_hypothesis_registry.json"
    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    expected_specification_hash = json_sha256(specification())
    if hypothesis.get("specification_sha256") != expected_specification_hash:
        raise ValueError("Phase A hypothesis registry hash mismatch")
    event_path = phase_a_root / "v11_r2_flow_event_cube.sqlite3"
    event = readonly_connection(event_path)
    source = readonly_connection(source_database)
    try:
        matrix = build_market_matrix(event=event, source=source)
        targets = evaluate_walk_forward(matrix)
    finally:
        event.close()
        source.close()
    gate = summarize_gate(targets)
    base_feature_groups = model_feature_sets(
        matrix["price_names"], matrix["flow_names"]
    )
    combined_names = tuple(matrix["price_names"]) + tuple(matrix["flow_names"])
    receipt = {
        "schema_version": PHASE_B_MARKET_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "started_at_utc": started_at,
        "timing_contract": TIMING_CONTRACT,
        "hypothesis_registry": {
            "path": str(hypothesis_path),
            "sha256": sha256_file(hypothesis_path),
            "specification_sha256": expected_specification_hash,
        },
        "sources": {
            "event_cube": {"path": str(event_path), "sha256": sha256_file(event_path)},
            "source_database": {"path": str(source_database), "bytes": source_database.stat().st_size},
        },
        "contract": {
            "target_anchors": list(TARGET_ANCHORS),
            "horizons": list(HORIZONS),
            "target_count": len(matrix["target_names"]),
            "outer_years": list(OUTER_YEARS),
            "purge_sessions": PURGE_SESSIONS,
            "alpha_grid": list(ALPHA_GRID),
            "date_centering_of_absolute_flow": False,
            "table_48_breadth_used": False,
            "phase_a_independent_family_breadth_used": True,
        },
        "feature_sets": {
            name: [combined_names[index] for index in indices]
            for name, indices in base_feature_groups.items()
        },
        "gate": gate,
        "targets": targets,
        "next_activation": (
            "PHASE_B_CLUSTER_STOCK_INTERPRETABLE"
            if gate["status"] == "PHASE_B_MARKET_SURVIVOR"
            else "MARKET_PATH_FAILED_BUT_CLUSTER_ROTATION_AND_AVOIDANCE_TESTS_STILL_REQUIRED"
        ),
        "phase_c_activation": "NOT_ACTIVATED",
    }
    output_path = output_root / "v11_r2_phase_b_market_receipt.json"
    write_json_atomic(output_path, receipt)
    return {
        "status": gate["status"],
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "gate": gate,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path, default=DEFAULT_SOURCE_DATABASE)
    parser.add_argument("--phase-a-root", type=Path, default=DEFAULT_PHASE_A_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        source_database=args.source_database,
        phase_a_root=args.phase_a_root,
        output_root=args.output_root,
        replace=args.replace,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PHASE_B_MARKET_SURVIVOR" else 3


if __name__ == "__main__":
    raise SystemExit(main())
