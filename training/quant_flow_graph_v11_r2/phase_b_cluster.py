"""Interpretable Phase B cluster rotation and downside-defense tournament."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import DEFAULT_SOURCE_DATABASE, TIMING_CONTRACT
from .hypotheses import specification
from .phase_a import json_sha256, readonly_connection, sha256_file, utc_now, write_json_atomic
from .phase_b_market import (
    ALPHA_GRID,
    OUTER_YEARS,
    PURGE_SESSIONS,
    PriceSeries,
    block_shuffle,
    build_market_matrix,
    future_targets,
    lag_matrix,
    load_price_series,
    metric,
    past_return,
    realized_volatility,
    ridge_fit,
    ridge_predict,
    rolling_mean,
    rolling_z,
    trailing_drawdown,
    tune_alpha,
)


PHASE_B_CLUSTER_SCHEMA_VERSION = "quant.etf_flow_v11_r2.phase_b_cluster.v1"
DEFAULT_PHASE_A_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/r2_phase_a"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/r2_phase_b_cluster"
)

# Frozen before reading Phase B cluster results. Each category has a liquid,
# economically interpretable US-listed price path. UNCLASSIFIED is explicitly
# excluded because it has no defensible target, not silently assigned.
CLUSTER_ANCHORS: Mapping[str, str] = {
    "DIVIDEND_INCOME": "SCHD",
    "GROWTH_MOMENTUM": "MTUM",
    "BROAD_MARKET_CORE_BETA": "SPY",
    "FINANCIALS": "XLF",
    "VALUE_QUALITY": "QUAL",
    "HIGH_YIELD_LOAN_CLO": "HYG",
    "CREDIT_CORPORATE_BOND": "LQD",
    "EMERGING_MARKETS": "EEM",
    "INFORMATION_TECHNOLOGY": "XLK",
    "ENERGY_OIL_GAS": "XLE",
    "RATES_TREASURY": "IEF",
    "CRYPTO_DIGITAL_ASSETS": "IBIT",
    "DEVELOPED_MARKETS": "EFA",
    "POWER_GRID_INFRASTRUCTURE": "PAVE",
    "SMALL_CAP_MID_CAP": "IWM",
    "REAL_ESTATE": "XLRE",
    "CHINA": "MCHI",
    "LARGE_CAP_MEGA_CAP": "SPY",
    "COMMUNICATION_SERVICES": "XLC",
    "METALS_MINING_MATERIALS": "XME",
    "AI_BROAD": "BOTZ",
    "HEALTH_CARE": "XLV",
    "INDUSTRIALS": "XLI",
    "CONSUMER_DISCRETIONARY": "XLY",
    "CURRENCY_FX": "UUP",
    "SEMICONDUCTOR_MEMORY": "SMH",
    "SPACE_DEFENSE": "ITA",
    "LOW_VOL_DEFENSIVE": "USMV",
    "SOFTWARE_CYBER": "IGV",
    "MATERIALS": "XLB",
    "UTILITIES": "XLU",
    "CONSUMER_STAPLES": "XLP",
    "INDIA": "INDA",
    "HEALTHCARE_BIOTECH": "XBI",
    "AGRICULTURE_FOOD": "DBA",
    "TRAVEL_LEISURE": "PEJ",
    "WATER": "PHO",
    "AUTO_EV_MOBILITY": "DRIV",
    "KOREA": "EWY",
    "NUCLEAR_URANIUM": "URA",
    "PHYSICAL_AI_ROBOTICS": "BOTZ",
    "JAPAN": "EWJ",
    "DATA_CENTER_CLOUD_INFRASTRUCTURE": "SKYY",
}
TARGET_NAMES = (
    "relative_return_5d_pct",
    "relative_return_20d_pct",
    "downside_defense_5d_pct",
    "downside_defense_20d_pct",
)


def cluster_flow_states(event: sqlite3.Connection) -> dict[tuple[str, str], dict[str, float]]:
    rows = event.execute(
        """
        WITH family_state AS (
          SELECT signal_date,cluster_family,independent_family_id,
            SUM(CASE WHEN observed_exact_t2=1 THEN fund_flow ELSE NULL END) family_flow,
            MAX(strict_eligible) strict_eligible,
            MAX(clean_eligible) clean_eligible,
            MAX(observed_exact_t2) observed,
            MAX(missing_exact_t2) missing,
            SUM(CASE WHEN observed_exact_t2=1 THEN assets_usd ELSE 0 END) observed_assets
          FROM etf_flow_events
          WHERE clean_eligible=1
          GROUP BY signal_date,cluster_family,independent_family_id
        )
        SELECT signal_date,cluster_family,
          COUNT(*) family_count,
          SUM(observed) observed_family_count,
          SUM(missing) missing_family_count,
          SUM(CASE WHEN family_flow>0 THEN 1 ELSE 0 END) positive_family_count,
          SUM(CASE WHEN family_flow<0 THEN 1 ELSE 0 END) negative_family_count,
          SUM(CASE WHEN family_flow=0 THEN 1 ELSE 0 END) zero_family_count,
          SUM(family_flow) signed_flow_usd,
          SUM(ABS(family_flow)) absolute_flow_usd,
          SUM(observed_assets) observed_assets_usd
        FROM family_state
        GROUP BY signal_date,cluster_family
        ORDER BY cluster_family,signal_date
        """
    )
    result: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        values = dict(row)
        family_count = float(values["family_count"] or 0)
        observed_count = float(values["observed_family_count"] or 0)
        positive = float(values["positive_family_count"] or 0)
        negative = float(values["negative_family_count"] or 0)
        signed = float(values["signed_flow_usd"] or 0)
        assets = float(values["observed_assets_usd"] or 0)
        result[(str(values["signal_date"]), str(values["cluster_family"]))] = {
            "cluster_family_count": family_count,
            "cluster_observed_family_count": observed_count,
            "cluster_missing_ratio": (
                float(values["missing_family_count"] or 0) / family_count
                if family_count
                else math.nan
            ),
            "cluster_coverage": observed_count / family_count if family_count else math.nan,
            "cluster_breadth_net": (
                (positive - negative) / observed_count if observed_count else math.nan
            ),
            "cluster_positive_share": positive / observed_count if observed_count else math.nan,
            "cluster_negative_share": negative / observed_count if observed_count else math.nan,
            "cluster_zero_share": (
                float(values["zero_family_count"] or 0) / observed_count
                if observed_count
                else math.nan
            ),
            "cluster_signed_flow_log": math.copysign(
                math.log1p(abs(signed) / 1_000_000.0), signed
            ),
            "cluster_absolute_flow_log": math.log1p(
                float(values["absolute_flow_usd"] or 0) / 1_000_000.0
            ),
            "cluster_flow_rate_pct": signed / assets * 100.0 if assets > 0 else math.nan,
            "cluster_special_effective_signed_flow_log": 0.0,
            "cluster_special_flow_rate_pct": 0.0,
        }
    special_rows = event.execute(
        """
        SELECT signal_date,cluster_family,
          SUM(fund_flow*effective_sign*ABS(target_multiple)) special_effective_flow,
          SUM(assets_usd) special_assets
        FROM etf_flow_events
        WHERE special_eligible=1 AND observed_exact_t2=1
        GROUP BY signal_date,cluster_family
        """
    )
    for date, cluster, special_flow, special_assets in special_rows:
        key = (str(date), str(cluster))
        if key not in result:
            continue
        flow = float(special_flow or 0)
        assets = float(special_assets or 0)
        result[key]["cluster_special_effective_signed_flow_log"] = math.copysign(
            math.log1p(abs(flow) / 1_000_000.0), flow
        )
        result[key]["cluster_special_flow_rate_pct"] = (
            flow / assets * 100.0 if assets > 0 else math.nan
        )
    return result


def cluster_catalog(event: sqlite3.Connection) -> dict[str, int]:
    return {
        str(row[0]): int(row[1])
        for row in event.execute(
            "SELECT cluster_family,COUNT(DISTINCT ticker) FROM etf_flow_events GROUP BY cluster_family"
        )
    }


def _cluster_sequences(
    states: Mapping[tuple[str, str], Mapping[str, float]],
    dates: Sequence[str],
) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = {
        key: dict(value) for key, value in states.items()
    }
    clusters = sorted({key[1] for key in states})
    date_index = {date: index for index, date in enumerate(dates)}
    for cluster in clusters:
        rows = [result.get((date, cluster), {}) for date in dates]
        for base_name in (
            "cluster_flow_rate_pct",
            "cluster_breadth_net",
            "cluster_coverage",
        ):
            values = np.asarray(
                [float(row.get(base_name, math.nan)) for row in rows], dtype=np.float64
            )
            mean5 = rolling_mean(values, 5)
            mean20 = rolling_mean(values, 20)
            z60 = rolling_z(values, 60)
            change5 = np.full(len(values), np.nan)
            change5[5:] = values[5:] - values[:-5]
            for date in dates:
                position = date_index[date]
                target = result.setdefault((date, cluster), {})
                target[f"{base_name}_mean_5"] = float(mean5[position])
                target[f"{base_name}_mean_20"] = float(mean20[position])
                target[f"{base_name}_z60"] = float(z60[position])
                target[f"{base_name}_change_5"] = float(change5[position])
    return result


def build_cluster_matrix(
    *, event: sqlite3.Connection, source: sqlite3.Connection
) -> dict[str, Any]:
    market = build_market_matrix(event=event, source=source)
    dates = market["dates"]
    date_to_market = {date: index for index, date in enumerate(dates)}
    states = _cluster_sequences(cluster_flow_states(event), dates)
    catalog = cluster_catalog(event)
    spy = load_price_series(source, "SPY")
    anchors = {
        symbol: load_price_series(source, symbol)
        for symbol in sorted(set(CLUSTER_ANCHORS.values()))
    }
    clusters = tuple(sorted(CLUSTER_ANCHORS))
    cluster_to_id = {cluster: index for index, cluster in enumerate(clusters)}
    market_price_names = tuple(market["price_names"])
    market_flow_names = tuple(market["flow_names"])
    cluster_price_names = (
        "cluster_ret_1d",
        "cluster_ret_5d",
        "cluster_ret_20d",
        "cluster_vol_20d",
        "cluster_drawdown_20d",
        "cluster_minus_spy_ret_5d",
    )
    cluster_flow_names = (
        "cluster_family_count",
        "cluster_observed_family_count",
        "cluster_missing_ratio",
        "cluster_coverage",
        "cluster_breadth_net",
        "cluster_positive_share",
        "cluster_negative_share",
        "cluster_zero_share",
        "cluster_signed_flow_log",
        "cluster_absolute_flow_log",
        "cluster_flow_rate_pct",
        "cluster_special_effective_signed_flow_log",
        "cluster_special_flow_rate_pct",
        "cluster_flow_rate_pct_mean_5",
        "cluster_flow_rate_pct_mean_20",
        "cluster_flow_rate_pct_z60",
        "cluster_flow_rate_pct_change_5",
        "cluster_breadth_net_mean_5",
        "cluster_breadth_net_mean_20",
        "cluster_breadth_net_z60",
        "cluster_breadth_net_change_5",
        "cluster_coverage_mean_5",
        "cluster_coverage_mean_20",
        "cluster_coverage_z60",
        "cluster_coverage_change_5",
        "drift_cluster_convergence",
        "cluster_minus_market_flow_rate",
    )
    row_dates: list[str] = []
    row_clusters: list[str] = []
    price_rows: list[list[float]] = []
    flow_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    excluded = defaultdict(int)
    for date in dates:
        market_position = date_to_market[date]
        price_date = market["price_dates"][market_position]
        market_price = market["price_matrix"][market_position]
        market_flow = market["flow_matrix"][market_position]
        drift_rate = market_flow[market_flow_names.index("drift_rate_pct")]
        for cluster in clusters:
            symbol = CLUSTER_ANCHORS[cluster]
            series = anchors[symbol]
            position = series.index.get(price_date)
            spy_position = spy.index.get(price_date)
            if position is None or spy_position is None:
                excluded["missing_anchor_price_date"] += 1
                continue
            state = states.get((date, cluster))
            if not state:
                excluded["missing_cluster_flow_state"] += 1
                continue
            cluster_price = [
                past_return(series, position, 1),
                past_return(series, position, 5),
                past_return(series, position, 20),
                realized_volatility(series, position, 20),
                trailing_drawdown(series, position, 20),
                past_return(series, position, 5) - past_return(spy, spy_position, 5),
            ]
            cluster_rate = float(state.get("cluster_flow_rate_pct", math.nan))
            convergence = (
                math.copysign(1.0, cluster_rate) * math.copysign(1.0, drift_rate)
                if math.isfinite(cluster_rate)
                and math.isfinite(drift_rate)
                and cluster_rate != 0
                and drift_rate != 0
                else 0.0
            )
            state_with_relations = {
                **state,
                "drift_cluster_convergence": convergence,
                "cluster_minus_market_flow_rate": cluster_rate - drift_rate,
            }
            one_hot = [0.0] * len(clusters)
            one_hot[cluster_to_id[cluster]] = 1.0
            price_rows.append(
                list(market_price) + cluster_price + one_hot
            )
            flow_rows.append(
                list(market_flow)
                + [float(state_with_relations.get(name, math.nan)) for name in cluster_flow_names]
            )
            targets = []
            for horizon in (5, 20):
                cluster_return, cluster_loss = future_targets(series, price_date, horizon)
                spy_return, spy_loss = future_targets(spy, price_date, horizon)
                targets.extend((cluster_return - spy_return, spy_loss - cluster_loss))
            # Interleaved loop produced return5, defense5, return20, defense20.
            target_rows.append(targets)
            row_dates.append(date)
            row_clusters.append(cluster)
    target_names = (
        "relative_return_5d_pct",
        "downside_defense_5d_pct",
        "relative_return_20d_pct",
        "downside_defense_20d_pct",
    )
    return {
        "dates": tuple(row_dates),
        "clusters": tuple(row_clusters),
        "price_names": market_price_names
        + cluster_price_names
        + tuple(f"cluster_id:{cluster}" for cluster in clusters),
        "price_matrix": np.asarray(price_rows, dtype=np.float64),
        "market_flow_names": market_flow_names,
        "cluster_flow_names": cluster_flow_names,
        "flow_names": market_flow_names + cluster_flow_names,
        "flow_matrix": np.asarray(flow_rows, dtype=np.float64),
        "target_names": target_names,
        "targets": np.asarray(target_rows, dtype=np.float64),
        "catalog": catalog,
        "anchor_mapping": dict(CLUSTER_ANCHORS),
        "excluded": dict(excluded),
        "unmapped_catalog": {
            cluster: count for cluster, count in catalog.items() if cluster not in CLUSTER_ANCHORS
        },
    }


def cluster_feature_sets(matrix: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    price_count = matrix["price_matrix"].shape[1]
    price = tuple(range(price_count))
    flow_names = matrix["flow_names"]
    flow_index = {name: price_count + index for index, name in enumerate(flow_names)}
    market_names = set(matrix["market_flow_names"])
    cluster_names = set(matrix["cluster_flow_names"])
    mask_names = {
        name
        for name in flow_names
        if "coverage" in name or "missing" in name or "zero" in name
    }
    market_dd_names = {
        name
        for name in market_names
        if name.startswith("drift_") or "breadth" in name or "diffusion" in name
    }
    cluster_dd_names = {
        name
        for name in cluster_names
        if name
        not in {
            "cluster_family_count",
            "cluster_observed_family_count",
        }
    }
    def make(names: set[str]) -> tuple[int, ...]:
        return price + tuple(flow_index[name] for name in flow_names if name in names)

    return {
        "price_only": price,
        "mask_only": make(mask_names),
        "market_drift_diffusion_only": make(market_dd_names | mask_names),
        "cluster_flow_only": make(cluster_dd_names | mask_names),
        "market_plus_cluster_dd": make(market_dd_names | cluster_dd_names | mask_names),
    }


def shuffle_within_cluster(
    flow: np.ndarray, clusters: Sequence[str], seed: int
) -> np.ndarray:
    result = np.empty_like(flow)
    for cluster in sorted(set(clusters)):
        indices = np.asarray([i for i, value in enumerate(clusters) if value == cluster])
        result[indices] = block_shuffle(flow[indices], seed + sum(map(ord, cluster)))
    return result


def cross_sectional_metrics(
    dates: Sequence[str], target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    correlations = []
    spreads = []
    avoidance = []
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, date in enumerate(dates):
        grouped[date].append(index)
    for date in sorted(grouped):
        indices = np.asarray(grouped[date], dtype=np.int64)
        if len(indices) < 6:
            continue
        actual = target[indices]
        forecast = prediction[indices]
        if np.std(actual) > 1e-12 and np.std(forecast) > 1e-12:
            rank_actual = np.argsort(np.argsort(actual)).astype(float)
            rank_forecast = np.argsort(np.argsort(forecast)).astype(float)
            correlations.append(float(np.corrcoef(rank_actual, rank_forecast)[0, 1]))
        count = max(1, len(indices) // 5)
        order = np.argsort(forecast)
        spreads.append(float(np.mean(actual[order[-count:]]) - np.mean(actual[order[:count]])))
        avoidance.append(float(np.mean(actual[order[:count]])))
    return {
        "mean_daily_rank_ic": float(np.mean(correlations)) if correlations else math.nan,
        "positive_daily_rank_ic_ratio": float(np.mean(np.asarray(correlations) > 0)) if correlations else math.nan,
        "mean_top_minus_bottom_spread": float(np.mean(spreads)) if spreads else math.nan,
        "mean_predicted_bottom_realized": float(np.mean(avoidance)) if avoidance else math.nan,
        "evaluated_date_count": len(spreads),
    }


def evaluate(matrix: Mapping[str, Any]) -> dict[str, Any]:
    dates = matrix["dates"]
    clusters = matrix["clusters"]
    price = matrix["price_matrix"]
    flow = matrix["flow_matrix"]
    combined = np.column_stack([price, flow])
    groups = cluster_feature_sets(matrix)
    dd_indices = groups["market_plus_cluster_dd"]
    lagged5 = np.column_stack([price, lag_matrix_by_cluster(flow, clusters, 5)])
    lagged20 = np.column_stack([price, lag_matrix_by_cluster(flow, clusters, 20)])
    groups["lagged_5"] = dd_indices
    groups["lagged_20"] = dd_indices
    controls = {"lagged_5": lagged5, "lagged_20": lagged20}
    receipts = {}
    for target_index, target_name in enumerate(matrix["target_names"]):
        target = matrix["targets"][:, target_index]
        predictions: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        actuals: list[np.ndarray] = []
        dates_out: list[str] = []
        folds = []
        for year in OUTER_YEARS:
            test = np.asarray(
                [i for i, date in enumerate(dates) if int(date[:4]) == year and np.isfinite(target[i])],
                dtype=np.int64,
            )
            if not len(test):
                continue
            test_start_date = dates[int(test[0])]
            unique_prior_dates = sorted({date for date in dates if date < test_start_date})
            allowed_train_dates = set(unique_prior_dates[:-PURGE_SESSIONS])
            train = np.asarray(
                [i for i, date in enumerate(dates) if date in allowed_train_dates and np.isfinite(target[i])],
                dtype=np.int64,
            )
            if len(train) < 2000:
                continue
            actuals.append(target[test])
            dates_out.extend(dates[i] for i in test)
            fold_models = {}
            for model_name, indices in groups.items():
                source_matrix = controls.get(model_name, combined)
                alpha = tune_alpha_cluster(source_matrix[:, indices], target, dates, train)
                model = ridge_fit(source_matrix[train][:, indices], target[train], alpha)
                prediction = ridge_predict(source_matrix[test][:, indices], model)
                predictions[model_name].append(prediction)
                fold_models[model_name] = {"alpha": alpha, **metric(target[test], prediction, True)}
            train_shuffle = shuffle_within_cluster(flow[train], [clusters[i] for i in train], year * 1009 + target_index)
            test_shuffle = shuffle_within_cluster(flow[test], [clusters[i] for i in test], year * 1013 + target_index)
            model = ridge_fit(
                np.column_stack([price[train], train_shuffle])[:, dd_indices],
                target[train],
                1.0,
            )
            prediction = ridge_predict(
                np.column_stack([price[test], test_shuffle])[:, dd_indices], model
            )
            predictions["date_block_shuffle"].append(prediction)
            fold_models["date_block_shuffle"] = {"alpha": 1.0, **metric(target[test], prediction, True)}
            folds.append(
                {
                    "outer_year": year,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_end_signal_date": max(dates[i] for i in train),
                    "test_start_signal_date": min(dates[i] for i in test),
                    "test_end_signal_date": max(dates[i] for i in test),
                    "purge_sessions": PURGE_SESSIONS,
                    "models": fold_models,
                }
            )
        actual = np.concatenate(actuals)
        pooled = {}
        for model_name, values in predictions.items():
            prediction = np.concatenate(values)
            pooled[model_name] = {
                **metric(actual, prediction, True),
                **cross_sectional_metrics(dates_out, actual, prediction),
            }
        price_mae = pooled["price_only"]["mae"]
        dd_mae = pooled["market_plus_cluster_dd"]["mae"]
        pooled["market_plus_cluster_dd"]["relative_mae_improvement_vs_price_pct"] = (
            (price_mae - dd_mae) / price_mae * 100.0
        )
        receipts[target_name] = {
            "rows": len(actual),
            "pooled": pooled,
            "folds": folds,
        }
    return receipts


def lag_matrix_by_cluster(
    matrix: np.ndarray, clusters: Sequence[str], sessions: int
) -> np.ndarray:
    result = np.full_like(matrix, np.nan)
    for cluster in sorted(set(clusters)):
        indices = np.asarray([i for i, value in enumerate(clusters) if value == cluster])
        result[indices] = lag_matrix(matrix[indices], sessions)
    return result


def tune_alpha_cluster(
    matrix: np.ndarray,
    target: np.ndarray,
    dates: Sequence[str],
    train: np.ndarray,
) -> float:
    years = sorted({int(dates[i][:4]) for i in train})
    if len(years) < 3:
        return 1.0
    validation_year = years[-1]
    validation_dates = sorted({dates[i] for i in train if int(dates[i][:4]) == validation_year})
    prior_dates = sorted({dates[i] for i in train if int(dates[i][:4]) < validation_year})
    allowed_inner_dates = set(prior_dates[:-PURGE_SESSIONS])
    inner = train[np.asarray([dates[i] in allowed_inner_dates for i in train])]
    validation = train[np.asarray([dates[i] in set(validation_dates) for i in train])]
    if len(inner) < 1000 or len(validation) < 200:
        return 1.0
    scores = []
    for alpha in ALPHA_GRID:
        model = ridge_fit(matrix[inner], target[inner], alpha)
        prediction = ridge_predict(matrix[validation], model)
        scores.append((float(np.mean(np.abs(prediction - target[validation]))), alpha))
    return min(scores)[1]


def summarize_gate(targets: Mapping[str, Any]) -> dict[str, Any]:
    counters = CounterLike()
    improvements = []
    fold_positive = 0
    fold_total = 0
    for target in targets.values():
        pooled = target["pooled"]
        dd = pooled["market_plus_cluster_dd"]
        dd_mae = dd["mae"]
        counters.add("dd_beats_price", dd_mae < pooled["price_only"]["mae"])
        counters.add("dd_beats_shuffle", dd_mae < pooled["date_block_shuffle"]["mae"])
        counters.add("dd_beats_lag5", dd_mae < pooled["lagged_5"]["mae"])
        counters.add(
            "cluster_adds_beyond_market",
            dd_mae < pooled["market_drift_diffusion_only"]["mae"],
        )
        counters.add(
            "rank_ic_beats_price",
            dd["mean_daily_rank_ic"] > pooled["price_only"]["mean_daily_rank_ic"],
        )
        counters.add(
            "basket_spread_beats_price",
            dd["mean_top_minus_bottom_spread"]
            > pooled["price_only"]["mean_top_minus_bottom_spread"],
        )
        improvements.append(dd["relative_mae_improvement_vs_price_pct"])
        for fold in target["folds"]:
            fold_total += 1
            fold_positive += (
                fold["models"]["market_plus_cluster_dd"]["mae"]
                < fold["models"]["price_only"]["mae"]
            )
    count = len(targets)
    checks = {
        "dd_beats_price_3_of_4": counters["dd_beats_price"] >= 3,
        "mean_mae_improvement_positive": bool(np.mean(improvements) > 0),
        "actual_beats_shuffle_3_of_4": counters["dd_beats_shuffle"] >= 3,
        "actual_beats_lag5_3_of_4": counters["dd_beats_lag5"] >= 3,
        "cluster_adds_beyond_market_2_of_4": counters["cluster_adds_beyond_market"] >= 2,
        "rank_ic_beats_price_3_of_4": counters["rank_ic_beats_price"] >= 3,
        "basket_spread_beats_price_2_of_4": counters["basket_spread_beats_price"] >= 2,
        "positive_half_outer_fold_targets": fold_positive >= math.ceil(fold_total / 2),
    }
    return {
        "status": "PHASE_B_CLUSTER_SURVIVOR" if all(checks.values()) else "PHASE_B_CLUSTER_FAIL",
        "checks": checks,
        "counters": {
            **dict(counters),
            "target_count": count,
            "mean_relative_mae_improvement_vs_price_pct": float(np.mean(improvements)),
            "positive_outer_fold_target_count": int(fold_positive),
            "outer_fold_target_count": int(fold_total),
        },
        "fixed_before_results": True,
    }


class CounterLike(defaultdict):
    def __init__(self) -> None:
        super().__init__(int)

    def add(self, key: str, condition: bool) -> None:
        self[key] += int(bool(condition))


def run(
    *, source_database: Path, phase_a_root: Path, output_root: Path, replace: bool
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
    expected_hash = json_sha256(specification())
    if hypothesis.get("specification_sha256") != expected_hash:
        raise ValueError("Phase A hypothesis registry hash mismatch")
    event_path = phase_a_root / "v11_r2_flow_event_cube.sqlite3"
    event = readonly_connection(event_path)
    source = readonly_connection(source_database)
    try:
        matrix = build_cluster_matrix(event=event, source=source)
        targets = evaluate(matrix)
    finally:
        event.close()
        source.close()
    gate = summarize_gate(targets)
    feature_groups = cluster_feature_sets(matrix)
    names = tuple(matrix["price_names"]) + tuple(matrix["flow_names"])
    receipt = {
        "schema_version": PHASE_B_CLUSTER_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "started_at_utc": started_at,
        "timing_contract": TIMING_CONTRACT,
        "hypothesis_registry_sha256": sha256_file(hypothesis_path),
        "hypothesis_specification_sha256": expected_hash,
        "source_event_cube_sha256": sha256_file(event_path),
        "contract": {
            "outer_years": list(OUTER_YEARS),
            "purge_sessions": PURGE_SESSIONS,
            "target_names": list(matrix["target_names"]),
            "anchor_mapping_frozen_before_results": dict(CLUSTER_ANCHORS),
            "table_48_breadth_used": False,
            "date_centering_used": False,
        },
        "scope": {
            "catalog_cluster_count": len(matrix["catalog"]),
            "mapped_cluster_count": len(CLUSTER_ANCHORS),
            "unmapped_catalog": matrix["unmapped_catalog"],
            "matrix_rows": len(matrix["dates"]),
            "excluded": matrix["excluded"],
        },
        "feature_sets": {
            model: [names[index] for index in indices]
            for model, indices in feature_groups.items()
        },
        "gate": gate,
        "targets": targets,
        "next_activation": (
            "PHASE_B_STOCK_PATH_REQUIRED"
            if gate["status"] == "PHASE_B_CLUSTER_SURVIVOR"
            else "CLUSTER_PATH_FAILED_BUT_STOCK_AVOIDANCE_TEST_STILL_REQUIRED"
        ),
        "phase_c_activation": "NOT_ACTIVATED",
        "limitations": [
            "cluster identity is current/static and must be refit inside each later graph fold",
            "representative anchor tests cluster-level economic paths; they do not substitute for stock-level holdings paths",
        ],
    }
    output_path = output_root / "v11_r2_phase_b_cluster_receipt.json"
    write_json_atomic(output_path, receipt)
    return {
        "status": gate["status"],
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "gate": gate,
        "scope": receipt["scope"],
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
    return 0 if result["status"] == "PHASE_B_CLUSTER_SURVIVOR" else 3


if __name__ == "__main__":
    raise SystemExit(main())
