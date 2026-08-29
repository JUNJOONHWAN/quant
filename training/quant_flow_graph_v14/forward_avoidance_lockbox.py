"""Run the preregistered v14 ETF-Flow forward avoidance micro-lockbox.

The experiment extends the audited v11 event cube through 2026-07-29 without
changing its historical prefix.  It then fits the frozen v12 CatBoost residual
and v13 deterministic graph-state variants on dates before a 20-session purge,
and evaluates exactly eleven previously unused signal dates.

The Massive Flow repair is a conservatively availability-gated historical
backfill, not an as-observed archive.  Consequently even a PASS is preliminary
and cannot activate trading, deployment, BF16 training, or NVFP4 conversion.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.contracts import (
    DEFAULT_ETFRADAR_ROOT,
    TIMING_CONTRACT,
)
from training.quant_flow_graph_v11_r2.phase_a import (
    build_event_cube,
    load_metadata,
    readonly_connection,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from training.quant_flow_graph_v11_r2.phase_b_stock import (
    PURGE_SESSIONS,
    TARGET_NAMES,
    build_stock_matrix_from_sources,
    regression_metrics,
    stock_cross_sectional_metrics,
)
from training.quant_flow_graph_v12.channel_ablation import classify_feature_channels
from training.quant_flow_graph_v12.residual_canary import (
    CATBOOST_PARAMETERS,
    CATBOOST_VERSION,
    capped_residual_prediction,
    date_balanced_weights,
    fit_predict_multioutput,
    residual_caps,
)
from training.quant_flow_graph_v13.adaptive_graph_state import (
    RANDOM_SEED,
    STATE_FIELDS,
    build_graph_state_features,
    causal_regime_schedule,
    state_feature_contract,
    topology_shuffle_state_inputs,
)


SCHEMA_VERSION = "quant.etf_flow_v14.forward_avoidance_lockbox.v1"
PREREGISTRATION_SHA256 = (
    "2808842bb52759bfe42c14f1357edac19900daf6196d1d85572dff83913b2c18"
)
BASE_END = "2026-07-14"
INCREMENTAL_DAILY_START = "2026-07-15"
REPAIR_FLOW_DATES = ("2026-07-15", "2026-07-16")
INCREMENTAL_FLOW_START = "2026-07-17"
TEST_DATES = (
    "2026-07-15",
    "2026-07-16",
    "2026-07-17",
    "2026-07-20",
    "2026-07-21",
    "2026-07-22",
    "2026-07-23",
    "2026-07-24",
    "2026-07-27",
    "2026-07-28",
    "2026-07-29",
)
PRIMARY_TARGETS = (
    "loss_5d_pct",
    "benchmark_downside_defense_20d_pct",
)
SECONDARY_AVOIDANCE_TARGETS = (
    "loss_20d_pct",
    "benchmark_downside_defense_5d_pct",
)

PRICE_MODEL = "price_only"
CURRENT_MODEL = "current_flow_structure"
FIXED_MODEL = "fixed_20_session_graph_state"
ADAPTIVE_MODEL = "adaptive_graph_state"
LAG5_MODEL = "adaptive_graph_state_lag5"
SHUFFLED_MODEL = "adaptive_graph_state_topology_shuffle"
MODEL_NAMES = (
    PRICE_MODEL,
    CURRENT_MODEL,
    FIXED_MODEL,
    ADAPTIVE_MODEL,
    LAG5_MODEL,
    SHUFFLED_MODEL,
)

DEFAULT_BASE_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/"
    "daily_observations.sqlite3"
)
DEFAULT_INCREMENTAL_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/incremental/"
    "normalized/daily_observations.sqlite3"
)
DEFAULT_REPAIRED_FLOW_CACHE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v1/"
    "repaired_flow_cache_20260715_20260722.sqlite3"
)
DEFAULT_OLD_EVENT_CUBE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/"
    "r2_phase_a/v11_r2_flow_event_cube.sqlite3"
)
DEFAULT_FAMILY_REGISTRY = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/"
    "r2_phase_a/v11_r2_etf_family_exposure_registry.sqlite3"
)
DEFAULT_GRAPH_DATASET_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/"
    "full_20180102_20260729_allpanel"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v14/"
    "forward_avoidance_lockbox_20260715_20260729"
)

EXPECTED_HASHES = {
    "incremental_database": (
        "275874b736568d1a5f6d2d4deb4bc7d494f576b77f4fb5d4751fd1d0ab4c3290"
    ),
    "repaired_flow_cache": (
        "8e49b20f74b73728bf65f1a27fe07117272cfd5b1f6136047a1f98190e8bc423"
    ),
    "old_event_cube": (
        "f50e2090312c56edf0681cf10028ef8a8adf074ac85740ed36412dfc51a24ec3"
    ),
    "family_registry": (
        "b8999da3b7cf0e1ec389385f430f7aa222924d58651ab094f24bff89f21c91e5"
    ),
    "graph_manifest": (
        "1dcbdbbd8127ec1659ebcc3c8123863c8e2fa1611c0636c86329daebc967179c"
    ),
}
EXPECTED_BASE_STAT = {
    "bytes": 100943228928,
    "mtime_ns": 1787844753218246628,
}


def _progress(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}, sort_keys=True), flush=True)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _attach_readonly(
    connection: sqlite3.Connection, schema: str, path: Path
) -> None:
    uri = f"file:{Path(path).resolve()}?mode=ro"
    connection.execute(f"ATTACH DATABASE ? AS {_quote_identifier(schema)}", (uri,))


def table_columns(connection: sqlite3.Connection, schema: str, table: str) -> tuple[str, ...]:
    rows = connection.execute(
        f"PRAGMA {_quote_identifier(schema)}.table_info({_quote_identifier(table)})"
    ).fetchall()
    return tuple(str(row[1]) for row in rows)


def open_union_source(
    *,
    base_database: Path,
    incremental_database: Path,
    repaired_flow_cache: Path,
) -> sqlite3.Connection:
    """Expose a read-only, non-overlapping PIT union under the v11 table names."""

    connection = sqlite3.connect(
        "file:v14_union_source?mode=memory&cache=private", uri=True
    )
    connection.row_factory = sqlite3.Row
    _attach_readonly(connection, "base", base_database)
    _attach_readonly(connection, "incremental", incremental_database)
    _attach_readonly(connection, "repair", repaired_flow_cache)
    base_daily = table_columns(connection, "base", "daily_observations")
    incremental_daily = table_columns(connection, "incremental", "daily_observations")
    if not base_daily or base_daily != incremental_daily:
        raise ValueError("base/incremental daily_observations schema mismatch")
    required_flow = {
        "ticker",
        "effective_date",
        "available_at_date",
        "processed_date",
        "fund_flow",
        "nav",
        "shares_outstanding",
    }
    for schema in ("base", "incremental"):
        missing = required_flow.difference(
            table_columns(connection, schema, "etf_flow_observations")
        )
        if missing:
            raise ValueError(f"{schema} Flow schema missing: {sorted(missing)}")
    repair_required = {
        "ticker",
        "effective_date",
        "available_session",
        "processed_date",
        "fund_flow",
        "nav",
        "shares_outstanding",
    }
    missing_repair = repair_required.difference(table_columns(connection, "repair", "flow"))
    if missing_repair:
        raise ValueError(f"repair Flow schema missing: {sorted(missing_repair)}")

    connection.execute(
        """
        CREATE TEMP VIEW daily_observations AS
        SELECT * FROM base.daily_observations WHERE trade_date<=
        """
        + repr(BASE_END)
        + " UNION ALL SELECT * FROM incremental.daily_observations WHERE trade_date>="
        + repr(INCREMENTAL_DAILY_START)
    )
    flow_projection = (
        "ticker,effective_date,available_at_date,processed_date,"
        "fund_flow,nav,shares_outstanding"
    )
    repair_dates = ",".join(repr(value) for value in REPAIR_FLOW_DATES)
    connection.execute(
        "CREATE TEMP VIEW etf_flow_observations AS "
        f"SELECT {flow_projection} FROM base.etf_flow_observations "
        f"WHERE effective_date<={BASE_END!r} UNION ALL "
        "SELECT ticker,effective_date,available_session AS available_at_date,"
        "processed_date,fund_flow,nav,shares_outstanding FROM repair.flow "
        f"WHERE effective_date IN ({repair_dates}) UNION ALL "
        f"SELECT {flow_projection} FROM incremental.etf_flow_observations "
        f"WHERE effective_date>={INCREMENTAL_FLOW_START!r}"
    )
    connection.execute("BEGIN")
    return connection


def union_source_audit(connection: sqlite3.Connection) -> dict[str, Any]:
    daily = connection.execute(
        "SELECT MIN(trade_date),MAX(trade_date),COUNT(*),COUNT(DISTINCT symbol) "
        "FROM daily_observations"
    ).fetchone()
    flow = connection.execute(
        "SELECT MIN(effective_date),MAX(effective_date),COUNT(*),COUNT(DISTINCT ticker) "
        "FROM etf_flow_observations"
    ).fetchone()
    repair = [
        dict(row)
        for row in connection.execute(
            "SELECT effective_date,COUNT(*) AS rows,COUNT(DISTINCT ticker) AS tickers,"
            "MIN(available_at_date) AS min_available,MAX(available_at_date) AS max_available "
            "FROM etf_flow_observations WHERE effective_date IN (?,?) "
            "GROUP BY effective_date ORDER BY effective_date",
            REPAIR_FLOW_DATES,
        )
    ]
    return {
        "daily": {
            "min_trade_date": str(daily[0]),
            "max_trade_date": str(daily[1]),
            "rows": int(daily[2]),
            "symbols": int(daily[3]),
        },
        "flow": {
            "min_effective_date": str(flow[0]),
            "max_effective_date": str(flow[1]),
            "rows": int(flow[2]),
            "tickers": int(flow[3]),
        },
        "repair_dates": repair,
    }


def verify_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    preregistration = Path(args.output_root) / "v14_forward_avoidance_preregistration.json"
    if sha256_file(preregistration) != PREREGISTRATION_SHA256:
        raise ValueError("v14 preregistration hash mismatch")
    stat = Path(args.base_database).stat()
    if stat.st_size != EXPECTED_BASE_STAT["bytes"] or stat.st_mtime_ns != EXPECTED_BASE_STAT["mtime_ns"]:
        raise ValueError("base database stat fingerprint mismatch")
    paths = {
        "incremental_database": Path(args.incremental_database),
        "repaired_flow_cache": Path(args.repaired_flow_cache),
        "old_event_cube": Path(args.old_event_cube),
        "family_registry": Path(args.family_registry),
        "graph_manifest": Path(args.graph_dataset_root) / "manifest.json",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_HASHES[name], "observed": digest}
        for name, digest in observed.items()
        if digest != EXPECTED_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"frozen input hash mismatch: {mismatches}")
    return {
        "preregistration": PREREGISTRATION_SHA256,
        "base_database": {**EXPECTED_BASE_STAT},
        **observed,
    }


def prefix_identity_audit(
    *, old_event_cube: Path, new_event_cube: Path, cutoff: str = BASE_END
) -> dict[str, Any]:
    """Compare every historical key and value using SQLite's NULL-safe IS."""

    connection = sqlite3.connect(
        "file:v14_prefix_identity?mode=memory&cache=private", uri=True
    )
    _attach_readonly(connection, "old", old_event_cube)
    _attach_readonly(connection, "new", new_event_cube)
    contracts = {
        "session_map": ("signal_date",),
        "daily_flow_state": ("signal_date",),
        "etf_flow_events": ("signal_date", "ticker"),
    }
    result: dict[str, Any] = {}
    try:
        for table, keys in contracts.items():
            old_columns = table_columns(connection, "old", table)
            new_columns = table_columns(connection, "new", table)
            if old_columns != new_columns:
                raise ValueError(f"prefix table schema mismatch: {table}")
            quoted_table = _quote_identifier(table)
            join = " AND ".join(
                f"o.{_quote_identifier(key)} IS n.{_quote_identifier(key)}" for key in keys
            )
            first_key = _quote_identifier(keys[0])
            value_columns = [name for name in old_columns if name not in keys]
            mismatch = " OR ".join(
                f"o.{_quote_identifier(name)} IS NOT n.{_quote_identifier(name)}"
                for name in value_columns
            ) or "0"
            old_count = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM old.{quoted_table} WHERE signal_date<=?",
                    (cutoff,),
                ).fetchone()[0]
            )
            new_count = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM new.{quoted_table} WHERE signal_date<=?",
                    (cutoff,),
                ).fetchone()[0]
            )
            missing = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM old.{quoted_table} o LEFT JOIN new.{quoted_table} n "
                    f"ON {join} WHERE o.signal_date<=? AND n.{first_key} IS NULL",
                    (cutoff,),
                ).fetchone()[0]
            )
            extra = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM new.{quoted_table} n LEFT JOIN old.{quoted_table} o "
                    f"ON {join} WHERE n.signal_date<=? AND o.{first_key} IS NULL",
                    (cutoff,),
                ).fetchone()[0]
            )
            changed = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM old.{quoted_table} o JOIN new.{quoted_table} n "
                    f"ON {join} WHERE o.signal_date<=? AND ({mismatch})",
                    (cutoff,),
                ).fetchone()[0]
            )
            result[table] = {
                "old_rows": old_count,
                "new_rows": new_count,
                "missing_keys": missing,
                "extra_keys": extra,
                "value_mismatches": changed,
                "passed": old_count == new_count and missing == extra == changed == 0,
            }
    finally:
        connection.close()
    result["passed"] = all(item["passed"] for item in result.values())
    if not result["passed"]:
        raise ValueError(f"historical event-cube prefix mismatch: {result}")
    return result


def split_indices(matrix: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dates = tuple(str(value) for value in matrix["date_values"])
    if any(value not in dates for value in TEST_DATES):
        missing = [value for value in TEST_DATES if value not in dates]
        raise ValueError(f"missing preregistered test dates: {missing}")
    first_test_code = dates.index(TEST_DATES[0])
    purge_boundary = first_test_code - PURGE_SESSIONS
    if purge_boundary <= 0:
        raise ValueError("insufficient training dates before purge")
    date_codes = np.asarray(matrix["date_codes"], dtype=np.int32)
    train = np.flatnonzero(date_codes < purge_boundary)
    test_codes = np.asarray([dates.index(value) for value in TEST_DATES], dtype=np.int32)
    test = np.flatnonzero(np.isin(date_codes, test_codes))
    actual_test_dates = tuple(dates[index] for index in np.unique(date_codes[test]))
    if actual_test_dates != TEST_DATES:
        raise ValueError(f"test date contract mismatch: {actual_test_dates}")
    return train, test, {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_date_count": int(len(np.unique(date_codes[train]))),
        "test_date_count": int(len(np.unique(date_codes[test]))),
        "train_start_signal_date": dates[int(np.min(date_codes[train]))],
        "train_end_signal_date": dates[int(np.max(date_codes[train]))],
        "purged_signal_date_start": dates[purge_boundary],
        "purged_signal_date_end": dates[first_test_code - 1],
        "test_signal_dates": list(actual_test_dates),
    }


def audit_test_window_relation_coverage(graph_dataset_root: Path) -> dict[str, Any]:
    """Audit only the preregistered test window, not unrelated old exclusions."""

    manifest = json.loads(
        (Path(graph_dataset_root) / "manifest.json").read_text(encoding="utf-8")
    )
    rows = {
        str(item["signal_date"]): {
            "relation_stock_coverage_ratio": float(
                item.get("relation_stock_coverage_ratio") or 0.0
            ),
            "stock_count": int(item.get("stock_count") or 0),
        }
        for item in manifest["snapshots"]
        if str(item["signal_date"]) in TEST_DATES
    }
    missing = [date for date in TEST_DATES if date not in rows]
    below = [
        date
        for date in TEST_DATES
        if date in rows and rows[date]["relation_stock_coverage_ratio"] < 0.95
    ]
    return {
        "dates": {date: rows[date] for date in TEST_DATES if date in rows},
        "missing_dates": missing,
        "below_95pct_dates": below,
        "passed": not missing and not below and len(rows) == len(TEST_DATES),
    }


def _sanitize(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if np.any(np.isinf(result)):
        result = result.copy()
        result[np.isinf(result)] = np.nan
    return result


def build_model_variants(matrix: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], dict[str, Any]]:
    price = _sanitize(matrix["price_matrix"])
    flow = _sanitize(matrix["flow_matrix"])
    date_codes = np.asarray(matrix["date_codes"], dtype=np.int32)
    symbol_codes = np.asarray(matrix["symbol_codes"], dtype=np.int32)
    contract = state_feature_contract(matrix)
    flow_names = tuple(contract["flow_names"])
    current_indices = tuple(contract["current_indices"])
    state_indices = tuple(contract["state_indices"])
    selected = flow[:, state_indices]
    selected_names = tuple(flow_names[index] for index in state_indices)
    if selected_names != STATE_FIELDS:
        raise ValueError("v14 state fields differ from frozen v13 contract")
    date_count = len(matrix["date_values"])
    symbol_count = len(matrix["symbol_values"])
    regime = causal_regime_schedule(
        flow=flow,
        flow_names=flow_names,
        date_codes=date_codes,
        date_count=date_count,
    )
    adaptive, state_names = build_graph_state_features(
        selected=selected,
        selected_names=selected_names,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        date_count=date_count,
        symbol_count=symbol_count,
        regime=regime,
        adaptive=True,
    )
    fixed, fixed_names = build_graph_state_features(
        selected=selected,
        selected_names=selected_names,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        date_count=date_count,
        symbol_count=symbol_count,
        regime=regime,
        adaptive=False,
    )
    lookup = np.full((date_count, symbol_count), -1, dtype=np.int32)
    lookup[date_codes, symbol_codes] = np.arange(len(flow), dtype=np.int32)
    lagged_selected = np.full_like(selected, np.nan)
    valid = date_codes >= 5
    source_rows = lookup[date_codes[valid] - 5, symbol_codes[valid]]
    found = source_rows >= 0
    destinations = np.flatnonzero(valid)[found]
    lagged_selected[destinations] = selected[source_rows[found]]
    lagged, lagged_names = build_graph_state_features(
        selected=lagged_selected,
        selected_names=selected_names,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        date_count=date_count,
        symbol_count=symbol_count,
        regime=regime,
        adaptive=True,
    )
    shuffled_selected = topology_shuffle_state_inputs(
        selected=selected,
        selected_names=selected_names,
        date_codes=date_codes,
        seed=RANDOM_SEED,
    )
    shuffled, shuffled_names = build_graph_state_features(
        selected=shuffled_selected,
        selected_names=selected_names,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        date_count=date_count,
        symbol_count=symbol_count,
        regime=regime,
        adaptive=True,
    )
    if not (state_names == fixed_names == lagged_names == shuffled_names):
        raise ValueError("v14 state/control feature contract mismatch")
    current = flow[:, current_indices]
    base = np.column_stack([price, current]).astype(np.float32)
    price_names = tuple(f"price::{value}" for value in matrix["price_names"])
    current_names = tuple(f"flow::{flow_names[index]}" for index in current_indices)
    base_names = price_names + current_names
    state_feature_names = base_names + tuple(state_names)
    return (
        {
            PRICE_MODEL: price,
            CURRENT_MODEL: base,
            FIXED_MODEL: np.column_stack([base, fixed]).astype(np.float32),
            ADAPTIVE_MODEL: np.column_stack([base, adaptive]).astype(np.float32),
            LAG5_MODEL: np.column_stack([base, lagged]).astype(np.float32),
            SHUFFLED_MODEL: np.column_stack([base, shuffled]).astype(np.float32),
        },
        {
            PRICE_MODEL: price_names,
            CURRENT_MODEL: base_names,
            FIXED_MODEL: state_feature_names,
            ADAPTIVE_MODEL: state_feature_names,
            LAG5_MODEL: state_feature_names,
            SHUFFLED_MODEL: state_feature_names,
        },
        {
            "current_feature_count": len(current_indices),
            "state_input_feature_count": len(state_indices),
            "state_output_feature_count": int(adaptive.shape[1]),
            "regime_gain": {
                "mean": float(np.nanmean(regime["gain"])),
                "minimum": float(np.nanmin(regime["gain"])),
                "maximum": float(np.nanmax(regime["gain"])),
                "sign_flip_date_count": int(np.sum(regime["sign_flip"] > 0.5)),
                "high_shock_date_count": int(np.sum(regime["shock"] >= 0.5)),
            },
        },
    )


def fit_lockbox(
    *, matrix: Mapping[str, Any], train: np.ndarray, test: np.ndarray, thread_count: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    variants, feature_names, state_diagnostics = build_model_variants(matrix)
    targets = np.asarray(matrix["targets"], dtype=np.float32)
    date_codes = np.asarray(matrix["date_codes"], dtype=np.int32)
    weights = date_balanced_weights(date_codes, train)
    caps = residual_caps(targets[train])
    raw_predictions: dict[str, np.ndarray] = {}
    top_features: dict[str, list[dict[str, Any]]] = {}
    fit_seconds: dict[str, float] = {}
    for model_name in MODEL_NAMES:
        raw, top, elapsed = fit_predict_multioutput(
            features=variants[model_name],
            targets=targets,
            train=train,
            test=test,
            weights=weights,
            feature_names=feature_names[model_name],
            thread_count=thread_count,
        )
        raw_predictions[model_name] = raw
        top_features[model_name] = top
        fit_seconds[model_name] = elapsed
        _progress(
            "v14_lockbox_model_fit",
            model=model_name,
            fit_seconds=elapsed,
            at_utc=utc_now(),
        )
        gc.collect()
    price_prediction = raw_predictions[PRICE_MODEL]
    predictions = {PRICE_MODEL: price_prediction}
    for model_name in MODEL_NAMES[1:]:
        predictions[model_name] = capped_residual_prediction(
            price_prediction, raw_predictions[model_name], caps
        )
    return predictions, {
        "fit_seconds": fit_seconds,
        "top_features": top_features,
        "state": state_diagnostics,
        "residual_caps": caps.tolist(),
    }


def evaluate_predictions(
    *,
    actual: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    test_date_codes: np.ndarray,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    unique_dates = np.unique(test_date_codes)
    for target_index, target_name in enumerate(TARGET_NAMES):
        loss_target = target_name.startswith("loss_")
        models: dict[str, Any] = {}
        for model_name in MODEL_NAMES:
            prediction = predictions[model_name][:, target_index]
            models[model_name] = {
                **regression_metrics(actual[:, target_index], prediction),
                **stock_cross_sectional_metrics(
                    date_codes=test_date_codes,
                    target=actual[:, target_index],
                    prediction=prediction,
                    loss_target=loss_target,
                ),
            }
        current_mae = models[CURRENT_MODEL]["mae"]
        for model_name in MODEL_NAMES:
            models[model_name]["relative_mae_improvement_vs_current_pct"] = (
                (current_mae - models[model_name]["mae"]) / current_mae * 100.0
            )
        daily_improvements: list[float] = []
        for date_code in unique_dates:
            rows = np.flatnonzero(test_date_codes == date_code)
            current_error = np.mean(
                np.abs(predictions[CURRENT_MODEL][rows, target_index] - actual[rows, target_index])
            )
            adaptive_error = np.mean(
                np.abs(predictions[ADAPTIVE_MODEL][rows, target_index] - actual[rows, target_index])
            )
            daily_improvements.append(float(current_error - adaptive_error))
        results[target_name] = {
            "rows": int(len(actual)),
            "models": models,
            "daily_mae_improvement_vs_current": daily_improvements,
            "positive_daily_mae_improvement_count": int(
                np.sum(np.asarray(daily_improvements) > 0.0)
            ),
            "evaluated_date_count": int(len(unique_dates)),
        }
    return results


def summarize_gate(
    *, targets: Mapping[str, Any], data_checks: Mapping[str, bool]
) -> dict[str, Any]:
    primary: dict[str, Any] = {}
    for target_name in PRIMARY_TARGETS:
        models = targets[target_name]["models"]
        adaptive = models[ADAPTIVE_MODEL]
        checks = {
            "adaptive_mae_beats_current": adaptive["mae"] < models[CURRENT_MODEL]["mae"],
            "adaptive_mae_beats_topology_shuffle": adaptive["mae"] < models[SHUFFLED_MODEL]["mae"],
            "adaptive_mae_beats_fixed_state": adaptive["mae"] < models[FIXED_MODEL]["mae"],
            "adaptive_mae_beats_lag5": adaptive["mae"] < models[LAG5_MODEL]["mae"],
            "adaptive_mean_daily_rank_ic_beats_current": adaptive["mean_daily_rank_ic"] > models[CURRENT_MODEL]["mean_daily_rank_ic"],
            "adaptive_mean_daily_rank_ic_beats_topology_shuffle": adaptive["mean_daily_rank_ic"] > models[SHUFFLED_MODEL]["mean_daily_rank_ic"],
            "adaptive_economic_basket_beats_current": adaptive["economic_basket_value"] > models[CURRENT_MODEL]["economic_basket_value"],
            "adaptive_economic_basket_beats_topology_shuffle": adaptive["economic_basket_value"] > models[SHUFFLED_MODEL]["economic_basket_value"],
            "positive_daily_mae_improvement_at_least_6_of_11": targets[target_name]["positive_daily_mae_improvement_count"] >= 6,
        }
        primary[target_name] = {
            "checks": checks,
            "passed": all(checks.values()),
        }
    all_data = all(bool(value) for value in data_checks.values())
    passed = all_data and all(item["passed"] for item in primary.values())
    return {
        "status": (
            "V14_FORWARD_AVOIDANCE_PRELIMINARY_PASS"
            if passed
            else "V14_FORWARD_AVOIDANCE_FAIL"
        ),
        "passed": passed,
        "fixed_before_results": True,
        "data_checks": dict(data_checks),
        "primary_targets": primary,
        "deployment_activation": False,
        "bf16_or_nvfp4_activation": False,
        "minimum_clean_future_dates_before_deployment_claim": 60,
        "observed_test_dates": len(TEST_DATES),
    }


def _write_predictions(
    path: Path,
    *,
    actual: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    date_codes: np.ndarray,
    symbol_codes: np.ndarray,
) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        actual=actual,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        **predictions,
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "v14_forward_avoidance_receipt.json"
    if receipt_path.exists() and not args.replace:
        raise FileExistsError(f"receipt already exists: {receipt_path}")
    frozen_hashes = verify_frozen_inputs(args)
    started_at = utc_now()
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "RUNNING",
            "stage": "extended_event_cube",
            "started_at_utc": started_at,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
    )
    event_path = output_root / "v14_extended_flow_event_cube.sqlite3"
    event_receipt_path = output_root / "v14_extended_flow_event_cube_receipt.json"
    source = open_union_source(
        base_database=Path(args.base_database),
        incremental_database=Path(args.incremental_database),
        repaired_flow_cache=Path(args.repaired_flow_cache),
    )
    try:
        source_audit = union_source_audit(source)
        if event_path.exists() and event_receipt_path.exists() and not args.replace:
            event_receipt = json.loads(event_receipt_path.read_text(encoding="utf-8"))
            if sha256_file(event_path) != event_receipt.get("sha256"):
                raise ValueError("existing v14 event cube hash mismatch")
        else:
            if event_path.exists() or event_receipt_path.exists():
                if not args.replace:
                    raise FileExistsError("partial v14 event cube exists")
                event_path.unlink(missing_ok=True)
                event_receipt_path.unlink(missing_ok=True)
            event_receipt = build_event_cube(
                source=source,
                metadata=load_metadata(Path(args.etfradar_root)),
                family_registry_path=Path(args.family_registry),
                output_path=event_path,
                start_date=None,
                end_date=TEST_DATES[-1],
            )
            prefix = prefix_identity_audit(
                old_event_cube=Path(args.old_event_cube),
                new_event_cube=event_path,
            )
            event_receipt = {
                **event_receipt,
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "union_source_audit": source_audit,
                "prefix_identity": prefix,
            }
            write_json_atomic(event_receipt_path, event_receipt)
        prefix = event_receipt.get("prefix_identity")
        if not isinstance(prefix, dict) or not prefix.get("passed"):
            raise ValueError("v14 event cube lacks a passing prefix identity receipt")
        write_json_atomic(
            output_root / "run_state.json",
            {
                "status": "RUNNING",
                "stage": "stock_matrix",
                "started_at_utc": started_at,
                "event_cube_sha256": event_receipt["sha256"],
                "updated_at_utc": utc_now(),
            },
        )
        with readonly_connection(event_path) as event:
            matrix = build_stock_matrix_from_sources(
                event=event,
                source=source,
                graph_dataset_root=Path(args.graph_dataset_root),
                progress=lambda payload: print(json.dumps(dict(payload), sort_keys=True), flush=True),
            )
        train, test, split = split_indices(matrix)
        relation_coverage = audit_test_window_relation_coverage(
            Path(args.graph_dataset_root)
        )
        data_checks = {
            "historical_prefix_exact": bool(prefix["passed"]),
            "timing_violation_count_zero": int(matrix["timing_violation_count"]) == 0,
            "exact_11_test_dates": split["test_date_count"] == len(TEST_DATES),
            "complete_12_target_test_rows": bool(
                np.all(np.isfinite(np.asarray(matrix["targets"])[test]))
            ),
            "relation_stock_coverage_at_least_95pct": bool(
                relation_coverage["passed"]
            ),
            "event_cube_signal_end_exact": event_receipt.get("signal_end") == TEST_DATES[-1],
        }
        if not all(data_checks.values()):
            raise ValueError(f"v14 data validity gate failed before fitting: {data_checks}")
        write_json_atomic(
            output_root / "run_state.json",
            {
                "status": "RUNNING",
                "stage": "frozen_models",
                "started_at_utc": started_at,
                "split": split,
                "updated_at_utc": utc_now(),
            },
        )
        predictions, fit_diagnostics = fit_lockbox(
            matrix=matrix,
            train=train,
            test=test,
            thread_count=int(args.thread_count),
        )
    finally:
        source.close()

    actual = np.asarray(matrix["targets"], dtype=np.float32)[test]
    test_date_codes_original = np.asarray(matrix["date_codes"], dtype=np.int32)[test]
    unique_codes = np.unique(test_date_codes_original)
    recode = {int(value): index for index, value in enumerate(unique_codes)}
    test_date_codes = np.asarray(
        [recode[int(value)] for value in test_date_codes_original], dtype=np.int16
    )
    test_symbol_codes = np.asarray(matrix["symbol_codes"], dtype=np.int32)[test]
    targets = evaluate_predictions(
        actual=actual,
        predictions=predictions,
        test_date_codes=test_date_codes,
    )
    gate = summarize_gate(targets=targets, data_checks=data_checks)
    predictions_path = output_root / "v14_forward_avoidance_predictions.npz"
    _write_predictions(
        predictions_path,
        actual=actual,
        predictions=predictions,
        date_codes=test_date_codes,
        symbol_codes=test_symbol_codes,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started_at,
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_sha256": sha256_file(Path(__file__)),
        "frozen_input_hashes": frozen_hashes,
        "event_cube": event_receipt,
        "graph_manifest_sha256": matrix["source_manifest_sha256"],
        "split": split,
        "scope": {
            "matrix_signal_date_start": matrix["date_values"][0],
            "matrix_signal_date_end": matrix["date_values"][-1],
            "matrix_signal_date_count": len(matrix["date_values"]),
            "matrix_stock_rows": len(matrix["targets"]),
            "matrix_stock_symbols": len(matrix["symbol_values"]),
            "matrix_clusters": len(matrix["clusters"]),
            "test_rows": int(len(test)),
            "test_dates": list(TEST_DATES),
            "target_count": len(TARGET_NAMES),
            "primary_targets": list(PRIMARY_TARGETS),
            "secondary_avoidance_targets": list(SECONDARY_AVOIDANCE_TARGETS),
            "no_row_or_symbol_sampling": True,
            "audit": matrix["audit"],
            "excluded": matrix["excluded"],
            "test_window_relation_coverage": relation_coverage,
        },
        "catboost": {
            "version": CATBOOST_VERSION,
            "parameters": CATBOOST_PARAMETERS,
            "thread_count": int(args.thread_count),
            "gpu_used": False,
        },
        "fit_diagnostics": fit_diagnostics,
        "targets": targets,
        "gate": gate,
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
        },
        "implementation_validity": {
            "historical_prefix_exact": True,
            "causal_forward_only_state": True,
            "twenty_session_purge": True,
            "date_balanced": True,
            "absolute_common_flow_date_centered": False,
            "table_48_breadth_used": False,
            "current_holdings_backfilled_historically": False,
            "new_fmp_features_used": False,
            "existing_v11_v12_v13_outputs_modified": False,
        },
        "limitations": [
            "only 11 forward signal dates are evaluated, below the preregistered 60-date deployment threshold",
            "the Massive repair is historical-window captured and PIT reconstructed, not an as-observed archive",
            "the graph snapshots and their labels existed before this run although their 11 dates were not used by v11-v13 evaluation",
            "a PASS is preliminary and cannot activate trading, deployment, BF16 training, or NVFP4 conversion",
        ],
    }
    write_json_atomic(receipt_path, receipt)
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "COMPLETE",
            "gate_status": gate["status"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "completed_at_utc": utc_now(),
        },
    )
    return receipt_path, receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    result.add_argument(
        "--incremental-database", type=Path, default=DEFAULT_INCREMENTAL_DATABASE
    )
    result.add_argument(
        "--repaired-flow-cache", type=Path, default=DEFAULT_REPAIRED_FLOW_CACHE
    )
    result.add_argument("--old-event-cube", type=Path, default=DEFAULT_OLD_EVENT_CUBE)
    result.add_argument("--family-registry", type=Path, default=DEFAULT_FAMILY_REGISTRY)
    result.add_argument("--etfradar-root", type=Path, default=DEFAULT_ETFRADAR_ROOT)
    result.add_argument(
        "--graph-dataset-root", type=Path, default=DEFAULT_GRAPH_DATASET_ROOT
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--thread-count", type=int, default=10)
    result.add_argument("--replace", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    path, receipt = run(args)
    print(
        json.dumps(
            {
                "path": str(path),
                "gate_status": receipt["gate"]["status"],
                "preregistration_sha256": receipt["preregistration_sha256"],
                "test_dates": receipt["scope"]["test_dates"],
                "test_rows": receipt["scope"]["test_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["gate"]["passed"] else 3
