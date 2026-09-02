"""Preregistered causal graph-state canary for ETF Flow.

The input signals are the already audited v11 point-in-time graph channels:
market Drift, direct ETF-to-stock pressure, and indirect all-ETF cluster
Diffusion.  This canary does not fit a Transformer or a neural graph network.
It asks a narrower question first: does causal, regime-adaptive state memory
improve the stable v12 current-Flow-plus-structure residual?

The state filter is deterministic and strictly forward-only.  A market-wide
shock schedule increases the observation gain after Drift or breadth changes,
so old state is forgotten faster.  Stale source coverage reduces that gain.
Fixed-memory, five-session-lag and within-date topology-shuffle controls use
the same features and CatBoost capacity.  Existing v11/v12 receipts are read
only and never overwritten.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.contracts import (
    DEFAULT_SOURCE_DATABASE,
    TIMING_CONTRACT,
)
from training.quant_flow_graph_v11_r2.phase_a import (
    readonly_connection,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from training.quant_flow_graph_v11_r2.phase_b_stock import (
    DEFAULT_GRAPH_DATASET_ROOT,
    DEFAULT_PHASE_A_ROOT,
    OUTER_YEARS,
    PURGE_SESSIONS,
    TARGET_NAMES,
    build_stock_matrix_from_sources,
    fold_indices,
    lag_flow_by_symbol,
    regression_metrics,
    stock_cross_sectional_metrics,
)
from training.quant_flow_graph_v12.channel_ablation import (
    DEFAULT_OUTPUT_ROOT as SOURCE_CHANNEL_OUTPUT_ROOT,
    classify_feature_channels,
)
from training.quant_flow_graph_v12.residual_canary import (
    CATBOOST_PARAMETERS,
    CATBOOST_VERSION,
    PRICE_MODEL,
    capped_residual_prediction,
    date_balanced_weights,
    fit_predict_multioutput,
    residual_caps,
)


SCHEMA_VERSION = "quant.etf_flow_v13.adaptive_graph_state_canary.v1"
PREREGISTRATION_SCHEMA_VERSION = (
    "quant.etf_flow_v13.adaptive_graph_state_canary_preregistration.v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v13/"
    "adaptive_graph_state_canary"
)

SOURCE_RESIDUAL_RECEIPT_SHA256 = (
    "951e120a8a34df0ebf3d970fac0acb80d1deb7142249879e6e9c44036cc70598"
)
SOURCE_CHANNEL_RECEIPT_SHA256 = (
    "af4134882b770301cd58698bb24423ab691ab6199b92f292f4ccf5066c02100d"
)
SOURCE_CHANNEL_RECEIPT_NAME = "v12_channel_ablation_receipt.json"
RANDOM_SEED = 20260828

SOURCE_PRICE_MODEL = PRICE_MODEL
SOURCE_CURRENT_MODEL = "source_full_current_no_rolling"
SOURCE_ORIGINAL_MODEL = "source_original_full"
FIXED_MODEL = "fixed_20_session_graph_state"
PRIMARY_MODEL = "adaptive_graph_state"
LAG5_MODEL = "adaptive_graph_state_lag5"
SHUFFLED_MODEL = "adaptive_graph_state_topology_shuffle"
MODEL_NAMES = (
    SOURCE_PRICE_MODEL,
    SOURCE_CURRENT_MODEL,
    SOURCE_ORIGINAL_MODEL,
    FIXED_MODEL,
    PRIMARY_MODEL,
    LAG5_MODEL,
    SHUFFLED_MODEL,
)

GLOBAL_STATE_FIELDS = (
    "drift_rate_pct",
    "independent_breadth_net",
    "diffusion_coverage",
    "stale_ratio",
)
STOCK_GRAPH_STATE_FIELDS = (
    "direct_clean_rate_net",
    "direct_family_breadth_net",
    "indirect_cluster_flow_rate_pct",
    "indirect_cluster_breadth_net",
    "direct_minus_indirect_rate",
    "indirect_minus_market_drift_rate",
    "direct_indirect_sign_convergence",
)
STATE_FIELDS = GLOBAL_STATE_FIELDS + STOCK_GRAPH_STATE_FIELDS

FIXED_GAIN = 2.0 / 21.0
MIN_GAIN = 0.05
MAX_GAIN = 0.85
BASE_GAIN = 0.10
SHOCK_GAIN = 0.55
SIGN_FLIP_GAIN = 0.20
SHOCK_NORMALIZER = 4.0
SCALE_DECAY = 0.95
MIN_STALE_TRUST = 0.25


def _progress(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def _sanitize(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if np.any(np.isinf(result)):
        result = result.copy()
        result[np.isinf(result)] = np.nan
    return result


def _first_finite_by_date(
    values: np.ndarray, date_codes: np.ndarray, date_count: int
) -> np.ndarray:
    """Collapse a repeated global field without using future dates."""

    result = np.full(date_count, np.nan, dtype=np.float64)
    for row, code in zip(np.asarray(values, dtype=np.float64), date_codes):
        code = int(code)
        if math.isfinite(row) and not math.isfinite(result[code]):
            result[code] = row
    return result


def causal_regime_schedule(
    *,
    flow: np.ndarray,
    flow_names: Sequence[str],
    date_codes: np.ndarray,
    date_count: int,
) -> dict[str, np.ndarray]:
    """Create a causal observation-gain schedule from market Flow only."""

    name_to_index = {str(name): index for index, name in enumerate(flow_names)}
    required = ("drift_rate_pct", "independent_breadth_net", "stale_ratio")
    missing = [name for name in required if name not in name_to_index]
    if missing:
        raise ValueError(f"missing regime fields: {missing}")
    drift = _first_finite_by_date(
        flow[:, name_to_index["drift_rate_pct"]], date_codes, date_count
    )
    breadth = _first_finite_by_date(
        flow[:, name_to_index["independent_breadth_net"]], date_codes, date_count
    )
    stale = _first_finite_by_date(
        flow[:, name_to_index["stale_ratio"]], date_codes, date_count
    )
    gain = np.full(date_count, BASE_GAIN, dtype=np.float64)
    shock = np.zeros(date_count, dtype=np.float64)
    sign_flip = np.zeros(date_count, dtype=np.float64)
    stale_trust = np.ones(date_count, dtype=np.float64)
    if date_count:
        gain[0] = 1.0
    previous = np.asarray([np.nan, np.nan], dtype=np.float64)
    scales = np.asarray([1e-3, 1e-3], dtype=np.float64)
    for date_index in range(date_count):
        current = np.asarray([drift[date_index], breadth[date_index]], dtype=np.float64)
        stale_value = stale[date_index]
        trust = 1.0 - stale_value if math.isfinite(stale_value) else MIN_STALE_TRUST
        trust = float(np.clip(trust, MIN_STALE_TRUST, 1.0))
        stale_trust[date_index] = trust
        if date_index == 0:
            for feature_index, value in enumerate(current):
                if math.isfinite(value):
                    previous[feature_index] = value
                    scales[feature_index] = max(abs(value) * 0.10, 1e-3)
            continue
        normalized_changes: list[float] = []
        flipped = False
        for feature_index, value in enumerate(current):
            prior = previous[feature_index]
            if not math.isfinite(value) or not math.isfinite(prior):
                if math.isfinite(value):
                    previous[feature_index] = value
                continue
            delta = abs(value - prior)
            normalized_changes.append(delta / max(scales[feature_index], 1e-6))
            flipped = flipped or (value * prior < 0.0)
            scales[feature_index] = (
                SCALE_DECAY * scales[feature_index]
                + (1.0 - SCALE_DECAY) * delta
            )
            previous[feature_index] = value
        shock_value = (
            min(max(normalized_changes) / SHOCK_NORMALIZER, 1.0)
            if normalized_changes
            else 0.0
        )
        shock[date_index] = shock_value
        sign_flip[date_index] = float(flipped)
        raw_gain = BASE_GAIN + SHOCK_GAIN * shock_value
        if flipped:
            raw_gain += SIGN_FLIP_GAIN
        gain[date_index] = float(
            np.clip(raw_gain * trust, MIN_GAIN, MAX_GAIN)
        )
    return {
        "gain": gain.astype(np.float32),
        "shock": shock.astype(np.float32),
        "sign_flip": sign_flip.astype(np.float32),
        "stale_trust": stale_trust.astype(np.float32),
    }


def _date_state(
    values: np.ndarray,
    date_codes: np.ndarray,
    date_count: int,
    gains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    series = _first_finite_by_date(values, date_codes, date_count)
    states = np.full(date_count, np.nan, dtype=np.float64)
    innovations = np.full(date_count, np.nan, dtype=np.float64)
    state = math.nan
    for date_index, value in enumerate(series):
        if not math.isfinite(value):
            states[date_index] = state
            continue
        if not math.isfinite(state):
            state = value
            innovation = 0.0
        else:
            innovation = value - state
            state = state + float(gains[date_index]) * innovation
        states[date_index] = state
        innovations[date_index] = innovation
    return states[date_codes], innovations[date_codes]


def _symbol_state(
    values: np.ndarray,
    date_codes: np.ndarray,
    symbol_codes: np.ndarray,
    symbol_count: int,
    gains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.full(symbol_count, np.nan, dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    innovation_output = np.full(len(values), np.nan, dtype=np.float64)
    order = np.lexsort((symbol_codes, date_codes))
    for row_index in order:
        value = float(values[row_index])
        symbol = int(symbol_codes[row_index])
        state = states[symbol]
        if not math.isfinite(value):
            output[row_index] = state
            continue
        if not math.isfinite(state):
            state = value
            innovation = 0.0
        else:
            innovation = value - state
            state = state + float(gains[int(date_codes[row_index])]) * innovation
        states[symbol] = state
        output[row_index] = state
        innovation_output[row_index] = innovation
    return output, innovation_output


def build_graph_state_features(
    *,
    selected: np.ndarray,
    selected_names: Sequence[str],
    date_codes: np.ndarray,
    symbol_codes: np.ndarray,
    date_count: int,
    symbol_count: int,
    regime: Mapping[str, np.ndarray],
    adaptive: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build causal state and innovation features for graph-derived signals."""

    names = tuple(str(name) for name in selected_names)
    if names != STATE_FIELDS:
        raise ValueError(f"unexpected state field contract: {names}")
    gains = (
        np.asarray(regime["gain"], dtype=np.float32)
        if adaptive
        else np.full(date_count, FIXED_GAIN, dtype=np.float32)
    )
    if date_count:
        gains[0] = 1.0
    state_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    global_set = set(GLOBAL_STATE_FIELDS)
    for feature_index, name in enumerate(names):
        values = np.asarray(selected[:, feature_index], dtype=np.float64)
        if name in global_set:
            state, innovation = _date_state(
                values, date_codes, date_count, gains
            )
        else:
            state, innovation = _symbol_state(
                values, date_codes, symbol_codes, symbol_count, gains
            )
        state_columns.extend((state, innovation))
        feature_names.extend((f"state::{name}", f"innovation::{name}"))
    for regime_name in ("gain", "shock", "sign_flip", "stale_trust"):
        state_columns.append(np.asarray(regime[regime_name])[date_codes])
        feature_names.append(f"regime::{regime_name}")
    return (
        np.column_stack(state_columns).astype(np.float32),
        tuple(feature_names),
    )


def topology_shuffle_state_inputs(
    *,
    selected: np.ndarray,
    selected_names: Sequence[str],
    date_codes: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Shuffle only stock-specific graph channels within each date."""

    result = np.asarray(selected, dtype=np.float32).copy()
    stock_indices = [
        index
        for index, name in enumerate(selected_names)
        if name in STOCK_GRAPH_STATE_FIELDS
    ]
    rng = np.random.default_rng(seed)
    for date_code in np.unique(date_codes):
        rows = np.flatnonzero(date_codes == date_code)
        if len(rows) < 2:
            continue
        permuted = rng.permutation(rows)
        result[np.ix_(rows, stock_indices)] = selected[
            np.ix_(permuted, stock_indices)
        ]
    return result


def preregistration() -> dict[str, Any]:
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "frozen_before_results": True,
        "purpose": (
            "test whether causal regime-adaptive memory over audited PIT graph "
            "Drift/direct/Diffusion channels improves the stable v12 current-Flow "
            "plus structure residual"
        ),
        "timing_contract": TIMING_CONTRACT,
        "scope": {
            "targets": list(TARGET_NAMES),
            "outer_years": list(OUTER_YEARS),
            "purge_sessions": PURGE_SESSIONS,
            "no_row_or_symbol_sampling": True,
        },
        "source_receipts": {
            "v12_residual_sha256": SOURCE_RESIDUAL_RECEIPT_SHA256,
            "v12_channel_ablation_sha256": SOURCE_CHANNEL_RECEIPT_SHA256,
        },
        "architecture": {
            "body": "deterministic causal graph-state filter plus CatBoost residual",
            "transformer_used": False,
            "generative_denoising_diffusion_used": False,
            "learned_gnn_used": False,
            "state_fields": list(STATE_FIELDS),
            "market_shock_fields": [
                "drift_rate_pct",
                "independent_breadth_net",
                "stale_ratio",
            ],
            "adaptive_gain": {
                "base": BASE_GAIN,
                "shock_weight": SHOCK_GAIN,
                "sign_flip_weight": SIGN_FLIP_GAIN,
                "minimum": MIN_GAIN,
                "maximum": MAX_GAIN,
                "stale_trust_floor": MIN_STALE_TRUST,
                "causal_scale_decay": SCALE_DECAY,
            },
            "fixed_memory_control_gain": FIXED_GAIN,
        },
        "fixed_estimator": {
            "library": "catboost",
            "version": CATBOOST_VERSION,
            "parameters": CATBOOST_PARAMETERS,
            "date_balanced_total_weight": True,
            "v12_capped_residual_adapter_reused_unchanged": True,
        },
        "controls": {
            "source_price_only": True,
            "source_full_current_no_rolling": True,
            "source_original_fixed_rolling": True,
            "fixed_20_session_state": True,
            "five_session_lag_before_state": True,
            "within_date_stock_topology_shuffle_before_state": True,
            "same_state_feature_count_and_model_capacity": True,
        },
        "gate_thresholds": {
            "forecast": {
                "mae_beats_source_current_targets": 8,
                "mean_mae_improvement_vs_source_current_positive": True,
                "worst_target_degradation_vs_source_current_at_most_pct": 0.5,
                "mae_beats_fixed_state_targets": 8,
                "mae_beats_lag5_targets": 8,
                "mae_beats_topology_shuffle_targets": 8,
                "positive_outer_fold_targets_vs_source_current": 36,
                "2025_and_2026_mean_improvement_nonnegative": True,
            },
            "basket": {
                "rank_ic_beats_source_current_targets": 8,
                "economic_basket_beats_source_current_targets": 8,
            },
            "avoidance": {
                "core_target_count": 4,
                "mae_beats_source_current": 3,
                "mae_beats_topology_shuffle": 3,
            },
        },
        "interpretation": {
            "historical_oos_is_not_new_lockbox": True,
            "pass_only_activates_learned_graph_ssm_canary": True,
            "future_forward_window_required_before_bf16_or_nvfp4": True,
            "no_trading_or_deployment_activation": True,
        },
        "prohibitions": {
            "post_result_retuning": True,
            "date_center_absolute_flow": True,
            "table_48_breadth": True,
            "historical_holdings_imputation": True,
            "current_holdings_used_as_historical": True,
            "new_fmp_features_in_this_canary": True,
            "gpu_use": True,
            "live_service_change": True,
        },
    }


def _source_receipt_paths() -> tuple[Path, Path]:
    return (
        SOURCE_CHANNEL_OUTPUT_ROOT.parent
        / "catboost_residual_canary"
        / "v12_residual_canary_receipt.json",
        SOURCE_CHANNEL_OUTPUT_ROOT / SOURCE_CHANNEL_RECEIPT_NAME,
    )


def verify_source_receipts() -> dict[str, str]:
    residual_path, channel_path = _source_receipt_paths()
    actual = {
        "v12_residual": sha256_file(residual_path),
        "v12_channel_ablation": sha256_file(channel_path),
    }
    expected = {
        "v12_residual": SOURCE_RESIDUAL_RECEIPT_SHA256,
        "v12_channel_ablation": SOURCE_CHANNEL_RECEIPT_SHA256,
    }
    if actual != expected:
        raise ValueError(f"source receipt hash mismatch: {actual} != {expected}")
    return actual


def _source_fold(year: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = SOURCE_CHANNEL_OUTPUT_ROOT / f"fold_{year}.npz"
    json_path = SOURCE_CHANNEL_OUTPUT_ROOT / f"fold_{year}.json"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if sha256_file(npz_path) != metadata.get("npz_sha256"):
        raise ValueError(f"source channel fold {year} hash mismatch")
    with np.load(npz_path, allow_pickle=False) as item:
        arrays = {
            "actual": item["actual"].copy(),
            "date_codes": item["date_codes"].copy(),
            SOURCE_PRICE_MODEL: item[PRICE_MODEL].copy(),
            SOURCE_CURRENT_MODEL: item["full_current_no_rolling"].copy(),
            SOURCE_ORIGINAL_MODEL: item["original_full"].copy(),
        }
    return arrays, metadata


def _fold_paths(output_root: Path, year: int) -> tuple[Path, Path]:
    return output_root / f"fold_{year}.npz", output_root / f"fold_{year}.json"


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_checkpoint(
    output_root: Path, year: int, preregistration_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    npz_path, json_path = _fold_paths(output_root, year)
    if not npz_path.exists() or not json_path.exists():
        return None
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"v13 fold {year} preregistration mismatch")
    if sha256_file(npz_path) != metadata.get("npz_sha256"):
        raise ValueError(f"v13 fold {year} checkpoint hash mismatch")
    with np.load(npz_path, allow_pickle=False) as item:
        arrays = {name: item[name].copy() for name in item.files}
    if set(arrays) != {"actual", "date_codes", *MODEL_NAMES}:
        raise ValueError(f"v13 fold {year} fields mismatch")
    return arrays, metadata


def _save_checkpoint(
    *,
    output_root: Path,
    year: int,
    preregistration_sha256: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    npz_path, json_path = _fold_paths(output_root, year)
    _write_npz_atomic(npz_path, arrays)
    payload = {
        **dict(metadata),
        "schema_version": "quant.etf_flow_v13.adaptive_graph_state_fold.v1",
        "preregistration_sha256": preregistration_sha256,
        "npz_sha256": sha256_file(npz_path),
        "generated_at_utc": utc_now(),
    }
    write_json_atomic(json_path, payload)
    return payload


def _fold_metrics(
    actual: np.ndarray, predictions: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    return {
        target_name: {
            model_name: regression_metrics(
                actual[:, target_index], prediction[:, target_index]
            )
            for model_name, prediction in predictions.items()
        }
        for target_index, target_name in enumerate(TARGET_NAMES)
    }


def _pooled_metrics(
    *,
    actual_parts: Sequence[np.ndarray],
    date_parts: Sequence[np.ndarray],
    prediction_parts: Mapping[str, Sequence[np.ndarray]],
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual = np.concatenate(actual_parts)
    dates = np.concatenate(date_parts)
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        pooled: dict[str, Any] = {}
        loss_target = target_name.startswith("loss_")
        for model_name in MODEL_NAMES:
            prediction = np.concatenate(prediction_parts[model_name])[:, target_index]
            pooled[model_name] = {
                **regression_metrics(actual[:, target_index], prediction),
                **stock_cross_sectional_metrics(
                    date_codes=dates,
                    target=actual[:, target_index],
                    prediction=prediction,
                    loss_target=loss_target,
                ),
            }
        source_mae = pooled[SOURCE_CURRENT_MODEL]["mae"]
        for model_name in MODEL_NAMES:
            pooled[model_name]["relative_mae_improvement_vs_source_current_pct"] = (
                (source_mae - pooled[model_name]["mae"]) / source_mae * 100.0
            )
        result[target_name] = {
            "rows": int(len(actual)),
            "pooled": pooled,
            "folds": [
                {
                    "outer_year": fold["outer_year"],
                    "models": fold["target_metrics"][target_name],
                }
                for fold in folds
            ],
        }
    return result


def state_feature_contract(matrix: Mapping[str, Any]) -> dict[str, Any]:
    flow_names = tuple(str(name) for name in matrix["flow_names"])
    missing = [name for name in STATE_FIELDS if name not in flow_names]
    if missing:
        raise ValueError(f"missing graph-state fields: {missing}")
    groups = classify_feature_channels(flow_names)
    return {
        "flow_names": flow_names,
        "current_indices": groups["full_current_no_rolling"],
        "state_indices": tuple(flow_names.index(name) for name in STATE_FIELDS),
    }


def evaluate(
    *,
    matrix: Mapping[str, Any],
    output_root: Path,
    preregistration_sha256: str,
    thread_count: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    price = _sanitize(matrix["price_matrix"])
    flow = _sanitize(matrix["flow_matrix"])
    targets = np.asarray(matrix["targets"], dtype=np.float32)
    date_codes = np.asarray(matrix["date_codes"], dtype=np.int32)
    symbol_codes = np.asarray(matrix["symbol_codes"], dtype=np.int32)
    contract = state_feature_contract(matrix)
    flow_names = contract["flow_names"]
    current_indices = contract["current_indices"]
    state_indices = contract["state_indices"]
    selected = flow[:, state_indices]
    selected_names = tuple(flow_names[index] for index in state_indices)
    date_count = len(matrix["date_values"])
    symbol_count = len(matrix["symbol_values"])
    regime = causal_regime_schedule(
        flow=flow,
        flow_names=flow_names,
        date_codes=date_codes,
        date_count=date_count,
    )
    adaptive_state, state_names = build_graph_state_features(
        selected=selected,
        selected_names=selected_names,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        date_count=date_count,
        symbol_count=symbol_count,
        regime=regime,
        adaptive=True,
    )
    fixed_state, fixed_names = build_graph_state_features(
        selected=selected,
        selected_names=selected_names,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        date_count=date_count,
        symbol_count=symbol_count,
        regime=regime,
        adaptive=False,
    )
    if fixed_names != state_names:
        raise ValueError("fixed/adaptive state feature contract mismatch")
    lagged_selected = lag_flow_by_symbol(
        selected,
        date_codes,
        symbol_codes,
        date_count,
        symbol_count,
        5,
    )
    lagged_state, lagged_names = build_graph_state_features(
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
    shuffled_state, shuffled_names = build_graph_state_features(
        selected=shuffled_selected,
        selected_names=selected_names,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        date_count=date_count,
        symbol_count=symbol_count,
        regime=regime,
        adaptive=True,
    )
    if not (state_names == lagged_names == shuffled_names):
        raise ValueError("state control feature contract mismatch")

    current = flow[:, current_indices]
    base = np.column_stack([price, current]).astype(np.float32)
    variants = {
        FIXED_MODEL: np.column_stack([base, fixed_state]).astype(np.float32),
        PRIMARY_MODEL: np.column_stack([base, adaptive_state]).astype(np.float32),
        LAG5_MODEL: np.column_stack([base, lagged_state]).astype(np.float32),
        SHUFFLED_MODEL: np.column_stack([base, shuffled_state]).astype(np.float32),
    }
    price_names = tuple(f"price::{name}" for name in matrix["price_names"])
    current_names = tuple(f"flow::{flow_names[index]}" for index in current_indices)
    feature_names = price_names + current_names + state_names

    actual_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    prediction_parts: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    fold_records: list[dict[str, Any]] = []
    for year in OUTER_YEARS:
        train, test = fold_indices(matrix, year)
        if len(train) < 50_000 or len(test) < 10_000:
            continue
        source_arrays, source_metadata = _source_fold(year)
        if not np.array_equal(source_arrays["actual"], targets[test]):
            raise ValueError(f"source fold {year} target mismatch")
        if not np.array_equal(source_arrays["date_codes"], date_codes[test]):
            raise ValueError(f"source fold {year} date mismatch")
        checkpoint = _load_checkpoint(output_root, year, preregistration_sha256)
        if checkpoint is not None:
            arrays, metadata = checkpoint
            if not np.array_equal(arrays["actual"], targets[test]):
                raise ValueError(f"v13 fold {year} target mismatch")
            predictions = {name: arrays[name] for name in MODEL_NAMES}
            metadata = {**metadata, "resumed": True}
        else:
            weights = date_balanced_weights(date_codes, train)
            caps = residual_caps(targets[train])
            predictions = {
                SOURCE_PRICE_MODEL: source_arrays[SOURCE_PRICE_MODEL],
                SOURCE_CURRENT_MODEL: source_arrays[SOURCE_CURRENT_MODEL],
                SOURCE_ORIGINAL_MODEL: source_arrays[SOURCE_ORIGINAL_MODEL],
            }
            fit_seconds: dict[str, float] = {}
            top_features: dict[str, list[dict[str, Any]]] = {}
            for model_name, features in variants.items():
                raw, top, elapsed = fit_predict_multioutput(
                    features=features,
                    targets=targets,
                    train=train,
                    test=test,
                    weights=weights,
                    feature_names=feature_names,
                    thread_count=thread_count,
                )
                predictions[model_name] = capped_residual_prediction(
                    source_arrays[SOURCE_PRICE_MODEL], raw, caps
                )
                fit_seconds[model_name] = elapsed
                top_features[model_name] = top
                if progress:
                    progress(
                        {
                            "stage": "v13_graph_state_model_fit",
                            "outer_year": year,
                            "variant": model_name,
                            "fit_seconds": elapsed,
                            "at_utc": utc_now(),
                        }
                    )
                gc.collect()
            train_codes = date_codes[train]
            test_codes = date_codes[test]
            metadata = {
                "outer_year": int(year),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_date_count": int(len(np.unique(train_codes))),
                "test_date_count": int(len(np.unique(test_codes))),
                "train_end_signal_date": matrix["date_values"][int(np.max(train_codes))],
                "test_start_signal_date": matrix["date_values"][int(np.min(test_codes))],
                "test_end_signal_date": matrix["date_values"][int(np.max(test_codes))],
                "source_channel_fold_sha256": source_metadata["npz_sha256"],
                "fit_seconds": fit_seconds,
                "top_features": top_features,
                "target_metrics": _fold_metrics(targets[test], predictions),
                "resumed": False,
            }
            arrays = {
                "actual": targets[test],
                "date_codes": date_codes[test],
                **predictions,
            }
            metadata = _save_checkpoint(
                output_root=output_root,
                year=year,
                preregistration_sha256=preregistration_sha256,
                arrays=arrays,
                metadata=metadata,
            )
        actual_parts.append(targets[test])
        date_parts.append(date_codes[test])
        for model_name, prediction in predictions.items():
            prediction_parts[model_name].append(prediction)
        fold_records.append(metadata)
        write_json_atomic(
            output_root / "run_state.json",
            {
                "status": "RUNNING",
                "stage": "outer_folds",
                "completed_outer_years": [item["outer_year"] for item in fold_records],
                "current_outer_year": year,
                "updated_at_utc": utc_now(),
            },
        )
        if progress:
            progress(
                {
                    "stage": "v13_graph_state_outer_fold_complete",
                    "outer_year": year,
                    "test_rows": len(test),
                    "resumed": bool(metadata.get("resumed")),
                    "at_utc": utc_now(),
                }
            )
    if not actual_parts:
        raise ValueError("no eligible v13 graph-state folds")
    diagnostics = {
        "current_feature_count": len(current_indices),
        "state_input_feature_count": len(state_indices),
        "state_output_feature_count": adaptive_state.shape[1],
        "model_feature_count": variants[PRIMARY_MODEL].shape[1],
        "regime_gain": {
            "mean": float(np.nanmean(regime["gain"])),
            "minimum": float(np.nanmin(regime["gain"])),
            "maximum": float(np.nanmax(regime["gain"])),
            "sign_flip_date_count": int(np.sum(regime["sign_flip"] > 0.5)),
            "high_shock_date_count": int(np.sum(regime["shock"] >= 0.5)),
        },
        "state_fields": list(selected_names),
    }
    return (
        _pooled_metrics(
            actual_parts=actual_parts,
            date_parts=date_parts,
            prediction_parts=prediction_parts,
            folds=fold_records,
        ),
        fold_records,
        diagnostics,
    )


def summarize_gate(targets: Mapping[str, Any]) -> dict[str, Any]:
    counters = defaultdict(int)
    core = defaultdict(int)
    improvements: list[float] = []
    positive_fold_targets = 0
    outer_fold_targets = 0
    yearly_improvements: defaultdict[int, list[float]] = defaultdict(list)
    core_names = {
        "loss_5d_pct",
        "loss_20d_pct",
        "benchmark_downside_defense_5d_pct",
        "benchmark_downside_defense_20d_pct",
    }
    for target_name, target in targets.items():
        pooled = target["pooled"]
        primary = pooled[PRIMARY_MODEL]
        source = pooled[SOURCE_CURRENT_MODEL]
        fixed = pooled[FIXED_MODEL]
        lagged = pooled[LAG5_MODEL]
        shuffled = pooled[SHUFFLED_MODEL]
        improvement = (source["mae"] - primary["mae"]) / source["mae"] * 100.0
        improvements.append(improvement)
        counters["mae_beats_source_current"] += primary["mae"] < source["mae"]
        counters["mae_beats_source_original"] += (
            primary["mae"] < pooled[SOURCE_ORIGINAL_MODEL]["mae"]
        )
        counters["mae_beats_fixed_state"] += primary["mae"] < fixed["mae"]
        counters["mae_beats_lag5"] += primary["mae"] < lagged["mae"]
        counters["mae_beats_topology_shuffle"] += primary["mae"] < shuffled["mae"]
        counters["rank_ic_beats_source_current"] += (
            primary["mean_daily_rank_ic"] > source["mean_daily_rank_ic"]
        )
        counters["economic_basket_beats_source_current"] += (
            primary["economic_basket_value"] > source["economic_basket_value"]
        )
        if target_name in core_names:
            core["mae_beats_source_current"] += primary["mae"] < source["mae"]
            core["mae_beats_topology_shuffle"] += primary["mae"] < shuffled["mae"]
        for fold in target["folds"]:
            outer_fold_targets += 1
            primary_mae = fold["models"][PRIMARY_MODEL]["mae"]
            source_mae = fold["models"][SOURCE_CURRENT_MODEL]["mae"]
            positive_fold_targets += primary_mae < source_mae
            yearly_improvements[int(fold["outer_year"])].append(
                (source_mae - primary_mae) / source_mae * 100.0
            )
    mean_improvement = float(np.mean(improvements))
    worst_improvement = float(np.min(improvements))
    yearly_means = {
        str(year): float(np.mean(values))
        for year, values in sorted(yearly_improvements.items())
    }
    late_regime_pass = all(yearly_means.get(str(year), -math.inf) >= 0.0 for year in (2025, 2026))
    forecast_pass = (
        counters["mae_beats_source_current"] >= 8
        and mean_improvement > 0.0
        and worst_improvement >= -0.5
        and counters["mae_beats_fixed_state"] >= 8
        and counters["mae_beats_lag5"] >= 8
        and counters["mae_beats_topology_shuffle"] >= 8
        and positive_fold_targets >= 36
        and late_regime_pass
    )
    basket_pass = (
        counters["rank_ic_beats_source_current"] >= 8
        and counters["economic_basket_beats_source_current"] >= 8
        and counters["mae_beats_topology_shuffle"] >= 8
        and late_regime_pass
    )
    avoidance_pass = (
        core["mae_beats_source_current"] >= 3
        and core["mae_beats_topology_shuffle"] >= 3
        and late_regime_pass
    )
    passed_paths = [
        name
        for name, passed in (
            ("FORECAST", forecast_pass),
            ("BASKET", basket_pass),
            ("AVOIDANCE", avoidance_pass),
        )
        if passed
    ]
    return {
        "status": (
            "V13_ADAPTIVE_GRAPH_STATE_PASS"
            if passed_paths
            else "V13_ADAPTIVE_GRAPH_STATE_FAIL"
        ),
        "passed_paths": passed_paths,
        "fixed_before_results": True,
        "historical_oos_not_clean_lockbox": True,
        "checks": {
            "forecast_path_pass": forecast_pass,
            "basket_path_pass": basket_pass,
            "avoidance_path_pass": avoidance_pass,
            "mae_beats_source_current_8_of_12": counters[
                "mae_beats_source_current"
            ] >= 8,
            "mean_improvement_vs_source_current_positive": mean_improvement > 0.0,
            "worst_degradation_vs_source_current_at_most_0_5pct": (
                worst_improvement >= -0.5
            ),
            "mae_beats_fixed_state_8_of_12": counters["mae_beats_fixed_state"] >= 8,
            "mae_beats_lag5_8_of_12": counters["mae_beats_lag5"] >= 8,
            "mae_beats_topology_shuffle_8_of_12": counters[
                "mae_beats_topology_shuffle"
            ] >= 8,
            "positive_half_outer_fold_targets": positive_fold_targets >= 36,
            "2025_and_2026_mean_improvement_nonnegative": late_regime_pass,
        },
        "counters": {
            **dict(counters),
            "avoidance_core": dict(core),
            "mean_relative_mae_improvement_vs_source_current_pct": mean_improvement,
            "worst_relative_mae_improvement_vs_source_current_pct": worst_improvement,
            "positive_outer_fold_target_count": int(positive_fold_targets),
            "outer_fold_target_count": int(outer_fold_targets),
            "yearly_mean_improvement_vs_source_current_pct": yearly_means,
            "target_count": len(targets),
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    preregistration_path = output_root / "v13_adaptive_graph_state_preregistration.json"
    frozen = preregistration()
    if preregistration_path.exists():
        existing = json.loads(preregistration_path.read_text(encoding="utf-8"))
        if existing != frozen:
            raise ValueError("existing v13 preregistration does not match source")
    else:
        write_json_atomic(preregistration_path, frozen)
    preregistration_sha256 = sha256_file(preregistration_path)
    source_receipts = verify_source_receipts()
    if args.preregister_only:
        return preregistration_path, {
            "status": "PREREGISTERED",
            "preregistration_sha256": preregistration_sha256,
            "source_receipts": source_receipts,
        }

    receipt_path = output_root / "v13_adaptive_graph_state_receipt.json"
    if receipt_path.exists() and not args.replace:
        raise FileExistsError(f"receipt already exists: {receipt_path}")
    phase_a_root = Path(args.phase_a_root)
    event_path = phase_a_root / "v11_r2_flow_event_cube.sqlite3"
    hypothesis_path = phase_a_root / "v11_r2_drift_diffusion_hypothesis_registry.json"
    if not event_path.exists() or not hypothesis_path.exists():
        raise FileNotFoundError("Phase A event cube or hypothesis registry is missing")
    started_at = utc_now()
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "RUNNING",
            "stage": "stock_matrix",
            "started_at_utc": started_at,
            "preregistration_sha256": preregistration_sha256,
        },
    )
    with readonly_connection(event_path) as event, readonly_connection(
        Path(args.source_database)
    ) as source:
        matrix = build_stock_matrix_from_sources(
            event=event,
            source=source,
            graph_dataset_root=Path(args.graph_dataset_root),
            progress=_progress,
        )
    targets, folds, diagnostics = evaluate(
        matrix=matrix,
        output_root=output_root,
        preregistration_sha256=preregistration_sha256,
        thread_count=args.thread_count,
        progress=_progress,
    )
    gate = summarize_gate(targets)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started_at,
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        "preregistration_sha256": preregistration_sha256,
        "source_sha256": sha256_file(Path(__file__)),
        "source_receipts": source_receipts,
        "source_event_cube_sha256": sha256_file(event_path),
        "source_graph_manifest_sha256": matrix["source_manifest_sha256"],
        "hypothesis_registry_sha256": sha256_file(hypothesis_path),
        "catboost": {
            "version": CATBOOST_VERSION,
            "parameters": CATBOOST_PARAMETERS,
            "thread_count": int(args.thread_count),
            "gpu_used": False,
        },
        "scope": {
            "signal_date_start": matrix["date_values"][0],
            "signal_date_end": matrix["date_values"][-1],
            "signal_date_count": len(matrix["date_values"]),
            "stock_symbol_count": len(matrix["symbol_values"]),
            "stock_row_count": len(matrix["targets"]),
            "cluster_count": len(matrix["clusters"]),
            "target_count": len(TARGET_NAMES),
            "audit": matrix["audit"],
            "excluded": matrix["excluded"],
            "timing_violation_count": matrix["timing_violation_count"],
            "no_row_or_symbol_sampling": True,
        },
        "state_diagnostics": diagnostics,
        "folds": folds,
        "targets": targets,
        "gate": gate,
        "next_activation": (
            "ELIGIBLE_FOR_LEARNED_GRAPH_SSM_CANARY_NOT_DEPLOYMENT"
            if gate["passed_paths"]
            else "KEEP_V12_CURRENT_STRUCTURE_RESIDUAL_NO_GRAPH_SSM_ACTIVATION"
        ),
        "implementation_validity": {
            "causal_forward_only_state": True,
            "date_balanced": True,
            "price_flow_lag_contract_preserved": True,
            "absolute_common_flow_date_centered": False,
            "table_48_breadth_used": False,
            "current_holdings_backfilled_historically": False,
            "new_fmp_features_used": False,
            "existing_v11_v12_outputs_modified": False,
        },
        "limitations": [
            "2021-2026 historical OOS informed design and is not a clean future lockbox",
            "the canary filters audited graph-propagated features; it is not yet a learned sparse GNN",
            "FMP Ultimate historical disclosures begin in 2019Q3 for SPY/QQQ and do not extend 2018",
            "a PASS only permits a learned BF16 graph-SSM canary and future forward test",
        ],
    }
    write_json_atomic(receipt_path, receipt)
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "COMPLETE",
            "gate_status": gate["status"],
            "passed_paths": gate["passed_paths"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "completed_at_utc": utc_now(),
        },
    )
    return receipt_path, receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase-a-root", type=Path, default=DEFAULT_PHASE_A_ROOT)
    result.add_argument(
        "--graph-dataset-root", type=Path, default=DEFAULT_GRAPH_DATASET_ROOT
    )
    result.add_argument("--source-database", type=Path, default=DEFAULT_SOURCE_DATABASE)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--thread-count", type=int, default=10)
    result.add_argument("--preregister-only", action="store_true")
    result.add_argument("--replace", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    path, payload = run(args)
    if payload.get("status") == "PREREGISTERED":
        print(json.dumps({"path": str(path), **payload}, indent=2, sort_keys=True))
        return 0
    summary = {
        "status": payload["gate"]["status"],
        "path": str(path),
        "sha256": sha256_file(path),
        "scope": payload["scope"],
        "gate": payload["gate"],
        "next_activation": payload["next_activation"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload["gate"]["status"].endswith("PASS") else 3


if __name__ == "__main__":
    raise SystemExit(main())
