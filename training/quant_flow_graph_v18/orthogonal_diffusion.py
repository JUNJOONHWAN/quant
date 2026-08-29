"""Prequential stock-level ETF-Flow diffusion/divergence residual gate.

The sealed v16 full-ETF model improved point forecast error, but its feature
importance was dominated by slow global ETF states.  v17 then showed that a
date-level confidence switch could not turn that state into a tradable basket
or avoidance edge.  This successor keeps absolute common Flow in a separate
Drift baseline and asks whether the stock-specific graph contribution contains
an orthogonal, prequential edge.

For each sealed v16 OOS stock row, the graph candidate minus the global-only
candidate is the model-implied Diffusion contribution.  Within each signal
date and target, that contribution is residualized against the price forecast,
the aggregate-v12 Flow increment, and the v16 global-Flow increment.  The
residual is the preregistered Flow-price divergence input.  No realized target
is used in feature construction.

One fixed-capacity CatBoost multi-output residual model is trained only on
earlier completed OOS years, with the final 20 sessions of the latest
calibration year purged.  The primary graph candidate and global-only, lagged,
ETF-axis-shuffled, and date-shuffled controls have identical capacity.  A pass
can authorize only a future prospective shadow lockbox, never deployment or
trading.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "quant.etf_flow_v18.orthogonal_diffusion.v1"
PREREGISTRATION_SCHEMA_VERSION = (
    "quant.etf_flow_v18.orthogonal_diffusion_preregistration.v1"
)
DEFAULT_V16_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v16/"
    "full_etf_identity_latent_walk_forward"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v18/"
    "orthogonal_diffusion_prequential"
)

CALIBRATION_YEAR = 2021
EVALUATION_YEARS = (2022, 2023, 2024, 2025, 2026)
ALL_YEARS = (CALIBRATION_YEAR, *EVALUATION_YEARS)
PURGE_SESSIONS = 20
RANDOM_SEED = 20260829

TARGET_NAMES = (
    "return_5d_pct",
    "upside_5d_pct",
    "loss_5d_pct",
    "benchmark_excess_return_5d_pct",
    "benchmark_upside_capture_5d_pct",
    "benchmark_downside_defense_5d_pct",
    "return_20d_pct",
    "upside_20d_pct",
    "loss_20d_pct",
    "benchmark_excess_return_20d_pct",
    "benchmark_upside_capture_20d_pct",
    "benchmark_downside_defense_20d_pct",
)
CORE_RISK_TARGETS = {
    "loss_5d_pct",
    "loss_20d_pct",
    "benchmark_downside_defense_5d_pct",
    "benchmark_downside_defense_20d_pct",
}

PRICE_MODEL = "price_only"
V12_MODEL = "v12_current_flow"
BASE_MODEL = "full_etf_global_only"
PRIMARY_CANDIDATE = "full_etf_query"
LAG5_CANDIDATE = "full_etf_query_lag5"
AXIS_SHUFFLE_CANDIDATE = "full_etf_axis_shuffle"
DATE_SHUFFLE_CANDIDATE = "full_etf_date_shuffle"
COMMON_ONLY_CANDIDATE = "common_only"
CANDIDATES = (
    COMMON_ONLY_CANDIDATE,
    PRIMARY_CANDIDATE,
    LAG5_CANDIDATE,
    AXIS_SHUFFLE_CANDIDATE,
    DATE_SHUFFLE_CANDIDATE,
)
CONTROL_CANDIDATES = (
    COMMON_ONLY_CANDIDATE,
    LAG5_CANDIDATE,
    AXIS_SHUFFLE_CANDIDATE,
    DATE_SHUFFLE_CANDIDATE,
)
CRITICAL_CONTROLS = (
    COMMON_ONLY_CANDIDATE,
    AXIS_SHUFFLE_CANDIDATE,
    DATE_SHUFFLE_CANDIDATE,
)
REQUIRED_V16_KEYS = {
    "actual",
    "date_codes",
    PRICE_MODEL,
    V12_MODEL,
    "full_etf_query_raw",
    PRIMARY_CANDIDATE,
    BASE_MODEL,
    LAG5_CANDIDATE,
    AXIS_SHUFFLE_CANDIDATE,
    DATE_SHUFFLE_CANDIDATE,
}

CATBOOST_VERSION = "1.2.10"
CATBOOST_PARAMETERS = {
    "loss_function": "MultiRMSE",
    "eval_metric": "MultiRMSE",
    "iterations": 128,
    "depth": 5,
    "learning_rate": 0.05,
    "l2_leaf_reg": 20.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.80,
    "rsm": 0.65,
    "leaf_estimation_iterations": 1,
    "random_seed": RANDOM_SEED,
    "task_type": "CPU",
    "allow_writing_files": False,
    "verbose": False,
}
RESIDUAL_SHRINKAGE = 0.25
RESIDUAL_CAP_QUANTILE = 0.90
ORTHOGONAL_RIDGE_ALPHA = 1.0
TOP_FEATURE_COUNT = 30
BOOTSTRAP_BLOCK_SESSIONS = 20
BOOTSTRAP_REPLICATIONS = 2_000

REFERENCE_PAPERS = (
    {
        "title": "A Flow-Based Explanation for Return Predictability",
        "url": "https://doi.org/10.1093/rfs/hhs085",
        "implication": (
            "Separate flow-induced demand from fundamentals and test temporary "
            "pressure/reversal rather than only unconditional market direction."
        ),
    },
    {
        "title": "Do ETFs Increase Volatility?",
        "url": "https://doi.org/10.1111/jofi.12727",
        "implication": (
            "ETF arbitrage can propagate non-fundamental shocks to constituents; "
            "stock-level topology and later reversal are therefore explicit gates."
        ),
    },
    {
        "title": "Connected Stocks",
        "url": "https://doi.org/10.1111/jofi.12149",
        "implication": (
            "Shared ownership predicts residual comovement and reversal, motivating "
            "a graph-specific control rather than a market-average Flow test."
        ),
    },
    {
        "title": "Double/debiased machine learning for treatment and structural parameters",
        "url": "https://doi.org/10.1111/ectj.12097",
        "implication": (
            "Orthogonalize the stock-specific Flow increment against price/global "
            "nuisance structure before testing its incremental predictive content."
        ),
    },
)


@dataclass(frozen=True)
class FoldData:
    year: int
    arrays: Mapping[str, np.ndarray]
    common_features: np.ndarray
    common_feature_names: tuple[str, ...]
    input_receipt: Mapping[str, Any]


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def progress(**payload: Any) -> None:
    print(json.dumps(_json_ready(payload), sort_keys=True), flush=True)


def load_fold(
    v16_root: Path, year: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = v16_root / f"fold_{year}.npz"
    json_path = v16_root / f"fold_{year}.json"
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"missing sealed v16 fold {year}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    npz_sha = sha256_file(npz_path)
    if npz_sha != metadata.get("prediction_sha256"):
        raise ValueError(f"v16 fold {year} prediction hash mismatch")
    with np.load(npz_path, allow_pickle=False) as item:
        if set(item.files) != REQUIRED_V16_KEYS:
            raise ValueError(f"v16 fold {year} key mismatch: {sorted(item.files)}")
        arrays = {name: np.asarray(item[name]) for name in item.files}
    actual = arrays["actual"]
    dates = arrays["date_codes"]
    if actual.ndim != 2 or actual.shape[1] != len(TARGET_NAMES):
        raise ValueError(f"v16 fold {year} target shape mismatch: {actual.shape}")
    if dates.shape != (len(actual),):
        raise ValueError(f"v16 fold {year} date shape mismatch")
    for name in REQUIRED_V16_KEYS - {"actual", "date_codes"}:
        if arrays[name].shape != actual.shape:
            raise ValueError(f"v16 fold {year} prediction shape mismatch: {name}")
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"v16 fold {year} non-finite prediction: {name}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(dates)):
        raise ValueError(f"v16 fold {year} non-finite identity arrays")
    return arrays, {
        "npz_path": str(npz_path),
        "npz_sha256": npz_sha,
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "row_count": int(len(actual)),
        "date_count": int(len(np.unique(dates))),
        "prediction_sha256_verified": True,
    }


def _cross_sectional_z(values: np.ndarray, date_codes: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(source, dtype=np.float32)
    for date_code in np.unique(date_codes):
        indices = np.flatnonzero(date_codes == date_code)
        block = source[indices].astype(np.float64)
        mean = np.mean(block, axis=0)
        scale = np.std(block, axis=0)
        scale[scale < 1e-9] = 1.0
        result[indices] = np.clip((block - mean) / scale, -10.0, 10.0)
    return result


def _cross_sectional_rank(values: np.ndarray, date_codes: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(source, dtype=np.float32)
    for date_code in np.unique(date_codes):
        indices = np.flatnonzero(date_codes == date_code)
        block = source[indices]
        count = len(indices)
        if count <= 1:
            continue
        ranks = np.empty_like(block, dtype=np.float32)
        for column in range(block.shape[1]):
            order = np.argsort(block[:, column], kind="mergesort")
            sorted_values = block[order, column]
            start = 0
            while start < count:
                end = start + 1
                while end < count and sorted_values[end] == sorted_values[start]:
                    end += 1
                average_rank = 0.5 * float(start + end - 1)
                ranks[order[start:end], column] = average_rank
                start = end
        result[indices] = 2.0 * ranks / float(count - 1) - 1.0
    return result


def orthogonal_diffusion_delta(
    *,
    candidate: np.ndarray,
    base: np.ndarray,
    price: np.ndarray,
    v12: np.ndarray,
    date_codes: np.ndarray,
    ridge_alpha: float = ORTHOGONAL_RIDGE_ALPHA,
) -> tuple[np.ndarray, dict[str, float]]:
    """Remove date-local price/global nuisance structure without using targets."""

    candidate = np.asarray(candidate, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    price = np.asarray(price, dtype=np.float32)
    v12 = np.asarray(v12, dtype=np.float32)
    delta = candidate - base
    result = np.zeros_like(delta, dtype=np.float32)
    max_abs_mean = 0.0
    max_abs_nuisance_corr = 0.0
    identity = np.eye(3, dtype=np.float64) * float(ridge_alpha)
    for date_code in np.unique(date_codes):
        indices = np.flatnonzero(date_codes == date_code)
        if len(indices) < 4:
            raise ValueError(f"date {date_code} has fewer than four stock rows")
        for target_index in range(delta.shape[1]):
            response = delta[indices, target_index].astype(np.float64)
            response -= np.mean(response)
            nuisance = np.column_stack(
                [
                    price[indices, target_index],
                    v12[indices, target_index] - price[indices, target_index],
                    base[indices, target_index] - price[indices, target_index],
                ]
            ).astype(np.float64)
            nuisance -= np.mean(nuisance, axis=0)
            scale = np.std(nuisance, axis=0)
            scale[scale < 1e-9] = 1.0
            nuisance /= scale
            beta = np.linalg.solve(
                nuisance.T @ nuisance + identity,
                nuisance.T @ response,
            )
            residual = response - nuisance @ beta
            result[indices, target_index] = residual.astype(np.float32)
            max_abs_mean = max(max_abs_mean, abs(float(np.mean(residual))))
            residual_scale = float(np.std(residual))
            if residual_scale > 1e-12:
                correlations = np.abs(
                    (nuisance.T @ residual)
                    / np.maximum(
                        np.linalg.norm(nuisance, axis=0)
                        * np.linalg.norm(residual),
                        1e-12,
                    )
                )
                max_abs_nuisance_corr = max(
                    max_abs_nuisance_corr, float(np.max(correlations))
                )
    return result, {
        "ridge_alpha": float(ridge_alpha),
        "max_abs_date_target_residual_mean": max_abs_mean,
        "max_abs_date_target_nuisance_correlation": max_abs_nuisance_corr,
        "absolute_common_flow_modified": False,
        "only_stock_specific_diffusion_branch_orthogonalized": True,
    }


def common_features(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    price = np.asarray(arrays[PRICE_MODEL], dtype=np.float32)
    v12 = np.asarray(arrays[V12_MODEL], dtype=np.float32)
    base = np.asarray(arrays[BASE_MODEL], dtype=np.float32)
    dates = np.asarray(arrays["date_codes"], dtype=np.int32)
    groups = {
        "price": price,
        "v12": v12,
        "global": base,
        "v12_minus_price": v12 - price,
        "global_minus_price": base - price,
    }
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for group_name, values in groups.items():
        blocks.append(values)
        names.extend(f"common::{group_name}::{target}" for target in TARGET_NAMES)
    for group_name, values in groups.items():
        blocks.append(_cross_sectional_z(values, dates))
        names.extend(
            f"common_z::{group_name}::{target}" for target in TARGET_NAMES
        )
    result = np.column_stack(blocks).astype(np.float32)
    return result, tuple(names)


def candidate_features(
    *, candidate_name: str, arrays: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    base = np.asarray(arrays[BASE_MODEL], dtype=np.float32)
    candidate = (
        base
        if candidate_name == COMMON_ONLY_CANDIDATE
        else np.asarray(arrays[candidate_name], dtype=np.float32)
    )
    price = np.asarray(arrays[PRICE_MODEL], dtype=np.float32)
    v12 = np.asarray(arrays[V12_MODEL], dtype=np.float32)
    dates = np.asarray(arrays["date_codes"], dtype=np.int32)
    orthogonal, audit = orthogonal_diffusion_delta(
        candidate=candidate,
        base=base,
        price=price,
        v12=v12,
        date_codes=dates,
    )
    orthogonal_z = _cross_sectional_z(orthogonal, dates)
    orthogonal_rank = _cross_sectional_rank(orthogonal, dates)
    price_z = _cross_sectional_z(price, dates)
    blocks = (
        orthogonal,
        orthogonal_z,
        orthogonal_rank,
        np.abs(orthogonal_z),
        orthogonal_z * price_z,
    )
    prefixes = (
        "diffusion_residual",
        "diffusion_z",
        "diffusion_rank",
        "diffusion_abs_z",
        "flow_price_divergence",
    )
    names = tuple(
        f"{prefix}::{target}" for prefix in prefixes for target in TARGET_NAMES
    )
    result = np.column_stack(blocks).astype(np.float32)
    audit = {
        **audit,
        "candidate": candidate_name,
        "feature_count": int(result.shape[1]),
        "feature_sha256": array_sha256(result),
        "target_values_used": False,
        "common_only_all_zero": bool(
            candidate_name != COMMON_ONLY_CANDIDATE or np.all(result == 0.0)
        ),
    }
    return result, names, audit


def date_balanced_weights(date_codes: np.ndarray) -> np.ndarray:
    selected = np.asarray(date_codes, dtype=np.int64)
    if not len(selected):
        raise ValueError("cannot weight empty calibration rows")
    _, inverse, counts = np.unique(selected, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / np.sum(weights)
    return weights.astype(np.float32)


def calibration_data(
    *,
    folds: Mapping[int, FoldData],
    candidate_blocks: Mapping[int, np.ndarray],
    test_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    years = [year for year in ALL_YEARS if year < test_year]
    if not years:
        raise ValueError(f"test year {test_year} has no earlier OOS calibration year")
    latest_year = max(years)
    features: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    audit: dict[str, Any] = {
        "source_years": [],
        "purged_from_year": latest_year,
        "purged_date_count": 0,
        "purged_row_count": 0,
    }
    for year in years:
        fold = folds[year]
        fold_dates = np.asarray(fold.arrays["date_codes"], dtype=np.int32)
        mask = np.ones(len(fold_dates), dtype=bool)
        if year == latest_year:
            unique_dates = np.unique(fold_dates)
            if len(unique_dates) <= PURGE_SESSIONS:
                raise ValueError(
                    f"calibration year {year} has only {len(unique_dates)} dates"
                )
            purged_dates = unique_dates[-PURGE_SESSIONS:]
            mask = ~np.isin(fold_dates, purged_dates)
            audit["purged_date_count"] = int(len(purged_dates))
            audit["purged_row_count"] = int(np.sum(~mask))
            audit["latest_used_date_code"] = int(np.max(fold_dates[mask]))
            audit["first_purged_date_code"] = int(np.min(purged_dates))
        block = np.column_stack(
            [fold.common_features, candidate_blocks[year]]
        ).astype(np.float32)
        base = np.asarray(fold.arrays[BASE_MODEL], dtype=np.float32)
        actual = np.asarray(fold.arrays["actual"], dtype=np.float32)
        features.append(block[mask])
        residuals.append((actual - base)[mask])
        dates.append(fold_dates[mask])
        audit["source_years"].append(
            {
                "year": year,
                "available_rows": int(len(mask)),
                "used_rows": int(np.sum(mask)),
                "available_dates": int(len(np.unique(fold_dates))),
                "used_dates": int(len(np.unique(fold_dates[mask]))),
            }
        )
    x = np.concatenate(features)
    y = np.concatenate(residuals)
    date_values = np.concatenate(dates)
    weights = date_balanced_weights(date_values)
    audit.update(
        {
            "calibration_rows": int(len(x)),
            "calibration_dates": int(len(np.unique(date_values))),
            "feature_count": int(x.shape[1]),
            "target_count": int(y.shape[1]),
            "date_balanced_total_weight": True,
            "same_year_adaptation": False,
        }
    )
    return x, y, weights, audit


def residual_caps(calibration_residuals: np.ndarray) -> np.ndarray:
    values = np.asarray(calibration_residuals, dtype=np.float64)
    median = np.median(values, axis=0)
    caps = RESIDUAL_SHRINKAGE * np.quantile(
        np.abs(values - median), RESIDUAL_CAP_QUANTILE, axis=0
    )
    return np.maximum(caps, 1e-6).astype(np.float32)


def apply_residual_adapter(
    *, base: np.ndarray, raw_residual: np.ndarray, caps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    correction = RESIDUAL_SHRINKAGE * np.asarray(raw_residual, dtype=np.float32)
    correction = np.clip(correction, -np.asarray(caps), np.asarray(caps))
    return (
        np.asarray(base, dtype=np.float32) + correction,
        correction.astype(np.float32),
    )


def fit_predict_residual(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    weights: np.ndarray,
    feature_names: Sequence[str],
    thread_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    try:
        import catboost
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("CatBoost is required in the isolated container") from exc
    if catboost.__version__ != CATBOOST_VERSION:
        raise RuntimeError(
            f"CatBoost version mismatch: {catboost.__version__} != {CATBOOST_VERSION}"
        )
    started = time.monotonic()
    model = catboost.CatBoostRegressor(
        **CATBOOST_PARAMETERS, thread_count=int(thread_count)
    )
    model.fit(
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_y, dtype=np.float32),
        sample_weight=np.asarray(weights, dtype=np.float32),
        verbose=False,
    )
    prediction = np.asarray(
        model.predict(np.asarray(test_x, dtype=np.float32)), dtype=np.float32
    )
    if prediction.shape != (len(test_x), len(TARGET_NAMES)):
        raise ValueError(f"unexpected residual prediction shape: {prediction.shape}")
    importance = np.asarray(model.get_feature_importance(), dtype=np.float64)
    order = np.argsort(importance)[::-1][:TOP_FEATURE_COUNT]
    top = [
        {"feature": str(feature_names[index]), "importance": float(importance[index])}
        for index in order
    ]
    elapsed = time.monotonic() - started
    del model
    gc.collect()
    return prediction, top, elapsed


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = target - prediction
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "correlation": _safe_correlation(target, prediction),
        "direction_accuracy": float(np.mean(np.sign(target) == np.sign(prediction))),
    }


def _rank_ic(actual: np.ndarray, prediction: np.ndarray) -> float:
    if np.std(actual) <= 1e-12 or np.std(prediction) <= 1e-12:
        return 0.0
    actual_rank = np.argsort(np.argsort(actual)).astype(np.float64)
    prediction_rank = np.argsort(np.argsort(prediction)).astype(np.float64)
    return _safe_correlation(actual_rank, prediction_rank)


def stock_cross_sectional_metrics(
    *,
    date_codes: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    loss_target: bool,
) -> dict[str, float | int]:
    rank_ics: list[float] = []
    spreads: list[float] = []
    tops: list[float] = []
    bottoms: list[float] = []
    basket_values: list[float] = []
    for date_code in np.unique(date_codes):
        indices = np.flatnonzero(date_codes == date_code)
        if len(indices) < 20:
            continue
        actual = np.asarray(target[indices], dtype=np.float64)
        forecast = np.asarray(prediction[indices], dtype=np.float64)
        rank_ics.append(_rank_ic(actual, forecast))
        basket_count = max(1, len(indices) // 10)
        order = np.argsort(forecast)
        bottom = float(np.mean(actual[order[:basket_count]]))
        top = float(np.mean(actual[order[-basket_count:]]))
        bottoms.append(bottom)
        tops.append(top)
        spreads.append(top - bottom)
        basket_values.append(-bottom if loss_target else top)
    mean_top = float(np.mean(tops)) if tops else math.nan
    mean_bottom = float(np.mean(bottoms)) if bottoms else math.nan
    return {
        "mean_daily_rank_ic": float(np.mean(rank_ics)) if rank_ics else math.nan,
        "positive_daily_rank_ic_ratio": (
            float(np.mean(np.asarray(rank_ics) > 0)) if rank_ics else math.nan
        ),
        "mean_top_minus_bottom_spread": (
            float(np.mean(spreads)) if spreads else math.nan
        ),
        "mean_predicted_top_realized": mean_top,
        "mean_predicted_bottom_realized": mean_bottom,
        "economic_basket_value": -mean_bottom if loss_target else mean_top,
        "economic_basket_p10": (
            float(np.quantile(basket_values, 0.10))
            if basket_values
            else math.nan
        ),
        "evaluated_date_count": int(len(spreads)),
    }


def full_metrics(
    *, date_codes: np.ndarray, actual: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        result[target_name] = {
            **regression_metrics(actual[:, target_index], prediction[:, target_index]),
            **stock_cross_sectional_metrics(
                date_codes=date_codes,
                target=actual[:, target_index],
                prediction=prediction[:, target_index],
                loss_target=target_name.startswith("loss_"),
            ),
        }
    return result


def _daily_gain_series(
    *,
    date_codes: np.ndarray,
    actual: np.ndarray,
    base: np.ndarray,
    prediction: np.ndarray,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    mae_gains: list[float] = []
    basket_gains: list[float] = []
    loss_target = TARGET_NAMES[target_index].startswith("loss_")
    for date_code in np.unique(date_codes):
        indices = np.flatnonzero(date_codes == date_code)
        y = actual[indices, target_index]
        b = base[indices, target_index]
        p = prediction[indices, target_index]
        mae_gains.append(float(np.mean(np.abs(y - b)) - np.mean(np.abs(y - p))))
        count = max(1, len(indices) // 10)
        base_order = np.argsort(b)
        prediction_order = np.argsort(p)
        if loss_target:
            base_basket = -float(np.mean(y[base_order[:count]]))
            prediction_basket = -float(np.mean(y[prediction_order[:count]]))
        else:
            base_basket = float(np.mean(y[base_order[-count:]]))
            prediction_basket = float(np.mean(y[prediction_order[-count:]]))
        basket_gains.append(prediction_basket - base_basket)
    return (
        np.asarray(mae_gains, dtype=np.float64),
        np.asarray(basket_gains, dtype=np.float64),
    )


def moving_block_bootstrap(
    values: np.ndarray,
    *,
    seed: int,
    replications: int = BOOTSTRAP_REPLICATIONS,
    block_sessions: int = BOOTSTRAP_BLOCK_SESSIONS,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("cannot bootstrap an empty series")
    block = min(int(block_sessions), len(values))
    block_count = int(math.ceil(len(values) / block))
    offsets = np.arange(block, dtype=np.int64)
    rng = np.random.default_rng(seed)
    means = np.empty(int(replications), dtype=np.float64)
    for replication in range(int(replications)):
        starts = rng.integers(0, len(values), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        means[replication] = float(np.mean(values[indices.ravel()[: len(values)]]))
    return {
        "observed_mean": float(np.mean(values)),
        "ci_lower_95": float(np.quantile(means, 0.025)),
        "ci_upper_95": float(np.quantile(means, 0.975)),
        "one_sided_probability_nonpositive": float(np.mean(means <= 0.0)),
        "block_sessions": int(block),
        "replications": int(replications),
        "date_count": int(len(values)),
    }


def preregistration(v16_root: Path) -> dict[str, Any]:
    receipt_path = v16_root / "v16_full_etf_identity_receipt.json"
    prereg_path = v16_root / "v16_full_etf_identity_preregistration.json"
    if not receipt_path.exists() or not prereg_path.exists():
        raise FileNotFoundError("sealed v16 receipt or preregistration is missing")
    fold_inputs: dict[str, Any] = {}
    for year in ALL_YEARS:
        fold_inputs[str(year)] = {
            "npz_sha256": sha256_file(v16_root / f"fold_{year}.npz"),
            "json_sha256": sha256_file(v16_root / f"fold_{year}.json"),
        }
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "frozen_before_results": True,
        "purpose": (
            "isolate a stock-specific all-ETF graph Diffusion edge from the "
            "absolute global Drift and price response already learned by v16"
        ),
        "economic_definition": {
            "drift": (
                "sealed v16 full_etf_global_only OOS prediction; absolute common "
                "ETF Flow is preserved and never date-centred"
            ),
            "diffusion": (
                "sealed graph candidate minus global-only candidate for the same "
                "stock/date/target"
            ),
            "divergence": (
                "Diffusion residual after date-local target-free projection on the "
                "price forecast, v12-minus-price increment, and global-minus-price "
                "increment"
            ),
            "convergence": (
                "whether that prior-OOS calibrated divergence improves 5d/20d "
                "forecast, ranking, basket, or downside identification"
            ),
        },
        "timing": {
            "input_predictions": "sealed v16 outer-year OOS predictions only",
            "calibration_year": CALIBRATION_YEAR,
            "evaluation_years": list(EVALUATION_YEARS),
            "training_rule": "each test year uses earlier completed OOS years only",
            "latest_calibration_year_purge_sessions": PURGE_SESSIONS,
            "same_year_adaptation": False,
            "future_target_used_as_feature": False,
            "test_threshold_tuning": False,
        },
        "scope": {
            "all_v16_stock_rows": True,
            "no_date_stock_target_or_etf_sampling": True,
            "targets": list(TARGET_NAMES),
            "primary": PRIMARY_CANDIDATE,
            "baseline": BASE_MODEL,
            "candidates": list(CANDIDATES),
        },
        "features": {
            "common": [
                "price_only_prediction",
                "v12_current_flow_prediction",
                "v16_global_only_prediction",
                "v12_minus_price_prediction",
                "global_minus_price_prediction",
                "date_local_zscores_of_each_common_block",
            ],
            "candidate_specific": [
                "orthogonal_diffusion_residual",
                "date_local_diffusion_zscore",
                "date_local_diffusion_rank",
                "absolute_diffusion_zscore",
                "diffusion_zscore_times_price_zscore",
            ],
            "orthogonal_ridge_alpha": ORTHOGONAL_RIDGE_ALPHA,
            "absolute_common_flow_modified": False,
            "only_stock_specific_diffusion_branch_orthogonalized": True,
            "realized_targets_used": False,
        },
        "meta_target": (
            "stock-level actual target minus sealed v16 global-only OOS prediction"
        ),
        "residual_adapter": {
            "shrinkage": RESIDUAL_SHRINKAGE,
            "cap_quantile": RESIDUAL_CAP_QUANTILE,
            "cap_source": "purged earlier-OOS calibration residuals only",
        },
        "estimator": {
            "library": "catboost",
            "version": CATBOOST_VERSION,
            "parameters": CATBOOST_PARAMETERS,
            "multi_target_single_model": True,
            "date_balanced_sample_weight": True,
            "same_capacity_for_primary_and_controls": True,
            "symbol_identity_feature": False,
        },
        "controls": {
            "common_only_no_diffusion": True,
            "five_session_lag": True,
            "etf_axis_shuffle": True,
            "date_shuffle": True,
            "sealed_always_query": True,
            "same_feature_width_and_estimator_capacity": True,
        },
        "inference": {
            "paired_unit": "signal date",
            "overlap_robust_method": "circular moving-block bootstrap",
            "block_sessions": BOOTSTRAP_BLOCK_SESSIONS,
            "replications": BOOTSTRAP_REPLICATIONS,
            "fixed_seed": RANDOM_SEED,
        },
        "fixed_gate": {
            "forecast_path": {
                "mae_beats_global_targets": 8,
                "rank_ic_beats_global_targets": 8,
                "mae_beats_each_critical_control_targets": 7,
                "mae_beats_lag5_targets": 7,
                "mean_mae_improvement_positive": True,
                "worst_target_mae_degradation_at_most_pct": 0.25,
                "positive_fold_targets": 30,
                "year_2025_and_2026_mean_improvement_nonnegative": True,
                "mae_gain_bootstrap_ci_lower_positive_targets": 4,
            },
            "basket_path": {
                "basket_beats_global_targets": 8,
                "rank_ic_beats_global_targets": 8,
                "basket_beats_each_control_targets": 7,
                "positive_fold_targets": 30,
                "year_2025_and_2026_mean_basket_gain_nonnegative": True,
                "basket_gain_bootstrap_ci_lower_positive_targets": 3,
            },
            "risk_filter_path": {
                "core_targets": sorted(CORE_RISK_TARGETS),
                "basket_beats_global_core_targets": 3,
                "basket_p10_beats_global_core_targets": 3,
                "basket_beats_each_control_core_targets": 3,
            },
        },
        "activation": {
            "historical_result_is_exploratory": True,
            "deployment_forbidden": True,
            "trading_forbidden": True,
            "bf16_training_activation_forbidden": True,
            "nvfp4_conversion_forbidden": True,
            "pass": "ELIGIBLE_FOR_FUTURE_PROSPECTIVE_SHADOW_LOCKBOX_ONLY",
            "fail": "NO_ORTHOGONAL_DIFFUSION_ACTIVATION",
        },
        "frozen_inputs": {
            "source_sha256": sha256_file(Path(__file__)),
            "v16_receipt_sha256": sha256_file(receipt_path),
            "v16_preregistration_sha256": sha256_file(prereg_path),
            "folds": fold_inputs,
        },
        "references": list(REFERENCE_PAPERS),
    }


def _checkpoint_paths(
    output_root: Path, candidate: str, year: int
) -> tuple[Path, Path]:
    stem = f"candidate_{candidate}_fold_{year}"
    return output_root / f"{stem}.npz", output_root / f"{stem}.json"


def load_checkpoint(
    *, output_root: Path, candidate: str, year: int, preregistration_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    npz_path, json_path = _checkpoint_paths(output_root, candidate, year)
    if not npz_path.exists() or not json_path.exists():
        return None
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"checkpoint preregistration mismatch: {candidate} {year}")
    if sha256_file(npz_path) != metadata.get("prediction_sha256"):
        raise ValueError(f"checkpoint hash mismatch: {candidate} {year}")
    with np.load(npz_path, allow_pickle=False) as item:
        arrays = {name: np.asarray(item[name]) for name in item.files}
    if set(arrays) != {"prediction", "raw_residual", "correction", "caps"}:
        raise ValueError(f"checkpoint key mismatch: {candidate} {year}")
    return arrays, metadata


def save_checkpoint(
    *,
    output_root: Path,
    candidate: str,
    year: int,
    preregistration_sha256: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    npz_path, json_path = _checkpoint_paths(output_root, candidate, year)
    write_npz_atomic(npz_path, arrays)
    result = {
        **dict(metadata),
        "candidate": candidate,
        "outer_year": int(year),
        "preregistration_sha256": preregistration_sha256,
        "prediction_path": str(npz_path),
        "prediction_sha256": sha256_file(npz_path),
    }
    write_json_atomic(json_path, result)
    return result


def bootstrap_primary(
    *, date_codes: np.ndarray, actual: np.ndarray, base: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        mae_gain, basket_gain = _daily_gain_series(
            date_codes=date_codes,
            actual=actual,
            base=base,
            prediction=prediction,
            target_index=target_index,
        )
        result[target_name] = {
            "mae_gain": moving_block_bootstrap(
                mae_gain, seed=RANDOM_SEED + target_index * 10
            ),
            "basket_gain": moving_block_bootstrap(
                basket_gain, seed=RANDOM_SEED + target_index * 10 + 1
            ),
        }
    return result


def gate_summary(
    *,
    pooled: Mapping[str, Mapping[str, Any]],
    yearly: Mapping[str, Mapping[str, Mapping[str, Any]]],
    bootstrap: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    primary = pooled[PRIMARY_CANDIDATE]
    base = pooled[BASE_MODEL]
    counters: defaultdict[str, int] = defaultdict(int)
    improvements: list[float] = []
    risk: defaultdict[str, int] = defaultdict(int)
    for target_name in TARGET_NAMES:
        p = primary[target_name]
        b = base[target_name]
        improvement = (b["mae"] - p["mae"]) / b["mae"] * 100.0
        improvements.append(improvement)
        counters["mae_beats_global"] += p["mae"] < b["mae"]
        counters["rank_ic_beats_global"] += (
            p["mean_daily_rank_ic"] > b["mean_daily_rank_ic"]
        )
        counters["basket_beats_global"] += (
            p["economic_basket_value"] > b["economic_basket_value"]
        )
        counters["basket_p10_beats_global"] += (
            p["economic_basket_p10"] > b["economic_basket_p10"]
        )
        counters["mae_bootstrap_ci_positive"] += (
            bootstrap[target_name]["mae_gain"]["ci_lower_95"] > 0.0
        )
        counters["basket_bootstrap_ci_positive"] += (
            bootstrap[target_name]["basket_gain"]["ci_lower_95"] > 0.0
        )
        for control in CONTROL_CANDIDATES:
            c = pooled[control][target_name]
            counters[f"mae_beats_{control}"] += p["mae"] < c["mae"]
            counters[f"basket_beats_{control}"] += (
                p["economic_basket_value"] > c["economic_basket_value"]
            )
        if target_name in CORE_RISK_TARGETS:
            risk["basket_beats_global"] += (
                p["economic_basket_value"] > b["economic_basket_value"]
            )
            risk["basket_p10_beats_global"] += (
                p["economic_basket_p10"] > b["economic_basket_p10"]
            )
            for control in CONTROL_CANDIDATES:
                c = pooled[control][target_name]
                risk[f"basket_beats_{control}"] += (
                    p["economic_basket_value"] > c["economic_basket_value"]
                )

    yearly_mae: dict[str, float] = {}
    yearly_basket: dict[str, float] = {}
    positive_mae_fold_targets = 0
    positive_basket_fold_targets = 0
    fold_target_count = 0
    for year, models in yearly.items():
        mae_values: list[float] = []
        basket_values: list[float] = []
        for target_name in TARGET_NAMES:
            p = models[PRIMARY_CANDIDATE][target_name]
            b = models[BASE_MODEL][target_name]
            mae_value = (b["mae"] - p["mae"]) / b["mae"] * 100.0
            basket_value = p["economic_basket_value"] - b["economic_basket_value"]
            mae_values.append(mae_value)
            basket_values.append(basket_value)
            positive_mae_fold_targets += mae_value > 0.0
            positive_basket_fold_targets += basket_value > 0.0
            fold_target_count += 1
        yearly_mae[year] = float(np.mean(mae_values))
        yearly_basket[year] = float(np.mean(basket_values))

    mean_improvement = float(np.mean(improvements))
    worst_improvement = float(np.min(improvements))
    critical_mae = all(
        counters[f"mae_beats_{control}"] >= 7 for control in CRITICAL_CONTROLS
    )
    all_control_basket = all(
        counters[f"basket_beats_{control}"] >= 7
        for control in CONTROL_CANDIDATES
    )
    all_control_risk = all(
        risk[f"basket_beats_{control}"] >= 3 for control in CONTROL_CANDIDATES
    )
    late_mae = all(yearly_mae.get(str(year), -math.inf) >= 0.0 for year in (2025, 2026))
    late_basket = all(
        yearly_basket.get(str(year), -math.inf) >= 0.0 for year in (2025, 2026)
    )
    forecast_path = (
        counters["mae_beats_global"] >= 8
        and counters["rank_ic_beats_global"] >= 8
        and critical_mae
        and counters[f"mae_beats_{LAG5_CANDIDATE}"] >= 7
        and mean_improvement > 0.0
        and worst_improvement >= -0.25
        and positive_mae_fold_targets >= 30
        and late_mae
        and counters["mae_bootstrap_ci_positive"] >= 4
    )
    basket_path = (
        counters["basket_beats_global"] >= 8
        and counters["rank_ic_beats_global"] >= 8
        and all_control_basket
        and positive_basket_fold_targets >= 30
        and late_basket
        and counters["basket_bootstrap_ci_positive"] >= 3
    )
    risk_filter_path = (
        risk["basket_beats_global"] >= 3
        and risk["basket_p10_beats_global"] >= 3
        and all_control_risk
    )
    passed_paths = [
        name
        for name, passed in (
            ("FORECAST", forecast_path),
            ("BASKET", basket_path),
            ("RISK_FILTER", risk_filter_path),
        )
        if passed
    ]
    return {
        "status": (
            "V18_ORTHOGONAL_DIFFUSION_PASS"
            if passed_paths
            else "V18_ORTHOGONAL_DIFFUSION_FAIL"
        ),
        "passed_paths": passed_paths,
        "fixed_before_results": True,
        "historical_oos_not_fresh_forward_lockbox": True,
        "checks": {
            "forecast_path_pass": forecast_path,
            "basket_path_pass": basket_path,
            "risk_filter_path_pass": risk_filter_path,
            "critical_control_mae_pass": critical_mae,
            "all_control_basket_pass": all_control_basket,
            "all_control_risk_filter_pass": all_control_risk,
            "late_2025_2026_mae_nonnegative": late_mae,
            "late_2025_2026_basket_nonnegative": late_basket,
        },
        "counters": {
            **dict(counters),
            "risk_core": dict(risk),
            "mean_mae_improvement_vs_global_pct": mean_improvement,
            "worst_target_mae_improvement_vs_global_pct": worst_improvement,
            "positive_mae_fold_target_count": int(positive_mae_fold_targets),
            "positive_basket_fold_target_count": int(positive_basket_fold_targets),
            "outer_fold_target_count": int(fold_target_count),
            "yearly_mean_mae_improvement_vs_global_pct": yearly_mae,
            "yearly_mean_basket_gain_vs_global": yearly_basket,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, Mapping[str, Any]]:
    v16_root = Path(args.v16_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "v18_orthogonal_diffusion_receipt.json"
    run_state_path = output_root / "run_state.json"
    prereg_path = output_root / "v18_orthogonal_diffusion_preregistration.json"
    if receipt_path.exists():
        raise FileExistsError(f"receipt already exists: {receipt_path}")

    proposed = preregistration(v16_root)
    if prereg_path.exists():
        existing = json.loads(prereg_path.read_text(encoding="utf-8"))
        if existing != proposed:
            raise ValueError("existing v18 preregistration differs from source/input")
    else:
        write_json_atomic(prereg_path, proposed)
    prereg_sha = sha256_file(prereg_path)
    if args.preregister_only:
        return prereg_path, {"preregistration_sha256": prereg_sha}
    if not args.expected_prereg_sha:
        raise ValueError("--expected-prereg-sha is required for the full run")
    if args.expected_prereg_sha != prereg_sha:
        raise ValueError(
            f"preregistration SHA mismatch: {args.expected_prereg_sha} != {prereg_sha}"
        )

    started_at = utc_now()
    write_json_atomic(
        run_state_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "RUNNING",
            "started_at_utc": started_at,
            "preregistration_sha256": prereg_sha,
        },
    )

    folds: dict[int, FoldData] = {}
    previous_max_date: int | None = None
    for year in ALL_YEARS:
        arrays, input_receipt = load_fold(v16_root, year)
        current_dates = np.asarray(arrays["date_codes"], dtype=np.int64)
        if previous_max_date is not None and int(np.min(current_dates)) <= previous_max_date:
            raise ValueError(f"overlapping or non-chronological v16 folds at {year}")
        previous_max_date = int(np.max(current_dates))
        common, names = common_features(arrays)
        folds[year] = FoldData(
            year=year,
            arrays=arrays,
            common_features=common,
            common_feature_names=names,
            input_receipt=input_receipt,
        )
        progress(
            stage="v18_fold_loaded",
            year=year,
            rows=len(arrays["actual"]),
            dates=len(np.unique(current_dates)),
            common_features=common.shape[1],
        )

    predictions: dict[str, dict[int, np.ndarray]] = {
        candidate: {} for candidate in CANDIDATES
    }
    corrections: dict[str, dict[int, np.ndarray]] = {
        candidate: {} for candidate in CANDIDATES
    }
    candidate_metadata: dict[str, dict[str, Any]] = {
        candidate: {} for candidate in CANDIDATES
    }
    feature_contract: tuple[str, ...] | None = None

    for candidate in CANDIDATES:
        candidate_blocks: dict[int, np.ndarray] = {}
        feature_audits: dict[int, dict[str, Any]] = {}
        candidate_names: tuple[str, ...] | None = None
        for year in ALL_YEARS:
            block, names, audit = candidate_features(
                candidate_name=candidate, arrays=folds[year].arrays
            )
            if candidate_names is None:
                candidate_names = names
            elif candidate_names != names:
                raise ValueError(f"candidate feature-name drift for {candidate}")
            candidate_blocks[year] = block
            feature_audits[year] = audit
        if candidate_names is None:
            raise AssertionError("candidate feature contract not built")
        names = folds[CALIBRATION_YEAR].common_feature_names + candidate_names
        if feature_contract is None:
            feature_contract = names
        elif feature_contract != names:
            raise ValueError("primary/control feature width or names differ")

        for year in EVALUATION_YEARS:
            started = time.monotonic()
            checkpoint = load_checkpoint(
                output_root=output_root,
                candidate=candidate,
                year=year,
                preregistration_sha256=prereg_sha,
            )
            if checkpoint is None:
                train_x, train_y, weights, calibration_audit = calibration_data(
                    folds=folds,
                    candidate_blocks=candidate_blocks,
                    test_year=year,
                )
                test_x = np.column_stack(
                    [folds[year].common_features, candidate_blocks[year]]
                ).astype(np.float32)
                raw_residual, top_features, fit_seconds = fit_predict_residual(
                    train_x=train_x,
                    train_y=train_y,
                    test_x=test_x,
                    weights=weights,
                    feature_names=names,
                    thread_count=int(args.thread_count),
                )
                caps = residual_caps(train_y)
                prediction, correction = apply_residual_adapter(
                    base=folds[year].arrays[BASE_MODEL],
                    raw_residual=raw_residual,
                    caps=caps,
                )
                arrays = {
                    "prediction": prediction.astype(np.float32),
                    "raw_residual": raw_residual.astype(np.float32),
                    "correction": correction.astype(np.float32),
                    "caps": caps.astype(np.float32),
                }
                metadata = save_checkpoint(
                    output_root=output_root,
                    candidate=candidate,
                    year=year,
                    preregistration_sha256=prereg_sha,
                    arrays=arrays,
                    metadata={
                        "schema_version": SCHEMA_VERSION,
                        "input": folds[year].input_receipt,
                        "calibration": calibration_audit,
                        "feature_audit": feature_audits[year],
                        "feature_count": int(len(names)),
                        "feature_names": list(names),
                        "top_features": top_features,
                        "fit_seconds": float(fit_seconds),
                        "elapsed_seconds": float(time.monotonic() - started),
                        "mean_absolute_raw_residual": float(
                            np.mean(np.abs(raw_residual))
                        ),
                        "mean_absolute_correction": float(np.mean(np.abs(correction))),
                        "cap_hit_ratio": float(
                            np.mean(np.abs(correction) >= caps[None, :] - 1e-8)
                        ),
                        "resumed": False,
                    },
                )
                del train_x, train_y, weights, test_x, raw_residual
                gc.collect()
            else:
                arrays, metadata = checkpoint
                metadata = {**metadata, "resumed": True}
            expected_shape = folds[year].arrays["actual"].shape
            if arrays["prediction"].shape != expected_shape:
                raise ValueError(f"checkpoint prediction shape mismatch: {candidate} {year}")
            predictions[candidate][year] = arrays["prediction"].astype(np.float32)
            corrections[candidate][year] = arrays["correction"].astype(np.float32)
            candidate_metadata[candidate][str(year)] = metadata
            progress(
                stage="v18_candidate_fold_complete",
                candidate=candidate,
                year=year,
                mean_absolute_correction=float(np.mean(np.abs(arrays["correction"]))),
                resumed=bool(metadata.get("resumed")),
            )
            write_json_atomic(
                run_state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "RUNNING",
                    "started_at_utc": started_at,
                    "updated_at_utc": utc_now(),
                    "preregistration_sha256": prereg_sha,
                    "completed_candidate": candidate,
                    "completed_outer_year": year,
                },
            )
        del candidate_blocks
        gc.collect()

    yearly_metrics: dict[str, Any] = {}
    pooled_actual: list[np.ndarray] = []
    pooled_dates: list[np.ndarray] = []
    pooled_base: list[np.ndarray] = []
    pooled_sealed_query: list[np.ndarray] = []
    pooled_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    for year in EVALUATION_YEARS:
        arrays = folds[year].arrays
        actual = np.asarray(arrays["actual"], dtype=np.float32)
        dates = np.asarray(arrays["date_codes"], dtype=np.int64)
        base = np.asarray(arrays[BASE_MODEL], dtype=np.float32)
        models: dict[str, Any] = {
            BASE_MODEL: full_metrics(date_codes=dates, actual=actual, prediction=base),
            "sealed_always_full_etf_query": full_metrics(
                date_codes=dates,
                actual=actual,
                prediction=np.asarray(arrays[PRIMARY_CANDIDATE], dtype=np.float32),
            ),
        }
        for candidate in CANDIDATES:
            models[candidate] = full_metrics(
                date_codes=dates,
                actual=actual,
                prediction=predictions[candidate][year],
            )
            pooled_predictions[candidate].append(predictions[candidate][year])
        yearly_metrics[str(year)] = models
        pooled_actual.append(actual)
        pooled_dates.append(dates)
        pooled_base.append(base)
        pooled_sealed_query.append(
            np.asarray(arrays[PRIMARY_CANDIDATE], dtype=np.float32)
        )

    actual_all = np.concatenate(pooled_actual)
    dates_all = np.concatenate(pooled_dates)
    base_all = np.concatenate(pooled_base)
    sealed_query_all = np.concatenate(pooled_sealed_query)
    pooled_metrics: dict[str, Any] = {
        BASE_MODEL: full_metrics(
            date_codes=dates_all, actual=actual_all, prediction=base_all
        ),
        "sealed_always_full_etf_query": full_metrics(
            date_codes=dates_all, actual=actual_all, prediction=sealed_query_all
        ),
    }
    for candidate in CANDIDATES:
        pooled_metrics[candidate] = full_metrics(
            date_codes=dates_all,
            actual=actual_all,
            prediction=np.concatenate(pooled_predictions[candidate]),
        )
    primary_all = np.concatenate(pooled_predictions[PRIMARY_CANDIDATE])
    bootstrap = bootstrap_primary(
        date_codes=dates_all,
        actual=actual_all,
        base=base_all,
        prediction=primary_all,
    )
    gate = gate_summary(
        pooled=pooled_metrics,
        yearly=yearly_metrics,
        bootstrap=bootstrap,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "started_at_utc": started_at,
        "preregistration_sha256": prereg_sha,
        "source_sha256": sha256_file(Path(__file__)),
        "v16_receipt_sha256": sha256_file(
            v16_root / "v16_full_etf_identity_receipt.json"
        ),
        "scope": {
            "calibration_year": CALIBRATION_YEAR,
            "evaluation_years": list(EVALUATION_YEARS),
            "evaluation_rows": int(len(actual_all)),
            "evaluation_dates": int(len(np.unique(dates_all))),
            "targets": list(TARGET_NAMES),
            "candidate_count": len(CANDIDATES),
            "no_sampling": True,
        },
        "timing": {
            "latest_calibration_year_purge_sessions": PURGE_SESSIONS,
            "same_year_adaptation": False,
            "timing_violations": 0,
        },
        "feature_contract": {
            "feature_count": int(len(feature_contract or ())),
            "feature_names": list(feature_contract or ()),
            "target_values_used_as_features": False,
            "absolute_common_flow_date_centered": False,
            "only_diffusion_branch_orthogonalized": True,
        },
        "estimator": {
            "library": "catboost",
            "version": CATBOOST_VERSION,
            "parameters": CATBOOST_PARAMETERS,
            "residual_shrinkage": RESIDUAL_SHRINKAGE,
            "residual_cap_quantile": RESIDUAL_CAP_QUANTILE,
        },
        "candidate_folds": candidate_metadata,
        "pooled_metrics": pooled_metrics,
        "yearly_metrics": yearly_metrics,
        "paired_date_block_bootstrap": bootstrap,
        "gate": gate,
        "implementation_validity": {
            "v16_hashes_verified": True,
            "prequential_prior_oos_only": True,
            "last_20_sessions_purged": True,
            "candidate_capacity_equal": True,
            "candidate_feature_width_equal": True,
            "target_free_feature_construction": True,
            "chronological_nonoverlapping_folds": True,
        },
        "limitations": [
            "The 2021-2026 historical OOS period was inspected by earlier v16/v17 work; v18 is exploratory rather than a fresh prospective lockbox.",
            "The divergence input is model-implied from sealed v16 predictions; raw latent ETF vectors were not refitted or retuned.",
            "Passing authorizes only a future prospective shadow lockbox and does not authorize deployment, trading, BF16 training, or NVFP4 conversion.",
        ],
        "next_activation": (
            "FUTURE_PROSPECTIVE_SHADOW_LOCKBOX_ONLY"
            if gate["passed_paths"]
            else "NO_ORTHOGONAL_DIFFUSION_ACTIVATION"
        ),
        "references": list(REFERENCE_PAPERS),
    }
    write_json_atomic(receipt_path, receipt)
    write_json_atomic(
        run_state_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "completed_at_utc": utc_now(),
            "preregistration_sha256": prereg_sha,
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "gate_status": gate["status"],
        },
    )
    progress(
        stage="v18_complete",
        status=gate["status"],
        passed_paths=gate["passed_paths"],
        receipt_path=str(receipt_path),
        receipt_sha256=sha256_file(receipt_path),
    )
    return receipt_path, receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--v16-root", default=str(DEFAULT_V16_ROOT))
    result.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    result.add_argument("--thread-count", type=int, default=10)
    result.add_argument("--preregister-only", action="store_true")
    result.add_argument("--expected-prereg-sha")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    path, payload = run(args)
    print(str(path), flush=True)
    if args.preregister_only:
        print(payload["preregistration_sha256"], flush=True)
        return 0
    return 0 if payload["gate"]["passed_paths"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
