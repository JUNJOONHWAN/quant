"""Prequential ETF-Flow confidence and avoidance gate.

v16 found a small full-ETF latent-state MAE improvement, but it did not beat
the five-session Flow control or economic-basket controls consistently.  This
module asks a narrower, operational question: can information already present
in a candidate Flow forecast identify, using only prior completed OOS years,
when to use that forecast instead of the sealed v12 forecast and when to
abstain?

The meta learner never sees in-sample base-model predictions.  Each outer year
is predicted from earlier v16 OOS folds only, and the final 20 signal sessions
of the most recent calibration year are purged.  The primary full-ETF query and
all controls receive the same deterministic model and feature contract.
"""

from __future__ import annotations

import argparse
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
from sklearn.ensemble import HistGradientBoostingRegressor


SCHEMA_VERSION = "quant.etf_flow_v17.prequential_gate.v2"
PREREGISTRATION_SCHEMA_VERSION = (
    "quant.etf_flow_v17.prequential_gate_preregistration.v2"
)
DEFAULT_V16_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v16/"
    "full_etf_identity_latent_walk_forward"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v17/"
    "prequential_confidence_gate_r2"
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
CORE_AVOIDANCE_TARGETS = {
    "loss_5d_pct",
    "loss_20d_pct",
    "benchmark_downside_defense_5d_pct",
    "benchmark_downside_defense_20d_pct",
}

PRICE_MODEL = "price_only"
BASE_MODEL = "v12_current_flow"
PRIMARY_CANDIDATE = "full_etf_query"
CANDIDATES = (
    PRIMARY_CANDIDATE,
    PRICE_MODEL,
    "full_etf_global_only",
    "full_etf_query_lag5",
    "full_etf_axis_shuffle",
    "full_etf_date_shuffle",
)
CRITICAL_CONTROLS = (
    PRICE_MODEL,
    "full_etf_axis_shuffle",
    "full_etf_date_shuffle",
)
STATE_CONTROLS = (
    "full_etf_global_only",
    "full_etf_query_lag5",
)
REQUIRED_V16_KEYS = {
    "actual",
    "date_codes",
    PRICE_MODEL,
    BASE_MODEL,
    "full_etf_query_raw",
    PRIMARY_CANDIDATE,
    "full_etf_global_only",
    "full_etf_query_lag5",
    "full_etf_axis_shuffle",
    "full_etf_date_shuffle",
}

SUMMARY_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
META_MODEL_PARAMETERS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 128,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 24,
    "l2_regularization": 10.0,
    "early_stopping": False,
    "random_state": RANDOM_SEED,
}
OUTCOME_NAMES = (
    "mae_gain",
    "basket_gain",
    "base_mae",
    "base_basket",
)

REFERENCE_PAPERS = (
    {
        "title": "Advances in Financial Machine Learning: Meta-Labeling",
        "url": "https://doi.org/10.1002/9781119482086",
        "implication": (
            "Use a secondary model to decide whether and how strongly to act "
            "on a primary forecast, while preserving chronological separation."
        ),
    },
    {
        "title": "On Calibration of Modern Neural Networks",
        "url": "https://proceedings.mlr.press/v70/guo17a.html",
        "implication": (
            "Predictive scores are not confidence by default; confidence must "
            "be evaluated out of sample."
        ),
    },
    {
        "title": "A Flow-Based Explanation for Return Predictability",
        "url": "https://academic.oup.com/rfs/article-abstract/25/12/3457/1594242",
        "implication": (
            "Flow-induced demand may be state dependent, so test timing and "
            "abstention rather than assuming a constant unconditional effect."
        ),
    },
)


@dataclass(frozen=True)
class DailyPanel:
    year: int
    candidate: str
    features: np.ndarray
    outcomes: np.ndarray
    date_codes: np.ndarray
    target_indices: np.ndarray
    feature_names: tuple[str, ...]


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
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.tobytes())
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def progress(**payload: Any) -> None:
    print(json.dumps(_json_ready(payload), sort_keys=True), flush=True)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _summary(values: np.ndarray) -> tuple[list[float], list[str]]:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, SUMMARY_QUANTILES)
    result = [float(np.mean(values)), float(np.std(values)), *map(float, quantiles)]
    names = ["mean", "std", "q10", "q25", "q50", "q75", "q90"]
    return result, names


def _daily_basket_value(
    *, actual: np.ndarray, prediction: np.ndarray, loss_target: bool
) -> float:
    count = max(1, len(actual) // 10)
    order = np.argsort(prediction)
    if loss_target:
        return -float(np.mean(actual[order[:count]]))
    return float(np.mean(actual[order[-count:]]))


def _daily_rank_ic(actual: np.ndarray, prediction: np.ndarray) -> float:
    if np.std(actual) <= 1e-12 or np.std(prediction) <= 1e-12:
        return 0.0
    actual_rank = np.argsort(np.argsort(actual)).astype(np.float64)
    predicted_rank = np.argsort(np.argsort(prediction)).astype(np.float64)
    return _safe_correlation(actual_rank, predicted_rank)


def load_fold(v16_root: Path, year: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = v16_root / f"fold_{year}.npz"
    json_path = v16_root / f"fold_{year}.json"
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"missing v16 fold {year}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if sha256_file(npz_path) != metadata.get("prediction_sha256"):
        raise ValueError(f"v16 fold {year} hash mismatch")
    with np.load(npz_path, allow_pickle=False) as item:
        if set(item.files) != REQUIRED_V16_KEYS:
            raise ValueError(f"v16 fold {year} key mismatch: {sorted(item.files)}")
        arrays = {name: np.asarray(item[name]) for name in item.files}
    row_count = len(arrays["actual"])
    if arrays["actual"].shape != (row_count, len(TARGET_NAMES)):
        raise ValueError(f"v16 fold {year} target shape mismatch")
    if arrays["date_codes"].shape != (row_count,):
        raise ValueError(f"v16 fold {year} date shape mismatch")
    for name in REQUIRED_V16_KEYS - {"actual", "date_codes"}:
        if arrays[name].shape != arrays["actual"].shape:
            raise ValueError(f"v16 fold {year} prediction shape mismatch for {name}")
    if not all(np.all(np.isfinite(arrays[name])) for name in REQUIRED_V16_KEYS):
        raise ValueError(f"v16 fold {year} contains non-finite values")
    return arrays, {
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "row_count": row_count,
        "date_count": int(len(np.unique(arrays["date_codes"]))),
        "prediction_sha256_verified": True,
    }


def _feature_row(
    *, base: np.ndarray, price: np.ndarray, candidate: np.ndarray, target_index: int
) -> tuple[list[float], tuple[str, ...]]:
    groups = {
        "base": base,
        "price": price,
        "candidate": candidate,
        "candidate_minus_base": candidate - base,
        "abs_candidate_minus_base": np.abs(candidate - base),
        "base_minus_price": base - price,
    }
    values: list[float] = []
    names: list[str] = []
    for prefix, group in groups.items():
        summary, summary_names = _summary(group)
        values.extend(summary)
        names.extend(f"{prefix}_{name}" for name in summary_names)
    values.extend(
        [
            _safe_correlation(candidate, base),
            _safe_correlation(candidate, price),
            _safe_correlation(base, price),
            float(np.mean(np.sign(candidate) == np.sign(base))),
            float(np.mean(np.sign(candidate) == np.sign(price))),
            float(np.mean(np.sign(base) == np.sign(price))),
        ]
    )
    names.extend(
        [
            "corr_candidate_base",
            "corr_candidate_price",
            "corr_base_price",
            "sign_agree_candidate_base",
            "sign_agree_candidate_price",
            "sign_agree_base_price",
        ]
    )
    for index, target_name in enumerate(TARGET_NAMES):
        values.append(float(index == target_index))
        names.append(f"target_{target_name}")
    values.extend(
        [
            float("_20d_" in f"_{TARGET_NAMES[target_index]}_"),
            float(TARGET_NAMES[target_index].startswith("loss_")),
        ]
    )
    names.extend(["horizon_20d", "loss_target"])
    return values, tuple(names)


def build_daily_panel(
    *, year: int, candidate_name: str, arrays: Mapping[str, np.ndarray]
) -> DailyPanel:
    actual = np.asarray(arrays["actual"], dtype=np.float32)
    dates = np.asarray(arrays["date_codes"], dtype=np.int32)
    price = np.asarray(arrays[PRICE_MODEL], dtype=np.float32)
    base = np.asarray(arrays[BASE_MODEL], dtype=np.float32)
    candidate = np.asarray(arrays[candidate_name], dtype=np.float32)
    feature_rows: list[list[float]] = []
    outcome_rows: list[list[float]] = []
    date_values: list[int] = []
    target_values: list[int] = []
    feature_names: tuple[str, ...] | None = None
    for date_code in np.unique(dates):
        indices = np.flatnonzero(dates == date_code)
        if len(indices) < 20:
            raise ValueError(f"year {year} date {date_code} has fewer than 20 rows")
        for target_index, target_name in enumerate(TARGET_NAMES):
            y = actual[indices, target_index]
            p = price[indices, target_index]
            b = base[indices, target_index]
            c = candidate[indices, target_index]
            row, names = _feature_row(
                base=b, price=p, candidate=c, target_index=target_index
            )
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise AssertionError("feature-name drift")
            base_mae = float(np.mean(np.abs(y - b)))
            candidate_mae = float(np.mean(np.abs(y - c)))
            loss_target = target_name.startswith("loss_")
            base_basket = _daily_basket_value(
                actual=y, prediction=b, loss_target=loss_target
            )
            candidate_basket = _daily_basket_value(
                actual=y, prediction=c, loss_target=loss_target
            )
            feature_rows.append(row)
            outcome_rows.append(
                [
                    base_mae - candidate_mae,
                    candidate_basket - base_basket,
                    base_mae,
                    base_basket,
                ]
            )
            date_values.append(int(date_code))
            target_values.append(target_index)
    if feature_names is None:
        raise ValueError(f"year {year} has no daily rows")
    return DailyPanel(
        year=year,
        candidate=candidate_name,
        features=np.asarray(feature_rows, dtype=np.float32),
        outcomes=np.asarray(outcome_rows, dtype=np.float32),
        date_codes=np.asarray(date_values, dtype=np.int32),
        target_indices=np.asarray(target_values, dtype=np.int16),
        feature_names=feature_names,
    )


def calibration_indices(
    panels: Sequence[DailyPanel], *, test_year: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not panels:
        raise ValueError("no calibration panels")
    features: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []
    audit: dict[str, Any] = {"years": [], "purged_daily_target_rows": 0}
    latest_year = max(panel.year for panel in panels)
    for panel in panels:
        if panel.year >= test_year:
            raise ValueError("future panel supplied to calibration")
        mask = np.ones(len(panel.date_codes), dtype=bool)
        if panel.year == latest_year:
            unique_dates = np.unique(panel.date_codes)
            purged_dates = set(unique_dates[-PURGE_SESSIONS:].tolist())
            mask = ~np.isin(panel.date_codes, list(purged_dates))
            audit["purged_daily_target_rows"] += int(np.sum(~mask))
            audit["purged_date_count"] = len(purged_dates)
            audit["purged_from_year"] = panel.year
        selected = int(np.sum(mask))
        if selected == 0:
            raise ValueError(f"empty calibration panel for {panel.year}")
        features.append(panel.features[mask])
        outcomes.append(panel.outcomes[mask])
        weights.append(np.full(selected, 1.0 / selected, dtype=np.float64))
        target_indices.append(panel.target_indices[mask])
        audit["years"].append(
            {
                "year": panel.year,
                "available_daily_target_rows": int(len(panel.date_codes)),
                "used_daily_target_rows": selected,
            }
        )
    x = np.concatenate(features)
    y = np.concatenate(outcomes)
    sample_weight = np.concatenate(weights)
    sample_weight *= len(sample_weight) / np.sum(sample_weight)
    audit["used_daily_target_rows"] = int(len(x))
    audit["feature_count"] = int(x.shape[1])
    audit["max_source_year"] = latest_year
    audit["target_indices"] = np.concatenate(target_indices)
    return x, y, {**audit, "sample_weight": sample_weight}


def fit_meta_models(
    *, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, seed_offset: int
) -> list[HistGradientBoostingRegressor]:
    models: list[HistGradientBoostingRegressor] = []
    for outcome_index in range(y.shape[1]):
        parameters = {
            **META_MODEL_PARAMETERS,
            "random_state": RANDOM_SEED + seed_offset + outcome_index,
        }
        model = HistGradientBoostingRegressor(**parameters)
        model.fit(x, y[:, outcome_index], sample_weight=sample_weight)
        models.append(model)
    return models


def decisions_from_predictions(
    *,
    panel: DailyPanel,
    meta_predictions: np.ndarray,
    calibration_outcomes: np.ndarray,
    calibration_target_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    if meta_predictions.shape != panel.outcomes.shape:
        raise ValueError("meta prediction shape mismatch")
    switch = (meta_predictions[:, 0] > 0.0) & (meta_predictions[:, 1] > 0.0)
    if len(calibration_target_indices) != len(calibration_outcomes):
        raise ValueError("calibration target identity mismatch")
    risk_thresholds = np.zeros(len(TARGET_NAMES), dtype=np.float32)
    basket_thresholds = np.zeros(len(TARGET_NAMES), dtype=np.float32)
    for target_index in range(len(TARGET_NAMES)):
        target_rows = calibration_target_indices == target_index
        risk_values = calibration_outcomes[target_rows, 2]
        basket_values = calibration_outcomes[target_rows, 3]
        if not len(risk_values):
            raise ValueError(f"no calibration risk values for target {target_index}")
        risk_thresholds[target_index] = float(np.median(risk_values))
        basket_thresholds[target_index] = float(np.median(basket_values))
    predicted_hybrid_basket = meta_predictions[:, 3] + np.where(
        switch, meta_predictions[:, 1], 0.0
    )
    safe = (
        meta_predictions[:, 2] <= risk_thresholds[panel.target_indices]
    ) & (
        predicted_hybrid_basket >= basket_thresholds[panel.target_indices]
    )
    return {
        "switch": switch,
        "safe": safe,
        "risk_thresholds": risk_thresholds,
        "basket_thresholds": basket_thresholds,
        "meta_predictions": meta_predictions.astype(np.float32),
    }


def _reshape_daily(values: np.ndarray, panel: DailyPanel) -> np.ndarray:
    dates = np.unique(panel.date_codes)
    date_to_index = {int(value): index for index, value in enumerate(dates)}
    if values.ndim == 1:
        result = np.zeros((len(dates), len(TARGET_NAMES)), dtype=values.dtype)
    else:
        result = np.zeros(
            (len(dates), len(TARGET_NAMES), values.shape[1]), dtype=values.dtype
        )
    if len(panel.date_codes) != len(panel.target_indices):
        raise ValueError("daily panel identity length mismatch")
    for row_index, (date_code, target_index) in enumerate(
        zip(panel.date_codes, panel.target_indices)
    ):
        result[date_to_index[int(date_code)], int(target_index)] = values[row_index]
    return result


def apply_daily_decisions(
    *,
    date_codes: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    panel: DailyPanel,
    switch: np.ndarray,
    safe: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_dates = np.unique(date_codes)
    switch_matrix = _reshape_daily(switch, panel).astype(bool)
    safe_matrix = _reshape_daily(safe, panel).astype(bool)
    if switch_matrix.shape != (len(unique_dates), len(TARGET_NAMES)):
        raise ValueError("daily switch matrix shape mismatch")
    date_to_index = {int(value): index for index, value in enumerate(unique_dates)}
    row_dates = np.asarray([date_to_index[int(value)] for value in date_codes])
    hybrid = np.asarray(base, dtype=np.float32).copy()
    safe_rows = np.zeros_like(hybrid, dtype=bool)
    for target_index in range(len(TARGET_NAMES)):
        use_candidate = switch_matrix[row_dates, target_index]
        hybrid[use_candidate, target_index] = candidate[use_candidate, target_index]
        safe_rows[:, target_index] = safe_matrix[row_dates, target_index]
    return hybrid, safe_rows, switch_matrix, safe_matrix


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    correlation = _safe_correlation(target, prediction)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "correlation": correlation,
        "direction_accuracy": float(np.mean((target >= 0) == (prediction >= 0))),
    }


def stock_cross_sectional_metrics(
    *,
    date_codes: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    loss_target: bool,
) -> dict[str, float]:
    rank_ics: list[float] = []
    spreads: list[float] = []
    tops: list[float] = []
    bottoms: list[float] = []
    daily_values: list[float] = []
    for date_code in np.unique(date_codes):
        indices = np.flatnonzero(date_codes == date_code)
        if len(indices) < 20:
            continue
        actual = target[indices]
        forecast = prediction[indices]
        rank_ics.append(_daily_rank_ic(actual, forecast))
        basket_count = max(1, len(indices) // 10)
        order = np.argsort(forecast)
        bottom = float(np.mean(actual[order[:basket_count]]))
        top = float(np.mean(actual[order[-basket_count:]]))
        bottoms.append(bottom)
        tops.append(top)
        spreads.append(top - bottom)
        daily_values.append(-bottom if loss_target else top)
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
            float(np.quantile(daily_values, 0.10)) if daily_values else math.nan
        ),
        "evaluated_date_count": len(spreads),
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


def safe_metrics(
    *,
    date_codes: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    safe_rows: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        mask = safe_rows[:, target_index]
        selected_dates = np.unique(date_codes[mask])
        total_dates = np.unique(date_codes)
        if not np.any(mask):
            result[target_name] = {
                "mae": math.inf,
                "rmse": math.inf,
                "correlation": -math.inf,
                "direction_accuracy": 0.0,
                "mean_daily_rank_ic": -math.inf,
                "positive_daily_rank_ic_ratio": 0.0,
                "mean_top_minus_bottom_spread": -math.inf,
                "mean_predicted_top_realized": -math.inf,
                "mean_predicted_bottom_realized": math.inf,
                "economic_basket_value": -math.inf,
                "economic_basket_p10": -math.inf,
                "evaluated_date_count": 0,
                "safe_date_count": 0,
                "total_date_count": int(len(total_dates)),
                "coverage": 0.0,
            }
            continue
        result[target_name] = {
            **regression_metrics(
                actual[mask, target_index], prediction[mask, target_index]
            ),
            **stock_cross_sectional_metrics(
                date_codes=date_codes[mask],
                target=actual[mask, target_index],
                prediction=prediction[mask, target_index],
                loss_target=target_name.startswith("loss_"),
            ),
            "safe_date_count": int(len(selected_dates)),
            "total_date_count": int(len(total_dates)),
            "coverage": float(len(selected_dates) / len(total_dates)),
        }
    return result


def preregistration(v16_root: Path) -> dict[str, Any]:
    receipt_path = v16_root / "v16_full_etf_identity_receipt.json"
    prereg_path = v16_root / "v16_full_etf_identity_preregistration.json"
    if not receipt_path.exists() or not prereg_path.exists():
        raise FileNotFoundError("sealed v16 receipt/preregistration missing")
    fold_inputs: dict[str, Any] = {}
    for year in ALL_YEARS:
        npz_path = v16_root / f"fold_{year}.npz"
        json_path = v16_root / f"fold_{year}.json"
        fold_inputs[str(year)] = {
            "npz_sha256": sha256_file(npz_path),
            "json_sha256": sha256_file(json_path),
        }
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "frozen_before_results": True,
        "purpose": (
            "test whether slow full-ETF latent state is useful as a strictly "
            "prequential confidence, model-selection, or avoidance gate"
        ),
        "timing": {
            "base_predictions": "sealed v16 outer-year OOS predictions only",
            "calibration_year": CALIBRATION_YEAR,
            "evaluation_years": list(EVALUATION_YEARS),
            "training_rule": "each test year uses earlier OOS years only",
            "latest_calibration_year_purge_sessions": PURGE_SESSIONS,
            "same_year_adaptation": False,
            "future_label_or_threshold_tuning": False,
        },
        "scope": {
            "all_v16_rows": True,
            "no_date_stock_or_target_sampling": True,
            "targets": list(TARGET_NAMES),
            "candidates": list(CANDIDATES),
            "primary_candidate": PRIMARY_CANDIDATE,
            "base_model": BASE_MODEL,
        },
        "features": {
            "source": "prediction distributions available at signal time",
            "target_values_used_as_features": False,
            "families": [
                "base_prediction_distribution",
                "price_prediction_distribution",
                "candidate_prediction_distribution",
                "candidate_minus_base_distribution",
                "absolute_candidate_minus_base_distribution",
                "base_minus_price_distribution",
                "prediction_correlations_and_sign_agreement",
                "fixed_target_identity_and_horizon",
            ],
        },
        "meta_targets": {
            "mae_gain": "daily MAE(v12) minus daily MAE(candidate)",
            "basket_gain": "daily economic basket(candidate) minus v12",
            "base_mae": "daily v12 MAE for risk/avoidance",
            "base_basket": "daily v12 economic basket value",
        },
        "decision_rule": {
            "switch_to_candidate": (
                "predicted_mae_gain > 0 AND predicted_basket_gain > 0"
            ),
            "otherwise": "use sealed v12 current-Flow prediction",
            "safe_date_target": (
                "predicted_base_mae <= target-specific calibration median AND "
                "predicted hybrid basket value >= target-specific calibration "
                "median; both medians use purged calibration data only"
            ),
            "thresholds_tuned_on_test": False,
        },
        "estimator": {
            "library": "sklearn HistGradientBoostingRegressor",
            "parameters": META_MODEL_PARAMETERS,
            "seed_rule": (
                "20260829 + candidate_index * 100 + meta_target_index"
            ),
            "one_model_per_meta_target": True,
            "same_capacity_for_primary_and_controls": True,
            "year_balanced_sample_weight": True,
        },
        "controls": {
            "price_only_meta_gate": True,
            "global_only_meta_gate": True,
            "lag5_meta_gate": True,
            "etf_axis_shuffle_meta_gate": True,
            "date_shuffle_meta_gate": True,
            "always_v12": True,
            "always_full_etf_query": True,
        },
        "fixed_gate": {
            "forecast_switch": {
                "mae_beats_v12_targets": 8,
                "mae_beats_always_query_targets": 8,
                "rank_ic_beats_v12_targets": 8,
                "basket_beats_v12_targets": 8,
                "basket_beats_always_query_targets": 8,
                "critical_control_mae_and_basket_targets_each": 8,
                "state_control_mae_targets_each": 8,
                "mean_mae_improvement_vs_v12_positive": True,
                "worst_target_mae_degradation_at_most_pct": 0.25,
                "positive_fold_targets": 30,
                "year_2025_and_2026_mean_improvement_nonnegative": True,
                "each_year_candidate_switch_fraction": [0.10, 0.90],
            },
            "avoidance": {
                "core_targets": sorted(CORE_AVOIDANCE_TARGETS),
                "coverage_each_core_target": [0.20, 0.80],
                "safe_mae_beats_unfiltered_v12_core_targets": 3,
                "safe_rank_ic_beats_unfiltered_v12_core_targets": 3,
                "safe_basket_beats_unfiltered_v12_core_targets": 3,
                "safe_basket_p10_beats_unfiltered_v12_core_targets": 3,
                "safe_basket_beats_axis_and_date_shuffle_each": 3,
            },
        },
        "activation": {
            "historical_result_is_exploratory": True,
            "deployment_forbidden": True,
            "bf16_set_transformer_activation_forbidden": True,
            "nvfp4_conversion_forbidden": True,
            "pass": "ELIGIBLE_FOR_FUTURE_PROSPECTIVE_SHADOW_LOCKBOX_ONLY",
            "fail": "NO_CONFIDENCE_OR_AVOIDANCE_ACTIVATION",
        },
        "frozen_inputs": {
            "source_sha256": sha256_file(Path(__file__)),
            "v16_receipt_sha256": sha256_file(receipt_path),
            "v16_preregistration_sha256": sha256_file(prereg_path),
            "folds": fold_inputs,
        },
        "references": list(REFERENCE_PAPERS),
    }


def _fold_checkpoint(
    *, output_root: Path, year: int, preregistration_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    npz_path = output_root / f"fold_{year}.npz"
    json_path = output_root / f"fold_{year}.json"
    if not npz_path.exists() or not json_path.exists():
        return None
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"v17 fold {year} preregistration mismatch")
    if sha256_file(npz_path) != metadata.get("decision_sha256"):
        raise ValueError(f"v17 fold {year} decision hash mismatch")
    with np.load(npz_path, allow_pickle=False) as item:
        arrays = {name: np.asarray(item[name]) for name in item.files}
    return arrays, metadata


def _save_fold_checkpoint(
    *,
    output_root: Path,
    year: int,
    preregistration_sha256: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    npz_path = output_root / f"fold_{year}.npz"
    json_path = output_root / f"fold_{year}.json"
    write_npz_atomic(npz_path, arrays)
    result = {
        **dict(metadata),
        "preregistration_sha256": preregistration_sha256,
        "decision_path": str(npz_path),
        "decision_sha256": sha256_file(npz_path),
    }
    write_json_atomic(json_path, result)
    return result


def _gate_summary(
    *,
    pooled: Mapping[str, Mapping[str, Any]],
    yearly: Mapping[str, Mapping[str, Mapping[str, Any]]],
    fold_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary_hybrid = pooled[f"hybrid_{PRIMARY_CANDIDATE}"]
    primary_safe = pooled[f"safe_{PRIMARY_CANDIDATE}"]
    base = pooled[BASE_MODEL]
    always_query = pooled[f"always_{PRIMARY_CANDIDATE}"]
    counters: defaultdict[str, int] = defaultdict(int)
    improvements: list[float] = []
    core: defaultdict[str, int] = defaultdict(int)

    for target_name in TARGET_NAMES:
        hybrid = primary_hybrid[target_name]
        baseline = base[target_name]
        always = always_query[target_name]
        safe = primary_safe[target_name]
        improvement = (baseline["mae"] - hybrid["mae"]) / baseline["mae"] * 100.0
        improvements.append(improvement)
        counters["mae_beats_v12"] += hybrid["mae"] < baseline["mae"]
        counters["mae_beats_always_query"] += hybrid["mae"] < always["mae"]
        counters["rank_ic_beats_v12"] += (
            hybrid["mean_daily_rank_ic"] > baseline["mean_daily_rank_ic"]
        )
        counters["basket_beats_v12"] += (
            hybrid["economic_basket_value"] > baseline["economic_basket_value"]
        )
        counters["basket_beats_always_query"] += (
            hybrid["economic_basket_value"] > always["economic_basket_value"]
        )
        for control in (*CRITICAL_CONTROLS, *STATE_CONTROLS):
            controlled = pooled[f"hybrid_{control}"][target_name]
            counters[f"mae_beats_{control}"] += hybrid["mae"] < controlled["mae"]
            counters[f"basket_beats_{control}"] += (
                hybrid["economic_basket_value"]
                > controlled["economic_basket_value"]
            )
        if target_name in CORE_AVOIDANCE_TARGETS:
            core["coverage_valid"] += 0.20 <= safe["coverage"] <= 0.80
            core["safe_mae_beats_v12"] += safe["mae"] < baseline["mae"]
            core["safe_rank_ic_beats_v12"] += (
                safe["mean_daily_rank_ic"] > baseline["mean_daily_rank_ic"]
            )
            core["safe_basket_beats_v12"] += (
                safe["economic_basket_value"] > baseline["economic_basket_value"]
            )
            core["safe_basket_p10_beats_v12"] += (
                safe["economic_basket_p10"] > baseline["economic_basket_p10"]
            )
            for control in ("full_etf_axis_shuffle", "full_etf_date_shuffle"):
                controlled = pooled[f"safe_{control}"][target_name]
                core[f"safe_basket_beats_{control}"] += (
                    safe["economic_basket_value"]
                    > controlled["economic_basket_value"]
                )

    yearly_improvements: dict[str, float] = {}
    positive_fold_targets = 0
    total_fold_targets = 0
    for year, models in yearly.items():
        values: list[float] = []
        for target_name in TARGET_NAMES:
            baseline = models[BASE_MODEL][target_name]
            hybrid = models[f"hybrid_{PRIMARY_CANDIDATE}"][target_name]
            value = (baseline["mae"] - hybrid["mae"]) / baseline["mae"] * 100.0
            values.append(value)
            positive_fold_targets += value > 0
            total_fold_targets += 1
        yearly_improvements[year] = float(np.mean(values))

    switch_fractions = {
        str(item["outer_year"]): float(
            item["candidates"][PRIMARY_CANDIDATE]["switch_fraction"]
        )
        for item in fold_metadata
    }
    switch_non_degenerate = all(
        0.10 <= value <= 0.90 for value in switch_fractions.values()
    )
    mean_improvement = float(np.mean(improvements))
    worst_improvement = float(np.min(improvements))
    critical_controls_pass = all(
        counters[f"mae_beats_{control}"] >= 8
        and counters[f"basket_beats_{control}"] >= 8
        for control in CRITICAL_CONTROLS
    )
    state_controls_pass = all(
        counters[f"mae_beats_{control}"] >= 8 for control in STATE_CONTROLS
    )
    forecast_switch = (
        counters["mae_beats_v12"] >= 8
        and counters["mae_beats_always_query"] >= 8
        and counters["rank_ic_beats_v12"] >= 8
        and counters["basket_beats_v12"] >= 8
        and counters["basket_beats_always_query"] >= 8
        and critical_controls_pass
        and state_controls_pass
        and mean_improvement > 0
        and worst_improvement >= -0.25
        and positive_fold_targets >= 30
        and yearly_improvements.get("2025", -math.inf) >= 0
        and yearly_improvements.get("2026", -math.inf) >= 0
        and switch_non_degenerate
    )
    avoidance = (
        core["coverage_valid"] == len(CORE_AVOIDANCE_TARGETS)
        and core["safe_mae_beats_v12"] >= 3
        and core["safe_rank_ic_beats_v12"] >= 3
        and core["safe_basket_beats_v12"] >= 3
        and core["safe_basket_p10_beats_v12"] >= 3
        and core["safe_basket_beats_full_etf_axis_shuffle"] >= 3
        and core["safe_basket_beats_full_etf_date_shuffle"] >= 3
    )
    passed_paths = [
        name
        for name, passed in (
            ("FORECAST_SWITCH", forecast_switch),
            ("AVOIDANCE", avoidance),
        )
        if passed
    ]
    return {
        "status": (
            "V17_PREQUENTIAL_GATE_PASS"
            if passed_paths
            else "V17_PREQUENTIAL_GATE_FAIL"
        ),
        "passed_paths": passed_paths,
        "fixed_before_results": True,
        "historical_oos_not_fresh_forward_lockbox": True,
        "checks": {
            "forecast_switch_path_pass": forecast_switch,
            "avoidance_path_pass": avoidance,
            "critical_controls_pass": critical_controls_pass,
            "state_controls_pass": state_controls_pass,
            "mean_mae_improvement_positive": mean_improvement > 0,
            "worst_target_degradation_at_most_0_25pct": worst_improvement >= -0.25,
            "positive_fold_targets_at_least_30": positive_fold_targets >= 30,
            "2025_and_2026_nonnegative": (
                yearly_improvements.get("2025", -math.inf) >= 0
                and yearly_improvements.get("2026", -math.inf) >= 0
            ),
            "switch_fraction_non_degenerate_each_year": switch_non_degenerate,
        },
        "counters": {
            **dict(counters),
            "avoidance_core": dict(core),
            "mean_mae_improvement_vs_v12_pct": mean_improvement,
            "worst_target_mae_improvement_vs_v12_pct": worst_improvement,
            "positive_fold_target_count": int(positive_fold_targets),
            "outer_fold_target_count": int(total_fold_targets),
            "yearly_mean_mae_improvement_vs_v12_pct": yearly_improvements,
            "primary_switch_fraction_by_year": switch_fractions,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, Mapping[str, Any]]:
    v16_root = Path(args.v16_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "v17_prequential_gate_receipt.json"
    run_state_path = output_root / "run_state.json"
    prereg_path = output_root / "v17_prequential_gate_preregistration.json"
    if receipt_path.exists():
        raise FileExistsError(f"receipt already exists: {receipt_path}")

    proposed = preregistration(v16_root)
    if prereg_path.exists():
        existing = json.loads(prereg_path.read_text(encoding="utf-8"))
        if existing != proposed:
            raise ValueError("existing v17 preregistration differs from current source/input")
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
    fold_arrays: dict[int, dict[str, np.ndarray]] = {}
    input_receipts: dict[str, Any] = {}
    panels: dict[str, dict[int, DailyPanel]] = {name: {} for name in CANDIDATES}
    for year in ALL_YEARS:
        arrays, input_receipt = load_fold(v16_root, year)
        fold_arrays[year] = arrays
        input_receipts[str(year)] = input_receipt
        for candidate in CANDIDATES:
            panels[candidate][year] = build_daily_panel(
                year=year, candidate_name=candidate, arrays=arrays
            )
        progress(stage="daily_panel", year=year, rows=len(arrays["actual"]))

    pooled_actual: list[np.ndarray] = []
    pooled_dates: list[np.ndarray] = []
    pooled_base: list[np.ndarray] = []
    pooled_always_query: list[np.ndarray] = []
    pooled_hybrid: dict[str, list[np.ndarray]] = defaultdict(list)
    pooled_safe_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    yearly_metrics: dict[str, Any] = {}
    fold_metadata: list[dict[str, Any]] = []

    for year in EVALUATION_YEARS:
        started = time.monotonic()
        arrays = fold_arrays[year]
        checkpoint = _fold_checkpoint(
            output_root=output_root,
            year=year,
            preregistration_sha256=prereg_sha,
        )
        if checkpoint is None:
            checkpoint_arrays: dict[str, np.ndarray] = {}
            candidates_metadata: dict[str, Any] = {}
            for candidate_index, candidate in enumerate(CANDIDATES):
                calibration = [
                    panels[candidate][source_year]
                    for source_year in ALL_YEARS
                    if source_year < year
                ]
                x, y, audit = calibration_indices(calibration, test_year=year)
                sample_weight = np.asarray(audit.pop("sample_weight"), dtype=np.float64)
                calibration_target_indices = np.asarray(
                    audit.pop("target_indices"), dtype=np.int16
                )
                models = fit_meta_models(
                    x=x,
                    y=y,
                    sample_weight=sample_weight,
                    seed_offset=candidate_index * 100,
                )
                test_panel = panels[candidate][year]
                meta_prediction = np.column_stack(
                    [model.predict(test_panel.features) for model in models]
                ).astype(np.float32)
                decision = decisions_from_predictions(
                    panel=test_panel,
                    meta_predictions=meta_prediction,
                    calibration_outcomes=y,
                    calibration_target_indices=calibration_target_indices,
                )
                switch = decision["switch"]
                safe = decision["safe"]
                checkpoint_arrays[f"switch__{candidate}"] = _reshape_daily(
                    switch, test_panel
                ).astype(np.uint8)
                checkpoint_arrays[f"safe__{candidate}"] = _reshape_daily(
                    safe, test_panel
                ).astype(np.uint8)
                checkpoint_arrays[f"meta__{candidate}"] = _reshape_daily(
                    meta_prediction, test_panel
                ).astype(np.float32)
                checkpoint_arrays[f"risk_thresholds__{candidate}"] = decision[
                    "risk_thresholds"
                ].astype(np.float32)
                checkpoint_arrays[f"basket_thresholds__{candidate}"] = decision[
                    "basket_thresholds"
                ].astype(np.float32)
                candidates_metadata[candidate] = {
                    "calibration": audit,
                    "feature_count": len(test_panel.feature_names),
                    "feature_names": list(test_panel.feature_names),
                    "switch_fraction": float(np.mean(switch)),
                    "safe_fraction": float(np.mean(safe)),
                    "daily_target_rows": int(len(test_panel.date_codes)),
                    "meta_prediction_sha256": array_sha256(meta_prediction),
                }
                progress(
                    stage="meta_gate",
                    year=year,
                    candidate=candidate,
                    switch_fraction=float(np.mean(switch)),
                    safe_fraction=float(np.mean(safe)),
                )
            checkpoint_arrays["daily_date_codes"] = np.unique(
                panels[PRIMARY_CANDIDATE][year].date_codes
            ).astype(np.int32)
            metadata = _save_fold_checkpoint(
                output_root=output_root,
                year=year,
                preregistration_sha256=prereg_sha,
                arrays=checkpoint_arrays,
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "outer_year": year,
                    "v16_input": input_receipts[str(year)],
                    "candidates": candidates_metadata,
                    "elapsed_seconds": float(time.monotonic() - started),
                    "resumed": False,
                },
            )
        else:
            checkpoint_arrays, metadata = checkpoint
            metadata = {**metadata, "resumed": True}

        year_hybrid: dict[str, np.ndarray] = {}
        year_safe: dict[str, np.ndarray] = {}
        checkpoint_date_codes = np.asarray(
            checkpoint_arrays["daily_date_codes"], dtype=np.int32
        )
        checkpoint_date_to_index = {
            int(value): index for index, value in enumerate(checkpoint_date_codes)
        }
        for candidate in CANDIDATES:
            panel = panels[candidate][year]
            switch_matrix = checkpoint_arrays[f"switch__{candidate}"].astype(bool)
            safe_matrix = checkpoint_arrays[f"safe__{candidate}"].astype(bool)
            switch = np.asarray(
                [
                    switch_matrix[
                        checkpoint_date_to_index[int(date_code)], int(target_index)
                    ]
                    for date_code, target_index in zip(
                        panel.date_codes, panel.target_indices
                    )
                ],
                dtype=bool,
            )
            safe = np.asarray(
                [
                    safe_matrix[
                        checkpoint_date_to_index[int(date_code)], int(target_index)
                    ]
                    for date_code, target_index in zip(
                        panel.date_codes, panel.target_indices
                    )
                ],
                dtype=bool,
            )
            hybrid, safe_rows, _, _ = apply_daily_decisions(
                date_codes=arrays["date_codes"],
                base=arrays[BASE_MODEL],
                candidate=arrays[candidate],
                panel=panel,
                switch=switch,
                safe=safe,
            )
            year_hybrid[candidate] = hybrid
            year_safe[candidate] = safe_rows
            pooled_hybrid[candidate].append(hybrid)
            pooled_safe_rows[candidate].append(safe_rows)

        global_dates = year * 1000 + arrays["date_codes"].astype(np.int64)
        models_for_year: dict[str, Any] = {
            BASE_MODEL: full_metrics(
                date_codes=global_dates,
                actual=arrays["actual"],
                prediction=arrays[BASE_MODEL],
            ),
            f"always_{PRIMARY_CANDIDATE}": full_metrics(
                date_codes=global_dates,
                actual=arrays["actual"],
                prediction=arrays[PRIMARY_CANDIDATE],
            ),
        }
        for candidate in CANDIDATES:
            models_for_year[f"hybrid_{candidate}"] = full_metrics(
                date_codes=global_dates,
                actual=arrays["actual"],
                prediction=year_hybrid[candidate],
            )
            models_for_year[f"safe_{candidate}"] = safe_metrics(
                date_codes=global_dates,
                actual=arrays["actual"],
                prediction=year_hybrid[candidate],
                safe_rows=year_safe[candidate],
            )
        yearly_metrics[str(year)] = models_for_year
        fold_metadata.append(metadata)
        pooled_actual.append(arrays["actual"])
        pooled_dates.append(global_dates)
        pooled_base.append(arrays[BASE_MODEL])
        pooled_always_query.append(arrays[PRIMARY_CANDIDATE])
        progress(stage="outer_year_complete", year=year)

    actual_all = np.concatenate(pooled_actual)
    dates_all = np.concatenate(pooled_dates)
    base_all = np.concatenate(pooled_base)
    query_all = np.concatenate(pooled_always_query)
    pooled_metrics: dict[str, Any] = {
        BASE_MODEL: full_metrics(
            date_codes=dates_all, actual=actual_all, prediction=base_all
        ),
        f"always_{PRIMARY_CANDIDATE}": full_metrics(
            date_codes=dates_all, actual=actual_all, prediction=query_all
        ),
    }
    for candidate in CANDIDATES:
        hybrid = np.concatenate(pooled_hybrid[candidate])
        safe_rows = np.concatenate(pooled_safe_rows[candidate])
        pooled_metrics[f"hybrid_{candidate}"] = full_metrics(
            date_codes=dates_all, actual=actual_all, prediction=hybrid
        )
        pooled_metrics[f"safe_{candidate}"] = safe_metrics(
            date_codes=dates_all,
            actual=actual_all,
            prediction=hybrid,
            safe_rows=safe_rows,
        )

    gate = _gate_summary(
        pooled=pooled_metrics,
        yearly=yearly_metrics,
        fold_metadata=fold_metadata,
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
        "meta_model": {
            "parameters": META_MODEL_PARAMETERS,
            "outcomes": list(OUTCOME_NAMES),
            "target_values_used_as_features": False,
        },
        "folds": fold_metadata,
        "pooled_metrics": pooled_metrics,
        "yearly_metrics": yearly_metrics,
        "gate": gate,
        "implementation_validity": {
            "v16_hashes_verified": True,
            "candidate_capacity_equal": True,
            "target_free_features": True,
            "prequential_prior_oos_only": True,
            "last_20_sessions_purged": True,
        },
        "limitations": [
            "The v16 historical OOS period has already been inspected; v17 is exploratory, not a fresh prospective lockbox.",
            "The gate sees distributions of base-model predictions, not raw ETF latent vectors or stock identities.",
            "Passing authorizes only a future shadow lockbox; it does not authorize deployment or trading.",
        ],
        "next_activation": (
            "FUTURE_PROSPECTIVE_SHADOW_LOCKBOX_ONLY"
            if gate["passed_paths"]
            else "NO_CONFIDENCE_OR_AVOIDANCE_ACTIVATION"
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
        stage="complete",
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
