"""Date-balanced nonlinear ETF Flow residual canary.

This module reuses the already audited v11-R2 point-in-time matrix builder but
changes the statistical unit and estimator. Every market date has equal total
training weight. A fixed-capacity CatBoost model is trained once on price-only
features and once on price plus Flow features. The primary forecast is a
pre-fixed, capped 25% move from the price prediction toward the enriched
prediction. Lagged Flow and topology-shuffled Flow receive identical model
capacity.

No existing v11 result is overwritten. A historical PASS remains exploratory
because the 2021-2026 period has already informed successor design choices.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.contracts import (
    DEFAULT_SOURCE_DATABASE,
    TIMING_CONTRACT,
)
from training.quant_flow_graph_v11_r2.phase_a import (
    json_sha256,
    readonly_connection,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from training.quant_flow_graph_v11_r2.phase_b_stock import (
    DEFAULT_GRAPH_DATASET_ROOT,
    DEFAULT_PHASE_A_ROOT,
    GLOBAL_FLOW_FIELDS,
    OUTER_YEARS,
    PURGE_SESSIONS,
    TARGET_NAMES,
    build_stock_matrix_from_sources,
    fold_indices,
    lag_flow_by_symbol,
    regression_metrics,
    stock_cross_sectional_metrics,
    topology_shuffle,
)


SCHEMA_VERSION = "quant.etf_flow_v12.date_balanced_residual_canary.v1"
PREREGISTRATION_SCHEMA_VERSION = (
    "quant.etf_flow_v12.date_balanced_residual_canary_preregistration.v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v12/"
    "catboost_residual_canary"
)

CATBOOST_VERSION = "1.2.10"
RANDOM_SEED = 20260828
RESIDUAL_SHRINKAGE = 0.25
RESIDUAL_CAP_QUANTILE = 0.90
TOP_FEATURE_COUNT = 30
PRIMARY_MODEL = "capped_flow_residual"
PRICE_MODEL = "price_only"
CONTROL_MODELS = (
    "lag5_capped_residual",
    "lag20_capped_residual",
    "topology_shuffle_capped_residual",
)
MODEL_NAMES = (
    PRICE_MODEL,
    "full_flow_raw",
    PRIMARY_MODEL,
    *CONTROL_MODELS,
)
CATBOOST_PARAMETERS = {
    "loss_function": "MultiRMSE",
    "eval_metric": "MultiRMSE",
    "iterations": 256,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 20.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.80,
    "rsm": 0.70,
    "leaf_estimation_iterations": 1,
    "random_seed": RANDOM_SEED,
    "task_type": "CPU",
    "allow_writing_files": False,
    "verbose": False,
}


def _progress(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def date_balanced_weights(date_codes: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Give every included signal date identical total sample weight."""

    selected = np.asarray(date_codes, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    if not len(selected):
        raise ValueError("cannot weight an empty split")
    unique, inverse, counts = np.unique(selected, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / len(unique)
    return weights.astype(np.float32)


def residual_caps(train_targets: np.ndarray) -> np.ndarray:
    """Scale correction caps using training targets only."""

    values = np.asarray(train_targets, dtype=np.float64)
    medians = np.median(values, axis=0)
    absolute = np.abs(values - medians)
    caps = RESIDUAL_SHRINKAGE * np.quantile(
        absolute, RESIDUAL_CAP_QUANTILE, axis=0
    )
    return np.maximum(caps, 1e-6).astype(np.float32)


def capped_residual_prediction(
    price_prediction: np.ndarray,
    enriched_prediction: np.ndarray,
    caps: np.ndarray,
) -> np.ndarray:
    """Apply the preregistered price-preserving residual adapter."""

    price = np.asarray(price_prediction, dtype=np.float32)
    enriched = np.asarray(enriched_prediction, dtype=np.float32)
    correction = RESIDUAL_SHRINKAGE * (enriched - price)
    correction = np.clip(correction, -np.asarray(caps), np.asarray(caps))
    return (price + correction).astype(np.float32)


def _sanitize_features(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if np.any(np.isinf(result)):
        result = result.copy()
        result[np.isinf(result)] = np.nan
    return result


def _catboost_regressor(thread_count: int):
    try:
        import catboost
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError(
            "CatBoost is missing; use the isolated v12 python_deps directory"
        ) from exc
    if catboost.__version__ != CATBOOST_VERSION:
        raise RuntimeError(
            f"CatBoost version mismatch: {catboost.__version__} != {CATBOOST_VERSION}"
        )
    return catboost.CatBoostRegressor(
        **CATBOOST_PARAMETERS,
        thread_count=int(thread_count),
    )


def fit_predict_multioutput(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    weights: np.ndarray,
    feature_names: Sequence[str],
    thread_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Fit one fixed-capacity model and return prediction plus importances."""

    started = time.monotonic()
    model = _catboost_regressor(thread_count)
    model.fit(
        _sanitize_features(features[train]),
        np.asarray(targets[train], dtype=np.float32),
        sample_weight=np.asarray(weights, dtype=np.float32),
        verbose=False,
    )
    prediction = np.asarray(
        model.predict(_sanitize_features(features[test])), dtype=np.float32
    )
    if prediction.shape != (len(test), len(TARGET_NAMES)):
        raise ValueError(f"unexpected prediction shape: {prediction.shape}")
    importance = np.asarray(model.get_feature_importance(), dtype=np.float64)
    order = np.argsort(importance)[::-1][:TOP_FEATURE_COUNT]
    top = [
        {
            "feature": str(feature_names[index]),
            "importance": float(importance[index]),
        }
        for index in order
    ]
    elapsed = time.monotonic() - started
    del model
    gc.collect()
    return prediction, top, elapsed


def preregistration() -> dict[str, Any]:
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "frozen_before_results": True,
        "purpose": (
            "detect small nonlinear incremental ETF Flow edge after correcting the "
            "effective statistical unit from stock rows to equally weighted dates"
        ),
        "timing_contract": TIMING_CONTRACT,
        "scope": {
            "targets": list(TARGET_NAMES),
            "outer_years": list(OUTER_YEARS),
            "purge_sessions": PURGE_SESSIONS,
            "no_row_or_symbol_sampling": True,
        },
        "estimator": {
            "library": "catboost",
            "version": CATBOOST_VERSION,
            "parameters": CATBOOST_PARAMETERS,
            "date_balanced_total_weight": True,
            "symbol_identity_feature": False,
        },
        "residual_adapter": {
            "primary_model": PRIMARY_MODEL,
            "shrinkage": RESIDUAL_SHRINKAGE,
            "cap_quantile": RESIDUAL_CAP_QUANTILE,
            "cap_source": "outer_train_targets_only",
            "formula": (
                "price_pred + clip(shrinkage * (enriched_pred - price_pred), "
                "+/- shrinkage*q90(abs(train_target-train_median)))"
            ),
        },
        "controls": {
            "equal_capacity_price_only": True,
            "flow_lag_sessions": [5, 20],
            "within_date_stock_topology_shuffle_seed": RANDOM_SEED,
            "global_flow_preserved_in_topology_shuffle": True,
            "same_model_capacity_for_all_variants": True,
        },
        "gate_thresholds": {
            "forecast": {
                "mae_beats_price_targets": 8,
                "mean_mae_improvement_positive": True,
                "worst_target_mae_degradation_at_most_pct": 1.0,
                "mae_beats_lag5_targets": 8,
                "mae_beats_lag20_targets": 8,
                "mae_beats_topology_shuffle_targets": 8,
                "positive_outer_fold_targets": 36,
            },
            "basket": {
                "rank_ic_beats_price_targets": 8,
                "economic_basket_beats_price_targets": 8,
                "mae_beats_topology_shuffle_targets": 8,
                "positive_outer_fold_targets": 36,
            },
            "avoidance": {
                "core_targets": [
                    "loss_5d_pct",
                    "loss_20d_pct",
                    "benchmark_downside_defense_5d_pct",
                    "benchmark_downside_defense_20d_pct",
                ],
                "mae_beats_price": 2,
                "rank_ic_beats_price": 3,
                "economic_basket_beats_price": 3,
                "mae_beats_topology_shuffle": 3,
            },
        },
        "interpretation": {
            "pass_is_exploratory_not_deployment_proof": True,
            "new_future_lockbox_required": True,
            "graph_diffusion_and_state_space_models_not_part_of_this_canary": True,
            "generative_denoising_diffusion_not_part_of_this_canary": True,
        },
        "prohibitions": {
            "date_center_absolute_flow": True,
            "table_48_breadth": True,
            "historical_holdings_imputation": True,
            "post_result_retuning": True,
            "new_fmp_features_in_this_canary": True,
            "gpu_use": True,
            "live_service_change": True,
        },
    }


def _fold_checkpoint_paths(output_root: Path, year: int) -> tuple[Path, Path]:
    return (
        output_root / f"fold_{year}.npz",
        output_root / f"fold_{year}.json",
    )


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_fold_checkpoint(
    *, output_root: Path, year: int, preregistration_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    npz_path, json_path = _fold_checkpoint_paths(output_root, year)
    if not npz_path.exists() or not json_path.exists():
        return None
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"fold {year} preregistration hash mismatch")
    with np.load(npz_path, allow_pickle=False) as item:
        arrays = {name: item[name].copy() for name in item.files}
    expected = {"actual", "date_codes", *MODEL_NAMES}
    if set(arrays) != expected:
        raise ValueError(f"fold {year} checkpoint fields mismatch")
    return arrays, metadata


def _save_fold_checkpoint(
    *,
    output_root: Path,
    year: int,
    preregistration_sha256: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    npz_path, json_path = _fold_checkpoint_paths(output_root, year)
    _write_npz_atomic(npz_path, arrays)
    payload = {
        **dict(metadata),
        "schema_version": "quant.etf_flow_v12.residual_canary_fold.v1",
        "outer_year": int(year),
        "preregistration_sha256": preregistration_sha256,
        "npz_sha256": sha256_file(npz_path),
        "generated_at_utc": utc_now(),
    }
    write_json_atomic(json_path, payload)
    return payload


def _fold_target_metrics(
    actual: np.ndarray, predictions: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        result[target_name] = {
            name: regression_metrics(
                actual[:, target_index], prediction[:, target_index]
            )
            for name, prediction in predictions.items()
        }
    return result


def _pooled_receipts(
    *,
    actual_parts: Sequence[np.ndarray],
    date_code_parts: Sequence[np.ndarray],
    prediction_parts: Mapping[str, Sequence[np.ndarray]],
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual = np.concatenate(actual_parts)
    dates = np.concatenate(date_code_parts)
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        loss_target = target_name.startswith("loss_")
        pooled: dict[str, Any] = {}
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
            price_mae = pooled.get(PRICE_MODEL, {}).get("mae")
            if price_mae is not None:
                pooled[model_name]["relative_mae_improvement_vs_price_pct"] = (
                    (price_mae - pooled[model_name]["mae"]) / price_mae * 100.0
                )
        result[target_name] = {
            "rows": int(len(actual)),
            "pooled": pooled,
            "folds": [
                {
                    "outer_year": fold["outer_year"],
                    "train_rows": fold["train_rows"],
                    "test_rows": fold["test_rows"],
                    "train_date_count": fold["train_date_count"],
                    "test_date_count": fold["test_date_count"],
                    "train_end_signal_date": fold["train_end_signal_date"],
                    "test_start_signal_date": fold["test_start_signal_date"],
                    "test_end_signal_date": fold["test_end_signal_date"],
                    "purge_sessions": PURGE_SESSIONS,
                    "models": fold["target_metrics"][target_name],
                }
                for fold in folds
            ],
        }
    return result


def evaluate_matrix(
    *,
    matrix: Mapping[str, Any],
    output_root: Path,
    preregistration_sha256: str,
    thread_count: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    price = _sanitize_features(matrix["price_matrix"])
    flow = _sanitize_features(matrix["flow_matrix"])
    targets = np.asarray(matrix["targets"], dtype=np.float32)
    price_names = tuple(f"price::{name}" for name in matrix["price_names"])
    flow_names = tuple(f"flow::{name}" for name in matrix["flow_names"])
    full_names = price_names + flow_names
    full = np.column_stack([price, flow]).astype(np.float32)
    lag5 = np.column_stack(
        [
            price,
            lag_flow_by_symbol(
                flow,
                matrix["date_codes"],
                matrix["symbol_codes"],
                len(matrix["date_values"]),
                len(matrix["symbol_values"]),
                5,
            ),
        ]
    ).astype(np.float32)
    lag20 = np.column_stack(
        [
            price,
            lag_flow_by_symbol(
                flow,
                matrix["date_codes"],
                matrix["symbol_codes"],
                len(matrix["date_values"]),
                len(matrix["symbol_values"]),
                20,
            ),
        ]
    ).astype(np.float32)
    shuffled = np.column_stack(
        [
            price,
            topology_shuffle(
                flow,
                matrix["date_codes"],
                len(GLOBAL_FLOW_FIELDS),
                seed=RANDOM_SEED,
            ),
        ]
    ).astype(np.float32)
    variants = {
        "full": full,
        "lag5": lag5,
        "lag20": lag20,
        "topology_shuffle": shuffled,
    }

    actual_parts: list[np.ndarray] = []
    date_code_parts: list[np.ndarray] = []
    prediction_parts: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    fold_records: list[dict[str, Any]] = []
    for year in OUTER_YEARS:
        train, test = fold_indices(matrix, year)
        if len(train) < 50_000 or len(test) < 10_000:
            continue
        checkpoint = _load_fold_checkpoint(
            output_root=output_root,
            year=year,
            preregistration_sha256=preregistration_sha256,
        )
        if checkpoint is not None:
            arrays, metadata = checkpoint
            if not np.array_equal(arrays["actual"], targets[test]):
                raise ValueError(f"fold {year} checkpoint target mismatch")
            if not np.array_equal(arrays["date_codes"], matrix["date_codes"][test]):
                raise ValueError(f"fold {year} checkpoint date mismatch")
            predictions = {name: arrays[name] for name in MODEL_NAMES}
            metadata = {**metadata, "resumed": True}
            if progress:
                progress(
                    {
                        "stage": "v12_residual_fold_resumed",
                        "outer_year": year,
                        "test_rows": len(test),
                        "at_utc": utc_now(),
                    }
                )
        else:
            weights = date_balanced_weights(matrix["date_codes"], train)
            price_prediction, price_top, price_seconds = fit_predict_multioutput(
                features=price,
                targets=targets,
                train=train,
                test=test,
                weights=weights,
                feature_names=price_names,
                thread_count=thread_count,
            )
            raw_predictions: dict[str, np.ndarray] = {}
            top_features: dict[str, list[dict[str, Any]]] = {
                PRICE_MODEL: price_top
            }
            fit_seconds = {PRICE_MODEL: price_seconds}
            for variant_name, variant_matrix in variants.items():
                prediction, top, elapsed = fit_predict_multioutput(
                    features=variant_matrix,
                    targets=targets,
                    train=train,
                    test=test,
                    weights=weights,
                    feature_names=full_names,
                    thread_count=thread_count,
                )
                raw_predictions[variant_name] = prediction
                top_features[variant_name] = top
                fit_seconds[variant_name] = elapsed
                if progress:
                    progress(
                        {
                            "stage": "v12_residual_model_fit",
                            "outer_year": year,
                            "variant": variant_name,
                            "fit_seconds": elapsed,
                            "at_utc": utc_now(),
                        }
                    )
            caps = residual_caps(targets[train])
            predictions = {
                PRICE_MODEL: price_prediction,
                "full_flow_raw": raw_predictions["full"],
                PRIMARY_MODEL: capped_residual_prediction(
                    price_prediction, raw_predictions["full"], caps
                ),
                "lag5_capped_residual": capped_residual_prediction(
                    price_prediction, raw_predictions["lag5"], caps
                ),
                "lag20_capped_residual": capped_residual_prediction(
                    price_prediction, raw_predictions["lag20"], caps
                ),
                "topology_shuffle_capped_residual": capped_residual_prediction(
                    price_prediction, raw_predictions["topology_shuffle"], caps
                ),
            }
            train_codes = matrix["date_codes"][train]
            test_codes = matrix["date_codes"][test]
            metadata = {
                "outer_year": int(year),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_date_count": int(len(np.unique(train_codes))),
                "test_date_count": int(len(np.unique(test_codes))),
                "train_end_signal_date": matrix["date_values"][int(np.max(train_codes))],
                "test_start_signal_date": matrix["date_values"][int(np.min(test_codes))],
                "test_end_signal_date": matrix["date_values"][int(np.max(test_codes))],
                "date_weight_total_min": float(
                    min(
                        np.sum(weights[train_codes == code])
                        for code in np.unique(train_codes)
                    )
                ),
                "date_weight_total_max": float(
                    max(
                        np.sum(weights[train_codes == code])
                        for code in np.unique(train_codes)
                    )
                ),
                "residual_caps": {
                    target: float(caps[index])
                    for index, target in enumerate(TARGET_NAMES)
                },
                "fit_seconds": fit_seconds,
                "top_features": top_features,
                "target_metrics": _fold_target_metrics(targets[test], predictions),
                "resumed": False,
            }
            arrays = {
                "actual": targets[test],
                "date_codes": matrix["date_codes"][test],
                **predictions,
            }
            metadata = _save_fold_checkpoint(
                output_root=output_root,
                year=year,
                preregistration_sha256=preregistration_sha256,
                arrays=arrays,
                metadata=metadata,
            )
        actual_parts.append(targets[test])
        date_code_parts.append(matrix["date_codes"][test])
        for model_name, prediction in predictions.items():
            prediction_parts[model_name].append(prediction)
        fold_records.append(metadata)
        write_json_atomic(
            output_root / "run_state.json",
            {
                "status": "RUNNING",
                "stage": "outer_folds",
                "completed_outer_years": [fold["outer_year"] for fold in fold_records],
                "current_outer_year": year,
                "updated_at_utc": utc_now(),
            },
        )
        if progress:
            progress(
                {
                    "stage": "v12_residual_outer_fold_complete",
                    "outer_year": year,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "resumed": bool(metadata.get("resumed")),
                    "at_utc": utc_now(),
                }
            )
    if not actual_parts:
        raise ValueError("no eligible v12 residual canary folds")
    return (
        _pooled_receipts(
            actual_parts=actual_parts,
            date_code_parts=date_code_parts,
            prediction_parts=prediction_parts,
            folds=fold_records,
        ),
        fold_records,
    )


def summarize_gate(targets: Mapping[str, Any]) -> dict[str, Any]:
    counters = defaultdict(int)
    improvements: list[float] = []
    positive_fold_targets = 0
    outer_fold_targets = 0
    core_names = {
        "loss_5d_pct",
        "loss_20d_pct",
        "benchmark_downside_defense_5d_pct",
        "benchmark_downside_defense_20d_pct",
    }
    core = defaultdict(int)
    for target_name, target in targets.items():
        pooled = target["pooled"]
        primary = pooled[PRIMARY_MODEL]
        price = pooled[PRICE_MODEL]
        lag5 = pooled["lag5_capped_residual"]
        lag20 = pooled["lag20_capped_residual"]
        shuffled = pooled["topology_shuffle_capped_residual"]
        improvement = (price["mae"] - primary["mae"]) / price["mae"] * 100.0
        improvements.append(improvement)
        counters["mae_beats_price"] += primary["mae"] < price["mae"]
        counters["raw_full_beats_price"] += (
            pooled["full_flow_raw"]["mae"] < price["mae"]
        )
        counters["mae_beats_lag5"] += primary["mae"] < lag5["mae"]
        counters["mae_beats_lag20"] += primary["mae"] < lag20["mae"]
        counters["mae_beats_topology_shuffle"] += primary["mae"] < shuffled["mae"]
        counters["rank_ic_beats_price"] += (
            primary["mean_daily_rank_ic"] > price["mean_daily_rank_ic"]
        )
        counters["economic_basket_beats_price"] += (
            primary["economic_basket_value"] > price["economic_basket_value"]
        )
        if target_name in core_names:
            core["mae_beats_price"] += primary["mae"] < price["mae"]
            core["rank_ic_beats_price"] += (
                primary["mean_daily_rank_ic"] > price["mean_daily_rank_ic"]
            )
            core["economic_basket_beats_price"] += (
                primary["economic_basket_value"] > price["economic_basket_value"]
            )
            core["mae_beats_topology_shuffle"] += primary["mae"] < shuffled["mae"]
        for fold in target["folds"]:
            outer_fold_targets += 1
            positive_fold_targets += (
                fold["models"][PRIMARY_MODEL]["mae"]
                < fold["models"][PRICE_MODEL]["mae"]
            )

    mean_improvement = float(np.mean(improvements))
    worst_improvement = float(np.min(improvements))
    forecast_pass = (
        counters["mae_beats_price"] >= 8
        and mean_improvement > 0
        and worst_improvement >= -1.0
        and counters["mae_beats_lag5"] >= 8
        and counters["mae_beats_lag20"] >= 8
        and counters["mae_beats_topology_shuffle"] >= 8
        and positive_fold_targets >= 36
    )
    basket_pass = (
        counters["rank_ic_beats_price"] >= 8
        and counters["economic_basket_beats_price"] >= 8
        and counters["mae_beats_topology_shuffle"] >= 8
        and positive_fold_targets >= 36
    )
    avoidance_pass = (
        core["mae_beats_price"] >= 2
        and core["rank_ic_beats_price"] >= 3
        and core["economic_basket_beats_price"] >= 3
        and core["mae_beats_topology_shuffle"] >= 3
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
        "status": "V12_RESIDUAL_CANARY_PASS" if passed_paths else "V12_RESIDUAL_CANARY_FAIL",
        "passed_paths": passed_paths,
        "fixed_before_results": True,
        "exploratory_historical_oos_only": True,
        "checks": {
            "forecast_path_pass": forecast_pass,
            "basket_path_pass": basket_pass,
            "avoidance_path_pass": avoidance_pass,
            "mae_beats_price_8_of_12": counters["mae_beats_price"] >= 8,
            "mean_mae_improvement_positive": mean_improvement > 0,
            "worst_target_degradation_at_most_1pct": worst_improvement >= -1.0,
            "mae_beats_lag5_8_of_12": counters["mae_beats_lag5"] >= 8,
            "mae_beats_lag20_8_of_12": counters["mae_beats_lag20"] >= 8,
            "mae_beats_topology_shuffle_8_of_12": counters[
                "mae_beats_topology_shuffle"
            ]
            >= 8,
            "positive_half_outer_fold_targets": positive_fold_targets >= 36,
        },
        "counters": {
            **dict(counters),
            "mean_relative_mae_improvement_vs_price_pct": mean_improvement,
            "worst_relative_mae_improvement_vs_price_pct": worst_improvement,
            "positive_outer_fold_target_count": int(positive_fold_targets),
            "outer_fold_target_count": int(outer_fold_targets),
            "target_count": len(targets),
            "avoidance_core": dict(core),
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    preregistration_path = output_root / "v12_residual_canary_preregistration.json"
    frozen = preregistration()
    if preregistration_path.exists():
        existing = json.loads(preregistration_path.read_text(encoding="utf-8"))
        if existing != frozen:
            raise ValueError("existing v12 preregistration does not match source")
    else:
        write_json_atomic(preregistration_path, frozen)
    preregistration_sha256 = sha256_file(preregistration_path)
    if args.preregister_only:
        return preregistration_path, {
            "status": "PREREGISTERED",
            "preregistration_sha256": preregistration_sha256,
        }

    receipt_path = output_root / "v12_residual_canary_receipt.json"
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
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "RUNNING",
            "stage": "outer_folds",
            "started_at_utc": started_at,
            "matrix_ready_at_utc": utc_now(),
            "preregistration_sha256": preregistration_sha256,
            "scope": {
                "signal_date_count": len(matrix["date_values"]),
                "stock_row_count": len(matrix["targets"]),
                "stock_symbol_count": len(matrix["symbol_values"]),
            },
        },
    )
    targets, folds = evaluate_matrix(
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
        "folds": folds,
        "targets": targets,
        "gate": gate,
        "next_activation": (
            "ELIGIBLE_FOR_GRAPH_DIFFUSION_CANARY_NOT_DEPLOYMENT"
            if gate["passed_paths"]
            else "NO_PREDICTIVE_ACTIVATION_FROM_RESIDUAL_CANARY"
        ),
        "implementation_validity": {
            "date_balanced": True,
            "price_flow_lag_contract_preserved": True,
            "absolute_common_flow_date_centered": False,
            "table_48_breadth_used": False,
            "new_fmp_features_used": False,
            "existing_v11_outputs_modified": False,
        },
        "limitations": [
            "2021-2026 historical OOS informed successor design and is not a clean new lockbox",
            "this canary uses audited aggregate Flow channels, not ETF-identity graph message passing",
            "2018 and partial-2019 PIT holdings remain excluded rather than imputed",
            "a PASS requires a new future forward window before BF16/NVFP4 or trading use",
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
        summary = {"path": str(path), **payload}
        print(json.dumps(summary, indent=2, sort_keys=True))
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
    return 0 if payload["gate"]["status"] == "V12_RESIDUAL_CANARY_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
