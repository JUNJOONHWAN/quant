"""Walk-forward ablation, calibration, final training, and live scoring."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .contracts import (
    HORIZONS,
    MODEL_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    TIMING_CONTRACT,
    model_feature_columns,
)
from .io_utils import sha256_file, utc_now, write_json_atomic


DEFAULT_PANEL = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/panel.sqlite3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/models"
)
VARIANTS = (
    "price",
    "price_benchmark_flow_t3",
    "price_benchmark_flow",
    "price_all_etf_flow",
    "full",
)
TARGET_KINDS = ("return", "upside", "loss")
COMMON_RULE = (
    "all_etf_flow_weight_coverage >= 0.50 and "
    "all_etf_flow_observed_count >= 5"
)
ROUND_TRIP_COST_PCT = 0.20
RANDOM_SEED = 1729


def _json_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _target_column(kind: str, horizon: int) -> str:
    return f"{kind}_{horizon}d_pct"


def _load_panel(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        frame = pd.read_sql_query("SELECT * FROM panel ORDER BY signal_date,symbol", connection)
    for column in frame.columns:
        if column in {
            "signal_date",
            "price_date",
            "flow_date",
            "legacy_flow_date",
            "symbol",
            "benchmark",
            "membership_source",
        }:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["signal_year"] = frame["signal_date"].str[:4].astype(int)
    return frame, metadata


def _date_weights(dates: pd.Series) -> np.ndarray:
    counts = dates.value_counts()
    weights = dates.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float)
    return weights / np.mean(weights)


def _usable_features(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    result = []
    for column in columns:
        values = frame[column]
        if values.notna().sum() and values.nunique(dropna=True) > 1:
            result.append(column)
    if not result:
        raise ValueError("no non-constant model features")
    return result


def _new_model(max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def _fit_model(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: str,
    max_iter: int,
) -> tuple[HistGradientBoostingRegressor, list[str], tuple[float, float]]:
    usable = _usable_features(frame, features)
    x = frame.loc[:, usable].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    y = frame[target].to_numpy(dtype=float)
    lower, upper = np.quantile(y, [0.005, 0.995])
    clipped = np.clip(y, lower, upper)
    model = _new_model(max_iter)
    model.fit(x, clipped, sample_weight=_date_weights(frame["signal_date"]))
    return model, usable, (float(lower), float(upper))


def _predict(
    model: HistGradientBoostingRegressor,
    frame: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    x = frame.loc[:, features].replace([np.inf, -np.inf], np.nan).to_numpy(
        dtype=np.float32
    )
    return np.asarray(model.predict(x), dtype=float)


def _daily_ic(y: np.ndarray, prediction: np.ndarray, dates: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame({"date": dates.to_numpy(), "y": y, "p": prediction})
    correlations = []
    for _, group in frame.groupby("date", sort=False):
        if len(group) < 20 or group["y"].nunique() < 2 or group["p"].nunique() < 2:
            continue
        value = group["y"].corr(group["p"], method="spearman")
        if pd.notna(value):
            correlations.append(float(value))
    return {
        "date_count": len(correlations),
        "mean": float(np.mean(correlations)) if correlations else None,
        "median": float(np.median(correlations)) if correlations else None,
        "positive_fraction": (
            float(np.mean(np.asarray(correlations) > 0)) if correlations else None
        ),
    }


def _decile_metrics(
    y: np.ndarray, prediction: np.ndarray, dates: pd.Series
) -> dict[str, Any]:
    frame = pd.DataFrame({"date": dates.to_numpy(), "y": y, "p": prediction})
    frame["rank"] = frame.groupby("date")["p"].rank(method="first", pct=True)
    frame["decile"] = np.minimum(10, np.maximum(1, np.ceil(frame["rank"] * 10))).astype(int)
    means = frame.groupby("decile")["y"].mean().reindex(range(1, 11))
    monotonic = means.corr(pd.Series(range(1, 11), index=means.index), method="spearman")
    return {
        "mean_actual_by_predicted_decile": {
            str(index): _json_number(value) for index, value in means.items()
        },
        "top_minus_bottom_actual_pct": _json_number(means.loc[10] - means.loc[1]),
        "decile_monotonic_spearman": _json_number(monotonic),
    }


def _regression_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    dates: pd.Series,
    calibration_residuals: np.ndarray,
    kind: str,
) -> dict[str, Any]:
    q10, q90 = np.quantile(calibration_residuals, [0.10, 0.90])
    lower = prediction + q10
    upper = prediction + q90
    result: dict[str, Any] = {
        "row_count": len(y),
        "mae_pct": float(mean_absolute_error(y, prediction)),
        "rmse_pct": float(math.sqrt(mean_squared_error(y, prediction))),
        "bias_pct": float(np.mean(prediction - y)),
        "r2": float(r2_score(y, prediction)),
        "overall_spearman": _json_number(
            pd.Series(y).corr(pd.Series(prediction), method="spearman")
        ),
        "daily_spearman_ic": _daily_ic(y, prediction, dates),
        "prediction_interval": {
            "nominal_coverage": 0.80,
            "empirical_coverage": float(np.mean((y >= lower) & (y <= upper))),
            "mean_width_pct": float(np.mean(upper - lower)),
            "calibration_residual_q10": float(q10),
            "calibration_residual_q90": float(q90),
        },
        "deciles": _decile_metrics(y, prediction, dates),
    }
    if kind == "return":
        result["directional_accuracy"] = float(
            np.mean((prediction >= 0.0) == (y >= 0.0))
        )
    return result


def _combined_metrics(
    frame: pd.DataFrame,
    horizon: int,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    data = pd.DataFrame(
        {
            "date": frame["signal_date"].to_numpy(),
            "actual_return": frame[_target_column("return", horizon)].to_numpy(float),
            "actual_loss": frame[_target_column("loss", horizon)].to_numpy(float),
            "pred_return": predictions["return"],
            "pred_upside": np.maximum(predictions["upside"], 0.0),
            "pred_loss": np.maximum(predictions["loss"], 0.0),
        }
    )
    data["score"] = (
        data["pred_return"] + 0.25 * data["pred_upside"] - 0.50 * data["pred_loss"]
    )
    data["score_rank"] = data.groupby("date")["score"].rank(method="first", pct=True)
    data["loss_rank"] = data.groupby("date")["pred_loss"].rank(method="first", pct=True)
    top = data[data["score_rank"] > 0.90]
    bottom = data[data["score_rank"] <= 0.10]
    retained = data[data["loss_rank"] <= 0.75]
    return {
        "score_contract": "pred_return + 0.25*pred_upside - 0.50*pred_loss",
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "top_decile_row_count": len(top),
        "top_decile_gross_return_pct": float(top["actual_return"].mean()),
        "top_decile_net_return_pct": float(
            top["actual_return"].mean() - ROUND_TRIP_COST_PCT
        ),
        "top_decile_win_rate": float((top["actual_return"] > 0).mean()),
        "top_decile_realized_loss_pct": float(top["actual_loss"].mean()),
        "bottom_decile_return_pct": float(bottom["actual_return"].mean()),
        "top_minus_bottom_return_pct": float(
            top["actual_return"].mean() - bottom["actual_return"].mean()
        ),
        "universe_equal_weight_return_pct": float(data["actual_return"].mean()),
        "universe_realized_loss_pct": float(data["actual_loss"].mean()),
        "loss_filter_retained_fraction": float(len(retained) / len(data)),
        "loss_filter_realized_loss_pct": float(retained["actual_loss"].mean()),
        "loss_filter_loss_reduction_pct": float(
            data["actual_loss"].mean() - retained["actual_loss"].mean()
        ),
        "loss_filter_return_pct": float(retained["actual_return"].mean()),
    }


def _fold_masks(
    frame: pd.DataFrame, sessions: Sequence[str], test_year: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    calibration_dates = sorted(
        frame.loc[frame["signal_year"] == test_year - 1, "signal_date"].unique()
    )
    test_dates = sorted(
        frame.loc[frame["signal_year"] == test_year, "signal_date"].unique()
    )
    if len(calibration_dates) < 100 or len(test_dates) < 20:
        return None
    positions = {value: index for index, value in enumerate(sessions)}
    calibration_start = positions[calibration_dates[0]]
    test_start = positions[test_dates[0]]
    train_last = sessions[calibration_start - horizon]
    calibration_last = sessions[test_start - horizon]
    train = (frame["signal_date"] <= train_last).to_numpy()
    calibration = (
        (frame["signal_date"] >= calibration_dates[0])
        & (frame["signal_date"] <= calibration_last)
    ).to_numpy()
    test = (frame["signal_year"] == test_year).to_numpy()
    if train.sum() < 20_000 or calibration.sum() < 10_000 or test.sum() < 1_000:
        return None
    return train, calibration, test


def _weighted_numeric_aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    weights = np.asarray([float(item.get("row_count") or 1.0) for item in items])
    result: dict[str, Any] = {"fold_count": len(items), "row_count": int(weights.sum())}
    keys = set.intersection(*(set(item) for item in items))
    for key in sorted(keys):
        if key == "row_count":
            continue
        values = [item.get(key) for item in items]
        if all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            result[key] = float(np.average(np.asarray(values, dtype=float), weights=weights))
        elif all(isinstance(value, Mapping) for value in values):
            result[key] = _weighted_numeric_aggregate(values)  # type: ignore[arg-type]
    return result


def _variant_score(aggregate: Mapping[str, Any]) -> float:
    targets = aggregate["targets"]
    combined = aggregate["combined"]
    ic = targets["return"].get("daily_spearman_ic", {}).get("mean") or 0.0
    loss_mae = targets["loss"].get("mae_pct") or 0.0
    return float(
        combined.get("top_decile_net_return_pct", 0.0)
        - 0.50 * combined.get("top_decile_realized_loss_pct", 0.0)
        + 10.0 * ic
        - 0.10 * loss_mae
    )


def _compare_variants(
    results: Mapping[str, Any], horizon: int, left: str, right: str
) -> dict[str, Any]:
    left_result = results[left][str(horizon)]["aggregate"]
    right_result = results[right][str(horizon)]["aggregate"]
    left_return = left_result["targets"]["return"]
    right_return = right_result["targets"]["return"]
    left_loss = left_result["targets"]["loss"]
    right_loss = right_result["targets"]["loss"]
    deltas = {
        "daily_return_ic": (
            (right_return["daily_spearman_ic"].get("mean") or 0.0)
            - (left_return["daily_spearman_ic"].get("mean") or 0.0)
        ),
        "top_decile_net_return_pct": (
            right_result["combined"].get("top_decile_net_return_pct", 0.0)
            - left_result["combined"].get("top_decile_net_return_pct", 0.0)
        ),
        "return_mae_pct": right_return.get("mae_pct", 0.0) - left_return.get("mae_pct", 0.0),
        "loss_mae_pct": right_loss.get("mae_pct", 0.0) - left_loss.get("mae_pct", 0.0),
        "loss_filter_reduction_pct": (
            right_result["combined"].get("loss_filter_loss_reduction_pct", 0.0)
            - left_result["combined"].get("loss_filter_loss_reduction_pct", 0.0)
        ),
    }
    favorable = sum(
        (
            deltas["daily_return_ic"] > 0,
            deltas["top_decile_net_return_pct"] > 0,
            deltas["return_mae_pct"] < 0,
            deltas["loss_mae_pct"] < 0,
            deltas["loss_filter_reduction_pct"] > 0,
        )
    )
    return {
        "baseline": left,
        "candidate": right,
        "deltas_candidate_minus_baseline": deltas,
        "favorable_metric_count_of_5": favorable,
        "verdict": "USEFUL" if favorable >= 3 else "NOT_PROVEN",
    }


def _save_joblib_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_and_train(
    *,
    panel_path: Path,
    output_root: Path,
    min_test_year: int,
    max_iter: int,
    replace: bool,
) -> dict[str, Any]:
    output_root = Path(output_root)
    report_path = output_root / "walk_forward_evaluation.json"
    checkpoint_path = output_root / "walk_forward_checkpoint.json"
    forecast_path = output_root / "forecasts_latest.csv"
    if report_path.exists() and not replace:
        raise FileExistsError(f"evaluation exists; pass --replace: {report_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    frame, panel_metadata = _load_panel(panel_path)
    common = frame[
        (frame["all_etf_flow_weight_coverage"] >= 0.50)
        & (frame["all_etf_flow_observed_count"] >= 5)
    ].copy()
    sessions = sorted(frame["signal_date"].unique())
    panel_sha256 = sha256_file(panel_path)
    all_results: dict[str, Any] = {variant: {} for variant in VARIANTS}
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("panel_sha256") == panel_sha256
            and checkpoint.get("max_iter") == max_iter
            and checkpoint.get("variants") == list(VARIANTS)
        ):
            stored = checkpoint.get("ablation_results")
            if isinstance(stored, dict):
                for variant in VARIANTS:
                    if isinstance(stored.get(variant), dict):
                        all_results[variant].update(stored[variant])
    eligible_years = sorted(
        int(year)
        for year in common["signal_year"].unique()
        if year >= min_test_year and year < 2100
    )
    for variant in VARIANTS:
        feature_contract = model_feature_columns(variant)
        for horizon in HORIZONS:
            if str(horizon) in all_results[variant]:
                continue
            target_columns = {
                kind: _target_column(kind, horizon) for kind in TARGET_KINDS
            }
            target_frame = common.dropna(subset=list(target_columns.values())).copy()
            fold_results = []
            for test_year in eligible_years:
                masks = _fold_masks(target_frame, sessions, int(test_year), horizon)
                if masks is None:
                    continue
                train_mask, calibration_mask, test_mask = masks
                train = target_frame.loc[train_mask]
                calibration = target_frame.loc[calibration_mask]
                test = target_frame.loc[test_mask]
                predictions = {}
                target_metrics = {}
                fit_audit = {}
                for kind, target in target_columns.items():
                    model, used_features, clip = _fit_model(
                        train, feature_contract, target, max_iter
                    )
                    calibration_prediction = _predict(model, calibration, used_features)
                    test_prediction = _predict(model, test, used_features)
                    residuals = calibration[target].to_numpy(float) - calibration_prediction
                    predictions[kind] = test_prediction
                    target_metrics[kind] = _regression_metrics(
                        test[target].to_numpy(float),
                        test_prediction,
                        test["signal_date"],
                        residuals,
                        kind,
                    )
                    fit_audit[kind] = {
                        "feature_count": len(used_features),
                        "label_clip_p005_p995": list(clip),
                    }
                fold_results.append(
                    {
                        "test_year": int(test_year),
                        "row_count": len(test),
                        "train_rows": len(train),
                        "calibration_rows": len(calibration),
                        "test_rows": len(test),
                        "train_signal_range": [
                            train["signal_date"].min(),
                            train["signal_date"].max(),
                        ],
                        "calibration_signal_range": [
                            calibration["signal_date"].min(),
                            calibration["signal_date"].max(),
                        ],
                        "test_signal_range": [
                            test["signal_date"].min(),
                            test["signal_date"].max(),
                        ],
                        "purge_sessions": horizon,
                        "fit_audit": fit_audit,
                        "targets": target_metrics,
                        "combined": _combined_metrics(test, horizon, predictions),
                    }
                )
            if not fold_results:
                raise RuntimeError(f"no valid walk-forward folds for {variant} h={horizon}")
            aggregate_targets = {
                kind: _weighted_numeric_aggregate(
                    [fold["targets"][kind] for fold in fold_results]
                )
                for kind in TARGET_KINDS
            }
            aggregate_combined = _weighted_numeric_aggregate(
                [
                    {**fold["combined"], "row_count": fold["row_count"]}
                    for fold in fold_results
                ]
            )
            aggregate = {
                "targets": aggregate_targets,
                "combined": aggregate_combined,
                "fold_count": len(fold_results),
                "test_rows": sum(fold["test_rows"] for fold in fold_results),
            }
            aggregate["selection_score"] = _variant_score(aggregate)
            all_results[variant][str(horizon)] = {
                "feature_contract": list(feature_contract),
                "folds": fold_results,
                "aggregate": aggregate,
            }
            write_json_atomic(
                checkpoint_path,
                {
                    "schema_version": "quant.spy_qqq_forecast_checkpoint.v2",
                    "generated_at_utc": utc_now(),
                    "panel_sha256": panel_sha256,
                    "max_iter": max_iter,
                    "variants": list(VARIANTS),
                    "ablation_results": all_results,
                },
            )

    selected = {}
    comparisons = {}
    for horizon in HORIZONS:
        ranked = sorted(
            VARIANTS,
            key=lambda name: all_results[name][str(horizon)]["aggregate"][
                "selection_score"
            ],
            reverse=True,
        )
        selected[str(horizon)] = {
            "variant": ranked[0],
            "selection_score": all_results[ranked[0]][str(horizon)]["aggregate"][
                "selection_score"
            ],
            "ranking": ranked,
        }
        comparisons[str(horizon)] = {
            "t2_vs_price": _compare_variants(
                all_results, horizon, "price", "price_benchmark_flow"
            ),
            "t2_vs_t3": _compare_variants(
                all_results,
                horizon,
                "price_benchmark_flow_t3",
                "price_benchmark_flow",
            ),
            "all_etf_vs_benchmark": _compare_variants(
                all_results,
                horizon,
                "price_benchmark_flow",
                "price_all_etf_flow",
            ),
            "fundamentals_vs_all_etf": _compare_variants(
                all_results, horizon, "price_all_etf_flow", "full"
            ),
        }

    latest_signal = frame["signal_date"].max()
    latest = frame[frame["signal_date"] == latest_signal].copy()
    forecast = latest[
        [
            "signal_date",
            "price_date",
            "flow_date",
            "symbol",
            "benchmark",
            "is_spy_member",
            "is_qqq_member",
            "reference_close",
            "all_etf_flow_observed_count",
            "all_etf_flow_weight_coverage",
        ]
    ].copy()
    model_manifest = {}
    for horizon in HORIZONS:
        variant = selected[str(horizon)]["variant"]
        features = model_feature_columns(variant)
        targets = {kind: _target_column(kind, horizon) for kind in TARGET_KINDS}
        labeled = common.dropna(subset=list(targets.values())).copy()
        labeled_dates = sorted(labeled["signal_date"].unique())
        if len(labeled_dates) < 400:
            raise RuntimeError("insufficient labeled dates for final model")
        calibration_start_index = max(1, len(labeled_dates) - 126)
        calibration_start = labeled_dates[calibration_start_index]
        positions = {value: index for index, value in enumerate(sessions)}
        train_last = sessions[positions[calibration_start] - horizon]
        calibration_train = labeled[labeled["signal_date"] <= train_last]
        calibration = labeled[labeled["signal_date"] >= calibration_start]
        bundle: dict[str, Any] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "timing_contract": TIMING_CONTRACT,
            "horizon_sessions": horizon,
            "variant": variant,
            "common_evaluation_rule": COMMON_RULE,
            "panel_sha256": sha256_file(panel_path),
            "models": {},
        }
        model_manifest[str(horizon)] = {
            "variant": variant,
            "labeled_rows": len(labeled),
            "labeled_signal_range": [
                labeled["signal_date"].min(),
                labeled["signal_date"].max(),
            ],
            "calibration_signal_range": [
                calibration["signal_date"].min(),
                calibration["signal_date"].max(),
            ],
            "targets": {},
        }
        live_predictions = {}
        for kind, target in targets.items():
            calibration_model, calibration_features, _ = _fit_model(
                calibration_train, features, target, max_iter
            )
            calibration_prediction = _predict(
                calibration_model, calibration, calibration_features
            )
            residuals = calibration[target].to_numpy(float) - calibration_prediction
            residual_quantiles = {
                "q10": float(np.quantile(residuals, 0.10)),
                "q50": float(np.quantile(residuals, 0.50)),
                "q90": float(np.quantile(residuals, 0.90)),
            }
            final_model, used_features, clip = _fit_model(
                labeled, features, target, max_iter
            )
            prediction = _predict(final_model, latest, used_features)
            if kind in {"upside", "loss"}:
                prediction = np.maximum(prediction, 0.0)
            live_predictions[kind] = prediction
            bundle["models"][kind] = {
                "target": target,
                "features": used_features,
                "label_clip_p005_p995": clip,
                "calibration_residual_quantiles": residual_quantiles,
                "estimator": final_model,
            }
            model_manifest[str(horizon)]["targets"][kind] = {
                "feature_count": len(used_features),
                "label_clip_p005_p995": list(clip),
                "calibration_residual_quantiles": residual_quantiles,
            }
            forecast[f"expected_{kind}_{horizon}d_pct"] = prediction
            forecast[f"{kind}_{horizon}d_p10_pct"] = prediction + residual_quantiles["q10"]
            forecast[f"{kind}_{horizon}d_p90_pct"] = prediction + residual_quantiles["q90"]
        forecast[f"utility_score_{horizon}d"] = (
            live_predictions["return"]
            + 0.25 * live_predictions["upside"]
            - 0.50 * live_predictions["loss"]
        )
        forecast[f"upside_to_loss_{horizon}d"] = np.divide(
            live_predictions["upside"],
            live_predictions["loss"],
            out=np.full_like(live_predictions["upside"], np.nan),
            where=live_predictions["loss"] > 0,
        )
        model_path = output_root / f"forecast_{horizon}d.joblib"
        _save_joblib_atomic(model_path, bundle)
        model_manifest[str(horizon)]["artifact"] = {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        }

    forecast["flow_quality"] = np.where(
        (forecast["all_etf_flow_weight_coverage"] >= 0.50)
        & (forecast["all_etf_flow_observed_count"] >= 5),
        "FULL",
        "LIMITED",
    )
    forecast = forecast.sort_values(
        ["utility_score_20d", "utility_score_5d"], ascending=False
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{forecast_path.name}.", dir=output_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        forecast.to_csv(temporary, index=False)
        os.replace(temporary, forecast_path)
    finally:
        temporary.unlink(missing_ok=True)
    forecast_json_path = output_root / "forecasts_latest.json"
    write_json_atomic(
        forecast_json_path,
        {
            "schema_version": "quant.spy_qqq_forecast_predictions.v2",
            "generated_at_utc": utc_now(),
            "signal_date": latest_signal,
            "timing_contract": TIMING_CONTRACT,
            "selected_variants": selected,
            "row_count": len(forecast),
            "rows": json.loads(forecast.to_json(orient="records")),
        },
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        "evaluation_design": {
            "walk_forward": "annual test folds; preceding year calibration; expanding training",
            "purge_rule": "target horizon sessions purged at train/calibration boundaries",
            "common_sample_rule": COMMON_RULE,
            "test_years_requested": eligible_years,
            "no_row_sampling": True,
            "label_training": "P0.5/P99.5 train-only winsorized squared-error conditional mean",
            "prediction_interval": "split-conformal residual P10/P90; empirical coverage reported",
            "transaction_cost_pct": ROUND_TRIP_COST_PCT,
            "model": {
                "class": "sklearn.ensemble.HistGradientBoostingRegressor",
                "max_iter": max_iter,
                "max_leaf_nodes": 31,
                "learning_rate": 0.05,
            },
        },
        "panel": {
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
            "schema_version": panel_metadata.get("schema_version"),
            "total_rows": len(frame),
            "common_rows": len(common),
            "signal_range": [frame["signal_date"].min(), frame["signal_date"].max()],
        },
        "ablation_results": all_results,
        "comparisons": comparisons,
        "selected_models": selected,
        "production_models": model_manifest,
        "latest_forecast": {
            "signal_date": latest_signal,
            "row_count": len(forecast),
            "csv_path": str(forecast_path),
            "csv_sha256": sha256_file(forecast_path),
            "json_path": str(forecast_json_path),
            "json_sha256": sha256_file(forecast_json_path),
        },
    }
    write_json_atomic(report_path, report)
    checkpoint_path.unlink(missing_ok=True)
    report["report_artifact"] = {
        "path": str(report_path),
        "bytes": report_path.stat().st_size,
        "sha256": sha256_file(report_path),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-test-year", type=int, default=2023)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_and_train(
        panel_path=args.panel,
        output_root=args.output_root,
        min_test_year=args.min_test_year,
        max_iter=args.max_iter,
        replace=args.replace,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
