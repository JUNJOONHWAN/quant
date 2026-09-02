"""Select target-specific point models and combine them with basket rankers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .contracts import HORIZONS, MODEL_SCHEMA_VERSION, TIMING_CONTRACT, model_feature_columns
from .evaluate import (
    COMMON_RULE,
    TARGET_KINDS,
    VARIANTS,
    _fit_model,
    _load_panel,
    _predict,
    _save_joblib_atomic,
    _target_column,
)
from .io_utils import sha256_file, utc_now, write_json_atomic, write_text_atomic


DEFAULT_PANEL = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/panel.sqlite3"
)
DEFAULT_EVALUATION = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/models/"
    "walk_forward_evaluation.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_EVALUATION.parent


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _prediction_interval_bounds(
    prediction: np.ndarray,
    residual_quantiles: Mapping[str, float],
    *,
    nonnegative: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply calibrated residual bounds while preserving the target support."""

    p10 = prediction + float(residual_quantiles["q10"])
    p90 = prediction + float(residual_quantiles["q90"])
    if nonnegative:
        p10 = np.maximum(p10, 0.0)
        p90 = np.maximum(p90, 0.0)
    return p10, np.maximum(p90, p10)


def select_point_variants(
    evaluation: Mapping[str, Any], horizon: int
) -> dict[str, dict[str, Any]]:
    """Minimize equally weighted relative MAE and RMSE for each numeric target."""

    result = {}
    for kind in TARGET_KINDS:
        metrics = {
            variant: evaluation["ablation_results"][variant][str(horizon)][
                "aggregate"
            ]["targets"][kind]
            for variant in VARIANTS
        }
        best_mae = min(float(item["mae_pct"]) for item in metrics.values())
        best_rmse = min(float(item["rmse_pct"]) for item in metrics.values())
        ranking = []
        for variant, item in metrics.items():
            score = 0.5 * (float(item["mae_pct"]) / best_mae - 1.0) + 0.5 * (
                float(item["rmse_pct"]) / best_rmse - 1.0
            )
            ranking.append(
                {
                    "variant": variant,
                    "normalized_error_score": score,
                    "mae_pct": float(item["mae_pct"]),
                    "rmse_pct": float(item["rmse_pct"]),
                    "daily_spearman_ic": item["daily_spearman_ic"].get("mean"),
                }
            )
        ranking.sort(
            key=lambda row: (
                row["normalized_error_score"],
                -(row["daily_spearman_ic"] or -math.inf),
            )
        )
        result[kind] = {
            "variant": ranking[0]["variant"],
            "selection_rule": (
                "minimum 0.5*(MAE/best_MAE-1)+0.5*(RMSE/best_RMSE-1); "
                "daily IC breaks exact ties"
            ),
            "ranking": ranking,
        }
    return result


def _fit_point_target(
    *,
    frame: pd.DataFrame,
    common: pd.DataFrame,
    sessions: Sequence[str],
    horizon: int,
    kind: str,
    variant: str,
    max_iter: int,
) -> tuple[dict[str, Any], np.ndarray]:
    target = _target_column(kind, horizon)
    labeled = common.dropna(subset=[target]).copy()
    labeled_dates = sorted(labeled["signal_date"].unique())
    calibration_start = labeled_dates[max(1, len(labeled_dates) - 126)]
    positions = {value: index for index, value in enumerate(sessions)}
    train_last = sessions[positions[calibration_start] - horizon]
    calibration_train = labeled[labeled["signal_date"] <= train_last]
    calibration = labeled[labeled["signal_date"] >= calibration_start]
    features = model_feature_columns(variant)
    calibration_model, calibration_features, _ = _fit_model(
        calibration_train, features, target, max_iter
    )
    residuals = calibration[target].to_numpy(float) - _predict(
        calibration_model, calibration, calibration_features
    )
    residual_quantiles = {
        "q10": float(np.quantile(residuals, 0.10)),
        "q50": float(np.quantile(residuals, 0.50)),
        "q90": float(np.quantile(residuals, 0.90)),
    }
    final_model, used_features, clip = _fit_model(labeled, features, target, max_iter)
    latest_signal = frame["signal_date"].max()
    latest = frame[frame["signal_date"] == latest_signal]
    prediction = _predict(final_model, latest, used_features)
    if kind in {"upside", "loss"}:
        prediction = np.maximum(prediction, 0.0)
    return (
        {
            "target": target,
            "variant": variant,
            "features": used_features,
            "label_clip_p005_p995": clip,
            "calibration_residual_quantiles": residual_quantiles,
            "calibration_signal_range": [
                calibration["signal_date"].min(),
                calibration["signal_date"].max(),
            ],
            "labeled_signal_range": [
                labeled["signal_date"].min(),
                labeled["signal_date"].max(),
            ],
            "labeled_rows": len(labeled),
            "estimator": final_model,
        },
        prediction,
    )


def finalize_models(
    *,
    panel_path: Path,
    evaluation_path: Path,
    output_root: Path,
    max_iter: int,
) -> dict[str, Any]:
    evaluation = _load_json(evaluation_path)
    frame, panel_metadata = _load_panel(panel_path)
    common = frame[
        (frame["all_etf_flow_weight_coverage"] >= 0.50)
        & (frame["all_etf_flow_observed_count"] >= 5)
    ].copy()
    sessions = sorted(frame["signal_date"].unique())
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
    production: dict[str, Any] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for horizon in HORIZONS:
        point_selection = select_point_variants(evaluation, horizon)
        ranking_variant = evaluation["selected_models"][str(horizon)]["variant"]
        ranking_path = output_root / f"forecast_{horizon}d.joblib"
        ranking_bundle = joblib.load(ranking_path)
        if ranking_bundle.get("variant") != ranking_variant:
            raise RuntimeError("ranking artifact variant disagrees with evaluation")
        point_bundle: dict[str, Any] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "timing_contract": TIMING_CONTRACT,
            "role": "target_specific_point_forecast",
            "horizon_sessions": horizon,
            "selection": point_selection,
            "models": {},
        }
        point_predictions = {}
        ranking_predictions = {}
        for kind in TARGET_KINDS:
            variant = point_selection[kind]["variant"]
            if variant == ranking_variant:
                model_item = ranking_bundle["models"][kind]
                prediction = _predict(
                    model_item["estimator"], latest, model_item["features"]
                )
                item = {**model_item, "variant": variant, "reused_ranking_model": True}
            else:
                item, prediction = _fit_point_target(
                    frame=frame,
                    common=common,
                    sessions=sessions,
                    horizon=horizon,
                    kind=kind,
                    variant=variant,
                    max_iter=max_iter,
                )
                item["reused_ranking_model"] = False
            if kind in {"upside", "loss"}:
                prediction = np.maximum(prediction, 0.0)
            point_bundle["models"][kind] = item
            point_predictions[kind] = prediction
            quantiles = item["calibration_residual_quantiles"]
            p10, p90 = _prediction_interval_bounds(
                prediction,
                quantiles,
                nonnegative=kind in {"upside", "loss"},
            )
            forecast[f"expected_{kind}_{horizon}d_pct"] = prediction
            forecast[f"{kind}_{horizon}d_p10_pct"] = p10
            forecast[f"{kind}_{horizon}d_p90_pct"] = p90
            rank_item = ranking_bundle["models"][kind]
            rank_prediction = _predict(
                rank_item["estimator"], latest, rank_item["features"]
            )
            if kind in {"upside", "loss"}:
                rank_prediction = np.maximum(rank_prediction, 0.0)
            ranking_predictions[kind] = rank_prediction
            forecast[f"ranking_{kind}_{horizon}d_pct"] = rank_prediction
        forecast[f"ranking_score_{horizon}d"] = (
            ranking_predictions["return"]
            + 0.25 * ranking_predictions["upside"]
            - 0.50 * ranking_predictions["loss"]
        )
        forecast[f"expected_upside_to_loss_{horizon}d"] = np.divide(
            point_predictions["upside"],
            point_predictions["loss"],
            out=np.full_like(point_predictions["upside"], np.nan),
            where=point_predictions["loss"] > 0,
        )
        point_path = output_root / f"point_forecast_{horizon}d.joblib"
        _save_joblib_atomic(point_path, point_bundle)
        production[str(horizon)] = {
            "point_selection": point_selection,
            "point_model": {
                "path": str(point_path),
                "bytes": point_path.stat().st_size,
                "sha256": sha256_file(point_path),
            },
            "ranking_variant": ranking_variant,
            "ranking_model": {
                "path": str(ranking_path),
                "bytes": ranking_path.stat().st_size,
                "sha256": sha256_file(ranking_path),
            },
        }
    forecast["flow_quality"] = np.where(
        (forecast["all_etf_flow_weight_coverage"] >= 0.50)
        & (forecast["all_etf_flow_observed_count"] >= 5),
        "FULL",
        "LIMITED",
    )
    forecast = forecast.sort_values(
        ["ranking_score_20d", "ranking_score_5d"], ascending=False
    )
    csv_path = output_root / "forecasts_latest_final.csv"
    write_text_atomic(csv_path, forecast.to_csv(index=False))
    json_path = output_root / "forecasts_latest_final.json"
    write_json_atomic(
        json_path,
        {
            "schema_version": "quant.spy_qqq_forecast_predictions.final.v2",
            "generated_at_utc": utc_now(),
            "signal_date": latest_signal,
            "timing_contract": TIMING_CONTRACT,
            "row_count": len(forecast),
            "production_models": production,
            "rows": json.loads(forecast.to_json(orient="records")),
        },
    )
    manifest_path = output_root / "production_manifest.json"
    manifest = {
        "schema_version": "quant.spy_qqq_forecast_production.v2",
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        "point_model_role": "numeric expected return/upside/loss",
        "ranking_model_role": "basket ordering and avoidance score",
        "ranking_score_contract": "return + 0.25*upside - 0.50*loss",
        "common_evaluation_rule": COMMON_RULE,
        "panel": {
            "path": str(panel_path),
            "schema_version": panel_metadata.get("schema_version"),
            "sha256": sha256_file(panel_path),
        },
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": sha256_file(evaluation_path),
        },
        "production_models": production,
        "forecast": {
            "signal_date": latest_signal,
            "row_count": len(forecast),
            "csv_path": str(csv_path),
            "csv_sha256": sha256_file(csv_path),
            "json_path": str(json_path),
            "json_sha256": sha256_file(json_path),
            "full_quality_rows": int((forecast["flow_quality"] == "FULL").sum()),
            "limited_quality_rows": int((forecast["flow_quality"] == "LIMITED").sum()),
        },
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-iter", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = finalize_models(
        panel_path=args.panel,
        evaluation_path=args.evaluation,
        output_root=args.output_root,
        max_iter=args.max_iter,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
