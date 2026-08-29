"""Exploratory, non-gating audit of the completed v14 lockbox predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.phase_a import sha256_file, utc_now, write_json_atomic
from training.quant_flow_graph_v11_r2.phase_b_stock import TARGET_NAMES
from training.quant_flow_graph_v14.forward_avoidance_lockbox import (
    ADAPTIVE_MODEL,
    CURRENT_MODEL,
    FIXED_MODEL,
    LAG5_MODEL,
    MODEL_NAMES,
    PRICE_MODEL,
    PRIMARY_TARGETS,
    PREREGISTRATION_SHA256,
    SHUFFLED_MODEL,
    TEST_DATES,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v14/"
    "forward_avoidance_lockbox_20260715_20260729"
)
COMPARISONS = (
    (ADAPTIVE_MODEL, CURRENT_MODEL),
    (ADAPTIVE_MODEL, PRICE_MODEL),
    (ADAPTIVE_MODEL, FIXED_MODEL),
    (ADAPTIVE_MODEL, LAG5_MODEL),
    (ADAPTIVE_MODEL, SHUFFLED_MODEL),
    (CURRENT_MODEL, PRICE_MODEL),
)


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    observations = int(positive) + int(negative)
    if observations <= 0:
        return math.nan
    tail = min(int(positive), int(negative))
    probability = sum(math.comb(observations, value) for value in range(tail + 1))
    return float(min(1.0, 2.0 * probability / (2**observations)))


def paired_daily_summary(model_error: np.ndarray, comparator_error: np.ndarray) -> dict[str, Any]:
    difference = np.asarray(comparator_error) - np.asarray(model_error)
    positive = int(np.sum(difference > 0.0))
    negative = int(np.sum(difference < 0.0))
    zero = int(np.sum(difference == 0.0))
    comparator_mean = float(np.mean(comparator_error))
    return {
        "date_balanced_mean_absolute_error_improvement": float(np.mean(difference)),
        "date_balanced_relative_mae_improvement_pct": float(
            np.mean(difference) / comparator_mean * 100.0
        ),
        "positive_dates": positive,
        "negative_dates": negative,
        "zero_dates": zero,
        "exact_two_sided_sign_p": exact_two_sided_sign_p(positive, negative),
        "daily_improvements": difference.tolist(),
    }


def audit(output_root: Path) -> dict[str, Any]:
    receipt_path = output_root / "v14_forward_avoidance_receipt.json"
    predictions_path = output_root / "v14_forward_avoidance_predictions.npz"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("v14 receipt preregistration mismatch")
    if receipt.get("predictions", {}).get("sha256") != sha256_file(predictions_path):
        raise ValueError("v14 predictions hash mismatch")
    with np.load(predictions_path, allow_pickle=False) as item:
        actual = item["actual"].copy()
        date_codes = item["date_codes"].copy()
        predictions = {name: item[name].copy() for name in MODEL_NAMES}
    if len(np.unique(date_codes)) != len(TEST_DATES):
        raise ValueError("v14 posthoc test-date count mismatch")

    target_results: dict[str, Any] = {}
    aggregate_counts = {
        f"{model}_beats_{comparator}_mae_targets": 0
        for model, comparator in COMPARISONS
    }
    aggregate_rank_counts = {
        f"{model}_beats_{comparator}_rank_ic_targets": 0
        for model, comparator in COMPARISONS
    }
    aggregate_basket_counts = {
        f"{model}_beats_{comparator}_economic_basket_targets": 0
        for model, comparator in COMPARISONS
    }
    relative_improvements: dict[str, list[float]] = {
        f"{model}_vs_{comparator}": [] for model, comparator in COMPARISONS
    }
    for target_index, target_name in enumerate(TARGET_NAMES):
        daily_errors: dict[str, np.ndarray] = {}
        for model_name in MODEL_NAMES:
            values = []
            for date_code in np.unique(date_codes):
                rows = np.flatnonzero(date_codes == date_code)
                values.append(
                    float(
                        np.mean(
                            np.abs(
                                predictions[model_name][rows, target_index]
                                - actual[rows, target_index]
                            )
                        )
                    )
                )
            daily_errors[model_name] = np.asarray(values, dtype=np.float64)
        comparisons: dict[str, Any] = {}
        receipt_models = receipt["targets"][target_name]["models"]
        for model, comparator in COMPARISONS:
            key = f"{model}_vs_{comparator}"
            summary = paired_daily_summary(
                daily_errors[model], daily_errors[comparator]
            )
            comparisons[key] = summary
            aggregate_counts[f"{model}_beats_{comparator}_mae_targets"] += (
                receipt_models[model]["mae"] < receipt_models[comparator]["mae"]
            )
            aggregate_rank_counts[
                f"{model}_beats_{comparator}_rank_ic_targets"
            ] += (
                receipt_models[model]["mean_daily_rank_ic"]
                > receipt_models[comparator]["mean_daily_rank_ic"]
            )
            aggregate_basket_counts[
                f"{model}_beats_{comparator}_economic_basket_targets"
            ] += (
                receipt_models[model]["economic_basket_value"]
                > receipt_models[comparator]["economic_basket_value"]
            )
            relative_improvements[key].append(
                (
                    receipt_models[comparator]["mae"]
                    - receipt_models[model]["mae"]
                )
                / receipt_models[comparator]["mae"]
                * 100.0
            )
        target_results[target_name] = {
            "test_dates": list(TEST_DATES),
            "daily_mae": {
                name: values.tolist() for name, values in daily_errors.items()
            },
            "comparisons": comparisons,
        }
    mean_relative = {
        key: float(np.mean(values)) for key, values in relative_improvements.items()
    }
    primary_composite: dict[str, Any] = {}
    for model, comparator in COMPARISONS:
        key = f"{model}_vs_{comparator}"
        per_target = []
        for target_name in PRIMARY_TARGETS:
            comparison = target_results[target_name]["comparisons"][key]
            comparator_daily = np.asarray(
                target_results[target_name]["daily_mae"][comparator]
            )
            difference = np.asarray(comparison["daily_improvements"])
            per_target.append(difference / np.maximum(comparator_daily, 1e-12))
        composite = np.mean(np.column_stack(per_target), axis=1)
        positive = int(np.sum(composite > 0.0))
        negative = int(np.sum(composite < 0.0))
        primary_composite[key] = {
            "mean_relative_daily_improvement_pct": float(np.mean(composite) * 100.0),
            "positive_dates": positive,
            "negative_dates": negative,
            "exact_two_sided_sign_p": exact_two_sided_sign_p(positive, negative),
            "daily_relative_improvements": composite.tolist(),
        }
    return {
        "schema_version": "quant.etf_flow_v14.forward_avoidance_posthoc_audit.v1",
        "generated_at_utc": utc_now(),
        "exploratory_not_gate": True,
        "gate_status_unchanged": receipt["gate"]["status"],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "receipt_sha256": sha256_file(receipt_path),
        "predictions_sha256": sha256_file(predictions_path),
        "scope": {
            "test_dates": list(TEST_DATES),
            "test_date_count": len(TEST_DATES),
            "test_rows": int(len(actual)),
            "target_count": len(TARGET_NAMES),
        },
        "aggregate_mae_target_counts": aggregate_counts,
        "aggregate_rank_ic_target_counts": aggregate_rank_counts,
        "aggregate_economic_basket_target_counts": aggregate_basket_counts,
        "mean_relative_mae_improvement_across_targets_pct": mean_relative,
        "primary_target_composite": primary_composite,
        "targets": target_results,
        "interpretation_contract": {
            "no_gate_change": True,
            "no_retuning": True,
            "eleven_dates_are_underpowered": True,
            "topology_or_state_attribution_requires_shuffle_fixed_and_lag_controls": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_root = Path(args.output_root)
    result = audit(output_root)
    path = output_root / "v14_forward_avoidance_posthoc_audit.json"
    write_json_atomic(path, result)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "gate_status_unchanged": result["gate_status_unchanged"],
                "aggregate_mae_target_counts": result["aggregate_mae_target_counts"],
                "mean_relative_mae_improvement_across_targets_pct": result[
                    "mean_relative_mae_improvement_across_targets_pct"
                ],
                "primary_target_composite": result["primary_target_composite"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
