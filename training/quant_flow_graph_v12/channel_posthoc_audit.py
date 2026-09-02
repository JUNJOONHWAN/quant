"""Paired-date posthoc audit for the frozen v12 channel ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.phase_a import sha256_file, utc_now, write_json_atomic
from training.quant_flow_graph_v11_r2.phase_b_stock import OUTER_YEARS, TARGET_NAMES
from training.quant_flow_graph_v12.channel_ablation import (
    DEFAULT_OUTPUT_ROOT,
    MODEL_NAMES,
)
from training.quant_flow_graph_v12.posthoc_audit import (
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_REPLICATIONS,
    HAC_LAGS,
    RANDOM_SEED,
    benjamini_hochberg,
    bootstrap_summary,
    circular_block_indices,
    date_means,
    hac_mean_test,
)


SCHEMA_VERSION = "quant.etf_flow_v12.channel_ablation_posthoc_paired_date.v1"
CONTRASTS = {
    "structure_vs_price": ("structure_mask_only", "price_only"),
    "current_dynamic_vs_price": ("current_dynamic_no_structure", "price_only"),
    "rolling_dynamic_vs_price": ("rolling_dynamic_no_structure", "price_only"),
    "all_dynamic_vs_price": ("all_dynamic_no_structure", "price_only"),
    "full_current_vs_price": ("full_current_no_rolling", "price_only"),
    "original_full_vs_price": ("original_full", "price_only"),
    "all_dynamic_vs_structure": (
        "all_dynamic_no_structure",
        "structure_mask_only",
    ),
    "rolling_dynamic_vs_current_dynamic": (
        "rolling_dynamic_no_structure",
        "current_dynamic_no_structure",
    ),
    "original_full_vs_all_dynamic": (
        "original_full",
        "all_dynamic_no_structure",
    ),
    "original_full_vs_structure": ("original_full", "structure_mask_only"),
    "original_full_vs_full_current": (
        "original_full",
        "full_current_no_rolling",
    ),
}


def load_daily_errors(
    output_root: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        model: {target: [] for target in TARGET_NAMES} for model in MODEL_NAMES
    }
    checkpoints: list[dict[str, Any]] = []
    all_dates: list[np.ndarray] = []
    for year in OUTER_YEARS:
        npz_path = output_root / f"fold_{year}.npz"
        metadata_path = output_root / f"fold_{year}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(npz_path) != metadata["npz_sha256"]:
            raise ValueError(f"channel fold {year} hash mismatch")
        with np.load(npz_path, allow_pickle=False) as item:
            actual = item["actual"].astype(np.float64)
            dates = item["date_codes"].astype(np.int64)
            predictions = {
                model: item[model].astype(np.float64) for model in MODEL_NAMES
            }
        unique_dates = np.unique(dates)
        all_dates.append(unique_dates)
        for target_index, target_name in enumerate(TARGET_NAMES):
            for model_name, prediction in predictions.items():
                error = np.abs(prediction[:, target_index] - actual[:, target_index])
                aggregated_dates, daily_error = date_means(dates, error)
                if not np.array_equal(aggregated_dates, unique_dates):
                    raise ValueError(f"channel fold {year} date mismatch")
                parts[model_name][target_name].append(daily_error)
        checkpoints.append(
            {
                "outer_year": year,
                "npz_sha256": metadata["npz_sha256"],
                "date_count": int(len(unique_dates)),
                "row_count": int(len(dates)),
            }
        )
    dates = np.concatenate(all_dates)
    if len(np.unique(dates)) != len(dates):
        raise ValueError("channel outer fold dates overlap")
    return (
        {
            model: {target: np.concatenate(values) for target, values in targets.items()}
            for model, targets in parts.items()
        },
        checkpoints,
    )


def audit(output_root: Path) -> dict[str, Any]:
    receipt_path = output_root / "v12_channel_ablation_receipt.json"
    preregistration_path = output_root / "v12_channel_ablation_preregistration.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    daily_errors, checkpoints = load_daily_errors(output_root)
    date_count = len(daily_errors["price_only"][TARGET_NAMES[0]])
    indices = circular_block_indices(
        count=date_count,
        block=BOOTSTRAP_BLOCK,
        replications=BOOTSTRAP_REPLICATIONS,
        seed=RANDOM_SEED,
    )
    contrasts: dict[str, Any] = {}
    for contrast_name, (primary_model, baseline_model) in CONTRASTS.items():
        target_results: dict[str, Any] = {}
        normalized_parts: list[np.ndarray] = []
        raw_p: dict[str, float] = {}
        for target_name in TARGET_NAMES:
            baseline = daily_errors[baseline_model][target_name]
            primary = daily_errors[primary_model][target_name]
            difference = baseline - primary
            normalized_parts.append(difference / np.mean(baseline))
            hac = {str(lag): hac_mean_test(difference, lag) for lag in HAC_LAGS}
            raw_p[target_name] = hac["60"]["one_sided_p_value"]
            target_results[target_name] = {
                "date_count": int(len(difference)),
                "date_balanced_primary_mae": float(np.mean(primary)),
                "date_balanced_baseline_mae": float(np.mean(baseline)),
                "relative_mae_improvement_pct": float(
                    np.mean(difference) / np.mean(baseline) * 100.0
                ),
                "positive_date_ratio": float(np.mean(difference > 0.0)),
                "hac": hac,
                "moving_block_bootstrap": bootstrap_summary(difference, indices),
            }
        adjusted = benjamini_hochberg(raw_p)
        for target_name in TARGET_NAMES:
            target_results[target_name]["hac60_bh_fdr_q_value"] = adjusted[target_name]
        composite = np.mean(np.column_stack(normalized_parts), axis=1) * 100.0
        contrasts[contrast_name] = {
            "primary_model": primary_model,
            "baseline_model": baseline_model,
            "target_hac60_bh_fdr_5pct_count": int(
                sum(value < 0.05 for value in adjusted.values())
            ),
            "target_block_ci_positive_count": int(
                sum(
                    result["moving_block_bootstrap"]["ci_2_5"] > 0.0
                    for result in target_results.values()
                )
            ),
            "composite": {
                "mean_relative_mae_improvement_pct": float(np.mean(composite)),
                "hac": {str(lag): hac_mean_test(composite, lag) for lag in HAC_LAGS},
                "moving_block_bootstrap": bootstrap_summary(composite, indices),
            },
            "targets": target_results,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "posthoc_after_attribution_results": True,
        "changes_models_or_predictions": False,
        "changes_attribution_labels": False,
        "source_receipt_path": str(receipt_path),
        "source_receipt_sha256": sha256_file(receipt_path),
        "source_preregistration_sha256": sha256_file(preregistration_path),
        "source_attribution_labels": receipt["attribution"]["labels"],
        "scope": {
            "date_count": int(date_count),
            "target_count": len(TARGET_NAMES),
            "outer_years": list(OUTER_YEARS),
            "checkpoints": checkpoints,
        },
        "method": {
            "independent_unit": "signal_date",
            "positive_contrast": "baseline_absolute_error_minus_primary_absolute_error",
            "hac_lags": list(HAC_LAGS),
            "moving_block_sessions": BOOTSTRAP_BLOCK,
            "moving_block_replications": BOOTSTRAP_REPLICATIONS,
            "random_seed": RANDOM_SEED,
            "multiple_testing": "BH FDR separately within each 12-target contrast",
        },
        "contrasts": contrasts,
        "limitations": [
            "posthoc inference cannot create a clean future lockbox",
            "channel significance is predictive attribution, not causal identification",
            "turnover, costs and portfolio constraints are not evaluated here",
        ],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_root = Path(args.output_root)
    payload = audit(output_root)
    output_path = output_root / "v12_channel_ablation_posthoc_paired_date_audit.json"
    write_json_atomic(output_path, payload)
    summary = {
        name: {
            "mean_relative_mae_improvement_pct": value["composite"][
                "mean_relative_mae_improvement_pct"
            ],
            "hac60_p": value["composite"]["hac"]["60"]["one_sided_p_value"],
            "block_ci": [
                value["composite"]["moving_block_bootstrap"]["ci_2_5"],
                value["composite"]["moving_block_bootstrap"]["ci_97_5"],
            ],
            "bh_fdr_target_count": value["target_hac60_bh_fdr_5pct_count"],
        }
        for name, value in payload["contrasts"].items()
    }
    print(
        json.dumps(
            {
                "status": "CHANNEL_POSTHOC_AUDIT_COMPLETE",
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "contrasts": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
