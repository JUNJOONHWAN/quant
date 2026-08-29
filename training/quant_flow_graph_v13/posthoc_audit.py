"""Paired-date inferential audit for the frozen v13 graph-state canary.

This script never trains, tunes, or changes a prediction. It is run only after
the predictive receipt exists and treats signal dates as the independent unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.phase_a import sha256_file, utc_now, write_json_atomic
from training.quant_flow_graph_v11_r2.phase_b_stock import OUTER_YEARS, TARGET_NAMES
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
from training.quant_flow_graph_v13.adaptive_graph_state import (
    DEFAULT_OUTPUT_ROOT,
    FIXED_MODEL,
    LAG5_MODEL,
    PRIMARY_MODEL,
    SHUFFLED_MODEL,
    SOURCE_CURRENT_MODEL,
    SOURCE_ORIGINAL_MODEL,
    SOURCE_PRICE_MODEL,
)


SCHEMA_VERSION = "quant.etf_flow_v13.adaptive_graph_state_posthoc_audit.v1"
COMPARATORS = {
    "source_current": SOURCE_CURRENT_MODEL,
    "source_original": SOURCE_ORIGINAL_MODEL,
    "fixed_state": FIXED_MODEL,
    "lag5": LAG5_MODEL,
    "topology_shuffle": SHUFFLED_MODEL,
    "price_only": SOURCE_PRICE_MODEL,
}


def load_daily_differences(
    output_root: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    differences = {
        comparator: {target: [] for target in TARGET_NAMES}
        for comparator in COMPARATORS
    }
    comparator_mae = {
        comparator: {target: [] for target in TARGET_NAMES}
        for comparator in COMPARATORS
    }
    checkpoints: list[dict[str, Any]] = []
    all_dates: list[np.ndarray] = []
    for year in OUTER_YEARS:
        npz_path = output_root / f"fold_{year}.npz"
        json_path = output_root / f"fold_{year}.json"
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if sha256_file(npz_path) != metadata["npz_sha256"]:
            raise ValueError(f"v13 fold {year} hash mismatch")
        with np.load(npz_path, allow_pickle=False) as item:
            actual = item["actual"].astype(np.float64)
            dates = item["date_codes"].astype(np.int64)
            primary = item[PRIMARY_MODEL].astype(np.float64)
            predictions = {
                name: item[field].astype(np.float64)
                for name, field in COMPARATORS.items()
            }
        unique_dates = np.unique(dates)
        all_dates.append(unique_dates)
        for target_index, target_name in enumerate(TARGET_NAMES):
            primary_error = np.abs(primary[:, target_index] - actual[:, target_index])
            for comparator_name, prediction in predictions.items():
                baseline_error = np.abs(
                    prediction[:, target_index] - actual[:, target_index]
                )
                daily_dates, daily_difference = date_means(
                    dates, baseline_error - primary_error
                )
                if not np.array_equal(daily_dates, unique_dates):
                    raise ValueError(f"v13 fold {year} date aggregation mismatch")
                _, daily_baseline = date_means(dates, baseline_error)
                differences[comparator_name][target_name].append(daily_difference)
                comparator_mae[comparator_name][target_name].append(daily_baseline)
        checkpoints.append(
            {
                "outer_year": year,
                "npz_path": str(npz_path),
                "npz_sha256": metadata["npz_sha256"],
                "date_count": int(len(unique_dates)),
                "row_count": int(len(dates)),
            }
        )
    merged_dates = np.concatenate(all_dates)
    if len(np.unique(merged_dates)) != len(merged_dates):
        raise ValueError("v13 outer fold dates overlap")
    return (
        {
            comparator: {
                target: np.concatenate(parts)
                for target, parts in target_map.items()
            }
            for comparator, target_map in differences.items()
        },
        {
            comparator: {
                target: np.concatenate(parts)
                for target, parts in target_map.items()
            }
            for comparator, target_map in comparator_mae.items()
        },
        checkpoints,
    )


def audit(output_root: Path) -> dict[str, Any]:
    receipt_path = output_root / "v13_adaptive_graph_state_receipt.json"
    preregistration_path = output_root / "v13_adaptive_graph_state_preregistration.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    differences, comparator_mae, checkpoints = load_daily_differences(output_root)
    date_count = len(next(iter(differences["source_current"].values())))
    bootstrap_indices = circular_block_indices(
        count=date_count,
        block=BOOTSTRAP_BLOCK,
        replications=BOOTSTRAP_REPLICATIONS,
        seed=RANDOM_SEED,
    )
    targets: dict[str, Any] = {}
    for target_name in TARGET_NAMES:
        comparisons: dict[str, Any] = {}
        for comparator_name in COMPARATORS:
            values = differences[comparator_name][target_name]
            baseline = comparator_mae[comparator_name][target_name]
            comparisons[comparator_name] = {
                "date_count": int(len(values)),
                "date_balanced_comparator_mae": float(np.mean(baseline)),
                "date_balanced_primary_mae": float(np.mean(baseline) - np.mean(values)),
                "relative_mae_improvement_pct": float(
                    np.mean(values) / np.mean(baseline) * 100.0
                ),
                "positive_date_ratio": float(np.mean(values > 0.0)),
                "hac": {str(lag): hac_mean_test(values, lag) for lag in HAC_LAGS},
                "moving_block_bootstrap": bootstrap_summary(values, bootstrap_indices),
            }
        targets[target_name] = comparisons

    raw_p = {
        target_name: targets[target_name]["source_current"]["hac"]["60"][
            "one_sided_p_value"
        ]
        for target_name in TARGET_NAMES
    }
    fdr = benjamini_hochberg(raw_p)
    for target_name in TARGET_NAMES:
        targets[target_name]["source_current"]["hac60_bh_fdr_q_value"] = fdr[
            target_name
        ]

    normalized = np.column_stack(
        [
            differences["source_current"][target]
            / np.mean(comparator_mae["source_current"][target])
            for target in TARGET_NAMES
        ]
    )
    composite = np.mean(normalized, axis=1) * 100.0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "posthoc_after_predictive_gate": True,
        "changes_model_or_prediction": False,
        "changes_original_gate": False,
        "source_receipt_path": str(receipt_path),
        "source_receipt_sha256": sha256_file(receipt_path),
        "source_preregistration_sha256": sha256_file(preregistration_path),
        "source_gate_status": receipt["gate"]["status"],
        "primary_model": PRIMARY_MODEL,
        "comparators": COMPARATORS,
        "scope": {
            "date_count": int(date_count),
            "target_count": len(TARGET_NAMES),
            "outer_years": list(OUTER_YEARS),
            "checkpoints": checkpoints,
        },
        "method": {
            "independent_unit": "signal_date",
            "paired_value": "comparator_absolute_error_minus_primary_absolute_error",
            "positive_value_means_primary_is_better": True,
            "hac_lags": list(HAC_LAGS),
            "moving_block_sessions": BOOTSTRAP_BLOCK,
            "moving_block_replications": BOOTSTRAP_REPLICATIONS,
            "multiple_testing": "BH over 12 HAC60 source-current comparisons",
        },
        "composite_source_current_comparison": {
            "hac": {str(lag): hac_mean_test(composite, lag) for lag in HAC_LAGS},
            "moving_block_bootstrap": bootstrap_summary(composite, bootstrap_indices),
        },
        "targets": targets,
        "limitations": [
            "post-result inference cannot create a clean future lockbox",
            "overlapping labels are addressed by HAC and block resampling",
            "significance does not establish implementable returns after costs",
        ],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = audit(Path(args.output_root))
    output_path = Path(args.output_root) / "v13_adaptive_graph_state_posthoc_audit.json"
    write_json_atomic(output_path, payload)
    print(
        json.dumps(
            {
                "status": "POSTHOC_AUDIT_COMPLETE",
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "source_gate_status": payload["source_gate_status"],
                "composite_source_current_comparison": payload[
                    "composite_source_current_comparison"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
