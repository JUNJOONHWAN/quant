"""Post-result paired-date significance audit for the frozen v12 canary.

This audit does not tune or retrain a model and is not a new predictive gate.
It treats each signal date, rather than each stock row, as the independent
statistical unit. Positive paired error differences mean the frozen primary
Flow prediction has lower MAE than the named comparator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.phase_a import sha256_file, utc_now, write_json_atomic
from training.quant_flow_graph_v11_r2.phase_b_stock import OUTER_YEARS, TARGET_NAMES
from training.quant_flow_graph_v12.residual_canary import (
    DEFAULT_OUTPUT_ROOT,
    PRIMARY_MODEL,
)


SCHEMA_VERSION = "quant.etf_flow_v12.residual_posthoc_paired_date_audit.v1"
COMPARATORS = {
    "price_only": "price_only",
    "lag5": "lag5_capped_residual",
    "lag20": "lag20_capped_residual",
    "topology_shuffle": "topology_shuffle_capped_residual",
}
HAC_LAGS = (20, 60)
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_REPLICATIONS = 5000
RANDOM_SEED = 20260828


def date_means(date_codes: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted dates and the equal-weight mean of rows within each date."""

    dates = np.asarray(date_codes, dtype=np.int64)
    observations = np.asarray(values, dtype=np.float64)
    unique, inverse = np.unique(dates, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=observations).astype(np.float64)
    return unique, sums / counts


def hac_mean_test(values: np.ndarray, lag: int) -> dict[str, float]:
    """One-sided Newey-West test that the paired mean is greater than zero."""

    series = np.asarray(values, dtype=np.float64)
    count = len(series)
    mean = float(np.mean(series))
    centered = series - mean
    long_run = float(np.dot(centered, centered) / count)
    effective_lag = min(int(lag), count - 1)
    for offset in range(1, effective_lag + 1):
        covariance = float(
            np.dot(centered[offset:], centered[:-offset]) / count
        )
        weight = 1.0 - offset / (effective_lag + 1.0)
        long_run += 2.0 * weight * covariance
    long_run = max(long_run, 0.0)
    standard_error = math.sqrt(long_run / count)
    if standard_error > 0:
        z_score = mean / standard_error
        one_sided_p = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    else:
        z_score = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        one_sided_p = 0.0 if mean > 0 else (1.0 if mean < 0 else 0.5)
    return {
        "mean_paired_mae_reduction": mean,
        "hac_lag": int(effective_lag),
        "hac_standard_error": float(standard_error),
        "z_score": float(z_score),
        "one_sided_p_value": float(one_sided_p),
    }


def circular_block_indices(
    *, count: int, block: int, replications: int, seed: int
) -> np.ndarray:
    if count <= 0 or block <= 0 or replications <= 0:
        raise ValueError("count, block and replications must be positive")
    generator = np.random.default_rng(seed)
    block_count = math.ceil(count / block)
    starts = generator.integers(0, count, size=(replications, block_count))
    offsets = np.arange(block, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % count
    return indices.reshape(replications, -1)[:, :count]


def bootstrap_summary(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    samples = np.mean(series[indices], axis=1)
    return {
        "replications": int(len(samples)),
        "block_sessions": BOOTSTRAP_BLOCK,
        "mean": float(np.mean(series)),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "probability_mean_positive": float(np.mean(samples > 0.0)),
    }


def benjamini_hochberg(values: Mapping[str, float]) -> dict[str, float]:
    names = list(values)
    raw = np.asarray([values[name] for name in names], dtype=np.float64)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 1.0
    count = len(raw)
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(raw[index]) * count / rank)
        adjusted[index] = min(running, 1.0)
    return {name: float(adjusted[index]) for index, name in enumerate(names)}


def load_daily_differences(
    output_root: Path,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    differences: dict[str, dict[str, list[np.ndarray]]] = {
        comparator: {target: [] for target in TARGET_NAMES}
        for comparator in COMPARATORS
    }
    comparator_mae: dict[str, dict[str, list[np.ndarray]]] = {
        comparator: {target: [] for target in TARGET_NAMES}
        for comparator in COMPARATORS
    }
    checkpoints: list[dict[str, Any]] = []
    date_parts: list[np.ndarray] = []
    for year in OUTER_YEARS:
        npz_path = output_root / f"fold_{year}.npz"
        metadata_path = output_root / f"fold_{year}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(npz_path) != metadata["npz_sha256"]:
            raise ValueError(f"fold {year} npz hash mismatch")
        with np.load(npz_path, allow_pickle=False) as item:
            actual = item["actual"].astype(np.float64)
            dates = item["date_codes"].astype(np.int64)
            primary = item[PRIMARY_MODEL].astype(np.float64)
            predictions = {
                name: item[field].astype(np.float64)
                for name, field in COMPARATORS.items()
            }
        unique_dates = np.unique(dates)
        date_parts.append(unique_dates)
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
                    raise ValueError(f"fold {year} date aggregation mismatch")
                _, daily_baseline_mae = date_means(dates, baseline_error)
                differences[comparator_name][target_name].append(daily_difference)
                comparator_mae[comparator_name][target_name].append(daily_baseline_mae)
        checkpoints.append(
            {
                "outer_year": year,
                "npz_path": str(npz_path),
                "npz_sha256": metadata["npz_sha256"],
                "date_count": int(len(unique_dates)),
                "row_count": int(len(dates)),
            }
        )
    all_dates = np.concatenate(date_parts)
    if len(np.unique(all_dates)) != len(all_dates):
        raise ValueError("outer fold dates overlap")
    merged = {
        comparator: {
            target: np.concatenate(parts)
            for target, parts in target_map.items()
        }
        for comparator, target_map in differences.items()
    }
    merged_mae = {
        comparator: {
            target: np.concatenate(parts)
            for target, parts in target_map.items()
        }
        for comparator, target_map in comparator_mae.items()
    }
    return merged, merged_mae, checkpoints


def audit(output_root: Path) -> dict[str, Any]:
    receipt_path = output_root / "v12_residual_canary_receipt.json"
    preregistration_path = output_root / "v12_residual_canary_preregistration.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    differences, comparator_mae, checkpoints = load_daily_differences(output_root)
    date_count = len(next(iter(differences["price_only"].values())))
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
                "date_balanced_primary_mae": float(
                    np.mean(baseline) - np.mean(values)
                ),
                "relative_mae_improvement_pct": float(
                    np.mean(values) / np.mean(baseline) * 100.0
                ),
                "positive_date_ratio": float(np.mean(values > 0.0)),
                "hac": {
                    str(lag): hac_mean_test(values, lag) for lag in HAC_LAGS
                },
                "moving_block_bootstrap": bootstrap_summary(
                    values, bootstrap_indices
                ),
            }
        targets[target_name] = comparisons

    raw_p = {
        target_name: targets[target_name]["price_only"]["hac"]["60"][
            "one_sided_p_value"
        ]
        for target_name in TARGET_NAMES
    }
    fdr = benjamini_hochberg(raw_p)
    for target_name in TARGET_NAMES:
        targets[target_name]["price_only"]["hac60_bh_fdr_q_value"] = fdr[target_name]

    normalized = np.column_stack(
        [
            differences["price_only"][target]
            / np.mean(comparator_mae["price_only"][target])
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
            "random_seed": RANDOM_SEED,
            "multiple_testing": "Benjamini-Hochberg over 12 HAC60 price comparisons",
        },
        "composite_price_comparison": {
            "definition": (
                "equal-target mean of each date's paired MAE reduction divided by "
                "that target's date-balanced price MAE, in percent"
            ),
            "hac": {str(lag): hac_mean_test(composite, lag) for lag in HAC_LAGS},
            "moving_block_bootstrap": bootstrap_summary(
                composite, bootstrap_indices
            ),
        },
        "targets": targets,
        "limitations": [
            "this is a post-result inferential audit and cannot create a clean lockbox",
            "overlapping 5/20-session labels are addressed with HAC and block resampling, not removed",
            "statistical significance does not establish implementable returns after turnover and costs",
        ],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = audit(Path(args.output_root))
    output_path = Path(args.output_root) / "v12_residual_posthoc_paired_date_audit.json"
    write_json_atomic(output_path, payload)
    print(
        json.dumps(
            {
                "status": "POSTHOC_AUDIT_COMPLETE",
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "source_gate_status": payload["source_gate_status"],
                "composite_price_comparison": payload[
                    "composite_price_comparison"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
