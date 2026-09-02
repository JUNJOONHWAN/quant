"""Fixed paired-date audit of the sealed v16 global ETF Drift forecast.

This audit does not fit or tune a model.  It separates three claims that were
conflated by earlier full-graph gates:

* distribution edge: global ETF state improves individual-stock 5d/20d target
  calibration relative to a price-only forecast;
* current-timing edge: the current global state beats a five-session-lagged
  state;
* stock-selection edge: global state improves daily cross-sectional baskets.

All arrays are sealed v16 outer-year OOS predictions.  Paired inference uses
signal dates and a 20-session circular moving-block bootstrap.  The historical
period was already inspected, so a pass is exploratory and can authorize only
a future prospective shadow lockbox.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from training.quant_flow_graph_v18.orthogonal_diffusion import (
        ALL_YEARS,
        AXIS_SHUFFLE_CANDIDATE,
        BASE_MODEL,
        BOOTSTRAP_BLOCK_SESSIONS,
        BOOTSTRAP_REPLICATIONS,
        DATE_SHUFFLE_CANDIDATE,
        LAG5_CANDIDATE,
        PRICE_MODEL,
        TARGET_NAMES,
        V12_MODEL,
        _daily_gain_series,
        full_metrics,
        load_fold,
        moving_block_bootstrap,
        sha256_file,
        utc_now,
        write_json_atomic,
    )
except ModuleNotFoundError:  # local staging tests
    from quant_flow_graph_v18.orthogonal_diffusion import (
        ALL_YEARS,
        AXIS_SHUFFLE_CANDIDATE,
        BASE_MODEL,
        BOOTSTRAP_BLOCK_SESSIONS,
        BOOTSTRAP_REPLICATIONS,
        DATE_SHUFFLE_CANDIDATE,
        LAG5_CANDIDATE,
        PRICE_MODEL,
        TARGET_NAMES,
        V12_MODEL,
        _daily_gain_series,
        full_metrics,
        load_fold,
        moving_block_bootstrap,
        sha256_file,
        utc_now,
        write_json_atomic,
    )


SCHEMA_VERSION = "quant.etf_flow_v19.global_drift_audit.v1"
PREREGISTRATION_SCHEMA_VERSION = "quant.etf_flow_v19.global_drift_audit_preregistration.v1"
DEFAULT_V16_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v16/"
    "full_etf_identity_latent_walk_forward"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v19/"
    "global_drift_edge_audit"
)
PRIMARY_MODEL = BASE_MODEL
COMPARATORS = (
    PRICE_MODEL,
    V12_MODEL,
    DATE_SHUFFLE_CANDIDATE,
    LAG5_CANDIDATE,
    AXIS_SHUFFLE_CANDIDATE,
)
CORE_POTENTIAL_TARGETS = {
    "upside_5d_pct",
    "loss_5d_pct",
    "upside_20d_pct",
    "loss_20d_pct",
}
RANDOM_SEED = 20260830


def preregistration(v16_root: Path) -> dict[str, Any]:
    receipt_path = v16_root / "v16_full_etf_identity_receipt.json"
    prereg_path = v16_root / "v16_full_etf_identity_preregistration.json"
    if not receipt_path.exists() or not prereg_path.exists():
        raise FileNotFoundError("sealed v16 receipt or preregistration missing")
    folds = {
        str(year): {
            "npz_sha256": sha256_file(v16_root / f"fold_{year}.npz"),
            "json_sha256": sha256_file(v16_root / f"fold_{year}.json"),
        }
        for year in ALL_YEARS
    }
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "frozen_before_audit_execution": True,
        "historical_metrics_partly_inspected_before_this_audit": True,
        "purpose": (
            "separate a slow global ETF Drift distribution-calibration edge from "
            "current-timing and cross-sectional stock-selection claims"
        ),
        "scope": {
            "outer_years": list(ALL_YEARS),
            "targets": list(TARGET_NAMES),
            "primary": PRIMARY_MODEL,
            "comparators": list(COMPARATORS),
            "no_row_date_stock_target_or_etf_sampling": True,
            "no_model_refit": True,
        },
        "claims": {
            "distribution_edge": (
                "global full-ETF state improves individual-stock target MAE versus "
                "price-only, v12 aggregate Flow, and a more-capacious date-shuffled model"
            ),
            "current_timing_edge": (
                "current global state improves MAE versus the five-session-lagged full model"
            ),
            "basket_edge": (
                "global state improves daily cross-sectional economic basket values"
            ),
            "stock_topology_edge": "not tested and cannot be claimed by v19",
        },
        "paired_inference": {
            "unit": "signal date",
            "method": "circular moving-block bootstrap",
            "block_sessions": BOOTSTRAP_BLOCK_SESSIONS,
            "replications": BOOTSTRAP_REPLICATIONS,
            "seed": RANDOM_SEED,
            "overlapping_5d_20d_targets_acknowledged": True,
        },
        "fixed_gate": {
            "distribution_forecast": {
                "mae_beats_price_targets": 10,
                "mae_beats_v12_targets": 8,
                "mae_beats_date_shuffle_targets": 8,
                "mae_gain_vs_price_ci_lower_positive_targets": 8,
                "mae_gain_vs_date_shuffle_ci_lower_positive_targets": 6,
                "positive_fold_targets_vs_price": 48,
                "year_2025_and_2026_mean_gain_vs_price_nonnegative": True,
            },
            "upside_downside_potential": {
                "core_targets": sorted(CORE_POTENTIAL_TARGETS),
                "mae_beats_price_core_targets": 4,
                "mae_beats_v12_core_targets": 3,
                "mae_beats_date_shuffle_core_targets": 3,
                "mae_gain_vs_price_ci_lower_positive_core_targets": 3,
                "positive_fold_core_targets_vs_price": 16,
            },
            "current_timing": {
                "mae_beats_lag5_targets": 8,
                "mae_gain_vs_lag5_ci_lower_positive_targets": 6,
                "positive_fold_targets_vs_lag5": 42,
                "year_2025_and_2026_mean_gain_vs_lag5_nonnegative": True,
            },
            "basket": {
                "basket_beats_price_targets": 8,
                "basket_beats_date_shuffle_targets": 8,
                "basket_gain_vs_price_ci_lower_positive_targets": 4,
                "positive_fold_targets_vs_price": 42,
                "year_2025_and_2026_mean_gain_vs_price_nonnegative": True,
            },
        },
        "activation": {
            "deployment_forbidden": True,
            "trading_forbidden": True,
            "stock_topology_claim_forbidden": True,
            "current_flow_claim_requires_current_timing_path": True,
            "pass": "ELIGIBLE_FOR_FUTURE_PROSPECTIVE_SHADOW_LOCKBOX_ONLY",
            "fail": "NO_GLOBAL_DRIFT_EDGE_ACTIVATION",
        },
        "frozen_inputs": {
            "source_sha256": sha256_file(Path(__file__)),
            "v16_receipt_sha256": sha256_file(receipt_path),
            "v16_preregistration_sha256": sha256_file(prereg_path),
            "folds": folds,
        },
    }


def _paired_bootstrap(
    *,
    date_codes: np.ndarray,
    actual: np.ndarray,
    primary: np.ndarray,
    comparator: np.ndarray,
    comparator_index: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        mae_gain, basket_gain = _daily_gain_series(
            date_codes=date_codes,
            actual=actual,
            base=comparator,
            prediction=primary,
            target_index=target_index,
        )
        result[target_name] = {
            "mae_gain": moving_block_bootstrap(
                mae_gain,
                seed=RANDOM_SEED + comparator_index * 1_000 + target_index * 10,
            ),
            "basket_gain": moving_block_bootstrap(
                basket_gain,
                seed=RANDOM_SEED + comparator_index * 1_000 + target_index * 10 + 1,
            ),
        }
    return result


def _gate(
    *,
    pooled: Mapping[str, Mapping[str, Any]],
    yearly: Mapping[str, Mapping[str, Mapping[str, Any]]],
    bootstrap: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> dict[str, Any]:
    primary = pooled[PRIMARY_MODEL]
    counters: defaultdict[str, int] = defaultdict(int)
    core: defaultdict[str, int] = defaultdict(int)
    for target_name in TARGET_NAMES:
        p = primary[target_name]
        for comparator in COMPARATORS:
            c = pooled[comparator][target_name]
            counters[f"mae_beats_{comparator}"] += p["mae"] < c["mae"]
            counters[f"basket_beats_{comparator}"] += (
                p["economic_basket_value"] > c["economic_basket_value"]
            )
            counters[f"mae_ci_positive_{comparator}"] += (
                bootstrap[comparator][target_name]["mae_gain"]["ci_lower_95"] > 0.0
            )
            counters[f"basket_ci_positive_{comparator}"] += (
                bootstrap[comparator][target_name]["basket_gain"]["ci_lower_95"] > 0.0
            )
            if target_name in CORE_POTENTIAL_TARGETS:
                core[f"mae_beats_{comparator}"] += p["mae"] < c["mae"]
                core[f"mae_ci_positive_{comparator}"] += (
                    bootstrap[comparator][target_name]["mae_gain"]["ci_lower_95"] > 0.0
                )

    yearly_mean_mae: dict[str, dict[str, float]] = {
        comparator: {} for comparator in COMPARATORS
    }
    yearly_mean_basket: dict[str, dict[str, float]] = {
        comparator: {} for comparator in COMPARATORS
    }
    positive_fold_mae: defaultdict[str, int] = defaultdict(int)
    positive_fold_basket: defaultdict[str, int] = defaultdict(int)
    positive_core_fold_mae: defaultdict[str, int] = defaultdict(int)
    for year, models in yearly.items():
        for comparator in COMPARATORS:
            mae_values: list[float] = []
            basket_values: list[float] = []
            for target_name in TARGET_NAMES:
                p = models[PRIMARY_MODEL][target_name]
                c = models[comparator][target_name]
                mae_gain = (c["mae"] - p["mae"]) / c["mae"] * 100.0
                basket_gain = p["economic_basket_value"] - c["economic_basket_value"]
                mae_values.append(mae_gain)
                basket_values.append(basket_gain)
                positive_fold_mae[comparator] += mae_gain > 0.0
                positive_fold_basket[comparator] += basket_gain > 0.0
                if target_name in CORE_POTENTIAL_TARGETS:
                    positive_core_fold_mae[comparator] += mae_gain > 0.0
            yearly_mean_mae[comparator][year] = float(np.mean(mae_values))
            yearly_mean_basket[comparator][year] = float(np.mean(basket_values))

    late_price_mae = all(
        yearly_mean_mae[PRICE_MODEL].get(str(year), -math.inf) >= 0.0
        for year in (2025, 2026)
    )
    late_lag_mae = all(
        yearly_mean_mae[LAG5_CANDIDATE].get(str(year), -math.inf) >= 0.0
        for year in (2025, 2026)
    )
    late_price_basket = all(
        yearly_mean_basket[PRICE_MODEL].get(str(year), -math.inf) >= 0.0
        for year in (2025, 2026)
    )
    distribution = (
        counters[f"mae_beats_{PRICE_MODEL}"] >= 10
        and counters[f"mae_beats_{V12_MODEL}"] >= 8
        and counters[f"mae_beats_{DATE_SHUFFLE_CANDIDATE}"] >= 8
        and counters[f"mae_ci_positive_{PRICE_MODEL}"] >= 8
        and counters[f"mae_ci_positive_{DATE_SHUFFLE_CANDIDATE}"] >= 6
        and positive_fold_mae[PRICE_MODEL] >= 48
        and late_price_mae
    )
    potential = (
        core[f"mae_beats_{PRICE_MODEL}"] == len(CORE_POTENTIAL_TARGETS)
        and core[f"mae_beats_{V12_MODEL}"] >= 3
        and core[f"mae_beats_{DATE_SHUFFLE_CANDIDATE}"] >= 3
        and core[f"mae_ci_positive_{PRICE_MODEL}"] >= 3
        and positive_core_fold_mae[PRICE_MODEL] >= 16
    )
    current_timing = (
        counters[f"mae_beats_{LAG5_CANDIDATE}"] >= 8
        and counters[f"mae_ci_positive_{LAG5_CANDIDATE}"] >= 6
        and positive_fold_mae[LAG5_CANDIDATE] >= 42
        and late_lag_mae
    )
    basket = (
        counters[f"basket_beats_{PRICE_MODEL}"] >= 8
        and counters[f"basket_beats_{DATE_SHUFFLE_CANDIDATE}"] >= 8
        and counters[f"basket_ci_positive_{PRICE_MODEL}"] >= 4
        and positive_fold_basket[PRICE_MODEL] >= 42
        and late_price_basket
    )
    passed_paths = [
        name
        for name, passed in (
            ("DISTRIBUTION_FORECAST", distribution),
            ("UPSIDE_DOWNSIDE_POTENTIAL", potential),
            ("CURRENT_TIMING", current_timing),
            ("BASKET", basket),
        )
        if passed
    ]
    return {
        "status": "V19_GLOBAL_DRIFT_PASS" if passed_paths else "V19_GLOBAL_DRIFT_FAIL",
        "passed_paths": passed_paths,
        "fixed_before_execution": True,
        "historical_oos_not_fresh_forward_lockbox": True,
        "checks": {
            "distribution_forecast_pass": distribution,
            "upside_downside_potential_pass": potential,
            "current_timing_pass": current_timing,
            "basket_pass": basket,
            "late_2025_2026_vs_price_mae_nonnegative": late_price_mae,
            "late_2025_2026_vs_lag5_mae_nonnegative": late_lag_mae,
            "late_2025_2026_vs_price_basket_nonnegative": late_price_basket,
        },
        "counters": {
            **dict(counters),
            "core_potential": dict(core),
            "positive_fold_mae": dict(positive_fold_mae),
            "positive_fold_basket": dict(positive_fold_basket),
            "positive_core_fold_mae": dict(positive_core_fold_mae),
            "yearly_mean_mae_improvement_pct": yearly_mean_mae,
            "yearly_mean_basket_gain": yearly_mean_basket,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, Mapping[str, Any]]:
    v16_root = Path(args.v16_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prereg_path = output_root / "v19_global_drift_preregistration.json"
    receipt_path = output_root / "v19_global_drift_receipt.json"
    state_path = output_root / "run_state.json"
    if receipt_path.exists():
        raise FileExistsError(f"receipt already exists: {receipt_path}")
    proposed = preregistration(v16_root)
    if prereg_path.exists():
        if json.loads(prereg_path.read_text(encoding="utf-8")) != proposed:
            raise ValueError("existing v19 preregistration differs from source/input")
    else:
        write_json_atomic(prereg_path, proposed)
    prereg_sha = sha256_file(prereg_path)
    if args.preregister_only:
        return prereg_path, {"preregistration_sha256": prereg_sha}
    if not args.expected_prereg_sha or args.expected_prereg_sha != prereg_sha:
        raise ValueError("exact --expected-prereg-sha is required")

    started_at = utc_now()
    write_json_atomic(
        state_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "RUNNING",
            "started_at_utc": started_at,
            "preregistration_sha256": prereg_sha,
        },
    )
    fold_arrays: dict[int, dict[str, np.ndarray]] = {}
    input_receipts: dict[str, Any] = {}
    previous_max: int | None = None
    for year in ALL_YEARS:
        arrays, input_receipt = load_fold(v16_root, year)
        dates = np.asarray(arrays["date_codes"], dtype=np.int64)
        if previous_max is not None and int(np.min(dates)) <= previous_max:
            raise ValueError(f"overlapping/nonchronological fold {year}")
        previous_max = int(np.max(dates))
        fold_arrays[year] = arrays
        input_receipts[str(year)] = input_receipt

    yearly: dict[str, Any] = {}
    pooled_arrays: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    pooled_actual: list[np.ndarray] = []
    pooled_dates: list[np.ndarray] = []
    for year in ALL_YEARS:
        arrays = fold_arrays[year]
        actual = np.asarray(arrays["actual"], dtype=np.float32)
        dates = np.asarray(arrays["date_codes"], dtype=np.int64)
        models = {
            model: full_metrics(
                date_codes=dates,
                actual=actual,
                prediction=np.asarray(arrays[model], dtype=np.float32),
            )
            for model in (PRIMARY_MODEL, *COMPARATORS)
        }
        yearly[str(year)] = models
        pooled_actual.append(actual)
        pooled_dates.append(dates)
        for model in (PRIMARY_MODEL, *COMPARATORS):
            pooled_arrays[model].append(np.asarray(arrays[model], dtype=np.float32))

    actual_all = np.concatenate(pooled_actual)
    dates_all = np.concatenate(pooled_dates)
    combined = {model: np.concatenate(parts) for model, parts in pooled_arrays.items()}
    pooled = {
        model: full_metrics(
            date_codes=dates_all,
            actual=actual_all,
            prediction=prediction,
        )
        for model, prediction in combined.items()
    }
    bootstrap = {
        comparator: _paired_bootstrap(
            date_codes=dates_all,
            actual=actual_all,
            primary=combined[PRIMARY_MODEL],
            comparator=combined[comparator],
            comparator_index=index,
        )
        for index, comparator in enumerate(COMPARATORS)
    }
    gate = _gate(pooled=pooled, yearly=yearly, bootstrap=bootstrap)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "started_at_utc": started_at,
        "preregistration_sha256": prereg_sha,
        "source_sha256": sha256_file(Path(__file__)),
        "scope": {
            "outer_years": list(ALL_YEARS),
            "rows": int(len(actual_all)),
            "dates": int(len(np.unique(dates_all))),
            "targets": list(TARGET_NAMES),
            "no_sampling": True,
            "model_refit": False,
        },
        "inputs": input_receipts,
        "pooled_metrics": pooled,
        "yearly_metrics": yearly,
        "paired_date_block_bootstrap": bootstrap,
        "gate": gate,
        "implementation_validity": {
            "v16_hashes_verified": True,
            "sealed_oos_predictions_only": True,
            "chronological_nonoverlapping_folds": True,
            "no_model_fit_or_threshold_tuning": True,
            "paired_signal_date_inference": True,
        },
        "limitations": [
            "Historical v16 metrics were already inspected; this is a fixed exploratory classification audit, not a fresh lockbox.",
            "The date-shuffle comparator is the more-capacious full query model, so it is conservative but not an exact global-only width match.",
            "A distribution or potential pass does not prove current daily Flow freshness, stock topology, basket alpha, execution alpha, or causality.",
        ],
        "next_activation": (
            "FUTURE_PROSPECTIVE_GLOBAL_DRIFT_SHADOW_ONLY"
            if gate["passed_paths"]
            else "NO_GLOBAL_DRIFT_EDGE_ACTIVATION"
        ),
    }
    write_json_atomic(receipt_path, receipt)
    write_json_atomic(
        state_path,
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
    print(json.dumps(payload["gate"], sort_keys=True), flush=True)
    return 0 if payload["gate"]["passed_paths"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
