"""Preregistered v12 attribution canary for Flow, structure and slow state.

The original v12 residual predictions are immutable inputs. This experiment
keeps the same CatBoost capacity, date weights and capped adapter while fitting
five mutually interpretable feature-channel variants. It asks whether the
observed edge belongs to dynamic fund Flow, slow rolling Flow state, structural
coverage/topology, or their interaction.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.contracts import (
    DEFAULT_SOURCE_DATABASE,
    TIMING_CONTRACT,
)
from training.quant_flow_graph_v11_r2.phase_a import (
    readonly_connection,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from training.quant_flow_graph_v11_r2.phase_b_stock import (
    DEFAULT_GRAPH_DATASET_ROOT,
    DEFAULT_PHASE_A_ROOT,
    DIRECT_MASK_FIELDS,
    GLOBAL_MASK_FIELDS,
    OUTER_YEARS,
    PURGE_SESSIONS,
    TARGET_NAMES,
    build_stock_matrix_from_sources,
    fold_indices,
    regression_metrics,
    stock_cross_sectional_metrics,
)
from training.quant_flow_graph_v12.residual_canary import (
    CATBOOST_PARAMETERS,
    CATBOOST_VERSION,
    DEFAULT_OUTPUT_ROOT as SOURCE_OUTPUT_ROOT,
    PRICE_MODEL,
    PRIMARY_MODEL,
    capped_residual_prediction,
    date_balanced_weights,
    fit_predict_multioutput,
    residual_caps,
)


SCHEMA_VERSION = "quant.etf_flow_v12.channel_ablation.v1"
PREREGISTRATION_SCHEMA_VERSION = "quant.etf_flow_v12.channel_ablation_preregistration.v1"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v12/"
    "channel_ablation_canary"
)
SOURCE_RECEIPT_SHA256 = "951e120a8a34df0ebf3d970fac0acb80d1deb7142249879e6e9c44036cc70598"
SOURCE_PREREGISTRATION_SHA256 = (
    "fd2efc9a723c33984dcc78fb0c5b525c071f087cd215fd10fce10c66ef36b16f"
)
STRUCTURAL_BASE_FIELDS = frozenset(
    GLOBAL_MASK_FIELDS
    + DIRECT_MASK_FIELDS
    + (
        "indirect_cluster_exposure_hhi",
    )
)
ROLLING_SUFFIXES = ("_mean_5", "_mean_20", "_z60", "_change_5")
VARIANT_NAMES = (
    "structure_mask_only",
    "current_dynamic_no_structure",
    "rolling_dynamic_no_structure",
    "all_dynamic_no_structure",
    "full_current_no_rolling",
)
MODEL_NAMES = (PRICE_MODEL, "original_full", *VARIANT_NAMES)


def _progress(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def rolling_base(name: str) -> str | None:
    for suffix in ROLLING_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def classify_feature_channels(flow_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    structure: list[int] = []
    current_dynamic: list[int] = []
    rolling_dynamic: list[int] = []
    full_current: list[int] = []
    for index, name in enumerate(flow_names):
        base = rolling_base(name)
        is_rolling = base is not None
        is_structure = name in STRUCTURAL_BASE_FIELDS or (
            base is not None and base in STRUCTURAL_BASE_FIELDS
        )
        if is_structure:
            structure.append(index)
        elif is_rolling:
            rolling_dynamic.append(index)
        else:
            current_dynamic.append(index)
        if not is_rolling:
            full_current.append(index)
    all_dynamic = current_dynamic + rolling_dynamic
    groups = {
        "structure_mask_only": tuple(structure),
        "current_dynamic_no_structure": tuple(current_dynamic),
        "rolling_dynamic_no_structure": tuple(rolling_dynamic),
        "all_dynamic_no_structure": tuple(all_dynamic),
        "full_current_no_rolling": tuple(full_current),
    }
    if set(groups["structure_mask_only"]) & set(groups["all_dynamic_no_structure"]):
        raise ValueError("structure and dynamic feature groups overlap")
    if set(groups["structure_mask_only"]) | set(groups["all_dynamic_no_structure"]) != set(
        range(len(flow_names))
    ):
        raise ValueError("structure plus dynamic groups do not cover all Flow features")
    if any(not indices for indices in groups.values()):
        raise ValueError(f"empty feature channel: {groups}")
    return groups


def preregistration() -> dict[str, Any]:
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "frozen_before_ablation_results": True,
        "source_predictive_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "source_predictive_preregistration_sha256": SOURCE_PREREGISTRATION_SHA256,
        "timing_contract": TIMING_CONTRACT,
        "scope": {
            "targets": list(TARGET_NAMES),
            "outer_years": list(OUTER_YEARS),
            "purge_sessions": PURGE_SESSIONS,
            "no_row_or_symbol_sampling": True,
        },
        "fixed_estimator": {
            "library": "catboost",
            "version": CATBOOST_VERSION,
            "parameters": CATBOOST_PARAMETERS,
            "date_balanced_total_weight": True,
            "residual_adapter_reused_unchanged": True,
        },
        "feature_channels": {
            "structure_mask_only": (
                "global availability/coverage/stale/zero masks, direct PIT connection/"
                "weight/coverage/age fields, indirect exposure HHI, including rolling "
                "derivatives whose base is structural"
            ),
            "current_dynamic_no_structure": (
                "non-rolling signed/rate/breadth/direct/indirect/relation Flow fields "
                "after structural fields are removed"
            ),
            "rolling_dynamic_no_structure": (
                "5/20-session means, z60 and change5 fields after structural lineages "
                "are removed"
            ),
            "all_dynamic_no_structure": (
                "union of current and rolling dynamic Flow, with all structural "
                "lineages removed"
            ),
            "full_current_no_rolling": (
                "all non-rolling Flow and structural fields; every rolling field removed"
            ),
        },
        "attribution_rules": {
            "dynamic_flow_edge": {
                "all_dynamic_beats_price_targets": 8,
                "all_dynamic_mean_improvement_vs_price_positive": True,
                "all_dynamic_beats_structure_targets": 8,
            },
            "structure_edge": {
                "structure_beats_price_targets": 8,
                "structure_mean_improvement_vs_price_positive": True,
            },
            "slow_state_dominant": {
                "rolling_dynamic_beats_current_dynamic_targets": 8,
            },
            "current_flow_edge": {
                "current_dynamic_beats_price_targets": 8,
                "current_dynamic_mean_improvement_vs_price_positive": True,
                "current_dynamic_beats_rolling_dynamic_targets": 8,
            },
            "full_requires_structure_interaction": {
                "original_full_beats_all_dynamic_targets": 8,
                "original_full_beats_structure_targets": 8,
            },
        },
        "interpretation": {
            "this_is_attribution_not_a_clean_lockbox": True,
            "does_not_change_original_predictive_gate": True,
            "does_not_activate_deployment_or_trading": True,
        },
        "prohibitions": {
            "post_result_hyperparameter_change": True,
            "new_fmp_features": True,
            "gpu_use": True,
            "date_center_absolute_flow": True,
            "table_48_breadth": True,
            "existing_receipt_modification": True,
        },
    }


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _fold_paths(output_root: Path, year: int) -> tuple[Path, Path]:
    return output_root / f"fold_{year}.npz", output_root / f"fold_{year}.json"


def _load_source_fold(year: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = SOURCE_OUTPUT_ROOT / f"fold_{year}.npz"
    metadata_path = SOURCE_OUTPUT_ROOT / f"fold_{year}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(npz_path) != metadata["npz_sha256"]:
        raise ValueError(f"source fold {year} hash mismatch")
    with np.load(npz_path, allow_pickle=False) as item:
        arrays = {
            "actual": item["actual"].copy(),
            "date_codes": item["date_codes"].copy(),
            PRICE_MODEL: item[PRICE_MODEL].copy(),
            "original_full": item[PRIMARY_MODEL].copy(),
        }
    return arrays, metadata


def _load_checkpoint(
    output_root: Path, year: int, preregistration_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    npz_path, json_path = _fold_paths(output_root, year)
    if not npz_path.exists() or not json_path.exists():
        return None
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"ablation fold {year} preregistration mismatch")
    if sha256_file(npz_path) != metadata.get("npz_sha256"):
        raise ValueError(f"ablation fold {year} npz hash mismatch")
    with np.load(npz_path, allow_pickle=False) as item:
        arrays = {name: item[name].copy() for name in item.files}
    if set(arrays) != {"actual", "date_codes", *MODEL_NAMES}:
        raise ValueError(f"ablation fold {year} fields mismatch")
    return arrays, metadata


def _save_checkpoint(
    *,
    output_root: Path,
    year: int,
    preregistration_sha256: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    npz_path, json_path = _fold_paths(output_root, year)
    _write_npz_atomic(npz_path, arrays)
    payload = {
        **dict(metadata),
        "schema_version": "quant.etf_flow_v12.channel_ablation_fold.v1",
        "preregistration_sha256": preregistration_sha256,
        "npz_sha256": sha256_file(npz_path),
        "generated_at_utc": utc_now(),
    }
    write_json_atomic(json_path, payload)
    return payload


def _fold_metrics(
    actual: np.ndarray, predictions: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    return {
        target_name: {
            model_name: regression_metrics(
                actual[:, target_index], prediction[:, target_index]
            )
            for model_name, prediction in predictions.items()
        }
        for target_index, target_name in enumerate(TARGET_NAMES)
    }


def _pooled_metrics(
    *,
    actual_parts: Sequence[np.ndarray],
    date_parts: Sequence[np.ndarray],
    prediction_parts: Mapping[str, Sequence[np.ndarray]],
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual = np.concatenate(actual_parts)
    dates = np.concatenate(date_parts)
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        pooled: dict[str, Any] = {}
        loss_target = target_name.startswith("loss_")
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
        result[target_name] = {
            "rows": int(len(actual)),
            "pooled": pooled,
            "folds": [
                {
                    "outer_year": fold["outer_year"],
                    "models": fold["target_metrics"][target_name],
                }
                for fold in folds
            ],
        }
    return result


def evaluate(
    *,
    matrix: Mapping[str, Any],
    output_root: Path,
    preregistration_sha256: str,
    thread_count: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    price = np.asarray(matrix["price_matrix"], dtype=np.float32)
    flow = np.asarray(matrix["flow_matrix"], dtype=np.float32)
    targets = np.asarray(matrix["targets"], dtype=np.float32)
    flow_names = tuple(str(name) for name in matrix["flow_names"])
    price_names = tuple(f"price::{name}" for name in matrix["price_names"])
    groups = classify_feature_channels(flow_names)
    variant_matrices = {
        name: np.column_stack([price, flow[:, indices]]).astype(np.float32)
        for name, indices in groups.items()
    }
    variant_feature_names = {
        name: price_names + tuple(f"flow::{flow_names[index]}" for index in indices)
        for name, indices in groups.items()
    }
    group_contract = {
        name: {
            "feature_count": len(indices),
            "features": [flow_names[index] for index in indices],
        }
        for name, indices in groups.items()
    }

    actual_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    prediction_parts: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    fold_records: list[dict[str, Any]] = []
    for year in OUTER_YEARS:
        train, test = fold_indices(matrix, year)
        if len(train) < 50_000 or len(test) < 10_000:
            continue
        source_arrays, source_metadata = _load_source_fold(year)
        if not np.array_equal(source_arrays["actual"], targets[test]):
            raise ValueError(f"source fold {year} target mismatch")
        if not np.array_equal(source_arrays["date_codes"], matrix["date_codes"][test]):
            raise ValueError(f"source fold {year} date mismatch")
        checkpoint = _load_checkpoint(output_root, year, preregistration_sha256)
        if checkpoint is not None:
            arrays, metadata = checkpoint
            predictions = {name: arrays[name] for name in MODEL_NAMES}
            metadata = {**metadata, "resumed": True}
        else:
            weights = date_balanced_weights(matrix["date_codes"], train)
            caps = residual_caps(targets[train])
            source_caps = np.asarray(
                [source_metadata["residual_caps"][target] for target in TARGET_NAMES],
                dtype=np.float32,
            )
            if not np.allclose(caps, source_caps, rtol=0.0, atol=1e-6):
                raise ValueError(f"source fold {year} residual cap mismatch")
            predictions = {
                PRICE_MODEL: source_arrays[PRICE_MODEL],
                "original_full": source_arrays["original_full"],
            }
            fit_seconds: dict[str, float] = {}
            top_features: dict[str, Any] = {}
            for variant_name in VARIANT_NAMES:
                raw, top, elapsed = fit_predict_multioutput(
                    features=variant_matrices[variant_name],
                    targets=targets,
                    train=train,
                    test=test,
                    weights=weights,
                    feature_names=variant_feature_names[variant_name],
                    thread_count=thread_count,
                )
                predictions[variant_name] = capped_residual_prediction(
                    predictions[PRICE_MODEL], raw, caps
                )
                fit_seconds[variant_name] = elapsed
                top_features[variant_name] = top
                del raw
                gc.collect()
                if progress:
                    progress(
                        {
                            "stage": "v12_channel_ablation_fit",
                            "outer_year": year,
                            "variant": variant_name,
                            "fit_seconds": elapsed,
                            "at_utc": utc_now(),
                        }
                    )
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
                "source_fold_npz_sha256": source_metadata["npz_sha256"],
                "fit_seconds": fit_seconds,
                "top_features": top_features,
                "target_metrics": _fold_metrics(targets[test], predictions),
                "resumed": False,
            }
            arrays = {
                "actual": targets[test],
                "date_codes": matrix["date_codes"][test],
                **predictions,
            }
            metadata = _save_checkpoint(
                output_root=output_root,
                year=year,
                preregistration_sha256=preregistration_sha256,
                arrays=arrays,
                metadata=metadata,
            )
        actual_parts.append(targets[test])
        date_parts.append(matrix["date_codes"][test])
        for model_name in MODEL_NAMES:
            prediction_parts[model_name].append(predictions[model_name])
        fold_records.append(metadata)
        write_json_atomic(
            output_root / "run_state.json",
            {
                "status": "RUNNING",
                "stage": "outer_folds",
                "completed_outer_years": [fold["outer_year"] for fold in fold_records],
                "updated_at_utc": utc_now(),
            },
        )
        if progress:
            progress(
                {
                    "stage": "v12_channel_ablation_fold_complete",
                    "outer_year": year,
                    "resumed": bool(metadata.get("resumed")),
                    "at_utc": utc_now(),
                }
            )
    if not fold_records:
        raise ValueError("no channel ablation folds")
    return (
        _pooled_metrics(
            actual_parts=actual_parts,
            date_parts=date_parts,
            prediction_parts=prediction_parts,
            folds=fold_records,
        ),
        fold_records,
        group_contract,
    )


def _beats(targets: Mapping[str, Any], left: str, right: str) -> int:
    return sum(
        target["pooled"][left]["mae"] < target["pooled"][right]["mae"]
        for target in targets.values()
    )


def _mean_improvement(targets: Mapping[str, Any], model: str, baseline: str) -> float:
    values = []
    for target in targets.values():
        model_mae = target["pooled"][model]["mae"]
        baseline_mae = target["pooled"][baseline]["mae"]
        values.append((baseline_mae - model_mae) / baseline_mae * 100.0)
    return float(np.mean(values))


def summarize_attribution(targets: Mapping[str, Any]) -> dict[str, Any]:
    counts = {
        "structure_beats_price": _beats(targets, "structure_mask_only", PRICE_MODEL),
        "current_dynamic_beats_price": _beats(
            targets, "current_dynamic_no_structure", PRICE_MODEL
        ),
        "rolling_dynamic_beats_price": _beats(
            targets, "rolling_dynamic_no_structure", PRICE_MODEL
        ),
        "all_dynamic_beats_price": _beats(
            targets, "all_dynamic_no_structure", PRICE_MODEL
        ),
        "full_current_beats_price": _beats(
            targets, "full_current_no_rolling", PRICE_MODEL
        ),
        "all_dynamic_beats_structure": _beats(
            targets, "all_dynamic_no_structure", "structure_mask_only"
        ),
        "rolling_dynamic_beats_current_dynamic": _beats(
            targets,
            "rolling_dynamic_no_structure",
            "current_dynamic_no_structure",
        ),
        "current_dynamic_beats_rolling_dynamic": _beats(
            targets,
            "current_dynamic_no_structure",
            "rolling_dynamic_no_structure",
        ),
        "original_full_beats_all_dynamic": _beats(
            targets, "original_full", "all_dynamic_no_structure"
        ),
        "original_full_beats_structure": _beats(
            targets, "original_full", "structure_mask_only"
        ),
        "original_full_beats_full_current": _beats(
            targets, "original_full", "full_current_no_rolling"
        ),
    }
    improvements = {
        model: _mean_improvement(targets, model, PRICE_MODEL)
        for model in ("original_full", *VARIANT_NAMES)
    }
    dynamic_flow_edge = (
        counts["all_dynamic_beats_price"] >= 8
        and improvements["all_dynamic_no_structure"] > 0
        and counts["all_dynamic_beats_structure"] >= 8
    )
    structure_edge = (
        counts["structure_beats_price"] >= 8
        and improvements["structure_mask_only"] > 0
    )
    slow_state_dominant = counts["rolling_dynamic_beats_current_dynamic"] >= 8
    current_flow_edge = (
        counts["current_dynamic_beats_price"] >= 8
        and improvements["current_dynamic_no_structure"] > 0
        and counts["current_dynamic_beats_rolling_dynamic"] >= 8
    )
    full_requires_structure_interaction = (
        counts["original_full_beats_all_dynamic"] >= 8
        and counts["original_full_beats_structure"] >= 8
    )
    labels = [
        label
        for label, passed in (
            ("DYNAMIC_FLOW_EDGE", dynamic_flow_edge),
            ("STRUCTURE_EDGE", structure_edge),
            ("SLOW_STATE_DOMINANT", slow_state_dominant),
            ("CURRENT_FLOW_EDGE", current_flow_edge),
            (
                "FULL_REQUIRES_STRUCTURE_INTERACTION",
                full_requires_structure_interaction,
            ),
        )
        if passed
    ]
    return {
        "status": "ATTRIBUTION_COMPLETE",
        "labels": labels,
        "checks": {
            "dynamic_flow_edge": dynamic_flow_edge,
            "structure_edge": structure_edge,
            "slow_state_dominant": slow_state_dominant,
            "current_flow_edge": current_flow_edge,
            "full_requires_structure_interaction": full_requires_structure_interaction,
        },
        "counters": counts,
        "mean_relative_mae_improvement_vs_price_pct": improvements,
        "graph_diffusion_activation": (
            "ELIGIBLE_FOR_GRAPH_DIFFUSION_ATTRIBUTION_CANARY"
            if dynamic_flow_edge or full_requires_structure_interaction
            else "NOT_ACTIVATED_BY_CHANNEL_ABLATION"
        ),
        "deployment_activation": "NOT_ACTIVATED",
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_receipt = SOURCE_OUTPUT_ROOT / "v12_residual_canary_receipt.json"
    source_preregistration = SOURCE_OUTPUT_ROOT / "v12_residual_canary_preregistration.json"
    if sha256_file(source_receipt) != SOURCE_RECEIPT_SHA256:
        raise ValueError("source residual receipt hash mismatch")
    if sha256_file(source_preregistration) != SOURCE_PREREGISTRATION_SHA256:
        raise ValueError("source residual preregistration hash mismatch")
    preregistration_path = output_root / "v12_channel_ablation_preregistration.json"
    frozen = preregistration()
    if preregistration_path.exists():
        existing = json.loads(preregistration_path.read_text(encoding="utf-8"))
        if existing != frozen:
            raise ValueError("existing channel preregistration does not match source")
    else:
        write_json_atomic(preregistration_path, frozen)
    preregistration_sha256 = sha256_file(preregistration_path)
    if args.preregister_only:
        return preregistration_path, {
            "status": "PREREGISTERED",
            "preregistration_sha256": preregistration_sha256,
        }
    receipt_path = output_root / "v12_channel_ablation_receipt.json"
    if receipt_path.exists() and not args.replace:
        raise FileExistsError(f"channel receipt already exists: {receipt_path}")

    phase_a_root = Path(args.phase_a_root)
    event_path = phase_a_root / "v11_r2_flow_event_cube.sqlite3"
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
    targets, folds, group_contract = evaluate(
        matrix=matrix,
        output_root=output_root,
        preregistration_sha256=preregistration_sha256,
        thread_count=args.thread_count,
        progress=_progress,
    )
    attribution = summarize_attribution(targets)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started_at,
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        "preregistration_sha256": preregistration_sha256,
        "source_sha256": sha256_file(Path(__file__)),
        "source_predictive_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "source_predictive_preregistration_sha256": SOURCE_PREREGISTRATION_SHA256,
        "scope": {
            "signal_date_start": matrix["date_values"][0],
            "signal_date_end": matrix["date_values"][-1],
            "signal_date_count": len(matrix["date_values"]),
            "stock_row_count": len(matrix["targets"]),
            "stock_symbol_count": len(matrix["symbol_values"]),
            "target_count": len(TARGET_NAMES),
            "timing_violation_count": matrix["timing_violation_count"],
            "no_row_or_symbol_sampling": True,
        },
        "feature_groups": group_contract,
        "folds": folds,
        "targets": targets,
        "attribution": attribution,
        "implementation_validity": {
            "original_predictions_reused_unchanged": True,
            "same_estimator_capacity": True,
            "date_balanced": True,
            "new_fmp_features_used": False,
            "gpu_used": False,
            "existing_receipt_modified": False,
        },
        "limitations": [
            "attribution uses the same historical period and is not a new lockbox",
            "feature ablation can identify useful channels but not a causal economic mechanism",
            "graph message passing and ETF identity remain outside this canary",
        ],
    }
    write_json_atomic(receipt_path, receipt)
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "COMPLETE",
            "attribution_labels": attribution["labels"],
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
        print(json.dumps({"path": str(path), **payload}, indent=2, sort_keys=True))
        return 0
    print(
        json.dumps(
            {
                "status": payload["attribution"]["status"],
                "path": str(path),
                "sha256": sha256_file(path),
                "scope": payload["scope"],
                "attribution": payload["attribution"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
