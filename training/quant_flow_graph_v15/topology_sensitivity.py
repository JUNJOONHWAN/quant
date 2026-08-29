"""Build and evaluate an isolated ETF-constituent topology sensitivity graph.

This is post-hoc data-quality research.  It preserves the sealed v14 prices,
targets, ETF Flow cube, training snapshots, model parameters, split, and random
seed.  Only the eleven v14 test snapshots may receive repaired constituent
edges.  A result from this module can explain sensitivity to missing holdings;
it can never replace or relabel the preregistered v14 verdict.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v14.forward_avoidance_lockbox import (
    ADAPTIVE_MODEL,
    DEFAULT_BASE_DATABASE,
    DEFAULT_ETFRADAR_ROOT,
    DEFAULT_FAMILY_REGISTRY,
    DEFAULT_INCREMENTAL_DATABASE,
    DEFAULT_OLD_EVENT_CUBE,
    DEFAULT_REPAIRED_FLOW_CACHE,
    EXPECTED_BASE_STAT,
    EXPECTED_HASHES,
    MODEL_NAMES,
    PRICE_MODEL,
    PRIMARY_TARGETS,
    SECONDARY_AVOIDANCE_TARGETS,
    TARGET_NAMES,
    TEST_DATES,
    TIMING_CONTRACT,
    _write_predictions,
    audit_test_window_relation_coverage,
    build_stock_matrix_from_sources,
    evaluate_predictions,
    fit_lockbox,
    open_union_source,
    readonly_connection,
    split_indices,
    summarize_gate,
    utc_now,
    write_json_atomic,
)

from .constituent_refresh import (
    DEFAULT_GRAPH_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPAIRED_GRAPH_ROOT,
    atomic_json,
    compare_graph_roots,
    sha256_file,
)


DEFAULT_HYBRID_ROOT = DEFAULT_OUTPUT_ROOT / "topology_only_graph_full_20180102_20260729"
DEFAULT_HYBRID_RECEIPT = DEFAULT_OUTPUT_ROOT / "v15_topology_only_graph_receipt.json"
DEFAULT_HYBRID_AUDIT = DEFAULT_OUTPUT_ROOT / "v15_topology_only_graph_impact_audit.json"
DEFAULT_V14_OUTPUT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v14/"
    "forward_avoidance_lockbox_20260715_20260729"
)
DEFAULT_SENSITIVITY_OUTPUT = DEFAULT_OUTPUT_ROOT / "topology_only_model_sensitivity"


def _manifest_rows(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["signal_date"]): row for row in manifest["snapshots"]}


def build_topology_only_graph(
    *,
    original_root: Path,
    repaired_root: Path,
    output_root: Path,
    receipt_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    original_root = Path(original_root).resolve()
    repaired_root = Path(repaired_root).resolve()
    output_root = Path(output_root).resolve()
    receipt_path = Path(receipt_path).resolve()
    audit_path = Path(audit_path).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    original_manifest_path = original_root / "manifest.json"
    repaired_manifest_path = repaired_root / "manifest.json"
    original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    repaired = json.loads(repaired_manifest_path.read_text(encoding="utf-8"))
    if original["etf_vocabulary"] != repaired["etf_vocabulary"]:
        raise ValueError("ETF vocabulary differs")
    original_rows = _manifest_rows(original)
    repaired_rows = _manifest_rows(repaired)
    missing = [date for date in TEST_DATES if date not in repaired_rows]
    if missing:
        raise ValueError(f"repaired graph missing test dates: {missing}")
    temporary = output_root.with_name(output_root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    hybrid = dict(original)
    hybrid_rows = []
    replaced_dates = []
    for row in original["snapshots"]:
        date = str(row["signal_date"])
        if date in TEST_DATES:
            replacement = dict(repaired_rows[date])
            for key in ("signal_date", "price_date", "flow_date", "stock_count"):
                if replacement.get(key) != row.get(key):
                    raise ValueError(f"snapshot identity mismatch for {date}: {key}")
            hybrid_rows.append(replacement)
            replaced_dates.append(date)
        else:
            hybrid_rows.append(dict(row))
    if tuple(replaced_dates) != TEST_DATES:
        raise ValueError(f"original graph test date mismatch: {replaced_dates}")
    hybrid["snapshots"] = hybrid_rows
    hybrid["edge_count"] = int(sum(int(row["edge_count"]) for row in hybrid_rows))
    hybrid["v15_topology_only_sensitivity"] = {
        "generated_at_utc": utc_now(),
        "posthoc": True,
        "original_manifest_sha256": sha256_file(original_manifest_path),
        "repaired_manifest_sha256": sha256_file(repaired_manifest_path),
        "replaced_test_dates": list(TEST_DATES),
        "training_snapshots_changed": False,
        "price_target_flow_cube_changed": False,
        "sealed_v14_verdict_changed": False,
    }
    atomic_json(temporary / "manifest.json", hybrid)
    for name in ("flow_values.npy", "flow_available_session_index.npy"):
        os.symlink(original_root / name, temporary / name)
    os.replace(temporary, output_root)
    audit = compare_graph_roots(original_root, output_root, TEST_DATES)
    if not audit["contracts"]["all_non_topology_arrays_equal"]:
        raise ValueError("hybrid changed a non-topology snapshot array")
    if not audit["contracts"]["all_model_flow_histories_equal"]:
        raise ValueError("hybrid changed the model-visible ETF Flow history")
    atomic_json(audit_path, audit)
    training_identity = all(
        dict(row) == dict(original_rows[str(row["signal_date"])])
        for row in hybrid_rows
        if str(row["signal_date"]) not in TEST_DATES
    )
    receipt = {
        "schema_version": "quant.etf_flow_v15.topology_only_graph.v1",
        "generated_at_utc": utc_now(),
        "ok": bool(training_identity),
        "original_graph": {
            "path": str(original_root),
            "manifest_sha256": sha256_file(original_manifest_path),
        },
        "repaired_test_graph": {
            "path": str(repaired_root),
            "manifest_sha256": sha256_file(repaired_manifest_path),
        },
        "hybrid_graph": {
            "path": str(output_root),
            "manifest_sha256": sha256_file(output_root / "manifest.json"),
            "flow_values_symlink": str((output_root / "flow_values.npy").resolve()),
            "flow_available_symlink": str(
                (output_root / "flow_available_session_index.npy").resolve()
            ),
        },
        "contracts": {
            "training_snapshot_rows_exactly_original": training_identity,
            "test_non_topology_arrays_equal": audit["contracts"][
                "all_non_topology_arrays_equal"
            ],
            "test_model_flow_histories_equal": audit["contracts"][
                "all_model_flow_histories_equal"
            ],
            "test_etf_vocabulary_equal": audit["contracts"]["etf_vocabulary_equal"],
            "test_changed_snapshot_count": audit["aggregate"][
                "changed_snapshot_count"
            ],
            "test_unchanged_snapshot_count": audit["aggregate"][
                "unchanged_snapshot_count"
            ],
        },
        "impact_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "interpretation_boundary": {
            "posthoc_data_quality_sensitivity": True,
            "changes_v14_clean_gate": False,
            "current_bulk_holdings_used": False,
            "only_historical_disclosure_overlay_edges_used": True,
        },
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _frozen_source_audit(args: argparse.Namespace) -> dict[str, Any]:
    base = Path(args.base_database)
    stat = base.stat()
    base_ok = (
        stat.st_size == EXPECTED_BASE_STAT["bytes"]
        and stat.st_mtime_ns == EXPECTED_BASE_STAT["mtime_ns"]
    )
    paths = {
        "incremental_database": Path(args.incremental_database),
        "repaired_flow_cache": Path(args.repaired_flow_cache),
        "old_event_cube": Path(args.old_event_cube),
        "family_registry": Path(args.family_registry),
        "graph_manifest": Path(args.original_graph_root) / "manifest.json",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_HASHES[name], "observed": digest}
        for name, digest in observed.items()
        if digest != EXPECTED_HASHES[name]
    }
    return {
        "passed": base_ok and not mismatches,
        "base_database": {
            "path": str(base),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "matches_v14": base_ok,
        },
        "observed_hashes": observed,
        "mismatches": mismatches,
    }


def _metric_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in TARGET_NAMES:
        models = {}
        for model in MODEL_NAMES:
            before = baseline[target]["models"][model]
            after = candidate[target]["models"][model]
            models[model] = {
                "mae_before": float(before["mae"]),
                "mae_after": float(after["mae"]),
                "mae_delta_after_minus_before": float(after["mae"] - before["mae"]),
                "mean_daily_rank_ic_before": float(before["mean_daily_rank_ic"]),
                "mean_daily_rank_ic_after": float(after["mean_daily_rank_ic"]),
                "economic_basket_before": float(before["economic_basket_value"]),
                "economic_basket_after": float(after["economic_basket_value"]),
            }
        result[target] = models
    return result


def run_topology_sensitivity(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "v15_topology_only_model_sensitivity_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    graph_receipt = json.loads(Path(args.hybrid_receipt).read_text(encoding="utf-8"))
    if not graph_receipt.get("ok"):
        raise ValueError("topology-only graph receipt is not valid")
    hybrid_manifest_sha = sha256_file(Path(args.hybrid_graph_root) / "manifest.json")
    if hybrid_manifest_sha != graph_receipt["hybrid_graph"]["manifest_sha256"]:
        raise ValueError("topology-only graph manifest hash mismatch")
    frozen = _frozen_source_audit(args)
    if not frozen["passed"]:
        raise ValueError(f"v14 frozen source mismatch: {frozen['mismatches']}")
    v14_receipt_path = Path(args.v14_output_root) / "v14_forward_avoidance_receipt.json"
    v14_predictions_path = Path(args.v14_output_root) / "v14_forward_avoidance_predictions.npz"
    baseline = json.loads(v14_receipt_path.read_text(encoding="utf-8"))
    if baseline["gate"]["passed"]:
        raise ValueError("expected sealed v14 FAIL receipt")
    event_path = Path(args.v14_output_root) / "v14_extended_flow_event_cube.sqlite3"
    event_receipt_path = Path(args.v14_output_root) / "v14_extended_flow_event_cube_receipt.json"
    event_receipt = json.loads(event_receipt_path.read_text(encoding="utf-8"))
    if sha256_file(event_path) != event_receipt["sha256"]:
        raise ValueError("sealed v14 extended event cube hash mismatch")
    started_at = utc_now()
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "RUNNING",
            "stage": "topology_only_stock_matrix",
            "started_at_utc": started_at,
        },
    )
    source = open_union_source(
        base_database=Path(args.base_database),
        incremental_database=Path(args.incremental_database),
        repaired_flow_cache=Path(args.repaired_flow_cache),
    )
    try:
        with readonly_connection(event_path) as event:
            matrix = build_stock_matrix_from_sources(
                event=event,
                source=source,
                graph_dataset_root=Path(args.hybrid_graph_root),
                progress=lambda payload: print(json.dumps(dict(payload), sort_keys=True), flush=True),
            )
        train, test, split = split_indices(matrix)
        relation_coverage = audit_test_window_relation_coverage(
            Path(args.hybrid_graph_root)
        )
        data_checks = {
            "sealed_v14_source_hashes_exact": frozen["passed"],
            "sealed_v14_event_cube_exact": True,
            "topology_only_graph_receipt_valid": True,
            "timing_violation_count_zero": int(matrix["timing_violation_count"]) == 0,
            "exact_11_test_dates": split["test_date_count"] == len(TEST_DATES),
            "complete_12_target_test_rows": bool(
                np.all(np.isfinite(np.asarray(matrix["targets"])[test]))
            ),
            "relation_stock_coverage_at_least_95pct": bool(relation_coverage["passed"]),
        }
        if not all(data_checks.values()):
            raise ValueError(f"v15 topology sensitivity data gate failed: {data_checks}")
        write_json_atomic(
            output_root / "run_state.json",
            {
                "status": "RUNNING",
                "stage": "frozen_models_no_retuning",
                "started_at_utc": started_at,
                "split": split,
                "updated_at_utc": utc_now(),
            },
        )
        predictions, diagnostics = fit_lockbox(
            matrix=matrix,
            train=train,
            test=test,
            thread_count=int(args.thread_count),
        )
    finally:
        source.close()
    actual = np.asarray(matrix["targets"], dtype=np.float32)[test]
    original_codes = np.asarray(matrix["date_codes"], dtype=np.int32)[test]
    unique_codes = np.unique(original_codes)
    recode = {int(value): index for index, value in enumerate(unique_codes)}
    date_codes = np.asarray([recode[int(value)] for value in original_codes], dtype=np.int16)
    symbol_codes = np.asarray(matrix["symbol_codes"], dtype=np.int32)[test]
    targets = evaluate_predictions(
        actual=actual,
        predictions=predictions,
        test_date_codes=date_codes,
    )
    counterfactual_gate = summarize_gate(targets=targets, data_checks=data_checks)
    with np.load(v14_predictions_path, allow_pickle=False) as prior:
        prediction_reproducibility = {
            model: {
                "shape_equal": list(prior[model].shape) == list(predictions[model].shape),
                "max_abs_delta": float(
                    np.max(np.abs(np.asarray(prior[model]) - predictions[model]))
                ),
                "exact_equal": bool(np.array_equal(prior[model], predictions[model])),
            }
            for model in MODEL_NAMES
        }
    if not prediction_reproducibility[PRICE_MODEL]["exact_equal"]:
        raise ValueError("price-only deterministic refit did not reproduce sealed v14")
    adaptive_deltas = [
        targets[target]["models"][ADAPTIVE_MODEL]["mae"]
        - baseline["targets"][target]["models"][ADAPTIVE_MODEL]["mae"]
        for target in TARGET_NAMES
    ]
    predictions_path = output_root / "v15_topology_only_model_sensitivity_predictions.npz"
    _write_predictions(
        predictions_path,
        actual=actual,
        predictions=predictions,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
    )
    status = (
        "V15_POSTHOC_TOPOLOGY_REPAIR_COUNTERFACTUAL_PASS"
        if counterfactual_gate["passed"]
        else "V15_POSTHOC_TOPOLOGY_REPAIR_STILL_FAIL"
    )
    receipt = {
        "schema_version": "quant.etf_flow_v15.topology_only_model_sensitivity.v1",
        "status": status,
        "started_at_utc": started_at,
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        "source_sha256": sha256_file(Path(__file__)),
        "frozen_source_audit": frozen,
        "inputs": {
            "sealed_v14_receipt": {
                "path": str(v14_receipt_path),
                "sha256": sha256_file(v14_receipt_path),
                "gate_status": baseline["gate"]["status"],
            },
            "sealed_v14_predictions": {
                "path": str(v14_predictions_path),
                "sha256": sha256_file(v14_predictions_path),
            },
            "sealed_v14_event_cube": {
                "path": str(event_path),
                "sha256": sha256_file(event_path),
            },
            "topology_only_graph_receipt": {
                "path": str(Path(args.hybrid_receipt)),
                "sha256": sha256_file(Path(args.hybrid_receipt)),
            },
            "topology_only_manifest_sha256": hybrid_manifest_sha,
        },
        "split": split,
        "scope": {
            "test_rows": int(len(test)),
            "test_dates": list(TEST_DATES),
            "target_count": len(TARGET_NAMES),
            "primary_targets": list(PRIMARY_TARGETS),
            "secondary_avoidance_targets": list(SECONDARY_AVOIDANCE_TARGETS),
            "no_row_or_symbol_sampling": True,
            "test_window_relation_coverage": relation_coverage,
        },
        "data_checks": data_checks,
        "fit_diagnostics": diagnostics,
        "prediction_reproducibility": prediction_reproducibility,
        "targets": targets,
        "metric_delta_vs_sealed_v14": _metric_delta(baseline["targets"], targets),
        "adaptive_summary": {
            "mae_improved_target_count": int(sum(value < 0 for value in adaptive_deltas)),
            "mae_worsened_target_count": int(sum(value > 0 for value in adaptive_deltas)),
            "mae_unchanged_target_count": int(sum(value == 0 for value in adaptive_deltas)),
            "mean_mae_delta_after_minus_before": float(np.mean(adaptive_deltas)),
        },
        "counterfactual_fixed_gate_checks": counterfactual_gate,
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
        },
        "interpretation": {
            "sealed_v14_verdict_remains": baseline["gate"]["status"],
            "this_is_a_clean_gate": False,
            "this_is_posthoc_data_quality_sensitivity": True,
            "retuning_performed": False,
            "current_bulk_holdings_used": False,
            "deployment_activation": False,
            "bf16_or_nvfp4_activation": False,
        },
    }
    write_json_atomic(receipt_path, receipt)
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "COMPLETE",
            "result_status": status,
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "completed_at_utc": utc_now(),
        },
    )
    return receipt_path, receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-graph")
    build.add_argument("--original-graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    build.add_argument("--repaired-graph-root", type=Path, default=DEFAULT_REPAIRED_GRAPH_ROOT)
    build.add_argument("--output-root", type=Path, default=DEFAULT_HYBRID_ROOT)
    build.add_argument("--receipt", type=Path, default=DEFAULT_HYBRID_RECEIPT)
    build.add_argument("--audit", type=Path, default=DEFAULT_HYBRID_AUDIT)
    run = commands.add_parser("run")
    run.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    run.add_argument("--incremental-database", type=Path, default=DEFAULT_INCREMENTAL_DATABASE)
    run.add_argument("--repaired-flow-cache", type=Path, default=DEFAULT_REPAIRED_FLOW_CACHE)
    run.add_argument("--old-event-cube", type=Path, default=DEFAULT_OLD_EVENT_CUBE)
    run.add_argument("--family-registry", type=Path, default=DEFAULT_FAMILY_REGISTRY)
    run.add_argument("--etfradar-root", type=Path, default=DEFAULT_ETFRADAR_ROOT)
    run.add_argument("--original-graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    run.add_argument("--hybrid-graph-root", type=Path, default=DEFAULT_HYBRID_ROOT)
    run.add_argument("--hybrid-receipt", type=Path, default=DEFAULT_HYBRID_RECEIPT)
    run.add_argument("--v14-output-root", type=Path, default=DEFAULT_V14_OUTPUT)
    run.add_argument("--output-root", type=Path, default=DEFAULT_SENSITIVITY_OUTPUT)
    run.add_argument("--thread-count", type=int, default=10)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build-graph":
        receipt = build_topology_only_graph(
            original_root=args.original_graph_root,
            repaired_root=args.repaired_graph_root,
            output_root=args.output_root,
            receipt_path=args.receipt,
            audit_path=args.audit,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    path, receipt = run_topology_sensitivity(args)
    print(
        json.dumps(
            {
                "receipt": str(path),
                "sha256": sha256_file(path),
                "status": receipt["status"],
                **receipt["adaptive_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
