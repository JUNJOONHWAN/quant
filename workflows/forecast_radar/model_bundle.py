"""Fit and load the immutable Forecast RADAR shadow model bundle.

The final refit deliberately reproduces the v16/v19 estimator family.  It does
not turn the historical OOS audit into a fresh lockbox and therefore publishes a
shadow model only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.phase_a import readonly_connection
from training.quant_flow_graph_v11_r2.phase_b_stock import (
    MIN_RELATION_COVERAGE,
    build_stock_matrix_from_sources,
)
from training.quant_flow_graph_v12.residual_canary import (
    date_balanced_weights,
    residual_caps,
)
from training.quant_flow_graph_v16.full_etf_latent import (
    STATE_NAMES,
    build_full_etf_panel,
)

from .contracts import (
    CATBOOST_PARAMETERS,
    CATBOOST_VERSION,
    DEFAULT_BASE_DATABASE,
    DEFAULT_GRAPH_DATASET_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PHASE_A_ROOT,
    DEFAULT_V16_ROOT,
    DEFAULT_V19_RECEIPT,
    LATENT_COMPONENTS,
    MODEL_SCHEMA_VERSION,
    RANDOM_SEED,
    TARGET_NAMES,
    TIMING_CONTRACT,
)
from .io import sha256_file, utc_now, write_json_atomic


GLOBAL_MODEL_KEY = "full_etf_global_only"
PRICE_MODEL_KEY = "price_only"


def _factor_names(prefix: str = "global") -> tuple[str, ...]:
    return tuple(
        f"{prefix}::{state}::{component:02d}"
        for state in STATE_NAMES
        for component in range(LATENT_COMPONENTS)
    )


def _fit_latent_full(panel: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import TruncatedSVD

    current = np.asarray(panel["current"], dtype=np.float32)
    active = np.any(np.abs(current) > 0, axis=0)
    if int(np.sum(active)) <= LATENT_COMPONENTS:
        raise ValueError("full ETF panel cannot support the frozen latent width")
    model = TruncatedSVD(
        n_components=LATENT_COMPONENTS,
        algorithm="randomized",
        n_iter=7,
        random_state=RANDOM_SEED,
    )
    model.fit(current[:, active])
    components = np.zeros((LATENT_COMPONENTS, current.shape[1]), dtype=np.float32)
    components[:, active] = np.asarray(model.components_, dtype=np.float32)
    states = {
        "current": current,
        "mean5": np.asarray(panel["mean5"], dtype=np.float32),
        "mean20": np.asarray(panel["mean20"], dtype=np.float32),
    }
    states["innovation"] = states["current"] - states["mean5"]
    states["convergence"] = states["mean5"] - states["mean20"]
    scores = np.column_stack(
        [states[name] @ components.T for name in STATE_NAMES]
    ).astype(np.float32)
    return scores, components, {
        "component_count": LATENT_COMPONENTS,
        "active_typed_column_count": int(np.sum(active)),
        "fit_date_count": int(len(current)),
        "explained_variance_ratio_sum": float(
            np.sum(model.explained_variance_ratio_)
        ),
        "singular_values": [float(value) for value in model.singular_values_],
        "target_free_fit": True,
    }


def _catboost() -> Any:
    try:
        import catboost
    except ImportError as exc:  # pragma: no cover - runtime dependency gate
        raise RuntimeError("CatBoost is missing from the Forecast RADAR runtime") from exc
    if catboost.__version__ != CATBOOST_VERSION:
        raise RuntimeError(
            f"CatBoost version mismatch: {catboost.__version__} != {CATBOOST_VERSION}"
        )
    return catboost


def _save_catboost_atomic(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".cbm", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        model.save_model(str(temporary), format="cbm")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fit_model(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    feature_names: Sequence[str],
    thread_count: int,
) -> tuple[Any, list[dict[str, Any]], float]:
    catboost = _catboost()
    model = catboost.CatBoostRegressor(
        **CATBOOST_PARAMETERS,
        thread_count=int(thread_count),
    )
    started = time.monotonic()
    pool = catboost.Pool(
        np.asarray(features, dtype=np.float32),
        label=np.asarray(targets, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
    )
    model.fit(pool, verbose=False)
    importance = np.asarray(model.get_feature_importance(), dtype=np.float64)
    order = np.argsort(importance)[::-1][:50]
    top = [
        {"feature": str(feature_names[index]), "importance": float(importance[index])}
        for index in order
    ]
    return model, top, time.monotonic() - started


def _fit_probability_calibration(v16_root: Path) -> dict[str, Any]:
    """Fit only on sealed historical OOS predictions, never final-refit residuals."""

    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    actual_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for year in range(2021, 2027):
        path = Path(v16_root) / f"fold_{year}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as item:
            actual_parts.append(np.asarray(item["actual"], dtype=np.float32))
            prediction_parts.append(
                np.asarray(item[GLOBAL_MODEL_KEY], dtype=np.float32)
            )
        sources.append({"year": year, "path": str(path), "sha256": sha256_file(path)})
    if not actual_parts:
        raise FileNotFoundError("sealed v16 OOS fold predictions are missing")
    actual = np.concatenate(actual_parts)
    prediction = np.concatenate(prediction_parts)
    result: dict[str, Any] = {"sources": sources, "rows": int(len(actual)), "heads": {}}
    for horizon, index in ((5, 0), (20, 6)):
        x = prediction[:, index].astype(np.float64)
        y = (actual[:, index] > 0.0).astype(np.float64)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        fitted = model.fit_transform(x, y)
        result["heads"][f"p_up_{horizon}d"] = {
            "target": TARGET_NAMES[index],
            "x_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
            "positive_rate": float(np.mean(y)),
            "brier_oos": float(brier_score_loss(y, fitted)),
            "calibration_source": "sealed_v16_historical_oos",
        }
    return result


def calibrate_probability(value: np.ndarray, head: Mapping[str, Any]) -> np.ndarray:
    return np.interp(
        np.asarray(value, dtype=np.float64),
        np.asarray(head["x_thresholds"], dtype=np.float64),
        np.asarray(head["y_thresholds"], dtype=np.float64),
    )


def observed_graph_symbols(graph_dataset_root: Path) -> list[str]:
    """Return the actual point-in-time stock universe stored in graph snapshots.

    The legacy graph manifest uses the scalar sentinel ``ALL_PANEL`` in
    ``requested_symbols``.  Treating that field as an iterable silently turns it
    into character symbols, so coverage receipts must be derived from the
    snapshot payloads instead.
    """

    graph_dataset_root = Path(graph_dataset_root)
    graph_manifest_path = graph_dataset_root / "manifest.json"
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    snapshots = graph_manifest.get("snapshots", [])
    if not snapshots:
        raise ValueError("graph manifest has no snapshots")
    observed: set[str] = set()
    for reference in snapshots:
        snapshot_path = Path(reference["path"])
        with np.load(snapshot_path, allow_pickle=False) as item:
            observed.update(str(value) for value in item["stock_symbols"])
    result = sorted(value for value in observed if value)
    if not result:
        raise ValueError("graph snapshots contain no stock symbols")
    return result


def validated_symbols_from_matrix(matrix: Mapping[str, Any]) -> list[str]:
    """Return symbols that actually contributed at least one complete row."""

    values = tuple(str(value) for value in matrix["symbol_values"])
    observed_codes = np.unique(np.asarray(matrix["symbol_codes"], dtype=np.int64))
    if np.any(observed_codes < 0) or np.any(observed_codes >= len(values)):
        raise ValueError("matrix contains an invalid symbol code")
    result = sorted({values[int(code)] for code in observed_codes})
    if not result:
        raise ValueError("matrix contains no observed stock symbols")
    return result


def validated_symbols_from_graph_training_scope(
    *, graph_dataset_root: Path, event_path: Path
) -> tuple[list[str], dict[str, Any]]:
    """Reconstruct the exact symbol eligibility used by the historical matrix."""

    graph_dataset_root = Path(graph_dataset_root)
    graph_manifest = json.loads(
        (graph_dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    event = readonly_connection(Path(event_path))
    try:
        timing = {
            str(row[0]): (str(row[1]), str(row[2]))
            for row in event.execute(
                "SELECT signal_date, price_date, flow_date FROM session_map"
            )
        }
    finally:
        event.close()
    observed: set[str] = set()
    accepted_dates: list[str] = []
    complete_rows = 0
    for reference in graph_manifest.get("snapshots", []):
        signal_date = str(reference["signal_date"])
        if int(signal_date[:4]) < 2020:
            continue
        if (
            float(reference.get("relation_stock_coverage_ratio") or 0.0)
            < MIN_RELATION_COVERAGE
        ):
            continue
        if timing.get(signal_date) != (
            str(reference["price_date"]),
            str(reference["flow_date"]),
        ):
            continue
        with np.load(reference["path"], allow_pickle=False) as item:
            symbols = np.asarray(item["stock_symbols"]).astype(str)
            target_mask = np.asarray(item["target_mask"], dtype=bool)
            targets = np.asarray(item["targets"], dtype=np.float32)
        complete = np.all(target_mask, axis=1) & np.all(np.isfinite(targets), axis=1)
        observed.update(str(value) for value in symbols[complete])
        complete_rows += int(np.sum(complete))
        accepted_dates.append(signal_date)
    result = sorted(observed)
    if not result or not accepted_dates:
        raise ValueError("no graph symbols matched the historical training scope")
    return result, {
        "date_count": len(accepted_dates),
        "date_start": accepted_dates[0],
        "date_end": accepted_dates[-1],
        "complete_row_count": complete_rows,
        "minimum_relation_coverage": MIN_RELATION_COVERAGE,
        "requires_complete_12_targets": True,
        "requires_exact_event_timing": True,
    }


def repair_manifest_coverage(
    *, graph_dataset_root: Path, output_root: Path
) -> dict[str, Any]:
    """Repair coverage metadata without touching any trained artifact."""

    manifest_path = Path(output_root) / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before_hashes = {
        key: sha256_file(Path(value["path"]))
        for key, value in manifest["artifacts"].items()
    }
    event_path = Path(manifest["sources"]["event_cube"]["path"])
    symbols, scope_audit = validated_symbols_from_graph_training_scope(
        graph_dataset_root=graph_dataset_root,
        event_path=event_path,
    )
    training = manifest.get("training", {})
    expected_scope = (
        int(training.get("dates", -1)),
        str(training.get("date_start")),
        str(training.get("date_end")),
        int(training.get("rows", -1)),
    )
    observed_scope = (
        scope_audit["date_count"],
        scope_audit["date_start"],
        scope_audit["date_end"],
        scope_audit["complete_row_count"],
    )
    if observed_scope != expected_scope:
        raise ValueError(
            f"coverage reconstruction differs from trained scope: "
            f"observed={observed_scope} expected={expected_scope}"
        )
    previous = manifest.get("coverage", {}).get("validated_core_symbols", [])
    previous_repair = manifest.get("coverage_repair")
    repair_history = list(manifest.get("coverage_repair_history", []))
    if previous_repair:
        repair_history.append(previous_repair)
    manifest["coverage"]["validated_core_symbols"] = symbols
    manifest["coverage"]["validated_core_symbol_count"] = len(symbols)
    manifest["coverage"]["symbol_source"] = (
        "historical_training_scope_complete_rows(graph_snapshot.stock_symbols)"
    )
    manifest["coverage"]["statement"] = (
        f"Historical OOS covers {len(symbols)} predominantly PIT SPY/QQQ-member "
        "symbols. It does not validate the general US-stock extrapolation tier."
    )
    if repair_history:
        manifest["coverage_repair_history"] = repair_history
    manifest["coverage_repair"] = {
        "repaired_at_utc": utc_now(),
        "reason": (
            "derive coverage from complete rows in the exact historical training "
            "scope; do not use requested_symbols ALL_PANEL or all-snapshot union"
        ),
        "previous_entry_count": len(previous),
        "previous_unique_count": len(set(previous)),
        "corrected_unique_count": len(symbols),
        "scope_audit": scope_audit,
        "model_artifacts_retrained": False,
        "model_artifact_hashes_before": before_hashes,
    }
    write_json_atomic(manifest_path, manifest)
    after_hashes = {
        key: sha256_file(Path(value["path"]))
        for key, value in manifest["artifacts"].items()
    }
    if before_hashes != after_hashes:
        raise RuntimeError("coverage repair unexpectedly changed a model artifact")
    manifest["coverage_repair"]["model_artifact_hashes_after"] = after_hashes
    write_json_atomic(manifest_path, manifest)
    return manifest


def train_final_bundle(
    *,
    phase_a_root: Path,
    graph_dataset_root: Path,
    source_database: Path,
    v16_root: Path,
    v19_receipt: Path,
    output_root: Path,
    thread_count: int,
    replace: bool,
) -> dict[str, Any]:
    output_root = Path(output_root)
    manifest_path = output_root / "model_manifest.json"
    if manifest_path.exists() and not replace:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    event_path = Path(phase_a_root) / "v11_r2_flow_event_cube.sqlite3"
    if not event_path.is_file():
        raise FileNotFoundError(event_path)
    v19 = json.loads(Path(v19_receipt).read_text(encoding="utf-8"))
    if v19.get("gate", {}).get("status") != "V19_GLOBAL_DRIFT_PASS":
        raise RuntimeError("v19 distribution/potential gate is not PASS")

    started = utc_now()
    event = readonly_connection(event_path)
    source = readonly_connection(source_database)
    try:
        matrix = build_stock_matrix_from_sources(
            event=event,
            source=source,
            graph_dataset_root=graph_dataset_root,
        )
        panel = build_full_etf_panel(
            event=event,
            date_values=matrix["date_values"],
        )
    finally:
        source.close()
        event.close()

    scores, components, latent_audit = _fit_latent_full(panel)
    price = np.asarray(matrix["price_matrix"], dtype=np.float32)
    flow = np.asarray(matrix["flow_matrix"], dtype=np.float32)
    targets = np.asarray(matrix["targets"], dtype=np.float32)
    global_rows = scores[np.asarray(matrix["date_codes"], dtype=np.int64)]
    enriched = np.column_stack([price, flow, global_rows]).astype(np.float32)
    weights = date_balanced_weights(
        np.asarray(matrix["date_codes"], dtype=np.int64),
        np.arange(len(targets), dtype=np.int64),
    )
    price_names = tuple(f"price::{name}" for name in matrix["price_names"])
    flow_names = tuple(f"v12::{name}" for name in matrix["flow_names"])
    global_names = _factor_names()
    price_model, price_top, price_seconds = _fit_model(
        price, targets, weights, price_names, thread_count
    )
    enriched_model, enriched_top, enriched_seconds = _fit_model(
        enriched,
        targets,
        weights,
        price_names + flow_names + global_names,
        thread_count,
    )
    caps = residual_caps(targets)
    calibration = _fit_probability_calibration(v16_root)

    price_path = output_root / "price_model.cbm"
    enriched_path = output_root / "global_drift_model.cbm"
    latent_path = output_root / "latent_state.npz"
    _save_catboost_atomic(price_model, price_path)
    _save_catboost_atomic(enriched_model, enriched_path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".latent_state.", suffix=".npz", dir=output_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            components=components,
            tickers=np.asarray(panel["tickers"], dtype="U32"),
            residual_caps=caps,
        )
        os.replace(temporary, latent_path)
    finally:
        temporary.unlink(missing_ok=True)
    calibration_path = output_root / "probability_calibration.json"
    write_json_atomic(calibration_path, calibration)

    validated_core_symbols = validated_symbols_from_matrix(matrix)
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "started_at_utc": started,
        "activation_status": "SHADOW_ONLY",
        "timing_contract": TIMING_CONTRACT,
        "target_names": list(TARGET_NAMES),
        "coverage": {
            "validated_core_symbols": validated_core_symbols,
            "validated_core_symbol_count": len(validated_core_symbols),
            "symbol_source": "unique(matrix.symbol_values[matrix.symbol_codes])",
            "statement": (
                f"Historical OOS covers {len(validated_core_symbols)} predominantly "
                "PIT SPY/QQQ-member symbols. It does not validate the general "
                "US-stock extrapolation tier."
            ),
            "general_universe_status": "EXTRAPOLATED_UNVALIDATED",
        },
        "training": {
            "rows": int(len(targets)),
            "dates": int(len(matrix["date_values"])),
            "date_start": str(matrix["date_values"][0]),
            "date_end": str(matrix["date_values"][-1]),
            "date_balanced": True,
            "price_fit_seconds": price_seconds,
            "enriched_fit_seconds": enriched_seconds,
            "catboost_version": CATBOOST_VERSION,
            "catboost_parameters": CATBOOST_PARAMETERS,
            "residual_adapter": "price + clip(0.25*(enriched-price), caps)",
            "historical_oos_not_fresh_lockbox": True,
        },
        "feature_contract": {
            "price_names": list(matrix["price_names"]),
            "flow_names": list(matrix["flow_names"]),
            "global_names": list(global_names),
            "latent_states": list(STATE_NAMES),
            "latent": latent_audit,
        },
        "top_features": {
            "price_model": price_top,
            "global_drift_model": enriched_top,
        },
        "sources": {
            "event_cube": {"path": str(event_path), "sha256": sha256_file(event_path)},
            "graph_manifest": {
                "path": str(Path(graph_dataset_root) / "manifest.json"),
                "sha256": sha256_file(Path(graph_dataset_root) / "manifest.json"),
            },
            "v19_receipt": {"path": str(v19_receipt), "sha256": sha256_file(v19_receipt)},
            "v19_passed_paths": v19["gate"].get("passed_paths", []),
        },
        "artifacts": {
            "price_model": {"path": str(price_path), "sha256": sha256_file(price_path)},
            "global_drift_model": {
                "path": str(enriched_path),
                "sha256": sha256_file(enriched_path),
            },
            "latent_state": {"path": str(latent_path), "sha256": sha256_file(latent_path)},
            "probability_calibration": {
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            },
        },
        "quality_gate": "PASS_SHADOW_BUNDLE",
    }
    write_json_atomic(manifest_path, manifest)
    del matrix, panel, scores, components, price, flow, targets, global_rows, enriched
    del price_model, enriched_model
    gc.collect()
    return manifest


def load_bundle(model_root: Path) -> dict[str, Any]:
    model_root = Path(model_root)
    manifest_path = model_root / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unexpected Forecast RADAR model schema")
    for item in manifest["artifacts"].values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"model artifact hash mismatch: {item['path']}")
    catboost = _catboost()
    price_model = catboost.CatBoostRegressor()
    price_model.load_model(manifest["artifacts"]["price_model"]["path"])
    enriched_model = catboost.CatBoostRegressor()
    enriched_model.load_model(manifest["artifacts"]["global_drift_model"]["path"])
    with np.load(manifest["artifacts"]["latent_state"]["path"], allow_pickle=False) as item:
        latent = {name: np.asarray(item[name]) for name in item.files}
    calibration = json.loads(
        Path(manifest["artifacts"]["probability_calibration"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    return {
        "manifest": manifest,
        "price_model": price_model,
        "enriched_model": enriched_model,
        "latent": latent,
        "calibration": calibration,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase-a-root", type=Path, default=DEFAULT_PHASE_A_ROOT)
    result.add_argument("--graph-dataset-root", type=Path, default=DEFAULT_GRAPH_DATASET_ROOT)
    result.add_argument("--source-database", type=Path, default=DEFAULT_BASE_DATABASE)
    result.add_argument("--v16-root", type=Path, default=DEFAULT_V16_ROOT)
    result.add_argument("--v19-receipt", type=Path, default=DEFAULT_V19_RECEIPT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_MODEL_ROOT)
    result.add_argument("--thread-count", type=int, default=10)
    result.add_argument("--replace", action="store_true")
    result.add_argument("--repair-coverage-only", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.repair_coverage_only:
        manifest = repair_manifest_coverage(
            graph_dataset_root=args.graph_dataset_root,
            output_root=args.output_root,
        )
    else:
        manifest = train_final_bundle(
            phase_a_root=args.phase_a_root,
            graph_dataset_root=args.graph_dataset_root,
            source_database=args.source_database,
            v16_root=args.v16_root,
            v19_receipt=args.v19_receipt,
            output_root=args.output_root,
            thread_count=args.thread_count,
            replace=args.replace,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
