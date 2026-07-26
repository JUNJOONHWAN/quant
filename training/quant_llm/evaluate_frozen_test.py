"""Evaluate frozen Qwen quant predictions and emit a release-bound gate report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from workflows.quant_ai_radar.model_runtime import (
    DETERMINISTIC_FACT_PRECISION,
    FLOW_SIGNAL_SOURCES,
    REGIMES,
    REQUIRED_RESPONSE_KEYS,
    SIGNALS,
    TASK_TYPES,
    canonicalize_deterministic_facts,
    judgement_prohibited_violations,
)


EVALUATION_SCHEMA = "quant.frozen_test_evaluation.v1"
THRESHOLDS = {
    "json_schema_valid_rate": 0.995,
    "facts_exact_rate": 0.99,
    "regime_accuracy": 0.95,
    "structured_signal_accuracy": 0.95,
    "counter_evidence_recall": 0.90,
    "unknown_recall": 0.90,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_artifact_set(adapter_root: Path, artifacts: list[Path]) -> tuple[str, list[dict[str, str]]]:
    root = adapter_root.expanduser().resolve()
    if not root.is_dir() or not artifacts:
        raise ValueError("adapter root and at least one artifact are required")
    rows: list[tuple[str, str]] = []
    public: list[dict[str, str]] = []
    for raw in artifacts:
        path = raw.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"adapter artifact is missing: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"adapter artifact escapes adapter root: {path}") from exc
        digest = sha256_file(path)
        rows.append((relative, digest))
        public.append({"path": str(path), "relative_path": relative, "sha256": digest})
    rows.sort()
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest(), public


def read_expected(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in result:
                raise ValueError(f"missing/duplicate frozen example_id at line {line_number}")
            response = json.loads(row.get("response") or "")
            if not isinstance(response, dict):
                raise ValueError(f"frozen response at line {line_number} is not an object")
            result[example_id] = response
    if not result:
        raise ValueError("frozen test file is empty")
    return result


def read_predictions(
    path: Path,
    endpoint_model: str,
    adapter_set_sha256: str,
    frozen_test_sha256: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in result:
                raise ValueError(f"missing/duplicate prediction example_id at line {line_number}")
            if row.get("endpoint_model") != endpoint_model:
                raise ValueError(f"prediction line {line_number} has a different endpoint model")
            if row.get("adapter_set_sha256") != adapter_set_sha256:
                raise ValueError(f"prediction line {line_number} has a different adapter set")
            if row.get("frozen_test_sha256") != frozen_test_sha256:
                raise ValueError(f"prediction line {line_number} has a different frozen test")
            response = row.get("response")
            if isinstance(response, str):
                response = json.loads(response)
            if not isinstance(response, dict):
                raise ValueError(f"prediction response at line {line_number} is not an object")
            result[example_id] = response
    return result


def schema_valid(value: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if set(value) != set(REQUIRED_RESPONSE_KEYS):
        return False
    interpretation = value.get("interpretation")
    expected_interpretation = expected.get("interpretation") or {}
    confidence = value.get("confidence")
    return bool(
        isinstance(value.get("facts"), dict)
        and isinstance(interpretation, dict)
        and interpretation.get("scope") == "data_interpretation_not_trade_execution"
        and interpretation.get("task_type") == expected_interpretation.get("task_type")
        and interpretation.get("task_type") in TASK_TYPES
        and interpretation.get("price_signal") in SIGNALS
        and interpretation.get("etf_flow_signal") in SIGNALS
        and interpretation.get("etf_flow_signal_source") in FLOW_SIGNAL_SOURCES
        and value.get("regime") in REGIMES
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0 <= float(confidence) <= 1
        and isinstance(value.get("counter_evidence"), list)
        and all(isinstance(item, str) for item in value["counter_evidence"])
        and isinstance(value.get("unknowns"), list)
        and all(isinstance(item, str) for item in value["unknowns"])
        and isinstance(value.get("conclusion"), str)
        and bool(value["conclusion"].strip())
    )


def recall(expected: Any, observed: Any) -> float:
    if not isinstance(expected, list) or not all(
        isinstance(item, str) for item in expected
    ):
        raise ValueError("frozen target recall fields must be string arrays")
    if not isinstance(observed, list) or not all(
        isinstance(item, str) for item in observed
    ):
        return 0.0
    expected_set = set(expected)
    observed_set = set(observed)
    return 1.0 if not expected_set else len(expected_set & observed_set) / len(expected_set)


def evaluate(expected: Mapping[str, Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expected_ids = set(expected)
    prediction_ids = set(predictions)
    shared = sorted(expected_ids & prediction_ids)
    total = len(expected_ids)
    metrics = {
        "json_schema_valid_rate": 0.0,
        "facts_exact_rate": 0.0,
        "regime_accuracy": 0.0,
        "structured_signal_accuracy": 0.0,
        "counter_evidence_recall": 0.0,
        "unknown_recall": 0.0,
    }
    counts = {"schema_valid": 0, "facts_exact": 0, "regime_exact": 0, "signals_exact": 0}
    counter_sum = unknown_sum = 0.0
    violations: list[dict[str, Any]] = []
    signal_keys = ("price_signal", "etf_flow_signal", "etf_flow_signal_source", "task_type")
    for example_id in shared:
        target = expected[example_id]
        observed = predictions[example_id]
        if schema_valid(observed, target):
            counts["schema_valid"] += 1
        if canonicalize_deterministic_facts(
            observed.get("facts")
        ) == canonicalize_deterministic_facts(target.get("facts")):
            counts["facts_exact"] += 1
        if observed.get("regime") == target.get("regime"):
            counts["regime_exact"] += 1
        raw_target_interpretation = target.get("interpretation")
        raw_observed_interpretation = observed.get("interpretation")
        target_interpretation = (
            raw_target_interpretation
            if isinstance(raw_target_interpretation, Mapping)
            else {}
        )
        observed_interpretation = (
            raw_observed_interpretation
            if isinstance(raw_observed_interpretation, Mapping)
            else {}
        )
        if all(observed_interpretation.get(key) == target_interpretation.get(key) for key in signal_keys):
            counts["signals_exact"] += 1
        counter_sum += recall(target.get("counter_evidence"), observed.get("counter_evidence"))
        unknown_sum += recall(target.get("unknowns"), observed.get("unknowns"))
        as_of = str((target.get("facts") or {}).get("as_of_date") or "")
        for violation in judgement_prohibited_violations(observed, as_of):
            violations.append({"example_id": example_id, "violation": violation})
    if total:
        metrics = {
            "json_schema_valid_rate": counts["schema_valid"] / total,
            "facts_exact_rate": counts["facts_exact"] / total,
            "regime_accuracy": counts["regime_exact"] / total,
            "structured_signal_accuracy": counts["signals_exact"] / total,
            "counter_evidence_recall": counter_sum / total,
            "unknown_recall": unknown_sum / total,
        }
    coverage = {
        "expected": total,
        "predicted": len(prediction_ids),
        "matched": len(shared),
        "missing": sorted(expected_ids - prediction_ids),
        "unexpected": sorted(prediction_ids - expected_ids),
    }
    gates = {
        "full_frozen_test_coverage": not coverage["missing"] and not coverage["unexpected"],
        **{
            f"{name}_gte_{threshold}": metrics[name] >= threshold
            for name, threshold in THRESHOLDS.items()
        },
        "zero_future_or_trade_violations": not violations,
    }
    return {
        "coverage": coverage,
        "counts": counts,
        "metrics": {key: round(value, 8) for key, value in metrics.items()},
        "thresholds": THRESHOLDS,
        "facts_comparison_contract": {
            "mode": "exact_after_declared_precision_canonicalization",
            "numeric_precision": DETERMINISTIC_FACT_PRECISION,
        },
        "prohibited_violation_count": len(violations),
        "prohibited_violations": violations[:100],
        "required_gates": gates,
        "status": "green" if all(gates.values()) else "red",
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".frozen-evaluation-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--endpoint-model", required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_manifest = args.dataset_manifest.expanduser().resolve()
    test_file = args.test_file.expanduser().resolve()
    predictions = args.predictions.expanduser().resolve()
    for path, label in ((dataset_manifest, "dataset manifest"), (test_file, "test file"), (predictions, "predictions")):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    test_entry = manifest.get("sealed_test_reference") or {}
    declared_test_path = Path(str(test_entry.get("path") or "")).expanduser().resolve()
    if declared_test_path != test_file:
        raise ValueError("--test-file does not match the sealed test reference")
    if test_entry.get("used_by_training") is not False:
        raise ValueError("sealed test reference is not explicitly excluded from training")
    declared = str(test_entry.get("sha256") or "")
    observed_test_sha = sha256_file(test_file)
    if declared != observed_test_sha:
        raise ValueError("frozen test SHA256 does not match the dataset manifest")
    adapter_sha, artifact_rows = adapter_artifact_set(args.adapter_root, args.artifact)
    result = evaluate(
        read_expected(test_file),
        read_predictions(predictions, args.endpoint_model, adapter_sha, observed_test_sha),
    )
    report = {
        "schema_version": EVALUATION_SCHEMA,
        "endpoint_model": args.endpoint_model,
        "adapter_root": str(args.adapter_root.expanduser().resolve()),
        "adapter_set_sha256": adapter_sha,
        "adapter_artifacts": artifact_rows,
        "dataset_manifest": {"path": str(dataset_manifest), "sha256": sha256_file(dataset_manifest)},
        "frozen_test": {"path": str(test_file), "sha256": observed_test_sha},
        "predictions": {"path": str(predictions), "sha256": sha256_file(predictions)},
        **result,
    }
    write_atomic(args.output.expanduser().resolve(), report)
    print(json.dumps({"status": report["status"], "output": str(args.output.expanduser().resolve())}))
    return 0 if report["status"] == "green" else 2


if __name__ == "__main__":
    raise SystemExit(main())
