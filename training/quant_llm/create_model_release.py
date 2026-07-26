"""Create the only release manifest accepted by Quant AI Radar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


RELEASE_SCHEMA = "quant.trained_model_release.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def build_release(
    *,
    model_id: str,
    endpoint_model: str,
    base_model: str,
    adapter_root: Path,
    artifacts: list[Path],
    dataset_manifest: Path,
    evaluation_report: Path,
) -> dict[str, Any]:
    root = adapter_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"adapter root is missing: {root}")
    if not artifacts:
        raise ValueError("at least one explicit adapter artifact is required")
    artifact_rows = []
    set_rows = []
    for raw in artifacts:
        path = raw.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"adapter artifact is missing: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"adapter artifact escapes adapter root: {path}") from exc
        digest = sha256_file(path)
        artifact_rows.append({"path": str(path), "sha256": digest})
        set_rows.append((relative, digest))
    set_rows.sort()
    adapter_set_sha = hashlib.sha256(
        canonical_json(set_rows).encode("utf-8")
    ).hexdigest()

    dataset = dataset_manifest.expanduser().resolve()
    evaluation = evaluation_report.expanduser().resolve()
    read_object(dataset, "dataset manifest")
    evaluation_value = read_object(evaluation, "evaluation report")
    if evaluation_value.get("schema_version") != "quant.frozen_test_evaluation.v1":
        raise ValueError("evaluation report schema is not the frozen-test contract")
    if evaluation_value.get("endpoint_model") != endpoint_model:
        raise ValueError("evaluation endpoint_model does not match the release")
    if evaluation_value.get("adapter_set_sha256") != adapter_set_sha:
        raise ValueError("evaluation was not run against the released adapter artifacts")
    evaluated_dataset = evaluation_value.get("dataset_manifest") or {}
    if evaluated_dataset.get("sha256") != sha256_file(dataset):
        raise ValueError("evaluation dataset manifest does not match the release")
    evaluation_inputs = {}
    for key in ("frozen_test", "predictions"):
        item = evaluation_value.get(key) or {}
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        expected = str(item.get("sha256") or "")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"evaluation {key} artifact is missing or changed")
        evaluation_inputs[key] = {"path": str(path), "sha256": expected}
    if evaluation_value.get("status") != "green":
        raise ValueError("evaluation report is not green")
    if int(evaluation_value.get("prohibited_violation_count", -1)) != 0:
        raise ValueError("evaluation report contains prohibited violations")
    required_gates = evaluation_value.get("required_gates")
    if not isinstance(required_gates, dict) or not required_gates:
        raise ValueError("evaluation report has no required gates")
    failed = sorted(key for key, value in required_gates.items() if value is not True)
    if failed:
        raise ValueError(f"evaluation report gates failed: {failed}")
    return {
        "schema_version": RELEASE_SCHEMA,
        "status": "accepted",
        "model_id": model_id,
        "endpoint_model": endpoint_model,
        "base_model": base_model,
        "adapter_root": str(root),
        "adapter_set_sha256": adapter_set_sha,
        "artifacts": artifact_rows,
        "dataset_manifest": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
        },
        "evaluation": {
            "path": str(evaluation),
            "sha256": sha256_file(evaluation),
        },
        "evaluation_inputs": evaluation_inputs,
        "deployment_contract": {
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "scope": "data_interpretation_not_trade_execution",
            "fallback_model_allowed": False,
        },
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".model-release-", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--endpoint-model", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = build_release(
        model_id=args.model_id,
        endpoint_model=args.endpoint_model,
        base_model=args.base_model,
        adapter_root=args.adapter_root,
        artifacts=args.artifact,
        dataset_manifest=args.dataset_manifest,
        evaluation_report=args.evaluation_report,
    )
    write_atomic(args.output, release)
    print(json.dumps({"status": "accepted", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
