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


def validate_merged_model(
    *,
    endpoint_model: str,
    base_model: str,
    adapter_artifact_rows: list[dict[str, str]],
    merged_manifest: Path,
    merged_model_root: Path,
) -> dict[str, Any]:
    """Bind a BF16 merge to the already-evaluated adapter release."""

    manifest_path = merged_manifest.expanduser().resolve()
    model_root = merged_model_root.expanduser().resolve()
    manifest = read_object(manifest_path, "merged-model manifest")
    if manifest.get("schema_version") != "quant.merged_hf_model.v1":
        raise ValueError("merged-model manifest schema is unsupported")
    if manifest.get("status") != "complete":
        raise ValueError("merged-model manifest is not complete")
    if manifest.get("model_name") != endpoint_model:
        raise ValueError("merged-model name does not match endpoint_model")
    if manifest.get("precision") != "bfloat16":
        raise ValueError("merged-model precision must be bfloat16")
    if not model_root.is_dir():
        raise ValueError(f"merged-model root is missing: {model_root}")
    if Path(base_model).expanduser().resolve() != model_root:
        raise ValueError("base_model must be the verified merged-model root")

    declared_files = manifest.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        raise ValueError("merged-model manifest has no model files")
    verified_files = []
    for index, row in enumerate(declared_files):
        if not isinstance(row, dict):
            raise ValueError(f"merged-model file {index} is not an object")
        relative = Path(str(row.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"merged-model file escapes root: {relative}")
        path = (model_root / relative).resolve()
        try:
            path.relative_to(model_root)
        except ValueError as exc:
            raise ValueError(f"merged-model file escapes root: {path}") from exc
        if not path.is_file():
            raise ValueError(f"merged-model file is missing: {path}")
        if path.stat().st_size != int(row.get("bytes") or -1):
            raise ValueError(f"merged-model file size mismatch: {path}")
        digest = sha256_file(path)
        if digest != str(row.get("sha256") or ""):
            raise ValueError(f"merged-model file SHA256 mismatch: {path}")
        verified_files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )

    source_adapter_hashes = sorted(
        str(row.get("sha256") or "") for row in adapter_artifact_rows
    )
    merged_adapter_rows = manifest.get("adapter_artifacts")
    if not isinstance(merged_adapter_rows, list):
        raise ValueError("merged-model manifest has no adapter artifacts")
    merged_adapter_hashes = sorted(
        str(row.get("sha256") or "")
        for row in merged_adapter_rows
        if isinstance(row, dict)
    )
    if merged_adapter_hashes != source_adapter_hashes:
        raise ValueError(
            "merged-model adapter artifacts do not match the evaluated release"
        )

    declared_content_sha = str(manifest.get("content_sha256") or "")
    content_core = dict(manifest)
    content_core.pop("content_sha256", None)
    observed_content_sha = hashlib.sha256(
        canonical_json(content_core).encode("utf-8")
    ).hexdigest()
    if declared_content_sha != observed_content_sha:
        raise ValueError("merged-model content SHA256 is invalid")
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "root": str(model_root),
        "model_name": endpoint_model,
        "precision": "bfloat16",
        "content_sha256": observed_content_sha,
        "total_bytes": sum(row["bytes"] for row in verified_files),
        "file_count": len(verified_files),
    }


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
    merged_manifest: Path | None = None,
    merged_model_root: Path | None = None,
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
    evaluation_endpoint_model = str(
        evaluation_value.get("endpoint_model") or ""
    )
    if (
        evaluation_endpoint_model != endpoint_model
        and merged_manifest is None
    ):
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
    release = {
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
            "evaluation_endpoint_model": evaluation_endpoint_model,
        },
    }
    if (merged_manifest is None) != (merged_model_root is None):
        raise ValueError(
            "merged_manifest and merged_model_root must be supplied together"
        )
    if merged_manifest is not None and merged_model_root is not None:
        release["merged_model"] = validate_merged_model(
            endpoint_model=endpoint_model,
            base_model=base_model,
            adapter_artifact_rows=artifact_rows,
            merged_manifest=merged_manifest,
            merged_model_root=merged_model_root,
        )
    return release


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
    parser.add_argument("--merged-manifest", type=Path)
    parser.add_argument("--merged-model-root", type=Path)
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
        merged_manifest=args.merged_manifest,
        merged_model_root=args.merged_model_root,
    )
    write_atomic(args.output, release)
    print(json.dumps({"status": "accepted", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
