"""Create an explicit task-balanced training view without deleting candidates."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


TASK_TYPES = (
    "etf_own_flow_analysis",
    "stock_constituent_flow_analysis",
    "all_stock_control_analysis",
)
DEFAULT_TRAIN_QUOTAS = {
    "etf_own_flow_analysis": 20_000,
    "stock_constituent_flow_analysis": 25_000,
    "all_stock_control_analysis": 15_000,
}


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_quotas(value: str) -> Dict[str, int]:
    result = {}
    for item in value.split(","):
        name, raw_count = item.split("=", 1)
        name = name.strip()
        if name not in TASK_TYPES:
            raise ValueError("unknown task type in quota: {}".format(name))
        count = int(raw_count)
        if count < 1:
            raise ValueError("task quotas must be positive")
        result[name] = count
    missing = set(TASK_TYPES) - set(result)
    if missing:
        raise ValueError("missing task quotas: {}".format(",".join(sorted(missing))))
    return result


def _select(path: Path, quotas: Mapping[str, int]) -> tuple:
    heaps = {task: [] for task in TASK_TYPES}
    candidate_counts = {task: 0 for task in TASK_TYPES}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = str((row.get("metadata") or {}).get("task_type") or "")
            if task not in heaps:
                raise ValueError("{}:{} unknown task_type {}".format(path, line_number, task))
            candidate_counts[task] += 1
            example_id = str(row.get("example_id") or "")
            if len(example_id) != 64:
                raise ValueError("{}:{} invalid example_id".format(path, line_number))
            key = int(example_id, 16)
            item = (-key, example_id, _canonical_json(row))
            heap = heaps[task]
            quota = int(quotas[task])
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif key < -heap[0][0]:
                heapq.heapreplace(heap, item)
    selected = []
    selected_counts = {}
    for task in TASK_TYPES:
        rows = sorted(heaps[task], key=lambda item: (-item[0], item[1]))
        selected_counts[task] = len(rows)
        selected.extend((item[1], item[2]) for item in rows)
    selected.sort(key=lambda item: item[0])
    return selected, candidate_counts, selected_counts


def build_balanced_training_set(
    candidate_root: Path,
    output_root: Path,
    *,
    train_quotas: Mapping[str, int] = DEFAULT_TRAIN_QUOTAS,
    validation_per_task: int = 512,
    replace: bool = False,
) -> dict:
    candidate = Path(candidate_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    candidate_manifest = json.loads(
        (candidate / "manifest.json").read_text(encoding="utf-8")
    )
    if validation_per_task < 1:
        raise ValueError("validation_per_task must be positive")
    validation_quotas = {task: validation_per_task for task in TASK_TYPES}
    selected = {}
    candidate_counts = {}
    selected_counts = {}
    for split, quotas in (("train", train_quotas), ("validation", validation_quotas)):
        rows, before, after = _select(candidate / "{}.jsonl".format(split), quotas)
        selected[split] = rows
        candidate_counts[split] = before
        selected_counts[split] = after
        missing = [task for task, count in after.items() if count == 0]
        if missing:
            raise ValueError(
                "candidate corpus has no {} examples for {}".format(
                    split, ",".join(missing)
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    targets = [output / "train.jsonl", output / "validation.jsonl", output / "manifest.json"]
    if any(path.exists() for path in targets) and not replace:
        raise FileExistsError("balanced output exists; pass --replace")
    staging = Path(tempfile.mkdtemp(prefix=".balanced-sft-", dir=str(output.parent)))
    try:
        files = {}
        for split in ("train", "validation"):
            path = staging / "{}.jsonl".format(split)
            with path.open("w", encoding="utf-8") as handle:
                for _, encoded in selected[split]:
                    handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            files[split] = {
                "filename": path.name,
                "rows": sum(selected_counts[split].values()),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        test_path = candidate / "test.jsonl"
        manifest = {
            **{
                key: candidate_manifest.get(key)
                for key in (
                    "dataset_contract_version",
                    "example_schema_version",
                    "input_packet_schema_required",
                    "model_family",
                    "framework",
                    "training_method",
                    "etf_flow_policy_id",
                    "future_returns_in_prompt",
                    "future_returns_in_target",
                    "trade_execution_target",
                    "target_origin",
                    "split_contract",
                    "preprocessing_policy",
                )
            },
            "schema_version": "quant.sft.balanced_manifest.v1",
            "required_split_files": ["train", "validation"],
            "candidate_manifest": str(candidate / "manifest.json"),
            "candidate_manifest_sha256": _sha256_file(candidate / "manifest.json"),
            "selection": {
                "method": "lowest example_id hash within time split and task type",
                "train_quotas": dict(train_quotas),
                "validation_quotas": validation_quotas,
                "candidate_counts": candidate_counts,
                "selected_counts": selected_counts,
                "candidate_corpus_deleted_or_modified": False,
                "representative_sample_claimed": False,
            },
            "sealed_test_reference": {
                "path": str(test_path),
                "rows": (candidate_manifest.get("files") or {}).get("test", {}).get("rows"),
                "sha256": _sha256_file(test_path),
                "used_by_training": False,
            },
            "files": files,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation"):
            os.replace(staging / "{}.jsonl".format(split), output / "{}.jsonl".format(split))
        os.replace(manifest_path, output / "manifest.json")
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--train-quotas",
        default=",".join("{}={}".format(key, DEFAULT_TRAIN_QUOTAS[key]) for key in TASK_TYPES),
    )
    parser.add_argument("--validation-per-task", type=int, default=512)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    result = build_balanced_training_set(
        args.candidate_root,
        args.output_root,
        train_quotas=_parse_quotas(args.train_quotas),
        validation_per_task=args.validation_per_task,
        replace=args.replace,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
