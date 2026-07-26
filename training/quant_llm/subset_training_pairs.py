"""Derive a deterministic readiness subset from a sealed pair selection."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from training.quant_llm.build_balanced_training_set import TASK_TYPES


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subset_pairs(
    source_root: Path,
    output_root: Path,
    *,
    per_split_task: int,
    replace: bool = False,
) -> dict:
    if per_split_task < 1:
        raise ValueError("per_split_task must be positive")
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_pairs = source / str(
        (source_manifest.get("pairs_file") or {}).get("filename") or "pairs.jsonl"
    )
    if _sha256_file(source_pairs) != (source_manifest.get("pairs_file") or {}).get("sha256"):
        raise ValueError("source pair selection SHA256 mismatch")
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / "pairs.jsonl", output / "manifest.json"]
    if any(path.exists() for path in targets) and not replace:
        raise FileExistsError("subset exists; pass --replace")

    heaps = {
        (split, task): []
        for split in ("train", "validation", "test")
        for task in TASK_TYPES
    }
    with source_pairs.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("split")), str(row.get("proxy_task_type")))
            if key not in heaps:
                raise ValueError("{}:{} unknown split/task".format(source_pairs, line_number))
            digest = str(row.get("pair_hash") or "")
            numeric = int(digest, 16)
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":"))
            item = (-numeric, digest, encoded)
            heap = heaps[key]
            if len(heap) < per_split_task:
                heapq.heappush(heap, item)
            elif numeric < -heap[0][0]:
                heapq.heapreplace(heap, item)

    selected = []
    counts = {split: {task: 0 for task in TASK_TYPES} for split in ("train", "validation", "test")}
    for (split, task), heap in heaps.items():
        for _, digest, encoded in heap:
            row = json.loads(encoded)
            selected.append(row)
            counts[split][task] += 1
    selected.sort(key=lambda row: (row["as_of_date"], row["symbol"], row["pair_hash"]))
    descriptor, name = tempfile.mkstemp(prefix=".pair-subset-", dir=str(output))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output / "pairs.jsonl")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    manifest = {
        **{
            key: source_manifest.get(key)
            for key in (
                "database",
                "database_size_bytes",
                "source",
                "requested_range",
                "split_contract",
                "selection_salt",
                "survivorship_policy",
                "proxy_contract",
            )
        },
        "schema_version": "quant.training_pair_subset.v1",
        "complete": True,
        "parent_manifest": str(source_manifest_path),
        "parent_manifest_sha256": _sha256_file(source_manifest_path),
        "parent_pairs_sha256": _sha256_file(source_pairs),
        "subset_method": "lowest pair_hash within split and proxy task",
        "per_split_proxy_task": per_split_task,
        "selected_counts": counts,
        "representative_sample_claimed": False,
        "pairs_file": {
            "filename": "pairs.jsonl",
            "rows": len(selected),
            "bytes": (output / "pairs.jsonl").stat().st_size,
            "sha256": _sha256_file(output / "pairs.jsonl"),
        },
    }
    manifest_path = output / "manifest.json"
    descriptor, name = tempfile.mkstemp(prefix=".subset-manifest-", dir=str(output))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--per-split-task", type=int, default=10)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    result = subset_pairs(
        args.source_root,
        args.output_root,
        per_split_task=args.per_split_task,
        replace=args.replace,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
