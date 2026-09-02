#!/usr/bin/env python3
"""Merge one accepted PEFT LoRA into a standalone Hugging Face model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def manifest_for(
    *,
    model_name: str,
    base_model: Path,
    adapter: Path,
    output: Path,
) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "merge_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    adapter_files = []
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter / name
        if not path.is_file():
            raise RuntimeError(f"required adapter artifact is missing: {path}")
        adapter_files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    core = {
        "schema_version": "quant.merged_hf_model.v1",
        "status": "complete",
        "model_name": model_name,
        "precision": "bfloat16",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base_model),
        "adapter": str(adapter),
        "adapter_artifacts": adapter_files,
        "output": str(output),
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files),
    }
    core["content_sha256"] = hashlib.sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    return core


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def merge(
    *,
    model_name: str,
    base_model: Path,
    adapter: Path,
    output: Path,
) -> dict[str, Any]:
    base = base_model.expanduser().resolve()
    lora = adapter.expanduser().resolve()
    target = output.expanduser().resolve()
    if not base.is_dir():
        raise RuntimeError(f"base model is missing: {base}")
    if not lora.is_dir():
        raise RuntimeError(f"adapter is missing: {lora}")
    if target.exists():
        raise RuntimeError(f"output already exists: {target}")
    temporary = target.with_name(f".{target.name}.incomplete")
    if temporary.exists():
        raise RuntimeError(f"incomplete merge path already exists: {temporary}")
    temporary.mkdir(parents=True)

    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    peft_model = PeftModel.from_pretrained(
        model,
        lora,
        is_trainable=False,
        local_files_only=True,
    )
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.config.name_or_path = model_name
    merged.save_pretrained(
        temporary,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        lora,
        local_files_only=True,
    )
    tokenizer.save_pretrained(temporary)
    os.replace(temporary, target)
    manifest = manifest_for(
        model_name=model_name,
        base_model=base,
        adapter=lora,
        output=target,
    )
    write_json(target / "merge_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(
        model_name=args.model_name,
        base_model=args.base_model,
        adapter=args.adapter,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": result["output"],
                "total_bytes": result["total_bytes"],
                "content_sha256": result["content_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
