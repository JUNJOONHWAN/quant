"""Audit every training row with the pinned Qwen tokenizer before NeMo runs.

The training configs deliberately disable truncation.  This command is the
fail-closed gate that proves the complete rendered conversation, including the
assistant target, fits the configured sequence length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_MODEL_PATH = Path("/home/zooh/models/Qwen3-8B-bf16")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(sorted_values: Sequence[int], fraction: float) -> Optional[int]:
    if not sorted_values:
        return None
    index = int(round((len(sorted_values) - 1) * fraction))
    return int(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def _token_count(value) -> int:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("unexpected batched tokenizer output")
        value = value[0]
    return len(value or [])


def _write_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".token-audit-", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def audit_token_lengths(
    dataset_root: Path,
    model_path: Path,
    *,
    max_length: int,
    splits: Sequence[str] = ("train", "validation"),
    max_error_examples: int = 20,
) -> dict:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    root = Path(dataset_root).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - image/runtime error path
        raise RuntimeError("transformers is required for the tokenizer audit") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("model tokenizer has no chat_template")

    full_lengths: List[int] = []
    assistant_lengths: List[int] = []
    task_counts: Counter[str] = Counter()
    split_counts: Dict[str, int] = {}
    files = {}
    errors = []
    max_row = None

    for split in splits:
        expected = (manifest.get("files") or {}).get(split) or {}
        path = root / str(expected.get("filename") or "{}.jsonl".format(split))
        if not path.is_file():
            errors.append({"split": split, "reason": "missing_split_file"})
            continue
        files[split] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                count += 1
                row = json.loads(line)
                context = str(row.get("context") or "")
                instruction = str(row.get("instruction") or "")
                response = str(row.get("response") or "")
                if not context or not instruction or not response:
                    if len(errors) < max_error_examples:
                        errors.append(
                            {
                                "split": split,
                                "line": line_number,
                                "example_id": row.get("example_id"),
                                "reason": "empty_conversation_field",
                            }
                        )
                    continue
                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ]
                full = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    truncation=False,
                )
                prompt = tokenizer.apply_chat_template(
                    messages[:2],
                    tokenize=True,
                    return_dict=True,
                    truncation=False,
                )
                full_length = _token_count(full)
                prompt_length = _token_count(prompt)
                assistant_length = full_length - prompt_length
                full_lengths.append(full_length)
                assistant_lengths.append(assistant_length)
                metadata = row.get("metadata") or {}
                task_counts[str(metadata.get("task_type") or "missing")] += 1
                if max_row is None or full_length > max_row["full_tokens"]:
                    max_row = {
                        "split": split,
                        "line": line_number,
                        "example_id": row.get("example_id"),
                        "symbol": metadata.get("symbol"),
                        "as_of_date": metadata.get("as_of_date"),
                        "task_type": metadata.get("task_type"),
                        "full_tokens": full_length,
                        "prompt_tokens": prompt_length,
                        "assistant_span_tokens": assistant_length,
                    }
                reason = None
                if assistant_length <= 0:
                    reason = "assistant_target_has_no_tokens"
                elif full_length > max_length:
                    reason = "conversation_exceeds_max_length"
                if reason and len(errors) < max_error_examples:
                    errors.append(
                        {
                            "split": split,
                            "line": line_number,
                            "example_id": row.get("example_id"),
                            "symbol": metadata.get("symbol"),
                            "as_of_date": metadata.get("as_of_date"),
                            "full_tokens": full_length,
                            "prompt_tokens": prompt_length,
                            "assistant_span_tokens": assistant_length,
                            "reason": reason,
                        }
                    )
        split_counts[split] = count

    ordered_full = sorted(full_lengths)
    ordered_assistant = sorted(assistant_lengths)
    overlength_count = sum(value > max_length for value in full_lengths)
    nonpositive_assistant_count = sum(value <= 0 for value in assistant_lengths)
    result = {
        "schema_version": "quant.token_length_audit.v1",
        "ok": not errors and overlength_count == 0 and nonpositive_assistant_count == 0,
        "dataset_root": str(root),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "model_path": str(model),
        "tokenizer_class": type(tokenizer).__name__,
        "max_length": max_length,
        "truncation_policy": "disabled_fail_closed",
        "splits_audited": list(splits),
        "split_counts": split_counts,
        "total_rows": len(full_lengths),
        "task_counts": dict(sorted(task_counts.items())),
        "overlength_count": overlength_count,
        "nonpositive_assistant_count": nonpositive_assistant_count,
        "full_token_quantiles": {
            "p50": _percentile(ordered_full, 0.50),
            "p90": _percentile(ordered_full, 0.90),
            "p95": _percentile(ordered_full, 0.95),
            "p99": _percentile(ordered_full, 0.99),
            "max": ordered_full[-1] if ordered_full else None,
        },
        "assistant_token_quantiles": {
            "min": ordered_assistant[0] if ordered_assistant else None,
            "p50": _percentile(ordered_assistant, 0.50),
            "p99": _percentile(ordered_assistant, 0.99),
            "max": ordered_assistant[-1] if ordered_assistant else None,
        },
        "max_row": max_row,
        "files": files,
        "errors": errors,
        "errors_truncated": (overlength_count + nonpositive_assistant_count) > len(errors),
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    if not splits:
        parser.error("--splits must contain at least one split")
    result = audit_token_lengths(
        args.dataset_root,
        args.model_path,
        max_length=args.max_length,
        splits=splits,
    )
    if args.output:
        _write_atomic(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
