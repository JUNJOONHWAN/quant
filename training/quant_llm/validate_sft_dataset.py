"""Fail-closed validation for the Qwen quant SFT dataset artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from quant_dataset.point_in_time import ETF_FLOW_POLICY_ID
from training.quant_llm import DATASET_SCHEMA_VERSION
from training.quant_llm.build_sft_dataset import RESPONSE_KEYS, assign_split


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(root: Path) -> dict:
    dataset_root = Path(root).expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    warnings = []
    seen = set()
    observed_counts: Dict[str, int] = {}
    contract = manifest.get("split_contract") or {}

    if manifest.get("etf_flow_policy_id") != ETF_FLOW_POLICY_ID:
        errors.append("manifest ETF Flow policy id mismatch")
    if manifest.get("future_returns_in_prompt") is not False:
        errors.append("manifest does not prohibit future returns in prompts")
    if manifest.get("future_returns_in_target") is not False:
        errors.append("manifest does not prohibit future returns in targets")
    preprocessing = manifest.get("preprocessing_policy") or {}
    if preprocessing.get("etf_liquidity_policy_id") != "asof_etf_liquidity_20s_v1":
        errors.append("manifest ETF liquidity policy id mismatch")
    if "present-day active lists forbidden" not in str(
        preprocessing.get("delisted_security_policy")
    ):
        errors.append("manifest does not prohibit survivorship filtering")

    required_splits = tuple(
        manifest.get("required_split_files") or ("train", "validation", "test")
    )
    for split in required_splits:
        expected = (manifest.get("files") or {}).get(split) or {}
        path = dataset_root / str(expected.get("filename") or "{}.jsonl".format(split))
        if not path.is_file():
            errors.append("missing {} split file".format(split))
            continue
        if expected.get("sha256") != _sha256_file(path):
            errors.append("{} SHA256 mismatch".format(split))
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    row = json.loads(line)
                    response = json.loads(row["response"])
                except Exception as exc:
                    errors.append("{}:{} invalid JSON: {}".format(split, line_number, exc))
                    continue
                if row.get("schema_version") != DATASET_SCHEMA_VERSION:
                    errors.append("{}:{} schema mismatch".format(split, line_number))
                example_id = row.get("example_id")
                if not example_id or example_id in seen:
                    errors.append("{}:{} duplicate/missing example_id".format(split, line_number))
                seen.add(example_id)
                metadata = row.get("metadata") or {}
                if metadata.get("split") != split:
                    errors.append("{}:{} metadata split mismatch".format(split, line_number))
                if metadata.get("etf_flow_policy_id") != ETF_FLOW_POLICY_ID:
                    errors.append("{}:{} ETF Flow policy mismatch".format(split, line_number))
                if metadata.get("contains_future_label") is not False:
                    errors.append("{}:{} future-label guard missing".format(split, line_number))
                if assign_split(metadata.get("as_of_date"), contract) != split:
                    errors.append("{}:{} date violates purged split".format(split, line_number))
                if ETF_FLOW_POLICY_ID not in str(row.get("context")):
                    errors.append("{}:{} prompt lacks ETF Flow policy".format(split, line_number))
                if set(response) != set(RESPONSE_KEYS):
                    errors.append("{}:{} response contract mismatch".format(split, line_number))
                if len(str(row.get("context") or "")) > 100_000:
                    warnings.append("{}:{} context exceeds 100k characters".format(split, line_number))
        observed_counts[split] = count
        if expected.get("rows") != count:
            errors.append("{} row-count mismatch".format(split))

    sealed_test = manifest.get("sealed_test_reference")
    if sealed_test:
        sealed_path = Path(str(sealed_test.get("path") or ""))
        if not sealed_path.is_file():
            errors.append("sealed test reference is missing")
        elif sealed_test.get("sha256") != _sha256_file(sealed_path):
            errors.append("sealed test reference SHA256 mismatch")
        if sealed_test.get("used_by_training") is not False:
            errors.append("sealed test is not marked training-inaccessible")

    return {
        "ok": not errors,
        "dataset_root": str(dataset_root),
        "observed_counts": observed_counts,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args(argv)
    result = validate_dataset(args.dataset_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
