"""Scan the full FMP daily universe and select deterministic task-proxy pairs.

This is the disk-safe first pass.  It scans every collected FMP symbol-session
row but stores only the lowest salted hashes within each time split and proxy
task.  Exact point-in-time task classification and all expensive preprocessing
gates happen later when the selected pairs are materialized in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from training.quant_llm.build_balanced_training_set import TASK_TYPES, _parse_quotas
from training.quant_llm.build_sft_dataset import (
    assign_split,
    load_trading_sessions,
    split_contract,
)


DEFAULT_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/"
    "daily_observations.sqlite3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/pair_selections/"
    "qwen3_8b_pairs_v2"
)
DEFAULT_TRAIN_QUOTAS = {
    "etf_own_flow_analysis": 40_000,
    "stock_constituent_flow_analysis": 50_000,
    "all_stock_control_analysis": 30_000,
}
SELECTION_SALT = "quant.qwen3_8b_pair_selection.v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pair-manifest-", dir=str(path.parent))
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


def _date_maps(connection: sqlite3.Connection) -> Tuple[Dict[str, str], Dict[str, str]]:
    flow_first = {
        str(ticker): str(first_date)
        for ticker, first_date in connection.execute(
            """
            SELECT ticker, MIN(processed_date)
            FROM etf_flow_versions
            WHERE ticker IS NOT NULL AND processed_date IS NOT NULL
            GROUP BY ticker
            """
        )
    }
    membership_first = {
        str(symbol): str(first_date)
        for symbol, first_date in connection.execute(
            """
            SELECT constituent_ticker, MIN(available_date)
            FROM etf_constituent_observations
            WHERE constituent_ticker IS NOT NULL AND constituent_ticker<>''
              AND available_date IS NOT NULL
            GROUP BY constituent_ticker
            """
        )
    }
    return flow_first, membership_first


def _proxy_task(
    symbol: str,
    as_of_date: str,
    flow_first: Mapping[str, str],
    membership_first: Mapping[str, str],
) -> str:
    if flow_first.get(symbol, "9999-12-31") <= as_of_date:
        return "etf_own_flow_analysis"
    if membership_first.get(symbol, "9999-12-31") <= as_of_date:
        return "stock_constituent_flow_analysis"
    return "all_stock_control_analysis"


def _pair_hash(split: str, task: str, symbol: str, as_of_date: str) -> str:
    payload = "{}:{}:{}:{}:{}".format(
        SELECTION_SALT, split, task, symbol, as_of_date
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _consider(heap: list, quota: int, item: tuple) -> None:
    key = int(item[0], 16)
    candidate = (-key,) + item
    if len(heap) < quota:
        heapq.heappush(heap, candidate)
    elif key < -heap[0][0]:
        heapq.heapreplace(heap, candidate)


def select_pairs(
    database_path: Path,
    output_root: Path,
    *,
    start_date: str,
    end_date: str,
    train_quotas: Mapping[str, int],
    validation_per_task: int,
    test_per_task: int,
    embargo_sessions: int,
    replace: bool,
) -> dict:
    database = Path(database_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if validation_per_task < 1 or test_per_task < 1:
        raise ValueError("validation/test per-task quotas must be positive")
    quotas = {
        "train": dict(train_quotas),
        "validation": {task: validation_per_task for task in TASK_TYPES},
        "test": {task: test_per_task for task in TASK_TYPES},
    }
    sessions = load_trading_sessions(database)
    contract = split_contract(sessions, "2024-01-01", "2025-01-01", embargo_sessions)
    targets = [output / "pairs.jsonl", output / "manifest.json"]
    if any(path.exists() for path in targets) and not replace:
        raise FileExistsError("pair selection exists; pass --replace or use a new root")

    heaps = {
        (split, task): [] for split in ("train", "validation", "test") for task in TASK_TYPES
    }
    proxy_counts = {
        split: {task: 0 for task in TASK_TYPES}
        for split in ("train", "validation", "test")
    }
    scanned = 0
    cheap_eligible = 0
    exclusions = {
        "outside_requested_range": 0,
        "purged_embargo_or_outside_split": 0,
        "nonpositive_or_missing_close": 0,
        "nonpositive_or_missing_volume": 0,
    }
    connection = sqlite3.connect("file:{}?mode=ro".format(database), uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        flow_first, membership_first = _date_maps(connection)
        cursor = connection.execute(
            """
            SELECT symbol, trade_date, close, volume
            FROM daily_observations
            WHERE source=? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date, symbol
            """,
            ("fmp", start_date, end_date),
        )
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            for symbol, as_of_date, close, volume in rows:
                scanned += 1
                if close is None or float(close) <= 0:
                    exclusions["nonpositive_or_missing_close"] += 1
                    continue
                if volume is None or float(volume) <= 0:
                    exclusions["nonpositive_or_missing_volume"] += 1
                    continue
                split = assign_split(str(as_of_date), contract)
                if split is None:
                    exclusions["purged_embargo_or_outside_split"] += 1
                    continue
                cheap_eligible += 1
                normalized_symbol = str(symbol).upper()
                task = _proxy_task(
                    normalized_symbol,
                    str(as_of_date),
                    flow_first,
                    membership_first,
                )
                proxy_counts[split][task] += 1
                digest = _pair_hash(split, task, normalized_symbol, str(as_of_date))
                _consider(
                    heaps[(split, task)],
                    int(quotas[split][task]),
                    (digest, normalized_symbol, str(as_of_date), split, task),
                )
    finally:
        connection.close()

    selected_rows = []
    selected_counts = {
        split: {task: 0 for task in TASK_TYPES}
        for split in ("train", "validation", "test")
    }
    for split in ("train", "validation", "test"):
        for task in TASK_TYPES:
            for item in heaps[(split, task)]:
                _, digest, symbol, as_of_date, _, _ = item
                selected_rows.append(
                    {
                        "pair_hash": digest,
                        "symbol": symbol,
                        "as_of_date": as_of_date,
                        "split": split,
                        "proxy_task_type": task,
                    }
                )
                selected_counts[split][task] += 1
    selected_rows.sort(
        key=lambda row: (
            row["as_of_date"],
            row["symbol"],
            row["proxy_task_type"],
            row["pair_hash"],
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pairs-", dir=str(output))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in selected_rows:
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
        "schema_version": "quant.training_pair_selection.v1",
        "complete": True,
        "database": str(database),
        "database_size_bytes": database.stat().st_size,
        "source": "fmp daily_observations full scan",
        "requested_range": {"from": start_date, "to": end_date},
        "split_contract": contract,
        "selection_salt": SELECTION_SALT,
        "selection_method": "lowest salted SHA256 within split and historical task proxy",
        "survivorship_policy": "present-day active lists are not used",
        "proxy_contract": {
            "etf_own_flow_analysis": "historical Massive ticker with processed_date <= as_of",
            "stock_constituent_flow_analysis": "historical FMP membership available_date <= as_of",
            "all_stock_control_analysis": "neither historical proxy is visible as-of",
            "exact_reclassification_required": True,
        },
        "scanned_fmp_symbol_sessions": scanned,
        "cheap_eligible_symbol_sessions": cheap_eligible,
        "exclusion_counts": exclusions,
        "proxy_counts": proxy_counts,
        "quotas": quotas,
        "selected_counts": selected_counts,
        "pairs_file": {
            "filename": "pairs.jsonl",
            "rows": len(selected_rows),
            "bytes": (output / "pairs.jsonl").stat().st_size,
            "sha256": _sha256_file(output / "pairs.jsonl"),
        },
        "full_packet_materialization": False,
        "reason": "source DB is preserved; all rich packets would require multi-terabyte duplicate storage",
        "representative_sample_claimed": False,
    }
    _write_atomic(output / "manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--from", dest="start_date", default="2017-01-01")
    parser.add_argument("--to", dest="end_date", default="2026-07-14")
    parser.add_argument(
        "--train-quotas",
        default=",".join(
            "{}={}".format(task, DEFAULT_TRAIN_QUOTAS[task]) for task in TASK_TYPES
        ),
    )
    parser.add_argument("--validation-per-task", type=int, default=5_000)
    parser.add_argument("--test-per-task", type=int, default=10_000)
    parser.add_argument("--embargo-sessions", type=int, default=20)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    if args.start_date > args.end_date:
        parser.error("--from must be <= --to")
    result = select_pairs(
        args.database,
        args.output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        train_quotas=_parse_quotas(args.train_quotas),
        validation_per_task=args.validation_per_task,
        test_per_task=args.test_per_task,
        embargo_sessions=args.embargo_sessions,
        replace=args.replace,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
