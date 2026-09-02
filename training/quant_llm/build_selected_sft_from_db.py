"""Materialize selected pairs directly into compact SFT rows with resume state.

Rich v3 packets are built and validated in memory, then discarded.  Immutable
raw artifacts remain in the quant source database; the SFT row retains bounded
evidence plus provenance-set digests.  This avoids multi-terabyte duplicated
packet storage while preserving a deterministic regeneration path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from quant_dataset.config import DEFAULT_SECRETS_PATH, load_credentials, resolve_data_root
from quant_dataset.pipeline import DatasetPipeline
from quant_dataset.point_in_time import ETF_FLOW_POLICY_ID
from training.quant_llm import DATASET_CONTRACT_VERSION, DATASET_SCHEMA_VERSION
from training.quant_llm.build_sft_dataset import (
    DEFAULT_MIN_ETF_MEDIAN_DOLLAR_VOLUME,
    DEFAULT_MIN_ETF_NONZERO_VOLUME_RATIO,
    DEFAULT_MIN_ETF_OBSERVED_SESSIONS,
    ETF_LIQUIDITY_POLICY_ID,
    ETF_LIQUIDITY_WINDOW_SESSIONS,
    REQUIRED_PACKET_SCHEMA,
    assign_split,
    build_example,
    canonical_json,
    packet_eligibility,
    validate_packet,
)


DEFAULT_PAIR_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/pair_selections/"
    "qwen3_8b_pairs_v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/"
    "qwen3_8b_candidate_v2"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _set_metadata(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute(
        """
        INSERT INTO metadata(key,value) VALUES (?,?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, str(value)),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialization_contract(
    *,
    pair_manifest_path: Path,
    pairs_path: Path,
    data_root: Path,
    lookback_days: int,
    min_etf_observed_sessions: int,
    min_etf_nonzero_volume_ratio: float,
    min_etf_median_dollar_volume: float,
) -> dict:
    data = Path(data_root).expanduser().resolve()
    return {
        "pair_manifest_sha256": _sha256_file(pair_manifest_path),
        "pairs_sha256": _sha256_file(pairs_path),
        "data_root": str(data),
        "database": str(data / "normalized" / "daily_observations.sqlite3"),
        "lookback_price_sessions": lookback_days,
        "flow_observations": min(20, lookback_days),
        "min_etf_observed_sessions": min_etf_observed_sessions,
        "min_etf_nonzero_volume_ratio": min_etf_nonzero_volume_ratio,
        "min_etf_median_dollar_volume": min_etf_median_dollar_volume,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
    }


def _verify_seed_pair_subset(seed_pairs_path: Path, extended_pairs_path: Path) -> dict:
    """Prove every sealed seed pair is unchanged in the extended selection."""

    seed_rows: Dict[str, str] = {}
    with Path(seed_pairs_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_hash = str(row.get("pair_hash") or "")
            if not pair_hash:
                raise ValueError("seed pairs line {} has no pair_hash".format(line_number))
            encoded = canonical_json(row)
            if pair_hash in seed_rows:
                raise ValueError("duplicate pair_hash in seed selection: {}".format(pair_hash))
            seed_rows[pair_hash] = encoded
    seed_count = len(seed_rows)
    matched = 0
    extended_count = 0
    seen_extended = set()
    with Path(extended_pairs_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_hash = str(row.get("pair_hash") or "")
            if not pair_hash:
                raise ValueError(
                    "extended pairs line {} has no pair_hash".format(line_number)
                )
            if pair_hash in seen_extended:
                raise ValueError(
                    "duplicate pair_hash in extended selection: {}".format(pair_hash)
                )
            seen_extended.add(pair_hash)
            extended_count += 1
            prior = seed_rows.get(pair_hash)
            if prior is not None:
                if prior != canonical_json(row):
                    raise ValueError(
                        "extended selection changed sealed seed pair {}".format(pair_hash)
                    )
                matched += 1
                del seed_rows[pair_hash]
    if seed_rows:
        raise ValueError(
            "extended selection is missing {} sealed seed pairs".format(len(seed_rows))
        )
    if extended_count < seed_count:
        raise ValueError("extended selection is smaller than the seed selection")
    return {
        "seed_pairs": seed_count,
        "extended_pairs": extended_count,
        "matched_unchanged_seed_pairs": matched,
        "additional_pairs": extended_count - seed_count,
    }


def _seed_state_from_completed_output(
    *,
    seed_output_root: Path,
    extended_pair_manifest_path: Path,
    extended_pairs_path: Path,
    destination_state_path: Path,
    contract_sha: str,
) -> dict:
    seed_root = Path(seed_output_root).expanduser().resolve()
    seed_manifest_path = seed_root / "manifest.json"
    seed_state_path = seed_root / "materialization_state.sqlite3"
    if not seed_manifest_path.is_file() or not seed_state_path.is_file():
        raise ValueError("seed output must contain a completed manifest and state database")
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    seed_selection = seed_manifest.get("input_pair_selection") or {}
    seed_pairs_path = Path(str(seed_selection.get("pairs") or ""))
    if not seed_pairs_path.is_file():
        raise ValueError("seed manifest pair file is missing")
    if _sha256_file(seed_pairs_path) != seed_selection.get("pairs_sha256"):
        raise ValueError("seed pair SHA256 differs from the completed manifest")
    expected_seed_examples = 0
    for split, file_info in (seed_manifest.get("files") or {}).items():
        split_path = seed_root / str(file_info.get("filename") or "")
        if not split_path.is_file():
            raise ValueError("seed {} split file is missing".format(split))
        if _sha256_file(split_path) != file_info.get("sha256"):
            raise ValueError("seed {} split SHA256 mismatch".format(split))
        expected_seed_examples += int(file_info.get("rows") or 0)
    extended_manifest = json.loads(
        Path(extended_pair_manifest_path).read_text(encoding="utf-8")
    )
    if seed_manifest.get("split_contract") != extended_manifest.get("split_contract"):
        raise ValueError("extended selection changed the sealed time split contract")
    subset = _verify_seed_pair_subset(seed_pairs_path, extended_pairs_path)
    expected_seed_pairs = int(seed_selection.get("selected_pairs") or 0)
    if subset["seed_pairs"] != expected_seed_pairs:
        raise ValueError("seed pair count differs from the completed candidate manifest")

    source = sqlite3.connect("file:{}?mode=ro".format(seed_state_path), uri=True)
    destination = sqlite3.connect(str(destination_state_path), timeout=120)
    try:
        seed_result_count = int(
            source.execute("SELECT COUNT(*) FROM pair_results").fetchone()[0]
        )
        seed_error_count = int(
            source.execute(
                "SELECT COUNT(*) FROM pair_results WHERE status='error'"
            ).fetchone()[0]
        )
        seed_example_count = int(source.execute("SELECT COUNT(*) FROM examples").fetchone()[0])
        state_contract = source.execute(
            "SELECT value FROM metadata WHERE key='contract_sha256'"
        ).fetchone()
        if (
            seed_result_count != expected_seed_pairs
            or seed_error_count
            or seed_example_count != expected_seed_examples
            or not state_contract
            or str(state_contract[0]) != str(seed_manifest.get("materialization_contract_sha256"))
        ):
            raise ValueError("seed state is incomplete or contains errors")
        source.backup(destination)
        destination.execute("PRAGMA journal_mode=WAL")
        now = _utc_now()
        prior_contract = destination.execute(
            "SELECT value FROM metadata WHERE key='contract_sha256'"
        ).fetchone()
        for key in (
            "completed_at_utc",
            "last_error",
            "last_as_of_date",
            "last_progress_at_utc",
        ):
            destination.execute("DELETE FROM metadata WHERE key=?", (key,))
        _set_metadata(destination, "seed_contract_sha256", prior_contract[0] if prior_contract else "")
        _set_metadata(destination, "contract_sha256", contract_sha)
        _set_metadata(destination, "seed_output_root", seed_root)
        _set_metadata(destination, "seed_candidate_manifest", seed_manifest_path)
        _set_metadata(destination, "seed_candidate_manifest_sha256", _sha256_file(seed_manifest_path))
        _set_metadata(destination, "seed_pair_count", expected_seed_pairs)
        _set_metadata(destination, "extension_started_at_utc", now)
        _set_metadata(destination, "started_at_utc", now)
        _set_metadata(destination, "processed_pairs", expected_seed_pairs)
        _set_metadata(destination, "run_status", "seeded_extension")
        destination.commit()
    finally:
        source.close()
        destination.close()
    return {
        **subset,
        "seed_output_root": str(seed_root),
        "seed_candidate_manifest": str(seed_manifest_path),
        "seed_candidate_manifest_sha256": _sha256_file(seed_manifest_path),
    }


def _write_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".selected-sft-manifest-", dir=str(path.parent))
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


def _open_state(path: Path, contract_sha: str) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pair_results (
            pair_hash TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            declared_split TEXT NOT NULL,
            proxy_task_type TEXT NOT NULL,
            status TEXT NOT NULL,
            actual_task_type TEXT,
            example_id TEXT,
            reasons_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS examples (
            example_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            task_type TEXT NOT NULL,
            packet_content_sha256 TEXT NOT NULL UNIQUE,
            encoded_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_examples_split_task_id
            ON examples(split, task_type, example_id);
        """
    )
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='contract_sha256'"
    ).fetchone()
    if existing and str(existing[0]) != contract_sha:
        raise ValueError("resume state contract differs; use a new output root")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key,value) VALUES ('contract_sha256',?)",
        (contract_sha,),
    )
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key,value) VALUES ('started_at_utc',?)",
        (_utc_now(),),
    )
    _set_metadata(connection, "run_status", "running")
    connection.commit()
    return connection


def _pair_groups(path: Path) -> Iterable[List[dict]]:
    group: List[dict] = []
    active_date = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            as_of = str(row.get("as_of_date") or "")
            if not as_of:
                raise ValueError("{}:{} missing as_of_date".format(path, line_number))
            if active_date is not None and as_of < active_date:
                raise ValueError("pairs file must be ordered by as_of_date")
            if active_date is not None and as_of != active_date:
                yield group
                group = []
            active_date = as_of
            group.append(row)
    if group:
        yield group


def _export_examples(
    state: sqlite3.Connection, output: Path, splits: Sequence[str]
) -> Dict[str, dict]:
    files = {}
    for split in splits:
        target = output / "{}.jsonl".format(split)
        descriptor, name = tempfile.mkstemp(prefix=".{}-".format(split), dir=str(output))
        temporary = Path(name)
        rows = 0
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for row in state.execute(
                    "SELECT encoded_json FROM examples WHERE split=? ORDER BY example_id",
                    (split,),
                ):
                    handle.write(str(row[0]) + "\n")
                    rows += 1
                handle.flush()
                os.fsync(handle.fileno())
            if rows == 0:
                raise ValueError("selected materialization produced empty {} split".format(split))
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        files[split] = {
            "filename": target.name,
            "rows": rows,
            "bytes": target.stat().st_size,
            "sha256": _sha256_file(target),
        }
    return files


def build_selected_dataset(
    *,
    pair_root: Path,
    data_root: Path,
    output_root: Path,
    secrets_file: Path,
    lookback_days: int,
    min_etf_observed_sessions: int,
    min_etf_nonzero_volume_ratio: float,
    min_etf_median_dollar_volume: float,
    timeout_seconds: float,
    retries: int,
    replace: bool,
    seed_output_root: Optional[Path] = None,
) -> dict:
    pairs_root = Path(pair_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    pair_manifest_path = pairs_root / "manifest.json"
    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    pairs_path = pairs_root / str(
        (pair_manifest.get("pairs_file") or {}).get("filename") or "pairs.jsonl"
    )
    expected_pairs_sha = (pair_manifest.get("pairs_file") or {}).get("sha256")
    selected_pair_count = int((pair_manifest.get("pairs_file") or {}).get("rows") or 0)
    actual_pairs_sha = _sha256_file(pairs_path)
    if expected_pairs_sha != actual_pairs_sha:
        raise ValueError("pair selection SHA256 mismatch")
    contract = _materialization_contract(
        pair_manifest_path=pair_manifest_path,
        pairs_path=pairs_path,
        data_root=data,
        lookback_days=lookback_days,
        min_etf_observed_sessions=min_etf_observed_sessions,
        min_etf_nonzero_volume_ratio=min_etf_nonzero_volume_ratio,
        min_etf_median_dollar_volume=min_etf_median_dollar_volume,
    )
    contract_sha = hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "materialization_state.sqlite3"
    existing_artifacts = [
        output / "train.jsonl",
        output / "validation.jsonl",
        output / "test.jsonl",
        output / "manifest.json",
    ]
    if replace:
        for path in existing_artifacts + [state_path, Path(str(state_path) + "-wal"), Path(str(state_path) + "-shm")]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    elif (output / "manifest.json").exists():
        raise FileExistsError("completed candidate dataset exists; use a new output root")

    seed_info = None
    if seed_output_root is not None and not state_path.exists():
        seed_info = _seed_state_from_completed_output(
            seed_output_root=seed_output_root,
            extended_pair_manifest_path=pair_manifest_path,
            extended_pairs_path=pairs_path,
            destination_state_path=state_path,
            contract_sha=contract_sha,
        )

    state = _open_state(state_path, contract_sha)
    _set_metadata(state, "pair_manifest", pair_manifest_path)
    _set_metadata(state, "pairs_file", pairs_path)
    _set_metadata(state, "expected_pairs", selected_pair_count)
    state.commit()
    if seed_info is None:
        metadata = {
            str(row[0]): str(row[1])
            for row in state.execute(
                "SELECT key,value FROM metadata WHERE key LIKE 'seed_%'"
            )
        }
        if metadata.get("seed_output_root"):
            seed_info = {
                "seed_output_root": metadata.get("seed_output_root"),
                "seed_candidate_manifest": metadata.get("seed_candidate_manifest"),
                "seed_candidate_manifest_sha256": metadata.get(
                    "seed_candidate_manifest_sha256"
                ),
                "seed_pairs": int(metadata.get("seed_pair_count") or 0),
                "extended_pairs": selected_pair_count,
                "additional_pairs": selected_pair_count
                - int(metadata.get("seed_pair_count") or 0),
                "resumed_from_seeded_state": True,
            }
    credentials = load_credentials(secrets_path=Path(secrets_file).expanduser())
    pipeline = DatasetPipeline(
        data_root=data,
        credentials=credentials,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    split_definition = pair_manifest.get("split_contract") or {}
    try:
        for group in _pair_groups(pairs_path):
            as_of = str(group[0]["as_of_date"])
            pending = []
            for pair in group:
                exists = state.execute(
                    "SELECT 1 FROM pair_results WHERE pair_hash=?",
                    (pair["pair_hash"],),
                ).fetchone()
                if not exists:
                    pending.append(pair)
            if not pending:
                continue
            symbols = sorted({str(pair["symbol"]).upper() for pair in pending})
            pipeline.quality.recompute(as_of, as_of, symbols)
            for pair in pending:
                pair_hash = str(pair["pair_hash"])
                symbol = str(pair["symbol"]).upper()
                declared_split = str(pair["split"])
                proxy_task = str(pair["proxy_task_type"])
                try:
                    actual_split = assign_split(as_of, split_definition)
                    if actual_split != declared_split:
                        raise ValueError("pair split does not match frozen split contract")
                    packet = pipeline.analysis_packet_for_pair(
                        symbol,
                        as_of,
                        lookback_days=lookback_days,
                        recompute_quality=False,
                    )
                    validate_packet(packet)
                    eligibility = packet_eligibility(
                        packet,
                        min_etf_observed_sessions=min_etf_observed_sessions,
                        min_etf_nonzero_volume_ratio=min_etf_nonzero_volume_ratio,
                        min_etf_median_dollar_volume=min_etf_median_dollar_volume,
                    )
                    if not eligibility["eligible"]:
                        state.execute(
                            """
                            INSERT INTO pair_results VALUES (?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                pair_hash,
                                symbol,
                                as_of,
                                declared_split,
                                proxy_task,
                                "excluded",
                                None,
                                None,
                                json.dumps(eligibility["reasons"], sort_keys=True),
                            ),
                        )
                        continue
                    example = build_example(packet)
                    example["metadata"]["split"] = declared_split
                    task_type = str(example["metadata"]["task_type"])
                    encoded = canonical_json(example)
                    content_packet = dict(packet)
                    content_packet.pop("packet_id", None)
                    packet_sha = hashlib.sha256(
                        canonical_json(content_packet).encode("utf-8")
                    ).hexdigest()
                    try:
                        state.execute(
                            "INSERT INTO examples VALUES (?,?,?,?,?)",
                            (
                                example["example_id"],
                                declared_split,
                                task_type,
                                packet_sha,
                                encoded,
                            ),
                        )
                        status = "eligible"
                    except sqlite3.IntegrityError:
                        status = "duplicate"
                    state.execute(
                        "INSERT INTO pair_results VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            pair_hash,
                            symbol,
                            as_of,
                            declared_split,
                            proxy_task,
                            status,
                            task_type,
                            example["example_id"],
                            "[]",
                        ),
                    )
                except Exception as exc:
                    state.execute(
                        "INSERT OR REPLACE INTO pair_results VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            pair_hash,
                            symbol,
                            as_of,
                            declared_split,
                            proxy_task,
                            "error",
                            None,
                            None,
                            json.dumps(["{}: {}".format(type(exc).__name__, exc)]),
                        ),
                    )
                    _set_metadata(state, "run_status", "error")
                    _set_metadata(state, "last_error", "{}: {}".format(type(exc).__name__, exc))
                    _set_metadata(state, "last_progress_at_utc", _utc_now())
                    state.commit()
                    raise
            processed_count = int(
                state.execute("SELECT COUNT(*) FROM pair_results").fetchone()[0]
            )
            _set_metadata(state, "processed_pairs", processed_count)
            _set_metadata(state, "last_as_of_date", as_of)
            _set_metadata(state, "last_progress_at_utc", _utc_now())
            state.commit()

        expected_pairs = selected_pair_count
        result_count = int(state.execute("SELECT COUNT(*) FROM pair_results").fetchone()[0])
        if result_count != expected_pairs:
            raise ValueError(
                "materialization incomplete: {} of {} pair results".format(
                    result_count, expected_pairs
                )
            )
        error_count = int(
            state.execute("SELECT COUNT(*) FROM pair_results WHERE status='error'").fetchone()[0]
        )
        if error_count:
            raise ValueError("materialization state contains {} errors".format(error_count))
        files = _export_examples(state, output, ("train", "validation", "test"))

        status_counts = {
            str(row[0]): int(row[1])
            for row in state.execute(
                "SELECT status,COUNT(*) FROM pair_results GROUP BY status"
            )
        }
        task_counts = defaultdict(dict)
        for split, task, count in state.execute(
            "SELECT split,task_type,COUNT(*) FROM examples GROUP BY split,task_type"
        ):
            task_counts[str(split)][str(task)] = int(count)
        transition_counts = defaultdict(lambda: defaultdict(dict))
        for split, proxy, actual, count in state.execute(
            """
            SELECT declared_split,proxy_task_type,COALESCE(actual_task_type,'excluded'),COUNT(*)
            FROM pair_results GROUP BY declared_split,proxy_task_type,COALESCE(actual_task_type,'excluded')
            """
        ):
            transition_counts[str(split)][str(proxy)][str(actual)] = int(count)
        exclusions = Counter()
        for row in state.execute(
            "SELECT reasons_json FROM pair_results WHERE status='excluded'"
        ):
            for reason in json.loads(row[0]):
                exclusions[str(reason)] += 1

        manifest = {
            "schema_version": "quant.sft.manifest.v1",
            "dataset_contract_version": DATASET_CONTRACT_VERSION,
            "example_schema_version": DATASET_SCHEMA_VERSION,
            "input_packet_schema_required": REQUIRED_PACKET_SCHEMA,
            "model_family": "Qwen/Qwen3-8B",
            "framework": "NVIDIA NeMo AutoModel",
            "training_method": "BF16 LoRA SFT answer-only loss",
            "etf_flow_policy_id": ETF_FLOW_POLICY_ID,
            "future_returns_in_prompt": False,
            "future_returns_in_target": False,
            "trade_execution_target": False,
            "target_origin": "deterministic_auditable_baseline_v1",
            "split_contract": split_definition,
            "input_pair_selection": {
                "manifest": str(pair_manifest_path),
                "manifest_sha256": _sha256_file(pair_manifest_path),
                "pairs": str(pairs_path),
                "pairs_sha256": actual_pairs_sha,
                "selected_pairs": expected_pairs,
                "all_rich_packets_materialized_to_disk": False,
                "rich_packets_built_and_validated_in_memory": True,
            },
            "materialization_contract": contract,
            "materialization_contract_sha256": contract_sha,
            "materialization_seed": seed_info,
            "pair_status_counts": status_counts,
            "actual_task_type_counts": dict(task_counts),
            "proxy_to_actual_transition_counts": {
                split: {proxy: dict(values) for proxy, values in proxies.items()}
                for split, proxies in transition_counts.items()
            },
            "preprocessing_exclusion_counts": dict(sorted(exclusions.items())),
            "preprocessing_policy": {
                "etf_liquidity_policy_id": ETF_LIQUIDITY_POLICY_ID,
                "trailing_window_sessions": ETF_LIQUIDITY_WINDOW_SESSIONS,
                "min_observed_sessions": min_etf_observed_sessions,
                "min_nonzero_volume_ratio": min_etf_nonzero_volume_ratio,
                "min_median_dollar_volume": min_etf_median_dollar_volume,
                "same_session_positive_volume_required": True,
                "all_security_min_observed_sessions": 5,
                "price_basis": "raw close only; adjusted absolute price is forbidden",
                "unexplained_raw_price_discontinuity_gate": "exclude at >=45 percent",
                "flow_version_dedupe": "ticker+effective_date; latest visible revision",
                "example_dedupe": "exact packet content and example id",
                "delisted_security_policy": (
                    "retain historical rows while traded; no rows after last price; "
                    "present-day active lists forbidden for historical filtering"
                ),
                "normalization": (
                    "per-AUM and trailing median/MAD from as-of-visible observations only"
                ),
                "prompt_compaction": {
                    "full_raw_artifacts_preserved_in_source_database": True,
                    "full_token_audit_required_before_training": True,
                    "training_truncation": "disabled_fail_closed",
                },
            },
            "selection": {
                "mode": "full_scan_then_deterministic_proxy_task_reservoir",
                "representative_sample_claimed": False,
                "source_database_deleted_or_modified": False,
            },
            "files": files,
        }
        _write_atomic(output / "manifest.json", manifest)
        _set_metadata(state, "run_status", "complete")
        _set_metadata(state, "processed_pairs", result_count)
        _set_metadata(state, "completed_at_utc", _utc_now())
        _set_metadata(state, "last_progress_at_utc", _utc_now())
        state.commit()
        return manifest
    finally:
        state.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--secrets-file", type=Path, default=DEFAULT_SECRETS_PATH)
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--min-etf-observed-sessions", type=int, default=DEFAULT_MIN_ETF_OBSERVED_SESSIONS)
    parser.add_argument("--min-etf-nonzero-volume-ratio", type=float, default=DEFAULT_MIN_ETF_NONZERO_VOLUME_RATIO)
    parser.add_argument("--min-etf-median-dollar-volume", type=float, default=DEFAULT_MIN_ETF_MEDIAN_DOLLAR_VOLUME)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed-output-root", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    if args.lookback_days < 21:
        parser.error("--lookback-days must be at least 21 for a 20-session return")
    result = build_selected_dataset(
        pair_root=args.pair_root,
        data_root=resolve_data_root(str(args.data_root) if args.data_root else None),
        output_root=args.output_root,
        secrets_file=args.secrets_file,
        lookback_days=args.lookback_days,
        min_etf_observed_sessions=args.min_etf_observed_sessions,
        min_etf_nonzero_volume_ratio=args.min_etf_nonzero_volume_ratio,
        min_etf_median_dollar_volume=args.min_etf_median_dollar_volume,
        timeout_seconds=args.timeout,
        retries=args.retries,
        replace=args.replace,
        seed_output_root=args.seed_output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
