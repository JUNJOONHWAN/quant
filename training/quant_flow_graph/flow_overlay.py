"""Build an isolated ETF Flow cache overlay without mutating canonical stores."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Sequence

from training.quant_forecast_v2.flow import FLOW_CACHE_SCHEMA, _available_session
from training.quant_forecast_v2.io_utils import sha256_file, utc_now, write_json_atomic
from training.quant_forecast_v2.source import SourceBundle, canonical_symbol

from .contracts import (
    DEFAULT_BASE_DATABASE,
    DEFAULT_FLOW_BACKFILL_DATABASE,
    DEFAULT_FLOW_CACHE,
    DEFAULT_FLOW_SOURCE_ROOT,
    DEFAULT_INCREMENTAL_DATABASE,
    DEFAULT_REPAIRED_FLOW_CACHE,
)


OVERLAY_SCHEMA_VERSION = "quant.etf_flow_cache_overlay.v1"


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    allowed = DEFAULT_FLOW_SOURCE_ROOT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"overlay output must stay below {allowed}: {resolved}")
    return resolved


def build_overlay_cache(
    *,
    base_cache: Path,
    backfill_database: Path,
    base_database: Path,
    incremental_database: Path | None,
    output_path: Path,
    replace: bool,
) -> dict[str, object]:
    """Copy the derived cache and insert only keys absent from the canonical cache."""

    base_cache = Path(base_cache)
    backfill_database = Path(backfill_database)
    output_path = _safe_output(output_path)
    if not base_cache.is_file():
        raise FileNotFoundError(base_cache)
    if not backfill_database.is_file():
        raise FileNotFoundError(backfill_database)
    if output_path.exists() and not replace:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with SourceBundle(base_database, incremental_database) as source:
        sessions = source.sessions()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".building", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    started_at = utc_now()
    try:
        shutil.copyfile(base_cache, temporary)
        connection = sqlite3.connect(temporary)
        source_connection = sqlite3.connect(
            f"file:{backfill_database}?mode=ro", uri=True
        )
        try:
            source_connection.execute("PRAGMA query_only=ON")
            schema = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if not schema or schema[0] != FLOW_CACHE_SCHEMA:
                raise ValueError("base flow cache schema mismatch")
            rows_before = int(connection.execute("SELECT COUNT(*) FROM flow").fetchone()[0])
            input_rows = 0
            invalid_rows = 0
            normalized: list[tuple[object, ...]] = []
            for row in source_connection.execute(
                "SELECT ticker,effective_date,processed_date,fund_flow,nav,"
                "shares_outstanding FROM etf_flow_observations "
                "WHERE provider='massive' ORDER BY effective_date,ticker"
            ):
                input_rows += 1
                ticker = canonical_symbol(row[0])
                effective = str(row[1])
                processed = str(row[2])
                fund = _number(row[3])
                nav = _number(row[4])
                shares = _number(row[5])
                available = _available_session(sessions, effective, processed)
                assets = (
                    nav * shares
                    if nav is not None and nav > 0 and shares is not None and shares > 0
                    else None
                )
                rate = fund / assets * 100.0 if fund is not None and assets else None
                if (
                    not ticker
                    or available is None
                    or fund is None
                    or nav is None
                    or shares is None
                    or rate is None
                    or abs(rate) > 100.0
                ):
                    invalid_rows += 1
                    continue
                normalized.append(
                    (ticker, effective, processed, available, rate, fund, nav, shares)
                )
            changes_before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO flow VALUES(?,?,?,?,?,?,?,?)", normalized
            )
            inserted_rows = connection.total_changes - changes_before
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES('overlay_schema_version',?)",
                (OVERLAY_SCHEMA_VERSION,),
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES('overlay_source_sha256',?)",
                (sha256_file(backfill_database),),
            )
            connection.commit()
            rows_after = int(connection.execute("SELECT COUNT(*) FROM flow").fetchone()[0])
            per_date = [
                {"effective_date": str(row[0]), "rows": int(row[1])}
                for row in connection.execute(
                    "SELECT effective_date,COUNT(*) FROM flow "
                    "WHERE effective_date BETWEEN '2026-07-14' AND '2026-07-22' "
                    "GROUP BY effective_date ORDER BY effective_date"
                )
            ]
        finally:
            source_connection.close()
            connection.close()
        if output_path.exists():
            output_path.unlink()
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    source_manifest_path = backfill_database.parent.parent / "state/dataset_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else None
    )
    receipt = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "started_at_utc": started_at,
        "policy": "copy canonical cache; insert only previously absent ticker/effective keys",
        "pit_limit": (
            "backfill is historical_window_captured, not an as-observed archive; "
            "eligibility is reconstructed from provider effective/processed dates"
        ),
        "base_cache": {
            "path": str(base_cache),
            "sha256": sha256_file(base_cache),
            "rows": rows_before,
        },
        "backfill_database": {
            "path": str(backfill_database),
            "sha256": sha256_file(backfill_database),
            "manifest": source_manifest,
            "input_rows": input_rows,
            "invalid_rows": invalid_rows,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": rows_after,
            "inserted_rows": inserted_rows,
            "duplicate_or_existing_rows": len(normalized) - inserted_rows,
        },
        "per_date": per_date,
        "quality_gate": "PASS" if rows_after == rows_before + inserted_rows else "FAIL",
    }
    write_json_atomic(output_path.with_suffix(".receipt.json"), receipt)
    if receipt["quality_gate"] != "PASS":
        raise RuntimeError("flow overlay receipt failed")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_FLOW_CACHE)
    parser.add_argument(
        "--backfill-database", type=Path, default=DEFAULT_FLOW_BACKFILL_DATABASE
    )
    parser.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    parser.add_argument(
        "--incremental-database", type=Path, default=DEFAULT_INCREMENTAL_DATABASE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPAIRED_FLOW_CACHE)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_overlay_cache(
        base_cache=args.base_cache,
        backfill_database=args.backfill_database,
        base_database=args.base_database,
        incremental_database=args.incremental_database,
        output_path=args.output,
        replace=args.replace,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
