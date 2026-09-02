"""Persistent ETF-to-security relation index for Quant AI Radar.

The source databases remain immutable/read-only.  This derived index converts
the expensive historical DISTINCT scans into a one-time build and then follows
append-only source markers.  A Radar run may consume the index only when it is
bound to the exact sealed Oracle source fingerprint and as-of date.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from quant_dataset.shared_market import SharedMarketBinding


SCHEMA_VERSION = "quant.ai_radar_relation_index.v1"
DEFAULT_RELATION_INDEX = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/"
    "shared/etf_relation_index.sqlite3"
)


class RelationIndexError(RuntimeError):
    """The derived relation index is missing, stale, or inconsistent."""


@dataclass(frozen=True)
class SourceSpec:
    key: str
    database_role: str
    table: str
    marker: str
    symbol_column: str
    relation_type: str
    visible_date_sql: str
    eligibility_sql: str = "1=1"


SOURCE_SPECS = (
    SourceSpec(
        key="base_flow_versions",
        database_role="base",
        table="etf_flow_versions",
        marker="id",
        symbol_column="ticker",
        relation_type="massive_etf_flow",
        visible_date_sql=(
            "CASE WHEN processed_date>effective_date "
            "THEN processed_date ELSE effective_date END"
        ),
        eligibility_sql="fund_flow IS NOT NULL",
    ),
    SourceSpec(
        key="incremental_flow_versions",
        database_role="incremental",
        table="etf_flow_versions",
        marker="id",
        symbol_column="ticker",
        relation_type="massive_etf_flow",
        visible_date_sql=(
            "CASE WHEN processed_date>effective_date "
            "THEN processed_date ELSE effective_date END"
        ),
        eligibility_sql="fund_flow IS NOT NULL",
    ),
    SourceSpec(
        key="base_constituent_snapshots",
        database_role="base",
        table="etf_constituent_snapshots",
        marker="rowid",
        symbol_column="etf_ticker",
        relation_type="fmp_etf_constituents",
        visible_date_sql=(
            "CASE WHEN available_date>effective_date "
            "THEN available_date ELSE effective_date END"
        ),
    ),
    SourceSpec(
        key="incremental_constituent_snapshots",
        database_role="incremental",
        table="etf_constituent_snapshots",
        marker="rowid",
        symbol_column="etf_ticker",
        relation_type="fmp_etf_constituents",
        visible_date_sql=(
            "CASE WHEN available_date>effective_date "
            "THEN available_date ELSE effective_date END"
        ),
    ),
    SourceSpec(
        key="base_constituent_members",
        database_role="base",
        table="etf_constituent_observations",
        marker="rowid",
        symbol_column="constituent_ticker",
        relation_type="fmp_etf_membership",
        visible_date_sql=(
            "CASE WHEN available_date>effective_date "
            "THEN available_date ELSE effective_date END"
        ),
        eligibility_sql=(
            "constituent_ticker IS NOT NULL AND TRIM(constituent_ticker)<>''"
        ),
    ),
    SourceSpec(
        key="incremental_constituent_members",
        database_role="incremental",
        table="etf_constituent_observations",
        marker="rowid",
        symbol_column="constituent_ticker",
        relation_type="fmp_etf_membership",
        visible_date_sql=(
            "CASE WHEN available_date>effective_date "
            "THEN available_date ELSE effective_date END"
        ),
        eligibility_sql=(
            "constituent_ticker IS NOT NULL AND TRIM(constituent_ticker)<>''"
        ),
    ),
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _connect_source(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS relation_symbols(
            symbol TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            first_visible_date TEXT NOT NULL,
            last_source_key TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY(symbol,relation_type)
        );
        CREATE INDEX IF NOT EXISTS idx_relation_symbols_visible
            ON relation_symbols(first_visible_date,relation_type,symbol);
        CREATE TABLE IF NOT EXISTS source_markers(
            source_key TEXT PRIMARY KEY,
            database_path TEXT NOT NULL,
            table_name TEXT NOT NULL,
            marker_column TEXT NOT NULL,
            last_marker INTEGER NOT NULL,
            updated_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        """
    )


def _upsert_relation(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    relation_type: str,
    first_visible_date: str,
    source_key: str,
    now: str,
) -> None:
    normalized = symbol.strip().upper()
    if not normalized:
        return
    try:
        visible = date.fromisoformat(str(first_visible_date)).isoformat()
    except ValueError as exc:
        raise RelationIndexError(
            f"invalid visible date from {source_key}: {first_visible_date}"
        ) from exc
    connection.execute(
        """
        INSERT INTO relation_symbols(
            symbol,relation_type,first_visible_date,last_source_key,updated_at_utc
        ) VALUES(?,?,?,?,?)
        ON CONFLICT(symbol,relation_type) DO UPDATE SET
            first_visible_date=MIN(
                relation_symbols.first_visible_date,
                excluded.first_visible_date
            ),
            last_source_key=excluded.last_source_key,
            updated_at_utc=excluded.updated_at_utc
        """,
        (normalized, relation_type, visible, source_key, now),
    )


def _source_marker(connection: sqlite3.Connection, spec: SourceSpec) -> int:
    row = connection.execute(
        f"SELECT COALESCE(MAX({spec.marker}),0) FROM {spec.table}"
    ).fetchone()
    return int(row[0] or 0)


def _existing_marker(
    connection: sqlite3.Connection, spec: SourceSpec
) -> tuple[bool, int]:
    row = connection.execute(
        "SELECT last_marker FROM source_markers WHERE source_key=?",
        (spec.key,),
    ).fetchone()
    return (row is not None, int(row[0]) if row else 0)


def _full_rows(
    source: sqlite3.Connection, spec: SourceSpec
) -> Iterable[sqlite3.Row]:
    return source.execute(
        f"""
        SELECT UPPER(TRIM({spec.symbol_column})) AS symbol,
               MIN({spec.visible_date_sql}) AS first_visible_date
        FROM {spec.table}
        WHERE {spec.eligibility_sql}
        GROUP BY UPPER(TRIM({spec.symbol_column}))
        """
    )


def _incremental_rows(
    source: sqlite3.Connection, spec: SourceSpec, previous_marker: int
) -> Iterable[sqlite3.Row]:
    return source.execute(
        f"""
        SELECT UPPER(TRIM({spec.symbol_column})) AS symbol,
               {spec.visible_date_sql} AS first_visible_date
        FROM {spec.table}
        WHERE {spec.marker}>? AND {spec.eligibility_sql}
        ORDER BY {spec.marker}
        """,
        (previous_marker,),
    )


def _database_for(binding: SharedMarketBinding, role: str) -> Path:
    if role == "base":
        return binding.base_database
    if role == "incremental":
        return binding.incremental_database
    raise RelationIndexError(f"unsupported database role: {role}")


def refresh_relation_index(
    binding: SharedMarketBinding,
    index_path: Path = DEFAULT_RELATION_INDEX,
) -> dict[str, Any]:
    """Build once, then ingest only source rows beyond persisted markers."""

    started = time.monotonic()
    resolved = Path(index_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    full_scan_sources: list[str] = []
    incremental_source_rows: dict[str, int] = {}
    source_markers: dict[str, int] = {}

    with sqlite3.connect(resolved, timeout=120) as index:
        index.row_factory = sqlite3.Row
        _initialize(index)
        index.execute("BEGIN IMMEDIATE")
        try:
            for spec in SOURCE_SPECS:
                database_path = _database_for(binding, spec.database_role)
                marker_exists, previous_marker = _existing_marker(index, spec)
                with _connect_source(database_path) as source:
                    current_marker = _source_marker(source, spec)
                    if previous_marker > current_marker:
                        raise RelationIndexError(
                            f"source marker regressed for {spec.key}: "
                            f"{previous_marker}>{current_marker}"
                        )
                    if not marker_exists:
                        rows = _full_rows(source, spec)
                        full_scan_sources.append(spec.key)
                    else:
                        rows = _incremental_rows(source, spec, previous_marker)
                    ingested = 0
                    for row in rows:
                        _upsert_relation(
                            index,
                            symbol=str(row["symbol"] or ""),
                            relation_type=spec.relation_type,
                            first_visible_date=str(row["first_visible_date"] or ""),
                            source_key=spec.key,
                            now=now,
                        )
                        ingested += 1
                index.execute(
                    """
                    INSERT INTO source_markers(
                        source_key,database_path,table_name,marker_column,
                        last_marker,updated_at_utc
                    ) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(source_key) DO UPDATE SET
                        database_path=excluded.database_path,
                        table_name=excluded.table_name,
                        marker_column=excluded.marker_column,
                        last_marker=excluded.last_marker,
                        updated_at_utc=excluded.updated_at_utc
                    """,
                    (
                        spec.key,
                        str(database_path),
                        spec.table,
                        spec.marker,
                        current_marker,
                        now,
                    ),
                )
                incremental_source_rows[spec.key] = ingested
                source_markers[spec.key] = current_marker

            relation_counts = {
                str(row["relation_type"]): int(row["row_count"])
                for row in index.execute(
                    """
                    SELECT relation_type,COUNT(*) AS row_count
                    FROM relation_symbols GROUP BY relation_type
                    ORDER BY relation_type
                    """
                )
            }
            core = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "index_path": str(resolved),
                "target_as_of_date": binding.target_as_of_date,
                "shared_source_fingerprint_sha256": (
                    binding.source_fingerprint_sha256
                ),
                "base_database": str(binding.base_database),
                "incremental_database": str(binding.incremental_database),
                "source_markers": source_markers,
                "relation_counts": relation_counts,
                "full_scan_sources": full_scan_sources,
                "source_rows_processed_this_refresh": incremental_source_rows,
                "refreshed_at_utc": now,
            }
            manifest = {
                **core,
                "content_sha256": _content_sha256(core),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            index.execute(
                """
                INSERT INTO metadata(key,value_json) VALUES('manifest',?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json
                """,
                (_canonical(manifest),),
            )
            index.commit()
        except Exception:
            index.rollback()
            raise
    return manifest


def load_verified_relation_index(
    binding: SharedMarketBinding,
    index_path: Path = DEFAULT_RELATION_INDEX,
) -> dict[str, Any]:
    """Fail closed unless the index is bound to the exact sealed source."""

    resolved = Path(index_path).expanduser().resolve()
    if not resolved.is_file():
        raise RelationIndexError(f"relation index is missing: {resolved}")
    with sqlite3.connect(f"file:{resolved}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT value_json FROM metadata WHERE key='manifest'"
        ).fetchone()
    if not row:
        raise RelationIndexError("relation index manifest is missing")
    try:
        manifest = json.loads(str(row[0]))
    except json.JSONDecodeError as exc:
        raise RelationIndexError("relation index manifest is invalid") from exc
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "target_as_of_date": binding.target_as_of_date,
        "shared_source_fingerprint_sha256": binding.source_fingerprint_sha256,
        "base_database": str(binding.base_database),
        "incremental_database": str(binding.incremental_database),
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RelationIndexError(
            "relation index binding mismatch: " + _canonical(mismatches)
        )
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"content_sha256", "elapsed_seconds"}
    }
    if manifest.get("content_sha256") != _content_sha256(core):
        raise RelationIndexError("relation index manifest content hash mismatch")
    return manifest


def visible_relation_sets(
    index_path: Path,
    as_of_date: str,
) -> dict[str, set[str]]:
    """Return every relation visible by ``as_of_date`` without history scans."""

    as_of = date.fromisoformat(as_of_date).isoformat()
    resolved = Path(index_path).expanduser().resolve()
    relations = {
        "massive_etf_flow": set(),
        "fmp_etf_constituents": set(),
        "fmp_etf_membership": set(),
    }
    with sqlite3.connect(f"file:{resolved}?mode=ro", uri=True) as connection:
        for symbol, relation_type in connection.execute(
            """
            SELECT symbol,relation_type FROM relation_symbols
            WHERE first_visible_date<=?
            ORDER BY relation_type,symbol
            """,
            (as_of,),
        ):
            if relation_type in relations:
                relations[str(relation_type)].add(str(symbol))
    return relations
