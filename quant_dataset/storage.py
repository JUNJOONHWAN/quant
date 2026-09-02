"""Immutable raw storage and SQLite persistence."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SCHEMA_VERSION = "1"
_SENSITIVE_NAME_PARTS = (
    "apikey",
    "api_key",
    "token",
    "secret",
    "password",
    "authorization",
    "credential",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sensitive_name(name: Any) -> bool:
    lowered = str(name).lower().replace("-", "_")
    return any(part in lowered for part in _SENSITIVE_NAME_PARTS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    return value


def redact_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for key, item in (value or {}).items():
        redacted[str(key)] = (
            "***REDACTED***" if _is_sensitive_name(key) else _redact_value(item)
        )
    return redacted


def _unredacted_sensitive_paths(value: Any, prefix: str = "request") -> List[str]:
    findings: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = "{}.{}".format(prefix, key)
            if _is_sensitive_name(key) and item != "***REDACTED***":
                findings.append(path)
            findings.extend(_unredacted_sensitive_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_unredacted_sensitive_paths(item, "{}[{}]".format(prefix, index)))
    return findings


def redact_url(url: str) -> str:
    split = urlsplit(url)
    safe_query = []
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        safe_query.append((key, "***REDACTED***" if _is_sensitive_name(key) else value))
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(safe_query), split.fragment))


def redacted_request_metadata(
    method: str,
    url: str,
    params: Optional[Mapping[str, Any]],
    logical_request: Optional[Mapping[str, Any]] = None,
) -> dict:
    return {
        "method": method.upper(),
        "url": redact_url(url),
        "params": redact_mapping(params),
        "logical_request": redact_mapping(logical_request),
    }


def _safe_segment(value: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value)).strip("._")
    return segment[:160] or "unknown"


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Atomically create path without ever replacing an existing artifact."""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".capture-", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary_path), str(path))
        except FileExistsError:
            pass
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class RawArtifact:
    artifact_id: int
    capture_event_id: int
    source: str
    dataset: str
    payload_sha256: str
    raw_path: Path
    metadata_path: Path
    captured_at_utc: str
    response_status: int
    payload_bytes: int


class Database:
    """SQLite source of truth for normalized rows, quality, and checkpoints."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root).expanduser()
        self.db_path = self.data_root / "normalized" / "daily_observations.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS raw_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            partition_key TEXT NOT NULL,
            request_fingerprint TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            raw_relative_path TEXT NOT NULL,
            metadata_relative_path TEXT NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT NOT NULL,
            captured_at_utc TEXT NOT NULL,
            response_status INTEGER NOT NULL,
            payload_bytes INTEGER NOT NULL,
            compressed_bytes INTEGER NOT NULL,
            UNIQUE(source, dataset, request_fingerprint, payload_sha256)
        );

        CREATE TABLE IF NOT EXISTS capture_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_artifact_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            partition_key TEXT NOT NULL,
            request_fingerprint TEXT NOT NULL,
            captured_at_utc TEXT NOT NULL,
            response_status INTEGER NOT NULL,
            payload_bytes INTEGER NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT NOT NULL,
            FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id)
        );

        CREATE TABLE IF NOT EXISTS daily_observation_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adjusted_close REAL,
            volume REAL,
            vwap REAL,
            transaction_count INTEGER,
            adjusted INTEGER,
            source_timestamp_ms INTEGER,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            source_row_index INTEGER NOT NULL,
            ingested_at_utc TEXT NOT NULL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(source, symbol, trade_date, raw_artifact_id),
            FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id),
            FOREIGN KEY(capture_event_id) REFERENCES capture_events(id)
        );

        CREATE TABLE IF NOT EXISTS daily_observations (
            source TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adjusted_close REAL,
            volume REAL,
            vwap REAL,
            transaction_count INTEGER,
            adjusted INTEGER,
            source_timestamp_ms INTEGER,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            source_row_index INTEGER NOT NULL,
            ingested_at_utc TEXT NOT NULL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(source, symbol, trade_date),
            FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id),
            FOREIGN KEY(capture_event_id) REFERENCES capture_events(id)
        );

        CREATE TABLE IF NOT EXISTS quality_checks (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            status TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            tolerances_json TEXT NOT NULL,
            computed_at_utc TEXT NOT NULL,
            PRIMARY KEY(symbol, trade_date)
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            job_id TEXT NOT NULL,
            source TEXT NOT NULL,
            item_key TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            scope_json TEXT NOT NULL,
            raw_artifact_id INTEGER,
            observation_count INTEGER,
            last_error TEXT,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY(job_id, source, item_key),
            FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id)
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            contract_json TEXT NOT NULL,
            endpoint_registry_version TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_observations_symbol_date
            ON daily_observations(symbol, trade_date);
        CREATE INDEX IF NOT EXISTS idx_observations_date_source
            ON daily_observations(trade_date, source);
        CREATE INDEX IF NOT EXISTS idx_quality_status_date
            ON quality_checks(status, trade_date);
        CREATE INDEX IF NOT EXISTS idx_raw_sha
            ON raw_artifacts(payload_sha256);
        CREATE INDEX IF NOT EXISTS idx_capture_events_artifact_time
            ON capture_events(raw_artifact_id, captured_at_utc);
        CREATE INDEX IF NOT EXISTS idx_observation_versions_symbol_date
            ON daily_observation_versions(symbol, trade_date, source);
        CREATE INDEX IF NOT EXISTS idx_checkpoint_status
            ON checkpoints(job_id, status);
        """
        with self.connect() as connection:
            connection.executescript(schema)
            connection.execute(
                "INSERT OR REPLACE INTO dataset_metadata(key, value) VALUES (?, ?)",
                ("schema_version", SCHEMA_VERSION),
            )

    def record_raw_artifact(self, values: Mapping[str, Any]) -> int:
        columns = (
            "source",
            "dataset",
            "partition_key",
            "request_fingerprint",
            "payload_sha256",
            "raw_relative_path",
            "metadata_relative_path",
            "request_json",
            "response_json",
            "captured_at_utc",
            "response_status",
            "payload_bytes",
            "compressed_bytes",
        )
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO raw_artifacts ({}) VALUES ({})".format(
                    ",".join(columns), ",".join("?" for _ in columns)
                ),
                tuple(values[column] for column in columns),
            )
            row = connection.execute(
                """
                SELECT id FROM raw_artifacts
                WHERE source=? AND dataset=? AND request_fingerprint=? AND payload_sha256=?
                """,
                (
                    values["source"],
                    values["dataset"],
                    values["request_fingerprint"],
                    values["payload_sha256"],
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError("raw artifact registration failed")
            return int(row["id"])

    def record_capture_event(self, values: Mapping[str, Any]) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO capture_events (
                    raw_artifact_id, source, dataset, partition_key,
                    request_fingerprint, captured_at_utc, response_status,
                    payload_bytes, request_json, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["raw_artifact_id"],
                    values["source"],
                    values["dataset"],
                    values["partition_key"],
                    values["request_fingerprint"],
                    values["captured_at_utc"],
                    values["response_status"],
                    values["payload_bytes"],
                    values["request_json"],
                    values["response_json"],
                ),
            )
            return int(cursor.lastrowid)

    def upsert_observations(self, observations: Sequence[Mapping[str, Any]]) -> int:
        if not observations:
            return 0
        version_sql = """
        INSERT OR IGNORE INTO daily_observation_versions (
            source, symbol, trade_date, open, high, low, close, adjusted_close,
            volume, vwap, transaction_count, adjusted, source_timestamp_ms,
            raw_artifact_id, capture_event_id, source_row_index, ingested_at_utc,
            extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        projection_sql = """
        INSERT INTO daily_observations (
            source, symbol, trade_date, open, high, low, close, adjusted_close,
            volume, vwap, transaction_count, adjusted, source_timestamp_ms,
            raw_artifact_id, capture_event_id, source_row_index, ingested_at_utc,
            extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, symbol, trade_date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            adjusted_close=excluded.adjusted_close,
            volume=excluded.volume,
            vwap=excluded.vwap,
            transaction_count=excluded.transaction_count,
            adjusted=excluded.adjusted,
            source_timestamp_ms=excluded.source_timestamp_ms,
            raw_artifact_id=excluded.raw_artifact_id,
            capture_event_id=excluded.capture_event_id,
            source_row_index=excluded.source_row_index,
            ingested_at_utc=excluded.ingested_at_utc,
            extra_json=excluded.extra_json
        """
        now = utc_now()
        values = []
        for item in observations:
            values.append(
                (
                    item["source"],
                    item["symbol"],
                    item["trade_date"],
                    item.get("open"),
                    item.get("high"),
                    item.get("low"),
                    item.get("close"),
                    item.get("adjusted_close"),
                    item.get("volume"),
                    item.get("vwap"),
                    item.get("transaction_count"),
                    item.get("adjusted"),
                    item.get("source_timestamp_ms"),
                    item["raw_artifact_id"],
                    item["capture_event_id"],
                    item.get("source_row_index", 0),
                    now,
                    canonical_json(item.get("extra", {})),
                )
            )
        with self.connect() as connection:
            connection.executemany(version_sql, values)
            connection.executemany(projection_sql, values)
        return len(values)

    def observation_pairs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> List[sqlite3.Row]:
        clauses = []
        parameters: List[Any] = []
        if start_date:
            clauses.append("trade_date >= ?")
            parameters.append(start_date)
        if end_date:
            clauses.append("trade_date <= ?")
            parameters.append(end_date)
        if symbols:
            clauses.append("symbol IN ({})".format(",".join("?" for _ in symbols)))
            parameters.extend(symbols)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as connection:
            return list(
                connection.execute(
                    "SELECT DISTINCT symbol, trade_date FROM daily_observations{} "
                    "ORDER BY symbol, trade_date".format(where),
                    parameters,
                ).fetchall()
            )

    def observations_for_pair(self, symbol: str, trade_date: str) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT o.*, r.payload_sha256, ce.captured_at_utc,
                           r.raw_relative_path
                    FROM daily_observations o
                    JOIN raw_artifacts r ON r.id=o.raw_artifact_id
                    JOIN capture_events ce ON ce.id=o.capture_event_id
                    WHERE o.symbol=? AND o.trade_date=?
                    ORDER BY o.source
                    """,
                    (symbol, trade_date),
                ).fetchall()
            )

    def observation_rows(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> List[sqlite3.Row]:
        clauses = []
        parameters: List[Any] = []
        if start_date:
            clauses.append("o.trade_date >= ?")
            parameters.append(start_date)
        if end_date:
            clauses.append("o.trade_date <= ?")
            parameters.append(end_date)
        if symbols:
            clauses.append("o.symbol IN ({})".format(",".join("?" for _ in symbols)))
            parameters.extend(symbols)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT o.*, r.payload_sha256, ce.captured_at_utc,
                           r.raw_relative_path
                    FROM daily_observations o
                    JOIN raw_artifacts r ON r.id=o.raw_artifact_id
                    JOIN capture_events ce ON ce.id=o.capture_event_id
                    {} ORDER BY o.symbol, o.trade_date, o.source
                    """.format(where),
                    parameters,
                ).fetchall()
            )

    def upsert_quality(self, values: Mapping[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO quality_checks (
                    symbol, trade_date, status, sources_json, metrics_json,
                    reasons_json, tolerances_json, computed_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, trade_date) DO UPDATE SET
                    status=excluded.status,
                    sources_json=excluded.sources_json,
                    metrics_json=excluded.metrics_json,
                    reasons_json=excluded.reasons_json,
                    tolerances_json=excluded.tolerances_json,
                    computed_at_utc=excluded.computed_at_utc
                """,
                (
                    values["symbol"],
                    values["trade_date"],
                    values["status"],
                    canonical_json(values["sources"]),
                    canonical_json(values["metrics"]),
                    canonical_json(values["reasons"]),
                    canonical_json(values["tolerances"]),
                    utc_now(),
                ),
            )

    def upsert_quality_many(self, records: Sequence[Mapping[str, Any]]) -> int:
        if not records:
            return 0
        sql = """
        INSERT INTO quality_checks (
            symbol, trade_date, status, sources_json, metrics_json,
            reasons_json, tolerances_json, computed_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, trade_date) DO UPDATE SET
            status=excluded.status,
            sources_json=excluded.sources_json,
            metrics_json=excluded.metrics_json,
            reasons_json=excluded.reasons_json,
            tolerances_json=excluded.tolerances_json,
            computed_at_utc=excluded.computed_at_utc
        """
        now = utc_now()
        values = [
            (
                item["symbol"],
                item["trade_date"],
                item["status"],
                canonical_json(item["sources"]),
                canonical_json(item["metrics"]),
                canonical_json(item["reasons"]),
                canonical_json(item["tolerances"]),
                now,
            )
            for item in records
        ]
        with self.connect() as connection:
            connection.executemany(sql, values)
        return len(values)

    def quality_for_pair(self, symbol: str, trade_date: str) -> Optional[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM quality_checks WHERE symbol=? AND trade_date=?",
                (symbol, trade_date),
            ).fetchone()

    def quality_counts(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> dict:
        clauses = []
        parameters: List[Any] = []
        if start_date:
            clauses.append("trade_date >= ?")
            parameters.append(start_date)
        if end_date:
            clauses.append("trade_date <= ?")
            parameters.append(end_date)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM quality_checks{} GROUP BY status".format(
                    where
                ),
                parameters,
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def ensure_checkpoint(
        self,
        job_id: str,
        source: str,
        item_key: str,
        scope: Mapping[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO checkpoints (
                    job_id, source, item_key, status, attempts, scope_json, updated_at_utc
                ) VALUES (?, ?, ?, 'pending', 0, ?, ?)
                """,
                (job_id, source, item_key, canonical_json(scope), utc_now()),
            )

    def register_job(
        self,
        job_id: str,
        job_type: str,
        contract: Mapping[str, Any],
        endpoint_registry_version: str,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, contract_json, endpoint_registry_version,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    contract_json=excluded.contract_json,
                    endpoint_registry_version=excluded.endpoint_registry_version,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (
                    job_id,
                    job_type,
                    canonical_json(contract),
                    endpoint_registry_version,
                    now,
                    now,
                ),
            )

    def checkpoint_status(self, job_id: str, source: str, item_key: str) -> Optional[str]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status FROM checkpoints WHERE job_id=? AND source=? AND item_key=?",
                (job_id, source, item_key),
            ).fetchone()
        return str(row["status"]) if row else None

    def completed_checkpoint_for_item(
        self, source: str, item_key: str, exclude_job_id: Optional[str] = None
    ) -> Optional[sqlite3.Row]:
        clauses = ["source=?", "item_key=?", "status='done'"]
        parameters: List[Any] = [source, item_key]
        if exclude_job_id:
            clauses.append("job_id<>?")
            parameters.append(exclude_job_id)
        with self.connect() as connection:
            return connection.execute(
                "SELECT raw_artifact_id, observation_count FROM checkpoints "
                "WHERE {} ORDER BY updated_at_utc DESC LIMIT 1".format(
                    " AND ".join(clauses)
                ),
                parameters,
            ).fetchone()

    def mark_checkpoint_running(self, job_id: str, source: str, item_key: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE checkpoints
                SET status='running', attempts=attempts+1, last_error=NULL, updated_at_utc=?
                WHERE job_id=? AND source=? AND item_key=?
                """,
                (utc_now(), job_id, source, item_key),
            )

    def mark_checkpoint_done(
        self,
        job_id: str,
        source: str,
        item_key: str,
        raw_artifact_id: int,
        observation_count: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE checkpoints
                SET status='done', raw_artifact_id=?, observation_count=?,
                    last_error=NULL, updated_at_utc=?
                WHERE job_id=? AND source=? AND item_key=?
                """,
                (
                    raw_artifact_id,
                    observation_count,
                    utc_now(),
                    job_id,
                    source,
                    item_key,
                ),
            )

    def mark_checkpoint_failed(
        self, job_id: str, source: str, item_key: str, error: str
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE checkpoints
                SET status='failed', last_error=?, updated_at_utc=?
                WHERE job_id=? AND source=? AND item_key=?
                """,
                (error[:1000], utc_now(), job_id, source, item_key),
            )

    def mark_checkpoint_not_entitled(
        self,
        job_id: str,
        source: str,
        item_key: str,
        raw_artifact_id: int,
        error: str,
    ) -> None:
        """Record a terminal FMP entitlement denial with its raw evidence."""

        with self.connect() as connection:
            artifact = connection.execute(
                "SELECT source, response_status FROM raw_artifacts WHERE id=?",
                (raw_artifact_id,),
            ).fetchone()
            if (
                artifact is None
                or str(artifact["source"]) != "fmp"
                or int(artifact["response_status"]) not in {402, 403}
            ):
                raise ValueError(
                    "not_entitled requires an FMP HTTP 402/403 raw artifact"
                )
            connection.execute(
                """
                UPDATE checkpoints
                SET status='not_entitled', raw_artifact_id=?, observation_count=0,
                    last_error=?, updated_at_utc=?
                WHERE job_id=? AND source=? AND item_key=?
                """,
                (
                    raw_artifact_id,
                    error[:1000],
                    utc_now(),
                    job_id,
                    source,
                    item_key,
                ),
            )

    def checkpoint_summary(self, job_id: str) -> dict:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count, COALESCE(SUM(observation_count), 0) AS rows
                FROM checkpoints WHERE job_id=? GROUP BY status ORDER BY status
                """,
                (job_id,),
            ).fetchall()
        return {
            str(row["status"]): {"items": int(row["count"]), "observations": int(row["rows"])}
            for row in rows
        }

    def raw_artifact_rows(self) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute("SELECT * FROM raw_artifacts ORDER BY id").fetchall())

    def capture_event_rows(self) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute("SELECT * FROM capture_events ORDER BY id").fetchall())

    def raw_artifacts_without_capture_event(self) -> List[int]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT r.id FROM raw_artifacts r
                LEFT JOIN capture_events e ON e.raw_artifact_id=r.id
                WHERE e.id IS NULL ORDER BY r.id
                """
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def counts(self) -> dict:
        with self.connect() as connection:
            result = {}
            for table in (
                "raw_artifacts",
                "capture_events",
                "daily_observation_versions",
                "daily_observations",
                "quality_checks",
                "checkpoints",
                "jobs",
            ):
                row = connection.execute("SELECT COUNT(*) AS count FROM " + table).fetchone()
                result[table] = int(row["count"])
            result["observations_by_source"] = {
                str(row["source"]): int(row["count"])
                for row in connection.execute(
                    "SELECT source, COUNT(*) AS count FROM daily_observations GROUP BY source"
                ).fetchall()
            }
            return result

    def invalid_observation_rows(self) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT source, symbol, trade_date, open, high, low, close, volume
                    FROM daily_observations
                    WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                       OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                       OR high < low OR open > high OR open < low
                       OR close > high OR close < low OR (volume IS NOT NULL AND volume < 0)
                    ORDER BY source, symbol, trade_date
                    """
                ).fetchall()
            )


class RawStore:
    """Content-addressed, append-only gzip payload storage."""

    def __init__(self, data_root: Path, database: Database):
        self.data_root = Path(data_root).expanduser()
        self.database = database

    def store(
        self,
        source: str,
        dataset: str,
        partition_key: str,
        payload: bytes,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> RawArtifact:
        safe_request = json.loads(canonical_json(request))
        request_fingerprint = sha256_bytes(canonical_json(safe_request).encode("utf-8"))
        payload_sha = sha256_bytes(payload)
        relative_directory = Path("raw") / _safe_segment(source) / _safe_segment(dataset) / _safe_segment(
            partition_key
        )
        base_name = "{}-{}".format(request_fingerprint[:16], payload_sha)
        raw_relative = relative_directory / (base_name + ".json.gz")
        metadata_relative = relative_directory / (base_name + ".metadata.json")
        raw_path = self.data_root / raw_relative
        metadata_path = self.data_root / metadata_relative
        compressed = gzip.compress(payload, compresslevel=9, mtime=0)
        captured_at = utc_now()
        metadata = {
            "schema_version": "quant.raw_capture.v1",
            "source": source,
            "dataset": dataset,
            "partition_key": partition_key,
            "request_fingerprint": request_fingerprint,
            "payload_sha256": payload_sha,
            "payload_bytes": len(payload),
            "compressed_bytes": len(compressed),
            "captured_at_utc": captured_at,
            "request": safe_request,
            "response": dict(response),
            "raw_relative_path": raw_relative.as_posix(),
        }
        _write_exclusive(raw_path, compressed)
        _write_exclusive(
            metadata_path,
            (json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
        )

        # If another process won the exclusive write, verify that the artifact
        # at the content-addressed path is exactly the requested payload.
        try:
            stored_payload = gzip.decompress(raw_path.read_bytes())
        except (OSError, EOFError) as error:
            raise RuntimeError("stored raw gzip is unreadable: {}".format(raw_path)) from error
        if sha256_bytes(stored_payload) != payload_sha:
            raise RuntimeError("immutable raw payload checksum mismatch: {}".format(raw_path))

        artifact_id = self.database.record_raw_artifact(
            {
                "source": source,
                "dataset": dataset,
                "partition_key": partition_key,
                "request_fingerprint": request_fingerprint,
                "payload_sha256": payload_sha,
                "raw_relative_path": raw_relative.as_posix(),
                "metadata_relative_path": metadata_relative.as_posix(),
                "request_json": canonical_json(safe_request),
                "response_json": canonical_json(response),
                "captured_at_utc": captured_at,
                "response_status": int(response.get("status_code", 0)),
                "payload_bytes": len(payload),
                "compressed_bytes": len(compressed),
            }
        )
        capture_event_id = self.database.record_capture_event(
            {
                "raw_artifact_id": artifact_id,
                "source": source,
                "dataset": dataset,
                "partition_key": partition_key,
                "request_fingerprint": request_fingerprint,
                "captured_at_utc": captured_at,
                "response_status": int(response.get("status_code", 0)),
                "payload_bytes": len(payload),
                "request_json": canonical_json(safe_request),
                "response_json": canonical_json(response),
            }
        )
        return RawArtifact(
            artifact_id=artifact_id,
            capture_event_id=capture_event_id,
            source=source,
            dataset=dataset,
            payload_sha256=payload_sha,
            raw_path=raw_path,
            metadata_path=metadata_path,
            captured_at_utc=captured_at,
            response_status=int(response.get("status_code", 0)),
            payload_bytes=len(payload),
        )

    def verify_all(self) -> dict:
        errors: List[dict] = []
        checked = 0
        for row in self.database.raw_artifact_rows():
            checked += 1
            raw_path = self.data_root / str(row["raw_relative_path"])
            metadata_path = self.data_root / str(row["metadata_relative_path"])
            if not raw_path.is_file():
                errors.append({"artifact_id": row["id"], "error": "raw_missing"})
                continue
            if not metadata_path.is_file():
                errors.append({"artifact_id": row["id"], "error": "metadata_missing"})
                continue
            try:
                payload = gzip.decompress(raw_path.read_bytes())
            except (OSError, EOFError):
                errors.append({"artifact_id": row["id"], "error": "gzip_invalid"})
                continue
            digest = sha256_bytes(payload)
            if digest != row["payload_sha256"]:
                errors.append(
                    {
                        "artifact_id": row["id"],
                        "error": "payload_sha256_mismatch",
                        "expected": row["payload_sha256"],
                        "actual": digest,
                    }
                )
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                errors.append({"artifact_id": row["id"], "error": "metadata_invalid"})
                continue
            if metadata.get("payload_sha256") != digest:
                errors.append({"artifact_id": row["id"], "error": "metadata_sha256_mismatch"})
            request = metadata.get("request", {})
            sensitive_paths = _unredacted_sensitive_paths(request)
            request_url = str(request.get("url", "")) if isinstance(request, dict) else ""
            for key, value in parse_qsl(urlsplit(request_url).query, keep_blank_values=True):
                if _is_sensitive_name(key) and value != "***REDACTED***":
                    sensitive_paths.append("request.url.query.{}".format(key))
            if sensitive_paths:
                errors.append(
                    {
                        "artifact_id": row["id"],
                        "error": "request_secret_not_redacted",
                        "paths": sorted(set(sensitive_paths)),
                    }
                )
        capture_events = self.database.capture_event_rows()
        for event in capture_events:
            try:
                request = json.loads(str(event["request_json"]))
            except ValueError:
                errors.append({"capture_event_id": event["id"], "error": "event_request_invalid"})
                continue
            sensitive_paths = _unredacted_sensitive_paths(request)
            request_url = str(request.get("url", "")) if isinstance(request, dict) else ""
            for key, value in parse_qsl(urlsplit(request_url).query, keep_blank_values=True):
                if _is_sensitive_name(key) and value != "***REDACTED***":
                    sensitive_paths.append("request.url.query.{}".format(key))
            if sensitive_paths:
                errors.append(
                    {
                        "capture_event_id": event["id"],
                        "error": "event_request_secret_not_redacted",
                        "paths": sorted(set(sensitive_paths)),
                    }
                )
        orphan_artifacts = self.database.raw_artifacts_without_capture_event()
        if orphan_artifacts:
            errors.append(
                {
                    "error": "capture_event_missing",
                    "artifact_ids": orphan_artifacts,
                }
            )
        return {
            "checked": checked,
            "capture_events": len(capture_events),
            "errors": errors,
            "ok": not errors,
        }
