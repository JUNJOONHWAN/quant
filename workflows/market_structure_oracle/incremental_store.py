"""Oracle-owned, resumable current-market delta store.

The immutable long-history database remains untouched.  Market Structure
Oracle is the only writer for FMP stock/ETF daily prices, Massive ETF Flow,
and periodic FMP ETF-constituent refreshes after that fixed boundary.
Consumers read only a sealed status receipt and the base+incremental SQLite
overlay; ETF RADAR is a separate application and is not a source or release
gate for this store.
"""

from __future__ import annotations

import fcntl
import csv
import hashlib
import io
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from quant_dataset.config import load_credentials
from quant_dataset.corporate_actions import capture_corporate_actions
from quant_dataset.pipeline import DatasetPipeline
from quant_dataset.providers import ApiRequestError
from quant_dataset.storage import canonical_json, redacted_request_metadata


PRICE_SOURCE = "fmp"
MASSIVE_SOURCE = "massive"
MIN_RELEASE_MARKET_ROWS = 10_000
INCREMENTAL_SCHEMA = "quant.market_structure_oracle.incremental.v5"
SOURCE_CONTRACT = (
    "oracle_owned_fmp_ultimate_bulk_price_lifecycle_events_"
    "massive_etf_flow_corporate_actions_"
    "no_etf_radar_dependency"
)
STATUS_FILE = "state/oracle_incremental_status.json"
LOCK_FILE = "state/oracle_incremental.lock"
ET = ZoneInfo("America/New_York")
FMP_LEGACY_EOD_BULK_URL = (
    "https://financialmodelingprep.com/api/v4/batch-historical-eod"
)
FMP_STABLE_EOD_BULK_URL = "https://financialmodelingprep.com/stable/eod-bulk"
FMP_STABLE_DELISTED_COMPANIES_URL = (
    "https://financialmodelingprep.com/stable/delisted-companies"
)
FMP_STABLE_MERGERS_ACQUISITIONS_URL = (
    "https://financialmodelingprep.com/stable/mergers-acquisitions-latest"
)
FMP_LIFECYCLE_PAGE_SIZE = 1_000
FMP_LIFECYCLE_MAX_PAGES = 100
DEFAULT_VERIFIED_CORPORATE_ACTIONS = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/"
    "incremental/state/verified_corporate_actions.json"
)
DEFAULT_PROTECTED_SYMBOLS = Path(
    "/home/zooh/Documents/GitHub/STOCK/favorites.json"
)


class IncrementalStoreError(RuntimeError):
    """The current-market delta cannot safely support an Oracle report."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    candidate = date(year, month, 1)
    while candidate.weekday() != weekday:
        candidate += timedelta(days=1)
    return candidate + timedelta(days=7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    candidate = (
        date(year, month + 1, 1) - timedelta(days=1)
        if month < 12
        else date(year, 12, 31)
    )
    while candidate.weekday() != weekday:
        candidate -= timedelta(days=1)
    return candidate


def _easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nyse_holidays(year: int) -> set[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    holidays.add(_observed(date(year + 1, 1, 1)))
    return holidays


def expected_nyse_sessions(start_date: str, end_date: str) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        return []
    holidays: set[date] = set()
    for year in range(start.year - 1, end.year + 2):
        holidays.update(_nyse_holidays(year))
    sessions: list[str] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            sessions.append(current.isoformat())
        current += timedelta(days=1)
    return sessions


def latest_closed_nyse_session(
    now: datetime | None = None, *, publish_grace_hour_et: int = 18
) -> str:
    """Return the latest fully closed and vendor-publishable NYSE session.

    The target is based on market completion, not provider availability.
    Materialization uses the source-preserving FMP legacy bulk first and the
    stable FMP per-symbol endpoint only for symbols not represented by the
    bulk response. Massive is reserved for ETF Flow in this contract.
    """

    if not 16 <= publish_grace_hour_et <= 24:
        raise ValueError("publish_grace_hour_et must be between 16 and 24")
    current = now or datetime.now(ET)
    if current.tzinfo is None:
        current = current.replace(tzinfo=ET)
    current = current.astimezone(ET)
    candidate = current.date()
    if publish_grace_hour_et == 24 or current.hour < publish_grace_hour_et:
        candidate -= timedelta(days=1)
    while not expected_nyse_sessions(candidate.isoformat(), candidate.isoformat()):
        candidate -= timedelta(days=1)
    return candidate.isoformat()


@contextmanager
def _writer_lock(incremental_root: Path) -> Iterator[None]:
    lock_path = incremental_root / LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
                },
                sort_keys=True,
            )
        )
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _initialize_oracle_tables(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS oracle_incremental_runs (
                target_as_of_date TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS oracle_snapshot_seals (
                target_as_of_date TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                source_contract TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                sealed_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS oracle_symbol_change_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_symbol TEXT NOT NULL,
                new_symbol TEXT NOT NULL,
                event_date TEXT NOT NULL,
                available_date TEXT NOT NULL,
                company_name TEXT,
                record_hash TEXT NOT NULL,
                raw_artifact_id INTEGER NOT NULL,
                capture_event_id INTEGER NOT NULL,
                source_row_index INTEGER NOT NULL,
                captured_at_utc TEXT NOT NULL,
                ingested_at_utc TEXT NOT NULL,
                UNIQUE(capture_event_id,source_row_index)
            );
            CREATE TABLE IF NOT EXISTS oracle_symbol_changes (
                old_symbol TEXT NOT NULL,
                new_symbol TEXT NOT NULL,
                event_date TEXT NOT NULL,
                first_available_date TEXT NOT NULL,
                company_name TEXT,
                record_hash TEXT NOT NULL,
                latest_version_id INTEGER NOT NULL,
                raw_artifact_id INTEGER NOT NULL,
                capture_event_id INTEGER NOT NULL,
                captured_at_utc TEXT NOT NULL,
                PRIMARY KEY(old_symbol,new_symbol,event_date),
                FOREIGN KEY(latest_version_id)
                    REFERENCES oracle_symbol_change_versions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_oracle_symbol_changes_visible
                ON oracle_symbol_changes(event_date,first_available_date);
            CREATE TABLE IF NOT EXISTS oracle_lifecycle_event_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_key TEXT NOT NULL,
                symbol TEXT NOT NULL,
                related_symbol TEXT,
                event_date TEXT NOT NULL,
                announcement_date TEXT,
                available_date TEXT NOT NULL,
                company_name TEXT,
                related_company_name TEXT,
                exchange TEXT,
                source_url TEXT,
                record_hash TEXT NOT NULL,
                raw_artifact_id INTEGER NOT NULL,
                capture_event_id INTEGER NOT NULL,
                source_row_index INTEGER NOT NULL,
                captured_at_utc TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                ingested_at_utc TEXT NOT NULL,
                UNIQUE(event_type,capture_event_id,source_row_index)
            );
            CREATE TABLE IF NOT EXISTS oracle_lifecycle_events (
                event_type TEXT NOT NULL,
                event_key TEXT NOT NULL,
                symbol TEXT NOT NULL,
                related_symbol TEXT,
                event_date TEXT NOT NULL,
                announcement_date TEXT,
                first_available_date TEXT NOT NULL,
                company_name TEXT,
                related_company_name TEXT,
                exchange TEXT,
                source_url TEXT,
                record_hash TEXT NOT NULL,
                latest_version_id INTEGER NOT NULL,
                raw_artifact_id INTEGER NOT NULL,
                capture_event_id INTEGER NOT NULL,
                captured_at_utc TEXT NOT NULL,
                PRIMARY KEY(event_type,event_key),
                FOREIGN KEY(latest_version_id)
                    REFERENCES oracle_lifecycle_event_versions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_oracle_lifecycle_events_visible
                ON oracle_lifecycle_events(event_date,first_available_date,event_type);
            """
        )


def _ingest_symbol_changes(
    database_path: Path, path: Path, as_of_date: str
) -> dict[str, Any]:
    if not path.is_file():
        raise IncrementalStoreError(f"FMP symbol-change ledger missing: {path}")
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            old_symbol = str(row["old_symbol"]).strip().upper()
            new_symbol = str(row["new_symbol"]).strip().upper()
            event_date = date.fromisoformat(str(row["event_date"])).isoformat()
            available_date = date.fromisoformat(
                str(row["available_date"])
            ).isoformat()
            raw_artifact_id = int(row["raw_artifact_id"])
            capture_event_id = int(row["capture_event_id"])
            source_row_index = int(row["source_row_index"])
            if not old_symbol or not new_symbol or old_symbol == new_symbol:
                raise ValueError("invalid old/new symbol pair")
            record_hash = hashlib.sha256(
                canonical_json(
                    {
                        "old_symbol": old_symbol,
                        "new_symbol": new_symbol,
                        "event_date": event_date,
                        "available_date": available_date,
                        "company_name": row.get("company_name"),
                    }
                ).encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "old_symbol": old_symbol,
                    "new_symbol": new_symbol,
                    "event_date": event_date,
                    "available_date": available_date,
                    "company_name": row.get("company_name"),
                    "record_hash": record_hash,
                    "raw_artifact_id": raw_artifact_id,
                    "capture_event_id": capture_event_id,
                    "source_row_index": source_row_index,
                    "captured_at_utc": str(row["captured_at_utc"]),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            invalid.append(
                {
                    "line": index + 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if invalid:
        raise IncrementalStoreError(
            f"FMP symbol-change normalization rejected {len(invalid)} rows"
        )
    inserted = 0
    with sqlite3.connect(database_path) as connection:
        for row in records:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO oracle_symbol_change_versions(
                    old_symbol,new_symbol,event_date,available_date,company_name,
                    record_hash,raw_artifact_id,capture_event_id,
                    source_row_index,captured_at_utc,ingested_at_utc
                ) VALUES(?,?,?,?,?,?,?,?,?,?,datetime('now'))
                """,
                (
                    row["old_symbol"],
                    row["new_symbol"],
                    row["event_date"],
                    row["available_date"],
                    row["company_name"],
                    row["record_hash"],
                    row["raw_artifact_id"],
                    row["capture_event_id"],
                    row["source_row_index"],
                    row["captured_at_utc"],
                ),
            )
            inserted += int(cursor.rowcount > 0)
            version = connection.execute(
                """
                SELECT id FROM oracle_symbol_change_versions
                WHERE capture_event_id=? AND source_row_index=?
                """,
                (row["capture_event_id"], row["source_row_index"]),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO oracle_symbol_changes(
                    old_symbol,new_symbol,event_date,first_available_date,
                    company_name,record_hash,latest_version_id,raw_artifact_id,
                    capture_event_id,captured_at_utc
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(old_symbol,new_symbol,event_date) DO UPDATE SET
                    first_available_date=MIN(
                        oracle_symbol_changes.first_available_date,
                        excluded.first_available_date
                    ),
                    company_name=COALESCE(
                        excluded.company_name,
                        oracle_symbol_changes.company_name
                    ),
                    record_hash=excluded.record_hash,
                    latest_version_id=excluded.latest_version_id,
                    raw_artifact_id=excluded.raw_artifact_id,
                    capture_event_id=excluded.capture_event_id,
                    captured_at_utc=excluded.captured_at_utc
                """,
                (
                    row["old_symbol"],
                    row["new_symbol"],
                    row["event_date"],
                    row["available_date"],
                    row["company_name"],
                    row["record_hash"],
                    int(version[0]),
                    row["raw_artifact_id"],
                    row["capture_event_id"],
                    row["captured_at_utc"],
                ),
            )
        visible = [
            list(row)
            for row in connection.execute(
                """
                SELECT old_symbol,new_symbol,event_date,first_available_date,
                       record_hash
                FROM oracle_symbol_changes
                WHERE event_date<=? AND first_available_date<=?
                ORDER BY event_date,old_symbol,new_symbol
                """,
                (as_of_date, as_of_date),
            )
        ]
        projection_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM oracle_symbol_changes"
            ).fetchone()[0]
        )
        version_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM oracle_symbol_change_versions"
            ).fetchone()[0]
        )
    return {
        "schema": "quant.oracle_symbol_lineage.v1",
        "source": "fmp_stable_symbol_change",
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "captured_event_count": len(records),
        "inserted_version_count": inserted,
        "projection_count": projection_count,
        "version_count": version_count,
        "visible_as_of_count": len(visible),
        "visible_projection_sha256": hashlib.sha256(
            canonical_json(visible).encode("utf-8")
        ).hexdigest(),
        "point_in_time_gate": (
            "event_date<=as_of and first_available_date<=as_of; "
            "old and new raw price symbols remain source-preserving"
        ),
    }


def _base_max_date(base_database: Path) -> str:
    with sqlite3.connect(
        f"file:{base_database}?mode=ro", uri=True
    ) as connection:
        value = connection.execute(
            "SELECT MAX(trade_date) FROM daily_observations WHERE source=?",
            (PRICE_SOURCE,),
        ).fetchone()[0]
    if not value:
        raise IncrementalStoreError("base FMP history has no daily observations")
    return str(value)


def _reference_fmp_rows(base_database: Path) -> int:
    base_end = _base_max_date(base_database)
    with sqlite3.connect(
        f"file:{base_database}?mode=ro", uri=True
    ) as connection:
        value = connection.execute(
            """
            SELECT COUNT(DISTINCT symbol) FROM daily_observations
            WHERE source='fmp' AND trade_date=? AND close>0
            """,
            (base_end,),
        ).fetchone()[0]
    return int(value or 0)


def _market_row_count(database_path: Path, trade_date: str) -> int:
    with sqlite3.connect(database_path) as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT symbol) FROM daily_observations
                WHERE source='fmp' AND trade_date=?
                """,
                (trade_date,),
            ).fetchone()[0]
        )


def _market_rows_by_source(
    database_path: Path, trade_date: str
) -> dict[str, int]:
    with sqlite3.connect(database_path) as connection:
        return {
            str(source): int(count)
            for source, count in connection.execute(
                """
                SELECT source,COUNT(DISTINCT symbol)
                FROM daily_observations WHERE trade_date=?
                GROUP BY source ORDER BY source
                """,
                (trade_date,),
            )
        }


def _observed_market_symbols(
    database_path: Path, trade_date: str
) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        return {
            str(row[0]).upper()
            for row in connection.execute(
                """
                SELECT DISTINCT symbol FROM daily_observations
                WHERE trade_date=? AND source='fmp'
                  AND close>0
                """,
                (trade_date,),
            )
        }


def _invalid_market_symbols(
    database_path: Path, trade_date: str
) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        return {
            str(row[0]).upper()
            for row in connection.execute(
                """
                SELECT q.symbol
                FROM quality_checks q
                JOIN daily_observations o
                  ON o.symbol=q.symbol AND o.trade_date=q.trade_date
                WHERE q.trade_date=? AND q.status='invalid'
                  AND o.source='fmp'
                """,
                (trade_date,),
            )
        }


def _load_protected_symbols(path: Path = DEFAULT_PROTECTED_SYMBOLS) -> list[str]:
    """Load symbols whose daily presence is required in addition to row count."""

    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IncrementalStoreError(
            f"protected symbol file is not valid JSON: {path}"
        ) from exc
    if isinstance(payload, dict):
        payload = payload.get("symbols") or payload.get("favorites") or []
    if not isinstance(payload, list):
        raise IncrementalStoreError(
            f"protected symbol file must contain a list: {path}"
        )
    result = []
    for value in payload:
        symbol = str(value or "").strip().upper()
        if symbol and symbol not in result:
            result.append(symbol)
    return result


def _read_symbol_file(path: Path) -> list[str]:
    if not path.exists():
        raise IncrementalStoreError(f"symbol file is missing: {path}")
    return sorted(
        {
            line.strip().upper()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )


def _coverage_checkpoint_rows(
    database_path: Path, job_id: str | None
) -> dict[str, dict[str, Any]]:
    if not job_id:
        return {}
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT item_key,status,observation_count,last_error,raw_artifact_id
            FROM checkpoints WHERE job_id=? AND source='fmp'
            """,
            (job_id,),
        ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for (
        item_key,
        status,
        observation_count,
        last_error,
        raw_artifact_id,
    ) in rows:
        symbol = str(item_key).split(":", 1)[0].upper()
        result[symbol] = {
            "status": str(status),
            "observation_count": int(observation_count or 0),
            "last_error": str(last_error or ""),
            "raw_artifact_id": (
                int(raw_artifact_id)
                if raw_artifact_id is not None
                else None
            ),
        }
    return result


def _reconcile_invalid_exact_rows(
    *,
    pipeline: DatasetPipeline,
    session: str,
    invalid_symbols: set[str],
    checkpoint_rows: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Restore exact FMP versions, then quarantine still-invalid projections.

    Raw artifacts and version rows remain immutable. Exact empty results become
    ``NO_BAR``. Exact rows that remain internally invalid become
    ``QUARANTINED_INVALID_BAR`` and are removed from the model-facing current
    projection.
    """

    candidates = sorted(
        symbol
        for symbol in invalid_symbols
        if (checkpoint_rows.get(symbol) or {}).get("status") == "done"
    )
    if not candidates:
        return {}
    outcomes: dict[str, str] = {}
    projection_columns = (
        "source,symbol,trade_date,open,high,low,close,adjusted_close,"
        "volume,vwap,transaction_count,adjusted,source_timestamp_ms,"
        "raw_artifact_id,capture_event_id,source_row_index,"
        "ingested_at_utc,extra_json"
    )
    with sqlite3.connect(pipeline.database.db_path) as connection:
        connection.row_factory = sqlite3.Row
        for symbol in candidates:
            checkpoint = checkpoint_rows[symbol]
            observation_count = int(
                checkpoint.get("observation_count") or 0
            )
            if observation_count == 0:
                connection.execute(
                    """
                    DELETE FROM daily_observations
                    WHERE source='fmp' AND symbol=? AND trade_date=?
                    """,
                    (symbol, session),
                )
                connection.execute(
                    "DELETE FROM quality_checks WHERE symbol=? AND trade_date=?",
                    (symbol, session),
                )
                outcomes[symbol] = "NO_BAR"
                continue
            raw_artifact_id = checkpoint.get("raw_artifact_id")
            if raw_artifact_id is None:
                continue
            version = connection.execute(
                f"""
                SELECT {projection_columns}
                FROM daily_observation_versions
                WHERE source='fmp' AND symbol=? AND trade_date=?
                  AND raw_artifact_id=?
                ORDER BY id DESC LIMIT 1
                """,
                (symbol, session, raw_artifact_id),
            ).fetchone()
            if version is None:
                continue
            values = tuple(version[key] for key in version.keys())
            connection.execute(
                """
                INSERT INTO daily_observations(
                    source,symbol,trade_date,open,high,low,close,adjusted_close,
                    volume,vwap,transaction_count,adjusted,
                    source_timestamp_ms,raw_artifact_id,capture_event_id,
                    source_row_index,ingested_at_utc,extra_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source,symbol,trade_date) DO UPDATE SET
                    open=excluded.open,high=excluded.high,low=excluded.low,
                    close=excluded.close,adjusted_close=excluded.adjusted_close,
                    volume=excluded.volume,vwap=excluded.vwap,
                    transaction_count=excluded.transaction_count,
                    adjusted=excluded.adjusted,
                    source_timestamp_ms=excluded.source_timestamp_ms,
                    raw_artifact_id=excluded.raw_artifact_id,
                    capture_event_id=excluded.capture_event_id,
                    source_row_index=excluded.source_row_index,
                    ingested_at_utc=excluded.ingested_at_utc,
                    extra_json=excluded.extra_json
                """,
                values,
            )
    positive_candidates = [
        symbol
        for symbol in candidates
        if int(
            (checkpoint_rows.get(symbol) or {}).get("observation_count") or 0
        )
        > 0
    ]
    if positive_candidates:
        pipeline.quality.recompute(
            session, session, positive_candidates
        )
    still_invalid = _invalid_market_symbols(
        pipeline.database.db_path, session
    ) & set(positive_candidates)
    if still_invalid:
        with sqlite3.connect(pipeline.database.db_path) as connection:
            for symbol in sorted(still_invalid):
                connection.execute(
                    """
                    DELETE FROM daily_observations
                    WHERE source='fmp' AND symbol=? AND trade_date=?
                    """,
                    (symbol, session),
                )
                connection.execute(
                    "DELETE FROM quality_checks WHERE symbol=? AND trade_date=?",
                    (symbol, session),
                )
                outcomes[symbol] = "QUARANTINED_INVALID_BAR"
    return outcomes


def _write_symbol_coverage_ledger(
    path: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        "\n".join(canonical_json(row) for row in rows) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": len(rows),
    }


def _repair_required_symbols(
    *,
    pipeline: DatasetPipeline,
    session: str,
    required_symbols: list[str],
    membership_basis_by_symbol: dict[str, str],
    ledger_path: Path,
) -> dict[str, Any]:
    """Classify every master symbol as BAR, NO_BAR, or ERROR.

    A successful exact-date FMP response with zero observations is a valid
    ``NO_BAR`` outcome. It is not silently converted into a missing/error row.
    Only request failures or unresolved checkpoints are ``ERROR``.
    """

    required = sorted(set(required_symbols))
    observed_before = _observed_market_symbols(
        pipeline.database.db_path,
        session,
    )
    invalid_before = _invalid_market_symbols(
        pipeline.database.db_path,
        session,
    )
    missing_before = [
        symbol for symbol in required if symbol not in observed_before
    ]
    repair_candidates = sorted(set(missing_before) | (invalid_before & set(required)))
    capture = None
    if repair_candidates:
        capture = pipeline.capture_daily(
            session,
            repair_candidates,
            source="fmp",
            continue_on_error=True,
        )
    checkpoint_rows = _coverage_checkpoint_rows(
        pipeline.database.db_path,
        str((capture or {}).get("job_id") or "") or None,
    )
    reconciled_invalid = _reconcile_invalid_exact_rows(
        pipeline=pipeline,
        session=session,
        invalid_symbols=invalid_before & set(required),
        checkpoint_rows=checkpoint_rows,
    )
    observed_after = _observed_market_symbols(
        pipeline.database.db_path,
        session,
    )
    invalid_after = _invalid_market_symbols(
        pipeline.database.db_path,
        session,
    )
    ledger_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    counts = {
        "BAR": 0,
        "NO_BAR": 0,
        "QUARANTINED_INVALID_BAR": 0,
        "ERROR": 0,
    }
    for symbol in required:
        checkpoint = checkpoint_rows.get(symbol) or {}
        if symbol in reconciled_invalid:
            outcome = reconciled_invalid[symbol]
            error = None
        elif symbol in observed_after and symbol not in invalid_after:
            outcome = "BAR"
            error = None
        elif symbol in invalid_after:
            outcome = "ERROR"
            error = "invalid FMP OHLC row after stable per-symbol repair"
            errors.append(symbol)
        elif (
            checkpoint.get("status") == "done"
            and int(checkpoint.get("observation_count") or 0) == 0
        ):
            outcome = "NO_BAR"
            error = None
        else:
            outcome = "ERROR"
            error = str(
                checkpoint.get("last_error")
                or "missing exact-date FMP completion checkpoint"
            )
            errors.append(symbol)
        counts[outcome] += 1
        ledger_rows.append(
            {
                "schema": "quant.oracle_daily_symbol_coverage.v1",
                "trade_date": session,
                "symbol": symbol,
                "membership_basis": membership_basis_by_symbol.get(
                    symbol, "unspecified"
                ),
                "outcome": outcome,
                "source": "fmp",
                "fallback_endpoint": (
                    "stable_historical_price_eod_full"
                    if symbol in repair_candidates
                    else None
                ),
                "error": error,
            }
        )
    ledger = _write_symbol_coverage_ledger(ledger_path, ledger_rows)
    complete = counts["ERROR"] == 0 and (
        counts["BAR"]
        + counts["NO_BAR"]
        + counts["QUARANTINED_INVALID_BAR"]
        == len(required)
    )
    return {
        "status": "complete" if complete else "incomplete",
        "required_count": len(required),
        "bar_count": counts["BAR"],
        "no_bar_count": counts["NO_BAR"],
        "quarantined_invalid_bar_count": counts[
            "QUARANTINED_INVALID_BAR"
        ],
        "error_count": counts["ERROR"],
        "observed_count": counts["BAR"],
        "missing_before_count": len(missing_before),
        "missing_before_sample": missing_before[:50],
        "invalid_before_count": len(invalid_before & set(required)),
        "invalid_before_sample": sorted(invalid_before & set(required))[:50],
        "invalid_no_bar_count": sum(
            outcome == "NO_BAR"
            for outcome in reconciled_invalid.values()
        ),
        "quarantined_invalid_bar_sample": [
            symbol
            for symbol, outcome in sorted(reconciled_invalid.items())
            if outcome == "QUARANTINED_INVALID_BAR"
        ][:50],
        "repaired_count": len(repair_candidates),
        "missing_after": errors,
        "error_symbols_sample": errors[:50],
        "ledger": ledger,
        "capture": capture,
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _daily_universe(
    pipeline: DatasetPipeline, target: str
) -> tuple[list[str], dict[str, Any]]:
    snapshot = pipeline.capture_fmp_active_universe(target)
    if snapshot.get("warnings"):
        raise IncrementalStoreError(
            "FMP active-master completeness warnings are release-blocking: "
            + ",".join(map(str, snapshot["warnings"]))
        )
    symbols_path = Path(str(snapshot["symbols_path"]))
    symbols = _read_symbol_file(symbols_path)
    if len(symbols) < MIN_RELEASE_MARKET_ROWS:
        raise IncrementalStoreError(
            f"FMP active US stock/ETF master is unexpectedly small: {len(symbols)}"
        )
    return symbols, snapshot


def _capture_fmp_stable_eod_bulk(
    *,
    pipeline: DatasetPipeline,
    target: str,
    allowed_symbols: set[str],
) -> dict[str, Any]:
    """Capture the Ultimate stable EOD bulk CSV with immutable raw evidence."""

    key = pipeline.credentials.fmp_api_key
    if not key:
        raise IncrementalStoreError("FMP_API_KEY is not configured")
    safe_request = redacted_request_metadata(
        "GET",
        FMP_STABLE_EOD_BULK_URL,
        {"date": target},
        {
            "endpoint_contract": "fmp_stable_eod_bulk",
            "date": target,
            "plan_contract": "ultimate_bulk_primary",
        },
    )
    response = None
    artifact = None
    for attempt in range(3):
        limiter = pipeline.http.rate_limiters.get("fmp")
        if limiter is not None:
            limiter.acquire()
        response = pipeline.http.session.get(
            FMP_STABLE_EOD_BULK_URL,
            params={"date": target},
            headers={"apikey": key},
            timeout=120,
        )
        payload = bytes(response.content)
        artifact = pipeline.raw_store.store(
            source="fmp",
            dataset="stable_eod_bulk",
            partition_key=target,
            payload=payload,
            request=safe_request,
            response={
                "status_code": int(response.status_code),
                "headers": {
                    "content-type": str(
                        response.headers.get("content-type") or ""
                    ),
                    "retry-after": str(
                        response.headers.get("retry-after") or ""
                    ),
                },
                "attempt": attempt + 1,
            },
        )
        if response.status_code != 429:
            break
        if attempt < 2:
            time.sleep(10)
    assert response is not None and artifact is not None
    if response.status_code < 200 or response.status_code >= 300:
        raise ApiRequestError(
            "fmp stable_eod_bulk returned HTTP {} "
            "(raw artifact id={})".format(
                response.status_code, artifact.artifact_id
            ),
            status_code=int(response.status_code),
            raw_artifact_id=artifact.artifact_id,
        )
    try:
        rows = list(
            csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")))
        )
    except (UnicodeDecodeError, csv.Error) as exc:
        raise IncrementalStoreError(
            "FMP stable EOD bulk CSV parsing failed "
            f"(raw artifact id={artifact.artifact_id})"
        ) from exc
    observations = []
    for index, row in enumerate(rows):
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol not in allowed_symbols or str(row.get("date") or "") != target:
            continue
        close = _number(row.get("close"))
        if close is None or close <= 0:
            continue
        observations.append(
            {
                "source": "fmp",
                "symbol": symbol,
                "trade_date": target,
                "open": _number(row.get("open")),
                "high": _number(row.get("high")),
                "low": _number(row.get("low")),
                "close": close,
                "adjusted_close": _number(row.get("adjClose")),
                "volume": _number(row.get("volume")),
                "vwap": None,
                "transaction_count": None,
                "adjusted": 1,
                "source_timestamp_ms": None,
                "raw_artifact_id": artifact.artifact_id,
                "capture_event_id": artifact.capture_event_id,
                "source_row_index": index,
                "extra": {
                    "endpoint_contract": (
                        "fmp_stable_eod_bulk"
                    ),
                    "plan_contract": "ultimate_bulk_primary",
                },
            }
        )
    inserted = pipeline.database.upsert_observations(observations)
    pipeline.quality.recompute(target, target)
    pipeline.write_manifest()
    return {
        "ok": True,
        "mode": "fmp_stable_eod_bulk",
        "raw_row_count": len(rows),
        "matched_us_universe_count": len(observations),
        "inserted_observation_count": inserted,
        "raw_artifact_id": artifact.artifact_id,
        "payload_sha256": artifact.payload_sha256,
        "plan_contract": "ultimate_bulk_primary",
    }


def _event_date(value: Any, field: str) -> str:
    try:
        return date.fromisoformat(str(value or "")[:10]).isoformat()
    except ValueError as exc:
        raise IncrementalStoreError(f"FMP lifecycle {field} is not YYYY-MM-DD") from exc


def _capture_fmp_lifecycle_events(
    *, pipeline: DatasetPipeline, target: str
) -> dict[str, Any]:
    """Register FMP Ultimate delistings and M&A as point-in-time Oracle events.

    These are separate from split adjustments: the raw source row, first capture
    date, immutable version and current projection are all retained so consumers
    can reconstruct membership without silently dropping disappeared tickers.
    """

    key = pipeline.credentials.fmp_api_key
    if not key:
        raise IncrementalStoreError("FMP_API_KEY is not configured")
    definitions = (
        ("delisted", "stable_delisted_companies", FMP_STABLE_DELISTED_COMPANIES_URL),
        ("merger_acquisition", "stable_mergers_acquisitions_latest", FMP_STABLE_MERGERS_ACQUISITIONS_URL),
    )
    captured: dict[str, Any] = {}
    normalized_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for event_type, dataset, url in definitions:
        raw_rows = 0
        artifacts: list[int] = []
        for page in range(FMP_LIFECYCLE_MAX_PAGES):
            result = pipeline.http.get_json(
                source="fmp",
                dataset=dataset,
                partition_key=target,
                url=url,
                params={"page": page, "limit": FMP_LIFECYCLE_PAGE_SIZE},
                headers={"apikey": key},
                logical_request={
                    "endpoint_contract": f"fmp_{dataset}",
                    "event_type": event_type,
                    "page": page,
                    "limit": FMP_LIFECYCLE_PAGE_SIZE,
                    "as_of_date": target,
                    "plan_contract": "ultimate_corporate_event_registry",
                },
            )
            rows = result.document
            if not isinstance(rows, list):
                raise IncrementalStoreError(
                    f"FMP {dataset} payload is not a list "
                    f"(raw artifact id={result.artifact.artifact_id})"
                )
            artifacts.append(int(result.artifact.artifact_id))
            raw_rows += len(rows)
            available_date = _event_date(
                result.artifact.captured_at_utc, "captured_at_utc"
            )
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    rejected.append(
                        {"event_type": event_type, "page": page, "index": index,
                         "reason": "row_not_object"}
                    )
                    continue
                try:
                    if event_type == "delisted":
                        symbol = str(row.get("symbol") or "").strip().upper()
                        related_symbol = None
                        event_date = _event_date(row.get("delistedDate"), "delistedDate")
                        announcement_date = None
                        company_name = str(row.get("companyName") or "").strip() or None
                        related_company_name = None
                        exchange = str(row.get("exchange") or "").strip() or None
                        source_url = FMP_STABLE_DELISTED_COMPANIES_URL
                    else:
                        symbol = str(row.get("symbol") or "").strip().upper()
                        related_symbol = (
                            str(row.get("targetedSymbol") or "").strip().upper() or None
                        )
                        event_date = _event_date(row.get("transactionDate"), "transactionDate")
                        announcement_date = (
                            _event_date(row.get("acceptedDate"), "acceptedDate")
                            if row.get("acceptedDate") else None
                        )
                        company_name = str(row.get("companyName") or "").strip() or None
                        related_company_name = (
                            str(row.get("targetedCompanyName") or "").strip() or None
                        )
                        exchange = None
                        source_url = str(row.get("link") or "").strip() or FMP_STABLE_MERGERS_ACQUISITIONS_URL
                    if not symbol:
                        raise ValueError("missing symbol")
                    if event_date > target:
                        continue
                    identity = {
                        "event_type": event_type,
                        "symbol": symbol,
                        "related_symbol": related_symbol,
                        "event_date": event_date,
                        "company_name": company_name,
                        "related_company_name": related_company_name,
                        "source_url": source_url,
                    }
                    event_key = hashlib.sha256(
                        canonical_json(identity).encode("utf-8")
                    ).hexdigest()
                    normalized_records.append(
                        {
                            **identity,
                            "event_key": event_key,
                            "announcement_date": announcement_date,
                            "available_date": available_date,
                            "exchange": exchange,
                            "record_hash": hashlib.sha256(
                                canonical_json({**identity, "raw": row}).encode("utf-8")
                            ).hexdigest(),
                            "raw_artifact_id": int(result.artifact.artifact_id),
                            "capture_event_id": int(result.artifact.capture_event_id),
                            "source_row_index": index,
                            "captured_at_utc": str(result.artifact.captured_at_utc),
                            "payload_json": canonical_json(row),
                        }
                    )
                except (IncrementalStoreError, TypeError, ValueError) as exc:
                    rejected.append(
                        {"event_type": event_type, "page": page, "index": index,
                         "reason": f"{type(exc).__name__}: {exc}"}
                    )
            if len(rows) < FMP_LIFECYCLE_PAGE_SIZE:
                break
        else:
            raise IncrementalStoreError(
                f"FMP {dataset} pagination exceeded {FMP_LIFECYCLE_MAX_PAGES} pages"
            )
        captured[event_type] = {
            "dataset": dataset,
            "raw_row_count": raw_rows,
            "raw_artifact_ids": artifacts,
        }
    if rejected:
        raise IncrementalStoreError(
            f"FMP lifecycle normalization rejected {len(rejected)} rows: {rejected[:3]}"
        )
    inserted = 0
    with sqlite3.connect(pipeline.database.db_path) as connection:
        for row in normalized_records:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO oracle_lifecycle_event_versions(
                    event_type,event_key,symbol,related_symbol,event_date,
                    announcement_date,available_date,company_name,related_company_name,
                    exchange,source_url,record_hash,raw_artifact_id,capture_event_id,
                    source_row_index,captured_at_utc,payload_json,ingested_at_utc
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
                """,
                tuple(row[key] for key in (
                    "event_type", "event_key", "symbol", "related_symbol", "event_date",
                    "announcement_date", "available_date", "company_name", "related_company_name",
                    "exchange", "source_url", "record_hash", "raw_artifact_id", "capture_event_id",
                    "source_row_index", "captured_at_utc", "payload_json",
                )),
            )
            inserted += int(cursor.rowcount > 0)
            version = connection.execute(
                """
                SELECT id FROM oracle_lifecycle_event_versions
                WHERE event_type=? AND capture_event_id=? AND source_row_index=?
                """,
                (row["event_type"], row["capture_event_id"], row["source_row_index"]),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO oracle_lifecycle_events(
                    event_type,event_key,symbol,related_symbol,event_date,
                    announcement_date,first_available_date,company_name,
                    related_company_name,exchange,source_url,record_hash,
                    latest_version_id,raw_artifact_id,capture_event_id,captured_at_utc
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(event_type,event_key) DO UPDATE SET
                    announcement_date=COALESCE(excluded.announcement_date,
                        oracle_lifecycle_events.announcement_date),
                    first_available_date=MIN(oracle_lifecycle_events.first_available_date,
                        excluded.first_available_date),
                    record_hash=excluded.record_hash,
                    latest_version_id=excluded.latest_version_id,
                    raw_artifact_id=excluded.raw_artifact_id,
                    capture_event_id=excluded.capture_event_id,
                    captured_at_utc=excluded.captured_at_utc
                """,
                tuple(row[key] for key in (
                    "event_type", "event_key", "symbol", "related_symbol", "event_date",
                    "announcement_date", "available_date", "company_name", "related_company_name",
                    "exchange", "source_url", "record_hash",
                )) + (int(version[0]),) + tuple(row[key] for key in (
                    "raw_artifact_id", "capture_event_id", "captured_at_utc",
                )),
            )
        visible = [
            list(row) for row in connection.execute(
                """
                SELECT event_type,event_key,symbol,related_symbol,event_date,
                       announcement_date,first_available_date,record_hash
                FROM oracle_lifecycle_events
                WHERE event_date<=? AND first_available_date<=?
                ORDER BY event_date,event_type,symbol,event_key
                """, (target, target)
            )
        ]
        projection_count = int(connection.execute(
            "SELECT COUNT(*) FROM oracle_lifecycle_events"
        ).fetchone()[0])
        version_count = int(connection.execute(
            "SELECT COUNT(*) FROM oracle_lifecycle_event_versions"
        ).fetchone()[0])
    return {
        "schema": "quant.oracle_lifecycle_event_registry.v1",
        "provider": "fmp_ultimate",
        "captured": captured,
        "normalized_record_count": len(normalized_records),
        "inserted_version_count": inserted,
        "projection_count": projection_count,
        "version_count": version_count,
        "visible_as_of_count": len(visible),
        "visible_projection_sha256": hashlib.sha256(
            canonical_json(visible).encode("utf-8")
        ).hexdigest(),
        "point_in_time_gate": (
            "event_date<=as_of and first_available_date<=as_of; "
            "delisted and merger/acquisition rows retain raw source versions"
        ),
    }


def _capture_current_session(
    *,
    pipeline: DatasetPipeline,
    session: str,
    allowed_reference_symbols: set[str],
) -> dict[str, Any]:
    """Capture one full Ultimate FMP daily file with immutable raw evidence."""

    attempts: list[dict[str, Any]] = []
    try:
        stable = _capture_fmp_stable_eod_bulk(
            pipeline=pipeline,
            target=session,
            allowed_symbols=allowed_reference_symbols,
        )
        attempts.append({"source": "fmp_stable_eod_bulk", "result": stable})
    except (ApiRequestError, IncrementalStoreError) as exc:
        attempts.append(
            {
                "source": "fmp_stable_eod_bulk",
                "status": "failed",
                "error": str(exc),
                "http_status": getattr(exc, "status_code", None),
                "raw_artifact_id": getattr(exc, "raw_artifact_id", None),
            }
        )
        raise IncrementalStoreError(
            f"FMP Ultimate stable EOD bulk capture failed for {session}; "
            "historical daily membership cannot be reconstructed safely"
        ) from exc
    return {
        "mode": "fmp_stable_eod_bulk",
        "attempts": attempts,
        "observed_fmp_bar_count": _market_row_count(
            pipeline.database.db_path, session
        ),
    }


def _archived_active_master(
    incremental_root: Path, session: str
) -> tuple[list[str], dict[str, Any]] | None:
    base = incremental_root / "state" / "active_universe"
    stem = "fmp_active_us_{}".format(session.replace("-", ""))
    manifest_path = base / (stem + ".manifest.json")
    symbols_path = base / (stem + ".symbols.txt")
    if not manifest_path.is_file() or not symbols_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IncrementalStoreError(
            f"invalid archived active-master manifest: {manifest_path}"
        ) from exc
    symbols = _read_symbol_file(symbols_path)
    expected_sha = str(manifest.get("symbols_sha256") or "")
    observed_sha = hashlib.sha256(symbols_path.read_bytes()).hexdigest()
    if expected_sha != observed_sha:
        raise IncrementalStoreError(
            f"archived active-master hash mismatch: {symbols_path}"
        )
    return symbols, {
        "manifest_path": str(manifest_path),
        "symbols_path": str(symbols_path),
        "symbols_sha256": observed_sha,
        "captured_at_utc": manifest.get("captured_at_utc"),
        "active_symbol_count": len(symbols),
    }


def _flow_summary(database_path: Path) -> dict[str, Any]:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT MAX(effective_date),MAX(processed_date),
                   COUNT(*),COUNT(DISTINCT ticker)
            FROM etf_flow_observations
            """
        ).fetchone()
    return {
        "latest_effective_date": row[0],
        "latest_processed_date": row[1],
        "record_count": int(row[2] or 0),
        "ticker_count": int(row[3] or 0),
    }


def _profile_etfs(base_database: Path) -> set[str]:
    result: set[str] = set()
    with sqlite3.connect(
        f"file:{base_database}?mode=ro", uri=True
    ) as connection:
        rows = connection.execute(
            """
            SELECT symbol,row_json FROM fmp_training_facts
            WHERE endpoint_id='company_information_company_profile_data'
            """
        ).fetchall()
    for symbol, raw in rows:
        try:
            document = json.loads(str(raw))
        except (TypeError, json.JSONDecodeError):
            continue
        if document.get("isEtf") is True and symbol:
            result.add(str(symbol).upper())
    return result


def constituent_refresh_candidates(
    *,
    base_database: Path,
    incremental_database: Path,
    target_as_of_date: str,
    stale_days: int,
    limit: int,
) -> dict[str, Any]:
    """Prioritize newly observed ETFs, then stale constituent snapshots."""

    if stale_days < 1 or limit < 0:
        raise ValueError("stale_days must be positive and limit non-negative")
    etfs = _profile_etfs(base_database)
    latest: dict[str, str] = {}
    for path in (base_database, incremental_database):
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            etfs.update(
                str(row[0]).upper()
                for row in connection.execute(
                    "SELECT DISTINCT ticker FROM etf_flow_observations"
                )
                if row[0]
            )
            for ticker, available in connection.execute(
                """
                SELECT etf_ticker,MAX(available_date)
                FROM etf_constituent_snapshots GROUP BY etf_ticker
                """
            ):
                key = str(ticker).upper()
                latest[key] = max(latest.get(key, ""), str(available))
    cutoff = (
        date.fromisoformat(target_as_of_date) - timedelta(days=stale_days)
    ).isoformat()
    missing = sorted(ticker for ticker in etfs if ticker not in latest)
    stale = sorted(
        ticker
        for ticker in etfs
        if ticker in latest and latest[ticker] < cutoff
    )
    selected = (missing + stale)[:limit]
    return {
        "discovered_etf_count": len(etfs),
        "missing_snapshot_count": len(missing),
        "stale_snapshot_count": len(stale),
        "selected": selected,
        "selected_count": len(selected),
        "pending_count": max(len(missing) + len(stale) - len(selected), 0),
        "stale_cutoff_available_date": cutoff,
    }


def _refresh_constituents(
    *,
    pipeline: DatasetPipeline,
    base_database: Path,
    database_path: Path,
    target: str,
    stale_days: int,
    max_etfs: int,
) -> dict[str, Any]:
    plan = constituent_refresh_candidates(
        base_database=base_database,
        incremental_database=database_path,
        target_as_of_date=target,
        stale_days=stale_days,
        limit=max_etfs,
    )
    if not plan["selected"]:
        return {**plan, "capture": None, "status": "current"}
    capture = pipeline.backfill_fmp_etf_constituents(
        (date.fromisoformat(target) - timedelta(days=400)).isoformat(),
        target,
        plan["selected"],
        universe_contract={
            "owner": "market_structure_oracle",
            "policy": "new_first_then_stale_bounded_daily_refresh",
            "target_as_of_date": target,
        },
        continue_on_error=True,
    )
    return {
        **plan,
        "capture": capture,
        "status": "complete" if capture.get("ok") else "partial",
    }


def _status_target(status_path: Path) -> str | None:
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if status.get("status") != "COMPLETE" or status.get("missing_sessions"):
        return None
    if status.get("schema") != INCREMENTAL_SCHEMA:
        return None
    if not ((status.get("snapshot_seal") or {}).get("receipt_sha256")):
        return None
    return str(status.get("target_as_of_date") or "") or None


def _materialize(
    *,
    base_database: Path,
    incremental_root: Path,
    target_as_of_date: str,
    constituent_stale_days: int,
    constituent_refresh_max_etfs: int,
    protected_symbols_path: Path = DEFAULT_PROTECTED_SYMBOLS,
) -> dict[str, Any]:
    base_end = _base_max_date(base_database)
    target = date.fromisoformat(target_as_of_date).isoformat()
    if target < base_end:
        raise IncrementalStoreError(
            f"target precedes immutable base cutoff: target={target} base={base_end}"
        )
    sessions = expected_nyse_sessions(
        (date.fromisoformat(base_end) + timedelta(days=1)).isoformat(), target
    )
    credentials = load_credentials()
    pipeline = DatasetPipeline(
        data_root=incremental_root, credentials=credentials
    )
    database_path = pipeline.database.db_path
    _initialize_oracle_tables(database_path)
    active_symbols, active_snapshot = _daily_universe(pipeline, target)
    symbol_lineage = _ingest_symbol_changes(
        database_path,
        Path(str(active_snapshot["symbol_changes_path"])),
        target,
    )
    reference_symbols = _read_symbol_file(
        Path(str(active_snapshot["reference_symbols_path"]))
    )
    if len(reference_symbols) < len(active_symbols):
        raise IncrementalStoreError(
            "FMP reference catalog is smaller than the active master"
        )
    session_rows: dict[str, int] = {}
    session_minimum_rows: dict[str, int] = {}
    session_rows_by_source: dict[str, dict[str, int]] = {}
    session_capture: dict[str, dict[str, Any]] = {}
    session_master: dict[str, dict[str, Any]] = {}
    repaired_sessions: list[str] = []
    protected_symbols = _load_protected_symbols(protected_symbols_path)
    coverage_by_session: dict[str, dict[str, Any]] = {}
    status_path = incremental_root / STATUS_FILE
    _atomic_json(
        status_path,
        {
            "status": "REPAIRING_COVERAGE",
            "schema": INCREMENTAL_SCHEMA,
            "source_contract": SOURCE_CONTRACT,
            "base_history_end": base_end,
            "target_as_of_date": target,
            "expected_sessions": sessions,
            "active_master": active_snapshot,
            "coverage_progress": {
                "completed_sessions": [],
                "remaining_sessions": sessions,
            },
            "database": str(database_path),
        },
    )
    for session in sessions:
        capture_result = _capture_current_session(
            pipeline=pipeline,
            session=session,
            allowed_reference_symbols=set(reference_symbols),
        )
        session_capture[session] = capture_result
        observed_before_gate = _observed_market_symbols(database_path, session)
        archived = _archived_active_master(incremental_root, session)
        timely_archived = False
        if archived is not None:
            captured_at = str(archived[1].get("captured_at_utc") or "")
            try:
                captured_date = date.fromisoformat(captured_at[:10])
                timely_archived = captured_date <= (
                    date.fromisoformat(session) + timedelta(days=4)
                )
            except ValueError:
                timely_archived = False
        if session == target:
            active_symbol_set = set(active_symbols)
            legacy_bar_additions = sorted(
                observed_before_gate - active_symbol_set
            )
            master_symbols = sorted(
                active_symbol_set | observed_before_gate
            )
            master_basis = (
                "captured_current_fmp_active_master_union_same_day_legacy_bars"
            )
            master_evidence = {
                "manifest_path": active_snapshot["manifest_path"],
                "symbols_path": active_snapshot["symbols_path"],
                "symbols_sha256": active_snapshot["symbols_sha256"],
                "same_day_legacy_bar_addition_count": len(
                    legacy_bar_additions
                ),
                "same_day_legacy_bar_additions": legacy_bar_additions,
            }
            membership_basis = {
                symbol: (
                    "captured_current_fmp_active_master"
                    if symbol in active_symbol_set
                    else "same_day_fmp_legacy_bar_not_in_active_directory"
                )
                for symbol in master_symbols
            }
        elif archived is not None and timely_archived:
            master_symbols, master_evidence = archived
            master_basis = "timely_archived_fmp_active_master"
            membership_basis = {
                symbol: master_basis for symbol in master_symbols
            }
        else:
            master_symbols = sorted(observed_before_gate)
            master_basis = "reconstructed_same_day_fmp_legacy_bar_presence"
            master_payload = ("\n".join(master_symbols) + "\n").encode("utf-8")
            master_evidence = {
                "symbols_path": None,
                "symbols_sha256": hashlib.sha256(master_payload).hexdigest(),
                "historical_active_snapshot_available": False,
                "lookahead_control": (
                    "only same-day FMP legacy bars are analytical members; "
                    "the current active list is not projected backward"
                ),
            }
            membership_basis = {
                symbol: master_basis for symbol in master_symbols
            }
        if not master_symbols:
            raise IncrementalStoreError(
                f"FMP daily master is empty for {session}"
            )
        for symbol in protected_symbols:
            membership_basis.setdefault(symbol, "protected_symbol_override")
        required_for_session = sorted(
            set(master_symbols) | set(protected_symbols)
        )
        coverage_gate = _repair_required_symbols(
            pipeline=pipeline,
            session=session,
            required_symbols=required_for_session,
            membership_basis_by_symbol=membership_basis,
            ledger_path=(
                incremental_root
                / "state"
                / "daily_coverage"
                / f"fmp_symbol_coverage_{session}.jsonl"
            ),
        )
        coverage_gate["mode"] = "fmp_daily_master_bar_no_bar_error_ledger"
        coverage_gate["master_basis"] = master_basis
        coverage_gate["master_symbol_count"] = len(master_symbols)
        coverage_gate["master_evidence"] = master_evidence
        session_master[session] = {
            "basis": master_basis,
            "symbol_count": len(master_symbols),
            **master_evidence,
        }
        coverage_by_session[session] = coverage_gate
        if coverage_gate["repaired_count"] and session not in repaired_sessions:
            repaired_sessions.append(session)
        if coverage_gate["status"] != "complete":
            raise IncrementalStoreError(
                f"FMP symbol coverage gate failed for {session}: "
                f"errors={coverage_gate['error_count']} "
                f"ledger={coverage_gate['ledger']['path']}"
            )
        count = _market_row_count(database_path, session)
        required_rows = min(MIN_RELEASE_MARKET_ROWS, len(master_symbols))
        if count < required_rows:
            raise IncrementalStoreError(
                f"Oracle FMP full-market row gate failed for {session}: "
                f"{count} < {required_rows}"
            )
        source_rows = _market_rows_by_source(database_path, session)
        if source_rows.get(MASSIVE_SOURCE, 0):
            raise IncrementalStoreError(
                f"non-contract Massive stock daily rows remain for {session}: "
                f"{source_rows[MASSIVE_SOURCE]}"
            )
        session_rows[session] = count
        session_minimum_rows[session] = required_rows
        session_rows_by_source[session] = source_rows
        completed_sessions = list(coverage_by_session)
        _atomic_json(
            status_path,
            {
                "status": "REPAIRING_COVERAGE",
                "schema": INCREMENTAL_SCHEMA,
                "source_contract": SOURCE_CONTRACT,
                "base_history_end": base_end,
                "target_as_of_date": target,
                "expected_sessions": sessions,
                "active_master": active_snapshot,
                "coverage_progress": {
                    "current_session": session,
                    "completed_sessions": completed_sessions,
                    "remaining_sessions": [
                        item for item in sessions if item not in completed_sessions
                    ],
                    "session_status": {
                        item: coverage_by_session[item]["status"]
                        for item in completed_sessions
                    },
                },
                "database": str(database_path),
            },
        )
    incomplete_sessions = {
        session: gate["missing_after"]
        for session, gate in coverage_by_session.items()
        if gate["status"] != "complete"
    }
    if incomplete_sessions:
        coverage_ledger = {
            "schema": "quant.oracle_symbol_coverage_ledger.v2",
            "status": "incomplete",
            "base_history_end": base_end,
            "target_as_of_date": target,
            "active_master": active_snapshot,
            "protected_symbols": protected_symbols,
            "sessions": coverage_by_session,
            "unresolved_session_count": len(incomplete_sessions),
            "unresolved_symbol_pairs": sum(
                len(symbols) for symbols in incomplete_sessions.values()
            ),
        }
        ledger_path = (
            incremental_root
            / "state"
            / f"symbol_coverage_ledger_{target}.json"
        )
        _atomic_json(ledger_path, coverage_ledger)
        _atomic_json(
            status_path,
            {
                "status": "INCOMPLETE_COVERAGE",
                "schema": INCREMENTAL_SCHEMA,
                "source_contract": SOURCE_CONTRACT,
                "base_history_end": base_end,
                "target_as_of_date": target,
                "expected_sessions": sessions,
                "database": str(database_path),
                "symbol_coverage_ledger": str(ledger_path),
                "unresolved_session_count": len(incomplete_sessions),
                "unresolved_symbol_pairs": coverage_ledger[
                    "unresolved_symbol_pairs"
                ],
            },
        )
        raise IncrementalStoreError(
            "full incremental symbol coverage remains incomplete: "
            f"sessions={len(incomplete_sessions)} "
            f"symbol_pairs={coverage_ledger['unresolved_symbol_pairs']} "
            f"ledger={ledger_path}"
        )
    corporate_actions = capture_corporate_actions(
        pipeline=pipeline,
        start_date=(
            date.fromisoformat(base_end) + timedelta(days=1)
        ).isoformat(),
        end_date=target,
        official_ledger_path=DEFAULT_VERIFIED_CORPORATE_ACTIONS,
    )
    if corporate_actions["invalid_row_count"]:
        raise IncrementalStoreError(
            "corporate-action normalization rejected provider rows: "
            f"{corporate_actions['invalid_row_count']}"
        )
    lifecycle_events = _capture_fmp_lifecycle_events(
        pipeline=pipeline,
        target=target,
    )
    flow_result = pipeline.capture_etf_flows(
        target,
        lookback_days=10,
        limit=5000,
        max_lag_days=4,
        resume=True,
        strict_freshness=True,
    )
    flow = _flow_summary(database_path)
    expected_flow_effective = sessions[-3] if len(sessions) >= 3 else base_end
    if (
        not flow_result.get("ok", True)
        or not flow["latest_effective_date"]
        or str(flow["latest_effective_date"]) < expected_flow_effective
    ):
        raise IncrementalStoreError(
            "Massive ETF Flow D+2 gate failed: "
            f"latest_effective={flow['latest_effective_date']} "
            f"expected_at_least={expected_flow_effective}"
        )
    constituent_refresh = _refresh_constituents(
        pipeline=pipeline,
        base_database=base_database,
        database_path=database_path,
        target=target,
        stale_days=constituent_stale_days,
        max_etfs=constituent_refresh_max_etfs,
    )
    receipt = {
        "schema": INCREMENTAL_SCHEMA,
        "source_contract": SOURCE_CONTRACT,
        "base_history_end": base_end,
        "target_as_of_date": target,
        "expected_sessions": sessions,
        "market_row_gate": {
            "minimum_rows": session_minimum_rows.get(target, 0),
            "minimum_rows_by_session": session_minimum_rows,
            "rows_by_session": session_rows,
            "rows_by_session_and_source": session_rows_by_source,
            "capture_by_session": session_capture,
            "source_priority": [
                "fmp_stable_eod_bulk",
                "fmp_per_symbol_eod",
            ],
            "massive_stock_daily_allowed": False,
        },
        "symbol_coverage_gate": {
            "universe_policy": "daily_fmp_active_master_with_historical_reconstruction",
            "active_master": active_snapshot,
            "session_master": session_master,
            "protected_source_path": str(protected_symbols_path),
            "protected_symbols": protected_symbols,
            "sessions": coverage_by_session,
            "status": "complete",
            "terminal_outcomes": [
                "BAR",
                "NO_BAR",
                "QUARANTINED_INVALID_BAR",
            ],
            "error_count": 0,
        },
        "etf_flow": {
            **flow,
            "expected_effective_date_at_least": expected_flow_effective,
            "capture": flow_result,
            "point_in_time_gate": (
                "effective_date,processed_date,available_at_date <= as_of"
            ),
        },
        "corporate_actions": corporate_actions,
        "lifecycle_events": lifecycle_events,
        "symbol_lineage": symbol_lineage,
        "etf_constituents": constituent_refresh,
        "database": str(database_path),
    }
    receipt_sha = hashlib.sha256(
        canonical_json(receipt).encode("utf-8")
    ).hexdigest()
    status = {
        "status": "COMPLETE",
        **receipt,
        "single_writer": "market_structure_oracle",
        "base_price_source": "FMP immutable long history",
        "incremental_price_source": (
            "FMP Ultimate stable/eod-bulk full-market daily, then exact "
            "per-symbol repair only for verified missing or invalid symbols"
        ),
        "missing_sessions": [],
        "repaired_sessions": repaired_sessions,
        "snapshot_seal": {
            "schema_version": "quant.oracle_snapshot_seal.v1",
            "receipt_sha256": receipt_sha,
            "source_contract": receipt["source_contract"],
        },
    }
    _atomic_json(incremental_root / STATUS_FILE, status)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO oracle_snapshot_seals
            VALUES (?,?,?,?,?,datetime('now'))
            """,
            (
                target,
                "quant.oracle_snapshot_seal.v1",
                receipt["source_contract"],
                receipt_sha,
                canonical_json(receipt),
            ),
        )
        connection.execute(
            "INSERT INTO oracle_incremental_runs VALUES (?,?,?,datetime('now'))",
            (target, "COMPLETE", canonical_json(status)),
        )
    return status


def ensure_oracle_snapshot(
    *,
    base_database: Path,
    incremental_root: Path,
    target_as_of_date: str | None = None,
    force_repair: bool = False,
    publish_grace_hour_et: int = 18,
    constituent_stale_days: int = 45,
    constituent_refresh_max_etfs: int = 50,
    protected_symbols_path: Path = DEFAULT_PROTECTED_SYMBOLS,
) -> dict[str, Any]:
    """Single-writer entrypoint; concurrent callers reuse one sealed snapshot."""

    target = target_as_of_date or latest_closed_nyse_session(
        publish_grace_hour_et=publish_grace_hour_et
    )
    status_path = incremental_root / STATUS_FILE
    with _writer_lock(incremental_root):
        if not force_repair and _status_target(status_path) == target:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            gate = status.get("symbol_coverage_gate") or {}
            current_protected = _load_protected_symbols(
                protected_symbols_path
            )
            if (
                gate.get("status") == "complete"
                and gate.get("protected_symbols") == current_protected
                and status.get("schema") == INCREMENTAL_SCHEMA
                and status.get("source_contract") == SOURCE_CONTRACT
            ):
                return {**status, "ensure_mode": "reused_existing_complete"}
        status = _materialize(
            base_database=base_database,
            incremental_root=incremental_root,
            target_as_of_date=target,
            constituent_stale_days=constituent_stale_days,
            constituent_refresh_max_etfs=constituent_refresh_max_etfs,
            protected_symbols_path=protected_symbols_path,
        )
        return {**status, "ensure_mode": "materialized_or_repaired"}


def materialize_incremental_store(
    *,
    base_database: Path,
    incremental_root: Path,
    target_as_of_date: str | None = None,
    constituent_stale_days: int = 45,
    constituent_refresh_max_etfs: int = 50,
) -> dict[str, Any]:
    """Compatibility entrypoint that forces an Oracle-owned repair."""

    return ensure_oracle_snapshot(
        base_database=base_database,
        incremental_root=incremental_root,
        target_as_of_date=target_as_of_date,
        force_repair=True,
        constituent_stale_days=constituent_stale_days,
        constituent_refresh_max_etfs=constituent_refresh_max_etfs,
    )
