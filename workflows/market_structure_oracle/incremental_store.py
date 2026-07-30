"""Oracle-owned, resumable current-market delta store.

The immutable long-history database remains untouched.  Market Structure
Oracle is the only writer for current Massive daily/ETF-flow rows and periodic
FMP ETF-constituent refreshes.  Consumers read only a sealed status receipt and
the base+incremental SQLite overlay; ETF RADAR is a separate application and
is not a source or release gate for this store.
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
STATUS_FILE = "state/oracle_incremental_status.json"
LOCK_FILE = "state/oracle_incremental.lock"
ET = ZoneInfo("America/New_York")
FMP_LEGACY_EOD_BULK_URL = (
    "https://financialmodelingprep.com/api/v4/batch-historical-eod"
)
DEFAULT_VERIFIED_CORPORATE_ACTIONS = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/"
    "incremental/state/verified_corporate_actions.json"
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
    Materialization tries Massive first and uses the source-preserving FMP
    legacy-bulk/per-symbol fallback when the configured Massive plan delays the
    same-day grouped bar.
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
            """
        )


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


def _reference_market_rows(base_database: Path) -> int:
    with sqlite3.connect(
        f"file:{base_database}?mode=ro", uri=True
    ) as connection:
        value = connection.execute(
            """
            SELECT MAX(n) FROM (
              SELECT COUNT(*) AS n FROM daily_observations
              WHERE source=? GROUP BY trade_date
            )
            """,
            (MASSIVE_SOURCE,),
        ).fetchone()[0]
    return int(value or MIN_RELEASE_MARKET_ROWS)


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
                WHERE source IN ('massive','fmp') AND trade_date=?
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
                WHERE trade_date=? AND source IN ('massive','fmp')
                  AND close>0
                """,
                (trade_date,),
            )
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
    snapshot = pipeline.capture_fmp_universe(target)
    symbols_path = Path(str(snapshot["symbols_path"]))
    symbols = sorted(
        {
            line.strip().upper()
            for line in symbols_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    if len(symbols) < MIN_RELEASE_MARKET_ROWS:
        raise IncrementalStoreError(
            f"FMP US equity/ETF universe is unexpectedly small: {len(symbols)}"
        )
    return symbols, snapshot


def _capture_fmp_legacy_eod_bulk(
    *,
    pipeline: DatasetPipeline,
    target: str,
    allowed_symbols: set[str],
) -> dict[str, Any]:
    """Capture the currently accessible legacy CSV endpoint with raw evidence."""

    key = pipeline.credentials.fmp_api_key
    if not key:
        raise IncrementalStoreError("FMP_API_KEY is not configured")
    safe_request = redacted_request_metadata(
        "GET",
        FMP_LEGACY_EOD_BULK_URL,
        {"date": target, "apikey": key},
        {
            "endpoint_contract": "fmp_v4_legacy_batch_historical_eod",
            "date": target,
            "support_status": "legacy_best_effort_not_starter_entitlement",
        },
    )
    response = None
    artifact = None
    for attempt in range(3):
        limiter = pipeline.http.rate_limiters.get("fmp")
        if limiter is not None:
            limiter.acquire()
        response = pipeline.http.session.get(
            FMP_LEGACY_EOD_BULK_URL,
            params={"date": target, "apikey": key},
            timeout=120,
        )
        payload = bytes(response.content)
        artifact = pipeline.raw_store.store(
            source="fmp",
            dataset="batch_historical_eod_v4_legacy",
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
            "fmp batch_historical_eod_v4_legacy returned HTTP {} "
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
            "FMP legacy EOD bulk CSV parsing failed "
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
                        "fmp_v4_legacy_batch_historical_eod"
                    ),
                    "legacy_best_effort": True,
                },
            }
        )
    inserted = pipeline.database.upsert_observations(observations)
    pipeline.quality.recompute(target, target)
    pipeline.write_manifest()
    return {
        "ok": True,
        "mode": "fmp_legacy_eod_bulk",
        "raw_row_count": len(rows),
        "matched_us_universe_count": len(observations),
        "inserted_observation_count": inserted,
        "raw_artifact_id": artifact.artifact_id,
        "payload_sha256": artifact.payload_sha256,
        "support_status": "legacy_best_effort_not_starter_entitlement",
    }


def _capture_current_session(
    *,
    pipeline: DatasetPipeline,
    session: str,
    massive_minimum_rows: int,
    fmp_minimum_rows: int,
) -> dict[str, Any]:
    """Massive -> FMP legacy bulk -> FMP per-symbol resumable fallback."""

    attempts: list[dict[str, Any]] = []
    try:
        massive = pipeline.capture_daily(
            session,
            [],
            source=MASSIVE_SOURCE,
            continue_on_error=False,
        )
        attempts.append({"source": "massive", "result": massive})
    except ApiRequestError as exc:
        attempts.append(
            {
                "source": "massive",
                "status": "failed",
                "error": str(exc),
                "http_status": exc.status_code,
                "raw_artifact_id": exc.raw_artifact_id,
            }
        )
    count = _market_row_count(pipeline.database.db_path, session)
    if count >= massive_minimum_rows:
        return {
            "mode": "massive_grouped_daily",
            "accepted_minimum_rows": massive_minimum_rows,
            "attempts": attempts,
        }

    symbols, universe = _daily_universe(pipeline, session)
    try:
        legacy = _capture_fmp_legacy_eod_bulk(
            pipeline=pipeline,
            target=session,
            allowed_symbols=set(symbols),
        )
        attempts.append({"source": "fmp_legacy_bulk", "result": legacy})
    except (ApiRequestError, IncrementalStoreError) as exc:
        attempts.append(
            {
                "source": "fmp_legacy_bulk",
                "status": "failed",
                "error": str(exc),
                "http_status": getattr(exc, "status_code", None),
                "raw_artifact_id": getattr(exc, "raw_artifact_id", None),
            }
        )
    count = _market_row_count(pipeline.database.db_path, session)
    if count >= fmp_minimum_rows:
        return {
            "mode": "fmp_legacy_eod_bulk",
            "accepted_minimum_rows": fmp_minimum_rows,
            "universe": universe,
            "attempts": attempts,
        }

    observed = _observed_market_symbols(pipeline.database.db_path, session)
    missing_symbols = [symbol for symbol in symbols if symbol not in observed]
    if not missing_symbols:
        raise IncrementalStoreError(
            "full-market row gate failed after all FMP universe symbols "
            f"were observed for {session}"
        )
    per_symbol = pipeline.capture_daily(
        session,
        missing_symbols,
        source="fmp",
        continue_on_error=True,
    )
    attempts.append(
        {
            "source": "fmp_per_symbol_eod",
            "already_covered_by_bulk": len(observed),
            "requested_missing_symbols": len(missing_symbols),
            "result": per_symbol,
        }
    )
    return {
        "mode": "fmp_per_symbol_eod",
        "accepted_minimum_rows": fmp_minimum_rows,
        "universe": universe,
        "attempts": attempts,
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
    if status.get("schema") != "quant.market_structure_oracle.incremental.v3":
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
    minimum_rows = max(
        MIN_RELEASE_MARKET_ROWS,
        int(_reference_market_rows(base_database) * 0.90),
    )
    fmp_minimum_rows = max(
        1, int(_reference_fmp_rows(base_database) * 0.90)
    )
    session_rows: dict[str, int] = {}
    session_minimum_rows: dict[str, int] = {}
    session_rows_by_source: dict[str, dict[str, int]] = {}
    session_capture: dict[str, dict[str, Any]] = {}
    repaired_sessions: list[str] = []
    for session in sessions:
        count = _market_row_count(database_path, session)
        source_rows = _market_rows_by_source(database_path, session)
        required_rows = (
            fmp_minimum_rows
            if source_rows.get("fmp", 0) >= fmp_minimum_rows
            and source_rows.get("massive", 0) < minimum_rows
            else minimum_rows
        )
        if count < required_rows:
            result = _capture_current_session(
                pipeline=pipeline,
                session=session,
                massive_minimum_rows=minimum_rows,
                fmp_minimum_rows=fmp_minimum_rows,
            )
            session_capture[session] = result
            required_rows = int(result["accepted_minimum_rows"])
            repaired_sessions.append(session)
            count = _market_row_count(database_path, session)
        if count < required_rows:
            raise IncrementalStoreError(
                f"Oracle full-market row gate failed for {session}: "
                f"{count} < {required_rows}"
            )
        session_rows[session] = count
        session_minimum_rows[session] = required_rows
        session_rows_by_source[session] = _market_rows_by_source(
            database_path, session
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
        "schema": "quant.market_structure_oracle.incremental.v3",
        "source_contract": "oracle_owned_fmp_massive_no_etf_radar_dependency",
        "base_history_end": base_end,
        "target_as_of_date": target,
        "expected_sessions": sessions,
        "market_row_gate": {
            "minimum_rows": minimum_rows,
            "massive_minimum_rows": minimum_rows,
            "fmp_minimum_rows": fmp_minimum_rows,
            "minimum_rows_by_session": session_minimum_rows,
            "rows_by_session": session_rows,
            "rows_by_session_and_source": session_rows_by_source,
            "capture_by_session": session_capture,
            "source_priority": [
                "massive_grouped_daily",
                "fmp_legacy_eod_bulk",
                "fmp_per_symbol_eod",
            ],
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
            "Massive grouped daily, then FMP legacy bulk, then resumable "
            "FMP per-symbol EOD"
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
) -> dict[str, Any]:
    """Single-writer entrypoint; concurrent callers reuse one sealed snapshot."""

    target = target_as_of_date or latest_closed_nyse_session(
        publish_grace_hour_et=publish_grace_hour_et
    )
    status_path = incremental_root / STATUS_FILE
    with _writer_lock(incremental_root):
        if not force_repair and _status_target(status_path) == target:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            return {**status, "ensure_mode": "reused_existing_complete"}
        status = _materialize(
            base_database=base_database,
            incremental_root=incremental_root,
            target_as_of_date=target,
            constituent_stale_days=constituent_stale_days,
            constituent_refresh_max_etfs=constituent_refresh_max_etfs,
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
