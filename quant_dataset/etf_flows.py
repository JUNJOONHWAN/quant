"""Massive ETF Global fund-flow capture, versioning, and PIT projection.

The contract in this module is intentionally separate from ETF holdings.  It
implements Massive endpoint ``partners_etf_fund_flows`` only and preserves the
provider's effective date and processed date as different facts.
"""

from __future__ import annotations

import math
import sqlite3
import uuid
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from .providers import (
    CredentialError,
    HttpCaptureClient,
    PayloadValidationError,
    normalize_symbol,
    validate_iso_date,
)
from .point_in_time import (
    ETF_FLOW_PIT_FILTER,
    ETF_FLOW_POLICY_ID,
    US_EQUITY_SESSION_SQL,
    derive_etf_flow_available_session,
    etf_flow_policy_manifest,
    normalize_trading_sessions,
)
from .storage import Database, RawArtifact, canonical_json, sha256_bytes, utc_now


MASSIVE_ETF_FLOW_ENDPOINT_ID = "partners_etf_fund_flows"
MASSIVE_ETF_FLOW_URL = "https://api.massive.com/etf-global/v1/fund-flows"
MASSIVE_ETF_FLOW_PATH = "/etf-global/v1/fund-flows"
MASSIVE_ETF_FLOW_MAX_LIMIT = 5000
# The live ETF Global endpoint accepts exactly one sortable field.  Pagination
# remains deterministic through Massive's opaque ``next_url`` cursor.
MASSIVE_ETF_FLOW_SORT = "processed_date.asc"
_SENSITIVE_QUERY_NAMES = {
    "apikey",
    "api_key",
    "authorization",
    "token",
    "access_token",
    "secret",
}
_STALE_STATUSES = {
    "empty_requested_window",
    "future_processed_date",
    "stale_source_date",
    "stale_repeated_hash",
}


@dataclass(frozen=True)
class EtfFlowPage:
    """One captured Massive page and its normalized rows."""

    artifact: RawArtifact
    records: List[dict]
    invalid_rows: List[dict]
    next_url: Optional[str]


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _normalized_tickers(tickers: Optional[Sequence[str]]) -> List[str]:
    result = []
    for ticker in tickers or []:
        normalized = normalize_symbol(ticker)
        if normalized not in result:
            result.append(normalized)
    return sorted(result)


def _safe_next_url(value: Any) -> Optional[str]:
    """Validate a provider cursor and remove any credential-like query fields."""

    if value in (None, ""):
        return None
    absolute = urljoin(MASSIVE_ETF_FLOW_URL, str(value).strip())
    split = urlsplit(absolute)
    if split.scheme.lower() != "https" or (split.hostname or "").lower() != "api.massive.com":
        raise PayloadValidationError("Massive ETF flow next_url has an untrusted host")
    if split.path.rstrip("/") != MASSIVE_ETF_FLOW_PATH:
        raise PayloadValidationError("Massive ETF flow next_url changed endpoint path")
    query = [
        (key, item)
        for key, item in parse_qsl(split.query, keep_blank_values=True)
        if key.lower().replace("-", "_") not in _SENSITIVE_QUERY_NAMES
    ]
    return urlunsplit(("https", "api.massive.com", split.path, urlencode(query), ""))


def _date_or_none(value: Any) -> Optional[str]:
    try:
        return validate_iso_date(str(value)[:10])
    except ValueError:
        return None


def _record_hash(record: Mapping[str, Any]) -> str:
    business_fields = {
        key: record.get(key)
        for key in (
            "provider",
            "endpoint_id",
            "ticker",
            "effective_date",
            "processed_date",
            "fund_flow",
            "nav",
            "shares_outstanding",
            "assets",
            "currency",
            "extra",
        )
    }
    return sha256_bytes(canonical_json(business_fields).encode("utf-8"))


class MassiveEtfFlowProvider:
    """Header-authenticated adapter for Massive ETF Global fund flows."""

    def __init__(self, http: HttpCaptureClient, api_key: Optional[str]):
        self.http = http
        self.api_key = api_key

    def capture_page(
        self,
        *,
        url: str,
        params: Mapping[str, Any],
        partition_key: str,
        page_number: int,
        contract_hash: str,
    ) -> EtfFlowPage:
        if not self.api_key:
            raise CredentialError("MASSIVE_API_KEY is not configured")
        safe_url = _safe_next_url(url)
        if not safe_url:
            raise ValueError("Massive ETF flow request URL is required")
        result = self.http.get_json(
            source="massive",
            dataset="etf_fund_flows",
            partition_key=partition_key,
            url=safe_url,
            params=dict(params),
            headers={"Authorization": "Bearer {}".format(self.api_key)},
            logical_request={
                "endpoint_id": MASSIVE_ETF_FLOW_ENDPOINT_ID,
                "endpoint_contract": "massive_etf_global_fund_flows",
                "contract_hash": contract_hash,
                "page_number": page_number,
            },
        )
        document = result.document
        if not isinstance(document, dict) or not isinstance(document.get("results"), list):
            raise PayloadValidationError(
                "Massive ETF fund-flow payload has no results list "
                "(raw artifact id={})".format(result.artifact.artifact_id)
            )

        recognized = {
            "composite_ticker",
            "ticker",
            "symbol",
            "effective_date",
            "processed_date",
            "fund_flow",
            "nav",
            "shares_outstanding",
            "assets",
            "aum",
            "currency",
        }
        records: List[dict] = []
        invalid_rows: List[dict] = []
        for index, row in enumerate(document.get("results") or []):
            if not isinstance(row, dict):
                invalid_rows.append({"source_row_index": index, "reason": "row_not_object"})
                continue
            try:
                ticker = normalize_symbol(
                    row.get("composite_ticker") or row.get("ticker") or row.get("symbol"),
                    uppercase=False,
                )
            except ValueError:
                invalid_rows.append({"source_row_index": index, "reason": "ticker_invalid"})
                continue
            effective_date = _date_or_none(row.get("effective_date"))
            processed_date = _date_or_none(row.get("processed_date"))
            if not effective_date or not processed_date:
                invalid_rows.append(
                    {
                        "source_row_index": index,
                        "reason": "effective_or_processed_date_invalid",
                    }
                )
                continue
            extra = {key: row[key] for key in sorted(row) if key not in recognized}
            record = {
                "provider": "massive",
                "endpoint_id": MASSIVE_ETF_FLOW_ENDPOINT_ID,
                "ticker": ticker,
                "effective_date": effective_date,
                "processed_date": processed_date,
                "fund_flow": _number(row.get("fund_flow")),
                "nav": _number(row.get("nav")),
                "shares_outstanding": _number(row.get("shares_outstanding")),
                "assets": _number(row.get("assets", row.get("aum"))),
                "currency": str(row.get("currency") or "").strip() or None,
                "available_at_date": processed_date,
                "availability_basis": "provider_processed_date",
                "pit_confidence": "date_only",
                "source_record_id": sha256_bytes(canonical_json(row).encode("utf-8")),
                "raw_artifact_id": result.artifact.artifact_id,
                "capture_event_id": result.artifact.capture_event_id,
                "source_row_index": index,
                "captured_at_utc": result.artifact.captured_at_utc,
                "extra": extra,
            }
            record["record_hash"] = _record_hash(record)
            records.append(record)

        return EtfFlowPage(
            artifact=result.artifact,
            records=records,
            invalid_rows=invalid_rows,
            next_url=_safe_next_url(document.get("next_url")),
        )


class EtfFlowStore:
    """Append-only ETF flow versions, latest projections, and resume state."""

    def __init__(self, database: Database, *, initialize_schema: bool = True):
        self.database = database
        self._trading_sessions_cache: Optional[Tuple[str, ...]] = None
        if initialize_schema:
            self.initialize()

    def _trading_sessions(self) -> Tuple[str, ...]:
        """Load the observed U.S. equity calendar once per packet-export run."""

        if self._trading_sessions_cache is None:
            with self.database.connect() as connection:
                rows = connection.execute(US_EQUITY_SESSION_SQL).fetchall()
            self._trading_sessions_cache = normalize_trading_sessions(
                row["trade_date"] for row in rows
            )
        return self._trading_sessions_cache

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS etf_flow_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            endpoint_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            effective_date TEXT NOT NULL,
            processed_date TEXT NOT NULL,
            fund_flow REAL,
            nav REAL,
            shares_outstanding REAL,
            assets REAL,
            currency TEXT,
            available_at_date TEXT NOT NULL,
            availability_basis TEXT NOT NULL,
            pit_confidence TEXT NOT NULL,
            record_hash TEXT NOT NULL,
            source_record_id TEXT NOT NULL,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            source_row_index INTEGER NOT NULL,
            captured_at_utc TEXT NOT NULL,
            ingested_at_utc TEXT NOT NULL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(capture_event_id, source_row_index),
            FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id),
            FOREIGN KEY(capture_event_id) REFERENCES capture_events(id)
        );

        CREATE TABLE IF NOT EXISTS etf_flow_observations (
            provider TEXT NOT NULL,
            endpoint_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            effective_date TEXT NOT NULL,
            processed_date TEXT NOT NULL,
            fund_flow REAL,
            nav REAL,
            shares_outstanding REAL,
            assets REAL,
            currency TEXT,
            available_at_date TEXT NOT NULL,
            availability_basis TEXT NOT NULL,
            pit_confidence TEXT NOT NULL,
            record_hash TEXT NOT NULL,
            source_record_id TEXT NOT NULL,
            version_id INTEGER NOT NULL,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            source_row_index INTEGER NOT NULL,
            captured_at_utc TEXT NOT NULL,
            ingested_at_utc TEXT NOT NULL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(provider, ticker, effective_date),
            FOREIGN KEY(version_id) REFERENCES etf_flow_versions(id),
            FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id),
            FOREIGN KEY(capture_event_id) REFERENCES capture_events(id)
        );

        CREATE TABLE IF NOT EXISTS etf_flow_runs (
            run_id TEXT PRIMARY KEY,
            contract_hash TEXT NOT NULL,
            series_key TEXT NOT NULL,
            job_type TEXT NOT NULL,
            contract_json TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            resume_next_url TEXT,
            page_count INTEGER NOT NULL DEFAULT 0,
            record_count INTEGER NOT NULL DEFAULT 0,
            invalid_row_count INTEGER NOT NULL DEFAULT 0,
            payload_set_sha256 TEXT,
            normalized_set_sha256 TEXT,
            min_processed_date TEXT,
            max_processed_date TEXT,
            min_effective_date TEXT,
            max_effective_date TEXT,
            freshness_status TEXT,
            repeated_payload_hash INTEGER NOT NULL DEFAULT 0,
            repeated_normalized_hash INTEGER NOT NULL DEFAULT 0,
            prior_run_id TEXT,
            max_lag_days INTEGER,
            last_error TEXT
        );

        CREATE TABLE IF NOT EXISTS etf_flow_run_pages (
            run_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            payload_sha256 TEXT NOT NULL,
            request_url TEXT NOT NULL,
            next_url TEXT,
            record_count INTEGER NOT NULL,
            invalid_row_count INTEGER NOT NULL,
            captured_at_utc TEXT NOT NULL,
            PRIMARY KEY(run_id, page_number),
            UNIQUE(run_id, capture_event_id),
            FOREIGN KEY(run_id) REFERENCES etf_flow_runs(run_id),
            FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id),
            FOREIGN KEY(capture_event_id) REFERENCES capture_events(id)
        );

        CREATE INDEX IF NOT EXISTS idx_etf_flow_versions_ticker_dates
            ON etf_flow_versions(ticker, effective_date, processed_date, captured_at_utc);
        CREATE INDEX IF NOT EXISTS idx_etf_flow_projection_ticker_dates
            ON etf_flow_observations(ticker, effective_date, processed_date);
        CREATE INDEX IF NOT EXISTS idx_etf_flow_runs_contract_status
            ON etf_flow_runs(contract_hash, status, started_at_utc);
        CREATE INDEX IF NOT EXISTS idx_etf_flow_runs_series_status
            ON etf_flow_runs(series_key, status, completed_at_utc);

        CREATE VIEW IF NOT EXISTS etf_flow_latest AS
        SELECT current.*
        FROM etf_flow_observations AS current
        WHERE NOT EXISTS (
            SELECT 1 FROM etf_flow_observations AS newer
            WHERE newer.provider=current.provider
              AND newer.ticker=current.ticker
              AND (
                    newer.effective_date>current.effective_date
                 OR (newer.effective_date=current.effective_date
                     AND newer.processed_date>current.processed_date)
              )
        );
        """
        with self.database.connect() as connection:
            connection.executescript(schema)

    def start_or_resume(
        self,
        *,
        contract_hash: str,
        series_key: str,
        job_type: str,
        contract: Mapping[str, Any],
        max_lag_days: Optional[int],
        resume: bool,
    ) -> Tuple[sqlite3.Row, bool]:
        with self.database.connect() as connection:
            row = None
            if resume:
                row = connection.execute(
                    """
                    SELECT * FROM etf_flow_runs
                    WHERE contract_hash=? AND status IN ('running','failed')
                    ORDER BY started_at_utc DESC LIMIT 1
                    """,
                    (contract_hash,),
                ).fetchone()
            if row is not None:
                connection.execute(
                    """
                    UPDATE etf_flow_runs
                    SET status='running', updated_at_utc=?, last_error=NULL
                    WHERE run_id=?
                    """,
                    (utc_now(), row["run_id"]),
                )
                return connection.execute(
                    "SELECT * FROM etf_flow_runs WHERE run_id=?", (row["run_id"],)
                ).fetchone(), True

            now = utc_now()
            run_id = "etf-flow-{}-{}".format(
                now.replace("-", "").replace(":", "").replace("+00:00", "Z"),
                uuid.uuid4().hex[:8],
            )
            connection.execute(
                """
                INSERT INTO etf_flow_runs (
                    run_id, contract_hash, series_key, job_type, contract_json,
                    status, started_at_utc, updated_at_utc, max_lag_days
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    run_id,
                    contract_hash,
                    series_key,
                    job_type,
                    canonical_json(contract),
                    now,
                    now,
                    max_lag_days,
                ),
            )
            return connection.execute(
                "SELECT * FROM etf_flow_runs WHERE run_id=?", (run_id,)
            ).fetchone(), False

    @staticmethod
    def _version_values(record: Mapping[str, Any], ingested_at: str) -> Tuple[Any, ...]:
        return (
            record["provider"],
            record["endpoint_id"],
            record["ticker"],
            record["effective_date"],
            record["processed_date"],
            record.get("fund_flow"),
            record.get("nav"),
            record.get("shares_outstanding"),
            record.get("assets"),
            record.get("currency"),
            record["available_at_date"],
            record["availability_basis"],
            record["pit_confidence"],
            record["record_hash"],
            record["source_record_id"],
            record["raw_artifact_id"],
            record["capture_event_id"],
            record["source_row_index"],
            record["captured_at_utc"],
            ingested_at,
            canonical_json(record.get("extra", {})),
        )

    def ingest_page(
        self,
        *,
        run_id: str,
        page_number: int,
        request_url: str,
        page: EtfFlowPage,
    ) -> None:
        version_sql = """
        INSERT OR IGNORE INTO etf_flow_versions (
            provider, endpoint_id, ticker, effective_date, processed_date,
            fund_flow, nav, shares_outstanding, assets, currency,
            available_at_date, availability_basis, pit_confidence, record_hash,
            source_record_id, raw_artifact_id, capture_event_id, source_row_index,
            captured_at_utc, ingested_at_utc, extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        projection_sql = """
        INSERT INTO etf_flow_observations (
            provider, endpoint_id, ticker, effective_date, processed_date,
            fund_flow, nav, shares_outstanding, assets, currency,
            available_at_date, availability_basis, pit_confidence, record_hash,
            source_record_id, version_id, raw_artifact_id, capture_event_id,
            source_row_index, captured_at_utc, ingested_at_utc, extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, ticker, effective_date) DO UPDATE SET
            endpoint_id=excluded.endpoint_id,
            processed_date=excluded.processed_date,
            fund_flow=excluded.fund_flow,
            nav=excluded.nav,
            shares_outstanding=excluded.shares_outstanding,
            assets=excluded.assets,
            currency=excluded.currency,
            available_at_date=excluded.available_at_date,
            availability_basis=excluded.availability_basis,
            pit_confidence=excluded.pit_confidence,
            record_hash=excluded.record_hash,
            source_record_id=excluded.source_record_id,
            version_id=excluded.version_id,
            raw_artifact_id=excluded.raw_artifact_id,
            capture_event_id=excluded.capture_event_id,
            source_row_index=excluded.source_row_index,
            captured_at_utc=excluded.captured_at_utc,
            ingested_at_utc=excluded.ingested_at_utc,
            extra_json=excluded.extra_json
        WHERE excluded.processed_date>etf_flow_observations.processed_date
           OR (
                excluded.processed_date=etf_flow_observations.processed_date
                AND excluded.captured_at_utc>etf_flow_observations.captured_at_utc
           )
        """
        ingested_at = utc_now()
        with self.database.connect() as connection:
            for record in page.records:
                values = self._version_values(record, ingested_at)
                connection.execute(version_sql, values)
                version = connection.execute(
                    """
                    SELECT id FROM etf_flow_versions
                    WHERE capture_event_id=? AND source_row_index=?
                    """,
                    (record["capture_event_id"], record["source_row_index"]),
                ).fetchone()
                if version is None:
                    raise RuntimeError("ETF flow version registration failed")
                projection_values = values[:15] + (int(version["id"]),) + values[15:]
                connection.execute(projection_sql, projection_values)

            connection.execute(
                """
                INSERT INTO etf_flow_run_pages (
                    run_id, page_number, raw_artifact_id, capture_event_id,
                    payload_sha256, request_url, next_url, record_count,
                    invalid_row_count, captured_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    page_number,
                    page.artifact.artifact_id,
                    page.artifact.capture_event_id,
                    page.artifact.payload_sha256,
                    _safe_next_url(request_url),
                    page.next_url,
                    len(page.records),
                    len(page.invalid_rows),
                    page.artifact.captured_at_utc,
                ),
            )
            totals = connection.execute(
                """
                SELECT COUNT(*) AS pages,
                       COALESCE(SUM(record_count),0) AS records,
                       COALESCE(SUM(invalid_row_count),0) AS invalid_rows
                FROM etf_flow_run_pages WHERE run_id=?
                """,
                (run_id,),
            ).fetchone()
            connection.execute(
                """
                UPDATE etf_flow_runs
                SET resume_next_url=?, page_count=?, record_count=?,
                    invalid_row_count=?, updated_at_utc=?
                WHERE run_id=?
                """,
                (
                    page.next_url,
                    int(totals["pages"]),
                    int(totals["records"]),
                    int(totals["invalid_rows"]),
                    utc_now(),
                    run_id,
                ),
            )

    def mark_failed(self, run_id: str, error: BaseException) -> None:
        message = "{}: {}".format(type(error).__name__, str(error))[:1000]
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE etf_flow_runs
                SET status='failed', last_error=?, updated_at_utc=?
                WHERE run_id=?
                """,
                (message, utc_now(), run_id),
            )

    def finalize(
        self,
        run_id: str,
        *,
        as_of_date: Optional[str],
        max_lag_days: Optional[int],
        historical: bool,
    ) -> sqlite3.Row:
        with self.database.connect() as connection:
            run = connection.execute(
                "SELECT * FROM etf_flow_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            pages = connection.execute(
                """
                SELECT * FROM etf_flow_run_pages
                WHERE run_id=? ORDER BY page_number
                """,
                (run_id,),
            ).fetchall()
            payload_hash = sha256_bytes(
                canonical_json([row["payload_sha256"] for row in pages]).encode("utf-8")
            )
            summary = connection.execute(
                """
                SELECT MIN(v.processed_date) AS min_processed_date,
                       MAX(v.processed_date) AS max_processed_date,
                       MIN(v.effective_date) AS min_effective_date,
                       MAX(v.effective_date) AS max_effective_date
                FROM etf_flow_versions v
                JOIN etf_flow_run_pages p ON p.capture_event_id=v.capture_event_id
                WHERE p.run_id=?
                """,
                (run_id,),
            ).fetchone()
            record_hashes = [
                str(row["record_hash"])
                for row in connection.execute(
                    """
                    SELECT v.record_hash
                    FROM etf_flow_versions v
                    JOIN etf_flow_run_pages p ON p.capture_event_id=v.capture_event_id
                    WHERE p.run_id=?
                    ORDER BY v.record_hash, v.source_row_index
                    """,
                    (run_id,),
                ).fetchall()
            ]
            normalized_hash = sha256_bytes(canonical_json(record_hashes).encode("utf-8"))
            prior = connection.execute(
                """
                SELECT * FROM etf_flow_runs
                WHERE series_key=? AND status='complete' AND run_id<>?
                ORDER BY completed_at_utc DESC LIMIT 1
                """,
                (run["series_key"], run_id),
            ).fetchone()
            repeated_payload = bool(
                prior and prior["payload_set_sha256"] == payload_hash
            )
            repeated_normalized = bool(
                prior and prior["normalized_set_sha256"] == normalized_hash
            )
            maximum_processed = summary["max_processed_date"]
            if not record_hashes:
                freshness = "empty_requested_window"
            elif historical:
                freshness = "historical_window_captured"
            elif as_of_date and maximum_processed:
                lag_days = (
                    date.fromisoformat(as_of_date)
                    - date.fromisoformat(str(maximum_processed))
                ).days
                if lag_days < 0:
                    freshness = "future_processed_date"
                elif max_lag_days is not None and lag_days > max_lag_days:
                    freshness = (
                        "stale_repeated_hash"
                        if repeated_payload or repeated_normalized
                        else "stale_source_date"
                    )
                elif repeated_payload or repeated_normalized:
                    freshness = "unchanged_repeated_hash"
                else:
                    freshness = "fresh"
            else:
                freshness = "freshness_not_evaluated"

            completed_at = utc_now()
            connection.execute(
                """
                UPDATE etf_flow_runs
                SET status='complete', completed_at_utc=?, updated_at_utc=?,
                    resume_next_url=NULL, payload_set_sha256=?,
                    normalized_set_sha256=?, min_processed_date=?,
                    max_processed_date=?, min_effective_date=?, max_effective_date=?,
                    freshness_status=?, repeated_payload_hash=?,
                    repeated_normalized_hash=?, prior_run_id=?, last_error=NULL
                WHERE run_id=?
                """,
                (
                    completed_at,
                    completed_at,
                    payload_hash,
                    normalized_hash,
                    summary["min_processed_date"],
                    maximum_processed,
                    summary["min_effective_date"],
                    summary["max_effective_date"],
                    freshness,
                    1 if repeated_payload else 0,
                    1 if repeated_normalized else 0,
                    prior["run_id"] if prior else None,
                    run_id,
                ),
            )
            return connection.execute(
                "SELECT * FROM etf_flow_runs WHERE run_id=?", (run_id,)
            ).fetchone()

    def run_request_urls(self, run_id: str) -> List[str]:
        with self.database.connect() as connection:
            return [
                str(row["request_url"])
                for row in connection.execute(
                    """
                    SELECT request_url FROM etf_flow_run_pages
                    WHERE run_id=? ORDER BY page_number
                    """,
                    (run_id,),
                ).fetchall()
            ]

    @staticmethod
    def _packet_document(
        as_of: str, selected: Sequence[Tuple[Mapping[str, Any], str]]
    ) -> dict:
        observations = [
            {
                "provider": str(row["provider"]),
                "endpoint_id": str(row["endpoint_id"]),
                "ticker": str(row["ticker"]),
                "effective_date": str(row["effective_date"]),
                "processed_date": str(row["processed_date"]),
                "fund_flow": row["fund_flow"],
                "nav": row["nav"],
                "shares_outstanding": row["shares_outstanding"],
                "assets": row["assets"],
                "currency": row["currency"],
                "available_at_date": training_available_session_date,
                "availability_basis": ETF_FLOW_POLICY_ID,
                "pit_confidence": "conservative_session_lag_fail_closed",
                "provider_available_at_date": str(row["available_at_date"]),
                "provider_availability_basis": str(row["availability_basis"]),
                "provider_pit_confidence": str(row["pit_confidence"]),
                "training_available_session_date": training_available_session_date,
                "training_availability_policy_id": ETF_FLOW_POLICY_ID,
                "captured_at_utc": str(row["captured_at_utc"]),
                "record_hash": str(row["record_hash"]),
            }
            for row, training_available_session_date in selected
        ]
        provenance = {
            str(row["payload_sha256"]): {
                "source": "massive",
                "dataset": "etf_fund_flows",
                "captured_at_utc": str(row["captured_at_utc"]),
                "raw_relative_path": str(row["raw_relative_path"]),
            }
            for row, _ in selected
        }
        latest = observations[-1] if observations else None
        return {
            "endpoint_id": MASSIVE_ETF_FLOW_ENDPOINT_ID,
            "as_of_date": as_of,
            "pit_filter": ETF_FLOW_PIT_FILTER,
            "availability_policy": etf_flow_policy_manifest(),
            "historical_backfill_is_true_point_in_time": False,
            "latest": latest,
            "observations": observations,
            "raw_provenance": provenance,
        }

    def packets_for_tickers(
        self,
        tickers: Sequence[str],
        as_of_date: str,
        lookback_records: int,
        *,
        chunk_size: int = 400,
    ) -> Dict[str, dict]:
        """Return exact PIT packets for many tickers without fetching all history.

        The effective date must precede the previous observed U.S. session for
        the conservative D+2 gate, while the processed date must precede the
        latest observed session for the next-session publication gate.  Those
        predicates are equivalent to the row-wise availability function at the
        requested as-of date, but allow SQLite to return only ranked revisions.
        """

        if lookback_records < 1:
            raise ValueError("lookback_records must be positive")
        if chunk_size < 1 or chunk_size > 400:
            raise ValueError("chunk_size must be between 1 and 400")
        normalized_tickers = _normalized_tickers(tickers)
        as_of = validate_iso_date(as_of_date)
        result = {
            ticker: self._packet_document(as_of, []) for ticker in normalized_tickers
        }
        if not normalized_tickers:
            return result
        sessions = self._trading_sessions()
        session_offset = bisect_right(sessions, as_of)
        if session_offset < 2:
            return result
        latest_visible_session = sessions[session_offset - 1]
        previous_visible_session = sessions[session_offset - 2]
        selected_by_ticker: Dict[str, List[Tuple[Mapping[str, Any], str]]] = {
            ticker: [] for ticker in normalized_tickers
        }
        direct_source_paths = getattr(self.database, "flow_source_paths", None)
        if callable(direct_source_paths):
            source_rows: List[dict] = []
            for source_rank, source_path in enumerate(direct_source_paths()):
                connection = sqlite3.connect(
                    "file:{}?mode=ro".format(Path(source_path)),
                    uri=True,
                    timeout=120,
                )
                connection.row_factory = sqlite3.Row
                try:
                    for ticker in normalized_tickers:
                        rows = connection.execute(
                            """
                            SELECT v.*,r.payload_sha256,r.raw_relative_path
                            FROM etf_flow_versions v
                            JOIN raw_artifacts r ON r.id=v.raw_artifact_id
                            WHERE v.ticker=?
                              AND v.effective_date<?
                              AND v.processed_date<?
                              AND v.id=(
                                SELECT v2.id
                                FROM etf_flow_versions v2
                                WHERE v2.ticker=v.ticker
                                  AND v2.effective_date=v.effective_date
                                  AND v2.processed_date<?
                                ORDER BY v2.processed_date DESC,
                                         v2.captured_at_utc DESC,v2.id DESC
                                LIMIT 1
                            )
                            ORDER BY v.effective_date DESC
                            LIMIT ?
                            """,
                            (
                                ticker,
                                previous_visible_session,
                                latest_visible_session,
                                latest_visible_session,
                                lookback_records,
                            ),
                        ).fetchall()
                        for row in rows:
                            value = dict(row)
                            value["_source_rank"] = source_rank
                            source_rows.append(value)
                finally:
                    connection.close()
            latest_by_effective: Dict[Tuple[str, str], dict] = {}
            for row in source_rows:
                key = (str(row["ticker"]), str(row["effective_date"]))
                rank = (
                    str(row["processed_date"]),
                    str(row["captured_at_utc"]),
                    int(row["id"]),
                    int(row["_source_rank"]),
                )
                current = latest_by_effective.get(key)
                current_rank = (
                    (
                        str(current["processed_date"]),
                        str(current["captured_at_utc"]),
                        int(current["id"]),
                        int(current["_source_rank"]),
                    )
                    if current
                    else None
                )
                if current_rank is None or rank > current_rank:
                    latest_by_effective[key] = row
            rows_by_ticker: Dict[str, List[dict]] = {
                ticker: [] for ticker in normalized_tickers
            }
            for row in latest_by_effective.values():
                rows_by_ticker[str(row["ticker"])].append(row)
            direct_rows = []
            for ticker in normalized_tickers:
                ordered = sorted(
                    rows_by_ticker[ticker],
                    key=lambda row: (
                        str(row["effective_date"]),
                        str(row["processed_date"]),
                        str(row["captured_at_utc"]),
                    ),
                )
                direct_rows.extend(ordered[-lookback_records:])
            for row in direct_rows:
                available = derive_etf_flow_available_session(
                    row["effective_date"], row["processed_date"], sessions
                )
                if available is None or available > as_of:
                    raise ValueError(
                        "bulk ETF-flow visibility predicate admitted a future row"
                    )
                selected_by_ticker[str(row["ticker"])].append((row, available))
        else:
            with self.database.connect() as connection:
                for offset in range(0, len(normalized_tickers), chunk_size):
                    chunk = normalized_tickers[offset : offset + chunk_size]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = connection.execute(
                    """
                    WITH revision_ranked AS (
                        SELECT v.*, r.payload_sha256, r.raw_relative_path,
                               ROW_NUMBER() OVER (
                                   PARTITION BY v.ticker, v.effective_date
                                   ORDER BY v.processed_date DESC,
                                            v.captured_at_utc DESC, v.id DESC
                               ) AS revision_rank
                        FROM etf_flow_versions v
                        JOIN raw_artifacts r ON r.id=v.raw_artifact_id
                        WHERE v.ticker IN ({})
                          AND v.effective_date < ?
                          AND v.processed_date < ?
                    ), effective_ranked AS (
                        SELECT revision_ranked.*,
                               ROW_NUMBER() OVER (
                                   PARTITION BY ticker ORDER BY effective_date DESC
                               ) AS effective_rank
                        FROM revision_ranked
                        WHERE revision_rank=1
                    )
                    SELECT * FROM effective_ranked
                    WHERE effective_rank<=?
                    ORDER BY ticker, effective_date
                    """.format(placeholders),
                    [
                        *chunk,
                        previous_visible_session,
                        latest_visible_session,
                        lookback_records,
                    ],
                    ).fetchall()
                    for row in rows:
                        available = derive_etf_flow_available_session(
                            row["effective_date"], row["processed_date"], sessions
                        )
                        if available is None or available > as_of:
                            raise ValueError(
                                "bulk ETF-flow visibility predicate admitted a future row"
                            )
                        selected_by_ticker[str(row["ticker"])].append((row, available))
        for ticker, selected in selected_by_ticker.items():
            result[ticker] = self._packet_document(as_of, selected)
        return result

    def packet_for_ticker(
        self, ticker: str, as_of_date: str, lookback_records: int
    ) -> dict:
        """Return only ETF-flow versions safely visible at an as-of session."""

        normalized_ticker = normalize_symbol(ticker)
        return self.packets_for_tickers(
            [normalized_ticker], as_of_date, lookback_records
        )[normalized_ticker]

    def verify(self) -> dict:
        errors: List[dict] = []
        warnings: List[dict] = []
        with self.database.connect() as connection:
            counts = {
                table: int(
                    connection.execute("SELECT COUNT(*) AS count FROM " + table).fetchone()[
                        "count"
                    ]
                )
                for table in (
                    "etf_flow_versions",
                    "etf_flow_observations",
                    "etf_flow_runs",
                    "etf_flow_run_pages",
                )
            }
            orphan_rows = connection.execute(
                """
                SELECT v.id FROM etf_flow_versions v
                LEFT JOIN raw_artifacts r ON r.id=v.raw_artifact_id
                LEFT JOIN capture_events e ON e.id=v.capture_event_id
                WHERE r.id IS NULL OR e.id IS NULL
                ORDER BY v.id
                """
            ).fetchall()
            if orphan_rows:
                errors.append(
                    {
                        "error": "etf_flow_provenance_missing",
                        "version_ids": [int(row["id"]) for row in orphan_rows],
                    }
                )
            projection_drift = connection.execute(
                """
                SELECT o.provider, o.ticker, o.effective_date
                FROM etf_flow_observations o
                JOIN etf_flow_versions v ON v.id=o.version_id
                WHERE o.record_hash<>v.record_hash
                   OR o.processed_date<>v.processed_date
                   OR o.capture_event_id<>v.capture_event_id
                """
            ).fetchall()
            if projection_drift:
                errors.append(
                    {
                        "error": "etf_flow_projection_drift",
                        "count": len(projection_drift),
                    }
                )
            stale_projection = connection.execute(
                """
                SELECT o.provider, o.ticker, o.effective_date
                FROM etf_flow_observations o
                WHERE EXISTS (
                    SELECT 1 FROM etf_flow_versions v
                    WHERE v.provider=o.provider
                      AND v.ticker=o.ticker
                      AND v.effective_date=o.effective_date
                      AND (
                            v.processed_date>o.processed_date
                         OR (v.processed_date=o.processed_date
                             AND v.captured_at_utc>o.captured_at_utc)
                      )
                )
                """
            ).fetchall()
            if stale_projection:
                errors.append(
                    {
                        "error": "etf_flow_projection_not_latest",
                        "count": len(stale_projection),
                    }
                )
            latest_daily = connection.execute(
                """
                SELECT * FROM etf_flow_runs
                WHERE job_type='capture_etf_flows' AND status='complete'
                ORDER BY completed_at_utc DESC LIMIT 1
                """
            ).fetchone()
            if latest_daily and latest_daily["freshness_status"] in _STALE_STATUSES:
                errors.append(
                    {
                        "error": "etf_flow_freshness_gate",
                        "run_id": str(latest_daily["run_id"]),
                        "status": str(latest_daily["freshness_status"]),
                        "max_processed_date": latest_daily["max_processed_date"],
                    }
                )
            elif latest_daily and latest_daily["freshness_status"] == "unchanged_repeated_hash":
                warnings.append(
                    {
                        "warning": "etf_flow_repeated_hash",
                        "run_id": str(latest_daily["run_id"]),
                        "max_processed_date": latest_daily["max_processed_date"],
                    }
                )
            incomplete = connection.execute(
                """
                SELECT run_id, status, last_error FROM etf_flow_runs
                WHERE status IN ('running','failed') ORDER BY started_at_utc
                """
            ).fetchall()
            if incomplete:
                warnings.append(
                    {
                        "warning": "etf_flow_incomplete_resumable_runs",
                        "runs": [
                            {"run_id": str(row["run_id"]), "status": str(row["status"])}
                            for row in incomplete
                        ],
                    }
                )
        return {
            "ok": not errors,
            "counts": counts,
            "errors": errors,
            "warnings": warnings,
            "latest_daily_run": dict(latest_daily) if latest_daily else None,
        }


class EtfFlowLayer:
    """Resumable paginator and public Phase 2 ETF-flow operations."""

    def __init__(
        self,
        database: Database,
        http: HttpCaptureClient,
        api_key: Optional[str],
        *,
        initialize_schema: bool = True,
    ):
        self.store = EtfFlowStore(
            database, initialize_schema=initialize_schema
        )
        self.provider = MassiveEtfFlowProvider(http, api_key)

    @staticmethod
    def _contract(
        *,
        job_type: str,
        start_date: str,
        end_date: str,
        as_of_date: Optional[str],
        tickers: Sequence[str],
        limit: int,
        lookback_days: Optional[int],
        max_lag_days: Optional[int],
    ) -> Tuple[dict, str, str]:
        contract = {
            "provider": "massive",
            "endpoint_id": MASSIVE_ETF_FLOW_ENDPOINT_ID,
            "method": "GET",
            "path": MASSIVE_ETF_FLOW_PATH,
            "processed_date_gte": start_date,
            "processed_date_lte": end_date,
            "as_of_date": as_of_date,
            "tickers": list(tickers),
            "limit": limit,
            "sort": MASSIVE_ETF_FLOW_SORT,
            "lookback_days": lookback_days,
            "max_lag_days": max_lag_days,
            "historical_filters_supported": True,
            "authentication": "Authorization_Bearer_header",
        }
        contract_hash = sha256_bytes(canonical_json(contract).encode("utf-8"))
        series = {
            "job_type": job_type,
            "provider": "massive",
            "endpoint_id": MASSIVE_ETF_FLOW_ENDPOINT_ID,
            "tickers": list(tickers),
            "limit": limit,
            "sort": MASSIVE_ETF_FLOW_SORT,
            "lookback_days": lookback_days,
        }
        series_key = sha256_bytes(canonical_json(series).encode("utf-8"))
        return contract, contract_hash, series_key

    def _capture_window(
        self,
        *,
        job_type: str,
        start_date: str,
        end_date: str,
        as_of_date: Optional[str],
        tickers: Optional[Sequence[str]],
        limit: int,
        lookback_days: Optional[int],
        max_lag_days: Optional[int],
        resume: bool,
        historical: bool,
    ) -> dict:
        start = validate_iso_date(start_date)
        end = validate_iso_date(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")
        if limit < 1 or limit > MASSIVE_ETF_FLOW_MAX_LIMIT:
            raise ValueError("limit must be between 1 and 5000")
        normalized_tickers = _normalized_tickers(tickers)
        contract, contract_hash, series_key = self._contract(
            job_type=job_type,
            start_date=start,
            end_date=end,
            as_of_date=as_of_date,
            tickers=normalized_tickers,
            limit=limit,
            lookback_days=lookback_days,
            max_lag_days=max_lag_days,
        )
        run, resumed = self.store.start_or_resume(
            contract_hash=contract_hash,
            series_key=series_key,
            job_type=job_type,
            contract=contract,
            max_lag_days=max_lag_days,
            resume=resume,
        )
        run_id = str(run["run_id"])
        page_number = int(run["page_count"]) + 1
        next_url = run["resume_next_url"]
        first_params: Dict[str, Any] = {
            "processed_date.gte": start,
            "processed_date.lte": end,
            "limit": limit,
            "sort": MASSIVE_ETF_FLOW_SORT,
        }
        if normalized_tickers:
            first_params["composite_ticker.any_of"] = ",".join(normalized_tickers)
        request_url = str(next_url) if next_url else MASSIVE_ETF_FLOW_URL
        params = {} if next_url else first_params
        seen_urls = set(self.store.run_request_urls(run_id))

        try:
            # A process can stop after atomically saving the terminal page but
            # before finalizing the run.  In that state page_count is nonzero
            # and resume_next_url is null; finalize without requesting page 1
            # again.
            terminal_page_already_saved = (
                resumed and int(run["page_count"]) > 0 and not run["resume_next_url"]
            )
            if not terminal_page_already_saved:
                while True:
                    safe_request_url = _safe_next_url(request_url)
                    if not safe_request_url:
                        raise PayloadValidationError("Massive ETF flow pagination URL is empty")
                    if safe_request_url in seen_urls:
                        raise PayloadValidationError(
                            "Massive ETF flow pagination cursor loop detected"
                        )
                    partition_key = "contract={}_page={:06d}".format(
                        contract_hash[:24], page_number
                    )
                    page = self.provider.capture_page(
                        url=safe_request_url,
                        params=params,
                        partition_key=partition_key,
                        page_number=page_number,
                        contract_hash=contract_hash,
                    )
                    self.store.ingest_page(
                        run_id=run_id,
                        page_number=page_number,
                        request_url=safe_request_url,
                        page=page,
                    )
                    seen_urls.add(safe_request_url)
                    if not page.next_url:
                        break
                    request_url = page.next_url
                    params = {}
                    page_number += 1
            completed = self.store.finalize(
                run_id,
                as_of_date=as_of_date,
                max_lag_days=max_lag_days,
                historical=historical,
            )
        except Exception as error:
            self.store.mark_failed(run_id, error)
            raise

        result = {
            "ok": True,
            "provider": "massive",
            "endpoint_id": MASSIVE_ETF_FLOW_ENDPOINT_ID,
            "endpoint": MASSIVE_ETF_FLOW_PATH,
            "historical_filters_supported": True,
            "run_id": run_id,
            "resumed": resumed,
            "processed_date_gte": start,
            "processed_date_lte": end,
            "tickers": normalized_tickers,
            "page_count": int(completed["page_count"]),
            "record_count": int(completed["record_count"]),
            "invalid_row_count": int(completed["invalid_row_count"]),
            "min_processed_date": completed["min_processed_date"],
            "max_processed_date": completed["max_processed_date"],
            "min_effective_date": completed["min_effective_date"],
            "max_effective_date": completed["max_effective_date"],
            "payload_set_sha256": completed["payload_set_sha256"],
            "normalized_set_sha256": completed["normalized_set_sha256"],
            "repeated_payload_hash": bool(completed["repeated_payload_hash"]),
            "repeated_normalized_hash": bool(completed["repeated_normalized_hash"]),
            "freshness_status": str(completed["freshness_status"]),
            "authentication": "Authorization_Bearer_header",
        }
        return result

    def capture_as_of(
        self,
        as_of_date: str,
        *,
        lookback_days: int = 7,
        tickers: Optional[Sequence[str]] = None,
        limit: int = MASSIVE_ETF_FLOW_MAX_LIMIT,
        max_lag_days: int = 4,
        resume: bool = True,
        strict_freshness: bool = False,
    ) -> dict:
        as_of = validate_iso_date(as_of_date)
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if max_lag_days < 0:
            raise ValueError("max_lag_days must be >= 0")
        start = (date.fromisoformat(as_of) - timedelta(days=lookback_days - 1)).isoformat()
        result = self._capture_window(
            job_type="capture_etf_flows",
            start_date=start,
            end_date=as_of,
            as_of_date=as_of,
            tickers=tickers,
            limit=limit,
            lookback_days=lookback_days,
            max_lag_days=max_lag_days,
            resume=resume,
            historical=False,
        )
        if strict_freshness and result["freshness_status"] in _STALE_STATUSES:
            result["ok"] = False
            result["freshness_gate"] = "failed"
        else:
            result["freshness_gate"] = (
                "warning"
                if result["freshness_status"] in _STALE_STATUSES
                or result["freshness_status"] == "unchanged_repeated_hash"
                else "passed"
            )
        return result

    def backfill(
        self,
        start_date: str,
        end_date: str,
        *,
        tickers: Optional[Sequence[str]] = None,
        limit: int = MASSIVE_ETF_FLOW_MAX_LIMIT,
        resume: bool = True,
    ) -> dict:
        return self._capture_window(
            job_type="backfill_etf_flows",
            start_date=start_date,
            end_date=end_date,
            as_of_date=None,
            tickers=tickers,
            limit=limit,
            lookback_days=None,
            max_lag_days=None,
            resume=resume,
            historical=True,
        )

    def verify(self) -> dict:
        return self.store.verify()

    def packet_for_ticker(
        self, ticker: str, as_of_date: str, lookback_records: int
    ) -> dict:
        return self.store.packet_for_ticker(ticker, as_of_date, lookback_records)

    def packets_for_tickers(
        self, tickers: Sequence[str], as_of_date: str, lookback_records: int
    ) -> Dict[str, dict]:
        return self.store.packets_for_tickers(tickers, as_of_date, lookback_records)
