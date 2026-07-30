"""Source-preserving point-in-time stock split ledger for Oracle.

Massive is the full-market discovery source. FMP is queried only for symbols
discovered in the requested window, and issuer/exchange evidence may provide an
earlier, explicitly documented availability date. Provider rows first observed
today are never backdated into an earlier analysis snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from .providers import CredentialError, PayloadValidationError, normalize_symbol
from .storage import canonical_json, utc_now


MASSIVE_SPLITS_ENDPOINT_ID = "stocks_splits_2"
MASSIVE_SPLITS_URL = "https://api.massive.com/stocks/v1/splits"
FMP_SPLITS_ENDPOINT_ID = "earnings_dividends_splits_stock_split_details"
FMP_SPLITS_URL = "https://financialmodelingprep.com/stable/splits"
SCHEMA_VERSION = "quant.oracle_corporate_actions.v1"
OFFICIAL_SCHEMA_VERSION = "quant.verified_corporate_actions.v1"
_SENSITIVE_QUERY_NAMES = {
    "apikey",
    "api_key",
    "authorization",
    "token",
    "access_token",
    "secret",
}


class CorporateActionStoreError(RuntimeError):
    """A corporate-action source or normalized row is unsafe."""


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _iso_date(value: Any, field: str) -> str:
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except (TypeError, ValueError) as exc:
        raise CorporateActionStoreError(f"invalid {field}: {value!r}") from exc


def _positive(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CorporateActionStoreError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise CorporateActionStoreError(f"{field} must be positive")
    return number


def _action_type(old_shares: float, new_shares: float) -> str:
    if old_shares > new_shares:
        return "reverse_split"
    if new_shares > old_shares:
        return "forward_split"
    raise CorporateActionStoreError("split ratio cannot be one-to-one")


def _safe_next_url(value: Any) -> str | None:
    if value in (None, ""):
        return None
    absolute = urljoin(MASSIVE_SPLITS_URL, str(value).strip())
    parsed = urlsplit(absolute)
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != "api.massive.com"
        or parsed.path.rstrip("/") != "/stocks/v1/splits"
    ):
        raise PayloadValidationError("Massive split next_url is untrusted")
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower().replace("-", "_") not in _SENSITIVE_QUERY_NAMES
    ]
    return urlunsplit(
        ("https", "api.massive.com", parsed.path, urlencode(query), "")
    )


def _base_record(
    *,
    provider: str,
    endpoint_id: str,
    provider_event_id: str,
    symbol: Any,
    effective_date: Any,
    available_date: Any,
    old_shares: Any,
    new_shares: Any,
    raw_artifact_id: int,
    capture_event_id: int,
    source_row_index: int,
    captured_at_utc: str,
    availability_basis: str,
    pit_confidence: str,
    source_type: str,
    source_name: str,
    source_url: str,
    announcement_date: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    old = _positive(old_shares, "old_shares")
    new = _positive(new_shares, "new_shares")
    normalized = {
        "provider": provider,
        "endpoint_id": endpoint_id,
        "provider_event_id": str(provider_event_id),
        "symbol": normalize_symbol(str(symbol)),
        "action_type": _action_type(old, new),
        "effective_date": _iso_date(effective_date, "effective_date"),
        "announcement_date": (
            _iso_date(announcement_date, "announcement_date")
            if announcement_date
            else None
        ),
        "available_date": _iso_date(available_date, "available_date"),
        "availability_basis": availability_basis,
        "pit_confidence": pit_confidence,
        "old_shares": old,
        "new_shares": new,
        "price_factor_for_prior_rows": old / new,
        "volume_factor_for_prior_rows": new / old,
        "source_type": source_type,
        "source_name": source_name,
        "source_url": source_url,
        "raw_artifact_id": int(raw_artifact_id),
        "capture_event_id": int(capture_event_id),
        "source_row_index": int(source_row_index),
        "captured_at_utc": str(captured_at_utc),
        "extra": dict(extra or {}),
    }
    normalized["record_hash"] = _sha(
        {
            key: normalized[key]
            for key in (
                "provider",
                "endpoint_id",
                "provider_event_id",
                "symbol",
                "action_type",
                "effective_date",
                "announcement_date",
                "available_date",
                "availability_basis",
                "old_shares",
                "new_shares",
                "source_type",
                "source_name",
                "source_url",
                "extra",
            )
        }
    )
    return normalized


def normalize_massive_rows(
    rows: Sequence[Any],
    *,
    artifact: Any,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date")
    available = _iso_date(artifact.captured_at_utc, "captured_at_utc")
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            invalid.append({"source_row_index": index, "reason": "row_not_object"})
            continue
        try:
            effective = _iso_date(row.get("execution_date"), "execution_date")
            if effective < start or effective > end:
                continue
            old = _positive(row.get("split_from"), "split_from")
            new = _positive(row.get("split_to"), "split_to")
            provider_id = str(row.get("id") or _sha(row))
            records.append(
                _base_record(
                    provider="massive",
                    endpoint_id=MASSIVE_SPLITS_ENDPOINT_ID,
                    provider_event_id=provider_id,
                    symbol=row.get("ticker"),
                    effective_date=effective,
                    available_date=available,
                    old_shares=old,
                    new_shares=new,
                    raw_artifact_id=artifact.artifact_id,
                    capture_event_id=artifact.capture_event_id,
                    source_row_index=index,
                    captured_at_utc=artifact.captured_at_utc,
                    availability_basis="first_observed_provider_capture_date",
                    pit_confidence="capture_date_only",
                    source_type="structured_provider",
                    source_name="Massive",
                    source_url=(
                        "https://massive.com/docs/rest/stocks/"
                        "corporate-actions/splits"
                    ),
                    extra={
                        key: row[key]
                        for key in sorted(row)
                        if key
                        not in {
                            "id",
                            "ticker",
                            "execution_date",
                            "split_from",
                            "split_to",
                        }
                    },
                )
            )
        except (CorporateActionStoreError, ValueError) as exc:
            invalid.append(
                {
                    "source_row_index": index,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return records, invalid


def normalize_fmp_rows(
    rows: Sequence[Any],
    *,
    artifact: Any,
    symbol: str,
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date")
    available = _iso_date(artifact.captured_at_utc, "captured_at_utc")
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            invalid.append({"source_row_index": index, "reason": "row_not_object"})
            continue
        try:
            effective = _iso_date(row.get("date"), "date")
            if effective < start or effective > end:
                continue
            new = _positive(row.get("numerator"), "numerator")
            old = _positive(row.get("denominator"), "denominator")
            event_symbol = normalize_symbol(str(row.get("symbol") or symbol))
            records.append(
                _base_record(
                    provider="fmp",
                    endpoint_id=FMP_SPLITS_ENDPOINT_ID,
                    provider_event_id=_sha(
                        {
                            "symbol": event_symbol,
                            "date": effective,
                            "numerator": new,
                            "denominator": old,
                        }
                    ),
                    symbol=event_symbol,
                    effective_date=effective,
                    available_date=available,
                    old_shares=old,
                    new_shares=new,
                    raw_artifact_id=artifact.artifact_id,
                    capture_event_id=artifact.capture_event_id,
                    source_row_index=index,
                    captured_at_utc=artifact.captured_at_utc,
                    availability_basis="first_observed_provider_capture_date",
                    pit_confidence="capture_date_only",
                    source_type="structured_provider",
                    source_name="Financial Modeling Prep",
                    source_url=(
                        "https://site.financialmodelingprep.com/"
                        "developer/docs/stable"
                    ),
                    extra={
                        "split_type": row.get("splitType"),
                    },
                )
            )
        except (CorporateActionStoreError, ValueError) as exc:
            invalid.append(
                {
                    "source_row_index": index,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return records, invalid


def normalize_official_events(
    payload: Mapping[str, Any],
    *,
    artifact: Any,
) -> list[dict[str, Any]]:
    if payload.get("schema_version") != OFFICIAL_SCHEMA_VERSION:
        raise CorporateActionStoreError(
            f"official event schema must be {OFFICIAL_SCHEMA_VERSION}"
        )
    records: list[dict[str, Any]] = []
    for index, row in enumerate(payload.get("events") or []):
        if not isinstance(row, Mapping):
            raise CorporateActionStoreError(f"official event {index} is invalid")
        source_type = str(row.get("source_type") or "")
        source_url = str(row.get("source_url") or "")
        if source_type not in {"official_issuer", "official_exchange"}:
            raise CorporateActionStoreError(
                f"official event {index} has invalid source_type"
            )
        if not source_url.startswith("https://"):
            raise CorporateActionStoreError(
                f"official event {index} requires an HTTPS source"
            )
        records.append(
            _base_record(
                provider=source_type,
                endpoint_id="official_corporate_action_notice",
                provider_event_id=_sha(
                    {
                        "symbol": row.get("symbol"),
                        "effective_date": row.get("effective_date"),
                        "source_url": source_url,
                    }
                ),
                symbol=row.get("symbol"),
                effective_date=row.get("effective_date"),
                announcement_date=(
                    row.get("announcement_date") or row.get("available_date")
                ),
                available_date=row.get("available_date"),
                old_shares=row.get("old_shares"),
                new_shares=row.get("new_shares"),
                raw_artifact_id=artifact.artifact_id,
                capture_event_id=artifact.capture_event_id,
                source_row_index=index,
                captured_at_utc=artifact.captured_at_utc,
                availability_basis="official_announcement_date",
                pit_confidence="official_date",
                source_type=source_type,
                source_name=str(row.get("source_name") or ""),
                source_url=source_url,
                extra={
                    key: row[key]
                    for key in sorted(row)
                    if key
                    not in {
                        "symbol",
                        "effective_date",
                        "announcement_date",
                        "available_date",
                        "old_shares",
                        "new_shares",
                        "source_type",
                        "source_name",
                        "source_url",
                    }
                },
            )
        )
    return records


def initialize_corporate_action_schema(database: Any) -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS corporate_action_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT NOT NULL,
        endpoint_id TEXT NOT NULL,
        provider_event_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        action_type TEXT NOT NULL,
        effective_date TEXT NOT NULL,
        announcement_date TEXT,
        available_date TEXT NOT NULL,
        availability_basis TEXT NOT NULL,
        pit_confidence TEXT NOT NULL,
        old_shares REAL NOT NULL,
        new_shares REAL NOT NULL,
        price_factor_for_prior_rows REAL NOT NULL,
        volume_factor_for_prior_rows REAL NOT NULL,
        source_type TEXT NOT NULL,
        source_name TEXT NOT NULL,
        source_url TEXT NOT NULL,
        record_hash TEXT NOT NULL,
        raw_artifact_id INTEGER NOT NULL,
        capture_event_id INTEGER NOT NULL,
        source_row_index INTEGER NOT NULL,
        captured_at_utc TEXT NOT NULL,
        ingested_at_utc TEXT NOT NULL,
        extra_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(provider, capture_event_id, source_row_index),
        FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id),
        FOREIGN KEY(capture_event_id) REFERENCES capture_events(id)
    );
    CREATE TABLE IF NOT EXISTS corporate_actions (
        provider TEXT NOT NULL,
        endpoint_id TEXT NOT NULL,
        provider_event_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        action_type TEXT NOT NULL,
        effective_date TEXT NOT NULL,
        announcement_date TEXT,
        available_date TEXT NOT NULL,
        availability_basis TEXT NOT NULL,
        pit_confidence TEXT NOT NULL,
        old_shares REAL NOT NULL,
        new_shares REAL NOT NULL,
        price_factor_for_prior_rows REAL NOT NULL,
        volume_factor_for_prior_rows REAL NOT NULL,
        source_type TEXT NOT NULL,
        source_name TEXT NOT NULL,
        source_url TEXT NOT NULL,
        record_hash TEXT NOT NULL,
        version_id INTEGER NOT NULL,
        raw_artifact_id INTEGER NOT NULL,
        capture_event_id INTEGER NOT NULL,
        source_row_index INTEGER NOT NULL,
        captured_at_utc TEXT NOT NULL,
        ingested_at_utc TEXT NOT NULL,
        extra_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY(provider, provider_event_id),
        FOREIGN KEY(version_id) REFERENCES corporate_action_versions(id),
        FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id),
        FOREIGN KEY(capture_event_id) REFERENCES capture_events(id)
    );
    CREATE INDEX IF NOT EXISTS idx_corporate_actions_symbol_dates
      ON corporate_actions(symbol,effective_date,available_date);
    CREATE INDEX IF NOT EXISTS idx_corporate_action_versions_symbol_dates
      ON corporate_action_versions(symbol,effective_date,available_date);
    """
    with database.connect() as connection:
        connection.executescript(schema)


class CorporateActionStore:
    """Append-only versions plus earliest-observed provider projection."""

    def __init__(self, database: Any):
        self.database = database
        initialize_corporate_action_schema(database)

    def ingest(self, records: Sequence[Mapping[str, Any]]) -> int:
        inserted = 0
        with self.database.connect() as connection:
            for record in records:
                now = utc_now()
                values = (
                    record["provider"],
                    record["endpoint_id"],
                    record["provider_event_id"],
                    record["symbol"],
                    record["action_type"],
                    record["effective_date"],
                    record.get("announcement_date"),
                    record["available_date"],
                    record["availability_basis"],
                    record["pit_confidence"],
                    record["old_shares"],
                    record["new_shares"],
                    record["price_factor_for_prior_rows"],
                    record["volume_factor_for_prior_rows"],
                    record["source_type"],
                    record["source_name"],
                    record["source_url"],
                    record["record_hash"],
                    record["raw_artifact_id"],
                    record["capture_event_id"],
                    record["source_row_index"],
                    record["captured_at_utc"],
                    now,
                    canonical_json(record.get("extra", {})),
                )
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO corporate_action_versions (
                      provider,endpoint_id,provider_event_id,symbol,action_type,
                      effective_date,announcement_date,available_date,
                      availability_basis,pit_confidence,old_shares,new_shares,
                      price_factor_for_prior_rows,volume_factor_for_prior_rows,
                      source_type,source_name,source_url,record_hash,
                      raw_artifact_id,capture_event_id,source_row_index,
                      captured_at_utc,ingested_at_utc,extra_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    values,
                )
                inserted += int(cursor.rowcount > 0)
                version = connection.execute(
                    """
                    SELECT id FROM corporate_action_versions
                    WHERE provider=? AND capture_event_id=? AND source_row_index=?
                    """,
                    (
                        record["provider"],
                        record["capture_event_id"],
                        record["source_row_index"],
                    ),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO corporate_actions (
                      provider,endpoint_id,provider_event_id,symbol,action_type,
                      effective_date,announcement_date,available_date,
                      availability_basis,pit_confidence,old_shares,new_shares,
                      price_factor_for_prior_rows,volume_factor_for_prior_rows,
                      source_type,source_name,source_url,record_hash,version_id,
                      raw_artifact_id,capture_event_id,source_row_index,
                      captured_at_utc,ingested_at_utc,extra_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(provider,provider_event_id) DO UPDATE SET
                      endpoint_id=excluded.endpoint_id,
                      symbol=excluded.symbol,
                      action_type=excluded.action_type,
                      effective_date=excluded.effective_date,
                      announcement_date=COALESCE(
                        corporate_actions.announcement_date,
                        excluded.announcement_date
                      ),
                      available_date=MIN(
                        corporate_actions.available_date,
                        excluded.available_date
                      ),
                      availability_basis=CASE
                        WHEN excluded.available_date <
                             corporate_actions.available_date
                        THEN excluded.availability_basis
                        ELSE corporate_actions.availability_basis END,
                      pit_confidence=CASE
                        WHEN excluded.available_date <
                             corporate_actions.available_date
                        THEN excluded.pit_confidence
                        ELSE corporate_actions.pit_confidence END,
                      old_shares=excluded.old_shares,
                      new_shares=excluded.new_shares,
                      price_factor_for_prior_rows=
                        excluded.price_factor_for_prior_rows,
                      volume_factor_for_prior_rows=
                        excluded.volume_factor_for_prior_rows,
                      source_type=excluded.source_type,
                      source_name=excluded.source_name,
                      source_url=excluded.source_url,
                      record_hash=excluded.record_hash,
                      version_id=excluded.version_id,
                      raw_artifact_id=excluded.raw_artifact_id,
                      capture_event_id=excluded.capture_event_id,
                      source_row_index=excluded.source_row_index,
                      captured_at_utc=excluded.captured_at_utc,
                      ingested_at_utc=excluded.ingested_at_utc,
                      extra_json=excluded.extra_json
                    """,
                    values[:18]
                    + (
                        int(version["id"]),
                        *values[18:],
                    ),
                )
        return inserted


def _official_capture(pipeline: Any, path: Path) -> tuple[Any, Mapping[str, Any]] | None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return None
    payload = resolved.read_bytes()
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorporateActionStoreError(
            f"official corporate-action ledger is invalid: {resolved}"
        ) from exc
    artifact = pipeline.raw_store.store(
        source="official",
        dataset="verified_corporate_actions",
        partition_key="bootstrap",
        payload=payload,
        request={
            "method": "LOCAL_IMPORT",
            "path": str(resolved),
            "logical_request": {
                "owner": "market_structure_oracle",
                "schema_version": OFFICIAL_SCHEMA_VERSION,
            },
        },
        response={"status_code": 200, "source": "local_verified_file"},
    )
    return artifact, document


def capture_corporate_actions(
    *,
    pipeline: Any,
    start_date: str,
    end_date: str,
    official_ledger_path: Path,
) -> dict[str, Any]:
    """Capture Massive discovery, FMP corroboration, and official notices."""

    if not pipeline.credentials.massive_api_key:
        raise CredentialError("MASSIVE_API_KEY is not configured")
    if not pipeline.credentials.fmp_api_key:
        raise CredentialError("FMP_API_KEY is not configured")
    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date")
    store = CorporateActionStore(pipeline.database)
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    pages = 0
    url: str | None = MASSIVE_SPLITS_URL
    params: dict[str, Any] = {
        "execution_date.gte": start,
        "execution_date.lte": end,
        "limit": 5000,
        "sort": "execution_date.asc",
    }
    while url:
        pages += 1
        if pages > 20:
            raise CorporateActionStoreError("Massive split pagination exceeded 20 pages")
        result = pipeline.http.get_json(
            source="massive",
            dataset="stock_splits",
            partition_key=f"{start}_{end}",
            url=url,
            params=params,
            headers={
                "Authorization": f"Bearer {pipeline.credentials.massive_api_key}"
            },
            logical_request={
                "endpoint_id": MASSIVE_SPLITS_ENDPOINT_ID,
                "start_date": start,
                "end_date": end,
                "page_number": pages,
            },
        )
        document = result.document
        if not isinstance(document, Mapping) or not isinstance(
            document.get("results"), list
        ):
            raise PayloadValidationError(
                "Massive split payload has no results list "
                f"(raw artifact id={result.artifact.artifact_id})"
            )
        normalized, rejected = normalize_massive_rows(
            document.get("results") or [],
            artifact=result.artifact,
            start_date=start,
            end_date=end,
        )
        records.extend(normalized)
        invalid.extend(rejected)
        url = _safe_next_url(document.get("next_url"))
        params = {}

    discovered_symbols = sorted({row["symbol"] for row in records})
    for symbol in discovered_symbols:
        result = pipeline.http.get_json(
            source="fmp",
            dataset="stock_splits",
            partition_key=symbol,
            url=FMP_SPLITS_URL,
            params={"symbol": symbol},
            headers={"apikey": pipeline.credentials.fmp_api_key},
            logical_request={
                "endpoint_id": FMP_SPLITS_ENDPOINT_ID,
                "symbol": symbol,
                "start_date": start,
                "end_date": end,
            },
        )
        if not isinstance(result.document, list):
            raise PayloadValidationError(
                "FMP split payload is not a list "
                f"(raw artifact id={result.artifact.artifact_id})"
            )
        normalized, rejected = normalize_fmp_rows(
            result.document,
            artifact=result.artifact,
            symbol=symbol,
            start_date=start,
            end_date=end,
        )
        records.extend(normalized)
        invalid.extend(rejected)

    official_capture = _official_capture(pipeline, official_ledger_path)
    if official_capture:
        artifact, document = official_capture
        records.extend(normalize_official_events(document, artifact=artifact))

    inserted_versions = store.ingest(records)
    summary = corporate_action_summary(pipeline.database.db_path, end)
    discovered_keys = {
        (
            row["symbol"],
            row["effective_date"],
            row["old_shares"],
            row["new_shares"],
        )
        for row in records
        if row["provider"] == "massive"
    }
    fmp_keys = {
        (
            row["symbol"],
            row["effective_date"],
            row["old_shares"],
            row["new_shares"],
        )
        for row in records
        if row["provider"] == "fmp"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "start_date": start,
        "end_date": end,
        "massive_pages": pages,
        "massive_discovered_event_count": len(discovered_keys),
        "fmp_corroborated_event_count": len(discovered_keys & fmp_keys),
        "uncorroborated_massive_event_count": len(discovered_keys - fmp_keys),
        "official_event_count": sum(
            row["source_type"] in {"official_issuer", "official_exchange"}
            for row in records
        ),
        "normalized_record_count": len(records),
        "invalid_row_count": len(invalid),
        "inserted_version_count": inserted_versions,
        "invalid_rows": invalid[:100],
        **summary,
        "point_in_time_gate": (
            "effective_date<=as_of and available_date<=as_of; "
            "provider rows use first observed capture date"
        ),
    }


def corporate_action_summary(database_path: Path, as_of_date: str) -> dict[str, Any]:
    as_of = _iso_date(as_of_date, "as_of_date")
    with sqlite3.connect(database_path) as connection:
        rows = [
            list(row)
            for row in connection.execute(
                """
                SELECT provider,provider_event_id,symbol,action_type,
                       effective_date,announcement_date,available_date,
                       old_shares,new_shares,record_hash,source_type,source_url
                FROM corporate_actions
                WHERE effective_date<=? AND available_date<=?
                ORDER BY symbol,effective_date,provider,provider_event_id
                """,
                (as_of, as_of),
            )
        ]
        total = int(
            connection.execute(
                "SELECT COUNT(*) FROM corporate_actions"
            ).fetchone()[0]
        )
        versions = int(
            connection.execute(
                "SELECT COUNT(*) FROM corporate_action_versions"
            ).fetchone()[0]
        )
    return {
        "projection_count": total,
        "version_count": versions,
        "visible_record_count": len(rows),
        "visible_projection_sha256": _sha(rows),
    }
