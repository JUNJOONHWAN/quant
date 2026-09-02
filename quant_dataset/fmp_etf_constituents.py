"""Historical FMP ETF constituent snapshots with point-in-time joins."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .providers import (
    CredentialError,
    HttpCaptureClient,
    PayloadValidationError,
    normalize_symbol,
    validate_iso_date,
)
from .point_in_time import (
    ETF_CONSTITUENT_POLICY_ID,
    US_EQUITY_SESSION_SQL,
    derive_constituent_available_session,
    etf_constituent_policy_manifest,
    normalize_trading_sessions,
)
from .storage import Database, RawArtifact, canonical_json, sha256_bytes, utc_now


FMP_ETF_HOLDING_DATES_URL = (
    "https://financialmodelingprep.com/api/v4/etf-holdings/portfolio-date"
)
FMP_ETF_HOLDINGS_URL = "https://financialmodelingprep.com/api/v4/etf-holdings"


@dataclass(frozen=True)
class ConstituentCapture:
    artifact: RawArtifact
    records: List[dict]
    invalid_rows: List[dict]
    available_date: str


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> Optional[str]:
    rendered = str(value or "").strip()
    return rendered or None


def _date_prefix(value: Any) -> Optional[str]:
    rendered = str(value or "").strip()
    if len(rendered) < 10:
        return None
    try:
        return validate_iso_date(rendered[:10])
    except ValueError:
        return None


def _constituent_key(row: Mapping[str, Any], ticker: Optional[str]) -> str:
    for prefix, value in (
        ("isin", row.get("isin")),
        ("cusip", row.get("cusip")),
        ("ticker", ticker),
        ("lei", row.get("lei")),
    ):
        rendered = _text(value)
        if rendered:
            return "{}:{}".format(prefix, rendered.upper())
    fallback = {
        "name": _text(row.get("name")),
        "title": _text(row.get("title")),
        "cik": _text(row.get("cik")),
    }
    return "rowhash:{}".format(
        sha256_bytes(canonical_json(fallback).encode("utf-8"))
    )


def read_etf_symbols_from_universe(path: Path) -> List[str]:
    result = []
    for line_number, line in enumerate(
        Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(
                "invalid universe JSONL at line {}".format(line_number)
            ) from error
        if not isinstance(row, dict) or row.get("is_etf") is not True:
            continue
        symbol = normalize_symbol(row.get("symbol"))
        if symbol not in result:
            result.append(symbol)
    return sorted(result)


def universe_jsonl_contract(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": sha256_bytes(payload),
        "etf_symbol_count": len(read_etf_symbols_from_universe(resolved)),
    }


def shard_symbols(
    symbols: Sequence[str], shard_count: int, shard_index: int
) -> List[str]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    normalized = sorted({normalize_symbol(symbol) for symbol in symbols})
    return normalized[shard_index::shard_count]


class FmpEtfConstituentProvider:
    def __init__(self, http: HttpCaptureClient, api_key: Optional[str]):
        self.http = http
        self.api_key = api_key

    def capture_dates(self, etf_ticker: str) -> Tuple[List[str], RawArtifact]:
        if not self.api_key:
            raise CredentialError("FMP_API_KEY is not configured")
        etf = normalize_symbol(etf_ticker)
        result = self.http.get_json(
            source="fmp",
            dataset="etf_holding_dates_v4",
            partition_key=etf,
            url=FMP_ETF_HOLDING_DATES_URL,
            params={"symbol": etf},
            headers={"apikey": self.api_key},
            logical_request={
                "endpoint_contract": "fmp_v4_etf_holding_dates",
                "etf_ticker": etf,
            },
        )
        if not isinstance(result.document, list):
            raise PayloadValidationError(
                "FMP ETF holding dates payload is not a list "
                "(raw artifact id={})".format(result.artifact.artifact_id)
            )
        dates = []
        for row in result.document:
            candidate = row.get("date") if isinstance(row, dict) else row
            parsed = _date_prefix(candidate)
            if parsed and parsed not in dates:
                dates.append(parsed)
        return sorted(dates), result.artifact

    def capture_snapshot(self, etf_ticker: str, effective_date: str) -> ConstituentCapture:
        if not self.api_key:
            raise CredentialError("FMP_API_KEY is not configured")
        etf = normalize_symbol(etf_ticker)
        requested_date = validate_iso_date(effective_date)
        result = self.http.get_json(
            source="fmp",
            dataset="etf_holdings_historical_v4",
            partition_key="{}_{}".format(etf, requested_date),
            url=FMP_ETF_HOLDINGS_URL,
            params={"symbol": etf, "date": requested_date},
            headers={"apikey": self.api_key},
            logical_request={
                "endpoint_contract": "fmp_v4_historical_etf_holdings",
                "etf_ticker": etf,
                "effective_date": requested_date,
            },
        )
        if not isinstance(result.document, list):
            raise PayloadValidationError(
                "FMP historical ETF holdings payload is not a list "
                "(raw artifact id={})".format(result.artifact.artifact_id)
            )
        records = []
        invalid_rows = []
        available_dates = []
        constituent_key_occurrences: Counter[str] = Counter()
        recognized = {
            "acceptanceTime",
            "assetCat",
            "balance",
            "cik",
            "cur_cd",
            "cusip",
            "date",
            "fairValLevel",
            "invCountry",
            "isin",
            "lei",
            "name",
            "pctVal",
            "symbol",
            "title",
            "units",
            "valUsd",
        }
        for index, row in enumerate(result.document):
            if not isinstance(row, dict):
                invalid_rows.append({"source_row_index": index, "reason": "row_not_object"})
                continue
            row_date = _date_prefix(row.get("date")) or requested_date
            if row_date != requested_date:
                invalid_rows.append(
                    {
                        "source_row_index": index,
                        "reason": "effective_date_mismatch",
                        "row_date": row_date,
                    }
                )
                continue
            ticker = None
            if _text(row.get("symbol")):
                try:
                    ticker = normalize_symbol(row.get("symbol"))
                except ValueError:
                    ticker = None
            acceptance_time = _text(row.get("acceptanceTime"))
            available_date = _date_prefix(acceptance_time) or requested_date
            available_dates.append(available_date)
            extra = {key: row[key] for key in sorted(row) if key not in recognized}
            base_constituent_key = _constituent_key(row, ticker)
            constituent_key_occurrences[base_constituent_key] += 1
            position_ordinal = constituent_key_occurrences[base_constituent_key]
            constituent_key = base_constituent_key
            if position_ordinal > 1:
                constituent_key = "{}#position:{:04d}".format(
                    base_constituent_key, position_ordinal
                )
            extra["base_constituent_key"] = base_constituent_key
            extra["position_ordinal"] = position_ordinal
            record = {
                "provider": "fmp",
                "etf_ticker": etf,
                "constituent_key": constituent_key,
                "constituent_ticker": ticker,
                "constituent_name": _text(row.get("name") or row.get("title")),
                "isin": _text(row.get("isin")),
                "cusip": _text(row.get("cusip")),
                "cik": _text(row.get("cik")),
                "lei": _text(row.get("lei")),
                "effective_date": requested_date,
                "acceptance_time": acceptance_time,
                "available_date": available_date,
                "availability_basis": (
                    "fmp_disclosure_acceptance_time"
                    if acceptance_time
                    else "effective_date_no_acceptance_time"
                ),
                "pit_confidence": (
                    "date_exact_timestamp_timezone_unverified"
                    if acceptance_time
                    else "date_only_approximate"
                ),
                "balance": _number(row.get("balance")),
                "value_usd": _number(row.get("valUsd")),
                "weight_percent": _number(row.get("pctVal")),
                "currency": _text(row.get("cur_cd")),
                "units": _text(row.get("units")),
                "asset_category": _text(row.get("assetCat")),
                "investment_country": _text(row.get("invCountry")),
                "raw_artifact_id": result.artifact.artifact_id,
                "capture_event_id": result.artifact.capture_event_id,
                "source_row_index": index,
                "captured_at_utc": result.artifact.captured_at_utc,
                "extra": extra,
            }
            records.append(record)
        snapshot_available_date = max(available_dates) if available_dates else requested_date
        return ConstituentCapture(
            artifact=result.artifact,
            records=records,
            invalid_rows=invalid_rows,
            available_date=snapshot_available_date,
        )


class FmpEtfConstituentStore:
    def __init__(self, database: Database, *, initialize_schema: bool = True):
        self.database = database
        self._trading_sessions_cache: Optional[Tuple[str, ...]] = None
        if initialize_schema:
            self.initialize()

    def _trading_sessions(self) -> Tuple[str, ...]:
        if self._trading_sessions_cache is None:
            with self.database.connect() as connection:
                rows = connection.execute(US_EQUITY_SESSION_SQL).fetchall()
            self._trading_sessions_cache = normalize_trading_sessions(
                row["trade_date"] for row in rows
            )
        return self._trading_sessions_cache

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS etf_constituent_available_dates (
            provider TEXT NOT NULL,
            etf_ticker TEXT NOT NULL,
            effective_date TEXT NOT NULL,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            captured_at_utc TEXT NOT NULL,
            PRIMARY KEY(provider, etf_ticker, effective_date)
        );

        CREATE TABLE IF NOT EXISTS etf_constituent_snapshots (
            provider TEXT NOT NULL,
            etf_ticker TEXT NOT NULL,
            effective_date TEXT NOT NULL,
            available_date TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            invalid_row_count INTEGER NOT NULL,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            captured_at_utc TEXT NOT NULL,
            PRIMARY KEY(provider, etf_ticker, effective_date)
        );

        CREATE TABLE IF NOT EXISTS etf_constituent_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            etf_ticker TEXT NOT NULL,
            constituent_key TEXT NOT NULL,
            constituent_ticker TEXT,
            constituent_name TEXT,
            isin TEXT,
            cusip TEXT,
            cik TEXT,
            lei TEXT,
            effective_date TEXT NOT NULL,
            acceptance_time TEXT,
            available_date TEXT NOT NULL,
            availability_basis TEXT NOT NULL,
            pit_confidence TEXT NOT NULL,
            balance REAL,
            value_usd REAL,
            weight_percent REAL,
            currency TEXT,
            units TEXT,
            asset_category TEXT,
            investment_country TEXT,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            source_row_index INTEGER NOT NULL,
            captured_at_utc TEXT NOT NULL,
            extra_json TEXT NOT NULL,
            UNIQUE(provider, etf_ticker, constituent_key, effective_date, raw_artifact_id)
        );

        CREATE TABLE IF NOT EXISTS etf_constituent_observations (
            provider TEXT NOT NULL,
            etf_ticker TEXT NOT NULL,
            constituent_key TEXT NOT NULL,
            constituent_ticker TEXT,
            constituent_name TEXT,
            isin TEXT,
            cusip TEXT,
            cik TEXT,
            lei TEXT,
            effective_date TEXT NOT NULL,
            acceptance_time TEXT,
            available_date TEXT NOT NULL,
            availability_basis TEXT NOT NULL,
            pit_confidence TEXT NOT NULL,
            balance REAL,
            value_usd REAL,
            weight_percent REAL,
            currency TEXT,
            units TEXT,
            asset_category TEXT,
            investment_country TEXT,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            source_row_index INTEGER NOT NULL,
            captured_at_utc TEXT NOT NULL,
            extra_json TEXT NOT NULL,
            PRIMARY KEY(provider, etf_ticker, constituent_key, effective_date)
        );

        CREATE INDEX IF NOT EXISTS idx_etf_constituent_etf_date
            ON etf_constituent_observations(etf_ticker, effective_date);
        CREATE INDEX IF NOT EXISTS idx_etf_constituent_symbol_date
            ON etf_constituent_observations(constituent_ticker, effective_date);
        CREATE INDEX IF NOT EXISTS idx_etf_constituent_snapshot_available
            ON etf_constituent_snapshots(etf_ticker, effective_date, available_date);
        """
        with self.database.connect() as connection:
            connection.executescript(schema)

    def ingest_dates(
        self, etf_ticker: str, dates: Sequence[str], artifact: RawArtifact
    ) -> None:
        with self.database.connect() as connection:
            connection.executemany(
                """
                INSERT INTO etf_constituent_available_dates (
                    provider, etf_ticker, effective_date, raw_artifact_id,
                    capture_event_id, captured_at_utc
                ) VALUES ('fmp', ?, ?, ?, ?, ?)
                ON CONFLICT(provider, etf_ticker, effective_date) DO UPDATE SET
                    raw_artifact_id=excluded.raw_artifact_id,
                    capture_event_id=excluded.capture_event_id,
                    captured_at_utc=excluded.captured_at_utc
                """,
                [
                    (
                        etf_ticker,
                        effective_date,
                        artifact.artifact_id,
                        artifact.capture_event_id,
                        artifact.captured_at_utc,
                    )
                    for effective_date in dates
                ],
            )

    def dates_for_etf(self, etf_ticker: str) -> List[str]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT effective_date FROM etf_constituent_available_dates
                WHERE provider='fmp' AND etf_ticker=?
                ORDER BY effective_date
                """,
                (normalize_symbol(etf_ticker),),
            ).fetchall()
        return [str(row["effective_date"]) for row in rows]

    @staticmethod
    def _values(record: Mapping[str, Any]) -> tuple:
        return (
            record["provider"],
            record["etf_ticker"],
            record["constituent_key"],
            record.get("constituent_ticker"),
            record.get("constituent_name"),
            record.get("isin"),
            record.get("cusip"),
            record.get("cik"),
            record.get("lei"),
            record["effective_date"],
            record.get("acceptance_time"),
            record["available_date"],
            record["availability_basis"],
            record["pit_confidence"],
            record.get("balance"),
            record.get("value_usd"),
            record.get("weight_percent"),
            record.get("currency"),
            record.get("units"),
            record.get("asset_category"),
            record.get("investment_country"),
            record["raw_artifact_id"],
            record["capture_event_id"],
            record["source_row_index"],
            record["captured_at_utc"],
            canonical_json(record.get("extra", {})),
        )

    def ingest_snapshot(
        self, etf_ticker: str, effective_date: str, capture: ConstituentCapture
    ) -> None:
        columns = """
            provider, etf_ticker, constituent_key, constituent_ticker,
            constituent_name, isin, cusip, cik, lei, effective_date,
            acceptance_time, available_date, availability_basis, pit_confidence,
            balance, value_usd, weight_percent, currency, units, asset_category,
            investment_country, raw_artifact_id, capture_event_id,
            source_row_index, captured_at_utc, extra_json
        """
        placeholders = ",".join("?" for _ in range(26))
        version_sql = "INSERT OR IGNORE INTO etf_constituent_versions ({}) VALUES ({})".format(
            columns, placeholders
        )
        observation_sql = "INSERT INTO etf_constituent_observations ({}) VALUES ({})".format(
            columns, placeholders
        )
        values = [self._values(record) for record in capture.records]
        with self.database.connect() as connection:
            connection.executemany(version_sql, values)
            connection.execute(
                "DELETE FROM etf_constituent_observations "
                "WHERE provider='fmp' AND etf_ticker=? AND effective_date=?",
                (etf_ticker, effective_date),
            )
            connection.executemany(observation_sql, values)
            connection.execute(
                """
                INSERT INTO etf_constituent_snapshots (
                    provider, etf_ticker, effective_date, available_date,
                    row_count, invalid_row_count, raw_artifact_id,
                    capture_event_id, captured_at_utc
                ) VALUES ('fmp', ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, etf_ticker, effective_date) DO UPDATE SET
                    available_date=excluded.available_date,
                    row_count=excluded.row_count,
                    invalid_row_count=excluded.invalid_row_count,
                    raw_artifact_id=excluded.raw_artifact_id,
                    capture_event_id=excluded.capture_event_id,
                    captured_at_utc=excluded.captured_at_utc
                """,
                (
                    etf_ticker,
                    effective_date,
                    capture.available_date,
                    len(capture.records),
                    len(capture.invalid_rows),
                    capture.artifact.artifact_id,
                    capture.artifact.capture_event_id,
                    capture.artifact.captured_at_utc,
                ),
            )

    @staticmethod
    def _packet_row(
        row: Mapping[str, Any], training_available_session_date: str
    ) -> dict:
        return {
            "etf_ticker": str(row["etf_ticker"]),
            "constituent_key": str(row["constituent_key"]),
            "constituent_ticker": row["constituent_ticker"],
            "constituent_name": row["constituent_name"],
            "isin": row["isin"],
            "cusip": row["cusip"],
            "effective_date": str(row["effective_date"]),
            "available_date": training_available_session_date,
            "availability_basis": ETF_CONSTITUENT_POLICY_ID,
            "pit_confidence": "conservative_next_session_fail_closed",
            "provider_available_date": str(row["available_date"]),
            "provider_availability_basis": str(row["availability_basis"]),
            "provider_pit_confidence": str(row["pit_confidence"]),
            "training_available_session_date": training_available_session_date,
            "training_availability_policy_id": ETF_CONSTITUENT_POLICY_ID,
            "balance": row["balance"],
            "value_usd": row["value_usd"],
            "weight_percent": row["weight_percent"],
            "currency": row["currency"],
            "units": row["units"],
            "asset_category": row["asset_category"],
            "investment_country": row["investment_country"],
        }

    @staticmethod
    def _availability_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        """Use the persisted business key; temp SQLite views have no rowid."""

        return (
            str(row["etf_ticker"]),
            str(row["constituent_key"]),
            str(row["effective_date"]),
        )

    def packet_for_symbol(self, symbol: str, as_of_date: str) -> dict:
        ticker = normalize_symbol(symbol)
        as_of = validate_iso_date(as_of_date)
        sessions = self._trading_sessions()
        connector = getattr(
            self.database,
            "connect_constituents",
            self.database.connect,
        )
        with connector() as connection:
            snapshot_candidates = connection.execute(
                """
                SELECT * FROM etf_constituent_snapshots
                WHERE provider='fmp' AND etf_ticker=?
                  AND effective_date<=? AND available_date<=?
                ORDER BY effective_date DESC
                """,
                (ticker, as_of, as_of),
            ).fetchall()
            etf_snapshot = None
            snapshot_training_available = None
            for candidate in snapshot_candidates:
                derived = derive_constituent_available_session(
                    candidate["available_date"], sessions
                )
                if derived is not None and derived <= as_of:
                    etf_snapshot = candidate
                    snapshot_training_available = derived
                    break
            holdings = []
            holding_availability = {}
            if etf_snapshot:
                candidates = connection.execute(
                        """
                        SELECT * FROM etf_constituent_observations
                        WHERE provider='fmp' AND etf_ticker=? AND effective_date=?
                          AND available_date<=?
                        ORDER BY weight_percent DESC, constituent_key
                        """,
                        (ticker, etf_snapshot["effective_date"], as_of),
                    ).fetchall()
                for row in candidates:
                    derived = derive_constituent_available_session(
                        row["available_date"], sessions
                    )
                    if derived is not None and derived <= as_of:
                        holdings.append(row)
                        holding_availability[self._availability_key(row)] = derived
            membership_candidates = connection.execute(
                """
                SELECT * FROM etf_constituent_observations
                WHERE provider='fmp' AND constituent_ticker=?
                  AND effective_date<=? AND available_date<=?
                ORDER BY etf_ticker, effective_date DESC, constituent_key
                """,
                (ticker, as_of, as_of),
            ).fetchall()
            eligible_memberships = []
            membership_availability = {}
            for row in membership_candidates:
                derived = derive_constituent_available_session(
                    row["available_date"], sessions
                )
                if derived is not None and derived <= as_of:
                    eligible_memberships.append(row)
                    membership_availability[self._availability_key(row)] = derived
            latest_effective_by_etf = {}
            for row in eligible_memberships:
                latest_effective_by_etf.setdefault(
                    str(row["etf_ticker"]), str(row["effective_date"])
                )
            memberships = [
                row
                for row in eligible_memberships
                if str(row["effective_date"])
                == latest_effective_by_etf[str(row["etf_ticker"])]
            ]
            composition = {}
            membership_pairs = sorted(
                {
                    (str(row["etf_ticker"]), str(row["effective_date"]))
                    for row in memberships
                }
            )
            for etf_ticker, effective_date in membership_pairs:
                # Keep both equality predicates on the observation table.
                # SQLite otherwise chooses the provider prefix of the 20M-row
                # primary index for a large VALUES join and scans the whole
                # FMP history instead of using idx_etf_constituent_etf_date.
                metric = connection.execute(
                    """
                    SELECT o.etf_ticker, o.effective_date,
                           COUNT(*) AS row_count,
                           SUM(CASE WHEN o.constituent_ticker IS NOT NULL THEN 1 ELSE 0 END)
                               AS ticker_count,
                           SUM(CASE WHEN o.constituent_ticker IS NULL THEN 1 ELSE 0 END)
                               AS unresolved_count,
                           SUM(CASE WHEN o.weight_percent < 0 THEN 1 ELSE 0 END)
                               AS negative_weight_count,
                           SUM(CASE WHEN o.constituent_ticker IS NOT NULL
                                         AND o.weight_percent > 0
                                    THEN o.weight_percent ELSE 0 END)
                               AS positive_ticker_weight_sum,
                           SUM(COALESCE(o.weight_percent, 0)) AS total_weight_sum
                    FROM etf_constituent_observations o
                    WHERE o.provider='fmp'
                      AND o.etf_ticker=? AND o.effective_date=?
                    GROUP BY o.etf_ticker, o.effective_date
                    """,
                    (etf_ticker, effective_date),
                ).fetchone()
                if metric is None:
                    continue
                row_count = int(metric["row_count"] or 0)
                ticker_count = int(metric["ticker_count"] or 0)
                negative_count = int(metric["negative_weight_count"] or 0)
                positive_weight = float(metric["positive_ticker_weight_sum"] or 0.0)
                reasons = []
                if row_count < 5:
                    reasons.append("snapshot_has_fewer_than_5_positions")
                if not row_count or ticker_count / row_count < 0.80:
                    reasons.append("ticker_coverage_below_80_percent")
                if not 70.0 <= positive_weight <= 130.0:
                    reasons.append("positive_ticker_weight_sum_outside_70_130")
                if negative_count:
                    reasons.append("negative_weights_present")
                composition[(str(metric["etf_ticker"]), str(metric["effective_date"]))] = {
                    "snapshot_row_count": row_count,
                    "ticker_constituent_count": ticker_count,
                    "unresolved_constituent_count": int(metric["unresolved_count"] or 0),
                    "negative_weight_count": negative_count,
                    "positive_ticker_weight_sum": positive_weight,
                    "total_weight_sum": float(metric["total_weight_sum"] or 0.0),
                    "direct_equity_proxy_eligible": not reasons,
                    "direct_equity_proxy_reasons": reasons,
                }
            artifact_ids = sorted(
                {
                    int(row["raw_artifact_id"])
                    for row in list(holdings) + list(memberships)
                }
            )
            provenance = {}
            for artifact_id in artifact_ids:
                row = connection.execute(
                    """
                    SELECT payload_sha256, raw_relative_path, captured_at_utc
                    FROM raw_artifacts WHERE id=?
                    """,
                    (artifact_id,),
                ).fetchone()
                if row:
                    provenance[str(row["payload_sha256"])] = {
                        "source": "fmp",
                        "captured_at_utc": str(row["captured_at_utc"]),
                        "raw_relative_path": str(row["raw_relative_path"]),
                    }
        membership_packets = []
        for row in memberships:
            payload = self._packet_row(
                row, membership_availability[self._availability_key(row)]
            )
            payload.update(
                composition.get(
                    (str(row["etf_ticker"]), str(row["effective_date"])),
                    {
                        "snapshot_row_count": 0,
                        "ticker_constituent_count": 0,
                        "unresolved_constituent_count": 0,
                        "negative_weight_count": 0,
                        "positive_ticker_weight_sum": 0.0,
                        "total_weight_sum": 0.0,
                        "direct_equity_proxy_eligible": False,
                        "direct_equity_proxy_reasons": ["snapshot_composition_missing"],
                    },
                )
            )
            membership_packets.append(payload)
        return {
            "source": "fmp_v4_historical_etf_holdings",
            "as_of_date": as_of,
            "constituent_snapshot_date": (
                str(etf_snapshot["effective_date"]) if etf_snapshot else None
            ),
            "constituent_snapshot_training_available_session_date": (
                snapshot_training_available if etf_snapshot else None
            ),
            "availability_policy": etf_constituent_policy_manifest(),
            "constituents": [
                self._packet_row(
                    row,
                    holding_availability[self._availability_key(row)],
                )
                for row in holdings
            ],
            "etf_memberships": membership_packets,
            "raw_provenance": provenance,
        }

    def counts(self) -> dict:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT 'available_dates' name, COUNT(*) count
                FROM etf_constituent_available_dates
                UNION ALL SELECT 'snapshots', COUNT(*) FROM etf_constituent_snapshots
                UNION ALL SELECT 'versions', COUNT(*) FROM etf_constituent_versions
                UNION ALL SELECT 'observations', COUNT(*) FROM etf_constituent_observations
                """
            ).fetchall()
        return {str(row["name"]): int(row["count"]) for row in rows}


class FmpEtfConstituentLayer:
    def __init__(
        self,
        database: Database,
        http: HttpCaptureClient,
        api_key: Optional[str],
        *,
        initialize_schema: bool = True,
    ):
        self.database = database
        self.provider = FmpEtfConstituentProvider(http, api_key)
        self.store = FmpEtfConstituentStore(
            database, initialize_schema=initialize_schema
        )

    def backfill(
        self,
        start_date: str,
        end_date: str,
        tickers: Sequence[str],
        universe_contract: Optional[Mapping[str, Any]] = None,
        continue_on_error: bool = True,
    ) -> dict:
        start = validate_iso_date(start_date)
        end = validate_iso_date(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")
        normalized = sorted({normalize_symbol(ticker) for ticker in tickers})
        if not normalized:
            raise ValueError("at least one ETF ticker is required")
        contract = {
            "provider": "fmp",
            "endpoint": "/api/v4/etf-holdings",
            "from": start,
            "to": end,
            "tickers": normalized,
            "universe_contract": dict(universe_contract or {}),
            "availability_gate": "acceptanceTime",
        }
        job_id = "etf-constituents:{}".format(
            sha256_bytes(canonical_json(contract).encode("utf-8"))[:16]
        )
        self.database.register_job(
            job_id, "backfill_fmp_etf_constituents", contract, "2026-07-14.fmp-v4"
        )
        results = Counter(
            {
                "etfs": len(normalized),
                "date_lists_done": 0,
                "date_lists_skipped": 0,
                "dates_discovered": 0,
                "done": 0,
                "empty": 0,
                "skipped": 0,
                "failed": 0,
                "records": 0,
            }
        )
        errors = []
        for etf in normalized:
            dates_item_key = "dates:{}".format(etf)
            dates_scope = {
                "etf_ticker": etf,
                "stage": "available_dates",
                "endpoint": "/api/v4/etf-holdings/portfolio-date",
            }
            self.database.ensure_checkpoint(
                job_id, "fmp", dates_item_key, dates_scope
            )
            try:
                if (
                    self.database.checkpoint_status(
                        job_id, "fmp", dates_item_key
                    )
                    == "done"
                ):
                    dates = self.store.dates_for_etf(etf)
                    results["date_lists_skipped"] += 1
                else:
                    self.database.mark_checkpoint_running(
                        job_id, "fmp", dates_item_key
                    )
                    dates, dates_artifact = self.provider.capture_dates(etf)
                    self.store.ingest_dates(etf, dates, dates_artifact)
                    self.database.mark_checkpoint_done(
                        job_id,
                        "fmp",
                        dates_item_key,
                        dates_artifact.artifact_id,
                        len(dates),
                    )
                    results["date_lists_done"] += 1
                eligible_dates = [item for item in dates if start <= item <= end]
                results["dates_discovered"] += len(eligible_dates)
            except Exception as error:
                self.database.mark_checkpoint_failed(
                    job_id,
                    "fmp",
                    dates_item_key,
                    "{}: {}".format(type(error).__name__, str(error)),
                )
                results["failed"] += 1
                errors.append(
                    {"etf_ticker": etf, "stage": "dates", "error": str(error)}
                )
                if not continue_on_error:
                    raise
                continue
            for effective_date in eligible_dates:
                item_key = "{}:{}".format(etf, effective_date)
                scope = {
                    "etf_ticker": etf,
                    "effective_date": effective_date,
                    "endpoint": "/api/v4/etf-holdings",
                }
                self.database.ensure_checkpoint(job_id, "fmp", item_key, scope)
                if self.database.checkpoint_status(job_id, "fmp", item_key) == "done":
                    results["skipped"] += 1
                    continue
                self.database.mark_checkpoint_running(job_id, "fmp", item_key)
                try:
                    capture = self.provider.capture_snapshot(etf, effective_date)
                    self.store.ingest_snapshot(etf, effective_date, capture)
                    self.database.mark_checkpoint_done(
                        job_id,
                        "fmp",
                        item_key,
                        capture.artifact.artifact_id,
                        len(capture.records),
                    )
                    results["records"] += len(capture.records)
                    results["done" if capture.records else "empty"] += 1
                except Exception as error:
                    self.database.mark_checkpoint_failed(
                        job_id,
                        "fmp",
                        item_key,
                        "{}: {}".format(type(error).__name__, str(error)),
                    )
                    results["failed"] += 1
                    errors.append(
                        {
                            "etf_ticker": etf,
                            "effective_date": effective_date,
                            "stage": "snapshot",
                            "error": str(error),
                        }
                    )
                    if not continue_on_error:
                        raise
        return {
            "ok": results["failed"] == 0,
            "job_id": job_id,
            **dict(results),
            "checkpoint_summary": self.database.checkpoint_summary(job_id),
            "database_counts": self.store.counts(),
            "errors": errors,
        }

    def packet_for_symbol(self, symbol: str, as_of_date: str) -> dict:
        return self.store.packet_for_symbol(symbol, as_of_date)

    def verify(self) -> dict:
        counts = self.store.counts()
        with self.database.connect() as connection:
            invalid = connection.execute(
                """
                SELECT COUNT(*) count FROM etf_constituent_observations
                WHERE effective_date>available_date
                """
            ).fetchone()
        errors = []
        if invalid and int(invalid["count"]):
            errors.append(
                {
                    "error": "invalid_etf_constituent_pit_or_weight",
                    "count": int(invalid["count"]),
                }
            )
        return {"ok": not errors, "counts": counts, "errors": errors}
