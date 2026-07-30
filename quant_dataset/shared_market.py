"""Read-only point-in-time overlay for the shared Oracle market store.

The long FMP history remains immutable.  Current sessions are written once by
the Market Structure Oracle incremental collector.  This module exposes both
databases through one read-only SQLite connection so Quant AI Radar can build
the same ``quant.analysis_packet.v3`` packets without recollecting or copying
source data.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from .corporate_actions import corporate_action_summary
from .storage import canonical_json


DEFAULT_BASE_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/"
    "normalized/daily_observations.sqlite3"
)
DEFAULT_INCREMENTAL_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/"
    "incremental/normalized/daily_observations.sqlite3"
)
DEFAULT_ORACLE_STATUS = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/"
    "incremental/state/oracle_incremental_status.json"
)


class SharedMarketStoreError(RuntimeError):
    """The Oracle source snapshot cannot safely support model inference."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(connection: sqlite3.Connection, sql: str, parameters=()) -> Any:
    row = connection.execute(sql, parameters).fetchone()
    return row[0] if row else None


def _table_columns(connection: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [
        str(row[1])
        for row in connection.execute(
            "PRAGMA {}.table_info({})".format(schema, table)
        ).fetchall()
    ]


def _assert_matching_schema(
    connection: sqlite3.Connection, tables: tuple[str, ...]
) -> None:
    mismatches = []
    for table in tables:
        base = _table_columns(connection, "main", table)
        incremental = _table_columns(connection, "oracle_incremental", table)
        if not base or base != incremental:
            mismatches.append(
                {
                    "table": table,
                    "base_columns": base,
                    "incremental_columns": incremental,
                }
            )
    if mismatches:
        raise SharedMarketStoreError(
            "base/incremental SQLite schemas differ: "
            + canonical_json(mismatches)
        )


@dataclass(frozen=True)
class SharedMarketBinding:
    base_database: Path
    incremental_database: Path
    oracle_status_path: Path
    base_history_end: str
    target_as_of_date: str
    latest_flow_effective_date: str
    latest_constituent_effective_date: str
    latest_constituent_available_date: str
    constituent_available_lag_days: int
    corporate_action_visible_record_count: int
    corporate_action_projection_sha256: str
    source_fingerprint_sha256: str
    source_fingerprint: dict[str, Any]

    def public_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "quant.shared_market_binding.v1",
            "status": "confirmed",
            "base_database": str(self.base_database),
            "incremental_database": str(self.incremental_database),
            "oracle_status_path": str(self.oracle_status_path),
            "base_history_end": self.base_history_end,
            "target_as_of_date": self.target_as_of_date,
            "latest_flow_effective_date": self.latest_flow_effective_date,
            "latest_constituent_effective_date": (
                self.latest_constituent_effective_date
            ),
            "latest_constituent_available_date": (
                self.latest_constituent_available_date
            ),
            "constituent_available_lag_days": (
                self.constituent_available_lag_days
            ),
            "corporate_action_visible_record_count": (
                self.corporate_action_visible_record_count
            ),
            "corporate_action_projection_sha256": (
                self.corporate_action_projection_sha256
            ),
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "single_writer": "market_structure_oracle_incremental_store",
            "consumer_mode": "sqlite_read_only_overlay",
        }


def load_shared_market_binding(
    *,
    base_database: Path = DEFAULT_BASE_DATABASE,
    incremental_database: Path = DEFAULT_INCREMENTAL_DATABASE,
    oracle_status_path: Path = DEFAULT_ORACLE_STATUS,
    max_constituent_available_lag_days: int = 45,
) -> SharedMarketBinding:
    """Verify the Oracle completion receipt and both source databases."""

    base = Path(base_database).expanduser().resolve()
    incremental = Path(incremental_database).expanduser().resolve()
    status_path = Path(oracle_status_path).expanduser().resolve()
    for label, path in (
        ("base database", base),
        ("incremental database", incremental),
        ("Oracle status", status_path),
    ):
        if not path.is_file():
            raise SharedMarketStoreError(f"{label} is missing: {path}")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SharedMarketStoreError(f"invalid Oracle status JSON: {status_path}") from exc
    if status.get("status") != "COMPLETE":
        raise SharedMarketStoreError(
            f"Oracle incremental store is not COMPLETE: {status.get('status')}"
        )
    status_schema = str(
        status.get("schema")
        or "quant.market_structure_oracle.incremental.v2"
    )
    if status_schema not in {
        "quant.market_structure_oracle.incremental.v2",
        "quant.market_structure_oracle.incremental.v3",
    }:
        raise SharedMarketStoreError(
            f"unsupported Oracle incremental schema: {status_schema}"
        )
    missing_sessions = list(status.get("missing_sessions") or [])
    if missing_sessions:
        raise SharedMarketStoreError(
            "Oracle incremental store has missing sessions: "
            + ",".join(map(str, missing_sessions))
        )
    target = str(status.get("target_as_of_date") or "")
    base_end = str(status.get("base_history_end") or "")
    try:
        date.fromisoformat(target)
        date.fromisoformat(base_end)
    except ValueError as exc:
        raise SharedMarketStoreError("Oracle status contains an invalid date") from exc
    status_database = str(status.get("database") or "")
    if status_database and Path(status_database).expanduser().resolve() != incremental:
        raise SharedMarketStoreError(
            "Oracle status is bound to a different incremental database"
        )

    with sqlite3.connect(f"file:{base}?mode=ro", uri=True) as base_connection:
        observed_base_end = _scalar(
            base_connection,
            "SELECT MAX(trade_date) FROM daily_observations WHERE source='fmp'",
        )
        constituent_count = int(
            _scalar(
                base_connection,
                "SELECT COUNT(*) FROM etf_constituent_observations",
            )
            or 0
        )
        constituent_effective = str(
            _scalar(
                base_connection,
                "SELECT MAX(effective_date) FROM etf_constituent_observations",
            )
            or ""
        )
        constituent_available = str(
            _scalar(
                base_connection,
                "SELECT MAX(available_date) FROM etf_constituent_observations",
            )
            or ""
        )
        base_flow_count = int(
            _scalar(base_connection, "SELECT COUNT(*) FROM etf_flow_versions") or 0
        )
    if str(observed_base_end or "") != base_end:
        raise SharedMarketStoreError(
            f"base cutoff mismatch: status={base_end} database={observed_base_end}"
        )

    with sqlite3.connect(
        f"file:{incremental}?mode=ro", uri=True
    ) as incremental_connection:
        target_rows = int(
            _scalar(
                incremental_connection,
                """
                SELECT COUNT(DISTINCT symbol) FROM daily_observations
                WHERE source IN ('massive','fmp') AND trade_date=?
                """,
                (target,),
            )
            or 0
        )
        minimum_rows = int(
            (
                (
                    (status.get("market_row_gate") or {}).get(
                        "minimum_rows_by_session"
                    )
                    or {}
                ).get(target)
                or (status.get("market_row_gate") or {}).get("minimum_rows")
            )
            or 0
        )
        flow_effective = str(
            _scalar(
                incremental_connection,
                "SELECT MAX(effective_date) FROM etf_flow_observations",
            )
            or ""
        )
        incremental_flow_count = int(
            _scalar(
                incremental_connection,
                "SELECT COUNT(*) FROM etf_flow_versions",
            )
            or 0
        )
        incremental_constituent_count = int(
            _scalar(
                incremental_connection,
                "SELECT COUNT(*) FROM etf_constituent_observations",
            )
            or 0
        )
        incremental_constituent_effective = str(
            _scalar(
                incremental_connection,
                "SELECT MAX(effective_date) FROM etf_constituent_observations",
            )
            or ""
        )
        incremental_constituent_available = str(
            _scalar(
                incremental_connection,
                "SELECT MAX(available_date) FROM etf_constituent_observations",
            )
            or ""
        )
        seal_row = incremental_connection.execute(
            """
            SELECT schema_version,source_contract,receipt_sha256,payload_json
            FROM oracle_snapshot_seals WHERE target_as_of_date=?
            """,
            (target,),
        ).fetchone()
        daily_rows = [
            list(row)
            for row in incremental_connection.execute(
                """
                SELECT source,trade_date,COUNT(*),MIN(raw_artifact_id),
                       MAX(raw_artifact_id)
                FROM daily_observations
                GROUP BY source,trade_date ORDER BY source,trade_date
                """
            ).fetchall()
        ]
    if minimum_rows <= 0 or target_rows < minimum_rows:
        raise SharedMarketStoreError(
            f"Oracle target market-row gate failed: {target_rows} < {minimum_rows}"
        )
    expected_flow = str(
        ((status.get("etf_flow") or {}).get("expected_effective_date_at_least"))
        or ""
    )
    if not flow_effective or (expected_flow and flow_effective < expected_flow):
        raise SharedMarketStoreError(
            f"Oracle ETF flow is stale: latest={flow_effective} expected>={expected_flow}"
        )
    if not seal_row:
        raise SharedMarketStoreError(
            f"Oracle target snapshot seal is missing: {target}"
        )
    seal_schema, source_contract, receipt_sha, receipt_json = map(str, seal_row)
    if seal_schema != "quant.oracle_snapshot_seal.v1":
        raise SharedMarketStoreError(
            f"unsupported Oracle snapshot seal: {seal_schema}"
        )
    if source_contract != "oracle_owned_fmp_massive_no_etf_radar_dependency":
        raise SharedMarketStoreError(
            f"unexpected Oracle source contract: {source_contract}"
        )
    observed_receipt_sha = hashlib.sha256(
        receipt_json.encode("utf-8")
    ).hexdigest()
    if observed_receipt_sha != receipt_sha:
        raise SharedMarketStoreError("Oracle snapshot receipt hash mismatch")
    status_seal_sha = str(
        ((status.get("snapshot_seal") or {}).get("receipt_sha256")) or ""
    )
    if status_seal_sha != receipt_sha:
        raise SharedMarketStoreError(
            "Oracle status and SQLite snapshot seal do not match"
        )
    if status_schema == "quant.market_structure_oracle.incremental.v3":
        try:
            corporate_summary = corporate_action_summary(incremental, target)
        except sqlite3.Error as exc:
            raise SharedMarketStoreError(
                "Oracle corporate-action ledger is missing or invalid"
            ) from exc
        expected_corporate = status.get("corporate_actions") or {}
        if (
            corporate_summary["visible_projection_sha256"]
            != expected_corporate.get("visible_projection_sha256")
            or corporate_summary["visible_record_count"]
            != expected_corporate.get("visible_record_count")
            or corporate_summary["projection_count"]
            != expected_corporate.get("projection_count")
        ):
            raise SharedMarketStoreError(
                "Oracle corporate-action ledger does not match the snapshot seal"
            )
    else:
        corporate_summary = {
            "projection_count": 0,
            "version_count": 0,
            "visible_record_count": 0,
            "visible_projection_sha256": hashlib.sha256(b"[]").hexdigest(),
        }

    effective_candidates = [
        value
        for value in (constituent_effective, incremental_constituent_effective)
        if value
    ]
    available_candidates = [
        value
        for value in (constituent_available, incremental_constituent_available)
        if value
    ]
    if constituent_count + incremental_constituent_count <= 0 or not available_candidates:
        raise SharedMarketStoreError("no point-in-time ETF constituent rows are available")
    latest_constituent_effective = max(effective_candidates)
    latest_constituent_available = max(available_candidates)
    constituent_lag = (
        date.fromisoformat(target)
        - date.fromisoformat(latest_constituent_available)
    ).days
    if constituent_lag < 0:
        raise SharedMarketStoreError(
            "ETF constituent availability date is after the Oracle target"
        )
    if constituent_lag > max_constituent_available_lag_days:
        raise SharedMarketStoreError(
            "ETF constituent snapshot is stale: "
            f"available={latest_constituent_available} target={target} "
            f"lag_days={constituent_lag} max={max_constituent_available_lag_days}"
        )

    fingerprint = {
        "schema_version": "quant.shared_market_source_fingerprint.v2",
        "base": {
            "path": str(base),
            "bytes": base.stat().st_size,
            "mtime_ns": base.stat().st_mtime_ns,
            "history_end": base_end,
            "etf_flow_version_count": base_flow_count,
            "etf_constituent_observation_count": constituent_count,
        },
        "incremental": {
            "path": str(incremental),
            "daily_rows": daily_rows,
            "etf_flow_version_count": incremental_flow_count,
            "etf_constituent_observation_count": incremental_constituent_count,
            "corporate_action_projection_count": corporate_summary[
                "projection_count"
            ],
            "corporate_action_version_count": corporate_summary[
                "version_count"
            ],
            "corporate_action_visible_record_count": corporate_summary[
                "visible_record_count"
            ],
            "corporate_action_projection_sha256": corporate_summary[
                "visible_projection_sha256"
            ],
            "snapshot_seal": {
                "schema_version": seal_schema,
                "source_contract": source_contract,
                "receipt_sha256": receipt_sha,
            },
        },
        "oracle_status_sha256": _sha256(status_path),
        "target_as_of_date": target,
        "latest_flow_effective_date": flow_effective,
        "latest_constituent_effective_date": latest_constituent_effective,
        "latest_constituent_available_date": latest_constituent_available,
    }
    fingerprint_sha = hashlib.sha256(
        canonical_json(fingerprint).encode("utf-8")
    ).hexdigest()
    return SharedMarketBinding(
        base_database=base,
        incremental_database=incremental,
        oracle_status_path=status_path,
        base_history_end=base_end,
        target_as_of_date=target,
        latest_flow_effective_date=flow_effective,
        latest_constituent_effective_date=latest_constituent_effective,
        latest_constituent_available_date=latest_constituent_available,
        constituent_available_lag_days=constituent_lag,
        corporate_action_visible_record_count=corporate_summary[
            "visible_record_count"
        ],
        corporate_action_projection_sha256=corporate_summary[
            "visible_projection_sha256"
        ],
        source_fingerprint_sha256=fingerprint_sha,
        source_fingerprint=fingerprint,
    )


class SharedReadOnlyDatabase:
    """Database-compatible read-only overlay for packet and universe readers."""

    read_only = True

    def __init__(self, binding: SharedMarketBinding):
        self.binding = binding
        self.db_path = binding.base_database
        self.data_root = binding.base_database.parent.parent

    def source_fingerprint(self, as_of_date: str) -> dict[str, Any]:
        if date.fromisoformat(as_of_date).isoformat() != self.binding.target_as_of_date:
            raise SharedMarketStoreError(
                "requested as-of does not match the sealed Oracle snapshot"
            )
        return {
            "as_of_date": as_of_date,
            "sha256": self.binding.source_fingerprint_sha256,
            "binding": self.binding.public_metadata(),
        }

    def latest_quality_date(self) -> str | None:
        """Read the current-side quality tail without opening union views."""

        source = (
            self.binding.incremental_database
            if self.binding.target_as_of_date > self.binding.base_history_end
            else self.binding.base_database
        )
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
            row = connection.execute(
                """
                SELECT MAX(q.trade_date)
                FROM quality_checks q
                JOIN daily_observations o
                  ON o.symbol=q.symbol AND o.trade_date=q.trade_date
                WHERE q.status IN ('pass','warn','single_source')
                  AND o.close>0 AND o.volume>0
                """
            ).fetchone()
        return str(row[0]) if row and row[0] else None

    def current_universe_price_rows(self, as_of_date: str) -> list[sqlite3.Row]:
        """Return target-day tradability directly from the owning source DB."""

        as_of = date.fromisoformat(as_of_date).isoformat()
        if as_of != self.binding.target_as_of_date:
            raise SharedMarketStoreError(
                "universe date does not match the sealed Oracle target"
            )
        source = (
            self.binding.incremental_database
            if as_of > self.binding.base_history_end
            else self.binding.base_database
        )
        connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            return connection.execute(
                """
                SELECT q.symbol,q.status,
                       MAX(
                         CASE WHEN o.volume>0 AND o.close>0 THEN 1 ELSE 0 END
                       ) AS tradable
                FROM quality_checks q
                JOIN daily_observations o
                  ON o.symbol=q.symbol AND o.trade_date=q.trade_date
                WHERE q.trade_date=?
                  AND q.status IN ('pass','warn','single_source')
                GROUP BY q.symbol,q.status
                ORDER BY q.symbol
                """,
                (as_of,),
            ).fetchall()
        finally:
            connection.close()

    def history_payload_rows(
        self,
        symbol: str,
        as_of_date: str,
        lookback_days: int,
    ) -> list[dict[str, Any]]:
        """Merge indexed base/current price windows without union-view scans."""

        as_of = date.fromisoformat(as_of_date).isoformat()
        if lookback_days < 1:
            raise SharedMarketStoreError("lookback_days must be positive")
        source_ranges = (
            (
                self.binding.base_database,
                "",
                min(as_of, self.binding.base_history_end),
            ),
            (
                self.binding.incremental_database,
                self.binding.base_history_end,
                as_of,
            ),
        )
        rows: list[dict[str, Any]] = []
        for path, after_date, through_date in source_ranges:
            if through_date <= after_date:
                continue
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                source_rows = connection.execute(
                    """
                    WITH selected_dates AS (
                        SELECT trade_date FROM (
                            SELECT DISTINCT trade_date
                            FROM daily_observations
                            WHERE symbol=? AND trade_date>?
                              AND trade_date<=?
                            ORDER BY trade_date DESC LIMIT ?
                        ) ORDER BY trade_date
                    )
                    SELECT o.*,r.payload_sha256,r.raw_relative_path,
                           ce.captured_at_utc
                    FROM daily_observations o
                    JOIN selected_dates d ON d.trade_date=o.trade_date
                    JOIN raw_artifacts r ON r.id=o.raw_artifact_id
                    JOIN capture_events ce ON ce.id=o.capture_event_id
                    WHERE o.symbol=?
                    ORDER BY o.trade_date,o.source
                    """,
                    (
                        symbol,
                        after_date,
                        through_date,
                        lookback_days,
                        symbol,
                    ),
                ).fetchall()
                rows.extend(dict(row) for row in source_rows)
            finally:
                connection.close()
        selected_dates = sorted(
            {str(row["trade_date"]) for row in rows},
            reverse=True,
        )[:lookback_days]
        selected = set(selected_dates)
        return sorted(
            (row for row in rows if str(row["trade_date"]) in selected),
            key=lambda row: (str(row["trade_date"]), str(row["source"])),
        )

    def corporate_action_rows(self, as_of_date: str) -> list[dict[str, Any]]:
        """Return only split evidence visible and effective by the snapshot."""

        as_of = date.fromisoformat(as_of_date).isoformat()
        if as_of != self.binding.target_as_of_date:
            raise SharedMarketStoreError(
                "corporate-action date does not match the sealed Oracle target"
            )
        if not self.binding.corporate_action_visible_record_count:
            return []
        connection = sqlite3.connect(
            f"file:{self.binding.incremental_database}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT c.*,r.payload_sha256,r.raw_relative_path
                FROM corporate_actions c
                JOIN raw_artifacts r ON r.id=c.raw_artifact_id
                WHERE c.effective_date<=? AND c.available_date<=?
                ORDER BY c.symbol,c.effective_date,c.provider,
                         c.provider_event_id
                """,
                (as_of, as_of),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def flow_source_paths(self) -> tuple[Path, Path]:
        """Expose immutable source files for indexed PIT flow reads."""

        return (
            self.binding.base_database,
            self.binding.incremental_database,
        )

    @contextmanager
    def connect_constituents(self) -> Iterator[sqlite3.Connection]:
        """Use native base indices when Oracle has no constituent overrides."""

        incremental_count = int(
            self.binding.source_fingerprint["incremental"].get(
                "etf_constituent_observation_count",
                0,
            )
        )
        if incremental_count:
            with self.connect() as connection:
                yield connection
            return
        connection = sqlite3.connect(
            f"file:{self.binding.base_database}?mode=ro",
            uri=True,
            timeout=120,
        )
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            f"file:{self.binding.base_database}?mode=ro",
            uri=True,
            timeout=120,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(
                "ATTACH DATABASE ? AS oracle_incremental",
                (f"file:{self.binding.incremental_database}?mode=ro",),
            )
            connection.execute("PRAGMA temp_store=MEMORY")
            tables = (
                "raw_artifacts",
                "capture_events",
                "daily_observation_versions",
                "daily_observations",
                "quality_checks",
                "etf_flow_versions",
                "etf_flow_observations",
                "etf_constituent_available_dates",
                "etf_constituent_snapshots",
                "etf_constituent_versions",
                "etf_constituent_observations",
            )
            _assert_matching_schema(connection, tables)
            offsets = {
                "raw": int(
                    _scalar(connection, "SELECT COALESCE(MAX(id),0) FROM main.raw_artifacts")
                    or 0
                ),
                "capture": int(
                    _scalar(
                        connection,
                        "SELECT COALESCE(MAX(id),0) FROM main.capture_events",
                    )
                    or 0
                ),
                "daily_version": int(
                    _scalar(
                        connection,
                        "SELECT COALESCE(MAX(id),0) FROM main.daily_observation_versions",
                    )
                    or 0
                ),
                "flow_version": int(
                    _scalar(
                        connection,
                        "SELECT COALESCE(MAX(id),0) FROM main.etf_flow_versions",
                    )
                    or 0
                ),
                "constituent_version": int(
                    _scalar(
                        connection,
                        "SELECT COALESCE(MAX(id),0) FROM main.etf_constituent_versions",
                    )
                    or 0
                ),
            }
            has_incremental_constituent_observations = bool(
                _scalar(
                    connection,
                    """
                    SELECT EXISTS(
                      SELECT 1
                      FROM oracle_incremental.etf_constituent_observations
                      LIMIT 1
                    )
                    """,
                )
            )
            if has_incremental_constituent_observations:
                constituent_observations_view = f"""
                CREATE TEMP VIEW etf_constituent_observations AS
                  SELECT b.* FROM main.etf_constituent_observations b
                  WHERE NOT EXISTS (
                    SELECT 1
                    FROM oracle_incremental.etf_constituent_observations i
                    WHERE i.provider=b.provider
                      AND i.etf_ticker=b.etf_ticker
                      AND i.constituent_key=b.constituent_key
                      AND i.effective_date=b.effective_date
                  )
                  UNION ALL
                  SELECT provider,etf_ticker,constituent_key,
                         constituent_ticker,constituent_name,isin,cusip,cik,lei,
                         effective_date,acceptance_time,available_date,
                         availability_basis,pit_confidence,balance,value_usd,
                         weight_percent,currency,units,asset_category,
                         investment_country,
                         raw_artifact_id+{offsets['raw']} AS raw_artifact_id,
                         capture_event_id+{offsets['capture']} AS capture_event_id,
                         source_row_index,captured_at_utc,extra_json
                  FROM oracle_incremental.etf_constituent_observations;
                """
            else:
                # Avoid a correlated NOT EXISTS over the 20M-row immutable
                # history when the current Oracle snapshot has no constituent
                # overrides. SQLite can push symbol/date predicates through
                # this direct view and use the base table indices.
                constituent_observations_view = """
                CREATE TEMP VIEW etf_constituent_observations AS
                  SELECT * FROM main.etf_constituent_observations;
                """
            base_end = self.binding.base_history_end.replace("'", "''")
            script = f"""
            CREATE TEMP VIEW raw_artifacts AS
              SELECT * FROM main.raw_artifacts
              UNION ALL
              SELECT id+{offsets['raw']} AS id,source,dataset,partition_key,
                     request_fingerprint,payload_sha256,raw_relative_path,
                     metadata_relative_path,request_json,response_json,
                     captured_at_utc,response_status,payload_bytes,compressed_bytes
              FROM oracle_incremental.raw_artifacts;

            CREATE TEMP VIEW capture_events AS
              SELECT * FROM main.capture_events
              UNION ALL
              SELECT id+{offsets['capture']} AS id,
                     raw_artifact_id+{offsets['raw']} AS raw_artifact_id,
                     source,dataset,partition_key,request_fingerprint,
                     captured_at_utc,response_status,payload_bytes,
                     request_json,response_json
              FROM oracle_incremental.capture_events;

            CREATE TEMP VIEW daily_observation_versions AS
              SELECT * FROM main.daily_observation_versions
              UNION ALL
              SELECT id+{offsets['daily_version']} AS id,source,symbol,trade_date,
                     open,high,low,close,adjusted_close,volume,vwap,
                     transaction_count,adjusted,source_timestamp_ms,
                     raw_artifact_id+{offsets['raw']} AS raw_artifact_id,
                     capture_event_id+{offsets['capture']} AS capture_event_id,
                     source_row_index,ingested_at_utc,extra_json
              FROM oracle_incremental.daily_observation_versions
              WHERE trade_date>'{base_end}';

            CREATE TEMP VIEW daily_observations AS
              SELECT * FROM main.daily_observations
              UNION ALL
              SELECT source,symbol,trade_date,open,high,low,close,adjusted_close,
                     volume,vwap,transaction_count,adjusted,source_timestamp_ms,
                     raw_artifact_id+{offsets['raw']} AS raw_artifact_id,
                     capture_event_id+{offsets['capture']} AS capture_event_id,
                     source_row_index,ingested_at_utc,extra_json
              FROM oracle_incremental.daily_observations
              WHERE trade_date>'{base_end}';

            CREATE TEMP VIEW quality_checks AS
              SELECT * FROM main.quality_checks
              UNION ALL
              SELECT * FROM oracle_incremental.quality_checks
              WHERE trade_date>'{base_end}';

            CREATE TEMP VIEW etf_flow_versions AS
              SELECT * FROM main.etf_flow_versions
              UNION ALL
              SELECT id+{offsets['flow_version']} AS id,provider,endpoint_id,
                     ticker,effective_date,processed_date,fund_flow,nav,
                     shares_outstanding,assets,currency,available_at_date,
                     availability_basis,pit_confidence,record_hash,
                     source_record_id,
                     raw_artifact_id+{offsets['raw']} AS raw_artifact_id,
                     capture_event_id+{offsets['capture']} AS capture_event_id,
                     source_row_index,captured_at_utc,ingested_at_utc,extra_json
              FROM oracle_incremental.etf_flow_versions;

            CREATE TEMP VIEW etf_flow_observations AS
              SELECT * FROM main.etf_flow_observations
              UNION ALL
              SELECT provider,endpoint_id,ticker,effective_date,processed_date,
                     fund_flow,nav,shares_outstanding,assets,currency,
                     available_at_date,availability_basis,pit_confidence,
                     record_hash,source_record_id,
                     version_id+{offsets['flow_version']} AS version_id,
                     raw_artifact_id+{offsets['raw']} AS raw_artifact_id,
                     capture_event_id+{offsets['capture']} AS capture_event_id,
                     source_row_index,captured_at_utc,ingested_at_utc,extra_json
              FROM oracle_incremental.etf_flow_observations;

            CREATE TEMP VIEW etf_constituent_available_dates AS
              SELECT b.* FROM main.etf_constituent_available_dates b
              WHERE NOT EXISTS (
                SELECT 1 FROM oracle_incremental.etf_constituent_available_dates i
                WHERE i.provider=b.provider AND i.etf_ticker=b.etf_ticker
                  AND i.effective_date=b.effective_date
              )
              UNION ALL
              SELECT provider,etf_ticker,effective_date,
                     raw_artifact_id+{offsets['raw']} AS raw_artifact_id,
                     capture_event_id+{offsets['capture']} AS capture_event_id,
                     captured_at_utc
              FROM oracle_incremental.etf_constituent_available_dates;

            CREATE TEMP VIEW etf_constituent_snapshots AS
              SELECT b.* FROM main.etf_constituent_snapshots b
              WHERE NOT EXISTS (
                SELECT 1 FROM oracle_incremental.etf_constituent_snapshots i
                WHERE i.provider=b.provider AND i.etf_ticker=b.etf_ticker
                  AND i.effective_date=b.effective_date
              )
              UNION ALL
              SELECT provider,etf_ticker,effective_date,available_date,row_count,
                     invalid_row_count,
                     raw_artifact_id+{offsets['raw']} AS raw_artifact_id,
                     capture_event_id+{offsets['capture']} AS capture_event_id,
                     captured_at_utc
              FROM oracle_incremental.etf_constituent_snapshots;

            CREATE TEMP VIEW etf_constituent_versions AS
              SELECT * FROM main.etf_constituent_versions
              UNION ALL
              SELECT id+{offsets['constituent_version']} AS id,provider,
                     etf_ticker,constituent_key,constituent_ticker,
                     constituent_name,isin,cusip,cik,lei,effective_date,
                     acceptance_time,available_date,availability_basis,
                     pit_confidence,balance,value_usd,weight_percent,currency,
                     units,asset_category,investment_country,
                     raw_artifact_id+{offsets['raw']} AS raw_artifact_id,
                     capture_event_id+{offsets['capture']} AS capture_event_id,
                     source_row_index,captured_at_utc,extra_json
              FROM oracle_incremental.etf_constituent_versions;

            {constituent_observations_view}
            """
            connection.executescript(script)
            yield connection
        finally:
            connection.close()

    def quality_for_pair(self, symbol: str, trade_date: str):
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM quality_checks WHERE symbol=? AND trade_date=?",
                (symbol, trade_date),
            ).fetchone()
