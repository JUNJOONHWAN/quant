"""Full-universe discovery for one point-in-time Quant AI Radar run."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator


ACCEPTED_QUALITY = ("pass", "warn", "single_source")


class UniverseError(RuntimeError):
    """Raised when a complete as-of universe cannot be proven."""


@dataclass(frozen=True)
class Candidate:
    symbol: str
    proxy_task_type: str
    quality_status: str
    relation_types: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "proxy_task_type": self.proxy_task_type,
            "quality_status": self.quality_status,
            "relation_types": list(self.relation_types),
        }


@contextmanager
def connect_readonly(database: Any) -> Iterator[sqlite3.Connection]:
    """Open either one SQLite file or a Database-compatible shared overlay."""

    connector = getattr(database, "connect", None)
    if callable(connector):
        with connector() as connection:
            yield connection
        return
    path = Path(database).expanduser().resolve()
    if not path.is_file():
        raise UniverseError(f"normalized dataset database is missing: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def resolve_as_of_date(database: Any, requested: str | None = None) -> str:
    if requested:
        return date.fromisoformat(requested).isoformat()
    latest_quality_date = getattr(database, "latest_quality_date", None)
    if callable(latest_quality_date):
        observed = latest_quality_date()
        if not observed:
            raise UniverseError("no quality-eligible positive-volume market date exists")
        return str(observed)
    with connect_readonly(database) as connection:
        row = connection.execute(
            """
            SELECT MAX(q.trade_date) AS trade_date
            FROM quality_checks q
            JOIN daily_observations o
              ON o.symbol=q.symbol AND o.trade_date=q.trade_date
            WHERE q.status IN ('pass','warn','single_source')
              AND o.close>0 AND o.volume>0
            """
        ).fetchone()
    if not row or not row["trade_date"]:
        raise UniverseError("no quality-eligible positive-volume market date exists")
    return str(row["trade_date"])


def dataset_source_fingerprint(database: Any, as_of_date: str) -> dict[str, Any]:
    """Build a retry-stable fingerprint from persisted source rows, not timestamps."""

    as_of = date.fromisoformat(as_of_date).isoformat()
    source_fingerprint = getattr(database, "source_fingerprint", None)
    if callable(source_fingerprint):
        return source_fingerprint(as_of)
    queries = {
        # MAX(rowid/id) is index-tail O(1). A new or revised persisted source
        # row changes the value, while a checkpoint-only retry leaves it stable.
        "daily_observation_versions": (
            "SELECT NULL,COALESCE(MAX(rowid),0) FROM daily_observation_versions"
        ),
        "etf_flow_versions": (
            "SELECT NULL,COALESCE(MAX(id),0) FROM etf_flow_versions"
        ),
        "etf_constituent_observations": (
            "SELECT NULL,COALESCE(MAX(rowid),0) FROM etf_constituent_observations"
        ),
        "quality_checks_as_of": (
            "SELECT COUNT(*),0 FROM quality_checks WHERE trade_date=?"
        ),
    }
    values: dict[str, Any] = {"as_of_date": as_of}
    with connect_readonly(database) as connection:
        for name, sql in queries.items():
            parameter_count = sql.count("?")
            row = connection.execute(sql, (as_of,) * parameter_count).fetchone()
            values[name] = {
                "row_count": int(row[0]) if row[0] is not None else None,
                "max_row_id": int(row[1]),
            }
    values["sha256"] = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return values


def _symbol_set(
    connection: sqlite3.Connection, sql: str, parameters: tuple[str, ...]
) -> set[str]:
    return {
        str(row[0]).strip().upper()
        for row in connection.execute(sql, parameters)
        if row[0]
    }


def scan_universe(
    database: Any,
    as_of_date: str,
    *,
    relation_index_path: Path | None = None,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Scan every observed symbol, then select every ETF-related candidate.

    No fixed ticker list or top-N gate is used here.  Non-related securities are
    counted as controls and excluded only because this product's factual basis
    is ETF flow plus point-in-time ETF membership.
    """

    as_of = date.fromisoformat(as_of_date).isoformat()
    indexed_price_reader = getattr(database, "current_universe_price_rows", None)
    if relation_index_path is not None and callable(indexed_price_reader):
        price_rows = indexed_price_reader(as_of)
    else:
        with connect_readonly(database) as connection:
            price_rows = connection.execute(
                """
                SELECT q.symbol, q.status,
                       MAX(
                         CASE WHEN o.volume>0 AND o.close>0 THEN 1 ELSE 0 END
                       ) AS tradable
                FROM quality_checks q
                JOIN daily_observations o
                  ON o.symbol=q.symbol AND o.trade_date=q.trade_date
                WHERE q.trade_date=?
                  AND q.status IN ('pass','warn','single_source')
                GROUP BY q.symbol, q.status
                ORDER BY q.symbol
                """,
                (as_of,),
            ).fetchall()
    if relation_index_path is None:
        with connect_readonly(database) as connection:
            flow_etfs = _symbol_set(
                connection,
                """
                SELECT DISTINCT ticker FROM etf_flow_observations
                WHERE effective_date<=? AND processed_date<=? AND fund_flow IS NOT NULL
                """,
                (as_of, as_of),
            )
            snapshot_etfs = _symbol_set(
                connection,
                """
                SELECT DISTINCT etf_ticker FROM etf_constituent_snapshots
                WHERE effective_date<=? AND available_date<=?
                """,
                (as_of, as_of),
            )
            membership_stocks = _symbol_set(
                connection,
                """
                SELECT DISTINCT constituent_ticker FROM etf_constituent_observations
                WHERE effective_date<=? AND available_date<=?
                  AND constituent_ticker IS NOT NULL AND constituent_ticker<>''
                """,
                (as_of, as_of),
            )
        relation_source = "direct_historical_scan"
    else:
        # Imported lazily so legacy single-file dataset scans remain usable
        # in tests and offline diagnostics.
        from workflows.quant_ai_radar.relation_index import (
            visible_relation_sets,
        )

        indexed = visible_relation_sets(relation_index_path, as_of)
        flow_etfs = indexed["massive_etf_flow"]
        snapshot_etfs = indexed["fmp_etf_constituents"]
        membership_stocks = indexed["fmp_etf_membership"]
        relation_source = "persistent_incremental_relation_index"

    # The source tables carry both effective and provider-available dates.  The
    # exact next-session visibility gate is applied again by analysis_packet_v3.
    # These sets are only a cheap superset for the full packet scan.
    candidate_rows: list[Candidate] = []
    quality_counts: Counter[str] = Counter()
    observed_symbols = 0
    positive_volume_symbols = 0
    no_etf_relation = 0
    relation_counts: Counter[str] = Counter()
    for row in price_rows:
        observed_symbols += 1
        quality_counts[str(row["status"])] += 1
        if not int(row["tradable"] or 0):
            continue
        positive_volume_symbols += 1
        symbol = str(row["symbol"]).strip().upper()
        relations = []
        if symbol in flow_etfs:
            relations.append("massive_etf_flow")
        if symbol in snapshot_etfs:
            relations.append("fmp_etf_constituents")
        if symbol in membership_stocks:
            relations.append("fmp_etf_membership")
        if not relations:
            no_etf_relation += 1
            continue
        for relation in relations:
            relation_counts[relation] += 1
        task_type = (
            "etf_own_flow_analysis"
            if symbol in flow_etfs or symbol in snapshot_etfs
            else "stock_constituent_flow_analysis"
        )
        candidate_rows.append(
            Candidate(
                symbol=symbol,
                proxy_task_type=task_type,
                quality_status=str(row["status"]),
                relation_types=tuple(relations),
            )
        )

    manifest = {
        "schema_version": "quant.ai_radar_universe.v1",
        "as_of_date": as_of,
        "scope": "all observed symbols; every ETF-related symbol receives full PIT packet eligibility review",
        "fixed_ticker_list_used": False,
        "top_n_selection_used": False,
        "observed_quality_symbols": observed_symbols,
        "positive_volume_symbols": positive_volume_symbols,
        "etf_related_candidates": len(candidate_rows),
        "all_stock_control_symbols": no_etf_relation,
        "quality_status_counts": dict(sorted(quality_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "source_relation_universe_counts": {
            "massive_etf_flow_tickers": len(flow_etfs),
            "fmp_etf_snapshot_tickers": len(snapshot_etfs),
            "fmp_etf_member_symbols": len(membership_stocks),
        },
        "relation_source": relation_source,
        "historical_relation_tables_scanned_this_run": (
            relation_index_path is None
        ),
        "control_policy": (
            "symbols with no visible ETF flow, ETF constituent snapshot, or ETF membership "
            "remain counted but are outside this ETF-grounded inference product"
        ),
    }
    return candidate_rows, manifest


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_candidates(path: Path, candidates: Iterable[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)
