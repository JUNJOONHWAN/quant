"""Read-only adapters for the sealed base store and Oracle increment."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def canonical_symbol(value: object) -> str:
    """Normalize the provider spellings used by prices, indexes, and holdings."""

    return str(value or "").strip().upper().replace(".", "-")


@dataclass(frozen=True)
class SnapshotMeta:
    origin: str
    etf_ticker: str
    effective_date: str
    provider_available_date: str
    row_count: int
    ticker_count: int
    negative_weight_count: int
    positive_ticker_weight_sum: float
    eligible: bool


class SourceBundle:
    """Two independent read-only SQLite connections with explicit precedence."""

    def __init__(self, base_database: Path, incremental_database: Path | None) -> None:
        self.base_database = Path(base_database)
        self.incremental_database = (
            Path(incremental_database) if incremental_database else None
        )
        if not self.base_database.is_file():
            raise FileNotFoundError(self.base_database)
        self.connections: dict[str, sqlite3.Connection] = {
            "base": self._connect(self.base_database)
        }
        if self.incremental_database and self.incremental_database.is_file():
            self.connections["incremental"] = self._connect(
                self.incremental_database
            )

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-262144")
        return connection

    def close(self) -> None:
        for connection in self.connections.values():
            connection.close()

    def __enter__(self) -> "SourceBundle":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _has_table(connection: sqlite3.Connection, table: str) -> bool:
        return bool(
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
        )

    def sessions(self) -> list[str]:
        values: set[str] = set()
        for connection in self.connections.values():
            for row in connection.execute(
                "SELECT trade_date FROM daily_observations "
                "WHERE source='fmp' AND symbol='SPY' AND close>0 AND volume>0 "
                "ORDER BY trade_date"
            ):
                values.add(str(row[0]))
        return sorted(values)

    def price_rows(self, symbol: str) -> list[sqlite3.Row]:
        """Return FMP daily rows, with the Oracle increment winning duplicates."""

        normalized = canonical_symbol(symbol)
        chosen: dict[str, sqlite3.Row] = {}
        for origin in ("base", "incremental"):
            connection = self.connections.get(origin)
            if connection is None:
                continue
            rows = connection.execute(
                "SELECT trade_date,open,high,low,close,adjusted_close,volume,vwap "
                "FROM daily_observations WHERE source='fmp' AND symbol=? "
                "ORDER BY trade_date",
                (normalized,),
            )
            for row in rows:
                chosen[str(row["trade_date"])] = row
        return [chosen[key] for key in sorted(chosen)]

    def latest_company_profiles(self) -> dict[str, dict[str, Any]]:
        """Return the latest FMP company profile per symbol.

        The base store is sealed, while a future Oracle increment may also carry
        profile facts.  As with prices, the increment wins duplicate symbols.
        """

        chosen: dict[str, dict[str, Any]] = {}
        for origin in ("base", "incremental"):
            connection = self.connections.get(origin)
            if connection is None or not self._has_table(
                connection, "fmp_training_facts"
            ):
                continue
            rows = connection.execute(
                """
                SELECT fact.symbol,fact.row_json
                FROM fmp_training_facts fact
                JOIN (
                  SELECT symbol,MAX(id) id
                  FROM fmp_training_facts
                  WHERE endpoint_id='company_information_company_profile_data'
                  GROUP BY symbol
                ) latest ON latest.id=fact.id
                """
            )
            for symbol, payload in rows:
                normalized = canonical_symbol(symbol)
                try:
                    parsed = json.loads(str(payload))
                except (TypeError, json.JSONDecodeError):
                    continue
                if normalized and isinstance(parsed, Mapping):
                    chosen[normalized] = dict(parsed)
        return chosen

    def active_us_exchange_stock_universe(
        self, price_date: str
    ) -> tuple[list[str], dict[str, Any]]:
        """Select the live U.S.-exchange individual-stock shadow universe.

        This is intentionally a live operational selection, not a historical
        point-in-time membership reconstruction.  A security must have a valid
        FMP close and volume on ``price_date``, an active latest FMP profile, be
        listed on NASDAQ/NYSE/AMEX, and not be classified as an ETF or fund.
        Foreign issuers and ADRs remain eligible when they are U.S.-listed.
        """

        priced: dict[str, str] = {}
        priced_by_origin: dict[str, int] = {}
        for origin in ("base", "incremental"):
            connection = self.connections.get(origin)
            if connection is None:
                continue
            count = 0
            for row in connection.execute(
                "SELECT symbol FROM daily_observations "
                "WHERE source='fmp' AND trade_date=? AND close>0 AND volume>0",
                (price_date,),
            ):
                symbol = canonical_symbol(row[0])
                if symbol:
                    priced[symbol] = origin
                    count += 1
            priced_by_origin[origin] = count

        profiles = self.latest_company_profiles()
        allowed_exchanges = {"NASDAQ", "NYSE", "AMEX"}
        exclusions = {
            "missing_profile": 0,
            "inactive": 0,
            "etf": 0,
            "fund": 0,
            "non_us_exchange": 0,
        }
        selected: list[str] = []
        for symbol in sorted(priced):
            profile = profiles.get(symbol)
            if profile is None:
                exclusions["missing_profile"] += 1
                continue
            if profile.get("isActivelyTrading") is not True:
                exclusions["inactive"] += 1
                continue
            if profile.get("isEtf") is True:
                exclusions["etf"] += 1
                continue
            if profile.get("isFund") is True:
                exclusions["fund"] += 1
                continue
            exchange = str(
                profile.get("exchangeShortName") or profile.get("exchange") or ""
            ).strip().upper()
            if exchange not in allowed_exchanges:
                exclusions["non_us_exchange"] += 1
                continue
            selected.append(symbol)

        return selected, {
            "selection_contract": (
                "LIVE_T_MINUS_1_VALID_PRICE_AND_VOLUME_ACTIVE_FMP_PROFILE_"
                "NASDAQ_NYSE_AMEX_EXCLUDING_ETF_AND_FUND"
            ),
            "selection_scope": "CURRENT_LIVE_GENERAL_UNIVERSE_SHADOW",
            "point_in_time_status": "NOT_HISTORICAL_PIT_DO_NOT_USE_FOR_OOS",
            "price_date": price_date,
            "priced_symbol_count": len(priced),
            "priced_rows_by_origin_before_precedence": priced_by_origin,
            "profile_symbol_count": len(profiles),
            "eligible_symbol_count": len(selected),
            "exclusions": exclusions,
        }

    def iter_flow_rows(self) -> Iterator[tuple]:
        """Yield normalized Massive flow records; increment wins duplicate keys."""

        incremental_keys: set[tuple[str, str]] = set()
        incremental = self.connections.get("incremental")
        if incremental is not None and self._has_table(
            incremental, "etf_flow_observations"
        ):
            for row in incremental.execute(
                "SELECT ticker,effective_date,processed_date,fund_flow,nav,"
                "shares_outstanding,available_at_date FROM etf_flow_observations "
                "WHERE provider='massive' ORDER BY effective_date,ticker"
            ):
                key = (canonical_symbol(row[0]), str(row[1]))
                incremental_keys.add(key)
                yield (
                    key[0],
                    key[1],
                    str(row[2]),
                    row[3],
                    row[4],
                    row[5],
                    str(row[6]),
                )
        base = self.connections["base"]
        for row in base.execute(
            "SELECT ticker,effective_date,processed_date,fund_flow,nav,"
            "shares_outstanding,available_at_date FROM etf_flow_observations "
            "WHERE provider='massive' ORDER BY effective_date,ticker"
        ):
            key = (canonical_symbol(row[0]), str(row[1]))
            if key in incremental_keys:
                continue
            yield (
                key[0],
                key[1],
                str(row[2]),
                row[3],
                row[4],
                row[5],
                str(row[6]),
            )

    @staticmethod
    def _snapshot_metadata_for(
        connection: sqlite3.Connection, origin: str
    ) -> Iterable[SnapshotMeta]:
        if not SourceBundle._has_table(connection, "etf_constituent_observations"):
            return []
        rows = connection.execute(
            """
            SELECT etf_ticker,effective_date,MAX(available_date) available_date,
                   COUNT(*) row_count,
                   SUM(CASE WHEN constituent_ticker IS NOT NULL
                                  AND constituent_ticker<>'' THEN 1 ELSE 0 END)
                       ticker_count,
                   SUM(CASE WHEN weight_percent<0 THEN 1 ELSE 0 END)
                       negative_weight_count,
                   SUM(CASE WHEN constituent_ticker IS NOT NULL
                                  AND constituent_ticker<>''
                                  AND weight_percent>0
                            THEN weight_percent ELSE 0 END)
                       positive_ticker_weight_sum
            FROM etf_constituent_observations
            WHERE provider='fmp'
            GROUP BY etf_ticker,effective_date
            ORDER BY etf_ticker,effective_date
            """
        )
        result = []
        for row in rows:
            count = int(row["row_count"] or 0)
            ticker_count = int(row["ticker_count"] or 0)
            negative = int(row["negative_weight_count"] or 0)
            positive_weight = float(row["positive_ticker_weight_sum"] or 0.0)
            eligible = bool(
                count >= 5
                and ticker_count / count >= 0.80
                and 70.0 <= positive_weight <= 130.0
                and negative == 0
            )
            result.append(
                SnapshotMeta(
                    origin=origin,
                    etf_ticker=canonical_symbol(row["etf_ticker"]),
                    effective_date=str(row["effective_date"]),
                    provider_available_date=str(row["available_date"]),
                    row_count=count,
                    ticker_count=ticker_count,
                    negative_weight_count=negative,
                    positive_ticker_weight_sum=positive_weight,
                    eligible=eligible,
                )
            )
        return result

    def snapshot_metadata(self) -> list[SnapshotMeta]:
        """Return snapshot metadata with incremental rows replacing base keys."""

        chosen: dict[tuple[str, str], SnapshotMeta] = {}
        for origin in ("base", "incremental"):
            connection = self.connections.get(origin)
            if connection is None:
                continue
            for item in self._snapshot_metadata_for(connection, origin):
                chosen[(item.etf_ticker, item.effective_date)] = item
        return [chosen[key] for key in sorted(chosen)]

    def snapshot_holdings(self, metadata: SnapshotMeta) -> dict[str, float]:
        connection = self.connections[metadata.origin]
        result: dict[str, float] = {}
        for row in connection.execute(
            "SELECT constituent_ticker,weight_percent "
            "FROM etf_constituent_observations "
            "WHERE provider='fmp' AND etf_ticker=? AND effective_date=? "
            "AND constituent_ticker IS NOT NULL AND constituent_ticker<>'' "
            "AND weight_percent>0",
            (metadata.etf_ticker, metadata.effective_date),
        ):
            symbol = canonical_symbol(row[0])
            if symbol:
                result[symbol] = result.get(symbol, 0.0) + float(row[1])
        return result

    def fmp_facts(self, symbol: str, endpoints: Sequence[str]) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in endpoints)
        if not placeholders:
            return []
        return list(
            self.connections["base"].execute(
                "SELECT endpoint_id,event_date,available_date,row_json "
                f"FROM fmp_training_facts WHERE symbol=? AND endpoint_id IN ({placeholders}) "
                "ORDER BY available_date,event_date,endpoint_id,id",
                (canonical_symbol(symbol), *endpoints),
            )
        )

    def source_fingerprint(self) -> dict:
        result = {}
        for origin, connection in self.connections.items():
            path = (
                self.base_database
                if origin == "base"
                else self.incremental_database
            )
            assert path is not None
            stat = path.stat()
            tables = {}
            for table, date_column in (
                ("daily_observations", "trade_date"),
                ("etf_flow_observations", "effective_date"),
                ("etf_constituent_observations", "effective_date"),
                ("fmp_training_facts", "available_date"),
            ):
                if not self._has_table(connection, table):
                    continue
                row = connection.execute(
                    f"SELECT COUNT(*),MIN({date_column}),MAX({date_column}) FROM {table}"
                ).fetchone()
                tables[table] = {
                    "rows": int(row[0]),
                    "min_date": row[1],
                    "max_date": row[2],
                }
            result[origin] = {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "tables": tables,
            }
        return result


def parse_json_row(value: object) -> dict:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
