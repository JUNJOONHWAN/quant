"""Read-only historical analogue search and realised-outcome statistics.

The accepted LoRA identifies the current point-in-time pattern.  This module
does not train a model: it finds comparable examples in the sealed SFT
materialization and calculates what happened afterwards from the price ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ANALOG_SCHEMA_VERSION = "quant.historical_analog_forecast.v1"
INDEX_SCHEMA_VERSION = "quant.historical_analog_index.v1"
HORIZONS = (5, 20, 60)
DEFAULT_EXAMPLE_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/"
    "qwen3_8b_candidate_v3/materialization_state.sqlite3"
)
DEFAULT_PRICE_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/"
    "daily_observations.sqlite3"
)
DEFAULT_ANALOG_INDEX = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/analog/"
    "historical_analog_index.sqlite3"
)

FEATURE_SCALES = {
    "return_1_session_pct": 4.0,
    "return_5_session_pct": 8.0,
    "return_20_session_pct": 18.0,
    "annualized_realized_volatility_pct": 25.0,
    "max_drawdown_in_packet_pct": 12.0,
    "latest_robust_zscore": 3.0,
    "latest_flow_to_assets_pct": 2.0,
    "net_weighted_flow_rate_contribution_pct": 2.0,
    "flow_breadth": 1.0,
    "log_median_dollar_volume": 4.0,
}
MISSING_FEATURE_PENALTY = 0.75


class HistoricalAnalogError(RuntimeError):
    """The historical-memory layer cannot produce an auditable result."""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _log_dollar_volume(value: Any) -> float | None:
    parsed = _number(value)
    if parsed is None or parsed < 0:
        return None
    return math.log10(1.0 + parsed)


def _breadth(flow: Mapping[str, Any]) -> float | None:
    eligible = _number(flow.get("eligible_etf_count"))
    positive = _number(flow.get("positive_etf_count"))
    negative = _number(flow.get("negative_etf_count"))
    if eligible is None or eligible <= 0 or positive is None or negative is None:
        return None
    return (positive - negative) / eligible


def feature_row_from_judgement(judgement: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the same point-in-time feature family from an 8B judgement."""

    facts = judgement.get("facts")
    interpretation = judgement.get("interpretation")
    if not isinstance(facts, Mapping) or not isinstance(interpretation, Mapping):
        raise HistoricalAnalogError("8B judgement has no deterministic facts contract")
    price = facts.get("price")
    own_flow = facts.get("etf_flow")
    exposure = facts.get("etf_flow_to_constituent")
    liquidity = facts.get("liquidity")
    price = price if isinstance(price, Mapping) else {}
    own_flow = own_flow if isinstance(own_flow, Mapping) else {}
    exposure = exposure if isinstance(exposure, Mapping) else {}
    liquidity = liquidity if isinstance(liquidity, Mapping) else {}
    return {
        "symbol": str(facts.get("symbol") or ""),
        "as_of_date": str(facts.get("as_of_date") or ""),
        "task_type": str(interpretation.get("task_type") or ""),
        "regime": str(judgement.get("regime") or ""),
        "price_signal": str(interpretation.get("price_signal") or ""),
        "flow_signal": str(interpretation.get("etf_flow_signal") or ""),
        "return_1_session_pct": _number(price.get("return_1_session_pct")),
        "return_5_session_pct": _number(price.get("return_5_session_pct")),
        "return_20_session_pct": _number(price.get("return_20_session_pct")),
        "annualized_realized_volatility_pct": _number(
            price.get("annualized_realized_volatility_pct")
        ),
        "max_drawdown_in_packet_pct": _number(
            price.get("max_drawdown_in_packet_pct")
        ),
        "latest_robust_zscore": _number(own_flow.get("latest_robust_zscore")),
        "latest_flow_to_assets_pct": _number(
            own_flow.get("latest_flow_to_assets_pct")
        ),
        "net_weighted_flow_rate_contribution_pct": _number(
            exposure.get("net_weighted_flow_rate_contribution_pct")
        ),
        "flow_breadth": _breadth(exposure),
        "eligible_etf_count": _number(exposure.get("eligible_etf_count")),
        "log_median_dollar_volume": _log_dollar_volume(
            liquidity.get("median_dollar_volume")
        ),
        "quality_status": str(facts.get("quality_status") or ""),
    }


def _feature_row_from_encoded(encoded: Mapping[str, Any]) -> dict[str, Any]:
    try:
        response = json.loads(str(encoded["response"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HistoricalAnalogError("SFT example response is invalid") from exc
    row = feature_row_from_judgement(response)
    metadata = encoded.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    row.update(
        {
            "example_id": str(encoded.get("example_id") or ""),
            "symbol": str(metadata.get("symbol") or row["symbol"]),
            "as_of_date": str(metadata.get("as_of_date") or row["as_of_date"]),
            "task_type": str(metadata.get("task_type") or row["task_type"]),
            "facts_json": json.dumps(
                response.get("facts") or {},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    if not row["example_id"] or not row["symbol"] or not row["as_of_date"]:
        raise HistoricalAnalogError("SFT example identity is incomplete")
    return row


def _open_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise HistoricalAnalogError(f"read-only source database is missing: {resolved}")
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _index_is_current(index_path: Path, source_identity: Mapping[str, Any]) -> bool:
    if not index_path.is_file():
        return False
    try:
        connection = sqlite3.connect(index_path)
        rows = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
    except (sqlite3.DatabaseError, OSError):
        return False
    finally:
        if "connection" in locals():
            connection.close()
    return (
        rows.get("schema_version") == INDEX_SCHEMA_VERSION
        and rows.get("source_identity_json")
        == json.dumps(source_identity, sort_keys=True, separators=(",", ":"))
    )


def build_analog_index(
    *, example_database: Path, index_path: Path, force: bool = False
) -> dict[str, Any]:
    """Build a disposable compact index without mutating the training corpus."""

    example_database = example_database.expanduser().resolve()
    index_path = index_path.expanduser().resolve()
    identity = _source_identity(example_database)
    if not force and _index_is_current(index_path, identity):
        with sqlite3.connect(index_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM analog_examples").fetchone()[0]
        return {"path": str(index_path), "row_count": count, "rebuilt": False}

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(f".{index_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = sqlite3.connect(temporary)
    output.execute(
        """
        CREATE TABLE analog_examples (
            example_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            task_type TEXT NOT NULL,
            regime TEXT NOT NULL,
            price_signal TEXT NOT NULL,
            flow_signal TEXT NOT NULL,
            return_1_session_pct REAL,
            return_5_session_pct REAL,
            return_20_session_pct REAL,
            annualized_realized_volatility_pct REAL,
            max_drawdown_in_packet_pct REAL,
            latest_robust_zscore REAL,
            latest_flow_to_assets_pct REAL,
            net_weighted_flow_rate_contribution_pct REAL,
            flow_breadth REAL,
            eligible_etf_count REAL,
            log_median_dollar_volume REAL,
            quality_status TEXT NOT NULL,
            facts_json TEXT NOT NULL
        )
        """
    )
    output.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    columns = (
        "example_id", "symbol", "as_of_date", "task_type", "regime",
        "price_signal", "flow_signal", "return_1_session_pct",
        "return_5_session_pct", "return_20_session_pct",
        "annualized_realized_volatility_pct", "max_drawdown_in_packet_pct",
        "latest_robust_zscore", "latest_flow_to_assets_pct",
        "net_weighted_flow_rate_contribution_pct", "flow_breadth",
        "eligible_etf_count", "log_median_dollar_volume", "quality_status",
        "facts_json",
    )
    placeholders = ",".join("?" for _ in columns)
    inserted = 0
    source = _open_read_only(example_database)
    try:
        cursor = source.execute("SELECT encoded_json FROM examples ORDER BY example_id")
        batch: list[tuple[Any, ...]] = []
        for raw in cursor:
            try:
                encoded = json.loads(str(raw[0]))
                row = _feature_row_from_encoded(encoded)
            except (json.JSONDecodeError, HistoricalAnalogError):
                continue
            batch.append(tuple(row.get(column) for column in columns))
            if len(batch) >= 1000:
                output.executemany(
                    f"INSERT INTO analog_examples ({','.join(columns)}) VALUES ({placeholders})",
                    batch,
                )
                inserted += len(batch)
                batch.clear()
        if batch:
            output.executemany(
                f"INSERT INTO analog_examples ({','.join(columns)}) VALUES ({placeholders})",
                batch,
            )
            inserted += len(batch)
        output.execute(
            "CREATE INDEX analog_lookup ON analog_examples(task_type, regime, as_of_date)"
        )
        output.execute(
            "CREATE INDEX analog_symbol_date ON analog_examples(symbol, as_of_date)"
        )
        output.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", INDEX_SCHEMA_VERSION),
                (
                    "source_identity_json",
                    json.dumps(identity, sort_keys=True, separators=(",", ":")),
                ),
                ("row_count", str(inserted)),
            ],
        )
        output.commit()
    except Exception:
        output.close()
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        source.close()
    output.close()
    os.replace(temporary, index_path)
    return {"path": str(index_path), "row_count": inserted, "rebuilt": True}


def feature_distance(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    """Fixed, deterministic normalized distance over point-in-time features."""

    total = 0.0
    weight = 0.0
    for name, scale in FEATURE_SCALES.items():
        left = _number(current.get(name))
        right = _number(candidate.get(name))
        weight += 1.0
        if left is None and right is None:
            continue
        if left is None or right is None:
            total += MISSING_FEATURE_PENALTY**2
            continue
        total += ((left - right) / scale) ** 2
    for name in ("price_signal", "flow_signal"):
        weight += 0.5
        if str(current.get(name) or "") != str(candidate.get(name) or ""):
            total += 0.5
    return math.sqrt(total / max(weight, 1.0))


def _price_rows(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
) -> tuple[str, list[sqlite3.Row]]:
    for source in ("fmp", "massive"):
        rows = connection.execute(
            """
            SELECT trade_date, close, adjusted_close
            FROM daily_observations
            WHERE source=? AND symbol=? AND trade_date>=? AND trade_date<=?
            ORDER BY trade_date
            LIMIT 80
            """,
            (source, symbol, start_date, end_date),
        ).fetchall()
        if rows and str(rows[0]["trade_date"]) == start_date:
            return source, rows
    return "", []


def _close(row: Mapping[str, Any]) -> float | None:
    adjusted = _number(row["adjusted_close"])
    if adjusted is not None and adjusted > 0:
        return adjusted
    close = _number(row["close"])
    return close if close is not None and close > 0 else None


def _return_pct(start: float, end: float) -> float:
    return (end / start - 1.0) * 100.0


def _exact_benchmark_return(
    connection: sqlite3.Connection, *, start_date: str, end_date: str
) -> float | None:
    for source in ("fmp", "massive"):
        rows = connection.execute(
            """
            SELECT trade_date, close, adjusted_close
            FROM daily_observations
            WHERE source=? AND symbol='SPY' AND trade_date IN (?, ?)
            ORDER BY trade_date
            """,
            (source, start_date, end_date),
        ).fetchall()
        if len(rows) == 2 and [str(row["trade_date"]) for row in rows] == [
            start_date,
            end_date,
        ]:
            start = _close(rows[0])
            end = _close(rows[1])
            if start is not None and end is not None:
                return _return_pct(start, end)
    return None


def realised_outcomes(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    analog_date: str,
    cutoff_date: str,
) -> dict[int, dict[str, Any]]:
    """Calculate only outcomes fully observable by cutoff_date."""

    source, rows = _price_rows(
        connection,
        symbol=symbol,
        start_date=analog_date,
        end_date=cutoff_date,
    )
    if not rows:
        return {}
    base = _close(rows[0])
    if base is None:
        return {}
    output: dict[int, dict[str, Any]] = {}
    for horizon in HORIZONS:
        if len(rows) <= horizon:
            continue
        end = _close(rows[horizon])
        path = [_close(row) for row in rows[1 : horizon + 1]]
        if end is None or any(value is None for value in path):
            continue
        values = [float(value) for value in path if value is not None]
        end_date = str(rows[horizon]["trade_date"])
        if end_date > cutoff_date:
            raise HistoricalAnalogError("walk-forward leakage gate failed")
        absolute = _return_pct(base, end)
        benchmark = _exact_benchmark_return(
            connection,
            start_date=analog_date,
            end_date=end_date,
        )
        output[horizon] = {
            "horizon_sessions": horizon,
            "outcome_end_date": end_date,
            "return_pct": absolute,
            "spy_return_pct": benchmark,
            "spy_excess_return_pct": (
                absolute - benchmark if benchmark is not None else None
            ),
            "maximum_favorable_excursion_pct": max(
                _return_pct(base, value) for value in values
            ),
            "maximum_adverse_excursion_pct": min(
                _return_pct(base, value) for value in values
            ),
            "price_source": source,
        }
    return output


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def aggregate_outcomes(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    returns = [float(row["return_pct"]) for row in rows]
    excess = [
        float(row["spy_excess_return_pct"])
        for row in rows
        if row.get("spy_excess_return_pct") is not None
    ]
    favorable = [float(row["maximum_favorable_excursion_pct"]) for row in rows]
    adverse = [float(row["maximum_adverse_excursion_pct"]) for row in rows]
    return {
        "sample_count": len(returns),
        "positive_probability_pct": _round(
            100.0 * sum(value > 0 for value in returns) / len(returns)
        ) if returns else None,
        "mean_return_pct": _round(statistics.fmean(returns)) if returns else None,
        "median_return_pct": _round(statistics.median(returns)) if returns else None,
        "p10_return_pct": _round(_percentile(returns, 0.10)),
        "p25_return_pct": _round(_percentile(returns, 0.25)),
        "p75_return_pct": _round(_percentile(returns, 0.75)),
        "p90_return_pct": _round(_percentile(returns, 0.90)),
        "spy_excess_sample_count": len(excess),
        "spy_outperformance_probability_pct": _round(
            100.0 * sum(value > 0 for value in excess) / len(excess)
        ) if excess else None,
        "median_spy_excess_return_pct": _round(statistics.median(excess))
        if excess else None,
        "median_maximum_favorable_excursion_pct": _round(
            statistics.median(favorable)
        ) if favorable else None,
        "median_maximum_adverse_excursion_pct": _round(
            statistics.median(adverse)
        ) if adverse else None,
    }


@dataclass
class HistoricalAnalogEngine:
    example_database: Path = DEFAULT_EXAMPLE_DATABASE
    price_database: Path = DEFAULT_PRICE_DATABASE
    index_path: Path = DEFAULT_ANALOG_INDEX
    neighbor_limit: int = 80
    per_symbol_limit: int = 2

    def forecast(
        self, *, judgement: Mapping[str, Any], analysis_as_of_date: str
    ) -> dict[str, Any]:
        current = feature_row_from_judgement(judgement)
        if current["as_of_date"] and current["as_of_date"] != analysis_as_of_date:
            raise HistoricalAnalogError(
                "8B facts as-of date does not match individual analysis date"
            )
        index_status = build_analog_index(
            example_database=self.example_database,
            index_path=self.index_path,
        )
        index = sqlite3.connect(self.index_path)
        index.row_factory = sqlite3.Row
        candidates = index.execute(
            """
            SELECT * FROM analog_examples
            WHERE task_type=? AND regime=? AND as_of_date<?
            ORDER BY as_of_date, symbol, example_id
            """,
            (current["task_type"], current["regime"], analysis_as_of_date),
        ).fetchall()
        index.close()
        if not candidates:
            raise HistoricalAnalogError(
                "no historical examples share the current task type and learned regime"
            )
        ranked = sorted(
            (
                (feature_distance(current, dict(row)), dict(row))
                for row in candidates
            ),
            key=lambda value: (
                value[0], value[1]["as_of_date"], value[1]["symbol"],
                value[1]["example_id"],
            ),
        )
        price = _open_read_only(self.price_database)
        selected: list[dict[str, Any]] = []
        per_symbol: dict[str, int] = {}
        try:
            for distance, row in ranked:
                symbol = str(row["symbol"])
                if per_symbol.get(symbol, 0) >= self.per_symbol_limit:
                    continue
                outcomes = realised_outcomes(
                    price,
                    symbol=symbol,
                    analog_date=str(row["as_of_date"]),
                    cutoff_date=analysis_as_of_date,
                )
                if not outcomes:
                    continue
                selected.append(
                    {
                        "example_id": row["example_id"],
                        "symbol": symbol,
                        "as_of_date": row["as_of_date"],
                        "distance": round(distance, 8),
                        "price_signal": row["price_signal"],
                        "flow_signal": row["flow_signal"],
                        "outcomes": outcomes,
                    }
                )
                per_symbol[symbol] = per_symbol.get(symbol, 0) + 1
                if len(selected) >= self.neighbor_limit:
                    break
        finally:
            price.close()
        if not selected:
            raise HistoricalAnalogError(
                "historical examples exist but no fully observed outcome is available"
            )
        horizon_statistics: dict[str, Any] = {}
        for horizon in HORIZONS:
            observed = [
                row["outcomes"][horizon]
                for row in selected
                if horizon in row["outcomes"]
            ]
            horizon_statistics[str(horizon)] = {
                "horizon_sessions": horizon,
                **aggregate_outcomes(observed),
            }
        top_analogs = []
        for row in selected[:12]:
            top_analogs.append(
                {
                    **{key: row[key] for key in (
                        "example_id", "symbol", "as_of_date", "distance",
                        "price_signal", "flow_signal",
                    )},
                    "outcomes": {
                        str(horizon): outcome
                        for horizon, outcome in row["outcomes"].items()
                    },
                }
            )
        payload = {
            "schema_version": ANALOG_SCHEMA_VERSION,
            "analysis_as_of_date": analysis_as_of_date,
            "current_pattern": {
                key: current.get(key)
                for key in (
                    "symbol", "task_type", "regime", "price_signal", "flow_signal"
                )
            },
            "matching_policy": {
                "task_type": "exact",
                "regime": "exact",
                "distance": "fixed_scaled_point_in_time_features_v1",
                "neighbor_limit": self.neighbor_limit,
                "per_symbol_limit": self.per_symbol_limit,
                "future_visibility_gate": (
                    "outcome_end_date_lte_analysis_as_of_date"
                ),
            },
            "candidate_count": len(candidates),
            "selected_analog_count": len(selected),
            "horizon_statistics": horizon_statistics,
            "top_analogs": top_analogs,
            "sources": {
                "training_materialization": _source_identity(self.example_database),
                "price_ledger": _source_identity(self.price_database),
                "derived_index": index_status,
                "benchmark": "SPY",
            },
        }
        payload["sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return payload
