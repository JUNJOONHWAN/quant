"""Build the full point-in-time SPY/QQQ constituent Forecast v2 panel."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict, deque
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .contracts import (
    FEATURE_GROUPS,
    IDENTITY_COLUMNS,
    PANEL_SCHEMA_VERSION,
    TARGET_COLUMNS,
    TIMING_CONTRACT,
    TimingRow,
)
from .features import (
    FUNDAMENTAL_ENDPOINTS,
    compute_fundamental_features,
    compute_price_features,
    next_session_strictly_after,
    price_frame,
)
from .flow import FlowCache, aggregate_symbol_flow, benchmark_flow_features, build_flow_cache
from .index_membership import (
    load_membership_evidence,
    reconstruct_memberships,
    validate_against_holdings,
)
from .io_utils import sha256_file, utc_now, write_json_atomic
from .source import SnapshotMeta, SourceBundle


DEFAULT_BASE_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/"
    "daily_observations.sqlite3"
)
DEFAULT_INCREMENTAL_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/incremental/"
    "normalized/daily_observations.sqlite3"
)
DEFAULT_INDEX_EVIDENCE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/evidence/"
    "fmp_spy_qqq_index_history_20260827.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2"
)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


FEATURE_COLUMNS = _unique(
    column for columns in FEATURE_GROUPS.values() for column in columns
)
PANEL_COLUMNS = _unique(
    (*IDENTITY_COLUMNS, "reference_close", *FEATURE_COLUMNS, *TARGET_COLUMNS)
)
TEXT_COLUMNS = {
    "signal_date",
    "price_date",
    "flow_date",
    "legacy_flow_date",
    "symbol",
    "benchmark",
    "membership_source",
}
INTEGER_COLUMNS = {"is_spy_member", "is_qqq_member"}


def _sql_identifier(value: str) -> str:
    if not value.replace("_", "").isalnum():
        raise ValueError(f"unsafe SQL identifier: {value}")
    return f'"{value}"'


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _create_panel_database(path: Path) -> sqlite3.Connection:
    definitions = []
    for column in PANEL_COLUMNS:
        if column in TEXT_COLUMNS:
            data_type = "TEXT"
        elif column in INTEGER_COLUMNS:
            data_type = "INTEGER"
        else:
            data_type = "REAL"
        definitions.append(f"{_sql_identifier(column)} {data_type}")
    connection = sqlite3.connect(path)
    connection.executescript(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY;"
    )
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.execute(
        f"CREATE TABLE panel({','.join(definitions)},"
        "PRIMARY KEY(signal_date,symbol)) WITHOUT ROWID"
    )
    connection.execute(
        "INSERT INTO metadata VALUES('schema_version',?)", (PANEL_SCHEMA_VERSION,)
    )
    connection.execute(
        "INSERT INTO metadata VALUES('timing_contract',?)", (TIMING_CONTRACT,)
    )
    return connection


def _append_live_session(sessions: list[str], live_signal_date: str | None) -> list[str]:
    if not live_signal_date:
        return sessions
    parsed = date.fromisoformat(live_signal_date)
    if live_signal_date in sessions:
        return sessions
    if live_signal_date <= sessions[-1]:
        raise ValueError("live signal date inside history is absent from SPY sessions")
    if parsed.weekday() >= 5:
        raise ValueError("live signal date is a weekend; supply a US trading session")
    if (parsed - date.fromisoformat(sessions[-1])).days > 4:
        raise ValueError("live signal date is too far beyond the last observed session")
    return [*sessions, live_signal_date]


def make_timing_rows(
    sessions: Sequence[str], start_date: str, end_date: str
) -> list[TimingRow]:
    result = []
    for signal_position in range(3, len(sessions)):
        signal_date = sessions[signal_position]
        if signal_date < start_date or signal_date > end_date:
            continue
        result.append(
            TimingRow(
                signal_date=signal_date,
                price_date=sessions[signal_position - 1],
                flow_date=sessions[signal_position - 2],
                price_position=signal_position - 1,
            )
        )
    return result


def _legacy_timing_rows(
    timing_rows: Sequence[TimingRow], sessions: Sequence[str]
) -> list[TimingRow]:
    positions = {value: index for index, value in enumerate(sessions)}
    return [
        TimingRow(
            signal_date=row.signal_date,
            price_date=row.price_date,
            flow_date=sessions[positions[row.signal_date] - 3],
            price_position=row.price_position,
        )
        for row in timing_rows
    ]


def _membership_symbols(
    memberships: Mapping[str, Mapping[str, frozenset[str]]],
    timing_rows: Sequence[TimingRow],
) -> tuple[list[str], dict[str, tuple[frozenset[str], frozenset[str]]]]:
    by_price_date = {}
    union: set[str] = set()
    for timing in timing_rows:
        spy = memberships["SPY"].get(timing.price_date, frozenset())
        qqq = memberships["QQQ"].get(timing.price_date, frozenset())
        by_price_date[timing.price_date] = (spy, qqq)
        union.update(spy)
        union.update(qqq)
    return sorted(union), by_price_date


def _row_values(record: Mapping[str, object]) -> tuple[object, ...]:
    values: list[object] = []
    for column in PANEL_COLUMNS:
        value = record.get(column)
        if column in TEXT_COLUMNS:
            values.append(None if value is None else str(value))
        elif column in INTEGER_COLUMNS:
            values.append(int(value or 0))
        else:
            values.append(_finite_or_none(value))
    return tuple(values)


def _price_phase(
    connection: sqlite3.Connection,
    source: SourceBundle,
    sessions: Sequence[str],
    timing_rows: Sequence[TimingRow],
    symbols: Sequence[str],
    memberships_by_date: Mapping[str, tuple[frozenset[str], frozenset[str]]],
) -> dict[str, Any]:
    spy_frame = price_frame(source.price_rows("SPY"), sessions)
    qqq_frame = price_frame(source.price_rows("QQQ"), sessions)
    insert_sql = (
        f"INSERT INTO panel({','.join(_sql_identifier(c) for c in PANEL_COLUMNS)}) "
        f"VALUES({','.join('?' for _ in PANEL_COLUMNS)})"
    )
    inserted = 0
    excluded_price_history = 0
    symbols_without_price = 0
    batch: list[tuple[object, ...]] = []
    legacy_dates = {
        row.signal_date: sessions[bisect.bisect_left(sessions, row.signal_date) - 3]
        for row in timing_rows
    }
    for symbol_number, symbol in enumerate(symbols, 1):
        rows = source.price_rows(symbol)
        if not rows:
            symbols_without_price += 1
            continue
        frame = price_frame(rows, sessions)
        spy_features = compute_price_features(frame, spy_frame, spy_frame, qqq_frame)
        qqq_features = compute_price_features(frame, qqq_frame, spy_frame, qqq_frame)
        fundamentals = compute_fundamental_features(
            source.fmp_facts(symbol, FUNDAMENTAL_ENDPOINTS), sessions
        )
        for timing in timing_rows:
            spy_members, qqq_members = memberships_by_date[timing.price_date]
            is_spy = symbol in spy_members
            is_qqq = symbol in qqq_members
            if not is_spy and not is_qqq:
                continue
            benchmark = "QQQ" if is_qqq else "SPY"
            feature_frame = qqq_features if benchmark == "QQQ" else spy_features
            price_values = feature_frame.loc[timing.price_date]
            reference_close = frame.at[timing.price_date, "close"]
            if not math.isfinite(reference_close) or not math.isfinite(
                float(price_values.get("ret_120d", np.nan))
            ):
                excluded_price_history += 1
                continue
            record: dict[str, object] = {
                "signal_date": timing.signal_date,
                "price_date": timing.price_date,
                "flow_date": timing.flow_date,
                "legacy_flow_date": legacy_dates[timing.signal_date],
                "symbol": symbol,
                "benchmark": benchmark,
                "is_spy_member": int(is_spy),
                "is_qqq_member": int(is_qqq),
                "membership_source": "fmp_index_history_reverse_v1",
                "reference_close": reference_close,
            }
            for column in FEATURE_GROUPS["price"]:
                if column not in INTEGER_COLUMNS:
                    record[column] = price_values.get(column)
            fundamental_values = fundamentals.loc[timing.price_date]
            for column in FEATURE_GROUPS["fmp_fundamentals"]:
                record[column] = fundamental_values.get(column)
            for column in TARGET_COLUMNS:
                record[column] = price_values.get(column)
            batch.append(_row_values(record))
            inserted += 1
            if len(batch) >= 10_000:
                connection.executemany(insert_sql, batch)
                batch.clear()
        if symbol_number % 100 == 0:
            connection.commit()
    if batch:
        connection.executemany(insert_sql, batch)
    connection.commit()
    return {
        "symbol_count": len(symbols),
        "symbols_without_price": symbols_without_price,
        "inserted_rows": inserted,
        "excluded_insufficient_price_history": excluded_price_history,
    }


def _snapshot_events(
    metadata: Sequence[SnapshotMeta], sessions: Sequence[str]
) -> tuple[list[tuple[str, SnapshotMeta]], dict[str, int]]:
    result = []
    reasons = defaultdict(int)
    for item in metadata:
        if not item.eligible:
            reasons["ineligible_snapshot"] += 1
            continue
        available = next_session_strictly_after(
            sessions, item.provider_available_date, 1
        )
        if not available:
            reasons["availability_after_calendar"] += 1
            continue
        if item.effective_date > available:
            reasons["effective_after_availability"] += 1
            continue
        result.append((available, item))
    result.sort(key=lambda pair: (pair[0], pair[1].effective_date, pair[1].etf_ticker))
    return result, dict(reasons)


class _ExposureState:
    def __init__(self, source: SourceBundle) -> None:
        self.source = source
        self.current: dict[str, tuple[str, dict[str, float]]] = {}
        self.reverse: dict[str, dict[str, float]] = defaultdict(dict)
        self.applied_snapshots = 0
        self.loaded_holding_rows = 0
        self.skipped_older_snapshot = 0

    def apply(self, item: SnapshotMeta) -> None:
        previous = self.current.get(item.etf_ticker)
        if previous and item.effective_date < previous[0]:
            self.skipped_older_snapshot += 1
            return
        if previous:
            for symbol in previous[1]:
                exposures = self.reverse.get(symbol)
                if exposures is not None:
                    exposures.pop(item.etf_ticker, None)
                    if not exposures:
                        self.reverse.pop(symbol, None)
        holdings = self.source.snapshot_holdings(item)
        self.current[item.etf_ticker] = (item.effective_date, holdings)
        for symbol, weight in holdings.items():
            self.reverse[symbol][item.etf_ticker] = weight
        self.applied_snapshots += 1
        self.loaded_holding_rows += len(holdings)


def _rolling_values(history: deque[float]) -> tuple[float, float, float]:
    values = np.asarray(history, dtype=float)
    finite = np.isfinite(values)
    tail5 = values[-5:]
    tail20 = values[-20:]
    net5 = float(np.nansum(tail5)) if np.isfinite(tail5).sum() >= 3 else math.nan
    net20 = float(np.nansum(tail20)) if np.isfinite(tail20).sum() >= 10 else math.nan
    window = values[finite]
    if len(window) >= 20 and float(np.std(window, ddof=1)) > 0:
        z60 = float((values[-1] - np.mean(window)) / np.std(window, ddof=1))
    else:
        z60 = math.nan
    return net5, net20, z60


def _benchmark_aliases(
    values: Mapping[str, float], benchmark: str, *, legacy: bool = False
) -> dict[str, float]:
    prefix = benchmark.lower()
    if legacy:
        return {
            "benchmark_flow_rate_t3": values[f"{prefix}_flow_rate_t3"],
            "benchmark_flow_rate_5d_t3_cutoff": values[
                f"{prefix}_flow_rate_5d_t3_cutoff"
            ],
            "benchmark_flow_rate_20d_t3_cutoff": values[
                f"{prefix}_flow_rate_20d_t3_cutoff"
            ],
            "benchmark_flow_z60_t3_cutoff": values[
                f"{prefix}_flow_z60_t3_cutoff"
            ],
        }
    return {
        "benchmark_flow_rate_t2": values[f"{prefix}_flow_rate_t2"],
        "benchmark_flow_rate_5d": values[f"{prefix}_flow_rate_5d"],
        "benchmark_flow_rate_20d": values[f"{prefix}_flow_rate_20d"],
        "benchmark_flow_z60": values[f"{prefix}_flow_z60"],
    }


def _flow_phase(
    connection: sqlite3.Connection,
    source: SourceBundle,
    cache: FlowCache,
    sessions: Sequence[str],
    timing_rows: Sequence[TimingRow],
    memberships_by_date: Mapping[str, tuple[frozenset[str], frozenset[str]]],
    metadata: Sequence[SnapshotMeta],
) -> dict[str, Any]:
    legacy_rows = _legacy_timing_rows(timing_rows, sessions)
    benchmark_t2 = benchmark_flow_features(cache, timing_rows, sessions)
    benchmark_t3 = benchmark_flow_features(
        cache, legacy_rows, sessions, suffix="_t3_cutoff"
    )
    events, excluded = _snapshot_events(metadata, sessions)
    event_position = 0
    exposure_state = _ExposureState(source)
    histories: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=60))
    last_position: dict[str, int] = {}
    session_positions = {value: index for index, value in enumerate(sessions)}
    update_columns = _unique(
        (
            *FEATURE_GROUPS["benchmark_flow"],
            *FEATURE_GROUPS["legacy_t3_flow"],
            *FEATURE_GROUPS["all_etf_flow"],
        )
    )
    update_sql = (
        "UPDATE panel SET "
        + ",".join(f"{_sql_identifier(column)}=?" for column in update_columns)
        + " WHERE signal_date=? AND symbol=?"
    )
    updated_rows = 0
    flow_missing_dates = []
    per_date_coverage = []
    for timing_number, timing in enumerate(timing_rows, 1):
        while event_position < len(events) and events[event_position][0] <= timing.price_date:
            exposure_state.apply(events[event_position][1])
            event_position += 1
        flows = cache.for_date(timing.flow_date, timing.signal_date)
        if "SPY" not in flows or "QQQ" not in flows:
            flow_missing_dates.append(timing.signal_date)
        price_rows = list(
            connection.execute(
                "SELECT symbol,benchmark,ret_5d FROM panel WHERE signal_date=?",
                (timing.signal_date,),
            )
        )
        raw_by_symbol: dict[str, dict[str, float]] = {}
        for symbol, _, _ in price_rows:
            current_position = session_positions[timing.signal_date]
            if last_position.get(symbol, current_position - 1) != current_position - 1:
                histories[symbol].clear()
            raw = aggregate_symbol_flow(exposure_state.reverse.get(symbol, {}), flows)
            histories[symbol].append(raw["all_etf_flow_net"])
            last_position[symbol] = current_position
            net5, net20, z60 = _rolling_values(histories[symbol])
            raw["all_etf_flow_net_5d"] = net5
            raw["all_etf_flow_net_20d"] = net20
            raw["all_etf_flow_net_z60"] = z60
            raw_by_symbol[str(symbol)] = raw
        if raw_by_symbol:
            rank_frame = pd.DataFrame.from_dict(raw_by_symbol, orient="index")
            for source_column, target_column in (
                ("all_etf_flow_net", "all_etf_flow_rank"),
                ("all_etf_flow_breadth", "all_etf_flow_breadth_rank"),
                ("all_etf_flow_weight_coverage", "all_etf_flow_coverage_rank"),
            ):
                rank_frame[target_column] = rank_frame[source_column].rank(
                    method="average", pct=True
                )
                for symbol, value in rank_frame[target_column].items():
                    raw_by_symbol[str(symbol)][target_column] = float(value)
        t2 = benchmark_t2[timing.price_date]
        t3 = benchmark_t3[timing.price_date]
        updates = []
        coverage_values = []
        for symbol, benchmark, ret_5d in price_rows:
            values: dict[str, float] = {**t2, **t3}
            values.update(_benchmark_aliases(t2, str(benchmark)))
            values.update(_benchmark_aliases(t3, str(benchmark), legacy=True))
            values["price_flow_interaction_5d"] = (
                float(ret_5d) * values["benchmark_flow_rate_5d"]
                if ret_5d is not None
                and math.isfinite(float(ret_5d))
                and math.isfinite(values["benchmark_flow_rate_5d"])
                else math.nan
            )
            values["price_flow_interaction_5d_t3_cutoff"] = (
                float(ret_5d) * values["benchmark_flow_rate_5d_t3_cutoff"]
                if ret_5d is not None
                and math.isfinite(float(ret_5d))
                and math.isfinite(values["benchmark_flow_rate_5d_t3_cutoff"])
                else math.nan
            )
            values.update(raw_by_symbol[str(symbol)])
            coverage = values.get("all_etf_flow_weight_coverage", math.nan)
            if math.isfinite(coverage):
                coverage_values.append(coverage)
            updates.append(
                tuple(_finite_or_none(values.get(column)) for column in update_columns)
                + (timing.signal_date, str(symbol))
            )
        if updates:
            connection.executemany(update_sql, updates)
            updated_rows += len(updates)
        if coverage_values:
            per_date_coverage.append(
                {
                    "signal_date": timing.signal_date,
                    "median_weight_coverage": float(np.median(coverage_values)),
                    "row_count": len(coverage_values),
                }
            )
        if timing_number % 20 == 0:
            connection.commit()
    connection.commit()
    return {
        "updated_rows": updated_rows,
        "flow_missing_date_count": len(flow_missing_dates),
        "flow_missing_dates": flow_missing_dates,
        "eligible_snapshot_event_count": len(events),
        "snapshot_exclusions": excluded,
        "applied_snapshot_count": exposure_state.applied_snapshots,
        "loaded_holding_rows": exposure_state.loaded_holding_rows,
        "skipped_older_snapshot_count": exposure_state.skipped_older_snapshot,
        "per_date_coverage": per_date_coverage,
    }


def _price_cross_section_ranks(connection: sqlite3.Connection) -> int:
    updated = 0
    dates = [row[0] for row in connection.execute("SELECT DISTINCT signal_date FROM panel")]
    sql = (
        "UPDATE panel SET momentum_rank_20d=?,volatility_rank_20d=?,size_rank=? "
        "WHERE signal_date=? AND symbol=?"
    )
    for number, signal_date in enumerate(dates, 1):
        frame = pd.read_sql_query(
            "SELECT symbol,ret_20d,realized_vol_20d,log_market_cap "
            "FROM panel WHERE signal_date=?",
            connection,
            params=(signal_date,),
        ).set_index("symbol")
        frame["momentum_rank_20d"] = frame["ret_20d"].rank(pct=True)
        frame["volatility_rank_20d"] = frame["realized_vol_20d"].rank(pct=True)
        frame["size_rank"] = frame["log_market_cap"].rank(pct=True)
        rows = [
            (
                _finite_or_none(row.momentum_rank_20d),
                _finite_or_none(row.volatility_rank_20d),
                _finite_or_none(row.size_rank),
                signal_date,
                str(symbol),
            )
            for symbol, row in frame.iterrows()
        ]
        connection.executemany(sql, rows)
        updated += len(rows)
        if number % 50 == 0:
            connection.commit()
    connection.commit()
    return updated


def _live_flow_capture_audit(
    source: SourceBundle, timing: TimingRow | None
) -> dict[str, Any]:
    if timing is None or "incremental" not in source.connections:
        return {"status": "not_requested"}
    connection = source.connections["incremental"]
    rows = list(
        connection.execute(
            "SELECT ticker,effective_date,processed_date,fund_flow,captured_at_utc "
            "FROM etf_flow_observations WHERE provider='massive' "
            "AND ticker IN ('SPY','QQQ') AND effective_date=? ORDER BY ticker",
            (timing.flow_date,),
        )
    )
    market_open = datetime.combine(
        date.fromisoformat(timing.signal_date),
        time(9, 30),
        tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(timezone.utc)
    parsed = []
    for row in rows:
        captured = datetime.fromisoformat(str(row[4]))
        parsed.append(
            {
                "ticker": row[0],
                "effective_date": row[1],
                "processed_date": row[2],
                "fund_flow": row[3],
                "captured_at_utc": row[4],
                "captured_before_t_open": captured < market_open,
            }
        )
    passed = {item["ticker"] for item in parsed} == {"SPY", "QQQ"} and all(
        item["captured_before_t_open"] for item in parsed
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "timing": {
            "signal_date": timing.signal_date,
            "price_date": timing.price_date,
            "flow_date": timing.flow_date,
            "market_open_utc": market_open.isoformat(),
        },
        "rows": parsed,
    }


def _quality_summary(
    connection: sqlite3.Connection,
    timing_rows: Sequence[TimingRow],
    sessions: Sequence[str],
) -> dict[str, Any]:
    count = int(connection.execute("SELECT COUNT(*) FROM panel").fetchone()[0])
    by_year = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            "SELECT substr(signal_date,1,4),COUNT(*) FROM panel GROUP BY 1 ORDER BY 1"
        )
    }
    target_counts = {
        column: int(
            connection.execute(
                f"SELECT COUNT(*) FROM panel WHERE {_sql_identifier(column)} IS NOT NULL"
            ).fetchone()[0]
        )
        for column in TARGET_COLUMNS
    }
    coverage = connection.execute(
        "SELECT COUNT(*),AVG(all_etf_flow_weight_coverage),"
        "AVG(all_etf_flow_count_coverage),AVG(all_etf_flow_observed_count),"
        "SUM(CASE WHEN all_etf_flow_weight_coverage>=0.5 "
        "AND all_etf_flow_observed_count>=5 THEN 1 ELSE 0 END) FROM panel"
    ).fetchone()
    timing_positions = {row.signal_date: row for row in timing_rows}
    session_positions = {value: index for index, value in enumerate(sessions)}
    violations = 0
    for signal, price, flow, legacy in connection.execute(
        "SELECT DISTINCT signal_date,price_date,flow_date,legacy_flow_date FROM panel"
    ):
        expected = timing_positions[str(signal)]
        signal_position = session_positions[str(signal)]
        if (
            price != expected.price_date
            or flow != expected.flow_date
            or str(price) != sessions[signal_position - 1]
            or str(flow) != sessions[signal_position - 2]
            or str(legacy) != sessions[signal_position - 3]
        ):
            violations += 1
    return {
        "row_count": count,
        "rows_by_signal_year": by_year,
        "target_non_null_rows": target_counts,
        "all_etf_flow": {
            "rows": int(coverage[0]),
            "mean_weight_coverage": coverage[1],
            "mean_count_coverage": coverage[2],
            "mean_observed_etf_count": coverage[3],
            "common_evaluation_rows": int(coverage[4] or 0),
            "common_evaluation_rule": (
                "weight coverage >= 0.50 and observed ETF count >= 5"
            ),
        },
        "timing_violation_count": violations,
        "timing_gate": "PASS" if violations == 0 else "FAIL",
    }


def build_panel(
    *,
    base_database: Path,
    incremental_database: Path | None,
    index_evidence: Path,
    output_root: Path,
    start_date: str,
    end_date: str | None,
    live_signal_date: str | None,
    replace: bool,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    panel_path = output_root / "panel.sqlite3"
    manifest_path = output_root / "panel_manifest.json"
    flow_cache_path = output_root / "flow_cache.sqlite3"
    if panel_path.exists() and not replace:
        raise FileExistsError(f"panel exists; pass --replace: {panel_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".panel.", suffix=".building", dir=output_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    started = utc_now()
    with SourceBundle(base_database, incremental_database) as source:
        observed_sessions = source.sessions()
        sessions = _append_live_session(observed_sessions, live_signal_date)
        requested_end = end_date or sessions[-1]
        timing_rows = make_timing_rows(sessions, start_date, requested_end)
        if not timing_rows:
            raise ValueError("requested window contains no forecast signal sessions")
        evidence = load_membership_evidence(index_evidence)
        memberships, reconstruction_audit = reconstruct_memberships(evidence, sessions)
        metadata = source.snapshot_metadata()
        membership_validation = validate_against_holdings(
            source, metadata, memberships, sessions
        )
        symbols, memberships_by_date = _membership_symbols(memberships, timing_rows)
        flow_cache_audit = build_flow_cache(
            source, sessions, flow_cache_path, replace=replace or not flow_cache_path.exists()
        )
        live_timing = (
            next((row for row in timing_rows if row.signal_date == live_signal_date), None)
            if live_signal_date
            else None
        )
        live_capture_audit = _live_flow_capture_audit(source, live_timing)
        if live_signal_date and live_capture_audit["status"] != "PASS":
            raise RuntimeError(
                "live T-2 ETF Flow is absent or was not captured before the T open"
            )
        connection = _create_panel_database(temporary)
        try:
            price_audit = _price_phase(
                connection,
                source,
                sessions,
                timing_rows,
                symbols,
                memberships_by_date,
            )
            with _flow_cache_context(flow_cache_path) as cache:
                flow_audit = _flow_phase(
                    connection,
                    source,
                    cache,
                    sessions,
                    timing_rows,
                    memberships_by_date,
                    metadata,
                )
            rank_updates = _price_cross_section_ranks(connection)
            connection.execute("CREATE INDEX panel_signal_idx ON panel(signal_date)")
            connection.execute("CREATE INDEX panel_symbol_idx ON panel(symbol,signal_date)")
            connection.commit()
            quality = _quality_summary(connection, timing_rows, sessions)
        finally:
            connection.close()
        if quality["timing_gate"] != "PASS":
            raise RuntimeError("panel timing integrity gate failed")
        os.replace(temporary, panel_path)
        manifest = {
            "schema_version": PANEL_SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "started_at_utc": started,
            "timing_contract": TIMING_CONTRACT,
            "requested_window": {
                "start_date": start_date,
                "end_date": requested_end,
                "live_signal_date": live_signal_date,
            },
            "observed_session_range": [observed_sessions[0], observed_sessions[-1]],
            "calendar_session_range": [sessions[0], sessions[-1]],
            "signal_range": [timing_rows[0].signal_date, timing_rows[-1].signal_date],
            "source_fingerprint": source.source_fingerprint(),
            "index_evidence": {
                "path": str(index_evidence),
                "sha256": sha256_file(index_evidence),
            },
            "membership_reconstruction": reconstruction_audit,
            "membership_validation": membership_validation,
            "live_flow_capture_audit": live_capture_audit,
            "flow_cache": flow_cache_audit,
            "price_phase": price_audit,
            "flow_phase": flow_audit,
            "price_rank_updates": rank_updates,
            "quality": quality,
            "panel": {
                "path": str(panel_path),
                "bytes": panel_path.stat().st_size,
                "sha256": sha256_file(panel_path),
            },
        }
        write_json_atomic(manifest_path, manifest)
        return manifest


class _flow_cache_context:
    def __init__(self, path: Path) -> None:
        self.cache = FlowCache(path)

    def __enter__(self) -> FlowCache:
        return self.cache

    def __exit__(self, *_: object) -> None:
        self.cache.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    parser.add_argument(
        "--incremental-database", type=Path, default=DEFAULT_INCREMENTAL_DATABASE
    )
    parser.add_argument("--index-evidence", type=Path, default=DEFAULT_INDEX_EVIDENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2018-01-02")
    parser.add_argument("--end-date")
    parser.add_argument("--live-signal-date")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_panel(
        base_database=args.base_database,
        incremental_database=args.incremental_database,
        index_evidence=args.index_evidence,
        output_root=args.output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        live_signal_date=args.live_signal_date,
        replace=args.replace,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
