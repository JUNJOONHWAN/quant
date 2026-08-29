"""Build a compact point-in-time temporal ETF-stock graph dataset.

The shared ETF flow cube is stored once.  Per-date snapshots contain only the
stock cross-section, targets, active ETF ids, and sparse holding edges, avoiding
duplication of a 60-session flow tensor for every signal date.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from training.quant_forecast_v2.features import (
    compute_price_features,
    next_session_strictly_after,
    price_frame,
)
from training.quant_forecast_v2.io_utils import sha256_file, utc_now, write_json_atomic
from training.quant_forecast_v2.source import SnapshotMeta, SourceBundle, canonical_symbol

from .contracts import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_BASE_DATABASE,
    DEFAULT_INCREMENTAL_DATABASE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PANEL,
    DEFAULT_REPAIRED_FLOW_CACHE,
    EDGE_FEATURE_COLUMNS,
    FLOW_COVERAGE_LOOKBACK_SESSIONS,
    FLOW_COVERAGE_MIN_RATIO,
    FLOW_COVERAGE_REFERENCE_QUANTILE,
    FLOW_ACTIVE_LOOKBACK_SESSIONS,
    FLOW_LOOKBACK_SESSIONS,
    FLOW_VALUE_COLUMNS,
    SMOKE_END_DATE,
    SMOKE_START_DATE,
    SMOKE_SYMBOLS,
    STOCK_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TIMING_CONTRACT,
)


BASE_TARGET_COLUMNS = tuple(
    name
    for horizon in (5, 20)
    for name in (
        f"return_{horizon}d_pct",
        f"upside_{horizon}d_pct",
        f"loss_{horizon}d_pct",
    )
)


def _finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _signed_log_millions(value: object) -> float:
    number = _finite(value)
    if not math.isfinite(number):
        return math.nan
    scaled = abs(number) / 1_000_000.0
    return math.copysign(math.log1p(scaled), number)


def derive_targets(
    stock_targets: Mapping[str, object], benchmark_targets: Mapping[str, object]
) -> np.ndarray:
    """Return absolute and benchmark-relative 5/20 session targets."""

    values: list[float] = []
    for horizon in (5, 20):
        stock_return = _finite(stock_targets.get(f"return_{horizon}d_pct"))
        stock_upside = _finite(stock_targets.get(f"upside_{horizon}d_pct"))
        stock_loss = _finite(stock_targets.get(f"loss_{horizon}d_pct"))
        benchmark_return = _finite(
            benchmark_targets.get(f"return_{horizon}d_pct")
        )
        benchmark_upside = _finite(
            benchmark_targets.get(f"upside_{horizon}d_pct")
        )
        benchmark_loss = _finite(benchmark_targets.get(f"loss_{horizon}d_pct"))
        values.extend(
            (
                stock_return,
                stock_upside,
                stock_loss,
                stock_return - benchmark_return,
                stock_upside - benchmark_upside,
                benchmark_loss - stock_loss,
            )
        )
    return np.asarray(values, dtype=np.float32)


def validate_timing_row(
    sessions: Sequence[str], signal_date: str, price_date: str, flow_date: str
) -> None:
    position = bisect.bisect_left(sessions, signal_date)
    if position < 2 or position >= len(sessions) or sessions[position] != signal_date:
        raise ValueError(f"signal date is not an observed session: {signal_date}")
    expected_price = sessions[position - 1]
    expected_flow = sessions[position - 2]
    if price_date != expected_price or flow_date != expected_flow:
        raise ValueError(
            "timing contract violation: "
            f"T={signal_date} price={price_date}/{expected_price} "
            f"flow={flow_date}/{expected_flow}"
        )


@dataclass(frozen=True)
class SnapshotEvent:
    available_session: str
    metadata: SnapshotMeta


def snapshot_events(
    metadata: Sequence[SnapshotMeta], sessions: Sequence[str]
) -> tuple[list[SnapshotEvent], dict[str, int]]:
    result: list[SnapshotEvent] = []
    excluded: dict[str, int] = defaultdict(int)
    for item in metadata:
        if not item.eligible:
            excluded["ineligible_snapshot"] += 1
            continue
        available = next_session_strictly_after(
            sessions, item.provider_available_date, 1
        )
        if not available:
            excluded["availability_after_calendar"] += 1
            continue
        if item.effective_date > available:
            excluded["effective_after_availability"] += 1
            continue
        result.append(SnapshotEvent(available, item))
    result.sort(
        key=lambda event: (
            event.available_session,
            event.metadata.effective_date,
            event.metadata.etf_ticker,
        )
    )
    return result, dict(excluded)


class ExposureState:
    """Latest disclosed holdings with reverse stock-to-ETF adjacency."""

    def __init__(self, source: SourceBundle) -> None:
        self.source = source
        self.current: dict[str, tuple[SnapshotMeta, dict[str, float]]] = {}
        self.reverse: dict[str, dict[str, float]] = defaultdict(dict)
        self.applied_snapshots = 0
        self.loaded_edges = 0

    def apply(self, item: SnapshotMeta) -> None:
        previous = self.current.get(item.etf_ticker)
        if previous and item.effective_date < previous[0].effective_date:
            return
        if previous:
            for symbol in previous[1]:
                reverse = self.reverse.get(symbol)
                if reverse is not None:
                    reverse.pop(item.etf_ticker, None)
                    if not reverse:
                        self.reverse.pop(symbol, None)
        holdings = self.source.snapshot_holdings(item)
        self.current[item.etf_ticker] = (item, holdings)
        for symbol, weight in holdings.items():
            self.reverse[symbol][item.etf_ticker] = weight
        self.applied_snapshots += 1
        self.loaded_edges += len(holdings)


def _initial_and_future_events(
    events: Sequence[SnapshotEvent], first_price_date: str, last_price_date: str
) -> tuple[list[SnapshotEvent], list[SnapshotEvent]]:
    latest: dict[str, SnapshotEvent] = {}
    future: list[SnapshotEvent] = []
    for event in events:
        if event.available_session <= first_price_date:
            previous = latest.get(event.metadata.etf_ticker)
            if previous is None or (
                event.available_session,
                event.metadata.effective_date,
            ) >= (
                previous.available_session,
                previous.metadata.effective_date,
            ):
                latest[event.metadata.etf_ticker] = event
        elif event.available_session <= last_price_date:
            future.append(event)
    initial = sorted(
        latest.values(),
        key=lambda event: (
            event.available_session,
            event.metadata.effective_date,
            event.metadata.etf_ticker,
        ),
    )
    return initial, future


def _panel_rows(
    panel_path: Path,
    start_date: str,
    end_date: str,
    symbols: Sequence[str] | None,
) -> dict[str, list[sqlite3.Row]]:
    columns = (
        "signal_date",
        "price_date",
        "flow_date",
        "symbol",
        "benchmark",
        *STOCK_FEATURE_COLUMNS,
        *BASE_TARGET_COLUMNS,
    )
    sql = f"SELECT {','.join(columns)} FROM panel WHERE signal_date BETWEEN ? AND ?"
    params: list[object] = [start_date, end_date]
    if symbols:
        sql += f" AND symbol IN ({','.join('?' for _ in symbols)})"
        params.extend(symbols)
    sql += " ORDER BY signal_date,symbol"
    connection = sqlite3.connect(f"file:{Path(panel_path)}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in connection.execute(sql, params):
            grouped[str(row["signal_date"])].append(row)
        return dict(grouped)
    finally:
        connection.close()


def _benchmark_target_frames(
    source: SourceBundle, sessions: Sequence[str]
) -> dict[str, object]:
    spy = price_frame(source.price_rows("SPY"), sessions)
    qqq = price_frame(source.price_rows("QQQ"), sessions)
    return {
        "SPY": compute_price_features(spy, spy, spy, qqq),
        "QQQ": compute_price_features(qqq, qqq, spy, qqq),
    }


def _flow_vocabulary(
    connection: sqlite3.Connection, metadata: Sequence[SnapshotMeta]
) -> tuple[list[str], dict[str, int], dict[str, object]]:
    tickers = {
        canonical_symbol(row[0])
        for row in connection.execute("SELECT DISTINCT ticker FROM flow")
        if canonical_symbol(row[0])
    }
    ordered = sorted(tickers)
    metadata_tickers = {item.etf_ticker for item in metadata if item.etf_ticker}
    metadata_without_flow = sorted(metadata_tickers - tickers)
    return (
        ordered,
        {ticker: index for index, ticker in enumerate(ordered)},
        {
            "flow_ticker_count": len(ordered),
            "metadata_ticker_count": len(metadata_tickers),
            "metadata_without_any_flow_count": len(metadata_without_flow),
            "metadata_without_any_flow": metadata_without_flow,
            "policy": "ETF nodes require at least one normalized Flow observation",
        },
    )


def flow_coverage_ratio(
    flow_counts: Mapping[str, int], sessions: Sequence[str], flow_date: str
) -> tuple[int, float, float]:
    """Compare exact Flow count with a robust prior PIT-visible lower envelope."""

    position = bisect.bisect_left(sessions, flow_date)
    start = max(0, position - FLOW_COVERAGE_LOOKBACK_SESSIONS)
    prior = [int(flow_counts.get(value, 0)) for value in sessions[start:position]]
    reference = (
        float(np.quantile(prior, FLOW_COVERAGE_REFERENCE_QUANTILE))
        if prior
        else 0.0
    )
    current = int(flow_counts.get(flow_date, 0))
    ratio = current / reference if reference > 0 else (1.0 if current > 0 else 0.0)
    return current, reference, ratio


def recent_visible_flow_ids(
    availability_window: np.ndarray, signal_position: int
) -> set[int]:
    """Return every ETF with at least one PIT-visible row in the lookback.

    The exact T-2 observation is deliberately not required here.  A missing
    current report stays missing in the tensor and its age remains observable;
    the ETF node is retained so convergence/divergence can use the full recent
    Flow universe instead of only today's reporters.
    """

    availability = np.asarray(availability_window)
    if availability.ndim != 2:
        raise ValueError("availability window must be [session, ETF]")
    visible = (availability >= 0) & (availability <= int(signal_position))
    return set(np.flatnonzero(visible.any(axis=0)).astype(int).tolist())


def _pit_visible_flow_counts(
    connection: sqlite3.Connection, sessions: Sequence[str]
) -> tuple[dict[str, int], dict[str, int]]:
    """Count rows usable at each date's own T decision, never eventual revisions."""

    positions = {value: index for index, value in enumerate(sessions)}
    counts: dict[str, int] = defaultdict(int)
    scanned = 0
    unavailable_at_own_signal = 0
    outside_calendar = 0
    for effective_date, available_session in connection.execute(
        "SELECT effective_date,available_session FROM flow"
    ):
        scanned += 1
        effective = str(effective_date)
        position = positions.get(effective)
        if position is None or position + 2 >= len(sessions):
            outside_calendar += 1
            continue
        own_signal = sessions[position + 2]
        if str(available_session) <= own_signal:
            counts[effective] += 1
        else:
            unavailable_at_own_signal += 1
    return dict(counts), {
        "rows_scanned": scanned,
        "rows_visible_at_own_t": sum(counts.values()),
        "rows_unavailable_at_own_t": unavailable_at_own_signal,
        "rows_outside_calendar": outside_calendar,
    }


def _materialize_flow_cube(
    connection: sqlite3.Connection,
    output_root: Path,
    sessions: Sequence[str],
    etf_to_id: Mapping[str, int],
    first_history_date: str,
    last_flow_date: str,
) -> dict[str, object]:
    first_position = bisect.bisect_left(sessions, first_history_date)
    last_position = bisect.bisect_right(sessions, last_flow_date) - 1
    if first_position < 0 or last_position < first_position:
        raise ValueError("invalid flow cube session range")
    cube_sessions = list(sessions[first_position : last_position + 1])
    session_to_local = {value: index for index, value in enumerate(cube_sessions)}
    global_session_position = {value: index for index, value in enumerate(sessions)}
    shape = (len(cube_sessions), len(etf_to_id), len(FLOW_VALUE_COLUMNS))
    values_path = output_root / "flow_values.npy"
    available_path = output_root / "flow_available_session_index.npy"
    values = np.lib.format.open_memmap(
        values_path, mode="w+", dtype=np.float32, shape=shape
    )
    values[:] = np.nan
    available = np.lib.format.open_memmap(
        available_path,
        mode="w+",
        dtype=np.int32,
        shape=(len(cube_sessions), len(etf_to_id)),
    )
    available[:] = -1
    previous_shares: dict[str, float] = {}
    inserted = 0
    excluded_availability = 0
    rows = connection.execute(
        "SELECT ticker,effective_date,available_session,flow_rate_pct,fund_flow,"
        "nav,shares_outstanding FROM flow WHERE effective_date BETWEEN ? AND ? "
        "ORDER BY ticker,effective_date",
        (first_history_date, last_flow_date),
    )
    for row in rows:
        ticker = canonical_symbol(row[0])
        effective = str(row[1])
        available_session = str(row[2])
        local_date = session_to_local.get(effective)
        etf_id = etf_to_id.get(ticker)
        available_index = global_session_position.get(available_session)
        if local_date is None or etf_id is None or available_index is None:
            excluded_availability += 1
            continue
        nav = _finite(row[5])
        shares = _finite(row[6])
        assets = nav * shares if nav > 0 and shares > 0 else math.nan
        prior = previous_shares.get(ticker, math.nan)
        share_change = (
            (shares / prior - 1.0) * 100.0
            if shares > 0 and prior > 0
            else math.nan
        )
        if shares > 0:
            previous_shares[ticker] = shares
        values[local_date, etf_id] = np.asarray(
            (
                _finite(row[3]),
                _signed_log_millions(row[4]),
                math.log1p(assets) if assets > 0 else math.nan,
                share_change,
            ),
            dtype=np.float32,
        )
        available[local_date, etf_id] = available_index
        inserted += 1
    values.flush()
    available.flush()
    return {
        "session_start_position": first_position,
        "session_end_position": last_position,
        "session_count": len(cube_sessions),
        "etf_count": len(etf_to_id),
        "feature_count": len(FLOW_VALUE_COLUMNS),
        "inserted_rows": inserted,
        "excluded_availability_rows": excluded_availability,
        "values_path": str(values_path),
        "available_path": str(available_path),
        "values_bytes": values_path.stat().st_size,
        "available_bytes": available_path.stat().st_size,
    }


def _save_snapshot(path: Path, **arrays: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_dataset(
    *,
    panel_path: Path,
    flow_cache_path: Path,
    base_database: Path,
    incremental_database: Path | None,
    output_root: Path,
    start_date: str,
    end_date: str,
    symbols: Sequence[str] | None,
    replace: bool,
    smoke_only: bool,
) -> dict[str, object]:
    output_root = Path(output_root)
    if output_root.exists():
        if not replace:
            raise FileExistsError(f"dataset output exists: {output_root}")
        resolved = output_root.resolve()
        allowed = Path(DEFAULT_OUTPUT_ROOT).resolve()
        if resolved != allowed and allowed not in resolved.parents:
            raise ValueError(f"refusing to replace output outside {allowed}: {resolved}")
        shutil.rmtree(resolved)
    output_root.mkdir(parents=True)
    snapshot_root = output_root / "snapshots"
    snapshot_root.mkdir()
    selected_symbols = (
        tuple(dict.fromkeys(canonical_symbol(value) for value in symbols))
        if symbols
        else None
    )
    grouped = _panel_rows(panel_path, start_date, end_date, selected_symbols)
    if not grouped:
        raise ValueError("no panel rows in requested graph window")
    started_at = utc_now()
    flow_connection = sqlite3.connect(
        f"file:{Path(flow_cache_path)}?mode=ro", uri=True
    )
    flow_connection.row_factory = sqlite3.Row
    flow_connection.execute("PRAGMA query_only=ON")
    with SourceBundle(base_database, incremental_database) as source:
        sessions = source.sessions()
        session_position = {value: index for index, value in enumerate(sessions)}
        first_row = grouped[min(grouped)][0]
        last_row = grouped[max(grouped)][0]
        first_price_date = str(first_row["price_date"])
        last_price_date = str(last_row["price_date"])
        first_flow_position = session_position[str(first_row["flow_date"])]
        first_history_position = max(0, first_flow_position - FLOW_LOOKBACK_SESSIONS + 1)
        first_history_date = sessions[first_history_position]
        last_flow_date = str(last_row["flow_date"])
        metadata = source.snapshot_metadata()
        events, event_exclusions = snapshot_events(metadata, sessions)
        initial_events, future_events = _initial_and_future_events(
            events, first_price_date, last_price_date
        )
        state = ExposureState(source)
        for event in initial_events:
            state.apply(event.metadata)
        etf_vocab, etf_to_id, vocabulary_audit = _flow_vocabulary(
            flow_connection, metadata
        )
        cube_audit = _materialize_flow_cube(
            flow_connection,
            output_root,
            sessions,
            etf_to_id,
            first_history_date,
            last_flow_date,
        )
        cube_available = np.load(
            Path(str(cube_audit["available_path"])), mmap_mode="r"
        )
        cube_start_position = int(cube_audit["session_start_position"])
        benchmark_frames = _benchmark_target_frames(source, sessions)
        pit_visible_flow_counts, flow_coverage_audit = _pit_visible_flow_counts(
            flow_connection, sessions
        )
        future_position = 0
        snapshot_receipts: list[dict[str, object]] = []
        coverage_exclusions: list[dict[str, object]] = []
        total_edges = 0
        total_stocks = 0
        requested_stocks = 0
        zero_edge_stocks = 0
        total_recent_flow_etfs = 0
        total_recent_without_current = 0
        timing_violations = 0
        for signal_date in sorted(grouped):
            rows = grouped[signal_date]
            price_date = str(rows[0]["price_date"])
            flow_date = str(rows[0]["flow_date"])
            try:
                validate_timing_row(sessions, signal_date, price_date, flow_date)
            except ValueError:
                timing_violations += 1
                raise
            while (
                future_position < len(future_events)
                and future_events[future_position].available_session <= price_date
            ):
                state.apply(future_events[future_position].metadata)
                future_position += 1
            requested_stocks += len(rows)
            signal_position = session_position[signal_date]
            flow_position = session_position[flow_date]
            local_flow_end = flow_position - cube_start_position
            local_flow_start = max(
                0, local_flow_end - FLOW_ACTIVE_LOOKBACK_SESSIONS + 1
            )
            recent_flow_ids = recent_visible_flow_ids(
                np.asarray(
                    cube_available[local_flow_start : local_flow_end + 1],
                    dtype=np.int32,
                ),
                signal_position,
            )
            visible_flow_ids: set[int] = set()
            for flow_row in flow_connection.execute(
                "SELECT ticker FROM flow WHERE effective_date=? AND available_session<=?",
                (flow_date, signal_date),
            ):
                ticker = canonical_symbol(flow_row[0])
                if ticker in etf_to_id:
                    visible_flow_ids.add(etf_to_id[ticker])
            _, reference_count, _ = flow_coverage_ratio(
                pit_visible_flow_counts, sessions, flow_date
            )
            observed_flow_count = len(visible_flow_ids)
            coverage_ratio = (
                observed_flow_count / reference_count
                if reference_count > 0
                else (1.0 if observed_flow_count > 0 else 0.0)
            )
            if coverage_ratio < FLOW_COVERAGE_MIN_RATIO:
                coverage_exclusions.append(
                    {
                        "signal_date": signal_date,
                        "price_date": price_date,
                        "flow_date": flow_date,
                        "stock_count": len(rows),
                        "observed_flow_etf_count": observed_flow_count,
                        "prior_flow_etf_lower_envelope": reference_count,
                        "coverage_ratio": coverage_ratio,
                        "minimum_ratio": FLOW_COVERAGE_MIN_RATIO,
                        "reason": "exact_t_minus_2_flow_cross_section_incomplete",
                    }
                )
                continue
            stock_symbols = [str(row["symbol"]) for row in rows]
            stock_x = np.asarray(
                [
                    [_finite(row[column]) for column in STOCK_FEATURE_COLUMNS]
                    for row in rows
                ],
                dtype=np.float32,
            )
            targets = []
            target_masks = []
            connected: set[str] = set()
            edge_records: list[tuple[int, str, float, float]] = []
            price_position = session_position[price_date]
            date_zero_edge_stocks = 0
            for stock_index, row in enumerate(rows):
                benchmark = str(row["benchmark"])
                benchmark_frame = benchmark_frames[benchmark]
                benchmark_row = benchmark_frame.loc[price_date]
                stock_target = {name: row[name] for name in BASE_TARGET_COLUMNS}
                benchmark_target = {
                    name: benchmark_row.get(name) for name in BASE_TARGET_COLUMNS
                }
                target = derive_targets(stock_target, benchmark_target)
                targets.append(target)
                target_masks.append(np.isfinite(target))
                exposures = state.reverse.get(str(row["symbol"]), {})
                if not exposures:
                    zero_edge_stocks += 1
                    date_zero_edge_stocks += 1
                for ticker, weight_percent in exposures.items():
                    current = state.current.get(ticker)
                    if current is None or ticker not in etf_to_id:
                        continue
                    effective_position = bisect.bisect_right(
                        sessions, current[0].effective_date
                    ) - 1
                    age_sessions = max(0, price_position - effective_position)
                    connected.add(ticker)
                    edge_records.append(
                        (
                            stock_index,
                            ticker,
                            float(weight_percent) / 100.0,
                            min(age_sessions / 252.0, 5.0),
                        )
                    )
            connected_ids = {etf_to_id[ticker] for ticker in connected}
            active_ids = recent_flow_ids | connected_ids
            ordered_ids = np.asarray(sorted(active_ids), dtype=np.int64)
            local_etf = {int(value): index for index, value in enumerate(ordered_ids)}
            current_observed = visible_flow_ids
            edge_index = np.empty((2, len(edge_records)), dtype=np.int64)
            edge_attr = np.empty(
                (len(edge_records), len(EDGE_FEATURE_COLUMNS)), dtype=np.float32
            )
            for index, (stock_index, ticker, weight, age) in enumerate(edge_records):
                global_id = etf_to_id[ticker]
                edge_index[:, index] = (stock_index, local_etf[global_id])
                edge_attr[index] = (
                    weight,
                    age,
                    float(global_id in current_observed),
                )
            observed_edges = int((edge_attr[:, 2] > 0.5).sum()) if len(edge_attr) else 0
            edge_weight = np.abs(edge_attr[:, 0]) if len(edge_attr) else np.asarray([])
            observed_edge_weight_ratio = (
                float(edge_weight[edge_attr[:, 2] > 0.5].sum() / edge_weight.sum())
                if len(edge_weight) and edge_weight.sum() > 0
                else 0.0
            )
            snapshot_path = snapshot_root / f"{signal_date}.npz"
            _save_snapshot(
                snapshot_path,
                stock_symbols=np.asarray(stock_symbols, dtype="U32"),
                stock_x=stock_x,
                targets=np.asarray(targets, dtype=np.float32),
                target_mask=np.asarray(target_masks, dtype=np.uint8),
                etf_ids=ordered_ids,
                edge_index=edge_index,
                edge_attr=edge_attr,
                signal_position=np.asarray(signal_position, dtype=np.int32),
                flow_position=np.asarray(flow_position, dtype=np.int32),
            )
            total_edges += len(edge_records)
            total_stocks += len(rows)
            total_recent_flow_etfs += len(recent_flow_ids)
            total_recent_without_current += len(recent_flow_ids - visible_flow_ids)
            snapshot_receipts.append(
                {
                    "signal_date": signal_date,
                    "price_date": price_date,
                    "flow_date": flow_date,
                    "path": str(snapshot_path),
                    "stock_count": len(rows),
                    "etf_count": len(ordered_ids),
                    "flow_observed_etf_count": observed_flow_count,
                    "recent_visible_flow_etf_count": len(recent_flow_ids),
                    "recent_without_current_flow_count": len(
                        recent_flow_ids - visible_flow_ids
                    ),
                    "connected_without_recent_flow_count": len(
                        connected_ids - recent_flow_ids
                    ),
                    "prior_flow_etf_lower_envelope": reference_count,
                    "flow_coverage_ratio": coverage_ratio,
                    "connected_etf_count": len(connected),
                    "edge_count": len(edge_records),
                    "zero_edge_stock_count": date_zero_edge_stocks,
                    "relation_stock_coverage_ratio": (
                        (len(rows) - date_zero_edge_stocks) / len(rows)
                        if rows
                        else 0.0
                    ),
                    "observed_edge_count": observed_edges,
                    "observed_edge_ratio": (
                        observed_edges / len(edge_records) if edge_records else 0.0
                    ),
                    "observed_edge_weight_ratio": observed_edge_weight_ratio,
                    "bytes": snapshot_path.stat().st_size,
                }
            )
        source_fingerprint = source.source_fingerprint()
    flow_connection.close()
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "started_at_utc": started_at,
        "smoke_only": bool(smoke_only),
        "timing_contract": TIMING_CONTRACT,
        "requested_window": {"start_date": start_date, "end_date": end_date},
        "requested_symbols": list(selected_symbols) if selected_symbols else "ALL_PANEL",
        "feature_contract": {
            "stock": list(STOCK_FEATURE_COLUMNS),
            "etf_flow": list(FLOW_VALUE_COLUMNS),
            "edge": list(EDGE_FEATURE_COLUMNS),
            "targets": list(TARGET_COLUMNS),
            "flow_lookback_sessions": FLOW_LOOKBACK_SESSIONS,
        },
        "source": {
            "panel": {
                "path": str(panel_path),
                "bytes": Path(panel_path).stat().st_size,
                "sha256": sha256_file(panel_path),
            },
            "flow_cache": {
                "path": str(flow_cache_path),
                "bytes": Path(flow_cache_path).stat().st_size,
                "sha256": sha256_file(flow_cache_path),
            },
            "databases": source_fingerprint,
        },
        "holdings": {
            "metadata_rows": len(metadata),
            "eligible_events": len(events),
            "event_exclusions": event_exclusions,
            "initial_events": len(initial_events),
            "post_start_pit_events_applied": future_position,
            "applied_snapshots": state.applied_snapshots,
            "loaded_edges_during_state_updates": state.loaded_edges,
        },
        "flow_cube": cube_audit,
        "flow_vocabulary_audit": vocabulary_audit,
        "etf_vocabulary": etf_vocab,
        "sessions": sessions,
        "requested_snapshot_count": len(grouped),
        "snapshot_count": len(snapshot_receipts),
        "excluded_snapshot_count": len(coverage_exclusions),
        "coverage_exclusions": coverage_exclusions,
        "requested_stock_row_count": requested_stocks,
        "stock_row_count": total_stocks,
        "edge_count": total_edges,
        "zero_edge_stock_count": zero_edge_stocks,
        "active_etf_selection": {
            "policy": (
                "all ETFs with at least one PIT-visible Flow row in the prior "
                f"{FLOW_ACTIVE_LOOKBACK_SESSIONS} sessions, union PIT-connected ETFs"
            ),
            "exact_t_minus_2_missing_policy": (
                "retain recent ETF node; preserve NaN/mask/reporting age; no T-3 fill"
            ),
            "snapshot_recent_flow_etf_total": total_recent_flow_etfs,
            "snapshot_recent_without_current_total": total_recent_without_current,
        },
        "timing_violation_count": timing_violations,
        "flow_coverage_contract": {
            "lookback_sessions": FLOW_COVERAGE_LOOKBACK_SESSIONS,
            "reference_quantile": FLOW_COVERAGE_REFERENCE_QUANTILE,
            "minimum_ratio": FLOW_COVERAGE_MIN_RATIO,
            "reference_mode": "prior_dates_pit_visible_at_each_own_t",
            "action": "exclude incomplete signal date; never fill from T-3",
        },
        "flow_coverage_audit": flow_coverage_audit,
        "quality_gate": (
            ("PASS" if not coverage_exclusions else "PASS_WITH_EXCLUSIONS")
            if timing_violations == 0 and total_stocks > 0 and total_edges > 0
            else "FAIL"
        ),
        "snapshots": snapshot_receipts,
    }
    if manifest["quality_gate"] not in {"PASS", "PASS_WITH_EXCLUSIONS"}:
        raise RuntimeError("graph dataset quality gate failed")
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument(
        "--flow-cache", type=Path, default=DEFAULT_REPAIRED_FLOW_CACHE
    )
    parser.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    parser.add_argument(
        "--incremental-database", type=Path, default=DEFAULT_INCREMENTAL_DATABASE
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_dataset(
        panel_path=args.panel,
        flow_cache_path=args.flow_cache,
        base_database=args.base_database,
        incremental_database=args.incremental_database,
        output_root=args.output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols,
        replace=args.replace,
        smoke_only=args.smoke_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def smoke_argv(output_root: Path, *, replace: bool = False) -> list[str]:
    result = [
        "--output-root",
        str(output_root),
        "--start-date",
        SMOKE_START_DATE,
        "--end-date",
        SMOKE_END_DATE,
        "--symbols",
        *SMOKE_SYMBOLS,
        "--smoke-only",
    ]
    if replace:
        result.append("--replace")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
