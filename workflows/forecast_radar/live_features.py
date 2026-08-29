"""Point-in-time live feature construction for Forecast RADAR."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.phase_b_cluster import (
    _cluster_sequences,
    cluster_flow_states,
)
from training.quant_flow_graph_v11_r2.phase_b_market import build_market_matrix
from training.quant_flow_graph_v11_r2.phase_b_stock import (
    DIRECT_FLOW_FIELDS,
    DIRECT_MASK_FIELDS,
    GLOBAL_FLOW_FIELDS,
    INDIRECT_BASE_FIELDS,
    INDIRECT_FLOW_FIELDS,
    RELATION_FIELDS,
    _event_arrays,
    add_symbol_rolling_features,
    aggregate_snapshot_features,
)
from training.quant_flow_graph_v16.full_etf_latent import (
    STATE_NAMES,
    _masked_rolling_mean,
    _row_scale,
)

from .contracts import LATENT_COMPONENTS
from .io import sha256_file, utc_now, write_json_atomic


TARGET_ANCHORS = ("SPY", "QQQ", "IWM", "RSP", "VTI", "DIA")


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _create_live_source_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE daily_observations(
          source TEXT NOT NULL,
          symbol TEXT NOT NULL,
          trade_date TEXT NOT NULL,
          open REAL,
          high REAL,
          low REAL,
          close REAL,
          adjusted_close REAL,
          volume REAL,
          PRIMARY KEY(source,symbol,trade_date)
        ) WITHOUT ROWID;
        CREATE TABLE etf_flow_observations(
          provider TEXT NOT NULL,
          ticker TEXT NOT NULL,
          effective_date TEXT NOT NULL,
          processed_date TEXT NOT NULL,
          fund_flow REAL,
          nav REAL,
          shares_outstanding REAL,
          available_at_date TEXT NOT NULL,
          PRIMARY KEY(provider,ticker,effective_date)
        ) WITHOUT ROWID;
        """
    )


def build_graph_session_overlay_database(
    *, incremental_database: Path, output_path: Path, signal_date: str
) -> dict[str, Any]:
    """Create a minimal read-only-source overlay containing a synthetic T session.

    The graph builder validates T against observed SPY sessions even though the
    decision is made before T has a closing bar.  The panel already supports
    this live session explicitly; this isolated overlay supplies the same
    calendar row without mutating the Oracle increment.
    """

    incremental_database = Path(incremental_database)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256_file(incremental_database)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".building", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute(
            "ATTACH DATABASE ? AS incremental",
            (f"file:{incremental_database}?mode=ro",),
        )
        connection.execute(
            "CREATE TABLE daily_observations AS "
            "SELECT * FROM incremental.daily_observations "
            "WHERE source='fmp' AND symbol IN ('SPY','QQQ')"
        )
        has_holdings = bool(
            connection.execute(
                "SELECT 1 FROM incremental.sqlite_master "
                "WHERE type='table' AND name='etf_constituent_observations'"
            ).fetchone()
        )
        if has_holdings:
            connection.execute(
                "CREATE TABLE etf_constituent_observations AS "
                "SELECT * FROM incremental.etf_constituent_observations"
            )
        columns = [
            str(row[1]) for row in connection.execute("PRAGMA table_info(daily_observations)")
        ]
        placeholders = ",".join("?" for _ in columns)
        synthetic_symbols: list[str] = []
        for symbol in ("SPY", "QQQ"):
            row = connection.execute(
                "SELECT * FROM daily_observations WHERE source='fmp' AND symbol=? "
                "AND trade_date<? AND close>0 AND volume>0 "
                "ORDER BY trade_date DESC LIMIT 1",
                (symbol, signal_date),
            ).fetchone()
            if row is None:
                raise ValueError(f"session overlay has no prior {symbol} row")
            values = list(row)
            values[columns.index("trade_date")] = signal_date
            connection.execute(
                f"INSERT INTO daily_observations VALUES({placeholders})", values
            )
            synthetic_symbols.append(symbol)
        connection.execute(
            "CREATE INDEX daily_symbol_date_idx "
            "ON daily_observations(source,symbol,trade_date)"
        )
        if has_holdings:
            connection.execute(
                "CREATE INDEX holdings_etf_date_idx ON "
                "etf_constituent_observations(provider,etf_ticker,effective_date)"
            )
        connection.commit()
        connection.execute("DETACH DATABASE incremental")
        row_count = int(
            connection.execute("SELECT COUNT(*) FROM daily_observations").fetchone()[0]
        )
        holding_count = (
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM etf_constituent_observations"
                ).fetchone()[0]
            )
            if has_holdings
            else 0
        )
    finally:
        connection.close()
    os.replace(temporary, output_path)
    source_hash_after = sha256_file(incremental_database)
    if source_hash_after != source_hash_before:
        raise RuntimeError("canonical incremental database changed during overlay build")
    receipt = {
        "schema_version": "quant.forecast_radar.graph_session_overlay.v1",
        "generated_at_utc": utc_now(),
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "signal_date": signal_date,
        "synthetic_session_symbols": synthetic_symbols,
        "synthetic_values_are_prior_close_copies": True,
        "used_for_targets_or_price_features": False,
        "daily_rows": row_count,
        "incremental_holding_rows": holding_count,
        "canonical_source_sha256_before": source_hash_before,
        "canonical_source_sha256_after": source_hash_after,
        "canonical_source_mutated": False,
        "quality_gate": "PASS",
    }
    write_json_atomic(output_path.with_suffix(".receipt.json"), receipt)
    return receipt


def build_live_source_database(
    *,
    base_database: Path,
    incremental_database: Path,
    output_path: Path,
    history_start: str,
    signal_date: str,
    replace: bool,
) -> dict[str, Any]:
    """Build an isolated recent union; neither canonical source is mutated."""

    output_path = Path(output_path)
    if output_path.exists() and not replace:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".building", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        _create_live_source_schema(connection)
        connection.execute(
            "ATTACH DATABASE ? AS base",
            (f"file:{Path(base_database)}?mode=ro",),
        )
        connection.execute(
            "ATTACH DATABASE ? AS incremental",
            (f"file:{Path(incremental_database)}?mode=ro",),
        )
        flow_sql = (
            "INSERT OR REPLACE INTO etf_flow_observations "
            "SELECT provider,ticker,effective_date,processed_date,fund_flow,nav,"
            "shares_outstanding,available_at_date FROM {origin}.etf_flow_observations "
            "WHERE provider='massive' AND effective_date>=?"
        )
        for origin in ("base", "incremental"):
            connection.execute(flow_sql.format(origin=origin), (history_start,))
        daily_sql = (
            "INSERT OR REPLACE INTO daily_observations "
            "SELECT source,symbol,trade_date,open,high,low,close,adjusted_close,volume "
            "FROM {origin}.daily_observations WHERE source='fmp' AND trade_date>=? "
            "AND (symbol IN (SELECT ticker FROM etf_flow_observations) "
            "OR symbol IN ('SPY','QQQ','IWM','RSP','VTI','DIA'))"
        )
        for origin in ("base", "incremental"):
            connection.execute(daily_sql.format(origin=origin), (history_start,))
        if not connection.execute(
            "SELECT 1 FROM daily_observations WHERE source='fmp' AND symbol='SPY' "
            "AND trade_date=?",
            (signal_date,),
        ).fetchone():
            connection.execute(
                "INSERT INTO daily_observations(source,symbol,trade_date) "
                "VALUES('fmp','SPY',?)",
                (signal_date,),
            )
        connection.execute(
            "CREATE INDEX daily_date_idx ON daily_observations(trade_date,symbol)"
        )
        connection.execute(
            "CREATE INDEX flow_date_idx ON etf_flow_observations(effective_date,ticker)"
        )
        connection.commit()
        daily = connection.execute(
            "SELECT COUNT(*),MIN(trade_date),MAX(trade_date),COUNT(DISTINCT symbol) "
            "FROM daily_observations"
        ).fetchone()
        flow = connection.execute(
            "SELECT COUNT(*),MIN(effective_date),MAX(effective_date),"
            "COUNT(DISTINCT ticker) FROM etf_flow_observations"
        ).fetchone()
        connection.execute("DETACH DATABASE base")
        connection.execute("DETACH DATABASE incremental")
    finally:
        connection.close()
    os.replace(temporary, output_path)
    receipt = {
        "schema_version": "quant.forecast_radar.live_source.v1",
        "generated_at_utc": utc_now(),
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "history_start": history_start,
        "signal_date": signal_date,
        "daily": {
            "rows": int(daily[0]),
            "min_date": daily[1],
            "max_date": daily[2],
            "symbols": int(daily[3]),
        },
        "flow": {
            "rows": int(flow[0]),
            "min_date": flow[1],
            "max_date": flow[2],
            "tickers": int(flow[3]),
        },
        "canonical_sources_mutated": False,
        "quality_gate": "PASS" if daily[0] and flow[0] else "FAIL",
    }
    write_json_atomic(output_path.with_suffix(".receipt.json"), receipt)
    return receipt


def build_live_stock_matrix(
    *,
    event: sqlite3.Connection,
    source: sqlite3.Connection,
    graph_dataset_root: Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build all current graph rows without requiring future targets."""

    manifest_path = Path(graph_dataset_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stock_feature_names = tuple(manifest["feature_contract"]["stock"])
    etf_vocabulary = tuple(str(value) for value in manifest["etf_vocabulary"])
    market = build_market_matrix(event=event, source=source)
    market_date_to_index = {date: index for index, date in enumerate(market["dates"])}
    market_flow_names = tuple(market["flow_names"])
    missing_global = [name for name in GLOBAL_FLOW_FIELDS if name not in market_flow_names]
    if missing_global:
        raise ValueError(f"missing global live Flow fields: {missing_global}")
    global_indices = [market_flow_names.index(name) for name in GLOBAL_FLOW_FIELDS]
    drift_index = market_flow_names.index("drift_rate_pct")
    states = cluster_flow_states(event)
    cluster_values = tuple(sorted({cluster for _, cluster in states}))
    cluster_to_code = {cluster: index for index, cluster in enumerate(cluster_values)}
    states = _cluster_sequences(states, tuple(market["dates"]))

    price_parts: list[np.ndarray] = []
    flow_parts: list[np.ndarray] = []
    date_code_parts: list[np.ndarray] = []
    symbol_code_parts: list[np.ndarray] = []
    date_values: list[str] = []
    symbols: list[str] = []
    symbol_to_code: dict[str, int] = {}
    audit: defaultdict[str, int] = defaultdict(int)
    for ref_number, ref in enumerate(manifest["snapshots"], 1):
        signal_date = str(ref["signal_date"])
        market_index = market_date_to_index.get(signal_date)
        if market_index is None:
            audit["no_event_date"] += 1
            continue
        with np.load(ref["path"], allow_pickle=False) as item:
            stock_symbols = tuple(str(value) for value in item["stock_symbols"])
            stock_x = np.asarray(item["stock_x"], dtype=np.float32)
            local_global_etf = np.asarray(item["etf_ids"], dtype=np.int64)
            edge_index = np.asarray(item["edge_index"], dtype=np.int64)
            edge_attr = np.asarray(item["edge_attr"], dtype=np.float32)
        local_tickers = tuple(etf_vocabulary[index] for index in local_global_etf)
        event_arrays, timing = _event_arrays(
            connection=event,
            signal_date=signal_date,
            local_tickers=local_tickers,
            cluster_to_code=cluster_to_code,
        )
        if timing != (str(ref["price_date"]), str(ref["flow_date"])):
            raise ValueError(f"live timing mismatch at {signal_date}: {timing}")
        cluster_state = np.full(
            (len(cluster_values), len(INDIRECT_BASE_FIELDS)), np.nan, dtype=np.float64
        )
        for cluster, cluster_code in cluster_to_code.items():
            state = states.get((signal_date, cluster), {})
            cluster_state[cluster_code] = [
                float(state.get(name, math.nan)) for name in INDIRECT_BASE_FIELDS
            ]
        drift_rate = float(market["flow_matrix"][market_index, drift_index])
        direct, indirect, snapshot_audit = aggregate_snapshot_features(
            stock_count=len(stock_symbols),
            edge_stock=edge_index[0],
            edge_etf=edge_index[1],
            edge_weight=edge_attr[:, 0],
            edge_age=edge_attr[:, 1],
            cluster_states=cluster_state,
            drift_rate=drift_rate,
            **event_arrays,
        )
        for key, value in snapshot_audit.items():
            audit[key] += int(value)
        direct_names = DIRECT_MASK_FIELDS + DIRECT_FLOW_FIELDS
        direct_index = {name: index for index, name in enumerate(direct_names)}
        indirect_rate = indirect[:, INDIRECT_BASE_FIELDS.index("cluster_flow_rate_pct")]
        direct_rate = direct[:, direct_index["direct_clean_rate_net"]]
        relations = np.column_stack(
            [
                direct_rate - indirect_rate,
                indirect_rate - drift_rate,
                np.sign(direct_rate) * np.sign(indirect_rate),
                stock_x[:, stock_feature_names.index("relative_ret_5d")] * indirect_rate,
                stock_x[:, stock_feature_names.index("drawdown_20d_pct")]
                * np.minimum(indirect_rate, 0.0),
            ]
        )
        global_row = market["flow_matrix"][market_index, global_indices]
        global_rows = np.repeat(global_row[None, :], len(stock_symbols), axis=0)
        for symbol in stock_symbols:
            if symbol not in symbol_to_code:
                symbol_to_code[symbol] = len(symbols)
                symbols.append(symbol)
        date_code = len(date_values)
        date_values.append(signal_date)
        price_parts.append(stock_x)
        flow_parts.append(
            np.column_stack([global_rows, direct, indirect, relations]).astype(np.float32)
        )
        date_code_parts.append(np.full(len(stock_symbols), date_code, dtype=np.int32))
        symbol_code_parts.append(
            np.asarray([symbol_to_code[symbol] for symbol in stock_symbols], dtype=np.int32)
        )
        audit["snapshots"] += 1
        audit["rows"] += len(stock_symbols)
        if progress and (ref_number == 1 or ref_number % 25 == 0):
            progress({"stage": "live_stock_matrix", "signal_date": signal_date})
    if not price_parts:
        raise ValueError("no live stock snapshots align with the event cube")
    flow = np.concatenate(flow_parts)
    date_codes = np.concatenate(date_code_parts)
    symbol_codes = np.concatenate(symbol_code_parts)
    base_flow_names = (
        GLOBAL_FLOW_FIELDS
        + DIRECT_MASK_FIELDS
        + DIRECT_FLOW_FIELDS
        + INDIRECT_FLOW_FIELDS
        + RELATION_FIELDS
    )
    flow, flow_names = add_symbol_rolling_features(
        flow=flow,
        flow_names=base_flow_names,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        symbol_count=len(symbols),
    )
    return {
        "date_values": tuple(date_values),
        "date_codes": date_codes,
        "symbol_values": tuple(symbols),
        "symbol_codes": symbol_codes,
        "price_names": stock_feature_names,
        "price_matrix": np.concatenate(price_parts),
        "flow_names": flow_names,
        "flow_matrix": np.asarray(flow, dtype=np.float32),
        "audit": dict(audit),
        "source_manifest_sha256": sha256_file(manifest_path),
    }


def project_fixed_latent(
    *,
    event: sqlite3.Connection,
    date_values: Sequence[str],
    tickers: Sequence[str],
    components: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    dates = tuple(str(value) for value in date_values)
    date_to_code = {value: index for index, value in enumerate(dates)}
    ticker_to_code = {str(value): index for index, value in enumerate(tickers)}
    clean = np.zeros((len(dates), len(tickers)), dtype=np.float32)
    special = np.zeros_like(clean)
    clean_observed = np.zeros_like(clean, dtype=bool)
    special_observed = np.zeros_like(clean_observed)
    rows = event.execute(
        "SELECT signal_date,ticker,clean_eligible,special_eligible,"
        "observed_exact_t2,true_zero,flow_rate_pct,effective_sign,target_multiple "
        "FROM etf_flow_events WHERE signal_date BETWEEN ? AND ? AND strict_eligible=1 "
        "ORDER BY signal_date,ticker",
        (dates[0], dates[-1]),
    )
    read = 0
    mapped = 0
    for row in rows:
        read += 1
        date_code = date_to_code.get(str(row[0]))
        ticker_code = ticker_to_code.get(str(row[1]))
        if date_code is None or ticker_code is None:
            continue
        mapped += 1
        exact = bool(row[4]) or bool(row[5])
        if not exact:
            continue
        value = float(row[6] or 0.0)
        if bool(row[2]):
            clean[date_code, ticker_code] = float(np.clip(value, -25.0, 25.0))
            clean_observed[date_code, ticker_code] = True
        if bool(row[3]):
            effective = value * float(row[7] or 0.0) * abs(float(row[8] or 0.0))
            special[date_code, ticker_code] = float(np.clip(effective, -50.0, 50.0))
            special_observed[date_code, ticker_code] = True
    raw = np.column_stack([clean, special]).astype(np.float32)
    observed = np.column_stack([clean_observed, special_observed])
    mean5_raw = _masked_rolling_mean(raw, observed, 5)
    mean20_raw = _masked_rolling_mean(raw, observed, 20)
    observed5 = _masked_rolling_mean(observed.astype(np.float32), observed, 5) > 0
    observed20 = _masked_rolling_mean(observed.astype(np.float32), observed, 20) > 0
    current = _row_scale(raw, observed)
    mean5 = _row_scale(mean5_raw, observed5)
    mean20 = _row_scale(mean20_raw, observed20)
    states = {
        "current": current,
        "mean5": mean5,
        "mean20": mean20,
        "innovation": current - mean5,
        "convergence": mean5 - mean20,
    }
    components = np.asarray(components, dtype=np.float32)
    if components.shape != (LATENT_COMPONENTS, raw.shape[1]):
        raise ValueError(
            f"latent component shape mismatch: {components.shape} vs {raw.shape[1]}"
        )
    scores = np.column_stack(
        [states[name] @ components.T for name in STATE_NAMES]
    ).astype(np.float32)
    return scores, {
        "rows_read": read,
        "rows_mapped_to_training_tickers": mapped,
        "training_ticker_count": len(tickers),
        "date_count": len(dates),
        "absolute_flow_date_centered": False,
    }
