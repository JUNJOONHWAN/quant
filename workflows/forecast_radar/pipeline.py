"""Daily Forecast RADAR batch, aggregation, publication, and query helpers."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training.quant_flow_graph.data import build_dataset
from training.quant_flow_graph_v11_r2.phase_a import (
    build_event_cube,
    load_metadata,
    readonly_connection,
)
from training.quant_forecast_v2.panel import (
    DEFAULT_INDEX_EVIDENCE,
    build_panel,
)
from training.quant_forecast_v2.source import SourceBundle

from .contracts import (
    COVERAGE_GENERAL_SHADOW,
    COVERAGE_VALIDATED_CORE,
    DEFAULT_BASE_DATABASE,
    DEFAULT_ETFRADAR_ROOT,
    DEFAULT_INCREMENTAL_DATABASE,
    DEFAULT_LIVE_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PHASE_A_ROOT,
    INFORMATION_VALUE_EVIDENCE,
    RUN_SCHEMA_VERSION,
    TARGET_NAMES,
    TIMING_CONTRACT,
)
from .io import sha256_file, utc_now, write_json_atomic
from .live_features import (
    build_graph_session_overlay_database,
    build_live_source_database,
    build_live_stock_matrix,
    project_fixed_latent,
)
from .model_bundle import calibrate_probability, load_bundle


def _next_weekday(value: str) -> str:
    current = date.fromisoformat(value) + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current.isoformat()


def infer_signal_date(base_database: Path, incremental_database: Path) -> dict[str, str]:
    with SourceBundle(base_database, incremental_database) as source:
        sessions = source.sessions()
    if len(sessions) < 3:
        raise ValueError("SPY session calendar is too short")
    signal = _next_weekday(sessions[-1])
    return {
        "signal_date": signal,
        "price_date": sessions[-1],
        "flow_date": sessions[-2],
    }


def _reference_close_map(panel_path: Path, signal_date: str) -> dict[str, float]:
    connection = sqlite3.connect(f"file:{Path(panel_path)}?mode=ro", uri=True)
    try:
        return {
            str(symbol): float(reference)
            for symbol, reference in connection.execute(
                "SELECT symbol,reference_close FROM panel WHERE signal_date=? "
                "AND reference_close IS NOT NULL",
                (signal_date,),
            )
        }
    finally:
        connection.close()


def _short_history_symbols(panel_path: Path, signal_date: str) -> set[str]:
    connection = sqlite3.connect(f"file:{Path(panel_path)}?mode=ro", uri=True)
    try:
        return {
            str(symbol)
            for (symbol,) in connection.execute(
                "SELECT symbol FROM panel WHERE signal_date=? AND ret_120d IS NULL",
                (signal_date,),
            )
        }
    finally:
        connection.close()


def _predict_latest(
    *,
    bundle: Mapping[str, Any],
    matrix: Mapping[str, Any],
    latent_scores: np.ndarray,
    signal_date: str,
    reference_close: Mapping[str, float],
    profiles: Mapping[str, Mapping[str, Any]],
    short_history_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    manifest = bundle["manifest"]
    if tuple(matrix["price_names"]) != tuple(manifest["feature_contract"]["price_names"]):
        raise ValueError("live price feature contract differs from model bundle")
    if tuple(matrix["flow_names"]) != tuple(manifest["feature_contract"]["flow_names"]):
        raise ValueError("live flow feature contract differs from model bundle")
    date_index = matrix["date_values"].index(signal_date)
    selected = np.flatnonzero(np.asarray(matrix["date_codes"]) == date_index)
    if not len(selected):
        raise ValueError(f"live matrix has no rows for {signal_date}")
    price = np.asarray(matrix["price_matrix"], dtype=np.float32)[selected]
    flow = np.asarray(matrix["flow_matrix"], dtype=np.float32)[selected]
    global_rows = np.repeat(
        np.asarray(latent_scores[date_index], dtype=np.float32)[None, :],
        len(selected),
        axis=0,
    )
    enriched = np.column_stack([price, flow, global_rows]).astype(np.float32)
    price_prediction = np.asarray(bundle["price_model"].predict(price), dtype=np.float32)
    enriched_prediction = np.asarray(
        bundle["enriched_model"].predict(enriched), dtype=np.float32
    )
    caps = np.asarray(bundle["latent"]["residual_caps"], dtype=np.float32)
    correction = 0.25 * (enriched_prediction - price_prediction)
    prediction = price_prediction + np.clip(correction, -caps, caps)
    p5 = calibrate_probability(
        prediction[:, 0], bundle["calibration"]["heads"]["p_up_5d"]
    )
    p20 = calibrate_probability(
        prediction[:, 6], bundle["calibration"]["heads"]["p_up_20d"]
    )
    validated = set(manifest["coverage"]["validated_core_symbols"])
    short_history = short_history_symbols or set()
    symbols = tuple(matrix["symbol_values"])
    symbol_codes = np.asarray(matrix["symbol_codes"], dtype=np.int64)[selected]
    rows: list[dict[str, Any]] = []
    for local, symbol_code in enumerate(symbol_codes):
        symbol = str(symbols[int(symbol_code)])
        profile = profiles.get(symbol, {})
        values = prediction[local]
        row = {
            "symbol": symbol,
            "sector": str(profile.get("sector") or "Unknown"),
            "industry": str(profile.get("industry") or "Unknown"),
            "reference_close": reference_close.get(symbol),
            "coverage_tier": (
                COVERAGE_VALIDATED_CORE if symbol in validated else COVERAGE_GENERAL_SHADOW
            ),
            "validation_status": (
                "HISTORICAL_OOS_CORE"
                if symbol in validated
                else (
                    "EXTRAPOLATED_UNVALIDATED_SHORT_HISTORY"
                    if symbol in short_history
                    else "EXTRAPOLATED_UNVALIDATED"
                )
            ),
            "p_up_5d": float(p5[local]),
            "p_up_20d": float(p20[local]),
        }
        for target_index, target_name in enumerate(TARGET_NAMES):
            row[target_name] = float(values[target_index])
        row["asymmetry_5d"] = row["upside_5d_pct"] - row["loss_5d_pct"]
        row["asymmetry_20d"] = row["upside_20d_pct"] - row["loss_20d_pct"]
        row["utility_5d"] = (
            row["p_up_5d"] * row["upside_5d_pct"]
            - (1.0 - row["p_up_5d"]) * row["loss_5d_pct"]
        )
        row["utility_20d"] = (
            row["p_up_20d"] * row["upside_20d_pct"]
            - (1.0 - row["p_up_20d"]) * row["loss_20d_pct"]
        )
        rows.append(row)
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]], key: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    if key is None:
        grouped["MARKET"] = list(rows)
    else:
        for row in rows:
            grouped.setdefault(str(row.get(key) or "Unknown"), []).append(row)
    result = []
    for name, items in sorted(grouped.items()):
        if not items:
            continue
        p5 = np.asarray([float(item["p_up_5d"]) for item in items])
        p20 = np.asarray([float(item["p_up_20d"]) for item in items])
        ret5 = np.asarray([float(item["return_5d_pct"]) for item in items])
        ret20 = np.asarray([float(item["return_20d_pct"]) for item in items])
        upside5 = np.asarray([float(item["upside_5d_pct"]) for item in items])
        loss5 = np.asarray([float(item["loss_5d_pct"]) for item in items])
        net5 = float(np.mean(p5 >= 0.55) - np.mean(p5 <= 0.45))
        net20 = float(np.mean(p20 >= 0.55) - np.mean(p20 <= 0.45))
        mean_probability = float(np.mean((p5 + p20) / 2.0))
        mean_breadth = (net5 + net20) / 2.0
        if mean_probability >= 0.58 and mean_breadth >= 0.20:
            label = "RISK_ON_SHADOW"
        elif mean_probability <= 0.42 and mean_breadth <= -0.20:
            label = "RISK_OFF_SHADOW"
        elif abs(mean_breadth) >= 0.10:
            label = "TRANSITION_SHADOW"
        else:
            label = "NEUTRAL_SHADOW"
        result.append(
            {
                "group": name,
                "stock_count": len(items),
                "validated_core_count": sum(
                    item["coverage_tier"] == COVERAGE_VALIDATED_CORE for item in items
                ),
                "mean_p_up_5d": float(np.mean(p5)),
                "mean_p_up_20d": float(np.mean(p20)),
                "net_breadth_5d": net5,
                "net_breadth_20d": net20,
                "median_return_5d_pct": float(np.median(ret5)),
                "median_return_20d_pct": float(np.median(ret20)),
                "median_upside_5d_pct": float(np.median(upside5)),
                "median_loss_5d_pct": float(np.median(loss5)),
                "return_dispersion_5d_pct": float(np.std(ret5, ddof=1)) if len(ret5) > 1 else 0.0,
                "loss_90pct_5d_pct": float(np.quantile(loss5, 0.90)),
                "regime_label": label,
                "regime_status": "SHADOW_HEURISTIC_FROM_STOCK_AGGREGATE",
            }
        )
    return result


def _build_coverage_gate(
    *,
    general_universe_symbols: Sequence[str],
    live_panel_symbols: Sequence[str],
    forecasts: Sequence[Mapping[str, Any]],
    panel_live_general_shadow_source_symbol_count: int,
) -> dict[str, Any]:
    """Validate live panel/forecast coverage without conflating source and model tiers."""
    candidate_symbols = {str(symbol) for symbol in general_universe_symbols}
    expected_panel_symbols = {str(symbol) for symbol in live_panel_symbols}
    forecast_symbols = {str(row["symbol"]) for row in forecasts}
    validated_core_count = sum(
        row["coverage_tier"] == COVERAGE_VALIDATED_CORE for row in forecasts
    )
    general_shadow_count = sum(
        row["coverage_tier"] == COVERAGE_GENERAL_SHADOW for row in forecasts
    )
    missing_general_candidates = sorted(candidate_symbols.difference(forecast_symbols))
    missing_live_panel_symbols = sorted(expected_panel_symbols.difference(forecast_symbols))
    unexpected_forecast_symbols = sorted(forecast_symbols.difference(expected_panel_symbols))
    gate = {
        "eligible_general_universe_symbol_count": len(candidate_symbols),
        "panel_live_symbol_count": len(expected_panel_symbols),
        "panel_live_general_shadow_source_symbol_count": int(
            panel_live_general_shadow_source_symbol_count
        ),
        "forecast_symbol_count": len(forecast_symbols),
        "forecast_general_shadow_symbol_count": general_shadow_count,
        "validated_core_forecast_count": validated_core_count,
        "short_history_forecast_count": sum(
            row["validation_status"] == "EXTRAPOLATED_UNVALIDATED_SHORT_HISTORY"
            for row in forecasts
        ),
        "missing_general_candidate_count": len(missing_general_candidates),
        "missing_general_candidate_symbols": missing_general_candidates,
        "missing_live_panel_symbol_count": len(missing_live_panel_symbols),
        "missing_live_panel_symbols": missing_live_panel_symbols,
        "unexpected_forecast_symbol_count": len(unexpected_forecast_symbols),
        "unexpected_forecast_symbols": unexpected_forecast_symbols,
        "general_universe_minimum_gate": len(candidate_symbols) >= 1_000,
        "live_panel_forecast_parity_gate": (
            not missing_live_panel_symbols and not unexpected_forecast_symbols
        ),
        "general_candidate_full_coverage_gate": not missing_general_candidates,
        "validated_core_minimum_gate": validated_core_count >= 400,
    }
    gate["status"] = (
        "PASS"
        if all(
            gate[key]
            for key in (
                "general_universe_minimum_gate",
                "live_panel_forecast_parity_gate",
                "general_candidate_full_coverage_gate",
                "validated_core_minimum_gate",
            )
        )
        else "FAIL"
    )
    return gate


def _write_forecast_database(
    *,
    path: Path,
    run: Mapping[str, Any],
    forecasts: Sequence[Mapping[str, Any]],
    market: Mapping[str, Any],
    sectors: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".building", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=FULL;
            CREATE TABLE run_manifest(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
            CREATE TABLE stock_forecasts(
              symbol TEXT PRIMARY KEY,sector TEXT,industry TEXT,reference_close REAL,
              coverage_tier TEXT NOT NULL,validation_status TEXT NOT NULL,
              p_up_5d REAL,p_up_20d REAL,
              return_5d_pct REAL,upside_5d_pct REAL,loss_5d_pct REAL,
              benchmark_excess_return_5d_pct REAL,
              benchmark_upside_capture_5d_pct REAL,
              benchmark_downside_defense_5d_pct REAL,
              return_20d_pct REAL,upside_20d_pct REAL,loss_20d_pct REAL,
              benchmark_excess_return_20d_pct REAL,
              benchmark_upside_capture_20d_pct REAL,
              benchmark_downside_defense_20d_pct REAL,
              asymmetry_5d REAL,asymmetry_20d REAL,utility_5d REAL,utility_20d REAL
            );
            CREATE TABLE market_regime(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
            CREATE TABLE sector_regimes(sector TEXT PRIMARY KEY,value_json TEXT NOT NULL);
            """
        )
        connection.executemany(
            "INSERT INTO run_manifest VALUES(?,?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in run.items()],
        )
        columns = (
            "symbol", "sector", "industry", "reference_close", "coverage_tier",
            "validation_status", "p_up_5d", "p_up_20d", *TARGET_NAMES,
            "asymmetry_5d", "asymmetry_20d", "utility_5d", "utility_20d",
        )
        connection.executemany(
            f"INSERT INTO stock_forecasts VALUES({','.join('?' for _ in columns)})",
            [tuple(row.get(column) for column in columns) for row in forecasts],
        )
        connection.executemany(
            "INSERT INTO market_regime VALUES(?,?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in market.items()],
        )
        connection.executemany(
            "INSERT INTO sector_regimes VALUES(?,?)",
            [
                (str(row["group"]), json.dumps(dict(row), ensure_ascii=False))
                for row in sectors
            ],
        )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"forecast sqlite integrity failure: {integrity}")
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, path)


def current_completed_run(live_root: Path, signal_date: str) -> dict[str, Any] | None:
    """Return a verified current run, or None when a fresh batch is required."""

    latest_path = Path(live_root) / "latest.json"
    if not latest_path.is_file():
        return None
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if latest.get("signal_date") != signal_date:
            return None
        summary_path = Path(latest["summary_path"])
        database_path = Path(latest["database_path"])
        if sha256_file(summary_path) != latest["summary_sha256"]:
            return None
        if sha256_file(database_path) != latest["database_sha256"]:
            return None
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema_version") != RUN_SCHEMA_VERSION:
            return None
        if summary.get("quality_gate") != "PASS_SHADOW_RUN":
            return None
        if summary.get("signal_date") != signal_date:
            return None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return {
        "quality_gate": "NOOP_ALREADY_CURRENT",
        "activation_status": latest.get("activation_status", "SHADOW_ONLY"),
        "run_id": latest["run_id"],
        "signal_date": signal_date,
        "price_date": summary.get("price_date"),
        "flow_date": summary.get("flow_date"),
        "stock_count": summary.get("stock_count"),
        "validated_core_count": summary.get("validated_core_count"),
        "general_shadow_count": summary.get("general_shadow_count"),
        "summary_path": str(summary_path),
        "database_path": str(database_path),
        "reason": "verified PASS_SHADOW_RUN already exists for signal_date",
    }


def run_daily(
    *,
    signal_date: str | None,
    if_needed: bool = False,
    base_database: Path = DEFAULT_BASE_DATABASE,
    incremental_database: Path = DEFAULT_INCREMENTAL_DATABASE,
    index_evidence: Path = DEFAULT_INDEX_EVIDENCE,
    phase_a_root: Path = DEFAULT_PHASE_A_ROOT,
    etfradar_root: Path = DEFAULT_ETFRADAR_ROOT,
    model_root: Path = DEFAULT_MODEL_ROOT,
    live_root: Path = DEFAULT_LIVE_ROOT,
) -> dict[str, Any]:
    timing = infer_signal_date(base_database, incremental_database)
    if signal_date is not None:
        timing["signal_date"] = signal_date
    signal = timing["signal_date"]
    if if_needed:
        current = current_completed_run(live_root, signal)
        if current is not None:
            return current
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = Path(live_root) / "runs" / run_id
    panel_root = run_root / "panel"
    graph_root = run_root / "graph"
    run_root.mkdir(parents=True, exist_ok=False)
    history_start = (date.fromisoformat(signal) - timedelta(days=420)).isoformat()

    with SourceBundle(base_database, incremental_database) as source:
        general_shadow_symbols, general_universe_audit = (
            source.active_us_exchange_stock_universe(timing["price_date"])
        )
        profiles = source.latest_company_profiles()
    if general_universe_audit["eligible_symbol_count"] < 1_000:
        raise RuntimeError(
            "general stock universe gate failed: fewer than 1000 eligible symbols"
        )

    panel_manifest = build_panel(
        base_database=base_database,
        incremental_database=incremental_database,
        index_evidence=index_evidence,
        output_root=panel_root,
        start_date=(date.fromisoformat(signal) - timedelta(days=300)).isoformat(),
        end_date=signal,
        live_signal_date=signal,
        replace=False,
        general_shadow_symbols=general_shadow_symbols,
    )
    session_overlay_path = run_root / "graph_session_overlay.sqlite3"
    session_overlay_receipt = build_graph_session_overlay_database(
        incremental_database=incremental_database,
        output_path=session_overlay_path,
        signal_date=signal,
    )
    graph_manifest = build_dataset(
        panel_path=panel_root / "panel.sqlite3",
        flow_cache_path=panel_root / "flow_cache.sqlite3",
        base_database=base_database,
        incremental_database=session_overlay_path,
        output_root=graph_root,
        start_date=(date.fromisoformat(signal) - timedelta(days=300)).isoformat(),
        end_date=signal,
        symbols=None,
        replace=False,
        smoke_only=False,
    )
    live_source_path = run_root / "live_source.sqlite3"
    source_receipt = build_live_source_database(
        base_database=base_database,
        incremental_database=incremental_database,
        output_path=live_source_path,
        history_start=history_start,
        signal_date=signal,
        replace=False,
    )
    event_path = run_root / "live_event_cube.sqlite3"
    source = readonly_connection(live_source_path)
    try:
        event_receipt = build_event_cube(
            source=source,
            metadata=load_metadata(etfradar_root),
            family_registry_path=(
                Path(phase_a_root) / "v11_r2_etf_family_exposure_registry.sqlite3"
            ),
            output_path=event_path,
            start_date=None,
            end_date=signal,
        )
    finally:
        source.close()
    event = readonly_connection(event_path)
    source = readonly_connection(live_source_path)
    try:
        matrix = build_live_stock_matrix(
            event=event,
            source=source,
            graph_dataset_root=graph_root,
        )
        bundle = load_bundle(model_root)
        latent_scores, latent_audit = project_fixed_latent(
            event=event,
            date_values=matrix["date_values"],
            tickers=tuple(str(value) for value in bundle["latent"]["tickers"]),
            components=np.asarray(bundle["latent"]["components"], dtype=np.float32),
        )
    finally:
        source.close()
        event.close()
    references = _reference_close_map(panel_root / "panel.sqlite3", signal)
    short_history = _short_history_symbols(panel_root / "panel.sqlite3", signal)
    forecasts = _predict_latest(
        bundle=bundle,
        matrix=matrix,
        latent_scores=latent_scores,
        signal_date=signal,
        reference_close=references,
        profiles=profiles,
        short_history_symbols=short_history,
    )
    validated_core_forecasts = [
        row
        for row in forecasts
        if row["coverage_tier"] == COVERAGE_VALIDATED_CORE
    ]
    general_shadow_forecasts = [
        row
        for row in forecasts
        if row["coverage_tier"] == COVERAGE_GENERAL_SHADOW
    ]
    coverage_gate = _build_coverage_gate(
        general_universe_symbols=general_shadow_symbols,
        live_panel_symbols=references,
        forecasts=forecasts,
        panel_live_general_shadow_source_symbol_count=int(
            panel_manifest["price_phase"]["live_general_shadow_inserted_symbol_count"]
        ),
    )
    if coverage_gate["status"] != "PASS":
        raise RuntimeError(f"Forecast RADAR coverage gate failed: {coverage_gate}")

    market_rows = _aggregate(validated_core_forecasts)
    if len(market_rows) != 1:
        raise RuntimeError("market aggregation did not produce exactly one row")
    exploratory_market_rows = _aggregate(forecasts)
    market = {
        **market_rows[0],
        "aggregation_scope": "VALIDATED_CORE_ONLY",
        "total_forecast_universe_count": len(forecasts),
        "general_shadow_count": len(general_shadow_forecasts),
        "exploratory_full_universe_shadow": exploratory_market_rows[0],
    }
    sectors = _aggregate(validated_core_forecasts, "sector")
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        **timing,
        "stock_count": len(forecasts),
        "validated_core_count": len(validated_core_forecasts),
        "general_shadow_count": len(general_shadow_forecasts),
        "activation_status": "SHADOW_ONLY",
        "model_manifest_sha256": sha256_file(Path(model_root) / "model_manifest.json"),
    }
    database_path = run_root / "forecast_radar.sqlite3"
    _write_forecast_database(
        path=database_path,
        run=run_manifest,
        forecasts=forecasts,
        market=market,
        sectors=sectors,
    )
    summary = {
        **run_manifest,
        "market": market,
        "sector_count": len(sectors),
        "top_utility_5d": sorted(
            validated_core_forecasts,
            key=lambda row: float(row["utility_5d"]),
            reverse=True,
        )[:20],
        "bottom_utility_5d": sorted(
            validated_core_forecasts, key=lambda row: float(row["utility_5d"])
        )[:20],
        "exploratory_general_universe_top_utility_5d": sorted(
            general_shadow_forecasts,
            key=lambda row: float(row["utility_5d"]),
            reverse=True,
        )[:20],
        "exploratory_general_universe_bottom_utility_5d": sorted(
            general_shadow_forecasts, key=lambda row: float(row["utility_5d"])
        )[:20],
        "coverage_gate": coverage_gate,
        "general_universe": general_universe_audit,
        "sources": {
            "panel_quality": panel_manifest["quality"],
            "graph_quality_gate": graph_manifest["quality_gate"],
            "live_source": source_receipt,
            "graph_session_overlay": session_overlay_receipt,
            "event_cube": event_receipt,
            "latent": latent_audit,
        },
        "artifacts": {
            "database": {"path": str(database_path), "sha256": sha256_file(database_path)},
        },
        "quality_gate": "PASS_SHADOW_RUN",
    }
    summary_path = run_root / "summary.json"
    write_json_atomic(summary_path, summary)
    latest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "signal_date": signal,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "database_path": str(database_path),
        "database_sha256": sha256_file(database_path),
        "activation_status": "SHADOW_ONLY",
    }
    write_json_atomic(Path(live_root) / "latest.json", latest)
    return summary


def query_latest(
    *,
    live_root: Path,
    symbol: str | None = None,
    sector: str | None = None,
) -> dict[str, Any]:
    latest = json.loads((Path(live_root) / "latest.json").read_text(encoding="utf-8"))
    summary_path = Path(latest["summary_path"])
    if sha256_file(summary_path) != latest["summary_sha256"]:
        raise ValueError("latest Forecast RADAR summary hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    latest_view = {
        **latest,
        "price_date": summary.get("price_date"),
        "flow_date": summary.get("flow_date"),
        "quality_gate": summary.get("quality_gate"),
        "stock_count": summary.get("stock_count"),
        "validated_core_count": summary.get("validated_core_count"),
        "general_shadow_count": summary.get("general_shadow_count"),
        "sector_count": summary.get("sector_count"),
    }
    database_path = Path(latest["database_path"])
    if sha256_file(database_path) != latest["database_sha256"]:
        raise ValueError("latest Forecast RADAR database hash mismatch")
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        probability_row = connection.execute(
            "SELECT COUNT(DISTINCT p_up_5d),COUNT(DISTINCT p_up_20d),"
            "MIN(p_up_5d),MAX(p_up_5d),MIN(p_up_20d),MAX(p_up_20d) "
            "FROM stock_forecasts"
        ).fetchone()
        probability_resolution = {
            "p_up_5d_distinct_count": int(probability_row[0]),
            "p_up_20d_distinct_count": int(probability_row[1]),
            "p_up_5d_min": float(probability_row[2]),
            "p_up_5d_max": float(probability_row[3]),
            "p_up_20d_min": float(probability_row[4]),
            "p_up_20d_max": float(probability_row[5]),
            "interpretation": "LOW_RESOLUTION_CALIBRATION_DO_NOT_OVERSTATE_DECIMALS",
        }
        if symbol:
            row = connection.execute(
                "SELECT * FROM stock_forecasts WHERE symbol=?",
                (symbol.strip().upper().replace(".", "-"),),
            ).fetchone()
            stock = dict(row) if row else None
            relative_position = (
                _stock_relative_position(connection, stock) if stock else None
            )
            return {
                "latest": latest_view,
                "probability_resolution": probability_resolution,
                "information_value": INFORMATION_VALUE_EVIDENCE,
                "interpretation_contract": {
                    "primary_use": "RANGE_RISK_MAP_AND_RELATIVE_POTENTIAL_RANKING",
                    "holder_questions": [
                        "HOW_MUCH_UPSIDE_POTENTIAL_REMAINS",
                        "HOW_LARGE_A_DOWNSIDE_EXCURSION_IS_PLAUSIBLE",
                        "WHETHER_UPSIDE_OR_DOWNSIDE_DOMINATES",
                        "HOW_THE_STOCK_RANKS_VERSUS_THE_UNIVERSE_AND_SECTOR",
                        "HOW_IT_MAY_CAPTURE_UPSIDE_OR_DEFEND_DOWNSIDE_VERSUS_BENCHMARK",
                    ],
                    "probability_warning": (
                        "p_up is low-resolution calibration context, not a precise hit probability"
                    ),
                    "trade_mapping": "NONE_INFORMATION_PRODUCT_ONLY",
                },
                "stock": stock,
                "relative_position": relative_position,
            }
        if sector:
            row = connection.execute(
                "SELECT value_json FROM sector_regimes WHERE lower(sector)=lower(?)",
                (sector,),
            ).fetchone()
            return {
                "latest": latest_view,
                "probability_resolution": probability_resolution,
                "sector": json.loads(row[0]) if row else None,
            }
        market = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key,value_json FROM market_regime")
        }
        return {
            "latest": latest_view,
            "probability_resolution": probability_resolution,
            "information_value": INFORMATION_VALUE_EVIDENCE,
            "market": market,
        }
    finally:
        connection.close()


RELATIVE_POSITION_METRICS = (
    "upside_5d_pct",
    "upside_20d_pct",
    "loss_5d_pct",
    "loss_20d_pct",
    "asymmetry_5d",
    "asymmetry_20d",
    "benchmark_downside_defense_5d_pct",
    "benchmark_downside_defense_20d_pct",
    "utility_5d",
    "utility_20d",
)


def _relative_metric(
    rows: Sequence[Mapping[str, Any]], metric: str, target: float
) -> dict[str, Any]:
    values = [
        float(row[metric])
        for row in rows
        if row.get(metric) is not None and math.isfinite(float(row[metric]))
    ]
    if not values:
        return {"value": target, "count": 0, "rank_high_to_low": None, "percentile": None}
    above = sum(value > target for value in values)
    below = sum(value < target for value in values)
    equal = len(values) - above - below
    return {
        "value": target,
        "count": len(values),
        "rank_high_to_low": above + 1,
        "tie_count": equal,
        "percentile": 100.0 * (below + 0.5 * equal) / len(values),
    }


def _stock_relative_position(
    connection: sqlite3.Connection, stock: Mapping[str, Any]
) -> dict[str, Any]:
    selected = ",".join(("symbol", "sector", *RELATIVE_POSITION_METRICS))
    rows = [dict(row) for row in connection.execute(f"SELECT {selected} FROM stock_forecasts")]
    sector = str(stock.get("sector") or "")
    sector_rows = [row for row in rows if str(row.get("sector") or "") == sector]
    universe: dict[str, Any] = {}
    sector_view: dict[str, Any] = {}
    for metric in RELATIVE_POSITION_METRICS:
        value = stock.get(metric)
        if value is None or not math.isfinite(float(value)):
            continue
        target = float(value)
        universe[metric] = _relative_metric(rows, metric, target)
        sector_view[metric] = _relative_metric(sector_rows, metric, target)
    return {
        "percentile_definition": (
            "0=lower predicted value, 100=higher predicted value; for loss metrics, "
            "a higher percentile means greater downside risk"
        ),
        "universe_count": len(rows),
        "sector": sector or None,
        "sector_count": len(sector_rows),
        "universe": universe,
        "sector_view": sector_view,
    }
