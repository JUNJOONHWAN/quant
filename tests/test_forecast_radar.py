from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from workflows.forecast_radar import cli
from workflows.forecast_radar.contracts import (
    COVERAGE_VALIDATED_CORE,
    TARGET_NAMES,
    TIMING_CONTRACT,
)
from workflows.forecast_radar.live_features import (
    build_graph_session_overlay_database,
    build_live_source_database,
)
from workflows.forecast_radar.model_bundle import (
    calibrate_probability,
    observed_graph_symbols,
    validated_symbols_from_matrix,
)
from workflows.forecast_radar.io import sha256_file
from workflows.forecast_radar.pipeline import (
    _aggregate,
    current_completed_run,
    query_latest,
)


def _source(path: Path, *, incremental: bool) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE daily_observations(
          source TEXT,symbol TEXT,trade_date TEXT,open REAL,high REAL,low REAL,
          close REAL,adjusted_close REAL,volume REAL,
          PRIMARY KEY(source,symbol,trade_date)
        );
        CREATE TABLE etf_flow_observations(
          provider TEXT,ticker TEXT,effective_date TEXT,processed_date TEXT,
          fund_flow REAL,nav REAL,shares_outstanding REAL,available_at_date TEXT,
          PRIMARY KEY(provider,ticker,effective_date)
        );
        """
    )
    close = 102.0 if incremental else 100.0
    connection.execute(
        "INSERT INTO daily_observations VALUES(?,?,?,?,?,?,?,?,?)",
        ("fmp", "SPY", "2026-08-25", close, close, close, close, close, 1_000_000),
    )
    connection.execute(
        "INSERT INTO etf_flow_observations VALUES(?,?,?,?,?,?,?,?)",
        (
            "massive",
            "SPY",
            "2026-08-25",
            "2026-08-25",
            1_000_000.0 if incremental else 500_000.0,
            500.0,
            1_000_000.0,
            "2026-08-26",
        ),
    )
    connection.commit()
    connection.close()


def test_contract_has_exact_t_minus_2_flow() -> None:
    assert "exactly T-2" in TIMING_CONTRACT
    assert len(TARGET_NAMES) == 12


def test_calibration_is_clipped_by_threshold_endpoints() -> None:
    head = {"x_thresholds": [-1.0, 0.0, 1.0], "y_thresholds": [0.2, 0.5, 0.8]}
    observed = calibrate_probability(np.asarray([-5.0, 0.0, 5.0]), head)
    assert np.allclose(observed, [0.2, 0.5, 0.8])


def test_coverage_comes_from_snapshot_symbols_not_all_panel_sentinel(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot.npz"
    np.savez_compressed(snapshot, stock_symbols=np.asarray(["A", "AAPL", "NVDA"]))
    (tmp_path / "manifest.json").write_text(
        '{"requested_symbols":"ALL_PANEL","snapshots":'
        f'[{{"path":"{snapshot}"}}]}}',
        encoding="utf-8",
    )
    assert observed_graph_symbols(tmp_path) == ["A", "AAPL", "NVDA"]


def test_validated_coverage_only_counts_symbols_with_training_rows() -> None:
    matrix = {
        "symbol_values": ("A", "A", "E", "L", "L", "L", "N", "P", "_", "AAPL"),
        "symbol_codes": np.asarray([1, 2, 5, 6, 7, 9, 9]),
    }
    assert validated_symbols_from_matrix(matrix) == ["A", "AAPL", "E", "L", "N", "P"]


def test_stock_aggregate_is_the_regime_source() -> None:
    rows = [
        {
            "coverage_tier": COVERAGE_VALIDATED_CORE,
            "p_up_5d": 0.70,
            "p_up_20d": 0.65,
            "return_5d_pct": 1.0,
            "return_20d_pct": 2.0,
            "upside_5d_pct": 3.0,
            "loss_5d_pct": 1.0,
        },
        {
            "coverage_tier": COVERAGE_VALIDATED_CORE,
            "p_up_5d": 0.68,
            "p_up_20d": 0.62,
            "return_5d_pct": 0.8,
            "return_20d_pct": 1.5,
            "upside_5d_pct": 2.5,
            "loss_5d_pct": 1.2,
        },
    ]
    result = _aggregate(rows)[0]
    assert result["group"] == "MARKET"
    assert result["stock_count"] == 2
    assert result["net_breadth_5d"] == 1.0
    assert result["regime_label"] == "RISK_ON_SHADOW"
    assert result["regime_status"] == "SHADOW_HEURISTIC_FROM_STOCK_AGGREGATE"


def test_live_source_is_isolated_and_increment_wins(tmp_path: Path) -> None:
    base = tmp_path / "base.sqlite3"
    incremental = tmp_path / "incremental.sqlite3"
    output = tmp_path / "live.sqlite3"
    _source(base, incremental=False)
    _source(incremental, incremental=True)
    receipt = build_live_source_database(
        base_database=base,
        incremental_database=incremental,
        output_path=output,
        history_start="2026-08-01",
        signal_date="2026-08-27",
        replace=False,
    )
    assert receipt["quality_gate"] == "PASS"
    assert receipt["canonical_sources_mutated"] is False
    connection = sqlite3.connect(output)
    try:
        assert connection.execute(
            "SELECT close FROM daily_observations WHERE symbol='SPY' "
            "AND trade_date='2026-08-25'"
        ).fetchone()[0] == 102.0
        assert connection.execute(
            "SELECT fund_flow FROM etf_flow_observations WHERE ticker='SPY'"
        ).fetchone()[0] == 1_000_000.0
        assert connection.execute(
            "SELECT COUNT(*) FROM daily_observations WHERE trade_date='2026-08-27'"
        ).fetchone()[0] == 1
    finally:
        connection.close()


def test_graph_session_overlay_isolated_and_adds_live_t(tmp_path: Path) -> None:
    incremental = tmp_path / "incremental.sqlite3"
    connection = sqlite3.connect(incremental)
    connection.execute(
        "CREATE TABLE daily_observations("
        "source TEXT,symbol TEXT,trade_date TEXT,open REAL,high REAL,low REAL,"
        "close REAL,adjusted_close REAL,volume REAL,vwap REAL)"
    )
    for symbol in ("SPY", "QQQ", "NVDA"):
        connection.execute(
            "INSERT INTO daily_observations VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("fmp", symbol, "2026-08-26", 1, 1, 1, 1, 1, 100, 1),
        )
    connection.commit()
    connection.close()
    source_hash = incremental.read_bytes()
    output = tmp_path / "overlay.sqlite3"
    receipt = build_graph_session_overlay_database(
        incremental_database=incremental,
        output_path=output,
        signal_date="2026-08-27",
    )
    assert receipt["quality_gate"] == "PASS"
    assert receipt["canonical_source_mutated"] is False
    assert incremental.read_bytes() == source_hash
    connection = sqlite3.connect(output)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM daily_observations WHERE trade_date='2026-08-27'"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM daily_observations WHERE symbol='NVDA'"
        ).fetchone()[0] == 0
    finally:
        connection.close()


def test_current_completed_run_is_idempotent_and_hash_gated(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "run-1"
    run.mkdir(parents=True)
    database = run / "forecast.sqlite3"
    database.write_bytes(b"immutable forecast")
    summary = run / "summary.json"
    summary.write_text(
        '{"quality_gate":"PASS_SHADOW_RUN","signal_date":"2026-08-27",'
        '"price_date":"2026-08-26","flow_date":"2026-08-25",'
        '"stock_count":479,"validated_core_count":477,"general_shadow_count":2}',
        encoding="utf-8",
    )
    latest = {
        "run_id": "run-1",
        "signal_date": "2026-08-27",
        "activation_status": "SHADOW_ONLY",
        "summary_path": str(summary),
        "summary_sha256": sha256_file(summary),
        "database_path": str(database),
        "database_sha256": sha256_file(database),
    }
    (tmp_path / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    observed = current_completed_run(tmp_path, "2026-08-27")
    assert observed is not None
    assert observed["quality_gate"] == "NOOP_ALREADY_CURRENT"
    database.write_bytes(b"tampered")
    assert current_completed_run(tmp_path, "2026-08-27") is None


def test_query_latest_exposes_timing_and_probability_resolution(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "run-1"
    run.mkdir(parents=True)
    database = run / "forecast.sqlite3"
    connection = sqlite3.connect(database)
    metric_columns = (
        "upside_5d_pct,upside_20d_pct,loss_5d_pct,loss_20d_pct,"
        "asymmetry_5d,asymmetry_20d,benchmark_downside_defense_5d_pct,"
        "benchmark_downside_defense_20d_pct,utility_5d,utility_20d"
    )
    connection.execute(
        f"CREATE TABLE stock_forecasts(symbol TEXT PRIMARY KEY,sector TEXT,"
        f"p_up_5d REAL,p_up_20d REAL,{metric_columns.replace(',', ' REAL,')} REAL)"
    )
    connection.executemany(
        "INSERT INTO stock_forecasts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            ("AAPL", "Technology", 0.51, 0.52, 2, 5, 1, 3, 1, 2, 0.1, 0.2, 0.3, 0.4),
            ("NVDA", "Technology", 0.53, 0.52, 4, 8, 3, 7, 1, 1, -0.2, -0.4, 0.6, 0.7),
            ("MSFT", "Technology", 0.51, 0.54, 3, 6, 2, 4, 1, 2, 0.0, 0.1, 0.4, 0.5),
        ),
    )
    connection.execute("CREATE TABLE sector_regimes(sector TEXT,value_json TEXT)")
    connection.execute("CREATE TABLE market_regime(key TEXT,value_json TEXT)")
    connection.commit()
    connection.close()
    summary = run / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "quality_gate": "PASS_SHADOW_RUN",
                "signal_date": "2026-08-27",
                "price_date": "2026-08-26",
                "flow_date": "2026-08-25",
                "stock_count": 3,
                "validated_core_count": 3,
                "general_shadow_count": 0,
                "sector_count": 2,
            }
        ),
        encoding="utf-8",
    )
    latest = {
        "run_id": "run-1",
        "signal_date": "2026-08-27",
        "activation_status": "SHADOW_ONLY",
        "summary_path": str(summary),
        "summary_sha256": sha256_file(summary),
        "database_path": str(database),
        "database_sha256": sha256_file(database),
    }
    (tmp_path / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    observed = query_latest(live_root=tmp_path, symbol="NVDA")
    assert observed["latest"]["price_date"] == "2026-08-26"
    assert observed["latest"]["flow_date"] == "2026-08-25"
    assert observed["probability_resolution"]["p_up_5d_distinct_count"] == 2
    assert observed["probability_resolution"]["p_up_20d_distinct_count"] == 2
    assert observed["stock"]["symbol"] == "NVDA"
    assert observed["relative_position"]["universe_count"] == 3
    assert observed["relative_position"]["universe"]["upside_5d_pct"]["rank_high_to_low"] == 1
    assert observed["relative_position"]["universe"]["loss_5d_pct"]["percentile"] > 80
    assert observed["information_value"]["validated_paths"] == [
        "DISTRIBUTION_FORECAST",
        "UPSIDE_DOWNSIDE_POTENTIAL",
    ]
    assert observed["interpretation_contract"]["trade_mapping"] == "NONE_INFORMATION_PRODUCT_ONLY"


def test_scheduled_daily_returns_one_compact_verified_payload(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_daily(**kwargs):
        calls.append(kwargs)
        return {"quality_gate": "NOOP_ALREADY_CURRENT", "run_id": "run-1"}

    def fake_query_latest(**kwargs):
        assert kwargs == {"live_root": tmp_path}
        return {
            "latest": {
                "run_id": "run-1",
                "quality_gate": "PASS_SHADOW_RUN",
                "database_sha256": "db-hash",
                "summary_sha256": "summary-hash",
            },
            "probability_resolution": {
                "p_up_5d_distinct_count": 11,
                "p_up_20d_distinct_count": 3,
            },
            "market": {"large": "must not leak into scheduled output"},
        }

    monkeypatch.setattr(cli, "run_daily", fake_run_daily)
    monkeypatch.setattr(cli, "query_latest", fake_query_latest)
    assert cli.main(["scheduled-daily", "--live-root", str(tmp_path)]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert calls == [
        {"signal_date": None, "if_needed": True, "live_root": tmp_path}
    ]
    assert observed["batch"]["quality_gate"] == "NOOP_ALREADY_CURRENT"
    assert observed["latest"]["database_sha256"] == "db-hash"
    assert observed["probability_resolution"]["p_up_5d_distinct_count"] == 11
    assert "market" not in observed
