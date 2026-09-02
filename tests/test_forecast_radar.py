from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_dataset.shared_market import SharedMarketBinding
from workflows.forecast_radar import cli
from workflows.forecast_radar.contracts import (
    COVERAGE_VALIDATED_CORE,
    RUN_SCHEMA_VERSION,
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
    _build_coverage_gate,
    current_completed_run,
    query_latest,
)
from workflows.forecast_radar.source_quality import validate_forecast_source_quality
from training.quant_forecast_v2.panel import _is_core_rank_reference, _rank_against_core
from training.quant_forecast_v2.source import SourceBundle


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


def _quality_binding(tmp_path: Path) -> SharedMarketBinding:
    target = "2026-08-26"
    previous = "2026-08-25"
    incremental = tmp_path / "incremental.sqlite3"
    status_path = tmp_path / "oracle_status.json"
    connection = sqlite3.connect(incremental)
    connection.executescript(
        """
        CREATE TABLE daily_observations(
          source TEXT,symbol TEXT,trade_date TEXT,open REAL,high REAL,low REAL,
          close REAL,adjusted_close REAL,volume REAL,raw_artifact_id INTEGER,
          capture_event_id INTEGER,source_row_index INTEGER,source_timestamp_ms INTEGER,
          PRIMARY KEY(source,symbol,trade_date)
        );
        CREATE TABLE quality_checks(
          symbol TEXT,trade_date TEXT,status TEXT,sources_json TEXT,metrics_json TEXT,
          reasons_json TEXT,tolerances_json TEXT,computed_at_utc TEXT,
          PRIMARY KEY(symbol,trade_date)
        );
        """
    )
    for offset, symbol in enumerate(("SPY", "QQQ", "IWM", "DIA"), start=1):
        for trade_date, close in ((previous, 100.0 + offset), (target, 101.0 + offset)):
            connection.execute(
                "INSERT INTO daily_observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "fmp", symbol, trade_date, close, close + 1.0, close - 1.0,
                    close, close, 1_000_000.0, 1, 1, offset, 0,
                ),
            )
        connection.execute(
            "INSERT INTO quality_checks VALUES(?,?,?,?,?,?,?,?)",
            (symbol, target, "single_source", "{}", "{}", "[]", "{}", "now"),
        )
    connection.commit()
    connection.close()
    status_path.write_text(
        json.dumps(
            {
                "target_as_of_date": target,
                "market_row_gate": {"rows_by_session": {target: 4}},
                "symbol_coverage_gate": {
                    "sessions": {
                        target: {
                            "status": "complete",
                            "error_count": 0,
                            "bar_count": 4,
                            "missing_after": [],
                            "invalid_before_count": 0,
                            "invalid_no_bar_count": 0,
                            "quarantined_invalid_bar_count": 0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return SharedMarketBinding(
        base_database=tmp_path / "base.sqlite3",
        incremental_database=incremental,
        oracle_status_path=status_path,
        base_history_end="2026-08-24",
        target_as_of_date=target,
        latest_flow_effective_date="2026-08-24",
        latest_constituent_effective_date="2026-08-24",
        latest_constituent_available_date="2026-08-24",
        constituent_available_lag_days=2,
        corporate_action_visible_record_count=0,
        corporate_action_projection_sha256="a" * 64,
        source_fingerprint_sha256="b" * 64,
        source_fingerprint={},
    )


def test_source_quality_blocks_invalid_target_bars_before_forecast_generation(
    tmp_path: Path,
) -> None:
    binding = _quality_binding(tmp_path)
    passed = validate_forecast_source_quality(binding)
    assert passed["status"] == "PASS"
    assert len(passed["data_fingerprint_sha256"]) == 64
    with sqlite3.connect(binding.incremental_database) as connection:
        connection.execute(
            "UPDATE daily_observations SET high=1.0 WHERE source='fmp' "
            "AND symbol='SPY' AND trade_date=?",
            (binding.target_as_of_date,),
        )
        connection.commit()
    with pytest.raises(RuntimeError, match="invalid_ohlcv_rows=1"):
        validate_forecast_source_quality(binding)


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


def test_coverage_gate_does_not_conflate_source_shadow_with_model_tier() -> None:
    additional_symbols = [f"SHADOW_{index}" for index in range(1_000)]
    additional_core_symbols = [f"CORE_{index}" for index in range(398)]
    forecasts = [
        {
            "symbol": "A",
            "coverage_tier": COVERAGE_VALIDATED_CORE,
            "validation_status": "HISTORICAL_OOS_CORE",
        },
        {
            "symbol": "CORE_ONLY",
            "coverage_tier": COVERAGE_VALIDATED_CORE,
            "validation_status": "HISTORICAL_OOS_CORE",
        },
        {
            "symbol": "JOBY",
            "coverage_tier": "GENERAL_UNIVERSE_SHADOW",
            "validation_status": "EXTRAPOLATED_UNVALIDATED",
        },
        {
            "symbol": "SHORT",
            "coverage_tier": "GENERAL_UNIVERSE_SHADOW",
            "validation_status": "EXTRAPOLATED_UNVALIDATED_SHORT_HISTORY",
        },
    ]
    forecasts.extend(
        {
            "symbol": symbol,
            "coverage_tier": "GENERAL_UNIVERSE_SHADOW",
            "validation_status": "EXTRAPOLATED_UNVALIDATED",
        }
        for symbol in additional_symbols
    )
    forecasts.extend(
        {
            "symbol": symbol,
            "coverage_tier": COVERAGE_VALIDATED_CORE,
            "validation_status": "HISTORICAL_OOS_CORE",
        }
        for symbol in additional_core_symbols
    )
    general_universe_symbols = ["A", "JOBY", "SHORT", *additional_symbols]
    gate = _build_coverage_gate(
        general_universe_symbols=general_universe_symbols,
        live_panel_symbols=[
            *general_universe_symbols,
            "CORE_ONLY",
            *additional_core_symbols,
        ],
        forecasts=forecasts,
        panel_live_general_shadow_source_symbol_count=len(general_universe_symbols),
    )
    assert gate["panel_live_general_shadow_source_symbol_count"] == 1_003
    assert gate["forecast_general_shadow_symbol_count"] == 1_002
    assert gate["live_panel_forecast_parity_gate"] is True
    assert gate["general_candidate_full_coverage_gate"] is True
    assert gate["status"] == "PASS"


def test_shadow_ranks_do_not_change_validated_core_ranks() -> None:
    values = np.asarray([10.0, 20.0, 15.0, 30.0])
    series = pd.Series(values, index=["CORE_A", "CORE_B", "S", "H"])
    core_mask = pd.Series(
        [True, True, False, False], index=series.index
    )
    observed = _rank_against_core(series, core_mask)
    assert observed["CORE_A"] == 0.5
    assert observed["CORE_B"] == 1.0
    assert observed["S"] == 0.5
    assert observed["H"] == 1.0


def test_short_history_index_member_is_not_a_core_rank_reference() -> None:
    assert _is_core_rank_reference(1, 0, 0.25) is True
    assert _is_core_rank_reference(0, 1, -0.10) is True
    assert _is_core_rank_reference(1, 0, None) is False
    assert _is_core_rank_reference(0, 0, 0.25) is False


def test_live_general_universe_keeps_us_listed_foreign_stock_and_excludes_funds(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE daily_observations(
          source TEXT,symbol TEXT,trade_date TEXT,open REAL,high REAL,low REAL,
          close REAL,adjusted_close REAL,volume REAL,vwap REAL
        );
        CREATE TABLE fmp_training_facts(
          id INTEGER PRIMARY KEY,endpoint_id TEXT,symbol TEXT,row_json TEXT
        );
        """
    )
    profiles = {
        "JOBY": {"exchangeShortName": "NYSE", "isActivelyTrading": True},
        "TSM": {"exchangeShortName": "NYSE", "country": "TW", "isActivelyTrading": True},
        "SPY": {
            "exchangeShortName": "AMEX",
            "isActivelyTrading": True,
            "isEtf": True,
        },
        "OLD": {"exchangeShortName": "NASDAQ", "isActivelyTrading": False},
    }
    for number, (symbol, profile) in enumerate(profiles.items(), 1):
        connection.execute(
            "INSERT INTO daily_observations VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("fmp", symbol, "2026-08-26", 1, 1, 1, 1, 1, 100, 1),
        )
        connection.execute(
            "INSERT INTO fmp_training_facts VALUES(?,?,?,?)",
            (
                number,
                "company_information_company_profile_data",
                symbol,
                json.dumps(profile),
            ),
        )
    connection.commit()
    connection.close()
    with SourceBundle(database, None) as source:
        symbols, audit = source.active_us_exchange_stock_universe("2026-08-26")
    assert symbols == ["JOBY", "TSM"]
    assert audit["point_in_time_status"] == "NOT_HISTORICAL_PIT_DO_NOT_USE_FOR_OOS"
    assert audit["exclusions"]["etf"] == 1
    assert audit["exclusions"]["inactive"] == 1


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
        json.dumps(
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "quality_gate": "PASS_SHADOW_RUN",
                "signal_date": "2026-08-27",
                "price_date": "2026-08-26",
                "flow_date": "2026-08-25",
                "stock_count": 479,
                "validated_core_count": 477,
                "general_shadow_count": 2,
                "source_status": {
                    "shared_oracle_store": {
                        "target_as_of_date": "2026-08-26",
                        "source_fingerprint_sha256": "a" * 64,
                    },
                    "forecast_source_data_quality": {
                        "status": "PASS",
                        "data_fingerprint_sha256": "c" * 64,
                    },
                },
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
        "oracle_target_as_of_date": "2026-08-26",
        "oracle_source_fingerprint_sha256": "a" * 64,
        "oracle_data_quality_status": "PASS",
        "oracle_data_quality_sha256": "c" * 64,
    }
    (tmp_path / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    observed = current_completed_run(
        tmp_path,
        "2026-08-27",
        oracle_target_as_of_date="2026-08-26",
        oracle_source_fingerprint_sha256="a" * 64,
        oracle_data_quality_sha256="c" * 64,
    )
    assert observed is not None
    assert observed["quality_gate"] == "NOOP_ALREADY_CURRENT"
    assert (
        current_completed_run(
            tmp_path,
            "2026-08-27",
            oracle_target_as_of_date="2026-08-26",
            oracle_source_fingerprint_sha256="b" * 64,
        )
        is None
    )
    assert (
        current_completed_run(
            tmp_path,
            "2026-08-27",
            oracle_target_as_of_date="2026-08-26",
            oracle_source_fingerprint_sha256="a" * 64,
            oracle_data_quality_sha256="d" * 64,
        )
        is None
    )
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
    observed = query_latest(
        live_root=tmp_path,
        symbol="NVDA",
        verify_current_oracle=False,
    )
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

    def fake_evaluate_outcomes(**kwargs):
        assert kwargs == {"live_root": tmp_path}
        return {
            "status": "RECORDED",
            "cohort_count": 477,
            "latest_signal_date": "2026-08-27",
        }

    monkeypatch.setattr(cli, "run_daily", fake_run_daily)
    monkeypatch.setattr(cli, "query_latest", fake_query_latest)
    monkeypatch.setattr(cli, "evaluate_outcomes", fake_evaluate_outcomes)
    assert cli.main(["scheduled-daily", "--live-root", str(tmp_path)]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert calls == [
        {"signal_date": None, "if_needed": True, "live_root": tmp_path}
    ]
    assert observed["batch"]["quality_gate"] == "NOOP_ALREADY_CURRENT"
    assert observed["latest"]["database_sha256"] == "db-hash"
    assert observed["probability_resolution"]["p_up_5d_distinct_count"] == 11
    assert observed["evaluation_477"]["cohort_count"] == 477
    assert "market" not in observed


def test_scheduled_daily_keeps_forecast_visible_when_evaluation_fails(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "run_daily",
        lambda **kwargs: {"quality_gate": "NOOP_ALREADY_CURRENT", "run_id": "run-1"},
    )
    monkeypatch.setattr(
        cli,
        "query_latest",
        lambda **kwargs: {
            "latest": {"run_id": "run-1", "quality_gate": "PASS_SHADOW_RUN"},
            "probability_resolution": {"p_up_5d_distinct_count": 11},
        },
    )

    def fail_evaluation(**kwargs):
        raise RuntimeError("ledger locked")

    monkeypatch.setattr(cli, "evaluate_outcomes", fail_evaluation)
    assert cli.main(["scheduled-daily", "--live-root", str(tmp_path)]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["latest"]["quality_gate"] == "PASS_SHADOW_RUN"
    assert observed["evaluation_477"] == {
        "status": "ERROR_RECORDING_FAILED",
        "error_type": "RuntimeError",
        "error": "ledger locked",
    }
