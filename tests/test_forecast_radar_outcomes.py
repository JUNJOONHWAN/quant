from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from workflows.forecast_radar.io import sha256_file
from workflows.forecast_radar.outcomes import evaluate_outcomes


def _cohort_hash(symbols: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{symbol}\n" for symbol in sorted(symbols)).encode("utf-8")
    ).hexdigest()


def _cohort_database(path: Path, symbols: list[str]) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE stock_forecasts(
          symbol TEXT PRIMARY KEY,sector TEXT,industry TEXT,
          coverage_tier TEXT,validation_status TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO stock_forecasts VALUES(?,?,?,?,?)",
        [
            (symbol, "Technology", "Test", "VALIDATED_CORE", "HISTORICAL_OOS_CORE")
            for symbol in symbols
        ],
    )
    connection.commit()
    connection.close()


def _price_database(path: Path, sessions: list[str], through: int) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE daily_observations(
          source TEXT,symbol TEXT,trade_date TEXT,open REAL,high REAL,low REAL,
          close REAL,adjusted_close REAL,volume REAL,vwap REAL,
          PRIMARY KEY(source,symbol,trade_date)
        )
        """
    )
    _append_prices(connection, sessions, 0, through)
    connection.commit()
    connection.close()


def _append_prices(
    connection: sqlite3.Connection, sessions: list[str], start: int, stop: int
) -> None:
    for index in range(start, stop):
        closes = {
            "SPY": 100.0 + 0.2 * index,
            "QQQ": 100.0 + 0.3 * index,
            "AAA": 100.0 + index,
            "BBB": 100.0 - 0.5 * index,
        }
        for symbol, close in closes.items():
            connection.execute(
                "INSERT INTO daily_observations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    "fmp",
                    symbol,
                    sessions[index],
                    close,
                    close + 1.0,
                    close - 1.0,
                    close,
                    close,
                    1_000_000.0,
                    close,
                ),
            )


def _run_database(path: Path, symbols: list[str], reference: dict[str, float]) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
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
        )
        """
    )
    for symbol in symbols:
        connection.execute(
            "INSERT INTO stock_forecasts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                symbol,
                "Technology",
                "Test",
                reference[symbol],
                "VALIDATED_CORE",
                "HISTORICAL_OOS_CORE",
                0.55,
                0.55,
                2.0,
                4.0,
                2.0,
                1.0,
                1.0,
                1.0,
                4.0,
                8.0,
                4.0,
                2.0,
                2.0,
                2.0,
                2.0,
                4.0,
                1.0,
                2.0,
            ),
        )
    connection.commit()
    connection.close()


def _live_run(live_root: Path, sessions: list[str]) -> None:
    run_id = "run-1"
    run_root = live_root / "runs" / run_id
    panel_root = run_root / "panel"
    panel_root.mkdir(parents=True)
    database = run_root / "forecast_radar.sqlite3"
    _run_database(database, ["AAA", "BBB"], {"AAA": 101.0, "BBB": 99.5})
    panel = sqlite3.connect(panel_root / "panel.sqlite3")
    panel.execute("CREATE TABLE panel(signal_date TEXT,symbol TEXT,benchmark TEXT)")
    panel.executemany(
        "INSERT INTO panel VALUES(?,?,?)",
        [(sessions[2], "AAA", "QQQ"), (sessions[2], "BBB", "SPY")],
    )
    panel.commit()
    panel.close()
    summary = {
        "schema_version": "quant.forecast_radar.daily_run.v2.full_universe_shadow",
        "quality_gate": "PASS_SHADOW_RUN",
        "activation_status": "SHADOW_ONLY",
        "run_id": run_id,
        "signal_date": sessions[2],
        "price_date": sessions[1],
        "flow_date": sessions[0],
        "generated_at_utc": "2026-08-01T00:00:00+00:00",
        "artifacts": {
            "database": {"path": str(database), "sha256": sha256_file(database)}
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_evaluation_ledger_is_immutable_idempotent_and_resolves_5d_20d(
    tmp_path: Path,
) -> None:
    symbols = ["AAA", "BBB"]
    sessions = [value.strftime("%Y-%m-%d") for value in pd.bdate_range("2026-07-01", periods=24)]
    cohort = tmp_path / "cohort.sqlite3"
    prices = tmp_path / "prices.sqlite3"
    live_root = tmp_path / "live"
    evaluation_root = tmp_path / "evaluation"
    _cohort_database(cohort, symbols)
    _price_database(prices, sessions, through=4)
    _live_run(live_root, sessions)

    first = evaluate_outcomes(
        live_root=live_root,
        evaluation_root=evaluation_root,
        cohort_source_database=cohort,
        cohort_count=2,
        cohort_sha256=_cohort_hash(symbols),
        base_database=prices,
        incremental_database=None,
    )
    assert first["forecast_count"] == 2
    assert first["outcomes_by_horizon"]["5"] == {"resolved": 0, "pending": 2}
    assert first["outcomes_by_horizon"]["20"] == {"resolved": 0, "pending": 2}

    connection = sqlite3.connect(prices)
    _append_prices(connection, sessions, 4, len(sessions))
    connection.commit()
    connection.close()
    second = evaluate_outcomes(
        live_root=live_root,
        evaluation_root=evaluation_root,
        cohort_source_database=cohort,
        cohort_count=2,
        cohort_sha256=_cohort_hash(symbols),
        base_database=prices,
        incremental_database=None,
    )
    assert second["forecast_count"] == 2
    assert second["outcomes_by_horizon"]["5"] == {"resolved": 2, "pending": 0}
    assert second["outcomes_by_horizon"]["20"] == {"resolved": 2, "pending": 0}
    assert second["overall_by_horizon"]["5"]["direction_accuracy"] == 0.5
    assert second["overall_by_horizon"]["20"]["direction_accuracy"] == 0.5

    ledger = sqlite3.connect(second["database"])
    assert ledger.execute("SELECT COUNT(*) FROM forecast_signals").fetchone()[0] == 2
    assert ledger.execute("SELECT COUNT(*) FROM horizon_outcomes").fetchone()[0] == 4
    assert ledger.execute("SELECT COUNT(*) FROM accuracy_by_symbol").fetchone()[0] == 4
    ledger.close()
    receipt = json.loads((evaluation_root / "latest.json").read_text(encoding="utf-8"))
    assert receipt["scope"]["cohort_count"] == 2
    assert receipt["contracts"]["trade_policy"] == "NONE_INFORMATION_EVALUATION_ONLY"


def test_evaluation_rejects_cohort_hash_drift(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.sqlite3"
    _cohort_database(cohort, ["AAA", "BBB"])
    with pytest.raises(RuntimeError, match="cohort verification failed"):
        evaluate_outcomes(
            live_root=tmp_path / "live",
            evaluation_root=tmp_path / "evaluation",
            cohort_source_database=cohort,
            cohort_count=2,
            cohort_sha256="0" * 64,
            base_database=tmp_path / "missing.sqlite3",
            incremental_database=None,
        )
