from __future__ import annotations

import sqlite3
import json
from pathlib import Path

import numpy as np
import pytest

from training.quant_flow_graph_v14.forward_avoidance_lockbox import (
    ADAPTIVE_MODEL,
    CURRENT_MODEL,
    FIXED_MODEL,
    LAG5_MODEL,
    PRIMARY_TARGETS,
    SHUFFLED_MODEL,
    TEST_DATES,
    open_union_source,
    prefix_identity_audit,
    split_indices,
    summarize_gate,
    audit_test_window_relation_coverage,
)


def _source_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE daily_observations(source TEXT,symbol TEXT,trade_date TEXT);
        CREATE TABLE etf_flow_observations(
          ticker TEXT,effective_date TEXT,available_at_date TEXT,
          processed_date TEXT,fund_flow REAL,nav REAL,shares_outstanding REAL
        );
        """
    )
    return connection


def test_union_source_uses_non_overlapping_frozen_boundaries(tmp_path: Path) -> None:
    base_path = tmp_path / "base.sqlite3"
    incremental_path = tmp_path / "incremental.sqlite3"
    repair_path = tmp_path / "repair.sqlite3"
    with _source_database(base_path) as item:
        item.executemany(
            "INSERT INTO daily_observations VALUES('fmp','SPY',?)",
            [("2026-07-14",), ("2026-07-15",)],
        )
        item.executemany(
            "INSERT INTO etf_flow_observations VALUES(?,?,?,?,?,?,?)",
            [
                ("SPY", "2026-07-14", "2026-07-16", "2026-07-15", 1, 2, 3),
                ("DROP", "2026-07-15", "2026-07-17", "2026-07-16", 1, 2, 3),
            ],
        )
    with _source_database(incremental_path) as item:
        item.executemany(
            "INSERT INTO daily_observations VALUES('fmp','SPY',?)",
            [("2026-07-14",), ("2026-07-15",)],
        )
        item.executemany(
            "INSERT INTO etf_flow_observations VALUES(?,?,?,?,?,?,?)",
            [
                ("DROP", "2026-07-16", "2026-07-20", "2026-07-17", 1, 2, 3),
                ("QQQ", "2026-07-17", "2026-07-21", "2026-07-20", 1, 2, 3),
            ],
        )
    with sqlite3.connect(repair_path) as item:
        item.execute(
            "CREATE TABLE flow(ticker TEXT,effective_date TEXT,processed_date TEXT,"
            "available_session TEXT,flow_rate_pct REAL,fund_flow REAL,nav REAL,"
            "shares_outstanding REAL)"
        )
        item.executemany(
            "INSERT INTO flow VALUES(?,?,?,?,?,?,?,?)",
            [
                ("SPY", "2026-07-15", "2026-07-16", "2026-07-17", 1, 1, 2, 3),
                ("IWM", "2026-07-16", "2026-07-17", "2026-07-20", 1, 1, 2, 3),
                ("DROP", "2026-07-17", "2026-07-20", "2026-07-21", 1, 1, 2, 3),
            ],
        )
    source = open_union_source(
        base_database=base_path,
        incremental_database=incremental_path,
        repaired_flow_cache=repair_path,
    )
    try:
        assert [row[0] for row in source.execute(
            "SELECT trade_date FROM daily_observations ORDER BY trade_date"
        )] == ["2026-07-14", "2026-07-15"]
        assert [tuple(row) for row in source.execute(
            "SELECT ticker,effective_date,available_at_date FROM etf_flow_observations "
            "ORDER BY effective_date"
        )] == [
            ("SPY", "2026-07-14", "2026-07-16"),
            ("SPY", "2026-07-15", "2026-07-17"),
            ("IWM", "2026-07-16", "2026-07-20"),
            ("QQQ", "2026-07-17", "2026-07-21"),
        ]
    finally:
        source.close()


def _event_database(path: Path, changed: bool = False) -> None:
    with sqlite3.connect(path) as item:
        item.executescript(
            """
            CREATE TABLE session_map(signal_date TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE daily_flow_state(signal_date TEXT PRIMARY KEY,value REAL);
            CREATE TABLE etf_flow_events(
              signal_date TEXT,ticker TEXT,value REAL,PRIMARY KEY(signal_date,ticker)
            );
            """
        )
        value = 2.0 if changed else 1.0
        item.execute("INSERT INTO session_map VALUES('2026-07-14','same')")
        item.execute("INSERT INTO daily_flow_state VALUES('2026-07-14',?)", (value,))
        item.execute("INSERT INTO etf_flow_events VALUES('2026-07-14','SPY',1.0)")
        item.execute("INSERT INTO session_map VALUES('2026-07-15','future')")


def test_prefix_identity_is_value_exact_and_ignores_new_dates(tmp_path: Path) -> None:
    old = tmp_path / "old.sqlite3"
    new = tmp_path / "new.sqlite3"
    _event_database(old)
    _event_database(new)
    assert prefix_identity_audit(old_event_cube=old, new_event_cube=new)["passed"]
    with sqlite3.connect(new) as item:
        item.execute("UPDATE daily_flow_state SET value=2 WHERE signal_date='2026-07-14'")
    with pytest.raises(ValueError, match="prefix mismatch"):
        prefix_identity_audit(old_event_cube=old, new_event_cube=new)


def test_split_applies_twenty_session_purge() -> None:
    historical = tuple(f"2026-05-{day:02d}" for day in range(1, 32))
    dates = historical + TEST_DATES
    matrix = {
        "date_values": dates,
        "date_codes": np.arange(len(dates), dtype=np.int32),
    }
    train, test, audit = split_indices(matrix)
    assert len(test) == len(TEST_DATES)
    assert audit["test_date_count"] == 11
    assert int(np.max(train)) == len(historical) - 21


def test_relation_coverage_is_scoped_to_preregistered_test_dates(tmp_path: Path) -> None:
    snapshots = [
        {
            "signal_date": "2020-01-02",
            "relation_stock_coverage_ratio": 0.0,
            "stock_count": 1,
        }
    ] + [
        {
            "signal_date": date,
            "relation_stock_coverage_ratio": 1.0,
            "stock_count": 479,
        }
        for date in TEST_DATES
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"snapshots": snapshots}), encoding="utf-8"
    )
    audit = audit_test_window_relation_coverage(tmp_path)
    assert audit["passed"]
    assert len(audit["dates"]) == 11


def _model_metrics(mae: float, rank: float, basket: float) -> dict[str, float]:
    return {
        "mae": mae,
        "mean_daily_rank_ic": rank,
        "economic_basket_value": basket,
    }


def test_primary_gate_requires_every_preregistered_check() -> None:
    targets = {}
    for target in PRIMARY_TARGETS:
        targets[target] = {
            "models": {
                ADAPTIVE_MODEL: _model_metrics(0.8, 0.3, 0.4),
                CURRENT_MODEL: _model_metrics(1.0, 0.2, 0.3),
                SHUFFLED_MODEL: _model_metrics(1.1, 0.1, 0.2),
                FIXED_MODEL: _model_metrics(0.9, 0.0, 0.0),
                LAG5_MODEL: _model_metrics(0.95, 0.0, 0.0),
            },
            "positive_daily_mae_improvement_count": 6,
        }
    result = summarize_gate(targets=targets, data_checks={"valid": True})
    assert result["status"] == "V14_FORWARD_AVOIDANCE_PRELIMINARY_PASS"
    targets[PRIMARY_TARGETS[0]]["positive_daily_mae_improvement_count"] = 5
    result = summarize_gate(targets=targets, data_checks={"valid": True})
    assert result["status"] == "V14_FORWARD_AVOIDANCE_FAIL"
