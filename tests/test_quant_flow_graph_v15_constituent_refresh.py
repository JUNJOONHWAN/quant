import json
import sqlite3
from argparse import Namespace
from pathlib import Path

import numpy as np

from training.quant_flow_graph_v15.constituent_refresh import (
    FMP_BULK_END_MESSAGE,
    _array_comparison,
    age_buckets,
    connected_snapshot,
    connected_universe_audit,
    compare_graph_roots,
    current_bulk_topology_audit,
    normalize_bulk_row,
    parse_bulk_payload,
    read_candidate_universe,
    run_combine,
)
from training.quant_flow_graph_v15.topology_sensitivity import build_topology_only_graph


def test_connected_snapshot_maps_local_to_global_etf_ids(tmp_path: Path):
    graph = tmp_path / "graph"
    (graph / "snapshots").mkdir(parents=True)
    path = graph / "snapshots" / "2026-07-15.npz"
    np.savez_compressed(
        path,
        edge_index=np.asarray([[0, 1, 1], [0, 1, 1]], dtype=np.int64),
        edge_attr=np.asarray([[0.1, 0.5, 1.0], [0.2, 1.0, 0.0], [0.3, 1.0, 0.0]], dtype=np.float32),
        etf_ids=np.asarray([2, 0], dtype=np.int64),
    )
    manifest = {"etf_vocabulary": ["AAA", "BBB", "CCC"]}
    row = {
        "signal_date": "2026-07-15",
        "price_date": "2026-07-14",
        "flow_date": "2026-07-13",
        "stock_count": 2,
        "path": str(path),
    }
    result = connected_snapshot(graph, manifest, row)
    assert result["connected"]["CCC"]["age_sessions"] == 126
    assert result["connected"]["AAA"]["age_sessions"] == 252
    assert result["connected"]["AAA"]["edge_count"] == 2
    assert result["connected"]["CCC"]["observed_exact_t2_in_graph"] is True


def test_age_buckets_are_exhaustive():
    values = [0, 63, 64, 126, 127, 252, 253, 504, 505, 1260]
    assert age_buckets(values) == {
        "age_0_63": 2,
        "age_64_126": 2,
        "age_127_252": 2,
        "age_253_504": 2,
        "age_over_504": 2,
    }


def test_candidate_universe_requires_ever_strict_eligibility():
    dates = ("2026-07-15", "2026-07-16")
    graph = {
        "manifest_path": "/manifest.json",
        "manifest_sha256": "abc",
        "snapshots": {
            date: {
                "price_date": date,
                "flow_date": date,
                "stock_count": 1,
                "edge_count": 2,
                "connected": {
                    "AAA": {"age_sessions": 10, "observed_exact_t2_in_graph": True},
                    "BBB": {"age_sessions": 600, "observed_exact_t2_in_graph": False},
                },
            }
            for date in dates
        },
    }
    states = {
        "2026-07-15": {
            "AAA": {"strict_eligible": 1, "clean_eligible": 1, "special_eligible": 0},
            "BBB": {"strict_eligible": 0, "clean_eligible": 0, "special_eligible": 0},
        },
        "2026-07-16": {
            "AAA": {"strict_eligible": 0, "clean_eligible": 0, "special_eligible": 0},
            "BBB": {"strict_eligible": 0, "clean_eligible": 0, "special_eligible": 0},
        },
    }
    base = [{"etf_ticker": "AAA", "effective_date": "2026-03-31", "available_date": "2026-05-01", "row_count": 10}]
    report, candidates = connected_universe_audit(graph, states, base, dates)
    assert report["counts"]["connected_etf_union"] == 2
    assert report["counts"]["ever_strict_connected_candidate_etfs"] == 1
    assert [row["symbol"] for row in candidates] == ["AAA"]


def test_read_candidate_universe_deduplicates_and_filters(tmp_path: Path):
    path = tmp_path / "universe.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"symbol": "spy", "is_etf": True}),
                json.dumps({"symbol": "SPY", "is_etf": True}),
                json.dumps({"symbol": "AAPL", "is_etf": False}),
            ]
        ),
        encoding="utf-8",
    )
    assert read_candidate_universe(path) == ["SPY"]


def test_parse_bulk_payload_accepts_fmp_csv():
    payload = b'"symbol","asset","sharesNumber","weightPercentage"\n"SPY","AAPL","10","7.1"\n'
    rows, payload_format, fields = parse_bulk_payload(payload)
    assert payload_format == "csv"
    assert fields == ["symbol", "asset", "sharesNumber", "weightPercentage"]
    assert rows == [
        {"symbol": "SPY", "asset": "AAPL", "sharesNumber": "10", "weightPercentage": "7.1"}
    ]


def test_normalize_bulk_row_maps_etf_and_asset():
    row = normalize_bulk_row(
        {
            "symbol": "spy",
            "asset": "aapl",
            "name": "Apple Inc.",
            "sharesNumber": "10",
            "weightPercentage": "7.1",
            "marketValue": "1000",
        }
    )
    assert row["etf_ticker"] == "SPY"
    assert row["constituent_ticker"] == "AAPL"
    assert row["shares"] == 10.0
    assert row["weight_percent"] == 7.1


def test_bulk_end_message_is_exact_and_not_csv():
    assert FMP_BULK_END_MESSAGE == b"Query Error: Invalid or missing query parameter - part"


def test_combine_preserves_source_and_adds_overlay_rows(tmp_path: Path):
    source = tmp_path / "source.sqlite3"
    overlay = tmp_path / "overlay.sqlite3"
    schema = """
      CREATE TABLE daily_observations(value INTEGER);
      CREATE TABLE etf_constituent_observations(
        provider TEXT NOT NULL, etf_ticker TEXT NOT NULL,
        constituent_key TEXT NOT NULL, effective_date TEXT NOT NULL,
        value TEXT,
        PRIMARY KEY(provider,etf_ticker,constituent_key,effective_date)
      );
    """
    with sqlite3.connect(source) as connection:
        connection.executescript(schema)
        connection.execute("INSERT INTO daily_observations VALUES (1)")
        connection.execute(
            "INSERT INTO etf_constituent_observations VALUES ('fmp','SPY','ticker:AAPL','2026-03-31','old')"
        )
        connection.execute(
            "INSERT INTO etf_constituent_observations VALUES ('fmp','QQQ','ticker:OLD','2026-06-30','stale')"
        )
    with sqlite3.connect(overlay) as connection:
        connection.executescript(schema)
        connection.execute(
            "INSERT INTO etf_constituent_observations VALUES ('fmp','QQQ','ticker:MSFT','2026-06-30','new')"
        )
    source_before = source.read_bytes()
    output = tmp_path / "combined.sqlite3"
    output_root = tmp_path / "output"
    assert (
        run_combine(
            Namespace(
                oracle_incremental=source,
                overlay_database=overlay,
                output_database=output,
                output_root=output_root,
                replace=False,
            )
        )
        == 0
    )
    assert source.read_bytes() == source_before
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT COUNT(*) FROM daily_observations").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM etf_constituent_observations").fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM etf_constituent_observations WHERE constituent_key='ticker:OLD'"
        ).fetchone()[0] == 0


def test_compare_graph_roots_maps_edges_and_separates_non_topology(tmp_path: Path):
    original = tmp_path / "original"
    repaired = tmp_path / "repaired"
    date = "2026-07-15"
    for root in (original, repaired):
        (root / "snapshots").mkdir(parents=True)
        np.save(
            root / "flow_values.npy",
            np.ones((9, 2, 4), dtype=np.float32),
        )
        np.save(
            root / "flow_available_session_index.npy",
            np.zeros((9, 2), dtype=np.int32),
        )
    common = {
        "stock_symbols": np.asarray(["AAPL", "MSFT"]),
        "stock_x": np.ones((2, 2), dtype=np.float32),
        "targets": np.ones((2, 1), dtype=np.float32),
        "target_mask": np.ones((2, 1), dtype=np.uint8),
        "etf_ids": np.asarray([1, 0], dtype=np.int64),
        "signal_position": np.asarray(10, dtype=np.int32),
        "flow_position": np.asarray(8, dtype=np.int32),
    }
    np.savez_compressed(
        original / "snapshots" / f"{date}.npz",
        **common,
        edge_index=np.asarray([[0, 1], [0, 1]], dtype=np.int64),
        edge_attr=np.asarray([[0.1, 0.5, 0.0], [0.2, 1.0, 0.0]], dtype=np.float32),
    )
    np.savez_compressed(
        repaired / "snapshots" / f"{date}.npz",
        **common,
        edge_index=np.asarray([[0, 0], [0, 1]], dtype=np.int64),
        edge_attr=np.asarray([[0.3, 0.1, 1.0], [0.2, 1.0, 0.0]], dtype=np.float32),
    )
    manifest = {
        "etf_vocabulary": ["QQQ", "SPY"],
        "feature_contract": {"flow_lookback_sessions": 60},
        "flow_cube": {"session_start_position": 0},
        "quality_gate": {"ok": True},
        "snapshots": [
            {
                "signal_date": date,
                "price_date": "2026-07-14",
                "flow_date": "2026-07-13",
            }
        ],
    }
    for root in (original, repaired):
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = compare_graph_roots(original, repaired, (date,))
    row = report["per_date"][date]
    assert row["all_non_topology_arrays_equal"] is True
    assert row["model_flow_history"]["equal"] is True
    assert row["edges"]["added"] == 1
    assert row["edges"]["removed"] == 1
    assert row["edges"]["common_with_changed_attributes"] == 1
    assert report["aggregate"]["changed_etf_union_count"] == 2
    assert report["aggregate"]["changed_stock_union_count"] == 2


def test_array_comparison_treats_matching_nan_positions_as_equal():
    original = np.asarray([1.0, np.nan, -2.0], dtype=np.float32)
    repaired = np.asarray([1.0, np.nan, -2.0], dtype=np.float32)
    result = _array_comparison(original, repaired)
    assert result["equal"] is True
    assert result["nan_mask_equal"] is True
    assert result["finite_value_changed_count"] == 0


def test_topology_only_graph_keeps_original_flow_and_training_rows(tmp_path: Path):
    original = tmp_path / "original"
    repaired = tmp_path / "repaired"
    output = tmp_path / "hybrid"
    for root in (original, repaired):
        (root / "snapshots").mkdir(parents=True)
        np.save(root / "flow_values.npy", np.ones((60, 1, 4), dtype=np.float32))
        np.save(
            root / "flow_available_session_index.npy",
            np.zeros((60, 1), dtype=np.int32),
        )
    rows = []
    for index, date in enumerate(
        (
            "2026-07-15",
            "2026-07-16",
            "2026-07-17",
            "2026-07-20",
            "2026-07-21",
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
        )
    ):
        common = {
            "stock_symbols": np.asarray(["AAPL"]),
            "stock_x": np.ones((1, 1), dtype=np.float32),
            "targets": np.ones((1, 1), dtype=np.float32),
            "target_mask": np.ones((1, 1), dtype=np.uint8),
            "etf_ids": np.asarray([0], dtype=np.int64),
            "signal_position": np.asarray(59, dtype=np.int32),
            "flow_position": np.asarray(59, dtype=np.int32),
        }
        old_path = original / "snapshots" / f"{date}.npz"
        new_path = repaired / "snapshots" / f"{date}.npz"
        np.savez_compressed(
            old_path,
            **common,
            edge_index=np.asarray([[0], [0]], dtype=np.int64),
            edge_attr=np.asarray([[0.1, 1.0, 0.0]], dtype=np.float32),
        )
        np.savez_compressed(
            new_path,
            **common,
            edge_index=np.asarray([[0], [0]], dtype=np.int64),
            edge_attr=np.asarray([[0.2, 0.1, 0.0]], dtype=np.float32),
        )
        rows.append(
            {
                "signal_date": date,
                "price_date": date,
                "flow_date": date,
                "stock_count": 1,
                "edge_count": 1,
                "path": str(old_path),
            }
        )
    original_manifest = {
        "etf_vocabulary": ["SPY"],
        "feature_contract": {"flow_lookback_sessions": 60},
        "flow_cube": {"session_start_position": 0},
        "quality_gate": "PASS",
        "edge_count": len(rows),
        "snapshots": rows,
    }
    repaired_manifest = dict(original_manifest)
    repaired_manifest["snapshots"] = [
        {**row, "path": str(repaired / "snapshots" / f"{row['signal_date']}.npz")}
        for row in rows
    ]
    (original / "manifest.json").write_text(json.dumps(original_manifest), encoding="utf-8")
    (repaired / "manifest.json").write_text(json.dumps(repaired_manifest), encoding="utf-8")
    receipt = build_topology_only_graph(
        original_root=original,
        repaired_root=repaired,
        output_root=output,
        receipt_path=tmp_path / "receipt.json",
        audit_path=tmp_path / "audit.json",
    )
    assert receipt["ok"] is True
    assert receipt["contracts"]["test_model_flow_histories_equal"] is True
    assert (output / "flow_values.npy").resolve() == (original / "flow_values.npy").resolve()


def test_current_bulk_audit_uses_candidate_and_stock_intersection(tmp_path: Path):
    graph = tmp_path / "graph"
    (graph / "snapshots").mkdir(parents=True)
    date = "2026-07-29"
    snapshot = graph / "snapshots" / f"{date}.npz"
    np.savez_compressed(
        snapshot,
        stock_symbols=np.asarray(["AAPL", "MSFT"]),
        stock_x=np.ones((2, 1), dtype=np.float32),
        targets=np.ones((2, 1), dtype=np.float32),
        target_mask=np.ones((2, 1), dtype=np.uint8),
        etf_ids=np.asarray([0], dtype=np.int64),
        edge_index=np.asarray([[0], [0]], dtype=np.int64),
        edge_attr=np.asarray([[0.1, 1.0, 0.0]], dtype=np.float32),
        signal_position=np.asarray(1, dtype=np.int32),
        flow_position=np.asarray(0, dtype=np.int32),
    )
    manifest = {
        "etf_vocabulary": ["SPY"],
        "snapshots": [
            {
                "signal_date": date,
                "price_date": "2026-07-28",
                "flow_date": "2026-07-27",
                "stock_count": 2,
                "path": str(snapshot),
            }
        ],
    }
    (graph / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    universe = tmp_path / "universe.jsonl"
    universe.write_text(
        json.dumps(
            {
                "symbol": "SPY",
                "is_etf": True,
                "strict_eligible_on_last_test_date": True,
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "bulk.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE fmp_etf_holder_bulk_parts(as_of_date TEXT,part INTEGER);
            CREATE TABLE fmp_etf_holder_bulk_rows(
              as_of_date TEXT,etf_ticker TEXT,constituent_ticker TEXT,
              weight_percent REAL,raw_json TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO fmp_etf_holder_bulk_parts VALUES ('2026-08-28',1)"
        )
        connection.execute(
            "INSERT INTO fmp_etf_holder_bulk_rows VALUES (?,?,?,?,?)",
            ("2026-08-28", "SPY", "MSFT", 5.0, json.dumps({"lastUpdated": "2026-08-27"})),
        )
    report = current_bulk_topology_audit(
        bulk_database=database,
        universe_path=universe,
        graph_root=graph,
        signal_date=date,
        as_of_date="2026-08-28",
    )
    comparison = report["strict_on_last_test_date"]
    assert comparison["historical_pair_count"] == 1
    assert comparison["current_bulk_pair_count"] == 1
    assert comparison["added_pair_count"] == 1
    assert comparison["removed_pair_count"] == 1
