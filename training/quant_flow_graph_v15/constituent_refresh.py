"""Audit and selectively refresh ETF constituent snapshots used by v14.

This lane is deliberately separate from the sealed v14 forward lockbox.  It
first derives the exact ETF set connected to stocks in the eleven v14 test
snapshots, intersects it with the ETF RADAR strict point-in-time eligibility
mask, and then asks FMP only for disclosure dates for that bounded universe.
Snapshot payloads are captured only when their (ETF, effective_date) key is
absent from the sealed base database.  Every new row is written to a separate
overlay and is post-hoc research evidence, never a clean v14 gate input.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from quant_dataset.config import DEFAULT_SECRETS_PATH, load_credentials
from quant_dataset.pipeline import DatasetPipeline
from quant_dataset.rate_limit import rate_limit_policy
from quant_dataset.storage import canonical_json, sha256_bytes


TEST_DATES = (
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
DEFAULT_GRAPH_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/"
    "full_20180102_20260729_allpanel"
)
DEFAULT_EVENT_CUBE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v14/"
    "forward_avoidance_lockbox_20260715_20260729/"
    "v14_extended_flow_event_cube.sqlite3"
)
DEFAULT_BASE_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/"
    "daily_observations.sqlite3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v15/"
    "fmp_constituent_refresh_20190930_20260714"
)
DEFAULT_ORACLE_INCREMENTAL = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/incremental/"
    "normalized/daily_observations.sqlite3"
)
DEFAULT_COMBINED_INCREMENTAL = (
    DEFAULT_OUTPUT_ROOT / "repaired_incremental" / "normalized" / "daily_observations.sqlite3"
)
DEFAULT_REPAIRED_GRAPH_ROOT = DEFAULT_OUTPUT_ROOT / "repaired_graph_20260715_20260729"
DEFAULT_FROM = "2019-09-30"
DEFAULT_TO = "2026-07-14"
FMP_ETF_HOLDER_BULK_URL = "https://financialmodelingprep.com/stable/etf-holder-bulk"
FMP_BULK_END_MESSAGE = b"Query Error: Invalid or missing query parameter - part"
AGE_BUCKETS = (
    ("age_0_63", 0, 63),
    ("age_64_126", 64, 126),
    ("age_127_252", 127, 252),
    ("age_253_504", 253, 504),
    ("age_over_504", 505, None),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, document: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def age_buckets(values: Sequence[int]) -> dict[str, int]:
    result = Counter()
    for value in values:
        for name, lower, upper in AGE_BUCKETS:
            if value >= lower and (upper is None or value <= upper):
                result[name] += 1
                break
    return {name: int(result[name]) for name, _, _ in AGE_BUCKETS}


def snapshot_path(graph_root: Path, manifest_row: Mapping[str, Any]) -> Path:
    candidate = Path(str(manifest_row.get("path") or ""))
    if candidate.is_file():
        return candidate
    return Path(graph_root) / "snapshots" / f"{manifest_row['signal_date']}.npz"


def connected_snapshot(
    graph_root: Path,
    manifest: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    vocabulary = list(manifest["etf_vocabulary"])
    path = snapshot_path(graph_root, manifest_row)
    with np.load(path, allow_pickle=False) as snapshot:
        edge_index = np.asarray(snapshot["edge_index"], dtype=np.int64)
        edge_attr = np.asarray(snapshot["edge_attr"], dtype=np.float64)
        etf_ids = np.asarray(snapshot["etf_ids"], dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"invalid edge_index shape: {path}")
        if edge_attr.shape != (edge_index.shape[1], 3):
            raise ValueError(f"invalid edge_attr shape: {path}")
        if edge_index.shape[1] and (
            int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= len(etf_ids)
        ):
            raise ValueError(f"local ETF index out of bounds: {path}")
        records: dict[str, dict[str, Any]] = {}
        for local_id in np.unique(edge_index[1]).tolist():
            mask = edge_index[1] == int(local_id)
            global_id = int(etf_ids[int(local_id)])
            if global_id < 0 or global_id >= len(vocabulary):
                raise ValueError(f"global ETF index out of bounds: {path}")
            ticker = str(vocabulary[global_id])
            ages_scaled = edge_attr[mask, 1]
            observed = edge_attr[mask, 2]
            if not np.allclose(ages_scaled, ages_scaled[0], atol=1e-6, rtol=0.0):
                raise ValueError(f"non-constant relation age for {ticker} on {path}")
            if not np.allclose(observed, observed[0], atol=1e-6, rtol=0.0):
                raise ValueError(f"non-constant current observation flag for {ticker} on {path}")
            records[ticker] = {
                "age_sessions": int(round(float(ages_scaled[0]) * 252.0)),
                "edge_count": int(mask.sum()),
                "stock_weight_sum": float(edge_attr[mask, 0].sum()),
                "observed_exact_t2_in_graph": bool(observed[0] > 0.5),
            }
    return {
        "signal_date": str(manifest_row["signal_date"]),
        "price_date": str(manifest_row["price_date"]),
        "flow_date": str(manifest_row["flow_date"]),
        "path": str(path),
        "sha256": sha256_file(path),
        "stock_count": int(manifest_row["stock_count"]),
        "edge_count": int(edge_index.shape[1]),
        "connected": records,
    }


def load_graph_connections(graph_root: Path, dates: Sequence[str]) -> dict[str, Any]:
    manifest_path = Path(graph_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_date = {str(row["signal_date"]): row for row in manifest["snapshots"]}
    missing = [date for date in dates if date not in by_date]
    if missing:
        raise ValueError(f"missing graph snapshots: {missing}")
    snapshots = {
        date: connected_snapshot(graph_root, manifest, by_date[date]) for date in dates
    }
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "etf_vocabulary_count": len(manifest["etf_vocabulary"]),
        "snapshots": snapshots,
    }


EVENT_COLUMNS = (
    "ticker",
    "observed_exact_t2",
    "true_zero",
    "missing_exact_t2",
    "stale_visible_state",
    "lifecycle_state",
    "strict_eligible",
    "clean_eligible",
    "special_eligible",
    "exclusion_reasons",
    "observation_channel",
)


def load_event_states(event_cube: Path, dates: Sequence[str]) -> dict[str, dict[str, dict[str, Any]]]:
    placeholders = ",".join("?" for _ in dates)
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with readonly_connection(event_cube) as connection:
        rows = connection.execute(
            f"SELECT signal_date,{','.join(EVENT_COLUMNS)} FROM etf_flow_events "
            f"WHERE signal_date IN ({placeholders}) ORDER BY signal_date,ticker",
            tuple(dates),
        )
        for row in rows:
            result[str(row["signal_date"])][str(row["ticker"])] = {
                key: row[key] for key in EVENT_COLUMNS if key != "ticker"
            }
    missing = [date for date in dates if date not in result]
    if missing:
        raise ValueError(f"missing event cube dates: {missing}")
    return dict(result)


def load_base_snapshot_rows(base_database: Path, tickers: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with readonly_connection(base_database) as connection:
        for start in range(0, len(tickers), 500):
            chunk = list(tickers[start : start + 500])
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                "SELECT etf_ticker,effective_date,available_date,row_count,"
                "invalid_row_count,captured_at_utc FROM etf_constituent_snapshots "
                f"WHERE provider='fmp' AND etf_ticker IN ({placeholders}) "
                "ORDER BY etf_ticker,effective_date",
                tuple(chunk),
            )
            result.extend(dict(row) for row in rows)
    return result


def connected_universe_audit(
    graph: Mapping[str, Any],
    event_states: Mapping[str, Mapping[str, Mapping[str, Any]]],
    base_snapshot_rows: Sequence[Mapping[str, Any]],
    dates: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_connected: set[str] = set()
    ever_strict: set[str] = set()
    per_ticker_dates: dict[str, list[str]] = defaultdict(list)
    per_date: dict[str, Any] = {}
    for date in dates:
        snapshot = graph["snapshots"][date]
        connected = snapshot["connected"]
        states = event_states[date]
        all_connected.update(connected)
        strict = {ticker for ticker in connected if int(states.get(ticker, {}).get("strict_eligible", 0)) == 1}
        clean = {ticker for ticker in connected if int(states.get(ticker, {}).get("clean_eligible", 0)) == 1}
        special = {ticker for ticker in connected if int(states.get(ticker, {}).get("special_eligible", 0)) == 1}
        ever_strict.update(strict)
        for ticker in strict:
            per_ticker_dates[ticker].append(date)
        ages_all = [int(row["age_sessions"]) for row in connected.values()]
        ages_strict = [int(connected[ticker]["age_sessions"]) for ticker in strict]
        ages_non_strict = [
            int(row["age_sessions"]) for ticker, row in connected.items() if ticker not in strict
        ]
        per_date[date] = {
            "price_date": snapshot["price_date"],
            "flow_date": snapshot["flow_date"],
            "stock_count": snapshot["stock_count"],
            "edge_count": snapshot["edge_count"],
            "connected_etf_count": len(connected),
            "connected_strict_eligible_count": len(strict),
            "connected_clean_eligible_count": len(clean),
            "connected_special_eligible_count": len(special),
            "connected_non_strict_count": len(connected) - len(strict),
            "connected_observed_exact_t2_count": sum(
                int(bool(row["observed_exact_t2_in_graph"])) for row in connected.values()
            ),
            "connected_missing_event_state_count": sum(ticker not in states for ticker in connected),
            "age_sessions_all": quantiles(ages_all),
            "age_sessions_strict": quantiles(ages_strict),
            "age_sessions_non_strict": quantiles(ages_non_strict),
            "age_buckets_all": age_buckets(ages_all),
            "age_buckets_strict": age_buckets(ages_strict),
        }
    rows_by_ticker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in base_snapshot_rows:
        rows_by_ticker[str(row["etf_ticker"])].append(row)
    latest_date = dates[-1]
    latest_connected = graph["snapshots"][latest_date]["connected"]
    candidate_rows = []
    for ticker in sorted(ever_strict):
        history = rows_by_ticker.get(ticker, [])
        last = history[-1] if history else None
        state = event_states[latest_date].get(ticker, {})
        relation = latest_connected.get(ticker)
        candidate_rows.append(
            {
                "symbol": ticker,
                "is_etf": True,
                "connected_test_date_count": len(per_ticker_dates[ticker]),
                "strict_eligible_on_last_test_date": bool(int(state.get("strict_eligible", 0))),
                "clean_eligible_on_last_test_date": bool(int(state.get("clean_eligible", 0))),
                "special_eligible_on_last_test_date": bool(int(state.get("special_eligible", 0))),
                "last_test_relation_age_sessions": (
                    int(relation["age_sessions"]) if relation is not None else None
                ),
                "base_snapshot_count": len(history),
                "base_latest_effective_date": str(last["effective_date"]) if last else None,
                "base_latest_available_date": str(last["available_date"]) if last else None,
                "base_latest_row_count": int(last["row_count"]) if last else None,
            }
        )
    report = {
        "schema_version": "quant.etf_flow_v15.connected_constituent_audit.v1",
        "generated_at_utc": utc_now(),
        "test_dates": list(dates),
        "graph_manifest_path": graph["manifest_path"],
        "graph_manifest_sha256": graph["manifest_sha256"],
        "event_cube_contract": {
            "absolute_common_flow_preserved": True,
            "small_etfs_removed_from_download_universe": True,
            "candidate_rule": "connected_to_test_stock AND strict_eligible_on_at_least_one_test_date",
        },
        "counts": {
            "connected_etf_union": len(all_connected),
            "ever_strict_connected_candidate_etfs": len(ever_strict),
            "never_strict_connected_excluded_etfs": len(all_connected - ever_strict),
            "candidate_with_any_base_snapshot": sum(bool(rows_by_ticker.get(t)) for t in ever_strict),
            "candidate_without_base_snapshot": sum(not rows_by_ticker.get(t) for t in ever_strict),
        },
        "per_date": per_date,
        "base_snapshot_scope": {
            "rows_for_connected_union": len(base_snapshot_rows),
            "tickers_with_rows": len(rows_by_ticker),
        },
        "interpretation_boundary": {
            "audit_is_model_gate": False,
            "historical_refresh_is_true_as_observed": False,
            "permitted_use": "posthoc data-quality and repaired-topology sensitivity only",
            "forbidden_use": "do not overwrite or re-label the sealed v14 forward gate",
        },
    }
    return report, candidate_rows


def run_audit(args: argparse.Namespace) -> int:
    dates = tuple(args.test_dates or TEST_DATES)
    graph = load_graph_connections(args.graph_root, dates)
    event_states = load_event_states(args.event_cube, dates)
    all_connected = sorted(
        {ticker for date in dates for ticker in graph["snapshots"][date]["connected"]}
    )
    base_rows = load_base_snapshot_rows(args.base_database, all_connected)
    report, candidates = connected_universe_audit(graph, event_states, base_rows, dates)
    args.output_root.mkdir(parents=True, exist_ok=True)
    universe_path = args.output_root / "v15_fmp_connected_strict_etf_universe.jsonl"
    report_path = args.output_root / "v15_connected_etf_pre_download_audit.json"
    atomic_jsonl(universe_path, candidates)
    report["candidate_universe"] = {
        "path": str(universe_path),
        "sha256": sha256_file(universe_path),
        "rows": len(candidates),
    }
    atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path), "sha256": sha256_file(report_path), **report["counts"]}, sort_keys=True))
    return 0


def read_candidate_universe(path: Path) -> list[str]:
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("is_etf") is True and row.get("symbol"):
            result.append(str(row["symbol"]).strip().upper())
    return sorted(set(result))


def base_snapshot_keys(base_database: Path, tickers: Sequence[str]) -> set[tuple[str, str]]:
    rows = load_base_snapshot_rows(base_database, tickers)
    return {(str(row["etf_ticker"]), str(row["effective_date"])) for row in rows}


def refresh_contract(args: argparse.Namespace, tickers: Sequence[str]) -> dict[str, Any]:
    universe_path = Path(args.universe)
    base_stat = Path(args.base_database).stat()
    return {
        "provider": "fmp",
        "purpose": "v15 connected strict-eligible ETF historical constituent gap overlay",
        "phase": args.phase,
        "from": args.start_date,
        "to": args.end_date,
        "tickers": list(tickers),
        "ticker_count": len(tickers),
        "universe_path": str(universe_path.resolve()),
        "universe_sha256": sha256_file(universe_path),
        "base_database": {
            "path": str(Path(args.base_database).resolve()),
            "bytes": int(base_stat.st_size),
            "mtime_ns": int(base_stat.st_mtime_ns),
        },
        "endpoints": {
            "dates": "/api/v4/etf-holdings/portfolio-date",
            "snapshots": "/api/v4/etf-holdings",
        },
        "download_rule": "capture snapshot only when key is absent from sealed base",
        "historical_backfill_is_true_point_in_time": False,
        "sealed_v14_gate_must_not_change": True,
    }


def overlay_counts(pipeline: DatasetPipeline) -> dict[str, int]:
    with pipeline.database.connect() as connection:
        tables = (
            "raw_artifacts",
            "capture_events",
            "etf_constituent_available_dates",
            "etf_constituent_snapshots",
            "etf_constituent_observations",
        )
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _first_text(row: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = str(row.get(name) or "").strip()
        if value:
            return value.upper()
    return None


def parse_bulk_payload(payload: bytes) -> tuple[list[dict[str, Any]], str, list[str]]:
    text = payload.decode("utf-8-sig")
    try:
        document = json.loads(text)
    except ValueError:
        reader = csv.DictReader(io.StringIO(text))
        rows = [dict(row) for row in reader]
        return rows, "csv", list(reader.fieldnames or [])
    if isinstance(document, list):
        rows = [dict(row) for row in document if isinstance(row, dict)]
        keys = sorted({key for row in rows for key in row})
        return rows, "json", keys
    raise ValueError("ETF holder bulk payload is neither a row list nor CSV")


def normalize_bulk_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "etf_ticker": _first_text(
            row,
            ("symbol", "etfSymbol", "etf_symbol", "holder", "etf", "fundSymbol"),
        ),
        "constituent_ticker": _first_text(
            row,
            ("asset", "holdingSymbol", "holding_symbol", "constituentSymbol", "ticker"),
        ),
        "constituent_name": next(
            (str(row.get(name)).strip() for name in ("name", "assetName", "holdingName") if row.get(name)),
            None,
        ),
        "isin": _first_text(row, ("isin", "ISIN")),
        "cusip": _first_text(row, ("cusip", "CUSIP")),
        "shares": _number_or_none(
            next((row.get(name) for name in ("sharesNumber", "shares", "balance") if row.get(name) not in (None, "")), None)
        ),
        "weight_percent": _number_or_none(
            next((row.get(name) for name in ("weightPercentage", "weight", "pctVal", "percentage") if row.get(name) not in (None, "")), None)
        ),
        "market_value": _number_or_none(
            next((row.get(name) for name in ("marketValue", "valueUsd", "value") if row.get(name) not in (None, "")), None)
        ),
    }


def initialize_bulk_schema(pipeline: DatasetPipeline) -> None:
    with pipeline.database.connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS fmp_etf_holder_bulk_parts (
              as_of_date TEXT NOT NULL,
              part INTEGER NOT NULL,
              captured_at_utc TEXT NOT NULL,
              payload_format TEXT NOT NULL,
              row_count INTEGER NOT NULL,
              raw_artifact_id INTEGER NOT NULL,
              payload_sha256 TEXT NOT NULL,
              field_names_json TEXT NOT NULL,
              PRIMARY KEY(as_of_date,part)
            );
            CREATE TABLE IF NOT EXISTS fmp_etf_holder_bulk_rows (
              as_of_date TEXT NOT NULL,
              part INTEGER NOT NULL,
              source_row_index INTEGER NOT NULL,
              etf_ticker TEXT,
              constituent_ticker TEXT,
              constituent_name TEXT,
              isin TEXT,
              cusip TEXT,
              shares REAL,
              weight_percent REAL,
              market_value REAL,
              raw_artifact_id INTEGER NOT NULL,
              raw_json TEXT NOT NULL,
              PRIMARY KEY(as_of_date,part,source_row_index)
            );
            CREATE INDEX IF NOT EXISTS idx_fmp_bulk_etf
              ON fmp_etf_holder_bulk_rows(as_of_date,etf_ticker);
            CREATE INDEX IF NOT EXISTS idx_fmp_bulk_constituent
              ON fmp_etf_holder_bulk_rows(as_of_date,constituent_ticker);
            """
        )


def checkpoint_payload(
    pipeline: DatasetPipeline, job_id: str, item_key: str
) -> tuple[bytes, dict[str, Any]]:
    with pipeline.database.connect() as connection:
        row = connection.execute(
            """
            SELECT r.id,r.raw_relative_path,r.payload_sha256,r.captured_at_utc
            FROM checkpoints c JOIN raw_artifacts r ON r.id=c.raw_artifact_id
            WHERE c.job_id=? AND c.source='fmp' AND c.item_key=? AND c.status='done'
            """,
            (job_id, item_key),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"missing completed bulk checkpoint payload: {item_key}")
    path = pipeline.data_root / str(row["raw_relative_path"])
    payload = gzip.decompress(path.read_bytes())
    return payload, dict(row)


def capture_bulk_part(
    pipeline: DatasetPipeline,
    api_key: str,
    part: int,
    retries: int,
    timeout: float,
) -> tuple[bytes, Any, bool]:
    safe_request = {
        "method": "GET",
        "url": FMP_ETF_HOLDER_BULK_URL,
        "params": {"part": part, "apikey": "***REDACTED***"},
        "logical_request": {"endpoint_contract": "fmp_stable_etf_holder_bulk", "part": part},
    }
    for attempt in range(retries + 1):
        limiter = pipeline.http.rate_limiters.get("fmp")
        if limiter is not None:
            limiter.acquire()
        response = pipeline.http.session.get(
            FMP_ETF_HOLDER_BULK_URL,
            params={"part": part, "apikey": api_key},
            timeout=timeout,
        )
        status = int(response.status_code)
        artifact = pipeline.raw_store.store(
            source="fmp",
            dataset="etf_holder_bulk_stable",
            partition_key=f"part_{part:04d}",
            payload=bytes(response.content),
            request=safe_request,
            response={
                "status_code": status,
                "content_type": response.headers.get("content-type"),
                "attempt": attempt + 1,
            },
        )
        if status == 429 and attempt < retries:
            time.sleep(float(response.headers.get("retry-after") or 60.0))
            continue
        terminal = status == 400 and bytes(response.content).strip() == FMP_BULK_END_MESSAGE
        if terminal:
            return bytes(response.content), artifact, True
        if status < 200 or status >= 300:
            raise RuntimeError(f"FMP ETF holder bulk part {part} returned HTTP {status} (raw artifact id={artifact.artifact_id})")
        return bytes(response.content), artifact, False
    raise AssertionError("bulk retry loop exhausted")


def prior_bulk_terminal_artifact(
    pipeline: DatasetPipeline, part: int
) -> tuple[bytes, dict[str, Any]] | None:
    with pipeline.database.connect() as connection:
        row = connection.execute(
            """
            SELECT id,raw_relative_path,payload_sha256,captured_at_utc
            FROM raw_artifacts
            WHERE source='fmp' AND dataset='etf_holder_bulk_stable'
              AND partition_key=? AND response_status=400
            ORDER BY id DESC LIMIT 1
            """,
            (f"part_{part:04d}",),
        ).fetchone()
    if row is None:
        return None
    payload = gzip.decompress((pipeline.data_root / str(row["raw_relative_path"])).read_bytes())
    if payload.strip() != FMP_BULK_END_MESSAGE:
        return None
    return payload, dict(row)


def existing_ingested_bulk_part(
    pipeline: DatasetPipeline, as_of_date: str, part: int
) -> dict[str, Any] | None:
    with pipeline.database.connect() as connection:
        row = connection.execute(
            """
            SELECT part,row_count,payload_format,payload_sha256,field_names_json
            FROM fmp_etf_holder_bulk_parts WHERE as_of_date=? AND part=?
            """,
            (as_of_date, part),
        ).fetchone()
    if row is None:
        return None
    return {
        "part": int(row["part"]),
        "rows": int(row["row_count"]),
        "payload_format": str(row["payload_format"]),
        "field_names": json.loads(str(row["field_names_json"])),
        "payload_sha256": str(row["payload_sha256"]),
        "reused_checkpoint": True,
    }


def ingest_bulk_part(
    pipeline: DatasetPipeline,
    as_of_date: str,
    part: int,
    rows: Sequence[Mapping[str, Any]],
    payload_format: str,
    field_names: Sequence[str],
    artifact: Mapping[str, Any] | Any,
) -> None:
    artifact_id = int(artifact["id"] if isinstance(artifact, Mapping) else artifact.artifact_id)
    captured_at = str(
        artifact["captured_at_utc"] if isinstance(artifact, Mapping) else artifact.captured_at_utc
    )
    payload_sha = str(
        artifact["payload_sha256"] if isinstance(artifact, Mapping) else artifact.payload_sha256
    )
    values = []
    for index, row in enumerate(rows):
        normalized = normalize_bulk_row(row)
        values.append(
            (
                as_of_date,
                part,
                index,
                normalized["etf_ticker"],
                normalized["constituent_ticker"],
                normalized["constituent_name"],
                normalized["isin"],
                normalized["cusip"],
                normalized["shares"],
                normalized["weight_percent"],
                normalized["market_value"],
                artifact_id,
                canonical_json(row),
            )
        )
    with pipeline.database.connect() as connection:
        connection.execute(
            "DELETE FROM fmp_etf_holder_bulk_rows WHERE as_of_date=? AND part=?",
            (as_of_date, part),
        )
        connection.executemany(
            """
            INSERT INTO fmp_etf_holder_bulk_rows (
              as_of_date,part,source_row_index,etf_ticker,constituent_ticker,
              constituent_name,isin,cusip,shares,weight_percent,market_value,
              raw_artifact_id,raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            values,
        )
        connection.execute(
            """
            INSERT INTO fmp_etf_holder_bulk_parts (
              as_of_date,part,captured_at_utc,payload_format,row_count,
              raw_artifact_id,payload_sha256,field_names_json
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(as_of_date,part) DO UPDATE SET
              captured_at_utc=excluded.captured_at_utc,
              payload_format=excluded.payload_format,
              row_count=excluded.row_count,
              raw_artifact_id=excluded.raw_artifact_id,
              payload_sha256=excluded.payload_sha256,
              field_names_json=excluded.field_names_json
            """,
            (
                as_of_date,
                part,
                captured_at,
                payload_format,
                len(rows),
                artifact_id,
                payload_sha,
                canonical_json(list(field_names)),
            ),
        )


def run_bulk(args: argparse.Namespace) -> int:
    credentials = load_credentials(secrets_path=args.secrets_file)
    if not credentials.fmp_api_key:
        raise RuntimeError("FMP_API_KEY is not configured")
    pipeline = DatasetPipeline(
        data_root=args.overlay_root,
        credentials=credentials,
        timeout_seconds=args.timeout,
        retries=args.retries,
    )
    initialize_bulk_schema(pipeline)
    job_id = f"v15-fmp-etf-holder-bulk:{args.as_of_date}"
    contract = {
        "provider": "fmp",
        "endpoint": "/stable/etf-holder-bulk",
        "account_tier_required": "ultimate",
        "as_of_capture_date": args.as_of_date,
        "start_part": args.start_part,
        "max_parts": args.max_parts,
        "purpose": "current/future topology only; never historical PIT imputation",
    }
    pipeline.database.register_job(job_id, "v15_fmp_etf_holder_bulk", contract, "2026-08-28.stable")
    part_receipts = []
    prior_digest = None
    stopped_on_empty = False
    stopped_on_invalid_part = False
    for part in range(args.start_part, args.start_part + args.max_parts):
        item_key = f"bulk-part:{part}"
        pipeline.database.ensure_checkpoint(job_id, "fmp", item_key, {"part": part})
        if pipeline.database.checkpoint_status(job_id, "fmp", item_key) == "done":
            ingested = existing_ingested_bulk_part(pipeline, args.as_of_date, part)
            if ingested is not None:
                if ingested["rows"] and ingested["payload_sha256"] == prior_digest:
                    raise RuntimeError(f"bulk pagination repeated payload at part {part}")
                prior_digest = ingested["payload_sha256"]
                part_receipts.append(ingested)
                print(json.dumps({"stage": "bulk", **ingested}, sort_keys=True), flush=True)
                continue
            payload, artifact = checkpoint_payload(pipeline, job_id, item_key)
            reused = True
            terminal = payload.strip() == FMP_BULK_END_MESSAGE
        else:
            pipeline.database.mark_checkpoint_running(job_id, "fmp", item_key)
            prior_terminal = prior_bulk_terminal_artifact(pipeline, part)
            if prior_terminal is not None:
                payload, artifact = prior_terminal
                terminal = True
                reused = True
            else:
                payload, artifact, terminal = capture_bulk_part(
                    pipeline, credentials.fmp_api_key, part, args.retries, args.timeout
                )
                reused = False
        if terminal:
            artifact_id = int(artifact["id"] if isinstance(artifact, Mapping) else artifact.artifact_id)
            pipeline.database.mark_checkpoint_done(job_id, "fmp", item_key, artifact_id, 0)
            part_receipts.append(
                {
                    "part": part,
                    "rows": 0,
                    "terminal_http_400_invalid_part": True,
                    "payload_sha256": sha256_bytes(payload),
                    "reused_checkpoint": reused,
                }
            )
            print(json.dumps({"stage": "bulk_terminal", **part_receipts[-1]}, sort_keys=True), flush=True)
            stopped_on_invalid_part = True
            break
        rows, payload_format, field_names = parse_bulk_payload(payload)
        payload_digest = sha256_bytes(payload)
        if rows and payload_digest == prior_digest:
            raise RuntimeError(f"bulk pagination repeated payload at part {part}")
        prior_digest = payload_digest
        ingest_bulk_part(
            pipeline,
            args.as_of_date,
            part,
            rows,
            payload_format,
            field_names,
            artifact,
        )
        if not reused:
            pipeline.database.mark_checkpoint_done(
                job_id,
                "fmp",
                item_key,
                int(artifact.artifact_id),
                len(rows),
            )
        part_receipts.append(
            {
                "part": part,
                "rows": len(rows),
                "payload_format": payload_format,
                "field_names": list(field_names),
                "payload_sha256": payload_digest,
                "reused_checkpoint": reused,
            }
        )
        print(json.dumps({"stage": "bulk", **part_receipts[-1]}, sort_keys=True), flush=True)
        if not rows:
            stopped_on_empty = True
            break
    with pipeline.database.connect() as connection:
        summary_row = connection.execute(
            """
            SELECT COUNT(*) rows,COUNT(DISTINCT etf_ticker) etfs,
                   COUNT(DISTINCT constituent_ticker) constituents,
                   SUM(CASE WHEN etf_ticker IS NULL THEN 1 ELSE 0 END) missing_etf,
                   SUM(CASE WHEN constituent_ticker IS NULL THEN 1 ELSE 0 END) missing_constituent
            FROM fmp_etf_holder_bulk_rows WHERE as_of_date=?
            """,
            (args.as_of_date,),
        ).fetchone()
    raw_verify = pipeline.raw_store.verify_all()
    receipt = {
        "schema_version": "quant.etf_flow_v15.fmp_etf_holder_bulk.v1",
        "generated_at_utc": utc_now(),
        "ok": bool(part_receipts)
        and raw_verify["ok"]
        and (stopped_on_empty or stopped_on_invalid_part or args.allow_partial),
        "contract": contract,
        "credential_status": credentials.status(),
        "rate_limit_policy": rate_limit_policy(),
        "parts": part_receipts,
        "stopped_on_empty_part": stopped_on_empty,
        "stopped_on_invalid_part": stopped_on_invalid_part,
        "summary": {key: int(summary_row[key] or 0) for key in summary_row.keys()},
        "raw_verify": raw_verify,
        "checkpoint_summary": pipeline.database.checkpoint_summary(job_id),
        "interpretation_boundary": {
            "current_capture_is_historical_pit": False,
            "permitted_use": "current topology and future as-observed captures",
            "forbidden_use": "do not backfill v14 historical topology from this current snapshot",
        },
    }
    receipt_path = args.output_root / "v15_fmp_etf_holder_bulk_receipt.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "sha256": sha256_file(receipt_path), "ok": receipt["ok"], **receipt["summary"]}, sort_keys=True))
    return 0 if receipt["ok"] else 2


def run_refresh(args: argparse.Namespace) -> int:
    if args.start_date > args.end_date:
        raise ValueError("from date must be <= to date")
    tickers = read_candidate_universe(args.universe)
    if not tickers:
        raise ValueError("candidate universe is empty")
    credentials = load_credentials(secrets_path=args.secrets_file)
    if not credentials.fmp_api_key:
        raise RuntimeError("FMP_API_KEY is not configured")
    contract = refresh_contract(args, tickers)
    job_id = "v15-fmp-constituent-refresh:" + sha256_bytes(
        canonical_json({**contract, "phase": "all"}).encode("utf-8")
    )[:16]
    pipeline = DatasetPipeline(
        data_root=args.overlay_root,
        credentials=credentials,
        timeout_seconds=args.timeout,
        retries=args.retries,
    )
    pipeline.database.register_job(job_id, "v15_fmp_constituent_refresh", contract, "2026-08-28.v1")
    base_keys = base_snapshot_keys(args.base_database, tickers)
    before = overlay_counts(pipeline)
    counters = Counter(
        {
            "candidate_etfs": len(tickers),
            "date_lists_done": 0,
            "date_lists_skipped": 0,
            "date_lists_failed": 0,
            "provider_dates_in_range": 0,
            "existing_base_keys": 0,
            "missing_snapshot_keys": 0,
            "snapshot_done": 0,
            "snapshot_empty": 0,
            "snapshot_skipped": 0,
            "snapshot_failed": 0,
            "snapshot_records": 0,
        }
    )
    errors: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for position, ticker in enumerate(tickers, 1):
        date_key = f"dates:{ticker}"
        pipeline.database.ensure_checkpoint(
            job_id,
            "fmp",
            date_key,
            {"stage": "available_dates", "etf_ticker": ticker},
        )
        try:
            if pipeline.database.checkpoint_status(job_id, "fmp", date_key) == "done":
                provider_dates = pipeline.etf_constituents.store.dates_for_etf(ticker)
                counters["date_lists_skipped"] += 1
            else:
                pipeline.database.mark_checkpoint_running(job_id, "fmp", date_key)
                provider_dates, artifact = pipeline.etf_constituents.provider.capture_dates(ticker)
                pipeline.etf_constituents.store.ingest_dates(ticker, provider_dates, artifact)
                pipeline.database.mark_checkpoint_done(job_id, "fmp", date_key, artifact.artifact_id, len(provider_dates))
                counters["date_lists_done"] += 1
        except Exception as error:
            pipeline.database.mark_checkpoint_failed(job_id, "fmp", date_key, f"{type(error).__name__}: {error}")
            counters["date_lists_failed"] += 1
            errors.append({"etf_ticker": ticker, "stage": "dates", "error": str(error)})
            if not args.continue_on_error:
                raise
            continue
        eligible = sorted(date for date in provider_dates if args.start_date <= date <= args.end_date)
        counters["provider_dates_in_range"] += len(eligible)
        for effective_date in eligible:
            if (ticker, effective_date) in base_keys:
                counters["existing_base_keys"] += 1
                continue
            missing_rows.append({"etf_ticker": ticker, "effective_date": effective_date})
        if position % 100 == 0 or position == len(tickers):
            print(json.dumps({"stage": "dates", "position": position, **dict(counters)}, sort_keys=True), flush=True)
    missing_rows.sort(key=lambda row: (row["etf_ticker"], row["effective_date"]))
    counters["missing_snapshot_keys"] = len(missing_rows)
    missing_path = args.output_root / "v15_fmp_missing_snapshot_keys.jsonl"
    atomic_jsonl(missing_path, missing_rows)
    if args.phase in ("download", "all"):
        for position, row in enumerate(missing_rows, 1):
            ticker = row["etf_ticker"]
            effective_date = row["effective_date"]
            item_key = f"snapshot:{ticker}:{effective_date}"
            pipeline.database.ensure_checkpoint(
                job_id,
                "fmp",
                item_key,
                {"stage": "snapshot", **row},
            )
            if pipeline.database.checkpoint_status(job_id, "fmp", item_key) == "done":
                counters["snapshot_skipped"] += 1
                continue
            pipeline.database.mark_checkpoint_running(job_id, "fmp", item_key)
            try:
                capture = pipeline.etf_constituents.provider.capture_snapshot(ticker, effective_date)
                pipeline.etf_constituents.store.ingest_snapshot(ticker, effective_date, capture)
                pipeline.database.mark_checkpoint_done(
                    job_id,
                    "fmp",
                    item_key,
                    capture.artifact.artifact_id,
                    len(capture.records),
                )
                counters["snapshot_records"] += len(capture.records)
                counters["snapshot_done" if capture.records else "snapshot_empty"] += 1
            except Exception as error:
                pipeline.database.mark_checkpoint_failed(job_id, "fmp", item_key, f"{type(error).__name__}: {error}")
                counters["snapshot_failed"] += 1
                errors.append({**row, "stage": "snapshot", "error": str(error)})
                if not args.continue_on_error:
                    raise
            if position % 50 == 0 or position == len(missing_rows):
                print(json.dumps({"stage": "snapshots", "position": position, **dict(counters)}, sort_keys=True), flush=True)
    after = overlay_counts(pipeline)
    overlap = 0
    with pipeline.database.connect() as connection:
        overlay_keys = {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT etf_ticker,effective_date FROM etf_constituent_snapshots WHERE provider='fmp'"
            )
        }
    overlap = len(overlay_keys & base_keys)
    receipt = {
        "schema_version": "quant.etf_flow_v15.fmp_constituent_refresh.v1",
        "generated_at_utc": utc_now(),
        "phase": args.phase,
        "job_id": job_id,
        "ok": counters["date_lists_failed"] == 0 and counters["snapshot_failed"] == 0 and overlap == 0,
        "contract": contract,
        "credential_status": credentials.status(),
        "rate_limit_policy": rate_limit_policy(),
        "counters": dict(counters),
        "overlay": {
            "root": str(args.overlay_root),
            "database": str(pipeline.database.db_path),
            "before": before,
            "after": after,
            "overlap_with_sealed_base_snapshot_keys": overlap,
        },
        "missing_keys": {
            "path": str(missing_path),
            "sha256": sha256_file(missing_path),
            "rows": len(missing_rows),
        },
        "checkpoint_summary": pipeline.database.checkpoint_summary(job_id),
        "errors": errors,
        "interpretation_boundary": {
            "historical_backfill_is_true_as_observed": False,
            "changes_v14_gate": False,
            "may_be_used_for": "posthoc repaired-topology sensitivity and future pipeline repair",
        },
    }
    receipt_path = args.output_root / f"v15_fmp_{args.phase}_receipt.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "sha256": sha256_file(receipt_path), "ok": receipt["ok"], **dict(counters)}, sort_keys=True))
    return 0 if receipt["ok"] else 2


def table_count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def run_combine(args: argparse.Namespace) -> int:
    source_path = Path(args.oracle_incremental).resolve()
    overlay_path = Path(args.overlay_database).resolve()
    output_path = Path(args.output_database).resolve()
    if not source_path.is_file() or not overlay_path.is_file():
        raise FileNotFoundError("combine inputs must exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if output_path.exists() and not args.replace:
        raise FileExistsError(output_path)
    if temporary.exists():
        if not args.replace:
            raise FileExistsError(temporary)
        temporary.unlink()
    source_stat_before = source_path.stat()
    source_sha_before = sha256_file(source_path)
    source = readonly_connection(source_path)
    destination = sqlite3.connect(str(temporary), timeout=120.0)
    try:
        source.backup(destination, pages=4096)
        destination.commit()
    finally:
        source.close()
        destination.close()
    connection = sqlite3.connect(str(temporary), timeout=120.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("ATTACH DATABASE ? AS overlay", (str(overlay_path),))
        main_columns = [
            str(row[1])
            for row in connection.execute("PRAGMA main.table_info(etf_constituent_observations)")
        ]
        overlay_columns = [
            str(row[1])
            for row in connection.execute("PRAGMA overlay.table_info(etf_constituent_observations)")
        ]
        if main_columns != overlay_columns:
            raise ValueError("constituent observation schema mismatch")
        overlapping_row_keys = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM overlay.etf_constituent_observations o
                JOIN main.etf_constituent_observations m
                  ON m.provider=o.provider AND m.etf_ticker=o.etf_ticker
                 AND m.constituent_key=o.constituent_key
                 AND m.effective_date=o.effective_date
                """
            ).fetchone()[0]
        )
        replacement_snapshot_keys = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT DISTINCT o.provider,o.etf_ticker,o.effective_date
                  FROM overlay.etf_constituent_observations o
                  WHERE EXISTS (
                    SELECT 1 FROM main.etf_constituent_observations m
                    WHERE m.provider=o.provider AND m.etf_ticker=o.etf_ticker
                      AND m.effective_date=o.effective_date
                  )
                )
                """
            ).fetchone()[0]
        )
        replaced_existing_rows = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM main.etf_constituent_observations m
                WHERE EXISTS (
                  SELECT 1 FROM overlay.etf_constituent_observations o
                  WHERE o.provider=m.provider AND o.etf_ticker=m.etf_ticker
                    AND o.effective_date=m.effective_date
                )
                """
            ).fetchone()[0]
        )
        daily_before = table_count(connection, "daily_observations")
        constituents_before = table_count(connection, "etf_constituent_observations")
        overlay_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM overlay.etf_constituent_observations"
            ).fetchone()[0]
        )
        overlay_snapshot_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT DISTINCT provider,etf_ticker,effective_date
                  FROM overlay.etf_constituent_observations
                )
                """
            ).fetchone()[0]
        )
        columns_sql = ",".join(f'"{column}"' for column in main_columns)
        connection.execute(
            """
            DELETE FROM main.etf_constituent_observations
            WHERE EXISTS (
              SELECT 1 FROM overlay.etf_constituent_observations o
              WHERE o.provider=main.etf_constituent_observations.provider
                AND o.etf_ticker=main.etf_constituent_observations.etf_ticker
                AND o.effective_date=main.etf_constituent_observations.effective_date
            )
            """
        )
        connection.execute(
            f"INSERT INTO main.etf_constituent_observations ({columns_sql}) "
            f"SELECT {columns_sql} FROM overlay.etf_constituent_observations"
        )
        connection.commit()
        daily_after = table_count(connection, "daily_observations")
        constituents_after = table_count(connection, "etf_constituent_observations")
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if daily_after != daily_before:
            raise ValueError("daily observation rows changed during constituent merge")
        expected_constituents = constituents_before - replaced_existing_rows + overlay_rows
        if constituents_after != expected_constituents:
            raise ValueError("constituent merge row delta mismatch")
        if integrity != "ok":
            raise ValueError(f"combined incremental integrity failed: {integrity}")
    finally:
        connection.close()
    source_stat_after = source_path.stat()
    source_sha_after = sha256_file(source_path)
    if (
        source_stat_after.st_size != source_stat_before.st_size
        or source_stat_after.st_mtime_ns != source_stat_before.st_mtime_ns
        or source_sha_after != source_sha_before
    ):
        raise RuntimeError("oracle incremental changed during read-only backup")
    if output_path.exists():
        output_path.unlink()
    os.replace(temporary, output_path)
    receipt = {
        "schema_version": "quant.etf_flow_v15.combined_incremental.v1",
        "generated_at_utc": utc_now(),
        "ok": True,
        "source_oracle_incremental": {
            "path": str(source_path),
            "bytes": int(source_stat_after.st_size),
            "mtime_ns": int(source_stat_after.st_mtime_ns),
            "sha256": source_sha_after,
            "unchanged": True,
        },
        "constituent_overlay": {
            "path": str(overlay_path),
            "sha256": sha256_file(overlay_path),
            "inserted_rows": overlay_rows,
            "new_snapshot_count": overlay_snapshot_count - replacement_snapshot_keys,
            "overlay_snapshot_count": overlay_snapshot_count,
            "replacement_snapshot_keys": replacement_snapshot_keys,
            "replaced_existing_rows": replaced_existing_rows,
            "overlapping_business_row_keys": overlapping_row_keys,
        },
        "combined_incremental": {
            "path": str(output_path),
            "bytes": int(output_path.stat().st_size),
            "sha256": sha256_file(output_path),
            "daily_observation_rows": daily_after,
            "constituent_rows_before": constituents_before,
            "constituent_rows_after": constituents_after,
            "integrity_check": integrity,
        },
        "interpretation_boundary": {
            "source_database_modified": False,
            "bulk_current_rows_merged": False,
            "historical_overlay_is_true_as_observed": False,
            "permitted_use": "posthoc repaired-topology sensitivity only",
        },
    }
    receipt_path = args.output_root / "v15_repaired_incremental_receipt.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "sha256": sha256_file(receipt_path), **receipt["combined_incremental"]}, sort_keys=True))
    return 0


def _snapshot_edge_map(
    graph_root: Path,
    manifest: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray], Path]:
    """Load one graph snapshot and map local edge indices to stable symbols."""
    path = snapshot_path(graph_root, manifest_row)
    with np.load(path, allow_pickle=False) as snapshot:
        arrays = {name: np.asarray(snapshot[name]).copy() for name in snapshot.files}
    required = {
        "stock_symbols",
        "stock_x",
        "targets",
        "target_mask",
        "etf_ids",
        "edge_index",
        "edge_attr",
        "signal_position",
        "flow_position",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"snapshot missing required arrays {missing}: {path}")
    edge_index = np.asarray(arrays["edge_index"], dtype=np.int64)
    edge_attr = np.asarray(arrays["edge_attr"], dtype=np.float64)
    stock_symbols = [str(value) for value in arrays["stock_symbols"].tolist()]
    etf_ids = np.asarray(arrays["etf_ids"], dtype=np.int64)
    vocabulary = [str(value) for value in manifest["etf_vocabulary"]]
    if edge_index.shape != (2, edge_attr.shape[0]):
        raise ValueError(f"edge shape mismatch: {path}")
    result: dict[tuple[str, str], np.ndarray] = {}
    for position in range(edge_index.shape[1]):
        stock_index = int(edge_index[0, position])
        local_etf_index = int(edge_index[1, position])
        if stock_index < 0 or stock_index >= len(stock_symbols):
            raise ValueError(f"stock index out of range: {path}")
        if local_etf_index < 0 or local_etf_index >= len(etf_ids):
            raise ValueError(f"local ETF index out of range: {path}")
        global_etf_index = int(etf_ids[local_etf_index])
        if global_etf_index < 0 or global_etf_index >= len(vocabulary):
            raise ValueError(f"global ETF index out of range: {path}")
        key = (stock_symbols[stock_index], vocabulary[global_etf_index])
        if key in result:
            raise ValueError(f"duplicate stock/ETF edge {key}: {path}")
        result[key] = edge_attr[position]
    return result, arrays, path


def _array_comparison(original: np.ndarray, repaired: np.ndarray) -> dict[str, Any]:
    original = np.asarray(original)
    repaired = np.asarray(repaired)
    same_shape = original.shape == repaired.shape
    result: dict[str, Any] = {
        "equal": False,
        "shape_original": list(original.shape),
        "shape_repaired": list(repaired.shape),
        "dtype_original": str(original.dtype),
        "dtype_repaired": str(repaired.dtype),
    }
    if not same_shape:
        return result
    if np.issubdtype(original.dtype, np.number) and np.issubdtype(
        repaired.dtype, np.number
    ):
        old_float = original.astype(np.float64, copy=False)
        new_float = repaired.astype(np.float64, copy=False)
        old_nan = np.isnan(old_float)
        new_nan = np.isnan(new_float)
        finite_both = np.isfinite(old_float) & np.isfinite(new_float)
        finite_delta = np.abs(new_float[finite_both] - old_float[finite_both])
        result.update(
            {
                "nan_mask_equal": bool(np.array_equal(old_nan, new_nan)),
                "nan_count_original": int(old_nan.sum()),
                "nan_count_repaired": int(new_nan.sum()),
                "finite_value_max_abs_delta": (
                    float(finite_delta.max()) if finite_delta.size else 0.0
                ),
                "finite_value_changed_count": int((finite_delta > 0.0).sum()),
            }
        )
        result["equal"] = bool(
            np.array_equal(original, repaired, equal_nan=True)
        )
    else:
        result["equal"] = bool(np.array_equal(original, repaired))
    return result


def _model_flow_history(
    manifest: Mapping[str, Any],
    snapshot_arrays: Mapping[str, np.ndarray],
    flow_values: np.ndarray,
    flow_available: np.ndarray,
) -> np.ndarray:
    """Reproduce the exact 60-session ETF tensor exposed by GraphDataset.load."""
    lookback = int(manifest.get("feature_contract", {}).get("flow_lookback_sessions", 60))
    cube_start = int(manifest["flow_cube"]["session_start_position"])
    flow_position = int(snapshot_arrays["flow_position"])
    signal_position = int(snapshot_arrays["signal_position"])
    etf_ids = np.asarray(snapshot_arrays["etf_ids"], dtype=np.int64)
    local_end = flow_position - cube_start
    local_start = local_end - lookback + 1
    if local_end < 0 or local_end >= flow_values.shape[0]:
        raise ValueError("flow cube does not cover snapshot flow position")
    history = np.full(
        (lookback, len(etf_ids), flow_values.shape[-1]),
        np.nan,
        dtype=np.float32,
    )
    if local_start < 0:
        source_start = 0
        target_start = -local_start
    else:
        source_start = local_start
        target_start = 0
    source = np.asarray(
        flow_values[source_start : local_end + 1, etf_ids], dtype=np.float32
    )
    availability = np.asarray(
        flow_available[source_start : local_end + 1, etf_ids], dtype=np.int32
    )
    visible = (availability >= 0) & (availability <= signal_position)
    source = np.where(visible[..., None], source, np.nan)
    history[target_start : target_start + len(source)] = source
    return np.transpose(history, (1, 0, 2))


def compare_graph_roots(
    original_root: Path,
    repaired_root: Path,
    dates: Sequence[str],
) -> dict[str, Any]:
    """Compare every requested snapshot without treating the repair as a clean gate."""
    roots = {"original": Path(original_root), "repaired": Path(repaired_root)}
    manifests: dict[str, dict[str, Any]] = {}
    rows_by_root: dict[str, dict[str, Mapping[str, Any]]] = {}
    for label, root in roots.items():
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[label] = manifest
        rows_by_root[label] = {
            str(row["signal_date"]): row for row in manifest["snapshots"]
        }
        missing = [date for date in dates if date not in rows_by_root[label]]
        if missing:
            raise ValueError(f"{label} graph missing dates: {missing}")
    vocabulary_equal = manifests["original"]["etf_vocabulary"] == manifests["repaired"]["etf_vocabulary"]
    if not vocabulary_equal:
        raise ValueError("ETF vocabulary changed; stable edge comparison is invalid")
    non_topology_names = (
        "stock_symbols",
        "stock_x",
        "targets",
        "target_mask",
        "etf_ids",
        "signal_position",
        "flow_position",
    )
    per_date: dict[str, Any] = {}
    flow_cubes = {
        label: {
            "values": np.load(root / "flow_values.npy", allow_pickle=False, mmap_mode="r"),
            "available": np.load(
                root / "flow_available_session_index.npy",
                allow_pickle=False,
                mmap_mode="r",
            ),
        }
        for label, root in roots.items()
    }
    changed_etfs: set[str] = set()
    changed_stocks: set[str] = set()
    total_added = 0
    total_removed = 0
    total_common_attr_changed = 0
    for date in dates:
        old_edges, old_arrays, old_path = _snapshot_edge_map(
            roots["original"], manifests["original"], rows_by_root["original"][date]
        )
        new_edges, new_arrays, new_path = _snapshot_edge_map(
            roots["repaired"], manifests["repaired"], rows_by_root["repaired"][date]
        )
        old_keys = set(old_edges)
        new_keys = set(new_edges)
        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)
        common = old_keys & new_keys
        attr_changed_pairs = []
        column_max_abs_delta = [0.0, 0.0, 0.0]
        column_changed_counts = [0, 0, 0]
        for key in common:
            delta = np.abs(new_edges[key] - old_edges[key])
            for column in range(3):
                column_max_abs_delta[column] = max(
                    column_max_abs_delta[column], float(delta[column])
                )
                if float(delta[column]) > 1e-7:
                    column_changed_counts[column] += 1
            if bool(np.any(delta > 1e-7)):
                attr_changed_pairs.append(key)
        changed_pairs = set(added) | set(removed) | set(attr_changed_pairs)
        changed_etfs.update(etf for _, etf in changed_pairs)
        changed_stocks.update(stock for stock, _ in changed_pairs)
        total_added += len(added)
        total_removed += len(removed)
        total_common_attr_changed += len(attr_changed_pairs)
        array_checks = {
            name: _array_comparison(old_arrays[name], new_arrays[name])
            for name in non_topology_names
        }
        model_flow_check = _array_comparison(
            _model_flow_history(
                manifests["original"],
                old_arrays,
                flow_cubes["original"]["values"],
                flow_cubes["original"]["available"],
            ),
            _model_flow_history(
                manifests["repaired"],
                new_arrays,
                flow_cubes["repaired"]["values"],
                flow_cubes["repaired"]["available"],
            ),
        )
        per_date[date] = {
            "price_date": str(rows_by_root["original"][date]["price_date"]),
            "flow_date": str(rows_by_root["original"][date]["flow_date"]),
            "paths": {"original": str(old_path), "repaired": str(new_path)},
            "sha256": {
                "original": sha256_file(old_path),
                "repaired": sha256_file(new_path),
            },
            "non_topology_arrays": array_checks,
            "all_non_topology_arrays_equal": all(
                check["equal"] for check in array_checks.values()
            ),
            "model_flow_history": model_flow_check,
            "edges": {
                "original": len(old_edges),
                "repaired": len(new_edges),
                "delta": len(new_edges) - len(old_edges),
                "added": len(added),
                "removed": len(removed),
                "common": len(common),
                "common_with_changed_attributes": len(attr_changed_pairs),
                "changed_pair_count": len(changed_pairs),
                "changed_stock_count": len({stock for stock, _ in changed_pairs}),
                "changed_etf_count": len({etf for _, etf in changed_pairs}),
                "attribute_columns": {
                    "weight": {
                        "changed_count": column_changed_counts[0],
                        "max_abs_delta": column_max_abs_delta[0],
                    },
                    "age_scaled_252": {
                        "changed_count": column_changed_counts[1],
                        "max_abs_delta": column_max_abs_delta[1],
                    },
                    "observed_exact_t2": {
                        "changed_count": column_changed_counts[2],
                        "max_abs_delta": column_max_abs_delta[2],
                    },
                },
                "added_examples": [list(key) for key in added[:20]],
                "removed_examples": [list(key) for key in removed[:20]],
                "changed_attribute_examples": [
                    {
                        "stock": stock,
                        "etf": etf,
                        "original": old_edges[(stock, etf)].tolist(),
                        "repaired": new_edges[(stock, etf)].tolist(),
                    }
                    for stock, etf in sorted(attr_changed_pairs)[:20]
                ],
            },
        }
    flow_files = {}
    for name in ("flow_values.npy", "flow_available_session_index.npy"):
        old_path = roots["original"] / name
        new_path = roots["repaired"] / name
        old_sha = sha256_file(old_path)
        new_sha = sha256_file(new_path)
        semantic = _array_comparison(
            np.load(old_path, allow_pickle=False, mmap_mode="r"),
            np.load(new_path, allow_pickle=False, mmap_mode="r"),
        )
        flow_files[name] = {
            "original_sha256": old_sha,
            "repaired_sha256": new_sha,
            "byte_equal": old_sha == new_sha,
            "semantic": semantic,
        }
    return {
        "schema_version": "quant.etf_flow_v15.repaired_graph_impact.v1",
        "generated_at_utc": utc_now(),
        "test_dates": list(dates),
        "roots": {
            label: {
                "path": str(root.resolve()),
                "manifest_sha256": sha256_file(root / "manifest.json"),
                "quality_gate": manifests[label].get("quality_gate"),
            }
            for label, root in roots.items()
        },
        "contracts": {
            "etf_vocabulary_equal": vocabulary_equal,
            "flow_files": flow_files,
            "all_non_topology_arrays_equal": all(
                row["all_non_topology_arrays_equal"] for row in per_date.values()
            ),
            "all_model_flow_histories_equal": all(
                row["model_flow_history"]["equal"] for row in per_date.values()
            ),
        },
        "aggregate": {
            "snapshot_count": len(dates),
            "changed_snapshot_count": sum(
                int(row["edges"]["changed_pair_count"] > 0) for row in per_date.values()
            ),
            "unchanged_snapshot_count": sum(
                int(row["edges"]["changed_pair_count"] == 0) for row in per_date.values()
            ),
            "added_edge_occurrences": total_added,
            "removed_edge_occurrences": total_removed,
            "common_edge_attribute_changed_occurrences": total_common_attr_changed,
            "changed_etf_union_count": len(changed_etfs),
            "changed_stock_union_count": len(changed_stocks),
            "changed_etfs": sorted(changed_etfs),
            "changed_stocks": sorted(changed_stocks),
        },
        "per_date": per_date,
        "interpretation_boundary": {
            "historical_backfill_is_true_as_observed": False,
            "changes_v14_clean_gate": False,
            "permitted_use": "posthoc repaired-topology and model-sensitivity analysis only",
        },
    }


def run_compare_graphs(args: argparse.Namespace) -> int:
    report = compare_graph_roots(
        args.original_graph_root,
        args.repaired_graph_root,
        tuple(args.test_dates or TEST_DATES),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "v15_repaired_graph_impact_audit.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "sha256": sha256_file(report_path),
                **report["aggregate"],
            },
            sort_keys=True,
        )
    )
    return 0


def current_bulk_topology_audit(
    *,
    bulk_database: Path,
    universe_path: Path,
    graph_root: Path,
    signal_date: str,
    as_of_date: str,
) -> dict[str, Any]:
    candidates = []
    for line in Path(universe_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("is_etf") is True and row.get("symbol"):
                candidates.append(row)
    candidate_symbols = sorted({str(row["symbol"]).upper() for row in candidates})
    last_strict = sorted(
        {
            str(row["symbol"]).upper()
            for row in candidates
            if bool(row.get("strict_eligible_on_last_test_date"))
        }
    )
    manifest_path = Path(graph_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {str(row["signal_date"]): row for row in manifest["snapshots"]}
    if signal_date not in rows:
        raise ValueError(f"graph missing signal date {signal_date}")
    historical_edges, arrays, snapshot_file = _snapshot_edge_map(
        graph_root, manifest, rows[signal_date]
    )
    stocks = {str(value) for value in arrays["stock_symbols"].tolist()}
    bulk_pairs: dict[tuple[str, str], float] = defaultdict(float)
    bulk_etfs: set[str] = set()
    etfs_with_stock_relation: set[str] = set()
    last_updated = Counter()
    counters = Counter()
    with readonly_connection(bulk_database) as connection:
        available = connection.execute(
            "SELECT COUNT(*) FROM fmp_etf_holder_bulk_parts WHERE as_of_date=?",
            (as_of_date,),
        ).fetchone()[0]
        if int(available) == 0:
            raise ValueError(f"bulk capture missing as_of_date {as_of_date}")
        for start in range(0, len(candidate_symbols), 400):
            chunk = candidate_symbols[start : start + 400]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT etf_ticker,constituent_ticker,weight_percent,raw_json "
                "FROM fmp_etf_holder_bulk_rows WHERE as_of_date=? "
                f"AND etf_ticker IN ({placeholders})"
            )
            for row in connection.execute(query, (as_of_date, *chunk)):
                ticker = str(row["etf_ticker"] or "").upper()
                constituent = str(row["constituent_ticker"] or "").upper()
                counters["candidate_bulk_rows"] += 1
                if ticker:
                    bulk_etfs.add(ticker)
                if not constituent:
                    counters["candidate_bulk_rows_missing_constituent_ticker"] += 1
                    continue
                if constituent not in stocks:
                    counters["candidate_bulk_rows_outside_graph_stock_universe"] += 1
                    continue
                etfs_with_stock_relation.add(ticker)
                weight = _number_or_none(row["weight_percent"])
                bulk_pairs[(constituent, ticker)] += float(weight or 0.0) / 100.0
                try:
                    raw = json.loads(str(row["raw_json"]))
                except ValueError:
                    raw = {}
                updated = str(raw.get("lastUpdated") or "UNKNOWN")
                last_updated[updated] += 1
    def compare(symbols: Sequence[str]) -> dict[str, Any]:
        selected = set(symbols)
        old = {pair for pair in historical_edges if pair[1] in selected}
        new = {pair for pair in bulk_pairs if pair[1] in selected}
        added = sorted(new - old)
        removed = sorted(old - new)
        common = old & new
        union = old | new
        return {
            "etf_count": len(selected),
            "bulk_etf_coverage_count": len(selected & bulk_etfs),
            "bulk_missing_etf_count": len(selected - bulk_etfs),
            "bulk_missing_etfs": sorted(selected - bulk_etfs),
            "bulk_etf_with_graph_stock_relation_count": len(
                selected & etfs_with_stock_relation
            ),
            "bulk_etfs_without_graph_stock_relation_count": len(
                (selected & bulk_etfs) - etfs_with_stock_relation
            ),
            "bulk_etfs_without_graph_stock_relation": sorted(
                (selected & bulk_etfs) - etfs_with_stock_relation
            ),
            "historical_pair_count": len(old),
            "current_bulk_pair_count": len(new),
            "common_pair_count": len(common),
            "added_pair_count": len(added),
            "removed_pair_count": len(removed),
            "pair_jaccard": float(len(common) / len(union)) if union else 1.0,
            "current_stock_coverage_count": len({stock for stock, _ in new}),
            "added_examples": [list(value) for value in added[:30]],
            "removed_examples": [list(value) for value in removed[:30]],
        }
    return {
        "schema_version": "quant.etf_flow_v15.current_bulk_topology_audit.v1",
        "generated_at_utc": utc_now(),
        "bulk": {
            "database": str(Path(bulk_database).resolve()),
            "database_sha256": sha256_file(bulk_database),
            "as_of_capture_date": as_of_date,
            "parts_present": int(available),
        },
        "historical_reference": {
            "graph_root": str(Path(graph_root).resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "signal_date": signal_date,
            "snapshot_path": str(snapshot_file),
            "snapshot_sha256": sha256_file(snapshot_file),
        },
        "universe": {
            "path": str(Path(universe_path).resolve()),
            "sha256": sha256_file(universe_path),
            "ever_strict_candidate_count": len(candidate_symbols),
            "strict_on_last_test_date_count": len(last_strict),
            "graph_stock_count": len(stocks),
        },
        "row_counters": dict(counters),
        "last_updated_for_in_scope_pairs": dict(sorted(last_updated.items())),
        "ever_strict_candidates": compare(candidate_symbols),
        "strict_on_last_test_date": compare(last_strict),
        "interpretation_boundary": {
            "current_bulk_is_historical_pit": False,
            "july_to_august_difference_is_provider_error": False,
            "july_to_august_difference_mixes_true_turnover_and_data_refresh": True,
            "permitted_use": "future topology coverage planning and daily as-observed capture",
            "forbidden_use": "do not impute this current topology into v14 history",
        },
    }


def run_current_bulk_audit(args: argparse.Namespace) -> int:
    report = current_bulk_topology_audit(
        bulk_database=args.bulk_database,
        universe_path=args.universe,
        graph_root=args.graph_root,
        signal_date=args.signal_date,
        as_of_date=args.as_of_date,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "v15_current_bulk_topology_audit.json"
    atomic_json(path, report)
    print(
        json.dumps(
            {
                "report": str(path),
                "sha256": sha256_file(path),
                "ever_strict_candidates": report["ever_strict_candidates"],
                "strict_on_last_test_date": report["strict_on_last_test_date"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m training.quant_flow_graph_v15")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="derive the exact connected strict ETF universe")
    audit.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    audit.add_argument("--event-cube", type=Path, default=DEFAULT_EVENT_CUBE)
    audit.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    audit.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    audit.add_argument("--test-dates", nargs="*")
    refresh = commands.add_parser("refresh", help="discover and selectively download FMP gaps")
    refresh.add_argument("--phase", choices=("discover", "download", "all"), required=True)
    refresh.add_argument("--base-database", type=Path, default=DEFAULT_BASE_DATABASE)
    refresh.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    refresh.add_argument("--overlay-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "overlay")
    refresh.add_argument("--universe", type=Path, default=DEFAULT_OUTPUT_ROOT / "v15_fmp_connected_strict_etf_universe.jsonl")
    refresh.add_argument("--from", dest="start_date", default=DEFAULT_FROM)
    refresh.add_argument("--to", dest="end_date", default=DEFAULT_TO)
    refresh.add_argument("--secrets-file", type=Path, default=DEFAULT_SECRETS_PATH)
    refresh.add_argument("--timeout", type=float, default=120.0)
    refresh.add_argument("--retries", type=int, default=3)
    refresh.add_argument("--fail-fast", action="store_false", dest="continue_on_error")
    refresh.set_defaults(continue_on_error=True)
    bulk = commands.add_parser("bulk", help="capture FMP Ultimate current ETF holdings in parts")
    bulk.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    bulk.add_argument("--overlay-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "overlay")
    bulk.add_argument("--as-of-date", default="2026-08-28")
    bulk.add_argument("--start-part", type=int, default=1)
    bulk.add_argument("--max-parts", type=int, default=100)
    bulk.add_argument("--allow-partial", action="store_true")
    bulk.add_argument("--secrets-file", type=Path, default=DEFAULT_SECRETS_PATH)
    bulk.add_argument("--timeout", type=float, default=300.0)
    bulk.add_argument("--retries", type=int, default=3)
    combine = commands.add_parser("combine", help="merge historical constituent overlay into a read-only Oracle copy")
    combine.add_argument("--oracle-incremental", type=Path, default=DEFAULT_ORACLE_INCREMENTAL)
    combine.add_argument(
        "--overlay-database",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "overlay" / "normalized" / "daily_observations.sqlite3",
    )
    combine.add_argument("--output-database", type=Path, default=DEFAULT_COMBINED_INCREMENTAL)
    combine.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    combine.add_argument("--replace", action="store_true")
    compare = commands.add_parser(
        "compare-graphs", help="compare original and repaired test graph snapshots"
    )
    compare.add_argument("--original-graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    compare.add_argument("--repaired-graph-root", type=Path, default=DEFAULT_REPAIRED_GRAPH_ROOT)
    compare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    compare.add_argument("--test-dates", nargs="*")
    current = commands.add_parser(
        "audit-current-bulk", help="compare current FMP bulk topology with the last test graph"
    )
    current.add_argument(
        "--bulk-database",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "overlay" / "normalized" / "daily_observations.sqlite3",
    )
    current.add_argument(
        "--universe",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "v15_fmp_connected_strict_etf_universe.jsonl",
    )
    current.add_argument("--graph-root", type=Path, default=DEFAULT_REPAIRED_GRAPH_ROOT)
    current.add_argument("--signal-date", default=TEST_DATES[-1])
    current.add_argument("--as-of-date", default="2026-08-28")
    current.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return run_audit(args)
    if args.command == "refresh":
        return run_refresh(args)
    if args.command == "bulk":
        return run_bulk(args)
    if args.command == "combine":
        return run_combine(args)
    if args.command == "compare-graphs":
        return run_compare_graphs(args)
    if args.command == "audit-current-bulk":
        return run_current_bulk_audit(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
