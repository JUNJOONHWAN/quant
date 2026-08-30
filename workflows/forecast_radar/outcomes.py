"""Immutable 477-stock forward outcome ledger for Forecast RADAR.

The ledger records each morning forecast before any outcome is known, then
resolves the 5- and 20-session terminal direction, upside excursion, and loss
excursion when the corresponding U.S. trading-session horizon matures.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training.quant_forecast_v2.features import price_frame
from training.quant_forecast_v2.source import SourceBundle

from .contracts import (
    COVERAGE_VALIDATED_CORE,
    DEFAULT_BASE_DATABASE,
    DEFAULT_EVALUATION_477_DATABASE,
    DEFAULT_EVALUATION_477_ROOT,
    DEFAULT_INCREMENTAL_DATABASE,
    DEFAULT_LIVE_ROOT,
    EVALUATION_477_COHORT_COUNT,
    EVALUATION_477_COHORT_SHA256,
    EVALUATION_477_COHORT_SOURCE_DATABASE,
)
from .io import sha256_file, utc_now, write_json_atomic


EVALUATION_SCHEMA_VERSION = "quant.forecast_radar.outcome_ledger.v1"
EVALUATION_COHORT_ID = "FORECAST_RADAR_VALIDATED_CORE_477_20260829"
HORIZONS = (5, 20)
VALIDATION_STATUS = "HISTORICAL_OOS_CORE"
PREDICTION_COLUMNS = (
    "symbol",
    "sector",
    "industry",
    "reference_close",
    "coverage_tier",
    "validation_status",
    "p_up_5d",
    "p_up_20d",
    "return_5d_pct",
    "upside_5d_pct",
    "loss_5d_pct",
    "benchmark_excess_return_5d_pct",
    "benchmark_upside_capture_5d_pct",
    "benchmark_downside_defense_5d_pct",
    "return_20d_pct",
    "upside_20d_pct",
    "loss_20d_pct",
    "benchmark_excess_return_20d_pct",
    "benchmark_upside_capture_20d_pct",
    "benchmark_downside_defense_20d_pct",
    "asymmetry_5d",
    "asymmetry_20d",
    "utility_5d",
    "utility_20d",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _symbols_sha256(symbols: Sequence[str]) -> str:
    payload = "".join(f"{symbol}\n" for symbol in sorted(symbols)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _direction(value: object, tolerance: float = 1e-12) -> str:
    number = _finite(value)
    if number is None or abs(number) <= tolerance:
        return "FLAT"
    return "UP" if number > 0 else "DOWN"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS cohort_members(
          cohort_id TEXT NOT NULL,
          symbol TEXT NOT NULL,
          source_sector TEXT,
          source_industry TEXT,
          PRIMARY KEY(cohort_id,symbol)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS ingestion_runs(
          run_id TEXT PRIMARY KEY,
          signal_date TEXT NOT NULL UNIQUE,
          price_date TEXT NOT NULL,
          flow_date TEXT NOT NULL,
          generated_at_utc TEXT NOT NULL,
          source_summary_path TEXT NOT NULL,
          source_database_path TEXT NOT NULL,
          source_database_sha256 TEXT NOT NULL,
          forecast_count INTEGER NOT NULL,
          ingested_at_utc TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS forecast_signals(
          run_id TEXT NOT NULL,
          signal_date TEXT NOT NULL,
          price_date TEXT NOT NULL,
          flow_date TEXT NOT NULL,
          symbol TEXT NOT NULL,
          sector TEXT,
          industry TEXT,
          benchmark TEXT NOT NULL,
          prediction_reference_close REAL NOT NULL,
          predicted_return_5d_pct REAL NOT NULL,
          predicted_upside_5d_pct REAL NOT NULL,
          predicted_loss_5d_pct REAL NOT NULL,
          predicted_return_20d_pct REAL NOT NULL,
          predicted_upside_20d_pct REAL NOT NULL,
          predicted_loss_20d_pct REAL NOT NULL,
          prediction_json TEXT NOT NULL,
          prediction_sha256 TEXT NOT NULL,
          recorded_at_utc TEXT NOT NULL,
          PRIMARY KEY(run_id,symbol),
          FOREIGN KEY(run_id) REFERENCES ingestion_runs(run_id)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS horizon_outcomes(
          run_id TEXT NOT NULL,
          symbol TEXT NOT NULL,
          horizon_sessions INTEGER NOT NULL,
          status TEXT NOT NULL,
          pending_reason TEXT,
          maturity_date TEXT,
          resolution_basis_reference_close REAL,
          prediction_reference_close_revision_pct REAL,
          actual_return_pct REAL,
          actual_upside_pct REAL,
          actual_loss_pct REAL,
          benchmark_return_pct REAL,
          benchmark_upside_pct REAL,
          benchmark_loss_pct REAL,
          actual_benchmark_excess_return_pct REAL,
          actual_benchmark_upside_capture_pct REAL,
          actual_benchmark_downside_defense_pct REAL,
          predicted_direction TEXT NOT NULL,
          actual_direction TEXT,
          direction_correct INTEGER,
          return_abs_error_pct_points REAL,
          upside_abs_error_pct_points REAL,
          loss_abs_error_pct_points REAL,
          benchmark_excess_return_abs_error_pct_points REAL,
          benchmark_upside_capture_abs_error_pct_points REAL,
          benchmark_downside_defense_abs_error_pct_points REAL,
          resolved_at_utc TEXT,
          PRIMARY KEY(run_id,symbol,horizon_sessions),
          FOREIGN KEY(run_id,symbol) REFERENCES forecast_signals(run_id,symbol),
          CHECK(horizon_sessions IN (5,20)),
          CHECK(status IN ('PENDING','RESOLVED'))
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS forecast_signal_date_idx
          ON forecast_signals(signal_date,symbol);
        CREATE INDEX IF NOT EXISTS outcome_status_idx
          ON horizon_outcomes(status,horizon_sessions);

        CREATE VIEW IF NOT EXISTS accuracy_by_signal_date AS
        SELECT f.signal_date,o.horizon_sessions,COUNT(*) resolved_count,
               AVG(o.direction_correct) direction_accuracy,
               AVG(o.return_abs_error_pct_points) return_mae_pct_points,
               AVG(o.upside_abs_error_pct_points) upside_mae_pct_points,
               AVG(o.loss_abs_error_pct_points) loss_mae_pct_points,
               AVG(o.benchmark_excess_return_abs_error_pct_points)
                 benchmark_excess_return_mae_pct_points,
               AVG(o.benchmark_upside_capture_abs_error_pct_points)
                 benchmark_upside_capture_mae_pct_points,
               AVG(o.benchmark_downside_defense_abs_error_pct_points)
                 benchmark_downside_defense_mae_pct_points
        FROM horizon_outcomes o
        JOIN forecast_signals f USING(run_id,symbol)
        WHERE o.status='RESOLVED'
        GROUP BY f.signal_date,o.horizon_sessions;

        CREATE VIEW IF NOT EXISTS accuracy_by_symbol AS
        SELECT f.symbol,o.horizon_sessions,COUNT(*) resolved_count,
               AVG(o.direction_correct) direction_accuracy,
               AVG(o.return_abs_error_pct_points) return_mae_pct_points,
               AVG(o.upside_abs_error_pct_points) upside_mae_pct_points,
               AVG(o.loss_abs_error_pct_points) loss_mae_pct_points,
               AVG(o.benchmark_excess_return_abs_error_pct_points)
                 benchmark_excess_return_mae_pct_points,
               AVG(o.benchmark_upside_capture_abs_error_pct_points)
                 benchmark_upside_capture_mae_pct_points,
               AVG(o.benchmark_downside_defense_abs_error_pct_points)
                 benchmark_downside_defense_mae_pct_points
        FROM horizon_outcomes o
        JOIN forecast_signals f USING(run_id,symbol)
        WHERE o.status='RESOLVED'
        GROUP BY f.symbol,o.horizon_sessions;

        CREATE VIEW IF NOT EXISTS accuracy_by_sector AS
        SELECT f.sector,o.horizon_sessions,COUNT(*) resolved_count,
               AVG(o.direction_correct) direction_accuracy,
               AVG(o.return_abs_error_pct_points) return_mae_pct_points,
               AVG(o.upside_abs_error_pct_points) upside_mae_pct_points,
               AVG(o.loss_abs_error_pct_points) loss_mae_pct_points,
               AVG(o.benchmark_excess_return_abs_error_pct_points)
                 benchmark_excess_return_mae_pct_points,
               AVG(o.benchmark_upside_capture_abs_error_pct_points)
                 benchmark_upside_capture_mae_pct_points,
               AVG(o.benchmark_downside_defense_abs_error_pct_points)
                 benchmark_downside_defense_mae_pct_points
        FROM horizon_outcomes o
        JOIN forecast_signals f USING(run_id,symbol)
        WHERE o.status='RESOLVED'
        GROUP BY f.sector,o.horizon_sessions;
        """
    )


def _load_cohort(
    source_database: Path,
    expected_count: int,
    expected_sha256: str,
) -> list[dict[str, Any]]:
    with _read_only(source_database) as source:
        rows = source.execute(
            """
            SELECT symbol,sector,industry
            FROM stock_forecasts
            WHERE coverage_tier=? AND validation_status=?
            ORDER BY symbol
            """,
            (COVERAGE_VALIDATED_CORE, VALIDATION_STATUS),
        ).fetchall()
    symbols = [str(row["symbol"]) for row in rows]
    digest = _symbols_sha256(symbols)
    if len(rows) != expected_count or digest != expected_sha256:
        raise RuntimeError(
            "frozen 477 cohort verification failed: "
            f"count={len(rows)} expected={expected_count} "
            f"sha256={digest} expected_sha256={expected_sha256}"
        )
    return [dict(row) for row in rows]


def _initialize_cohort(
    connection: sqlite3.Connection,
    cohort: Sequence[Mapping[str, Any]],
    cohort_source_database: Path,
    cohort_sha256: str,
) -> None:
    existing = [
        str(row[0])
        for row in connection.execute(
            "SELECT symbol FROM cohort_members WHERE cohort_id=? ORDER BY symbol",
            (EVALUATION_COHORT_ID,),
        )
    ]
    expected = [str(row["symbol"]) for row in cohort]
    if existing and existing != expected:
        raise RuntimeError("existing evaluation cohort differs from frozen cohort")
    if not existing:
        connection.executemany(
            """
            INSERT INTO cohort_members(cohort_id,symbol,source_sector,source_industry)
            VALUES(?,?,?,?)
            """,
            [
                (
                    EVALUATION_COHORT_ID,
                    str(row["symbol"]),
                    row.get("sector"),
                    row.get("industry"),
                )
                for row in cohort
            ],
        )
    metadata = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "cohort_id": EVALUATION_COHORT_ID,
        "cohort_count": str(len(cohort)),
        "cohort_symbols_sha256": cohort_sha256,
        "cohort_source_database": str(cohort_source_database),
        "direction_contract": "SIGN_OF_PREDICTED_TERMINAL_RETURN_NOT_P_UP",
        "timing_contract": (
            "signal T; price/reference close T-1; ETF Flow T-2; "
            "horizon spans the next H SPY trading sessions after price_date"
        ),
    }
    for key, value in metadata.items():
        prior = connection.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        if prior is not None and str(prior[0]) != value:
            raise RuntimeError(f"immutable metadata drift for {key}")
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key,value) VALUES(?,?)", (key, value)
        )


def _candidate_runs(live_root: Path) -> list[dict[str, Any]]:
    by_signal_date: dict[str, dict[str, Any]] = {}
    runs_root = live_root / "runs"
    if not runs_root.is_dir():
        return []
    for summary_path in sorted(runs_root.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("quality_gate") != "PASS_SHADOW_RUN":
            continue
        required = ("run_id", "signal_date", "price_date", "flow_date", "generated_at_utc")
        if any(not summary.get(key) for key in required):
            continue
        summary["_summary_path"] = str(summary_path)
        signal_date = str(summary["signal_date"])
        prior = by_signal_date.get(signal_date)
        # The first completed PASS for a signal date is the immutable morning
        # observation.  Later reruns cannot rewrite the forward record.
        if prior is None or str(summary["generated_at_utc"]) < str(prior["generated_at_utc"]):
            by_signal_date[signal_date] = summary
    return sorted(by_signal_date.values(), key=lambda row: (row["signal_date"], row["generated_at_utc"]))


def _read_run_predictions(
    summary: Mapping[str, Any], cohort_symbols: Sequence[str]
) -> tuple[Path, list[dict[str, Any]], dict[str, str]]:
    artifact = summary.get("artifacts", {}).get("database", {})
    database_path = Path(str(artifact.get("path") or ""))
    expected_sha256 = str(artifact.get("sha256") or "")
    if not database_path.is_file() or not expected_sha256:
        raise RuntimeError(f"run {summary['run_id']} has no verified forecast database")
    actual_sha256 = sha256_file(database_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"run {summary['run_id']} database SHA mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    placeholders = ",".join("?" for _ in cohort_symbols)
    with _read_only(database_path) as source:
        rows = source.execute(
            f"SELECT {','.join(PREDICTION_COLUMNS)} FROM stock_forecasts "
            f"WHERE symbol IN ({placeholders}) ORDER BY symbol",
            tuple(cohort_symbols),
        ).fetchall()
    if len(rows) != len(cohort_symbols):
        found = {str(row["symbol"]) for row in rows}
        missing = sorted(set(cohort_symbols) - found)
        raise RuntimeError(
            f"run {summary['run_id']} lacks frozen cohort coverage: "
            f"found={len(rows)} expected={len(cohort_symbols)} missing={missing[:20]}"
        )
    for row in rows:
        if (
            row["coverage_tier"] != COVERAGE_VALIDATED_CORE
            or row["validation_status"] != VALIDATION_STATUS
        ):
            raise RuntimeError(
                f"run {summary['run_id']} changed validation status for {row['symbol']}"
            )

    panel_path = Path(str(summary["_summary_path"])).parent / "panel" / "panel.sqlite3"
    if not panel_path.is_file():
        raise RuntimeError(f"run {summary['run_id']} has no retained panel database")
    with _read_only(panel_path) as panel:
        benchmark_rows = panel.execute(
            f"SELECT symbol,benchmark FROM panel WHERE signal_date=? "
            f"AND symbol IN ({placeholders}) ORDER BY symbol",
            (summary["signal_date"], *cohort_symbols),
        ).fetchall()
    benchmarks = {str(row["symbol"]): str(row["benchmark"]) for row in benchmark_rows}
    if len(benchmarks) != len(cohort_symbols):
        missing = sorted(set(cohort_symbols) - set(benchmarks))
        raise RuntimeError(
            f"run {summary['run_id']} lacks benchmark provenance: {missing[:20]}"
        )
    return database_path, [dict(row) for row in rows], benchmarks


def _ingest_runs(
    connection: sqlite3.Connection,
    live_root: Path,
    cohort_symbols: Sequence[str],
) -> dict[str, int]:
    counters = {"candidate_runs": 0, "new_runs": 0, "existing_runs": 0}
    for summary in _candidate_runs(live_root):
        counters["candidate_runs"] += 1
        run_id = str(summary["run_id"])
        existing = connection.execute(
            "SELECT run_id FROM ingestion_runs WHERE signal_date=?",
            (summary["signal_date"],),
        ).fetchone()
        if existing is not None:
            counters["existing_runs"] += 1
            continue
        database_path, rows, benchmarks = _read_run_predictions(summary, cohort_symbols)
        now = utc_now()
        connection.execute(
            """
            INSERT INTO ingestion_runs(
              run_id,signal_date,price_date,flow_date,generated_at_utc,
              source_summary_path,source_database_path,source_database_sha256,
              forecast_count,ingested_at_utc
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                summary["signal_date"],
                summary["price_date"],
                summary["flow_date"],
                summary["generated_at_utc"],
                summary["_summary_path"],
                str(database_path),
                summary["artifacts"]["database"]["sha256"],
                len(rows),
                now,
            ),
        )
        for row in rows:
            payload = {key: row.get(key) for key in PREDICTION_COLUMNS}
            payload_json = _canonical_json(payload)
            payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            symbol = str(row["symbol"])
            connection.execute(
                """
                INSERT INTO forecast_signals(
                  run_id,signal_date,price_date,flow_date,symbol,sector,industry,
                  benchmark,prediction_reference_close,
                  predicted_return_5d_pct,predicted_upside_5d_pct,predicted_loss_5d_pct,
                  predicted_return_20d_pct,predicted_upside_20d_pct,predicted_loss_20d_pct,
                  prediction_json,prediction_sha256,recorded_at_utc
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    summary["signal_date"],
                    summary["price_date"],
                    summary["flow_date"],
                    symbol,
                    row.get("sector"),
                    row.get("industry"),
                    benchmarks[symbol],
                    row["reference_close"],
                    row["return_5d_pct"],
                    row["upside_5d_pct"],
                    row["loss_5d_pct"],
                    row["return_20d_pct"],
                    row["upside_20d_pct"],
                    row["loss_20d_pct"],
                    payload_json,
                    payload_sha256,
                    now,
                ),
            )
            for horizon in HORIZONS:
                connection.execute(
                    """
                    INSERT INTO horizon_outcomes(
                      run_id,symbol,horizon_sessions,status,pending_reason,
                      predicted_direction
                    ) VALUES(?,?,?,'PENDING','HORIZON_NOT_MATURE',?)
                    """,
                    (
                        run_id,
                        symbol,
                        horizon,
                        _direction(row[f"return_{horizon}d_pct"]),
                    ),
                )
        counters["new_runs"] += 1
    return counters


def _frame(
    source: SourceBundle,
    sessions: Sequence[str],
    symbol: str,
    cache: dict[str, Any],
):
    if symbol not in cache:
        cache[symbol] = price_frame(source.price_rows(symbol), sessions)
    return cache[symbol]


def _realized_metrics(frame: Any, price_date: str, future_dates: Sequence[str]) -> dict[str, float] | None:
    try:
        reference = float(frame.at[price_date, "close"])
        future_close = float(frame.at[future_dates[-1], "close"])
        future_high = float(frame.loc[list(future_dates), "high"].max(skipna=False))
        future_low = float(frame.loc[list(future_dates), "low"].min(skipna=False))
    except (KeyError, TypeError, ValueError):
        return None
    values = (reference, future_close, future_high, future_low)
    if not all(math.isfinite(value) and value > 0 for value in values):
        return None
    return {
        "reference_close": reference,
        "return_pct": (future_close / reference - 1.0) * 100.0,
        "upside_pct": max(future_high / reference - 1.0, 0.0) * 100.0,
        "loss_pct": max(1.0 - future_low / reference, 0.0) * 100.0,
    }


def _resolve_outcomes(
    connection: sqlite3.Connection,
    base_database: Path,
    incremental_database: Path | None,
) -> dict[str, int]:
    counters = {"pending_checked": 0, "resolved_now": 0, "price_gaps": 0}
    pending = connection.execute(
        """
        SELECT o.run_id,o.symbol,o.horizon_sessions,f.price_date,f.benchmark,
               f.prediction_reference_close,
               CASE o.horizon_sessions
                 WHEN 5 THEN f.predicted_return_5d_pct
                 ELSE f.predicted_return_20d_pct END predicted_return_pct,
               CASE o.horizon_sessions
                 WHEN 5 THEN f.predicted_upside_5d_pct
                 ELSE f.predicted_upside_20d_pct END predicted_upside_pct,
               CASE o.horizon_sessions
                 WHEN 5 THEN f.predicted_loss_5d_pct
                 ELSE f.predicted_loss_20d_pct END predicted_loss_pct,
               json_extract(f.prediction_json,
                 CASE o.horizon_sessions WHEN 5
                   THEN '$.benchmark_excess_return_5d_pct'
                   ELSE '$.benchmark_excess_return_20d_pct' END)
                 predicted_benchmark_excess_return_pct,
               json_extract(f.prediction_json,
                 CASE o.horizon_sessions WHEN 5
                   THEN '$.benchmark_upside_capture_5d_pct'
                   ELSE '$.benchmark_upside_capture_20d_pct' END)
                 predicted_benchmark_upside_capture_pct,
               json_extract(f.prediction_json,
                 CASE o.horizon_sessions WHEN 5
                   THEN '$.benchmark_downside_defense_5d_pct'
                   ELSE '$.benchmark_downside_defense_20d_pct' END)
                 predicted_benchmark_downside_defense_pct
        FROM horizon_outcomes o
        JOIN forecast_signals f USING(run_id,symbol)
        WHERE o.status='PENDING'
        ORDER BY f.price_date,o.horizon_sessions,o.symbol
        """
    ).fetchall()
    if not pending:
        return counters
    with SourceBundle(base_database, incremental_database) as source:
        sessions = source.sessions()
        session_positions = {session: position for position, session in enumerate(sessions)}
        cache: dict[str, Any] = {}
        for row in pending:
            counters["pending_checked"] += 1
            position = session_positions.get(str(row["price_date"]))
            horizon = int(row["horizon_sessions"])
            if position is None:
                connection.execute(
                    """
                    UPDATE horizon_outcomes SET pending_reason='PRICE_DATE_NOT_IN_SPY_CALENDAR'
                    WHERE run_id=? AND symbol=? AND horizon_sessions=?
                    """,
                    (row["run_id"], row["symbol"], horizon),
                )
                counters["price_gaps"] += 1
                continue
            maturity_position = position + horizon
            if maturity_position >= len(sessions):
                connection.execute(
                    """
                    UPDATE horizon_outcomes SET pending_reason='HORIZON_NOT_MATURE'
                    WHERE run_id=? AND symbol=? AND horizon_sessions=?
                    """,
                    (row["run_id"], row["symbol"], horizon),
                )
                continue
            future_dates = sessions[position + 1 : maturity_position + 1]
            stock = _realized_metrics(
                _frame(source, sessions, str(row["symbol"]), cache),
                str(row["price_date"]),
                future_dates,
            )
            benchmark = _realized_metrics(
                _frame(source, sessions, str(row["benchmark"]), cache),
                str(row["price_date"]),
                future_dates,
            )
            if stock is None or benchmark is None:
                connection.execute(
                    """
                    UPDATE horizon_outcomes SET pending_reason='ADJUSTED_OHLC_OR_BENCHMARK_GAP',
                      maturity_date=?
                    WHERE run_id=? AND symbol=? AND horizon_sessions=?
                    """,
                    (sessions[maturity_position], row["run_id"], row["symbol"], horizon),
                )
                counters["price_gaps"] += 1
                continue
            prediction_reference = float(row["prediction_reference_close"])
            revision_pct = (
                (stock["reference_close"] / prediction_reference - 1.0) * 100.0
                if prediction_reference > 0
                else None
            )
            actual_direction = _direction(stock["return_pct"])
            predicted_direction = _direction(row["predicted_return_pct"])
            actual_benchmark_excess_return = (
                stock["return_pct"] - benchmark["return_pct"]
            )
            actual_benchmark_upside_capture = (
                stock["upside_pct"] - benchmark["upside_pct"]
            )
            actual_benchmark_downside_defense = (
                benchmark["loss_pct"] - stock["loss_pct"]
            )
            connection.execute(
                """
                UPDATE horizon_outcomes SET
                  status='RESOLVED',pending_reason=NULL,maturity_date=?,
                  resolution_basis_reference_close=?,
                  prediction_reference_close_revision_pct=?,
                  actual_return_pct=?,actual_upside_pct=?,actual_loss_pct=?,
                  benchmark_return_pct=?,benchmark_upside_pct=?,benchmark_loss_pct=?,
                  actual_benchmark_excess_return_pct=?,
                  actual_benchmark_upside_capture_pct=?,
                  actual_benchmark_downside_defense_pct=?,
                  actual_direction=?,direction_correct=?,
                  return_abs_error_pct_points=?,upside_abs_error_pct_points=?,
                  loss_abs_error_pct_points=?,
                  benchmark_excess_return_abs_error_pct_points=?,
                  benchmark_upside_capture_abs_error_pct_points=?,
                  benchmark_downside_defense_abs_error_pct_points=?,
                  resolved_at_utc=?
                WHERE run_id=? AND symbol=? AND horizon_sessions=?
                """,
                (
                    sessions[maturity_position],
                    stock["reference_close"],
                    revision_pct,
                    stock["return_pct"],
                    stock["upside_pct"],
                    stock["loss_pct"],
                    benchmark["return_pct"],
                    benchmark["upside_pct"],
                    benchmark["loss_pct"],
                    actual_benchmark_excess_return,
                    actual_benchmark_upside_capture,
                    actual_benchmark_downside_defense,
                    actual_direction,
                    int(predicted_direction == actual_direction),
                    abs(stock["return_pct"] - float(row["predicted_return_pct"])),
                    abs(stock["upside_pct"] - float(row["predicted_upside_pct"])),
                    abs(stock["loss_pct"] - float(row["predicted_loss_pct"])),
                    abs(
                        actual_benchmark_excess_return
                        - float(row["predicted_benchmark_excess_return_pct"])
                    ),
                    abs(
                        actual_benchmark_upside_capture
                        - float(row["predicted_benchmark_upside_capture_pct"])
                    ),
                    abs(
                        actual_benchmark_downside_defense
                        - float(row["predicted_benchmark_downside_defense_pct"])
                    ),
                    utc_now(),
                    row["run_id"],
                    row["symbol"],
                    horizon,
                ),
            )
            counters["resolved_now"] += 1
    return counters


def _metric_summary(rows: Iterable[sqlite3.Row]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        return {
            "resolved_count": 0,
            "direction_accuracy": None,
            "return_mae_pct_points": None,
            "upside_mae_pct_points": None,
            "loss_mae_pct_points": None,
            "benchmark_excess_return_mae_pct_points": None,
            "benchmark_upside_capture_mae_pct_points": None,
            "benchmark_downside_defense_mae_pct_points": None,
        }
    return {
        "resolved_count": len(values),
        "direction_accuracy": sum(int(row["direction_correct"]) for row in values) / len(values),
        "return_mae_pct_points": sum(float(row["return_abs_error_pct_points"]) for row in values) / len(values),
        "upside_mae_pct_points": sum(float(row["upside_abs_error_pct_points"]) for row in values) / len(values),
        "loss_mae_pct_points": sum(float(row["loss_abs_error_pct_points"]) for row in values) / len(values),
        "benchmark_excess_return_mae_pct_points": sum(
            float(row["benchmark_excess_return_abs_error_pct_points"])
            for row in values
        )
        / len(values),
        "benchmark_upside_capture_mae_pct_points": sum(
            float(row["benchmark_upside_capture_abs_error_pct_points"])
            for row in values
        )
        / len(values),
        "benchmark_downside_defense_mae_pct_points": sum(
            float(row["benchmark_downside_defense_abs_error_pct_points"])
            for row in values
        )
        / len(values),
    }


def _build_summary(
    connection: sqlite3.Connection,
    evaluation_database: Path,
    cohort_count: int,
    ingestion: Mapping[str, int],
    resolution: Mapping[str, int],
) -> dict[str, Any]:
    run_count = int(connection.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0])
    signal_count = int(
        connection.execute("SELECT COUNT(DISTINCT signal_date) FROM forecast_signals").fetchone()[0]
    )
    forecast_count = int(connection.execute("SELECT COUNT(*) FROM forecast_signals").fetchone()[0])
    latest_signal = connection.execute("SELECT MAX(signal_date) FROM forecast_signals").fetchone()[0]
    status_counts: dict[str, dict[str, int]] = {}
    overall: dict[str, Any] = {}
    rolling: dict[str, Any] = {}
    for horizon in HORIZONS:
        counts = {
            str(row["status"]): int(row["count"])
            for row in connection.execute(
                "SELECT status,COUNT(*) count FROM horizon_outcomes "
                "WHERE horizon_sessions=? GROUP BY status",
                (horizon,),
            )
        }
        status_counts[str(horizon)] = {
            "resolved": counts.get("RESOLVED", 0),
            "pending": counts.get("PENDING", 0),
        }
        resolved = connection.execute(
            "SELECT * FROM horizon_outcomes WHERE horizon_sessions=? AND status='RESOLVED'",
            (horizon,),
        ).fetchall()
        overall[str(horizon)] = _metric_summary(resolved)
        rolling[str(horizon)] = {}
        signal_dates = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT f.signal_date
                FROM horizon_outcomes o JOIN forecast_signals f USING(run_id,symbol)
                WHERE o.horizon_sessions=? AND o.status='RESOLVED'
                ORDER BY f.signal_date DESC
                """,
                (horizon,),
            )
        ]
        for window in (20, 60, 120):
            selected = signal_dates[:window]
            if selected:
                placeholders = ",".join("?" for _ in selected)
                rows = connection.execute(
                    f"SELECT o.* FROM horizon_outcomes o "
                    f"JOIN forecast_signals f USING(run_id,symbol) "
                    f"WHERE o.horizon_sessions=? AND o.status='RESOLVED' "
                    f"AND f.signal_date IN ({placeholders})",
                    (horizon, *selected),
                ).fetchall()
            else:
                rows = []
            value = _metric_summary(rows)
            value["signal_date_count"] = len(selected)
            rolling[str(horizon)][str(window)] = value
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": "RECORDED",
        "scope": {
            "cohort_id": EVALUATION_COHORT_ID,
            "cohort_count": cohort_count,
            "cohort_symbols_sha256": _symbols_sha256(
                [row[0] for row in connection.execute(
                    "SELECT symbol FROM cohort_members WHERE cohort_id=? ORDER BY symbol",
                    (EVALUATION_COHORT_ID,),
                )]
            ),
            "not_full_live_universe": True,
        },
        "contracts": {
            "direction": "SIGN_OF_PREDICTED_TERMINAL_RETURN_NOT_P_UP",
            "horizons_in_spy_trading_sessions": list(HORIZONS),
            "actual_upside": "MAX_SPLIT_ADJUSTED_HIGH_OVER_NEXT_H_SESSIONS_VS_PRICE_DATE_CLOSE",
            "actual_loss": "PRICE_DATE_CLOSE_MINUS_MIN_SPLIT_ADJUSTED_LOW_OVER_NEXT_H_SESSIONS",
            "canonical_run_per_signal_date": "FIRST_COMPLETED_PASS_SHADOW_RUN",
            "trade_policy": "NONE_INFORMATION_EVALUATION_ONLY",
        },
        "database": str(evaluation_database),
        "ingestion_this_call": dict(ingestion),
        "resolution_this_call": dict(resolution),
        "totals": {
            "run_count": run_count,
            "signal_date_count": signal_count,
            "forecast_count": forecast_count,
            "expected_forecast_count": signal_count * cohort_count,
            "latest_signal_date": latest_signal,
            "outcomes_by_horizon": status_counts,
        },
        "overall_by_horizon": overall,
        "rolling_signal_dates_by_horizon": rolling,
    }


def evaluate_outcomes(
    *,
    live_root: Path = DEFAULT_LIVE_ROOT,
    evaluation_root: Path = DEFAULT_EVALUATION_477_ROOT,
    evaluation_database: Path | None = None,
    cohort_source_database: Path = EVALUATION_477_COHORT_SOURCE_DATABASE,
    cohort_count: int = EVALUATION_477_COHORT_COUNT,
    cohort_sha256: str = EVALUATION_477_COHORT_SHA256,
    base_database: Path = DEFAULT_BASE_DATABASE,
    incremental_database: Path | None = DEFAULT_INCREMENTAL_DATABASE,
) -> dict[str, Any]:
    """Ingest immutable daily predictions and resolve all matured outcomes."""

    live_root = Path(live_root)
    evaluation_root = Path(evaluation_root)
    database = Path(evaluation_database or (evaluation_root / DEFAULT_EVALUATION_477_DATABASE.name))
    cohort = _load_cohort(Path(cohort_source_database), cohort_count, cohort_sha256)
    cohort_symbols = [str(row["symbol"]) for row in cohort]
    connection = _connect(database)
    try:
        _create_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        _initialize_cohort(
            connection,
            cohort,
            Path(cohort_source_database),
            cohort_sha256,
        )
        ingestion = _ingest_runs(connection, live_root, cohort_symbols)
        resolution = _resolve_outcomes(
            connection,
            Path(base_database),
            Path(incremental_database) if incremental_database else None,
        )
        summary = _build_summary(
            connection, database, cohort_count, ingestion, resolution
        )
        if summary["totals"]["forecast_count"] != summary["totals"]["expected_forecast_count"]:
            raise RuntimeError("evaluation ledger forecast coverage parity failed")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    summary["database_sha256"] = sha256_file(database)
    write_json_atomic(evaluation_root / "latest.json", summary)
    return {
        "status": summary["status"],
        "cohort_count": cohort_count,
        "latest_signal_date": summary["totals"]["latest_signal_date"],
        "forecast_count": summary["totals"]["forecast_count"],
        "outcomes_by_horizon": summary["totals"]["outcomes_by_horizon"],
        "overall_by_horizon": summary["overall_by_horizon"],
        "database": str(database),
        "database_sha256": summary["database_sha256"],
        "direction_contract": summary["contracts"]["direction"],
    }
