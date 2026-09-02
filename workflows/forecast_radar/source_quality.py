"""Fail-closed Oracle source-data checks for Forecast RADAR."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from quant_dataset.shared_market import SharedMarketBinding


QUALITY_SCHEMA = "quant.forecast_radar.source_quality.v1"
ALLOWED_QUALITY_STATUSES = frozenset({"pass", "warn", "single_source"})
REQUIRED_BENCHMARKS = ("SPY", "QQQ", "IWM", "DIA")
MINIMUM_COVERAGE_RATIO = 0.80
MAXIMUM_BENCHMARK_ABS_RETURN = 0.30


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_positive(value: object) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _finite_nonnegative(value: object) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= 0.0
    except (TypeError, ValueError):
        return False


def validate_forecast_source_quality(binding: SharedMarketBinding) -> dict[str, Any]:
    """Validate the sealed target bars before a Forecast RADAR can exist.

    The Oracle completion receipt establishes source authority. This additional
    consumer-side gate verifies that its target-day payload is internally
    coherent and completely represented before a derived report is generated
    or read. Any failed check raises and therefore leaves ``latest.json``
    untouched.
    """

    try:
        status = json.loads(Path(binding.oracle_status_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Forecast source-data quality cannot read Oracle status") from exc
    target = binding.target_as_of_date
    if str(status.get("target_as_of_date") or "") != target:
        raise RuntimeError("Forecast source-data quality target differs from Oracle binding")

    market_gate = _as_mapping(status.get("market_row_gate"))
    expected_rows = _as_mapping(market_gate.get("rows_by_session")).get(target)
    coverage = _as_mapping(status.get("symbol_coverage_gate"))
    coverage_session = _as_mapping(_as_mapping(coverage.get("sessions")).get(target))
    failures: list[str] = []
    try:
        expected_rows = int(expected_rows)
    except (TypeError, ValueError):
        expected_rows = 0
    if expected_rows <= 0:
        failures.append("sealed_target_row_count_missing")
    if not coverage_session:
        failures.append("sealed_coverage_session_missing")

    with sqlite3.connect(
        f"file:{Path(binding.incremental_database)}?mode=ro", uri=True
    ) as connection:
        source_rows = [
            list(row)
            for row in connection.execute(
                """
                SELECT symbol,open,high,low,close,adjusted_close,volume,
                       raw_artifact_id,capture_event_id,source_row_index,
                       source_timestamp_ms
                FROM daily_observations
                WHERE source='fmp' AND trade_date=?
                ORDER BY symbol
                """,
                (target,),
            )
        ]
        quality_rows = [
            list(row)
            for row in connection.execute(
                """
                SELECT symbol,status,sources_json,metrics_json,reasons_json,
                       tolerances_json
                FROM quality_checks
                WHERE trade_date=?
                ORDER BY symbol
                """,
                (target,),
            )
        ]
        duplicate_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT symbol FROM daily_observations
                  WHERE source='fmp' AND trade_date=?
                  GROUP BY symbol HAVING COUNT(*) > 1
                )
                """,
                (target,),
            ).fetchone()[0]
        )
        previous_date_row = connection.execute(
            """
            SELECT MAX(trade_date) FROM daily_observations
            WHERE source='fmp' AND symbol='SPY' AND trade_date<?
            """,
            (target,),
        ).fetchone()
        previous_date = str(previous_date_row[0] or "") if previous_date_row else ""
        previous_row_count = int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT symbol) FROM daily_observations
                WHERE source='fmp' AND trade_date=?
                """,
                (previous_date,),
            ).fetchone()[0]
            or 0
        )
        benchmark_rows = {
            str(row[0]): (float(row[1]), float(row[2]))
            for row in connection.execute(
                """
                SELECT current.symbol,current.close,previous.close
                FROM daily_observations AS current
                JOIN daily_observations AS previous
                  ON previous.source=current.source
                 AND previous.symbol=current.symbol
                 AND previous.trade_date=?
                WHERE current.source='fmp' AND current.trade_date=?
                  AND current.symbol IN ('SPY','QQQ','IWM','DIA')
                ORDER BY current.symbol
                """,
                (previous_date, target),
            )
        }

    invalid_bar_count = 0
    for row in source_rows:
        _symbol, open_, high, low, close, adjusted_close, volume, *_rest = row
        valid_bar = (
            _finite_positive(open_)
            and _finite_positive(high)
            and _finite_positive(low)
            and _finite_positive(close)
            and _finite_nonnegative(volume)
            and float(low) <= min(float(open_), float(close))
            and float(high) >= max(float(open_), float(close))
            and (adjusted_close is None or _finite_positive(adjusted_close))
        )
        if not valid_bar:
            invalid_bar_count += 1
    if len(source_rows) != expected_rows:
        failures.append(f"target_row_count={len(source_rows)} expected={expected_rows}")
    if duplicate_count:
        failures.append(f"duplicate_target_symbols={duplicate_count}")
    if invalid_bar_count:
        failures.append(f"invalid_ohlcv_rows={invalid_bar_count}")

    quality_status_counts = Counter(str(row[1]) for row in quality_rows)
    invalid_quality = sorted(
        item for item in quality_status_counts if item not in ALLOWED_QUALITY_STATUSES
    )
    if len(quality_rows) != len(source_rows):
        failures.append(
            f"quality_row_count={len(quality_rows)} target_rows={len(source_rows)}"
        )
    if invalid_quality:
        failures.append("invalid_quality_statuses=" + ",".join(invalid_quality))

    coverage_status = str(coverage_session.get("status") or "")
    coverage_errors = int(coverage_session.get("error_count") or 0)
    coverage_bar_count = int(coverage_session.get("bar_count") or 0)
    coverage_missing_after = list(coverage_session.get("missing_after") or [])
    coverage_invalid = sum(
        int(coverage_session.get(key) or 0)
        for key in (
            "invalid_before_count",
            "invalid_no_bar_count",
            "quarantined_invalid_bar_count",
        )
    )
    if coverage_status != "complete" or coverage_errors:
        failures.append(f"coverage_status={coverage_status or 'missing'} errors={coverage_errors}")
    if coverage_bar_count != len(source_rows):
        failures.append(
            f"coverage_bar_count={coverage_bar_count} target_rows={len(source_rows)}"
        )
    if coverage_missing_after or coverage_invalid:
        failures.append(
            f"coverage_unresolved={len(coverage_missing_after)} invalid={coverage_invalid}"
        )

    coverage_ratio = (
        float(len(source_rows)) / float(previous_row_count)
        if previous_row_count > 0
        else 0.0
    )
    if previous_row_count <= 0 or coverage_ratio < MINIMUM_COVERAGE_RATIO:
        failures.append(
            f"coverage_ratio={coverage_ratio:.6f} minimum={MINIMUM_COVERAGE_RATIO:.2f}"
        )
    missing_benchmarks = sorted(set(REQUIRED_BENCHMARKS).difference(benchmark_rows))
    if missing_benchmarks:
        failures.append("missing_benchmarks=" + ",".join(missing_benchmarks))
    benchmark_returns = {
        symbol: (close / previous_close) - 1.0
        for symbol, (close, previous_close) in benchmark_rows.items()
        if previous_close > 0.0
    }
    implausible = sorted(
        symbol
        for symbol, daily_return in benchmark_returns.items()
        if abs(daily_return) > MAXIMUM_BENCHMARK_ABS_RETURN
    )
    if implausible:
        failures.append("implausible_benchmark_returns=" + ",".join(implausible))

    evidence = {
        "schema": QUALITY_SCHEMA,
        "target_as_of_date": target,
        "oracle_source_fingerprint_sha256": binding.source_fingerprint_sha256,
        "expected_target_rows": expected_rows,
        "target_rows": source_rows,
        "quality_rows": quality_rows,
        "quality_status_counts": dict(sorted(quality_status_counts.items())),
        "duplicate_target_symbol_count": duplicate_count,
        "invalid_ohlcv_row_count": invalid_bar_count,
        "coverage_session": {
            "status": coverage_status,
            "error_count": coverage_errors,
            "bar_count": coverage_bar_count,
            "missing_after_count": len(coverage_missing_after),
            "invalid_count": coverage_invalid,
        },
        "previous_date": previous_date,
        "previous_row_count": previous_row_count,
        "coverage_ratio": coverage_ratio,
        "benchmark_returns": benchmark_returns,
    }
    data_fingerprint = _canonical_sha256(evidence)
    if failures:
        raise RuntimeError(
            "Forecast RADAR source-data quality gate failed: " + "; ".join(failures)
        )
    return {
        "schema": QUALITY_SCHEMA,
        "status": "PASS",
        "target_as_of_date": target,
        "oracle_source_fingerprint_sha256": binding.source_fingerprint_sha256,
        "data_fingerprint_sha256": data_fingerprint,
        "target_row_count": len(source_rows),
        "quality_row_count": len(quality_rows),
        "quality_status_counts": dict(sorted(quality_status_counts.items())),
        "coverage_ratio": coverage_ratio,
        "benchmark_returns": benchmark_returns,
    }
