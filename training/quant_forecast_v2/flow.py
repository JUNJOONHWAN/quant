"""Massive flow cache and benchmark/full-ETF flow transformations."""

from __future__ import annotations

import bisect
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import TimingRow
from .source import SourceBundle


FLOW_CACHE_SCHEMA = "quant.forecast_v2.flow_cache.v1"


def _available_session(
    sessions: Sequence[str], effective_date: str, processed_date: str
) -> str | None:
    effective_position = bisect.bisect_right(sessions, effective_date) + 1
    processed_position = bisect.bisect_right(sessions, processed_date)
    if effective_position >= len(sessions) or processed_position >= len(sessions):
        return None
    return max(sessions[effective_position], sessions[processed_position])


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_flow_cache(
    source: SourceBundle,
    sessions: Sequence[str],
    output_path: Path,
    *,
    replace: bool = False,
) -> dict:
    """Materialize only normalized fields and index by effective date."""

    output_path = Path(output_path)
    if output_path.exists() and not replace:
        with sqlite3.connect(f"file:{output_path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            count = connection.execute("SELECT COUNT(*) FROM flow").fetchone()[0]
        if row and row[0] == FLOW_CACHE_SCHEMA:
            return {"status": "reused", "path": str(output_path), "rows": int(count)}
        raise ValueError(f"existing flow cache has a different contract: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".building", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE flow(
              ticker TEXT NOT NULL,
              effective_date TEXT NOT NULL,
              processed_date TEXT NOT NULL,
              available_session TEXT NOT NULL,
              flow_rate_pct REAL NOT NULL,
              fund_flow REAL NOT NULL,
              nav REAL NOT NULL,
              shares_outstanding REAL NOT NULL,
              PRIMARY KEY(ticker,effective_date)
            ) WITHOUT ROWID;
            """
        )
        connection.execute(
            "INSERT INTO metadata VALUES('schema_version',?)", (FLOW_CACHE_SCHEMA,)
        )
        batch = []
        input_rows = 0
        excluded = 0
        for ticker, effective, processed, fund_flow, nav, shares, _ in source.iter_flow_rows():
            input_rows += 1
            available = _available_session(sessions, effective, processed)
            fund = _number(fund_flow)
            nav_value = _number(nav)
            share_value = _number(shares)
            assets = (
                nav_value * share_value
                if nav_value and nav_value > 0 and share_value and share_value > 0
                else None
            )
            rate = fund / assets * 100.0 if fund is not None and assets else None
            if (
                available is None
                or fund is None
                or nav_value is None
                or share_value is None
                or rate is None
                or abs(rate) > 100.0
            ):
                excluded += 1
                continue
            batch.append(
                (
                    ticker,
                    effective,
                    processed,
                    available,
                    rate,
                    fund,
                    nav_value,
                    share_value,
                )
            )
            if len(batch) >= 20_000:
                connection.executemany(
                    "INSERT OR REPLACE INTO flow VALUES(?,?,?,?,?,?,?,?)", batch
                )
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT OR REPLACE INTO flow VALUES(?,?,?,?,?,?,?,?)", batch
            )
        connection.execute("CREATE INDEX flow_effective_idx ON flow(effective_date)")
        connection.execute("CREATE INDEX flow_available_idx ON flow(available_session)")
        connection.commit()
        count = int(connection.execute("SELECT COUNT(*) FROM flow").fetchone()[0])
    finally:
        connection.close()
    os.replace(temporary, output_path)
    return {
        "status": "built",
        "path": str(output_path),
        "input_rows": input_rows,
        "rows": count,
        "excluded_rows": excluded,
    }


class FlowCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        self.connection.close()

    def for_date(self, effective_date: str, available_by: str) -> dict[str, float]:
        return {
            str(row[0]): float(row[1])
            for row in self.connection.execute(
                "SELECT ticker,flow_rate_pct FROM flow "
                "WHERE effective_date=? AND available_session<=?",
                (effective_date, available_by),
            )
        }

    def ticker_history(self, ticker: str) -> pd.DataFrame:
        rows = list(
            self.connection.execute(
                "SELECT effective_date,available_session,flow_rate_pct FROM flow "
                "WHERE ticker=? ORDER BY effective_date",
                (ticker,),
            )
        )
        if not rows:
            return pd.DataFrame(columns=["available_session", "flow_rate_pct"])
        return pd.DataFrame.from_records(
            rows, columns=["effective_date", "available_session", "flow_rate_pct"]
        ).set_index("effective_date")


def benchmark_flow_features(
    cache: FlowCache,
    timing_rows: Sequence[TimingRow],
    sessions: Sequence[str],
    *,
    suffix: str = "",
) -> dict[str, dict[str, float]]:
    histories = {ticker: cache.ticker_history(ticker) for ticker in ("SPY", "QQQ")}
    result: dict[str, dict[str, float]] = {}
    session_positions = {value: index for index, value in enumerate(sessions)}
    for timing in timing_rows:
        values: dict[str, float] = {}
        flow_position = session_positions[timing.flow_date]
        dates_60 = sessions[max(0, flow_position - 59) : flow_position + 1]
        for ticker in ("SPY", "QQQ"):
            history = histories[ticker]
            visible = history.reindex(dates_60)
            # Flow is the one source whose live cutoff is signal session T:
            # provider D=T-2 is collected before the T open.  Prices,
            # fundamentals, and constituent disclosures remain cut at T-1.
            available = visible["available_session"].fillna("9999-12-31")
            rates = visible["flow_rate_pct"].where(available <= timing.signal_date)
            current = rates.iloc[-1] if len(rates) else np.nan
            lag = "t3" if suffix else "t2"
            values[f"{ticker.lower()}_flow_rate_{lag}"] = float(current)
            values[f"{ticker.lower()}_flow_rate_5d{suffix}"] = float(
                rates.tail(5).sum(min_count=3)
            )
            values[f"{ticker.lower()}_flow_rate_20d{suffix}"] = float(
                rates.tail(20).sum(min_count=10)
            )
            window = rates.dropna()
            if len(window) >= 20 and window.std(ddof=1) > 0:
                values[f"{ticker.lower()}_flow_z60{suffix}"] = float(
                    (current - window.mean()) / window.std(ddof=1)
                )
            else:
                values[f"{ticker.lower()}_flow_z60{suffix}"] = math.nan
        lag = "t3" if suffix else "t2"
        values[f"qqq_minus_spy_flow_{lag}"] = (
            values[f"qqq_flow_rate_{lag}"] - values[f"spy_flow_rate_{lag}"]
        )
        result[timing.price_date] = values
    return result


def aggregate_symbol_flow(
    exposures: Mapping[str, float], flows: Mapping[str, float]
) -> dict[str, float]:
    total_count = len(exposures)
    total_weight = sum(weight for weight in exposures.values() if weight > 0)
    contributions = []
    observed_weight = 0.0
    positive_count = 0
    negative_count = 0
    for etf, weight in exposures.items():
        rate = flows.get(etf)
        if rate is None or not math.isfinite(rate) or weight <= 0:
            continue
        contribution = rate * weight / 100.0
        contributions.append(contribution)
        observed_weight += weight
        positive_count += int(contribution > 0)
        negative_count += int(contribution < 0)
    gross = sum(abs(value) for value in contributions)
    positive = sum(value for value in contributions if value > 0)
    negative = sum(value for value in contributions if value < 0)
    ordered = sorted((abs(value) for value in contributions), reverse=True)
    shares = [value / gross for value in ordered] if gross > 0 else []
    observed = len(contributions)
    return {
        "all_etf_exposure_count": float(total_count),
        "all_etf_flow_observed_count": float(observed),
        "all_etf_flow_count_coverage": observed / total_count if total_count else math.nan,
        "all_etf_holding_weight_sum": total_weight,
        "all_etf_observed_weight_sum": observed_weight,
        "all_etf_flow_weight_coverage": observed_weight / total_weight
        if total_weight > 0
        else math.nan,
        "all_etf_flow_positive_count": float(positive_count),
        "all_etf_flow_negative_count": float(negative_count),
        "all_etf_flow_breadth": (positive_count - negative_count) / observed
        if observed
        else math.nan,
        "all_etf_flow_net": sum(contributions) if contributions else math.nan,
        "all_etf_flow_positive": positive if contributions else math.nan,
        "all_etf_flow_negative": negative if contributions else math.nan,
        "all_etf_flow_gross": gross if contributions else math.nan,
        "all_etf_flow_max_abs_contribution": ordered[0] if ordered else math.nan,
        "all_etf_flow_top3_abs_share": sum(shares[:3]) if shares else math.nan,
        "all_etf_flow_hhi": sum(value * value for value in shares) if shares else math.nan,
    }
