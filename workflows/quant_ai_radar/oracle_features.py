"""Point-in-time market features built only from the sealed Oracle store."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from quant_dataset.shared_market import SharedReadOnlyDatabase
from quant_dataset.storage import canonical_json
from workflows.quant_ai_radar.universe import Candidate


FEATURE_SCHEMA = "quant.oracle_market_features.v1"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _return(values: Sequence[float], periods: int) -> float | None:
    if len(values) <= periods or values[-periods - 1] == 0:
        return None
    return (values[-1] / values[-periods - 1] - 1.0) * 100.0


def _robust_z(values: Sequence[float]) -> float | None:
    if len(values) < 5:
        return None
    center = median(values)
    mad = median(abs(value - center) for value in values)
    if mad <= 0:
        return 0.0
    return 0.67448975 * (values[-1] - center) / mad


def _chunks(values: Sequence[str], size: int = 500) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def _price_history(
    database: SharedReadOnlyDatabase,
    symbols: Sequence[str],
    as_of_date: str,
) -> dict[str, list[tuple[str, float, float]]]:
    start = (date.fromisoformat(as_of_date) - timedelta(days=100)).isoformat()
    result: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    with database.connect() as connection:
        for chunk in _chunks(symbols):
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                WITH ranked AS (
                  SELECT symbol,trade_date,close,volume,
                         ROW_NUMBER() OVER (
                           PARTITION BY symbol,trade_date
                           ORDER BY CASE source WHEN 'massive' THEN 0 ELSE 1 END
                         ) AS source_rank
                  FROM daily_observations
                  WHERE symbol IN ({placeholders})
                    AND trade_date BETWEEN ? AND ?
                    AND close>0
                )
                SELECT symbol,trade_date,close,volume FROM ranked
                WHERE source_rank=1 ORDER BY symbol,trade_date
                """,
                (*chunk, start, as_of_date),
            )
            for symbol, trade_date, close, volume in rows:
                result[str(symbol)].append(
                    (str(trade_date), float(close), float(volume or 0.0))
                )
    return result


def _flow_history(
    database: SharedReadOnlyDatabase,
    tickers: Sequence[str],
    as_of_date: str,
) -> dict[str, list[dict[str, Any]]]:
    start = (date.fromisoformat(as_of_date) - timedelta(days=220)).isoformat()
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with database.connect() as connection:
        for chunk in _chunks(tickers):
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT ticker,effective_date,processed_date,available_at_date,
                       fund_flow,assets
                FROM etf_flow_observations
                WHERE ticker IN ({placeholders})
                  AND effective_date BETWEEN ? AND ?
                  AND effective_date<=?
                  AND processed_date<=?
                  AND available_at_date<=?
                ORDER BY ticker,effective_date
                """,
                (*chunk, start, as_of_date, as_of_date, as_of_date, as_of_date),
            )
            for row in rows:
                fund_flow = _finite(row[4])
                assets = _finite(row[5])
                flow_rate = (
                    fund_flow / assets * 100.0
                    if fund_flow is not None and assets not in (None, 0.0)
                    else None
                )
                result[str(row[0])].append(
                    {
                        "effective_date": str(row[1]),
                        "processed_date": str(row[2]),
                        "available_at_date": str(row[3]),
                        "fund_flow": fund_flow,
                        "assets": assets,
                        "flow_to_assets_pct": flow_rate,
                    }
                )
    return result


def _profile_sectors(base_database: Path) -> dict[str, str]:
    sectors: dict[str, str] = {}
    with sqlite3.connect(
        f"file:{base_database}?mode=ro", uri=True
    ) as connection:
        rows = connection.execute(
            """
            SELECT symbol,row_json FROM fmp_training_facts
            WHERE endpoint_id='company_information_company_profile_data'
            """
        )
        for symbol, raw in rows:
            try:
                document = json.loads(str(raw))
            except (TypeError, json.JSONDecodeError):
                continue
            sector = str(document.get("sector") or "").strip()
            if symbol and sector:
                sectors[str(symbol).upper()] = sector
    return sectors


def _visible_constituents(
    database: SharedReadOnlyDatabase,
    etfs: Sequence[str],
    as_of_date: str,
) -> list[dict[str, Any]]:
    if not etfs:
        return []
    result: list[dict[str, Any]] = []
    with database.connect_constituents() as connection:
        for chunk in _chunks(etfs, 100):
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                WITH latest AS (
                  SELECT etf_ticker,MAX(effective_date) AS effective_date
                  FROM etf_constituent_observations
                  WHERE etf_ticker IN ({placeholders})
                    AND effective_date<=? AND available_date<=?
                  GROUP BY etf_ticker
                )
                SELECT o.etf_ticker,o.constituent_ticker,o.weight_percent,
                       o.effective_date,o.available_date
                FROM etf_constituent_observations o
                JOIN latest l
                  ON l.etf_ticker=o.etf_ticker
                 AND l.effective_date=o.effective_date
                WHERE o.available_date<=?
                  AND o.constituent_ticker IS NOT NULL
                """,
                (*chunk, as_of_date, as_of_date, as_of_date),
            )
            result.extend(
                {
                    "etf_ticker": str(row[0]),
                    "constituent_ticker": str(row[1]),
                    "weight_percent": float(row[2] or 0.0),
                    "effective_date": str(row[3]),
                    "available_date": str(row[4]),
                }
                for row in rows
            )
    return result


def _etf_rows(
    etf_symbols: Sequence[str],
    prices: Mapping[str, Sequence[tuple[str, float, float]]],
    flows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for ticker in etf_symbols:
        history = list(prices.get(ticker) or [])
        flow_history = list(flows.get(ticker) or [])
        if not history or not flow_history:
            continue
        closes = [row[1] for row in history]
        flow_rates = [
            float(row["flow_to_assets_pct"])
            for row in flow_history
            if row.get("flow_to_assets_pct") is not None
        ]
        latest = flow_history[-1]
        latest_rate = _finite(latest.get("flow_to_assets_pct"))
        zscore = _robust_z(flow_rates)
        ret_5d = _return(closes, 5)
        ret_21d = _return(closes, 21)
        flow_5d = sum(flow_rates[-5:]) if flow_rates else 0.0
        flow_21d = sum(flow_rates[-21:]) if flow_rates else 0.0
        score = (
            abs(zscore or 0.0) * 20.0
            + min(abs(flow_5d) * 50.0, 50.0)
            + min(abs(ret_21d or 0.0), 25.0)
        )
        flow_sign = 1 if flow_5d > 0 else -1 if flow_5d < 0 else 0
        price_sign = 1 if (ret_5d or 0) > 0 else -1 if (ret_5d or 0) < 0 else 0
        state = (
            "confirmed_accumulation"
            if flow_sign > 0 and price_sign > 0
            else "confirmed_distribution"
            if flow_sign < 0 and price_sign < 0
            else "flow_price_divergence"
        )
        rows.append(
            {
                "ticker": ticker,
                "priority_score": round(score, 6),
                "state": state,
                "latest_effective_date": latest["effective_date"],
                "latest_processed_date": latest["processed_date"],
                "latest_available_at_date": latest["available_at_date"],
                "latest_flow_to_assets_pct": latest_rate,
                "latest_robust_zscore": zscore,
                "flow_5d_to_assets": flow_5d,
                "flow_21d_to_assets": flow_21d,
                "ret_5d": ret_5d,
                "ret_21d": ret_21d,
                "dollar_volume": history[-1][1] * history[-1][2],
            }
        )
    rows.sort(key=lambda row: (-row["priority_score"], row["ticker"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def _stock_rows(
    candidate_stocks: set[str],
    constituents: Sequence[Mapping[str, Any]],
    etf_by_ticker: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    contributions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in constituents:
        stock = str(row["constituent_ticker"])
        if stock not in candidate_stocks:
            continue
        etf = etf_by_ticker.get(str(row["etf_ticker"]))
        if not etf:
            continue
        flow_rate = _finite(etf.get("latest_flow_to_assets_pct")) or 0.0
        weighted = flow_rate * float(row["weight_percent"]) / 100.0
        contributions[stock].append(
            {
                "etf_ticker": row["etf_ticker"],
                "weighted_flow_rate_contribution_pct": weighted,
                "weight_percent": row["weight_percent"],
                "constituent_effective_date": row["effective_date"],
                "constituent_available_date": row["available_date"],
                "flow_training_available_session_date": etf[
                    "latest_available_at_date"
                ],
            }
        )
    result = []
    for symbol, items in contributions.items():
        items.sort(
            key=lambda item: (
                -abs(item["weighted_flow_rate_contribution_pct"]),
                item["etf_ticker"],
            )
        )
        net = sum(
            float(item["weighted_flow_rate_contribution_pct"]) for item in items
        )
        result.append(
            {
                "symbol": symbol,
                "priority_score": abs(net) * 1000.0 + len(items),
                "net_weighted_flow_rate_contribution_pct": net,
                "eligible_etf_count": len(items),
                "top_contributing_etfs": items[:8],
            }
        )
    result.sort(key=lambda row: (-row["priority_score"], row["symbol"]))
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    return result


def _rotation_clusters(
    etf_rows: Sequence[Mapping[str, Any]],
    constituents: Sequence[Mapping[str, Any]],
    sectors: Mapping[str, str],
) -> list[dict[str, Any]]:
    holdings_by_etf: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in constituents:
        holdings_by_etf[str(row["etf_ticker"])].append(row)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    related: dict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for etf in etf_rows:
        ticker = str(etf["ticker"])
        sector_weights: defaultdict[str, float] = defaultdict(float)
        for holding in holdings_by_etf.get(ticker, []):
            stock = str(holding["constituent_ticker"])
            sector = sectors.get(stock, "Unclassified")
            weight = max(float(holding["weight_percent"]), 0.0)
            sector_weights[sector] += weight
            related[sector][stock] += weight
        dominant = (
            max(sector_weights, key=sector_weights.get)
            if sector_weights
            else "Unclassified"
        )
        grouped[dominant].append(etf)
    clusters = []
    for sector, members in grouped.items():
        members = sorted(
            members,
            key=lambda row: (-float(row["priority_score"]), str(row["ticker"])),
        )
        flows_5 = [float(row["flow_5d_to_assets"]) for row in members]
        flows_21 = [float(row["flow_21d_to_assets"]) for row in members]
        returns_5 = [
            float(row["ret_5d"]) for row in members if row.get("ret_5d") is not None
        ]
        returns_21 = [
            float(row["ret_21d"])
            for row in members
            if row.get("ret_21d") is not None
        ]
        score = sum(float(row["priority_score"]) for row in members) / len(members)
        clusters.append(
            {
                "integrated_cluster": sector,
                "integrated_state": (
                    "rotation_in"
                    if sum(flows_5) > 0 and median(returns_5 or [0.0]) > 0
                    else "rotation_out"
                    if sum(flows_5) < 0 and median(returns_5 or [0.0]) < 0
                    else "mixed"
                ),
                "integrated_score": score,
                "breadth_score": (
                    sum(1 for row in members if (row.get("ret_5d") or 0) > 0)
                    / len(members)
                    * 100.0
                ),
                "ticker_count": len(members),
                "top_ticker": members[0]["ticker"],
                "representative_tickers": [
                    row["ticker"] for row in members[:5]
                ],
                "median_fmp_ret_5d": median(returns_5 or [0.0]),
                "median_fmp_ret_21d": median(returns_21 or [0.0]),
                "flow_5d_to_assets": sum(flows_5),
                "flow_21d_to_assets": sum(flows_21),
                "top_related_stocks": [
                    f"{symbol}:{weight:.2f}%"
                    for symbol, weight in sorted(
                        related[sector].items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:8]
                ],
            }
        )
    clusters.sort(
        key=lambda row: (-abs(float(row["integrated_score"])), row["integrated_cluster"])
    )
    for rank, row in enumerate(clusters, 1):
        row["rank"] = rank
    return clusters


def build_oracle_market_features(
    database: SharedReadOnlyDatabase,
    candidates: Sequence[Candidate],
    as_of_date: str,
) -> dict[str, Any]:
    """Build one deterministic, hash-bound, lookahead-safe feature snapshot."""

    if as_of_date != database.binding.target_as_of_date:
        raise ValueError("feature date must match the sealed Oracle snapshot")
    etf_symbols = sorted(
        candidate.symbol
        for candidate in candidates
        if candidate.proxy_task_type == "etf_own_flow_analysis"
    )
    stock_symbols = {
        candidate.symbol
        for candidate in candidates
        if candidate.proxy_task_type == "stock_constituent_flow_analysis"
    }
    prices = _price_history(database, etf_symbols, as_of_date)
    flows = _flow_history(database, etf_symbols, as_of_date)
    etfs = _etf_rows(etf_symbols, prices, flows)
    relation_etfs = [str(row["ticker"]) for row in etfs[:256]]
    constituents = _visible_constituents(database, relation_etfs, as_of_date)
    etf_by_ticker = {str(row["ticker"]): row for row in etfs}
    stocks = _stock_rows(stock_symbols, constituents, etf_by_ticker)
    clusters = _rotation_clusters(
        etfs[:128],
        constituents,
        _profile_sectors(database.binding.base_database),
    )
    binding = {
        "schema_version": "quant.oracle_feature_binding.v1",
        "release_id": f"oracle-{as_of_date}",
        "trade_date_us": as_of_date,
        "source_fingerprint_sha256": (
            database.binding.source_fingerprint_sha256
        ),
        "producer": "market_structure_oracle_single_writer",
        "point_in_time_policy": (
            "effective_date,processed_date,available_at_date,"
            "constituent.available_date <= as_of"
        ),
    }
    payload = {
        "schema_version": FEATURE_SCHEMA,
        "binding": binding,
        "etfs": etfs,
        "stocks": stocks,
        "integrated_rotation_clusters": clusters,
        "accumulation_clusters": [
            {
                **row,
                "accum_cluster": row["integrated_cluster"],
                "cluster_score": row["integrated_score"],
                "selection_state": row["integrated_state"],
            }
            for row in clusters
        ],
        "master_eligibility_counts": {
            "candidate_etf": len(etf_symbols),
            "material_flow_price_evidence": len(etfs),
        },
        "master_flow_status_counts": {
            "visible_pit_flow": len(etfs),
            "missing_or_immaterial": max(len(etf_symbols) - len(etfs), 0),
        },
    }
    payload["snapshot_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload
