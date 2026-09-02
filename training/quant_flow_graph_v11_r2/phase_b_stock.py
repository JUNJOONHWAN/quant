"""Phase B stock-level all-ETF Flow propagation and avoidance tournament.

The stock path deliberately separates three economic channels:

* market-wide Drift from every eligible ETF family;
* direct constituent pressure from ETFs that actually hold a stock at T-1;
* indirect Diffusion from every eligible ETF in the stock's PIT cluster exposures.

No absolute Flow field is date-centred.  Duplicate ETF families are collapsed
only for breadth and exposure topology; their actual dollar Flow remains in the
direct pressure sum.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import DEFAULT_SOURCE_DATABASE, TIMING_CONTRACT
from .hypotheses import specification
from .phase_a import json_sha256, readonly_connection, sha256_file, utc_now, write_json_atomic
from .phase_b_cluster import _cluster_sequences, cluster_flow_states
from .phase_b_market import build_market_matrix


PHASE_B_STOCK_SCHEMA_VERSION = "quant.etf_flow_v11_r2.phase_b_stock.v1"
DEFAULT_PHASE_A_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/r2_phase_a"
)
DEFAULT_GRAPH_DATASET_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/"
    "full_20180102_20260729_allpanel"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/r2_phase_b_stock"
)

TARGET_NAMES = (
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
)
OUTER_YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
PURGE_SESSIONS = 20
RIDGE_ALPHA = 100.0
MIN_RELATION_COVERAGE = 0.95

# Frozen before the first stock-path result is read.
GLOBAL_FLOW_FIELDS = (
    "eligible_signed_flow_log",
    "clean_signed_flow_log",
    "special_effective_signed_flow_log",
    "drift_signed_flow_log",
    "drift_rate_pct",
    "independent_breadth_net",
    "diffusion_coverage",
    "observed_ratio",
    "missing_ratio",
    "stale_ratio",
    "true_zero_observed_ratio",
    "positive_family_share",
    "negative_family_share",
    "anchor_spy_flow_rate",
    "anchor_qqq_flow_rate",
    "anchor_vti_flow_rate",
    "anchor_rsp_flow_rate",
    "anchor_iwm_flow_rate",
    "anchor_dia_flow_rate",
    "drift_rate_pct_mean_5",
    "drift_rate_pct_mean_20",
    "drift_rate_pct_z60",
    "drift_rate_pct_change_5",
    "independent_breadth_net_mean_5",
    "independent_breadth_net_mean_20",
    "independent_breadth_net_z60",
    "independent_breadth_net_change_5",
    "diffusion_coverage_mean_5",
    "diffusion_coverage_mean_20",
    "diffusion_coverage_z60",
    "diffusion_coverage_change_5",
)
GLOBAL_MASK_FIELDS = (
    "diffusion_coverage",
    "observed_ratio",
    "missing_ratio",
    "stale_ratio",
    "true_zero_observed_ratio",
)
DIRECT_MASK_FIELDS = (
    "direct_connected_etf_count",
    "direct_known_identity_etf_count",
    "direct_clean_observed_etf_count",
    "direct_special_observed_etf_count",
    "direct_holding_weight_sum",
    "direct_clean_observed_weight_coverage",
    "direct_identity_weight_coverage",
    "direct_independent_family_count",
    "direct_cluster_count",
    "direct_holdings_age_scaled_mean",
)
DIRECT_FLOW_FIELDS = (
    "direct_clean_rate_net",
    "direct_clean_rate_gross",
    "direct_clean_usd_log",
    "direct_family_breadth_net",
    "direct_positive_family_share",
    "direct_negative_family_share",
    "direct_family_contribution_hhi",
    "direct_special_effective_rate_net",
    "direct_special_effective_usd_log",
)
INDIRECT_BASE_FIELDS = (
    "cluster_flow_rate_pct",
    "cluster_breadth_net",
    "cluster_coverage",
    "cluster_zero_share",
    "cluster_signed_flow_log",
    "cluster_absolute_flow_log",
    "cluster_special_effective_signed_flow_log",
    "cluster_special_flow_rate_pct",
    "cluster_flow_rate_pct_mean_5",
    "cluster_flow_rate_pct_mean_20",
    "cluster_flow_rate_pct_z60",
    "cluster_breadth_net_mean_5",
    "cluster_breadth_net_mean_20",
    "cluster_breadth_net_z60",
)
INDIRECT_FLOW_FIELDS = tuple(f"indirect_{name}" for name in INDIRECT_BASE_FIELDS) + (
    "indirect_cluster_exposure_hhi",
    "indirect_positive_cluster_exposure_share",
    "indirect_negative_cluster_exposure_share",
    "indirect_cluster_rate_dispersion",
    "indirect_drift_convergence",
)
RELATION_FIELDS = (
    "direct_minus_indirect_rate",
    "indirect_minus_market_drift_rate",
    "direct_indirect_sign_convergence",
    "relative_momentum_x_indirect_rate",
    "drawdown_x_outflow_pressure",
)
ROLLING_BASE_FIELDS = (
    "direct_clean_rate_net",
    "direct_family_breadth_net",
    "indirect_cluster_flow_rate_pct",
)
ROLLING_SUFFIXES = ("mean_5", "mean_20", "z60", "change_5")
ROLLING_FIELDS = tuple(
    f"{name}_{suffix}" for name in ROLLING_BASE_FIELDS for suffix in ROLLING_SUFFIXES
)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _signed_log_dollars(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values) / 1_000_000.0)


def _bincount(
    indices: np.ndarray, weights: np.ndarray | None, size: int
) -> np.ndarray:
    return np.bincount(indices, weights=weights, minlength=size).astype(np.float64)


def _unique_groups(
    stock_ids: np.ndarray,
    secondary_ids: np.ndarray,
    secondary_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = stock_ids.astype(np.int64) * int(secondary_width) + secondary_ids.astype(np.int64)
    unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return unique, first, inverse, unique // int(secondary_width)


def aggregate_snapshot_features(
    *,
    stock_count: int,
    edge_stock: np.ndarray,
    edge_etf: np.ndarray,
    edge_weight: np.ndarray,
    edge_age: np.ndarray,
    family_code: np.ndarray,
    cluster_code: np.ndarray,
    clean_observed: np.ndarray,
    special_observed: np.ndarray,
    flow_rate: np.ndarray,
    fund_flow: np.ndarray,
    effective_sign: np.ndarray,
    target_multiple: np.ndarray,
    cluster_states: np.ndarray,
    drift_rate: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Aggregate one PIT ETF-stock graph without date-centering Flow."""

    edge_stock = np.asarray(edge_stock, dtype=np.int64)
    edge_etf = np.asarray(edge_etf, dtype=np.int64)
    edge_weight = np.asarray(edge_weight, dtype=np.float64)
    edge_age = np.asarray(edge_age, dtype=np.float64)
    connected = _bincount(edge_stock, None, stock_count)
    total_weight = _bincount(edge_stock, edge_weight, stock_count)
    weighted_age = _safe_ratio(
        _bincount(edge_stock, edge_weight * edge_age, stock_count), total_weight
    )

    edge_family = family_code[edge_etf]
    edge_cluster = cluster_code[edge_etf]
    known = (edge_family >= 0) & (edge_cluster >= 0)
    known_count = _bincount(edge_stock[known], None, stock_count)
    known_weight = _bincount(edge_stock[known], edge_weight[known], stock_count)

    clean = clean_observed[edge_etf] & np.isfinite(flow_rate[edge_etf])
    special = special_observed[edge_etf] & np.isfinite(flow_rate[edge_etf])
    clean_count = _bincount(edge_stock[clean], None, stock_count)
    special_count = _bincount(edge_stock[special], None, stock_count)
    clean_weight = _bincount(edge_stock[clean], edge_weight[clean], stock_count)

    clean_rate_contribution = edge_weight[clean] * flow_rate[edge_etf[clean]]
    clean_usd_contribution = edge_weight[clean] * fund_flow[edge_etf[clean]]
    clean_rate_net = _bincount(edge_stock[clean], clean_rate_contribution, stock_count)
    clean_rate_gross = _bincount(
        edge_stock[clean], np.abs(clean_rate_contribution), stock_count
    )
    clean_usd_net = _bincount(edge_stock[clean], clean_usd_contribution, stock_count)

    special_multiplier = (
        effective_sign[edge_etf[special]] * np.abs(target_multiple[edge_etf[special]])
    )
    special_rate_net = _bincount(
        edge_stock[special],
        edge_weight[special] * flow_rate[edge_etf[special]] * special_multiplier,
        stock_count,
    )
    special_usd_net = _bincount(
        edge_stock[special],
        edge_weight[special] * fund_flow[edge_etf[special]] * special_multiplier,
        stock_count,
    )

    family_count = np.zeros(stock_count, dtype=np.float64)
    cluster_count = np.zeros(stock_count, dtype=np.float64)
    family_breadth = np.full(stock_count, np.nan, dtype=np.float64)
    family_positive_share = np.full(stock_count, np.nan, dtype=np.float64)
    family_negative_share = np.full(stock_count, np.nan, dtype=np.float64)
    family_hhi = np.full(stock_count, np.nan, dtype=np.float64)
    indirect = np.full(
        (stock_count, len(INDIRECT_BASE_FIELDS) + 5), np.nan, dtype=np.float64
    )

    if np.any(known):
        family_width = int(np.max(edge_family[known])) + 1
        _, first, inverse, family_stock = _unique_groups(
            edge_stock[known], edge_family[known], family_width
        )
        family_cluster = edge_cluster[known][first]
        family_max_weight = np.zeros(len(first), dtype=np.float64)
        np.maximum.at(family_max_weight, inverse, edge_weight[known])
        family_count = _bincount(family_stock, None, stock_count)

        cluster_width = cluster_states.shape[0]
        _, cluster_first, cluster_inverse, cluster_stock = _unique_groups(
            family_stock, family_cluster, cluster_width
        )
        grouped_cluster_code = family_cluster[cluster_first]
        grouped_cluster_weight = _bincount(
            cluster_inverse, family_max_weight, len(cluster_first)
        )
        cluster_count = _bincount(cluster_stock, None, stock_count)
        cluster_weight_sum = _bincount(
            cluster_stock, grouped_cluster_weight, stock_count
        )
        cluster_weight_sq = _bincount(
            cluster_stock, np.square(grouped_cluster_weight), stock_count
        )
        indirect[:, len(INDIRECT_BASE_FIELDS)] = _safe_ratio(
            cluster_weight_sq, np.square(cluster_weight_sum)
        )

        for feature_index in range(len(INDIRECT_BASE_FIELDS)):
            values = cluster_states[grouped_cluster_code, feature_index]
            finite = np.isfinite(values)
            numerator = _bincount(
                cluster_stock[finite],
                grouped_cluster_weight[finite] * values[finite],
                stock_count,
            )
            denominator = _bincount(
                cluster_stock[finite], grouped_cluster_weight[finite], stock_count
            )
            indirect[:, feature_index] = _safe_ratio(numerator, denominator)

        rate = cluster_states[grouped_cluster_code, 0]
        finite_rate = np.isfinite(rate)
        rate_weight = _bincount(
            cluster_stock[finite_rate], grouped_cluster_weight[finite_rate], stock_count
        )
        positive_weight = _bincount(
            cluster_stock[finite_rate & (rate > 0)],
            grouped_cluster_weight[finite_rate & (rate > 0)],
            stock_count,
        )
        negative_weight = _bincount(
            cluster_stock[finite_rate & (rate < 0)],
            grouped_cluster_weight[finite_rate & (rate < 0)],
            stock_count,
        )
        indirect[:, len(INDIRECT_BASE_FIELDS) + 1] = _safe_ratio(
            positive_weight, rate_weight
        )
        indirect[:, len(INDIRECT_BASE_FIELDS) + 2] = _safe_ratio(
            negative_weight, rate_weight
        )
        mean_rate = indirect[:, 0]
        centered_sq = np.square(rate[finite_rate] - mean_rate[cluster_stock[finite_rate]])
        variance = _safe_ratio(
            _bincount(
                cluster_stock[finite_rate],
                grouped_cluster_weight[finite_rate] * centered_sq,
                stock_count,
            ),
            rate_weight,
        )
        indirect[:, len(INDIRECT_BASE_FIELDS) + 3] = np.sqrt(
            np.maximum(variance, 0.0)
        )
        if math.isfinite(drift_rate) and drift_rate != 0:
            aligned = np.sign(rate[finite_rate]) * math.copysign(1.0, drift_rate)
            indirect[:, len(INDIRECT_BASE_FIELDS) + 4] = _safe_ratio(
                _bincount(
                    cluster_stock[finite_rate],
                    grouped_cluster_weight[finite_rate] * aligned,
                    stock_count,
                ),
                rate_weight,
            )
        else:
            indirect[:, len(INDIRECT_BASE_FIELDS) + 4] = 0.0

    clean_known = clean & (edge_family >= 0)
    if np.any(clean_known):
        family_width = int(np.max(edge_family[clean_known])) + 1
        _, _, inverse, clean_family_stock = _unique_groups(
            edge_stock[clean_known], edge_family[clean_known], family_width
        )
        family_contribution = _bincount(
            inverse,
            edge_weight[clean_known] * flow_rate[edge_etf[clean_known]],
            int(np.max(inverse)) + 1,
        )
        family_gross = _bincount(
            clean_family_stock, np.abs(family_contribution), stock_count
        )
        family_hhi = _safe_ratio(
            _bincount(clean_family_stock, np.square(family_contribution), stock_count),
            np.square(family_gross),
        )
        observed_family_count = _bincount(clean_family_stock, None, stock_count)
        positive_family = _bincount(
            clean_family_stock[family_contribution > 0], None, stock_count
        )
        negative_family = _bincount(
            clean_family_stock[family_contribution < 0], None, stock_count
        )
        family_breadth = _safe_ratio(
            positive_family - negative_family, observed_family_count
        )
        family_positive_share = _safe_ratio(positive_family, observed_family_count)
        family_negative_share = _safe_ratio(negative_family, observed_family_count)

    direct_mask = np.column_stack(
        [
            connected,
            known_count,
            clean_count,
            special_count,
            total_weight,
            _safe_ratio(clean_weight, total_weight),
            _safe_ratio(known_weight, total_weight),
            family_count,
            cluster_count,
            weighted_age,
        ]
    )
    direct_flow = np.column_stack(
        [
            clean_rate_net,
            clean_rate_gross,
            _signed_log_dollars(clean_usd_net),
            family_breadth,
            family_positive_share,
            family_negative_share,
            family_hhi,
            special_rate_net,
            _signed_log_dollars(special_usd_net),
        ]
    )
    audit = {
        "edge_count": int(len(edge_stock)),
        "known_identity_edge_count": int(np.sum(known)),
        "clean_observed_edge_count": int(np.sum(clean)),
        "special_observed_edge_count": int(np.sum(special)),
    }
    return np.column_stack([direct_mask, direct_flow]), indirect, audit


def _rolling_mean(values: np.ndarray, window: int, minimum: int) -> np.ndarray:
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(clean)])
    counts = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    result = np.full(len(values), np.nan, dtype=np.float64)
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    count = counts[end] - counts[start]
    valid = count >= minimum
    result[valid] = (cumulative[end[valid]] - cumulative[start[valid]]) / count[valid]
    return result


def _rolling_z(values: np.ndarray, window: int, minimum: int) -> np.ndarray:
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(clean)])
    cumulative_sq = np.concatenate([[0.0], np.cumsum(np.square(clean))])
    counts = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    result = np.full(len(values), np.nan, dtype=np.float64)
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    count = counts[end] - counts[start]
    total = cumulative[end] - cumulative[start]
    total_sq = cumulative_sq[end] - cumulative_sq[start]
    variance = np.full(len(values), np.nan, dtype=np.float64)
    enough = count >= minimum
    variance[enough] = (
        total_sq[enough] - np.square(total[enough]) / count[enough]
    ) / np.maximum(count[enough] - 1, 1)
    valid = enough & finite & (variance > 1e-12)
    result[valid] = (values[valid] - total[valid] / count[valid]) / np.sqrt(
        variance[valid]
    )
    return result


def add_symbol_rolling_features(
    *,
    flow: np.ndarray,
    flow_names: Sequence[str],
    date_codes: np.ndarray,
    symbol_codes: np.ndarray,
    symbol_count: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    derived = np.full((len(flow), len(ROLLING_FIELDS)), np.nan, dtype=np.float32)
    name_to_index = {name: index for index, name in enumerate(flow_names)}
    date_count = int(np.max(date_codes)) + 1
    output_index = 0
    for base_name in ROLLING_BASE_FIELDS:
        source_column = name_to_index[base_name]
        for symbol_code in range(symbol_count):
            indices = np.flatnonzero(symbol_codes == symbol_code)
            if not len(indices):
                continue
            values = np.full(date_count, np.nan, dtype=np.float64)
            values[date_codes[indices]] = flow[indices, source_column]
            mean5 = _rolling_mean(values, 5, 3)
            mean20 = _rolling_mean(values, 20, 10)
            z60 = _rolling_z(values, 60, 20)
            change5 = np.full(len(values), np.nan, dtype=np.float64)
            if len(values) > 5:
                change5[5:] = values[5:] - values[:-5]
            derived[indices, output_index : output_index + 4] = np.column_stack(
                [mean5, mean20, z60, change5]
            )[date_codes[indices]].astype(np.float32)
        output_index += 4
    return np.column_stack([flow, derived]), tuple(flow_names) + ROLLING_FIELDS


def _event_arrays(
    *,
    connection: sqlite3.Connection,
    signal_date: str,
    local_tickers: Sequence[str],
    cluster_to_code: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], tuple[str, str]]:
    ticker_to_local = {ticker: index for index, ticker in enumerate(local_tickers)}
    size = len(local_tickers)
    family_code = np.full(size, -1, dtype=np.int32)
    cluster_code = np.full(size, -1, dtype=np.int16)
    clean_observed = np.zeros(size, dtype=bool)
    special_observed = np.zeros(size, dtype=bool)
    flow_rate = np.full(size, np.nan, dtype=np.float64)
    fund_flow = np.zeros(size, dtype=np.float64)
    effective_sign = np.ones(size, dtype=np.float64)
    target_multiple = np.ones(size, dtype=np.float64)
    family_to_code: dict[str, int] = {}
    price_date = ""
    flow_date = ""
    rows = connection.execute(
        """
        SELECT price_date,flow_date,ticker,observed_exact_t2,fund_flow,flow_rate_pct,
          clean_eligible,special_eligible,effective_sign,target_multiple,
          cluster_family,independent_family_id
        FROM etf_flow_events
        WHERE signal_date=?
        """,
        (signal_date,),
    )
    for row in rows:
        price_date = str(row[0])
        flow_date = str(row[1])
        local = ticker_to_local.get(str(row[2]))
        if local is None:
            continue
        observed = bool(row[3])
        fund_flow[local] = float(row[4] or 0.0)
        flow_rate[local] = float(row[5]) if row[5] is not None else math.nan
        clean_observed[local] = observed and bool(row[6])
        special_observed[local] = observed and bool(row[7])
        effective_sign[local] = float(row[8] or 0.0)
        target_multiple[local] = float(row[9] or 0.0)
        cluster_code[local] = cluster_to_code.get(str(row[10]), -1)
        family = str(row[11])
        if family not in family_to_code:
            family_to_code[family] = len(family_to_code)
        family_code[local] = family_to_code[family]
    return {
        "family_code": family_code,
        "cluster_code": cluster_code,
        "clean_observed": clean_observed,
        "special_observed": special_observed,
        "flow_rate": flow_rate,
        "fund_flow": fund_flow,
        "effective_sign": effective_sign,
        "target_multiple": target_multiple,
    }, (price_date, flow_date)


def build_stock_matrix_from_sources(
    *,
    event: sqlite3.Connection,
    source: sqlite3.Connection,
    graph_dataset_root: Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    manifest_path = graph_dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "quant.etf_flow_graph_dataset.v2":
        raise ValueError("unexpected graph dataset schema")
    if tuple(manifest["feature_contract"]["targets"]) != TARGET_NAMES:
        raise ValueError("stock target contract mismatch")
    stock_feature_names = tuple(manifest["feature_contract"]["stock"])
    etf_vocabulary = tuple(str(value) for value in manifest["etf_vocabulary"])
    symbol_values = list(sorted(str(value) for value in manifest["requested_symbols"]))
    symbol_to_code = {symbol: index for index, symbol in enumerate(symbol_values)}

    market = build_market_matrix(event=event, source=source)
    market_date_to_index = {date: index for index, date in enumerate(market["dates"])}
    market_flow_names = tuple(market["flow_names"])
    missing_global = [name for name in GLOBAL_FLOW_FIELDS if name not in market_flow_names]
    if missing_global:
        raise ValueError(f"missing global Flow fields: {missing_global}")
    global_indices = [market_flow_names.index(name) for name in GLOBAL_FLOW_FIELDS]
    drift_index = market_flow_names.index("drift_rate_pct")

    states = cluster_flow_states(event)
    cluster_values = tuple(sorted({cluster for _, cluster in states}))
    cluster_to_code = {cluster: index for index, cluster in enumerate(cluster_values)}
    states = _cluster_sequences(states, tuple(market["dates"]))

    price_parts: list[np.ndarray] = []
    flow_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    date_code_parts: list[np.ndarray] = []
    symbol_code_parts: list[np.ndarray] = []
    date_values: list[str] = []
    audit = defaultdict(int)
    excluded = defaultdict(int)
    timing_violations: list[dict[str, str]] = []
    refs = tuple(manifest["snapshots"])
    for ref_number, ref in enumerate(refs, 1):
        signal_date = str(ref["signal_date"])
        market_index = market_date_to_index.get(signal_date)
        if market_index is None:
            excluded["no_v11_event_date"] += 1
            continue
        if float(ref.get("relation_stock_coverage_ratio") or 0.0) < MIN_RELATION_COVERAGE:
            excluded["relation_coverage_below_95pct"] += 1
            continue
        if int(signal_date[:4]) < 2020:
            excluded["pre_2020_relation_window"] += 1
            continue
        with np.load(ref["path"], allow_pickle=False) as item:
            stock_symbols = tuple(str(value) for value in item["stock_symbols"])
            stock_x = item["stock_x"].astype(np.float32, copy=True)
            targets = item["targets"].astype(np.float32, copy=True)
            target_mask = item["target_mask"].astype(bool, copy=True)
            local_global_etf = item["etf_ids"].astype(np.int64, copy=True)
            edge_index = item["edge_index"].astype(np.int64, copy=True)
            edge_attr = item["edge_attr"].astype(np.float32, copy=True)
        for symbol in stock_symbols:
            if symbol not in symbol_to_code:
                symbol_to_code[symbol] = len(symbol_values)
                symbol_values.append(symbol)
                audit["symbols_added_beyond_requested_manifest"] += 1
        local_tickers = tuple(etf_vocabulary[index] for index in local_global_etf)
        event_arrays, timing = _event_arrays(
            connection=event,
            signal_date=signal_date,
            local_tickers=local_tickers,
            cluster_to_code=cluster_to_code,
        )
        if timing != (str(ref["price_date"]), str(ref["flow_date"])):
            timing_violations.append(
                {
                    "signal_date": signal_date,
                    "graph_price_date": str(ref["price_date"]),
                    "graph_flow_date": str(ref["flow_date"]),
                    "event_price_date": timing[0],
                    "event_flow_date": timing[1],
                }
            )
            continue

        cluster_state = np.full(
            (len(cluster_values), len(INDIRECT_BASE_FIELDS)), np.nan, dtype=np.float64
        )
        for cluster, cluster_code in cluster_to_code.items():
            state = states.get((signal_date, cluster), {})
            cluster_state[cluster_code] = [
                float(state.get(name, math.nan)) for name in INDIRECT_BASE_FIELDS
            ]
        drift_rate = float(market["flow_matrix"][market_index, drift_index])
        direct, indirect, snapshot_audit = aggregate_snapshot_features(
            stock_count=len(stock_symbols),
            edge_stock=edge_index[0],
            edge_etf=edge_index[1],
            edge_weight=edge_attr[:, 0],
            edge_age=edge_attr[:, 1],
            cluster_states=cluster_state,
            drift_rate=drift_rate,
            **event_arrays,
        )
        for key, value in snapshot_audit.items():
            audit[key] += value
        audit["snapshot_count"] += 1
        audit["raw_stock_row_count"] += len(stock_symbols)

        direct_names = DIRECT_MASK_FIELDS + DIRECT_FLOW_FIELDS
        direct_index = {name: index for index, name in enumerate(direct_names)}
        indirect_rate = indirect[:, INDIRECT_BASE_FIELDS.index("cluster_flow_rate_pct")]
        direct_rate = direct[:, direct_index["direct_clean_rate_net"]]
        relative_index = stock_feature_names.index("relative_ret_5d")
        drawdown_index = stock_feature_names.index("drawdown_20d_pct")
        relations = np.column_stack(
            [
                direct_rate - indirect_rate,
                indirect_rate - drift_rate,
                np.sign(direct_rate) * np.sign(indirect_rate),
                stock_x[:, relative_index] * indirect_rate,
                stock_x[:, drawdown_index] * np.minimum(indirect_rate, 0.0),
            ]
        )
        global_row = market["flow_matrix"][market_index, global_indices]
        global_rows = np.repeat(global_row[None, :], len(stock_symbols), axis=0)
        complete = np.all(target_mask, axis=1) & np.all(np.isfinite(targets), axis=1)
        excluded["incomplete_12_target_stock_rows"] += int(np.sum(~complete))
        if not np.any(complete):
            excluded["no_complete_target_snapshot"] += 1
            continue
        date_code = len(date_values)
        date_values.append(signal_date)
        price_parts.append(stock_x[complete])
        flow_parts.append(
            np.column_stack([global_rows, direct, indirect, relations])[complete].astype(
                np.float32
            )
        )
        target_parts.append(targets[complete])
        date_code_parts.append(np.full(int(np.sum(complete)), date_code, dtype=np.int32))
        symbol_code_parts.append(
            np.asarray(
                [symbol_to_code[stock_symbols[index]] for index in np.flatnonzero(complete)],
                dtype=np.int16,
            )
        )
        audit["complete_target_stock_row_count"] += int(np.sum(complete))
        if progress and (
            audit["snapshot_count"] == 1 or ref_number == len(refs) or audit["snapshot_count"] % 100 == 0
        ):
            progress(
                {
                    "stage": "phase_b_stock_matrix",
                    "signal_date": signal_date,
                    "completed_snapshots": audit["snapshot_count"],
                    "total_manifest_snapshots": len(refs),
                    "complete_stock_rows": audit["complete_target_stock_row_count"],
                    "edges_processed": audit["edge_count"],
                    "at_utc": utc_now(),
                }
            )

    if timing_violations:
        raise ValueError(f"stock timing violations: {timing_violations[:3]}")
    if not price_parts:
        raise ValueError("no eligible stock snapshots")
    price = np.concatenate(price_parts).astype(np.float32)
    flow = np.concatenate(flow_parts).astype(np.float32)
    targets = np.concatenate(target_parts).astype(np.float32)
    date_codes = np.concatenate(date_code_parts)
    symbol_codes = np.concatenate(symbol_code_parts)
    base_flow_names = (
        GLOBAL_FLOW_FIELDS
        + DIRECT_MASK_FIELDS
        + DIRECT_FLOW_FIELDS
        + INDIRECT_FLOW_FIELDS
        + RELATION_FIELDS
    )
    flow, flow_names = add_symbol_rolling_features(
        flow=flow,
        flow_names=base_flow_names,
        date_codes=date_codes,
        symbol_codes=symbol_codes,
        symbol_count=len(symbol_values),
    )
    return {
        "date_values": tuple(date_values),
        "date_codes": date_codes,
        "symbol_values": tuple(symbol_values),
        "symbol_codes": symbol_codes,
        "price_names": stock_feature_names,
        "price_matrix": price,
        "flow_names": flow_names,
        "flow_matrix": flow,
        "target_names": TARGET_NAMES,
        "targets": targets,
        "clusters": cluster_values,
        "audit": dict(audit),
        "excluded": dict(excluded),
        "timing_violation_count": len(timing_violations),
        "source_manifest_sha256": sha256_file(manifest_path),
    }


def feature_groups(matrix: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    price_count = len(matrix["price_names"])
    flow_names = tuple(matrix["flow_names"])
    flow_index = {name: price_count + index for index, name in enumerate(flow_names)}
    price = tuple(range(price_count))

    def combine(names: Sequence[str]) -> tuple[int, ...]:
        return price + tuple(flow_index[name] for name in names)

    masks = tuple(dict.fromkeys(GLOBAL_MASK_FIELDS + DIRECT_MASK_FIELDS))
    global_flow = tuple(GLOBAL_FLOW_FIELDS)
    direct = tuple(dict.fromkeys(DIRECT_MASK_FIELDS + DIRECT_FLOW_FIELDS + ROLLING_FIELDS[:8]))
    indirect = tuple(
        dict.fromkeys(
            DIRECT_MASK_FIELDS
            + INDIRECT_FLOW_FIELDS
            + RELATION_FIELDS
            + ROLLING_FIELDS[8:]
        )
    )
    direct_plus_global = tuple(dict.fromkeys(global_flow + direct))
    indirect_plus_global = tuple(dict.fromkeys(global_flow + indirect))
    full = tuple(flow_names)
    return {
        "price_only": price,
        "mask_only": combine(masks),
        "global_drift_diffusion": combine(global_flow),
        "direct_holdings_only": combine(direct),
        "indirect_all_etf_only": combine(indirect_plus_global),
        "direct_plus_global": combine(direct_plus_global),
        "full_stock_drift_diffusion": combine(full),
        "full_special_channel_off": combine(
            tuple(name for name in full if "special" not in name)
        ),
    }


def price_capacity_controls(price: np.ndarray, count: int) -> np.ndarray:
    """Create leakage-free price transforms matching the Flow feature count."""

    source = np.nan_to_num(np.asarray(price, dtype=np.float32), nan=0.0)
    clipped = np.clip(source, -1_000.0, 1_000.0)
    candidates = [
        np.sign(clipped) * np.log1p(np.abs(clipped)),
        np.sign(clipped) * np.sqrt(np.abs(clipped)),
        np.square(np.clip(clipped, -100.0, 100.0)),
        clipped * np.roll(clipped, 1, axis=1),
        clipped * np.roll(clipped, 3, axis=1),
    ]
    result = np.column_stack(candidates)
    if result.shape[1] < count:
        repeats = math.ceil(count / result.shape[1])
        result = np.tile(result, (1, repeats))
    return result[:, :count].astype(np.float32)


def lag_flow_by_symbol(
    flow: np.ndarray,
    date_codes: np.ndarray,
    symbol_codes: np.ndarray,
    date_count: int,
    symbol_count: int,
    sessions: int,
) -> np.ndarray:
    lookup = np.full((date_count, symbol_count), -1, dtype=np.int32)
    lookup[date_codes, symbol_codes] = np.arange(len(flow), dtype=np.int32)
    result = np.full_like(flow, np.nan)
    valid = date_codes >= sessions
    source = lookup[date_codes[valid] - sessions, symbol_codes[valid]]
    found = source >= 0
    destination = np.flatnonzero(valid)[found]
    result[destination] = flow[source[found]]
    return result


def topology_shuffle(
    flow: np.ndarray,
    date_codes: np.ndarray,
    global_feature_count: int,
    seed: int,
) -> np.ndarray:
    """Destroy stock exposure topology while preserving each date's Flow cross-section."""

    result = flow.copy()
    rng = np.random.default_rng(seed)
    for date_code in np.unique(date_codes):
        indices = np.flatnonzero(date_codes == date_code)
        permutation = rng.permutation(indices)
        result[indices, global_feature_count:] = flow[
            permutation, global_feature_count:
        ]
    return result


def fold_indices(matrix: Mapping[str, Any], year: int) -> tuple[np.ndarray, np.ndarray]:
    date_values = matrix["date_values"]
    date_codes = matrix["date_codes"]
    test_date_codes = np.asarray(
        [index for index, date in enumerate(date_values) if int(date[:4]) == year],
        dtype=np.int32,
    )
    if not len(test_date_codes):
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    first_test = int(test_date_codes[0])
    train_last_exclusive = first_test - PURGE_SESSIONS
    train = np.flatnonzero(date_codes < train_last_exclusive)
    test = np.flatnonzero(np.isin(date_codes, test_date_codes))
    return train, test


def _standardized_sufficient_statistics(
    matrix: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.nan_to_num(matrix[train], nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(matrix[test], nan=0.0, posinf=0.0, neginf=0.0)
    means = np.mean(train_x, axis=0, dtype=np.float64)
    scales = np.std(train_x, axis=0, dtype=np.float64)
    scales[scales < 1e-9] = 1.0
    train_z = np.clip((train_x - means) / scales, -20.0, 20.0).astype(np.float32)
    test_z = np.clip((test_x - means) / scales, -20.0, 20.0).astype(np.float32)
    target_mean = np.mean(targets[train], axis=0, dtype=np.float64)
    centered = (targets[train] - target_mean).astype(np.float32)
    xtx = (train_z.T @ train_z).astype(np.float64)
    xty = (train_z.T @ centered).astype(np.float64)
    return xtx, xty, test_z, target_mean, scales


def fit_predict_groups(
    *,
    matrix: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    groups: Mapping[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    xtx, xty, test_z, target_mean, _ = _standardized_sufficient_statistics(
        matrix, targets, train, test
    )
    predictions: dict[str, np.ndarray] = {}
    for name, raw_indices in groups.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        gram = xtx[np.ix_(indices, indices)].copy()
        gram.flat[:: len(indices) + 1] += RIDGE_ALPHA
        beta = np.linalg.solve(gram, xty[indices])
        predictions[name] = (test_z[:, indices] @ beta + target_mean).astype(np.float32)
    return predictions


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    if np.std(target) > 1e-12 and np.std(prediction) > 1e-12:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    else:
        correlation = math.nan
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "correlation": correlation,
        "direction_accuracy": float(np.mean((target >= 0) == (prediction >= 0))),
    }


def stock_cross_sectional_metrics(
    *,
    date_codes: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    loss_target: bool,
) -> dict[str, float]:
    rank_ics: list[float] = []
    spreads: list[float] = []
    tops: list[float] = []
    bottoms: list[float] = []
    for date_code in np.unique(date_codes):
        indices = np.flatnonzero(date_codes == date_code)
        if len(indices) < 20:
            continue
        actual = target[indices]
        forecast = prediction[indices]
        if np.std(actual) > 1e-12 and np.std(forecast) > 1e-12:
            rank_actual = np.argsort(np.argsort(actual)).astype(np.float64)
            rank_forecast = np.argsort(np.argsort(forecast)).astype(np.float64)
            rank_ics.append(float(np.corrcoef(rank_actual, rank_forecast)[0, 1]))
        basket_count = max(1, len(indices) // 10)
        order = np.argsort(forecast)
        bottom = float(np.mean(actual[order[:basket_count]]))
        top = float(np.mean(actual[order[-basket_count:]]))
        bottoms.append(bottom)
        tops.append(top)
        spreads.append(top - bottom)
    mean_top = float(np.mean(tops)) if tops else math.nan
    mean_bottom = float(np.mean(bottoms)) if bottoms else math.nan
    return {
        "mean_daily_rank_ic": float(np.mean(rank_ics)) if rank_ics else math.nan,
        "positive_daily_rank_ic_ratio": (
            float(np.mean(np.asarray(rank_ics) > 0)) if rank_ics else math.nan
        ),
        "mean_top_minus_bottom_spread": (
            float(np.mean(spreads)) if spreads else math.nan
        ),
        "mean_predicted_top_realized": mean_top,
        "mean_predicted_bottom_realized": mean_bottom,
        "economic_basket_value": -mean_bottom if loss_target else mean_top,
        "evaluated_date_count": len(spreads),
    }


def _pooled_receipts(
    *,
    matrix: Mapping[str, Any],
    actual_parts: Sequence[np.ndarray],
    date_code_parts: Sequence[np.ndarray],
    predictions: Mapping[str, Sequence[np.ndarray]],
    fold_receipts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    actual = np.concatenate(actual_parts)
    dates = np.concatenate(date_code_parts)
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        pooled: dict[str, Any] = {}
        loss_target = target_name.startswith("loss_")
        for model_name, parts in predictions.items():
            prediction = np.concatenate(parts)[:, target_index]
            pooled[model_name] = {
                **regression_metrics(actual[:, target_index], prediction),
                **stock_cross_sectional_metrics(
                    date_codes=dates,
                    target=actual[:, target_index],
                    prediction=prediction,
                    loss_target=loss_target,
                ),
            }
        capacity_mae = pooled["price_capacity_matched"]["mae"]
        full_mae = pooled["full_stock_drift_diffusion"]["mae"]
        pooled["full_stock_drift_diffusion"][
            "relative_mae_improvement_vs_price_capacity_pct"
        ] = (capacity_mae - full_mae) / capacity_mae * 100.0
        result[target_name] = {
            "rows": len(actual),
            "pooled": pooled,
            "folds": list(fold_receipts[target_name]),
        }
    return result


def evaluate_stock_matrix(
    matrix: Mapping[str, Any], progress: Callable[[Mapping[str, Any]], None] | None = None
) -> dict[str, Any]:
    price = matrix["price_matrix"]
    flow = matrix["flow_matrix"]
    targets = matrix["targets"]
    combined = np.column_stack([price, flow]).astype(np.float32)
    groups = feature_groups(matrix)
    full_indices = groups["full_stock_drift_diffusion"]
    flow_count = flow.shape[1]
    capacity = np.column_stack(
        [price, price_capacity_controls(price, flow_count)]
    ).astype(np.float32)
    lag5 = np.column_stack(
        [
            price,
            lag_flow_by_symbol(
                flow,
                matrix["date_codes"],
                matrix["symbol_codes"],
                len(matrix["date_values"]),
                len(matrix["symbol_values"]),
                5,
            ),
        ]
    ).astype(np.float32)
    lag20 = np.column_stack(
        [
            price,
            lag_flow_by_symbol(
                flow,
                matrix["date_codes"],
                matrix["symbol_codes"],
                len(matrix["date_values"]),
                len(matrix["symbol_values"]),
                20,
            ),
        ]
    ).astype(np.float32)
    shuffled = np.column_stack(
        [
            price,
            topology_shuffle(
                flow,
                matrix["date_codes"],
                len(GLOBAL_FLOW_FIELDS),
                seed=20260828,
            ),
        ]
    ).astype(np.float32)

    predictions: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    actual_parts: list[np.ndarray] = []
    date_code_parts: list[np.ndarray] = []
    fold_receipts: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for year in OUTER_YEARS:
        train, test = fold_indices(matrix, year)
        if len(train) < 50_000 or len(test) < 10_000:
            continue
        actual_parts.append(targets[test])
        date_code_parts.append(matrix["date_codes"][test])
        main_predictions = fit_predict_groups(
            matrix=combined,
            targets=targets,
            train=train,
            test=test,
            groups=groups,
        )
        capacity_prediction = fit_predict_groups(
            matrix=capacity,
            targets=targets,
            train=train,
            test=test,
            groups={"price_capacity_matched": tuple(range(capacity.shape[1]))},
        )["price_capacity_matched"]
        lag5_prediction = fit_predict_groups(
            matrix=lag5,
            targets=targets,
            train=train,
            test=test,
            groups={"lagged_5": full_indices},
        )["lagged_5"]
        lag20_prediction = fit_predict_groups(
            matrix=lag20,
            targets=targets,
            train=train,
            test=test,
            groups={"lagged_20": full_indices},
        )["lagged_20"]
        shuffle_prediction = fit_predict_groups(
            matrix=shuffled,
            targets=targets,
            train=train,
            test=test,
            groups={"topology_shuffle": full_indices},
        )["topology_shuffle"]
        fold_predictions = {
            **main_predictions,
            "price_capacity_matched": capacity_prediction,
            "lagged_5": lag5_prediction,
            "lagged_20": lag20_prediction,
            "topology_shuffle": shuffle_prediction,
        }
        for model_name, prediction in fold_predictions.items():
            predictions[model_name].append(prediction)
        train_last_code = int(np.max(matrix["date_codes"][train]))
        test_codes = matrix["date_codes"][test]
        for target_index, target_name in enumerate(TARGET_NAMES):
            fold_receipts[target_name].append(
                {
                    "outer_year": year,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_end_signal_date": matrix["date_values"][train_last_code],
                    "test_start_signal_date": matrix["date_values"][int(np.min(test_codes))],
                    "test_end_signal_date": matrix["date_values"][int(np.max(test_codes))],
                    "purge_sessions": PURGE_SESSIONS,
                    "models": {
                        model_name: regression_metrics(
                            targets[test, target_index], prediction[:, target_index]
                        )
                        for model_name, prediction in fold_predictions.items()
                    },
                }
            )
        if progress:
            progress(
                {
                    "stage": "phase_b_stock_oos",
                    "completed_outer_year": year,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "at_utc": utc_now(),
                }
            )
    if not actual_parts:
        raise ValueError("no eligible stock OOS folds")
    return _pooled_receipts(
        matrix=matrix,
        actual_parts=actual_parts,
        date_code_parts=date_code_parts,
        predictions=predictions,
        fold_receipts=fold_receipts,
    )


def summarize_gate(targets: Mapping[str, Any]) -> dict[str, Any]:
    counters = defaultdict(int)
    improvements: list[float] = []
    positive_fold_targets = 0
    outer_fold_targets = 0
    core_names = {
        "loss_5d_pct",
        "loss_20d_pct",
        "benchmark_downside_defense_5d_pct",
        "benchmark_downside_defense_20d_pct",
    }
    core = defaultdict(int)
    for target_name, target in targets.items():
        pooled = target["pooled"]
        full = pooled["full_stock_drift_diffusion"]
        capacity = pooled["price_capacity_matched"]
        counters["full_beats_price"] += full["mae"] < pooled["price_only"]["mae"]
        counters["full_beats_price_capacity"] += full["mae"] < capacity["mae"]
        counters["full_beats_lag5"] += full["mae"] < pooled["lagged_5"]["mae"]
        counters["full_beats_lag20"] += full["mae"] < pooled["lagged_20"]["mae"]
        counters["full_beats_topology_shuffle"] += (
            full["mae"] < pooled["topology_shuffle"]["mae"]
        )
        counters["indirect_adds_beyond_direct_global"] += (
            full["mae"] < pooled["direct_plus_global"]["mae"]
        )
        counters["rank_ic_beats_capacity"] += (
            full["mean_daily_rank_ic"] > capacity["mean_daily_rank_ic"]
        )
        counters["economic_basket_beats_capacity"] += (
            full["economic_basket_value"] > capacity["economic_basket_value"]
        )
        improvements.append(full["relative_mae_improvement_vs_price_capacity_pct"])
        if target_name in core_names:
            core["mae_beats_capacity"] += full["mae"] < capacity["mae"]
            core["rank_ic_beats_capacity"] += (
                full["mean_daily_rank_ic"] > capacity["mean_daily_rank_ic"]
            )
            core["economic_basket_beats_capacity"] += (
                full["economic_basket_value"] > capacity["economic_basket_value"]
            )
            core["beats_topology_shuffle"] += (
                full["mae"] < pooled["topology_shuffle"]["mae"]
            )
        for fold in target["folds"]:
            outer_fold_targets += 1
            positive_fold_targets += (
                fold["models"]["full_stock_drift_diffusion"]["mae"]
                < fold["models"]["price_capacity_matched"]["mae"]
            )

    mean_improvement = float(np.mean(improvements))
    forecast_pass = (
        counters["full_beats_price_capacity"] >= 8
        and mean_improvement > 0
        and counters["full_beats_topology_shuffle"] >= 8
        and counters["full_beats_lag5"] >= 8
        and counters["indirect_adds_beyond_direct_global"] >= 7
        and positive_fold_targets >= 36
    )
    basket_pass = (
        counters["rank_ic_beats_capacity"] >= 8
        and counters["economic_basket_beats_capacity"] >= 8
        and counters["full_beats_topology_shuffle"] >= 8
        and positive_fold_targets >= 36
    )
    avoidance_pass = (
        core["mae_beats_capacity"] >= 2
        and core["rank_ic_beats_capacity"] >= 3
        and core["economic_basket_beats_capacity"] >= 3
        and core["beats_topology_shuffle"] >= 3
    )
    passed_paths = [
        name
        for name, passed in (
            ("FORECAST", forecast_pass),
            ("BASKET", basket_pass),
            ("AVOIDANCE", avoidance_pass),
        )
        if passed
    ]
    return {
        "status": "PHASE_B_STOCK_PASS" if passed_paths else "PHASE_B_STOCK_FAIL",
        "passed_paths": passed_paths,
        "fixed_before_results": True,
        "checks": {
            "forecast_path_pass": forecast_pass,
            "basket_path_pass": basket_pass,
            "avoidance_path_pass": avoidance_pass,
            "full_beats_price_capacity_8_of_12": counters[
                "full_beats_price_capacity"
            ]
            >= 8,
            "mean_mae_improvement_positive": mean_improvement > 0,
            "full_beats_topology_shuffle_8_of_12": counters[
                "full_beats_topology_shuffle"
            ]
            >= 8,
            "full_beats_lag5_8_of_12": counters["full_beats_lag5"] >= 8,
            "indirect_adds_7_of_12": counters[
                "indirect_adds_beyond_direct_global"
            ]
            >= 7,
            "positive_half_outer_fold_targets": positive_fold_targets >= 36,
        },
        "counters": {
            **dict(counters),
            "mean_relative_mae_improvement_vs_price_capacity_pct": mean_improvement,
            "positive_outer_fold_target_count": int(positive_fold_targets),
            "outer_fold_target_count": int(outer_fold_targets),
            "target_count": len(targets),
            "avoidance_core": dict(core),
        },
    }


def _progress(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def preregistration() -> dict[str, Any]:
    return {
        "schema_version": "quant.etf_flow_v11_r2.phase_b_stock_preregistration.v1",
        "frozen_before_results": True,
        "timing_contract": TIMING_CONTRACT,
        "outer_years": list(OUTER_YEARS),
        "purge_sessions": PURGE_SESSIONS,
        "ridge_alpha": RIDGE_ALPHA,
        "targets": list(TARGET_NAMES),
        "channels": {
            "global_drift": list(GLOBAL_FLOW_FIELDS),
            "direct_pit_holdings": list(DIRECT_MASK_FIELDS + DIRECT_FLOW_FIELDS),
            "indirect_all_etf_diffusion": list(INDIRECT_FLOW_FIELDS),
            "relations": list(RELATION_FIELDS + ROLLING_FIELDS),
        },
        "controls": [
            "equal_width_nonlinear_price_capacity",
            "flow_lag_5_sessions",
            "flow_lag_20_sessions",
            "within_date_stock_topology_shuffle",
            "direct_plus_global_without_indirect",
            "special_channel_off",
        ],
        "gate_thresholds": {
            "forecast": {
                "mae_beats_equal_width_price_targets": 8,
                "mean_mae_improvement_positive": True,
                "mae_beats_topology_shuffle_targets": 8,
                "mae_beats_lag5_targets": 8,
                "indirect_adds_beyond_direct_global_targets": 7,
                "positive_outer_fold_targets": 36,
            },
            "basket": {
                "rank_ic_beats_equal_width_price_targets": 8,
                "economic_basket_beats_equal_width_price_targets": 8,
                "mae_beats_topology_shuffle_targets": 8,
                "positive_outer_fold_targets": 36,
            },
            "avoidance": {
                "core_targets": [
                    "loss_5d_pct",
                    "loss_20d_pct",
                    "benchmark_downside_defense_5d_pct",
                    "benchmark_downside_defense_20d_pct",
                ],
                "mae_beats_equal_width_price": 2,
                "rank_ic_beats_equal_width_price": 3,
                "economic_basket_beats_equal_width_price": 3,
                "mae_beats_topology_shuffle": 3,
            },
        },
        "prohibitions": {
            "date_center_absolute_flow": True,
            "table_48_breadth": True,
            "historical_holdings_imputation": True,
            "post_result_retuning": True,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output_root = Path(args.output_root)
    preregistration_name = "v11_r2_phase_b_stock_preregistration.json"
    existing_names = (
        {path.name for path in output_root.iterdir()} if output_root.exists() else set()
    )
    if existing_names - {preregistration_name} and not args.replace:
        raise FileExistsError(f"output root already populated: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    preregistration_path = output_root / preregistration_name
    frozen_preregistration = preregistration()
    if preregistration_path.exists():
        existing_preregistration = json.loads(
            preregistration_path.read_text(encoding="utf-8")
        )
        if existing_preregistration != frozen_preregistration:
            raise ValueError("existing stock preregistration does not match source")
    else:
        write_json_atomic(preregistration_path, frozen_preregistration)
    preregistration_sha256 = sha256_file(preregistration_path)
    phase_a_root = Path(args.phase_a_root)
    event_path = phase_a_root / "v11_r2_flow_event_cube.sqlite3"
    hypothesis_path = phase_a_root / "v11_r2_drift_diffusion_hypothesis_registry.json"
    if not event_path.exists() or not hypothesis_path.exists():
        raise FileNotFoundError("Phase A event cube or hypothesis registry is missing")
    started_at = utc_now()
    with readonly_connection(event_path) as event, readonly_connection(
        Path(args.source_database)
    ) as source:
        matrix = build_stock_matrix_from_sources(
            event=event,
            source=source,
            graph_dataset_root=Path(args.graph_dataset_root),
            progress=_progress,
        )
        targets = evaluate_stock_matrix(matrix, progress=_progress)
    gate = summarize_gate(targets)
    receipt = {
        "schema_version": PHASE_B_STOCK_SCHEMA_VERSION,
        "started_at_utc": started_at,
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        "hypothesis_registry_sha256": sha256_file(hypothesis_path),
        "hypothesis_specification_sha256": json_sha256(specification()),
        "stock_preregistration_sha256": preregistration_sha256,
        "stock_source_sha256": sha256_file(Path(__file__)),
        "source_event_cube_sha256": sha256_file(event_path),
        "source_graph_manifest_sha256": matrix["source_manifest_sha256"],
        "contract": {
            "outer_years": list(OUTER_YEARS),
            "purge_sessions": PURGE_SESSIONS,
            "ridge_alpha": RIDGE_ALPHA,
            "target_count": len(TARGET_NAMES),
            "min_relation_coverage": MIN_RELATION_COVERAGE,
            "date_centering_of_absolute_flow": False,
            "direct_dollar_flow_preserved": True,
            "independent_family_breadth": True,
            "indirect_uses_all_eligible_etfs_in_exposed_clusters": True,
            "table_48_breadth_used": False,
            "price_capacity_comparator_equal_width": True,
        },
        "feature_contract": {
            "price": list(matrix["price_names"]),
            "flow": list(matrix["flow_names"]),
            "targets": list(matrix["target_names"]),
            "groups": {
                name: len(indices) for name, indices in feature_groups(matrix).items()
            },
        },
        "scope": {
            "signal_date_start": matrix["date_values"][0],
            "signal_date_end": matrix["date_values"][-1],
            "signal_date_count": len(matrix["date_values"]),
            "stock_symbol_count": len(matrix["symbol_values"]),
            "stock_row_count": len(matrix["targets"]),
            "cluster_count": len(matrix["clusters"]),
            "audit": matrix["audit"],
            "excluded": matrix["excluded"],
            "timing_violation_count": matrix["timing_violation_count"],
        },
        "targets": targets,
        "gate": gate,
        "phase_c_activation": (
            "ACTIVATED_FOR_" + "_AND_".join(gate["passed_paths"])
            if gate["passed_paths"]
            else "NOT_ACTIVATED"
        ),
        "limitations": [
            "ETF cluster identity is current/static; PIT holdings edges themselves are date-local",
            "2018 and partial-2019 holdings are excluded rather than imputed",
            "indirect Diffusion is cluster-mediated and cannot identify an unobserved latent relation outside the 44-family taxonomy",
        ],
    }
    receipt_path = output_root / "v11_r2_phase_b_stock_receipt.json"
    write_json_atomic(receipt_path, receipt)
    return receipt_path, receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase-a-root", type=Path, default=DEFAULT_PHASE_A_ROOT)
    result.add_argument(
        "--graph-dataset-root", type=Path, default=DEFAULT_GRAPH_DATASET_ROOT
    )
    result.add_argument(
        "--source-database", type=Path, default=DEFAULT_SOURCE_DATABASE
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--replace", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    receipt_path, receipt = run(args)
    summary = {
        "status": receipt["gate"]["status"],
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "scope": receipt["scope"],
        "gate": receipt["gate"],
        "phase_c_activation": receipt["phase_c_activation"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if receipt["gate"]["status"] == "PHASE_B_STOCK_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
