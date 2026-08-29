"""Frozen contracts for the temporal ETF-stock graph forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from training.quant_forecast_v2.contracts import PRICE_FEATURES


DATASET_SCHEMA_VERSION: Final = "quant.etf_flow_graph_dataset.v2"
MODEL_SCHEMA_VERSION: Final = "quant.etf_flow_graph_forecaster.v6"
RECEIPT_SCHEMA_VERSION: Final = "quant.etf_flow_graph_training_receipt.v6"

TIMING_CONTRACT: Final = (
    "signal/trade session T / price and non-flow inputs through T-1 / "
    "Massive ETF Flow effective session exactly T-2 and captured by T / "
    "PIT holdings available no later than T-1 / no silent T-3 fallback"
)

FLOW_LOOKBACK_SESSIONS: Final = 60
FLOW_ACTIVE_LOOKBACK_SESSIONS: Final = FLOW_LOOKBACK_SESSIONS
FLOW_COVERAGE_LOOKBACK_SESSIONS: Final = 20
FLOW_COVERAGE_REFERENCE_QUANTILE: Final = 0.10
FLOW_COVERAGE_MIN_RATIO: Final = 0.50
PURGE_SESSIONS: Final = 20

STOCK_FEATURE_COLUMNS: Final = tuple(PRICE_FEATURES)
FLOW_VALUE_COLUMNS: Final = (
    "flow_rate_pct",
    "signed_log_fund_flow_millions",
    "log_assets",
    "shares_change_pct",
)
EDGE_FEATURE_COLUMNS: Final = (
    "holding_weight_fraction",
    "snapshot_age_sessions_scaled",
    "current_flow_observed",
)

TARGET_COLUMNS: Final = tuple(
    name
    for horizon in (5, 20)
    for name in (
        f"return_{horizon}d_pct",
        f"upside_{horizon}d_pct",
        f"loss_{horizon}d_pct",
        f"benchmark_excess_return_{horizon}d_pct",
        f"benchmark_upside_capture_{horizon}d_pct",
        f"benchmark_downside_defense_{horizon}d_pct",
    )
)

DIRECTION_TARGET_INDICES: Final = tuple(
    TARGET_COLUMNS.index(name)
    for name in (
        "return_5d_pct",
        "benchmark_excess_return_5d_pct",
        "return_20d_pct",
        "benchmark_excess_return_20d_pct",
    )
)

COMMON_FLOW_TARGET_INDICES: Final = tuple(
    TARGET_COLUMNS.index(name)
    for name in (
        "return_5d_pct",
        "upside_5d_pct",
        "loss_5d_pct",
        "return_20d_pct",
        "upside_20d_pct",
        "loss_20d_pct",
    )
)

ROTATION_FLOW_TARGET_INDICES: Final = tuple(
    TARGET_COLUMNS.index(name)
    for name in (
        "benchmark_excess_return_5d_pct",
        "benchmark_upside_capture_5d_pct",
        "benchmark_downside_defense_5d_pct",
        "benchmark_excess_return_20d_pct",
        "benchmark_upside_capture_20d_pct",
        "benchmark_downside_defense_20d_pct",
    )
)

DEFAULT_PANEL: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/panel.sqlite3"
)
DEFAULT_FLOW_CACHE: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/flow_cache.sqlite3"
)
DEFAULT_BASE_DATABASE: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/"
    "daily_observations.sqlite3"
)
DEFAULT_INCREMENTAL_DATABASE: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/incremental/"
    "normalized/daily_observations.sqlite3"
)
DEFAULT_FLOW_SOURCE_ROOT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v1"
)
DEFAULT_OUTPUT_ROOT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6"
)
DEFAULT_FLOW_BACKFILL_DATABASE: Final = (
    DEFAULT_FLOW_SOURCE_ROOT
    / "source_backfill_20260715_20260721/normalized/daily_observations.sqlite3"
)
DEFAULT_REPAIRED_FLOW_CACHE: Final = (
    DEFAULT_FLOW_SOURCE_ROOT / "repaired_flow_cache_20260715_20260722.sqlite3"
)

SMOKE_START_DATE: Final = "2026-06-29"
SMOKE_END_DATE: Final = "2026-07-29"
SMOKE_SYMBOLS: Final = (
    "AAPL",
    "AMZN",
    "AVGO",
    "BRK-B",
    "GOOG",
    "GOOGL",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
)
