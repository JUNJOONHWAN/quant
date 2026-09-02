"""Frozen contracts for the v11-R2 fund-flow research pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Final


PHASE_A_AUDIT_SCHEMA_VERSION: Final = "quant.etf_flow_v11_r2.phase_a_audit.v1"
FAMILY_REGISTRY_SCHEMA_VERSION: Final = "quant.etf_flow_v11_r2.family_registry.v1"
HYPOTHESIS_REGISTRY_SCHEMA_VERSION: Final = (
    "quant.etf_flow_v11_r2.hypothesis_registry.v1"
)
EVENT_CUBE_SCHEMA_VERSION: Final = "quant.etf_flow_v11_r2.event_cube.v1"

TIMING_CONTRACT: Final = (
    "decision/signal session T; ETF and stock price/liquidity through T-1; "
    "Massive ETF Flow effective session exactly T-2 and available by T; "
    "PIT holdings available no later than T-1; no silent T-3 Flow fill"
)

MIN_ASSETS_USD: Final = 50_000_000.0
MIN_DOLLAR_VOLUME_USD: Final = 1_000_000.0
MIN_PRICE_USD: Final = 3.0
ACTIVE_LOOKBACK_SESSIONS: Final = 60
STALE_AFTER_SESSIONS: Final = 5
NEW_LIFECYCLE_SESSIONS: Final = 20

FORBIDDEN_DIFFUSION_INPUTS: Final = (
    "48_MASSIVE_ACCUM_CLUSTER.flow_breadth",
)
FORBIDDEN_TRANSFORMS: Final = (
    "cross_sectional_date_centering_of_absolute_flow",
    "date_mean_subtraction_of_fund_flow",
)

DEFAULT_SOURCE_DATABASE: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/"
    "daily_observations.sqlite3"
)
DEFAULT_ETFRADAR_ROOT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/ETFRADAR"
)
DEFAULT_OUTPUT_ROOT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/r2_phase_a"
)

AUDIT_FILENAME: Final = "v11_r2_pit_eligibility_audit.json"
FAMILY_REGISTRY_FILENAME: Final = "v11_r2_etf_family_exposure_registry.sqlite3"
HYPOTHESIS_REGISTRY_FILENAME: Final = (
    "v11_r2_drift_diffusion_hypothesis_registry.json"
)
EVENT_CUBE_FILENAME: Final = "v11_r2_flow_event_cube.sqlite3"
EVENT_CUBE_MANIFEST_FILENAME: Final = "v11_r2_flow_event_cube_manifest.json"

ANCHOR_TICKERS: Final = (
    "SPY",
    "QQQ",
    "VTI",
    "RSP",
    "IWM",
    "DIA",
)
