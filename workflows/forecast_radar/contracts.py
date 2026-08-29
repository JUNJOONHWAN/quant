"""Frozen contracts for the Forecast RADAR shadow product."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from training.quant_flow_graph.contracts import TARGET_COLUMNS
from training.quant_flow_graph_v11_r2.contracts import TIMING_CONTRACT


SCHEMA_VERSION: Final = "quant.forecast_radar.v1"
MODEL_SCHEMA_VERSION: Final = "quant.forecast_radar.model_bundle.v1"
RUN_SCHEMA_VERSION: Final = "quant.forecast_radar.daily_run.v1"
TARGET_NAMES: Final = tuple(TARGET_COLUMNS)

COVERAGE_VALIDATED_CORE: Final = "VALIDATED_CORE"
COVERAGE_GENERAL_SHADOW: Final = "GENERAL_UNIVERSE_SHADOW"

DEFAULT_DATA_ROOT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/forecast_radar"
)
DEFAULT_MODEL_ROOT: Final = DEFAULT_DATA_ROOT / "model"
DEFAULT_LIVE_ROOT: Final = DEFAULT_DATA_ROOT / "live"
DEFAULT_RESEARCH_ROOT: Final = DEFAULT_DATA_ROOT / "research"

DEFAULT_BASE_DATABASE: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/"
    "daily_observations.sqlite3"
)
DEFAULT_INCREMENTAL_DATABASE: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/incremental/"
    "normalized/daily_observations.sqlite3"
)
DEFAULT_GRAPH_DATASET_ROOT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/"
    "full_20180102_20260729_allpanel"
)
DEFAULT_PHASE_A_ROOT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v11/r2_phase_a"
)
DEFAULT_V16_ROOT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v16/"
    "full_etf_identity_latent_walk_forward"
)
DEFAULT_V19_RECEIPT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v19/"
    "global_drift_edge_audit/v19_global_drift_receipt.json"
)
V19_RECEIPT_SHA256: Final = (
    "b2996c25d97d2b2eb3625fc4ec61981acabee496bba774d8627af7c2ab22006e"
)
DEFAULT_ETFRADAR_ROOT: Final = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/ETFRADAR"
)

# Frozen, source-linked interpretation contract. These values come from the
# immutable v19 receipt above. They describe where historical OOS information
# survived; they are not current-price forecasts or a trading policy.
INFORMATION_VALUE_EVIDENCE: Final = {
    "status": "USEFUL_FOR_DISTRIBUTION_AND_POTENTIAL_NOT_TRADE_DIRECTION",
    "validated_paths": ["DISTRIBUTION_FORECAST", "UPSIDE_DOWNSIDE_POTENTIAL"],
    "not_validated": [
        "PRECISE_RETURN_DIRECTION",
        "CURRENT_TIMING_EDGE",
        "STABLE_BASKET_ALPHA",
        "AUTOMATED_TRADING_POLICY",
    ],
    "historical_oos": {
        "years": [2021, 2022, 2023, 2024, 2025, 2026],
        "date_count": 1383,
        "row_count": 673081,
        "historical_oos_not_fresh_forward_lockbox": True,
    },
    "recent_oos_2026": {
        "upside_5d": {
            "mean_daily_rank_ic": 0.28998382241371395,
            "top_minus_bottom_realized_spread_pct_points": 5.36167722028,
        },
        "upside_20d": {
            "mean_daily_rank_ic": 0.29849062572014706,
            "top_minus_bottom_realized_spread_pct_points": 11.781394920420114,
        },
        "loss_5d": {
            "mean_daily_rank_ic": 0.2878425943859433,
            "top_minus_bottom_realized_spread_pct_points": 4.5046570694889345,
        },
        "loss_20d": {
            "mean_daily_rank_ic": 0.31846394903516,
            "top_minus_bottom_realized_spread_pct_points": 8.640680180912117,
        },
        "return_5d_mean_daily_rank_ic": -0.0072500848339163695,
        "return_20d_mean_daily_rank_ic": -0.05219838618484046,
        "mean_mae_improvement_vs_price_only_pct": 0.41784194155017335,
    },
    "etf_flow_incremental_evidence": {
        "mae_beats_price_only_target_count": 12,
        "target_count": 12,
        "block_bootstrap_ci_positive_target_count": 12,
        "interpretation": "MEASURABLE_BUT_SMALL_AND_DIMINISHING_IN_2026",
    },
    "source": {
        "path": str(DEFAULT_V19_RECEIPT),
        "sha256": V19_RECEIPT_SHA256,
    },
}

MIN_PRICE_USD: Final = 3.0
MIN_DOLLAR_VOLUME_USD: Final = 1_000_000.0
MIN_PRICE_HISTORY_ROWS: Final = 200
GENERAL_EXCHANGES: Final = frozenset(
    {"NASDAQ", "NYSE", "AMEX", "NYSE ARCA", "NASDAQ GLOBAL SELECT"}
)

LATENT_COMPONENTS: Final = 32
LATENT_STATES: Final = ("current", "mean5", "mean20", "innovation", "convergence")
RESIDUAL_SHRINKAGE: Final = 0.25
RANDOM_SEED: Final = 20260828

CATBOOST_VERSION: Final = "1.2.10"
CATBOOST_PARAMETERS: Final = {
    "loss_function": "MultiRMSE",
    "eval_metric": "MultiRMSE",
    "iterations": 256,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 20.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.80,
    "rsm": 0.70,
    "leaf_estimation_iterations": 1,
    "random_seed": RANDOM_SEED,
    "task_type": "CPU",
    "allow_writing_files": False,
    "verbose": False,
}

CORE_VALIDATION_STATEMENT: Final = (
    "Historical OOS covers the listed predominantly PIT SPY/QQQ-member symbols. "
    "It does not validate the general US-stock extrapolation tier."
)
