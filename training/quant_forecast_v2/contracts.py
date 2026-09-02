"""Shared contracts for the quantitative Forecast v2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


PANEL_SCHEMA_VERSION: Final = "quant.spy_qqq_forecast_panel.v2"
MODEL_SCHEMA_VERSION: Final = "quant.spy_qqq_forecast_model.v2"
REPORT_SCHEMA_VERSION: Final = "quant.spy_qqq_forecast_evaluation.v2"
TIMING_CONTRACT: Final = (
    "signal session T / price and fundamentals through T-1 close / "
    "ETF Flow effective session exactly T-2 and captured before T open / "
    "flow visibility checked at T; all other sources available by T-1"
)
HORIZONS: Final = (5, 20)


IDENTITY_COLUMNS: Final = (
    "signal_date",
    "price_date",
    "flow_date",
    "legacy_flow_date",
    "symbol",
    "benchmark",
    "is_spy_member",
    "is_qqq_member",
    "membership_source",
)

PRICE_FEATURES: Final = (
    "is_spy_member",
    "is_qqq_member",
    "ret_1d",
    "ret_2d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "realized_vol_5d",
    "realized_vol_20d",
    "realized_vol_60d",
    "atr_14d_pct",
    "rsi_14d",
    "close_to_sma20_pct",
    "close_to_sma50_pct",
    "close_to_sma200_pct",
    "drawdown_20d_pct",
    "drawdown_60d_pct",
    "drawdown_252d_pct",
    "log_dollar_volume_20d",
    "volume_ratio_20d",
    "benchmark_ret_5d",
    "benchmark_ret_20d",
    "benchmark_ret_60d",
    "relative_ret_5d",
    "relative_ret_20d",
    "relative_ret_60d",
    "beta_20d",
    "beta_60d",
    "corr_20d",
    "corr_60d",
    "spy_ret_5d",
    "qqq_ret_5d",
    "spy_vol_20d",
    "qqq_vol_20d",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "momentum_rank_20d",
    "volatility_rank_20d",
    "size_rank",
)

BENCHMARK_FLOW_FEATURES: Final = (
    "spy_flow_rate_t2",
    "spy_flow_rate_5d",
    "spy_flow_rate_20d",
    "spy_flow_z60",
    "qqq_flow_rate_t2",
    "qqq_flow_rate_5d",
    "qqq_flow_rate_20d",
    "qqq_flow_z60",
    "benchmark_flow_rate_t2",
    "benchmark_flow_rate_5d",
    "benchmark_flow_rate_20d",
    "benchmark_flow_z60",
    "qqq_minus_spy_flow_t2",
    "price_flow_interaction_5d",
)

LEGACY_T3_FLOW_FEATURES: Final = tuple(
    column.replace("_t2", "_t3")
    for column in BENCHMARK_FLOW_FEATURES
    if column.endswith("_t2")
) + (
    "spy_flow_rate_5d_t3_cutoff",
    "spy_flow_rate_20d_t3_cutoff",
    "spy_flow_z60_t3_cutoff",
    "qqq_flow_rate_5d_t3_cutoff",
    "qqq_flow_rate_20d_t3_cutoff",
    "qqq_flow_z60_t3_cutoff",
    "benchmark_flow_rate_5d_t3_cutoff",
    "benchmark_flow_rate_20d_t3_cutoff",
    "benchmark_flow_z60_t3_cutoff",
    "price_flow_interaction_5d_t3_cutoff",
)

FULL_ETF_FLOW_FEATURES: Final = (
    "all_etf_exposure_count",
    "all_etf_flow_observed_count",
    "all_etf_flow_count_coverage",
    "all_etf_holding_weight_sum",
    "all_etf_observed_weight_sum",
    "all_etf_flow_weight_coverage",
    "all_etf_flow_positive_count",
    "all_etf_flow_negative_count",
    "all_etf_flow_breadth",
    "all_etf_flow_net",
    "all_etf_flow_positive",
    "all_etf_flow_negative",
    "all_etf_flow_gross",
    "all_etf_flow_max_abs_contribution",
    "all_etf_flow_top3_abs_share",
    "all_etf_flow_hhi",
    "all_etf_flow_net_5d",
    "all_etf_flow_net_20d",
    "all_etf_flow_net_z60",
    "all_etf_flow_rank",
    "all_etf_flow_breadth_rank",
    "all_etf_flow_coverage_rank",
)

FMP_FUNDAMENTAL_FEATURES: Final = (
    "log_market_cap",
    "market_cap_to_20d_avg",
    "revenue_yoy",
    "net_margin",
    "operating_margin",
    "gross_margin",
    "free_cash_flow_margin",
    "operating_cash_flow_margin",
    "debt_to_assets",
    "cash_to_assets",
    "current_ratio",
    "financial_statement_age_days",
    "fundamental_coverage",
)

FEATURE_GROUPS: Final = {
    "price": PRICE_FEATURES,
    "benchmark_flow": BENCHMARK_FLOW_FEATURES,
    "legacy_t3_flow": LEGACY_T3_FLOW_FEATURES,
    "all_etf_flow": FULL_ETF_FLOW_FEATURES,
    "fmp_fundamentals": FMP_FUNDAMENTAL_FEATURES,
}

TARGET_COLUMNS: Final = tuple(
    item
    for horizon in HORIZONS
    for item in (
        f"return_{horizon}d_pct",
        f"upside_{horizon}d_pct",
        f"loss_{horizon}d_pct",
    )
)


@dataclass(frozen=True)
class TimingRow:
    """One look-ahead-safe reference row."""

    signal_date: str
    price_date: str
    flow_date: str
    price_position: int


def model_feature_columns(variant: str) -> tuple[str, ...]:
    """Return the ordered feature contract for an ablation variant."""

    variants = {
        "price": ("price",),
        "price_benchmark_flow_t3": ("price", "legacy_t3_flow"),
        "price_benchmark_flow": ("price", "benchmark_flow"),
        "price_all_etf_flow": ("price", "benchmark_flow", "all_etf_flow"),
        "full": (
            "price",
            "benchmark_flow",
            "all_etf_flow",
            "fmp_fundamentals",
        ),
    }
    try:
        groups = variants[variant]
    except KeyError as exc:
        raise ValueError(f"unknown feature variant: {variant}") from exc
    return tuple(column for group in groups for column in FEATURE_GROUPS[group])
