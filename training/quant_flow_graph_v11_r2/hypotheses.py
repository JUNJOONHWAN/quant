"""Pre-register the interpretable v11-R2 hypothesis tournament."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import (
    FORBIDDEN_DIFFUSION_INPUTS,
    FORBIDDEN_TRANSFORMS,
    HYPOTHESIS_REGISTRY_SCHEMA_VERSION,
    TIMING_CONTRACT,
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def specification() -> dict[str, Any]:
    """Return the timestamp-free specification whose digest freezes Phase B."""

    hypotheses = [
        {
            "id": "H1_DRIFT_DIRECTION",
            "claim": "absolute market-wide Flow Drift improves broad-market 5d/20d direction and drawdown forecasts beyond equal-capacity price-only",
            "primary_targets": [
                "SPY_return_5d",
                "SPY_return_20d",
                "QQQ_return_5d",
                "QQQ_return_20d",
                "broad_market_drawdown_5d_20d",
            ],
        },
        {
            "id": "H2_DIFFUSION_PERSISTENCE",
            "claim": "independent breadth, reach, and persistence distinguish continuation from anchor-only Flow",
            "primary_targets": ["continuation_5d_20d", "downside_defense_5d_20d"],
        },
        {
            "id": "H3_ANCHOR_CLUSTER_STOCK_SEQUENCE",
            "claim": "Flow propagates anchor to cluster to holdings-weighted stock breadth in temporal order",
            "primary_targets": ["lead_lag_order", "time_shuffle_degradation"],
        },
        {
            "id": "H4_INDIRECT_STOCK_EDGE",
            "claim": "all-ETF cluster paths add stock information beyond direct holding ETFs",
            "primary_targets": [
                "relative_return_5d_20d",
                "mfe_5d_20d",
                "mae_5d_20d",
            ],
        },
        {
            "id": "H5_BEARISH_AVOIDANCE",
            "claim": "negative Drift with broad internal propagation improves bottom-basket avoidance",
            "primary_targets": ["mae", "drawdown", "cvar", "avoidance_utility"],
        },
        {
            "id": "H6_INDEPENDENT_CONVERGENCE",
            "claim": "family-adjusted convergence survives duplicate ETF adjustment",
            "primary_targets": ["independent_breadth", "duplicate_off_placebo"],
        },
        {
            "id": "H7_EFFECTIVE_EXPOSURE_SIGN",
            "claim": "typed leverage inverse defensive signs outperform raw special-product Flow",
            "primary_targets": ["sign_stability", "inverse_sign_placebo"],
        },
        {
            "id": "H8_PHASE_TRANSITION",
            "claim": "Drift-Diffusion state transitions forecast continuation, exhaustion, and reversal",
            "primary_targets": ["stage_transition_logloss", "resolution_5d_20d"],
        },
        {
            "id": "H9_FLOW_BEYOND_PRICE_BREADTH",
            "claim": "Flow diffusion retains incremental value after price-breadth controls",
            "primary_targets": ["flow_incremental_oos", "price_breadth_control"],
        },
        {
            "id": "H10_CALIBRATION",
            "claim": "probability and interval heads remain calibrated by regime and horizon",
            "primary_targets": ["brier", "ece", "interval_coverage", "sharpness"],
        },
    ]
    controls = [
        "price_only_equal_capacity",
        "mask_coverage_only",
        "drift_only",
        "diffusion_only",
        "drift_plus_diffusion",
        "raw_flow_no_graph",
        "direct_holdings_only",
        "legacy_friction_only_ddm",
        "etf_axis_flow_shuffle",
        "date_block_flow_shuffle",
        "five_session_lag",
        "twenty_session_lag",
        "within_cluster_shuffle",
        "holdings_edge_shuffle",
        "graph_rewiring",
        "inverse_sign_placebo",
        "special_channel_off",
        "duplicate_adjustment_off",
        "future_flow_negative_control",
    ]
    return {
        "schema_version": HYPOTHESIS_REGISTRY_SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_PHASE_B",
        "timing_contract": TIMING_CONTRACT,
        "definitions": {
            "drift": "market-wide absolute fund-flow direction; never date-centered",
            "diffusion": "propagation from broad anchors through independent families and clusters to stocks",
            "legacy_diffusion": "friction/counter-flow dispersion; forbidden as the new diffusion target",
        },
        "forbidden_inputs": list(FORBIDDEN_DIFFUSION_INPUTS),
        "forbidden_transforms": list(FORBIDDEN_TRANSFORMS),
        "hypotheses": hypotheses,
        "required_controls": controls,
        "walk_forward": {
            "outer_split": "calendar_year_by_signal_date",
            "purge_sessions": 20,
            "embargo_sessions": 20,
            "same_date_train_test_split": "FORBIDDEN",
            "fit_inside_train_fold": [
                "scaler",
                "family_clustering",
                "taxonomy",
                "lead_lag_edges",
            ],
            "researcher_contaminated_secondary_oos": [2024, 2025],
            "final_confirmation": "prospective_shadow_60_then_120_sessions",
        },
        "statistics": [
            "moving_block_bootstrap",
            "newey_west_hac",
            "white_reality_check_or_hansen_spa",
            "hypothesis_family_fdr",
        ],
        "decision_gate": {
            "FORECAST_ALPHA": [
                "drift_plus_diffusion_beats_equal_capacity_price_only",
                "actual_flow_beats_shuffled_lagged_rewired",
                "all_etf_diffusion_beats_direct_holdings_only",
                "pre_registered_target_family_consistency",
                "same_sign_in_at_least_two_historical_outer_folds",
                "survives_multiple_testing_correction",
                "positive_after_cost_basket_utility",
                "prospective_calibration_pass",
            ],
            "AVOIDANCE_FILTER": "only downside family passes",
            "BASKET_CONSTRUCTION": "only cross-sectional rank family passes",
            "FAIL": "no pre-registered utility family passes",
        },
    }


def build_registry(*, generated_at_utc: str) -> dict[str, Any]:
    spec = specification()
    digest = hashlib.sha256(canonical_json(spec)).hexdigest()
    return {
        **spec,
        "generated_at_utc": generated_at_utc,
        "specification_sha256": digest,
    }


def write_registry(path: Path, *, generated_at_utc: str) -> dict[str, Any]:
    registry = build_registry(generated_at_utc=generated_at_utc)
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return registry
