"""Build source-preserving Qwen SFT rows from quant analysis packets.

The builder never uses a future return as an input or target.  Its first-stage
targets are deterministic, auditable interpretations of facts visible in a
``quant.analysis_packet.v3`` packet.  Expert/teacher labels can be introduced
later as a separate contract instead of silently fabricating them here.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import heapq
import json
import math
import os
import shutil
import sqlite3
import statistics
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from quant_dataset.point_in_time import (
    ETF_CONSTITUENT_POLICY_ID,
    ETF_FLOW_POLICY_ID,
    US_EQUITY_SESSION_SQL,
)
from quant_dataset.etf_flow_exposure import (
    ETF_CONSTITUENT_FLOW_POLICY_ID,
    flow_quality_reasons,
)
from training.quant_llm import DATASET_CONTRACT_VERSION, DATASET_SCHEMA_VERSION


DEFAULT_DATABASE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/"
    "daily_observations.sqlite3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/"
    "qwen3_8b_candidate_v2"
)
REQUIRED_PACKET_SCHEMA = "quant.analysis_packet.v3"
ETF_LIQUIDITY_POLICY_ID = "asof_etf_liquidity_20s_v1"
ETF_LIQUIDITY_WINDOW_SESSIONS = 20
DEFAULT_MIN_ETF_OBSERVED_SESSIONS = 10
DEFAULT_MIN_ETF_NONZERO_VOLUME_RATIO = 0.75
DEFAULT_MIN_ETF_MEDIAN_DOLLAR_VOLUME = 1_000_000.0
MAX_PROMPT_PRICE_ROWS = 8
MAX_PROMPT_FLOW_ROWS = 5
MAX_PROMPT_RELATION_ROWS = 4
MAX_PROMPT_EXPOSURE_ROWS = 3
MAX_TARGET_EXPOSURE_ROWS = 3
MAX_PROMPT_PROVENANCE_HASHES = 2
RESPONSE_KEYS = (
    "facts",
    "interpretation",
    "counter_evidence",
    "unknowns",
    "regime",
    "confidence",
    "conclusion",
)
FORBIDDEN_PACKET_KEYS = {
    "future_return",
    "forward_return",
    "realized_future_return",
    "future_price",
    "forward_price",
    "next_day_return",
    "target_return",
    "trade_outcome",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_date(value: object) -> str:
    return date.fromisoformat(str(value)).isoformat()


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _as_number(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def resolve_packet_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.extend(matches)
    unique = sorted({path.expanduser().resolve() for path in paths if path.is_file()})
    if not unique:
        raise ValueError("no packet JSONL files matched --packets")
    return unique


def load_trading_sessions(database_path: Path) -> Tuple[str, ...]:
    database = Path(database_path).expanduser()
    if not database.is_file():
        raise FileNotFoundError("daily observation database not found: {}".format(database))
    connection = sqlite3.connect("file:{}?mode=ro".format(database), uri=True)
    try:
        rows = connection.execute(US_EQUITY_SESSION_SQL).fetchall()
    finally:
        connection.close()
    sessions = tuple(_canonical_date(row[0]) for row in rows)
    if not sessions:
        raise ValueError("daily observation database has no trading sessions")
    return sessions


def _embargo_cutoff(
    sessions: Sequence[str], next_split_start: str, embargo_sessions: int
) -> str:
    prior = [session for session in sessions if session < next_split_start]
    if len(prior) <= embargo_sessions:
        raise ValueError(
            "not enough sessions before {} for {}-session embargo".format(
                next_split_start, embargo_sessions
            )
        )
    return prior[-embargo_sessions - 1]


def split_contract(
    sessions: Sequence[str],
    validation_start: str,
    test_start: str,
    embargo_sessions: int,
) -> dict:
    validation = _canonical_date(validation_start)
    test = _canonical_date(test_start)
    if validation >= test:
        raise ValueError("validation_start must be before test_start")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative")
    return {
        "train": {
            "start": sessions[0],
            "end": _embargo_cutoff(sessions, validation, embargo_sessions),
        },
        "validation": {
            "start": validation,
            "end": _embargo_cutoff(sessions, test, embargo_sessions),
        },
        "test": {"start": test, "end": sessions[-1]},
        "embargo_sessions": embargo_sessions,
        "purged_ranges": [
            {
                "after": _embargo_cutoff(sessions, validation, embargo_sessions),
                "before": validation,
            },
            {
                "after": _embargo_cutoff(sessions, test, embargo_sessions),
                "before": test,
            },
        ],
    }


def assign_split(as_of_date: str, contract: Mapping[str, Any]) -> Optional[str]:
    as_of = _canonical_date(as_of_date)
    for name in ("train", "validation", "test"):
        bounds = contract[name]
        if bounds["start"] <= as_of <= bounds["end"]:
            return name
    return None


def _forbidden_key_path(value: Any, prefix: str = "packet") -> Optional[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_PACKET_KEYS or lowered.startswith("future_"):
                return "{}.{}".format(prefix, key)
            found = _forbidden_key_path(item, "{}.{}".format(prefix, key))
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _forbidden_key_path(item, "{}[{}]".format(prefix, index))
            if found:
                return found
    return None


def validate_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != REQUIRED_PACKET_SCHEMA:
        raise ValueError(
            "packet {} uses {}; {} is required".format(
                packet.get("packet_id"), packet.get("schema_version"), REQUIRED_PACKET_SCHEMA
            )
        )
    as_of = _canonical_date(packet.get("as_of_date"))
    forbidden = _forbidden_key_path(packet)
    if forbidden:
        raise ValueError("future/target field is forbidden in SFT input: {}".format(forbidden))

    for row in packet.get("history") or []:
        if _canonical_date(row.get("trade_date")) > as_of:
            raise ValueError("price history exceeds packet as_of_date")

    flow = packet.get("etf_flow") or {}
    policy = flow.get("availability_policy") or {}
    if policy.get("policy_id") != ETF_FLOW_POLICY_ID:
        raise ValueError("ETF Flow packet is missing the required session-lag policy")
    effective_dates = set()
    for row in flow.get("observations") or []:
        available = _canonical_date(row.get("training_available_session_date"))
        if row.get("training_availability_policy_id") != ETF_FLOW_POLICY_ID:
            raise ValueError("ETF Flow observation policy id mismatch")
        if available > as_of:
            raise ValueError("ETF Flow observation is not available at packet as_of_date")
        if _canonical_date(row.get("effective_date")) > as_of:
            raise ValueError("ETF Flow effective_date exceeds packet as_of_date")
        if _canonical_date(row.get("processed_date")) > as_of:
            raise ValueError("ETF Flow processed_date exceeds packet as_of_date")
        if row.get("available_at_date") != available:
            raise ValueError("ETF Flow available_at_date is not the derived training date")
        effective = _canonical_date(row.get("effective_date"))
        if effective in effective_dates:
            raise ValueError("duplicate ETF Flow effective_date survived packet dedupe")
        effective_dates.add(effective)

    constituents = packet.get("etf_constituents") or {}
    constituent_policy = constituents.get("availability_policy") or {}
    if constituent_policy.get("policy_id") != ETF_CONSTITUENT_POLICY_ID:
        raise ValueError("ETF constituent packet is missing next-session policy")
    for row in list(constituents.get("constituents") or []) + list(
        constituents.get("etf_memberships") or []
    ):
        if _canonical_date(row.get("effective_date")) > as_of:
            raise ValueError("ETF constituent effective_date exceeds packet as_of_date")
        if _canonical_date(row.get("available_date")) > as_of:
            raise ValueError("ETF constituent available_date exceeds packet as_of_date")
        if row.get("training_availability_policy_id") != ETF_CONSTITUENT_POLICY_ID:
            raise ValueError("ETF constituent observation policy id mismatch")

    exposure = packet.get("etf_flow_to_constituent") or {}
    if exposure.get("policy_id") != ETF_CONSTITUENT_FLOW_POLICY_ID:
        raise ValueError("ETF-to-constituent Flow packet policy is missing")
    for row in exposure.get("rows") or []:
        if _canonical_date(row.get("membership_effective_date")) > as_of:
            raise ValueError("ETF membership exposure effective_date exceeds as-of")
        if _canonical_date(row.get("membership_available_date")) > as_of:
            raise ValueError("ETF membership exposure is not available as-of")
        if _canonical_date(row.get("flow_effective_date")) > as_of:
            raise ValueError("constituent ETF Flow effective_date exceeds as-of")
        if _canonical_date(row.get("flow_processed_date")) > as_of:
            raise ValueError("constituent ETF Flow processed_date exceeds as-of")
        if _canonical_date(row.get("flow_training_available_session_date")) > as_of:
            raise ValueError("constituent ETF Flow exposure is not available as-of")
        if row.get("flow_availability_policy_id") != ETF_FLOW_POLICY_ID:
            raise ValueError("constituent ETF Flow availability policy mismatch")


def _preferred_price_rows(packet: Mapping[str, Any]) -> List[dict]:
    preferred = {"fmp": 0, "massive": 1}
    result = []
    for day in packet.get("history") or []:
        candidates = [
            row for row in day.get("sources") or [] if _as_number(row.get("close"))
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (preferred.get(str(row.get("source")), 9), str(row.get("source"))))
        selected = candidates[0]
        raw_close = _as_number(selected.get("close"))
        result.append(
            {
                "trade_date": _canonical_date(day["trade_date"]),
                "source": selected.get("source"),
                "close": raw_close,
                "raw_close": raw_close,
                "price_basis": "raw_close_pit_conservative",
                "volume": _as_number(selected.get("volume")),
            }
        )
    return result


def _return_pct(values: Sequence[float], periods: int) -> Optional[float]:
    if len(values) <= periods or values[-periods - 1] <= 0:
        return None
    return (values[-1] / values[-periods - 1] - 1.0) * 100.0


def _price_facts(rows: Sequence[Mapping[str, Any]]) -> dict:
    closes = [float(row["close"]) for row in rows if row.get("close") is not None]
    log_returns = [math.log(right / left) for left, right in zip(closes, closes[1:]) if left > 0 and right > 0]
    volatility = (
        statistics.stdev(log_returns) * math.sqrt(252.0) * 100.0
        if len(log_returns) >= 5
        else None
    )
    peak = None
    max_drawdown = None
    for close in closes:
        peak = close if peak is None else max(peak, close)
        drawdown = (close / peak - 1.0) * 100.0
        max_drawdown = drawdown if max_drawdown is None else min(max_drawdown, drawdown)
    return {
        "observed_sessions": len(closes),
        "latest_close": _round(closes[-1]) if closes else None,
        "return_1_session_pct": _round(_return_pct(closes, 1)),
        "return_5_session_pct": _round(_return_pct(closes, 5)),
        "return_20_session_pct": _round(_return_pct(closes, 20)),
        "annualized_realized_volatility_pct": _round(volatility),
        "max_drawdown_in_packet_pct": _round(max_drawdown),
    }


def _liquidity_facts(rows: Sequence[Mapping[str, Any]], window_sessions: int = 20) -> dict:
    window = list(rows[-window_sessions:])
    volumes = [_as_number(row.get("volume")) for row in window]
    nonzero = [value for value in volumes if value is not None and value > 0]
    dollar_volumes = [
        float(row["close"]) * float(volume)
        for row, volume in zip(window, volumes)
        if row.get("close") is not None and volume is not None and volume > 0
    ]
    return {
        "policy_id": ETF_LIQUIDITY_POLICY_ID,
        "window_sessions": window_sessions,
        "observed_sessions": len(window),
        "latest_trade_date": window[-1].get("trade_date") if window else None,
        "latest_volume": _round(volumes[-1]) if volumes else None,
        "nonzero_volume_sessions": len(nonzero),
        "nonzero_volume_ratio": _round(len(nonzero) / len(window), 6) if window else 0.0,
        "median_share_volume": _round(statistics.median(nonzero)) if nonzero else None,
        "median_dollar_volume": _round(statistics.median(dollar_volumes)) if dollar_volumes else None,
    }


def packet_eligibility(
    packet: Mapping[str, Any],
    *,
    min_etf_observed_sessions: int = DEFAULT_MIN_ETF_OBSERVED_SESSIONS,
    min_etf_nonzero_volume_ratio: float = DEFAULT_MIN_ETF_NONZERO_VOLUME_RATIO,
    min_etf_median_dollar_volume: float = DEFAULT_MIN_ETF_MEDIAN_DOLLAR_VOLUME,
) -> dict:
    """Apply only as-of ETF quality gates; never use today's active list."""

    price_rows = _preferred_price_rows(packet)
    relation = packet.get("etf_constituents") or {}
    flow_rows = list((packet.get("etf_flow") or {}).get("observations") or [])
    is_etf = bool(flow_rows) or bool(relation.get("constituents")) or bool(
        relation.get("constituent_snapshot_date")
    )
    liquidity = _liquidity_facts(price_rows, ETF_LIQUIDITY_WINDOW_SESSIONS)
    reasons = []
    if liquidity["latest_trade_date"] != packet.get("as_of_date"):
        reasons.append("security_no_same_session_price")
    latest_volume = _as_number(liquidity.get("latest_volume"))
    if latest_volume is None or latest_volume <= 0:
        reasons.append("security_zero_or_missing_latest_volume")
    if liquidity["observed_sessions"] < 5:
        reasons.append("security_fewer_than_5_trailing_sessions")
    closes = [
        float(row["close"])
        for row in price_rows
        if _as_number(row.get("close")) is not None and float(row["close"]) > 0
    ]
    one_session_returns = [right / left - 1.0 for left, right in zip(closes, closes[1:])]
    max_abs_return = max((abs(value) for value in one_session_returns), default=0.0)
    if max_abs_return >= 0.45:
        reasons.append("raw_price_discontinuity_ge_45pct_without_pit_corporate_action")
    if is_etf:
        if liquidity["observed_sessions"] < min_etf_observed_sessions:
            reasons.append("etf_insufficient_trailing_sessions")
        if float(liquidity["nonzero_volume_ratio"] or 0.0) < min_etf_nonzero_volume_ratio:
            reasons.append("etf_low_nonzero_volume_ratio")
        median_dollar_volume = _as_number(liquidity.get("median_dollar_volume"))
        if median_dollar_volume is None or median_dollar_volume < min_etf_median_dollar_volume:
            reasons.append("etf_low_median_dollar_volume")
        if flow_rows:
            latest_flow = flow_rows[-1]
            fund_flow = _as_number(latest_flow.get("fund_flow"))
            assets = _as_number(latest_flow.get("assets"))
            nav = _as_number(latest_flow.get("nav"))
            shares = _as_number(latest_flow.get("shares_outstanding"))
            estimated_assets = (
                assets
                if assets is not None and assets > 0
                else nav * shares
                if nav is not None and nav > 0 and shares is not None and shares > 0
                else None
            )
            flow_rate = (
                fund_flow / estimated_assets * 100.0
                if fund_flow is not None and estimated_assets not in (None, 0.0)
                else None
            )
            reasons.extend(
                flow_quality_reasons(
                    as_of_date=str(packet.get("as_of_date") or ""),
                    effective_date=latest_flow.get("effective_date"),
                    processed_date=latest_flow.get("processed_date"),
                    available_date=latest_flow.get(
                        "training_available_session_date"
                    ),
                    flow_to_net_assets_pct=flow_rate,
                )
            )
    return {
        "eligible": not reasons,
        "is_etf_as_of_packet": is_etf,
        "liquidity": liquidity,
        "max_absolute_raw_one_session_return": _round(max_abs_return),
        "reasons": reasons,
        "delisting_policy": (
            "retain pre-delisting observations; require same-session price; "
            "never filter historical rows with a present-day active list"
        ),
    }


def _flow_facts(observations: Sequence[Mapping[str, Any]]) -> dict:
    numeric = [float(value) for value in (_as_number(row.get("fund_flow")) for row in observations) if value is not None]
    latest = observations[-1] if observations else None
    latest_flow = _as_number(latest.get("fund_flow")) if latest else None
    latest_assets = _as_number(latest.get("assets")) if latest else None
    latest_nav = _as_number(latest.get("nav")) if latest else None
    latest_shares = (
        _as_number(latest.get("shares_outstanding")) if latest else None
    )
    if latest_assets is None or latest_assets <= 0:
        latest_assets = (
            latest_nav * latest_shares
            if latest_nav is not None
            and latest_nav > 0
            and latest_shares is not None
            and latest_shares > 0
            else None
        )
    currencies = sorted({str(row.get("currency")) for row in observations if row.get("currency")})
    comparable = len(currencies) <= 1
    median_flow = statistics.median(numeric) if numeric else None
    deviations = [abs(value - median_flow) for value in numeric] if median_flow is not None else []
    mad = statistics.median(deviations) if deviations else None
    robust_z = (
        0.6745 * (latest_flow - median_flow) / mad
        if latest_flow is not None and median_flow is not None and mad not in (None, 0.0)
        else None
    )
    return {
        "policy_id": ETF_FLOW_POLICY_ID,
        "visible_observations": len(observations),
        "latest_effective_date": latest.get("effective_date") if latest else None,
        "latest_training_available_session_date": (
            latest.get("training_available_session_date") if latest else None
        ),
        "latest_fund_flow": _round(latest_flow),
        "latest_flow_to_assets_pct": _round(
            latest_flow / latest_assets * 100.0
            if latest_flow is not None and latest_assets not in (None, 0.0)
            else None
        ),
        "sum_last_5_visible_flows": _round(sum(numeric[-5:])) if numeric and comparable else None,
        "sum_last_20_visible_flows": _round(sum(numeric[-20:])) if numeric and comparable else None,
        "median_visible_flow": _round(median_flow),
        "median_absolute_deviation": _round(mad),
        "latest_robust_zscore": _round(robust_z),
        "latest_extreme_outlier_flag": abs(robust_z) > 8.0 if robust_z is not None else False,
        "currency_values": currencies,
        "flows_comparable_for_sum": comparable,
        "dedupe_key": "ticker+effective_date",
        "revision_policy": "latest revision whose derived availability is <= as_of",
    }


def _compact_constituents(
    packet: Mapping[str, Any],
    limit: int = MAX_PROMPT_RELATION_ROWS,
    *,
    include_constituents: bool = True,
    include_memberships: bool = True,
) -> dict:
    relation = packet.get("etf_constituents") or {}

    def top(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
        ordered = sorted(
            rows,
            key=lambda row: (
                -(_as_number(row.get("weight_percent")) or -1.0),
                str(row.get("constituent_key") or row.get("etf_ticker") or ""),
            ),
        )
        keys = (
            "etf_ticker",
            "constituent_ticker",
            "constituent_name",
            "effective_date",
            "available_date",
            "weight_percent",
        )
        return [{key: row.get(key) for key in keys} for row in ordered[:limit]]

    constituents = relation.get("constituents") or []
    memberships = relation.get("etf_memberships") or []
    compact_constituents = top(constituents) if include_constituents else []
    compact_memberships = top(memberships) if include_memberships else []
    return {
        "snapshot_date": relation.get("constituent_snapshot_date"),
        "constituent_count": len(constituents),
        "constituents_in_prompt": compact_constituents,
        "constituents_omitted_count": max(
            0, len(constituents) - len(compact_constituents)
        ),
        "membership_count": len(memberships),
        "memberships_in_prompt": compact_memberships,
        "memberships_omitted_count": max(
            0, len(memberships) - len(compact_memberships)
        ),
    }


def _signals(price: Mapping[str, Any], flow: Mapping[str, Any]) -> Tuple[str, str]:
    price_value = price.get("return_20_session_pct")
    if price_value is None:
        price_value = price.get("return_5_session_pct")
    flow_value = flow.get("sum_last_20_visible_flows")
    if flow_value is None:
        flow_value = flow.get("sum_last_5_visible_flows")

    def sign(value: object) -> str:
        number = _as_number(value)
        if number is None:
            return "unknown"
        if number > 0:
            return "positive"
        if number < 0:
            return "negative"
        return "flat"

    return sign(price_value), sign(flow_value)


def _exposure_facts(
    packet: Mapping[str, Any], row_limit: int = MAX_TARGET_EXPOSURE_ROWS
) -> dict:
    exposure = packet.get("etf_flow_to_constituent") or {}
    rows = list(exposure.get("rows") or [])
    return {
        "policy_id": exposure.get("policy_id"),
        "eligible_etf_count": int(exposure.get("eligible_etf_count") or 0),
        "excluded_etf_count": int(exposure.get("excluded_etf_count") or 0),
        "net_weighted_flow_rate_contribution_pct": _round(
            _as_number(exposure.get("net_weighted_flow_rate_contribution_pct"))
        ),
        "positive_etf_count": int(exposure.get("positive_etf_count") or 0),
        "negative_etf_count": int(exposure.get("negative_etf_count") or 0),
        "explicit_currency_aggregate_available": bool(
            exposure.get("aggregate_currency")
            and exposure.get("net_allocated_flow_in_explicit_currency") is not None
        ),
        "top_contributing_etfs": [
            {
                key: row.get(key)
                for key in (
                    "etf_ticker",
                    "membership_weight_percent",
                    "flow_effective_date",
                    "flow_training_available_session_date",
                    "fund_flow_reported_units",
                    "allocated_flow_reported_units",
                    "flow_to_estimated_net_assets_pct",
                    "weighted_flow_rate_contribution_pct",
                )
            }
            for row in rows[:row_limit]
        ],
        "omitted_contributing_etf_count": max(0, len(rows) - row_limit),
    }


def _compact_price_evidence(rows: Sequence[Mapping[str, Any]]) -> dict:
    selected = list(rows[-MAX_PROMPT_PRICE_ROWS:])
    return {
        "visible_row_count": len(rows),
        "rows_in_prompt": [
            {
                key: row.get(key)
                for key in ("trade_date", "source", "close", "volume")
            }
            for row in selected
        ],
        "rows_omitted_count": max(0, len(rows) - len(selected)),
    }


def _compact_flow_evidence(rows: Sequence[Mapping[str, Any]]) -> dict:
    selected = list(rows[-MAX_PROMPT_FLOW_ROWS:])
    return {
        "visible_row_count": len(rows),
        "rows_in_prompt": [
            {
                key: row.get(key)
                for key in (
                    "effective_date",
                    "processed_date",
                    "training_available_session_date",
                    "fund_flow",
                    "nav",
                    "shares_outstanding",
                    "currency",
                )
            }
            for row in selected
        ],
        "rows_omitted_count": max(0, len(rows) - len(selected)),
    }


def _compact_exposure_evidence(
    exposure: Mapping[str, Any], row_limit: int = MAX_PROMPT_EXPOSURE_ROWS
) -> dict:
    rows = list(exposure.get("rows") or [])
    selected = rows[:row_limit]
    summary_keys = (
        "policy_id",
        "eligible_etf_count",
        "excluded_etf_count",
        "positive_etf_count",
        "negative_etf_count",
        "net_weighted_flow_rate_contribution_pct",
        "aggregate_currency",
        "net_allocated_flow_in_explicit_currency",
    )
    row_keys = (
        "etf_ticker",
        "membership_effective_date",
        "membership_available_date",
        "membership_weight_percent",
        "flow_effective_date",
        "flow_processed_date",
        "flow_training_available_session_date",
        "fund_flow_reported_units",
        "allocated_flow_reported_units",
        "flow_to_estimated_net_assets_pct",
        "weighted_flow_rate_contribution_pct",
        "provider_currency",
    )
    summary = {key: exposure.get(key) for key in summary_keys}
    summary["net_weighted_flow_rate_contribution_pct"] = _round(
        _as_number(exposure.get("net_weighted_flow_rate_contribution_pct"))
    )
    return {
        "summary": summary,
        "rows_in_prompt": [
            {key: row.get(key) for key in row_keys} for row in selected
        ],
        "rows_omitted_count": max(0, len(rows) - len(selected)),
    }


def _provenance_summary(rows: Sequence[Mapping[str, Any]]) -> dict:
    hashes = sorted(
        {
            str(row.get("payload_sha256"))
            for row in rows
            if row.get("payload_sha256")
        }
    )
    digest = hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()
    return {
        "raw_artifact_count": len(rows),
        "distinct_payload_count": len(hashes),
        "sources": sorted(
            {str(row.get("source")) for row in rows if row.get("source")}
        ),
        "sorted_payload_sha256_set_digest": digest,
        "sample_payload_sha256s": hashes[:MAX_PROMPT_PROVENANCE_HASHES],
        "all_raw_artifacts_retained_in_packet": True,
    }


def _compact_quality(quality: Mapping[str, Any]) -> dict:
    return {
        "status": quality.get("status"),
        "sources": quality.get("sources") or [],
        "reasons": quality.get("reasons") or [],
    }


def build_example(packet: Mapping[str, Any]) -> dict:
    validate_packet(packet)
    price_rows = _preferred_price_rows(packet)
    flow_rows = list((packet.get("etf_flow") or {}).get("observations") or [])
    price = _price_facts(price_rows)
    liquidity = _liquidity_facts(price_rows)
    flow = _flow_facts(flow_rows)
    exposure = _exposure_facts(
        packet, row_limit=0 if flow_rows else MAX_TARGET_EXPOSURE_ROWS
    )
    price_signal, flow_signal = _signals(price, flow)
    flow_signal_source = "own_etf_flow"
    if flow_signal == "unknown":
        exposure_value = _as_number(
            exposure.get("net_weighted_flow_rate_contribution_pct")
        )
        if exposure_value is not None:
            flow_signal = (
                "positive" if exposure_value > 0 else "negative" if exposure_value < 0 else "flat"
            )
            flow_signal_source = "constituent_etf_flow_exposure"
        else:
            flow_signal_source = "none"
    task_type = (
        "etf_own_flow_analysis"
        if flow_rows
        else "stock_constituent_flow_analysis"
        if exposure["eligible_etf_count"]
        else "all_stock_control_analysis"
    )
    if "unknown" in (price_signal, flow_signal):
        regime = "insufficient_joint_evidence"
    elif price_signal == "positive" and flow_signal == "positive":
        regime = "price_flow_positive_confirmation"
    elif price_signal == "negative" and flow_signal == "negative":
        regime = "price_flow_negative_confirmation"
    elif price_signal == "positive" and flow_signal == "negative":
        regime = "price_up_flow_out_divergence"
    elif price_signal == "negative" and flow_signal == "positive":
        regime = "price_down_flow_in_divergence"
    else:
        regime = "mixed_or_flat"

    counter_evidence = []
    if price_signal not in ("unknown", "flat") and flow_signal not in ("unknown", "flat") and price_signal != flow_signal:
        counter_evidence.append("price_and_etf_flow_signals_diverge")
    quality_status = (packet.get("quality") or {}).get("status")
    if quality_status not in ("pass",):
        counter_evidence.append("price_quality_status_{}".format(quality_status or "unknown"))

    unknowns = ["historical_backfill_not_true_as_observed_point_in_time"]
    if price["observed_sessions"] < 5:
        unknowns.append("insufficient_price_history_for_short_horizon_statistics")
    if not flow_rows:
        unknowns.append("no_etf_flow_visible_under_session_lag_policy")
    if flow.get("flows_comparable_for_sum") is False:
        unknowns.append("mixed_flow_currencies_prevent_aggregation")

    confidence = 0.25
    confidence += min(price["observed_sessions"], 20) / 20.0 * 0.30
    confidence += min(len(flow_rows), 20) / 20.0 * 0.25
    confidence += 0.15 if quality_status == "pass" else 0.05
    confidence -= 0.10 if counter_evidence else 0.0
    confidence = max(0.0, min(confidence, 0.95))

    facts = {
        "symbol": packet.get("symbol"),
        "as_of_date": packet.get("as_of_date"),
        "price": price,
        "liquidity": liquidity,
        "etf_flow": flow,
        "etf_flow_to_constituent": exposure,
        "etf_relations": {
            "constituent_count": len((packet.get("etf_constituents") or {}).get("constituents") or []),
            "membership_count": len((packet.get("etf_constituents") or {}).get("etf_memberships") or []),
        },
        "quality_status": quality_status,
    }
    interpretation = {
        "price_signal": price_signal,
        "etf_flow_signal": flow_signal,
        "etf_flow_signal_source": flow_signal_source,
        "relationship": regime,
        "scope": "data_interpretation_not_trade_execution",
        "task_type": task_type,
    }
    response = {
        "facts": facts,
        "interpretation": interpretation,
        "counter_evidence": counter_evidence,
        "unknowns": unknowns,
        "regime": regime,
        "confidence": _round(confidence, 4),
        "conclusion": "Evidence supports {} as of {}; no future outcome is used.".format(
            regime, packet.get("as_of_date")
        ),
    }

    provenance = (packet.get("provenance") or {}).get("raw_artifacts") or []
    evidence = {
        "symbol": packet.get("symbol"),
        "as_of_date": packet.get("as_of_date"),
        "price_history": _compact_price_evidence(price_rows),
        "price_asof_facts": price,
        "etf_flow_observations": _compact_flow_evidence(flow_rows),
        "etf_flow_asof_facts": flow,
        "etf_relations": _compact_constituents(
            packet,
            include_constituents=bool(flow_rows),
            include_memberships=not bool(flow_rows),
        ),
        "etf_flow_to_constituent": _compact_exposure_evidence(
            packet.get("etf_flow_to_constituent") or {},
            row_limit=0 if flow_rows else MAX_PROMPT_EXPOSURE_ROWS,
        ),
        "quality": _compact_quality(packet.get("quality") or {}),
        "liquidity": liquidity,
        "provenance": _provenance_summary(provenance),
        "historical_backfill_is_true_point_in_time": False,
    }
    context = (
        "You are a quant data-analysis model. Use only the supplied as-of evidence. "
        "Separate facts, interpretation, counter-evidence, unknowns, regime, confidence, "
        "and conclusion. Never infer or mention a future return, future price, or trade "
        "execution. Massive ETF Flow is visible only under policy {}.\nEVIDENCE_JSON={}".format(
            ETF_FLOW_POLICY_ID, canonical_json(evidence)
        )
    )
    instruction = (
        "이 시점에 실제로 이용 가능했던 가격·ETF Flow·ETF 구성 관계만 해석하고, "
        "미래 성과나 매매 지시 없이 지정된 구조로 답하라. /no_think"
    )
    packet_id = str(packet.get("packet_id") or hashlib.sha256(canonical_json(packet).encode("utf-8")).hexdigest())
    example_id = hashlib.sha256(
        "{}:{}".format(DATASET_CONTRACT_VERSION, packet_id).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "example_id": example_id,
        "context": context,
        "instruction": instruction,
        "response": canonical_json(response),
        "metadata": {
            "packet_id": packet_id,
            "input_packet_schema": REQUIRED_PACKET_SCHEMA,
            "symbol": packet.get("symbol"),
            "as_of_date": packet.get("as_of_date"),
            "etf_flow_policy_id": ETF_FLOW_POLICY_ID,
            "target_origin": "deterministic_auditable_baseline_v1",
            "contains_future_label": False,
            "contains_trade_instruction": False,
            "task_type": task_type,
        },
    }


def iter_packets(paths: Sequence[Path]) -> Iterator[Tuple[Path, int, dict]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                document = json.loads(line)
                if not isinstance(document, dict):
                    raise ValueError("{}:{} packet is not an object".format(path, line_number))
                yield path, line_number, document


def build_dataset(
    packet_paths: Sequence[Path],
    output_root: Path,
    sessions: Sequence[str],
    *,
    validation_start: str = "2024-01-01",
    test_start: str = "2025-01-01",
    embargo_sessions: int = 20,
    max_per_split: int = 0,
    min_etf_observed_sessions: int = DEFAULT_MIN_ETF_OBSERVED_SESSIONS,
    min_etf_nonzero_volume_ratio: float = DEFAULT_MIN_ETF_NONZERO_VOLUME_RATIO,
    min_etf_median_dollar_volume: float = DEFAULT_MIN_ETF_MEDIAN_DOLLAR_VOLUME,
    replace: bool = False,
) -> dict:
    paths = [Path(path).expanduser().resolve() for path in packet_paths]
    contract = split_contract(sessions, validation_start, test_start, embargo_sessions)
    output = Path(output_root).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    targets = {name: output / "{}.jsonl".format(name) for name in ("train", "validation", "test")}
    manifest_target = output / "manifest.json"
    existing = [path for path in list(targets.values()) + [manifest_target] if path.exists()]
    if existing and not replace:
        raise FileExistsError("output exists; pass --replace: {}".format(existing[0]))

    staging = Path(tempfile.mkdtemp(prefix=".quant-sft-build-", dir=str(output.parent)))
    handles = {}
    heaps: Dict[str, list] = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    input_rows = 0
    purged_rows = 0
    duplicate_rows_skipped = 0
    exclusion_counts: Dict[str, int] = {}
    task_type_counts: Dict[str, int] = {}
    dedupe = None
    try:
        dedupe = sqlite3.connect(str(staging / "dedupe.sqlite3"))
        dedupe.execute(
            """
            CREATE TABLE seen_examples (
                example_id TEXT PRIMARY KEY,
                packet_content_sha256 TEXT NOT NULL UNIQUE
            )
            """
        )
        if max_per_split <= 0:
            handles = {
                name: (staging / "{}.jsonl".format(name)).open("w", encoding="utf-8")
                for name in targets
            }
        for path, line_number, packet in iter_packets(paths):
            input_rows += 1
            try:
                validate_packet(packet)
            except Exception as exc:
                raise ValueError("{}:{}: {}".format(path, line_number, exc)) from exc
            split = assign_split(packet.get("as_of_date"), contract)
            if split is None:
                purged_rows += 1
                continue
            eligibility = packet_eligibility(
                packet,
                min_etf_observed_sessions=min_etf_observed_sessions,
                min_etf_nonzero_volume_ratio=min_etf_nonzero_volume_ratio,
                min_etf_median_dollar_volume=min_etf_median_dollar_volume,
            )
            if not eligibility["eligible"]:
                for reason in eligibility["reasons"]:
                    exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
                continue
            example = build_example(packet)
            task_type = str(example["metadata"]["task_type"])
            example["metadata"]["split"] = split
            encoded = canonical_json(example)
            content_packet = dict(packet)
            content_packet.pop("packet_id", None)
            packet_content_sha = hashlib.sha256(
                canonical_json(content_packet).encode("utf-8")
            ).hexdigest()
            try:
                dedupe.execute(
                    "INSERT INTO seen_examples(example_id, packet_content_sha256) VALUES (?, ?)",
                    (example["example_id"], packet_content_sha),
                )
            except sqlite3.IntegrityError:
                existing = dedupe.execute(
                    "SELECT packet_content_sha256 FROM seen_examples WHERE example_id=?",
                    (example["example_id"],),
                ).fetchone()
                same_content = dedupe.execute(
                    "SELECT example_id FROM seen_examples WHERE packet_content_sha256=?",
                    (packet_content_sha,),
                ).fetchone()
                if (existing and existing[0] == packet_content_sha) or same_content:
                    duplicate_rows_skipped += 1
                    continue
                raise ValueError("example_id collision with different packet content")
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            if max_per_split > 0:
                key = int(example["example_id"], 16)
                heap = heaps[split]
                item = (-key, example["example_id"], encoded)
                if len(heap) < max_per_split:
                    heapq.heappush(heap, item)
                elif key < -heap[0][0]:
                    heapq.heapreplace(heap, item)
            else:
                handles[split].write(encoded + "\n")
                counts[split] += 1
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles = {}

        if max_per_split > 0:
            for name, heap in heaps.items():
                selected = sorted(heap, key=lambda item: (-item[0], item[1]))
                path = staging / "{}.jsonl".format(name)
                with path.open("w", encoding="utf-8") as handle:
                    for _, _, encoded in selected:
                        handle.write(encoded + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                counts[name] = len(selected)

        empty = [name for name, count in counts.items() if count == 0]
        if empty:
            raise ValueError("empty required dataset split(s): {}".format(",".join(empty)))

        files = {}
        for name in targets:
            path = staging / "{}.jsonl".format(name)
            files[name] = {
                "filename": path.name,
                "rows": counts[name],
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        manifest = {
            "schema_version": "quant.sft.manifest.v1",
            "dataset_contract_version": DATASET_CONTRACT_VERSION,
            "example_schema_version": DATASET_SCHEMA_VERSION,
            "input_packet_schema_required": REQUIRED_PACKET_SCHEMA,
            "model_family": "Qwen/Qwen3-8B",
            "framework": "NVIDIA NeMo AutoModel",
            "training_method": "BF16 LoRA SFT answer-only loss",
            "etf_flow_policy_id": ETF_FLOW_POLICY_ID,
            "future_returns_in_prompt": False,
            "future_returns_in_target": False,
            "trade_execution_target": False,
            "target_origin": "deterministic_auditable_baseline_v1",
            "split_contract": contract,
            "selection": {
                "mode": "full_corpus" if max_per_split <= 0 else "lowest_example_id_hash",
                "max_per_split": max_per_split or None,
                "representative_sample_claimed": False,
            },
            "input_packet_files": [str(path) for path in paths],
            "input_rows_scanned": input_rows,
            "purged_embargo_or_out_of_range_rows": purged_rows,
            "duplicate_rows_skipped": duplicate_rows_skipped,
            "preprocessing_exclusion_counts": dict(sorted(exclusion_counts.items())),
            "eligible_task_type_counts_before_optional_hash_cap": dict(
                sorted(task_type_counts.items())
            ),
            "preprocessing_policy": {
                "etf_liquidity_policy_id": ETF_LIQUIDITY_POLICY_ID,
                "trailing_window_sessions": ETF_LIQUIDITY_WINDOW_SESSIONS,
                "min_observed_sessions": min_etf_observed_sessions,
                "min_nonzero_volume_ratio": min_etf_nonzero_volume_ratio,
                "min_median_dollar_volume": min_etf_median_dollar_volume,
                "same_session_positive_volume_required": True,
                "all_security_min_observed_sessions": 5,
                "price_basis": "raw close only; adjusted absolute price is forbidden",
                "unexplained_raw_price_discontinuity_gate": "exclude at >=45 percent",
                "flow_version_dedupe": "ticker+effective_date; latest revision visible as-of",
                "example_dedupe": "exact packet content and example id",
                "delisted_security_policy": (
                    "retain historical rows while traded; no rows after last price; "
                    "present-day active lists forbidden for historical filtering"
                ),
                "normalization": (
                    "per-AUM and trailing median/MAD computed only from observations "
                    "visible at each as-of date; no full-period scaler"
                ),
                "prompt_compaction": {
                    "price_rows": MAX_PROMPT_PRICE_ROWS,
                    "flow_rows": MAX_PROMPT_FLOW_ROWS,
                    "relation_rows_per_direction": MAX_PROMPT_RELATION_ROWS,
                    "exposure_rows": MAX_PROMPT_EXPOSURE_ROWS,
                    "target_exposure_rows": MAX_TARGET_EXPOSURE_ROWS,
                    "provenance_hash_samples": MAX_PROMPT_PROVENANCE_HASHES,
                    "all_raw_artifacts_retained_in_packet": True,
                    "full_token_audit_required_before_training": True,
                },
            },
            "files": files,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())

        output.mkdir(parents=True, exist_ok=True)
        for name, target in targets.items():
            os.replace(staging / "{}.jsonl".format(name), target)
        os.replace(manifest_path, manifest_target)
        return manifest
    finally:
        if dedupe is not None:
            dedupe.close()
        for handle in handles.values():
            handle.close()
        shutil.rmtree(staging, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", nargs="+", required=True, help="packet JSONL paths/globs")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--embargo-sessions", type=int, default=20)
    parser.add_argument(
        "--min-etf-observed-sessions",
        type=int,
        default=DEFAULT_MIN_ETF_OBSERVED_SESSIONS,
    )
    parser.add_argument(
        "--min-etf-nonzero-volume-ratio",
        type=float,
        default=DEFAULT_MIN_ETF_NONZERO_VOLUME_RATIO,
    )
    parser.add_argument(
        "--min-etf-median-dollar-volume",
        type=float,
        default=DEFAULT_MIN_ETF_MEDIAN_DOLLAR_VOLUME,
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=0,
        help="0 keeps the full corpus; positive values are explicit smoke-only hash caps",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    packet_paths = resolve_packet_paths(args.packets)
    sessions = load_trading_sessions(args.database)
    manifest = build_dataset(
        packet_paths,
        args.output_root,
        sessions,
        validation_start=args.validation_start,
        test_start=args.test_start,
        embargo_sessions=args.embargo_sessions,
        max_per_split=args.max_per_split,
        min_etf_observed_sessions=args.min_etf_observed_sessions,
        min_etf_nonzero_volume_ratio=args.min_etf_nonzero_volume_ratio,
        min_etf_median_dollar_volume=args.min_etf_median_dollar_volume,
        replace=args.replace,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
