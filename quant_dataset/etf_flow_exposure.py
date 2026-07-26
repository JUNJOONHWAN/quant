"""Point-in-time ETF-flow attribution from funds to constituent stocks."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence


ETF_CONSTITUENT_FLOW_POLICY_ID = "pit_etf_flow_to_constituent_weight_v1"


def _number(value: object):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value, digits: int = 8):
    return round(value, digits) if value is not None and math.isfinite(value) else None


def build_constituent_flow_exposure(
    symbol: str,
    as_of_date: str,
    memberships: Sequence[Mapping[str, Any]],
    flow_packets: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Allocate each visible ETF flow by its visible constituent weight.

    Duplicate positions inside one ETF snapshot are summed before the ETF flow
    is applied, preventing the fund flow itself from being counted twice.
    Currency is not invented: the Massive fund-flow endpoint does not expose a
    reporting-currency field reliably, so raw allocated amounts remain labeled
    as provider-reported units.
    """

    grouped: Dict[str, dict] = {}
    for row in memberships:
        etf = str(row.get("etf_ticker") or "").strip().upper()
        if not etf:
            continue
        item = grouped.setdefault(
            etf,
            {
                "etf_ticker": etf,
                "membership_effective_date": row.get("effective_date"),
                "membership_available_date": row.get("available_date"),
                "membership_weight_percent": 0.0,
                "source_position_count": 0,
                "direct_equity_proxy_eligible": bool(
                    row.get("direct_equity_proxy_eligible")
                ),
                "direct_equity_proxy_reasons": list(
                    row.get("direct_equity_proxy_reasons") or []
                ),
            },
        )
        weight = _number(row.get("weight_percent"))
        if weight is not None and weight > 0:
            item["membership_weight_percent"] += weight
        item["source_position_count"] += 1
        if row.get("available_date") and str(row["available_date"]) > str(
            item["membership_available_date"] or ""
        ):
            item["membership_available_date"] = row["available_date"]
        if not row.get("direct_equity_proxy_eligible"):
            item["direct_equity_proxy_eligible"] = False
            item["direct_equity_proxy_reasons"] = sorted(
                set(item["direct_equity_proxy_reasons"])
                | set(row.get("direct_equity_proxy_reasons") or [])
            )

    rows = []
    exclusion_counts: Dict[str, int] = {}
    excluded_etfs = 0
    for etf in sorted(grouped):
        membership = grouped[etf]
        reasons = []
        if not membership["direct_equity_proxy_eligible"]:
            reasons.append("etf_snapshot_not_direct_equity_proxy")
        if membership["membership_weight_percent"] <= 0:
            reasons.append("membership_weight_nonpositive_or_missing")
        if str(membership.get("membership_available_date") or "") > as_of_date:
            reasons.append("membership_not_available_as_of")
        flow_packet = flow_packets.get(etf) or {}
        flow = flow_packet.get("latest")
        if not flow:
            reasons.append("no_etf_flow_visible_as_of")
        elif str(flow.get("training_available_session_date") or "") > as_of_date:
            reasons.append("etf_flow_not_available_as_of")
        fund_flow = _number(flow.get("fund_flow")) if flow else None
        if fund_flow is None:
            reasons.append("fund_flow_missing_or_nonfinite")
        if reasons:
            excluded_etfs += 1
            for reason in reasons:
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            continue

        weight = membership["membership_weight_percent"]
        nav = _number(flow.get("nav"))
        shares = _number(flow.get("shares_outstanding"))
        estimated_net_assets = (
            nav * shares if nav is not None and nav > 0 and shares is not None and shares > 0 else None
        )
        flow_rate_pct = (
            fund_flow / estimated_net_assets * 100.0
            if estimated_net_assets not in (None, 0.0)
            else None
        )
        allocated = fund_flow * weight / 100.0
        rows.append(
            {
                **membership,
                "flow_effective_date": flow.get("effective_date"),
                "flow_processed_date": flow.get("processed_date"),
                "flow_training_available_session_date": flow.get(
                    "training_available_session_date"
                ),
                "flow_availability_policy_id": flow.get(
                    "training_availability_policy_id"
                ),
                "fund_flow_reported_units": _round(fund_flow),
                "nav": _round(nav),
                "shares_outstanding": _round(shares),
                "estimated_net_assets_reported_units": _round(estimated_net_assets),
                "flow_to_estimated_net_assets_pct": _round(flow_rate_pct),
                "allocated_flow_reported_units": _round(allocated),
                "weighted_flow_rate_contribution_pct": _round(
                    flow_rate_pct * weight / 100.0 if flow_rate_pct is not None else None
                ),
                "provider_currency": flow.get("currency"),
            }
        )

    rows.sort(
        key=lambda row: (
            -abs(float(row.get("allocated_flow_reported_units") or 0.0)),
            row["etf_ticker"],
        )
    )
    weighted_rates = [
        float(row["weighted_flow_rate_contribution_pct"])
        for row in rows
        if row.get("weighted_flow_rate_contribution_pct") is not None
    ]
    known_currencies = sorted(
        {str(row["provider_currency"]) for row in rows if row.get("provider_currency")}
    )
    same_known_currency = len(known_currencies) == 1 and len(rows) > 0
    allocated_values = [float(row["allocated_flow_reported_units"]) for row in rows]
    return {
        "policy_id": ETF_CONSTITUENT_FLOW_POLICY_ID,
        "symbol": symbol,
        "as_of_date": as_of_date,
        "allocation_rule": "fund_flow * point_in_time_constituent_weight_percent / 100",
        "flow_visibility_rule": "massive_etf_flow_us_sessions_v1",
        "membership_visibility_rule": (
            "fmp_etf_constituent_next_us_session_v1: "
            "training_available_session_date <= as_of"
        ),
        "duplicate_position_rule": "sum weights within ETF before applying fund flow once",
        "currency_policy": (
            "do not label provider fund-flow units as USD; aggregate raw allocated "
            "amounts only when one explicit non-null currency is shared"
        ),
        "direct_equity_proxy_policy": (
            "use only PIT snapshots with sufficient ticker coverage, plausible "
            "positive weight sum, and no negative weights"
        ),
        "eligible_etf_count": len(rows),
        "excluded_etf_count": excluded_etfs,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "net_weighted_flow_rate_contribution_pct": _round(sum(weighted_rates))
        if weighted_rates
        else None,
        "positive_etf_count": sum(
            1 for value in allocated_values if value > 0
        ),
        "negative_etf_count": sum(
            1 for value in allocated_values if value < 0
        ),
        "aggregate_currency": known_currencies[0] if same_known_currency else None,
        "net_allocated_flow_in_explicit_currency": _round(sum(allocated_values))
        if same_known_currency
        else None,
        "rows": rows,
    }
