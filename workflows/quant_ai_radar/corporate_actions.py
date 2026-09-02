"""Verified point-in-time corporate-action adjustments for Radar packets."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping


DEFAULT_VERIFIED_CORPORATE_ACTIONS = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/oracle/"
    "incremental/state/verified_corporate_actions.json"
)
SCHEMA_VERSION = "quant.verified_corporate_actions.v1"
ORACLE_SCHEMA_VERSION = "quant.oracle_corporate_actions.v1"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,31}$")
PRICE_FIELDS = ("open", "high", "low", "close", "adjusted_close", "vwap")


class CorporateActionError(RuntimeError):
    """A verified corporate-action ledger or packet adjustment is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _iso_date(value: Any, field: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as exc:
        raise CorporateActionError(f"invalid {field}: {value!r}") from exc


def _positive_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CorporateActionError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise CorporateActionError(f"{field} must be positive: {value!r}")
    return number


def load_verified_corporate_actions(
    path: Path, *, as_of_date: str
) -> dict[str, Any]:
    """Load official actions available and effective by the analysis date."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "path": str(resolved),
            "sha256": None,
            "events": [],
            "events_by_symbol": {},
        }
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorporateActionError(
            f"invalid corporate-action ledger: {resolved}: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise CorporateActionError(
            f"corporate-action ledger schema must be {SCHEMA_VERSION}"
        )
    as_of = _iso_date(as_of_date, "as_of_date")
    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(payload.get("events") or []):
        if not isinstance(event, Mapping):
            raise CorporateActionError(f"event {index} must be an object")
        symbol = str(event.get("symbol") or "").strip().upper()
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise CorporateActionError(f"event {index} has invalid symbol")
        action_type = str(event.get("action_type") or "")
        if action_type not in {"split", "reverse_split", "forward_split"}:
            raise CorporateActionError(
                f"event {index} has unsupported action_type: {action_type!r}"
            )
        effective = _iso_date(event.get("effective_date"), "effective_date")
        available = _iso_date(event.get("available_date"), "available_date")
        if available > as_of or effective > as_of:
            continue
        old_shares = _positive_number(event.get("old_shares"), "old_shares")
        new_shares = _positive_number(event.get("new_shares"), "new_shares")
        source_url = str(event.get("source_url") or "").strip()
        source_type = str(event.get("source_type") or "")
        if not source_url.startswith("https://"):
            raise CorporateActionError(
                f"event {index} requires an HTTPS official source"
            )
        if source_type not in {"official_issuer", "official_exchange"}:
            raise CorporateActionError(
                f"event {index} requires an official source_type"
            )
        normalized.append(
            {
                "symbol": symbol,
                "action_type": action_type,
                "effective_date": effective,
                "available_date": available,
                "announcement_date": _iso_date(
                    event.get("announcement_date") or available,
                    "announcement_date",
                ),
                "old_shares": old_shares,
                "new_shares": new_shares,
                "price_factor_for_prior_rows": old_shares / new_shares,
                "volume_factor_for_prior_rows": new_shares / old_shares,
                "source_type": source_type,
                "source_name": str(event.get("source_name") or "").strip(),
                "source_url": source_url,
                "verification_status": "official",
                "corroborating_sources": [
                    {
                        "provider": source_type,
                        "source_name": str(
                            event.get("source_name") or ""
                        ).strip(),
                        "source_url": source_url,
                    }
                ],
            }
        )
    normalized.sort(
        key=lambda item: (
            item["symbol"],
            item["effective_date"],
            item["source_url"],
        )
    )
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for event in normalized:
        by_symbol.setdefault(event["symbol"], []).append(event)
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "events": normalized,
        "events_by_symbol": by_symbol,
    }


def load_oracle_corporate_actions(
    database: Any, *, as_of_date: str
) -> dict[str, Any]:
    """Load only official or cross-provider split events from sealed Oracle."""

    as_of = _iso_date(as_of_date, "as_of_date")
    rows = list(database.corporate_action_rows(as_of))
    grouped: dict[tuple[str, str, float, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise CorporateActionError("Oracle corporate action has invalid symbol")
        effective = _iso_date(row.get("effective_date"), "effective_date")
        available = _iso_date(row.get("available_date"), "available_date")
        if effective > as_of or available > as_of:
            raise CorporateActionError(
                "Oracle returned a future corporate-action row"
            )
        old = _positive_number(row.get("old_shares"), "old_shares")
        new = _positive_number(row.get("new_shares"), "new_shares")
        grouped.setdefault((symbol, effective, old, new), []).append(row)

    normalized: list[dict[str, Any]] = []
    for (symbol, effective, old, new), evidence in sorted(grouped.items()):
        official = [
            row
            for row in evidence
            if str(row.get("source_type") or "")
            in {"official_issuer", "official_exchange"}
        ]
        providers = {
            str(row.get("provider") or "")
            for row in evidence
            if str(row.get("provider") or "") in {"massive", "fmp"}
        }
        if official:
            verification_status = "official"
            preferred = official[0]
        elif providers == {"massive", "fmp"}:
            verification_status = "cross_provider"
            preferred = next(
                row
                for row in evidence
                if str(row.get("provider") or "") == "massive"
            )
        else:
            continue
        corroborating_sources = [
            {
                "provider": str(row.get("provider") or ""),
                "source_name": str(row.get("source_name") or ""),
                "source_url": str(row.get("source_url") or ""),
                "available_date": str(row.get("available_date") or ""),
                "payload_sha256": str(row.get("payload_sha256") or ""),
            }
            for row in sorted(
                evidence,
                key=lambda item: (
                    str(item.get("provider") or ""),
                    str(item.get("source_url") or ""),
                ),
            )
        ]
        normalized.append(
            {
                "symbol": symbol,
                "action_type": (
                    "reverse_split" if old > new else "forward_split"
                ),
                "effective_date": effective,
                "available_date": min(
                    str(row.get("available_date") or "") for row in evidence
                ),
                "announcement_date": str(
                    preferred.get("announcement_date")
                    or preferred.get("available_date")
                ),
                "old_shares": old,
                "new_shares": new,
                "price_factor_for_prior_rows": old / new,
                "volume_factor_for_prior_rows": new / old,
                "source_type": str(preferred.get("source_type") or ""),
                "source_name": str(preferred.get("source_name") or ""),
                "source_url": str(preferred.get("source_url") or ""),
                "verification_status": verification_status,
                "corroborating_sources": corroborating_sources,
            }
        )
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for event in normalized:
        by_symbol.setdefault(event["symbol"], []).append(event)
    digest = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": ORACLE_SCHEMA_VERSION,
                "as_of_date": as_of,
                "events": normalized,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "path": str(
            getattr(
                getattr(database, "binding", None),
                "incremental_database",
                "oracle_incremental_database",
            )
        ),
        "sha256": digest,
        "source_row_count": len(rows),
        "events": normalized,
        "events_by_symbol": by_symbol,
    }


def adjust_packet_for_verified_corporate_actions(
    packet: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize historical packet rows to the latest visible split basis."""

    symbol = str(packet.get("symbol") or "").upper()
    events = list((ledger.get("events_by_symbol") or {}).get(symbol) or [])
    if not events:
        return dict(packet)
    adjusted = copy.deepcopy(dict(packet))
    applied: list[dict[str, Any]] = []
    for event in events:
        effective = str(event["effective_date"])
        price_factor = float(event["price_factor_for_prior_rows"])
        volume_factor = float(event["volume_factor_for_prior_rows"])
        affected_rows = 0
        for day in adjusted.get("history") or []:
            if str(day.get("trade_date") or "") >= effective:
                continue
            for row in day.get("sources") or []:
                original_close = row.get("close")
                for field in PRICE_FIELDS:
                    value = row.get(field)
                    if isinstance(value, (int, float)) and not isinstance(
                        value, bool
                    ):
                        row[field] = float(value) * price_factor
                volume = row.get("volume")
                if isinstance(volume, (int, float)) and not isinstance(
                    volume, bool
                ):
                    row["volume"] = float(volume) * volume_factor
                row["raw_close_before_corporate_action"] = original_close
                row["corporate_action_adjustment"] = {
                    "effective_date": effective,
                    "price_factor": price_factor,
                    "volume_factor": volume_factor,
                    "ledger_sha256": ledger.get("sha256"),
                }
                affected_rows += 1
        applied.append({**event, "affected_source_rows": affected_rows})
    adjusted["verified_corporate_actions"] = {
        "schema_version": SCHEMA_VERSION,
        "ledger_sha256": ledger.get("sha256"),
        "events": applied,
        "basis": "latest_effective_split_basis_visible_as_of",
    }
    provenance = adjusted.setdefault("provenance", {})
    provenance["verified_corporate_actions"] = {
        "ledger_sha256": ledger.get("sha256"),
        "event_count": len(applied),
        "official_sources_only": all(
            event.get("verification_status") == "official"
            for event in applied
        ),
        "verification_policy": "official_or_massive_fmp_cross_provider",
        "point_in_time_gate": "available_date_and_effective_date_lte_as_of",
    }
    packet_without_id = dict(adjusted)
    packet_without_id.pop("packet_id", None)
    adjusted["packet_id"] = hashlib.sha256(
        _canonical_json(packet_without_id).encode("utf-8")
    ).hexdigest()
    return adjusted
