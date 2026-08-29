"""Reconstruct historical SPY and QQQ membership from FMP change events."""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

from .source import SnapshotMeta, SourceBundle, canonical_symbol


ENDPOINTS = {
    "SPY": ("indexes_s_and_p_500_index", "indexes_historical_s_and_p_500"),
    "QQQ": ("indexes_nasdaq_index", "indexes_historical_nasdaq"),
}


def _parse_date(value: object) -> str | None:
    rendered = str(value or "").strip()
    if not rendered:
        return None
    for pattern in ("%Y-%m-%d", "%B %d, %Y", "%B %d %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(rendered, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def _endpoint_data(payload: Mapping, endpoint_id: str) -> list[dict]:
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, Mapping):
        raise ValueError("FMP index evidence has no endpoints object")
    endpoint = endpoints.get(endpoint_id)
    data = endpoint.get("data") if isinstance(endpoint, Mapping) else None
    if not isinstance(data, list) or not data:
        raise ValueError(f"FMP index evidence missing data: {endpoint_id}")
    return [row for row in data if isinstance(row, dict)]


def load_membership_evidence(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FMP index evidence must be an object")
    return payload


def reconstruct_memberships(
    payload: Mapping, sessions: Sequence[str]
) -> tuple[dict[str, dict[str, frozenset[str]]], dict]:
    """Reverse current lists through additions/removals, batched by effective day."""

    normalized_sessions = sorted(set(sessions))
    if not normalized_sessions:
        raise ValueError("session calendar is empty")
    all_memberships: dict[str, dict[str, frozenset[str]]] = {}
    audit: dict[str, dict] = {}
    for benchmark, (current_id, history_id) in ENDPOINTS.items():
        current = {
            canonical_symbol(row.get("symbol"))
            for row in _endpoint_data(payload, current_id)
            if canonical_symbol(row.get("symbol"))
        }
        events: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: {"added": set(), "removed": set()}
        )
        invalid_events = 0
        for row in _endpoint_data(payload, history_id):
            effective = _parse_date(row.get("dateAdded")) or _parse_date(row.get("date"))
            symbol = canonical_symbol(row.get("symbol"))
            removed = canonical_symbol(row.get("removedTicker"))
            if not effective or not symbol:
                invalid_events += 1
                continue
            position = bisect.bisect_left(normalized_sessions, effective)
            if position >= len(normalized_sessions):
                # A future announced/effective change is not in the current price calendar.
                continue
            session = normalized_sessions[position]
            events[session]["added"].add(symbol)
            if removed:
                events[session]["removed"].add(removed)

        state = set(current)
        by_date: dict[str, frozenset[str]] = {}
        for session in reversed(normalized_sessions):
            by_date[session] = frozenset(state)
            event = events.get(session)
            if event:
                state.difference_update(event["added"])
                state.update(event["removed"])
        all_memberships[benchmark] = by_date
        counts = [len(by_date[session]) for session in normalized_sessions]
        audit[benchmark] = {
            "current_count": len(current),
            "history_event_count": sum(
                len(value["added"]) for value in events.values()
            ),
            "invalid_event_count": invalid_events,
            "session_count": len(by_date),
            "min_member_count": min(counts),
            "median_member_count": median(counts),
            "max_member_count": max(counts),
            "reconstruction_method": (
                "current FMP index list reversed through dateAdded batches; "
                "effective dates mapped to first observed US session on or after dateAdded"
            ),
        }
    return all_memberships, audit


def _last_session_on_or_before(sessions: Sequence[str], value: str) -> str | None:
    position = bisect.bisect_right(sessions, value) - 1
    return sessions[position] if position >= 0 else None


def validate_against_holdings(
    source: SourceBundle,
    metadata: Iterable[SnapshotMeta],
    memberships: Mapping[str, Mapping[str, frozenset[str]]],
    sessions: Sequence[str],
) -> dict:
    """Compare reconstructed index membership with FMP ETF disclosure snapshots."""

    result = {}
    for benchmark in ENDPOINTS:
        rows = []
        for item in metadata:
            if item.etf_ticker != benchmark:
                continue
            session = _last_session_on_or_before(sessions, item.effective_date)
            if not session:
                continue
            disclosed = set(source.snapshot_holdings(item))
            reconstructed = set(memberships[benchmark].get(session, frozenset()))
            if not disclosed or not reconstructed:
                continue
            intersection = disclosed & reconstructed
            union = disclosed | reconstructed
            rows.append(
                {
                    "effective_date": item.effective_date,
                    "session": session,
                    "disclosed_count": len(disclosed),
                    "reconstructed_count": len(reconstructed),
                    "intersection_count": len(intersection),
                    "jaccard": len(intersection) / len(union),
                    "disclosure_recall": len(intersection) / len(disclosed),
                    "reconstruction_precision": len(intersection) / len(reconstructed),
                }
            )
        jaccards = sorted(row["jaccard"] for row in rows)
        recall = sorted(row["disclosure_recall"] for row in rows)
        p10_index = max(0, int(len(jaccards) * 0.10) - 1) if jaccards else 0
        summary = {
            "snapshot_count": len(rows),
            "median_jaccard": median(jaccards) if jaccards else None,
            "p10_jaccard": jaccards[p10_index] if jaccards else None,
            "median_disclosure_recall": median(recall) if recall else None,
            "gate": "PASS"
            if jaccards and median(jaccards) >= 0.90 and jaccards[p10_index] >= 0.80
            else "FAIL",
            "rows": rows,
        }
        result[benchmark] = summary
    return result
