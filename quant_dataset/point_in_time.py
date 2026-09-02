"""Fail-closed point-in-time availability policies for training packets.

Provider dates describe the source record, but they do not always describe the
first U.S. trading session on which a model could have consumed that record.
This module keeps those two concepts separate and derives training visibility
from the observed daily-price calendar.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
from typing import Iterable, Optional, Sequence, Tuple


ETF_FLOW_POLICY_ID = "massive_etf_flow_us_sessions_v1"
ETF_FLOW_EFFECTIVE_LAG_SESSIONS = 2
ETF_FLOW_PROCESSED_LAG_SESSIONS = 1
ETF_CONSTITUENT_POLICY_ID = "fmp_etf_constituent_next_us_session_v1"
US_EQUITY_CALENDAR_SYMBOLS = ("SPY", "QQQ")
US_EQUITY_SESSION_SQL = """
    SELECT trade_date
    FROM daily_observations
    WHERE symbol IN ('SPY', 'QQQ')
      AND close > 0
      AND volume > 0
    GROUP BY trade_date
    ORDER BY trade_date
"""
ETF_FLOW_PIT_FILTER = (
    "training_available_session_date<=as_of; "
    "training_available_session_date=max("
    "second_us_session_after_effective_date,"
    "first_us_session_after_processed_date)"
)


def _iso_date(value: object) -> Optional[str]:
    """Return a canonical ISO date or ``None`` for malformed input."""

    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed.isoformat()


def normalize_trading_sessions(values: Iterable[object]) -> Tuple[str, ...]:
    """Return sorted unique ISO sessions, dropping malformed calendar rows."""

    sessions = {_iso_date(value) for value in values}
    sessions.discard(None)
    return tuple(sorted(sessions))


def nth_session_strictly_after(
    sessions: Sequence[str], reference_date: object, count: int
) -> Optional[str]:
    """Return the ``count``-th observed session after a date, or fail closed.

    ``None`` means the local price calendar does not yet prove that the required
    delay elapsed.  Callers must exclude the source row instead of falling back
    to calendar-day arithmetic.
    """

    reference = _iso_date(reference_date)
    if reference is None or count < 1:
        return None
    index = bisect_right(sessions, reference) + count - 1
    if index >= len(sessions):
        return None
    return sessions[index]


def derive_etf_flow_available_session(
    effective_date: object,
    processed_date: object,
    sessions: Sequence[str],
) -> Optional[str]:
    """Derive the first session on which an ETF flow may enter training.

    Massive historical rows are exposed no earlier than two U.S. trading
    sessions after ``effective_date``.  Because ``processed_date`` has no
    verified publication timestamp, one complete trading session is also
    required after it.  The later bound wins.
    """

    effective_bound = nth_session_strictly_after(
        sessions, effective_date, ETF_FLOW_EFFECTIVE_LAG_SESSIONS
    )
    processed_bound = nth_session_strictly_after(
        sessions, processed_date, ETF_FLOW_PROCESSED_LAG_SESSIONS
    )
    if effective_bound is None or processed_bound is None:
        return None
    return max(effective_bound, processed_bound)


def derive_constituent_available_session(
    provider_available_date: object, sessions: Sequence[str]
) -> Optional[str]:
    """Require one complete U.S. session after date-only acceptance evidence."""

    return nth_session_strictly_after(sessions, provider_available_date, 1)


def etf_flow_policy_manifest() -> dict:
    """Machine-readable policy embedded in manifests and training packets."""

    return {
        "policy_id": ETF_FLOW_POLICY_ID,
        "calendar": (
            "positive-volume SPY/QQQ daily_observations sessions; isolated "
            "weekend rows from other instruments are excluded"
        ),
        "effective_date_min_lag_sessions": ETF_FLOW_EFFECTIVE_LAG_SESSIONS,
        "processed_date_min_lag_sessions": ETF_FLOW_PROCESSED_LAG_SESSIONS,
        "available_session_rule": (
            "max(second session strictly after effective_date, "
            "first session strictly after processed_date)"
        ),
        "missing_or_incomplete_calendar_action": "exclude_row_fail_closed",
        "provider_timestamp_precision": "date_only",
        "historical_backfill_is_true_point_in_time": False,
    }


def etf_constituent_policy_manifest() -> dict:
    return {
        "policy_id": ETF_CONSTITUENT_POLICY_ID,
        "provider_availability_field": "acceptanceTime date",
        "training_available_session_rule": (
            "first positive-volume SPY/QQQ session strictly after acceptance date"
        ),
        "rationale": "acceptance timestamp timezone and intraday usability are unverified",
        "missing_or_incomplete_calendar_action": "exclude_row_fail_closed",
        "historical_backfill_is_true_point_in_time": False,
    }
