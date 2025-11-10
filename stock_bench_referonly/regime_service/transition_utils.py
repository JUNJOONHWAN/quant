"""Shared helpers to derive recent regime transitions and formatting."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytz


DEFAULT_TZ = pytz.timezone("America/New_York")


def label_state(value: int) -> str:
    if value > 0:
        return "Risk-On"
    if value < 0:
        return "Risk-Off"
    return "Neutral"


def format_price(value: Optional[float]) -> str:
    try:
        if value is None:
            return "N/A"
        text = f"{float(value):.2f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
    except Exception:
        return "N/A"


def format_float(value: Optional[float], precision: int) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return "N/A"


def fusion_weights_series(snapshot: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    series_ta = snapshot.get("fusion_wTA_series")
    if not isinstance(series_ta, list):
        series_ta = []
    series_flow = snapshot.get("fusion_wFlow_series")
    if not isinstance(series_flow, list):
        series_flow = []
    return series_ta, series_flow


def fusion_weights_last(snapshot: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    w_ta = snapshot.get("fusion_wTA_last")
    w_flow = snapshot.get("fusion_wFlow_last")
    return (
        float(w_ta) if isinstance(w_ta, (int, float)) else None,
        float(w_flow) if isinstance(w_flow, (int, float)) else None,
    )


def _series_bench(snapshot: Dict[str, Any]) -> List[float]:
    ser = snapshot.get("series_bench")
    if isinstance(ser, list) and ser:
        return ser
    series_map = snapshot.get("series", {}) if isinstance(snapshot, dict) else {}
    ser = series_map.get("QQQ")
    if isinstance(ser, list):
        return ser
    return []


def asof_basis_price(snapshot: Dict[str, Any]) -> Optional[float]:
    ser = _series_bench(snapshot)
    try:
        if ser:
            return float(ser[-1])
    except Exception:
        return None
    return None


def asof_basis_date(snapshot: Dict[str, Any]) -> Optional[str]:
    if not isinstance(snapshot, dict):
        return None
    asof = snapshot.get("asof", {}) or {}
    return (
        asof.get("fusion_last_date")
        or asof.get("inter_last_date")
        or asof.get("today_utc")
        or snapshot.get("date")
    )


def scores_at_date(snapshot: Dict[str, Any], date_str: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not isinstance(snapshot, dict):
        return (None, None, None)
    fdates = snapshot.get("fusion_dates") or []
    scores = snapshot.get("fusion_score_series") or []
    wta_series, wflow_series = fusion_weights_series(snapshot)
    idx: Optional[int] = None
    if date_str and isinstance(fdates, list):
        try:
            idx = fdates.index(date_str)
        except Exception:
            for j in range(len(fdates) - 1, -1, -1):
                try:
                    if str(fdates[j]) <= str(date_str):
                        idx = j
                        break
                except Exception:
                    continue

    def _pick(arr: List[Any], pos: Optional[int]) -> Optional[float]:
        if pos is None or pos < 0:
            return None
        if pos < len(arr):
            try:
                return float(arr[pos])
            except Exception:
                return None
        return None

    score = _pick(scores, idx)
    wta = _pick(wta_series, idx)
    wflow = _pick(wflow_series, idx)
    if score is None:
        score_val = snapshot.get("fusion_score_last")
        try:
            score = float(score_val) if score_val is not None else None
        except Exception:
            score = None
    if wta is None or wflow is None:
        wta_last, wflow_last = fusion_weights_last(snapshot)
        if wta is None:
            wta = wta_last
        if wflow is None:
            wflow = wflow_last
    return (score, wta, wflow)


def _decide_hit(new_state: int, metric: Optional[float]) -> Optional[bool]:
    if metric is None:
        return None
    if new_state > 0:
        return metric >= 0
    if new_state < 0:
        return metric <= 0
    return metric >= 0


def compute_transitions(
    states: List[int],
    dates: List[str],
    prices: Optional[List[float]],
    limit: int = 10,
    *,
    asof: Optional[Dict[str, Any]] = None,
    prices_open: Optional[List[float]] = None,
    tz=None,
) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    if not states or not dates:
        return transitions

    states_seq = list(states)
    dates_seq = list(dates)
    n = min(len(states_seq), len(dates_seq))
    if n <= 0:
        return transitions
    states_seq = states_seq[-n:]
    dates_seq = dates_seq[-n:]

    price_series: List[Optional[float]] = []
    if isinstance(prices, list):
        for val in prices:
            try:
                price_series.append(float(val))
            except (TypeError, ValueError):
                price_series.append(None)
    length_offset = len(price_series) - len(states_seq)
    if length_offset < 0:
        price_series = [None] * (-length_offset) + price_series
        length_offset = 0

    def price_at(index: int) -> Optional[float]:
        pos = length_offset + index
        if 0 <= pos < len(price_series):
            return price_series[pos]
        return None

    open_series: List[Optional[float]] = []
    open_offset = 0
    if isinstance(prices_open, list):
        for val in prices_open:
            try:
                open_series.append(float(val))
            except (TypeError, ValueError):
                open_series.append(None)
        open_offset = len(open_series) - len(states_seq)
        if open_offset < 0:
            open_series = [None] * (-open_offset) + open_series
            open_offset = 0

    def open_at(index: int) -> Optional[float]:
        if not open_series:
            return None
        pos = open_offset + index
        if 0 <= pos < len(open_series):
            return open_series[pos]
        return None

    change_indices = [i for i in range(1, len(states_seq)) if states_seq[i] != states_seq[i - 1]]
    if not change_indices:
        return transitions

    final_price = price_at(len(states_seq) - 1)
    last_date = dates_seq[-1] if dates_seq else None
    asof_data = asof or {}
    today_str = asof_data.get("today_utc") if isinstance(asof_data, dict) else None
    intraday_applied = bool(asof_data.get("intraday_base_applied")) if isinstance(asof_data, dict) else False
    tzinfo = tz or DEFAULT_TZ
    try:
        now_et = datetime.now(tzinfo)
    except Exception:
        now_et = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(DEFAULT_TZ)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    next_change_idx = len(states_seq) - 1
    for idx in reversed(change_indices):
        end_idx = min(next_change_idx, len(dates_seq) - 1)
        end_date = dates_seq[end_idx]
        end_price = price_at(end_idx)
        if end_idx == len(states_seq) - 1:
            if intraday_applied and final_price is not None:
                end_price = final_price
            else:
                if last_date and today_str and str(last_date) == str(today_str):
                    if now_et >= market_open:
                        end_price = open_at(end_idx) or None
                    else:
                        end_price = None
                elif final_price is not None:
                    end_price = final_price

        start_idx = min(idx, len(dates_seq) - 1)
        start_date = dates_seq[start_idx]
        start_price = price_at(idx)
        prev_state_value = states_seq[idx - 1]

        price_delta: Optional[float] = None
        return_pct: Optional[float] = None
        if start_price is not None and end_price is not None and start_price != 0:
            price_delta = end_price - start_price
            return_pct = end_price / start_price - 1.0

        metric_value = price_delta if price_delta is not None else return_pct
        hit = _decide_hit(states_seq[idx], metric_value)

        transitions.append(
            {
                "date": start_date,
                "state": states_seq[idx],
                "state_label": label_state(states_seq[idx]),
                "prev_state": prev_state_value,
                "prev_state_label": label_state(prev_state_value),
                "end_date": end_date,
                "price_start": start_price,
                "price_end": end_price,
                "price_delta": price_delta,
                "return_pct": return_pct,
                "hit": hit,
            }
        )

        if len(transitions) >= limit:
            break
        next_change_idx = idx

    return transitions


def _fmt_md(date_str: Optional[str]) -> str:
    try:
        if not date_str:
            return "N/A"
        dt = datetime.strptime(str(date_str), "%Y-%m-%d")
        return dt.strftime("%m/%d")
    except Exception:
        try:
            # Accept already mm/dd
            dt = datetime.fromisoformat(str(date_str))
            return dt.strftime("%m/%d")
        except Exception:
            s = str(date_str)
            if len(s) >= 10 and s[4] == '-' and s[7] == '-':
                return s[5:7] + "/" + s[8:10]
            return s


def _next_calendar_day(date_str: Optional[str]) -> Optional[str]:
    try:
        if not date_str:
            return None
        dt = datetime.strptime(str(date_str), "%Y-%m-%d")
        from datetime import timedelta
        return (dt + timedelta(days=1)).strftime("%m/%d")
    except Exception:
        return None


def build_recent_transition_lines(
    snapshot: Dict[str, Any],
    title: str,
    *,
    limit: int = 10,
    tz=None,
) -> List[str]:
    states = snapshot.get("states", []) or []
    dates = snapshot.get("dates", []) or []
    bench = snapshot.get("series_bench")
    if not isinstance(bench, list):
        bench = (snapshot.get("series", {}) or {}).get("QQQ") or []
    transitions = compute_transitions(
        states,
        dates,
        bench,
        limit=limit,
        asof=snapshot.get("asof", {}),
        prices_open=snapshot.get("series_bench_open"),
        tz=tz,
    )
    if not transitions:
        return []

    bench_list = bench if isinstance(bench, list) else []
    latest_state = {
        "date": snapshot.get("date"),
        "end_date": snapshot.get("date"),
        "state": snapshot.get("state", 0),
        "state_label": label_state(int(snapshot.get("state", 0))),
        "prev_state": transitions[0].get("state") if transitions else 0,
        "prev_state_label": transitions[0].get("state_label") if transitions else "Neutral",
        "price_start": transitions[0].get("price_end"),
        "price_end": bench_list[-1] if bench_list else None,
        "price_delta": None,
        "return_pct": None,
        "hit": None,
    }
    if transitions and latest_state["date"] != transitions[0].get("date"):
        transitions.insert(0, latest_state)

    header_date = asof_basis_date(snapshot) or "N/A"
    header_price = asof_basis_price(snapshot)
    asof = snapshot.get("asof", {}) or {}
    src = asof.get("override_source")
    src_sfx = f" · source={src}" if isinstance(src, str) and src else ""
    lines: List[str] = [
        title,
        f"  · 현재 스냅샷 기준: 데이터 기준일자 {_fmt_md(header_date)} · 데이터 기준가격(QQQ) {format_price(header_price)}{src_sfx}",
    ]

    for idx, transition in enumerate(transitions, start=1):
        date_val = transition.get("date") or "N/A"
        prev_label = transition.get("prev_state_label") or "Neutral"
        curr_label = transition.get("state_label") or "Neutral"
        price_start = transition.get("price_start")
        hit = transition.get("hit")
        hit_txt = "적중" if hit else ("미적중" if hit is not None else "N/A")
        score, wta, wflow = scores_at_date(snapshot, date_val)
        # format dates without year; signal date = next calendar day of data date
        sig_date = _next_calendar_day(date_val) or _fmt_md(date_val)
        dv = _fmt_md(date_val)
        # day return percentage
        rpct = transition.get("return_pct")
        rpct_txt = f"{rpct*100:+.2f}%" if isinstance(rpct, (int, float)) else "N/A"
        line = (
            f"{idx:02d}. 신호일자 {sig_date} · 데이터일자 {dv} · 레짐 {curr_label} · "
            f"QQQ {format_price(price_start)} · 변화율 {rpct_txt} ({hit_txt}) · "
            f"score {format_float(score, 3)} · wTA {format_float(wta, 2)}"
        )
        lines.append(line)

    return lines


def build_recent_transition_markdown(
    snapshot: Dict[str, Any],
    *,
    title: str = "🌙 최근 전환 10회 (각 항목별 데이터 기준 + score/wTA/wFLOW)",
    limit: int = 10,
    tz=None,
) -> str:
    lines = build_recent_transition_lines(snapshot, title, limit=limit, tz=tz)
    if not lines:
        return "\n"
    return "\n".join(lines) + "\n"


__all__ = [
    "asof_basis_date",
    "asof_basis_price",
    "build_recent_transition_lines",
    "build_recent_transition_markdown",
    "compute_transitions",
    "format_float",
    "format_price",
    "fusion_weights_last",
    "fusion_weights_series",
    "label_state",
    "scores_at_date",
]
