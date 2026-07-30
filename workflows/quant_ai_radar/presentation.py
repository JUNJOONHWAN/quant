"""Korean, renderer-owned labels and compact numeric presentation."""

from __future__ import annotations

import math
from typing import Any


MARKET_STATE_LABELS = {
    "risk_on": "위험선호",
    "risk_off": "위험회피",
    "rotation": "섹터 순환",
    "mixed": "혼조",
    "insufficient_evidence": "판단 근거 부족",
}

ROTATION_STATE_LABELS = {
    "rotation_in": "자금 유입과 가격 확산",
    "rotation_out": "자금 이탈과 가격 약화",
    "mixed": "방향이 엇갈리는 혼조",
}

REGIME_LABELS = {
    "price_flow_positive_confirmation": "가격·ETF 자금 동반 강세",
    "price_flow_negative_confirmation": "가격·ETF 자금 동반 약세",
    "price_up_flow_out_divergence": "가격 상승·ETF 자금 유출 괴리",
    "price_down_flow_in_divergence": "가격 하락·ETF 자금 유입 괴리",
    "mixed_or_flat": "가격·ETF 자금 혼조",
    "insufficient_joint_evidence": "공동 근거 부족",
}

SIGNAL_LABELS = {
    "positive": "강세",
    "negative": "약세",
    "neutral": "중립",
    "unknown": "미확인",
}

TASK_LABELS = {
    "etf_own_flow_analysis": "ETF 자체 자금 흐름",
    "stock_constituent_flow_analysis": "ETF 구성종목 전달 자금",
    "all_stock_control_analysis": "가격 중심 대조 분석",
}


def label_market_state(value: Any) -> str:
    text = str(value or "")
    return MARKET_STATE_LABELS.get(text, text or "미확인")


def label_rotation_state(value: Any) -> str:
    text = str(value or "")
    return ROTATION_STATE_LABELS.get(text, text or "미확인")


def label_regime(value: Any) -> str:
    text = str(value or "")
    return REGIME_LABELS.get(text, text or "미확인")


def label_signal(value: Any) -> str:
    text = str(value or "")
    return SIGNAL_LABELS.get(text, text or "미확인")


def label_task(value: Any) -> str:
    text = str(value or "")
    return TASK_LABELS.get(text, text or "미확인")


def whole(value: Any, *, signed: bool = False, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:,.0f}{suffix}"


def confidence_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if number <= 1:
        number *= 100
    return whole(number, suffix="%")


def bar_width(value: Any, *, minimum: float = 0, maximum: float = 100) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number) or maximum <= minimum:
        return 0
    return int(round(max(0.0, min(1.0, (number - minimum) / (maximum - minimum))) * 100))
