"""Single-stock deep research framework with DDM context.

This module is deliberately data-provider agnostic at the core: callers pass
FMP/Massive/Barchart/community evidence as dictionaries, and the pure functions
turn those inputs into scored hypotheses, a walk-forward backtest, Monte Carlo
simulation, and a 420px HTML report.
"""

from __future__ import annotations

import html
import json
import math
import random
import re
import statistics
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]
PriceSeries = List[Tuple[str, float]]

DEFAULT_HORIZONS: Dict[str, int] = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}

DEFAULT_CONTEXT_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "XLK",
    "SMH",
    "XLY",
    "XLI",
    "XLF",
    "XLE",
    "XLV",
    "XLP",
    "TLT",
    "GLD",
    "UUP",
]

SECTOR_ETF_MAP = {
    "technology": ["XLK", "QQQ", "SMH"],
    "communication services": ["XLC", "QQQ"],
    "consumer cyclical": ["XLY", "IWM"],
    "consumer defensive": ["XLP"],
    "financial services": ["XLF"],
    "healthcare": ["XLV", "IWM"],
    "industrials": ["XLI", "ITA", "XAR"],
    "energy": ["XLE", "USO", "CL=F"],
    "basic materials": ["XLB", "HG=F", "GLD"],
    "real estate": ["XLRE", "TLT"],
    "utilities": ["XLU", "TLT"],
}

MOAT_KEYWORDS = (
    "platform",
    "subscription",
    "recurring",
    "license",
    "network",
    "ecosystem",
    "mission critical",
    "proprietary",
    "patent",
    "data",
    "scale",
    "switching cost",
    "installed base",
)

MILESTONE_KEYWORDS = (
    "earnings",
    "guidance",
    "contract",
    "launch",
    "approval",
    "fda",
    "faa",
    "shipment",
    "production",
    "backlog",
    "rpo",
    "partnership",
    "trial",
    "phase",
    "investor day",
)

RAW_SEC_TAXONOMY_RE = re.compile(
    r"\b(?:us-gaap|s-gaap|dei|xbrli|iso4217|ifrs-full|Member)\b",
    re.IGNORECASE,
)
RAW_SEC_QNAME_RE = re.compile(r"\b[a-z]{1,12}(?:-[a-z]+)?:[A-Za-z0-9_]+")
RAW_SEC_ZERO_ID_RE = re.compile(r"\b0{3,}\d{5,}\b")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _round(value: Any, digits: int = 2) -> Optional[float]:
    number = as_float(value)
    if number is None:
        return None
    return round(number, digits)


def _parse_percent_metric(value: Any) -> Optional[float]:
    text = str(value or "").replace(",", "").replace("$", "").strip()
    if not text:
        return None
    number = as_float(text.replace("%", ""))
    if number is not None:
        return number
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return as_float(match.group(0)) if match else None


def _exposure_upper_pct(value: Any) -> Optional[float]:
    numbers = [
        number
        for number in (_parse_percent_metric(part) for part in re.findall(r"[-+]?\d+(?:\.\d+)?%?", str(value or "")))
        if number is not None
    ]
    return max(numbers) if numbers else None


def _looks_like_raw_sec_metadata(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return False
    taxonomy_hits = len(RAW_SEC_TAXONOMY_RE.findall(text))
    qname_hits = len(RAW_SEC_QNAME_RE.findall(text))
    zero_id_hits = len(RAW_SEC_ZERO_ID_RE.findall(text))
    member_hits = len(re.findall(r"\b[A-Za-z0-9]+Member\b", text))
    prose_marks = len(re.findall(r"[.!?]", text))
    word_count = len(re.findall(r"[A-Za-z가-힣]{3,}", text))
    if qname_hits >= 2 or taxonomy_hits >= 3:
        return True
    if member_hits >= 2 and word_count < 45:
        return True
    return bool(zero_id_hits and (qname_hits or taxonomy_hits) and prose_marks == 0)


def fmt_num(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    abs_number = abs(number)
    if abs_number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.{digits}f}B"
    if abs_number >= 1_000_000:
        return f"{number / 1_000_000:.{digits}f}M"
    if abs_number >= 1_000:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def fmt_money(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    return "$" + fmt_num(number, digits)


def fmt_pct(value: Any, digits: int = 1, already_pct: bool = True) -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    if not already_pct:
        number *= 100.0
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.{digits}f}%"


def flatten_records(data: Any) -> List[JsonDict]:
    if data is None:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "historical", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    return []


def first_record(data: Any) -> JsonDict:
    rows = flatten_records(data)
    return rows[0] if rows else {}


def find_value(row: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    if not isinstance(row, Mapping):
        return None
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return row[key]
    lower = {str(key).lower(): key for key in row}
    for key in candidates:
        real_key = lower.get(str(key).lower())
        if real_key is not None and row[real_key] not in (None, ""):
            return row[real_key]
    return None


def records_for(datasets: Mapping[str, Any], key: str) -> List[JsonDict]:
    return flatten_records(datasets.get(key))


def series_from_records(records: Sequence[Mapping[str, Any]]) -> PriceSeries:
    points: PriceSeries = []
    for row in records:
        close = as_float(find_value(row, ["close", "adjClose", "c", "price"]))
        raw_date = find_value(row, ["date", "t", "timestamp", "time"])
        if close is None or raw_date is None:
            continue
        label = str(raw_date)
        if isinstance(raw_date, (int, float)):
            try:
                label = datetime.fromtimestamp(raw_date / 1000, tz=timezone.utc).date().isoformat()
            except Exception:
                label = str(raw_date)
        else:
            label = label[:10]
        if len(label) >= 10:
            points.append((label[:10], float(close)))
    dedup = {date_key: price for date_key, price in points}
    return sorted(dedup.items(), key=lambda item: item[0])


def price_series_from_datasets(datasets: Mapping[str, Any], symbol: str) -> PriceSeries:
    rows = records_for(datasets, "fmp_eod_full")
    if rows:
        return series_from_records(rows)
    rows = records_for(datasets, "massive_daily_aggs")
    return series_from_records(rows)


def pct_change(start: Any, end: Any) -> Optional[float]:
    left = as_float(start)
    right = as_float(end)
    if left in (None, 0) or right is None:
        return None
    return (right / left - 1.0) * 100.0


def series_return(series: PriceSeries, days: int) -> Optional[float]:
    if len(series) <= days:
        return None
    return pct_change(series[-days - 1][1], series[-1][1])


def moving_average(values: Sequence[float], window: int) -> Optional[float]:
    if len(values) < window or window <= 0:
        return None
    return sum(values[-window:]) / window


def stdev(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return None
    return statistics.stdev(clean)


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return clean[low]
    return clean[low] + (clean[high] - clean[low]) * (rank - low)


def daily_returns(series: PriceSeries) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for idx in range(1, len(series)):
        prev = series[idx - 1][1]
        curr = series[idx][1]
        if prev:
            out[series[idx][0]] = curr / prev - 1.0
    return out


def daily_return_list(series: PriceSeries) -> List[float]:
    returns = []
    for idx in range(1, len(series)):
        prev = series[idx - 1][1]
        curr = series[idx][1]
        if prev:
            returns.append(curr / prev - 1.0)
    return returns


def pearson_corr(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    if len(pairs) < 3:
        return None
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


def sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def bounded_signal(value: Any, scale: float) -> Optional[float]:
    number = as_float(value)
    if number is None:
        return None
    if scale <= 0:
        return None
    return clamp(number / scale, -2.0, 2.0)


def weighted_average(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    usable = [(float(v), float(w)) for v, w in pairs if w > 0 and math.isfinite(v)]
    if not usable:
        return None
    total = sum(weight for _, weight in usable)
    if total <= 0:
        return None
    return sum(value * weight for value, weight in usable) / total


def weighted_stdev(pairs: Sequence[Tuple[float, float]], center: float) -> Optional[float]:
    usable = [(float(v), float(w)) for v, w in pairs if w > 0 and math.isfinite(v)]
    if not usable:
        return None
    total = sum(weight for _, weight in usable)
    if total <= 0:
        return None
    variance = sum(weight * (value - center) ** 2 for value, weight in usable) / total
    return math.sqrt(max(0.0, variance))


def compute_rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
    if len(values) <= period:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for idx in range(len(values) - period, len(values)):
        change = values[idx] - values[idx - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def max_drawdown_pct(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            drawdown = min(drawdown, value / peak - 1.0)
    return drawdown * 100.0


def realized_vol_pct(series: PriceSeries, lookback: int = 63) -> Optional[float]:
    returns = daily_return_list(series)[-lookback:]
    if len(returns) < 10:
        return None
    sigma = stdev(returns)
    if sigma is None:
        return None
    return sigma * math.sqrt(252) * 100.0


def compute_technical_snapshot(series: PriceSeries) -> JsonDict:
    values = [price for _, price in series]
    latest = values[-1] if values else None
    sma20 = moving_average(values, 20)
    sma50 = moving_average(values, 50)
    sma200 = moving_average(values, 200)
    rsi14 = compute_rsi(values)
    returns = {name: series_return(series, days) for name, days in DEFAULT_HORIZONS.items()}
    vol21 = realized_vol_pct(series, 21)
    vol63 = realized_vol_pct(series, 63)
    drawdown = max_drawdown_pct(values[-252:])
    score = 50.0
    if latest is not None and sma20:
        score += 8 if latest > sma20 else -8
    if latest is not None and sma50:
        score += 8 if latest > sma50 else -8
    if latest is not None and sma200:
        score += 7 if latest > sma200 else -7
    score += clamp((returns.get("1m") or 0.0) * 0.7, -14.0, 14.0)
    score += clamp((returns.get("3m") or 0.0) * 0.25, -10.0, 10.0)
    if rsi14 is not None:
        if rsi14 > 78:
            score -= 8
        elif rsi14 > 68:
            score -= 3
        elif 45 <= rsi14 <= 62:
            score += 4
        elif rsi14 < 35:
            score -= 5
    if vol63 is not None and vol63 > 95:
        score -= 8
    elif vol63 is not None and vol63 < 35:
        score += 3
    return {
        "latest_price": _round(latest),
        "price_date": series[-1][0] if series else None,
        "sma20": _round(sma20),
        "sma50": _round(sma50),
        "sma200": _round(sma200),
        "rsi14": _round(rsi14),
        "realized_vol_21d_pct": _round(vol21),
        "realized_vol_63d_pct": _round(vol63),
        "max_drawdown_252d_pct": _round(drawdown),
        "returns_pct": {key: _round(value) for key, value in returns.items()},
        "score": round(clamp(score), 1),
    }


def choose_context_symbols(
    symbol: str,
    *,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    peers: Optional[Sequence[str]] = None,
    max_symbols: int = 18,
) -> List[str]:
    selected: List[str] = []

    def add(items: Iterable[str]) -> None:
        for item in items:
            clean = str(item or "").strip().upper()
            if not clean or clean == symbol.upper():
                continue
            if clean not in selected:
                selected.append(clean)

    add(["SPY", "QQQ", "IWM", "DIA", "VTI"])
    sector_key = str(sector or "").strip().lower()
    add(SECTOR_ETF_MAP.get(sector_key, []))
    industry_key = str(industry or "").strip().lower()
    if "semiconductor" in industry_key:
        add(["SMH", "SOXX"])
    if "aerospace" in industry_key or "defense" in industry_key:
        add(["ITA", "XAR", "XLI"])
    if "software" in industry_key:
        add(["IGV", "XLK", "QQQ"])
    if "biotech" in industry_key:
        add(["IBB", "XBI", "XLV"])
    add(peers or [])
    add(["TLT", "UUP", "GLD"])
    return selected[:max_symbols]


def _latest_by_date(series: PriceSeries, asof_date: str) -> PriceSeries:
    return [(dt_key, price) for dt_key, price in series if dt_key <= asof_date]


def build_single_stock_ddm_signal(
    *,
    symbol: str,
    target_series: PriceSeries,
    context_series: Mapping[str, PriceSeries],
    topstep_pulse: Optional[Mapping[str, Any]] = None,
    asof_date: Optional[str] = None,
    corr_lookback_bars: int = 45,
    min_overlap: int = 20,
    min_abs_corr: float = 0.25,
) -> JsonDict:
    """Build a drift/diffusion signal for one stock.

    Drift is correlated-basket pressure. Diffusion is disagreement,
    resistance, and realized volatility around that pressure.
    """

    target = symbol.upper()
    if not target_series:
        return {
            "enabled": False,
            "status": "insufficient_target_history",
            "target": target,
            "note": "개별주 DDM을 계산할 가격 시계열이 없습니다.",
        }
    asof = asof_date or target_series[-1][0]
    target_slice = _latest_by_date(target_series, asof)
    target_returns = daily_returns(target_slice)
    if len(target_returns) < min_overlap:
        return {
            "enabled": False,
            "status": "insufficient_target_history",
            "target": target,
            "asof_date": asof,
            "note": "개별주 DDM 상관 바스켓을 만들 과거 수익률 수가 부족합니다.",
        }

    target_dates = sorted(target_returns)[-max(min_overlap, corr_lookback_bars) :]
    components: List[JsonDict] = []
    for ctx_symbol, ctx_series_raw in context_series.items():
        ctx_symbol_clean = str(ctx_symbol).upper()
        if ctx_symbol_clean == target:
            continue
        ctx_series = _latest_by_date(ctx_series_raw, asof)
        ctx_returns = daily_returns(ctx_series)
        pairs = [
            (target_returns[dt_key], ctx_returns[dt_key])
            for dt_key in target_dates
            if dt_key in target_returns and dt_key in ctx_returns
        ]
        if len(pairs) < min_overlap:
            continue
        corr = pearson_corr(pairs)
        if corr is None or abs(corr) < min_abs_corr:
            continue
        ret_5 = series_return(ctx_series, 5)
        ret_21 = series_return(ctx_series, 21)
        impulse_parts = [
            (bounded_signal(ret_5, 5.0), 0.62),
            (bounded_signal(ret_21, 12.0), 0.38),
        ]
        usable = [(value, weight) for value, weight in impulse_parts if value is not None]
        impulse_base = weighted_average(usable)
        if impulse_base is None:
            continue
        signed_pressure = (1.0 if corr >= 0 else -1.0) * impulse_base
        weight = abs(corr)
        components.append(
            {
                "symbol": ctx_symbol_clean,
                "corr": round(corr, 3),
                "weight": round(weight, 3),
                "direction": "positive_corr" if corr >= 0 else "negative_corr",
                "return_5d_pct": _round(ret_5),
                "return_21d_pct": _round(ret_21),
                "impulse": round(impulse_base, 3),
                "signed_pressure": round(signed_pressure, 3),
                "support": signed_pressure > 0,
            }
        )

    if not components:
        return {
            "enabled": False,
            "status": "no_correlated_basket",
            "target": target,
            "asof_date": asof,
            "note": "현재 lookback에서 충분히 상관된 주식/ETF 바스켓이 없습니다.",
        }

    pressure_pairs = [
        (float(item["signed_pressure"]), float(item["weight"]))
        for item in components
    ]
    raw_drift = weighted_average(pressure_pairs) or 0.0
    dispersion = weighted_stdev(pressure_pairs, raw_drift) or 0.0
    support_pressure = sum(
        max(0.0, float(item["signed_pressure"])) * float(item["weight"])
        for item in components
    )
    resistance_pressure = sum(
        max(0.0, -float(item["signed_pressure"])) * float(item["weight"])
        for item in components
    )
    total_pressure = support_pressure + resistance_pressure
    agreement = support_pressure / total_pressure if total_pressure > 0 else 0.5
    resistance_ratio = resistance_pressure / total_pressure if total_pressure > 0 else 0.5

    macro_prior = 0.0
    if topstep_pulse:
        macro_prior = clamp(as_float(topstep_pulse.get("risk_on_score")) or 0.0, -2.5, 2.5) / 2.5
    drift = raw_drift + macro_prior * 0.18
    vol63 = realized_vol_pct(target_slice, 63) or 0.0
    vol_penalty = clamp((vol63 - 55.0) / 100.0, 0.0, 0.45)
    diffusion = dispersion + resistance_ratio * 1.15 + vol_penalty
    evidence = drift / (diffusion + 0.35)
    confidence = clamp(sigmoid(evidence * 1.65) * 100.0)
    high_corr_count = sum(1 for item in components if abs(float(item["corr"])) >= 0.35)

    if drift > 0.35 and agreement >= 0.60 and confidence >= 62 and high_corr_count >= 2:
        status = "boost"
    elif drift > 0.15 and agreement >= 0.55 and confidence >= 55:
        status = "constructive"
    elif drift <= -0.10 or agreement < 0.47:
        status = "block"
    else:
        status = "neutral"

    ranked = sorted(
        components,
        key=lambda item: abs(float(item["signed_pressure"]) * float(item["weight"])),
        reverse=True,
    )
    return {
        "enabled": True,
        "status": status,
        "target": target,
        "asof_date": asof,
        "corr_lookback_bars": corr_lookback_bars,
        "min_overlap": min_overlap,
        "min_abs_corr": min_abs_corr,
        "correlated_count": len(components),
        "high_corr_count": high_corr_count,
        "drift": round(drift, 4),
        "raw_drift": round(raw_drift, 4),
        "macro_prior": round(macro_prior, 4),
        "diffusion": round(diffusion, 4),
        "dispersion": round(dispersion, 4),
        "evidence": round(evidence, 4),
        "confidence_pct": round(confidence, 1),
        "agreement_pct": round(agreement * 100.0, 1),
        "support_pressure": round(support_pressure, 4),
        "resistance_pressure": round(resistance_pressure, 4),
        "support": [item for item in ranked if item.get("support")][:8],
        "resistance": [item for item in ranked if not item.get("support")][:8],
        "components": ranked[:18],
        "note": "Drift는 개별주와 상관된 ETF/피어/매크로 바스켓 압력, diffusion은 내부 불일치·저항·변동성입니다.",
    }


def _sum_recent(rows: Sequence[Mapping[str, Any]], field: str, count: int = 4) -> Optional[float]:
    values = [as_float(find_value(row, [field])) for row in rows[:count]]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean)


def _ratio(num: Any, den: Any, pct: bool = True) -> Optional[float]:
    left = as_float(num)
    right = as_float(den)
    if left is None or right in (None, 0):
        return None
    out = left / right
    return out * 100.0 if pct else out


def compute_fundamental_snapshot(datasets: Mapping[str, Any]) -> JsonDict:
    profile = first_record(datasets.get("fmp_profile"))
    quote = first_record(datasets.get("fmp_quote"))
    income_q = records_for(datasets, "fmp_income_quarter")
    income_a = records_for(datasets, "fmp_income_annual")
    cashflow_q = records_for(datasets, "fmp_cashflow_quarter")
    balance_q = records_for(datasets, "fmp_balance_quarter")
    ratios_q = first_record(datasets.get("fmp_ratios_quarter"))
    metrics_q = first_record(datasets.get("fmp_key_metrics_quarter"))
    scores = first_record(datasets.get("fmp_financial_scores"))

    latest_income = income_q[0] if income_q else {}
    prior_year_quarter = income_q[4] if len(income_q) > 4 else {}
    latest_balance = balance_q[0] if balance_q else {}
    latest_cashflow = cashflow_q[0] if cashflow_q else {}
    revenue_ttm = _sum_recent(income_q, "revenue")
    if revenue_ttm is None and income_a:
        revenue_ttm = as_float(find_value(income_a[0], ["revenue"]))
    market_cap = (
        as_float(find_value(quote, ["marketCap"]))
        or as_float(find_value(profile, ["mktCap", "marketCap"]))
        or as_float(find_value(metrics_q, ["marketCap"]))
    )
    revenue_q = as_float(find_value(latest_income, ["revenue"]))
    revenue_yoy = pct_change(
        find_value(prior_year_quarter, ["revenue"]),
        revenue_q,
    ) if prior_year_quarter else None
    gross_profit_q = as_float(find_value(latest_income, ["grossProfit"]))
    gross_margin = _ratio(gross_profit_q, revenue_q)
    net_income_q = as_float(find_value(latest_income, ["netIncome"]))
    fcf_q = as_float(find_value(latest_cashflow, ["freeCashFlow"]))
    operating_cf_q = as_float(find_value(latest_cashflow, ["operatingCashFlow", "netCashProvidedByOperatingActivities"]))
    fcf_margin = _ratio(fcf_q, revenue_q)
    cash = as_float(find_value(latest_balance, ["cashAndShortTermInvestments", "cashAndCashEquivalents"]))
    total_debt = as_float(find_value(latest_balance, ["totalDebt", "shortTermDebt", "longTermDebt"]))
    current_assets = as_float(find_value(latest_balance, ["totalCurrentAssets"]))
    current_liabilities = as_float(find_value(latest_balance, ["totalCurrentLiabilities"]))
    current_ratio = (
        as_float(find_value(ratios_q, ["currentRatio"]))
        or _ratio(current_assets, current_liabilities, pct=False)
    )
    total_equity = as_float(find_value(latest_balance, ["totalStockholdersEquity", "totalEquity"]))
    debt_to_equity = _ratio(total_debt, total_equity, pct=False)
    ps_ratio = _ratio(market_cap, revenue_ttm, pct=False)
    pe_ratio = as_float(find_value(quote, ["pe"])) or as_float(find_value(metrics_q, ["peRatio"]))
    piotroski = as_float(find_value(scores, ["piotroskiScore", "piotroski"]))
    altman_z = as_float(find_value(scores, ["altmanZScore", "altmanZ"]))
    rd_expense = as_float(find_value(latest_income, ["researchAndDevelopmentExpenses"]))
    rd_ratio = _ratio(rd_expense, revenue_q)

    quality = 50.0
    if revenue_yoy is not None:
        quality += clamp(revenue_yoy * 0.35, -15, 18)
    if gross_margin is not None:
        quality += clamp((gross_margin - 35.0) * 0.35, -10, 14)
    if fcf_margin is not None:
        quality += clamp(fcf_margin * 0.55, -18, 16)
    if current_ratio is not None:
        quality += 6 if current_ratio >= 1.5 else -8 if current_ratio < 0.9 else 0
    if debt_to_equity is not None:
        quality += 5 if debt_to_equity <= 0.4 else -7 if debt_to_equity > 1.5 else 0
    if piotroski is not None:
        quality += (piotroski - 5.0) * 2.0
    if altman_z is not None:
        quality += 5 if altman_z >= 3 else -8 if altman_z < 1.8 else 0

    value = 50.0
    growth_adj = revenue_yoy or 0.0
    if ps_ratio is not None:
        value += clamp(growth_adj * 0.25, -8, 14)
        value -= clamp((ps_ratio - 6.0) * 2.2, -12, 28)
    if pe_ratio is not None and pe_ratio > 0:
        value += 8 if pe_ratio < 22 else -clamp((pe_ratio - 28.0) * 0.45, 0, 18)
    elif net_income_q is not None and net_income_q < 0:
        value -= 8
    if fcf_margin is not None:
        value += clamp(fcf_margin * 0.35, -10, 12)
    if cash is not None and total_debt is not None and market_cap:
        net_cash_pct = (cash - total_debt) / market_cap * 100.0
        value += clamp(net_cash_pct * 0.35, -8, 10)
    else:
        net_cash_pct = None

    return {
        "sector": find_value(profile, ["sector"]),
        "industry": find_value(profile, ["industry"]),
        "description": find_value(profile, ["description", "companyDescription"]),
        "market_cap": _round(market_cap),
        "revenue_q": _round(revenue_q),
        "revenue_ttm": _round(revenue_ttm),
        "revenue_yoy_pct": _round(revenue_yoy),
        "gross_margin_pct": _round(gross_margin),
        "free_cash_flow_q": _round(fcf_q),
        "free_cash_flow_margin_pct": _round(fcf_margin),
        "operating_cash_flow_q": _round(operating_cf_q),
        "net_income_q": _round(net_income_q),
        "cash_and_short_investments": _round(cash),
        "total_debt": _round(total_debt),
        "net_cash_pct_market_cap": _round(net_cash_pct),
        "current_ratio": _round(current_ratio),
        "debt_to_equity": _round(debt_to_equity),
        "price_to_sales_ttm": _round(ps_ratio),
        "pe_ratio": _round(pe_ratio),
        "piotroski_score": _round(piotroski, 1),
        "altman_z": _round(altman_z, 2),
        "rd_ratio_pct": _round(rd_ratio),
        "quality_score": round(clamp(quality), 1),
        "value_score": round(clamp(value), 1),
    }


def compute_relative_snapshot(
    *,
    symbol: str,
    target_series: PriceSeries,
    context_series: Mapping[str, PriceSeries],
    datasets: Mapping[str, Any],
) -> JsonDict:
    target_1m = series_return(target_series, 21)
    target_3m = series_return(target_series, 63)
    context_rows: List[JsonDict] = []
    for ctx, series in context_series.items():
        ret_1m = series_return(series, 21)
        ret_3m = series_return(series, 63)
        if ret_1m is None and ret_3m is None:
            continue
        context_rows.append(
            {
                "symbol": ctx,
                "return_1m_pct": _round(ret_1m),
                "return_3m_pct": _round(ret_3m),
            }
        )
    ctx_1m = [row["return_1m_pct"] for row in context_rows if row.get("return_1m_pct") is not None]
    ctx_3m = [row["return_3m_pct"] for row in context_rows if row.get("return_3m_pct") is not None]
    rel_1m = target_1m - statistics.median(ctx_1m) if target_1m is not None and ctx_1m else None
    rel_3m = target_3m - statistics.median(ctx_3m) if target_3m is not None and ctx_3m else None

    peer_quotes = records_for(datasets, "fmp_peer_quotes")
    peer_changes = []
    for row in peer_quotes:
        change = as_float(find_value(row, ["changesPercentage", "changePercentage", "changesPercentage"]))
        if change is not None:
            peer_changes.append((str(find_value(row, ["symbol"]) or ""), change))
    target_peer_change = None
    for peer, change in peer_changes:
        if peer.upper() == symbol.upper():
            target_peer_change = change
            break
    if target_peer_change is not None and len(peer_changes) > 2:
        below = sum(1 for _, change in peer_changes if change <= target_peer_change)
        peer_percentile = below / len(peer_changes) * 100.0
    else:
        peer_percentile = None

    score = 50.0
    if rel_1m is not None:
        score += clamp(rel_1m * 1.1, -18, 18)
    if rel_3m is not None:
        score += clamp(rel_3m * 0.55, -14, 14)
    if peer_percentile is not None:
        score += (peer_percentile - 50.0) * 0.18
    return {
        "score": round(clamp(score), 1),
        "target_return_1m_pct": _round(target_1m),
        "target_return_3m_pct": _round(target_3m),
        "relative_1m_vs_context_pct": _round(rel_1m),
        "relative_3m_vs_context_pct": _round(rel_3m),
        "peer_1d_percentile": _round(peer_percentile),
        "context_rows": sorted(context_rows, key=lambda row: str(row.get("symbol")))[:16],
    }


def compute_upside_snapshot(
    *,
    target_series: PriceSeries,
    datasets: Mapping[str, Any],
    ddm_signal: Mapping[str, Any],
) -> JsonDict:
    quote = first_record(datasets.get("fmp_quote"))
    price_target = first_record(datasets.get("fmp_price_target_summary")) or first_record(
        datasets.get("fmp_price_target_consensus")
    )
    dcf = first_record(datasets.get("fmp_dcf")) or first_record(datasets.get("fmp_levered_dcf"))
    current_price = (
        as_float(find_value(quote, ["price", "lastSalePrice"]))
        or (target_series[-1][1] if target_series else None)
    )
    target_price = as_float(find_value(price_target, ["targetConsensus", "priceTargetAverage", "targetMean", "targetMedian"]))
    dcf_price = as_float(find_value(dcf, ["dcf", "DCF", "value"]))
    target_upside = pct_change(current_price, target_price)
    dcf_upside = pct_change(current_price, dcf_price)
    values = [price for _, price in target_series[-252:]]
    high_252 = max(values) if values else None
    low_252 = min(values) if values else None
    high_retest_upside = pct_change(current_price, high_252)
    downside_to_low = pct_change(current_price, low_252)
    upside_inputs = [value for value in (target_upside, dcf_upside, high_retest_upside) if value is not None]
    median_upside = statistics.median(upside_inputs) if upside_inputs else None
    ddm_evidence = as_float(ddm_signal.get("evidence")) if ddm_signal.get("enabled") else 0.0
    score = 50.0
    if median_upside is not None:
        score += clamp(median_upside * 0.75, -24, 26)
    if high_retest_upside is not None:
        score += clamp(high_retest_upside * 0.35, -8, 10)
    score += clamp((ddm_evidence or 0.0) * 14.0, -12, 12)
    return {
        "score": round(clamp(score), 1),
        "current_price": _round(current_price),
        "analyst_target_price": _round(target_price),
        "analyst_target_upside_pct": _round(target_upside),
        "dcf_price": _round(dcf_price),
        "dcf_upside_pct": _round(dcf_upside),
        "high_252d": _round(high_252),
        "low_252d": _round(low_252),
        "high_retest_upside_pct": _round(high_retest_upside),
        "downside_to_252d_low_pct": _round(downside_to_low),
        "median_upside_proxy_pct": _round(median_upside),
    }


def compute_moat_snapshot(datasets: Mapping[str, Any], fundamentals: Mapping[str, Any]) -> JsonDict:
    description = str(fundamentals.get("description") or "").lower()
    keyword_hits = [word for word in MOAT_KEYWORDS if word in description]
    gross_margin = as_float(fundamentals.get("gross_margin_pct"))
    rd_ratio = as_float(fundamentals.get("rd_ratio_pct"))
    revenue_growth = as_float(fundamentals.get("revenue_yoy_pct"))
    fcf_margin = as_float(fundamentals.get("free_cash_flow_margin_pct"))
    holders = records_for(datasets, "fmp_institutional_holders")
    score = 45.0
    if gross_margin is not None:
        score += clamp((gross_margin - 35.0) * 0.32, -8, 14)
    if rd_ratio is not None:
        score += clamp(rd_ratio * 0.20, 0, 8)
    if revenue_growth is not None and revenue_growth > 10:
        score += clamp(revenue_growth * 0.18, 0, 8)
    if fcf_margin is not None and fcf_margin > 0:
        score += clamp(fcf_margin * 0.25, 0, 8)
    score += min(len(keyword_hits), 6) * 2.5
    if len(holders) >= 20:
        score += 4
    return {
        "score": round(clamp(score), 1),
        "keyword_hits": keyword_hits[:10],
        "institutional_holder_rows": len(holders),
        "read": "해자 증거가 숫자와 사업 설명 양쪽에서 확인됩니다." if score >= 65 else "해자 후보는 있으나 수치로 더 확인해야 합니다.",
    }


def build_roadmap_snapshot(datasets: Mapping[str, Any]) -> JsonDict:
    news_rows = records_for(datasets, "fmp_legacy_stock_news") or records_for(datasets, "fmp_news_stock_search")
    massive_news = records_for(datasets, "massive_news")
    sec_rows = records_for(datasets, "fmp_sec_symbol")
    eight_k = records_for(datasets, "massive_8k")
    earnings = records_for(datasets, "fmp_earnings")
    today = date.today()
    events: List[JsonDict] = []
    for row in earnings[:8]:
        raw_date = str(find_value(row, ["date", "fillingDate", "acceptedDate"]) or "")[:10]
        if raw_date:
            events.append(
                {
                    "date": raw_date,
                    "type": "earnings",
                    "title": "Earnings / reported EPS check",
                    "source": "FMP earnings",
                }
            )
    combined_news = list(news_rows[:20]) + list(massive_news[:12])
    keyword_count = 0
    for row in combined_news:
        title = str(find_value(row, ["title", "headline"]) or "")
        lower = title.lower()
        hits = [kw for kw in MILESTONE_KEYWORDS if kw in lower]
        if hits:
            keyword_count += 1
            events.append(
                {
                    "date": str(find_value(row, ["publishedDate", "published_utc", "date"]) or "")[:10],
                    "type": "news_milestone",
                    "title": title[:160],
                    "source": find_value(row, ["site", "publisher", "source"]),
                    "keywords": hits[:4],
                }
            )
    for row in list(sec_rows[:12]) + list(eight_k[:8]):
        form_type = str(find_value(row, ["formType", "form_type", "form"]) or "")
        if form_type.upper() in {"8-K", "10-Q", "10-K", "S-1", "424B", "DEF 14A"}:
            events.append(
                {
                    "date": str(find_value(row, ["filingDate", "filing_date", "acceptedDate"]) or "")[:10],
                    "type": "filing",
                    "title": form_type or "SEC filing",
                    "source": "FMP/Massive SEC",
                }
            )
    future_events = [item for item in events if str(item.get("date") or "") >= today.isoformat()]
    score = 45.0 + min(len(events), 12) * 2.0 + min(keyword_count, 8) * 2.0
    if future_events:
        score += 8
    if len(eight_k) >= 3:
        score += 4
    return {
        "score": round(clamp(score), 1),
        "event_count": len(events),
        "future_event_count": len(future_events),
        "milestone_news_count": keyword_count,
        "events": sorted(events, key=lambda item: str(item.get("date") or ""), reverse=True)[:14],
        "read": "로드맵을 추적할 촉매와 공시 이벤트가 충분합니다." if score >= 65 else "로드맵은 확인 가능하지만 숫자 milestone 추적을 더 쌓아야 합니다.",
    }


def build_hypotheses(
    *,
    fundamentals: Mapping[str, Any],
    relative: Mapping[str, Any],
    upside: Mapping[str, Any],
    roadmap: Mapping[str, Any],
    moat: Mapping[str, Any],
    ddm_signal: Mapping[str, Any],
) -> List[JsonDict]:
    quality = as_float(fundamentals.get("quality_score")) or 50.0
    value = as_float(fundamentals.get("value_score")) or 50.0
    relative_score = as_float(relative.get("score")) or 50.0
    upside_score = as_float(upside.get("score")) or 50.0
    roadmap_score = as_float(roadmap.get("score")) or 50.0
    moat_score = as_float(moat.get("score")) or 50.0
    ddm_bonus = clamp((as_float(ddm_signal.get("evidence")) or 0.0) * 10.0, -8.0, 8.0)

    return [
        {
            "id": "value_investment_merit",
            "question": "가치투자할 가치가 있는 회사인가",
            "score": round(clamp(quality * 0.58 + value * 0.42), 1),
            "evidence": [
                f"Quality {quality:.1f}/100, Value {value:.1f}/100",
                f"매출 YoY {fmt_pct(fundamentals.get('revenue_yoy_pct'))}, FCF margin {fmt_pct(fundamentals.get('free_cash_flow_margin_pct'))}",
                f"P/S {fmt_num(fundamentals.get('price_to_sales_ttm'))}, Net cash/MC {fmt_pct(fundamentals.get('net_cash_pct_market_cap'))}",
            ],
            "counter": "밸류에이션 지표가 비싸거나 FCF가 불안정하면 장기 가치투자보다 이벤트 성장주로 분류합니다.",
        },
        {
            "id": "relative_attractiveness",
            "question": "지금 상대투자매력이 높은가",
            "score": round(clamp(relative_score + ddm_bonus * 0.4), 1),
            "evidence": [
                f"1개월 상대수익률 {fmt_pct(relative.get('relative_1m_vs_context_pct'))}",
                f"3개월 상대수익률 {fmt_pct(relative.get('relative_3m_vs_context_pct'))}",
                f"피어 1일 percentile {fmt_pct(relative.get('peer_1d_percentile'))}",
            ],
            "counter": "상대수익률이 좋아도 diffusion이 높으면 추격보다 눌림/확인 구간으로 봅니다.",
        },
        {
            "id": "remaining_upside",
            "question": "그럼에도 상승 여력이 많은가",
            "score": round(clamp(upside_score + ddm_bonus), 1),
            "evidence": [
                f"애널리스트 목표가 upside {fmt_pct(upside.get('analyst_target_upside_pct'))}",
                f"DCF proxy upside {fmt_pct(upside.get('dcf_upside_pct'))}",
                f"DDM evidence {fmt_num(ddm_signal.get('evidence'), 3)} / confidence {fmt_pct(ddm_signal.get('confidence_pct'))}",
            ],
            "counter": "목표가·DCF·52주 고점 재돌파 여지가 모두 낮으면 모멘텀이 있어도 기대수익/리스크가 얇습니다.",
        },
        {
            "id": "roadmap_milestones",
            "question": "로드맵과 마일스톤이 이상적인가",
            "score": round(clamp(roadmap_score), 1),
            "evidence": [
                f"추적 이벤트 {roadmap.get('event_count')}개, 향후 이벤트 {roadmap.get('future_event_count')}개",
                str(roadmap.get("read") or ""),
            ],
            "counter": "뉴스 키워드가 많아도 매출·마진·현금흐름으로 이어지는 milestone이 아니면 점수를 제한합니다.",
        },
        {
            "id": "moat",
            "question": "해자가 있는가",
            "score": round(clamp(moat_score), 1),
            "evidence": [
                f"Moat score {moat_score:.1f}/100",
                "키워드: " + (", ".join(moat.get("keyword_hits") or []) or "N/A"),
                f"기관 보유 데이터 rows {moat.get('institutional_holder_rows')}",
            ],
            "counter": "텍스트상 해자 후보가 있어도 고마진·반복매출·현금흐름으로 확인되지 않으면 보수적으로 둡니다.",
        },
    ]


def build_gostop_moment(
    *,
    ddm_signal: Mapping[str, Any],
    hypotheses: Sequence[Mapping[str, Any]],
    technical: Mapping[str, Any],
) -> JsonDict:
    """Translate DDM pressure into a GoStop-style decision moment.

    ETF GoStop uses DDM as a boundary filter: supportive drift can allow risk,
    while high diffusion/resistance blocks or delays it. For a single stock we
    keep the same shape but blend in company hypothesis quality so the moment
    is not just a short-term chart signal.
    """

    if not ddm_signal.get("enabled"):
        return {
            "enabled": False,
            "status": "unavailable",
            "gate": "WAIT",
            "score": 50.0,
            "reason": "DDM context basket이 부족해 GoStop moment를 확정하지 않습니다.",
        }
    scores = [as_float(item.get("score")) for item in hypotheses]
    clean_scores = [score for score in scores if score is not None]
    hypothesis_avg = sum(clean_scores) / len(clean_scores) if clean_scores else 50.0
    tech_score = as_float(technical.get("score")) or 50.0
    drift = as_float(ddm_signal.get("drift")) or 0.0
    diffusion = as_float(ddm_signal.get("diffusion")) or 0.0
    evidence = as_float(ddm_signal.get("evidence")) or 0.0
    confidence = as_float(ddm_signal.get("confidence_pct")) or 50.0
    agreement = as_float(ddm_signal.get("agreement_pct")) or 50.0

    moment_score = clamp(
        50.0
        + evidence * 28.0
        + (hypothesis_avg - 50.0) * 0.30
        + (tech_score - 50.0) * 0.18
        + (confidence - 50.0) * 0.08
    )
    go_boundary = 62.0
    stop_boundary = 45.0
    if moment_score >= 74 and drift > 0 and agreement >= 58:
        status = "GO_MOMENT"
        gate = "GO"
        exposure = "60-100%"
        reason = "drift가 diffusion을 충분히 이기고 가설 평균도 우호적입니다."
    elif moment_score >= go_boundary and drift > 0:
        status = "PROBING_GO"
        gate = "GO_SMALL"
        exposure = "25-60%"
        reason = "GO boundary 위지만 diffusion 또는 가설 확인이 일부 남아 있습니다."
    elif moment_score <= stop_boundary or drift <= -0.10:
        status = "STOP_MOMENT"
        gate = "STOP"
        exposure = "0%"
        reason = "역방향 drift 또는 높은 diffusion 때문에 신규 리스크를 차단합니다."
    else:
        status = "WAIT_MOMENT"
        gate = "WAIT"
        exposure = "0-25%"
        reason = "GO/STOP boundary 사이의 증거 부족 구간입니다."

    return {
        "enabled": True,
        "status": status,
        "gate": gate,
        "score": round(moment_score, 1),
        "suggested_exposure": exposure,
        "drift": round(drift, 4),
        "diffusion": round(diffusion, 4),
        "evidence": round(evidence, 4),
        "confidence_pct": round(confidence, 1),
        "agreement_pct": round(agreement, 1),
        "hypothesis_avg_score": round(hypothesis_avg, 1),
        "technical_score": round(tech_score, 1),
        "go_boundary": go_boundary,
        "stop_boundary": stop_boundary,
        "distance_to_go": round(moment_score - go_boundary, 1),
        "distance_to_stop": round(moment_score - stop_boundary, 1),
        "reason": reason,
        "formula": "score = 50 + evidence*28 + hypothesis_avg_delta*0.30 + technical_delta*0.18 + confidence_delta*0.08",
    }


def _score_at_index(series: PriceSeries, idx: int) -> float:
    sub = series[: idx + 1]
    snap = compute_technical_snapshot(sub)
    return as_float(snap.get("score")) or 50.0


def _ddm_score_at_index(
    *,
    symbol: str,
    target_series: PriceSeries,
    context_series: Mapping[str, PriceSeries],
    idx: int,
) -> float:
    asof = target_series[idx][0]
    sub_target = target_series[: idx + 1]
    sub_context = {
        sym: _latest_by_date(series, asof)
        for sym, series in context_series.items()
    }
    signal = build_single_stock_ddm_signal(
        symbol=symbol,
        target_series=sub_target,
        context_series=sub_context,
        asof_date=asof,
        corr_lookback_bars=45,
        min_overlap=20,
    )
    if not signal.get("enabled"):
        return 50.0
    evidence = as_float(signal.get("evidence")) or 0.0
    confidence = as_float(signal.get("confidence_pct")) or 50.0
    return clamp(50.0 + evidence * 32.0 + (confidence - 50.0) * 0.18)


def run_walk_forward_backtest(
    *,
    symbol: str,
    target_series: PriceSeries,
    context_series: Mapping[str, PriceSeries],
    horizons: Mapping[str, int] = DEFAULT_HORIZONS,
    lookback: int = 80,
) -> JsonDict:
    if len(target_series) < lookback + 30:
        return {
            "enabled": False,
            "status": "insufficient_history",
            "note": "백테스트에 필요한 가격 히스토리가 부족합니다.",
            "bars": len(target_series),
        }
    strategy_curve = [1.0]
    buyhold_curve = [1.0]
    signals: List[JsonDict] = []
    for idx in range(lookback, len(target_series) - 1):
        tech = _score_at_index(target_series, idx)
        ddm_score = _ddm_score_at_index(
            symbol=symbol,
            target_series=target_series,
            context_series=context_series,
            idx=idx,
        )
        realized_vol = realized_vol_pct(target_series[: idx + 1], 63)
        vol_score = 55.0 if realized_vol is None else clamp(72.0 - max(0.0, realized_vol - 35.0) * 0.35)
        research_score = clamp(tech * 0.45 + ddm_score * 0.38 + vol_score * 0.17)
        exposure = 1.0 if research_score >= 62 else 0.5 if research_score >= 50 else 0.0
        day_return = target_series[idx + 1][1] / target_series[idx][1] - 1.0
        strategy_curve.append(strategy_curve[-1] * (1.0 + day_return * exposure))
        buyhold_curve.append(buyhold_curve[-1] * (1.0 + day_return))
        signals.append(
            {
                "date": target_series[idx][0],
                "score": round(research_score, 1),
                "technical_score": round(tech, 1),
                "ddm_score": round(ddm_score, 1),
                "exposure": exposure,
                "next_day_return_pct": round(day_return * 100.0, 2),
            }
        )

    forward: List[JsonDict] = []
    for name, days in horizons.items():
        rows = []
        for offset, signal in enumerate(signals):
            idx = lookback + offset
            if idx + days >= len(target_series):
                continue
            if float(signal["score"]) < 62:
                continue
            fwd = target_series[idx + days][1] / target_series[idx][1] - 1.0
            rows.append(fwd * 100.0)
        forward.append(
            {
                "horizon": name,
                "days": days,
                "signal_count": len(rows),
                "avg_forward_return_pct": _round(sum(rows) / len(rows) if rows else None),
                "median_forward_return_pct": _round(statistics.median(rows) if rows else None),
                "win_rate_pct": _round(sum(1 for value in rows if value > 0) / len(rows) * 100.0 if rows else None),
                "p10_pct": _round(percentile(rows, 0.10) if rows else None),
                "p90_pct": _round(percentile(rows, 0.90) if rows else None),
            }
        )
    strategy_total = (strategy_curve[-1] - 1.0) * 100.0
    buyhold_total = (buyhold_curve[-1] - 1.0) * 100.0
    exposed = [row for row in signals if row["exposure"] > 0]
    return {
        "enabled": True,
        "bars": len(target_series),
        "test_days": len(signals),
        "strategy_total_return_pct": _round(strategy_total),
        "buyhold_total_return_pct": _round(buyhold_total),
        "excess_return_pct": _round(strategy_total - buyhold_total),
        "strategy_max_drawdown_pct": _round(max_drawdown_pct(strategy_curve)),
        "buyhold_max_drawdown_pct": _round(max_drawdown_pct(buyhold_curve)),
        "avg_exposure_pct": _round(sum(row["exposure"] for row in signals) / len(signals) * 100.0 if signals else None),
        "exposed_next_day_hit_rate_pct": _round(
            sum(1 for row in exposed if row["next_day_return_pct"] > 0) / len(exposed) * 100.0 if exposed else None
        ),
        "forward_horizons": forward,
        "recent_signals": signals[-8:],
        "note": "전일 종가까지의 DDM/기술 점수로 다음 거래일 노출을 정하는 walk-forward 근사입니다.",
    }


def run_monte_carlo_simulation(
    *,
    target_series: PriceSeries,
    ddm_signal: Mapping[str, Any],
    horizons: Mapping[str, int] = DEFAULT_HORIZONS,
    paths: int = 2000,
    seed: int = 20260527,
) -> JsonDict:
    returns = daily_return_list(target_series)[-252:]
    if len(returns) < 40:
        return {
            "enabled": False,
            "status": "insufficient_history",
            "note": "시뮬레이션에 필요한 일일 수익률 표본이 부족합니다.",
        }
    sigma = stdev(returns) or 0.0
    raw_base_mean = sum(returns) / len(returns)
    # Single-stock deep research must not project a parabolic recent tape as a
    # new unconditional regime. Keep the realized volatility sample, but shrink
    # and cap the drift so Monte Carlo is a stress tool, not a promise machine.
    base_mean = clamp(raw_base_mean * 0.15, -0.0015, 0.0015)
    evidence = as_float(ddm_signal.get("evidence")) if ddm_signal.get("enabled") else 0.0
    drift_adjustment = clamp(evidence or 0.0, -1.2, 1.2) * sigma * 0.018
    rng = random.Random(seed)
    output = []
    for name, days in horizons.items():
        terminal_returns: List[float] = []
        max_drawdowns: List[float] = []
        for _ in range(max(100, paths)):
            equity = 1.0
            peak = 1.0
            path_dd = 0.0
            for _day in range(days):
                sampled = rng.choice(returns)
                ret = sampled - raw_base_mean + base_mean + drift_adjustment
                equity *= 1.0 + ret
                peak = max(peak, equity)
                path_dd = min(path_dd, equity / peak - 1.0)
            terminal_returns.append((equity - 1.0) * 100.0)
            max_drawdowns.append(path_dd * 100.0)
        output.append(
            {
                "horizon": name,
                "days": days,
                "p10_return_pct": _round(percentile(terminal_returns, 0.10)),
                "p50_return_pct": _round(percentile(terminal_returns, 0.50)),
                "p90_return_pct": _round(percentile(terminal_returns, 0.90)),
                "prob_gain_pct": _round(sum(1 for value in terminal_returns if value > 0) / len(terminal_returns) * 100.0),
                "prob_loss_10pct_pct": _round(sum(1 for value in terminal_returns if value <= -10.0) / len(terminal_returns) * 100.0),
                "median_max_drawdown_pct": _round(percentile(max_drawdowns, 0.50)),
            }
        )
    return {
        "enabled": True,
        "paths": max(100, paths),
        "seed": seed,
        "sample_days": len(returns),
        "raw_daily_mean_pct": _round(raw_base_mean * 100.0, 3),
        "daily_mean_pct": _round(base_mean * 100.0, 3),
        "daily_vol_pct": _round(sigma * 100.0, 3),
        "ddm_drift_adjustment_daily_pct": _round(drift_adjustment * 100.0, 3),
        "mean_shrinkage_applied": True,
        "decision_use": "stress_test_only",
        "horizons": output,
        "note": "최근 252거래일 수익률 bootstrap은 유지하되 평균수익률은 shrink/cap했습니다. 결론용 예측값이 아니라 변동성 stress test입니다.",
    }


def build_source_status(
    *,
    inventory: Sequence[Mapping[str, Any]],
    topstep_pulse: Optional[Mapping[str, Any]],
    external_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
    access_probe_summary: Optional[Mapping[str, Any]] = None,
) -> Tuple[JsonDict, List[str]]:
    counts: Dict[str, Dict[str, int]] = {}
    for item in inventory:
        provider = str(item.get("provider") or "unknown")
        status = str(item.get("status") or "unknown")
        counts.setdefault(provider, {})
        counts[provider][status] = counts[provider].get(status, 0) + 1
    source_status: JsonDict = {}
    fmp_ok = counts.get("FMP", {}).get("ok", 0)
    fmp_error = counts.get("FMP", {}).get("error", 0)
    massive_ok = counts.get("Massive", {}).get("ok", 0)
    massive_error = counts.get("Massive", {}).get("error", 0)
    source_status["FMP"] = f"확인됨: {fmp_ok}개 endpoint ok, {fmp_error}개 제한/실패."
    source_status["Massive"] = f"확인됨: {massive_ok}개 endpoint ok, {massive_error}개 제한/실패."
    if topstep_pulse and not topstep_pulse.get("error"):
        source_status["TopstepX futures MCP"] = (
            f"확인됨: tone={topstep_pulse.get('tone')}, risk_on_score={fmt_num(topstep_pulse.get('risk_on_score'))}."
        )
    else:
        source_status["TopstepX futures MCP"] = "부분 제한/실패: futures pulse가 없거나 오류가 기록됐습니다."
    evidence = list(external_evidence or [])
    evidence_text = json.dumps(evidence, ensure_ascii=False).lower()
    barchart_total = 0
    barchart_ok = 0
    for item in evidence:
        pages = item.get("barchart") if isinstance(item, Mapping) else None
        if isinstance(pages, list):
            barchart_total += len(pages)
            barchart_ok += sum(1 for page in pages if isinstance(page, Mapping) and page.get("status") == "ok")
    has_barchart = barchart_total > 0
    has_barchart_confirmed = barchart_ok > 0
    has_community = any(token in evidence_text for token in ("reddit", "naver", "cafe", "community"))
    has_web = any(token in evidence_text for token in ("web", "search", "sec.gov", "official"))
    source_status["Barchart Premier"] = (
        f"확인됨/부분 제한: Chrome 자동 evidence {barchart_ok}/{barchart_total} 페이지 반영." if has_barchart_confirmed else
        "실패/부분 제한: Barchart evidence는 생성됐지만 확인된 페이지가 없습니다." if has_barchart else
        "부분 제한: 이번 CLI 실행에는 Barchart Chrome evidence JSON이 입력되지 않았습니다."
    )
    source_status["Community/Web"] = (
        "확인됨/부분 제한: 커뮤니티/웹 evidence JSON으로 반영." if (has_community or has_web) else
        "부분 제한: FMP/Massive 뉴스는 반영했지만 네이버/Reddit/일반 웹 evidence JSON은 입력되지 않았습니다."
    )
    if access_probe_summary:
        source_status["Access probe"] = "확인됨: 최신 access_probe 요약을 보고서 메타에 포함했습니다."
    unused: List[str] = []
    if not has_barchart_confirmed:
        unused.append(
            "Barchart Premier: Chrome 자동 evidence에서 확인된 페이지가 없거나 제한됐습니다. 대체로 FMP/Massive 옵션 계약·뉴스·가격 데이터를 사용했으며, 옵션 flow/감마/맥스페인/put-call 신뢰도는 낮아집니다. 다음 실행에서는 Chrome 로그인 상태와 AppleScript JavaScript 실행 권한을 확인해야 합니다."
        )
    if not has_community:
        unused.append(
            "커뮤니티: 네이버 카페/Reddit listing evidence가 입력되지 않아 개인투자자 분위기 점수는 계산하지 않았습니다. 대체로 FMP/Massive 뉴스와 가격/수급 데이터를 사용했으며, crowding/FOMO 판단 신뢰도는 제한됩니다."
        )
    if not topstep_pulse or topstep_pulse.get("error"):
        unused.append(
            "TopstepX/선물 MCP: futures pulse가 실패했거나 입력되지 않아 DDM의 macro_prior가 약화됐습니다. 다음 실행에서는 TopstepX env와 callable MCP 노출을 확인해야 합니다."
        )
    return source_status, unused


def build_investment_conclusion(
    *,
    hypotheses: Sequence[Mapping[str, Any]],
    technical: Mapping[str, Any],
    ddm_signal: Mapping[str, Any],
    gostop_moment: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    score_by_id = {str(item.get("id")): as_float(item.get("score")) or 50.0 for item in hypotheses}
    overall = (
        score_by_id.get("value_investment_merit", 50.0) * 0.23
        + score_by_id.get("relative_attractiveness", 50.0) * 0.17
        + score_by_id.get("remaining_upside", 50.0) * 0.18
        + score_by_id.get("roadmap_milestones", 50.0) * 0.16
        + score_by_id.get("moat", 50.0) * 0.18
        + (as_float(technical.get("score")) or 50.0) * 0.08
    )
    if ddm_signal.get("enabled"):
        overall += clamp((as_float(ddm_signal.get("evidence")) or 0.0) * 4.0, -4, 4)
    if gostop_moment and gostop_moment.get("enabled"):
        overall += clamp(((as_float(gostop_moment.get("score")) or 50.0) - 50.0) * 0.08, -3, 3)
    overall = clamp(overall)
    if overall >= 78:
        label = "High Conviction Candidate"
        stance = "가치·상대매력·DDM이 함께 맞는 구간입니다. 그래도 실거래는 분할과 이벤트 체크가 필요합니다."
    elif overall >= 64:
        label = "Selective Accumulate"
        stance = "투자 매력은 있으나 한두 축의 확인이 더 필요합니다. 눌림/마일스톤 확인 기반 접근이 적절합니다."
    elif overall >= 50:
        label = "Watch / Prove It"
        stance = "관심 후보지만 현재는 가격, diffusion, 또는 가치 축 중 하나가 충분히 증명되지 않았습니다."
    else:
        label = "Avoid / De-risk"
        stance = "현재 수치 조합은 신규 리스크를 정당화하기 어렵습니다."
    return {
        "score": round(overall, 1),
        "label": label,
        "stance": stance,
        "one_line": (
            f"{label}: 종합 점수 {overall:.1f}/100, "
            f"DDM {ddm_signal.get('status', 'N/A')}, "
            f"GoStop {((gostop_moment or {}).get('gate') or 'N/A')}."
        ),
    }


def build_horizon_scores(
    *,
    hypotheses: Sequence[Mapping[str, Any]],
    technical: Mapping[str, Any],
    ddm_signal: Mapping[str, Any],
    gostop_moment: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    scores = {str(item.get("id")): as_float(item.get("score")) or 50.0 for item in hypotheses}
    tech = as_float(technical.get("score")) or 50.0
    ddm = 50.0
    if ddm_signal.get("enabled"):
        ddm = clamp(50.0 + (as_float(ddm_signal.get("evidence")) or 0.0) * 30.0)
    moment = as_float((gostop_moment or {}).get("score")) or ddm
    return {
        "1w": round(clamp(tech * 0.32 + ddm * 0.26 + moment * 0.24 + scores.get("relative_attractiveness", 50.0) * 0.18), 1),
        "1m": round(clamp(tech * 0.25 + ddm * 0.25 + moment * 0.22 + scores.get("relative_attractiveness", 50.0) * 0.16 + scores.get("remaining_upside", 50.0) * 0.12), 1),
        "3m": round(clamp(ddm * 0.20 + moment * 0.18 + scores.get("relative_attractiveness", 50.0) * 0.22 + scores.get("remaining_upside", 50.0) * 0.22 + scores.get("roadmap_milestones", 50.0) * 0.18), 1),
        "6m": round(clamp(scores.get("value_investment_merit", 50.0) * 0.28 + scores.get("remaining_upside", 50.0) * 0.25 + scores.get("roadmap_milestones", 50.0) * 0.22 + scores.get("moat", 50.0) * 0.25), 1),
        "1y": round(clamp(scores.get("value_investment_merit", 50.0) * 0.34 + scores.get("moat", 50.0) * 0.30 + scores.get("roadmap_milestones", 50.0) * 0.18 + scores.get("remaining_upside", 50.0) * 0.18), 1),
    }


def build_research_quality_audit(
    *,
    upside: Mapping[str, Any],
    research_intelligence: Mapping[str, Any],
    simulation: Mapping[str, Any],
    backtest: Mapping[str, Any],
    gostop_moment: Mapping[str, Any],
    hypotheses: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Detect contradictions that must be resolved before publishing.

    This is the intelligence layer that prevents a data-rich report from
    becoming a source dump. It asks whether valuation, options heat, SEC
    parsing, simulation, and the final action all tell a coherent story.
    """

    ledger: List[JsonDict] = []

    def add_issue(
        severity: str,
        issue_type: str,
        evidence: str,
        required_resolution: str,
        impact: str,
    ) -> None:
        ledger.append(
            {
                "severity": severity,
                "type": issue_type,
                "evidence": evidence,
                "required_resolution": required_resolution,
                "impact": impact,
            }
        )

    scores = {str(item.get("id")): as_float(item.get("score")) or 50.0 for item in hypotheses}
    remaining_score = scores.get("remaining_upside", 50.0)
    dcf_upside = as_float(upside.get("dcf_upside_pct"))
    gate = str(gostop_moment.get("gate") or "WAIT")
    exposure_upper = _exposure_upper_pct(gostop_moment.get("suggested_exposure"))
    option_metrics = (research_intelligence.get("barchart") or {}).get("option_metrics") or {}
    iv_percentile = _parse_percent_metric(option_metrics.get("iv_percentile"))
    iv_rank = _parse_percent_metric(option_metrics.get("iv_rank"))
    iv_heat = max(value for value in (iv_percentile, iv_rank, 0.0) if value is not None)
    sec_quality = research_intelligence.get("sec") or {}
    raw_xbrl_count = int(as_float(sec_quality.get("raw_xbrl_snippet_count")) or 0)

    simulation_warnings: List[str] = []
    p50_thresholds = {"3m": 60.0, "6m": 120.0, "1y": 250.0}
    for row in simulation.get("horizons") or []:
        horizon = str(row.get("horizon") or "")
        p50 = as_float(row.get("p50_return_pct"))
        prob_gain = as_float(row.get("prob_gain_pct"))
        threshold = p50_thresholds.get(horizon)
        if threshold is not None and p50 is not None and p50 > threshold:
            simulation_warnings.append(f"{horizon} p50 {p50:.1f}% > sanity cap {threshold:.0f}%")
        if horizon in {"3m", "6m", "1y"} and prob_gain is not None and prob_gain >= 99.5:
            simulation_warnings.append(f"{horizon} 상승확률 {prob_gain:.1f}%는 과적합 가능성이 높음")
    if simulation_warnings:
        add_issue(
            "high",
            "simulation_overfit",
            "; ".join(simulation_warnings[:4]),
            "Monte Carlo는 stress-only로 낮추고 기대수익/승률을 투자 결론 근거에서 제외합니다.",
            "점수와 비중을 자동 보수화합니다.",
        )

    backtest_warnings: List[str] = []
    for row in backtest.get("forward_horizons") or []:
        horizon = str(row.get("horizon") or "")
        win_rate = as_float(row.get("win_rate_pct"))
        signal_count = int(as_float(row.get("signal_count")) or 0)
        if horizon in {"3m", "6m", "1y"} and signal_count >= 3 and win_rate is not None and win_rate >= 99.5:
            backtest_warnings.append(f"{horizon} 신호 {signal_count}개 승률 {win_rate:.1f}%")
    if backtest_warnings:
        add_issue(
            "medium",
            "backtest_perfect_hit_rate",
            "; ".join(backtest_warnings[:4]),
            "완벽한 승률은 edge가 아니라 표본/국면 편향 신호로 취급합니다.",
            "백테스트는 방향 검증 보조로만 사용합니다.",
        )

    if dcf_upside is not None and dcf_upside <= -20.0 and gate in {"GO", "GO_SMALL"}:
        add_issue(
            "high",
            "valuation_action_conflict",
            f"DCF upside {dcf_upside:.1f}%인데 GoStop gate가 {gate}.",
            "좋은 회사/강한 모멘텀과 좋은 진입을 분리해 pullback 또는 event-only로 낮춥니다.",
            "GO 결론과 60% 이상 노출을 허용하지 않습니다.",
        )
    if remaining_score < 50.0 and gate in {"GO", "GO_SMALL"}:
        add_issue(
            "medium",
            "remaining_upside_conflict",
            f"remaining-upside score {remaining_score:.1f}/100인데 GoStop gate가 {gate}.",
            "상승 여력 부족을 모멘텀 thesis와 분리해 노출 상한을 둡니다.",
            "단기 추격보다 눌림/이벤트 확인형으로 번역합니다.",
        )
    if iv_heat >= 90.0 and gate in {"GO", "GO_SMALL"}:
        add_issue(
            "high" if iv_heat >= 95.0 else "medium",
            "options_heat_action_conflict",
            f"IV rank/percentile max {iv_heat:.1f}%인데 GoStop gate가 {gate}.",
            "옵션 과열은 방향 신호가 아니라 event premium/vol crush 리스크로 별도 처리합니다.",
            "진입은 pullback, spread, event hedge, 또는 post-event 확인으로 제한합니다.",
        )
    if exposure_upper is not None and exposure_upper >= 60.0 and any(
        item.get("type") in {"valuation_action_conflict", "options_heat_action_conflict"}
        for item in ledger
    ):
        add_issue(
            "high",
            "sizing_conflict",
            f"suggested exposure upper {exposure_upper:.0f}% conflicts with valuation/IV brakes.",
            "비중 제안은 품질 게이트 이후에만 산출합니다.",
            "노출 상한을 0-25% 또는 0-50%로 재설정합니다.",
        )
    if raw_xbrl_count:
        add_issue(
            "medium",
            "sec_raw_xbrl_leakage",
            f"{raw_xbrl_count} SEC snippets looked like taxonomy/meta rather than prose.",
            "MD&A/risk/liquidity/customer/capex 문장으로 재추출하기 전까지 SEC insight 점수를 낮춥니다.",
            "공시 근거 신뢰도를 부분 제한으로 표시합니다.",
        )

    high_count = sum(1 for item in ledger if item.get("severity") == "high")
    medium_count = sum(1 for item in ledger if item.get("severity") == "medium")
    if high_count or medium_count >= 2:
        verdict = "requires_revision"
    elif medium_count:
        verdict = "pass_with_warnings"
    else:
        verdict = "pass"

    market_conflict = any(
        item.get("type") in {
            "valuation_action_conflict",
            "options_heat_action_conflict",
            "sizing_conflict",
            "remaining_upside_conflict",
        }
        for item in ledger
    )
    if high_count and market_conflict:
        action_override = {
            "enabled": True,
            "gate": "WAIT",
            "status": "QUALITY_GATED_WAIT",
            "suggested_exposure": "0-25%",
            "score_cap": 62.0,
            "label": "Momentum Extended / Pullback Required",
            "stance": "모멘텀은 인정하지만 DCF/IV/비중 충돌이 해소되지 않아 추격 진입은 보류합니다.",
        }
    elif verdict != "pass":
        action_override = {
            "enabled": True,
            "gate": "WAIT" if gate == "GO" else gate,
            "status": "QUALITY_GATED_SELECTIVE",
            "suggested_exposure": "0-50%",
            "score_cap": 70.0,
            "label": "Selective / Evidence Conflict",
            "stance": "근거는 있으나 품질 게이트가 경고를 냈기 때문에 결론은 보수화합니다.",
        }
    else:
        action_override = {"enabled": False}

    return {
        "enabled": True,
        "verdict": verdict,
        "high_count": high_count,
        "medium_count": medium_count,
        "contradiction_ledger": ledger,
        "action_override": action_override,
        "simulation_quality": {
            "use_for_decision": not simulation_warnings,
            "warnings": simulation_warnings,
            "rule": "3m p50>60%, 6m p50>120%, 1y p50>250%, or gain probability >=99.5% means overfit/stress-only.",
        },
        "sec_quality": {
            "parse_status": sec_quality.get("parse_status") or "unknown",
            "raw_xbrl_snippet_count": raw_xbrl_count,
            "use_for_scoring": raw_xbrl_count == 0,
        },
        "decision_rule": "좋은 회사/좋은 모멘텀/좋은 진입은 분리합니다. DCF downside, IV heat, SEC parse failure, perfect backtest, overfit simulation이 있으면 score/exposure/action을 강등합니다.",
    }


def apply_quality_gate_to_gostop(
    gostop_moment: Mapping[str, Any],
    quality_audit: Mapping[str, Any],
) -> JsonDict:
    out = dict(gostop_moment)
    override = quality_audit.get("action_override") or {}
    if not override.get("enabled"):
        out["quality_gate_applied"] = False
        return out
    out["quality_gate_applied"] = True
    out["original_gate"] = gostop_moment.get("gate")
    out["original_status"] = gostop_moment.get("status")
    out["original_suggested_exposure"] = gostop_moment.get("suggested_exposure")
    out["gate"] = override.get("gate", out.get("gate"))
    out["status"] = override.get("status", out.get("status"))
    out["suggested_exposure"] = override.get("suggested_exposure", out.get("suggested_exposure"))
    out["reason"] = (
        f"품질 게이트 적용: {override.get('stance')} "
        f"원래 판단은 {gostop_moment.get('gate')} / {gostop_moment.get('suggested_exposure')}였습니다. "
        f"{gostop_moment.get('reason')}"
    )
    return out


def apply_quality_gate_to_conclusion(
    conclusion: Mapping[str, Any],
    quality_audit: Mapping[str, Any],
) -> JsonDict:
    out = dict(conclusion)
    override = quality_audit.get("action_override") or {}
    if not override.get("enabled"):
        out["quality_gate_applied"] = False
        return out
    original_score = as_float(out.get("score")) or 50.0
    cap = as_float(override.get("score_cap")) or original_score
    score = min(original_score, cap)
    out.update(
        {
            "quality_gate_applied": True,
            "original_label": conclusion.get("label"),
            "original_score": conclusion.get("score"),
            "score": round(score, 1),
            "label": override.get("label") or conclusion.get("label"),
            "stance": override.get("stance") or conclusion.get("stance"),
        }
    )
    out["one_line"] = (
        f"{out['label']}: 품질 게이트 후 종합 점수 {score:.1f}/100. "
        f"원래 결론 {conclusion.get('label')} {original_score:.1f}/100은 "
        f"{quality_audit.get('verdict')} 판정으로 보수화했습니다."
    )
    return out


def apply_quality_gate_to_horizon_scores(
    horizon_scores: Mapping[str, Any],
    quality_audit: Mapping[str, Any],
) -> JsonDict:
    override = quality_audit.get("action_override") or {}
    if not override.get("enabled"):
        return dict(horizon_scores)
    caps = {"1w": 65.0, "1m": 62.0, "3m": 58.0, "6m": 55.0, "1y": 58.0}
    if override.get("suggested_exposure") == "0-25%":
        caps.update({"1w": 58.0, "1m": 56.0, "3m": 54.0, "6m": 52.0, "1y": 55.0})
    return {
        key: round(min(as_float(value) or 50.0, caps.get(key, 70.0)), 1)
        for key, value in horizon_scores.items()
    }


def _clean_evidence_text(value: Any, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _find_labeled_number(text: str, label: str) -> Optional[str]:
    pattern = re.compile(
        re.escape(label) + r"\s*[:\n]?\s*([+-]?\$?\d[\d,]*(?:\.\d+)?%?(?:\s*x\s*\d+)?)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _sentences_with_keywords(text: str, keywords: Sequence[str], *, limit: int = 8) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
    out: List[str] = []
    lowered_keywords = [word.lower() for word in keywords]
    for chunk in chunks:
        compact = _clean_evidence_text(chunk, 260)
        if len(compact) < 25:
            continue
        lowered = compact.lower()
        if any(word in lowered for word in lowered_keywords):
            if compact not in out:
                out.append(compact)
        if len(out) >= limit:
            break
    return out


def build_research_intelligence(
    *,
    symbol: str,
    external_evidence: Sequence[Mapping[str, Any]],
    datasets: Mapping[str, Any],
    hypotheses: Sequence[Mapping[str, Any]],
    ddm_signal: Mapping[str, Any],
    technical: Mapping[str, Any],
) -> JsonDict:
    """Turn raw page captures and filings into an explicit research thesis.

    This is intentionally extractive and auditable: the report shows which
    Barchart pages, community pages, and SEC documents drove each inference.
    """

    barchart_pages: List[Mapping[str, Any]] = []
    reddit_matches: List[Mapping[str, Any]] = []
    cafe_payloads: List[Mapping[str, Any]] = []
    sec_docs: List[Mapping[str, Any]] = []
    sec_submissions: Mapping[str, Any] = {}
    for item in external_evidence:
        pages = item.get("barchart") if isinstance(item, Mapping) else None
        if isinstance(pages, list):
            barchart_pages.extend(page for page in pages if isinstance(page, Mapping))
        matches = item.get("matched_rows") if isinstance(item, Mapping) else None
        if isinstance(matches, list):
            reddit_matches.extend(row for row in matches if isinstance(row, Mapping))
        cafe = item.get("cafe") if isinstance(item, Mapping) else None
        if isinstance(cafe, Mapping):
            cafe_payloads.append(cafe)
        web = item.get("web") if isinstance(item, Mapping) else None
        sec = web.get("sec") if isinstance(web, Mapping) else None
        if isinstance(sec, Mapping):
            docs = sec.get("filing_documents")
            if isinstance(docs, list):
                sec_docs.extend(doc for doc in docs if isinstance(doc, Mapping))
            submissions = sec.get("submissions")
            if isinstance(submissions, Mapping):
                sec_submissions = submissions

    barchart_ok = [page for page in barchart_pages if page.get("status") == "ok"]
    barchart_by_label = {str(page.get("label")): page for page in barchart_ok}
    all_barchart_text = "\n".join(str(page.get("text") or "") for page in barchart_ok)
    overview_text = str((barchart_by_label.get("Overview") or {}).get("text") or "")
    opinion_text = str((barchart_by_label.get("Barchart Opinion") or {}).get("text") or "")
    technical_text = str((barchart_by_label.get("Technical Analysis") or {}).get("text") or "")
    options_text = "\n".join(
        str((barchart_by_label.get(label) or {}).get("text") or "")
        for label in (
            "Options Prices",
            "Volatility & Greeks",
            "Options Flow",
            "Unusual Options Activity",
            "Put/Call Ratio",
            "Gamma Exposure",
            "Max Pain & Vol Skew",
            "Expected Move",
            "Options Data",
        )
    )
    news_text = "\n".join(
        str((barchart_by_label.get(label) or {}).get("text") or "")
        for label in ("Overview", "News")
    )

    option_metrics = {
        "implied_volatility": _find_labeled_number(options_text or overview_text, "Implied Volatility"),
        "historical_volatility": _find_labeled_number(options_text or overview_text, "Historical Volatility"),
        "iv_percentile": _find_labeled_number(options_text or overview_text, "IV Percentile"),
        "iv_rank": _find_labeled_number(options_text or overview_text, "IV Rank"),
        "expected_move": _find_labeled_number(options_text or overview_text, "Expected Move"),
        "put_call_vol_ratio": _find_labeled_number(options_text or overview_text, "Put/Call Vol Ratio"),
        "put_call_oi_ratio": _find_labeled_number(options_text or overview_text, "Put/Call OI Ratio"),
        "open_interest": _find_labeled_number(options_text or overview_text, "Today's Open Interest"),
    }
    turning_points = {
        "resistance_3": _find_labeled_number(overview_text, "3rd Resistance Point"),
        "resistance_2": _find_labeled_number(overview_text, "2nd Resistance Point"),
        "resistance_1": _find_labeled_number(overview_text, "1st Resistance Point"),
        "support_1": _find_labeled_number(overview_text, "1st Support Level"),
        "support_2": _find_labeled_number(overview_text, "2nd Support Level"),
        "support_3": _find_labeled_number(overview_text, "3rd Support Level"),
    }
    opinion_snippets = _sentences_with_keywords(
        opinion_text + "\n" + overview_text + "\n" + technical_text,
        ["strong buy", "buy", "overbought", "trend", "relative strength", "reversal"],
        limit=7,
    )
    news_snippets = _sentences_with_keywords(
        news_text,
        ["price target", "volatility", "dram", "ai", "undervalued", "math problem", "earnings", "manufacturing"],
        limit=8,
    )
    sec_snippets: List[str] = []
    raw_xbrl_snippet_count = 0
    for doc in sec_docs:
        targeted = doc.get("targeted_snippets")
        if isinstance(targeted, list):
            for snippet in targeted:
                compact = _clean_evidence_text(snippet, 360)
                if _looks_like_raw_sec_metadata(compact):
                    raw_xbrl_snippet_count += 1
                    continue
                if compact:
                    sec_snippets.append(
                        f"{doc.get('form')} {doc.get('filing_date')}: {compact}"
                    )
                if len(sec_snippets) >= 8:
                    break
        if len(sec_snippets) >= 8:
            break
        text = str(doc.get("text") or "")
        hits = _sentences_with_keywords(
            text,
            ["risk", "demand", "supply", "customer", "inventory", "manufacturing", "china", "competition", "capital"],
            limit=3,
        )
        for hit in hits:
            if _looks_like_raw_sec_metadata(hit):
                raw_xbrl_snippet_count += 1
                continue
            sec_snippets.append(
                f"{doc.get('form')} {doc.get('filing_date')}: {hit}"
            )
        if len(sec_snippets) >= 8:
            break
    sec_limitations: List[str] = []
    if raw_xbrl_snippet_count:
        sec_limitations.append(
            f"원시 XBRL/taxonomy 형태 snippet {raw_xbrl_snippet_count}개는 공시 인사이트에서 제외했습니다."
        )
    if sec_docs and not sec_snippets:
        sec_limitations.append(
            "SEC 문서는 수집됐지만 사람이 읽을 수 있는 MD&A/risk/liquidity/customer/capex 문장 추출이 부족합니다."
        )
    if sec_snippets:
        sec_parse_status = "prose_snippets_ok"
    elif sec_docs:
        sec_parse_status = "text_parse_limited"
    else:
        sec_parse_status = "unavailable"

    cafe_pages: List[Mapping[str, Any]] = []
    cafe_latest_ok = 0
    cafe_mobile_search_ok = 0
    cafe_internal_search_ok = 0
    cafe_search_ok = 0
    cafe_home_ok = 0
    cafe_text = ""
    for cafe in cafe_payloads:
        for home_key in ("mobile_home", "home"):
            home = cafe.get(home_key)
            if isinstance(home, Mapping):
                cafe_pages.append(home)
                cafe_home_ok += 1 if home.get("status") == "ok" else 0
                cafe_text += "\n" + str(home.get("text") or "")
        latest = cafe.get("latest_list_pages")
        if isinstance(latest, list):
            cafe_pages.extend(page for page in latest if isinstance(page, Mapping))
            cafe_latest_ok += sum(1 for page in latest if isinstance(page, Mapping) and page.get("status") == "ok")
            cafe_text += "\n".join(str(page.get("text") or "") for page in latest if isinstance(page, Mapping))
        mobile_search = cafe.get("mobile_search_pages")
        if isinstance(mobile_search, list):
            cafe_pages.extend(page for page in mobile_search if isinstance(page, Mapping))
            cafe_mobile_search_ok += sum(
                1 for page in mobile_search if isinstance(page, Mapping) and page.get("status") == "ok"
            )
            cafe_text += "\n".join(
                str(page.get("text") or "") for page in mobile_search if isinstance(page, Mapping)
            )
        internal_search = cafe.get("internal_search_pages")
        if isinstance(internal_search, list):
            cafe_pages.extend(page for page in internal_search if isinstance(page, Mapping))
            cafe_internal_search_ok += sum(
                1 for page in internal_search if isinstance(page, Mapping) and page.get("status") == "ok"
            )
            cafe_text += "\n".join(
                str(page.get("text") or "") for page in internal_search if isinstance(page, Mapping)
            )
        pages = cafe.get("pages")
        if isinstance(pages, list):
            cafe_pages.extend(page for page in pages if isinstance(page, Mapping))
            cafe_search_ok += sum(1 for page in pages if isinstance(page, Mapping) and page.get("status") == "ok")
            cafe_text += "\n".join(str(page.get("text") or "") for page in pages if isinstance(page, Mapping))
    community_titles = [
        _clean_evidence_text(row.get("title"), 180)
        for row in reddit_matches[:8]
        if row.get("title")
    ]
    cafe_keywords = []
    for keyword in ("미국주식", "나스닥", "엔비디아", "etf", "주식", symbol.upper(), "반도체", "금리", "선물"):
        count = cafe_text.lower().count(keyword.lower())
        if count:
            cafe_keywords.append({"keyword": keyword, "count": count})

    scores = {str(item.get("id")): as_float(item.get("score")) or 50.0 for item in hypotheses}
    ddm_evidence = as_float(ddm_signal.get("evidence")) or 0.0
    tech_score = as_float(technical.get("score")) or 50.0
    option_risk = 0.0
    iv_rank = as_float(str(option_metrics.get("iv_rank") or "").replace("%", ""))
    iv_percentile = as_float(str(option_metrics.get("iv_percentile") or "").replace("%", ""))
    if (iv_rank is not None and iv_rank >= 80) or (iv_percentile is not None and iv_percentile >= 80):
        option_risk += 12.0
    if "overbought" in (opinion_text + overview_text + technical_text).lower():
        option_risk += 8.0
    thesis_score = clamp(
        (scores.get("value_investment_merit", 50.0) * 0.18)
        + (scores.get("relative_attractiveness", 50.0) * 0.16)
        + (scores.get("remaining_upside", 50.0) * 0.16)
        + (scores.get("roadmap_milestones", 50.0) * 0.14)
        + (scores.get("moat", 50.0) * 0.14)
        + tech_score * 0.10
        + clamp(50.0 + ddm_evidence * 35.0) * 0.12
        - option_risk * 0.22
    )
    if thesis_score >= 78:
        thesis = "AI-cycle memory winner thesis is supported, but execution should respect volatility and event risk."
    elif thesis_score >= 62:
        thesis = "Constructive thesis, but entry quality depends on pullback, options stress, and milestone confirmation."
    elif thesis_score >= 48:
        thesis = "Balanced thesis with unresolved valuation or catalyst gaps."
    else:
        thesis = "Risk/reward is not yet proven by the combined evidence set."

    confirms = []
    if ddm_signal.get("status") in ("boost", "constructive"):
        confirms.append(f"DDM {ddm_signal.get('status')} with evidence {fmt_num(ddm_signal.get('evidence'), 3)}.")
    if opinion_snippets:
        confirms.append("Barchart technical/opinion text supports the trend.")
    if news_snippets:
        confirms.append("Barchart/news narratives identify memory, AI, or price-target catalysts.")
    if sec_snippets:
        confirms.append("SEC official filings were pulled for risk and business context.")
    challenges = []
    if option_risk:
        challenges.append("Options/technical evidence indicates elevated volatility or overbought risk.")
    if scores.get("remaining_upside", 50.0) < 55:
        challenges.append("Remaining-upside hypothesis is weak relative to current price run-up.")
    if raw_xbrl_snippet_count:
        challenges.append("SEC evidence contained raw XBRL taxonomy/meta snippets; those were filtered and SEC insight is parse-limited.")
    if cafe_mobile_search_ok == 0 and cafe_latest_ok == 0:
        challenges.append("Naver Cafe mobile/internal pages did not yield full article rows; search fallback remains partial.")
    if not sec_docs:
        challenges.append("SEC filing document text was unavailable; only filing index/companyfacts were used.")

    return {
        "enabled": True,
        "method": "extractive evidence synthesis over Barchart page text, SEC filing text, community rows, DDM, and numeric hypotheses",
        "thesis_score": round(thesis_score, 1),
        "thesis": thesis,
        "confirmed_evidence": confirms[:8],
        "challenge_evidence": challenges[:8],
        "barchart": {
            "confirmed_pages": len(barchart_ok),
            "labels": [str(page.get("label")) for page in barchart_ok[:40]],
            "opinion_snippets": opinion_snippets,
            "news_snippets": news_snippets,
            "option_metrics": option_metrics,
            "turning_points": turning_points,
        },
        "sec": {
            "company": sec_submissions.get("name"),
            "sic": sec_submissions.get("sic"),
            "sic_description": sec_submissions.get("sicDescription"),
            "recent_forms": sec_submissions.get("forms", [])[:12],
            "filing_documents_count": len([doc for doc in sec_docs if doc.get("status") == "ok"]),
            "filing_snippets": sec_snippets[:8],
            "parse_status": sec_parse_status,
            "raw_xbrl_snippet_count": raw_xbrl_snippet_count,
            "limitations": sec_limitations,
        },
        "community": {
            "reddit_matched_count": len(reddit_matches),
            "reddit_titles": community_titles,
            "naver_home_ok": cafe_home_ok,
            "naver_latest_pages_ok": cafe_latest_ok,
            "naver_mobile_search_pages_ok": cafe_mobile_search_ok,
            "naver_internal_search_pages_ok": cafe_internal_search_ok,
            "naver_search_pages_ok": cafe_search_ok,
            "naver_keywords": cafe_keywords[:12],
        },
        "open_questions": [
            "Is the next earnings print strong enough to validate the current implied-volatility premium?",
            "Do DRAM/NAND contract prices keep rising after the current AI restocking wave?",
            "Does options positioning create a squeeze tailwind or a post-event volatility crush risk?",
            "Are SEC filing risks around cyclicality, capex, customer concentration, or export controls worsening?",
        ],
    }


def build_stock_research_payload(
    *,
    symbol: str,
    company_name: Optional[str],
    datasets: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    context_series: Mapping[str, PriceSeries],
    topstep_pulse: Optional[Mapping[str, Any]] = None,
    external_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
    access_probe_summary: Optional[Mapping[str, Any]] = None,
    simulation_paths: int = 2000,
) -> JsonDict:
    target_series = price_series_from_datasets(datasets, symbol)
    technical = compute_technical_snapshot(target_series)
    fundamentals = compute_fundamental_snapshot(datasets)
    ddm_signal = build_single_stock_ddm_signal(
        symbol=symbol,
        target_series=target_series,
        context_series=context_series,
        topstep_pulse=topstep_pulse,
    )
    relative = compute_relative_snapshot(
        symbol=symbol,
        target_series=target_series,
        context_series=context_series,
        datasets=datasets,
    )
    upside = compute_upside_snapshot(
        target_series=target_series,
        datasets=datasets,
        ddm_signal=ddm_signal,
    )
    moat = compute_moat_snapshot(datasets, fundamentals)
    roadmap = build_roadmap_snapshot(datasets)
    hypotheses = build_hypotheses(
        fundamentals=fundamentals,
        relative=relative,
        upside=upside,
        roadmap=roadmap,
        moat=moat,
        ddm_signal=ddm_signal,
    )
    initial_gostop_moment = build_gostop_moment(
        ddm_signal=ddm_signal,
        hypotheses=hypotheses,
        technical=technical,
    )
    backtest = run_walk_forward_backtest(
        symbol=symbol,
        target_series=target_series,
        context_series=context_series,
    )
    simulation = run_monte_carlo_simulation(
        target_series=target_series,
        ddm_signal=ddm_signal,
        paths=simulation_paths,
    )
    source_status, unused = build_source_status(
        inventory=inventory,
        topstep_pulse=topstep_pulse,
        external_evidence=external_evidence,
        access_probe_summary=access_probe_summary,
    )
    research_intelligence = build_research_intelligence(
        symbol=symbol,
        external_evidence=external_evidence or [],
        datasets=datasets,
        hypotheses=hypotheses,
        ddm_signal=ddm_signal,
        technical=technical,
    )
    quality_audit = build_research_quality_audit(
        upside=upside,
        research_intelligence=research_intelligence,
        simulation=simulation,
        backtest=backtest,
        gostop_moment=initial_gostop_moment,
        hypotheses=hypotheses,
    )
    gostop_moment = apply_quality_gate_to_gostop(
        initial_gostop_moment,
        quality_audit,
    )
    conclusion = build_investment_conclusion(
        hypotheses=hypotheses,
        technical=technical,
        ddm_signal=ddm_signal,
        gostop_moment=gostop_moment,
    )
    conclusion = apply_quality_gate_to_conclusion(conclusion, quality_audit)
    horizon_scores = build_horizon_scores(
        hypotheses=hypotheses,
        technical=technical,
        ddm_signal=ddm_signal,
        gostop_moment=gostop_moment,
    )
    horizon_scores = apply_quality_gate_to_horizon_scores(horizon_scores, quality_audit)
    return {
        "symbol": symbol.upper(),
        "company": company_name or symbol.upper(),
        "generated_at_utc": _now_utc(),
        "process_contract": {
            "target_runtime_minutes": 60,
            "phases": [
                "0-10m data inventory/access check",
                "10-20m hypothesis setup and endpoint collection",
                "20-35m DDM context/futures/relative basket",
                "35-50m walk-forward backtest and Monte Carlo",
                "50-60m 420px report QA and source limitation statement",
            ],
        },
        "investment_conclusion": conclusion,
        "horizon_scores": horizon_scores,
        "technical": technical,
        "fundamentals": fundamentals,
        "relative": relative,
        "upside": upside,
        "roadmap": roadmap,
        "moat": moat,
        "ddm_signal": ddm_signal,
        "gostop_moment": gostop_moment,
        "research_quality_audit": quality_audit,
        "research_intelligence": research_intelligence,
        "hypotheses": hypotheses,
        "backtest": backtest,
        "simulation": simulation,
        "source_status": source_status,
        "unused_data_source_statement": unused,
        "access_probe_summary": access_probe_summary or {},
        "external_evidence_summary": summarize_external_evidence(external_evidence or []),
        "source_inventory": list(inventory),
    }


def summarize_external_evidence(evidence: Sequence[Mapping[str, Any]]) -> JsonDict:
    files = []
    for item in evidence:
        files.append(
            {
                "source": item.get("_source_file") or item.get("source") or "inline",
                "keys": sorted(str(key) for key in item.keys())[:20],
            }
        )
    return {"count": len(files), "files": files}


def _tone_class(score: Any) -> str:
    number = as_float(score)
    if number is None:
        return "neutral"
    if number >= 70:
        return "good"
    if number >= 52:
        return "warn"
    return "bad"


def _bar(score: Any) -> str:
    number = clamp(as_float(score) or 0.0)
    return f'<div class="bar"><span style="width:{number:.1f}%"></span></div>'


def _metric(label: str, value: Any, note: str = "", tone: str = "") -> str:
    cls = f"metric {tone}".strip()
    return (
        f'<div class="{cls}"><div class="m-label">{esc(label)}</div>'
        f'<div class="m-value">{esc(value)}</div>'
        f'<div class="m-note">{esc(note)}</div></div>'
    )


def _list_items(items: Iterable[Any]) -> str:
    return "".join(f"<li>{esc(item)}</li>" for item in items)


def _small_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[Tuple[str, str]], limit: int = 8) -> str:
    if not rows:
        return '<div class="empty">데이터 없음</div>'
    head = "".join(f"<th>{esc(label)}</th>" for _, label in columns)
    body = []
    for row in rows[:limit]:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)[:160]
            cells.append(f"<td>{esc(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_stock_research_html(payload: Mapping[str, Any]) -> str:
    symbol = payload.get("symbol")
    company = payload.get("company")
    conclusion = payload.get("investment_conclusion") or {}
    score = as_float(conclusion.get("score")) or 0.0
    horizon_scores = payload.get("horizon_scores") or {}
    technical = payload.get("technical") or {}
    fundamentals = payload.get("fundamentals") or {}
    ddm = payload.get("ddm_signal") or {}
    gostop_moment = payload.get("gostop_moment") or {}
    research_intelligence = payload.get("research_intelligence") or {}
    hypotheses = payload.get("hypotheses") or []
    backtest = payload.get("backtest") or {}
    simulation = payload.get("simulation") or {}
    roadmap = payload.get("roadmap") or {}
    source_status = payload.get("source_status") or {}
    unused = payload.get("unused_data_source_statement") or []
    quality_audit = payload.get("research_quality_audit") or {}

    horizon_cards = "".join(
        _metric(name, f"{value}/100", "기간 적합도", _tone_class(value))
        for name, value in horizon_scores.items()
    )
    hypothesis_cards = []
    for item in hypotheses:
        item_score = item.get("score")
        evidence = item.get("evidence") or []
        hypothesis_cards.append(
            '<section class="card">'
            f'<div class="rowHead"><h2>{esc(item.get("question"))}</h2><strong class="{_tone_class(item_score)}">{esc(item_score)}/100</strong></div>'
            + _bar(item_score)
            + f'<ul>{_list_items(evidence)}</ul>'
            + f'<p class="counter">{esc(item.get("counter"))}</p>'
            + "</section>"
        )

    support_rows = ddm.get("support") or []
    resistance_rows = ddm.get("resistance") or []
    backtest_metrics = [
        _metric("Strategy", fmt_pct(backtest.get("strategy_total_return_pct")), "walk-forward", _tone_class((as_float(backtest.get("strategy_total_return_pct")) or 0) + 50)),
        _metric("Buy&Hold", fmt_pct(backtest.get("buyhold_total_return_pct")), "same window"),
        _metric("Excess", fmt_pct(backtest.get("excess_return_pct")), "strategy - B&H", _tone_class((as_float(backtest.get("excess_return_pct")) or 0) + 55)),
        _metric("Avg Exposure", fmt_pct(backtest.get("avg_exposure_pct")), "risk used"),
    ]
    simulation_rows = simulation.get("horizons") or []
    source_rows = [
        {"source": key, "status": value}
        for key, value in source_status.items()
    ]
    ri_barchart = research_intelligence.get("barchart") or {}
    ri_sec = research_intelligence.get("sec") or {}
    ri_community = research_intelligence.get("community") or {}
    quality_rows = quality_audit.get("contradiction_ledger") or []
    quality_override = quality_audit.get("action_override") or {}
    simulation_quality = quality_audit.get("simulation_quality") or {}
    sec_quality = quality_audit.get("sec_quality") or {}
    sec_limitations = ri_sec.get("limitations") or []
    option_metrics_rows = [
        {"metric": key, "value": value}
        for key, value in (ri_barchart.get("option_metrics") or {}).items()
        if value not in (None, "")
    ]
    turning_point_rows = [
        {"level": key, "value": value}
        for key, value in (ri_barchart.get("turning_points") or {}).items()
        if value not in (None, "")
    ]
    css = """
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#edf1f3;color:#172026;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Apple SD Gothic Neo,Noto Sans KR,sans-serif;letter-spacing:0}
body{width:420px;max-width:420px;margin:0 auto}
.page{width:420px;max-width:420px;min-height:100vh;padding:14px 12px 24px;background:#f6f8f8}
.hero,.card{background:#fff;border:1px solid #d9e1e5;border-radius:8px;box-shadow:0 6px 16px rgba(28,42,50,.05)}
.hero{padding:15px 14px;background:linear-gradient(180deg,#fffefa 0,#eef5f6 100%)}
.eyebrow{font-size:10.5px;font-weight:850;color:#607077;text-transform:uppercase}
h1{margin:5px 0 6px;font-size:24px;line-height:1.13;letter-spacing:0;color:#14242c}
.meta{font-size:11px;color:#67757b;line-height:1.4}
.heroGrid{display:grid;grid-template-columns:92px 1fr;gap:12px;align-items:center;margin-top:12px}
.score{width:82px;height:82px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(#2f8f6b 0 var(--score),#dfe7ea var(--score) 100%);border:6px solid #fff;box-shadow:inset 0 0 0 1px rgba(0,0,0,.08);font-size:27px;font-weight:950}
.stance strong{display:block;font-size:16px;line-height:1.25;color:#17313b}.stance p{margin:5px 0 0;font-size:12.5px;line-height:1.45;color:#33454d}
.cards,.grid2,.grid3{display:grid;gap:8px}.grid2{grid-template-columns:1fr 1fr}.grid3{grid-template-columns:1fr 1fr}
.card{margin-top:12px;padding:12px}
h2{margin:0 0 9px;font-size:15.5px;line-height:1.25;color:#17272e}h3{margin:10px 0 6px;font-size:12.5px;color:#24353d}
p,li{font-size:12px;line-height:1.45;color:#34444b}p{margin:7px 0}ul{margin:7px 0 0;padding-left:16px}li{margin:4px 0}
.metric{min-height:72px;padding:9px;border:1px solid #dfe6e9;border-radius:7px;background:#fbfcfc}.metric.good{background:#e8f5ef;border-color:#bcdcca}.metric.warn{background:#fbf3df;border-color:#e6d19d}.metric.bad{background:#f9e9e7;border-color:#e2b6b1}
.m-label{font-size:10.5px;color:#657278;font-weight:850}.m-value{margin-top:3px;font-size:17px;line-height:1.15;font-weight:950;overflow-wrap:anywhere}.m-note{margin-top:4px;font-size:10px;line-height:1.25;color:#68757a}
.rowHead{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}.rowHead strong{font-size:15px}.good{color:#1f7a52}.warn{color:#9b6c00}.bad{color:#b23b34}.neutral{color:#5b6870}
.bar{height:9px;margin:8px 0;background:#e5ecef;border-radius:999px;overflow:hidden}.bar span{display:block;height:100%;background:linear-gradient(90deg,#bf4b4b,#d4a33a,#2f8f6b);border-radius:999px}
.counter{padding:8px;border-left:3px solid #8aa9bd;background:#f3f7f8;border-radius:6px;color:#425158}
.flag{padding:8px;border:1px solid #e2c16d;background:#fff8e6;border-radius:7px;color:#55420f;font-size:11.5px;line-height:1.42}
table{width:100%;border-collapse:collapse;table-layout:fixed;margin-top:7px}th,td{border-top:1px solid #e0e7ea;padding:7px 5px;text-align:left;vertical-align:top;font-size:11.2px;line-height:1.35;overflow-wrap:anywhere}th{font-size:10px;color:#5b6970;background:#f2f6f7}
.empty{padding:10px;border:1px dashed #ccd6da;border-radius:7px;background:#f8fafb;color:#6f7d83;font-size:12px}
.mini{font-size:11px;color:#66747a}.source{font-size:11px;line-height:1.42;color:#4a5960}
@page{size:420px 920px;margin:10px}
"""
    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=420, initial-scale=1">
<title>{esc(symbol)} Deep Research</title>
<style>:root{{--score:{clamp(score):.1f}%}}{css}</style>
</head>
<body>
<div class="page">
  <header class="hero">
    <div class="eyebrow">Single Stock Deep Research · DDM</div>
    <h1>{esc(company)} ({esc(symbol)})</h1>
    <div class="meta">생성 UTC {esc(payload.get('generated_at_utc'))} · 목표 프로세스 60분 · 투자기간 1주~1년</div>
    <div class="heroGrid">
      <div class="score">{score:.0f}</div>
      <div class="stance"><strong>{esc(conclusion.get('label'))}</strong><p>{esc(conclusion.get('stance'))}</p></div>
    </div>
  </header>

  <section class="card">
    <h2>기간별 적합도</h2>
    <div class="grid3">{horizon_cards}</div>
  </section>

  <section class="card">
    <h2>딥리서치 품질 게이트</h2>
    <div class="grid2">
      {_metric('Verdict', quality_audit.get('verdict'), 'quality audit', 'bad' if quality_audit.get('verdict') == 'requires_revision' else 'warn' if quality_audit.get('verdict') == 'pass_with_warnings' else 'good')}
      {_metric('Conflict', f"H{quality_audit.get('high_count', 0)} / M{quality_audit.get('medium_count', 0)}", 'high / medium')}
      {_metric('Action', quality_override.get('status') or 'no override', quality_override.get('label') or '')}
      {_metric('Exposure Cap', quality_override.get('suggested_exposure') or gostop_moment.get('suggested_exposure'), 'after gate')}
    </div>
    <p class="counter">{esc(quality_audit.get('decision_rule'))}</p>
    {_small_table(quality_rows, [('severity','등급'),('type','충돌'),('evidence','근거'),('required_resolution','해소 규칙'),('impact','영향')], limit=8)}
    <h3>시뮬레이션 품질</h3>
    <p class="flag">decision use: {esc(simulation_quality.get('use_for_decision'))} · {esc(simulation_quality.get('rule'))}</p>
    <ul>{_list_items(simulation_quality.get('warnings') or [])}</ul>
    <h3>SEC 파싱 품질</h3>
    <p class="flag">parse status: {esc(sec_quality.get('parse_status'))} · raw XBRL filtered: {esc(sec_quality.get('raw_xbrl_snippet_count'))}</p>
    <ul>{_list_items(sec_limitations)}</ul>
  </section>

  <section class="card">
    <h2>DDM 맥락 강화</h2>
    <div class="grid2">
      {_metric('Drift', fmt_num(ddm.get('drift'), 3), ddm.get('status'), _tone_class((as_float(ddm.get('evidence')) or 0) * 30 + 50))}
      {_metric('Diffusion', fmt_num(ddm.get('diffusion'), 3), '내부 저항/불일치')}
      {_metric('Evidence', fmt_num(ddm.get('evidence'), 3), 'drift / diffusion')}
      {_metric('Confidence', fmt_pct(ddm.get('confidence_pct')), f"{ddm.get('correlated_count', 0)} correlated")}
    </div>
    <h3>지지 압력</h3>
    {_small_table(support_rows, [('symbol','symbol'),('corr','corr'),('return_5d_pct','5D'),('signed_pressure','pressure')], limit=6)}
    <h3>저항 압력</h3>
    {_small_table(resistance_rows, [('symbol','symbol'),('corr','corr'),('return_5d_pct','5D'),('signed_pressure','pressure')], limit=6)}
  </section>

  <section class="card">
    <h2>GoStop Moment</h2>
    <div class="grid2">
      {_metric('Gate', gostop_moment.get('gate'), gostop_moment.get('status'), _tone_class(gostop_moment.get('score')))}
      {_metric('Moment Score', f"{gostop_moment.get('score')}/100", gostop_moment.get('suggested_exposure'), _tone_class(gostop_moment.get('score')))}
      {_metric('GO 거리', fmt_num(gostop_moment.get('distance_to_go')), 'GO boundary 62')}
      {_metric('STOP 거리', fmt_num(gostop_moment.get('distance_to_stop')), 'STOP boundary 45')}
    </div>
    <p class="counter">{esc(gostop_moment.get('reason'))}</p>
    <p class="mini">{esc(gostop_moment.get('formula'))}</p>
  </section>

  <section class="card">
    <h2>AI 리서치 추론</h2>
    <div class="grid2">
      {_metric('Thesis Score', f"{research_intelligence.get('thesis_score')}/100", 'evidence synthesis', _tone_class(research_intelligence.get('thesis_score')))}
      {_metric('Barchart Pages', ri_barchart.get('confirmed_pages'), '본문 분석 반영')}
      {_metric('SEC Docs', ri_sec.get('filing_documents_count'), '공식 filing 원문')}
      {_metric('Naver Cafe', f"{ri_community.get('naver_home_ok')}/{ri_community.get('naver_mobile_search_pages_ok')}/{ri_community.get('naver_search_pages_ok')}", 'home/mobile/fallback')}
    </div>
    <p class="counter">{esc(research_intelligence.get('thesis'))}</p>
    <h3>확인 근거</h3>
    <ul>{_list_items(research_intelligence.get('confirmed_evidence') or [])}</ul>
    <h3>반대 근거 / 남은 리스크</h3>
    <ul>{_list_items(research_intelligence.get('challenge_evidence') or [])}</ul>
    <h3>Barchart 옵션·레벨</h3>
    {_small_table(option_metrics_rows, [('metric','metric'),('value','value')], limit=10)}
    {_small_table(turning_point_rows, [('level','level'),('value','value')], limit=8)}
    <h3>Barchart 분석글/뉴스 신호</h3>
    <ul>{_list_items((ri_barchart.get('opinion_snippets') or [])[:4] + (ri_barchart.get('news_snippets') or [])[:4])}</ul>
    <h3>SEC 공식 filing 신호</h3>
    <p class="mini">parse status: {esc(ri_sec.get('parse_status'))} · raw XBRL filtered: {esc(ri_sec.get('raw_xbrl_snippet_count'))}</p>
    <ul>{_list_items(sec_limitations)}</ul>
    <ul>{_list_items(ri_sec.get('filing_snippets') or [])}</ul>
  </section>

  {''.join(hypothesis_cards)}

  <section class="card">
    <h2>핵심 수치</h2>
    <div class="grid2">
      {_metric('현재가', fmt_money(technical.get('latest_price')), technical.get('price_date'))}
      {_metric('기술 점수', f"{technical.get('score')}/100", f"RSI {fmt_num(technical.get('rsi14'))}", _tone_class(technical.get('score')))}
      {_metric('Quality', f"{fundamentals.get('quality_score')}/100", f"매출 YoY {fmt_pct(fundamentals.get('revenue_yoy_pct'))}", _tone_class(fundamentals.get('quality_score')))}
      {_metric('Value', f"{fundamentals.get('value_score')}/100", f"P/S {fmt_num(fundamentals.get('price_to_sales_ttm'))}", _tone_class(fundamentals.get('value_score')))}
    </div>
  </section>

  <section class="card">
    <h2>과거 검증</h2>
    <div class="grid2">{''.join(backtest_metrics)}</div>
    <h3>Forward horizon hit</h3>
    {_small_table(backtest.get('forward_horizons') or [], [('horizon','기간'),('signal_count','신호수'),('avg_forward_return_pct','평균'),('win_rate_pct','승률'),('p10_pct','p10'),('p90_pct','p90')], limit=8)}
    <p class="mini">{esc(backtest.get('note'))}</p>
  </section>

  <section class="card">
    <h2>시뮬레이션</h2>
    <p class="flag">품질 게이트 기준: {esc(simulation.get('decision_use'))}. {esc('; '.join(simulation_quality.get('warnings') or []))}</p>
    {_small_table(simulation_rows, [('horizon','기간'),('p10_return_pct','p10'),('p50_return_pct','p50'),('p90_return_pct','p90'),('prob_gain_pct','상승확률'),('prob_loss_10pct_pct','-10%확률')], limit=8)}
    <p class="mini">{esc(simulation.get('note'))}</p>
  </section>

  <section class="card">
    <h2>로드맵 / 마일스톤</h2>
    <p>{esc(roadmap.get('read'))}</p>
    {_small_table(roadmap.get('events') or [], [('date','date'),('type','type'),('title','title'),('source','source')], limit=10)}
  </section>

  <section class="card">
    <h2>데이터 소스 상태</h2>
    {_small_table(source_rows, [('source','source'),('status','status')], limit=12)}
    <h3>데이터 소스 미사용 사유서</h3>
    <ul>{_list_items(unused)}</ul>
  </section>
</div>
</body>
</html>
"""
    return html_doc


def save_stock_research_report(
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
    stamp: Optional[str] = None,
) -> JsonDict:
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol = str(payload.get("symbol") or "STOCK").upper()
    run_stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{symbol}_deep_research_{run_stamp}_raw.json"
    html_path = output_dir / f"{symbol}_deep_research_{run_stamp}_mobile420.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    html_path.write_text(render_stock_research_html(payload), encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path)}
