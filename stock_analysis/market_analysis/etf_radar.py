#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF GoStop: post-close ETF flow/NAV/price entry gate.

The module combines Massive ETF Global fund-flow data with ETF market prices
from FMP. The output is not a broad ETF discovery radar. It is a Go/Stop gate
that answers whether fresh risk entry is allowed after the close.
"""

from __future__ import annotations

import csv
import datetime as dt
import html as html_lib
import json
import math
import os
import re
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from .dgx_paths import choose_file_path, existing_nonlegacy_file

try:  # pragma: no cover - present in the repo environment.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


DEFAULT_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "TQQQ",
    "XLK",
    "XLF",
    "XLE",
    "XLY",
    "XLP",
    "XLV",
    "XLI",
    "XLC",
    "XLU",
    "XLB",
    "XLRE",
    "SMH",
    "TLT",
    "HYG",
    "LQD",
    "GLD",
    "USO",
]
DEFAULT_GOSTOP_HISTORY_DB_PATH = Path("sweet_spot_reports/etf_gostop_history.sqlite")
DEFAULT_GOSTOP_ICLOUD_HISTORY_DB_PATH = choose_file_path(
    "ETF_GOSTOP_ICLOUD_HISTORY_DB_PATH",
    "ETF GoStop",
    "data",
    "etf_gostop_history.sqlite",
    legacy_mac_path=Path(
        "/home/zooh/Documents/DGX_Outputs/STOCK/ETF GoStop/data/etf_gostop_history.sqlite"
    ),
)
KST = dt.timezone(dt.timedelta(hours=9), name="KST")

RISK_ETFS = {"SPY", "QQQ", "IWM", "DIA", "VTI", "TQQQ", "XLK", "SMH", "XLY"}
DEFENSIVE_ETFS = {"TLT", "GLD", "XLP", "XLV", "XLU", "LQD"}
SECTOR_ETFS = {"XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLC", "XLU", "XLB", "XLRE", "SMH"}
DEFAULT_TUNE_HORIZONS = [5, 10, 21]
ADAPTIVE_EXPOSURE_CANDIDATES = [0, 10, 25, 40, 60, 80, 100]
TQQQ_BOOST_CANDIDATES = [0, 5, 10, 15, 20, 25, 33]
FLOW_PERSISTENCE_LOOKBACK_DAYS = 760
BUY_HOLD_BASE_EXPOSURE_PCT = 100
DDM_CORR_LOOKBACK_BARS = 45
DDM_MIN_CORR_OVERLAP = 20
DDM_MIN_ABS_CORR = 0.25
DDM_HIGH_ABS_CORR = 0.35
DDM_HIGH_PRECISION_REDUCTION_TACTICALS = {"WATCH_REBOUND", "TACTICAL_GO_SMALL"}
DDM_TARGET_DERIVED_EXCLUSIONS = {
    # QQQ-derived leveraged/inverse products are crowding/boost context, not
    # independent correlated-basket evidence for QQQ drift.
    "QQQ": {"TQQQ", "QLD", "SQQQ", "PSQ", "QID"},
}
SIMILARITY_FEATURES = [
    "gostop_score",
    "greed_score",
    "risk_price_5d",
    "risk_flow_5d_aum",
    "broad_flow_5d_aum",
    "small_cyclical_flow_5d_aum",
    "credit_flow_5d_aum",
    "defensive_flow_5d_aum",
    "leverage_flow_5d_aum",
    "sector_flow_5d_total",
    "total_flow_5d",
    "nowcast_score",
    "nowcast_risk_day",
    "nowcast_risk_breadth",
    "nowcast_rel_volume",
    "distribution_count",
    "fragile_rally_count",
    "quiet_redemption_count",
    "ddm_drift",
    "ddm_diffusion",
    "ddm_evidence",
    "ddm_confidence",
    "ddm_agreement",
    "ddm_correlated_count",
    "ddm_support_pressure",
    "ddm_resistance_pressure",
]


def _now_utc_and_kst() -> Tuple[dt.datetime, dt.datetime]:
    """Return one coherent timestamp pair for report issue-time fields."""

    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    return now_utc, now_utc.astimezone(KST)


def _date_counts(values: Iterable[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))

FMP_BASE_URL = "https://financialmodelingprep.com"
MASSIVE_BASE_URL = "https://api.massive.com"


class EtfRadarError(RuntimeError):
    """Raised when the report cannot be built at all."""


@dataclass
class PricePoint:
    date: str
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None


@dataclass
class FlowPoint:
    composite_ticker: str
    effective_date: str
    processed_date: str
    fund_flow: Optional[float]
    nav: Optional[float]
    shares_outstanding: Optional[float]


@dataclass
class RadarRow:
    symbol: str
    price_date: Optional[str]
    latest_price: Optional[float]
    price_1d_pct: Optional[float]
    price_5d_pct: Optional[float]
    price_20d_pct: Optional[float]
    flow_date: Optional[str]
    processed_date: Optional[str]
    fund_flow_latest: Optional[float]
    fund_flow_5d: Optional[float]
    fund_flow_20d: Optional[float]
    flow_zscore: Optional[float]
    flow_aum_latest_pct: Optional[float]
    flow_aum_5d_pct: Optional[float]
    nav: Optional[float]
    nav_gap_pct: Optional[float]
    nav_stale_days: Optional[int]
    shares_change_5d_pct: Optional[float]
    signal: str
    nuance: str
    warnings: List[str]


@dataclass
class GoStopDecision:
    action: str
    score: int
    mode: str
    max_new_risk_pct: int
    headline: str
    reasons: List[str]
    blocks: List[str]
    entry_candidates: List[str]
    stop_candidates: List[str]


@dataclass
class NowCastRow:
    symbol: str
    price: Optional[float]
    day_change_pct: Optional[float]
    rel_volume: Optional[float]
    above_sma20_pct: Optional[float]
    above_sma50_pct: Optional[float]
    above_sma200_pct: Optional[float]
    signal: str


def _load_env() -> None:
    if load_dotenv is None:
        return
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    load_dotenv(override=False)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except Exception:
        return None


def _parse_date(value: Any) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _fmt_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return ("%+." + str(digits) + "f%%") % value


def _fmt_num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:+.{digits}f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:+.{digits}f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:+.{digits}f}K"
    return f"{value:+.{digits}f}"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return statistics.median(clean)


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old in (None, 0):
        return None
    return (new / old - 1.0) * 100.0


def _sum(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean)


def _first_nonempty(*values: Any) -> Optional[str]:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return None


class MarketDataClient:
    """Small HTTP client for FMP and Massive read-only market data."""

    def __init__(
        self,
        *,
        fmp_key: Optional[str] = None,
        massive_key: Optional[str] = None,
        timeout: int = 20,
        pause_sec: float = 0.15,
    ) -> None:
        _load_env()
        self.fmp_key = fmp_key or os.getenv("FMP_API_KEY", "").strip()
        self.massive_key = massive_key or os.getenv("MASSIVE_API_KEY", "").strip()
        self.timeout = timeout
        self.pause_sec = pause_sec
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "STOCK-etf-radar/1.0"})

    def _get_json(self, url: str, params: Mapping[str, Any]) -> Any:
        time.sleep(self.pause_sec)
        response = self.session.get(url, params=dict(params), timeout=self.timeout)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"text": response.text[:2000]}

    def fmp_quote(self, symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not self.fmp_key:
            # Try local fundamental data as fallback
            for p in [
                Path(__file__).resolve().parents[2] / "STOCK" / "sweet_spot_reports" / "etf_fundamental_data.json",
                Path(__file__).resolve().parents[1] / "STOCK" / "sweet_spot_reports" / "etf_fundamental_data.json",
            ]:
                if p.exists():
                    try:
                        data = json.load(open(p))
                        price_data = data.get("price_data", {})
                        return {
                            s: {
                                "symbol": s,
                                "price": p_data.get("latest_price"),
                                "priceAvg50": p_data.get("fiftyDayAverage"),
                                "priceAvg200": p_data.get("twoHundredDayAverage"),
                                "fiftyTwoWeekHigh": p_data.get("fiftyTwoWeekHigh"),
                                "fiftyTwoWeekLow": p_data.get("fiftyTwoWeekLow"),
                                "trailingPE": p_data.get("trailingPE"),
                                "dividendYield": p_data.get("dividendYield"),
                            }
                            for s, p_data in price_data.items()
                        }
                    except Exception:
                        pass
            return {}
        clean = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not clean:
            return {}
        url = f"{FMP_BASE_URL}/api/v3/quote/{','.join(clean)}"
        data = self._get_json(url, {"apikey": self.fmp_key})
        rows = data if isinstance(data, list) else []
        return {str(row.get("symbol", "")).upper(): row for row in rows if isinstance(row, dict)}

    def fmp_historical_prices(
        self, symbol: str, *, from_date: str, to_date: str
    ) -> List[PricePoint]:
        if not self.fmp_key:
            # Try local price history as fallback
            for p in [
                Path(__file__).resolve().parents[2] / "STOCK" / "sweet_spot_reports" / "etf_price_history.json",
                Path(__file__).resolve().parents[1] / "STOCK" / "sweet_spot_reports" / "etf_price_history.json",
            ]:
                if p.exists():
                    try:
                        data = json.load(open(p))
                        bars = data.get(symbol.upper(), {}).get("bars", [])
                        return [
                            PricePoint(
                                date=b["date"][:10],
                                close=_to_float(b["close"]),
                                open=_to_float(b["open"]),
                                high=_to_float(b["high"]),
                                low=_to_float(b["low"]),
                                volume=_to_float(b["volume"]),
                            )
                            for b in bars
                            if _to_float(b["close"]) is not None
                        ]
                    except Exception:
                        pass
            return []
        url = f"{FMP_BASE_URL}/api/v3/historical-price-full/{symbol.upper()}"
        data = self._get_json(
            url,
            {"apikey": self.fmp_key, "from": from_date, "to": to_date},
        )
        rows = data.get("historical") if isinstance(data, dict) else []
        points: List[PricePoint] = []
        if not isinstance(rows, list):
            return points
        for row in rows:
            if not isinstance(row, dict):
                continue
            date = _first_nonempty(row.get("date"))
            close = _to_float(row.get("close"))
            if not date or close is None:
                continue
            points.append(
                PricePoint(
                    date=date[:10],
                    close=close,
                    open=_to_float(row.get("open")),
                    high=_to_float(row.get("high")),
                    low=_to_float(row.get("low")),
                    volume=_to_float(row.get("volume")),
                )
            )
        return sorted(points, key=lambda item: item.date)

    def massive_fund_flows(
        self,
        symbols: Sequence[str],
        *,
        processed_date_gte: Optional[str] = None,
        limit: int = 5000,
    ) -> List[FlowPoint]:
        if not self.massive_key:
            return []
        clean = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not clean:
            return []
        params: Dict[str, Any] = {
            "apiKey": self.massive_key,
            "composite_ticker.any_of": ",".join(clean),
            "limit": int(limit),
            "sort": "processed_date.desc",
        }
        if processed_date_gte:
            params["processed_date.gte"] = processed_date_gte
        url = f"{MASSIVE_BASE_URL}/etf-global/v1/fund-flows"
        try:
            data = self._get_json(url, params)
            rows = data.get("results") if isinstance(data, dict) else []
            if isinstance(rows, list) and rows:
                return self._parse_flow_rows(rows)
        except Exception:
            # Some entitlement or filter combinations can fail; per-ticker
            # fallback keeps the report useful and localizes data gaps.
            pass

        all_rows: List[Dict[str, Any]] = []
        for symbol in clean:
            ticker_params = dict(params)
            ticker_params.pop("composite_ticker.any_of", None)
            ticker_params["composite_ticker"] = symbol
            try:
                data = self._get_json(url, ticker_params)
            except Exception:
                continue
            rows = data.get("results") if isinstance(data, dict) else []
            if isinstance(rows, list):
                all_rows.extend(row for row in rows if isinstance(row, dict))
        return self._parse_flow_rows(all_rows)

    @staticmethod
    def _parse_flow_rows(rows: Sequence[Mapping[str, Any]]) -> List[FlowPoint]:
        points: List[FlowPoint] = []
        for row in rows:
            ticker = _first_nonempty(row.get("composite_ticker"))
            effective_date = _first_nonempty(row.get("effective_date"))
            processed_date = _first_nonempty(row.get("processed_date"))
            if not ticker or not (effective_date or processed_date):
                continue
            points.append(
                FlowPoint(
                    composite_ticker=ticker.upper(),
                    effective_date=(effective_date or processed_date or "")[:10],
                    processed_date=(processed_date or effective_date or "")[:10],
                    fund_flow=_to_float(row.get("fund_flow")),
                    nav=_to_float(row.get("nav")),
                    shares_outstanding=_to_float(row.get("shares_outstanding")),
                )
            )
        return sorted(
            points,
            key=lambda item: (
                item.composite_ticker,
                item.effective_date or "",
                item.processed_date or "",
            ),
        )


def _latest_price_on_or_before(
    prices: Sequence[PricePoint], target_date: Optional[str]
) -> Optional[PricePoint]:
    if not prices:
        return None
    if not target_date:
        return prices[-1]
    target = _parse_date(target_date)
    if not target:
        return prices[-1]
    candidates = [point for point in prices if (_parse_date(point.date) or dt.date.min) <= target]
    return candidates[-1] if candidates else None


def _last_change(prices: Sequence[PricePoint], bars: int) -> Optional[float]:
    if len(prices) <= bars:
        return None
    return _pct_change(prices[-1].close, prices[-1 - bars].close)


def _last_n_flows(flows: Sequence[FlowPoint], n: int) -> List[FlowPoint]:
    return list(flows[-n:]) if flows else []


def _flow_zscore(flows: Sequence[FlowPoint], window: int = 20) -> Optional[float]:
    values = [point.fund_flow for point in flows if point.fund_flow is not None]
    if len(values) < 6:
        return None
    latest = values[-1]
    history = values[-window - 1 : -1] if len(values) > window else values[:-1]
    if len(history) < 5:
        return None
    try:
        std = statistics.pstdev(history)
    except statistics.StatisticsError:
        return None
    if std <= 0:
        return None
    return (latest - statistics.mean(history)) / std


def _aum_from_flow(point: Optional[FlowPoint]) -> Optional[float]:
    if point is None or point.nav is None or point.shares_outstanding is None:
        return None
    aum = point.nav * point.shares_outstanding
    return aum if aum > 0 else None


def _flow_aum_pct(flow: Optional[float], aum: Optional[float]) -> Optional[float]:
    if flow is None or not aum:
        return None
    return flow / aum * 100.0


def _shares_change(flows: Sequence[FlowPoint], bars: int) -> Optional[float]:
    clean = [point for point in flows if point.shares_outstanding is not None]
    if len(clean) <= bars:
        return None
    return _pct_change(clean[-1].shares_outstanding, clean[-1 - bars].shares_outstanding)


def classify_etf_row(row: RadarRow) -> Tuple[str, str]:
    """Classify an ETF from cross-signals, preserving subtle conflicts."""

    price5 = row.price_5d_pct
    flow5 = row.fund_flow_5d
    flow5_aum = row.flow_aum_5d_pct
    z = row.flow_zscore
    nav_gap = row.nav_gap_pct
    shares5 = row.shares_change_5d_pct

    price_up = price5 is not None and price5 >= 1.0
    price_down = price5 is not None and price5 <= -1.0
    flow_up = (flow5 is not None and flow5 > 0) or (flow5_aum is not None and flow5_aum >= 0.10)
    flow_down = (flow5 is not None and flow5 < 0) or (flow5_aum is not None and flow5_aum <= -0.10)
    flow_hot = (z is not None and z >= 1.25) or (flow5_aum is not None and flow5_aum >= 0.35)
    flow_cold = (z is not None and z <= -1.25) or (flow5_aum is not None and flow5_aum <= -0.35)
    premium = nav_gap is not None and nav_gap >= 0.20
    discount = nav_gap is not None and nav_gap <= -0.20
    shares_expand = shares5 is not None and shares5 >= 0.10
    shares_shrink = shares5 is not None and shares5 <= -0.10

    if price_up and flow_hot and premium:
        return (
            "greed_overheat",
            "가격 상승, 강한 유입, NAV 프리미엄이 동시에 나타나 단기 과열 수급입니다.",
        )
    if price_up and flow_up and not premium:
        if shares_expand:
            return (
                "healthy_risk_on",
                "가격 상승과 유입, 발행주식수 증가가 동행해 수급이 가격을 확인합니다.",
            )
        return (
            "constructive_inflow",
            "가격 상승과 유입이 동행하지만 creation 확인은 약해 건강한 추세의 초입에 가깝습니다.",
        )
    if price_up and (flow_down or shares_shrink):
        return (
            "fragile_rally",
            "가격은 오르지만 flow나 shares가 빠져 랠리의 질이 약합니다.",
        )
    if price_down and flow_up:
        return (
            "dip_buying",
            "가격 약세에도 자금이 들어와 저가매수 또는 방어적 매집 신호입니다.",
        )
    if price_down and flow_down and discount:
        return (
            "stress_outflow",
            "가격 하락, 유출, NAV 디스카운트가 겹쳐 유동성 스트레스에 가깝습니다.",
        )
    if price_down and flow_down:
        return (
            "distribution",
            "가격 약세와 유출이 같이 나타나 분배/리스크 축소 흐름입니다.",
        )
    if not price_up and not price_down and flow_hot:
        return (
            "quiet_accumulation",
            "가격은 조용하지만 평소보다 강한 유입이 있어 선행 매집 가능성이 있습니다.",
        )
    if not price_up and not price_down and flow_cold:
        return (
            "quiet_redemption",
            "가격은 버티지만 유출이 커 내부 수급은 약해지는 중입니다.",
        )
    if premium:
        return (
            "premium_watch",
            "가격 방향은 중립적이나 NAV 프리미엄이 있어 단기 수요 과열을 감시해야 합니다.",
        )
    if discount:
        return (
            "discount_watch",
            "가격 방향은 중립적이나 NAV 디스카운트가 있어 매도 압력 또는 가격 괴리를 감시해야 합니다.",
        )
    return (
        "mixed_neutral",
        "가격, flow, NAV가 한 방향으로 정렬되지 않아 전환 구간으로 봅니다.",
    )


def build_radar_rows(
    *,
    symbols: Sequence[str],
    price_map: Mapping[str, Sequence[PricePoint]],
    flow_map: Mapping[str, Sequence[FlowPoint]],
    asof_date: Optional[str] = None,
) -> List[RadarRow]:
    rows: List[RadarRow] = []
    asof = _parse_date(asof_date) or dt.date.today()
    for symbol in symbols:
        sym = symbol.upper()
        prices = list(price_map.get(sym) or [])
        flows = list(flow_map.get(sym) or [])
        latest_price = prices[-1] if prices else None
        latest_flow = flows[-1] if flows else None
        nav_point = latest_flow
        nav_price_point = _latest_price_on_or_before(prices, latest_flow.effective_date if latest_flow else None)
        aum = _aum_from_flow(latest_flow)
        flow_5 = _sum(point.fund_flow for point in _last_n_flows(flows, 5))
        flow_20 = _sum(point.fund_flow for point in _last_n_flows(flows, 20))
        nav_gap = _pct_change(nav_price_point.close if nav_price_point else None, nav_point.nav if nav_point else None)
        stale_days: Optional[int] = None
        warnings: List[str] = []
        if latest_flow:
            flow_date = _parse_date(latest_flow.effective_date)
            if flow_date:
                stale_days = (asof - flow_date).days
                if stale_days > 3:
                    warnings.append(f"flow/NAV 기준일이 {stale_days}일 전입니다.")
        else:
            warnings.append("Massive fund flow 데이터가 없습니다.")
        if not latest_price:
            warnings.append("가격 데이터가 없습니다.")
        if nav_gap is None:
            warnings.append("NAV 괴리를 계산할 수 없습니다.")

        row = RadarRow(
            symbol=sym,
            price_date=latest_price.date if latest_price else None,
            latest_price=latest_price.close if latest_price else None,
            price_1d_pct=_last_change(prices, 1),
            price_5d_pct=_last_change(prices, 5),
            price_20d_pct=_last_change(prices, 20),
            flow_date=latest_flow.effective_date if latest_flow else None,
            processed_date=latest_flow.processed_date if latest_flow else None,
            fund_flow_latest=latest_flow.fund_flow if latest_flow else None,
            fund_flow_5d=flow_5,
            fund_flow_20d=flow_20,
            flow_zscore=_flow_zscore(flows),
            flow_aum_latest_pct=_flow_aum_pct(latest_flow.fund_flow if latest_flow else None, aum),
            flow_aum_5d_pct=_flow_aum_pct(flow_5, aum),
            nav=latest_flow.nav if latest_flow else None,
            nav_gap_pct=nav_gap,
            nav_stale_days=stale_days,
            shares_change_5d_pct=_shares_change(flows, 5),
            signal="pending",
            nuance="",
            warnings=warnings,
        )
        signal, nuance = classify_etf_row(row)
        row.signal = signal
        row.nuance = nuance
        rows.append(row)
    return rows


def build_data_freshness_audit(rows: Sequence[RadarRow]) -> Dict[str, Any]:
    """Summarize price vs flow/NAV timing without changing decision fields."""

    total = len(rows)
    price_dates = _date_counts(row.price_date for row in rows)
    flow_dates = _date_counts(row.flow_date for row in rows)
    processed_dates = _date_counts(row.processed_date for row in rows)
    price_asof = max(price_dates) if price_dates else None
    flow_asof = max(flow_dates) if flow_dates else None
    processed_asof = max(processed_dates) if processed_dates else None
    stale_symbols = [
        row.symbol
        for row in rows
        if row.nav_stale_days is not None and row.nav_stale_days > 3
    ]
    missing_price_symbols = [
        row.symbol for row in rows if not row.price_date or row.latest_price is None
    ]
    missing_flow_symbols = [row.symbol for row in rows if not row.flow_date]
    max_nav_stale_days = max(
        [row.nav_stale_days for row in rows if row.nav_stale_days is not None],
        default=None,
    )
    flow_lag_days: Optional[int] = None
    price_date = _parse_date(price_asof)
    flow_date = _parse_date(flow_asof)
    if price_date and flow_date:
        flow_lag_days = (price_date - flow_date).days

    if missing_price_symbols:
        status = "price_missing"
    elif missing_flow_symbols:
        status = "flow_limited"
    elif stale_symbols:
        status = "stale"
    elif flow_lag_days and flow_lag_days > 0:
        status = f"price_fresh_flow_t_minus_{flow_lag_days}"
    else:
        status = "fresh"

    if status == "fresh":
        interpretation = "가격과 Flow/NAV 기준일이 맞아 수급 해석 신뢰도가 정상입니다."
    elif status.startswith("price_fresh_flow_t_minus_"):
        interpretation = (
            f"가격은 {price_asof} 기준으로 최신이나 Flow/NAV는 {flow_asof} 기준 "
            f"T-{flow_lag_days}입니다. DDM은 사용할 수 있지만 당일 수급 확정이 아니라 "
            "가격/nowcast + 지연 flow 혼합 신호로 해석해야 합니다."
        )
    elif status == "stale":
        interpretation = (
            "Flow/NAV가 3일 초과 지연된 종목이 있어 수급 기반 결론 신뢰도를 낮춰야 합니다."
        )
    else:
        interpretation = "가격 또는 Flow/NAV 일부가 비어 있어 수급 기반 결론은 부분 제한입니다."

    futures_options_note = (
        "Flow/NAV가 당일 비어 있거나 T-1이면 NQ/ES/RTY/VIX 선물과 QQQ 옵션 flow·gamma·put/call은 "
        "수급 공백을 메우는 참고축으로만 보고, 원천 확인 전 GoStop 점수에는 직접 반영하지 않습니다."
    )

    return {
        "status": status,
        "price_asof_date": price_asof,
        "flow_nav_asof_date": flow_asof,
        "flow_processed_asof_date": processed_asof,
        "flow_lag_calendar_days": flow_lag_days,
        "max_nav_stale_days": max_nav_stale_days,
        "price_date_counts": price_dates,
        "flow_date_counts": flow_dates,
        "processed_date_counts": processed_dates,
        "price_symbols_current": price_dates.get(price_asof or "", 0),
        "flow_symbols_current": flow_dates.get(flow_asof or "", 0),
        "total_symbols": total,
        "missing_price_symbols": missing_price_symbols,
        "missing_flow_symbols": missing_flow_symbols,
        "stale_flow_symbols": stale_symbols,
        "interpretation": interpretation,
        "futures_options_reference_line": futures_options_note,
    }


def build_ddm_input_audit(
    ddm_signal: Mapping[str, Any],
    freshness: Mapping[str, Any],
) -> Dict[str, Any]:
    components = list(ddm_signal.get("all_components") or ddm_signal.get("components") or [])
    required = [
        "corr",
        "price_5d_pct",
        "flow_aum_5d_pct",
        "day_change_pct",
        "rel_volume",
        "impulse",
        "signed_pressure",
    ]
    missing_fields = [
        {"symbol": item.get("symbol"), "missing_fields": [key for key in required if item.get(key) is None]}
        for item in components
        if any(item.get(key) is None for key in required)
    ]
    freshness_status = str(freshness.get("status") or "unknown")
    if not ddm_signal.get("enabled"):
        quality = "disabled"
    elif missing_fields:
        quality = "partial_component_fields"
    elif freshness_status == "fresh":
        quality = "fresh"
    elif freshness_status.startswith("price_fresh_flow_t_minus_"):
        quality = "usable_with_flow_lag"
    elif freshness_status == "stale":
        quality = "stale_limited"
    else:
        quality = "limited"
    return {
        "quality": quality,
        "ddm_asof_date": ddm_signal.get("asof_date"),
        "price_asof_date": freshness.get("price_asof_date"),
        "flow_nav_asof_date": freshness.get("flow_nav_asof_date"),
        "flow_lag_calendar_days": freshness.get("flow_lag_calendar_days"),
        "component_count": ddm_signal.get("correlated_count"),
        "stored_component_count": len(components),
        "display_component_count": len(ddm_signal.get("components") or []),
        "components_truncated_for_display": len(components) > len(ddm_signal.get("components") or []),
        "missing_component_fields": missing_fields,
        "interpretation": freshness.get("interpretation"),
    }


def _rolling_flow_aum_snapshot(flows: Sequence[FlowPoint]) -> Dict[str, Any]:
    clean = sorted(
        [point for point in flows if point.fund_flow is not None],
        key=lambda item: (item.effective_date, item.processed_date),
    )
    if not clean:
        return {
            "available": False,
            "interpretation": "원천 flow 시계열이 없어 지속성 판단을 할 수 없습니다.",
        }
    latest = clean[-1]

    def window_sum(n: int) -> Optional[float]:
        points = clean[-n:]
        return _sum(point.fund_flow for point in points)

    def window_pct_at(index: int, n: int) -> Optional[float]:
        if index + 1 < n:
            return None
        point = clean[index]
        flow_sum = _sum(item.fund_flow for item in clean[index + 1 - n : index + 1])
        return _flow_aum_pct(flow_sum, _aum_from_flow(point))

    def percentile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
        clean_values = sorted(float(value) for value in values if value is not None)
        if not clean_values:
            return None
        if len(clean_values) == 1:
            return clean_values[0]
        pos = (len(clean_values) - 1) * q
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return clean_values[low]
        return clean_values[low] * (high - pos) + clean_values[high] * (pos - low)

    def percentile_rank(values: Sequence[Optional[float]], current: Optional[float]) -> Optional[float]:
        clean_values = sorted(float(value) for value in values if value is not None)
        if current is None or not clean_values:
            return None
        return sum(1 for value in clean_values if value <= current) / len(clean_values) * 100.0

    rolling_5d = [
        {
            "date": clean[index].effective_date,
            "flow_5d_aum_pct": window_pct_at(index, 5),
            "flow_20d_aum_pct": window_pct_at(index, 20),
            "flow_60d_aum_pct": window_pct_at(index, 60),
        }
        for index in range(len(clean))
        if window_pct_at(index, 5) is not None
    ]
    consecutive_positive_5d = 0
    for item in reversed(rolling_5d):
        value = _to_float(item.get("flow_5d_aum_pct"))
        if value is not None and value > 0:
            consecutive_positive_5d += 1
        else:
            break

    latest_aum = _aum_from_flow(latest)
    latest_pct = _flow_aum_pct(latest.fund_flow, latest_aum)
    flow_5d = window_sum(5)
    flow_20d = window_sum(20)
    flow_60d = window_sum(60)
    current_5d_pct = _flow_aum_pct(flow_5d, latest_aum)
    current_20d_pct = _flow_aum_pct(flow_20d, latest_aum)
    current_60d_pct = _flow_aum_pct(flow_60d, latest_aum)
    latest_date = _parse_date(latest.effective_date)
    pre90_rows = [
        item
        for item in rolling_5d
        if latest_date
        and (_parse_date(item.get("date")) or latest_date) < latest_date - dt.timedelta(days=90)
    ]

    def pre90_stats(key: str, current_value: Optional[float]) -> Dict[str, Any]:
        values = [item.get(key) for item in pre90_rows]
        clean_values = [float(value) for value in values if value is not None]
        return {
            "n": len(clean_values),
            "mean": _mean(clean_values),
            "median": _median(clean_values),
            "p90": percentile(clean_values, 0.90),
            "p95": percentile(clean_values, 0.95),
            "current_percentile_rank": percentile_rank(clean_values, current_value),
        }

    return {
        "available": True,
        "latest_effective_date": latest.effective_date,
        "latest_processed_date": latest.processed_date,
        "latest_flow": latest.fund_flow,
        "latest_flow_aum_pct": latest_pct,
        "flow_5d": flow_5d,
        "flow_5d_aum_pct": current_5d_pct,
        "flow_20d": flow_20d,
        "flow_20d_aum_pct": current_20d_pct,
        "flow_60d": flow_60d,
        "flow_60d_aum_pct": current_60d_pct,
        "consecutive_positive_5d_windows": consecutive_positive_5d,
        "pre90_distribution": {
            "flow_5d_aum_pct": pre90_stats("flow_5d_aum_pct", current_5d_pct),
            "flow_20d_aum_pct": pre90_stats("flow_20d_aum_pct", current_20d_pct),
            "flow_60d_aum_pct": pre90_stats("flow_60d_aum_pct", current_60d_pct),
        },
        "recent_5d_aum_pct": rolling_5d[-10:],
    }


def build_flow_persistence_audit(
    *,
    rows: Sequence[RadarRow],
    flow_map: Optional[Mapping[str, Sequence[FlowPoint]]] = None,
    freshness: Optional[Mapping[str, Any]] = None,
    market_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Explain whether ETF flow strength is persistent, current, or stale."""

    row_by_symbol = {row.symbol.upper(): row for row in rows}
    flow_map = flow_map or {}
    freshness = freshness or {}
    market_state = market_state or {}
    qqq_row = row_by_symbol.get("QQQ")
    qqq_snapshot = _rolling_flow_aum_snapshot(flow_map.get("QQQ") or [])
    if not qqq_snapshot.get("available") and qqq_row:
        qqq_snapshot = {
            "available": False,
            "latest_effective_date": qqq_row.flow_date,
            "latest_processed_date": qqq_row.processed_date,
            "latest_flow": qqq_row.fund_flow_latest,
            "latest_flow_aum_pct": qqq_row.flow_aum_latest_pct,
            "flow_5d": qqq_row.fund_flow_5d,
            "flow_5d_aum_pct": qqq_row.flow_aum_5d_pct,
            "flow_20d": qqq_row.fund_flow_20d,
            "flow_20d_aum_pct": None,
            "flow_60d": None,
            "flow_60d_aum_pct": None,
            "consecutive_positive_5d_windows": None,
            "recent_5d_aum_pct": [],
        }

    risk_rows = [row for row in rows if row.symbol in RISK_ETFS]
    top_risk = sorted(
        [
            {
                "symbol": row.symbol,
                "flow_5d_aum_pct": row.flow_aum_5d_pct,
                "latest_flow_aum_pct": row.flow_aum_latest_pct,
                "price_5d_pct": row.price_5d_pct,
            }
            for row in risk_rows
            if row.flow_aum_5d_pct is not None
        ],
        key=lambda item: float(item.get("flow_5d_aum_pct") or 0.0),
        reverse=True,
    )[:5]
    weak_risk = sorted(
        [
            {
                "symbol": row.symbol,
                "flow_5d_aum_pct": row.flow_aum_5d_pct,
                "latest_flow_aum_pct": row.flow_aum_latest_pct,
                "price_5d_pct": row.price_5d_pct,
            }
            for row in risk_rows
            if row.flow_aum_5d_pct is not None
        ],
        key=lambda item: float(item.get("flow_5d_aum_pct") or 0.0),
    )[:5]

    flow_lag = _to_float(freshness.get("flow_lag_calendar_days"))
    stale_symbols = list(freshness.get("stale_flow_symbols") or [])
    if stale_symbols:
        stale_read = "일부 flow/NAV가 3일 초과 지연되어 stale 제한이 있습니다."
        freshness_status = "stale_limited"
    elif flow_lag is not None and flow_lag > 0:
        stale_read = (
            f"가격보다 flow/NAV가 T-{int(flow_lag)}입니다. stale은 아니지만 당일 수급 확정으로 보지 않습니다."
        )
        freshness_status = "lagged_usable"
    else:
        stale_read = "가격과 flow/NAV 기준일이 맞아 stale 징후는 없습니다."
        freshness_status = "fresh"

    q_latest_pct = _to_float(qqq_snapshot.get("latest_flow_aum_pct"))
    q_5d_pct = _to_float(qqq_snapshot.get("flow_5d_aum_pct"))
    q_20d_pct = _to_float(qqq_snapshot.get("flow_20d_aum_pct"))
    q_60d_pct = _to_float(qqq_snapshot.get("flow_60d_aum_pct"))
    pre90 = qqq_snapshot.get("pre90_distribution") or {}
    q_5d_rank = _to_float((pre90.get("flow_5d_aum_pct") or {}).get("current_percentile_rank"))
    q_20d_rank = _to_float((pre90.get("flow_20d_aum_pct") or {}).get("current_percentile_rank"))
    q_60d_rank = _to_float((pre90.get("flow_60d_aum_pct") or {}).get("current_percentile_rank"))
    if q_latest_pct is not None and q_latest_pct < -0.25 and q_5d_pct is not None and q_5d_pct < 0.10:
        qqq_read = (
            "QQQ 단독 flow는 현재 강하다고 보기 어렵습니다. 20D/60D 누적은 플러스여도 "
            "최근 1일 유출과 5D 둔화가 같이 보입니다."
        )
    elif q_5d_rank is not None and q_5d_rank < 60 and q_20d_rank is not None and q_20d_rank < 85:
        qqq_read = (
            "QQQ 단독 flow는 90일 이전 장기 분포 대비 평범한 구간입니다. "
            "장기 누적 플러스만으로 비정상적 유입이라고 보기는 어렵습니다."
        )
    elif q_20d_pct is not None and q_20d_pct > 0 and q_60d_pct is not None and q_60d_pct > 0:
        qqq_read = (
            "QQQ 단독 flow는 장기 누적 유입이 남아 있습니다. 다만 최신/5D 값과 함께 봐야 합니다."
        )
    elif q_5d_pct is not None and q_5d_pct > 0:
        qqq_read = "QQQ 단독 flow는 단기 순유입이나, 장기 지속성은 제한적으로 봅니다."
    else:
        qqq_read = "QQQ 단독 flow는 강한 지속 유입으로 확인되지 않습니다."

    top_symbols = ", ".join(
        f"{item['symbol']} {_fmt_pct(_to_float(item.get('flow_5d_aum_pct')))}"
        for item in top_risk[:4]
    )
    market_read = (
        f"RADAR의 risk-flow 평균은 QQQ 하나가 아니라 {', '.join(sorted(RISK_ETFS))} 평균입니다. "
        f"이번 강도는 주로 {top_symbols or 'N/A'}가 끌어올렸습니다."
    )
    interpretation = f"{qqq_read} {market_read} {stale_read}"
    return {
        "status": freshness_status,
        "source": "Massive ETF Global fund-flows + FMP price",
        "qqq": qqq_snapshot,
        "qqq_pre90_percentile_rank": {
            "flow_5d_aum_pct": q_5d_rank,
            "flow_20d_aum_pct": q_20d_rank,
            "flow_60d_aum_pct": q_60d_rank,
        },
        "risk_flow_5d_aum_avg_pct": market_state.get("risk_flow_5d_aum_avg_pct"),
        "risk_flow_5d_total": market_state.get("risk_flow_5d_total"),
        "broad_flow_5d_aum_pct": market_state.get("broad_flow_5d_aum_avg_pct"),
        "leverage_flow_5d_aum_pct": market_state.get("leverage_flow_5d_aum_pct"),
        "top_risk_contributors": top_risk,
        "weak_risk_contributors": weak_risk,
        "qqq_read": qqq_read,
        "market_read": market_read,
        "stale_read": stale_read,
        "interpretation": interpretation,
    }


def _first_metric_window(
    evidence: Mapping[str, Any],
    term: str,
    *,
    label: Optional[str] = None,
) -> str:
    term_lower = term.lower()
    label_lower = label.lower() if label else None
    for page in evidence.get("metric_windows") or []:
        if label_lower and label_lower not in str(page.get("label") or "").lower():
            continue
        for item in page.get("metric_windows") or []:
            if term_lower in str(item.get("term") or "").lower():
                return str(item.get("window") or "")
    return ""


def _metric_after_label(text: str, label: str) -> Optional[str]:
    pattern = re.escape(label) + r"\s*:?\s*(?:\|\s*)?([-+]?\d[\d,]*(?:\.\d+)?%?)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).replace(",", "")


def _metric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    return _to_float(str(value).replace("%", "").replace(",", "").strip())


def _metric_from_page_phrases(
    pages: Sequence[Mapping[str, Any]],
    *,
    label: str,
    pattern: str,
) -> Optional[str]:
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    label_lower = label.lower()
    for page in pages:
        if label_lower not in str(page.get("label") or "").lower():
            continue
        for phrase in page.get("core_phrases") or []:
            match = compiled.search(str(phrase))
            if match:
                return match.group(1).replace(",", "")
    return None


def _distance_pct(level: Any, price: Any) -> Optional[float]:
    level_value = _metric_value(level)
    price_value = _metric_value(price)
    if level_value is None or price_value in (None, 0):
        return None
    return (level_value / price_value - 1.0) * 100.0


def build_options_futures_reference(
    evidence: Optional[Mapping[str, Any]],
    *,
    evidence_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize Barchart options/futures evidence as a non-scoring reference."""

    if not evidence:
        return {
            "enabled": False,
            "source": "Barchart Premier",
            "status": "not_collected",
            "role": "reference_only_not_scoring",
            "evidence_path": evidence_path,
            "interpretation": (
                "Barchart 옵션/선물 참고 데이터가 붙지 않았습니다. GoStop 점수에는 "
                "FMP 가격과 Massive flow/NAV만 반영했습니다."
            ),
            "limitations": ["Barchart reference evidence not provided."],
        }

    pages = [page for page in evidence.get("pages") or [] if isinstance(page, Mapping)]
    qqq_pages = [
        page
        for page in pages
        if "/etfs-funds/quotes/qqq/" in str(
            page.get("final_url") or page.get("requested_url") or ""
        ).lower()
    ]
    qqq_confirmed = [page for page in qqq_pages if page.get("status") == "ok"]
    futures_pages = [
        page
        for page in pages
        if "futures" in str(page.get("final_url") or page.get("requested_url") or "").lower()
        and page.get("status") == "ok"
    ]
    limitations = list(evidence.get("limitations") or [])

    options_data_iv = _first_metric_window(evidence, "Implied Volatility", label="Options Data")
    options_prices_iv = _first_metric_window(evidence, "Implied Volatility", label="Options Prices")
    gamma_window = _first_metric_window(evidence, "IV Rank", label="Gamma Exposure")
    option_prices_volume = _first_metric_window(evidence, "Volume Ratio", label="Options Prices")
    option_prices_oi = _first_metric_window(evidence, "Open Interest Ratio", label="Options Prices")

    metrics = {
        "latest_underlying_price": _metric_from_page_phrases(
            pages,
            label="Overview",
            pattern=r"Last Price\s+([-+]?\d[\d,]*(?:\.\d+)?)",
        ),
        "iv_rank_pct": _metric_after_label(_first_metric_window(evidence, "IV Rank"), "IV Rank"),
        "iv_percentile_pct": _metric_after_label(
            _first_metric_window(evidence, "IV Percentile"),
            "IV Percentile",
        ),
        "implied_volatility_pct": _metric_after_label(options_data_iv, "Implied Volatility"),
        "atm_implied_volatility_pct": _metric_after_label(
            options_prices_iv,
            "Implied Volatility (ATM)",
        ),
        "historical_volatility_pct": _metric_after_label(options_data_iv, "Historic Volatility"),
        "put_call_volume_ratio": _metric_after_label(option_prices_volume, "Put/Call Volume Ratio"),
        "put_call_oi_ratio": _metric_after_label(option_prices_oi, "Put/Call Open Interest Ratio"),
        "put_open_interest_total": _metric_after_label(option_prices_volume, "Put Open Interest Total"),
        "call_open_interest_total": _metric_after_label(option_prices_volume, "Call Open Interest Total"),
        "gamma_flip_point": _metric_after_label(gamma_window, "gamma flip point is"),
        "put_wall": _metric_after_label(gamma_window, "QQQ put wall is"),
        "call_wall": _metric_after_label(gamma_window, "QQQ call wall is"),
    }
    metrics["call_wall_distance_pct"] = _distance_pct(
        metrics.get("call_wall"),
        metrics.get("latest_underlying_price"),
    )
    metrics["gamma_flip_distance_pct"] = _distance_pct(
        metrics.get("gamma_flip_point"),
        metrics.get("latest_underlying_price"),
    )
    metrics["put_wall_distance_pct"] = _distance_pct(
        metrics.get("put_wall"),
        metrics.get("latest_underlying_price"),
    )

    iv_rank = _metric_value(metrics.get("iv_rank_pct"))
    iv_percentile = _metric_value(metrics.get("iv_percentile_pct"))
    implied_vol = _metric_value(metrics.get("implied_volatility_pct"))
    hist_vol = _metric_value(metrics.get("historical_volatility_pct"))
    pc_volume = _metric_value(metrics.get("put_call_volume_ratio"))
    pc_oi = _metric_value(metrics.get("put_call_oi_ratio"))
    call_wall_distance = _to_float(metrics.get("call_wall_distance_pct"))
    gamma_flip_distance = _to_float(metrics.get("gamma_flip_distance_pct"))
    volatility_premium = (
        implied_vol - hist_vol
        if implied_vol is not None and hist_vol is not None
        else None
    )

    volatility_read = "IV는 중립권입니다."
    if iv_percentile is not None and iv_percentile >= 70:
        volatility_read = (
            f"IV Percentile {metrics.get('iv_percentile_pct')}로 최근 분포 대비 옵션값이 비싼 편입니다. "
            "방향이 맞아도 신규 추격의 손익분기점이 올라간 상태입니다."
        )
    elif iv_rank is not None and iv_rank >= 50:
        volatility_read = (
            f"IV Rank {metrics.get('iv_rank_pct')}로 변동성 가격이 중상단입니다. "
            "레버리지 증가는 확인 후가 낫습니다."
        )
    if volatility_premium is not None and volatility_premium > 4:
        volatility_read += (
            f" Implied Vol {metrics.get('implied_volatility_pct')}가 Historical Vol "
            f"{metrics.get('historical_volatility_pct')}보다 약 {volatility_premium:.2f}pt 높아 "
            "옵션 시장이 실제 변동보다 큰 움직임을 가격에 넣고 있습니다."
        )

    put_call_read = "Put/Call은 뚜렷한 쏠림이 제한적입니다."
    if pc_volume is not None and pc_oi is not None and pc_volume > 1.2 and pc_oi > 1.2:
        put_call_read = (
            f"Put/Call volume {metrics.get('put_call_volume_ratio')}, OI "
            f"{metrics.get('put_call_oi_ratio')}로 풋 수요와 기존 풋 포지션이 콜보다 큽니다. "
            "이는 약세 단정이 아니라 상승 추격에 대한 헤지 비용과 하방 보험 수요가 커졌다는 뜻입니다."
        )
    elif pc_volume is not None and pc_volume < 0.8:
        put_call_read = (
            f"Put/Call volume {metrics.get('put_call_volume_ratio')}로 콜 선호가 강합니다. "
            "방향성은 우호적일 수 있지만 과열 추격 위험도 같이 봐야 합니다."
        )

    gamma_read = "감마 기준 레벨은 참고 수준입니다."
    if call_wall_distance is not None and gamma_flip_distance is not None:
        gamma_read = (
            f"현재가 {metrics.get('latest_underlying_price')}는 gamma flip "
            f"{metrics.get('gamma_flip_point')}보다 약 {abs(gamma_flip_distance):.2f}% 위에 있어 "
            "단기 체제는 아직 붕괴 쪽보다 지지 쪽입니다. 다만 call wall "
            f"{metrics.get('call_wall')}까지 거리가 약 {call_wall_distance:.2f}%라 상단 감마 저항이 가깝습니다."
        )
        if call_wall_distance <= 1.0:
            gamma_read += " 그래서 GO 신호라도 TQQQ boost를 크게 키우기보다 작게 유지하는 해석이 맞습니다."

    if not qqq_confirmed:
        status = "실패"
    elif qqq_pages and len(qqq_confirmed) == len(qqq_pages) and futures_pages:
        status = "확인됨"
    else:
        status = "부분 제한"

    interpretation_bits: List[str] = []
    if metrics.get("iv_rank_pct") or metrics.get("iv_percentile_pct"):
        interpretation_bits.append(
            f"IV Rank {metrics.get('iv_rank_pct') or 'N/A'}, "
            f"IV Percentile {metrics.get('iv_percentile_pct') or 'N/A'}"
        )
    if metrics.get("implied_volatility_pct") or metrics.get("atm_implied_volatility_pct"):
        interpretation_bits.append(
            f"IV {metrics.get('implied_volatility_pct') or 'N/A'}, "
            f"ATM IV {metrics.get('atm_implied_volatility_pct') or 'N/A'}"
        )
    if metrics.get("put_call_volume_ratio") or metrics.get("put_call_oi_ratio"):
        interpretation_bits.append(
            f"Put/Call volume {metrics.get('put_call_volume_ratio') or 'N/A'}, "
            f"OI {metrics.get('put_call_oi_ratio') or 'N/A'}"
        )
    if metrics.get("gamma_flip_point"):
        interpretation_bits.append(
            f"Gamma flip {metrics.get('gamma_flip_point')}, "
            f"put wall {metrics.get('put_wall') or 'N/A'}, "
            f"call wall {metrics.get('call_wall') or 'N/A'}"
        )
    interpretation = (
        "; ".join(interpretation_bits)
        if interpretation_bits
        else "Barchart QQQ 옵션 페이지는 열렸지만 핵심 수치 정규화가 제한됐습니다."
    )
    interpretation += (
        ". 이 블록은 flow/NAV T-1 공백을 읽는 참고축이며 GoStop 점수, QQQ exposure, "
        "TQQQ boost, AutoTrade2 주문 신호에는 직접 반영하지 않습니다."
    )
    ai_explanation = {
        "headline": "옵션은 방향 GO를 부정하지 않지만, 추격 레버리지는 낮게 보라는 경고입니다.",
        "volatility_read": volatility_read,
        "put_call_read": put_call_read,
        "gamma_read": gamma_read,
        "action_translation": (
            "가격/DDM이 우호적이면 QQQ 유지 또는 기본 GO는 가능하지만, IV가 비싸고 "
            "put/call 방어 수요가 높으며 call wall이 가까우면 TQQQ는 작은 boost만 허용하는 쪽이 일관됩니다."
        ),
    }

    return {
        "enabled": True,
        "source": evidence.get("source") or "Barchart Premier normal Chrome",
        "status": status,
        "role": "reference_only_not_scoring",
        "generated_at_utc": evidence.get("generated_at_utc"),
        "evidence_path": evidence_path,
        "confirmed_pages": evidence.get("confirmed_pages"),
        "total_pages": evidence.get("total_pages"),
        "qqq_options_pages_confirmed": len(qqq_confirmed),
        "qqq_options_pages_total": len(qqq_pages),
        "futures_pages_confirmed": len(futures_pages),
        "metrics": metrics,
        "page_statuses": [
            {
                "label": page.get("label"),
                "status": page.get("status"),
                "final_url": page.get("final_url"),
                "body_length": page.get("body_length"),
            }
            for page in pages
        ],
        "interpretation": interpretation,
        "ai_explanation": ai_explanation,
        "limitations": limitations,
    }


def build_options_ddm_explanation(
    *,
    options_ref: Mapping[str, Any],
    ddm_signal: Mapping[str, Any],
    freshness: Mapping[str, Any],
    qqq_decision: Optional[Mapping[str, Any]] = None,
    tqqq_boost: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Explain how non-scoring options evidence should be read next to DDM."""

    qqq_decision = qqq_decision or {}
    tqqq_boost = tqqq_boost or {}
    ddm_status = str(ddm_signal.get("status") or "unknown")
    drift = _to_float(ddm_signal.get("drift"))
    diffusion = _to_float(ddm_signal.get("diffusion"))
    evidence = _to_float(ddm_signal.get("evidence"))
    confidence = _to_float(ddm_signal.get("confidence_pct"))
    agreement = _to_float(ddm_signal.get("agreement_pct"))
    support = _to_float(ddm_signal.get("support_pressure"))
    resistance = _to_float(ddm_signal.get("resistance_pressure"))
    support_ratio = (
        support / resistance
        if support is not None and resistance not in (None, 0)
        else None
    )
    flow_lag = freshness.get("flow_lag_calendar_days")
    option_ai = options_ref.get("ai_explanation") or {}
    options_status = str(options_ref.get("status") or "not_collected")
    options_available = bool(
        options_ref.get("enabled")
        and options_status not in {"not_collected", "실패"}
        and option_ai
    )

    if not ddm_signal.get("enabled"):
        ddm_read = "DDM 계산이 비활성 또는 데이터 부족이라 옵션 참고와 결합해 판단할 수 없습니다."
    else:
        drift_word = "상방" if (drift or 0) > 0 else "하방" if (drift or 0) < 0 else "중립"
        diffusion_vs_drift = (
            abs(diffusion) > abs(drift)
            if diffusion is not None and drift is not None
            else False
        )
        ddm_read = (
            f"DDM은 {ddm_status}입니다. Drift {_fmt_num(drift, 3)}는 {drift_word} 압력을 보지만, "
            f"Diffusion {_fmt_num(diffusion, 3)}"
        )
        if diffusion_vs_drift:
            ddm_read += "이 Drift 절대값보다 커서 신호가 깨끗한 단일 추세라기보다 내부 저항/불일치가 큽니다."
        else:
            ddm_read += "은 Drift보다 작아 방향 신호가 상대적으로 더 선명합니다."
        if evidence is not None and confidence is not None:
            ddm_read += (
                f" Evidence {_fmt_num(evidence, 3)}, confidence {_fmt_pct(confidence)}라 "
                "강한 확신 구간은 아닙니다."
            )
        if agreement is not None:
            ddm_read += f" 상관 바스켓 agreement는 {_fmt_pct(agreement)}입니다."
        if support_ratio is not None:
            if support_ratio >= 1.02:
                pressure_read = "support가 resistance보다 우세해 QQQ 유지/GO를 보강합니다."
            elif support_ratio <= 0.98:
                pressure_read = "resistance가 support보다 우세해 TQQQ boost를 차단하거나 감시해야 합니다."
            else:
                pressure_read = "support와 resistance가 비슷해 QQQ 기본 노출은 유지하되 boost 확신은 낮습니다."
            ddm_read += (
                f" Support pressure가 resistance의 약 {support_ratio:.2f}배라 "
                f"{pressure_read}"
            )
        if flow_lag:
            ddm_read += (
                f" 다만 Flow/NAV가 가격보다 T-{flow_lag}라 당일 수급 확정 신호가 아니라 "
                "전일 flow와 최신 가격을 섞은 판독입니다."
            )

    decision_action = qqq_decision.get("action") or "N/A"
    exposure = qqq_decision.get("recommended_exposure_pct", "N/A")
    exposure_num = _to_float(exposure)
    boost = tqqq_boost.get("tqqq_boost_pct", 0)
    ddm_constructive = bool(
        (drift or 0) > 0
        and (evidence is None or evidence >= 0)
        and (support_ratio is None or support_ratio >= 1.0)
        and ddm_status not in {"block", "fail", "failed"}
    )
    if exposure_num is not None and exposure_num <= 0:
        exposure_read = (
            f"결론적으로 QQQ action {decision_action}, exposure {exposure}%는 유지가 아니라 "
            "현금/회피 상태입니다. 가격 반등 신호가 일부 있어도 DDM 위험 overlay가 hard block으로 "
            "기본 QQQ 노출까지 차단한 판독입니다. "
        )
        risk_posture_read = (
            "QQQ 기본 노출을 유지하지 않고 현금/관찰 상태를 우선하며, 옵션 비용과 감마 레벨은 "
            "재진입 확인용 참고축으로만 분리해서 봅니다."
        )
    elif exposure_num is not None and exposure_num < 100:
        exposure_read = (
            f"결론적으로 QQQ action {decision_action}, exposure {exposure}%는 전면 유지가 아니라 "
            "부분 노출/탐색 구간입니다. DDM과 가격 fresh 조건이 충돌해 기본 노출을 줄인 상태입니다. "
        )
        risk_posture_read = (
            "QQQ 기본 노출을 일부만 유지하고, 옵션 비용과 감마 레벨을 확인해 추가 레버리지 추격은 "
            "분리해서 판단하는 상태입니다."
        )
    elif ddm_constructive:
        fusion_read = (
            f"결론적으로 QQQ action {decision_action}, exposure {exposure}%는 DDM의 상방 압력과 "
            "가격 fresh 조건이 함께 지지해 유지됩니다."
        )
        risk_posture_read = (
            "QQQ 기본 노출은 유지하되, 옵션 비용과 감마 레벨을 확인해 새 레버리지 추격은 "
            "분리해서 판단하는 상태입니다."
        )
    else:
        fusion_read = (
            f"결론적으로 QQQ action {decision_action}, exposure {exposure}%는 DDM 상방 확신 때문이 아니라 "
            "QQQ buy-and-hold 기본 정책과 가격 fresh 조건 때문에 유지됩니다. DDM은 boost 확신을 낮추는 "
            f"감시/차단 축이라 TQQQ boost는 {boost}%로 제한됩니다."
        )
        risk_posture_read = (
            "QQQ 기본 노출은 유지하되, 옵션 비용과 감마 레벨을 확인해 새 레버리지 추격은 "
            "분리해서 판단하는 상태입니다."
        )
    if exposure_num is not None and exposure_num < 100:
        fusion_read = (
            exposure_read
            + f"DDM은 boost 확신을 낮추는 감시/차단 축이라 TQQQ boost는 {boost}%로 제한됩니다."
        )
    if options_available:
        fusion_read += (
            " 옵션 쪽은 IV/put-call/gamma를 reference_only_not_scoring으로 보며, 추격 비용과 상단 저항이 "
            f"확인될 때 TQQQ boost {boost}% sizing을 더 보수적으로 읽습니다."
        )
        risk_read = (
            f"이 조합은 '{risk_posture_read}' 옵션 블록은 점수 입력이 아니라 "
            "DDM/flow 공백을 읽는 품질 감사와 sizing 설명입니다."
        )
        option_read = " ".join(
            str(option_ai.get(key) or "")
            for key in ("volatility_read", "put_call_read", "gamma_read")
            if option_ai.get(key)
        )
    else:
        option_read = (
            "Barchart 옵션/선물 참고 데이터가 미수집이라 IV, put/call, gamma flip, put/call wall 해석은 "
            "보류합니다. 이 상태에서는 옵션 비용이나 call wall을 확인된 근거처럼 쓰지 않습니다."
        )
        risk_read = (
            "이번 결합 해석은 FMP 가격, Massive flow/NAV, DDM, nowcast 중심입니다. 옵션/선물 공백은 "
            "source limitation이며 GoStop 점수나 TQQQ boost 판단의 원천 확인으로 취급하지 않습니다."
        )

    return {
        "headline": (
            "DDM은 QQQ 기본 노출과 TQQQ boost를 분리해 읽어야 합니다."
            if options_available
            else "Barchart 옵션 미수집: QQQ/DDM 해석과 옵션 해석을 분리합니다."
        ),
        "ddm_read": ddm_read,
        "option_read": option_read,
        "fusion_read": fusion_read,
        "risk_read": risk_read,
        "source_note": (
            "Barchart 옵션/선물은 reference_only_not_scoring이며, DDM은 FMP price + "
            "Massive ETF flow/NAV 기반입니다."
        ),
    }


def _avg_for_group(rows: Sequence[RadarRow], symbols: set, attr: str) -> Optional[float]:
    return _mean(getattr(row, attr) for row in rows if row.symbol in symbols)


def _sum_for_group(rows: Sequence[RadarRow], symbols: set, attr: str) -> Optional[float]:
    return _sum(getattr(row, attr) for row in rows if row.symbol in symbols)


def classify_market_state(rows: Sequence[RadarRow]) -> Dict[str, Any]:
    risk_price = _avg_for_group(rows, RISK_ETFS, "price_5d_pct")
    risk_flow_aum = _avg_for_group(rows, RISK_ETFS, "flow_aum_5d_pct")
    risk_nav_gap = _avg_for_group(rows, RISK_ETFS, "nav_gap_pct")
    defensive_flow_aum = _avg_for_group(rows, DEFENSIVE_ETFS, "flow_aum_5d_pct")
    broad_flow_aum = _avg_for_group(rows, {"SPY", "QQQ"}, "flow_aum_5d_pct")
    small_cyclical_flow_aum = _avg_for_group(rows, {"IWM", "DIA"}, "flow_aum_5d_pct")
    credit_flow_aum = _avg_for_group(rows, {"HYG", "LQD"}, "flow_aum_5d_pct")
    risk_flow_total = _sum_for_group(rows, RISK_ETFS, "fund_flow_5d")
    defensive_flow_total = _sum_for_group(rows, DEFENSIVE_ETFS, "fund_flow_5d")
    sector_flow_total = _sum_for_group(rows, SECTOR_ETFS, "fund_flow_5d")
    total_flow_5d = _sum_for_group(rows, {row.symbol for row in rows}, "fund_flow_5d")

    tqqq = next((row for row in rows if row.symbol == "TQQQ"), None)
    leverage_component = tqqq.flow_aum_5d_pct if tqqq else None
    price_component = _clamp((risk_price or 0.0) / 4.0, -1.0, 1.0) * 20.0
    flow_component = _clamp((risk_flow_aum or 0.0) / 0.50, -1.0, 1.0) * 20.0
    nav_component = _clamp((risk_nav_gap or 0.0) / 0.30, -1.0, 1.0) * 10.0
    leverage_score = _clamp((leverage_component or 0.0) / 1.0, -1.0, 1.0) * 10.0
    defensive_drag = _clamp((defensive_flow_aum or 0.0) / 0.50, -1.0, 1.0) * 8.0
    greed_score = int(round(_clamp(50.0 + price_component + flow_component + nav_component + leverage_score - defensive_drag, 0.0, 100.0)))

    signal_counts: Dict[str, int] = {}
    for row in rows:
        signal_counts[row.signal] = signal_counts.get(row.signal, 0) + 1

    if greed_score >= 76 or signal_counts.get("greed_overheat", 0) >= 2:
        label = "greed_overheat"
        summary = "리스크 ETF의 가격/유입/NAV 프리미엄이 강해 과열 쪽으로 기울었습니다."
    elif (risk_price or 0) > 1.0 and (risk_flow_aum or 0) > 0.05:
        label = "healthy_risk_on"
        summary = "가격 상승과 ETF 유입이 동행해 위험선호가 비교적 건강합니다."
    elif (risk_price or 0) > 1.0 and (risk_flow_aum or 0) < -0.05:
        label = "fragile_rally"
        summary = "가격은 오르지만 ETF 자금은 빠져 랠리 신뢰도가 낮습니다."
    elif (risk_price or 0) < -1.0 and (risk_flow_aum or 0) < -0.05:
        label = "risk_off_distribution"
        summary = "가격 하락과 자금 유출이 동행해 리스크 축소 흐름입니다."
    elif (risk_price or 0) < 0 and (defensive_flow_aum or 0) > 0.05:
        label = "defensive_rotation"
        summary = "주식 리스크는 약하고 방어/채권/금 쪽으로 자금이 기웁니다."
    else:
        label = "mixed_transition"
        summary = "가격과 flow가 완전히 정렬되지 않은 전환/혼조 구간입니다."

    strongest = sorted(
        rows,
        key=lambda row: (
            row.flow_aum_5d_pct if row.flow_aum_5d_pct is not None else -999.0,
            row.price_5d_pct if row.price_5d_pct is not None else -999.0,
        ),
        reverse=True,
    )[:5]
    weakest = sorted(
        rows,
        key=lambda row: (
            row.flow_aum_5d_pct if row.flow_aum_5d_pct is not None else 999.0,
            row.price_5d_pct if row.price_5d_pct is not None else 999.0,
        ),
    )[:5]

    return {
        "label": label,
        "summary": summary,
        "greed_score": greed_score,
        "risk_price_5d_avg_pct": risk_price,
        "risk_flow_5d_aum_avg_pct": risk_flow_aum,
        "risk_nav_gap_avg_pct": risk_nav_gap,
        "defensive_flow_5d_aum_avg_pct": defensive_flow_aum,
        "broad_flow_5d_aum_avg_pct": broad_flow_aum,
        "small_cyclical_flow_5d_aum_avg_pct": small_cyclical_flow_aum,
        "credit_flow_5d_aum_avg_pct": credit_flow_aum,
        "leverage_flow_5d_aum_pct": leverage_component,
        "risk_flow_5d_total": risk_flow_total,
        "defensive_flow_5d_total": defensive_flow_total,
        "sector_flow_5d_total": sector_flow_total,
        "total_flow_5d": total_flow_5d,
        "signal_counts": signal_counts,
        "strongest": [row.symbol for row in strongest],
        "weakest": [row.symbol for row in weakest],
    }


def _row_map(rows: Sequence[RadarRow]) -> Dict[str, RadarRow]:
    return {str(row.symbol).upper(): row for row in rows}


def _flow_aum(row: Optional[RadarRow]) -> Optional[float]:
    return row.flow_aum_5d_pct if row is not None else None


def _price_5d(row: Optional[RadarRow]) -> Optional[float]:
    return row.price_5d_pct if row is not None else None


def _is_positive(value: Optional[float], threshold: float = 0.0) -> bool:
    return value is not None and value > threshold


def _is_negative(value: Optional[float], threshold: float = 0.0) -> bool:
    return value is not None and value < threshold


def classify_gostop_decision(
    rows: Sequence[RadarRow],
    market_state: Mapping[str, Any],
    *,
    warnings: Optional[Sequence[str]] = None,
) -> GoStopDecision:
    """Convert radar readings into a fresh-entry Go/Stop gate.

    The score intentionally penalizes crowded or conflicted markets. A high
    greed score can therefore produce WAIT rather than GO when chase risk is
    high.
    """

    by_symbol = _row_map(rows)
    spy = by_symbol.get("SPY")
    qqq = by_symbol.get("QQQ")
    iwm = by_symbol.get("IWM")
    dia = by_symbol.get("DIA")
    tqqq = by_symbol.get("TQQQ")
    hyg = by_symbol.get("HYG")
    lqd = by_symbol.get("LQD")

    risk_price = _to_float(market_state.get("risk_price_5d_avg_pct"))
    risk_flow = _to_float(market_state.get("risk_flow_5d_aum_avg_pct"))
    risk_nav_gap = _to_float(market_state.get("risk_nav_gap_avg_pct"))
    defensive_flow = _to_float(market_state.get("defensive_flow_5d_aum_avg_pct"))
    sector_flow = _to_float(market_state.get("sector_flow_5d_total"))
    greed_score = int(market_state.get("greed_score") or 50)
    signal_counts = dict(market_state.get("signal_counts") or {})

    score = 50.0
    reasons: List[str] = []
    blocks: List[str] = []

    if risk_price is None:
        score -= 5
        blocks.append("Risk ETF 5D 가격 평균을 계산할 수 없습니다.")
    elif risk_price >= 1.0:
        score += 12
        reasons.append(f"Risk ETF 5D 가격 평균이 {_fmt_pct(risk_price)}로 양호합니다.")
    elif risk_price > 0:
        score += 5
        reasons.append(f"Risk ETF 가격은 약한 플러스입니다({_fmt_pct(risk_price)}).")
    elif risk_price <= -1.0:
        score -= 16
        blocks.append(f"Risk ETF 5D 가격 평균이 {_fmt_pct(risk_price)}로 약합니다.")
    else:
        score -= 4
        blocks.append(f"Risk ETF 가격 모멘텀이 부족합니다({_fmt_pct(risk_price)}).")

    if risk_flow is None:
        score -= 8
        blocks.append("Risk ETF 5D flow/AUM 평균을 계산할 수 없습니다.")
    elif risk_flow >= 0.10:
        score += 16
        reasons.append(f"Risk ETF flow/AUM이 {_fmt_pct(risk_flow, 3)}로 신규 자금을 확인합니다.")
    elif risk_flow > 0:
        score += 8
        reasons.append(f"Risk ETF flow/AUM이 플러스입니다({_fmt_pct(risk_flow, 3)}).")
    elif risk_flow <= -0.10:
        score -= 18
        blocks.append(f"Risk ETF flow/AUM이 {_fmt_pct(risk_flow, 3)}로 유출 우위입니다.")
    else:
        score -= 6
        blocks.append(f"Risk ETF flow/AUM이 약합니다({_fmt_pct(risk_flow, 3)}).")

    spy_qqq_positive = sum(
        1 for row in (spy, qqq) if _is_positive(_flow_aum(row), 0.0)
    )
    if spy_qqq_positive == 2:
        score += 8
        reasons.append("SPY와 QQQ 모두 5D flow/AUM이 플러스입니다.")
    elif spy_qqq_positive == 1:
        score += 3
        reasons.append("SPY/QQQ 중 하나만 flow가 시장을 지지합니다.")
    else:
        score -= 8
        blocks.append("SPY와 QQQ flow가 신규 진입을 확인하지 못합니다.")

    credit_flow = _mean([_flow_aum(hyg), _flow_aum(lqd)])
    if credit_flow is not None:
        if credit_flow <= -0.05:
            score -= 12
            blocks.append(f"HYG/LQD credit flow가 약합니다({_fmt_pct(credit_flow, 3)}).")
        elif credit_flow > 0:
            score += 6
            reasons.append(f"Credit ETF flow가 시장 확산을 보강합니다({_fmt_pct(credit_flow, 3)}).")

    small_cyclical_weak = (
        (_is_negative(_price_5d(iwm), -0.50) or _is_negative(_flow_aum(iwm), -0.05))
        and (_is_negative(_price_5d(dia), -0.50) or _is_negative(_flow_aum(dia), -0.05))
    )
    if small_cyclical_weak:
        score -= 8
        blocks.append("IWM/DIA가 약해 broad risk-on 확산이 부족합니다.")

    if defensive_flow is not None:
        if defensive_flow > 0.08 and (risk_price or 0.0) <= 0:
            score -= 10
            blocks.append(f"Defensive flow가 강해 방어 로테이션입니다({_fmt_pct(defensive_flow, 3)}).")
        elif defensive_flow < -0.05 and (risk_price or 0.0) > 0:
            score += 4
            reasons.append("방어 ETF 유입이 강하지 않아 risk entry에 부담이 덜합니다.")

    if sector_flow is not None and sector_flow > 0:
        score += 3
        reasons.append("섹터 ETF 5D flow 총합이 플러스입니다.")

    overheat_count = int(signal_counts.get("greed_overheat", 0) or 0)
    stress_count = int(signal_counts.get("stress_outflow", 0) or 0)
    distribution_count = int(signal_counts.get("distribution", 0) or 0)
    fragile_count = int(signal_counts.get("fragile_rally", 0) or 0)
    tqqq_flow = _flow_aum(tqqq)

    chase_risk = False
    if greed_score >= 76:
        chase_risk = True
        score -= 10
        blocks.append(f"Greed score가 {greed_score}/100으로 추격 리스크가 큽니다.")
    if risk_nav_gap is not None and risk_nav_gap >= 0.20:
        chase_risk = True
        score -= 8
        blocks.append(f"Risk ETF NAV 프리미엄이 큽니다({_fmt_pct(risk_nav_gap, 3)}).")
    if overheat_count >= 2:
        chase_risk = True
        score -= 8
        blocks.append(f"과열 ETF 신호가 {overheat_count}개입니다.")
    if tqqq_flow is not None and tqqq_flow >= 0.75:
        chase_risk = True
        score -= 8
        blocks.append(f"TQQQ flow/AUM이 {_fmt_pct(tqqq_flow, 3)}로 레버리지 crowding이 큽니다.")

    if stress_count or distribution_count:
        penalty = min(18, stress_count * 8 + distribution_count * 4)
        score -= penalty
        blocks.append(f"분배/스트레스 신호가 {stress_count + distribution_count}개 있습니다.")
    if fragile_count >= 2:
        score -= 6
        blocks.append(f"가격 상승 대비 flow 약화 신호가 {fragile_count}개입니다.")

    data_warnings = list(warnings or [])
    if data_warnings:
        score -= min(10, 2 * len(data_warnings))
        blocks.append("데이터 경고가 있어 판정 신뢰도를 낮춥니다.")

    score_int = int(round(_clamp(score, 0.0, 100.0)))
    hard_stop = (
        (risk_price is not None and risk_price <= -1.0 and risk_flow is not None and risk_flow <= -0.05)
        or stress_count >= 2
        or (credit_flow is not None and credit_flow <= -0.08 and small_cyclical_weak and (risk_price or 0.0) < 0)
    )

    if hard_stop or score_int < 35:
        action = "STOP"
        mode = "no_fresh_entry"
        max_new_risk_pct = 0
        headline = "신규 진입 중단. 가격/flow/credit 중 하나 이상이 리스크 축소를 가리킵니다."
    elif chase_risk and score_int >= 50:
        action = "WAIT"
        mode = "pullback_only"
        max_new_risk_pct = 0
        headline = "방향성은 살아 있지만 추격 진입은 금지입니다. 눌림과 flow 유지 확인이 필요합니다."
    elif score_int >= 72 and not blocks:
        action = "GO"
        mode = "normal_entry"
        max_new_risk_pct = 100
        headline = "신규 진입 허용. 가격, flow, credit 확인이 대체로 정렬되어 있습니다."
    elif score_int >= 58:
        action = "GO_SMALL"
        mode = "selective_entry"
        max_new_risk_pct = 50
        headline = "선별 소액 진입만 허용. 시장 전체보다 강한 ETF/섹터만 봅니다."
    elif score_int >= 45:
        action = "WAIT"
        mode = "watch_only"
        max_new_risk_pct = 0
        headline = "대기. 신호가 혼재되어 신규 진입보다 확인이 우선입니다."
    else:
        action = "STOP"
        mode = "no_fresh_entry"
        max_new_risk_pct = 0
        headline = "신규 진입 중단. Go 조건이 부족합니다."

    positive_signals = {
        "healthy_risk_on",
        "constructive_inflow",
        "dip_buying",
        "quiet_accumulation",
    }
    negative_signals = {
        "greed_overheat",
        "fragile_rally",
        "distribution",
        "stress_outflow",
        "quiet_redemption",
    }
    entry_candidates = [
        row.symbol
        for row in sorted(rows, key=_row_sort_key, reverse=True)
        if row.signal in positive_signals
    ][:6]
    stop_candidates = [
        row.symbol
        for row in sorted(rows, key=_row_sort_key, reverse=True)
        if row.signal in negative_signals
    ][:6]

    return GoStopDecision(
        action=action,
        score=score_int,
        mode=mode,
        max_new_risk_pct=max_new_risk_pct,
        headline=headline,
        reasons=reasons[:6],
        blocks=blocks[:8],
        entry_candidates=entry_candidates,
        stop_candidates=stop_candidates,
    )


def _row_sort_key(row: RadarRow) -> Tuple[float, float]:
    flow = row.flow_aum_5d_pct if row.flow_aum_5d_pct is not None else 0.0
    price = row.price_5d_pct if row.price_5d_pct is not None else 0.0
    return (abs(flow), abs(price))


def _rows_from_payload(payload: Mapping[str, Any]) -> List[RadarRow]:
    return [
        RadarRow(**row) if isinstance(row, dict) else row
        for row in payload.get("rows", [])
    ]


def _latest_price_date(rows: Sequence[RadarRow]) -> Optional[str]:
    dates = [row.price_date for row in rows if row.price_date]
    return max(dates) if dates else None


def _price_lookup(rows: Sequence[RadarRow]) -> Dict[str, float]:
    return {
        row.symbol: float(row.latest_price)
        for row in rows
        if row.latest_price is not None
    }


def _avg_forward_return(
    symbols: Sequence[str],
    *,
    current_prices: Mapping[str, float],
    prior_prices: Mapping[str, float],
) -> Optional[float]:
    returns: List[float] = []
    for symbol in symbols:
        current = current_prices.get(symbol)
        prior = prior_prices.get(symbol)
        change = _pct_change(current, prior)
        if change is not None:
            returns.append(change)
    return _mean(returns)


def _pearson_corr(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    if len(pairs) < 5:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return None
    return num / (den_x * den_y)


def _daily_return_map(
    prices: Sequence[PricePoint],
    *,
    asof_date: str,
) -> Dict[str, float]:
    asof = _parse_date(asof_date) or dt.date.max
    clean = sorted(
        [
            point
            for point in prices
            if point.close is not None and (_parse_date(point.date) or dt.date.max) <= asof
        ],
        key=lambda point: point.date,
    )
    returns: Dict[str, float] = {}
    prev_close: Optional[float] = None
    for point in clean:
        if prev_close not in (None, 0):
            change = _pct_change(point.close, prev_close)
            if change is not None:
                returns[point.date] = change
        prev_close = point.close
    return returns


def _bounded_signal(value: Optional[float], scale: float) -> Optional[float]:
    number = _to_float(value)
    if number is None or scale <= 0:
        return None
    return _clamp(number / scale, -3.0, 3.0)


def _weighted_avg(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    total_weight = sum(weight for _, weight in pairs if weight > 0)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in pairs if weight > 0) / total_weight


def _weighted_stdev(pairs: Sequence[Tuple[float, float]], mean_value: float) -> Optional[float]:
    total_weight = sum(weight for _, weight in pairs if weight > 0)
    if total_weight <= 0:
        return None
    variance = sum(
        weight * (value - mean_value) ** 2
        for value, weight in pairs
        if weight > 0
    ) / total_weight
    return math.sqrt(max(0.0, variance))


def _sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def build_ddm_signal(
    *,
    symbols: Sequence[str],
    price_map: Mapping[str, Sequence[PricePoint]],
    rows: Sequence[RadarRow],
    nowcast: Optional[Mapping[str, Any]] = None,
    asof_date: str,
    target: str = "QQQ",
    corr_lookback_bars: int = DDM_CORR_LOOKBACK_BARS,
    min_overlap: int = DDM_MIN_CORR_OVERLAP,
    min_abs_corr: float = DDM_MIN_ABS_CORR,
) -> Dict[str, Any]:
    """Estimate QQQ drift/diffusion from correlated ETF basket pressure.

    Only data available at ``asof_date`` is used. Positive-correlation ETFs add
    evidence when they rise/attract flow. Negative-correlation ETFs add evidence
    when they fall/lose flow. Diffusion is the internal disagreement and
    resistance inside that correlated basket.
    """

    target_symbol = target.upper()
    target_derived_exclusions = DDM_TARGET_DERIVED_EXCLUSIONS.get(target_symbol, set())
    excluded_self_proxy_symbols: List[str] = []
    target_returns = _daily_return_map(
        price_map.get(target_symbol) or [],
        asof_date=asof_date,
    )
    if len(target_returns) < min_overlap:
        return {
            "enabled": False,
            "status": "insufficient_target_history",
            "target": target_symbol,
            "note": "QQQ 상관 바스켓을 계산할 과거 가격 데이터가 부족합니다.",
        }

    row_by_symbol = {row.symbol.upper(): row for row in rows}
    now_rows = {
        str(row.get("symbol") or "").upper(): row
        for row in (nowcast or {}).get("rows", [])
        if isinstance(row, Mapping)
    }
    target_dates = sorted(target_returns)[-max(min_overlap, int(corr_lookback_bars)) :]
    components: List[Dict[str, Any]] = []

    for symbol in symbols:
        sym = symbol.upper()
        if sym == target_symbol:
            continue
        if sym in target_derived_exclusions:
            excluded_self_proxy_symbols.append(sym)
            continue
        symbol_returns = _daily_return_map(price_map.get(sym) or [], asof_date=asof_date)
        pairs = [
            (target_returns[date], symbol_returns[date])
            for date in target_dates
            if date in target_returns and date in symbol_returns
        ]
        if len(pairs) < min_overlap:
            continue
        corr = _pearson_corr(pairs)
        if corr is None or abs(corr) < min_abs_corr:
            continue

        row = row_by_symbol.get(sym)
        now_row = now_rows.get(sym) or {}
        price_component = _bounded_signal(row.price_5d_pct if row else None, 3.0)
        flow_component = _bounded_signal(row.flow_aum_5d_pct if row else None, 1.0)
        day_component = _bounded_signal(now_row.get("day_change_pct"), 1.5)
        parts = [
            (price_component, 0.50),
            (flow_component, 0.35),
            (day_component, 0.15),
        ]
        impulse_base = _weighted_avg(
            [(value, weight) for value, weight in parts if value is not None]
        )
        if impulse_base is None:
            continue
        rel_volume = _to_float(now_row.get("rel_volume"))
        if rel_volume is not None:
            volume_multiplier = 1.0 + _clamp((rel_volume - 1.0) * 0.25, -0.15, 0.25)
        else:
            volume_multiplier = 1.0
        impulse = impulse_base * volume_multiplier
        direction = 1.0 if corr >= 0 else -1.0
        signed_pressure = direction * impulse
        weight = abs(corr)
        components.append(
            {
                "symbol": sym,
                "corr": corr,
                "direction": "positive_corr" if corr >= 0 else "negative_corr",
                "weight": weight,
                "price_5d_pct": row.price_5d_pct if row else None,
                "flow_aum_5d_pct": row.flow_aum_5d_pct if row else None,
                "day_change_pct": now_row.get("day_change_pct"),
                "rel_volume": rel_volume,
                "impulse": impulse,
                "signed_pressure": signed_pressure,
                "qqq_support": signed_pressure > 0,
            }
        )

    if not components:
        return {
            "enabled": False,
            "status": "no_correlated_basket",
            "target": target_symbol,
            "excluded_self_proxy_symbols": sorted(set(excluded_self_proxy_symbols)),
            "note": "현재 lookback에서 QQQ와 충분히 상관된 ETF 바스켓이 없습니다.",
        }

    pressure_pairs = [
        (float(item["signed_pressure"]), float(item["weight"]))
        for item in components
    ]
    drift = _weighted_avg(pressure_pairs) or 0.0
    dispersion = _weighted_stdev(pressure_pairs, drift) or 0.0
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
    diffusion = dispersion + resistance_ratio * 1.25
    evidence = drift / (diffusion + 0.35)
    high_corr_count = sum(1 for item in components if abs(float(item["corr"])) >= DDM_HIGH_ABS_CORR)
    confidence = _clamp(_sigmoid(evidence * 1.65) * 100.0, 0.0, 100.0)

    if drift > 0 and agreement >= 0.60 and confidence >= 65 and high_corr_count >= 3:
        boundary = "boost"
        boost_cap_multiplier = 1.0
    elif drift > 0 and agreement >= 0.55 and confidence >= 58:
        boundary = "constructive"
        boost_cap_multiplier = 0.5
    elif drift <= 0 or agreement < 0.50:
        boundary = "block"
        boost_cap_multiplier = 0.0
    else:
        boundary = "neutral"
        boost_cap_multiplier = 0.25

    sorted_components = sorted(
        components,
        key=lambda item: abs(float(item["signed_pressure"]) * float(item["weight"])),
        reverse=True,
    )
    support = [item for item in sorted_components if item.get("qqq_support")]
    resistance = [item for item in sorted_components if not item.get("qqq_support")]
    return {
        "enabled": True,
        "status": boundary,
        "target": target_symbol,
        "asof_date": asof_date,
        "corr_lookback_bars": corr_lookback_bars,
        "min_overlap": min_overlap,
        "min_abs_corr": min_abs_corr,
        "excluded_self_proxy_symbols": sorted(set(excluded_self_proxy_symbols)),
        "correlated_count": len(components),
        "high_corr_count": high_corr_count,
        "drift": drift,
        "diffusion": diffusion,
        "dispersion": dispersion,
        "evidence": evidence,
        "confidence_pct": confidence,
        "agreement_pct": agreement * 100.0,
        "support_pressure": support_pressure,
        "resistance_pressure": resistance_pressure,
        "boost_cap_multiplier": boost_cap_multiplier,
        "support": support[:8],
        "resistance": resistance[:8],
        "components": sorted_components[:16],
        "all_components": sorted_components,
        "note": (
            "Drift는 QQQ와 상관된 ETF 바스켓의 순방향 압력, diffusion은 내부 저항/불일치입니다. "
            "QQQ 파생 레버리지/인버스 ETF는 독립 DDM 증거가 아니라 crowding/boost 맥락으로만 봅니다."
        ),
    }


def _gostop_feature_snapshot(payload: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    state = payload.get("market_state") or {}
    decision = payload.get("gostop") or {}
    nowcast = payload.get("nowcast") or {}
    tactical = payload.get("tactical_overlay") or {}
    qqq_decision = payload.get("qqq_decision") or {}
    ddm = payload.get("ddm_signal") or {}
    counts = state.get("signal_counts") or {}
    return {
        "gostop_score": _to_float(decision.get("score")),
        "qqq_exposure_pct": _to_float(qqq_decision.get("recommended_exposure_pct")),
        "nowcast_score": _to_float(nowcast.get("score")),
        "nowcast_risk_day": _to_float(nowcast.get("risk_day_avg_pct")),
        "nowcast_risk_breadth": (
            _to_float(nowcast.get("risk_positive_count")) / _to_float(nowcast.get("risk_count"))
            if _to_float(nowcast.get("risk_positive_count")) is not None
            and _to_float(nowcast.get("risk_count")) not in (None, 0)
            else None
        ),
        "nowcast_rel_volume": _to_float(nowcast.get("risk_rel_volume_median")),
        "tactical_probe_pct": _to_float(tactical.get("max_tactical_risk_pct")),
        "greed_score": _to_float(state.get("greed_score")),
        "risk_price_5d": _to_float(state.get("risk_price_5d_avg_pct")),
        "risk_flow_5d_aum": _to_float(state.get("risk_flow_5d_aum_avg_pct")),
        "risk_nav_gap": _to_float(state.get("risk_nav_gap_avg_pct")),
        "defensive_flow_5d_aum": _to_float(state.get("defensive_flow_5d_aum_avg_pct")),
        "broad_flow_5d_aum": _to_float(state.get("broad_flow_5d_aum_avg_pct")),
        "small_cyclical_flow_5d_aum": _to_float(state.get("small_cyclical_flow_5d_aum_avg_pct")),
        "credit_flow_5d_aum": _to_float(state.get("credit_flow_5d_aum_avg_pct")),
        "leverage_flow_5d_aum": _to_float(state.get("leverage_flow_5d_aum_pct")),
        "risk_flow_5d_total": _to_float(state.get("risk_flow_5d_total")),
        "defensive_flow_5d_total": _to_float(state.get("defensive_flow_5d_total")),
        "sector_flow_5d_total": _to_float(state.get("sector_flow_5d_total")),
        "total_flow_5d": _to_float(state.get("total_flow_5d")),
        "distribution_count": _to_float(counts.get("distribution")),
        "fragile_rally_count": _to_float(counts.get("fragile_rally")),
        "stress_outflow_count": _to_float(counts.get("stress_outflow")),
        "quiet_accumulation_count": _to_float(counts.get("quiet_accumulation")),
        "quiet_redemption_count": _to_float(counts.get("quiet_redemption")),
        "ddm_drift": _to_float(ddm.get("drift")),
        "ddm_diffusion": _to_float(ddm.get("diffusion")),
        "ddm_evidence": _to_float(ddm.get("evidence")),
        "ddm_confidence": _to_float(ddm.get("confidence_pct")),
        "ddm_agreement": _to_float(ddm.get("agreement_pct")),
        "ddm_correlated_count": _to_float(ddm.get("correlated_count")),
        "ddm_support_pressure": _to_float(ddm.get("support_pressure")),
        "ddm_resistance_pressure": _to_float(ddm.get("resistance_pressure")),
    }


def _rolling_feature_correlations(
    feature_targets: Sequence[Tuple[Dict[str, Optional[float]], Dict[str, float]]]
) -> List[Dict[str, Any]]:
    correlations: List[Dict[str, Any]] = []
    feature_names = sorted({name for item, _ in feature_targets for name in item})
    for feature_name in feature_names:
        for target_name in (
            "action_utility",
            "qqq_decision_utility",
            "qqq_strategy_return",
            "tactical_utility",
            "qqq_return",
            "entry_excess",
            "stop_utility",
        ):
            pairs: List[Tuple[float, float]] = []
            for features, targets in feature_targets:
                feature_value = features.get(feature_name)
                target_value = targets.get(target_name)
                if feature_value is not None and target_value is not None:
                    pairs.append((feature_value, target_value))
            corr = _pearson_corr(pairs)
            if corr is not None:
                correlations.append(
                    {
                        "feature": feature_name,
                        "target": target_name,
                        "corr": corr,
                        "n": len(pairs),
                    }
                )
    correlations.sort(key=lambda item: abs(float(item.get("corr") or 0.0)), reverse=True)
    return correlations


def _hit_rate(values: Sequence[float], threshold: float) -> Optional[float]:
    if not values:
        return None
    hits = [1.0 if value > threshold else 0.0 for value in values]
    return (_mean(hits) or 0.0) * 100.0


def _score_threshold_tuning(
    score_returns: Sequence[Tuple[int, float]],
    *,
    hit_threshold_pct: float,
    thresholds: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    threshold_values = list(thresholds or range(35, 81, 5))
    results: List[Dict[str, Any]] = []
    for threshold in threshold_values:
        utilities: List[float] = []
        go_count = 0
        for score, qqq_return in score_returns:
            is_go = score >= threshold
            go_count += 1 if is_go else 0
            utilities.append(qqq_return if is_go else -qqq_return)
        if not utilities:
            continue
        hit_rate = _hit_rate(utilities, hit_threshold_pct)
        go_rate = go_count / len(utilities) * 100.0
        avg_utility = _mean(utilities)
        # Prefer thresholds that make money, hit often, and still trade often
        # enough to avoid picking a trivial always-stop policy.
        objective = (avg_utility or 0.0) + ((hit_rate or 0.0) - 50.0) * 0.02
        if go_rate < 10.0 or go_rate > 90.0:
            objective -= 0.25
        results.append(
            {
                "entry_score_min": threshold,
                "n": len(utilities),
                "go_rate_pct": go_rate,
                "hit_rate_pct": hit_rate,
                "avg_utility_pct": avg_utility,
                "objective": objective,
            }
        )
    results.sort(key=lambda item: float(item.get("objective") or -999.0), reverse=True)
    return results


def _tactical_exposure_pct(tactical: Mapping[str, Any]) -> float:
    action = str(tactical.get("action") or "").upper()
    if action in {"GO", "GO_SMALL", "TACTICAL_GO_SMALL", "WATCH_REBOUND"}:
        return _clamp(float(_to_float(tactical.get("max_tactical_risk_pct")) or 0.0), 0.0, 100.0)
    return 0.0


def _gostop_exposure_pct(gostop: Mapping[str, Any]) -> float:
    action = str(gostop.get("action") or "").upper()
    if action not in {"GO", "GO_SMALL"}:
        return 0.0
    default = 100.0 if action == "GO" else 50.0
    exposure = _to_float(gostop.get("max_new_risk_pct"))
    return _clamp(exposure if exposure is not None else default, 0.0, 100.0)


def _qqq_policy_key(swing_action: str, tactical_action: Optional[str]) -> str:
    tactical_key = str(tactical_action or "").upper()
    if tactical_key in {
        "GO",
        "GO_SMALL",
        "TACTICAL_GO_SMALL",
        "WATCH_REBOUND",
        "DELAY_ENTRY",
        "STOP",
        "WAIT",
    }:
        return tactical_key
    return str(swing_action or "UNKNOWN").upper()


def _adaptive_context_exposure_cap(policy_key: str, swing_action: str) -> int:
    key = str(policy_key or "").upper()
    swing = str(swing_action or "").upper()
    if swing == "STOP":
        if key in {"WATCH_REBOUND", "TACTICAL_GO_SMALL"}:
            return 60
        if key == "STOP":
            return 25
        return 40
    if swing == "WAIT":
        if key in {"WATCH_REBOUND", "TACTICAL_GO_SMALL"}:
            return 60
        return 40
    if key == "GO_SMALL":
        return 60
    return 100


def _qqq_decision_utility(qqq_return: float, exposure_pct: float) -> float:
    """Directional QQQ timing utility.

    If the report says to own QQQ, QQQ going up is correct. If it says to hold
    cash, QQQ going down is correct. Position sizing is evaluated separately.
    """

    return qqq_return if exposure_pct > 0 else -qqq_return


def _qqq_strategy_return(qqq_return: float, exposure_pct: float) -> float:
    return qqq_return * (_clamp(exposure_pct, 0.0, 100.0) / 100.0)


def _qqq_missed_upside(qqq_return: float, exposure_pct: float) -> float:
    if qqq_return <= 0:
        return 0.0
    return qqq_return * (1.0 - _clamp(exposure_pct, 0.0, 100.0) / 100.0)


def _qqq_drawdown_avoided(qqq_return: float, exposure_pct: float) -> float:
    if qqq_return >= 0:
        return 0.0
    return -qqq_return * (1.0 - _clamp(exposure_pct, 0.0, 100.0) / 100.0)


def _tqqq_boost_context_cap(entry: Mapping[str, Any], qqq_exposure_pct: float) -> int:
    exposure = _clamp(qqq_exposure_pct, 0.0, 100.0)
    if exposure < 40:
        return 0
    policy_key = str(entry.get("policy_key") or entry.get("qqq_decision") or entry.get("action") or "").upper()
    action = str(entry.get("qqq_decision") or "").upper()
    if action == "GO" or exposure >= 80:
        cap = 33
    elif action in {"GO_SMALL", "TACTICAL_GO_SMALL"} or policy_key in {"GO_SMALL", "TACTICAL_GO_SMALL", "WATCH_REBOUND"}:
        cap = 20
    else:
        cap = 10
    return int(min(cap, exposure))


def _ddm_adjusted_boost_cap(entry: Mapping[str, Any], base_cap: int) -> Tuple[int, Dict[str, Any]]:
    ddm = entry.get("ddm_signal") or {}
    if not ddm.get("enabled"):
        return base_cap, {
            "enabled": False,
            "base_cap_pct": base_cap,
            "adjusted_cap_pct": base_cap,
            "reason": "DDM 데이터가 없어 기존 TQQQ cap을 유지합니다.",
        }

    confidence = _to_float(ddm.get("confidence_pct")) or 0.0
    drift = _to_float(ddm.get("drift")) or 0.0
    agreement = _to_float(ddm.get("agreement_pct")) or 0.0
    multiplier = _to_float(ddm.get("boost_cap_multiplier"))
    if multiplier is None:
        multiplier = 1.0
    adjusted = int(round(base_cap * _clamp(multiplier, 0.0, 1.0)))
    if drift <= 0 or confidence < 55.0 or agreement < 50.0:
        adjusted = 0
        reason = "DDM drift/확신도가 부족해 TQQQ boost를 차단합니다."
    elif adjusted < base_cap:
        reason = "DDM이 우호적이지만 내부 저항이 남아 TQQQ cap을 축소합니다."
    else:
        reason = "DDM drift와 상관 바스켓 합의가 충분해 TQQQ cap을 유지합니다."
    return adjusted, {
        "enabled": True,
        "status": ddm.get("status"),
        "base_cap_pct": base_cap,
        "adjusted_cap_pct": adjusted,
        "confidence_pct": confidence,
        "drift": drift,
        "diffusion": _to_float(ddm.get("diffusion")),
        "evidence": _to_float(ddm.get("evidence")),
        "agreement_pct": agreement,
        "correlated_count": ddm.get("correlated_count"),
        "high_corr_count": ddm.get("high_corr_count"),
        "reason": reason,
    }


def _ddm_buyhold_risk_overlay(
    ddm: Mapping[str, Any],
    *,
    learned_exposure_pct: float,
    swing_action: str = "",
    tactical_action: str = "",
) -> Dict[str, Any]:
    """Use DDM only as a risk-avoidance overlay over QQQ buy-and-hold."""

    learned = _clamp(learned_exposure_pct, 0.0, 100.0)
    if not ddm.get("enabled"):
        return {
            "enabled": False,
            "applied": False,
            "source": "legacy_no_ddm",
            "base_exposure_pct": learned,
            "exposure_pct": learned,
            "multiplier": learned / 100.0 if learned else 0.0,
            "reason": "DDM 데이터가 없어 기존 learned exposure를 사용합니다.",
        }

    status = str(ddm.get("status") or "neutral").lower()
    drift = _to_float(ddm.get("drift")) or 0.0
    diffusion = _to_float(ddm.get("diffusion")) or 0.0
    evidence = _to_float(ddm.get("evidence")) or 0.0
    confidence = _to_float(ddm.get("confidence_pct")) or 50.0
    agreement = _to_float(ddm.get("agreement_pct")) or 50.0
    exposure = float(BUY_HOLD_BASE_EXPOSURE_PCT)
    severity = "none"
    tactical_key = str(tactical_action or "").upper()

    if status == "boost":
        exposure = 100.0
        reason = "DDM drift가 강하고 diffusion 저항이 낮아 QQQ buy-and-hold를 유지합니다."
    elif status == "constructive":
        exposure = 100.0
        reason = "DDM이 우호적이어서 QQQ buy-and-hold 기본 비중을 유지합니다."
    elif status == "block":
        if evidence <= -0.55 or confidence <= 32.0 or (drift <= -0.40 and agreement < 36.0):
            exposure = 0.0
            severity = "hard"
            reason = "DDM 역방향 압력이 극단적이라 QQQ buy-and-hold 비중을 크게 줄입니다."
        elif (
            evidence <= -0.38
            or confidence <= 36.5
            or (drift <= -0.30 and agreement < 38.0)
            or (diffusion >= 2.40 and agreement < 40.0)
        ):
            exposure = 25.0
            severity = "medium"
            reason = "DDM 역방향 압력과 내부 저항이 충분히 커서 QQQ buy-and-hold 비중을 줄입니다."
        else:
            exposure = 100.0
            severity = "soft_watch"
            reason = "DDM soft block은 감축하지 않고 QQQ buy-and-hold를 유지하며 감시합니다."
    else:
        exposure = 100.0
        reason = "DDM이 명확한 위험 회피 경계에 닿지 않아 QQQ buy-and-hold를 유지합니다."

    if (
        status == "block"
        and severity in {"hard", "medium"}
        and tactical_key not in DDM_HIGH_PRECISION_REDUCTION_TACTICALS
    ):
        exposure = 100.0
        severity = f"{severity}_late_stop_watch"
        reason = (
            "DDM 위험은 강하지만 tactical 충돌 신호가 없어 QQQ buy-and-hold를 유지합니다. "
            "순수 STOP 구간의 후행 감축은 낮은 정밀도로 분류합니다."
        )
    if (
        tactical_key == "DELAY_ENTRY"
        and status == "block"
        and severity in {"hard", "medium"}
    ):
        exposure = min(exposure, 40.0)
        severity = severity if severity != "none" else "tactical_delay"

    exposure = _clamp(exposure, 0.0, 100.0)
    return {
        "enabled": True,
        "applied": True,
        "source": "ddm_buyhold_risk_overlay",
        "status": status,
        "severity": severity,
        "base_exposure_pct": BUY_HOLD_BASE_EXPOSURE_PCT,
        "learned_reference_exposure_pct": learned,
        "exposure_pct": exposure,
        "multiplier": exposure / 100.0,
        "confidence_pct": confidence,
        "drift": drift,
        "diffusion": diffusion,
        "evidence": evidence,
        "agreement_pct": agreement,
        "reason": reason,
    }


def _portfolio_return_with_tqqq(
    qqq_return: float,
    tqqq_return: Optional[float],
    qqq_exposure_pct: float,
    tqqq_boost_pct: float,
) -> float:
    boost = _clamp(tqqq_boost_pct, 0.0, qqq_exposure_pct)
    qqq_alloc = _clamp(qqq_exposure_pct - boost, 0.0, 100.0)
    tqqq_ret = qqq_return if tqqq_return is None else tqqq_return
    return qqq_return * (qqq_alloc / 100.0) + tqqq_ret * (boost / 100.0)


def _downside_deviation(values: Sequence[float]) -> Optional[float]:
    downsides = [min(0.0, value) for value in values]
    if not downsides:
        return None
    return math.sqrt(sum(value * value for value in downsides) / len(downsides))


def _adaptive_exposure_policy(
    entries: Sequence[Mapping[str, Any]],
    *,
    hit_threshold_pct: float,
    candidates: Optional[Sequence[int]] = None,
    min_samples: int = 6,
) -> Dict[str, Any]:
    """Choose QQQ exposure per signal bucket from rolling point-in-time results."""

    candidate_values = [
        int(_clamp(float(value), 0.0, 100.0))
        for value in (candidates or ADAPTIVE_EXPOSURE_CANDIDATES)
    ]
    candidate_values = sorted(set(candidate_values))
    buckets: Dict[str, List[float]] = {}
    for entry in entries:
        key = str(entry.get("policy_key") or entry.get("qqq_decision") or entry.get("action") or "UNKNOWN").upper()
        qqq_return = _to_float(entry.get("qqq_return_pct"))
        if qqq_return is None:
            continue
        buckets.setdefault(key, []).append(qqq_return)

    policy: Dict[str, Any] = {}
    for key, returns in sorted(buckets.items()):
        trials: List[Dict[str, Any]] = []
        for exposure in candidate_values:
            decision_utilities = [
                _qqq_decision_utility(qqq_return, exposure)
                for qqq_return in returns
            ]
            strategy_returns = [
                _qqq_strategy_return(qqq_return, exposure)
                for qqq_return in returns
            ]
            missed = [
                _qqq_missed_upside(qqq_return, exposure)
                for qqq_return in returns
            ]
            avoided = [
                _qqq_drawdown_avoided(qqq_return, exposure)
                for qqq_return in returns
            ]
            hit_rate = _hit_rate(decision_utilities, hit_threshold_pct)
            avg_strategy = _mean(strategy_returns)
            avg_missed = _mean(missed)
            avg_avoided = _mean(avoided)
            objective = (
                (avg_strategy or 0.0)
                + ((hit_rate or 0.0) - 50.0) * 0.015
                + (avg_avoided or 0.0) * 0.20
                - (avg_missed or 0.0) * 0.10
            )
            if len(returns) < min_samples:
                objective -= 0.35
            trials.append(
                {
                    "exposure_pct": exposure,
                    "n": len(returns),
                    "hit_rate_pct": hit_rate,
                    "avg_decision_utility_pct": _mean(decision_utilities),
                    "avg_strategy_return_pct": avg_strategy,
                    "avg_missed_upside_pct": avg_missed,
                    "avg_drawdown_avoided_pct": avg_avoided,
                    "objective": objective,
                }
            )
        trials.sort(key=lambda item: float(item.get("objective") or -999.0), reverse=True)
        selected = trials[0] if trials else {}
        policy[key] = {
            "selected_exposure_pct": selected.get("exposure_pct"),
            "n": len(returns),
            "confidence": "ok" if len(returns) >= min_samples else "thin",
            "selected": selected,
            "candidates": trials[:6],
        }

    adaptive_hits: List[float] = []
    adaptive_utilities: List[float] = []
    adaptive_strategy_returns: List[float] = []
    adaptive_vs_buy_hold: List[float] = []
    adaptive_missed: List[float] = []
    adaptive_avoided: List[float] = []
    for entry in entries:
        qqq_return = _to_float(entry.get("qqq_return_pct"))
        if qqq_return is None:
            continue
        key = str(entry.get("policy_key") or entry.get("qqq_decision") or entry.get("action") or "UNKNOWN").upper()
        bucket = policy.get(key) or {}
        selected = bucket.get("selected") or {}
        exposure = _to_float(bucket.get("selected_exposure_pct"))
        if exposure is None:
            exposure = _to_float(selected.get("exposure_pct"))
        if exposure is None or bucket.get("confidence") == "thin":
            exposure = _to_float(entry.get("qqq_exposure_pct")) or 0.0
        exposure = min(
            exposure,
            _adaptive_context_exposure_cap(key, str(entry.get("action") or "")),
        )
        decision_utility = _qqq_decision_utility(qqq_return, exposure)
        strategy_return = _qqq_strategy_return(qqq_return, exposure)
        adaptive_hits.append(1.0 if decision_utility > hit_threshold_pct else 0.0)
        adaptive_utilities.append(decision_utility)
        adaptive_strategy_returns.append(strategy_return)
        adaptive_vs_buy_hold.append(strategy_return - qqq_return)
        adaptive_missed.append(_qqq_missed_upside(qqq_return, exposure))
        adaptive_avoided.append(_qqq_drawdown_avoided(qqq_return, exposure))

    return {
        "enabled": True,
        "min_samples": min_samples,
        "candidate_exposures_pct": candidate_values,
        "buckets": policy,
        "aggregate": {
            "n": len(adaptive_utilities),
            "hit_rate_pct": (_mean(adaptive_hits) or 0.0) * 100.0 if adaptive_hits else None,
            "avg_decision_utility_pct": _mean(adaptive_utilities),
            "avg_strategy_return_pct": _mean(adaptive_strategy_returns),
            "avg_vs_buy_hold_pct": _mean(adaptive_vs_buy_hold),
            "avg_missed_upside_pct": _mean(adaptive_missed),
            "avg_drawdown_avoided_pct": _mean(adaptive_avoided),
        },
        "note": "매일 재계산한 과거 QQQ forward 성과로 신호 버킷별 권장 exposure를 고릅니다.",
    }


def _adaptive_policy_exposure_for_entry(
    entry: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    fallback_source: str,
) -> Dict[str, Any]:
    key = str(entry.get("policy_key") or entry.get("qqq_decision") or entry.get("action") or "UNKNOWN").upper()
    baseline = _to_float(entry.get("qqq_exposure_pct")) or 0.0
    bucket = (policy.get("buckets") or {}).get(key) or {}
    selected = bucket.get("selected") or {}
    raw_exposure = _to_float(bucket.get("selected_exposure_pct"))
    if raw_exposure is None:
        raw_exposure = _to_float(selected.get("exposure_pct"))

    source = fallback_source
    exposure = baseline
    if raw_exposure is not None and bucket.get("confidence") != "thin":
        exposure = raw_exposure
        source = "adaptive"
    cap = _adaptive_context_exposure_cap(key, str(entry.get("action") or ""))
    exposure = min(_clamp(exposure, 0.0, 100.0), cap)
    return {
        "policy_key": key,
        "exposure_pct": exposure,
        "raw_exposure_pct": raw_exposure,
        "context_cap_pct": cap,
        "source": source,
        "bucket_n": bucket.get("n"),
        "bucket_confidence": bucket.get("confidence"),
    }


def _feature_stats(
    entries: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for name in feature_names:
        values = [
            float(value)
            for entry in entries
            for value in [_to_float((entry.get("features") or {}).get(name))]
            if value is not None
        ]
        if len(values) < 3:
            continue
        mean_value = sum(values) / len(values)
        stdev = statistics.pstdev(values)
        if stdev > 0:
            stats[name] = (mean_value, stdev)
    return stats


def _feature_distance(
    current_features: Mapping[str, Any],
    prior_features: Mapping[str, Any],
    stats: Mapping[str, Tuple[float, float]],
    feature_names: Sequence[str],
) -> Tuple[Optional[float], int]:
    total = 0.0
    count = 0
    for name in feature_names:
        current_value = _to_float(current_features.get(name))
        prior_value = _to_float(prior_features.get(name))
        stat = stats.get(name)
        if current_value is None or prior_value is None or stat is None:
            continue
        _, scale = stat
        if scale <= 0:
            continue
        diff = (current_value - prior_value) / scale
        total += diff * diff
        count += 1
    if count < 5:
        return None, count
    return math.sqrt(total / count), count


def _similarity_adaptive_exposure_for_entry(
    entry: Mapping[str, Any],
    prior_entries: Sequence[Mapping[str, Any]],
    *,
    hit_threshold_pct: float,
    min_neighbors: int = 12,
    max_neighbors: int = 35,
) -> Dict[str, Any]:
    current_features = entry.get("features") or {}
    if len(prior_entries) < min_neighbors or not current_features:
        return {
            "source": "insufficient_history",
            "exposure_pct": _to_float(entry.get("qqq_exposure_pct")) or 0.0,
            "neighbors": 0,
        }

    policy_key = str(entry.get("policy_key") or entry.get("qqq_decision") or entry.get("action") or "UNKNOWN").upper()
    stats = _feature_stats(prior_entries, SIMILARITY_FEATURES)
    scored: List[Tuple[float, Mapping[str, Any]]] = []
    for prior in prior_entries:
        prior_return = _to_float(prior.get("qqq_return_pct"))
        if prior_return is None:
            continue
        distance, overlap = _feature_distance(
            current_features,
            prior.get("features") or {},
            stats,
            SIMILARITY_FEATURES,
        )
        if distance is None:
            continue
        prior_key = str(prior.get("policy_key") or prior.get("qqq_decision") or prior.get("action") or "UNKNOWN").upper()
        if prior_key != policy_key:
            distance += 0.30
        scored.append((distance, prior))

    scored.sort(key=lambda item: item[0])
    if len(scored) < min_neighbors:
        return {
            "source": "insufficient_neighbors",
            "policy_key": policy_key,
            "exposure_pct": _to_float(entry.get("qqq_exposure_pct")) or 0.0,
            "neighbors": len(scored),
        }

    neighbors = [item[1] for item in scored[: min(max_neighbors, len(scored))]]
    cap = _adaptive_context_exposure_cap(policy_key, str(entry.get("action") or ""))
    candidates = [
        exposure
        for exposure in ADAPTIVE_EXPOSURE_CANDIDATES
        if exposure <= cap
    ] or [cap]
    trials: List[Dict[str, Any]] = []
    neighbor_returns = [
        float(_to_float(item.get("qqq_return_pct")) or 0.0)
        for item in neighbors
    ]
    for exposure in candidates:
        utilities = [
            _qqq_decision_utility(qqq_return, exposure)
            for qqq_return in neighbor_returns
        ]
        strategy_returns = [
            _qqq_strategy_return(qqq_return, exposure)
            for qqq_return in neighbor_returns
        ]
        missed = [
            _qqq_missed_upside(qqq_return, exposure)
            for qqq_return in neighbor_returns
        ]
        avoided = [
            _qqq_drawdown_avoided(qqq_return, exposure)
            for qqq_return in neighbor_returns
        ]
        hit_rate = _hit_rate(utilities, hit_threshold_pct)
        avg_strategy = _mean(strategy_returns)
        avg_avoided = _mean(avoided)
        avg_missed = _mean(missed)
        objective = (
            (avg_strategy or 0.0)
            + ((hit_rate or 0.0) - 50.0) * 0.015
            + (avg_avoided or 0.0) * 0.35
            - (avg_missed or 0.0) * 0.05
        )
        trials.append(
            {
                "exposure_pct": exposure,
                "hit_rate_pct": hit_rate,
                "avg_decision_utility_pct": _mean(utilities),
                "avg_strategy_return_pct": avg_strategy,
                "avg_missed_upside_pct": avg_missed,
                "avg_drawdown_avoided_pct": avg_avoided,
                "objective": objective,
            }
        )

    trials.sort(key=lambda item: float(item.get("objective") or -999.0), reverse=True)
    selected = trials[0] if trials else {}
    return {
        "source": "similarity",
        "policy_key": policy_key,
        "exposure_pct": selected.get("exposure_pct"),
        "raw_exposure_pct": selected.get("exposure_pct"),
        "context_cap_pct": cap,
        "neighbors": len(neighbors),
        "avg_neighbor_distance": _mean([item[0] for item in scored[: len(neighbors)]]),
        "selected": selected,
        "candidates": trials[:6],
        "features": [name for name in SIMILARITY_FEATURES if name in stats],
    }


def _similarity_tqqq_boost_for_entry(
    entry: Mapping[str, Any],
    prior_entries: Sequence[Mapping[str, Any]],
    *,
    qqq_exposure_pct: float,
    hit_threshold_pct: float,
    min_neighbors: int = 12,
    max_neighbors: int = 35,
) -> Dict[str, Any]:
    """Choose a TQQQ sleeve from prior similar ETF flow/price patterns."""

    base_cap = _tqqq_boost_context_cap(entry, qqq_exposure_pct)
    cap, ddm_gate = _ddm_adjusted_boost_cap(entry, base_cap)
    if cap <= 0:
        return {
            "enabled": True,
            "source": "ddm_block" if base_cap > 0 else "low_qqq_exposure",
            "qqq_exposure_pct": qqq_exposure_pct,
            "tqqq_boost_pct": 0,
            "qqq_alloc_pct": int(round(_clamp(qqq_exposure_pct, 0.0, 100.0))),
            "effective_beta": _clamp(qqq_exposure_pct, 0.0, 100.0) / 100.0,
            "context_cap_pct": cap,
            "base_context_cap_pct": base_cap,
            "ddm_gate": ddm_gate,
            "ddm_signal": entry.get("ddm_signal") or {},
            "reason": ddm_gate.get("reason") or "QQQ exposure가 낮아 TQQQ sleeve를 열지 않습니다.",
        }

    current_features = entry.get("features") or {}
    if len(prior_entries) < min_neighbors or not current_features:
        return {
            "enabled": True,
            "source": "insufficient_history",
            "tqqq_boost_pct": 0,
            "context_cap_pct": cap,
            "base_context_cap_pct": base_cap,
            "ddm_gate": ddm_gate,
            "ddm_signal": entry.get("ddm_signal") or {},
            "neighbors": 0,
        }

    policy_key = str(entry.get("policy_key") or entry.get("qqq_decision") or entry.get("action") or "UNKNOWN").upper()
    stats = _feature_stats(prior_entries, SIMILARITY_FEATURES)
    scored: List[Tuple[float, Mapping[str, Any]]] = []
    for prior in prior_entries:
        if _to_float(prior.get("qqq_return_pct")) is None:
            continue
        if _to_float(prior.get("tqqq_return_pct")) is None:
            continue
        distance, _ = _feature_distance(
            current_features,
            prior.get("features") or {},
            stats,
            SIMILARITY_FEATURES,
        )
        if distance is None:
            continue
        prior_key = str(prior.get("policy_key") or prior.get("qqq_decision") or prior.get("action") or "UNKNOWN").upper()
        if prior_key != policy_key:
            distance += 0.30
        current_ddm_status = str((entry.get("ddm_signal") or {}).get("status") or "")
        prior_ddm_status = str((prior.get("ddm_signal") or {}).get("status") or "")
        if current_ddm_status and prior_ddm_status and current_ddm_status != prior_ddm_status:
            distance += 0.20
        scored.append((distance, prior))

    scored.sort(key=lambda item: item[0])
    if len(scored) < min_neighbors:
        return {
            "enabled": True,
            "source": "insufficient_neighbors",
            "policy_key": policy_key,
            "tqqq_boost_pct": 0,
            "context_cap_pct": cap,
            "base_context_cap_pct": base_cap,
            "ddm_gate": ddm_gate,
            "ddm_signal": entry.get("ddm_signal") or {},
            "neighbors": len(scored),
        }

    neighbors = [item[1] for item in scored[: min(max_neighbors, len(scored))]]
    candidates = [value for value in TQQQ_BOOST_CANDIDATES if value <= cap]
    if 0 not in candidates:
        candidates.insert(0, 0)
    trials: List[Dict[str, Any]] = []
    for boost in sorted(set(candidates)):
        portfolio_returns: List[float] = []
        base_returns: List[float] = []
        buy_hold_excesses: List[float] = []
        boost_excesses: List[float] = []
        for prior in neighbors:
            qqq_return = float(_to_float(prior.get("qqq_return_pct")) or 0.0)
            tqqq_return = _to_float(prior.get("tqqq_return_pct"))
            portfolio_return = _portfolio_return_with_tqqq(
                qqq_return,
                tqqq_return,
                qqq_exposure_pct,
                boost,
            )
            base_return = _qqq_strategy_return(qqq_return, qqq_exposure_pct)
            portfolio_returns.append(portfolio_return)
            base_returns.append(base_return)
            buy_hold_excesses.append(portfolio_return - qqq_return)
            boost_excesses.append(portfolio_return - base_return)

        hit_rate = _hit_rate(boost_excesses, hit_threshold_pct / 2.0)
        avg_return = _mean(portfolio_returns)
        avg_boost_excess = _mean(boost_excesses)
        avg_buy_hold_excess = _mean(buy_hold_excesses)
        downside = _downside_deviation(portfolio_returns)
        worst_return = min(portfolio_returns) if portfolio_returns else None
        objective = (
            (avg_return or 0.0)
            + (avg_boost_excess or 0.0) * 0.75
            + ((hit_rate or 0.0) - 50.0) * 0.01
            - (downside or 0.0) * 0.60
            - max(0.0, -(worst_return or 0.0)) * 0.08
        )
        trials.append(
            {
                "tqqq_boost_pct": boost,
                "effective_beta": (qqq_exposure_pct - boost + 3 * boost) / 100.0,
                "hit_rate_pct": hit_rate,
                "avg_portfolio_return_pct": avg_return,
                "avg_boost_excess_pct": avg_boost_excess,
                "avg_vs_qqq_buy_hold_pct": avg_buy_hold_excess,
                "downside_deviation_pct": downside,
                "worst_return_pct": worst_return,
                "objective": objective,
            }
        )

    trials.sort(key=lambda item: float(item.get("objective") or -999.0), reverse=True)
    selected = trials[0] if trials else {}
    selected_boost = int(selected.get("tqqq_boost_pct") or 0)
    return {
        "enabled": True,
        "source": "similarity",
        "policy_key": policy_key,
        "qqq_exposure_pct": qqq_exposure_pct,
        "tqqq_boost_pct": selected_boost,
        "qqq_alloc_pct": int(round(max(0.0, qqq_exposure_pct - selected_boost))),
        "effective_beta": selected.get("effective_beta"),
        "context_cap_pct": cap,
        "base_context_cap_pct": base_cap,
        "ddm_gate": ddm_gate,
        "ddm_signal": entry.get("ddm_signal") or {},
        "neighbors": len(neighbors),
        "avg_neighbor_distance": _mean([item[0] for item in scored[: len(neighbors)]]),
        "selected": selected,
        "candidates": trials[:6],
        "note": "QQQ exposure 안에서 일부를 TQQQ로 치환하는 boost sleeve입니다.",
    }


def _walk_forward_adaptive_backtest(
    entries: Sequence[Mapping[str, Any]],
    *,
    hit_threshold_pct: float,
    warmup_signals: int = 20,
    min_bucket_samples: int = 6,
) -> Dict[str, Any]:
    """Evaluate daily re-learning without lookahead.

    For each signal date, only prior signal outcomes are used to choose the
    adaptive exposure. The current signal's forward QQQ return is then scored
    out of sample.
    """

    ordered = sorted(entries, key=lambda item: str(item.get("date") or ""))
    prior: List[Mapping[str, Any]] = []
    evaluated_rows: List[Dict[str, Any]] = []
    hits: List[float] = []
    utilities: List[float] = []
    strategy_returns: List[float] = []
    vs_buy_hold: List[float] = []
    missed: List[float] = []
    avoided: List[float] = []
    boosted_hits: List[float] = []
    boosted_returns: List[float] = []
    boosted_vs_buy_hold: List[float] = []
    boosted_vs_qqq_strategy: List[float] = []
    active_boost_excesses: List[float] = []
    boost_values: List[float] = []
    risk_reduction_hits: List[float] = []
    risk_reduction_excesses: List[float] = []
    risk_reduction_missed: List[float] = []
    risk_reduction_avoided: List[float] = []
    risk_reduction_breakdown: Dict[str, Dict[str, float]] = {}
    ddm_confidences: List[float] = []
    ddm_evidences: List[float] = []
    ddm_overlay_count = 0
    learned_count = 0
    similarity_count = 0
    boost_count = 0

    for entry in ordered:
        qqq_return = _to_float(entry.get("qqq_return_pct"))
        if qqq_return is None:
            prior.append(entry)
            continue
        ddm_signal = entry.get("ddm_signal") or {}
        if ddm_signal.get("enabled"):
            ddm_confidence = _to_float(ddm_signal.get("confidence_pct"))
            ddm_evidence = _to_float(ddm_signal.get("evidence"))
            if ddm_confidence is not None:
                ddm_confidences.append(ddm_confidence)
            if ddm_evidence is not None:
                ddm_evidences.append(ddm_evidence)
        if len(prior) >= warmup_signals:
            exposure_info = _similarity_adaptive_exposure_for_entry(
                entry,
                prior,
                hit_threshold_pct=hit_threshold_pct,
            )
            if exposure_info.get("source") == "similarity":
                learned_count += 1
                similarity_count += 1
            else:
                policy = _adaptive_exposure_policy(
                    prior,
                    hit_threshold_pct=hit_threshold_pct,
                    min_samples=min_bucket_samples,
                )
                exposure_info = _adaptive_policy_exposure_for_entry(
                    entry,
                    policy,
                    fallback_source="thin_bucket",
                )
                if exposure_info.get("source") == "adaptive":
                    learned_count += 1
        else:
            exposure_info = _adaptive_policy_exposure_for_entry(
                entry,
                {"buckets": {}},
                fallback_source="warmup",
            )

        learned_exposure = float(exposure_info.get("exposure_pct") or 0.0)
        ddm_overlay = _ddm_buyhold_risk_overlay(
            ddm_signal,
            learned_exposure_pct=learned_exposure,
            swing_action=str(entry.get("action") or ""),
            tactical_action=str(entry.get("tactical_action") or ""),
        )
        if ddm_overlay.get("applied"):
            ddm_overlay_count += 1
            exposure = float(ddm_overlay.get("exposure_pct") or 0.0)
        else:
            exposure = learned_exposure
        boost_info = (
            _similarity_tqqq_boost_for_entry(
                entry,
                prior,
                qqq_exposure_pct=exposure,
                hit_threshold_pct=hit_threshold_pct,
            )
            if len(prior) >= warmup_signals
            else {
                "enabled": True,
                "source": "warmup",
                "tqqq_boost_pct": 0,
                "context_cap_pct": 0,
            }
        )
        boost_pct = float(_to_float(boost_info.get("tqqq_boost_pct")) or 0.0)
        decision_utility = _qqq_decision_utility(qqq_return, exposure)
        strategy_return = _qqq_strategy_return(qqq_return, exposure)
        tqqq_return = _to_float(entry.get("tqqq_return_pct"))
        boosted_return = _portfolio_return_with_tqqq(
            qqq_return,
            tqqq_return,
            exposure,
            boost_pct,
        )
        boost_excess = boosted_return - strategy_return
        hit = decision_utility > hit_threshold_pct
        risk_reduction_excess = strategy_return - qqq_return
        risk_reduction_hit: Optional[bool] = None
        hits.append(1.0 if hit else 0.0)
        utilities.append(decision_utility)
        strategy_returns.append(strategy_return)
        vs_buy_hold.append(strategy_return - qqq_return)
        boosted_returns.append(boosted_return)
        boosted_vs_buy_hold.append(boosted_return - qqq_return)
        boosted_vs_qqq_strategy.append(boost_excess)
        boost_values.append(boost_pct)
        if boost_pct > 0:
            boost_count += 1
            boosted_hits.append(1.0 if boost_excess > hit_threshold_pct / 2.0 else 0.0)
            active_boost_excesses.append(boost_excess)
        missed_upside = _qqq_missed_upside(qqq_return, exposure)
        drawdown_avoided = _qqq_drawdown_avoided(qqq_return, exposure)
        missed.append(missed_upside)
        avoided.append(drawdown_avoided)
        if exposure < BUY_HOLD_BASE_EXPOSURE_PCT:
            risk_reduction_hit = risk_reduction_excess > hit_threshold_pct / 2.0
            risk_reduction_hits.append(1.0 if risk_reduction_hit else 0.0)
            risk_reduction_excesses.append(risk_reduction_excess)
            risk_reduction_missed.append(missed_upside)
            risk_reduction_avoided.append(drawdown_avoided)
            breakdown_key = "{}|{}|{}".format(
                exposure_info.get("policy_key") or "unknown",
                ddm_overlay.get("status") or "unknown",
                ddm_overlay.get("severity") or "unknown",
            )
            bucket = risk_reduction_breakdown.setdefault(
                breakdown_key,
                {
                    "signals": 0.0,
                    "hits": 0.0,
                    "excess_sum": 0.0,
                    "missed_sum": 0.0,
                    "avoided_sum": 0.0,
                },
            )
            bucket["signals"] += 1.0
            bucket["hits"] += 1.0 if risk_reduction_hit else 0.0
            bucket["excess_sum"] += risk_reduction_excess
            bucket["missed_sum"] += missed_upside
            bucket["avoided_sum"] += drawdown_avoided
        evaluated_rows.append(
            {
                "date": entry.get("date"),
                "policy_key": exposure_info.get("policy_key"),
                "source": exposure_info.get("source"),
                "exposure_pct": exposure,
                "learned_exposure_pct": learned_exposure,
                "ddm_overlay_source": ddm_overlay.get("source"),
                "ddm_overlay_status": ddm_overlay.get("status"),
                "ddm_overlay_severity": ddm_overlay.get("severity"),
                "raw_exposure_pct": exposure_info.get("raw_exposure_pct"),
                "context_cap_pct": exposure_info.get("context_cap_pct"),
                "bucket_n": exposure_info.get("bucket_n"),
                "neighbors": exposure_info.get("neighbors"),
                "avg_neighbor_distance": exposure_info.get("avg_neighbor_distance"),
                "qqq_return_pct": qqq_return,
                "tqqq_return_pct": tqqq_return,
                "decision_utility_pct": decision_utility,
                "strategy_return_pct": strategy_return,
                "tqqq_boost_pct": boost_pct,
                "boost_source": boost_info.get("source"),
                "boost_neighbors": boost_info.get("neighbors"),
                "ddm_status": ddm_signal.get("status"),
                "ddm_confidence_pct": ddm_signal.get("confidence_pct"),
                "ddm_evidence": ddm_signal.get("evidence"),
                "boosted_strategy_return_pct": boosted_return,
                "boost_excess_pct": boost_excess,
                "vs_buy_hold_pct": strategy_return - qqq_return,
                "risk_reduction_excess_pct": risk_reduction_excess,
                "risk_reduction_hit": risk_reduction_hit,
                "boosted_vs_buy_hold_pct": boosted_return - qqq_return,
                "missed_upside_pct": missed_upside,
                "drawdown_avoided_pct": drawdown_avoided,
                "hit": hit,
            }
        )
        prior.append(entry)

    return {
        "enabled": True,
        "warmup_signals": warmup_signals,
        "min_bucket_samples": min_bucket_samples,
        "evaluated_signals": len(evaluated_rows),
        "learned_signals": learned_count,
        "similarity_signals": similarity_count,
        "ddm_overlay_signals": ddm_overlay_count,
        "boost_signals": boost_count,
        "risk_reduction_signals": len(risk_reduction_hits),
        "hit_rate_pct": (_mean(hits) or 0.0) * 100.0 if hits else None,
        "avg_decision_utility_pct": _mean(utilities),
        "avg_strategy_return_pct": _mean(strategy_returns),
        "avg_vs_buy_hold_pct": _mean(vs_buy_hold),
        "avg_missed_upside_pct": _mean(missed),
        "avg_drawdown_avoided_pct": _mean(avoided),
        "risk_reduction_hit_rate_pct": (
            (_mean(risk_reduction_hits) or 0.0) * 100.0
            if risk_reduction_hits
            else None
        ),
        "avg_risk_reduction_excess_pct": _mean(risk_reduction_excesses),
        "avg_risk_reduction_missed_upside_pct": _mean(risk_reduction_missed),
        "avg_risk_reduction_drawdown_avoided_pct": _mean(risk_reduction_avoided),
        "risk_reduction_breakdown": [
            {
                "bucket": bucket_key,
                "signals": int(values["signals"]),
                "hit_rate_pct": (
                    values["hits"] / values["signals"] * 100.0
                    if values["signals"]
                    else None
                ),
                "avg_excess_pct": (
                    values["excess_sum"] / values["signals"]
                    if values["signals"]
                    else None
                ),
                "avg_missed_upside_pct": (
                    values["missed_sum"] / values["signals"]
                    if values["signals"]
                    else None
                ),
                "avg_drawdown_avoided_pct": (
                    values["avoided_sum"] / values["signals"]
                    if values["signals"]
                    else None
                ),
            }
            for bucket_key, values in sorted(
                risk_reduction_breakdown.items(),
                key=lambda item: (item[1]["hits"] / item[1]["signals"], item[1]["signals"]),
                reverse=True,
            )
        ],
        "boost_hit_rate_pct": (_mean(boosted_hits) or 0.0) * 100.0 if boosted_hits else None,
        "avg_tqqq_boost_pct": _mean(boost_values),
        "avg_boosted_strategy_return_pct": _mean(boosted_returns),
        "avg_boosted_vs_buy_hold_pct": _mean(boosted_vs_buy_hold),
        "avg_boost_excess_vs_qqq_strategy_pct": _mean(boosted_vs_qqq_strategy),
        "avg_active_boost_excess_pct": _mean(active_boost_excesses),
        "boosted_downside_deviation_pct": _downside_deviation(boosted_returns),
        "ddm_enabled_signals": len(ddm_confidences),
        "avg_ddm_confidence_pct": _mean(ddm_confidences),
        "avg_ddm_evidence": _mean(ddm_evidences),
        "recent": evaluated_rows[-8:],
        "note": "각 날짜 이전 신호만으로 ETF flow/price 유사 패턴 policy를 재학습한 walk-forward 결과입니다.",
    }


def build_qqq_decision(
    *,
    gostop: Mapping[str, Any],
    tactical: Optional[Mapping[str, Any]] = None,
    nowcast: Optional[Mapping[str, Any]] = None,
    auto_tune: Optional[Mapping[str, Any]] = None,
    ddm_signal: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the final QQQ go/stop decision and target exposure.

    ETF rotation, flow/NAV, credit, and NowCast are inputs. The primary output
    is whether QQQ exposure should be opened, sized down, or avoided.
    """

    tactical = tactical or {}
    nowcast = nowcast or {}
    auto_tune = auto_tune or {}
    ddm_signal = ddm_signal or {}
    selected = auto_tune.get("selected") or {}

    swing_action = str(gostop.get("action") or "UNKNOWN").upper()
    swing_exposure = _gostop_exposure_pct(gostop)
    tactical_action = str(tactical.get("action") or "").upper()
    tactical_exposure = _tactical_exposure_pct(tactical) if tactical else 0.0
    policy_key = _qqq_policy_key(swing_action, tactical_action)

    tuned_cap = _to_float(selected.get("tuned_max_new_risk_pct"))
    tuned_gate_open = selected.get("tuned_gate_open")
    if tuned_gate_open is False and tuned_cap is None:
        tuned_cap = 0.0
    if tuned_cap is not None and swing_action in {"GO", "GO_SMALL"}:
        swing_exposure = min(swing_exposure, _clamp(tuned_cap, 0.0, 100.0))

    if tactical.get("enabled"):
        exposure = tactical_exposure
        if tactical_action in {"GO", "GO_SMALL"}:
            exposure = min(exposure, swing_exposure) if tuned_cap is not None else exposure
        elif tactical_action in {"TACTICAL_GO_SMALL", "WATCH_REBOUND"}:
            # A strong live reversal can allow a small probe even when the
            # slower swing gate remains closed.
            exposure = max(exposure, 0.0)
        elif tactical_action == "DELAY_ENTRY":
            exposure = 0.0
    else:
        exposure = swing_exposure

    adaptive_policy = selected.get("adaptive_exposure_policy") or auto_tune.get("adaptive_exposure_policy") or {}
    adaptive_bucket = (adaptive_policy.get("buckets") or {}).get(policy_key) or {}
    adaptive_selected = adaptive_bucket.get("selected") or {}
    adaptive_exposure = _to_float(
        adaptive_bucket.get("selected_exposure_pct")
        if adaptive_bucket.get("selected_exposure_pct") is not None
        else adaptive_selected.get("exposure_pct")
    )
    adaptive_raw_exposure = adaptive_exposure
    adaptive_cap = _adaptive_context_exposure_cap(policy_key, swing_action)
    if adaptive_exposure is not None:
        adaptive_exposure = min(adaptive_exposure, adaptive_cap)
    adaptive_applied = False
    if adaptive_exposure is not None and adaptive_bucket.get("confidence") != "thin":
        exposure = adaptive_exposure
        adaptive_applied = True

    pattern_decision = selected.get("pattern_adaptive_decision") or auto_tune.get("pattern_adaptive_decision") or {}
    pattern_exposure = _to_float(pattern_decision.get("exposure_pct"))
    pattern_applied = False
    if pattern_exposure is not None and pattern_decision.get("source") == "similarity":
        exposure = pattern_exposure
        pattern_applied = True

    learned_reference_exposure = _clamp(exposure, 0.0, 100.0)
    ddm_overlay = _ddm_buyhold_risk_overlay(
        ddm_signal,
        learned_exposure_pct=learned_reference_exposure,
        swing_action=swing_action,
        tactical_action=tactical_action,
    )
    ddm_overlay_applied = bool(ddm_overlay.get("applied"))
    if ddm_overlay_applied:
        exposure = float(ddm_overlay.get("exposure_pct") or 0.0)
    exposure = _clamp(exposure, 0.0, 100.0)
    if exposure >= 75:
        action = "GO"
    elif exposure >= 25:
        action = "TACTICAL_GO_SMALL" if tactical_action == "TACTICAL_GO_SMALL" else "GO_SMALL"
    elif exposure > 0:
        action = "WATCH_REBOUND"
    elif tactical_action == "WATCH_REBOUND":
        action = "WATCH_REBOUND"
    elif swing_action == "WAIT":
        action = "WAIT"
    else:
        action = "STOP"

    reasons = [
        f"Swing gate는 {swing_action}이며 기본 QQQ exposure는 {int(round(swing_exposure))}%입니다."
    ]
    blocks: List[str] = []
    if tactical.get("enabled"):
        reasons.append(
            f"Tactical overlay는 {tactical_action or 'N/A'}이며 탐색 exposure는 "
            f"{int(round(tactical_exposure))}%입니다."
        )
    if nowcast.get("enabled"):
        reasons.append(
            f"NowCast score는 {nowcast.get('score', 'N/A')}/100, "
            f"risk ETF 당일 평균은 {_fmt_pct(nowcast.get('risk_day_avg_pct'))}입니다."
        )
    if auto_tune.get("enabled"):
        if tuned_gate_open is False:
            blocks.append("학습 기준 swing entry gate는 아직 닫혀 있습니다.")
        elif tuned_gate_open is True:
            reasons.append("학습 기준 swing entry gate가 열려 있습니다.")
    if adaptive_bucket:
        learned_exposure = adaptive_bucket.get("selected_exposure_pct")
        learned_n = adaptive_bucket.get("n")
        if adaptive_applied:
            reasons.append(
                f"Adaptive policy가 `{policy_key}` 버킷 과거 {learned_n}개 신호 기준 "
                f"QQQ exposure {learned_exposure}%를 선택했습니다."
            )
            if adaptive_raw_exposure is not None and adaptive_raw_exposure > adaptive_cap:
                blocks.append(
                    f"다만 `{swing_action}` 컨텍스트 리스크 캡으로 adaptive exposure를 "
                    f"{int(round(adaptive_raw_exposure))}% -> {adaptive_cap}%로 제한했습니다."
                )
        else:
            blocks.append(
                f"Adaptive policy의 `{policy_key}` 버킷 표본이 부족해 기본 exposure를 유지합니다."
            )
    if pattern_decision:
        if pattern_applied:
            reasons.append(
                "Similarity policy가 ETF flow/price 유사 과거 "
                f"{pattern_decision.get('neighbors')}개 패턴 기준 QQQ exposure "
                f"{int(round(pattern_exposure or 0))}%를 선택했습니다."
            )
        else:
            blocks.append(
                f"Similarity policy는 `{pattern_decision.get('source', 'unknown')}` 상태라 "
                "기본/adaptive bucket exposure를 유지합니다."
            )
    if ddm_overlay_applied:
        if exposure >= BUY_HOLD_BASE_EXPOSURE_PCT:
            reasons.append(
                "DDM risk overlay가 QQQ buy-and-hold 기본 비중 100%를 유지했습니다."
            )
        else:
            blocks.append(
                "DDM risk overlay가 QQQ buy-and-hold 100%를 "
                f"{int(round(exposure))}%로 축소했습니다 "
                f"(confidence {_fmt_pct(ddm_overlay.get('confidence_pct'))}, "
                f"drift {_fmt_num(ddm_overlay.get('drift'), 3)}, "
                f"diffusion {_fmt_num(ddm_overlay.get('diffusion'), 3)})."
            )
    for item in gostop.get("blocks") or []:
        blocks.append(str(item))
    for item in tactical.get("blocks") or []:
        blocks.append(str(item))

    if action == "GO":
        headline = "QQQ 신규/유지 진입이 가능합니다."
    elif action in {"GO_SMALL", "TACTICAL_GO_SMALL"}:
        headline = "QQQ 부분 진입만 허용합니다."
    elif action == "WATCH_REBOUND":
        headline = "QQQ 반등은 감지됐지만 swing 확정 전이라 소액 탐색 또는 관찰 단계입니다."
    elif action == "WAIT":
        headline = "QQQ 방향 확인 전까지 신규 진입을 보류합니다."
    else:
        headline = "QQQ 신규 진입을 중단합니다."

    return {
        "enabled": True,
        "target": "QQQ",
        "action": action,
        "recommended_exposure_pct": int(round(exposure)),
        "policy_key": policy_key,
        "base_policy": "QQQ_BUY_AND_HOLD_WITH_DDM_RISK_AVOIDANCE",
        "buy_hold_base_exposure_pct": BUY_HOLD_BASE_EXPOSURE_PCT,
        "learned_reference_exposure_pct": int(round(learned_reference_exposure)),
        "ddm_risk_overlay_applied": ddm_overlay_applied,
        "ddm_risk_overlay": ddm_overlay,
        "adaptive_policy_applied": adaptive_applied,
        "adaptive_raw_exposure_pct": int(round(adaptive_raw_exposure)) if adaptive_raw_exposure is not None else None,
        "adaptive_context_cap_pct": adaptive_cap,
        "adaptive_policy_bucket": adaptive_bucket or None,
        "pattern_policy_applied": pattern_applied,
        "pattern_policy_decision": pattern_decision or None,
        "swing_action": swing_action,
        "swing_exposure_pct": int(round(swing_exposure)),
        "tactical_action": tactical_action or None,
        "tactical_exposure_pct": int(round(tactical_exposure)),
        "headline": headline,
        "reasons": reasons[:6],
        "blocks": blocks[:8],
        "evaluation_rule": (
            "exposure>0이면 QQQ forward return이 양수일 때 HIT, "
            "exposure=0이면 QQQ forward return이 음수일 때 HIT"
        ),
    }


def _tactical_utility(qqq_return: float, tactical: Mapping[str, Any]) -> float:
    """Evaluate tactical exposure against an all-in/all-out benchmark.

    100% exposure receives full QQQ forward return, 0% receives the inverse
    stop utility, and partial probe states scale the return by exposure.
    """

    exposure = _tactical_exposure_pct(tactical) / 100.0
    if exposure <= 0:
        return -qqq_return
    return qqq_return * exposure


def _objective(
    avg_value: Optional[float],
    hit_rate_pct: Optional[float],
    n: int,
) -> float:
    score = (avg_value or 0.0) + ((hit_rate_pct or 0.0) - 50.0) * 0.02
    if n < 15:
        score -= 0.50
    elif n < 30:
        score -= 0.20
    return score


def _forward_return_from_prices(
    prices: Sequence[PricePoint],
    asof_date: str,
    horizon_bars: int,
) -> Optional[float]:
    clean = [point for point in prices if point.close is not None]
    if not clean:
        return None
    asof = _parse_date(asof_date)
    if not asof:
        return None
    idx = None
    for pos, point in enumerate(clean):
        point_date = _parse_date(point.date)
        if point_date and point_date <= asof:
            idx = pos
    if idx is None:
        return None
    future_idx = idx + max(1, int(horizon_bars))
    if future_idx >= len(clean):
        return None
    return _pct_change(clean[future_idx].close, clean[idx].close)


def _slice_price_map(
    price_map: Mapping[str, Sequence[PricePoint]], asof_date: str
) -> Dict[str, List[PricePoint]]:
    asof = _parse_date(asof_date) or dt.date.min
    return {
        symbol: [
            point
            for point in points
            if (_parse_date(point.date) or dt.date.min) <= asof
        ]
        for symbol, points in price_map.items()
    }


def _slice_flow_map(
    flow_map: Mapping[str, Sequence[FlowPoint]], asof_date: str
) -> Dict[str, List[FlowPoint]]:
    asof = _parse_date(asof_date) or dt.date.min
    sliced: Dict[str, List[FlowPoint]] = {}
    for symbol, points in flow_map.items():
        # processed_date is the conservative availability gate. effective_date
        # can describe the underlying flow date before the API made it usable.
        sliced[symbol] = [
            point
            for point in points
            if (_parse_date(point.processed_date) or dt.date.min) <= asof
        ]
    return sliced


def _historical_quote_map(
    price_map: Mapping[str, Sequence[PricePoint]],
    asof_date: str,
) -> Dict[str, Dict[str, Any]]:
    """Build a quote-like snapshot from historical bars for backtests."""

    quote_map: Dict[str, Dict[str, Any]] = {}
    asof = _parse_date(asof_date) or dt.date.min
    for symbol, points in price_map.items():
        clean = [
            point
            for point in points
            if point.close is not None and (_parse_date(point.date) or dt.date.min) <= asof
        ]
        if not clean:
            continue
        latest = clean[-1]
        prev = clean[-2] if len(clean) >= 2 else None
        closes = [float(point.close) for point in clean]
        volumes = [
            float(point.volume)
            for point in clean[-20:]
            if point.volume is not None
        ]
        avg_volume = _mean(volumes)
        quote_map[symbol] = {
            "symbol": symbol,
            "price": latest.close,
            "changesPercentage": _pct_change(latest.close, prev.close) if prev else None,
            "volume": latest.volume,
            "avgVolume": avg_volume,
            "priceAvg50": _sma(closes, 50),
            "priceAvg200": _sma(closes, 200),
        }
    return quote_map


def recompute_gostop_backtest(
    *,
    symbols: Sequence[str],
    price_map: Mapping[str, Sequence[PricePoint]],
    flow_map: Mapping[str, Sequence[FlowPoint]],
    end_date: str,
    history_days: int = 60,
    horizon_bars: int = 5,
    hit_threshold_pct: float = 0.25,
) -> Dict[str, Any]:
    """Recompute historical GoStop decisions from point-in-time data slices."""

    bounded_price_map = _slice_price_map(price_map, end_date)
    bounded_flow_map = _slice_flow_map(flow_map, end_date)
    benchmark = "QQQ" if bounded_price_map.get("QQQ") else "SPY"
    benchmark_prices = list(bounded_price_map.get(benchmark) or [])
    end = _parse_date(end_date)
    if not benchmark_prices or not end:
        return {
            "evaluated_signals": 0,
            "note": "백테스트 기준 가격 데이터가 부족합니다.",
        }

    start = end - dt.timedelta(days=max(1, int(history_days)))
    candidate_dates = [
        point.date
        for point in benchmark_prices
        if start <= (_parse_date(point.date) or dt.date.min) <= end
    ]

    entries: List[Dict[str, Any]] = []
    action_hits: List[float] = []
    action_utilities: List[float] = []
    qqq_decision_hits: List[float] = []
    qqq_decision_utilities: List[float] = []
    qqq_strategy_returns: List[float] = []
    qqq_buy_hold_excesses: List[float] = []
    qqq_missed_upsides: List[float] = []
    qqq_drawdown_avoided: List[float] = []
    tactical_hits: List[float] = []
    tactical_utilities: List[float] = []
    entry_excesses: List[float] = []
    entry_hits: List[float] = []
    stop_utilities: List[float] = []
    stop_hits: List[float] = []
    feature_targets: List[Tuple[Dict[str, Optional[float]], Dict[str, float]]] = []
    score_returns: List[Tuple[int, float]] = []
    signal_stats: Dict[str, Dict[str, List[float]]] = {}

    for asof in candidate_dates:
        qqq_ret = _forward_return_from_prices(
            bounded_price_map.get("QQQ") or [], asof, horizon_bars
        )
        if qqq_ret is None:
            continue
        spy_ret = _forward_return_from_prices(
            bounded_price_map.get("SPY") or [], asof, horizon_bars
        )
        tqqq_ret = _forward_return_from_prices(
            bounded_price_map.get("TQQQ") or [], asof, horizon_bars
        )
        sliced_prices = _slice_price_map(bounded_price_map, asof)
        sliced_flows = _slice_flow_map(bounded_flow_map, asof)
        rows = build_radar_rows(
            symbols=symbols,
            price_map=sliced_prices,
            flow_map=sliced_flows,
            asof_date=asof,
        )
        state = classify_market_state(rows)
        decision = classify_gostop_decision(rows, state)
        nowcast = build_nowcast_overlay(
            symbols=symbols,
            price_map=sliced_prices,
            quote_map=_historical_quote_map(bounded_price_map, asof),
            gostop=asdict(decision),
        )
        tactical = build_tactical_overlay(
            gostop=asdict(decision),
            nowcast=nowcast,
        )
        ddm_signal = build_ddm_signal(
            symbols=symbols,
            price_map=sliced_prices,
            rows=rows,
            nowcast=nowcast,
            asof_date=asof,
        )
        policy_key = _qqq_policy_key(decision.action, str(tactical.get("action") or ""))
        qqq_decision = build_qqq_decision(
            gostop=asdict(decision),
            tactical=tactical,
            nowcast=nowcast,
            ddm_signal=ddm_signal,
        )
        qqq_exposure = float(qqq_decision.get("recommended_exposure_pct") or 0.0)
        score_returns.append((decision.score, qqq_ret))

        entry_returns = [
            _forward_return_from_prices(
                bounded_price_map.get(symbol) or [], asof, horizon_bars
            )
            for symbol in decision.entry_candidates
        ]
        stop_returns = [
            _forward_return_from_prices(
                bounded_price_map.get(symbol) or [], asof, horizon_bars
            )
            for symbol in decision.stop_candidates
        ]
        entry_ret = _mean(entry_returns)
        stop_ret = _mean(stop_returns)
        entry_excess = entry_ret - qqq_ret if entry_ret is not None else None
        stop_utility = qqq_ret - stop_ret if stop_ret is not None else None
        qqq_decision_utility = _qqq_decision_utility(qqq_ret, qqq_exposure)
        qqq_strategy_return = _qqq_strategy_return(qqq_ret, qqq_exposure)
        qqq_buy_hold_excess = qqq_strategy_return - qqq_ret
        missed_upside = _qqq_missed_upside(qqq_ret, qqq_exposure)
        drawdown_avoided = _qqq_drawdown_avoided(qqq_ret, qqq_exposure)
        action_utility = qqq_decision_utility
        action_hit = action_utility > hit_threshold_pct
        tactical_utility = _tactical_utility(qqq_ret, tactical)
        tactical_hit = tactical_utility > hit_threshold_pct

        action_hits.append(1.0 if action_hit else 0.0)
        action_utilities.append(action_utility)
        qqq_decision_hits.append(1.0 if action_hit else 0.0)
        qqq_decision_utilities.append(qqq_decision_utility)
        qqq_strategy_returns.append(qqq_strategy_return)
        qqq_buy_hold_excesses.append(qqq_buy_hold_excess)
        qqq_missed_upsides.append(missed_upside)
        qqq_drawdown_avoided.append(drawdown_avoided)
        tactical_hits.append(1.0 if tactical_hit else 0.0)
        tactical_utilities.append(tactical_utility)
        if entry_excess is not None:
            entry_excesses.append(entry_excess)
            entry_hits.append(1.0 if entry_excess > hit_threshold_pct else 0.0)
        if stop_utility is not None:
            stop_utilities.append(stop_utility)
            stop_hits.append(1.0 if stop_utility > hit_threshold_pct else 0.0)

        for row in rows:
            symbol_ret = _forward_return_from_prices(
                bounded_price_map.get(row.symbol) or [], asof, horizon_bars
            )
            if symbol_ret is None:
                continue
            stats = signal_stats.setdefault(
                row.signal,
                {
                    "returns": [],
                    "entry_excess": [],
                    "stop_utility": [],
                },
            )
            stats["returns"].append(symbol_ret)
            stats["entry_excess"].append(symbol_ret - qqq_ret)
            stats["stop_utility"].append(qqq_ret - symbol_ret)

        feature_snapshot = _gostop_feature_snapshot(
            {
                "market_state": state,
                "gostop": asdict(decision),
                "nowcast": nowcast,
                "tactical_overlay": tactical,
                "qqq_decision": qqq_decision,
                "ddm_signal": ddm_signal,
            }
        )
        feature_targets.append(
            (
                feature_snapshot,
                {
                    "action_utility": action_utility,
                    "qqq_decision_utility": qqq_decision_utility,
                    "qqq_strategy_return": qqq_strategy_return,
                    "tactical_utility": tactical_utility,
                    "qqq_return": qqq_ret,
                    "entry_excess": entry_excess if entry_excess is not None else 0.0,
                    "stop_utility": stop_utility if stop_utility is not None else 0.0,
                },
            )
        )
        entries.append(
            {
                "date": asof,
                "action": decision.action,
                "qqq_decision": qqq_decision.get("action"),
                "policy_key": policy_key,
                "qqq_exposure_pct": qqq_exposure,
                "score": decision.score,
                "mode": decision.mode,
                "tactical_action": tactical.get("action"),
                "tactical_probe_pct": tactical.get("max_tactical_risk_pct"),
                "ddm_signal": ddm_signal,
                "qqq_decision_utility_pct": qqq_decision_utility,
                "qqq_strategy_return_pct": qqq_strategy_return,
                "qqq_vs_buy_hold_pct": qqq_buy_hold_excess,
                "missed_upside_pct": missed_upside,
                "drawdown_avoided_pct": drawdown_avoided,
                "tactical_utility_pct": tactical_utility,
                "qqq_return_pct": qqq_ret,
                "spy_return_pct": spy_ret,
                "tqqq_return_pct": tqqq_ret,
                "entry_basket_return_pct": entry_ret,
                "entry_excess_qqq_pct": entry_excess,
                "stop_basket_return_pct": stop_ret,
                "stop_utility_pct": stop_utility,
                "action_hit": action_hit,
                "features": feature_snapshot,
            }
        )

    signal_summary: List[Dict[str, Any]] = []
    for signal, values in signal_stats.items():
        entry_values = values.get("entry_excess") or []
        stop_values = values.get("stop_utility") or []
        signal_summary.append(
            {
                "signal": signal,
                "n": len(values.get("returns") or []),
                "avg_forward_return_pct": _mean(values.get("returns") or []),
                "avg_entry_excess_qqq_pct": _mean(entry_values),
                "entry_hit_rate_pct": _hit_rate(entry_values, hit_threshold_pct),
                "avg_stop_utility_pct": _mean(stop_values),
                "stop_hit_rate_pct": _hit_rate(stop_values, hit_threshold_pct),
            }
        )
    signal_summary.sort(
        key=lambda item: max(
            abs(float(item.get("avg_entry_excess_qqq_pct") or 0.0)),
            abs(float(item.get("avg_stop_utility_pct") or 0.0)),
        ),
        reverse=True,
    )
    threshold_results = _score_threshold_tuning(
        score_returns,
        hit_threshold_pct=hit_threshold_pct,
    )
    adaptive_policy = _adaptive_exposure_policy(
        entries,
        hit_threshold_pct=hit_threshold_pct,
    )
    walk_forward = _walk_forward_adaptive_backtest(
        entries,
        hit_threshold_pct=hit_threshold_pct,
    )

    return {
        "evaluated_signals": len(entries),
        "history_days": history_days,
        "horizon_bars": horizon_bars,
        "hit_threshold_pct": hit_threshold_pct,
        "action_hit_rate_pct": (_mean(action_hits) or 0.0) * 100.0 if entries else None,
        "avg_action_utility_pct": _mean(action_utilities),
        "qqq_decision_hit_rate_pct": (_mean(qqq_decision_hits) or 0.0) * 100.0 if entries else None,
        "avg_qqq_decision_utility_pct": _mean(qqq_decision_utilities),
        "avg_qqq_strategy_return_pct": _mean(qqq_strategy_returns),
        "avg_qqq_vs_buy_hold_pct": _mean(qqq_buy_hold_excesses),
        "avg_missed_upside_pct": _mean(qqq_missed_upsides),
        "avg_drawdown_avoided_pct": _mean(qqq_drawdown_avoided),
        "tactical_hit_rate_pct": (_mean(tactical_hits) or 0.0) * 100.0 if tactical_hits else None,
        "avg_tactical_utility_pct": _mean(tactical_utilities),
        "avg_entry_excess_qqq_pct": _mean(entry_excesses),
        "entry_hit_rate_pct": (_mean(entry_hits) or 0.0) * 100.0 if entry_hits else None,
        "avg_stop_utility_pct": _mean(stop_utilities),
        "stop_hit_rate_pct": (_mean(stop_hits) or 0.0) * 100.0 if stop_hits else None,
        "best_score_threshold": threshold_results[0] if threshold_results else None,
        "score_thresholds": threshold_results[:8],
        "adaptive_exposure_policy": adaptive_policy,
        "walk_forward_adaptive": walk_forward,
        "signal_stats": signal_summary,
        "rolling_correlations": _rolling_feature_correlations(feature_targets)[:10],
        "recent": entries[-8:],
        "learning_entries": entries,
        "note": "FMP/Massive 과거 데이터만으로 point-in-time GoStop과 Tactical Overlay를 재계산합니다.",
    }


def _select_horizon(
    summaries: Sequence[Mapping[str, Any]],
    objective_key: str,
) -> Optional[Mapping[str, Any]]:
    usable = [
        item for item in summaries
        if item.get("evaluated_signals") and item.get(objective_key) is not None
    ]
    if not usable:
        return None
    return max(usable, key=lambda item: float(item.get(objective_key) or -999.0))


def _recommended_signals(
    signal_stats: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    hit_key: str,
    min_samples: int,
) -> List[Dict[str, Any]]:
    recommended: List[Dict[str, Any]] = []
    for item in signal_stats:
        n = int(item.get("n") or 0)
        metric = _to_float(item.get(metric_key))
        hit_rate = _to_float(item.get(hit_key))
        if n < min_samples or metric is None:
            continue
        if metric > 0 and (hit_rate is None or hit_rate >= 45.0):
            recommended.append(
                {
                    "signal": item.get("signal"),
                    "n": n,
                    metric_key: metric,
                    hit_key: hit_rate,
                }
            )
    recommended.sort(key=lambda item: float(item.get(metric_key) or 0.0), reverse=True)
    return recommended


def _symbols_for_signals(
    rows: Sequence[RadarRow],
    signals: Sequence[str],
    *,
    fallback: Sequence[str],
    limit: int = 6,
) -> List[str]:
    allowed = {str(signal) for signal in signals if signal}
    if not allowed:
        return list(fallback)[:limit]
    symbols = [
        row.symbol
        for row in sorted(rows, key=_row_sort_key, reverse=True)
        if row.signal in allowed
    ]
    return symbols[:limit]


def _sma(values: Sequence[float], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    return sum(values[-window:]) / float(window)


def _quote_float(quote: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _to_float(quote.get(key))
        if value is not None:
            return value
    return None


def build_nowcast_overlay(
    *,
    symbols: Sequence[str],
    price_map: Mapping[str, Sequence[PricePoint]],
    quote_map: Mapping[str, Mapping[str, Any]],
    gostop: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build an intraday/now price layer over slower T+1 GoStop signals."""

    rows: List[NowCastRow] = []
    for symbol in symbols:
        sym = symbol.upper()
        quote = quote_map.get(sym) or {}
        if not quote:
            continue
        price = _quote_float(quote, "price")
        day_change = _quote_float(quote, "changesPercentage")
        volume = _quote_float(quote, "volume")
        avg_volume = _quote_float(quote, "avgVolume")
        price_avg50 = _quote_float(quote, "priceAvg50")
        price_avg200 = _quote_float(quote, "priceAvg200")
        closes = [
            float(point.close)
            for point in price_map.get(sym, [])
            if point.close is not None
        ]
        sma20 = _sma(closes, 20)
        above_sma20 = _pct_change(price, sma20)
        above_sma50 = _pct_change(price, price_avg50)
        above_sma200 = _pct_change(price, price_avg200)
        rel_volume = volume / avg_volume if volume is not None and avg_volume else None
        if day_change is not None and day_change >= 1.0 and rel_volume is not None and rel_volume >= 1.10:
            signal = "live_accumulation"
        elif day_change is not None and day_change >= 1.0:
            signal = "live_price_strength"
        elif day_change is not None and day_change <= -1.0 and rel_volume is not None and rel_volume >= 1.10:
            signal = "live_distribution"
        elif day_change is not None and day_change <= -1.0:
            signal = "live_price_weakness"
        elif above_sma20 is not None and above_sma20 > 0 and above_sma50 is not None and above_sma50 > 0:
            signal = "live_constructive"
        else:
            signal = "live_mixed"
        rows.append(
            NowCastRow(
                symbol=sym,
                price=price,
                day_change_pct=day_change,
                rel_volume=rel_volume,
                above_sma20_pct=above_sma20,
                above_sma50_pct=above_sma50,
                above_sma200_pct=above_sma200,
                signal=signal,
            )
        )

    if not rows:
        return {
            "enabled": False,
            "status": "no_quote_data",
            "note": "실시간 quote 데이터가 없어 NowCast를 건너뜁니다.",
        }

    by_symbol = {row.symbol: row for row in rows}
    risk_rows = [row for row in rows if row.symbol in RISK_ETFS]
    defensive_rows = [row for row in rows if row.symbol in DEFENSIVE_ETFS]
    risk_day = _mean(row.day_change_pct for row in risk_rows)
    defensive_day = _mean(row.day_change_pct for row in defensive_rows)
    risk_positive = sum(1 for row in risk_rows if (row.day_change_pct or 0.0) > 0)
    risk_count = len([row for row in risk_rows if row.day_change_pct is not None])
    above50_count = sum(1 for row in risk_rows if (row.above_sma50_pct or -999.0) > 0)
    rel_volume_median = _median(row.rel_volume for row in risk_rows)
    spy = by_symbol.get("SPY")
    qqq = by_symbol.get("QQQ")
    iwm = by_symbol.get("IWM")
    dia = by_symbol.get("DIA")
    hyg = by_symbol.get("HYG")
    lqd = by_symbol.get("LQD")

    score = 50.0
    reasons: List[str] = []
    warnings: List[str] = []
    if risk_day is not None:
        if risk_day >= 1.0:
            score += 16
            reasons.append(f"Risk ETF 당일 평균 상승률이 {_fmt_pct(risk_day)}입니다.")
        elif risk_day > 0:
            score += 7
            reasons.append(f"Risk ETF 당일 평균이 플러스입니다({_fmt_pct(risk_day)}).")
        elif risk_day <= -1.0:
            score -= 16
            warnings.append(f"Risk ETF 당일 평균이 {_fmt_pct(risk_day)}로 약합니다.")
        else:
            score -= 5
            warnings.append(f"Risk ETF 당일 모멘텀이 약합니다({_fmt_pct(risk_day)}).")
    if risk_count:
        positive_ratio = risk_positive / risk_count
        if positive_ratio >= 0.70:
            score += 10
            reasons.append(f"Risk ETF 중 {risk_positive}/{risk_count}개가 상승 중입니다.")
        elif positive_ratio <= 0.35:
            score -= 10
            warnings.append(f"Risk ETF 상승 확산이 {risk_positive}/{risk_count}개로 약합니다.")
    if qqq and (qqq.day_change_pct or 0.0) > 1.0:
        score += 8
        reasons.append(f"QQQ 당일 상승률이 {_fmt_pct(qqq.day_change_pct)}입니다.")
    if iwm and dia and (iwm.day_change_pct or 0.0) > 0 and (dia.day_change_pct or 0.0) > 0:
        score += 5
        reasons.append("IWM/DIA가 같이 올라 장중 확산은 개선됐습니다.")
    elif iwm and dia:
        score -= 5
        warnings.append("IWM/DIA 동반 확인이 약해 broad risk-on은 아직 제한적입니다.")
    credit_ok = bool(hyg and lqd and (hyg.day_change_pct or 0.0) > 0 and (lqd.day_change_pct or 0.0) > 0)
    if credit_ok:
        score += 5
        reasons.append("HYG/LQD가 당일 플러스로 credit 가격은 완화됐습니다.")
    else:
        score -= 5
        warnings.append("HYG/LQD 당일 가격 확인이 부족합니다.")
    if risk_count and above50_count / risk_count >= 0.60:
        score += 6
        reasons.append("Risk ETF 다수가 50일선 위에 있습니다.")
    if rel_volume_median is not None and rel_volume_median >= 1.10 and risk_day is not None and risk_day > 0:
        score += 6
        reasons.append(f"Risk ETF 거래량 중앙값이 평소의 {rel_volume_median:.2f}배입니다.")

    score_int = int(round(_clamp(score, 0.0, 100.0)))
    base_action = str(gostop.get("action") or "").upper()
    if base_action in {"STOP", "WAIT"} and score_int >= 70:
        status = "live_relief_against_t1_stop"
        stance = "장중 반등은 강하지만 T+1 수급 확인 전까지 신규 스윙 진입은 보류합니다."
    elif base_action in {"STOP", "WAIT"} and score_int >= 55:
        status = "live_bounce_watch"
        stance = "장중 반등은 있으나 T+1 GoStop을 뒤집을 정도는 아닙니다."
    elif base_action in {"STOP", "WAIT"}:
        status = "live_confirms_t1_stop"
        stance = "장중 기술 신호도 T+1 Stop/Wait 판단을 확인합니다."
    elif base_action in {"GO", "GO_SMALL"} and score_int < 45:
        status = "live_fades_t1_go"
        stance = "T+1 Go 신호가 있어도 장중 가격 확인이 약해 진입을 늦춥니다."
    else:
        status = "live_confirms_t1_go"
        stance = "T+1 Go 신호를 장중 가격이 확인합니다."

    leaders = sorted(
        rows,
        key=lambda row: row.day_change_pct if row.day_change_pct is not None else -999.0,
        reverse=True,
    )[:8]
    laggards = sorted(
        rows,
        key=lambda row: row.day_change_pct if row.day_change_pct is not None else 999.0,
    )[:8]
    return {
        "enabled": True,
        "status": status,
        "score": score_int,
        "stance": stance,
        "risk_day_avg_pct": risk_day,
        "defensive_day_avg_pct": defensive_day,
        "risk_positive_count": risk_positive,
        "risk_count": risk_count,
        "risk_above_sma50_count": above50_count,
        "risk_rel_volume_median": rel_volume_median,
        "reasons": reasons[:6],
        "warnings": warnings[:6],
        "leaders": [asdict(row) for row in leaders],
        "laggards": [asdict(row) for row in laggards],
        "rows": [asdict(row) for row in rows],
        "note": "FMP quote 기반 당일 가격/거래량/이평선 레이어입니다. ETF flow/NAV 기반 GoStop보다 빠르지만 수급 확정력은 낮습니다.",
    }


def build_tactical_overlay(
    *,
    gostop: Mapping[str, Any],
    nowcast: Mapping[str, Any],
) -> Dict[str, Any]:
    """Blend slower T+1 GoStop with same-day quote breadth.

    GoStop remains the swing-entry source of truth because ETF fund flow/NAV is
    processed with a lag. This overlay prevents a strong same-day reversal from
    being hidden behind a stale-looking STOP label.
    """

    base_action = str(gostop.get("action") or "").upper()
    now_score = _to_float(nowcast.get("score"))
    risk_day = _to_float(nowcast.get("risk_day_avg_pct"))
    rel_volume = _to_float(nowcast.get("risk_rel_volume_median"))
    risk_positive = _to_float(nowcast.get("risk_positive_count"))
    risk_count = _to_float(nowcast.get("risk_count"))
    breadth = (risk_positive / risk_count) if risk_positive is not None and risk_count else None

    reasons: List[str] = []
    blocks: List[str] = []
    if now_score is not None:
        reasons.append(f"NowCast score가 {int(round(now_score))}/100입니다.")
    if risk_day is not None:
        if risk_day >= 1.0:
            reasons.append(f"Risk ETF 당일 평균이 {_fmt_pct(risk_day)}로 강합니다.")
        elif risk_day <= 0:
            blocks.append(f"Risk ETF 당일 평균이 {_fmt_pct(risk_day)}로 약합니다.")
    if breadth is not None:
        if breadth >= 0.80:
            reasons.append(f"Risk ETF 상승 확산이 {int(risk_positive or 0)}/{int(risk_count or 0)}로 넓습니다.")
        elif breadth <= 0.50:
            blocks.append(f"Risk ETF 상승 확산이 {int(risk_positive or 0)}/{int(risk_count or 0)}로 제한적입니다.")
    if rel_volume is not None:
        if rel_volume >= 1.10:
            reasons.append(f"거래량 중앙값이 {rel_volume:.2f}x로 가격 반등을 확인합니다.")
        else:
            blocks.append(f"거래량 중앙값이 {rel_volume:.2f}x로 강한 수급 확정은 아직 부족합니다.")

    if not nowcast.get("enabled"):
        return {
            "enabled": False,
            "action": base_action or "UNKNOWN",
            "max_tactical_risk_pct": 0,
            "swing_action": base_action or "UNKNOWN",
            "headline": "NowCast 데이터가 없어 tactical overlay를 계산하지 못했습니다.",
            "reasons": reasons,
            "blocks": blocks,
        }

    if base_action in {"GO", "GO_SMALL"}:
        if now_score is not None and now_score < 45:
            action = "DELAY_ENTRY"
            risk_pct = 0
            headline = "T+1 Go 신호가 있어도 오늘 가격 확인이 약해 진입을 늦춥니다."
        else:
            action = base_action
            risk_pct = int(gostop.get("max_new_risk_pct") or 0)
            headline = "T+1 Go 신호와 오늘 가격 확인이 대체로 정렬되어 있습니다."
    elif (
        now_score is not None
        and now_score >= 90
        and risk_day is not None
        and risk_day >= 1.0
        and breadth is not None
        and breadth >= 0.80
    ):
        if rel_volume is not None and rel_volume >= 1.10:
            action = "TACTICAL_GO_SMALL"
            risk_pct = 25
            headline = "T+1 STOP이지만 오늘 가격·거래량 확인이 강해 단기 소액 진입 후보로 격상합니다."
        else:
            action = "WATCH_REBOUND"
            risk_pct = 10
            headline = "T+1 STOP이지만 오늘 가격 확산은 강합니다. 거래량/다음 flow 확인 전까지 추격 대신 관찰·소액 탐색만 허용합니다."
    elif now_score is not None and now_score >= 70:
        action = "WATCH_REBOUND"
        risk_pct = 0
        headline = "T+1 STOP/WAIT이지만 오늘 반등이 있어 다음 flow 확인 대상입니다."
    else:
        action = base_action or "STOP"
        risk_pct = 0
        headline = "오늘 가격 확인도 T+1 GoStop을 뒤집을 정도는 아닙니다."

    return {
        "enabled": True,
        "action": action,
        "max_tactical_risk_pct": risk_pct,
        "swing_action": base_action or "UNKNOWN",
        "swing_max_new_risk_pct": int(gostop.get("max_new_risk_pct") or 0),
        "headline": headline,
        "reasons": reasons[:6],
        "blocks": blocks[:6],
    }


def auto_tune_gostop(
    *,
    symbols: Sequence[str],
    price_map: Mapping[str, Sequence[PricePoint]],
    flow_map: Mapping[str, Sequence[FlowPoint]],
    rows: Sequence[RadarRow],
    current_decision: Mapping[str, Any],
    end_date: str,
    current_context: Optional[Mapping[str, Any]] = None,
    history_days: int = 180,
    horizons: Optional[Sequence[int]] = None,
    hit_threshold_pct: float = 0.25,
    min_signal_samples: int = 5,
) -> Dict[str, Any]:
    """Tune GoStop evaluation horizon and candidate filters.

    This is intentionally an overlay rather than a silent rewrite of the core
    rule set. It selects robust horizons and signal groups from point-in-time
    recomputation, then applies those filters to today's candidates.
    """

    clean_horizons = sorted(
        {max(1, int(value)) for value in (horizons or DEFAULT_TUNE_HORIZONS)}
    )
    if not clean_horizons:
        clean_horizons = list(DEFAULT_TUNE_HORIZONS)

    summaries: List[Dict[str, Any]] = []
    for horizon in clean_horizons:
        result = recompute_gostop_backtest(
            symbols=symbols,
            price_map=price_map,
            flow_map=flow_map,
            end_date=end_date,
            history_days=history_days,
            horizon_bars=horizon,
            hit_threshold_pct=hit_threshold_pct,
        )
        evaluated = int(result.get("evaluated_signals") or 0)
        action_hit = _to_float(result.get("action_hit_rate_pct"))
        entry_hit = _to_float(result.get("entry_hit_rate_pct"))
        stop_hit = _to_float(result.get("stop_hit_rate_pct"))
        action_utility = _to_float(result.get("avg_action_utility_pct"))
        qqq_hit = _to_float(result.get("qqq_decision_hit_rate_pct"))
        qqq_utility = _to_float(result.get("avg_qqq_decision_utility_pct"))
        qqq_strategy = _to_float(result.get("avg_qqq_strategy_return_pct"))
        qqq_vs_hold = _to_float(result.get("avg_qqq_vs_buy_hold_pct"))
        tactical_hit = _to_float(result.get("tactical_hit_rate_pct"))
        tactical_utility = _to_float(result.get("avg_tactical_utility_pct"))
        entry_excess = _to_float(result.get("avg_entry_excess_qqq_pct"))
        stop_utility = _to_float(result.get("avg_stop_utility_pct"))
        best_threshold = result.get("best_score_threshold") or {}
        adaptive_policy = result.get("adaptive_exposure_policy") or {}
        walk_forward = result.get("walk_forward_adaptive") or {}
        walk_forward_hit = _to_float(walk_forward.get("hit_rate_pct"))
        walk_forward_strategy = _to_float(walk_forward.get("avg_strategy_return_pct"))
        boost_strategy = _to_float(walk_forward.get("avg_boosted_strategy_return_pct"))
        boost_vs_hold = _to_float(walk_forward.get("avg_boosted_vs_buy_hold_pct"))
        boost_excess = _to_float(walk_forward.get("avg_boost_excess_vs_qqq_strategy_pct"))
        summary = {
            "horizon_bars": horizon,
            "evaluated_signals": evaluated,
            "action_hit_rate_pct": action_hit,
            "avg_action_utility_pct": action_utility,
            "action_objective": _objective(action_utility, action_hit, evaluated),
            "qqq_decision_hit_rate_pct": qqq_hit,
            "avg_qqq_decision_utility_pct": qqq_utility,
            "avg_qqq_strategy_return_pct": qqq_strategy,
            "avg_qqq_vs_buy_hold_pct": qqq_vs_hold,
            "qqq_decision_objective": _objective(qqq_utility, qqq_hit, evaluated),
            "walk_forward_hit_rate_pct": walk_forward_hit,
            "avg_walk_forward_strategy_return_pct": walk_forward_strategy,
            "avg_walk_forward_vs_buy_hold_pct": _to_float(walk_forward.get("avg_vs_buy_hold_pct")),
            "walk_forward_objective": _objective(walk_forward_strategy, walk_forward_hit, evaluated),
            "avg_boosted_strategy_return_pct": boost_strategy,
            "avg_boosted_vs_buy_hold_pct": boost_vs_hold,
            "avg_boost_excess_vs_qqq_strategy_pct": boost_excess,
            "boost_objective": (boost_strategy or 0.0) + (boost_excess or 0.0) * 0.5,
            "tactical_hit_rate_pct": tactical_hit,
            "avg_tactical_utility_pct": tactical_utility,
            "tactical_objective": _objective(tactical_utility, tactical_hit, evaluated),
            "avg_entry_excess_qqq_pct": entry_excess,
            "entry_hit_rate_pct": entry_hit,
            "entry_objective": _objective(entry_excess, entry_hit, evaluated),
            "avg_stop_utility_pct": stop_utility,
            "stop_hit_rate_pct": stop_hit,
            "stop_objective": _objective(stop_utility, stop_hit, evaluated),
            "best_score_threshold": best_threshold,
            "adaptive_exposure_policy": adaptive_policy,
            "walk_forward_adaptive": walk_forward,
            "learning_entries": result.get("learning_entries") or [],
            "signal_stats": result.get("signal_stats") or [],
        }
        summaries.append(summary)

    action_choice = _select_horizon(summaries, "walk_forward_objective")
    tactical_choice = _select_horizon(summaries, "tactical_objective")
    entry_choice = _select_horizon(summaries, "entry_objective")
    stop_choice = _select_horizon(summaries, "stop_objective")
    if not (action_choice and entry_choice and stop_choice):
        return {
            "enabled": True,
            "status": "insufficient_data",
            "history_days": history_days,
            "horizons": clean_horizons,
            "note": "자동 튜닝에 필요한 과거 재계산 샘플이 부족합니다.",
            "horizon_summaries": summaries,
        }

    entry_signals = _recommended_signals(
        entry_choice.get("signal_stats") or [],
        metric_key="avg_entry_excess_qqq_pct",
        hit_key="entry_hit_rate_pct",
        min_samples=min_signal_samples,
    )
    stop_signals = _recommended_signals(
        stop_choice.get("signal_stats") or [],
        metric_key="avg_stop_utility_pct",
        hit_key="stop_hit_rate_pct",
        min_samples=min_signal_samples,
    )
    entry_signal_names = [str(item.get("signal")) for item in entry_signals]
    stop_signal_names = [str(item.get("signal")) for item in stop_signals]
    tuned_entry_candidates = _symbols_for_signals(
        rows,
        entry_signal_names,
        fallback=current_decision.get("entry_candidates") or [],
    )
    tuned_stop_candidates = _symbols_for_signals(
        rows,
        stop_signal_names,
        fallback=current_decision.get("stop_candidates") or [],
    )

    threshold_info = action_choice.get("best_score_threshold") or {}
    entry_score_min = int(threshold_info.get("entry_score_min") or 58)
    current_score = int(current_decision.get("score") or 0)
    base_risk = int(current_decision.get("max_new_risk_pct") or 0)
    tuned_gate_open = current_score >= entry_score_min
    tuned_max_new_risk_pct = base_risk if tuned_gate_open else 0
    qqq_metric = _to_float(action_choice.get("avg_qqq_decision_utility_pct")) or 0.0
    qqq_strategy_metric = _to_float(action_choice.get("avg_qqq_strategy_return_pct")) or 0.0
    walk_forward_hit_metric = _to_float(action_choice.get("walk_forward_hit_rate_pct"))
    walk_forward_strategy_metric = _to_float(action_choice.get("avg_walk_forward_strategy_return_pct"))
    walk_forward_vs_hold_metric = _to_float(action_choice.get("avg_walk_forward_vs_buy_hold_pct"))
    boosted_strategy_metric = _to_float(action_choice.get("avg_boosted_strategy_return_pct"))
    boosted_vs_hold_metric = _to_float(action_choice.get("avg_boosted_vs_buy_hold_pct"))
    boost_excess_metric = _to_float(action_choice.get("avg_boost_excess_vs_qqq_strategy_pct"))
    adaptive_policy = action_choice.get("adaptive_exposure_policy") or {}
    walk_forward_adaptive = action_choice.get("walk_forward_adaptive") or {}
    pattern_adaptive_decision: Dict[str, Any] = {}
    tqqq_boost_decision: Dict[str, Any] = {}
    if current_context:
        current_tactical = current_context.get("tactical_overlay") or {}
        current_nowcast = current_context.get("nowcast") or {}
        base_qqq_decision = build_qqq_decision(
            gostop=current_decision,
            tactical=current_tactical,
            nowcast=current_nowcast,
            ddm_signal=current_context.get("ddm_signal") or {},
        )
        current_features = _gostop_feature_snapshot(
            {
                "market_state": current_context.get("market_state") or {},
                "gostop": current_decision,
                "nowcast": current_nowcast,
                "tactical_overlay": current_tactical,
                "qqq_decision": base_qqq_decision,
                "ddm_signal": current_context.get("ddm_signal") or {},
            }
        )
        current_entry = {
            "date": end_date,
            "action": str(current_decision.get("action") or "").upper(),
            "qqq_decision": base_qqq_decision.get("action"),
            "policy_key": _qqq_policy_key(
                str(current_decision.get("action") or "").upper(),
                str(current_tactical.get("action") or ""),
            ),
            "qqq_exposure_pct": base_qqq_decision.get("recommended_exposure_pct"),
            "ddm_signal": current_context.get("ddm_signal") or {},
            "features": current_features,
        }
        pattern_adaptive_decision = _similarity_adaptive_exposure_for_entry(
            current_entry,
            action_choice.get("learning_entries") or [],
            hit_threshold_pct=hit_threshold_pct,
        )
        qqq_exposure_for_boost = _to_float(base_qqq_decision.get("recommended_exposure_pct")) or 0.0
        if not (current_context.get("ddm_signal") or {}).get("enabled"):
            pattern_exposure_for_boost = _to_float(pattern_adaptive_decision.get("exposure_pct"))
            if pattern_exposure_for_boost is not None:
                qqq_exposure_for_boost = pattern_exposure_for_boost
        tqqq_boost_decision = _similarity_tqqq_boost_for_entry(
            current_entry,
            action_choice.get("learning_entries") or [],
            qqq_exposure_pct=qqq_exposure_for_boost,
            hit_threshold_pct=hit_threshold_pct,
        )
    tactical_metric = _to_float(tactical_choice.get("avg_tactical_utility_pct")) if tactical_choice else None
    entry_metric = _to_float(entry_choice.get("avg_entry_excess_qqq_pct")) or 0.0
    stop_metric = _to_float(stop_choice.get("avg_stop_utility_pct")) or 0.0
    evaluated = int(action_choice.get("evaluated_signals") or 0)
    if evaluated >= 40 and (entry_metric > 0 or stop_metric > 0):
        confidence = "medium"
    else:
        confidence = "low"
    if evaluated >= 80 and qqq_metric > 0 and entry_metric > 0 and stop_metric > 0:
        confidence = "high"

    if qqq_metric <= 0 and entry_metric <= 0 and stop_metric <= 0:
        status = "observe_only"
    elif tuned_gate_open:
        status = "tuned_entry_allowed"
    else:
        status = "tuned_entry_closed"

    return {
        "enabled": True,
        "status": status,
        "confidence": confidence,
        "history_days": history_days,
        "horizons": clean_horizons,
        "selected": {
            "action_horizon_bars": action_choice.get("horizon_bars"),
            "tactical_horizon_bars": tactical_choice.get("horizon_bars") if tactical_choice else None,
            "entry_horizon_bars": entry_choice.get("horizon_bars"),
            "stop_horizon_bars": stop_choice.get("horizon_bars"),
            "entry_score_min": entry_score_min,
            "tuned_gate_open": tuned_gate_open,
            "tuned_max_new_risk_pct": tuned_max_new_risk_pct,
            "avg_qqq_decision_utility_pct": qqq_metric,
            "avg_qqq_strategy_return_pct": qqq_strategy_metric,
            "walk_forward_hit_rate_pct": walk_forward_hit_metric,
            "avg_walk_forward_strategy_return_pct": walk_forward_strategy_metric,
            "avg_walk_forward_vs_buy_hold_pct": walk_forward_vs_hold_metric,
            "avg_boosted_strategy_return_pct": boosted_strategy_metric,
            "avg_boosted_vs_buy_hold_pct": boosted_vs_hold_metric,
            "avg_boost_excess_vs_qqq_strategy_pct": boost_excess_metric,
            "adaptive_exposure_policy": adaptive_policy,
            "walk_forward_adaptive": walk_forward_adaptive,
            "pattern_adaptive_decision": pattern_adaptive_decision,
            "tqqq_boost_decision": tqqq_boost_decision,
            "avg_tactical_utility_pct": tactical_metric,
        },
        "recommended_entry_signals": entry_signals[:6],
        "recommended_stop_signals": stop_signals[:6],
        "tuned_entry_candidates": tuned_entry_candidates,
        "tuned_stop_candidates": tuned_stop_candidates,
        "adaptive_exposure_policy": adaptive_policy,
        "walk_forward_adaptive": walk_forward_adaptive,
        "pattern_adaptive_decision": pattern_adaptive_decision,
        "tqqq_boost_decision": tqqq_boost_decision,
        "horizon_summaries": [
            {
                key: value
                for key, value in item.items()
                if key not in {"signal_stats", "learning_entries"}
            }
            for item in summaries
        ],
        "note": "5/10/21일 등 복수 horizon에서 QQQ 결정/entry/stop/tactical을 따로 고르는 자동 튜닝 오버레이입니다.",
    }


def evaluate_gostop_history(
    current_report: Mapping[str, Any],
    *,
    history_dir: str = "sweet_spot_reports",
    max_reports: int = 60,
    min_age_days: int = 1,
) -> Dict[str, Any]:
    """Evaluate saved GoStop reports against the current price snapshot.

    This is an as-of-now performance audit, not a backtest engine. It reads
    prior saved JSON payloads, compares their candidate/avoid baskets with the
    current prices, and reports whether the prior Go/Stop decision added value.
    """

    current_rows = _rows_from_payload(current_report)
    current_prices = _price_lookup(current_rows)
    current_date_text = _latest_price_date(current_rows) or (current_report.get("data_window") or {}).get("to")
    current_date = _parse_date(current_date_text)
    if not current_date or not current_prices:
        return {
            "evaluated_reports": 0,
            "note": "현재 가격 스냅샷이 없어 성과 평가를 건너뜁니다.",
        }

    root = Path(history_dir)
    if not root.exists():
        return {
            "evaluated_reports": 0,
            "note": "아직 저장된 GoStop 히스토리가 없습니다.",
        }

    files = sorted(root.glob("etf_gostop_*.json"), key=lambda path: path.name, reverse=True)
    if max_reports > 0:
        files = files[:max_reports]

    current_generated = str(current_report.get("generated_at_utc") or "")
    entries: List[Dict[str, Any]] = []
    action_hits: List[float] = []
    action_utilities: List[float] = []
    qqq_decision_hits: List[float] = []
    qqq_decision_utilities: List[float] = []
    qqq_strategy_returns: List[float] = []
    qqq_buy_hold_excesses: List[float] = []
    qqq_missed_upsides: List[float] = []
    qqq_drawdown_avoided: List[float] = []
    entry_excesses: List[float] = []
    stop_utilities: List[float] = []
    feature_targets: List[Tuple[Dict[str, Optional[float]], Dict[str, float]]] = []

    for path in files:
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(prior.get("generated_at_utc") or "") == current_generated:
            continue
        prior_gostop = prior.get("gostop") or {}
        action = str(prior_gostop.get("action") or "").upper()
        if action not in {"GO", "GO_SMALL", "WAIT", "STOP"}:
            continue
        prior_rows = _rows_from_payload(prior)
        prior_date_text = _latest_price_date(prior_rows) or (prior.get("data_window") or {}).get("to")
        prior_date = _parse_date(prior_date_text)
        if not prior_date:
            continue
        age_days = (current_date - prior_date).days
        if age_days < min_age_days:
            continue
        prior_prices = _price_lookup(prior_rows)
        qqq_ret = _avg_forward_return(["QQQ"], current_prices=current_prices, prior_prices=prior_prices)
        spy_ret = _avg_forward_return(["SPY"], current_prices=current_prices, prior_prices=prior_prices)
        if qqq_ret is None:
            continue
        prior_qqq_decision = prior.get("qqq_decision") or build_qqq_decision(
            gostop=prior_gostop,
            tactical=prior.get("tactical_overlay") or {},
            nowcast=prior.get("nowcast") or {},
            auto_tune=prior.get("auto_tune") or {},
        )
        qqq_exposure = float(prior_qqq_decision.get("recommended_exposure_pct") or 0.0)

        entry_ret = _avg_forward_return(
            prior_gostop.get("entry_candidates") or [],
            current_prices=current_prices,
            prior_prices=prior_prices,
        )
        stop_ret = _avg_forward_return(
            prior_gostop.get("stop_candidates") or [],
            current_prices=current_prices,
            prior_prices=prior_prices,
        )
        qqq_decision_utility = _qqq_decision_utility(qqq_ret, qqq_exposure)
        qqq_strategy_return = _qqq_strategy_return(qqq_ret, qqq_exposure)
        qqq_buy_hold_excess = qqq_strategy_return - qqq_ret
        missed_upside = _qqq_missed_upside(qqq_ret, qqq_exposure)
        drawdown_avoided = _qqq_drawdown_avoided(qqq_ret, qqq_exposure)
        action_utility = qqq_decision_utility
        action_hit = action_utility > 0.25
        entry_excess = entry_ret - qqq_ret if entry_ret is not None else None
        stop_utility = qqq_ret - stop_ret if stop_ret is not None else None

        action_hits.append(1.0 if action_hit else 0.0)
        action_utilities.append(action_utility)
        qqq_decision_hits.append(1.0 if action_hit else 0.0)
        qqq_decision_utilities.append(qqq_decision_utility)
        qqq_strategy_returns.append(qqq_strategy_return)
        qqq_buy_hold_excesses.append(qqq_buy_hold_excess)
        qqq_missed_upsides.append(missed_upside)
        qqq_drawdown_avoided.append(drawdown_avoided)
        if entry_excess is not None:
            entry_excesses.append(entry_excess)
        if stop_utility is not None:
            stop_utilities.append(stop_utility)
        feature_targets.append(
            (
                _gostop_feature_snapshot(prior),
                {
                    "action_utility": action_utility,
                    "qqq_decision_utility": qqq_decision_utility,
                    "qqq_strategy_return": qqq_strategy_return,
                    "qqq_return": qqq_ret,
                    "entry_excess": entry_excess if entry_excess is not None else 0.0,
                    "stop_utility": stop_utility if stop_utility is not None else 0.0,
                },
            )
        )

        entries.append(
            {
                "file": path.name,
                "date": prior_date.isoformat(),
                "age_days": age_days,
                "action": action,
                "qqq_decision": prior_qqq_decision.get("action"),
                "qqq_exposure_pct": qqq_exposure,
                "score": prior_gostop.get("score"),
                "qqq_return_pct": qqq_ret,
                "spy_return_pct": spy_ret,
                "qqq_decision_utility_pct": qqq_decision_utility,
                "qqq_strategy_return_pct": qqq_strategy_return,
                "qqq_vs_buy_hold_pct": qqq_buy_hold_excess,
                "missed_upside_pct": missed_upside,
                "drawdown_avoided_pct": drawdown_avoided,
                "entry_basket_return_pct": entry_ret,
                "entry_excess_qqq_pct": entry_excess,
                "stop_basket_return_pct": stop_ret,
                "stop_utility_pct": stop_utility,
                "action_hit": action_hit,
            }
        )

    note = (
        "저장된 GoStop JSON을 현재 가격으로 재평가합니다."
        if entries
        else "평가 가능한 과거 GoStop JSON이 아직 없습니다."
    )
    return {
        "evaluated_reports": len(entries),
        "current_price_date": current_date.isoformat(),
        "action_hit_rate_pct": (_mean(action_hits) or 0.0) * 100.0 if entries else None,
        "avg_action_utility_pct": _mean(action_utilities),
        "qqq_decision_hit_rate_pct": (_mean(qqq_decision_hits) or 0.0) * 100.0 if entries else None,
        "avg_qqq_decision_utility_pct": _mean(qqq_decision_utilities),
        "avg_qqq_strategy_return_pct": _mean(qqq_strategy_returns),
        "avg_qqq_vs_buy_hold_pct": _mean(qqq_buy_hold_excesses),
        "avg_missed_upside_pct": _mean(qqq_missed_upsides),
        "avg_drawdown_avoided_pct": _mean(qqq_drawdown_avoided),
        "avg_entry_excess_qqq_pct": _mean(entry_excesses),
        "avg_stop_utility_pct": _mean(stop_utilities),
        "rolling_correlations": _rolling_feature_correlations(feature_targets)[:10],
        "recent": entries[:8],
        "note": note,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = [RadarRow(**row) if isinstance(row, dict) else row for row in report.get("rows", [])]
    state = report.get("market_state") or {}
    gostop = report.get("gostop") or {}
    nowcast = report.get("nowcast") or {}
    tactical = report.get("tactical_overlay") or {}
    ddm_signal = report.get("ddm_signal") or {}
    qqq_decision = report.get("qqq_decision") or build_qqq_decision(
        gostop=gostop,
        tactical=tactical,
        nowcast=nowcast,
        auto_tune=report.get("auto_tune") or {},
    )
    tqqq_boost = report.get("tqqq_boost") or {}
    tqqq_eff_beta = _to_float(tqqq_boost.get("effective_beta"))
    tqqq_eff_beta_text = f"{tqqq_eff_beta:.2f}" if tqqq_eff_beta is not None else "N/A"
    auto_tune = report.get("auto_tune") or {}
    performance = report.get("performance") or {}
    backtest = report.get("history_backtest") or {}
    generated = report.get("generated_at_utc")
    generated_kst = report.get("generated_at_kst")
    publish_date_kst = report.get("publish_date_kst")
    data_window = report.get("data_window") or {}
    freshness = report.get("data_freshness") or build_data_freshness_audit(rows)
    ddm_audit = report.get("ddm_input_audit") or build_ddm_input_audit(ddm_signal, freshness)
    flow_audit = report.get("flow_persistence_audit") or build_flow_persistence_audit(
        rows=rows,
        freshness=freshness,
        market_state=state,
    )
    options_ref = report.get("options_futures_reference") or build_options_futures_reference(None)
    options_ddm_explain = report.get("options_ddm_explanation") or build_options_ddm_explanation(
        options_ref=options_ref,
        ddm_signal=ddm_signal,
        freshness=freshness,
        qqq_decision=qqq_decision,
        tqqq_boost=tqqq_boost,
    )
    data_asof = freshness.get("price_asof_date") or _latest_price_date(rows) or data_window.get("to") or ddm_signal.get("asof_date") or "N/A"
    warnings = report.get("warnings") or []

    lines: List[str] = []
    lines.append("# ETF GoStop QQQ Decision Report")
    lines.append("")
    lines.append(f"- 발행일(KST): `{publish_date_kst or 'N/A'}`")
    lines.append(f"- 발행 시각(KST): `{generated_kst or 'N/A'}`")
    lines.append(f"- 생성 시각(UTC): `{generated}`")
    lines.append(f"- 가격 기준일(미국장): `{data_asof}`")
    lines.append(f"- Flow/NAV 기준일: `{freshness.get('flow_nav_asof_date') or 'N/A'}`")
    lines.append(f"- 데이터 신선도: `{freshness.get('status')}` - {freshness.get('interpretation')}")
    lines.append(f"- 장기 Flow 감사: `{flow_audit.get('status')}` - {flow_audit.get('interpretation')}")
    lines.append(f"- DDM 입력 감사: `{ddm_audit.get('quality')}`")
    lines.append(
        f"- Barchart 옵션/선물 참고: `{options_ref.get('status')}` - "
        f"{options_ref.get('interpretation')}"
    )
    lines.append(f"- 옵션+DDM 해석: {options_ddm_explain.get('headline')}")
    lines.append(f"- 가격/flow 조회 범위: `{data_window.get('from')}` ~ `{data_window.get('to')}`")
    lines.append(
        f"- QQQ 최종 결정: **{qqq_decision.get('action', 'N/A')}** "
        f"(권장 exposure {qqq_decision.get('recommended_exposure_pct', 'N/A')}%)"
    )
    if tqqq_boost.get("enabled"):
        lines.append(
            f"- TQQQ boost: **{tqqq_boost.get('tqqq_boost_pct', 0)}%** "
            f"(QQQ alloc {tqqq_boost.get('qqq_alloc_pct', qqq_decision.get('recommended_exposure_pct', 0))}%, "
            f"effective beta {tqqq_eff_beta_text})"
        )
    lines.append(f"- QQQ 판단: {qqq_decision.get('headline', 'N/A')}")
    if ddm_signal.get("enabled"):
        lines.append(
            f"- DDM confidence: **{_fmt_pct(ddm_signal.get('confidence_pct'))}** "
            f"(`{ddm_signal.get('status', 'unknown')}`, evidence {_fmt_num(ddm_signal.get('evidence'), 2)})"
        )
    lines.append(
        f"- Swing GoStop: **{gostop.get('action', 'N/A')}** "
        f"({gostop.get('score', 'N/A')}/100, `{gostop.get('mode', 'unknown')}`)"
    )
    lines.append(f"- Swing 신규 한도: **{gostop.get('max_new_risk_pct', 'N/A')}%**")
    lines.append(f"- Swing 판단: {gostop.get('headline', 'N/A')}")
    if tactical.get("enabled"):
        lines.append(
            f"- Tactical overlay: **{tactical.get('action', 'N/A')}** "
            f"(탐색 한도 {tactical.get('max_tactical_risk_pct', 0)}%)"
        )
        lines.append(f"- Tactical 판단: {tactical.get('headline', 'N/A')}")
    lines.append(f"- 시장 상태: **{state.get('label', 'unknown')}**")
    lines.append(f"- Greed score: **{state.get('greed_score', 'N/A')}/100**")
    lines.append(f"- 시장 요약: {state.get('summary', 'N/A')}")
    lines.append("")

    lines.append("## Long-Term Flow Persistence Audit")
    lines.append("")
    qqq_flow = flow_audit.get("qqq") or {}
    qqq_rank = flow_audit.get("qqq_pre90_percentile_rank") or {}
    lines.append(f"- 판정: {flow_audit.get('interpretation')}")
    lines.append(
        "- QQQ latest flow: "
        f"**{_fmt_num(_to_float(qqq_flow.get('latest_flow')), 2)}** "
        f"({_fmt_pct(_to_float(qqq_flow.get('latest_flow_aum_pct')))} of AUM)"
    )
    lines.append(
        "- QQQ trailing flow: "
        f"5D {_fmt_num(_to_float(qqq_flow.get('flow_5d')), 2)} "
        f"({_fmt_pct(_to_float(qqq_flow.get('flow_5d_aum_pct')))}), "
        f"20D {_fmt_num(_to_float(qqq_flow.get('flow_20d')), 2)} "
        f"({_fmt_pct(_to_float(qqq_flow.get('flow_20d_aum_pct')))}), "
        f"60D {_fmt_num(_to_float(qqq_flow.get('flow_60d')), 2)} "
        f"({_fmt_pct(_to_float(qqq_flow.get('flow_60d_aum_pct')))})"
    )
    if qqq_rank:
        lines.append(
            "- QQQ pre-90 percentile rank: "
            f"5D {_fmt_pct(_to_float(qqq_rank.get('flow_5d_aum_pct')), 1)}, "
            f"20D {_fmt_pct(_to_float(qqq_rank.get('flow_20d_aum_pct')), 1)}, "
            f"60D {_fmt_pct(_to_float(qqq_rank.get('flow_60d_aum_pct')), 1)}"
        )
    if qqq_flow.get("consecutive_positive_5d_windows") is not None:
        lines.append(
            f"- QQQ positive 5D windows in a row: **{qqq_flow.get('consecutive_positive_5d_windows')}**"
        )
    contributors = flow_audit.get("top_risk_contributors") or []
    if contributors:
        lines.append("")
        lines.append("| Risk contributor | 5D Flow/AUM | Latest Flow/AUM | 5D Price |")
        lines.append("|---|---:|---:|---:|")
        for item in contributors[:5]:
            lines.append(
                f"| {item.get('symbol')} | {_fmt_pct(_to_float(item.get('flow_5d_aum_pct')))} "
                f"| {_fmt_pct(_to_float(item.get('latest_flow_aum_pct')))} "
                f"| {_fmt_pct(_to_float(item.get('price_5d_pct')))} |"
            )
    lines.append("")

    lines.append("## Drift-Diffusion Confidence")
    lines.append("")
    if ddm_signal.get("enabled"):
        lines.append(f"- Target: **{ddm_signal.get('target', 'QQQ')}**")
        lines.append(f"- Status: **{ddm_signal.get('status', 'unknown')}**")
        lines.append(
            f"- 입력 감사: **{ddm_audit.get('quality')}** "
            f"(price `{ddm_audit.get('price_asof_date')}`, flow `{ddm_audit.get('flow_nav_asof_date')}`)"
        )
        lines.append(f"- Confidence: **{_fmt_pct(ddm_signal.get('confidence_pct'))}**")
        lines.append(f"- Drift: **{_fmt_num(ddm_signal.get('drift'), 3)}**")
        lines.append(f"- Diffusion: **{_fmt_num(ddm_signal.get('diffusion'), 3)}**")
        lines.append(f"- Evidence ratio: **{_fmt_num(ddm_signal.get('evidence'), 3)}**")
        lines.append(f"- Agreement: **{_fmt_pct(ddm_signal.get('agreement_pct'))}**")
        lines.append(
            f"- Correlated basket: **{ddm_signal.get('correlated_count', 0)}개** "
            f"(high corr {ddm_signal.get('high_corr_count', 0)}개)"
        )
        lines.append(f"- Support pressure: **{_fmt_num(ddm_signal.get('support_pressure'), 3)}**")
        lines.append(f"- Resistance pressure: **{_fmt_num(ddm_signal.get('resistance_pressure'), 3)}**")
        lines.append("")
        lines.append("### DDM 해석")
        lines.append("")
        lines.append(f"- {options_ddm_explain.get('ddm_read')}")
        lines.append(f"- {options_ddm_explain.get('fusion_read')}")
        ddm_support = ddm_signal.get("support") or []
        ddm_resistance = ddm_signal.get("resistance") or []
        if ddm_support or ddm_resistance:
            lines.append("")
            lines.append("| Side | ETF | Corr | 5D Price | Flow/AUM | Day | Signed Pressure |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
            for side, items in (("support", ddm_support[:5]), ("resistance", ddm_resistance[:5])):
                for item in items:
                    lines.append(
                        "| `{}` | `{}` | {} | {} | {} | {} | {} |".format(
                            side,
                            item.get("symbol"),
                            _fmt_num(item.get("corr"), 2),
                            _fmt_pct(item.get("price_5d_pct")),
                            _fmt_pct(item.get("flow_aum_5d_pct"), 3),
                            _fmt_pct(item.get("day_change_pct")),
                            _fmt_num(item.get("signed_pressure"), 3),
                        )
                    )
    else:
        lines.append(f"- {ddm_signal.get('note') or 'DDM 계산 데이터가 부족합니다.'}")
    lines.append("")

    lines.append("## QQQ 최종 결정")
    lines.append("")
    lines.append(f"- Target: **{qqq_decision.get('target', 'QQQ')}**")
    lines.append(f"- Decision: **{qqq_decision.get('action', 'N/A')}**")
    lines.append(f"- Recommended exposure: **{qqq_decision.get('recommended_exposure_pct', 'N/A')}%**")
    lines.append(f"- Base policy: **{qqq_decision.get('base_policy', 'N/A')}**")
    lines.append(f"- Buy&Hold base exposure: **{qqq_decision.get('buy_hold_base_exposure_pct', 'N/A')}%**")
    lines.append(f"- Learned reference exposure: **{qqq_decision.get('learned_reference_exposure_pct', 'N/A')}%**")
    lines.append(f"- Swing gate: **{qqq_decision.get('swing_action', 'N/A')}**")
    lines.append(f"- Tactical action: **{qqq_decision.get('tactical_action') or 'N/A'}**")
    lines.append(f"- Adaptive policy key: **{qqq_decision.get('policy_key', 'N/A')}**")
    risk_overlay = qqq_decision.get("ddm_risk_overlay") or {}
    if qqq_decision.get("ddm_risk_overlay_applied"):
        lines.append(
            f"- DDM risk overlay: **{risk_overlay.get('status', 'unknown')} / "
            f"{risk_overlay.get('severity', 'none')}**, multiplier **{_fmt_pct((risk_overlay.get('multiplier') or 0) * 100, 0)}**"
        )
    lines.append(
        "- Adaptive 적용: **{}**".format(
            "YES" if qqq_decision.get("adaptive_policy_applied") else "NO"
        )
    )
    if qqq_decision.get("adaptive_policy_applied"):
        lines.append(
            f"- Adaptive raw/cap: **{qqq_decision.get('adaptive_raw_exposure_pct', 'N/A')}% / "
            f"{qqq_decision.get('adaptive_context_cap_pct', 'N/A')}%**"
        )
    pattern_decision = qqq_decision.get("pattern_policy_decision") or {}
    if pattern_decision:
        lines.append(
            f"- Similarity policy: **{pattern_decision.get('source', 'N/A')}**, "
            f"neighbors **{pattern_decision.get('neighbors', 'N/A')}**, "
            f"exposure **{_fmt_pct(pattern_decision.get('exposure_pct'), 0)}**"
        )
    lines.append(f"- 평가 규칙: {qqq_decision.get('evaluation_rule', 'N/A')}")
    if qqq_decision.get("reasons"):
        lines.append("")
        lines.append("### QQQ 진입 근거")
        lines.append("")
        for reason in qqq_decision.get("reasons") or []:
            lines.append(f"- {reason}")
    if qqq_decision.get("blocks"):
        lines.append("")
        lines.append("### QQQ 제한 요인")
        lines.append("")
        for block in qqq_decision.get("blocks") or []:
            lines.append(f"- {block}")
    lines.append("")

    lines.append("## TQQQ Boost Sleeve")
    lines.append("")
    if tqqq_boost.get("enabled"):
        lines.append(f"- Source: **{tqqq_boost.get('source', 'N/A')}**")
        lines.append(f"- TQQQ boost: **{_fmt_pct(tqqq_boost.get('tqqq_boost_pct'), 0)}**")
        lines.append(f"- QQQ allocation after boost: **{_fmt_pct(tqqq_boost.get('qqq_alloc_pct'), 0)}**")
        eff_beta = _to_float(tqqq_boost.get("effective_beta"))
        lines.append(f"- Effective QQQ beta: **{eff_beta:.2f}**" if eff_beta is not None else "- Effective QQQ beta: **N/A**")
        lines.append(f"- Context cap: **{_fmt_pct(tqqq_boost.get('context_cap_pct'), 0)}**")
        lines.append(f"- Similar neighbors: **{tqqq_boost.get('neighbors', 'N/A')}**")
        ddm_gate = tqqq_boost.get("ddm_gate") or {}
        if ddm_gate:
            lines.append(
                "- DDM gate: "
                f"**{ddm_gate.get('status', 'unknown')}**, "
                f"cap **{_fmt_pct(ddm_gate.get('base_cap_pct'), 0)} -> {_fmt_pct(ddm_gate.get('adjusted_cap_pct'), 0)}**, "
                f"confidence **{_fmt_pct(ddm_gate.get('confidence_pct'))}**"
            )
            if ddm_gate.get("reason"):
                lines.append(f"- DDM gate reason: {ddm_gate.get('reason')}")
        selected_boost = tqqq_boost.get("selected") or {}
        if selected_boost:
            lines.append(f"- Neighbor avg portfolio return: **{_fmt_pct(selected_boost.get('avg_portfolio_return_pct'))}**")
            lines.append(f"- Neighbor boost excess vs QQQ strategy: **{_fmt_pct(selected_boost.get('avg_boost_excess_pct'))}**")
            lines.append(f"- Neighbor downside deviation: **{_fmt_pct(selected_boost.get('downside_deviation_pct'))}**")
            lines.append(f"- Neighbor worst return: **{_fmt_pct(selected_boost.get('worst_return_pct'))}**")
        if tqqq_boost.get("candidates"):
            lines.append("")
            lines.append("| TQQQ Boost | Eff Beta | Avg Port | Boost Excess | vs QQQ Hold | Downside | Worst | Hit |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for item in tqqq_boost.get("candidates") or []:
                eff = _to_float(item.get("effective_beta"))
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        _fmt_pct(item.get("tqqq_boost_pct"), 0),
                        f"{eff:.2f}" if eff is not None else "N/A",
                        _fmt_pct(item.get("avg_portfolio_return_pct")),
                        _fmt_pct(item.get("avg_boost_excess_pct")),
                        _fmt_pct(item.get("avg_vs_qqq_buy_hold_pct")),
                        _fmt_pct(item.get("downside_deviation_pct")),
                        _fmt_pct(item.get("worst_return_pct")),
                        _fmt_pct(item.get("hit_rate_pct")),
                    )
                )
    else:
        lines.append("- TQQQ boost가 비활성화되어 있습니다.")
    lines.append("")

    lines.append("## GoStop 판정")
    lines.append("")
    lines.append(f"- Action: **{gostop.get('action', 'N/A')}**")
    lines.append(f"- Mode: `{gostop.get('mode', 'unknown')}`")
    lines.append(f"- Max new risk: **{gostop.get('max_new_risk_pct', 'N/A')}%**")
    entry_candidates = gostop.get("entry_candidates") or []
    stop_candidates = gostop.get("stop_candidates") or []
    lines.append(f"- 후보로 볼 수 있는 쪽: `{', '.join(entry_candidates) if entry_candidates else 'N/A'}`")
    lines.append(f"- 멈추거나 추격 금지할 쪽: `{', '.join(stop_candidates) if stop_candidates else 'N/A'}`")
    if gostop.get("reasons"):
        lines.append("")
        lines.append("### GO 근거")
        lines.append("")
        for reason in gostop.get("reasons") or []:
            lines.append(f"- {reason}")
    if gostop.get("blocks"):
        lines.append("")
        lines.append("### STOP/WAIT 근거")
        lines.append("")
        for block in gostop.get("blocks") or []:
            lines.append(f"- {block}")
    lines.append("")

    lines.append("## Tactical Overlay")
    lines.append("")
    if tactical:
        lines.append(f"- Action: **{tactical.get('action', 'N/A')}**")
        lines.append(f"- Swing source-of-truth: **{tactical.get('swing_action', gostop.get('action', 'N/A'))}**")
        lines.append(f"- Tactical probe max: **{tactical.get('max_tactical_risk_pct', 0)}%**")
        lines.append(f"- 판단: {tactical.get('headline', 'N/A')}")
        if tactical.get("reasons"):
            lines.append("")
            lines.append("### Tactical 확인")
            lines.append("")
            for item in tactical.get("reasons") or []:
                lines.append(f"- {item}")
        if tactical.get("blocks"):
            lines.append("")
            lines.append("### Tactical 제한")
            lines.append("")
            for item in tactical.get("blocks") or []:
                lines.append(f"- {item}")
    else:
        lines.append("- Tactical overlay가 없습니다.")
    lines.append("")

    lines.append("## NowCast")
    lines.append("")
    if nowcast.get("enabled"):
        lines.append(f"- Status: **{nowcast.get('status', 'unknown')}**")
        lines.append(f"- Score: **{nowcast.get('score', 'N/A')}/100**")
        lines.append(f"- 판단: {nowcast.get('stance', 'N/A')}")
        lines.append(f"- Risk ETF 당일 평균: **{_fmt_pct(nowcast.get('risk_day_avg_pct'))}**")
        lines.append(
            f"- Risk ETF 상승 확산: **{nowcast.get('risk_positive_count', 'N/A')}/"
            f"{nowcast.get('risk_count', 'N/A')}**"
        )
        lines.append(f"- Risk ETF 50일선 상회: **{nowcast.get('risk_above_sma50_count', 'N/A')}개**")
        rel_vol = nowcast.get("risk_rel_volume_median")
        lines.append(f"- Risk ETF 거래량 중앙값: **{rel_vol:.2f}x**" if rel_vol is not None else "- Risk ETF 거래량 중앙값: **N/A**")
        if nowcast.get("reasons"):
            lines.append("")
            lines.append("### NowCast 확인")
            lines.append("")
            for item in nowcast.get("reasons") or []:
                lines.append(f"- {item}")
        if nowcast.get("warnings"):
            lines.append("")
            lines.append("### NowCast 경고")
            lines.append("")
            for item in nowcast.get("warnings") or []:
                lines.append(f"- {item}")
        lines.append("")
        lines.append("| Leader | 1D | Rel Vol | vs 20D | vs 50D | Signal |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for row in nowcast.get("leaders") or []:
            lines.append(
                "| `{}` | {} | {} | {} | {} | `{}` |".format(
                    row.get("symbol"),
                    _fmt_pct(row.get("day_change_pct")),
                    f"{row.get('rel_volume'):.2f}x" if row.get("rel_volume") is not None else "N/A",
                    _fmt_pct(row.get("above_sma20_pct")),
                    _fmt_pct(row.get("above_sma50_pct")),
                    row.get("signal"),
                )
            )
    else:
        lines.append(f"- {nowcast.get('note') or 'NowCast가 비활성화되어 있습니다.'}")
    lines.append("")

    lines.append("## 자동 튜닝")
    lines.append("")
    if auto_tune.get("enabled"):
        selected = auto_tune.get("selected") or {}
        lines.append(f"- Status: **{auto_tune.get('status', 'unknown')}**")
        lines.append(f"- Confidence: **{auto_tune.get('confidence', 'unknown')}**")
        lines.append(f"- 평가 기간: **최근 {auto_tune.get('history_days')}일**")
        lines.append(
            "- 선택 horizon: "
            f"action **{selected.get('action_horizon_bars')}일**, "
            f"tactical **{selected.get('tactical_horizon_bars')}일**, "
            f"entry **{selected.get('entry_horizon_bars')}일**, "
            f"stop **{selected.get('stop_horizon_bars')}일**"
        )
        lines.append(f"- 추천 entry score 최소값: **{selected.get('entry_score_min')}**")
        lines.append(f"- 튜닝 기준 신규 리스크 한도: **{selected.get('tuned_max_new_risk_pct')}%**")
        if selected.get("avg_qqq_decision_utility_pct") is not None:
            lines.append(f"- QQQ 결정 평균 utility: **{_fmt_pct(selected.get('avg_qqq_decision_utility_pct'))}**")
        if selected.get("avg_qqq_strategy_return_pct") is not None:
            lines.append(f"- QQQ exposure 전략 평균 수익: **{_fmt_pct(selected.get('avg_qqq_strategy_return_pct'))}**")
        if selected.get("walk_forward_hit_rate_pct") is not None:
            lines.append(f"- Walk-forward adaptive hit: **{_fmt_pct(selected.get('walk_forward_hit_rate_pct'))}**")
        if selected.get("avg_walk_forward_strategy_return_pct") is not None:
            lines.append(f"- Walk-forward adaptive 전략 수익: **{_fmt_pct(selected.get('avg_walk_forward_strategy_return_pct'))}**")
        if selected.get("avg_walk_forward_vs_buy_hold_pct") is not None:
            lines.append(f"- Walk-forward adaptive vs Buy&Hold: **{_fmt_pct(selected.get('avg_walk_forward_vs_buy_hold_pct'))}**")
        if selected.get("avg_boosted_strategy_return_pct") is not None:
            lines.append(f"- TQQQ boost 적용 후 전략 수익: **{_fmt_pct(selected.get('avg_boosted_strategy_return_pct'))}**")
        if selected.get("avg_boosted_vs_buy_hold_pct") is not None:
            lines.append(f"- TQQQ boost 적용 후 vs Buy&Hold: **{_fmt_pct(selected.get('avg_boosted_vs_buy_hold_pct'))}**")
        if selected.get("avg_boost_excess_vs_qqq_strategy_pct") is not None:
            lines.append(f"- TQQQ boost 초과수익 vs QQQ 전략: **{_fmt_pct(selected.get('avg_boost_excess_vs_qqq_strategy_pct'))}**")
        if selected.get("avg_tactical_utility_pct") is not None:
            lines.append(f"- Tactical 평균 utility: **{_fmt_pct(selected.get('avg_tactical_utility_pct'))}**")
        pattern_decision = selected.get("pattern_adaptive_decision") or auto_tune.get("pattern_adaptive_decision") or {}
        if pattern_decision:
            lines.append(
                "- Current similarity policy: "
                f"source **{pattern_decision.get('source', 'N/A')}**, "
                f"neighbors **{pattern_decision.get('neighbors', 'N/A')}**, "
                f"exposure **{_fmt_pct(pattern_decision.get('exposure_pct'), 0)}**"
            )
        walk_forward = selected.get("walk_forward_adaptive") or auto_tune.get("walk_forward_adaptive") or {}
        if walk_forward.get("enabled"):
            lines.append("")
            lines.append("### Walk-Forward Adaptive 평가")
            lines.append("")
            lines.append(f"- Warmup signals: **{walk_forward.get('warmup_signals')}**")
            lines.append(f"- Learned signals: **{walk_forward.get('learned_signals')}/{walk_forward.get('evaluated_signals')}**")
            lines.append(f"- Similarity signals: **{walk_forward.get('similarity_signals')}**")
            lines.append(f"- DDM risk overlay signals: **{walk_forward.get('ddm_overlay_signals')}**")
            lines.append(f"- Hit: **{_fmt_pct(walk_forward.get('hit_rate_pct'))}**")
            lines.append(f"- Strategy: **{_fmt_pct(walk_forward.get('avg_strategy_return_pct'))}**")
            lines.append(f"- vs Buy&Hold: **{_fmt_pct(walk_forward.get('avg_vs_buy_hold_pct'))}**")
            lines.append(f"- Missed up: **{_fmt_pct(walk_forward.get('avg_missed_upside_pct'))}**")
            lines.append(f"- Drawdown avoided: **{_fmt_pct(walk_forward.get('avg_drawdown_avoided_pct'))}**")
            lines.append(f"- Risk reduction signals: **{walk_forward.get('risk_reduction_signals')}**")
            lines.append(f"- Risk reduction hit: **{_fmt_pct(walk_forward.get('risk_reduction_hit_rate_pct'))}**")
            lines.append(f"- Risk reduction excess vs Buy&Hold: **{_fmt_pct(walk_forward.get('avg_risk_reduction_excess_pct'))}**")
            lines.append(f"- Reduction missed up: **{_fmt_pct(walk_forward.get('avg_risk_reduction_missed_upside_pct'))}**")
            lines.append(f"- Reduction drawdown avoided: **{_fmt_pct(walk_forward.get('avg_risk_reduction_drawdown_avoided_pct'))}**")
            breakdown = walk_forward.get("risk_reduction_breakdown") or []
            if breakdown:
                lines.append("")
                lines.append("#### Risk Reduction Precision Breakdown")
                lines.append("")
                lines.append("| Bucket | Signals | Hit | Excess | Missed Up | Drawdown Avoided |")
                lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
                for item in breakdown[:6]:
                    lines.append(
                        "| `{}` | {} | {} | {} | {} | {} |".format(
                            item.get("bucket"),
                            item.get("signals"),
                            _fmt_pct(item.get("hit_rate_pct")),
                            _fmt_pct(item.get("avg_excess_pct")),
                            _fmt_pct(item.get("avg_missed_upside_pct")),
                            _fmt_pct(item.get("avg_drawdown_avoided_pct")),
                        )
                    )
            lines.append(f"- Boost signals: **{walk_forward.get('boost_signals')}**")
            lines.append(f"- Boost hit: **{_fmt_pct(walk_forward.get('boost_hit_rate_pct'))}**")
            lines.append(f"- Avg TQQQ boost: **{_fmt_pct(walk_forward.get('avg_tqqq_boost_pct'), 0)}**")
            lines.append(f"- Boosted strategy: **{_fmt_pct(walk_forward.get('avg_boosted_strategy_return_pct'))}**")
            lines.append(f"- Boosted vs Buy&Hold: **{_fmt_pct(walk_forward.get('avg_boosted_vs_buy_hold_pct'))}**")
            lines.append(f"- Boost excess vs QQQ strategy: **{_fmt_pct(walk_forward.get('avg_boost_excess_vs_qqq_strategy_pct'))}**")
            lines.append(f"- Active boost excess: **{_fmt_pct(walk_forward.get('avg_active_boost_excess_pct'))}**")
            lines.append(f"- Boosted downside deviation: **{_fmt_pct(walk_forward.get('boosted_downside_deviation_pct'))}**")
            lines.append(f"- DDM enabled signals: **{walk_forward.get('ddm_enabled_signals')}**")
            lines.append(f"- Avg DDM confidence: **{_fmt_pct(walk_forward.get('avg_ddm_confidence_pct'))}**")
            lines.append(f"- Avg DDM evidence: **{_fmt_num(walk_forward.get('avg_ddm_evidence'), 3)}**")
        adaptive_policy = selected.get("adaptive_exposure_policy") or auto_tune.get("adaptive_exposure_policy") or {}
        buckets = adaptive_policy.get("buckets") or {}
        if buckets:
            aggregate = adaptive_policy.get("aggregate") or {}
            if aggregate:
                lines.append("")
                lines.append(
                    "- Adaptive policy 인샘플 참고 성과: "
                    f"Hit **{_fmt_pct(aggregate.get('hit_rate_pct'))}**, "
                    f"Strategy **{_fmt_pct(aggregate.get('avg_strategy_return_pct'))}**, "
                    f"vs Buy&Hold **{_fmt_pct(aggregate.get('avg_vs_buy_hold_pct'))}**, "
                    f"Missed up **{_fmt_pct(aggregate.get('avg_missed_upside_pct'))}**, "
                    f"Drawdown avoided **{_fmt_pct(aggregate.get('avg_drawdown_avoided_pct'))}**"
                )
            lines.append("")
            lines.append("### Adaptive QQQ Exposure Policy")
            lines.append("")
            lines.append("| Signal Bucket | N | Exposure | Hit | Strategy | Missed Up | Drawdown Avoided | Confidence |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            preferred = ["GO", "GO_SMALL", "TACTICAL_GO_SMALL", "WATCH_REBOUND", "WAIT", "STOP", "DELAY_ENTRY"]
            ordered_keys = [key for key in preferred if key in buckets] + [
                key for key in sorted(buckets) if key not in preferred
            ]
            for key in ordered_keys[:10]:
                bucket = buckets.get(key) or {}
                selected_bucket = bucket.get("selected") or {}
                lines.append(
                    "| `{}` | {} | {} | {} | {} | {} | {} | `{}` |".format(
                        key,
                        bucket.get("n"),
                        _fmt_pct(bucket.get("selected_exposure_pct"), 0),
                        _fmt_pct(selected_bucket.get("hit_rate_pct")),
                        _fmt_pct(selected_bucket.get("avg_strategy_return_pct")),
                        _fmt_pct(selected_bucket.get("avg_missed_upside_pct")),
                        _fmt_pct(selected_bucket.get("avg_drawdown_avoided_pct")),
                        bucket.get("confidence", "unknown"),
                    )
                )
        tuned_entries = auto_tune.get("tuned_entry_candidates") or []
        tuned_stops = auto_tune.get("tuned_stop_candidates") or []
        lines.append(f"- 튜닝 entry 후보: `{', '.join(tuned_entries) if tuned_entries else 'N/A'}`")
        lines.append(f"- 튜닝 stop/avoid 후보: `{', '.join(tuned_stops) if tuned_stops else 'N/A'}`")
        entry_signals = auto_tune.get("recommended_entry_signals") or []
        stop_signals = auto_tune.get("recommended_stop_signals") or []
        if entry_signals:
            lines.append("")
            lines.append("### Entry에 유리했던 신호")
            lines.append("")
            lines.append("| Signal | N | Entry vs QQQ | Hit |")
            lines.append("| --- | ---: | ---: | ---: |")
            for item in entry_signals:
                lines.append(
                    "| `{}` | {} | {} | {} |".format(
                        item.get("signal"),
                        item.get("n"),
                        _fmt_pct(item.get("avg_entry_excess_qqq_pct")),
                        _fmt_pct(item.get("entry_hit_rate_pct")),
                    )
                )
        if stop_signals:
            lines.append("")
            lines.append("### Stop/Avoid에 유리했던 신호")
            lines.append("")
            lines.append("| Signal | N | Stop Utility | Hit |")
            lines.append("| --- | ---: | ---: | ---: |")
            for item in stop_signals:
                lines.append(
                    "| `{}` | {} | {} | {} |".format(
                        item.get("signal"),
                        item.get("n"),
                        _fmt_pct(item.get("avg_stop_utility_pct")),
                        _fmt_pct(item.get("stop_hit_rate_pct")),
                    )
                )
        summaries = auto_tune.get("horizon_summaries") or []
        if summaries:
            lines.append("")
            lines.append("### Horizon별 튜닝 결과")
            lines.append("")
            lines.append("| Horizon | N | WF Hit | WF Strat | WF vs Hold | Boosted Strat | Boost Excess | QQQ Hit | QQQ Strat | Tactical Utility | Entry vs QQQ | Stop Utility | Score Min |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for item in summaries:
                threshold = item.get("best_score_threshold") or {}
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        item.get("horizon_bars"),
                        item.get("evaluated_signals"),
                        _fmt_pct(item.get("walk_forward_hit_rate_pct")),
                        _fmt_pct(item.get("avg_walk_forward_strategy_return_pct")),
                        _fmt_pct(item.get("avg_walk_forward_vs_buy_hold_pct")),
                        _fmt_pct(item.get("avg_boosted_strategy_return_pct")),
                        _fmt_pct(item.get("avg_boost_excess_vs_qqq_strategy_pct")),
                        _fmt_pct(item.get("qqq_decision_hit_rate_pct")),
                        _fmt_pct(item.get("avg_qqq_strategy_return_pct")),
                        _fmt_pct(item.get("avg_tactical_utility_pct")),
                        _fmt_pct(item.get("avg_entry_excess_qqq_pct")),
                        _fmt_pct(item.get("avg_stop_utility_pct")),
                        threshold.get("entry_score_min", "N/A"),
                    )
                )
    else:
        lines.append("- 자동 튜닝이 비활성화되어 있습니다.")
    lines.append("")

    lines.append("## QQQ 의사결정 성과 평가")
    lines.append("")
    evaluated = performance.get("evaluated_reports") or 0
    if evaluated:
        lines.append(f"- 평가된 과거 GoStop 리포트: **{evaluated}개**")
        lines.append(f"- 현재 가격 기준일: `{performance.get('current_price_date')}`")
        lines.append(f"- QQQ decision hit rate: **{_fmt_pct(performance.get('qqq_decision_hit_rate_pct'))}**")
        lines.append(f"- 평균 QQQ decision utility: **{_fmt_pct(performance.get('avg_qqq_decision_utility_pct'))}**")
        lines.append(f"- 평균 QQQ exposure 전략 수익: **{_fmt_pct(performance.get('avg_qqq_strategy_return_pct'))}**")
        lines.append(f"- QQQ buy&hold 대비: **{_fmt_pct(performance.get('avg_qqq_vs_buy_hold_pct'))}**")
        lines.append(f"- 평균 missed upside: **{_fmt_pct(performance.get('avg_missed_upside_pct'))}**")
        lines.append(f"- 평균 drawdown avoided: **{_fmt_pct(performance.get('avg_drawdown_avoided_pct'))}**")
        lines.append("")
        lines.append("### 보조 ETF/avoid 평가")
        lines.append("")
        lines.append(f"- Entry basket 평균 QQQ 초과: **{_fmt_pct(performance.get('avg_entry_excess_qqq_pct'))}**")
        lines.append(f"- Stop basket 회피 utility: **{_fmt_pct(performance.get('avg_stop_utility_pct'))}**")
        lines.append("")
        lines.append("| Signal Date | Age | QQQ Decision | Exposure | QQQ | Strategy | Missed Up | Drawdown Avoided | Hit |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for item in performance.get("recent") or []:
            lines.append(
                "| {} | {}d | `{}` | {} | {} | {} | {} | {} | {} |".format(
                    item.get("date"),
                    item.get("age_days"),
                    item.get("qqq_decision") or item.get("action"),
                    _fmt_pct(item.get("qqq_exposure_pct"), 0),
                    _fmt_pct(item.get("qqq_return_pct")),
                    _fmt_pct(item.get("qqq_strategy_return_pct")),
                    _fmt_pct(item.get("missed_upside_pct")),
                    _fmt_pct(item.get("drawdown_avoided_pct")),
                    "HIT" if item.get("action_hit") else "MISS",
                )
            )
        correlations = performance.get("rolling_correlations") or []
        lines.append("")
        if correlations:
            lines.append("### Rolling Feature Correlation")
            lines.append("")
            lines.append("| Feature | Target | Corr | N |")
            lines.append("| --- | --- | ---: | ---: |")
            for item in correlations[:8]:
                corr = item.get("corr")
                lines.append(
                    "| `{}` | `{}` | {} | {} |".format(
                        item.get("feature"),
                        item.get("target"),
                        f"{corr:+.2f}" if corr is not None else "N/A",
                        item.get("n"),
                    )
                )
        else:
            lines.append("- Rolling correlation은 최소 5개 이상 평가 샘플이 쌓이면 표시됩니다.")
    else:
        lines.append(f"- {performance.get('note') or '아직 평가할 과거 GoStop JSON이 없습니다.'}")
    lines.append("")

    lines.append("## 과거 재계산 백테스트")
    lines.append("")
    evaluated_signals = backtest.get("evaluated_signals") or 0
    if evaluated_signals:
        lines.append(f"- 재계산 신호 수: **{evaluated_signals}개**")
        lines.append(f"- 평가 기간: **최근 {backtest.get('history_days')}일**")
        lines.append(f"- 평가 horizon: **{backtest.get('horizon_bars')}거래일**")
        lines.append(f"- QQQ decision hit rate: **{_fmt_pct(backtest.get('qqq_decision_hit_rate_pct'))}**")
        lines.append(f"- 평균 QQQ decision utility: **{_fmt_pct(backtest.get('avg_qqq_decision_utility_pct'))}**")
        lines.append(f"- 평균 QQQ exposure 전략 수익: **{_fmt_pct(backtest.get('avg_qqq_strategy_return_pct'))}**")
        lines.append(f"- QQQ buy&hold 대비: **{_fmt_pct(backtest.get('avg_qqq_vs_buy_hold_pct'))}**")
        lines.append(f"- 평균 missed upside: **{_fmt_pct(backtest.get('avg_missed_upside_pct'))}**")
        lines.append(f"- 평균 drawdown avoided: **{_fmt_pct(backtest.get('avg_drawdown_avoided_pct'))}**")
        lines.append(f"- Tactical hit rate: **{_fmt_pct(backtest.get('tactical_hit_rate_pct'))}**")
        lines.append(f"- 평균 tactical utility: **{_fmt_pct(backtest.get('avg_tactical_utility_pct'))}**")
        lines.append(f"- Entry basket 평균 QQQ 초과: **{_fmt_pct(backtest.get('avg_entry_excess_qqq_pct'))}**")
        lines.append(f"- Stop basket 회피 utility: **{_fmt_pct(backtest.get('avg_stop_utility_pct'))}**")
        walk_forward = backtest.get("walk_forward_adaptive") or {}
        if walk_forward.get("enabled"):
            lines.append("")
            lines.append("### Walk-Forward Adaptive")
            lines.append("")
            lines.append(f"- Warmup signals: **{walk_forward.get('warmup_signals')}**")
            lines.append(f"- Learned signals: **{walk_forward.get('learned_signals')}/{walk_forward.get('evaluated_signals')}**")
            lines.append(f"- Similarity signals: **{walk_forward.get('similarity_signals')}**")
            lines.append(f"- DDM risk overlay signals: **{walk_forward.get('ddm_overlay_signals')}**")
            lines.append(f"- Hit: **{_fmt_pct(walk_forward.get('hit_rate_pct'))}**")
            lines.append(f"- Strategy: **{_fmt_pct(walk_forward.get('avg_strategy_return_pct'))}**")
            lines.append(f"- vs Buy&Hold: **{_fmt_pct(walk_forward.get('avg_vs_buy_hold_pct'))}**")
            lines.append(f"- Missed up: **{_fmt_pct(walk_forward.get('avg_missed_upside_pct'))}**")
            lines.append(f"- Drawdown avoided: **{_fmt_pct(walk_forward.get('avg_drawdown_avoided_pct'))}**")
            lines.append(f"- Risk reduction signals: **{walk_forward.get('risk_reduction_signals')}**")
            lines.append(f"- Risk reduction hit: **{_fmt_pct(walk_forward.get('risk_reduction_hit_rate_pct'))}**")
            lines.append(f"- Risk reduction excess vs Buy&Hold: **{_fmt_pct(walk_forward.get('avg_risk_reduction_excess_pct'))}**")
            lines.append(f"- Reduction missed up: **{_fmt_pct(walk_forward.get('avg_risk_reduction_missed_upside_pct'))}**")
            lines.append(f"- Reduction drawdown avoided: **{_fmt_pct(walk_forward.get('avg_risk_reduction_drawdown_avoided_pct'))}**")
            lines.append(f"- Boost signals: **{walk_forward.get('boost_signals')}**")
            lines.append(f"- Boost hit: **{_fmt_pct(walk_forward.get('boost_hit_rate_pct'))}**")
            lines.append(f"- Avg TQQQ boost: **{_fmt_pct(walk_forward.get('avg_tqqq_boost_pct'), 0)}**")
            lines.append(f"- Boosted strategy: **{_fmt_pct(walk_forward.get('avg_boosted_strategy_return_pct'))}**")
            lines.append(f"- Boosted vs Buy&Hold: **{_fmt_pct(walk_forward.get('avg_boosted_vs_buy_hold_pct'))}**")
            lines.append(f"- Boost excess vs QQQ strategy: **{_fmt_pct(walk_forward.get('avg_boost_excess_vs_qqq_strategy_pct'))}**")
            lines.append(f"- Active boost excess: **{_fmt_pct(walk_forward.get('avg_active_boost_excess_pct'))}**")
            lines.append(f"- Boosted downside deviation: **{_fmt_pct(walk_forward.get('boosted_downside_deviation_pct'))}**")
            lines.append(f"- DDM enabled signals: **{walk_forward.get('ddm_enabled_signals')}**")
            lines.append(f"- Avg DDM confidence: **{_fmt_pct(walk_forward.get('avg_ddm_confidence_pct'))}**")
            lines.append(f"- Avg DDM evidence: **{_fmt_num(walk_forward.get('avg_ddm_evidence'), 3)}**")
            if walk_forward.get("recent"):
                lines.append("")
                lines.append("| Date | Bucket | Source | Learned | Final | TQQQ Boost | DDM | QQQ | Strategy | Risk Excess | Boosted | Boost Excess | Hit |")
                lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
                for item in walk_forward.get("recent") or []:
                    lines.append(
                        "| {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                            item.get("date"),
                            item.get("policy_key"),
                            item.get("source"),
                            _fmt_pct(item.get("learned_exposure_pct"), 0),
                            _fmt_pct(item.get("exposure_pct"), 0),
                            _fmt_pct(item.get("tqqq_boost_pct"), 0),
                            _fmt_pct(item.get("ddm_confidence_pct")),
                            _fmt_pct(item.get("qqq_return_pct")),
                            _fmt_pct(item.get("strategy_return_pct")),
                            _fmt_pct(item.get("risk_reduction_excess_pct")),
                            _fmt_pct(item.get("boosted_strategy_return_pct")),
                            _fmt_pct(item.get("boost_excess_pct")),
                            "HIT" if item.get("hit") else "MISS",
                        )
                    )
        lines.append("")
        lines.append("| Signal Date | QQQ Decision | Exposure | Swing | Tactical | Score | QQQ Forward | Strategy | Missed Up | Drawdown Avoided | Hit |")
        lines.append("| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for item in backtest.get("recent") or []:
            lines.append(
                "| {} | `{}` | {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} |".format(
                    item.get("date"),
                    item.get("qqq_decision") or item.get("action"),
                    _fmt_pct(item.get("qqq_exposure_pct"), 0),
                    item.get("action"),
                    item.get("tactical_action"),
                    item.get("score"),
                    _fmt_pct(item.get("qqq_return_pct")),
                    _fmt_pct(item.get("qqq_strategy_return_pct")),
                    _fmt_pct(item.get("missed_upside_pct")),
                    _fmt_pct(item.get("drawdown_avoided_pct")),
                    "HIT" if item.get("action_hit") else "MISS",
                )
            )
        correlations = backtest.get("rolling_correlations") or []
        lines.append("")
        if correlations:
            lines.append("### Recomputed Rolling Correlation")
            lines.append("")
            lines.append("| Feature | Target | Corr | N |")
            lines.append("| --- | --- | ---: | ---: |")
            for item in correlations[:8]:
                corr = item.get("corr")
                lines.append(
                    "| `{}` | `{}` | {} | {} |".format(
                        item.get("feature"),
                        item.get("target"),
                        f"{corr:+.2f}" if corr is not None else "N/A",
                        item.get("n"),
                    )
                )
        else:
            lines.append("- Rolling correlation은 최소 5개 이상 재계산 샘플이 쌓이면 표시됩니다.")
    else:
        lines.append(f"- {backtest.get('note') or '재계산 백테스트가 비활성화되어 있습니다.'}")
    lines.append("")

    risk_price = _fmt_pct(state.get("risk_price_5d_avg_pct"))
    risk_flow = _fmt_pct(state.get("risk_flow_5d_aum_avg_pct"), 3)
    nav_gap = _fmt_pct(state.get("risk_nav_gap_avg_pct"), 3)
    defensive = _fmt_pct(state.get("defensive_flow_5d_aum_avg_pct"), 3)
    lines.append("## 핵심 계기판")
    lines.append("")
    lines.append(f"- Risk ETF 평균 5일 가격 변화: **{risk_price}**")
    lines.append(f"- Risk ETF 평균 5일 flow/AUM: **{risk_flow}**")
    lines.append(f"- Risk ETF 평균 NAV 괴리: **{nav_gap}**")
    lines.append(f"- Defensive ETF 평균 5일 flow/AUM: **{defensive}**")
    lines.append(f"- 강한 쪽: `{', '.join(state.get('strongest') or [])}`")
    lines.append(f"- 약한 쪽: `{', '.join(state.get('weakest') or [])}`")
    lines.append("")

    interesting = sorted(rows, key=_row_sort_key, reverse=True)[:12]
    lines.append("## ETF 신호 Top")
    lines.append("")
    lines.append("| ETF | Signal | 5D 가격 | 5D flow | 5D flow/AUM | NAV 괴리 | Shares 5D | 해석 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in interesting:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | {} |".format(
                row.symbol,
                row.signal,
                _fmt_pct(row.price_5d_pct),
                _fmt_num(row.fund_flow_5d),
                _fmt_pct(row.flow_aum_5d_pct, 3),
                _fmt_pct(row.nav_gap_pct, 3),
                _fmt_pct(row.shares_change_5d_pct, 3),
                row.nuance,
            )
        )
    lines.append("")

    lines.append("## 전체 ETF 테이블")
    lines.append("")
    lines.append("| ETF | Price Date | Flow Date | Price | 1D | 5D | 20D | Latest Flow | Flow z | NAV | NAV Gap | Warning |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.symbol,
                row.price_date or "N/A",
                row.flow_date or "N/A",
                f"{row.latest_price:.2f}" if row.latest_price is not None else "N/A",
                _fmt_pct(row.price_1d_pct),
                _fmt_pct(row.price_5d_pct),
                _fmt_pct(row.price_20d_pct),
                _fmt_num(row.fund_flow_latest),
                f"{row.flow_zscore:+.2f}" if row.flow_zscore is not None else "N/A",
                f"{row.nav:.2f}" if row.nav is not None else "N/A",
                _fmt_pct(row.nav_gap_pct, 3),
                "; ".join(row.warnings) if row.warnings else "",
            )
        )
    lines.append("")

    if warnings:
        lines.append("## 데이터 주의")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Barchart 옵션·선물 참고")
    lines.append("")
    lines.append(f"- Status: **{options_ref.get('status', 'not_collected')}**")
    lines.append(f"- Role: `{options_ref.get('role', 'reference_only_not_scoring')}`")
    lines.append(f"- Evidence: `{options_ref.get('evidence_path') or 'N/A'}`")
    lines.append(
        f"- Confirmed pages: **{options_ref.get('confirmed_pages', 0)}/"
        f"{options_ref.get('total_pages', 0)}** "
        f"(QQQ options {options_ref.get('qqq_options_pages_confirmed', 0)}/"
        f"{options_ref.get('qqq_options_pages_total', 0)}, futures "
        f"{options_ref.get('futures_pages_confirmed', 0)})"
    )
    lines.append(f"- 해석: {options_ref.get('interpretation')}")
    if options_ref.get("ai_explanation"):
        option_ai = options_ref.get("ai_explanation") or {}
        lines.append("")
        lines.append("### 옵션 AI 해석")
        lines.append("")
        for key in ("volatility_read", "put_call_read", "gamma_read", "action_translation"):
            if option_ai.get(key):
                lines.append(f"- {option_ai.get(key)}")
    if options_ddm_explain.get("option_read") or options_ddm_explain.get("risk_read"):
        lines.append("")
        lines.append("### 옵션+DDM 결합 해석")
        lines.append("")
        if options_ddm_explain.get("option_read"):
            lines.append(f"- {options_ddm_explain.get('option_read')}")
        if options_ddm_explain.get("risk_read"):
            lines.append(f"- {options_ddm_explain.get('risk_read')}")
    metrics = options_ref.get("metrics") or {}
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for label, key in (
        ("Latest QQQ Price", "latest_underlying_price"),
        ("IV Rank", "iv_rank_pct"),
        ("IV Percentile", "iv_percentile_pct"),
        ("Implied Volatility", "implied_volatility_pct"),
        ("ATM IV", "atm_implied_volatility_pct"),
        ("Historical Volatility", "historical_volatility_pct"),
        ("Put/Call Volume Ratio", "put_call_volume_ratio"),
        ("Put/Call OI Ratio", "put_call_oi_ratio"),
        ("Put OI Total", "put_open_interest_total"),
        ("Call OI Total", "call_open_interest_total"),
        ("Gamma Flip", "gamma_flip_point"),
        ("Gamma Flip Distance", "gamma_flip_distance_pct"),
        ("Put Wall", "put_wall"),
        ("Put Wall Distance", "put_wall_distance_pct"),
        ("Call Wall", "call_wall"),
        ("Call Wall Distance", "call_wall_distance_pct"),
    ):
        value = metrics.get(key)
        if key.endswith("_distance_pct") and value is not None:
            value = f"{_to_float(value):+.2f}%" if _to_float(value) is not None else value
        lines.append(f"| {label} | `{value or 'N/A'}` |")
    if options_ref.get("limitations"):
        lines.append("")
        lines.append("### Barchart 제한")
        lines.append("")
        for item in options_ref.get("limitations") or []:
            lines.append(f"- {item}")
    lines.append("")

    lines.append("## 판독 기준")
    lines.append("")
    lines.append("- `GO`: 신규 진입 가능")
    lines.append("- `GO_SMALL`: 강한 쪽만 선별 소액 진입")
    lines.append("- `WAIT`: 방향성은 보되 신규 추격 금지")
    lines.append("- `STOP`: 신규 진입 중단")
    lines.append("- `healthy_risk_on`: 가격 상승과 자금 유입이 동행")
    lines.append("- `greed_overheat`: 가격 상승, 강한 유입, NAV 프리미엄이 동시 발생")
    lines.append("- `fragile_rally`: 가격은 오르지만 flow/shares가 약함")
    lines.append("- `dip_buying`: 가격 약세에도 유입 발생")
    lines.append("- `stress_outflow`: 가격 하락, 유출, NAV 디스카운트 동시 발생")
    lines.append("- `quiet_accumulation/redemption`: 가격보다 flow가 먼저 움직이는 구간")
    lines.append(f"- Flow/옵션·선물 참고: {freshness.get('futures_options_reference_line')}")
    return "\n".join(lines) + "\n"


def generate_etf_radar_report(
    *,
    symbols: Optional[Sequence[str]] = None,
    lookback_days: int = 90,
    asof_date: Optional[str] = None,
    client: Optional[MarketDataClient] = None,
    recompute_history_days: int = 0,
    recompute_horizon_bars: int = 5,
    nowcast: bool = True,
    auto_tune: bool = True,
    tune_history_days: int = 180,
    tune_horizon_bars: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Fetch data and build the ETF Radar report payload."""

    universe = [symbol.upper() for symbol in (symbols or DEFAULT_UNIVERSE)]
    if not universe:
        raise EtfRadarError("ETF universe is empty")
    end_date = _parse_date(asof_date) or dt.date.today()
    tune_horizons = list(tune_horizon_bars or DEFAULT_TUNE_HORIZONS)
    max_horizon = max([int(recompute_horizon_bars)] + [int(value) for value in tune_horizons])
    analysis_lookback_days = max(
        30,
        int(lookback_days),
        int(recompute_history_days) + max_horizon + 45 if recompute_history_days > 0 else 0,
        int(tune_history_days) + max_horizon + 45 if auto_tune else 0,
    )
    data_client = client or MarketDataClient()

    def fetch_inputs(target_end: dt.date) -> Tuple[str, str, Dict[str, List[FlowPoint]], Dict[str, List[PricePoint]]]:
        target_start = target_end - dt.timedelta(days=analysis_lookback_days)
        start_text = target_start.isoformat()
        end_text = target_end.isoformat()
        flow_points = data_client.massive_fund_flows(
            universe,
            processed_date_gte=start_text,
            limit=min(max(len(universe) * (analysis_lookback_days + 20), 1000), 5000),
        )
        fetched_flow_map: Dict[str, List[FlowPoint]] = {symbol: [] for symbol in universe}
        for point in flow_points:
            if point.composite_ticker in fetched_flow_map:
                fetched_flow_map[point.composite_ticker].append(point)
        for points in fetched_flow_map.values():
            points.sort(key=lambda item: (item.effective_date, item.processed_date))

        fetched_price_map: Dict[str, List[PricePoint]] = {}
        for symbol in universe:
            try:
                fetched_price_map[symbol] = data_client.fmp_historical_prices(
                    symbol, from_date=start_text, to_date=end_text
                )
            except Exception:
                fetched_price_map[symbol] = []
        return start_text, end_text, fetched_flow_map, fetched_price_map

    def fetch_flow_persistence_inputs(target_end: dt.date) -> Dict[str, List[FlowPoint]]:
        target_start = target_end - dt.timedelta(days=FLOW_PERSISTENCE_LOOKBACK_DAYS)
        start_text = target_start.isoformat()
        audit_map: Dict[str, List[FlowPoint]] = {symbol: [] for symbol in universe}
        # This audit is explanatory only. Fetch per symbol so long histories are
        # not truncated by the ETF Global batch limit.
        for symbol in universe:
            try:
                points = data_client.massive_fund_flows(
                    [symbol],
                    processed_date_gte=start_text,
                    limit=5000,
                )
            except Exception:
                points = []
            audit_map[symbol] = sorted(
                [point for point in points if point.composite_ticker == symbol],
                key=lambda item: (item.effective_date, item.processed_date),
            )
        return audit_map

    start, end, flow_map, price_map = fetch_inputs(end_date)

    rows = build_radar_rows(
        symbols=universe,
        price_map=price_map,
        flow_map=flow_map,
        asof_date=end,
    )
    actual_price_date = _latest_price_date(rows)
    if actual_price_date and actual_price_date != end:
        actual_end_date = _parse_date(actual_price_date)
        if actual_end_date:
            end_date = actual_end_date
            start, end, flow_map, price_map = fetch_inputs(end_date)
            rows = build_radar_rows(
                symbols=universe,
                price_map=price_map,
                flow_map=flow_map,
                asof_date=end,
            )
    market_state = classify_market_state(rows)
    warnings: List[str] = []
    missing_flow = [row.symbol for row in rows if not flow_map.get(row.symbol)]
    missing_price = [row.symbol for row in rows if not price_map.get(row.symbol)]
    if missing_flow:
        warnings.append("Fund flow 미수신: " + ", ".join(missing_flow))
    if missing_price:
        warnings.append("가격 미수신: " + ", ".join(missing_price))
    stale = [row.symbol for row in rows if row.nav_stale_days is not None and row.nav_stale_days > 3]
    if stale:
        warnings.append("NAV/flow 기준일 지연: " + ", ".join(stale))
    gostop = classify_gostop_decision(rows, market_state, warnings=warnings)
    quote_map: Dict[str, Dict[str, Any]] = {}
    if nowcast:
        try:
            quote_map = data_client.fmp_quote(universe)
        except Exception:
            quote_map = {}

    now_utc, now_kst = _now_utc_and_kst()
    payload: Dict[str, Any] = {
        "generated_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_at_kst": now_kst.isoformat(),
        "publish_date_kst": now_kst.date().isoformat(),
        "data_window": {"from": start, "to": end, "lookback_days": lookback_days},
        "universe": universe,
        "market_state": market_state,
        "gostop": asdict(gostop),
        "rows": [asdict(row) for row in rows],
        "warnings": warnings,
        "markdown": "",
    }
    if nowcast:
        payload["nowcast"] = build_nowcast_overlay(
            symbols=universe,
            price_map=price_map,
            quote_map=quote_map,
            gostop=payload["gostop"],
        )
        payload["tactical_overlay"] = build_tactical_overlay(
            gostop=payload["gostop"],
            nowcast=payload["nowcast"],
        )
    payload["ddm_signal"] = build_ddm_signal(
        symbols=universe,
        price_map=price_map,
        rows=rows,
        nowcast=payload.get("nowcast") or {},
        asof_date=end,
    )
    payload["data_freshness"] = build_data_freshness_audit(rows)
    payload["ddm_input_audit"] = build_ddm_input_audit(
        payload["ddm_signal"],
        payload["data_freshness"],
    )
    flow_persistence_map = fetch_flow_persistence_inputs(end_date)
    payload["flow_persistence_audit"] = build_flow_persistence_audit(
        rows=rows,
        flow_map=flow_persistence_map,
        freshness=payload["data_freshness"],
        market_state=market_state,
    )
    if recompute_history_days > 0:
        payload["history_backtest"] = recompute_gostop_backtest(
            symbols=universe,
            price_map=price_map,
            flow_map=flow_map,
            end_date=end,
            history_days=int(recompute_history_days),
            horizon_bars=int(recompute_horizon_bars),
        )
    if auto_tune:
        payload["auto_tune"] = auto_tune_gostop(
            symbols=universe,
            price_map=price_map,
            flow_map=flow_map,
            rows=rows,
            current_decision=payload["gostop"],
            end_date=end,
            current_context=payload,
            history_days=int(tune_history_days),
            horizons=tune_horizons,
        )
    payload["qqq_decision"] = build_qqq_decision(
        gostop=payload["gostop"],
        tactical=payload.get("tactical_overlay") or {},
        nowcast=payload.get("nowcast") or {},
        auto_tune=payload.get("auto_tune") or {},
        ddm_signal=payload.get("ddm_signal") or {},
    )
    payload["tqqq_boost"] = (
        ((payload.get("auto_tune") or {}).get("selected") or {}).get("tqqq_boost_decision")
        or (payload.get("auto_tune") or {}).get("tqqq_boost_decision")
        or {"enabled": False, "source": "no_auto_tune", "tqqq_boost_pct": 0}
    )
    payload["markdown"] = render_markdown(payload)
    return payload


def _html_escape(value: Any) -> str:
    return html_lib.escape("" if value is None else str(value), quote=True)


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def _metric_tone(value: Optional[float], *, positive_good: bool = True) -> str:
    number = _to_float(value)
    if number is None:
        return "muted"
    if number == 0:
        return "muted"
    good = number > 0 if positive_good else number < 0
    return "good" if good else "bad"


def _gostop_position_snapshot(report: Mapping[str, Any]) -> Dict[str, Any]:
    decision = report.get("qqq_decision") or {}
    boost = report.get("tqqq_boost") or {}
    qqq_exposure = _to_float(decision.get("recommended_exposure_pct")) or 0.0
    boost_pct = _to_float(boost.get("tqqq_boost_pct")) or 0.0
    qqq_alloc = _to_float(boost.get("qqq_alloc_pct"))
    if qqq_alloc is None:
        qqq_alloc = max(0.0, qqq_exposure - boost_pct)
    effective_beta = _to_float(boost.get("effective_beta"))
    if effective_beta is None:
        effective_beta = (qqq_alloc + boost_pct * 3.0) / 100.0

    max_boost = float(max(TQQQ_BOOST_CANDIDATES) or 33)
    thermometer = _clamp(qqq_exposure * 0.60 + boost_pct * (40.0 / max_boost), 0.0, 100.0)
    if boost_pct > 0:
        label = "TQQQ BOOST"
        read = "QQQ buy-and-hold 일부를 TQQQ boost로 치환하는 공격 구간입니다."
    elif thermometer >= 82:
        label = "MAX RISK"
        read = "QQQ 기본 비중이 매우 높은 공격 구간입니다."
    elif thermometer >= 58:
        label = "QQQ BUY&HOLD"
        read = "기본 QQQ buy-and-hold를 유지하고 boost/감축은 제한합니다."
    elif thermometer >= 30:
        label = "REDUCED RISK"
        read = "QQQ 기본 비중을 줄인 방어 구간입니다."
    else:
        label = "DEFENSIVE CUT"
        read = "현금 비중이 큰 강한 회피 구간입니다."
    return {
        "qqq_exposure_pct": qqq_exposure,
        "tqqq_boost_pct": boost_pct,
        "qqq_alloc_pct": qqq_alloc,
        "effective_beta": effective_beta,
        "thermometer_score": thermometer,
        "thermometer_label": label,
        "thermometer_read": read,
    }


def build_gostop_ledger_row(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten the daily GoStop report into a Google-Sheets-ready audit row."""

    position = _gostop_position_snapshot(report)
    decision = report.get("qqq_decision") or {}
    overlay = decision.get("ddm_risk_overlay") or {}
    ddm = report.get("ddm_signal") or {}
    state = report.get("market_state") or {}
    nowcast = report.get("nowcast") or {}
    boost = report.get("tqqq_boost") or {}
    output_paths = report.get("output_paths") or report.get("paths") or {}
    selected = (report.get("auto_tune") or {}).get("selected") or {}
    walk_forward = selected.get("walk_forward_adaptive") or {}
    boosted_vs_hold = _to_float(walk_forward.get("avg_boosted_vs_buy_hold_pct"))
    boosted_return = _to_float(walk_forward.get("avg_boosted_strategy_return_pct"))
    buy_hold_return = (
        boosted_return - boosted_vs_hold
        if boosted_return is not None and boosted_vs_hold is not None
        else None
    )
    window = report.get("data_window") or {}
    rows = _rows_from_payload(report)
    asof = _latest_price_date(rows) or window.get("to") or ddm.get("asof_date")
    return {
        "run_at_utc": report.get("generated_at_utc"),
        "asof_date": asof,
        "markdown_path": output_paths.get("markdown"),
        "json_path": output_paths.get("json"),
        "html_path": output_paths.get("html"),
        "icloud_html_path": output_paths.get("icloud_html"),
        "google_sheet_url": os.getenv(
            "ETF_GOSTOP_SHEET_URL",
            "https://docs.google.com/spreadsheets/d/16yR0Q1ctCsFAa5FDkTfQTiQSefX_JFn62QjybW_anHk/edit",
        ),
        "action": decision.get("action"),
        "policy_key": decision.get("policy_key"),
        "headline": decision.get("headline"),
        "thermometer_score": round(position["thermometer_score"], 2),
        "thermometer_label": position["thermometer_label"],
        "qqq_exposure_pct": round(position["qqq_exposure_pct"], 2),
        "qqq_alloc_pct": round(position["qqq_alloc_pct"], 2),
        "tqqq_boost_pct": round(position["tqqq_boost_pct"], 2),
        "effective_beta": round(position["effective_beta"], 4),
        "buy_hold_return_pct": buy_hold_return,
        "qqq_strategy_return_pct": walk_forward.get("avg_strategy_return_pct"),
        "boosted_strategy_return_pct": boosted_return,
        "boosted_vs_buy_hold_pct": boosted_vs_hold,
        "boost_signals": walk_forward.get("boost_signals"),
        "boost_hit_rate_pct": walk_forward.get("boost_hit_rate_pct"),
        "active_boost_excess_pct": walk_forward.get("avg_active_boost_excess_pct"),
        "risk_reduction_signals": walk_forward.get("risk_reduction_signals"),
        "risk_reduction_hit_rate_pct": walk_forward.get("risk_reduction_hit_rate_pct"),
        "risk_reduction_excess_pct": walk_forward.get("avg_risk_reduction_excess_pct"),
        "evaluated_signals": walk_forward.get("evaluated_signals"),
        "learned_signals": walk_forward.get("learned_signals"),
        "similarity_signals": walk_forward.get("similarity_signals"),
        "ddm_status": ddm.get("status"),
        "ddm_corr_lookback_bars": ddm.get("corr_lookback_bars"),
        "ddm_min_abs_corr": ddm.get("min_abs_corr"),
        "ddm_correlated_count": ddm.get("correlated_count"),
        "ddm_high_corr_count": ddm.get("high_corr_count"),
        "ddm_overlay_severity": overlay.get("severity"),
        "ddm_confidence_pct": ddm.get("confidence_pct"),
        "ddm_drift": ddm.get("drift"),
        "ddm_diffusion": ddm.get("diffusion"),
        "ddm_evidence": ddm.get("evidence"),
        "ddm_agreement_pct": ddm.get("agreement_pct"),
        "ddm_support_pressure": ddm.get("support_pressure"),
        "ddm_resistance_pressure": ddm.get("resistance_pressure"),
        "ddm_boost_cap_multiplier": ddm.get("boost_cap_multiplier"),
        "ddm_support_symbols": ",".join(
            str(item.get("symbol") or "")
            for item in (ddm.get("support") or [])
            if isinstance(item, Mapping)
        ),
        "ddm_resistance_symbols": ",".join(
            str(item.get("symbol") or "")
            for item in (ddm.get("resistance") or [])
            if isinstance(item, Mapping)
        ),
        "market_state": state.get("label"),
        "greed_score": state.get("greed_score"),
        "nowcast_score": nowcast.get("score"),
        "nowcast_risk_day_pct": nowcast.get("risk_day_avg_pct"),
        "qqq_decision_json": _compact_json(decision),
        "tqqq_boost_json": _compact_json(boost),
        "ddm_support_json": _compact_json(ddm.get("support") or []),
        "ddm_resistance_json": _compact_json(ddm.get("resistance") or []),
        "ddm_components_json": _compact_json(ddm.get("components") or []),
        "risk_reduction_breakdown_json": _compact_json(
            walk_forward.get("risk_reduction_breakdown") or []
        ),
        "walk_forward_recent_json": _compact_json(walk_forward.get("recent") or []),
        "market_state_json": _compact_json(state),
        "nowcast_leaders_json": _compact_json(nowcast.get("leaders") or []),
        "etf_rows_json": _compact_json(report.get("rows") or []),
    }


def append_gostop_ledger(report: Mapping[str, Any], ledger_path: str) -> str:
    """Append a report row to a local CSV that can be imported by Google Sheets."""

    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = build_gostop_ledger_row(report)
    headers = list(row.keys())
    existing_rows: List[Dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_headers = list(reader.fieldnames or [])
            existing_rows = [dict(item) for item in reader]
        headers = existing_headers + [key for key in row if key not in existing_headers]
        if headers != existing_headers:
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
                writer.writeheader()
                for item in existing_rows:
                    writer.writerow(item)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return str(path)


def _gostop_history_db_paths() -> List[Path]:
    local_raw = os.getenv("ETF_GOSTOP_HISTORY_DB_PATH", str(DEFAULT_GOSTOP_HISTORY_DB_PATH))
    local = Path(os.path.expandvars(local_raw))
    cloud_default = str(DEFAULT_GOSTOP_ICLOUD_HISTORY_DB_PATH)
    cloud_raw = os.getenv("ETF_GOSTOP_ICLOUD_HISTORY_DB_PATH", cloud_default)
    paths = [local]
    if cloud_raw:
        paths.append(Path(os.path.expandvars(cloud_raw)))
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            unique.append(path.expanduser())
    return unique


def _init_gostop_history_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS etf_gostop_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at_utc TEXT,
            asof_date TEXT NOT NULL,
            universe_key TEXT NOT NULL DEFAULT 'ETF_GOSTOP',
            action TEXT,
            policy_key TEXT,
            headline TEXT,
            thermometer_score REAL,
            thermometer_label TEXT,
            qqq_exposure_pct REAL,
            qqq_alloc_pct REAL,
            tqqq_boost_pct REAL,
            effective_beta REAL,
            buy_hold_return_pct REAL,
            qqq_strategy_return_pct REAL,
            boosted_strategy_return_pct REAL,
            boosted_vs_buy_hold_pct REAL,
            boost_hit_rate_pct REAL,
            risk_reduction_hit_rate_pct REAL,
            ddm_status TEXT,
            ddm_confidence_pct REAL,
            ddm_drift REAL,
            ddm_diffusion REAL,
            market_state TEXT,
            greed_score REAL,
            nowcast_score REAL,
            markdown_path TEXT,
            json_path TEXT,
            html_path TEXT,
            icloud_html_path TEXT,
            row_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(asof_date, universe_key)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_gostop_obs_month "
        "ON etf_gostop_observations(universe_key, asof_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_gostop_obs_run "
        "ON etf_gostop_observations(run_at_utc)"
    )
    conn.commit()


def _gostop_history_values(row: Mapping[str, Any], universe_key: str = "ETF_GOSTOP") -> Dict[str, Any]:
    return {
        "run_at_utc": row.get("run_at_utc"),
        "asof_date": row.get("asof_date"),
        "universe_key": universe_key,
        "action": row.get("action"),
        "policy_key": row.get("policy_key"),
        "headline": row.get("headline"),
        "thermometer_score": _to_float(row.get("thermometer_score")),
        "thermometer_label": row.get("thermometer_label"),
        "qqq_exposure_pct": _to_float(row.get("qqq_exposure_pct")),
        "qqq_alloc_pct": _to_float(row.get("qqq_alloc_pct")),
        "tqqq_boost_pct": _to_float(row.get("tqqq_boost_pct")),
        "effective_beta": _to_float(row.get("effective_beta")),
        "buy_hold_return_pct": _to_float(row.get("buy_hold_return_pct")),
        "qqq_strategy_return_pct": _to_float(row.get("qqq_strategy_return_pct")),
        "boosted_strategy_return_pct": _to_float(row.get("boosted_strategy_return_pct")),
        "boosted_vs_buy_hold_pct": _to_float(row.get("boosted_vs_buy_hold_pct")),
        "boost_hit_rate_pct": _to_float(row.get("boost_hit_rate_pct")),
        "risk_reduction_hit_rate_pct": _to_float(row.get("risk_reduction_hit_rate_pct")),
        "ddm_status": row.get("ddm_status"),
        "ddm_confidence_pct": _to_float(row.get("ddm_confidence_pct")),
        "ddm_drift": _to_float(row.get("ddm_drift")),
        "ddm_diffusion": _to_float(row.get("ddm_diffusion")),
        "market_state": row.get("market_state"),
        "greed_score": _to_float(row.get("greed_score")),
        "nowcast_score": _to_float(row.get("nowcast_score")),
        "markdown_path": row.get("markdown_path"),
        "json_path": row.get("json_path"),
        "html_path": row.get("html_path"),
        "icloud_html_path": row.get("icloud_html_path"),
        "row_json": json.dumps(dict(row), ensure_ascii=False, default=str),
    }


def upsert_gostop_history_row(
    row: Mapping[str, Any],
    *,
    db_paths: Optional[Sequence[Path]] = None,
    universe_key: str = "ETF_GOSTOP",
) -> List[str]:
    """Upsert one GoStop ledger row into SQLite history databases."""

    values = _gostop_history_values(row, universe_key=universe_key)
    if not values.get("asof_date"):
        raise ValueError("GoStop history row requires asof_date")
    written: List[str] = []
    for db_path in db_paths or _gostop_history_db_paths():
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            _init_gostop_history_db(conn)
            conn.execute(
                """
                INSERT INTO etf_gostop_observations (
                    run_at_utc, asof_date, universe_key, action, policy_key, headline,
                    thermometer_score, thermometer_label, qqq_exposure_pct,
                    qqq_alloc_pct, tqqq_boost_pct, effective_beta,
                    buy_hold_return_pct, qqq_strategy_return_pct,
                    boosted_strategy_return_pct, boosted_vs_buy_hold_pct,
                    boost_hit_rate_pct, risk_reduction_hit_rate_pct,
                    ddm_status, ddm_confidence_pct, ddm_drift, ddm_diffusion,
                    market_state, greed_score, nowcast_score,
                    markdown_path, json_path, html_path, icloud_html_path, row_json
                ) VALUES (
                    :run_at_utc, :asof_date, :universe_key, :action, :policy_key, :headline,
                    :thermometer_score, :thermometer_label, :qqq_exposure_pct,
                    :qqq_alloc_pct, :tqqq_boost_pct, :effective_beta,
                    :buy_hold_return_pct, :qqq_strategy_return_pct,
                    :boosted_strategy_return_pct, :boosted_vs_buy_hold_pct,
                    :boost_hit_rate_pct, :risk_reduction_hit_rate_pct,
                    :ddm_status, :ddm_confidence_pct, :ddm_drift, :ddm_diffusion,
                    :market_state, :greed_score, :nowcast_score,
                    :markdown_path, :json_path, :html_path, :icloud_html_path, :row_json
                )
                ON CONFLICT(asof_date, universe_key) DO UPDATE SET
                    run_at_utc=excluded.run_at_utc,
                    action=excluded.action,
                    policy_key=excluded.policy_key,
                    headline=excluded.headline,
                    thermometer_score=excluded.thermometer_score,
                    thermometer_label=excluded.thermometer_label,
                    qqq_exposure_pct=excluded.qqq_exposure_pct,
                    qqq_alloc_pct=excluded.qqq_alloc_pct,
                    tqqq_boost_pct=excluded.tqqq_boost_pct,
                    effective_beta=excluded.effective_beta,
                    buy_hold_return_pct=excluded.buy_hold_return_pct,
                    qqq_strategy_return_pct=excluded.qqq_strategy_return_pct,
                    boosted_strategy_return_pct=excluded.boosted_strategy_return_pct,
                    boosted_vs_buy_hold_pct=excluded.boosted_vs_buy_hold_pct,
                    boost_hit_rate_pct=excluded.boost_hit_rate_pct,
                    risk_reduction_hit_rate_pct=excluded.risk_reduction_hit_rate_pct,
                    ddm_status=excluded.ddm_status,
                    ddm_confidence_pct=excluded.ddm_confidence_pct,
                    ddm_drift=excluded.ddm_drift,
                    ddm_diffusion=excluded.ddm_diffusion,
                    market_state=excluded.market_state,
                    greed_score=excluded.greed_score,
                    nowcast_score=excluded.nowcast_score,
                    markdown_path=excluded.markdown_path,
                    json_path=excluded.json_path,
                    html_path=excluded.html_path,
                    icloud_html_path=excluded.icloud_html_path,
                    row_json=excluded.row_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                values,
            )
            conn.commit()
        written.append(str(path))
    return written


def append_gostop_history_db(report: Mapping[str, Any]) -> List[str]:
    """Write the current GoStop report to local and iCloud SQLite history DBs."""

    return upsert_gostop_history_row(build_gostop_ledger_row(report))


def _extract_google_sheet_id(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    marker = "/spreadsheets/d/"
    if marker in raw:
        return raw.split(marker, 1)[1].split("/", 1)[0]
    return raw


def _sheet_range(sheet_name: Optional[str], a1_range: str) -> str:
    name = str(sheet_name or "").strip()
    if not name:
        return a1_range
    escaped = name.replace("'", "''")
    return f"'{escaped}'!{a1_range}"


def append_gostop_google_sheet(
    report: Mapping[str, Any],
    *,
    credentials_path: Optional[str] = None,
    spreadsheet_id: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> str:
    """Append a GoStop audit row to Google Sheets.

    Service-account credentials remain supported. On DGX, a stale Mac service
    account path is ignored and the existing ``~/.clasprc.json`` OAuth token is
    used as the fallback Sheets API credential.
    """

    credentials_file = existing_nonlegacy_file(
        str(
            credentials_path
            or os.getenv("ETF_GOSTOP_GOOGLE_CREDENTIALS")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            or ""
        ).strip()
    )

    sheet_id = _extract_google_sheet_id(
        spreadsheet_id
        or os.getenv("ETF_GOSTOP_SHEET_ID")
        or os.getenv("ETF_GOSTOP_SHEET_URL")
        or "https://docs.google.com/spreadsheets/d/16yR0Q1ctCsFAa5FDkTfQTiQSefX_JFn62QjybW_anHk/edit"
    )
    if not sheet_id:
        raise RuntimeError("Google Sheets spreadsheet id is not configured")

    try:
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2.credentials import Credentials
        from google.oauth2 import service_account
    except Exception as exc:
        raise RuntimeError(f"google-auth is not available: {exc}") from exc

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    auth_mode = "service_account"
    if credentials_file:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=scopes,
        )
    else:
        clasprc = Path(os.getenv("CLASPRC_PATH", str(Path.home() / ".clasprc.json"))).expanduser()
        if not clasprc.exists():
            raise RuntimeError(
                "Google Sheets credentials are not configured and ~/.clasprc.json is missing"
            )
        data = json.loads(clasprc.read_text(encoding="utf-8"))
        token_data = data.get("tokens", {}).get("default") if isinstance(data.get("tokens"), dict) else None
        if not isinstance(token_data, dict):
            token_data = data.get("tokens") if isinstance(data.get("tokens"), dict) else data
        required = ("client_id", "client_secret", "refresh_token")
        if not isinstance(token_data, dict) or not all(token_data.get(key) for key in required):
            raise RuntimeError(f"Could not find clasp OAuth refresh token fields in {clasprc}")
        credentials = Credentials(
            token=None,
            refresh_token=str(token_data["refresh_token"]),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=str(token_data["client_id"]),
            client_secret=str(token_data["client_secret"]),
            scopes=scopes,
        )
        auth_mode = "clasp_oauth"
    session = AuthorizedSession(credentials)
    base_url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values"
    target_sheet = sheet_name if sheet_name is not None else os.getenv("ETF_GOSTOP_SHEET_NAME", "")
    header_range = _sheet_range(target_sheet, "A1:ZZ1")
    row = build_gostop_ledger_row(report)
    desired_headers = list(row.keys())

    header_resp = session.get(f"{base_url}/{requests.utils.quote(header_range, safe='')}")
    header_resp.raise_for_status()
    values = header_resp.json().get("values") or []
    headers = [str(item) for item in values[0]] if values else []
    if not headers:
        headers = desired_headers
    else:
        headers = headers + [key for key in desired_headers if key not in headers]

    update_resp = session.put(
        f"{base_url}/{requests.utils.quote(header_range, safe='')}",
        params={"valueInputOption": "RAW"},
        json={"values": [headers]},
    )
    update_resp.raise_for_status()

    append_range = _sheet_range(target_sheet, "A1")
    append_resp = session.post(
        f"{base_url}/{requests.utils.quote(append_range, safe='')}:append",
        params={"valueInputOption": "RAW", "insertDataOption": "INSERT_ROWS"},
        json={"values": [[row.get(header, "") for header in headers]]},
    )
    append_resp.raise_for_status()
    updated_range = (append_resp.json().get("updates") or {}).get("updatedRange") or append_range
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit#range={updated_range}&auth={auth_mode}"


def _html_metric(label: str, value: str, caption: str = "", tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return (
        '<td class="metricCell" width="25%">'
        f'<div class="metricBox"><div class="metricLabel">{_html_escape(label)}</div>'
        f'<div class="metricValue{tone_class}">{_html_escape(value)}</div>'
        f'<div class="metricCaption">{_html_escape(caption)}</div></div></td>'
    )


def _html_bar(value: Optional[float], *, color: str = "green") -> str:
    number = _clamp(float(_to_float(value) or 0.0), 0.0, 100.0)
    return (
        '<span class="barbox">'
        f'<span class="bar {color}" style="display:block;width:{number:.0f}%"></span>'
        "</span>"
        f"{number:.1f}%"
    )


def render_gostop_html(report: Mapping[str, Any]) -> str:
    """Render an email/browser friendly ETF GoStop HTML report."""

    position = _gostop_position_snapshot(report)
    decision = report.get("qqq_decision") or {}
    overlay = decision.get("ddm_risk_overlay") or {}
    boost = report.get("tqqq_boost") or {}
    ddm = report.get("ddm_signal") or {}
    state = report.get("market_state") or {}
    nowcast = report.get("nowcast") or {}
    selected = (report.get("auto_tune") or {}).get("selected") or {}
    walk_forward = selected.get("walk_forward_adaptive") or {}
    window = report.get("data_window") or {}
    score = position["thermometer_score"]
    marker = _clamp(score, 0.0, 100.0)
    boosted_vs_hold = _to_float(walk_forward.get("avg_boosted_vs_buy_hold_pct"))
    boosted_return = _to_float(walk_forward.get("avg_boosted_strategy_return_pct"))
    buy_hold_return = (
        boosted_return - boosted_vs_hold
        if boosted_return is not None and boosted_vs_hold is not None
        else None
    )

    def component_rows(items: Sequence[Mapping[str, Any]]) -> str:
        rows_html: List[str] = []
        for item in items[:8]:
            rows_html.append(
                "<tr>"
                f"<td><strong>{_html_escape(item.get('symbol'))}</strong></td>"
                f"<td>{_fmt_num(_to_float(item.get('corr')), 2)}</td>"
                f"<td>{_fmt_num(_to_float(item.get('signed_pressure')), 2)}</td>"
                f"<td>{_fmt_pct(_to_float(item.get('price_5d_pct')))}</td>"
                f"<td>{_fmt_pct(_to_float(item.get('flow_aum_5d_pct')))}</td>"
                f"<td>{_fmt_pct(_to_float(item.get('day_change_pct')))}</td>"
                "</tr>"
            )
        return "".join(rows_html) or (
            '<tr><td colspan="6" class="muted">No correlated basket data.</td></tr>'
        )

    breakdown_rows = []
    for item in (walk_forward.get("risk_reduction_breakdown") or [])[:6]:
        breakdown_rows.append(
            "<tr>"
            f"<td style=\"text-align:left;\"><strong>{_html_escape(item.get('bucket'))}</strong></td>"
            f"<td>{_html_escape(item.get('signals'))}</td>"
            f"<td>{_html_bar(_to_float(item.get('hit_rate_pct')), color='blue')}</td>"
            f"<td class=\"{_metric_tone(_to_float(item.get('avg_excess_pct')))}\">"
            f"{_fmt_pct(_to_float(item.get('avg_excess_pct')))}</td>"
            f"<td>{_fmt_pct(_to_float(item.get('avg_drawdown_avoided_pct')))}</td>"
            "</tr>"
        )
    if not breakdown_rows:
        breakdown_rows.append('<tr><td colspan="5" class="muted">No reduction events.</td></tr>')

    metric_row = "".join(
        [
            _html_metric(
                "QQQ Buy&Hold",
                _fmt_pct(buy_hold_return),
                "워크포워드 평균 기준",
                _metric_tone(buy_hold_return),
            ),
            _html_metric(
                "QQQ Strategy",
                _fmt_pct(_to_float(walk_forward.get("avg_strategy_return_pct"))),
                "감축만 적용",
                _metric_tone(_to_float(walk_forward.get("avg_strategy_return_pct"))),
            ),
            _html_metric(
                "Boosted Strategy",
                _fmt_pct(boosted_return),
                "TQQQ boost 포함",
                _metric_tone(boosted_return),
            ),
            _html_metric(
                "vs Buy&Hold",
                _fmt_pct(boosted_vs_hold),
                "누적 감사 핵심값",
                _metric_tone(boosted_vs_hold),
            ),
        ]
    )

    css = """
	body{margin:0;padding:0;background:#f6f8fb;color:#14213d;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Segoe UI",Arial,sans-serif;letter-spacing:0;font-size:13px;line-height:19px;font-variant-numeric:tabular-nums;-webkit-text-size-adjust:100%;overflow-x:hidden;}
	body *{box-sizing:border-box}.container{width:760px;max-width:760px}.shell{background:#fff;border:1px solid #d7dee8;border-radius:8px;overflow:hidden}.inner{padding:28px 30px}
	.brandText{font-size:30px;line-height:36px;font-weight:800;color:#07143d}.headline{font-size:22px;line-height:28px;font-weight:800;margin:12px 0 4px}
	.headlineSub{font-size:13px;line-height:19px;color:#5d6878;max-width:100%;overflow-wrap:break-word}.divider{height:1px;background:#c5ceda}.metaCell{text-align:left;padding:12px 10px;border-right:1px solid #d6dde8;vertical-align:top}
	.metaLabel{font-size:11px;line-height:15px;color:#5d6878;text-transform:uppercase;font-weight:800}.metaValue{font-size:13px;line-height:18px;font-weight:750;margin-top:3px;color:#14213d;overflow-wrap:normal;word-break:normal}.section{padding:22px 0 0;margin-top:6px;border-top:1px solid #e7edf5}
	.sectionTitle{font-size:18px;line-height:24px;font-weight:800;color:#07143d;margin:0 0 11px}.sub{color:#5d6878;font-size:13px;line-height:19px}
	.metricCell{padding:9px 7px 0 0;vertical-align:top}.metricBox{border:1px solid #ccd5e2;border-radius:4px;background:#fff;padding:13px 10px;text-align:center;min-height:94px}
	.metricLabel{font-size:12px;line-height:16px;font-weight:750;color:#0b4aa2}.metricValue{font-size:24px;line-height:30px;font-weight:800;margin:8px 0 5px;white-space:nowrap}
	.metricCaption{font-size:12px;line-height:17px;color:#5d6878}.good{color:#087a35}.bad{color:#c1121f}.muted{color:#5d6878}.blueText{color:#0b4aa2}
	.tableWrap{border:1px solid #ccd5e2;border-radius:4px;overflow:hidden}table.data{border-collapse:collapse;width:100%;font-size:12px;line-height:17px}
	table.data th,table.data td{border-bottom:1px solid #dfe5ee;padding:8px 7px;text-align:center;vertical-align:middle}table.data th{font-size:12px;line-height:16px;background:#fbfcfe;font-weight:750;color:#344054}
	.barbox{height:9px;background:#e4e9f1;border-radius:999px;overflow:hidden;width:78px;display:inline-block;vertical-align:middle;margin-right:6px}.bar{height:9px;border-radius:999px}.green{background:#169b45}.blue{background:#1565d8}.orange{background:#f79009}
	.noteBox{background:#fbfcfe;border:1px solid #e3e9f2;border-radius:4px;padding:11px 12px;margin-top:9px}.noteLine{font-size:13px;line-height:20px;color:#4f5b6d;margin:0 0 5px;overflow-wrap:break-word}.noteLine:last-child{margin-bottom:0}
	.miniTitle{font-size:18px;line-height:24px;font-weight:800;margin:20px 0 9px;color:#07143d}.thermoCard{border:1px solid #ccd5e2;border-radius:4px;background:#fff;padding:15px 14px;margin-top:8px}
	.thermoHead{font-size:12px;line-height:17px;color:#5d6878;font-weight:750;text-transform:uppercase}.thermoHead strong{display:block;font-size:22px;line-height:28px;color:#07143d;text-transform:none;margin-top:3px}
	.thermoTrack{height:17px;border-radius:999px;margin:14px 0 5px;position:relative;box-shadow:inset 0 0 0 1px rgba(11,22,68,.14)}.thermoMarker{position:absolute;top:-5px;width:4px;height:27px;background:#07143d;border-radius:3px;box-shadow:0 0 0 2px #fff}
	.thermoScale td{font-size:12px;line-height:16px;color:#5d6878;font-weight:750}.thermoRead{font-size:13px;line-height:20px;color:#14213d;font-weight:750;margin-top:12px}
	.thermoDrivers{margin-top:12px;border-collapse:separate;border-spacing:6px}.thermoDriver{border:1px solid #e3e9f2;border-radius:4px;padding:10px 8px;background:#fbfcfe;vertical-align:top}
	.thermoDriverLabel{font-size:11px;line-height:15px;color:#5d6878;font-weight:800;text-transform:uppercase}.thermoDriverValue{font-size:20px;line-height:26px;color:#07143d;font-weight:800;margin-top:3px}.thermoDriverRead{font-size:12px;line-height:17px;color:#5d6878;margin-top:2px}
	.footer{border-top:1px solid #dfe5ee;margin-top:18px;padding-top:14px;text-align:center;color:#5d6878;font-size:12px;line-height:18px}
	@media only screen and (max-width:620px){.pagePad{padding:12px 0!important}.container{width:100%!important;max-width:100%!important}.inner{padding:18px 14px!important}.metricCell,.metaCell{display:block!important;width:100%!important;border-right:0!important}.scroll{overflow-x:auto!important}}
	"""
    generated = report.get("generated_at_utc") or ""
    generated_kst = report.get("generated_at_kst") or ""
    generated_display = str(generated_kst or generated)
    if "T" in generated_display:
        date_part, time_part = generated_display.split("T", 1)
        generated_display = f"{date_part} {time_part[:5]}"
        if "+09:00" in str(generated_kst):
            generated_display += " KST"
        elif str(generated).endswith("Z"):
            generated_display += " UTC"
    decision_display = str(decision.get("action") or "N/A").replace("_", " ")
    publish_date_kst = report.get("publish_date_kst") or "N/A"
    rows = _rows_from_payload(report)
    freshness = report.get("data_freshness") or build_data_freshness_audit(rows)
    ddm_audit = report.get("ddm_input_audit") or build_ddm_input_audit(ddm, freshness)
    flow_audit = report.get("flow_persistence_audit") or build_flow_persistence_audit(
        rows=rows,
        freshness=freshness,
        market_state=state,
    )
    options_ref = report.get("options_futures_reference") or build_options_futures_reference(None)
    options_ddm_explain = report.get("options_ddm_explanation") or build_options_ddm_explanation(
        options_ref=options_ref,
        ddm_signal=ddm,
        freshness=freshness,
        qqq_decision=decision,
        tqqq_boost=boost,
    )
    options_metrics = options_ref.get("metrics") or {}
    asof = _latest_price_date(rows) or window.get("to") or ddm.get("asof_date") or "N/A"
    qqq_flow = flow_audit.get("qqq") or {}
    qqq_rank = flow_audit.get("qqq_pre90_percentile_rank") or {}
    flow_contributor_rows = "".join(
        "<tr>"
        f"<td style=\"text-align:left;\"><strong>{_html_escape(item.get('symbol'))}</strong></td>"
        f"<td>{_fmt_pct(_to_float(item.get('flow_5d_aum_pct')))}</td>"
        f"<td>{_fmt_pct(_to_float(item.get('latest_flow_aum_pct')))}</td>"
        f"<td>{_fmt_pct(_to_float(item.get('price_5d_pct')))}</td>"
        "</tr>"
        for item in (flow_audit.get("top_risk_contributors") or [])[:5]
    ) or '<tr><td colspan="4" class="muted">No flow contributor data.</td></tr>'
    options_rows = "".join(
        "<tr>"
        f"<td style=\"text-align:left;\">{_html_escape(label)}</td>"
        f"<td><strong>{_html_escape((f'{_to_float(options_metrics.get(key)):+.2f}%' if key.endswith('_distance_pct') and _to_float(options_metrics.get(key)) is not None else options_metrics.get(key)) or 'N/A')}</strong></td>"
        "</tr>"
        for label, key in (
            ("Latest QQQ Price", "latest_underlying_price"),
            ("IV Rank", "iv_rank_pct"),
            ("IV Percentile", "iv_percentile_pct"),
            ("Implied Volatility", "implied_volatility_pct"),
            ("ATM IV", "atm_implied_volatility_pct"),
            ("Historical Volatility", "historical_volatility_pct"),
            ("Put/Call Volume", "put_call_volume_ratio"),
            ("Put/Call OI", "put_call_oi_ratio"),
            ("Gamma Flip", "gamma_flip_point"),
            ("Gamma Flip Distance", "gamma_flip_distance_pct"),
            ("Put Wall", "put_wall"),
            ("Put Wall Distance", "put_wall_distance_pct"),
            ("Call Wall", "call_wall"),
            ("Call Wall Distance", "call_wall_distance_pct"),
        )
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{css}</style></head>
	<body><table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f4f7fb;"><tr><td align="center" class="pagePad" style="padding:24px 10px;">
<table role="presentation" class="container" cellspacing="0" cellpadding="0"><tr><td class="shell"><div class="inner">
<div class="brandText">ETF GoStop</div><div class="headline">QQQ Timing & Boost Audit</div>
<div class="headlineSub">QQQ buy-and-hold base · TQQQ boost/reduction thermometer · daily learning ledger</div>
<div class="divider"></div>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr>
<td class="metaCell" width="20%"><div class="metaLabel">Publish Date KST</div><div class="metaValue">{_html_escape(publish_date_kst)}</div></td>
<td class="metaCell" width="20%"><div class="metaLabel">Price As-of US</div><div class="metaValue">{_html_escape(freshness.get('price_asof_date') or asof)}</div></td>
<td class="metaCell" width="20%"><div class="metaLabel">Flow/NAV As-of</div><div class="metaValue">{_html_escape(freshness.get('flow_nav_asof_date') or 'N/A')}</div></td>
	<td class="metaCell" width="20%"><div class="metaLabel">Generated KST</div><div class="metaValue">{_html_escape(generated_display)}</div></td>
	<td class="metaCell" width="20%" style="border-right:0;"><div class="metaLabel">Decision</div><div class="metaValue">{_html_escape(decision_display)} · {_html_escape(position['qqq_exposure_pct'])}%</div></td>
</tr></table><div class="divider"></div>
<div class="noteBox"><div class="noteLine"><strong>Data freshness:</strong> {_html_escape(freshness.get('status'))} · {_html_escape(freshness.get('interpretation'))}</div></div>
<div class="section"><div class="sectionTitle">Long-Term Flow Persistence Audit</div><div class="noteBox"><div class="noteLine"><strong>Status:</strong> {_html_escape(flow_audit.get('status'))} · {_html_escape(flow_audit.get('interpretation'))}</div><div class="noteLine"><strong>QQQ:</strong> latest {_fmt_num(_to_float(qqq_flow.get('latest_flow')), 2)} ({_fmt_pct(_to_float(qqq_flow.get('latest_flow_aum_pct')))} AUM), 5D {_fmt_pct(_to_float(qqq_flow.get('flow_5d_aum_pct')))}, 20D {_fmt_pct(_to_float(qqq_flow.get('flow_20d_aum_pct')))}, 60D {_fmt_pct(_to_float(qqq_flow.get('flow_60d_aum_pct')))} · pre-90 ranks 5D {_fmt_pct(_to_float(qqq_rank.get('flow_5d_aum_pct')), 1)}, 20D {_fmt_pct(_to_float(qqq_rank.get('flow_20d_aum_pct')), 1)}, 60D {_fmt_pct(_to_float(qqq_rank.get('flow_60d_aum_pct')), 1)}</div></div><div class="scroll"><div class="tableWrap"><table class="data"><thead><tr><th>Risk contributor</th><th>5D Flow/AUM</th><th>Latest Flow/AUM</th><th>5D Price</th></tr></thead><tbody>{flow_contributor_rows}</tbody></table></div></div></div>
<div class="miniTitle">QQQ Position Thermometer</div>
<div class="thermoCard"><div class="thermoHead"><span>0 Cash · 60 QQQ Buy&Hold · 100 Max Boost</span><strong>{score:.0f}/100 · {_html_escape(position['thermometer_label'])}</strong></div>
<div class="thermoTrack" style="background:linear-gradient(90deg,#c1121f 0%,#f79009 34%,#f1c95b 58%,#169b45 100%);"><div class="thermoMarker" style="left:{marker:.1f}%;">&nbsp;</div></div>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" class="thermoScale"><tr><td>Cut</td><td style="text-align:center;">QQQ Base</td><td style="text-align:right;">TQQQ Boost</td></tr></table>
<div class="thermoRead">{_html_escape(position['thermometer_read'])} {_html_escape(decision.get('headline') or '')}</div>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" class="thermoDrivers"><tr>
<td class="thermoDriver" width="33%"><div class="thermoDriverLabel">QQQ Exposure</div><div class="thermoDriverValue">{position['qqq_exposure_pct']:.0f}%</div><div class="thermoDriverRead">base sleeve</div></td>
<td class="thermoDriver" width="33%"><div class="thermoDriverLabel">TQQQ Boost</div><div class="thermoDriverValue">{position['tqqq_boost_pct']:.0f}%</div><div class="thermoDriverRead">{_html_escape(boost.get('source'))}</div></td>
<td class="thermoDriver" width="33%"><div class="thermoDriverLabel">Effective Beta</div><div class="thermoDriverValue">{position['effective_beta']:.2f}x</div><div class="thermoDriverRead">QQQ equivalent</div></td>
</tr><tr>
<td class="thermoDriver" width="33%"><div class="thermoDriverLabel">DDM Confidence</div><div class="thermoDriverValue">{_fmt_pct(_to_float(ddm.get('confidence_pct')))}</div><div class="thermoDriverRead">{_html_escape(ddm.get('status'))} / {_html_escape(overlay.get('severity'))}</div></td>
<td class="thermoDriver" width="33%"><div class="thermoDriverLabel">Boost Hit</div><div class="thermoDriverValue">{_fmt_pct(_to_float(walk_forward.get('boost_hit_rate_pct')))}</div><div class="thermoDriverRead">{_html_escape(walk_forward.get('boost_signals'))} signals</div></td>
<td class="thermoDriver" width="33%"><div class="thermoDriverLabel">Reduction Hit</div><div class="thermoDriverValue">{_fmt_pct(_to_float(walk_forward.get('risk_reduction_hit_rate_pct')))}</div><div class="thermoDriverRead">{_html_escape(walk_forward.get('risk_reduction_signals'))} signals</div></td>
</tr></table></div>
<div class="noteBox"><div class="noteLine">Thermometer score is position sizing, not a market-greed score. QQQ 100% sits near 60; TQQQ boost pushes right; risk cuts push left.</div></div>
<div class="section"><div class="sectionTitle">Buy&Hold Audit</div><table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr>{metric_row}</tr></table></div>
<div class="section"><div class="sectionTitle">Learning Status</div><div class="scroll"><div class="tableWrap"><table class="data"><thead><tr><th>Evaluated</th><th>Learned</th><th>Similarity</th><th>Boost Signals</th><th>Reduction Signals</th><th>Boost Excess</th><th>Reduction Excess</th></tr></thead><tbody><tr>
<td>{_html_escape(walk_forward.get('evaluated_signals'))}</td><td>{_html_escape(walk_forward.get('learned_signals'))}</td><td>{_html_escape(walk_forward.get('similarity_signals'))}</td>
<td>{_html_escape(walk_forward.get('boost_signals'))}</td><td>{_html_escape(walk_forward.get('risk_reduction_signals'))}</td>
<td class="{_metric_tone(_to_float(walk_forward.get('avg_active_boost_excess_pct')))}">{_fmt_pct(_to_float(walk_forward.get('avg_active_boost_excess_pct')))}</td>
<td class="{_metric_tone(_to_float(walk_forward.get('avg_risk_reduction_excess_pct')))}">{_fmt_pct(_to_float(walk_forward.get('avg_risk_reduction_excess_pct')))}</td>
</tr></tbody></table></div></div></div>
<div class="section"><div class="sectionTitle">Risk Reduction Precision</div><div class="scroll"><div class="tableWrap"><table class="data"><thead><tr><th>Bucket</th><th>Signals</th><th>Hit</th><th>Excess</th><th>Drawdown Avoided</th></tr></thead><tbody>{''.join(breakdown_rows)}</tbody></table></div></div></div>
<div class="section"><div class="sectionTitle">Drift-Diffusion Read</div><div class="scroll"><div class="tableWrap"><table class="data"><thead><tr><th>Status</th><th>Drift</th><th>Diffusion</th><th>Evidence</th><th>Agreement</th><th>Support</th><th>Resistance</th></tr></thead><tbody><tr>
<td><strong>{_html_escape(ddm.get('status'))}</strong></td><td>{_fmt_num(_to_float(ddm.get('drift')), 3)}</td><td>{_fmt_num(_to_float(ddm.get('diffusion')), 3)}</td><td>{_fmt_num(_to_float(ddm.get('evidence')), 3)}</td>
<td>{_fmt_pct(_to_float(ddm.get('agreement_pct')))}</td><td>{_fmt_num(_to_float(ddm.get('support_pressure')), 2)}</td><td>{_fmt_num(_to_float(ddm.get('resistance_pressure')), 2)}</td>
</tr></tbody></table></div></div><div class="noteBox"><div class="noteLine">{_html_escape(ddm.get('note') or '')}</div></div></div>
<div class="noteBox"><div class="noteLine"><strong>DDM input audit:</strong> {_html_escape(ddm_audit.get('quality'))} · price {_html_escape(ddm_audit.get('price_asof_date'))} / flow {_html_escape(ddm_audit.get('flow_nav_asof_date'))} · components {_html_escape(ddm_audit.get('stored_component_count'))}</div></div>
<div class="noteBox"><div class="noteLine"><strong>DDM explanation:</strong> {_html_escape(options_ddm_explain.get('ddm_read') or '')}</div><div class="noteLine">{_html_escape(options_ddm_explain.get('fusion_read') or '')}</div></div>
<div class="section"><div class="sectionTitle">Correlated Basket Support</div><div class="scroll"><div class="tableWrap"><table class="data"><thead><tr><th>ETF</th><th>Corr</th><th>Pressure</th><th>5D Price</th><th>5D Flow/AUM</th><th>1D</th></tr></thead><tbody>{component_rows(ddm.get('support') or [])}</tbody></table></div></div></div>
<div class="section"><div class="sectionTitle">Correlated Basket Resistance</div><div class="scroll"><div class="tableWrap"><table class="data"><thead><tr><th>ETF</th><th>Corr</th><th>Pressure</th><th>5D Price</th><th>5D Flow/AUM</th><th>1D</th></tr></thead><tbody>{component_rows(ddm.get('resistance') or [])}</tbody></table></div></div></div>
<div class="section"><div class="sectionTitle">Market Context</div><div class="noteBox"><div class="noteLine">State: {_html_escape(state.get('label'))} · Greed { _html_escape(state.get('greed_score'))}/100 · NowCast { _html_escape(nowcast.get('score'))}/100 · Risk day {_fmt_pct(_to_float(nowcast.get('risk_day_avg_pct')))}</div><div class="noteLine">{_html_escape(state.get('summary') or '')}</div></div></div>
	<div class="section"><div class="sectionTitle">Barchart Options/Futures Reference</div><div class="noteBox"><div class="noteLine"><strong>Status:</strong> {_html_escape(options_ref.get('status'))} · QQQ options {_html_escape(options_ref.get('qqq_options_pages_confirmed') or 0)}/{_html_escape(options_ref.get('qqq_options_pages_total') or 0)} · futures {_html_escape(options_ref.get('futures_pages_confirmed') or 0)}</div><div class="noteLine">{_html_escape(options_ref.get('interpretation') or '')}</div><div class="noteLine"><strong>Options explanation:</strong> {_html_escape(options_ddm_explain.get('option_read') or '')}</div><div class="noteLine"><strong>Combined read:</strong> {_html_escape(options_ddm_explain.get('risk_read') or '')}</div><div class="noteLine">Evidence: {_html_escape(options_ref.get('evidence_path') or 'N/A')}</div></div><div class="scroll"><div class="tableWrap"><table class="data"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{options_rows}</tbody></table></div></div></div>
<div class="footer">{_html_escape(freshness.get('futures_options_reference_line'))}<br>Automatically generated by ETF GoStop. Local CSV ledger is Google-Sheets-ready. Not investment advice.</div>
</div></td></tr></table></td></tr></table></body></html>"""


def save_report(
    report: Mapping[str, Any],
    *,
    output_dir: str = "sweet_spot_reports",
    write_json: bool = True,
    write_html: bool = False,
    icloud_dir: Optional[str] = None,
    append_ledger: bool = False,
    ledger_path: Optional[str] = None,
    append_history_db: bool = True,
    append_google_sheet: bool = False,
    google_credentials_path: Optional[str] = None,
    google_sheet_id: Optional[str] = None,
    google_sheet_name: Optional[str] = None,
) -> Dict[str, str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    md_path = Path(output_dir) / f"etf_gostop_{stamp}.md"
    md_path.write_text(str(report.get("markdown") or ""), encoding="utf-8")
    paths = {"markdown": str(md_path)}
    if write_json:
        json_path = Path(output_dir) / f"etf_gostop_{stamp}.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json"] = str(json_path)
    if write_html:
        html_doc = str(report.get("html") or render_gostop_html(report))
        html_path = Path(output_dir) / f"etf_gostop_{stamp}.html"
        html_path.write_text(html_doc, encoding="utf-8")
        paths["html"] = str(html_path)
        if icloud_dir:
            try:
                cloud_dir = Path(icloud_dir)
                cloud_dir.mkdir(parents=True, exist_ok=True)
                cloud_path = cloud_dir / html_path.name
                cloud_path.write_text(html_doc, encoding="utf-8")
                paths["icloud_html"] = str(cloud_path)
            except Exception as exc:
                paths["icloud_error"] = str(exc)
    if append_ledger:
        ledger = ledger_path or str(Path(output_dir) / "etf_gostop_sheet_rows.csv")
        try:
            ledger_report = dict(report)
            ledger_report["output_paths"] = dict(paths)
            paths["ledger_csv"] = append_gostop_ledger(ledger_report, ledger)
        except Exception as exc:
            paths["ledger_error"] = str(exc)
    if append_history_db:
        try:
            db_report = dict(report)
            db_report["output_paths"] = dict(paths)
            db_paths = append_gostop_history_db(db_report)
            if db_paths:
                paths["history_db"] = db_paths[0]
            if len(db_paths) > 1:
                paths["icloud_history_db"] = db_paths[1]
        except Exception as exc:
            paths["history_db_error"] = str(exc)
    if append_google_sheet:
        try:
            sheet_report = dict(report)
            sheet_report["output_paths"] = dict(paths)
            paths["google_sheet"] = append_gostop_google_sheet(
                sheet_report,
                credentials_path=google_credentials_path,
                spreadsheet_id=google_sheet_id,
                sheet_name=google_sheet_name,
            )
        except Exception as exc:
            paths["google_sheet_error"] = str(exc)
    return paths


def compact_slack_text(report: Mapping[str, Any], *, max_rows: int = 8) -> str:
    state = report.get("market_state") or {}
    gostop = report.get("gostop") or {}
    nowcast = report.get("nowcast") or {}
    qqq_decision = report.get("qqq_decision") or {}
    tqqq_boost = report.get("tqqq_boost") or {}
    ddm_signal = report.get("ddm_signal") or {}
    auto_tune = report.get("auto_tune") or {}
    rows = [RadarRow(**row) if isinstance(row, dict) else row for row in report.get("rows", [])]
    interesting = sorted(rows, key=_row_sort_key, reverse=True)[:max_rows]
    lines = [
        "*ETF GoStop QQQ Decision*",
        (
            f"QQQ: *{qqq_decision.get('action', 'N/A')}* "
            f"exposure {qqq_decision.get('recommended_exposure_pct', 'N/A')}%"
        ),
        str(qqq_decision.get("headline") or ""),
        (
            f"Swing GoStop: *{gostop.get('action', 'N/A')}* "
            f"({gostop.get('score', 'N/A')}/100, {gostop.get('mode', 'unknown')})"
        ),
        f"Swing 신규 한도: *{gostop.get('max_new_risk_pct', 'N/A')}%*",
        str(gostop.get("headline") or ""),
        f"시장: *{state.get('label', 'unknown')}* / Greed {state.get('greed_score', 'N/A')}/100",
        str(state.get("summary") or ""),
        "",
        "핵심 ETF:",
    ]
    if auto_tune.get("enabled"):
        selected = auto_tune.get("selected") or {}
        lines.insert(
            4,
            (
                "튜닝: "
                f"{auto_tune.get('status', 'unknown')} / "
                f"entry {selected.get('entry_horizon_bars', 'N/A')}D, "
                f"stop {selected.get('stop_horizon_bars', 'N/A')}D, "
                f"risk {selected.get('tuned_max_new_risk_pct', 'N/A')}%"
            ),
        )
    if tqqq_boost.get("enabled") and _to_float(tqqq_boost.get("tqqq_boost_pct")):
        lines.insert(
            3,
            (
                "TQQQ boost: "
                f"{_fmt_pct(tqqq_boost.get('tqqq_boost_pct'), 0)} "
                f"(QQQ alloc {_fmt_pct(tqqq_boost.get('qqq_alloc_pct'), 0)}, "
                f"beta {(_to_float(tqqq_boost.get('effective_beta')) or 0.0):.2f})"
            ),
        )
    if ddm_signal.get("enabled"):
        lines.insert(
            3,
            (
                "DDM: "
                f"{ddm_signal.get('status', 'unknown')} "
                f"conf {_fmt_pct(ddm_signal.get('confidence_pct'))}, "
                f"drift {_fmt_num(ddm_signal.get('drift'), 2)}, "
                f"diff {_fmt_num(ddm_signal.get('diffusion'), 2)}"
            ),
        )
    if nowcast.get("enabled"):
        lines.insert(
            4,
            (
                "NowCast: "
                f"{nowcast.get('status', 'unknown')} "
                f"({nowcast.get('score', 'N/A')}/100, "
                f"risk 1D {_fmt_pct(nowcast.get('risk_day_avg_pct'))})"
            ),
        )
    for row in interesting:
        lines.append(
            "- `{}` {} | 5D {} | flow/AUM {} | NAV {} | {}".format(
                row.symbol,
                row.signal,
                _fmt_pct(row.price_5d_pct),
                _fmt_pct(row.flow_aum_5d_pct, 3),
                _fmt_pct(row.nav_gap_pct, 3),
                row.nuance,
            )
        )
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("주의: " + " / ".join(str(item) for item in warnings[:3]))
    return "\n".join(lines)


generate_etf_gostop_report = generate_etf_radar_report


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="ETF GoStop close report")
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--output-dir", default="sweet_spot_reports")
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--print", action="store_true", dest="print_report")
    args = parser.parse_args()

    report_payload = generate_etf_radar_report(
        symbols=[item.strip() for item in args.symbols.split(",") if item.strip()],
        lookback_days=args.lookback_days,
        asof_date=args.asof_date,
    )
    saved = save_report(
        report_payload,
        output_dir=args.output_dir,
        write_json=not args.no_json,
    )
    if args.print_report:
        print(report_payload["markdown"])
    else:
        print(json.dumps(saved, ensure_ascii=False, indent=2))
