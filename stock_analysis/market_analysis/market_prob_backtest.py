"""
Market probability backtester.

Reads the JSONL history recorded by market_report.generate_market_report and
evaluates how well the predicted P(Up|x) matched subsequent QQQ (or bench) moves.

Design highlights:
- No lookahead: comparisons use trading-day offsets (horizon) into bench closes.
- History entries are deduplicated by (date, symbol, horizon) keeping the newest.
- Outputs summary stats + per-trade rows for webapp consumption.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from regime_service import at2_get_ticker_series
from market_analysis.market_report import FEATURES

HISTORY_FILE = os.path.join("ml_cache", "market_prob_history.jsonl")
DEFAULT_WINDOW = 2000


def _load_history(path: str = HISTORY_FILE) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    entries.append(data)
            except Exception:
                continue
    return entries


def _dedup(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for entry in entries:
        date = str(entry.get("asof_date") or "")
        if not date:
            continue
        horizon = int(entry.get("horizon_days", 0))
        sym = str(entry.get("base_symbol") or "QQQ").upper()
        key = (date, horizon, sym)
        prev = latest.get(key)
        ts = str(entry.get("ts_utc") or "")
        if prev is None or ts > str(prev.get("ts_utc") or ""):
            latest[key] = entry
    out = list(latest.values())
    out.sort(key=lambda e: (e.get("asof_date") or "", e.get("base_symbol") or ""))
    return out


def _parse_date(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s))
    except Exception:
        return None


def _load_price_series(base_symbol: str, window: int = DEFAULT_WINDOW) -> Tuple[List[str], List[float]]:
    payload = at2_get_ticker_series(window=window, preset=None, use_realtime=False)
    dates = payload.get("dates") or []
    if not isinstance(dates, list) or not dates:
        raise RuntimeError("레짐 페이로드에서 날짜 배열을 찾을 수 없습니다.")
    series_map = payload.get("series") or {}
    prices = None
    if isinstance(series_map, dict):
        prices = (
            series_map.get(base_symbol)
            or series_map.get(base_symbol.upper())
            or series_map.get("QQQ")
            or series_map.get("SPY")
        )
    if not isinstance(prices, list) or not prices:
        raise RuntimeError(f"{base_symbol} 가격 시계열을 찾을 수 없습니다.")
    n = min(len(dates), len(prices))
    if n == 0:
        raise RuntimeError("가격 시계열 길이가 0입니다.")
    return [str(dates[i]) for i in range(-n, 0)], [float(prices[i]) for i in range(-n, 0)]


@dataclass
class BacktestResult:
    stats: Dict[str, Any]
    rows: List[Dict[str, Any]]
    markdown: str
    json_path: str


def build_labeled_rows(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    horizon_days: int = 5,
    base_symbol: str = "QQQ",
    history_file: str = HISTORY_FILE,
    window: int = DEFAULT_WINDOW,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    entries = _dedup(_load_history(history_file))
    if not entries:
        raise RuntimeError("확률 리포트 히스토리가 없습니다. 먼저 리포트를 실행해 기록을 남겨주세요.")
    base_symbol = base_symbol.upper()
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    filtered: List[Dict[str, Any]] = []
    for entry in entries:
        if int(entry.get("horizon_days", 0)) != int(horizon_days):
            continue
        sym = str(entry.get("base_symbol") or "QQQ").upper()
        if sym != base_symbol:
            continue
        d = _parse_date(entry.get("asof_date"))
        if d is None:
            continue
        if start_dt and d < start_dt:
            continue
        if end_dt and d > end_dt:
            continue
        filtered.append({**entry, "_asof_dt": d})
    if not filtered:
        raise RuntimeError("조건에 맞는 기록이 없습니다. 기간/지평/심볼을 확인하세요.")

    dates, prices = _load_price_series(base_symbol, window=window)
    index_map = {str(date): idx for idx, date in enumerate(dates)}

    rows: List[Dict[str, Any]] = []
    for entry in filtered:
        asof = str(entry.get("asof_date"))
        idx = index_map.get(asof)
        prob = entry.get("prob")
        if idx is None or prob is None:
            continue
        future_idx = idx + int(horizon_days)
        if future_idx >= len(prices):
            continue
        price_now = prices[idx]
        price_future = prices[future_idx]
        if price_now <= 0 or price_future <= 0:
            continue
        realized = price_future / price_now - 1.0
        actual_up = 1 if realized > 0 else 0
        feat_map = entry.get("features") or {}
        vec: List[float] = []
        for name in FEATURES:
            val = feat_map.get(name)
            try:
                num = float(val)
                if math.isnan(num):
                    vec.append(float("nan"))
                else:
                    vec.append(num)
            except (TypeError, ValueError):
                vec.append(float("nan"))
        rows.append(
            {
                "asof_date": asof,
                "future_date": dates[future_idx],
                "prob": float(prob),
                "prob_raw": entry.get("prob_raw"),
                "realized_return": realized,
                "actual_up": actual_up,
                "predicted_up": None,
                "asof_price": price_now,
                "future_price": price_future,
                "horizon_days": horizon_days,
                "features_vector": vec,
                "features_map": feat_map,
            }
        )
    if not rows:
        raise RuntimeError("가격 데이터와 매칭되는 기록이 없습니다. window 값을 늘리거나 기간을 조정하세요.")

    rows.sort(key=lambda r: r["asof_date"])
    meta = {
        "samples": len(rows),
        "start_date": rows[0]["asof_date"],
        "end_date": rows[-1]["asof_date"],
        "base_symbol": base_symbol,
        "horizon_days": horizon_days,
    }
    return rows, meta


def run_backtest(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    horizon_days: int = 5,
    prob_threshold: float = 0.5,
    base_symbol: str = "QQQ",
    history_file: str = HISTORY_FILE,
    window: int = DEFAULT_WINDOW,
) -> BacktestResult:
    rows, meta = build_labeled_rows(
        start_date=start_date,
        end_date=end_date,
        horizon_days=horizon_days,
        base_symbol=base_symbol,
        history_file=history_file,
        window=window,
    )

    for row in rows:
        prob = row.get("prob")
        row["predicted_up"] = 1 if prob is not None and float(prob) >= prob_threshold else 0

    probs = [r["prob"] for r in rows]
    outcomes = [r["actual_up"] for r in rows]
    rets = [r["realized_return"] for r in rows]
    accuracy = float(np.mean([int(r["actual_up"] == r["predicted_up"]) for r in rows]))
    hit_rate = float(np.mean(outcomes))
    brier = float(np.mean([(p - y) ** 2 for p, y in zip(probs, outcomes)]))
    avg_prob = float(np.mean(probs))
    avg_ret = float(np.mean(rets))
    avg_ret_up = float(np.mean([r for r in rets if r > 0])) if any(r > 0 for r in rets) else 0.0
    avg_ret_dn = float(np.mean([r for r in rets if r <= 0])) if any(r <= 0 for r in rets) else 0.0
    avg_prob_up = float(np.mean([p for p, y in zip(probs, outcomes) if y == 1])) if any(outcomes) else None
    avg_prob_dn = (
        float(np.mean([p for p, y in zip(probs, outcomes) if y == 0])) if any(y == 0 for y in outcomes) else None
    )

    stats = {
        "samples": meta["samples"],
        "start_date": meta["start_date"],
        "end_date": meta["end_date"],
        "horizon_days": horizon_days,
        "base_symbol": meta["base_symbol"],
        "threshold": prob_threshold,
        "accuracy": accuracy,
        "hit_rate": hit_rate,
        "brier": brier,
        "avg_prob": avg_prob,
        "avg_return": avg_ret,
        "avg_return_up": avg_ret_up,
        "avg_return_down": avg_ret_dn,
        "avg_prob_actual_up": avg_prob_up,
        "avg_prob_actual_down": avg_prob_dn,
    }

    lines = [
        "### 📈 확률 백테스트 결과",
        f"- 표본 수: {stats['samples']} · 기간: {stats['start_date']} → {stats['end_date']}",
        f"- 기준 심볼: {meta['base_symbol']} · 지평 H={horizon_days} 거래일",
        f"- 정확도(Threshold {prob_threshold:.2f}): {accuracy*100:.1f}%",
        f"- Hit-rate(실제 상승 비중): {hit_rate*100:.1f}%",
        f"- Brier score: {brier:.4f}",
        f"- 평균 확률 P(Up): {avg_prob*100:.1f}% | 평균 수익률: {avg_ret*100:.2f}%",
    ]
    if avg_prob_up is not None and avg_prob_dn is not None:
        lines.append(
            f"- 실제 상승일 확률 평균: {avg_prob_up*100:.1f}% · 하락일 확률 평균: {avg_prob_dn*100:.1f}%"
        )
    markdown = "\n".join(lines)

    fd, path = tempfile.mkstemp(prefix="market_prob_backtest_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        json.dump({"stats": stats, "rows": rows}, fp, ensure_ascii=False, indent=2)

    return BacktestResult(stats=stats, rows=rows, markdown=markdown, json_path=path)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Market probability backtester")
    parser.add_argument("--start", type=str, default=None, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=5, help="지평 (거래일)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Up 판정 확률 임계값")
    parser.add_argument("--symbol", type=str, default="QQQ", help="기준 심볼 (기본 QQQ)")
    parser.add_argument("--history", type=str, default=HISTORY_FILE, help="히스토리 파일 경로")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="가격 시계열 윈도우(거래일)")
    args = parser.parse_args()
    result = run_backtest(
        start_date=args.start,
        end_date=args.end,
        horizon_days=args.horizon,
        prob_threshold=args.threshold,
        base_symbol=args.symbol,
        history_file=args.history,
        window=args.window,
    )
    print(result.markdown)
    print(f"\nJSON 저장: {result.json_path}")


if __name__ == "__main__":
    _main()
#  truncated due to message size
