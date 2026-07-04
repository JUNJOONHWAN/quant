"""Train Gaussian NB + Platt model for market probability reports."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from market_analysis.market_report import (
    FEATURES,
    fit_gaussian_nb,
    platt_scale,
    predict_proba_nb,
    save_model,
)
from market_analysis.market_prob_backtest import (
    DEFAULT_WINDOW,
    HISTORY_FILE,
    build_labeled_rows,
)


def _build_dataset(
    start: str | None,
    end: str | None,
    horizon: int,
    base_symbol: str,
    history_file: str,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, _ = build_labeled_rows(
        start_date=start,
        end_date=end,
        horizon_days=horizon,
        base_symbol=base_symbol,
        history_file=history_file,
        window=window,
    )
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for row in rows:
        vec = np.array(row["features_vector"], dtype=float)
        if not np.isfinite(vec).any():
            continue
        X_list.append(vec)
        y_list.append(int(row["actual_up"]))
    if not X_list:
        raise RuntimeError("유효한 피처가 있는 표본이 없습니다. 히스토리를 더 수집하세요.")
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y


def train_model(
    *,
    start: str | None,
    end: str | None,
    horizon: int,
    base_symbol: str,
    history_file: str,
    window: int,
    min_samples: int,
) -> dict:
    X, y = _build_dataset(start, end, horizon, base_symbol, history_file, window)
    if len(y) < min_samples:
        raise RuntimeError(f"표본이 {len(y)}개뿐입니다. --min-samples 값을 낮추거나 더 많은 히스토리를 확보하세요.")
    params = fit_gaussian_nb(X, y)
    raw_probs: List[float] = []
    for vec in X:
        p, _ = predict_proba_nb(params, np.nan_to_num(vec, nan=0.0))
        raw_probs.append(p)
    A, B = platt_scale(np.array(raw_probs), y)
    save_model(params, (A, B))
    return {
        "path": os.path.join("ml_cache", "market_prob_nb.json"),
        "samples": len(y),
        "min_samples": min_samples,
        "start": start,
        "end": end,
        "horizon": horizon,
        "symbol": base_symbol,
    }


def notify_slack(stats: dict) -> None:
    webhook = os.getenv("SCREENING_SLACK_HOOK", "").strip()
    if not webhook:
        return
    text = (
        "*Market Prob 모델 재학습 완료*\n"
        f"- 기간: {stats.get('start') or '전체'} → {stats.get('end') or '현재'}\n"
        f"- 표본 수: {stats.get('samples')} (요구 {stats.get('min_samples')})\n"
        f"- 심볼/지평: {stats.get('symbol')} / H={stats.get('horizon')}일\n"
        f"- 모델 파일: `{stats.get('path')}`"
    )
    try:
        requests.post(webhook, json={"text": text}, timeout=5)
    except Exception as exc:
        print(f"⚠️ Slack 통보 실패: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train market probability model from history")
    parser.add_argument("--start", type=str, default=None, help="학습 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="학습 종료일 (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=5, help="H 지평(거래일)")
    parser.add_argument("--symbol", type=str, default="QQQ", help="기준 심볼")
    parser.add_argument("--history", type=str, default=HISTORY_FILE, help="히스토리 JSONL 경로")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="가격 시계열 윈도우")
    parser.add_argument("--min-samples", type=int, default=200, help="최소 학습 표본 수")
    args = parser.parse_args()
    stats = train_model(
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        base_symbol=args.symbol,
        history_file=args.history,
        window=args.window,
        min_samples=args.min_samples,
    )
    print(f"✅ 모델 저장: {stats['path']}")
    print(f"표본 수: {stats['samples']} (요구 {stats['min_samples']}) · H={stats['horizon']}일 · 심볼={stats['symbol']}")
    notify_slack(stats)


if __name__ == "__main__":
    main()
