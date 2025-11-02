import os
import math
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import requests

# --- Minimal, compatibility-first realtime regime engine (FMP-backed) ---

# Symbols and clusters (match front-end JS defaults)
BASE_SYMBOL = "QQQ"
_strategy_sym_env = os.getenv("REGIME_STRATEGY_SYMBOL", "TQQQ")
STRATEGY_SYMBOL = (_strategy_sym_env or "TQQQ").upper()
_bench_sym_env = os.getenv("REGIME_BENCH_SYMBOL", BASE_SYMBOL)
BENCH_SYMBOL = (_bench_sym_env or BASE_SYMBOL).upper()
try:
    NEUTRAL_BENCH_WEIGHT = float(os.getenv("REGIME_NEUTRAL_BENCH_WEIGHT", "0.33"))
except ValueError:
    NEUTRAL_BENCH_WEIGHT = 0.33
try:
    RISK_ON_BENCH_WEIGHT = float(os.getenv("REGIME_RISK_ON_BENCH_WEIGHT", "1.0"))
except ValueError:
    RISK_ON_BENCH_WEIGHT = 1.0
RISK_SYMBOLS = ["IWM", "SPY", "BTC-USD"]
SAFE_SYMBOLS = ["TLT", "GLD"]
_all_symbols_seed: List[str] = [BASE_SYMBOL]
for sym in [STRATEGY_SYMBOL, BENCH_SYMBOL]:
    if sym and sym not in _all_symbols_seed:
        _all_symbols_seed.append(sym)
for sym in ["IWM", "SPY", "TLT", "GLD", "BTC-USD"]:
    if sym not in _all_symbols_seed:
        _all_symbols_seed.append(sym)
ALL_SYMBOLS = _all_symbols_seed
SIGNAL_SYMBOLS = ["IWM", "SPY", "TLT", "GLD", "BTC-USD"]
CLUSTERS = {
    "risk": ["IWM", "SPY", "BTC-USD"],
    "safe": ["TLT", "GLD"],
}
SIGNAL_PAIR_KEY = "IWM|BTC-USD"

# Window options
WINDOWS = [20, 30, 60]

# Historical coverage settings (ensure enough lead-in for rolling windows and parity with JS)
MIN_HISTORY_YEARS = 6  # At least ~2019 baseline
MIN_HISTORY_START = pd.Timestamp("2019-01-01")
MIN_HISTORY_LEAD_DAYS = 365  # extra buffer so indicators have lookback runway

# FFL baseline knobs (subset, aligned with JS defaults where practical)
RISK_CFG_FFL = {
    "lookbacks": {"momentum": 10, "vol": 20, "breadth": 5},
    "p": 1.5,
    "zSat": 2.0,
    "lambda": 0.25,
    "thresholds": {
        "jOn": +0.10,
        "jOff": -0.08,
        "scoreOn": 0.60,
        "scoreOff": 0.40,
        "breadthOn": 0.50,
        "mmFragile": 0.88,
        "mmOff": 0.96,
        "pconOn": 0.55,
        "pconOff": 0.40,
        "mmHi": 0.90,
        "downAll": 0.60,
        "corrConeDays": 5,
        "driftMinDays": 3,
        "driftCool": 2,
        "offMinDays": 2,
        "vOn": +0.05,
        "vOff": -0.05,
        "kOn": 0.60,
    },
    "kLambda": 1.0,
    "exp": {
        "lam": 0.75,
        "rOn": 0.76,
        "rOff": -0.05,
        "breadth1dMin": 0.45,
        "d0AnyPos": True,
        "d0BothPosHiCorr": False,
        "ti": {"win": 52, "onK": 0.81, "offK": 0.37, "hiCorrScale": 1.25, "strongK": 1.5},
        "lev": {"r0": 0.30, "r1": 1.05, "min": 1.0, "max": 3.0, "damp": 3.0},
    },
    "expTune": {
        "macroOn": 0.514,
        "macroOff": 0.371,
        "qOn": 0.715,
        "qOff": 0.244,
        "aS": 0.187,
        "aJ": 0.095,
        "bS": 0.106,
        "bJ": 0.058,
        "gSPos": 0.052,
        "gSNeg": 0.065,
        "confirmOn": 2,
        "confirmOff": 1,
        "hazardHigh": 0.513,
        "hazardDrop": 0.033,
        "hazardLookback": 5,
        "wC": 0.5,
        "wF": 0.3,
        "wS": 0.2,
    },
    "stabTune": {
        "fast": 21,
        "slow": 63,
        "zWin": 126,
        "zUp": 2.50,
        "zDown": 2.50,
        "slopeMin": 0.0200,
        "neutralLo": 0.30,
        "neutralHi": 0.40,
        "lagUp": 3,
        "lagDown": 4,
        "onFluxEase": 0.02,
        "confirmOnMin": 2,
        "leadOnWindow": 6,
        "downGrace": 6,
        "hazardWindow": 9,
        "offFluxTighten": 0.03,
        "confirmOffMin": 1,
        "onOverrideMargin": 0.01,
        "upOffHarden": 0.02,
        "upConfirmOffMin": 2,
    },
}


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def _tanh_clip(x: float, sat: float = 2.0) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(np.tanh(x / max(1e-9, sat)))


def ema(seq: List[float], n: int) -> List[float]:
    if n <= 1:
        return list(seq)
    out: List[float] = []
    alpha = 2 / (n + 1)
    prev = None
    for v in seq:
        if not np.isfinite(v):
            out.append(np.nan)
            continue
        if prev is None or not np.isfinite(prev):
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def clamp01(value: Optional[float], default: float = 0.0) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(min(1.0, max(0.0, value)))


def normalize_range(value: Optional[float], lo: float, hi: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    if hi <= lo:
        return 0.0
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def sigmoid(value: Optional[float], k: float = 0.85) -> float:
    if value is None or not np.isfinite(value):
        return 1.0
    return float(1.0 / (1.0 + math.exp(-k * value)))


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        num = float(value)
        return num if np.isfinite(num) else default
    except Exception:
        return default


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return float(arr[mid])
    return float(0.5 * (arr[mid - 1] + arr[mid]))


def rolling_sigma_mad(series: List[Optional[float]], end: int, window: int) -> Optional[float]:
    if window <= 0 or end < 0:
        return None
    xs: List[float] = []
    start = max(0, end - window + 1)
    for idx in range(start, end + 1):
        val = series[idx]
        if val is not None and np.isfinite(val):
            xs.append(float(val))
    if len(xs) < 5:
        return None
    med = _median(xs)
    if med is None:
        return None
    deviations = [abs(x - med) for x in xs]
    mad = _median(deviations)
    if mad is None or mad <= 0:
        return None
    return float(1.4826 * mad)


def to_returns(prices: List[float]) -> List[float]:
    arr: List[float] = []
    for i in range(1, len(prices)):
        a, b = prices[i - 1], prices[i]
        if np.isfinite(a) and np.isfinite(b) and a != 0:
            arr.append(math.log(b / a))
        else:
            arr.append(0.0)
    return arr


def top_eigenvalue(mat: np.ndarray) -> float:
    try:
        vals = np.linalg.eigvalsh(mat)
        if vals.size == 0:
            return 0.0
        return float(np.max(vals))
    except Exception:
        return 0.0


def rolling_corr_matrix(ret_mat: np.ndarray) -> np.ndarray:
    # ret_mat: shape (n_assets, window)
    if ret_mat.shape[1] < 2:
        return np.eye(ret_mat.shape[0])
    return np.corrcoef(ret_mat)


def top_eigenvalue_ratio(mat: np.ndarray) -> float:
    try:
        vals = np.linalg.eigvalsh(mat)
        vals = np.sort(vals)
        lam1 = float(vals[-1]) if len(vals) else 0.0
        n = mat.shape[0]
        return lam1 / max(1, n)
    except Exception:
        return np.nan


def stability_index(mat: np.ndarray) -> float:
    # Weighted |corr| average with higher weight on risk-safe pairs
    labels = SIGNAL_SYMBOLS
    idx = {s: i for i, s in enumerate(labels)}
    w_sum = 0.0
    num = 0.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = labels[i]
            b = labels[j]
            w = 2.0 if ((a in RISK_SYMBOLS and b in SAFE_SYMBOLS) or (a in SAFE_SYMBOLS and b in RISK_SYMBOLS)) else 1.0
            c = mat[i, j]
            if np.isfinite(c):
                num += w * abs(float(c))
                w_sum += w
    return float(num / w_sum) if w_sum > 0 else np.nan


def sub_indices(mat: np.ndarray) -> Dict[str, float]:
    # stockCrypto(+): |corr(IWM/SPY, BTC)| avg
    # traditional(+): |corr(IWM, SPY, TLT, GLD)| avg
    # safeNegative(-): max(0, -corr(stock, safe)) avg
    labels = SIGNAL_SYMBOLS
    idx = {s: i for i, s in enumerate(labels)}
    out = {"stockCrypto": np.nan, "traditional": np.nan, "safeNegative": np.nan}
    vals_sc = []
    for s in ["IWM", "SPY"]:
        c = mat[idx[s], idx["BTC-USD"]]
        if np.isfinite(c):
            vals_sc.append(abs(float(c)))
    out["stockCrypto"] = float(np.mean(vals_sc)) if vals_sc else np.nan
    trad_pairs = [("IWM", "SPY"), ("IWM", "TLT"), ("IWM", "GLD"), ("SPY", "TLT"), ("SPY", "GLD"), ("TLT", "GLD")]
    vals_trad = []
    for a, b in trad_pairs:
        c = mat[idx[a], idx[b]]
        if np.isfinite(c):
            vals_trad.append(abs(float(c)))
    out["traditional"] = float(np.mean(vals_trad)) if vals_trad else np.nan
    vals_sn = []
    for s in ["IWM", "SPY"]:
        for t in ["TLT", "GLD"]:
            c = mat[idx[s], idx[t]]
            if np.isfinite(c):
                vals_sn.append(max(0.0, -float(c)))
    out["safeNegative"] = float(np.mean(vals_sn)) if vals_sn else np.nan
    return out


def rolling_std_from_series(series: List[float], idx: int, lookback: int) -> Optional[float]:
    if lookback < 2 or idx < lookback:
        return None
    window = []
    for i in range(idx - lookback + 1, idx + 1):
        a = series[i - 1]
        b = series[i]
        if np.isfinite(a) and np.isfinite(b) and a != 0:
            window.append(b / a - 1.0)
    if len(window) < 2:
        return None
    arr = np.array(window, dtype=float)
    mu = float(np.mean(arr))
    var = float(np.var(arr, ddof=1))
    return math.sqrt(max(0.0, var))


def z_momentum(series: List[float], idx: int, k: int, v: int, z_sat: float) -> Optional[float]:
    if idx < k or k <= 0 or v <= 1:
        return None
    base = series[idx - k]
    latest = series[idx]
    if not (np.isfinite(base) and np.isfinite(latest)) or base == 0:
        return None
    mom = latest / base - 1.0
    std = rolling_std_from_series(series, idx, v)
    if std is None or std == 0:
        return None
    return _tanh_clip(mom / std, z_sat)


def frob_diff(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    diff = a - b
    return float(np.sqrt(np.mean(np.square(diff))))


def rolling_z(values: List[Optional[float]], i: int, win: int) -> Optional[float]:
    xs = [float(v) for v in values[max(0, i - win + 1): i + 1] if v is not None and np.isfinite(v)]
    if len(xs) < 5:
        return None
    arr = np.array(xs, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd <= 0:
        return None
    v = values[i]
    if v is None or not np.isfinite(v):
        return None
    return (float(v) - mu) / sd


# ----------------------------------------------------------------------------
# Data fetch (FMP)
# ----------------------------------------------------------------------------

def _map_symbol_for_fmp(sym: str) -> str:
    if sym == "BTC-USD":
        return "BTCUSD"
    return sym


def fetch_daily_history_fmp(symbols: List[str], from_date: str, api_key: str) -> Dict[str, pd.DataFrame]:
    base = "https://financialmodelingprep.com/api/v3/historical-price-full/"
    out: Dict[str, pd.DataFrame] = {}
    sess = requests.Session()
    for sym in symbols:
        mapped = _map_symbol_for_fmp(sym)
        def _fetch(url: str) -> List[Dict[str, Any]]:
            r = sess.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            return data.get("historical") or []
        url_primary = f"{base}{mapped}?from={from_date}&apikey={api_key}"
        hist = _fetch(url_primary)
        if not hist:
            # retry with serietype=line (ETF 등 일부 티커에서 필요)
            url_alt = f"{base}{mapped}?from={from_date}&serietype=line&apikey={api_key}"
            hist = _fetch(url_alt)
        if not hist:
            raise RuntimeError(f"FMP 응답에 {sym} 데이터가 없습니다. API 키 권한을 확인하세요.")
        rows = []
        for row in hist:
            d = row.get("date")
            # Prefer adjusted close to stay consistent with Alpha Vantage / JS pipeline.
            ac = row.get("adjClose")
            close = ac if isinstance(ac, (int, float)) else row.get("close")
            open_raw = row.get("open")
            if not d or not isinstance(close, (int, float)):
                continue
            # Build adjusted open when possible to stay aligned with adjusted close.
            if isinstance(ac, (int, float)) and isinstance(row.get("close"), (int, float)) and row.get("close"):
                factor = float(ac) / float(row.get("close"))
                adj_open = float(open_raw) * factor if isinstance(open_raw, (int, float)) else None
            else:
                adj_open = float(open_raw) if isinstance(open_raw, (int, float)) else None
            rows.append(
                {
                    "date": d,
                    "adj_close": float(close),
                    "adj_open": float(adj_open) if adj_open is not None else None,
                }
            )
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # Normalize to tz-naive UTC date to avoid tz-aware/naive comparisons downstream
        dt = pd.to_datetime(df["date"], utc=True, errors="coerce")
        dt = dt.dt.tz_convert(None).dt.normalize()
        df["date"] = dt
        df = df.set_index("date").sort_index()
        out[sym] = df
    return out


def fetch_realtime_quotes_fmp(symbols: List[str], api_key: str) -> Dict[str, Tuple[pd.Timestamp, float, Optional[float]]]:
    base = "https://financialmodelingprep.com/api/v3/quote/"
    out: Dict[str, Tuple[pd.Timestamp, float, Optional[float]]] = {}
    sess = requests.Session()
    q_syms = ",".join([_map_symbol_for_fmp(s) for s in symbols])
    url = f"{base}{q_syms}?apikey={api_key}"
    r = sess.get(url, timeout=15)
    r.raise_for_status()
    rows = r.json() if isinstance(r.json(), list) else []
    by_symbol = {row.get("symbol"): row for row in rows if isinstance(row, dict)}
    for sym in symbols:
        key = _map_symbol_for_fmp(sym)
        row = by_symbol.get(key)
        if not row:
            continue
        price_raw = row.get("price")
        prev_close_raw = row.get("previousClose")
        price: Optional[float] = None
        if isinstance(price_raw, (int, float)) and price_raw > 0:
            price = float(price_raw)
        elif isinstance(prev_close_raw, (int, float)) and prev_close_raw > 0:
            price = float(prev_close_raw)
        else:
            # Skip realtime patch when FMP returns 0.0 (or missing) to avoid zeroing out history.
            continue
        ts = row.get("timestamp") or row.get("lastUpdated")
        try:
            t = pd.to_datetime(ts, unit="s", utc=True) if isinstance(ts, (int, float)) else pd.to_datetime(ts, utc=True)
        except Exception:
            t = pd.Timestamp.utcnow()
        prev_close = float(prev_close_raw) if isinstance(prev_close_raw, (int, float)) else None
        out[sym] = (t, price, prev_close)
    return out


def patch_with_realtime_last_price(series_map: Dict[str, pd.Series], quotes: Dict[str, Tuple[pd.Timestamp, float, Optional[float]]]) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for sym, s in series_map.items():
        q = quotes.get(sym)
        if not q:
            out[sym] = s.copy()
            continue
        ts, price, prev_close = q
        # normalize to date (naive, no tz) to be consistent with historical series
        day = pd.Timestamp(year=ts.year, month=ts.month, day=ts.day, tz="UTC").tz_convert(None).normalize()
        base = s.copy()
        if day in base.index:
            base.loc[day] = float(price)
        else:
            # Append a new day using realtime price
            append = pd.Series([float(price)], index=pd.to_datetime([day]))
            base = pd.concat([base, append]).sort_index()
        out[sym] = base
    return out


def _align_by_intersection(series_map: Dict[str, pd.Series], keys: List[str]) -> Tuple[List[pd.Timestamp], Dict[str, pd.Series]]:
    """Match JS alignSeries: keep only dates present for all keys.
    Dates are normalized to naive midnight for stable set operations.
    """
    # Build intersection of date indexes across requested keys
    inter: Optional[pd.DatetimeIndex] = None
    normed: Dict[str, pd.Series] = {}
    for k in keys:
        s = series_map.get(k)
        if s is None or s.empty:
            continue
        # normalize index to naive midnight
        idx = pd.to_datetime(s.index)
        # Robust tz handling: drop tz if present, otherwise no-op
        try:
            idx = idx.tz_localize(None)
        except Exception:
            try:
                idx = idx.tz_convert(None)
            except Exception:
                pass
        idx = idx.normalize()
        ss = pd.Series(s.values, index=idx)
        normed[k] = ss.sort_index()
        inter = idx if inter is None else inter.intersection(idx)
    if inter is None:
        return [], {k: pd.Series(dtype=float) for k in keys}
    inter = inter.sort_values()
    aligned = {k: normed.get(k, pd.Series(dtype=float)).reindex(inter).dropna() for k in keys}
    # ensure all have same index after dropna; build final intersection again to be safe
    inter2 = inter
    for k in keys:
        inter2 = inter2.intersection(aligned[k].index)
    inter2 = inter2.sort_values()
    aligned = {k: aligned[k].reindex(inter2) for k in keys}
    return list(inter2), aligned


# ----------------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------------

@dataclass
class RegimeSeries:
    dates: List[str]
    score: List[Optional[float]]
    state: List[int]
    executed_state: List[int]
    mm: List[Optional[float]]
    guard: List[Optional[float]]
    jflux: List[Optional[float]]
    fint: List[Optional[float]]


def compute_window_metrics(prices: Dict[str, List[float]], dates: List[pd.Timestamp], window: int) -> Dict[str, Any]:
    """JS parity:
    - returns are computed on price transitions (length = len(dates)-1)
    - rolling window is applied over returns indexes [start..end]
    - record date = dates[end] (aligns with analysis_data/app.js and precomputed.json)
      NOTE: Previously this used dates[end+1], which shifted Python one day ahead
      versus the front-end and precomputed, causing Classic/FFL misalignment.
    """
    idx = {s: i for i, s in enumerate(SIGNAL_SYMBOLS)}
    series = {s: prices[s] for s in SIGNAL_SYMBOLS}
    ret = {s: to_returns(series[s]) for s in SIGNAL_SYMBOLS}
    ret_len = max(0, len(dates) - 1)
    records: List[Dict[str, Any]] = []
    stability_vals: List[float] = []
    prev_mat: Optional[np.ndarray] = None
    full_flux: List[Optional[float]] = []
    for end in range(window - 1, ret_len):
        start = end - window + 1
        mat = np.zeros((len(SIGNAL_SYMBOLS), len(SIGNAL_SYMBOLS)))
        for a in range(len(SIGNAL_SYMBOLS)):
            for b in range(len(SIGNAL_SYMBOLS)):
                ra = ret[SIGNAL_SYMBOLS[a]][start:end+1]
                rb = ret[SIGNAL_SYMBOLS[b]][start:end+1]
                if len(ra) != len(rb) or len(ra) < 2:
                    mat[a, b] = np.nan
                else:
                    mat[a, b] = np.corrcoef(ra, rb)[0, 1]
        stab = stability_index(mat)
        stability_vals.append(stab)
        # JS sets record.date to analysisDates[endIndex], which corresponds to prices[end+1].
        records.append({
            "date": dates[end + 1].strftime("%Y-%m-%d") if (end + 1) < len(dates) else dates[-1].strftime("%Y-%m-%d"),
            "stability": stab,
            "sub": sub_indices(mat),
            "matrix": mat,
        })
        full_flux.append(frob_diff(mat, prev_mat))
        prev_mat = mat
    # Smoothed and delta (EMA3-EMA10 on stability)
    # JS app.js uses EMA10 as smoothed, and EMA3-EMA10 delta.
    sm10 = ema(stability_vals, 10)
    sm3 = ema(stability_vals, 3)
    for i, rec in enumerate(records):
        rec["smoothed"] = sm10[i]
        rec["delta"] = (sm3[i] - sm10[i]) if np.isfinite(sm3[i]) and np.isfinite(sm10[i]) else np.nan
    avg180 = float(np.nanmean(stability_vals[-min(180, len(stability_vals)):])) if stability_vals else np.nan
    return {"records": records, "average180": avg180, "fullFlux": full_flux}


def compute_classic(metrics: Dict[str, Any]) -> Tuple[List[str], List[float], List[int], List[float], List[float]]:
    recs = metrics["records"]
    dates = [r["date"] for r in recs]
    # Pair corr: IWM|BTC-USD approximation via matrix
    # Map indices
    idx = {s: i for i, s in enumerate(SIGNAL_SYMBOLS)}
    sc_corr: List[float] = []
    safe_neg: List[float] = []
    for r in recs:
        mat = r["matrix"]
        if isinstance(mat, np.ndarray):
            sc = mat[idx["IWM"], idx["BTC-USD"]]
        else:
            sc = np.nan
        sc_corr.append(float(sc) if np.isfinite(sc) else 0.0)
        safe_neg.append(float(r["sub"].get("safeNegative", 0.0)))
    w_sc, w_safe = 0.70, 0.30
    score = [max(0.0, min(1.0, w_sc * max(0.0, sc) + w_safe * sn)) for sc, sn in zip(sc_corr, safe_neg)]
    th = {"scoreOn": 0.65, "scoreOff": 0.30, "corrOn": 0.50, "corrMinOn": 0.20, "corrOff": -0.10}
    state: List[int] = []
    for sc, s in zip(sc_corr, score):
        if sc >= th["corrOn"]:
            state.append(1)
        elif sc <= th["corrOff"] or s <= th["scoreOff"]:
            state.append(-1)
        elif s >= th["scoreOn"] and sc >= th["corrMinOn"]:
            state.append(1)
        else:
            state.append(0)
    exec_state = [0] + state[:-1] if state else []
    return dates, score, state, sc_corr, safe_neg


def quantile(xs: List[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = min(max(q, 0.0), 1.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    w = pos - lo
    return float(s[lo] * (1 - w) + s[hi] * w)


def top_eigenvector(mat: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> Optional[np.ndarray]:
    try:
        n = mat.shape[0]
        v = np.ones(n, dtype=float)
        v /= np.linalg.norm(v) if np.linalg.norm(v) else 1.0
        for _ in range(max_iter):
            w = mat @ v
            norm = np.linalg.norm(w)
            if norm == 0:
                break
            nv = w / norm
            if np.linalg.norm(nv - v) < tol:
                v = nv
                break
            v = nv
        return v
    except Exception:
        return None


def rolling_return(series: List[float], idx: int, lookback: int) -> Optional[float]:
    if lookback <= 0 or idx < lookback or idx >= len(series):
        return None
    start = series[idx - lookback]
    end = series[idx]
    if not (np.isfinite(start) and np.isfinite(end)) or start == 0:
        return None
    return end / start - 1.0


def compute_ffl(prices: Dict[str, List[float]], dates: List[pd.Timestamp], metrics: Dict[str, Any], window: int, variant: str = "stab") -> Dict[str, Any]:
    recs = metrics.get("records") or []
    length = len(recs)
    if length == 0:
        return {}

    labels = SIGNAL_SYMBOLS
    idx_map = {symbol: idx for idx, symbol in enumerate(labels)}
    window_offset = max(1, window - 1)

    lookbacks = RISK_CFG_FFL.get("lookbacks", {})
    mom_lb = int(lookbacks.get("momentum", 10))
    vol_lb = int(lookbacks.get("vol", 20))
    z_sat = float(RISK_CFG_FFL.get("zSat", 2.0))
    p_power = float(RISK_CFG_FFL.get("p", 1.5))
    lambda_flux = float(RISK_CFG_FFL.get("lambda", 0.25))
    k_lambda = float(RISK_CFG_FFL.get("kLambda", 1.0))

    price_len = len(dates)
    z_cache: Dict[str, Dict[int, Optional[float]]] = {symbol: {} for symbol in labels}

    def z_value(symbol: str, idx: int) -> Optional[float]:
        cache = z_cache[symbol]
        if idx in cache:
            return cache[idx]
        series = prices.get(symbol)
        if not isinstance(series, list) or idx < 0 or idx >= len(series):
            cache[idx] = None
            return None
        value = z_momentum(series, idx, mom_lb, vol_lb, z_sat)
        cache[idx] = value
        return value

    pair_series = (metrics.get("pairs") or {}).get(SIGNAL_PAIR_KEY)
    correlations = pair_series.get("correlation") if isinstance(pair_series, dict) else None
    sc_corr: List[float] = []
    if correlations:
        for i in range(length):
            value = correlations[i] if i < len(correlations) else None
            sc_corr.append(safe_float(value, float("nan")))
    else:
        sc_corr = [float("nan")] * length

    safe_neg_series = [safe_float(rec.get("sub", {}).get("safeNegative", 0.0), 0.0) for rec in recs]
    dates_str = [rec.get("date") for rec in recs]

    dates_idx = pd.to_datetime(pd.Index(dates))
    risk_candidates = [BASE_SYMBOL, "IWM", "BTC-USD"]
    risk_set = [sym for sym in risk_candidates if sym in prices]
    r_star_series: Optional[pd.Series] = None
    if risk_set:
        for sym in risk_set:
            px = pd.Series(prices.get(sym, []), index=dates_idx)
            log_ret = np.log(px).diff()
            pos2 = (log_ret.clip(lower=0.0) ** 2).ewm(span=60, adjust=False).mean()
            neg2 = (log_ret.clip(upper=0.0) ** 2).ewm(span=60, adjust=False).mean()
            ratio = (neg2 / (pos2 + 1e-12)).clip(0, 5)
            if r_star_series is None:
                r_star_series = ratio
            else:
                r_star_series = pd.concat([r_star_series, ratio], axis=1).max(axis=1)
    if r_star_series is None:
        r_star_series = pd.Series(1.0, index=dates_idx)
    else:
        r_star_series = r_star_series.reindex(dates_idx).fillna(1.0)
    R_star = r_star_series.values

    def bear_damp(idx: int, r_thr: float = 1.2, beta: float = 0.5, gmin: float = 0.5) -> float:
        if 0 <= idx < len(R_star):
            x_val = max(0.0, float(R_star[idx]) - r_thr)
            return max(gmin, 1.0 / (1.0 + beta * x_val))
        return 1.0

    mm: List[Optional[float]] = [None] * length
    mm_trend: List[Optional[float]] = [None] * length
    jflux: List[Optional[float]] = [None] * length
    risk_beta_flux: List[Optional[float]] = [None] * length
    flux_raw: List[Optional[float]] = [None] * length
    flux_intensity: List[Optional[float]] = [None] * length
    flux_slope: List[Optional[float]] = [None] * length
    guard: List[Optional[float]] = [None] * length
    score: List[Optional[float]] = [None] * length
    apdf: List[Optional[float]] = [None] * length
    pcon: List[Optional[float]] = [None] * length
    vpc1: List[Optional[float]] = [None] * length
    kappa: List[Optional[float]] = [None] * length
    co_down_all: List[Optional[float]] = [None] * length
    combo_mom: List[Optional[float]] = [None] * length
    breadth: List[Optional[float]] = [None] * length
    diffusion_scores: List[Optional[float]] = [None] * length
    full_flux: List[Optional[float]] = [None] * length
    full_flux_z: List[Optional[float]] = [None] * length
    far: List[Optional[float]] = [None] * length
    fragile: List[bool] = [False] * length

    full_flux_window = min(63, max(15, length // 4 if length // 4 else 15))

    prev_mat: Optional[np.ndarray] = None
    for i, rec in enumerate(recs):
        matrix_raw = rec.get("matrix")
        matrix: Optional[np.ndarray]
        if isinstance(matrix_raw, np.ndarray):
            matrix = matrix_raw.astype(float)
        elif isinstance(matrix_raw, (list, tuple)):
            try:
                matrix = np.array(matrix_raw, dtype=float)
            except Exception:
                matrix = None
        else:
            matrix = None
        if matrix is not None and matrix.ndim != 2:
            matrix = None
        if matrix is not None and matrix.shape[0] != len(labels):
            try:
                matrix = np.array(matrix[:len(labels), :len(labels)], dtype=float)
            except Exception:
                matrix = None

        mm_val = None
        if matrix is not None and matrix.size > 0:
            mm_val = top_eigenvalue(matrix) / max(1, matrix.shape[0])
        mm[i] = mm_val if mm_val is not None and np.isfinite(mm_val) else None
        if i > 0 and mm[i] is not None and mm[i - 1] is not None:
            mm_trend[i] = float(mm[i] - mm[i - 1])
        else:
            mm_trend[i] = None

        price_index = window_offset + i
        zr: Dict[str, Optional[float]] = {symbol: z_value(symbol, price_index) for symbol in labels}

        weight_sum = 0.0
        flux_sum = 0.0
        abs_sum = 0.0
        apdf_weighted = 0.0
        pcon_weighted = 0.0
        weights_all = 0.0

        if matrix is not None:
            rows, cols = matrix.shape
            for safe_sym in CLUSTERS["safe"]:
                ia = idx_map[safe_sym]
                if ia >= rows:
                    continue
                for risk_sym in CLUSTERS["risk"]:
                    ib = idx_map[risk_sym]
                    if ib >= cols:
                        continue
                    coef = matrix[ia, ib]
                    if not np.isfinite(coef):
                        continue
                    weight = abs(coef) ** p_power
                    if weight <= 0:
                        continue
                    s_val = zr.get(safe_sym)
                    r_val = zr.get(risk_sym)
                    s_norm = float(s_val) if s_val is not None and np.isfinite(s_val) else 0.0
                    r_norm = float(r_val) if r_val is not None and np.isfinite(r_val) else 0.0
                    diff = r_norm - s_norm
                    weight_sum += weight
                    flux_sum += weight * diff
                    abs_sum += weight * abs(diff)
                    apdf_weighted += weight * diff
                    pcon_weighted += weight * (1.0 if r_norm > s_norm else 0.0)
                    weights_all += weight
            for ai in range(len(CLUSTERS["risk"])):
                sym_a = CLUSTERS["risk"][ai]
                ia = idx_map[sym_a]
                if ia >= rows:
                    continue
                for bi in range(ai + 1, len(CLUSTERS["risk"])):
                    sym_b = CLUSTERS["risk"][bi]
                    ib = idx_map[sym_b]
                    if ib >= cols:
                        continue
                    coef = matrix[ia, ib]
                    if not np.isfinite(coef):
                        continue
                    weight = abs(coef) ** p_power
                    if weight <= 0:
                        continue
                    a_norm = float(zr.get(sym_a)) if zr.get(sym_a) is not None and np.isfinite(zr.get(sym_a)) else 0.0
                    b_norm = float(zr.get(sym_b)) if zr.get(sym_b) is not None and np.isfinite(zr.get(sym_b)) else 0.0
                    apdf_weighted += weight * (0.5 * (a_norm + b_norm))
                    pcon_weighted += weight * (1.0 if (a_norm > 0 and b_norm > 0) else 0.0)
                    weights_all += weight
            for ai in range(len(CLUSTERS["safe"])):
                sym_a = CLUSTERS["safe"][ai]
                ia = idx_map[sym_a]
                if ia >= rows:
                    continue
                for bi in range(ai + 1, len(CLUSTERS["safe"])):
                    sym_b = CLUSTERS["safe"][bi]
                    ib = idx_map[sym_b]
                    if ib >= cols:
                        continue
                    coef = matrix[ia, ib]
                    if not np.isfinite(coef):
                        continue
                    weight = abs(coef) ** p_power
                    if weight <= 0:
                        continue
                    a_norm = float(zr.get(sym_a)) if zr.get(sym_a) is not None and np.isfinite(zr.get(sym_a)) else 0.0
                    b_norm = float(zr.get(sym_b)) if zr.get(sym_b) is not None and np.isfinite(zr.get(sym_b)) else 0.0
                    apdf_weighted += weight * (-0.5 * (a_norm + b_norm))
                    pcon_weighted += weight * (1.0 if (a_norm <= 0 and b_norm <= 0) else 0.0)
                    weights_all += weight
        if weight_sum > 0:
            jbar = flux_sum / weight_sum
            jnorm = _tanh_clip(jbar, lambda_flux)
        else:
            jbar = 0.0
            jnorm = 0.0
        jflux_val = jnorm * bear_damp(price_index)
        jflux[i] = jflux_val
        flux_raw[i] = jbar if weight_sum > 0 else 0.0
        flux_intensity[i] = abs_sum / weight_sum if weight_sum > 0 else 0.0
        if i > 0 and jflux_val is not None and jflux[i - 1] is not None:
            flux_slope[i] = jflux_val - jflux[i - 1]
        else:
            flux_slope[i] = None

        matrix_for_diff = matrix.copy() if matrix is not None else None
        prev_matrix = prev_mat.copy() if prev_mat is not None else None
        full_flux_val = frob_diff(matrix_for_diff, prev_matrix) if matrix_for_diff is not None else None
        full_flux[i] = full_flux_val if full_flux_val is not None and np.isfinite(full_flux_val) else None
        z_full = rolling_z(full_flux, i, full_flux_window)
        full_flux_z[i] = z_full if z_full is not None and np.isfinite(z_full) else None

        if weights_all > 0:
            apdf_val = math.tanh((apdf_weighted / weights_all) / (lambda_flux or 0.25))
            apdf[i] = apdf_val
            pcon_val = pcon_weighted / weights_all
            pcon[i] = clamp01(pcon_val)
        else:
            apdf[i] = None
            pcon[i] = None

        if matrix_for_diff is not None:
            try:
                eig_vec = top_eigenvector(matrix_for_diff)
            except Exception:
                eig_vec = None
            if eig_vec is not None:
                num_proj = 0.0
                den_proj = 0.0
                for sym, weight in zip(labels, eig_vec):
                    z_val = z_value(sym, price_index)
                    z_norm = float(z_val) if z_val is not None and np.isfinite(z_val) else 0.0
                    num_proj += float(weight) * z_norm
                    den_proj += abs(float(weight))
                vproj = num_proj / den_proj if den_proj > 0 else 0.0
                vpc1[i] = math.tanh(vproj / 0.5)
            else:
                vpc1[i] = None
        else:
            vpc1[i] = None

        all_z: List[float] = []
        risk_z: List[float] = []
        for sym in labels:
            val = z_value(sym, price_index)
            if val is not None and np.isfinite(val):
                all_z.append(float(val))
        for sym in CLUSTERS["risk"]:
            val = z_value(sym, price_index)
            if val is not None and np.isfinite(val):
                risk_z.append(float(val))
        co_down_all[i] = (sum(1 for value in all_z if value < 0) / len(all_z)) if all_z else None
        combo_mom[i] = float(np.mean(risk_z)) if risk_z else None
        breadth[i] = (sum(1 for value in risk_z if value > 0) / len(risk_z)) if risk_z else None

        mm_pen = normalize_range(mm_val, 0.85, 0.97)
        safe_pen = normalize_range(safe_neg_series[i], 0.35, 0.60)
        delta_val = safe_float(rec.get("delta"), float("nan"))
        delta_pen = normalize_range(max(0.0, -delta_val) if np.isfinite(delta_val) else 0.0, 0.015, 0.05)
        flux_guard = sigmoid(full_flux_z[i], 0.85)
        guard_val = 0.4 * mm_pen + 0.2 * safe_pen + 0.2 * delta_pen + 0.2 * flux_guard
        guard[i] = guard_val

        flux_score = 0.5 * (1.0 + (jflux_val if jflux_val is not None else 0.0)) if jflux_val is not None else None
        combo_norm = clamp01(0.5 * ((combo_mom[i] if combo_mom[i] is not None else 0.0) + 1.0)) if combo_mom[i] is not None else None
        breadth_norm = clamp01(breadth[i]) if breadth[i] is not None else None
        guard_relief = clamp01(1.0 - guard_val)
        components: List[Tuple[float, float]] = []
        if flux_score is not None:
            components.append((0.5, float(flux_score)))
        if combo_norm is not None:
            components.append((0.2, float(combo_norm)))
        if breadth_norm is not None:
            components.append((0.2, float(breadth_norm)))
        if guard_relief is not None:
            components.append((0.1, float(guard_relief)))
        if components:
            total_weight = sum(weight for weight, _ in components)
            aggregated = sum(weight * value for weight, value in components) / total_weight if total_weight > 0 else (flux_score or 0.5)
        else:
            aggregated = flux_score if flux_score is not None else 0.5
        score[i] = float(aggregated)

        mt = max(0.0, mm_trend[i]) if mm_trend[i] is not None and np.isfinite(mm_trend[i]) else 0.0
        fs_neg = max(0.0, -(flux_slope[i] if flux_slope[i] is not None and np.isfinite(flux_slope[i]) else 0.0))
        diff_val = (jflux_val if jflux_val is not None else 0.0) - (0.50 * mt) - (0.15 * fs_neg)
        diffusion_scores[i] = diff_val

        if prev_mat is not None and matrix_for_diff is not None:
            rbw = 0.0
            rbsum = 0.0
            rows_prev, cols_prev = prev_mat.shape
            for ai, sym_a in enumerate(CLUSTERS["risk"]):
                ia = idx_map[sym_a]
                if ia >= matrix_for_diff.shape[0] or ia >= rows_prev:
                    continue
                for bi in range(ai + 1, len(CLUSTERS["risk"])):
                    sym_b = CLUSTERS["risk"][bi]
                    ib = idx_map[sym_b]
                    if ib >= matrix_for_diff.shape[1] or ib >= cols_prev:
                        continue
                    curr = matrix_for_diff[ia, ib]
                    prev_val = prev_mat[ia, ib]
                    if not np.isfinite(curr) or not np.isfinite(prev_val):
                        continue
                    delta = curr - prev_val
                    weight = abs(curr) ** p_power
                    if weight <= 0:
                        continue
                    dir_sum = 0.0
                    za = zr.get(sym_a)
                    zb = zr.get(sym_b)
                    if za is not None and np.isfinite(za):
                        dir_sum += float(za)
                    if zb is not None and np.isfinite(zb):
                        dir_sum += float(zb)
                    dir_sign = 1.0 if dir_sum >= 0 else -1.0
                    rbw += weight
                    rbsum += weight * delta * dir_sign
            rb_bar = (rbsum / rbw) if rbw > 0 else 0.0
            risk_beta_flux[i] = float(np.tanh(rb_bar / (lambda_flux or 0.25)))
        else:
            risk_beta_flux[i] = None

        prev_mat = matrix_for_diff.copy() if matrix_for_diff is not None else None

    thresholds = RISK_CFG_FFL.get("thresholds", {})
    valid_flux = [float(x) for x in jflux if x is not None and np.isfinite(x)]
    valid_score = [float(x) for x in score if x is not None and np.isfinite(x)]
    dyn_on_flux = float(thresholds.get("jOn", 0.10))
    dyn_off_flux = float(thresholds.get("jOff", -0.08))
    dyn_on_score = float(thresholds.get("scoreOn", 0.60))
    dyn_off_score = float(thresholds.get("scoreOff", 0.40))

    exp_tune = RISK_CFG_FFL.get("expTune", {})
    q_on = float(exp_tune.get("qOn", 0.715))
    q_off = float(exp_tune.get("qOff", 0.244))

    if variant != "classic" and len(valid_flux) >= 50:
        dyn_on_flux = max(dyn_on_flux, quantile(valid_flux, q_on))
        dyn_off_flux = min(dyn_off_flux, quantile(valid_flux, q_off))
    if variant != "classic" and len(valid_score) >= 50:
        dyn_on_score = max(dyn_on_score, quantile(valid_score, 0.75))
        dyn_off_score = min(dyn_off_score, quantile(valid_score, 0.25))

    stab_cfg = RISK_CFG_FFL.get("stabTune", {})
    s_fast_n = int(stab_cfg.get("fast", 21))
    s_slow_n = int(stab_cfg.get("slow", 63))
    s_z_win = int(stab_cfg.get("zWin", 126))
    s_neutral_lo = float(stab_cfg.get("neutralLo", 0.30))
    s_neutral_hi = float(stab_cfg.get("neutralHi", 0.40))
    s_min = float(stab_cfg.get("slopeMin", 0.02))
    s_z_up = float(stab_cfg.get("zUp", 2.50))
    s_z_down = float(stab_cfg.get("zDown", 2.50))

    Sseries = [safe_float(rec.get("stability"), 0.0) for rec in recs]
    s_ema_fast = ema(Sseries, s_fast_n)
    s_ema_slow = ema(Sseries, s_slow_n)
    s_slope: List[Optional[float]] = [None] * length
    s_sigma: List[Optional[float]] = [None] * length
    s_z_values: List[Optional[float]] = [None] * length
    s_up_seq_values: List[int] = [0] * length
    s_down_seq_values: List[int] = [0] * length
    up_seq = 0
    down_seq = 0
    for i in range(length):
        v_fast = s_ema_fast[i] if np.isfinite(s_ema_fast[i]) else None
        v_slow = s_ema_slow[i] if np.isfinite(s_ema_slow[i]) else None
        slope = (v_fast - v_slow) if (v_fast is not None and v_slow is not None) else None
        s_slope[i] = slope
        sigma = rolling_sigma_mad(s_slope, i, s_z_win)
        s_sigma[i] = sigma
        if slope is not None and sigma is not None and sigma > 0 and np.isfinite(slope):
            z_val = slope / sigma
        else:
            z_val = None
        s_z_values[i] = z_val
        up_cond = z_val is not None and np.isfinite(z_val) and slope is not None and slope >= s_min and z_val >= s_z_up
        down_cond = z_val is not None and np.isfinite(z_val) and slope is not None and slope <= -s_min and z_val <= -s_z_down
        up_seq = up_seq + 1 if up_cond else 0
        down_seq = down_seq + 1 if down_cond else 0
        s_up_seq_values[i] = up_seq
        s_down_seq_values[i] = down_seq

    state: List[int] = [0] * length
    on_cand = 0
    off_cand = 0
    prev_state = 0
    drift_seq = 0
    drift_cooldown = 0
    in_drift_epoch = False
    stab_neutral_run = 0
    s_down_run = 0

    for i in range(length):
        flux_val = jflux[i] if jflux[i] is not None else 0.0
        diff_val = diffusion_scores[i] if diffusion_scores[i] is not None else flux_val
        score_val = score[i] if score[i] is not None else 0.5
        mm_val = mm[i] if mm[i] is not None else 0.0
        combo_val = combo_mom[i]
        breadth_val = breadth[i]
        guard_val = guard[i] if guard[i] is not None else 1.0
        vpc1_val = vpc1[i] if vpc1[i] is not None else 0.0

        if flux_val is not None and np.isfinite(flux_val) and np.isfinite(diff_val) and np.isfinite(vpc1_val):
            kappa_val = abs(diff_val) / (abs(diff_val) + k_lambda * abs(vpc1_val) + 1e-6)
        else:
            kappa_val = None
        kappa[i] = kappa_val

        guard_soft = 0.95
        guard_hard = 0.98
        breadth_gate = (breadth_val if breadth_val is not None and np.isfinite(breadth_val) else 0.0) >= (thresholds.get("breadthOn", 0.5) * 0.6)

        s_val_now = s_slope[i] if s_slope[i] is not None and np.isfinite(s_slope[i]) else None
        s_z_now = s_z_values[i] if s_z_values[i] is not None and np.isfinite(s_z_values[i]) else None
        if variant == "stab":
            if s_val_now is not None and s_val_now < -max(1e-6, s_min):
                s_down_run += 1
            else:
                s_down_run = 0
        stab_up_trend = variant == "stab" and s_val_now is not None and s_val_now > 0
        stab_plunge = variant == "stab" and s_val_now is not None and s_z_now is not None and s_z_now <= -s_z_down and s_val_now < 0

        pcon_val = pcon[i]
        apdf_val = apdf[i]
        pcon_ok_base = (pcon_val is None) or (pcon_val >= thresholds.get("pconOn", 0.55))
        apdf_ok = (apdf_val is None) or (apdf_val >= -0.05)
        pcon_ok = pcon_ok_base or (diff_val >= (dyn_on_flux + 0.07))

        idx_price = window_offset + i
        bench_returns = prices.get(BASE_SYMBOL, [])
        bench10 = rolling_return(bench_returns, idx_price, 10) if isinstance(bench_returns, list) else None
        bench20 = rolling_return(bench_returns, idx_price, 20) if isinstance(bench_returns, list) else None
        hi_corr_bear = (mm_val >= 0.90) and ((bench10 if bench10 is not None else -1e-9) <= 0)
        co_down_val = co_down_all[i]
        hi_corr_drift = ((co_down_val is not None and co_down_val >= thresholds.get("downAll", 0.60) and flux_val <= 0) or
                         ((mm_val >= thresholds.get("mmHi", 0.90)) and ((bench20 if bench20 is not None else -1e-9) <= 0)))

        if hi_corr_drift:
            drift_seq += 1
            drift_cooldown = 0
        else:
            drift_seq = 0
            drift_cooldown += 1
        if drift_seq >= int(thresholds.get("driftMinDays", 3)):
            in_drift_epoch = True
        if in_drift_epoch and drift_cooldown >= int(thresholds.get("driftCool", 2)):
            in_drift_epoch = False

        dyn_on_adj = dyn_on_flux + (0.05 if mm_val >= 0.94 else (0.03 if mm_val >= 0.90 else 0.0))
        stricter = (not hi_corr_bear) or (
            (diff_val >= dyn_on_adj + 0.05) and
            (pcon_val is None or pcon_val >= 0.65) and
            (apdf_val is None or apdf_val >= 0.0) and
            (combo_val is None or combo_val >= 0.10)
        )
        on_main = (
            (diff_val >= dyn_on_adj) and
            pcon_ok and apdf_ok and stricter and breadth_gate and
            guard_val < guard_soft and mm_val < thresholds.get("mmOff", 0.96)
        )
        on_alt = (
            (not hi_corr_bear) and
            (risk_beta_flux[i] is not None and risk_beta_flux[i] >= 0.06) and
            (combo_val is None or combo_val >= 0.10) and
            guard_val < 0.90
        )
        strong_on = (
            (diff_val >= dyn_on_adj + 0.03) and
            ((kappa_val is None) or (kappa_val >= thresholds.get("kOn", 0.60))) and
            (pcon_val is None or pcon_val >= 0.65) and
            (vpc1_val >= thresholds.get("vOn", 0.05)) and
            guard_val < guard_soft and
            mm_val < thresholds.get("mmOff", 0.96)
        )
        on_raw = (on_main or on_alt or strong_on) and (not hi_corr_drift)

        guard_only = ((guard_val >= guard_hard) or (mm_val >= thresholds.get("mmOff", 0.96))) and not (diff_val <= dyn_off_flux)
        off_by_rel = (
            ((vpc1_val <= thresholds.get("vOff", -0.05)) and (abs(vpc1_val) >= 0.05)) or
            (diff_val <= dyn_off_flux and (kappa_val is None or kappa_val < 0.55))
        )
        off_raw = (
            off_by_rel or
            (diff_val <= dyn_off_flux) or
            guard_only or
            ((pcon_val is not None and pcon_val <= thresholds.get("pconOff", 0.40)) and mm_val >= 0.92)
        )

        risk3 = None
        idx_three = idx_price
        returns_list = []
        for sym in CLUSTERS["risk"]:
            series = prices.get(sym, [])
            ret = rolling_return(series, idx_three, 3) if isinstance(series, list) else None
            if ret is not None and np.isfinite(ret):
                returns_list.append(ret)
        if returns_list:
            risk3 = sum(returns_list) / len(returns_list)
        block_on_high_corr_down = risk3 is not None and risk3 <= 0 and mm_val >= 0.90

        raw_on = (on_raw and not block_on_high_corr_down)
        raw_off = off_raw
        if variant == "stab":
            if stab_up_trend:
                raw_off = False
            if stab_plunge:
                raw_off = True
            lag_down = int(stab_cfg.get("lagDown", 4))
            if not stab_up_trend and not stab_plunge and s_down_run >= max(1, lag_down):
                raw_off = True

        hi_corr_risk = (mm_val >= 0.90) or ((mm_trend[i] or 0) > 0.005) or ((full_flux_z[i] or 0) >= 1.5)
        accel = (flux_slope[i] or 0) > 0 and (mm_trend[i] or 0) <= 0
        strong_pcon = pcon_val is not None and pcon_val >= 0.68
        confirm_on_days = 1 if strong_on else (3 if hi_corr_risk else (1 if strong_pcon or accel else 2))

        on_cand = on_cand + 1 if raw_on else 0
        off_cand = off_cand + 1 if raw_off else 0

        if variant == "stab":
            s_now = Sseries[i] if np.isfinite(Sseries[i]) else None
            slope_abs = abs(s_slope[i]) if s_slope[i] is not None and np.isfinite(s_slope[i]) else None
            flux_abs = abs(jflux[i]) if jflux[i] is not None and np.isfinite(jflux[i]) else None
            mid_band = s_now is not None and s_neutral_lo <= s_now <= s_neutral_hi
            tiny_slope = slope_abs is not None and slope_abs < 0.005
            tiny_flux = flux_abs is not None and flux_abs < 0.03
            if mid_band and tiny_slope and tiny_flux:
                stab_neutral_run += 1
            else:
                stab_neutral_run = 0

        decided = prev_state
        if in_drift_epoch:
            decided = -1
        else:
            if prev_state == 1:
                if raw_off:
                    lag_up = max(1, int(stab_cfg.get("lagUp", 3)))
                    up_lead_active = (s_up_seq_values[i] >= lag_up)
                    if variant == "stab" and up_lead_active:
                        need = max(1, int(stab_cfg.get("upConfirmOffMin", 2)))
                        decided = -1 if off_cand >= need else 1
                    else:
                        decided = -1
                else:
                    decided = 1
            elif prev_state == -1:
                decided = 1 if on_cand >= confirm_on_days else -1
            else:
                if off_cand >= 1:
                    decided = -1
                elif on_cand >= confirm_on_days:
                    decided = 1
                else:
                    decided = 0

        if variant == "stab" and stab_neutral_run >= 2:
            decided = 0

        state[i] = decided
        prev_state = decided

        fragile[i] = decided >= 0 and (guard_val >= thresholds.get("mmFragile", 0.88) or (guard_val >= guard_soft and guard_val < guard_hard))
        if mm_val > 0:
            far[i] = abs(jflux[i]) / (mm_val + 1e-9) if jflux[i] is not None else None
        else:
            far[i] = None

    executed_state = [0] + state[:-1]

    result = {
        "dates": dates_str,
        "score": score,
        "riskScore": score,
        "state": state,
        "executedState": executed_state,
        "executed_state": executed_state,
        "fragile": fragile,
        "guard": guard,
        "mm": mm,
        "far": far,
        "fflFlux": jflux,
        "riskBetaFlux": risk_beta_flux,
        "apdf": apdf,
        "pcon": pcon,
        "diffusionScore": diffusion_scores,
        "fluxSlope": flux_slope,
        "mmTrend": mm_trend,
        "fullFlux": full_flux,
        "fullFluxZ": full_flux_z,
        "fluxRaw": flux_raw,
        "fluxIntensity": flux_intensity,
        "comboMomentum": combo_mom,
        "breadth": breadth,
        "vPC1": vpc1,
        "kappa": kappa,
        "coDownAll": co_down_all,
        "scCorr": sc_corr,
        "safeNeg": safe_neg_series,
        "diagnostics": {
            "fluxThresholds": {"on": dyn_on_flux, "off": dyn_off_flux},
            "scoreThresholds": {"on": dyn_on_score, "off": dyn_off_score},
            "scoreLatest": score[-1] if score else None,
        },
    }
    return result

def compute_fusion(classic: Tuple[List[str], List[float], List[int], List[float], List[float]],
                   ffl: Dict[str, Any],
                   qqq_prices: List[float],
                   window: int,
                   all_dates: List[str],
                   win: int = 40,
                   lam: float = 0.50,
                   tau: float = 4.0,
                   floor: float = 0.10) -> Dict[str, Any]:
    dates, c_score, c_state, _, _ = classic
    f_state = ffl.get("state", [])
    n = min(len(dates), len(f_state))
    # align returns to regime dates (use log returns of QQQ)
    # JS parity: use arithmetic returns for Fusion and align with gIdx-1
    def to_returns_arith(px: List[float]) -> List[float]:
        out: List[float] = []
        for i in range(1, len(px)):
            a, b = px[i - 1], px[i]
            if np.isfinite(a) and np.isfinite(b) and a != 0:
                out.append(b / a - 1.0)
            else:
                out.append(0.0)
        return out
    r = to_returns_arith(qqq_prices)
    # gIdx = index of first regime date within full analysis dates
    first_date = dates[0] if dates else None
    try:
        gIdx = all_dates.index(first_date) if (first_date is not None) else (window - 1)
    except ValueError:
        gIdx = window - 1
    start_idx = max(0, gIdx - 1)
    r_aligned = r[start_idx:start_idx + n]
    if len(r_aligned) < n:
        r_aligned = ([0.0] * (n - len(r_aligned))) + r_aligned
    if len(r_aligned) < n:
        r_aligned = ([0.0] * (n - len(r_aligned))) + r_aligned
    pred_c_now = [1 if s > 0 else (-1 if s < 0 else 0) for s in c_state[:n]]
    pred_f_now = [1 if s > 0 else (-1 if s < 0 else 0) for s in f_state[:n]]
    pred_c = [0] + pred_c_now[:-1]
    pred_f = [0] + pred_f_now[:-1]
    def rolling_hit(pred: List[int], ret: List[float], W: int) -> List[float]:
        out = [0.5] * len(ret)
        buf: List[Tuple[int, int]] = []
        hits = 0
        tot = 0
        for i in range(len(ret)):
            p = pred[i]
            y = 1 if ret[i] > 0 else (-1 if ret[i] < 0 else 0)
            if p != 0:
                ok = 1 if p == y else 0
                buf.append((ok, 1))
                hits += ok
                tot += 1
            else:
                buf.append((0, 0))
            if len(buf) > W:
                ok0, t0 = buf.pop(0)
                hits -= ok0
                tot -= t0
            out[i] = (hits / tot) if tot > 0 else 0.5
        return out
    def rolling_ic(pred: List[int], ret: List[float], W: int) -> List[float]:
        out = [0.0] * len(ret)
        bx: List[int] = []
        by: List[float] = []
        sum_xy = 0.0
        sum_y = 0.0
        sum_y2 = 0.0
        for i in range(len(ret)):
            x = pred[i]
            y = ret[i]
            bx.append(x)
            by.append(y)
            sum_xy += x * y
            sum_y += y
            sum_y2 += y * y
            if len(bx) > W:
                x0 = bx.pop(0)
                y0 = by.pop(0)
                sum_xy -= x0 * y0
                sum_y -= y0
                sum_y2 -= y0 * y0
            nwin = len(by)
            mu = sum_y / nwin if nwin else 0.0
            var = (sum_y2 / nwin) - mu * mu if nwin else 0.0
            sd = math.sqrt(max(1e-12, var))
            out[i] = (sum_xy / nwin) / sd if nwin and sd > 0 else 0.0
        return out
    hit_c = rolling_hit(pred_c, r_aligned, win)
    hit_f = rolling_hit(pred_f, r_aligned, win)
    ic_c = rolling_ic(pred_c, r_aligned, win)
    ic_f = rolling_ic(pred_f, r_aligned, win)
    score_c = [lam * (h - 0.5) + (1 - lam) * ic for h, ic in zip(hit_c, ic_c)]
    score_f = [lam * (h - 0.5) + (1 - lam) * ic for h, ic in zip(hit_f, ic_f)]
    w_c = []
    w_f = []
    for a, b in zip(score_c, score_f):
        ez_a = math.exp(tau * a)
        ez_b = math.exp(tau * b)
        wa = ez_a / (ez_a + ez_b)
        wb = 1.0 - wa
        wa = max(floor, wa)
        wb = max(floor, wb)
        s = wa + wb
        w_c.append(wa / s)
        w_f.append(wb / s)
    fused_raw = [w_c[i] * pred_c_now[i] + w_f[i] * pred_f_now[i] for i in range(n)]
    mm_arr = ffl.get("mm", [])
    mm = [(mm_arr[i] if i < len(mm_arr) else None) for i in range(n)]
    mm_frag = RISK_CFG_FFL["thresholds"]["mmFragile"]
    mm_off = RISK_CFG_FFL["thresholds"]["mmOff"]
    state = []
    for i in range(n):
        v = fused_raw[i]
        m = mm[i] if (mm[i] is not None and np.isfinite(mm[i])) else 0.0
        if m >= mm_off:
            state.append(-1)
        elif m >= mm_frag and v > 0:
            state.append(0)
        elif v >= 0.20:
            state.append(1)
        elif v <= -0.20:
            state.append(-1)
        else:
            state.append(0)
    exec_state = [0] + state[:-1]
    score = [max(0.0, min(1.0, (v + 1.0) / 2.0)) for v in fused_raw]
    return {
        "dates": dates[:n],
        "state": state,
        "executed_state": exec_state,
        "score": score,
        "wClassic": w_c,
        "wFFL": w_f,
    }


def leveraged_return(r: float, leverage: int = 3) -> float:
    return max(-0.99, leverage * r)


def backtest_from_state(
    prices_close: List[float],
    dates: List[str],
    state: List[int],
    leverage: int = 3,
    delay_days: int = 1,
    *,
    price_mode: str = "close",
    prices_open: Optional[List[float]] = None,
    strategy_close: Optional[List[float]] = None,
    strategy_open: Optional[List[float]] = None,
    benchmark_close: Optional[List[float]] = None,
    benchmark_open: Optional[List[float]] = None,
    neutral_bench_weight: Optional[float] = None,
    risk_on_bench_weight: Optional[float] = None,
) -> Dict[str, Any]:
    # Use arithmetic returns to match JS/UI backtest and CSV expectations
    mode = "open" if price_mode == "open" else "close"

    def _select_series(close: Optional[List[float]], open_: Optional[List[float]], fallback: Optional[List[float]]) -> List[float]:
        candidate = close if isinstance(close, list) and close else fallback
        open_candidate = open_ if isinstance(open_, list) and open_ else None
        if mode == "open" and open_candidate:
            return open_candidate
        return candidate or []

    def _to_returns(series: Optional[List[float]]) -> List[float]:
        out: List[float] = [0.0]
        if not isinstance(series, list):
            return out
        for i in range(1, len(series)):
            a, b = series[i - 1], series[i]
            if np.isfinite(a) and np.isfinite(b) and a != 0:
                out.append(b / a - 1.0)
            else:
                out.append(0.0)
        return out

    series_base = _select_series(prices_close, prices_open, prices_close)
    series_strategy = _select_series(strategy_close, strategy_open, series_base)
    series_bench = _select_series(benchmark_close, benchmark_open, prices_close)

    rets_base = _to_returns(series_base)
    rets_strategy = _to_returns(series_strategy)
    rets_bench = _to_returns(series_bench)

    neutral_weight = (
        NEUTRAL_BENCH_WEIGHT if neutral_bench_weight is None else neutral_bench_weight
    )
    risk_on_weight = (
        RISK_ON_BENCH_WEIGHT if risk_on_bench_weight is None else risk_on_bench_weight
    )

    # Align: dates length == len(series_base). Strategy rets length equals len(dates)
    # We map each regime day to return at same index (already aligned to window slicing externally)
    strat_eq: List[float] = []
    bh_eq: List[float] = []
    asset_eq: List[float] = []
    s = 1.0
    b = 1.0
    a = 1.0
    aligned_prices_bench: List[Optional[float]] = []
    aligned_prices_strategy: List[Optional[float]] = []

    def _aligned_price(series: List[float], idx: int) -> Optional[float]:
        if not isinstance(series, list) or not series:
            return None
        # Use idx+1 to align with return at position idx; fallback to last available
        target = idx + 1
        if target < len(series):
            return series[target]
        if len(series) >= 1:
            return series[-1]
        return None

    use_strategy = any(abs(x) > 1e-12 for x in rets_strategy[1:]) if len(rets_strategy) > 1 else False
    # Build executed state with delay
    exec_state = [0] * len(state)
    for i in range(len(state)):
        j = i - delay_days
        exec_state[i] = state[j] if j >= 0 else 0
    for i in range(len(state)):
        r_base = rets_base[i] if i < len(rets_base) else 0.0
        r_strategy = rets_strategy[i] if i < len(rets_strategy) else r_base
        if not use_strategy:
            r_strategy = r_base
        r_bench = rets_bench[i] if i < len(rets_bench) else r_base
        if exec_state[i] > 0:
            s *= 1.0 + risk_on_weight * r_strategy
        elif exec_state[i] < 0:
            s *= 1.0  # cash
        else:
            s *= 1.0 + neutral_weight * r_strategy
        b *= 1.0 + r_bench
        a *= 1.0 + r_strategy
        strat_eq.append(s)
        bh_eq.append(b)
        asset_eq.append(a)
        aligned_prices_bench.append(_aligned_price(series_bench, i))
        aligned_prices_strategy.append(_aligned_price(series_strategy, i))
    return {
        "dates": dates,
        "equity_strategy": strat_eq,
        "equity_bh": bh_eq,
        "equity_asset": asset_eq,
        "prices_benchmark": aligned_prices_bench,
        "prices_strategy": aligned_prices_strategy,
        "cum_strategy": strat_eq[-1] if strat_eq else 1.0,
        "cum_bh": bh_eq[-1] if bh_eq else 1.0,
        "price_mode": mode,
    }


def compute_realtime_regime(window: int = 30, use_realtime: bool = True, years: int = 5) -> Dict[str, Any]:
    """Fetch prices (FMP) and compute Classic, FFL-STAB, and FLL-Fusion with optional realtime patch.

    Returns a dict suitable for UI rendering.
    """
    if window not in WINDOWS:
        window = 30
    effective_years = max(years, MIN_HISTORY_YEARS)
    # Use tz-naive UTC date to avoid tz-aware vs tz-naive comparisons
    from datetime import datetime
    today_utc_naive = pd.Timestamp(datetime.utcnow().date())  # naive UTC calendar day
    from_date = today_utc_naive - pd.Timedelta(days=365 * effective_years + MIN_HISTORY_LEAD_DAYS)
    if from_date > MIN_HISTORY_START:
        from_date = MIN_HISTORY_START
    from_str = from_date.strftime("%Y-%m-%d")
    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set. Please add it to .env")
    # 1) Daily history (adj close + adj open)
    hist_frames = fetch_daily_history_fmp(ALL_SYMBOLS, from_str, api_key)
    close_map: Dict[str, pd.Series] = {}
    open_map: Dict[str, pd.Series] = {}
    for sym in ALL_SYMBOLS:
        df = hist_frames.get(sym)
        if df is None or df.empty:
            continue
        close_series = df["adj_close"].astype(float)
        open_series = df.get("adj_open")
        if open_series is not None:
            open_series = open_series.astype(float)
        else:
            open_series = pd.Series(index=close_series.index, dtype=float)
        # Fallback: fill missing adjusted open with adjusted close
        open_series = open_series.fillna(close_series)
        close_map[sym] = close_series
        open_map[sym] = open_series
    # 2) Apply realtime patch on raw close_map (before alignment).
    patched_close_map = close_map
    if use_realtime:
        try:
            quotes = fetch_realtime_quotes_fmp(ALL_SYMBOLS, api_key)
            patched_close_map = patch_with_realtime_last_price(close_map, quotes)
        except Exception:
            # best-effort; keep daily
            patched_close_map = close_map
    # 3) Align by intersection like JS (only dates that actually exist for all symbols).
    #    This naturally removes BTC 주말/공휴일 인덱스와 ETF 휴장일을 제외해 precomputed와 동일한 표본을 구성.
    inter_dates, inter_series_close = _align_by_intersection(patched_close_map, ALL_SYMBOLS)
    if not inter_dates or len(inter_dates) < window + 5:
        raise RuntimeError("Insufficient intersected history after alignment")
    idx_inter = pd.Index(inter_dates)
    # Reindex opens to the same intersection; fill gaps with corresponding close.
    inter_series_open: Dict[str, pd.Series] = {}
    for sym in ALL_SYMBOLS:
        base_open = open_map.get(sym, pd.Series(dtype=float))
        aligned_open = base_open.reindex(idx_inter).ffill()
        inter_series_open[sym] = aligned_open.fillna(inter_series_close[sym])
    dates = [d.strftime("%Y-%m-%d") for d in inter_dates]
    prices = {s: inter_series_close[s].astype(float).tolist() for s in ALL_SYMBOLS if s in inter_series_close}
    prices_open = {s: inter_series_open[s].astype(float).tolist() for s in ALL_SYMBOLS if s in inter_series_open}
    # Compute window metrics for SIGNAL set
    date_idx = inter_dates
    metrics = compute_window_metrics(prices, date_idx, window)
    # Classic
    classic = compute_classic(metrics)
    # FFL port (stab variant default for UI parity)
    ffl = compute_ffl(prices, date_idx, metrics, window, variant="stab")
    # FLL-Fusion
    all_dates = [d.strftime("%Y-%m-%d") for d in date_idx]
    fusion = compute_fusion(classic, ffl, prices[BASE_SYMBOL], window, all_dates)
    # Backtests (provide both 1d and 2d delays)
    # JS parity: segment should start at (window-1) to yield R+1 prices for R records
    start_idx = max(0, window - 1)
    base_close_full = prices.get(BASE_SYMBOL, [])
    base_open_full = prices_open.get(BASE_SYMBOL, [])
    strategy_close_full = prices.get(STRATEGY_SYMBOL, []) if STRATEGY_SYMBOL in prices else None
    strategy_open_full = prices_open.get(STRATEGY_SYMBOL, []) if STRATEGY_SYMBOL in prices_open else None
    bench_close_full = prices.get(BENCH_SYMBOL, []) if BENCH_SYMBOL in prices else None
    bench_open_full = prices_open.get(BENCH_SYMBOL, []) if BENCH_SYMBOL in prices_open else None

    def _slice(series: Optional[List[float]]) -> List[float]:
        if not isinstance(series, list):
            return []
        if start_idx <= 0:
            return series[:]
        if start_idx >= len(series):
            return []
        return series[start_idx:]

    qqq_segment = _slice(base_close_full)
    qqq_open_segment = _slice(base_open_full)
    strategy_segment = _slice(strategy_close_full) if strategy_close_full is not None else None
    strategy_open_segment = _slice(strategy_open_full) if strategy_open_full is not None else None
    bench_segment = _slice(bench_close_full) if bench_close_full is not None else None
    bench_open_segment = _slice(bench_open_full) if bench_open_full is not None else None

    def _build_synthetic(base: List[float], factor: float = 3.0) -> List[float]:
        if not base:
            return []
        out = [float(base[0])]
        level = float(base[0])
        for i in range(1, len(base)):
            prev = base[i - 1]
            curr = base[i]
            if isinstance(prev, (int, float)) and isinstance(curr, (int, float)) and prev:
                ret = curr / prev - 1.0
                level *= max(0.01, 1.0 + factor * ret)
            out.append(level)
        return out

    def _blend_strategy(base: List[float], existing: Optional[List[float]], factor: float = 3.0) -> List[float]:
        if not base:
            return []
        synthetic = _build_synthetic(base, factor=factor)
        if not existing:
            return synthetic
        blended: List[float] = []
        for i in range(len(base)):
            if i < len(existing) and isinstance(existing[i], (int, float)):
                blended.append(float(existing[i]))
            else:
                blended.append(synthetic[i])
        return blended

    strategy_segment = _blend_strategy(qqq_segment, strategy_segment, factor=3.0)
    strategy_open_segment = _blend_strategy(qqq_open_segment, strategy_open_segment, factor=3.0)
    bt0_close = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=0,
        price_mode="close",
        strategy_close=strategy_segment,
        strategy_open=strategy_open_segment,
        benchmark_close=bench_segment,
        benchmark_open=None,
        neutral_bench_weight=NEUTRAL_BENCH_WEIGHT,
        risk_on_bench_weight=RISK_ON_BENCH_WEIGHT,
    )
    bt1_close = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=1,
        price_mode="close",
        strategy_close=strategy_segment,
        strategy_open=strategy_open_segment,
        benchmark_close=bench_segment,
        benchmark_open=None,
        neutral_bench_weight=NEUTRAL_BENCH_WEIGHT,
        risk_on_bench_weight=RISK_ON_BENCH_WEIGHT,
    )
    bt2_close = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=2,
        price_mode="close",
        strategy_close=strategy_segment,
        strategy_open=strategy_open_segment,
        benchmark_close=bench_segment,
        benchmark_open=None,
        neutral_bench_weight=NEUTRAL_BENCH_WEIGHT,
        risk_on_bench_weight=RISK_ON_BENCH_WEIGHT,
    )
    bt1_open = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=1,
        price_mode="open",
        prices_open=qqq_open_segment,
        strategy_close=strategy_segment,
        strategy_open=strategy_open_segment,
        benchmark_close=bench_segment,
        benchmark_open=bench_open_segment,
        neutral_bench_weight=NEUTRAL_BENCH_WEIGHT,
        risk_on_bench_weight=RISK_ON_BENCH_WEIGHT,
    )
    bt2_open = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=2,
        price_mode="open",
        prices_open=qqq_open_segment,
        strategy_close=strategy_segment,
        strategy_open=strategy_open_segment,
        benchmark_close=bench_segment,
        benchmark_open=bench_open_segment,
        neutral_bench_weight=NEUTRAL_BENCH_WEIGHT,
        risk_on_bench_weight=RISK_ON_BENCH_WEIGHT,
    )
    series_payload: Dict[str, List[float]] = {BASE_SYMBOL: qqq_segment}
    if STRATEGY_SYMBOL and strategy_segment:
        series_payload[STRATEGY_SYMBOL] = strategy_segment
    if BENCH_SYMBOL and bench_segment:
        series_payload[BENCH_SYMBOL] = bench_segment
    series_open_payload: Dict[str, List[float]] = {BASE_SYMBOL: qqq_open_segment}
    if STRATEGY_SYMBOL and strategy_open_segment:
        series_open_payload[STRATEGY_SYMBOL] = strategy_open_segment
    if BENCH_SYMBOL and bench_open_segment:
        series_open_payload[BENCH_SYMBOL] = bench_open_segment
    return {
        "window": window,
        "dates": ffl["dates"],
        "classic": {"score": classic[1], "state": classic[2]},
        "ffl_stab": ffl,
        "fusion": fusion,
        "series": series_payload,
        "series_open": series_open_payload,
        "stability": [float(r.get("stability", np.nan)) for r in metrics["records"]],
        "smoothed": [float(r.get("smoothed", np.nan)) for r in metrics["records"]],
        "delta": [float(r.get("delta", np.nan)) for r in metrics["records"]],
        "sub": {
            "stockCrypto": [float(r.get("sub", {}).get("stockCrypto", np.nan)) for r in metrics["records"]],
            "traditional": [float(r.get("sub", {}).get("traditional", np.nan)) for r in metrics["records"]],
            "safeNegative": [float(r.get("sub", {}).get("safeNegative", np.nan)) for r in metrics["records"]],
        },
        "backtest": {
            "ffl_stab": {
                "delay0": bt0_close,
                "delay1": bt1_close,
                "delay2": bt2_close,
                "delay0_close": bt0_close,
                "delay1_close": bt1_close,
                "delay2_close": bt2_close,
                "delay1_open": bt1_open,
                "delay2_open": bt2_open,
            },
        },
    }


if __name__ == "__main__":
    # Simple CLI for debugging
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--no-realtime", action="store_true")
    args = parser.parse_args()
    res = compute_realtime_regime(window=args.window, use_realtime=(not args.no_realtime))
    print(json.dumps({k: (len(v) if isinstance(v, list) else (list(v.keys()) if isinstance(v, dict) else type(v).__name__)) for k, v in res.items()}, ensure_ascii=False, indent=2))
