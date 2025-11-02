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
RISK_SYMBOLS = ["IWM", "SPY", "BTC-USD"]
SAFE_SYMBOLS = ["TLT", "GLD"]
ALL_SYMBOLS = ["QQQ", "IWM", "SPY", "TLT", "GLD", "BTC-USD"]
SIGNAL_SYMBOLS = ["IWM", "SPY", "TLT", "GLD", "BTC-USD"]

# Window options
WINDOWS = [20, 30, 60]

# FFL baseline knobs (subset, aligned with JS defaults where practical)
RISK_CFG_FFL = {
    "lookbacks": {"momentum": 10, "vol": 20},
    "p": 1.5,
    "zSat": 2.0,
    "lambda": 0.25,
    "thresholds": {
        "scoreOn": 0.60,
        "scoreOff": 0.40,
        "jOn": +0.10,
        "jOff": -0.08,
        "breadthOn": 0.50,
        "mmFragile": 0.88,
        "mmOff": 0.96,
        "vOn": +0.05,
        "vOff": -0.05,
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


def to_returns(prices: List[float]) -> List[float]:
    arr: List[float] = []
    for i in range(1, len(prices)):
        a, b = prices[i - 1], prices[i]
        if np.isfinite(a) and np.isfinite(b) and a != 0:
            arr.append(math.log(b / a))
        else:
            arr.append(0.0)
    return arr


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
        url = f"{base}{_map_symbol_for_fmp(sym)}?from={from_date}&apikey={api_key}"
        r = sess.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        hist = data.get("historical") or []
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
        df["date"] = pd.to_datetime(df["date"])
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
        price = float(row.get("price") or row.get("previousClose") or 0.0)
        ts = row.get("timestamp") or row.get("lastUpdated")
        try:
            t = pd.to_datetime(ts, unit="s", utc=True) if isinstance(ts, (int, float)) else pd.to_datetime(ts, utc=True)
        except Exception:
            t = pd.Timestamp.utcnow()
        prev_close = row.get("previousClose")
        out[sym] = (t, price, float(prev_close) if isinstance(prev_close, (int, float)) else None)
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
        idx = pd.to_datetime(s.index).tz_localize(None).normalize()
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
    - record date = dates[end+1]
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
        records.append({
            # record date aligns to returns index end -> prices index end+1
            "date": dates[end + 1].strftime("%Y-%m-%d"),
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
    # Port of computeRiskSeriesFFL with classic/exp/stab variants (simplified parity where possible)
    recs = metrics["records"]
    n = len(recs)
    labels = SIGNAL_SYMBOLS
    idx_map = {s: i for i, s in enumerate(labels)}
    window_offset = window - 1
    # z-momentum per asset for entire date range
    k = RISK_CFG_FFL["lookbacks"]["momentum"]
    v = RISK_CFG_FFL["lookbacks"]["vol"]
    z: Dict[str, List[Optional[float]]] = {s: [None] * len(dates) for s in labels}
    for s in labels:
        series = prices[s]
        for i in range(len(dates)):
            z[s][i] = z_momentum(series, i, k, v, RISK_CFG_FFL["zSat"])
    # --- BearGuard: downside-asymmetry damper ---
    dates_idx = pd.to_datetime(pd.Index(dates))
    risk_candidates = ["QQQ", "IWM", "BTC-USD"]
    risk_set = [s for s in risk_candidates if s in prices]
    r_star_series: Optional[pd.Series] = None
    if risk_set:
        for sym in risk_set:
            px = pd.Series(prices[sym], index=dates_idx)
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

    def _bear_damp(idx: int, R_thr: float = 1.2, beta: float = 0.5, gmin: float = 0.5) -> float:
        if 0 <= idx < len(R_star):
            x = max(0.0, float(R_star[idx]) - R_thr)
            return max(gmin, 1.0 / (1.0 + beta * x))
        return 1.0
    # --- end BearGuard ---
    # Stability slope z for STAB/EXP rules
    Svals = [float(r.get("stability", np.nan)) for r in recs]
    s_fast = ema(Svals, RISK_CFG_FFL["stabTune"]["fast"])
    s_slow = ema(Svals, RISK_CFG_FFL["stabTune"]["slow"])
    s_slope = [((s_fast[i] - s_slow[i]) if np.isfinite(s_fast[i]) and np.isfinite(s_slow[i]) else np.nan) for i in range(n)]
    s_z: List[Optional[float]] = [None] * n
    for i in range(n):
        s_z[i] = rolling_z(s_slope, i, RISK_CFG_FFL["stabTune"]["zWin"])
    # Iterate records for flux/guard metrics
    mm: List[Optional[float]] = [None] * n
    mm_trend: List[Optional[float]] = [None] * n
    jflux: List[Optional[float]] = [None] * n
    flux_raw: List[Optional[float]] = [None] * n
    flux_intensity: List[Optional[float]] = [None] * n
    flux_slope: List[Optional[float]] = [None] * n
    guard: List[Optional[float]] = [None] * n
    score: List[Optional[float]] = [None] * n
    apdf: List[Optional[float]] = [None] * n
    pcon: List[Optional[float]] = [None] * n
    vpc1: List[Optional[float]] = [None] * n
    kappa: List[Optional[float]] = [None] * n
    risk_beta_flux: List[Optional[float]] = [None] * n
    co_down_all: List[Optional[float]] = [None] * n
    combo_mom: List[Optional[float]] = [None] * n
    breadth: List[Optional[float]] = [None] * n
    prev_mat: Optional[np.ndarray] = None
    for i, r in enumerate(recs):
        mat = r["matrix"]
        # Absorption (market mode)
        mm[i] = top_eigenvalue_ratio(mat)
        mm_trend[i] = (mm[i] - mm[i - 1]) if i > 0 and np.isfinite(mm[i]) and np.isfinite(mm[i - 1]) else None
        # Flux accumulators
        p = RISK_CFG_FFL["p"]
        num = 0.0
        den = 0.0
        abs_sum = 0.0
        apdf_w = 0.0
        pcon_w = 0.0
        w_all = 0.0
        # risk-safe pairs
        for s_safe in SAFE_SYMBOLS:
            for s_risk in RISK_SYMBOLS:
                ia = idx_map[s_safe]; ib = idx_map[s_risk]
                coef = float(mat[ia, ib]) if np.isfinite(mat[ia, ib]) else 0.0
                w = abs(coef) ** p
                # JS mapping: record i corresponds to price index (window_offset + i + 1)
                pi = window_offset + i + 1
                zi = z[s_risk][pi] if pi < len(dates) else None
                zs = z[s_safe][pi] if pi < len(dates) else None
                if zi is None or zs is None:
                    continue
                diff = float(zi) - float(zs)
                num += w * diff
                den += w
                abs_sum += w * abs(diff)
                apdf_w += w * (float(zi) - float(zs))
                pcon_w += w * (1.0 if float(zi) > float(zs) else 0.0)
                w_all += w
        # risk-risk pairs (APDF/PCON)
        for a in range(len(RISK_SYMBOLS)):
            for b in range(a + 1, len(RISK_SYMBOLS)):
                ia = idx_map[RISK_SYMBOLS[a]]; ib = idx_map[RISK_SYMBOLS[b]]
                coef = float(mat[ia, ib]) if np.isfinite(mat[ia, ib]) else 0.0
                w = abs(coef) ** p
                pi = window_offset + i + 1
                za = z[RISK_SYMBOLS[a]][pi] if pi < len(dates) else None
                zb = z[RISK_SYMBOLS[b]][pi] if pi < len(dates) else None
                if za is None or zb is None:
                    continue
                apdf_w += w * (0.5 * (float(za) + float(zb)))
                pcon_w += w * (1.0 if (float(za) > 0 and float(zb) > 0) else 0.0)
                w_all += w
        # safe-safe pairs (APDF/PCON)
        for a in range(len(SAFE_SYMBOLS)):
            for b in range(a + 1, len(SAFE_SYMBOLS)):
                ia = idx_map[SAFE_SYMBOLS[a]]; ib = idx_map[SAFE_SYMBOLS[b]]
                coef = float(mat[ia, ib]) if np.isfinite(mat[ia, ib]) else 0.0
                w = abs(coef) ** p
                pi = window_offset + i + 1
                za = z[SAFE_SYMBOLS[a]][pi] if pi < len(dates) else None
                zb = z[SAFE_SYMBOLS[b]][pi] if pi < len(dates) else None
                if za is None or zb is None:
                    continue
                apdf_w += w * (-0.5 * (float(za) + float(zb)))
                pcon_w += w * (1.0 if (float(za) <= 0 and float(zb) <= 0) else 0.0)
                w_all += w
        jbar = (num / den) if den > 0 else 0.0
        jn = _tanh_clip(jbar, RISK_CFG_FFL["lambda"]) if den > 0 else 0.0
        jflux[i] = jn * _bear_damp(window_offset + i)
        flux_raw[i] = (jbar if den > 0 else 0.0)
        flux_intensity[i] = (abs_sum / den) if den > 0 else 0.0
        flux_slope[i] = (jflux[i] - jflux[i - 1]) if i > 0 and jflux[i] is not None and jflux[i - 1] is not None else None
        # APDF/PCON finalization
        apdf[i] = float(np.tanh((apdf_w / w_all) / (RISK_CFG_FFL["lambda"] or 0.25))) if w_all > 0 else None
        pcon[i] = float(min(1.0, max(0.0, pcon_w / w_all))) if w_all > 0 else None
        # v_PC1
        try:
            e1 = top_eigenvector(mat)
            if e1 is not None:
                numv = 0.0; denv = 0.0
                for ksym, wv in zip(labels, e1):
                    pi = window_offset + i + 1
                    zi = z[ksym][pi] if pi < len(dates) else None
                    zi = float(zi) if zi is not None and np.isfinite(zi) else 0.0
                    numv += float(wv) * zi
                    denv += abs(float(wv))
                vproj = (numv / denv) if denv > 0 else 0.0
                vpc1[i] = float(np.tanh(vproj / 0.5))
            else:
                vpc1[i] = None
        except Exception:
            vpc1[i] = None
        # coDownAll / combo / breadth
        all_z = []
        risk_z = []
        for s in labels:
            pi = window_offset + i + 1
            zi = z[s][pi] if pi < len(dates) else None
            if zi is not None and np.isfinite(zi):
                all_z.append(float(zi))
        for s in RISK_SYMBOLS:
            pi = window_offset + i + 1
            zi = z[s][pi] if pi < len(dates) else None
            if zi is not None and np.isfinite(zi):
                risk_z.append(float(zi))
        co_down_all[i] = (sum(1 for v in all_z if v < 0) / len(all_z)) if all_z else None
        combo_mom[i] = (float(np.mean(risk_z)) if risk_z else None)
        breadth[i] = (sum(1 for v in risk_z if v > 0) / len(risk_z)) if risk_z else None
        # guard
        safe_neg = float(r.get("sub", {}).get("safeNegative", 0.0))
        mm_pen = 0.0 if not np.isfinite(mm[i]) else min(1.0, max(0.0, (mm[i] - 0.85) / (0.97 - 0.85)))
        safe_pen = min(1.0, max(0.0, (safe_neg - 0.35) / (0.60 - 0.35)))
        delta_pen = 0.0
        if np.isfinite(r.get("delta", np.nan)):
            delta_pen = min(1.0, max(0.0, max(0.0, -float(r["delta"])) / (0.05 - 0.015)))
        zf = rolling_z(metrics.get("fullFlux", []), i, min(63, max(15, n // 4)))
        flux_guard = 1 / (1 + math.exp(-0.85 * zf)) if zf is not None else 1.0
        guard_val = 0.4 * mm_pen + 0.2 * safe_pen + 0.2 * delta_pen + 0.2 * flux_guard
        guard[i] = float(guard_val)
        # flux score components
        flux_score = 0.5 * (1 + (jflux[i] if jflux[i] is not None else 0.0))
        combo_norm = 0.5 * (1 + (combo_mom[i] if combo_mom[i] is not None else 0.0))
        breadth_norm = (breadth[i] if breadth[i] is not None else 0.0)
        guard_relief = max(0.0, 1.0 - guard_val)
        score[i] = float(0.5 * flux_score + 0.2 * combo_norm + 0.2 * breadth_norm + 0.1 * guard_relief)
        # risk beta flux (requires prev mat)
        if prev_mat is not None:
            rbw = 0.0
            rbsum = 0.0
            for a in range(len(RISK_SYMBOLS)):
                for b in range(a + 1, len(RISK_SYMBOLS)):
                    ia = idx_map[RISK_SYMBOLS[a]]; ib = idx_map[RISK_SYMBOLS[b]]
                    curr = float(mat[ia, ib]) if np.isfinite(mat[ia, ib]) else None
                    prev = float(prev_mat[ia, ib]) if np.isfinite(prev_mat[ia, ib]) else None
                    if curr is None or prev is None:
                        continue
                    delta = curr - prev
                    w = abs(curr) ** p
                    dir_sum = 0.0
                    za = z[RISK_SYMBOLS[a]][window_offset + i]
                    zb = z[RISK_SYMBOLS[b]][window_offset + i]
                    if za is not None:
                        dir_sum += float(za)
                    if zb is not None:
                        dir_sum += float(zb)
                    dir_sign = 1.0 if dir_sum >= 0 else -1.0
                    rbw += w
                    rbsum += w * delta * dir_sign
            rb_bar = (rbsum / rbw) if rbw > 0 else 0.0
            risk_beta_flux[i] = float(np.tanh(rb_bar / (RISK_CFG_FFL["lambda"] or 0.25)))
        else:
            risk_beta_flux[i] = None
        prev_mat = mat
    # dynamic thresholds (variant != classic)
    th = RISK_CFG_FFL["thresholds"]
    valid_flux = [float(x) for x in jflux if x is not None and np.isfinite(x)]
    valid_score = [float(x) for x in score if x is not None and np.isfinite(x)]
    dyn_on_flux = th["jOn"]
    dyn_off_flux = th["jOff"]
    dyn_on_score = th["scoreOn"]
    dyn_off_score = th["scoreOff"]
    if variant != "classic" and len(valid_flux) >= 50:
        dyn_on_flux = max(th["jOn"], quantile(valid_flux, 0.715))
        dyn_off_flux = min(th["jOff"], quantile(valid_flux, 0.244))
    if variant != "classic" and len(valid_score) >= 50:
        dyn_on_score = max(th["scoreOn"], quantile(valid_score, 0.75))
        dyn_off_score = min(th["scoreOff"], quantile(valid_score, 0.25))
    # state machine
    state: List[int] = []
    prev = 0
    drift_seq = 0
    drift_cool = 0
    in_drift = False
    def bench_ret(sym: str, i_local: int, lb: int) -> Optional[float]:
        idx = window_offset + i_local
        return rolling_return(prices[sym], idx, lb)
    for i in range(n):
        mmv = mm[i] if (mm[i] is not None and np.isfinite(mm[i])) else 0.0
        gv = guard[i] if guard[i] is not None else 1.0
        jv = jflux[i] if jflux[i] is not None else 0.0
        sc = score[i] if score[i] is not None else 0.5
        combo = combo_mom[i] if combo_mom[i] is not None else 0.0
        br = breadth[i] if breadth[i] is not None else 0.0
        # hi-corr bear & drift
        b10 = bench_ret("QQQ", i, 10)
        b20 = bench_ret("QQQ", i, 20)
        hi_corr_bear = (mmv >= 0.90) and ((b10 if b10 is not None else -1e-9) <= 0)
        hi_corr_drift = ((co_down_all[i] is not None and co_down_all[i] >= 0.60 and jv <= 0) or (mmv >= 0.90 and (b20 if b20 is not None else -1e-9) <= 0))
        if hi_corr_drift:
            drift_seq += 1; drift_cool = 0
        else:
            drift_seq = 0; drift_cool += 1
        if drift_seq >= (RISK_CFG_FFL["thresholds"].get("driftMinDays", 3)):
            in_drift = True
        if in_drift and drift_cool >= (RISK_CFG_FFL["thresholds"].get("driftCool", 2)):
            in_drift = False
        # EXP veto/booster placeholders (parity close; full EXP timing can be added)
        exp_ok_on = True
        exp_force_off = False
        # main gates
        pcon_ok = (pcon[i] is None) or (pcon[i] >= (RISK_CFG_FFL.get("thresholds", {}).get("pconOn", 0.55))) or (jv >= dyn_on_flux + 0.07)
        apdf_ok = (apdf[i] is None) or (apdf[i] >= -0.05)
        dyn_on_adj = dyn_on_flux + (0.05 if mmv >= 0.94 else (0.03 if mmv >= 0.90 else 0.0))
        stricter = (not hi_corr_bear) or ( (jv >= dyn_on_adj + 0.05) and ((pcon[i] or 1.0) >= 0.65) and ((apdf[i] or 0.0) >= 0.0) and (combo >= 0.10) )
        on_main = (jv >= dyn_on_adj) and pcon_ok and apdf_ok and stricter and (gv < 0.95) and (mmv < th["mmOff"]) and (br >= (th.get("breadthOn", 0.5) * 0.6)) and exp_ok_on
        # alt/strong on
        on_alt = (not hi_corr_bear) and (risk_beta_flux[i] is not None and risk_beta_flux[i] >= 0.06) and (combo >= 0.10) and (gv < 0.90)
        kappa_val = None
        if vpc1[i] is not None and jflux[i] is not None:
            lam = 1.0
            kappa_val = abs(jflux[i]) / (abs(vpc1[i]) * lam + 1e-6)
            kappa[i] = kappa_val
        strong_on = (jv >= dyn_on_adj + 0.03) and ((kappa_val is None) or (kappa_val >= 0.60)) and ((pcon[i] or 1.0) >= 0.65) and ((vpc1[i] or 0.0) >= th.get("vOn", 0.05)) and (gv < 0.95) and (mmv < th["mmOff"])
        on_raw = (on_main or on_alt or strong_on) and (not hi_corr_drift)
        # off gates
        guard_only = ((gv >= 0.98) or (mmv >= th["mmOff"])) and not (jv <= dyn_off_flux)
        off_by_rel = (((vpc1[i] or 0.0) <= th.get("vOff", -0.05)) and (abs(vpc1[i] or 0.0) >= 0.05)) or ((jv <= dyn_off_flux) and ((kappa_val or 0.0) < 0.55)) or exp_force_off
        off_raw = off_by_rel or (jv <= dyn_off_flux) or guard_only or ((pcon[i] or 1.0) <= (RISK_CFG_FFL.get("thresholds", {}).get("pconOff", 0.40)) and mmv >= 0.92)
        # decisions
        decided = prev
        if in_drift:
            decided = -1
        else:
            if prev == 1:
                decided = -1 if off_raw else 1
            elif prev == -1:
                decided = 1 if on_raw else -1
            else:
                decided = -1 if off_raw else (1 if on_raw else 0)
        state.append(decided)
        prev = decided
    executed_state = [0] + state[:-1]
    return {
        "dates": [r["date"] for r in recs],
        "score": score,
        "state": state,
        "executedState": executed_state,
        "mm": mm,
        "mmTrend": mm_trend,
        "guard": guard,
        "fflFlux": jflux,
        "fluxIntensity": flux_intensity,
        "fluxSlope": flux_slope,
        "riskBetaFlux": risk_beta_flux,
        "apdf": apdf,
        "pcon": pcon,
        "vPC1": vpc1,
        "kappa": kappa,
        "fluxRaw": flux_raw,
        "coDownAll": co_down_all,
        "comboMomentum": combo_mom,
        "breadth": breadth,
        "stabZ": s_z,
        "diagnostics": {
            "fluxThresholds": {"on": dyn_on_flux, "off": dyn_off_flux},
            "scoreThresholds": {"on": dyn_on_score, "off": dyn_off_score},
        },
    }
    recs = metrics["records"]
    full_flux = metrics.get("fullFlux", [])
    idx_map = {s: i for i, s in enumerate(SIGNAL_SYMBOLS)}
    window_offset = window - 1
    # z-momentum per asset
    k = RISK_CFG_FFL["lookbacks"]["momentum"]
    v = RISK_CFG_FFL["lookbacks"]["vol"]
    z: Dict[str, List[Optional[float]]] = {s: [None] * len(dates) for s in SIGNAL_SYMBOLS}
    for s in SIGNAL_SYMBOLS:
        series = prices[s]
        for i in range(len(dates)):
            idx = i
            val = z_momentum(series, idx, k, v, RISK_CFG_FFL["zSat"])
            z[s][i] = val
    # mm, guard pieces, flux
    mm: List[Optional[float]] = [None] * len(recs)
    jflux: List[Optional[float]] = [None] * len(recs)
    fint: List[Optional[float]] = [None] * len(recs)
    guard: List[Optional[float]] = [None] * len(recs)
    score: List[Optional[float]] = [None] * len(recs)
    # stability slope z (stab tune)
    Svals = [float(r.get("stability", np.nan)) for r in recs]
    S_fast = ema(Svals, RISK_CFG_FFL["stabTune"]["fast"])
    S_slow = ema(Svals, RISK_CFG_FFL["stabTune"]["slow"])
    S_slope = [((S_fast[i] - S_slow[i]) if np.isfinite(S_fast[i]) and np.isfinite(S_slow[i]) else np.nan) for i in range(len(recs))]
    S_z: List[Optional[float]] = [None] * len(recs)
    for i in range(len(recs)):
        S_z[i] = rolling_z(S_slope, i, RISK_CFG_FFL["stabTune"]["zWin"])
    # compute per record
    prev_mat: Optional[np.ndarray] = None
    for i, r in enumerate(recs):
        mat = r["matrix"]
        mm[i] = top_eigenvalue_ratio(mat)
        # directional flux using risk-safe pairs weighted by |coef|^p
        p = RISK_CFG_FFL["p"]
        num = 0.0
        den = 0.0
        abs_sum = 0.0
        for s_safe in SAFE_SYMBOLS:
            for s_risk in RISK_SYMBOLS:
                ia = idx_map[s_safe]
                ib = idx_map[s_risk]
                coef = float(mat[ia, ib]) if np.isfinite(mat[ia, ib]) else 0.0
                w = abs(coef) ** p
                zr = z[s_risk][window_offset + i] if window_offset + i < len(dates) else None
                zs = z[s_safe][window_offset + i] if window_offset + i < len(dates) else None
                if not (zr is None or zs is None):
                    diff = float(zr) - float(zs)
                    num += w * diff
                    den += w
                    abs_sum += w * abs(diff)
        jbar = (num / den) if den > 0 else 0.0
        jn = _tanh_clip(jbar, RISK_CFG_FFL["lambda"]) if den > 0 else 0.0
        jflux[i] = jn * _bear_damp(window_offset + i)
        fint[i] = (abs_sum / den) if den > 0 else 0.0
        # guard components
        safe_neg = float(r["sub"].get("safeNegative", 0.0))
        mm_pen = 0.0 if not np.isfinite(mm[i]) else min(1.0, max(0.0, (mm[i] - 0.85) / (0.97 - 0.85)))
        safe_pen = min(1.0, max(0.0, (safe_neg - 0.35) / (0.60 - 0.35)))
        delta_pen = 0.0
        if np.isfinite(r.get("delta", np.nan)):
            delta_pen = min(1.0, max(0.0, max(0.0, -float(r["delta"])) / (0.05 - 0.015)))
        flux_guard = None
        if i < len(full_flux) and full_flux[i] is not None:
            zf = rolling_z(full_flux, i, min(63, max(15, len(recs)//4)))
            flux_guard = 1 / (1 + math.exp(-0.85 * zf)) if zf is not None else None
        guard_val = 0.4 * mm_pen + 0.2 * safe_pen + 0.2 * delta_pen + 0.2 * (flux_guard if flux_guard is not None else 1.0)
        guard[i] = float(guard_val)
        # display score (blend)
        breadth = None
        risk_zs = []
        for sym in RISK_SYMBOLS:
            zi = z[sym][window_offset + i] if window_offset + i < len(dates) else None
            if zi is not None and np.isfinite(zi):
                risk_zs.append(zi)
        if risk_zs:
            breadth = sum(1 for v in risk_zs if v > 0) / len(risk_zs)
        combo = float(np.mean(risk_zs)) if risk_zs else 0.0
        flux_score = 0.5 * (1 + (jn if jn is not None else 0.0))
        combo_norm = 0.5 * (1 + combo)
        breadth_norm = breadth if breadth is not None else 0.0
        guard_relief = max(0.0, 1.0 - guard_val)
        score[i] = float(0.5 * flux_score + 0.2 * combo_norm + 0.2 * breadth_norm + 0.1 * guard_relief)
    # state machine (stab simplified)
    th = RISK_CFG_FFL["thresholds"]
    state: List[int] = []
    prev = 0
    for i in range(len(recs)):
        bready = True  # breadth gate simplified (computed in score)
        mmv = mm[i] if np.isfinite(mm[i]) else 0.0
        jn = jflux[i] if jflux[i] is not None else 0.0
        gv = guard[i] if guard[i] is not None else 1.0
        slope = S_slope[i]
        zsig = S_z[i]
        stab_up = (np.isfinite(slope) and np.isfinite(zsig) and slope > RISK_CFG_FFL["stabTune"]["slopeMin"] and zsig >= RISK_CFG_FFL["stabTune"]["zUp"])  # monthly uptrend
        stab_plunge = (np.isfinite(slope) and np.isfinite(zsig) and slope < -RISK_CFG_FFL["stabTune"]["slopeMin"] and zsig <= -RISK_CFG_FFL["stabTune"]["zDown"])  # monthly downshock
        # gates
        on_gate = (jn >= th["jOn"]) and (gv < 0.95) and (mmv < th["mmOff"]) and bready
        off_gate = (jn <= th["jOff"]) or (mmv >= th["mmOff"]) or (gv >= 1.0)
        # stab overrides
        if stab_plunge:
            state.append(-1)
        elif prev == 1 and stab_up and not off_gate:
            state.append(1)
        else:
            if on_gate:
                state.append(1)
            elif off_gate:
                state.append(-1)
            else:
                state.append(0)
        prev = state[-1]
    exec_state = [0] + state[:-1]
    return RegimeSeries(
        dates=[r["date"] for r in recs],
        score=[(s if s is not None else np.nan) for s in score],
        state=state,
        executed_state=exec_state,
        mm=mm,
        guard=guard,
        jflux=jflux,
        fint=fint,
    )


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


def leveraged_return(r: float, leverage: int = 3, weight: float = 1.0) -> float:
    if not np.isfinite(r):
        return 0.0
    eff_weight = float(weight) if np.isfinite(weight) else 1.0
    return max(-0.99, eff_weight * leverage * r)


def backtest_from_state(
    prices_close: List[float],
    dates: List[str],
    state: List[int],
    leverage: int = 3,
    delay_days: int = 1,
    *,
    price_mode: str = "close",
    prices_open: Optional[List[float]] = None,
    neutral_weight: float = 1.0 / 3.0,
) -> Dict[str, Any]:
    # Use arithmetic returns to match JS/UI backtest and CSV expectations
    mode = "open" if price_mode == "open" else "close"
    series = prices_close
    if mode == "open" and isinstance(prices_open, list) and len(prices_open) >= 2:
        series = [
            float(prices_open[idx])
            if idx < len(prices_open) and np.isfinite(prices_open[idx])
            else float(prices_close[idx]) if idx < len(prices_close) and np.isfinite(prices_close[idx])
            else np.nan
            for idx in range(len(prices_close))
        ]
    rets: List[float] = [0.0]
    for i in range(1, len(series)):
        a, b = series[i - 1], series[i]
        if np.isfinite(a) and np.isfinite(b) and a != 0:
            rets.append(b / a - 1.0)
        else:
            rets.append(0.0)
    if len(rets) < len(state) + 1:
        rets.extend([0.0] * (len(state) + 1 - len(rets)))
    # Align: dates length == len(series). Strategy rets length equals len(dates)
    # We map each regime day to return at same index (already aligned to window slicing externally)
    strat_eq = []
    bh_eq = []
    base_returns: List[float] = []
    strat_returns: List[float] = []
    s = 1.0
    b = 1.0
    # Build executed state with delay
    exec_state = [0] * len(state)
    for i in range(len(state)):
        j = i - delay_days
        exec_state[i] = state[j] if j >= 0 else 0
    for i in range(len(state)):
        # Map regime day i to corresponding benchmark return r[i]
        idx_ret = i + 1
        r = rets[idx_ret] if idx_ret < len(rets) else 0.0
        base_returns.append(r)
        if exec_state[i] > 0:
            step_ret = leveraged_return(r, leverage, 1.0)
        elif exec_state[i] < 0:
            step_ret = 0.0
        else:
            step_ret = leveraged_return(r, leverage, neutral_weight)
        strat_returns.append(step_ret)
        s *= 1.0 + step_ret
        b *= 1.0 + r
        strat_eq.append(s)
        bh_eq.append(b)
    return {
        "dates": dates,
        "equity_strategy": strat_eq,
        "equity_bh": bh_eq,
        "cum_strategy": strat_eq[-1] if strat_eq else 1.0,
        "cum_bh": bh_eq[-1] if bh_eq else 1.0,
        "price_mode": mode,
        "base_returns": base_returns,
        "strategy_returns": strat_returns,
        "executed_state": exec_state,
    }


def compute_realtime_regime(window: int = 30, use_realtime: bool = True, years: int = 5) -> Dict[str, Any]:
    """Fetch prices (FMP) and compute Classic, FFL-STAB, and FLL-Fusion with optional realtime patch.

    Returns a dict suitable for UI rendering.
    """
    if window not in WINDOWS:
        window = 30
    from_date = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=365 * years + 10)
    from_str = from_date.strftime("%Y-%m-%d")
    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set. Please add it to .env")
    # 1) Daily history (adj close + adj open)
    hist_frames = fetch_daily_history_fmp(ALL_SYMBOLS, from_str, api_key)
    close_map: Dict[str, pd.Series] = {}
    open_map: Dict[str, pd.Series] = {}
    idx_union: Optional[pd.Index] = None
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
        idx_union = close_series.index if idx_union is None else idx_union.union(close_series.index)
    if idx_union is None or len(idx_union) < window + 5:
        raise RuntimeError("Insufficient historical data from FMP")
    idx_union = idx_union.sort_values()
    # 2) Forward-fill gaps and apply realtime patch (close prices only)
    aligned_close: Dict[str, pd.Series] = {}
    aligned_open: Dict[str, pd.Series] = {}
    for sym in ALL_SYMBOLS:
        base_close = close_map.get(sym, pd.Series(dtype=float)).reindex(idx_union).ffill()
        if base_close.isna().all():
            raise RuntimeError(f"Insufficient data for {sym}")
        aligned_close[sym] = base_close
        base_open = open_map.get(sym, pd.Series(dtype=float)).reindex(idx_union)
        base_open = base_open.ffill().fillna(base_close)
        aligned_open[sym] = base_open
    if use_realtime:
        try:
            quotes = fetch_realtime_quotes_fmp(ALL_SYMBOLS, api_key)
            aligned_close = patch_with_realtime_last_price(aligned_close, quotes)
        except Exception:
            # best-effort; keep daily
            pass
    # 3) Align by intersection like JS (drop weekends introduced by BTC series)
    inter_dates, inter_series_close = _align_by_intersection(aligned_close, ALL_SYMBOLS)
    if not inter_dates or len(inter_dates) < window + 5:
        raise RuntimeError("Insufficient intersected history after alignment")
    idx_inter = pd.Index(inter_dates)
    inter_series_open: Dict[str, pd.Series] = {
        sym: aligned_open[sym].reindex(idx_inter).ffill().fillna(inter_series_close[sym])
        for sym in ALL_SYMBOLS
    }
    dates = [d.strftime("%Y-%m-%d") for d in inter_dates]
    prices = {s: inter_series_close[s].astype(float).tolist() for s in ALL_SYMBOLS}
    prices_open = {s: inter_series_open[s].astype(float).tolist() for s in ALL_SYMBOLS}
    # Compute window metrics for SIGNAL set
    date_idx = inter_dates
    metrics = compute_window_metrics(prices, date_idx, window)
    # Classic
    classic = compute_classic(metrics)
    # FFL port (stab variant default for UI parity)
    ffl = compute_ffl(prices, date_idx, metrics, window, variant="stab")
    # FLL-Fusion
    all_dates = [d.strftime("%Y-%m-%d") for d in date_idx]
    fusion = compute_fusion(classic, ffl, prices["QQQ"], window, all_dates)
    # Backtests (provide both 1d and 2d delays)
    # JS parity: segment should start at (window-1) to yield R+1 prices for R records
    start_idx = max(0, window - 1)
    qqq_segment = prices["QQQ"][start_idx:]
    qqq_open_segment = prices_open.get("QQQ", [])[start_idx:] if "QQQ" in prices_open else []
    bt0_close = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=0,
        price_mode="close",
    )
    bt1_close = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=1,
        price_mode="close",
    )
    bt2_close = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=2,
        price_mode="close",
    )
    bt1_open = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=1,
        price_mode="open",
        prices_open=qqq_open_segment,
    )
    bt2_open = backtest_from_state(
        qqq_segment,
        ffl["dates"],
        ffl["state"],
        delay_days=2,
        price_mode="open",
        prices_open=qqq_open_segment,
    )
    return {
        "window": window,
        "dates": ffl["dates"],
        "classic": {"score": classic[1], "state": classic[2]},
        "ffl_stab": ffl,
        "fusion": fusion,
        "series": {"QQQ": qqq_segment},
        "series_open": {"QQQ": qqq_open_segment},
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
