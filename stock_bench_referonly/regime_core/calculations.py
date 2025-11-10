"""Core calculation helpers shared by realtime regime engines."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SIGNAL_SYMBOLS: List[str] = []
RISK_SYMBOLS: List[str] = []
SAFE_SYMBOLS: List[str] = []
BASE_SYMBOL: str = "QQQ"
STAB_DI_GUARD_ENABLED: bool = True
STAB_DI_SYMBOL: str = "QQQ"
STAB_DI_PERIOD: int = 14
STAB_DI_MIN_PLUS: float = 20.0
STAB_DI_MIN_GAP: float = 3.0
RISK_CFG_FFL: Dict[str, Any] = {}
NEUTRAL_BENCH_WEIGHT: float = 0.33
RISK_ON_BENCH_WEIGHT: float = 1.0


def configure_calculations(
    *,
    signal_symbols: List[str],
    risk_symbols: List[str],
    safe_symbols: List[str],
    base_symbol: str,
    risk_cfg_ffl: Dict[str, Any],
    stab_di_guard_enabled: bool,
    stab_di_symbol: str,
    stab_di_period: int,
    stab_di_min_plus: float,
    stab_di_min_gap: float,
    neutral_bench_weight: float,
    risk_on_bench_weight: float,
) -> None:
    """Configure symbol universe and guard knobs for shared calculations."""
    global SIGNAL_SYMBOLS, RISK_SYMBOLS, SAFE_SYMBOLS
    global BASE_SYMBOL, RISK_CFG_FFL
    global STAB_DI_GUARD_ENABLED, STAB_DI_SYMBOL, STAB_DI_PERIOD
    global STAB_DI_MIN_PLUS, STAB_DI_MIN_GAP
    global NEUTRAL_BENCH_WEIGHT, RISK_ON_BENCH_WEIGHT

    SIGNAL_SYMBOLS = list(signal_symbols)
    RISK_SYMBOLS = list(risk_symbols)
    SAFE_SYMBOLS = list(safe_symbols)
    BASE_SYMBOL = base_symbol
    RISK_CFG_FFL = copy.deepcopy(risk_cfg_ffl)
    STAB_DI_GUARD_ENABLED = bool(stab_di_guard_enabled)
    STAB_DI_SYMBOL = stab_di_symbol
    STAB_DI_PERIOD = int(stab_di_period)
    STAB_DI_MIN_PLUS = float(stab_di_min_plus)
    STAB_DI_MIN_GAP = float(stab_di_min_gap)
    NEUTRAL_BENCH_WEIGHT = float(neutral_bench_weight)
    RISK_ON_BENCH_WEIGHT = float(risk_on_bench_weight)


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


def corr_safe(x: List[float], y: List[float]) -> float:
    """Correlation that returns 0.0 when undefined instead of NaN/warnings.

    - If stdev is zero on either side, treat correlation as 0.0 (no co-move info)
    - If lengths mismatch or too short, return NaN to signal unusable window
    """
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    ax = np.array(x, dtype=float)
    ay = np.array(y, dtype=float)
    sx = float(np.std(ax, ddof=1))
    sy = float(np.std(ay, ddof=1))
    if sx <= 0 or sy <= 0:
        return 0.0
    cx = ax - float(np.mean(ax))
    cy = ay - float(np.mean(ay))
    num = float(np.dot(cx, cy))
    den = (len(ax) - 1) * sx * sy
    if den == 0:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def rolling_corr_matrix(ret_mat: np.ndarray) -> np.ndarray:
    # ret_mat: shape (n_assets, window)
    if ret_mat.shape[1] < 2:
        return np.eye(ret_mat.shape[0])
    a = np.array(ret_mat, dtype=float)
    a[~np.isfinite(a)] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(a)
    if not np.isfinite(corr).all():
        corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


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
    # use module-level corr_safe

    for end in range(window - 1, ret_len):
        start = end - window + 1
        mat = np.zeros((len(SIGNAL_SYMBOLS), len(SIGNAL_SYMBOLS)))
        for a in range(len(SIGNAL_SYMBOLS)):
            for b in range(len(SIGNAL_SYMBOLS)):
                ra = ret[SIGNAL_SYMBOLS[a]][start:end+1]
                rb = ret[SIGNAL_SYMBOLS[b]][start:end+1]
                mat[a, b] = corr_safe(ra, rb)
        stab = stability_index(mat)
        stability_vals.append(stab)
        # JS sets record.date to the price date at the end index of the window.
        # Use dates[end] to match analysis_data/app.js computeWindowMetrics and precomputed.json.
        records.append({
            "date": dates[end].strftime("%Y-%m-%d"),
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
    risk_candidates = [BASE_SYMBOL, "IWM", "BTC-USD"]
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

    # --- Optional DI+/DI- close-based proxy (bullish hold guard)
    # Controlled by code-level settings (see file top: STAB_DI_*)
    def _compute_di_proxy_close(px: List[float], period: int = 14) -> Tuple[List[float], List[float]]:
        n = len(px)
        dm_plus_raw = [0.0] * n
        dm_minus_raw = [0.0] * n
        tr_raw = [0.0] * n
        for i in range(1, n):
            diff = float(px[i]) - float(px[i - 1]) if np.isfinite(px[i]) and np.isfinite(px[i - 1]) else 0.0
            up = max(0.0, diff)
            dn = max(0.0, -diff)
            dm_plus_raw[i] = up
            dm_minus_raw[i] = dn
            tr_raw[i] = abs(diff)
        # Smooth with EMA to avoid requiring Wilder's recursive seed
        dm_plus = ema(dm_plus_raw, max(2, period))
        dm_minus = ema(dm_minus_raw, max(2, period))
        tr = ema(tr_raw, max(2, period))
        di_p = [0.0] * n
        di_m = [0.0] * n
        for i in range(n):
            t = tr[i]
            if not np.isfinite(t) or t <= 0:
                di_p[i] = 0.0
                di_m[i] = 0.0
            else:
                di_p[i] = 100.0 * float(dm_plus[i]) / float(t)
                di_m[i] = 100.0 * float(dm_minus[i]) / float(t)
        return di_p, di_m

    di_guard_enabled = STAB_DI_GUARD_ENABLED
    di_period = STAB_DI_PERIOD
    di_min_plus = STAB_DI_MIN_PLUS
    di_min_gap = STAB_DI_MIN_GAP
    di_symbol = (STAB_DI_SYMBOL or BASE_SYMBOL).upper()
    di_plus_all: Optional[List[float]] = None
    di_minus_all: Optional[List[float]] = None
    if di_guard_enabled and di_symbol in prices:
        try:
            di_p, di_m = _compute_di_proxy_close(prices[di_symbol], period=di_period)
            di_plus_all, di_minus_all = di_p, di_m
        except Exception:
            di_plus_all, di_minus_all = None, None
    # Stability slope z for STAB/EXP rules
    Svals = [float(r.get("stability", np.nan)) for r in recs]
    s_fast = ema(Svals, RISK_CFG_FFL["stabTune"]["fast"])
    s_slow = ema(Svals, RISK_CFG_FFL["stabTune"]["slow"])
    s_slope = [((s_fast[i] - s_slow[i]) if np.isfinite(s_fast[i]) and np.isfinite(s_slow[i]) else np.nan) for i in range(n)]
    s_z: List[Optional[float]] = [None] * n
    for i in range(n):
        s_z[i] = rolling_z(s_slope, i, RISK_CFG_FFL["stabTune"]["zWin"])
    # Uptrend lead window (persistence): when stability slope-z shows sustained up-shock,
    # keep an up-lead window that resists brief Off signals.
    lag_up = int(max(1, RISK_CFG_FFL["stabTune"].get("lagUp", 3)))
    lead_on_win = int(max(1, RISK_CFG_FFL["stabTune"].get("leadOnWindow", 6)))
    up_lead = [0] * n
    up_seq = 0
    lead_left = 0
    for i in range(n):
        zi = s_z[i]
        sl = s_slope[i]
        up_cond = (zi is not None and np.isfinite(zi) and sl is not None and np.isfinite(sl)
                   and zi >= RISK_CFG_FFL["stabTune"]["zUp"] and sl >= RISK_CFG_FFL["stabTune"]["slopeMin"])
        up_seq = (up_seq + 1) if up_cond else 0
        if up_seq >= lag_up:
            lead_left = max(lead_left, lead_on_win)
        if lead_left > 0:
            up_lead[i] = 1
            lead_left -= 1
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
                # Parity with JS: record i maps to price index (window_offset + i)
                # Using +1 here would leak one-day-ahead information (lookahead bias).
                pi = window_offset + i
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
                pi = window_offset + i
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
                pi = window_offset + i
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
                    pi = window_offset + i
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
            pi = window_offset + i
            zi = z[s][pi] if pi < len(dates) else None
            if zi is not None and np.isfinite(zi):
                all_z.append(float(zi))
        for s in RISK_SYMBOLS:
            pi = window_offset + i
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
        b10 = bench_ret(BASE_SYMBOL, i, 10)
        b20 = bench_ret(BASE_SYMBOL, i, 20)
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
    up_confirm_off_min = int(max(1, RISK_CFG_FFL["stabTune"].get("upConfirmOffMin", 2)))
    up_off_harden = float(RISK_CFG_FFL["stabTune"].get("upOffHarden", 0.02) or 0.0)
    off_streak = 0
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
        # Harden Off threshold slightly during up-lead
        j_off_thr = th["jOff"] - (up_off_harden if up_lead[i] else 0.0)
        off_gate = (jn <= j_off_thr) or (mmv >= th["mmOff"]) or (gv >= 1.0)

        # Optional DI+ bullish hold: if currently Long and DI+ is clearly dominant,
        # ignore a soft Off (from j/guard) unless absorption is at hard-off level.
        di_bull = False
        if di_guard_enabled and di_plus_all is not None and di_minus_all is not None:
            px_idx = (window - 1) + i  # align rec i to price index
            if 0 <= px_idx < len(di_plus_all):
                dip = di_plus_all[px_idx]
                dim = di_minus_all[px_idx]
                if np.isfinite(dip) and np.isfinite(dim):
                    if (dip >= di_min_plus) and (dip >= dim + di_min_gap):
                        di_bull = True

        off_gate_final = off_gate
        if prev == 1 and di_bull and mmv < th["mmOff"]:
            # keep holding in uptrend unless hard absorption off triggers
            off_gate_final = False
        # Up-lead persistence: require consecutive Off during up-lead before flipping
        if prev == 1 and up_lead[i] and mmv < th["mmOff"] and not stab_plunge:
            if off_gate:
                off_streak += 1
            else:
                off_streak = 0
            if off_streak < up_confirm_off_min:
                off_gate_final = False
        else:
            off_streak = 0
        # stab overrides
        if stab_plunge:
            state.append(-1)
        elif prev == 1 and (stab_up or di_bull) and not off_gate_final:
            state.append(1)
        else:
            if on_gate:
                state.append(1)
            elif off_gate_final:
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


# ----------------------------------------------------------------------------
# Presets for gate tuning
# ----------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    # Current base (conservative default)
    "base": {
        "kappa": 2.50, "theta": 0.0,
        "ew_tilt": 0.80, "ew_dR": 0.80, "ew_chi": 0.60, "ew_FQI": 0.00, "ew_dT": -0.25, "G1": 0.66,
        "dr_tilt": 1.10, "dr_dR": 1.10, "dr_chi": 0.90, "dr_FQI": -0.25, "G2": 0.50,
        "cr_enabled": True, "cr_dR": 0.90, "cr_tilt": 0.80, "cr_eta": 0.80, "cr_FQI": 0.00, "cr_zR": 0.40, "cr_rsi_max": 50.0, "G3": 0.50, "cr_hold_days": 3,
    },
    # Minimize drawdown, stronger caps and holds
    "defensive": {
        "kappa": 2.20, "theta": 0.0,
        "ew_tilt": 0.80, "ew_dR": 0.80, "ew_chi": 0.60, "ew_FQI": 0.00, "ew_dT": -0.25, "G1": 0.66,
        "dr_tilt": 1.10, "dr_dR": 1.10, "dr_chi": 0.90, "dr_FQI": -0.25, "G2": 0.33,
        "cr_enabled": True, "cr_dR": 0.90, "cr_tilt": 0.80, "cr_eta": 0.80, "cr_FQI": 0.00, "cr_zR": 0.40, "cr_rsi_max": 50.0, "G3": 0.50, "cr_hold_days": 3,
        "shock_z": 2.0,
    },
    # Defensive+ : defensive 베이스 + STAB 기반 Off 오버레이만 추가
    # - defensive 본체는 그대로 유지
    # - Classic/FFL 엄격 오프 게이트 비활성화 (기존과 동일)
    # - STAB 하드 오프( jOff<=-0.08 또는 mmOff>=0.96 ) 시 노출을 추가 캡(stab_off_cap)으로 제한
    #   ⇒ 사용 의도: 약한 플립은 무시하고, 안정성 붕괴 시에만 확실히 방어
    "defensive_plus": {
        "kappa": 2.20, "theta": 0.0,
        "ew_tilt": 0.80, "ew_dR": 0.80, "ew_chi": 0.60, "ew_FQI": 0.00, "ew_dT": -0.25, "G1": 0.66,
        "dr_tilt": 1.10, "dr_dR": 1.10, "dr_chi": 0.90, "dr_FQI": -0.25, "G2": 0.33,
        "cr_enabled": True, "cr_dR": 0.90, "cr_tilt": 0.80, "cr_eta": 0.80, "cr_FQI": 0.00, "cr_zR": 0.40, "cr_rsi_max": 50.0, "G3": 0.50, "cr_hold_days": 3,
        "shock_z": 2.0,
        # 엄격 오프 게이트는 사용하지 않음
        "use_classic_off_gate": False,
        "use_ffl_off_gate": False,
        # STAB 하드오프 오버레이 사용 (FFL+STAB)
        "use_ffl_stab_off_gate": True,
        "stab_off_cap": 0.20,
        "use_ffl_stab_soft_cap": True,
        "stab_soft_cap": 0.40,
    },
    # Balance rally participation vs. cuts
    "balanced": {
        "kappa": 2.40, "theta": 0.0,
        "ew_tilt": 0.85, "ew_dR": 0.85, "ew_chi": 0.60, "ew_FQI": 0.00, "ew_dT": -0.25, "G1": 0.75,
        "dr_tilt": 1.10, "dr_dR": 1.10, "dr_chi": 0.90, "dr_FQI": -0.25, "G2": 0.50,
        "cr_enabled": True, "cr_dR": 0.90, "cr_tilt": 0.80, "cr_eta": 0.80, "cr_FQI": 0.00, "cr_zR": 0.40, "cr_rsi_max": 50.0, "G3": 0.66, "cr_hold_days": 2,
        "shock_z": 2.0,
    },
    # Balanced+ : balanced 베이스 + STAB 하드‑오프 오버레이(cap=0.20)
    "balanced_plus": {
        "kappa": 2.40, "theta": 0.0,
        "ew_tilt": 0.85, "ew_dR": 0.85, "ew_chi": 0.60, "ew_FQI": 0.00, "ew_dT": -0.25, "G1": 0.75,
        "dr_tilt": 1.10, "dr_dR": 1.10, "dr_chi": 0.90, "dr_FQI": -0.25, "G2": 0.50,
        "cr_enabled": True, "cr_dR": 0.90, "cr_tilt": 0.80, "cr_eta": 0.80, "cr_FQI": 0.00, "cr_zR": 0.40, "cr_rsi_max": 50.0, "G3": 0.66, "cr_hold_days": 2,
        "shock_z": 2.0,
        "use_classic_off_gate": False,
        "use_ffl_off_gate": False,
        "use_ffl_stab_off_gate": True,
        "stab_off_cap": 0.20,
        "use_ffl_stab_soft_cap": True,
        "stab_soft_cap": 0.40,
    },
    # Rally‑friendly: relax caps and triggers to preserve upside
    "aggressive": {
        # Upside-friendly: make caps rare in expansions, push wTA to extremes,
        # and ease +1 attainment while keeping crash guards.
        "kappa": 3.00, "theta": 0.0,
        # EW/DR triggers (raise thresholds so caps fire less often)
        "ew_tilt": 1.05, "ew_dR": 1.05, "ew_chi": 0.75, "ew_FQI": -0.15, "ew_dT": -0.35, "G1": 0.95,
        "dr_tilt": 1.25, "dr_dR": 1.25, "dr_chi": 1.10, "dr_FQI": -0.35, "G2": 0.90,
        # Cascade risk: keep but soften cap and shorten hold to recover upside faster
        "cr_enabled": True, "cr_dR": 1.10, "cr_tilt": 0.95, "cr_eta": 0.95, "cr_FQI": -0.10, "cr_zR": 0.60, "cr_rsi_max": 45.0, "G3": 0.80, "cr_hold_days": 1,
        # Shock cap less sensitive
        "shock_z": 2.4,
        # Mapping tweaks
        "wmin": 0.05, "wmax": 0.98,
        "neutral_band": (0.30, 0.60),
    },
    # Aggressive+: keep core logic, relax caps to preserve upside further.
    # Parameters only (A2 tuned); logic unchanged.
    "aggressive_plus": {
        # Upside-friendly (근본 로직 불변, 파라미터만 조정)
        "kappa": 3.00, "theta": 0.0,

        # EW/DR caps disabled (keep triggers for diagnostics; cap=1.00) g1 1.0 g2 1.0 에서 g2 0.8
        "ew_tilt": 1.05, "ew_dR": 1.05, "ew_chi": 0.75, "ew_FQI": -0.15, "ew_dT": -0.35, "G1": 1.00,
        "dr_tilt": 1.25, "dr_dR": 1.25, "dr_chi": 1.10, "dr_FQI": -0.35, "G2": 1.00,   

        # Cascade Risk (급락 선행) 유지(aggressive 동급)
        "cr_enabled": True,
        "cr_dR": 1.10, "cr_tilt": 0.95, "cr_eta": 0.95,
        "cr_FQI": -0.15, "cr_zR": 0.60, "cr_rsi_max": 45.0,
        "G3": 0.70, "cr_hold_days": 2,    #기존 0.80에 1일인데 우선 0.7 에 2일로 늘림

        # Shock sensitivity / mapping (aggressive 동급)
        "shock_z": 2.4,
        "wmin": 0.05, "wmax": 0.98,
        "neutral_band": (0.30, 0.60),

        # Flow 재진입 가속 (상승 초입 업캡처 개선)
        "on_thr": 0.20,
        "off_thr": -0.30,
    },
}



def compute_fusion(
    classic: Tuple[List[str], List[float], List[int], List[float], List[float]],
    ffl: Dict[str, Any],
    qqq_prices: List[float],
    window: int,
    all_dates: List[str],
    prices_all: Optional[Dict[str, List[float]]] = None,
    # Flow regime thresholds (zJ based)
    on_thr: float = 0.30,
    off_thr: float = -0.30,
    # Uncertainty score S weights
    a_chi: float = 0.40,
    b_eta: float = 0.20,
    c_R: float = 0.30,
    d_dR: float = 0.10,
    # TA weight mapping: wTA = σ(κ·(S−θ)), clipped to [wmin, wmax]
    kappa: float = 2.50,
    theta: float = 0.0,
    wmin: float = 0.10,
    wmax: float = 0.90,
    # Shock cap: z(|g|) > shock_z & g < 0 → pos ≤ 1/3
    shock_z: float = 2.0,
    # Exposure band for state mapping
    neutral_band: Tuple[float, float] = (0.33, 0.66),
    # Early‑Warning / De‑Risk gates (2/5, 2/4)
    ew_tilt: float = 0.80,  # z(tilt−) ≥ 0.80
    ew_dR: float = 0.80,    # z(ΔR)    ≥ 0.80
    ew_chi: float = 0.60,   # z(χ)     ≥ 0.60
    ew_FQI: float = 0.00,   # FQI      ≤ 0.00
    ew_dT: float = -0.25,   # Δ(TQI−FFQI) ≤ −0.25 (10d)
    G1: float = 0.66,       # EW cap
    dr_tilt: float = 1.10,
    dr_dR: float = 1.10,
    dr_chi: float = 0.90,
    dr_FQI: float = -0.25,
                   G2: float = 0.50,       # DR cap (conservative)
                   # --- Cascade Risk (CR) gate: front‑run fast drawdowns (optional) ---
                   cr_enabled: bool = True,
                   cr_dR: float = 0.90,    # z(ΔR) >= 0.90
                   cr_tilt: float = 0.80,  # z(tilt−) >= 0.80 (or eta >= 0.80)
                   cr_eta: float = 0.80,   # z(η) >= 0.80
                   cr_FQI: float = 0.00,   # FQI <= 0.00
                   cr_zR: float = 0.40,    # z(R) >= 0.40
                   cr_rsi_max: float = 50.0, # RSI <= 50 (trend weakening)
                   G3: float = 0.50,       # CR cap (medium)
                   cr_hold_days: int = 3,  # hold cap for N days once triggered
                   # --- Strict Off gate (optional, additive) ---
                   use_classic_off_gate: bool = False,
                   use_ffl_off_gate: bool = False,
                   off_cap: float = 0.20,
                   # --- STAB overlays (FFL+STAB) ---
                   use_ffl_stab_off_gate: bool = False,
                   stab_off_cap: float = 0.0,
                   use_ffl_stab_soft_cap: bool = False,
                   stab_soft_cap: float = 0.40,
                   ) -> Dict[str, Any]:
    """
    Compute TA+Fusion (NewMix) with Early‑Warning / De‑Risk gates.

    Ingredients
    - Flow: D⁺/D⁻ from flow‑gradient g, μ slope, J (mu−D weighted) and absorption ratio R
    - Uncertainty S: 0.4·z(χ) + 0.2·z(η) + 0.3·z(R) + 0.1·z(ΔR)
    - TA weight: wTA = σ(κ·(S−θ)) ∈ [wmin, wmax]
    - Exposure: pos = wTA·pos_TA + (1−wTA)·pos_Flow
    - Gates: EW(2/5) with cap G1, DR(2/4) with cap G2; shock cap for large down‑shock
    - No lookahead: decision at t, execution at t+1

    Returns
    - dates, state, executed_state, score (pos)
    - wTA, wFlow — TA/Flow exposure weights
    - diag: latest diagnostics (S, wTA, z‑scores, FQI/TQI/FFQI, tilt− z, TFI/FFI, gate status, advice)
    """
    import math
    # -------- helpers (local) --------
    def ar_ret(px: List[float]) -> List[float]:
        out=[0.0]
        for i in range(1,len(px)):
            a,b=px[i-1],px[i]; out.append((b/a-1.0) if (a and b) else 0.0)
        return out
    def rolling_mean(a, w):
        out=[np.nan]*len(a); s=0.0; q=0
        for i,x in enumerate(a):
            if np.isfinite(x): s+=x; q+=1
            if i>=w:
                y=a[i-w]; 
                if np.isfinite(y): s-=y; q-=1
            out[i]=(s/q) if q>0 else np.nan
        return out
    def rolling_std(a, w):
        m=rolling_mean(a,w); out=[np.nan]*len(a)
        for i in range(len(a)):
            if i<w-1 or not np.isfinite(m[i]): continue
            s2=0.0; c=0
            for j in range(i-w+1,i+1):
                x=a[j]
                if np.isfinite(x): d=x-m[i]; s2+=d*d; c+=1
            out[i]=math.sqrt(s2/c) if c>1 else np.nan
        return out
    def zscore_series(a, w=252, minp=63):
        m=rolling_mean(a,w); s=rolling_std(a,w); out=[]
        for i,x in enumerate(a):
            mi=m[i]; si=s[i]
            out.append(((x-mi)/si) if (i>=minp-1 and np.isfinite(mi) and np.isfinite(si) and si>0) else np.nan)
        return out
    def pct_rank_exp(a, w=252, minp=63):
        out=[np.nan]*len(a)
        for i in range(len(a)):
            lo=max(0,i-w+1); win=[v for v in a[lo:i+1] if np.isfinite(v)]
            if len(win)<minp: continue
            last=a[i]; ords=sorted(win); r=sum(v<=last for v in ords)-1
            out[i]=(r/(len(win)-1)) if len(win)>1 else np.nan
        return out
    def rsi_list(px, period=14):
        out=[np.nan]*len(px); prev=None; gains=[];losses=[]
        for i,x in enumerate(px):
            if prev is None: prev=x; continue
            d=x-prev; prev=x
            gains.append(max(0.0,d)); losses.append(max(0.0,-d))
            if len(gains)>period: gains.pop(0); losses.pop(0)
            if i<period: continue
            ag=sum(gains)/len(gains); al=sum(losses)/len(losses)
            if al<=1e-12: out[i]=100.0
            else: out[i]=100.0 - (100.0/(1.0+ag/al))
        return out
    def rolling_slope(a, win=63, minp=20):
        out=[np.nan]*len(a)
        for i in range(len(a)):
            lo=max(0,i-win+1); seg=[t for t in a[lo:i+1] if np.isfinite(t)]
            if len(seg)<max(minp,int(win*0.3)): continue
            m=len(seg); xs=np.arange(m); ys=np.array(seg,float)
            xm=xs.mean(); ym=ys.mean(); den=((xs-xm)**2).sum()
            if den<=0: continue
            out[i]=float(((xs-xm)*(ys-ym)).sum()/den)
        return out
    def logret_list(px):
        out=[np.nan]*len(px); prev=None
        for i,x in enumerate(px):
            if prev is None or not (x and prev): prev=x; continue
            out[i]=math.log(x)-math.log(prev); prev=x
        return out

    # -------- data --------
    if prices_all is None:
        raise ValueError("prices_all is required (QQQ,IWM,SPY,TLT,GLD,BTC-USD,TQQQ).")
    L=len(all_dates)
    def _crop(sym): 
        arr=prices_all.get(sym, []); return arr[:L]
    QQQ=_crop("QQQ"); IWM=_crop("IWM"); SPY=_crop("SPY"); TLT=_crop("TLT"); GLD=_crop("GLD"); BTC=_crop("BTC-USD")
    # --- Flow blocks ---
    rI=logret_list(IWM); rS=logret_list(SPY); rT=logret_list(TLT); rG=logret_list(GLD); rB=logret_list(BTC)
    zI=zscore_series(rI); zS=zscore_series(rS); zT=zscore_series(rT); zG=zscore_series(rG); zB=zscore_series(rB)
    flow=[ (np.nanmean([zI[i],zS[i],zB[i]]) - np.nanmean([zT[i],zG[i]])) if any(np.isfinite(v) for v in [zI[i],zS[i],zB[i],zT[i],zG[i]]) else np.nan for i in range(L) ]
    x=[]; acc=0.0
    for v in flow:
        if np.isfinite(v): acc+=v
        x.append(acc if np.isfinite(v) else np.nan)
    g=[0.0]; 
    for i in range(1,L):
        a=flow[i-1]; b=flow[i]
        g.append((b-a) if (np.isfinite(a) and np.isfinite(b)) else 0.0)
    g_pos=[max(0.0,v) for v in g]; g_neg=[max(0.0,-v) for v in g]
    def roll_mean(a,w): return rolling_mean(a,w)
    W=63
    D_plus =[math.sqrt(max(0.0,(roll_mean([u*u for u in g_pos],W)[i] or 0.0))) for i in range(L)]
    D_minus=[math.sqrt(max(0.0,(roll_mean([u*u for u in g_neg],W)[i] or 0.0))) for i in range(L)]
    mu=rolling_slope(x,W); mu_pos=[max(0.0, v if np.isfinite(v) else 0.0) for v in mu]; mu_neg=[max(0.0, -v if np.isfinite(v) else 0.0) for v in mu]
    J_plus =[ (mu_pos[i] - D_plus[i]*g_pos[i]) for i in range(L) ]
    J_minus=[ (mu_neg[i] - D_minus[i]*g_neg[i]) for i in range(L) ]
    J=[ (J_plus[i]-J_minus[i]) for i in range(L) ]
    # Absorption ratio R
    def absorption_ratio(win_returns):
        mat=np.array(win_returns,float)
        if not np.isfinite(mat).all(): return np.nan
        # Safe corr: build covariance-like via corr_safe
        n = mat.shape[0]
        C = np.ones((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    C[i, j] = 1.0
                else:
                    C[i, j] = corr_safe(list(mat[i, :]), list(mat[j, :]))
        vals=np.linalg.eigvalsh(C); vals.sort(); top=vals[-1]; tot=vals.sum()
        return float(top/tot) if tot!=0 else np.nan
    R=[np.nan]*L
    for i in range(L):
        if i < W-1: continue
        win=[
            [rI[j] for j in range(i-W+1,i+1)],
            [rS[j] for j in range(i-W+1,i+1)],
            [rT[j] for j in range(i-W+1,i+1)],
            [rG[j] for j in range(i-W+1,i+1)],
            [rB[j] for j in range(i-W+1,i+1)],
        ]
        R[i]=absorption_ratio(win)
    dR=[ abs(R[i]-R[i-1]) if i>0 and np.isfinite(R[i]) and np.isfinite(R[i-1]) else np.nan for i in range(L) ]

    # Uncertainty and TA weight
    eps=1e-8
    chi=[ (D_plus[i]+D_minus[i]) / ((mu_pos[i]+mu_neg[i])+eps) for i in range(L) ]
    eta=[ (abs(D_plus[i]-D_minus[i])) / ((D_plus[i]+D_minus[i])+eps) for i in range(L) ]
    z_chi=zscore_series(chi); z_eta=zscore_series(eta); z_R=zscore_series(R); z_dR=zscore_series(dR)
    S=[ a_chi*(z_chi[i] if np.isfinite(z_chi[i]) else 0.0) + b_eta*(z_eta[i] if np.isfinite(z_eta[i]) else 0.0) + c_R*(z_R[i] if np.isfinite(z_R[i]) else 0.0) + d_dR*(z_dR[i] if np.isfinite(z_dR[i]) else 0.0) for i in range(L) ]
    wTA=[]
    for i in range(L):
        s=S[i]
        if not np.isfinite(s): wTA.append(0.5)
        else:
            w=1.0/(1.0+math.exp(-kappa*(s-theta))); wTA.append(min(wmax,max(wmin,w)))

    # TA regime on QQQ
    SMA50=rolling_mean(QQQ,50); SMA200=rolling_mean(QQQ,200)
    def slope20(a): return rolling_slope(a,20,10)
    slope50=slope20(SMA50)
    trend_up=[1.0 if (i<len(QQQ) and np.isfinite(QQQ[i]) and np.isfinite(SMA200[i]) and QQQ[i]>SMA200[i]) else 0.0 for i in range(L)]
    rsi14=rsi_list(QQQ,14); mom_s=[ min(1.0,max(0.0,((rsi14[i]-30.0)/40.0))) if i<len(rsi14) and np.isfinite(rsi14[i]) else np.nan for i in range(L) ]
    rv20 = rolling_std(logret_list(QQQ),20)
    vol_s = pct_rank_exp(rv20,252,63)
    pr_slope = pct_rank_exp(slope50,252,63)
    # Robust TA score: average only finite components; default to 0.5 when none.
    TA_score=[]
    for i in range(L):
        cands = []
        ps = pr_slope[i] if i < len(pr_slope) else np.nan
        vs = vol_s[i] if i < len(vol_s) else np.nan
        comp1 = (0.6 * trend_up[i] + 0.4 * (ps if np.isfinite(ps) else 0.0)) if np.isfinite(trend_up[i]) else np.nan
        if np.isfinite(comp1):
            cands.append(comp1)
        if i < len(mom_s) and np.isfinite(mom_s[i]):
            cands.append(float(mom_s[i]))
        comp3 = (1.0 - vs) if np.isfinite(vs) else np.nan
        if np.isfinite(comp3):
            cands.append(float(comp3))
        if cands:
            TA_score.append(float(np.mean(cands)))
        else:
            TA_score.append(0.5)
    # map TA score -> state
    TA_state=[]; st=0
    for i in range(L):
        v=TA_score[i]
        if np.isfinite(v):
            if v>=0.60: st=1
            elif v<=0.40: st=-1
        TA_state.append(st)
    pos_ta=[ (1.0 if s==1 else (1.0/3.0 if s==0 else 0.0)) for s in TA_state ]

    # FFL regime from z(J)
    zJ=zscore_series(J); FFL_state=[]; st=0
    for i in range(L):
        v=zJ[i]
        if np.isfinite(v):
            if v>=on_thr: st=1
            elif v<=off_thr: st=-1
        FFL_state.append(st)
    pos_ffl=[ (1.0 if s==1 else (1.0/3.0 if s==0 else 0.0)) for s in FFL_state ]

    # Base mix
    pos_mix=[ min(1.0,max(0.0, wTA[i]*pos_ta[i] + (1.0-wTA[i])*pos_ffl[i])) for i in range(L) ]

    # --- Gates ---
    # Shock cap
    abs_g=[abs(v) for v in g]; z_abs_g=zscore_series(abs_g)
    shock=[ (1.0/3.0 if ( (z_abs_g[i] is not None and np.isfinite(z_abs_g[i]) and z_abs_g[i]>shock_z) and (g[i]<0) ) else 1.0) for i in range(L) ]
    # FQI, TQI, FFQI (need returns; we compute only for gating label; exposure shift is handled externally in backtest)
    # Here, approximate with QQQ returns as proxy; in live trading you'd compute with STRATEGY_SYMBOL series aligned.
    # For gating decision stability, proxy suffices.
    def pnl_series(exp):
        # t-1 decision vs t return using QQQ as proxy
        r=ar_ret(QQQ); out=[0.0]*L
        for i in range(1,L):
            out[i]=(exp[i-1] or 0.0) * r[i]
        return out
    # TQI/FFQI as rolling IR (mean/std) over 63
    def rolling_ir_list(pnl, win=63):
        out=[np.nan]*L
        for i in range(L):
            lo=max(0,i-win+1); seg=[t for t in pnl[lo:i+1] if np.isfinite(t)]
            if len(seg)<30: continue
            m=float(np.mean(seg)); s=float(np.std(seg))
            out[i]=(m/s) if s>1e-12 else np.nan
        return out
    pnl_ta=pnl_series(pos_ta); pnl_ff=pnl_series(pos_ffl)
    TQI=rolling_ir_list(pnl_ta,63); FFQI=rolling_ir_list(pnl_ff,63)
    # FQI: corr(J_{t-1}, r_t) with QQQ proxy
    rQ=ar_ret(QQQ)
    FQI=[np.nan]*L
    for i in range(L):
        lo=max(0,i-62); js=[J[j-1] for j in range(lo,i+1) if j-1>=0 and np.isfinite(J[j-1]) and np.isfinite(rQ[j])]
        rs=[rQ[j]      for j in range(lo,i+1) if j-1>=0 and np.isfinite(J[j-1]) and np.isfinite(rQ[j])]
        if len(js)>=30:
            # use safe correlation to avoid NaN and runtime warnings on zero-variance windows
            FQI[i]=float(corr_safe(js,rs))
    # tilt-
    tilt_minus=[ (D_minus[i]-D_plus[i]) / ((D_minus[i]+D_plus[i])+1e-8) for i in range(L) ]
    z_tilt=zscore_series(tilt_minus)
    # dT
    dT=[]
    for i in range(L):
        prev=i-10
        cur=( (TQI[i] if np.isfinite(TQI[i]) else 0.0) - (FFQI[i] if np.isfinite(FFQI[i]) else 0.0) )
        prv=( (TQI[prev] if (prev>=0 and np.isfinite(TQI[prev])) else 0.0) - (FFQI[prev] if (prev>=0 and np.isfinite(FFQI[prev])) else 0.0) )
        dT.append(cur-prv)

    # Build gates
    gate_ew=[]
    gate_dr=[]
    for i in range(L):
        hits_ew=0
        if (z_tilt[i] is not None and np.isfinite(z_tilt[i]) and z_tilt[i] >= ew_tilt): hits_ew+=1
        if (z_dR[i]   is not None and np.isfinite(z_dR[i])   and z_dR[i]   >= ew_dR): hits_ew+=1
        if (z_chi[i]  is not None and np.isfinite(z_chi[i])  and z_chi[i]  >= ew_chi): hits_ew+=1
        if (FQI[i]    is not None and np.isfinite(FQI[i])    and FQI[i]    <= ew_FQI): hits_ew+=1
        if (dT[i]     is not None and np.isfinite(dT[i])     and dT[i]     <= ew_dT ): hits_ew+=1
        gate_ew.append(G1 if hits_ew>=2 else 1.0)

        hits_dr=0
        if (z_tilt[i] is not None and np.isfinite(z_tilt[i]) and z_tilt[i] >= dr_tilt): hits_dr+=1
        if (z_dR[i]   is not None and np.isfinite(z_dR[i])   and z_dR[i]   >= dr_dR): hits_dr+=1
        if (z_chi[i]  is not None and np.isfinite(z_chi[i])  and z_chi[i]  >= dr_chi): hits_dr+=1
        if (FQI[i]    is not None and np.isfinite(FQI[i]) and FQI[i] <= dr_FQI and ( (TQI[i] if np.isfinite(TQI[i]) else 0.0) - (FFQI[i] if np.isfinite(FFQI[i]) else 0.0) ) <= 0.0 ): hits_dr+=1
        gate_dr.append(G2 if hits_dr>=2 else 1.0)

    # Cascade Risk (CR) — front‑run fast drawdowns by combining coupling and directional stress
    gate_cr=[1.0]*L
    if cr_enabled:
        streak=0
        for i in range(L):
            # build hit count
            hc=0
            if z_dR[i] is not None and np.isfinite(z_dR[i]) and z_dR[i] >= cr_dR: hc += 1
            # either tilt− or eta can represent directional dispersion asymmetry
            if (z_tilt[i] is not None and np.isfinite(z_tilt[i]) and z_tilt[i] >= cr_tilt) or \
               (z_eta[i] is not None and np.isfinite(z_eta[i]) and z_eta[i] >= cr_eta):
                hc += 1
            if FQI[i] is not None and np.isfinite(FQI[i]) and FQI[i] <= cr_FQI: hc += 1
            if z_R[i] is not None and np.isfinite(z_R[i]) and z_R[i] >= cr_zR: hc += 1
            # RSI condition (trend softening)
            rsi = rsi14[i] if i < len(rsi14) else None
            if isinstance(rsi, (int,float)) and np.isfinite(rsi) and rsi <= cr_rsi_max: hc += 1
            # trigger if 3+ conditions
            if hc >= 3:
                streak = max(streak, cr_hold_days)
            # recovery: if core drivers revert, allow decay
            if (z_chi[i] is not None and np.isfinite(z_chi[i]) and z_chi[i] < 0.0) and \
               (z_dR[i]  is not None and np.isfinite(z_dR[i])  and z_dR[i]  < 0.0) and \
               (z_tilt[i] is not None and np.isfinite(z_tilt[i]) and z_tilt[i] < 0.0):
                streak = min(streak, 1)
            # apply cap while active
            if streak > 0:
                gate_cr[i] = min(gate_cr[i], G3)
                streak -= 1

    # Combine
    pos_gate=[]
    lo,hi = neutral_band
    # Strict-Off gate from Classic/FFL (optional):
    # If Classic or FFL indicates Off at t, cap exposure additionally (off_cap).
    # Use provided 'classic' tuple for Classic state when available.
    classic_state: List[int] = []
    try:
        classic_state = (classic[2] if isinstance(classic, (list, tuple)) and len(classic) >= 3 else [])  # type: ignore
    except Exception:
        classic_state = []

    # Pre-extract FFL-STAB series for optional STAB overlay
    ffl_mm: List[Optional[float]] = []
    ffl_flux: List[Optional[float]] = []
    try:
        ffl_mm = ffl.get("mm", []) if isinstance(ffl, dict) else []
        ffl_flux = ffl.get("fflFlux", []) if isinstance(ffl, dict) else []
    except Exception:
        ffl_mm, ffl_flux = [], []

    th_ffl = RISK_CFG_FFL.get("thresholds", {})
    j_off_hard = float(th_ffl.get("jOff", -0.08))
    mm_off_hard = float(th_ffl.get("mmOff", 0.96))
    mm_fragile = float(th_ffl.get("mmFragile", 0.88))

    for i in range(L):
        g_cap = min(shock[i], gate_ew[i], gate_dr[i], gate_cr[i])
        # Optional strict-off caps from Classic/FFL states
        if off_cap < 1.0 and (use_classic_off_gate or use_ffl_off_gate):
            extra = 1.0
            if use_classic_off_gate and i < len(classic_state) and classic_state[i] == -1:
                extra = min(extra, max(0.0, float(off_cap)))
            if use_ffl_off_gate and i < len(FFL_state) and FFL_state[i] == -1:
                extra = min(extra, max(0.0, float(off_cap)))
            g_cap = min(g_cap, extra)
        # Optional STAB hard-off overlay (uses FFL-STAB signals)
        if use_ffl_stab_off_gate and stab_off_cap <= 1.0:
            rec_idx = i - (window - 1)
            if 0 <= rec_idx < max(len(ffl_mm), len(ffl_flux)):
                mmv = ffl_mm[rec_idx] if rec_idx < len(ffl_mm) else None
                jv = ffl_flux[rec_idx] if rec_idx < len(ffl_flux) else None
                hard_off = False
                if isinstance(mmv, (int, float)) and np.isfinite(mmv) and mmv >= mm_off_hard:
                    hard_off = True
                if not hard_off and isinstance(jv, (int, float)) and np.isfinite(jv) and jv <= j_off_hard:
                    hard_off = True
                if hard_off:
                    g_cap = min(g_cap, max(0.0, float(stab_off_cap)))
        # Optional STAB soft overlay: fragile absorption or high guard → cap to soft level
        if use_ffl_stab_soft_cap and stab_soft_cap < 1.0:
            rec_idx = i - (window - 1)
            if 0 <= rec_idx < max(len(ffl_mm), len(ffl_flux)):
                mmv = ffl_mm[rec_idx] if rec_idx < len(ffl_mm) else None
                guard_all = ffl.get("guard", []) if isinstance(ffl, dict) else []
                gv = None
                try:
                    gv = guard_all[rec_idx] if 0 <= rec_idx < len(guard_all) else None
                except Exception:
                    gv = None
                soft_on = False
                if isinstance(mmv, (int, float)) and np.isfinite(mmv) and (mmv >= mm_fragile) and (not (mmv >= mm_off_hard)):
                    soft_on = True
                if not soft_on and isinstance(gv, (int, float)) and np.isfinite(gv) and gv >= 0.95:
                    soft_on = True
                if soft_on:
                    g_cap = min(g_cap, max(0.0, float(stab_soft_cap)))
        pos_gate.append(min(1.0, max(0.0, pos_mix[i] * g_cap)))

    # Map to state {-1,0,1}
    state=[]
    for v in pos_gate:
        if v >= hi: state.append(1)
        elif v <= (lo*0.5): state.append(-1)
        else: state.append(0)

    # Executed state is t-1 decision
    exec_state=[0]+state[:-1]
    score=[ min(1.0, max(0.0, v)) for v in pos_gate ]
    # Align Fusion dates to the full input calendar to guarantee that
    # intraday-patched days (e.g., today) are represented in `fusion.dates`.
    # Classic dates may lag one day when window/min-period filters apply, so
    # relying on them can drop the realtime day from the label. Keep `all_dates`.
    dates = all_dates[:L]
    w_c=[ min(1.0, max(0.0, wTA[i])) for i in range(min(L,len(dates))) ]
    w_f=[ 1.0 - w_c[i] for i in range(len(w_c)) ]

    # --- Diagnostics snapshot (last day) ---
    last = min(len(score), L) - 1 if L > 0 and len(score) > 0 else -1
    diag: Dict[str, Any] = {}
    if last >= 0:
        # Helper to coerce to float or None
        def fval(arr, idx=last):
            try:
                v = arr[idx]
                return float(v) if (v is not None and np.isfinite(v)) else None
            except Exception:
                return None
        # TFI / FFI
        zS = zscore_series(S)
        qdiff = [
            ((TQI[i] if (i < len(TQI) and np.isfinite(TQI[i])) else 0.0) -
             (FFQI[i] if (i < len(FFQI) and np.isfinite(FFQI[i])) else 0.0))
            for i in range(L)
        ]
        zQ = zscore_series(qdiff)
        zFQI = zscore_series(FQI)
        tfi = None
        if fval(zS) is not None and fval(zQ) is not None and fval(zFQI) is not None:
            tfi = float(fval(zS) + 0.5 * fval(zQ) - 0.5 * fval(zFQI))
        ffi = (-tfi) if (tfi is not None) else None
        # FDD (drift dominance)
        fdd = None
        if chi[last] is not None and np.isfinite(chi[last]) and chi[last] != 0:
            fdd = float(1.0 / chi[last])
        # Gate hits/cap labels + detailed gate diagnostics (EW/DR/CR/Shock/STAB)
        cap_ew = float(gate_ew[last]) if last < len(gate_ew) else 1.0
        cap_dr = float(gate_dr[last]) if last < len(gate_dr) else 1.0
        cap_shock = float(shock[last]) if last < len(shock) else 1.0
        cap_cr = float(gate_cr[last]) if last < len(gate_cr) else 1.0
        cap = min(cap_shock, cap_ew, cap_dr, cap_cr)
        if cap < 1.0:
            if cap == cap_shock:
                cap_label = "ShockCap"
            elif cap == cap_dr:
                cap_label = f"DR(G2={G2:.2f})"
            elif cap == cap_cr:
                cap_label = f"CR(G3={G3:.2f})"
            else:
                cap_label = f"EW(G1={G1:.2f})"
        else:
            cap_label = "None"
        # --- EW details (2/5) ---
        ew_items = []
        # thresholds provided via function args
        ew_hits = 0
        _ew_metrics = {
            "tilt_z": (fval(z_tilt), ew_tilt, ">="),
            "dR_z": (fval(z_dR), ew_dR, ">="),
            "chi_z": (fval(z_chi), ew_chi, ">="),
            "FQI": (fval(FQI), ew_FQI, "<="),
            "dT": (fval(dT), ew_dT, "<="),
        }
        for k,(v,thr,op) in _ew_metrics.items():
            hit = None
            if v is not None:
                hit = (v >= thr) if op==">=" else (v <= thr)
                ew_hits += int(bool(hit))
            ew_items.append({"name":k,"value":v,"thr":thr,"op":op,"hit":hit})
        # --- DR details (2/4) ---
        dr_items = []
        dr_hits = 0
        _dr_metrics = {
            "tilt_z": (fval(z_tilt), dr_tilt, ">="),
            "dR_z": (fval(z_dR), dr_dR, ">="),
            "chi_z": (fval(z_chi), dr_chi, ">="),
            "FFQI": (fval(FFQI), dr_FQI, "<="),
        }
        for k,(v,thr,op) in _dr_metrics.items():
            hit = None
            if v is not None:
                hit = (v >= thr) if op==">=" else (v <= thr)
                dr_hits += int(bool(hit))
            dr_items.append({"name":k,"value":v,"thr":thr,"op":op,"hit":hit})
        # --- CR details (>=3 triggers) ---
        cr_items = []
        cr_hits = 0
        v_dR = fval(z_dR); v_tilt = fval(z_tilt); v_eta = fval(z_eta)
        v_FQI = fval(FQI); v_zR = fval(z_R)
        v_RSI = None
        try:
            v_RSI = fval(rsi14)
        except Exception:
            v_RSI = None
        # Each condition
        conds = [
            {"name":"dR_z","value":v_dR,"thr":cr_dR,"op":">="},
            {"name":"tilt_z_or_eta_z","value":max(v_tilt if v_tilt is not None else -1e9, v_eta if v_eta is not None else -1e9),"thr":max(cr_tilt, cr_eta),"op":">= (either)"},
            {"name":"FQI","value":v_FQI,"thr":cr_FQI,"op":"<="},
            {"name":"R_z","value":v_zR,"thr":cr_zR,"op":">="},
            {"name":"RSI","value":v_RSI,"thr":cr_rsi_max,"op":"<="},
        ]
        for c in conds:
            v,thr,op = c["value"], c["thr"], c["op"]
            hit = None
            if v is not None:
                if op.startswith(">="):
                    hit = v >= thr
                else:
                    hit = v <= thr
            c["hit"] = hit
            if hit:
                cr_hits += 1
            cr_items.append(c)
        # --- Shock details ---
        z_abs = fval(z_abs_g)
        g_last = g[last] if last < len(g) else None
        shock_active = (z_abs is not None and g_last is not None and z_abs > shock_z and g_last < 0)
        # --- STAB overlay details ---
        stab_overlay = None
        stab_overlay_cap = None
        stab_soft_overlay = None
        stab_soft_cap_out = None
        try:
            if use_ffl_stab_off_gate:
                rec_idx = last - (window - 1)
                mmv = fval(ffl.get("mm", []), idx=rec_idx) if isinstance(ffl, dict) else None
                jv  = fval(ffl.get("fflFlux", []), idx=rec_idx) if isinstance(ffl, dict) else None
                j_off_hard = float(RISK_CFG_FFL.get("thresholds",{}).get("jOff", -0.08))
                mm_off_hard = float(RISK_CFG_FFL.get("thresholds",{}).get("mmOff", 0.96))
                stab_overlay = ( (mmv is not None and mmv >= mm_off_hard) or (jv is not None and jv <= j_off_hard) )
                stab_overlay_cap = float(stab_off_cap)
            if use_ffl_stab_soft_cap:
                rec_idx = last - (window - 1)
                mmv = fval(ffl.get("mm", []), idx=rec_idx) if isinstance(ffl, dict) else None
                gv  = fval(ffl.get("guard", []), idx=rec_idx) if isinstance(ffl, dict) else None
                mm_fragile = float(RISK_CFG_FFL.get("thresholds",{}).get("mmFragile", 0.88))
                mm_off_hard = float(RISK_CFG_FFL.get("thresholds",{}).get("mmOff", 0.96))
                soft_on = False
                if mmv is not None and mmv >= mm_fragile and not (mmv >= mm_off_hard):
                    soft_on = True
                if gv is not None and gv >= 0.95:
                    soft_on = True
                stab_soft_overlay = soft_on
                stab_soft_cap_out = float(stab_soft_cap)
        except Exception:
            stab_overlay, stab_overlay_cap = None, None
            stab_soft_overlay, stab_soft_cap_out = None, None
        # 63d signal concordance (TA vs Flow)
        conc = None
        try:
            wlen = 63
            i0 = max(0, last - wlen + 1)
            agree = 0
            tot = 0
            for i in range(i0, last + 1):
                a = TA_state[i] if i < len(TA_state) else 0
                b = FFL_state[i] if i < len(FFL_state) else 0
                if a == 0 and b == 0:
                    continue
                if (a > 0 and b > 0) or (a < 0 and b < 0) or (a == 0 and b == 0):
                    agree += 1
                tot += 1
            conc = float(agree / tot) if tot > 0 else None
        except Exception:
            conc = None
        # Classification
        regime_label = "혼합"
        wt = fval(wTA)
        if tfi is not None and wt is not None:
            if (tfi >= 0.5) and (wt >= 0.60):
                regime_label = "TA-우위"
            elif (-tfi >= 0.5) and (wt <= 0.40):
                regime_label = "Flow-우위"
            elif (-tfi >= 0.2) and (0.40 < wt < 0.60):
                regime_label = "Flow-우위혼합"
        # Advice (concise, Korean)
        advice_lines = []
        advice_lines.append("오늘 운용 의견 (TQQQ 기준)")
        advice_lines.append(f"- 현재 판단: {regime_label}")
        advice_lines.append("- 운용 프레임: pos = wTA·pos_TA + (1−wTA)·pos_Flow; 급락 쇼크 시 pos≤1/3 캡")
        advice_lines.append("- 엔트리/추세는 TA 기준 우선(SMA200 상회·RSI>50 ⇒ 노출 하한 1/3 유지)")
        advice_lines.append("- Flow(zJ)·FDD 회복 시 비중 상향; 변동성 확산(η↑/ΔR↑) 시 전일 고점 리테스트 후 분할 증액")
        advice_lines.append(
            f"- 게이트: EW {ew_hits}/5 (cap={G1:.2f}), DR {dr_hits}/4 (cap={G2:.2f}), CR {cr_hits}/5 (cap={G3:.2f}), Shock={'on' if shock_active else 'off'}"
        )
        if use_ffl_stab_off_gate:
            advice_lines.append(
                f"- STAB 오버레이: {'on' if (stab_overlay is True and stab_overlay_cap is not None) else 'off'}"
                + (f" (cap={stab_overlay_cap:.2f})" if stab_overlay_cap is not None else "")
            )
        # Populate diagnostics
        diag = {
            "wTA": wt,
            "S": fval(S),
            "z_chi": fval(z_chi),
            "z_eta": fval(z_eta),
            "z_R": fval(z_R),
            "z_dR": fval(z_dR),
            "tilt_z": fval(z_tilt),
            "FQI": fval(FQI),
            "TQI": fval(TQI),
            "FFQI": fval(FFQI),
            "dT": fval(dT),
            "zJ": fval(zJ),
            "FDD": fdd,
            "TFI": (float(tfi) if tfi is not None else None),
            "FFI": (float(ffi) if ffi is not None else None),
            "gate_cap": cap,
            "gate_label": cap_label,
            "ew_cap": cap_ew,
            "dr_cap": cap_dr,
            "shock_cap": cap_shock,
            "cr_cap": cap_cr,
            "EW": {"count": ew_hits, "total": 5, "cap": cap_ew, "items": ew_items},
            "DR": {"count": dr_hits, "total": 4, "cap": cap_dr, "items": dr_items},
            "CR": {"count": cr_hits, "need": 3, "total": 5, "cap": cap_cr, "items": cr_items},
            "Shock": {"active": shock_active, "z_abs_g": z_abs, "thr": shock_z, "cap": cap_shock},
            "STAB": {
                "overlay": stab_overlay,
                "cap": stab_overlay_cap,
                "enabled": bool(use_ffl_stab_off_gate),
                "soft_overlay": stab_soft_overlay,
                "soft_cap": stab_soft_cap_out,
                "soft_enabled": bool(use_ffl_stab_soft_cap),
            },
            "CONC": conc,
            "regime_label": regime_label,
            "advice": "\n".join(advice_lines),
        }

    w_c_trim = w_c[:len(score)]
    w_f_trim = w_f[:len(score)]

    return {
        "dates": dates[:len(score)],
        "state": state[:len(score)],
        "executed_state": exec_state[:len(score)],
        "score": score[:len(score)],
        "wTA": w_c_trim,
        "wFlow": w_f_trim,
        "mode": "FFL+TA",
        "diag": diag,
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
