"""Market Probabilistic Report (single-file module).

Generates a deep, professional market report with:
- Probability of upward direction P(Up|x) for horizon H
- Feature contributions (log-likelihood ratios)
- Breadth, yield curve, sectors snapshot (when FMP is available)
- Calendar highlights (when FMP is available)

Design goals:
- Additive, drop-in module with a single API: generate_market_report(...)
- No changes required elsewhere; webapp can import and render markdown later
- FMP network calls are optional; graceful degradation with SoT payload fallback

Usage (CLI):
    python3 market_analysis/market_report.py --print --horizon 5

Environment:
- FMP_API_KEY: Financial Modeling Prep API key (optional)
- REGIME_BASE_SYMBOL or REGIME_BENCH_SYMBOL to override base benchmark (default QQQ)

Artifacts:
- ml_cache/market_prob_nb.json  (Naive Bayes + Platt scaling params)
"""

from __future__ import annotations

import os
import io
import json
import re
import math
import time
import random
import logging
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "numpy/pandas가 필요합니다. `pip install numpy pandas` 후 다시 실행하세요."
    ) from exc

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception as exc:  # pragma: no cover
    raise RuntimeError("requests/urllib3가 필요합니다. `pip install requests` 후 다시 실행하세요.") from exc


# Optional: use insights.resolve_effective_index if available for SoT payloads
try:  # pragma: no cover - optional import
    from .insights import resolve_effective_index  # type: ignore
except Exception:  # pragma: no cover
    resolve_effective_index = None  # fallback later


LOG = logging.getLogger("market_report")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler("logs/market_prob_report.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    LOG.addHandler(sh)


HISTORY_FILE = os.path.join("ml_cache", "market_prob_history.jsonl")


def _history_path() -> str:
    os.makedirs("ml_cache", exist_ok=True)
    return HISTORY_FILE


def _append_history(entry: Dict[str, Any]) -> None:
    try:
        path = _history_path()
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        LOG.warning(f"히스토리 기록 실패: {exc}")


def _extract_bench_snapshot(payload: Optional[Dict[str, Any]], base_symbol: str) -> Tuple[Optional[str], Optional[float]]:
    if not payload or not isinstance(payload, dict):
        return None, None
    dates = payload.get("dates") or []
    if not isinstance(dates, list) or not dates:
        return None, None
    dates_str = [str(d) for d in dates]
    series_map = payload.get("series") or {}
    bench_series = None
    if isinstance(series_map, dict):
        bench_series = series_map.get(base_symbol) or series_map.get(base_symbol.upper()) or series_map.get("QQQ")
    if not isinstance(bench_series, list) or not bench_series:
        bench_series = payload.get("series_bench") or payload.get("series_bench_close")
    if not isinstance(bench_series, list) or not bench_series:
        return dates_str[-1], None
    idx = min(len(bench_series) - 1, len(dates_str) - 1)
    try:
        return dates_str[-1], float(bench_series[idx])
    except Exception:
        return dates_str[-1], None


def _record_report_history(
    report: Dict[str, Any],
    *,
    base_symbol: str,
    horizon_days: int,
    sot_payload: Optional[Dict[str, Any]],
) -> None:
    if not sot_payload:
        return
    asof_date, bench_close = _extract_bench_snapshot(sot_payload, base_symbol)
    if not asof_date:
        return
    prob = ((report.get("prob") or {}).get("p_up"))
    prob_raw = ((report.get("prob") or {}).get("p_up_raw"))
    entry = {
        "ts_utc": dt.datetime.utcnow().isoformat() + "Z",
        "asof_date": asof_date,
        "horizon_days": int(horizon_days),
        "base_symbol": base_symbol,
        "prob": float(prob) if prob is not None else None,
        "prob_raw": float(prob_raw) if prob_raw is not None else None,
        "bench_close": bench_close,
        "drivers": report.get("drivers"),
        "features": report.get("features"),
        "refs": {
            "sources": (report.get("refs") or {}).get("ctx_fmp", {}).get("sources"),
            "quotes_count": (report.get("refs") or {}).get("ctx_fmp", {}).get("quotes_count"),
        },
    }
    manifest = sot_payload.get("manifest") or {}
    if manifest:
        entry["manifest"] = {
            "origin": manifest.get("origin"),
            "mode": manifest.get("mode"),
            "preset": manifest.get("preset"),
        }
    try:
        _append_history(entry)
    except Exception:
        pass


# ------------------------------------------------------------
# Small, local FMP client (graceful if no key/network)
# ------------------------------------------------------------

class FMPClient:
    BASES = [
        "https://financialmodelingprep.com/stable",
        "https://financialmodelingprep.com/api/v3",
        "https://financialmodelingprep.com/api/v4",
    ]
    RETRYABLE_EXC = (
        requests.exceptions.SSLError,
        requests.exceptions.ConnectionError,
        requests.exceptions.ReadTimeout,
    )

    def __init__(self, api_key: Optional[str] = None, pause: float = 0.8, timeout: int = 20):
        self.api_key = api_key or os.getenv("FMP_API_KEY", "").strip()
        self.pause = pause
        self.timeout = timeout
        session = requests.Session()
        session.trust_env = False
        session.headers.update(
            {
                "User-Agent": "curl/8.6.0",
                "Accept": "application/json",
                "Connection": "keep-alive",
            }
        )
        retry = Retry(
            total=6,
            connect=6,
            read=6,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "OPTIONS"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        self._session = session

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        p = dict(params or {})
        if self.api_key:
            p.setdefault("apikey", self.api_key)
        last_exc: Optional[Exception] = None
        for attempt in range(6):
            try:
                resp = self._session.get(url, params=p, timeout=(10, self.timeout))
                resp.raise_for_status()
                data = resp.json()
                time.sleep(self.pause + random.uniform(0.0, 0.4))
                return data
            except self.RETRYABLE_EXC as exc:
                last_exc = exc
                if attempt >= 5:
                    raise
                backoff = 0.6 * (attempt + 1)
                time.sleep(backoff)
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("FMP request failed without exception")

    def _try(self, paths: List[str], params: Optional[Dict[str, Any]] = None) -> Any:
        last_err: Optional[Exception] = None
        for base in self.BASES:
            for path in paths:
                try:
                    return self._get(f"{base}/{path.lstrip('/')}", params)
                except Exception as e:  # pragma: no cover - network path
                    last_err = e
                    continue
        if last_err:
            raise last_err
        raise RuntimeError("FMP endpoints unreachable")

    def _try_list(self, paths: List[str], params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Try multiple paths; return first non-empty list payload.

        Some FMP endpoints return an object or empty array under certain plans; this helper ensures
        we only accept a non-empty list, otherwise continue to next candidate.
        """
        last_err: Optional[Exception] = None
        for base in self.BASES:
            for path in paths:
                try:
                    data = self._get(f"{base}/{path.lstrip('/')}", params)
                    if isinstance(data, list) and data:
                        return data
                except Exception as e:
                    last_err = e
                    continue
        if last_err:
            raise last_err
        return []

    # --- Selected endpoints used for the report ---
    def sectors_performance(self) -> Any:
        return self._try(["sector-performance"])  # v3

    def etf_holdings(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch ETF constituents via documented v3 endpoint with API key enforced."""
        sym = (symbol or "").strip().upper()
        if not sym:
            return []

        if not self.api_key:
            LOG.info("FMP_API_KEY 미설정 → ETF holdings 생략(%s)", sym)
            return []

        url = f"https://financialmodelingprep.com/api/v3/etf-holder/{sym}"
        try:
            data = self._get(url)
            if isinstance(data, list):
                return data
        except Exception as exc:
            LOG.info("ETF holdings(v3) 실패(%s): %s", sym, exc)
        return []

    def batch_quote(self, symbols: List[str]) -> Any:
        """Fetch quotes for many symbols with graceful fallback.

        Some plans return HTTP 200 with an empty list for batch-quote; using _try_list
        lets us continue to the next endpoint until we receive real data.
        """
        if not symbols:
            return []
        syms = ",".join(symbols)
        return self._try_list([f"batch-quote?symbols={syms}", f"quote/{syms}"])  # v3

    def treasury_rates(self, frm: Optional[str] = None, to: Optional[str] = None) -> Any:
        p: Dict[str, Any] = {}
        if frm:
            p["from"] = frm
        if to:
            p["to"] = to
        return self._try(["treasury-rates"], p)  # v4

    def economics_calendar(self, frm: str, to: str) -> Any:
        return self._try([f"economics-calendar?from={frm}&to={to}"])  # v3


# ------------------------------------------------------------
# Feature engineering helpers (breadth, momentum, curve)
# ------------------------------------------------------------

def _iqr_scale(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
    iqr = max(1e-9, float(q3 - q1))
    return (x - float(q1)) / iqr


def _winsorize(x: np.ndarray, p: float = 0.01) -> np.ndarray:
    lo, hi = np.nanpercentile(x, 100 * p), np.nanpercentile(x, 100 * (1 - p))
    return np.clip(x, lo, hi)


def adv_decline_ratio(quotes_df: pd.DataFrame) -> float:
    # Prefer percentage if present; values may include "%" suffix
    cp = quotes_df.get("changesPercentage")
    if cp is not None and len(cp) > 0:
        try:
            cp = pd.to_numeric(pd.Series(cp).astype(str).str.rstrip("%"), errors="coerce")
        except Exception:
            cp = pd.Series(dtype=float)
    else:
        cp = pd.Series(dtype=float)
    if cp.notna().any():
        pos = (cp > 0).sum()
        neg = (cp < 0).sum()
    else:
        ch = quotes_df.get("change")
        if ch is not None and len(ch) > 0:
            ch = pd.to_numeric(pd.Series(ch), errors="coerce")
        else:
            ch = pd.Series(dtype=float)
        if ch.notna().any():
            pos = (ch > 0).sum()
            neg = (ch < 0).sum()
        else:
            pos = 0
            neg = 0
    if pos == 0 and neg == 0:
        # fallback using price vs prevClose
        if {"price", "previousClose"} <= set(quotes_df.columns):
            diff = quotes_df["price"] - quotes_df["previousClose"]
            pos = (diff > 0).sum()
            neg = (diff < 0).sum()
    return float(pos) / max(1.0, float(neg)) if (pos or neg) else float("nan")


def pct_above_ma(quotes_df: pd.DataFrame, field: str = "priceAvg50") -> float:
    if {"price", field} <= set(quotes_df.columns):
        try:
            return float((quotes_df["price"] > quotes_df[field]).mean())
        except Exception:
            return float("nan")
    return float("nan")


def nh_nl_ratio(quotes_df: pd.DataFrame) -> float:
    if {"price", "yearHigh", "yearLow"} <= set(quotes_df.columns):
        nh = (quotes_df["price"] >= quotes_df["yearHigh"] * 0.995).sum()
        nl = (quotes_df["price"] <= quotes_df["yearLow"] * 1.005).sum()
        return float(nh) / max(1.0, float(nl))
    return float("nan")


def realized_vol(prices: pd.Series, window: int = 20) -> float:
    r = np.log(pd.to_numeric(prices, errors="coerce")).diff().dropna()
    if len(r) < max(2, window // 2):
        return float("nan")
    return float(r.rolling(window).std().iloc[-1] * math.sqrt(252))


def yield_spreads(curve: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if curve is None or curve.empty:
        return out
    # Build tolerant alias map: handle '3M'|'MONTH3'|'THREEMONTH', '2Y'|'YEAR2', '10Y'|'YEAR10', '30Y'|'YEAR30'
    up = {str(c).upper(): c for c in curve.columns}
    def _pick(*aliases: str) -> Optional[str]:
        for a in aliases:
            key = a.upper()
            if key in up:
                return up[key]
        return None

    c3m = _pick("3M", "MONTH3", "THREEMONTH", "M3")
    c2y = _pick("2Y", "YEAR2", "Y2", "2YEAR")
    c10y = _pick("10Y", "YEAR10", "Y10", "10YEAR")
    c30y = _pick("30Y", "YEAR30", "Y30", "30YEAR")
    cols_needed = [c3m, c2y, c10y, c30y]
    if any(c is None for c in cols_needed):
        return out
    last = curve.iloc[-1]
    def gf(cname: str) -> float:
        return float(pd.to_numeric(last.get(cname), errors="coerce"))

    try:
        ten = gf(c10y)
        two = gf(c2y)
        trm = gf(c3m)
        thrty = gf(c30y)
    except Exception:
        return out
    out["SPR_10Y_3M"] = ten - trm
    out["SPR_10Y_2Y"] = ten - two
    out["CURVATURE"] = thrty + trm - 2.0 * ten
    return out


# ------------------------------------------------------------
# Simple Gaussian Naive Bayes with Platt scaling
# ------------------------------------------------------------

@dataclass
class NBParams:
    mu_up: np.ndarray
    var_up: np.ndarray
    mu_dn: np.ndarray
    var_dn: np.ndarray
    prior_up: float


def fit_gaussian_nb(X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> NBParams:
    mask_up = y == 1
    mask_dn = y == 0
    up, dn = X[mask_up], X[mask_dn]
    mu_up = np.nanmean(up, axis=0)
    var_up = np.nanvar(up, axis=0) + eps
    mu_dn = np.nanmean(dn, axis=0)
    var_dn = np.nanvar(dn, axis=0) + eps
    prior_up = float(mask_up.mean()) if len(y) else 0.5
    return NBParams(mu_up, var_up, mu_dn, var_dn, prior_up)


def _log_pdf_gaussian(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2 * np.pi * var) + ((x - mu) ** 2) / var)


def predict_proba_nb(params: NBParams, x: np.ndarray) -> Tuple[float, np.ndarray]:
    # ignore NaNs feature-wise
    finite = np.isfinite(x)
    if not finite.any():
        return 0.5, np.zeros_like(x)
    ll_up = _log_pdf_gaussian(x[finite], params.mu_up[finite], params.var_up[finite]).sum()
    ll_dn = _log_pdf_gaussian(x[finite], params.mu_dn[finite], params.var_dn[finite]).sum()
    ll_up += math.log(max(1e-9, params.prior_up))
    ll_dn += math.log(max(1e-9, 1.0 - params.prior_up))
    log_den = np.logaddexp(ll_up, ll_dn)
    p_up = float(np.exp(ll_up - log_den))
    contrib = np.zeros_like(x)
    # feature-wise contribution where finite
    lr = _log_pdf_gaussian(x[finite], params.mu_up[finite], params.var_up[finite]) - _log_pdf_gaussian(
        x[finite], params.mu_dn[finite], params.var_dn[finite]
    )
    contrib[finite] = lr
    return p_up, contrib


def _stabilize_nb_params(params: NBParams, var_floor: float = 1e-3) -> None:
    params.var_up = np.clip(params.var_up, var_floor, None)
    params.var_dn = np.clip(params.var_dn, var_floor, None)


def platt_scale(p: np.ndarray, y: np.ndarray, max_iter: int = 200, lr: float = 0.1) -> Tuple[float, float]:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p))
    A, B = 1.0, 0.0
    for _ in range(max_iter):
        s = 1.0 / (1.0 + np.exp(-(A * z + B)))
        gA = float(((s - y) * z).mean())
        gB = float((s - y).mean())
        A -= lr * gA
        B -= lr * gB
    return float(A), float(B)


def apply_platt(p: float, A: float, B: float) -> float:
    p = float(np.clip(p, 1e-5, 1 - 1e-5))
    z = math.log(p / (1 - p))
    return float(1.0 / (1.0 + math.exp(-(A * z + B))))


# ------------------------------------------------------------
# Model persistence
# ------------------------------------------------------------

def _model_path() -> str:
    os.makedirs("ml_cache", exist_ok=True)
    return os.path.join("ml_cache", "market_prob_nb.json")


def save_model(params: NBParams, calib: Tuple[float, float]) -> None:
    data = {
        "mu_up": params.mu_up.tolist(),
        "var_up": params.var_up.tolist(),
        "mu_dn": params.mu_dn.tolist(),
        "var_dn": params.var_dn.tolist(),
        "prior_up": params.prior_up,
        "calib": [float(calib[0]), float(calib[1])],
        "ts": dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(_model_path(), "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_model() -> Optional[Tuple[NBParams, Tuple[float, float]]]:
    path = _model_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        params = NBParams(
            mu_up=np.array(d["mu_up"], dtype=float),
            var_up=np.array(d["var_up"], dtype=float),
            mu_dn=np.array(d["mu_dn"], dtype=float),
            var_dn=np.array(d["var_dn"], dtype=float),
            prior_up=float(d.get("prior_up", 0.5)),
        )
        calib = tuple(d.get("calib", [1.0, 0.0]))  # type: ignore
        return params, (float(calib[0]), float(calib[1]))  # type: ignore
    except Exception as e:
        LOG.warning(f"모델 로드 실패: {e}")
        return None


# ------------------------------------------------------------
# Feature computation
# ------------------------------------------------------------

FEATURES = [
    "ADR",
    "Pct>MA50",
    "Pct>MA200",
    "NH/NL",
    "RV20",
    "RV60",
    "SPR_10Y_3M",
    "SPR_10Y_2Y",
    "CURVATURE",
]


ETF_BASKET = ["SPY", "VOO", "IVV", "QQQ"]

FEATURE_CLAMPS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "ADR": (0.05, 5.0),
    "NH/NL": (0.05, 8.0),
    "Pct>MA50": (0.0, 1.0),
    "Pct>MA200": (0.0, 1.0),
}


def _apply_feature_clamps(x: np.ndarray) -> np.ndarray:
    capped = x.copy()
    for idx, name in enumerate(FEATURES):
        bounds = FEATURE_CLAMPS.get(name)
        if not bounds:
            continue
        lo, hi = bounds
        val = capped[idx]
        if not np.isfinite(val):
            continue
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        capped[idx] = float(val)
    return capped


def _features_from_etf(fmp: FMPClient) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute breadth from ETF holdings (default: SPY/VOO/IVV)."""
    ctx: Dict[str, Any] = {"sources": [], "holdings": []}
    holdings: List[str] = []
    pattern = re.compile(r"^[A-Z0-9.\-]{1,10}$")
    for etf in ETF_BASKET:
        try:
            rows = fmp.etf_holdings(etf)
        except Exception as exc:
            LOG.info(f"ETF holdings 실패({etf}): {exc}")
        if not rows:
            continue
        ctx["sources"].append(f"FMP etf-holder/{etf}")
        for item in rows:
            ticker = None
            if isinstance(item, dict):
                for key in ("assetSymbol", "symbol", "asset", "ticker"):
                    if item.get(key):
                        ticker = str(item[key]).upper()
                        break
            elif isinstance(item, str):
                ticker = item.upper()
            if ticker and pattern.match(ticker) and ticker not in holdings:
                holdings.append(ticker)
    ctx["holdings"] = holdings

    quotes_df: Optional[pd.DataFrame] = None
    if holdings:
        rows: List[dict] = []
        chunk = 50
        for i in range(0, len(holdings), chunk):
            try:
                rows.extend(fmp.batch_quote(holdings[i : i + chunk]))
            except Exception as exc:
                LOG.info(f"batch_quote 실패(chunk {i}): {exc}")
        if rows:
            quotes_df = pd.DataFrame(rows)
            if not quotes_df.empty:
                for col in ("price", "previousClose", "change", "priceAvg50", "priceAvg200", "yearHigh", "yearLow"):
                    if col in quotes_df.columns:
                        quotes_df[col] = pd.to_numeric(quotes_df[col], errors="coerce")
                if "changesPercentage" in quotes_df.columns:
                    quotes_df["changesPercentage"] = pd.to_numeric(
                        quotes_df["changesPercentage"].astype(str).str.rstrip("%"),
                        errors="coerce",
                    )
            else:
                quotes_df = None
    else:
        LOG.warning("ETF 기반 브레드스: holdings 목록이 비었습니다.")

    # Compute breadth features from whatever quotes_df we obtained
    adr = adv_decline_ratio(quotes_df) if isinstance(quotes_df, pd.DataFrame) and not quotes_df.empty else float("nan")
    p50 = pct_above_ma(quotes_df, "priceAvg50") if isinstance(quotes_df, pd.DataFrame) and not quotes_df.empty else float("nan")
    p200 = pct_above_ma(quotes_df, "priceAvg200") if isinstance(quotes_df, pd.DataFrame) and not quotes_df.empty else float("nan")
    nhnl = nh_nl_ratio(quotes_df) if isinstance(quotes_df, pd.DataFrame) and not quotes_df.empty else float("nan")
    ctx["quotes_count"] = int(len(quotes_df)) if isinstance(quotes_df, pd.DataFrame) else 0

    # Yield curve via FMP
    to = dt.date.today().isoformat()
    frm = (dt.date.today() - dt.timedelta(days=45)).isoformat()
    curve_ts = {"dates": [], "spr_10y_3m": [], "spr_10y_2y": [], "curvature": []}
    spreads: Dict[str, float] = {}
    try:
        curve_raw = fmp.treasury_rates(frm, to)
        curve_df = pd.DataFrame(curve_raw)
        cols = {c: c.upper() for c in curve_df.columns}
        curve_df = curve_df.rename(columns=cols)
        date_col = next((c for c in curve_df.columns if str(c).upper().startswith("DATE")), None)
        if date_col:
            curve_df = curve_df.sort_values(date_col)
        spreads = yield_spreads(curve_df)
        # alias mapping for robustness (e.g., MONTH3/YEAR2/YEAR10/YEAR30)
        up = {str(c).upper(): c for c in curve_df.columns}
        def _pick(*aliases: str):
            for a in aliases:
                if a.upper() in up:
                    return up[a.upper()]
            return None
        c3m = _pick("3M", "MONTH3", "THREEMONTH", "M3")
        c2y = _pick("2Y", "YEAR2", "Y2", "2YEAR")
        c10y = _pick("10Y", "YEAR10", "Y10", "10YEAR")
        c30y = _pick("30Y", "YEAR30", "Y30", "30YEAR")
        if all([c3m, c2y, c10y, c30y]):
            for c in [c3m, c2y, c10y, c30y]:
                curve_df[c] = pd.to_numeric(curve_df[c], errors="coerce")
            dd = curve_df.dropna(subset=[c3m, c2y, c10y, c30y]).copy()
            curve_ts["dates"] = [str(x) for x in dd[date_col].tolist()] if (date_col in dd.columns) else [str(i) for i in range(len(dd))]
            curve_ts["spr_10y_3m"] = (dd[c10y] - dd[c3m]).astype(float).tolist()
            curve_ts["spr_10y_2y"] = (dd[c10y] - dd[c2y]).astype(float).tolist()
            curve_ts["curvature"] = (dd[c30y] + dd[c3m] - 2.0 * dd[c10y]).astype(float).tolist()
        ctx["sources"].append("FMP treasury-rates")
    except Exception:
        pass

    x = np.array(
        [
            adr,
            p50,
            p200,
            nhnl,
            float("nan"),  # RV20 via SoT later
            float("nan"),  # RV60 via SoT later
            spreads.get("SPR_10Y_3M", float("nan")),
            spreads.get("SPR_10Y_2Y", float("nan")),
            spreads.get("CURVATURE", float("nan")),
        ],
        dtype=float,
    )
    ctx["curve_ts"] = curve_ts
    # include sectors snapshot if available
    try:
        secs = fmp.sectors_performance()
        if isinstance(secs, list) and secs:
            ctx["sectors"] = secs
    except Exception:
        pass
    return x, ctx


def _features_from_sot(payload: Dict[str, Any], base_symbol: str = "QQQ") -> Tuple[np.ndarray, Dict[str, Any]]:
    ctx: Dict[str, Any] = {"sources": ["SoT payload"], "bench": base_symbol}
    dates = payload.get("dates", []) or []
    series_map = payload.get("series", {}) or {}
    bench = series_map.get(base_symbol) or series_map.get("QQQ") or []
    if not bench and isinstance(series_map, dict):
        # Try close/open fallback keys
        for k in ("series_bench", "series_bench_close"):
            v = payload.get(k)
            if isinstance(v, list) and v:
                bench = v
                break
    s = pd.Series(bench)
    rv20 = realized_vol(s, 20)
    rv60 = realized_vol(s, 60)
    # fallback features not available from SoT: use NaN
    x = np.array(
        [
            float("nan"),  # ADR
            float("nan"),  # %>MA50
            float("nan"),  # %>MA200
            float("nan"),  # NH/NL
            rv20,
            rv60,
            float("nan"),  # 10Y-3M
            float("nan"),  # 10Y-2Y
            float("nan"),  # curvature
        ],
        dtype=float,
    )
    ctx["len"] = len(s)
    ctx["dates"] = dates[-2:]
    return x, ctx


def _merge_feature_vectors(x_fmp: Optional[np.ndarray], x_sot: Optional[np.ndarray]) -> np.ndarray:
    if x_fmp is None and x_sot is None:
        return np.full(len(FEATURES), np.nan, dtype=float)
    if x_fmp is None:
        return x_sot  # type: ignore
    if x_sot is None:
        return x_fmp
    # Prefer FMP where available; fill NaNs from SoT
    out = x_fmp.copy()
    mask = ~np.isfinite(out)
    out[mask] = x_sot[mask]
    return out


def _features_from_precomputed(path: str = "precomputed.json") -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Offline fallback using local precomputed.json.

    Approximates breadth over the small asset set; not perfect but better than empty.
    """
    ctx: Dict[str, Any] = {"sources": ["precomputed"], "assets": []}
    if not os.path.exists(path):
        return None, ctx
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        assets = data.get("assets", []) or []
        series = data.get("priceSeries", {}) or {}
        ctx["assets"] = [a.get("symbol") for a in assets if isinstance(a, dict) and a.get("symbol")]
        # Build small-panel breadth
        panel: Dict[str, List[float]] = {}
        for sym in ctx["assets"]:
            arr = series.get(sym)
            if isinstance(arr, list) and len(arr) >= 200:
                panel[sym] = [float(x) for x in arr if isinstance(x, (int, float))]
        if not panel:
            return None, ctx
        import numpy as _np
        # compute 1d returns, MA50/200 flags, NH/NL proximity flags
        pos = 0
        neg = 0
        above50 = 0
        above200 = 0
        near_high = 0
        near_low = 0
        rv20_list = []
        rv60_list = []
        for sym, px in panel.items():
            s = _np.array(px, dtype=float)
            if len(s) < 200:
                continue
            r = _np.diff(_np.log(s))
            if len(r) >= 1:
                last_r = r[-1]
                pos += int(last_r > 0)
                neg += int(last_r < 0)
            ma50 = _np.mean(s[-50:])
            ma200 = _np.mean(s[-200:])
            last = s[-1]
            above50 += int(last > ma50)
            above200 += int(last > ma200)
            h52 = _np.max(s[-252:])
            l52 = _np.min(s[-252:])
            near_high += int(last >= 0.995 * h52)
            near_low += int(last <= 1.005 * l52)
            # realized vol per asset
            if len(r) >= 20:
                rv20_list.append(float(_np.std(r[-20:]) * math.sqrt(252)))
            if len(r) >= 60:
                rv60_list.append(float(_np.std(r[-60:]) * math.sqrt(252)))
        # Protect against zero division
        adr = float(pos) / max(1.0, float(neg)) if (pos or neg) else float("nan")
        total = max(1, len(panel))
        p50 = float(above50) / float(total)
        p200 = float(above200) / float(total)
        nhnl = float(near_high) / max(1.0, float(near_low)) if (near_high or near_low) else float("nan")
        rv20 = float(_np.nanmean(rv20_list)) if rv20_list else float("nan")
        rv60 = float(_np.nanmean(rv60_list)) if rv60_list else float("nan")
        x = _np.array(
            [
                adr,
                p50,
                p200,
                nhnl,
                rv20,
                rv60,
                float("nan"),
                float("nan"),
                float("nan"),
            ],
            dtype=float,
        )
        return x, ctx
    except Exception as e:
        LOG.info(f"precomputed fallback 실패: {e}")
        return None, ctx


# ------------------------------------------------------------
# Report builder
# ------------------------------------------------------------

def _pick_base_symbol(payload: Optional[Dict[str, Any]]) -> str:
    base = (os.getenv("REGIME_BASE_SYMBOL") or os.getenv("REGIME_BENCH_SYMBOL") or "SPY").upper()
    if payload and isinstance(payload.get("series"), dict):
        if base not in payload["series"] and "QQQ" in payload["series"]:
            base = "QQQ"
    return base


def _format_pct(x: Optional[float], digits: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "N/A"


def _format_num(x: Optional[float], digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def _driver_table(drivers: List[Tuple[str, float]]) -> str:
    lines = ["| 순위 | 드라이버 | 기여(LLR) |", "|---:|:--|--:|"]
    for i, (name, val) in enumerate(drivers, 1):
        lines.append(f"| {i} | {name} | {_format_num(val, 3)} |")
    return "\n".join(lines)


def _detect_divergences(p_up: float, features: Dict[str, float]) -> List[str]:
    notes: List[str] = []
    spr = features.get("SPR_10Y_3M")
    rv = features.get("RV20")
    adr = features.get("ADR")
    if p_up >= 0.6 and rv and rv > 0.35:
        notes.append("상승확률은 높은데 단기 변동성↑ → 추격 리스크")
    if p_up <= 0.4 and adr and adr > 1.8:
        notes.append("하락확률 우세지만 참가도↑ → 되돌림 가능성")
    if spr is not None and spr < 0 and p_up >= 0.55:
        notes.append("역전 수익률곡선 vs 상승확률 → 중기/단기 시그널 충돌")
    return notes


def _build_narrative(
    p_up: float, drivers: List[Tuple[str, float]], feats: Dict[str, Optional[float]], include_header: bool = True
) -> str:
    """Compose a concise, human-readable narrative and conclusion."""
    lines: List[str] = []
    # 1) Headline by probability
    if p_up >= 0.65:
        headline = "단기 상승 우세 (신뢰 높음)"
    elif p_up >= 0.55:
        headline = "약한 상승 우세 (혼조 요소 존재)"
    elif p_up > 0.45:
        headline = "혼조/박스 구간"
    else:
        headline = "단기 하락 우세"

    # 2) Primary drivers
    top = drivers[:3] if drivers else []
    drv_txt = ", ".join([f"{n}({val:+.3f})" for n, val in top]) if top else "(드라이버 부족)"

    # 3) Macro cues from spreads/vol
    spr2 = feats.get("SPR_10Y_2Y")
    spr3 = feats.get("SPR_10Y_3M")
    curv = feats.get("CURVATURE")
    rv20 = feats.get("RV20")
    cues: List[str] = []
    try:
        if isinstance(spr2, (int, float)):
            cues.append(f"10Y−2Y={spr2:+.2f}")
        if isinstance(spr3, (int, float)):
            cues.append(f"10Y−3M={spr3:+.2f}")
        if isinstance(curv, (int, float)):
            cues.append(f"Curve={curv:+.2f}")
        if isinstance(rv20, (int, float)):
            cues.append(f"RV20={rv20:.3f}")
    except Exception:
        pass
    cue_txt = ", ".join(cues) if cues else "거시/변동성 단서 부족"

    # 4) Breadth quick check
    adr = feats.get("ADR")
    ma50 = feats.get("Pct>MA50")
    ma200 = feats.get("Pct>MA200")
    nhnl = feats.get("NH/NL")
    if all(v is None for v in [adr, ma50, ma200, nhnl]):
        breadth = "브레드스 결측(구성/배치 응답 필요)"
    else:
        parts = []
        if isinstance(adr, (int, float)):
            parts.append(f"ADR={adr:.2f}")
        if isinstance(ma50, (int, float)):
            parts.append(f"%>MA50={ma50:.2%}")
        if isinstance(ma200, (int, float)):
            parts.append(f"%>MA200={ma200:.2%}")
        if isinstance(nhnl, (int, float)):
            parts.append(f"NH/NL={nhnl:.2f}")
        breadth = ", ".join(parts)

    # 5) Compose
    if include_header:
        lines.append("### 🧠 서술형 요약")
    lines.append(f"- 결론: **{headline}** · P(Up|H)={p_up:.1%}")
    lines.append(f"- 주된 근거(Top drivers): {drv_txt}")
    lines.append(f"- 거시/변동성 단서: {cue_txt}")
    lines.append(f"- 참가도(브레드스): {breadth}")
    lines.append("- 해석 가이드: 확률은 방향성, 브레드스는 랠리의 질, 스프레드는 거시·유동성 축을 의미. 세 축이 같은 방향이면 신뢰↑, 엇갈리면 분할·헷지 고려.")
    return "\n".join(lines)


def _build_driver_notes(
    drivers: List[Tuple[str, float]], feats: Dict[str, Optional[float]], p_up: float
) -> List[str]:
    notes: List[str] = []

    def _pct(val: Optional[float]) -> str:
        try:
            return f"{val:.1%}"
        except Exception:
            return "N/A"

    for name, llr in drivers:
        raw = feats.get(name)
        if raw is None:
            continue
        if name == "NH/NL":
            clamp_hi = FEATURE_CLAMPS.get("NH/NL", (None, None))[1]
            if clamp_hi is not None and raw >= clamp_hi:
                notes.append(
                    f"- NH/NL={raw:.2f}가 모델 상한({clamp_hi})을 넘어서 '과열 후 되돌림' 패턴으로 분류되어 로그우도비 {llr:+.3f}."
                )
            elif llr < 0:
                notes.append(
                    f"- NH/NL={raw:.2f}: 신고 대비 신저 비율이 둔화되어 되돌림 가능성을 키워 LLR {llr:+.3f}."
                )
            else:
                notes.append(f"- NH/NL={raw:.2f}: 신고 강세가 유지돼 상승 쪽으로 LLR {llr:+.3f}.")
            continue
        if name in {"RV20", "RV60"}:
            notes.append(
                f"- {name}={raw:.3f}: 실현 변동성이 {'높아져' if llr < 0 else '안정돼'} "
                f"{'하락' if llr < 0 else '상승'} 시나리오에 LLR {llr:+.3f}만큼 기여."
            )
            continue
        if name == "ADR":
            notes.append(
                f"- ADR={raw:.2f}: 상승 종목 비중이 높지만 상승/하락 클래스 평균 차이가 작아 LLR {llr:+.3f} (중립에 가까움)."
            )
            continue
        if name.startswith("Pct>MA"):
            notes.append(
                f"- {name}={_pct(raw)}: 장기/중기 추세 위 종목 비중 변동으로 LLR {llr:+.3f}."
            )
            continue
        notes.append(f"- {name}={raw:.3f}: LLR {llr:+.3f}.")

    adr = feats.get("ADR")
    if isinstance(adr, (int, float)):
        if p_up < 0.4 and adr > 1.5:
            notes.append(
                f"- 확률은 {p_up:.1%}로 낮지만 ADR={adr:.2f}>1.5라 참가도는 살아 있습니다. 하락 시나리오에도 급반등이 나올 수 있어 분할/헷지를 권장합니다."
            )
        elif p_up > 0.6 and adr < 1.0:
            notes.append(
                f"- 확률은 {p_up:.1%}로 높지만 ADR={adr:.2f}<1이라 상승 폭이 좁습니다. 추격 매수보다는 분할 접근이 유리합니다."
            )
    return notes


def generate_market_report(
    *,
    horizon_days: int = 5,
    lookback_days: int = 1260,
    use_cache: bool = True,
    auto_calibrate: bool = True,
    benchmark: Optional[str] = None,
    sot_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute market upward probability and build a detailed Markdown report.

    Parameters
    ----------
    horizon_days : int
        Forward horizon (business days) for direction label definition (for docs only here).
    lookback_days : int
        Historical lookback for fitting (when data/feature cache is available).
    use_cache : bool
        If True, try loading model params from ml_cache first.
    auto_calibrate : bool
        If True and we have calibration labels, apply Platt scaling (no-op otherwise).
    benchmark : Optional[str]
        Base symbol for realized vol from SoT. Defaults to env or QQQ.
    sot_payload : Optional[dict]
        Shared SoT payload to enrich features without FMP.
    """

    base_symbol = (benchmark or _pick_base_symbol(sot_payload)).upper()
    fmp = FMPClient()

    # 1) Build feature vector from FMP and/or SoT
    x_fmp: Optional[np.ndarray] = None
    ctx_fmp: Dict[str, Any] = {}
    try:
        x_fmp, ctx_fmp = _features_from_etf(fmp)
    except Exception as e:
        LOG.warning(f"피처 생성 실패(FMP ETF 경로): {e}")
        x_fmp = None

    x_sot: Optional[np.ndarray] = None
    ctx_sot: Dict[str, Any] = {}
    # Prefer SPY as bench when ETF(SPY/VOO/IVV) 기반 브레드스를 사용
    if sot_payload and isinstance(sot_payload, dict):
        try:
            series_map = sot_payload.get("series", {}) or {}
            if "SPY" in series_map:
                base_symbol = "SPY"
        except Exception:
            pass
    if sot_payload:
        try:
            x_sot, ctx_sot = _features_from_sot(sot_payload, base_symbol)
        except Exception as e:
            LOG.warning(f"SoT 피처 생성 실패: {e}")
            x_sot = None

    # merge order: prefer Holdings path (FMP/Yahoo) → SoT
    x = _merge_feature_vectors(x_fmp, x_sot)
    x = _apply_feature_clamps(x)
    x_display = x.copy()

    # Provide realized vol if FMP lacked it but SoT had it
    features_map = {k: float(v) if np.isfinite(v) else None for k, v in zip(FEATURES, x_display)}

    # 2) Load model (or cold-start defaults)
    cold_start = False
    loaded = load_model() if use_cache else None
    if loaded is not None:
        params, calib = loaded
        _stabilize_nb_params(params)
    else:
        cold_start = True
        zeros = np.zeros_like(x)
        base_var = np.ones_like(x) * 10.0
        params = NBParams(mu_up=zeros, var_up=base_var, mu_dn=zeros, var_dn=base_var, prior_up=0.5)
        calib = (1.0, 0.0)

    # 3) Inference
    p_raw, contrib = predict_proba_nb(params, np.nan_to_num(x, nan=0.0))
    if cold_start:
        p_up = 0.5
        contrib = np.zeros_like(contrib)
    else:
        p_up = apply_platt(p_raw, *calib) if auto_calibrate else float(p_raw)
    order = np.argsort(-np.abs(contrib))
    top = [(FEATURES[i], float(contrib[i])) for i in order[:5]]

    # 4) Optional sector snapshot & calendar (FMP)
    sectors_md = ""
    cal_md = ""
    if fmp.enabled:
        try:
            secs = fmp.sectors_performance()
            if isinstance(secs, dict) and "sectorPerformance" in secs:
                secs = secs["sectorPerformance"]
            if isinstance(secs, list) and secs:
                df = pd.DataFrame(secs)

                def _lc(name: str) -> str:
                    return str(name).strip().lower()

                df = df.rename(columns={c: _lc(c) for c in df.columns})
                sector_col = next((c for c in df.columns if _lc(c) in {"sector", "name"}), None)
                pct_col = next(
                    (c for c in df.columns if _lc(c) in {"changespercentage", "changepercent", "changes"}), None
                )
                if sector_col and pct_col:
                    df = df[[sector_col, pct_col]].rename(columns={sector_col: "sector", pct_col: "pct_raw"})
                    df["pct_clean"] = (
                        df["pct_raw"]
                        .astype(str)
                        .str.replace("%", "", regex=False)
                        .str.replace("+", "", regex=False)
                        .str.replace(",", "", regex=False)
                        .str.strip()
                    )
                    df["pct"] = pd.to_numeric(df["pct_clean"], errors="coerce")
                    df = df.dropna(subset=["pct"])
                    df["frac"] = df["pct"].astype(float) / 100.0
                    df = df.dropna(subset=["sector", "frac"]).sort_values("frac", ascending=False)
                    lines = ["| 섹터 | 변화율(1d) |", "|:--|--:|"]
                    for _, row in df.iterrows():
                        frac = row.get("frac")
                        sector = row.get("sector", "-")
                        if pd.isna(frac):
                            continue
                        lines.append(f"| {sector} | {_format_pct(float(frac), 2)} |")
                    sectors_md = "\n".join(lines)
        except Exception as e:
            LOG.info(f"섹터 스냅샷 실패: {e}")
        try:
            end = dt.date.today()
            start = end + dt.timedelta(days=1)
            cal = fmp.economics_calendar(start.isoformat(), (end + dt.timedelta(days=5)).isoformat())
            if isinstance(cal, list) and cal:
                cal = cal[:8]
                lines = ["| 날짜(ET) | 이벤트 | 실제/예상 | 영향 |", "|:--|:--|:--|:--|"]
                for it in cal:
                    date = it.get("date") or it.get("publishedDate") or ""
                    name = it.get("event") or it.get("name") or it.get("title") or "-"
                    actual = it.get("actual") or "-"
                    forecast = it.get("forecast") or it.get("consensus") or "-"
                    impact = it.get("impact") or it.get("importance") or "-"
                    lines.append(f"| {date} | {name} | {actual}/{forecast} | {impact} |")
                cal_md = "\n".join(lines)
        except Exception as e:
            LOG.info(f"캘린더 수집 실패: {e}")

    # 5) Divergences
    div_notes = _detect_divergences(p_up, features_map)

    # 6) Compose markdown with narrative
    today_et = dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M ET")
    lines: List[str] = []
    lines.append("## 📊 확률 기반 시장 리포트 (Fusion PDF)")
    lines.append("")
    lines.append(f"- 기준(ET): {today_et} · 벤치마크: {base_symbol}")
    lines.append(f"- 예측 지평: H={horizon_days} 영업일")
    lines.append(f"- 상승 확률 P(Up|x): {_format_pct(p_up,1)} (원시 {_format_pct(p_raw,1)})")
    lines.append("")
    if cold_start:
        lines.append("- ⚠️ 모델 파라미터 미적재: 중립 폴백(확률 50%)")
        lines.append("")
    narrative_text = _build_narrative(p_up, top, features_map, include_header=True)
    narrative_plain = _build_narrative(p_up, top, features_map, include_header=False)
    lines.append(narrative_text)
    lines.append("")
    if div_notes:
        lines.append("- 상충/특이점: " + "; ".join(div_notes))
        lines.append("")
    lines.append("### 🔎 주요 드라이버(로그우도비)")
    lines.append(_driver_table(top))
    lines.append("")
    driver_notes = _build_driver_notes(top, features_map, float(p_up))
    if driver_notes:
        lines.append("### 💬 드라이버 해석 가이드")
        lines.extend(driver_notes)
        lines.append("")

    lines.append("### 🧭 특징값 요약")
    kv = [
        ("ADR", features_map.get("ADR")),
        ("%>MA50", features_map.get("Pct>MA50")),
        ("%>MA200", features_map.get("Pct>MA200")),
        ("NH/NL", features_map.get("NH/NL")),
        ("RV20", features_map.get("RV20")),
        ("RV60", features_map.get("RV60")),
        ("10Y-3M", features_map.get("SPR_10Y_3M")),
        ("10Y-2Y", features_map.get("SPR_10Y_2Y")),
        ("Curve(30Y+3M-2*10Y)", features_map.get("CURVATURE")),
    ]
    tbl = ["| 지표 | 값 |", "|:--|--:|"]
    for k, v in kv:
        tbl.append(f"| {k} | {_format_num(v,3)} |")
    lines.append("\n".join(tbl))
    lines.append("")
    # Feature glossary
    lines.append("#### ℹ️ 지표 설명(간단)")
    lines.append("- ADR: 상승/하락 종목 수 비율(>1 상승 우세)")
    lines.append("- %>MA50/%>MA200: 50/200일선 상회 비중(참가도·추세)")
    lines.append("- NH/NL: 52주 신고/신저 근접 비율(추세 견고/약화)")
    lines.append("- RV20/60: 20/60일 실현변동성(연환산). 높을수록 변동성 환경")
    lines.append("- 10Y−3M/10Y−2Y: 장단기 금리 경사. +는 정상화·완화, −는 역전 리스크")
    lines.append("- Curve: (30Y+3M−2×10Y) 수익률곡선 곡률. +는 정상화 방향")
    lines.append("")

    if sectors_md:
        lines.append("### 🗂️ 섹터 상대강도(1일)")
        lines.append(sectors_md)
        lines.append("")
    if cal_md:
        lines.append("### 🗓️ 경제 캘린더(예정)")
        lines.append(cal_md)
        lines.append("")

    srcs: List[str] = []
    if fmp.enabled:
        srcs.append("FMP(sector-performance, etf-holder(SPY/VOO/IVV), batch-quote, treasury-rates, economics-calendar)")
    if sot_payload:
        srcs.append("SoT(payload series)")
    # no local file fallback by policy
    if not srcs:
        srcs.append("No external sources (degraded)")

    lines.append("### ℹ️ 데이터 기준/출처")
    lines.append("- 출처: " + ", ".join(srcs))
    lines.append("- 결측/장애 시 보고서는 축소 모드로 출력됩니다.")

    markdown = "\n".join(lines)

    result = {
        "prob": {"p_up": float(p_up), "p_up_raw": float(p_raw)},
        "drivers": top,
        "features": features_map,
        "markdown": markdown,
        "narrative": narrative_plain,
        "driver_notes": driver_notes,
        "refs": {
            "base_symbol": base_symbol,
            "horizon_days": horizon_days,
            "fmp_enabled": fmp.enabled,
            "ctx_fmp": ctx_fmp,
            "ctx_sot": ctx_sot,
            "policy": {
                "no_local_files": True,
                "breadth_universe": "ETF(SPY/VOO/IVV)",
                "no_yahoo_fallback": True,
                "feature_clamps": {
                    name: {"min": bounds[0], "max": bounds[1]} for name, bounds in FEATURE_CLAMPS.items()
                },
                "cold_start": cold_start,
            },
        },
    }
    try:
        _record_report_history(
            result,
            base_symbol=base_symbol,
            horizon_days=horizon_days,
            sot_payload=sot_payload,
        )
    except Exception:
        pass
    return result


# ------------------------------------------------------------
# CLI entry (for quick preview / manual run)
# ------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> Any:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Market Probabilistic Report")
    p.add_argument("--horizon", type=int, default=5, help="Horizon days for label doc")
    p.add_argument("--lookback", type=int, default=1260, help="Lookback days (doc)")
    p.add_argument("--no-cache", action="store_true", help="Ignore cached model params")
    p.add_argument("--no-calib", action="store_true", help="Disable Platt scaling")
    p.add_argument("--benchmark", type=str, default=None, help="Base symbol (default env or QQQ)")
    p.add_argument("--print", action="store_true", help="Print markdown to stdout")
    p.add_argument("--json", action="store_true", help="Print JSON to stdout")
    return p.parse_args(argv)


def _main() -> None:  # pragma: no cover
    args = _parse_args()
    rep = generate_market_report(
        horizon_days=args.horizon,
        lookback_days=args.lookback,
        use_cache=not args.no_cache,
        auto_calibrate=not args.no_calib,
        benchmark=args.benchmark,
        sot_payload=None,
    )
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return
    if args.print or not args.json:
        print(rep.get("markdown", ""))


if __name__ == "__main__":  # pragma: no cover
    _main()
