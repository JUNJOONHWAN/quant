"""Shared regime payload helpers for AutoTrade2 and the web app."""

from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytz
import requests

from regime_backtest.simulator import run_tomorrow_open_backtest
from regime_data.fmp_io import fetch_yahoo_quotes


logger = logging.getLogger("regime_service.signals")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def _default_symbol() -> str:
    symbol = os.getenv("AUTOTRADE2_SYMBOL") or os.getenv("REGIME_STRATEGY_SYMBOL") or "TQQQ"
    return symbol.upper()


DEFAULT_SYMBOL = _default_symbol()
REGIME_CACHE_FILE = os.getenv("AUTOTRADE2_REGIME_CACHE", "autotrade2_regime_cache.json")
FMP_QUOTE_SYMBOLS = ["QQQ", "SPY", "IWM", "TLT", "GLD", "BTC-USD", "TQQQ"]


class RegimeFetcher:
    """Wrapper around realtime_regime_newgate with caching & overrides."""

    def __init__(
        self,
        window: int = 30,
        *,
        symbol: Optional[str] = None,
        cache_file: Optional[str] = None,
    ) -> None:
        self.window = window
        self.symbol = (symbol or DEFAULT_SYMBOL or "TQQQ").upper()
        self.cache_file = cache_file or REGIME_CACHE_FILE
        candidates: List[str] = []
        for cand in [self.symbol, self.symbol.upper(), "TQQQ", "QQQ"]:
            if not cand:
                continue
            key = cand.upper()
            if key not in candidates:
                candidates.append(key)
        self._symbol_candidates = candidates

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        if not self.cache_file or not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return None

    def _save_cache(self, payload: Dict[str, Any]) -> None:
        if not self.cache_file:
            return
        try:
            with open(self.cache_file, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("레짐 캐시 저장 실패: %s", exc)

    # ------------------------------------------------------------------
    # Core fetch/normalize
    # ------------------------------------------------------------------
    def _normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        dates = (payload.get("fusion", {}) or {}).get("dates") or payload.get("dates", [])
        fusion = payload.get("fusion", {}) or {}
        states = fusion.get("state", []) or []
        if not dates or not states:
            raise RuntimeError("레짐 데이터가 비어 있습니다")
        series_map = (payload.get("series", {}) or {})
        series_open_map = (payload.get("series_open", {}) or {})

        def _pickup(targets: List[str], source: Dict[str, Any]) -> Optional[List[float]]:
            for target in targets:
                base = source.get(target)
                if isinstance(base, list):
                    return base
                base = source.get(target.upper())
                if isinstance(base, list):
                    return base
            return None

        symbol_series = _pickup(self._symbol_candidates, series_map)
        if symbol_series is None:
            symbol_series = series_map.get("QQQ") or series_map.get("SPY")
        bench_series = series_map.get("QQQ") or series_map.get("SPY")
        bench_open = series_open_map.get("QQQ") or series_open_map.get("SPY")

        executed = [0] + states[:-1] if states else []
        f_score = (fusion.get("score", []) or [])
        f_wTA = fusion.get("wTA") if isinstance(fusion.get("wTA"), list) else []
        f_wFlow = fusion.get("wFlow") if isinstance(fusion.get("wFlow"), list) else []
        f_dates = fusion.get("dates") or dates
        f_engine = fusion.get("engine") or (payload.get("asof", {}) or {}).get("fusion_engine")
        f_preset = fusion.get("preset") or payload.get("fusion_preset")

        score_last = f_score[-1] if f_score else None
        wTA_last = f_wTA[-1] if f_wTA else None
        wFlow_last = f_wFlow[-1] if f_wFlow else None
        asof = payload.get("asof", {}) or {}
        last_is_today = False
        try:
            t = asof.get("today_utc")
            dlast = f_dates[-1] if f_dates else None
            last_is_today = bool(t and dlast and str(dlast) == str(t))
        except Exception:
            last_is_today = False

        return {
            "state": states[-1],
            "date": dates[-1],
            "states": states,
            "dates": dates,
            "series_close": symbol_series,
            "series_bench": bench_series,
            "series_bench_open": bench_open,
            "executed_states": executed,
            "diag": fusion.get("diag", {}),
            "mode": fusion.get("mode"),
            "asof": asof,
            "fusion_engine": f_engine,
            "fusion_preset": f_preset,
            "fusion_score_last": score_last,
            "fusion_wTA_last": wTA_last,
            "fusion_wFlow_last": wFlow_last,
            "fusion_score_series": f_score,
            "fusion_wTA_series": f_wTA,
            "fusion_wFlow_series": f_wFlow,
            "fusion_dates": f_dates,
            "fusion_last_is_today": last_is_today,
            "series": payload.get("series"),
            "series_open": payload.get("series_open"),
        }

    def snapshot(self, use_realtime: bool) -> Dict[str, Any]:
        from realtime_regime_newgate import compute_realtime_regime  # type: ignore

        overrides = self._auto_premarket_overrides() if use_realtime else {}
        params: Dict[str, Any] = {"window": self.window, "use_realtime": use_realtime}
        if overrides:
            params["override_last"] = overrides
        payload = compute_realtime_regime(**params)
        return self._normalize(payload)

    def latest_state(self) -> Dict[str, Any]:
        from realtime_regime_newgate import compute_realtime_regime  # type: ignore

        try:
            overrides = self._auto_premarket_overrides()
            params: Dict[str, Any] = {"window": self.window, "use_realtime": True}
            if overrides:
                params["override_last"] = overrides
            payload = compute_realtime_regime(**params)
            self._save_cache(payload)
        except Exception as exc:
            logger.error("realtime_regime 호출 실패: %s", exc)
            payload = self._load_cache()
            if not payload:
                raise
        return self._normalize(payload)

    # ------------------------------------------------------------------
    # Override helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _et_session_stage() -> str:
        try:
            et = pytz.timezone("America/New_York")
            now = datetime.now(et)
            pre = now.replace(hour=9, minute=30, second=0, microsecond=0)
            close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            if now < pre:
                return "premarket"
            if now >= close:
                return "post"
            return "regular"
        except Exception:
            return "regular"

    def _auto_premarket_overrides(self) -> Dict[str, float]:
        stage = self._et_session_stage()
        if stage == "regular":
            return {}
        symbols = FMP_QUOTE_SYMBOLS
        # 1) Try FMP quote endpoint
        api_key = os.getenv("FMP_API_KEY")
        if api_key:
            try:
                url = f"https://financialmodelingprep.com/api/v3/quote/{','.join(symbols)}?apikey={api_key}"
                resp = requests.get(url, timeout=6)
                resp.raise_for_status()
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else []
                if isinstance(data, list):
                    out: Dict[str, float] = {}
                    for item in data:
                        try:
                            sym = str(item.get("symbol") or "").upper()
                            if not sym:
                                continue
                            val = None
                            if stage == "premarket":
                                val = item.get("preMarketPrice") or item.get("premarketPrice")
                            elif stage == "post":
                                val = item.get("postMarketPrice") or item.get("afterHoursPrice")
                            if isinstance(val, (int, float)) and val > 0:
                                out[sym] = float(val)
                        except Exception:
                            continue
                    if out:
                        # Mark source/stage for downstream reporting
                        out["__source__"] = "FMP"
                        out["__stage__"] = stage
                        return out
            except Exception:
                pass
        # 2) Fallback to Yahoo Finance (source: https://query1.finance.yahoo.com/v7/finance/quote)
        try:
            y_quotes = fetch_yahoo_quotes(symbols)
        except Exception:
            y_quotes = {}
        if not y_quotes:
            return {}
        out: Dict[str, float] = {}
        for sym in symbols:
            info = y_quotes.get(sym.upper())
            if not info:
                continue
            if stage == "premarket":
                val = info.get("preMarketPrice")
            else:
                val = info.get("postMarketPrice")
            if not isinstance(val, (int, float)) or val <= 0:
                continue
            out[sym.upper()] = float(val)
        if out:
            out["__source__"] = "Yahoo"
            out["__stage__"] = stage
        return out


# ----------------------------------------------------------------------
# Shared helpers for payload exposure (SoT APIs)
# ----------------------------------------------------------------------

def _clone_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(data)


def _extract_preset_hint(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    fusion = payload.get("fusion")
    if isinstance(fusion, dict):
        preset = fusion.get("preset")
        if isinstance(preset, str) and preset:
            return preset
    manifest = payload.get("manifest")
    if isinstance(manifest, dict):
        preset = manifest.get("preset")
        if isinstance(preset, str) and preset:
            return preset
    preset = payload.get("fusion_preset")
    if isinstance(preset, str) and preset:
        return preset
    return None


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    trimmed = value.strip()
    if trimmed.endswith("UTC"):
        trimmed = trimmed[:-3].strip()
    if trimmed.endswith("Z"):
        trimmed = trimmed[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(trimmed)
    except Exception:
        dt = None
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(trimmed, fmt)
                break
            except Exception:
                continue
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_cache_stale(payload: Dict[str, Any], max_age_minutes: int) -> bool:
    if max_age_minutes <= 0:
        return False
    asof = payload.get("asof", {}) if isinstance(payload, dict) else {}
    quote_ts = None
    if isinstance(asof, dict):
        quote_ts = asof.get("quote_ts_max") or asof.get("quote_ts_last")
    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    if quote_ts is None and isinstance(manifest, dict):
        quote_ts = manifest.get("as_of_ts")
    dt_quote = _parse_iso8601(quote_ts)
    if dt_quote is None:
        return False
    now_utc = datetime.now(timezone.utc)
    age = now_utc - dt_quote
    return age > timedelta(minutes=max_age_minutes)


def _ensure_manifest(
    payload: Dict[str, Any],
    *,
    origin: str,
    mode: str,
    use_realtime: bool,
    intraday_base: bool,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = payload.setdefault("manifest", {})
    manifest.setdefault("origin", origin)
    manifest.setdefault("mode", mode)
    manifest.setdefault("use_realtime", use_realtime)
    manifest.setdefault("intraday_base", intraday_base)
    manifest.setdefault("tz", "America/New_York")
    manifest.setdefault("anchor", "ET_16:00")
    manifest.setdefault("as_of_ts", datetime.now(timezone.utc).isoformat())
    if preset:
        manifest.setdefault("preset", preset)
    return payload


# ----------------------------------------------------------------------
# Public payload/backtest helpers (SoT API)
# ----------------------------------------------------------------------

def at2_get_payload_close_raw(window: int = 30, preset: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    from realtime_regime_newgate import compute_realtime_regime  # type: ignore

    fetcher = RegimeFetcher(window)
    try:
        cached_payload = fetcher._load_cache()
    except Exception:
        cached_payload = None
    preset_hint = _extract_preset_hint(cached_payload)
    preset_arg: Optional[str] = preset if preset is not None else preset_hint

    params: Dict[str, Any] = {"window": window, "use_realtime": False}
    if preset_arg is not None:
        params["preset"] = preset_arg
    payload = compute_realtime_regime(**params)
    if isinstance(payload, dict):
        _ensure_manifest(
            payload,
            origin="autotrade2.close",
            mode="close",
            use_realtime=False,
            intraday_base=False,
            preset=preset_arg,
        )
    return payload


def at2_get_payload_now_raw(window: int = 30, preset: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    from realtime_regime_newgate import compute_realtime_regime  # type: ignore

    fetcher = RegimeFetcher(window)
    auto_override = bool(kwargs.pop("auto_override", True))
    force_refresh = bool(kwargs.pop("force_refresh", False))
    prefer_cache = kwargs.pop("prefer_cache", True)
    cache_max_age = int(kwargs.pop("cache_max_age", 10))

    overrides: Dict[str, float] = {}
    if auto_override:
        try:
            overrides = fetcher._auto_premarket_overrides()
        except Exception:
            overrides = {}

    cached_payload: Optional[Dict[str, Any]] = None
    preset_hint: Optional[str] = None
    if prefer_cache and not force_refresh:
        try:
            cached_raw = fetcher._load_cache()
        except Exception:
            cached_raw = None
        if isinstance(cached_raw, dict) and cached_raw:
            preset_hint = _extract_preset_hint(cached_raw)
            if not _is_cache_stale(cached_raw, cache_max_age):
                cached_payload = _clone_payload(cached_raw)
                payload = _ensure_manifest(
                    cached_payload,
                    origin="autotrade2.cache_realtime",
                    mode="realtime",
                    use_realtime=True,
                    intraday_base=True,
                    preset=preset_hint,
                )
                return payload

    preset_arg: Optional[str] = preset if preset is not None else preset_hint
    params: Dict[str, Any] = {"window": window, "use_realtime": True}
    if auto_override and overrides:
        params["override_last"] = overrides
    if preset_arg is not None:
        params["preset"] = preset_arg

    try:
        payload = compute_realtime_regime(**params)
    except TypeError:
        params.pop("override_last", None)
        payload = compute_realtime_regime(window=window, use_realtime=True)

    if isinstance(payload, dict):
        _ensure_manifest(
            payload,
            origin="autotrade2.realtime",
            mode="realtime",
            use_realtime=True,
            intraday_base=True,
            preset=preset_arg,
        )
        # annotate override source/stage if present
        try:
            if auto_override and overrides and isinstance(overrides, dict):
                src = overrides.get("__source__")
                stg = overrides.get("__stage__")
                if src or stg:
                    asof = payload.setdefault("asof", {})
                    if src:
                        asof["override_source"] = src
                    if stg:
                        asof["override_stage"] = stg
        except Exception:
            pass
    return payload


def at2_get_ticker_series(
    window: int = 30,
    preset: Optional[str] = None,
    use_realtime: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    payload = (
        at2_get_payload_now_raw(window=window, preset=preset, **kwargs)
        if use_realtime
        else at2_get_payload_close_raw(window=window, preset=preset, **kwargs)
    )
    return {
        "dates": payload.get("dates", []),
        "series": payload.get("series", {}) or {},
        "series_open": payload.get("series_open", {}) or {},
    }


def at2_backtest_close(
    start_date: str,
    end_date: str,
    *,
    window: int = 30,
    preset: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    payload = at2_get_payload_close_raw(window=window, preset=preset, **kwargs)
    return run_tomorrow_open_backtest(
        payload,
        start_date=start_date,
        end_date=end_date,
    )


__all__ = [
    "RegimeFetcher",
    "at2_get_payload_close_raw",
    "at2_get_payload_now_raw",
    "at2_get_ticker_series",
    "at2_backtest_close",
]
