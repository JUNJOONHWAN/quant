"""Market narrative helpers shared by UI and reports."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple


BASE_SYMBOL_DEFAULT = (os.getenv("REGIME_BASE_SYMBOL") or os.getenv("REGIME_BENCH_SYMBOL") or "QQQ").upper()


def resolve_effective_index(
    dates: List[str],
    asof: Dict[str, Any],
    *,
    base_symbol: Optional[str] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """Determine which index to use (today vs fallback) based on intraday metadata."""

    idx_last = len(dates) - 1
    info: Dict[str, Any] = {
        "used_fallback": False,
        "reason": "",
        "fusion_last": None,
        "today": None,
        "base_stubbed": False,
        "intraday_base": False,
        "quotes_received": False,
    }
    if idx_last < 0:
        return idx_last, idx_last, info

    fusion_last = None
    today = None
    intraday_base = False
    quotes_received = False
    stubbed_like_list: List[str] = []
    if isinstance(asof, dict):
        fusion_last = asof.get("fusion_last_date")
        today = asof.get("today_utc")
        intraday_base = bool(asof.get("intraday_base_applied"))
        quotes_received = bool(asof.get("quotes_received"))
        stubbed_like = asof.get("today_stubbed_like") or []
        if isinstance(stubbed_like, list):
            stubbed_like_list = stubbed_like

    base = (base_symbol or BASE_SYMBOL_DEFAULT or "QQQ").upper()
    stubbed_upper = {str(s).upper() for s in stubbed_like_list}
    base_stubbed = base in stubbed_upper

    idx_eff = idx_last
    disable_fallback = os.getenv("FUSION_DISABLE_RT_FALLBACK", "1") == "1"
    if (not disable_fallback) and idx_last > 0:
        if (fusion_last == today) and (base_stubbed or (not intraday_base) or (not quotes_received)):
            idx_eff = idx_last - 1
            info["used_fallback"] = True
            reasons = []
            if base_stubbed:
                reasons.append(f"{base} stubbed")
            if not intraday_base:
                reasons.append("intraday base not patched")
            if not quotes_received:
                reasons.append("quotes missing")
            info["reason"] = ", ".join(reasons) if reasons else "realtime unavailable"

    info["fusion_last"] = fusion_last
    info["today"] = today
    info["base_stubbed"] = base_stubbed
    info["intraday_base"] = intraday_base
    info["quotes_received"] = quotes_received

    if idx_eff < 0:
        idx_eff = 0
    return idx_last, idx_eff, info


def build_market_narrative(
    payload: Dict[str, Any],
    *,
    base_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a structured narrative + markdown text from SoT payload."""

    dates = payload.get("dates", []) or []
    asof = payload.get("asof", {}) or {}
    base = base_symbol or BASE_SYMBOL_DEFAULT
    idx_last, idx_eff, idx_meta = resolve_effective_index(dates, asof, base_symbol=base)
    if idx_eff < 0 or not dates:
        return {"text": "", "label": "", "notes": [], "info": idx_meta}

    fusion = payload.get("fusion", {}) or {}
    state_arr = fusion.get("state", []) or []
    score_arr = fusion.get("score", []) or []
    diag = fusion.get("diag") or {}

    raw_pos = score_arr[idx_eff] if idx_eff < len(score_arr) else None
    pos = float(raw_pos) if raw_pos is not None else None
    raw_state = state_arr[idx_eff] if idx_eff < len(state_arr) else 0
    st = int(raw_state) if raw_state is not None else 0

    ew = diag.get("EW") or {}
    dr = diag.get("DR") or {}
    cr = diag.get("CR") or {}
    shock = diag.get("Shock") or {}
    gate_cap = float(diag.get("gate_cap") or 1.0)
    cap_ew = float(diag.get("ew_cap") or ew.get("cap") or 1.0)
    cap_dr = float(diag.get("dr_cap") or dr.get("cap") or 1.0)
    cap_cr = float(diag.get("cr_cap") or cr.get("cap") or 1.0)
    cap_sh = float(diag.get("shock_cap") or shock.get("cap") or 1.0)

    z_chi = float(diag.get("z_chi") or 0.0)
    z_eta = float(diag.get("z_eta") or 0.0)
    z_R = float(diag.get("z_R") or 0.0)
    z_dR = float(diag.get("z_dR") or 0.0)
    FQI = float(diag.get("FQI") or 0.0)
    TFI = float(diag.get("TFI") or 0.0)
    wTA = float(diag.get("wTA") or fusion.get("wTA", [0])[-1] if fusion.get("wTA") else 0.0)
    regime_label = str(diag.get("regime_label") or "")
    shock_active = bool(shock.get("active"))

    stab = payload.get("stability", []) or []
    smoo = payload.get("smoothed", []) or []
    delt = payload.get("delta", []) or []
    stab_v = float(stab[idx_eff]) if idx_eff < len(stab) and stab[idx_eff] is not None else float("nan")
    smoo_v = float(smoo[idx_eff]) if idx_eff < len(smoo) and smoo[idx_eff] is not None else float("nan")
    delt_v = float(delt[idx_eff]) if idx_eff < len(delt) and delt[idx_eff] is not None else float("nan")

    sub = payload.get("sub", {}) or {}
    sc = sub.get("stockCrypto", []) or []
    tr = sub.get("traditional", []) or []
    sn = sub.get("safeNegative", []) or []
    sc_v = float(sc[idx_eff]) if idx_eff < len(sc) and sc[idx_eff] is not None else float("nan")
    tr_v = float(tr[idx_eff]) if idx_eff < len(tr) and tr[idx_eff] is not None else float("nan")
    sn_v = float(sn[idx_eff]) if idx_eff < len(sn) and sn[idx_eff] is not None else float("nan")

    def fmt(x: Any, digits: int = 2) -> str:
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return "N/A"

    def nz(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
            if math.isfinite(v):
                return v
        except Exception:
            pass
        return default

    label = ""
    notes: List[str] = []
    strong_pos = (pos is not None and pos >= 0.70)
    mid_pos = (pos is not None and pos >= 0.50)
    very_quiet = math.isfinite(delt_v) and abs(delt_v) < 0.005
    rotation = (
        math.isfinite(sc_v)
        and math.isfinite(tr_v)
        and ((sc_v > 0 and tr_v < 0) or (tr_v > 0 and sc_v < 0))
        and (not math.isfinite(sn_v) or abs(sn_v) < 0.3)
    )
    gates_on = (cap_ew < 1.0) or (cap_dr < 1.0) or (cap_cr < 1.0) or (cap_sh < 1.0) or (gate_cap < 1.0)
    high_cascade = (z_R >= 0.90 or z_dR >= 1.00)

    if st > 0:
        if strong_pos and nz(delt_v, 0.0) > 0.01 and not gates_on and nz(sn_v, 0.0) <= 0.0:
            label = "모멘텀 돌파형 상승"
            notes.append("pos≥0.70, ΔStab>0, 게이트 제한 없음, 방어자금 감소")
        elif mid_pos and nz(delt_v, 0.0) > 0 and (gates_on or high_cascade):
            label = "되돌림/리스크 동반 상승"
            notes.append("ΔStab>0 & 게이트 제한/동조화↑")
        else:
            label = "완만한 상승/추세 유지"
            notes.append("pos 중간 · 게이트 제한 크지 않음")
    elif st == 0:
        if very_quiet and (not rotation):
            label = "압축적 횡보(코일)"
            notes.append("ΔStab≈0, 하위지표 진폭 작음")
        elif rotation:
            label = "순환적 횡보(섹터 로테이션)"
            notes.append("Stock/Crypto vs Traditional 엇갈림")
        else:
            label = "불확실성 박스/조정"
            notes.append("ΔStab 약하고 혼조")
    else:
        if shock_active or cap_sh < 1.0 or (cap_dr < 1.0 and high_cascade):
            label = "급락/캐스케이드 위험 하락"
            notes.append("Shock/DR 활성 또는 결합 급등")
        elif nz(sn_v, 0.0) > 0.30 and nz(sc_v, 0.0) < 0 and nz(tr_v, 0.0) < 0:
            label = "방어적 위험회피 하락"
            notes.append("Safe-NEG↑ & 위험자산 동반 약세")
        else:
            label = "완만한 조정/하락"
            notes.append("게이트 제한 약함 · 완만한 하락")

    # ---- Rich, reference-style narrative ---------------------------------
    def arrow(v: float, thr: float = 0.0) -> str:
        try:
            if not math.isfinite(v):
                return "·"
            if v > thr:
                return "▲"
            if v < -thr:
                return "▼"
            return "·"
        except Exception:
            return "·"

    ew_cnt, ew_tot = ew.get("count"), ew.get("total")
    dr_cnt, dr_tot = dr.get("count"), dr.get("total")
    cr_cnt, cr_tot = cr.get("count"), cr.get("total")
    stab_overlay = bool((diag.get("STAB") or {}).get("overlay"))
    basis = "실시간" if bool(asof.get("intraday_base_applied")) else "장마감"
    f_last = asof.get("fusion_last_date") or dates[idx_eff]
    src = asof.get("override_source")

    lines = [
        "### 🧭 시장 상황 해설",
        f"- 현재 해석: **{label or '정보 부족'}**" + (f" · 레짐태그 {regime_label}" if regime_label else ""),
        f"- 게이트: GateCap {fmt(gate_cap)} · EW {ew_cnt}/{ew_tot} (cap {fmt(cap_ew)}) · DR {dr_cnt}/{dr_tot} (cap {fmt(cap_dr)}) · CR {cr_cnt}/{cr_tot} (cap {fmt(cap_cr)}) · Shock {'on' if shock_active else 'off'} · STAB {'on' if stab_overlay else 'off'}",
        f"- 추세/모멘텀: Stability EMA10 {fmt(smoo_v)} ({arrow(smoo_v)}) · Δ(3-10) {fmt(delt_v)} ({arrow(delt_v, 0.005)}) · score {fmt(pos,3)} · wTA {fmt(wTA)}",
        f"- 자금/섹터: 주식-암호화폐 {fmt(sc_v)} ({arrow(sc_v, 0.02)}) · 전통 {fmt(tr_v)} ({arrow(tr_v, 0.02)}) · Safe-NEG {fmt(sn_v)} ({arrow(sn_v, 0.02)})",
        f"- 참고지표: FQI {fmt(FQI)} · TFI {fmt(TFI)} · 동조화 z_R {fmt(z_R,2)} / z_dR {fmt(z_dR,2)}",
        f"- 기준(ET): {basis} {f_last}" + (f" · source={src}" if isinstance(src, str) and src else ""),
    ]
    # Natural-language synthesis with numbers embedded
    try:
        gate_txt = (
            "제약 없음" if gate_cap >= 0.95 else ("부분 제약" if gate_cap >= 0.60 else "강한 제약")
        )
        mom_txt = (
            "강한 모멘텀" if (pos is not None and pos >= 0.70) else ("완만한 모멘텀" if (pos is not None and pos >= 0.50) else "모멘텀 약함")
        )
        slope_txt = "상승 기울기" if (math.isfinite(delt_v) and delt_v > 0) else ("하락 기울기" if (math.isfinite(delt_v) and delt_v < 0) else "보합")
        risk_bias = "위험선호(Stock/Crypto↑, 전통↓)" if (math.isfinite(sc_v) and math.isfinite(tr_v) and sc_v > 0 and tr_v <= 0) else (
            "방어선호(전통↑)" if (math.isfinite(tr_v) and tr_v > 0 and (not math.isfinite(sc_v) or sc_v <= 0)) else "혼조")
        shock_txt = "Shock 가드 ON" if shock_active else "Shock 가드 OFF"
        coh_txt = "동조화 높음" if (z_R >= 0.9 or z_dR >= 1.0) else "동조화 보통"
        lines.append(
            f"- 해설: 게이트 {gate_txt}(GateCap {fmt(gate_cap)})이고 {mom_txt}(score {fmt(pos,3)}, wTA {fmt(wTA)})입니다. "
            f"안정성은 {slope_txt}(EMA10 {fmt(smoo_v)}, Δ {fmt(delt_v)})이며, 자금흐름은 {risk_bias}입니다. {shock_txt}, {coh_txt}."
        )
    except Exception:
        pass
    # Divergence/Anomaly hints
    try:
        divergences: List[str] = []
        # 상승(>0)인데 리스크 지표가 경고
        if st > 0 and (
            (gate_cap < 0.95) or (z_R >= 0.9) or (z_dR >= 1.0) or (TFI < 0.0) or (FQI < 0.0)
        ):
            divergences.append("상승 중이지만 결합/확산↑ 또는 게이트 제약 → 변동성 주의")
        # 하락(<0)인데 회복 신호가 동반
        if st < 0 and (
            (wTA >= 0.60 and nz(delt_v, 0.0) > 0) or (FQI > 0.10) or (z_eta < -0.4)
        ):
            divergences.append("약세이지만 TA/Flow 회복 단서 → 되돌림 랠리 가능성")
        # 혼합 구간 강조
        if 0.45 <= wTA <= 0.65 and (FQI * TFI) <= 0:
            divergences.append("wTA≈0.5 & 품질 혼조 → 방향성 신뢰도 낮음")
        if divergences:
            lines.append("- 상충/특이점: " + "; ".join(divergences))
    except Exception:
        pass

    if notes:
        lines.append("- 관찰 포인트: " + "; ".join(notes))
    if idx_meta.get("used_fallback"):
        reason = idx_meta.get("reason") or "실시간 호가 미수신"
        lines.append(f"- ⚠️ {reason} → 전일 데이터 기준")

    text = "\n".join(lines)
    metrics = {
        "position": pos,
        "state": st,
        "gate_cap": gate_cap,
        "z_chi": z_chi,
        "z_eta": z_eta,
        "z_R": z_R,
        "z_dR": z_dR,
        "FQI": FQI,
        "TFI": TFI,
        "stability": stab_v,
        "delta": delt_v,
        "stockCrypto": sc_v,
        "traditional": tr_v,
        "safeNegative": sn_v,
    }

    return {
        "text": text,
        "label": label,
        "notes": notes,
        "info": idx_meta,
        "metrics": metrics,
        "refs": {
            "basis": basis,
            "fusion_last": f_last,
            "source": src,
            "gates": {
                "gate_cap": gate_cap,
                "EW": {"count": ew_cnt, "total": ew_tot, "cap": cap_ew},
                "DR": {"count": dr_cnt, "total": dr_tot, "cap": cap_dr},
                "CR": {"count": cr_cnt, "total": cr_tot, "cap": cap_cr},
                "Shock": {"active": shock_active, "cap": cap_sh},
                "STAB_overlay": stab_overlay,
            },
        },
    }


__all__ = ["resolve_effective_index", "build_market_narrative"]
