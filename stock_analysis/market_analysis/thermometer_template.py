"""Reusable 420px thermometer report renderer.

The renderer is intentionally dependency-free so Codex sessions and local
scripts can use the same mobile HTML shell for market, ETF, BTC, and single
stock thermometer reports.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dgx_paths import choose_output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_PATH = PROJECT_ROOT / "reports/templates/thermometer_mobile420_base.html"
LOCAL_THERMOMETER_DIR = PROJECT_ROOT / "sweet_spot_reports"
ICLOUD_THERMOMETER_DIR = (
    choose_output_dir(
        "STOCK_THERMOMETER_DIR",
        "온도계",
        legacy_mac_path=Path("/home/zooh/Documents/DGX_Outputs/STOCK/온도계"),
    )
)


def _text(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return max(0.0, min(100.0, out))


ABBR = {
    "현재가": "가",
    "변화율": "변",
    "위험점수": "Risk",
    "시간": "시각",
    "대상": "대상",
    "헤드라인/상태": "헤드",
    "항목": "항목",
    "상태": "상태",
    "처리": "근거",
    "값/상태": "값",
    "근거": "근거",
    "선물 보강": "선물",
    "옵션 확장": "옵션",
    "Barchart Chrome": "옵션웹",
    "Barchart 시장 대시보드": "시장웹",
    "Barchart Chrome 추출 요약": "웹요약",
    "옵션/감마·Barchart 필수 체크": "옵션·감마",
    "뉴스/기관/이벤트": "뉴스·이벤트",
    "리더/방어 기술 흐름": "TA·방어",
    "핵심 자산 히트맵": "자산 Heat",
    "온도 구성 점수": "온도 Score",
    "데이터 소스 상태": "소스 상태",
    "데이터 소스 미사용 사유서": "제한 사유",
    "시장 구동축": "구동축",
    "가격·VIX·시장 폭": "가격·VIX",
    "옵션·선물·Barchart": "옵션·선물",
    "뉴스·커뮤니티": "뉴스·커뮤",
    "데이터 감사": "감사",
    "커뮤니티 분위기": "커뮤 Mood",
    "카나리아 조기경보": "Canary",
    "크레딧/방어 자산": "방어자산",
}


KIND_INFO = {
    "score_table": "온도 구성 요소별 점수입니다. 50은 중립, 70 이상은 우호, 45 미만은 경계로 봅니다.",
    "heatmap": "자산별 당일 온도를 색과 숫자로 압축한 화면입니다.",
    "bar_chart": "막대가 길수록 해당 항목이 현재 온도에 더 우호적입니다.",
    "technical_analysis": "가격 흐름과 리더·방어 자산의 기술적 방향을 함께 본 모듈입니다.",
    "sentiment_shift": "뉴스와 커뮤니티 분위기의 변화 방향입니다. 점수보다 변화가 더 중요합니다.",
    "vix_analysis": "VIX 상승은 위험 확대, 하락은 위험 완화로 반영합니다.",
    "canary_board": "시장의 온도 변화를 먼저 감지하는 조기경보 축입니다.",
    "breadth_analysis": "상승 종목 비율, 거래량, 동일가중 괴리로 장의 폭을 봅니다.",
    "data_freshness": "데이터가 실제 확인됐는지와 제한 사항을 기록합니다.",
    "table": "세부 체크 결과입니다. 긴 문구는 짧게 줄이고 원문은 셀 설명에 남깁니다.",
    "checklist": "프로세스에서 확인해야 하는 항목 목록입니다.",
    "status_list": "대시보드나 페이지별 확인 상태입니다.",
    "note": "데이터 한계와 해석 주의점을 적은 감사 메모입니다.",
}


def _short_label(value: Any) -> str:
    text = public_text(value)
    return ABBR.get(text, text)


def public_text(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = (
        ("TopstepX futures pulse", "선물 pulse"),
        ("TopstepX/선물 MCP", "선물 데이터"),
        ("TopstepX", "선물"),
        ("Topstep/Barchart", "선물/옵션"),
        ("Barchart Chrome", "옵션웹"),
        ("Barchart Premier", "옵션웹"),
        ("Barchart", "옵션웹"),
        ("Chrome 로그인 세션", "로그인 웹"),
        ("Chrome", "웹"),
        ("Massive grouped daily", "시장폭"),
        ("Massive", "시장폭"),
        ("FMP 뉴스", "뉴스"),
        ("FMP quote", "가격 quote"),
        ("FMP VIX", "VIX"),
        ("FMP", "가격"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _short(value: Any, limit: int = 38) -> str:
    text = public_text(value)
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _summary_class(*values: Any) -> str:
    text = " ".join(public_text(value) for value in values if value is not None)
    return " summaryCard" if "종합" in text else ""


def info_icon(text: Any) -> str:
    tooltip = _short(text, 190)
    if not tooltip:
        return ""
    return (
        '<button class="info" type="button" aria-label="설명" tabindex="0">'
        'i<span class="tip">%s</span></button>'
    ) % _text(tooltip)


def display_source_name(value: Any) -> str:
    text = str(value or "")
    if "TopstepX" in text:
        return "선물"
    if text == "Barchart Chrome":
        return "옵션웹"
    if text == "FMP 뉴스":
        return "뉴스"
    if text == "FMP":
        return "가격"
    if text == "Massive":
        return "시장폭"
    return ABBR.get(text, text)


def tone_color(score: Any) -> str:
    score_f = _num(score)
    if score_f >= 70:
        return "#2f8f6b"
    if score_f >= 55:
        return "#b8872e"
    if score_f >= 45:
        return "#6d6a9f"
    return "#bf4b4b"


def tone_name(score: Any) -> str:
    score_f = _num(score)
    if score_f >= 70:
        return "good"
    if score_f >= 55:
        return "warn"
    if score_f >= 45:
        return "violet"
    return "bad"


def status_tone(value: Any) -> str:
    text = str(value or "").lower()
    if any(word in text for word in ("실패", "bad", "risk-off", "de-risk", "약화", "악화")):
        return "bad"
    if any(word in text for word in ("부분", "warn", "neutral", "혼조", "중립", "예비")):
        return "warn"
    if any(word in text for word in ("확인", "good", "risk-on", "buy", "개선", "양호")):
        return "good"
    return "blue"


def heat_color(score: Any) -> str:
    score_f = _num(score)
    if score_f >= 70:
        return "linear-gradient(180deg, #dfeee8, #f7fbf9)"
    if score_f >= 55:
        return "linear-gradient(180deg, #f0e2c3, #fbf6ec)"
    if score_f >= 45:
        return "linear-gradient(180deg, #e4e3f1, #f7f7fb)"
    return "linear-gradient(180deg, #efdada, #fbf1f1)"


def render_meter(score: Any) -> str:
    score_f = _num(score)
    return (
        '<div class="meter"><div style="width:%.0f%%;background:%s"></div></div>'
        % (score_f, tone_color(score_f))
    )


def render_tags(tags: Sequence[Any]) -> str:
    if not tags:
        return ""
    html = []
    for tag in tags:
        if isinstance(tag, Mapping):
            label = tag.get("label") or tag.get("name") or tag.get("text")
            tone = tag.get("tone") or tag.get("status") or status_tone(label)
        else:
            label = tag
            tone = status_tone(tag)
        html.append('<span class="tag %s">%s</span>' % (_text(tone), _text(_short(label, 18))))
    return '<div class="tagrow">%s</div>' % "".join(html)


def render_decision_grid(report: Mapping[str, Any]) -> str:
    points = report.get("decision_points") or []
    if not points:
        points = [
            {
                "label": item.get("name"),
                "value": "%.0f/100" % _num(item.get("score")),
                "note": (item.get("lines") or [""])[0],
                "score": item.get("score"),
            }
            for item in (report.get("assets") or [])[:4]
        ]
    cards = []
    for point in points[:4]:
        score = point.get("score")
        tone = point.get("tone") or (tone_name(score) if score is not None else status_tone(point.get("value")))
        cards.append(
            """
<div class="decisionCard">
  <span class="label">{label}</span>
  <span class="value {tone}">{value}</span>
  <div class="note">{note}</div>
</div>""".format(
                label=_text(_short_label(point.get("label"))),
                value=_text(_short(point.get("value"), 16)),
                tone=_text(tone),
                note=_text(_short(point.get("note"), 54)),
            )
        )
    return '<div class="decisionGrid">%s</div>' % "".join(cards) if cards else ""


def render_ai_diagnosis(report: Mapping[str, Any]) -> str:
    diagnosis = report.get("ai_diagnosis") or {}
    if not isinstance(diagnosis, Mapping):
        diagnosis = {}
    if not diagnosis:
        score = _num(report.get("overall_score"))
        verdict = report.get("verdict_label") or "Neutral"
        if score >= 70:
            state = "위험선호 우위"
            stance = "추세 추종"
        elif score >= 55:
            state = "완만한 위험선호"
            stance = "분할 확인"
        elif score >= 45:
            state = "중립권 압축"
            stance = "관망 우선"
        else:
            state = "방어 우위"
            stance = "리스크 축소"
        diagnosis = {
            "state": state,
            "stance": stance,
            "summary": (
                f"{verdict} {score:.0f}/100 구간입니다. 방향성은 아직 확정되지 않았고, "
                "가격·선물·변동성·커뮤니티 축의 동조 여부를 확인해야 합니다."
            ),
            "cards": [
                {"label": "Bias", "value": state, "note": "현재 온도계가 읽는 시장 성격"},
                {"label": "Mode", "value": stance, "note": "매수·매도 전 기본 태도"},
                {"label": "Watch", "value": "VIX·선물", "note": "온도 변화가 빠르게 드러나는 축"},
                {"label": "Check", "value": "옵션·커뮤", "note": "과열·공포 확인용 보조 축"},
            ],
        }
    cards = []
    for card in (diagnosis.get("cards") or [])[:4]:
        cards.append(
            """
<div class="diagCard">
  <span>{label}</span>
  <b>{value}</b>
  <p>{note}</p>
</div>""".format(
                label=_text(_short(card.get("label"), 14)),
                value=_text(_short(card.get("value"), 24)),
                note=_text(_short(card.get("note"), 54)),
            )
        )
    return """
<section class="section diagnosis">
  <div class="diagnosisHead">
    <div>
      <div class="diagnosisKicker">AI DIAGNOSIS</div>
      <h2>AI 종합 진단</h2>
    </div>
    {info}
  </div>
  <div class="diagnosisState">{state} · {stance}</div>
  <p class="diagnosisText">{summary}</p>
  <div class="diagnosisGrid">{cards}</div>
</section>""".format(
        info=info_icon("온도계의 여러 데이터 축을 한 문장으로 합성한 현재 시장 상태 설명입니다."),
        state=_text(_short(diagnosis.get("state"), 22)),
        stance=_text(_short(diagnosis.get("stance"), 18)),
        summary=_text(_short(diagnosis.get("summary"), 190)),
        cards="".join(cards),
    )


def render_asset_cards(assets: Sequence[Mapping[str, Any]]) -> str:
    if not assets:
        return ""
    cards = []
    for item in assets:
        score = _num(item.get("score"))
        lines = item.get("details") or item.get("lines") or []
        line_html = "".join("<p>%s</p>" % _text(_short(line, 78)) for line in lines)
        score_tone = tone_name(score)
        price = _short(item.get("price"), 24)
        change = _short(item.get("change"), 18)
        primary_html = ""
        if price or change:
            primary_html = """
  <div class="assetPrimary">
    <strong>{price}</strong>
    <span class="{tone}">{change}</span>
  </div>""".format(
                price=_text(price),
                change=_text(change),
                tone=_text(score_tone),
            )
        cards.append(
            """
<div class="card">
  <div class="assetTop"><h2>{name}</h2><span class="tag">{label}</span></div>
  {primary}
  <div class="assetScore {tone}">{score:.0f}<small>/100</small></div>
  {meter}
  {lines}
</div>""".format(
                name=_text(_short_label(item.get("name"))),
                label=_text(_short(item.get("label"), 16)),
                primary=primary_html,
                tone=score_tone,
                score=score,
                meter=render_meter(score),
                lines=line_html,
            )
        )
    return '<div class="cards">%s</div>' % "".join(cards)


def render_score_table(module: Mapping[str, Any]) -> str:
    rows = []
    for row in module.get("rows") or []:
        score = _num(row.get("score"))
        rows.append(
            """
<div class="scoreRow">
  <div class="scoreName">{name}</div>
  {meter}
  <div class="scoreValue">{score:.0f}</div>
</div>""".format(
                name=_text(_short_label(row.get("name"))),
                meter=render_meter(score),
                score=score,
            )
        )
    return '<div class="scoreTable">%s</div>' % "".join(rows)


def render_heatmap(module: Mapping[str, Any]) -> str:
    cells = []
    for cell in module.get("cells") or []:
        score = _num(cell.get("score"))
        cells.append(
            """
<div class="heatCell{summary_class}" style="background:{color}">
  <span class="heatLabel">{label}</span>
  <span class="heatValue">{value}</span>
</div>""".format(
                color=heat_color(score),
                summary_class=_summary_class(cell.get("label")),
                label=_text(_short_label(cell.get("label"))),
                value=_text(_short(cell.get("value", "%.0f" % score), 14)),
            )
        )
    return '<div class="heatmap">%s</div>' % "".join(cells)


def render_bar_chart(module: Mapping[str, Any]) -> str:
    bars = []
    for row in module.get("bars") or []:
        value = _num(row.get("value"))
        bars.append(
            """
<div class="barRow">
  <div class="scoreName">{label}</div>
  <div class="barTrack"><div class="barFill" style="width:{value:.0f}%;background:{color}"></div></div>
  <div class="scoreValue">{value:.0f}</div>
</div>""".format(
                label=_text(_short_label(row.get("label"))),
                value=value,
                color=_text(row.get("color") or tone_color(value)),
            )
        )
    return '<div class="bars">%s</div>' % "".join(bars)


def render_technical_analysis(module: Mapping[str, Any]) -> str:
    rows = []
    for item in module.get("indicators") or []:
        score = _num(item.get("score"), 50.0)
        rows.append(
            """
<div class="techItem{summary_class}">
  <div class="techTop">
    <b>{name}</b>
    <span style="color:{color}">{signal}</span>
  </div>
  <div class="techMeta">{value}</div>
  {meter}
  <p>{note}</p>
</div>""".format(
                name=_text(_short_label(item.get("name"))),
                summary_class=_summary_class(item.get("name")),
                signal=_text(_short(item.get("signal") or item.get("status"), 18)),
                value=_text(_short(item.get("value"), 22)),
                color=_text(item.get("color") or tone_color(score)),
                meter=render_meter(score),
                note=_text(_short(item.get("note"), 72)),
            )
        )
    return '<div class="technicalGrid">%s</div>' % "".join(rows)


def render_sentiment_shift(module: Mapping[str, Any]) -> str:
    points = module.get("points") or []
    if not points:
        return '<p class="note">센티먼트 변화 데이터가 부족합니다.</p>'
    rows = []
    previous = None
    compare_sequential = bool(module.get("compare_sequential"))
    for point in points:
        score = _num(point.get("score"), 50.0)
        delta = None
        delta_label = str(point.get("status") or point.get("delta_label") or "수집")
        delta_class = "flat"
        if "delta" in point:
            delta = _num(point.get("delta"), 0.0)
        elif compare_sequential:
            delta = None if previous is None else score - previous
            previous = score
            delta_label = "첫값" if delta is None else delta_label
        if delta is not None:
            if delta > 0:
                delta_label = "+%.0f" % delta
                delta_class = "up"
            elif delta < 0:
                delta_label = "%.0f" % delta
                delta_class = "down"
            else:
                delta_label = "0"
        rows.append(
            """
<div class="sentimentPoint{summary_class}">
  <div class="sentimentTime">{label}</div>
  <div class="sentimentTrack"><div style="width:{score:.0f}%;background:{color}"></div></div>
  <div class="sentimentScore">{score:.0f}</div>
  <div class="sentimentDelta {delta_class}">{delta_label}</div>
  <p>{note}</p>
</div>""".format(
                label=_text(_short_label(point.get("label") or point.get("time"))),
                summary_class=_summary_class(point.get("label") or point.get("time")),
                score=score,
                color=_text(point.get("color") or tone_color(score)),
                delta_class=delta_class,
                delta_label=_text(delta_label),
                note=_text(_short(point.get("note"), 72)),
            )
        )
    return '<div class="sentimentFlow">%s</div>' % "".join(rows)


def _signed_pct(value: Any) -> str:
    try:
        value_f = float(value)
    except Exception:
        return _text(value)
    return "%+.2f%%" % value_f


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def render_vix_analysis(module: Mapping[str, Any]) -> str:
    current = module.get("current")
    change_pct = module.get("change_pct")
    score = _num(module.get("score"), 50.0)
    metrics = module.get("metrics") or []
    levels = module.get("levels") or []
    metric_html = []
    for metric in metrics:
        metric_html.append(
            """
<div class="vixMetric{summary_class}">
  <span>{label}</span>
  <b style="color:{color}">{value}</b>
</div>""".format(
                label=_text(_short_label(metric.get("label"))),
                summary_class=_summary_class(metric.get("label")),
                value=_text(_short(metric.get("value"), 16)),
                color=_text(metric.get("color") or tone_color(metric.get("score", score))),
            )
        )
    level_html = []
    for level in levels:
        level_score = _num(level.get("score"), 50.0)
        level_html.append(
            """
<div class="vixLevel">
  <div class="techTop"><b>{name}</b><span style="color:{color}">{signal}</span></div>
  <div class="techMeta">{value}</div>
  {meter}
</div>""".format(
                name=_text(_short_label(level.get("name"))),
                signal=_text(_short(level.get("signal"), 16)),
                value=_text(_short(level.get("value"), 22)),
                color=_text(level.get("color") or tone_color(level_score)),
                meter=render_meter(level_score),
            )
        )
    return """
<div class="vixPanel">
  <div class="vixMain">
    <div>
      <span>VIX</span>
      <strong>{current}</strong>
    </div>
    <div>
      <span>변</span>
      <strong style="color:{change_color}">{change}</strong>
    </div>
    <div>
      <span>Risk</span>
      <strong style="color:{score_color}">{score:.0f}</strong>
    </div>
  </div>
  {meter}
  <p title="{summary_title}">{summary}</p>
  <div class="vixMetrics">{metrics}</div>
  <div class="vixLevels">{levels}</div>
</div>""".format(
        current=_text(current),
        change=_signed_pct(change_pct),
        change_color=("#c83c3c" if _float_or_zero(change_pct) > 0 else "#15995e"),
        score=score,
        score_color=tone_color(score),
        meter=render_meter(score),
        summary=_text(_short(module.get("summary"), 82)),
        summary_title=_text(public_text(module.get("summary"))),
        metrics="".join(metric_html),
        levels="".join(level_html),
    )


def render_canary_board(module: Mapping[str, Any]) -> str:
    items = []
    for item in module.get("items") or []:
        score = _num(item.get("score"), 50.0)
        delta = item.get("delta")
        delta_text = "" if delta is None else _signed_pct(delta) if item.get("delta_is_pct") else _text(delta)
        items.append(
            """
<div class="canaryItem{summary_class}">
  <div class="canaryTop">
    <b>{name}</b>
    <span style="color:{color}">{status}</span>
  </div>
  <div class="canaryScore">{score:.0f}<small>/100</small>{delta}</div>
  {meter}
  <div class="techMeta">{trigger}</div>
  <p>{note}</p>
</div>""".format(
                name=_text(_short_label(item.get("name"))),
                summary_class=_summary_class(item.get("name")),
                status=_text(_short(item.get("status") or item.get("signal"), 16)),
                color=_text(item.get("color") or tone_color(score)),
                score=score,
                delta=('<em>%s</em>' % delta_text) if delta_text else "",
                meter=render_meter(score),
                trigger=_text(_short(item.get("trigger") or item.get("source"), 46)),
                note=_text(_short(item.get("note"), 72)),
            )
        )
    return '<div class="canaryGrid">%s</div>' % "".join(items)


def render_breadth_analysis(module: Mapping[str, Any]) -> str:
    cards = []
    for item in module.get("metrics") or []:
        score = _num(item.get("score"), 50.0)
        cards.append(
            """
<div class="breadthCard{summary_class}">
  <span>{label}</span>
  <strong style="color:{color}">{value}</strong>
  {meter}
  <p>{note}</p>
</div>""".format(
                label=_text(_short_label(item.get("label"))),
                summary_class=_summary_class(item.get("label")),
                value=_text(_short(item.get("value"), 18)),
                color=_text(item.get("color") or tone_color(score)),
                meter=render_meter(score),
                note=_text(_short(item.get("note"), 58)),
            )
        )
    return """
<div class="breadthPanel">
  <p title="{summary_title}">{summary}</p>
  <div class="breadthGrid">{cards}</div>
</div>""".format(
        summary=_text(_short(module.get("summary"), 82)),
        summary_title=_text(public_text(module.get("summary"))),
        cards="".join(cards),
    )


def render_data_freshness(module: Mapping[str, Any]) -> str:
    rows = []
    for item in module.get("sources") or []:
        status = str(item.get("status") or "부분 제한")
        status_class = "ok" if status == "확인됨" else "bad" if status == "실패" else "warn"
        rows.append(
            """
<div class="freshRow">
  <div>
    <b>{name}</b>
    <span>{detail}</span>
  </div>
  <div class="{status_class}">{status}</div>
  <div>{asof}</div>
</div>""".format(
                name=_text(display_source_name(item.get("name"))),
                detail=_text(_short(item.get("detail"), 64)),
                status_class=status_class,
                status=_text(status),
                asof=_text(item.get("asof") or item.get("age")),
            )
        )
    return '<div class="freshnessList">%s</div>' % "".join(rows)


def render_line_chart(module: Mapping[str, Any]) -> str:
    points = [float(p) for p in (module.get("points") or []) if isinstance(p, (int, float))]
    if len(points) < 2:
        return '<p class="note">차트 데이터가 부족합니다.</p>'
    min_v = min(points)
    max_v = max(points)
    span = max(max_v - min_v, 1e-9)
    coords = []
    for idx, value in enumerate(points):
        x = 10 + idx * (280 / max(1, len(points) - 1))
        y = 106 - ((value - min_v) / span) * 82
        coords.append("%.1f,%.1f" % (x, y))
    return """
<svg class="miniChart" viewBox="0 0 300 132" role="img" aria-label="{label}">
  <line x1="10" y1="106" x2="290" y2="106" stroke="#c9dcff" />
  <line x1="10" y1="24" x2="10" y2="106" stroke="#c9dcff" />
  <polyline points="{points}" fill="none" stroke="#2878ff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
  <circle cx="290" cy="{last_y}" r="4" fill="#ffd34e" stroke="#17213a" stroke-width="2" />
</svg>""".format(
        label=_text(module.get("title") or "line chart"),
        points=" ".join(coords),
        last_y=coords[-1].split(",")[1],
    )


def render_table(module: Mapping[str, Any]) -> str:
    headers = module.get("headers") or []
    rows = module.get("rows") or []
    head_html = "".join("<th>%s</th>" % _text(_short_label(h)) for h in headers)
    body_rows = []
    for row in rows:
        cells = []
        for c in row:
            cells.append(
                '<td title="{full}">{short}</td>'.format(
                    full=_text(public_text(c)),
                    short=_text(_short(c, 46)),
                )
            )
        body_rows.append("<tr>%s</tr>" % "".join(cells))
    return "<table><tr>%s</tr>%s</table>" % (head_html, "".join(body_rows))


def render_checklist(module: Mapping[str, Any]) -> str:
    items = module.get("items") or []
    return "<ul>%s</ul>" % "".join("<li>%s</li>" % _text(_short(item, 76)) for item in items)


def render_status_list(module: Mapping[str, Any]) -> str:
    items = []
    for item in module.get("items") or []:
        items.append(
            '<div class="statusItem%s" title="%s"><b>%s</b><br>%s</div>'
            % (
                _summary_class(item.get("name")),
                _text(public_text(item.get("status"))),
                _text(_short_label(item.get("name"))),
                _text(_short(item.get("status"), 58)),
            )
        )
    return '<div class="statusList">%s</div>' % "".join(items)


def render_module(module: Mapping[str, Any]) -> str:
    kind = str(module.get("type") or "note")
    title_raw = module.get("title")
    subtitle_raw = module.get("subtitle")
    title = _text(_short_label(title_raw))
    subtitle = _text(_short(subtitle_raw, 86))
    if kind == "score_table":
        body = render_score_table(module)
    elif kind == "heatmap":
        body = render_heatmap(module)
    elif kind == "bar_chart":
        body = render_bar_chart(module)
    elif kind == "technical_analysis":
        body = render_technical_analysis(module)
    elif kind == "sentiment_shift":
        body = render_sentiment_shift(module)
    elif kind == "vix_analysis":
        body = render_vix_analysis(module)
    elif kind == "canary_board":
        body = render_canary_board(module)
    elif kind == "breadth_analysis":
        body = render_breadth_analysis(module)
    elif kind == "data_freshness":
        body = render_data_freshness(module)
    elif kind == "line_chart":
        body = render_line_chart(module)
    elif kind == "table":
        body = render_table(module)
    elif kind == "checklist":
        body = render_checklist(module)
    elif kind == "status_list":
        body = render_status_list(module)
    else:
        body = '<p title="%s">%s</p>' % (_text(public_text(module.get("body"))), _text(_short(module.get("body"), 180)))
    sub_html = '<p class="moduleSub">%s</p>' % subtitle if subtitle else ""
    module_info = module.get("info") or subtitle_raw or KIND_INFO.get(kind)
    tag_html = '<span class="tag">%s</span>' % _text(_short(module.get("tag"), 12)) if module.get("tag") else ""
    return """
<div class="module">
  <div class="moduleHead"><h2>{title}</h2><div style="display:flex;gap:5px;align-items:center">{tag}{info}</div></div>
  {subtitle}
  {body}
</div>""".format(
        title=title,
        tag=tag_html,
        info=info_icon(module_info),
        subtitle=sub_html,
        body=body,
    )


def render_modules(modules: Iterable[Mapping[str, Any]]) -> str:
    return "".join(render_module(module) for module in modules)


PHASES = (
    ("drivers", "구동축", "Score · Heat · Canary"),
    ("assets", "자산별", "QQQ · IWM · BTC"),
    ("market", "가격·VIX", "Price · Vol · Breadth"),
    ("derivatives", "옵션·선물", "Gamma · P/C · MaxPain · Futures"),
    ("sentiment", "뉴스·커뮤", "News · Cafe · Reddit"),
    ("audit", "감사", "Source · Fresh · Limits"),
)


PHASE_INFO = {
    "drivers": "시장 온도가 오르는지 내리는지 가장 먼저 보는 요약판입니다.",
    "assets": "QQQ, IWM, BTC를 따로 보며 서로 같은 방향인지 확인합니다.",
    "market": "가격, VIX, 시장 폭, 방어자산이 같은 신호를 내는지 점검합니다.",
    "derivatives": "옵션, 감마, 풋콜, 맥스페인, 선물 흐름을 묶어 과열과 방어를 봅니다.",
    "sentiment": "뉴스와 커뮤니티가 낙관, 공포, 관망 중 어디에 가까운지 봅니다.",
    "audit": "데이터가 실제 확인됐는지, 제한된 축은 무엇인지 남깁니다.",
}


def infer_phase(module: Mapping[str, Any]) -> str:
    explicit = str(module.get("phase") or "").strip()
    if explicit:
        return explicit
    title = str(module.get("title") or "")
    kind = str(module.get("type") or "")
    tag = str(module.get("tag") or "")
    haystack = f"{title} {kind} {tag}".lower()
    if any(word in haystack for word in ("score", "heat", "카나리아", "구성 점수", "히트맵")):
        return "drivers"
    if any(word in haystack for word in ("asset", "개별", "자산별")):
        return "assets"
    if any(word in haystack for word in ("vix", "breadth", "시장 폭", "크레딧", "technical", "리더", "방어")):
        return "market"
    if any(word in haystack for word in ("option", "gamma", "max pain", "put/call", "barchart", "선물", "옵션", "감마")):
        return "derivatives"
    if any(word in haystack for word in ("sentiment", "community", "커뮤니티", "뉴스", "기관", "event")):
        return "sentiment"
    if any(word in haystack for word in ("source", "fresh", "데이터", "limits", "미사용", "감사")):
        return "audit"
    return "market"


def render_phase_section(
    phase_id: str,
    number: int,
    title: str,
    subtitle: str,
    modules: Sequence[Mapping[str, Any]],
    extra_html: str = "",
) -> str:
    body = extra_html + render_modules(modules)
    if not body:
        return ""
    return """
<section class="phase phase-{phase_id}">
  <div class="phaseHead">
    <div class="phaseNo">{number}</div>
    <div><h2>{title}</h2><p>{subtitle}</p></div>
    {info}
  </div>
  <div class="phaseBody">{body}</div>
</section>""".format(
        phase_id=_text(phase_id),
        number=number,
        title=_text(_short_label(title)),
        subtitle=_text(subtitle),
        info=info_icon(PHASE_INFO.get(phase_id, subtitle)),
        body=body,
    )


def render_phases(report: Mapping[str, Any]) -> str:
    modules = list(report.get("modules") or [])
    grouped = {phase_id: [] for phase_id, _, _ in PHASES}
    for module in modules:
        if isinstance(module, Mapping):
            grouped.setdefault(infer_phase(module), []).append(module)
    html = []
    for idx, (phase_id, title, subtitle) in enumerate(PHASES, start=1):
        extra = render_asset_cards(report.get("assets") or []) if phase_id == "assets" else ""
        html.append(render_phase_section(phase_id, idx, title, subtitle, grouped.get(phase_id, []), extra))
    return "".join(html)


def render_thermometer_html(report: Mapping[str, Any], *, template_path: Path | None = None) -> str:
    template = (template_path or DEFAULT_TEMPLATE_PATH).read_text(encoding="utf-8")
    overall_score = _num(report.get("overall_score"))
    replacements = {
        "{{TITLE}}": _text(report.get("title") or "온도계 리포트"),
        "{{EYEBROW}}": _text(report.get("eyebrow") or "THERMOMETER"),
        "{{HEADING}}": _text(report.get("heading") or report.get("title") or "온도계 리포트"),
        "{{META}}": str(report.get("meta_html") or _text(report.get("meta") or "")),
        "{{CONFIDENCE}}": _text(report.get("confidence") or "Medium confidence"),
        "{{VERDICT_LABEL}}": _text(report.get("verdict_label") or "Neutral"),
        "{{OVERALL_SCORE}}": "%.0f" % overall_score,
        "{{OVERALL_COLOR}}": tone_color(overall_score),
        "{{HERO_TAGS}}": render_tags(report.get("hero_tags") or []),
        "{{SUMMARY}}": _text(public_text(report.get("summary") or "")),
        "{{SUMMARY_TONE}}": _text(report.get("summary_tone") or status_tone(report.get("verdict_label"))),
        "{{DECISION_GRID}}": render_decision_grid(report),
        "{{AI_DIAGNOSIS}}": render_ai_diagnosis(report),
        "{{PHASES}}": render_phases(report),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def write_thermometer_html(report: Mapping[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_thermometer_html(report), encoding="utf-8")
    return output_path


def write_thermometer_archive(report: Mapping[str, Any], filename: str) -> Mapping[str, Path]:
    """Write one thermometer report to the local cache and iCloud archive."""
    local_path = LOCAL_THERMOMETER_DIR / filename
    icloud_path = ICLOUD_THERMOMETER_DIR / filename
    write_thermometer_html(report, local_path)
    write_thermometer_html(report, icloud_path)
    return {"local": local_path, "icloud": icloud_path}
