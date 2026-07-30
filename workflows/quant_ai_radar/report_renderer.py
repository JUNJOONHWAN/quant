"""Offline, dependency-free HTML/JSON output for Quant AI Radar."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_support import build_market_dashboard, build_security_brief
from .presentation import (
    bar_width,
    confidence_pct,
    label_market_state,
    label_regime,
    label_rotation_state,
    label_signal,
    label_task,
    whole,
)
from .report_narratives import validate_report_narratives


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _list_items(values: Sequence[Any]) -> str:
    return "".join(f"<li>{_e(value)}</li>" for value in values) or "<li>없음</li>"


def _market_evidence_item(
    row: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> str:
    evidence_id = str(row.get("evidence_id") or "")
    evidence = catalog.get(evidence_id)
    if evidence is None:
        return f"<li><code>{_e(evidence_id)}</code> · 근거 데이터 없음</li>"
    return (
        f"<li><code>{_e(evidence_id)}</code>"
        f"<pre>{_e(_canonical(evidence))}</pre></li>"
    )


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title>
<style>
:root{{--bg:#0b1020;--panel:#131b31;--line:#26324f;--text:#eef3ff;
--muted:#9ba9c8;--accent:#72e0bd;--warn:#ffcc72;--bad:#ff8e9d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Noto Sans KR",sans-serif}}
main{{width:100%;max-width:420px;margin:0 auto;padding:14px}} a{{color:var(--accent)}}
h1,h2,h3{{line-height:1.2}} .meta,.muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:minmax(0,1fr);gap:12px}}
.card{{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}}
.metric{{font-size:24px;font-weight:750}} .pill{{display:inline-block;padding:3px 9px;
border:1px solid var(--line);border-radius:999px;margin:2px;color:var(--accent)}}
table{{width:100%;table-layout:fixed;border-collapse:collapse}} th,td{{padding:7px;
border-bottom:1px solid var(--line);text-align:left;vertical-align:top;
overflow-wrap:anywhere}} th{{color:var(--muted)}}
input{{width:100%;padding:12px;background:#0d1528;color:var(--text);border:1px solid
var(--line);border-radius:10px;margin:10px 0 16px}} pre{{white-space:pre-wrap;
word-break:break-word;background:#09101e;border:1px solid var(--line);padding:12px;
border-radius:10px;max-height:520px;overflow:auto}}
.positive{{color:var(--accent)}} .negative{{color:var(--bad)}}
.bar{{height:9px;background:#09101e;border-radius:999px;overflow:hidden;margin:6px 0}}
.bar>span{{display:block;height:100%;background:linear-gradient(90deg,#72e0bd,#7aa7ff);
border-radius:999px}} .callout{{border-left:3px solid var(--accent);padding-left:12px}}
@media(max-width:420px){{main{{padding:10px}}th:nth-child(n+4),td:nth-child(n+4){{display:none}}}}
</style>
</head><body><main>{body}</main></body></html>
"""


def _security_html(
    result: Mapping[str, Any],
    coverage: Mapping[str, Any],
    as_of_date: str,
    navigation_html: str | None = None,
    narrative: Mapping[str, Any] | None = None,
) -> str:
    judgement = result["judgement"]
    interpretation = judgement.get("interpretation") or {}
    facts = judgement.get("facts") or {}
    brief = build_security_brief(judgement)
    narrative = narrative or {}
    confirmation_rows = "".join(
        "<li><strong>{}</strong> · {} <span class=\"muted\">({})</span></li>".format(
            _e(row.get("label")),
            _e(row.get("interpretation")),
            _e(row.get("value")),
        )
        for row in brief["confirmations"]
    )
    contradiction_rows = "".join(
        "<li><strong>{}</strong> · {} <span class=\"muted\">({})</span></li>".format(
            _e(row.get("label")),
            _e(row.get("interpretation")),
            _e(row.get("value")),
        )
        for row in brief["contradictions"]
    )
    contributor_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _e(row.get("etf_ticker")),
            _e(row.get("weighted_flow_rate_contribution_pct")),
            _e(row.get("membership_weight_percent")),
            _e(row.get("flow_training_available_session_date")),
        )
        for row in brief["flow"]["top_contributing_etfs"]
    )
    if navigation_html is None:
        navigation_html = (
            '<p><a href="../security_index.html">← 종목 목록</a> · '
            '<a href="../market_report.html">시장 리포트</a></p>'
        )
    body = f"""
{navigation_html}
<h1>{_e(result["symbol"])}</h1>
<p class="meta">기준일 {_e(as_of_date)} · {_e(label_task(result["task_type"]))} ·
분석 전용, 주문 연결 없음</p>
<div class="grid">
 <section class="card"><div class="muted">가격·ETF 자금 국면</div>
  <div class="metric">{_e(label_regime(judgement.get("regime")))}</div></section>
 <section class="card"><div class="muted">AI 판단 신뢰도</div>
  <div class="metric">{_e(confidence_pct(judgement.get("confidence")))}</div>
  <div class="bar"><span style="width:{bar_width((judgement.get('confidence') or 0) * 100)}%"></span></div></section>
 <section class="card"><div class="muted">가격 / ETF 자금</div>
  <div class="metric">{_e(label_signal(interpretation.get("price_signal")))} /
  {_e(label_signal(interpretation.get("etf_flow_signal")))}</div></section>
 <section class="card"><div class="muted">분석 우선순위</div>
  <div class="metric">{_e(whole(coverage.get("priority_score")))}</div>
  <div>{''.join(f'<span class="pill">{_e(x)}</span>' for x in coverage.get('selection_reasons') or [])}</div>
 </section>
 <section class="card"><div class="muted">근거 강도</div>
  <div class="metric">{_e(whole(brief["data_quality"]["evidence_strength_score"], suffix="/100"))}</div>
  <div>{_e(brief["data_quality"]["status"])}</div></section>
</div>
<section class="card"><h2>근거 기반 결론</h2><p>{_e(brief["conclusion"])}</p></section>
<div class="grid">
 <section class="card"><h2>가격 구조</h2><p>{_e(brief["price"]["summary"])}</p></section>
 <section class="card"><h2>ETF Flow 전달</h2><p>{_e(brief["flow"]["summary"])}</p></section>
 <section class="card"><h2>가격–Flow 관계</h2><p>{_e(brief["relationship"]["summary"])}</p></section>
</div>
<section class="card"><h2>학습 모델 해석</h2>
 <p>{_e(judgement.get("conclusion"))}</p>
</section>
{f'''<section class="card callout"><h2>전체 시장 속 이 종목</h2>
 <h3>{_e(narrative.get("headline"))}</h3>
 <p><strong>소속 흐름:</strong> {_e(narrative.get("group_context"))}</p>
 <p><strong>ETF 전달 경로:</strong> {_e(narrative.get("etf_transmission"))}</p>
 <p><strong>반대 근거:</strong> {_e(narrative.get("counterpoint"))}</p>
 <p><strong>다음 확인:</strong> {_e(narrative.get("watch_condition"))}</p>
</section>''' if narrative else ""}
<div class="grid">
 <section class="card"><h2>확인 증거</h2><ul>{confirmation_rows or '<li>없음</li>'}</ul></section>
 <section class="card"><h2>반대 증거</h2><ul>{contradiction_rows or '<li>없음</li>'}</ul></section>
 <section class="card"><h2>미확인 사항</h2><ul>{_list_items(brief["unknowns"])}</ul></section>
</div>
<section class="card"><h2>상위 ETF→종목 기여 경로</h2>
 <table><thead><tr><th>ETF</th><th>가중 Flow 기여도 %</th>
 <th>보유 비중 %</th><th>학습 가시일</th></tr></thead>
 <tbody>{contributor_rows or '<tr><td colspan="4">ETF 자체 Flow 분석 또는 기여 경로 없음</td></tr>'}</tbody></table>
</section>
<section class="card"><h2>다음 확인 조건</h2><ul>{_list_items(brief["confirmation_conditions"])}</ul></section>
<details class="card"><summary>결정론적 원본 사실</summary><pre>{_e(_canonical(facts))}</pre></details>
"""
    return _page(f"{result['symbol']} Quant AI Radar", body)


def render_single_security_html(
    path: Path,
    *,
    result: Mapping[str, Any],
    as_of_date: str,
    coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render one requested security without requiring a daily market report."""

    resolved = Path(path).expanduser().resolve()
    _atomic_text(
        resolved,
        _security_html(
            result,
            coverage or {},
            as_of_date,
            navigation_html="",
        ),
    )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def render_reports(
    *,
    run_dir: Path,
    report: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    coverage_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write market and per-security reports, then hash every artifact."""

    validate_report_narratives(report)
    root = Path(run_dir).expanduser().resolve()
    coverage = {str(row["symbol"]): row for row in coverage_ledger}
    narratives = report.get("multistage_narratives") or {}
    security_narratives = {
        str(row.get("symbol") or ""): row
        for row in narratives.get("security_explanations") or []
    }
    security_dir = root / "security_reports"
    rows = []
    artifact_paths = []
    for result in sorted(results, key=lambda item: str(item["symbol"])):
        symbol = str(result["symbol"])
        value = {
            "schema_version": "quant.ai_radar_security_report.v1",
            "as_of_date": report["as_of_date"],
            "scope": "data_interpretation_not_trade_execution",
            "selection": coverage.get(symbol, {}),
            **dict(result),
        }
        json_path = security_dir / f"{symbol}.json"
        html_path = security_dir / f"{symbol}.html"
        _atomic_text(json_path, _canonical(value))
        _atomic_text(
            html_path,
            _security_html(
                result,
                coverage.get(symbol, {}),
                str(report["as_of_date"]),
                narrative=security_narratives.get(symbol),
            ),
        )
        artifact_paths.extend((json_path, html_path))
        judgement = result["judgement"]
        interpretation = judgement.get("interpretation") or {}
        rows.append(
            {
                "symbol": symbol,
                "task_type": label_task(result["task_type"]),
                "regime": label_regime(judgement.get("regime")),
                "confidence": confidence_pct(judgement.get("confidence")),
                "relationship": label_regime(interpretation.get("relationship")),
                "priority_score": coverage.get(symbol, {}).get("priority_score"),
                "path": f"security_reports/{symbol}.html",
            }
        )

    table_rows = "".join(
        f"""<tr data-search="{_e(' '.join(str(v) for v in row.values()))}">
<td><a href="{_e(row['path'])}">{_e(row['symbol'])}</a></td>
<td>{_e(row['task_type'])}</td><td>{_e(row['regime'])}</td>
<td>{_e(row['relationship'])}</td><td>{_e(row['confidence'])}</td>
<td>{_e(row['priority_score'])}</td></tr>"""
        for row in rows
    )
    index_body = f"""
<p><a href="market_report.html">← 시장 리포트</a></p>
<h1>선택 ETF·종목 상세 분석</h1>
<p class="meta">기준일 {_e(report["as_of_date"])} · {len(rows)}건 ·
전체 정량 스캔 후 동적 선택</p>
<input id="q" placeholder="ticker, regime, 관계 검색">
<div class="card"><table id="t"><thead><tr><th>Symbol</th><th>Task</th>
<th>Regime</th><th>관계</th><th>신뢰도</th><th>우선순위</th></tr></thead>
<tbody>{table_rows}</tbody></table></div>
<script>
const q=document.getElementById('q');q.addEventListener('input',()=>{{
 const v=q.value.toLowerCase();document.querySelectorAll('#t tbody tr').forEach(r=>{{
  r.style.display=r.dataset.search.toLowerCase().includes(v)?'':'none';
 }});
}});
</script>"""
    security_index = root / "security_index.html"
    _atomic_text(security_index, _page("Quant AI Radar 종목 분석", index_body))
    artifact_paths.append(security_index)

    market = report["market_judgement"]
    aggregate = report["aggregate"]
    selection = report["selection"]
    dashboard = report.get("market_dashboard") or build_market_dashboard(
        aggregate,
        report.get("oracle_market") or {},
    )
    quality = report.get("quality_audit") or {}
    editorial = narratives.get("editorial") or {}
    strict_narratives = report.get("schema_version") == "quant.ai_radar_report.v2"
    editorial_headline = (
        editorial.get("headline")
        if strict_narratives
        else editorial.get("headline") or "오늘의 시장 구조"
    )
    editorial_summary = (
        editorial.get("executive_summary")
        if strict_narratives
        else editorial.get("executive_summary") or market.get("summary")
    )
    editorial_rotation = (
        editorial.get("rotation_summary")
        if strict_narratives
        else editorial.get("rotation_summary") or market.get("summary")
    )
    editorial_selection = (
        editorial.get("selection_summary")
        if strict_narratives
        else editorial.get("selection_summary")
        or "가격과 ETF 자금이 함께 확인된 후보를 분리해 봅니다."
    )
    editorial_risk = (
        editorial.get("risk_summary")
        if strict_narratives
        else editorial.get("risk_summary")
        or "괴리와 반대 근거를 함께 확인해야 합니다."
    )
    sector_narratives = {
        str(row.get("cluster") or ""): row
        for row in narratives.get("sector_explanations") or []
    }
    evidence_catalog = report.get("market_evidence_catalog") or {}
    confirmations = "".join(
        _market_evidence_item(row, evidence_catalog)
        for row in market.get("confirmations") or []
    )
    contradictions = "".join(
        _market_evidence_item(row, evidence_catalog)
        for row in market.get("contradictions") or []
    )
    rotation_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _e(row.get("cluster")),
            _e(label_rotation_state(row.get("state"))),
            _e(whole(row.get("score"))),
            _e(whole(row.get("breadth_score"), suffix="%")),
            _e(whole(row.get("median_return_5d_pct"), signed=True, suffix="%")),
        )
        for row in dashboard.get("rotation_clusters") or []
    )
    etf_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _e(row.get("symbol")),
            _e(label_regime(row.get("regime"))),
            _e(whole(row.get("latest_robust_zscore"), signed=True)),
            _e(row.get("latest_effective_date")),
        )
        for row in dashboard.get("leading_etfs") or []
    )
    stock_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _e(row.get("symbol")),
            _e(label_regime(row.get("regime"))),
            _e(whole(row.get("net_weighted_flow_rate_contribution_pct"), signed=True, suffix="%")),
            _e(whole(row.get("eligible_etf_count"))),
        )
        for row in dashboard.get("affected_stocks") or []
    )
    candidate_lanes = dashboard.get("candidate_lanes") or {}

    def candidate_rows(key: str) -> str:
        return "".join(
            "<tr><td><a href=\"security_reports/{}.html\">{}</a></td>"
            "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                _e(row.get("symbol")),
                _e(row.get("symbol")),
                _e(label_regime(row.get("regime"))),
                _e(confidence_pct(row.get("confidence"))),
                _e(
                    whole(row.get("latest_robust_zscore"), signed=True)
                    if row.get("latest_robust_zscore") is not None
                    else whole(
                        row.get("net_weighted_flow_rate_contribution_pct"),
                        signed=True,
                        suffix="%",
                    )
                ),
            )
            for row in candidate_lanes.get(key) or []
        )

    positive_candidates = candidate_rows(
        "positive_confirmation_etfs"
    ) + candidate_rows("positive_confirmation_stocks")
    negative_candidates = candidate_rows(
        "negative_confirmation_etfs"
    ) + candidate_rows("negative_confirmation_stocks")
    divergence_candidates = candidate_rows("divergence_etfs") + candidate_rows(
        "divergence_stocks"
    )
    score_pills = "".join(
        f'<span class="pill">{_e(name)} {_e(score)}/10</span>'
        for name, score in (quality.get("scores") or {}).items()
    )
    breadth = dashboard.get("breadth") or {}
    completed_ai_count = quality.get(
        "security_report_count",
        selection.get("selected_count"),
    )
    sector_explanation_cards = "".join(
        f"""<section class="card"><h3>{_e(row.get("cluster"))} ·
{_e(label_rotation_state(next((item.get("state") for item in dashboard.get("rotation_clusters") or [] if item.get("cluster") == row.get("cluster")), "")))}</h3>
<p><strong>{_e(row.get("headline"))}</strong></p>
<p>{_e(row.get("explanation"))}</p>
<p class="muted">관련 종목: {_e(row.get("stock_context"))}</p>
<p class="muted">반대 근거: {_e(row.get("counterpoint"))}</p></section>"""
        for row in narratives.get("sector_explanations") or []
    )
    market_body = f"""
<h1>Quant AI Radar</h1>
<p class="meta">기준일 {_e(report["as_of_date"])} · 생성 {_e(report["generated_at_kst"])}
· 참고용 분석 · 실주문 미연결</p>
<div class="grid">
 <section class="card"><div class="muted">시장 국면</div>
  <div class="metric">{_e(label_market_state(market.get("market_state")))}</div></section>
 <section class="card"><div class="muted">AI 판단 신뢰도</div>
  <div class="metric">{_e(confidence_pct(market.get("confidence")))}</div>
  <div class="bar"><span style="width:{bar_width((market.get('confidence') or 0) * 100)}%"></span></div></section>
 <section class="card"><div class="muted">전체 ETF 관련 후보</div>
  <div class="metric">{_e(selection.get("full_candidate_count"))}</div></section>
 <section class="card"><div class="muted">AI 상세 분석</div>
  <div class="metric">{_e(completed_ai_count)}</div></section>
</div>
<section class="card callout"><h2>{_e(editorial_headline)}</h2>
<p>{_e(editorial_summary)}</p>
<p><strong>회전:</strong> {_e(editorial_rotation)}</p>
<p><strong>후보군:</strong> {_e(editorial_selection)}</p>
<p><strong>위험:</strong> {_e(editorial_risk)}</p>
<p><a href="security_index.html">ETF·종목 상세 분석 보기 →</a></p></section>
<section class="card"><h2>정확 집계 기반 시장 구조</h2>
 <p>{_e(dashboard.get("interpretation"))}</p>
 <div class="grid">
  <div><div class="muted">가격 강세 비중</div><div class="metric">{_e(whole(breadth.get("price_positive_pct"), suffix="%"))}</div>
  <div class="bar"><span style="width:{bar_width(breadth.get('price_positive_pct'))}%"></span></div></div>
  <div><div class="muted">ETF 자금 강세 비중</div><div class="metric">{_e(whole(breadth.get("etf_flow_positive_pct"), suffix="%"))}</div>
  <div class="bar"><span style="width:{bar_width(breadth.get('etf_flow_positive_pct'))}%"></span></div></div>
  <div><div class="muted">동시 확인 국면</div><div class="metric">{_e(whole(breadth.get("confirmation_count")))}</div></div>
  <div><div class="muted">가격·자금 괴리</div><div class="metric">{_e(whole(breadth.get("divergence_count")))}</div></div>
 </div>
</section>
<section class="card"><h2>AI Radar 판단 후보군</h2>
 <p class="muted">가격과 ETF Flow가 함께 확인된 관찰 후보와 약세 위험 후보를
 분리합니다. 매수·매도 주문 신호가 아니며, 괴리 후보는 방향 확정 전에 추가
 확인이 필요합니다.</p>
 <div class="grid">
  <div><h3>강세 확인 관찰</h3><table><thead><tr><th>종목</th><th>판정</th><th>신뢰도</th><th>자금</th></tr></thead>
   <tbody>{positive_candidates or '<tr><td colspan="4">동시 확인 후보 없음</td></tr>'}</tbody></table></div>
  <div><h3>약세 확인 위험</h3><table><thead><tr><th>종목</th><th>판정</th><th>신뢰도</th><th>자금</th></tr></thead>
   <tbody>{negative_candidates or '<tr><td colspan="4">동시 약세 후보 없음</td></tr>'}</tbody></table></div>
  <div><h3>가격–ETF 자금 괴리</h3><table><thead><tr><th>종목</th><th>판정</th><th>신뢰도</th><th>자금</th></tr></thead>
   <tbody>{divergence_candidates or '<tr><td colspan="4">괴리 후보 없음</td></tr>'}</tbody></table></div>
 </div>
</section>
<div class="grid">
 <section class="card"><h2>확인 증거</h2><ul>{confirmations or '<li>없음</li>'}</ul></section>
 <section class="card"><h2>반대 증거</h2><ul>{contradictions or '<li>없음</li>'}</ul></section>
 <section class="card"><h2>미확인 사항</h2><ul>{_list_items(market.get("unknowns") or [])}</ul></section>
</div>
<section class="card"><h2>섹터·테마 회전 수치</h2>
 <table><thead><tr><th>분류</th><th>상태</th><th>강도</th><th>확산</th><th>최근</th></tr></thead>
 <tbody>{rotation_rows or '<tr><td colspan="5">확인 가능한 회전 cluster 없음</td></tr>'}</tbody></table>
</section>
<h2>섹터별 AI 해설</h2>
<div class="grid">{sector_explanation_cards}</div>
<div class="grid">
 <section class="card"><h2>주요 ETF</h2><table><thead><tr><th>ETF</th><th>판정</th><th>자금 이상도</th><th>유효일</th></tr></thead><tbody>{etf_rows}</tbody></table></section>
 <section class="card"><h2>영향 종목</h2><table><thead><tr><th>종목</th><th>판정</th><th>가중 자금</th><th>ETF 수</th></tr></thead><tbody>{stock_rows}</tbody></table></section>
</div>
<section class="card"><h2>품질 게이트</h2><p>{score_pills or '실행 전'}</p>
 <p class="muted">모든 항목 8.0/10 이상일 때만 reference publish 가능</p></section>
<details class="card"><summary>전체 집계 원본</summary><pre>{_e(_canonical(aggregate))}</pre></details>
"""
    market_html = root / "market_report.html"
    _atomic_text(market_html, _page("Quant AI Radar 시장 리포트", market_body))
    artifact_paths.append(market_html)

    manifest_core = {
        "schema_version": "quant.ai_radar_rendered_reports.v1",
        "as_of_date": report["as_of_date"],
        "market_report_html": str(market_html),
        "security_index_html": str(security_index),
        "security_report_count": len(rows),
        "artifacts": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(artifact_paths)
        ],
    }
    manifest_core["content_sha256"] = hashlib.sha256(
        json.dumps(
            manifest_core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = root / "rendered_reports_manifest.json"
    _atomic_text(manifest_path, _canonical(manifest_core))
    return {**manifest_core, "manifest_path": str(manifest_path)}
