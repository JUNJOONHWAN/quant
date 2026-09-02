"""Render the Forecast v2 walk-forward evidence and current forecasts to HTML."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .io_utils import sha256_file, utc_now, write_text_atomic


DEFAULT_EVALUATION = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/models/"
    "walk_forward_evaluation.json"
)
DEFAULT_PANEL_MANIFEST = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/panel_manifest.json"
)
DEFAULT_PRODUCTION_MANIFEST = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/models/"
    "production_manifest.json"
)
DEFAULT_ACCESS_PROBE = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/evidence/"
    "market_access_probe_20260827.json"
)
DEFAULT_OVERFIT_AUDIT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/v2/evidence/"
    "nested_walk_forward_overfit_audit_20260827.json"
)
DEFAULT_OUTPUT = Path(
    "/home/zooh/Documents/DGX_Outputs/STOCK/리포트/AI_RADAR_FORECAST_V2/"
    "AI_RADAR_Forecast_v2_워크포워드_평가.html"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _number(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _percent(value: object, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):,.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _status_class(value: str) -> str:
    return "good" if value in {"PASS", "USEFUL", "FULL"} else "warn"


def _variant_label(value: str) -> str:
    return {
        "price": "가격만",
        "price_benchmark_flow_t3": "가격 + SPY/QQQ Flow (T-3 대조)",
        "price_benchmark_flow": "가격 + SPY/QQQ Flow (T-2)",
        "price_all_etf_flow": "가격 + 전체 ETF Flow (T-2)",
        "full": "전체 ETF Flow + FMP 펀더멘털",
    }.get(value, value)


def _comparison_rows(report: Mapping[str, Any], horizon: int) -> str:
    rows = []
    labels = {
        "t2_vs_price": "T-2 벤치마크 Flow vs 가격",
        "t2_vs_t3": "T-2 vs 기존 T-3",
        "all_etf_vs_benchmark": "전체 ETF Flow vs SPY/QQQ Flow",
        "fundamentals_vs_all_etf": "FMP 펀더멘털 추가",
    }
    for key, item in report["comparisons"][str(horizon)].items():
        delta = item["deltas_candidate_minus_baseline"]
        verdict = item["verdict"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels[key])}</td>"
            f"<td class='{_status_class(verdict)}'>{html.escape(verdict)}</td>"
            f"<td>{_number(delta['daily_return_ic'], 4)}</td>"
            f"<td>{_number(delta['top_decile_net_return_pct'])}</td>"
            f"<td>{_number(delta['return_mae_pct'])}</td>"
            f"<td>{_number(delta['loss_mae_pct'])}</td>"
            f"<td>{item['favorable_metric_count_of_5']}/5</td>"
            "</tr>"
        )
    return "".join(rows)


def _ablation_rows(report: Mapping[str, Any], horizon: int) -> str:
    rows = []
    selected = report["selected_models"][str(horizon)]["variant"]
    for variant, horizons in report["ablation_results"].items():
        aggregate = horizons[str(horizon)]["aggregate"]
        ret = aggregate["targets"]["return"]
        loss = aggregate["targets"]["loss"]
        combined = aggregate["combined"]
        rows.append(
            "<tr>"
            f"<td>{'★ ' if variant == selected else ''}{html.escape(_variant_label(variant))}</td>"
            f"<td>{aggregate['fold_count']}</td>"
            f"<td>{aggregate['test_rows']:,}</td>"
            f"<td>{_number(ret.get('mae_pct'))}</td>"
            f"<td>{_number(ret.get('daily_spearman_ic', {}).get('mean'), 4)}</td>"
            f"<td>{_percent(ret.get('directional_accuracy'), 1)}</td>"
            f"<td>{_number(loss.get('mae_pct'))}</td>"
            f"<td>{_number(combined.get('top_decile_net_return_pct'))}</td>"
            f"<td>{_number(combined.get('top_minus_bottom_return_pct'))}</td>"
            f"<td>{_number(combined.get('loss_filter_loss_reduction_pct'))}</td>"
            f"<td>{_number(aggregate.get('selection_score'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _point_metric_rows(
    evaluation: Mapping[str, Any], production: Mapping[str, Any]
) -> str:
    rows = []
    target_labels = {
        "return": "종가수익",
        "upside": "상승여력",
        "loss": "손실폭",
    }
    for horizon in (5, 20):
        selections = production["production_models"][str(horizon)]["point_selection"]
        for target in ("return", "upside", "loss"):
            variant = selections[target]["variant"]
            metrics = evaluation["ablation_results"][variant][str(horizon)][
                "aggregate"
            ]["targets"][target]
            interval = metrics["prediction_interval"]
            rows.append(
                "<tr>"
                f"<td>{horizon}일</td>"
                f"<td>{target_labels[target]}</td>"
                f"<td>{html.escape(_variant_label(variant))}</td>"
                f"<td>{_number(metrics['mae_pct'])}</td>"
                f"<td>{_number(metrics['rmse_pct'])}</td>"
                f"<td>{_number(metrics['r2'], 4)}</td>"
                f"<td>{_number(metrics['daily_spearman_ic']['mean'], 4)}</td>"
                f"<td>{_percent(interval['empirical_coverage'], 1)}</td>"
                f"<td>{_number(interval['mean_width_pct'])}</td>"
                "</tr>"
            )
    return "".join(rows)


def _overfit_rows(audit: Mapping[str, Any]) -> str:
    rows = []
    for horizon in (5, 20):
        item = audit["horizons"][str(horizon)]
        nested = item["nested_prequential"]
        variant_path = " → ".join(
            f"{row['test_year']} {_variant_label(row['ranking_variant'])}"
            for row in nested
        )
        positive_spreads = sum(
            float(row["ranking_test"]["top_minus_bottom_return_pct"]) > 0
            for row in nested
        )
        positive_top = sum(
            float(row["ranking_test"]["top_decile_net_return_pct"]) > 0
            for row in nested
        )
        verdict = (
            "조건부 통과·shadow 필요"
            if len({row["ranking_variant"] for row in nested}) == 1
            and positive_spreads == len(nested)
            else "연구용·선택 불안정"
        )
        rows.append(
            "<tr>"
            f"<td>{horizon}일</td>"
            f"<td>{html.escape(_variant_label(item['aggregate_selected_ranking_variant']))}</td>"
            f"<td>{html.escape(variant_path)}</td>"
            f"<td>{positive_top}/{len(nested)}</td>"
            f"<td>{positive_spreads}/{len(nested)}</td>"
            f"<td class='{_status_class('PASS' if verdict.startswith('조건부') else 'WARN')}'>{verdict}</td>"
            "</tr>"
        )
    return "".join(rows)


def _forecast_rows(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['symbol']))}</td>"
            f"<td>{html.escape(str(row['benchmark']))}</td>"
            f"<td>{_number(row['reference_close'], 2)}</td>"
            f"<td>{_number(row['expected_return_5d_pct'])}</td>"
            f"<td>{_number(row['expected_upside_5d_pct'])}</td>"
            f"<td>{_number(row['expected_loss_5d_pct'])}</td>"
            f"<td>{_number(row['return_5d_p10_pct'])} ~ {_number(row['return_5d_p90_pct'])}</td>"
            f"<td>{_number(row['expected_return_20d_pct'])}</td>"
            f"<td>{_number(row['expected_upside_20d_pct'])}</td>"
            f"<td>{_number(row['expected_loss_20d_pct'])}</td>"
            f"<td>{_number(row['loss_20d_p90_pct'])}</td>"
            f"<td>{_number(row['ranking_score_20d'])}</td>"
            f"<td class='{_status_class(str(row['flow_quality']))}'>{html.escape(str(row['flow_quality']))}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_report(
    *,
    evaluation_path: Path,
    panel_manifest_path: Path,
    production_manifest_path: Path,
    access_probe_path: Path,
    overfit_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    evaluation = _load(evaluation_path)
    panel = _load(panel_manifest_path)
    production = _load(production_manifest_path)
    access_probe = _load(access_probe_path)
    overfit_audit = _load(overfit_audit_path)
    forecast_path = Path(production["forecast"]["csv_path"])
    forecasts = pd.read_csv(forecast_path)
    live_audit = panel["live_flow_capture_audit"]
    quality = panel["quality"]
    membership = panel["membership_validation"]
    point_selections = {
        str(horizon): production["production_models"][str(horizon)]["point_selection"]
        for horizon in (5, 20)
    }

    def selected_metrics(horizon: int, target: str) -> Mapping[str, Any]:
        variant = point_selections[str(horizon)][target]["variant"]
        return evaluation["ablation_results"][variant][str(horizon)]["aggregate"][
            "targets"
        ][target]

    five_return = selected_metrics(5, "return")
    twenty_return = selected_metrics(20, "return")
    five_upside = selected_metrics(5, "upside")
    five_loss = selected_metrics(5, "loss")
    twenty_upside = selected_metrics(20, "upside")
    twenty_loss = selected_metrics(20, "loss")
    generated = utc_now()
    html_text = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI RADAR Forecast v2 워크포워드 평가</title>
<style>
:root{{--bg:#07101f;--panel:#0f1c2e;--line:#27405f;--text:#e8f1fb;--muted:#9eb1c8;--blue:#5fb3ff;--green:#61d39b;--amber:#ffc76a;}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(160deg,#07101f,#0a1628 55%,#07101f);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}}
main{{max-width:1500px;margin:auto;padding:32px 24px 80px}} h1{{font-size:32px;margin:0 0 6px}} h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}} h3{{color:var(--blue)}} p,li{{color:var(--muted)}} .lead{{font-size:17px;color:var(--text)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:20px 0}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}} .card b{{display:block;font-size:22px;color:var(--blue)}}
.good{{color:var(--green);font-weight:700}} .warn{{color:var(--amber);font-weight:700}} code{{color:#b9d9ff}} a{{color:#7fc4ff}} .scroll{{overflow:auto;border:1px solid var(--line);border-radius:10px}}
table{{width:100%;border-collapse:collapse;white-space:nowrap;background:#0b1728}} th,td{{padding:9px 10px;border-bottom:1px solid #1e3550;text-align:right}} th{{position:sticky;top:0;background:#13243a;color:#b9d9ff}} td:first-child,th:first-child{{text-align:left}} .note{{border-left:3px solid var(--blue);padding:10px 14px;background:#0c1b30}}
</style></head><body><main>
<h1>AI RADAR Forecast v2</h1>
<p class="lead">SPY·QQQ 구성 종목의 5·20거래일 기대수익, 기대상승여력, 기대손실을 전체 ETF Flow와 FMP 시점일치 정보로 예측한 워크포워드 결과</p>
<div class="cards">
 <div class="card">신호일<b>{html.escape(str(evaluation['latest_forecast']['signal_date']))}</b></div>
 <div class="card">가격 컷오프<b>T-1 종가</b></div>
 <div class="card">ETF Flow<b>T-2 거래일</b></div>
 <div class="card">현재 예측 종목<b>{production['forecast']['row_count']:,}개</b></div>
 <div class="card">전체 패널<b>{evaluation['panel']['total_rows']:,}행</b></div>
 <div class="card">공통 평가 표본<b>{evaluation['panel']['common_rows']:,}행</b></div>
</div>
<div class="note"><b>확정된 운영 타이밍</b><br>{html.escape(evaluation['timing_contract'])}<br>
현재 수집 게이트: <span class="{_status_class(live_audit['status'])}">{html.escape(live_audit['status'])}</span>. T-2가 장전까지 없으면 예측을 오래된 T-3로 조용히 대체하지 않는다.</div>

<h2>핵심 판정</h2>
<ul>
 <li><b>독립 매수·매도 방향신호로는 부족하다.</b> 최종 종가수익 모델의 방향 정확도는 5일 {_percent(five_return['directional_accuracy'])}, 20일 {_percent(twenty_return['directional_accuracy'])}, R²는 각각 {_number(five_return['r2'], 4)}, {_number(twenty_return['r2'], 4)}다. 기대수익 숫자는 오차범위와 함께 참고해야 한다.</li>
 <li><b>주 용도는 상승여력·손실위험의 횡단면 순위와 회피 필터다.</b> 일별 순위 IC는 5일 상승 {_number(five_upside['daily_spearman_ic']['mean'], 3)} / 손실 {_number(five_loss['daily_spearman_ic']['mean'], 3)}, 20일 상승 {_number(twenty_upside['daily_spearman_ic']['mean'], 3)} / 손실 {_number(twenty_loss['daily_spearman_ic']['mean'], 3)}로 종가수익 IC보다 뚜렷하다.</li>
 <li><b>ETF Flow는 기간·목적별로 선별 사용한다.</b> 5일 SPY/QQQ T-2 Flow의 가격 대비 증분은 {html.escape(evaluation['comparisons']['5']['t2_vs_price']['verdict'])}이고 5일 바스켓 순위는 {html.escape(_variant_label(production['production_models']['5']['ranking_variant']))} 모델이다. 20일 T-2는 T-3보다 {html.escape(evaluation['comparisons']['20']['t2_vs_t3']['verdict'])}이며, 수치 기대수익·기대손실은 각각 {html.escape(_variant_label(point_selections['20']['return']['variant']))}, {html.escape(_variant_label(point_selections['20']['loss']['variant']))}, 바스켓 순위는 {html.escape(_variant_label(production['production_models']['20']['ranking_variant']))} 모델을 사용한다.</li>
</ul>

<h2>결론을 만드는 기준</h2>
<p>모든 모델은 같은 공통 표본에서 연도별 확장형 워크포워드로 비교했다. 학습·보정·테스트 경계에는 각 목표의 5/20거래일 purge를 넣었다. 아래 ★는 <b>바스켓 순위용</b> 테스트 결과로 선택된 모델이며, 수치 기대값 모델은 다음 역할별 표와 목표별 오차 표를 따른다. T-3는 과거 계약과의 타이밍 대조군일 뿐 운영 입력이 아니다.</p>

<h2>과적합 감사: 과거 OOS만으로 다음 해 선택</h2>
<p>2023년 OOS를 첫 선택자료로 삼고, 테스트 연도 Y의 모델은 Y 이전에 끝난 OOS 연도만으로 골라 2024~2026에 적용했다. 이는 현재 집계값을 보고 같은 기간 모델을 고르는 선택 편향을 줄이지만, 설계자가 이미 이 기간을 보았으므로 완전히 봉인된 미래 증명은 아니다.</p>
<div class="scroll"><table><thead><tr><th>기간</th><th>전체기간 집계 선택</th><th>과거 OOS만 쓴 다음해 선택 경로</th><th>상위10% 순수익 양수</th><th>상하위 스프레드 양수</th><th>판정</th></tr></thead><tbody>{_overfit_rows(overfit_audit)}</tbody></table></div>
<p class="note"><b>운영 게이트:</b> 5일 바스켓 순위는 선택 불안정으로 연구·shadow 전용이다. 20일도 종가수익 단독 신호가 아니라 상승여력·손실위험 순위와 회피 필터로만 조건부 사용한다. 최종 일반화 판정은 이 계약을 동결한 뒤 미래 데이터에서 재튜닝 없이 평가해야 한다.</p>

<h2>최종 역할별 모델</h2>
<p>수치 기대값은 목표별 MAE와 RMSE가 가장 안정적인 모델을, 바스켓 순서는 비용 차감 상위 수익·손실 회피·IC를 합친 별도 순위 모델을 사용한다. Flow가 개선하지 못한 목표에 Flow를 억지로 넣지 않는다.</p>
<div class="scroll"><table><thead><tr><th>기간</th><th>기대수익</th><th>기대상승</th><th>기대손실</th><th>바스켓 순위</th></tr></thead><tbody>
<tr><td>5일</td><td>{html.escape(_variant_label(production['production_models']['5']['point_selection']['return']['variant']))}</td><td>{html.escape(_variant_label(production['production_models']['5']['point_selection']['upside']['variant']))}</td><td>{html.escape(_variant_label(production['production_models']['5']['point_selection']['loss']['variant']))}</td><td>{html.escape(_variant_label(production['production_models']['5']['ranking_variant']))}</td></tr>
<tr><td>20일</td><td>{html.escape(_variant_label(production['production_models']['20']['point_selection']['return']['variant']))}</td><td>{html.escape(_variant_label(production['production_models']['20']['point_selection']['upside']['variant']))}</td><td>{html.escape(_variant_label(production['production_models']['20']['point_selection']['loss']['variant']))}</td><td>{html.escape(_variant_label(production['production_models']['20']['ranking_variant']))}</td></tr>
</tbody></table></div>

<h3>수치 예측 모델의 워크포워드 오차·구간</h3>
<p>커버리지는 명목 80% P10~P90 구간이 실제 테스트값을 포함한 비율이다. 폭이 넓을수록 현재 불확실성이 크다는 뜻이다.</p>
<div class="scroll"><table><thead><tr><th>기간</th><th>목표</th><th>수치 모델</th><th>MAE</th><th>RMSE</th><th>R²</th><th>일별 IC</th><th>P10~P90 커버리지</th><th>평균 구간폭</th></tr></thead><tbody>{_point_metric_rows(evaluation, production)}</tbody></table></div>

<h3>5거래일</h3><div class="scroll"><table><thead><tr><th>모델</th><th>Fold</th><th>테스트 행</th><th>수익 MAE</th><th>일별 IC</th><th>방향 정확도</th><th>손실 MAE</th><th>상위10% 순수익</th><th>상하위 스프레드</th><th>회피 손실감소</th><th>선택점수</th></tr></thead><tbody>{_ablation_rows(evaluation,5)}</tbody></table></div>
<h3>20거래일</h3><div class="scroll"><table><thead><tr><th>모델</th><th>Fold</th><th>테스트 행</th><th>수익 MAE</th><th>일별 IC</th><th>방향 정확도</th><th>손실 MAE</th><th>상위10% 순수익</th><th>상하위 스프레드</th><th>회피 손실감소</th><th>선택점수</th></tr></thead><tbody>{_ablation_rows(evaluation,20)}</tbody></table></div>

<h2>Flow·펀더멘털 증분 검정</h2>
<h3>5거래일</h3><div class="scroll"><table><thead><tr><th>비교</th><th>판정</th><th>일별 IC Δ</th><th>상위10% 순수익 Δ</th><th>수익 MAE Δ</th><th>손실 MAE Δ</th><th>유리한 지표</th></tr></thead><tbody>{_comparison_rows(evaluation,5)}</tbody></table></div>
<h3>20거래일</h3><div class="scroll"><table><thead><tr><th>비교</th><th>판정</th><th>일별 IC Δ</th><th>상위10% 순수익 Δ</th><th>수익 MAE Δ</th><th>손실 MAE Δ</th><th>유리한 지표</th></tr></thead><tbody>{_comparison_rows(evaluation,20)}</tbody></table></div>

<h2>현재 전 종목 Forecast</h2>
<p>기대상승·기대손실은 T-1 종가 대비 이후 구간의 최대 유리/불리 움직임이다. P10~P90은 시간순 보정 표본으로 만든 경험적 예측구간이다. 표는 별도 20일 바스켓 순위점수 순이며 투자 주문이 아니라 후보 정렬·회피 필터 입력이다.</p>
<div class="scroll" style="max-height:760px"><table><thead><tr><th>종목</th><th>기준</th><th>T-1 종가</th><th>5일 기대수익</th><th>5일 상승여력</th><th>5일 기대손실</th><th>5일 수익 P10~P90</th><th>20일 기대수익</th><th>20일 상승여력</th><th>20일 기대손실</th><th>20일 손실 P90</th><th>20일 순위점수</th><th>Flow 품질</th></tr></thead><tbody>{_forecast_rows(forecasts)}</tbody></table></div>

<h2>데이터 및 품질 게이트</h2>
<ul>
 <li>FMP Ultimate: <b class="good">확인됨</b> — 가격, S&amp;P 500/Nasdaq-100 현재·변경 이력, ETF 구성 공시, 시점일치 재무·시가총액.</li>
 <li><a href="https://massive.com/docs/rest/partners/etf-global/fundflows">Massive ETF Global Fund Flows</a>: <b class="good">확인됨</b> — fund flow, NAV, shares outstanding, effective/processed date. 공급자에 행이 없는 날을 0 Flow로 만들지 않고 결측으로 보존했다.</li>
 <li>패널 타이밍: <b class="{_status_class(quality['timing_gate'])}">{html.escape(quality['timing_gate'])}</b>, 위반 {quality['timing_violation_count']}건.</li>
 <li>SPY 멤버십 공시 대조: <b class="{_status_class(membership['SPY']['gate'])}">{membership['SPY']['gate']}</b>; QQQ: <b class="{_status_class(membership['QQQ']['gate'])}">{membership['QQQ']['gate']}</b>.</li>
 <li>시장 접근 probe: <code>{html.escape(str(access_probe_path))}</code> / SHA-256 <code>{sha256_file(access_probe_path)}</code>.</li>
 <li>Nested walk-forward 과적합 감사: <code>{html.escape(str(overfit_audit_path))}</code> / 상태 <b class="warn">{html.escape(overfit_audit['status'])}</b>.</li>
 <li>노하우 DB: <code>/Users/zooh/.codex/knowledge/stock_research_knowhow_db.json</code>; 적용 workflow: apps_script_etf_radar_signal_eval, stock_etf_gostop_report의 PIT·워크포워드·원천 상태 규칙.</li>
</ul>

<h2>방법 선택의 연구 근거</h2>
<ul>
 <li><a href="https://www.nber.org/papers/w25398">Gu, Kelly, Xiu — Empirical Asset Pricing via Machine Learning</a>: 비선형 모델, 모멘텀·유동성·변동성, 단일 종목 잡음을 줄이는 횡단면 평가.</li>
 <li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2158468">Staer — Fund Flows and Underlying Returns: The Case of ETFs</a>: ETF flow의 단기 가격압력과 이후 반전 가능성.</li>
 <li><a href="https://www.nber.org/papers/w22829">Ben-David, Franzoni, Moussawi — Exchange Traded Funds</a>: ETF 거래·차익거래가 구성 종목의 변동성·상관·유동성으로 전달되는 경로.</li>
 <li><a href="https://www.nber.org/papers/w11357">Coval &amp; Stafford — Asset Fire Sales</a>: 공통 보유자금의 유출입으로 인한 가격압력과 반전.</li>
 <li><a href="https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html">Romano, Patterson, Candès — Conformalized Quantile Regression</a>: 시점분리 보정으로 예측구간의 경험적 커버리지를 직접 검사.</li>
</ul>

<h2>데이터 소스 미사용 사유서</h2>
<ul>
 <li><b>TopstepX/선물:</b> 이번 목표는 2018~현재의 종목별 종가·ETF 보유·ETF Flow 분포 예측이며, 과거 전 기간에 동일 정의로 봉인된 선물 패널이 입력 계약에 없어서 모델 피처로 사용하지 않았다. 대체는 SPY/QQQ 가격·변동성·횡단면 상태다. 따라서 장중 선물 충격을 반영하는 실시간 온도계가 아니라 장전 일별 Forecast라는 제한이 있다.</li>
 <li><b>Barchart Premier 옵션/감마:</b> 과거 전 기간 point-in-time 옵션 체인이 현재 저장소에 봉인돼 있지 않아 사용하면 기간별 결측·현재 정보 혼입이 생긴다. 옵션 의사결정은 별도 실시간 레이어에서 Forecast와 합쳐야 한다.</li>
 <li><b>Massive ETF constituents/profile/analytics:</b> 현재 권한 probe에서 403으로 부분 제한됐다. FMP Ultimate의 ETF 공시와 역사적 지수 변경 이력으로 대체했다. Massive 구성 종목 권한이 열리면 공급자 간 PIT 일치율을 추가 검정할 수 있다.</li>
</ul>
<p class="note">이 결과는 투자 수익을 보장하지 않는다. 특히 기업행사, 장중 뉴스, 옵션·선물 급변은 일별 T-1/T-2 입력 이후 발생할 수 있다. 기대손실 P90과 Flow 품질을 함께 보고 포지션 크기·회피 여부를 판단해야 한다.</p>
<p>Generated UTC: {html.escape(generated)} · Evaluation SHA-256: <code>{sha256_file(evaluation_path)}</code> · Panel manifest SHA-256: <code>{sha256_file(panel_manifest_path)}</code></p>
</main></body></html>"""
    write_text_atomic(output_path, html_text)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "forecast_rows": len(forecasts),
        "generated_at_utc": generated,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--panel-manifest", type=Path, default=DEFAULT_PANEL_MANIFEST)
    parser.add_argument(
        "--production-manifest", type=Path, default=DEFAULT_PRODUCTION_MANIFEST
    )
    parser.add_argument("--access-probe", type=Path, default=DEFAULT_ACCESS_PROBE)
    parser.add_argument("--overfit-audit", type=Path, default=DEFAULT_OVERFIT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = render_report(
        evaluation_path=args.evaluation,
        panel_manifest_path=args.panel_manifest,
        production_manifest_path=args.production_manifest,
        access_probe_path=args.access_probe,
        overfit_audit_path=args.overfit_audit,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
