import gradio as gr
import json
import subprocess
import asyncio
import os
import sys
import glob
import logging
import re
import tempfile
import shutil
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import io
from pathlib import Path
import hashlib
import time
import pytz
import requests
from market_analysis.insights import build_market_narrative, resolve_effective_index
try:
    from market_analysis.market_report import generate_market_report  # 확률 리포트 엔진
except Exception as _mr_exc:  # pragma: no cover
    generate_market_report = None  # type: ignore
try:
    from market_analysis.market_prob_backtest import run_backtest as run_market_prob_backtest
except Exception as _mpb_exc:  # pragma: no cover
    run_market_prob_backtest = None  # type: ignore
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None  # graceful fallback
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

# === Use AutoTrade2 SoT API (no fallback) ===
try:
    from regime_service import (
        at2_backtest_close,
        at2_get_payload_close_raw,
        at2_get_payload_now_raw,
        at2_get_ticker_series,
        build_recent_transition_markdown,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "AutoTrade2 SoT API unavailable; ensure regime_service is accessible before launching the web app"
    ) from exc



# AutoTrade2 SoT만 사용하므로 별도 RegimeFetcher 스텁은 제공하지 않습니다.

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTML 변환 함수 제거됨 - Markdown 직접 사용으로 변경

# HTML 변환 함수들 제거됨 - Markdown 직접 사용으로 변경


class StockAnalysisWebApp:
    """주식 분석 웹앱"""
    
    def __init__(self):
        self.current_module = "pd.py"
        self.favorites_file = "favorites.json"
        self.available_modules = self._scan_available_modules()
        self.last_analysis_result = None  # 마지막 분석 결과 저장
        self._last_realtime_payload: Optional[Dict[str, Any]] = None

    # === StockAnalysisWebApp methods: route everything via AutoTrade2 ===

    def _fetch_payload_via_autotrade2(
        self,
        window_val: int = 30,
        use_real: bool = True,
        preset: Optional[str] = None,
        *,
        auto_override: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        레짐·Classic·FFL-STAB·Fusion·series 전부 AutoTrade2에서 수신.
        실시간(use_real=True)은 항상 강제 재계산하며 캐시를 사용하지 않는다.
        """
        if use_real:
            return at2_get_payload_now_raw(  # type: ignore
                window=window_val,
                preset=preset,
                auto_override=auto_override,
                force_refresh=True,
                prefer_cache=False,
                **kwargs,
            )
        return at2_get_payload_close_raw(window=window_val, preset=preset, **kwargs)  # type: ignore


    def _backtest_via_autotrade2(
        self,
        start_date: str,
        end_date: str,
        window_val: int = 30,
        preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        백테스트 시계열도 AutoTrade2에서 수신.
        """
        return at2_backtest_close(start_date, end_date, window=window_val, preset=preset)  # type: ignore

    def _tickers_via_autotrade2(
        self,
        window_val: int = 30,
        use_real: bool = False,
        preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        return at2_get_ticker_series(  # type: ignore
            window=window_val,
            preset=preset,
            use_realtime=use_real,
        )

    def _classic_ffl_states_via_autotrade2(
        self,
        window_val: int = 30,
        use_real: bool = False,
        preset: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Classic/FFL-STAB/Fusion 상태 블록 추출.
        """
        payload = self._fetch_payload_via_autotrade2(window_val=window_val, use_real=use_real, preset=preset)
        return (
            payload.get("classic", {}) or {},
            payload.get("ffl_stab", {}) or {},
            payload.get("fusion", {}) or {},
        )
        
    def _scan_available_modules(self) -> List[str]:
        """사용 가능한 분석 모듈 스캔"""
        try:
            modules = []
            for file in Path(".").glob("*.py"):
                if file.name not in ["stock_analysis_webapp.py", "__init__.py"]:
                    modules.append(file.name)
            
            if "pd.py" not in modules:
                modules.insert(0, "pd.py")
                
            return sorted(modules)
        except Exception as e:
            print(f"모듈 스캔 실패: {e}")
            return ["pd.py"]
    
    def load_favorites(self) -> List[str]:
        """favorites.json 파일 로드"""
        try:
            if os.path.exists(self.favorites_file):
                with open(self.favorites_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        return [str(item) for item in data if item]  # 빈 값 제거
                    else:
                        return []
            else:
                default_list = ["ACHR", "JOBY", "SLDP", "NVDA", "QBTS", "MRVL", "RKLB", "GOOGL", "QS"]
                self.save_favorites(default_list)
                return default_list
        except Exception as e:
            print(f"Favorites 로드 실패: {e}")
            return ["ACHR", "JOBY", "SLDP", "NVDA", "QBTS"]  # 기본값 반환
    
    def save_favorites(self, ticker_list: List[str]) -> bool:
        """favorites.json 파일 저장"""
        try:
            # 빈 값 제거 및 문자열 변환
            clean_list = [str(ticker).strip().upper() for ticker in ticker_list if ticker and str(ticker).strip()]
            
            with open(self.favorites_file, "w", encoding="utf-8") as f:
                json.dump(clean_list, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Favorites 저장 실패: {e}")
            return False
    
    def update_module(self, module_name: str) -> str:
        """분석 모듈 변경"""
        try:
            if not module_name or not isinstance(module_name, str):
                return "❌ 올바른 파일명을 입력하세요."
                
            module_name = module_name.strip()
            if not module_name.endswith('.py'):
                return "❌ .py 파일만 지원됩니다."
            
            if os.path.exists(module_name):
                self.current_module = module_name
                return f"✅ 모듈이 '{module_name}'로 변경되었습니다."
            else:
                return f"❌ '{module_name}' 파일을 찾을 수 없습니다."
        except Exception as e:
            return f"❌ 모듈 변경 중 오류: {str(e)[:100]}"
    
    def get_current_module_info(self) -> str:
        """현재 모듈 정보 반환"""
        try:
            info = f"**현재 분석 모듈:** `{self.current_module}`\n\n**사용 가능한 모듈:**\n"
            info += "\n".join([f"- {module}" for module in self.available_modules])
            return info
        except Exception as e:
            return f"**현재 분석 모듈:** `{self.current_module}`\n\n모듈 정보 로드 실패: {e}"





    
    
    # === Realtime Regime (FMP) ===
    def _plot_regime_states(self, payload: Dict[str, Any]):
        # Use unified effective dates (ET basis, drop today stub when appropriate)
        dates: List[Any] = self._effective_dates(payload)
        fusion = payload.get("fusion", {}) or {}
        states_raw = fusion.get("state") or payload.get("states") or []
        if not dates or not states_raw:
            return go.Figure() if go is not None else {}

        n = min(len(dates), len(states_raw))
        dates = list(dates)[-n:]

        def _align(series: Any) -> List[Optional[float]]:
            if not isinstance(series, list):
                return [None] * n
            if len(series) < n:
                return [None] * (n - len(series)) + series
            return series[-n:]

        states = []
        for val in states_raw[-n:]:
            try:
                states.append(int(val))
            except Exception:
                states.append(0)
        scores = _align(fusion.get("score"))
        wta_series = _align(fusion.get("wTA"))
        wflow_series = _align(fusion.get("wFlow"))

        def _color_for_state(state: int) -> str:
            if state > 0:
                return "#2ecc71"
            if state < 0:
                return "#c0392b"
            return "#95a5a6"

        colors = [_color_for_state(s) for s in states]

        def _label_state(state: int) -> str:
            if state > 0:
                return "Risk-On"
            if state < 0:
                return "Risk-Off"
            return "Neutral"

        hover_text = []
        for idx, state in enumerate(states):
            text = f"{dates[idx]}<br>상태: {_label_state(state)}"
            if isinstance(scores[idx], (int, float)):
                text += f"<br>score: {scores[idx]:.3f}"
            extras = []
            if isinstance(wta_series[idx], (int, float)):
                extras.append(f"wTA {wta_series[idx]:.2f}")
            if isinstance(wflow_series[idx], (int, float)):
                extras.append(f"wFlow {wflow_series[idx]:.2f}")
            if extras:
                text += "<br>" + " · ".join(extras)
            hover_text.append(text)

        if go is not None:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=states,
                    name="Fusion State",
                    marker_color=colors,
                    hovertext=hover_text,
                    hovertemplate="%{hovertext}<extra></extra>",
                    opacity=0.85,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=scores,
                    mode="lines+markers",
                    name="Score",
                    yaxis="y2",
                    line=dict(color="#2980b9", width=2),
                    marker=dict(size=5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=wta_series,
                    mode="lines",
                    name="wTA",
                    yaxis="y2",
                    line=dict(color="#8e44ad", dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=wflow_series,
                    mode="lines",
                    name="wFlow",
                    yaxis="y2",
                    line=dict(color="#f39c12", dash="dash"),
                )
            )
            fig.update_layout(
                height=360,
                barmode="relative",
                legend=dict(orientation="h"),
                yaxis=dict(title="State", range=[-1.2, 1.2], dtick=1, zeroline=True),
                yaxis2=dict(title="Score / Weights", overlaying="y", side="right", range=[-0.1, 1.05]),
                margin=dict(t=30, l=40, r=40, b=40),
            )
            return fig

        if plt is None:
            return {}
        fig, ax1 = plt.subplots(figsize=(8, 3.6))
        ax1.bar(dates, states, color=colors, alpha=0.8)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_yticks([-1, 0, 1])
        ax1.set_ylabel("State")
        ax1.set_title("Fusion Signal (Bar)")
        ax2 = ax1.twinx()
        if any(isinstance(v, (int, float)) for v in scores):
            ax2.plot(dates, scores, label="Score", color="#2980b9", linewidth=1.8)
        if any(isinstance(v, (int, float)) for v in wta_series):
            ax2.plot(dates, wta_series, label="wTA", color="#8e44ad", linestyle="--")
        if any(isinstance(v, (int, float)) for v in wflow_series):
            ax2.plot(dates, wflow_series, label="wFlow", color="#f39c12", linestyle="-.")
        ax2.set_ylim(-0.1, 1.05)
        ax2.set_ylabel("Score / Weights")
        handles, labels = [], []
        for ax in (ax1, ax2):
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        if handles:
            ax2.legend(handles, labels, loc="upper center", ncol=4)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    def _effective_dates(self, payload: Dict[str, Any]) -> List[Any]:
        """Return dates aligned for display (ET basis with today stub fallback),
        with sanitation: drop invalid/future trailing dates while preserving order."""
        try:
            src = (payload.get("fusion", {}) or {}).get("dates") or payload.get("dates") or []
            if not isinstance(src, list) or not src:
                return []
            asof = payload.get("asof", {}) or {}
            today = asof.get("today_utc") or asof.get("fusion_last_date")
            intraday = bool(asof.get("intraday_base_applied"))
            seq = [str(d) for d in src if isinstance(d, (str, bytes))]
            # Clamp trailing future dates if any (defensive against bad stubs)
            if isinstance(today, str):
                while seq and str(seq[-1]) > str(today):
                    seq.pop()
            # If not intraday and last equals today, drop today stub
            if len(seq) >= 2 and not intraday and isinstance(today, str) and seq[-1] == today:
                seq = seq[:-1]
            return list(seq)
        except Exception:
            return list((payload.get("dates") or []))

    def _plot_stability(self, payload: Dict[str, Any]):
        dates = self._effective_dates(payload)
        n = len(dates)
        stab = payload.get("stability", [])
        smoothed = payload.get("smoothed", [])
        sub = payload.get("sub", {}) or {}

        def _align(arr):
            if not isinstance(arr, list):
                return [None] * n
            if len(arr) < n:
                return [None] * (n - len(arr)) + list(arr)
            return list(arr[-n:])

        stab_a = _align(stab)
        smoo_a = _align(smoothed)
        sc = _align(sub.get("stockCrypto")) if sub else [None] * n
        tr = _align(sub.get("traditional")) if sub else [None] * n
        sn = _align(sub.get("safeNegative")) if sub else [None] * n

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=smoo_a, mode="lines", name="Stability(EMA10)"))
            if any(v is not None for v in stab_a):
                fig.add_trace(go.Scatter(x=dates, y=stab_a, mode="lines", name="Stability", visible="legendonly"))
            if any(v is not None for v in sc):
                fig.add_trace(go.Scatter(x=dates, y=sc, mode="lines", name="Stock-Crypto(+)", visible="legendonly"))
            if any(v is not None for v in tr):
                fig.add_trace(go.Scatter(x=dates, y=tr, mode="lines", name="Traditional(+)", visible="legendonly"))
            if any(v is not None for v in sn):
                fig.add_trace(go.Scatter(x=dates, y=sn, mode="lines", name="Safe-NEG(-)", visible="legendonly"))
            fig.update_layout(height=360, legend=dict(orientation="h"))
            return fig
        if plt is None:
            return {}
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(dates, smoo_a, label="Stability(EMA10)")
        if any(v is not None for v in stab_a):
            ax.plot(dates, stab_a, label="Stability")
        ax.legend(loc="upper center", ncol=2)
        ax.set_title("Stability & Sub-Indices")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # NOTE: duplicate _plot_stability (unaligned) removed to keep single, aligned implementation above.

    # NOTE: legacy _plot_backtest (class-level) removed; realtime tab uses the aligned nested version.
    
    async def run_analysis(self, tickers: List[str]) -> Dict:
        """분석 실행"""
        output_file = None
        try:
            # 입력 검증
            if not tickers or not isinstance(tickers, list):
                return {"error": "분석할 티커를 입력하세요."}
            
            # 빈 값 제거 및 정제
            clean_tickers = [str(t).strip().upper() for t in tickers if t and str(t).strip()]
            
            if not clean_tickers:
                return {"error": "유효한 티커가 없습니다."}
            
            if len(clean_tickers) > 15:
                return {"error": "최대 15개 티커까지만 분석 가능합니다."}
            
            # 임시 출력 파일
            output_file = f"temp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # 명령어 구성
            cmd = [sys.executable, self.current_module, "--tickers"] + clean_tickers + ["--output", output_file]
            
            print(f"🔄 실행 명령어: {' '.join(cmd)}")
            
            # 프로세스 실행 (타임아웃 적용)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="."
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)  # 5분 타임아웃
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                return {"error": "분석 시간이 초과되었습니다. (5분 제한)"}
            
            if process.returncode == 0:
                # 결과 파일 읽기
                if os.path.exists(output_file):
                    try:
                        with open(output_file, "r", encoding="utf-8") as f:
                            result = json.load(f)
                        # 결과를 인스턴스 변수에 저장
                        self.last_analysis_result = result
                        return result
                    except json.JSONDecodeError:
                        return {"error": "분석 결과 파일 형식이 올바르지 않습니다."}
                    except Exception as e:
                        return {"error": f"결과 파일 읽기 실패: {str(e)[:100]}"}
                else:
                    return {"error": "결과 파일이 생성되지 않았습니다."}
            else:
                error_msg = stderr.decode('utf-8', errors='replace') if stderr else "알 수 없는 오류"
                return {"error": f"분석 실행 실패: {error_msg[:500]}"}
                
        except Exception as e:
            return {"error": f"분석 중 오류 발생: {str(e)[:200]}"}
        finally:
            # 임시 파일 정리
            if output_file and os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
    
    def _format_report_header(self, result: Dict) -> List[str]:
        """보고서 헤더 섹션 생성"""
        return [
            "# 📊 AI 주식 분석 보고서",
            "",
            f"**생성 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**분석 모듈:** `{self.current_module}`",
            f"**원본 분석 시간:** {result.get('timestamp', 'N/A')}",
            "",
            "---"
        ]
    
    def _format_market_regime_section(self, result: Dict) -> List[str]:
        """시장 레짐 섹션 생성"""
        if "market_regime" not in result or not isinstance(result["market_regime"], dict):
            return []
        
        regime = result["market_regime"]
        regime_emoji = self._get_regime_emoji(regime.get('regime_type', 'NEUTRAL'))
        
        return [
            f"## {regime_emoji} 시장 레짐 분석",
            "",
            f"- **현재 레짐:** {regime.get('regime_type', 'N/A')} (확신도: {regime.get('confidence', 0):.1%})",
            f"- **SPY 변화:** {regime.get('spy_change', 0):+.2f}%",
            f"- **QQQ 변화:** {regime.get('qqq_change', 0):+.2f}%",
            f"- **VIX 수준:** {regime.get('vix_level', 0):.1f}",
            ""
        ]
    
    def _format_portfolio_summary_section(self, result: Dict) -> List[str]:
        """포트폴리오 요약 섹션 생성"""
        if "summary" not in result or not isinstance(result["summary"], dict):
            return []
        
        summary = result["summary"]
        portfolio_weights = result.get("portfolio_weights", {})
        cash_weight = portfolio_weights.get("CASH", 0) if isinstance(portfolio_weights, dict) else 0
        
        return [
            "## 💰 포트폴리오 요약",
            "",
            f"- **분석 종목 수:** {summary.get('total_stocks', 0)}개",
            f"- **투자 비중:** {summary.get('invested_ratio', 0):.1%}",
            f"- **현금 비중:** {abs(cash_weight):.1%}",
            f"- **STRONG_BUY:** {summary.get('strong_buy_count', 0)}개",
            f"- **극단적 조정:** {summary.get('extreme_adjustments', 0)}개 종목",
            ""
        ]
    
    def _format_portfolio_allocation_section(self, result: Dict) -> List[str]:
        """포트폴리오 배분 섹션 생성"""
        if "portfolio_weights" not in result or not isinstance(result["portfolio_weights"], dict):
            return []
        
        weights = result["portfolio_weights"]
        section = ["## 📈 추천 포트폴리오 배분", ""]
        
        # 투자 종목만 필터링하고 정렬
        invested_stocks = {k: v for k, v in weights.items() 
                         if k != "CASH" and isinstance(v, (int, float)) and v > 0.001}
        
        if invested_stocks:
            sorted_stocks = sorted(invested_stocks.items(), key=lambda x: x[1], reverse=True)
            
            for ticker, weight in sorted_stocks:
                safe_ticker = str(ticker).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                section.append(f"- **{safe_ticker}:** {weight:.1%}")
            
            # 현금 표시
            cash_weight = weights.get("CASH", 0)
            if isinstance(cash_weight, (int, float)) and abs(cash_weight) > 0.001:
                section.append(f"- **현금:** {abs(cash_weight):.1%}")
        else:
            section.append("- **투자 추천 종목이 없습니다** (100% 현금 보유 권장)")
        
        section.append("")
        return section
    
    def _format_stock_details_section(self, result: Dict, original_ticker_order: List[str] = None) -> List[str]:
        """종목별 상세 분석 섹션 생성"""
        if "signals" not in result or not isinstance(result["signals"], dict):
            return []
        
        signals = result["signals"]
        section = ["## 🎯 종목별 상세 분석", ""]
        
        # 티커 순서 결정
        if original_ticker_order:
            ordered_tickers = [ticker for ticker in original_ticker_order if ticker in signals]
            for ticker in signals:
                if ticker not in ordered_tickers:
                    ordered_tickers.append(ticker)
        else:
            ordered_tickers = list(signals.keys())
        
        for ticker in ordered_tickers:
            if ticker not in signals:
                continue
                
            data = signals[ticker]
            if not isinstance(data, dict):
                continue
            
            signal = data.get("signal", "UNKNOWN")
            signal_emoji = self._get_signal_emoji(signal)
            
            # 안전한 데이터 추출
            weight = data.get("weight", 0) if isinstance(data.get("weight"), (int, float)) else 0
            final_score = data.get("final_score", 0) if isinstance(data.get("final_score"), (int, float)) else 0
            
            # 기본 정보
            safe_ticker = str(ticker).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            section.append(f"### {signal_emoji} {safe_ticker} - {signal}")
            section.append(f"- **최종 점수:** {final_score:.1f}/100")
            section.append(f"- **포트폴리오 비중:** {weight:.1%}")
            
            # 5축 점수
            if "axis_scores" in data and isinstance(data["axis_scores"], dict):
                scores = data["axis_scores"]
                section.append(f"- **5축 점수:**")
                section.append(f"  - 펀더멘탈: {scores.get('fundamental', 0):.0f}")
                section.append(f"  - 기술적: {scores.get('technical', 0):.0f}")
                section.append(f"  - R&D: {scores.get('rnd', 0):.0f}")
                section.append(f"  - 센티멘트: {scores.get('sentiment', 0):.0f}")
                section.append(f"  - 해자성: {scores.get('moat', 0):.0f}")
            
            # 축별 가중치 정보
            if "axis_weights" in data and isinstance(data["axis_weights"], dict):
                weights_info = data["axis_weights"]
                if weights_info:
                    section.append(f"- **축별 가중치:**")
                    
                    axis_names = {
                        'fundamental': '펀더멘탈', 'technical': '기술적', 'rnd': 'R&D',
                        'sentiment': '센티멘트', 'moat': '해자성'
                    }
                    
                    sorted_weights = sorted(weights_info.items(), key=lambda x: x[1], reverse=True)
                    
                    for i, (axis, weight_val) in enumerate(sorted_weights):
                        axis_name = axis_names.get(axis, axis)
                        if i == 0:
                            section.append(f"  - **{axis_name}: {weight_val:.1%} (주도축)**")
                        else:
                            section.append(f"  - {axis_name}: {weight_val:.1%}")
            
            # AI 근거
            if data.get("ai_rationale") and isinstance(data.get("ai_rationale"), str):
                rationale = data["ai_rationale"][:300]
                if len(data["ai_rationale"]) > 300:
                    rationale += "..."
                safe_rationale = rationale.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                section.append(f"- **AI 분석:** {safe_rationale}")
            
            section.append("")
        
        return section

    def format_analysis_report(self, result: Dict, original_ticker_order: List[str] = None) -> str:
        """분석 결과를 보고서 형태로 포맷팅 (순서 유지)"""
        if not result or not isinstance(result, dict):
            return "❌ **분석 결과가 없습니다.**"
            
        if "error" in result:
            return f"❌ **분석 실패**\n\n{result['error']}"
        
        try:
            report = []
            
            # 섹션별로 분리된 메서드 호출
            report.extend(self._format_report_header(result))
            report.extend(self._format_market_regime_section(result))
            report.extend(self._format_portfolio_summary_section(result))
            report.extend(self._format_portfolio_allocation_section(result))
            report.extend(self._format_stock_details_section(result, original_ticker_order))
            
            # 푸터
            report.append("---")
            report.append("**⚠️ 투자 유의사항**")
            report.append("")
            report.append("본 보고서는 AI 기반 분석 결과이며, 투자 판단의 참고 자료로만 활용하시기 바랍니다.")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"❌ **보고서 생성 중 오류 발생**\n\n{str(e)[:200]}"
    
    def get_analysis_text_for_copy(self) -> str:
        """복사용 텍스트 생성"""
        if not self.last_analysis_result:
            return "❌ 복사할 분석 결과가 없습니다. 먼저 분석을 실행하세요."
        
        try:
            # Markdown 형식을 제거한 순수 텍스트 버전 생성
            result = self.last_analysis_result
            lines = []
            
            lines.append("=== AI 주식 분석 결과 ===")
            lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"분석 모듈: {self.current_module}")
            lines.append("")
            
            # 시장 레짐
            if "market_regime" in result and isinstance(result["market_regime"], dict):
                regime = result["market_regime"]
                lines.append("=== 시장 레짐 분석 ===")
                lines.append(f"현재 레짐: {regime.get('regime_type', 'N/A')} (확신도: {regime.get('confidence', 0):.1%})")
                lines.append(f"SPY 변화: {regime.get('spy_change', 0):+.2f}%")
                lines.append(f"QQQ 변화: {regime.get('qqq_change', 0):+.2f}%")
                lines.append(f"VIX 수준: {regime.get('vix_level', 0):.1f}")
                lines.append("")
            
            # 포트폴리오 요약
            if "summary" in result and isinstance(result["summary"], dict):
                summary = result["summary"]
                lines.append("=== 포트폴리오 요약 ===")
                lines.append(f"분석 종목 수: {summary.get('total_stocks', 0)}개")
                lines.append(f"투자 비중: {summary.get('invested_ratio', 0):.1%}")
                lines.append(f"STRONG_BUY: {summary.get('strong_buy_count', 0)}개")
                lines.append("")
            
            # 종목별 분석
            if "signals" in result and isinstance(result["signals"], dict):
                lines.append("=== 종목별 분석 결과 ===")
                for ticker, data in result["signals"].items():
                    if not isinstance(data, dict):
                        continue
                    
                    lines.append(f"\n[{ticker}]")
                    lines.append(f"신호: {data.get('signal', 'N/A')}")
                    lines.append(f"최종 점수: {data.get('final_score', 0):.1f}/100")
                    lines.append(f"포트폴리오 비중: {data.get('weight', 0):.1%}")
                    
                    if "axis_scores" in data and isinstance(data["axis_scores"], dict):
                        scores = data["axis_scores"]
                        lines.append(f"5축 점수: 펀더멘탈({scores.get('fundamental', 0):.0f}) 기술적({scores.get('technical', 0):.0f}) R&D({scores.get('rnd', 0):.0f}) 센티멘트({scores.get('sentiment', 0):.0f}) 해자성({scores.get('moat', 0):.0f})")
                    
                    if data.get("ai_rationale"):
                        lines.append(f"AI 분석: {data['ai_rationale'][:200]}...")
            
            lines.append("\n=== 투자 유의사항 ===")
            lines.append("본 결과는 AI 기반 분석이며, 투자 판단의 참고 자료로만 활용하시기 바랍니다.")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"❌ 복사용 텍스트 생성 오류: {str(e)[:100]}"
    
    def get_analysis_json_for_download(self) -> tuple[str, str]:
        """다운로드용 JSON 파일 생성 (파일경로, 메시지 반환)"""
        if not self.last_analysis_result:
            return None, "❌ 분석 결과가 없습니다"
        
        try:
            # 다운로드용 JSON에 메타데이터 추가
            download_data = {
                "metadata": {
                    "export_time": datetime.now().isoformat(),
                    "analysis_module": self.current_module,
                    "export_version": "1.0"
                },
                "analysis_result": self.last_analysis_result
            }
            
            # 임시 파일 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"stock_analysis_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(download_data, f, indent=2, ensure_ascii=False)
            
            return filename, f"✅ JSON 파일이 생성되었습니다: {filename}"
            
        except Exception as e:
            return None, f"❌ JSON 파일 생성 오류: {str(e)[:100]}"
    
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            import glob
            temp_files = glob.glob("stock_analysis_*.json")
            for file in temp_files:
                try:
                    # 1시간 이상 된 파일만 삭제
                    if os.path.exists(file):
                        file_time = os.path.getmtime(file)
                        current_time = datetime.now().timestamp()
                        if current_time - file_time > 3600:  # 1시간
                            os.remove(file)
                except:
                    pass
        except Exception:
            pass
    
    def _get_signal_emoji(self, signal: str) -> str:
        """시그널별 이모지 반환"""
        emoji_map = {
            "STRONG_BUY": "🚀",
            "BUY": "📈", 
            "HOLD": "⚖️",
            "WEAK_HOLD": "📉",
            "AVOID": "⛔"
        }
        return emoji_map.get(str(signal), "❓")
    
    def _get_regime_emoji(self, regime: str) -> str:
        """레짐별 이모지 반환"""
        emoji_map = {
            "GROWTH": "🌱",
            "MOMENTUM": "🚀",
            "DEFENSIVE": "🛡️",
            "CRISIS": "⚠️",
            "NEUTRAL": "⚖️"
        }
        return emoji_map.get(str(regime), "🌍")
    
    def format_favorites_display(self, favorites: List[str]) -> str:
        """Favorites를 보기 좋게 포맷팅"""
        try:
            if not favorites or not isinstance(favorites, list):
                return "❌ 즐겨찾기가 비어있습니다."
            
            clean_favorites = [str(f) for f in favorites if f]
            if not clean_favorites:
                return "❌ 즐겨찾기가 비어있습니다."
            
            display = ["# 📋 즐겨찾기 종목 목록"]
            display.append("")
            display.append(f"**총 {len(clean_favorites)}개 종목 (순서대로):**")
            display.append("")
            
            for i, ticker in enumerate(clean_favorites, 1):
                safe_ticker = str(ticker).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                display.append(f"{i:2d}. **{safe_ticker}**")
            
            return "\n".join(display)
        except Exception as e:
            return f"❌ 즐겨찾기 표시 오류: {str(e)[:100]}"
    
    def format_favorites_for_editing(self, favorites: List[str]) -> str:
        """편집을 위한 JSON 형태로 포맷팅"""
        try:
            if not favorites or not isinstance(favorites, list):
                return '[\n  "ACHR",\n  "JOBY",\n  "SLDP",\n  "NVDA",\n  "QBTS"\n]'
            
            clean_favorites = [str(f) for f in favorites if f]
            return json.dumps(clean_favorites, indent=2, ensure_ascii=False)
        except Exception as e:
            return '[\n  "ACHR",\n  "JOBY",\n  "SLDP",\n  "NVDA",\n  "QBTS"\n]'
    
    def validate_and_save_favorites(self, json_text: str) -> tuple[str, List[str]]:
        """JSON 텍스트 검증 후 favorites 저장"""
        try:
            if not json_text or not isinstance(json_text, str):
                return "❌ 올바른 JSON 텍스트를 입력하세요.", []
            
            # JSON 파싱
            parsed_data = json.loads(json_text.strip())
            
            # 리스트 형태만 허용
            if not isinstance(parsed_data, list):
                return "❌ JSON은 리스트 형태여야 합니다.", []
            
            if len(parsed_data) == 0:
                return "❌ 최소 하나의 종목이 필요합니다.", []
            
            if len(parsed_data) > 20:  # 제한 완화
                return f"❌ 최대 20개 종목까지만 가능합니다. (현재 {len(parsed_data)}개)", []
            
            # 개별 티커 검증
            clean_tickers = []
            for i, ticker in enumerate(parsed_data):
                if not ticker or not str(ticker).strip():
                    continue  # 빈 값 건너뛰기
                
                ticker_str = str(ticker).strip().upper()
                
                # 기본적인 티커 형식 검사
                import re
                if re.match(r'^[A-Za-z0-9.-]+$', ticker_str) and len(ticker_str) <= 20:
                    clean_tickers.append(ticker_str)
                else:
                    return f"❌ '{ticker}'는 올바르지 않은 티커 형식입니다.", []
            
            if not clean_tickers:
                return "❌ 유효한 티커가 없습니다.", []
            
            # 저장
            if self.save_favorites(clean_tickers):
                success_msg = f"✅ favorites.json 저장 완료!\n\n{len(clean_tickers)}개 종목이 순서대로 저장되었습니다."
                return success_msg, clean_tickers
            else:
                return "❌ 파일 저장에 실패했습니다.", []
                
        except json.JSONDecodeError as e:
            return f"❌ JSON 형식 오류: 올바른 JSON 형식으로 입력하세요.", []
        except Exception as e:
            return f"❌ 저장 실패: {str(e)[:100]}", []
    
    def load_ml_parameters(self) -> Dict:
        """ML 최적화된 파라미터 로드"""
        try:
            if os.path.exists("ml_parameters.json"):
                with open("ml_parameters.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # ML 파라미터가 비활성화되었는지 확인
                if data.get('disabled', False):
                    return None
                    
                # 새로운 ML 파라미터 구조 처리 (v6.0)
                if 'current_parameters' in data:
                    current_params = data['current_parameters']
                    main_weights = current_params.get('main_scoring_weights', {})
                    detailed_weights = current_params.get('detailed_scoring_weights', {})
                    multipliers = current_params.get('sweet_spot_multipliers', {})
                    deeptech_multipliers = current_params.get('deeptech_category_multipliers', {})
                    
                    # 새로운 ML v6.0 파라미터 구조로 반환 
                    ml_config = {
                        # 메인 점수 가중치 (ML 최적화된 값)
                        "pattern_score": main_weights.get('pattern_score', 0.25),
                        "convergence_score": main_weights.get('convergence_score', 0.25), 
                        "growth_score": main_weights.get('growth_score', 0.20),
                        "tech_score": main_weights.get('tech_score', 0.20),
                        "institutional_score": main_weights.get('institutional_score', 0.05),
                        "financial_score": main_weights.get('financial_score', 0.05),
                        
                        # Sweet Spot 단계별 배수 (ML 최적화된 값)
                        "early_recovery_multiplier": multipliers.get('early_recovery_multiplier', 1.3),
                        "golden_time_multiplier": multipliers.get('golden_time_multiplier', 1.5),
                        "late_recovery_multiplier": multipliers.get('late_recovery_multiplier', 0.8),
                        "overheated_penalty": multipliers.get('overheated_penalty', 0.6),
                        
                        # 딥테크 카테고리 배수 (ML 최적화된 값)
                        "ai_computing": deeptech_multipliers.get('ai_computing', 1.2),
                        "quantum_tech": deeptech_multipliers.get('quantum_tech', 1.3),
                        "bio_health_tech": deeptech_multipliers.get('bio_health_tech', 1.25),
                        "emerging_tech": deeptech_multipliers.get('emerging_tech', 1.15),
                        "semiconductor": deeptech_multipliers.get('semiconductor', 1.1),
                        "mobility_tech": deeptech_multipliers.get('mobility_tech', 1.2),
                        "energy_materials": deeptech_multipliers.get('energy_materials', 1.1),
                        "security_fintech": deeptech_multipliers.get('security_fintech', 1.0),
                        
                        # 상세 가중치 정보 (28개 파라미터)
                        "detailed_weights": detailed_weights,
                        
                        # ML 메타데이터
                        "is_ml_optimized": True,
                        "ml_version": data.get('metadata', {}).get('version', '6.0'),
                        "last_updated": data.get('metadata', {}).get('last_updated', 'Unknown'),
                        "parameters_count": data.get('metadata', {}).get('parameters_count', 28)
                    }
                    
                    return ml_config
                    
        except Exception as e:
            print(f"ML 파라미터 로드 실패: {e}")
            pass
        
        return None

# 웹앱 인스턴스 생성
app = StockAnalysisWebApp()

# Gradio 인터페이스 함수들
def update_analysis_module(module_name: str):
    """분석 모듈 업데이트"""
    try:
        result = app.update_module(module_name)
        module_info = app.get_current_module_info()
        return result, module_info
    except Exception as e:
        return f"❌ 오류 발생: {str(e)[:100]}", app.get_current_module_info()

def load_and_display_favorites():
    """즐겨찾기 로드 및 표시"""
    try:
        favorites = app.load_favorites()
        display = app.format_favorites_display(favorites)
        edit_json = app.format_favorites_for_editing(favorites)
        return display, favorites, edit_json
    except Exception as e:
        error_msg = f"❌ 로드 오류: {str(e)[:100]}"
        return error_msg, [], "[]"

def save_edited_favorites(json_text: str):
    """편집된 favorites 저장"""
    try:
        result_msg, new_favorites = app.validate_and_save_favorites(json_text)
        display = app.format_favorites_display(new_favorites)
        editor_json = app.format_favorites_for_editing(new_favorites)
        return result_msg, display, new_favorites, editor_json
    except Exception as e:
        error_msg = f"❌ 저장 오류: {str(e)[:100]}"
        return error_msg, "오류 발생", [], json_text

def run_analysis_from_favorites(favorites_data: List[str]):
    """즐겨찾기로 분석 실행"""
    try:
        if not favorites_data or not isinstance(favorites_data, list):
            return "❌ 즐겨찾기가 비어있습니다.", ""
        return run_custom_analysis(favorites_data)
    except Exception as e:
        return f"❌ 분석 오류: {str(e)[:100]}", ""

def run_custom_analysis(tickers_input):
    """커스텀 티커로 분석 실행"""
    loop = None
    try:
        # 티커 파싱
        if isinstance(tickers_input, str):
            tickers = [t.strip().upper() for t in tickers_input.replace(',', ' ').split() if t.strip()]
        elif isinstance(tickers_input, list):
            tickers = [str(t).strip().upper() for t in tickers_input if t and str(t).strip()]
        else:
            return "❌ 올바른 티커 형식을 입력하세요.", ""
        
        if not tickers:
            return "❌ 분석할 티커를 입력하세요.", ""
        
        if len(tickers) > 15:
            return "❌ 최대 15개 티커까지만 분석 가능합니다.", ""
        
        # 비동기 분석 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(app.run_analysis(tickers))
        
        # 보고서 생성 (티커 순서 유지)
        report = app.format_analysis_report(result, tickers)
        
        # 복사용 텍스트 생성
        copy_text = app.get_analysis_text_for_copy()
        
        return report, copy_text
        
    except Exception as e:
        return f"❌ 분석 중 오류 발생: {str(e)[:200]}", ""
    finally:
        # 이벤트 루프 정리
        if loop:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
            except Exception:
                pass

def get_copy_text():
    """복사용 텍스트 반환"""
    return app.get_analysis_text_for_copy()

def create_json_file():
    """JSON 파일 생성 및 반환"""
    try:
        # 임시 파일 정리
        app.cleanup_temp_files()
        
        filename, message = app.get_analysis_json_for_download()
        if filename and os.path.exists(filename):
            return filename, message
        else:
            return None, message
    except Exception as e:
        return None, f"❌ 파일 생성 오류: {str(e)[:100]}"

def show_copy_textbox():
    """복사용 텍스트박스 표시"""
    copy_text = get_copy_text()
    return gr.update(visible=True, value=copy_text)

def show_json_download():
    """JSON 다운로드 준비"""
    filename, message = create_json_file()
    if filename:
        return gr.update(visible=True, value=filename), gr.update(visible=True, value=message)
    else:
        return gr.update(visible=False), gr.update(visible=True, value=message)


def create_interface():
    initial_favorites = app.load_favorites()
    initial_favorites_json = app.format_favorites_for_editing(initial_favorites)
    with gr.Blocks(
        title="AI 주식 분석 웹앱", 
        theme=gr.themes.Soft(),
        css="""
        /* 최적화된 웹폰트 임포트 - 2개 폰트만 사용 */
        /* NOTE: Constructable Stylesheets 제한으로 @import 금지. 시스템 폰트+fallback 사용. */
    
        /* 통합된 폰트 설정 */
        *, body, .gr-box, .gr-form, .gr-panel, .gr-input, .gr-textbox {
            font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
            font-weight: 400;
            line-height: 1.5;
        }
    
        /* 제목과 버튼 */
        h1, h2, h3, h4, h5, h6, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-button {
            font-family: 'Noto Sans KR', sans-serif !important;
            font-weight: 600;
        }
    
        /* 코드와 데이터 (모노스페이스) */
        code, pre, .gr-code, .dataframe, .gr-dataframe td, .gr-number {
            font-family: 'Fira Code', 'Consolas', monospace !important;
            font-weight: 400;
        }
    
        .analysis-output { 
            max-height: 800px; 
            overflow-y: auto; 
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 8px;
        }
        .favorites-display {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
        }
        .export-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
    
        /* 회복단계별 색상 코딩 */
        .recovery-stage-sweet-spot {
            background: linear-gradient(135deg, #d4f4dd, #c8f4de);
            border-left: 4px solid #28a745;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
        }
    
        .recovery-stage-overheated {
            background: linear-gradient(135deg, #fff3cd, #fdf6ce);
            border-left: 4px solid #fd7e14;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
        }
    
        .recovery-stage-extreme {
            background: linear-gradient(135deg, #f8d7da, #f9dbde);
            border-left: 4px solid #dc3545;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
        }
    
        .recovery-stage-bottom {
            background: linear-gradient(135deg, #e2e3e5, #e9ecef);
            border-left: 4px solid #6c757d;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
        }
    
        /* 경고 및 알림 스타일 */
        .alert-success {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
    
        .alert-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
    
        .alert-danger {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
    
        /* 파일 테이블 행별 하이라이트 */
        .dataframe tr:hover {
            background-color: #f8f9fa !important;
        }
    
        /* Sweet Spot 하이라이트 */
        .sweet-spot-highlight {
            background: linear-gradient(90deg, #e8f5e8, #f0f8f0);
            font-weight: bold;
        }
    
        /* 과열 구간 하이라이트 */
        .overheated-highlight {
            background: linear-gradient(90deg, #fff8e8, #fffaf0);
            font-weight: bold;
        }
    
        /* 필터 옵션 스타일 */
        .filter-panel {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border: 1px solid #e9ecef;
        }
    
        /* 상태 메시지 애니메이션 */
        .status-message {
            transition: all 0.3s ease;
        }
    
        /* 버튼 호버 효과 강화 */
        .gr-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        """
    ) as interface:
    
        # 상태 변수
        favorites_state = gr.State(value=initial_favorites)
    
        # 헤더
        gr.Markdown("""
        # 📊 AI 주식 분석 웹앱
        *다축 가중치 조정 기반 주식 분석 시스템*
        """)

        with gr.Accordion("⚙️ 분석 모듈(엔진) 설정", open=False):
            gr.Markdown("기본 엔진은 `pd.py`이며, `Purple2.1.py` 등 다른 분석 스크립트로 즉시 전환할 수 있습니다.")
            module_dropdown = gr.Dropdown(
                choices=app.available_modules,
                value=app.current_module,
                label="분석 모듈 선택 (엔진 교체)"
            )
            module_apply_btn = gr.Button("엔진 교체/저장", variant="primary")
            module_status = gr.Markdown("")
            module_info_md = gr.Markdown(app.get_current_module_info())
    
        with gr.Tabs():
            # 📡 Realtime Market Regime (FMP)
            with gr.Tab("📡 Realtime Regime"):
                # Unified execution modes: exactly three
                EXEC_CHOICES = ["T+0 종가(LOC)", "T+1 시초가", "T+1 종가"]

                def _decode_exec_choice(label: Optional[str]) -> tuple[int, str]:
                    """Map unified label to (delay_days, price_mode)."""
                    if not label:
                        return 1, "open"
                    s = str(label)
                    if "T+0" in s:
                        return 0, "close"
                    if "시초가" in s:
                        return 1, "open"
                    return 1, "close"

                gr.Markdown("## FMP 기반 실시간 레짐 (Classic · FLL-STAB · FLL-Fusion)")
                with gr.Row():
                    window_dd = gr.Dropdown(choices=[20, 30, 60], value=30, label="창(Window)", scale=1)
                    use_rt = gr.Checkbox(value=True, label="실시간 가격 사용")
                    exec_mode_dd = gr.Dropdown(choices=EXEC_CHOICES, value="T+1 시초가", label="체결 모드", scale=1)
                    refresh_btn = gr.Button("🔄 새로고침", variant="primary")
                premarket_box = gr.Textbox(
                    label="프리마켓 환산가(JSON, 예: {\"QQQ\": 448.2, \"SPY\": 552.0, \"IWM\": 228.5, \"GLD\": 221.7, \"BTC-USD\": 71234.0, \"TLT\": 91.20})",
                    placeholder="입력 시 오늘자 종가를 해당 값으로 덮어씁니다. 비워두면 FMP 실시간/전일 데이터 사용.",
                    lines=2,
                )
                auto_pre = gr.Checkbox(value=True, label="프리마켓/애프터 자동 반영(FMP)")
                with gr.Row():
                    range_dd = gr.Dropdown(choices=[30, 60, 180, 360, "맞춤"], value=180, label="표시 기간(일) 또는 맞춤", scale=1)
                    start_box = gr.Textbox(label="맞춤 시작일(YYYY-MM-DD)", placeholder="예: 2024-01-01", scale=1)
                    end_box = gr.Textbox(label="맞춤 종료일(YYYY-MM-DD)", placeholder="예: 2025-10-30", scale=1)
                rt_summary = gr.Markdown("실시간 요약 준비 중…")
                rt_narrative = gr.Markdown(visible=True)
                ewdr_md = gr.Markdown(visible=True)
                rt_transitions = gr.Markdown(visible=True)
                reg_fig = gr.Plot(label="레짐 상태", show_label=True)
                stab_fig = gr.Plot(label="Stability & Sub-Indices", show_label=True)
                bt_fig = gr.Plot(label="백테스트", show_label=True)
                bt_stats = gr.Markdown(visible=True)
                with gr.Row():
                    bt_csv_cls_btn = gr.DownloadButton(label="📥 Classic CSV", value=None)
                    bt_csv_stab_btn = gr.DownloadButton(label="📥 FLL-STAB CSV", value=None)
                    bt_csv_fus_btn = gr.DownloadButton(label="📥 FLL-Fusion CSV", value=None)

                BASE_SYMBOL = "QQQ"

                # === 공용 해설/인덱스 헬퍼는 market_analysis.insights 로 이동 ===
            
                def _pick_fusion_snapshot(payload: Dict[str, Any], mode: str) -> Tuple[Dict[str, Any], List[str], int]:
                    """
                    mode: 'rt' 또는 'y'
                    반환: (fusion_obj, dates, idx_last)
            
                    규칙:
                      - 'rt': payload['fusion'] 우선. dates는 fusion.dates → payload.dates.
                      - 'y' : payload['fusion_prev'] 있으면 그걸 사용.
                              없으면 '전일 날짜'로 rt 달력에서 인덱스를 찾아 대체.
                    """
                    asof = payload.get("asof", {}) or {}
                    fu_rt = payload.get("fusion", {}) or {}
                    fu_y  = payload.get("fusion_prev", {}) or {}
            
                    dates_rt = fu_rt.get("dates") or payload.get("dates") or []
                    dates_y  = fu_y.get("dates") or []
            
                    if mode == "rt":
                        dates = dates_rt if dates_rt else []
                        idx_last = len(dates) - 1
                        return fu_rt, dates, max(idx_last, -1)
            
                    # mode == 'y'
                    if dates_y and isinstance(dates_y, list):
                        idx_last = len(dates_y) - 1
                        return fu_y, dates_y, max(idx_last, -1)
            
                    # 전일 날짜로 rt 달력에서 위치 찾기
                    if not isinstance(dates_rt, list) or not dates_rt:
                        return fu_rt, [], -1
            
                    prev_date = asof.get("prev_close_date")
                    if not prev_date:
                        prev_date = dates_rt[-2] if len(dates_rt) >= 2 else dates_rt[-1]
            
                    try:
                        j = dates_rt.index(prev_date)
                    except ValueError:
                        j = len(dates_rt) - 2 if len(dates_rt) >= 2 else len(dates_rt) - 1
            
                    return fu_rt, dates_rt, j
                # === INSERT END ===
            

                #중복? def _format_realtime_summary

                def _format_realtime_summary(payload: Dict[str, Any]) -> str:
                    try:
                        asof = payload.get("asof", {}) or {}
                        fu_rt = payload.get("fusion", {}) or {}
                        fu_y  = payload.get("fusion_prev", {}) or {}
                        cmod  = payload.get("classic", {}) or {}
                        fmod  = payload.get("ffl_stab", {}) or {}
            
                        dates_rt = fu_rt.get("dates") or payload.get("dates") or []
                        if not isinstance(dates_rt, list) or not dates_rt:
                            return "❌ 데이터가 없습니다."
                        n_rt = len(dates_rt)
            
                        # 표시용 인덱스(T)
                        _, idx_eff_raw, idx_meta = resolve_effective_index(dates_rt, asof, base_symbol=BASE_SYMBOL)
                        idx_rt = max(0, min(idx_eff_raw, n_rt - 1))
            
                        def pick_tail(arr: Optional[List[Any]], base_len: int, i: int) -> Optional[float]:
                            if not isinstance(arr, list) or base_len <= 0 or i < 0:
                                return None
                            m = len(arr)
                            if m == 0:
                                return None
                            if m >= base_len:
                                j = (m - base_len) + i
                                return arr[j] if 0 <= j < m else None
                            left_pad = base_len - m
                            if i < left_pad:
                                return None
                            j = i - left_pad
                            return arr[j] if 0 <= j < m else None
            
                        def lab_from_val(v: Optional[float]) -> str:
                            try:
                                iv = int(v) if v is not None else None
                                return "Risk-On" if iv and iv > 0 else ("Risk-Off" if iv and iv < 0 else "Neutral")
                            except Exception:
                                return "N/A"
            
                        def fmt(x, d=3):
                            try: return f"{float(x):.{d}f}"
                            except Exception: return "N/A"
            
                        def fmt_px(x):
                            try: return f"${float(x):.2f}"
                            except Exception: return "$N/A"
            
                        # === RT ===
                        st_rt = fu_rt.get("state") or []
                        sc_rt = fu_rt.get("score") or []
                        wC_rt = fu_rt.get("wTA") or []
                        wF_rt = fu_rt.get("wFlow") or []
                        diag_rt = fu_rt.get("diag") or {}
                        fu_state_rt = lab_from_val(pick_tail(st_rt, n_rt, idx_rt))
                        fu_score_rt = pick_tail(sc_rt, n_rt, idx_rt)
                        fu_wC_rt    = pick_tail(wC_rt, n_rt, idx_rt)
                        fu_wF_rt    = pick_tail(wF_rt, n_rt, idx_rt)
                        if isinstance(diag_rt, dict):
                            diag_wta_rt = diag_rt.get("wTA")
                            if isinstance(diag_wta_rt, (int, float)):
                                fu_wC_rt = diag_wta_rt
                                fu_wF_rt = 1.0 - diag_wta_rt

                        # === 전일(Y) ===
                        diag_y = {}
                        if isinstance(fu_y.get("dates"), list) and fu_y.get("dates"):
                            dates_y = fu_y["dates"]; n_y = len(dates_y); idx_y = n_y - 1
                            st_y = fu_y.get("state") or []; sc_y = fu_y.get("score") or []
                            wC_y = fu_y.get("wTA") or []; wF_y = fu_y.get("wFlow") or []
                            diag_y = fu_y.get("diag") or {}
                            fu_state_y = lab_from_val(pick_tail(st_y, n_y, idx_y))
                            fu_score_y = pick_tail(sc_y, n_y, idx_y)
                            fu_wC_y    = pick_tail(wC_y, n_y, idx_y)
                            fu_wF_y    = pick_tail(wF_y, n_y, idx_y)
                            date_y = dates_y[idx_y]
                        else:
                            prev_date = asof.get("prev_close_date") or (dates_rt[idx_rt - 1] if idx_rt - 1 >= 0 else dates_rt[idx_rt])
                            try:
                                idx_y0 = dates_rt.index(prev_date)
                            except ValueError:
                                idx_y0 = idx_rt - 1 if idx_rt - 1 >= 0 else idx_rt
                            idx_y = max(0, min(idx_y0, n_rt - 1))
                            fu_state_y = lab_from_val(pick_tail(st_rt, n_rt, idx_y))
                            fu_score_y = pick_tail(sc_rt, n_rt, idx_y)
                            fu_wC_y    = pick_tail(wC_rt, n_rt, idx_y)
                            fu_wF_y    = pick_tail(wF_rt, n_rt, idx_y)
                            date_y     = dates_rt[idx_y]
                            diag_y     = diag_rt

                        if isinstance(diag_y, dict):
                            diag_wta_y = diag_y.get("wTA")
                            if isinstance(diag_wta_y, (int, float)):
                                fu_wC_y = diag_wta_y
                                fu_wF_y = 1.0 - diag_wta_y
            
                        # === 가격(QQQ) ===
                        bench_close = payload.get("series_bench")
                        if not isinstance(bench_close, list):
                            bench_close = (payload.get("series", {}) or {}).get("QQQ")
                        bench_open  = payload.get("series_bench_open")
                        if not isinstance(bench_open, list):
                            bench_open = (payload.get("series_open", {}) or {}).get("QQQ")
            
                        def tail_num(arr: Optional[List[Any]], base_len: int, i: int) -> Optional[float]:
                            v = pick_tail(arr, base_len, i)
                            try: return float(v) if v is not None else None
                            except Exception: return None
            
                        q_rt = tail_num(bench_close, n_rt, idx_rt)
                        q_y  = tail_num(bench_close, (len(fu_y.get("dates")) if fu_y.get("dates") else n_rt), idx_y)
            
                        intraday = bool(asof.get("intraday_base_applied"))
                        if (not intraday) and dates_rt[idx_rt] == asof.get("today_utc"):
                            q_rt = tail_num(bench_open, n_rt, idx_rt) or q_rt
            
                        # Classic / STAB
                        def pick_mod(mod: Dict[str, Any], n_base: int, i: int):
                            st  = lab_from_val(pick_tail(mod.get("state"), n_base, i))
                            sc  = fmt(pick_tail(mod.get("score"), n_base, i))
                            mm  = fmt(pick_tail(mod.get("mm"), n_base, i))
                            jn  = fmt(pick_tail(mod.get("fflFlux"), n_base, i))
                            fint= fmt(pick_tail(mod.get("fluxIntensity"), n_base, i))
                            return st, sc, mm, jn, fint
            
                        c_state, c_score, _, _, _ = pick_mod(cmod, n_rt, idx_rt)
                        f_state, f_score, f_mm, f_jn, f_fint = pick_mod(fmod, n_rt, idx_rt)
            
                        engine = fu_rt.get("engine") or asof.get("fusion_engine") or "newgate"
                        preset = fu_rt.get("preset") or payload.get("fusion_preset") or "default"
                        fusion_last = asof.get("fusion_last_date") or dates_rt[idx_rt]
            
                        ts_max = asof.get("quote_ts_max")
                        ts_sfx = "" if (isinstance(ts_max, str) and ts_max.strip().upper().endswith("UTC")) else " UTC"
                        basis  = "실시간 가격" if intraday else "장 마감가"
            
                        lines = [
                            "### 📡 Realtime Regime",
                            f"- 데이터 기준(ET): {basis} (ET {fusion_last})" + (f" · quote_ts_max={ts_max}{ts_sfx}" if ts_max else ""),
                            f"- Fusion(현재 기준) 엔진={engine} · preset={preset} · 레짐={fu_state_rt} · score={fmt(fu_score_rt)} · wTA={fmt(fu_wC_rt,2)} · wFlow={fmt(fu_wF_rt,2)}",
                            f"- Fusion(전일 종가 기준 ET {date_y}) · 레짐={fu_state_y} · score={fmt(fu_score_y)} · wTA={fmt(fu_wC_y,2)} · wFlow={fmt(fu_wF_y,2)}",
                            f"- QQQ as-of({fusion_last}): {fmt_px(q_rt)}",
                            f"- QQQ 전일 종가({date_y}): {fmt_px(q_y)}",
                            f"- Classic: {c_state} · score {c_score}",
                            f"- FLL-STAB: {f_state} · score {f_score} · Absorption(mm) {f_mm} · J_norm {f_jn} · FINT {f_fint}",
                        ]
                        try:
                            src = (asof or {}).get("override_source")
                            if isinstance(src, str) and src:
                                lines[1] = lines[1] + f" · source={src}"
                        except Exception:
                            pass
                        if idx_meta.get("used_fallback"):
                            reason = idx_meta.get("reason") or "실시간 호가 미수신"
                            lines.append(f"- ⚠️ {reason} → 전일 데이터 기준")
            
                        return "\n".join(lines)
                    except Exception as e:
                        return f"❌ 요약 생성 실패: {e}"
                    
                # 시장 해설은 market_analysis.insights.build_market_narrative 사용
                def _plot_regime_states(payload: Dict[str, Any]):
                    return app._plot_regime_states(payload)

                def _plot_stability(payload: Dict[str, Any]):
                    return app._plot_stability(payload)

                def _plot_backtest(payload: Dict[str, Any], delay_days: int, price_mode: str):
                    dates = app._effective_dates(payload)
                    n = len(dates)
                    state_stab_full = payload.get("ffl_stab", {}).get("state", [])
                    state_cls_full = payload.get("classic", {}).get("state", [])
                    state_fus_full = payload.get("fusion", {}).get("state", [])
                    def tail(arr):
                        if not isinstance(arr, list):
                            return []
                        return arr[-n:] if len(arr) >= n else [0] * (n - len(arr)) + arr
                    state_stab = tail(state_stab_full)
                    state_cls = tail(state_cls_full)
                    state_fus = tail(state_fus_full)
                    series_map = payload.get("series", {}) or {}
                    series_open_map = payload.get("series_open", {}) or {}
                    qqq_close = series_map.get("QQQ")
                    qqq_open = series_open_map.get("QQQ")
                    tqqq_close = series_map.get("TQQQ")
                    tqqq_open = series_open_map.get("TQQQ")
                    requested_mode = "open" if price_mode == "open" else "close"
                    has_open = isinstance(qqq_open, list) and len(qqq_open) >= 2
                    price_mode = "open" if (requested_mode == "open" and has_open) else "close"
                    qqq_segment = qqq_open if price_mode == "open" else qqq_close
                    tqqq_segment = tqqq_open if price_mode == "open" and isinstance(tqqq_open, list) else tqqq_close
                    # 길이 보정: dates(n)에 맞춰 가격 시리즈는 n+1 또는 최소 n로 맞춘다
                    def trim_px(px, extra=0):
                        if not isinstance(px, list):
                            return []
                        need = n + extra
                        if len(px) >= need:
                            return px[-need:]
                        return px
                    qqq_segment = trim_px(qqq_segment, extra=1)
                    tqqq_segment = trim_px(tqqq_segment, extra=1)
                    def write_tmp_csv(bdata: Optional[bytes], tag: str):
                        if not bdata:
                            return None
                        fd, path = tempfile.mkstemp(prefix=f"backtest_{tag}_{price_mode}_delay{delay_days}_", suffix=".csv")
                        with os.fdopen(fd, "wb") as f:
                            f.write(bdata)
                        return path
                    if not dates or not state_stab or not qqq_segment:
                        empty_fig = go.Figure() if go else (plt.figure() if plt else None)
                        return empty_fig, "데이터가 부족합니다.", None, None, None

                    def compute_returns(px: List[float]) -> List[float]:
                        out: List[float] = []
                        if not isinstance(px, list) or len(px) < 2:
                            return out
                        for i in range(1, len(px)):
                            a, b = px[i - 1], px[i]
                            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a != 0:
                                out.append(b / a - 1.0)
                            else:
                                out.append(0.0)
                        return out

                    rets = compute_returns(qqq_segment)
                    if len(rets) < n:
                        rets.extend([0.0] * (n - len(rets)))
                    else:
                        rets = rets[-n:]
                    rets_strategy = compute_returns(tqqq_segment)
                    if len(rets_strategy) < n:
                        rets_strategy.extend([0.0] * (n - len(rets_strategy)))
                    else:
                        rets_strategy = rets_strategy[-n:]
                    neutral_weight = 0.40

                    def build_eq(state_arr: List[int]):
                        exec_state = []
                        for i in range(n):
                            j = i - delay_days
                            exec_state.append(state_arr[j] if j >= 0 else 0)
                        eq = []
                        s = 1.0
                        strat_rets = []
                        for i in range(n):
                            r = rets[i] if i < len(rets) else 0.0
                            rs = rets_strategy[i] if i < len(rets_strategy) else r
                            if exec_state[i] > 0:
                                strat_rets.append(rs)
                                s *= (1.0 + rs)
                            elif exec_state[i] < 0:
                                strat_rets.append(0.0)
                                s *= 1.0
                            else:
                                strat_rets.append(neutral_weight * rs)
                                s *= (1.0 + neutral_weight * rs)
                            eq.append(s)
                        return eq, exec_state, strat_rets

                    eq_stab, exec_stab, rets_stab = build_eq(state_stab)
                    eq_cls, exec_cls, rets_cls = build_eq(state_cls)
                    eq_fus, exec_fus, rets_fus = build_eq(state_fus)

                    eq_b = []
                    s = 1.0
                    for i in range(n):
                        r = rets[i] if i < len(rets) else 0.0
                        s *= (1.0 + r)
                        eq_b.append(s)
                    eq_asset = []
                    a = 1.0
                    for i in range(n):
                        rs = rets_strategy[i] if i < len(rets_strategy) else 0.0
                        a *= (1.0 + rs)
                        eq_asset.append(a)
                    prices_bench = []
                    prices_strategy = []
                    def price_at(series, idx):
                        if not isinstance(series, list) or not series:
                            return None
                        target = idx + 1
                        if target < len(series):
                            return series[target]
                        return series[-1]
                    for i in range(n):
                        prices_bench.append(price_at(qqq_segment, i))
                        prices_strategy.append(price_at(tqqq_segment, i))

                    def build_csv_bytes(state_arr, exec_arr, strat_rets, eq_arr, eq_asset_arr, price_b_arr, price_s_arr, tag: str):
                        buf = io.StringIO()
                        buf.write("date,regime,executed,ret_bench,ret_strategy,price_qqq,price_tqqq,eq_strategy,eq_benchmark,eq_tqqq\n")
                        for i in range(len(dates)):
                            d = dates[i]
                            reg = state_arr[i] if i < len(state_arr) else 0
                            exe = exec_arr[i] if i < len(exec_arr) else 0
                            rb = rets[i] if i < len(rets) else 0.0
                            rs = strat_rets[i] if i < len(strat_rets) else 0.0
                            es = eq_arr[i] if i < len(eq_arr) else 1.0
                            eb = eq_b[i] if i < len(eq_b) else 1.0
                            ea = eq_asset_arr[i] if i < len(eq_asset_arr) else 1.0
                            pb = price_b_arr[i] if i < len(price_b_arr) and price_b_arr[i] is not None else ""
                            ps = price_s_arr[i] if i < len(price_s_arr) and price_s_arr[i] is not None else ""
                            pb_val = f"{pb:.4f}" if isinstance(pb, (int, float)) else ""
                            ps_val = f"{ps:.4f}" if isinstance(ps, (int, float)) else ""
                            buf.write(f"{d},{reg},{exe},{rb:.8f},{rs:.8f},{pb_val},{ps_val},{es:.8f},{eb:.8f},{ea:.8f}\n")
                        return write_tmp_csv(buf.getvalue().encode("utf-8"), tag)

                    csv_cls = build_csv_bytes(state_cls, exec_cls, rets_cls, eq_cls, eq_asset, prices_bench, prices_strategy, "classic")
                    csv_stab = build_csv_bytes(state_stab, exec_stab, rets_stab, eq_stab, eq_asset, prices_bench, prices_strategy, "fll_stab")
                    csv_fus = build_csv_bytes(state_fus, exec_fus, rets_fus, eq_fus, eq_asset, prices_bench, prices_strategy, "fll_fusion")
                    if go is not None:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dates, y=eq_cls, mode="lines", name="Classic"))
                        fig.add_trace(go.Scatter(x=dates, y=eq_stab, mode="lines", name="FLL-STAB"))
                        fig.add_trace(go.Scatter(x=dates, y=eq_fus, mode="lines", name="FLL-Fusion"))
                        fig.add_trace(go.Scatter(x=dates, y=eq_b, mode="lines", name="벤치(QQQ)"))
                        if eq_asset:
                            fig.add_trace(go.Scatter(x=dates, y=eq_asset, mode="lines", name="벤치(TQQQ)", line=dict(dash="dot")))
                        fig.update_layout(height=360, legend=dict(orientation="h"))
                        def last(x):
                            return x[-1] if x else 1.0
                        mode_label = ("T+0 종가(LOC)" if delay_days == 0 else f"T+1 {'시초가' if price_mode=='open' else '종가'}")
                        rng = f" · 기간: {dates[0]} ~ {dates[-1]}" if dates else ""
                        stats = f"체결({mode_label}) · 누적: Classic {last(eq_cls):.3f} · STAB {last(eq_stab):.3f} · Fusion {last(eq_fus):.3f} · 벤치(QQQ) {last(eq_b):.3f}{rng}"
                        if eq_asset:
                            stats += f" · 벤치(TQQQ) {last(eq_asset):.3f}"
                        return fig, stats, csv_cls, csv_stab, csv_fus
                    if plt is None:
                        return None, "", None, None, None
                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    ax.plot(dates, eq_cls, label="Classic")
                    ax.plot(dates, eq_stab, label="FLL-STAB")
                    ax.plot(dates, eq_fus, label="FLL-Fusion")
                    ax.plot(dates, eq_b, label="벤치(QQQ)")
                    if eq_asset:
                        ax.plot(dates, eq_asset, label="벤치(TQQQ)", linestyle=":")
                    ax.legend(loc="upper center", ncol=5)
                    ax.set_title("Backtest")
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    def last(x):
                        return x[-1] if x else 1.0
                    mode_label = ("T+0 종가(LOC)" if delay_days == 0 else f"T+1 {'시초가' if price_mode=='open' else '종가'}")
                    stats = f"체결({mode_label}) · 누적: Classic {last(eq_cls):.3f} · STAB {last(eq_stab):.3f} · Fusion {last(eq_fus):.3f} · 벤치(QQQ) {last(eq_b):.3f}"
                    if eq_asset:
                        stats += f" · 벤치(TQQQ) {last(eq_asset):.3f}"
                    return fig, stats, csv_cls, csv_stab, csv_fus

                def _apply_range(payload: Dict[str, Any], rng: Any, start_s: Optional[str], end_s: Optional[str]) -> Dict[str, Any]:
                    dates = payload.get("dates", [])
                    if not dates:
                        return payload
                    # 1) Non-custom range: keep last N days
                    if rng != "맞춤":
                        try:
                            n = int(rng)
                        except Exception:
                            n = 180
                        n = max(1, n)
                        i0 = max(0, len(dates) - n)
                        i1 = len(dates) - 1
                        indices = (i0, i1)
                    else:
                        # 2) Custom start/end
                        def norm(s):
                            if not s:
                                return None
                            s = str(s).strip()
                            return s if s else None
                        s0 = norm(start_s)
                        s1 = norm(end_s)
                        def in_range(d):
                            if s0 and d < s0:
                                return False
                            if s1 and d > s1:
                                return False
                            return True
                        idxs = [i for i, d in enumerate(dates) if in_range(d)]
                        if not idxs:
                            return payload
                        indices = (idxs[0], idxs[-1])
                    i0, i1 = indices
                    def slice_list(a):
                        return a[i0:i1+1] if isinstance(a, list) else a
                    pl = dict(payload)
                    pl["dates"] = slice_list(dates)
                    # classic
                    if "classic" in pl:
                        c = dict(pl["classic"])
                        for k in ["score", "state"]:
                            if isinstance(c.get(k), list):
                                c[k] = slice_list(c[k])
                        pl["classic"] = c
                    # ffl_stab
                    if "ffl_stab" in pl:
                        f = dict(pl["ffl_stab"])
                        for k, v in list(f.items()):
                            if isinstance(v, list) and len(v) == len(dates):
                                f[k] = slice_list(v)
                        pl["ffl_stab"] = f
                    # fusion
                    if "fusion" in pl:
                        fu = dict(pl["fusion"])
                        # include 'dates' to ensure range slicing applies to fusion calendar
                        for k in ["dates", "score", "state", "executed_state", "wTA", "wFlow"]:
                            if isinstance(fu.get(k), list):
                                fu[k] = slice_list(fu[k])
                        pl["fusion"] = fu
                    # stability/sub
                    for k in ["stability", "smoothed", "delta"]:
                        if isinstance(pl.get(k), list) and len(pl[k]) == len(dates):
                            pl[k] = slice_list(pl[k])
                    if "sub" in pl and isinstance(pl["sub"], dict):
                        sub = dict(pl["sub"])
                        for sk in ["stockCrypto", "traditional", "safeNegative"]:
                            if isinstance(sub.get(sk), list) and len(sub[sk]) == len(dates):
                                sub[sk] = slice_list(sub[sk])
                        pl["sub"] = sub
                    # series for backtest: account for leading offset (series often start at window-1)
                    total_dates = len(dates)
                    def slice_series(arr: List[float]) -> List[float]:
                        if not isinstance(arr, list):
                            return []
                        offset = max(0, total_dates - len(arr))
                        s = max(0, i0 - offset)
                        # +2 to keep an extra point for returns alignment
                        e = min(len(arr), i1 - offset + 2)
                        if s >= e:
                            return []
                        return arr[s:e]
                    if isinstance(pl.get("series"), dict):
                        out_series = {}
                        for sym, data in (payload.get("series") or {}).items():
                            if isinstance(data, list):
                                out_series[sym] = slice_series(data)
                        if out_series:
                            pl["series"] = out_series
                    if isinstance(pl.get("series_open"), dict):
                        out_open = {}
                        for sym, data in (payload.get("series_open") or {}).items():
                            if isinstance(data, list):
                                out_open[sym] = slice_series(data)
                        if out_open:
                            pl["series_open"] = out_open
                    # Maintain bench series for downstream alignment (match AutoTrade2)
                    bench_close = None
                    bench_open = None
                    if isinstance(pl.get("series"), dict):
                        bench_close = pl["series"].get("QQQ") or pl["series"].get("SPY")
                    if isinstance(pl.get("series_open"), dict):
                        bench_open = pl["series_open"].get("QQQ") or pl["series_open"].get("SPY")
                    if bench_close is not None:
                        pl["series_bench"] = bench_close
                    if bench_open is not None:
                        pl["series_bench_open"] = bench_open
                    return pl

                def _format_ewdr(payload: Dict[str, Any]) -> str:
                    try:
                        fu = payload.get("fusion", {}) or {}
                        diag = fu.get("diag") or {}
                        if not diag:
                            return ""
                        def fmt(x, d=3):
                            try:
                                return f"{float(x):.{d}f}"
                            except Exception:
                                return "N/A"
                        # --- Risk assessment (EW/DR pre-warning) ---
                        ew = diag.get("EW") or {}
                        dr = diag.get("DR") or {}
                        cr = diag.get("CR") or {}
                        shock = (diag.get("Shock") or {}).get("active")
                        ew_hits = int(ew.get("count") or 0)
                        dr_hits = int(dr.get("count") or 0)
                        z_chi = float(diag.get("z_chi") or 0.0)
                        z_eta = float(diag.get("z_eta") or 0.0)
                        z_R   = float(diag.get("z_R") or 0.0)
                        z_dR  = float(diag.get("z_dR") or 0.0)
                        tilt_z = float(diag.get("tilt_z") or 0.0)
                        FQI = float(diag.get("FQI") or 0.0)
                        TFI = float(diag.get("TFI") or 0.0)
                        cap_ew = float(diag.get("ew_cap") or 1.0)
                        cap_dr = float(diag.get("dr_cap") or 1.0)
                        cap_cr = float(diag.get("cr_cap") or 1.0)
                        cap_sh = float(diag.get("shock_cap") or 1.0)
                        gate_cap = float(diag.get("gate_cap") or min(cap_ew, cap_dr, cap_cr, cap_sh, 1.0))
                        # Prefer narrative refs if available for consistency with insights
                        try:
                            nar = payload.get("_market_narrative") or {}
                            grefs = (nar.get("refs") or {}).get("gates") or {}
                            if grefs:
                                gate_cap = float(grefs.get("gate_cap") or gate_cap)
                                cap_ew = float(((grefs.get("EW") or {}).get("cap")) or cap_ew)
                                cap_dr = float(((grefs.get("DR") or {}).get("cap")) or cap_dr)
                                cap_cr = float(((grefs.get("CR") or {}).get("cap")) or cap_cr)
                                cap_sh = float(((grefs.get("Shock") or {}).get("cap")) or cap_sh)
                        except Exception:
                            pass
                        # Heuristic thresholds
                        severe = (
                            bool(shock) or cap_sh < 1.0 or
                            ew_hits >= 2 or dr_hits >= 2 or
                            z_dR >= 1.10 or z_eta >= 0.90 or z_R >= 0.90 or
                            (FQI <= -0.25 and TFI <= -0.50)
                        )
                        caution = (
                            (not severe) and (
                                cap_ew < 1.0 or cap_dr < 1.0 or cap_cr < 1.0 or
                                abs(z_chi) >= 0.60 or z_R >= 0.60 or z_eta >= 0.60 or z_dR >= 0.80 or
                                tilt_z <= -0.40 or FQI < 0.0
                            )
                        )
                        if severe:
                            risk_label = "🚨 심각"
                        elif caution:
                            risk_label = "⚠️ 주의"
                        else:
                            risk_label = "✅ 정상"
                        lines = []
                        lines.append("### ⚠️ 사전(EW/DR) 리스크 지표")
                        # headline
                        reasons = []
                        if shock or cap_sh < 1.0:
                            reasons.append("Shock")
                        if cap_ew < 1.0:
                            reasons.append("EW")
                        if cap_dr < 1.0:
                            reasons.append("DR")
                        if cap_cr < 1.0:
                            reasons.append("CR")
                        if z_dR >= 0.8 or z_eta >= 0.8:
                            reasons.append("확산 급변/비대칭")
                        if FQI < 0.0:
                            reasons.append("Flow 약화")
                        if TFI < 0.0:
                            reasons.append("레짐 약세")
                        reason_s = (" · 트리거: " + ", ".join(reasons)) if reasons else ""
                        lines.append(f"- 현재 리스크 판단: {risk_label}{reason_s}")
                        lines.append(
                            f"- wTA {fmt(diag.get('wTA'),2)} · S {fmt(diag.get('S'))} · z(χ/η/R/ΔR) {fmt(diag.get('z_chi'))} / {fmt(diag.get('z_eta'))} / {fmt(diag.get('z_R'))} / {fmt(diag.get('z_dR'))}"
                        )
                        lines.append(
                            f"- 품질: FQI {fmt(diag.get('FQI'))} · TQI {fmt(diag.get('TQI'))} · FFQI {fmt(diag.get('FFQI'))} · tilt− z {fmt(diag.get('tilt_z'))} · TFI {fmt(diag.get('TFI'))} ({diag.get('regime_label','-')})"
                        )
                        conc = diag.get('CONC')
                        if conc is not None:
                            lines.append(f"- 일치도(CONC, 63d): {fmt(conc,3)}")
                        # Compute gate label consistently with sub-caps
                        active_gates = []
                        if cap_sh < 1.0:
                            active_gates.append('Shock')
                        if cap_dr < 1.0:
                            active_gates.append('DR')
                        if cap_ew < 1.0:
                            active_gates.append('EW')
                        if cap_cr < 1.0:
                            active_gates.append('CR')
                        gate_label = ' + '.join(active_gates) if active_gates else 'None'
                        lines.append(
                            f"- 게이트: {gate_label} (cap={fmt(gate_cap,2)} · EW {fmt(cap_ew,2)} · DR {fmt(cap_dr,2)} · CR {fmt(cap_cr,2)} · Shock {fmt(cap_sh,2)})"
                        )
                        # Add divergence hints using narrative metrics if available
                        try:
                            nar = payload.get("_market_narrative") or {}
                            metrics = nar.get("metrics") or {}
                            st = int((nar.get("state") if isinstance(nar.get("state"), (int, float)) else (fu.get("state") or [0])[-1]) or 0)
                            wTA_v = float(metrics.get("wTA") if metrics.get("wTA") is not None else diag.get("wTA") or 0.0)
                            delta_v = float(metrics.get("delta") if metrics.get("delta") is not None else 0.0)
                            zR_v = float(metrics.get("z_R") if metrics.get("z_R") is not None else z_R)
                            zdR_v = float(metrics.get("z_dR") if metrics.get("z_dR") is not None else z_dR)
                            FQI_v = float(metrics.get("FQI") if metrics.get("FQI") is not None else FQI)
                            TFI_v = float(metrics.get("TFI") if metrics.get("TFI") is not None else TFI)
                            anomalies = []
                            if st > 0 and (gate_cap < 0.95 or zR_v >= 0.9 or zdR_v >= 1.0 or TFI_v < 0.0 or FQI_v < 0.0):
                                anomalies.append("상승 vs 리스크 경고(게이트/동조화/품질)")
                            if st < 0 and ((wTA_v >= 0.60 and delta_v > 0) or (FQI_v > 0.10)):
                                anomalies.append("약세 vs 회복 단서(wTA/Δ/FQI)")
                            if anomalies:
                                lines.append(f"- 상충/특이점: {', '.join(anomalies)}")
                        except Exception:
                            pass
                        adv = diag.get("advice")
                        if isinstance(adv, str) and adv:
                            lines.append("")
                            lines.append(adv)
                        # Friendly glossary
                        lines.append("")
                        lines.append("#### ℹ️ 용어 설명(간단)")
                        lines.append("- S(불확실성): 확산/결합의 이상 징후 종합 점수. 높을수록 Flow 불확실·TA 비중↑")
                        lines.append("- wTA: TA 가중(0~1). 높을수록 TA 쪽 노출을 크게 반영")
                        lines.append("- z(χ): 확산 대비 드리프트 약화(↑는 추세 약화). z(η): 하/상방 확산의 비대칭(↑는 한쪽 쏠림)")
                        lines.append("- z(R), z(ΔR): 결합도/결합 급변. ↑면 동조화 커져 동시하락/급락 민감도↑")
                        lines.append("- FQI: Flow 신호의 단기 예측력(상관). + 우위면 Flow, 0이면 중립")
                        lines.append("- TQI/FFQI: 최근 성과(위험조정). TQI>FFQI면 TA 질 우세")
                        lines.append("- TFI/FFI: 오늘의 레짐 스코어. TFI≥0.5 & wTA≥0.6 → TA‑우위 / 반대면 Flow‑우위")
                        lines.append("- 게이트: EW/DR 조건 충족 시 노출 상한 적용(EW:≈2/3, DR:≈1/3), Shock: 급락일 당일 1/3")
                        return "\n".join(lines)
                    except Exception as e:
                        return f"❌ EW/DR 지표 생성 실패: {e}"
                    
                # 오토트레이드2와 맞춤
                def _format_transitions(payload: Dict[str, Any]) -> str:
                    """웹앱에서도 AutoTrade2와 동일한 전환 로그를 표시."""
                    try:
                        return build_recent_transition_markdown(
                            payload,
                            title="🌙 최근 전환 10회 (각 항목별 데이터 기준 + score/wTA/wFLOW)",
                            limit=10,
                        )
                    except Exception as e:
                        return f"❌ 전환 요약 생성 실패: {e}\n"

                    
                def _parse_override_json(text: Optional[str]) -> dict:
                    if not text:
                        return {}
                    try:
                        data = json.loads(text)
                        if isinstance(data, dict):
                            out = {}
                            for k, v in data.items():
                                try:
                                    out[str(k).upper()] = float(v)
                                except Exception:
                                    continue
                            return out
                    except Exception:
                        pass
                    return {}

                def _run_realtime(window_val: int, use_real: bool, exec_choice: str, rng: Any,
                                  start_s: Optional[str], end_s: Optional[str],
                                  premarket_json: Optional[str], auto_pre_enabled: bool = True):
                    empty_fig = (go.Figure() if go else (plt.figure() if plt else None))
                    sync_dates_local: Optional[List[Any]] = None
                    try:
                        payload = app._fetch_payload_via_autotrade2(
                            window_val=window_val,
                            use_real=use_real,
                            auto_override=bool(auto_pre_enabled),
                        )
                        try:
                            payload_prev = app._fetch_payload_via_autotrade2(
                                window_val=window_val,
                                use_real=False,
                            )
                        except Exception as close_err:
                            print(f"[WARN] close-mode payload fetch failed: {close_err}")
                            payload_prev = None
                        close_snapshot: Optional[Dict[str, Any]] = None
                        if payload_prev:
                            payload.setdefault("_fusion_prev_source", "close_payload")
                    except Exception as e:
                        return (f"❌ 데이터 로드 실패: {e}", "", "", "", empty_fig, empty_fig, empty_fig, "", None, None, None)

                    overrides_user = _parse_override_json(premarket_json)
                    if overrides_user and isinstance(payload, dict):
                        for target_key in ("series", "series_open"):
                            series_map = payload.get(target_key)
                            if not isinstance(series_map, dict):
                                continue
                            for sym, val in overrides_user.items():
                                sym_key = str(sym).upper()
                                arr = series_map.get(sym_key)
                                if isinstance(arr, list) and arr:
                                    try:
                                        arr[-1] = float(val)
                                    except Exception:
                                        continue
                        payload["_user_override_used"] = True
                        payload.setdefault("manifest", {})["manual_override"] = True

                    if isinstance(payload, dict):
                        payload.setdefault("manifest", {})["auto_override"] = bool(auto_pre_enabled)

                    # ③.5 달력 기준을 먼저 통일 (fusion.dates → payload.dates)
                    try:
                        fu_dates = (payload.get("fusion") or {}).get("dates")
                        if isinstance(fu_dates, list) and fu_dates:
                            payload["dates"] = list(fu_dates)
                    except Exception:
                        pass
                    if payload_prev and isinstance(payload_prev, dict):
                        try:
                            fu_dates_prev = (payload_prev.get("fusion") or {}).get("dates")
                            if isinstance(fu_dates_prev, list) and fu_dates_prev:
                                payload_prev["dates"] = list(fu_dates_prev)
                        except Exception:
                            pass

                    # ④ 범위 적용 (통일된 달력 기준으로 슬라이싱)
                    delay_val, price_mode = _decode_exec_choice(exec_choice)
                    payload = _apply_range(payload, rng, start_s, end_s)
                    if payload_prev:
                        payload_prev = _apply_range(payload_prev, rng, start_s, end_s)

                    # ⑤ 전일 스냅샷 주입 및 prev_close_date 설정
                    if payload_prev:
                        fusion_prev = payload_prev.get("fusion") or {}
                        if isinstance(fusion_prev, (dict, list)):
                            fusion_prev = copy.deepcopy(fusion_prev)
                        payload["fusion_prev"] = fusion_prev
                        prev_dates = payload["fusion_prev"].get("dates") or payload_prev.get("dates") or []
                        if prev_dates:
                            payload.setdefault("asof", {})["prev_close_date"] = prev_dates[-1]
                        else:
                            rt_dates = (payload.get("fusion") or {}).get("dates") or payload.get("dates") or []
                            if len(rt_dates) >= 2:
                                payload.setdefault("asof", {})["prev_close_date"] = rt_dates[-2]
                    else:
                        rt_dates = (payload.get("fusion") or {}).get("dates") or payload.get("dates") or []
                        if len(rt_dates) >= 2:
                            payload.setdefault("asof", {})["prev_close_date"] = rt_dates[-2]

                    # ⑤.5 모든 차트의 x축을 단일 달력으로 강제 동기화 (payload 기준)
                    try:
                        sync_dates = app._effective_dates(payload)
                        if isinstance(sync_dates, list) and sync_dates:
                            payload["dates"] = list(sync_dates)
                            sync_dates_local = list(sync_dates)
                    except Exception:
                        pass

                    # ⑥ 출력 생성
                    summary = _format_realtime_summary(payload)
                    narrative_data = build_market_narrative(payload, base_symbol=BASE_SYMBOL)
                    payload["_market_narrative"] = narrative_data
                    narrative = narrative_data.get("text", "")
                    transitions_md = _format_transitions(payload)
                    ewdr = _format_ewdr(payload)
                    reg = _plot_regime_states(payload)
                    stab = _plot_stability(payload)
                    # 백테스트 입력: T+0(LOC)은 실시간 payload, T+1(open/close)은 전일 스냅샷 사용
                    use_prev = (delay_val == 1)
                    payload_for_bt = payload_prev if (use_prev and payload_prev) else payload
                    if isinstance(payload_for_bt, dict) and sync_dates_local:
                        try:
                            payload_for_bt["dates"] = list(sync_dates_local)
                        except Exception:
                            pass
                    bt, stats, csv_cls, csv_stab, csv_fus = _plot_backtest(payload_for_bt, delay_val, price_mode)
            
                    return (summary, narrative, transitions_md, ewdr, reg, stab, bt, stats, csv_cls, csv_stab, csv_fus)


                refresh_btn.click(
                    _run_realtime,
                    inputs=[window_dd, use_rt, exec_mode_dd, range_dd, start_box, end_box, premarket_box, auto_pre],
                    outputs=[rt_summary, rt_narrative, rt_transitions, ewdr_md, reg_fig, stab_fig, bt_fig, bt_stats, bt_csv_cls_btn, bt_csv_stab_btn, bt_csv_fus_btn],
                )

                # run once on mount
                window_dd.change(
                    _run_realtime,
                    inputs=[window_dd, use_rt, exec_mode_dd, range_dd, start_box, end_box, premarket_box, auto_pre],
                    outputs=[rt_summary, rt_narrative, rt_transitions, ewdr_md, reg_fig, stab_fig, bt_fig, bt_stats, bt_csv_cls_btn, bt_csv_stab_btn, bt_csv_fus_btn],
                )
                use_rt.change(
                    _run_realtime,
                    inputs=[window_dd, use_rt, exec_mode_dd, range_dd, start_box, end_box, premarket_box, auto_pre],
                    outputs=[rt_summary, rt_narrative, rt_transitions, ewdr_md, reg_fig, stab_fig, bt_fig, bt_stats, bt_csv_cls_btn, bt_csv_stab_btn, bt_csv_fus_btn],
                )

            # 📊 확률 기반 시장 리포트 (Market Report)
            with gr.Tab("📊 확률 리포트"):
                gr.Markdown("""
                ## 확률 기반 전문 리포트
                - FMP 신호군(+SoT)을 결합한 가우시안 Naive Bayes Fusion으로 P(Up|x)를 산출합니다.
                - FMP_API_KEY가 없거나 네트워크 제한 시, SoT 기반 최소 리포트로 축소되어 출력됩니다.
                """)
                with gr.Row():
                    mr_h = gr.Slider(minimum=3, maximum=20, value=5, step=1, label="예측 지평(H, 영업일)", scale=1)
                    mr_use_rt = gr.Checkbox(value=True, label="실시간 SoT 사용", scale=1)
                    mr_sort = gr.Radio(choices=["절대값 순", "부호 순(양→음)"], value="절대값 순", label="드라이버 정렬", scale=1)
                    mr_btn = gr.Button("🧮 리포트 생성", variant="primary", scale=1)
                mr_help_controls = gr.Markdown("""
                **도움말**
                - 예측 지평(H): H영업일 뒤 수익률이 양일 확률(P(Up|H)). H가 길수록 신호는 완만·보수적.
                - 실시간 SoT: AutoTrade2 SoT를 즉시 계산해 벤치(RV)와 지표를 갱신.
                - 드라이버 정렬: 절대값(영향력 큰 순) 또는 부호(상승 기여 먼저)로 정렬.
                """)
                mr_conclusion = gr.Markdown(visible=True)
                mr_md = gr.Markdown("리포트를 생성하세요.")
                with gr.Row():
                    mr_gauge = gr.Plot(label="상승 확률 게이지(도넛)", show_label=True)
                    mr_drivers = gr.Plot(label="드라이버 Top5 (LLR)", show_label=True)
                mr_help_gauge = gr.Markdown("""
                **게이지·점선 안내**
                - 도넛은 P(Up|H)를 의미(녹=상승, 적=하락). 중앙의 %가 확률입니다.
                - 회색 점선 링은 ‘불확실성 대역’(피처 완결도 기반). 두꺼울수록 신뢰 낮음.
                - 드라이버(LLR): +는 상승 쪽 기여, −는 하락 쪽 기여. 길이는 영향력 크기.
                """)
                with gr.Row():
                    mr_spr1 = gr.Plot(label="10Y-3M", show_label=True)
                    mr_spr2 = gr.Plot(label="10Y-2Y", show_label=True)
                    mr_curv = gr.Plot(label="Curve", show_label=True)
                mr_help_spreads = gr.Markdown("""
                **스프레드/곡률 해석**
                - 10Y−3M: +면 단기금리 하락/완화, −면 초단기 역전(스트레스) 신호.
                - 10Y−2Y: +면 스티프닝(정상화), −면 장단기 역전(경기둔화 리스크).
                - Curve(30Y+3M−2×10Y): +면 정상화·경사 회복 경향.
                """)
                with gr.Row():
                    mr_sectors = gr.Plot(label="섹터 히트맵", show_label=True)
                mr_help_heatmap = gr.Markdown("""
                **섹터 히트맵**
                - 색: 1일 변화율(녹=상승, 적=하락), 면적: |변화율| 크기.
                - 무엇이 시장을 끌어올리는지/누르는지 한눈에 확인.
                """)
                with gr.Row():
                    mr_json_btn = gr.DownloadButton(label="💾 JSON 다운로드", value=None)
                mr_diag = gr.Markdown(visible=True)

                def _run_market_report(h: int, use_real: bool, sort_mode: str):
                    if generate_market_report is None:
                        empty = (go.Figure() if go else None)
                        return ("❌ market_report 모듈을 불러올 수 없습니다.", None, empty, empty, empty, empty, empty, empty)
                    try:
                        payload = app._fetch_payload_via_autotrade2(
                            window_val=30,
                            use_real=use_real,
                            auto_override=True,
                        )
                    except Exception as e:
                        empty = (go.Figure() if go else None)
                        return (f"❌ SoT 페치 실패: {e}", None, empty, empty, empty, empty, empty, empty)

                    def _feature_defs():
                        return {
                            "ADR": "Adv/Decl 비율(상승 종목 수 / 하락 종목 수)",
                            "Pct>MA50": "ETF(SPY/VOO/IVV) 보유종목 중 50일선 상회 비중",
                            "Pct>MA200": "ETF(SPY/VOO/IVV) 보유종목 중 200일선 상회 비중",
                            "NH/NL": "52주 신고/신저 비율(근접 기준)",
                            "RV20": "20일 실현변동성(연환산)",
                            "RV60": "60일 실현변동성(연환산)",
                            "SPR_10Y_3M": "미 국채 10년-3개월 스프레드",
                            "SPR_10Y_2Y": "미 국채 10년-2년 스프레드",
                            "CURVATURE": "수익률곡선 곡률(30Y+3M-2*10Y)",
                        }

                    def _build_gauge(prob: float, rep: dict):
                        if not go:
                            return None
                        try:
                            p = max(0.0, min(1.0, float(prob)))
                        except Exception:
                            p = 0.5
                        # Base donut
                        fig = go.Figure(
                            data=[
                                go.Pie(
                                    values=[p, 1 - p],
                                    labels=["P(Up)", "P(Down)"],
                                    hole=0.6,
                                    marker=dict(colors=["#2ecc71", "#e74c3c"]),
                                    textinfo="label+percent",
                                    sort=False,
                                )
                            ]
                        )
                        # Confidence band as dotted ring
                        try:
                            features = (rep.get("features") or {})
                            total = max(1, len(features))
                            avail = sum(1 for v in features.values() if isinstance(v, (int, float)) and not (v is None))
                            completeness = avail / total if total else 0.0
                            band = max(0.06, min(0.20, 0.18 - 0.10 * completeness))  # narrower with more features
                            plo = max(0.0, p - band / 2)
                            phi = min(1.0, p + band / 2)
                            segs = 72
                            vals = [1] * segs
                            colors = []
                            for i in range(segs):
                                a0 = i / segs
                                in_band = (a0 >= plo) and (a0 <= phi)
                                if in_band and (i % 2 == 0):
                                    colors.append("rgba(127,140,141,0.9)")  # dotted segments
                                else:
                                    colors.append("rgba(0,0,0,0)")
                            fig.add_trace(
                                go.Pie(
                                    values=vals,
                                    labels=[str(i) for i in range(segs)],
                                    hole=0.72,
                                    marker=dict(colors=colors),
                                    textinfo="none",
                                    sort=False,
                                )
                            )
                        except Exception:
                            pass
                        fig.update_layout(
                            showlegend=False,
                            height=320,
                            annotations=[dict(text=f"{p*100:.1f}%", x=0.5, y=0.5, font_size=22, showarrow=False)],
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        return fig

                    def _build_drivers_bar(drivers, mode: str):
                        if not go or not isinstance(drivers, list) or not drivers:
                            return None
                        defs = _feature_defs()
                        if mode.startswith("절대값"):
                            drivers = sorted(drivers, key=lambda t: abs(float(t[1])), reverse=True)
                        else:
                            # positive first by magnitude, then negatives by magnitude
                            pos = [(n, float(v)) for n, v in drivers if float(v) >= 0]
                            neg = [(n, float(v)) for n, v in drivers if float(v) < 0]
                            pos.sort(key=lambda t: abs(t[1]), reverse=True)
                            neg.sort(key=lambda t: abs(t[1]), reverse=True)
                            drivers = pos + neg
                        names = [str(n) for n, _ in drivers]
                        vals = [float(v) for _, v in drivers]
                        hover = [f"{n}: {defs.get(n, '')}<br>LLR={float(v):+.3f}" for n, v in drivers]
                        fig = go.Figure(
                            data=[go.Bar(
                                x=vals,
                                y=names,
                                orientation="h",
                                marker=dict(color=["#2ecc71" if v>=0 else "#e74c3c" for v in vals]),
                                hovertext=hover,
                                hovertemplate="%{hovertext}<extra></extra>",
                            )]
                        )
                        fig.update_layout(
                            height=320,
                            margin=dict(l=80, r=20, t=10, b=10),
                            xaxis=dict(title="기여(LLR)", zeroline=True, zerolinewidth=1, zerolinecolor="#7f8c8d"),
                            yaxis=dict(title="드라이버"),
                        )
                        return fig

                    def _build_curve(dates, values, title):
                        if not go or not isinstance(values, list) or len(values) < 2:
                            return None
                        try:
                            xs = dates if isinstance(dates, list) and len(dates) == len(values) else None
                            if xs is None:
                                xs = list(range(len(values)))
                            else:
                                coerced = []
                                for d in xs:
                                    if isinstance(d, (int, float)):
                                        coerced.append(d)
                                    else:
                                        try:
                                            coerced.append(datetime.fromisoformat(str(d)).date())
                                        except Exception:
                                            coerced.append(str(d))
                                xs = coerced
                            fig = go.Figure(
                                data=[
                                    go.Scatter(
                                        x=xs,
                                        y=values,
                                        mode="lines+markers",
                                        line=dict(color="#3498db", width=2),
                                        marker=dict(size=6, color="#1abc9c"),
                                    )
                                ]
                            )
                            fig.add_hline(y=0.0, line=dict(color="#7f8c8d", width=1, dash="dash"))
                            fig.update_layout(
                                title=title,
                                height=260,
                                margin=dict(l=40, r=10, t=40, b=40),
                                xaxis_title="날짜",
                                yaxis_title="수준",
                            )
                            return fig
                        except Exception:
                            return None
                    
                    def _build_sectors_treemap(sectors):
                        if not go or not isinstance(sectors, list) or not sectors:
                            return None
                        try:
                            labels = []
                            parents = []
                            values = []
                            colors = []
                            for it in sectors:
                                sec = it.get('sector') or it.get('name') or 'N/A'
                                chg = it.get('changesPercentage')
                                try:
                                    if isinstance(chg, str) and chg.endswith('%'):
                                        chg = float(chg[:-1])
                                    else:
                                        chg = float(chg)
                                except Exception:
                                    chg = None
                                labels.append(str(sec))
                                parents.append("")
                                # size by absolute performance (with floor)
                                size = abs(chg) if isinstance(chg, (int, float)) else 1.0
                                values.append(max(1e-3, size))
                                colors.append(chg if isinstance(chg, (int, float)) else 0.0)
                            fig = go.Figure(
                                go.Treemap(
                                    labels=labels,
                                    parents=parents,
                                    values=values,
                                    marker=dict(colors=colors, colorscale=[[0,"#e74c3c"],[0.5,"#f1c40f"],[1,"#2ecc71"]], colorbar=dict(title="%")),
                                    hovertemplate="%{label}<br>% 변화=%{color:.2f}%<extra></extra>",
                                )
                            )
                            fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=340)
                            return fig
                        except Exception:
                            return None
                    try:
                        rep = generate_market_report(horizon_days=int(h), sot_payload=payload)
                        md = rep.get("markdown") or "(빈 리포트)"
                        concl = rep.get("narrative") or ""
                        # write JSON temp file
                        import tempfile, json, os
                        fd, path = tempfile.mkstemp(prefix="market_report_", suffix=".json")
                        with os.fdopen(fd, "w", encoding="utf-8") as fp:
                            json.dump(rep, fp, ensure_ascii=False, indent=2)
                        p_up = float(((rep.get("prob") or {}).get("p_up")) or 0.5)
                        gauge = _build_gauge(p_up, rep)
                        drivers = rep.get("drivers") or []
                        drv_fig = _build_drivers_bar(drivers, sort_mode)
                        # curves
                        cts = (((rep.get("refs") or {}).get("ctx_fmp") or {}).get("curve_ts")) or {}
                        c_dates = cts.get("dates") or []
                        fig1 = _build_curve(c_dates, cts.get("spr_10y_3m"), "10Y-3M")
                        fig2 = _build_curve(c_dates, cts.get("spr_10y_2y"), "10Y-2Y")
                        fig3 = _build_curve(c_dates, cts.get("curvature"), "Curve")
                        # sectors treemap
                        secs = (((rep.get('refs') or {}).get('ctx_fmp') or {}).get('sectors')) or []
                        sec_fig = _build_sectors_treemap(secs)
                        # diagnostics
                        ctx_f = (rep.get('refs') or {}).get('ctx_fmp') or {}
                        qn = ctx_f.get('quotes_count')
                        srcs = ctx_f.get('sources') or []
                        diag = f"**데이터 진단**: sources={srcs} · quotes_count={qn}"
                        return (concl, md, path, gauge, drv_fig, fig1, fig2, fig3, sec_fig, diag)
                    except Exception as e:
                        empty = (go.Figure() if go else None)
                        return (f"❌ 리포트 생성 실패: {e}", f"❌ 리포트 생성 실패: {e}", None, empty, empty, empty, empty, empty, empty, f"❌ {e}")

                mr_btn.click(
                    _run_market_report,
                    inputs=[mr_h, mr_use_rt, mr_sort],
                    outputs=[mr_conclusion, mr_md, mr_json_btn, mr_gauge, mr_drivers, mr_spr1, mr_spr2, mr_curv, mr_sectors, mr_diag],
                )

            # 📈 확률 백테스트
            with gr.Tab("📈 확률 백테스트"):
                gr.Markdown("""
                ## 확률 히스토리 기반 QQQ 검증
                - 확률 리포트 실행 시 저장된 기록을 사용해 H거래일 뒤 QQQ 수익률과 비교합니다.
                - 룩어헤드 없이 trading-day offset으로 평가하므로, 충분한 히스토리가 쌓여야 합니다.
                """)
                with gr.Row():
                    bt_start = gr.Textbox(label="시작일 (YYYY-MM-DD)", placeholder="옵션", scale=1)
                    bt_end = gr.Textbox(label="종료일 (YYYY-MM-DD)", placeholder="옵션", scale=1)
                    bt_h = gr.Slider(minimum=3, maximum=20, value=5, step=1, label="지평 H(거래일)", scale=1)
                    bt_thresh = gr.Slider(minimum=0.3, maximum=0.8, value=0.5, step=0.05, label="상승 판정 임계값", scale=1)
                    bt_btn = gr.Button("📈 백테스트 실행", variant="primary", scale=1)
                bt_summary = gr.Markdown("히스토리를 기록한 뒤 백테스트를 실행하세요.")
                bt_plot = gr.Plot(label="P(Up) vs QQQ 수익률", show_label=True)
                bt_headers = ["신호일", "결과일", "P(Up)%", "QQQ %", "실제", "판정", "적중"]
                bt_table = gr.Dataframe(headers=bt_headers, value=pd.DataFrame(columns=bt_headers), visible=True)
                bt_download = gr.DownloadButton(label="💾 JSON 다운로드", value=None)
                bt_diag = gr.Markdown("")

                def _build_backtest_plot(rows, threshold):
                    if not go or not rows:
                        return None
                    dates = [r["asof_date"] for r in rows]
                    probs = [float(r["prob"]) for r in rows]
                    rets = [float(r["realized_return"]) * 100 for r in rows]
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=probs,
                            name="P(Up)",
                            mode="lines+markers",
                            line=dict(color="#2ecc71", width=2),
                            marker=dict(size=6),
                        )
                    )
                    fig.add_trace(
                        go.Bar(
                            x=dates,
                            y=rets,
                            name="QQQ 수익률(%)",
                            yaxis="y2",
                            marker_color="#95a5a6",
                            opacity=0.65,
                        )
                    )
                    fig.add_hline(y=threshold, line=dict(color="#e67e22", dash="dot"), annotation_text="임계값", annotation_position="top left")
                    fig.update_layout(
                        height=360,
                        margin=dict(l=40, r=40, t=30, b=60),
                        yaxis=dict(title="P(Up)", range=[0, 1]),
                        yaxis2=dict(title="QQQ %", overlaying="y", side="right"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                        xaxis=dict(title="신호일"),
                    )
                    return fig

                def _run_prob_backtest(start: str, end: str, horizon: int, threshold: float):
                    if run_market_prob_backtest is None:
                        empty = (go.Figure() if go else None)
                        return ("❌ market_prob_backtest 모듈을 불러올 수 없습니다.", empty, pd.DataFrame(), None, "모듈 import 실패")
                    try:
                        result = run_market_prob_backtest(
                            start_date=start or None,
                            end_date=end or None,
                            horizon_days=int(horizon),
                            prob_threshold=float(threshold),
                            base_symbol="QQQ",
                        )
                    except Exception as e:
                        empty = (go.Figure() if go else None)
                        return (f"❌ 백테스트 실패: {e}", empty, pd.DataFrame(), None, f"❌ {e}")
                    fig = _build_backtest_plot(result.rows, threshold)
                    df = pd.DataFrame(
                        [
                            {
                                "신호일": r["asof_date"],
                                "결과일": r["future_date"],
                                "P(Up)%": round(r["prob"] * 100, 2),
                                "QQQ %": round(r["realized_return"] * 100, 2),
                                "실제": "상승" if r["actual_up"] else "하락",
                                "판정": "상승" if r["predicted_up"] else "하락",
                                "적중": "✅" if r["actual_up"] == r["predicted_up"] else "❌",
                            }
                            for r in result.rows
                        ]
                    )
                    diag = (
                        f"표본 {result.stats['samples']} · 정확도 {result.stats['accuracy']*100:.1f}% · "
                        f"Hit-rate {result.stats['hit_rate']*100:.1f}%"
                    )
                    return (result.markdown, fig, df, result.json_path, diag)

                bt_btn.click(
                    _run_prob_backtest,
                    inputs=[bt_start, bt_end, bt_h, bt_thresh],
                    outputs=[bt_summary, bt_plot, bt_table, bt_download, bt_diag],
                )
                range_dd.change(
                    _run_realtime,
                    inputs=[window_dd, use_rt, exec_mode_dd, range_dd, start_box, end_box, premarket_box, auto_pre],
                    outputs=[rt_summary, rt_narrative, rt_transitions, ewdr_md, reg_fig, stab_fig, bt_fig, bt_stats, bt_csv_cls_btn, bt_csv_stab_btn, bt_csv_fus_btn],
                )
                exec_mode_dd.change(
                    _run_realtime,
                    inputs=[window_dd, use_rt, exec_mode_dd, range_dd, start_box, end_box, premarket_box, auto_pre],
                    outputs=[rt_summary, rt_narrative, rt_transitions, ewdr_md, reg_fig, stab_fig, bt_fig, bt_stats, bt_csv_cls_btn, bt_csv_stab_btn, bt_csv_fus_btn],
                )
                auto_pre.change(
                    _run_realtime,
                    inputs=[window_dd, use_rt, exec_mode_dd, range_dd, start_box, end_box, premarket_box, auto_pre],
                    outputs=[rt_summary, rt_narrative, rt_transitions, ewdr_md, reg_fig, stab_fig, bt_fig, bt_stats, bt_csv_cls_btn, bt_csv_stab_btn, bt_csv_fus_btn],
                )
            # 🎯 커스텀 분석 탭
            with gr.Tab("🎯 커스텀 분석"):
                gr.Markdown("## 직접 입력한 종목 분석")
            
                with gr.Row():
                    tickers_input = gr.Textbox(
                        label="분석할 종목 (티커) - 최대 15개",
                        placeholder="AAPL, MSFT, GOOGL 또는 AAPL MSFT GOOGL",
                        lines=2,
                        scale=3
                    )
                    analyze_custom_btn = gr.Button("🚀 분석 실행", variant="primary", scale=1)
            
                gr.Markdown("""
                **입력 예시:** `AAPL, MSFT, GOOGL` 또는 `AAPL MSFT GOOGL`  
                **새로운 기능:** 축별 가중치 정보도 함께 표시됩니다!
                """)
            
                custom_analysis_output = gr.Markdown(elem_classes=["analysis-output"])
            
                # 내보내기 버튼들
                with gr.Row(elem_classes=["export-buttons"]):
                    copy_btn = gr.Button("📋 결과 복사", variant="secondary")
                    json_download_btn = gr.Button("💾 JSON 다운로드", variant="secondary")
            
                # 복사용 텍스트박스
                copy_text_box = gr.Textbox(
                    label="복사용 텍스트 (전체 선택 후 복사하세요)",
                    lines=15,
                    visible=False,
                    max_lines=25
                )
            
                # JSON 다운로드
                json_file = gr.File(
                    label="JSON 파일 다운로드",
                    visible=False
                )
            
                json_status = gr.Markdown(visible=False)
        
            # ⭐ 즐겨찾기 분석
            with gr.Tab("⭐ 즐겨찾기 분석"):
                gr.Markdown("## 저장된 포트폴리오 분석")
            
                with gr.Row():
                    with gr.Column(scale=1):
                        load_fav_btn = gr.Button("📋 즐겨찾기 로드", variant="secondary")
                        analyze_fav_btn = gr.Button("🚀 분석 실행", variant="primary")
                
                    with gr.Column(scale=2):
                        favorites_display = gr.Markdown(
                            "📋 즐겨찾기를 로드하려면 버튼을 클릭하세요.",
                            elem_classes=["favorites-display"]
                        )
                        fav_status = gr.Markdown("")
            
                fav_analysis_output = gr.Markdown(elem_classes=["analysis-output"])
            
                # 즐겨찾기 분석 내보내기 버튼들
                with gr.Row(elem_classes=["export-buttons"]):
                    fav_copy_btn = gr.Button("📋 결과 복사", variant="secondary")
                    fav_json_download_btn = gr.Button("💾 JSON 다운로드", variant="secondary")
            
                # 즐겨찾기용 복사/다운로드
                fav_copy_text_box = gr.Textbox(
                    label="복사용 텍스트 (전체 선택 후 복사하세요)",
                    lines=15,
                    visible=False,
                    max_lines=25
                )
            
                fav_json_file = gr.File(
                    label="JSON 파일 다운로드",
                    visible=False
                )
            
                fav_json_status = gr.Markdown(visible=False)
        
            # 📝 즐겨찾기 편집
            with gr.Tab("📝 즐겨찾기 편집"):
                favorites_json_editor = gr.Code(
                    label="favorites.json 내용",
                    language="json",
                    value=initial_favorites_json
                )
            
                with gr.Row():
                    save_json_btn = gr.Button("💾 저장", variant="primary")
                    reload_btn = gr.Button("🔄 다시 로드", variant="secondary")
            
                edit_result = gr.Markdown("")
            
                gr.Markdown("""
                **형식 예시:**
                ```
                [
                  "ACHR",
                  "JOBY",
                  "SLDP",
                  "NVDA"
                ]
                ```
                """)

        def _load_favorites_for_analysis():
            display, favorites, edit_json = load_and_display_favorites()
            status_msg = "✅ 즐겨찾기를 로드했습니다."
            return display, favorites, edit_json, status_msg

        def _reload_favorites_for_editor():
            display, favorites, edit_json = load_and_display_favorites()
            status_msg = "✅ favorites.json을 다시 불러왔습니다."
            return edit_json, favorites, display, status_msg

        # 커스텀 분석 이벤트
        analyze_custom_btn.click(
            fn=run_custom_analysis,
            inputs=[tickers_input],
            outputs=[custom_analysis_output, copy_text_box]
        )
    
        tickers_input.submit(
            fn=run_custom_analysis,
            inputs=[tickers_input],
            outputs=[custom_analysis_output, copy_text_box]
        )
    
        # 복사 버튼 이벤트 (커스텀 분석)
        copy_btn.click(
            fn=show_copy_textbox,
            outputs=[copy_text_box]
        )
    
        # JSON 다운로드 버튼 이벤트 (커스텀 분석)
        json_download_btn.click(
            fn=show_json_download,
            outputs=[json_file, json_status]
        )
    
        # 복사 버튼 이벤트 (즐겨찾기 분석)
        fav_copy_btn.click(
            fn=show_copy_textbox,
            outputs=[fav_copy_text_box]
        )
    
        # JSON 다운로드 버튼 이벤트 (즐겨찾기 분석)
        fav_json_download_btn.click(
            fn=show_json_download,
            outputs=[fav_json_file, fav_json_status]
        )

        # 즐겨찾기 로드/분석/편집 이벤트
        load_fav_btn.click(
            fn=_load_favorites_for_analysis,
            outputs=[favorites_display, favorites_state, favorites_json_editor, fav_status]
        )

        analyze_fav_btn.click(
            fn=run_analysis_from_favorites,
            inputs=[favorites_state],
            outputs=[fav_analysis_output, fav_copy_text_box]
        )

        reload_btn.click(
            fn=_reload_favorites_for_editor,
            outputs=[favorites_json_editor, favorites_state, favorites_display, edit_result]
        )

        save_json_btn.click(
            fn=save_edited_favorites,
            inputs=[favorites_json_editor],
            outputs=[edit_result, favorites_display, favorites_state, favorites_json_editor]
        )

        module_apply_btn.click(
            fn=update_analysis_module,
            inputs=[module_dropdown],
            outputs=[module_status, module_info_md]
        )
    
    return interface


# 웹앱 실행
def simple_auth(_, password):
    """간단한 비밀번호 인증 (사용자명 무시)"""
    return password == "5632"

if __name__ == "__main__":
    try:
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            auth=simple_auth,
            share=False,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"웹앱 실행 오류: {e}")
