#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced 퀀트 시스템 1.8.2 - Gradio UI 인터페이스
3단 구조 시스템용 웹 인터페이스
"""
import gradio as gr
import json
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def create_three_tier_interface(system=None):
    """3단 구조 Enhanced 시스템용 Gradio 인터페이스 생성"""
    
    # 시스템 객체가 없으면 임포트해서 생성
    if system is None:
        try:
            from ..analysis.scoring_engine import Enhanced182ThreeTierSystem
            system = Enhanced182ThreeTierSystem()
            logger.info("✅ 시스템 객체 자동 생성 완료")
        except Exception as e:
            logger.error(f"❌ 시스템 객체 생성 실패: {e}")
            raise

    def analyze_single_symbol(symbol: str, enable_gap_boost: bool = True, enable_pattern: bool = True) -> tuple:
        """단일 종목 분석"""
        try:
            if not symbol or not symbol.strip():
                return "❌ 종목명을 입력하세요", "", "", "", ""
            
            symbol = symbol.strip().upper()
            logger.info(f"🎯 단일 종목 분석 시작: {symbol}")
            
            # 부스터 설정 임시 변경
            original_pattern_enabled = system.pattern_enabled
            system.pattern_enabled = enable_pattern
            
            # 비동기 분석 실행
            result = asyncio.run(system.analyze_symbol_comprehensive(symbol))
            
            # 설정 복구
            system.pattern_enabled = original_pattern_enabled
            
            if 'error' in result:
                return f"❌ 분석 실패: {result['error']}", "", "", "", ""
            
            # 결과 포맷팅
            summary_html = format_analysis_summary(result)
            vtnf_html = format_vtnf_scores(result)
            gap_html = format_gap_analysis(result) if enable_gap_boost else "갭 부스터 비활성화"
            pattern_html = format_pattern_analysis(result) if enable_pattern else "패턴 인식 비활성화"
            recommendation_html = format_investment_recommendation(result)
            
            logger.info(f"✅ {symbol} 분석 완료")
            return summary_html, vtnf_html, gap_html, pattern_html, recommendation_html
            
        except Exception as e:
            error_msg = f"❌ 분석 중 오류: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return error_msg, "", "", "", ""

    def analyze_multiple_symbols(symbols_text: str, max_symbols: int = 5) -> tuple:
        """다중 종목 분석"""
        try:
            if not symbols_text or not symbols_text.strip():
                return "❌ 종목 목록을 입력하세요", ""
            
            # 종목 파싱
            symbols = []
            for line in symbols_text.strip().split('\n'):
                symbol = line.strip().upper()
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
            
            if not symbols:
                return "❌ 유효한 종목이 없습니다", ""
            
            symbols = symbols[:max_symbols]  # 최대 개수 제한
            logger.info(f"🎯 다중 종목 분석 시작: {symbols}")
            
            # 비동기 분석 실행
            result = asyncio.run(system.analyze_multiple_symbols(symbols))
            
            if 'error' in result:
                return f"❌ 분석 실패: {result['error']}", ""
            
            # 결과 포맷팅
            portfolio_html = format_portfolio_analysis(result)
            comparison_html = format_symbols_comparison(result)
            
            logger.info(f"✅ 다중 종목 분석 완료: {len(symbols)}개")
            return portfolio_html, comparison_html
            
        except Exception as e:
            error_msg = f"❌ 다중 분석 중 오류: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return error_msg, ""

    def get_system_status() -> str:
        """시스템 상태 조회"""
        try:
            status = system.get_system_status()
            return format_system_status(status)
        except Exception as e:
            return f"❌ 상태 조회 실패: {str(e)}"

    def clear_all_caches() -> str:
        """모든 캐시 정리"""
        try:
            cleared_items = []
            
            # 갭 부스터 캐시 정리
            if hasattr(system, 'gap_booster'):
                system.gap_booster.clear_cache()
                cleared_items.append("갭 부스터 캐시")
            
            # 패턴 부스터 캐시 정리
            if hasattr(system, 'pattern_booster'):
                system.pattern_booster.clear_cache()
                cleared_items.append("패턴 인식 캐시")
            
            if cleared_items:
                return f"✅ 캐시 정리 완료: {', '.join(cleared_items)}"
            else:
                return "ℹ️ 정리할 캐시가 없습니다"
                
        except Exception as e:
            return f"❌ 캐시 정리 실패: {str(e)}"

    def refresh_sector_cache() -> str:
        """섹터 캐시 갱신"""
        try:
            logger.info("🔄 섹터 캐시 수동 갱신 시작...")
            asyncio.run(system._async_initialize_sector_cache())
            return "✅ 섹터 캐시 갱신 완료"
        except Exception as e:
            return f"❌ 섹터 캐시 갱신 실패: {str(e)}"

    # CSS 스타일
    custom_css = """
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .analysis-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .recommendation-buy { background: #d4edda; border-left: 4px solid #28a745; }
    .recommendation-sell { background: #f8d7da; border-left: 4px solid #dc3545; }
    .recommendation-hold { background: #fff3cd; border-left: 4px solid #ffc107; }
    .metric-value { font-size: 1.2em; font-weight: bold; color: #495057; }
    .section-title { 
        color: #2c3e50; 
        border-bottom: 2px solid #3498db; 
        padding-bottom: 5px; 
        margin-bottom: 15px;
    }
    """

    # 메인 인터페이스 구성
    with gr.Blocks(css=custom_css, title="Enhanced 퀀트 시스템 1.8.2", theme=gr.themes.Default()) as interface:
        
        # 헤더
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #4CAF50, #45a049); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🚀 Enhanced 퀀트 시스템 1.8.2</h1>
            <h3>3단 구조 AI 투자 분석 시스템</h3>
            <p>📊 Tier 1: 딥리서치 섹터 가중치 | 🔥 Tier 2: Gemini Flash 분석 | 🎯 Tier 3: 갭필터 + 패턴 부스터</p>
        </div>
        """)

        with gr.Tabs() as tabs:
            
            # 탭 1: 단일 종목 분석
            with gr.Tab("📈 단일 종목 분석"):
                with gr.Row():
                    with gr.Column(scale=1):
                        symbol_input = gr.Textbox(
                            label="📊 종목 코드/티커",
                            placeholder="예: AAPL, TSLA, 005930",
                            value="",
                            lines=1
                        )
                        
                        with gr.Row():
                            enable_gap = gr.Checkbox(
                                label="🎯 갭필터 부스터 활성화",
                                value=True
                            )
                            enable_pattern = gr.Checkbox(
                                label="🔍 패턴 인식 활성화", 
                                value=True
                            )
                        
                        analyze_btn = gr.Button(
                            "🎯 종합 분석 시작",
                            variant="primary",
                            size="lg"
                        )
                        
                        gr.HTML("""
                        <div style="margin-top: 20px; padding: 15px; background: #e3f2fd; border-radius: 8px;">
                            <h4>💡 사용법</h4>
                            <ul>
                                <li>🇺🇸 미국: AAPL, MSFT, TSLA 등</li>
                                <li>🇰🇷 한국: 005930, 000660 등</li>
                                <li>⚡ 실시간 AI 분석 (15-30초 소요)</li>
                                <li>🎯 갭필터: AI vs 실제값 델타 분석</li>
                                <li>🔍 패턴: 15분 차트 패턴 인식</li>
                            </ul>
                        </div>
                        """)
                    
                    with gr.Column(scale=2):
                        # 분석 결과 출력
                        analysis_summary = gr.HTML(label="📊 분석 요약")
                        
                        with gr.Row():
                            vtnf_scores = gr.HTML(label="📈 VTNF 점수")
                            gap_analysis_output = gr.HTML(label="🎯 갭필터 분석")
                        
                        with gr.Row():
                            pattern_output = gr.HTML(label="🔍 패턴 인식")
                            recommendation_output = gr.HTML(label="💼 투자 제안")

                # 분석 버튼 이벤트
                analyze_btn.click(
                    fn=analyze_single_symbol,
                    inputs=[symbol_input, enable_gap, enable_pattern],
                    outputs=[analysis_summary, vtnf_scores, gap_analysis_output, pattern_output, recommendation_output]
                )

            # 탭 2: 다중 종목 분석
            with gr.Tab("📊 포트폴리오 분석"):
                with gr.Row():
                    with gr.Column(scale=1):
                        symbols_input = gr.Textbox(
                            label="📋 종목 목록 (한 줄에 하나씩)",
                            placeholder="AAPL\nMSFT\nTSLA\nNVDA\nGOOGL",
                            lines=8
                        )
                        
                        max_symbols_slider = gr.Slider(
                            minimum=2,
                            maximum=10,
                            value=5,
                            step=1,
                            label="📊 최대 분석 종목 수"
                        )
                        
                        portfolio_analyze_btn = gr.Button(
                            "📊 포트폴리오 분석",
                            variant="primary",
                            size="lg"
                        )
                        
                        gr.HTML("""
                        <div style="margin-top: 20px; padding: 15px; background: #f3e5f5; border-radius: 8px;">
                            <h4>📊 포트폴리오 기능</h4>
                            <ul>
                                <li>🔍 최대 10개 종목 동시 분석</li>
                                <li>📈 상대 성과 비교</li>
                                <li>⚖️ 리스크 분산 평가</li>
                                <li>🎯 탑픽 추천</li>
                                <li>⚠️ 리스크 종목 식별</li>
                            </ul>
                        </div>
                        """)
                    
                    with gr.Column(scale=2):
                        portfolio_summary = gr.HTML(label="📊 포트폴리오 요약")
                        symbols_comparison = gr.HTML(label="📈 종목 비교")

                # 포트폴리오 분석 버튼 이벤트
                portfolio_analyze_btn.click(
                    fn=analyze_multiple_symbols,
                    inputs=[symbols_input, max_symbols_slider],
                    outputs=[portfolio_summary, symbols_comparison]
                )

            # 탭 3: 시스템 관리
            with gr.Tab("⚙️ 시스템 관리"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h3>📊 시스템 상태</h3>")
                        
                        status_btn = gr.Button("🔍 상태 확인", variant="secondary")
                        system_status_output = gr.HTML()
                        
                        gr.HTML("<hr>")
                        gr.HTML("<h3>🧹 캐시 관리</h3>")
                        
                        with gr.Row():
                            clear_cache_btn = gr.Button("🧹 캐시 정리", variant="secondary")
                            refresh_sector_btn = gr.Button("🔄 섹터 캐시 갱신", variant="secondary")
                        
                        cache_status_output = gr.HTML()
                        
                        gr.HTML("""
                        <div style="margin-top: 30px; padding: 20px; background: #fff3e0; border-radius: 8px;">
                            <h4>🛠️ 시스템 정보</h4>
                            <ul>
                                <li><strong>버전:</strong> Enhanced 1.8.2</li>
                                <li><strong>아키텍처:</strong> 3단 구조</li>
                                <li><strong>AI 엔진:</strong> Gemini 2.0 Flash + Perplexity</li>
                                <li><strong>데이터:</strong> KIS API + yfinance</li>
                                <li><strong>캐시:</strong> 섹터(48h) + 갭/패턴(5-10min)</li>
                            </ul>
                        </div>
                        """)
                    
                    with gr.Column():
                        gr.HTML("<h3>📈 성능 모니터링</h3>")
                        
                        # 실시간 차트나 성능 지표를 여기에 추가할 수 있음
                        gr.HTML("""
                        <div style="padding: 20px; background: #e8f5e8; border-radius: 8px; margin-bottom: 20px;">
                            <h4>✅ 시스템 상태</h4>
                            <p>• 모든 컴포넌트 정상 가동 중</p>
                            <p>• AI 분석 엔진 활성화</p>
                            <p>• 실시간 데이터 연결 양호</p>
                        </div>
                        """)
                        
                        gr.HTML("<h3>📚 사용 가이드</h3>")
                        gr.HTML("""
                        <div style="padding: 20px; background: #f5f5f5; border-radius: 8px;">
                            <h4>🎯 3단 구조 분석 과정</h4>
                            <ol>
                                <li><strong>Tier 1:</strong> 섹터별 VTNF 가중치 적용</li>
                                <li><strong>Tier 2:</strong> Gemini Flash로 N점수 실시간 계산</li>
                                <li><strong>Tier 3:</strong> 갭필터 + 패턴 부스터로 최종 조정</li>
                            </ol>
                            
                            <h4>📊 VTNF 스코어 의미</h4>
                            <ul>
                                <li><strong>V (Value):</strong> 재무지표, 밸류에이션</li>
                                <li><strong>T (Technical):</strong> 기술적 분석, 차트</li>
                                <li><strong>N (News):</strong> 뉴스, 감정, 트렌드</li>
                                <li><strong>F (Flow):</strong> 자금 흐름, 기관 매매</li>
                            </ul>
                        </div>
                        """)

                # 시스템 관리 버튼 이벤트들
                status_btn.click(
                    fn=get_system_status,
                    outputs=system_status_output
                )
                
                clear_cache_btn.click(
                    fn=clear_all_caches,
                    outputs=cache_status_output
                )
                
                refresh_sector_btn.click(
                    fn=refresh_sector_cache,
                    outputs=cache_status_output
                )

        # 푸터
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; background: #f8f9fa; border-radius: 10px;">
            <p style="color: #6c757d;">
                🚀 Enhanced 퀀트 시스템 1.8.2 | 
                ⚡ Powered by Gemini AI & Perplexity | 
                📊 Real-time Market Analysis
            </p>
            <p style="color: #6c757d; font-size: 0.9em;">
                ⚠️ 투자 결정은 본인 책임입니다. 이 시스템은 참고용으로만 사용하세요.
            </p>
        </div>
        """)

    return interface

# HTML 포맷팅 함수들

def format_analysis_summary(result: dict) -> str:
    """분석 요약 HTML 포맷팅"""
    try:
        symbol = result.get('symbol', 'N/A')
        sector = result.get('sector', 'Unknown')
        overall_score = result.get('overall_score', 0)
        analysis_time = result.get('analysis_metadata', {}).get('analysis_time', 0)
        
        # 점수에 따른 색상
        if overall_score >= 7.5:
            score_class = "status-good"
            score_emoji = "🚀"
        elif overall_score >= 6.0:
            score_class = "status-warning" 
            score_emoji = "⚡"
        else:
            score_class = "status-error"
            score_emoji = "⚠️"
        
        return f"""
        <div class="score-card">
            <h2>{score_emoji} {symbol} 종합 분석</h2>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 class="{score_class}">최종 점수: {overall_score:.2f}/10</h3>
                    <p><strong>섹터:</strong> {sector}</p>
                    <p><strong>분석 시간:</strong> {analysis_time:.1f}초</p>
                </div>
                <div style="font-size: 3em;">{score_emoji}</div>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div class='analysis-card'>❌ 요약 포맷팅 오류: {str(e)}</div>"

def format_vtnf_scores(result: dict) -> str:
    """VTNF 점수 HTML 포맷팅"""
    try:
        vtnf_scores = result.get('vtnf_scores', {})
        final_scores = result.get('final_scores', {})
        sector_weights = result.get('sector_weights', {})
        
        html = '<div class="analysis-card"><h3 class="section-title">📈 VTNF 상세 점수</h3>'
        
        for component in ['V', 'T', 'N', 'F']:
            score_data = vtnf_scores.get(component, {})
            final_data = final_scores.get(component, {})
            weight = sector_weights.get(component, 0.25)
            
            if isinstance(score_data, dict):
                raw_score = score_data.get('score', 0)
                source = score_data.get('source', 'unknown')
            else:
                raw_score = score_data
                source = 'direct'
            
            weighted_score = final_data.get('weighted_score', 0)
            
            component_names = {
                'V': 'Value (가치)', 
                'T': 'Technical (기술적)', 
                'N': 'News (뉴스)', 
                'F': 'Flow (자금흐름)'
            }
            
            html += f"""
            <div style="margin: 10px 0; padding: 10px; border-left: 4px solid #3498db; background: #f8f9fa;">
                <strong>{component_names[component]}</strong><br>
                원점수: <span class="metric-value">{raw_score:.2f}</span> | 
                가중치: {weight:.2f} | 
                최종: <span class="metric-value">{weighted_score:.2f}</span><br>
                <small>데이터 소스: {source}</small>
            </div>
            """
        
        html += '</div>'
        return html
        
    except Exception as e:
        return f"<div class='analysis-card'>❌ VTNF 포맷팅 오류: {str(e)}</div>"

def format_gap_analysis(result: dict) -> str:
    """갭 분석 HTML 포맷팅"""
    try:
        gap_analysis = result.get('gap_analysis', {})
        
        if not gap_analysis or not gap_analysis.get('gap_detected'):
            return """
            <div class="analysis-card">
                <h3 class="section-title">🎯 갭필터 분석</h3>
                <p>ℹ️ 유의미한 갭이 감지되지 않았습니다</p>
            </div>
            """
        
        gap_score = gap_analysis.get('gap_score', 0)
        boost_type = gap_analysis.get('boost_type', 'neutral')
        confidence = gap_analysis.get('confidence', 0)
        position_adjustment = gap_analysis.get('position_adjustment', 0)
        
        boost_emoji = "🚀" if boost_type == 'synergy' else "⚠️" if boost_type == 'risk' else "➡️"
        boost_text = "시너지 부스트" if boost_type == 'synergy' else "리스크 감지" if boost_type == 'risk' else "중립"
        
        return f"""
        <div class="analysis-card">
            <h3 class="section-title">🎯 갭필터 분석</h3>
            <div style="text-align: center; margin: 15px 0;">
                <div style="font-size: 2em;">{boost_emoji}</div>
                <h4>{boost_text}</h4>
            </div>
            
            <div style="margin: 10px 0;">
                <strong>갭 점수:</strong> <span class="metric-value">{gap_score:+.2f}</span><br>
                <strong>신뢰도:</strong> <span class="metric-value">{confidence:.1%}</span><br>
                <strong>포지션 조정:</strong> <span class="metric-value">{position_adjustment:+.1f}%</span>
            </div>
            
            <div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <small><strong>AI 독립 판단 vs 실제 VTNF 델타 분석</strong><br>
                AI가 종목을 다르게 평가하는 정도를 측정하여 시장 미스프라이싱을 탐지합니다.</small>
            </div>
        </div>
        """
        
    except Exception as e:
        return f"<div class='analysis-card'>❌ 갭 분석 포맷팅 오류: {str(e)}</div>"

def format_pattern_analysis(result: dict) -> str:
    """패턴 분석 HTML 포맷팅"""
    try:
        pattern_analysis = result.get('pattern_analysis', {})
        
        if not pattern_analysis or pattern_analysis.get('pattern_detected') == 'None':
            return """
            <div class="analysis-card">
                <h3 class="section-title">🔍 패턴 인식</h3>
                <p>ℹ️ 명확한 패턴이 감지되지 않았습니다</p>
            </div>
            """
        
        pattern_name = pattern_analysis.get('pattern_detected', 'Unknown')
        pattern_score = pattern_analysis.get('pattern_score', 5.0)
        direction = pattern_analysis.get('direction_bias', 'Neutral')
        confidence = pattern_analysis.get('confidence', 50)
        entry_timing = pattern_analysis.get('entry_timing', 'Wait')
        
        direction_emoji = "📈" if "bullish" in direction.lower() else "📉" if "bearish" in direction.lower() else "➡️"
        
        return f"""
        <div class="analysis-card">
            <h3 class="section-title">🔍 패턴 인식 (15분)</h3>
            
            <div style="text-align: center; margin: 15px 0;">
                <div style="font-size: 2em;">{direction_emoji}</div>
                <h4>{pattern_name.replace('_', ' ')}</h4>
            </div>
            
            <div style="margin: 10px 0;">
                <strong>패턴 점수:</strong> <span class="metric-value">{pattern_score:.1f}/10</span><br>
                <strong>방향성:</strong> <span class="metric-value">{direction}</span><br>
                <strong>신뢰도:</strong> <span class="metric-value">{confidence}%</span><br>
                <strong>진입 타이밍:</strong> <span class="metric-value">{entry_timing}</span>
            </div>
            
            <div style="background: #f3e5f5; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <small><strong>Gemini 2.5 Flash 패턴 인식</strong><br>
                15분 차트 기준 단기 트레이딩 최적화 패턴 분석</small>
            </div>
        </div>
        """
        
    except Exception as e:
        return f"<div class='analysis-card'>❌ 패턴 분석 포맷팅 오류: {str(e)}</div>"

def format_investment_recommendation(result: dict) -> str:
    """투자 제안 HTML 포맷팅"""
    try:
        recommendation = result.get('investment_recommendation', {})
        
        action = recommendation.get('action', 'HOLD')
        confidence = recommendation.get('confidence', '중간')
        position_size = recommendation.get('recommended_position_size', '5%')
        
        # 액션별 스타일
        if action in ['STRONG_BUY', 'BUY']:
            card_class = "recommendation-buy"
            action_emoji = "🚀" if action == 'STRONG_BUY' else "📈"
        elif action in ['SELL', 'WEAK_SELL']:
            card_class = "recommendation-sell"
            action_emoji = "📉" if action == 'SELL' else "⚠️"
        else:
            card_class = "recommendation-hold"
            action_emoji = "⏸️"
        
        action_text = {
            'STRONG_BUY': '강력 매수',
            'BUY': '매수',
            'HOLD': '보유',
            'WEAK_SELL': '약한 매도',
            'SELL': '매도'
        }.get(action, action)
        
        html = f"""
        <div class="analysis-card {card_class}">
            <h3 class="section-title">💼 투자 제안</h3>
            
            <div style="text-align: center; margin: 15px 0;">
                <div style="font-size: 3em;">{action_emoji}</div>
                <h2>{action_text}</h2>
                <p><strong>신뢰도:</strong> {confidence}</p>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>권장 포지션:</strong> {position_size}<br>
                <strong>투자 기간:</strong> {recommendation.get('time_horizon', '단기')}<br>
                <strong>손절매:</strong> {recommendation.get('stop_loss_suggestion', 'N/A')}<br>
                <strong>수익 목표:</strong> {recommendation.get('profit_target_suggestion', 'N/A')}
            </div>
        """
        
        # 강점/약점
        strengths = recommendation.get('key_strengths', [])
        weaknesses = recommendation.get('key_weaknesses', [])
        
        if strengths:
            html += "<div style='margin: 10px 0;'><strong>💪 주요 강점:</strong><ul>"
            for strength in strengths[:3]:
                html += f"<li>{strength}</li>"
            html += "</ul></div>"
        
        if weaknesses:
            html += "<div style='margin: 10px 0;'><strong>⚠️ 주요 약점:</strong><ul>"
            for weakness in weaknesses[:3]:
                html += f"<li>{weakness}</li>"
            html += "</ul></div>"
        
        html += "</div>"
        return html
        
    except Exception as e:
        return f"<div class='analysis-card'>❌ 투자 제안 포맷팅 오류: {str(e)}</div>"

def format_portfolio_analysis(result: dict) -> str:
    """포트폴리오 분석 HTML 포맷팅"""
    try:
        portfolio_summary = result.get('portfolio_summary', {})
        symbols_data = result.get('symbols', {})
        
        if not portfolio_summary:
            return "<div class='analysis-card'>❌ 포트폴리오 데이터 없음</div>"
        
        avg_score = portfolio_summary.get('average_score', 0)
        action_dist = portfolio_summary.get('action_distribution', {})
        top_picks = portfolio_summary.get('top_picks', [])
        risk_symbols = portfolio_summary.get('risk_symbols', [])
        
        html = f"""
        <div class="score-card">
            <h2>📊 포트폴리오 요약</h2>
            <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                <div style="text-align: center;">
                    <h3>평균 점수</h3>
                    <div class="metric-value" style="font-size: 2em; color: white;">{avg_score:.2f}</div>
                </div>
                <div style="text-align: center;">
                    <h3>분석 종목</h3>
                    <div class="metric-value" style="font-size: 2em; color: white;">{len(symbols_data)}</div>
                </div>
            </div>
        </div>
        
        <div class="analysis-card">
            <h3 class="section-title">📈 투자 액션 분포</h3>
            <div style="display: flex; justify-content: space-around; margin: 15px 0;">
        """
        
        action_colors = {
            'STRONG_BUY': '#28a745', 'BUY': '#17a2b8',
            'HOLD': '#ffc107', 'WEAK_SELL': '#fd7e14', 'SELL': '#dc3545'
        }
        
        for action, count in action_dist.items():
            color = action_colors.get(action, '#6c757d')
            html += f"""
                <div style="text-align: center;">
                    <div style="background: {color}; color: white; padding: 10px; border-radius: 5px;">
                        <div style="font-weight: bold;">{count}</div>
                        <div style="font-size: 0.8em;">{action.replace('_', ' ')}</div>
                    </div>
                </div>
            """
        
        html += "</div></div>"
        
        # 탑픽과 리스크 종목
        if top_picks:
            html += f"""
            <div class="analysis-card recommendation-buy">
                <h3>🏆 탑픽 ({len(top_picks)}개)</h3>
                <p>{', '.join(top_picks)}</p>
            </div>
            """
        
        if risk_symbols:
            html += f"""
            <div class="analysis-card recommendation-sell">
                <h3>⚠️ 리스크 종목 ({len(risk_symbols)}개)</h3>
                <p>{', '.join(risk_symbols)}</p>
            </div>
            """
        
        return html
        
    except Exception as e:
        return f"<div class='analysis-card'>❌ 포트폴리오 포맷팅 오류: {str(e)}</div>"

def format_symbols_comparison(result: dict) -> str:
    """종목 비교 HTML 포맷팅"""
    try:
        symbols_data = result.get('symbols', {})
        
        if not symbols_data:
            return "<div class='analysis-card'>❌ 비교할 종목 데이터 없음</div>"
        
        html = """
        <div class="analysis-card">
            <h3 class="section-title">📈 종목별 상세 비교</h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 10px; border: 1px solid #ddd;">종목</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">점수</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">액션</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">섹터</th>
                            <th style="padding: 10px; border: 1px solid #ddd;">V-T-N-F</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # 점수순으로 정렬
        sorted_symbols = sorted(
            symbols_data.items(),
            key=lambda x: x[1].get('overall_score', 0),
            reverse=True
        )
        
        for symbol, data in sorted_symbols:
            score = data.get('overall_score', 0)
            action = data.get('investment_recommendation', {}).get('action', 'HOLD')
            sector = data.get('sector', 'Unknown')
            
            # VTNF 점수 추출
            final_scores = data.get('final_scores', {})
            vtnf_text = "-".join([
                f"{final_scores.get(c, {}).get('raw_score', 0):.1f}" 
                for c in ['V', 'T', 'N', 'F']
            ])
            
            # 점수별 색상
            score_color = "#28a745" if score >= 7.5 else "#ffc107" if score >= 6.0 else "#dc3545"
            
            html += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">{symbol}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; color: {score_color}; font-weight: bold;">{score:.2f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{action.replace('_', ' ')}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{sector}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; font-family: monospace;">{vtnf_text}</td>
                </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        return html
        
    except Exception as e:
        return f"<div class='analysis-card'>❌ 종목 비교 포맷팅 오류: {str(e)}</div>"

def format_system_status(status: dict) -> str:
    """시스템 상태 HTML 포맷팅"""
    try:
        components = status.get('components', {})
        api_status = status.get('api_status', {})
        
        html = """
        <div class="analysis-card">
            <h3 class="section-title">🛠️ 시스템 상태</h3>
        """
        
        # 컴포넌트 상태
        html += "<h4>📊 주요 컴포넌트</h4>"
        
        sector_cache = components.get('sector_cache', {})
        sector_status = "✅ 정상" if sector_cache.get('valid') else "⚠️ 만료"
        
        gap_booster = components.get('gap_booster', {})
        gap_status = "✅ 활성화" if gap_booster.get('enabled') else "❌ 비활성화"
        
        pattern_booster = components.get('pattern_booster', {})
        pattern_status = "✅ 활성화" if pattern_booster.get('enabled') else "❌ 비활성화"
        
        html += f"""
        <ul>
            <li><strong>섹터 캐시:</strong> {sector_status}</li>
            <li><strong>갭 부스터:</strong> {gap_status}</li>
            <li><strong>패턴 인식:</strong> {pattern_status}</li>
        </ul>
        """
        
        # API 상태
        html += "<h4>🔌 API 연결 상태</h4><ul>"
        
        kis_status = "✅ 연결됨" if api_status.get('kis_token_manager') else "❌ 미연결"
        gemini_status = "✅ 설정됨" if api_status.get('gemini_config') else "❌ 미설정"
        perplexity_status = "✅ 설정됨" if api_status.get('perplexity_config') else "❌ 미설정"
        
        html += f"""
            <li><strong>한투 API:</strong> {kis_status}</li>
            <li><strong>Gemini API:</strong> {gemini_status}</li>
            <li><strong>Perplexity API:</strong> {perplexity_status}</li>
        </ul>
        """
        
        # 시스템 정보
        html += f"""
        <h4>ℹ️ 시스템 정보</h4>
        <ul>
            <li><strong>버전:</strong> {status.get('system_version', 'Unknown')}</li>
            <li><strong>마지막 업데이트:</strong> {status.get('timestamp', 'Unknown')}</li>
        </ul>
        </div>
        """
        
        return html
        
    except Exception as e:
        return f"<div class='analysis-card'>❌ 상태 포맷팅 오류: {str(e)}</div>"
