#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 향상된 리포트 생성기

4개 카테고리별 15개씩 종목 분석:
1. 점수 변동 큰 순 15개
2. 현재 점수 높은 순 15개  
3. 예상 수익률 높은 순 15개
4. 실제 수익률 높은 순 15개
"""

import json
import logging
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import statistics
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockAnalysis:
    """종목 분석 데이터"""
    symbol: str
    company_name: str
    original_score: float  # 이전 점수 (ML 최적화 전)
    current_score: float   # 현재 점수 (ML 최적화 후)
    score_change: float    # 점수 변동 (current - original)
    expected_return: Optional[float] = None  # 예상수익률 (%)
    actual_return: Optional[float] = None    # 실제수익률 (%)
    tech_category: str = 'unknown'
    recovery_stage: str = 'unknown'
    volume_signals: Dict[str, bool] = None
    guide_confidence: str = 'medium'
    selection_date: str = ''
    days_held: int = 0

class EnhancedReportGenerator:
    """향상된 리포트 생성기"""
    
    def __init__(self):
        self.sweet_spot_db_file = "sweet_spot_database.json"
        self.predictions_file = "ml_predictions_history.json"
        self.ml_parameters_file = "ml_parameters.json"
        
        # 카테고리별 개수
        self.items_per_category = 15
    
    def calculate_expected_return_from_score(self, score: float) -> float:
        """ML 점수를 예상수익률로 변환 (prediction_system.py와 동일한 로직)"""
        try:
            # 점수 절대값 기반 예상수익률 계산
            if score >= 80:
                return 12.0 + (score - 80) * 0.5  # 80점 이상: 12%+ 예상
            elif score >= 70:
                return 8.0 + (score - 70) * 0.4   # 70-80점: 8-12% 예상
            elif score >= 60:
                return 4.0 + (score - 60) * 0.4   # 60-70점: 4-8% 예상
            elif score >= 50:
                return 0.0 + (score - 50) * 0.4   # 50-60점: 0-4% 예상
            else:
                return -5.0 + score * 0.1          # 50점 미만: 부정적 예상
        except:
            return 0.0
        
    def load_sweet_spot_database(self) -> Dict:
        """Sweet Spot 데이터베이스 로드"""
        try:
            if os.path.exists(self.sweet_spot_db_file):
                with open(self.sweet_spot_db_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Sweet Spot DB 로드 실패: {e}")
            return {}
    
    def load_predictions_history(self) -> List[Dict]:
        """예측 히스토리 로드"""
        try:
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"예측 히스토리 로드 실패: {e}")
            return []
    
    async def generate_enhanced_report(self) -> str:
        """4개 카테고리 향상된 리포트 생성"""
        try:
            logger.info("📊 향상된 4개 카테고리 리포트 생성 시작")
            
            # 데이터 로드
            sweet_spot_db = self.load_sweet_spot_database()
            predictions_history = self.load_predictions_history()
            
            picks = sweet_spot_db.get('picks', [])
            if not picks:
                logger.warning("Sweet Spot 종목 데이터가 없습니다")
                return "❌ 분석할 종목 데이터가 없습니다."
            
            # 종목별 분석 데이터 생성
            stock_analyses = await self.create_stock_analyses(picks, predictions_history)
            
            if not stock_analyses:
                logger.warning("분석 가능한 종목이 없습니다")
                return "❌ 분석 가능한 종목이 없습니다."
            
            # 4개 카테고리별 분석
            report_sections = []
            
            # 1. 점수 변동 큰 순 15개
            score_changed_section = self.generate_score_change_section(stock_analyses)
            report_sections.append(score_changed_section)
            
            # 2. 현재 점수 높은 순 15개
            high_score_section = self.generate_high_score_section(stock_analyses)
            report_sections.append(high_score_section)
            
            # 3. 예상 수익률 높은 순 15개
            expected_return_section = self.generate_expected_return_section(stock_analyses)
            report_sections.append(expected_return_section)
            
            # 4. 실제 수익률 높은 순 15개
            actual_return_section = self.generate_actual_return_section(stock_analyses)
            report_sections.append(actual_return_section)
            
            # 전체 리포트 조합
            header = self.generate_report_header(len(stock_analyses))
            summary = self.generate_summary_section(stock_analyses)
            footer = self.generate_report_footer()
            
            full_report = f"{header}\n\n{summary}\n\n" + "\n\n".join(report_sections) + f"\n\n{footer}"
            
            logger.info("✅ 향상된 4개 카테고리 리포트 생성 완료")
            return full_report
            
        except Exception as e:
            logger.error(f"향상된 리포트 생성 실패: {e}")
            return f"❌ 리포트 생성 실패: {str(e)}"
    
    async def create_stock_analyses(self, picks: List[Dict], predictions_history: List[Dict]) -> List[StockAnalysis]:
        """종목별 분석 데이터 생성"""
        try:
            stock_analyses = []
            
            # 예측 데이터를 심볼별로 정리
            predictions_map = {}
            for pred in predictions_history:
                symbol = pred.get('symbol')
                if symbol:
                    if symbol not in predictions_map:
                        predictions_map[symbol] = []
                    predictions_map[symbol].append(pred)
            
            # 현재 ML 파라미터로 점수 계산
            current_params = await self.load_current_ml_parameters()
            
            for pick in picks:
                try:
                    symbol = pick.get('symbol')
                    if not symbol:
                        continue
                    
                    # 기본 정보
                    company_name = pick.get('company_name', symbol)
                    selection_score = pick.get('selection_score', 0)  # 이전 점수 (ML 최적화 전)
                    current_score = pick.get('recent_score', selection_score)  # 현재 점수 (ML 최적화 후)
                    
                    # 점수 변동 계산
                    score_change = current_score - selection_score
                    
                    # 예측 수익률 (최신 예측 또는 점수 기반 계산)
                    expected_return = None
                    if symbol in predictions_map:
                        latest_prediction = max(predictions_map[symbol], 
                                             key=lambda x: x.get('prediction_date', ''))
                        expected_return = latest_prediction.get('next_week_expected_return')
                    
                    # 예상수익률이 없는 경우 현재 점수 기반으로 계산
                    if expected_return is None:
                        expected_return = self.calculate_expected_return_from_score(current_score)
                    
                    # 실제 수익률
                    actual_return = pick.get('current_return_pct', pick.get('current_return', 0.0))
                    
                    # 기술 카테고리
                    tech_category = pick.get('tech_category', 'unknown')
                    
                    # 회복 단계
                    recovery_stage = pick.get('recovery_stage', 'unknown')
                    
                    # 거래량 신호
                    volume_signals = {
                        'volume_surge': pick.get('volume_surge', False),
                        'volume_trend': pick.get('volume_trend', False)
                    }
                    
                    # 투자가이드 신뢰도
                    guide_confidence = 'high' if pick.get('investment_guide', {}).get('confidence_level') == 'high' else 'medium'
                    
                    # 보유 일수
                    selection_date = pick.get('selection_date', '')
                    days_held = pick.get('days_held', 0)
                    
                    analysis = StockAnalysis(
                        symbol=symbol,
                        company_name=company_name,
                        original_score=selection_score,  # 이전 점수
                        current_score=current_score,     # 현재 점수
                        score_change=score_change,       # 점수 변동
                        expected_return=expected_return, # 예상수익률
                        actual_return=actual_return,     # 실제수익률
                        tech_category=tech_category,
                        recovery_stage=recovery_stage,
                        volume_signals=volume_signals,
                        guide_confidence=guide_confidence,
                        selection_date=selection_date,
                        days_held=days_held
                    )
                    
                    stock_analyses.append(analysis)
                    
                except Exception as e:
                    logger.warning(f"{pick.get('symbol', 'Unknown')} 분석 실패: {e}")
                    continue
            
            logger.info(f"종목별 분석 완료: {len(stock_analyses)}개 종목")
            return stock_analyses
            
        except Exception as e:
            logger.error(f"종목별 분석 데이터 생성 실패: {e}")
            return []
    
    async def load_current_ml_parameters(self) -> Dict:
        """현재 ML 파라미터 로드"""
        try:
            if os.path.exists(self.ml_parameters_file):
                with open(self.ml_parameters_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('current_parameters', {})
            return {}
        except Exception as e:
            logger.warning(f"ML 파라미터 로드 실패: {e}")
            return {}
    
    def generate_score_change_section(self, stock_analyses: List[StockAnalysis]) -> str:
        """1. 점수 변동 큰 순 15개 섹션"""
        try:
            # 점수 변동 절댓값 기준 정렬
            sorted_stocks = sorted(stock_analyses, 
                                 key=lambda x: abs(x.score_change), reverse=True)
            top_stocks = sorted_stocks[:self.items_per_category]
            
            section = "## 📈 1. 점수 변동 큰 순 TOP 15\n\n"
            section += "| 순위 | 종목 | 이전점수 | 현재점수 | 점수변동 | 실제수익률 | 예상수익률 | 딥테크 |\n"
            section += "|------|------|----------|----------|----------|------------|------------|--------|\n"
            
            for i, stock in enumerate(top_stocks, 1):
                change_indicator = "📈" if stock.score_change > 0 else "📉" if stock.score_change < 0 else "➡️"
                actual_indicator = "📈" if stock.actual_return and stock.actual_return > 0 else "📉" if stock.actual_return and stock.actual_return < 0 else "➡️"
                expected_indicator = "📈" if stock.expected_return and stock.expected_return > 0 else "📉" if stock.expected_return and stock.expected_return < 0 else "➡️"
                
                actual_text = f"{actual_indicator}{stock.actual_return:+.1f}%" if stock.actual_return is not None else "N/A"
                expected_text = f"{expected_indicator}{stock.expected_return:+.1f}%" if stock.expected_return is not None else "N/A"
                
                section += f"| {i} | {stock.symbol} | {stock.original_score:.1f} | {stock.current_score:.1f} | {change_indicator}{stock.score_change:+.1f} | {actual_text} | {expected_text} | {stock.tech_category[:10]} |\n"
            
            # 분석 요약
            positive_changes = sum(1 for s in top_stocks if s.score_change > 0)
            negative_changes = sum(1 for s in top_stocks if s.score_change < 0)
            avg_change = np.mean([s.score_change for s in top_stocks])
            
            section += f"\n**📊 점수 변동 분석:**\n"
            section += f"- 상승: {positive_changes}개, 하락: {negative_changes}개\n"
            section += f"- 평균 변동: {avg_change:+.2f}점\n"
            section += f"- 최대 변동: {max([abs(s.score_change) for s in top_stocks]):.1f}점\n"
            
            return section
            
        except Exception as e:
            logger.error(f"점수 변동 섹션 생성 실패: {e}")
            return "❌ 점수 변동 섹션 생성 실패"
    
    def generate_high_score_section(self, stock_analyses: List[StockAnalysis]) -> str:
        """2. 현재 점수 높은 순 15개 섹션"""
        try:
            # 현재 점수 기준 정렬
            sorted_stocks = sorted(stock_analyses, 
                                 key=lambda x: x.current_score, reverse=True)
            top_stocks = sorted_stocks[:self.items_per_category]
            
            section = "## 🏆 2. 현재 점수 높은 순 TOP 15\n\n"
            section += "| 순위 | 종목 | 이전점수 | 현재점수 | 점수변동 | 실제수익률 | 예상수익률 | 회복단계 |\n"
            section += "|------|------|----------|----------|----------|------------|------------|----------|\n"
            
            for i, stock in enumerate(top_stocks, 1):
                change_indicator = "📈" if stock.score_change > 0 else "📉" if stock.score_change < 0 else "➡️"
                actual_indicator = "📈" if stock.actual_return and stock.actual_return > 0 else "📉" if stock.actual_return and stock.actual_return < 0 else "➡️"
                expected_indicator = "📈" if stock.expected_return and stock.expected_return > 0 else "📉" if stock.expected_return and stock.expected_return < 0 else "➡️"
                
                actual_text = f"{actual_indicator}{stock.actual_return:+.1f}%" if stock.actual_return is not None else "N/A"
                expected_text = f"{expected_indicator}{stock.expected_return:+.1f}%" if stock.expected_return is not None else "N/A"
                
                section += f"| {i} | {stock.symbol} | {stock.original_score:.1f} | {stock.current_score:.1f} | {change_indicator}{stock.score_change:+.1f} | {actual_text} | {expected_text} | {stock.recovery_stage} |\n"
            
            # 분석 요약
            avg_score = np.mean([s.current_score for s in top_stocks])
            high_scores = sum(1 for s in top_stocks if s.current_score >= 70)
            with_expected = sum(1 for s in top_stocks if s.expected_return is not None)
            
            section += f"\n**🏆 고득점 종목 분석:**\n"
            section += f"- 평균 점수: {avg_score:.1f}점\n"
            section += f"- 70점 이상: {high_scores}개 종목\n"
            section += f"- 예측 수익률 보유: {with_expected}개 종목\n"
            
            return section
            
        except Exception as e:
            logger.error(f"고득점 섹션 생성 실패: {e}")
            return "❌ 고득점 섹션 생성 실패"
    
    def generate_expected_return_section(self, stock_analyses: List[StockAnalysis]) -> str:
        """3. 예상 수익률 높은 순 15개 섹션"""
        try:
            # 예상 수익률이 있는 종목들만 필터링 후 정렬
            stocks_with_expected = [s for s in stock_analyses if s.expected_return is not None]
            sorted_stocks = sorted(stocks_with_expected, 
                                 key=lambda x: x.expected_return, reverse=True)
            top_stocks = sorted_stocks[:self.items_per_category]
            
            section = "## 🎯 3. 예상 수익률 높은 순 TOP 15\n\n"
            section += "| 순위 | 종목 | 이전점수 | 현재점수 | 점수변동 | 실제수익률 | 예상수익률 | 예측정확도 |\n"
            section += "|------|------|----------|----------|----------|------------|------------|------------|\n"
            
            for i, stock in enumerate(top_stocks, 1):
                change_indicator = "📈" if stock.score_change > 0 else "📉" if stock.score_change < 0 else "➡️"
                actual_indicator = "📈" if stock.actual_return and stock.actual_return > 0 else "📉" if stock.actual_return and stock.actual_return < 0 else "➡️"
                expected_indicator = "📈" if stock.expected_return and stock.expected_return > 0 else "📉" if stock.expected_return and stock.expected_return < 0 else "➡️"
                
                actual_text = f"{actual_indicator}{stock.actual_return:+.1f}%" if stock.actual_return is not None else "진행중"
                expected_text = f"{expected_indicator}{stock.expected_return:+.1f}%" if stock.expected_return is not None else "N/A"
                
                # 예측 정확도 계산
                prediction_accuracy = ""
                if stock.actual_return is not None and stock.expected_return is not None:
                    error = abs(stock.actual_return - stock.expected_return)
                    if error <= 3:
                        prediction_accuracy = "🎯정확"
                    elif error <= 8:
                        prediction_accuracy = "⚖️보통"
                    else:
                        prediction_accuracy = "❌오차큼"
                else:
                    prediction_accuracy = "⏳진행중"
                
                section += f"| {i} | {stock.symbol} | {stock.original_score:.1f} | {stock.current_score:.1f} | {change_indicator}{stock.score_change:+.1f} | {actual_text} | {expected_text} | {prediction_accuracy} |\n"
            
            # 분석 요약
            if top_stocks:
                avg_expected = np.mean([s.expected_return for s in top_stocks])
                positive_predictions = sum(1 for s in top_stocks if s.expected_return > 0)
                
                section += f"\n**🎯 예상 수익률 분석:**\n"
                section += f"- 평균 예상 수익률: {avg_expected:+.1f}%\n"
                section += f"- 상승 예측: {positive_predictions}/{len(top_stocks)}개 종목\n"
                section += f"- 예측 보유 종목: {len(stocks_with_expected)}개\n"
            else:
                section += "\n**⚠️ 예측 수익률 데이터가 없습니다.**\n"
            
            return section
            
        except Exception as e:
            logger.error(f"예상 수익률 섹션 생성 실패: {e}")
            return "❌ 예상 수익률 섹션 생성 실패"
    
    def generate_actual_return_section(self, stock_analyses: List[StockAnalysis]) -> str:
        """4. 실제 수익률 높은 순 15개 섹션"""
        try:
            # 실제 수익률이 있는 종목들만 필터링 후 정렬
            stocks_with_actual = [s for s in stock_analyses if s.actual_return is not None]
            sorted_stocks = sorted(stocks_with_actual, 
                                 key=lambda x: x.actual_return, reverse=True)
            top_stocks = sorted_stocks[:self.items_per_category]
            
            section = "## 💰 4. 실제 수익률 높은 순 TOP 15\n\n"
            section += "| 순위 | 종목 | 이전점수 | 현재점수 | 점수변동 | 실제수익률 | 예상수익률 | 보유일 |\n"
            section += "|------|------|----------|----------|----------|------------|------------|--------|\n"
            
            for i, stock in enumerate(top_stocks, 1):
                change_indicator = "📈" if stock.score_change > 0 else "📉" if stock.score_change < 0 else "➡️"
                actual_indicator = "📈" if stock.actual_return and stock.actual_return > 0 else "📉" if stock.actual_return and stock.actual_return < 0 else "➡️"
                expected_indicator = "📈" if stock.expected_return and stock.expected_return > 0 else "📉" if stock.expected_return and stock.expected_return < 0 else "➡️"
                
                actual_text = f"{actual_indicator}{stock.actual_return:+.1f}%" if stock.actual_return is not None else "N/A"
                expected_text = f"{expected_indicator}{stock.expected_return:+.1f}%" if stock.expected_return is not None else "N/A"
                
                section += f"| {i} | {stock.symbol} | {stock.original_score:.1f} | {stock.current_score:.1f} | {change_indicator}{stock.score_change:+.1f} | {actual_text} | {expected_text} | {stock.days_held}일 |\n"
            
            # 분석 요약
            if top_stocks:
                avg_actual = np.mean([s.actual_return for s in top_stocks])
                positive_returns = sum(1 for s in top_stocks if s.actual_return > 0)
                high_returns = sum(1 for s in top_stocks if s.actual_return > 10)
                
                # 예측 정확도 통계
                accurate_predictions = 0
                total_predictions = 0
                for s in top_stocks:
                    if s.expected_return is not None:
                        total_predictions += 1
                        error = abs(s.actual_return - s.expected_return)
                        if error <= 5:  # 5% 이내 오차
                            accurate_predictions += 1
                
                section += f"\n**💰 실제 수익률 분석:**\n"
                section += f"- 평균 실제 수익률: {avg_actual:+.1f}%\n"
                section += f"- 상승 종목: {positive_returns}/{len(top_stocks)}개 ({positive_returns/len(top_stocks):.1%})\n"
                section += f"- 고수익(10%+): {high_returns}개 종목\n"
                
                if total_predictions > 0:
                    section += f"- 예측 정확도: {accurate_predictions}/{total_predictions}개 ({accurate_predictions/total_predictions:.1%})\n"
            else:
                section += "\n**⚠️ 실제 수익률 데이터가 없습니다.**\n"
            
            return section
            
        except Exception as e:
            logger.error(f"실제 수익률 섹션 생성 실패: {e}")
            return "❌ 실제 수익률 섹션 생성 실패"
    
    def generate_report_header(self, total_stocks: int) -> str:
        """리포트 헤더 생성"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"""# 📊 향상된 4개 카테고리 스톡 분석 리포트

**생성 시간:** {current_time}  
**분석 종목 수:** {total_stocks}개  
**리포트 구성:** 각 카테고리별 상위 15개 종목  

## 📋 분석 카테고리
1. **점수 변동 큰 순**: ML 파라미터 변화에 가장 민감한 종목들
2. **현재 점수 높은 순**: Sweet Spot 알고리즘 기준 최고 점수 종목들  
3. **예상 수익률 높은 순**: ML 예측 기준 수익률 전망이 좋은 종목들
4. **실제 수익률 높은 순**: 현재까지 실제 성과가 우수한 종목들"""
    
    def generate_summary_section(self, stock_analyses: List[StockAnalysis]) -> str:
        """요약 섹션 생성"""
        try:
            total_stocks = len(stock_analyses)
            
            # 전체 통계
            avg_score = np.mean([s.current_score for s in stock_analyses])
            score_changes = [s.score_change for s in stock_analyses]
            avg_score_change = np.mean(score_changes)
            
            # 수익률 통계
            actual_returns = [s.actual_return for s in stock_analyses if s.actual_return is not None]
            expected_returns = [s.expected_return for s in stock_analyses if s.expected_return is not None]
            
            avg_actual = np.mean(actual_returns) if actual_returns else 0
            avg_expected = np.mean(expected_returns) if expected_returns else 0
            
            # 카테고리별 분포
            category_dist = {}
            for stock in stock_analyses:
                cat = stock.tech_category
                category_dist[cat] = category_dist.get(cat, 0) + 1
            
            top_categories = sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            
            summary = f"""## 🎯 전체 요약

**📊 기본 통계:**
- 평균 점수: {avg_score:.1f}점
- 평균 점수 변동: {avg_score_change:+.2f}점
- 실제 수익률 보유: {len(actual_returns)}개 종목 (평균: {avg_actual:+.1f}%)
- 예측 수익률 보유: {len(expected_returns)}개 종목 (평균: {avg_expected:+.1f}%)

**🏢 딥테크 카테고리 분포:**"""
            
            for cat, count in top_categories:
                percentage = count / total_stocks * 100
                summary += f"\n- {cat}: {count}개 ({percentage:.1f}%)"
            
            # 성과 하이라이트
            if actual_returns:
                best_performer = max(stock_analyses, key=lambda x: x.actual_return if x.actual_return else -100)
                worst_performer = min([s for s in stock_analyses if s.actual_return is not None], 
                                    key=lambda x: x.actual_return)
                
                summary += f"""

**🏆 성과 하이라이트:**
- 최고 수익: {best_performer.symbol} ({best_performer.actual_return:+.1f}%)
- 최저 수익: {worst_performer.symbol} ({worst_performer.actual_return:+.1f}%)"""
            
            return summary
            
        except Exception as e:
            logger.error(f"요약 섹션 생성 실패: {e}")
            return "❌ 요약 섹션 생성 실패"
    
    def generate_report_footer(self) -> str:
        """리포트 푸터 생성"""
        return f"""---

## ℹ️ 범례 및 참고사항

**아이콘 설명:**
- 📈 상승 / 📉 하락 / ➡️ 변동없음
- 🎯 높은 정확도 / ⚖️ 보통 정확도 / ❌ 낮은 정확도
- 🔥 거래량 급증 / 📊 거래량 트렌드 / - 신호없음

**투자 유의사항:**
- 본 리포트는 참고용이며 투자 권유가 아닙니다
- 실제 투자시 추가적인 분석과 위험 관리가 필요합니다
- ML 예측은 과거 데이터 기반이며 미래 수익을 보장하지 않습니다

**데이터 출처:**
- Sweet Spot Database: 종목 선정 및 점수 정보
- ML Predictions: 머신러닝 기반 수익률 예측
- Real-time Data: yfinance 및 FMP API

*Report Generated by Enhanced ML+AI System v4.0*"""

# 독립 실행 테스트
async def main():
    """테스트 실행"""
    generator = EnhancedReportGenerator()
    
    logger.info("=== 향상된 리포트 생성기 테스트 ===")
    
    # 리포트 생성
    report = await generator.generate_enhanced_report()
    
    if "❌" not in report:
        logger.info("✅ 리포트 생성 성공")
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_report_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 리포트 저장: {filename}")
        
        # 미리보기 출력
        print("\n" + "="*60)
        print(report[:1000] + "..." if len(report) > 1000 else report)
        print("="*60)
    else:
        logger.error("❌ 리포트 생성 실패")
        print(report)

if __name__ == "__main__":
    asyncio.run(main())