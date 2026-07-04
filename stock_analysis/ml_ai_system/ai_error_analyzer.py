#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 AI 오차 분석 엔진

예측 오차를 심층 분석하여 빗나간 이유를 파악하고
Perplexity AI를 활용해 시장 인사이트를 제공하는 시스템
"""

import json
import logging
import asyncio
import os
import requests
import aiohttp
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
class ErrorPattern:
    """오차 패턴"""
    pattern_type: str
    affected_symbols: List[str]
    common_characteristics: Dict[str, Any]
    error_magnitude: float
    frequency: int
    potential_causes: List[str]

@dataclass
class SectorAnalysis:
    """섹터별 분석"""
    sector: str
    symbol_count: int
    average_error: float
    worst_performer: str
    best_performer: str
    sector_bias: float  # 과대/과소평가 정도
    market_events: List[str]

@dataclass
class AIAnalysisResult:
    """AI 분석 결과"""
    analysis_date: str
    total_errors_analyzed: int
    major_error_patterns: List[ErrorPattern]
    sector_analysis: List[SectorAnalysis]
    market_context: Dict[str, Any]
    ai_insights: Dict[str, str]
    parameter_adjustment_suggestions: Dict[str, float]
    confidence_score: float

class AIErrorAnalyzer:
    """AI 오차 분석 엔진"""
    
    def __init__(self):
        self.perplexity_api_key = os.getenv('PPPP_API_KEY', '')
        self.perplexity_base_url = "https://api.perplexity.ai/chat/completions"
        
        # 분석 결과 저장 파일
        self.analysis_history_file = "ai_analysis_history.json"
        
        # 오차 임계값 설정
        self.high_error_threshold = 10.0  # 10% 이상 오차는 큰 오차
        self.direction_error_weight = 2.0  # 방향 틀렸을 때 가중치
        
        # 분석 히스토리 로드
        self.analysis_history = self.load_analysis_history()
    
    def load_analysis_history(self) -> List[Dict]:
        """분석 히스토리 로드"""
        try:
            if os.path.exists(self.analysis_history_file):
                with open(self.analysis_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"분석 히스토리 로드 실패: {e}")
            return []
    
    def save_analysis_history(self):
        """분석 히스토리 저장"""
        try:
            with open(self.analysis_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_history, f, indent=2, ensure_ascii=False)
            logger.info(f"분석 히스토리 저장 완료: {len(self.analysis_history)}개 기록")
        except Exception as e:
            logger.error(f"분석 히스토리 저장 실패: {e}")
    
    async def analyze_prediction_errors(self, validation_results: List[Dict]) -> AIAnalysisResult:
        """예측 오차 심층 분석"""
        try:
            logger.info(f"🔍 AI 오차 분석 시작: {len(validation_results)}개 예측 분석")
            
            # 1. 오차 패턴 추출
            error_patterns = self.extract_error_patterns(validation_results)
            
            # 2. 섹터별 분석
            sector_analysis = await self.analyze_by_sector(validation_results)
            
            # 3. 시장 컨텍스트 분석
            market_context = await self.analyze_market_context(validation_results)
            
            # 4. Perplexity AI 심층 분석
            ai_insights = await self.get_perplexity_insights(
                error_patterns, sector_analysis, market_context
            )
            
            # 5. 파라미터 조정 제안 생성
            adjustment_suggestions = self.generate_parameter_adjustments(
                error_patterns, sector_analysis, ai_insights
            )
            
            # 6. 신뢰도 점수 계산
            confidence_score = self.calculate_analysis_confidence(
                error_patterns, ai_insights
            )
            
            analysis_result = AIAnalysisResult(
                analysis_date=datetime.now().strftime('%Y-%m-%d'),
                total_errors_analyzed=len(validation_results),
                major_error_patterns=error_patterns,
                sector_analysis=sector_analysis,
                market_context=market_context,
                ai_insights=ai_insights,
                parameter_adjustment_suggestions=adjustment_suggestions,
                confidence_score=confidence_score
            )
            
            # 분석 결과 저장
            self.analysis_history.append(asdict(analysis_result))
            self.save_analysis_history()
            
            logger.info(f"✅ AI 오차 분석 완료: 신뢰도 {confidence_score:.1%}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"AI 오차 분석 실패: {e}")
            return None
    
    def extract_error_patterns(self, validation_results: List[Dict]) -> List[ErrorPattern]:
        """오차 패턴 추출"""
        try:
            patterns = []
            
            # 큰 오차 종목들 분석
            high_error_results = [
                r for r in validation_results 
                if r.get('error', 0) > self.high_error_threshold
            ]
            
            if high_error_results:
                # 과대평가 패턴 (예측이 실제보다 높음)
                overestimated = [
                    r for r in high_error_results
                    if r['next_week_expected_return'] > r['actual_return']
                ]
                
                if overestimated:
                    common_chars = self.find_common_characteristics(overestimated)
                    patterns.append(ErrorPattern(
                        pattern_type="과대평가",
                        affected_symbols=[r['symbol'] for r in overestimated],
                        common_characteristics=common_chars,
                        error_magnitude=np.mean([r['error'] for r in overestimated]),
                        frequency=len(overestimated),
                        potential_causes=self.identify_overestimation_causes(common_chars)
                    ))
                
                # 과소평가 패턴 (예측이 실제보다 낮음)
                underestimated = [
                    r for r in high_error_results
                    if r['next_week_expected_return'] < r['actual_return']
                ]
                
                if underestimated:
                    common_chars = self.find_common_characteristics(underestimated)
                    patterns.append(ErrorPattern(
                        pattern_type="과소평가",
                        affected_symbols=[r['symbol'] for r in underestimated],
                        common_characteristics=common_chars,
                        error_magnitude=np.mean([r['error'] for r in underestimated]),
                        frequency=len(underestimated),
                        potential_causes=self.identify_underestimation_causes(common_chars)
                    ))
            
            # 방향성 오류 패턴
            direction_errors = [
                r for r in validation_results
                if not r.get('direction_correct', True)
            ]
            
            if direction_errors:
                common_chars = self.find_common_characteristics(direction_errors)
                patterns.append(ErrorPattern(
                    pattern_type="방향성_오류",
                    affected_symbols=[r['symbol'] for r in direction_errors],
                    common_characteristics=common_chars,
                    error_magnitude=np.mean([r['error'] for r in direction_errors]) * self.direction_error_weight,
                    frequency=len(direction_errors),
                    potential_causes=self.identify_direction_error_causes(common_chars)
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"오차 패턴 추출 실패: {e}")
            return []
    
    def find_common_characteristics(self, error_results: List[Dict]) -> Dict[str, Any]:
        """오차 종목들의 공통 특성 찾기"""
        try:
            if not error_results:
                return {}
            
            # Sweet Spot DB에서 종목 정보 가져오기
            sweet_spot_db = self.load_sweet_spot_database()
            symbol_info = {stock['symbol']: stock for stock in sweet_spot_db.get('picks', [])}
            
            characteristics = {}
            symbols = [r['symbol'] for r in error_results]
            
            # 딥테크 카테고리 분포
            categories = []
            recovery_stages = []
            scores = []
            recovery_percents = []
            
            for symbol in symbols:
                if symbol in symbol_info:
                    info = symbol_info[symbol]
                    categories.append(info.get('tech_category', 'unknown'))
                    recovery_stages.append(info.get('recovery_stage', 'unknown'))
                    scores.append(info.get('selection_score', 0))
                    recovery_percents.append(info.get('recovery_from_low_percent', 0))
            
            if categories:
                characteristics['dominant_category'] = max(set(categories), key=categories.count)
                characteristics['category_distribution'] = dict(zip(*np.unique(categories, return_counts=True)))
            
            if recovery_stages:
                characteristics['dominant_recovery_stage'] = max(set(recovery_stages), key=recovery_stages.count)
            
            if scores:
                characteristics['average_score'] = np.mean(scores)
                characteristics['score_range'] = [min(scores), max(scores)]
            
            if recovery_percents:
                characteristics['average_recovery_percent'] = np.mean(recovery_percents)
                characteristics['recovery_range'] = [min(recovery_percents), max(recovery_percents)]
            
            # 예측 관련 특성
            predictions = [r['next_week_expected_return'] for r in error_results]
            actuals = [r['actual_return'] for r in error_results]
            
            characteristics['prediction_range'] = [min(predictions), max(predictions)]
            characteristics['actual_range'] = [min(actuals), max(actuals)]
            characteristics['prediction_bias'] = np.mean(predictions) - np.mean(actuals)
            
            return characteristics
            
        except Exception as e:
            logger.warning(f"공통 특성 찾기 실패: {e}")
            return {}
    
    def identify_overestimation_causes(self, characteristics: Dict) -> List[str]:
        """과대평가 원인 파악"""
        causes = []
        
        # 딥테크 카테고리별 원인
        dominant_category = characteristics.get('dominant_category')
        if dominant_category in ['quantum_tech', 'ai_computing']:
            causes.append("하이프 섹터 - 기대와 현실의 괴리")
        
        # 회복 단계별 원인
        avg_recovery = characteristics.get('average_recovery_percent', 0)
        if avg_recovery > 100:
            causes.append("과열 구간 - Sweet Spot 이탈")
        
        # 점수별 원인
        avg_score = characteristics.get('average_score', 0)
        if avg_score > 70:
            causes.append("높은 점수 편향 - 과도한 낙관론")
        
        # 예측 편향
        bias = characteristics.get('prediction_bias', 0)
        if bias > 5:
            causes.append("시스템적 낙관 편향")
        
        return causes if causes else ["원인 불명"]
    
    def identify_underestimation_causes(self, characteristics: Dict) -> List[str]:
        """과소평가 원인 파악"""
        causes = []
        
        # 회복 단계별 원인
        avg_recovery = characteristics.get('average_recovery_percent', 0)
        if avg_recovery < 30:
            causes.append("초기 회복 가속도 과소평가")
        
        # 딥테크 카테고리별 원인
        dominant_category = characteristics.get('dominant_category')
        if dominant_category in ['bio_health_tech', 'mobility_tech']:
            causes.append("기술 혁신 속도 과소평가")
        
        # 점수별 원인
        avg_score = characteristics.get('average_score', 0)
        if avg_score < 50:
            causes.append("낮은 점수 편향 - 과도한 보수주의")
        
        return causes if causes else ["돌발 호재 발생"]
    
    def identify_direction_error_causes(self, characteristics: Dict) -> List[str]:
        """방향성 오류 원인 파악"""
        causes = []
        
        # 회복 단계별 원인
        dominant_stage = characteristics.get('dominant_recovery_stage')
        if dominant_stage == 'late':
            causes.append("후기 회복 단계 - 반전 위험 증가")
        
        # 예측 범위별 원인
        pred_range = characteristics.get('prediction_range', [0, 0])
        if abs(pred_range[1] - pred_range[0]) > 20:
            causes.append("예측 불확실성 높음")
        
        causes.append("예상치 못한 시장 이벤트")
        
        return causes
    
    async def analyze_by_sector(self, validation_results: List[Dict]) -> List[SectorAnalysis]:
        """섹터별 분석"""
        try:
            # Sweet Spot DB에서 섹터 정보 가져오기
            sweet_spot_db = self.load_sweet_spot_database()
            symbol_info = {stock['symbol']: stock for stock in sweet_spot_db.get('picks', [])}
            
            # 섹터별 그룹화
            sector_groups = {}
            for result in validation_results:
                symbol = result['symbol']
                sector = symbol_info.get(symbol, {}).get('tech_category', 'unknown')
                
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(result)
            
            sector_analyses = []
            for sector, results in sector_groups.items():
                if len(results) < 2:  # 최소 2개 종목 이상
                    continue
                
                errors = [r['error'] for r in results]
                predictions = [r['next_week_expected_return'] for r in results]
                actuals = [r['actual_return'] for r in results]
                
                # 최고/최악 종목
                sorted_by_error = sorted(results, key=lambda x: x['error'])
                best_performer = sorted_by_error[0]['symbol']
                worst_performer = sorted_by_error[-1]['symbol']
                
                # 섹터 편향 (예측 - 실제)
                sector_bias = np.mean(predictions) - np.mean(actuals)
                
                # 해당 섹터 시장 이벤트 (추후 Perplexity AI로 보강)
                market_events = await self.get_sector_events(sector)
                
                sector_analyses.append(SectorAnalysis(
                    sector=sector,
                    symbol_count=len(results),
                    average_error=np.mean(errors),
                    worst_performer=worst_performer,
                    best_performer=best_performer,
                    sector_bias=sector_bias,
                    market_events=market_events
                ))
            
            return sorted(sector_analyses, key=lambda x: x.average_error, reverse=True)
            
        except Exception as e:
            logger.error(f"섹터별 분석 실패: {e}")
            return []
    
    async def get_sector_events(self, sector: str) -> List[str]:
        """섹터별 주요 이벤트 조회 (간단 버전)"""
        try:
            # 섹터별 일반적인 이벤트들 (추후 실시간 뉴스 API로 보강 가능)
            sector_events_map = {
                'ai_computing': ['AI 칩 수요 급증', 'ChatGPT 열풍', '엔비디아 실적'],
                'quantum_tech': ['양자컴퓨팅 투자 증가', 'IBM 양자 발표', '정부 R&D 지원'],
                'bio_health_tech': ['FDA 승인 소식', '임상 결과 발표', '바이오투자 확대'],
                'mobility_tech': ['자율주행 규제 완화', 'EV 보급 확산', '배터리 기술 혁신'],
                'semiconductor': ['반도체 슈퍼사이클', '지정학적 긴장', 'CHIPS Act'],
                'energy_materials': ['ESG 투자 증가', '배터리 소재 수급', '탄소중립 정책']
            }
            
            return sector_events_map.get(sector, ['일반적 시장 변동'])
            
        except:
            return ['이벤트 정보 없음']
    
    async def analyze_market_context(self, validation_results: List[Dict]) -> Dict[str, Any]:
        """시장 컨텍스트 분석"""
        try:
            context = {}
            
            # 분석 기간 설정
            if validation_results:
                pred_dates = [r['prediction_date'] for r in validation_results]
                start_date = min(pred_dates)
                end_date = max(pred_dates)
                context['analysis_period'] = f"{start_date} ~ {end_date}"
            
            # 전체 시장 성과
            all_predictions = [r['next_week_expected_return'] for r in validation_results]
            all_actuals = [r['actual_return'] for r in validation_results]
            
            context['market_sentiment'] = {
                'predicted_avg': np.mean(all_predictions),
                'actual_avg': np.mean(all_actuals),
                'sentiment_shift': np.mean(all_actuals) - np.mean(all_predictions)
            }
            
            # 변동성 분석
            context['volatility'] = {
                'prediction_volatility': np.std(all_predictions),
                'actual_volatility': np.std(all_actuals),
                'volatility_surprise': np.std(all_actuals) - np.std(all_predictions)
            }
            
            # 방향성 분석
            direction_correct = [r.get('direction_correct', False) for r in validation_results]
            context['direction_analysis'] = {
                'accuracy': np.mean(direction_correct),
                'total_predictions': len(direction_correct),
                'correct_predictions': sum(direction_correct)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"시장 컨텍스트 분석 실패: {e}")
            return {}
    
    async def get_perplexity_insights(self, error_patterns: List[ErrorPattern], 
                                   sector_analysis: List[SectorAnalysis], 
                                   market_context: Dict) -> Dict[str, str]:
        """Perplexity AI 심층 분석"""
        try:
            if not self.perplexity_api_key:
                logger.warning("Perplexity API 키가 없어 기본 분석으로 대체")
                return self.get_fallback_insights(error_patterns, sector_analysis)
            
            # AI 분석 프롬프트 생성
            prompt = self.create_analysis_prompt(error_patterns, sector_analysis, market_context)
            
            # Perplexity API 호출
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.perplexity_api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 주식 시장 전문가입니다. 예측 오차 패턴을 분석하여 구체적이고 실행 가능한 인사이트를 제공해주세요."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3
                }
                
                async with session.post(self.perplexity_base_url, 
                                      headers=headers, 
                                      json=payload,
                                      timeout=30) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        ai_content = result['choices'][0]['message']['content']
                        return self.parse_ai_response(ai_content)
                    else:
                        logger.error(f"Perplexity API 오류: {response.status}")
                        return self.get_fallback_insights(error_patterns, sector_analysis)
        
        except Exception as e:
            logger.error(f"Perplexity AI 분석 실패: {e}")
            return self.get_fallback_insights(error_patterns, sector_analysis)
    
    def create_analysis_prompt(self, error_patterns: List[ErrorPattern], 
                             sector_analysis: List[SectorAnalysis], 
                             market_context: Dict) -> str:
        """AI 분석 프롬프트 생성"""
        prompt = f"""
주식 예측 오차 패턴을 분석해주세요:

## 오차 패턴 요약:
"""
        
        for pattern in error_patterns[:3]:  # 상위 3개 패턴
            prompt += f"""
- **{pattern.pattern_type}**: {pattern.frequency}개 종목, 평균 오차 {pattern.error_magnitude:.1f}%
  - 영향받은 종목: {', '.join(pattern.affected_symbols[:5])}
  - 추정 원인: {', '.join(pattern.potential_causes)}
"""
        
        prompt += f"""
## 섹터별 성과:
"""
        for sector in sector_analysis[:3]:  # 상위 3개 섹터
            prompt += f"""
- **{sector.sector}**: 평균 오차 {sector.average_error:.1f}%, 편향 {sector.sector_bias:+.1f}%
  - 최악: {sector.worst_performer}, 최고: {sector.best_performer}
"""
        
        prompt += f"""
## 시장 컨텍스트:
- 예측 평균: {market_context.get('market_sentiment', {}).get('predicted_avg', 0):.1f}%
- 실제 평균: {market_context.get('market_sentiment', {}).get('actual_avg', 0):.1f}%
- 방향 정확도: {market_context.get('direction_analysis', {}).get('accuracy', 0):.1%}

질문:
1. 왜 이런 오차 패턴이 발생했을까요?
2. 현재 시장에서 놓치고 있는 주요 트렌드는 무엇인가요?
3. ML 파라미터를 어떻게 조정해야 할까요?
4. 앞으로 주의해야 할 점은 무엇인가요?

구체적이고 실행 가능한 답변을 부탁드립니다.
"""
        
        return prompt
    
    def parse_ai_response(self, ai_content: str) -> Dict[str, str]:
        """AI 응답 파싱"""
        try:
            # 간단한 섹션별 파싱
            insights = {}
            
            sections = ['오차 원인', '놓친 트렌드', '파라미터 조정', '주의사항']
            current_section = 'general'
            current_content = []
            
            lines = ai_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 섹션 헤더 감지
                for section in sections:
                    if section in line or any(keyword in line for keyword in ['1.', '2.', '3.', '4.']):
                        if current_content:
                            insights[current_section] = ' '.join(current_content)
                        current_section = section.replace(' ', '_').lower()
                        current_content = []
                        break
                
                current_content.append(line)
            
            # 마지막 섹션 처리
            if current_content:
                insights[current_section] = ' '.join(current_content)
            
            # 전체 분석이 하나로 되어있는 경우
            if not insights or len(insights) == 1:
                insights = {
                    'overall_analysis': ai_content,
                    'summary': ai_content[:200] + '...' if len(ai_content) > 200 else ai_content
                }
            
            return insights
            
        except Exception as e:
            logger.warning(f"AI 응답 파싱 실패: {e}")
            return {'raw_response': ai_content}
    
    def get_fallback_insights(self, error_patterns: List[ErrorPattern], 
                            sector_analysis: List[SectorAnalysis]) -> Dict[str, str]:
        """Perplexity API 실패시 대체 분석"""
        insights = {}
        
        # 기본 패턴 분석
        if error_patterns:
            dominant_pattern = max(error_patterns, key=lambda x: x.frequency)
            insights['dominant_error'] = f"{dominant_pattern.pattern_type}: {dominant_pattern.frequency}개 종목 영향"
            insights['main_cause'] = ', '.join(dominant_pattern.potential_causes[:2])
        
        # 섹터 분석
        if sector_analysis:
            worst_sector = max(sector_analysis, key=lambda x: x.average_error)
            insights['worst_sector'] = f"{worst_sector.sector}: 평균 {worst_sector.average_error:.1f}% 오차"
        
        # 일반적 권장사항
        insights['general_recommendation'] = "시장 변동성 증가, 파라미터 조정 필요"
        
        return insights
    
    def generate_parameter_adjustments(self, error_patterns: List[ErrorPattern],
                                     sector_analysis: List[SectorAnalysis],
                                     ai_insights: Dict[str, str]) -> Dict[str, float]:
        """파라미터 조정 제안 생성"""
        try:
            adjustments = {}
            
            # 오차 패턴 기반 조정
            for pattern in error_patterns:
                if pattern.pattern_type == "과대평가":
                    # 낙관 편향 완화
                    adjustments['golden_time_multiplier'] = -0.15
                    adjustments['pattern_score_weight'] = -0.05
                    
                elif pattern.pattern_type == "과소평가":
                    # 보수 편향 완화
                    adjustments['early_recovery_multiplier'] = +0.10
                    adjustments['growth_score_weight'] = +0.05
                    
                elif pattern.pattern_type == "방향성_오류":
                    # 기술지표 가중치 증가
                    adjustments['convergence_score_weight'] = +0.05
                    adjustments['technical_confluence_weight'] = +0.03
            
            # 섹터별 조정
            for sector in sector_analysis:
                if sector.average_error > 8.0:  # 8% 이상 오차
                    # 해당 섹터 배수 감소
                    if sector.sector_bias > 5.0:  # 과대평가
                        adjustments[f'{sector.sector}_multiplier'] = -0.10
                    elif sector.sector_bias < -5.0:  # 과소평가
                        adjustments[f'{sector.sector}_multiplier'] = +0.10
            
            # AI 인사이트 기반 추가 조정
            ai_text = ' '.join(ai_insights.values()).lower()
            
            if '거래량' in ai_text or 'volume' in ai_text:
                adjustments['volume_signal_weight'] = +0.05
                
            if '변동성' in ai_text or 'volatility' in ai_text:
                adjustments['volatility_compression_weight'] = +0.03
                
            if '기관' in ai_text or 'institutional' in ai_text:
                adjustments['institutional_score_weight'] = +0.02
            
            return adjustments
            
        except Exception as e:
            logger.error(f"파라미터 조정 제안 실패: {e}")
            return {}
    
    def calculate_analysis_confidence(self, error_patterns: List[ErrorPattern],
                                    ai_insights: Dict[str, str]) -> float:
        """분석 신뢰도 계산"""
        try:
            confidence = 0.5  # 기본값
            
            # 패턴의 명확성
            if error_patterns:
                pattern_strength = sum(p.frequency for p in error_patterns) / len(error_patterns)
                confidence += min(0.3, pattern_strength * 0.05)
            
            # AI 인사이트 품질
            total_insight_length = sum(len(text) for text in ai_insights.values())
            if total_insight_length > 500:  # 충분한 분석량
                confidence += 0.15
            elif total_insight_length > 200:
                confidence += 0.10
            
            # Perplexity API 사용 여부
            if 'raw_response' not in ai_insights and ai_insights:
                confidence += 0.05  # API 사용시 약간의 보너스
            
            return max(0.3, min(0.95, confidence))
            
        except:
            return 0.6
    
    def load_sweet_spot_database(self) -> Dict:
        """Sweet Spot 데이터베이스 로드"""
        try:
            with open('sweet_spot_database.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Sweet Spot DB 로드 실패: {e}")
            return {}

# 독립 실행 테스트
async def main():
    """테스트 실행"""
    analyzer = AIErrorAnalyzer()
    
    # 테스트 데이터 생성
    test_validation_results = [
        {
            'symbol': 'IONQ',
            'prediction_date': '2025-09-02',
            'next_week_expected_return': 8.5,
            'actual_return': 3.2,
            'error': 5.3,
            'direction_correct': True
        },
        {
            'symbol': 'MBLY',
            'prediction_date': '2025-09-02',
            'next_week_expected_return': 5.2,
            'actual_return': -2.1,
            'error': 7.3,
            'direction_correct': False
        }
    ]
    
    logger.info("=== AI 오차 분석 엔진 테스트 ===")
    
    analysis = await analyzer.analyze_prediction_errors(test_validation_results)
    
    if analysis:
        logger.info(f"✅ 분석 완료: 신뢰도 {analysis.confidence_score:.1%}")
        logger.info(f"   주요 패턴: {len(analysis.major_error_patterns)}개")
        logger.info(f"   섹터 분석: {len(analysis.sector_analysis)}개")
        logger.info(f"   조정 제안: {len(analysis.parameter_adjustment_suggestions)}개")
    else:
        logger.error("❌ 분석 실패")

if __name__ == "__main__":
    asyncio.run(main())