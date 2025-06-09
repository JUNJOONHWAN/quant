#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini 갭필터 부스터
AI 독립 판단 vs 실제 VTNF 델타 분석
"""
import re
import json
import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class GeminiGapFilterBooster:
    """Gemini Flash로 VTNF 갭 분석 (AI 독립 판단 vs 실제 VTNF 델타)"""

    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.gap_cache = {}  # 10분간 캐싱
        self.cache_duration = timedelta(minutes=10)
        
        # 갭 임계값 설정
        self.gap_thresholds = {
            'minor': 0.5,     # 작은 갭
            'moderate': 1.0,  # 중간 갭
            'major': 1.5,     # 큰 갭
            'extreme': 2.0    # 극단적 갭
        }
        
        # 부스트 강도 설정
        self.boost_settings = {
            'max_boost': 2.0,      # 최대 부스트
            'confidence_weight': 0.3,  # 신뢰도 가중치
            'delta_weight': 0.4,       # 델타 가중치
            'comprehensive_weight': 0.3  # 종합 판단 가중치
        }
        
        logger.info("🎯 Gemini 갭필터 부스터 초기화 완료")

    async def calculate_vtnf_gap_boost(self, symbol: str, vtnf_scores: dict) -> dict:
        """✅ AI 독립 판단 vs VTNF 실제값 델타 계산"""
        try:
            logger.info(f"🎯 {symbol} AI 델타 갭필터 분석 시작...")
            
            # 캐시 확인
            cache_key = f"{symbol}_gap_{datetime.now().strftime('%H:%M')[:-1]}0"
            if cache_key in self.gap_cache:
                cached_time, cached_result = self.gap_cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    logger.info(f"📋 {symbol} 갭 분석 캐시 사용")
                    return cached_result

            # 1. ✅ AI의 독립적인 종합 판단 (섹터 무관)
            ai_comprehensive_query = self._build_comprehensive_query(symbol)
            
            # 2. ✅ AI의 개별 VTNF 예측 (실제값과 비교용)
            ai_vtnf_predictions = self._build_vtnf_prediction_queries(symbol)
            
            # 3. 병렬 API 호출
            comprehensive_task = self._get_ai_score(ai_comprehensive_query)
            prediction_tasks = {k: self._get_ai_score(v) for k, v in ai_vtnf_predictions.items()}
            
            # 결과 수집
            ai_comprehensive = await comprehensive_task
            ai_predictions = {}
            for k, task in prediction_tasks.items():
                ai_predictions[k.lower()] = await task
                
            # 4. ✅ 실제 VTNF 값 정규화 (소문자로 통일)
            actual_vtnf = {k.lower(): float(v) for k, v in vtnf_scores.items() if v is not None}
            
            # 5. ✅ 델타 계산 (AI 예측 - 실제값)
            deltas = {}
            total_delta = 0
            valid_factors = 0
            
            for factor in ['v', 't', 'n', 'f']:
                if factor in actual_vtnf and factor in ai_predictions:
                    predicted = float(ai_predictions[factor])
                    actual = float(actual_vtnf[factor])
                    delta = predicted - actual
                    deltas[factor] = delta
                    total_delta += delta
                    valid_factors += 1
                else:
                    deltas[factor] = 0.0
                    
            avg_delta = total_delta / max(valid_factors, 1)
            
            # 6. ✅ AI 종합 판단 vs VTNF 평균의 갭
            actual_vtnf_avg = sum(actual_vtnf.values()) / max(len(actual_vtnf), 1)
            comprehensive_gap = ai_comprehensive - actual_vtnf_avg
            
            # 7. ✅ 통합 갭 스코어 계산
            # AI가 개별 요소들을 다르게 보는 정도(델타) + 종합 판단의 차이
            integrated_gap = (comprehensive_gap * 0.6) + (avg_delta * 0.4)
            
            # 로깅
            logger.info(f"🔍 {symbol} AI 델타 분석:")
            logger.info(f"   AI 종합 판단: {ai_comprehensive}")
            logger.info(f"   실제 VTNF 평균: {actual_vtnf_avg:.2f}")
            logger.info(f"   종합 갭: {comprehensive_gap:+.2f}")
            logger.info(f"   AI 예측 VTNF: {ai_predictions}")
            logger.info(f"   실제 VTNF: {actual_vtnf}")
            logger.info(f"   델타: {deltas}")
            logger.info(f"   평균 델타: {avg_delta:+.2f}")
            logger.info(f"   통합 갭: {integrated_gap:+.2f}")
            
            # 8. ✅ 갭 유의성 평가
            gap_significance = self._evaluate_gap_significance(
                integrated_gap, comprehensive_gap, avg_delta, deltas
            )
            
            # 9. ✅ 부스트 결정
            result = self._calculate_boost_result(
                symbol, integrated_gap, comprehensive_gap, avg_delta, deltas,
                gap_significance, ai_comprehensive, ai_predictions, actual_vtnf
            )
            
            # 캐시 저장
            self.gap_cache[cache_key] = (datetime.now(), result)
            
            return result
                
        except Exception as e:
            logger.error(f"❌ {symbol} AI 델타 갭필터 실패: {e}")
            return self._get_error_result(symbol, str(e))

    def _build_comprehensive_query(self, symbol: str) -> str:
        """종합 판단 쿼리 구성"""
        return f"""
You are an elite institutional portfolio manager analyzing {symbol} with complete independence from sector classifications.

**COMPREHENSIVE INVESTMENT ANALYSIS**

Analyze {symbol} based on its unique characteristics, not sector stereotypes:

**Company-Specific Dynamics**
- What unique business model or strategic positioning does {symbol} have?
- How does it differ from typical companies in its sector?
- What company-specific catalysts or risks exist?

**Cross-Sector Opportunities**
- Does {symbol} operate across multiple sectors or have hybrid characteristics?
- Are there technology/innovation elements regardless of sector classification?
- What non-traditional revenue streams or growth drivers exist?

**Market Perception vs Reality**
- Is the market categorizing this company correctly?
- What aspects might algorithmic trading or sector ETFs be missing?
- Are there hidden value drivers the market overlooks?

**Adaptive Scoring Framework**
Instead of sector-based analysis, evaluate based on:
- Innovation & Disruption Potential (regardless of sector)
- Execution & Management Quality
- Market Position & Competitive Moat
- Financial Flexibility & Capital Efficiency
- Catalyst Timing & Momentum

**INDEPENDENT CONVICTION SCORE (0-10):**
Rate {symbol} purely on its individual merits, ignoring sector conventions.
Consider how this specific company might surprise the market.

CONVICTION SCORE: [0-10]
"""

    def _build_vtnf_prediction_queries(self, symbol: str) -> dict:
        """VTNF 예측 쿼리 구성"""
        return {
            'V': f"""Analyze {symbol}'s FUNDAMENTAL VALUE ignoring sector norms.
Consider: unique financial metrics, hidden assets, cash generation ability, 
non-traditional value drivers specific to this company.
Look at P/E ratios, revenue growth, profit margins, debt levels, but focus on 
what makes THIS company different from sector averages.
PREDICTED V SCORE (0-10):""",
            
            'T': f"""Analyze {symbol}'s TECHNICAL DYNAMICS beyond sector patterns.
Consider: unique price behavior, volume anomalies, algorithmic trading patterns,
company-specific technical setups that differ from sector.
Evaluate moving averages, RSI, MACD, volume patterns specific to this stock.
PREDICTED T SCORE (0-10):""",
            
            'N': f"""Analyze {symbol}'s NEWS & SENTIMENT with fresh perspective.
Consider: company-specific catalysts, management changes, product launches,
partnerships that transcend typical sector news flow.
Recent earnings, analyst upgrades/downgrades, industry disruption potential.
PREDICTED N SCORE (0-10):""",
            
            'F': f"""Analyze {symbol}'s INSTITUTIONAL FLOWS independently.
Consider: smart money movements specific to this stock, unusual options activity,
insider behavior that differs from sector trends.
ETF flows, hedge fund positions, institutional ownership changes.
PREDICTED F SCORE (0-10):"""
        }

    async def _get_ai_score(self, query: str) -> float:
        """AI API 호출하여 점수 추출"""
        try:
            gemini_config = self.api_manager.get_gemini_config()
            api_key = gemini_config['api_key']
            base_url = gemini_config['base_url']
            
            if not api_key or api_key == 'GEM':
                logger.warning("Gemini API 키 없음, 기본값 반환")
                return 6.0
            
            url = f"{base_url}/models/gemini-2.0-flash-exp:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            
            payload = {
                "contents": [{
                    "parts": [{"text": query}]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "topK": 1,
                    "topP": 0.1,
                    "maxOutputTokens": 200
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                        
                        # 점수 추출
                        score = self._extract_score_from_text(content)
                        return score
                    else:
                        logger.warning(f"Gemini API 오류: {response.status}")
                        return 6.0
                        
        except Exception as e:
            logger.warning(f"AI 점수 추출 실패: {e}")
            return 6.0

    def _extract_score_from_text(self, text: str) -> float:
        """텍스트에서 점수 추출"""
        try:
            # 다양한 패턴으로 점수 추출 시도
            patterns = [
                r'CONVICTION SCORE:\s*(\d+(?:\.\d+)?)',
                r'PREDICTED [VTNF] SCORE.*?(\d+(?:\.\d+)?)',
                r'SCORE.*?(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)/10',
                r'(\d+(?:\.\d+)?)\s*점',
                r'(\d+(?:\.\d+)?)\s*out of 10'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    score = float(matches[0])
                    # 점수 범위 검증 및 정규화
                    if 0 <= score <= 10:
                        return score
                    elif score > 10:
                        return min(score / 10, 10.0) if score <= 100 else 10.0
            
            # 패턴이 없으면 숫자만 추출
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
            for num_str in numbers:
                num = float(num_str)
                if 0 <= num <= 10:
                    return num
                elif 10 < num <= 100:
                    return num / 10
            
            # 기본값
            return 6.0
            
        except Exception as e:
            logger.warning(f"점수 추출 실패: {e}")
            return 6.0

    def _evaluate_gap_significance(self, integrated_gap: float, comprehensive_gap: float, 
                                  avg_delta: float, deltas: dict) -> dict:
        """갭 유의성 평가"""
        try:
            # 1. 갭 크기 평가
            abs_integrated_gap = abs(integrated_gap)
            abs_comprehensive_gap = abs(comprehensive_gap)
            abs_avg_delta = abs(avg_delta)
            
            # 2. 임계값 기반 분류
            gap_level = 'minor'
            if abs_integrated_gap >= self.gap_thresholds['extreme']:
                gap_level = 'extreme'
            elif abs_integrated_gap >= self.gap_thresholds['major']:
                gap_level = 'major'
            elif abs_integrated_gap >= self.gap_thresholds['moderate']:
                gap_level = 'moderate'
            
            # 3. 일관성 평가 (모든 델타가 같은 방향인지)
            delta_values = [v for v in deltas.values() if v != 0]
            if delta_values:
                positive_deltas = sum(1 for d in delta_values if d > 0)
                negative_deltas = sum(1 for d in delta_values if d < 0)
                consistency = abs(positive_deltas - negative_deltas) / len(delta_values)
            else:
                consistency = 0
            
            # 4. 신뢰도 계산
            confidence = min(
                (abs_integrated_gap / self.gap_thresholds['moderate']) * 0.4 +
                consistency * 0.3 +
                (min(abs_comprehensive_gap, 2.0) / 2.0) * 0.3,
                1.0
            )
            
            # 5. 유의성 판단
            is_significant = (
                abs_integrated_gap >= self.gap_thresholds['minor'] and
                confidence >= 0.3
            )
            
            # 6. 갭 강도 계산
            strength = min(abs_integrated_gap, self.boost_settings['max_boost'])
            
            # 7. 발산 요소 분석
            divergence_factors = []
            if abs_comprehensive_gap > 1.0:
                divergence_factors.append('종합_판단_차이')
            if abs_avg_delta > 0.8:
                divergence_factors.append('개별_요소_차이')
            if consistency > 0.7:
                divergence_factors.append('일관된_방향성')
            
            return {
                'is_significant': is_significant,
                'gap_level': gap_level,
                'strength': strength,
                'confidence': confidence,
                'consistency': consistency,
                'divergence_factors': divergence_factors
            }
            
        except Exception as e:
            logger.error(f"갭 유의성 평가 실패: {e}")
            return {
                'is_significant': False,
                'gap_level': 'minor',
                'strength': 0,
                'confidence': 0,
                'consistency': 0,
                'divergence_factors': []
            }

    def _calculate_position_adjustment(self, integrated_gap: float, boost_strength: float, 
                                     confidence: float) -> float:
        """포지션 조정 비율 계산"""
        try:
            # 기본 조정 비율
            base_adjustment = integrated_gap * 10  # -20 ~ +20 범위
            
            # 신뢰도와 강도로 조정
            confidence_factor = confidence ** 0.5
            strength_factor = min(boost_strength / self.boost_settings['max_boost'], 1.0)
            
            # 최종 조정 비율
            position_adjustment = base_adjustment * confidence_factor * strength_factor
            
            # 범위 제한 (-30% ~ +30%)
            return max(-30, min(30, position_adjustment))
            
        except Exception as e:
            logger.error(f"포지션 조정 계산 실패: {e}")
            return 0.0

    def _calculate_boost_result(self, symbol: str, integrated_gap: float, comprehensive_gap: float,
                               avg_delta: float, deltas: dict, gap_significance: dict,
                               ai_comprehensive: float, ai_predictions: dict, actual_vtnf: dict) -> dict:
        """부스트 결과 계산"""
        try:
            if gap_significance['is_significant']:
                boost_type = 'synergy' if integrated_gap > 0 else 'risk'
                boost_strength = gap_significance['strength']
                position_adjustment = self._calculate_position_adjustment(
                    integrated_gap, boost_strength, gap_significance['confidence']
                )
                
                return {
                    'gap_detected': True,
                    'gap_score': round(integrated_gap, 2),
                    'integrated_gap': round(integrated_gap, 2),
                    'comprehensive_gap': round(comprehensive_gap, 2),
                    'avg_delta': round(avg_delta, 2),
                    'deltas': {k: round(v, 2) for k, v in deltas.items()},
                    'boost_type': boost_type,
                    'boost_strength': round(boost_strength, 2),
                    'position_adjustment': round(position_adjustment, 1),
                    'ai_comprehensive': ai_comprehensive,
                    'ai_predictions': ai_predictions,
                    'actual_vtnf': actual_vtnf,
                    'confidence': gap_significance['confidence'],
                    'gap_level': gap_significance['gap_level'],
                    'consistency': gap_significance['consistency'],
                    'divergence_factors': gap_significance['divergence_factors'],
                    'method': 'ai_independent_delta',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'gap_detected': False,
                    'gap_score': round(integrated_gap, 2),
                    'integrated_gap': round(integrated_gap, 2),
                    'comprehensive_gap': round(comprehensive_gap, 2),
                    'avg_delta': round(avg_delta, 2),
                    'deltas': {k: round(v, 2) for k, v in deltas.items()},
                    'boost_type': 'neutral',
                    'boost_strength': 0,
                    'position_adjustment': 0,
                    'confidence': gap_significance['confidence'],
                    'gap_level': gap_significance['gap_level'],
                    'method': 'ai_independent_delta',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"부스트 결과 계산 실패: {e}")
            return self._get_error_result(symbol, str(e))

    def _get_error_result(self, symbol: str, error_msg: str) -> dict:
        """에러 결과 반환"""
        return {
            'gap_detected': False,
            'gap_score': 0,
            'integrated_gap': 0,
            'comprehensive_gap': 0,
            'avg_delta': 0,
            'deltas': {'v': 0, 't': 0, 'n': 0, 'f': 0},
            'boost_type': 'error',
            'boost_strength': 0,
            'position_adjustment': 0,
            'confidence': 0,
            'gap_level': 'none',
            'error': error_msg,
            'method': 'ai_independent_delta',
            'timestamp': datetime.now().isoformat()
        }

    def clear_cache(self):
        """캐시 정리"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, (timestamp, _) in self.gap_cache.items():
            if current_time - timestamp > self.cache_duration:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.gap_cache[key]
        
        logger.info(f"🧹 갭 부스터 캐시 정리: {len(expired_keys)}개 항목 삭제")

    def get_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        return {
            'total_entries': len(self.gap_cache),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60,
            'gap_thresholds': self.gap_thresholds,
            'boost_settings': self.boost_settings
        }
