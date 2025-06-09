#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini 2.5 Flash 패턴 인식 부스터
15분 자동매매 최적화 패턴 분석
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

class GeminiPatternRecognitionBooster:
    """🔍 Gemini 2.5 Flash 패턴 인식 부스터 (15분 자동매매 최적화)"""

    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.pattern_cache = {}  # 5분간 캐싱
        self.cache_duration = timedelta(minutes=5)
        
        # 지원하는 패턴 타입들
        self.pattern_types = [
            'Head_and_Shoulders', 'Inverse_Head_and_Shoulders',
            'Cup_and_Handle', 'Double_Top', 'Double_Bottom',
            'Ascending_Triangle', 'Descending_Triangle', 'Symmetrical_Triangle',
            'Bullish_Flag', 'Bearish_Flag', 'Bullish_Pennant', 'Bearish_Pennant',
            'Rising_Wedge', 'Falling_Wedge',
            'Bullish_Engulfing', 'Bearish_Engulfing',
            'Hammer', 'Inverted_Hammer', 'Doji', 'Shooting_Star',
            'Support_Breakout', 'Resistance_Breakout',
            'Channel_Up', 'Channel_Down', 'Sideways_Channel'
        ]
        
        # 패턴별 가중치
        self.pattern_weights = {
            'Head_and_Shoulders': 0.9, 'Inverse_Head_and_Shoulders': 0.9,
            'Cup_and_Handle': 0.8, 'Double_Top': 0.8, 'Double_Bottom': 0.8,
            'Ascending_Triangle': 0.7, 'Descending_Triangle': 0.7,
            'Bullish_Flag': 0.75, 'Bearish_Flag': 0.75,
            'Support_Breakout': 0.85, 'Resistance_Breakout': 0.85,
            'Bullish_Engulfing': 0.6, 'Bearish_Engulfing': 0.6,
            'default': 0.5
        }
        
        # 점수 계산 가중치
        self.score_weights = {
            'pattern_strength': 0.25,
            'breakout_probability': 0.3,
            'confidence': 0.2,
            'volume_confirmation': 0.15,
            'risk_assessment': 0.1
        }
        
        logger.info("🔍 Gemini 패턴 인식 부스터 초기화 완료")

    async def analyze_patterns_15min(self, symbol: str, thinking_budget: int = 5000) -> dict:
        """🎯 15분 자동매매용 패턴 분석 (Gemini 2.5 Flash)"""
        try:
            logger.info(f"🔍 {symbol} 패턴 인식 분석 시작 (thinking_budget: {thinking_budget})...")

            # 캐시 확인 (5분)
            cache_key = f"{symbol}_pattern_{datetime.now().strftime('%H:%M')[:-1]}0"
            if cache_key in self.pattern_cache:
                cached_time, cached_result = self.pattern_cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    logger.info(f"📋 {symbol} 패턴 캐시 사용")
                    return cached_result

            # 패턴 분석 쿼리 구성
            query = self._build_pattern_analysis_query(symbol)

            # Gemini 2.5 Flash with thinking budget
            result = await self._call_gemini_25_flash(query, thinking_budget=thinking_budget)

            if result.get('content'):
                pattern_data = self._parse_pattern_result(result['content'])

                # 패턴 점수 계산
                pattern_score = self._calculate_pattern_score(pattern_data)

                final_result = {
                    'pattern_detected': pattern_data.get('pattern', 'None'),
                    'pattern_strength': pattern_data.get('strength', 5),
                    'breakout_probability': pattern_data.get('breakout_prob', 50),
                    'direction_bias': pattern_data.get('direction', 'Neutral'),
                    'confidence': pattern_data.get('confidence', 50),
                    'entry_timing': pattern_data.get('timing', 'Wait'),
                    'support_level': pattern_data.get('support', 0),
                    'resistance_level': pattern_data.get('resistance', 0),
                    'volume_confirmation': pattern_data.get('volume_confirm', 'Weak'),
                    'risk_level': pattern_data.get('risk', 50),
                    'pattern_score': pattern_score,
                    'reasoning': pattern_data.get('reasoning', ''),
                    'additional_signals': pattern_data.get('additional_signals', []),
                    'timeframe_analysis': pattern_data.get('timeframe_analysis', {}),
                    'data_source': 'Gemini 2.5 Flash Pattern Recognition',
                    'thinking_budget_used': thinking_budget,
                    'cache_key': cache_key,
                    'timestamp': datetime.now().isoformat()
                }

                # 캐시 저장 (5분)
                self.pattern_cache[cache_key] = (datetime.now(), final_result)

                logger.info(f"✅ {symbol} 패턴 분석 완료: {final_result['pattern_detected']} (점수: {pattern_score:.1f})")
                return final_result
            else:
                logger.warning(f"⚠️ {symbol} 패턴 분석 실패, 기본값 사용")
                return self._get_neutral_pattern_result(symbol)

        except Exception as e:
            logger.error(f"❌ {symbol} 패턴 인식 실패: {str(e)}")
            return self._get_error_pattern_result(symbol, str(e))

    def _build_pattern_analysis_query(self, symbol: str) -> str:
        """패턴 분석 쿼리 구성"""
        return f"""
Analyze {symbol} stock chart patterns for short-term trading (15-minute intervals).

**COMPREHENSIVE PATTERN ANALYSIS**

Identify and evaluate the following:

**1. Primary Pattern Recognition**
- Pattern Type: {', '.join(self.pattern_types[:10])}
- Pattern Completion: How well-formed is the pattern? (0-10 scale)
- Pattern Reliability: Historical success rate for this pattern type

**2. Technical Setup Analysis**
- Current Price Position: Above/below key moving averages
- Volume Analysis: Is volume supporting the pattern formation?
- Momentum Indicators: RSI, MACD, Stochastic alignment
- Support/Resistance: Key levels for breakout/breakdown

**3. Breakout Assessment**
- Breakout Probability: Likelihood in next 15-30 minutes (0-100%)
- Direction Bias: Bullish/Bearish/Neutral with confidence
- Entry Timing: Immediate/Wait_for_breakout/Avoid
- Target Levels: Expected move distance

**4. Risk Management**
- Pattern Failure Probability (0-100%)
- Stop-loss recommendations
- Risk-reward ratio
- Maximum position size suggestion

**5. Multi-Timeframe Context**
- 1-minute: Micro patterns and noise
- 5-minute: Short-term momentum
- 15-minute: Primary pattern timeframe
- 1-hour: Broader trend context

**6. Additional Technical Signals**
- Candlestick patterns at key levels
- Gap analysis (opening gaps, filling patterns)
- Options flow or unusual activity indicators
- Sector/market correlation impact

Consider recent 2-3 day price action, current market sentiment, and {symbol}-specific news flow.

**REPLY FORMAT:**
PATTERN: [pattern_name from list above]
STRENGTH: [0-10]
BREAKOUT_PROB: [0-100]%
DIRECTION: [Bullish/Bearish/Neutral]
CONFIDENCE: [0-100]%
TIMING: [Immediate/Wait/Avoid]
SUPPORT: $[price]
RESISTANCE: $[price]
VOLUME_CONFIRM: [Strong/Moderate/Weak/None]
RISK: [0-100]%
TARGET_MOVE: [percentage expected]
TIMEFRAME_1M: [brief comment]
TIMEFRAME_5M: [brief comment]  
TIMEFRAME_1H: [brief comment]
ADDITIONAL_SIGNALS: [comma-separated list]
REASONING: [detailed explanation of pattern setup and expectations]
"""

    async def _call_gemini_25_flash(self, query: str, thinking_budget: int = 5000) -> dict:
        """Gemini 2.5 Flash API 호출"""
        try:
            gemini_config = self.api_manager.get_gemini_config()
            api_key = gemini_config['api_key']
            base_url = gemini_config['base_url']
            
            if not api_key or api_key == 'GEM':
                logger.warning("Gemini API 키 없음, 빈 결과 반환")
                return {}
            
            # Gemini 2.5 Flash 엔드포인트
            url = f"{base_url}/models/gemini-2.0-flash-thinking-exp:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            
            payload = {
                "contents": [{
                    "parts": [{"text": query}]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "topK": 40,
                    "topP": 0.8,
                    "maxOutputTokens": 1000
                },
                "systemInstruction": {
                    "parts": [{"text": "You are an expert technical analyst specializing in short-term trading patterns. Provide precise, actionable analysis for 15-minute timeframe trading."}]
                }
            }
            
            # thinking budget 추가 (있는 경우)
            if thinking_budget > 0:
                payload["generationConfig"]["thinkingBudget"] = thinking_budget
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=20) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # content 추출
                        candidates = result.get('candidates', [])
                        if candidates:
                            content = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                            return {'content': content, 'status': 'success'}
                        else:
                            logger.warning("Gemini 응답에 candidates 없음")
                            return {}
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API 오류: {response.status} - {error_text}")
                        return {}
                        
        except asyncio.TimeoutError:
            logger.error("Gemini API 타임아웃")
            return {}
        except Exception as e:
            logger.error(f"Gemini API 호출 실패: {e}")
            return {}

    def _parse_pattern_result(self, content: str) -> dict:
        """패턴 분석 결과 파싱"""
        try:
            result = {}
            
            # 정규식 패턴들
            patterns = {
                'pattern': r'PATTERN:\s*([^\n]+)',
                'strength': r'STRENGTH:\s*(\d+(?:\.\d+)?)',
                'breakout_prob': r'BREAKOUT_PROB:\s*(\d+(?:\.\d+)?)%?',
                'direction': r'DIRECTION:\s*([^\n]+)',
                'confidence': r'CONFIDENCE:\s*(\d+(?:\.\d+)?)%?',
                'timing': r'TIMING:\s*([^\n]+)',
                'support': r'SUPPORT:\s*\$?(\d+(?:\.\d+)?)',
                'resistance': r'RESISTANCE:\s*\$?(\d+(?:\.\d+)?)',
                'volume_confirm': r'VOLUME_CONFIRM:\s*([^\n]+)',
                'risk': r'RISK:\s*(\d+(?:\.\d+)?)%?',
                'target_move': r'TARGET_MOVE:\s*(\d+(?:\.\d+)?)%?',
                'reasoning': r'REASONING:\s*([^\n]+(?:\n(?!PATTERN:|STRENGTH:|BREAKOUT_PROB:|DIRECTION:|CONFIDENCE:|TIMING:|SUPPORT:|RESISTANCE:|VOLUME_CONFIRM:|RISK:|TARGET_MOVE:|TIMEFRAME_|ADDITIONAL_SIGNALS:)[^\n]+)*)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    
                    # 숫자 필드 변환
                    if key in ['strength', 'breakout_prob', 'confidence', 'risk', 'target_move']:
                        try:
                            result[key] = float(value)
                        except ValueError:
                            result[key] = 50  # 기본값
                    elif key in ['support', 'resistance']:
                        try:
                            result[key] = float(value)
                        except ValueError:
                            result[key] = 0
                    else:
                        result[key] = value
                else:
                    # 기본값 설정
                    defaults = {
                        'pattern': 'No_Clear_Pattern',
                        'strength': 5,
                        'breakout_prob': 50,
                        'direction': 'Neutral',
                        'confidence': 50,
                        'timing': 'Wait',
                        'support': 0,
                        'resistance': 0,
                        'volume_confirm': 'Weak',
                        'risk': 50,
                        'target_move': 0,
                        'reasoning': 'No clear pattern identified'
                    }
                    result[key] = defaults.get(key, '')
            
            # 타임프레임 분석 파싱
            timeframe_analysis = {}
            for tf in ['1M', '5M', '1H']:
                tf_pattern = f'TIMEFRAME_{tf}:\\s*([^\n]+)'
                match = re.search(tf_pattern, content, re.IGNORECASE)
                if match:
                    timeframe_analysis[tf] = match.group(1).strip()
            result['timeframe_analysis'] = timeframe_analysis
            
            # 추가 시그널 파싱
            additional_signals = []
            signals_pattern = r'ADDITIONAL_SIGNALS:\s*([^\n]+)'
            match = re.search(signals_pattern, content, re.IGNORECASE)
            if match:
                signals_text = match.group(1).strip()
                additional_signals = [s.strip() for s in signals_text.split(',') if s.strip()]
            result['additional_signals'] = additional_signals
            
            return result
            
        except Exception as e:
            logger.error(f"패턴 결과 파싱 실패: {e}")
            return self._get_default_pattern_data()

    def _calculate_pattern_score(self, pattern_data: dict) -> float:
        """패턴 점수 계산"""
        try:
            # 각 요소별 점수 (0-10 스케일)
            strength_score = min(pattern_data.get('strength', 5), 10) / 10
            breakout_score = min(pattern_data.get('breakout_prob', 50), 100) / 100
            confidence_score = min(pattern_data.get('confidence', 50), 100) / 100
            
            # 볼륨 확인 점수
            volume_confirm = pattern_data.get('volume_confirm', 'Weak').lower()
            volume_score = {
                'strong': 1.0,
                'moderate': 0.7,
                'weak': 0.4,
                'none': 0.1
            }.get(volume_confirm, 0.4)
            
            # 리스크 점수 (낮을수록 좋음)
            risk_level = pattern_data.get('risk', 50)
            risk_score = max(0, (100 - risk_level) / 100)
            
            # 패턴 타입 가중치
            pattern_type = pattern_data.get('pattern', 'default')
            pattern_weight = self.pattern_weights.get(pattern_type, self.pattern_weights['default'])
            
            # 가중 평균 계산
            weighted_score = (
                strength_score * self.score_weights['pattern_strength'] +
                breakout_score * self.score_weights['breakout_probability'] +
                confidence_score * self.score_weights['confidence'] +
                volume_score * self.score_weights['volume_confirmation'] +
                risk_score * self.score_weights['risk_assessment']
            )
            
            # 패턴 가중치 적용
            final_score = weighted_score * pattern_weight * 10  # 0-10 스케일
            
            return round(final_score, 2)
            
        except Exception as e:
            logger.error(f"패턴 점수 계산 실패: {e}")
            return 5.0

    def _get_neutral_pattern_result(self, symbol: str) -> dict:
        """중립 패턴 결과"""
        return {
            'pattern_detected': 'No_Clear_Pattern',
            'pattern_strength': 5,
            'breakout_probability': 50,
            'direction_bias': 'Neutral',
            'confidence': 50,
            'entry_timing': 'Wait',
            'support_level': 0,
            'resistance_level': 0,
            'volume_confirmation': 'Weak',
            'risk_level': 50,
            'pattern_score': 5.0,
            'reasoning': 'No clear pattern identified in current timeframe',
            'additional_signals': [],
            'timeframe_analysis': {},
            'data_source': 'Pattern Recognition Fallback',
            'thinking_budget_used': 0,
            'cache_key': '',
            'timestamp': datetime.now().isoformat()
        }

    def _get_error_pattern_result(self, symbol: str, error_msg: str) -> dict:
        """에러 패턴 결과"""
        result = self._get_neutral_pattern_result(symbol)
        result.update({
            'pattern_detected': 'Analysis_Error',
            'data_source': 'Pattern Recognition Error',
            'reasoning': f'Analysis failed: {error_msg}',
            'error': error_msg
        })
        return result

    def _get_default_pattern_data(self) -> dict:
        """기본 패턴 데이터"""
        return {
            'pattern': 'No_Clear_Pattern',
            'strength': 5,
            'breakout_prob': 50,
            'direction': 'Neutral',
            'confidence': 50,
            'timing': 'Wait',
            'support': 0,
            'resistance': 0,
            'volume_confirm': 'Weak',
            'risk': 50,
            'target_move': 0,
            'reasoning': 'Default pattern data',
            'timeframe_analysis': {},
            'additional_signals': []
        }

    async def analyze_multiple_patterns(self, symbols: List[str], thinking_budget: int = 3000) -> dict:
        """여러 종목 패턴 동시 분석"""
        try:
            logger.info(f"🔍 다중 패턴 분석 시작: {len(symbols)}개 종목")
            
            # 동시 분석 (최대 5개까지)
            symbols = symbols[:5]
            tasks = [
                self.analyze_patterns_15min(symbol, thinking_budget)
                for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 정리
            analysis_results = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    analysis_results[symbol] = self._get_error_pattern_result(symbol, str(result))
                else:
                    analysis_results[symbol] = result
            
            # 요약 통계
            summary = self._generate_pattern_summary(analysis_results)
            
            return {
                'symbols': analysis_results,
                'summary': summary,
                'total_analyzed': len(symbols),
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"다중 패턴 분석 실패: {e}")
            return {
                'symbols': {},
                'summary': {},
                'total_analyzed': 0,
                'error': str(e)
            }

    def _generate_pattern_summary(self, results: dict) -> dict:
        """패턴 분석 요약 생성"""
        try:
            if not results:
                return {}
            
            # 통계 계산
            pattern_counts = {}
            total_score = 0
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            for symbol, data in results.items():
                pattern = data.get('pattern_detected', 'Unknown')
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                total_score += data.get('pattern_score', 5)
                
                direction = data.get('direction_bias', 'Neutral').lower()
                if 'bullish' in direction:
                    bullish_count += 1
                elif 'bearish' in direction:
                    bearish_count += 1
                else:
                    neutral_count += 1
            
            avg_score = total_score / len(results) if results else 5.0
            
            # 가장 빈번한 패턴
            most_common_pattern = max(pattern_counts.items(), key=lambda x: x[1]) if pattern_counts else ('None', 0)
            
            return {
                'average_pattern_score': round(avg_score, 2),
                'direction_distribution': {
                    'bullish': bullish_count,
                    'bearish': bearish_count,
                    'neutral': neutral_count
                },
                'pattern_distribution': pattern_counts,
                'most_common_pattern': most_common_pattern[0],
                'total_patterns_found': len([p for p in pattern_counts.keys() if p != 'No_Clear_Pattern'])
            }
            
        except Exception as e:
            logger.error(f"패턴 요약 생성 실패: {e}")
            return {}

    def clear_cache(self):
        """캐시 정리"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, (timestamp, _) in self.pattern_cache.items():
            if current_time - timestamp > self.cache_duration:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.pattern_cache[key]
        
        logger.info(f"🧹 패턴 인식 캐시 정리: {len(expired_keys)}개 항목 삭제")

    def get_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        return {
            'total_entries': len(self.pattern_cache),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60,
            'supported_patterns': len(self.pattern_types),
            'pattern_weights': self.pattern_weights,
            'score_weights': self.score_weights
        }

    def get_supported_patterns(self) -> List[str]:
        """지원하는 패턴 목록 반환"""
        return self.pattern_types.copy()
