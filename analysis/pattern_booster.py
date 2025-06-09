"""
Gemini 2.5 Flash 패턴 인식 부스터
"""
import re
import aiohttp
from datetime import datetime

class GeminiPatternRecognitionBooster:
    """🔍 Gemini 2.5 Flash 패턴 인식 부스터 (15분 자동매매 최적화)"""

    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.pattern_cache = {}  # 5분간 캐싱
        self.pattern_types = [
            'Head_and_Shoulders', 'Cup_and_Handle', 'Double_Top', 'Double_Bottom',
            'Triangle', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Hammer',
            'Doji', 'Flag', 'Pennant', 'Wedge'
        ]

    async def analyze_patterns_15min(self, symbol: str, thinking_budget: int = 5000) -> dict:
        """🎯 15분 자동매매용 패턴 분석 (Gemini 2.5 Flash)"""
        try:
            print(f"🔍 {symbol} 패턴 인식 분석 시작 (thinking_budget: {thinking_budget})...")

            # 캐시 확인 (5분)
            cache_key = f"{symbol}_pattern_{datetime.now().strftime('%H:%M')[:-1]}0"
            if cache_key in self.pattern_cache:
                print(f"📋 {symbol} 패턴 캐시 사용")
                return self.pattern_cache[cache_key]

            query = f"""
            Analyze {symbol} stock chart patterns for short-term trading (15-minute intervals).

            Identify and evaluate:
            1. **Current Pattern Type**: Head & Shoulders, Cup & Handle, Double Top/Bottom, Triangle, etc.
            2. **Pattern Strength**: How well-formed is the pattern? (0-10 scale)
            3. **Breakout Probability**: Likelihood of breakout in next 15-30 minutes (0-100%)
            4. **Direction Bias**: Bullish/Bearish/Neutral with confidence level
            5. **Entry Timing**: Immediate/Wait_for_breakout/Avoid
            6. **Support/Resistance Levels**: Key price levels to watch
            7. **Volume Confirmation**: Is volume supporting the pattern?
            8. **Risk Assessment**: Pattern failure probability (0-100%)

            Consider recent 2-3 day price action and current market momentum.

            Reply format:
            PATTERN: [pattern_name]
            STRENGTH: [0-10]
            BREAKOUT_PROB: [0-100]%
            DIRECTION: [Bullish/Bearish/Neutral]
            CONFIDENCE: [0-100]%
            TIMING: [Immediate/Wait/Avoid]
            SUPPORT: $[price]
            RESISTANCE: $[price]
            VOLUME_CONFIRM: [Yes/No/Weak]
            RISK: [0-100]%
            REASONING: [brief explanation]
            """

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
                    'data_source': 'Gemini 2.5 Flash Pattern Recognition',
                    'thinking_budget_used': thinking_budget,
                    'cache_key': cache_key
                }

                # 캐시 저장 (5분)
                self.pattern_cache[cache_key] = final_result

                print(f"✅ {symbol} 패턴 분석 완료: {final_result['pattern_detected']} (점수: {pattern_score:.1f})")
                return final_result
            else:
                print(f"⚠️ {symbol} 패턴 분석 실패, 기본값 사용")
                return self._get_neutral_pattern_result(symbol)

        except Exception as e:
            print(f"❌ {symbol} 패턴 인식 실패: {str(e)}")
            return self._get_neutral_pattern_result(symbol)
    
    # 나머지 메서드들...
