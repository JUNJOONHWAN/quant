"""
Gemini 갭필터 부스터
"""
import re
import numpy as np
import aiohttp

class GeminiGapFilterBooster:
    """Gemini Flash로 VTNF 갭 분석 (AI 독립 판단 vs 실제 VTNF 델타)"""

    def __init__(self, api_manager):
        self.api_manager = api_manager
        
    async def calculate_vtnf_gap_boost(self, symbol: str, vtnf_scores: dict) -> dict:
        """✅ AI 독립 판단 vs VTNF 실제값 델타 계산"""
        try:
            print(f"🎯 {symbol} AI 델타 갭필터 분석 시작...")
            
            # 1. ✅ AI의 독립적인 종합 판단 (섹터 무관)
            ai_comprehensive_query = f"""
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

            # 2. ✅ AI의 개별 VTNF 예측 (실제값과 비교용)
            ai_vtnf_predictions = {
                'V': f"""Analyze {symbol}'s FUNDAMENTAL VALUE ignoring sector norms.
Consider: unique financial metrics, hidden assets, cash generation ability, 
non-traditional value drivers specific to this company.
PREDICTED V SCORE (0-10):""",
                
                'T': f"""Analyze {symbol}'s TECHNICAL DYNAMICS beyond sector patterns.
Consider: unique price behavior, volume anomalies, algorithmic trading patterns,
company-specific technical setups that differ from sector.
PREDICTED T SCORE (0-10):""",
                
                'N': f"""Analyze {symbol}'s NEWS & SENTIMENT with fresh perspective.
Consider: company-specific catalysts, management changes, product launches,
partnerships that transcend typical sector news flow.
PREDICTED N SCORE (0-10):""",
                
                'F': f"""Analyze {symbol}'s INSTITUTIONAL FLOWS independently.
Consider: smart money movements specific to this stock, unusual options activity,
insider behavior that differs from sector trends.
PREDICTED F SCORE (0-10):"""
            }
            
            # 3. 병렬 API 호출
            comprehensive_task = self._get_ai_score(ai_comprehensive_query)
            prediction_tasks = {k: self._get_ai_score(v) for k, v in ai_vtnf_predictions.items()}
            
            # 결과 수집
            ai_comprehensive = await comprehensive_task
            ai_predictions = {}
            for k, task in prediction_tasks.items():
                ai_predictions[k.lower()] = await task
                
            # 4. ✅ 실제 VTNF 값 정규화 (소문자로 통일)
            actual_vtnf = {k.lower(): v for k, v in vtnf_scores.items()}
            
            # 5. ✅ 델타 계산 (AI 예측 - 실제값)
            deltas = {}
            total_delta = 0
            for factor in ['v', 't', 'n', 'f']:
                predicted = ai_predictions.get(factor, 6.0)
                actual = actual_vtnf.get(factor, 6.0)
                delta = predicted - actual
                deltas[factor] = delta
                total_delta += delta
                
            avg_delta = total_delta / 4
            
            # 6. ✅ AI 종합 판단 vs VTNF 평균의 갭
            actual_vtnf_avg = sum(actual_vtnf.values()) / len(actual_vtnf)
            comprehensive_gap = ai_comprehensive - actual_vtnf_avg
            
            # 7. ✅ 통합 갭 스코어 계산
            # AI가 개별 요소들을 다르게 보는 정도(델타) + 종합 판단의 차이
            integrated_gap = (comprehensive_gap * 0.6) + (avg_delta * 0.4)
            
            print(f"🔍 {symbol} AI 델타 분석:")
            print(f"   AI 종합 판단: {ai_comprehensive}")
            print(f"   실제 VTNF 평균: {actual_vtnf_avg:.2f}")
            print(f"   종합 갭: {comprehensive_gap:+.2f}")
            print(f"   AI 예측 VTNF: {ai_predictions}")
            print(f"   실제 VTNF: {actual_vtnf}")
            print(f"   델타: {deltas}")
            print(f"   평균 델타: {avg_delta:+.2f}")
            print(f"   통합 갭: {integrated_gap:+.2f}")
            
            # 8. ✅ 갭 유의성 평가
            gap_significance = self._evaluate_gap_significance(
                integrated_gap, comprehensive_gap, avg_delta, deltas
            )
            
            # 9. ✅ 부스트 결정
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
                    'divergence_factors': gap_significance['divergence_factors'],
                    'method': 'ai_independent_delta'
                }
            else:
                return {
                    'gap_detected': False,
                    'gap_score': round(integrated_gap, 2),
                    'integrated_gap': round(integrated_gap, 2),
                    'comprehensive_gap': round(comprehensive_gap, 2),
                    'avg_delta': round(avg_delta, 2),
                    'boost_type': 'neutral',
                    'boost_strength': 0,
                    'position_adjustment': 0,
                    'confidence': gap_significance['confidence'],
                    'method': 'ai_independent_delta'
                }
                
        except Exception as e:
            print(f"❌ {symbol} AI 델타 갭필터 실패: {e}")
            return {
                'gap_detected': False,
                'gap_score': 0,
                'integrated_gap': 0,
                'boost_type': 'error',
                'error': str(e)
            }
    
    # 나머지 메서드들... (_evaluate_gap_significance, _calculate_position_adjustment 등)
