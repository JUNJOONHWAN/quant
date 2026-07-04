#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_trend_validator.py - Perplexity API 기반 범용 트렌드 검증 시스템
"""

import os
import json
import logging
import aiohttp
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """AI 검증 결과"""
    parameter: str
    ml_change: str  
    validation_score: int  # 0-100
    status: str  # 적절함/주의필요/재검토필요
    confidence: int  # AI 판단 신뢰도
    key_factors: List[str]
    risks: List[str]
    recommended_adjustment: float  # 1.0=그대로, 0.8=20%완화, 0.5=50%완화
    reasoning: str

@dataclass
class TrendValidationReport:
    """전체 검증 리포트"""
    validation_timestamp: str
    overall_confidence: float
    parameter_validations: List[ValidationResult]
    market_context_summary: str
    key_trends: List[str]
    risk_alerts: List[str]

class PerplexityAPIClient:
    """Perplexity API 클라이언트"""
    
    def __init__(self):
        self.api_key = os.getenv("PPPP_API_KEY", "")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = os.getenv("PPPP_MODEL", "sonar-pro")
        
        if not self.api_key:
            logger.warning("PPPP_API_KEY가 설정되지 않았습니다")
            
    async def query(self, prompt: str, max_tokens: int = 2000) -> str:
        """Perplexity API 쿼리"""
        try:
            if not self.api_key:
                logger.error("Perplexity API 키가 없습니다")
                return "API 키 없음"
                
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "당신은 딥테크 투자 전문가입니다. 객관적이고 데이터 기반으로 분석하며, JSON 형태로 구조화된 답변을 제공합니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Perplexity API 오류: {response.status} - {error_text}")
                        return f"API 오류: {response.status}"
                        
        except asyncio.TimeoutError:
            logger.error("Perplexity API 타임아웃")
            return "API 타임아웃"
        except Exception as e:
            logger.error(f"Perplexity API 호출 실패: {e}")
            return f"API 오류: {str(e)}"

class UniversalTrendValidator:
    """범용 트렌드 검증기"""
    
    def __init__(self):
        self.api_client = PerplexityAPIClient()
        
    def format_parameter_changes(self, parameter_changes: Dict) -> str:
        """파라미터 변경사항 포맷팅"""
        try:
            formatted_changes = []
            
            # 점수 가중치 변경사항
            if 'scoring_parameters' in parameter_changes:
                weights = parameter_changes['scoring_parameters'].get('weights', {})
                for weight_name, value in weights.items():
                    formatted_changes.append(f"- {weight_name}: {value:.3f}")
                    
                # 기술 가중치 변경사항
                tech_mult = parameter_changes['scoring_parameters'].get('tech_multipliers', {})
                for tech, multiplier in tech_mult.items():
                    formatted_changes.append(f"- {tech}_multiplier: {multiplier:.2f}")
                    
            # 필터링 파라미터 변경사항
            if 'filtering_parameters' in parameter_changes:
                filters = parameter_changes['filtering_parameters']
                key_filters = ['min_crash_percent', 'min_recovery_percent', 'min_institutional_holders', 'min_total_score']
                for filter_name in key_filters:
                    if filter_name in filters:
                        formatted_changes.append(f"- {filter_name}: {filters[filter_name]}")
                        
            return "\n".join(formatted_changes)
            
        except Exception as e:
            logger.error(f"파라미터 포맷팅 실패: {e}")
            return "포맷팅 오류"
            
    def get_universal_validation_prompt(self, parameter_changes: Dict) -> str:
        """시점에 무관한 범용 검증 프롬프트"""
        
        changes_text = self.format_parameter_changes(parameter_changes)
        
        return f"""
딥테크 투자 파라미터 변경에 대한 트렌드 검증 분석을 요청합니다.

제안된 파라미터 변경사항:
{changes_text}

다음 5가지 관점에서 각 주요 변경사항을 검증해주세요:

1. **기술 성숙도 사이클 관점**
   - 해당 기술이 현재 어느 단계인가? (연구/개발/상용화/성숙)
   - 제안된 가중치가 성숙도와 일치하는가?

2. **산업 채택 속도 관점**  
   - 기업들의 실제 도입/투자 속도는?
   - 제안된 변경이 현실적 채택 속도와 부합하는가?
   
3. **규제 환경 관점**
   - 현재 규제 트렌드 (강화/완화/중립)
   - 규제 변화가 해당 섹터에 미치는 영향
   
4. **자본 흐름 관점**
   - VC/기관투자 트렌드 (집중/분산/회피)
   - 제안된 가중치가 자본 흐름과 일치하는가?
   
5. **글로벌 경쟁 관점**
   - 지정학적 요소 (미중 기술패권, 공급망)
   - 국가별 기술 우위 변화

다음 JSON 형식으로 주요 변경사항 3-5개에 대해 답변해주세요:

```json
[
  {{
    "parameter": "ai_ml_weight",
    "change": "+20%",
    "validation_score": 85,
    "status": "적절함",
    "confidence": 90,
    "key_factors": ["기업 AI 도입 가속", "규제 명확화 진행"],
    "risks": ["과도한 밸류에이션", "경쟁 심화"],
    "recommended_adjustment": 1.0,
    "reasoning": "AI/ML 섹터는 현재 상용화 단계 진입으로 가중치 증가가 적절함. 엔터프라이즈 도입 가속화와 규제 환경 명확화가 긍정적 요인."
  }}
]
```

validation_score: 0-100 (트렌드 부합도)
status: "적절함"/"주의필요"/"재검토필요"
recommended_adjustment: 1.0(그대로)/0.8(20%완화)/0.5(50%완화)

객관적이고 데이터 기반으로 판단해주세요.
"""
        
    def get_historical_validation_prompt(self, enriched_ml_results: Dict) -> str:
        """과거 데이터를 포함한 강화된 검증 프롬프트"""
        
        # 현재 ML 제안 파라미터
        changes_text = self.format_parameter_changes(enriched_ml_results.get("optimized_parameters", {}))
        
        # 과거 데이터 포맷팅
        historical_context = enriched_ml_results.get("historical_context", {})
        
        # ML 파라미터 변동 이력
        ml_history_text = "없음"
        if historical_context.get("ml_parameter_history"):
            ml_history_lines = []
            for hist in historical_context["ml_parameter_history"]:
                ml_history_lines.append(f"- {hist['date']}: 신뢰도 {hist['confidence']}, 조정 {hist['adjustments_made']}건, 피트니스 {hist['fitness_score']:.1f}")
            ml_history_text = "\n".join(ml_history_lines)
        
        # 카테고리별 실제 성과
        performance_text = "없음"
        if historical_context.get("category_performance"):
            perf_lines = []
            for cat, perf in historical_context["category_performance"].items():
                perf_lines.append(f"- {cat}: 평균수익률 {perf['avg_return']:+.1f}%, 승률 {perf['win_rate']:.0f}%, {perf['total_picks']}개 종목")
            performance_text = "\n".join(perf_lines)
        
        return f"""
딥테크 투자 파라미터 변경에 대한 **실제 데이터 기반** 트렌드 검증 분석을 요청합니다.

## 현재 ML이 제안한 파라미터 변경:
{changes_text}

## 과거 ML 최적화 이력:
{ml_history_text}

## 카테고리별 실제 Sweet Spot 성과 데이터:
{performance_text}

**과거 데이터를 바탕으로 다음 관점에서 각 주요 변경사항을 검증해주세요:**

1. **과거 성과 기반 타당성**
   - 해당 카테고리의 실제 수익률/승률이 변경을 뒷받침하는가?
   - 과거 유사한 조정이 있었다면 그 결과는?

2. **ML 최적화 트렌드 분석**
   - 최근 ML 신뢰도/피트니스 점수 변화 추이
   - 과거 조정이 실제로 성과 개선에 기여했는가?

3. **카테고리별 성과 차별화**
   - 실제로 성과가 좋은 카테고리 vs 부진한 카테고리 구분
   - 제안된 가중치가 실제 성과와 일치하는가?

4. **데이터 기반 위험 요소**
   - 승률이 낮거나 수익률이 부진한 카테고리의 과도한 가중치 증가
   - 샘플 수가 적은 카테고리에 대한 성급한 판단

5. **Sweet Spot 특화 고려사항**
   - 급락 후 회복 패턴에서의 카테고리별 특성
   - 현재 시장 환경에서의 Sweet Spot 전략 유효성

**다음 JSON 형식으로 주요 변경사항 3-5개에 대해 답변해주세요:**

```json
[
  {{
    "parameter": "ai_computing_weight",
    "change": "+15%",
    "validation_score": 75,
    "status": "적절함",
    "confidence": 85,
    "key_factors": ["실제 수익률 +3.0% 양호", "43개 종목 충분한 샘플", "승률 48.6% 평균적"],
    "risks": ["시장 과열시 조정 위험", "경쟁 심화"],
    "recommended_adjustment": 0.9,
    "reasoning": "ai_computing 카테고리는 실제 평균 수익률 +3.0%로 다른 카테고리 대비 우수한 성과를 보임. 43개 종목으로 충분한 샘플 확보. 다만 승률 48.6%는 평균 수준으로 +15% 증가는 과도할 수 있어 90% 적용 권장."
  }}
]
```

validation_score: 0-100 (실제 데이터 기반 타당성 점수)
recommended_adjustment: 1.0(그대로)/0.9(10%완화)/0.8(20%완화)/0.5(50%완화)

**실제 성과 데이터를 최우선으로 고려하여 객관적으로 평가해주세요.**
"""

    async def validate_ml_results_with_history(self, enriched_ml_results: Dict) -> TrendValidationReport:
        """과거 데이터를 포함한 강화된 ML 결과 검증"""
        try:
            logger.info("🧠 AI 트렌드 검증 시작 (과거 데이터 기반)")
            
            # 과거 데이터 기반 강화된 프롬프트 생성
            prompt = self.get_historical_validation_prompt(enriched_ml_results)
            ai_response = await self.api_client.query(prompt)
            
            if "API 오류" in ai_response or "API 키 없음" in ai_response or "API 타임아웃" in ai_response:
                logger.warning(f"AI 검증 실패: {ai_response}")
                return self.create_fallback_validation(enriched_ml_results)
                
            # AI 응답 파싱
            validations = self.parse_ai_response(ai_response)
            
            # 전체 신뢰도 계산
            overall_confidence = self.calculate_overall_confidence(validations)
            
            report = TrendValidationReport(
                validation_timestamp=datetime.now().isoformat(),
                overall_confidence=overall_confidence,
                parameter_validations=validations,
                market_context_summary=f"과거 데이터 기반 평가 - {len(enriched_ml_results.get('historical_context', {}).get('category_performance', {}))}개 카테고리 성과 분석",
                key_trends=self.extract_key_trends(validations),
                risk_alerts=self.extract_risk_alerts(validations)
            )
            
            logger.info(f"✅ AI 검증 완료: 신뢰도 {overall_confidence:.1%}, {len(validations)}개 파라미터 검증")
            return report
            
        except Exception as e:
            logger.error(f"AI 검증 실패: {e}")
            return self.create_fallback_validation(enriched_ml_results)

    async def validate_ml_results(self, ml_results: Dict) -> TrendValidationReport:
        """ML 결과를 AI로 검증 (호환성 유지)"""
        try:
            logger.info("🧠 AI 트렌드 검증 시작")
            
            # Perplexity API 호출
            prompt = self.get_universal_validation_prompt(ml_results.get("optimized_parameters", {}))
            ai_response = await self.api_client.query(prompt)
            
            if "API 오류" in ai_response or "API 키 없음" in ai_response or "API 타임아웃" in ai_response:
                logger.warning(f"AI 검증 실패: {ai_response}")
                return self.create_fallback_validation(ml_results)
                
            # AI 응답 파싱
            validations = self.parse_ai_response(ai_response)
            
            # 전체 신뢰도 계산
            overall_confidence = self.calculate_overall_confidence(validations)
            
            report = TrendValidationReport(
                validation_timestamp=datetime.now().isoformat(),
                overall_confidence=overall_confidence,
                parameter_validations=validations,
                market_context_summary=self.extract_market_context(ai_response),
                key_trends=self.extract_key_trends(validations),
                risk_alerts=self.extract_risk_alerts(validations)
            )
            
            logger.info(f"AI 검증 완료: 전체 신뢰도 {overall_confidence:.1%}")
            return report
            
        except Exception as e:
            logger.error(f"AI 검증 실패: {e}")
            return self.create_fallback_validation(ml_results)
            
    def parse_ai_response(self, ai_response: str) -> List[ValidationResult]:
        """AI 응답 파싱"""
        try:
            # JSON 추출
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', ai_response, re.DOTALL)
            if not json_match:
                # JSON 블록이 없으면 전체에서 JSON 찾기
                json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                
            if json_match:
                json_text = json_match.group(1) if '```json' in ai_response else json_match.group(0)
                validations_data = json.loads(json_text)
                
                validations = []
                for item in validations_data:
                    validation = ValidationResult(
                        parameter=item.get("parameter", "unknown"),
                        ml_change=item.get("change", "0%"),
                        validation_score=int(item.get("validation_score", 50)),
                        status=item.get("status", "주의필요"),
                        confidence=int(item.get("confidence", 50)),
                        key_factors=item.get("key_factors", []),
                        risks=item.get("risks", []),
                        recommended_adjustment=float(item.get("recommended_adjustment", 1.0)),
                        reasoning=item.get("reasoning", "분석 정보 부족")
                    )
                    validations.append(validation)
                    
                return validations
                
            else:
                logger.warning("AI 응답에서 JSON을 찾을 수 없음")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            return []
        except Exception as e:
            logger.error(f"AI 응답 파싱 실패: {e}")
            return []
            
    def create_fallback_validation(self, ml_results: Dict) -> TrendValidationReport:
        """API 실패시 폴백 검증"""
        logger.info("폴백 검증 모드 사용")
        
        fallback_validations = []
        
        # 주요 파라미터에 대한 보수적 검증
        optimized_params = ml_results.get("optimized_parameters", {})
        if "scoring_parameters" in optimized_params:
            weights = optimized_params["scoring_parameters"].get("weights", {})
            tech_mult = optimized_params["scoring_parameters"].get("tech_multipliers", {})
            
            # 가중치 검증
            for weight_name, value in weights.items():
                validation = ValidationResult(
                    parameter=weight_name,
                    ml_change=f"{value:.3f}",
                    validation_score=70,  # 중간 점수
                    status="주의필요",
                    confidence=60,
                    key_factors=["AI 검증 불가"],
                    risks=["검증되지 않은 변경"],
                    recommended_adjustment=0.8,  # 20% 완화
                    reasoning="AI 검증 실패로 보수적 적용"
                )
                fallback_validations.append(validation)
                
        return TrendValidationReport(
            validation_timestamp=datetime.now().isoformat(),
            overall_confidence=0.6,  # 낮은 신뢰도
            parameter_validations=fallback_validations,
            market_context_summary="AI 검증 서비스 이용 불가",
            key_trends=["검증 데이터 부족"],
            risk_alerts=["AI 검증 없이 파라미터 변경"]
        )
        
    def calculate_overall_confidence(self, validations: List[ValidationResult]) -> float:
        """전체 신뢰도 계산"""
        if not validations:
            return 0.5
            
        total_confidence = sum(v.confidence * (v.validation_score / 100) for v in validations)
        return min(total_confidence / len(validations) / 100, 1.0)
        
    def extract_market_context(self, ai_response: str) -> str:
        """시장 맥락 요약 추출"""
        try:
            # 첫 번째 문단이나 요약 부분 추출
            lines = ai_response.split('\n')
            context_lines = []
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['시장', '환경', '트렌드', '상황']):
                    context_lines.append(line.strip())
                    if len(context_lines) >= 3:
                        break
                        
            return " ".join(context_lines) if context_lines else "시장 맥락 분석 정보 없음"
            
        except:
            return "시장 맥락 추출 실패"
            
    def extract_key_trends(self, validations: List[ValidationResult]) -> List[str]:
        """주요 트렌드 추출"""
        trends = []
        for validation in validations:
            trends.extend(validation.key_factors)
        return list(set(trends))[:5]  # 중복 제거 후 상위 5개
        
    def extract_risk_alerts(self, validations: List[ValidationResult]) -> List[str]:
        """리스크 알림 추출"""
        risks = []
        for validation in validations:
            if validation.validation_score < 60:  # 낮은 점수는 리스크
                risks.extend(validation.risks)
        return list(set(risks))[:5]  # 중복 제거 후 상위 5개

class TrendValidationManager:
    """트렌드 검증 관리자"""
    
    def __init__(self):
        self.validator = UniversalTrendValidator()
        self.validation_history_file = "ai_validation_history.json"
        self.ml_params_file = "ml_parameters.json"
        self.sweet_spot_db_file = "sweet_spot_database.json"
        
    async def validate_parameters(self, ml_results: Dict) -> TrendValidationReport:
        """파라미터 검증 실행 (과거 데이터 기반 강화)"""
        try:
            # 1. 과거 데이터 수집
            historical_data = self.collect_historical_data()
            
            # 2. ML 결과와 과거 데이터 결합
            enriched_ml_results = {**ml_results, "historical_context": historical_data}
            
            # 3. 강화된 검증 실행
            report = await self.validator.validate_ml_results_with_history(enriched_ml_results)
            self.save_validation_history(report)
            return report
            
        except Exception as e:
            logger.error(f"검증 관리 실패: {e}")
            return self.validator.create_fallback_validation(ml_results)
    
    def collect_historical_data(self) -> Dict:
        """과거 ML 파라미터 변동 이력과 성과 데이터 수집"""
        try:
            historical_data = {
                "ml_parameter_history": [],
                "category_performance": {},
                "market_context": {}
            }
            
            # 1. ML 파라미터 변동 이력 수집
            try:
                with open(self.ml_params_file, 'r', encoding='utf-8') as f:
                    ml_data = json.load(f)
                    
                opt_history = ml_data.get('optimization_history', [])
                # 최근 5개 기록만 사용
                recent_history = sorted(opt_history, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
                
                for hist in recent_history:
                    historical_data["ml_parameter_history"].append({
                        "date": hist.get('timestamp', ''),
                        "confidence": hist.get('confidence', 0),
                        "adjustments_made": hist.get('adjustments_made', 0),
                        "fitness_score": hist.get('fitness_score', 0),
                        "expected_improvement": hist.get('expected_improvement', 0)
                    })
                    
            except Exception as e:
                logger.warning(f"ML 이력 수집 실패: {e}")
            
            # 2. Sweet Spot 카테고리별 성과 수집
            try:
                with open(self.sweet_spot_db_file, 'r', encoding='utf-8') as f:
                    sweet_spot_data = json.load(f)
                    
                picks = sweet_spot_data.get('picks', [])
                category_stats = {}
                
                for pick in picks:
                    cat = pick.get('tech_category', 'unknown')
                    return_pct = pick.get('current_return_pct')
                    
                    if cat not in category_stats:
                        category_stats[cat] = {"returns": [], "count": 0}
                    
                    category_stats[cat]["count"] += 1
                    if return_pct is not None:
                        category_stats[cat]["returns"].append(return_pct)
                
                # 카테고리별 통계 계산
                for cat, stats in category_stats.items():
                    if stats["returns"]:
                        avg_return = sum(stats["returns"]) / len(stats["returns"])
                        win_rate = sum(1 for r in stats["returns"] if r > 0) / len(stats["returns"]) * 100
                    else:
                        avg_return = 0
                        win_rate = 0
                        
                    historical_data["category_performance"][cat] = {
                        "avg_return": round(avg_return, 1),
                        "win_rate": round(win_rate, 1),
                        "total_picks": stats["count"],
                        "analyzed_picks": len(stats["returns"])
                    }
                    
            except Exception as e:
                logger.warning(f"Sweet Spot 성과 수집 실패: {e}")
            
            # 3. 기본 시장 컨텍스트 (향후 확장 가능)
            historical_data["market_context"] = {
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "total_categories": len(historical_data["category_performance"]),
                "ml_history_records": len(historical_data["ml_parameter_history"])
            }
            
            logger.info(f"📊 AI 평가용 과거 데이터 수집 완료: {len(historical_data['category_performance'])}개 카테고리, {len(historical_data['ml_parameter_history'])}개 ML 이력")
            return historical_data
            
        except Exception as e:
            logger.error(f"과거 데이터 수집 실패: {e}")
            return {"ml_parameter_history": [], "category_performance": {}, "market_context": {}}
            
    def save_validation_history(self, report: TrendValidationReport):
        """검증 히스토리 저장"""
        try:
            history = []
            if os.path.exists(self.validation_history_file):
                with open(self.validation_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    
            # 새 검증 결과 추가
            history.append({
                "timestamp": report.validation_timestamp,
                "overall_confidence": report.overall_confidence,
                "validation_count": len(report.parameter_validations),
                "key_trends": report.key_trends,
                "risk_alerts": report.risk_alerts
            })
            
            # 최근 50개만 보관
            history = history[-50:]
            
            with open(self.validation_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"검증 히스토리 저장 실패: {e}")

if __name__ == "__main__":
    async def test_validation():
        validator = TrendValidationManager()
        
        # 테스트 ML 결과
        test_ml_results = {
            "optimized_parameters": {
                "scoring_parameters": {
                    "weights": {
                        "pattern_weight": 0.28,
                        "growth_weight": 0.22,
                        "tech_weight": 0.25
                    },
                    "tech_multipliers": {
                        "ai_ml": 1.8,
                        "quantum": 1.2,
                        "biotech": 1.4
                    }
                }
            }
        }
        
        report = await validator.validate_parameters(test_ml_results)
        
        print("=== AI 트렌드 검증 결과 ===")
        print(f"전체 신뢰도: {report.overall_confidence:.1%}")
        print(f"검증 항목: {len(report.parameter_validations)}개")
        print(f"주요 트렌드: {', '.join(report.key_trends)}")
        
        for validation in report.parameter_validations:
            print(f"\n{validation.parameter}: {validation.status} ({validation.validation_score}점)")
            print(f"  조정 권장: {validation.recommended_adjustment:.1f}")
            print(f"  이유: {validation.reasoning}")
            
    asyncio.run(test_validation())