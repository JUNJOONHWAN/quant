#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 ML/AI 수렴 추적 시스템

ML과 AI의 파워 밸런스를 동적으로 추적하고 조정
예측 정확도 개선 패턴을 모니터링하여 수렴 상태 판단
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
class WeeklyPerformance:
    """주간 성과 기록"""
    week_date: str
    ml_accuracy: float
    ai_contribution: float
    combined_accuracy: float
    prediction_count: int
    mae: float
    direction_accuracy: float
    best_performers: List[str]
    worst_performers: List[str]
    major_adjustments_applied: int

@dataclass
class PowerBalance:
    """파워 밸런스"""
    ml_power: float
    ai_power: float
    balance_reason: str
    confidence_level: float
    adjustment_history: List[Dict[str, float]]

@dataclass
class ConvergenceState:
    """수렴 상태"""
    status: str  # 'improving', 'converging', 'converged', 'diverging'
    confidence: float
    weeks_in_state: int
    improvement_rate: float
    stability_score: float
    next_milestone: str

class ConvergenceTracker:
    """ML/AI 수렴 추적 시스템"""
    
    def __init__(self):
        self.performance_history_file = "ml_ai_performance_history.json"
        self.convergence_state_file = "convergence_state.json"
        
        # 파워 밸런스 초기값
        self.current_power_balance = PowerBalance(
            ml_power=0.7,
            ai_power=0.3,
            balance_reason="초기 설정",
            confidence_level=0.6,
            adjustment_history=[]
        )
        
        # 수렴 기준값
        self.convergence_thresholds = {
            'accuracy_target': 0.85,  # 85% 목표 정확도
            'stability_threshold': 0.02,  # 2% 미만 변동시 안정
            'convergence_weeks': 4,  # 4주 연속 안정시 수렴
            'improvement_threshold': 0.01,  # 1% 미만 개선시 수렴
            'min_predictions': 15  # 최소 예측 수
        }
        
        # 성과 히스토리 로드
        self.performance_history = self.load_performance_history()
        self.convergence_state = self.load_convergence_state()
    
    def load_performance_history(self) -> List[Dict]:
        """성과 히스토리 로드"""
        try:
            if os.path.exists(self.performance_history_file):
                with open(self.performance_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"성과 히스토리 로드 실패: {e}")
            return []
    
    def save_performance_history(self):
        """성과 히스토리 저장"""
        try:
            with open(self.performance_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False)
            logger.info(f"성과 히스토리 저장 완료: {len(self.performance_history)}주")
        except Exception as e:
            logger.error(f"성과 히스토리 저장 실패: {e}")
    
    def load_convergence_state(self) -> Optional[ConvergenceState]:
        """수렴 상태 로드"""
        try:
            if os.path.exists(self.convergence_state_file):
                with open(self.convergence_state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ConvergenceState(**data)
            return ConvergenceState(
                status='improving',
                confidence=0.5,
                weeks_in_state=0,
                improvement_rate=0.0,
                stability_score=0.0,
                next_milestone='첫 번째 검증 완료'
            )
        except Exception as e:
            logger.error(f"수렴 상태 로드 실패: {e}")
            return None
    
    def save_convergence_state(self):
        """수렴 상태 저장"""
        try:
            with open(self.convergence_state_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.convergence_state), f, indent=2, ensure_ascii=False)
            logger.info(f"수렴 상태 저장 완료: {self.convergence_state.status}")
        except Exception as e:
            logger.error(f"수렴 상태 저장 실패: {e}")
    
    async def update_weekly_performance(self, validation_result: Dict,
                                      ai_optimization_result: Dict) -> WeeklyPerformance:
        """주간 성과 업데이트"""
        try:
            logger.info("📊 주간 성과 업데이트 시작")
            
            # 현재 주 날짜
            current_week = datetime.now().strftime('%Y-W%U')
            
            # ML 정확도
            ml_accuracy = validation_result.get('ml_accuracy', 0)
            
            # AI 기여도 (AI 조정 후 예상 개선도)
            ai_contribution = ai_optimization_result.get('expected_improvement', 0) / 100
            
            # 결합 정확도 (ML + AI 조정 효과)
            combined_accuracy = min(0.95, ml_accuracy + ai_contribution)
            
            # 성과 기록 생성
            weekly_performance = WeeklyPerformance(
                week_date=current_week,
                ml_accuracy=ml_accuracy,
                ai_contribution=ai_contribution,
                combined_accuracy=combined_accuracy,
                prediction_count=validation_result.get('total_predictions', 0),
                mae=validation_result.get('mae', 0),
                direction_accuracy=validation_result.get('direction_accuracy', 0),
                best_performers=[pred['symbol'] for pred in validation_result.get('best_predictions', [])[:3]],
                worst_performers=[pred['symbol'] for pred in validation_result.get('worst_predictions', [])[:3]],
                major_adjustments_applied=ai_optimization_result.get('total_adjustments', 0)
            )
            
            # 히스토리에 추가
            self.performance_history.append(asdict(weekly_performance))
            
            # 최근 20주만 유지
            if len(self.performance_history) > 20:
                self.performance_history = self.performance_history[-20:]
            
            self.save_performance_history()
            
            logger.info(f"✅ 주간 성과 기록: ML={ml_accuracy:.1%}, AI기여={ai_contribution:.1%}, 결합={combined_accuracy:.1%}")
            return weekly_performance
            
        except Exception as e:
            logger.error(f"주간 성과 업데이트 실패: {e}")
            return None
    
    def update_power_balance(self, weekly_performance: WeeklyPerformance) -> PowerBalance:
        """파워 밸런스 업데이트"""
        try:
            logger.info("⚖️ ML/AI 파워 밸런스 업데이트")
            
            # 최근 성과 분석
            recent_weeks = self.performance_history[-4:] if len(self.performance_history) >= 4 else self.performance_history
            
            if len(recent_weeks) < 2:
                logger.info("성과 데이터 부족, 기본 밸런스 유지")
                return self.current_power_balance
            
            # ML vs AI 성과 비교
            ml_trend = self.calculate_ml_trend(recent_weeks)
            ai_effectiveness = self.calculate_ai_effectiveness(recent_weeks)
            
            # 새로운 파워 밸런스 계산
            new_ml_power, new_ai_power = self.calculate_new_balance(
                ml_trend, ai_effectiveness, weekly_performance
            )
            
            # 밸런스 변화 이유 생성
            balance_reason = self.generate_balance_reason(
                ml_trend, ai_effectiveness, new_ml_power - self.current_power_balance.ml_power
            )
            
            # 신뢰도 계산
            confidence_level = self.calculate_balance_confidence(recent_weeks)
            
            # 파워 밸런스 업데이트
            self.current_power_balance = PowerBalance(
                ml_power=new_ml_power,
                ai_power=new_ai_power,
                balance_reason=balance_reason,
                confidence_level=confidence_level,
                adjustment_history=self.current_power_balance.adjustment_history + [{
                    'date': datetime.now().isoformat(),
                    'ml_power': new_ml_power,
                    'ai_power': new_ai_power,
                    'reason': balance_reason
                }]
            )
            
            # 조정 히스토리는 최근 10개만 유지
            if len(self.current_power_balance.adjustment_history) > 10:
                self.current_power_balance.adjustment_history = self.current_power_balance.adjustment_history[-10:]
            
            logger.info(f"🔄 파워 밸런스 조정: ML={new_ml_power:.2f}, AI={new_ai_power:.2f}")
            logger.info(f"   이유: {balance_reason}")
            
            return self.current_power_balance
            
        except Exception as e:
            logger.error(f"파워 밸런스 업데이트 실패: {e}")
            return self.current_power_balance
    
    def calculate_ml_trend(self, recent_weeks: List[Dict]) -> float:
        """ML 성과 트렌드 계산"""
        try:
            ml_accuracies = [week['ml_accuracy'] for week in recent_weeks]
            
            if len(ml_accuracies) < 2:
                return 0.0
            
            # 최근 4주 트렌드 계산
            if len(ml_accuracies) >= 4:
                recent_avg = np.mean(ml_accuracies[-2:])  # 최근 2주
                older_avg = np.mean(ml_accuracies[-4:-2])  # 이전 2주
                trend = recent_avg - older_avg
            else:
                trend = ml_accuracies[-1] - ml_accuracies[0]
            
            return trend
            
        except:
            return 0.0
    
    def calculate_ai_effectiveness(self, recent_weeks: List[Dict]) -> float:
        """AI 효과성 계산"""
        try:
            ai_contributions = [week.get('ai_contribution', 0) for week in recent_weeks]
            combined_improvements = []
            
            for week in recent_weeks:
                ml_acc = week.get('ml_accuracy', 0)
                ai_contrib = week.get('ai_contribution', 0)
                combined_acc = week.get('combined_accuracy', ml_acc)
                
                # 실제 결합 효과 계산
                actual_improvement = combined_acc - ml_acc
                combined_improvements.append(actual_improvement)
            
            # AI의 평균 기여도 vs 예상 기여도
            avg_ai_contribution = np.mean(ai_contributions)
            avg_actual_improvement = np.mean(combined_improvements)
            
            # AI 효과성 = 실제 개선 / 예상 개선
            if avg_ai_contribution > 0:
                effectiveness = avg_actual_improvement / avg_ai_contribution
            else:
                effectiveness = 0.0
            
            return max(0.0, min(2.0, effectiveness))  # 0-2 범위로 제한
            
        except:
            return 0.5
    
    def calculate_new_balance(self, ml_trend: float, ai_effectiveness: float,
                            current_performance: WeeklyPerformance) -> Tuple[float, float]:
        """새로운 파워 밸런스 계산"""
        try:
            current_ml = self.current_power_balance.ml_power
            current_ai = self.current_power_balance.ai_power
            
            # ML 성과에 따른 조정
            ml_adjustment = 0.0
            if ml_trend > 0.05:  # 5% 이상 개선
                ml_adjustment += 0.1
            elif ml_trend < -0.05:  # 5% 이상 악화
                ml_adjustment -= 0.1
            elif ml_trend > 0.02:  # 2% 이상 개선
                ml_adjustment += 0.05
            elif ml_trend < -0.02:  # 2% 이상 악화
                ml_adjustment -= 0.05
            
            # AI 효과성에 따른 조정
            ai_adjustment = 0.0
            if ai_effectiveness > 1.2:  # AI가 예상보다 120% 효과적
                ai_adjustment += 0.1
            elif ai_effectiveness < 0.5:  # AI가 예상보다 50% 미만 효과적
                ai_adjustment -= 0.05
            elif ai_effectiveness > 0.8:  # AI가 80% 이상 효과적
                ai_adjustment += 0.05
            
            # 현재 성과 기반 보정
            if current_performance.combined_accuracy > 0.85:  # 85% 이상 정확도
                # 성과가 좋으면 현재 밸런스 유지하는 방향으로
                ml_adjustment *= 0.5
                ai_adjustment *= 0.5
            elif current_performance.combined_accuracy < 0.65:  # 65% 미만 정확도
                # 성과가 나쁘면 더 적극적 조정
                ml_adjustment *= 1.5
                ai_adjustment *= 1.5
            
            # 새로운 밸런스 계산
            new_ml_power = current_ml + ml_adjustment - ai_adjustment * 0.5
            new_ai_power = 1.0 - new_ml_power
            
            # 범위 제한
            new_ml_power = max(0.3, min(0.9, new_ml_power))
            new_ai_power = max(0.1, min(0.7, new_ai_power))
            
            # 정규화
            total = new_ml_power + new_ai_power
            new_ml_power /= total
            new_ai_power /= total
            
            return new_ml_power, new_ai_power
            
        except:
            return 0.7, 0.3
    
    def generate_balance_reason(self, ml_trend: float, ai_effectiveness: float,
                              power_change: float) -> str:
        """밸런스 변화 이유 생성"""
        reasons = []
        
        if abs(power_change) < 0.02:
            return "안정적 성과로 밸런스 유지"
        
        if power_change > 0.05:
            # ML 파워 증가
            if ml_trend > 0.05:
                reasons.append("ML 성과 대폭 개선")
            if ai_effectiveness < 0.7:
                reasons.append("AI 효과성 제한적")
        elif power_change < -0.05:
            # AI 파워 증가
            if ai_effectiveness > 1.0:
                reasons.append("AI 조정 효과 우수")
            if ml_trend < -0.02:
                reasons.append("ML 성과 개선 필요")
        else:
            # 소폭 조정
            if ml_trend > 0.02:
                reasons.append("ML 소폭 개선")
            if ai_effectiveness > 0.8:
                reasons.append("AI 기여도 증가")
        
        return " + ".join(reasons) if reasons else "성과 기반 미세 조정"
    
    def calculate_balance_confidence(self, recent_weeks: List[Dict]) -> float:
        """밸런스 신뢰도 계산"""
        try:
            if len(recent_weeks) < 3:
                return 0.5
            
            # 성과 일관성 계산
            accuracies = [week['combined_accuracy'] for week in recent_weeks]
            consistency = 1.0 - (np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 1.0)
            
            # 예측 수 기반 신뢰도
            avg_predictions = np.mean([week.get('prediction_count', 0) for week in recent_weeks])
            prediction_confidence = min(1.0, avg_predictions / self.convergence_thresholds['min_predictions'])
            
            # 종합 신뢰도
            confidence = (consistency * 0.7 + prediction_confidence * 0.3)
            
            return max(0.3, min(0.95, confidence))
            
        except:
            return 0.6
    
    def analyze_convergence(self, weekly_performance: WeeklyPerformance) -> ConvergenceState:
        """수렴 상태 분석"""
        try:
            logger.info("🎯 수렴 상태 분석")
            
            if len(self.performance_history) < 3:
                self.convergence_state.status = 'improving'
                self.convergence_state.next_milestone = f"{3 - len(self.performance_history)}주 더 필요"
                return self.convergence_state
            
            # 최근 성과 데이터
            recent_performances = self.performance_history[-4:] if len(self.performance_history) >= 4 else self.performance_history
            accuracies = [perf['combined_accuracy'] for perf in recent_performances]
            
            # 개선률 계산
            improvement_rate = self.calculate_improvement_rate(accuracies)
            
            # 안정성 점수 계산
            stability_score = self.calculate_stability_score(accuracies)
            
            # 수렴 상태 판단
            new_status = self.determine_convergence_status(
                accuracies, improvement_rate, stability_score
            )
            
            # 상태 변화 추적
            if new_status == self.convergence_state.status:
                weeks_in_state = self.convergence_state.weeks_in_state + 1
            else:
                weeks_in_state = 1
            
            # 신뢰도 계산
            confidence = self.calculate_convergence_confidence(
                accuracies, improvement_rate, stability_score, weeks_in_state
            )
            
            # 다음 마일스톤 설정
            next_milestone = self.determine_next_milestone(new_status, accuracies, weeks_in_state)
            
            # 수렴 상태 업데이트
            self.convergence_state = ConvergenceState(
                status=new_status,
                confidence=confidence,
                weeks_in_state=weeks_in_state,
                improvement_rate=improvement_rate,
                stability_score=stability_score,
                next_milestone=next_milestone
            )
            
            self.save_convergence_state()
            
            logger.info(f"📈 수렴 상태: {new_status} (신뢰도: {confidence:.1%}, {weeks_in_state}주 지속)")
            return self.convergence_state
            
        except Exception as e:
            logger.error(f"수렴 상태 분석 실패: {e}")
            return self.convergence_state
    
    def calculate_improvement_rate(self, accuracies: List[float]) -> float:
        """개선률 계산"""
        try:
            if len(accuracies) < 2:
                return 0.0
            
            if len(accuracies) >= 4:
                # 최근 2주 vs 이전 2주
                recent_avg = np.mean(accuracies[-2:])
                older_avg = np.mean(accuracies[-4:-2])
                return recent_avg - older_avg
            else:
                # 최신 vs 최초
                return accuracies[-1] - accuracies[0]
                
        except:
            return 0.0
    
    def calculate_stability_score(self, accuracies: List[float]) -> float:
        """안정성 점수 계산"""
        try:
            if len(accuracies) < 2:
                return 0.0
            
            # 표준편차 기반 안정성 (낮을수록 안정적)
            std_dev = np.std(accuracies)
            avg_accuracy = np.mean(accuracies)
            
            if avg_accuracy > 0:
                coefficient_of_variation = std_dev / avg_accuracy
                stability = max(0.0, 1.0 - coefficient_of_variation * 5)  # 5배 페널티
            else:
                stability = 0.0
            
            return stability
            
        except:
            return 0.0
    
    def determine_convergence_status(self, accuracies: List[float],
                                   improvement_rate: float, stability_score: float) -> str:
        """수렴 상태 판단"""
        try:
            current_accuracy = accuracies[-1] if accuracies else 0
            
            # 목표 달성 및 안정성 확인
            if (current_accuracy >= self.convergence_thresholds['accuracy_target'] and 
                stability_score >= 0.8 and
                abs(improvement_rate) <= self.convergence_thresholds['improvement_threshold']):
                return 'converged'
            
            # 수렴 중 (목표에 가깝고 안정적)
            elif (current_accuracy >= 0.75 and 
                  stability_score >= 0.7 and
                  abs(improvement_rate) <= self.convergence_thresholds['stability_threshold']):
                return 'converging'
            
            # 발산 (성과 악화)
            elif improvement_rate < -0.05:
                return 'diverging'
            
            # 개선 중
            else:
                return 'improving'
                
        except:
            return 'improving'
    
    def calculate_convergence_confidence(self, accuracies: List[float],
                                       improvement_rate: float, stability_score: float,
                                       weeks_in_state: int) -> float:
        """수렴 신뢰도 계산"""
        try:
            base_confidence = 0.5
            
            # 성과 수준 보너스
            current_accuracy = accuracies[-1] if accuracies else 0
            if current_accuracy >= 0.85:
                base_confidence += 0.3
            elif current_accuracy >= 0.75:
                base_confidence += 0.2
            elif current_accuracy >= 0.65:
                base_confidence += 0.1
            
            # 안정성 보너스
            base_confidence += stability_score * 0.2
            
            # 상태 지속 기간 보너스
            base_confidence += min(0.2, weeks_in_state * 0.05)
            
            # 데이터 충분성 보너스
            if len(accuracies) >= 4:
                base_confidence += 0.1
            
            return max(0.2, min(0.95, base_confidence))
            
        except:
            return 0.6
    
    def determine_next_milestone(self, status: str, accuracies: List[float],
                               weeks_in_state: int) -> str:
        """다음 마일스톤 결정"""
        try:
            current_accuracy = accuracies[-1] if accuracies else 0
            
            if status == 'converged':
                return "성과 유지 및 시스템 안정성 모니터링"
            
            elif status == 'converging':
                needed_weeks = self.convergence_thresholds['convergence_weeks'] - weeks_in_state
                if needed_weeks > 0:
                    return f"{needed_weeks}주 더 안정적 성과 유지"
                else:
                    return "수렴 달성 임박"
            
            elif status == 'improving':
                if current_accuracy < 0.65:
                    return "예측 정확도 65% 달성"
                elif current_accuracy < 0.75:
                    return "예측 정확도 75% 달성"
                else:
                    return "안정성 개선 (변동성 감소)"
            
            elif status == 'diverging':
                return "성과 회복 및 파라미터 재조정"
            
            else:
                return "데이터 축적 및 성과 모니터링"
                
        except:
            return "성과 개선 지속"
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """수렴 현황 요약"""
        try:
            recent_performance = self.performance_history[-1] if self.performance_history else None
            
            summary = {
                'current_status': self.convergence_state.status,
                'status_confidence': self.convergence_state.confidence,
                'weeks_tracking': len(self.performance_history),
                'current_power_balance': {
                    'ml_power': self.current_power_balance.ml_power,
                    'ai_power': self.current_power_balance.ai_power,
                    'balance_reason': self.current_power_balance.balance_reason
                },
                'performance_metrics': {
                    'latest_accuracy': recent_performance['combined_accuracy'] if recent_performance else 0,
                    'improvement_rate': self.convergence_state.improvement_rate,
                    'stability_score': self.convergence_state.stability_score
                },
                'milestones': {
                    'next_target': self.convergence_state.next_milestone,
                    'target_accuracy': self.convergence_thresholds['accuracy_target'],
                    'weeks_in_current_state': self.convergence_state.weeks_in_state
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"수렴 현황 요약 실패: {e}")
            return {}
    
    def get_power_balance(self) -> Tuple[float, float]:
        """현재 파워 밸런스 반환"""
        return self.current_power_balance.ml_power, self.current_power_balance.ai_power

# 독립 실행 테스트
async def main():
    """테스트 실행"""
    tracker = ConvergenceTracker()
    
    # 테스트 데이터
    test_validation = {
        'ml_accuracy': 0.75,
        'total_predictions': 18,
        'mae': 8.2,
        'direction_accuracy': 0.78,
        'best_predictions': [{'symbol': 'IONQ'}, {'symbol': 'RGTI'}],
        'worst_predictions': [{'symbol': 'MBLY'}]
    }
    
    test_ai_optimization = {
        'expected_improvement': 5.5,
        'total_adjustments': 12
    }
    
    logger.info("=== ML/AI 수렴 추적 시스템 테스트 ===")
    
    # 성과 업데이트
    weekly_perf = await tracker.update_weekly_performance(test_validation, test_ai_optimization)
    if weekly_perf:
        logger.info(f"✅ 주간 성과 기록: {weekly_perf.combined_accuracy:.1%}")
    
    # 파워 밸런스 업데이트
    power_balance = tracker.update_power_balance(weekly_perf)
    logger.info(f"⚖️ 파워 밸런스: ML={power_balance.ml_power:.2f}, AI={power_balance.ai_power:.2f}")
    
    # 수렴 분석
    convergence = tracker.analyze_convergence(weekly_perf)
    logger.info(f"📈 수렴 상태: {convergence.status} (신뢰도: {convergence.confidence:.1%})")
    
    # 현황 요약
    summary = tracker.get_convergence_summary()
    logger.info(f"📊 현재 정확도: {summary['performance_metrics']['latest_accuracy']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())