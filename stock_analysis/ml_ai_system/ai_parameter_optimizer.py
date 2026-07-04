#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ AI 주도 파라미터 최적화 엔진

AI 오차 분석 결과를 바탕으로 59개 ML 파라미터를 적극적으로 조정
ML vs AI 파워 밸런스에 따라 조정 강도를 동적으로 변경
"""

import json
import logging
import asyncio
import os
import copy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParameterAdjustment:
    """파라미터 조정 기록"""
    parameter_name: str
    current_value: float
    suggested_value: float
    adjustment_magnitude: float
    adjustment_reason: str
    confidence: float
    error_pattern_source: str

@dataclass
class OptimizationResult:
    """최적화 결과"""
    optimization_date: str
    total_adjustments: int
    major_adjustments: List[ParameterAdjustment]
    minor_adjustments: List[ParameterAdjustment]
    ml_power_factor: float
    ai_power_factor: float
    optimization_confidence: float
    expected_improvement: float
    risk_assessment: Dict[str, Any]

class AIParameterOptimizer:
    """AI 주도 파라미터 최적화 엔진"""
    
    def __init__(self):
        self.ml_parameters_file = "ml_parameters.json"
        self.optimization_history_file = "ai_optimization_history.json"
        
        # 조정 가능한 파라미터 정의 (필터링 제외)
        self.adjustable_parameters = {
            # 메인 스코어링 가중치 (6개) - 합계 1.0 유지
            'main_weights': [
                'pattern_score', 'convergence_score', 'growth_score', 
                'tech_score', 'institutional_score', 'financial_score'
            ],
            
            # 세부 스코어링 가중치 (28개) - 각 카테고리별 합계 1.0 유지
            'pattern_weights': [
                'crash_depth_weight', 'recovery_velocity_weight', 'recovery_position_weight',
                'volatility_compression_weight', 'support_strength_weight', 
                'breakout_proximity_weight', 'volume_pattern_weight', 'pattern_similarity_weight'
            ],
            'convergence_weights': [
                'rsi_recovery_weight', 'macd_timing_weight', 'bollinger_squeeze_weight',
                'moving_avg_convergence_weight', 'volume_oscillator_weight', 'technical_confluence_weight'
            ],
            'growth_weights': [
                'revenue_acceleration_weight', 'pipeline_strength_weight',
                'partnership_catalyst_weight', 'market_expansion_weight', 'regulatory_tailwind_weight'
            ],
            'tech_weights': [
                'innovation_cycle_position_weight', 'tech_adoption_curve_weight',
                'scaling_readiness_weight', 'tech_validation_weight'
            ],
            'institutional_weights': [
                'institutional_flow_weight', 'analyst_momentum_weight', 'insider_signal_weight'
            ],
            'financial_weights': [
                'cash_adequacy_weight', 'debt_management_weight'
            ],
            
            # Sweet Spot 배수 (5개) - 독립적 조정
            'sweet_spot_multipliers': [
                'early_recovery_multiplier', 'mid_recovery_multiplier', 'late_recovery_multiplier',
                'golden_time_multiplier', 'overheated_penalty'
            ],
            
            # 딥테크 카테고리 배수 (8개) - 독립적 조정
            'deeptech_multipliers': [
                'ai_computing', 'quantum_tech', 'bio_health_tech', 'mobility_tech',
                'semiconductor', 'energy_materials', 'security_fintech', 'emerging_tech'
            ],
            
            # 거래량 신호 가중치 (4개) - 독립적 조정
            'volume_weights': [
                'spike_signal_weight', 'trend_signal_weight',
                'combined_signal_weight', 'volume_quality_weight'
            ],
            
            # 서브카테고리 가중치 (8개 - 실제 16개 중 주요 8개)
            'subcategory_weights': [
                'machine_learning', 'spatial_computing', 'evtol', 'robotics',
                'biotech_ai', 'neural_interface', 'energy_storage', 'new_materials'
            ]
        }
        
        # 조정 제한 설정
        self.adjustment_limits = {
            'max_single_adjustment': 0.25,  # 한번에 최대 25% 조정
            'max_cumulative_adjustment': 0.50,  # 누적 최대 50% 조정
            'min_weight': 0.02,  # 최소 가중치 2%
            'max_weight': 0.80,  # 최대 가중치 80%
            'min_multiplier': 0.3,  # 최소 배수
            'max_multiplier': 3.0   # 최대 배수
        }
        
        # 최적화 히스토리 로드
        self.optimization_history = self.load_optimization_history()
    
    def load_optimization_history(self) -> List[Dict]:
        """최적화 히스토리 로드"""
        try:
            if os.path.exists(self.optimization_history_file):
                with open(self.optimization_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"최적화 히스토리 로드 실패: {e}")
            return []
    
    def save_optimization_history(self):
        """최적화 히스토리 저장"""
        try:
            with open(self.optimization_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_history, f, indent=2, ensure_ascii=False)
            logger.info(f"최적화 히스토리 저장 완료: {len(self.optimization_history)}개 기록")
        except Exception as e:
            logger.error(f"최적화 히스토리 저장 실패: {e}")
    
    async def optimize_parameters(self, ai_analysis_result: Dict,
                                ml_power_factor: float = 0.7,
                                ai_power_factor: float = 0.3) -> OptimizationResult:
        """AI 주도 파라미터 최적화"""
        try:
            logger.info(f"⚡ AI 파라미터 최적화 시작: ML파워={ml_power_factor:.2f}, AI파워={ai_power_factor:.2f}")
            
            # 현재 ML 파라미터 로드
            current_params = self.load_ml_parameters()
            if not current_params:
                logger.error("ML 파라미터 로드 실패")
                return None
            
            # 조정 계획 생성
            adjustment_plan = self.generate_adjustment_plan(
                ai_analysis_result, ml_power_factor, ai_power_factor
            )
            
            # 조정 적용
            optimized_params, applied_adjustments = await self.apply_adjustments(
                current_params, adjustment_plan
            )
            
            # 최적화 결과 생성
            optimization_result = OptimizationResult(
                optimization_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                total_adjustments=len(applied_adjustments),
                major_adjustments=[adj for adj in applied_adjustments if adj.adjustment_magnitude > 0.1],
                minor_adjustments=[adj for adj in applied_adjustments if adj.adjustment_magnitude <= 0.1],
                ml_power_factor=ml_power_factor,
                ai_power_factor=ai_power_factor,
                optimization_confidence=self.calculate_optimization_confidence(applied_adjustments),
                expected_improvement=self.estimate_improvement(applied_adjustments),
                risk_assessment=self.assess_optimization_risks(applied_adjustments)
            )
            
            # 최적화된 파라미터 저장
            await self.save_optimized_parameters(optimized_params, optimization_result)
            
            # 히스토리에 추가
            self.optimization_history.append(asdict(optimization_result))
            self.save_optimization_history()
            
            logger.info(f"✅ AI 파라미터 최적화 완료: {len(applied_adjustments)}개 조정, 신뢰도 {optimization_result.optimization_confidence:.1%}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"AI 파라미터 최적화 실패: {e}")
            return None
    
    def generate_adjustment_plan(self, ai_analysis: Dict, 
                               ml_power: float, ai_power: float) -> List[ParameterAdjustment]:
        """조정 계획 생성"""
        try:
            adjustment_plan = []
            
            # AI 분석에서 조정 제안 추출
            ai_suggestions = ai_analysis.get('parameter_adjustment_suggestions', {})
            error_patterns = ai_analysis.get('major_error_patterns', [])
            sector_analysis = ai_analysis.get('sector_analysis', [])
            
            # 1. 오차 패턴 기반 조정
            for pattern in error_patterns:
                pattern_adjustments = self.generate_pattern_based_adjustments(
                    pattern, ai_power
                )
                adjustment_plan.extend(pattern_adjustments)
            
            # 2. 섹터별 성과 기반 조정
            for sector in sector_analysis:
                sector_adjustments = self.generate_sector_based_adjustments(
                    sector, ai_power
                )
                adjustment_plan.extend(sector_adjustments)
            
            # 3. AI 직접 제안 기반 조정
            for param_name, adjustment_value in ai_suggestions.items():
                if self.is_adjustable_parameter(param_name):
                    adjustment_plan.append(self.create_adjustment(
                        param_name, adjustment_value, ai_power,
                        "AI 직접 제안", "AI 분석 결과"
                    ))
            
            # 4. 파워 밸런스 기반 조정 강도 적용
            for adjustment in adjustment_plan:
                # AI 파워가 높을수록 더 적극적 조정
                base_magnitude = abs(adjustment.adjustment_magnitude)
                adjusted_magnitude = base_magnitude * ai_power * 2.0  # 최대 2배 증폭
                
                # ML 파워가 높으면 보수적 조정
                if ml_power > 0.7:
                    adjusted_magnitude *= 0.5  # 50% 감소
                
                adjustment.adjustment_magnitude = min(
                    adjusted_magnitude, self.adjustment_limits['max_single_adjustment']
                )
                adjustment.confidence *= ai_power  # AI 파워에 비례한 신뢰도
            
            # 중복 제거 및 우선순위 정렬
            adjustment_plan = self.deduplicate_and_prioritize(adjustment_plan)
            
            logger.info(f"조정 계획 생성 완료: {len(adjustment_plan)}개 조정")
            return adjustment_plan
            
        except Exception as e:
            logger.error(f"조정 계획 생성 실패: {e}")
            return []
    
    def generate_pattern_based_adjustments(self, error_pattern: Dict, 
                                         ai_power: float) -> List[ParameterAdjustment]:
        """오차 패턴 기반 조정"""
        adjustments = []
        
        try:
            pattern_type = error_pattern.get('pattern_type', '')
            error_magnitude = error_pattern.get('error_magnitude', 0)
            affected_symbols = error_pattern.get('affected_symbols', [])
            
            # 조정 강도 계산 (오차 크기와 AI 파워에 비례)
            adjustment_strength = min(0.2, error_magnitude * 0.01 * ai_power)
            
            if pattern_type == "과대평가":
                # 낙관 편향 완화
                adjustments.extend([
                    self.create_adjustment(
                        'golden_time_multiplier', -adjustment_strength * 1.5,
                        ai_power * 0.8, "과대평가 패턴", pattern_type
                    ),
                    self.create_adjustment(
                        'pattern_score', -adjustment_strength,
                        ai_power * 0.7, "패턴 스코어 과신", pattern_type
                    ),
                    self.create_adjustment(
                        'early_recovery_multiplier', -adjustment_strength * 0.8,
                        ai_power * 0.6, "초기 회복 과대평가", pattern_type
                    )
                ])
                
            elif pattern_type == "과소평가":
                # 보수 편향 완화
                adjustments.extend([
                    self.create_adjustment(
                        'growth_score', +adjustment_strength,
                        ai_power * 0.8, "성장성 과소평가", pattern_type
                    ),
                    self.create_adjustment(
                        'tech_score', +adjustment_strength * 0.8,
                        ai_power * 0.7, "기술력 과소평가", pattern_type
                    ),
                    self.create_adjustment(
                        'revenue_acceleration_weight', +adjustment_strength,
                        ai_power * 0.6, "매출 가속화 과소평가", pattern_type
                    )
                ])
                
            elif pattern_type == "방향성_오류":
                # 기술지표 강화
                adjustments.extend([
                    self.create_adjustment(
                        'convergence_score', +adjustment_strength,
                        ai_power * 0.9, "기술지표 보강 필요", pattern_type
                    ),
                    self.create_adjustment(
                        'macd_timing_weight', +adjustment_strength * 0.7,
                        ai_power * 0.8, "MACD 타이밍 개선", pattern_type
                    ),
                    self.create_adjustment(
                        'rsi_recovery_weight', +adjustment_strength * 0.6,
                        ai_power * 0.7, "RSI 회복 신호 강화", pattern_type
                    )
                ])
            
        except Exception as e:
            logger.warning(f"패턴 기반 조정 생성 실패: {e}")
        
        return adjustments
    
    def generate_sector_based_adjustments(self, sector_analysis: Dict,
                                        ai_power: float) -> List[ParameterAdjustment]:
        """섹터별 성과 기반 조정"""
        adjustments = []
        
        try:
            sector = sector_analysis.get('sector', '')
            average_error = sector_analysis.get('average_error', 0)
            sector_bias = sector_analysis.get('sector_bias', 0)
            
            # 섹터 오차가 크면 해당 섹터 배수 조정
            if average_error > 8.0:  # 8% 이상 오차
                adjustment_strength = min(0.15, average_error * 0.01 * ai_power)
                
                # 딥테크 카테고리 배수 조정
                sector_param_map = {
                    'ai_computing': 'ai_computing',
                    'quantum_tech': 'quantum_tech', 
                    'bio_health_tech': 'bio_health_tech',
                    'mobility_tech': 'mobility_tech',
                    'semiconductor': 'semiconductor',
                    'energy_materials': 'energy_materials'
                }
                
                if sector in sector_param_map:
                    multiplier_param = f"{sector_param_map[sector]}_multiplier"
                    
                    if sector_bias > 5.0:  # 과대평가
                        adjustments.append(self.create_adjustment(
                            multiplier_param, -adjustment_strength,
                            ai_power * 0.8, f"{sector} 섹터 과대평가", "섹터 분석"
                        ))
                    elif sector_bias < -5.0:  # 과소평가
                        adjustments.append(self.create_adjustment(
                            multiplier_param, +adjustment_strength,
                            ai_power * 0.8, f"{sector} 섹터 과소평가", "섹터 분석"
                        ))
            
        except Exception as e:
            logger.warning(f"섹터 기반 조정 생성 실패: {e}")
        
        return adjustments
    
    def create_adjustment(self, param_name: str, adjustment_value: float,
                         confidence: float, reason: str, source: str) -> ParameterAdjustment:
        """조정 객체 생성"""
        return ParameterAdjustment(
            parameter_name=param_name,
            current_value=0.0,  # 나중에 실제 값으로 업데이트
            suggested_value=0.0,  # 나중에 계산
            adjustment_magnitude=abs(adjustment_value),
            adjustment_reason=reason,
            confidence=confidence,
            error_pattern_source=source
        )
    
    def is_adjustable_parameter(self, param_name: str) -> bool:
        """조정 가능한 파라미터인지 확인"""
        for category in self.adjustable_parameters.values():
            if param_name in category or param_name.endswith('_weight') or param_name.endswith('_multiplier'):
                return True
        return False
    
    def deduplicate_and_prioritize(self, adjustments: List[ParameterAdjustment]) -> List[ParameterAdjustment]:
        """중복 제거 및 우선순위 정렬"""
        # 파라미터별 그룹화
        param_groups = {}
        for adj in adjustments:
            if adj.parameter_name not in param_groups:
                param_groups[adj.parameter_name] = []
            param_groups[adj.parameter_name].append(adj)
        
        # 각 파라미터별로 최고 신뢰도 조정 선택
        final_adjustments = []
        for param_name, param_adjustments in param_groups.items():
            best_adjustment = max(param_adjustments, key=lambda x: x.confidence)
            final_adjustments.append(best_adjustment)
        
        # 조정 강도별 우선순위 정렬
        return sorted(final_adjustments, key=lambda x: x.adjustment_magnitude, reverse=True)
    
    async def apply_adjustments(self, current_params: Dict, 
                              adjustment_plan: List[ParameterAdjustment]) -> Tuple[Dict, List[ParameterAdjustment]]:
        """조정 적용"""
        try:
            optimized_params = copy.deepcopy(current_params)
            applied_adjustments = []
            
            for adjustment in adjustment_plan:
                try:
                    # 현재 값 찾기 및 새로운 값 계산
                    current_value, param_path = self.find_parameter_value(
                        optimized_params, adjustment.parameter_name
                    )
                    
                    if current_value is None:
                        logger.warning(f"파라미터 찾기 실패: {adjustment.parameter_name}")
                        continue
                    
                    # 새로운 값 계산
                    if adjustment.parameter_name.endswith('_multiplier'):
                        # 배수는 곱셈 조정
                        new_value = current_value * (1 + adjustment.adjustment_magnitude)
                        new_value = max(self.adjustment_limits['min_multiplier'],
                                      min(self.adjustment_limits['max_multiplier'], new_value))
                    else:
                        # 가중치는 덧셈 조정
                        adjustment_sign = 1 if '+' in adjustment.adjustment_reason or '증가' in adjustment.adjustment_reason else -1
                        new_value = current_value + (adjustment.adjustment_magnitude * adjustment_sign)
                        new_value = max(self.adjustment_limits['min_weight'],
                                      min(self.adjustment_limits['max_weight'], new_value))
                    
                    # 조정 적용
                    self.set_parameter_value(optimized_params, param_path, new_value)
                    
                    # 적용된 조정 기록
                    adjustment.current_value = current_value
                    adjustment.suggested_value = new_value
                    applied_adjustments.append(adjustment)
                    
                    logger.info(f"   {adjustment.parameter_name}: {current_value:.3f} → {new_value:.3f}")
                    
                except Exception as e:
                    logger.warning(f"조정 적용 실패 ({adjustment.parameter_name}): {e}")
                    continue
            
            # 가중치 정규화 (합계 1.0 유지)
            optimized_params = self.normalize_weight_groups(optimized_params)
            
            logger.info(f"조정 적용 완료: {len(applied_adjustments)}개 파라미터")
            return optimized_params, applied_adjustments
            
        except Exception as e:
            logger.error(f"조정 적용 실패: {e}")
            return current_params, []
    
    def find_parameter_value(self, params: Dict, param_name: str) -> Tuple[Optional[float], List[str]]:
        """파라미터 값 찾기"""
        try:
            # 메인 가중치
            if param_name in params.get('main_scoring_weights', {}):
                return params['main_scoring_weights'][param_name], ['main_scoring_weights', param_name]
            
            # 세부 가중치
            for category in ['pattern_scoring', 'convergence_scoring', 'growth_scoring', 
                           'tech_scoring', 'institutional_scoring', 'financial_scoring']:
                if category in params.get('detailed_scoring_weights', {}):
                    if param_name in params['detailed_scoring_weights'][category]:
                        return (params['detailed_scoring_weights'][category][param_name], 
                               ['detailed_scoring_weights', category, param_name])
            
            # Sweet Spot 배수
            if param_name in params.get('sweet_spot_multipliers', {}):
                return params['sweet_spot_multipliers'][param_name], ['sweet_spot_multipliers', param_name]
            
            # 딥테크 배수
            if param_name.endswith('_multiplier'):
                base_name = param_name.replace('_multiplier', '')
                if base_name in params.get('deeptech_category_multipliers', {}):
                    return (params['deeptech_category_multipliers'][base_name], 
                           ['deeptech_category_multipliers', base_name])
            
            # 거래량 가중치
            if param_name in params.get('volume_signal_weights', {}):
                return params['volume_signal_weights'][param_name], ['volume_signal_weights', param_name]
            
            return None, []
            
        except Exception as e:
            logger.warning(f"파라미터 값 찾기 실패 ({param_name}): {e}")
            return None, []
    
    def set_parameter_value(self, params: Dict, param_path: List[str], value: float):
        """파라미터 값 설정"""
        try:
            current = params
            for key in param_path[:-1]:
                current = current[key]
            current[param_path[-1]] = value
        except Exception as e:
            logger.warning(f"파라미터 값 설정 실패: {e}")
    
    def normalize_weight_groups(self, params: Dict) -> Dict:
        """가중치 그룹별 정규화"""
        try:
            # 메인 가중치 정규화
            if 'main_scoring_weights' in params:
                main_weights = params['main_scoring_weights']
                total = sum(main_weights.values())
                if total > 0:
                    for key in main_weights:
                        main_weights[key] = main_weights[key] / total
            
            # 세부 가중치 정규화
            if 'detailed_scoring_weights' in params:
                for category in params['detailed_scoring_weights']:
                    weights = params['detailed_scoring_weights'][category]
                    total = sum(weights.values())
                    if total > 0:
                        for key in weights:
                            weights[key] = weights[key] / total
            
            return params
            
        except Exception as e:
            logger.warning(f"가중치 정규화 실패: {e}")
            return params
    
    def calculate_optimization_confidence(self, adjustments: List[ParameterAdjustment]) -> float:
        """최적화 신뢰도 계산"""
        try:
            if not adjustments:
                return 0.3
            
            # 평균 조정 신뢰도
            avg_confidence = np.mean([adj.confidence for adj in adjustments])
            
            # 조정 일관성 (같은 방향 조정이 많으면 신뢰도 증가)
            consistency_bonus = 0.0
            if len(adjustments) > 3:
                similar_adjustments = sum(1 for adj in adjustments 
                                        if adj.error_pattern_source == adjustments[0].error_pattern_source)
                consistency_bonus = min(0.2, similar_adjustments * 0.05)
            
            return min(0.95, avg_confidence + consistency_bonus)
            
        except:
            return 0.5
    
    def estimate_improvement(self, adjustments: List[ParameterAdjustment]) -> float:
        """예상 개선도 추정"""
        try:
            if not adjustments:
                return 0.0
            
            # 조정 강도 기반 개선도 추정
            major_adjustments = [adj for adj in adjustments if adj.adjustment_magnitude > 0.1]
            minor_adjustments = [adj for adj in adjustments if adj.adjustment_magnitude <= 0.1]
            
            estimated_improvement = len(major_adjustments) * 2.0 + len(minor_adjustments) * 0.5
            
            # 신뢰도로 가중
            avg_confidence = np.mean([adj.confidence for adj in adjustments])
            estimated_improvement *= avg_confidence
            
            return min(15.0, estimated_improvement)  # 최대 15% 개선
            
        except:
            return 0.0
    
    def assess_optimization_risks(self, adjustments: List[ParameterAdjustment]) -> Dict[str, Any]:
        """최적화 리스크 평가"""
        try:
            risks = {
                'risk_level': 'low',
                'risk_factors': [],
                'mitigation_strategies': []
            }
            
            # 큰 조정이 많으면 리스크 증가
            major_adjustments = [adj for adj in adjustments if adj.adjustment_magnitude > 0.2]
            if len(major_adjustments) > 5:
                risks['risk_level'] = 'high'
                risks['risk_factors'].append("다수 파라미터 대폭 조정")
                risks['mitigation_strategies'].append("단계적 조정 권장")
            
            # 신뢰도가 낮으면 리스크 증가
            avg_confidence = np.mean([adj.confidence for adj in adjustments])
            if avg_confidence < 0.6:
                risks['risk_level'] = 'medium' if risks['risk_level'] == 'low' else 'high'
                risks['risk_factors'].append("AI 분석 신뢰도 낮음")
                risks['mitigation_strategies'].append("보수적 조정 권장")
            
            return risks
            
        except:
            return {'risk_level': 'unknown', 'risk_factors': ['평가 실패']}
    
    async def save_optimized_parameters(self, optimized_params: Dict, 
                                      optimization_result: OptimizationResult):
        """최적화된 파라미터 저장"""
        try:
            # ml_parameters.json 백업
            backup_file = f"ml_parameters_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if os.path.exists(self.ml_parameters_file):
                with open(self.ml_parameters_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
            
            # 최적화된 파라미터 저장
            final_params = {
                **optimized_params,
                'metadata': {
                    **optimized_params.get('metadata', {}),
                    'last_ai_optimization': datetime.now().isoformat(),
                    'ai_optimization_count': optimized_params.get('metadata', {}).get('ai_optimization_count', 0) + 1,
                    'optimization_confidence': optimization_result.optimization_confidence,
                    'adjustments_applied': optimization_result.total_adjustments
                }
            }
            
            with open(self.ml_parameters_file, 'w', encoding='utf-8') as f:
                json.dump(final_params, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 최적화된 파라미터 저장 완료: {backup_file}")
            
        except Exception as e:
            logger.error(f"최적화된 파라미터 저장 실패: {e}")
    
    def load_ml_parameters(self) -> Dict:
        """ML 파라미터 로드"""
        try:
            with open(self.ml_parameters_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"ML 파라미터 로드 실패: {e}")
            return {}

# 독립 실행 테스트
async def main():
    """테스트 실행"""
    optimizer = AIParameterOptimizer()
    
    # 테스트 AI 분석 결과
    test_ai_analysis = {
        'major_error_patterns': [
            {
                'pattern_type': '과대평가',
                'error_magnitude': 12.5,
                'affected_symbols': ['IONQ', 'RGTI'],
                'potential_causes': ['양자 기술 하이프']
            }
        ],
        'sector_analysis': [
            {
                'sector': 'quantum_tech',
                'average_error': 10.2,
                'sector_bias': 8.5
            }
        ],
        'parameter_adjustment_suggestions': {
            'golden_time_multiplier': -0.1,
            'quantum_tech_multiplier': -0.15
        }
    }
    
    logger.info("=== AI 파라미터 최적화 엔진 테스트 ===")
    
    # 최적화 실행
    result = await optimizer.optimize_parameters(
        test_ai_analysis, ml_power_factor=0.6, ai_power_factor=0.4
    )
    
    if result:
        logger.info(f"✅ 최적화 완료: {result.total_adjustments}개 조정")
        logger.info(f"   신뢰도: {result.optimization_confidence:.1%}")
        logger.info(f"   예상 개선: {result.expected_improvement:.1f}%")
        logger.info(f"   리스크: {result.risk_assessment['risk_level']}")
    else:
        logger.error("❌ 최적화 실패")

if __name__ == "__main__":
    asyncio.run(main())