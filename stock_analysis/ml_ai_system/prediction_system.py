#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ML+AI 통합 예측 생성/검증 시스템

ML이 예측하고, 1주 후 검증하여 AI와의 파워 밸런스를 조정하는 시스템
Sweet Spot 데이터베이스와 완전 연동
"""

import json
import logging
import asyncio
import os
import yfinance as yf
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
class PredictionRecord:
    """예측 기록"""
    symbol: str
    prediction_date: str
    current_score: float
    next_week_expected_return: float
    prediction_confidence: float
    key_factors: Dict[str, float]
    actual_return: Optional[float] = None
    validation_date: Optional[str] = None
    error: Optional[float] = None
    direction_correct: Optional[bool] = None

@dataclass
class ValidationResult:
    """검증 결과"""
    total_predictions: int
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    direction_accuracy: float  # 방향성 정확도
    ml_accuracy: float  # ML 전체 정확도
    best_predictions: List[Dict]
    worst_predictions: List[Dict]
    sector_accuracy: Dict[str, float]

class PredictionSystem:
    """ML+AI 통합 예측 생성/검증 시스템"""
    
    def __init__(self):
        self.sweet_spot_db_file = "sweet_spot_database.json"
        self.predictions_file = "ml_predictions_history.json"
        
        # ML/AI 파워 밸런스 초기값
        self.ml_confidence = 0.7  # ML 초기 신뢰도
        self.ai_confidence = 0.3  # AI 초기 신뢰도
        
        # 예측 기록
        self.prediction_history = self.load_prediction_history()
        
    def load_prediction_history(self) -> List[Dict]:
        """예측 히스토리 로드"""
        try:
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"예측 히스토리 로드 실패: {e}")
            return []
    
    def save_prediction_history(self):
        """예측 히스토리 저장"""
        try:
            with open(self.predictions_file, 'w', encoding='utf-8') as f:
                json.dump(self.prediction_history, f, indent=2, ensure_ascii=False)
            logger.info(f"예측 히스토리 저장 완료: {len(self.prediction_history)}개 기록")
        except Exception as e:
            logger.error(f"예측 히스토리 저장 실패: {e}")
    
    async def generate_weekly_predictions(self, limit: int = 20) -> List[PredictionRecord]:
        """주간 예측 생성 - 상위 Sweet Spot 종목들"""
        try:
            logger.info(f"🎯 주간 예측 생성 시작: 상위 {limit}개 종목")
            
            # Sweet Spot DB 로드
            sweet_spot_stocks = self.load_sweet_spot_stocks()
            if not sweet_spot_stocks:
                logger.error("Sweet Spot 종목 데이터가 없습니다")
                return []
            
            # 상위 종목 선별 (점수 높은 순)
            sorted_stocks = sorted(sweet_spot_stocks, 
                                 key=lambda x: x.get('selection_score', 0), reverse=True)
            top_stocks = sorted_stocks[:limit]
            
            predictions = []
            ml_params = self.load_ml_parameters()
            
            for stock in top_stocks:
                try:
                    prediction = await self.generate_single_prediction(stock, ml_params)
                    if prediction:
                        predictions.append(prediction)
                        logger.info(f"✅ {stock['symbol']}: {prediction.next_week_expected_return:+.1f}% 예측")
                    
                except Exception as e:
                    logger.warning(f"{stock.get('symbol', 'Unknown')} 예측 실패: {e}")
                    continue
            
            # 예측 기록에 추가 (ML 최적화 정보 포함)
            for pred in predictions:
                pred_dict = asdict(pred)
                
                # ML 최적화 정보 추가 (동적으로 추가된 속성)
                if hasattr(pred, 'ml_optimization'):
                    pred_dict['ml_optimization'] = pred.ml_optimization
                
                # 확장된 구조로 저장
                enhanced_pred = {
                    **pred_dict,
                    'current_metrics': {
                        'actual_return': pred_dict.get('actual_return', 0.0),
                        'selection_score': pred_dict.get('current_score', 0.0)
                    },
                    'ml_based_prediction': {
                        'expected_return': pred_dict.get('next_week_expected_return', 0.0),
                        'confidence': pred_dict.get('prediction_confidence', 0.0),
                        'key_factors': pred_dict.get('key_factors', {})
                    },
                    'validation': {
                        'actual_return_next_week': None,
                        'prediction_error': None,
                        'direction_correct': None,
                        'validation_date': None
                    }
                }
                
                self.prediction_history.append(enhanced_pred)
            
            self.save_prediction_history()
            
            logger.info(f"📊 주간 예측 완료: {len(predictions)}개 종목")
            return predictions
            
        except Exception as e:
            logger.error(f"주간 예측 생성 실패: {e}")
            return []
    
    async def generate_single_prediction(self, stock: Dict, ml_params: Dict) -> Optional[PredictionRecord]:
        """개별 종목 예측 생성 - ML 파라미터 기반"""
        try:
            symbol = stock['symbol']
            
            # 이전 점수 (최적화 전)
            original_score = stock.get('selection_score', 0)
            
            # 현재 ML 파라미터로 점수 계산 (최적화 후)
            optimized_score = await self.calculate_current_score(stock, ml_params)
            
            # ML 파라미터 기반 예상수익률 계산
            ml_based_prediction = await self.calculate_ml_based_expected_return(
                stock, original_score, optimized_score, ml_params
            )
            
            # 신뢰도 계산
            confidence = self.calculate_prediction_confidence(stock, ml_params)
            
            # 핵심 요인 분석
            key_factors = self.analyze_key_factors(stock, ml_params)
            
            # ML 최적화 정보 추가
            ml_optimization_info = {
                'old_score': original_score,
                'new_score': optimized_score,
                'score_change': optimized_score - original_score,
                'score_change_pct': ((optimized_score - original_score) / original_score * 100) if original_score > 0 else 0
            }
            
            prediction_record = PredictionRecord(
                symbol=symbol,
                prediction_date=datetime.now().strftime('%Y-%m-%d'),
                current_score=optimized_score,
                next_week_expected_return=ml_based_prediction,
                prediction_confidence=confidence,
                key_factors=key_factors
            )
            
            # ML 최적화 정보를 prediction_record에 추가 (동적으로)
            prediction_record.ml_optimization = ml_optimization_info
            
            logger.info(f"✅ {symbol}: 점수 {original_score:.1f}→{optimized_score:.1f} ({optimized_score-original_score:+.1f}), 예상수익률 {ml_based_prediction:+.1f}%")
            
            return prediction_record
            
        except Exception as e:
            logger.error(f"{stock.get('symbol')} 개별 예측 실패: {e}")
            return None
    
    async def calculate_current_score(self, stock: Dict, ml_params: Dict) -> float:
        """현재 ML 파라미터로 종목 점수 계산"""
        try:
            # ScoreRecalculator 사용하여 현재 점수 계산
            from score_recalculator import ScoreRecalculator
            
            score_calculator = ScoreRecalculator()
            
            # 캐시된 데이터 가져오기 (ml_optimizer DataCache 사용)
            from .ml_optimizer import DataCache
            cache = DataCache()
            stock_data = await cache.get_cached_data(stock['symbol'], 'complete_data')
            
            if not stock_data:
                logger.warning(f"{stock['symbol']} 캐시된 데이터 없음")
                return stock.get('selection_score', 0)
            
            # 현재 파라미터로 점수 계산
            current_params = ml_params.get('current_parameters', {})
            final_score = score_calculator.calculate_score_with_parameters(
                stock['symbol'], stock_data, current_params
            )
            
            return final_score
            
        except Exception as e:
            logger.warning(f"{stock['symbol']} 현재 점수 계산 실패: {e}")
            return stock.get('selection_score', 0)
    
    async def calculate_ml_based_expected_return(self, stock: Dict, original_score: float, optimized_score: float, ml_params: Dict) -> float:
        """ML 파라미터 최적화 기반 예상수익률 계산"""
        try:
            symbol = stock.get('symbol', 'Unknown')
            
            # 1. 점수 변화 계산
            score_change = optimized_score - original_score
            score_change_pct = (score_change / original_score * 100) if original_score > 0 else 0
            
            # 2. 현재 실제 수익률 가져오기
            current_return = stock.get('current_return', 0.0)  # Sweet Spot DB에서
            if current_return == 0.0:
                # 폴백: Sweet Spot DB 직접 조회
                current_return = await self.get_actual_return_from_sweet_spot_db(symbol)
            
            # 3. 파라미터 민감도 계산 (이 종목이 새 파라미터에 얼마나 유리한지)
            param_sensitivity = self.calculate_parameter_sensitivity(stock, ml_params)
            
            # 4. ML 기반 예상수익률 계산 공식
            # 핵심: 점수 절대값이 높을수록 높은 수익률 예상 (ML이 학습한 패턴)
            
            # 4-1. 점수 절대값 기반 예상수익률 (60%) - 핵심 신호
            # 점수가 높을수록 더 높은 수익률 기대 (ML이 학습한 상관관계)
            if optimized_score >= 80:
                score_based_return = 12.0 + (optimized_score - 80) * 0.5  # 80점 이상: 12%+ 예상
            elif optimized_score >= 70:
                score_based_return = 8.0 + (optimized_score - 70) * 0.4   # 70-80점: 8-12% 예상
            elif optimized_score >= 60:
                score_based_return = 4.0 + (optimized_score - 60) * 0.4   # 60-70점: 4-8% 예상
            elif optimized_score >= 50:
                score_based_return = 0.0 + (optimized_score - 50) * 0.4   # 50-60점: 0-4% 예상
            else:
                score_based_return = -5.0 + optimized_score * 0.1          # 50점 미만: 부정적 예상
            
            # 4-2. 현재 모멘텀 반영 (25%) - 트렌드 지속성
            momentum_component = current_return * 0.25  # 현재 추세의 25% 지속
            
            # 4-3. 점수 향상도 보너스 (15%) - 개선 신호
            improvement_bonus = 0.0
            if score_change > 0:
                improvement_bonus = min(score_change * 0.2, 3.0)  # 점수 향상시 보너스
            elif score_change < -5:  # 점수가 크게 하락한 경우만 패널티
                improvement_bonus = max(score_change * 0.1, -2.0)
            
            # 4-4. 최종 예상수익률
            expected_return = (
                score_based_return +    # 점수 절대값 기반 (60%)
                momentum_component +    # 모멘텀 지속 (25%)
                improvement_bonus       # 개선 보너스 (15%)
            )
            
            # 5. Sweet Spot 위치별 조정
            recovery_percent = stock.get('recovery_from_low_percent', 50)
            if 30 <= recovery_percent <= 80:  # 골든타임
                expected_return *= 1.1  # 10% 보너스
            elif recovery_percent > 150:  # 과열 구간
                expected_return *= 0.8   # 20% 패널티
            
            # 6. 현실적 범위로 제한 (-30% ~ +50%)
            expected_return = max(-30.0, min(50.0, expected_return))
            
            logger.debug(f"{symbol}: 점수변화 {score_change:+.1f}({score_change_pct:+.1f}%), "
                        f"현재수익률 {current_return:+.1f}%, 민감도 {param_sensitivity:.2f} "
                        f"→ 예상수익률 {expected_return:+.1f}%")
            
            return expected_return
            
        except Exception as e:
            logger.warning(f"{stock.get('symbol', 'Unknown')} ML 기반 예상수익률 계산 실패: {e}")
            # 폴백: 점수 변화만으로 단순 계산
            if original_score > 0:
                return ((optimized_score - original_score) / original_score) * 20  # 점수 1% 변화 = 수익률 0.2%
            return 0.0
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except:
            return 50.0
    
    def calculate_parameter_sensitivity(self, stock: Dict, ml_params: Dict) -> float:
        """파라미터 민감도 계산 - 새 파라미터가 이 종목에 얼마나 유리한지"""
        try:
            sensitivity = 0.0
            current_params = ml_params.get('current_parameters', {})
            
            # 1. 딥테크 카테고리 친화도
            tech_category = stock.get('tech_category', '')
            category_multipliers = current_params.get('deeptech_category_multipliers', {})
            if tech_category in category_multipliers:
                category_boost = category_multipliers[tech_category]
                sensitivity += (category_boost - 1.0) * 0.4  # -0.4 ~ +0.4 범위
            
            # 2. Sweet Spot 단계별 친화도
            recovery_percent = stock.get('recovery_from_low_percent', 50)
            multipliers = current_params.get('sweet_spot_multipliers', {})
            
            if 30 <= recovery_percent <= 80:  # 골든타임
                golden_multiplier = multipliers.get('golden_time_multiplier', 1.0)
                sensitivity += (golden_multiplier - 1.0) * 0.3
            elif recovery_percent < 30:  # 초기 회복
                early_multiplier = multipliers.get('early_recovery_multiplier', 1.0)
                sensitivity += (early_multiplier - 1.0) * 0.2
            elif recovery_percent > 150:  # 과열
                penalty = multipliers.get('overheated_penalty', 1.0)
                sensitivity += (penalty - 1.0) * 0.4  # 패널티 반영
            
            # 3. 거래량 신호 친화도
            if stock.get('volume_surge', False):
                volume_weights = current_params.get('volume_signal_weights', {})
                spike_weight = volume_weights.get('spike_signal_weight', 1.0)
                sensitivity += (spike_weight - 1.0) * 0.1
            
            # -1.0 ~ +1.0 범위로 제한
            return max(-1.0, min(1.0, sensitivity))
            
        except Exception as e:
            logger.warning(f"파라미터 민감도 계산 실패: {e}")
            return 0.0
    
    async def get_actual_return_from_sweet_spot_db(self, symbol: str) -> float:
        """Sweet Spot DB에서 실제 수익률 가져오기"""
        try:
            with open(self.sweet_spot_db_file, 'r', encoding='utf-8') as f:
                db = json.load(f)
            
            picks = db.get('picks', [])
            for pick in picks:
                if pick.get('symbol') == symbol:
                    return pick.get('current_return', 0.0)
            
            logger.debug(f"{symbol} Sweet Spot DB에서 수익률 정보 없음")
            return 0.0
            
        except Exception as e:
            logger.warning(f"Sweet Spot DB 수익률 조회 실패: {e}")
            return 0.0
    
    def calculate_prediction_confidence(self, stock: Dict, ml_params: Dict) -> float:
        """예측 신뢰도 계산"""
        try:
            confidence = 0.5  # 기본값
            
            # 점수 기반 신뢰도
            score = stock.get('selection_score', 0)
            if score > 70:
                confidence += 0.3
            elif score > 50:
                confidence += 0.2
            elif score < 30:
                confidence -= 0.2
            
            # Sweet Spot 위치 기반
            recovery_percent = stock.get('recovery_from_low_percent', 50)
            if 30 <= recovery_percent <= 80:
                confidence += 0.15
            
            # 거래량 신호 기반
            if stock.get('volume_surge', False) and stock.get('volume_trend', False):
                confidence += 0.1
            
            # ML/AI 과거 성과 반영
            confidence = confidence * self.ml_confidence + 0.1 * self.ai_confidence
            
            return max(0.1, min(0.95, confidence))
            
        except:
            return 0.5
    
    def analyze_key_factors(self, stock: Dict, ml_params: Dict) -> Dict[str, float]:
        """핵심 요인 분석"""
        try:
            factors = {}
            
            # 패턴 신호
            recovery_percent = stock.get('recovery_from_low_percent', 50)
            if 30 <= recovery_percent <= 80:
                factors['sweet_spot_signal'] = 0.25
            
            # 거래량 신호
            volume_weight = 0.0
            if stock.get('volume_surge', False):
                volume_weight += 0.15
            if stock.get('volume_trend', False):
                volume_weight += 0.10
            if volume_weight > 0:
                factors['volume_signal'] = volume_weight
            
            # 딥테크 카테고리 부스트
            tech_category = stock.get('tech_category', '')
            if tech_category in ['ai_computing', 'quantum_tech', 'bio_health_tech']:
                factors['tech_category_boost'] = 0.20
            
            # 타이밍 점수
            timing_score = stock.get('timing_score', 0)
            if timing_score > 0.7:
                factors['timing_signal'] = timing_score * 0.15
            
            return factors
            
        except:
            return {'unknown_factor': 0.1}
    
    async def validate_predictions(self, weeks_back: int = 1) -> Optional[ValidationResult]:
        """예측 검증 - 실제 수익률과 비교"""
        try:
            logger.info(f"🔍 {weeks_back}주 전 예측 검증 시작")
            
            # 검증할 예측 찾기 (weeks_back주 전)
            target_date = (datetime.now() - timedelta(weeks=weeks_back)).strftime('%Y-%m-%d')
            predictions_to_validate = [
                p for p in self.prediction_history 
                if p['prediction_date'] == target_date and p.get('actual_return') is None
            ]
            
            if not predictions_to_validate:
                logger.warning(f"{weeks_back}주 전 검증할 예측이 없습니다")
                return None
            
            logger.info(f"검증 대상: {len(predictions_to_validate)}개 예측")
            
            # 실제 수익률 계산
            validation_results = []
            for pred in predictions_to_validate:
                try:
                    actual_return = await self.calculate_actual_return(
                        pred['symbol'], pred['prediction_date']
                    )
                    
                    if actual_return is not None:
                        # 예측 기록 업데이트
                        pred['actual_return'] = actual_return
                        pred['validation_date'] = datetime.now().strftime('%Y-%m-%d')
                        pred['error'] = abs(actual_return - pred['next_week_expected_return'])
                        pred['direction_correct'] = (
                            (pred['next_week_expected_return'] > 0) == (actual_return > 0)
                        )
                        
                        validation_results.append(pred)
                        
                except Exception as e:
                    logger.warning(f"{pred['symbol']} 실제 수익률 계산 실패: {e}")
                    continue
            
            # 검증 결과 저장
            self.save_prediction_history()
            
            if not validation_results:
                logger.error("검증 완료된 예측이 없습니다")
                return None
            
            # 검증 메트릭 계산
            return self.calculate_validation_metrics(validation_results)
            
        except Exception as e:
            logger.error(f"예측 검증 실패: {e}")
            return None
    
    async def calculate_actual_return(self, symbol: str, prediction_date: str) -> Optional[float]:
        """실제 수익률 계산"""
        try:
            # 예측 날짜로부터 1주 후 실제 수익률 계산
            pred_date = datetime.strptime(prediction_date, '%Y-%m-%d')
            end_date = pred_date + timedelta(days=7)
            
            # yfinance로 데이터 가져오기
            stock = yf.Ticker(symbol)
            hist = stock.history(start=pred_date.strftime('%Y-%m-%d'), 
                               end=(end_date + timedelta(days=3)).strftime('%Y-%m-%d'))
            
            if len(hist) < 2:
                logger.warning(f"{symbol} 충분한 가격 데이터 없음")
                return None
            
            start_price = hist.iloc[0]['Close']
            end_price = hist.iloc[-1]['Close']
            
            actual_return = ((end_price - start_price) / start_price) * 100
            
            return actual_return
            
        except Exception as e:
            logger.warning(f"{symbol} 실제 수익률 계산 실패: {e}")
            return None
    
    def calculate_validation_metrics(self, validation_results: List[Dict]) -> ValidationResult:
        """검증 메트릭 계산"""
        try:
            # 기본 통계
            errors = [r['error'] for r in validation_results]
            predictions = [r['next_week_expected_return'] for r in validation_results]
            actuals = [r['actual_return'] for r in validation_results]
            directions = [r['direction_correct'] for r in validation_results]
            
            # MAE (Mean Absolute Error)
            mae = np.mean(errors)
            
            # RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.mean([(a - p) ** 2 for p, a in zip(predictions, actuals)]))
            
            # 방향성 정확도
            direction_accuracy = np.mean(directions)
            
            # ML 전체 정확도 (MAE 기반, 낮을수록 좋음)
            ml_accuracy = max(0.0, 1.0 - mae / 20.0)  # 20% 오차시 0점
            
            # 최고/최악 예측
            sorted_results = sorted(validation_results, key=lambda x: x['error'])
            best_predictions = sorted_results[:3]
            worst_predictions = sorted_results[-3:]
            
            # 섹터별 정확도 (Sweet Spot DB에서 가져올 필요)
            sector_accuracy = self.calculate_sector_accuracy(validation_results)
            
            return ValidationResult(
                total_predictions=len(validation_results),
                mae=mae,
                rmse=rmse,
                direction_accuracy=direction_accuracy,
                ml_accuracy=ml_accuracy,
                best_predictions=best_predictions,
                worst_predictions=worst_predictions,
                sector_accuracy=sector_accuracy
            )
            
        except Exception as e:
            logger.error(f"검증 메트릭 계산 실패: {e}")
            return None
    
    def calculate_sector_accuracy(self, validation_results: List[Dict]) -> Dict[str, float]:
        """섹터별 정확도 계산"""
        try:
            # Sweet Spot DB에서 섹터 정보 가져오기
            sweet_spot_stocks = self.load_sweet_spot_stocks()
            sector_map = {s['symbol']: s.get('tech_category', 'unknown') for s in sweet_spot_stocks}
            
            sector_errors = {}
            for result in validation_results:
                sector = sector_map.get(result['symbol'], 'unknown')
                if sector not in sector_errors:
                    sector_errors[sector] = []
                sector_errors[sector].append(result['error'])
            
            sector_accuracy = {}
            for sector, errors in sector_errors.items():
                mae = np.mean(errors)
                accuracy = max(0.0, 1.0 - mae / 20.0)
                sector_accuracy[sector] = accuracy
            
            return sector_accuracy
            
        except Exception as e:
            logger.warning(f"섹터별 정확도 계산 실패: {e}")
            return {}
    
    def update_ml_ai_confidence(self, validation_result: ValidationResult):
        """ML/AI 신뢰도 업데이트"""
        try:
            # ML 성과 기반 신뢰도 조정
            if validation_result.ml_accuracy > 0.8:
                # ML이 매우 정확하면 ML 신뢰도 증가
                self.ml_confidence = min(0.9, self.ml_confidence + 0.05)
                self.ai_confidence = 1.0 - self.ml_confidence
            elif validation_result.ml_accuracy < 0.5:
                # ML이 부정확하면 AI 신뢰도 증가
                self.ai_confidence = min(0.6, self.ai_confidence + 0.1)
                self.ml_confidence = 1.0 - self.ai_confidence
            
            logger.info(f"📊 신뢰도 업데이트: ML={self.ml_confidence:.2f}, AI={self.ai_confidence:.2f}")
            
            # 신뢰도 저장
            confidence_data = {
                'ml_confidence': self.ml_confidence,
                'ai_confidence': self.ai_confidence,
                'last_updated': datetime.now().isoformat(),
                'last_ml_accuracy': validation_result.ml_accuracy
            }
            
            with open('ml_ai_confidence.json', 'w', encoding='utf-8') as f:
                json.dump(confidence_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"신뢰도 업데이트 실패: {e}")
    
    def load_sweet_spot_stocks(self) -> List[Dict]:
        """Sweet Spot 종목 로드"""
        try:
            with open(self.sweet_spot_db_file, 'r', encoding='utf-8') as f:
                db = json.load(f)
                return db.get('picks', [])
        except Exception as e:
            logger.error(f"Sweet Spot DB 로드 실패: {e}")
            return []
    
    def load_ml_parameters(self) -> Dict:
        """ML 파라미터 로드"""
        try:
            with open('ml_parameters.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"ML 파라미터 로드 실패: {e}")
            return {}

# 독립 실행 테스트
async def main():
    """테스트 실행"""
    prediction_system = PredictionSystem()
    
    logger.info("=== ML+AI 예측 시스템 테스트 ===")
    
    # 1. 예측 생성 테스트
    logger.info("1️⃣ 예측 생성 테스트")
    predictions = await prediction_system.generate_weekly_predictions(5)
    
    if predictions:
        logger.info(f"✅ {len(predictions)}개 예측 생성 성공")
        for pred in predictions[:3]:
            logger.info(f"   {pred.symbol}: {pred.next_week_expected_return:+.1f}% (신뢰도: {pred.prediction_confidence:.2f})")
    
    # 2. 예측 검증 테스트 (과거 데이터가 있는 경우)
    logger.info("2️⃣ 예측 검증 테스트")
    validation = await prediction_system.validate_predictions(weeks_back=1)
    
    if validation:
        logger.info(f"✅ 검증 완료: MAE={validation.mae:.2f}%, 방향정확도={validation.direction_accuracy:.1%}")
        logger.info(f"   ML 정확도: {validation.ml_accuracy:.1%}")
    else:
        logger.info("검증할 과거 예측이 없습니다")

if __name__ == "__main__":
    asyncio.run(main())