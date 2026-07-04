#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Prediction Validator - ML 예측 검증 자동화 시스템

실시간으로 ML 예측의 정확도를 추적하고 검증하는 시스템
Sweet Spot 데이터베이스와 연동하여 종목별 성과를 자동 추적

API 정보:
- FMP API Rate Limit: 300 calls/분 (일일 43,200 calls 가능)
"""

import json
import logging
import asyncio
import os
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationMetric:
    """검증 지표"""
    metric_name: str
    current_value: float
    target_value: float
    accuracy: float
    trend: str  # 'improving', 'stable', 'declining'
    confidence: float

@dataclass
class StockPerformanceRecord:
    """개별 종목 성과 기록"""
    symbol: str
    prediction_date: str
    predicted_score: float
    predicted_category: str
    actual_performance_1w: Optional[float] = None
    actual_performance_1m: Optional[float] = None
    actual_performance_3m: Optional[float] = None
    volume_change_1w: Optional[float] = None
    sector_relative_performance: Optional[float] = None
    validation_status: str = 'pending'  # 'pending', 'validated', 'failed'
    
class FMPDataProvider:
    """FMP API 데이터 제공자 (yfinance 대체용)"""
    
    def __init__(self):
        self.api_key = os.getenv('FMP_API_KEY', '')
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.rate_limiter = {'last_call': 0, 'min_interval': 0.2}  # 300 calls/min = 0.2초 간격
        
    async def _make_request(self, url: str) -> Dict:
        """Rate limit 적용된 API 요청"""
        # Rate limiting
        now = time.time()
        elapsed = now - self.rate_limiter['last_call']
        if elapsed < self.rate_limiter['min_interval']:
            await asyncio.sleep(self.rate_limiter['min_interval'] - elapsed)
        
        self.rate_limiter['last_call'] = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"FMP API 오류: {response.status} - {url}")
                        return {}
        except Exception as e:
            logger.error(f"FMP API 요청 실패: {e}")
            return {}
    
    async def get_historical_prices(self, symbol: str, days: int = 30) -> List[Dict]:
        """종목 과거 가격 데이터 조회"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days + 5)).strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={self.api_key}"
        data = await self._make_request(url)
        
        if 'historical' in data:
            return data['historical']
        return []
    
    async def get_stock_performance(self, symbol: str, start_date: datetime, periods: List[int] = [7, 30, 90]) -> Dict[str, float]:
        """종목별 실제 성과 계산"""
        try:
            historical_data = await self.get_historical_prices(symbol, max(periods) + 10)
            
            if not historical_data:
                return {}
                
            # 날짜 기준으로 정렬
            historical_data.sort(key=lambda x: x['date'])
            
            # 시작 날짜에 가장 가까운 가격 찾기
            start_price = None
            start_volume = None
            
            for data_point in historical_data:
                data_date = datetime.strptime(data_point['date'], '%Y-%m-%d')
                if data_date.date() >= start_date.date():
                    start_price = data_point['close']
                    start_volume = data_point['volume']
                    break
            
            if start_price is None:
                return {}
            
            performance = {}
            
            for period in periods:
                target_date = start_date + timedelta(days=period)
                
                # 해당 기간에 가장 가까운 거래일 찾기
                closest_data = None
                min_diff = timedelta(days=float('inf'))
                
                for data_point in historical_data:
                    data_date = datetime.strptime(data_point['date'], '%Y-%m-%d')
                    diff = abs(data_date.date() - target_date.date())
                    
                    if diff <= min_diff and data_date >= start_date:
                        min_diff = diff
                        closest_data = data_point
                
                if closest_data is not None and min_diff <= timedelta(days=3):
                    end_price = closest_data['close']
                    performance[f'{period}d'] = ((end_price - start_price) / start_price) * 100
                    
                    # 거래량 변화 계산 (1주일만)
                    if period == 7 and start_volume and start_volume > 0:
                        end_volume = closest_data['volume']
                        performance['volume_change_7d'] = ((end_volume - start_volume) / start_volume) * 100
            
            return performance
            
        except Exception as e:
            logger.warning(f"FMP 종목 성과 계산 실패 ({symbol}): {e}")
            return {}
    
    async def get_etf_performance(self, etf_symbol: str, start_date: datetime, days: int = 7) -> float:
        """ETF 성과 조회"""
        try:
            historical_data = await self.get_historical_prices(etf_symbol, days + 5)
            
            if len(historical_data) < 2:
                return 0.0
            
            # 날짜 기준 정렬
            historical_data.sort(key=lambda x: x['date'])
            
            # 시작일과 종료일에 가장 가까운 데이터 찾기
            start_price = None
            end_price = None
            
            for data_point in historical_data:
                data_date = datetime.strptime(data_point['date'], '%Y-%m-%d')
                
                if start_price is None and data_date.date() >= start_date.date():
                    start_price = data_point['close']
                
                if data_date.date() >= (start_date + timedelta(days=days)).date():
                    end_price = data_point['close']
                    break
            
            # 최신 데이터를 종료가격으로 사용 (정확한 날짜 매칭 안될 경우)
            if start_price and not end_price:
                end_price = historical_data[-1]['close']
            
            if start_price and end_price:
                return ((end_price - start_price) / start_price) * 100
                
        except Exception as e:
            logger.warning(f"FMP ETF 성과 조회 실패 ({etf_symbol}): {e}")
            
        return 0.0


class PredictionValidator:
    """🎯 ML 예측 검증 자동화 시스템"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.sweet_spot_db = self.data_dir / "sweet_spot_database.json"
        self.validation_history = self.data_dir / "validation_history.json"
        self.performance_metrics = self.data_dir / "performance_metrics.json"
        self.screening_data_dir = self.data_dir / "screening_data"
        self.fmp_provider = FMPDataProvider()
        
        # 검증 기준 설정
        self.validation_thresholds = {
            'prediction_accuracy_1w': 0.65,    # 1주일 예측 정확도 65% 목표
            'prediction_accuracy_1m': 0.60,    # 1개월 예측 정확도 60% 목표  
            'sector_outperformance': 0.55,     # 섹터 대비 55% 승률 목표
            'sweet_spot_success_rate': 0.70,   # Sweet Spot 성공률 70% 목표
            'volume_prediction_accuracy': 0.58, # 거래량 예측 정확도 58% 목표
        }
        
        # 8개 최적화 섹터 매핑
        self.sector_etf_mapping = {
            'ai_computing': 'QQQ',       # Tech-heavy
            'quantum_tech': 'XLK',       # Technology 
            'mobility_tech': 'XLI',      # Industrial
            'semiconductor': 'SMH',      # Semiconductor ETF
            'bio_health_tech': 'XLV',    # Healthcare
            'energy_materials': 'XLE',   # Energy
            'security_fintech': 'XLF',   # Financial
            'emerging_tech': 'QQQ'       # General Tech
        }
        
    def load_sweet_spot_database(self) -> Dict:
        """Sweet Spot 데이터베이스 로드"""
        try:
            if self.sweet_spot_db.exists():
                with open(self.sweet_spot_db, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Sweet Spot DB 로드 실패: {e}")
            return {}
    
    def save_validation_history(self, records: List[StockPerformanceRecord]):
        """검증 히스토리 저장"""
        try:
            data = [asdict(record) for record in records]
            with open(self.validation_history, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"검증 히스토리 저장 실패: {e}")
    
    def load_validation_history(self) -> List[StockPerformanceRecord]:
        """검증 히스토리 로드"""
        try:
            if self.validation_history.exists():
                with open(self.validation_history, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [StockPerformanceRecord(**record) for record in data]
            return []
        except Exception as e:
            logger.error(f"검증 히스토리 로드 실패: {e}")
            return []
    
    async def fetch_stock_performance(self, symbol: str, start_date: datetime, 
                                    periods: List[int] = [7, 30, 90]) -> Dict[str, float]:
        """종목별 실제 성과 조회 (FMP API 사용)"""
        return await self.fmp_provider.get_stock_performance(symbol, start_date, periods)
    
    async def fetch_sector_performance(self, sector: str, start_date: datetime, days: int = 7) -> float:
        """섹터 ETF 성과 조회 (FMP API 사용)"""
        etf_symbol = self.sector_etf_mapping.get(sector, 'QQQ')
        return await self.fmp_provider.get_etf_performance(etf_symbol, start_date, days)
    
    async def validate_recent_predictions(self, days_back: int = 7) -> Dict[str, Any]:
        """최근 예측들을 검증"""
        logger.info(f"📊 최근 {days_back}일간 예측 검증 중...")
        
        # Sweet Spot 데이터베이스에서 검증 대상 추출
        sweet_spot_db = self.load_sweet_spot_database()
        validation_records = []
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for symbol, pick_data in sweet_spot_db.items():
            if not isinstance(pick_data, dict):
                continue
                
            pick_date_str = pick_data.get('first_picked_date', '')
            if not pick_date_str:
                continue
                
            try:
                pick_date = datetime.fromisoformat(pick_date_str.replace('KST', '').strip())
                
                # 검증 대상 기간 내의 종목들
                if pick_date >= cutoff_date:
                    record = StockPerformanceRecord(
                        symbol=symbol,
                        prediction_date=pick_date_str,
                        predicted_score=pick_data.get('initial_score', 0.0),
                        predicted_category=pick_data.get('tech_category', 'unknown')
                    )
                    
                    # 실제 성과 조회
                    performance = await self.fetch_stock_performance(pick_date, [7, 30])
                    
                    if performance:
                        record.actual_performance_1w = performance.get('7d')
                        record.actual_performance_1m = performance.get('30d') 
                        record.volume_change_1w = performance.get('volume_change_7d')
                        
                        # 섹터 상대 성과 계산
                        sector_perf = await self.fetch_sector_performance(
                            record.predicted_category, pick_date, 7
                        )
                        if record.actual_performance_1w is not None:
                            record.sector_relative_performance = record.actual_performance_1w - sector_perf
                            
                        record.validation_status = 'validated'
                    else:
                        record.validation_status = 'failed'
                    
                    validation_records.append(record)
                    
            except Exception as e:
                logger.warning(f"종목 검증 실패 ({symbol}): {e}")
                continue
        
        # 검증 결과 분석
        validation_results = self.analyze_validation_results(validation_records)
        
        # 검증 히스토리 저장
        self.save_validation_history(validation_records)
        
        logger.info(f"✅ 검증 완료: {len(validation_records)}개 종목, 전체 정확도: {validation_results['overall_accuracy']:.2%}")
        
        return {
            'validation_records': validation_records,
            'analysis': validation_results,
            'validated_count': len([r for r in validation_records if r.validation_status == 'validated']),
            'total_validated': validation_results.get('total_validated', 0)
        }
    
    def analyze_validation_results(self, records: List[StockPerformanceRecord]) -> Dict[str, Any]:
        """검증 결과 분석"""
        
        if not records:
            return {
                'overall_accuracy': 0.0, 
                'metrics': [],
                'total_validated': 0,
                'sector_performance': {},
                'summary': {
                    '1w_accuracy': 0.0,
                    'sector_outperform_rate': 0.0,
                    'sweet_spot_success_rate': 0.0,
                    'volume_signal_accuracy': 0.0
                }
            }
        
        validated_records = [r for r in records if r.validation_status == 'validated' and r.actual_performance_1w is not None]
        
        if not validated_records:
            return {
                'overall_accuracy': 0.0, 
                'metrics': [],
                'total_validated': 0,
                'sector_performance': {},
                'summary': {
                    '1w_accuracy': 0.0,
                    'sector_outperform_rate': 0.0,
                    'sweet_spot_success_rate': 0.0,
                    'volume_signal_accuracy': 0.0
                }
            }
        
        # 1주일 예측 정확도 (방향성 기준)
        correct_predictions_1w = 0
        correct_predictions_1m = 0
        outperformed_sector = 0
        positive_volume_signals = 0
        
        sweet_spot_successes = 0
        sweet_spot_total = 0
        
        sector_performance = {}
        
        for record in validated_records:
            # 1주일 방향성 정확도
            predicted_positive = record.predicted_score > 50  # 50점 이상이면 긍정적 예측
            actual_positive = record.actual_performance_1w > 0
            
            if predicted_positive == actual_positive:
                correct_predictions_1w += 1
            
            # 1개월 방향성 정확도 (데이터 있는 경우)
            if record.actual_performance_1m is not None:
                actual_positive_1m = record.actual_performance_1m > 0
                if predicted_positive == actual_positive_1m:
                    correct_predictions_1m += 1
            
            # 섹터 대비 성과
            if record.sector_relative_performance is not None and record.sector_relative_performance > 0:
                outperformed_sector += 1
            
            # 거래량 신호 정확도
            if record.volume_change_1w is not None and record.volume_change_1w > 10:  # 10% 이상 증가
                positive_volume_signals += 1
            
            # Sweet Spot 성공률 (15% 이상 수익)
            sweet_spot_total += 1
            if record.actual_performance_1w > 15:  # 15% 이상 수익
                sweet_spot_successes += 1
                
            # 섹터별 성과 집계
            sector = record.predicted_category
            if sector not in sector_performance:
                sector_performance[sector] = {'count': 0, 'sum': 0.0}
            sector_performance[sector]['count'] += 1
            sector_performance[sector]['sum'] += record.actual_performance_1w
        
        # 지표 계산
        total_validated = len(validated_records)
        
        metrics = [
            ValidationMetric(
                metric_name='prediction_accuracy_1w',
                current_value=correct_predictions_1w / total_validated,
                target_value=self.validation_thresholds['prediction_accuracy_1w'],
                accuracy=min(1.0, (correct_predictions_1w / total_validated) / self.validation_thresholds['prediction_accuracy_1w']),
                trend=self.calculate_trend('prediction_accuracy_1w'),
                confidence=0.8 if total_validated >= 10 else 0.6
            ),
            ValidationMetric(
                metric_name='sector_outperformance',
                current_value=outperformed_sector / total_validated,
                target_value=self.validation_thresholds['sector_outperformance'],
                accuracy=min(1.0, (outperformed_sector / total_validated) / self.validation_thresholds['sector_outperformance']),
                trend=self.calculate_trend('sector_outperformance'),
                confidence=0.7
            ),
            ValidationMetric(
                metric_name='sweet_spot_success_rate',
                current_value=sweet_spot_successes / sweet_spot_total if sweet_spot_total > 0 else 0,
                target_value=self.validation_thresholds['sweet_spot_success_rate'],
                accuracy=min(1.0, (sweet_spot_successes / sweet_spot_total) / self.validation_thresholds['sweet_spot_success_rate']) if sweet_spot_total > 0 else 0,
                trend=self.calculate_trend('sweet_spot_success_rate'),
                confidence=0.9 if sweet_spot_total >= 5 else 0.5
            ),
            ValidationMetric(
                metric_name='volume_prediction_accuracy',
                current_value=positive_volume_signals / total_validated,
                target_value=self.validation_thresholds['volume_prediction_accuracy'],
                accuracy=min(1.0, (positive_volume_signals / total_validated) / self.validation_thresholds['volume_prediction_accuracy']),
                trend=self.calculate_trend('volume_prediction_accuracy'),
                confidence=0.6
            )
        ]
        
        # 전체 정확도 계산 (가중평균)
        overall_accuracy = sum(m.accuracy * m.confidence for m in metrics) / sum(m.confidence for m in metrics)
        
        # 섹터별 평균 성과
        sector_avg_performance = {}
        for sector, data in sector_performance.items():
            sector_avg_performance[sector] = data['sum'] / data['count'] if data['count'] > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'metrics': metrics,
            'total_validated': total_validated,
            'sector_performance': sector_avg_performance,
            'summary': {
                '1w_accuracy': correct_predictions_1w / total_validated,
                'sector_outperform_rate': outperformed_sector / total_validated,
                'sweet_spot_success_rate': sweet_spot_successes / sweet_spot_total if sweet_spot_total > 0 else 0,
                'volume_signal_accuracy': positive_volume_signals / total_validated
            }
        }
    
    def calculate_trend(self, metric_name: str) -> str:
        """지표 트렌드 계산 (이전 데이터와 비교)"""
        # 간단한 트렌드 계산 (실제로는 과거 데이터와 비교)
        try:
            if self.performance_metrics.exists():
                with open(self.performance_metrics, 'r', encoding='utf-8') as f:
                    historical_data = json.load(f)
                    
                recent_values = historical_data.get(metric_name, [])
                if len(recent_values) >= 2:
                    current = recent_values[-1]
                    previous = recent_values[-2]
                    
                    if current > previous * 1.05:  # 5% 이상 향상
                        return 'improving'
                    elif current < previous * 0.95:  # 5% 이상 하락
                        return 'declining'
                    else:
                        return 'stable'
            
            return 'stable'
        except Exception:
            return 'stable'
    
    def save_performance_metrics(self, analysis: Dict[str, Any]):
        """성과 지표 히스토리 저장"""
        try:
            historical_data = {}
            
            if self.performance_metrics.exists():
                with open(self.performance_metrics, 'r', encoding='utf-8') as f:
                    historical_data = json.load(f)
            
            # 현재 시점의 지표들 저장
            current_timestamp = datetime.now().isoformat()
            
            for metric in analysis['metrics']:
                metric_name = metric.metric_name
                
                if metric_name not in historical_data:
                    historical_data[metric_name] = []
                
                historical_data[metric_name].append({
                    'timestamp': current_timestamp,
                    'value': metric.current_value,
                    'accuracy': metric.accuracy,
                    'confidence': metric.confidence
                })
                
                # 최근 30개 데이터만 보관 (메모리 절약)
                if len(historical_data[metric_name]) > 30:
                    historical_data[metric_name] = historical_data[metric_name][-30:]
            
            with open(self.performance_metrics, 'w', encoding='utf-8') as f:
                json.dump(historical_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"성과 지표 저장 실패: {e}")
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """검증 리포트 생성"""
        analysis = validation_results['analysis']
        
        report_lines = [
            f"📊 **ML 예측 검증 리포트** ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
            "=" * 50,
            "",
            f"🎯 **전체 성과**: {analysis['overall_accuracy']:.1%}",
            f"📈 **검증 종목 수**: {analysis['total_validated']}개",
            "",
            "🔍 **세부 지표**:"
        ]
        
        for metric in analysis['metrics']:
            status_emoji = "✅" if metric.accuracy > 0.8 else "⚡" if metric.accuracy > 0.6 else "❌"
            trend_emoji = {"improving": "📈", "stable": "➡️", "declining": "📉"}[metric.trend]
            
            report_lines.extend([
                f"  {status_emoji} **{metric.metric_name}**:",
                f"    현재: {metric.current_value:.1%} | 목표: {metric.target_value:.1%} | 달성도: {metric.accuracy:.1%}",
                f"    트렌드: {trend_emoji} {metric.trend} | 신뢰도: {metric.confidence:.1%}",
                ""
            ])
        
        # 섹터별 성과
        if analysis.get('sector_performance'):
            report_lines.extend([
                "🏗️ **8개 섹터별 성과**:"
            ])
            
            for sector, avg_perf in analysis['sector_performance'].items():
                perf_emoji = "🔥" if avg_perf > 10 else "⚡" if avg_perf > 0 else "❄️"
                report_lines.append(f"  {perf_emoji} {sector}: {avg_perf:+.1f}%")
        
        # 요약 및 추천사항
        report_lines.extend([
            "",
            "💡 **AI 분석 결과**:",
        ])
        
        if analysis['overall_accuracy'] > 0.8:
            report_lines.append("✅ 현재 ML 모델이 매우 효과적으로 작동하고 있습니다.")
        elif analysis['overall_accuracy'] > 0.6:
            report_lines.append("⚡ ML 모델이 적절히 작동하나, 개선 여지가 있습니다.")
        else:
            report_lines.append("❌ ML 모델의 성과가 기대에 미치지 못합니다. 가중치 조정 필요.")
        
        # 구체적 개선사항
        underperforming_metrics = [m for m in analysis['metrics'] if m.accuracy < 0.7]
        if underperforming_metrics:
            report_lines.extend([
                "",
                "🔧 **개선 필요 지표**:"
            ])
            for metric in underperforming_metrics[:3]:  # 상위 3개만
                report_lines.append(f"  • {metric.metric_name}: {metric.current_value:.1%} → {metric.target_value:.1%} 목표")
        
        return "\n".join(report_lines)
    
    async def daily_validation_cycle(self) -> Dict[str, Any]:
        """📅 일일 검증 사이클"""
        logger.info("🎯 일일 예측 검증 사이클 시작...")
        
        try:
            # 최근 7일간 예측 검증
            validation_results = await self.validate_recent_predictions(days_back=7)
            
            # 성과 지표 저장
            self.save_performance_metrics(validation_results['analysis'])
            
            # 검증 리포트 생성
            report = self.generate_validation_report(validation_results)
            
            # 리포트 파일 저장
            report_file = self.data_dir / f"validation_report_{datetime.now().strftime('%Y%m%d')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            result = {
                'success': True,
                'validated_count': validation_results.get('validated_count', 0),
                'overall_accuracy': validation_results['analysis']['overall_accuracy'],
                'report': report,
                'report_file': str(report_file)
            }
            
            logger.info(f"✅ 일일 검증 완료: {result['validated_count']}개 종목, 정확도: {result['overall_accuracy']:.2%}")
            
        except Exception as e:
            import traceback
            logger.error(f"일일 검증 사이클 실패: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")
            result = {'success': False, 'error': str(e)}
        
        return result


async def main():
    """테스트 및 데모 실행"""
    logger.info("🎯 Prediction Validator 데모 시작")
    
    validator = PredictionValidator()
    
    # 일일 검증 사이클 테스트
    result = await validator.daily_validation_cycle()
    
    if result['success']:
        print("✅ 검증 성공!")
        print(f"📊 검증 종목 수: {result['validated_count']}")
        print(f"📈 전체 정확도: {result['overall_accuracy']:.2%}")
        print(f"📄 리포트 생성: {result['report_file']}")
    else:
        print(f"❌ 검증 실패: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())