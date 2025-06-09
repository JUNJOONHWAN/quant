#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced 통합 시스템 (메인 분석 엔진)
3단 구조: 딥리서치 섹터 + Gemini Flash + 갭필터 부스터
"""
import asyncio
import json
import time
import requests
import aiohttp
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging
import traceback

# 로컬 모듈 임포트
from .sector_cache import SectorWeightCache
from .sector_mapper import SectorMappingEngine  
from .gap_booster import GeminiGapFilterBooster
from .pattern_booster import GeminiPatternRecognitionBooster
from ..core.api_manager import UnifiedAPIManager
from ..core.token_manager import KisTokenManager
from ..core.data_validator import DataValidationManager
from ..models.perplexity_models import PerplexityModelManager

logger = logging.getLogger(__name__)

class Enhanced182ThreeTierSystem:
    """3단 구조 Enhanced 시스템: 딥리서치 섹터 + Gemini Flash + 갭필터 부스터"""

    def __init__(self):
        # 기본 API 관리자들
        self.api_manager = UnifiedAPIManager()
        self.model_manager = PerplexityModelManager()
        self.kis_token_manager = None
        self.kis_base_url = "https://openapivts.koreainvestment.com:29443"

        # 한투 자동 초기화
        self._auto_initialize_kis()

        # 🆕 3단 구조 핵심 컴포넌트들
        self.sector_cache = SectorWeightCache(cache_hours=48)
        self.sector_mapper = SectorMappingEngine(self.api_manager)
        self.gap_booster = GeminiGapFilterBooster(self.api_manager)

        # 🆕 패턴 인식 부스터 추가
        self.pattern_booster = GeminiPatternRecognitionBooster(self.api_manager)
        self.pattern_enabled = True  # ✅ 패턴 인식 on/off 스위치

        # 하이브리드 전략
        self.use_hybrid_gemini = True
        logger.info("🔥 하이브리드 Gemini: N점수(Flash) + 갭필터(Flash-Lite)")
        logger.info("🔍 패턴 인식 부스터 초기화 완료 (Gemini 2.5 Flash)")

        # ✅ 캐시 없으면 즉시 갱신!
        if not self.sector_cache.is_valid():
            logger.info("🔄 섹터 캐시 없음 - 즉시 갱신 시작...")
            self._initialize_sector_cache()
        else:
            logger.info("✅ 섹터 캐시 유효함")
            
        # 데이터 검증 매니저 추가
        self.data_validator = DataValidationManager()
        self.fallback_enabled = True  # 폴백 데이터 사용 여부

        logger.info("🚀 Enhanced 1.8.2 3단 구조 시스템 초기화")
        logger.info("📊 Tier 1: 딥리서치 섹터 가중치 (48시간 캐싱)")
        logger.info("🔥 Tier 2: Gemini Flash N점수 (실시간)")
        logger.info("🎯 Tier 3: 갭필터 부스터 Lite (시너지/리스크)")

    def _auto_initialize_kis(self):
        """한투 API 자동 초기화"""
        try:
            kis_config = self.api_manager.get_kis_config()
            if kis_config['app_key'] and kis_config['app_secret']:
                self.kis_token_manager = KisTokenManager(
                    kis_config['app_key'], 
                    kis_config['app_secret']
                )
                logger.info("✅ 한투 API 초기화 완료")
            else:
                logger.warning("⚠️ 한투 API 키 없음")
        except Exception as e:
            logger.error(f"❌ 한투 API 초기화 실패: {e}")

    def _initialize_sector_cache(self):
        """섹터 캐시 초기화 (동기 실행)"""
        try:
            import asyncio
            if asyncio.get_running_loop():
                # 이미 이벤트 루프가 있으면 태스크로 실행
                asyncio.create_task(self._async_initialize_sector_cache())
            else:
                # 새 이벤트 루프 실행
                asyncio.run(self._async_initialize_sector_cache())
        except RuntimeError:
            # 동기적으로 처리
            logger.info("🔄 섹터 캐시 동기 초기화...")
            pass

    async def _async_initialize_sector_cache(self):
        """비동기 섹터 캐시 초기화"""
        try:
            logger.info("🔄 섹터 가중치 캐시 갱신 중...")
            
            # 기본 섹터들
            sectors = [
                "Technology", "Healthcare", "Financials", "ConsumerCyclical", 
                "Industrials", "Energy", "Utilities", "RealEstate"
            ]
            
            sector_weights = {}
            for sector in sectors:
                try:
                    weights = await self._calculate_sector_weights_deep_research(sector)
                    sector_weights[sector] = weights
                    logger.info(f"   ✅ {sector}: V{weights['V']:.2f} T{weights['T']:.2f} N{weights['N']:.2f} F{weights['F']:.2f}")
                    await asyncio.sleep(0.5)  # API 레이트 리밋
                except Exception as e:
                    logger.warning(f"   ⚠️ {sector} 가중치 계산 실패: {e}")
                    sector_weights[sector] = {'V': 0.30, 'T': 0.30, 'N': 0.25, 'F': 0.15}
            
            # 캐시 저장
            self.sector_cache.save_cache(sector_weights)
            logger.info("✅ 섹터 가중치 캐시 갱신 완료")
            
        except Exception as e:
            logger.error(f"❌ 섹터 캐시 초기화 실패: {e}")

    async def _calculate_sector_weights_deep_research(self, sector: str) -> dict:
        """딥리서치로 섹터별 VTNF 가중치 계산"""
        try:
            perplexity_config = self.api_manager.get_perplexity_config()
            api_key = perplexity_config['api_key']
            
            if not api_key or api_key == 'PPL':
                return {'V': 0.30, 'T': 0.30, 'N': 0.25, 'F': 0.15}
            
            query = f"""
{sector} 섹터 투자 전략에서 VTNF 요소별 최적 가중치를 분석해주세요.

**VTNF 요소 설명:**
- V (Value): 재무지표, PER, 성장률, 수익성
- T (Technical): 기술적 분석, 차트 패턴, 모멘텀  
- N (News): 뉴스 감정, 시장 심리, 소셜 트렌드
- F (Flow): 자금 흐름, 기관 매매, 옵션 플로우

**분석 요청:**
1. {sector} 섹터의 특성상 어떤 요소가 가장 중요한가?
2. 시장 사이클에 따른 요소별 영향력은?
3. 현재 시장 환경에서의 최적 가중치는?

**응답 형식:**
V: [0.15-0.45 범위의 소수점 2자리]
T: [0.15-0.45 범위의 소수점 2자리]  
N: [0.10-0.35 범위의 소수점 2자리]
F: [0.10-0.25 범위의 소수점 2자리]

합계가 1.00이 되도록 정확히 계산해주세요.
"""

            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar-pro",
                "messages": [{"role": "user", "content": query}],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        
                        # 가중치 추출
                        weights = self._extract_sector_weights(content)
                        return weights
                    else:
                        logger.warning(f"Perplexity API 오류: {response.status}")
                        return {'V': 0.30, 'T': 0.30, 'N': 0.25, 'F': 0.15}
                        
        except Exception as e:
            logger.error(f"섹터 가중치 계산 실패: {e}")
            return {'V': 0.30, 'T': 0.30, 'N': 0.25, 'F': 0.15}

    def _extract_sector_weights(self, content: str) -> dict:
        """섹터 가중치 추출"""
        try:
            import re
            
            # V, T, N, F 가중치 추출
            v_match = re.search(r'V:\s*([0-9.]+)', content)
            t_match = re.search(r'T:\s*([0-9.]+)', content)  
            n_match = re.search(r'N:\s*([0-9.]+)', content)
            f_match = re.search(r'F:\s*([0-9.]+)', content)
            
            if all([v_match, t_match, n_match, f_match]):
                weights = {
                    'V': float(v_match.group(1)),
                    'T': float(t_match.group(1)),
                    'N': float(n_match.group(1)), 
                    'F': float(f_match.group(1))
                }
                
                # 합계 검증 및 정규화
                total = sum(weights.values())
                if 0.8 <= total <= 1.2:  # 허용 오차 범위
                    for key in weights:
                        weights[key] = round(weights[key] / total, 2)
                    return weights
            
            # 추출 실패 시 기본값
            return {'V': 0.30, 'T': 0.30, 'N': 0.25, 'F': 0.15}
            
        except Exception as e:
            logger.error(f"가중치 추출 실패: {e}")
            return {'V': 0.30, 'T': 0.30, 'N': 0.25, 'F': 0.15}

    async def analyze_symbol_comprehensive(self, symbol: str) -> dict:
        """종목 종합 분석 (3단 구조)"""
        try:
            logger.info(f"🎯 {symbol} 종합 분석 시작...")
            start_time = time.time()
            
            # 1단계: 섹터 매핑 및 가중치 조회
            sector = await self.sector_mapper.get_sector(symbol)
            sector_weights = self.sector_cache.get_sector_weight(sector)
            
            logger.info(f"📊 {symbol} 섹터: {sector}")
            logger.info(f"⚖️ 가중치: V{sector_weights['V']:.2f} T{sector_weights['T']:.2f} N{sector_weights['N']:.2f} F{sector_weights['F']:.2f}")
            
            # 2단계: VTNF 점수 병렬 계산
            vtnf_tasks = {
                'V': self._calculate_v_score_enhanced(symbol),
                'T': self._calculate_t_score_enhanced(symbol),
                'N': self._calculate_n_score_gemini_flash(symbol),
                'F': self._calculate_f_score_enhanced(symbol)
            }
            
            vtnf_results = await asyncio.gather(*vtnf_tasks.values(), return_exceptions=True)
            vtnf_scores = {}
            
            for i, (component, task) in enumerate(vtnf_tasks.items()):
                result = vtnf_results[i]
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ {component} 점수 계산 실패: {result}")
                    vtnf_scores[component] = self._get_fallback_score(component)
                else:
                    vtnf_scores[component] = result
            
            # 데이터 검증
            validation_results = {}
            for component, score_data in vtnf_scores.items():
                validation = self.data_validator.validate_component_data(
                    f"{component.lower()}_score", 
                    score_data if isinstance(score_data, dict) else {'score': score_data}
                )
                validation_results[component] = validation
            
            # 3단계: 갭필터 부스터 (선택적)
            gap_analysis = {}
            if hasattr(self, 'gap_booster'):
                try:
                    # 점수만 추출
                    simple_scores = {k: v.get('score', 6.0) if isinstance(v, dict) else v 
                                   for k, v in vtnf_scores.items()}
                    gap_analysis = await self.gap_booster.calculate_vtnf_gap_boost(symbol, simple_scores)
                except Exception as e:
                    logger.warning(f"⚠️ {symbol} 갭 부스터 실패: {e}")
            
            # 4단계: 패턴 인식 (선택적)
            pattern_analysis = {}
            if self.pattern_enabled and hasattr(self, 'pattern_booster'):
                try:
                    pattern_analysis = await self.pattern_booster.analyze_patterns_15min(symbol)
                except Exception as e:
                    logger.warning(f"⚠️ {symbol} 패턴 인식 실패: {e}")
            
            # 5단계: 종합 점수 계산
            final_scores = self._calculate_weighted_scores(vtnf_scores, sector_weights)
            overall_score = self._calculate_overall_score(final_scores, gap_analysis, pattern_analysis)
            
            # 6단계: 투자 제안 생성
            investment_recommendation = self._generate_investment_recommendation(
                symbol, sector, final_scores, overall_score, gap_analysis, pattern_analysis
            )
            
            elapsed_time = time.time() - start_time
            
            result = {
                'symbol': symbol,
                'sector': sector,
                'sector_weights': sector_weights,
                'vtnf_scores': vtnf_scores,
                'final_scores': final_scores,
                'overall_score': overall_score,
                'gap_analysis': gap_analysis,
                'pattern_analysis': pattern_analysis,
                'validation_results': validation_results,
                'investment_recommendation': investment_recommendation,
                'analysis_metadata': {
                    'analysis_time': elapsed_time,
                    'timestamp': datetime.now().isoformat(),
                    'data_sources': {
                        'sector_mapping': 'yfinance + AI',
                        'vtnf_calculation': 'KIS + Gemini + yfinance',
                        'gap_analysis': 'Gemini Flash',
                        'pattern_recognition': 'Gemini 2.5 Flash' if self.pattern_enabled else 'disabled'
                    },
                    'cache_status': {
                        'sector_cache': 'hit' if self.sector_cache.is_valid() else 'miss',
                        'gap_cache': 'enabled',
                        'pattern_cache': 'enabled' if self.pattern_enabled else 'disabled'
                    }
                }
            }
            
            logger.info(f"✅ {symbol} 종합 분석 완료 (소요시간: {elapsed_time:.1f}초)")
            logger.info(f"📊 최종 점수: {overall_score:.2f}/10")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {symbol} 종합 분석 실패: {e}")
            logger.error(traceback.format_exc())
            return self._get_error_result(symbol, str(e))

    async def _calculate_v_score_enhanced(self, symbol: str) -> dict:
        """V 점수 계산 (한투 API 우선, yfinance 폴백)"""
        try:
            # 1. 한투 API 시도
            if self.kis_token_manager:
                kis_data = await self._fetch_kis_financial_data(symbol)
                if kis_data and kis_data.get('success'):
                    return self._calculate_v_from_kis_data(kis_data['data'])
            
            # 2. yfinance 폴백
            yf_data = await self._fetch_yfinance_data(symbol)
            if yf_data:
                return self._calculate_v_from_yfinance_data(yf_data)
            
            # 3. 최종 폴백
            return self._get_fallback_score('V')
            
        except Exception as e:
            logger.error(f"V 점수 계산 실패: {e}")
            return self._get_fallback_score('V')

    async def _calculate_t_score_enhanced(self, symbol: str) -> dict:
        """T 점수 계산 (한투 API + yfinance)"""
        try:
            # yfinance에서 기술적 지표 계산
            yf_data = await self._fetch_yfinance_data(symbol)
            if yf_data:
                return self._calculate_t_from_yfinance_data(yf_data)
            
            return self._get_fallback_score('T')
            
        except Exception as e:
            logger.error(f"T 점수 계산 실패: {e}")
            return self._get_fallback_score('T')

    async def _calculate_n_score_gemini_flash(self, symbol: str) -> dict:
        """N 점수 계산 (Gemini Flash)"""
        try:
            gemini_config = self.api_manager.get_gemini_config()
            api_key = gemini_config['api_key']
            
            if not api_key or api_key == 'GEM':
                return self._get_fallback_score('N')
            
            query = f"""
{symbol} 주식의 최근 뉴스와 시장 심리를 분석하여 N점수(0-10)를 계산해주세요.

분석 요소:
1. 최근 1주일 뉴스 감정 분석
2. 애널리스트 리포트 및 등급 변화
3. 소셜미디어 언급량 및 감정
4. 기관투자자 동향
5. 업종 관련 이슈 및 전망

N점수: [0-10]
"""

            url = f"{gemini_config['base_url']}/models/gemini-2.0-flash-exp:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            
            payload = {
                "contents": [{"parts": [{"text": query}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 300
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                        
                        score = self._extract_score_from_text(content)
                        return {
                            'score': score,
                            'source': 'Gemini Flash',
                            'content': content[:200],
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        return self._get_fallback_score('N')
                        
        except Exception as e:
            logger.error(f"N 점수 계산 실패: {e}")
            return self._get_fallback_score('N')

    async def _calculate_f_score_enhanced(self, symbol: str) -> dict:
        """F 점수 계산 (자금 흐름 분석)"""
        try:
            # yfinance에서 거래량 및 기관 데이터
            yf_data = await self._fetch_yfinance_data(symbol)
            if yf_data:
                return self._calculate_f_from_yfinance_data(yf_data)
            
            return self._get_fallback_score('F')
            
        except Exception as e:
            logger.error(f"F 점수 계산 실패: {e}")
            return self._get_fallback_score('F')

    async def _fetch_kis_financial_data(self, symbol: str) -> dict:
        """한투 API에서 재무 데이터 조회"""
        try:
            if not self.kis_token_manager:
                return {'success': False, 'error': 'KIS token manager not initialized'}
            
            headers = self.kis_token_manager.get_auth_headers("FHKST01010100")
            
            # 기본 정보 조회
            url = f"{self.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {'success': True, 'data': data}
                    else:
                        return {'success': False, 'error': f'API error: {response.status}'}
                        
        except Exception as e:
            logger.error(f"한투 API 조회 실패: {e}")
            return {'success': False, 'error': str(e)}

    async def _fetch_yfinance_data(self, symbol: str) -> dict:
        """yfinance에서 데이터 조회"""
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            def get_yf_data():
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="3mo")
                return {'info': info, 'hist': hist}
            
            # 별도 스레드에서 실행
            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(executor, get_yf_data)
                return result
                
        except Exception as e:
            logger.error(f"yfinance 조회 실패: {e}")
            return None

    def _calculate_v_from_kis_data(self, data: dict) -> dict:
        """한투 데이터로 V 점수 계산"""
        try:
            # 한투 API 데이터 파싱 및 점수 계산
            # 실제 구현 필요
            score = 6.0  # 임시
            
            return {
                'score': score,
                'source': 'KIS API',
                'components': {
                    'per': 0,
                    'pbr': 0,
                    'revenue_growth': 0,
                    'profit_margin': 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"KIS V 점수 계산 실패: {e}")
            return self._get_fallback_score('V')

    def _calculate_v_from_yfinance_data(self, data: dict) -> dict:
        """yfinance 데이터로 V 점수 계산"""
        try:
            info = data.get('info', {})
            
            # 기본 지표들
            pe_ratio = info.get('forwardPE') or info.get('trailingPE', 20)
            pb_ratio = info.get('priceToBook', 1.5)
            revenue_growth = info.get('revenueGrowth', 0) * 100
            profit_margin = info.get('profitMargins', 0) * 100
            debt_to_equity = info.get('debtToEquity', 50)
            
            # 점수 계산 (0-10)
            pe_score = max(0, min(10, 10 - (pe_ratio - 15) * 0.2)) if pe_ratio else 5
            pb_score = max(0, min(10, 10 - (pb_ratio - 1) * 2)) if pb_ratio else 5
            growth_score = max(0, min(10, 5 + revenue_growth * 0.2))
            margin_score = max(0, min(10, profit_margin * 0.4))
            debt_score = max(0, min(10, 10 - debt_to_equity * 0.1))
            
            final_score = (pe_score * 0.25 + pb_score * 0.2 + growth_score * 0.25 + 
                          margin_score * 0.2 + debt_score * 0.1)
            
            return {
                'score': round(final_score, 2),
                'source': 'yfinance',
                'components': {
                    'per': pe_ratio,
                    'pbr': pb_ratio,
                    'revenue_growth': revenue_growth,
                    'profit_margin': profit_margin,
                    'debt_to_equity': debt_to_equity
                },
                'sub_scores': {
                    'pe_score': round(pe_score, 2),
                    'pb_score': round(pb_score, 2),
                    'growth_score': round(growth_score, 2),
                    'margin_score': round(margin_score, 2),
                    'debt_score': round(debt_score, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"yfinance V 점수 계산 실패: {e}")
            return self._get_fallback_score('V')

    def _calculate_t_from_yfinance_data(self, data: dict) -> dict:
        """yfinance 데이터로 T 점수 계산"""
        try:
            hist = data.get('hist')
            if hist is None or hist.empty:
                return self._get_fallback_score('T')
            
            # 이동평균 계산
            hist['MA5'] = hist['Close'].rolling(5).mean()
            hist['MA10'] = hist['Close'].rolling(10).mean()
            hist['MA20'] = hist['Close'].rolling(20).mean()
            hist['MA50'] = hist['Close'].rolling(50).mean()
            
            current_price = hist['Close'].iloc[-1]
            
            # 이동평균 위 개수
            ma_above_count = 0
            for ma in ['MA5', 'MA10', 'MA20', 'MA50']:
                if len(hist) > 50 and not pd.isna(hist[ma].iloc[-1]):
                    if current_price > hist[ma].iloc[-1]:
                        ma_above_count += 1
            
            # RSI 계산
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # 볼륨 분석
            volume_ma = hist['Volume'].rolling(20).mean()
            volume_ratio = hist['Volume'].iloc[-1] / volume_ma.iloc[-1] if not pd.isna(volume_ma.iloc[-1]) else 1
            
            # T 점수 계산
            ma_score = (ma_above_count / 4) * 4  # 0-4점
            rsi_score = 2 if 30 <= current_rsi <= 70 else (1 if 20 <= current_rsi <= 80 else 0)
            volume_score = min(4, volume_ratio * 2)  # 0-4점
            
            final_score = ma_score + rsi_score + volume_score
            
            return {
                'score': round(final_score, 2),
                'source': 'yfinance',
                'components': {
                    'ma_above_count': ma_above_count,
                    'rsi': round(current_rsi, 2),
                    'volume_ratio': round(volume_ratio, 2),
                    'current_price': round(current_price, 2)
                },
                'sub_scores': {
                    'ma_score': round(ma_score, 2),
                    'rsi_score': round(rsi_score, 2),
                    'volume_score': round(volume_score, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"T 점수 계산 실패: {e}")
            return self._get_fallback_score('T')

    def _calculate_f_from_yfinance_data(self, data: dict) -> dict:
        """yfinance 데이터로 F 점수 계산"""
        try:
            info = data.get('info', {})
            hist = data.get('hist')
            
            # 기관 소유 비율
            institutional_pct = info.get('heldPercentInstitutions', 0) * 100
            
            # 거래량 트렌드
            if hist is not None and not hist.empty:
                volume_trend = (hist['Volume'].tail(5).mean() / hist['Volume'].tail(20).mean()) if len(hist) >= 20 else 1
            else:
                volume_trend = 1
            
            # 유동성 지표
            avg_volume = info.get('averageVolume', 0)
            float_shares = info.get('floatShares', info.get('sharesOutstanding', 1))
            liquidity_ratio = avg_volume / float_shares if float_shares > 0 else 0
            
            # F 점수 계산
            institutional_score = min(5, institutional_pct * 0.1)
            volume_score = min(3, (volume_trend - 0.8) * 10) if volume_trend > 0.8 else 0
            liquidity_score = min(2, liquidity_ratio * 1000)
            
            final_score = institutional_score + volume_score + liquidity_score
            
            return {
                'score': round(final_score, 2),
                'source': 'yfinance',
                'components': {
                    'institutional_ownership': round(institutional_pct, 2),
                    'volume_trend': round(volume_trend, 2),
                    'liquidity_ratio': round(liquidity_ratio * 1000, 4),
                    'avg_volume': avg_volume
                },
                'sub_scores': {
                    'institutional_score': round(institutional_score, 2),
                    'volume_score': round(volume_score, 2),
                    'liquidity_score': round(liquidity_score, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"F 점수 계산 실패: {e}")
            return self._get_fallback_score('F')

    def _extract_score_from_text(self, text: str) -> float:
        """텍스트에서 점수 추출"""
        try:
            import re
            
            patterns = [
                r'N점수:\s*(\d+(?:\.\d+)?)',
                r'점수:\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)/10',
                r'(\d+(?:\.\d+)?)\s*점'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    score = float(matches[0])
                    if 0 <= score <= 10:
                        return score
                    elif score > 10:
                        return min(score / 10, 10.0)
            
            return 6.0  # 기본값
            
        except Exception:
            return 6.0

    def _get_fallback_score(self, component: str) -> dict:
        """폴백 점수 반환"""
        fallback_scores = {'V': 6.0, 'T': 5.5, 'N': 6.0, 'F': 5.0}
        
        return {
            'score': fallback_scores.get(component, 6.0),
            'source': 'fallback',
            'components': {},
            'error': 'Data source unavailable',
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_weighted_scores(self, vtnf_scores: dict, sector_weights: dict) -> dict:
        """가중치 적용 점수 계산"""
        try:
            weighted_scores = {}
            
            for component in ['V', 'T', 'N', 'F']:
                score_data = vtnf_scores.get(component, {})
                raw_score = score_data.get('score', 6.0) if isinstance(score_data, dict) else score_data
                weight = sector_weights.get(component, 0.25)
                weighted_score = raw_score * weight
                
                weighted_scores[component] = {
                    'raw_score': round(raw_score, 2),
                    'weight': weight,
                    'weighted_score': round(weighted_score, 2)
                }
            
            return weighted_scores
            
        except Exception as e:
            logger.error(f"가중치 점수 계산 실패: {e}")
            return {}

    def _calculate_overall_score(self, final_scores: dict, gap_analysis: dict, pattern_analysis: dict) -> float:
        """종합 점수 계산"""
        try:
            # 기본 VTNF 점수 합계
            base_score = sum(score_data.get('weighted_score', 0) for score_data in final_scores.values())
            
            # 갭 부스터 조정
            gap_adjustment = 0
            if gap_analysis.get('gap_detected'):
                position_adj = gap_analysis.get('position_adjustment', 0)
                gap_adjustment = position_adj * 0.01  # 퍼센트를 점수로 변환
            
            # 패턴 부스터 조정
            pattern_adjustment = 0
            if pattern_analysis.get('pattern_detected', 'None') != 'None':
                pattern_score = pattern_analysis.get('pattern_score', 5.0)
                pattern_adjustment = (pattern_score - 5.0) * 0.1  # -0.5 ~ +0.5 범위
            
            overall = base_score + gap_adjustment + pattern_adjustment
            return round(max(0, min(10, overall)), 2)
            
        except Exception as e:
            logger.error(f"종합 점수 계산 실패: {e}")
            return 6.0

    def _generate_investment_recommendation(self, symbol: str, sector: str, final_scores: dict, 
                                          overall_score: float, gap_analysis: dict, pattern_analysis: dict) -> dict:
        """투자 제안 생성"""
        try:
            # 기본 투자 액션
            if overall_score >= 7.5:
                action = "STRONG_BUY"
                confidence = "높음"
            elif overall_score >= 6.5:
                action = "BUY"
                confidence = "중간"
            elif overall_score >= 5.5:
                action = "HOLD"
                confidence = "중간"
            elif overall_score >= 4.5:
                action = "WEAK_SELL"
                confidence = "중간"
            else:
                action = "SELL"
                confidence = "높음"
            
            # 갭 분석 반영
            gap_impact = ""
            if gap_analysis.get('gap_detected'):
                boost_type = gap_analysis.get('boost_type', 'neutral')
                if boost_type == 'synergy':
                    gap_impact = "AI 시너지 효과로 상향 조정"
                elif boost_type == 'risk':
                    gap_impact = "AI 리스크 요소로 하향 조정"
            
            # 패턴 분석 반영
            pattern_impact = ""
            if pattern_analysis.get('pattern_detected', 'None') != 'None':
                pattern_name = pattern_analysis.get('pattern_detected')
                direction = pattern_analysis.get('direction_bias', 'Neutral')
                pattern_impact = f"{pattern_name} 패턴, {direction} 신호"
            
            # 리스크 요소
            risk_factors = []
            if overall_score < 6.0:
                risk_factors.append("전반적 점수 부진")
            if gap_analysis.get('boost_type') == 'risk':
                risk_factors.append("AI 분석 리스크 감지")
            if pattern_analysis.get('risk_level', 50) > 70:
                risk_factors.append("패턴 실패 위험 높음")
            
            return {
                'action': action,
                'confidence': confidence,
                'overall_score': overall_score,
                'target_score_range': self._get_target_range(overall_score),
                'key_strengths': self._identify_strengths(final_scores),
                'key_weaknesses': self._identify_weaknesses(final_scores),
                'gap_impact': gap_impact,
                'pattern_impact': pattern_impact,
                'risk_factors': risk_factors,
                'recommended_position_size': self._calculate_position_size(overall_score, gap_analysis),
                'time_horizon': self._suggest_time_horizon(action, pattern_analysis),
                'stop_loss_suggestion': self._suggest_stop_loss(overall_score, pattern_analysis),
                'profit_target_suggestion': self._suggest_profit_target(overall_score, gap_analysis),
                'next_review_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            logger.error(f"투자 제안 생성 실패: {e}")
            return {
                'action': 'HOLD',
                'confidence': '낮음',
                'error': str(e)
            }

    def _get_target_range(self, score: float) -> str:
        """목표 점수 범위"""
        if score >= 8:
            return "8.0-10.0 (매우 우수)"
        elif score >= 7:
            return "7.0-8.0 (우수)"
        elif score >= 6:
            return "6.0-7.0 (양호)"
        elif score >= 5:
            return "5.0-6.0 (보통)"
        else:
            return "4.0-5.0 (부진)"

    def _identify_strengths(self, final_scores: dict) -> List[str]:
        """강점 식별"""
        strengths = []
        for component, data in final_scores.items():
            if data.get('raw_score', 0) >= 7:
                component_names = {'V': '가치', 'T': '기술적', 'N': '뉴스', 'F': '자금'}
                strengths.append(f"{component_names.get(component, component)} 지표 우수")
        return strengths

    def _identify_weaknesses(self, final_scores: dict) -> List[str]:
        """약점 식별"""
        weaknesses = []
        for component, data in final_scores.items():
            if data.get('raw_score', 0) <= 4:
                component_names = {'V': '가치', 'T': '기술적', 'N': '뉴스', 'F': '자금'}
                weaknesses.append(f"{component_names.get(component, component)} 지표 부진")
        return weaknesses

    def _calculate_position_size(self, overall_score: float, gap_analysis: dict) -> str:
        """포지션 크기 제안"""
        base_size = min(20, max(5, overall_score * 2))  # 5-20% 기본 범위
        
        if gap_analysis.get('gap_detected'):
            adjustment = gap_analysis.get('position_adjustment', 0)
            adjusted_size = base_size + (adjustment * 0.3)  # 조정 적용
            adjusted_size = max(2, min(30, adjusted_size))  # 2-30% 제한
            return f"{adjusted_size:.1f}% (갭 조정 적용)"
        
        return f"{base_size:.1f}%"

    def _suggest_time_horizon(self, action: str, pattern_analysis: dict) -> str:
        """투자 기간 제안"""
        if action in ['STRONG_BUY', 'BUY']:
            return "중장기 (3-12개월)"
        elif action == 'HOLD':
            return "단기 모니터링 (1-3개월)"
        else:
            return "즉시 검토 필요"

    def _suggest_stop_loss(self, overall_score: float, pattern_analysis: dict) -> str:
        """손절매 제안"""
        base_stop = max(5, min(15, 15 - overall_score))
        
        if pattern_analysis.get('support_level'):
            return f"패턴 지지선 하향 돌파 시 ({base_stop:.1f}% 가이드)"
        
        return f"현재가 대비 -{base_stop:.1f}%"

    def _suggest_profit_target(self, overall_score: float, gap_analysis: dict) -> str:
        """수익 목표 제안"""
        base_target = max(10, overall_score * 3)
        
        if gap_analysis.get('boost_type') == 'synergy':
            enhanced_target = base_target * 1.3
            return f"+{enhanced_target:.1f}% (시너지 부스트)"
        
        return f"+{base_target:.1f}%"

    def _get_error_result(self, symbol: str, error_msg: str) -> dict:
        """에러 결과 반환"""
        return {
            'symbol': symbol,
            'error': error_msg,
            'overall_score': 5.0,
            'investment_recommendation': {
                'action': 'HOLD',
                'confidence': '낮음',
                'error': '분석 중 오류 발생'
            },
            'timestamp': datetime.now().isoformat()
        }

    async def analyze_multiple_symbols(self, symbols: List[str]) -> dict:
        """다중 종목 분석"""
        try:
            logger.info(f"🎯 다중 종목 분석 시작: {len(symbols)}개")
            
            # 최대 10개까지 제한
            symbols = symbols[:10]
            
            # 병렬 분석
            tasks = [self.analyze_symbol_comprehensive(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 정리
            analysis_results = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    analysis_results[symbol] = self._get_error_result(symbol, str(result))
                else:
                    analysis_results[symbol] = result
            
            # 포트폴리오 요약
            portfolio_summary = self._generate_portfolio_summary(analysis_results)
            
            return {
                'symbols': analysis_results,
                'portfolio_summary': portfolio_summary,
                'total_analyzed': len(symbols),
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"다중 종목 분석 실패: {e}")
            return {
                'symbols': {},
                'error': str(e),
                'total_analyzed': 0
            }

    def _generate_portfolio_summary(self, results: dict) -> dict:
        """포트폴리오 요약 생성"""
        try:
            if not results:
                return {}
            
            scores = [r.get('overall_score', 5.0) for r in results.values()]
            actions = [r.get('investment_recommendation', {}).get('action', 'HOLD') for r in results.values()]
            
            action_counts = {}
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            return {
                'average_score': round(sum(scores) / len(scores), 2),
                'score_distribution': {
                    'excellent': len([s for s in scores if s >= 8]),
                    'good': len([s for s in scores if 7 <= s < 8]),
                    'fair': len([s for s in scores if 6 <= s < 7]),
                    'poor': len([s for s in scores if s < 6])
                },
                'action_distribution': action_counts,
                'top_picks': [symbol for symbol, data in results.items() 
                             if data.get('overall_score', 0) >= 7.5],
                'risk_symbols': [symbol for symbol, data in results.items() 
                               if data.get('overall_score', 10) <= 4.5]
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 요약 생성 실패: {e}")
            return {}

    def get_system_status(self) -> dict:
        """시스템 상태 반환"""
        return {
            'system_version': '1.8.2',
            'components': {
                'sector_cache': {
                    'valid': self.sector_cache.is_valid(),
                    'last_updated': self.sector_cache.last_updated.isoformat() if self.sector_cache.last_updated else None,
                    'cache_hours': self.sector_cache.cache_hours
                },
                'gap_booster': {
                    'enabled': hasattr(self, 'gap_booster'),
                    'cache_stats': self.gap_booster.get_cache_stats() if hasattr(self, 'gap_booster') else {}
                },
                'pattern_booster': {
                    'enabled': self.pattern_enabled,
                    'cache_stats': self.pattern_booster.get_cache_stats() if hasattr(self, 'pattern_booster') else {}
                },
                'data_validator': {
                    'strict_mode': self.data_validator.strict_mode if hasattr(self, 'data_validator') else False
                }
            },
            'api_status': {
                'kis_token_manager': self.kis_token_manager is not None,
                'gemini_config': bool(self.api_manager.get_gemini_config().get('api_key')),
                'perplexity_config': bool(self.api_manager.get_perplexity_config().get('api_key'))
            },
            'timestamp': datetime.now().isoformat()
        }
