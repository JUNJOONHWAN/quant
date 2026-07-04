#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_optimizer.py - Enhanced Real Data-Based Backtesting Genetic Algorithm Optimizer
v6.0 - Sweet Spot Detailed Parameters Support (60 parameters)
"""

import os
import json
import random
import logging
import numpy as np
import pickle
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import copy
import gc
from score_recalculator import ScoreRecalculator
from collections import OrderedDict
import aiofiles

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ml_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# Data Classes - WITH HYBRID COMPATIBILITY
# ================================================================

@dataclass
class BacktestResult:
    """Backtest result data structure"""
    total_picks: int
    win_rate: float
    average_return: float
    best_return: float
    worst_return: float
    std_dev: float
    sharpe_ratio: float
    max_drawdown: float
    fitness_score: float
    selected_symbols: List[str]
    backtest_period: str
    parameters_used: Dict = None
    # Alignment metrics (score vs DB 2-week returns)
    alignment_spearman_all: Optional[float] = None
    alignment_spearman_topk: Optional[float] = None
    alignment_pearson_all: Optional[float] = None
    alignment_pearson_topk: Optional[float] = None
    alignment_n_all: int = 0
    alignment_n_topk: int = 0

@dataclass
class Individual:
    """Individual in genetic algorithm"""
    parameters: Dict
    fitness: float = 0.0
    backtest_result: Optional[BacktestResult] = None
    generation: int = 0

# *** HYBRID COMPATIBILITY: OptimizationResult class added ***
@dataclass
class OptimizationResult:
    """ML Optimization result for Hybrid compatibility"""
    best_parameters: Dict
    fitness_score: float
    optimization_confidence: float  # 0.0 ~ 1.0
    expected_improvement: float     # Expected improvement percentage
    convergence_generation: int     # Generation where convergence occurred

# ================================================================
# Enhanced Parameter Validator (v6.0 - 60 parameters)
# ================================================================

class ParameterValidator:
    """Parameter validation and normalization - Sweet Spot v6.0 Enhanced"""
    
    @staticmethod
    def validate_parameters(params: Dict) -> Dict:
        """Validate and normalize ALL parameters including detailed weights (60 total)"""
        validated = copy.deepcopy(params)
        
        # 1. Main weights validation (6 parameters)
        weights = validated.get('main_scoring_weights', {})
        for key, value in weights.items():
            weights[key] = max(0.01, min(1.0, value))
        
        # Normalize main weights
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] = weights[key] / total
        
        # 2. Sweet Spot v6.0 세부 가중치 검증 및 정규화 (28 parameters)
        if 'detailed_scoring_weights' in validated:
            detailed_weights = validated['detailed_scoring_weights']
            
            # 각 카테고리별 검증 및 정규화
            category_constraints = {
                'pattern_scoring': {'min': 0.02, 'max': 0.30, 'keys': 8},
                'convergence_scoring': {'min': 0.03, 'max': 0.35, 'keys': 6},
                'growth_scoring': {'min': 0.08, 'max': 0.40, 'keys': 5},
                'tech_scoring': {'min': 0.10, 'max': 0.45, 'keys': 4},
                'institutional_scoring': {'min': 0.15, 'max': 0.70, 'keys': 3},
                'financial_scoring': {'min': 0.25, 'max': 0.80, 'keys': 2}
            }
            
            for category, category_weights in detailed_weights.items():
                if isinstance(category_weights, dict):
                    constraints = category_constraints.get(category, {'min': 0.02, 'max': 0.40})
                    
                    # 범위 제한
                    for key, value in category_weights.items():
                        category_weights[key] = max(constraints['min'], min(constraints['max'], value))
                    
                    # 정규화
                    total_category = sum(category_weights.values())
                    if total_category > 0:
                        for key in category_weights:
                            category_weights[key] = category_weights[key] / total_category
        
        # 3. Sweet Spot multipliers validation (5 parameters)
        multipliers = validated.get('sweet_spot_multipliers', {})
        for key, value in multipliers.items():
            if 'penalty' in key:
                multipliers[key] = max(0.1, min(0.9, value))
            elif 'multiplier' in key:
                multipliers[key] = max(0.5, min(2.0, value))
        
        # 4. Deep tech category multipliers validation (8 parameters)
        tech_multipliers = validated.get('deeptech_category_multipliers', {})
        for key, value in tech_multipliers.items():
            tech_multipliers[key] = max(0.5, min(1.5, value))
        
        # 5. Volume signal weights validation (새로 추가 - 4 parameters)
        if 'volume_signal_weights' in validated:
            volume_weights = validated['volume_signal_weights']
            for key, value in volume_weights.items():
                volume_weights[key] = max(1.0, min(1.8, value))
        
        # 6. Deeptech subcategory weights validation (새로 추가 - 8 parameters)
        if 'deeptech_subcategory_weights' in validated:
            deeptech_sub_weights = validated['deeptech_subcategory_weights']
            for category, subcategories in deeptech_sub_weights.items():
                if isinstance(subcategories, dict):
                    for subcat_key, subcat_value in subcategories.items():
                        subcategories[subcat_key] = max(0.8, min(1.8, subcat_value))
        
        # 7. Threshold validation (기존 4 parameters)
        validated['min_crash_percent'] = max(10, min(50, validated.get('min_crash_percent', 20)))
        validated['min_recovery_percent'] = max(5, min(30, validated.get('min_recovery_percent', 15)))
        validated['volume_spike_multiplier'] = max(1.2, min(5.0, validated.get('volume_spike_multiplier', 2.5)))
        validated['volume_trend_multiplier'] = max(1.0, min(2.0, validated.get('volume_trend_multiplier', 1.3)))
        
        return validated

# ================================================================
# LRU Cache Manager
# ================================================================

class LRUCache:
    """Memory efficient LRU cache"""
    
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Store data in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        gc.collect()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }

# ================================================================
# Enhanced ML Parameter Manager (v6.0 - 60 parameters)
# ================================================================

class MLParameterManager:
    """ML parameter management - Sweet Spot v6.0 Enhanced"""
    
    def __init__(self, config_file="ml_parameters.json"):
        self.config_file = config_file
        self.parameters = self._load_parameters()
        self.backup_file = config_file.replace('.json', '_backup.json')
        self.validator = ParameterValidator()
    
    def _load_parameters(self) -> Dict:
        """Load parameters from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info("Loaded existing ML parameters")
                    return data
            return self._get_default_parameters()
        except Exception as e:
            logger.warning(f"Failed to load parameters: {e}")
            return self._get_default_parameters()

    def get_current_parameters(self) -> Dict:
        """Return the current validated parameter set (read-only view)."""
        try:
            return self.parameters.get('current_parameters', {})
        except Exception:
            return {}
    
    def _get_default_parameters(self) -> Dict:
        """Get default Sweet Spot v6.0 parameters WITH detailed weights (60 total)"""
        return {
            'current_parameters': {
                # 메인 스코어링 가중치 (6개)
                'main_scoring_weights': {
                    'pattern_score': 0.25,
                    'convergence_score': 0.30,
                    'growth_score': 0.25,
                    'tech_score': 0.20,
                    'institutional_score': 0.10,
                    'financial_score': 0.05
                },
                
                # Sweet Spot v6.0 세부 가중치 (28개)
                'detailed_scoring_weights': {
                    'pattern_scoring': {
                        'crash_depth_weight': 0.20,
                        'recovery_velocity_weight': 0.18,
                        'recovery_position_weight': 0.15,
                        'volatility_compression_weight': 0.12,
                        'support_strength_weight': 0.15,
                        'breakout_proximity_weight': 0.10,
                        'volume_pattern_weight': 0.08,
                        'pattern_similarity_weight': 0.02
                    },
                    'convergence_scoring': {
                        'rsi_recovery_weight': 0.25,
                        'macd_timing_weight': 0.25,
                        'bollinger_squeeze_weight': 0.20,
                        'moving_avg_convergence_weight': 0.15,
                        'volume_oscillator_weight': 0.10,
                        'technical_confluence_weight': 0.05
                    },
                    'growth_scoring': {
                        'revenue_acceleration_weight': 0.30,
                        'pipeline_strength_weight': 0.25,
                        'partnership_catalyst_weight': 0.20,
                        'market_expansion_weight': 0.15,
                        'regulatory_tailwind_weight': 0.10
                    },
                    'tech_scoring': {
                        'innovation_cycle_position_weight': 0.35,
                        'tech_adoption_curve_weight': 0.25,
                        'scaling_readiness_weight': 0.25,
                        'tech_validation_weight': 0.15
                    },
                    'institutional_scoring': {
                        'institutional_flow_weight': 0.50,
                        'analyst_momentum_weight': 0.30,
                        'insider_signal_weight': 0.20
                    },
                    'financial_scoring': {
                        'cash_adequacy_weight': 0.70,
                        'debt_management_weight': 0.30
                    }
                },
                
                # Sweet Spot 배수 (5개)
                'sweet_spot_multipliers': {
                    'early_recovery_multiplier': 1.3,
                    'mid_recovery_multiplier': 1.0,
                    'late_recovery_multiplier': 0.8,
                    'golden_time_multiplier': 1.5,
                    'overheated_penalty': 0.6
                },
                
                # 딥테크 카테고리 배수 (8개)
                'deeptech_category_multipliers': {
                    'ai_computing': 1.2,
                    'quantum_tech': 1.3,
                    'bio_health_tech': 1.25,
                    'mobility_tech': 1.2,
                    'semiconductor': 1.1,
                    'energy_materials': 1.1,
                    'security_fintech': 1.0,
                    'emerging_tech': 1.15
                },
                
                # 거래량 신호 가중치 (새로 추가 - 4개)
                'volume_signal_weights': {
                    'spike_signal_weight': 1.25,
                    'trend_signal_weight': 1.15,
                    'combined_signal_weight': 1.35,
                    'volume_quality_weight': 1.20
                },
                
                # 딥테크 서브카테고리 가중치 (새로 추가 - 8개, 각 2개씩)
                'deeptech_subcategory_weights': {
                    'ai_computing': {
                        'machine_learning': 1.2,
                        'spatial_computing': 1.15
                    },
                    'mobility_tech': {
                        'evtol': 1.3,
                        'robotics': 1.25
                    },
                    'bio_health_tech': {
                        'biotech_ai': 1.35,
                        'neural_interface': 1.4
                    },
                    'energy_materials': {
                        'energy_storage': 1.15,
                        'new_materials': 1.1
                    }
                },
                
                # 기존 임계값 (4개)
                'min_crash_percent': 20,
                'min_recovery_percent': 15,
                'volume_spike_multiplier': 2.5,
                'volume_trend_multiplier': 1.3
            },
            'metadata': {
                'version': 'v6.0-detailed',
                'parameters_count': 60,  # 6 + 28 + 5 + 8 + 4 + 8 + 4 = 63개 (실제로는 더 많음)
                'last_optimization': datetime.now().isoformat(),
                'optimization_count': 0,
                'detailed_weights_enabled': True
            }
        }
    
    async def backup_current(self):
        """Backup current parameters"""
        try:
            async with aiofiles.open(self.backup_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.parameters, indent=2, ensure_ascii=False))
            logger.info("Parameters backed up successfully")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
    
    async def update_parameters(self, new_params: Dict, fitness_score: float = None):
        """Update parameters"""
        try:
            await self.backup_current()
            
            validated_params = self.validator.validate_parameters(new_params)
            
            if 'current_parameters' not in self.parameters:
                self.parameters['current_parameters'] = {}
            
            self.parameters['current_parameters'].update(validated_params)
            
            if 'metadata' not in self.parameters:
                self.parameters['metadata'] = {}
            
            self.parameters['metadata']['last_optimization'] = datetime.now().isoformat()
            self.parameters['metadata']['optimization_count'] = \
                self.parameters['metadata'].get('optimization_count', 0) + 1
            
            if fitness_score:
                self.parameters['metadata']['last_fitness_score'] = fitness_score
            
            async with aiofiles.open(self.config_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.parameters, indent=2, ensure_ascii=False))
            
            logger.info(f"ML parameters updated (fitness: {fitness_score:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to save parameters: {e}")

# ================================================================
# Enhanced Data Caching System
# ================================================================

class DataCache:
    """API data caching manager with TTL support"""
    
    def __init__(self, cache_dir="ml_cache", memory_cache_size=100, max_concurrent_file_ops: int = 32):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_index_file = os.path.join(cache_dir, "cache_index.json")
        self.cache_index = self._load_cache_index()

        # Memory cache (LRU)
        self.memory_cache = LRUCache(memory_cache_size)

        # Limit the number of simultaneous file handles to avoid OS ulimit exhaustion
        self._max_file_ops = max(1, int(max_concurrent_file_ops))
        self._file_semaphore: Optional[asyncio.Semaphore] = None
        
        # Statistics
        self.disk_hits = 0
        self.disk_misses = 0
        
        # TTL by data type (days)
        self.cache_ttl = {
            'prices': 1,
            'profile': 7,
            'institutional': 3,
            'sec_filings': 7,
            'news': 1,
            'financials': 30,
            'complete_data': 1
        }
    
    def _load_cache_index(self) -> Dict:
        """Load cache index"""
        try:
            if os.path.exists(self.cache_index_file):
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            return {}
    
    async def _save_cache_index(self):
        """Save cache index"""
        try:
            async with aiofiles.open(self.cache_index_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.cache_index, indent=2))
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def get_cache_key(self, symbol: str, data_type: str) -> str:
        """Generate cache key"""
        return f"{symbol}_{data_type}"
    
    def get_cache_file(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def is_cached(self, symbol: str, data_type: str, max_age_days: int = None) -> bool:
        """Check if data is cached and valid"""
        cache_key = self.get_cache_key(symbol, data_type)
        
        # Use type-specific TTL if not provided
        if max_age_days is None:
            max_age_days = self.cache_ttl.get(data_type, 7)
        
        # Check memory cache
        if self.memory_cache.get(cache_key) is not None:
            return True
        
        # Check disk cache
        if cache_key not in self.cache_index:
            return False
        
        # Check cache expiry
        try:
            cached_time = datetime.fromisoformat(self.cache_index[cache_key]['timestamp'])
            age_days = (datetime.now() - cached_time).days
            return age_days <= max_age_days
        except:
            return False
    
    async def get_cached_data(self, symbol: str, data_type: str) -> Optional[Any]:
        """Load cached data (memory -> disk)"""
        cache_key = self.get_cache_key(symbol, data_type)

        # Check memory cache
        data = self.memory_cache.get(cache_key)
        if data is not None:
            logger.debug(f"Memory cache hit: {cache_key}")
            return data

        # Check disk cache
        try:
            cache_file = self.get_cache_file(cache_key)
            if os.path.exists(cache_file):
                semaphore = self._get_file_semaphore()
                async with semaphore:
                    async with aiofiles.open(cache_file, 'rb') as f:
                        content = await f.read()
                    data = pickle.loads(content)

                self.memory_cache.put(cache_key, data)
                self.disk_hits += 1
                logger.debug(f"Disk cache hit: {cache_key}")
                return data
            
            self.disk_misses += 1
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load cache {symbol}/{data_type}: {e}")
            self.disk_misses += 1
            return None
    
    async def save_to_cache(self, symbol: str, data_type: str, data: Any):
        """Save data to cache (memory + disk)"""
        try:
            cache_key = self.get_cache_key(symbol, data_type)

            # Save to memory cache
            self.memory_cache.put(cache_key, data)

            # Save to disk cache
            cache_file = self.get_cache_file(cache_key)
            serialized_data = pickle.dumps(data)

            semaphore = self._get_file_semaphore()
            async with semaphore:
                async with aiofiles.open(cache_file, 'wb') as f:
                    await f.write(serialized_data)

            # Update index
            self.cache_index[cache_key] = {
                'symbol': symbol,
                'data_type': data_type,
                'timestamp': datetime.now().isoformat(),
                'file': cache_file
            }
            await self._save_cache_index()

            logger.debug(f"Cached: {cache_key}")

        except Exception as e:
            logger.error(f"Failed to cache {symbol}/{data_type}: {e}")

    def _get_file_semaphore(self) -> asyncio.Semaphore:
        """Lazily create a semaphore tied to the current event loop"""
        if self._file_semaphore is None:
            self._file_semaphore = asyncio.Semaphore(self._max_file_ops)
        return self._file_semaphore
    
    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        total_disk = self.disk_hits + self.disk_misses
        disk_hit_rate = (self.disk_hits / total_disk * 100) if total_disk > 0 else 0
        
        return {
            'memory': memory_stats,
            'disk': {
                'hits': self.disk_hits,
                'misses': self.disk_misses,
                'hit_rate': disk_hit_rate,
                'total_cached': len(self.cache_index)
            }
        }
    
    def cleanup_memory(self):
        """Clean up memory"""
        self.memory_cache.clear()
        gc.collect()

# ================================================================
# API Rate Limiter
# ================================================================

class APIRateLimiter:
    """API rate limiting manager"""
    
    def __init__(self, max_concurrent=3, calls_per_minute=150):
        env_max_concurrent = int(os.getenv("ML_FMP_MAX_CONCURRENT", max_concurrent))
        env_calls_per_min = int(os.getenv("ML_FMP_CALLS_PER_MIN", calls_per_minute))
        self.base_delay = float(os.getenv("ML_FMP_BASE_DELAY", "0.20"))

        self.semaphore = asyncio.Semaphore(max(1, env_max_concurrent))
        self.calls_per_minute = max(1, env_calls_per_min)
        self.call_times = []
        self.total_calls = 0

    async def acquire(self):
        """Acquire API call permission"""
        await self.semaphore.acquire()
        
        # Check calls per minute
        now = time.time()
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        if len(self.call_times) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.call_times[0])
            logger.warning(f"API rate limit reached, waiting {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
        
        self.call_times.append(now)
        self.total_calls += 1

    def release(self):
        """Release API call permission"""
        self.semaphore.release()

# ================================================================
# Enhanced Backtesting Engine
# ================================================================

class RealBacktestEngine:
    """Real data-based backtesting engine - REQUIRES REAL DATA"""
    
    def __init__(self):
        self.cache = DataCache()
        self.fmp_api_key = os.getenv("FMP_API_KEY", "")
        self.rate_limiter = APIRateLimiter()
        self.fmp_client = None
        self.screener = None
        self.score_recalculator = ScoreRecalculator()
        # 최근 백테스트 요약 정보 (로그 요약용)
        self.last_backtest_summary: Dict[str, Any] = {}
        # DB 신뢰 모드: 스크리닝 결과를 신뢰하고 추가 게이트 최소화
        self.trust_db: bool = True
        # 평가 심볼 상한 (None이면 전체 평가)
        self.test_symbols_cap: Optional[int] = None
        # 점수 컷 (None이면 컷 미적용)
        self.min_score_threshold: Optional[float] = None
        # 최소 히스토리 일수 (적응형 윈도우 사용)
        self.min_history_days: int = 15
        # Top-K 선택 (각 개체에서 total_score 상위 K만 수익률 집계). None이면 전체 사용
        self.top_k: Optional[int] = 30
        # 세대 간 데이터 재사용을 위한 프리패치 캐시
        self._prefetched_symbol_data: Dict[str, Dict] = {}
        self._prefetch_lock = asyncio.Lock()
        
    def set_training_mode(self, enabled: bool):
        """Enable training-only features in score recalculation (kept off in production)."""
        try:
            self.score_recalculator.set_training_mode(bool(enabled))
        except Exception:
            pass

    async def initialize(self):
        """Async initialization"""
        try:
            from screening_app import FMPClient, InnovativeStockScreener
            self.fmp_client = FMPClient()
            self.screener = InnovativeStockScreener(self.fmp_client)
            await self.fmp_client.__aenter__()
            logger.info("FMP API client initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import screening_app: {e}")
            raise RuntimeError("Real backtesting requires FMP API client. Please ensure screening_app module is available.")
        except Exception as e:
            logger.error(f"Failed to initialize FMP client: {e}")
            raise RuntimeError(f"Failed to initialize FMP API client: {e}")
    
    async def cleanup(self):
        """Resource cleanup"""
        if self.fmp_client:
            try:
                await self.fmp_client.__aexit__(None, None, None)
                logger.info("FMP API client cleaned up")
            except:
                pass
        
        self.cache.cleanup_memory()
        self.reset_prefetched_data()
    
    async def _api_call_with_retry(self, api_func, *args, max_retries=3, **kwargs):
        """API call with retry logic"""
        for attempt in range(max_retries):
            try:
                await self.rate_limiter.acquire()
                try:
                    result = await api_func(*args, **kwargs)
                    if self.rate_limiter.base_delay > 0:
                        await asyncio.sleep(self.rate_limiter.base_delay)
                    return result
                finally:
                    self.rate_limiter.release()
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    return None
                
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}), retrying in {wait_time:.1f}s: {e}")
                await asyncio.sleep(wait_time)
        
        return None
    
    async def download_stock_data_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Download multiple stock data in parallel"""
        if not self.fmp_client:
            raise RuntimeError("FMP API client is required for real backtesting")
        
        results: Dict[str, Dict] = {}
        cached_symbols: Dict[str, Dict] = {}
        missing_symbols: List[str] = []

        # 기존 프리패치 데이터 활용
        async with self._prefetch_lock:
            for symbol in symbols:
                cached = self._prefetched_symbol_data.get(symbol)
                if cached is not None:
                    cached_symbols[symbol] = cached
                else:
                    missing_symbols.append(symbol)

        results.update(cached_symbols)

        if missing_symbols:
            tasks = [asyncio.create_task(self.download_stock_data(symbol)) for symbol in missing_symbols]
            for symbol, task in zip(missing_symbols, tasks):
                try:
                    data = await task
                    if data:
                        results[symbol] = data
                        async with self._prefetch_lock:
                            self._prefetched_symbol_data[symbol] = data
                except Exception as e:
                    logger.error(f"Failed to download data for {symbol}: {e}")

        # 요청된 순서에 맞춰 반환 (부족 데이터는 제외)
        return {symbol: results[symbol] for symbol in symbols if symbol in results}
    
    async def download_stock_data(self, symbol: str) -> Dict:
        """Download stock data - REAL DATA ONLY"""
        
        # Check if fully cached
        if self.cache.is_cached(symbol, 'complete_data'):
            cached = await self.cache.get_cached_data(symbol, 'complete_data')
            if cached:
                logger.debug(f"Using cached data: {symbol}")
                return cached
        
        # Require API client - NO MOCK DATA
        if not self.fmp_client:
            raise RuntimeError("Real backtesting requires FMP API client")
        
        # Parallel API calls
        stock_data = {}
        
        try:
            # Prepare concurrent API calls
            tasks = {
                'prices': self._api_call_with_retry(
                    self.fmp_client.get_historical_prices, symbol, 360
                ),
                'profile': self._api_call_with_retry(
                    self.fmp_client.get_company_profile, symbol
                ),
                'institutional': self._api_call_with_retry(
                    self.fmp_client.get_institutional_holders, symbol
                ),
                'sec_filings': self._api_call_with_retry(
                    self.fmp_client.get_sec_filings, symbol, limit=10
                ),
                'news': self._api_call_with_retry(
                    self.fmp_client.get_stock_news, symbol, limit=5
                )
            }
            
            # Execute all API calls
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Process results
            for i, (key, task) in enumerate(tasks.items()):
                result = results[i]
                if isinstance(result, Exception):
                    logger.warning(f"{symbol} {key} API call failed: {result}")
                    stock_data[key] = [] if key in ['institutional', 'sec_filings', 'news'] else {}
                else:
                    stock_data[key] = result if result else ([] if key in ['institutional', 'sec_filings', 'news'] else {})
            
            # Financial data (separate call)
            balance_sheet = await self._api_call_with_retry(
                self.fmp_client.get_balance_sheet, symbol
            )
            income_statements = await self._api_call_with_retry(
                self.fmp_client.get_income_statement, symbol, limit=4
            )
            
            stock_data['financials'] = {
                'balance_sheet': balance_sheet if balance_sheet else {},
                'income_statements': income_statements if income_statements else [],
                'cash_flows': [],
                'ratios': []
            }
            
            # Validate data quality
            if not stock_data.get('prices') or len(stock_data['prices']) < 60:
                logger.warning(f"{symbol}: Insufficient price data")
                return {}
            
            # Cache valid data
            await self.cache.save_to_cache(symbol, 'complete_data', stock_data)
            logger.info(f"Real data downloaded: {symbol}")
            
            return stock_data
            
        except Exception as e:
            logger.error(f"{symbol} API call failed: {e}")
            return {}

    def reset_prefetched_data(self):
        """세대 간 공유한 프리패치 데이터를 초기화"""
        self._prefetched_symbol_data.clear()
    
    def _analyze_pattern(self, prices: List[Dict], parameters: Dict) -> Dict:
        """Pattern analysis (enhanced version)"""
        try:
            close_prices = [float(p.get('close', 0)) for p in prices if p.get('close')]
            if len(close_prices) < 180:
                return {'has_pattern': False, 'score': 0}
            
            # Price statistics
            max_price = max(close_prices)
            min_price = min(close_prices)
            current_price = close_prices[0]
            
            # Find peaks and troughs
            max_idx = close_prices.index(max_price)
            min_idx = close_prices.index(min_price)
            
            # Calculate crash/recovery
            crash_percent = ((max_price - min_price) / max_price * 100) if max_price > 0 else 0
            recovery_percent = ((current_price - min_price) / min_price * 100) if min_price > 0 else 0
            
            min_crash = parameters.get('min_crash_percent', 20)
            min_recovery = parameters.get('min_recovery_percent', 15)
            
            has_pattern = crash_percent >= min_crash and recovery_percent >= min_recovery
            
            # Sweet Spot determination
            is_sweet_spot = 15 <= recovery_percent <= 150 and max_idx > min_idx
            
            # Recovery stage classification
            if recovery_percent < 15:
                recovery_stage = 'bottom'
            elif recovery_percent < 40:
                recovery_stage = 'early'
            elif recovery_percent < 120:
                recovery_stage = 'mid'
            else:
                recovery_stage = 'late'
            
            # Score calculation
            score = 0
            if has_pattern:
                base_score = min((crash_percent + recovery_percent) / 2, 90)
                
                # Pattern quality bonus
                if max_idx > min_idx:
                    base_score *= 1.1
                if 20 <= recovery_percent <= 100:
                    base_score *= 1.15
                
                score = min(base_score, 100)
            
            return {
                'has_pattern': has_pattern,
                'score': score,
                'crash_percent': crash_percent,
                'recovery_percent': recovery_percent,
                'is_sweet_spot': is_sweet_spot,
                'recovery_stage': recovery_stage,
                'max_idx': max_idx,
                'min_idx': min_idx
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {'has_pattern': False, 'score': 0}
    
    def _is_deep_tech(self, profile: Dict) -> bool:
        """Deep Tech company identification"""
        text_fields = [
            profile.get('description', ''),
            profile.get('companyName', ''),
            profile.get('industry', ''),
            profile.get('sector', '')
        ]
        
        combined_text = ' '.join(text_fields).lower()
        
        deep_tech_patterns = {
            'ai_computing': ['artificial intelligence', 'ai', 'machine learning', 'ml', 'neural network', 'deep learning'],
            'quantum_tech': ['quantum', 'quantum computing', 'quantum technology'],
            'bio_health_tech': ['biotechnology', 'biotech', 'pharmaceutical', 'drug discovery', 'gene therapy'],
            'semiconductor': ['semiconductor', 'chip', 'microprocessor', 'cpu', 'gpu'],
            'mobility_tech': ['autonomous', 'self-driving', 'robotics', 'automation'],
            'energy_materials': ['renewable energy', 'solar', 'battery', 'energy storage'],
            'emerging_tech': ['space', 'satellite', 'aerospace', 'rocket'],
            'security_fintech': ['fintech', 'blockchain', 'cryptocurrency', 'digital payment']
        }
        
        for category, keywords in deep_tech_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                return True
        
        return False
    
    def _get_tech_category(self, profile: Dict) -> str:
        """Technology category classification"""
        text = ' '.join([
            profile.get('description', ''),
            profile.get('industry', ''),
            profile.get('companyName', '')
        ]).lower()
        
        category_keywords = {
            'ai_computing': ['artificial intelligence', 'ai', 'machine learning', 'neural', 'deep learning'],
            'quantum_tech': ['quantum'],
            'bio_health_tech': ['biotech', 'pharmaceutical', 'drug', 'gene'],
            'semiconductor': ['semiconductor', 'chip', 'microprocessor'],
            'mobility_tech': ['autonomous', 'self-driving', 'robotics'],
            'energy_materials': ['energy', 'battery', 'solar', 'renewable'],
            'emerging_tech': ['space', 'satellite', 'aerospace'],
            'security_fintech': ['fintech', 'blockchain', 'crypto', 'payment']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general_tech'
    
    def _analyze_growth_signals(self, stock_data: Dict, parameters: Dict) -> float:
        """Growth signal analysis"""
        score = 0
        
        # SEC filings analysis
        filings = stock_data.get('sec_filings', [])
        growth_keywords = ['partnership', 'acquisition', 'merger', 'funding', 'approval', 'expansion', 'launch']
        
        for filing in filings[:5]:
            filing_text = f"{filing.get('title', '')} {filing.get('type', '')}".lower()
            matches = sum(1 for keyword in growth_keywords if keyword in filing_text)
            score += matches * 8
        
        # News analysis
        news = stock_data.get('news', [])
        positive_keywords = ['partnership', 'agreement', 'launch', 'approval', 'expansion', 'breakthrough']
        
        for article in news[:5]:
            title = article.get('title', '').lower()
            sentiment = article.get('sentiment', 'neutral')
            
            matches = sum(1 for keyword in positive_keywords if keyword in title)
            article_score = matches * 6
            
            if sentiment == 'positive':
                article_score *= 1.2
            elif sentiment == 'negative':
                article_score *= 0.5
            
            score += article_score
        
        # Institutional changes
        institutional = stock_data.get('institutional', [])
        for holder in institutional:
            change = holder.get('change', 0)
            if change > 0:
                score += min(change * 2, 15)
        
        return min(score, 100)
    
    def _analyze_convergence(self, prices: List[Dict]) -> float:
        """Convergence pattern analysis"""
        if len(prices) < 30:
            return 0
        
        periods = [7, 14, 30]
        volatilities = []
        
        for period in periods:
            if len(prices) >= period:
                period_prices = [float(p.get('close', 0)) for p in prices[:period] if p.get('close')]
                if len(period_prices) >= period * 0.8:
                    volatility = np.std(period_prices) / np.mean(period_prices) if np.mean(period_prices) > 0 else 1
                    volatilities.append(volatility)
        
        if not volatilities:
            return 0
        
        avg_volatility = np.mean(volatilities)
        
        # Volatility-based scoring
        if avg_volatility < 0.03:
            return 90
        elif avg_volatility < 0.05:
            return 75
        elif avg_volatility < 0.08:
            return 60
        elif avg_volatility < 0.12:
            return 40
        elif avg_volatility < 0.18:
            return 20
        else:
            return 0
    
    def _analyze_financial_health(self, financials: Dict) -> float:
        """Financial health analysis"""
        score = 0
        
        # Cash and debt analysis
        balance_sheet = financials.get('balance_sheet', {})
        cash = balance_sheet.get('cash', 0)
        debt = balance_sheet.get('totalDebt', 0)
        
        if cash > 0:
            score += 15
            if debt > 0:
                debt_ratio = debt / cash
                if debt_ratio < 0.5:
                    score += 10
                elif debt_ratio < 1.0:
                    score += 5
        
        # Profitability analysis
        income_statements = financials.get('income_statements', [])
        if income_statements and len(income_statements) > 0:
            recent_income = income_statements[0]
            revenue = recent_income.get('revenue', 0)
            net_income = recent_income.get('netIncome', 0)
            
            if revenue > 0:
                score += 20
                if net_income > 0:
                    score += 15
                    profit_margin = net_income / revenue
                    if profit_margin > 0.1:
                        score += 10
        
        return min(score, 100)
    
    async def run_real_screening(self, symbol: str, parameters: Dict, stock_data: Dict) -> Optional[Dict]:
        """Run real screening logic using ScoreRecalculator for consistency"""
        try:
            # Use ScoreRecalculator for unified scoring logic
            total_score = self.score_recalculator.calculate_score_with_parameters(
                symbol, stock_data, parameters
            )
            
            if total_score == 0:
                return None
            
            # Get pattern analysis for additional info
            prices = stock_data.get('prices', [])
            pattern_result = self._analyze_pattern(prices, parameters)
            
            return {
                'symbol': symbol,
                'total_score': total_score,
                'is_sweet_spot': pattern_result.get('is_sweet_spot', False),
                'recovery_stage': pattern_result.get('recovery_stage', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"{symbol} screening failed: {e}")
            return None
    
    async def run_backtest_with_parameters(self, parameters: Dict, 
                                          test_symbols: List[str] = None) -> BacktestResult:
        """Run backtest with real data"""
        try:
            # Ensure API client is available
            if not self.fmp_client:
                raise RuntimeError("FMP API client is required for real backtesting")
            
            if not test_symbols:
                test_symbols = self._get_test_symbols_from_db()
            
            if not test_symbols:
                test_symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'META', 'GOOGL', 'AMZN', 'NFLX']
            
            logger.info(f"Starting backtest: {len(test_symbols)} symbols")
            
            # 평가 심볼 결정
            if self.trust_db and (self.test_symbols_cap is None):
                selected_symbols = list(test_symbols)  # 전체 평가
            else:
                cap = self.test_symbols_cap if self.test_symbols_cap is not None else 20
                max_test_stocks = min(len(test_symbols), cap)
                selected_symbols = test_symbols[:max_test_stocks]
            
            # Parallel data download
            logger.info("Downloading stock data in parallel...")
            stock_data_batch = await self.download_stock_data_batch(selected_symbols)
            
            if not stock_data_batch:
                raise RuntimeError("Failed to download any stock data")
            
            # Parallel screening (score computation per symbol)
            screening_tasks = []
            for symbol in selected_symbols:
                if symbol in stock_data_batch:
                    task = asyncio.create_task(
                        self.run_real_screening(symbol, parameters, stock_data_batch[symbol])
                    )
                    screening_tasks.append((symbol, task))
            
            # Collect screening results
            selected_stocks: List[str] = []
            returns: List[float] = []
            score_passed = 0
            all_candidates: List[Tuple[str, float]] = []  # (symbol, score)
            prices_map: Dict[str, List[Dict]] = {}

            for symbol, task in screening_tasks:
                try:
                    result = await task
                    threshold = self.min_score_threshold
                    if result:
                        score = float(result.get('total_score', 0))
                        if threshold is None or score >= threshold:
                            all_candidates.append((symbol, score))
                            prices_map[symbol] = stock_data_batch[symbol].get('prices', [])
                            score_passed += 1
                except Exception as e:
                    logger.warning(f"{symbol} backtest failed: {e}")
                    continue

            # Top-K 선택 적용 (설정된 경우)
            if self.top_k is not None and len(all_candidates) > self.top_k:
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                chosen = all_candidates[: self.top_k]
            else:
                chosen = all_candidates

            # DB의 2주 수익률(current_return_pct) 사용하여 수익률 집계
            db_returns = self._load_db_2w_returns()
            missing = 0
            top_scores: List[float] = []
            for sym, sc in chosen:
                r = db_returns.get(sym)
                if r is None:
                    missing += 1
                    continue
                try:
                    return_pct = float(r)
                except Exception:
                    missing += 1
                    continue
                selected_stocks.append(sym)
                returns.append(return_pct)
                top_scores.append(sc)
                status = "🟢" if return_pct > 0 else "🔴"
                logger.info(f"{status} {sym}: Score {sc:.1f}, Return {return_pct:+.1f}% (DB 2w)")
            if missing > 0:
                logger.info(f"DB 2w return missing for {missing} symbols (skipped)")
            
            # Calculate performance metrics
            if not returns:
                logger.warning("No stocks selected")
                return BacktestResult(
                    total_picks=0, win_rate=0, average_return=0,
                    best_return=0, worst_return=0, std_dev=0,
                    sharpe_ratio=0, max_drawdown=0, fitness_score=0.1,
                    selected_symbols=[], backtest_period="60d->30d"
                )

            # Compute alignment metrics (All candidates vs DB returns, and Top-K)
            def _rank(vals: List[float]) -> List[float]:
                try:
                    arr = np.array(vals, dtype=float)
                    order = arr.argsort()
                    ranks = np.empty(len(arr), dtype=float)
                    ranks[order] = np.arange(len(arr), dtype=float)
                    # handle ties: average ranks
                    unique_vals, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
                    for i, cnt in enumerate(counts):
                        if cnt > 1:
                            idx = np.where(inverse == i)[0]
                            avg_rank = ranks[idx].mean()
                            ranks[idx] = avg_rank
                    return ranks.tolist()
                except Exception:
                    return [float('nan')] * len(vals)

            def _pearson(x: List[float], y: List[float]) -> Optional[float]:
                try:
                    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
                        return None
                    X = np.array(x, dtype=float)
                    Y = np.array(y, dtype=float)
                    sx = X.std()
                    sy = Y.std()
                    if sx == 0 or sy == 0:
                        return None
                    return float(np.corrcoef(X, Y)[0, 1])
                except Exception:
                    return None

            # All candidates with available DB returns
            all_scores: List[float] = []
            all_returns: List[float] = []
            for sym, sc in all_candidates:
                r = db_returns.get(sym)
                if r is None:
                    continue
                try:
                    rv = float(r)
                except Exception:
                    continue
                all_scores.append(sc)
                all_returns.append(rv)

            spearman_all = None
            pearson_all = None
            n_all = len(all_scores)
            if n_all >= 5:
                try:
                    spearman_all = _pearson(_rank(all_scores), _rank(all_returns))
                    pearson_all = _pearson(all_scores, all_returns)
                except Exception:
                    spearman_all = None
                    pearson_all = None

            # Top-K alignment
            spearman_topk = None
            pearson_topk = None
            n_topk = len(top_scores)
            if n_topk >= 5:
                try:
                    spearman_topk = _pearson(_rank(top_scores), _rank(returns))
                    pearson_topk = _pearson(top_scores, returns)
                except Exception:
                    spearman_topk = None
                    pearson_topk = None

            # Log alignment
            def _fmt(v: Optional[float]) -> str:
                return f"{v:.2f}" if isinstance(v, float) and not np.isnan(v) else "N/A"

            logger.info("\n🔗 Score ↔ 2주 수익 정렬력 (Alignment)")
            logger.info(f"   All: n={n_all}, Spearman={_fmt(spearman_all)}, Pearson={_fmt(pearson_all)}")
            logger.info(f"   Top-{self.top_k or 'All'}: n={n_topk}, Spearman={_fmt(spearman_topk)}, Pearson={_fmt(pearson_topk)}")
            
            # Statistics
            total_picks = len(returns)
            positive_returns = [r for r in returns if r > 0]
            win_rate = len(positive_returns) / total_picks
            average_return = np.mean(returns)
            best_return = max(returns)
            worst_return = min(returns)
            std_dev = np.std(returns)
            
            # Sharpe ratio
            sharpe_ratio = (average_return) / std_dev if std_dev > 0 else 0
            
            # Max drawdown
            max_drawdown = abs(worst_return) if worst_return < 0 else 0
            
            # Enhanced fitness score calculation WITH alignment (Spearman)
            risk_adjusted_return = average_return / (std_dev + 1e-6)
            
            # Alignment components: map Spearman from [-1,1] to [0,1]
            def _to01(v: Optional[float]) -> float:
                try:
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return 0.0
                    return max(0.0, min(1.0, (float(v) + 1.0) / 2.0))
                except Exception:
                    return 0.0
            align_all = _to01(spearman_all)
            align_topk = _to01(spearman_topk)
            
            # Weighted blend: prioritize rank alignment, keep basket metrics as secondary
            fitness_score = (
                align_all * 60.0 +                 # Global rank alignment (primary)
                align_topk * 20.0 +                # Top-K rank alignment
                risk_adjusted_return * 10.0 +      # Risk-adjusted return
                win_rate * 20.0 +                  # Win rate
                min(sharpe_ratio * 5.0, 10.0) +    # Sharpe ratio (capped)
                max(0.0, 10.0 - max_drawdown / 2)  # Drawdown penalty (smaller)
            )
            
            logger.info(f"Backtest complete: {total_picks} picks, Win rate {win_rate:.1%}, Avg return {average_return:+.1f}%")

            # Save summary for final optimization log
            self.last_backtest_summary = {
                'db_symbols_loaded': len(test_symbols),
                'tested_symbols': len(selected_symbols),
                'score_threshold': self.min_score_threshold if self.min_score_threshold is not None else 'None',
                'score_passed': score_passed,
                'picks': total_picks,
                'backtest_period': "db_current_return_pct(2w)",
                'top_k': self.top_k if self.top_k is not None else 'None',
                'alignment': {
                    'spearman_all': spearman_all,
                    'pearson_all': pearson_all,
                    'spearman_topk': spearman_topk,
                    'pearson_topk': pearson_topk,
                    'n_all': n_all,
                    'n_topk': n_topk
                }
            }
            
            return BacktestResult(
                total_picks=total_picks,
                win_rate=win_rate,
                average_return=average_return,
                best_return=best_return,
                worst_return=worst_return,
                std_dev=std_dev,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                fitness_score=max(fitness_score, 0.1),
                selected_symbols=selected_stocks,
                backtest_period="60d->30d",
                parameters_used=parameters,
                alignment_spearman_all=spearman_all,
                alignment_spearman_topk=spearman_topk,
                alignment_pearson_all=pearson_all,
                alignment_pearson_topk=pearson_topk,
                alignment_n_all=n_all,
                alignment_n_topk=n_topk
            )
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return BacktestResult(
                total_picks=0, win_rate=0, average_return=0,
                best_return=0, worst_return=0, std_dev=0,
                sharpe_ratio=0, max_drawdown=0, fitness_score=0.1,
                selected_symbols=[], backtest_period="60d->30d"
            )

    def _compute_return_pct(self, prices: List[Dict]) -> Optional[float]:
        """적응형 수익률 계산: 가용 히스토리에 맞춰 윈도우 선택"""
        try:
            if not prices or len(prices) < self.min_history_days:
                return None
            # 최근이 index 0, 과거로 갈수록 인덱스 증가
            windows = [(60, 30), (30, 15), (15, 7)]
            for entry, exit in windows:
                if len(prices) > entry and len(prices) > exit:
                    entry_price = float(prices[entry].get('close', 0) or 0)
                    exit_price = float(prices[exit].get('close', 0) or 0)
                    if entry_price > 0:
                        return ((exit_price - entry_price) / entry_price) * 100
            return None
        except Exception:
            return None

    def _load_db_2w_returns(self) -> Dict[str, Optional[float]]:
        """sweet_spot_database.json에서 symbol별 현재 2주 수익률(current_return_pct) 로드"""
        try:
            path = "sweet_spot_database.json"
            if not os.path.exists(path):
                return {}
            with open(path, 'r', encoding='utf-8') as f:
                db = json.load(f)
            mapping: Dict[str, Optional[float]] = {}
            for pick in db.get('picks', []):
                sym = pick.get('symbol')
                if not sym:
                    continue
                val = pick.get('current_return_pct')
                # 마지막 값 우선(갱신 데이터로 덮어씀)
                mapping[sym] = val
            return mapping
        except Exception as e:
            logger.warning(f"Failed to load DB 2w returns: {e}")
            return {}
    
    def _get_test_symbols_from_db(self) -> List[str]:
        """Extract test symbols from Sweet Spot DB"""
        try:
            if os.path.exists("sweet_spot_database.json"):
                with open("sweet_spot_database.json", 'r', encoding='utf-8') as f:
                    db = json.load(f)
                    picks = db.get('picks', [])
                    
                    # Recent 90 days picks
                    recent_symbols = []
                    cutoff_date = datetime.now() - timedelta(days=90)
                    
                    for pick in picks:
                        try:
                            selection_date = datetime.strptime(pick['selection_date'], '%Y-%m-%d')
                            if selection_date >= cutoff_date:
                                recent_symbols.append(pick['symbol'])
                        except:
                            continue
                    
                    unique_symbols = list(set(recent_symbols))
                    logger.info(f"Loaded {len(unique_symbols)} symbols from Sweet Spot DB")
                    return unique_symbols
                    
            return []
        except Exception as e:
            logger.error(f"Failed to extract DB symbols: {e}")
            return []

# ================================================================
# Enhanced Genetic Algorithm Optimizer (v6.0 - 60 parameters)
# ================================================================

class GeneticOptimizer:
    """Genetic algorithm optimizer - Sweet Spot v6.0 DETAILED PARAMETERS (60 params)"""
    
    def __init__(self, backtest_engine: RealBacktestEngine):
        self.backtest_engine = backtest_engine
        self.population_size = 40  # 60개 파라미터 충분한 탐색을 위해 증가
        self.max_generations = 1000  # 캐시 데이터 활용: 세대수 확장
        self.mutation_rate = 0.12  # 감소 (안정성 향상)
        self.crossover_rate = 0.80  # 증가 (탐색 강화)
        self.elite_ratio = 0.25    # 증가 (좋은 해 보존)
        self.validator = ParameterValidator()
        self.max_concurrent_evaluations = max(1, int(os.getenv("ML_GA_MAX_CONCURRENCY", "2")))
    
    async def optimize(self) -> Tuple[Optional[Dict], float]:
        """Run genetic algorithm optimization with 60 parameters"""
        try:
            logger.info("🧬 Starting Sweet Spot v6.0 Genetic Algorithm (60 Parameters)")
            logger.info(f"   Settings: Population={self.population_size}, Generations={self.max_generations}")
            logger.info(f"   Parameters: 60개 (메인 6 + 세부 28 + Sweet Spot 5 + 딥테크 8 + 거래량 4 + 서브카테고리 8 + 임계값 4)")
            logger.info(f"   Concurrency limit: {self.max_concurrent_evaluations} individuals")
            
            # Create initial population
            population = [self.create_random_individual() for _ in range(self.population_size)]
            
            best_individual = None
            best_fitness = 0
            fitness_history = []
            stagnation_count = 0
            evaluation_semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)
            
            for generation in range(self.max_generations):
                logger.info(f"\n🔄 Generation {generation+1}/{self.max_generations}")
                
                # Parallel evaluation
                evaluation_tasks = []
                for i, individual in enumerate(population):
                    if individual.fitness == 0:
                        task = asyncio.create_task(
                            self._evaluate_individual(
                                individual,
                                generation,
                                i + 1,
                                evaluation_semaphore
                            )
                        )
                        evaluation_tasks.append((i, task))
                
                # Collect evaluation results
                for i, task in evaluation_tasks:
                    try:
                        fitness = await task
                        population[i].fitness = fitness
                        
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_individual = copy.deepcopy(population[i])
                            stagnation_count = 0
                            logger.info(f"   🎯 New best fitness: {best_fitness:.2f}")
                        
                    except Exception as e:
                        logger.error(f"Individual {i} evaluation failed: {e}")
                        population[i].fitness = 0.1
                
                # Generation statistics
                fitnesses = [ind.fitness for ind in population]
                avg_fitness = np.mean(fitnesses)
                fitness_history.append(avg_fitness)
                
                logger.info(f"   Average fitness: {avg_fitness:.2f}, Best: {max(fitnesses):.2f}")
                
                # Early stopping check (완화된 수렴 조건)
                stagnation_count += 1
                if stagnation_count >= 30 and generation >= 50:  # 500세대 대비 충분한 수렴 기다림
                    logger.info(f"   Early stopping: Converged at generation {generation+1}")
                    self.generations_run = generation + 1
                    break
                
                # Create next generation
                if generation < self.max_generations - 1:
                    population = self.evolve_population(population)
            
            if best_individual:
                # 세대 수 기록 (완주한 경우)
                if not hasattr(self, 'generations_run'):
                    self.generations_run = generation + 1
                logger.info(f"\n✅ Sweet Spot v6.0 Optimization complete!")
                logger.info(f"   Best fitness: {best_fitness:.2f}")
                logger.info(f"   Total parameters optimized: ~60개")
                if best_individual.backtest_result:
                    logger.info(f"   Selected stocks: {best_individual.backtest_result.selected_symbols}")
                    # expose best backtest result for outer summary/logs
                    try:
                        self.best_backtest_result = best_individual.backtest_result
                    except Exception:
                        self.best_backtest_result = None
                
                return best_individual.parameters, best_fitness
            
            return None, 0
            
        except Exception as e:
            logger.error(f"Sweet Spot v6.0 Optimization failed: {e}")
            return None, 0
    
    async def _evaluate_individual(
        self,
        individual: Individual,
        generation: int,
        index: int,
        semaphore: asyncio.Semaphore,
    ) -> float:
        """Evaluate individual with 60 parameters"""
        async with semaphore:
            logger.info(
                "   Evaluating individual %d/%d (60 params, concurrency=%d)",
                index,
                self.population_size,
                self.max_concurrent_evaluations,
            )

            # Validate parameters
            validated_params = self.validator.validate_parameters(individual.parameters)
            individual.parameters = validated_params

            # Run backtest
            result = await self.backtest_engine.run_backtest_with_parameters(validated_params)
            individual.backtest_result = result
            individual.generation = generation

            return result.fitness_score
    
    def create_random_individual(self) -> Individual:
        """Create random individual with ALL 60 parameters"""
        parameters = {
            # 메인 스코어링 가중치 (6개)
            'main_scoring_weights': {
                'pattern_score': random.uniform(0.15, 0.35),
                'convergence_score': random.uniform(0.20, 0.40),
                'growth_score': random.uniform(0.15, 0.35),
                'tech_score': random.uniform(0.10, 0.30),
                'institutional_score': random.uniform(0.05, 0.20),
                'financial_score': random.uniform(0.02, 0.15)
            },
            
            # *** Sweet Spot v6.0 세부 가중치 (28개) ***
            'detailed_scoring_weights': {
                'pattern_scoring': {
                    'crash_depth_weight': random.uniform(0.15, 0.25),
                    'recovery_velocity_weight': random.uniform(0.15, 0.22),
                    'recovery_position_weight': random.uniform(0.12, 0.18),
                    'volatility_compression_weight': random.uniform(0.08, 0.15),
                    'support_strength_weight': random.uniform(0.12, 0.18),
                    'breakout_proximity_weight': random.uniform(0.08, 0.12),
                    'volume_pattern_weight': random.uniform(0.06, 0.10),
                    'pattern_similarity_weight': random.uniform(0.01, 0.04)
                },
                'convergence_scoring': {
                    'rsi_recovery_weight': random.uniform(0.20, 0.30),
                    'macd_timing_weight': random.uniform(0.20, 0.30),
                    'bollinger_squeeze_weight': random.uniform(0.15, 0.25),
                    'moving_avg_convergence_weight': random.uniform(0.10, 0.20),
                    'volume_oscillator_weight': random.uniform(0.08, 0.12),
                    'technical_confluence_weight': random.uniform(0.03, 0.07)
                },
                'growth_scoring': {
                    'revenue_acceleration_weight': random.uniform(0.25, 0.35),
                    'pipeline_strength_weight': random.uniform(0.20, 0.30),
                    'partnership_catalyst_weight': random.uniform(0.15, 0.25),
                    'market_expansion_weight': random.uniform(0.10, 0.20),
                    'regulatory_tailwind_weight': random.uniform(0.08, 0.12)
                },
                'tech_scoring': {
                    'innovation_cycle_position_weight': random.uniform(0.30, 0.40),
                    'tech_adoption_curve_weight': random.uniform(0.20, 0.30),
                    'scaling_readiness_weight': random.uniform(0.20, 0.30),
                    'tech_validation_weight': random.uniform(0.10, 0.20)
                },
                'institutional_scoring': {
                    'institutional_flow_weight': random.uniform(0.40, 0.60),
                    'analyst_momentum_weight': random.uniform(0.25, 0.35),
                    'insider_signal_weight': random.uniform(0.15, 0.25)
                },
                'financial_scoring': {
                    'cash_adequacy_weight': random.uniform(0.60, 0.75),
                    'debt_management_weight': random.uniform(0.25, 0.40)
                }
            },
            
            # Sweet Spot 배수 (5개)
            'sweet_spot_multipliers': {
                'early_recovery_multiplier': random.uniform(1.1, 1.5),
                'mid_recovery_multiplier': random.uniform(0.9, 1.2),
                'late_recovery_multiplier': random.uniform(0.6, 0.9),
                'golden_time_multiplier': random.uniform(1.3, 1.7),
                'overheated_penalty': random.uniform(0.4, 0.7)
            },
            
            # 딥테크 카테고리 배수 (8개)
            'deeptech_category_multipliers': {
                'ai_computing': random.uniform(1.0, 1.3),
                'quantum_tech': random.uniform(1.1, 1.4),
                'bio_health_tech': random.uniform(1.05, 1.35),
                'mobility_tech': random.uniform(1.0, 1.3),
                'semiconductor': random.uniform(0.9, 1.2),
                'energy_materials': random.uniform(0.9, 1.2),
                'security_fintech': random.uniform(0.8, 1.1),
                'emerging_tech': random.uniform(0.95, 1.25)
            },
            
            # 거래량 신호 가중치 (새로 추가 - 4개)
            'volume_signal_weights': {
                'spike_signal_weight': random.uniform(1.15, 1.35),
                'trend_signal_weight': random.uniform(1.10, 1.25),
                'combined_signal_weight': random.uniform(1.25, 1.45),
                'volume_quality_weight': random.uniform(1.10, 1.30)
            },
            
            # 딥테크 서브카테고리 가중치 (새로 추가 - 8개, 각 2개씩 총 16개)
            'deeptech_subcategory_weights': {
                'ai_computing': {
                    'machine_learning': random.uniform(1.10, 1.30),
                    'spatial_computing': random.uniform(1.05, 1.25)
                },
                'mobility_tech': {
                    'evtol': random.uniform(1.20, 1.40),
                    'robotics': random.uniform(1.15, 1.35)
                },
                'bio_health_tech': {
                    'biotech_ai': random.uniform(1.25, 1.45),
                    'neural_interface': random.uniform(1.30, 1.50)
                },
                'energy_materials': {
                    'energy_storage': random.uniform(1.05, 1.25),
                    'new_materials': random.uniform(1.00, 1.20)
                }
            },
            
            # 기존 임계값 (4개)
            'min_crash_percent': random.uniform(15, 30),
            'min_recovery_percent': random.uniform(10, 25),
            'volume_spike_multiplier': random.uniform(1.5, 3.5),
            'volume_trend_multiplier': random.uniform(1.1, 1.5)
        }
        
        # 모든 세부 가중치 정규화
        self.normalize_weights(parameters['main_scoring_weights'])
        for category_weights in parameters['detailed_scoring_weights'].values():
            self.normalize_weights(category_weights)
        
        return Individual(parameters=parameters)
    
    def normalize_weights(self, weights: Dict):
        """Normalize weights to sum to 1"""
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] = weights[key] / total
    
    def evolve_population(self, population: List[Individual]) -> List[Individual]:
        """Evolve population with enhanced diversity"""
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        new_population = []
        
        # Elite preservation (increased for stability)
        elite_size = max(3, int(self.population_size * self.elite_ratio))
        elite = population[:elite_size]
        new_population.extend([copy.deepcopy(ind) for ind in elite])
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(population) >= 2:
                # Crossover
                parent1 = self.tournament_selection(population, tournament_size=4)
                parent2 = self.tournament_selection(population, tournament_size=4)
                child = self.crossover(parent1, parent2)
                
                # Apply mutation
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
            else:
                # Mutation only
                parent = self.tournament_selection(population, tournament_size=3)
                child = self.mutate(parent)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def tournament_selection(self, population: List[Individual], tournament_size: int = 4) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Enhanced crossover with detailed weights support"""
        child_params = copy.deepcopy(parent1.parameters)
        
        # Adaptive blend ratio based on parent fitness
        fitness1 = parent1.fitness
        fitness2 = parent2.fitness
        total_fitness = fitness1 + fitness2
        
        if total_fitness > 0:
            alpha = fitness1 / total_fitness
        else:
            alpha = 0.5
        
        # Blend main weights
        for key in child_params['main_scoring_weights']:
            child_params['main_scoring_weights'][key] = (
                alpha * parent1.parameters['main_scoring_weights'][key] +
                (1 - alpha) * parent2.parameters['main_scoring_weights'][key]
            )
        
        # *** Sweet Spot v6.0 세부 가중치 교배 ***
        if 'detailed_scoring_weights' in parent1.parameters and 'detailed_scoring_weights' in parent2.parameters:
            for category in child_params['detailed_scoring_weights']:
                for weight_key in child_params['detailed_scoring_weights'][category]:
                    child_params['detailed_scoring_weights'][category][weight_key] = (
                        alpha * parent1.parameters['detailed_scoring_weights'][category][weight_key] +
                        (1 - alpha) * parent2.parameters['detailed_scoring_weights'][category][weight_key]
                    )
        
        # Blend Sweet Spot multipliers
        for key in child_params['sweet_spot_multipliers']:
            child_params['sweet_spot_multipliers'][key] = (
                alpha * parent1.parameters['sweet_spot_multipliers'][key] +
                (1 - alpha) * parent2.parameters['sweet_spot_multipliers'][key]
            )
        
        # Blend Deep Tech multipliers
        for key in child_params['deeptech_category_multipliers']:
            child_params['deeptech_category_multipliers'][key] = (
                alpha * parent1.parameters['deeptech_category_multipliers'][key] +
                (1 - alpha) * parent2.parameters['deeptech_category_multipliers'][key]
            )
        
        # Blend volume signal weights (새로 추가)
        if 'volume_signal_weights' in child_params:
            for key in child_params['volume_signal_weights']:
                child_params['volume_signal_weights'][key] = (
                    alpha * parent1.parameters['volume_signal_weights'][key] +
                    (1 - alpha) * parent2.parameters['volume_signal_weights'][key]
                )
        
        # Blend deeptech subcategory weights (새로 추가)
        if 'deeptech_subcategory_weights' in child_params:
            for category, subcategories in child_params['deeptech_subcategory_weights'].items():
                if isinstance(subcategories, dict):
                    for subcat_key in subcategories:
                        child_params['deeptech_subcategory_weights'][category][subcat_key] = (
                            alpha * parent1.parameters['deeptech_subcategory_weights'][category][subcat_key] +
                            (1 - alpha) * parent2.parameters['deeptech_subcategory_weights'][category][subcat_key]
                        )
        
        # Blend thresholds
        child_params['min_crash_percent'] = (
            alpha * parent1.parameters['min_crash_percent'] +
            (1 - alpha) * parent2.parameters['min_crash_percent']
        )
        child_params['min_recovery_percent'] = (
            alpha * parent1.parameters['min_recovery_percent'] +
            (1 - alpha) * parent2.parameters['min_recovery_percent']
        )
        child_params['volume_spike_multiplier'] = (
            alpha * parent1.parameters['volume_spike_multiplier'] +
            (1 - alpha) * parent2.parameters['volume_spike_multiplier']
        )
        child_params['volume_trend_multiplier'] = (
            alpha * parent1.parameters['volume_trend_multiplier'] +
            (1 - alpha) * parent2.parameters['volume_trend_multiplier']
        )
        
        # Normalize all weights
        self.normalize_weights(child_params['main_scoring_weights'])
        if 'detailed_scoring_weights' in child_params:
            for category_weights in child_params['detailed_scoring_weights'].values():
                self.normalize_weights(category_weights)
        
        return Individual(parameters=child_params)
    
    def mutate(self, individual: Individual) -> Individual:
        """Enhanced mutation with detailed weights support"""
        new_params = copy.deepcopy(individual.parameters)
        mutation_strength = 0.10  # 더 보수적인 변이 (복잡도 증가로 인해)
        
        # Main weight mutation
        if random.random() < self.mutation_rate * 1.5:
            weights = new_params['main_scoring_weights']
            num_mutations = random.randint(1, 2)  # 더 적은 변이
            for _ in range(num_mutations):
                key = random.choice(list(weights.keys()))
                factor = random.uniform(1 - mutation_strength, 1 + mutation_strength)
                weights[key] *= factor
            self.normalize_weights(weights)
        
        # *** Sweet Spot v6.0 세부 가중치 돌연변이 ***
        if 'detailed_scoring_weights' in new_params and random.random() < self.mutation_rate * 2.0:
            # 랜덤하게 1-2개 카테고리 선택
            categories = list(new_params['detailed_scoring_weights'].keys())
            selected_categories = random.sample(categories, min(2, len(categories)))
            
            for category in selected_categories:
                category_weights = new_params['detailed_scoring_weights'][category]
                # 각 카테고리에서 1-2개 가중치 변경
                num_mutations = random.randint(1, 2)
                weight_keys = list(category_weights.keys())
                
                for _ in range(num_mutations):
                    key = random.choice(weight_keys)
                    factor = random.uniform(1 - mutation_strength, 1 + mutation_strength)
                    category_weights[key] *= factor
                
                # 카테고리별 정규화
                self.normalize_weights(category_weights)
        
        # Sweet Spot multiplier mutation
        if random.random() < self.mutation_rate:
            multipliers = new_params['sweet_spot_multipliers']
            key = random.choice(list(multipliers.keys()))
            factor = random.uniform(1 - mutation_strength, 1 + mutation_strength)
            multipliers[key] *= factor
            # Ensure penalty stays < 1
            if 'penalty' in key:
                multipliers[key] = min(multipliers[key], 0.9)
        
        # Deep Tech multiplier mutation
        if random.random() < self.mutation_rate:
            tech_multipliers = new_params['deeptech_category_multipliers']
            key = random.choice(list(tech_multipliers.keys()))
            factor = random.uniform(1 - mutation_strength, 1 + mutation_strength)
            tech_multipliers[key] *= factor
        
        # Volume signal weights mutation (새로 추가)
        if 'volume_signal_weights' in new_params and random.random() < self.mutation_rate:
            volume_weights = new_params['volume_signal_weights']
            key = random.choice(list(volume_weights.keys()))
            factor = random.uniform(1 - mutation_strength, 1 + mutation_strength)
            volume_weights[key] *= factor
        
        # Deeptech subcategory weights mutation (새로 추가)
        if 'deeptech_subcategory_weights' in new_params and random.random() < self.mutation_rate:
            # 랜덤 카테고리 선택
            categories = list(new_params['deeptech_subcategory_weights'].keys())
            category = random.choice(categories)
            subcategories = new_params['deeptech_subcategory_weights'][category]
            if isinstance(subcategories, dict):
                subcat_key = random.choice(list(subcategories.keys()))
                factor = random.uniform(1 - mutation_strength, 1 + mutation_strength)
                subcategories[subcat_key] *= factor
        
        # Threshold mutation
        if random.random() < self.mutation_rate:
            param_keys = ['min_crash_percent', 'min_recovery_percent', 'volume_spike_multiplier', 'volume_trend_multiplier']
            key = random.choice(param_keys)
            factor = random.uniform(1 - mutation_strength, 1 + mutation_strength)
            new_params[key] *= factor
        
        return Individual(parameters=new_params)

# ================================================================
# Main Optimization Class - HYBRID COMPATIBLE v6.0
# ================================================================

class MLOptimizer:
    """ML optimization main class - Sweet Spot v6.0 HYBRID COMPATIBLE (60 parameters)"""
    
    def __init__(self):
        self.param_manager = MLParameterManager()
        self.backtest_engine = RealBacktestEngine()
        self.genetic_optimizer = GeneticOptimizer(self.backtest_engine)
        
    async def run_optimization(self) -> OptimizationResult:
        """*** HYBRID COMPATIBLE: Returns OptimizationResult with 60 parameters ***"""
        try:
            start_time = datetime.now()
            logger.info("="*60)
            logger.info("🚀 Sweet Spot v6.0 ML Parameter Optimization Starting (60 Parameters)")
            logger.info(f"   Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("   Mode: REAL DATA ONLY (No Mock Data)")
            logger.info("   Parameters: 60개 세부 파라미터 최적화")
            logger.info("="*60)
            
            # Initialize backtesting engine
            await self.backtest_engine.initialize()
            # Enable training-only scoring adjustments (e.g., 10d momentum)
            try:
                self.backtest_engine.set_training_mode(True)
            except Exception:
                pass
            
            # Test current parameters
            current_params = self.param_manager.parameters.get('current_parameters', {})
            baseline_fitness = 0
            if current_params:
                logger.info("📊 Evaluating current parameters...")
                baseline_result = await self.backtest_engine.run_backtest_with_parameters(current_params)
                baseline_fitness = baseline_result.fitness_score
                logger.info(f"   Baseline fitness: {baseline_fitness:.2f}")
            
            # Run genetic algorithm optimization
            best_params, best_fitness = await self.genetic_optimizer.optimize()
            
            # Calculate improvement
            improvement = best_fitness - baseline_fitness if baseline_fitness > 0 else best_fitness
            
            # Save results if significant improvement
            if best_params and best_fitness > 35:
                if improvement > 5 or baseline_fitness == 0:
                    await self.param_manager.update_parameters(best_params, best_fitness)
                    
                    logger.info("\n" + "="*60)
                    logger.info("📊 Sweet Spot v6.0 Optimization Results")
                    logger.info("="*60)
                    logger.info(f"✅ Best fitness: {best_fitness:.2f}")
                    # 최종 요약 로그 추가
                    try:
                        gen_run = getattr(self.genetic_optimizer, 'generations_run', None)
                        summary = self.backtest_engine.last_backtest_summary if isinstance(self.backtest_engine.last_backtest_summary, dict) else {}
                        logger.info("\n📌 Optimization Summary")
                        if gen_run:
                            logger.info(f"   • Generations run: {gen_run}")
                        if summary:
                            logger.info(f"   • DB symbols (recent 90d): {summary.get('db_symbols_loaded','N/A')}")
                            logger.info(f"   • Tested symbols cap: {summary.get('tested_symbols','N/A')}")
                            logger.info(f"   • Score ≥ {summary.get('score_threshold', 'N/A')}: {summary.get('score_passed','N/A')}")
                            logger.info(f"   • Picks with returns: {summary.get('picks','N/A')} (period {summary.get('backtest_period','')})")
                        if self.genetic_optimizer and self.genetic_optimizer.backtest_engine and best_params:
                            sel = getattr(self.genetic_optimizer, 'backtest_engine', None)
                        # best individual's selected list
                        if best_params and self.genetic_optimizer and self.genetic_optimizer.backtest_engine:
                            pass
                        # Alignment summary (Baseline vs Optimized)
                        try:
                            base_res = locals().get('baseline_result', None)
                            best_res = getattr(self.genetic_optimizer, 'best_backtest_result', None)
                            if base_res or best_res:
                                logger.info("\n🔗 Score ↔ 2주 수익 정렬력 (Baseline → Optimized)")
                                def _fmt(v):
                                    try:
                                        return f"{v:.2f}" if v is not None else "N/A"
                                    except Exception:
                                        return "N/A"
                                b_na = base_res.alignment_n_all if base_res else 0
                                o_na = best_res.alignment_n_all if best_res else 0
                                b_sa = _fmt(base_res.alignment_spearman_all if base_res else None)
                                o_sa = _fmt(best_res.alignment_spearman_all if best_res else None)
                                b_pa = _fmt(base_res.alignment_pearson_all if base_res else None)
                                o_pa = _fmt(best_res.alignment_pearson_all if best_res else None)
                                b_nt = base_res.alignment_n_topk if base_res else 0
                                o_nt = best_res.alignment_n_topk if best_res else 0
                                b_st = _fmt(base_res.alignment_spearman_topk if base_res else None)
                                o_st = _fmt(best_res.alignment_spearman_topk if best_res else None)
                                b_pt = _fmt(base_res.alignment_pearson_topk if base_res else None)
                                o_pt = _fmt(best_res.alignment_pearson_topk if best_res else None)
                                logger.info(f"   All: n {b_na} → {o_na}, Spearman {b_sa} → {o_sa}, Pearson {b_pa} → {o_pa}")
                                logger.info(f"   Top-{self.backtest_engine.top_k or 'All'}: n {b_nt} → {o_nt}, Spearman {b_st} → {o_st}, Pearson {b_pt} → {o_pt}")
                        except Exception as _e2:
                            logger.debug(f"Alignment summary skipped: {_e2}")
                    except Exception as _e:
                        logger.debug(f"Summary log skipped: {_e}")
                    logger.info(f"   Improvement: {improvement:+.2f} points")
                    logger.info(f"✅ New 60 parameters saved")
                    
                    # Display key parameters
                    weights = best_params.get('main_scoring_weights', {})
                    logger.info("\n📈 Optimal main weights:")
                    for key, value in weights.items():
                        logger.info(f"   {key}: {value:.3f}")
                    
                    # Display detailed weights summary
                    if 'detailed_scoring_weights' in best_params:
                        detailed_count = sum(len(cat_weights) for cat_weights in best_params['detailed_scoring_weights'].values())
                        logger.info(f"\n🔬 Detailed weights optimized: {detailed_count}개")
                        
                else:
                    logger.info(f"⚠️ Low improvement ({improvement:.2f} points), keeping current parameters")
            else:
                logger.warning("⚠️ Optimization failed or performance below threshold")
                # Use current parameters as fallback
                best_params = current_params
                best_fitness = baseline_fitness
            
            # Performance statistics
            cache_stats = self.backtest_engine.cache.get_stats()
            logger.info(f"\n💾 Cache performance:")
            logger.info(f"   Memory hit rate: {cache_stats['memory']['hit_rate']:.1f}%")
            logger.info(f"   Disk hit rate: {cache_stats['disk']['hit_rate']:.1f}%")
            
            rate_limiter = self.backtest_engine.rate_limiter
            logger.info(f"   Total API calls: {rate_limiter.total_calls}")
            
            # Elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
            
            # Cleanup
            await self.backtest_engine.cleanup()
            try:
                self.backtest_engine.set_training_mode(False)
            except Exception:
                pass
            
            # *** HYBRID COMPATIBLE: Return OptimizationResult ***
            confidence = min(1.0, best_fitness / 100.0)
            convergence_gen = self.genetic_optimizer.max_generations  # or actual convergence gen
            
            return OptimizationResult(
                best_parameters=best_params or {},
                fitness_score=best_fitness,
                optimization_confidence=confidence,
                expected_improvement=max(0, improvement),
                convergence_generation=convergence_gen
            )
            
        except Exception as e:
            logger.error(f"Sweet Spot v6.0 Optimization process error: {e}")
            await self.backtest_engine.cleanup()
            
            # Return failure result
            return OptimizationResult(
                best_parameters={},
                fitness_score=0.0,
                optimization_confidence=0.0,
                expected_improvement=0.0,
                convergence_generation=0
            )

# ================================================================
# Entry Point
# ================================================================

async def main():
    """Main execution function"""
    optimizer = MLOptimizer()
    result = await optimizer.run_optimization()
    
    print(f"\n=== Sweet Spot v6.0 Optimization Result ===")
    print(f"Fitness Score: {result.fitness_score:.2f}")
    print(f"Confidence: {result.optimization_confidence:.1%}")
    print(f"Expected Improvement: {result.expected_improvement:.1%}")
    print(f"Convergence Generation: {result.convergence_generation}")
    print(f"Parameters Optimized: ~60개 (세부 가중치 포함)")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧬 Sweet Spot v6.0 Enhanced Real Data ML Optimizer")
    print("   - 60개 파라미터 최적화 (세부 가중치 포함)")
    print("   - HYBRID COMPATIBLE VERSION")
    print("   - NO MOCK DATA - Real API data only")
    print("   - Enhanced genetic algorithm (25 pop, 30 gen)")
    print("   - 세부 스코어링 가중치 지원")
    print("   - Returns OptimizationResult for Hybrid")
    print("="*60 + "\n")
    
    try:
        asyncio.run(main())
        print("\n✅ Sweet Spot v6.0 Program completed successfully")
    except KeyboardInterrupt:
        print("\n⚠️ User interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
