#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 설정 관리자
Enhanced 퀀트 시스템용 설정 클래스들
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from pydantic import BaseSettings, validator, Field
from .env_loader import get_env_loader, EnvironmentLoader

logger = logging.getLogger(__name__)

class AppSettings(BaseSettings):
    """앱 기본 설정"""
    
    # 기본 설정
    environment: str = Field(default="development", description="실행 환경")
    debug: bool = Field(default=True, description="디버그 모드")
    log_level: str = Field(default="INFO", description="로그 레벨")
    
    # 서버 설정
    host: str = Field(default="0.0.0.0", description="서버 호스트")
    port: int = Field(default=7860, description="서버 포트")
    share: bool = Field(default=False, description="Gradio 공유 활성화")
    
    # 보안 설정
    cors_origins: List[str] = Field(default=["*"], description="CORS 허용 도메인")
    max_request_size: int = Field(default=10485760, description="최대 요청 크기 (10MB)")
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'production', 'testing']
        if v.lower() not in allowed:
            raise ValueError(f'environment must be one of {allowed}')
        return v.lower()
    
    @validator('log_level')
    def validate_log_level(cls, v):
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f'log_level must be one of {allowed}')
        return v.upper()
    
    @validator('port')
    def validate_port(cls, v):
        if not (1000 <= v <= 65535):
            raise ValueError('port must be between 1000 and 65535')
        return v
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

class APISettings(BaseSettings):
    """API 설정"""
    
    # 한국투자증권 API
    kis_app_key: Optional[str] = Field(default=None, description="KIS 앱 키")
    kis_app_secret: Optional[str] = Field(default=None, description="KIS 앱 시크릿")
    kis_base_url: str = Field(default="https://openapivts.koreainvestment.com:29443", description="KIS API 기본 URL")
    
    # AI API들
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API 키")
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta", description="Gemini API 기본 URL")
    
    perplexity_api_key: Optional[str] = Field(default=None, description="Perplexity API 키")
    perplexity_base_url: str = Field(default="https://api.perplexity.ai", description="Perplexity API 기본 URL")
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API 키")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API 기본 URL")
    
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API 키")
    anthropic_base_url: str = Field(default="https://api.anthropic.com", description="Anthropic API 기본 URL")
    
    # API 공통 설정
    api_timeout: int = Field(default=30, description="API 타임아웃 (초)")
    retry_count: int = Field(default=3, description="재시도 횟수")
    retry_delay: float = Field(default=1.0, description="재시도 지연 (초)")
    rate_limit_calls: int = Field(default=100, description="분당 API 호출 제한")
    
    @validator('*', pre=True)
    def load_from_env(cls, v, field):
        """환경변수에서 값 로드 (별칭 포함)"""
        if v is not None:
            return v
        
        # 별칭 매핑
        aliases = {
            'kis_app_key': ['MKEY', 'KIS_APP_KEY', 'KIS_KEY'],
            'kis_app_secret': ['MKEYS', 'KIS_APP_SECRET', 'KIS_SECRET'],
            'gemini_api_key': ['GEM', 'GEMINI_API_KEY', 'GEMINI_KEY'],
            'perplexity_api_key': ['PPL', 'PERPLEXITY_API_KEY', 'PERPLEXITY_KEY'],
            'openai_api_key': ['OPENAI_KEY', 'OPENAI_API_KEY'],
            'anthropic_api_key': ['ANTHROPIC_KEY', 'ANTHROPIC_API_KEY']
        }
        
        field_name = field.name
        if field_name in aliases:
            for alias in aliases[field_name]:
                value = os.getenv(alias)
                if value:
                    return value
        
        return os.getenv(field_name.upper())
    
    @validator('api_timeout')
    def validate_timeout(cls, v):
        if not (5 <= v <= 300):
            raise ValueError('api_timeout must be between 5 and 300 seconds')
        return v
    
    @validator('retry_count')
    def validate_retry_count(cls, v):
        if not (1 <= v <= 10):
            raise ValueError('retry_count must be between 1 and 10')
        return v
    
    def get_kis_config(self) -> Dict[str, str]:
        """한투 API 설정 반환"""
        return {
            'app_key': self.kis_app_key or '',
            'app_secret': self.kis_app_secret or '',
            'base_url': self.kis_base_url
        }
    
    def get_gemini_config(self) -> Dict[str, str]:
        """Gemini API 설정 반환"""
        return {
            'api_key': self.gemini_api_key or '',
            'base_url': self.gemini_base_url
        }
    
    def get_perplexity_config(self) -> Dict[str, str]:
        """Perplexity API 설정 반환"""
        return {
            'api_key': self.perplexity_api_key or '',
            'base_url': self.perplexity_base_url
        }
    
    def get_openai_config(self) -> Dict[str, str]:
        """OpenAI API 설정 반환"""
        return {
            'api_key': self.openai_api_key or '',
            'base_url': self.openai_base_url
        }
    
    def get_anthropic_config(self) -> Dict[str, str]:
        """Anthropic API 설정 반환"""
        return {
            'api_key': self.anthropic_api_key or '',
            'base_url': self.anthropic_base_url
        }
    
    def validate_required_keys(self) -> Dict[str, bool]:
        """필수 API 키 검증"""
        return {
            'kis': bool(self.kis_app_key and self.kis_app_secret),
            'gemini': bool(self.gemini_api_key),
            'perplexity': bool(self.perplexity_api_key),
            'openai': bool(self.openai_api_key),
            'anthropic': bool(self.anthropic_api_key)
        }
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

class AnalysisSettings(BaseSettings):
    """분석 설정"""
    
    # 종목 분석 설정
    max_symbols: int = Field(default=10, description="최대 분석 종목 수")
    cache_hours: int = Field(default=48, description="캐시 유지 시간")
    strict_validation: bool = Field(default=True, description="엄격한 데이터 검증")
    
    # VTNF 설정
    default_v_weight: float = Field(default=0.30, description="V 기본 가중치")
    default_t_weight: float = Field(default=0.30, description="T 기본 가중치")
    default_n_weight: float = Field(default=0.25, description="N 기본 가중치")
    default_f_weight: float = Field(default=0.15, description="F 기본 가중치")
    
    # 부스터 설정
    gap_filter_enabled: bool = Field(default=True, description="갭필터 부스터 활성화")
    pattern_enabled: bool = Field(default=True, description="패턴 인식 활성화")
    thinking_budget: int = Field(default=5000, description="AI 사고 예산")
    
    # 임계값 설정
    buy_threshold: float = Field(default=7.5, description="매수 임계값")
    sell_threshold: float = Field(default=4.5, description="매도 임계값")
    gap_significance_threshold: float = Field(default=0.5, description="갭 유의성 임계값")
    pattern_confidence_threshold: float = Field(default=70.0, description="패턴 신뢰도 임계값")
    
    # 캐시 설정
    sector_cache_hours: int = Field(default=48, description="섹터 캐시 시간")
    gap_cache_minutes: int = Field(default=10, description="갭 캐시 시간 (분)")
    pattern_cache_minutes: int = Field(default=5, description="패턴 캐시 시간 (분)")
    
    @validator('max_symbols')
    def validate_max_symbols(cls, v):
        if not (1 <= v <= 50):
            raise ValueError('max_symbols must be between 1 and 50')
        return v
    
    @validator('cache_hours')
    def validate_cache_hours(cls, v):
        if not (1 <= v <= 168):  # 최대 1주일
            raise ValueError('cache_hours must be between 1 and 168')
        return v
    
    @validator('thinking_budget')
    def validate_thinking_budget(cls, v):
        if not (1000 <= v <= 50000):
            raise ValueError('thinking_budget must be between 1000 and 50000')
        return v
    
    @validator('default_v_weight', 'default_t_weight', 'default_n_weight', 'default_f_weight')
    def validate_weights(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('weights must be between 0.0 and 1.0')
        return v
    
    @validator('buy_threshold', 'sell_threshold')
    def validate_thresholds(cls, v):
        if not (0.0 <= v <= 10.0):
            raise ValueError('thresholds must be between 0.0 and 10.0')
        return v
    
    def validate_weight_sum(self) -> bool:
        """가중치 합계 검증"""
        total = self.default_v_weight + self.default_t_weight + self.default_n_weight + self.default_f_weight
        return abs(total - 1.0) < 0.01  # 소수점 오차 허용
    
    def get_default_weights(self) -> Dict[str, float]:
        """기본 VTNF 가중치 반환"""
        return {
            'V': self.default_v_weight,
            'T': self.default_t_weight,
            'N': self.default_n_weight,
            'F': self.default_f_weight
        }
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

class DatabaseSettings(BaseSettings):
    """데이터베이스 설정 (향후 확장용)"""
    
    # 파일 기반 저장소
    data_dir: str = Field(default="data", description="데이터 디렉토리")
    cache_dir: str = Field(default="cache", description="캐시 디렉토리")
    logs_dir: str = Field(default="logs", description="로그 디렉토리")
    
    # SQLite 설정
    sqlite_db_path: str = Field(default="data/quant_system.db", description="SQLite DB 경로")
    sqlite_timeout: int = Field(default=30, description="SQLite 타임아웃")
    
    # Redis 설정 (선택사항)
    redis_host: str = Field(default="localhost", description="Redis 호스트")
    redis_port: int = Field(default=6379, description="Redis 포트")
    redis_password: Optional[str] = Field(default=None, description="Redis 비밀번호")
    redis_db: int = Field(default=0, description="Redis DB 번호")
    
    def ensure_directories(self):
        """필요한 디렉토리 생성"""
        for dir_name in [self.data_dir, self.cache_dir, self.logs_dir]:
            Path(dir_name).mkdir(exist_ok=True)
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

class GlobalSettings:
    """전역 설정 관리자"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self.env_loader: Optional[EnvironmentLoader] = None
        
        # 환경변수 로드
        self._load_environment()
        
        # 설정 객체들 생성
        self.app = AppSettings()
        self.api = APISettings()
        self.analysis = AnalysisSettings()
        self.database = DatabaseSettings()
        
        # 디렉토리 생성
        self.database.ensure_directories()
        
        logger.info("⚙️ 전역 설정 초기화 완료")
    
    def _load_environment(self):
        """환경변수 로드"""
        try:
            from .env_loader import get_env_loader
            self.env_loader = get_env_loader()
            self.env_loader.env_file = Path(self.env_file)
            success = self.env_loader.load_env_file(strict_mode=False)
            
            if success:
                logger.info("✅ 환경변수 로드 성공")
            else:
                logger.warning("⚠️ 환경변수 로드 부분 실패")
                
        except Exception as e:
            logger.error(f"❌ 환경변수 로드 실패: {e}")
    
    def validate_all(self) -> bool:
        """모든 설정 검증"""
        try:
            validation_results = []
            
            # API 키 검증
            api_keys = self.api.validate_required_keys()
            kis_valid = api_keys['kis']
            gemini_valid = api_keys['gemini']
            perplexity_valid = api_keys['perplexity']
            
            if not (kis_valid and gemini_valid and perplexity_valid):
                missing = [k for k, v in api_keys.items() if not v and k in ['kis', 'gemini', 'perplexity']]
                logger.error(f"❌ 필수 API 키 누락: {missing}")
                validation_results.append(False)
            else:
                logger.info("✅ 필수 API 키 검증 완료")
                validation_results.append(True)
            
            # VTNF 가중치 검증
            if not self.analysis.validate_weight_sum():
                logger.error("❌ VTNF 가중치 합계가 1.0이 아님")
                validation_results.append(False)
            else:
                logger.info("✅ VTNF 가중치 검증 완료")
                validation_results.append(True)
            
            # 포트 충돌 검사
            if self.app.port in [80, 443, 22, 21, 25]:
                logger.warning(f"⚠️ 시스템 포트 사용 중: {self.app.port}")
            
            return all(validation_results)
            
        except Exception as e:
            logger.error(f"❌ 설정 검증 중 오류: {e}")
            return False
    
    def export_config(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """설정 내보내기"""
        config = {
            'app': self.app.dict(),
            'analysis': self.analysis.dict(),
            'database': self.database.dict(),
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'environment': self.app.environment,
                'version': '1.8.2'
            }
        }
        
        if include_sensitive:
            config['api'] = self.api.dict()
        else:
            # 민감 정보 마스킹
            api_dict = self.api.dict()
            sensitive_keys = ['api_key', 'app_key', 'app_secret']
            
            for key, value in api_dict.items():
                if any(sensitive in key for sensitive in sensitive_keys) and value:
                    api_dict[key] = f"{'*' * 8}...{value[-4:]}" if len(value) > 4 else "****"
            
            config['api'] = api_dict
        
        return config
    
    def save_config(self, file_path: str = "config/current_settings.json"):
        """설정을 파일로 저장"""
        try:
            config_path = Path(file_path)
            config_path.parent.mkdir(exist_ok=True)
            
            config = self.export_config(include_sensitive=False)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 설정 저장 완료: {config_path}")
            
        except Exception as e:
            logger.error(f"❌ 설정 저장 실패: {e}")
    
    def load_config(self, file_path: str = "config/current_settings.json") -> bool:
        """파일에서 설정 로드"""
        try:
            config_path = Path(file_path)
            if not config_path.exists():
                logger.warning(f"⚠️ 설정 파일 없음: {config_path}")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 설정 적용 (API 키는 제외)
            if 'app' in config:
                for key, value in config['app'].items():
                    if hasattr(self.app, key):
                        setattr(self.app, key, value)
            
            if 'analysis' in config:
                for key, value in config['analysis'].items():
                    if hasattr(self.analysis, key):
                        setattr(self.analysis, key, value)
            
            logger.info(f"📂 설정 로드 완료: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 설정 로드 실패: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            'version': '1.8.2',
            'environment': self.app.environment,
            'debug_mode': self.app.debug,
            'server': {
                'host': self.app.host,
                'port': self.app.port,
                'share': self.app.share
            },
            'analysis': {
                'max_symbols': self.analysis.max_symbols,
                'cache_hours': self.analysis.cache_hours,
                'gap_filter_enabled': self.analysis.gap_filter_enabled,
                'pattern_enabled': self.analysis.pattern_enabled
            },
            'api_status': self.api.validate_required_keys(),
            'directories': {
                'data': self.database.data_dir,
                'cache': self.database.cache_dir,
                'logs': self.database.logs_dir
            }
        }
    
    def print_summary(self):
        """설정 요약 출력"""
        print("\n⚙️ Enhanced 퀀트 시스템 설정 요약")
        print("=" * 60)
        
        print(f"🌍 환경: {self.app.environment}")
        print(f"🏠 서버: {self.app.host}:{self.app.port}")
        print(f"📊 최대 종목: {self.analysis.max_symbols}개")
        print(f"💾 캐시: {self.analysis.cache_hours}시간")
        
        print(f"\n🔧 부스터 상태:")
        print(f"  갭필터: {'✅' if self.analysis.gap_filter_enabled else '❌'}")
        print(f"  패턴인식: {'✅' if self.analysis.pattern_enabled else '❌'}")
        
        print(f"\n🔑 API 상태:")
        api_status = self.api.validate_required_keys()
        for service, status in api_status.items():
            print(f"  {service}: {'✅' if status else '❌'}")
        
        print(f"\n📁 디렉토리:")
        print(f"  데이터: {self.database.data_dir}")
        print(f"  캐시: {self.database.cache_dir}")
        print(f"  로그: {self.database.logs_dir}")

# 전역 설정 인스턴스
_global_settings_instance: Optional[GlobalSettings] = None

def get_settings() -> GlobalSettings:
    """전역 설정 싱글톤 인스턴스"""
    global _global_settings_instance
    if _global_settings_instance is None:
        _global_settings_instance = GlobalSettings()
    return _global_settings_instance

def reload_settings(env_file: str = ".env") -> GlobalSettings:
    """설정 재로드"""
    global _global_settings_instance
    _global_settings_instance = GlobalSettings(env_file)
    return _global_settings_instance

# 편의 함수들
def get_app_settings() -> AppSettings:
    """앱 설정 반환"""
    return get_settings().app

def get_api_settings() -> APISettings:
    """API 설정 반환"""
    return get_settings().api

def get_analysis_settings() -> AnalysisSettings:
    """분석 설정 반환"""
    return get_settings().analysis

def get_database_settings() -> DatabaseSettings:
    """데이터베이스 설정 반환"""
    return get_settings().database

if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 설정 시스템 테스트")
    print("=" * 50)
    
    # 설정 초기화
    settings = GlobalSettings()
    
    # 설정 검증
    is_valid = settings.validate_all()
    print(f"설정 검증: {'✅ 성공' if is_valid else '❌ 실패'}")
    
    # 설정 요약 출력
    settings.print_summary()
    
    # 설정 저장/로드 테스트
    settings.save_config("test_settings.json")
    load_success = settings.load_config("test_settings.json")
    print(f"\n설정 저장/로드: {'✅ 성공' if load_success else '❌ 실패'}")
    
    # 시스템 정보
    system_info = settings.get_system_info()
    print(f"\n시스템 정보: {json.dumps(system_info, indent=2, ensure_ascii=False)}")
