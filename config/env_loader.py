#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
환경 변수 로드 모듈 (개선된 버전)
Enhanced 퀀트 시스템용 환경설정 관리
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# 환경변수 별칭 매핑
ENV_ALIASES = {
    'KIS_APP_KEY': ['MKEY', 'KIS_APP_KEY', 'KIS_KEY'],
    'KIS_APP_SECRET': ['MKEYS', 'KIS_APP_SECRET', 'KIS_SECRET'],
    'GEMINI_API_KEY': ['GEM', 'GEMINI_API_KEY', 'GEMINI_KEY'],
    'PERPLEXITY_API_KEY': ['PPL', 'PERPLEXITY_API_KEY', 'PERPLEXITY_KEY'],
    'OPENAI_API_KEY': ['OPENAI_KEY', 'OPENAI_API_KEY'],
    'ANTHROPIC_API_KEY': ['ANTHROPIC_KEY', 'ANTHROPIC_API_KEY']
}

# 필수 환경변수
REQUIRED_VARIABLES = [
    'KIS_APP_KEY', 'KIS_APP_SECRET', 
    'GEMINI_API_KEY', 'PERPLEXITY_API_KEY'
]

# 선택적 환경변수
OPTIONAL_VARIABLES = [
    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
    'ENVIRONMENT', 'DEBUG', 'LOG_LEVEL',
    'HOST', 'PORT', 'SHARE',
    'MAX_SYMBOLS', 'CACHE_HOURS',
    'THINKING_BUDGET', 'PATTERN_ENABLED'
]

# 기본값
DEFAULT_VALUES = {
    'ENVIRONMENT': 'development',
    'DEBUG': 'true',
    'LOG_LEVEL': 'INFO',
    'HOST': '0.0.0.0',
    'PORT': '7860',
    'SHARE': 'false',
    'MAX_SYMBOLS': '10',
    'CACHE_HOURS': '48',
    'THINKING_BUDGET': '5000',
    'PATTERN_ENABLED': 'true',
    'STRICT_VALIDATION': 'true',
    'USE_CACHE': 'true',
    'API_TIMEOUT': '30',
    'RETRY_COUNT': '3'
}

class EnvironmentLoader:
    """환경변수 로더 클래스"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self.loaded_vars: Dict[str, str] = {}
        self.missing_vars: List[str] = []
        self.validation_errors: List[str] = []
        self.load_timestamp: Optional[datetime] = None
        
    def load_env_file(self, strict_mode: bool = False) -> bool:
        """환경 변수 파일 로드 with 검증"""
        try:
            logger.info(f"🔧 환경변수 파일 로드 시작: {self.env_file}")
            
            # 1. .env 파일 로드
            if self.env_file.exists():
                self._load_from_file()
                logger.info(f"✅ {self.env_file} 파일 로드 완료")
            else:
                logger.warning(f"⚠️ {self.env_file} 파일 없음")
                if strict_mode:
                    self._create_sample_env_file()
                    return False
            
            # 2. 시스템 환경변수도 확인
            self._load_from_system()
            
            # 3. 별칭 처리
            self._resolve_aliases()
            
            # 4. 기본값 적용
            self._apply_defaults()
            
            # 5. 검증
            validation_result = self._validate_variables(strict_mode)
            
            # 6. 환경변수 설정
            self._set_environment_variables()
            
            self.load_timestamp = datetime.now()
            
            if validation_result:
                logger.info("✅ 환경변수 로드 및 검증 완료")
                return True
            else:
                if strict_mode:
                    logger.error("❌ 환경변수 검증 실패 (strict mode)")
                    return False
                else:
                    logger.warning("⚠️ 일부 환경변수 누락, 기본값 사용")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ 환경변수 로드 실패: {e}")
            return False
    
    def _load_from_file(self):
        """파일에서 환경변수 로드"""
        try:
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 빈 줄이나 주석 스킵
                    if not line or line.startswith('#'):
                        continue
                    
                    # KEY=VALUE 파싱
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 따옴표 제거
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        self.loaded_vars[key] = value
                        logger.debug(f"로드: {key}={'*' * min(len(value), 8)}")
                    else:
                        logger.warning(f"잘못된 형식 (라인 {line_num}): {line}")
                        
        except Exception as e:
            logger.error(f"파일 로드 오류: {e}")
            raise
    
    def _load_from_system(self):
        """시스템 환경변수에서 로드 (우선순위 높음)"""
        for key in REQUIRED_VARIABLES + OPTIONAL_VARIABLES:
            if key in os.environ:
                self.loaded_vars[key] = os.environ[key]
                logger.debug(f"시스템에서 로드: {key}")
    
    def _resolve_aliases(self):
        """환경변수 별칭 해결"""
        for main_key, aliases in ENV_ALIASES.items():
            if main_key not in self.loaded_vars:
                for alias in aliases:
                    if alias in self.loaded_vars:
                        self.loaded_vars[main_key] = self.loaded_vars[alias]
                        logger.debug(f"별칭 해결: {alias} -> {main_key}")
                        break
                    elif alias in os.environ:
                        self.loaded_vars[main_key] = os.environ[alias]
                        logger.debug(f"시스템 별칭 해결: {alias} -> {main_key}")
                        break
    
    def _apply_defaults(self):
        """기본값 적용"""
        for key, default_value in DEFAULT_VALUES.items():
            if key not in self.loaded_vars:
                self.loaded_vars[key] = default_value
                logger.debug(f"기본값 적용: {key}={default_value}")
    
    def _validate_variables(self, strict_mode: bool) -> bool:
        """환경변수 검증"""
        self.missing_vars = []
        self.validation_errors = []
        
        # 필수 변수 확인
        for var in REQUIRED_VARIABLES:
            if var not in self.loaded_vars or not self.loaded_vars[var]:
                self.missing_vars.append(var)
        
        # 값 검증
        validation_rules = {
            'PORT': lambda x: x.isdigit() and 1000 <= int(x) <= 65535,
            'MAX_SYMBOLS': lambda x: x.isdigit() and 1 <= int(x) <= 50,
            'CACHE_HOURS': lambda x: x.isdigit() and 1 <= int(x) <= 168,
            'THINKING_BUDGET': lambda x: x.isdigit() and 1000 <= int(x) <= 50000,
            'API_TIMEOUT': lambda x: x.isdigit() and 5 <= int(x) <= 300,
            'RETRY_COUNT': lambda x: x.isdigit() and 1 <= int(x) <= 10
        }
        
        for key, validator in validation_rules.items():
            if key in self.loaded_vars:
                if not validator(self.loaded_vars[key]):
                    self.validation_errors.append(f"{key}: 잘못된 값 '{self.loaded_vars[key]}'")
        
        # API 키 길이 검증
        api_key_min_lengths = {
            'KIS_APP_KEY': 10,
            'KIS_APP_SECRET': 10,
            'GEMINI_API_KEY': 20,
            'PERPLEXITY_API_KEY': 20
        }
        
        for key, min_length in api_key_min_lengths.items():
            if key in self.loaded_vars and len(self.loaded_vars[key]) < min_length:
                self.validation_errors.append(f"{key}: 너무 짧음 (최소 {min_length}자)")
        
        # 결과 출력
        if self.missing_vars:
            logger.warning(f"⚠️ 누락된 필수 환경변수: {self.missing_vars}")
        
        if self.validation_errors:
            logger.error(f"❌ 검증 오류: {self.validation_errors}")
        
        return len(self.missing_vars) == 0 and len(self.validation_errors) == 0
    
    def _set_environment_variables(self):
        """환경변수를 시스템에 설정"""
        for key, value in self.loaded_vars.items():
            os.environ[key] = value
    
    def _create_sample_env_file(self):
        """샘플 .env 파일 생성"""
        sample_content = self._generate_sample_env_content()
        
        try:
            sample_file = self.env_file.with_suffix('.env.sample')
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(sample_content)
            logger.info(f"📝 샘플 파일 생성: {sample_file}")
            logger.info("   실제 API 키로 수정 후 .env로 이름을 변경하세요")
        except Exception as e:
            logger.error(f"❌ 샘플 파일 생성 실패: {e}")
    
    def _generate_sample_env_content(self) -> str:
        """샘플 환경변수 파일 내용 생성"""
        return f"""# Enhanced 퀀트 시스템 1.8.2 환경설정
# 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# =============================================================================
# 🔐 API 키 설정 (필수 - 실제 값으로 교체하세요!)
# =============================================================================

# 한국투자증권 API
KIS_APP_KEY=your_kis_app_key_here
KIS_APP_SECRET=your_kis_app_secret_here

# 별칭 (선택사항 - 위의 값과 동일하게 설정)
MKEY=your_kis_app_key_here
MKEYS=your_kis_app_secret_here

# Gemini AI API
GEMINI_API_KEY=your_gemini_api_key_here
GEM=your_gemini_api_key_here

# Perplexity AI API
PERPLEXITY_API_KEY=your_perplexity_api_key_here
PPL=your_perplexity_api_key_here

# OpenAI API (선택사항)
# OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API (선택사항)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# =============================================================================
# ⚙️ 시스템 설정
# =============================================================================

# 환경 설정
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# 서버 설정
HOST=0.0.0.0
PORT=7860
SHARE=false

# =============================================================================
# 📊 분석 설정
# =============================================================================

# 종목 분석
MAX_SYMBOLS=10
CACHE_HOURS=48
STRICT_VALIDATION=true

# AI 설정
THINKING_BUDGET=5000
PATTERN_ENABLED=true

# API 설정
API_TIMEOUT=30
RETRY_COUNT=3
USE_CACHE=true

# =============================================================================
# 📝 설정 가이드
# =============================================================================

# 1. KIS API 키 발급: https://apiportal.koreainvestment.com/
# 2. Gemini API 키: https://aistudio.google.com/app/apikey
# 3. Perplexity API 키: https://www.perplexity.ai/settings/api

# ⚠️ 보안 주의사항:
# - API 키는 절대 공유하지 마세요
# - .env 파일을 Git에 커밋하지 마세요
# - 주기적으로 API 키를 갱신하세요
"""

    def get_env_var(self, key: str, default: str = "") -> str:
        """환경변수 안전하게 가져오기"""
        return self.loaded_vars.get(key, os.getenv(key, default))
    
    def get_bool_env(self, key: str, default: bool = False) -> bool:
        """불린 환경변수 가져오기"""
        value = self.get_env_var(key, str(default).lower())
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_int_env(self, key: str, default: int = 0) -> int:
        """정수 환경변수 가져오기"""
        try:
            return int(self.get_env_var(key, str(default)))
        except ValueError:
            logger.warning(f"잘못된 정수 값: {key}, 기본값 사용")
            return default
    
    def get_float_env(self, key: str, default: float = 0.0) -> float:
        """실수 환경변수 가져오기"""
        try:
            return float(self.get_env_var(key, str(default)))
        except ValueError:
            logger.warning(f"잘못된 실수 값: {key}, 기본값 사용")
            return default
    
    def export_config(self) -> Dict[str, Any]:
        """설정 내보내기 (민감 정보 마스킹)"""
        config = {}
        sensitive_keys = ['KEY', 'SECRET', 'TOKEN', 'PASSWORD']
        
        for key, value in self.loaded_vars.items():
            if any(sensitive in key.upper() for sensitive in sensitive_keys):
                config[key] = f"{'*' * 8}...{value[-4:]}" if len(value) > 4 else "****"
            else:
                config[key] = value
        
        return {
            'variables': config,
            'missing_vars': self.missing_vars,
            'validation_errors': self.validation_errors,
            'load_timestamp': self.load_timestamp.isoformat() if self.load_timestamp else None
        }
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """API 키 유효성 검사"""
        validation_result = {
            'kis': bool(self.get_env_var('KIS_APP_KEY') and self.get_env_var('KIS_APP_SECRET')),
            'gemini': bool(self.get_env_var('GEMINI_API_KEY')),
            'perplexity': bool(self.get_env_var('PERPLEXITY_API_KEY')),
            'openai': bool(self.get_env_var('OPENAI_API_KEY')),
            'anthropic': bool(self.get_env_var('ANTHROPIC_API_KEY'))
        }
        
        logger.info("🔑 API 키 상태:")
        for service, is_valid in validation_result.items():
            status = "✅" if is_valid else "❌"
            logger.info(f"   {service}: {status}")
        
        return validation_result
    
    def get_config_summary(self) -> str:
        """설정 요약 문자열"""
        summary = [
            f"환경변수 로드 상태: {'✅ 성공' if self.load_timestamp else '❌ 실패'}",
            f"로드된 변수 수: {len(self.loaded_vars)}",
            f"누락된 필수 변수: {len(self.missing_vars)}",
            f"검증 오류: {len(self.validation_errors)}"
        ]
        
        if self.load_timestamp:
            summary.append(f"마지막 로드: {self.load_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(summary)

# 전역 인스턴스
_env_loader_instance: Optional[EnvironmentLoader] = None

def get_env_loader() -> EnvironmentLoader:
    """환경변수 로더 싱글톤 인스턴스"""
    global _env_loader_instance
    if _env_loader_instance is None:
        _env_loader_instance = EnvironmentLoader()
    return _env_loader_instance

# 편의 함수들
def load_env_file(env_file: str = ".env", strict_mode: bool = False) -> bool:
    """환경 변수 파일 로드"""
    loader = get_env_loader()
    loader.env_file = Path(env_file)
    return loader.load_env_file(strict_mode)

def validate_api_keys() -> Dict[str, bool]:
    """API 키 유효성 검사"""
    loader = get_env_loader()
    return loader.validate_api_keys()

def create_sample_env_file(env_file: str = ".env") -> bool:
    """샘플 .env 파일 생성"""
    try:
        loader = EnvironmentLoader(env_file)
        loader._create_sample_env_file()
        return True
    except Exception as e:
        logger.error(f"샘플 파일 생성 실패: {e}")
        return False

def get_env_var(key: str, default: str = "") -> str:
    """환경변수 가져오기"""
    loader = get_env_loader()
    return loader.get_env_var(key, default)

def get_bool_env(key: str, default: bool = False) -> bool:
    """불린 환경변수 가져오기"""
    loader = get_env_loader()
    return loader.get_bool_env(key, default)

def get_int_env(key: str, default: int = 0) -> int:
    """정수 환경변수 가져오기"""
    loader = get_env_loader()
    return loader.get_int_env(key, default)

def check_environment_health() -> Dict[str, Any]:
    """환경 상태 종합 검사"""
    loader = get_env_loader()
    
    return {
        'env_file_exists': loader.env_file.exists(),
        'loaded_variables': len(loader.loaded_vars),
        'missing_required': loader.missing_vars,
        'validation_errors': loader.validation_errors,
        'api_keys_status': loader.validate_api_keys(),
        'load_timestamp': loader.load_timestamp.isoformat() if loader.load_timestamp else None,
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd()
        }
    }

if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 환경변수 로더 테스트")
    print("=" * 50)
    
    # 환경변수 로드
    success = load_env_file(strict_mode=False)
    print(f"로드 결과: {'✅ 성공' if success else '❌ 실패'}")
    
    # API 키 검증
    api_status = validate_api_keys()
    print(f"\nAPI 키 상태: {api_status}")
    
    # 환경 상태 검사
    health = check_environment_health()
    print(f"\n환경 상태:")
    for key, value in health.items():
        if key != 'system_info':
            print(f"  {key}: {value}")
    
    # 설정 요약
    loader = get_env_loader()
    print(f"\n설정 요약:\n{loader.get_config_summary()}")
