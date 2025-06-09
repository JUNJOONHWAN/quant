#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 설정 관리자
"""
import os
from typing import Optional
from pydantic import BaseSettings, validator
from .env_loader import load_env_file

class AppSettings(BaseSettings):
    """앱 기본 설정"""
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

class APISettings(BaseSettings):
    """API 설정"""
    kis_app_key: Optional[str] = None
    kis_app_secret: Optional[str] = None
    gemini_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    
    @validator('*', pre=True)
    def load_from_env(cls, v, field):
        """환경변수에서 값 로드 (별칭 포함)"""
        if v is not None:
            return v
        
        # 별칭 매핑
        aliases = {
            'kis_app_key': ['MKEY', 'KIS_APP_KEY'],
            'kis_app_secret': ['MKEYS', 'KIS_APP_SECRET'],
            'gemini_api_key': ['GEM', 'GEMINI_API_KEY'],
            'perplexity_api_key': ['PPL', 'PERPLEXITY_API_KEY']
        }
        
        field_name = field.name
        if field_name in aliases:
            for alias in aliases[field_name]:
                value = os.getenv(alias)
                if value:
                    return value
        
        return os.getenv(field_name.upper())
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

class AnalysisSettings(BaseSettings):
    """분석 설정"""
    max_symbols: int = 10
    cache_hours: int = 48
    strict_validation: bool = True
    pattern_enabled: bool = True
    gap_filter_enabled: bool = True
    
    class Config:
        env_file = '.env'

class GlobalSettings:
    """전역 설정 관리자"""
    
    def __init__(self):
        # 환경변수 로드
        load_env_file()
        
        # 설정 객체들 생성
        self.app = AppSettings()
        self.api = APISettings()
        self.analysis = AnalysisSettings()
    
    def validate_all(self) -> bool:
        """모든 설정 검증"""
        try:
            # API 키 검증
            required_keys = [
                self.api.kis_app_key,
                self.api.kis_app_secret,
                self.api.gemini_api_key,
                self.api.perplexity_api_key
            ]
            
            if not all(required_keys):
                print("❌ 필수 API 키가 누락되었습니다")
                return False
            
            print("✅ 모든 설정 검증 완료")
            return True
            
        except Exception as e:
            print(f"❌ 설정 검증 실패: {e}")
            return False

# 전역 설정 인스턴스
settings = GlobalSettings()
