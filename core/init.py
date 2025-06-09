#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core 패키지 초기화
핵심 API 관리 및 데이터 처리 모듈들
"""

try:
    from .api_manager import UnifiedAPIManager
    from .token_manager import KisTokenManager
    from .data_validator import DataValidationManager
except ImportError as e:
    print(f"⚠️ Core 모듈 import 오류: {e}")

__all__ = [
    'UnifiedAPIManager',
    'KisTokenManager', 
    'DataValidationManager'
]

__version__ = '1.8.2'
