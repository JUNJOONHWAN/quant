#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models 패키지 초기화
AI 모델 관리 및 설정 모듈들
"""

try:
    from .perplexity_models import PerplexityModelManager
except ImportError as e:
    print(f"⚠️ Models 모듈 import 오류: {e}")

__all__ = [
    'PerplexityModelManager'
]

__version__ = '1.8.2'
