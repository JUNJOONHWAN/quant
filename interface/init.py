#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface 패키지 초기화
웹 UI 및 사용자 인터페이스 모듈들
"""

try:
    from .gradio_ui import create_three_tier_interface
    from .gradio_ui import (
        format_analysis_summary,
        format_vtnf_scores,
        format_gap_analysis,
        format_pattern_analysis,
        format_investment_recommendation,
        format_portfolio_analysis,
        format_symbols_comparison,
        format_system_status
    )
except ImportError as e:
    print(f"⚠️ Interface 모듈 import 오류: {e}")

__all__ = [
    'create_three_tier_interface',
    'format_analysis_summary',
    'format_vtnf_scores', 
    'format_gap_analysis',
    'format_pattern_analysis',
    'format_investment_recommendation',
    'format_portfolio_analysis',
    'format_symbols_comparison',
    'format_system_status'
]

__version__ = '1.8.2'
