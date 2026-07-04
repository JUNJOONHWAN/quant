#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_ai_system/__init__.py - ML/AI 시스템 패키지 초기화 파일

이 패키지는 Sweet Spot 전략을 위한 완전한 ML+AI 하이브리드 시스템을 제공합니다:
- ML 최적화 엔진: 파라미터 자동 최적화
- AI 오차 분석: Perplexity AI 기반 심층 분석 
- 예측 시스템: 주간 수익률 예측 및 검증
- 수렴 추적: ML/AI 파워 밸런스 관리
"""

# Core ML Components
from .ml_optimizer import MLOptimizer, OptimizationResult, ParameterValidator, DataCache, MLParameterManager
from .recursive_ml_optimizer import RecursiveMLOptimizer

# Prediction System (optional heavy deps)
try:
    from .prediction_system import PredictionSystem
except Exception:
    PredictionSystem = None  # optional
from .prediction_validator import PredictionValidator

# AI Analysis Components (optional external deps)
try:
    from .ai_error_analyzer import AIErrorAnalyzer
except Exception:
    AIErrorAnalyzer = None
try:
    from .ai_parameter_optimizer import AIParameterOptimizer
except Exception:
    AIParameterOptimizer = None
from .ai_trend_validator import TrendValidationManager, TrendValidationReport, ValidationResult

# Convergence and Tracking (optional)
try:
    from .convergence_tracker import ConvergenceTracker
except Exception:
    ConvergenceTracker = None

# Report Generation (optional)
try:
    from .enhanced_report_generator import EnhancedReportGenerator
except Exception:
    EnhancedReportGenerator = None

# Package version
__version__ = "2.0.0"

# Package metadata
__author__ = "Sweet Spot ML+AI System"
__description__ = "Complete ML+AI hybrid system for Sweet Spot investment strategy"

# Export main components for easy import
__all__ = [
    # Core ML
    "MLOptimizer",
    "RecursiveMLOptimizer", 
    "OptimizationResult",
    "ParameterValidator",
    "DataCache",
    "MLParameterManager",
    
    # Prediction
    "PredictionSystem",
    "PredictionValidator",
    
    # AI Analysis
    "AIErrorAnalyzer", 
    "AIParameterOptimizer",
    "TrendValidationManager",
    "TrendValidationReport",
    "ValidationResult",
    
    # Tracking
    "ConvergenceTracker",
    
    # Reporting
    "EnhancedReportGenerator"
]
