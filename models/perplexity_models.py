#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
퍼플렉시티 모델 관리자
Perplexity API 모델 통합 관리 - 딥리서치 포함
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PerplexityModel:
    """퍼플렉시티 모델 정보"""
    api_model: str
    display_name: str
    category: str
    context_length: int
    max_output_tokens: int
    model_type: str
    features: List[str]
    pricing_tier: str
    description: str
    use_cases: List[str]
    recommended: bool = False
    deprecated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

class PerplexityModelManager:
    """퍼플렉시티 API 모델 통합 관리 - 딥리서치 포함"""

    def __init__(self):
        self.models = self._initialize_models()
        self.model_groups = self._group_models()
        self.default_model = "sonar-pro"
        self.fallback_model = "sonar"
        
        logger.info(f"🔍 Perplexity 모델 매니저 초기화: {len(self.models)}개 모델")

    def _initialize_models(self) -> Dict[str, PerplexityModel]:
        """2025년 6월 최신 Perplexity 모델 정의"""
        models = {}
        
        # 🔍 Sonar Models (핵심 검색 모델들)
        models["sonar-pro"] = PerplexityModel(
            api_model="sonar-pro",
            display_name="🚀 Sonar Pro (검색 + 복잡 쿼리)",
            category="sonar",
            context_length=200000,  # 200k tokens
            max_output_tokens=8000,
            model_type="Chat Completion",
            features=["web_search", "real_time_data", "advanced_reasoning"],
            pricing_tier="premium",
            description="복잡한 쿼리와 깊은 콘텐츠 이해에 최적화된 고급 검색 모델",
            use_cases=[
                "심화 시장 분석", "복합 데이터 검색", "전문가 수준 리서치",
                "다각도 정보 종합", "트렌드 분석"
            ],
            recommended=True
        )
        
        models["sonar"] = PerplexityModel(
            api_model="sonar",
            display_name="⚡ Sonar (경량 검색)",
            category="sonar",
            context_length=128000,  # 128k tokens
            max_output_tokens=4000,
            model_type="Chat Completion",
            features=["web_search", "real_time_data"],
            pricing_tier="standard",
            description="빠르고 비용 효율적인 경량 검색 모델",
            use_cases=[
                "기본 정보 검색", "뉴스 요약", "빠른 팩트 체크",
                "일반적인 질의응답", "실시간 데이터 조회"
            ],
            recommended=False
        )

        # 📊 Deep Research Model (핵심)
        models["sonar-deep-research"] = PerplexityModel(
            api_model="sonar-deep-research",
            display_name="📊 Sonar Deep Research (심화 연구)",
            category="deep_research",
            context_length=128000,
            max_output_tokens=8000,
            model_type="Deep Research",
            features=["deep_analysis", "comprehensive_reports", "multi_source_synthesis"],
            pricing_tier="premium",
            description="상세한 보고서와 심층 인사이트 생성에 최적화",
            use_cases=[
                "종합 투자 분석", "시장 심화 리서치", "경쟁사 분석",
                "업계 트렌드 보고서", "전략적 의사결정 지원"
            ],
            recommended=True
        )

        # 🧠 Reasoning Models (추론 특화)
        models["sonar-reasoning-pro"] = PerplexityModel(
            api_model="sonar-reasoning-pro",
            display_name="🧠 Sonar Reasoning Pro (고급 추론)",
            category="reasoning",
            context_length=128000,
            max_output_tokens=8000,
            model_type="Chain of Thought",
            features=["advanced_reasoning", "step_by_step_analysis", "logical_deduction"],
            pricing_tier="premium",
            description="복잡한 논리적 추론과 단계별 분석에 특화",
            use_cases=[
                "복합 투자 전략 수립", "리스크 시나리오 분석", 
                "논리적 의사결정 지원", "복잡한 문제 해결"
            ]
        )

        models["sonar-reasoning"] = PerplexityModel(
            api_model="sonar-reasoning",
            display_name="🔍 Sonar Reasoning (기본 추론)",
            category="reasoning",
            context_length=128000,
            max_output_tokens=4000,
            model_type="Chain of Thought",
            features=["basic_reasoning", "step_by_step_analysis"],
            pricing_tier="standard",
            description="기본적인 추론과 단계별 분석 제공",
            use_cases=[
                "기본 분석 논리", "단계별 설명", "추론 과정 시각화"
            ]
        )

        # 💬 Chat Model (오프라인)
        models["r1-1776"] = PerplexityModel(
            api_model="r1-1776",
            display_name="💬 R1-1776 (오프라인 채팅)",
            category="chat",
            context_length=128000,
            max_output_tokens=4000,
            model_type="Chat Completion",
            features=["offline_chat", "no_web_search"],
            pricing_tier="standard",
            description="웹 검색 없는 순수 대화형 모델",
            use_cases=[
                "일반 대화", "텍스트 생성", "창작 지원", "코딩 도움"
            ]
        )

        return models

    def _group_models(self) -> Dict[str, List[str]]:
        """모델 그룹핑"""
        groups = {
            "recommended": [],
            "sonar": [],
            "deep_research": [],
            "reasoning": [],
            "chat": [],
            "premium": [],
            "standard": []
        }
        
        for model_id, model in self.models.items():
            if model.recommended:
                groups["recommended"].append(model_id)
            
            groups[model.category].append(model_id)
            groups[model.pricing_tier].append(model_id)
        
        return groups

    def get_model_info(self, model_id: str) -> Optional[PerplexityModel]:
        """모델 정보 조회"""
        return self.models.get(model_id)

    def get_recommended_models(self) -> List[PerplexityModel]:
        """추천 모델 목록"""
        return [
            self.models[model_id] 
            for model_id in self.model_groups["recommended"]
        ]

    def get_models_by_category(self, category: str) -> List[PerplexityModel]:
        """카테고리별 모델 목록"""
        return [
            self.models[model_id] 
            for model_id in self.model_groups.get(category, [])
        ]

    def get_best_model_for_task(self, task_type: str) -> str:
        """작업 유형별 최적 모델 추천"""
        task_models = {
            "deep_research": "sonar-deep-research",
            "market_analysis": "sonar-pro", 
            "news_sentiment": "sonar-pro",
            "quick_search": "sonar",
            "reasoning": "sonar-reasoning-pro",
            "chat": "r1-1776",
            "sector_analysis": "sonar-deep-research",
            "gap_analysis": "sonar-pro",
            "pattern_analysis": "sonar-reasoning"
        }
        
        return task_models.get(task_type, self.default_model)

    def validate_model(self, model_id: str) -> bool:
        """모델 유효성 검사"""
        return model_id in self.models

    def get_model_context_limit(self, model_id: str) -> int:
        """모델 컨텍스트 길이 조회"""
        model = self.models.get(model_id)
        return model.context_length if model else 128000

    def get_model_max_output(self, model_id: str) -> int:
        """모델 최대 출력 토큰 수"""
        model = self.models.get(model_id)
        return model.max_output_tokens if model else 4000

    def estimate_cost_tier(self, model_id: str) -> str:
        """비용 등급 추정"""
        model = self.models.get(model_id)
        return model.pricing_tier if model else "standard"

    def get_model_features(self, model_id: str) -> List[str]:
        """모델 기능 목록"""
        model = self.models.get(model_id)
        return model.features if model else []

    def has_web_search(self, model_id: str) -> bool:
        """웹 검색 기능 여부"""
        features = self.get_model_features(model_id)
        return "web_search" in features

    def has_deep_research(self, model_id: str) -> bool:
        """딥 리서치 기능 여부"""
        features = self.get_model_features(model_id)
        return "deep_analysis" in features or "comprehensive_reports" in features

    def has_reasoning(self, model_id: str) -> bool:
        """추론 기능 여부"""
        features = self.get_model_features(model_id)
        return any(f in features for f in ["advanced_reasoning", "basic_reasoning", "logical_deduction"])

    def get_fallback_model(self, preferred_model: str) -> str:
        """폴백 모델 추천"""
        if not self.validate_model(preferred_model):
            return self.fallback_model
        
        model = self.models[preferred_model]
        
        # 같은 카테고리 내에서 폴백
        category_models = self.get_models_by_category(model.category)
        if len(category_models) > 1:
            for fallback in category_models:
                if fallback.api_model != preferred_model and not fallback.deprecated:
                    return fallback.api_model
        
        return self.fallback_model

    def get_model_selection_guide(self) -> Dict[str, Any]:
        """모델 선택 가이드"""
        return {
            "quick_tasks": {
                "model": "sonar",
                "description": "빠른 검색과 기본 분석",
                "suitable_for": ["뉴스 요약", "기본 정보 검색", "빠른 팩트체크"]
            },
            "comprehensive_analysis": {
                "model": "sonar-pro", 
                "description": "종합적인 분석과 복잡한 쿼리",
                "suitable_for": ["시장 분석", "복합 데이터 검색", "전문가 수준 리서치"]
            },
            "deep_research": {
                "model": "sonar-deep-research",
                "description": "심층 연구와 상세 보고서",
                "suitable_for": ["투자 분석", "경쟁사 분석", "업계 트렌드 리포트"]
            },
            "reasoning_tasks": {
                "model": "sonar-reasoning-pro",
                "description": "논리적 추론과 단계별 분석", 
                "suitable_for": ["전략 수립", "리스크 분석", "복잡한 의사결정"]
            },
            "offline_chat": {
                "model": "r1-1776",
                "description": "웹 검색 없는 순수 대화",
                "suitable_for": ["일반 대화", "텍스트 생성", "창작 지원"]
            }
        }

    def get_usage_statistics(self) -> Dict[str, Any]:
        """사용 통계 (시뮬레이션)"""
        return {
            "total_models": len(self.models),
            "recommended_models": len(self.model_groups["recommended"]),
            "categories": {
                category: len(models) 
                for category, models in self.model_groups.items()
                if category not in ["premium", "standard", "recommended"]
            },
            "pricing_tiers": {
                "premium": len(self.model_groups["premium"]),
                "standard": len(self.model_groups["standard"])
            },
            "features_coverage": {
                "web_search": len([m for m in self.models.values() if "web_search" in m.features]),
                "deep_analysis": len([m for m in self.models.values() if "deep_analysis" in m.features]),
                "reasoning": len([m for m in self.models.values() if any(f in m.features for f in ["advanced_reasoning", "basic_reasoning"])])
            }
        }

    def export_models_info(self) -> Dict[str, Any]:
        """모델 정보 내보내기"""
        return {
            "models": {
                model_id: model.to_dict() 
                for model_id, model in self.models.items()
            },
            "groups": self.model_groups,
            "metadata": {
                "default_model": self.default_model,
                "fallback_model": self.fallback_model,
                "last_updated": datetime.now().isoformat(),
                "total_models": len(self.models)
            }
        }

    def print_models_summary(self):
        """모델 요약 출력"""
        print("\n🔍 Perplexity 모델 매니저 요약")
        print("=" * 50)
        
        print(f"📊 총 모델 수: {len(self.models)}")
        print(f"⭐ 추천 모델: {len(self.model_groups['recommended'])}개")
        print(f"🔧 기본 모델: {self.default_model}")
        
        print("\n📂 카테고리별 분포:")
        for category, models in self.model_groups.items():
            if category not in ["premium", "standard", "recommended"]:
                print(f"  • {category}: {len(models)}개")
        
        print("\n⭐ 추천 모델들:")
        for model in self.get_recommended_models():
            print(f"  • {model.display_name}")
            print(f"    용도: {', '.join(model.use_cases[:2])}")

    def __str__(self) -> str:
        return f"PerplexityModelManager({len(self.models)} models)"

    def __repr__(self) -> str:
        return f"PerplexityModelManager(models={list(self.models.keys())})"

# 전역 인스턴스 (싱글톤 패턴)
_perplexity_manager_instance = None

def get_perplexity_manager() -> PerplexityModelManager:
    """퍼플렉시티 모델 매니저 싱글톤 인스턴스"""
    global _perplexity_manager_instance
    if _perplexity_manager_instance is None:
        _perplexity_manager_instance = PerplexityModelManager()
    return _perplexity_manager_instance

# 편의 함수들
def get_best_model_for_research() -> str:
    """리서치용 최적 모델"""
    return "sonar-deep-research"

def get_best_model_for_analysis() -> str:
    """분석용 최적 모델"""
    return "sonar-pro"

def get_best_model_for_reasoning() -> str:
    """추론용 최적 모델"""
    return "sonar-reasoning-pro"

def is_premium_model(model_id: str) -> bool:
    """프리미엄 모델 여부"""
    manager = get_perplexity_manager()
    model = manager.get_model_info(model_id)
    return model.pricing_tier == "premium" if model else False

# 모델 호환성 체크
def check_model_compatibility(model_id: str, required_features: List[str]) -> bool:
    """모델 호환성 검사"""
    manager = get_perplexity_manager()
    model_features = manager.get_model_features(model_id)
    return all(feature in model_features for feature in required_features)

if __name__ == "__main__":
    # 테스트 및 데모
    manager = PerplexityModelManager()
    manager.print_models_summary()
    
    # 모델 선택 가이드 출력
    print("\n📖 모델 선택 가이드:")
    guide = manager.get_model_selection_guide()
    for task, info in guide.items():
        print(f"\n• {task.replace('_', ' ').title()}:")
        print(f"  모델: {info['model']}")
        print(f"  설명: {info['description']}")
        print(f"  적합한 용도: {', '.join(info['suitable_for'])}")
