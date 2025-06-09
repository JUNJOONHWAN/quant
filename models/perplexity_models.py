"""
퍼플렉시티 모델 관리자
"""

class PerplexityModelManager:
    """퍼플렉시티 API 모델 통합 관리 - 딥리서치 포함"""

    def __init__(self):
        self.models = self._initialize_models()
        self.model_groups = self._group_models()

    def _initialize_models(self) -> dict:
        """2025년 6월 핵심 모델 + 딥리서치"""
        return {
            # 🔍 Sonar Models (핵심만)
            "sonar-pro": {
                "display_name": "🚀 Sonar Pro (검색 + 복잡 쿼리)",
                "category": "sonar",
                "modes": ["medium", "low"],
                "api_model": "sonar-pro",
                "description": "복잡한 쿼리와 깊은 콘텐츠 이해에 최적화된 고급 검색 모델"
            },
            "sonar": {
                "display_name": "⚡ Sonar (경량 검색)",
                "category": "sonar",
                "modes": ["medium", "low"],
                "api_model": "sonar",
                "description": "빠르고 비용 효율적인 경량 검색 모델"
            },

            # 📊 Deep Research Model (핵심)
            "sonar-deep-research": {
                "display_name": "📊 Sonar Deep Research (심화 연구)",
                "category": "deep_research",
                "modes": ["standard"],
                "api_model": "sonar-deep-research",
                "description": "상세한 보고서와 심층 인사이트 생성에 최적화"
            }
        }

    # 나머지 메서드들...
