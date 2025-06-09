"""
Enhanced 통합 시스템 (메인 분석 엔진)
"""
from .sector_cache import SectorWeightCache
from .sector_mapper import SectorMappingEngine  
from .gap_booster import GeminiGapFilterBooster
from .pattern_booster import GeminiPatternRecognitionBooster
from ..core.api_manager import UnifiedAPIManager
from ..core.token_manager import KisTokenManager
from ..core.data_validator import DataValidationManager
from ..models.perplexity_models import PerplexityModelManager

class Enhanced182ThreeTierSystem:
    """3단 구조 Enhanced 시스템: 딥리서치 섹터 + Gemini Flash + 갭필터 부스터"""

    def __init__(self):
        # 기본 API 관리자들
        self.api_manager = UnifiedAPIManager()
        self.model_manager = PerplexityModelManager()
        self.kis_token_manager = None
        self.kis_base_url = "https://openapivts.koreainvestment.com:29443"

        # 한투 자동 초기화
        self._auto_initialize_kis()

        # 🆕 3단 구조 핵심 컴포넌트들
        self.sector_cache = SectorWeightCache(cache_hours=48)
        self.sector_mapper = SectorMappingEngine(self.api_manager)
        self.gap_booster = GeminiGapFilterBooster(self.api_manager)

        # 🆕 패턴 인식 부스터 추가
        self.pattern_booster = GeminiPatternRecognitionBooster(self.api_manager)
        self.pattern_enabled = True  # ✅ 패턴 인식 on/off 스위치

        # 하이브리드 전략
        self.use_hybrid_gemini = True
        print("🔥 하이브리드 Gemini: N점수(Flash) + 갭필터(Flash-Lite)")
        print("🔍 패턴 인식 부스터 초기화 완료 (Gemini 2.5 Flash)")

        # ✅ 캐시 없으면 즉시 갱신!
        if not self.sector_cache.is_valid():
            print("🔄 섹터 캐시 없음 - 즉시 갱신 시작...")
            self._initialize_sector_cache()
        else:
            print("✅ 섹터 캐시 유효함")
            
        # 데이터 검증 매니저 추가
        self.data_validator = DataValidationManager()
        self.fallback_enabled = True  # 폴백 데이터 사용 여부

        print("🚀 Enhanced 1.8.2 3단 구조 시스템 초기화")
        print("📊 Tier 1: 딥리서치 섹터 가중치 (12시간 캐싱)")
        print("🔥 Tier 2: Gemini Flash N점수 (실시간)")
        print("🎯 Tier 3: 갭필터 부스터 Lite (시너지/리스크)")

    # 나머지 메서드들...
