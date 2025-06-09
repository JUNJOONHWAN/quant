"""
섹터 가중치 캐시 시스템
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

class SectorWeightCache:
    """섹터별 VTNF 가중치 캐시 시스템"""

    def __init__(self, cache_hours: int = 48):
        self.cache_file = Path("sector_weights_cache.json")
        self.cache_hours = cache_hours
        self.sector_weights = {}
        self.last_updated = None
        self.load_cache()

    def load_cache(self):
        """캐시된 섹터 가중치 로드"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.last_updated = datetime.fromisoformat(data['last_updated'])

                    if datetime.now() - self.last_updated < timedelta(hours=self.cache_hours):
                        self.sector_weights = data['sector_weights']
                        print(f"✅ 섹터 가중치 캐시 로드: {self.last_updated.strftime('%H:%M')}")
                        return True
                    else:
                        print(f"⏰ 섹터 가중치 캐시 만료 ({self.cache_hours}시간 초과)")

        except Exception as e:
            print(f"❌ 섹터 캐시 로드 실패: {e}")

        return False

    def save_cache(self, sector_weights: dict):
        """섹터 가중치 캐시 저장"""
        try:
            self.sector_weights = sector_weights
            self.last_updated = datetime.now()

            cache_data = {
                'sector_weights': sector_weights,
                'last_updated': self.last_updated.isoformat(),
                'cache_hours': self.cache_hours
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            print(f"💾 섹터 가중치 캐시 저장: {len(sector_weights)}개 섹터")

        except Exception as e:
            print(f"❌ 섹터 캐시 저장 실패: {e}")

    def is_valid(self) -> bool:
        """캐시 유효성 검사"""
        if not self.last_updated or not self.sector_weights:
            return False
        return datetime.now() - self.last_updated < timedelta(hours=self.cache_hours)

    def get_sector_weight(self, sector: str) -> dict:
        """특정 섹터의 가중치 반환"""
        default_weights = {'V': 0.30, 'T': 0.30, 'N': 0.25, 'F': 0.15}
        return self.sector_weights.get(sector, default_weights)
