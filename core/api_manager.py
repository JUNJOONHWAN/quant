"""
통합 API 관리자
"""
import os
import json
from datetime import datetime
from pathlib import Path

class UnifiedAPIManager:
    """모든 API 키를 통합 관리하는 클래스"""

    def __init__(self):
        self.settings_file = Path("unified_api_settings.json")
        self.settings = self.load_settings()

    def load_settings(self) -> dict:
        """저장된 API 설정 로드"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    print("✅ API 설정 로드 완료")
                    return settings
        except Exception as e:
            print(f"⚠️ 설정 로드 실패: {str(e)}")

        return {
            "kis_app_key": os.getenv("KIS_APP_KEY", "") or os.getenv("MKEY", ""),
            "kis_app_secret": os.getenv("KIS_APP_SECRET", "") or os.getenv("MKEYS", ""),
            "perplexity_api_key": os.getenv("PERPLEXITY_API_KEY", "") or os.getenv("PPL", ""),
            "gemini_api_key": os.getenv("GEMINI_API_KEY", "") or os.getenv("GEM", ""),
            "selected_model": "gemini-2.0-flash",
            "last_updated": datetime.now().isoformat()
        }

    def get_kis_config(self) -> dict:
        """한투 API 설정 반환"""
        return {
            'app_key': self.settings.get('kis_app_key', ''),
            'app_secret': self.settings.get('kis_app_secret', '')
        }

    def get_perplexity_config(self) -> dict:
        """Perplexity API 설정 반환"""
        return {
            'api_key': self.settings.get('perplexity_api_key', 'PPL'),
            'base_url': 'https://api.perplexity.ai'
        }

    def get_gemini_config(self) -> dict:
        """Gemini API 설정 반환"""
        return {
            'api_key': self.settings.get('gemini_api_key', 'GEM'),
            'base_url': 'https://generativelanguage.googleapis.com/v1beta'
        }
