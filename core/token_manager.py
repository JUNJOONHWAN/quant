"""
한투 API 토큰 관리자
"""
import os
import json
import time
import requests
from datetime import datetime

class KisTokenManager:
    """한투 API 토큰 관리"""

    def __init__(self, app_key: str, app_secret: str):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = "https://openapivts.koreainvestment.com:29443"
        self.token_file = "kis_unified_token.json"

    def get_valid_token(self) -> str | None:
        """유효한 토큰 반환"""
        token = self.load_token()
        if token:
            return token
        return self.get_new_token()

    def load_token(self) -> str | None:
        """저장된 토큰 로드"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("expires_at", 0) > time.time():
                        return data["access_token"]
                    else:
                        os.remove(self.token_file)
            except Exception:
                pass
        return None

    def get_new_token(self) -> str | None:
        """새 토큰 발급"""
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            headers = {"content-type": "application/json"}
            body = {
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret
            }

            response = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
            response.raise_for_status()

            token_data = response.json()

            if "access_token" in token_data:
                access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 86400)

                with open(self.token_file, "w", encoding='utf-8') as f:
                    json.dump({
                        "access_token": access_token,
                        "expires_at": time.time() + expires_in - 3600,
                        "created_at": datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)

                print("✅ 한투 토큰 발급 성공!")
                return access_token
            else:
                print(f"❌ 한투 토큰 발급 실패: {token_data}")
                return None

        except Exception as e:
            print(f"❌ 한투 토큰 발급 오류: {str(e)}")
            return None

    def get_auth_headers(self, tr_id: str = "") -> dict:
        """인증 헤더 생성"""
        token = self.get_valid_token()
        if not token:
            raise Exception("한투 토큰 발급 실패")

        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P"
        }
