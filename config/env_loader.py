#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
환경 변수 로드 모듈 (개선된 버전)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_env_file(env_file: str = ".env") -> bool:
    """환경 변수 파일 로드 with 검증"""
    try:
        env_path = Path(env_file)
        
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ {env_file} 파일 로드 완료")
            
            # 필수 환경변수 검증
            required_vars = [
                'KIS_APP_KEY', 'KIS_APP_SECRET', 
                'GEMINI_API_KEY', 'PERPLEXITY_API_KEY'
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var) and not os.getenv(var.replace('_', '').replace('API', '').replace('KEY', '')):
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"⚠️ 누락된 환경변수: {missing_vars}")
                return False
            
            print("✅ 모든 필수 환경변수 확인됨")
            return True
        else:
            print(f"⚠️ {env_file} 파일 없음")
            print("📝 .env 파일을 생성하고 API 키들을 설정하세요")
            create_sample_env_file()
            return False
            
    except Exception as e:
        print(f"❌ 환경변수 로드 실패: {e}")
        return False

def create_sample_env_file():
    """샘플 .env 파일 생성"""
    sample_content = """# API 키들 (실제 값으로 교체하세요!)
KIS_APP_KEY=your_kis_app_key_here
KIS_APP_SECRET=your_kis_app_secret_here
MKEY=your_kis_app_key_here
MKEYS=your_kis_app_secret_here

GEMINI_API_KEY=your_gemini_api_key_here
GEM=your_gemini_api_key_here

PERPLEXITY_API_KEY=your_perplexity_api_key_here
PPL=your_perplexity_api_key_here

# 시스템 설정
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
MAX_SYMBOLS=10
CACHE_HOURS=48

# 서버 설정
HOST=0.0.0.0
PORT=7860
SHARE=false
"""
    
    try:
        with open('.env.sample', 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print("📝 .env.sample 파일이 생성되었습니다")
        print("   실제 API 키로 수정 후 .env로 이름을 변경하세요")
    except Exception as e:
        print(f"❌ 샘플 파일 생성 실패: {e}")

def get_env_var(key: str, default: str = "") -> str:
    """환경변수 안전하게 가져오기"""
    return os.getenv(key, default)

def validate_api_keys() -> dict:
    """API 키 유효성 검사"""
    validation_result = {
        'kis': bool(get_env_var('KIS_APP_KEY') or get_env_var('MKEY')),
        'gemini': bool(get_env_var('GEMINI_API_KEY') or get_env_var('GEM')),
        'perplexity': bool(get_env_var('PERPLEXITY_API_KEY') or get_env_var('PPL')),
    }
    
    print("🔑 API 키 상태:")
    for service, is_valid in validation_result.items():
        status = "✅" if is_valid else "❌"
        print(f"   {service}: {status}")
    
    return validation_result
