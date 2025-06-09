#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
환경 변수 로드 모듈
"""
import os
from pathlib import Path

def load_env_file():
    """환경 변수 파일 로드"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print("✅ .env 파일 로드 완료")
    else:
        print("⚠️ .env 파일 없음")
