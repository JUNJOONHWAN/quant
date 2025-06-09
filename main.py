#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced 퀀트 시스템 1.8.2 - 메인 실행 파일
3단 구조: 딥리서치 섹터 + Gemini Flash + 갭필터 부스터
"""
import asyncio
import sys
import os
import logging
import signal
import traceback
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from config.env_loader import load_env_file, validate_api_keys
    from config.settings import GlobalSettings
    from analysis.scoring_engine import Enhanced182ThreeTierSystem
    from interface.gradio_ui import create_three_tier_interface
    from utils.async_utils import run_async_safely
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("📁 현재 디렉토리에서 실행했는지 확인하세요")
    print("📦 필요한 패키지가 설치되었는지 확인하세요: pip install -r requirements.txt")
    sys.exit(1)

def setup_logging():
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / "app.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def validate_environment():
    """환경 검증"""
    logger = logging.getLogger(__name__)
    
    # Python 버전 확인
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.error(f"❌ Python 3.8+ 필요 (현재: {python_version.major}.{python_version.minor})")
        return False
    
    # 필수 디렉토리 생성
    required_dirs = ["logs", "cache", "config", "analysis", "interface", "utils"]
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    logger.info("✅ 환경 검증 완료")
    return True

def check_dependencies():
    """필수 패키지 확인"""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        "gradio", "aiohttp", "requests", "numpy", "pandas", 
        "yfinance", "pydantic", "pathlib"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ 누락된 패키지: {missing_packages}")
        logger.error("다음 명령어로 설치하세요: pip install -r requirements.txt")
        return False
    
    logger.info("✅ 모든 필수 패키지 확인됨")
    return True

def signal_handler(signum, frame):
    """시그널 핸들러 (Ctrl+C 등)"""
    logger = logging.getLogger(__name__)
    logger.info(f"🛑 시그널 {signum} 받음. 시스템 종료 중...")
    sys.exit(0)

def print_startup_banner():
    """시작 배너 출력"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                Enhanced 퀀트 시스템 1.8.2                    ║
║                  3단 구조 AI 투자 분석                       ║
╠══════════════════════════════════════════════════════════════╣
║ 📊 Tier 1: 딥리서치 섹터 가중치 (48시간 캐싱)               ║
║ 🔥 Tier 2: Gemini Flash N점수 (실시간)                      ║
║ 🎯 Tier 3: 갭필터 부스터 + 패턴 인식                        ║
║ 🚀 Enhanced VTNF 통합 시스템                                ║
╠══════════════════════════════════════════════════════════════╣
║ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST                  ║
║ Python: {sys.version.split()[0]}                                      ║
║ 환경: {'Production' if not os.getenv('DEBUG') else 'Development'}                                       ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

async def initialize_system():
    """비동기 시스템 초기화"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🔧 시스템 초기화 중...")
        
        # Enhanced 시스템 초기화
        system = Enhanced182ThreeTierSystem()
        
        # 시스템 상태 확인
        logger.info("📊 시스템 구성요소 확인:")
        logger.info(f"   ✅ 섹터 캐시: {'유효' if system.sector_cache.is_valid() else '갱신 필요'}")
        logger.info(f"   ✅ 갭 부스터: {'활성화' if hasattr(system, 'gap_booster') else '비활성화'}")
        logger.info(f"   ✅ 패턴 인식: {'활성화' if system.pattern_enabled else '비활성화'}")
        logger.info(f"   ✅ 데이터 검증: {'엄격' if system.data_validator.strict_mode else '관대'}")
        
        return system
        
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 실패: {e}")
        logger.error(f"상세 오류:\n{traceback.format_exc()}")
        raise

def create_gradio_interface(system):
    """Gradio 인터페이스 생성"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🌐 Gradio 인터페이스 생성 중...")
        
        # Gradio 인터페이스 생성 (시스템 객체를 전달)
        interface = create_three_tier_interface(system)
        
        logger.info("✅ Gradio 인터페이스 생성 완료")
        return interface
        
    except Exception as e:
        logger.error(f"❌ Gradio 인터페이스 생성 실패: {e}")
        logger.error(f"상세 오류:\n{traceback.format_exc()}")
        raise

def launch_server(interface, settings):
    """서버 실행"""
    logger = logging.getLogger(__name__)
    
    try:
        host = settings.app.host
        port = settings.app.port
        share = settings.app.share
        
        logger.info(f"🚀 서버 시작 중...")
        logger.info(f"   📍 주소: http://{host}:{port}")
        logger.info(f"   🌐 공유: {'활성화' if share else '비활성화'}")
        
        # 서버 실행 (blocking)
        interface.launch(
            server_name=host,
            server_port=port,
            share=share,
            show_error=True,
            quiet=False,
            inbrowser=True if not share else False,
            favicon_path=None,
            ssl_verify=False
        )
        
    except Exception as e:
        logger.error(f"❌ 서버 실행 실패: {e}")
        logger.error(f"상세 오류:\n{traceback.format_exc()}")
        raise

def main():
    """메인 실행 함수"""
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 시작 배너 출력
    print_startup_banner()
    
    # 로깅 설정
    logger = setup_logging()
    logger.info("🚀 Enhanced 퀀트 시스템 시작")
    
    try:
        # 1. 환경 검증
        if not validate_environment():
            logger.error("❌ 환경 검증 실패")
            sys.exit(1)
        
        # 2. 패키지 확인
        if not check_dependencies():
            logger.error("❌ 패키지 확인 실패")
            sys.exit(1)
        
        # 3. 환경변수 로드
        logger.info("📋 환경변수 로드 중...")
        if not load_env_file():
            logger.warning("⚠️ 환경변수 로드 실패, 기본값 사용")
        
        # 4. API 키 검증
        api_status = validate_api_keys()
        if not all(api_status.values()):
            logger.warning("⚠️ 일부 API 키가 누락됨. 일부 기능이 제한될 수 있습니다.")
        
        # 5. 설정 로드
        logger.info("⚙️ 설정 로드 중...")
        settings = GlobalSettings()
        if not settings.validate_all():
            logger.error("❌ 설정 검증 실패")
            sys.exit(1)
        
        # 6. 시스템 초기화 (비동기)
        logger.info("🔧 비동기 시스템 초기화...")
        system = run_async_safely(initialize_system())
        
        # 7. Gradio 인터페이스 생성
        interface = create_gradio_interface(system)
        
        # 8. 서버 실행
        logger.info("🎉 시스템 초기화 완료! 서버 시작...")
        launch_server(interface, settings)
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자가 중단했습니다 (Ctrl+C)")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ 시스템 실행 중 오류 발생: {e}")
        logger.error(f"상세 오류:\n{traceback.format_exc()}")
        
        # 디버깅 정보 출력
        logger.error("🔍 디버깅 정보:")
        logger.error(f"   Python 버전: {sys.version}")
        logger.error(f"   작업 디렉토리: {os.getcwd()}")
        logger.error(f"   Python 경로: {sys.path[:3]}...")
        
        sys.exit(1)
        
    finally:
        logger.info("🏁 Enhanced 퀀트 시스템 종료")

if __name__ == "__main__":
    main()

