"""
비동기 처리 유틸리티
"""
import asyncio
import concurrent.futures

def run_async_safely(coro):
    """EC2 환경용 안전한 비동기 함수 실행"""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # 기존 루프가 있는 경우 동기적으로 처리
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            raise e
