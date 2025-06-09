#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비동기 처리 유틸리티
Enhanced 퀀트 시스템용 비동기 작업 관리
"""
import asyncio
import concurrent.futures
import functools
import time
import logging
from typing import Any, Callable, Coroutine, List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def run_async_safely(coro: Coroutine) -> Any:
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

async def run_with_timeout(coro: Coroutine, timeout_seconds: float = 30.0) -> Any:
    """타임아웃과 함께 코루틴 실행"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"비동기 작업 타임아웃: {timeout_seconds}초")
        raise
    except Exception as e:
        logger.error(f"비동기 작업 실패: {e}")
        raise

async def retry_async(
    coro_func: Callable[..., Coroutine],
    args: tuple = (),
    kwargs: dict = None,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """재시도 로직을 포함한 비동기 함수 실행"""
    kwargs = kwargs or {}
    
    for attempt in range(max_retries + 1):
        try:
            result = await coro_func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"재시도 성공: {attempt + 1}번째 시도")
            return result
            
        except exceptions as e:
            if attempt == max_retries:
                logger.error(f"최대 재시도 횟수 초과: {max_retries}회")
                raise
            
            wait_time = delay * (backoff_factor ** attempt)
            logger.warning(f"재시도 {attempt + 1}/{max_retries}: {e}, {wait_time:.1f}초 대기")
            await asyncio.sleep(wait_time)

async def gather_with_concurrency(
    coroutines: List[Coroutine], 
    max_concurrency: int = 5,
    return_exceptions: bool = True
) -> List[Any]:
    """동시 실행 개수를 제한하는 gather"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def limited_coro(coro):
        async with semaphore:
            return await coro
    
    limited_coroutines = [limited_coro(coro) for coro in coroutines]
    return await asyncio.gather(*limited_coroutines, return_exceptions=return_exceptions)

async def batch_process(
    items: List[Any],
    async_func: Callable[[Any], Coroutine],
    batch_size: int = 10,
    max_concurrency: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """배치 단위로 비동기 처리"""
    results = []
    total_items = len(items)
    
    for i in range(0, total_items, batch_size):
        batch = items[i:i + batch_size]
        batch_coroutines = [async_func(item) for item in batch]
        
        logger.info(f"배치 처리 중: {i + 1}-{min(i + batch_size, total_items)}/{total_items}")
        
        batch_results = await gather_with_concurrency(
            batch_coroutines, 
            max_concurrency=max_concurrency
        )
        
        results.extend(batch_results)
        
        if progress_callback:
            progress_callback(min(i + batch_size, total_items), total_items)
        
        # 배치 간 짧은 휴식
        if i + batch_size < total_items:
            await asyncio.sleep(0.1)
    
    return results

class AsyncRateLimiter:
    """비동기 레이트 리미터"""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_called = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """레이트 리미트 획득"""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_called
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_called = time.time()

class AsyncTaskManager:
    """비동기 작업 관리자"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def submit_task(
        self, 
        task_id: str, 
        coro: Coroutine,
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> str:
        """작업 제출"""
        async with self._lock:
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                logger.warning(f"최대 동시 작업 수 초과: {self.max_concurrent_tasks}")
                return None
            
            if timeout:
                coro = asyncio.wait_for(coro, timeout=timeout)
            
            task = asyncio.create_task(coro)
            task.add_done_callback(
                lambda t: asyncio.create_task(self._task_completed(task_id, t))
            )
            
            self.active_tasks[task_id] = task
            logger.info(f"작업 시작: {task_id}")
            return task_id
    
    async def _task_completed(self, task_id: str, task: asyncio.Task):
        """작업 완료 콜백"""
        async with self._lock:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            result_info = {
                'task_id': task_id,
                'completed_at': datetime.now(),
                'success': not task.exception(),
                'result': None,
                'error': None
            }
            
            if task.exception():
                result_info['error'] = str(task.exception())
                logger.error(f"작업 실패: {task_id} - {task.exception()}")
            else:
                try:
                    result_info['result'] = task.result()
                    logger.info(f"작업 완료: {task_id}")
                except Exception as e:
                    result_info['error'] = str(e)
            
            self.completed_tasks.append(result_info)
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """특정 작업 완료 대기"""
        if task_id not in self.active_tasks:
            return None
        
        try:
            if timeout:
                return await asyncio.wait_for(self.active_tasks[task_id], timeout=timeout)
            else:
                return await self.active_tasks[task_id]
        except asyncio.TimeoutError:
            logger.warning(f"작업 대기 타임아웃: {task_id}")
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """작업 취소"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            del self.active_tasks[task_id]
            logger.info(f"작업 취소: {task_id}")
            return True
        return False
    
    def get_active_tasks(self) -> List[str]:
        """활성 작업 목록"""
        return list(self.active_tasks.keys())
    
    def get_completed_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """완료된 작업 목록"""
        return self.completed_tasks[-limit:]

async def schedule_periodic_task(
    coro_func: Callable[..., Coroutine],
    interval_seconds: float,
    args: tuple = (),
    kwargs: dict = None,
    max_iterations: Optional[int] = None,
    stop_on_error: bool = False
) -> List[Any]:
    """주기적 작업 스케줄링"""
    kwargs = kwargs or {}
    results = []
    iteration = 0
    
    while max_iterations is None or iteration < max_iterations:
        try:
            start_time = time.time()
            result = await coro_func(*args, **kwargs)
            results.append(result)
            
            # 다음 실행까지 대기
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_seconds - elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            
            iteration += 1
            
        except Exception as e:
            logger.error(f"주기적 작업 오류: {e}")
            if stop_on_error:
                break
            results.append(e)
            await asyncio.sleep(interval_seconds)
    
    return results

async def async_cache_with_ttl(
    cache_dict: Dict[str, Tuple[Any, float]],
    key: str,
    coro_func: Callable[..., Coroutine],
    ttl_seconds: float = 300,
    args: tuple = (),
    kwargs: dict = None
) -> Any:
    """TTL이 있는 비동기 캐시"""
    kwargs = kwargs or {}
    current_time = time.time()
    
    # 캐시 확인
    if key in cache_dict:
        value, timestamp = cache_dict[key]
        if current_time - timestamp < ttl_seconds:
            logger.debug(f"캐시 히트: {key}")
            return value
    
    # 캐시 미스 또는 만료
    logger.debug(f"캐시 미스: {key}")
    result = await coro_func(*args, **kwargs)
    cache_dict[key] = (result, current_time)
    
    return result

def async_timing_decorator(func: Callable) -> Callable:
    """비동기 함수 실행 시간 측정 데코레이터"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func.__name__} 실행 시간: {elapsed_time:.3f}초")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"{func.__name__} 실행 실패 ({elapsed_time:.3f}초): {e}")
            raise
    return wrapper

def sync_to_async(func: Callable) -> Callable[..., Coroutine]:
    """동기 함수를 비동기로 변환"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, func, *args, **kwargs)
    return wrapper

class AsyncContextManager:
    """비동기 컨텍스트 매니저 유틸리티"""
    
    def __init__(self, setup_coro: Coroutine, cleanup_coro: Optional[Coroutine] = None):
        self.setup_coro = setup_coro
        self.cleanup_coro = cleanup_coro
        self.resource = None
    
    async def __aenter__(self):
        self.resource = await self.setup_coro
        return self.resource
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_coro:
            await self.cleanup_coro

# 편의 함수들
async def sleep_with_jitter(base_seconds: float, jitter_factor: float = 0.1):
    """지터를 포함한 sleep"""
    import random
    jitter = random.uniform(-jitter_factor, jitter_factor) * base_seconds
    sleep_time = max(0.01, base_seconds + jitter)
    await asyncio.sleep(sleep_time)

async def wait_for_condition(
    condition_func: Callable[[], bool],
    timeout: float = 30.0,
    check_interval: float = 0.5
) -> bool:
    """조건 만족까지 대기"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        await asyncio.sleep(check_interval)
    
    return False

# 성능 모니터링
class AsyncPerformanceMonitor:
    """비동기 성능 모니터"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    async def measure(self, name: str, coro: Coroutine) -> Any:
        """코루틴 실행 시간 측정"""
        start_time = time.time()
        try:
            result = await coro
            elapsed = time.time() - start_time
            
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(elapsed)
            
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{name} 실행 실패 ({elapsed:.3f}초): {e}")
            raise
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """성능 통계 조회"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        times = self.metrics[name]
        return {
            'count': len(times),
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'total': sum(times)
        }

# 전역 인스턴스들
performance_monitor = AsyncPerformanceMonitor()
task_manager = AsyncTaskManager()

# 테스트용 함수
async def test_async_utils():
    """비동기 유틸리티 테스트"""
    logger.info("🧪 비동기 유틸리티 테스트 시작")
    
    # 기본 함수 테스트
    async def dummy_task(value: int) -> int:
        await asyncio.sleep(0.1)
        return value * 2
    
    # 배치 처리 테스트
    items = list(range(10))
    results = await batch_process(items, dummy_task, batch_size=3, max_concurrency=2)
    logger.info(f"배치 처리 결과: {results}")
    
    # 재시도 테스트
    async def failing_task():
        import random
        if random.random() < 0.7:
            raise ValueError("Random failure")
        return "success"
    
    try:
        result = await retry_async(failing_task, max_retries=3)
        logger.info(f"재시도 결과: {result}")
    except Exception as e:
        logger.error(f"재시도 실패: {e}")
    
    logger.info("✅ 비동기 유틸리티 테스트 완료")

if __name__ == "__main__":
    # 테스트 실행
    run_async_safely(test_async_utils())
