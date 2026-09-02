"""Cross-process rate limiting for shared DGX market-data credentials."""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional


DEFAULT_RATE_LIMIT_ROOT = Path("~/.cache/quant_dataset/rate_limits").expanduser()


@dataclass(frozen=True)
class RateLimitSpec:
    name: str
    max_calls: int
    period_seconds: float
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


RATE_LIMIT_SPECS = {
    "fmp": RateLimitSpec(
        name="fmp",
        max_calls=240,
        period_seconds=60.0,
        rationale="80_percent_of_300_per_minute_shared_key_limit_with_external_headroom",
    ),
    "massive": RateLimitSpec(
        name="massive",
        max_calls=2,
        period_seconds=1.0,
        rationale="conservative_shared_rest_limit_grouped_jobs_remain_serial",
    ),
}


class FileWindowRateLimiter:
    """A sliding-window limiter shared by every process using the same file."""

    def __init__(
        self,
        path: Path,
        spec: RateLimitSpec,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ):
        self.path = Path(path).expanduser()
        self.spec = spec
        self.sleep = sleep
        self.clock = clock
        if self.spec.max_calls < 1 or self.spec.period_seconds <= 0:
            raise ValueError("invalid rate-limit specification")

    @staticmethod
    def _read_timestamps(handle) -> list:
        handle.seek(0)
        try:
            document = json.loads(handle.read() or "[]")
        except ValueError:
            return []
        if not isinstance(document, list):
            return []
        result = []
        for value in document:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result.append(float(value))
        return sorted(result)

    @staticmethod
    def _write_timestamps(handle, timestamps: list) -> None:
        handle.seek(0)
        handle.truncate()
        json.dump(timestamps, handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())

    def acquire(self) -> float:
        """Reserve one request slot, blocking only when the shared window is full."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        waited = 0.0
        while True:
            with self.path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                now = float(self.clock())
                cutoff = now - self.spec.period_seconds
                timestamps = [item for item in self._read_timestamps(handle) if item > cutoff]
                if len(timestamps) < self.spec.max_calls:
                    timestamps.append(now)
                    self._write_timestamps(handle, timestamps)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    return waited
                delay = max(
                    timestamps[0] + self.spec.period_seconds - now + 0.001,
                    0.001,
                )
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            self.sleep(delay)
            waited += delay


def build_default_rate_limiters(
    root: Path = DEFAULT_RATE_LIMIT_ROOT,
) -> Dict[str, FileWindowRateLimiter]:
    base = Path(root).expanduser()
    return {
        source: FileWindowRateLimiter(base / (source + ".json"), spec)
        for source, spec in RATE_LIMIT_SPECS.items()
    }


def rate_limit_policy() -> dict:
    return {source: spec.to_dict() for source, spec in RATE_LIMIT_SPECS.items()}
