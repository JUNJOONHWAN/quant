"""Read-only Spark readiness gate; it never stops a live model service."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_IMAGE = "nvcr.io/nvidia/nemo-automodel:26.06.00"
DEFAULT_MODEL = Path("/home/zooh/models/Qwen3-8B-bf16")
MODE_MIN_FREE_MIB = {"smoke": 36_000, "full": 48_000}


def _run(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _integer_query(command: Sequence[str]) -> Optional[int]:
    result = _run(command)
    if result.returncode:
        return None
    values = []
    for line in result.stdout.splitlines():
        token = line.strip().split()[0].replace("MiB", "") if line.strip() else ""
        try:
            values.append(int(token))
        except ValueError:
            continue
    return min(values) if values else None


def _mem_available_mib(meminfo_path: Path = Path("/proc/meminfo")) -> Optional[int]:
    """Read unified-memory availability without relying on a login-shell PATH."""

    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                return int(fields[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def inspect_readiness(mode: str, image: str, model_path: Path, min_free_mib: Optional[int] = None) -> dict:
    required_free = min_free_mib if min_free_mib is not None else MODE_MIN_FREE_MIB[mode]
    gpu_free = _integer_query(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    )
    memory_available = _mem_available_mib()
    docker_ps = _run(["docker", "ps", "--format", "{{.Names}}"])
    running = sorted(line.strip() for line in docker_ps.stdout.splitlines() if line.strip())
    model_services = [
        name for name in running if any(token in name.lower() for token in ("qwen", "gemma", "vllm"))
    ]
    image_present = _run(["docker", "image", "inspect", image]).returncode == 0
    model = Path(model_path)
    model_present = (model / "config.json").is_file() and any(model.glob("*.safetensors"))
    measured = [value for value in (gpu_free, memory_available) if value is not None]
    effective_free = min(measured) if measured else None
    errors = []
    if not image_present:
        errors.append("NeMo AutoModel image is not present")
    if not model_present:
        errors.append("Qwen3-8B BF16 model files are not complete")
    if effective_free is None:
        errors.append("free unified memory could not be measured")
    elif effective_free < required_free:
        errors.append(
            "free unified memory {} MiB is below {} MiB {} gate".format(
                effective_free, required_free, mode
            )
        )
    return {
        "ok": not errors,
        "mode": mode,
        "image": image,
        "image_present": image_present,
        "model_path": str(model),
        "model_present": model_present,
        "gpu_free_mib": gpu_free,
        "system_available_mib": memory_available,
        "effective_free_mib": effective_free,
        "required_free_mib": required_free,
        "running_model_services": model_services,
        "automatic_service_stop": False,
        "errors": errors,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODE_MIN_FREE_MIB), default="smoke")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--min-free-mib", type=int)
    args = parser.parse_args(argv)
    result = inspect_readiness(args.mode, args.image, args.model_path, args.min_free_mib)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
