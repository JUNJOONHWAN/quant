"""Finish Qwen3-8B LoRA training, frozen evaluation, and release creation.

This is the durable post-training lane.  It never mutates the training dataset,
never resumes FMP collection, and never enables Quant AI Radar timers.  It waits
for the already-running training unit, resumes only when the final adapter is
missing, serves that exact adapter on an isolated local endpoint, evaluates the
entire sealed test set, and creates a release only after every gate is green.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from training.quant_llm.evaluate_frozen_test import adapter_artifact_set
from workflows.quant_ai_radar.model_runtime import load_model_release


REPO = Path("/home/zooh/Documents/GitHub/quant")
DATA_ROOT = Path("/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM")
MODEL_ROOT = Path("/home/zooh/models/Qwen3-8B-bf16")
CHECKPOINT_ROOT = DATA_ROOT / "checkpoints/qwen3_8b_quant_lora_v1"
DATASET_MANIFEST = DATA_ROOT / "datasets/qwen3_8b_sft_v2/manifest.json"
FROZEN_TEST = DATA_ROOT / "datasets/qwen3_8b_candidate_v3/test.jsonl"
EVALUATION_ROOT = DATA_ROOT / "evaluations/qwen3_8b_quant_lora_v1"
PREDICTIONS = EVALUATION_ROOT / "predictions.jsonl"
EVALUATION_REPORT = EVALUATION_ROOT / "frozen_test_evaluation.json"
RELEASE_MANIFEST = DATA_ROOT / "releases/qwen3_8b_quant_lora_v1/release_manifest.json"
STATUS_PATH = DATA_ROOT / "status/qwen3_8b_training_completion.json"
STATUS_HISTORY = DATA_ROOT / "status/qwen3_8b_training_completion_history.jsonl"
RADAR_ENV = Path("/home/zooh/.config/quant/quant-ai-radar.env")
TRAINING_UNIT = "quant-qwen3-lora-train.service"
TRAIN_SCRIPT = REPO / "training/quant_llm/run_train.sh"
ENDPOINT_MODEL = "qwen3-8b-quant-lora-v1"
BASE_SERVED_MODEL = "qwen3-8b-base"
ENDPOINT = "http://127.0.0.1:8018/v1/chat/completions"
MODELS_ENDPOINT = "http://127.0.0.1:8018/v1/models"
VLLM_CONTAINER = "quant-qwen3-lora-vllm-8018"
VLLM_IMAGE = "vllm/vllm-openai:v0.25.0"
CHECKPOINT_PERMISSION_IMAGE = os.environ.get(
    "NEMO_AUTOMODEL_IMAGE", "nvcr.io/nvidia/nemo-automodel:26.06.00"
)
# NeMo StepScheduler is zero-based while the progress total is a count.  With
# max_steps=14028, is_last_step becomes true at step 14027 and that number is
# used in the final epoch_<E>_step_<S> checkpoint directory.
EXPECTED_FINAL_CHECKPOINT_STEP = 14027
CHECKPOINT_RE = re.compile(r"^epoch_(\d+)_step_(\d+)$")
STEP_LOG_RE = re.compile(r"step\s+(\d+)\s+\|\s+epoch")


@dataclass(frozen=True)
class AdapterCandidate:
    checkpoint_dir: Path
    model_dir: Path
    epoch: int
    step: int
    weights: Path
    config: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamps() -> dict[str, str]:
    now = datetime.now(timezone.utc)
    return {
        "checked_at_utc": now.isoformat(),
        "checked_at_kst": now.astimezone(ZoneInfo("Asia/Seoul")).isoformat(),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_status(phase: str, **details: Any) -> dict[str, Any]:
    value = {
        "schema_version": "quant.training_completion_status.v1",
        "phase": phase,
        **timestamps(),
        **details,
    }
    write_json_atomic(STATUS_PATH, value)
    STATUS_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return value


def discover_final_adapter(
    checkpoint_root: Path, expected_final_step: int
) -> AdapterCandidate | None:
    candidates: list[AdapterCandidate] = []
    if not checkpoint_root.is_dir():
        return None
    for path in checkpoint_root.iterdir():
        if not path.is_dir():
            continue
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        epoch, step = (int(value) for value in match.groups())
        model_dir = path / "model"
        weights = model_dir / "adapter_model.safetensors"
        config = model_dir / "adapter_config.json"
        if step < expected_final_step:
            continue
        if not weights.is_file() or weights.stat().st_size <= 0:
            continue
        if not config.is_file() or config.stat().st_size <= 0:
            continue
        candidates.append(
            AdapterCandidate(path, model_dir, epoch, step, weights, config)
        )
    return max(candidates, key=lambda item: (item.step, item.epoch), default=None)


def run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("RUN " + " ".join(command), flush=True)
    return subprocess.run(
        list(command),
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        check=check,
        text=True,
    )


def build_checkpoint_permission_command(
    candidate: AdapterCandidate, artifacts: Sequence[Path]
) -> list[str]:
    container_paths: list[str] = []
    for artifact in artifacts:
        try:
            relative = artifact.relative_to(candidate.checkpoint_dir)
        except ValueError as exc:
            raise ValueError(
                f"checkpoint artifact is outside candidate directory: {artifact}"
            ) from exc
        container_paths.append(str(Path("/checkpoint") / relative))
    return [
        "docker",
        "run",
        "--rm",
        "--user",
        "0:0",
        "-v",
        f"{candidate.checkpoint_dir}:/checkpoint",
        "--entrypoint",
        "/usr/bin/chmod",
        CHECKPOINT_PERMISSION_IMAGE,
        "a+r",
        *container_paths,
    ]


def ensure_candidate_artifacts_readable(candidate: AdapterCandidate) -> None:
    artifacts = (candidate.weights, candidate.config)
    unreadable = [
        artifact for artifact in artifacts if not os.access(artifact, os.R_OK)
    ]
    if not unreadable:
        return
    run(build_checkpoint_permission_command(candidate, unreadable))
    still_unreadable = [
        artifact for artifact in artifacts if not os.access(artifact, os.R_OK)
    ]
    if still_unreadable:
        raise PermissionError(
            "checkpoint artifacts remain unreadable after permission repair: "
            + ", ".join(str(path) for path in still_unreadable)
        )


def training_unit_active() -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", TRAINING_UNIT],
        check=False,
    ).returncode == 0


def latest_logged_step() -> int | None:
    result = subprocess.run(
        ["journalctl", "--user", "-u", TRAINING_UNIT, "-n", "600", "-o", "cat"],
        check=False,
        capture_output=True,
        text=True,
    )
    matches = [int(value) for value in STEP_LOG_RE.findall(result.stdout)]
    return max(matches, default=None)


def wait_for_training(wait_seconds: int, expected_final_step: int) -> AdapterCandidate | None:
    while training_unit_active():
        candidate = discover_final_adapter(CHECKPOINT_ROOT, expected_final_step)
        record_status(
            "training",
            training_unit=TRAINING_UNIT,
            latest_logged_step=latest_logged_step(),
            expected_final_step=expected_final_step,
            final_adapter_ready=candidate is not None,
        )
        time.sleep(wait_seconds)
    return discover_final_adapter(CHECKPOINT_ROOT, expected_final_step)


def port_is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def endpoint_models() -> set[str]:
    try:
        with urllib.request.urlopen(MODELS_ENDPOINT, timeout=10) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return set()
    return {
        str(item.get("id"))
        for item in value.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }


def build_vllm_command(candidate: AdapterCandidate) -> list[str]:
    return [
        "docker",
        "run",
        "-d",
        "--name",
        VLLM_CONTAINER,
        "--restart",
        "unless-stopped",
        "--gpus",
        "all",
        "--ipc=host",
        "-p",
        "127.0.0.1:8018:8000",
        "-v",
        f"{MODEL_ROOT}:/model:ro",
        "-v",
        f"{candidate.model_dir}:/adapter:ro",
        VLLM_IMAGE,
        "/model",
        "--served-model-name",
        BASE_SERVED_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.70",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "8",
        "--enable-lora",
        "--max-lora-rank",
        "8",
        "--lora-modules",
        f"{ENDPOINT_MODEL}=/adapter",
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
        "--generation-config",
        "vllm",
    ]


def container_exists() -> bool:
    return subprocess.run(
        ["docker", "inspect", VLLM_CONTAINER],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def inspect_candidate_container() -> dict[str, Any] | None:
    result = subprocess.run(
        ["docker", "inspect", VLLM_CONTAINER],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = json.loads(result.stdout)
    return value[0] if isinstance(value, list) and value and isinstance(value[0], dict) else None


def container_matches_candidate(value: dict[str, Any], candidate: AdapterCandidate) -> bool:
    mounts = {
        str(item.get("Destination")): str(item.get("Source"))
        for item in value.get("Mounts", [])
        if isinstance(item, dict)
    }
    config = value.get("Config") or {}
    command = [str(item) for item in config.get("Cmd", [])]
    return bool(
        config.get("Image") == VLLM_IMAGE
        and mounts.get("/model") == str(MODEL_ROOT)
        and mounts.get("/adapter") == str(candidate.model_dir)
        and f"{ENDPOINT_MODEL}=/adapter" in command
    )


def start_candidate_server(candidate: AdapterCandidate) -> None:
    inspection = inspect_candidate_container()
    matching_container = inspection is not None and container_matches_candidate(inspection, candidate)
    if ENDPOINT_MODEL in endpoint_models() and matching_container:
        return
    if inspection is not None:
        if matching_container:
            run(["docker", "start", VLLM_CONTAINER], check=False)
            for _ in range(18):
                if ENDPOINT_MODEL in endpoint_models():
                    return
                time.sleep(10)
        run(["docker", "stop", "--time", "30", VLLM_CONTAINER], check=False)
        run(["docker", "rm", VLLM_CONTAINER])
    elif port_is_listening("127.0.0.1", 8018):
        raise RuntimeError("port 8018 is occupied by an unverified process")
    run(build_vllm_command(candidate))
    for _ in range(120):
        if ENDPOINT_MODEL in endpoint_models():
            return
        time.sleep(10)
    raise RuntimeError("candidate vLLM endpoint did not expose the LoRA model within 20 minutes")


def candidate_adapter_sha(candidate: AdapterCandidate) -> str:
    return adapter_artifact_set(
        candidate.checkpoint_dir.parent, [candidate.weights, candidate.config]
    )[0]


def collect_predictions(candidate: AdapterCandidate) -> None:
    adapter_sha = candidate_adapter_sha(candidate)
    frozen_test_sha = sha256_file(FROZEN_TEST)
    run(
        [
            sys.executable,
            "-m",
            "training.quant_llm.collect_frozen_predictions",
            "--test-file",
            str(FROZEN_TEST),
            "--endpoint",
            ENDPOINT,
            "--endpoint-model",
            ENDPOINT_MODEL,
            "--adapter-set-sha256",
            adapter_sha,
            "--frozen-test-sha256",
            frozen_test_sha,
            "--output",
            str(PREDICTIONS),
            "--workers",
            "4",
        ]
    )


def evaluate(candidate: AdapterCandidate) -> int:
    result = run(
        [
            sys.executable,
            "-m",
            "training.quant_llm.evaluate_frozen_test",
            "--dataset-manifest",
            str(DATASET_MANIFEST),
            "--test-file",
            str(FROZEN_TEST),
            "--predictions",
            str(PREDICTIONS),
            "--endpoint-model",
            ENDPOINT_MODEL,
            "--adapter-root",
            str(CHECKPOINT_ROOT),
            "--artifact",
            str(candidate.weights),
            "--artifact",
            str(candidate.config),
            "--output",
            str(EVALUATION_REPORT),
        ],
        check=False,
    )
    return result.returncode


def create_release(candidate: AdapterCandidate) -> None:
    run(
        [
            sys.executable,
            "-m",
            "training.quant_llm.create_model_release",
            "--model-id",
            ENDPOINT_MODEL,
            "--endpoint-model",
            ENDPOINT_MODEL,
            "--base-model",
            str(MODEL_ROOT),
            "--adapter-root",
            str(CHECKPOINT_ROOT),
            "--artifact",
            str(candidate.weights),
            "--artifact",
            str(candidate.config),
            "--dataset-manifest",
            str(DATASET_MANIFEST),
            "--evaluation-report",
            str(EVALUATION_REPORT),
            "--output",
            str(RELEASE_MANIFEST),
        ]
    )


def ensure_radar_env() -> None:
    expected = {
        "QUANT_AI_MODEL_ENDPOINT": ENDPOINT,
        "QUANT_AI_RELEASE_MANIFEST": str(RELEASE_MANIFEST),
        "QUANT_AI_WORKERS": "4",
    }
    if RADAR_ENV.is_file():
        observed: dict[str, str] = {}
        for line in RADAR_ENV.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                observed[key.strip()] = value.strip()
        mismatched = {key: value for key, value in expected.items() if observed.get(key) != value}
        if mismatched:
            raise RuntimeError(f"existing Quant AI Radar environment conflicts with accepted release: {mismatched}")
        return
    RADAR_ENV.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".quant-ai-radar.env.", dir=RADAR_ENV.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for key, value in expected.items():
                handle.write(f"{key}={value}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, RADAR_ENV)
    finally:
        temporary.unlink(missing_ok=True)


def accepted_release_status() -> dict[str, Any] | None:
    if not RELEASE_MANIFEST.is_file():
        return None
    binding = load_model_release(RELEASE_MANIFEST)
    return {
        "release_manifest": str(RELEASE_MANIFEST),
        "release_manifest_sha256": sha256_file(RELEASE_MANIFEST),
        "model_id": binding.model_id,
        "endpoint_model": binding.endpoint_model,
        "adapter_root": str(binding.adapter_root),
        "adapter_set_sha256": binding.adapter_set_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-final-step", type=int, default=EXPECTED_FINAL_CHECKPOINT_STEP
    )
    parser.add_argument("--wait-seconds", type=int, default=300)
    parser.add_argument("--max-resume-attempts", type=int, default=5)
    args = parser.parse_args()
    if args.expected_final_step <= 0 or args.wait_seconds <= 0:
        raise ValueError("step and wait values must be positive")

    accepted = accepted_release_status()
    if accepted is not None:
        ensure_radar_env()
        record_status("accepted", **accepted)
        return 0

    candidate = wait_for_training(args.wait_seconds, args.expected_final_step)
    resume_attempts = 0
    while candidate is None:
        if resume_attempts >= args.max_resume_attempts:
            record_status(
                "training_incomplete",
                expected_final_step=args.expected_final_step,
                resume_attempts=resume_attempts,
            )
            return 1
        resume_attempts += 1
        record_status(
            "resuming_training",
            expected_final_step=args.expected_final_step,
            resume_attempt=resume_attempts,
        )
        result = run(["/usr/bin/bash", str(TRAIN_SCRIPT)], check=False)
        candidate = discover_final_adapter(CHECKPOINT_ROOT, args.expected_final_step)
        if result.returncode != 0 and candidate is None:
            time.sleep(args.wait_seconds)

    ensure_candidate_artifacts_readable(candidate)
    checkpoint_details = {
        "checkpoint_dir": str(candidate.checkpoint_dir),
        "checkpoint_epoch": candidate.epoch,
        "checkpoint_step": candidate.step,
        "adapter_weights_sha256": sha256_file(candidate.weights),
        "adapter_config_sha256": sha256_file(candidate.config),
    }
    record_status("serving_candidate", **checkpoint_details)
    start_candidate_server(candidate)
    record_status("collecting_frozen_predictions", **checkpoint_details)
    collect_predictions(candidate)
    record_status("evaluating_frozen_test", **checkpoint_details)
    evaluation_code = evaluate(candidate)
    evaluation = json.loads(EVALUATION_REPORT.read_text(encoding="utf-8"))
    if evaluation_code != 0 or evaluation.get("status") != "green":
        record_status(
            "evaluation_red",
            evaluation_report=str(EVALUATION_REPORT),
            metrics=evaluation.get("metrics"),
            failed_gates=sorted(
                key
                for key, value in (evaluation.get("required_gates") or {}).items()
                if value is not True
            ),
            **checkpoint_details,
        )
        return 2
    record_status("creating_release", **checkpoint_details)
    create_release(candidate)
    accepted = accepted_release_status()
    if accepted is None:
        raise RuntimeError("release command returned without a verifiable manifest")
    ensure_radar_env()
    record_status(
        "accepted",
        evaluation_report=str(EVALUATION_REPORT),
        metrics=evaluation.get("metrics"),
        **checkpoint_details,
        **accepted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
