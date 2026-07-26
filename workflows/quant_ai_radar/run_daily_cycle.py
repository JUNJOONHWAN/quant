#!/usr/bin/env python3
"""Run data refresh followed by the accepted-model full-universe radar."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from workflows.quant_ai_radar.universe import write_json


KST = ZoneInfo("Asia/Seoul")
QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")
STATE_PATH = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/status/daily_cycle.json"
)


class CycleError(RuntimeError):
    pass


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise CycleError(f"required environment variable is missing: {name}")
    return value


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=QUANT_ROOT, text=True, check=False)
    if completed.returncode != 0:
        raise CycleError(
            f"stage failed ({completed.returncode}): {' '.join(command)}"
        )


def main() -> int:
    model_endpoint = _required_env("QUANT_AI_MODEL_ENDPOINT")
    release_manifest = os.environ.get(
        "QUANT_AI_RELEASE_MANIFEST",
        "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/"
        "qwen3_8b_quant_lora_v1/release_manifest.json",
    )
    command = [
        sys.executable,
        "-m",
        "workflows.quant_ai_radar.refresh_daily_data",
    ]
    state = {
        "schema_version": "quant.ai_radar_daily_cycle.v1",
        "status": "running_data_refresh",
        "started_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
    }
    write_json(STATE_PATH, state)
    try:
        _run(command)
        state["status"] = "running_full_universe_inference"
        write_json(STATE_PATH, state)
        command = [
            sys.executable,
            "-m",
            "workflows.quant_ai_radar.run_quant_ai_radar",
            "--release-manifest",
            release_manifest,
            "--model-endpoint",
            model_endpoint,
            "--workers",
            os.environ.get("QUANT_AI_WORKERS", "4"),
        ]
        token_file = os.environ.get("QUANT_AI_MODEL_TOKEN_FILE", "").strip()
        if token_file:
            command.extend(["--model-token-file", token_file])
        _run(command)
    except Exception as exc:
        state.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
            }
        )
        write_json(STATE_PATH, state)
        print(json.dumps(state, ensure_ascii=False))
        return 1
    state.update(
        {
            "status": "complete",
            "completed_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        }
    )
    write_json(STATE_PATH, state)
    print(json.dumps(state, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
