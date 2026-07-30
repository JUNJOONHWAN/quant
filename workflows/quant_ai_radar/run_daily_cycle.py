#!/usr/bin/env python3
"""Run shared-data prepare followed by prioritized accepted-model analysis."""

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


def build_stage_commands(
    *,
    model_endpoint: str,
    release_manifest: str,
    workers: str,
    token_file: str = "",
    max_constituent_available_lag_days: str = "45",
    constituent_stale_days: str = "45",
    constituent_refresh_max_etfs: str = "50",
    publish_grace_hour_et: str = "18",
    max_ai_etfs: str = "64",
    max_ai_stocks: str = "192",
) -> list[list[str]]:
    prepare = [
        sys.executable,
        "-m",
        "workflows.quant_ai_radar.prepare_shared_data",
        "--max-constituent-available-lag-days",
        max_constituent_available_lag_days,
        "--constituent-stale-days",
        constituent_stale_days,
        "--constituent-refresh-max-etfs",
        constituent_refresh_max_etfs,
        "--publish-grace-hour-et",
        publish_grace_hour_et,
    ]
    radar = [
        sys.executable,
        "-m",
        "workflows.quant_ai_radar.run_quant_ai_radar",
        "--release-manifest",
        release_manifest,
        "--model-endpoint",
        model_endpoint,
        "--workers",
        workers,
        "--max-constituent-available-lag-days",
        max_constituent_available_lag_days,
        "--max-ai-etfs",
        max_ai_etfs,
        "--max-ai-stocks",
        max_ai_stocks,
    ]
    if token_file:
        radar.extend(["--model-token-file", token_file])
    return [prepare, radar]


def main() -> int:
    model_endpoint = _required_env("QUANT_AI_MODEL_ENDPOINT")
    release_manifest = os.environ.get(
        "QUANT_AI_RELEASE_MANIFEST",
        "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/"
        "qwen3_8b_quant_lora_v1/release_manifest.json",
    )
    commands = build_stage_commands(
        model_endpoint=model_endpoint,
        release_manifest=release_manifest,
        workers=os.environ.get("QUANT_AI_WORKERS", "4"),
        token_file=os.environ.get("QUANT_AI_MODEL_TOKEN_FILE", "").strip(),
        max_constituent_available_lag_days=os.environ.get(
            "QUANT_AI_MAX_CONSTITUENT_AVAILABLE_LAG_DAYS", "45"
        ),
        constituent_stale_days=os.environ.get(
            "QUANT_AI_CONSTITUENT_STALE_DAYS", "45"
        ),
        constituent_refresh_max_etfs=os.environ.get(
            "QUANT_AI_CONSTITUENT_REFRESH_MAX_ETFS", "50"
        ),
        publish_grace_hour_et=os.environ.get(
            "QUANT_AI_ORACLE_PUBLISH_GRACE_HOUR_ET", "18"
        ),
        max_ai_etfs=os.environ.get("QUANT_AI_MAX_ETFS", "64"),
        max_ai_stocks=os.environ.get("QUANT_AI_MAX_STOCKS", "192"),
    )
    state = {
        "schema_version": "quant.ai_radar_daily_cycle.v1",
        "status": "running_shared_oracle_store_prepare",
        "started_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "source_ownership": "market_structure_oracle_single_writer",
        "duplicate_fmp_massive_collection": False,
    }
    write_json(STATE_PATH, state)
    try:
        _run(commands[0])
        state["status"] = "running_full_scan_prioritized_inference"
        write_json(STATE_PATH, state)
        _run(commands[1])
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
