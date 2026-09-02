#!/usr/bin/env python3
"""Capture fail-closed DGX runtime evidence for Quant AI Radar activation.

The functional shadow gate and the host-runtime gate are deliberately separate.
An NVIDIA graphics-context allocation warning does not invalidate already
hash-verified analysis output when vLLM stayed healthy, but it does prevent that
run from counting toward automatic timer activation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from workflows.quant_ai_radar.universe import write_json


KST = ZoneInfo("Asia/Seoul")
DEFAULT_CONTAINER = "quant-qwen3-lora-vllm-8018"
DEFAULT_EXPECTED_MODEL = "qwen3-8b-quant-lora-v1"
DEFAULT_MODEL_ENDPOINT = "http://127.0.0.1:8018/v1/models"
RADAR_UNITS = (
    "quant-ai-radar-daily.service",
    "quant-ai-radar-daily.timer",
    "quant-ai-radar-relations-weekly.service",
    "quant-ai-radar-relations-weekly.timer",
)
FMP_BACKFILL_UNIT = "quant-fmp-training-backfill.service"
TIMESTAMP_PATTERN = re.compile(r"^(20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


class RuntimeAuditError(RuntimeError):
    """Required DGX runtime evidence could not be collected."""


def _run(command: Sequence[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _unit_state(unit: str, *, user: bool) -> dict[str, str]:
    prefix = ["systemctl"]
    if user:
        prefix.append("--user")
    enabled = _run([*prefix, "is-enabled", unit])[1] or "unknown"
    active = _run([*prefix, "is-active", unit])[1] or "unknown"
    return {"enabled": enabled, "active": active}


def summarize_kernel_events(text: str) -> dict[str, Any]:
    """Classify kernel failures without conflating NVRM warnings with OOM-kill."""

    buckets: dict[str, int] = {}
    nvrm_lines = 0
    linux_oom_kill_lines = 0
    nvidia_xid_lines = 0
    for line in text.splitlines():
        lowered = line.lower()
        if "nv_err_no_memory" in lowered:
            nvrm_lines += 1
            match = TIMESTAMP_PATTERN.match(line)
            key = match.group(1) if match else "timestamp_unavailable"
            buckets[key] = buckets.get(key, 0) + 1
        if (
            "oom-kill" in lowered
            or "killed process" in lowered
            or "out of memory: kill process" in lowered
        ):
            linux_oom_kill_lines += 1
        if "nvrm: xid" in lowered:
            nvidia_xid_lines += 1
    return {
        "nvrm_nv_err_no_memory_lines": nvrm_lines,
        "nvrm_event_buckets": buckets,
        "linux_oom_kill_lines": linux_oom_kill_lines,
        "nvidia_xid_lines": nvidia_xid_lines,
    }


def _fmp_backfill_processes() -> list[dict[str, Any]]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = [
                token.decode(errors="replace")
                for token in entry.joinpath("cmdline").read_bytes().split(b"\0")
                if token
            ]
        except OSError:
            continue
        if not command or not Path(command[0]).name.startswith("python"):
            continue
        module_match = any(
            command[index : index + 2] == ["-m", "quant_dataset.fmp_training"]
            for index in range(max(len(command) - 1, 0))
        )
        script_match = any(
            Path(token).name in {"fmp_training.py", "fmp_backfill.py"}
            for token in command[1:]
        )
        if module_match or script_match:
            matches.append({"pid": int(entry.name), "command": command})
    return sorted(matches, key=lambda item: item["pid"])


def classify_runtime_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return manual-readiness and stricter timer-activation decisions."""

    kernel = evidence.get("kernel_events") or {}
    vllm = evidence.get("vllm") or {}
    docker = evidence.get("docker") or {}
    radar_units = evidence.get("radar_units") or {}
    fmp = evidence.get("fmp_backfill") or {}
    expected_model = str(evidence.get("expected_model") or "")
    served_models = set(evidence.get("served_models") or [])

    gates = {
        "docker_boot_recovery": (
            docker.get("enabled") == "enabled"
            and docker.get("active") == "active"
        ),
        "vllm_healthy": (
            vllm.get("running") is True
            and vllm.get("oom_killed") is False
            and int(vllm.get("restart_count") or 0) == 0
            and vllm.get("restart_policy") == "unless-stopped"
        ),
        "accepted_model_served": bool(expected_model)
        and expected_model in served_models,
        "user_linger_enabled": evidence.get("user_linger") is True,
        "radar_units_still_disabled": all(
            (radar_units.get(unit) or {}).get("enabled") == "disabled"
            and (radar_units.get(unit) or {}).get("active") == "inactive"
            for unit in RADAR_UNITS
        ),
        "fmp_history_backfill_paused": (
            (fmp.get("unit") or {}).get("enabled") == "disabled"
            and (fmp.get("unit") or {}).get("active") == "inactive"
            and not (fmp.get("processes") or [])
        ),
        "no_linux_oom_kill": int(kernel.get("linux_oom_kill_lines") or 0) == 0,
        "no_nvidia_xid": int(kernel.get("nvidia_xid_lines") or 0) == 0,
    }
    manual_ready = all(gates.values())
    nvrm_count = int(kernel.get("nvrm_nv_err_no_memory_lines") or 0)
    contention_clear = nvrm_count == 0
    timer_eligible = manual_ready and contention_clear
    if not manual_ready:
        status = "fail"
    elif not contention_clear:
        status = "pass_with_resource_contention_watchpoint"
    else:
        status = "pass"
    return {
        "status": status,
        "manual_reference_ready": manual_ready,
        "timer_activation_eligible": timer_eligible,
        "gates": gates,
        "watchpoints": (
            [
                {
                    "code": "nvrm_nv_err_no_memory_during_audit_window",
                    "count": nvrm_count,
                    "effect": (
                        "functional output remains usable because vLLM did not "
                        "OOM/restart and hashes passed; this run does not count "
                        "toward timer activation"
                    ),
                }
            ]
            if nvrm_count
            else []
        ),
    }


def collect_runtime_evidence(args: argparse.Namespace) -> dict[str, Any]:
    journal_command = ["journalctl", "-k", "-b", "--no-pager", "-o", "short-iso"]
    if args.since:
        journal_command.extend(["--since", args.since])
    if args.until:
        journal_command.extend(["--until", args.until])
    return_code, journal, error = _run(journal_command)
    if return_code != 0:
        raise RuntimeAuditError(f"kernel journal unavailable: {error}")

    return_code, raw_inspect, error = _run(["docker", "inspect", args.container])
    if return_code != 0:
        raise RuntimeAuditError(f"vLLM container inspect failed: {error}")
    try:
        inspected = json.loads(raw_inspect)[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise RuntimeAuditError("vLLM container inspect returned invalid JSON") from exc
    state = inspected.get("State") or {}
    restart = (inspected.get("HostConfig") or {}).get("RestartPolicy") or {}

    request = urllib.request.Request(args.model_endpoint, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            model_payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeAuditError(f"vLLM model endpoint failed: {exc}") from exc
    served_models = [
        str(item.get("id"))
        for item in model_payload.get("data") or []
        if isinstance(item, Mapping) and item.get("id")
    ]

    docker_unit = _unit_state("docker.service", user=False)
    radar_units = {unit: _unit_state(unit, user=True) for unit in RADAR_UNITS}
    fmp_unit = _unit_state(FMP_BACKFILL_UNIT, user=True)
    linger_code, linger_value, linger_error = _run(
        [
            "loginctl",
            "show-user",
            os.environ.get("USER", "zooh"),
            "-p",
            "Linger",
            "--value",
        ]
    )
    if linger_code != 0:
        raise RuntimeAuditError(f"loginctl linger query failed: {linger_error}")

    evidence = {
        "schema_version": "quant.ai_radar_runtime_readiness.v1",
        "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "audit_window": {
            "since": args.since or "current_boot",
            "until": args.until or "collection_time",
        },
        "expected_model": args.expected_model,
        "served_models": served_models,
        "kernel_events": summarize_kernel_events(journal),
        "docker": docker_unit,
        "vllm": {
            "container": args.container,
            "running": state.get("Running") is True,
            "oom_killed": state.get("OOMKilled") is True,
            "restart_count": int(inspected.get("RestartCount") or 0),
            "restart_policy": str(restart.get("Name") or ""),
            "started_at": state.get("StartedAt"),
        },
        "user_linger": linger_value.strip().lower() == "yes",
        "radar_units": radar_units,
        "fmp_backfill": {
            "unit": fmp_unit,
            "processes": _fmp_backfill_processes(),
        },
    }
    evidence.update(classify_runtime_evidence(evidence))
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--expected-model", default=DEFAULT_EXPECTED_MODEL)
    parser.add_argument("--model-endpoint", default=DEFAULT_MODEL_ENDPOINT)
    parser.add_argument("--timeout", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        audit = collect_runtime_evidence(args)
    except RuntimeAuditError as exc:
        audit = {
            "schema_version": "quant.ai_radar_runtime_readiness.v1",
            "status": "fail",
            "manual_reference_ready": False,
            "timer_activation_eligible": False,
            "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    output = args.run_dir.expanduser().resolve() / "runtime_readiness_audit.json"
    write_json(output, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0 if audit.get("manual_reference_ready") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
