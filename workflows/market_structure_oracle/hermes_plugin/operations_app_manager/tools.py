"""Operations-worker tools for the independent application manager."""

from __future__ import annotations

import json
from typing import Any

from hermes_constants import get_default_hermes_root

from .manager import AppManager, AppManagerError


OPERATIONS_APP_RUN_SCHEMA: dict[str, Any] = {
    "name": "operations_app_run",
    "description": (
        "Run one registered independent application through App Manager. "
        "This tool is accepted only from a live Operations Role Shell worker; "
        "App Manager never creates or selects a worker."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "app_id": {"type": "string"},
            "trigger": {
                "type": "string",
                "enum": ["manual", "cron", "worker", "direct"],
            },
            "source_job_id": {"type": "string"},
            "request_id": {
                "type": "string",
                "pattern": "^[a-f0-9]{32}$",
            },
            "managed": {"type": "boolean"},
            "preflight_only": {"type": "boolean"},
            "input": {
                "type": "object",
                "description": (
                    "Application-owned JSON request. App Manager seals it in "
                    "an immutable run input file and passes only the file path."
                ),
            },
        },
        "required": ["app_id", "trigger", "request_id"],
        "additionalProperties": False,
    },
}


def handle_operations_app_run(args: dict[str, Any], **_kwargs: Any) -> str:
    manager = AppManager(home=get_default_hermes_root())
    try:
        receipt = manager.run(
            str(args.get("app_id") or ""),
            trigger=str(args.get("trigger") or "worker"),
            source_job_id=str(args.get("source_job_id") or "").strip() or None,
            managed=bool(args.get("managed", True)),
            request_id=str(args.get("request_id") or ""),
            preflight_only=bool(args.get("preflight_only", False)),
            request_input=args.get("input"),
        )
    except AppManagerError as exc:
        return json.dumps(
            {"ok": False, "error": str(exc)},
            ensure_ascii=False,
            sort_keys=True,
        )

    receipt_path = (
        manager.runs_dir
        / str(receipt["app_id"])
        / f"{receipt['run_id']}.json"
    )
    return json.dumps(
        {
            "ok": receipt["status"] in {"PASS", "PREFLIGHT_PASS"},
            "app_id": receipt["app_id"],
            "request_id": receipt["run_id"],
            "status": receipt["status"],
            "exit_code": receipt["exit_code"],
            "managed_completion_claim_allowed": receipt[
                "managed_completion_claim_allowed"
            ],
            "operations_worker": receipt["operations_worker"],
            "operations_role_shell_id": receipt["operations_role_shell_id"],
            "operations_worker_executor_id": receipt[
                "operations_worker_executor_id"
            ],
            "operations_worker_task_id": receipt["operations_worker_task_id"],
            "receipt_path": str(receipt_path),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
