"""CLI for the Operations App Manager."""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .manager import AppManager, AppManagerError


def _load_completed_receipt(
    manager: AppManager,
    *,
    app_id: str,
    request_id: str,
    task_id: str,
    expected_status: str,
) -> dict[str, Any] | None:
    receipt_path = manager.runs_dir / app_id / f"{request_id}.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if str(receipt.get("operations_worker_task_id") or "") != task_id:
        raise AppManagerError(
            f"App Manager receipt task provenance mismatch: {task_id}"
        )
    if (
        str(receipt.get("status") or "") != expected_status
        or int(receipt.get("exit_code") if receipt.get("exit_code") is not None else 1)
        != 0
    ):
        return None
    return receipt


def register_cli(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="apps_action")

    list_p = sub.add_parser("list", aliases=["ls"], help="List registered applications")
    list_p.add_argument("--json", action="store_true")

    show_p = sub.add_parser("show", help="Show one application manifest and latest receipt")
    show_p.add_argument("app_id")
    show_p.add_argument("--json", action="store_true")

    status_p = sub.add_parser("status", help="Show registry and runtime health")
    status_p.add_argument("--json", action="store_true")

    verify_p = sub.add_parser("verify", help="Validate manifests, entrypoints, capabilities, and cron links")
    verify_p.add_argument("app_id", nargs="?")
    verify_p.add_argument("--json", action="store_true")

    reconcile_p = sub.add_parser(
        "reconcile",
        help="Normalize every manifest to the current Operations-worker contract",
    )
    reconcile_p.add_argument("--json", action="store_true")

    register_p = sub.add_parser("register", help="Register an application manifest")
    register_p.add_argument("manifest")
    register_p.add_argument("--replace", action="store_true")
    register_p.add_argument("--json", action="store_true")

    import_p = sub.add_parser("import-cron", help="Register existing script cron jobs as independent apps")
    import_p.add_argument("--attach", action="store_true", help="Replace cron script with a managed wrapper")
    import_p.add_argument("--include-agent-jobs", action="store_true")
    import_p.add_argument("--json", action="store_true")

    schedule_p = sub.add_parser("schedule", help="Create or reconcile an app-owned cron schedule")
    schedule_p.add_argument("app_id")
    schedule_p.add_argument("--json", action="store_true")

    pause_p = sub.add_parser("pause", help="Pause an app's linked cron schedule")
    pause_p.add_argument("app_id")
    pause_p.add_argument("--reason", default="paused by Operations App Manager")
    pause_p.add_argument("--json", action="store_true")

    resume_p = sub.add_parser("resume", help="Resume an app's linked cron schedule")
    resume_p.add_argument("app_id")
    resume_p.add_argument("--json", action="store_true")

    run_p = sub.add_parser("run", help="Run a registered application")
    run_p.add_argument("app_id")
    run_p.add_argument(
        "--trigger",
        choices=["manual", "cron", "worker", "direct"],
        default="manual",
    )
    run_p.add_argument("--source-job-id", default="")
    run_p.add_argument("--unmanaged", action="store_true")
    run_p.add_argument("--passthrough", action="store_true")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--preflight-only", action="store_true")
    run_p.add_argument("--json", action="store_true")

    execute_p = sub.add_parser(
        "execute",
        help="Internal: execute an app after the Operations worker has accepted it",
    )
    execute_p.add_argument("app_id")
    execute_p.add_argument(
        "--trigger",
        choices=["manual", "cron", "worker", "direct"],
        default="worker",
    )
    execute_p.add_argument("--source-job-id", default="")
    execute_p.add_argument("--request-id", required=True)
    execute_p.add_argument("--passthrough", action="store_true")
    execute_p.add_argument("--preflight-only", action="store_true")
    execute_p.add_argument("--json", action="store_true")

    cap_p = sub.add_parser("capabilities", help="Inspect or refresh capability inventory")
    cap_p.add_argument("--refresh", action="store_true")
    cap_p.add_argument("--json", action="store_true")

    parser.set_defaults(func=apps_command)


def _emit(payload, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if isinstance(payload, str):
        print(payload)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _run_cli_json(command: list[str], *, timeout: int = 60) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise AppManagerError(
            f"command failed rc={completed.returncode}: {' '.join(command)}; "
            f"stderr={completed.stderr[-1000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AppManagerError(
            f"command returned invalid JSON: {' '.join(command)}"
        ) from exc
    if not isinstance(payload, dict):
        raise AppManagerError(
            f"command returned non-object JSON: {' '.join(command)}"
        )
    return payload


def _operations_request_workspace(manager: AppManager, app_id: str) -> Path:
    """Return an isolated workspace for the Operations worker request card."""
    workspace = (
        manager.root / "request-workspaces" / manager._validate_id(app_id)
    ).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _request_operations_app_run(
    manager: AppManager,
    app_id: str,
    *,
    trigger: str,
    source_job_id: str | None,
    managed: bool,
    dry_run: bool,
    preflight_only: bool,
) -> dict[str, Any]:
    """Ask Hermes to route an app request through the Operations Role Shell."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import supervisor_registry as registry

    manifest = manager._load_manifest(app_id)
    verification = manager._verify_one(manifest)
    if not verification["ok"]:
        raise AppManagerError(
            f"application verification failed: {', '.join(verification['errors'])}"
        )
    if preflight_only and not isinstance(manifest.get("preflight"), dict):
        raise AppManagerError(f"application has no preflight contract: {app_id}")
    with kb.connect_closing() as conn:
        operations_shell = registry.get_shell(conn, shell_key="operations")
    if operations_shell is None:
        raise AppManagerError("active Operations Role Shell is unavailable")

    request_id = uuid.uuid4().hex
    if dry_run:
        default_worker = str(
            (manifest.get("execution") or {}).get("default_worker") or ""
        ).strip()
        return {
            "schema": "hermes-app-run-receipt-v1",
            "run_id": request_id,
            "app_id": app_id,
            "trigger": trigger,
            "source_job_id": source_job_id,
            "managed": bool(managed),
            "managed_completion_claim_allowed": False,
            "status": "DRY_RUN",
            "exit_code": 0,
            "operations_worker_required": True,
            "operations_worker": default_worker,
            "operations_worker_context": False,
            "operations_worker_dispatched": False,
            "operations_worker_dispatch_owner": "hermes-role-shell",
            "operations_worker_routed_by_hermes": False,
            "app_manager_created_kanban_card": False,
            "app_manager_created_worker": False,
            "operations_role_shell_id": operations_shell.id,
            "supervisor_controller_selected_agent": False,
            "multitool_called_at_runtime": False,
            "preflight_only": bool(preflight_only),
            "stdout": "",
            "stderr": "",
        }

    hermes = shutil.which("hermes")
    if not hermes:
        raise AppManagerError("hermes executable not found")
    timeout_seconds = max(
        60,
        int((manifest.get("runtime") or {}).get("timeout_seconds", 7200)),
    )
    request_workspace = _operations_request_workspace(manager, app_id)
    tool_args = {
        "app_id": app_id,
        "trigger": trigger,
        "source_job_id": source_job_id or "",
        "request_id": request_id,
        "managed": bool(managed),
        "preflight_only": bool(preflight_only),
    }
    expected_status = "PREFLIGHT_PASS" if preflight_only else "PASS"
    body = (
        "Manage one registered independent application through App Manager. "
        "Call the `operations_app_run` tool exactly once with these arguments:\n\n"
        f"{json.dumps(tool_args, ensure_ascii=False, sort_keys=True)}\n\n"
        "Do not use Terminal to run the application and do not create or select "
        "another worker. The App Manager tool is the only application execution "
        "surface. Record the returned receipt as Timeline output, then complete "
        f"this card only when ok=true and status={expected_status}. Otherwise "
        "block with the exact failed gate."
    )
    create_command = [
        hermes,
        "kanban",
        "create",
        f"Operations app request: {app_id}",
        "--body",
        body,
        "--role-shell",
        operations_shell.id,
        "--workspace",
        f"dir:{request_workspace}",
        "--idempotency-key",
        f"operations-app-request:{app_id}:{request_id}",
        "--max-runtime",
        str(timeout_seconds + 300),
        "--max-retries",
        "1",
        "--created-by",
        "operations-app-requester",
        "--json",
    ]
    task = _run_cli_json(create_command)
    task_id = str(task.get("id") or "")
    if not task_id:
        raise AppManagerError("Kanban create returned no Operations task id")

    with contextlib.suppress(Exception):
        _run_cli_json([hermes, "kanban", "dispatch", "--max", "1", "--json"])

    deadline = time.monotonic() + timeout_seconds + 300
    last_status = ""
    while time.monotonic() < deadline:
        detail = _run_cli_json(
            [hermes, "kanban", "show", task_id, "--json"],
            timeout=60,
        )
        task_detail = dict(detail.get("task") or {})
        last_status = str(task_detail.get("status") or "")
        receipt = _load_completed_receipt(
            manager,
            app_id=app_id,
            request_id=request_id,
            task_id=task_id,
            expected_status=expected_status,
        )
        if receipt is not None:
            if last_status not in {"done", "completed"}:
                with contextlib.suppress(Exception):
                    _run_cli_json(
                        [
                            hermes,
                            "kanban",
                            "complete",
                            task_id,
                            "--result",
                            json.dumps(
                                {
                                    "status": expected_status,
                                    "app_id": app_id,
                                    "run_id": receipt.get("run_id"),
                                    "reconciled_from_app_receipt": True,
                                },
                                ensure_ascii=False,
                            ),
                            "--json",
                        ]
                    )
            return receipt
        if last_status in {"done", "completed"}:
            raise AppManagerError(
                f"Operations worker completed without successful App Manager receipt: {task_id}"
            )
        if last_status in {"blocked", "failed", "archived"}:
            runs = list(detail.get("runs") or [])
            latest_run = runs[-1] if runs else {}
            raise AppManagerError(
                f"Operations task {last_status}: {task_id}; "
                f"error={latest_run.get('error') or task_detail.get('result') or ''}"
            )
        time.sleep(2)
    raise AppManagerError(
        f"Operations task timed out: {task_id}; status={last_status}"
    )


def apps_command(args: argparse.Namespace) -> int:
    action = getattr(args, "apps_action", None)
    if not action:
        print(
            "Usage: hermes apps "
            "{list|show|status|verify|register|import-cron|run|capabilities}"
        )
        return 2

    manager = AppManager()
    try:
        if action in {"list", "ls"}:
            payload = manager.list_apps()
        elif action == "show":
            payload = manager.show(args.app_id)
        elif action == "status":
            payload = manager.status()
        elif action == "verify":
            payload = manager.verify(args.app_id or None)
        elif action == "reconcile":
            payload = manager.reconcile_manifests()
        elif action == "register":
            payload = manager.register_manifest(Path(args.manifest), replace=args.replace)
        elif action == "import-cron":
            payload = manager.import_cron(
                attach=args.attach,
                include_agent_jobs=args.include_agent_jobs,
            )
        elif action == "schedule":
            payload = manager.ensure_schedule(args.app_id)
        elif action == "pause":
            payload = manager.set_schedule_enabled(
                args.app_id,
                enabled=False,
                reason=args.reason,
            )
        elif action == "resume":
            payload = manager.set_schedule_enabled(args.app_id, enabled=True)
        elif action == "run":
            receipt = _request_operations_app_run(
                manager,
                args.app_id,
                trigger=args.trigger,
                source_job_id=args.source_job_id or None,
                managed=not args.unmanaged,
                dry_run=args.dry_run,
                preflight_only=args.preflight_only,
            )
            if args.passthrough:
                stdout = str(receipt.get("stdout") or "")
                stderr = str(receipt.get("stderr") or "")
                if stdout:
                    print(stdout, end="" if stdout.endswith("\n") else "\n")
                if stderr:
                    import sys

                    print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
                return int(receipt.get("exit_code") or 0)
            payload = receipt
        elif action == "execute":
            receipt = manager.run(
                args.app_id,
                trigger=args.trigger,
                source_job_id=args.source_job_id or None,
                request_id=args.request_id,
                preflight_only=args.preflight_only,
            )
            if args.passthrough:
                stdout = str(receipt.get("stdout") or "")
                stderr = str(receipt.get("stderr") or "")
                if stdout:
                    print(stdout, end="" if stdout.endswith("\n") else "\n")
                if stderr:
                    import sys

                    print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
                return int(receipt.get("exit_code") or 0)
            payload = receipt
        elif action == "capabilities":
            payload = manager.capabilities(refresh=args.refresh)
        else:
            print(f"Unknown apps action: {action}")
            return 2
        _emit(payload, bool(getattr(args, "json", False)))
        if isinstance(payload, dict) and payload.get("ok") is False:
            return 1
        return 0
    except AppManagerError as exc:
        if bool(getattr(args, "json", False)):
            _emit({"ok": False, "error": str(exc)}, True)
        else:
            print(f"operations-app-manager: {exc}")
        return 1
