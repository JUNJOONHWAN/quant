"""Independent application registry and execution ledger.

Operations is deliberately a control plane:

* app manifests own logic, dependencies, agent profile, skills, and gates;
* cron and workers call the same app runner;
* Multitool provisions capabilities out of band and is never a runtime hop;
* the Hermes supervisor adapter registry is not consulted for app execution.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home


SCHEMA = "hermes-independent-app-v1"
RECEIPT_SCHEMA = "hermes-app-run-receipt-v1"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,95}$")
REQUEST_INPUT_MAX_BYTES = 32 * 1024


class AppManagerError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9._-]+", "-", str(text or "").lower()).strip("-._")
    return value[:80] or "app"


class AppManager:
    def __init__(self, home: Path | None = None):
        self.home = (home or get_hermes_home()).expanduser().resolve()
        self.root = self.home / "operations"
        self.apps_dir = self.root / "apps"
        self.runs_dir = self.root / "runs"
        self.locks_dir = self.root / "locks"
        self.backups_dir = self.root / "backups"
        self.scripts_dir = self.home / "scripts"
        self.capabilities_file = self.root / "capabilities.json"
        for directory in (
            self.apps_dir,
            self.runs_dir,
            self.locks_dir,
            self.backups_dir,
            self.scripts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _validate_id(self, app_id: str) -> str:
        app_id = str(app_id or "").strip()
        if not SAFE_ID.fullmatch(app_id):
            raise AppManagerError(f"invalid app_id: {app_id!r}")
        return app_id

    def _manifest_path(self, app_id: str) -> Path:
        return self.apps_dir / self._validate_id(app_id) / "app.json"

    def _load_manifest(self, app_id: str) -> dict[str, Any]:
        path = self._manifest_path(app_id)
        if not path.is_file():
            raise AppManagerError(f"application is not registered: {app_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise AppManagerError(f"invalid manifest {path}: {exc}") from exc
        self._validate_manifest(data)
        data["_manifest_path"] = str(path)
        return data

    def _validate_manifest(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise AppManagerError("manifest must be a JSON object")
        if data.get("schema") != SCHEMA:
            raise AppManagerError(f"manifest schema must be {SCHEMA}")
        self._validate_id(str(data.get("app_id") or ""))
        runtime = data.get("runtime")
        if not isinstance(runtime, dict):
            raise AppManagerError("manifest.runtime must be an object")
        if runtime.get("kind") not in {"script", "command", "agent"}:
            raise AppManagerError("runtime.kind must be script, command, or agent")
        if runtime.get("kind") in {"script", "command"} and not runtime.get("entrypoint"):
            raise AppManagerError("script/command runtime requires entrypoint")
        if runtime.get("kind") == "agent":
            if not runtime.get("profile"):
                raise AppManagerError("agent runtime requires an independent profile")
            if not runtime.get("prompt_file") and not runtime.get("prompt"):
                raise AppManagerError("agent runtime requires prompt_file or prompt")

    def register_manifest(self, source: Path, *, replace: bool = False) -> dict[str, Any]:
        source = source.expanduser().resolve()
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except Exception as exc:
            raise AppManagerError(f"cannot read manifest {source}: {exc}") from exc
        self._validate_manifest(data)
        app_id = str(data["app_id"])
        destination = self._manifest_path(app_id)
        if destination.exists() and not replace:
            raise AppManagerError(f"application already registered: {app_id}")
        data = dict(data)
        data["registered_at"] = data.get("registered_at") or _utc_now()
        data["updated_at"] = _utc_now()
        _atomic_json(destination, data)
        return {"ok": True, "app_id": app_id, "manifest": str(destination)}

    def list_apps(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.apps_dir.glob("*/app.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                app_id = str(manifest.get("app_id") or path.parent.name)
                latest = self._latest_receipt(app_id)
                result.append(
                    {
                        "app_id": app_id,
                        "name": manifest.get("name") or app_id,
                        "category": manifest.get("category") or "other",
                        "managed": bool(manifest.get("managed", True)),
                        "runtime_kind": (manifest.get("runtime") or {}).get("kind"),
                        "cron_job_id": (manifest.get("schedule") or {}).get("cron_job_id"),
                        "cron_enabled": (manifest.get("schedule") or {}).get("enabled"),
                        "latest_status": (latest or {}).get("status"),
                        "latest_run_at": (latest or {}).get("finished_at"),
                    }
                )
            except Exception as exc:
                result.append({"app_id": path.parent.name, "error": str(exc)})
        return result

    def show(self, app_id: str) -> dict[str, Any]:
        manifest = self._load_manifest(app_id)
        manifest.pop("_manifest_path", None)
        return {
            "ok": True,
            "manifest": manifest,
            "latest_receipt": self._latest_receipt(app_id),
        }

    def _latest_receipt(self, app_id: str) -> dict[str, Any] | None:
        latest = self.runs_dir / self._validate_id(app_id) / "latest.json"
        if not latest.is_file():
            return None
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return None

    def capabilities(self, *, refresh: bool = False) -> dict[str, Any]:
        if refresh or not self.capabilities_file.is_file():
            news_fused_runtime = (
                self.root / "agents" / "etf-radar-news-fused-research"
            )
            barchart_service = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "is-active",
                    "qagent-logged-in-browser.service",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            barchart_collector = Path(
                "/home/zooh/Documents/GitHub/STOCK/scripts/"
                "collect_thermometer_chrome_evidence.py"
            )
            advanced_browser_skill = (
                news_fused_runtime / "skills" / "advanced-browser-ops" / "SKILL.md"
            )
            barchart_available = (
                barchart_service.returncode == 0
                and barchart_service.stdout.strip() == "active"
                and barchart_collector.is_file()
                and advanced_browser_skill.is_file()
            )
            payload = {
                "schema": "hermes-app-capabilities-v1",
                "refreshed_at": _utc_now(),
                "provisioner": "multitool-out-of-band",
                "runtime_calls_multitool": False,
                "capabilities": {
                    "filesystem": {"available": True, "source": "python-runtime"},
                    "terminal": {
                        "available": bool(shutil.which("bash") and shutil.which("python3")),
                        "source": "host-runtime",
                    },
                    "web": {
                        "available": (news_fused_runtime / "config.toml").is_file(),
                        "source": "news-fused-internal-research-runtime",
                    },
                    "browser": {
                        "available": (news_fused_runtime / "config.toml").is_file(),
                        "source": "news-fused-internal-research-runtime",
                    },
                    "advanced-internet-search": {
                        "available": (
                            news_fused_runtime / "skills" / "advanced-internet-search" / "SKILL.md"
                        ).is_file(),
                        "source": "news-fused-internal-research-skill",
                    },
                    "advanced-browser-ops": {
                        "available": advanced_browser_skill.is_file(),
                        "source": "news-fused-internal-browser-skill",
                    },
                    "barchart-premier": {
                        "available": barchart_available,
                        "source": (
                            "logged-in qagent browser service + CDP collector + "
                            "advanced-browser-ops"
                        ),
                        "service_status": barchart_service.stdout.strip(),
                        "collector": str(barchart_collector),
                    },
                    "etf-radar-news-fused": {
                        "available": (
                            news_fused_runtime / "skills" / "etf-radar-news-fused" / "SKILL.md"
                        ).is_file(),
                        "source": "news-fused-internal-research-skill",
                    },
                },
            }
            _atomic_json(self.capabilities_file, payload)
        try:
            return json.loads(self.capabilities_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise AppManagerError(f"cannot read capabilities: {exc}") from exc

    def _verify_one(self, manifest: dict[str, Any]) -> dict[str, Any]:
        app_id = str(manifest["app_id"])
        errors: list[str] = []
        warnings: list[str] = []
        runtime = manifest["runtime"]
        execution = manifest.get("execution")
        if not isinstance(execution, dict):
            errors.append("execution_contract_missing")
            execution = {}
        bypass = bool(execution.get("bypass_operations_worker", False))
        if bypass:
            if not str(execution.get("bypass_reason") or "").strip():
                errors.append("operations_worker_bypass_reason_missing")
        elif execution.get("default_worker") != "hermes-worker-general":
            errors.append("default_operations_worker_missing")
        kind = runtime["kind"]
        if kind in {"script", "command"}:
            entry = Path(str(runtime["entrypoint"])).expanduser()
            if not entry.is_absolute():
                entry = (Path(str(runtime.get("workdir") or self.home)) / entry).resolve()
            if not entry.is_file():
                errors.append(f"entrypoint_missing:{entry}")
        else:
            profile = self.home / "profiles" / str(runtime["profile"])
            if not (profile / "config.yaml").is_file():
                errors.append(f"profile_missing:{runtime['profile']}")
            prompt_file = runtime.get("prompt_file")
            if prompt_file and not Path(str(prompt_file)).expanduser().is_file():
                errors.append(f"prompt_file_missing:{prompt_file}")

        caps = self.capabilities(refresh=False).get("capabilities") or {}
        required = list((manifest.get("capabilities") or {}).get("required") or [])
        for capability in required:
            if not bool((caps.get(str(capability)) or {}).get("available")):
                errors.append(f"capability_missing:{capability}")

        schedule = manifest.get("schedule") or {}
        cron_id = schedule.get("cron_job_id")
        if cron_id:
            try:
                jobs_payload = json.loads(
                    (self.home / "cron" / "jobs.json").read_text(encoding="utf-8")
                )
                jobs = (
                    list(jobs_payload.get("jobs") or [])
                    if isinstance(jobs_payload, dict)
                    else list(jobs_payload)
                )
                job = next(
                    (
                        item
                        for item in jobs
                        if str(item.get("id") or "") == str(cron_id)
                    ),
                    None,
                )
                if not job:
                    errors.append(f"cron_job_missing:{cron_id}")
                else:
                    wrapper = str(job.get("script") or "")
                    expected = str(schedule.get("wrapper") or "")
                    if expected and Path(wrapper).name != Path(expected).name:
                        warnings.append(f"cron_not_attached:{wrapper}")
            except Exception as exc:
                warnings.append(f"cron_check_failed:{exc}")
        return {
            "app_id": app_id,
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    def reconcile_manifests(self) -> dict[str, Any]:
        updated: list[str] = []
        for item in self.list_apps():
            app_id = str(item.get("app_id") or "")
            if not app_id or item.get("error"):
                continue
            manifest = self._load_manifest(app_id)
            execution = dict(manifest.get("execution") or {})
            if not bool(execution.get("bypass_operations_worker", False)):
                execution["default_worker"] = "hermes-worker-general"
                execution["bypass_operations_worker"] = False
                execution.pop("bypass_reason", None)
            elif not str(execution.get("bypass_reason") or "").strip():
                raise AppManagerError(
                    f"{app_id}: explicit worker bypass has no bypass_reason"
                )
            execution.setdefault("worker_skills", [])
            manifest.pop("_manifest_path", None)
            manifest["execution"] = execution
            manifest["updated_at"] = _utc_now()
            _atomic_json(self._manifest_path(app_id), manifest)
            updated.append(app_id)
        return {
            "ok": True,
            "updated_count": len(updated),
            "default_worker": "hermes-worker-general",
            "apps": updated,
        }

    def verify(self, app_id: str | None = None) -> dict[str, Any]:
        manifests = [self._load_manifest(app_id)] if app_id else [
            self._load_manifest(item["app_id"])
            for item in self.list_apps()
            if item.get("app_id") and not item.get("error")
        ]
        results = [self._verify_one(item) for item in manifests]
        return {
            "ok": all(item["ok"] for item in results),
            "registered_count": len(results),
            "valid_count": sum(1 for item in results if item["ok"]),
            "invalid_count": sum(1 for item in results if not item["ok"]),
            "results": results,
        }

    def status(self) -> dict[str, Any]:
        apps = self.list_apps()
        verification = self.verify()
        cron = {"available": False}
        try:
            from cron.jobs import load_jobs

            jobs = load_jobs()
            cron = {
                "available": True,
                "total_jobs": len(jobs),
                "managed_jobs": sum(
                    1
                    for job in jobs
                    if str(job.get("script") or "").startswith("operations_app_")
                ),
            }
        except Exception as exc:
            cron = {"available": False, "error": str(exc)}
        return {
            "ok": bool(verification["ok"]),
            "operations_role": "program-manager",
            "runtime_multitool_dependency": False,
            "supervisor_adapter_dependency": False,
            "registered_apps": len(apps),
            "verification": verification,
            "cron": cron,
        }

    def _resolve_script(self, script: str) -> Path:
        path = Path(str(script)).expanduser()
        if not path.is_absolute():
            path = self.scripts_dir / path
        return path.resolve()

    def _wrapper_path(self, job_id: str) -> Path:
        return self.scripts_dir / f"operations_app_{job_id}.py"

    def _write_wrapper(self, app_id: str, job_id: str) -> Path:
        wrapper = self._wrapper_path(job_id)
        code = f'''#!/usr/bin/env python3
"""Generated Operations App Manager cron wrapper."""
import shutil
import subprocess
import sys

hermes = shutil.which("hermes")
if not hermes:
    print("operations-app-manager: hermes executable not found", file=sys.stderr)
    raise SystemExit(127)
command = [
    hermes,
    "apps",
    "run",
    {app_id!r},
    "--trigger",
    "cron",
    "--source-job-id",
    {job_id!r},
    "--passthrough",
]
raise SystemExit(subprocess.run(command, check=False).returncode)
'''
        wrapper.write_text(code, encoding="utf-8")
        os.chmod(wrapper, 0o700)
        return wrapper

    def import_cron(
        self,
        *,
        attach: bool = False,
        include_agent_jobs: bool = False,
    ) -> dict[str, Any]:
        try:
            from cron.jobs import JOBS_FILE, load_jobs, update_job
        except Exception as exc:
            raise AppManagerError(f"cron subsystem unavailable: {exc}") from exc

        jobs = load_jobs()
        backup = None
        if attach and Path(JOBS_FILE).is_file():
            backup = self.backups_dir / (
                f"jobs.before-app-import.{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
            )
            shutil.copy2(JOBS_FILE, backup)
            os.chmod(backup, 0o600)

        imported: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for job in jobs:
            job_id = str(job.get("id") or "")
            script = str(job.get("script") or "").strip()
            if not script:
                skipped.append({"job_id": job_id, "reason": "no_script"})
                continue
            if not bool(job.get("no_agent")) and not include_agent_jobs:
                skipped.append({"job_id": job_id, "reason": "agent_job_excluded"})
                continue

            existing_wrapper = Path(script).name.startswith("operations_app_")
            app_id = _slug(str(job.get("name") or f"cron-{job_id}"))
            existing_manifest = self._manifest_path(app_id)
            if existing_wrapper and existing_manifest.is_file():
                skipped.append({"job_id": job_id, "reason": "already_attached"})
                continue

            original_script = self._resolve_script(script)
            category = "report" if any(
                token in app_id
                for token in ("etf", "radar", "thermometer", "capital-tides", "gostop", "flow-dashboard")
            ) else "operations"
            manifest = {
                "schema": SCHEMA,
                "app_id": app_id,
                "name": str(job.get("name") or app_id),
                "description": "Imported from an existing Hermes cron script.",
                "category": category,
                "managed": True,
                "ownership": {
                    "control_plane": "operations",
                    "logic_owner": "application",
                    "multitool_runtime_dependency": False,
                    "supervisor_adapter_dependency": False,
                },
                "runtime": {
                    "kind": "script",
                    "entrypoint": str(original_script),
                    "workdir": job.get("workdir"),
                    "timeout_seconds": 7200,
                    "agent_selection": "not-applicable",
                },
                "execution": {
                    "default_worker": "hermes-worker-general",
                    "bypass_operations_worker": False,
                    "worker_skills": [],
                },
                "capabilities": {
                    "required": ["filesystem", "terminal"],
                    "declared_legacy_toolsets": list(job.get("enabled_toolsets") or []),
                },
                "final_gates": [{"type": "exit_code", "equals": 0}],
                "schedule": {
                    "owner": "operations",
                    "executor": "hermes-cron",
                    "cron_job_id": job_id,
                    "expression": (job.get("schedule") or {}).get("expr"),
                    "timezone": "Asia/Seoul",
                    "enabled": bool(job.get("enabled")),
                    "wrapper": str(self._wrapper_path(job_id)),
                },
                "source": {
                    "type": "hermes-cron-import",
                    "cron_job_id": job_id,
                    "original_script": script,
                    "original_script_sha256": _sha256(original_script),
                    "imported_at": _utc_now(),
                },
            }
            destination = self._manifest_path(app_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            manifest["registered_at"] = _utc_now()
            manifest["updated_at"] = _utc_now()
            _atomic_json(destination, manifest)

            wrapper = self._write_wrapper(app_id, job_id)
            if attach:
                updated = update_job(job_id, {"script": wrapper.name})
                if not updated:
                    raise AppManagerError(f"failed to attach cron job {job_id}")
            imported.append(
                {
                    "app_id": app_id,
                    "job_id": job_id,
                    "attached": bool(attach),
                    "original_script": str(original_script),
                    "wrapper": str(wrapper),
                }
            )
        return {
            "ok": True,
            "job_count": len(jobs),
            "imported_count": len(imported),
            "skipped_count": len(skipped),
            "attached": bool(attach),
            "backup": str(backup) if backup else None,
            "imported": imported,
            "skipped": skipped,
        }

    def ensure_schedule(self, app_id: str) -> dict[str, Any]:
        manifest = self._load_manifest(app_id)
        schedule = dict(manifest.get("schedule") or {})
        expression = str(schedule.get("expression") or "").strip()
        if not expression:
            raise AppManagerError(f"application has no schedule expression: {app_id}")
        if str(schedule.get("timezone") or "") != "Asia/Seoul":
            raise AppManagerError("managed app schedules must declare timezone=Asia/Seoul")
        try:
            from cron.jobs import create_job, get_job, pause_job, resume_job, update_job
        except Exception as exc:
            raise AppManagerError(f"cron subsystem unavailable: {exc}") from exc

        cron_job_id = str(schedule.get("cron_job_id") or "")
        job = get_job(cron_job_id) if cron_job_id else None
        created = False
        if not job:
            bootstrap_id = f"bootstrap-{_slug(app_id)}"
            bootstrap_wrapper = self._write_wrapper(app_id, bootstrap_id)
            job = create_job(
                prompt=f"Operations-managed independent app: {app_id}",
                schedule=expression,
                name=str(manifest.get("name") or app_id),
                script=bootstrap_wrapper.name,
                no_agent=True,
                workdir=manifest["runtime"].get("workdir"),
                deliver="local",
            )
            cron_job_id = str(job["id"])
            created = True

        wrapper = self._write_wrapper(app_id, cron_job_id)
        job = update_job(
            cron_job_id,
            {
                "name": str(manifest.get("name") or app_id),
                "script": wrapper.name,
                "schedule": expression,
                "workdir": manifest["runtime"].get("workdir"),
                "no_agent": True,
            },
        )
        if not job:
            raise AppManagerError(f"failed to reconcile cron job: {cron_job_id}")
        desired_enabled = bool(schedule.get("enabled", True))
        if desired_enabled and not job.get("enabled"):
            job = resume_job(cron_job_id)
        elif not desired_enabled and job.get("enabled"):
            job = pause_job(cron_job_id, reason="disabled by application manifest")

        schedule.update(
            {
                "owner": "operations",
                "executor": "hermes-cron",
                "cron_job_id": cron_job_id,
                "wrapper": str(wrapper),
                "enabled": desired_enabled,
            }
        )
        manifest.pop("_manifest_path", None)
        manifest["schedule"] = schedule
        manifest["updated_at"] = _utc_now()
        _atomic_json(self._manifest_path(app_id), manifest)
        return {
            "ok": True,
            "app_id": app_id,
            "created": created,
            "cron_job_id": cron_job_id,
            "expression": expression,
            "timezone": "Asia/Seoul",
            "enabled": bool((job or {}).get("enabled")),
            "wrapper": str(wrapper),
        }

    def set_schedule_enabled(
        self,
        app_id: str,
        *,
        enabled: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        manifest = self._load_manifest(app_id)
        schedule = dict(manifest.get("schedule") or {})
        cron_job_id = str(schedule.get("cron_job_id") or "")
        if not cron_job_id:
            raise AppManagerError(f"application has no linked cron job: {app_id}")
        try:
            from cron.jobs import pause_job, resume_job
        except Exception as exc:
            raise AppManagerError(f"cron subsystem unavailable: {exc}") from exc
        job = resume_job(cron_job_id) if enabled else pause_job(cron_job_id, reason=reason)
        if not job:
            raise AppManagerError(f"linked cron job not found: {cron_job_id}")
        schedule["enabled"] = bool(enabled)
        manifest.pop("_manifest_path", None)
        manifest["schedule"] = schedule
        manifest["updated_at"] = _utc_now()
        _atomic_json(self._manifest_path(app_id), manifest)
        return {
            "ok": True,
            "app_id": app_id,
            "cron_job_id": cron_job_id,
            "enabled": bool(job.get("enabled")),
            "state": job.get("state"),
        }

    @contextlib.contextmanager
    def _run_lock(self, app_id: str) -> Iterator[None]:
        path = self.locks_dir / f"{self._validate_id(app_id)}.lock"
        with path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AppManagerError(f"application already running: {app_id}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _build_command(self, manifest: dict[str, Any]) -> tuple[list[str], Path | None]:
        runtime = manifest["runtime"]
        kind = runtime["kind"]
        workdir = Path(str(runtime.get("workdir"))).expanduser() if runtime.get("workdir") else None
        if kind == "script":
            entry = Path(str(runtime["entrypoint"])).expanduser()
            if entry.suffix.lower() in {".sh", ".bash"}:
                return ["bash", str(entry)], workdir
            return [sys.executable, str(entry)], workdir
        if kind == "command":
            command = runtime.get("command")
            if isinstance(command, list) and command:
                return [str(item) for item in command], workdir
            return [str(runtime["entrypoint"])], workdir

        hermes = shutil.which("hermes")
        if not hermes:
            raise AppManagerError("hermes executable not found")
        prompt = str(runtime.get("prompt") or "")
        if runtime.get("prompt_file"):
            prompt = Path(str(runtime["prompt_file"])).expanduser().read_text(encoding="utf-8")
        command = [hermes, "-p", str(runtime["profile"]), "-z", prompt]
        if runtime.get("model"):
            command.extend(["-m", str(runtime["model"])])
        if runtime.get("provider"):
            command.extend(["--provider", str(runtime["provider"])])
        toolsets = list(runtime.get("toolsets") or [])
        if toolsets:
            command.extend(["-t", ",".join(str(item) for item in toolsets)])
        skills = list(runtime.get("skills") or [])
        if skills:
            command.extend(["--skills", ",".join(str(item) for item in skills)])
        return command, workdir

    def _check_gates(
        self,
        manifest: dict[str, Any],
        *,
        exit_code: int,
        stdout: str,
    ) -> tuple[bool, list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        gates = list(manifest.get("final_gates") or [{"type": "exit_code", "equals": 0}])
        for gate in gates:
            gate_type = str(gate.get("type") or "")
            passed = False
            detail = ""
            if gate_type == "exit_code":
                expected = int(gate.get("equals", 0))
                passed = exit_code == expected
                detail = f"actual={exit_code},expected={expected}"
            elif gate_type == "stdout_nonempty":
                passed = bool(stdout.strip())
                detail = f"chars={len(stdout)}"
            elif gate_type == "path_exists":
                path = Path(str(gate.get("path") or "")).expanduser()
                passed = path.exists()
                detail = str(path)
            elif gate_type == "json_field":
                path = Path(str(gate.get("path") or "")).expanduser()
                field = str(gate.get("field") or "")
                expected = gate.get("equals")
                try:
                    value: Any = json.loads(path.read_text(encoding="utf-8"))
                    for part in field.split("."):
                        value = value[part]
                    passed = value == expected
                    detail = f"actual={value!r},expected={expected!r}"
                except Exception as exc:
                    detail = f"error={exc}"
            else:
                detail = "unknown_gate"
            results.append({"type": gate_type, "passed": passed, "detail": detail})
        return all(item["passed"] for item in results), results

    def _persist_receipt(self, app_id: str, receipt: dict[str, Any]) -> None:
        run_dir = self.runs_dir / self._validate_id(app_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / f"{receipt['run_id']}.json", receipt)
        _atomic_json(run_dir / "latest.json", receipt)
        ledger = self.root / "runs.jsonl"
        lock = self.root / ".runs.lock"
        with lock.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            try:
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
        os.chmod(ledger, 0o600)

    def _seal_request_input(
        self,
        app_id: str,
        run_id: str,
        request_input: dict[str, Any] | None,
    ) -> tuple[Path | None, str | None, int]:
        if request_input is None:
            return None, None, 0
        if not isinstance(request_input, dict):
            raise AppManagerError("application input must be one JSON object")
        try:
            encoded = json.dumps(
                request_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise AppManagerError(
                f"application input is not finite JSON: {exc}"
            ) from exc
        if len(encoded) > REQUEST_INPUT_MAX_BYTES:
            raise AppManagerError(
                "application input exceeds "
                f"{REQUEST_INPUT_MAX_BYTES} UTF-8 bytes"
            )
        input_path = (
            self.runs_dir
            / self._validate_id(app_id)
            / "inputs"
            / f"{run_id}.json"
        )
        _atomic_json(input_path, request_input)
        return input_path, hashlib.sha256(encoded).hexdigest(), len(encoded)

    def run(
        self,
        app_id: str,
        *,
        trigger: str = "manual",
        source_job_id: str | None = None,
        managed: bool = True,
        dry_run: bool = False,
        request_id: str | None = None,
        preflight_only: bool = False,
        request_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = self._load_manifest(app_id)
        execution = dict(manifest.get("execution") or {})
        bypass = bool(execution.get("bypass_operations_worker", False))
        if bypass:
            reason = str(execution.get("bypass_reason") or "").strip()
            if not reason:
                raise AppManagerError(
                    "bypass_operations_worker requires an explicit bypass_reason"
                )
            return self.execute_direct(
                app_id,
                trigger=trigger,
                source_job_id=source_job_id,
                managed=managed,
                dry_run=dry_run,
                request_id=request_id,
                preflight_only=preflight_only,
                request_input=request_input,
            )
        context = self._bound_operations_context()
        return self.execute_direct(
            app_id,
            trigger=trigger,
            source_job_id=source_job_id,
            managed=managed,
            dry_run=dry_run,
            request_id=request_id,
            preflight_only=preflight_only,
            request_input=request_input,
            operations_context=context,
        )

    def _bound_operations_context(self) -> Any:
        """Return verified Operations Role Shell provenance for this worker."""
        try:
            from hermes_cli.external_cli_adapter import (
                ExternalCLIAdapterError,
                _load_context,
            )
        except ImportError as exc:
            raise AppManagerError(
                "managed app execution requires a live Operations Role Shell worker"
            ) from exc
        try:
            context = _load_context(dict(os.environ))
        except ExternalCLIAdapterError as exc:
            raise AppManagerError(
                "managed app execution requires a live Operations Role Shell worker"
            ) from exc
        if context.shell_key != "operations":
            raise AppManagerError(
                "managed app execution is allowed only from the Operations Role Shell"
            )
        return context

    def execute_direct(
        self,
        app_id: str,
        *,
        trigger: str = "worker",
        source_job_id: str | None = None,
        managed: bool = True,
        dry_run: bool = False,
        request_id: str | None = None,
        preflight_only: bool = False,
        request_input: dict[str, Any] | None = None,
        operations_context: Any | None = None,
    ) -> dict[str, Any]:
        manifest = self._load_manifest(app_id)
        execution = dict(manifest.get("execution") or {})
        bypass = bool(execution.get("bypass_operations_worker", False))
        default_worker = (
            str(execution.get("default_worker") or "").strip() or None
        )
        bypass_reason = str(execution.get("bypass_reason") or "").strip()
        if operations_context is None and not (bypass and bypass_reason):
            raise AppManagerError(
                "App Manager execution requires Operations Role Shell provenance"
            )
        verification = self._verify_one(manifest)
        if not verification["ok"]:
            raise AppManagerError(
                f"application verification failed: {', '.join(verification['errors'])}"
            )
        if source_job_id and not re.fullmatch(
            r"[A-Za-z0-9._-]{1,128}", source_job_id
        ):
            raise AppManagerError(f"invalid source_job_id: {source_job_id!r}")
        command, workdir = self._build_command(manifest)
        run_id = str(request_id or uuid.uuid4().hex)
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise AppManagerError(f"invalid request_id: {run_id}")
        input_path, input_sha256, input_bytes = self._seal_request_input(
            app_id,
            run_id,
            request_input,
        )
        started = _utc_now()
        env = os.environ.copy()
        env.update(
            {
                "HERMES_HOME": str(self.home),
                "OPERATIONS_APP_ID": app_id,
                "OPERATIONS_APP_RUN_ID": run_id,
                "OPERATIONS_APP_TRIGGER": trigger,
                "OPERATIONS_APP_MANAGED": "1" if managed else "0",
            }
        )
        if input_path is not None:
            env["OPERATIONS_APP_INPUT_FILE"] = str(input_path)
            env["OPERATIONS_APP_INPUT_SHA256"] = str(input_sha256)
        if preflight_only:
            preflight = manifest.get("preflight")
            if not isinstance(preflight, dict):
                raise AppManagerError(f"application has no preflight contract: {app_id}")
            for key, value in dict(preflight.get("env") or {}).items():
                key_text = str(key)
                if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", key_text):
                    raise AppManagerError(f"invalid preflight env key: {key_text!r}")
                env[key_text] = str(value)
        if source_job_id:
            env["OPERATIONS_APP_SOURCE_JOB_ID"] = source_job_id

        with self._run_lock(app_id):
            begin = time.monotonic()
            if dry_run:
                exit_code, stdout, stderr = 0, "", ""
            else:
                timeout = max(1, int((manifest["runtime"]).get("timeout_seconds", 7200)))
                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(workdir) if workdir else None,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                    )
                    exit_code = int(completed.returncode)
                    stdout = completed.stdout or ""
                    stderr = completed.stderr or ""
                except subprocess.TimeoutExpired as exc:
                    exit_code = 124
                    stdout = str(exc.stdout or "")
                    stderr = f"timeout after {exc.timeout}s"
            gate_ok, gates = self._check_gates(
                manifest,
                exit_code=exit_code,
                stdout=stdout,
            )
            status = (
                "DRY_RUN"
                if dry_run
                else (
                    "PREFLIGHT_PASS"
                    if preflight_only and gate_ok
                    else ("PASS" if gate_ok else "FAIL")
                )
            )
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "run_id": run_id,
                "app_id": app_id,
                "trigger": trigger,
                "source_job_id": source_job_id,
                "managed": bool(managed),
                "managed_completion_claim_allowed": bool(
                    managed and gate_ok and not dry_run and not preflight_only
                ),
                "runtime_kind": manifest["runtime"]["kind"],
                "agent_profile": manifest["runtime"].get("profile"),
                "supervisor_adapter_used": False,
                "multitool_called_at_runtime": False,
                "operations_worker_required": not bool(
                    (manifest.get("execution") or {}).get(
                        "bypass_operations_worker", False
                    )
                ),
                "operations_worker": default_worker if not bypass else None,
                "operations_worker_context": operations_context is not None,
                "operations_worker_dispatched": operations_context is not None,
                "operations_worker_dispatch_owner": (
                    "hermes-role-shell" if operations_context else None
                ),
                "operations_worker_routed_by_hermes": (
                    operations_context is not None
                ),
                "app_manager_created_kanban_card": False,
                "app_manager_created_worker": False,
                "operations_role_shell_id": (
                    operations_context.shell_id if operations_context else None
                ),
                "operations_worker_executor_id": (
                    operations_context.executor_id if operations_context else None
                ),
                "operations_worker_task_id": (
                    operations_context.task_id if operations_context else None
                ),
                "operations_worker_run_id": (
                    operations_context.run_id if operations_context else None
                ),
                "supervisor_controller_selected_agent": False,
                "preflight_only": bool(preflight_only),
                "request_input_present": input_path is not None,
                "request_input_file": str(input_path) if input_path else None,
                "request_input_sha256": input_sha256,
                "request_input_bytes": input_bytes,
                "started_at": started,
                "finished_at": _utc_now(),
                "duration_seconds": round(time.monotonic() - begin, 3),
                "exit_code": exit_code,
                "status": status,
                "gates": gates,
                "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
                "stdout": stdout,
                "stderr": stderr,
                "command_executable": Path(command[0]).name,
            }
            self._persist_receipt(app_id, receipt)
            return receipt
