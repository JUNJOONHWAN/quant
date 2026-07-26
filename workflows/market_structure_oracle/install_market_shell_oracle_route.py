#!/home/zooh/.hermes/hermes-agent/venv/bin/python
"""Install the managed Oracle route as a new immutable market shell version."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from hermes_cli import kanban_db as kb
from hermes_cli import supervisor_registry as registry


ROUTE_MARKER = "MANAGED_ORACLE_ROUTE_V1"
ROUTE_INSTRUCTIONS = """\
[MANAGED_ORACLE_ROUTE_V1 — highest priority inside the market shell]
When the exact user request explicitly contains `오라클` or `Oracle`, use the
registered `market-structure-oracle` application as the exclusive analysis
runtime. This is a frozen-source historical Oracle request, not an ordinary
live-market collection request. Do not run question_verification, external
market-data tools, open-world research, browser collection, or data enrichment
for this route.

Resolve the requested sector or theme to the application's canonical scope and
run only:

  hermes apps run market-structure-oracle --input-json <JSON_OBJECT> --json

The JSON object contains the exact user `query` plus either canonical `scope`
or an explicit `etfs` list. Whole-market Oracle requests use
`hermes apps run market-structure-oracle --json`. Never execute the Python
entrypoint directly. Require a managed App Manager receipt with status PASS,
Operations Role Shell provenance, and output paths. Read the Oracle JSON result
and answer with its global context, conditional scope structure, absolute and
QQQ-relative scenarios, validation confidence, limitations, and exact result
paths. The app's frozen full-market daily observations and ETF Flow D+2 contract
satisfy the data-source requirement for this explicit route. If the managed app
fails, block with the exact App Manager gate; do not silently fall back to a
different analysis.

[All other market requests]
"""


def _run_json(command: list[str]) -> Any:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}; "
            f"{completed.stderr[-1000:]}"
        )
    return json.loads(completed.stdout)


def _latest_market_shell(hermes: str) -> dict[str, Any]:
    rows = _run_json([hermes, "supervisor", "shell", "list", "--json"])
    market = [row for row in rows if row.get("shell_key") == "market"]
    if not market:
        raise RuntimeError("market role shell is not registered")
    return max(market, key=lambda row: int(row["version"]))


def _migrate_bindings(
    *,
    source_shell_id: str,
    target_shell_id: str,
) -> list[dict[str, Any]]:
    migrated: list[dict[str, Any]] = []
    with kb.connect_closing() as connection:
        source = registry.list_bindings(
            connection, shell_id=source_shell_id
        )
        target = {
            binding.executor_id: binding
            for binding in registry.list_bindings(
                connection, shell_id=target_shell_id
            )
        }
        for old in source:
            if old.executor_id in target:
                continue
            new = registry.bind_executor(
                connection,
                shell_id=target_shell_id,
                executor_id=old.executor_id,
                priority=old.priority,
                weight=old.weight,
                capability_cap=old.capability_cap,
                constraints=old.constraints,
                responsibility=old.responsibility,
                assignment_note=old.assignment_note,
                assigned_by=old.assigned_by,
            )
            if not old.enabled:
                registry.set_binding_enabled(connection, new.id, False)
            migrated.append(
                {
                    "binding_id": new.id,
                    "executor_id": new.executor_id,
                    "priority": new.priority,
                    "responsibility": new.responsibility,
                    "enabled": old.enabled,
                }
            )
    return migrated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    hermes = shutil.which("hermes")
    if not hermes:
        raise RuntimeError("hermes executable not found")
    latest = _latest_market_shell(hermes)
    contract = dict(latest["contract"])
    existing = str(contract.get("instructions") or "")
    if ROUTE_MARKER in existing:
        migrated = _migrate_bindings(
            source_shell_id=str(latest["supersedes_shell_id"]),
            target_shell_id=str(latest["id"]),
        )
        print(
            json.dumps(
                {
                    "status": "NOOP",
                    "shell_id": latest["id"],
                    "version": latest["version"],
                    "route_marker": ROUTE_MARKER,
                    "bindings_migrated": migrated,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    contract["instructions"] = ROUTE_INSTRUCTIONS + existing
    contract["managed_application_routes"] = {
        "oracle": {
            "app_id": "market-structure-oracle",
            "exclusive_when_request_contains": ["오라클", "Oracle"],
            "external_overlay": False,
            "flow_policy": "ETF Flow D+2",
            "execution_surface": "hermes apps run",
        }
    }
    preview = {
        "status": "DRY_RUN" if not args.apply else "APPLYING",
        "supersedes_shell_id": latest["id"],
        "next_version": int(latest["version"]) + 1,
        "route_marker": ROUTE_MARKER,
        "required_capabilities": latest["required_capabilities"],
        "allowed_capabilities": latest["allowed_capabilities"],
        "evidence_policy": latest["evidence_policy"],
    }
    if not args.apply:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
    ) as contract_file:
        json.dump(contract, contract_file, ensure_ascii=False)
        contract_file.flush()
        command = [
            hermes,
            "supervisor",
            "shell",
            "add-version",
            "--key",
            "market",
            "--name",
            str(latest["name"]),
            "--description",
            str(latest["description"]),
            "--contract",
            contract_file.name,
            "--evidence-policy",
            json.dumps(
                latest["evidence_policy"],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
        for capability in latest["required_capabilities"]:
            command.extend(["--required-capability", str(capability)])
        for capability in latest["allowed_capabilities"]:
            command.extend(["--allowed-capability", str(capability)])
        created = _run_json(command)
    migrated = _migrate_bindings(
        source_shell_id=str(latest["id"]),
        target_shell_id=str(created["id"]),
    )
    print(
        json.dumps(
            {
                **preview,
                "status": "INSTALLED",
                "created": created,
                "bindings_migrated": migrated,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise
