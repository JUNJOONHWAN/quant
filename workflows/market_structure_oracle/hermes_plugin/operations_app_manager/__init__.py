"""Operations App Manager user plugin.

The plugin adds an operator CLI only. Applications remain independent
processes and do not become model tools in the Hermes supervisor.
"""

from __future__ import annotations

from .cli import apps_command, register_cli
from .tools import OPERATIONS_APP_RUN_SCHEMA, handle_operations_app_run


def register(ctx) -> None:
    ctx.register_tool(
        name="operations_app_run",
        toolset="cronjob",
        schema=OPERATIONS_APP_RUN_SCHEMA,
        handler=handle_operations_app_run,
        description=(
            "Operations Role Shell host tool for running registered independent "
            "applications through App Manager."
        ),
        emoji="⚙️",
    )
    ctx.register_cli_command(
        name="apps",
        help="Manage independent applications, cron attachments, and run receipts",
        setup_fn=register_cli,
        handler_fn=apps_command,
        description=(
            "Program-manager surface for independent applications. "
            "Owns registry, lifecycle, schedules, capability validation, "
            "and receipts without absorbing application logic."
        ),
    )
