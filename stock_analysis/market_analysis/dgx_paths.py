"""DGX-aware path helpers for STOCK/quant report outputs.

The Mac iCloud tree is a legacy sync target. On DGX, user-facing report copies
should land under ``/home/zooh/Documents/DGX_Outputs/STOCK`` unless an explicit
non-legacy path is provided or the legacy path is actually mounted.
"""

from __future__ import annotations

import os
from pathlib import Path


DGX_OUTPUT_ROOT = Path(
    os.getenv("DGX_STOCK_OUTPUT_ROOT", "/home/zooh/Documents/DGX_Outputs/STOCK")
).expanduser()


def _expanded(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def is_legacy_mac_path(path: Path) -> bool:
    text = str(path)
    return (
        text.startswith("/Users/zooh/")
        or text.startswith("/Volumes/")
        or "Mobile Documents" in text
        or "CloudDocs" in text
        or "CloudStorage" in text
        or "GoogleDrive-" in text
    )


def choose_output_dir(
    env_name: str,
    *relative_parts: str,
    legacy_mac_path: Path | None = None,
) -> Path:
    explicit = os.getenv(env_name, "").strip()
    if explicit:
        candidate = _expanded(explicit)
        if candidate.exists() or not is_legacy_mac_path(candidate):
            return candidate

    if legacy_mac_path is not None and legacy_mac_path.exists():
        return legacy_mac_path.expanduser()

    return DGX_OUTPUT_ROOT.joinpath(*relative_parts)


def choose_file_path(
    env_name: str,
    *relative_parts: str,
    legacy_mac_path: Path | None = None,
) -> Path:
    explicit = os.getenv(env_name, "").strip()
    if explicit:
        candidate = _expanded(explicit)
        if candidate.exists() or not is_legacy_mac_path(candidate):
            return candidate
    if legacy_mac_path is not None and legacy_mac_path.exists():
        return legacy_mac_path.expanduser()
    return DGX_OUTPUT_ROOT.joinpath(*relative_parts)


def knowledge_db_path(env_name: str = "STOCK_KNOWHOW_DB") -> Path:
    explicit = os.getenv(env_name, "").strip()
    if explicit:
        candidate = _expanded(explicit)
        if candidate.exists() or not is_legacy_mac_path(candidate):
            return candidate

    for candidate in (
        Path("/home/zooh/.codex/knowledge/stock_research_knowhow_db.json"),
        Path.home() / ".codex/knowledge/stock_research_knowhow_db.json",
    ):
        if candidate.exists():
            return candidate
    return Path("/home/zooh/.codex/knowledge/stock_research_knowhow_db.json")


def existing_nonlegacy_file(value: str | None) -> str:
    if not value:
        return ""
    candidate = _expanded(value)
    if candidate.exists():
        return str(candidate)
    if is_legacy_mac_path(candidate):
        return ""
    return str(candidate)
