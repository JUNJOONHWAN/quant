"""Verified reader for immutable ETF RADAR releases.

The Quant AI workflow consumes the existing ETF RADAR release instead of
re-downloading its 6k+ FMP/Massive ETF payloads.  Only a COMPLETE release whose
manifest hashes match the files is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Iterable


REQUIRED_TABLES = (
    "02_ETF_MASTER",
    "04_DAILY_QUOTES",
    "05_HIST_RETURNS",
    "36_MASSIVE_FLOW_FUSION",
    "47_EARLY_ACCUMULATION_RADAR",
    "48_MASSIVE_ACCUM_CLUSTER",
    "49_INTEGRATED_ROTATION_RADAR",
    "55_MASSIVE_ACCUM_MEMBER_CACHE",
)
EVIDENCE_TABLES = (
    "02_ETF_MASTER",
    "36_MASSIVE_FLOW_FUSION",
    "47_EARLY_ACCUMULATION_RADAR",
    "48_MASSIVE_ACCUM_CLUSTER",
    "49_INTEGRATED_ROTATION_RADAR",
    "55_MASSIVE_ACCUM_MEMBER_CACHE",
)


class EtfRadarReleaseError(RuntimeError):
    """Raised when ETF RADAR evidence is incomplete or hash-invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EtfRadarReleaseError(f"required ETF RADAR file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EtfRadarReleaseError(f"invalid ETF RADAR JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EtfRadarReleaseError(f"ETF RADAR JSON must be an object: {path}")
    return value


def discover_release(data_root: Path, as_of_date: str) -> Path:
    root = Path(data_root).expanduser().resolve()
    as_of = date.fromisoformat(as_of_date).isoformat()
    candidates: list[tuple[str, str, Path]] = []
    for tier in ("hot", "warm", "archive"):
        tier_root = root / "releases" / tier
        if not tier_root.is_dir():
            continue
        for manifest_path in tier_root.rglob("release_manifest.json"):
            try:
                manifest = _read_object(manifest_path)
            except EtfRadarReleaseError:
                continue
            trade_date = str(manifest.get("trade_date_us") or "")
            if (
                manifest.get("schema_version") == "etfradar-release-v1"
                and manifest.get("complete") is True
                and trade_date
                and trade_date <= as_of
                and manifest_path.parent.joinpath("COMPLETE").is_file()
            ):
                candidates.append(
                    (trade_date, str(manifest.get("created_at_kst") or ""), manifest_path.parent)
                )
    if not candidates:
        raise EtfRadarReleaseError(
            f"no COMPLETE ETF RADAR release exists on or before {as_of}"
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def verify_release(
    release_dir: Path, required_tables: Iterable[str] = REQUIRED_TABLES
) -> dict[str, Any]:
    release = Path(release_dir).expanduser().resolve()
    manifest_path = release / "release_manifest.json"
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != "etfradar-release-v1":
        raise EtfRadarReleaseError("unsupported ETF RADAR release schema")
    if manifest.get("complete") is not True or not release.joinpath("COMPLETE").is_file():
        raise EtfRadarReleaseError(f"ETF RADAR release is not COMPLETE: {release}")
    table_map = {
        str(item.get("sheet_name")): item
        for item in manifest.get("tables") or []
        if isinstance(item, dict)
    }
    verified = []
    for table_name in required_tables:
        table = table_map.get(table_name)
        if not table:
            raise EtfRadarReleaseError(f"release is missing required table: {table_name}")
        table_dir = release / "tables" / table_name
        files = table.get("files")
        if not isinstance(files, list) or not files:
            raise EtfRadarReleaseError(f"release table has no file manifest: {table_name}")
        verified_files = []
        for item in files:
            if not isinstance(item, dict) or not item.get("relative_path"):
                raise EtfRadarReleaseError(f"invalid file manifest in table {table_name}")
            path = table_dir / str(item["relative_path"])
            expected = str(item.get("sha256") or "")
            if not path.is_file():
                raise EtfRadarReleaseError(f"release table file is missing: {path}")
            observed = _sha256(path)
            if observed != expected:
                raise EtfRadarReleaseError(
                    f"release SHA mismatch: table={table_name} file={path.name}"
                )
            verified_files.append(
                {"relative_path": str(path.relative_to(release)), "sha256": observed}
            )
        meta = _read_object(table_dir / "meta.json")
        if int(meta.get("row_count", -1)) != int(table.get("row_count", -2)):
            raise EtfRadarReleaseError(f"release row-count metadata mismatch: {table_name}")
        verified.append(
            {
                "table": table_name,
                "row_count": int(table["row_count"]),
                "files": verified_files,
            }
        )
    return {
        "schema_version": "quant.etfradar_release_binding.v1",
        "release_id": manifest.get("release_id"),
        "trade_date_us": manifest.get("trade_date_us"),
        "release_path": str(release),
        "release_manifest_sha256": _sha256(manifest_path),
        "complete": True,
        "tables": verified,
    }


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_scalar(value.item())
    return str(value)


def read_table_rows(release_dir: Path, table_name: str) -> list[dict[str, Any]]:
    """Read a verified full Parquet table; previews are never accepted as full data."""

    path = Path(release_dir).expanduser().resolve() / "tables" / table_name / "data.parquet"
    try:
        import pandas as pd
    except ImportError as exc:
        raise EtfRadarReleaseError(
            "pandas plus a Parquet engine is required to consume ETF RADAR releases"
        ) from exc
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise EtfRadarReleaseError(f"failed to read full ETF RADAR table: {path}: {exc}") from exc
    return [
        {str(key): _json_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def load_release_evidence(data_root: Path, as_of_date: str) -> dict[str, Any]:
    release = discover_release(data_root, as_of_date)
    binding = verify_release(release)
    tables = {
        table_name: read_table_rows(release, table_name)
        for table_name in EVIDENCE_TABLES
    }
    observed = {name: len(rows) for name, rows in tables.items()}
    expected = {
        item["table"]: item["row_count"]
        for item in binding["tables"]
        if item["table"] in EVIDENCE_TABLES
    }
    if observed != expected:
        raise EtfRadarReleaseError(
            f"ETF RADAR Parquet row counts do not match release manifest: {observed} != {expected}"
        )
    return {"binding": binding, "tables": tables}
