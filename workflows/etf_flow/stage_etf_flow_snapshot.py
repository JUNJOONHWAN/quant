#!/usr/bin/env python3
"""Stage a completed ETF Flow source package into the canonical quant tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")
REQUIRED_INPUTS = (
    "massive_flows.json",
    "fmp_quotes.json",
    "analyst_estimates.json",
    "barchart_qqq.json",
)


class StageError(RuntimeError):
    """Raised when the completed source package cannot be staged intact."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_required(source: Path, destination: Path) -> dict[str, str]:
    if not source.is_file():
        raise StageError(f"Required source artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise StageError(f"Hash mismatch after staging: {source}")
    return {"source": str(source), "staged": str(destination), "sha256": sha256(destination)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-input-dir", type=Path, required=True)
    parser.add_argument("--source-reports-dir", type=Path, required=True)
    parser.add_argument("--quant-data-root", type=Path, required=True)
    parser.add_argument("--report-date", required=True)
    args = parser.parse_args()

    source_input_dir = args.source_input_dir.resolve()
    source_reports_dir = args.source_reports_dir.resolve()
    quant_data_root = args.quant_data_root.resolve()
    snapshot_dir = quant_data_root / "snapshots" / args.report_date
    reports_dir = quant_data_root / "daily_reports"
    staged = [
        copy_required(source_input_dir / name, snapshot_dir / name)
        for name in REQUIRED_INPUTS
    ]
    reports = sorted(source_reports_dir.glob("etf-flow-report-*.md"))
    if not reports:
        raise StageError(f"No ETF Flow daily reports found: {source_reports_dir}")
    staged_reports = [copy_required(report, reports_dir / report.name) for report in reports]
    manifest = {
        "schema_version": "1.0",
        "report_date": args.report_date,
        "staged_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "role": "quant canonical analysis input",
        "source_role": "STOCK runtime/legacy collection compatibility only",
        "inputs": staged,
        "daily_reports": staged_reports,
    }
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "snapshot_dir": str(snapshot_dir), "reports_dir": str(reports_dir), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StageError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        raise SystemExit(1)
