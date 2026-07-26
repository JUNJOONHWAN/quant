#!/usr/bin/env python3
"""Run the canonical quant ETF Flow analysis and publish the current artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")
QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")
WORKFLOW = QUANT_ROOT / "workflows" / "etf_flow"
LEGACY_INPUT = Path("/home/zooh/Documents/GitHub/STOCK/quant/etf-flow-report/input")
REPORT_ROOT = Path("/home/zooh/Documents/DGX_Outputs/STOCK/ETF Flow")
LEGACY_REPORTS = REPORT_ROOT


def _valid_existing_analysis(path: Path, packet_path: Path, report_date: str) -> bool:
    """Accept only a fresh analysis bound to the exact packet bytes."""
    if not path.is_file() or not packet_path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if str(value.get("generated_at_kst", ""))[:10] != report_date:
        return False
    if value.get("packet_sha256") != hashlib.sha256(packet_path.read_bytes()).hexdigest():
        return False
    expected_sections = len(packet.get("prompt_matrix") or [])
    if not isinstance(value.get("sections"), list) or len(value["sections"]) != expected_sections:
        return False
    news = value.get("news")
    if not isinstance(news, dict) or news.get("source_status") not in {"confirmed", "partial", "failed"}:
        return False
    if not isinstance(news.get("items"), list) or not isinstance(news.get("limitations"), list):
        return False
    return value.get("coverage") is not None and value.get("market_statement") is not None


def _semantic_packet_sha256(packet_path: Path) -> str:
    """Hash analysis inputs while ignoring staging timestamps and directory names."""
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.pop("generated_at_kst", None)
    for source_file in packet.get("source_files") or []:
        path = source_file.get("path")
        if path:
            source_file["path"] = Path(path).name
    canonical = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reuse_matching_accepted_analysis(
    *, artifact_dir: Path, packet_path: Path, report_date: str
) -> str | None:
    """Rebind a prior accepted analysis only when every semantic packet input matches."""
    current_digest = _semantic_packet_sha256(packet_path)
    for candidate in sorted((REPORT_ROOT / "data" / "artifacts").glob("*/analysis.json"), reverse=True):
        if candidate.parent == artifact_dir:
            continue
        candidate_date = candidate.parent.name
        candidate_packet = candidate.with_name("packet.json")
        if not _valid_existing_analysis(candidate, candidate_packet, candidate_date):
            continue
        if _semantic_packet_sha256(candidate_packet) != current_digest:
            continue
        accepted = json.loads(candidate.read_text(encoding="utf-8"))
        rebound = copy.deepcopy(accepted)
        rebound["generated_at_kst"] = datetime.now(KST).isoformat(timespec="seconds")
        rebound["packet_sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
        rebound["analysis_reuse"] = {
            "mode": "exact_semantic_packet_match",
            "source_analysis": str(candidate),
            "source_report_date": candidate_date,
            "semantic_packet_sha256": current_digest,
            "reason": "Routine cron is model-independent; no new analysis text was generated.",
        }
        news = rebound.get("news")
        if isinstance(news, dict):
            news["source_status"] = "partial"
            news_limitations = news.setdefault("limitations", [])
            news_limitations.append(
                f"{candidate_date} 승인 분석의 뉴스 보강을 재사용했으며 "
                f"{report_date} 현재 뉴스 재수집은 수행하지 않았습니다."
            )
        rebound.setdefault("limitations", []).append(
            f"분석 입력이 {candidate_date} 승인 패킷과 정확히 동일하여 기존 분석을 재사용했습니다."
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.joinpath("analysis.json").write_text(
            json.dumps(rebound, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return str(candidate)
    return None


class RefreshError(RuntimeError):
    """Raised when a required quant refresh stage fails."""


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=WORKFLOW, text=True, check=False)
    if completed.returncode != 0:
        raise RefreshError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--site-access-token-file", type=Path)
    parser.add_argument(
        "--analysis-strategy",
        choices=["primary", "evidence_editor", "conservative_editor", "coverage_editor"],
        default="primary",
    )
    parser.add_argument("--report-date")
    parser.add_argument("--timeout", type=int, default=420)
    args = parser.parse_args()
    report_date = args.report_date or datetime.now(KST).date().isoformat()
    data_root = REPORT_ROOT / "data"
    artifact_dir = data_root / "artifacts" / report_date
    state_path = artifact_dir / "publish_state.json"
    try:
        run([
            "python3", str(WORKFLOW / "stage_etf_flow_snapshot.py"),
            "--source-input-dir", str(LEGACY_INPUT),
            "--source-reports-dir", str(LEGACY_REPORTS),
            "--quant-data-root", str(data_root),
            "--report-date", report_date,
        ])
        run([
            "python3", str(WORKFLOW / "build_etf_flow_analysis_packet.py"),
            "--input-dir", str(data_root / "snapshots" / report_date),
            "--reports-dir", str(data_root / "daily_reports"),
            "--output", str(artifact_dir / "packet.json"),
        ])
        selected_analysis = artifact_dir / "analysis.json"
        analysis_reused = False
        analysis_reused_from = None
        if args.analysis_strategy == "primary" and _valid_existing_analysis(
            selected_analysis, artifact_dir / "packet.json", report_date
        ):
            analysis_reused = True
            existing = json.loads(selected_analysis.read_text(encoding="utf-8"))
            analysis_reused_from = (
                (existing.get("analysis_reuse") or {}).get("source_analysis")
                or str(selected_analysis)
            )
        elif args.analysis_strategy == "primary":
            analysis_reused_from = _reuse_matching_accepted_analysis(
                artifact_dir=artifact_dir,
                packet_path=artifact_dir / "packet.json",
                report_date=report_date,
            )
            if analysis_reused_from is None:
                raise RefreshError(
                    "analysis_required: no accepted analysis matches the current semantic packet; "
                    "routine cron will not invoke Hermes or any vLLM-backed model"
                )
            analysis_reused = True
        else:
            run([
                "python3", str(WORKFLOW / "run_hermes_etf_flow_analysis.py"),
                "--input-dir", str(data_root / "snapshots" / report_date),
                "--reports-dir", str(data_root / "daily_reports"),
                "--output-dir", str(artifact_dir),
                "--timeout", str(args.timeout),
            ])
        if args.analysis_strategy != "primary":
            selected_analysis = artifact_dir / f"analysis-{args.analysis_strategy}.json"
            run([
                "python3", str(WORKFLOW / "refine_etf_flow_analysis.py"),
                "--packet", str(artifact_dir / "packet.json"),
                "--analysis", str(artifact_dir / "analysis.json"),
                "--output", str(selected_analysis),
                "--strategy", args.analysis_strategy,
                "--timeout", str(args.timeout),
            ])
        publish_command = [
            "python3", str(WORKFLOW / "publish_etf_flow_analysis.py"),
            "--analysis", str(selected_analysis),
            "--snapshot-dir", str(data_root / "snapshots" / report_date),
            "--report-date", report_date,
            "--endpoint", args.endpoint,
            "--token-file", str(args.token_file),
        ]
        if args.site_access_token_file:
            publish_command.extend(["--site-access-token-file", str(args.site_access_token_file)])
        run(publish_command)
        state = {
            "status": "ok",
            "report_date": report_date,
            "analysis_strategy": args.analysis_strategy,
            "analysis_path": str(selected_analysis),
            "analysis_reused": analysis_reused,
            "analysis_reused_from": analysis_reused_from,
            "published_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
            "news_source_status": json.loads(selected_analysis.read_text(encoding="utf-8"))["news"]["source_status"],
            "news_item_count": len(json.loads(selected_analysis.read_text(encoding="utf-8"))["news"]["items"]),
        }
    except RefreshError as exc:
        state = {"status": "error", "report_date": report_date, "error": str(exc), "failed_at_kst": datetime.now(KST).isoformat(timespec="seconds")}
        artifact_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(state, ensure_ascii=False))
        return 1
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
