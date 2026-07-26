#!/usr/bin/env python3
"""Run the DGX quant Hermes prompt matrix against a validated ETF Flow packet."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from build_etf_flow_analysis_packet import PacketError, build_packet


KST = ZoneInfo("Asia/Seoul")
QUANT_ROOT = Path("/home/zooh/Documents/GitHub/quant")


class AnalysisError(RuntimeError):
    """Raised when Hermes returns an incomplete analysis response."""


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AnalysisError("Hermes response did not contain a valid JSON object.") from exc
        if isinstance(value, dict):
            return value
    raise AnalysisError("Hermes response did not contain a JSON object.")


def _run_hermes(*, prompt: str, response_path: Path, timeout: int) -> dict[str, Any]:
    response_path.unlink(missing_ok=True)
    started_at = time.time()
    command = [
        "hermes",
        "--oneshot",
        prompt,
        "--skills",
        "quant-market-analysis",
    ]
    completed = subprocess.run(
        command,
        cwd=QUANT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if not response_path.is_file():
        # Hermes occasionally shortens the requested workflow path to
        # <quant>/artifacts/<run>/responses. Move only a fresh file into the
        # canonical DGX response path so no report artifact remains in source.
        run_name = response_path.parent.parent.name
        alternate_path = QUANT_ROOT / "artifacts" / run_name / "responses" / response_path.name
        if alternate_path.is_file() and alternate_path.stat().st_mtime >= started_at:
            response_path.parent.mkdir(parents=True, exist_ok=True)
            alternate_path.replace(response_path)
    if response_path.is_file() and response_path.stat().st_mtime >= started_at:
        # The prompt intentionally asks Hermes to write only the contract file
        # and emit no chat summary.  The one-shot CLI currently reports that
        # valid empty-final-response case as non-zero, so the fresh validated
        # artifact is the authoritative success signal.
        return _json_object(response_path.read_text(encoding="utf-8"))
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}"
        raise AnalysisError(f"DGX Hermes invocation failed: {detail}")
    raise AnalysisError("Hermes completed without writing the required analysis response file.")


def _run_validated(
    *,
    prompt: str,
    response_path: Path,
    timeout: int,
    validate: Any,
    label: str,
) -> dict[str, Any]:
    retry_prompt = prompt
    errors: list[str] = []
    for attempt in range(1, 4):
        try:
            return validate(_run_hermes(prompt=retry_prompt, response_path=response_path, timeout=timeout))
        except AnalysisError as exc:
            errors.append(str(exc))
            if attempt == 3:
                raise AnalysisError(f"Hermes {label} failed contract validation after 3 attempts: {exc}") from exc
            retry_prompt = (
                f"{prompt}\n\nYour previous response was rejected by the deterministic contract: {exc}. "
                "Rewrite the JSON now. Preserve the required schema and obey every exact evidence and coverage constraint."
            )
    raise AnalysisError(f"Hermes {label} failed without a response: {'; '.join(errors)}")


def _section_prompt(
    packet_path: Path,
    section: dict[str, Any],
    response_path: Path,
    evidence_catalog: dict[str, str],
) -> str:
    expected = {
        "id": section["id"],
        "headline": "short Korean headline",
        "summary": "2-4 Korean sentences grounded only in the packet",
        "stance": "risk_on|neutral|risk_off|mixed",
        "confidence": "high|medium|low",
        "evidence": [{"id": "evidence_catalog id", "value": "copy the catalog value exactly", "meaning": "why it matters"}],
        "limitations": ["coverage or timing limit"],
    }
    return f"""You are the DGX Hermes market-analysis agent. Load the quant-market-analysis skill and follow its preflight requirements. The validated ETF Flow packet is at {packet_path} and the already-verified news supplement is at {response_path.parent / 'news.json'}. Do not recollect, modify, or fabricate market data. Treat these two artifacts as the sole factual sources for this section.

Answer this analysis question: {section['question']}
Use these packet keys: {', '.join(section['packet_keys'])}.

The following is the complete allowed evidence id:value map. Cite exactly two or three entries from this map only, and copy each paired value exactly. An id not listed here will fail validation. Do not convert units and do not use a numeric claim outside cited catalog values:
{json.dumps(evidence_catalog, ensure_ascii=False)}

Write Korean only, apart from ETF tickers and the prescribed enum values. Keep the summary to two to four polished Korean sentences. The packet's report_continuity.coverage is authoritative: never say that prior reports are unavailable when observed_reports is greater than zero, and mention only dates in missing_report_dates as missing. Do not infer missing dates from the flow-data lag.

Use the file tool to write ONLY one JSON object matching this shape to {response_path}. Do not write to any other path. Do not summarize the answer in chat after writing:
{json.dumps(expected, ensure_ascii=False)}
"""


def _coverage_summary(packet: dict[str, Any]) -> str:
    report_coverage = packet["report_continuity"]["coverage"]
    return (
        f"{report_coverage['observed_reports']}/{report_coverage['expected_reports']} daily reports; "
        f"{len(packet['flow_sessions'])} ETF flow sessions observed"
    )


def _synthesis_prompt(
    packet_path: Path,
    section_path: Path,
    response_path: Path,
    coverage_summary: str,
    evidence_catalog: dict[str, str],
) -> str:
    expected = {
        "market_statement": {
            "headline": "Korean market statement",
            "body": "3-5 Korean sentences",
            "stance": "risk_on|neutral|risk_off|mixed",
            "confidence": "high|medium|low",
            "key_levels": [{"id": "evidence_catalog id", "value": "copy the catalog value exactly", "meaning": "interpretation"}],
        },
        "mood_change": {
            "state": "strengthening|weakening|rotation|unchanged|insufficient_evidence",
            "headline": "Korean headline",
            "summary": "Korean comparison to prior daily reports",
            "evidence": [{"id": "evidence_catalog id", "value": "copy the catalog value exactly", "meaning": "change"}],
        },
        "weekly_continuity": {
            "state": "persistent|reversing|choppy|insufficient_evidence",
            "headline": "Korean headline",
            "summary": "Korean 7-day continuity description",
            "observed_sessions": coverage_summary,
            "missing_coverage_effect": "Korean limitation",
        },
        "limitations": ["Korean limitation"],
    }
    return f"""You are the DGX Hermes market-analysis agent. Load the quant-market-analysis skill and follow its preflight requirements. Read the validated ETF Flow packet at {packet_path}, the news supplement at {response_path.parent / 'news.json'}, and the six section analyses at {section_path}. These artifacts are the only factual sources. Do not recollect, modify, or fabricate market data.

Synthesize a concise market statement, the change from prior daily reports, and the observed weekly continuity. Preserve every missing-date or source-timing limit from the packet. Never label an incomplete seven-day window as complete.

Use exactly two or three key_levels and exactly two or three mood_change evidence entries from this allowed id:value map. Copy the paired values exactly; an id not listed here will fail validation. Do not convert units or make numeric claims outside cited catalog values:
{json.dumps(evidence_catalog, ensure_ascii=False)}

Use the file tool to write ONLY one JSON object matching this shape to {response_path}. Do not write to any other path and do not summarize the answer in chat after writing:
{json.dumps(expected, ensure_ascii=False)}

Write Korean only, apart from ETF tickers and the prescribed enum values. The exact observed-session string is "{coverage_summary}"; set weekly_continuity.observed_sessions to exactly that text. report_continuity.coverage has observed reports, so do not state that prior daily reports are unavailable. The only missing report date is the packet's missing_report_dates value; do not add dates based on flow lag.
"""


def _news_prompt(packet_path: Path, response_path: Path) -> str:
    expected = {
        "source_status": "confirmed|partial|failed",
        "items": [
            {
                "headline": "article headline",
                "published_at": "ISO time or source-published date",
                "source": "publisher",
                "url": "https URL",
                "relevance": "Korean explanation tied to the ETF flow packet",
                "affected_sections": ["flow_structure", "daily_mood_shift"],
            }
        ],
        "limitations": ["Korean data-access or timing limitation"],
    }
    return f"""You are the DGX Hermes market-analysis agent. Load the quant-market-analysis skill and follow its preflight requirements. The validated ETF Flow packet is at {packet_path}.

Use the configured DGX market-analysis MCP route to collect current news that can explain or challenge this ETF Flow packet. Prefer mcp_stock_market_data_fmp_stock_news for SPY, QQQ, SMH, TLT, HYG, and the highest-flow ETFs. If that MCP source is insufficient, use Hermes web search only to add reputable, directly linked articles. Do not change, replace, or re-collect the validated ETF Flow source values.

Return three to eight articles when available. Each item must include a real source URL, its published date, and a Korean relevance explanation. Explicitly mark the news source partial or failed when access or coverage is incomplete. Do not invent a headline, date, source, or URL.

Use the file tool to write ONLY one JSON object matching this shape to {response_path}. Do not write to any other path and do not summarize the answer in chat after writing:
{json.dumps(expected, ensure_ascii=False)}
"""


def _validate_evidence(items: Any, catalog: dict[str, str], context: str) -> None:
    if not isinstance(items, list) or not items:
        raise AnalysisError(f"{context} has no evidence.")
    for item in items:
        if not isinstance(item, dict) or item.get("id") not in catalog:
            raise AnalysisError(f"{context} cited an unknown evidence id.")
        if str(item.get("value")) != catalog[str(item["id"])]:
            raise AnalysisError(f"{context} changed the display value for evidence {item['id']}.")


def _validate_section(value: dict[str, Any], section_id: str, catalog: dict[str, str]) -> dict[str, Any]:
    required = ("id", "headline", "summary", "stance", "confidence", "evidence", "limitations")
    missing = [key for key in required if key not in value]
    if missing:
        raise AnalysisError(f"Hermes section {section_id} is missing fields: {missing}")
    if value["id"] != section_id:
        raise AnalysisError(f"Hermes section id mismatch: expected {section_id}, got {value['id']}")
    _validate_evidence(value["evidence"], catalog, f"Hermes section {section_id}")
    return value


def _validate_news(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("source_status") not in {"confirmed", "partial", "failed"}:
        raise AnalysisError("Hermes news supplement must report source_status.")
    items = value.get("items")
    if not isinstance(items, list):
        raise AnalysisError("Hermes news supplement items must be an array.")
    for item in items:
        required = ("headline", "published_at", "source", "url", "relevance", "affected_sections")
        if not isinstance(item, dict) or any(not item.get(key) for key in required):
            raise AnalysisError("Hermes news item is missing a required field.")
        if not re.match(r"https?://", str(item["url"])):
            raise AnalysisError("Hermes news item does not contain an absolute HTTP URL.")
        if not isinstance(item["affected_sections"], list):
            raise AnalysisError("Hermes news affected_sections must be an array.")
    if not isinstance(value.get("limitations"), list):
        raise AnalysisError("Hermes news supplement limitations must be an array.")
    return value


def _validate_synthesis(value: dict[str, Any], catalog: dict[str, str], coverage_summary: str) -> dict[str, Any]:
    required = ("market_statement", "mood_change", "weekly_continuity", "limitations")
    missing = [key for key in required if key not in value]
    if missing:
        raise AnalysisError(f"Hermes synthesis is missing fields: {missing}")
    if not isinstance(value["limitations"], list):
        raise AnalysisError("Hermes synthesis limitations must be an array.")
    _validate_evidence(value["market_statement"].get("key_levels"), catalog, "Hermes market statement")
    _validate_evidence(value["mood_change"].get("evidence"), catalog, "Hermes mood change")
    if value["weekly_continuity"].get("observed_sessions") != coverage_summary:
        raise AnalysisError("Hermes weekly continuity did not preserve the validated coverage summary.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet = build_packet(args.input_dir, args.reports_dir)
    packet_path = args.output_dir / "packet.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    response_dir = args.output_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    evidence_catalog = {item["id"]: item["value"] for item in packet["evidence_catalog"]}
    coverage_summary = _coverage_summary(packet)
    news = _run_validated(
        validate=_validate_news,
        label="news supplement",
        prompt=_news_prompt(packet_path, response_dir / "news.json"),
        response_path=response_dir / "news.json",
        timeout=args.timeout,
    )
    sections = []
    for section in packet["prompt_matrix"]:
        result = _run_validated(
            validate=lambda value, section_id=section["id"]: _validate_section(value, section_id, evidence_catalog),
            label=f"section {section['id']}",
            prompt=_section_prompt(packet_path, section, response_dir / f"{section['id']}.json", evidence_catalog),
            response_path=response_dir / f"{section['id']}.json",
            timeout=args.timeout,
        )
        sections.append(result)
    sections_path = args.output_dir / "sections.json"
    sections_path.write_text(json.dumps(sections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    synthesis = _run_validated(
        validate=lambda value: _validate_synthesis(value, evidence_catalog, coverage_summary),
        label="synthesis",
        prompt=_synthesis_prompt(
            packet_path,
            sections_path,
            response_dir / "synthesis.json",
            coverage_summary,
            evidence_catalog,
        ),
        response_path=response_dir / "synthesis.json",
        timeout=args.timeout,
    )
    artifact = {
        "schema_version": "1.0",
        "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "analysis_owner": packet["analysis_owner"],
        "source_status": packet["source_status"],
        "source_descriptions": packet["source_descriptions"],
        "prompt_matrix": packet["prompt_matrix"],
        "evidence_catalog": packet["evidence_catalog"],
        "news": news,
        "market_statement": synthesis["market_statement"],
        "mood_change": synthesis["mood_change"],
        "weekly_continuity": synthesis["weekly_continuity"],
        "sections": sections,
        "coverage": {
            "flow_sessions": packet["flow_sessions"],
            "report_continuity": packet["report_continuity"],
        },
        "limitations": synthesis["limitations"],
        "packet_sha256": __import__("hashlib").sha256(packet_path.read_bytes()).hexdigest(),
    }
    output_path = args.output_dir / "analysis.json"
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output_path), "sections": len(sections), "packet_sha256": artifact["packet_sha256"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PacketError, AnalysisError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        raise SystemExit(1)
