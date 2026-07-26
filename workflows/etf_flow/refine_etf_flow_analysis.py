#!/usr/bin/env python3
"""Run an on-demand Hermes analysis alternative against an accepted ETF Flow packet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from run_hermes_etf_flow_analysis import AnalysisError, _coverage_summary, _run_hermes, _validate_section, _validate_synthesis


FORBIDDEN_SCRIPT = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisError(f"Required JSON file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisError(f"Expected a JSON object: {path}")
    return value


def strategy_instruction(strategy: str) -> str:
    instructions = {
        "evidence_editor": "Rewrite the primary answer as polished Korean. Correct mixed scripts, unsupported session wording, and imprecise coverage language while keeping the exact evidence contract.",
        "conservative_editor": "Rewrite the primary answer with a conservative analyst stance. Separate observed facts from interpretation, lower confidence where coverage is incomplete, and avoid directional language that the packet cannot prove.",
        "coverage_editor": "Rewrite the primary answer with coverage and chronology as the priority. Clearly distinguish observed ETF Flow sessions, daily-report coverage, data lag, and unknown periods without inventing missing session dates.",
    }
    try:
        return instructions[strategy]
    except KeyError as exc:
        raise AnalysisError(f"Unknown analysis alternative: {strategy}") from exc


def editable_texts(candidate: dict[str, Any]) -> list[str]:
    texts = [
        str(candidate["market_statement"]["headline"]),
        str(candidate["market_statement"]["body"]),
        str(candidate["mood_change"]["headline"]),
        str(candidate["mood_change"]["summary"]),
        str(candidate["weekly_continuity"]["headline"]),
        str(candidate["weekly_continuity"]["summary"]),
        str(candidate["weekly_continuity"]["missing_coverage_effect"]),
    ]
    for section in candidate["sections"]:
        texts.extend([str(section["headline"]), str(section["summary"])])
    return texts


def validate_editor_quality(candidate: dict[str, Any], packet: dict[str, Any]) -> None:
    if any(FORBIDDEN_SCRIPT.search(text) for text in editable_texts(candidate)):
        raise AnalysisError("Alternative contains non-Korean CJK characters in dashboard text.")
    flow_dates = {item["date"] for item in packet["flow_sessions"]}
    for text in editable_texts(candidate):
        for date in re.findall(r"20\d{2}-\d{2}-\d{2}", text):
            if date not in flow_dates and re.search(rf"{re.escape(date)}[^.\n]{{0,32}}(?:flow|세션)", text, flags=re.IGNORECASE):
                raise AnalysisError(f"Alternative labeled unobserved date {date} as an ETF Flow session.")
        if re.search(r"(?:매\s*세션|매번|각\s*세션)[^.\n]{0,96}(?:반도체|\+\$1\.95B)", text):
            raise AnalysisError("Alternative converted a category aggregate into a per-session claim.")
        if re.search(r"7/9[^.\n]{0,48}(?:flow|세션).{0,24}(?:부재|누락)", text, flags=re.IGNORECASE):
            raise AnalysisError("Alternative invented a missing 7/9 ETF Flow session.")


def build_prompt(
    packet_path: Path,
    analysis_path: Path,
    response_path: Path,
    coverage_summary: str,
    catalog: dict[str, str],
    strategy: str,
) -> str:
    expected = {
        "market_statement": {"headline": "Korean", "body": "Korean", "stance": "risk_on|neutral|risk_off|mixed", "confidence": "high|medium|low", "key_levels": [{"id": "allowed", "value": "exact", "meaning": "Korean"}]},
        "mood_change": {"state": "strengthening|weakening|rotation|unchanged|insufficient_evidence", "headline": "Korean", "summary": "Korean", "evidence": [{"id": "allowed", "value": "exact", "meaning": "Korean"}]},
        "weekly_continuity": {"state": "persistent|reversing|choppy|insufficient_evidence", "headline": "Korean", "summary": "Korean", "observed_sessions": coverage_summary, "missing_coverage_effect": "Korean"},
        "sections": [{"id": "existing section id", "headline": "Korean", "summary": "Korean", "stance": "risk_on|neutral|risk_off|mixed", "confidence": "high|medium|low", "evidence": [{"id": "allowed", "value": "exact", "meaning": "Korean"}], "limitations": ["Korean"]}],
        "limitations": ["Korean"],
    }
    observed_flow_dates = [item["date"] for item in read_json(packet_path)["flow_sessions"]]
    return f"""You are an on-demand Hermes analysis alternative in the DGX quant market-analysis workflow. Read the validated packet at {packet_path} and primary candidate at {analysis_path}. Do not collect data, search the web, or modify news/source fields.

Selected alternative: {strategy}. {strategy_instruction(strategy)}

Return every field below. Preserve exactly six existing section ids and copy every id:value pair from the allowed map without conversion. Use two or three evidence items per section, market statement, and mood change. Do not introduce numeric claims outside cited evidence.

Observed ETF Flow session dates are exactly {observed_flow_dates}. Never call another date an ETF Flow session or a missing Flow session. Daily report coverage is exactly {coverage_summary}; the sole missing daily report date is 2026-07-10. Do not call 2026-07-09 missing, and do not turn a category aggregate into a per-session claim.

Write natural Korean only, apart from tickers, exact evidence values, and enum values. Do not use Chinese Han characters, mixed-script fragments, or untranslated analyst jargon.

Allowed evidence id:value map:
{json.dumps(catalog, ensure_ascii=False)}

Use the file tool to write ONLY one JSON object matching this shape to {response_path}. Do not write elsewhere or summarize in chat:
{json.dumps(expected, ensure_ascii=False)}
"""


def normalize_sections(value: Any, expected_ids: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(expected_ids):
        raise AnalysisError("Alternative did not return exactly six sections.")
    by_id = {section.get("id"): section for section in value if isinstance(section, dict)}
    if set(by_id) != set(expected_ids):
        raise AnalysisError("Alternative did not return the six original section ids.")
    return [by_id[section_id] for section_id in expected_ids]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=["evidence_editor", "conservative_editor", "coverage_editor"], required=True)
    parser.add_argument("--timeout", type=int, default=420)
    args = parser.parse_args()

    packet = read_json(args.packet)
    original = read_json(args.analysis)
    catalog = {item["id"]: item["value"] for item in packet["evidence_catalog"]}
    coverage_summary = _coverage_summary(packet)
    response_path = args.output.parent / "responses" / f"{args.strategy}.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = [section["id"] for section in original.get("sections", [])]
    base_prompt = build_prompt(args.packet, args.analysis, response_path, coverage_summary, catalog, args.strategy)
    last_error: AnalysisError | None = None
    result: dict[str, Any] | None = None
    for attempt in range(1, 4):
        prompt = base_prompt if last_error is None else (
            f"{base_prompt}\n\nYour previous alternative response was rejected: {last_error}. "
            "Rewrite it now with only Korean dashboard prose and every coverage rule intact."
        )
        try:
            edited = _run_hermes(prompt=prompt, response_path=response_path, timeout=args.timeout)
            sections = normalize_sections(edited.get("sections"), expected_ids)
            for section in sections:
                _validate_section(section, section["id"], catalog)
            synthesis = _validate_synthesis(edited, catalog, coverage_summary)
            candidate = dict(original)
            candidate.update({
                "market_statement": synthesis["market_statement"],
                "mood_change": synthesis["mood_change"],
                "weekly_continuity": synthesis["weekly_continuity"],
                "sections": sections,
                "limitations": synthesis["limitations"],
                "analysis_strategy": args.strategy,
            })
            validate_editor_quality(candidate, packet)
            result = candidate
            break
        except AnalysisError as exc:
            last_error = exc
    if result is None:
        raise AnalysisError(f"Alternative {args.strategy} failed after 3 attempts: {last_error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "strategy": args.strategy}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        raise SystemExit(1)
