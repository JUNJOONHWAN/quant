#!/usr/bin/env python3
"""Build the immutable input packet used by the quant Hermes ETF Flow analyst.

The script intentionally does not collect or repair market data. It validates
the completed ETF Flow snapshot, calculates transparent rollups, and preserves
coverage gaps for the Hermes prompt layer to disclose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")

CATEGORY_DEFINITIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("large_cap", "Large-cap indices", ("SPY", "QQQ", "IWM", "RSP", "VOO")),
    ("sector", "GICS sectors", ("XLK", "XLF", "XLE", "XLV", "XLU", "XLRE", "XLC", "XLY", "XLP", "XLB", "XLI")),
    ("semiconductor", "Semiconductors", ("SMH", "SOXL")),
    ("thematic", "Thematic", ("IBB", "ROBO", "ICLN")),
    ("fixed_income", "Fixed income", ("TLT", "IEF", "HYG", "LQD")),
    ("commodity", "Commodities", ("GLD", "SLV")),
    ("leveraged", "Leveraged and inverse", ("TQQQ", "SQQQ", "SPXL", "UPRO")),
)

PROMPT_MATRIX = (
    {
        "id": "flow_structure",
        "title": "Flow structure",
        "question": "Which ETF-flow clusters are directing risk appetite, and where is the strongest cross-category disagreement?",
        "packet_keys": ("category_rollups", "market_anchors"),
    },
    {
        "id": "price_confirmation",
        "title": "Price and volume confirmation",
        "question": "Which flow signals are confirmed or contradicted by price changes and relative volume? Do not infer intraday facts beyond the packet.",
        "packet_keys": ("category_rollups", "market_anchors"),
    },
    {
        "id": "options_risk",
        "title": "Options risk map",
        "question": "How do QQQ volatility, put-call ratios, gamma flip, and option walls change the interpretation of the ETF-flow signal?",
        "packet_keys": ("options", "market_anchors"),
    },
    {
        "id": "rotation_and_defense",
        "title": "Rotation and defense",
        "question": "Describe the rotation between equity beta, credit, duration, commodities, and leverage. State what would invalidate this reading.",
        "packet_keys": ("category_rollups", "market_anchors", "analyst_context"),
    },
    {
        "id": "daily_mood_shift",
        "title": "Change from daily reports",
        "question": "Compare the current validated snapshot with the preceding daily ETF Flow reports. Classify the change as strengthening, weakening, rotation, unchanged, or insufficient evidence, and cite the exact observations.",
        "packet_keys": ("report_continuity", "flow_sessions", "source_status"),
    },
    {
        "id": "weekly_continuity",
        "title": "Weekly continuity",
        "question": "Assess whether the observed flow sessions form a persistent, reversing, choppy, or insufficient weekly pattern. Preserve missing sessions and explain their effect on confidence.",
        "packet_keys": ("flow_sessions", "report_continuity", "category_rollups"),
    },
)


class PacketError(RuntimeError):
    """Raised when a supposedly validated ETF Flow snapshot is incomplete."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PacketError(f"Required source file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PacketError(f"Invalid JSON in source file: {path}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_usd(value: float) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1_000_000_000:.2f}B"


def _date_key(row: dict[str, Any]) -> str:
    return str(row.get("processed_date") or row.get("effective_date") or "")[:10]


def _parse_report_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    report_date = re.search(r"ETF Flow Daily Report\s+[—-]\s+(\d{4}-\d{2}-\d{2})", text)
    asof = re.search(r"report_date=(\d{4}-\d{2}-\d{2}),\s*latest_flow=(\d{4}-\d{2}-\d{2})\s*\(D\+(\d+)", text)
    source_status = re.search(r"\| Barchart Options \|[^\n]*\|\s*([^|]+)\|", text)
    insights = re.search(r"## 11\. Key Insights\s*(.*?)(?:\n---|\Z)", text, re.DOTALL)
    return {
        "report_date": report_date.group(1) if report_date else path.stem.removeprefix("etf-flow-report-"),
        "latest_flow_date": asof.group(2) if asof else None,
        "flow_lag_days": int(asof.group(3)) if asof else None,
        "barchart_status": source_status.group(1).strip() if source_status else None,
        "key_insights": [line.strip() for line in (insights.group(1).splitlines() if insights else []) if line.strip().startswith("-")],
        "path": str(path),
        "sha256": _sha256(path),
    }


def _coverage_window(latest_report_date: str, report_dates: Iterable[str]) -> dict[str, Any]:
    latest = date.fromisoformat(latest_report_date)
    expected = [latest - timedelta(days=offset) for offset in range(6, -1, -1)]
    reported = set(report_dates)
    return {
        "window_start": expected[0].isoformat(),
        "window_end": expected[-1].isoformat(),
        "expected_report_dates": [item.isoformat() for item in expected],
        "observed_report_dates": sorted(reported),
        "missing_report_dates": [item.isoformat() for item in expected if item.isoformat() not in reported],
        "observed_reports": len(reported),
        "expected_reports": len(expected),
    }


def _latest_by_symbol(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        symbol = str(row.get("composite_ticker") or "").upper()
        if symbol:
            grouped[symbol].append(row)
    return {
        symbol: sorted(values, key=_date_key)[-1]
        for symbol, values in grouped.items()
        if values
    }


def _build_category_rollups(
    latest: dict[str, dict[str, Any]],
    quotes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rollups: list[dict[str, Any]] = []
    for category_id, label, symbols in CATEGORY_DEFINITIONS:
        members = []
        for symbol in symbols:
            flow = latest.get(symbol)
            quote = quotes.get(symbol)
            if not flow or not quote:
                raise PacketError(f"Required {category_id} member is missing from the validated snapshot: {symbol}")
            members.append(
                {
                    "symbol": symbol,
                    "processed_date": _date_key(flow),
                    "fund_flow": _float(flow.get("fund_flow")),
                    "nav": _float(flow.get("nav")),
                    "shares_outstanding": _float(flow.get("shares_outstanding")),
                    "price": _float(quote.get("price")),
                    "change_percent": _float(quote.get("changesPercentage")),
                    "volume_ratio": _float(quote.get("volume")) / _float(quote.get("avgVolume")) if _float(quote.get("avgVolume")) else None,
                }
            )
        inflows = sorted((item for item in members if item["fund_flow"] > 0), key=lambda item: item["fund_flow"], reverse=True)
        outflows = sorted((item for item in members if item["fund_flow"] < 0), key=lambda item: item["fund_flow"])
        change_values = [item["change_percent"] for item in members]
        rollups.append(
            {
                "id": category_id,
                "label": label,
                "members": members,
                "net_flow": sum(item["fund_flow"] for item in members),
                "inflow": sum(item["fund_flow"] for item in members if item["fund_flow"] > 0),
                "outflow": sum(item["fund_flow"] for item in members if item["fund_flow"] < 0),
                "average_price_change_percent": fmean(change_values) if change_values else None,
                "top_inflows": inflows[:3],
                "top_outflows": outflows[:3],
            }
        )
    return rollups


def _build_flow_sessions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        session_date = _date_key(row)
        if session_date:
            sessions[session_date].append(row)
    output = []
    for session_date, values in sorted(sessions.items()):
        symbol_map = {str(row.get("composite_ticker") or "").upper(): row for row in values}
        output.append(
            {
                "date": session_date,
                "net_flow": sum(_float(row.get("fund_flow")) for row in values),
                "positive_etfs": sum(1 for row in values if _float(row.get("fund_flow")) > 0),
                "negative_etfs": sum(1 for row in values if _float(row.get("fund_flow")) < 0),
                "qqq_flow": _float(symbol_map.get("QQQ", {}).get("fund_flow")) if "QQQ" in symbol_map else None,
                "spy_flow": _float(symbol_map.get("SPY", {}).get("fund_flow")) if "SPY" in symbol_map else None,
                "hyg_flow": _float(symbol_map.get("HYG", {}).get("fund_flow")) if "HYG" in symbol_map else None,
                "tlt_flow": _float(symbol_map.get("TLT", {}).get("fund_flow")) if "TLT" in symbol_map else None,
            }
        )
    return output


def _build_evidence_catalog(
    category_rollups: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    options: dict[str, Any],
    flow_sessions: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> list[dict[str, str]]:
    """Give Hermes exact display strings to cite without unit conversion."""

    catalog: list[dict[str, str]] = []

    def add(evidence_id: str, value: str, description: str) -> None:
        catalog.append({"id": evidence_id, "value": value, "description": description})

    for rollup in category_rollups:
        category_id = str(rollup["id"])
        add(f"category.{category_id}.net_flow", _format_usd(_float(rollup["net_flow"])), f"{rollup['label']} net ETF flow")
        add(
            f"category.{category_id}.average_price_change_percent",
            f"{_float(rollup['average_price_change_percent']):+.2f}%",
            f"{rollup['label']} equal-weight average price change",
        )
        for member in list(rollup["top_inflows"]) + list(rollup["top_outflows"]):
            symbol = str(member["symbol"])
            add(
                f"category.{category_id}.{symbol}.flow",
                _format_usd(_float(member["fund_flow"])),
                f"{symbol} latest ETF fund flow on {member['processed_date']}",
            )
    for anchor in anchors:
        add(f"anchor.{anchor['symbol']}.flow", _format_usd(_float(anchor["flow"])), f"{anchor['symbol']} latest ETF fund flow on {anchor['flow_date']}")
        add(f"anchor.{anchor['symbol']}.price_change", f"{_float(anchor['change_percent']):+.2f}%", f"{anchor['symbol']} quote change")
    for key, value in (options.get("metrics") or {}).items():
        unit = "%" if key in {"iv", "iv_rank", "iv_percentile", "hist_volatility"} else ""
        add(f"options.{key}", f"{_float(value):.2f}{unit}", f"QQQ option metric {key}")
    for session in flow_sessions:
        session_date = str(session["date"])
        add(f"session.{session_date}.net_flow", _format_usd(_float(session["net_flow"])), f"Aggregate observed ETF fund flow on {session_date}")
        for symbol in ("qqq", "spy", "hyg", "tlt"):
            value = session.get(f"{symbol}_flow")
            if value is not None:
                add(f"session.{session_date}.{symbol}_flow", _format_usd(_float(value)), f"{symbol.upper()} observed ETF fund flow on {session_date}")
    add("coverage.report_count", f"{coverage['observed_reports']}/{coverage['expected_reports']}", "Daily reports observed in the latest seven calendar days")
    add("coverage.missing_report_dates", ", ".join(coverage["missing_report_dates"]) or "none", "Missing daily ETF report dates")
    return catalog


def build_packet(input_dir: Path, reports_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    reports_dir = reports_dir.resolve()
    source_paths = {
        "massive_flows": input_dir / "massive_flows.json",
        "fmp_quotes": input_dir / "fmp_quotes.json",
        "analyst_estimates": input_dir / "analyst_estimates.json",
        "barchart_qqq": input_dir / "barchart_qqq.json",
    }
    massive = _read_json(source_paths["massive_flows"])
    quotes_raw = _read_json(source_paths["fmp_quotes"])
    analysts = _read_json(source_paths["analyst_estimates"])
    options = _read_json(source_paths["barchart_qqq"])
    rows = massive.get("results") if isinstance(massive, dict) else None
    if not isinstance(rows, list):
        raise PacketError("Massive flow payload must contain a results list.")
    if not isinstance(quotes_raw, list) or not isinstance(analysts, dict) or not isinstance(options, dict):
        raise PacketError("FMP/Barchart source shapes do not match the ETF Flow contract.")
    quotes = {str(item.get("symbol") or "").upper(): item for item in quotes_raw if isinstance(item, dict) and item.get("symbol")}
    latest = _latest_by_symbol([row for row in rows if isinstance(row, dict)])
    required_symbols = {symbol for _, _, symbols in CATEGORY_DEFINITIONS for symbol in symbols}
    missing_flows = sorted(required_symbols - set(latest))
    missing_quotes = sorted(required_symbols - set(quotes))
    if missing_flows or missing_quotes:
        raise PacketError(f"Validated ETF universe is incomplete; missing flows={missing_flows}, quotes={missing_quotes}")
    metrics = options.get("metrics")
    if not isinstance(metrics, dict) or len(metrics) < 9:
        raise PacketError("Barchart QQQ options payload must contain all nine metrics.")
    if len(analysts) < 7:
        raise PacketError("FMP analyst payload has fewer than the seven expected ticker groups.")

    report_paths = sorted(reports_dir.glob("etf-flow-report-*.md"))
    reports = [_parse_report_summary(path) for path in report_paths]
    if not reports:
        raise PacketError(f"No daily ETF Flow reports found under {reports_dir}")
    latest_report = max(reports, key=lambda item: item["report_date"])
    coverage = _coverage_window(latest_report["report_date"], [item["report_date"] for item in reports if item["report_date"] >= (date.fromisoformat(latest_report["report_date"]) - timedelta(days=6)).isoformat()])
    category_rollups = _build_category_rollups(latest, quotes)
    flow_sessions = _build_flow_sessions([row for row in rows if isinstance(row, dict)])
    anchors = ("SPY", "QQQ", "IWM", "TQQQ", "SQQQ", "TLT", "HYG", "LQD", "GLD")

    return {
        "schema_version": "1.0",
        "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "analysis_owner": {
            "project": "/home/zooh/Documents/GitHub/quant",
            "skill": "quant-market-analysis",
            "runtime": "DGX Hermes",
        },
        "source_status": [
            {"source": "Massive ETF Flow", "status": "confirmed", "records": len(rows), "latest_flow_date": max(_date_key(row) for row in rows if _date_key(row))},
            {"source": "FMP Quotes", "status": "confirmed", "records": len(quotes)},
            {"source": "Barchart QQQ Options", "status": "confirmed", "metrics": len(metrics)},
            {"source": "FMP Analyst", "status": "confirmed", "ticker_groups": len(analysts)},
        ],
        "source_descriptions": {
            "Massive ETF Flow": "Confirmed end-of-day ETF fund-flow observations; it is lagged relative to quotes.",
            "FMP Quotes": "Price, change, volume, and average-volume snapshot used to confirm or challenge the flow reading.",
            "Barchart QQQ Options": "QQQ implied volatility, put/call, gamma pivot, and option-wall context.",
            "FMP Analyst": "Consensus target context only; it is not a flow or timing signal.",
        },
        "source_files": [{"name": name, "path": str(path), "sha256": _sha256(path)} for name, path in source_paths.items()],
        "category_rollups": category_rollups,
        "market_anchors": [
            {
                "symbol": symbol,
                "flow": _float(latest[symbol].get("fund_flow")),
                "flow_date": _date_key(latest[symbol]),
                "price": _float(quotes[symbol].get("price")),
                "change_percent": _float(quotes[symbol].get("changesPercentage")),
                "volume_ratio": _float(quotes[symbol].get("volume")) / _float(quotes[symbol].get("avgVolume")) if _float(quotes[symbol].get("avgVolume")) else None,
            }
            for symbol in anchors
        ],
        "options": {"symbol": options.get("symbol"), "date": options.get("date"), "crawled_at": options.get("crawled_at"), "metrics": metrics},
        "analyst_context": analysts,
        "flow_sessions": flow_sessions,
        "report_continuity": {"coverage": coverage, "reports": [item for item in reports if item["report_date"] >= coverage["window_start"]]},
        "prompt_matrix": list(PROMPT_MATRIX),
        "evidence_catalog": _build_evidence_catalog(
            category_rollups,
            [
                {
                    "symbol": symbol,
                    "flow": _float(latest[symbol].get("fund_flow")),
                    "flow_date": _date_key(latest[symbol]),
                    "price": _float(quotes[symbol].get("price")),
                    "change_percent": _float(quotes[symbol].get("changesPercentage")),
                    "volume_ratio": _float(quotes[symbol].get("volume")) / _float(quotes[symbol].get("avgVolume")) if _float(quotes[symbol].get("avgVolume")) else None,
                }
                for symbol in anchors
            ],
            {"metrics": metrics},
            flow_sessions,
            coverage,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = build_packet(args.input_dir, args.reports_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "schema_version": packet["schema_version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
