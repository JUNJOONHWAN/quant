#!/usr/bin/env python3
"""Publish a validated quant ETF Flow snapshot and analysis to the dashboard API."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


class PublishError(RuntimeError):
    """Raised when the dashboard refuses a quant analysis artifact."""


KST = ZoneInfo("Asia/Seoul")


def read_json(path: Path):
    if not path.is_file():
        raise PublishError(f"Required dashboard artifact is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PublishError(f"Required dashboard artifact is invalid JSON: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--site-access-token-file", type=Path)
    args = parser.parse_args()
    if not args.analysis.is_file():
        raise PublishError(f"Analysis artifact is missing: {args.analysis}")
    if not args.token_file.is_file():
        raise PublishError(f"Dashboard publish token file is missing: {args.token_file}")
    analysis = read_json(args.analysis)
    market = {
        "flows": read_json(args.snapshot_dir / "massive_flows.json"),
        "quotes": read_json(args.snapshot_dir / "fmp_quotes.json"),
        "analysts": read_json(args.snapshot_dir / "analyst_estimates.json"),
        "options": read_json(args.snapshot_dir / "barchart_qqq.json"),
    }
    document = {
        "schema_version": "1.0",
        "published_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "report_date": args.report_date,
        "market": market,
        "analysis": analysis,
    }
    payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise PublishError("Dashboard publish token is empty.")
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "content-length": str(len(payload)),
        "user-agent": "quant-etf-flow-publisher/1.0",
    }
    if args.site_access_token_file:
        if not args.site_access_token_file.is_file():
            raise PublishError(f"Site access token file is missing: {args.site_access_token_file}")
        site_access_token = args.site_access_token_file.read_text(encoding="utf-8").strip()
        if not site_access_token:
            raise PublishError("Site access token is empty.")
        headers["oai-sites-authorization"] = f"Bearer {site_access_token}"
    request = urllib.request.Request(
        args.endpoint,
        data=payload,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            body = json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise PublishError(f"Dashboard publish request failed: {exc}") from exc
    if response.status != 200 or body.get("status") != "ok":
        raise PublishError(f"Dashboard publish was rejected: {body}")
    print(json.dumps({
        "status": "ok",
        "endpoint": args.endpoint,
        "published_at_kst": body.get("published_at_kst"),
        "generated_at_kst": body.get("generated_at_kst"),
        "report_date": body.get("report_date"),
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublishError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        raise SystemExit(1)
