#!/usr/bin/env python3
"""Append a review outcome that can drive the next ETF Flow strategy selection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")
WORKFLOW = Path("/home/zooh/Documents/GitHub/quant/workflows/etf_flow")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejected-strategy", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--next-strategy", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reviewer", default="hermes_or_operator")
    args = parser.parse_args()

    registry = json.loads((WORKFLOW / "strategy_registry.json").read_text(encoding="utf-8"))
    known = {item["id"] for item in registry["strategies"]}
    if args.rejected_strategy not in known or args.next_strategy not in known:
        raise SystemExit("unknown strategy in feedback record")
    if not args.artifact.is_file():
        raise SystemExit(f"artifact is missing: {args.artifact}")

    log_path = WORKFLOW / registry["feedback_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "reviewer": args.reviewer,
        "rejected_strategy": args.rejected_strategy,
        "reason": args.reason,
        "next_strategy": args.next_strategy,
        "artifact": str(args.artifact),
        "status": "pending_alternative_run",
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
