#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/run-market-report.sh [output-directory]
# HORIZON and LOOKBACK can be overridden via env vars.

OUT_DIR=${1:-market_report}
HORIZON=${HORIZON:-5}
LOOKBACK=${LOOKBACK:-1260}

mkdir -p "$OUT_DIR"

JSON_PATH="$OUT_DIR/report.json"
MD_PATH="$OUT_DIR/report.md"
SUMMARY_PATH="$OUT_DIR/summary.txt"

python3 report_engine/market_analysis/market_report.py \
  --json \
  --horizon "$HORIZON" \
  --lookback "$LOOKBACK" \
  > "$JSON_PATH"

python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
json_path = out_dir / "report.json"
data = json.loads(json_path.read_text(encoding="utf-8"))

markdown = data.get("markdown", "").strip()
md_path = out_dir / "report.md"
md_path.write_text(markdown + "\n", encoding="utf-8")

prob = data.get("prob", {}).get("p_up")
drivers = data.get("drivers", [])

summary_lines = []
if prob is not None:
    summary_lines.append(f"Upward probability: {prob * 100:.2f}%")
if drivers:
    top = ", ".join(f"{name}:{value:+.3f}" for name, value in drivers[:5])
    summary_lines.append(f"Top drivers: {top}")

summary = "\n".join(summary_lines)
(out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")

print(summary)
PY

echo "📄 Market report JSON: $JSON_PATH"
echo "📝 Markdown preview: $MD_PATH"
echo "ℹ️ Summary: $SUMMARY_PATH"
