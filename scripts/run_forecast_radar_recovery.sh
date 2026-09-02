#!/usr/bin/env bash
# Regenerate the Forecast RADAR shadow artifact only when the sealed Oracle
# source fingerprint is not already represented by the current run.
set -euo pipefail

readonly QUANT_DIR='/home/zooh/Documents/GitHub/quant'
readonly FORECAST_ROOT='/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/forecast_radar'
readonly PYTHON_BIN="${FORECAST_RADAR_PYTHON:-${FORECAST_ROOT}/runtime/venv/bin/python}"
readonly STATE_PATH="${FORECAST_ROOT}/status/forecast_radar_recovery.json"
readonly COUNTER_PATH="${VECTORMAN_COUNTER_PATH:-${QUANT_DIR}/counter.txt}"

if [[ "$PWD" != "$QUANT_DIR" ]]; then
  cd "$QUANT_DIR"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' "Forecast RADAR runtime is unavailable: $PYTHON_BIN" >&2
  exit 1
fi

readonly RUN_LOG="$(mktemp /tmp/vectorman-forecast-radar-recovery.XXXXXX)"
trap 'rm -f "$RUN_LOG"' EXIT
if ! "$PYTHON_BIN" -m workflows.forecast_radar.cli scheduled-daily >"$RUN_LOG" 2>&1; then
  printf '%s\n' 'Forecast RADAR scheduled recovery failed; inspect the redacted state receipt' >&2
  exit 1
fi

"$PYTHON_BIN" - "$RUN_LOG" "$STATE_PATH" <<'PY'
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

run_payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
state_path = Path(sys.argv[2])
batch = run_payload.get("batch") if isinstance(run_payload.get("batch"), dict) else {}
latest = run_payload.get("latest") if isinstance(run_payload.get("latest"), dict) else {}
allowed = {"PASS_SHADOW_RUN", "NOOP_ALREADY_CURRENT"}
quality_gate = str(batch.get("quality_gate") or "")
target = str(latest.get("oracle_target_as_of_date") or "")[:10]
price_date = str(latest.get("price_date") or "")[:10]
fingerprint = str(latest.get("oracle_source_fingerprint_sha256") or "")
seal = str(latest.get("oracle_snapshot_seal_sha256") or "")
data_quality_status = str(latest.get("oracle_data_quality_status") or "")
data_quality_fingerprint = str(latest.get("oracle_data_quality_sha256") or "")
if quality_gate not in allowed:
    raise SystemExit(f"Forecast RADAR quality gate failed: {quality_gate or 'missing'}")
if not target or price_date != target:
    raise SystemExit(f"Forecast RADAR price/Oracle mismatch: price={price_date or 'missing'} oracle={target or 'missing'}")
if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
    raise SystemExit("Forecast RADAR Oracle source fingerprint is missing")
if not re.fullmatch(r"[0-9a-f]{64}", seal):
    raise SystemExit("Forecast RADAR Oracle snapshot seal is missing")
if data_quality_status != "PASS":
    raise SystemExit("Forecast RADAR source-data quality gate did not pass")
if not re.fullmatch(r"[0-9a-f]{64}", data_quality_fingerprint):
    raise SystemExit("Forecast RADAR source-data quality fingerprint is missing")
state = {
    "schema": "vectorman.forecast-radar-recovery/v1",
    "status": "complete",
    "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
    "source_owner": "market_structure_oracle_single_writer",
    "oracle": {
        "target_as_of_date": target,
        "source_fingerprint_sha256": fingerprint,
        "snapshot_seal_sha256": seal,
        "data_quality_status": data_quality_status,
        "data_quality_sha256": data_quality_fingerprint,
    },
    "forecast": {
        "run_id": latest.get("run_id"),
        "signal_date": latest.get("signal_date"),
        "price_date": price_date,
        "quality_gate": quality_gate,
        "activation_status": latest.get("activation_status"),
    },
    "recovery_action": "already_current" if quality_gate == "NOOP_ALREADY_CURRENT" else "regenerated",
}
state_path.parent.mkdir(parents=True, exist_ok=True)
temporary = state_path.with_name(state_path.name + ".tmp")
temporary.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, state_path)
print(json.dumps(state, ensure_ascii=False, sort_keys=True))
PY

printf 'completed=1\n' >"$COUNTER_PATH"
