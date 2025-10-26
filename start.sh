#!/usr/bin/env bash
# Orchestrate data refresh + verification from the repository root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

log() {
  printf '==> %s\n' "$*"
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_tool() {
  local tool=$1
  command -v "$tool" >/dev/null 2>&1 || die "'$tool' 명령을 찾을 수 없습니다. 설치 후 다시 실행하세요."
}

require_file() {
  local path=$1
  [[ -f "$path" ]] || die "필수 파일이 없습니다: $path"
}

require_tool python3
require_tool npm
require_file scripts/fetch_long_history.py
require_file scripts/generate-data.js

if [[ -z "${ALPHAVANTAGE_API_KEY:-}" ]]; then
  die "환경변수 ALPHAVANTAGE_API_KEY 가 설정되어 있지 않습니다. Alpha Vantage 키를 내보낸 뒤 실행하세요."
fi

log "1/3 장기 시계열(2017-) 다운로드"
python3 scripts/fetch_long_history.py --force

log "2/3 Alpha Vantage 기반 지표 생성"
npm run generate:data

log "3/3 단위 테스트 실행"
npm test

log "요약"
python3 - "$ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
history_path = root / 'static_site' / 'data' / 'historical_prices.json'
precomputed_path = root / 'static_site' / 'data' / 'precomputed.json'

def describe_history():
    if not history_path.exists():
        print(f" - 장기 캐시 파일 없음: {history_path}")
        return
    data = json.loads(history_path.read_text())
    assets = data.get('assets') or []
    sample = assets[0] if assets else {}
    rows = len(sample.get('dates') or [])
    print(
        " - Long history: {start} → {end} / {rows} 영업일 (예: {symbol})".format(
            start=data.get('startDate', '?'),
            end=data.get('endDate', '?'),
            rows=rows,
            symbol=sample.get('symbol', 'n/a'),
        )
    )

def describe_precomputed():
    if not precomputed_path.exists():
        print(f" - 사전계산 파일 없음: {precomputed_path}")
        return
    data = json.loads(precomputed_path.read_text())
    dates = data.get('analysisDates') or []
    label = f"{dates[0]} → {dates[-1]}" if dates else 'n/a'
    print(
        " - precomputed.json: {label} / 샘플 {count}일 / 생성시각 {generated}".format(
            label=label,
            count=len(dates),
            generated=data.get('generatedAt', 'n/a'),
        )
    )

describe_history()
describe_precomputed()
PY

log "완료: static_site/data/ 에 최신 JSON이 준비되었습니다."
