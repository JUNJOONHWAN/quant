#!/usr/bin/env python3
"""Build a hash-bound US stock/ETF symbol list from an FMP universe JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


US_DELISTED_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "OTC", "PNK"}
CANONICAL_US_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9-]{0,63}$")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".symbols-", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build(input_path: Path, output_path: Path) -> dict:
    symbols = []
    excluded = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("analysis_eligible"):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError("missing symbol at line {}".format(line_number))
        sources = set(row.get("sources") or [])
        exchange = str(row.get("exchange") or "").strip().upper()
        if sources == {"symbol_change"} and not CANONICAL_US_SYMBOL.fullmatch(symbol):
            canonical_symbols = sorted(
                {
                    str(event.get("new_symbol") or "").strip().upper()
                    for event in row.get("symbol_change_events") or []
                    if isinstance(event, dict)
                    and CANONICAL_US_SYMBOL.fullmatch(
                        str(event.get("new_symbol") or "").strip().upper()
                    )
                }
            )
            excluded.append(
                {
                    "symbol": symbol,
                    "canonical_symbols": canonical_symbols,
                    "reason": "non_canonical_symbol_change_alias",
                }
            )
            continue
        if sources == {"delisted_companies"} and exchange not in US_DELISTED_EXCHANGES:
            excluded.append(
                {
                    "symbol": symbol,
                    "exchange": exchange or None,
                    "reason": "non_us_delisted_reference",
                }
            )
            continue
        symbols.append(symbol)
    symbols = sorted(set(symbols))
    payload = ("\n".join(symbols) + "\n").encode("utf-8")
    input_payload = input_path.read_bytes()
    manifest = {
        "schema_version": "quant.fmp_us_equity_etf_symbols.v1",
        "input_jsonl": str(input_path.resolve()),
        "input_jsonl_sha256": hashlib.sha256(input_payload).hexdigest(),
        "output_symbols": str(output_path.resolve()),
        "output_symbols_sha256": hashlib.sha256(payload).hexdigest(),
        "symbol_count": len(symbols),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "scope": "US stocks, ETFs, and US OTC; mutual funds excluded",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    _atomic_write(output_path, payload)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {"ok": True, **manifest, "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-symbols", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input_jsonl.expanduser(), args.output_symbols.expanduser())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
