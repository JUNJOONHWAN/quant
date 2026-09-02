"""Export resumable monthly v3 analysis-packet shards and a hash manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from quant_dataset.config import DEFAULT_SECRETS_PATH, load_credentials, resolve_data_root
from quant_dataset.pipeline import DatasetPipeline
from training.quant_llm.build_sft_dataset import load_trading_sessions


DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/packets/v3_monthly"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".packet-manifest-", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _monthly_ranges(
    sessions: Sequence[str], start_date: str, end_date: str
) -> List[Tuple[str, str]]:
    selected = [session for session in sessions if start_date <= session <= end_date]
    grouped: "OrderedDict[str, List[str]]" = OrderedDict()
    for session in selected:
        grouped.setdefault(session[:7], []).append(session)
    return [(rows[0], rows[-1]) for rows in grouped.values()]


def _normalize_symbols(value: Optional[str]) -> List[str]:
    return sorted(
        {
            item.strip().upper()
            for item in (value or "").split(",")
            if item.strip()
        }
    )


def _normalize_statuses(value: str) -> List[str]:
    statuses = sorted({item.strip() for item in value.split(",") if item.strip()})
    if not statuses:
        raise ValueError("at least one quality status is required")
    return statuses


def export_monthly_shards(
    *,
    data_root: Path,
    output_root: Path,
    secrets_file: Path,
    start_date: str,
    end_date: str,
    symbols: Sequence[str],
    lookback_days: int,
    quality_statuses: Sequence[str],
    timeout_seconds: float,
    retries: int,
    replace: bool,
) -> dict:
    data = Path(data_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    database = data / "normalized" / "daily_observations.sqlite3"
    sessions = load_trading_sessions(database)
    ranges = _monthly_ranges(sessions, start_date, end_date)
    if not ranges:
        raise ValueError("requested range contains no U.S. sessions")
    contract = {
        "packet_schema": "quant.analysis_packet.v3",
        "data_root": str(data),
        "database": str(database),
        "database_size_bytes_at_start": database.stat().st_size,
        "from": start_date,
        "to": end_date,
        "symbols": list(symbols),
        "symbol_scope": "explicit" if symbols else "all quality-eligible symbol-session pairs",
        "lookback_days": lookback_days,
        "quality_statuses": list(quality_statuses),
        "sharding": "one shard per calendar month, bounded to observed U.S. sessions",
        "ordering": "trade_date_then_symbol",
    }
    contract_sha = hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()
    manifest_path = output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    existing = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") != contract_sha and not replace:
            raise ValueError("existing packet manifest contract differs; use a new output root")
    manifest = {
        "schema_version": "quant.packet_shard_manifest.v1",
        "contract": contract,
        "contract_sha256": contract_sha,
        "complete": False,
        "completed_shards": 0,
        "total_shards": len(ranges),
        "total_packets": 0,
        "total_bytes": 0,
        "shards": [],
    }
    previous_by_name: Dict[str, dict] = {
        str(row.get("filename")): row for row in (existing or {}).get("shards", [])
    }
    credentials = load_credentials(secrets_path=Path(secrets_file).expanduser())
    pipeline = DatasetPipeline(
        data_root=data,
        credentials=credentials,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )

    for shard_start, shard_end in ranges:
        filename = "analysis_packets_{}_{}.v3.jsonl".format(
            shard_start.replace("-", ""), shard_end.replace("-", "")
        )
        path = output / filename
        previous = previous_by_name.get(filename)
        reusable = bool(
            previous
            and path.is_file()
            and path.stat().st_size == int(previous.get("bytes") or -1)
            and _sha256_file(path) == previous.get("sha256")
        )
        if reusable and not replace:
            result = {
                "output": str(path),
                "packets": int(previous["packets"]),
                "bytes": int(previous["bytes"]),
                "sha256": str(previous["sha256"]),
                "resumed_verified_existing": True,
            }
        else:
            if path.exists() and not replace:
                raise ValueError(
                    "unverified existing shard {}; pass --replace or use a new output root".format(
                        path
                    )
                )
            result = pipeline.export_packets(
                shard_start,
                shard_end,
                path,
                symbols=list(symbols) or None,
                lookback_days=lookback_days,
                quality_statuses=quality_statuses,
            )
            result["resumed_verified_existing"] = False
        shard = {
            "filename": filename,
            "from": shard_start,
            "to": shard_end,
            "packets": int(result["packets"]),
            "bytes": int(result["bytes"]),
            "sha256": str(result["sha256"]),
            "resumed_verified_existing": bool(result["resumed_verified_existing"]),
        }
        manifest["shards"].append(shard)
        manifest["completed_shards"] = len(manifest["shards"])
        manifest["total_packets"] += shard["packets"]
        manifest["total_bytes"] += shard["bytes"]
        _write_atomic(manifest_path, manifest)

    manifest["complete"] = manifest["completed_shards"] == manifest["total_shards"]
    _write_atomic(manifest_path, manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--secrets-file", type=Path, default=DEFAULT_SECRETS_PATH)
    parser.add_argument("--from", dest="start_date", required=True)
    parser.add_argument("--to", dest="end_date", required=True)
    parser.add_argument("--symbols", help="optional comma-separated readiness subset")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=21,
        help="price sessions per packet; ETF Flow observations are capped at 20",
    )
    parser.add_argument("--quality-statuses", default="pass,warn,single_source")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    if args.start_date > args.end_date:
        parser.error("--from must be <= --to")
    if args.lookback_days < 1:
        parser.error("--lookback-days must be positive")
    result = export_monthly_shards(
        data_root=resolve_data_root(str(args.data_root) if args.data_root else None),
        output_root=args.output_root,
        secrets_file=args.secrets_file,
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=_normalize_symbols(args.symbols),
        lookback_days=args.lookback_days,
        quality_statuses=_normalize_statuses(args.quality_statuses),
        timeout_seconds=args.timeout,
        retries=args.retries,
        replace=args.replace,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
