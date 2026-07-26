#!/usr/bin/env python3
"""Report hard completion gates for the long-running quant dataset backfills."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


DEFAULT_DATA_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET"
)
DEFAULT_UNIVERSE_STEM = "fmp_us_all_20260714"
DEFAULT_EOD_SYMBOLS_STEM = "fmp_us_equity_etf_20260714"


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def _load_jsonl_etfs(path: Path) -> Set[str]:
    result = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("is_etf") is True:
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol:
                    result.add(symbol)
    return result


def _load_symbols(path: Path) -> Set[str]:
    return {
        line.strip().upper()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }


def _status_counts(
    connection: sqlite3.Connection,
    job_id: str,
    item_prefix: Optional[str] = None,
    invert_prefix: bool = False,
) -> Dict[str, int]:
    sql = "SELECT status, COUNT(*) count FROM checkpoints WHERE job_id=?"
    parameters: List[Any] = [job_id]
    if item_prefix is not None:
        sql += " AND item_key {}LIKE ?".format("NOT " if invert_prefix else "")
        parameters.append(item_prefix + "%")
    sql += " GROUP BY status"
    return {
        str(row["status"]): int(row["count"])
        for row in connection.execute(sql, parameters)
    }


def _service_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def _job_rows(connection: sqlite3.Connection, job_type: str) -> List[sqlite3.Row]:
    return list(
        connection.execute(
            "SELECT * FROM jobs WHERE job_type=? ORDER BY updated_at_utc DESC",
            (job_type,),
        )
    )


def _matching_eod_job(
    connection: sqlite3.Connection, universe_sha256: str
) -> Optional[Tuple[sqlite3.Row, dict]]:
    for row in _job_rows(connection, "backfill"):
        contract = json.loads(row["contract_json"])
        if (
            contract.get("from") == "2017-01-01"
            and contract.get("to") == "2026-07-14"
            and contract.get("sources") == ["fmp"]
            and contract.get("symbol_universe", {}).get("sha256")
            == universe_sha256
        ):
            return row, contract
    return None


def _constituent_jobs(
    connection: sqlite3.Connection, universe_sha256: str
) -> List[Tuple[sqlite3.Row, dict]]:
    result = []
    for row in _job_rows(connection, "backfill_fmp_etf_constituents"):
        contract = json.loads(row["contract_json"])
        if (
            contract.get("from") == "2017-01-01"
            and contract.get("to") == "2026-07-14"
            and contract.get("universe_contract", {}).get("sha256")
            == universe_sha256
        ):
            result.append((row, contract))
    return result


def build_status(
    data_root: Path, universe_stem: str, eod_symbols_stem: str
) -> dict:
    state_root = data_root / "state" / "universe"
    symbols_path = state_root / (eod_symbols_stem + ".symbols.txt")
    universe_path = state_root / (universe_stem + ".jsonl")
    manifest_path = state_root / (universe_stem + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    symbols_sha256 = __import__("hashlib").sha256(symbols_path.read_bytes()).hexdigest()
    jsonl_sha256 = str(manifest["jsonl_sha256"])
    target_symbols = _load_symbols(symbols_path)
    target_etfs = _load_jsonl_etfs(universe_path)

    database_path = data_root / "normalized" / "daily_observations.sqlite3"
    with _connect(database_path) as connection:
        eod_match = _matching_eod_job(connection, symbols_sha256)
        if eod_match:
            eod_row, _ = eod_match
            eod_statuses = _status_counts(connection, str(eod_row["job_id"]))
        else:
            eod_row = None
            eod_statuses = {}
        fmp_observation = connection.execute(
            """
            SELECT COUNT(*) rows, COUNT(DISTINCT symbol) symbols,
                   MIN(trade_date) min_date, MAX(trade_date) max_date
            FROM daily_observations WHERE source='fmp'
            """
        ).fetchone()
        eod_done = int(eod_statuses.get("done", 0))
        eod_complete = (
            eod_row is not None
            and eod_done == len(target_symbols)
            and set(eod_statuses).issubset({"done"})
        )

        constituent_matches = _constituent_jobs(connection, jsonl_sha256)
        covered_etfs: Set[str] = set()
        duplicate_etfs: Set[str] = set()
        date_statuses: Counter[str] = Counter()
        snapshot_statuses: Counter[str] = Counter()
        constituent_job_ids = []
        for row, contract in constituent_matches:
            job_id = str(row["job_id"])
            constituent_job_ids.append(job_id)
            tickers = {str(item).upper() for item in contract.get("tickers", [])}
            duplicate_etfs.update(covered_etfs.intersection(tickers))
            covered_etfs.update(tickers)
            date_statuses.update(_status_counts(connection, job_id, "dates:"))
            snapshot_statuses.update(
                _status_counts(connection, job_id, "dates:", invert_prefix=True)
            )

        available_pairs = {
            (str(row["etf_ticker"]), str(row["effective_date"]))
            for row in connection.execute(
                """
                SELECT etf_ticker, effective_date
                FROM etf_constituent_available_dates
                WHERE provider='fmp' AND effective_date BETWEEN '2017-01-01' AND '2026-07-14'
                """
            )
            if str(row["etf_ticker"]) in target_etfs
        }
        snapshot_pairs = {
            (str(row["etf_ticker"]), str(row["effective_date"]))
            for row in connection.execute(
                """
                SELECT etf_ticker, effective_date
                FROM etf_constituent_snapshots
                WHERE provider='fmp' AND effective_date BETWEEN '2017-01-01' AND '2026-07-14'
                """
            )
            if str(row["etf_ticker"]) in target_etfs
        }
        constituent_counts = connection.execute(
            """
            SELECT COUNT(*) observations, COUNT(DISTINCT etf_ticker) etfs
            FROM etf_constituent_observations WHERE provider='fmp'
            """
        ).fetchone()
        missing_etfs = sorted(target_etfs - covered_etfs)
        missing_snapshots = len(available_pairs - snapshot_pairs)
        constituents_complete = (
            not missing_etfs
            and not duplicate_etfs
            and int(date_statuses.get("done", 0)) == len(target_etfs)
            and set(date_statuses).issubset({"done"})
            and set(snapshot_statuses).issubset({"done"})
            and missing_snapshots == 0
        )

        flow_runs = []
        for row in connection.execute(
            "SELECT * FROM etf_flow_runs ORDER BY updated_at_utc DESC"
        ):
            contract = json.loads(row["contract_json"])
            if (
                row["job_type"] == "backfill_etf_flows"
                and contract.get("processed_date_gte") == "2017-01-01"
                and contract.get("processed_date_lte") == "2026-07-14"
            ):
                flow_runs.append((row, contract))
        flow_row = flow_runs[0][0] if flow_runs else None
        flow_complete = bool(
            flow_row and str(flow_row["status"]) in {"complete", "completed"}
        )

    services = {
        "fmp_eod": _service_state("quant-fmp-eod-backfill.service"),
        "massive_etf_flow": _service_state("quant-massive-etf-flow.service"),
        "fmp_etf_constituent_shards": {
            str(index): _service_state(
                "quant-fmp-etf-constituents@{}.service".format(index)
            )
            for index in range(4)
        },
    }
    return {
        "ok": True,
        "data_root": str(data_root),
        "universe": {
            "symbols": len(target_symbols),
            "etfs": len(target_etfs),
            "symbols_sha256": symbols_sha256,
            "jsonl_sha256": jsonl_sha256,
            "delisted_history_warning": (
                "FMP delisted-companies pagination stopped after page 0 with HTTP 402; "
                "inactive symbols from stock-list remain included"
            ),
        },
        "fmp_eod": {
            "complete": eod_complete,
            "job_id": str(eod_row["job_id"]) if eod_row else None,
            "checkpoint_statuses": eod_statuses,
            "target_symbols": len(target_symbols),
            "observed_symbols": int(fmp_observation["symbols"] or 0),
            "rows": int(fmp_observation["rows"] or 0),
            "min_date": fmp_observation["min_date"],
            "max_date": fmp_observation["max_date"],
        },
        "fmp_etf_constituents": {
            "complete": constituents_complete,
            "job_ids": constituent_job_ids,
            "target_etfs": len(target_etfs),
            "covered_etfs": len(covered_etfs),
            "missing_etfs": len(missing_etfs),
            "missing_etf_examples": missing_etfs[:20],
            "duplicate_etfs": len(duplicate_etfs),
            "date_checkpoint_statuses": dict(date_statuses),
            "snapshot_checkpoint_statuses": dict(snapshot_statuses),
            "expected_snapshots": len(available_pairs),
            "missing_snapshots": missing_snapshots,
            "observations": int(constituent_counts["observations"] or 0),
            "observed_etfs": int(constituent_counts["etfs"] or 0),
        },
        "massive_etf_flow": {
            "complete": flow_complete,
            "run_id": str(flow_row["run_id"]) if flow_row else None,
            "status": str(flow_row["status"]) if flow_row else None,
            "pages": int(flow_row["page_count"]) if flow_row else 0,
            "records": int(flow_row["record_count"]) if flow_row else 0,
            "last_error": flow_row["last_error"] if flow_row else None,
        },
        "services": services,
        "overall_complete": eod_complete and constituents_complete and flow_complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--universe-stem", default=DEFAULT_UNIVERSE_STEM)
    parser.add_argument("--eod-symbols-stem", default=DEFAULT_EOD_SYMBOLS_STEM)
    args = parser.parse_args()
    status = build_status(
        args.data_root.expanduser(), args.universe_stem, args.eod_symbols_stem
    )
    print(json.dumps(status, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if status["overall_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
