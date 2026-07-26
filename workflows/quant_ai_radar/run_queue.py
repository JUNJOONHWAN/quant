"""Restartable SQLite ledger for full-universe Quant AI inference."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .universe import Candidate


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RadarQueue:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=120)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS run_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS items (
                    symbol TEXT PRIMARY KEY,
                    proxy_task_type TEXT NOT NULL,
                    actual_task_type TEXT,
                    quality_status TEXT NOT NULL,
                    relation_types_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    packet_id TEXT,
                    eligibility_json TEXT,
                    prompt_sha256 TEXT,
                    response_sha256 TEXT,
                    result_json TEXT,
                    exclusion_reason TEXT,
                    error TEXT,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_radar_queue_status
                    ON items(status, symbol);
                """
            )
            connection.execute(
                "UPDATE items SET status='pending', error='interrupted_while_running' "
                "WHERE status='running'"
            )

    def bind_metadata(self, values: Mapping[str, Any]) -> None:
        with self.connect() as connection:
            existing = {
                str(row["key"]): json.loads(str(row["value_json"]))
                for row in connection.execute("SELECT * FROM run_metadata")
            }
            for key, value in values.items():
                if key in existing and existing[key] != value:
                    raise ValueError(
                        f"run queue metadata mismatch for {key}: {existing[key]!r} != {value!r}"
                    )
                connection.execute(
                    "INSERT OR REPLACE INTO run_metadata(key,value_json) VALUES(?,?)",
                    (key, json.dumps(value, sort_keys=True, ensure_ascii=False)),
                )

    def seed(self, candidates: Iterable[Candidate]) -> int:
        rows = list(candidates)
        with self.connect() as connection:
            for item in rows:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO items(
                        symbol,proxy_task_type,quality_status,relation_types_json,
                        status,updated_at_utc
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        item.symbol,
                        item.proxy_task_type,
                        item.quality_status,
                        json.dumps(item.relation_types, sort_keys=True),
                        "pending",
                        utc_now(),
                    ),
                )
        return len(rows)

    def pending(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM items WHERE status IN ('pending','error') ORDER BY symbol"
                )
            ]

    def mark_running(self, symbol: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE items SET status='running',attempt_count=attempt_count+1,
                    error=NULL,updated_at_utc=? WHERE symbol=?
                """,
                (utc_now(), symbol),
            )

    def mark_excluded(
        self, symbol: str, eligibility: Mapping[str, Any], reason: str
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE items SET status='excluded',eligibility_json=?,
                    exclusion_reason=?,updated_at_utc=? WHERE symbol=?
                """,
                (
                    json.dumps(eligibility, sort_keys=True, ensure_ascii=False),
                    reason,
                    utc_now(),
                    symbol,
                ),
            )

    def mark_done(
        self,
        *,
        symbol: str,
        actual_task_type: str,
        packet_id: str,
        eligibility: Mapping[str, Any],
        prompt_sha256: str,
        response_sha256: str,
        result: Mapping[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE items SET status='done',actual_task_type=?,packet_id=?,
                    eligibility_json=?,prompt_sha256=?,response_sha256=?,result_json=?,
                    exclusion_reason=NULL,error=NULL,updated_at_utc=? WHERE symbol=?
                """,
                (
                    actual_task_type,
                    packet_id,
                    json.dumps(eligibility, sort_keys=True, ensure_ascii=False),
                    prompt_sha256,
                    response_sha256,
                    json.dumps(result, sort_keys=True, ensure_ascii=False),
                    utc_now(),
                    symbol,
                ),
            )

    def mark_error(self, symbol: str, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE items SET status='error',error=?,updated_at_utc=? WHERE symbol=?",
                (error[:4000], utc_now(), symbol),
            )

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    "SELECT status,COUNT(*) count FROM items GROUP BY status ORDER BY status"
                )
            }

    def done_results(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT symbol,actual_task_type,result_json FROM items "
                "WHERE status='done' ORDER BY symbol"
            ).fetchall()
        return [
            {
                "symbol": str(row["symbol"]),
                "task_type": str(row["actual_task_type"]),
                "judgement": json.loads(str(row["result_json"])),
            }
            for row in rows
        ]

    def exclusions(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT exclusion_reason,COUNT(*) count FROM items "
                "WHERE status='excluded' GROUP BY exclusion_reason"
            ).fetchall()
        return {str(row["exclusion_reason"]): int(row["count"]) for row in rows}
