"""Generic, source-preserving FMP feature backfills for daily quant training."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from .providers import (
    ApiRequestError,
    CredentialError,
    HttpCaptureClient,
    normalize_symbol,
    validate_iso_date,
)
from .storage import Database, RawArtifact, canonical_json, utc_now


FMP_STABLE_BASE_URL = "https://financialmodelingprep.com"
FMP_TRAINING_SCHEMA_VERSION = "quant.fmp_training.v1"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _rows(document: Any) -> List[Mapping[str, Any]]:
    if isinstance(document, list):
        return [row for row in document if isinstance(row, Mapping)]
    if isinstance(document, Mapping):
        for key in ("data", "results", "historical"):
            value = document.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, Mapping)]
        return [document]
    return []


def _date_value(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if not value:
            continue
        text = str(value).strip()[:10]
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            continue
    return None


def _entity_key(row: Mapping[str, Any], fallback: str) -> str:
    for key in (
        "symbol",
        "ticker",
        "composite_ticker",
        "cik",
        "name",
        "companyName",
        "exchange",
        "sector",
        "industry",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return "{}:{}".format(key, str(value).strip())
    return fallback


def _variants(values: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    return values or ({},)


@dataclass(frozen=True)
class FmpWorkItem:
    endpoint_id: str
    path: str
    entity_key: str
    params: Mapping[str, Any]
    pagination: bool
    page_size: int
    max_pages: int

    @property
    def item_key(self) -> str:
        digest = _sha256_json(
            {
                "endpoint_id": self.endpoint_id,
                "entity_key": self.entity_key,
                "params": dict(self.params),
            }
        )[:20]
        return "{}:{}:{}".format(self.endpoint_id, self.entity_key[:80], digest)


class FmpTrainingStore:
    """Append-only generic FMP facts linked to immutable raw artifacts."""

    def __init__(self, database: Database):
        self.database = database
        self.initialize()

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS fmp_training_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint_id TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            row_sha256 TEXT NOT NULL,
            symbol TEXT,
            cik TEXT,
            period TEXT,
            event_date TEXT,
            available_date TEXT,
            availability_basis TEXT NOT NULL,
            row_json TEXT NOT NULL,
            raw_artifact_id INTEGER NOT NULL,
            capture_event_id INTEGER NOT NULL,
            source_row_index INTEGER NOT NULL,
            ingested_at_utc TEXT NOT NULL,
            UNIQUE(endpoint_id, entity_key, row_sha256, raw_artifact_id),
            FOREIGN KEY(raw_artifact_id) REFERENCES raw_artifacts(id),
            FOREIGN KEY(capture_event_id) REFERENCES capture_events(id)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_training_endpoint_entity
            ON fmp_training_facts(endpoint_id, entity_key);
        CREATE INDEX IF NOT EXISTS idx_fmp_training_symbol_available
            ON fmp_training_facts(symbol, available_date);
        CREATE INDEX IF NOT EXISTS idx_fmp_training_event_date
            ON fmp_training_facts(event_date);
        CREATE TABLE IF NOT EXISTS fmp_training_runs (
            job_id TEXT PRIMARY KEY,
            plan_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            generated_work_items INTEGER NOT NULL DEFAULT 0,
            done_items INTEGER NOT NULL DEFAULT 0,
            empty_items INTEGER NOT NULL DEFAULT 0,
            failed_items INTEGER NOT NULL DEFAULT 0,
            started_at_utc TEXT NOT NULL,
            completed_at_utc TEXT,
            FOREIGN KEY(job_id) REFERENCES jobs(job_id)
        );
        """
        with self.database.connect() as connection:
            connection.executescript(schema)

    def start_run(self, job_id: str, plan_sha256: str) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO fmp_training_runs (
                    job_id, plan_sha256, status, started_at_utc
                ) VALUES (?, ?, 'running', ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status='running', completed_at_utc=NULL
                """,
                (job_id, plan_sha256, utc_now()),
            )

    def finish_run(self, job_id: str, result: Mapping[str, Any]) -> None:
        status = "complete" if int(result.get("failed", 0)) == 0 else "incomplete"
        generated = sum(
            int(result.get(key, 0))
            for key in ("done", "empty", "not_entitled", "skipped", "failed")
        )
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE fmp_training_runs SET
                    status=?, generated_work_items=?, done_items=?, empty_items=?,
                    failed_items=?, completed_at_utc=?
                WHERE job_id=?
                """,
                (
                    status,
                    generated,
                    int(result.get("done", 0))
                    + int(result.get("skipped", 0))
                    - int(result.get("skipped_not_entitled", 0)),
                    int(result.get("empty", 0)),
                    int(result.get("failed", 0)),
                    utc_now(),
                    job_id,
                ),
            )

    def insert_rows(
        self,
        endpoint_id: str,
        requested_entity: str,
        rows: Sequence[Mapping[str, Any]],
        artifact: RawArtifact,
    ) -> int:
        if not rows:
            return 0
        now = utc_now()
        values = []
        for index, row in enumerate(rows):
            rendered = canonical_json(row)
            accepted = _date_value(
                row,
                (
                    "acceptedDate",
                    "acceptanceTime",
                    "acceptedDateTime",
                    "filingDate",
                    "fillingDate",
                    "publishedDate",
                    "publishedDateTime",
                ),
            )
            event_date = _date_value(
                row,
                (
                    "date",
                    "calendarDate",
                    "transactionDate",
                    "recordDate",
                    "paymentDate",
                    "fiscalDateEnding",
                    "reportedDate",
                    "year",
                ),
            )
            available_date = accepted or event_date or artifact.captured_at_utc[:10]
            values.append(
                (
                    endpoint_id,
                    _entity_key(row, requested_entity),
                    hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                    str(row.get("symbol") or row.get("ticker") or "").strip() or None,
                    str(row.get("cik") or "").strip() or None,
                    str(row.get("period") or row.get("quarter") or "").strip() or None,
                    event_date,
                    available_date,
                    "provider_acceptance_or_publish_date" if accepted else "event_date_or_capture_only",
                    rendered,
                    artifact.artifact_id,
                    artifact.capture_event_id,
                    index,
                    now,
                )
            )
        with self.database.connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO fmp_training_facts (
                    endpoint_id, entity_key, row_sha256, symbol, cik, period,
                    event_date, available_date, availability_basis, row_json,
                    raw_artifact_id, capture_event_id, source_row_index,
                    ingested_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            return int(connection.total_changes - before)

    def dimension_values(self, endpoint_id: str, keys: Sequence[str]) -> List[str]:
        values = set()
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT row_json FROM fmp_training_facts WHERE endpoint_id=?",
                (endpoint_id,),
            )
            for item in rows:
                try:
                    row = json.loads(item["row_json"])
                except (TypeError, ValueError):
                    continue
                if not isinstance(row, Mapping):
                    continue
                for key in keys:
                    value = row.get(key)
                    if value not in (None, ""):
                        values.add(str(value).strip())
                        break
        return sorted(value for value in values if value)

    def pagination_progress(
        self,
        endpoint_id: str,
        entity_key: str,
        base_params: Mapping[str, Any],
    ) -> tuple[int, set[str], int]:
        """Return first missing page, prior payload hashes, and existing fact rows."""

        comparable = {
            str(key): value
            for key, value in base_params.items()
            if str(key) not in {"page", "limit"}
        }
        with self.database.connect() as connection:
            try:
                artifacts = connection.execute(
                    """
                    SELECT payload_sha256, request_json
                    FROM raw_artifacts
                    WHERE source='fmp' AND dataset=?
                      AND json_extract(request_json, '$.logical_request.entity_key')=?
                    ORDER BY id
                    """,
                    (endpoint_id, entity_key),
                ).fetchall()
            except Exception:
                artifacts = connection.execute(
                    "SELECT payload_sha256, request_json FROM raw_artifacts "
                    "WHERE source='fmp' AND dataset=? ORDER BY id",
                    (endpoint_id,),
                ).fetchall()
            if entity_key == "scope=global":
                fact_row = connection.execute(
                    "SELECT COUNT(*) count FROM fmp_training_facts WHERE endpoint_id=?",
                    (endpoint_id,),
                ).fetchone()
            else:
                fact_row = connection.execute(
                    "SELECT COUNT(*) count FROM fmp_training_facts "
                    "WHERE endpoint_id=? AND entity_key=?",
                    (endpoint_id, entity_key),
                ).fetchone()
        pages: Dict[int, str] = {}
        for artifact in artifacts:
            try:
                request = json.loads(artifact["request_json"])
                logical = request.get("logical_request") or {}
                if str(logical.get("entity_key")) != entity_key:
                    continue
                params = dict(logical.get("params") or {})
                page = int(params.pop("page"))
                params.pop("limit", None)
            except (KeyError, TypeError, ValueError):
                continue
            if params == comparable:
                pages[page] = str(artifact["payload_sha256"])
        first_missing = 0
        while first_missing in pages:
            first_missing += 1
        return first_missing, set(pages.values()), int(fact_row["count"])

    def counts(self) -> dict:
        with self.database.connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) count FROM fmp_training_facts"
            ).fetchone()
            rows = connection.execute(
                "SELECT endpoint_id, COUNT(*) count FROM fmp_training_facts "
                "GROUP BY endpoint_id ORDER BY endpoint_id"
            ).fetchall()
        return {
            "facts": int(total["count"]),
            "by_endpoint": {str(row["endpoint_id"]): int(row["count"]) for row in rows},
        }


class FmpTrainingBackfill:
    """Execute a classified endpoint plan with resumable work-item checkpoints."""

    def __init__(
        self,
        database: Database,
        http: HttpCaptureClient,
        api_key: Optional[str],
    ):
        self.database = database
        self.http = http
        self.api_key = api_key
        self.store = FmpTrainingStore(database)
        self.blocked_dimensions: List[dict] = []

    @staticmethod
    def load_plan(path: Path) -> dict:
        plan = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(plan, dict) or not isinstance(plan.get("endpoints"), list):
            raise ValueError("invalid FMP training plan")
        return plan

    @staticmethod
    def _read_symbols(path: Path) -> List[str]:
        return sorted(
            {
                normalize_symbol(line)
                for line in Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            }
        )

    @staticmethod
    def _read_etfs(path: Path) -> List[str]:
        result = set()
        with Path(path).expanduser().open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, Mapping) and row.get("is_etf") is True and row.get("symbol"):
                    result.add(normalize_symbol(str(row["symbol"])))
        return sorted(result)

    def _dimension_values(
        self,
        spec: Mapping[str, Any],
        symbols: Sequence[str],
        etfs: Sequence[str],
    ) -> List[str]:
        mode = str(spec.get("mode") or "global")
        if mode == "global":
            return ["global"]
        if mode == "per_symbol":
            return list(symbols)
        if mode == "per_etf":
            return list(etfs)
        if mode == "per_value":
            return [str(value) for value in spec.get("values", [])]
        if mode == "per_discovered":
            return self.store.dimension_values(
                str(spec["source_endpoint_id"]),
                [str(key) for key in spec.get("source_keys", ["symbol"])],
            )
        raise ValueError("unsupported FMP training collection mode: {}".format(mode))

    def _work_items(
        self,
        plan: Mapping[str, Any],
        symbols: Sequence[str],
        etfs: Sequence[str],
        start_date: str,
        end_date: str,
        endpoint_ids: Optional[Sequence[str]],
    ) -> Iterator[FmpWorkItem]:
        self.blocked_dimensions = []
        selected = set(endpoint_ids or [])
        for endpoint in plan["endpoints"]:
            if endpoint.get("action") not in {"backfill", "snapshot"}:
                continue
            endpoint_id = str(endpoint["id"])
            if selected and endpoint_id not in selected:
                continue
            collection = dict(endpoint.get("collection") or {})
            dimension_param = collection.get("dimension_param")
            dimensions = self._dimension_values(collection, symbols, etfs)
            if not dimensions:
                self.blocked_dimensions.append(
                    {
                        "endpoint_id": endpoint_id,
                        "mode": collection.get("mode"),
                        "source_endpoint_id": collection.get("source_endpoint_id"),
                    }
                )
                continue
            variants = [dict(item) for item in _variants(collection.get("variants", []))]
            if collection.get("date_windows") == "year":
                windows = []
                for year in range(date.fromisoformat(start_date).year, date.fromisoformat(end_date).year + 1):
                    left = max(date(year, 1, 1), date.fromisoformat(start_date)).isoformat()
                    right = min(date(year, 12, 31), date.fromisoformat(end_date)).isoformat()
                    windows.append({"from": left, "to": right})
            else:
                windows = [{"from": start_date, "to": end_date}] if collection.get("include_date_range") else [{}]
            for dimension, variant, window in itertools.product(dimensions, variants, windows):
                params: Dict[str, Any] = dict(collection.get("static_params") or {})
                params.update(variant)
                params.update(window)
                if dimension_param:
                    if collection.get("batch_size"):
                        raise ValueError("batch dimensions are expanded separately")
                    params[str(dimension_param)] = dimension
                yield FmpWorkItem(
                    endpoint_id=endpoint_id,
                    path=str(endpoint["path"]),
                    entity_key="{}={}".format(dimension_param or "scope", dimension),
                    params=params,
                    pagination=bool(collection.get("pagination")),
                    page_size=max(1, int(collection.get("page_size") or 100)),
                    max_pages=max(1, int(collection.get("max_pages") or 10000)),
                )

    def _capture_item(self, item: FmpWorkItem) -> tuple[RawArtifact, int]:
        start_page = 0
        total = 0
        last_artifact: Optional[RawArtifact] = None
        seen_payloads = set()
        if item.pagination:
            start_page, seen_payloads, total = self.store.pagination_progress(
                item.endpoint_id, item.entity_key, item.params
            )
        for page in range(start_page, item.max_pages):
            params = dict(item.params)
            if item.pagination:
                params.update({"page": page, "limit": item.page_size})
            result = self.http.get_json(
                source="fmp",
                dataset=item.endpoint_id,
                partition_key="{}_page_{}".format(item.entity_key, page),
                url=FMP_STABLE_BASE_URL + item.path,
                params=params,
                headers={"apikey": self.api_key},
                logical_request={
                    "endpoint_contract": "fmp_training_generic_v1",
                    "endpoint_id": item.endpoint_id,
                    "entity_key": item.entity_key,
                    "page": page if item.pagination else None,
                    "params": params,
                },
            )
            last_artifact = result.artifact
            if result.artifact.payload_sha256 in seen_payloads:
                break
            seen_payloads.add(result.artifact.payload_sha256)
            rows = _rows(result.document)
            total += self.store.insert_rows(
                item.endpoint_id, item.entity_key, rows, result.artifact
            )
            if not item.pagination or len(rows) < item.page_size:
                break
        if last_artifact is None:
            raise RuntimeError("FMP work item produced no raw artifact")
        return last_artifact, total

    def backfill(
        self,
        plan_path: Path,
        symbols_path: Path,
        universe_jsonl: Path,
        start_date: str,
        end_date: str,
        endpoint_ids: Optional[Sequence[str]] = None,
        continue_on_error: bool = True,
    ) -> dict:
        if not self.api_key:
            raise CredentialError("FMP_API_KEY is not configured")
        start = validate_iso_date(start_date)
        end = validate_iso_date(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")
        plan_path = Path(plan_path).expanduser()
        symbols_path = Path(symbols_path).expanduser()
        universe_jsonl = Path(universe_jsonl).expanduser()
        plan = self.load_plan(plan_path)
        symbols = self._read_symbols(symbols_path)
        etfs = self._read_etfs(universe_jsonl)
        contract = {
            "schema_version": FMP_TRAINING_SCHEMA_VERSION,
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "symbols_sha256": hashlib.sha256(symbols_path.read_bytes()).hexdigest(),
            "universe_sha256": hashlib.sha256(universe_jsonl.read_bytes()).hexdigest(),
            "from": start,
            "to": end,
            "endpoint_ids": sorted(endpoint_ids or []),
        }
        job_id = "fmp-training:{}".format(_sha256_json(contract)[:16])
        self.database.register_job(
            job_id, "backfill_fmp_training", contract, FMP_TRAINING_SCHEMA_VERSION
        )
        self.store.start_run(job_id, contract["plan_sha256"])
        result = {
            "job_id": job_id,
            "done": 0,
            "empty": 0,
            "not_entitled": 0,
            "skipped": 0,
            "skipped_not_entitled": 0,
            "failed": 0,
            "observations": 0,
            "errors": [],
        }
        for item in self._work_items(plan, symbols, etfs, start, end, endpoint_ids):
            scope = {
                "endpoint_id": item.endpoint_id,
                "entity_key": item.entity_key,
                "params": dict(item.params),
                "pagination": item.pagination,
            }
            self.database.ensure_checkpoint(job_id, "fmp_training", item.item_key, scope)
            checkpoint_status = self.database.checkpoint_status(
                job_id, "fmp_training", item.item_key
            )
            if checkpoint_status in {"done", "not_entitled"}:
                result["skipped"] += 1
                if checkpoint_status == "not_entitled":
                    result["skipped_not_entitled"] += 1
                continue
            prior = self.database.completed_checkpoint_for_item(
                "fmp_training", item.item_key, exclude_job_id=job_id
            )
            if prior and prior["raw_artifact_id"] is not None:
                self.database.mark_checkpoint_done(
                    job_id,
                    "fmp_training",
                    item.item_key,
                    int(prior["raw_artifact_id"]),
                    int(prior["observation_count"] or 0),
                )
                result["skipped"] += 1
                continue
            self.database.mark_checkpoint_running(job_id, "fmp_training", item.item_key)
            try:
                artifact, count = self._capture_item(item)
                self.database.mark_checkpoint_done(
                    job_id, "fmp_training", item.item_key, artifact.artifact_id, count
                )
                result["done" if count else "empty"] += 1
                result["observations"] += count
            except ApiRequestError as error:
                if (
                    error.status_code in {402, 403}
                    and error.raw_artifact_id is not None
                ):
                    self.database.mark_checkpoint_not_entitled(
                        job_id,
                        "fmp_training",
                        item.item_key,
                        error.raw_artifact_id,
                        "{}: {}".format(type(error).__name__, str(error)),
                    )
                    result["not_entitled"] += 1
                    continue
                self.database.mark_checkpoint_failed(
                    job_id,
                    "fmp_training",
                    item.item_key,
                    "{}: {}".format(type(error).__name__, str(error)),
                )
                result["failed"] += 1
                if len(result["errors"]) < 100:
                    result["errors"].append(
                        {
                            "endpoint_id": item.endpoint_id,
                            "entity_key": item.entity_key,
                            "error": str(error),
                        }
                    )
                if not continue_on_error:
                    raise
            except Exception as error:
                self.database.mark_checkpoint_failed(
                    job_id,
                    "fmp_training",
                    item.item_key,
                    "{}: {}".format(type(error).__name__, str(error)),
                )
                result["failed"] += 1
                if len(result["errors"]) < 100:
                    result["errors"].append(
                        {
                            "endpoint_id": item.endpoint_id,
                            "entity_key": item.entity_key,
                            "error": str(error),
                        }
                    )
                if not continue_on_error:
                    raise
        if self.blocked_dimensions:
            result["failed"] += len(self.blocked_dimensions)
            result["errors"].extend(
                {**item, "error": "missing_collection_dimension"}
                for item in self.blocked_dimensions[: max(0, 100 - len(result["errors"]))]
            )
        result["blocked_dimensions"] = list(self.blocked_dimensions)
        result["checkpoint_summary"] = self.database.checkpoint_summary(job_id)
        result["fact_counts"] = self.store.counts()
        result["ok"] = result["failed"] == 0
        self.store.finish_run(job_id, result)
        return result

    def verify(self) -> dict:
        errors = []
        with self.database.connect() as connection:
            orphan = connection.execute(
                """
                SELECT f.id FROM fmp_training_facts f
                LEFT JOIN raw_artifacts r ON r.id=f.raw_artifact_id
                LEFT JOIN capture_events c ON c.id=f.capture_event_id
                WHERE r.id IS NULL OR c.id IS NULL
                LIMIT 100
                """
            ).fetchall()
        if orphan:
            errors.append({"error": "orphan_fmp_training_facts", "ids": [row["id"] for row in orphan]})
        return {"ok": not errors, "counts": self.store.counts(), "errors": errors}
