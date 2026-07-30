"""Orchestration, quality checks, verification, and packet export."""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .config import CredentialSet
from .etf_flows import (
    MASSIVE_ETF_FLOW_ENDPOINT_ID,
    MASSIVE_ETF_FLOW_PATH,
    EtfFlowLayer,
)
from .etf_flow_exposure import (
    ETF_CONSTITUENT_FLOW_POLICY_ID,
    build_constituent_flow_exposure,
)
from .fmp_etf_constituents import FmpEtfConstituentLayer
from .fmp_training import FmpTrainingBackfill
from .fmp_universe import FmpUniverseCollector
from .providers import (
    CaptureResult,
    DatasetError,
    FmpProvider,
    HttpCaptureClient,
    MassiveProvider,
    normalize_symbol,
    validate_iso_date,
)
from .point_in_time import etf_constituent_policy_manifest, etf_flow_policy_manifest
from .rate_limit import build_default_rate_limiters, rate_limit_policy
from .storage import Database, RawStore, canonical_json, sha256_bytes, utc_now


ENDPOINT_REGISTRY_VERSION = "2026-07-16.etf-flow-pit-training-v1"
SOURCE_CAPABILITY_POLICY = {
    "massive_grouped_daily": {
        "enabled": True,
        "role": "preferred_full_universe_daily_path",
        "endpoint": "/v2/aggs/grouped/locale/us/market/stocks/{date}",
        "request_granularity": "one_request_per_date",
    },
    "fmp_historical_eod_full": {
        "enabled": True,
        "role": "per_symbol_range_path",
        "endpoint": "/stable/historical-price-eod/full",
        "request_granularity": "one_request_per_symbol_and_range",
    },
    "fmp_eod_bulk": {
        "enabled": False,
        "role": "not_used",
        "endpoint": "/stable/eod-bulk",
        "latest_observed_http_status": 402,
        "latest_observed_classification": "not_entitled",
        "latest_observed_date": "2026-07-14",
        "reprobe_required_if_subscription_changes": True,
    },
    "phase2_universe_reference": {
        "enabled": True,
        "implementation_status": "implemented_fmp_snapshot",
        "category": "universe_and_reference",
        "sources": [
            "company-screener",
            "stock-list",
            "etf-list",
            "actively-trading-list",
            "delisted-companies",
            "symbol-change",
        ],
    },
    "phase2_corporate_actions": {
        "enabled": False,
        "implementation_status": "planned_not_implemented",
        "category": "dividends_splits_ticker_events",
    },
    "phase2_float_short": {
        "enabled": False,
        "implementation_status": "planned_not_implemented",
        "category": "float_short_interest_short_volume",
    },
    "phase2_news_sec": {
        "enabled": False,
        "implementation_status": "planned_not_implemented",
        "category": "news_sec_filings_and_risk_factors",
    },
    "phase2_macro": {
        "enabled": False,
        "implementation_status": "planned_not_implemented",
        "category": "treasury_economic_and_market_status",
    },
    "phase2_etf_flows": {
        "enabled": True,
        "implementation_status": "implemented",
        "category": "etf_flows_nav_and_shares_outstanding",
        "endpoint_id": MASSIVE_ETF_FLOW_ENDPOINT_ID,
        "endpoint": MASSIVE_ETF_FLOW_PATH,
        "historical_filters": ["processed_date.gte", "processed_date.lte"],
        "pagination": "next_url_with_page_resume",
        "authentication": "Authorization_Bearer_header",
        "not_holdings": True,
    },
    "phase2_etf_constituents": {
        "enabled": True,
        "implementation_status": "implemented_fmp_v4_historical",
        "category": "etf_constituent_point_in_time_relationships",
        "endpoint": "/api/v4/etf-holdings",
        "availability_gate": "acceptanceTime",
        "live_entitlement_evidence": {
            "fmp_v4_historical": 200,
            "fmp_stable_holdings": 402,
            "massive_constituents": 403,
            "checked_date": "2026-07-14",
        },
    },
    "phase2_options": {
        "enabled": False,
        "implementation_status": "planned_not_implemented",
        "category": "option_contracts_daily_bars_and_indicators",
    },
    "phase3_fmp_training_features": {
        "enabled": True,
        "implementation_status": "implemented_generic_endpoint_backfill",
        "category": (
            "fundamentals_analyst_corporate_actions_ownership_filings_news_"
            "macro_technical_reference"
        ),
        "entitlement_source": "live_263_endpoint_access_probe",
        "raw_policy": "immutable_gzip_sha256_redacted_request_metadata",
        "normalized_policy": "generic_source_preserving_fact_rows",
    },
}


@dataclass(frozen=True)
class QualityTolerances:
    price_relative: float = 0.005
    price_absolute: float = 0.02
    volume_relative: float = 0.15
    vwap_relative: float = 0.01
    hard_mismatch_multiplier: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _row_validation_reasons(row: Mapping[str, Any]) -> List[str]:
    reasons = []
    values = {field: row[field] for field in ("open", "high", "low", "close")}
    for field, value in values.items():
        if not _finite_number(value) or float(value) <= 0:
            reasons.append("{}:invalid_{}".format(row["source"], field))
    if reasons:
        return reasons
    high = float(values["high"])
    low = float(values["low"])
    if high < low:
        reasons.append("{}:high_below_low".format(row["source"]))
    if not low <= float(values["open"]) <= high:
        reasons.append("{}:open_outside_range".format(row["source"]))
    if not low <= float(values["close"]) <= high:
        reasons.append("{}:close_outside_range".format(row["source"]))
    volume = row["volume"]
    if volume is not None and (not _finite_number(volume) or float(volume) < 0):
        reasons.append("{}:invalid_volume".format(row["source"]))
    return reasons


class QualityEngine:
    def __init__(self, database: Database, tolerances: QualityTolerances):
        self.database = database
        self.tolerances = tolerances

    def evaluate(self, symbol: str, trade_date: str, rows: Sequence[Mapping[str, Any]]) -> dict:
        by_source = {str(row["source"]): row for row in rows}
        sources = sorted(by_source)
        reasons: List[str] = []
        for source in sources:
            reasons.extend(_row_validation_reasons(by_source[source]))
        if reasons:
            return {
                "symbol": symbol,
                "trade_date": trade_date,
                "status": "invalid",
                "sources": sources,
                "metrics": {},
                "reasons": sorted(set(reasons)),
                "tolerances": self.tolerances.to_dict(),
            }
        if "fmp" not in by_source or "massive" not in by_source:
            missing = sorted({"fmp", "massive"} - set(by_source))
            return {
                "symbol": symbol,
                "trade_date": trade_date,
                "status": "single_source",
                "sources": sources,
                "metrics": {},
                "reasons": ["missing_source:{}".format(item) for item in missing],
                "tolerances": self.tolerances.to_dict(),
            }

        fmp = by_source["fmp"]
        massive = by_source["massive"]
        metrics: Dict[str, Any] = {}
        soft_mismatches: List[str] = []
        hard_mismatches: List[str] = []
        hard_multiplier = self.tolerances.hard_mismatch_multiplier

        for field in ("open", "high", "low", "close"):
            left_value = fmp["adjusted_close"] if field == "close" and fmp["adjusted_close"] else fmp[field]
            right_value = massive[field]
            left = float(left_value)
            right = float(right_value)
            absolute = abs(left - right)
            relative = _relative_difference(left, right)
            within = (
                absolute <= self.tolerances.price_absolute
                or relative <= self.tolerances.price_relative
            )
            metrics[field] = {
                "fmp": left,
                "massive": right,
                "absolute_difference": absolute,
                "relative_difference": relative,
                "within_tolerance": within,
            }
            if not within:
                soft_mismatches.append(field)
                if (
                    absolute > self.tolerances.price_absolute * hard_multiplier
                    and relative > self.tolerances.price_relative * hard_multiplier
                ):
                    hard_mismatches.append(field)

        if fmp["volume"] is not None and massive["volume"] is not None:
            left = float(fmp["volume"])
            right = float(massive["volume"])
            relative = _relative_difference(left, right)
            within = relative <= self.tolerances.volume_relative
            metrics["volume"] = {
                "fmp": left,
                "massive": right,
                "relative_difference": relative,
                "within_tolerance": within,
            }
            if not within:
                soft_mismatches.append("volume")
                if relative > self.tolerances.volume_relative * hard_multiplier:
                    hard_mismatches.append("volume")
        else:
            metrics["volume"] = {"within_tolerance": None, "reason": "not_comparable"}

        if fmp["vwap"] is not None and massive["vwap"] is not None:
            left = float(fmp["vwap"])
            right = float(massive["vwap"])
            relative = _relative_difference(left, right)
            within = relative <= self.tolerances.vwap_relative
            metrics["vwap"] = {
                "fmp": left,
                "massive": right,
                "relative_difference": relative,
                "within_tolerance": within,
            }
            if not within:
                soft_mismatches.append("vwap")
                if relative > self.tolerances.vwap_relative * hard_multiplier:
                    hard_mismatches.append("vwap")
        else:
            metrics["vwap"] = {"within_tolerance": None, "reason": "not_comparable"}

        status = "fail" if hard_mismatches else ("warn" if soft_mismatches else "pass")
        reasons = ["mismatch:{}".format(field) for field in soft_mismatches]
        reasons.extend("hard_mismatch:{}".format(field) for field in hard_mismatches)
        return {
            "symbol": symbol,
            "trade_date": trade_date,
            "status": status,
            "sources": sources,
            "metrics": metrics,
            "reasons": sorted(set(reasons)),
            "tolerances": self.tolerances.to_dict(),
        }

    def recompute(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> dict:
        rows = self.database.observation_rows(start_date, end_date, symbols)
        grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["symbol"]), str(row["trade_date"]))].append(row)
        records = [
            self.evaluate(symbol, trade_date, grouped[(symbol, trade_date)])
            for symbol, trade_date in sorted(grouped)
        ]
        self.database.upsert_quality_many(records)
        return dict(Counter(record["status"] for record in records))


def _normalize_sources(source: str) -> Set[str]:
    if source == "both":
        return {"fmp", "massive"}
    if source in ("fmp", "massive"):
        return {source}
    raise ValueError("source must be one of: both, fmp, massive")


def _normalize_symbols(symbols: Optional[Sequence[str]]) -> List[str]:
    result = []
    for symbol in symbols or []:
        normalized = normalize_symbol(symbol)
        if normalized not in result:
            result.append(normalized)
    return sorted(result)


def _job_id(prefix: str, values: Mapping[str, Any]) -> str:
    digest = sha256_bytes(canonical_json(values).encode("utf-8"))[:16]
    return "{}:{}".format(prefix, digest)


def _weekday_dates(start: str, end: str) -> List[str]:
    current = date.fromisoformat(start)
    final = date.fromisoformat(end)
    result = []
    while current <= final:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


class DatasetPipeline:
    """High-level daily capture and export API."""

    def __init__(
        self,
        data_root: Path,
        credentials: CredentialSet,
        session: Optional[Any] = None,
        timeout_seconds: float = 120.0,
        retries: int = 3,
        sleep=None,
        tolerances: Optional[QualityTolerances] = None,
        rate_limiters: Optional[Mapping[str, Any]] = None,
        database: Optional[Database] = None,
        read_only: bool = False,
    ):
        self.data_root = Path(data_root).expanduser()
        if not read_only:
            self.data_root.mkdir(parents=True, exist_ok=True)
        self.credentials = credentials
        if read_only and database is None:
            raise ValueError("read_only DatasetPipeline requires an explicit database")
        self.database = database or Database(self.data_root)
        self.raw_store = RawStore(self.data_root, self.database)
        http_kwargs = {
            "raw_store": self.raw_store,
            "session": session,
            "timeout_seconds": timeout_seconds,
            "retries": retries,
        }
        if sleep is not None:
            http_kwargs["sleep"] = sleep
        http_kwargs["rate_limiters"] = (
            build_default_rate_limiters() if rate_limiters is None else rate_limiters
        )
        self.http = HttpCaptureClient(**http_kwargs)
        self.fmp = FmpProvider(self.http, credentials.fmp_api_key)
        self.fmp_universe = FmpUniverseCollector(
            self.data_root, self.http, credentials.fmp_api_key
        )
        self.massive = MassiveProvider(self.http, credentials.massive_api_key)
        self.etf_flows = EtfFlowLayer(
            self.database,
            self.http,
            credentials.massive_api_key,
            initialize_schema=not read_only,
        )
        self.etf_constituents = FmpEtfConstituentLayer(
            self.database,
            self.http,
            credentials.fmp_api_key,
            initialize_schema=not read_only,
        )
        self.fmp_training = (
            None
            if read_only
            else FmpTrainingBackfill(
                self.database, self.http, credentials.fmp_api_key
            )
        )
        self.tolerances = tolerances or QualityTolerances()
        self.quality = QualityEngine(self.database, self.tolerances)
        self._packet_flow_cache_date: Optional[str] = None
        self._packet_flow_cache: Dict[str, dict] = {}

    def _manifest(self) -> dict:
        return {
            "schema_version": "quant.dataset_manifest.v1",
            "updated_at_utc": utc_now(),
            "data_root": str(self.data_root),
            "database_relative_path": str(self.database.db_path.relative_to(self.data_root)),
            "source_capability_policy": SOURCE_CAPABILITY_POLICY,
            "rate_limit_policy": rate_limit_policy(),
            "endpoint_registry_version": ENDPOINT_REGISTRY_VERSION,
            "implemented_scope": "daily_backbone_plus_etf_relations_and_fmp_training_features",
            "implemented_extensions": [
                {
                    "phase": 2,
                    "provider": "massive",
                    "endpoint_id": MASSIVE_ETF_FLOW_ENDPOINT_ID,
                    "dataset": "ETF Global fund flows, NAV, and shares outstanding",
                },
                {
                    "phase": 2,
                    "provider": "fmp",
                    "endpoint_id": "fmp_v4_historical_etf_holdings",
                    "dataset": "historical ETF constituents and weights",
                },
                {
                    "phase": 3,
                    "provider": "fmp",
                    "endpoint_id": "live_classified_stable_catalog",
                    "dataset": (
                        "fundamentals, analyst, corporate actions, ownership, "
                        "filings, news, macro, technical, and reference facts"
                    ),
                },
            ],
            "layers": {
                "raw": "immutable gzip plus SHA256 and redacted request metadata",
                "normalized": (
                    "SQLite daily observations plus append-only ETF flow versions "
                    "and point-in-time ETF constituent snapshots plus generic "
                    "source-preserving FMP training facts"
                ),
                "training_packets": (
                    "deterministic unlabeled analysis inputs with ticker/date PIT "
                    "ETF-flow and ETF-constituent joins"
                ),
            },
            "historical_backfill_is_true_point_in_time": False,
            "etf_flow_pit_contract": {
                "event_date": "effective_date",
                "provider_processed_date": "processed_date",
                "available_date": "derived training_available_session_date",
                "capture_time": "captured_at_utc",
                "availability_policy": etf_flow_policy_manifest(),
                "confidence": "conservative_session_lag_fail_closed",
            },
            "etf_constituent_pit_contract": {
                "event_date": "effective_date",
                "provider_available_date": "acceptanceTime date",
                "available_date": "derived next U.S. trading session",
                "capture_time": "captured_at_utc",
                "availability_policy": etf_constituent_policy_manifest(),
                "confidence": "conservative_next_session_fail_closed",
            },
            "etf_flow_to_constituent_contract": {
                "policy_id": ETF_CONSTITUENT_FLOW_POLICY_ID,
                "allocation": "fund_flow * PIT constituent weight / 100",
                "membership_gate": "FMP acceptanceTime available_date <= as_of",
                "flow_gate": "massive_etf_flow_us_sessions_v1",
                "duplicate_positions": "sum within ETF before applying flow once",
                "currency_policy": "do not invent USD when provider currency is absent",
                "survivorship_policy": "do not use present-day membership or active status",
            },
        }

    def write_manifest(self) -> Path:
        path = self.data_root / "state" / "dataset_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(self._manifest(), indent=2, sort_keys=True) + "\n").encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest-", dir=str(path.parent))
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary_path), str(path))
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        return path

    def preflight(self, require_keys: bool = True) -> dict:
        self.write_manifest()
        state_directory = self.data_root / "state"
        descriptor, name = tempfile.mkstemp(prefix=".write-check-", dir=str(state_directory))
        os.close(descriptor)
        Path(name).unlink()
        key_status = self.credentials.status()
        missing = [name for name, item in key_status.items() if not item["configured"]]
        ok = not missing or not require_keys
        return {
            "ok": ok,
            "data_root": str(self.data_root),
            "database": str(self.database.db_path),
            "writable": True,
            "credentials": key_status,
            "missing_credentials": missing,
            "source_capability_policy": SOURCE_CAPABILITY_POLICY,
            "rate_limit_policy": rate_limit_policy(),
            "manifest": str(self.data_root / "state" / "dataset_manifest.json"),
            "network_requests_performed": 0,
        }

    def capture_fmp_universe(self, as_of_date: str) -> dict:
        """Capture current, ETF, delisted, and symbol-change FMP evidence."""

        result = self.fmp_universe.capture(as_of_date)
        self.write_manifest()
        return result

    def _capture_fmp_task(
        self, job_id: str, symbol: str, start_date: str, end_date: str
    ) -> Tuple[str, int]:
        item_key = "{}:{}:{}".format(symbol, start_date, end_date)
        scope = {"symbol": symbol, "from": start_date, "to": end_date}
        self.database.ensure_checkpoint(job_id, "fmp", item_key, scope)
        if self.database.checkpoint_status(job_id, "fmp", item_key) == "done":
            return "skipped", 0
        prior = self.database.completed_checkpoint_for_item(
            "fmp", item_key, exclude_job_id=job_id
        )
        if prior and prior["raw_artifact_id"] is not None:
            self.database.mark_checkpoint_done(
                job_id,
                "fmp",
                item_key,
                int(prior["raw_artifact_id"]),
                int(prior["observation_count"] or 0),
            )
            return "skipped", 0
        self.database.mark_checkpoint_running(job_id, "fmp", item_key)
        try:
            result = self.fmp.capture_range(symbol, start_date, end_date)
            count = self.database.upsert_observations(result.observations)
            self.quality.recompute(start_date, end_date, [symbol])
            self.database.mark_checkpoint_done(
                job_id, "fmp", item_key, result.artifact.artifact_id, count
            )
            return ("done" if count else "empty"), count
        except Exception as error:
            self.database.mark_checkpoint_failed(
                job_id, "fmp", item_key, "{}: {}".format(type(error).__name__, str(error))
            )
            raise

    def _capture_massive_task(
        self,
        job_id: str,
        trade_date: str,
        adjusted: bool,
        include_otc: bool,
    ) -> Tuple[str, int]:
        item_key = trade_date
        scope = {
            "date": trade_date,
            "adjusted": adjusted,
            "include_otc": include_otc,
            "full_universe": True,
        }
        self.database.ensure_checkpoint(job_id, "massive", item_key, scope)
        if self.database.checkpoint_status(job_id, "massive", item_key) == "done":
            return "skipped", 0
        self.database.mark_checkpoint_running(job_id, "massive", item_key)
        try:
            result = self.massive.capture_date(trade_date, adjusted, include_otc)
            count = self.database.upsert_observations(result.observations)
            self.quality.recompute(trade_date, trade_date)
            self.database.mark_checkpoint_done(
                job_id, "massive", item_key, result.artifact.artifact_id, count
            )
            return "done", count
        except Exception as error:
            self.database.mark_checkpoint_failed(
                job_id, "massive", item_key, "{}: {}".format(type(error).__name__, str(error))
            )
            raise

    def capture_daily(
        self,
        trade_date: str,
        symbols: Optional[Sequence[str]],
        source: str = "both",
        adjusted: bool = True,
        include_otc: bool = False,
        continue_on_error: bool = True,
    ) -> dict:
        normalized_date = validate_iso_date(trade_date)
        normalized_symbols = _normalize_symbols(symbols)
        sources = _normalize_sources(source)
        if "fmp" in sources and not normalized_symbols:
            raise ValueError("at least one --symbols value is required for FMP")
        contract = {
            "date": normalized_date,
            "symbols": normalized_symbols,
            "sources": sorted(sources),
            "adjusted": adjusted,
            "include_otc": include_otc,
        }
        job_id = _job_id("daily", contract)
        self.database.register_job(job_id, "capture_daily", contract, ENDPOINT_REGISTRY_VERSION)
        results = {
            "done": 0,
            "empty": 0,
            "skipped": 0,
            "failed": 0,
            "observations": 0,
            "errors": [],
        }

        if "massive" in sources:
            try:
                status, count = self._capture_massive_task(
                    job_id, normalized_date, adjusted, include_otc
                )
                results[status] += 1
                results["observations"] += count
            except Exception as error:
                results["failed"] += 1
                results["errors"].append(
                    {"source": "massive", "item": normalized_date, "error": str(error)}
                )
                if not continue_on_error:
                    raise

        if "fmp" in sources:
            for symbol in normalized_symbols:
                try:
                    status, count = self._capture_fmp_task(
                        job_id, symbol, normalized_date, normalized_date
                    )
                    results[status] += 1
                    results["observations"] += count
                except Exception as error:
                    results["failed"] += 1
                    results["errors"].append(
                        {"source": "fmp", "item": symbol, "error": str(error)}
                    )
                    if not continue_on_error:
                        raise
        self.write_manifest()
        results.update(
            {
                "job_id": job_id,
                "checkpoint_summary": self.database.checkpoint_summary(job_id),
                "quality": self.database.quality_counts(normalized_date, normalized_date),
                "ok": results["failed"] == 0,
            }
        )
        return results

    def backfill(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[Sequence[str]],
        source: str = "both",
        adjusted: bool = True,
        include_otc: bool = False,
        continue_on_error: bool = True,
        symbol_universe: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        start = validate_iso_date(start_date)
        end = validate_iso_date(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")
        normalized_symbols = _normalize_symbols(symbols)
        sources = _normalize_sources(source)
        if "fmp" in sources and not normalized_symbols:
            raise ValueError("at least one --symbols value is required for FMP")
        contract = {
            "from": start,
            "to": end,
            "symbols": normalized_symbols,
            "sources": sorted(sources),
            "adjusted": adjusted,
            "include_otc": include_otc,
            "symbol_universe": dict(symbol_universe or {}),
        }
        job_id = _job_id("backfill", contract)
        self.database.register_job(job_id, "backfill", contract, ENDPOINT_REGISTRY_VERSION)
        results = {
            "done": 0,
            "empty": 0,
            "skipped": 0,
            "failed": 0,
            "observations": 0,
            "errors": [],
        }

        if "fmp" in sources:
            for symbol in normalized_symbols:
                try:
                    status, count = self._capture_fmp_task(job_id, symbol, start, end)
                    results[status] += 1
                    results["observations"] += count
                except Exception as error:
                    results["failed"] += 1
                    results["errors"].append(
                        {"source": "fmp", "item": symbol, "error": str(error)}
                    )
                    if not continue_on_error:
                        raise

        if "massive" in sources:
            for trade_date in _weekday_dates(start, end):
                try:
                    status, count = self._capture_massive_task(
                        job_id, trade_date, adjusted, include_otc
                    )
                    results[status] += 1
                    results["observations"] += count
                except Exception as error:
                    results["failed"] += 1
                    results["errors"].append(
                        {"source": "massive", "item": trade_date, "error": str(error)}
                    )
                    if not continue_on_error:
                        raise
        self.write_manifest()
        results.update(
            {
                "job_id": job_id,
                "checkpoint_summary": self.database.checkpoint_summary(job_id),
                "quality": self.database.quality_counts(start, end),
                "ok": results["failed"] == 0,
            }
        )
        return results

    def capture_etf_flows(
        self,
        as_of_date: str,
        *,
        lookback_days: int = 7,
        tickers: Optional[Sequence[str]] = None,
        limit: int = 5000,
        max_lag_days: int = 4,
        resume: bool = True,
        strict_freshness: bool = False,
    ) -> dict:
        """Capture a recent ETF-flow processed-date window as of one date."""

        result = self.etf_flows.capture_as_of(
            as_of_date,
            lookback_days=lookback_days,
            tickers=tickers,
            limit=limit,
            max_lag_days=max_lag_days,
            resume=resume,
            strict_freshness=strict_freshness,
        )
        self.write_manifest()
        return result

    def backfill_etf_flows(
        self,
        start_date: str,
        end_date: str,
        *,
        tickers: Optional[Sequence[str]] = None,
        limit: int = 5000,
        resume: bool = True,
    ) -> dict:
        """Backfill the endpoint's documented historical processed-date range."""

        result = self.etf_flows.backfill(
            start_date,
            end_date,
            tickers=tickers,
            limit=limit,
            resume=resume,
        )
        self.write_manifest()
        return result

    def backfill_fmp_etf_constituents(
        self,
        start_date: str,
        end_date: str,
        tickers: Sequence[str],
        *,
        universe_contract: Optional[Mapping[str, Any]] = None,
        continue_on_error: bool = True,
    ) -> dict:
        """Backfill historical ETF constituent snapshots with PIT gates."""

        result = self.etf_constituents.backfill(
            start_date,
            end_date,
            tickers,
            universe_contract=universe_contract,
            continue_on_error=continue_on_error,
        )
        self.write_manifest()
        return result

    def backfill_fmp_training(
        self,
        plan_path: Path,
        symbols_path: Path,
        universe_jsonl: Path,
        start_date: str,
        end_date: str,
        *,
        endpoint_ids: Optional[Sequence[str]] = None,
        continue_on_error: bool = True,
    ) -> dict:
        """Backfill all FMP training facts selected by a classified catalog plan."""

        result = self.fmp_training.backfill(
            plan_path,
            symbols_path,
            universe_jsonl,
            start_date,
            end_date,
            endpoint_ids=endpoint_ids,
            continue_on_error=continue_on_error,
        )
        self.write_manifest()
        return result

    def verify(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> dict:
        start = validate_iso_date(start_date) if start_date else None
        end = validate_iso_date(end_date) if end_date else None
        normalized_symbols = _normalize_symbols(symbols)
        if start and end and start > end:
            raise ValueError("start_date must be <= end_date")
        raw = self.raw_store.verify_all()
        recomputed = self.quality.recompute(start, end, normalized_symbols or None)
        invalid_rows = [dict(row) for row in self.database.invalid_observation_rows()]
        errors = list(raw["errors"])
        errors.extend(
            {
                "source": row["source"],
                "symbol": row["symbol"],
                "trade_date": row["trade_date"],
                "error": "invalid_normalized_observation",
            }
            for row in invalid_rows
        )
        if recomputed.get("fail"):
            errors.append(
                {"error": "cross_source_hard_mismatch", "count": recomputed["fail"]}
            )
        etf_flow = self.etf_flows.verify()
        errors.extend(etf_flow["errors"])
        etf_constituents = self.etf_constituents.verify()
        errors.extend(etf_constituents["errors"])
        fmp_training = self.fmp_training.verify()
        errors.extend(fmp_training["errors"])
        return {
            "ok": not errors,
            "raw": raw,
            "etf_flow": etf_flow,
            "etf_constituents": etf_constituents,
            "fmp_training": fmp_training,
            "quality_recomputed": recomputed,
            "quality_totals": self.database.quality_counts(start, end),
            "database_counts": self.database.counts(),
            "invalid_observation_count": len(invalid_rows),
            "errors": errors,
        }

    def _export_pairs(
        self,
        start_date: str,
        end_date: str,
        symbols: Sequence[str],
        quality_statuses: Sequence[str],
    ) -> Iterator[Tuple[str, str]]:
        clauses = ["q.trade_date >= ?", "q.trade_date <= ?"]
        parameters: List[Any] = [start_date, end_date]
        if symbols:
            clauses.append("q.symbol IN ({})".format(",".join("?" for _ in symbols)))
            parameters.extend(symbols)
        if quality_statuses:
            clauses.append("q.status IN ({})".format(",".join("?" for _ in quality_statuses)))
            parameters.extend(quality_statuses)
        with self.database.connect() as connection:
            cursor = connection.execute(
                "SELECT q.symbol, q.trade_date FROM quality_checks q WHERE {} "
                "ORDER BY q.trade_date, q.symbol".format(" AND ".join(clauses)),
                parameters,
            )
            while True:
                rows = cursor.fetchmany(2000)
                if not rows:
                    break
                for row in rows:
                    yield str(row["symbol"]), str(row["trade_date"])

    def _history_dates(self, symbol: str, as_of_date: str, lookback_days: int) -> List[str]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT trade_date FROM daily_observations
                WHERE symbol=? AND trade_date<=?
                ORDER BY trade_date DESC LIMIT ?
                """,
                (symbol, as_of_date, lookback_days),
            ).fetchall()
        return sorted(str(row["trade_date"]) for row in rows)

    def _history_payload(
        self, symbol: str, as_of_date: str, lookback_days: int
    ) -> Tuple[List[dict], dict]:
        """Fetch an entire lookback window in one indexed query."""

        indexed_reader = getattr(self.database, "history_payload_rows", None)
        if callable(indexed_reader):
            rows = indexed_reader(symbol, as_of_date, lookback_days)
        else:
            with self.database.connect() as connection:
                rows = connection.execute(
                    """
                    WITH selected_dates AS (
                        SELECT trade_date FROM (
                            SELECT DISTINCT trade_date
                            FROM daily_observations
                            WHERE symbol=? AND trade_date<=?
                            ORDER BY trade_date DESC LIMIT ?
                        ) ORDER BY trade_date
                    )
                    SELECT o.*, r.payload_sha256, r.raw_relative_path,
                           ce.captured_at_utc
                    FROM daily_observations o
                    JOIN selected_dates d ON d.trade_date=o.trade_date
                    JOIN raw_artifacts r ON r.id=o.raw_artifact_id
                    JOIN capture_events ce ON ce.id=o.capture_event_id
                    WHERE o.symbol=?
                    ORDER BY o.trade_date, o.source
                    """,
                    (symbol, as_of_date, lookback_days, symbol),
                ).fetchall()
        history_by_date: Dict[str, List[dict]] = {}
        provenance = {}
        for row in rows:
            trade_date = str(row["trade_date"])
            history_by_date.setdefault(trade_date, []).append(
                self._observation_payload(row)
            )
            provenance[str(row["payload_sha256"])] = {
                "source": str(row["source"]),
                "captured_at_utc": str(row["captured_at_utc"]),
                "raw_relative_path": str(row["raw_relative_path"]),
            }
        return (
            [
                {"trade_date": trade_date, "sources": history_by_date[trade_date]}
                for trade_date in sorted(history_by_date)
            ],
            provenance,
        )

    @staticmethod
    def _observation_payload(row: Mapping[str, Any]) -> dict:
        return {
            "source": str(row["source"]),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "adjusted_close": row["adjusted_close"],
            "volume": row["volume"],
            "vwap": row["vwap"],
            "transaction_count": row["transaction_count"],
            "adjusted": bool(row["adjusted"]) if row["adjusted"] is not None else None,
        }

    def _flow_packet_cached(
        self, ticker: str, as_of_date: str, lookback_records: int
    ) -> dict:
        if self._packet_flow_cache_date != as_of_date:
            self._packet_flow_cache_date = as_of_date
            self._packet_flow_cache = {}
        key = "{}:{}".format(ticker, lookback_records)
        if key not in self._packet_flow_cache:
            self._packet_flow_cache[key] = self.etf_flows.packet_for_ticker(
                ticker, as_of_date, lookback_records
            )
        return copy.deepcopy(self._packet_flow_cache[key])

    def _constituent_flow_exposure(
        self, symbol: str, as_of_date: str, etf_constituents: Mapping[str, Any]
    ) -> Tuple[dict, dict]:
        memberships = list(etf_constituents.get("etf_memberships") or [])
        eligible_tickers = sorted(
            {
                str(row.get("etf_ticker"))
                for row in memberships
                if row.get("etf_ticker") and row.get("direct_equity_proxy_eligible")
            }
        )
        if self._packet_flow_cache_date != as_of_date:
            self._packet_flow_cache_date = as_of_date
            self._packet_flow_cache = {}
        missing = [
            ticker
            for ticker in eligible_tickers
            if "{}:1".format(ticker) not in self._packet_flow_cache
        ]
        if missing:
            bulk_packets = self.etf_flows.packets_for_tickers(missing, as_of_date, 1)
            for ticker, packet in bulk_packets.items():
                self._packet_flow_cache["{}:1".format(ticker)] = packet
        packets = {
            ticker: copy.deepcopy(self._packet_flow_cache["{}:1".format(ticker)])
            for ticker in eligible_tickers
        }
        provenance = {}
        for packet in packets.values():
            provenance.update(packet.get("raw_provenance") or {})
        exposure = build_constituent_flow_exposure(
            symbol, as_of_date, memberships, packets
        )
        return exposure, provenance

    def _build_analysis_packet(
        self, symbol: str, as_of_date: str, lookback_days: int
    ) -> dict:
        history, raw_provenance = self._history_payload(
            symbol, as_of_date, lookback_days
        )
        etf_flow = self._flow_packet_cached(
            symbol, as_of_date, min(20, lookback_days)
        )
        raw_provenance.update(etf_flow.pop("raw_provenance"))
        etf_constituents = self.etf_constituents.packet_for_symbol(
            symbol, as_of_date
        )
        raw_provenance.update(etf_constituents.pop("raw_provenance"))
        etf_flow_to_constituent, exposure_provenance = (
            self._constituent_flow_exposure(
                symbol, as_of_date, etf_constituents
            )
        )
        raw_provenance.update(exposure_provenance)
        quality_row = self.database.quality_for_pair(symbol, as_of_date)
        if quality_row is None:
            raise ValueError(
                "quality row missing for {} {}".format(symbol, as_of_date)
            )
        quality = {
            "status": quality_row["status"],
            "sources": json.loads(quality_row["sources_json"]),
            "metrics": json.loads(quality_row["metrics_json"]),
            "reasons": json.loads(quality_row["reasons_json"]),
            "tolerances": json.loads(quality_row["tolerances_json"]),
        }
        packet = {
            "schema_version": "quant.analysis_packet.v3",
            "symbol": symbol,
            "as_of_date": as_of_date,
            "history": history,
            "etf_flow": etf_flow,
            "etf_constituents": etf_constituents,
            "etf_flow_to_constituent": etf_flow_to_constituent,
            "quality": quality,
            "provenance": {
                "raw_artifacts": [
                    {"payload_sha256": digest, **raw_provenance[digest]}
                    for digest in sorted(raw_provenance)
                ],
                "sources_retained_separately": True,
                "historical_backfill_is_true_point_in_time": False,
            },
        }
        packet["packet_id"] = sha256_bytes(canonical_json(packet).encode("utf-8"))
        return packet

    def analysis_packet_for_pair(
        self,
        symbol: str,
        as_of_date: str,
        *,
        lookback_days: int = 21,
        recompute_quality: bool = True,
    ) -> dict:
        """Build one deterministic v3 packet without writing an intermediate file."""

        normalized = normalize_symbol(symbol)
        as_of = validate_iso_date(as_of_date)
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if recompute_quality:
            self.quality.recompute(as_of, as_of, [normalized])
        return self._build_analysis_packet(normalized, as_of, lookback_days)

    def export_packets(
        self,
        start_date: str,
        end_date: str,
        output_path: Path,
        symbols: Optional[Sequence[str]] = None,
        lookback_days: int = 21,
        quality_statuses: Sequence[str] = ("pass", "warn", "single_source"),
    ) -> dict:
        start = validate_iso_date(start_date)
        end = validate_iso_date(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        normalized_symbols = _normalize_symbols(symbols)
        allowed_statuses = {"pass", "warn", "fail", "single_source", "invalid"}
        statuses = sorted(set(quality_statuses))
        unknown = set(statuses) - allowed_statuses
        if unknown:
            raise ValueError("unknown quality status: {}".format(",".join(sorted(unknown))))
        self.quality.recompute(start, end, normalized_symbols or None)
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".packets-", dir=str(output.parent))
        temporary_path = Path(temporary_name)
        packet_count = 0
        byte_count = 0
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for symbol, as_of_date in self._export_pairs(
                    start, end, normalized_symbols, statuses
                ):
                    packet = self._build_analysis_packet(
                        symbol, as_of_date, lookback_days
                    )
                    payload = (canonical_json(packet) + "\n").encode("utf-8")
                    handle.write(payload)
                    digest.update(payload)
                    packet_count += 1
                    byte_count += len(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary_path), str(output))
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        return {
            "ok": True,
            "output": str(output),
            "packets": packet_count,
            "bytes": byte_count,
            "sha256": digest.hexdigest(),
            "streaming_write": True,
            "pair_order": "trade_date_then_symbol",
            "unlabeled": True,
            "contains_expert_answers": False,
        }
