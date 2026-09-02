"""Immutable FMP universe snapshots for survivorship-aware backfills."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .providers import (
    ApiRequestError,
    CredentialError,
    HttpCaptureClient,
    PayloadValidationError,
    normalize_symbol,
    validate_iso_date,
)
from .storage import canonical_json, sha256_bytes, utc_now


FMP_COMPANY_SCREENER_URL = (
    "https://financialmodelingprep.com/stable/company-screener"
)
FMP_STOCK_LIST_URL = "https://financialmodelingprep.com/stable/stock-list"
FMP_ETF_LIST_URL = "https://financialmodelingprep.com/stable/etf-list"
FMP_ACTIVE_LIST_URL = (
    "https://financialmodelingprep.com/stable/actively-trading-list"
)
FMP_DELISTED_URL = "https://financialmodelingprep.com/stable/delisted-companies"
FMP_SYMBOL_CHANGE_URL = "https://financialmodelingprep.com/stable/symbol-change"

FMP_UNIVERSE_SCREENER_LIMIT = 10000
FMP_UNIVERSE_PAGE_LIMIT = 100
FMP_UNIVERSE_MAX_PAGES = 500
_US_EXCHANGES = ("NASDAQ", "NYSE", "AMEX")
_US_DELISTED_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "OTC", "PNK"}
_US_STYLE_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9-]{0,15}$")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".universe-", dir=str(path.parent)
    )
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


def _bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1", "yes"}:
        return True
    if str(value).strip().lower() in {"false", "0", "no"}:
        return False
    return None


def _text(value: Any) -> Optional[str]:
    rendered = str(value or "").strip()
    return rendered or None


def _symbol(value: Any) -> Optional[str]:
    try:
        return normalize_symbol(str(value or ""))
    except ValueError:
        return None


class FmpUniverseCollector:
    """Capture current, delisted, ETF, and symbol-change universe evidence."""

    def __init__(
        self,
        data_root: Path,
        http: HttpCaptureClient,
        api_key: Optional[str],
    ):
        self.data_root = Path(data_root).expanduser()
        self.http = http
        self.api_key = api_key

    def _capture_list(
        self,
        *,
        dataset: str,
        url: str,
        params: Mapping[str, Any],
        partition_key: str,
    ) -> Tuple[List[dict], int, str]:
        if not self.api_key:
            raise CredentialError("FMP_API_KEY is not configured")
        result = self.http.get_json(
            source="fmp",
            dataset=dataset,
            partition_key=partition_key,
            url=url,
            params=dict(params),
            headers={"apikey": self.api_key},
            logical_request={
                "endpoint_contract": "fmp_universe_reference",
                "dataset": dataset,
                "partition_key": partition_key,
            },
        )
        if not isinstance(result.document, list):
            raise PayloadValidationError(
                "FMP {} payload is not a list (raw artifact id={})".format(
                    dataset, result.artifact.artifact_id
                )
            )
        rows = [dict(row) for row in result.document if isinstance(row, dict)]
        return rows, result.artifact.artifact_id, result.artifact.payload_sha256

    def _capture_paged(
        self,
        *,
        dataset: str,
        url: str,
        base_params: Optional[Mapping[str, Any]],
        as_of_date: str,
    ) -> Tuple[List[dict], List[int], List[str], List[str]]:
        rows: List[dict] = []
        artifact_ids: List[int] = []
        payload_hashes: List[str] = []
        warnings: List[str] = []
        seen_hashes = set()
        for page in range(FMP_UNIVERSE_MAX_PAGES):
            params = dict(base_params or {})
            params.update({"page": page, "limit": FMP_UNIVERSE_PAGE_LIMIT})
            try:
                page_rows, artifact_id, payload_hash = self._capture_list(
                    dataset=dataset,
                    url=url,
                    params=params,
                    partition_key="{}_page_{:04d}".format(as_of_date, page),
                )
            except ApiRequestError as error:
                if page == 0:
                    raise
                warnings.append(
                    "{}_pagination_stopped_page_{}_{}".format(
                        dataset, page, type(error).__name__
                    )
                )
                break
            artifact_ids.append(artifact_id)
            payload_hashes.append(payload_hash)
            if payload_hash in seen_hashes:
                warnings.append("{}_pagination_repeated_payload_page_{}".format(dataset, page))
                break
            seen_hashes.add(payload_hash)
            rows.extend(page_rows)
            if len(page_rows) < FMP_UNIVERSE_PAGE_LIMIT:
                break
        else:
            warnings.append("{}_pagination_hit_max_pages".format(dataset))
        return rows, artifact_ids, payload_hashes, warnings

    @staticmethod
    def _merge_record(
        records: Dict[str, dict],
        row: Mapping[str, Any],
        source_dataset: str,
        *,
        forced_symbol: Optional[str] = None,
    ) -> None:
        symbol = _symbol(forced_symbol or row.get("symbol"))
        if not symbol:
            return
        record = records.setdefault(
            symbol,
            {
                "symbol": symbol,
                "company_name": None,
                "country": None,
                "exchange": None,
                "exchange_short_name": None,
                "is_etf": None,
                "is_fund": None,
                "is_actively_trading": None,
                "ipo_date": None,
                "delisted_date": None,
                "sources": [],
                "symbol_change_events": [],
            },
        )
        mappings = {
            "company_name": row.get("companyName") or row.get("name"),
            "country": row.get("country"),
            "exchange": row.get("exchange"),
            "exchange_short_name": row.get("exchangeShortName"),
            "ipo_date": row.get("ipoDate"),
            "delisted_date": row.get("delistedDate"),
        }
        for key, value in mappings.items():
            if record.get(key) in (None, "") and _text(value):
                record[key] = _text(value)
        for key, source_key in (
            ("is_etf", "isEtf"),
            ("is_fund", "isFund"),
            ("is_actively_trading", "isActivelyTrading"),
        ):
            parsed = _bool_or_none(row.get(source_key))
            if parsed is not None:
                record[key] = parsed
        if source_dataset == "etf_list":
            record["is_etf"] = True
        if source_dataset == "delisted_companies":
            record["is_actively_trading"] = False
        if source_dataset not in record["sources"]:
            record["sources"].append(source_dataset)

    def capture(self, as_of_date: str) -> dict:
        as_of = validate_iso_date(as_of_date)
        captured_at = utc_now()
        records: Dict[str, dict] = {}
        artifacts: List[dict] = []
        warnings: List[str] = []

        screener_requests = [
            (
                "us_current_stocks",
                {
                    "country": "US",
                    "isEtf": "false",
                    "isFund": "false",
                    "limit": FMP_UNIVERSE_SCREENER_LIMIT,
                },
            ),
            (
                "us_current_etfs",
                {
                    "country": "US",
                    "isEtf": "true",
                    "limit": FMP_UNIVERSE_SCREENER_LIMIT,
                },
            ),
        ]
        for exchange in _US_EXCHANGES:
            screener_requests.append(
                (
                    "us_exchange_{}".format(exchange.lower()),
                    {
                        "country": "US",
                        "exchange": exchange,
                        "limit": FMP_UNIVERSE_SCREENER_LIMIT,
                    },
                )
            )

        for dataset, params in screener_requests:
            rows, artifact_id, payload_hash = self._capture_list(
                dataset="company_screener_{}".format(dataset),
                url=FMP_COMPANY_SCREENER_URL,
                params=params,
                partition_key=as_of,
            )
            artifacts.append(
                {
                    "dataset": dataset,
                    "artifact_id": artifact_id,
                    "payload_sha256": payload_hash,
                    "row_count": len(rows),
                }
            )
            if len(rows) == FMP_UNIVERSE_SCREENER_LIMIT:
                warnings.append("{}_possible_limit_cap".format(dataset))
            for row in rows:
                self._merge_record(records, row, dataset)

        for dataset, url in (
            ("stock_list", FMP_STOCK_LIST_URL),
            ("etf_list", FMP_ETF_LIST_URL),
            ("actively_trading_list", FMP_ACTIVE_LIST_URL),
        ):
            rows, artifact_id, payload_hash = self._capture_list(
                dataset=dataset,
                url=url,
                params={},
                partition_key=as_of,
            )
            artifacts.append(
                {
                    "dataset": dataset,
                    "artifact_id": artifact_id,
                    "payload_sha256": payload_hash,
                    "row_count": len(rows),
                }
            )
            for row in rows:
                symbol = _symbol(row.get("symbol"))
                if symbol in records or (
                    dataset == "etf_list" and symbol and _US_STYLE_SYMBOL.fullmatch(symbol)
                ):
                    self._merge_record(records, row, dataset)

        delisted, ids, hashes, paged_warnings = self._capture_paged(
            dataset="delisted_companies",
            url=FMP_DELISTED_URL,
            base_params={},
            as_of_date=as_of,
        )
        warnings.extend(paged_warnings)
        artifacts.append(
            {
                "dataset": "delisted_companies",
                "artifact_ids": ids,
                "payload_sha256": hashes,
                "row_count": len(delisted),
            }
        )
        excluded_non_us_delisted = 0
        for row in delisted:
            symbol = _symbol(row.get("symbol"))
            exchange = str(row.get("exchange") or "").strip().upper()
            if symbol not in records and exchange not in _US_DELISTED_EXCHANGES:
                excluded_non_us_delisted += 1
                continue
            self._merge_record(records, row, "delisted_companies")
        if excluded_non_us_delisted:
            warnings.append(
                "excluded_non_us_delisted_companies_{}".format(
                    excluded_non_us_delisted
                )
            )

        changes, ids, hashes, paged_warnings = self._capture_paged(
            dataset="symbol_change",
            url=FMP_SYMBOL_CHANGE_URL,
            base_params={},
            as_of_date=as_of,
        )
        warnings.extend(paged_warnings)
        artifacts.append(
            {
                "dataset": "symbol_change",
                "artifact_ids": ids,
                "payload_sha256": hashes,
                "row_count": len(changes),
            }
        )
        for row in changes:
            event = {
                "date": _text(row.get("date")),
                "old_symbol": _symbol(row.get("oldSymbol")),
                "new_symbol": _symbol(row.get("newSymbol")),
                "company_name": _text(row.get("companyName")),
            }
            for symbol in (event["old_symbol"], event["new_symbol"]):
                if not symbol:
                    continue
                if symbol not in records and not _US_STYLE_SYMBOL.fullmatch(symbol):
                    continue
                self._merge_record(
                    records,
                    row,
                    "symbol_change",
                    forced_symbol=symbol,
                )
                if event not in records[symbol]["symbol_change_events"]:
                    records[symbol]["symbol_change_events"].append(event)

        normalized = []
        for symbol in sorted(records):
            record = records[symbol]
            record["sources"] = sorted(record["sources"])
            record["symbol_change_events"] = sorted(
                record["symbol_change_events"],
                key=lambda item: (
                    item.get("date") or "",
                    item.get("old_symbol") or "",
                    item.get("new_symbol") or "",
                ),
            )
            record["as_of_date"] = as_of
            record["captured_at_utc"] = captured_at
            record["availability_basis"] = "captured_reference_snapshot"
            record["pit_confidence"] = (
                "listing_dates_partial"
                if record.get("ipo_date") or record.get("delisted_date")
                else "retrospective_current_reference"
            )
            record["analysis_eligible"] = not bool(record.get("is_fund"))
            record["eligibility_reason"] = (
                "security_or_etf" if record["analysis_eligible"] else "mutual_fund"
            )
            normalized.append(record)

        base = self.data_root / "state" / "universe"
        stem = "fmp_us_all_{}".format(as_of.replace("-", ""))
        jsonl_path = base / (stem + ".jsonl")
        symbols_path = base / (stem + ".symbols.txt")
        manifest_path = base / (stem + ".manifest.json")
        jsonl_payload = (
            "\n".join(canonical_json(row) for row in normalized) + "\n"
        ).encode("utf-8")
        symbols = [row["symbol"] for row in normalized if row["analysis_eligible"]]
        symbols_payload = ("\n".join(symbols) + "\n").encode("utf-8")
        source_counts = Counter(
            source for row in normalized for source in row.get("sources", [])
        )
        manifest = {
            "schema_version": "quant.fmp_universe_snapshot.v1",
            "provider": "fmp",
            "as_of_date": as_of,
            "captured_at_utc": captured_at,
            "row_count": len(normalized),
            "eligible_symbol_count": len(symbols),
            "etf_count": sum(row.get("is_etf") is True for row in normalized),
            "delisted_count": sum(bool(row.get("delisted_date")) for row in normalized),
            "excluded_non_us_delisted_count": excluded_non_us_delisted,
            "source_membership_counts": dict(sorted(source_counts.items())),
            "jsonl_path": str(jsonl_path),
            "jsonl_sha256": sha256_bytes(jsonl_payload),
            "symbols_path": str(symbols_path),
            "symbols_sha256": sha256_bytes(symbols_payload),
            "raw_artifacts": artifacts,
            "warnings": sorted(set(warnings)),
            "survivorship_bias_controls": [
                "current US screener shards",
                "ETF directory",
                "delisted companies",
                "symbol changes",
            ],
            "historical_backfill_is_true_point_in_time": False,
        }
        manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        _atomic_write(jsonl_path, jsonl_payload)
        _atomic_write(symbols_path, symbols_payload)
        _atomic_write(manifest_path, manifest_payload)
        return {
            "ok": True,
            "provider": "fmp",
            "as_of_date": as_of,
            "row_count": len(normalized),
            "eligible_symbol_count": len(symbols),
            "etf_count": manifest["etf_count"],
            "delisted_count": manifest["delisted_count"],
            "symbols_path": str(symbols_path),
            "symbols_sha256": manifest["symbols_sha256"],
            "jsonl_path": str(jsonl_path),
            "manifest_path": str(manifest_path),
            "warnings": manifest["warnings"],
        }


def symbol_file_contract(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    payload = resolved.read_bytes()
    symbols = [
        normalize_symbol(line)
        for line in payload.decode("utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return {
        "path": str(resolved),
        "sha256": sha256_bytes(payload),
        "symbol_count": len(set(symbols)),
    }


def read_symbol_file(path: Path) -> List[str]:
    contract = symbol_file_contract(path)
    del contract
    result = []
    for line in Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        symbol = normalize_symbol(line)
        if symbol not in result:
            result.append(symbol)
    return sorted(result)
