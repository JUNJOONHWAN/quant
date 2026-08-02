"""Daily FMP active US stock/ETF master snapshots.

This collector is intentionally separate from ``fmp_universe.py``.  The
older collector builds a broad survivorship-aware research universe.  This
module builds the operational, date-stamped master used to prove daily
post-training price coverage.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

from .providers import (
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
FMP_STABLE_ACTIVE_LIST_URL = (
    "https://financialmodelingprep.com/stable/actively-trading-list"
)
FMP_STABLE_STOCK_LIST_URL = "https://financialmodelingprep.com/stable/stock-list"
FMP_STABLE_ETF_LIST_URL = "https://financialmodelingprep.com/stable/etf-list"
FMP_STABLE_SYMBOL_CHANGE_URL = (
    "https://financialmodelingprep.com/stable/symbol-change"
)
FMP_LEGACY_AVAILABLE_TRADED_URL = (
    "https://financialmodelingprep.com/api/v3/available-traded/list"
)
FMP_LEGACY_STOCK_LIST_URL = (
    "https://financialmodelingprep.com/api/v3/stock/list"
)

FMP_ACTIVE_SCREENER_PAGE_SIZE = 1000
FMP_ACTIVE_SCREENER_MAX_PAGES = 30
FMP_SYMBOL_CHANGE_LIMIT = 10000
FMP_ACTIVE_EXCHANGES = ("NASDAQ", "NYSE", "AMEX", "CBOE")
_FMP_EXCHANGE_ALIASES = {
    "NASDAQ": "NASDAQ",
    "NASDAQ GLOBAL MARKET": "NASDAQ",
    "NASDAQ GLOBAL SELECT": "NASDAQ",
    "NASDAQ CAPITAL MARKET": "NASDAQ",
    "NYSE": "NYSE",
    "NEW YORK STOCK EXCHANGE": "NYSE",
    "AMEX": "AMEX",
    "NYSE AMERICAN": "AMEX",
    "NYSE ARCA": "AMEX",
    "CBOE": "CBOE",
    "BATS": "CBOE",
    "CBOE BZX": "CBOE",
}


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".active-universe-", dir=str(path.parent)
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


def _symbol(value: Any) -> Optional[str]:
    try:
        return normalize_symbol(str(value or ""))
    except ValueError:
        return None


def _bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _canonical_exchange(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("exchangeShortName", "exchange", "exchangeName"):
        value = str(row.get(key) or "").strip().upper()
        if value in _FMP_EXCHANGE_ALIASES:
            return _FMP_EXCHANGE_ALIASES[value]
    return None


def _security_type(row: Mapping[str, Any]) -> str:
    value = str(row.get("type") or "").strip().lower()
    if value:
        return value
    if _bool_or_none(row.get("isEtf")) is True:
        return "etf"
    if _bool_or_none(row.get("isFund")) is True:
        return "fund"
    return "stock"


class FmpActiveUniverseCollector:
    """Capture the current active FMP stock/ETF master and reference catalog."""

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
        legacy_query_auth: bool = False,
    ) -> tuple[list[dict], dict[str, Any]]:
        if not self.api_key:
            raise CredentialError("FMP_API_KEY is not configured")
        request_params = dict(params)
        headers: dict[str, str] = {}
        if legacy_query_auth:
            request_params["apikey"] = self.api_key
        else:
            headers["apikey"] = self.api_key
        result = self.http.get_json(
            source="fmp",
            dataset=dataset,
            partition_key=partition_key,
            url=url,
            params=request_params,
            headers=headers,
            logical_request={
                "endpoint_contract": "fmp_active_us_daily_master",
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
        return rows, {
            "dataset": dataset,
            "artifact_id": result.artifact.artifact_id,
            "capture_event_id": result.artifact.capture_event_id,
            "captured_at_utc": result.artifact.captured_at_utc,
            "payload_sha256": result.artifact.payload_sha256,
            "row_count": len(rows),
        }

    def _capture_screener_exchange(
        self, as_of: str, exchange: str
    ) -> tuple[list[dict], list[dict[str, Any]], list[str]]:
        rows: list[dict] = []
        artifacts: list[dict[str, Any]] = []
        warnings: list[str] = []
        seen_hashes: set[str] = set()
        for page in range(FMP_ACTIVE_SCREENER_MAX_PAGES):
            page_rows, artifact = self._capture_list(
                dataset="active_company_screener_{}".format(exchange.lower()),
                url=FMP_COMPANY_SCREENER_URL,
                params={
                    "exchange": exchange,
                    "isActivelyTrading": "true",
                    "limit": FMP_ACTIVE_SCREENER_PAGE_SIZE,
                    "page": page,
                },
                partition_key="{}_{}_page_{:03d}".format(
                    as_of, exchange.lower(), page
                ),
            )
            artifacts.append({**artifact, "exchange": exchange, "page": page})
            payload_hash = str(artifact["payload_sha256"])
            if payload_hash in seen_hashes:
                warnings.append(
                    "screener_repeated_payload_{}_page_{}".format(exchange, page)
                )
                break
            seen_hashes.add(payload_hash)
            rows.extend(page_rows)
            if len(page_rows) < FMP_ACTIVE_SCREENER_PAGE_SIZE:
                break
        else:
            warnings.append("screener_hit_max_pages_{}".format(exchange))
        return rows, artifacts, warnings

    @staticmethod
    def _merge(
        records: dict[str, dict[str, Any]],
        row: Mapping[str, Any],
        source: str,
    ) -> None:
        symbol = _symbol(row.get("symbol"))
        exchange = _canonical_exchange(row)
        if not symbol or not exchange:
            return
        record = records.setdefault(
            symbol,
            {
                "symbol": symbol,
                "name": None,
                "exchange": exchange,
                "security_type": _security_type(row),
                "is_etf": None,
                "is_fund": None,
                "is_actively_trading": None,
                "sources": [],
            },
        )
        record["exchange"] = record.get("exchange") or exchange
        record["name"] = record.get("name") or (
            str(row.get("companyName") or row.get("name") or "").strip() or None
        )
        if record.get("security_type") in {"", "stock"}:
            record["security_type"] = _security_type(row)
        for target, key in (
            ("is_etf", "isEtf"),
            ("is_fund", "isFund"),
            ("is_actively_trading", "isActivelyTrading"),
        ):
            parsed = _bool_or_none(row.get(key))
            if parsed is not None:
                record[target] = parsed
        if source == "stable_etf_list":
            record["is_etf"] = True
            record["security_type"] = "etf"
        if source not in record["sources"]:
            record["sources"].append(source)

    def capture(self, as_of_date: str) -> dict[str, Any]:
        as_of = validate_iso_date(as_of_date)
        captured_at = utc_now()
        artifacts: list[dict[str, Any]] = []
        warnings: list[str] = []
        screener_rows: list[dict] = []
        for exchange in FMP_ACTIVE_EXCHANGES:
            rows, exchange_artifacts, exchange_warnings = (
                self._capture_screener_exchange(as_of, exchange)
            )
            screener_rows.extend(rows)
            artifacts.extend(exchange_artifacts)
            warnings.extend(exchange_warnings)

        stable_active, artifact = self._capture_list(
            dataset="stable_actively_trading_list",
            url=FMP_STABLE_ACTIVE_LIST_URL,
            params={},
            partition_key=as_of,
        )
        artifacts.append(artifact)
        stable_stocks, artifact = self._capture_list(
            dataset="stable_stock_list",
            url=FMP_STABLE_STOCK_LIST_URL,
            params={},
            partition_key=as_of,
        )
        artifacts.append(artifact)
        stable_etfs, artifact = self._capture_list(
            dataset="stable_etf_list",
            url=FMP_STABLE_ETF_LIST_URL,
            params={},
            partition_key=as_of,
        )
        artifacts.append(artifact)
        available_traded, artifact = self._capture_list(
            dataset="legacy_available_traded_list",
            url=FMP_LEGACY_AVAILABLE_TRADED_URL,
            params={},
            partition_key=as_of,
            legacy_query_auth=True,
        )
        artifacts.append(artifact)
        legacy_stock_list, artifact = self._capture_list(
            dataset="legacy_stock_list",
            url=FMP_LEGACY_STOCK_LIST_URL,
            params={},
            partition_key=as_of,
            legacy_query_auth=True,
        )
        artifacts.append(artifact)
        symbol_changes: list[dict[str, Any]] = []
        symbol_change_exclusions: list[dict[str, Any]] = []
        change_rows, artifact = self._capture_list(
            dataset="stable_symbol_change",
            url=FMP_STABLE_SYMBOL_CHANGE_URL,
            params={"limit": FMP_SYMBOL_CHANGE_LIMIT},
            partition_key=as_of,
        )
        artifacts.append(artifact)
        if len(change_rows) >= FMP_SYMBOL_CHANGE_LIMIT:
            raise PayloadValidationError(
                "FMP symbol-change reached the configured limit; "
                "full ticker lineage is not proven"
            )
        for index, row in enumerate(change_rows):
            old_symbol = _symbol(row.get("oldSymbol"))
            new_symbol = _symbol(row.get("newSymbol"))
            event_date = str(row.get("date") or "")[:10]
            if not old_symbol or not new_symbol or len(event_date) != 10:
                raise PayloadValidationError(
                    "FMP symbol-change contains a malformed lineage row "
                    f"at source index {index}"
                )
            if old_symbol == new_symbol:
                symbol_change_exclusions.append(
                    {
                        "source_row_index": index,
                        "reason": "provider_no_op_same_symbol",
                        "old_symbol": old_symbol,
                        "new_symbol": new_symbol,
                        "event_date": event_date,
                    }
                )
                continue
            symbol_changes.append(
                {
                    "schema": "quant.fmp_symbol_change_event.v1",
                    "old_symbol": old_symbol,
                    "new_symbol": new_symbol,
                    "event_date": event_date,
                    "available_date": captured_at[:10],
                    "captured_at_utc": captured_at,
                    "raw_artifact_id": artifact["artifact_id"],
                    "capture_event_id": artifact["capture_event_id"],
                    "source_row_index": index,
                    "company_name": (
                        str(row.get("companyName") or "").strip() or None
                    ),
                }
            )

        active_symbols = {
            symbol
            for symbol in (_symbol(row.get("symbol")) for row in stable_active)
            if symbol
        }
        records: dict[str, dict[str, Any]] = {}
        core_symbols: set[str] = set()
        extended_symbols: set[str] = set()
        reference_symbols: set[str] = set()

        for row in screener_rows:
            symbol = _symbol(row.get("symbol"))
            exchange = _canonical_exchange(row)
            is_etf = _bool_or_none(row.get("isEtf"))
            is_fund = _bool_or_none(row.get("isFund"))
            if not symbol or not exchange:
                continue
            if is_etf is True or is_fund is False:
                self._merge(records, row, "active_company_screener")
                core_symbols.add(symbol)
                extended_symbols.add(symbol)

        for row in available_traded:
            symbol = _symbol(row.get("symbol"))
            exchange = _canonical_exchange(row)
            security_type = _security_type(row)
            if not symbol or not exchange or security_type not in {"stock", "etf"}:
                continue
            self._merge(records, row, "legacy_available_traded_list")
            extended_symbols.add(symbol)
            reference_symbols.add(symbol)

        for row in stable_etfs:
            symbol = _symbol(row.get("symbol"))
            if not symbol or not _canonical_exchange(row):
                continue
            if symbol in active_symbols:
                self._merge(records, row, "stable_etf_list")
                extended_symbols.add(symbol)
            reference_symbols.add(symbol)

        for source, rows in (
            ("stable_stock_list", stable_stocks),
            ("legacy_stock_list", legacy_stock_list),
        ):
            for row in rows:
                symbol = _symbol(row.get("symbol"))
                exchange = _canonical_exchange(row)
                security_type = _security_type(row)
                if (
                    not symbol
                    or not exchange
                    or security_type not in {"stock", "etf"}
                ):
                    continue
                reference_symbols.add(symbol)
                if symbol in extended_symbols:
                    self._merge(records, row, source)

        reference_symbols.update(extended_symbols)
        normalized = []
        for symbol in sorted(extended_symbols):
            record = records[symbol]
            record["sources"] = sorted(record["sources"])
            record["as_of_date"] = as_of
            record["captured_at_utc"] = captured_at
            record["availability_basis"] = "captured_current_fmp_active_master"
            record["core_company_or_etf"] = symbol in core_symbols
            normalized.append(record)

        base = self.data_root / "state" / "active_universe"
        stem = "fmp_active_us_{}".format(as_of.replace("-", ""))
        jsonl_path = base / (stem + ".jsonl")
        symbols_path = base / (stem + ".symbols.txt")
        core_path = base / (stem + ".core_symbols.txt")
        reference_path = base / (stem + ".reference_symbols.txt")
        symbol_changes_path = base / (stem + ".symbol_changes.jsonl")
        manifest_path = base / (stem + ".manifest.json")
        jsonl_payload = (
            "\n".join(canonical_json(row) for row in normalized) + "\n"
        ).encode("utf-8")
        symbols_payload = (
            "\n".join(sorted(extended_symbols)) + "\n"
        ).encode("utf-8")
        core_payload = ("\n".join(sorted(core_symbols)) + "\n").encode("utf-8")
        reference_payload = (
            "\n".join(sorted(reference_symbols)) + "\n"
        ).encode("utf-8")
        symbol_changes_payload = (
            "\n".join(
                canonical_json(row)
                for row in sorted(
                    symbol_changes,
                    key=lambda row: (
                        row["event_date"],
                        row["old_symbol"],
                        row["new_symbol"],
                    ),
                )
            )
            + "\n"
        ).encode("utf-8")
        exchange_counts = Counter(
            str(row["exchange"]) for row in normalized
        )
        manifest = {
            "schema_version": "quant.fmp_active_us_daily_master.v1",
            "provider": "fmp",
            "as_of_date": as_of,
            "captured_at_utc": captured_at,
            "exchange_scope": list(FMP_ACTIVE_EXCHANGES),
            "country_filter_used": False,
            "active_symbol_count": len(extended_symbols),
            "core_symbol_count": len(core_symbols),
            "reference_symbol_count": len(reference_symbols),
            "exchange_counts": dict(sorted(exchange_counts.items())),
            "jsonl_path": str(jsonl_path),
            "jsonl_sha256": sha256_bytes(jsonl_payload),
            "symbols_path": str(symbols_path),
            "symbols_sha256": sha256_bytes(symbols_payload),
            "core_symbols_path": str(core_path),
            "core_symbols_sha256": sha256_bytes(core_payload),
            "reference_symbols_path": str(reference_path),
            "reference_symbols_sha256": sha256_bytes(reference_payload),
            "symbol_changes_path": str(symbol_changes_path),
            "symbol_changes_sha256": sha256_bytes(symbol_changes_payload),
            "symbol_change_event_count": len(symbol_changes),
            "symbol_change_exclusion_count": len(
                symbol_change_exclusions
            ),
            "symbol_change_exclusions": symbol_change_exclusions,
            "raw_artifacts": artifacts,
            "warnings": sorted(set(warnings)),
            "membership_contract": {
                "core": (
                    "FMP company-screener isActivelyTrading=true on each "
                    "NASDAQ/NYSE/AMEX/CBOE shard, excluding mutual funds"
                ),
                "extended": (
                    "core plus FMP available-traded stock/ETF variants and "
                    "active FMP ETF directory members"
                ),
                "reference": (
                    "FMP legacy/stable stock and ETF catalogs on the four "
                    "venues; used only to classify same-day legacy bars"
                ),
            },
        }
        manifest_payload = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        _atomic_write(jsonl_path, jsonl_payload)
        _atomic_write(symbols_path, symbols_payload)
        _atomic_write(core_path, core_payload)
        _atomic_write(reference_path, reference_payload)
        _atomic_write(symbol_changes_path, symbol_changes_payload)
        _atomic_write(manifest_path, manifest_payload)
        return {
            "ok": True,
            "provider": "fmp",
            "as_of_date": as_of,
            "active_symbol_count": len(extended_symbols),
            "core_symbol_count": len(core_symbols),
            "reference_symbol_count": len(reference_symbols),
            "symbols_path": str(symbols_path),
            "symbols_sha256": manifest["symbols_sha256"],
            "core_symbols_path": str(core_path),
            "reference_symbols_path": str(reference_path),
            "reference_symbols_sha256": manifest["reference_symbols_sha256"],
            "symbol_changes_path": str(symbol_changes_path),
            "symbol_changes_sha256": manifest["symbol_changes_sha256"],
            "symbol_change_event_count": len(symbol_changes),
            "symbol_change_exclusion_count": len(
                symbol_change_exclusions
            ),
            "jsonl_path": str(jsonl_path),
            "manifest_path": str(manifest_path),
            "warnings": manifest["warnings"],
        }
