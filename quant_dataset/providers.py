"""FMP and Massive HTTP capture/parsing adapters."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

import requests

from .storage import RawArtifact, RawStore, redacted_request_metadata


FMP_EOD_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
MASSIVE_GROUPED_DAILY_URL = (
    "https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{date}"
)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class DatasetError(RuntimeError):
    """Base error for safe, user-visible dataset failures."""


class CredentialError(DatasetError):
    pass


class ApiRequestError(DatasetError):
    """HTTP/API failure with optional source-preserving response evidence."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        raw_artifact_id: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.raw_artifact_id = raw_artifact_id


class PayloadValidationError(DatasetError):
    pass


@dataclass(frozen=True)
class CaptureResult:
    artifact: RawArtifact
    observations: List[dict]


@dataclass(frozen=True)
class HttpCaptureResult:
    artifact: RawArtifact
    document: Any


def validate_iso_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("date must use YYYY-MM-DD: {!r}".format(value)) from error
    return parsed.isoformat()


def normalize_symbol(value: str, uppercase: bool = True) -> str:
    """Validate a provider symbol without losing provider-significant case.

    FMP and user-supplied lookup symbols are conventionally upper-case. Massive
    can return distinct mixed-case tickers (for example, preferred securities),
    so callers parsing Massive payloads must pass ``uppercase=False``.
    """

    symbol = str(value).strip()
    if uppercase:
        symbol = symbol.upper()
    if not symbol or len(symbol) > 64:
        raise ValueError("invalid symbol: {!r}".format(value))
    return symbol


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _integer(value: Any) -> Optional[int]:
    numeric = _number(value)
    return int(numeric) if numeric is not None else None


def _safe_response_headers(headers: Mapping[str, Any]) -> dict:
    allowed = {
        "content-type",
        "content-length",
        "etag",
        "last-modified",
        "date",
        "x-request-id",
        "request-id",
        "retry-after",
    }
    return {
        str(key).lower(): str(value)
        for key, value in (headers or {}).items()
        if str(key).lower() in allowed
    }


def _retry_after_seconds(headers: Mapping[str, Any], fallback: float) -> float:
    value = None
    for key, item in (headers or {}).items():
        if str(key).lower() == "retry-after":
            value = str(item).strip()
            break
    if value:
        try:
            return min(max(float(value), 0.0), 120.0)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
                seconds = parsed.timestamp() - time.time()
                return min(max(seconds, 0.0), 120.0)
            except (TypeError, ValueError, OverflowError):
                pass
    return min(max(fallback, 0.0), 120.0)


class HttpCaptureClient:
    """HTTP client that persists every received response before interpreting it."""

    def __init__(
        self,
        raw_store: RawStore,
        session: Optional[Any] = None,
        timeout_seconds: float = 120.0,
        retries: int = 3,
        sleep=time.sleep,
        rate_limiters: Optional[Mapping[str, Any]] = None,
    ):
        self.raw_store = raw_store
        self.session = session or requests.Session()
        self.timeout_seconds = float(timeout_seconds)
        self.retries = max(0, int(retries))
        self.sleep = sleep
        self.rate_limiters = dict(rate_limiters or {})

    def get_json(
        self,
        source: str,
        dataset: str,
        partition_key: str,
        url: str,
        params: Mapping[str, Any],
        headers: Optional[Mapping[str, Any]],
        logical_request: Mapping[str, Any],
    ) -> HttpCaptureResult:
        safe_request = redacted_request_metadata("GET", url, params, logical_request)
        last_transport_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            limiter = self.rate_limiters.get(source)
            if limiter is not None:
                limiter.acquire()
            try:
                response = self.session.get(
                    url,
                    params=dict(params),
                    headers=dict(headers or {}),
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as error:
                last_transport_error = error
                if attempt < self.retries:
                    self.sleep(min(2.0 ** attempt, 30.0))
                    continue
                raise ApiRequestError(
                    "{} transport failure after {} attempts ({})".format(
                        source, attempt + 1, type(error).__name__
                    )
                ) from error
            except Exception as error:
                # Test transports and alternate sessions may not derive from
                # requests.RequestException. Never include their raw message,
                # which can contain a credential-bearing URL.
                last_transport_error = error
                if attempt < self.retries:
                    self.sleep(min(2.0 ** attempt, 30.0))
                    continue
                raise ApiRequestError(
                    "{} transport failure after {} attempts ({})".format(
                        source, attempt + 1, type(error).__name__
                    )
                ) from error

            status_code = int(getattr(response, "status_code", 0))
            payload = bytes(getattr(response, "content", b""))
            response_headers = dict(getattr(response, "headers", {}) or {})
            artifact = self.raw_store.store(
                source=source,
                dataset=dataset,
                partition_key=partition_key,
                payload=payload,
                request=safe_request,
                response={
                    "status_code": status_code,
                    "headers": _safe_response_headers(response_headers),
                    "attempt": attempt + 1,
                },
            )
            if status_code in RETRYABLE_STATUS_CODES and attempt < self.retries:
                fallback = 60.0 if status_code == 429 else 2.0 ** attempt
                self.sleep(_retry_after_seconds(response_headers, fallback))
                continue
            if status_code < 200 or status_code >= 300:
                raise ApiRequestError(
                    "{} {} returned HTTP {} (raw artifact id={})".format(
                        source, dataset, status_code, artifact.artifact_id
                    ),
                    status_code=status_code,
                    raw_artifact_id=artifact.artifact_id,
                )
            try:
                document = json.loads(payload.decode("utf-8-sig"))
            except (UnicodeDecodeError, ValueError) as error:
                raise PayloadValidationError(
                    "{} {} returned invalid JSON (raw artifact id={})".format(
                        source, dataset, artifact.artifact_id
                    )
                ) from error
            return HttpCaptureResult(artifact=artifact, document=document)

        # The loop always returns or raises. Keep a defensive error for static
        # analyzers and future refactors.
        raise ApiRequestError(
            "{} request failed ({})".format(
                source, type(last_transport_error).__name__ if last_transport_error else "unknown"
            )
        )


class FmpProvider:
    """FMP stable historical EOD, intentionally one symbol per date range."""

    def __init__(self, http: HttpCaptureClient, api_key: Optional[str]):
        self.http = http
        self.api_key = api_key

    def capture_range(self, symbol: str, start_date: str, end_date: str) -> CaptureResult:
        if not self.api_key:
            raise CredentialError("FMP_API_KEY is not configured")
        normalized_symbol = normalize_symbol(symbol)
        start = validate_iso_date(start_date)
        end = validate_iso_date(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")
        result = self.http.get_json(
            source="fmp",
            dataset="historical_price_eod_full",
            partition_key="{}_{}".format(start, end),
            url=FMP_EOD_URL,
            params={
                "symbol": normalized_symbol,
                "from": start,
                "to": end,
            },
            headers={"apikey": self.api_key},
            logical_request={
                "endpoint_contract": "fmp_per_symbol_range",
                "symbol": normalized_symbol,
                "from": start,
                "to": end,
            },
        )
        document = result.document
        if isinstance(document, list):
            rows = document
        elif isinstance(document, dict) and isinstance(document.get("historical"), list):
            rows = document["historical"]
        elif isinstance(document, dict) and isinstance(document.get("data"), list):
            rows = document["data"]
        else:
            raise PayloadValidationError(
                "FMP historical EOD payload has no row list (raw artifact id={})".format(
                    result.artifact.artifact_id
                )
            )

        observations = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            trade_date = row.get("date")
            try:
                normalized_date = validate_iso_date(str(trade_date))
            except ValueError:
                continue
            if normalized_date < start or normalized_date > end:
                continue
            observations.append(
                {
                    "source": "fmp",
                    "symbol": normalize_symbol(row.get("symbol") or normalized_symbol),
                    "trade_date": normalized_date,
                    "open": _number(row.get("open")),
                    "high": _number(row.get("high")),
                    "low": _number(row.get("low")),
                    "close": _number(row.get("close")),
                    "adjusted_close": _number(row.get("adjClose")),
                    "volume": _number(row.get("volume")),
                    "vwap": _number(row.get("vwap")),
                    "transaction_count": None,
                    "adjusted": 1,
                    "source_timestamp_ms": None,
                    "raw_artifact_id": result.artifact.artifact_id,
                    "capture_event_id": result.artifact.capture_event_id,
                    "source_row_index": index,
                    "extra": {
                        "change": _number(row.get("change")),
                        "change_percent": _number(
                            row.get("changePercent", row.get("changePercentage"))
                        ),
                    },
                }
            )
        return CaptureResult(artifact=result.artifact, observations=observations)


class MassiveProvider:
    """Massive grouped daily full-universe capture, one request per date."""

    def __init__(self, http: HttpCaptureClient, api_key: Optional[str]):
        self.http = http
        self.api_key = api_key

    def capture_date(
        self,
        trade_date: str,
        adjusted: bool = True,
        include_otc: bool = False,
    ) -> CaptureResult:
        if not self.api_key:
            raise CredentialError("MASSIVE_API_KEY is not configured")
        normalized_date = validate_iso_date(trade_date)
        result = self.http.get_json(
            source="massive",
            dataset="grouped_daily_us_stocks",
            partition_key=normalized_date,
            url=MASSIVE_GROUPED_DAILY_URL.format(date=normalized_date),
            params={
                "adjusted": "true" if adjusted else "false",
                "include_otc": "true" if include_otc else "false",
            },
            headers={"Authorization": "Bearer {}".format(self.api_key)},
            logical_request={
                "endpoint_contract": "massive_grouped_daily_full_universe",
                "date": normalized_date,
                "adjusted": bool(adjusted),
                "include_otc": bool(include_otc),
            },
        )
        document = result.document
        if not isinstance(document, dict) or not isinstance(document.get("results", []), list):
            raise PayloadValidationError(
                "Massive grouped daily payload has no results list (raw artifact id={})".format(
                    result.artifact.artifact_id
                )
            )
        observations = []
        for index, row in enumerate(document.get("results") or []):
            if not isinstance(row, dict) or not row.get("T"):
                continue
            try:
                symbol = normalize_symbol(row["T"], uppercase=False)
            except ValueError:
                continue
            observations.append(
                {
                    "source": "massive",
                    "symbol": symbol,
                    "trade_date": normalized_date,
                    "open": _number(row.get("o")),
                    "high": _number(row.get("h")),
                    "low": _number(row.get("l")),
                    "close": _number(row.get("c")),
                    "adjusted_close": None,
                    "volume": _number(row.get("v")),
                    "vwap": _number(row.get("vw")),
                    "transaction_count": _integer(row.get("n")),
                    "adjusted": 1 if adjusted else 0,
                    "source_timestamp_ms": _integer(row.get("t")),
                    "raw_artifact_id": result.artifact.artifact_id,
                    "capture_event_id": result.artifact.capture_event_id,
                    "source_row_index": index,
                    "extra": {"otc": bool(row.get("otc", False))},
                }
            )
        return CaptureResult(artifact=result.artifact, observations=observations)
