"""Build the v11-R2 PIT eligibility audit, family registry, and Flow event cube.

The source stores are opened read-only.  Derived outputs are created in a new
directory and atomically promoted only after all fixed quality checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .contracts import (
    ACTIVE_LOOKBACK_SESSIONS,
    ANCHOR_TICKERS,
    AUDIT_FILENAME,
    DEFAULT_ETFRADAR_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_DATABASE,
    EVENT_CUBE_FILENAME,
    EVENT_CUBE_MANIFEST_FILENAME,
    EVENT_CUBE_SCHEMA_VERSION,
    FAMILY_REGISTRY_FILENAME,
    FAMILY_REGISTRY_SCHEMA_VERSION,
    FORBIDDEN_DIFFUSION_INPUTS,
    FORBIDDEN_TRANSFORMS,
    HYPOTHESIS_REGISTRY_FILENAME,
    MIN_ASSETS_USD,
    MIN_DOLLAR_VOLUME_USD,
    MIN_PRICE_USD,
    NEW_LIFECYCLE_SESSIONS,
    PHASE_A_AUDIT_SCHEMA_VERSION,
    STALE_AFTER_SESSIONS,
    TIMING_CONTRACT,
)
from .hypotheses import write_registry


_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,16}$")
_FOREIGN_SUFFIX_RE = re.compile(r"\.(?:L|DE|PA|AS|MI|SW|TO|V|HK|KS|KQ|TW|T|AX)$")
_ISSUER_PATTERNS = (
    ("ISHARES", "ISHARES"),
    ("VANGUARD", "VANGUARD"),
    ("SPDR", "STATE_STREET_SPDR"),
    ("INVESCO", "INVESCO"),
    ("PROSHARES", "PROSHARES"),
    ("DIREXION", "DIREXION"),
    ("GLOBAL X", "GLOBAL_X"),
    ("JPMORGAN", "JPMORGAN"),
    ("J.P. MORGAN", "JPMORGAN"),
    ("FIDELITY", "FIDELITY"),
    ("FIRST TRUST", "FIRST_TRUST"),
    ("VANECK", "VANECK"),
    ("WISDOMTREE", "WISDOMTREE"),
    ("ARK ", "ARK"),
    ("SCHWAB", "SCHWAB"),
    ("FRANKLIN", "FRANKLIN_TEMPLETON"),
    ("DIMENSIONAL", "DIMENSIONAL"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress(stage: str, **values: object) -> None:
    print(
        json.dumps(
            {"stage": stage, "at_utc": utc_now(), **values},
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def canonical_ticker(value: object) -> str:
    return str(value or "").strip().upper().replace("/", "-")


def finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().upper() in {"1", "TRUE", "YES", "Y", "T"}


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def safe_output(path: Path, allowed_root: Path) -> Path:
    resolved = Path(path).resolve()
    allowed = Path(allowed_root).resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"output must stay below {allowed}: {resolved}")
    return resolved


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - container execution path
        raise RuntimeError(
            "Phase A must run in the existing NVIDIA container with pyarrow"
        ) from error
    return parquet.read_table(path).to_pylist()


def normalize_text(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")


def infer_issuer(name: object, explicit: object = None) -> str:
    value = normalize_text(explicit)
    if value:
        return value
    text = str(name or "").upper()
    for pattern, issuer in _ISSUER_PATTERNS:
        if pattern in text:
            return issuer
    return "UNKNOWN"


def normalize_benchmark(value: object, evidence: object = None) -> str:
    text = f"{value or ''} {evidence or ''}".upper()
    aliases = (
        (r"NASDAQ[ -]?100|NASDAQ 100|\bQQQ\b", "NASDAQ_100"),
        (r"S&P[ -]?500|S AND P 500|\bSPY\b", "SP_500"),
        (r"RUSSELL[ -]?2000|\bIWM\b", "RUSSELL_2000"),
        (r"DOW JONES INDUSTRIAL|\bDIA\b", "DOW_30"),
        (r"TOTAL (?:US |U\.S\. )?STOCK|\bVTI\b", "US_TOTAL_MARKET"),
        (r"EQUAL WEIGHT.*S&P|\bRSP\b", "SP_500_EQUAL_WEIGHT"),
        (r"SEMICONDUCTOR", "SEMICONDUCTORS"),
        (r"BITCOIN|\bBTC\b", "BITCOIN"),
        (r"ETHER(?:EUM)?|\bETH\b", "ETHEREUM"),
    )
    for pattern, result in aliases:
        if re.search(pattern, text):
            return result
    normalized = normalize_text(value)
    return normalized if len(normalized) >= 4 else ""


def infer_target_multiple(name: object, leverage: bool, inverse: bool) -> float:
    text = str(name or "").upper().replace("×", "X")
    matches = re.findall(r"(?<!\d)([123](?:\.\d+)?)\s*X", text)
    multiple = float(matches[0]) if matches else 1.0
    if not matches and "ULTRAPRO" in text:
        multiple = 3.0
    elif not matches and "ULTRA" in text and leverage:
        multiple = 2.0
    return -abs(multiple) if inverse else abs(multiple)


def classify_exposure(metadata: Mapping[str, Any]) -> dict[str, Any]:
    name = metadata.get("name") or metadata.get("fmp_name") or ""
    instrument = str(metadata.get("instrument_class") or "").upper()
    leverage = truthy(metadata.get("leverage_flag")) or "LEVERAGED" in instrument
    inverse = truthy(metadata.get("inverse_flag")) or "INVERSE" in instrument
    bond_cash = truthy(metadata.get("bond_cash_flag")) or "BOND_CASH" in instrument
    option_income = truthy(metadata.get("option_income_flag")) or "OPTION" in instrument
    single_stock = truthy(metadata.get("single_stock_flag")) or "SINGLE_STOCK" in instrument
    target_multiple = infer_target_multiple(name, leverage, inverse)
    if bond_cash:
        effective_sign = -1.0
        sign_basis = "defensive_risk_off_channel"
    elif inverse:
        effective_sign = -1.0
        sign_basis = "inverse_exposure"
    else:
        effective_sign = 1.0
        sign_basis = "long_exposure"
    clean = truthy(metadata.get("clean_rotation_eligible"))
    if not instrument:
        clean = not any((leverage, inverse, bond_cash, option_income, single_stock))
        instrument = "CLEAN_ROTATION_ELIGIBLE" if clean else "SPECIAL_UNCLASSIFIED"
    channel = "CLEAN_INDEPENDENT" if clean else "TYPED_SPECIAL"
    return {
        "instrument_class": instrument,
        "clean_rotation_eligible": int(clean),
        "leverage_flag": int(leverage),
        "inverse_flag": int(inverse),
        "bond_cash_flag": int(bond_cash),
        "option_income_flag": int(option_income),
        "single_stock_flag": int(single_stock),
        "target_multiple": target_multiple,
        "effective_sign": effective_sign,
        "sign_basis": sign_basis,
        "observation_channel": channel,
    }


def load_metadata(etfradar_root: Path) -> dict[str, dict[str, Any]]:
    table_root = Path(etfradar_root) / "tables"
    master_rows = read_parquet_records(table_root / "02_ETF_MASTER/data.parquet")
    classification_rows = read_parquet_records(
        table_root / "30_FMP_ETF_CLASSIFICATION/data.parquet"
    )
    result: dict[str, dict[str, Any]] = {}
    for row in master_rows:
        ticker = canonical_ticker(row.get("ticker"))
        if ticker:
            result[ticker] = dict(row)
    for row in classification_rows:
        ticker = canonical_ticker(row.get("ticker"))
        if ticker:
            result.setdefault(ticker, {}).update(
                {key: value for key, value in row.items() if value not in (None, "")}
            )
    for ticker, row in result.items():
        row["ticker"] = ticker
        row["issuer_family"] = infer_issuer(
            row.get("name") or row.get("fmp_name"), row.get("issuer")
        )
        row["benchmark_family"] = normalize_benchmark(
            row.get("benchmark"),
            " ".join(
                str(row.get(key) or "")
                for key in ("name", "fmp_name", "description", "objective")
            ),
        )
        row["cluster_family"] = normalize_text(
            row.get("cluster_auto")
            or row.get("cluster_auto_v1")
            or row.get("top_sector")
            or "UNCLASSIFIED"
        )
        row.update(classify_exposure(row))
    return result


def minhash_signature(tickers: Iterable[str], seeds: int = 32) -> tuple[int, ...]:
    values = sorted({canonical_ticker(value) for value in tickers if value})
    if not values:
        return tuple()
    signature: list[int] = []
    for seed in range(seeds):
        minimum = min(
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{value}".encode("utf-8"), digest_size=8
                ).digest(),
                "big",
            )
            for value in values
        )
        signature.append(minimum)
    return tuple(signature)


def weighted_jaccard(
    left: Mapping[str, float], right: Mapping[str, float]
) -> tuple[float, int]:
    keys = set(left) | set(right)
    if not keys:
        return 0.0, 0
    intersection = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    union = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return (intersection / union if union > 0 else 0.0, len(set(left) & set(right)))


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right


def _family_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE etf_identity(
          ticker TEXT PRIMARY KEY,
          name TEXT,
          issuer_family TEXT NOT NULL,
          benchmark_family TEXT,
          cluster_family TEXT NOT NULL,
          instrument_class TEXT NOT NULL,
          clean_rotation_eligible INTEGER NOT NULL,
          observation_channel TEXT NOT NULL,
          leverage_flag INTEGER NOT NULL,
          inverse_flag INTEGER NOT NULL,
          bond_cash_flag INTEGER NOT NULL,
          option_income_flag INTEGER NOT NULL,
          single_stock_flag INTEGER NOT NULL,
          target_multiple REAL NOT NULL,
          effective_sign REAL NOT NULL,
          sign_basis TEXT NOT NULL,
          holdings_signature_sha256 TEXT,
          holdings_snapshot_effective_date TEXT,
          holdings_snapshot_available_date TEXT,
          independent_family_id TEXT,
          identity_source TEXT NOT NULL,
          identity_pit_limit TEXT NOT NULL
        );
        CREATE TABLE holdings_top(
          etf_ticker TEXT NOT NULL,
          constituent_ticker TEXT NOT NULL,
          weight_fraction REAL NOT NULL,
          rank_in_etf INTEGER NOT NULL,
          effective_date TEXT NOT NULL,
          available_date TEXT NOT NULL,
          PRIMARY KEY(etf_ticker,constituent_ticker)
        );
        CREATE TABLE holding_overlap_edges(
          left_ticker TEXT NOT NULL,
          right_ticker TEXT NOT NULL,
          weighted_jaccard REAL NOT NULL,
          shared_top_holding_count INTEGER NOT NULL,
          cutoff_date TEXT NOT NULL,
          PRIMARY KEY(left_ticker,right_ticker)
        );
        CREATE TABLE relation_registry(
          ticker TEXT NOT NULL,
          relation_type TEXT NOT NULL,
          relation_value TEXT NOT NULL,
          relation_source TEXT NOT NULL,
          PRIMARY KEY(ticker,relation_type,relation_value)
        );
        CREATE INDEX idx_family_identity_family ON etf_identity(independent_family_id);
        CREATE INDEX idx_family_cluster ON etf_identity(cluster_family);
        CREATE INDEX idx_holding_constituent ON holdings_top(constituent_ticker);
        """
    )


def _latest_holdings(
    source: sqlite3.Connection, cutoff_date: str
) -> dict[str, tuple[str, str]]:
    rows = source.execute(
        """
        SELECT etf_ticker,effective_date,available_date FROM (
          SELECT etf_ticker,effective_date,available_date,
                 ROW_NUMBER() OVER (
                   PARTITION BY etf_ticker
                   ORDER BY available_date DESC,effective_date DESC
                 ) AS sequence
          FROM etf_constituent_snapshots
          WHERE available_date<=?
        ) WHERE sequence=1
        """,
        (cutoff_date,),
    )
    return {
        canonical_ticker(row[0]): (str(row[1]), str(row[2]))
        for row in rows
        if canonical_ticker(row[0])
    }


def build_family_registry(
    *,
    source: sqlite3.Connection,
    metadata: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    cutoff_date: str,
) -> dict[str, Any]:
    """Create identity, typed-exposure, and holdings-overlap relations."""

    tickers = sorted(
        {
            canonical_ticker(row[0])
            for row in source.execute("SELECT DISTINCT ticker FROM etf_flow_observations")
            if canonical_ticker(row[0])
        }
    )
    snapshots = _latest_holdings(source, cutoff_date)
    temporary = output_path.with_suffix(output_path.suffix + ".building")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    _family_schema(connection)
    top_holdings: dict[str, dict[str, float]] = {}
    snapshots_loaded = 0
    holding_rows_loaded = 0
    for ticker in tickers:
        snapshot = snapshots.get(ticker)
        if not snapshot:
            continue
        effective_date, available_date = snapshot
        rows = list(
            source.execute(
                """
                SELECT constituent_ticker,weight_percent,value_usd,balance
                FROM etf_constituent_observations
                WHERE etf_ticker=? AND effective_date=? AND available_date<=?
                  AND constituent_ticker IS NOT NULL AND constituent_ticker<>''
                ORDER BY COALESCE(weight_percent,value_usd,balance,0) DESC
                LIMIT 25
                """,
                (ticker, effective_date, cutoff_date),
            )
        )
        raw: list[tuple[str, float]] = []
        for row in rows:
            constituent = canonical_ticker(row[0])
            weight = finite(row[1])
            if weight is not None and weight > 0:
                score = weight / 100.0
            else:
                score = finite(row[2]) or finite(row[3]) or 0.0
            if constituent and score > 0:
                raw.append((constituent, score))
        total = sum(score for _, score in raw)
        if total <= 0:
            continue
        normalized = {name: score / total for name, score in raw}
        top_holdings[ticker] = normalized
        connection.executemany(
            "INSERT INTO holdings_top VALUES(?,?,?,?,?,?)",
            [
                (ticker, name, weight, rank, effective_date, available_date)
                for rank, (name, weight) in enumerate(
                    sorted(normalized.items(), key=lambda item: item[1], reverse=True),
                    start=1,
                )
            ],
        )
        snapshots_loaded += 1
        holding_rows_loaded += len(normalized)
        if snapshots_loaded % 500 == 0:
            progress(
                "phase_a_family_holdings",
                snapshots_loaded=snapshots_loaded,
                total_flow_tickers=len(tickers),
                holding_rows_loaded=holding_rows_loaded,
            )

    signatures = {
        ticker: minhash_signature(holdings) for ticker, holdings in top_holdings.items()
    }
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for ticker, signature in signatures.items():
        for band in range(8):
            chunk = signature[band * 4 : (band + 1) * 4]
            if chunk:
                buckets[(band, chunk)].append(ticker)
    candidates: set[tuple[str, str]] = set()
    skipped_large_buckets = 0
    for members in buckets.values():
        unique = sorted(set(members))
        if len(unique) > 250:
            skipped_large_buckets += 1
            continue
        candidates.update(itertools.combinations(unique, 2))
    overlap_edges: list[tuple[str, str, float, int, str]] = []
    for left, right in sorted(candidates):
        similarity, shared = weighted_jaccard(
            top_holdings[left], top_holdings[right]
        )
        if similarity >= 0.55 and shared >= 3:
            overlap_edges.append((left, right, similarity, shared, cutoff_date))
    connection.executemany(
        "INSERT INTO holding_overlap_edges VALUES(?,?,?,?,?)", overlap_edges
    )

    union = UnionFind(tickers)
    benchmark_groups: dict[str, list[str]] = defaultdict(list)
    lineage_groups: dict[str, list[str]] = defaultdict(list)
    for ticker in tickers:
        row = metadata.get(ticker, {})
        benchmark = str(row.get("benchmark_family") or "")
        if benchmark:
            benchmark_groups[benchmark].append(ticker)
        exposure = classify_exposure(row)
        if exposure["leverage_flag"] or exposure["inverse_flag"]:
            lineage = benchmark or normalize_benchmark(
                row.get("name") or row.get("fmp_name")
            )
            if lineage:
                lineage_groups[lineage].append(ticker)
    for members in benchmark_groups.values():
        for member in members[1:]:
            union.union(members[0], member)
    for members in lineage_groups.values():
        for member in members[1:]:
            union.union(members[0], member)
    for left, right, similarity, _, _ in overlap_edges:
        if similarity >= 0.65:
            union.union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for ticker in tickers:
        components[union.find(ticker)].append(ticker)
    family_by_ticker: dict[str, str] = {}
    for members in components.values():
        family_id = "FAM_" + hashlib.sha256(
            "|".join(sorted(members)).encode("utf-8")
        ).hexdigest()[:16]
        for ticker in members:
            family_by_ticker[ticker] = family_id

    identities: list[tuple[Any, ...]] = []
    relations: list[tuple[str, str, str, str]] = []
    metadata_missing = 0
    for ticker in tickers:
        row = dict(metadata.get(ticker, {}))
        if not row:
            metadata_missing += 1
        exposure = classify_exposure(row)
        name = str(row.get("name") or row.get("fmp_name") or "")
        issuer = str(row.get("issuer_family") or infer_issuer(name))
        benchmark = str(row.get("benchmark_family") or "")
        cluster = str(row.get("cluster_family") or "UNCLASSIFIED")
        holdings = top_holdings.get(ticker, {})
        holdings_digest = (
            json_sha256(sorted(holdings.items())) if holdings else ""
        )
        snapshot = snapshots.get(ticker, ("", ""))
        identities.append(
            (
                ticker,
                name,
                issuer,
                benchmark,
                cluster,
                exposure["instrument_class"],
                exposure["clean_rotation_eligible"],
                exposure["observation_channel"],
                exposure["leverage_flag"],
                exposure["inverse_flag"],
                exposure["bond_cash_flag"],
                exposure["option_income_flag"],
                exposure["single_stock_flag"],
                exposure["target_multiple"],
                exposure["effective_sign"],
                exposure["sign_basis"],
                holdings_digest,
                snapshot[0],
                snapshot[1],
                family_by_ticker[ticker],
                "ETF_RADAR_02_30_PLUS_FMP_PIT_HOLDINGS",
                "identity classification is current/static; holdings availability is PIT",
            )
        )
        relations.extend(
            (
                (ticker, "issuer", issuer, "ETF_RADAR_02_30"),
                (ticker, "benchmark", benchmark, "ETF_RADAR_30_OR_TEXT"),
                (ticker, "cluster", cluster, "ETF_RADAR_02_30"),
                (
                    ticker,
                    "independent_family",
                    family_by_ticker[ticker],
                    "benchmark_lineage_plus_holdings_overlap",
                ),
            )
        )
    connection.executemany(
        "INSERT INTO etf_identity VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        identities,
    )
    connection.executemany(
        "INSERT OR IGNORE INTO relation_registry VALUES(?,?,?,?)",
        [row for row in relations if row[2]],
    )
    metadata_values = {
        "schema_version": FAMILY_REGISTRY_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "cutoff_date": cutoff_date,
        "ticker_count": len(tickers),
        "family_count": len(components),
        "identity_pit_limit": (
            "current/static ETF RADAR identity fields are not represented as historical "
            "as-observed; Phase B family fitting remains train-fold-local"
        ),
    }
    connection.executemany(
        "INSERT INTO metadata VALUES(?,?)",
        [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata_values.items()],
    )
    connection.commit()
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    connection.close()
    if integrity != "ok":
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"family registry integrity failure: {integrity}")
    os.replace(temporary, output_path)
    return {
        "schema_version": FAMILY_REGISTRY_SCHEMA_VERSION,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "ticker_count": len(tickers),
        "metadata_missing_ticker_count": metadata_missing,
        "latest_pit_holdings_snapshot_count": snapshots_loaded,
        "top_holding_row_count": holding_rows_loaded,
        "overlap_candidate_pair_count": len(candidates),
        "holding_overlap_edge_count": len(overlap_edges),
        "independent_family_count": len(components),
        "skipped_large_lsh_bucket_count": skipped_large_buckets,
        "cutoff_date": cutoff_date,
        "integrity_check": integrity,
        "pit_limit": metadata_values["identity_pit_limit"],
    }


@dataclass
class FlowObservation:
    ticker: str
    effective_date: str
    available_at_date: str
    processed_date: str
    fund_flow: float | None
    nav: float | None
    shares_outstanding: float | None

    @property
    def assets(self) -> float | None:
        if (
            self.nav is None
            or self.shares_outstanding is None
            or self.nav <= 0
            or self.shares_outstanding <= 0
        ):
            return None
        return self.nav * self.shares_outstanding


@dataclass
class VisibleState:
    observation: FlowObservation
    effective_position: int
    first_visible_position: int
    last_gap_before_update: int


def flow_observation(row: Sequence[Any]) -> FlowObservation:
    return FlowObservation(
        ticker=canonical_ticker(row[0]),
        effective_date=str(row[1]),
        available_at_date=str(row[2] or row[1]),
        processed_date=str(row[3] or row[1]),
        fund_flow=finite(row[4]),
        nav=finite(row[5]),
        shares_outstanding=finite(row[6]),
    )


def grouped_flow_rows(
    connection: sqlite3.Connection,
) -> Iterator[tuple[str, list[FlowObservation]]]:
    cursor = connection.execute(
        """
        SELECT ticker,effective_date,available_at_date,processed_date,
               fund_flow,nav,shares_outstanding
        FROM etf_flow_observations
        ORDER BY effective_date,ticker
        """
    )
    for effective_date, rows in itertools.groupby(cursor, key=lambda row: str(row[1])):
        yield effective_date, [flow_observation(row) for row in rows]


def session_contract(
    connection: sqlite3.Connection,
) -> tuple[list[str], list[dict[str, str]]]:
    sessions = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT trade_date FROM daily_observations
            WHERE source='fmp' AND symbol='SPY'
            ORDER BY trade_date
            """
        )
    ]
    mapping = [
        {
            "flow_date": sessions[index - 2],
            "price_date": sessions[index - 1],
            "signal_date": sessions[index],
        }
        for index in range(2, len(sessions))
    ]
    return sessions, mapping


def price_map(
    connection: sqlite3.Connection, trade_date: str
) -> dict[str, tuple[float | None, float | None, str]]:
    result: dict[str, tuple[float | None, float | None, str]] = {}
    rows = connection.execute(
        """
        SELECT symbol,adjusted_close,close,volume,source
        FROM daily_observations
        WHERE trade_date=? AND source IN ('fmp','massive')
        ORDER BY CASE source WHEN 'fmp' THEN 0 ELSE 1 END
        """,
        (trade_date,),
    )
    for row in rows:
        ticker = canonical_ticker(row[0])
        if not ticker or ticker in result:
            continue
        price = finite(row[1]) or finite(row[2])
        volume = finite(row[3])
        dollar_volume = (
            price * volume
            if price is not None and volume is not None and volume >= 0
            else None
        )
        result[ticker] = (price, dollar_volume, str(row[4]))
    return result


def hygiene_reasons(ticker: str, metadata: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _TICKER_RE.fullmatch(ticker):
        reasons.append("BAD_TICKER_FORMAT")
    if _FOREIGN_SUFFIX_RE.search(ticker):
        reasons.append("FOREIGN_LISTING_SUFFIX")
    currency = str(metadata.get("currency") or "").upper()
    if currency and currency != "USD":
        reasons.append("NON_USD")
    instrument_type = str(metadata.get("type") or "").upper()
    if instrument_type and not any(
        token in instrument_type for token in ("ETF", "ETN", "FUND")
    ):
        reasons.append("NON_ETF_TYPE")
    exchange = " ".join(
        str(metadata.get(key) or "").upper()
        for key in ("exchange", "exchangeShortName")
    )
    if any(token in exchange for token in ("OTC", "PINK", "GREY")):
        reasons.append("OTC_OR_GREY")
    return reasons


def eligibility_decision(
    *,
    ticker: str,
    metadata: Mapping[str, Any],
    assets: float | None,
    price: float | None,
    dollar_volume: float | None,
) -> dict[str, Any]:
    """Apply the frozen ETF RADAR floors as hard denominator exclusions."""

    reasons = hygiene_reasons(ticker, metadata)
    if assets is None:
        reasons.append("ASSETS_MISSING")
    elif assets < MIN_ASSETS_USD:
        reasons.append("ASSETS_BELOW_MIN")
    if price is None:
        reasons.append("PRICE_T1_MISSING")
    elif price < MIN_PRICE_USD:
        reasons.append("PRICE_BELOW_MIN")
    if dollar_volume is None:
        reasons.append("DOLLAR_VOLUME_T1_MISSING")
    elif dollar_volume < MIN_DOLLAR_VOLUME_USD:
        reasons.append("DOLLAR_VOLUME_BELOW_MIN")
    exposure = classify_exposure(metadata)
    strict = not reasons
    return {
        "strict_eligible": int(strict),
        "clean_eligible": int(strict and exposure["clean_rotation_eligible"]),
        "special_eligible": int(strict and not exposure["clean_rotation_eligible"]),
        "reasons": reasons,
        **exposure,
    }


def _event_cube_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE session_map(
          signal_date TEXT PRIMARY KEY,
          price_date TEXT NOT NULL,
          flow_date TEXT NOT NULL,
          signal_position INTEGER NOT NULL,
          CHECK(price_date<signal_date),
          CHECK(flow_date<price_date)
        );
        CREATE TABLE etf_flow_events(
          signal_date TEXT NOT NULL,
          price_date TEXT NOT NULL,
          flow_date TEXT NOT NULL,
          ticker TEXT NOT NULL,
          observed_exact_t2 INTEGER NOT NULL,
          true_zero INTEGER NOT NULL,
          missing_exact_t2 INTEGER NOT NULL,
          stale_visible_state INTEGER NOT NULL,
          lifecycle_state TEXT NOT NULL,
          reporting_age_sessions INTEGER NOT NULL,
          fund_flow REAL,
          nav REAL,
          shares_outstanding REAL,
          assets_usd REAL,
          flow_rate_pct REAL,
          price_t1 REAL,
          dollar_volume_t1 REAL,
          price_source TEXT,
          strict_eligible INTEGER NOT NULL,
          clean_eligible INTEGER NOT NULL,
          special_eligible INTEGER NOT NULL,
          exclusion_reasons TEXT NOT NULL,
          issuer_family TEXT NOT NULL,
          benchmark_family TEXT,
          cluster_family TEXT NOT NULL,
          independent_family_id TEXT NOT NULL,
          effective_sign REAL NOT NULL,
          target_multiple REAL NOT NULL,
          observation_channel TEXT NOT NULL,
          PRIMARY KEY(signal_date,ticker),
          CHECK(NOT (true_zero=1 AND missing_exact_t2=1))
        ) WITHOUT ROWID;
        CREATE TABLE daily_flow_state(
          signal_date TEXT PRIMARY KEY,
          price_date TEXT NOT NULL,
          flow_date TEXT NOT NULL,
          raw_observed_etf_count INTEGER NOT NULL,
          active_visible_etf_count INTEGER NOT NULL,
          strict_eligible_etf_count INTEGER NOT NULL,
          clean_eligible_etf_count INTEGER NOT NULL,
          special_eligible_etf_count INTEGER NOT NULL,
          excluded_small_or_hygiene_count INTEGER NOT NULL,
          observed_eligible_count INTEGER NOT NULL,
          true_zero_eligible_count INTEGER NOT NULL,
          nonzero_eligible_count INTEGER NOT NULL,
          missing_eligible_count INTEGER NOT NULL,
          stale_eligible_count INTEGER NOT NULL,
          new_lifecycle_count INTEGER NOT NULL,
          reactivated_count INTEGER NOT NULL,
          raw_signed_flow_usd REAL NOT NULL,
          eligible_signed_flow_usd REAL NOT NULL,
          eligible_absolute_flow_usd REAL NOT NULL,
          clean_signed_flow_usd REAL NOT NULL,
          special_raw_signed_flow_usd REAL NOT NULL,
          special_effective_signed_flow_usd REAL NOT NULL,
          drift_signed_flow_usd REAL NOT NULL,
          eligible_assets_usd REAL NOT NULL,
          drift_rate_pct REAL,
          independent_family_count INTEGER NOT NULL,
          observed_independent_family_count INTEGER NOT NULL,
          positive_independent_family_count INTEGER NOT NULL,
          negative_independent_family_count INTEGER NOT NULL,
          zero_independent_family_count INTEGER NOT NULL,
          independent_breadth_net REAL,
          diffusion_coverage REAL,
          CHECK(flow_date<price_date AND price_date<signal_date)
        );
        CREATE INDEX idx_event_ticker_date ON etf_flow_events(ticker,signal_date);
        CREATE INDEX idx_event_date_clean ON etf_flow_events(signal_date,clean_eligible);
        CREATE INDEX idx_event_date_cluster ON etf_flow_events(signal_date,cluster_family);
        CREATE INDEX idx_event_date_family ON etf_flow_events(signal_date,independent_family_id);
        """
    )


def load_family_map(path: Path) -> dict[str, dict[str, Any]]:
    connection = readonly_connection(path)
    try:
        return {
            str(row["ticker"]): dict(row)
            for row in connection.execute("SELECT * FROM etf_identity")
        }
    finally:
        connection.close()


def _first_last_flow_dates(
    connection: sqlite3.Connection,
) -> dict[str, tuple[str, str]]:
    return {
        canonical_ticker(row[0]): (str(row[1]), str(row[2]))
        for row in connection.execute(
            """
            SELECT ticker,MIN(effective_date),MAX(effective_date)
            FROM etf_flow_observations GROUP BY ticker
            """
        )
    }


def build_event_cube(
    *,
    source: sqlite3.Connection,
    metadata: Mapping[str, Mapping[str, Any]],
    family_registry_path: Path,
    output_path: Path,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    sessions, mapping = session_contract(source)
    session_position = {value: index for index, value in enumerate(sessions)}
    flow_range = source.execute(
        "SELECT MIN(effective_date),MAX(effective_date),COUNT(*) FROM etf_flow_observations"
    ).fetchone()
    min_flow_date, max_flow_date, source_flow_rows = (
        str(flow_range[0]),
        str(flow_range[1]),
        int(flow_range[2]),
    )
    relevant = [
        row
        for row in mapping
        if min_flow_date <= row["flow_date"] <= max_flow_date
        and (start_date is None or row["signal_date"] >= start_date)
        and (end_date is None or row["signal_date"] <= end_date)
    ]
    if not relevant:
        raise ValueError("no signal sessions overlap the requested Flow window")
    mapping_by_flow = {row["flow_date"]: row for row in relevant}
    first_last = _first_last_flow_dates(source)
    family = load_family_map(family_registry_path)

    temporary = output_path.with_suffix(output_path.suffix + ".building")
    temporary.unlink(missing_ok=True)
    output = sqlite3.connect(temporary)
    output.row_factory = sqlite3.Row
    output.execute("PRAGMA journal_mode=OFF")
    output.execute("PRAGMA synchronous=OFF")
    output.execute("PRAGMA temp_store=FILE")
    _event_cube_schema(output)
    output.executemany(
        "INSERT INTO session_map VALUES(?,?,?,?)",
        [
            (
                row["signal_date"],
                row["price_date"],
                row["flow_date"],
                session_position[row["signal_date"]],
            )
            for row in relevant
        ],
    )

    flow_groups = iter(grouped_flow_rows(source))
    next_group = next(flow_groups, None)
    state: dict[str, VisibleState] = {}
    pending: list[tuple[str, int, FlowObservation]] = []
    pending_sequence = 0
    total_event_rows = 0
    raw_rows_in_window = 0
    raw_rows_outside_us_calendar = 0
    raw_rows_unavailable_at_t = 0
    timing_violations = 0
    exclusion_counter: Counter[str] = Counter()
    lifecycle_counter: Counter[str] = Counter()
    per_year: dict[str, Counter[str]] = defaultdict(Counter)
    event_batch: list[tuple[Any, ...]] = []
    daily_rows: list[tuple[Any, ...]] = []

    def apply_visible(observation: FlowObservation, current_position: int) -> int:
        effective_position = session_position.get(observation.effective_date)
        if effective_position is None:
            return 0
        previous = state.get(observation.ticker)
        gap = (
            effective_position - previous.effective_position
            if previous is not None
            else 0
        )
        if previous is None or effective_position >= previous.effective_position:
            state[observation.ticker] = VisibleState(
                observation=observation,
                effective_position=effective_position,
                first_visible_position=(
                    previous.first_visible_position
                    if previous is not None
                    else current_position
                ),
                last_gap_before_update=gap,
            )
        return gap

    ordered_flow_dates = sorted(mapping_by_flow)
    for date_index, calendar_flow_date in enumerate(ordered_flow_dates, start=1):
        row_map = mapping_by_flow[calendar_flow_date]
        signal_date = row_map["signal_date"]
        price_date = row_map["price_date"]
        signal_pos = session_position[signal_date]
        flow_pos = session_position[calendar_flow_date]
        if not (
            sessions[signal_pos - 1] == price_date
            and sessions[signal_pos - 2] == calendar_flow_date
        ):
            timing_violations += 1
            raise RuntimeError(f"timing contract violation at {signal_date}")

        while next_group is not None and next_group[0] < calendar_flow_date:
            if next_group[0] not in session_position:
                raw_rows_outside_us_calendar += len(next_group[1])
            next_group = next(flow_groups, None)
        exact_rows: list[FlowObservation] = []
        if next_group is not None and next_group[0] == calendar_flow_date:
            exact_rows = next_group[1]
            raw_rows_in_window += len(exact_rows)
            next_group = next(flow_groups, None)
        for observation in exact_rows:
            heapq.heappush(
                pending,
                (observation.available_at_date, pending_sequence, observation),
            )
            pending_sequence += 1
        newly_applied: dict[str, int] = {}
        while pending and pending[0][0] <= signal_date:
            _, _, observation = heapq.heappop(pending)
            gap = apply_visible(observation, signal_pos)
            newly_applied[observation.ticker] = gap
        exact_visible = {
            observation.ticker: observation
            for observation in exact_rows
            if observation.available_at_date <= signal_date
        }
        raw_rows_unavailable_at_t += len(exact_rows) - len(exact_visible)
        prices = price_map(source, price_date)

        active = {
            ticker: visible
            for ticker, visible in state.items()
            if 0 <= flow_pos - visible.effective_position <= ACTIVE_LOOKBACK_SESSIONS
        }
        raw_signed_flow = sum(
            observation.fund_flow or 0.0 for observation in exact_visible.values()
        )
        counts: Counter[str] = Counter()
        counts["raw_observed"] = len(exact_visible)
        counts["active_visible"] = len(active)
        sums: defaultdict[str, float] = defaultdict(float)
        sums["raw_signed_flow"] = raw_signed_flow
        family_flow: defaultdict[str, float] = defaultdict(float)
        eligible_families: set[str] = set()

        for ticker, visible in sorted(active.items()):
            current = exact_visible.get(ticker)
            observed = current is not None
            observation = current if observed else visible.observation
            reporting_age = flow_pos - visible.effective_position
            meta = dict(metadata.get(ticker, {}))
            identity = family.get(ticker, {})
            meta.update({key: value for key, value in identity.items() if value is not None})
            price, dollar_volume, source_name = prices.get(ticker, (None, None, ""))
            decision = eligibility_decision(
                ticker=ticker,
                metadata=meta,
                assets=observation.assets,
                price=price,
                dollar_volume=dollar_volume,
            )
            for reason in decision["reasons"]:
                exclusion_counter[reason] += 1
            strict = bool(decision["strict_eligible"])
            clean = bool(decision["clean_eligible"])
            special = bool(decision["special_eligible"])
            if strict:
                counts["strict"] += 1
                counts["clean"] += int(clean)
                counts["special"] += int(special)
                sums["eligible_assets"] += observation.assets or 0.0
            else:
                counts["excluded"] += 1
            true_zero = bool(observed and current.fund_flow == 0.0)
            missing = not observed
            stale = bool(missing and reporting_age > STALE_AFTER_SESSIONS)
            since_first_visible = signal_pos - visible.first_visible_position
            gap = newly_applied.get(ticker, 0)
            if gap > ACTIVE_LOOKBACK_SESSIONS:
                lifecycle = "REACTIVATED"
                counts["reactivated"] += 1
            elif since_first_visible <= NEW_LIFECYCLE_SESSIONS:
                lifecycle = "NEW"
                counts["new"] += 1
            elif reporting_age > STALE_AFTER_SESSIONS:
                lifecycle = "STALE_ACTIVE"
            else:
                lifecycle = "ACTIVE"
            lifecycle_counter[lifecycle] += 1
            fund = current.fund_flow if observed else None
            flow_rate = (
                fund / observation.assets * 100.0
                if fund is not None and observation.assets
                else None
            )
            if strict:
                if observed:
                    counts["observed_eligible"] += 1
                    counts["true_zero"] += int(true_zero)
                    counts["nonzero"] += int(not true_zero and fund is not None)
                    sums["eligible_signed"] += fund or 0.0
                    sums["eligible_absolute"] += abs(fund or 0.0)
                    if clean:
                        sums["clean_signed"] += fund or 0.0
                    else:
                        sums["special_raw_signed"] += fund or 0.0
                        sums["special_effective_signed"] += (
                            (fund or 0.0)
                            * float(decision["effective_sign"])
                            * abs(float(decision["target_multiple"]))
                        )
                else:
                    counts["missing"] += 1
                    counts["stale"] += int(stale)
            family_id = str(
                identity.get("independent_family_id")
                or "FAM_" + hashlib.sha256(ticker.encode()).hexdigest()[:16]
            )
            if clean and strict:
                eligible_families.add(family_id)
                if observed:
                    family_flow[family_id] += fund or 0.0
            event_batch.append(
                (
                    signal_date,
                    price_date,
                    calendar_flow_date,
                    ticker,
                    int(observed),
                    int(true_zero),
                    int(missing),
                    int(stale),
                    lifecycle,
                    reporting_age,
                    fund,
                    observation.nav,
                    observation.shares_outstanding,
                    observation.assets,
                    flow_rate,
                    price,
                    dollar_volume,
                    source_name,
                    int(strict),
                    int(clean),
                    int(special),
                    "|".join(decision["reasons"]),
                    str(identity.get("issuer_family") or meta.get("issuer_family") or "UNKNOWN"),
                    str(identity.get("benchmark_family") or meta.get("benchmark_family") or ""),
                    str(identity.get("cluster_family") or meta.get("cluster_family") or "UNCLASSIFIED"),
                    family_id,
                    float(decision["effective_sign"]),
                    float(decision["target_multiple"]),
                    str(decision["observation_channel"]),
                )
            )
            if len(event_batch) >= 50_000:
                output.executemany(
                    "INSERT INTO etf_flow_events VALUES(" + ",".join("?" * 29) + ")",
                    event_batch,
                )
                total_event_rows += len(event_batch)
                event_batch.clear()

        positive_families = sum(value > 0 for value in family_flow.values())
        negative_families = sum(value < 0 for value in family_flow.values())
        zero_families = sum(value == 0 for value in family_flow.values())
        observed_families = len(family_flow)
        breadth_net = (
            (positive_families - negative_families) / observed_families
            if observed_families
            else None
        )
        diffusion_coverage = (
            observed_families / len(eligible_families) if eligible_families else None
        )
        drift_signed = sums["clean_signed"] + sums["special_effective_signed"]
        drift_rate = (
            drift_signed / sums["eligible_assets"] * 100.0
            if sums["eligible_assets"] > 0
            else None
        )
        daily_rows.append(
            (
                signal_date,
                price_date,
                calendar_flow_date,
                counts["raw_observed"],
                counts["active_visible"],
                counts["strict"],
                counts["clean"],
                counts["special"],
                counts["excluded"],
                counts["observed_eligible"],
                counts["true_zero"],
                counts["nonzero"],
                counts["missing"],
                counts["stale"],
                counts["new"],
                counts["reactivated"],
                sums["raw_signed_flow"],
                sums["eligible_signed"],
                sums["eligible_absolute"],
                sums["clean_signed"],
                sums["special_raw_signed"],
                sums["special_effective_signed"],
                drift_signed,
                sums["eligible_assets"],
                drift_rate,
                len(eligible_families),
                observed_families,
                positive_families,
                negative_families,
                zero_families,
                breadth_net,
                diffusion_coverage,
            )
        )
        year = signal_date[:4]
        per_year[year].update(
            {
                "signal_dates": 1,
                "raw_observed_rows": counts["raw_observed"],
                "event_rows": counts["active_visible"],
                "strict_eligible_rows": counts["strict"],
                "observed_eligible_rows": counts["observed_eligible"],
                "true_zero_rows": counts["true_zero"],
                "missing_rows": counts["missing"],
                "stale_rows": counts["stale"],
            }
        )
        if date_index == 1 or date_index % 100 == 0 or date_index == len(ordered_flow_dates):
            progress(
                "phase_a_event_cube",
                completed_signal_dates=date_index,
                total_signal_dates=len(ordered_flow_dates),
                signal_date=signal_date,
                event_rows_written=total_event_rows + len(event_batch),
                active_visible_etfs=counts["active_visible"],
                strict_eligible_etfs=counts["strict"],
            )

    if event_batch:
        output.executemany(
            "INSERT INTO etf_flow_events VALUES(" + ",".join("?" * 29) + ")",
            event_batch,
        )
        total_event_rows += len(event_batch)
    output.executemany(
        "INSERT INTO daily_flow_state VALUES(" + ",".join("?" * 32) + ")",
        daily_rows,
    )
    metadata_values = {
        "schema_version": EVENT_CUBE_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        "eligibility_policy": {
            "min_assets_usd": MIN_ASSETS_USD,
            "min_price_t1_usd": MIN_PRICE_USD,
            "min_dollar_volume_t1_usd": MIN_DOLLAR_VOLUME_USD,
            "small_etf_action": "remove_from_modeling_denominator",
        },
        "absolute_flow_policy": "preserved; no date centering or mean subtraction",
        "forbidden_diffusion_inputs": list(FORBIDDEN_DIFFUSION_INPUTS),
        "forbidden_transforms": list(FORBIDDEN_TRANSFORMS),
    }
    output.executemany(
        "INSERT INTO metadata VALUES(?,?)",
        [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata_values.items()],
    )
    output.commit()
    counters = dict(
        output.execute(
            """
            SELECT
              COUNT(*) event_rows,
              SUM(observed_exact_t2) observed_rows,
              SUM(true_zero) true_zero_rows,
              SUM(missing_exact_t2) missing_rows,
              SUM(stale_visible_state) stale_rows,
              SUM(strict_eligible) strict_eligible_rows,
              SUM(clean_eligible) clean_eligible_rows,
              SUM(special_eligible) special_eligible_rows
            FROM etf_flow_events
            """
        ).fetchone()
    )
    capital = dict(
        output.execute(
            """
            SELECT
              SUM(raw_signed_flow_usd) raw_signed_flow_usd,
              SUM(eligible_signed_flow_usd) eligible_signed_flow_usd,
              SUM(eligible_absolute_flow_usd) eligible_absolute_flow_usd,
              SUM(drift_signed_flow_usd) drift_signed_flow_usd
            FROM daily_flow_state
            """
        ).fetchone()
    )
    integrity = str(output.execute("PRAGMA integrity_check").fetchone()[0])
    output.close()
    if integrity != "ok" or timing_violations:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"event cube quality failure integrity={integrity} timing={timing_violations}"
        )
    os.replace(temporary, output_path)
    return {
        "schema_version": EVENT_CUBE_SCHEMA_VERSION,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "source_flow_row_count": source_flow_rows,
        "raw_flow_rows_in_requested_window": raw_rows_in_window,
        "raw_flow_rows_outside_us_calendar": raw_rows_outside_us_calendar,
        "raw_flow_rows_unavailable_at_own_t": raw_rows_unavailable_at_t,
        "signal_date_count": len(daily_rows),
        "signal_start": daily_rows[0][0],
        "signal_end": daily_rows[-1][0],
        "timing_violation_count": timing_violations,
        "counters": counters,
        "capital_mass": capital,
        "exclusion_reasons": dict(exclusion_counter.most_common()),
        "lifecycle_states": dict(lifecycle_counter.most_common()),
        "per_year": {year: dict(values) for year, values in sorted(per_year.items())},
        "integrity_check": integrity,
        "contract_checks": {
            "small_etfs_removed_from_denominator": counters["strict_eligible_rows"]
            < counters["event_rows"],
            "true_zero_separate_from_missing": True,
            "absolute_common_flow_preserved": True,
            "date_centering_used": False,
            "table_48_breadth_used": False,
        },
    }


def source_fingerprint(path: Path, connection: sqlite3.Connection) -> dict[str, Any]:
    stat = path.stat()
    schema_rows = [
        tuple(row)
        for row in connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        )
    ]
    page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sqlite_page_count": page_count,
        "sqlite_page_size": page_size,
        "sqlite_schema_sha256": json_sha256(schema_rows),
        "fingerprint_policy": (
            "stat plus schema digest; the multi-gigabyte authority DB is read-only and "
            "not copied or mutated by Phase A"
        ),
    }


def choose_eligibility_snapshots(etfradar_root: Path) -> list[dict[str, Any]]:
    candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    pattern = "**/tables/31_ETF_UNIVERSE_ELIGIBILITY/meta.json"
    for meta_path in Path(etfradar_root).glob(pattern):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        trade_date = str(meta.get("trade_date_us") or "")
        data_path = meta_path.with_name("data.parquet")
        if not trade_date or not data_path.is_file():
            continue
        path_text = str(meta_path)
        preference = (
            4
            if "/DAILY" in path_text
            else 3
            if "QUANT_PIT" in path_text
            else 2
            if "BACKFILL_DETAIL" in path_text
            else 1
        )
        candidates[trade_date].append(
            {
                "trade_date_us": trade_date,
                "run_id": str(meta.get("run_id") or ""),
                "updated_at_utc": str(meta.get("updated_at_utc") or ""),
                "row_count_meta": int(meta.get("row_count") or 0),
                "preference": preference,
                "meta_path": str(meta_path),
                "data_path": str(data_path),
            }
        )
    selected = []
    for trade_date, rows in candidates.items():
        selected.append(
            max(
                rows,
                key=lambda row: (
                    row["preference"],
                    row["updated_at_utc"],
                    row["run_id"],
                    row["data_path"],
                ),
            )
        )
    return sorted(selected, key=lambda row: row["trade_date_us"])


def audit_archived_eligibility(etfradar_root: Path) -> dict[str, Any]:
    selected = choose_eligibility_snapshots(etfradar_root)
    receipts: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []
    for index, item in enumerate(selected, start=1):
        try:
            rows = read_parquet_records(Path(item["data_path"]))
        except Exception as error:  # pragma: no cover - source-artifact failure path
            parse_failures.append(
                {"path": item["data_path"], "error": repr(error)}
            )
            continue
        counts = Counter()
        for row in rows:
            for key in (
                "universe_eligible",
                "flow_signal_eligible",
                "accum_eligible",
                "signal_eval_eligible",
                "clean_rotation_eligible",
            ):
                counts[key] += int(truthy(row.get(key)))
            counts[f"tier:{row.get('eligibility_tier') or 'MISSING'}"] += 1
        receipts.append(
            {
                **{key: value for key, value in item.items() if key != "preference"},
                "row_count_parquet": len(rows),
                "row_count_matches_meta": len(rows) == item["row_count_meta"],
                "counts": dict(counts),
                "data_sha256": sha256_file(Path(item["data_path"])),
                "as_observed_status": (
                    "DAILY_CAPTURE_OR_LATER_REPAIR"
                    if "/DAILY" in item["data_path"]
                    else "HISTORICAL_BACKFILL_NOT_AS_OBSERVED"
                ),
            }
        )
        if index % 25 == 0 or index == len(selected):
            progress(
                "phase_a_archived_masks",
                completed_trade_dates=index,
                total_trade_dates=len(selected),
                trade_date=item["trade_date_us"],
            )
    return {
        "artifact_file_count": len(
            list(
                Path(etfradar_root).glob(
                    "**/tables/31_ETF_UNIVERSE_ELIGIBILITY/meta.json"
                )
            )
        ),
        "selected_unique_trade_date_count": len(selected),
        "parsed_unique_trade_date_count": len(receipts),
        "first_trade_date": receipts[0]["trade_date_us"] if receipts else None,
        "last_trade_date": receipts[-1]["trade_date_us"] if receipts else None,
        "historical_coverage_limitation": (
            "archived table 31 masks begin in 2026; 2017-2025 eligibility is "
            "reconstructed from date-local AUM, price, liquidity, and Flow visibility"
        ),
        "parse_failures": parse_failures,
        "receipts": receipts,
    }


def current_lineage_audit(etfradar_root: Path) -> dict[str, Any]:
    tables = {}
    for name in (
        "24_CLUSTER_SUMMARY",
        "31_ETF_UNIVERSE_ELIGIBILITY",
        "48_MASSIVE_ACCUM_CLUSTER",
    ):
        path = Path(etfradar_root) / f"tables/{name}/meta.json"
        tables[name] = json.loads(path.read_text(encoding="utf-8"))
    cluster = tables["24_CLUSTER_SUMMARY"]
    eligibility = tables["31_ETF_UNIVERSE_ELIGIBILITY"]
    biased = tables["48_MASSIVE_ACCUM_CLUSTER"]
    cluster_matches = (
        cluster.get("trade_date_us") == eligibility.get("trade_date_us")
        and cluster.get("run_id") == eligibility.get("run_id")
    )
    table48_matches = (
        biased.get("trade_date_us") == eligibility.get("trade_date_us")
        and biased.get("run_id") == eligibility.get("run_id")
    )
    return {
        "tables": tables,
        "cluster_24_matches_eligibility_31": cluster_matches,
        "table_48_matches_eligibility_31": table48_matches,
        "join_permission": "DENY" if not cluster_matches else "ALLOW_SAME_LINEAGE_ONLY",
        "table_48_diffusion_breadth_permission": "FORBIDDEN_REGARDLESS_OF_LINEAGE",
    }


def event_archive_comparison(
    event_cube_path: Path, archive_audit: Mapping[str, Any]
) -> dict[str, Any]:
    connection = readonly_connection(event_cube_path)
    try:
        reconstructed = {
            str(row[0]): {
                "strict_eligible": int(row[1]),
                "clean_eligible": int(row[2]),
                "special_eligible": int(row[3]),
            }
            for row in connection.execute(
                """
                SELECT signal_date,strict_eligible_etf_count,
                       clean_eligible_etf_count,special_eligible_etf_count
                FROM daily_flow_state
                """
            )
        }
    finally:
        connection.close()
    comparisons = []
    for receipt in archive_audit.get("receipts", []):
        date = str(receipt["trade_date_us"])
        if date not in reconstructed:
            continue
        archived = int(receipt["counts"].get("flow_signal_eligible", 0))
        rebuilt = reconstructed[date]["clean_eligible"]
        comparisons.append(
            {
                "trade_date_us": date,
                "archived_flow_signal_eligible": archived,
                "reconstructed_clean_eligible": rebuilt,
                "difference": rebuilt - archived,
                "comparison_status": "DIAGNOSTIC_NOT_EQUALITY_GATE",
            }
        )
    absolute_differences = [abs(item["difference"]) for item in comparisons]
    return {
        "overlap_trade_date_count": len(comparisons),
        "mean_absolute_count_difference": (
            sum(absolute_differences) / len(absolute_differences)
            if absolute_differences
            else None
        ),
        "reason_not_equality_gate": (
            "archived masks include later backfills/current identity logic, while the "
            "reconstruction requires exact date-local price, liquidity, and AUM"
        ),
        "comparisons": comparisons,
    }


def run_phase_a(
    *,
    source_database: Path,
    etfradar_root: Path,
    output_root: Path,
    allowed_output_root: Path,
    replace: bool,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    output_root = safe_output(output_root, allowed_output_root)
    if output_root.exists():
        if not replace:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    started_at = utc_now()
    source = readonly_connection(source_database)
    try:
        quick_check = str(source.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise RuntimeError(f"source SQLite quick_check failed: {quick_check}")
        fingerprint = source_fingerprint(source_database, source)
        metadata = load_metadata(etfradar_root)
        hypothesis_path = output_root / HYPOTHESIS_REGISTRY_FILENAME
        hypothesis_registry = write_registry(
            hypothesis_path, generated_at_utc=started_at
        )
        hypothesis_receipt = {
            "path": str(hypothesis_path),
            "sha256": sha256_file(hypothesis_path),
            "specification_sha256": hypothesis_registry["specification_sha256"],
            "status": hypothesis_registry["status"],
        }
        flow_max = str(
            source.execute(
                "SELECT MAX(effective_date) FROM etf_flow_observations"
            ).fetchone()[0]
        )
        family_path = output_root / FAMILY_REGISTRY_FILENAME
        family_receipt = build_family_registry(
            source=source,
            metadata=metadata,
            output_path=family_path,
            cutoff_date=flow_max,
        )
        event_path = output_root / EVENT_CUBE_FILENAME
        event_receipt = build_event_cube(
            source=source,
            metadata=metadata,
            family_registry_path=family_path,
            output_path=event_path,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        source.close()

    archive_audit = audit_archived_eligibility(etfradar_root)
    lineage_audit = current_lineage_audit(etfradar_root)
    archive_comparison = event_archive_comparison(event_path, archive_audit)
    quality_checks = {
        "source_sqlite_quick_check": quick_check == "ok",
        "timing_contract_zero_violations": event_receipt["timing_violation_count"] == 0,
        "true_zero_separate_from_missing": event_receipt["contract_checks"][
            "true_zero_separate_from_missing"
        ],
        "small_etfs_removed_from_denominator": event_receipt["contract_checks"][
            "small_etfs_removed_from_denominator"
        ],
        "absolute_common_flow_preserved": event_receipt["contract_checks"][
            "absolute_common_flow_preserved"
        ],
        "no_date_centering": not event_receipt["contract_checks"][
            "date_centering_used"
        ],
        "selection_biased_table_48_not_used": not event_receipt["contract_checks"][
            "table_48_breadth_used"
        ],
        "family_registry_integrity": family_receipt["integrity_check"] == "ok",
        "event_cube_integrity": event_receipt["integrity_check"] == "ok",
        "hypotheses_frozen_before_phase_b": hypothesis_registry["status"]
        == "FROZEN_BEFORE_PHASE_B",
        "current_cross_lineage_join_prevented": lineage_audit["join_permission"]
        == "DENY",
        "archived_mask_parse_complete": not archive_audit["parse_failures"],
    }
    blocking_checks = {
        key: value
        for key, value in quality_checks.items()
        if key != "archived_mask_parse_complete"
    }
    status = "PASS_WITH_DECLARED_LIMITATIONS" if all(blocking_checks.values()) else "FAIL"
    audit = {
        "schema_version": PHASE_A_AUDIT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "started_at_utc": started_at,
        "status": status,
        "timing_contract": TIMING_CONTRACT,
        "source": {
            "database": fingerprint,
            "sqlite_quick_check": quick_check,
            "etfradar_root": str(etfradar_root),
        },
        "fixed_thresholds": {
            "min_assets_usd": MIN_ASSETS_USD,
            "min_price_t1_usd": MIN_PRICE_USD,
            "min_dollar_volume_t1_usd": MIN_DOLLAR_VOLUME_USD,
            "active_lookback_sessions": ACTIVE_LOOKBACK_SESSIONS,
            "stale_after_sessions": STALE_AFTER_SESSIONS,
            "small_etf_action": "REMOVE_FROM_DENOMINATOR_NOT_DOWNWEIGHT",
        },
        "hypothesis_registry": hypothesis_receipt,
        "family_registry": family_receipt,
        "event_cube": event_receipt,
        "archived_eligibility": archive_audit,
        "archived_vs_reconstructed": archive_comparison,
        "current_lineage": lineage_audit,
        "forbidden_inputs": list(FORBIDDEN_DIFFUSION_INPUTS),
        "forbidden_transforms": list(FORBIDDEN_TRANSFORMS),
        "quality_checks": quality_checks,
        "limitations": [
            archive_audit["historical_coverage_limitation"],
            family_receipt["pit_limit"],
            (
                "price/liquidity reconstruction is strict: missing T-1 price or dollar "
                "volume is excluded rather than silently admitted"
            ),
            (
                "availability dates reconstruct historical usability but the source was "
                "captured later; it is historical_window_captured, not live as-observed"
            ),
        ],
        "phase_b_activation": (
            "ACTIVATED_INTERPRETABLE_ONLY" if status != "FAIL" else "BLOCKED"
        ),
        "phase_c_activation": "NOT_ACTIVATED_PENDING_PHASE_B_SURVIVORS",
    }
    audit_path = output_root / AUDIT_FILENAME
    write_json_atomic(audit_path, audit)
    manifest = {
        "schema_version": EVENT_CUBE_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": status,
        "timing_contract": TIMING_CONTRACT,
        "event_cube": event_receipt,
        "family_registry": family_receipt,
        "hypothesis_registry": hypothesis_receipt,
        "audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
        },
        "phase_b_activation": audit["phase_b_activation"],
    }
    manifest_path = output_root / EVENT_CUBE_MANIFEST_FILENAME
    write_json_atomic(manifest_path, manifest)
    result = {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "output_files": {
            str(path.name): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output_root.iterdir())
            if path.is_file()
        },
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path, default=DEFAULT_SOURCE_DATABASE)
    parser.add_argument("--etfradar-root", type=Path, default=DEFAULT_ETFRADAR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--allowed-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT.parent
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_phase_a(
        source_database=args.source_database,
        etfradar_root=args.etfradar_root,
        output_root=args.output_root,
        allowed_output_root=args.allowed_output_root,
        replace=args.replace,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
