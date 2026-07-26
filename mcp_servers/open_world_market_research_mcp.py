#!/usr/bin/env python3
"""Card-scoped open-world market research MCP.

The market Role Shell remains the analysis owner. This server preserves the
question-verification packet, manages hypotheses and a bounded search frontier,
collects public-web leads, accepts evidence from other market MCPs, and decides
whether the research loop has enough independent evidence to stop.

It never places orders, reads accounts, changes schedulers, or turns search
snippets into confirmed facts.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests


SERVER_NAME = "open-world-market-research"
SERVER_VERSION = "0.1.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"
DEFAULT_DB_PATH = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/"
    "OPEN_WORLD_MARKET_RESEARCH/research.sqlite3"
)
DEFAULT_SEARCH_URL = "https://html.duckduckgo.com/html/"
MAX_TEXT = 4000

JsonDict = dict[str, Any]
ToolHandler = Callable[[JsonDict], JsonDict]

VERIFICATION_STATES = {
    "INSUFFICIENT",
    "CONTRADICTORY",
    "NOVEL_DISCOVERY",
    "FORCED_EXPANSION",
}
EVIDENCE_STANCES = {"support", "challenge", "neutral"}
CONFIRMED_STATES = {"CONFIRMED"}
ALLOWED_SOURCE_STATES = {
    "CONFIRMED",
    "PARTIAL_LIMIT",
    "NOT_DUE",
    "EOD_ONLY",
    "ESTIMATE_ONLY",
    "UNVERIFIED_CONTRACT",
    "UNVERIFIED_UNIT",
    "NOT_APPLICABLE",
    "INTENTIONAL_NOT_USED",
    "PAUSED",
    "RECOVERING",
    "FAILED",
    "SEARCH_LEAD",
}


class ToolInputError(ValueError):
    """Raised for a malformed MCP tool request."""


class ResearchError(RuntimeError):
    """Raised when a research session cannot progress."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JsonDict
    handler: ToolHandler

    def as_mcp_tool(self) -> JsonDict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def database_path() -> Path:
    raw = os.getenv("OPEN_WORLD_MARKET_RESEARCH_DB_PATH", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def compact_text(value: Any, limit: int = MAX_TEXT) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def get_str(
    args: Mapping[str, Any],
    name: str,
    default: Optional[str] = None,
) -> str:
    value = args.get(name, default)
    if value is None:
        raise ToolInputError(f"Missing required argument: {name}")
    if not isinstance(value, str):
        raise ToolInputError(f"Argument {name} must be a string")
    value = compact_text(value)
    if not value:
        raise ToolInputError(f"Argument {name} cannot be empty")
    return value


def get_optional_str(args: Mapping[str, Any], name: str) -> Optional[str]:
    value = args.get(name)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ToolInputError(f"Argument {name} must be a string")
    return compact_text(value)


def get_int(
    args: Mapping[str, Any],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError(f"Argument {name} must be an integer")
    if not minimum <= value <= maximum:
        raise ToolInputError(
            f"Argument {name} must be between {minimum} and {maximum}"
        )
    return value


def get_string_list(
    args: Mapping[str, Any],
    name: str,
    *,
    required: bool = False,
    maximum: int = 50,
) -> list[str]:
    raw = args.get(name)
    if raw is None:
        if required:
            raise ToolInputError(f"Missing required argument: {name}")
        return []
    if not isinstance(raw, list):
        raise ToolInputError(f"Argument {name} must be an array")
    values = [compact_text(item, 1000) for item in raw]
    values = [item for item in values if item]
    if required and not values:
        raise ToolInputError(f"Argument {name} cannot be empty")
    return values[:maximum]


def base_schema(
    properties: JsonDict,
    required: Optional[list[str]] = None,
) -> JsonDict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


SCHEMA_SQL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS research_sessions (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    as_of TEXT,
    market_scope TEXT,
    verification_state TEXT NOT NULL,
    verification_json TEXT NOT NULL,
    status TEXT NOT NULL,
    max_rounds INTEGER NOT NULL,
    rounds INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_hypotheses (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES research_sessions(id),
    statement TEXT NOT NULL,
    falsifier TEXT,
    origin TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, statement)
);
CREATE TABLE IF NOT EXISTS research_frontier (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES research_sessions(id),
    query TEXT NOT NULL,
    rationale TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, query)
);
CREATE TABLE IF NOT EXISTS research_evidence (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES research_sessions(id),
    hypothesis_id TEXT REFERENCES research_hypotheses(id),
    stance TEXT NOT NULL,
    source_family TEXT NOT NULL,
    source TEXT NOT NULL,
    url TEXT,
    title TEXT,
    excerpt TEXT NOT NULL,
    source_status TEXT NOT NULL,
    observed_at TEXT,
    as_of TEXT,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, content_sha256)
);
CREATE TABLE IF NOT EXISTS research_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES research_sessions(id),
    query TEXT NOT NULL,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_hypotheses_session
    ON research_hypotheses(session_id);
CREATE INDEX IF NOT EXISTS idx_research_frontier_session
    ON research_frontier(session_id,status,priority);
CREATE INDEX IF NOT EXISTS idx_research_evidence_session
    ON research_evidence(session_id,hypothesis_id,source_status);
"""


def connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def require_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM research_sessions WHERE id=?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise ToolInputError(f"Unknown research session: {session_id}")
    return row


def split_claims(question: str) -> list[str]:
    parts = re.split(r"[?!.;\n]+|(?:\s+(?:그리고|하지만|또는|vs\.?|versus)\s+)", question)
    claims = [compact_text(part, 500) for part in parts]
    claims = [part for part in claims if len(part) >= 4]
    return claims[:8] or [question]


def seed_hypotheses(question: str, claims: list[str]) -> list[dict[str, str]]:
    primary = claims[0] if claims else question
    return [
        {
            "statement": f"핵심 주장: {primary}",
            "falsifier": "독립된 최신 원천 데이터가 핵심 주장과 반대 방향을 보인다.",
            "origin": "question_verification",
        },
        {
            "statement": "대체 설명이 핵심 주장보다 관측된 시장 움직임을 더 잘 설명한다.",
            "falsifier": "가격·수급·공시·거시의 독립 증거가 핵심 주장으로 수렴한다.",
            "origin": "counter_hypothesis",
        },
        {
            "statement": "관측된 움직임은 새로운 펀더멘털보다 포지셔닝·유동성·시점 효과다.",
            "falsifier": "공식 이벤트나 실적·가이던스 변화가 동일 시점에 확인된다.",
            "origin": "market_microstructure_hypothesis",
        },
    ]


def normalize_hypotheses(
    raw: Any,
    *,
    question: str,
    claims: list[str],
) -> list[dict[str, str]]:
    if raw is None:
        return seed_hypotheses(question, claims)
    if not isinstance(raw, list):
        raise ToolInputError("Argument hypotheses must be an array")
    result: list[dict[str, str]] = []
    for item in raw[:20]:
        if isinstance(item, str):
            statement = compact_text(item, 1000)
            falsifier = ""
            origin = "market_role_shell"
        elif isinstance(item, dict):
            statement = compact_text(item.get("statement"), 1000)
            falsifier = compact_text(item.get("falsifier"), 1000)
            origin = compact_text(item.get("origin") or "market_role_shell", 100)
        else:
            raise ToolInputError("Each hypothesis must be a string or object")
        if statement:
            result.append(
                {
                    "statement": statement,
                    "falsifier": falsifier,
                    "origin": origin,
                }
            )
    return result or seed_hypotheses(question, claims)


def frontier_queries(
    question: str,
    claims: list[str],
    hypotheses: list[dict[str, str]],
) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = [
        (question, "원 질문의 열린세계 탐색", 100),
        (f"{question} official filing investor relations", "공식 원문 탐색", 95),
        (f"{question} risk counter evidence", "반증·리스크 탐색", 90),
        (f"{question} competitor supply chain policy", "관계·대체 설명 탐색", 80),
    ]
    for claim in claims[:4]:
        rows.append((f"{claim} primary source data", "주장별 원천 검증", 88))
    for hypothesis in hypotheses[:4]:
        falsifier = hypothesis.get("falsifier")
        if falsifier:
            rows.append((falsifier, "가설 반증 조건 탐색", 92))
    seen: set[str] = set()
    unique: list[tuple[str, str, int]] = []
    for query, rationale, priority in rows:
        key = query.casefold()
        if key not in seen:
            seen.add(key)
            unique.append((compact_text(query, 1000), rationale, priority))
    return unique[:16]


def insert_hypotheses(
    conn: sqlite3.Connection,
    session_id: str,
    hypotheses: Iterable[Mapping[str, Any]],
) -> list[str]:
    ids: list[str] = []
    now = utc_now()
    for hypothesis in hypotheses:
        statement = compact_text(hypothesis.get("statement"), 1000)
        if not statement:
            continue
        hypothesis_id = stable_id("hyp", session_id, statement.casefold())
        conn.execute(
            "INSERT OR IGNORE INTO research_hypotheses "
            "(id,session_id,statement,falsifier,origin,status,score,created_at) "
            "VALUES(?,?,?,?,?,'unresolved',0,?)",
            (
                hypothesis_id,
                session_id,
                statement,
                compact_text(hypothesis.get("falsifier"), 1000) or None,
                compact_text(hypothesis.get("origin") or "open_world", 100),
                now,
            ),
        )
        ids.append(hypothesis_id)
    return ids


def insert_frontier(
    conn: sqlite3.Connection,
    session_id: str,
    rows: Iterable[tuple[str, str, int]],
) -> list[str]:
    ids: list[str] = []
    now = utc_now()
    for query, rationale, priority in rows:
        query = compact_text(query, 1000)
        if not query:
            continue
        frontier_id = stable_id("frontier", session_id, query.casefold())
        conn.execute(
            "INSERT OR IGNORE INTO research_frontier "
            "(id,session_id,query,rationale,status,priority,created_at,updated_at) "
            "VALUES(?,?,?,?,'pending',?,?,?)",
            (
                frontier_id,
                session_id,
                query,
                compact_text(rationale, 500),
                int(priority),
                now,
                now,
            ),
        )
        ids.append(frontier_id)
    return ids


def tool_health(_: JsonDict) -> JsonDict:
    with connect() as conn:
        counts = {
            "sessions": conn.execute(
                "SELECT COUNT(*) FROM research_sessions"
            ).fetchone()[0],
            "evidence": conn.execute(
                "SELECT COUNT(*) FROM research_evidence"
            ).fetchone()[0],
        }
    return {
        "ok": True,
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "database": str(database_path()),
        "search_backend": (
            "searxng"
            if os.getenv("OPEN_WORLD_RESEARCH_SEARCH_URL", "").strip()
            or os.getenv("SEARXNG_URL", "").strip()
            else "duckduckgo_html"
        ),
        "safety": {
            "trade_writes": False,
            "account_access": False,
            "scheduler_writes": False,
            "search_snippets_are_confirmed_evidence": False,
        },
        "counts": counts,
    }


def tool_research_start(args: JsonDict) -> JsonDict:
    question = get_str(args, "question")
    verification_state = get_str(
        args, "verification_state", "INSUFFICIENT"
    ).upper()
    if verification_state not in VERIFICATION_STATES:
        raise ToolInputError(
            "verification_state must be one of "
            + ", ".join(sorted(VERIFICATION_STATES))
        )
    claims = get_string_list(args, "claims") or split_claims(question)
    hypotheses = normalize_hypotheses(
        args.get("hypotheses"),
        question=question,
        claims=claims,
    )
    max_rounds = get_int(args, "max_rounds", 6, minimum=1, maximum=20)
    as_of = get_optional_str(args, "as_of")
    market_scope = get_optional_str(args, "market_scope") or "global"
    reason = get_optional_str(args, "expansion_reason") or verification_state
    now = utc_now()
    session_id = stable_id(
        "research",
        question.casefold(),
        as_of or "",
        market_scope.casefold(),
        now,
    )
    verification = {
        "schema": "quant.market_question_verification.v1",
        "state": verification_state,
        "claims": claims,
        "expansion_reason": reason,
        "existing_source_families": get_string_list(
            args, "existing_source_families"
        ),
        "requested_by": "market_role_shell",
    }
    with connect() as conn:
        conn.execute(
            "INSERT INTO research_sessions "
            "(id,question,as_of,market_scope,verification_state,"
            "verification_json,status,max_rounds,rounds,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,'researching',?,0,?,?)",
            (
                session_id,
                question,
                as_of,
                market_scope,
                verification_state,
                json.dumps(verification, ensure_ascii=False, sort_keys=True),
                max_rounds,
                now,
                now,
            ),
        )
        hypothesis_ids = insert_hypotheses(conn, session_id, hypotheses)
        frontier_ids = insert_frontier(
            conn,
            session_id,
            frontier_queries(question, claims, hypotheses),
        )
        conn.commit()
    return {
        "schema": "quant.open_world_market_research_session.v1",
        "session_id": session_id,
        "status": "researching",
        "question_verification": verification,
        "open_world_required": True,
        "hypothesis_ids": hypothesis_ids,
        "frontier_ids": frontier_ids,
        "next_action": "search_frontier_then_verify_leads_with_independent_sources",
    }


class DuckResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._anchor: Optional[dict[str, str]] = None
        self._snippet = False
        self._text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
    ) -> None:
        values = {key: value or "" for key, value in attrs}
        classes = set(values.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._anchor = {"url": values.get("href", ""), "title": ""}
            self._text = []
        elif "result__snippet" in classes:
            self._snippet = True
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._anchor is not None or self._snippet:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor is not None:
            self._anchor["title"] = compact_text(" ".join(self._text), 500)
            self.results.append(self._anchor)
            self._anchor = None
            self._text = []
        elif self._snippet:
            snippet = compact_text(" ".join(self._text), 1200)
            if self.results and snippet and not self.results[-1].get("snippet"):
                self.results[-1]["snippet"] = snippet
            self._snippet = False
            self._text = []


def normalize_result_url(value: str) -> str:
    value = html.unescape(value or "")
    parsed = urlparse(value)
    if parsed.netloc.endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    if value.startswith("//"):
        return "https:" + value
    return value


def search_web(query: str, limit: int) -> tuple[str, list[dict[str, str]]]:
    configured = (
        os.getenv("OPEN_WORLD_RESEARCH_SEARCH_URL", "").strip()
        or os.getenv("SEARXNG_URL", "").strip()
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 Chrome/124 Safari/537.36"
        )
    }
    timeout = 20
    if configured:
        response = requests.get(
            configured.rstrip("/") + "/search",
            params={"q": query, "format": "json", "language": "all"},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": compact_text(item.get("title"), 500),
                    "url": compact_text(item.get("url"), 2000),
                    "snippet": compact_text(item.get("content"), 1200),
                }
            )
        return "searxng", results[:limit]
    response = requests.post(
        DEFAULT_SEARCH_URL,
        data={"q": query},
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    parser = DuckResultParser()
    parser.feed(response.text)
    results = [
        {
            **item,
            "url": normalize_result_url(item.get("url", "")),
        }
        for item in parser.results
        if item.get("title") and item.get("url")
    ]
    return "duckduckgo_html", results[:limit]


def next_frontier(
    conn: sqlite3.Connection,
    session_id: str,
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM research_frontier "
        "WHERE session_id=? AND status='pending' "
        "ORDER BY priority DESC,created_at,id LIMIT 1",
        (session_id,),
    ).fetchone()


def insert_search_lead(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    backend: str,
    result: Mapping[str, Any],
) -> Optional[str]:
    title = compact_text(result.get("title"), 500)
    excerpt = compact_text(result.get("snippet") or title, 1200)
    url = compact_text(result.get("url"), 2000)
    if not excerpt or not url:
        return None
    digest = hashlib.sha256(
        f"{backend}\x1f{url}\x1f{excerpt}".encode("utf-8")
    ).hexdigest()
    evidence_id = stable_id("evidence", session_id, digest)
    conn.execute(
        "INSERT OR IGNORE INTO research_evidence "
        "(id,session_id,hypothesis_id,stance,source_family,source,url,title,"
        "excerpt,source_status,observed_at,as_of,content_sha256,created_at) "
        "VALUES(?,?,NULL,'neutral','open_web_search',?,?,?,?,"
        "'SEARCH_LEAD',?,NULL,?,?)",
        (
            evidence_id,
            session_id,
            backend,
            url,
            title,
            excerpt,
            utc_now(),
            digest,
            utc_now(),
        ),
    )
    return evidence_id


def tool_research_search(args: JsonDict) -> JsonDict:
    session_id = get_str(args, "session_id")
    explicit_query = get_optional_str(args, "query")
    max_results = get_int(args, "max_results", 8, minimum=1, maximum=20)
    with connect() as conn:
        session = require_session(conn, session_id)
        if session["status"] not in {"researching", "bounded_limit"}:
            raise ResearchError(
                f"Session {session_id} is not open for search: {session['status']}"
            )
        frontier = next_frontier(conn, session_id)
        if explicit_query:
            query = explicit_query
            frontier_id = stable_id("frontier", session_id, query.casefold())
            insert_frontier(
                conn,
                session_id,
                [(query, "시장 Role Shell 지정 탐색", 100)],
            )
        elif frontier is not None:
            query = str(frontier["query"])
            frontier_id = str(frontier["id"])
        else:
            raise ResearchError("No pending research frontier remains")
        try:
            backend, results = search_web(query, max_results)
            status = "confirmed" if results else "empty"
            error = None
        except Exception as exc:
            backend = (
                "searxng"
                if os.getenv("OPEN_WORLD_RESEARCH_SEARCH_URL", "").strip()
                or os.getenv("SEARXNG_URL", "").strip()
                else "duckduckgo_html"
            )
            results = []
            status = "failed"
            error = compact_text(f"{type(exc).__name__}: {exc}", 1000)
        evidence_ids = [
            item
            for item in (
                insert_search_lead(
                    conn,
                    session_id=session_id,
                    backend=backend,
                    result=result,
                )
                for result in results
            )
            if item
        ]
        now = utc_now()
        conn.execute(
            "UPDATE research_frontier SET status=?,updated_at=? "
            "WHERE id=? AND session_id=?",
            ("searched" if results else status, now, frontier_id, session_id),
        )
        conn.execute(
            "INSERT INTO research_attempts "
            "(session_id,query,backend,status,result_count,error,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                session_id,
                query,
                backend,
                status,
                len(results),
                error,
                now,
            ),
        )
        conn.execute(
            "UPDATE research_sessions SET rounds=rounds+1,updated_at=? WHERE id=?",
            (now, session_id),
        )
        conn.commit()
    return {
        "schema": "quant.open_world_search_result.v1",
        "session_id": session_id,
        "query": query,
        "backend": backend,
        "status": status,
        "result_count": len(results),
        "results": results,
        "evidence_ids": evidence_ids,
        "evidence_status": "SEARCH_LEAD",
        "confirmation_rule": (
            "Search snippets are leads only. Verify with an official, structured, "
            "or independently fetched source before using them as market evidence."
        ),
        "error": error,
    }


def normalize_evidence_item(
    session_id: str,
    item: Mapping[str, Any],
) -> tuple[str, tuple[Any, ...]]:
    hypothesis_id = compact_text(item.get("hypothesis_id"), 100) or None
    stance = compact_text(item.get("stance") or "neutral", 30).casefold()
    if stance not in EVIDENCE_STANCES:
        raise ToolInputError(
            "Evidence stance must be support, challenge, or neutral"
        )
    source_family = compact_text(item.get("source_family"), 200)
    source = compact_text(item.get("source"), 500)
    excerpt = compact_text(item.get("excerpt"), 4000)
    if not source_family or not source or not excerpt:
        raise ToolInputError(
            "Each evidence item requires source_family, source, and excerpt"
        )
    source_status = compact_text(
        item.get("source_status") or "UNVERIFIED_CONTRACT", 60
    ).upper()
    if source_status not in ALLOWED_SOURCE_STATES:
        raise ToolInputError(
            f"Unsupported evidence source_status: {source_status}"
        )
    url = compact_text(item.get("url"), 2000) or None
    title = compact_text(item.get("title"), 500) or None
    observed_at = compact_text(item.get("observed_at"), 100) or utc_now()
    as_of = compact_text(item.get("as_of"), 100) or None
    digest = hashlib.sha256(
        "\x1f".join(
            [
                hypothesis_id or "",
                stance,
                source_family,
                source,
                url or "",
                excerpt,
                source_status,
                as_of or "",
            ]
        ).encode("utf-8")
    ).hexdigest()
    evidence_id = stable_id("evidence", session_id, digest)
    values = (
        evidence_id,
        session_id,
        hypothesis_id,
        stance,
        source_family,
        source,
        url,
        title,
        excerpt,
        source_status,
        observed_at,
        as_of,
        digest,
        utc_now(),
    )
    return evidence_id, values


def tool_evidence_add(args: JsonDict) -> JsonDict:
    session_id = get_str(args, "session_id")
    items = args.get("items")
    if not isinstance(items, list) or not items:
        raise ToolInputError("Argument items must be a non-empty array")
    if len(items) > 100:
        raise ToolInputError("Argument items cannot exceed 100 evidence rows")
    added: list[str] = []
    duplicates: list[str] = []
    with connect() as conn:
        require_session(conn, session_id)
        valid_hypotheses = {
            str(row["id"])
            for row in conn.execute(
                "SELECT id FROM research_hypotheses WHERE session_id=?",
                (session_id,),
            )
        }
        for raw in items:
            if not isinstance(raw, dict):
                raise ToolInputError("Each evidence item must be an object")
            evidence_id, values = normalize_evidence_item(session_id, raw)
            hypothesis_id = values[2]
            if hypothesis_id and hypothesis_id not in valid_hypotheses:
                raise ToolInputError(
                    f"Evidence references unknown hypothesis: {hypothesis_id}"
                )
            cursor = conn.execute(
                "INSERT OR IGNORE INTO research_evidence "
                "(id,session_id,hypothesis_id,stance,source_family,source,url,"
                "title,excerpt,source_status,observed_at,as_of,content_sha256,"
                "created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values,
            )
            (added if cursor.rowcount else duplicates).append(evidence_id)
        conn.execute(
            "UPDATE research_sessions SET updated_at=? WHERE id=?",
            (utc_now(), session_id),
        )
        conn.commit()
    return {
        "schema": "quant.open_world_market_evidence_add.v1",
        "session_id": session_id,
        "added_ids": added,
        "duplicate_ids": duplicates,
        "added_count": len(added),
        "duplicate_count": len(duplicates),
    }


def add_evaluation_frontier(
    conn: sqlite3.Connection,
    session_id: str,
    hypothesis: Mapping[str, Any],
) -> list[str]:
    statement = compact_text(hypothesis.get("statement"), 1000)
    falsifier = compact_text(hypothesis.get("falsifier"), 1000)
    rows = [
        (
            f"{statement} latest primary source evidence",
            "미해결 가설의 지지 증거",
            85,
        ),
        (
            falsifier or f"{statement} counter evidence alternative explanation",
            "미해결 가설의 반증 증거",
            95,
        ),
    ]
    return insert_frontier(conn, session_id, rows)


def tool_research_evaluate(args: JsonDict) -> JsonDict:
    session_id = get_str(args, "session_id")
    new_hypotheses = normalize_hypotheses(
        args.get("new_hypotheses") or [],
        question="",
        claims=[],
    ) if args.get("new_hypotheses") else []
    with connect() as conn:
        session = require_session(conn, session_id)
        new_hypothesis_ids = insert_hypotheses(
            conn, session_id, new_hypotheses
        )
        hypotheses = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM research_hypotheses "
                "WHERE session_id=? ORDER BY created_at,id",
                (session_id,),
            )
        ]
        evaluations: list[dict[str, Any]] = []
        frontier_ids: list[str] = []
        for hypothesis in hypotheses:
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT stance,source_family,source_status FROM research_evidence "
                    "WHERE session_id=? AND hypothesis_id=?",
                    (session_id, hypothesis["id"]),
                )
            ]
            confirmed = [
                row for row in rows if row["source_status"] in CONFIRMED_STATES
            ]
            support_families = {
                row["source_family"]
                for row in confirmed
                if row["stance"] == "support"
            }
            challenge_families = {
                row["source_family"]
                for row in confirmed
                if row["stance"] == "challenge"
            }
            if support_families and challenge_families:
                status = "mixed"
            elif len(support_families) >= 2:
                status = "supported"
            elif len(challenge_families) >= 2:
                status = "challenged"
            else:
                status = "unresolved"
            score = max(
                -1.0,
                min(
                    1.0,
                    (len(support_families) - len(challenge_families)) / 2.0,
                ),
            )
            conn.execute(
                "UPDATE research_hypotheses SET status=?,score=? WHERE id=?",
                (status, score, hypothesis["id"]),
            )
            if status in {"mixed", "unresolved"}:
                frontier_ids.extend(
                    add_evaluation_frontier(conn, session_id, hypothesis)
                )
            evaluations.append(
                {
                    "hypothesis_id": hypothesis["id"],
                    "statement": hypothesis["statement"],
                    "status": status,
                    "score": score,
                    "confirmed_support_families": sorted(support_families),
                    "confirmed_challenge_families": sorted(challenge_families),
                    "all_evidence_count": len(rows),
                }
            )
        rounds = int(session["rounds"])
        max_rounds = int(session["max_rounds"])
        all_resolved = bool(evaluations) and all(
            row["status"] in {"supported", "challenged"} for row in evaluations
        )
        if all_resolved:
            status = "sufficient"
            stop_reason = "independent_confirmed_sources_resolved_all_hypotheses"
            open_world_required = False
        elif rounds >= max_rounds:
            status = "bounded_limit"
            stop_reason = "max_rounds_reached_with_unresolved_or_mixed_hypotheses"
            open_world_required = False
        else:
            status = "researching"
            stop_reason = "continue_high_value_frontier"
            open_world_required = True
        now = utc_now()
        conn.execute(
            "UPDATE research_sessions SET status=?,updated_at=? WHERE id=?",
            (status, now, session_id),
        )
        pending = conn.execute(
            "SELECT COUNT(*) FROM research_frontier "
            "WHERE session_id=? AND status='pending'",
            (session_id,),
        ).fetchone()[0]
        conn.commit()
    return {
        "schema": "quant.open_world_market_research_evaluation.v1",
        "session_id": session_id,
        "status": status,
        "open_world_required": open_world_required,
        "stop_reason": stop_reason,
        "rounds": rounds,
        "max_rounds": max_rounds,
        "hypotheses": evaluations,
        "new_hypothesis_ids": new_hypothesis_ids,
        "new_frontier_ids": sorted(set(frontier_ids)),
        "pending_frontier_count": int(pending),
    }


def tool_research_export(args: JsonDict) -> JsonDict:
    session_id = get_str(args, "session_id")
    max_evidence = get_int(
        args, "max_evidence", 500, minimum=1, maximum=2000
    )
    with connect() as conn:
        session = dict(require_session(conn, session_id))
        session["question_verification"] = json.loads(
            session.pop("verification_json")
        )
        hypotheses = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM research_hypotheses "
                "WHERE session_id=? ORDER BY created_at,id",
                (session_id,),
            )
        ]
        frontier = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM research_frontier "
                "WHERE session_id=? ORDER BY priority DESC,created_at,id",
                (session_id,),
            )
        ]
        evidence = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM research_evidence "
                "WHERE session_id=? ORDER BY created_at,id LIMIT ?",
                (session_id, max_evidence),
            )
        ]
        attempts = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM research_attempts "
                "WHERE session_id=? ORDER BY id",
                (session_id,),
            )
        ]
        total_evidence = conn.execute(
            "SELECT COUNT(*) FROM research_evidence WHERE session_id=?",
            (session_id,),
        ).fetchone()[0]
    return {
        "schema": "quant.open_world_market_research_export.v1",
        "session": session,
        "hypotheses": hypotheses,
        "frontier": frontier,
        "evidence": evidence,
        "attempts": attempts,
        "completeness": {
            "total_evidence": int(total_evidence),
            "returned_evidence": len(evidence),
            "truncated": int(total_evidence) > len(evidence),
        },
    }


def object_array_schema(properties: JsonDict, required: list[str]) -> JsonDict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOLS: dict[str, ToolSpec] = {
    "market_research_health": ToolSpec(
        "market_research_health",
        "Report open-world research MCP health and safety boundaries.",
        base_schema({}),
        tool_health,
    ),
    "market_research_start": ToolSpec(
        "market_research_start",
        (
            "Start a bounded open-world market research session after the market "
            "Role Shell has verified the question and decided existing data is "
            "insufficient, contradictory, or likely to yield novel discovery."
        ),
        base_schema(
            {
                "question": {"type": "string"},
                "verification_state": {
                    "type": "string",
                    "enum": sorted(VERIFICATION_STATES),
                },
                "claims": {"type": "array", "items": {"type": "string"}},
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "statement": {"type": "string"},
                                    "falsifier": {"type": "string"},
                                    "origin": {"type": "string"},
                                },
                                "required": ["statement"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                },
                "existing_source_families": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "expansion_reason": {"type": "string"},
                "as_of": {"type": "string"},
                "market_scope": {"type": "string"},
                "max_rounds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            ["question", "verification_state"],
        ),
        tool_research_start,
    ),
    "market_research_search": ToolSpec(
        "market_research_search",
        (
            "Search one high-value frontier query. Returned snippets are SEARCH_LEAD "
            "only and require independent confirmation."
        ),
        base_schema(
            {
                "session_id": {"type": "string"},
                "query": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            ["session_id"],
        ),
        tool_research_search,
    ),
    "market_research_add_evidence": ToolSpec(
        "market_research_add_evidence",
        (
            "Attach provenance-rich evidence from FMP, Massive, Topstep, KIS/KRX/"
            "DART, Barchart, official filings, public web, or other verified sources."
        ),
        base_schema(
            {
                "session_id": {"type": "string"},
                "items": object_array_schema(
                    {
                        "hypothesis_id": {"type": "string"},
                        "stance": {
                            "type": "string",
                            "enum": sorted(EVIDENCE_STANCES),
                        },
                        "source_family": {"type": "string"},
                        "source": {"type": "string"},
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "excerpt": {"type": "string"},
                        "source_status": {
                            "type": "string",
                            "enum": sorted(ALLOWED_SOURCE_STATES),
                        },
                        "observed_at": {"type": "string"},
                        "as_of": {"type": "string"},
                    },
                    ["stance", "source_family", "source", "excerpt", "source_status"],
                ),
            },
            ["session_id", "items"],
        ),
        tool_evidence_add,
    ),
    "market_research_evaluate": ToolSpec(
        "market_research_evaluate",
        (
            "Evaluate hypotheses against independent confirmed evidence, add newly "
            "discovered hypotheses, and return the next frontier or bounded stop reason."
        ),
        base_schema(
            {
                "session_id": {"type": "string"},
                "new_hypotheses": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "statement": {"type": "string"},
                                    "falsifier": {"type": "string"},
                                    "origin": {"type": "string"},
                                },
                                "required": ["statement"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                },
            },
            ["session_id"],
        ),
        tool_research_evaluate,
    ),
    "market_research_export": ToolSpec(
        "market_research_export",
        "Export the complete question, hypotheses, frontier, evidence, and attempt trace.",
        base_schema(
            {
                "session_id": {"type": "string"},
                "max_evidence": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2000,
                },
            },
            ["session_id"],
        ),
        tool_research_export,
    ),
}


def success_response(message_id: Any, result: Any) -> JsonDict:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def error_response(message_id: Any, code: int, message: str) -> JsonDict:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": compact_text(message, 2000)},
    }


def tool_result(payload: Any, *, is_error: bool = False) -> JsonDict:
    return {
        "content": [{"type": "text", "text": json_text(payload)}],
        "isError": is_error,
    }


def handle_initialize(message_id: Any, params: JsonDict) -> JsonDict:
    protocol = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
    return success_response(
        message_id,
        {
            "protocolVersion": protocol,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        },
    )


def handle_tools_call(message_id: Any, params: JsonDict) -> JsonDict:
    name = params.get("name")
    if not isinstance(name, str) or not name:
        return error_response(message_id, -32602, "tools/call requires a tool name")
    spec = TOOLS.get(name)
    if spec is None:
        return error_response(message_id, -32602, f"Unknown tool: {name}")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return error_response(message_id, -32602, "Tool arguments must be an object")
    try:
        payload = spec.handler(arguments)
        return success_response(message_id, tool_result(payload))
    except (ToolInputError, ResearchError) as exc:
        return success_response(
            message_id,
            tool_result(
                {
                    "error": compact_text(exc),
                    "type": (
                        "invalid_arguments"
                        if isinstance(exc, ToolInputError)
                        else "research_state_error"
                    ),
                },
                is_error=True,
            ),
        )
    except Exception as exc:
        return success_response(
            message_id,
            tool_result(
                {
                    "error": compact_text(f"{type(exc).__name__}: {exc}"),
                    "type": "internal_error",
                },
                is_error=True,
            ),
        )


def handle_message(message: JsonDict) -> Optional[JsonDict]:
    message_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return error_response(message_id, -32602, "params must be an object")
    if (
        message_id is None
        and isinstance(method, str)
        and method.startswith("notifications/")
    ):
        return None
    if method == "initialize":
        return handle_initialize(message_id, params)
    if method == "ping":
        return success_response(message_id, {})
    if method == "tools/list":
        return success_response(
            message_id,
            {"tools": [TOOLS[name].as_mcp_tool() for name in sorted(TOOLS)]},
        )
    if method == "tools/call":
        return handle_tools_call(message_id, params)
    if method == "resources/list":
        return success_response(message_id, {"resources": []})
    if method == "prompts/list":
        return success_response(message_id, {"prompts": []})
    if message_id is None:
        return None
    return error_response(message_id, -32601, f"Method not found: {method}")


def serve_stdio() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            response = error_response(None, -32700, f"Parse error: {exc}")
        else:
            response = (
                handle_message(message)
                if isinstance(message, dict)
                else error_response(None, -32600, "Invalid JSON-RPC message")
            )
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


def run_self_test() -> int:
    problems: list[str] = []
    for name, spec in TOOLS.items():
        if spec.name != name:
            problems.append(f"Tool key/name mismatch: {name} != {spec.name}")
        schema = spec.as_mcp_tool().get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            problems.append(f"Tool has invalid schema: {name}")
    payload = {
        "ok": not problems,
        "server": SERVER_NAME,
        "tool_count": len(TOOLS),
        "tools": sorted(TOOLS),
        "problems": problems,
    }
    print(json_text(payload))
    return 0 if not problems else 1


def list_tools() -> int:
    print(
        json_text(
            [
                {"name": TOOLS[name].name, "description": TOOLS[name].description}
                for name in sorted(TOOLS)
            ]
        )
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open-world market research MCP server"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--list-tools", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.list_tools:
        return list_tools()
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
