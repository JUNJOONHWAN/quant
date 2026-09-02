from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mcp_servers import open_world_market_research_mcp as research


class OpenWorldMarketResearchMcpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "research.sqlite3"
        self.env = patch.dict(
            os.environ,
            {
                "OPEN_WORLD_MARKET_RESEARCH_DB_PATH": str(self.db_path),
                "OPEN_WORLD_RESEARCH_SEARCH_URL": "",
                "SEARXNG_URL": "",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tempdir.cleanup()

    def start_session(self, *, max_rounds: int = 6) -> dict:
        return research.tool_research_start(
            {
                "question": "NVDA 상승은 실적 때문인가 공급 부족 때문인가?",
                "verification_state": "CONTRADICTORY",
                "claims": ["NVDA 상승 원인", "실적과 공급 부족의 상대 영향"],
                "hypotheses": [
                    {
                        "statement": "실적 상향이 주된 상승 원인이다.",
                        "falsifier": "실적 변화 없이 공급 제약 뉴스만 가격을 설명한다.",
                    }
                ],
                "existing_source_families": ["fmp_quote"],
                "max_rounds": max_rounds,
            }
        )

    def test_tools_are_structured_and_do_not_expose_trade_actions(self):
        self.assertEqual(
            set(research.TOOLS),
            {
                "market_research_health",
                "market_research_start",
                "market_research_search",
                "market_research_add_evidence",
                "market_research_evaluate",
                "market_research_export",
            },
        )
        text = json.dumps(
            [spec.as_mcp_tool() for spec in research.TOOLS.values()]
        ).casefold()
        self.assertNotIn("place_order", text)
        self.assertNotIn("account_balance", text)

    def test_start_preserves_question_verification_and_builds_frontier(self):
        payload = self.start_session()
        self.assertTrue(payload["open_world_required"])
        self.assertEqual(
            payload["question_verification"]["state"], "CONTRADICTORY"
        )
        self.assertEqual(len(payload["hypothesis_ids"]), 1)
        self.assertGreaterEqual(len(payload["frontier_ids"]), 4)
        exported = research.tool_research_export(
            {"session_id": payload["session_id"]}
        )
        self.assertEqual(
            exported["session"]["question_verification"]["requested_by"],
            "market_role_shell",
        )
        self.assertEqual(exported["session"]["status"], "researching")

    def test_invalid_verification_state_is_rejected(self):
        with self.assertRaises(research.ToolInputError):
            research.tool_research_start(
                {
                    "question": "QQQ?",
                    "verification_state": "MAYBE",
                }
            )

    def test_search_results_remain_leads_not_confirmed_evidence(self):
        session = self.start_session()
        response = Mock()
        response.raise_for_status.return_value = None
        response.text = """
        <div class="result">
          <a class="result__a"
             href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ffiling">
             Example filing
          </a>
          <a class="result__snippet">Primary-source candidate snippet</a>
        </div>
        """
        with patch.object(research.requests, "post", return_value=response):
            payload = research.tool_research_search(
                {
                    "session_id": session["session_id"],
                    "query": "NVDA official filing",
                    "max_results": 5,
                }
            )
        self.assertEqual(payload["status"], "confirmed")
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["evidence_status"], "SEARCH_LEAD")
        exported = research.tool_research_export(
            {"session_id": session["session_id"]}
        )
        self.assertEqual(exported["evidence"][0]["source_status"], "SEARCH_LEAD")
        self.assertEqual(
            exported["evidence"][0]["url"], "https://example.com/filing"
        )

    def test_searxng_json_backend_is_supported(self):
        session = self.start_session()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {
                    "title": "Official release",
                    "url": "https://example.org/release",
                    "content": "Release evidence lead",
                }
            ]
        }
        with patch.dict(
            os.environ,
            {"OPEN_WORLD_RESEARCH_SEARCH_URL": "http://search.local"},
            clear=False,
        ), patch.object(research.requests, "get", return_value=response) as get:
            payload = research.tool_research_search(
                {
                    "session_id": session["session_id"],
                    "query": "NVDA release",
                }
            )
        self.assertEqual(payload["backend"], "searxng")
        get.assert_called_once()
        self.assertEqual(payload["results"][0]["title"], "Official release")

    def test_two_independent_confirmed_families_resolve_hypothesis(self):
        session = self.start_session()
        hypothesis_id = session["hypothesis_ids"][0]
        added = research.tool_evidence_add(
            {
                "session_id": session["session_id"],
                "items": [
                    {
                        "hypothesis_id": hypothesis_id,
                        "stance": "support",
                        "source_family": "sec_filing",
                        "source": "SEC 10-Q",
                        "url": "https://sec.example/10q",
                        "excerpt": "Revenue guidance increased.",
                        "source_status": "CONFIRMED",
                    },
                    {
                        "hypothesis_id": hypothesis_id,
                        "stance": "support",
                        "source_family": "earnings_call",
                        "source": "Company earnings call",
                        "url": "https://ir.example/call",
                        "excerpt": "Management raised the outlook.",
                        "source_status": "CONFIRMED",
                    },
                ],
            }
        )
        self.assertEqual(added["added_count"], 2)
        evaluated = research.tool_research_evaluate(
            {"session_id": session["session_id"]}
        )
        self.assertEqual(evaluated["status"], "sufficient")
        self.assertFalse(evaluated["open_world_required"])
        self.assertEqual(
            evaluated["hypotheses"][0]["status"], "supported"
        )

    def test_conflicting_confirmed_evidence_keeps_research_open(self):
        session = self.start_session()
        hypothesis_id = session["hypothesis_ids"][0]
        research.tool_evidence_add(
            {
                "session_id": session["session_id"],
                "items": [
                    {
                        "hypothesis_id": hypothesis_id,
                        "stance": "support",
                        "source_family": "sec_filing",
                        "source": "SEC",
                        "excerpt": "Guidance increased.",
                        "source_status": "CONFIRMED",
                    },
                    {
                        "hypothesis_id": hypothesis_id,
                        "stance": "challenge",
                        "source_family": "options_flow",
                        "source": "Barchart",
                        "excerpt": "Positioning led the move.",
                        "source_status": "CONFIRMED",
                    },
                ],
            }
        )
        evaluated = research.tool_research_evaluate(
            {
                "session_id": session["session_id"],
                "new_hypotheses": [
                    {
                        "statement": "옵션 감마가 실적 반응을 증폭했다.",
                        "falsifier": "감마 지표와 가격 반응 시점이 불일치한다.",
                        "origin": "discovered_from_conflict",
                    }
                ],
            }
        )
        self.assertEqual(evaluated["status"], "researching")
        self.assertTrue(evaluated["open_world_required"])
        self.assertEqual(evaluated["hypotheses"][0]["status"], "mixed")
        self.assertEqual(len(evaluated["new_hypothesis_ids"]), 1)
        self.assertGreater(evaluated["pending_frontier_count"], 0)

    def test_partial_and_search_lead_do_not_resolve_hypothesis(self):
        session = self.start_session()
        hypothesis_id = session["hypothesis_ids"][0]
        research.tool_evidence_add(
            {
                "session_id": session["session_id"],
                "items": [
                    {
                        "hypothesis_id": hypothesis_id,
                        "stance": "support",
                        "source_family": "open_web_search",
                        "source": "search",
                        "excerpt": "Possible evidence",
                        "source_status": "SEARCH_LEAD",
                    },
                    {
                        "hypothesis_id": hypothesis_id,
                        "stance": "support",
                        "source_family": "massive",
                        "source": "Massive",
                        "excerpt": "Entitlement-limited response",
                        "source_status": "PARTIAL_LIMIT",
                    },
                ],
            }
        )
        evaluated = research.tool_research_evaluate(
            {"session_id": session["session_id"]}
        )
        self.assertEqual(evaluated["hypotheses"][0]["status"], "unresolved")

    def test_evidence_is_deduplicated_by_content(self):
        session = self.start_session()
        item = {
            "hypothesis_id": session["hypothesis_ids"][0],
            "stance": "support",
            "source_family": "sec",
            "source": "10-Q",
            "excerpt": "Same evidence",
            "source_status": "CONFIRMED",
        }
        first = research.tool_evidence_add(
            {"session_id": session["session_id"], "items": [item]}
        )
        second = research.tool_evidence_add(
            {"session_id": session["session_id"], "items": [item]}
        )
        self.assertEqual(first["added_count"], 1)
        self.assertEqual(second["duplicate_count"], 1)

    def test_max_rounds_returns_bounded_limit_not_false_confidence(self):
        session = self.start_session(max_rounds=1)
        response = Mock()
        response.raise_for_status.return_value = None
        response.text = ""
        with patch.object(research.requests, "post", return_value=response):
            research.tool_research_search(
                {
                    "session_id": session["session_id"],
                    "query": "unresolved question",
                }
            )
        evaluated = research.tool_research_evaluate(
            {"session_id": session["session_id"]}
        )
        self.assertEqual(evaluated["status"], "bounded_limit")
        self.assertFalse(evaluated["open_world_required"])
        self.assertIn("unresolved", evaluated["stop_reason"])

    def test_stdio_tools_list_and_call_survive_bad_input(self):
        listed = research.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        self.assertEqual(len(listed["result"]["tools"]), len(research.TOOLS))
        called = research.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "market_research_start",
                    "arguments": {"question": "x", "verification_state": "BAD"},
                },
            }
        )
        self.assertTrue(called["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
