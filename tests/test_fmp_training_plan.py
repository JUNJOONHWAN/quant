from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_fmp_training_plan.py"
SPEC = importlib.util.spec_from_file_location("build_fmp_training_plan", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FmpTrainingPlanTest(unittest.TestCase):
    def test_every_endpoint_is_classified_and_live_402_is_terminal(self):
        registry = [
            {
                "id": "statements_income_statement",
                "category": "Statements",
                "title": "Income Statement",
                "path": "/stable/income-statement",
                "plan_access": "starter",
                "sample_params": {"symbol": "AAPL"},
                "query_parameters": [{"name": "symbol"}],
            },
            {
                "id": "form_13f_positions_summary",
                "category": "Form 13F",
                "title": "Positions Summary",
                "path": "/stable/institutional-ownership/symbol-positions-summary",
                "plan_access": "ultimate",
                "sample_params": {"symbol": "AAPL"},
                "query_parameters": [{"name": "symbol"}],
            },
        ]
        probe = {
            "generated_at_utc": "2026-07-15T00:00:00Z",
            "providers": {
                "FMP": {
                    "records": [
                        {"id": "statements_income_statement", "status": "accessible"},
                        {
                            "id": "form_13f_positions_summary",
                            "status": "api_error",
                            "status_code": 402,
                        },
                    ]
                }
            },
        }
        plan = MODULE.build_plan(registry, probe, "2017-01-01", "2026-07-14")
        self.assertEqual(plan["endpoint_count"], 2)
        by_id = {row["id"]: row for row in plan["endpoints"]}
        self.assertEqual(by_id["statements_income_statement"]["action"], "backfill")
        self.assertEqual(
            by_id["statements_income_statement"]["collection"]["mode"], "per_symbol"
        )
        self.assertEqual(by_id["form_13f_positions_summary"]["action"], "not_entitled")


if __name__ == "__main__":
    unittest.main()
