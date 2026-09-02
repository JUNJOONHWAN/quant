from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_fmp_us_equity_etf_symbols.py"
SPEC = importlib.util.spec_from_file_location("build_fmp_symbols", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class BuildFmpSymbolsTest(unittest.TestCase):
    def test_symbol_change_alias_is_preserved_in_universe_but_not_eod_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "universe.jsonl"
            target = root / "symbols.txt"
            rows = [
                {
                    "symbol": "FDXF",
                    "analysis_eligible": True,
                    "sources": ["stock_list", "symbol_change"],
                    "exchange": "NYSE",
                },
                {
                    "symbol": "FDXF.V",
                    "analysis_eligible": True,
                    "sources": ["symbol_change"],
                    "symbol_change_events": [
                        {"old_symbol": "FDXF.V", "new_symbol": "FDXF"}
                    ],
                },
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = MODULE.build(source, target)
            self.assertEqual(target.read_text(encoding="utf-8"), "FDXF\n")
            self.assertEqual(result["symbol_count"], 1)
            self.assertEqual(result["excluded_count"], 1)
            self.assertEqual(
                result["excluded"][0]["reason"], "non_canonical_symbol_change_alias"
            )
            self.assertEqual(result["excluded"][0]["canonical_symbols"], ["FDXF"])


if __name__ == "__main__":
    unittest.main()
