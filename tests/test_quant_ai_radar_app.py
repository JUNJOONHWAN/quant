from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from workflows.quant_ai_radar import app_cli


class QuantAiRadarAppTest(unittest.TestCase):
    def test_env_file_is_parsed_without_overwriting_explicit_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "radar.env"
            path.write_text(
                "# comment\n"
                "QUANT_AI_MODEL_ENDPOINT=http://file/v1/chat/completions\n"
                "QUANT_AI_WORKERS='8'\n",
                encoding="utf-8",
            )
            environment = {
                "QUANT_AI_MODEL_ENDPOINT": "http://explicit/v1/chat/completions"
            }
            loaded = app_cli.load_env_file(path, environ=environment)
        self.assertEqual(
            environment["QUANT_AI_MODEL_ENDPOINT"],
            "http://explicit/v1/chat/completions",
        )
        self.assertEqual(environment["QUANT_AI_WORKERS"], "8")
        self.assertEqual(
            loaded["QUANT_AI_MODEL_ENDPOINT"],
            "http://explicit/v1/chat/completions",
        )

    def test_sealed_request_requires_matching_canonical_sha(self):
        request = {
            "action": "analyze",
            "symbols": ["AAPL", "NVDA"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            digest = hashlib.sha256(
                app_cli._canonical_bytes(request)
            ).hexdigest()
            with mock.patch.dict(
                os.environ,
                {"OPERATIONS_APP_INPUT_SHA256": digest},
                clear=False,
            ):
                self.assertEqual(app_cli.load_sealed_request(path), request)
            with mock.patch.dict(
                os.environ,
                {"OPERATIONS_APP_INPUT_SHA256": "0" * 64},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    app_cli.AppCliError, "SHA-256 mismatch"
                ):
                    app_cli.load_sealed_request(path)

    def test_daily_command_preserves_caps_and_adds_shadow_and_smoke(self):
        with mock.patch.dict(
            os.environ,
            {
                "QUANT_AI_MODEL_ENDPOINT": (
                    "http://127.0.0.1:8018/v1/chat/completions"
                ),
                "QUANT_AI_RELEASE_MANIFEST": "/tmp/release.json",
            },
            clear=False,
        ):
            commands = app_cli.build_daily_commands(
                shadow=True,
                workers=8,
                max_ai_etfs=64,
                max_ai_stocks=192,
                smoke_max_items=2,
            )
        radar = commands[1]
        self.assertIn("--shadow", radar)
        self.assertEqual(radar[radar.index("--workers") + 1], "8")
        self.assertEqual(radar[radar.index("--max-ai-etfs") + 1], "64")
        self.assertEqual(radar[radar.index("--max-ai-stocks") + 1], "192")
        self.assertEqual(radar[radar.index("--smoke-max-items") + 1], "2")

    def test_daily_cli_none_values_fall_back_to_environment(self):
        with mock.patch.dict(
            os.environ,
            {
                "QUANT_AI_WORKERS": "6",
                "QUANT_AI_MAX_ETFS": "40",
                "QUANT_AI_MAX_STOCKS": "120",
            },
            clear=False,
        ):
            values = app_cli._daily_values(
                {
                    "shadow": False,
                    "workers": None,
                    "max_ai_etfs": None,
                    "max_ai_stocks": None,
                    "smoke_max_items": 0,
                }
            )
        self.assertEqual(values["workers"], 6)
        self.assertEqual(values["max_ai_etfs"], 40)
        self.assertEqual(values["max_ai_stocks"], 120)

    def test_on_demand_symbols_bypass_daily_selection_via_existing_entrypoint(self):
        with mock.patch.dict(
            os.environ,
            {
                "QUANT_AI_MODEL_ENDPOINT": (
                    "http://127.0.0.1:8018/v1/chat/completions"
                ),
                "QUANT_AI_RELEASE_MANIFEST": "/tmp/release.json",
            },
            clear=False,
        ):
            command = app_cli.build_analyze_command(["AAPL", "NVDA"])
        self.assertIn("workflows.quant_ai_radar.analyze_on_demand", command)
        self.assertNotIn("--max-ai-stocks", command)
        self.assertEqual(
            app_cli.normalize_symbols(["aapl,nvda", "AAPL"]),
            ["AAPL", "NVDA"],
        )

    def test_request_rejects_unknown_fields_and_mixed_direct_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            path.write_text(
                json.dumps({"action": "status", "trade": True}),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"OPERATIONS_APP_INPUT_SHA256": ""},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    app_cli.AppCliError, "unsupported application request fields"
                ):
                    app_cli.load_sealed_request(path)

    def test_oracle_status_is_summarized_without_embedded_artifact_history(self):
        summary = app_cli._oracle_summary(
            {
                "status": "COMPLETE",
                "target_as_of_date": "2026-07-28",
                "base_history_end": "2026-07-14",
                "database": "/tmp/oracle.sqlite3",
                "missing_sessions": [],
                "etf_flow": {
                    "latest_effective_date": "2026-07-28",
                    "latest_processed_date": "2026-07-28",
                    "record_count": 34996,
                    "ticker_count": 5418,
                    "capture": {"large": "not returned"},
                },
                "etf_radar_release_artifacts": {"large": "not returned"},
            }
        )
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["missing_session_count"], 0)
        self.assertEqual(summary["flow_record_count"], 34996)
        self.assertNotIn("capture", summary)
        self.assertNotIn("etf_radar_release_artifacts", summary)


if __name__ == "__main__":
    unittest.main()
