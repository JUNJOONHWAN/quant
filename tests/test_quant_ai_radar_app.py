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
                "QUANT_AI_MODEL_TIMEOUT_SECONDS": "420",
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
        self.assertEqual(radar[radar.index("--timeout") + 1], "420")
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

    def test_production_daily_requires_email_completion(self):
        radar = {
            "status": "complete",
            "as_of_date": "2026-07-29",
            "run_dir": "/tmp/run",
            "report": "/tmp/run/market_report.json",
            "production_latest_published": True,
            "queue_counts": {"done": 255, "error": 0},
        }
        with mock.patch.object(
            app_cli, "daily_completion_status", return_value=None
        ), mock.patch.object(
            app_cli, "build_daily_commands", return_value=[["prepare"], ["radar"]]
        ), mock.patch.object(
            app_cli,
            "latest_closed_nyse_session",
            return_value="2026-07-29",
        ):
            with mock.patch.object(
                app_cli,
                "run_json_command",
                side_effect=[
                    {
                        "status": "complete",
                        "target_as_of_date": "2026-07-29",
                        "binding": {
                            "source_fingerprint_sha256": "a" * 64,
                        },
                    },
                    radar,
                ],
            ):
                with mock.patch.object(
                    app_cli,
                    "deliver_daily_report",
                    return_value={
                        "status": "DONE",
                        "complete": True,
                        "message_id": "gmail-id",
                    },
                ) as deliver:
                    with mock.patch.object(app_cli, "write_json"):
                        result = app_cli.run_daily(
                            shadow=False,
                            workers=4,
                            max_ai_etfs=64,
                            max_ai_stocks=192,
                        )
        deliver.assert_called_once_with(Path("/tmp/run/market_report.json"))
        self.assertTrue(result["email_delivery"]["complete"])

    def test_production_daily_skips_completed_oracle_target(self):
        completed = {
            "as_of_date": "2026-07-29",
            "run_dir": "/tmp/run",
            "report": "/tmp/run/market_report.json",
            "queue_counts": {"done": 255, "error": 0},
            "email_delivery": {
                "status": "DONE",
                "complete": True,
                "message_id": "gmail-id",
            },
        }
        with mock.patch.object(
            app_cli, "daily_completion_status", return_value=completed
        ), mock.patch.object(
            app_cli,
            "build_daily_commands",
            return_value=[["prepare"], ["radar"]],
        ) as build, mock.patch.object(
            app_cli,
            "run_json_command",
            return_value={
                "status": "complete",
                "target_as_of_date": "2026-07-29",
                "binding": {
                    "source_fingerprint_sha256": "a" * 64,
                },
            },
        ) as run, mock.patch.object(
            app_cli,
            "latest_closed_nyse_session",
            return_value="2026-07-29",
        ), mock.patch.object(app_cli, "write_json"):
            result = app_cli.run_daily(
                shadow=False,
                workers=4,
                max_ai_etfs=64,
                max_ai_stocks=192,
            )
        build.assert_called_once()
        run.assert_called_once_with(["prepare"])
        self.assertTrue(result["generation_skipped"])
        self.assertEqual(result["engine_status"], "already_complete")
        self.assertIn("generated_today", result["skip_reason"])

    def test_production_daily_rejects_stale_oracle_target_after_prepare(self):
        with mock.patch.object(
            app_cli,
            "build_daily_commands",
            return_value=[["prepare"], ["radar"]],
        ), mock.patch.object(
            app_cli,
            "run_json_command",
            return_value={
                "status": "complete",
                "target_as_of_date": "2026-07-29",
                "binding": {
                    "source_fingerprint_sha256": "a" * 64,
                },
            },
        ) as run, mock.patch.object(
            app_cli,
            "latest_closed_nyse_session",
            return_value="2026-07-30",
        ), mock.patch.object(app_cli, "write_json"):
            with self.assertRaisesRegex(
                app_cli.AppCliError, "not yet vendor-publishable"
            ):
                app_cli.run_daily(
                    shadow=False,
                    workers=4,
                    max_ai_etfs=64,
                    max_ai_stocks=192,
                )
        run.assert_called_once_with(["prepare"])

    def test_generated_today_kst_is_a_completion_gate(self):
        today = app_cli.datetime.now(app_cli.KST).date().isoformat()
        self.assertTrue(
            app_cli._generated_today_kst(
                {"generated_at_kst": f"{today}T07:15:00+09:00"}
            )
        )
        self.assertFalse(
            app_cli._generated_today_kst(
                {"generated_at_kst": "2026-07-30T22:09:46+09:00"}
            )
        )

    def test_shadow_daily_never_sends_email(self):
        radar = {
            "status": "shadow_complete_not_published",
            "as_of_date": "2026-07-29",
            "production_latest_published": False,
        }
        with mock.patch.object(
            app_cli, "build_daily_commands", return_value=[["prepare"], ["radar"]]
        ):
            with mock.patch.object(
                app_cli,
                "run_json_command",
                side_effect=[{"status": "COMPLETE"}, radar],
            ):
                with mock.patch.object(
                    app_cli, "deliver_daily_report"
                ) as deliver:
                    with mock.patch.object(app_cli, "write_json"):
                        result = app_cli.run_daily(
                            shadow=True,
                            workers=4,
                            max_ai_etfs=64,
                            max_ai_stocks=192,
                        )
        deliver.assert_not_called()
        self.assertEqual(result["email_delivery"]["status"], "NOT_REQUIRED")

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

    def test_natural_question_routes_explicit_ticker_to_fresh_analysis(self):
        with mock.patch.object(
            app_cli,
            "run_analysis",
            return_value={"status": "PASS", "results": [{"symbol": "AAPL"}]},
        ) as run_analysis:
            result = app_cli.run_question("AI Radar에 AAPL 분석해줘")
        run_analysis.assert_called_once_with(["AAPL"])
        self.assertEqual(result["intent"], "explicit_symbol_analysis")
        self.assertEqual(
            result["answer_basis"], "fresh_on_demand_trained_model_inference"
        )

    def test_natural_candidate_question_uses_only_fresh_green_report(self):
        report = {
            "as_of_date": "2026-07-29",
            "quality_audit": {
                "schema_version": app_cli.QUALITY_SCHEMA_VERSION,
                "status": "green",
                "publishable_reference_report": True,
                "scores": {"flow_evidence_quality": 10.0},
            },
            "market_judgement": {
                "market_state": "rotation",
                "confidence": 0.7,
                "summary": "검증된 시장 요약",
            },
            "market_dashboard": {
                "candidate_lanes": {
                    "positive_confirmation_stocks": [{"symbol": "AAPL"}]
                }
            },
            "source_status": {"quant_dataset": {"status": "confirmed"}},
        }

        def read_json(path):
            if Path(path).name == "latest.json":
                return report
            return {"target_as_of_date": "2026-07-29"}

        with mock.patch.object(app_cli, "_read_json", side_effect=read_json):
            with mock.patch.object(app_cli, "write_json"):
                result = app_cli.run_question("강세와 약세 종목 후보는?")
        self.assertEqual(result["intent"], "candidates")
        self.assertEqual(
            result["candidate_lanes"]["positive_confirmation_stocks"][0][
                "symbol"
            ],
            "AAPL",
        )

    def test_natural_question_fails_closed_when_report_is_stale(self):
        report = {
            "as_of_date": "2026-07-28",
            "quality_audit": {
                "schema_version": app_cli.QUALITY_SCHEMA_VERSION,
                "status": "green",
                "publishable_reference_report": True,
                "scores": {"flow_evidence_quality": 10.0},
            },
        }

        def read_json(path):
            if Path(path).name == "latest.json":
                return report
            return {"target_as_of_date": "2026-07-29"}

        with mock.patch.object(app_cli, "_read_json", side_effect=read_json):
            with self.assertRaisesRegex(app_cli.AppCliError, "stale"):
                app_cli.run_question("오늘 시장 분석 보여줘")

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

    def test_status_marks_old_report_stale_during_current_source_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            as_of_date = "2026-07-30"
            report_path = root / "runs" / as_of_date / "market_report.json"
            latest_path = root / "status" / "latest.json"
            app_state_path = root / "status" / "app_cli.json"
            oracle_path = root / "oracle.json"
            report_path.parent.mkdir(parents=True)
            latest_path.parent.mkdir(parents=True)
            report = {
                "as_of_date": as_of_date,
                "deployment_mode": "reference_publish",
                "selected_model_scope_complete": True,
                "queue_counts": {"done": 253, "excluded": 3},
                "source_status": {
                    "shared_oracle_store": {
                        "source_fingerprint_sha256": "a" * 64,
                    }
                },
            }
            encoded = json.dumps(report, sort_keys=True)
            report_path.write_text(encoded, encoding="utf-8")
            latest_path.write_text(encoded, encoding="utf-8")
            app_state_path.write_text(
                json.dumps(
                    {
                        "status": "running_full_scan_prioritized_inference",
                        "prepare": {
                            "binding": {
                                "source_fingerprint_sha256": "b" * 64,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            oracle_path.write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "target_as_of_date": as_of_date,
                        "missing_sessions": [],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                app_cli, "DEFAULT_OUTPUT_ROOT", root
            ), mock.patch.object(
                app_cli, "APP_STATE_PATH", app_state_path
            ), mock.patch.object(
                app_cli, "DEFAULT_ORACLE_STATUS", oracle_path
            ), mock.patch.object(
                app_cli,
                "endpoint_status",
                return_value={"status": "confirmed"},
            ), mock.patch.object(
                app_cli,
                "email_delivery_status",
                return_value={"status": "MISSING", "complete": False},
            ) as email_status:
                result = app_cli.status()
        self.assertEqual(result["status"], "running")
        self.assertEqual(
            result["latest_published"]["publication_status"],
            "STALE_ORACLE_SOURCE",
        )
        self.assertFalse(
            result["latest_published"]["source_matches_current_oracle"]
        )
        self.assertEqual(result["email_delivery"]["status"], "MISSING")
        email_status.assert_called_once_with(
            as_of_date,
            report_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            source_fingerprint_sha256="b" * 64,
        )

    def test_symbols_file_loads_and_seals_the_complete_favorites_universe(self):
        favorites = [
            "RKLB", "IONQ", "VRT", "MPWR", "STX", "PL", "NVDA",
            "CRDO", "ALAB", "QCOM", "GOOG", "SQQQ", "TZA",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "favorites.json"
            raw = json.dumps(favorites).encode("utf-8")
            path.write_bytes(raw)
            symbols, source = app_cli.load_symbols_file(path)
        self.assertEqual(symbols, favorites)
        self.assertEqual(source["symbol_count"], 13)
        self.assertEqual(source["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(source["path"], str(path.resolve()))

    def test_on_demand_analysis_requires_exact_requested_symbol_coverage(self):
        receipt = {
            "status": "complete",
            "as_of_date": "2026-08-03",
            "requested_symbols": ["NVDA", "QCOM"],
            "results": [
                {"symbol": "NVDA", "status": "complete"},
            ],
        }
        with mock.patch.object(
            app_cli, "run_json_command", return_value=receipt
        ), mock.patch.object(
            app_cli, "build_analyze_command", return_value=["analyze"]
        ), mock.patch.object(app_cli, "write_json"):
            with self.assertRaisesRegex(
                app_cli.AppCliError, "completed_symbols coverage mismatch"
            ):
                app_cli.run_analysis(["NVDA", "QCOM"])

    def test_on_demand_analysis_reports_complete_coverage_and_source(self):
        receipt = {
            "status": "complete",
            "as_of_date": "2026-08-03",
            "requested_symbols": ["NVDA", "QCOM"],
            "results": [
                {"symbol": "NVDA", "status": "complete"},
                {"symbol": "QCOM", "status": "complete"},
            ],
        }
        source = {"path": "/tmp/favorites.json", "symbol_count": 2}
        with mock.patch.object(
            app_cli, "run_json_command", return_value=receipt
        ), mock.patch.object(
            app_cli, "build_analyze_command", return_value=["analyze"]
        ), mock.patch.object(app_cli, "write_json"):
            result = app_cli.run_analysis(
                ["NVDA", "QCOM"], symbols_source=source
            )
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(result["requested_symbol_count"], 2)
        self.assertEqual(result["completed_symbol_count"], 2)
        self.assertEqual(result["requested_symbols"], ["NVDA", "QCOM"])
        self.assertEqual(result["completed_symbols"], ["NVDA", "QCOM"])
        self.assertEqual(result["symbols_source"], source)


if __name__ == "__main__":
    unittest.main()
