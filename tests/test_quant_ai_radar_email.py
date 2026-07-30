from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from workflows.quant_ai_radar import email_delivery


def _report() -> dict:
    return {
        "schema_version": "quant.ai_radar_report.v1",
        "as_of_date": "2026-07-29",
        "deployment_mode": "reference_publish",
        "selected_model_scope_complete": True,
        "selection": {"selected_count": 255},
        "market_judgement": {"market_state": "ROTATION"},
        "quality_audit": {
            "schema_version": "quant.ai_radar_quality_audit.v2",
            "status": "green",
            "publishable_reference_report": True,
            "scores": {
                "data_integrity": 10.0,
                "flow_evidence_quality": 10.0,
                "report_usability": 10.0,
            },
        },
    }


class QuantAiRadarEmailTest(unittest.TestCase):
    def test_mobile_contract_requires_420px(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.html"
            path.write_text(
                "<!doctype html><html><head>"
                '<meta name="viewport" content="width=device-width">'
                "<style>main{max-width:420px}</style></head><body>"
                + ("accepted report " * 500)
                + "</body></html>",
                encoding="utf-8",
            )
            result = email_delivery.validate_mobile_report(path)
        self.assertTrue(result["complete"])
        self.assertTrue(result["checks"]["max_width_420"])

    def test_delivery_is_gmail_proven_and_same_date_deduped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "2026-07-29"
            run_dir.mkdir()
            report_path = run_dir / "market_report.json"
            report_path.write_text(json.dumps(_report()), encoding="utf-8")
            attachment_path = run_dir / "market_report.html"
            attachment_path.write_text(
                "<!doctype html><html><head>"
                '<meta name="viewport" content="width=device-width">'
                "<style>main{max-width:420px}</style></head><body>"
                + ("market evidence " * 500)
                + "</body></html>",
                encoding="utf-8",
            )
            oauth = root / "oauth.json"
            oauth.write_text(
                json.dumps(
                    {
                        "client_id": "id",
                        "client_secret": "secret",
                        "refresh_token": "refresh",
                    }
                ),
                encoding="utf-8",
            )
            recipient = root / "recipient"
            recipient.write_text("radar@example.com\n", encoding="utf-8")
            environment = {
                "QUANT_AI_RADAR_GMAIL_OAUTH_FILE": str(oauth),
                "QUANT_AI_RADAR_GMAIL_RECIPIENT_FILE": str(recipient),
            }
            state_dir = root / "state"
            with mock.patch.object(
                email_delivery,
                "_send_gmail_api",
                return_value="gmail-message-id",
            ) as sender:
                first = email_delivery.deliver_daily_report(
                    report_path,
                    state_dir=state_dir,
                    environ=environment,
                )
                second = email_delivery.deliver_daily_report(
                    report_path,
                    state_dir=state_dir,
                    environ=environment,
                )
                email_artifact_exists = (
                    run_dir / "market_report_email_420.html"
                ).is_file()
        self.assertEqual(first["send_status"], "DONE")
        self.assertTrue(first["complete"])
        self.assertLessEqual(
            first["html_contract"]["bytes"],
            email_delivery.MAX_GMAIL_INLINE_BYTES,
        )
        self.assertTrue(email_artifact_exists)
        self.assertEqual(second["send_status"], "SKIP_ALREADY_SENT")
        sender.assert_called_once()

    def test_delivery_fails_closed_on_non_green_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "2026-07-29"
            run_dir.mkdir()
            report = _report()
            report["quality_audit"]["status"] = "red"
            report_path = run_dir / "market_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(email_delivery.EmailDeliveryError):
                email_delivery.deliver_daily_report(
                    report_path,
                    state_dir=root / "state",
                    environ={},
                )


if __name__ == "__main__":
    unittest.main()
