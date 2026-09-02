import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "etf_flow"
    / "refresh_quant_etf_flow_site.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("refresh_quant_etf_flow_site_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refresh_builds_packet_before_primary_analysis_reuse(tmp_path, monkeypatch):
    module = _load_module()
    report_root = tmp_path / "ETF Flow"
    artifact_dir = report_root / "data" / "artifacts" / "2026-07-17"
    artifact_dir.mkdir(parents=True)
    analysis_path = artifact_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps({"news": {"source_status": "confirmed", "items": []}}),
        encoding="utf-8",
    )
    calls = []

    monkeypatch.setattr(module, "REPORT_ROOT", report_root)
    monkeypatch.setattr(module, "LEGACY_INPUT", tmp_path / "input")
    monkeypatch.setattr(module, "LEGACY_REPORTS", report_root)
    monkeypatch.setattr(module, "_valid_existing_analysis", lambda *args: True)
    monkeypatch.setattr(module, "run", lambda command: calls.append(Path(command[1]).name))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--endpoint",
            "https://example.invalid/publish",
            "--token-file",
            str(tmp_path / "token"),
            "--report-date",
            "2026-07-17",
        ],
    )

    assert module.main() == 0
    assert calls == [
        "stage_etf_flow_snapshot.py",
        "build_etf_flow_analysis_packet.py",
        "publish_etf_flow_analysis.py",
    ]
    state = json.loads((artifact_dir / "publish_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "ok"
