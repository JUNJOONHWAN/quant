import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "etf_flow"
MODULE_PATH = WORKFLOW / "run_hermes_etf_flow_analysis.py"


def _load_module():
    sys.path.insert(0, str(WORKFLOW))
    spec = importlib.util.spec_from_file_location("run_hermes_etf_flow_analysis_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_hermes_accepts_fresh_contract_file_when_cli_has_no_final_response(
    tmp_path, monkeypatch
):
    module = _load_module()
    response_path = tmp_path / "response.json"
    expected = {"source_status": "partial", "items": [], "limitations": []}

    def fake_run(*args, **kwargs):
        response_path.write_text(json.dumps(expected), encoding="utf-8")
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="hermes -z: no final response was produced; treating the run as failed.",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._run_hermes(
        prompt="write the contract file only",
        response_path=response_path,
        timeout=30,
    ) == expected
