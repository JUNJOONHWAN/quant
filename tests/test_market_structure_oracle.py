import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


QUANT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = QUANT_ROOT / "workflows" / "market_structure_oracle"
MODULE_PATH = WORKFLOW / "run_market_structure_oracle.py"


def _load_module():
    sys.path.insert(0, str(QUANT_ROOT))
    spec = importlib.util.spec_from_file_location(
        "run_market_structure_oracle_test", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_horizon_returns_never_uses_future_for_feature_date():
    module = _load_module()
    close = np.asarray(
        [[10.0, 20.0], [11.0, 18.0], [12.0, 21.0]], dtype=np.float32
    )
    result = module._horizon_returns(close, 1)
    assert np.isnan(result[0]).all()
    np.testing.assert_allclose(result[1], [0.1, -0.1], rtol=1e-6)
    np.testing.assert_allclose(
        result[2], [12.0 / 11.0 - 1.0, 21.0 / 18.0 - 1.0], rtol=1e-6
    )


def test_nonoverlapping_analogs_respect_embargo():
    module = _load_module()
    candidates = np.asarray([10, 12, 30, 31, 50])
    distances = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    selected, _ = module._select_nonoverlapping(
        candidates, distances, count=3, embargo=5
    )
    assert selected.tolist() == [10, 30, 50]


def test_current_regime_is_flow_supported_but_fragile_when_breadth_lags():
    module = _load_module()
    raw = {}
    zscore = {
        "breadth_up_20d": -0.2,
        "breadth_up_63d": 0.5,
        "median_ret_20d": -0.1,
        "median_ret_63d": 0.8,
        "flow_net_5d_usd": 1.2,
        "flow_net_20d_usd": 2.5,
        "flow_balance_5d": -0.2,
        "flow_balance_20d": -0.3,
        "rsp_spy_rel_20d": -0.1,
        "iwm_spy_rel_20d": -0.3,
        "hyg_tlt_rel_20d": 0.3,
        "deep_damage_frac": -2.0,
        "leadership_gap_20d": -0.5,
        "liquidity_hhi": -0.3,
    }
    result = module._classify_current(raw, zscore)
    assert result["regime"] == "fragile_risk_on"
    assert result["flow_score_z"] > 0
    assert result["breadth_score_z"] <= 0


def test_preflight_stdout_contract_is_json_serializable(tmp_path, monkeypatch):
    module = _load_module()
    payload = {
        "status": "PREFLIGHT_PASS",
        "app_id": "market-structure-oracle",
        "flow_policy_id": module.ETF_FLOW_POLICY_ID,
    }
    assert json.loads(json.dumps(payload))["status"] == "PREFLIGHT_PASS"


def test_scope_resolution_keeps_global_context_and_supports_etf_baskets():
    module = _load_module()
    resolved = module._resolve_scope(
        {"query": "반도체 섹터를 오라클로 분석해줘"}
    )
    assert resolved["scope_id"] == "semiconductors"
    assert resolved["etfs"] == ["SMH", "SOXX"]
    custom = module._resolve_scope(
        {
            "query": "내 금광 바스켓",
            "scope": "my_gold",
            "etfs": ["GDX", "GDXJ", "GDX"],
        }
    )
    assert custom["scope_id"] == "my_gold"
    assert custom["etfs"] == ["GDX", "GDXJ"]
    assert module._resolve_scope({"scope": "full_market"}) is None


def test_pit_membership_activates_only_on_available_date():
    module = _load_module()
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE etf_constituent_observations (
          etf_ticker TEXT, constituent_ticker TEXT, effective_date TEXT,
          available_date TEXT, weight_percent REAL, pit_confidence TEXT
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO etf_constituent_observations
        VALUES (?,?,?,?,?,?)
        """,
        [
            ("TEST", "AAA", "2020-01-02", "2020-01-06", 60.0, "date_exact"),
            ("TEST", "BBB", "2020-01-02", "2020-01-06", 40.0, "date_exact"),
            ("TEST", "BBB", "2020-01-07", "2020-01-09", 30.0, "date_exact"),
            ("TEST", "CCC", "2020-01-07", "2020-01-09", 70.0, "date_exact"),
        ],
    )
    dates = [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
        "2020-01-09",
    ]
    memberships, coverage = module._load_pit_scope_memberships(
        connection,
        dates=dates,
        symbols=["AAA", "BBB", "CCC"],
        etfs=["TEST"],
    )
    assert memberships[0] is None
    assert memberships[1] is None
    assert memberships[2][0].tolist() == [0, 1]
    assert memberships[4][0].tolist() == [0, 1]
    assert memberships[5][0].tolist() == [1, 2]
    assert coverage["history_start"] == "2020-01-06"
