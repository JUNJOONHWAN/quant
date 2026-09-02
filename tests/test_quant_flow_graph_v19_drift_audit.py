import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


STAGING_ROOT = Path(__file__).parent
V18_STAGING = STAGING_ROOT.parent / "quant_v18_staging"
REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(V18_STAGING), str(STAGING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

REPO_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "quant_flow_graph_v19"
    / "drift_audit.py"
)
STAGING_MODULE_PATH = STAGING_ROOT / "quant_flow_graph_v19" / "drift_audit.py"
MODULE_PATH = REPO_MODULE_PATH if REPO_MODULE_PATH.exists() else STAGING_MODULE_PATH
SPEC = importlib.util.spec_from_file_location("v19_drift_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GlobalDriftAuditTest(unittest.TestCase):
    def synthetic_arrays(self, *, date_offset=0, dates=25, rows_per_date=20):
        rows = dates * rows_per_date
        targets = len(MODULE.TARGET_NAMES)
        date_codes = np.repeat(
            np.arange(date_offset, date_offset + dates, dtype=np.int32),
            rows_per_date,
        )
        grid = np.arange(rows * targets, dtype=np.float32).reshape(rows, targets)
        actual = np.sin(grid / 37.0).astype(np.float32)
        global_only = (actual + 0.05 * np.cos(grid / 19.0)).astype(np.float32)
        price = (actual + 0.25 * np.cos(grid / 19.0)).astype(np.float32)
        v12 = (actual + 0.18 * np.cos(grid / 19.0)).astype(np.float32)
        lag = (actual + 0.12 * np.cos(grid / 17.0)).astype(np.float32)
        date_shuffle = (actual + 0.30 * np.cos(grid / 23.0)).astype(np.float32)
        axis = (actual + 0.08 * np.cos(grid / 29.0)).astype(np.float32)
        return {
            "actual": actual,
            "date_codes": date_codes,
            MODULE.PRICE_MODEL: price,
            MODULE.V12_MODEL: v12,
            "full_etf_query_raw": global_only,
            "full_etf_query": global_only,
            MODULE.BASE_MODEL: global_only,
            MODULE.LAG5_CANDIDATE: lag,
            MODULE.AXIS_SHUFFLE_CANDIDATE: axis,
            MODULE.DATE_SHUFFLE_CANDIDATE: date_shuffle,
        }

    def test_preregistration_separates_claims_and_forbids_deployment(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"distribution_edge"', source)
        self.assertIn('"current_timing_edge"', source)
        self.assertIn('"stock_topology_claim_forbidden": True', source)
        self.assertIn('"deployment_forbidden": True', source)

    def test_gate_can_pass_distribution_without_current_timing(self):
        metric = {
            "mae": 1.0,
            "economic_basket_value": 1.0,
            "economic_basket_p10": 0.0,
            "mean_daily_rank_ic": 0.0,
        }
        pooled = {MODULE.PRIMARY_MODEL: {name: dict(metric) for name in MODULE.TARGET_NAMES}}
        for comparator in MODULE.COMPARATORS:
            pooled[comparator] = {}
            for name in MODULE.TARGET_NAMES:
                value = dict(metric)
                value["mae"] = 1.2 if comparator != MODULE.LAG5_CANDIDATE else 0.9
                value["economic_basket_value"] = 0.8
                pooled[comparator][name] = value
        yearly = {}
        for year in MODULE.ALL_YEARS:
            yearly[str(year)] = {
                model: {name: dict(values[name]) for name in MODULE.TARGET_NAMES}
                for model, values in pooled.items()
            }
        bootstrap = {}
        for comparator in MODULE.COMPARATORS:
            bootstrap[comparator] = {}
            for name in MODULE.TARGET_NAMES:
                lower = -0.1 if comparator == MODULE.LAG5_CANDIDATE else 0.1
                bootstrap[comparator][name] = {
                    "mae_gain": {"ci_lower_95": lower},
                    "basket_gain": {"ci_lower_95": 0.1},
                }
        gate = MODULE._gate(pooled=pooled, yearly=yearly, bootstrap=bootstrap)
        self.assertTrue(gate["checks"]["distribution_forecast_pass"])
        self.assertFalse(gate["checks"]["current_timing_pass"])

    def test_small_end_to_end_uses_all_rows_and_writes_receipt(self):
        original_bootstrap = MODULE.moving_block_bootstrap

        def fast_bootstrap(values, *, seed):
            del seed
            mean = float(np.mean(values))
            return {
                "observed_mean": mean,
                "ci_lower_95": mean,
                "ci_upper_95": mean,
                "one_sided_probability_nonpositive": float(mean <= 0.0),
                "block_sessions": 20,
                "replications": 1,
                "date_count": len(values),
            }

        MODULE.moving_block_bootstrap = fast_bootstrap
        self.addCleanup(setattr, MODULE, "moving_block_bootstrap", original_bootstrap)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v16 = root / "v16"
            output = root / "v19"
            v16.mkdir()
            (v16 / "v16_full_etf_identity_receipt.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (v16 / "v16_full_etf_identity_preregistration.json").write_text(
                "{}\n", encoding="utf-8"
            )
            for index, year in enumerate(MODULE.ALL_YEARS):
                arrays = self.synthetic_arrays(date_offset=100 + index * 100)
                npz_path = v16 / f"fold_{year}.npz"
                np.savez_compressed(npz_path, **arrays)
                (v16 / f"fold_{year}.json").write_text(
                    json.dumps({"prediction_sha256": MODULE.sha256_file(npz_path)})
                    + "\n",
                    encoding="utf-8",
                )
            prereg_args = argparse.Namespace(
                v16_root=str(v16),
                output_root=str(output),
                preregister_only=True,
                expected_prereg_sha=None,
            )
            prereg_path, prereg = MODULE.run(prereg_args)
            self.assertTrue(prereg_path.exists())
            run_args = argparse.Namespace(
                v16_root=str(v16),
                output_root=str(output),
                preregister_only=False,
                expected_prereg_sha=prereg["preregistration_sha256"],
            )
            receipt_path, receipt = MODULE.run(run_args)
            self.assertTrue(receipt_path.exists())
            self.assertEqual(receipt["scope"]["rows"], len(MODULE.ALL_YEARS) * 25 * 20)
            self.assertEqual(receipt["scope"]["dates"], len(MODULE.ALL_YEARS) * 25)
            self.assertFalse(receipt["scope"]["model_refit"])
            self.assertEqual(
                json.loads((output / "run_state.json").read_text())["status"],
                "COMPLETE",
            )


if __name__ == "__main__":
    unittest.main()
