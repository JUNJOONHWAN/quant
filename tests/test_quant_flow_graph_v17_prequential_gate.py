import importlib.util
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "quant_flow_graph_v17"
    / "prequential_gate.py"
)
STAGING_MODULE_PATH = (
    Path(__file__).parent / "quant_flow_graph_v17" / "prequential_gate.py"
)
MODULE_PATH = REPO_MODULE_PATH if REPO_MODULE_PATH.exists() else STAGING_MODULE_PATH
SPEC = importlib.util.spec_from_file_location("v17_gate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PrequentialGateTest(unittest.TestCase):
    def synthetic_arrays(self, dates=25, rows_per_date=20, date_offset=0):
        row_count = dates * rows_per_date
        date_codes = np.repeat(
            np.arange(date_offset, date_offset + dates, dtype=np.int32),
            rows_per_date,
        )
        grid = np.arange(row_count * len(MODULE.TARGET_NAMES), dtype=np.float32)
        price = (grid.reshape(row_count, -1) % 31 - 15) / 10.0
        base = price + 0.1
        query = base + np.sin(grid.reshape(row_count, -1) / 17.0) * 0.2
        actual = base + np.cos(grid.reshape(row_count, -1) / 23.0) * 0.3
        arrays = {
            "actual": actual.astype(np.float32),
            "date_codes": date_codes,
            MODULE.PRICE_MODEL: price.astype(np.float32),
            MODULE.BASE_MODEL: base.astype(np.float32),
        }
        for candidate in MODULE.CANDIDATES:
            arrays[candidate] = (
                price if candidate == MODULE.PRICE_MODEL else query
            ).astype(np.float32)
        arrays["full_etf_query_raw"] = query.astype(np.float32)
        return arrays

    def test_features_do_not_use_targets(self):
        first = self.synthetic_arrays()
        second = {key: value.copy() for key, value in first.items()}
        second["actual"] *= -7.0
        panel_first = MODULE.build_daily_panel(
            year=2021,
            candidate_name=MODULE.PRIMARY_CANDIDATE,
            arrays=first,
        )
        panel_second = MODULE.build_daily_panel(
            year=2021,
            candidate_name=MODULE.PRIMARY_CANDIDATE,
            arrays=second,
        )
        np.testing.assert_array_equal(panel_first.features, panel_second.features)
        self.assertFalse(
            np.array_equal(panel_first.outcomes, panel_second.outcomes)
        )

    def test_latest_calibration_year_purges_twenty_dates(self):
        panel_2021 = MODULE.build_daily_panel(
            year=2021,
            candidate_name=MODULE.PRIMARY_CANDIDATE,
            arrays=self.synthetic_arrays(dates=25),
        )
        x, y, audit = MODULE.calibration_indices([panel_2021], test_year=2022)
        self.assertEqual(audit["purged_date_count"], 20)
        self.assertEqual(audit["purged_daily_target_rows"], 20 * 12)
        self.assertEqual(len(x), 5 * 12)
        self.assertEqual(len(y), 5 * 12)
        self.assertEqual(len(audit["sample_weight"]), 5 * 12)

    def test_daily_decisions_switch_complete_date_target_groups(self):
        arrays = self.synthetic_arrays(dates=25, date_offset=504)
        panel = MODULE.build_daily_panel(
            year=2022,
            candidate_name=MODULE.PRIMARY_CANDIDATE,
            arrays=arrays,
        )
        switch = (panel.target_indices % 2 == 0)
        safe = (panel.date_codes % 2 == 0)
        hybrid, safe_rows, switch_matrix, safe_matrix = MODULE.apply_daily_decisions(
            date_codes=arrays["date_codes"],
            base=arrays[MODULE.BASE_MODEL],
            candidate=arrays[MODULE.PRIMARY_CANDIDATE],
            panel=panel,
            switch=switch,
            safe=safe,
        )
        np.testing.assert_array_equal(
            hybrid[:, 0], arrays[MODULE.PRIMARY_CANDIDATE][:, 0]
        )
        np.testing.assert_array_equal(hybrid[:, 1], arrays[MODULE.BASE_MODEL][:, 1])
        self.assertEqual(switch_matrix.shape, (25, 12))
        self.assertEqual(safe_matrix.shape, (25, 12))
        self.assertTrue(np.all(safe_rows[arrays["date_codes"] == 504]))
        self.assertFalse(np.any(safe_rows[arrays["date_codes"] == 505]))

    def test_preregistered_activation_never_deploys(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"deployment_forbidden": True', source)
        self.assertIn('"nvfp4_conversion_forbidden": True', source)
        self.assertIn("FUTURE_PROSPECTIVE_SHADOW_LOCKBOX_ONLY", source)

    def test_small_end_to_end_run_writes_complete_receipt(self):
        production_parameters = dict(MODULE.META_MODEL_PARAMETERS)
        MODULE.META_MODEL_PARAMETERS["max_iter"] = 2
        self.addCleanup(
            lambda: MODULE.META_MODEL_PARAMETERS.update(production_parameters)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v16_root = root / "v16"
            output_root = root / "v17"
            v16_root.mkdir()
            (v16_root / "v16_full_etf_identity_receipt.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (v16_root / "v16_full_etf_identity_preregistration.json").write_text(
                "{}\n", encoding="utf-8"
            )
            for year in MODULE.ALL_YEARS:
                arrays = self.synthetic_arrays(
                    dates=25, date_offset=(year - 2020) * 300
                )
                npz_path = v16_root / f"fold_{year}.npz"
                np.savez_compressed(npz_path, **arrays)
                (v16_root / f"fold_{year}.json").write_text(
                    json.dumps(
                        {"prediction_sha256": MODULE.sha256_file(npz_path)}
                    )
                    + "\n",
                    encoding="utf-8",
                )
            prereg_args = argparse.Namespace(
                v16_root=str(v16_root),
                output_root=str(output_root),
                preregister_only=True,
                expected_prereg_sha=None,
            )
            prereg_path, prereg_payload = MODULE.run(prereg_args)
            self.assertTrue(prereg_path.exists())
            run_args = argparse.Namespace(
                v16_root=str(v16_root),
                output_root=str(output_root),
                preregister_only=False,
                expected_prereg_sha=prereg_payload["preregistration_sha256"],
            )
            receipt_path, receipt = MODULE.run(run_args)
            self.assertTrue(receipt_path.exists())
            self.assertEqual(receipt["scope"]["evaluation_years"], [2022, 2023, 2024, 2025, 2026])
            self.assertEqual(receipt["timing"]["timing_violations"], 0)
            self.assertIn(receipt["gate"]["status"], {
                "V17_PREQUENTIAL_GATE_PASS",
                "V17_PREQUENTIAL_GATE_FAIL",
            })
            state = json.loads((output_root / "run_state.json").read_text())
            self.assertEqual(state["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
