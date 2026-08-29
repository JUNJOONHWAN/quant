import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "quant_flow_graph_v18"
    / "orthogonal_diffusion.py"
)
STAGING_MODULE_PATH = (
    Path(__file__).parent / "quant_flow_graph_v18" / "orthogonal_diffusion.py"
)
MODULE_PATH = REPO_MODULE_PATH if REPO_MODULE_PATH.exists() else STAGING_MODULE_PATH
SPEC = importlib.util.spec_from_file_location("v18_orthogonal_diffusion", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OrthogonalDiffusionTest(unittest.TestCase):
    def synthetic_arrays(self, *, dates=25, rows_per_date=20, date_offset=0):
        row_count = dates * rows_per_date
        target_count = len(MODULE.TARGET_NAMES)
        date_codes = np.repeat(
            np.arange(date_offset, date_offset + dates, dtype=np.int32),
            rows_per_date,
        )
        row = np.arange(row_count, dtype=np.float32)[:, None]
        target = np.arange(target_count, dtype=np.float32)[None, :]
        price = np.sin(row / 17.0 + target / 7.0).astype(np.float32)
        v12 = (price + 0.05 * np.cos(row / 11.0 + target)).astype(np.float32)
        global_only = (price + 0.08 * np.sin(row / 13.0 + target)).astype(
            np.float32
        )
        graph_delta = (
            0.15 * np.cos(row / 19.0 + target / 3.0)
            + 0.04 * (price - np.mean(price, axis=0))
        ).astype(np.float32)
        query = global_only + graph_delta
        actual = (global_only + 0.5 * graph_delta + 0.1 * np.sin(row / 5.0)).astype(
            np.float32
        )
        arrays = {
            "actual": actual,
            "date_codes": date_codes,
            MODULE.PRICE_MODEL: price,
            MODULE.V12_MODEL: v12,
            MODULE.BASE_MODEL: global_only,
            MODULE.PRIMARY_CANDIDATE: query,
            MODULE.LAG5_CANDIDATE: global_only + np.roll(graph_delta, rows_per_date * 5, axis=0),
            MODULE.AXIS_SHUFFLE_CANDIDATE: global_only + graph_delta[::-1],
            MODULE.DATE_SHUFFLE_CANDIDATE: global_only + np.roll(graph_delta, rows_per_date, axis=0),
            "full_etf_query_raw": query + graph_delta,
        }
        return {key: np.asarray(value) for key, value in arrays.items()}

    def test_target_mutation_does_not_change_features(self):
        first = self.synthetic_arrays()
        second = {key: value.copy() for key, value in first.items()}
        second["actual"] *= -9.0
        common_first, _ = MODULE.common_features(first)
        common_second, _ = MODULE.common_features(second)
        candidate_first, _, _ = MODULE.candidate_features(
            candidate_name=MODULE.PRIMARY_CANDIDATE, arrays=first
        )
        candidate_second, _, _ = MODULE.candidate_features(
            candidate_name=MODULE.PRIMARY_CANDIDATE, arrays=second
        )
        np.testing.assert_array_equal(common_first, common_second)
        np.testing.assert_array_equal(candidate_first, candidate_second)
        self.assertFalse(
            np.array_equal(
                first["actual"] - first[MODULE.BASE_MODEL],
                second["actual"] - second[MODULE.BASE_MODEL],
            )
        )

    def test_orthogonal_delta_is_date_centered_and_nuisance_orthogonal(self):
        arrays = self.synthetic_arrays()
        residual, audit = MODULE.orthogonal_diffusion_delta(
            candidate=arrays[MODULE.PRIMARY_CANDIDATE],
            base=arrays[MODULE.BASE_MODEL],
            price=arrays[MODULE.PRICE_MODEL],
            v12=arrays[MODULE.V12_MODEL],
            date_codes=arrays["date_codes"],
            ridge_alpha=1e-8,
        )
        for date_code in np.unique(arrays["date_codes"]):
            block = residual[arrays["date_codes"] == date_code]
            self.assertLess(float(np.max(np.abs(np.mean(block, axis=0)))), 1e-5)
        self.assertLess(audit["max_abs_date_target_nuisance_correlation"], 1e-4)
        self.assertFalse(audit["absolute_common_flow_modified"])

    def test_common_only_has_equal_width_and_zero_diffusion(self):
        arrays = self.synthetic_arrays()
        primary, primary_names, _ = MODULE.candidate_features(
            candidate_name=MODULE.PRIMARY_CANDIDATE, arrays=arrays
        )
        common, common_names, audit = MODULE.candidate_features(
            candidate_name=MODULE.COMMON_ONLY_CANDIDATE, arrays=arrays
        )
        self.assertEqual(primary.shape, common.shape)
        self.assertEqual(primary_names, common_names)
        self.assertTrue(np.all(common == 0.0))
        self.assertTrue(audit["common_only_all_zero"])

    def test_calibration_purges_latest_twenty_dates(self):
        folds = {}
        blocks = {}
        for year, offset in ((2021, 100), (2022, 200)):
            arrays = self.synthetic_arrays(date_offset=offset)
            common, names = MODULE.common_features(arrays)
            block, _, _ = MODULE.candidate_features(
                candidate_name=MODULE.PRIMARY_CANDIDATE, arrays=arrays
            )
            folds[year] = MODULE.FoldData(
                year=year,
                arrays=arrays,
                common_features=common,
                common_feature_names=names,
                input_receipt={},
            )
            blocks[year] = block
        x, y, weights, audit = MODULE.calibration_data(
            folds=folds, candidate_blocks=blocks, test_year=2022
        )
        self.assertEqual(audit["purged_date_count"], 20)
        self.assertEqual(audit["calibration_dates"], 5)
        self.assertEqual(len(x), 5 * 20)
        self.assertEqual(y.shape, (5 * 20, len(MODULE.TARGET_NAMES)))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=6)

    def test_activation_contract_forbids_deployment(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"deployment_forbidden": True', source)
        self.assertIn('"trading_forbidden": True', source)
        self.assertIn('"nvfp4_conversion_forbidden": True', source)
        self.assertIn("FUTURE_PROSPECTIVE_SHADOW_LOCKBOX_ONLY", source)

    def test_small_end_to_end_writes_receipt(self):
        original_fit = MODULE.fit_predict_residual
        original_bootstrap = MODULE.BOOTSTRAP_REPLICATIONS

        def fake_fit(*, train_x, train_y, test_x, weights, feature_names, thread_count):
            del train_x, weights, feature_names, thread_count
            mean = np.mean(train_y, axis=0, dtype=np.float64).astype(np.float32)
            return (
                np.repeat(mean[None, :], len(test_x), axis=0),
                [],
                0.0,
            )

        MODULE.fit_predict_residual = fake_fit
        MODULE.BOOTSTRAP_REPLICATIONS = 20
        self.addCleanup(setattr, MODULE, "fit_predict_residual", original_fit)
        self.addCleanup(setattr, MODULE, "BOOTSTRAP_REPLICATIONS", original_bootstrap)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v16_root = root / "v16"
            output_root = root / "v18"
            v16_root.mkdir()
            (v16_root / "v16_full_etf_identity_receipt.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (v16_root / "v16_full_etf_identity_preregistration.json").write_text(
                "{}\n", encoding="utf-8"
            )
            for index, year in enumerate(MODULE.ALL_YEARS):
                arrays = self.synthetic_arrays(date_offset=100 + index * 100)
                npz_path = v16_root / f"fold_{year}.npz"
                np.savez_compressed(npz_path, **arrays)
                (v16_root / f"fold_{year}.json").write_text(
                    json.dumps({"prediction_sha256": MODULE.sha256_file(npz_path)})
                    + "\n",
                    encoding="utf-8",
                )
            prereg_args = argparse.Namespace(
                v16_root=str(v16_root),
                output_root=str(output_root),
                thread_count=1,
                preregister_only=True,
                expected_prereg_sha=None,
            )
            prereg_path, prereg_payload = MODULE.run(prereg_args)
            self.assertTrue(prereg_path.exists())
            run_args = argparse.Namespace(
                v16_root=str(v16_root),
                output_root=str(output_root),
                thread_count=1,
                preregister_only=False,
                expected_prereg_sha=prereg_payload["preregistration_sha256"],
            )
            receipt_path, receipt = MODULE.run(run_args)
            self.assertTrue(receipt_path.exists())
            self.assertEqual(receipt["scope"]["evaluation_rows"], 5 * 25 * 20)
            self.assertEqual(receipt["timing"]["timing_violations"], 0)
            self.assertIn(
                receipt["gate"]["status"],
                {
                    "V18_ORTHOGONAL_DIFFUSION_PASS",
                    "V18_ORTHOGONAL_DIFFUSION_FAIL",
                },
            )
            state = json.loads((output_root / "run_state.json").read_text())
            self.assertEqual(state["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
