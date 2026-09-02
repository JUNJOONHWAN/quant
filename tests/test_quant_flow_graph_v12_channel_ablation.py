from __future__ import annotations

import unittest

from training.quant_flow_graph_v12.channel_ablation import (
    classify_feature_channels,
    preregistration,
    rolling_base,
)


class V12ChannelAblationTests(unittest.TestCase):
    def test_rolling_base_recognizes_fixed_suffixes(self) -> None:
        self.assertEqual(rolling_base("drift_rate_pct_mean_20"), "drift_rate_pct")
        self.assertEqual(rolling_base("direct_clean_rate_net_z60"), "direct_clean_rate_net")
        self.assertIsNone(rolling_base("direct_clean_rate_net"))

    def test_channel_partition_separates_structure_and_dynamic(self) -> None:
        names = (
            "observed_ratio",
            "diffusion_coverage_mean_20",
            "drift_rate_pct",
            "drift_rate_pct_mean_20",
        )
        groups = classify_feature_channels(names)
        self.assertEqual(groups["structure_mask_only"], (0, 1))
        self.assertEqual(groups["current_dynamic_no_structure"], (2,))
        self.assertEqual(groups["rolling_dynamic_no_structure"], (3,))
        self.assertEqual(groups["all_dynamic_no_structure"], (2, 3))
        self.assertEqual(groups["full_current_no_rolling"], (0, 2))

    def test_preregistration_freezes_source_receipt_and_no_sampling(self) -> None:
        payload = preregistration()
        self.assertTrue(payload["frozen_before_ablation_results"])
        self.assertTrue(payload["scope"]["no_row_or_symbol_sampling"])
        self.assertTrue(payload["prohibitions"]["new_fmp_features"])
        self.assertTrue(payload["interpretation"]["does_not_change_original_predictive_gate"])


if __name__ == "__main__":
    unittest.main()
