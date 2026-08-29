from __future__ import annotations

import unittest

import numpy as np

from training.quant_flow_graph_v12.posthoc_audit import (
    benjamini_hochberg,
    circular_block_indices,
    date_means,
    hac_mean_test,
)


class V12PosthocAuditTests(unittest.TestCase):
    def test_date_means_equalize_unequal_row_counts(self) -> None:
        dates, means = date_means(
            np.asarray([0, 0, 0, 1]), np.asarray([1.0, 2.0, 3.0, 10.0])
        )
        np.testing.assert_array_equal(dates, np.asarray([0, 1]))
        np.testing.assert_allclose(means, np.asarray([2.0, 10.0]))

    def test_hac_positive_constant_has_zero_p_value(self) -> None:
        result = hac_mean_test(np.ones(100), 20)
        self.assertEqual(result["mean_paired_mae_reduction"], 1.0)
        self.assertEqual(result["one_sided_p_value"], 0.0)

    def test_circular_blocks_have_expected_shape_and_bounds(self) -> None:
        indices = circular_block_indices(
            count=31, block=7, replications=50, seed=3
        )
        self.assertEqual(indices.shape, (50, 31))
        self.assertGreaterEqual(int(np.min(indices)), 0)
        self.assertLess(int(np.max(indices)), 31)

    def test_bh_adjustment_is_monotone_and_bounded(self) -> None:
        adjusted = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.5})
        self.assertLessEqual(adjusted["a"], adjusted["b"])
        self.assertLessEqual(adjusted["b"], adjusted["c"])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted.values()))


if __name__ == "__main__":
    unittest.main()
