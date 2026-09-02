from __future__ import annotations

import unittest

from training.quant_flow_graph_v12.channel_posthoc_audit import CONTRASTS


class V12ChannelPosthocAuditTests(unittest.TestCase):
    def test_key_contrasts_have_distinct_primary_and_baseline(self) -> None:
        self.assertIn("all_dynamic_vs_structure", CONTRASTS)
        self.assertIn("rolling_dynamic_vs_current_dynamic", CONTRASTS)
        self.assertIn("original_full_vs_full_current", CONTRASTS)
        for primary, baseline in CONTRASTS.values():
            self.assertNotEqual(primary, baseline)


if __name__ == "__main__":
    unittest.main()
