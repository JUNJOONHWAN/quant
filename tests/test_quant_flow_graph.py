from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.quant_flow_graph.contracts import (
    DATASET_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    TARGET_COLUMNS,
)
from training.quant_flow_graph.data import (
    derive_targets,
    flow_coverage_ratio,
    recent_visible_flow_ids,
    validate_timing_row,
)


class QuantFlowGraphDataTests(unittest.TestCase):
    def test_timing_is_price_t1_flow_t2(self) -> None:
        sessions = ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"]
        validate_timing_row(
            sessions,
            signal_date="2026-08-27",
            price_date="2026-08-26",
            flow_date="2026-08-25",
        )
        with self.assertRaises(ValueError):
            validate_timing_row(
                sessions,
                signal_date="2026-08-27",
                price_date="2026-08-26",
                flow_date="2026-08-24",
            )

    def test_targets_include_upside_capture_and_downside_defense(self) -> None:
        stock = {
            "return_5d_pct": 5,
            "upside_5d_pct": 8,
            "loss_5d_pct": 3,
            "return_20d_pct": 10,
            "upside_20d_pct": 14,
            "loss_20d_pct": 6,
        }
        benchmark = {
            "return_5d_pct": 2,
            "upside_5d_pct": 4,
            "loss_5d_pct": 5,
            "return_20d_pct": 3,
            "upside_20d_pct": 9,
            "loss_20d_pct": 11,
        }
        actual = derive_targets(stock, benchmark)
        self.assertEqual(actual.shape, (len(TARGET_COLUMNS),))
        np.testing.assert_allclose(actual, [5, 8, 3, 3, 4, 2, 10, 14, 6, 7, 5, 5])

    def test_flow_coverage_detects_missing_cross_section(self) -> None:
        sessions = [f"2026-07-{day:02d}" for day in range(1, 23)]
        counts = {session: 5000 for session in sessions}
        counts["2026-07-21"] = 25
        current, reference, ratio = flow_coverage_ratio(
            counts, sessions, "2026-07-21"
        )
        self.assertEqual(current, 25)
        self.assertEqual(reference, 5000)
        self.assertAlmostEqual(ratio, 0.005)

    def test_flow_coverage_preserves_recurring_low_reporting_cadence(self) -> None:
        sessions = [f"2026-07-{day:02d}" for day in range(1, 23)]
        counts = {
            session: (800 if index % 3 == 0 else 2000)
            for index, session in enumerate(sessions)
        }
        current, reference, ratio = flow_coverage_ratio(
            counts, sessions, "2026-07-22"
        )
        self.assertEqual(current, 800)
        self.assertEqual(reference, 800)
        self.assertEqual(ratio, 1.0)

    def test_recent_flow_universe_retains_nonreporters_at_exact_t2(self) -> None:
        availability = np.asarray(
            [
                [1, -1, 1, -1],
                [-1, 2, -1, -1],
                [-1, -1, 5, -1],
            ],
            dtype=np.int32,
        )
        # At signal position 3, ETFs 0/1 remain active from recent PIT-visible
        # observations even though neither reports in the final row. ETF 2's
        # final row is not yet visible and ETF 3 has no recent Flow.
        self.assertEqual(recent_visible_flow_ids(availability, 3), {0, 1, 2})

    def test_loader_masks_flow_unavailable_at_signal(self) -> None:
        try:
            import torch  # noqa: F401
            from training.quant_flow_graph.train import GraphDataset
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            values = np.full((3, 1, 4), np.nan, dtype=np.float32)
            values[0, 0, 0] = 1.0
            values[1, 0, 0] = 2.0
            values[2, 0, 0] = 3.0
            np.save(root / "flow_values.npy", values)
            # Last effective observation is only available after signal position 2.
            np.save(
                root / "flow_available_session_index.npy",
                np.asarray([[0], [2], [3]], dtype=np.int32),
            )
            snapshot_path = snapshots / "2026-01-06.npz"
            np.savez_compressed(
                snapshot_path,
                stock_symbols=np.asarray(["A"], dtype="U32"),
                stock_x=np.zeros((1, 2), dtype=np.float32),
                targets=np.zeros((1, len(TARGET_COLUMNS)), dtype=np.float32),
                target_mask=np.ones((1, len(TARGET_COLUMNS)), dtype=np.uint8),
                etf_ids=np.asarray([0], dtype=np.int64),
                edge_index=np.asarray([[0], [0]], dtype=np.int64),
                edge_attr=np.asarray([[0.1, 0.0, 0.0]], dtype=np.float32),
                signal_position=np.asarray(2, dtype=np.int32),
                flow_position=np.asarray(2, dtype=np.int32),
            )
            manifest = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "quality_gate": "PASS_WITH_EXCLUSIONS",
                "sessions": ["2026-01-02", "2026-01-05", "2026-01-06"],
                "flow_cube": {"session_start_position": 0},
                "etf_vocabulary": ["ETF"],
                "snapshots": [
                    {"signal_date": "2026-01-06", "path": str(snapshot_path)}
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded = GraphDataset(root).load("2026-01-06")
            self.assertEqual(loaded.stock_ids.tolist(), [0])
            self.assertEqual(float(loaded.etf_x[0, -2, 0]), 2.0)
            self.assertTrue(np.isnan(loaded.etf_x[0, -1, 0]))


class QuantFlowGraphModelTests(unittest.TestCase):
    def test_model_schema_is_v6(self) -> None:
        self.assertEqual(MODEL_SCHEMA_VERSION, "quant.etf_flow_graph_forecaster.v6")

    def test_rotation_correlation_loss_does_not_replace_common_target(self) -> None:
        try:
            import torch
            from training.quant_flow_graph.train import (
                _masked_cross_sectional_correlation_loss,
            )
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        values = torch.tensor(
            [[1.0, 6.0], [2.0, 4.0], [3.0, 2.0], [4.0, 0.0]]
        )
        mask = torch.ones_like(values, dtype=torch.bool)
        loss = _masked_cross_sectional_correlation_loss(
            values,
            values * 3.0 + 7.0,
            mask,
            (0, 1),
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_ai_specific_gate_rejects_wrong_rotation_query(self) -> None:
        from training.quant_flow_graph.ai_specific_gate import (
            summarize_acceptance_gate,
        )

        base_targets = {}
        ai_targets = {}
        for name in TARGET_COLUMNS:
            base_targets[name] = {
                "zero_flow_max_abs_dynamic_pct": 0.0,
                "mean_abs_dynamic_flow_input_effect_pct": 0.20,
                "mean_abs_common_flow_pct": 0.15,
                "mean_abs_rotation_flow_pct": 0.15,
                "price_mae_pct": 2.0,
                "graph_mae_pct": 1.9,
                "price_minus_graph_mae_pct": 0.1,
                "flow_specific_vs_shuffled_mae_pct": 0.05,
                "flow_timeliness_vs_lagged_mae_pct": 0.05,
                "flow_incremental_vs_relation_mae_pct": 0.04,
                "flow_incremental_vs_zero_flow_mae_pct": 0.03,
                "common_flow_incremental_mae_pct": 0.02,
                "rotation_flow_incremental_mae_pct": 0.02,
            }
            ai_targets[name] = {
                "shuffled_rotation_query_minus_full_mae_pct": 0.0,
            }
        result = summarize_acceptance_gate(
            {"targets": base_targets}, ai_targets
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(
            result["checks"]["correct_rotation_query_beats_shuffled_4_of_6"]
        )

    def test_ai_specific_gate_accepts_stock_conditioned_oos_edge(self) -> None:
        from training.quant_flow_graph.ai_specific_gate import (
            summarize_acceptance_gate,
        )

        base_targets = {}
        ai_targets = {}
        for name in TARGET_COLUMNS:
            base_targets[name] = {
                "zero_flow_max_abs_dynamic_pct": 0.0,
                "mean_abs_dynamic_flow_input_effect_pct": 0.20,
                "mean_abs_common_flow_pct": 0.15,
                "mean_abs_rotation_flow_pct": 0.15,
                "price_mae_pct": 2.0,
                "graph_mae_pct": 1.9,
                "price_minus_graph_mae_pct": 0.1,
                "flow_specific_vs_shuffled_mae_pct": 0.05,
                "flow_timeliness_vs_lagged_mae_pct": 0.05,
                "flow_incremental_vs_relation_mae_pct": 0.04,
                "flow_incremental_vs_zero_flow_mae_pct": 0.03,
                "common_flow_incremental_mae_pct": 0.02,
                "rotation_flow_incremental_mae_pct": 0.02,
            }
            ai_targets[name] = {
                "shuffled_rotation_query_minus_full_mae_pct": 0.02,
            }
        result = summarize_acceptance_gate(
            {"targets": base_targets}, ai_targets
        )
        self.assertEqual(result["status"], "PASS")

    def test_holding_weighted_pool_uses_exact_pit_edges(self) -> None:
        try:
            import torch
            from training.quant_flow_graph.model import holding_weighted_pool
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        actual = holding_weighted_pool(
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
            torch.tensor([[0, 0, 1], [0, 1, 0]]),
            torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            2,
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([[0.25, 1.50], [1.00, 0.00]]),
        )

    def test_temporal_encoder_uses_feature_masks_and_reporting_age(self) -> None:
        try:
            import torch
            from training.quant_flow_graph.model import ETFTemporalEncoder
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        torch.manual_seed(5)
        encoder = ETFTemporalEncoder(
            input_dim=4,
            hidden_dim=8,
            heads=2,
            layers=1,
            max_lookback=3,
            dropout=0.0,
        )
        raw = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])
        fully_observed = torch.cat((raw, torch.ones_like(raw)), dim=-1)
        partially_observed = torch.cat(
            (raw, torch.tensor([[[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]])),
            dim=-1,
        )
        full_value, _, full_key, full_age = encoder(fully_observed)
        partial_value, _, partial_key, partial_age = encoder(partially_observed)
        self.assertFalse(torch.allclose(full_value, partial_value))
        self.assertFalse(torch.allclose(full_key, partial_key))
        torch.testing.assert_close(full_age, torch.zeros_like(full_age))
        self.assertGreater(float(partial_age.max()), 0.0)
        zero_values = torch.cat((torch.zeros_like(raw), torch.ones_like(raw)), dim=-1)
        zero_value, _, _, _ = encoder(zero_values)
        torch.testing.assert_close(zero_value, torch.zeros_like(zero_value))

    def test_graph_model_bf16_autocast_cuda(self) -> None:
        try:
            import torch
            from training.quant_flow_graph.model import ETFStockGraphForecaster
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 runtime is required")
        model = ETFStockGraphForecaster(
            stock_input_dim=4,
            stock_vocabulary_size=2,
            etf_input_dim=4,
            edge_dim=3,
            etf_vocabulary_size=3,
            target_dim=2,
            direction_dim=1,
            hidden_dim=16,
            heads=4,
            temporal_layers=1,
            set_layers=1,
            graph_layers=1,
            inducing_points=2,
            max_lookback=3,
            dropout=0.0,
        ).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                stock_x=torch.zeros(1, 4, device="cuda"),
                stock_ids=torch.tensor([0], device="cuda"),
                etf_x=torch.zeros(2, 3, 4, device="cuda"),
                etf_ids=torch.tensor([0, 1], device="cuda"),
                edge_index=torch.tensor([[0, 0], [0, 1]], device="cuda"),
                edge_attr=torch.tensor(
                    [[0.1, 0.0, 1.0], [0.2, 0.0, 1.0]], device="cuda"
                ),
            )
        self.assertTrue(torch.isfinite(output.residual_point).all())

    def test_graph_model_shapes_and_flow_ablation(self) -> None:
        try:
            import torch
            from training.quant_flow_graph.model import ETFStockGraphForecaster
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        torch.manual_seed(7)
        model = ETFStockGraphForecaster(
            stock_input_dim=8,
            stock_vocabulary_size=4,
            etf_input_dim=10,
            edge_dim=3,
            etf_vocabulary_size=7,
            target_dim=len(TARGET_COLUMNS),
            direction_dim=4,
            hidden_dim=32,
            heads=4,
            temporal_layers=1,
            set_layers=1,
            graph_layers=1,
            inducing_points=4,
            max_lookback=6,
            dropout=0.0,
        )
        etf_values = torch.randn(3, 6, 5)
        etf_masks = torch.ones(3, 6, 5)
        inputs = {
            "stock_x": torch.randn(2, 8),
            "stock_ids": torch.tensor([0, 2]),
            "etf_x": torch.cat((etf_values, etf_masks), dim=-1),
            "etf_ids": torch.tensor([1, 3, 5]),
            "edge_index": torch.tensor([[0, 0, 1], [0, 1, 2]]),
            "edge_attr": torch.tensor(
                [[0.10, 0.1, 1.0], [0.05, 0.2, 1.0], [0.20, 0.3, 0.0]]
            ),
        }
        full = model(**inputs)
        neutral = model(**inputs, mask_dynamic_flow=True)
        no_global = model(**inputs, mask_global_flow=True)
        no_linked = model(**inputs, mask_linked_flow=True)
        flow_only = model(**inputs, mask_relation=True)
        shuffled_inputs = dict(inputs)
        shuffled_inputs["etf_x"] = inputs["etf_x"][torch.tensor([2, 0, 1])]
        shuffled = model(**shuffled_inputs)
        zero_value_inputs = dict(inputs)
        zero_value_inputs["etf_x"] = torch.cat(
            (torch.zeros_like(etf_values), etf_masks), dim=-1
        )
        zero_value = model(**zero_value_inputs)
        self.assertEqual(full.residual_point.shape, (2, len(TARGET_COLUMNS)))
        self.assertEqual(full.relation_residual.shape, (2, len(TARGET_COLUMNS)))
        self.assertEqual(full.dynamic_flow_residual.shape, (2, len(TARGET_COLUMNS)))
        self.assertEqual(full.global_flow_residual.shape, (2, len(TARGET_COLUMNS)))
        self.assertEqual(full.common_flow_residual.shape, (2, len(TARGET_COLUMNS)))
        self.assertEqual(full.rotation_flow_residual.shape, (2, len(TARGET_COLUMNS)))
        self.assertEqual(full.linked_flow_residual.shape, (2, len(TARGET_COLUMNS)))
        self.assertEqual(full.residual_quantiles.shape, (2, len(TARGET_COLUMNS), 3))
        self.assertEqual(full.flow_reconstruction.shape, (3, 6, 5))
        self.assertEqual(full.direct_flow_context.shape, (2, 32))
        self.assertEqual(full.attention.shape, (4, 3))
        self.assertEqual(full.relation_attention.shape, (4, 3))
        self.assertEqual(full.global_etf_attention.shape, (4, 4, 3))
        self.assertEqual(full.global_stock_attention.shape, (4, 2, 4))
        self.assertEqual(full.common_factor_attention.shape, (4, 1, 4))
        self.assertEqual(full.global_flow_context.shape, (2, 32))
        self.assertEqual(full.etf_reporting_age.shape, (3, 5))
        torch.testing.assert_close(
            full.attention[:, :2].sum(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            full.attention[:, 2], torch.ones(4), atol=1e-5, rtol=1e-5
        )
        self.assertTrue(
            torch.all(full.residual_quantiles[..., 0] <= full.residual_quantiles[..., 1])
        )
        self.assertTrue(
            torch.all(full.residual_quantiles[..., 1] <= full.residual_quantiles[..., 2])
        )
        torch.testing.assert_close(
            full.residual_point,
            full.relation_residual + full.dynamic_flow_residual,
        )
        torch.testing.assert_close(
            full.dynamic_flow_residual,
            full.common_flow_residual
            + full.rotation_flow_residual
            + full.linked_flow_residual,
        )
        self.assertFalse(
            torch.allclose(
                full.global_flow_residual.mean(dim=0),
                torch.zeros(len(TARGET_COLUMNS)),
            )
        )
        torch.testing.assert_close(
            neutral.dynamic_flow_residual,
            torch.zeros_like(neutral.dynamic_flow_residual),
        )
        torch.testing.assert_close(
            flow_only.relation_residual,
            torch.zeros_like(flow_only.relation_residual),
        )
        torch.testing.assert_close(
            no_global.global_flow_residual,
            torch.zeros_like(no_global.global_flow_residual),
        )
        torch.testing.assert_close(
            no_linked.linked_flow_residual,
            torch.zeros_like(no_linked.linked_flow_residual),
        )
        self.assertTrue(torch.all(full.relation_gate < full.common_flow_gate))
        self.assertTrue(torch.all(full.relation_gate < full.rotation_flow_gate))
        self.assertTrue(torch.all(full.relation_gate < full.linked_flow_gate))
        self.assertFalse(torch.allclose(full.residual_point, neutral.residual_point))
        self.assertFalse(
            torch.allclose(full.dynamic_flow_residual, shuffled.dynamic_flow_residual)
        )
        torch.testing.assert_close(
            zero_value.dynamic_flow_residual,
            torch.zeros_like(zero_value.dynamic_flow_residual),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            zero_value.direct_flow_context,
            torch.zeros_like(zero_value.direct_flow_context),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            zero_value.global_flow_context,
            torch.zeros_like(zero_value.global_flow_context),
            atol=0.0,
            rtol=0.0,
        )

    def test_stock_without_edges_stays_finite(self) -> None:
        try:
            import torch
            from training.quant_flow_graph.model import ETFStockGraphForecaster
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        model = ETFStockGraphForecaster(
            stock_input_dim=4,
            stock_vocabulary_size=2,
            etf_input_dim=4,
            edge_dim=3,
            etf_vocabulary_size=2,
            target_dim=2,
            direction_dim=1,
            hidden_dim=16,
            heads=4,
            temporal_layers=1,
            set_layers=1,
            graph_layers=1,
            inducing_points=2,
            max_lookback=3,
            dropout=0.0,
        )
        output = model(
            stock_x=torch.zeros(1, 4),
            stock_ids=torch.tensor([0]),
            etf_x=torch.zeros(2, 3, 4),
            etf_ids=torch.tensor([0, 1]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 3)),
        )
        self.assertTrue(torch.isfinite(output.residual_point).all())
        torch.testing.assert_close(
            output.residual_point, torch.zeros_like(output.residual_point)
        )
        torch.testing.assert_close(
            output.relation_residual, torch.zeros_like(output.relation_residual)
        )
        torch.testing.assert_close(
            output.dynamic_flow_residual,
            torch.zeros_like(output.dynamic_flow_residual),
        )

    def test_all_etf_flow_reaches_stock_without_holding_edge(self) -> None:
        try:
            import torch
            from training.quant_flow_graph.model import ETFStockGraphForecaster
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        torch.manual_seed(19)
        model = ETFStockGraphForecaster(
            stock_input_dim=4,
            stock_vocabulary_size=2,
            etf_input_dim=4,
            edge_dim=3,
            etf_vocabulary_size=3,
            target_dim=2,
            direction_dim=1,
            hidden_dim=16,
            heads=4,
            temporal_layers=1,
            set_layers=1,
            graph_layers=1,
            inducing_points=3,
            max_lookback=3,
            dropout=0.0,
        )
        flow_values = torch.tensor(
            [
                [[1.0, -1.0], [2.0, 0.5], [0.5, 1.5]],
                [[-2.0, 1.0], [-1.0, 2.0], [1.0, -0.5]],
                [[0.25, 0.75], [1.25, -1.5], [2.0, 1.0]],
            ]
        )
        output = model(
            stock_x=torch.randn(2, 4),
            stock_ids=torch.tensor([0, 1]),
            etf_x=torch.cat((flow_values, torch.ones_like(flow_values)), dim=-1),
            etf_ids=torch.tensor([0, 1, 2]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 3)),
        )
        torch.testing.assert_close(
            output.linked_flow_residual,
            torch.zeros_like(output.linked_flow_residual),
        )
        torch.testing.assert_close(
            output.relation_residual,
            torch.zeros_like(output.relation_residual),
        )
        self.assertFalse(
            torch.allclose(
                output.global_flow_residual,
                torch.zeros_like(output.global_flow_residual),
            )
        )
        self.assertFalse(
            torch.allclose(
                output.common_flow_residual,
                torch.zeros_like(output.common_flow_residual),
            )
        )
        torch.testing.assert_close(
            output.residual_point,
            output.global_flow_residual,
        )

    def test_same_price_state_gets_stock_specific_global_flow(self) -> None:
        try:
            import torch
            from training.quant_flow_graph.model import ETFStockGraphForecaster
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        torch.manual_seed(29)
        model = ETFStockGraphForecaster(
            stock_input_dim=4,
            stock_vocabulary_size=2,
            etf_input_dim=4,
            edge_dim=3,
            etf_vocabulary_size=3,
            target_dim=2,
            direction_dim=1,
            hidden_dim=16,
            heads=4,
            temporal_layers=1,
            set_layers=1,
            graph_layers=1,
            inducing_points=3,
            max_lookback=3,
            dropout=0.0,
        )
        identical_price_state = torch.tensor([[0.5, -0.5, 1.0, 1.0]]).repeat(2, 1)
        flow_values = torch.tensor(
            [
                [[1.0, -1.0], [2.0, 0.5], [0.5, 1.5]],
                [[-2.0, 1.0], [-1.0, 2.0], [1.0, -0.5]],
                [[0.25, 0.75], [1.25, -1.5], [2.0, 1.0]],
            ]
        )
        output = model(
            stock_x=identical_price_state,
            stock_ids=torch.tensor([0, 1]),
            etf_x=torch.cat((flow_values, torch.ones_like(flow_values)), dim=-1),
            etf_ids=torch.tensor([0, 1, 2]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 3)),
        )
        self.assertFalse(
            torch.allclose(
                output.global_flow_residual[0],
                output.global_flow_residual[1],
            )
        )

    def test_single_stock_common_flow_is_not_centered_away(self) -> None:
        try:
            import torch
            from training.quant_flow_graph.model import ETFStockGraphForecaster
        except ImportError:
            self.skipTest("PyTorch runtime is intentionally container-only")
        torch.manual_seed(31)
        model = ETFStockGraphForecaster(
            stock_input_dim=4,
            stock_vocabulary_size=1,
            etf_input_dim=4,
            edge_dim=3,
            etf_vocabulary_size=2,
            target_dim=2,
            direction_dim=1,
            hidden_dim=16,
            heads=4,
            temporal_layers=1,
            set_layers=1,
            graph_layers=1,
            inducing_points=2,
            max_lookback=3,
            dropout=0.0,
        )
        values = torch.tensor(
            [
                [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
                [[-0.5, 1.0], [0.5, 2.0], [1.5, 3.0]],
            ]
        )
        output = model(
            stock_x=torch.tensor([[0.25, -0.25, 1.0, 1.0]]),
            stock_ids=torch.tensor([0]),
            etf_x=torch.cat((values, torch.ones_like(values)), dim=-1),
            etf_ids=torch.tensor([0, 1]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 3)),
        )
        self.assertGreater(float(output.common_flow_residual.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
