"""Strict OOS gate for common Flow plus stock-specific rotation information.

The common all-ETF Flow component is intentional in v6 and is never erased by
cross-sectional centering.  This diagnostic instead requires the common branch
to help absolute outcomes, the convergence/divergence rotation branch to help
benchmark-relative outcomes, and the correct stock query to beat a fixed
permutation out of sample.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from training.quant_forecast_v2.io_utils import sha256_file, utc_now

from .contracts import (
    COMMON_FLOW_TARGET_INDICES,
    MODEL_SCHEMA_VERSION,
    ROTATION_FLOW_TARGET_INDICES,
    TARGET_COLUMNS,
)
from .model import PriceBaseline
from .train import (
    FeatureStats,
    GraphDataset,
    _autocast,
    _device,
    _inputs,
    build_graph_model,
    predict_price,
)


GATE_SCHEMA_VERSION = "quant.etf_flow_graph_ai_specific_gate.v2"


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_acceptance_gate(
    base_metrics: Mapping[str, object],
    ai_specific_targets: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """Apply thresholds fixed before reading the candidate result."""

    targets = base_metrics["targets"]
    target_count = len(TARGET_COLUMNS)
    common_names = [TARGET_COLUMNS[index] for index in COMMON_FLOW_TARGET_INDICES]
    rotation_names = [TARGET_COLUMNS[index] for index in ROTATION_FLOW_TARGET_INDICES]

    def count_base(
        metric: str,
        threshold: float = 0.0,
        names: Sequence[str] = TARGET_COLUMNS,
    ) -> int:
        return sum(float(targets[name][metric]) > threshold for name in names)

    def count_ai(
        metric: str,
        threshold: float = 0.0,
        names: Sequence[str] = TARGET_COLUMNS,
    ) -> int:
        return sum(
            float(ai_specific_targets[name][metric]) > threshold
            for name in names
        )

    zero_exact_count = sum(
        float(targets[name]["zero_flow_max_abs_dynamic_pct"]) == 0.0
        for name in TARGET_COLUMNS
    )
    relative_improvements = [
        (
            float(targets[name]["price_mae_pct"])
            - float(targets[name]["graph_mae_pct"])
        )
        / max(float(targets[name]["price_mae_pct"]), 1e-12)
        * 100.0
        for name in TARGET_COLUMNS
    ]
    checks = {
        "zero_flow_exact_all_targets": zero_exact_count == target_count,
        "material_flow_input_effect_10_of_12": count_base(
            "mean_abs_dynamic_flow_input_effect_pct", 0.01
        )
        >= 10,
        "material_common_flow_5_of_6": count_base(
            "mean_abs_common_flow_pct", 0.01, common_names
        ) >= 5,
        "material_rotation_flow_5_of_6": count_base(
            "mean_abs_rotation_flow_pct", 0.01, rotation_names
        ) >= 5,
        "full_beats_price_6_of_12": count_base("price_minus_graph_mae_pct") >= 6,
        "mean_relative_improvement_vs_price_nonnegative": float(
            np.mean(relative_improvements)
        )
        >= 0.0,
        "actual_flow_beats_shuffled_7_of_12": count_base(
            "flow_specific_vs_shuffled_mae_pct"
        )
        >= 7,
        "timely_flow_beats_5_session_lag_7_of_12": count_base(
            "flow_timeliness_vs_lagged_mae_pct"
        ) >= 7,
        "full_beats_relation_6_of_12": count_base(
            "flow_incremental_vs_relation_mae_pct"
        )
        >= 6,
        "full_beats_zero_flow_6_of_12": count_base(
            "flow_incremental_vs_zero_flow_mae_pct"
        )
        >= 6,
        "common_flow_improves_absolute_4_of_6": count_base(
            "common_flow_incremental_mae_pct", names=common_names
        ) >= 4,
        "rotation_flow_improves_relative_4_of_6": count_base(
            "rotation_flow_incremental_mae_pct", names=rotation_names
        ) >= 4,
        "correct_rotation_query_beats_shuffled_4_of_6": count_ai(
            "shuffled_rotation_query_minus_full_mae_pct",
            names=rotation_names,
        ) >= 4,
    }
    counters = {
        "target_count": target_count,
        "zero_flow_exact_count": zero_exact_count,
        "material_flow_input_effect_count": count_base(
            "mean_abs_dynamic_flow_input_effect_pct", 0.01
        ),
        "material_common_flow_absolute_count": count_base(
            "mean_abs_common_flow_pct", 0.01, common_names
        ),
        "material_rotation_flow_relative_count": count_base(
            "mean_abs_rotation_flow_pct", 0.01, rotation_names
        ),
        "full_beats_price_count": count_base("price_minus_graph_mae_pct"),
        "mean_relative_improvement_vs_price_pct": float(
            np.mean(relative_improvements)
        ),
        "actual_flow_beats_shuffled_count": count_base(
            "flow_specific_vs_shuffled_mae_pct"
        ),
        "timely_flow_beats_5_session_lag_count": count_base(
            "flow_timeliness_vs_lagged_mae_pct"
        ),
        "full_beats_relation_count": count_base(
            "flow_incremental_vs_relation_mae_pct"
        ),
        "full_beats_zero_flow_count": count_base(
            "flow_incremental_vs_zero_flow_mae_pct"
        ),
        "common_flow_improves_absolute_count": count_base(
            "common_flow_incremental_mae_pct", names=common_names
        ),
        "rotation_flow_improves_relative_count": count_base(
            "rotation_flow_incremental_mae_pct", names=rotation_names
        ),
        "correct_rotation_query_beats_shuffled_count": count_ai(
            "shuffled_rotation_query_minus_full_mae_pct",
            names=rotation_names,
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counters": counters,
        "policy": {
            "purpose": (
                "Require common all-ETF Flow, convergence/divergence rotation, "
                "correct stock conditioning, timeliness, and OOS incremental value."
            ),
            "common_flow": (
                "A common market/benchmark Flow effect is intentional and is not "
                "cross-sectionally centered or rejected as a broadcast shortcut."
            ),
            "linked_flow": (
                "Measured separately but not required to pass because the primary "
                "hypothesis is the all-ETF global Flow path."
            ),
        },
    }


@torch.no_grad()
def run_diagnostic(args: argparse.Namespace) -> dict[str, object]:
    device = _device(args.device, args.cuda_memory_fraction)
    dataset = GraphDataset(args.dataset_root)
    fold = json.loads(args.fold_receipt.read_text(encoding="utf-8"))
    if fold.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("fold model schema does not match the current diagnostic")
    checkpoint_path = Path(fold["checkpoint"]["path"])
    if sha256_file(checkpoint_path) != fold["checkpoint"]["sha256"]:
        raise ValueError("checkpoint digest mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["configuration"]
    stock_stats = FeatureStats(
        np.asarray(checkpoint["stock_stats"]["mean"], dtype=np.float32),
        np.asarray(checkpoint["stock_stats"]["std"], dtype=np.float32),
    )
    etf_stats = FeatureStats(
        np.asarray(checkpoint["etf_stats"]["mean"], dtype=np.float32),
        np.asarray(checkpoint["etf_stats"]["std"], dtype=np.float32),
    )
    target_stats = FeatureStats(
        np.asarray(checkpoint["target_stats"]["mean"], dtype=np.float32),
        np.asarray(checkpoint["target_stats"]["std"], dtype=np.float32),
    )
    test_start, test_end = fold["outer_test"]
    dates = [date for date in dataset.dates if test_start <= date <= test_end]
    if not dates:
        raise ValueError("fold has no matching test dates")
    sample = dataset.load(dates[0])
    price_model = PriceBaseline(
        sample.stock_x.shape[-1] * 2,
        int(configuration["hidden_dim"]),
        sample.targets.shape[-1],
        float(configuration["dropout"]),
    ).to(device)
    graph_model = build_graph_model(
        dataset,
        sample,
        hidden_dim=int(configuration["hidden_dim"]),
        heads=int(configuration["heads"]),
        temporal_layers=int(configuration["temporal_layers"]),
        set_layers=int(configuration["set_layers"]),
        graph_layers=int(configuration["graph_layers"]),
        inducing_points=int(configuration["inducing_points"]),
        dropout=float(configuration["dropout"]),
    ).to(device)
    price_model.load_state_dict(checkpoint["price_model"])
    graph_model.load_state_dict(checkpoint["graph_model"])
    price_model.eval()
    graph_model.eval()
    _, target_std = target_stats.tensors(device)

    targets = []
    masks = []
    full_predictions = []
    shuffled_query_predictions = []
    rotation_components = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    for date in dates:
        snapshot = dataset.load(date)
        baseline = predict_price(
            price_model,
            snapshot,
            stock_stats,
            target_stats,
            device,
            args.bf16,
        )
        inputs = _inputs(snapshot, stock_stats, etf_stats, device)
        with _autocast(device, args.bf16):
            output = graph_model(**inputs)
            shuffled_inputs = dict(inputs)
            permutation = torch.randperm(
                inputs["stock_x"].shape[0], generator=generator
            ).to(device)
            shuffled_inputs["stock_x"] = inputs["stock_x"][permutation]
            shuffled_inputs["stock_ids"] = inputs["stock_ids"][permutation]
            shuffled_query_output = graph_model(**shuffled_inputs)
        relation = (output.relation_residual.float() * target_std).cpu().numpy()
        linked = (output.linked_flow_residual.float() * target_std).cpu().numpy()
        common_flow = (output.common_flow_residual.float() * target_std).cpu().numpy()
        rotation_flow = (
            output.rotation_flow_residual.float() * target_std
        ).cpu().numpy()
        shuffled_rotation = (
            shuffled_query_output.rotation_flow_residual.float() * target_std
        ).cpu().numpy()
        common = baseline + relation + common_flow + linked
        targets.append(snapshot.targets)
        masks.append(snapshot.target_mask)
        full_predictions.append(common + rotation_flow)
        shuffled_query_predictions.append(common + shuffled_rotation)
        rotation_components.append(rotation_flow)

    target = np.concatenate(targets)
    mask = np.concatenate(masks).astype(bool)
    full = np.concatenate(full_predictions)
    shuffled_query = np.concatenate(shuffled_query_predictions)
    rotation = np.concatenate(rotation_components)
    target_metrics = {}
    for index, name in enumerate(TARGET_COLUMNS):
        valid = mask[:, index] & np.isfinite(target[:, index])
        full_mae = float(np.mean(np.abs(full[valid, index] - target[valid, index])))
        shuffled_mae = float(
            np.mean(np.abs(shuffled_query[valid, index] - target[valid, index]))
        )
        target_metrics[name] = {
            "rows": int(valid.sum()),
            "full_mae_pct": full_mae,
            "shuffled_stock_query_mae_pct": shuffled_mae,
            "shuffled_rotation_query_minus_full_mae_pct": shuffled_mae - full_mae,
            "mean_abs_rotation_flow_pct": float(
                np.mean(np.abs(rotation[valid, index]))
            ),
        }

    # Add one derived field used by the fixed gate without mutating training output.
    base_metrics = json.loads(json.dumps(fold["metrics"]))
    for name in TARGET_COLUMNS:
        item = base_metrics["targets"][name]
        item["price_minus_graph_mae_pct"] = (
            float(item["price_mae_pct"]) - float(item["graph_mae_pct"])
        )
    gate = summarize_acceptance_gate(base_metrics, target_metrics)
    result = {
        "schema_version": GATE_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": gate["status"],
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "dataset_root": str(dataset.root),
        "dataset_manifest_sha256": sha256_file(dataset.root / "manifest.json"),
        "fold_receipt": str(args.fold_receipt),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "dates": [dates[0], dates[-1]],
        "date_count": len(dates),
        "row_count": int(len(target)),
        "controls": {
            "shuffled_stock_query": (
                "Deterministically permute stock state and identity before the "
                "rotation Flow-to-stock attention while retaining the actual "
                "common and linked Flow components; seed fixed before evaluation."
            ),
        },
        "targets": target_metrics,
        "acceptance_gate": gate,
        "side_effects": {
            "orders": 0,
            "emails": 0,
            "sheets_writes": 0,
            "scheduler_changes": 0,
            "service_changes": 0,
            "deployments": 0,
        },
    }
    _write_json_atomic(args.output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fold-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=9417)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_diagnostic(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
