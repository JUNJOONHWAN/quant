"""BF16 smoke and expanding walk-forward training for ETF Flow graphs."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from training.quant_forecast_v2.io_utils import sha256_file, utc_now

from .contracts import (
    COMMON_FLOW_TARGET_INDICES,
    DATASET_SCHEMA_VERSION,
    DIRECTION_TARGET_INDICES,
    FLOW_LOOKBACK_SESSIONS,
    MODEL_SCHEMA_VERSION,
    PURGE_SESSIONS,
    RECEIPT_SCHEMA_VERSION,
    ROTATION_FLOW_TARGET_INDICES,
    TARGET_COLUMNS,
)
from .model import (
    ETFStockGraphForecaster,
    PriceBaseline,
    normalized_with_mask,
    parameter_count,
)


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def tensors(self, device: torch.device) -> tuple[Tensor, Tensor]:
        return (
            torch.as_tensor(self.mean, dtype=torch.float32, device=device),
            torch.as_tensor(self.std, dtype=torch.float32, device=device),
        )


@dataclass(frozen=True)
class Snapshot:
    signal_date: str
    stock_symbols: np.ndarray
    stock_ids: np.ndarray
    stock_x: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    etf_ids: np.ndarray
    etf_x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray


class GraphDataset:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError("dataset schema mismatch")
        if self.manifest.get("quality_gate") not in {
            "PASS",
            "PASS_WITH_EXCLUSIONS",
        }:
            raise ValueError("dataset quality gate is not PASS")
        self.sessions = tuple(self.manifest["sessions"])
        self.cube_start = int(
            self.manifest["flow_cube"]["session_start_position"]
        )
        self.flow_values = np.load(self.root / "flow_values.npy", mmap_mode="r")
        self.flow_available = np.load(
            self.root / "flow_available_session_index.npy", mmap_mode="r"
        )
        self.refs = tuple(self.manifest["snapshots"])
        self.by_date = {str(ref["signal_date"]): ref for ref in self.refs}
        stock_symbols: set[str] = set()
        for ref in self.refs:
            with np.load(ref["path"], allow_pickle=False) as item:
                stock_symbols.update(str(value) for value in item["stock_symbols"])
        self.stock_vocabulary = tuple(sorted(stock_symbols))
        self.stock_id_by_symbol = {
            symbol: index for index, symbol in enumerate(self.stock_vocabulary)
        }

    @property
    def dates(self) -> list[str]:
        return sorted(self.by_date)

    def load(self, signal_date: str) -> Snapshot:
        ref = self.by_date[signal_date]
        with np.load(ref["path"], allow_pickle=False) as item:
            stock_symbols = item["stock_symbols"].copy()
            stock_ids = np.asarray(
                [self.stock_id_by_symbol[str(symbol)] for symbol in stock_symbols],
                dtype=np.int64,
            )
            stock_x = item["stock_x"].astype(np.float32, copy=True)
            targets = item["targets"].astype(np.float32, copy=True)
            target_mask = item["target_mask"].astype(bool, copy=True)
            etf_ids = item["etf_ids"].astype(np.int64, copy=True)
            edge_index = item["edge_index"].astype(np.int64, copy=True)
            edge_attr = item["edge_attr"].astype(np.float32, copy=True)
            signal_position = int(item["signal_position"])
            flow_position = int(item["flow_position"])
        local_end = flow_position - self.cube_start
        local_start = local_end - FLOW_LOOKBACK_SESSIONS + 1
        if local_end < 0:
            raise ValueError(f"flow cube does not cover {signal_date}")
        history = np.full(
            (FLOW_LOOKBACK_SESSIONS, len(etf_ids), self.flow_values.shape[-1]),
            np.nan,
            dtype=np.float32,
        )
        if local_start < 0:
            source_start = 0
            target_start = -local_start
        else:
            source_start = local_start
            target_start = 0
        source = np.asarray(
            self.flow_values[source_start : local_end + 1, etf_ids],
            dtype=np.float32,
        )
        availability = np.asarray(
            self.flow_available[source_start : local_end + 1, etf_ids],
            dtype=np.int32,
        )
        visible = (availability >= 0) & (availability <= signal_position)
        source = np.where(visible[..., None], source, np.nan)
        history[target_start : target_start + len(source)] = source
        etf_x = np.transpose(history, (1, 0, 2))
        return Snapshot(
            signal_date=signal_date,
            stock_symbols=stock_symbols,
            stock_ids=stock_ids,
            stock_x=stock_x,
            targets=targets,
            target_mask=target_mask,
            etf_ids=etf_ids,
            etf_x=etf_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )


def _stats(values: Iterable[np.ndarray], width: int) -> FeatureStats:
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    count = np.zeros(width, dtype=np.float64)
    for value in values:
        flat = np.asarray(value, dtype=np.float64).reshape(-1, width)
        finite = np.isfinite(flat)
        clean = np.where(finite, flat, 0.0)
        total += clean.sum(axis=0)
        total_sq += np.square(clean).sum(axis=0)
        count += finite.sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(total_sq, count, out=np.ones_like(total), where=count > 0)
    variance = np.maximum(variance - np.square(mean), 1e-6)
    return FeatureStats(mean.astype(np.float32), np.sqrt(variance).astype(np.float32))


def fit_feature_stats(
    dataset: GraphDataset, dates: Sequence[str]
) -> tuple[FeatureStats, FeatureStats, FeatureStats]:
    if not dates:
        raise ValueError("cannot fit feature statistics on an empty date set")
    first = dataset.load(dates[0])
    stock_width = first.stock_x.shape[-1]
    etf_width = first.etf_x.shape[-1]
    target_width = first.targets.shape[-1]
    stock_stats = _stats(
        (dataset.load(date).stock_x for date in dates), stock_width
    )
    etf_stats = _stats((dataset.load(date).etf_x for date in dates), etf_width)
    target_stats = _stats(
        (dataset.load(date).targets for date in dates), target_width
    )
    return stock_stats, etf_stats, target_stats


def _device(name: str, cuda_memory_fraction: float) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but unsupported on this CUDA device")
        if not 0.0 < cuda_memory_fraction <= 1.0:
            raise ValueError("cuda_memory_fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction)
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_huber(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask & torch.isfinite(target)
    if not valid.any():
        return prediction.sum() * 0.0
    return F.huber_loss(prediction[valid], target[valid], delta=1.0)


def _pinball_loss(
    quantiles: Tensor, target: Tensor, mask: Tensor, levels: Sequence[float]
) -> Tensor:
    valid = mask & torch.isfinite(target)
    if not valid.any():
        return quantiles.sum() * 0.0
    losses = []
    for index, level in enumerate(levels):
        error = target - quantiles[..., index]
        loss = torch.maximum(level * error, (level - 1.0) * error)
        losses.append(loss[valid].mean())
    return torch.stack(losses).mean()


def _pairwise_rank_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask & torch.isfinite(target)
    prediction = prediction[valid]
    target = target[valid]
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    target_delta = target[:, None] - target[None, :]
    pred_delta = prediction[:, None] - prediction[None, :]
    usable = target_delta.abs() > 1e-6
    if not usable.any():
        return pred_delta.sum() * 0.0
    sign = target_delta.sign()
    return F.softplus(-sign[usable] * pred_delta[usable]).mean()


def _masked_cross_sectional_correlation_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    indices: Sequence[int],
) -> Tensor:
    """Encourage correct stock ordering for sector-rotation targets."""

    predicted = prediction[:, indices]
    actual = target[:, indices]
    valid = mask[:, indices] & torch.isfinite(actual)
    predicted = torch.where(valid, predicted, torch.zeros_like(predicted))
    actual = torch.where(valid, actual, torch.zeros_like(actual))
    count = valid.sum(dim=0).clamp_min(1)
    predicted_mean = predicted.sum(dim=0) / count
    actual_mean = actual.sum(dim=0) / count
    predicted = torch.where(
        valid, predicted - predicted_mean, torch.zeros_like(predicted)
    )
    actual = torch.where(valid, actual - actual_mean, torch.zeros_like(actual))
    numerator = (predicted * actual).sum(dim=0)
    denominator = predicted.square().sum(dim=0).sqrt() * actual.square().sum(dim=0).sqrt()
    usable = (valid.sum(dim=0) >= 3) & (denominator > 1e-8)
    safe_denominator = torch.where(usable, denominator, torch.ones_like(denominator))
    losses = torch.where(
        usable,
        1.0 - numerator / safe_denominator,
        numerator * 0.0,
    )
    return losses.sum() / usable.sum().clamp_min(1)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def _inputs(
    snapshot: Snapshot,
    stock_stats: FeatureStats,
    etf_stats: FeatureStats,
    device: torch.device,
) -> dict[str, Tensor]:
    stock_mean, stock_std = stock_stats.tensors(device)
    etf_mean, etf_std = etf_stats.tensors(device)
    stock_raw = torch.as_tensor(snapshot.stock_x, dtype=torch.float32, device=device)
    etf_raw = torch.as_tensor(snapshot.etf_x, dtype=torch.float32, device=device)
    return {
        "stock_x": normalized_with_mask(stock_raw, stock_mean, stock_std),
        "stock_ids": torch.as_tensor(
            snapshot.stock_ids,
            dtype=torch.long,
            device=device,
        ),
        "etf_x": normalized_with_mask(etf_raw, etf_mean, etf_std),
        "etf_ids": torch.as_tensor(snapshot.etf_ids, dtype=torch.long, device=device),
        "edge_index": torch.as_tensor(
            snapshot.edge_index, dtype=torch.long, device=device
        ),
        "edge_attr": torch.as_tensor(
            snapshot.edge_attr, dtype=torch.float32, device=device
        ),
    }


def fit_price_baseline(
    dataset: GraphDataset,
    dates: Sequence[str],
    stock_stats: FeatureStats,
    target_stats: FeatureStats,
    *,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    bf16: bool,
) -> PriceBaseline:
    sample = dataset.load(dates[0])
    model = PriceBaseline(
        sample.stock_x.shape[-1] * 2,
        hidden_dim,
        sample.targets.shape[-1],
        dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    target_mean, target_std = target_stats.tensors(device)
    for _ in range(epochs):
        model.train()
        for date in dates:
            snapshot = dataset.load(date)
            stock_mean, stock_std = stock_stats.tensors(device)
            stock_raw = torch.as_tensor(
                snapshot.stock_x, dtype=torch.float32, device=device
            )
            stock_x = normalized_with_mask(stock_raw, stock_mean, stock_std)
            target = torch.as_tensor(
                snapshot.targets, dtype=torch.float32, device=device
            )
            mask = torch.as_tensor(snapshot.target_mask, dtype=torch.bool, device=device)
            normalized_target = (target - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, bf16):
                prediction = model(stock_x)
                loss = _masked_huber(prediction, normalized_target, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict_price(
    model: PriceBaseline,
    snapshot: Snapshot,
    stock_stats: FeatureStats,
    target_stats: FeatureStats,
    device: torch.device,
    bf16: bool,
) -> np.ndarray:
    model.eval()
    stock_mean, stock_std = stock_stats.tensors(device)
    stock_raw = torch.as_tensor(snapshot.stock_x, dtype=torch.float32, device=device)
    stock_x = normalized_with_mask(stock_raw, stock_mean, stock_std)
    target_mean, target_std = target_stats.tensors(device)
    with _autocast(device, bf16):
        prediction = model(stock_x)
    raw = prediction.float() * target_std + target_mean
    return raw.cpu().numpy()


def build_graph_model(
    dataset: GraphDataset,
    sample: Snapshot,
    *,
    hidden_dim: int,
    heads: int,
    temporal_layers: int,
    set_layers: int,
    graph_layers: int,
    inducing_points: int,
    dropout: float,
) -> ETFStockGraphForecaster:
    return ETFStockGraphForecaster(
        stock_input_dim=sample.stock_x.shape[-1] * 2,
        stock_vocabulary_size=len(dataset.stock_vocabulary),
        etf_input_dim=sample.etf_x.shape[-1] * 2,
        edge_dim=sample.edge_attr.shape[-1],
        etf_vocabulary_size=len(dataset.manifest["etf_vocabulary"]),
        target_dim=sample.targets.shape[-1],
        direction_dim=len(DIRECTION_TARGET_INDICES),
        hidden_dim=hidden_dim,
        heads=heads,
        temporal_layers=temporal_layers,
        set_layers=set_layers,
        graph_layers=graph_layers,
        inducing_points=inducing_points,
        max_lookback=FLOW_LOOKBACK_SESSIONS,
        dropout=dropout,
    )


def pretrain_flow_encoder(
    model: ETFStockGraphForecaster,
    dataset: GraphDataset,
    dates: Sequence[str],
    etf_stats: FeatureStats,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    bf16: bool,
    mask_probability: float = 0.15,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    if epochs <= 0:
        return
    optimizer = torch.optim.AdamW(
        list(model.etf_temporal.parameters())
        + list(model.reconstruction_head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    etf_mean, etf_std = etf_stats.tensors(device)
    raw_width = len(etf_stats.mean)
    for epoch in range(epochs):
        model.train()
        for date in dates:
            snapshot = dataset.load(date)
            raw = torch.as_tensor(snapshot.etf_x, dtype=torch.float32, device=device)
            etf_x = normalized_with_mask(raw, etf_mean, etf_std)
            observed = etf_x[..., raw_width:].any(dim=-1)
            selected = observed & (torch.rand_like(observed.float()) < mask_probability)
            if not selected.any():
                continue
            masked = etf_x.clone()
            masked[selected] = 0.0
            target = etf_x[..., :raw_width]
            feature_mask = etf_x[..., raw_width:] > 0
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, bf16):
                reconstruction = model.reconstruct_flow(masked)
                usable = selected.unsqueeze(-1) & feature_mask
                loss = F.huber_loss(reconstruction[usable], target[usable], delta=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if progress_callback is not None:
            progress_callback(epoch + 1, epochs)


def fit_graph_residual(
    model: ETFStockGraphForecaster,
    dataset: GraphDataset,
    dates: Sequence[str],
    residuals: Mapping[str, np.ndarray],
    baselines: Mapping[str, np.ndarray],
    stock_stats: FeatureStats,
    etf_stats: FeatureStats,
    target_stats: FeatureStats,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    bf16: bool,
    flow_only_loss_weight: float,
    common_flow_only_loss_weight: float,
    rotation_flow_only_loss_weight: float,
    linked_flow_only_loss_weight: float,
    rotation_correlation_loss_weight: float,
    relation_only_loss_weight: float,
    flow_reconstruction_loss_weight: float,
    relation_gate_l1_weight: float,
    flow_encoder_lr_scale: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    flow_parameters = list(model.etf_temporal.parameters()) + list(
        model.reconstruction_head.parameters()
    )
    flow_parameter_ids = {id(parameter) for parameter in flow_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in flow_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": learning_rate},
            {
                "params": flow_parameters,
                "lr": learning_rate * flow_encoder_lr_scale,
            },
        ],
        weight_decay=weight_decay,
    )
    _, target_std = target_stats.tensors(device)
    rank_indices = (3, 9)
    relative_target_indices = ROTATION_FLOW_TARGET_INDICES
    for epoch in range(epochs):
        model.train()
        for date in dates:
            snapshot = dataset.load(date)
            inputs = _inputs(snapshot, stock_stats, etf_stats, device)
            residual = torch.as_tensor(
                residuals[date], dtype=torch.float32, device=device
            )
            normalized_residual = residual / target_std
            actual = torch.as_tensor(
                snapshot.targets, dtype=torch.float32, device=device
            )
            baseline = torch.as_tensor(
                baselines[date], dtype=torch.float32, device=device
            )
            mask = torch.as_tensor(snapshot.target_mask, dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, bf16):
                output = model(**inputs)
                point_loss = _masked_huber(
                    output.residual_point, normalized_residual, mask
                )
                relation_only_loss = _masked_huber(
                    output.relation_residual, normalized_residual, mask
                )
                flow_only_loss = _masked_huber(
                    output.dynamic_flow_residual, normalized_residual, mask
                )
                common_mask = torch.zeros_like(mask)
                common_mask[:, COMMON_FLOW_TARGET_INDICES] = mask[
                    :, COMMON_FLOW_TARGET_INDICES
                ]
                rotation_mask = torch.zeros_like(mask)
                rotation_mask[:, ROTATION_FLOW_TARGET_INDICES] = mask[
                    :, ROTATION_FLOW_TARGET_INDICES
                ]
                common_flow_only_loss = _masked_huber(
                    output.common_flow_residual,
                    normalized_residual,
                    common_mask,
                )
                rotation_flow_only_loss = _masked_huber(
                    output.rotation_flow_residual,
                    normalized_residual,
                    rotation_mask,
                )
                linked_flow_only_loss = _masked_huber(
                    output.linked_flow_residual, normalized_residual, mask
                )
                rotation_correlation_loss = _masked_cross_sectional_correlation_loss(
                    output.rotation_flow_residual,
                    normalized_residual,
                    mask,
                    relative_target_indices,
                )
                quantile_loss = _pinball_loss(
                    output.residual_quantiles,
                    normalized_residual,
                    mask,
                    (0.1, 0.5, 0.9),
                )
                direction_target = actual[:, DIRECTION_TARGET_INDICES] > 0
                direction_mask = mask[:, DIRECTION_TARGET_INDICES]
                direction_loss = F.binary_cross_entropy_with_logits(
                    output.direction_logits[direction_mask],
                    direction_target.to(output.direction_logits.dtype)[direction_mask],
                )
                final_raw = baseline + output.residual_point * target_std
                rank_loss = torch.stack(
                    [
                        _pairwise_rank_loss(
                            final_raw[:, index], actual[:, index], mask[:, index]
                        )
                        for index in rank_indices
                    ]
                ).mean()
                raw_width = inputs["etf_x"].shape[-1] // 2
                reconstruction_target = inputs["etf_x"][..., :raw_width]
                reconstruction_mask = inputs["etf_x"][..., raw_width:] > 0
                if reconstruction_mask.any():
                    reconstruction_loss = F.huber_loss(
                        output.flow_reconstruction[reconstruction_mask],
                        reconstruction_target[reconstruction_mask],
                        delta=1.0,
                    )
                else:
                    reconstruction_loss = output.flow_reconstruction.sum() * 0.0
                loss = (
                    point_loss
                    + 0.35 * quantile_loss
                    + 0.20 * direction_loss
                    + 0.10 * rank_loss
                    + relation_only_loss_weight * relation_only_loss
                    + flow_only_loss_weight * flow_only_loss
                    + common_flow_only_loss_weight * common_flow_only_loss
                    + rotation_flow_only_loss_weight * rotation_flow_only_loss
                    + linked_flow_only_loss_weight * linked_flow_only_loss
                    + rotation_correlation_loss_weight * rotation_correlation_loss
                    + flow_reconstruction_loss_weight * reconstruction_loss
                    + relation_gate_l1_weight * output.relation_gate.mean()
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if progress_callback is not None:
            progress_callback(epoch + 1, epochs)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = (start + end - 1) / 2.0
        start = end
    result = np.empty_like(sorted_ranks)
    result[order] = sorted_ranks
    return result


def _spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(prediction) & np.isfinite(target)
    if finite.sum() < 3:
        return math.nan
    a = _rank(prediction[finite])
    b = _rank(target[finite])
    if a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate(
    price_model: PriceBaseline,
    graph_model: ETFStockGraphForecaster,
    dataset: GraphDataset,
    dates: Sequence[str],
    stock_stats: FeatureStats,
    etf_stats: FeatureStats,
    target_stats: FeatureStats,
    *,
    device: torch.device,
    bf16: bool,
) -> dict[str, object]:
    price_model.eval()
    graph_model.eval()
    _, target_std = target_stats.tensors(device)
    collected_target = []
    collected_mask = []
    collected_price = []
    collected_full = []
    collected_relation = []
    collected_flow = []
    collected_global_flow = []
    collected_common_flow = []
    collected_rotation_flow = []
    collected_linked_flow = []
    collected_relation_prediction = []
    collected_flow_prediction = []
    collected_global_flow_prediction = []
    collected_common_flow_prediction = []
    collected_rotation_flow_prediction = []
    collected_linked_flow_prediction = []
    collected_shuffled_prediction = []
    collected_lagged_prediction = []
    collected_shuffled_query_prediction = []
    collected_zero_flow_prediction = []
    collected_zero_flow_residual = []
    collected_common_alignment = []
    collected_rotation_alignment = []
    factor_convergence_by_date = []
    factor_dispersion_by_date = []
    collected_logits = []
    daily_ic: dict[str, list[float]] = {"return_5d": [], "return_20d": []}
    control_daily_ic: dict[str, dict[str, list[float]]] = {
        variant: {"return_5d": [], "return_20d": []}
        for variant in (
            "price",
            "relation_only",
            "flow_only",
            "global_flow_only",
            "common_flow_only",
            "rotation_flow_only",
            "linked_flow_only",
            "without_global_flow",
            "without_common_flow",
            "without_rotation_flow",
            "without_linked_flow",
            "shuffled_flow",
            "lagged_flow_5_sessions",
            "shuffled_stock_query",
            "zero_flow",
        )
    }
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(1729)
    for date in dates:
        snapshot = dataset.load(date)
        baseline = predict_price(
            price_model, snapshot, stock_stats, target_stats, device, bf16
        )
        inputs = _inputs(snapshot, stock_stats, etf_stats, device)
        with _autocast(device, bf16):
            full = graph_model(**inputs)
            shuffled_inputs = dict(inputs)
            permutation = torch.randperm(
                inputs["etf_x"].shape[0],
                generator=shuffle_generator,
            ).to(device)
            shuffled_inputs["etf_x"] = inputs["etf_x"][permutation]
            raw_width = inputs["etf_x"].shape[-1] // 2
            lagged_inputs = dict(inputs)
            lagged_etf_x = torch.zeros_like(inputs["etf_x"])
            lag_sessions = min(5, inputs["etf_x"].shape[1])
            if lag_sessions < inputs["etf_x"].shape[1]:
                lagged_etf_x[:, lag_sessions:] = inputs["etf_x"][:, :-lag_sessions]
            lagged_inputs["etf_x"] = lagged_etf_x
            query_inputs = dict(inputs)
            stock_permutation = torch.randperm(
                inputs["stock_x"].shape[0],
                generator=shuffle_generator,
            ).to(device)
            query_inputs["stock_x"] = inputs["stock_x"][stock_permutation]
            query_inputs["stock_ids"] = inputs["stock_ids"][stock_permutation]
            zero_flow_inputs = dict(inputs)
            zero_flow_inputs["etf_x"] = torch.cat(
                (
                    torch.zeros_like(inputs["etf_x"][..., :raw_width]),
                    inputs["etf_x"][..., raw_width:],
                ),
                dim=-1,
            )
            shuffled = graph_model(**shuffled_inputs)
            lagged = graph_model(**lagged_inputs)
            shuffled_query = graph_model(**query_inputs)
            zero_flow = graph_model(**zero_flow_inputs)
        full_residual = (full.residual_point.float() * target_std).cpu().numpy()
        relation = (full.relation_residual.float() * target_std).cpu().numpy()
        flow = (full.dynamic_flow_residual.float() * target_std).cpu().numpy()
        global_flow = (full.global_flow_residual.float() * target_std).cpu().numpy()
        common_flow = (full.common_flow_residual.float() * target_std).cpu().numpy()
        rotation_flow = (full.rotation_flow_residual.float() * target_std).cpu().numpy()
        linked_flow = (full.linked_flow_residual.float() * target_std).cpu().numpy()
        shuffled_residual = (
            shuffled.residual_point.float() * target_std
        ).cpu().numpy()
        lagged_residual = (
            lagged.residual_point.float() * target_std
        ).cpu().numpy()
        shuffled_query_rotation = (
            shuffled_query.rotation_flow_residual.float() * target_std
        ).cpu().numpy()
        zero_flow_raw_residual = (
            zero_flow.residual_point.float() * target_std
        ).cpu().numpy()
        zero_flow_dynamic_residual = (
            zero_flow.dynamic_flow_residual.float() * target_std
        ).cpu().numpy()
        final = baseline + full_residual
        relation_prediction = baseline + relation
        flow_prediction = baseline + flow
        global_flow_prediction = baseline + global_flow
        common_flow_prediction = baseline + common_flow
        rotation_flow_prediction = baseline + rotation_flow
        linked_flow_prediction = baseline + linked_flow
        without_global_flow_prediction = baseline + relation + linked_flow
        without_common_flow_prediction = baseline + relation + rotation_flow + linked_flow
        without_rotation_flow_prediction = baseline + relation + common_flow + linked_flow
        without_linked_flow_prediction = baseline + relation + global_flow
        shuffled_prediction = baseline + shuffled_residual
        lagged_prediction = baseline + lagged_residual
        shuffled_query_prediction = (
            baseline + relation + common_flow + shuffled_query_rotation + linked_flow
        )
        zero_flow_prediction = baseline + zero_flow_raw_residual
        collected_target.append(snapshot.targets)
        collected_mask.append(snapshot.target_mask)
        collected_price.append(baseline)
        collected_full.append(final)
        collected_relation.append(relation)
        collected_flow.append(flow)
        collected_global_flow.append(global_flow)
        collected_common_flow.append(common_flow)
        collected_rotation_flow.append(rotation_flow)
        collected_linked_flow.append(linked_flow)
        collected_relation_prediction.append(relation_prediction)
        collected_flow_prediction.append(flow_prediction)
        collected_global_flow_prediction.append(global_flow_prediction)
        collected_common_flow_prediction.append(common_flow_prediction)
        collected_rotation_flow_prediction.append(rotation_flow_prediction)
        collected_linked_flow_prediction.append(linked_flow_prediction)
        collected_shuffled_prediction.append(shuffled_prediction)
        collected_lagged_prediction.append(lagged_prediction)
        collected_shuffled_query_prediction.append(shuffled_query_prediction)
        collected_zero_flow_prediction.append(zero_flow_prediction)
        collected_zero_flow_residual.append(zero_flow_dynamic_residual)
        collected_common_alignment.append(
            full.common_price_alignment.float().cpu().numpy()
        )
        collected_rotation_alignment.append(
            full.rotation_price_alignment.float().cpu().numpy()
        )
        factor_convergence_by_date.append(float(full.factor_convergence.float().cpu()))
        factor_dispersion_by_date.append(float(full.factor_dispersion.float().cpu()))
        collected_logits.append(full.direction_logits.float().cpu().numpy())
        daily_ic["return_5d"].append(
            _spearman(final[:, 0], snapshot.targets[:, 0])
        )
        daily_ic["return_20d"].append(
            _spearman(final[:, 6], snapshot.targets[:, 6])
        )
        controls = {
            "price": baseline,
            "relation_only": relation_prediction,
            "flow_only": flow_prediction,
            "global_flow_only": global_flow_prediction,
            "common_flow_only": common_flow_prediction,
            "rotation_flow_only": rotation_flow_prediction,
            "linked_flow_only": linked_flow_prediction,
            "without_global_flow": without_global_flow_prediction,
            "without_common_flow": without_common_flow_prediction,
            "without_rotation_flow": without_rotation_flow_prediction,
            "without_linked_flow": without_linked_flow_prediction,
            "shuffled_flow": shuffled_prediction,
            "lagged_flow_5_sessions": lagged_prediction,
            "shuffled_stock_query": shuffled_query_prediction,
            "zero_flow": zero_flow_prediction,
        }
        for variant, prediction in controls.items():
            control_daily_ic[variant]["return_5d"].append(
                _spearman(prediction[:, 0], snapshot.targets[:, 0])
            )
            control_daily_ic[variant]["return_20d"].append(
                _spearman(prediction[:, 6], snapshot.targets[:, 6])
            )
    target = np.concatenate(collected_target)
    mask = np.concatenate(collected_mask).astype(bool)
    price = np.concatenate(collected_price)
    final = np.concatenate(collected_full)
    relation = np.concatenate(collected_relation)
    flow = np.concatenate(collected_flow)
    global_flow = np.concatenate(collected_global_flow)
    common_flow = np.concatenate(collected_common_flow)
    rotation_flow = np.concatenate(collected_rotation_flow)
    linked_flow = np.concatenate(collected_linked_flow)
    relation_prediction = np.concatenate(collected_relation_prediction)
    flow_prediction = np.concatenate(collected_flow_prediction)
    global_flow_prediction = np.concatenate(collected_global_flow_prediction)
    common_flow_prediction = np.concatenate(collected_common_flow_prediction)
    rotation_flow_prediction = np.concatenate(collected_rotation_flow_prediction)
    linked_flow_prediction = np.concatenate(collected_linked_flow_prediction)
    shuffled_prediction = np.concatenate(collected_shuffled_prediction)
    lagged_prediction = np.concatenate(collected_lagged_prediction)
    shuffled_query_prediction = np.concatenate(collected_shuffled_query_prediction)
    zero_flow_prediction = np.concatenate(collected_zero_flow_prediction)
    zero_flow_residual = np.concatenate(collected_zero_flow_residual)
    without_global_flow_prediction = price + relation + linked_flow
    without_common_flow_prediction = price + relation + rotation_flow + linked_flow
    without_rotation_flow_prediction = price + relation + common_flow + linked_flow
    without_linked_flow_prediction = price + relation + global_flow
    common_alignment = np.concatenate(collected_common_alignment)
    rotation_alignment = np.concatenate(collected_rotation_alignment)
    logits = np.concatenate(collected_logits)
    target_metrics = {}
    for index, name in enumerate(TARGET_COLUMNS):
        valid = mask[:, index] & np.isfinite(target[:, index])
        price_mae = float(np.mean(np.abs(price[valid, index] - target[valid, index])))
        graph_mae = float(np.mean(np.abs(final[valid, index] - target[valid, index])))
        relation_mae = float(
            np.mean(np.abs(relation_prediction[valid, index] - target[valid, index]))
        )
        flow_mae = float(
            np.mean(np.abs(flow_prediction[valid, index] - target[valid, index]))
        )
        global_flow_mae = float(
            np.mean(
                np.abs(global_flow_prediction[valid, index] - target[valid, index])
            )
        )
        common_flow_mae = float(
            np.mean(
                np.abs(common_flow_prediction[valid, index] - target[valid, index])
            )
        )
        rotation_flow_mae = float(
            np.mean(
                np.abs(rotation_flow_prediction[valid, index] - target[valid, index])
            )
        )
        linked_flow_mae = float(
            np.mean(
                np.abs(linked_flow_prediction[valid, index] - target[valid, index])
            )
        )
        shuffled_mae = float(
            np.mean(np.abs(shuffled_prediction[valid, index] - target[valid, index]))
        )
        lagged_mae = float(
            np.mean(np.abs(lagged_prediction[valid, index] - target[valid, index]))
        )
        shuffled_query_mae = float(
            np.mean(
                np.abs(shuffled_query_prediction[valid, index] - target[valid, index])
            )
        )
        zero_flow_mae = float(
            np.mean(np.abs(zero_flow_prediction[valid, index] - target[valid, index]))
        )
        without_global_flow_mae = float(
            np.mean(
                np.abs(
                    without_global_flow_prediction[valid, index]
                    - target[valid, index]
                )
            )
        )
        without_common_flow_mae = float(
            np.mean(
                np.abs(
                    without_common_flow_prediction[valid, index]
                    - target[valid, index]
                )
            )
        )
        without_rotation_flow_mae = float(
            np.mean(
                np.abs(
                    without_rotation_flow_prediction[valid, index]
                    - target[valid, index]
                )
            )
        )
        without_linked_flow_mae = float(
            np.mean(
                np.abs(
                    without_linked_flow_prediction[valid, index]
                    - target[valid, index]
                )
            )
        )
        target_metrics[name] = {
            "rows": int(valid.sum()),
            "price_mae_pct": price_mae,
            "relation_only_mae_pct": relation_mae,
            "flow_only_mae_pct": flow_mae,
            "global_flow_only_mae_pct": global_flow_mae,
            "common_flow_only_mae_pct": common_flow_mae,
            "rotation_flow_only_mae_pct": rotation_flow_mae,
            "linked_flow_only_mae_pct": linked_flow_mae,
            "shuffled_flow_mae_pct": shuffled_mae,
            "lagged_flow_5_sessions_mae_pct": lagged_mae,
            "shuffled_stock_query_mae_pct": shuffled_query_mae,
            "zero_flow_mae_pct": zero_flow_mae,
            "without_global_flow_mae_pct": without_global_flow_mae,
            "without_common_flow_mae_pct": without_common_flow_mae,
            "without_rotation_flow_mae_pct": without_rotation_flow_mae,
            "without_linked_flow_mae_pct": without_linked_flow_mae,
            "graph_mae_pct": graph_mae,
            "flow_incremental_vs_relation_mae_pct": relation_mae - graph_mae,
            "flow_incremental_vs_zero_flow_mae_pct": zero_flow_mae - graph_mae,
            "flow_specific_vs_shuffled_mae_pct": shuffled_mae - graph_mae,
            "flow_timeliness_vs_lagged_mae_pct": lagged_mae - graph_mae,
            "stock_query_specific_mae_pct": shuffled_query_mae - graph_mae,
            "global_flow_incremental_mae_pct": without_global_flow_mae - graph_mae,
            "common_flow_incremental_mae_pct": without_common_flow_mae - graph_mae,
            "rotation_flow_incremental_mae_pct": without_rotation_flow_mae - graph_mae,
            "linked_flow_incremental_mae_pct": without_linked_flow_mae - graph_mae,
            "mean_abs_relation_edge_pct": float(np.mean(np.abs(relation[valid, index]))),
            "mean_abs_dynamic_flow_edge_pct": float(np.mean(np.abs(flow[valid, index]))),
            "mean_abs_global_flow_pct": float(
                np.mean(np.abs(global_flow[valid, index]))
            ),
            "mean_abs_common_flow_pct": float(
                np.mean(np.abs(common_flow[valid, index]))
            ),
            "mean_abs_rotation_flow_pct": float(
                np.mean(np.abs(rotation_flow[valid, index]))
            ),
            "mean_abs_linked_flow_pct": float(
                np.mean(np.abs(linked_flow[valid, index]))
            ),
            "mean_abs_dynamic_flow_input_effect_pct": float(
                np.mean(np.abs(flow[valid, index] - zero_flow_residual[valid, index]))
            ),
            "zero_flow_max_abs_dynamic_pct": float(
                np.max(np.abs(zero_flow_residual[valid, index]))
            ),
        }
    direction_actual = target[:, DIRECTION_TARGET_INDICES] > 0
    direction_valid = mask[:, DIRECTION_TARGET_INDICES]
    direction_pred = logits > 0
    direction = {}
    for index, target_index in enumerate(DIRECTION_TARGET_INDICES):
        valid = direction_valid[:, index]
        direction[TARGET_COLUMNS[target_index]] = {
            "rows": int(valid.sum()),
            "accuracy": float(
                np.mean(direction_pred[valid, index] == direction_actual[valid, index])
            ),
        }

    def alignment_resolution(values: np.ndarray) -> dict[str, object]:
        groups: dict[str, object] = {}
        for label, selector in (
            ("convergence", values >= 0.0),
            ("divergence", values < 0.0),
        ):
            metrics: dict[str, object] = {}
            for index, name in enumerate(TARGET_COLUMNS):
                valid = selector & mask[:, index] & np.isfinite(target[:, index])
                if not valid.any():
                    continue
                metrics[name] = {
                    "rows": int(valid.sum()),
                    "actual_mean_pct": float(np.mean(target[valid, index])),
                    "price_mae_pct": float(
                        np.mean(np.abs(price[valid, index] - target[valid, index]))
                    ),
                    "graph_mae_pct": float(
                        np.mean(np.abs(final[valid, index] - target[valid, index]))
                    ),
                }
                metrics[name]["graph_improvement_vs_price_pct"] = (
                    float(metrics[name]["price_mae_pct"])
                    - float(metrics[name]["graph_mae_pct"])
                )
            groups[label] = {
                "row_count": int(selector.sum()),
                "targets": metrics,
            }
        return groups

    return {
        "dates": [dates[0], dates[-1]],
        "date_count": len(dates),
        "row_count": int(len(target)),
        "targets": target_metrics,
        "direction": direction,
        "daily_spearman_ic": {
            key: {
                "mean": float(np.nanmean(value)),
                "valid_dates": int(np.isfinite(value).sum()),
            }
            for key, value in daily_ic.items()
        },
        "control_daily_spearman_ic": {
            variant: {
                key: {
                    "mean": float(np.nanmean(value)),
                    "valid_dates": int(np.isfinite(value).sum()),
                }
                for key, value in horizons.items()
            }
            for variant, horizons in control_daily_ic.items()
        },
        "flow_input_controls": {
            "shuffle": "deterministic_random_etf_axis_seed_1729",
            "lag": "all Flow values and masks delayed by five sessions",
            "stock_query_shuffle": (
                "stock state and identity permuted; actual common/linked components retained"
            ),
            "zero_values": "normalized_values_zero_observation_masks_preserved",
        },
        "convergence_divergence_diagnostics": {
            "definition": (
                "learned latent agreement/opposition between all-ETF Flow factors "
                "and each stock price state; diagnostic only, never a label input"
            ),
            "factor_convergence_mean": float(np.mean(factor_convergence_by_date)),
            "factor_dispersion_mean": float(np.mean(factor_dispersion_by_date)),
            "common_flow_price": alignment_resolution(common_alignment),
            "rotation_flow_price": alignment_resolution(rotation_alignment),
        },
        "learned_gates": {
            "relation": {
                name: float(value)
                for name, value in zip(
                    TARGET_COLUMNS,
                    torch.sigmoid(graph_model.relation_gate_logits).detach().cpu().tolist(),
                )
            },
            "common_flow": {
                name: float(value)
                for name, value in zip(
                    TARGET_COLUMNS,
                    torch.sigmoid(graph_model.common_flow_gate_logits)
                    .detach()
                    .cpu()
                    .tolist(),
                )
            },
            "rotation_flow": {
                name: float(value)
                for name, value in zip(
                    TARGET_COLUMNS,
                    torch.sigmoid(graph_model.rotation_flow_gate_logits)
                    .detach()
                    .cpu()
                    .tolist(),
                )
            },
            "linked_flow": {
                name: float(value)
                for name, value in zip(
                    TARGET_COLUMNS,
                    torch.sigmoid(graph_model.linked_flow_gate_logits)
                    .detach()
                    .cpu()
                    .tolist(),
                )
            },
        },
        "interpretation": (
            "SMOKE_ONLY_PIPELINE_CHECK"
            if dataset.manifest.get("smoke_only")
            else "OUT_OF_SAMPLE_RESEARCH"
        ),
    }


def _jsonable_stats(stats: FeatureStats) -> dict[str, list[float]]:
    return {"mean": stats.mean.tolist(), "std": stats.std.tolist()}


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


def _torch_save_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _chronological_smoke_split(dates: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    if len(dates) < 10:
        raise ValueError("smoke needs at least 10 signal dates")
    price_end = max(4, int(len(dates) * 0.50))
    graph_end = max(price_end + 3, int(len(dates) * 0.80))
    graph_end = min(graph_end, len(dates) - 2)
    return list(dates[:price_end]), list(dates[price_end:graph_end]), list(dates[graph_end:])


def _train_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "seed": args.seed,
        "device": args.device,
        "bf16": args.bf16,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "temporal_layers": args.temporal_layers,
        "set_layers": args.set_layers,
        "graph_layers": args.graph_layers,
        "inducing_points": args.inducing_points,
        "dropout": args.dropout,
        "pretrain_epochs": args.pretrain_epochs,
        "price_epochs": args.price_epochs,
        "graph_epochs": args.graph_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "flow_only_loss_weight": args.flow_only_loss_weight,
        "common_flow_only_loss_weight": args.common_flow_only_loss_weight,
        "rotation_flow_only_loss_weight": args.rotation_flow_only_loss_weight,
        "linked_flow_only_loss_weight": args.linked_flow_only_loss_weight,
        "rotation_correlation_loss_weight": args.rotation_correlation_loss_weight,
        "relation_only_loss_weight": args.relation_only_loss_weight,
        "flow_reconstruction_loss_weight": args.flow_reconstruction_loss_weight,
        "relation_gate_l1_weight": args.relation_gate_l1_weight,
        "flow_encoder_lr_scale": args.flow_encoder_lr_scale,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    _seed_everything(args.seed)
    device = _device(args.device, args.cuda_memory_fraction)
    dataset = GraphDataset(args.dataset_root)
    if not dataset.manifest.get("smoke_only"):
        raise ValueError("train-smoke requires a dataset marked smoke_only")
    price_dates, graph_dates, validation_dates = _chronological_smoke_split(
        dataset.dates
    )
    stats_dates = price_dates + graph_dates
    stock_stats, etf_stats, target_stats = fit_feature_stats(dataset, stats_dates)
    price_model = fit_price_baseline(
        dataset,
        price_dates,
        stock_stats,
        target_stats,
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.price_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
    )
    baselines = {}
    residuals = {}
    for date in graph_dates:
        snapshot = dataset.load(date)
        baseline = predict_price(
            price_model, snapshot, stock_stats, target_stats, device, args.bf16
        )
        baselines[date] = baseline
        residuals[date] = snapshot.targets - baseline
    sample = dataset.load(graph_dates[0])
    graph_model = build_graph_model(
        dataset,
        sample,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        temporal_layers=args.temporal_layers,
        set_layers=args.set_layers,
        graph_layers=args.graph_layers,
        inducing_points=args.inducing_points,
        dropout=args.dropout,
    ).to(device)
    pretrain_flow_encoder(
        graph_model,
        dataset,
        price_dates + graph_dates,
        etf_stats,
        device=device,
        epochs=args.pretrain_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
    )
    fit_graph_residual(
        graph_model,
        dataset,
        graph_dates,
        residuals,
        baselines,
        stock_stats,
        etf_stats,
        target_stats,
        device=device,
        epochs=args.graph_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        flow_only_loss_weight=args.flow_only_loss_weight,
        common_flow_only_loss_weight=args.common_flow_only_loss_weight,
        rotation_flow_only_loss_weight=args.rotation_flow_only_loss_weight,
        linked_flow_only_loss_weight=args.linked_flow_only_loss_weight,
        rotation_correlation_loss_weight=args.rotation_correlation_loss_weight,
        relation_only_loss_weight=args.relation_only_loss_weight,
        flow_reconstruction_loss_weight=args.flow_reconstruction_loss_weight,
        relation_gate_l1_weight=args.relation_gate_l1_weight,
        flow_encoder_lr_scale=args.flow_encoder_lr_scale,
    )
    metrics = evaluate(
        price_model,
        graph_model,
        dataset,
        validation_dates,
        stock_stats,
        etf_stats,
        target_stats,
        device=device,
        bf16=args.bf16,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "smoke_checkpoint.pt"
    _torch_save_atomic(
        checkpoint_path,
        {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "price_model": price_model.state_dict(),
            "graph_model": graph_model.state_dict(),
            "stock_stats": _jsonable_stats(stock_stats),
            "etf_stats": _jsonable_stats(etf_stats),
            "target_stats": _jsonable_stats(target_stats),
            "configuration": _train_configuration(args),
        },
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "scope": "SMOKE_ONLY_NOT_PERFORMANCE_EVIDENCE",
        "dataset_root": str(dataset.root),
        "dataset_window": dataset.manifest["requested_window"],
        "requested_symbols": dataset.manifest["requested_symbols"],
        "split": {
            "price_baseline": [price_dates[0], price_dates[-1]],
            "oos_residual_training": [graph_dates[0], graph_dates[-1]],
            "validation": [validation_dates[0], validation_dates[-1]],
            "overlap_count": 0,
        },
        "configuration": _train_configuration(args),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "bf16_supported": bool(
                torch.cuda.is_bf16_supported() if device.type == "cuda" else False
            ),
            "precision": "BF16_AUTOCAST_FP32_MASTER" if args.bf16 else "FP32",
        },
        "parameters": {
            "price": parameter_count(price_model),
            "graph": parameter_count(graph_model),
        },
        "metrics": metrics,
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "side_effects": {
            "orders": 0,
            "emails": 0,
            "sheets_writes": 0,
            "scheduler_changes": 0,
            "service_changes": 0,
            "deployments": 0,
        },
    }
    _write_json_atomic(output_root / "smoke_receipt.json", receipt)
    return receipt


def _inner_oof_blocks(
    train_dates: Sequence[str], folds: int, purge_sessions: int
) -> list[tuple[list[str], list[str]]]:
    initial = max(60, len(train_dates) // 2)
    remaining = len(train_dates) - initial
    if remaining <= folds:
        raise ValueError("not enough dates for expanding OOF residual folds")
    boundaries = np.linspace(initial, len(train_dates), folds + 1, dtype=int)
    result = []
    for index in range(folds):
        validation_start = int(boundaries[index])
        validation_end = int(boundaries[index + 1])
        training_end = validation_start - purge_sessions
        if training_end < 40 or validation_end <= validation_start:
            continue
        result.append(
            (
                list(train_dates[:training_end]),
                list(train_dates[validation_start:validation_end]),
            )
        )
    if not result:
        raise ValueError("no valid inner OOF blocks after purge")
    return result


def run_walk_forward(args: argparse.Namespace) -> dict[str, object]:
    _seed_everything(args.seed)
    device = _device(args.device, args.cuda_memory_fraction)
    dataset = GraphDataset(args.dataset_root)
    dates = dataset.dates
    years = sorted({date[:4] for date in dates})
    test_years = years[args.min_train_years :]
    if args.test_year_start:
        test_years = [year for year in test_years if year >= args.test_year_start]
    if args.test_year_end:
        test_years = [year for year in test_years if year <= args.test_year_end]
    if args.max_folds:
        test_years = test_years[: args.max_folds]
    if not test_years:
        raise ValueError("no walk-forward outer folds selected")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_manifest_sha256 = sha256_file(dataset.root / "manifest.json")
    configuration = _train_configuration(args)
    fold_receipts = []
    resumed_fold_count = 0

    def progress(
        *,
        status: str,
        test_year: str | None,
        phase: str,
        fold_number: int | None = None,
    ) -> None:
        _write_json_atomic(
            output_root / "run_state.json",
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "updated_at_utc": utc_now(),
                "status": status,
                "dataset_root": str(dataset.root),
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "configuration": configuration,
                "selected_test_years": test_years,
                "completed_test_years": [item["test_year"] for item in fold_receipts],
                "current_test_year": test_year,
                "current_fold_number": fold_number,
                "phase": phase,
                "resume_enabled": bool(args.resume),
            },
        )

    progress(status="RUNNING", test_year=None, phase="initializing")
    for fold_number, test_year in enumerate(test_years, 1):
        fold_receipt_path = output_root / f"fold_{test_year}.json"
        checkpoint = output_root / f"fold_{test_year}.pt"
        if args.resume and fold_receipt_path.is_file() and checkpoint.is_file():
            prior = json.loads(fold_receipt_path.read_text(encoding="utf-8"))
            checkpoint_receipt = prior.get("checkpoint") or {}
            reusable = bool(
                prior.get("test_year") == test_year
                and prior.get("model_schema_version") == MODEL_SCHEMA_VERSION
                and prior.get("dataset_manifest_sha256") == dataset_manifest_sha256
                and prior.get("configuration") == configuration
                and checkpoint_receipt.get("sha256") == sha256_file(checkpoint)
            )
            if reusable:
                fold_receipts.append(prior)
                resumed_fold_count += 1
                progress(
                    status="RUNNING",
                    test_year=test_year,
                    fold_number=fold_number,
                    phase="reused_verified_fold",
                )
                continue
        test_dates = [date for date in dates if date.startswith(test_year)]
        first_test_index = dates.index(test_dates[0])
        train_end = first_test_index - args.purge_sessions
        outer_train_dates = dates[:train_end]
        if len(outer_train_dates) < 120:
            continue
        fold_started_at = utc_now()
        progress(
            status="RUNNING",
            test_year=test_year,
            fold_number=fold_number,
            phase="feature_statistics",
        )
        stock_stats, etf_stats, target_stats = fit_feature_stats(
            dataset, outer_train_dates
        )
        oof_baselines: dict[str, np.ndarray] = {}
        oof_residuals: dict[str, np.ndarray] = {}
        inner_blocks = _inner_oof_blocks(
            outer_train_dates, args.inner_folds, args.purge_sessions
        )
        for inner_number, (inner_train, inner_validation) in enumerate(
            inner_blocks, 1
        ):
            progress(
                status="RUNNING",
                test_year=test_year,
                fold_number=fold_number,
                phase=f"inner_oof_price_{inner_number}_of_{len(inner_blocks)}",
            )
            inner_price = fit_price_baseline(
                dataset,
                inner_train,
                stock_stats,
                target_stats,
                device=device,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                epochs=args.price_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                bf16=args.bf16,
            )
            for date in inner_validation:
                snapshot = dataset.load(date)
                baseline = predict_price(
                    inner_price,
                    snapshot,
                    stock_stats,
                    target_stats,
                    device,
                    args.bf16,
                )
                oof_baselines[date] = baseline
                oof_residuals[date] = snapshot.targets - baseline
            del inner_price
            if device.type == "cuda":
                torch.cuda.empty_cache()
        residual_dates = sorted(oof_residuals)
        progress(
            status="RUNNING",
            test_year=test_year,
            fold_number=fold_number,
            phase="final_price_baseline",
        )
        final_price = fit_price_baseline(
            dataset,
            outer_train_dates,
            stock_stats,
            target_stats,
            device=device,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            epochs=args.price_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
        )
        sample = dataset.load(residual_dates[0])
        graph_model = build_graph_model(
            dataset,
            sample,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            temporal_layers=args.temporal_layers,
            set_layers=args.set_layers,
            graph_layers=args.graph_layers,
            inducing_points=args.inducing_points,
            dropout=args.dropout,
        ).to(device)
        progress(
            status="RUNNING",
            test_year=test_year,
            fold_number=fold_number,
            phase="flow_reconstruction_pretrain",
        )
        pretrain_flow_encoder(
            graph_model,
            dataset,
            outer_train_dates,
            etf_stats,
            device=device,
            epochs=args.pretrain_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            progress_callback=lambda completed, total: progress(
                status="RUNNING",
                test_year=test_year,
                fold_number=fold_number,
                phase=f"flow_reconstruction_pretrain_epoch_{completed}_of_{total}",
            ),
        )
        progress(
            status="RUNNING",
            test_year=test_year,
            fold_number=fold_number,
            phase="graph_residual_training",
        )
        fit_graph_residual(
            graph_model,
            dataset,
            residual_dates,
            oof_residuals,
            oof_baselines,
            stock_stats,
            etf_stats,
            target_stats,
            device=device,
            epochs=args.graph_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            flow_only_loss_weight=args.flow_only_loss_weight,
            common_flow_only_loss_weight=args.common_flow_only_loss_weight,
            rotation_flow_only_loss_weight=args.rotation_flow_only_loss_weight,
            linked_flow_only_loss_weight=args.linked_flow_only_loss_weight,
            rotation_correlation_loss_weight=args.rotation_correlation_loss_weight,
            relation_only_loss_weight=args.relation_only_loss_weight,
            flow_reconstruction_loss_weight=args.flow_reconstruction_loss_weight,
            relation_gate_l1_weight=args.relation_gate_l1_weight,
            flow_encoder_lr_scale=args.flow_encoder_lr_scale,
            progress_callback=lambda completed, total: progress(
                status="RUNNING",
                test_year=test_year,
                fold_number=fold_number,
                phase=f"graph_residual_training_epoch_{completed}_of_{total}",
            ),
        )
        progress(
            status="RUNNING",
            test_year=test_year,
            fold_number=fold_number,
            phase="outer_evaluation",
        )
        metrics = evaluate(
            final_price,
            graph_model,
            dataset,
            test_dates,
            stock_stats,
            etf_stats,
            target_stats,
            device=device,
            bf16=args.bf16,
        )
        progress(
            status="RUNNING",
            test_year=test_year,
            fold_number=fold_number,
            phase="checkpointing",
        )
        _torch_save_atomic(
            checkpoint,
            {
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "price_model": final_price.state_dict(),
                "graph_model": graph_model.state_dict(),
                "stock_stats": _jsonable_stats(stock_stats),
                "etf_stats": _jsonable_stats(etf_stats),
                "target_stats": _jsonable_stats(target_stats),
                "configuration": _train_configuration(args),
            },
        )
        fold_receipt = {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "fold": fold_number,
            "test_year": test_year,
            "outer_train": [outer_train_dates[0], outer_train_dates[-1]],
            "outer_test": [test_dates[0], test_dates[-1]],
            "purge_sessions": args.purge_sessions,
            "fold_started_at_utc": fold_started_at,
            "fold_completed_at_utc": utc_now(),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "configuration": configuration,
            "oof_residual_dates": [residual_dates[0], residual_dates[-1]],
            "oof_residual_date_count": len(residual_dates),
            "metrics": metrics,
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            },
        }
        _write_json_atomic(fold_receipt_path, fold_receipt)
        fold_receipts.append(fold_receipt)
        progress(
            status="RUNNING",
            test_year=test_year,
            fold_number=fold_number,
            phase="fold_complete",
        )
        del final_price, graph_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS" if fold_receipts else "FAIL",
        "scope": "EXPANDING_WALK_FORWARD_RESEARCH",
        "dataset_root": str(dataset.root),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "configuration": configuration,
        "purge_sessions": args.purge_sessions,
        "inner_folds": args.inner_folds,
        "resume_enabled": bool(args.resume),
        "resumed_fold_count": resumed_fold_count,
        "folds": fold_receipts,
        "side_effects": {
            "orders": 0,
            "emails": 0,
            "sheets_writes": 0,
            "scheduler_changes": 0,
            "service_changes": 0,
            "deployments": 0,
        },
    }
    _write_json_atomic(output_root / "walk_forward_receipt.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError("walk-forward produced no accepted folds")
    progress(status="PASS", test_year=None, phase="complete")
    return receipt


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=float(os.environ.get("QUANT_FLOW_GRAPH_CUDA_MEMORY_FRACTION", "0.15")),
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--inducing-points", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--price-epochs", type=int, default=2)
    parser.add_argument("--graph-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--flow-only-loss-weight", type=float, default=0.25)
    parser.add_argument("--common-flow-only-loss-weight", type=float, default=0.15)
    parser.add_argument("--rotation-flow-only-loss-weight", type=float, default=0.15)
    parser.add_argument("--linked-flow-only-loss-weight", type=float, default=0.10)
    parser.add_argument("--rotation-correlation-loss-weight", type=float, default=0.10)
    parser.add_argument("--relation-only-loss-weight", type=float, default=0.05)
    parser.add_argument("--flow-reconstruction-loss-weight", type=float, default=0.05)
    parser.add_argument("--relation-gate-l1-weight", type=float, default=0.01)
    parser.add_argument("--flow-encoder-lr-scale", type=float, default=0.20)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("train-smoke")
    add_training_arguments(smoke)
    walk = subparsers.add_parser("walk-forward")
    add_training_arguments(walk)
    walk.set_defaults(price_epochs=10, graph_epochs=15, pretrain_epochs=3)
    walk.add_argument("--purge-sessions", type=int, default=PURGE_SESSIONS)
    walk.add_argument("--inner-folds", type=int, default=3)
    walk.add_argument("--min-train-years", type=int, default=3)
    walk.add_argument("--test-year-start")
    walk.add_argument("--test-year-end")
    walk.add_argument("--max-folds", type=int)
    walk.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "train-smoke":
        receipt = run_smoke(args)
    else:
        receipt = run_walk_forward(args)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
