"""BF16-ready temporal Set/Graph Transformer modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def normalized_with_mask(values: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Normalize finite values and append an explicit observation mask."""

    observed = torch.isfinite(values)
    normalized = (torch.nan_to_num(values, nan=0.0) - mean) / std.clamp_min(1e-6)
    normalized = torch.where(observed, normalized, torch.zeros_like(normalized))
    return torch.cat((normalized, observed.to(normalized.dtype)), dim=-1)


class MAB(nn.Module):
    """Multihead attention block used by the Set Transformer."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        attended, _ = self.attention(query, key_value, key_value, need_weights=False)
        hidden = self.norm1(query + self.dropout(attended))
        return self.norm2(hidden + self.dropout(self.ffn(hidden)))


class ISAB(nn.Module):
    """Induced Set Attention Block: all elements, bounded inducing set."""

    def __init__(
        self, hidden_dim: int, heads: int, inducing_points: int, dropout: float
    ) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, inducing_points, hidden_dim))
        nn.init.xavier_uniform_(self.inducing)
        self.to_inducing = MAB(hidden_dim, heads, dropout)
        self.to_elements = MAB(hidden_dim, heads, dropout)

    def forward(self, values: Tensor) -> Tensor:
        inducing = self.inducing.expand(values.shape[0], -1, -1)
        summary = self.to_inducing(inducing, values)
        return self.to_elements(values, summary)


class ETFTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        max_lookback: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim % 2:
            raise ValueError("ETF input must contain values followed by masks")
        self.raw_input_dim = input_dim // 2
        self.input_projection = nn.Linear(
            self.raw_input_dim, hidden_dim, bias=False
        )
        self.mask_gate = nn.Linear(self.raw_input_dim, hidden_dim, bias=False)
        self.reporting_key = nn.Linear(
            self.raw_input_dim * 2, hidden_dim, bias=False
        )
        self.position = nn.Parameter(torch.empty(1, max_lookback, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim, bias=False)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        length = values.shape[1]
        raw_values = values[..., : self.raw_input_dim]
        feature_mask = values[..., self.raw_input_dim :] > 0
        observed = feature_mask.any(dim=-1)
        masked_values = raw_values * feature_mask.to(raw_values.dtype)
        hidden = self.input_projection(masked_values)
        # Feature-specific availability changes how observed values are encoded,
        # while multiplication keeps an all-zero Flow tensor exactly neutral.
        hidden = hidden * (2.0 * torch.sigmoid(self.mask_gate(feature_mask.to(hidden.dtype))))
        hidden = hidden + hidden * self.position[:, :length]
        tokens = self.norm(self.encoder(hidden))
        weights = observed.to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        positions = torch.arange(length, device=values.device).view(1, length, 1)
        last_observed = torch.where(
            feature_mask,
            positions,
            torch.full_like(positions, -1),
        ).amax(dim=1)
        reporting_age = torch.where(
            last_observed >= 0,
            (length - 1 - last_observed).to(values.dtype) / max(length - 1, 1),
            torch.ones_like(last_observed, dtype=values.dtype),
        )
        coverage = feature_mask.to(values.dtype).mean(dim=1)
        reporting_key = self.reporting_key(torch.cat((coverage, reporting_age), dim=-1))
        return pooled, tokens, reporting_key, reporting_age


class BipartiteGraphAttention(nn.Module):
    """Stock queries attend only to ETFs linked by PIT holding edges."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        edge_dim: int,
        dropout: float,
        *,
        preserve_zero: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.preserve_zero = preserve_zero
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=not preserve_zero)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=not preserve_zero)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, heads),
        )
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=not preserve_zero)
        self.norm = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=not preserve_zero,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        stocks: Tensor,
        etfs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        *,
        key_etfs: Tensor | None = None,
        residual_connection: bool = True,
    ) -> tuple[Tensor, Tensor]:
        stock_count, hidden_dim = stocks.shape
        etf_count = etfs.shape[0]
        query = self.query(stocks).view(stock_count, self.heads, self.head_dim)
        key_source = etfs if key_etfs is None else key_etfs
        key = self.key(key_source).view(etf_count, self.heads, self.head_dim)
        value = self.value(etfs).view(etf_count, self.heads, self.head_dim)
        if not edge_index.numel():
            attention = torch.empty(
                (self.heads, 0), dtype=stocks.dtype, device=stocks.device
            )
            hidden = self.norm(stocks) if residual_connection else torch.zeros_like(stocks)
            return hidden, attention
        stock_ids = edge_index[0].long()
        etf_ids = edge_index[1].long()
        edge_logits = (
            (query[stock_ids] * key[etf_ids]).sum(dim=-1)
            / math.sqrt(self.head_dim)
        ).float()
        learned_bias = self.edge_bias(edge_attr).float()
        weight_bias = torch.log(edge_attr[:, 0].float().clamp_min(1e-6)).unsqueeze(1)
        edge_logits = edge_logits + learned_bias + weight_bias
        scatter_index = stock_ids.unsqueeze(1).expand(-1, self.heads)
        maximum = torch.full(
            (stock_count, self.heads),
            -torch.inf,
            dtype=torch.float32,
            device=stocks.device,
        )
        maximum.scatter_reduce_(
            0, scatter_index, edge_logits, reduce="amax", include_self=True
        )
        unnormalized = torch.exp(edge_logits - maximum[stock_ids])
        denominator = torch.zeros(
            (stock_count, self.heads), dtype=torch.float32, device=stocks.device
        )
        denominator.scatter_add_(0, scatter_index, unnormalized)
        edge_attention = unnormalized / denominator[stock_ids].clamp_min(1e-12)
        messages = edge_attention.unsqueeze(-1) * value[etf_ids].float()
        context = torch.zeros(
            (stock_count, self.heads, self.head_dim),
            dtype=torch.float32,
            device=stocks.device,
        )
        context.scatter_add_(
            0,
            stock_ids[:, None, None].expand_as(messages),
            messages,
        )
        context = context.reshape(stock_count, hidden_dim).to(stocks.dtype)
        valid_stock = denominator.sum(dim=1) > 0
        context = self.output(context)
        context = torch.where(valid_stock.unsqueeze(-1), context, torch.zeros_like(context))
        if residual_connection:
            hidden = self.norm(stocks + self.dropout(context))
        else:
            hidden = self.norm(self.dropout(context))
        return hidden, edge_attention.transpose(0, 1)


class ZeroPreservingFusion(nn.Module):
    """Nonlinear fusion whose exact zero input always returns exact zero."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        hidden = F.gelu(self.norm(self.projection(values)))
        return self.output(self.dropout(hidden))


class ZeroPreservingCrossAttention(nn.Module):
    """Cross-attention where query/key identity cannot leak without Flow value."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        query_hidden = self.query(query).view(-1, self.heads, self.head_dim)
        key_hidden = self.key(key).view(-1, self.heads, self.head_dim)
        value_hidden = self.value(value).view(-1, self.heads, self.head_dim)
        logits = torch.einsum(
            "qhd,khd->hqk",
            query_hidden,
            key_hidden,
        ).float() / math.sqrt(self.head_dim)
        attention = torch.softmax(logits, dim=-1)
        dropped_attention = F.dropout(
            attention,
            p=self.dropout.p,
            training=self.training,
        )
        context = torch.einsum(
            "hqk,khd->qhd",
            dropped_attention,
            value_hidden.float(),
        )
        context = context.reshape(query.shape[0], -1).to(query.dtype)
        hidden = self.norm1(self.output(context))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        return hidden, attention


def holding_weighted_pool(
    etfs: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    stock_count: int,
) -> Tensor:
    """Direct PIT holding-weighted ETF aggregation with no learned shortcut."""

    result = torch.zeros(
        (stock_count, etfs.shape[-1]),
        dtype=etfs.dtype,
        device=etfs.device,
    )
    if not edge_index.numel():
        return result
    stock_ids = edge_index[0].long()
    etf_ids = edge_index[1].long()
    weights = edge_attr[:, 0].to(etfs.dtype).clamp_min(0.0)
    denominator = torch.zeros(
        stock_count,
        dtype=etfs.dtype,
        device=etfs.device,
    )
    denominator.scatter_add_(0, stock_ids, weights)
    normalized = weights / denominator[stock_ids].clamp_min(1e-12)
    messages = normalized.unsqueeze(-1) * etfs[etf_ids]
    result.scatter_add_(
        0,
        stock_ids.unsqueeze(-1).expand_as(messages),
        messages,
    )
    return result


def latent_alignment(left: Tensor, right: Tensor) -> Tensor:
    """Cosine agreement in the learned Flow/price space, with zero neutrality."""

    numerator = (left.float() * right.float()).sum(dim=-1)
    denominator = left.float().norm(dim=-1) * right.float().norm(dim=-1)
    return torch.where(
        denominator > 1e-8,
        numerator / denominator.clamp_min(1e-8),
        torch.zeros_like(numerator),
    )


class OrderedQuantileHead(nn.Module):
    """Residual q10 <= q50 <= q90 by construction."""

    def __init__(self, hidden_dim: int, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.projection = nn.Linear(hidden_dim, target_dim * 3)

    def forward(self, hidden: Tensor) -> Tensor:
        raw = self.projection(hidden).view(-1, self.target_dim, 3)
        median = raw[..., 1]
        lower = median - F.softplus(raw[..., 0])
        upper = median + F.softplus(raw[..., 2])
        return torch.stack((lower, median, upper), dim=-1)


class PriceBaseline(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, target_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.point = nn.Linear(hidden_dim, target_dim)

    def forward(self, stock_x: Tensor) -> Tensor:
        return self.point(self.encoder(stock_x))


@dataclass
class GraphForecastOutput:
    residual_point: Tensor
    relation_residual: Tensor
    dynamic_flow_residual: Tensor
    global_flow_residual: Tensor
    common_flow_residual: Tensor
    rotation_flow_residual: Tensor
    linked_flow_residual: Tensor
    residual_quantiles: Tensor
    direction_logits: Tensor
    attention: Tensor
    relation_attention: Tensor
    global_etf_attention: Tensor
    global_stock_attention: Tensor
    common_factor_attention: Tensor
    global_flow_context: Tensor
    common_flow_context: Tensor
    rotation_flow_context: Tensor
    direct_flow_context: Tensor
    etf_temporal_tokens: Tensor
    flow_reconstruction: Tensor
    relation_gate: Tensor
    common_flow_gate: Tensor
    rotation_flow_gate: Tensor
    linked_flow_gate: Tensor
    factor_convergence: Tensor
    factor_dispersion: Tensor
    common_price_alignment: Tensor
    rotation_price_alignment: Tensor
    etf_reporting_age: Tensor


class ETFStockGraphForecaster(nn.Module):
    def __init__(
        self,
        *,
        stock_input_dim: int,
        stock_vocabulary_size: int,
        etf_input_dim: int,
        edge_dim: int,
        etf_vocabulary_size: int,
        target_dim: int,
        direction_dim: int,
        hidden_dim: int = 128,
        heads: int = 8,
        temporal_layers: int = 2,
        set_layers: int = 2,
        graph_layers: int = 2,
        inducing_points: int = 32,
        max_lookback: int = 60,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.etf_temporal = ETFTemporalEncoder(
            etf_input_dim,
            hidden_dim,
            heads,
            temporal_layers,
            max_lookback,
            dropout,
        )
        self.etf_identity = nn.Embedding(etf_vocabulary_size, hidden_dim)
        self.stock_identity = nn.Embedding(stock_vocabulary_size, hidden_dim)
        self.relation_set_blocks = nn.ModuleList(
            ISAB(hidden_dim, heads, inducing_points, dropout)
            for _ in range(set_layers)
        )
        self.global_flow_inducing = nn.Parameter(
            torch.empty(inducing_points, hidden_dim)
        )
        nn.init.xavier_uniform_(self.global_flow_inducing)
        self.etf_to_global_flow = ZeroPreservingCrossAttention(
            hidden_dim,
            heads,
            dropout,
        )
        self.flow_consensus_query = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.flow_consensus_query)
        self.factor_to_consensus = ZeroPreservingCrossAttention(
            hidden_dim,
            heads,
            dropout,
        )
        self.convergence_factor_fusion = ZeroPreservingFusion(
            hidden_dim * 3,
            hidden_dim,
            dropout,
        )
        self.global_stock_query = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rotation_flow_to_stock = ZeroPreservingCrossAttention(
            hidden_dim,
            heads,
            dropout,
        )
        self.stock_encoder = nn.Sequential(
            nn.Linear(stock_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.relation_graph_blocks = nn.ModuleList(
            BipartiteGraphAttention(hidden_dim, heads, edge_dim, dropout)
            for _ in range(graph_layers)
        )
        self.flow_graph_blocks = nn.ModuleList(
            BipartiteGraphAttention(
                hidden_dim,
                heads,
                edge_dim,
                dropout,
                preserve_zero=True,
            )
            for _ in range(graph_layers)
        )
        self.relation_global_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
        )
        self.relation_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.linked_flow_fusion = ZeroPreservingFusion(
            hidden_dim * 2,
            hidden_dim,
            dropout,
        )
        self.price_flow_probe = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.common_price_fusion = ZeroPreservingFusion(
            hidden_dim * 3,
            hidden_dim,
            dropout,
        )
        self.rotation_price_fusion = ZeroPreservingFusion(
            hidden_dim * 3,
            hidden_dim,
            dropout,
        )
        self.linked_price_fusion = ZeroPreservingFusion(
            hidden_dim * 3,
            hidden_dim,
            dropout,
        )
        self.combined_flow_fusion = ZeroPreservingFusion(
            hidden_dim * 2,
            hidden_dim,
            dropout,
        )
        self.full_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.relation_head = nn.Linear(hidden_dim, target_dim)
        self.common_flow_head = nn.Linear(hidden_dim, target_dim, bias=False)
        self.rotation_flow_head = nn.Linear(hidden_dim, target_dim, bias=False)
        self.linked_flow_head = nn.Linear(hidden_dim, target_dim, bias=False)
        self.relation_gate_logits = nn.Parameter(torch.full((target_dim,), -4.0))
        self.common_flow_gate_logits = nn.Parameter(torch.full((target_dim,), -2.0))
        self.rotation_flow_gate_logits = nn.Parameter(torch.full((target_dim,), -2.0))
        self.linked_flow_gate_logits = nn.Parameter(torch.full((target_dim,), -2.0))
        self.quantile_head = OrderedQuantileHead(hidden_dim, target_dim)
        self.direction_head = nn.Linear(hidden_dim, direction_dim)
        self.reconstruction_head = nn.Linear(
            hidden_dim,
            etf_input_dim // 2,
            bias=False,
        )

    def forward(
        self,
        *,
        stock_x: Tensor,
        stock_ids: Tensor,
        etf_x: Tensor,
        etf_ids: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        mask_dynamic_flow: bool = False,
        mask_global_flow: bool = False,
        mask_common_flow: bool = False,
        mask_rotation_flow: bool = False,
        mask_linked_flow: bool = False,
        mask_relation: bool = False,
    ) -> GraphForecastOutput:
        mask_global_flow = mask_global_flow or mask_dynamic_flow
        mask_common_flow = mask_common_flow or mask_global_flow
        mask_rotation_flow = mask_rotation_flow or mask_global_flow
        mask_linked_flow = mask_linked_flow or mask_dynamic_flow
        flow_etf_hidden, tokens, reporting_key, reporting_age = self.etf_temporal(
            etf_x
        )
        reconstruction = self.reconstruction_head(tokens)
        if mask_dynamic_flow:
            flow_etf_hidden = torch.zeros_like(flow_etf_hidden)

        etf_identity_hidden = self.etf_identity(etf_ids.long())
        relation_etf_hidden = etf_identity_hidden
        relation_set_hidden = relation_etf_hidden.unsqueeze(0)
        for block in self.relation_set_blocks:
            relation_set_hidden = block(relation_set_hidden)
        relation_etf_hidden = relation_set_hidden.squeeze(0)

        stock_hidden = self.stock_encoder(stock_x)
        stock_identity_hidden = self.stock_identity(stock_ids.long())
        global_stock_query = self.global_stock_query(
            torch.cat((stock_hidden, stock_identity_hidden), dim=-1)
        )
        global_flow_factors, global_etf_attention = self.etf_to_global_flow(
            self.global_flow_inducing,
            etf_identity_hidden + reporting_key + flow_etf_hidden,
            flow_etf_hidden,
        )
        consensus_context, common_factor_attention = self.factor_to_consensus(
            self.flow_consensus_query,
            global_flow_factors,
            global_flow_factors,
        )
        expanded_consensus = consensus_context.expand_as(global_flow_factors)
        factor_deviation = global_flow_factors - expanded_consensus
        convergence_factors = self.convergence_factor_fusion(
            torch.cat(
                (
                    global_flow_factors * expanded_consensus,
                    factor_deviation,
                    factor_deviation.abs(),
                ),
                dim=-1,
            )
        )
        rotation_flow_base, global_stock_attention = self.rotation_flow_to_stock(
            global_stock_query,
            convergence_factors,
            convergence_factors,
        )
        common_flow_base = consensus_context.expand_as(stock_hidden)
        price_flow_probe = torch.tanh(self.price_flow_probe(stock_hidden))
        common_flow_context = common_flow_base + self.common_price_fusion(
            torch.cat(
                (
                    common_flow_base,
                    common_flow_base * price_flow_probe,
                    common_flow_base.abs() * price_flow_probe,
                ),
                dim=-1,
            )
        )
        rotation_flow_context = rotation_flow_base + self.rotation_price_fusion(
            torch.cat(
                (
                    rotation_flow_base,
                    rotation_flow_base * price_flow_probe,
                    rotation_flow_base.abs() * price_flow_probe,
                ),
                dim=-1,
            )
        )
        if mask_common_flow:
            common_flow_context = torch.zeros_like(common_flow_context)
        if mask_rotation_flow:
            rotation_flow_context = torch.zeros_like(rotation_flow_context)
        global_flow_context = common_flow_context + rotation_flow_context

        relation_stock_hidden = stock_hidden
        relation_attention = torch.empty(0, device=stock_x.device)
        for block in self.relation_graph_blocks:
            relation_stock_hidden, relation_attention = block(
                relation_stock_hidden,
                relation_etf_hidden,
                edge_index,
                edge_attr,
            )

        linked_etf_hidden = flow_etf_hidden
        if mask_linked_flow:
            linked_etf_hidden = torch.zeros_like(linked_etf_hidden)
        linked_flow_stock_hidden = torch.zeros_like(stock_hidden)
        flow_attention = torch.empty(0, device=stock_x.device)
        for block in self.flow_graph_blocks:
            linked_flow_stock_hidden, flow_attention = block(
                stock_hidden + linked_flow_stock_hidden,
                linked_etf_hidden,
                edge_index,
                edge_attr,
                key_etfs=linked_etf_hidden + reporting_key,
                residual_connection=False,
            )
        direct_flow_context = holding_weighted_pool(
            linked_etf_hidden,
            edge_index,
            edge_attr,
            stock_hidden.shape[0],
        )

        relation_global = relation_etf_hidden.mean(dim=0, keepdim=True).expand_as(
            relation_stock_hidden
        )
        relation_global_gate = self.relation_global_gate(
            torch.cat((relation_stock_hidden, relation_global), dim=-1)
        )
        relation_fused = self.relation_fusion(
            torch.cat(
                (relation_stock_hidden, relation_global_gate * relation_global),
                dim=-1,
            )
        )
        linked_flow_context = direct_flow_context + self.linked_flow_fusion(
            torch.cat((linked_flow_stock_hidden, direct_flow_context), dim=-1)
        )
        linked_flow_context = linked_flow_context + self.linked_price_fusion(
            torch.cat(
                (
                    linked_flow_context,
                    linked_flow_context * price_flow_probe,
                    linked_flow_context.abs() * price_flow_probe,
                ),
                dim=-1,
            )
        )
        if mask_relation:
            relation_fused = torch.zeros_like(relation_fused)
        if mask_linked_flow:
            linked_flow_context = torch.zeros_like(linked_flow_context)

        relation_gate = torch.sigmoid(self.relation_gate_logits).unsqueeze(0)
        common_flow_gate = torch.sigmoid(self.common_flow_gate_logits).unsqueeze(0)
        rotation_flow_gate = torch.sigmoid(self.rotation_flow_gate_logits).unsqueeze(0)
        linked_flow_gate = torch.sigmoid(self.linked_flow_gate_logits).unsqueeze(0)
        relation_residual = relation_gate * self.relation_head(relation_fused)
        common_flow_residual = common_flow_gate * self.common_flow_head(
            common_flow_context
        )
        rotation_flow_residual = rotation_flow_gate * self.rotation_flow_head(
            rotation_flow_context
        )
        global_flow_residual = common_flow_residual + rotation_flow_residual
        linked_flow_residual = linked_flow_gate * self.linked_flow_head(
            linked_flow_context
        )
        stock_has_edge = torch.zeros(
            stock_hidden.shape[0], dtype=torch.bool, device=stock_hidden.device
        )
        if edge_index.numel():
            stock_has_edge[edge_index[0].long()] = True
        edge_mask = stock_has_edge.to(stock_hidden.dtype).unsqueeze(-1)
        relation_residual = relation_residual * edge_mask
        linked_flow_residual = linked_flow_residual * edge_mask
        if mask_relation:
            relation_residual = torch.zeros_like(relation_residual)
        if mask_common_flow:
            common_flow_residual = torch.zeros_like(common_flow_residual)
        if mask_rotation_flow:
            rotation_flow_residual = torch.zeros_like(rotation_flow_residual)
        global_flow_residual = common_flow_residual + rotation_flow_residual
        if mask_linked_flow:
            linked_flow_residual = torch.zeros_like(linked_flow_residual)
        dynamic_flow_residual = global_flow_residual + linked_flow_residual
        residual_point = relation_residual + dynamic_flow_residual
        combined_flow_context = (
            global_flow_context
            + linked_flow_context
            + self.combined_flow_fusion(
                torch.cat((global_flow_context, linked_flow_context), dim=-1)
            )
        )
        full_hidden = self.full_fusion(
            torch.cat((relation_fused, combined_flow_context), dim=-1)
        )
        return GraphForecastOutput(
            residual_point=residual_point,
            relation_residual=relation_residual,
            dynamic_flow_residual=dynamic_flow_residual,
            global_flow_residual=global_flow_residual,
            common_flow_residual=common_flow_residual,
            rotation_flow_residual=rotation_flow_residual,
            linked_flow_residual=linked_flow_residual,
            residual_quantiles=self.quantile_head(full_hidden),
            direction_logits=self.direction_head(full_hidden),
            attention=flow_attention,
            relation_attention=relation_attention,
            global_etf_attention=global_etf_attention,
            global_stock_attention=global_stock_attention,
            common_factor_attention=common_factor_attention,
            global_flow_context=global_flow_context,
            common_flow_context=common_flow_context,
            rotation_flow_context=rotation_flow_context,
            direct_flow_context=direct_flow_context,
            etf_temporal_tokens=tokens,
            flow_reconstruction=reconstruction,
            relation_gate=relation_gate,
            common_flow_gate=common_flow_gate,
            rotation_flow_gate=rotation_flow_gate,
            linked_flow_gate=linked_flow_gate,
            factor_convergence=latent_alignment(
                global_flow_factors, expanded_consensus
            ).mean(),
            factor_dispersion=factor_deviation.float().square().mean().sqrt(),
            common_price_alignment=latent_alignment(
                common_flow_base, price_flow_probe
            ),
            rotation_price_alignment=latent_alignment(
                rotation_flow_base, price_flow_probe
            ),
            etf_reporting_age=reporting_age,
        )

    def reconstruct_flow(self, etf_x: Tensor) -> Tensor:
        _, tokens, _, _ = self.etf_temporal(etf_x)
        return self.reconstruction_head(tokens)


def parameter_count(module: nn.Module) -> Mapping[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}
