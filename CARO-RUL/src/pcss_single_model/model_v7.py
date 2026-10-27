from __future__ import annotations

import math

import torch
from torch import nn

from pcss_single_model.model_v2 import PointDecoder, StandardCausalBlock
from pcss_single_model.model_v4 import OrderedQuantileRiskHead


class SequenceTCN(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(
            *(StandardCausalBlock(hidden, dilation) for dilation in (1, 2, 4, 8))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(x))


class CausalDegradationAttentionPooling(nn.Module):
    """Module I: pool causal history using state, trend, and reliability.

    Unlike last-step readout, CDAP can down-weight isolated acquisition plateaus
    while retaining the direction of recent degradation. It is a single
    trainable pooling module, not a collection of independently fitted models.
    """

    def __init__(self, hidden: int, feature_count: int = 14, residual_scale: float = 0.35):
        super().__init__()
        self.feature_count = feature_count
        self.residual_scale = residual_scale
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Conv1d(hidden, hidden, 1, bias=False)
        self.value = nn.Conv1d(hidden, hidden, 1, bias=False)
        self.recency = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3 + feature_count + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Tanh()
        )

    def forward(self, sequence: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last = sequence[..., -1]
        query = self.query(last).unsqueeze(-1)
        scores = (self.key(sequence) * query).sum(dim=1) / math.sqrt(sequence.shape[1])
        time = torch.linspace(-1.0, 0.0, sequence.shape[-1], device=sequence.device)
        scores = scores + torch.nn.functional.softplus(self.recency) * time
        # Lower reliability reduces attention without masking the entire step.
        f = self.feature_count
        reliability = 1.0 - raw[:, 2 * f : 3 * f].mean(dim=1).clamp(0.0, 1.0)
        scores = scores + torch.log(reliability.clamp_min(0.05))
        attention = torch.softmax(scores, dim=-1)
        pooled = (self.value(sequence) * attention.unsqueeze(1)).sum(dim=-1)
        recent = sequence[..., -16:].mean(dim=-1)
        earlier = sequence[..., -32:-16].mean(dim=-1)
        trend = recent - earlier
        missing_rate = raw[:, 2 * f : 3 * f].mean(dim=-1)
        age = raw[:, 3 * f, -1:]
        gate = self.gate(torch.cat([last, pooled, trend, missing_rate, age], dim=-1))
        update = self.update(torch.cat([pooled - last, trend], dim=-1))
        return last + self.residual_scale * gate * update, attention


class CDAPRUL(nn.Module):
    def __init__(self, backbone, pool, decoder):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence = self.backbone(x)
        if self.pool is None:
            hidden, attention = sequence[..., -1], None
        else:
            hidden, attention = self.pool(sequence, x)
        output = self.decoder(hidden)
        if attention is not None:
            output["temporal_attention"] = attention
        return output


def build_model(name: str, channels: int, hidden: int, bins: int):
    del bins
    variants = {
        "tcn_point": (False, False),
        "cdap_point": (True, False),
        "tcn_oqrh": (False, True),
        "caro_rul": (True, True),
    }
    if name not in variants:
        raise KeyError(name)
    use_pool, use_quantiles = variants[name]
    # Identical construction stream gives paired backbone/head initialization.
    backbone = SequenceTCN(channels, hidden)
    point = PointDecoder(hidden)
    candidate_pool = CausalDegradationAttentionPooling(hidden)
    quantile = OrderedQuantileRiskHead(hidden)
    return CDAPRUL(
        backbone,
        candidate_pool if use_pool else None,
        quantile if use_quantiles else point,
    )
