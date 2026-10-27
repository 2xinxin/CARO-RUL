from __future__ import annotations

import math

import torch

from pcss_single_model.model_v4 import OrderedQuantileRiskHead
from pcss_single_model.model_v7 import CDAPRUL, SequenceTCN
from pcss_single_model.model_v9 import IntervalAwareOrderedRiskHead


class MechanismCDAP(torch.nn.Module):
    """CDAP with one prespecified mechanism removed for reviewer ablation."""

    def __init__(
        self,
        hidden: int,
        *,
        use_recency: bool = True,
        use_reliability: bool = True,
        use_trend: bool = True,
        feature_count: int = 14,
        residual_scale: float = 0.10,
    ):
        super().__init__()
        self.feature_count = feature_count
        self.residual_scale = residual_scale
        self.use_recency = use_recency
        self.use_reliability = use_reliability
        self.use_trend = use_trend
        self.query = torch.nn.Linear(hidden, hidden, bias=False)
        self.key = torch.nn.Conv1d(hidden, hidden, 1, bias=False)
        self.value = torch.nn.Conv1d(hidden, hidden, 1, bias=False)
        self.recency = torch.nn.Parameter(torch.tensor(1.0))
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden * 3 + feature_count + 1, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Sigmoid(),
        )
        self.update = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden * 2), torch.nn.Linear(hidden * 2, hidden),
            torch.nn.Tanh(),
        )

    def forward(self, sequence: torch.Tensor, raw: torch.Tensor):
        last = sequence[..., -1]
        query = self.query(last).unsqueeze(-1)
        scores = (self.key(sequence) * query).sum(dim=1) / math.sqrt(sequence.shape[1])
        if self.use_recency:
            time = torch.linspace(-1.0, 0.0, sequence.shape[-1], device=sequence.device)
            scores = scores + torch.nn.functional.softplus(self.recency) * time
        f = self.feature_count
        if self.use_reliability:
            reliability = 1.0 - raw[:, 2 * f : 3 * f].mean(dim=1).clamp(0.0, 1.0)
            scores = scores + torch.log(reliability.clamp_min(0.05))
        attention = torch.softmax(scores, dim=-1)
        pooled = (self.value(sequence) * attention.unsqueeze(1)).sum(dim=-1)
        if self.use_trend:
            recent = sequence[..., -16:].mean(dim=-1)
            earlier = sequence[..., -32:-16].mean(dim=-1)
            trend = recent - earlier
        else:
            trend = torch.zeros_like(last)
        missing_rate = raw[:, 2 * f : 3 * f].mean(dim=-1)
        age = raw[:, 3 * f, -1:]
        gate = self.gate(torch.cat([last, pooled, trend, missing_rate, age], dim=-1))
        update = self.update(torch.cat([pooled - last, trend], dim=-1))
        return last + self.residual_scale * gate * update, attention


def build_model(name: str, channels: int, hidden: int, bins: int):
    del bins
    flags = {
        "caro_no_recency": (False, True, True),
        "caro_no_reliability": (True, False, True),
        "caro_no_trend": (True, True, False),
        "caro_ordered_only": (True, True, True),
    }
    if name not in flags:
        raise KeyError(name)
    pool = MechanismCDAP(
        hidden,
        use_recency=flags[name][0],
        use_reliability=flags[name][1],
        use_trend=flags[name][2],
    )
    decoder = (
        OrderedQuantileRiskHead(hidden)
        if name == "caro_ordered_only"
        else IntervalAwareOrderedRiskHead(hidden, risk_margin=0.10)
    )
    return CDAPRUL(SequenceTCN(channels, hidden), pool, decoder)
