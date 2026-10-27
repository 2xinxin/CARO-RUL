from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pcss_single_model.model_v2 import PointDecoder, StandardCausalBlock


class CausalReliabilityNormalization(nn.Module):
    """Module I: observation-aware within-window normalization.

    Statistics use only the already observed causal window. Missing entries do
    not contribute to the moments, and a learnable bounded residual preserves
    absolute amplitude whenever normalization is not useful.
    """

    def __init__(self, dynamic_channels: int = 28):
        super().__init__()
        self.dynamic_channels = dynamic_channels
        self.mix = nn.Parameter(torch.zeros(dynamic_channels))
        self.affine_weight = nn.Parameter(torch.ones(dynamic_channels))
        self.affine_bias = nn.Parameter(torch.zeros(dynamic_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dynamic_channels
        dynamic = x[:, :d]
        # The 14 masks correspond to values; repeat them for first differences.
        missing = x[:, d : d + 14]
        observed = 1.0 - torch.cat([missing, missing], dim=1).clamp(0.0, 1.0)
        count = observed.sum(dim=-1, keepdim=True).clamp_min(2.0)
        mean = (dynamic * observed).sum(dim=-1, keepdim=True) / count
        variance = ((dynamic - mean).square() * observed).sum(dim=-1, keepdim=True) / count
        normalized = (dynamic - mean) / torch.sqrt(variance + 1e-4)
        normalized = (
            normalized * self.affine_weight[None, :, None]
            + self.affine_bias[None, :, None]
        )
        strength = 0.5 * torch.tanh(self.mix)[None, :, None]
        fused = dynamic + strength * (normalized - dynamic)
        return torch.cat([fused, x[:, d:]], dim=1)


class HiddenMorphologyBranch(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.left = 2 * dilation
        self.depthwise = nn.Conv1d(
            width, width, 3, dilation=dilation, groups=width, bias=False
        )
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(F.pad(x, (self.left, 0)))
        return F.gelu(self.norm(self.pointwise(y)))


class ReliabilityGatedMorphologyFusion(nn.Module):
    """Module II: fuse degradation morphology at three causal time scales."""

    def __init__(self, width: int, feature_count: int = 14):
        super().__init__()
        self.feature_count = feature_count
        self.branches = nn.ModuleList(
            HiddenMorphologyBranch(width, dilation) for dilation in (1, 4, 16)
        )
        self.gate = nn.Sequential(
            nn.Conv1d(width + feature_count + 1, width, 1),
            nn.GELU(),
            nn.Conv1d(width, 3, 1),
        )
        self.output = nn.Conv1d(width, width, 1)
        self.strength = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.feature_count
        missing = raw[:, 2 * f : 3 * f]
        age = raw[:, 3 * f : 3 * f + 1]
        reliability = 1.0 - missing.clamp(0.0, 1.0)
        weights = torch.softmax(
            self.gate(torch.cat([hidden, reliability, age], dim=1)), dim=1
        )
        branch_outputs = torch.stack([branch(hidden) for branch in self.branches], dim=1)
        fused = (branch_outputs * weights.unsqueeze(2)).sum(dim=1)
        update = torch.tanh(self.output(fused))
        hidden = hidden + 0.35 * torch.tanh(self.strength) * update
        return hidden, weights


class CRMFEncoder(nn.Module):
    def __init__(self, channels: int, hidden: int, use_crn: bool, use_rgmf: bool):
        super().__init__()
        self.normalizer = CausalReliabilityNormalization() if use_crn else None
        self.input = nn.Conv1d(channels, hidden, 1)
        self.early = nn.Sequential(
            StandardCausalBlock(hidden, 1), StandardCausalBlock(hidden, 2)
        )
        self.fusion = ReliabilityGatedMorphologyFusion(hidden) if use_rgmf else None
        self.late = nn.Sequential(
            StandardCausalBlock(hidden, 4), StandardCausalBlock(hidden, 8)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        original = x
        if self.normalizer is not None:
            x = self.normalizer(x)
        hidden = self.early(self.input(x))
        weights = None
        if self.fusion is not None:
            hidden, weights = self.fusion(hidden, original)
        hidden = self.late(hidden)
        return hidden[..., -1], weights


class CRMRUL(nn.Module):
    def __init__(self, channels: int, hidden: int, use_crn: bool, use_rgmf: bool):
        super().__init__()
        self.encoder = CRMFEncoder(channels, hidden, use_crn, use_rgmf)
        self.decoder = PointDecoder(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, weights = self.encoder(x)
        output = self.decoder(hidden)
        if weights is not None:
            output["scale_weights"] = weights
        return output


def build_model(name: str, channels: int, hidden: int, bins: int) -> nn.Module:
    del bins
    variants = {
        "tcn_point": (False, False),
        "crn_tcn": (True, False),
        "rgmf_tcn": (False, True),
        "crm_rul": (True, True),
    }
    if name not in variants:
        raise KeyError(name)
    return CRMRUL(channels, hidden, *variants[name])
