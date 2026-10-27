from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StandardCausalBlock(nn.Module):
    """The same full-convolution residual block used by the frozen TCN baseline."""

    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation, padding=self.pad)
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)[..., : -self.pad]
        return x + F.gelu(self.norm(y))


class StandardTCNEncoder(nn.Module):
    """Matched single-network backbone; no lightweight surrogate is used."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(
            *(StandardCausalBlock(hidden, dilation) for dilation in (1, 2, 4, 8))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(x))[..., -1]


class CausalMorphologyBranch(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.left = 2 * dilation
        self.conv = nn.Conv1d(
            width, width, 3, dilation=dilation, groups=width, bias=False
        )
        self.mix = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mix(self.conv(F.pad(x, (self.left, 0))))
        return F.gelu(self.norm(y))


class ReliabilityConditionedMorphologyAdapter(nn.Module):
    """Module I (QMME): a bounded residual adapter inside one TCN.

    The adapter extracts short-, medium-, and long-range pulse morphology from
    value and first-difference streams.  Missingness rates and normalized age
    select the useful temporal scale.  A zero-initialized residual coefficient
    makes the initial network exactly equivalent to the standard TCN.
    """

    def __init__(self, feature_count: int, hidden: int):
        super().__init__()
        half = hidden // 2
        self.feature_count = feature_count
        self.value_projection = nn.Conv1d(feature_count, half, 1)
        self.delta_projection = nn.Conv1d(feature_count, hidden - half, 1)
        self.branches = nn.ModuleList(
            CausalMorphologyBranch(hidden, dilation) for dilation in (1, 4, 16)
        )
        self.selector = nn.Sequential(
            nn.Linear(hidden + feature_count + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
            nn.Softmax(dim=-1),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Tanh()
        )
        self.strength = nn.Parameter(torch.zeros(()))

    def forward(
        self, x: torch.Tensor, backbone_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.feature_count
        values = x[:, :f]
        differences = x[:, f : 2 * f]
        missingness = x[:, 2 * f : 3 * f].mean(dim=-1)
        age = x[:, 3 * f, -1:]
        projected = torch.cat(
            [self.value_projection(values), self.delta_projection(differences)], dim=1
        )
        branch_sequences = [branch(projected) for branch in self.branches]
        states = torch.stack(
            [
                0.5 * sequence[..., -1]
                + 0.5 * sequence[..., -16:].mean(dim=-1)
                for sequence in branch_sequences
            ],
            dim=1,
        )
        weights = self.selector(
            torch.cat([backbone_state, missingness, age], dim=-1)
        )
        morphology = (states * weights.unsqueeze(-1)).sum(dim=1)
        update = self.residual(torch.cat([backbone_state, morphology], dim=-1))
        # At initialization tanh(strength)=0, so this is exactly the TCN state.
        adapted = backbone_state + 0.35 * torch.tanh(self.strength) * update
        return adapted, weights


class PointDecoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.point = nn.Linear(hidden, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"prediction": torch.sigmoid(self.point(self.norm(hidden))).squeeze(-1)}


class ResidualMonotoneOrdinalDecoder(nn.Module):
    """Module II (MOD): direct RUL plus a bounded monotone ordinal correction.

    The survival logits are guaranteed to decrease with the ordered RUL
    thresholds.  Their implied expectation can correct the direct prediction,
    but the zero-initialized coupling prevents the ordinal branch from
    overpowering the calibrated point estimate.
    """

    def __init__(self, hidden: int, bins: int):
        super().__init__()
        self.bins = bins
        self.norm = nn.LayerNorm(hidden)
        self.point = nn.Linear(hidden, 1)
        self.anchor = nn.Linear(hidden, 1)
        self.steps = nn.Linear(hidden, bins - 1)
        self.span = nn.Linear(hidden, 1)
        self.coupling = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.norm(hidden)
        direct = torch.sigmoid(self.point(h)).squeeze(-1)
        proportions = torch.softmax(self.steps(h), dim=-1)
        span = 4.0 + 8.0 * torch.sigmoid(self.span(h))
        drops = proportions * span
        cumulative = torch.cat(
            [torch.zeros_like(span), torch.cumsum(drops, dim=-1)], dim=-1
        )
        logits = self.anchor(h) - cumulative
        survival = torch.sigmoid(logits)
        ordinal = survival.mean(dim=-1)
        correction = 0.35 * torch.tanh(self.coupling) * (ordinal - direct)
        prediction = torch.clamp(direct + correction, 0.0, 1.0)
        return {
            "prediction": prediction,
            "point_prediction": direct,
            "ordinal_prediction": ordinal,
            "survival_logits": logits,
            "survival": survival,
        }


class QOMRULv2(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        bins: int,
        use_qmme: bool,
        use_mod: bool,
    ):
        super().__init__()
        self.encoder = StandardTCNEncoder(channels, hidden)
        self.adapter = (
            ReliabilityConditionedMorphologyAdapter(14, hidden) if use_qmme else None
        )
        self.decoder = (
            ResidualMonotoneOrdinalDecoder(hidden, bins)
            if use_mod
            else PointDecoder(hidden)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        weights = None
        if self.adapter is not None:
            hidden, weights = self.adapter(x, hidden)
        output = self.decoder(hidden)
        if weights is not None:
            output["scale_weights"] = weights
        return output


def build_model(name: str, channels: int, hidden: int, bins: int) -> nn.Module:
    variants = {
        "tcn_point": (False, False),
        "qmme_point": (True, False),
        "tcn_mod": (False, True),
        "qom_rul": (True, True),
    }
    if name not in variants:
        raise KeyError(name)
    qmme, mod = variants[name]
    return QOMRULv2(channels, hidden, bins, qmme, mod)
