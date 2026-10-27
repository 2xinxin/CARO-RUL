from __future__ import annotations

import torch
from torch import nn

from pcss_single_model.model_v2 import PointDecoder
from pcss_single_model.model_v3 import CRMFEncoder


class OrderedQuantileRiskHead(nn.Module):
    """Module II: one-pass ordered conditional RUL quantiles.

    The head parameterizes four non-crossing quantiles.  The 0.375 quantile is
    the operational prediction, reflecting the asymmetric cost of dangerous
    RUL overestimation without averaging checkpoints or architectures.
    """

    def __init__(self, hidden: int, quantile_weight: float = 0.35):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.center = nn.Linear(hidden, 1)
        self.gaps = nn.Linear(hidden, 3)
        nn.init.constant_(self.gaps.bias, -2.0)
        self.quantile_weight = quantile_weight
        self.register_buffer(
            "levels", torch.tensor([0.25, 0.375, 0.50, 0.75], dtype=torch.float32)
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.norm(hidden)
        q50 = torch.sigmoid(self.center(h)).squeeze(-1)
        gap = torch.sigmoid(self.gaps(h))
        q375 = q50 * (1.0 - 0.5 * gap[:, 0])
        q25 = q375 * (1.0 - 0.5 * gap[:, 1])
        q75 = q50 + (1.0 - q50) * 0.5 * gap[:, 2]
        quantiles = torch.stack([q25, q375, q50, q75], dim=-1)
        return {
            "prediction": q375,
            "median_prediction": q50,
            "quantiles": quantiles,
            "quantile_levels": self.levels,
            "quantile_weight": self.quantile_weight,
        }


class RQMRUL(nn.Module):
    def __init__(
        self, channels: int, hidden: int, use_rgmf: bool, use_quantile_head: bool
    ):
        super().__init__()
        # CRN is deliberately disabled: P004 showed no meaningful contribution.
        self.encoder = CRMFEncoder(channels, hidden, use_crn=False, use_rgmf=use_rgmf)
        self.decoder = (
            OrderedQuantileRiskHead(hidden) if use_quantile_head else PointDecoder(hidden)
        )

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
        "rgmf_point": (True, False),
        "tcn_oqrh": (False, True),
        "rqm_rul": (True, True),
    }
    if name not in variants:
        raise KeyError(name)
    return RQMRUL(channels, hidden, *variants[name])
