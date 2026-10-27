from __future__ import annotations

import torch

from pcss_single_model.model_v4 import OrderedQuantileRiskHead
from pcss_single_model.model_v7 import CDAPRUL, CausalDegradationAttentionPooling, SequenceTCN
from pcss_single_model.model_v2 import PointDecoder


class IntervalAwareOrderedRiskHead(OrderedQuantileRiskHead):
    """Ordered quantiles with a single-pass uncertainty risk margin."""

    def __init__(self, hidden: int, risk_margin: float = 0.10):
        super().__init__(hidden)
        self.risk_margin = risk_margin

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(hidden)
        q25, q375, _q50, q75 = output["quantiles"].unbind(dim=-1)
        spread = q75 - q25
        output["prediction"] = torch.clamp(q375 - self.risk_margin * spread, 0.0, 1.0)
        output["predictive_spread"] = spread
        return output


def build_model(name: str, channels: int, hidden: int, bins: int):
    del bins
    variants = {
        "tcn_point": (False, False),
        "cdap_point": (True, False),
        "tcn_iorh": (False, True),
        "caro_rul": (True, True),
    }
    if name not in variants:
        raise KeyError(name)
    use_pool, use_risk = variants[name]
    backbone = SequenceTCN(channels, hidden)
    point = PointDecoder(hidden)
    candidate_pool = CausalDegradationAttentionPooling(hidden, residual_scale=0.10)
    risk_head = IntervalAwareOrderedRiskHead(hidden, risk_margin=0.10)
    return CDAPRUL(
        backbone,
        candidate_pool if use_pool else None,
        risk_head if use_risk else point,
    )
