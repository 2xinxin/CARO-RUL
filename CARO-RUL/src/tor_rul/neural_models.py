from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PointHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = torch.sigmoid(self.net(hidden).squeeze(-1))
        return {"prediction": prediction}


class MLPRegressor(nn.Module):
    def __init__(self, channels: int, length: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(), nn.Linear(channels * length, hidden * 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden * 2, hidden), nn.GELU(),
        )
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.body(x))


class CNNRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.body(x))


class RecurrentRegressor(nn.Module):
    def __init__(self, kind: str, channels: int, hidden: int, bidirectional: bool = False):
        super().__init__()
        recurrent = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = recurrent(channels, hidden, batch_first=True, bidirectional=bidirectional)
        self.head = PointHead(hidden * (2 if bidirectional else 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output, _ = self.rnn(x.transpose(1, 2))
        return self.head(output[:, -1])


class CausalConvBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        pad = 2 * dilation
        self.pad = pad
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation, padding=pad)
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)[..., : -self.pad] if self.pad else self.conv(x)
        return F.gelu(self.norm(y)) + x


class TCNRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(*(CausalConvBlock(hidden, d) for d in (1, 2, 4, 8)))
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = self.blocks(self.input(x))
        return self.head(y[..., -1])


class DLinearRegressor(nn.Module):
    def __init__(self, channels: int, length: int, hidden: int):
        super().__init__()
        self.kernel = 25
        self.trend = nn.Linear(length, hidden)
        self.residual = nn.Linear(length, hidden)
        self.mix = nn.Linear(channels * hidden * 2, hidden)
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        trend = F.avg_pool1d(x, self.kernel, stride=1, padding=self.kernel // 2)
        residual = x - trend
        hidden = torch.cat([self.trend(trend), self.residual(residual)], dim=1).flatten(1)
        return self.head(F.gelu(self.mix(hidden)))


class TransformerRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int, length: int, sparse: bool = False):
        super().__init__()
        self.embed = nn.Linear(channels, hidden)
        self.position = nn.Parameter(torch.zeros(1, length, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=0.1, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.sparse = sparse
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.embed(x.transpose(1, 2)) + self.position[:, : x.shape[-1]]
        if self.sparse:
            stride = max(1, int(math.log2(tokens.shape[1])))
            anchors = tokens[:, ::stride]
            tokens = torch.cat([anchors, tokens[:, -min(16, tokens.shape[1]) :]], dim=1)
        hidden = self.encoder(tokens).mean(dim=1)
        return self.head(hidden)


class NBeatsRegressor(MLPRegressor):
    pass


class PatchTSTRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int, length: int):
        super().__init__()
        self.patch = 16
        self.embed = nn.Conv1d(channels, hidden, self.patch, stride=8)
        n_tokens = 1 + max(1, (length - self.patch) // 8)
        self.position = nn.Parameter(torch.zeros(1, n_tokens, hidden))
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.embed(x).transpose(1, 2)
        tokens = self.encoder(tokens + self.position[:, : tokens.shape[1]])
        return self.head(tokens.mean(1))


class TimesBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(width, width * 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(width * 2, width, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1)
        weight = spectrum.abs().mean(dim=1, keepdim=True)
        filtered = torch.fft.irfft(spectrum * torch.sigmoid(weight), n=x.shape[-1], dim=-1)
        return x + self.conv(filtered)


class TimesNetRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(TimesBlock(hidden), TimesBlock(hidden))
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.blocks(self.input(x)).mean(-1))


class PathformerRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, hidden, k, stride=max(1, k // 2)), nn.GELU(), nn.AdaptiveAvgPool1d(1))
            for k in (4, 8, 16)
        ])
        self.router = nn.Sequential(nn.Linear(channels, 3), nn.Softmax(dim=-1))
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = x.mean(-1)
        weights = self.router(pooled)
        paths = torch.stack([branch(x).squeeze(-1) for branch in self.branches], dim=1)
        return self.head((paths * weights.unsqueeze(-1)).sum(1))


class SAMCDRegressor(TransformerRegressor):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        result = super().forward(x)
        result["prediction"] = F.dropout(result["prediction"], p=0.15, training=self.training)
        return result


class CMRDE(nn.Module):
    """Causal multi-resolution trend-residual encoder."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(CausalConvBlock(hidden, d), CausalConvBlock(hidden, d * 2))
            for d in (1, 4, 16)
        ])
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3 + 4, hidden), nn.GELU(), nn.Linear(hidden, 3), nn.Softmax(dim=-1)
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.input(x)
        branches = torch.stack([branch(base)[..., -1] for branch in self.branches], dim=1)
        state = torch.stack([
            x[:, 1, -1], x[:, 1, -16:].mean(-1), x[:, 2, -1], x[:, 3, -1]
        ], dim=-1)
        weights = self.gate(torch.cat([branches.flatten(1), state], dim=-1))
        fused = (branches * weights.unsqueeze(-1)).sum(1)
        residual = base[..., -16:].std(dim=-1, unbiased=False)
        return self.out(torch.cat([fused, residual], dim=-1)), weights


class SurvivalHazardDecoder(nn.Module):
    def __init__(self, hidden: int, bins: int):
        super().__init__()
        self.bins = bins
        self.net = nn.Sequential(
            nn.Linear(hidden + 4, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 2)
        )

    def forward(
        self, hidden: torch.Tensor, age: torch.Tensor, rate: torch.Tensor,
        drawdown: torch.Tensor, rebound: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        params = self.net(torch.cat([hidden, age, rate, drawdown, rebound], dim=-1))
        location = torch.sigmoid(params[:, :1])
        concentration = F.softplus(params[:, 1:2]) + 8.0
        grid = torch.linspace(0.0, 1.0, self.bins + 1, device=hidden.device)[None, :]
        survival_edges = torch.sigmoid((location - grid) * concentration)
        survival = survival_edges[:, 1:]
        mass = (survival_edges[:, :-1] - survival_edges[:, 1:]).clamp_min(1e-8)
        mass = mass / mass.sum(dim=-1, keepdim=True)
        centers = (grid[:, :-1] + grid[:, 1:]) * 0.5
        prediction = (mass * centers).sum(dim=-1)
        hazard = (mass / survival_edges[:, :-1].clamp_min(1e-6)).clamp(1e-5, 1.0 - 1e-5)
        return {
            "prediction": prediction, "hazard": hazard, "survival": survival,
            "location": location.squeeze(-1), "concentration": concentration.squeeze(-1),
        }


class DASHRUL(nn.Module):
    def __init__(self, channels: int, hidden: int, bins: int):
        super().__init__()
        self.encoder = CMRDE(channels, hidden)
        self.decoder = SurvivalHazardDecoder(hidden, bins)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, weights = self.encoder(x)
        age = x[:, 5, -1:].contiguous()
        rate = x[:, 1, -16:].mean(-1, keepdim=True)
        drawdown = x[:, 2, -1:].contiguous()
        rebound = x[:, 3, -1:].contiguous()
        result = self.decoder(hidden, age, rate, drawdown, rebound)
        result["branch_weights"] = weights
        return result


class PopulationAnchoredSurvivalDecoder(nn.Module):
    """Absolute-life decoder anchored only by the current training population."""

    def __init__(self, hidden: int, bins: int):
        super().__init__()
        self.bins = bins
        self.register_buffer("prior_log_life", torch.tensor(0.70, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(hidden + 5, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 2)
        )

    def set_prior_lifetime(self, lifetime: float) -> None:
        normalized = math.log1p(max(float(lifetime), 1.0)) / math.log1p(1_000_000.0)
        self.prior_log_life.fill_(normalized)

    def forward(
        self,
        hidden: torch.Tensor,
        age: torch.Tensor,
        rate: torch.Tensor,
        drawdown: torch.Tensor,
        rebound: torch.Tensor,
        scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        params = self.net(torch.cat([hidden, age, rate, drawdown, rebound, scale], dim=-1))
        total_log = (self.prior_log_life + 0.30 * torch.tanh(params[:, 0])).clamp(0.05, 0.999)
        total_lifetime = torch.expm1(total_log * math.log1p(1_000_000.0))
        observed_age = torch.expm1(age.squeeze(-1) * math.log1p(50000.0))
        remaining = F.softplus((total_lifetime - observed_age) / 100.0) * 100.0
        prediction = torch.log1p(remaining) / math.log1p(1_000_000.0)
        concentration = F.softplus(params[:, 1]) + 8.0
        grid = torch.linspace(0.0, 1.0, self.bins + 1, device=hidden.device)[None, :]
        edges = torch.sigmoid((prediction[:, None] - grid) * concentration[:, None])
        survival = edges[:, 1:]
        mass = (edges[:, :-1] - edges[:, 1:]).clamp_min(1e-8)
        mass = mass / mass.sum(dim=-1, keepdim=True)
        hazard = (mass / edges[:, :-1].clamp_min(1e-6)).clamp(1e-5, 1.0 - 1e-5)
        return {
            "prediction": prediction,
            "hazard": hazard,
            "survival": survival,
            "total_lifetime": total_lifetime,
            "concentration": concentration,
        }


class CAPSRUL(nn.Module):
    """CMRDE plus population-anchored survival decoder."""

    def __init__(self, channels: int, hidden: int, bins: int):
        super().__init__()
        self.encoder = CMRDE(channels, hidden)
        self.decoder = PopulationAnchoredSurvivalDecoder(hidden, bins)

    def set_prior_lifetime(self, lifetime: float) -> None:
        self.decoder.set_prior_lifetime(lifetime)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, weights = self.encoder(x)
        age = x[:, 5, -1:].contiguous()
        rate = x[:, 1, -16:].mean(-1, keepdim=True)
        drawdown = x[:, 2, -1:].contiguous()
        rebound = x[:, 3, -1:].contiguous()
        scale = x[:, 6, -1:].contiguous()
        result = self.decoder(hidden, age, rate, drawdown, rebound, scale)
        result["branch_weights"] = weights
        return result


class CMRDEDirect(nn.Module):
    """CMRDE with a generic point head: PASD ablation."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.encoder = CMRDE(channels, hidden)
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, weights = self.encoder(x)
        result = self.head(hidden)
        result["branch_weights"] = weights
        return result


class CAPSNoCMRDE(nn.Module):
    """Standard causal TCN with PASD: CMRDE ablation."""

    def __init__(self, channels: int, hidden: int, bins: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(*(CausalConvBlock(hidden, d) for d in (1, 2, 4, 8)))
        self.decoder = PopulationAnchoredSurvivalDecoder(hidden, bins)

    def set_prior_lifetime(self, lifetime: float) -> None:
        self.decoder.set_prior_lifetime(lifetime)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.input(x))[..., -1]
        return self.decoder(
            hidden,
            x[:, 5, -1:].contiguous(),
            x[:, 1, -16:].mean(-1, keepdim=True),
            x[:, 2, -1:].contiguous(),
            x[:, 3, -1:].contiguous(),
            x[:, 6, -1:].contiguous(),
        )


class ScaleAdaptiveCausalMixer(nn.Module):
    """TCN backbone plus zero-initialized multi-resolution degradation context."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.backbone = nn.Sequential(*(CausalConvBlock(hidden, d) for d in (1, 2, 4, 8)))
        self.context = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2),
                nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            for kernel_size in (3, 9, 25)
        ])
        self.router = nn.Sequential(
            nn.Linear(hidden + 4, hidden), nn.GELU(), nn.Linear(hidden, 3), nn.Softmax(dim=-1)
        )
        self.context_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(self.input(x))
        base = features[..., -1]
        contexts = torch.stack([branch(features) for branch in self.context], dim=1)
        state = torch.stack([
            x[:, 1, -1], x[:, 1, -16:].mean(-1), x[:, 2, -1], x[:, 3, -1]
        ], dim=-1)
        weights = self.router(torch.cat([base, state], dim=-1))
        mixed = (contexts * weights.unsqueeze(-1)).sum(1)
        # Preserve the proven TCN representation and permit only a bounded
        # multi-resolution correction. The zero initialization is an exact
        # TCN-equivalent starting point.
        hidden = base + 0.10 * torch.tanh(self.context_scale) * mixed
        return hidden, weights


class MonotonicLifeCalibrationDecoder(nn.Module):
    """Fuse neural RUL with a source-only monotonic scale-to-life calibration."""

    def __init__(self, hidden: int, base_low: float = 0.10, base_high: float = 0.50):
        super().__init__()
        self.base_low = float(base_low)
        self.base_high = float(base_high)
        self.direct = PointHead(hidden)
        self.gate = nn.Sequential(nn.Linear(hidden + 3, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(3.0))
        self.register_buffer("calibration_slope", torch.tensor(1.0))
        self.register_buffer("calibration_intercept", torch.tensor(0.0))

    def set_population(self, scales: list[float], lifetimes: list[float]) -> None:
        x = torch.log(torch.as_tensor(scales, dtype=torch.float64).clamp_min(1e-8)).numpy()
        y = torch.log(torch.as_tensor(lifetimes, dtype=torch.float64).clamp_min(1.0)).numpy()
        slope, intercept = __import__("numpy").polyfit(x, y, 1)
        self.calibration_slope.fill_(max(float(slope), 0.0))
        self.calibration_intercept.fill_(float(intercept))

    def forward(self, hidden: torch.Tensor, age: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
        direct_log = self.direct(hidden)["prediction"]
        scale_value = torch.expm1(scale.squeeze(-1) * math.log1p(10000.0)).clamp_min(1e-8)
        total_lifetime = torch.exp(
            self.calibration_intercept + self.calibration_slope * torch.log(scale_value)
        ).clamp(1.0, 1_000_000.0)
        observed_age = torch.expm1(age.squeeze(-1) * math.log1p(50000.0))
        clock_rul = F.softplus((total_lifetime - observed_age) / 100.0) * 100.0
        direct_rul = torch.expm1(direct_log * math.log1p(1_000_000.0))
        calibration_log = torch.log1p(clock_rul) / math.log1p(1_000_000.0)
        gate_features = torch.cat([
            hidden, age, scale,
            (calibration_log - direct_log).unsqueeze(-1),
        ], dim=-1)
        # A bounded residual calibration prevents the three-device fit from
        # replacing the neural estimator when it must extrapolate.
        calibration_weight = self.base_low + (self.base_high - self.base_low) * torch.sigmoid(
            self.gate(gate_features).squeeze(-1)
        )
        prediction_rul = (
            (1.0 - calibration_weight) * direct_rul + calibration_weight * clock_rul
        )
        prediction = torch.log1p(prediction_rul) / math.log1p(1_000_000.0)
        return {
            "prediction": prediction,
            "direct_prediction": direct_log,
            "calibrated_total_lifetime": total_lifetime,
            "calibration_weight": calibration_weight,
        }


class SACMDirect(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.encoder = ScaleAdaptiveCausalMixer(channels, hidden)
        self.head = PointHead(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, weights = self.encoder(x)
        result = self.head(hidden)
        result["scale_weights"] = weights
        return result


class MOSAICRUL(nn.Module):
    """SACM encoder plus monotonic life-calibration decoder."""

    def __init__(self, channels: int, hidden: int, use_sacm: bool = True):
        super().__init__()
        self.use_sacm = use_sacm
        if use_sacm:
            self.encoder = ScaleAdaptiveCausalMixer(channels, hidden)
        else:
            self.input = nn.Conv1d(channels, hidden, 1)
            self.blocks = nn.Sequential(*(CausalConvBlock(hidden, d) for d in (1, 2, 4, 8)))
        self.decoder = MonotonicLifeCalibrationDecoder(hidden)

    def set_population(self, scales: list[float], lifetimes: list[float]) -> None:
        self.decoder.set_population(scales, lifetimes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.use_sacm:
            hidden, weights = self.encoder(x)
        else:
            hidden = self.blocks(self.input(x))[..., -1]
            weights = None
        result = self.decoder(hidden, x[:, 5, -1:].contiguous(), x[:, 6, -1:].contiguous())
        if weights is not None:
            result["scale_weights"] = weights
        return result


class StateConditionedModulation(nn.Module):
    """Bounded identity-initialized FiLM for cross-device state adaptation."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden * 2)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        state = torch.stack([
            x[:, 5, -1], x[:, 6, -1],
            x[:, 1, -16:].mean(-1), x[:, 2, -1],
        ], dim=-1)
        gamma, beta = self.net(state).chunk(2, dim=-1)
        return hidden * (1.0 + 0.10 * torch.tanh(gamma)) + 0.10 * torch.tanh(beta)


class SCTCNDirect(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(*(CausalConvBlock(hidden, d) for d in (1, 2, 4, 8)))
        # Construct the point head before the new identity module so that this
        # ablation and TCN have matching initial backbone/head weights per seed.
        self.head = PointHead(hidden)
        self.modulation = StateConditionedModulation(hidden)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.input(x))[..., -1]
        return self.head(self.modulation(hidden, x))


class SCMOLCRUL(nn.Module):
    """State-conditioned TCN plus monotonic calibration and bounded fusion."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(*(CausalConvBlock(hidden, d) for d in (1, 2, 4, 8)))
        self.decoder = MonotonicLifeCalibrationDecoder(hidden)
        self.modulation = StateConditionedModulation(hidden)

    def set_population(self, scales: list[float], lifetimes: list[float]) -> None:
        self.decoder.set_population(scales, lifetimes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.input(x))[..., -1]
        hidden = self.modulation(hidden, x)
        return self.decoder(hidden, x[:, 5, -1:].contiguous(), x[:, 6, -1:].contiguous())


class ExtrapolationAwareCalibrationDecoder(MonotonicLifeCalibrationDecoder):
    """Increase calibration continuously outside the source scale envelope."""

    def __init__(
        self,
        hidden: int,
        base_low: float = 0.10,
        base_high: float = 0.50,
        outside_weight: float = 0.95,
        envelope_decay: float = 8.0,
    ):
        super().__init__(hidden, base_low=base_low, base_high=base_high)
        self.outside_weight = float(outside_weight)
        self.envelope_decay = float(envelope_decay)
        self.register_buffer("min_log_scale", torch.tensor(0.0))
        self.register_buffer("max_log_scale", torch.tensor(10.0))

    def set_population(self, scales: list[float], lifetimes: list[float]) -> None:
        super().set_population(scales, lifetimes)
        log_scales = torch.log(torch.as_tensor(scales, dtype=torch.float32).clamp_min(1e-8))
        self.min_log_scale.copy_(log_scales.min())
        self.max_log_scale.copy_(log_scales.max())

    def forward(self, hidden: torch.Tensor, age: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
        result = super().forward(hidden, age, scale)
        direct_log = result["direct_prediction"]
        direct_rul = torch.expm1(direct_log * math.log1p(1_000_000.0))
        observed_age = torch.expm1(age.squeeze(-1) * math.log1p(50000.0))
        clock_rul = F.softplus(
            (result["calibrated_total_lifetime"] - observed_age) / 100.0
        ) * 100.0
        log_scale = torch.log(
            torch.expm1(scale.squeeze(-1) * math.log1p(10000.0)).clamp_min(1e-8)
        )
        outside_distance = (
            F.relu(self.min_log_scale - log_scale) + F.relu(log_scale - self.max_log_scale)
        )
        outside_strength = 1.0 - torch.exp(-self.envelope_decay * outside_distance)
        base_weight = result["calibration_weight"]
        weight = base_weight + (self.outside_weight - base_weight) * outside_strength
        prediction_rul = (1.0 - weight) * direct_rul + weight * clock_rul
        result["prediction"] = torch.log1p(prediction_rul) / math.log1p(1_000_000.0)
        result["calibration_weight"] = weight
        result["outside_strength"] = outside_strength
        return result


class EACALMRUL(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        use_modulation: bool,
        *,
        base_low: float = 0.10,
        base_high: float = 0.50,
        outside_weight: float = 0.95,
        envelope_decay: float = 8.0,
    ):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(*(CausalConvBlock(hidden, d) for d in (1, 2, 4, 8)))
        self.decoder = ExtrapolationAwareCalibrationDecoder(
            hidden,
            base_low=base_low,
            base_high=base_high,
            outside_weight=outside_weight,
            envelope_decay=envelope_decay,
        )
        self.modulation = StateConditionedModulation(hidden) if use_modulation else None

    def set_population(self, scales: list[float], lifetimes: list[float]) -> None:
        self.decoder.set_population(scales, lifetimes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.input(x))[..., -1]
        if self.modulation is not None:
            hidden = self.modulation(hidden, x)
        return self.decoder(hidden, x[:, 5, -1:].contiguous(), x[:, 6, -1:].contiguous())


def build_model(name: str, channels: int, length: int, hidden: int, bins: int) -> nn.Module:
    if name in {"cdap_point", "tcn_iorh", "caro_rul"}:
        # Import lazily so the publication-facing CARO modules and every
        # comparator can be trained by the same benchmark runner.
        from pcss_single_model.model_v9 import build_model as build_caro_model

        return build_caro_model(name, channels, hidden, bins)
    if name in {
        "caro_no_recency", "caro_no_reliability", "caro_no_trend",
        "caro_ordered_only",
    }:
        from pcss_single_model.model_v10 import build_model as build_mechanism_model

        return build_mechanism_model(name, channels, hidden, bins)
    if name in {
        "dlinear_faithful", "patchtst_faithful", "timesnet_faithful",
        "informer_faithful", "nbeats_faithful", "pathformer_adapted",
        "sa_mcd_adapted",
    }:
        from .faithful_models import build_faithful_model

        return build_faithful_model(name, channels, length, hidden)
    if name == "mlp":
        return MLPRegressor(channels, length, hidden)
    if name == "cnn":
        return CNNRegressor(channels, hidden)
    if name in {"rnn", "gru", "lstm"}:
        return RecurrentRegressor(name, channels, hidden)
    if name == "bilstm":
        return RecurrentRegressor("lstm", channels, hidden, bidirectional=True)
    if name == "tcn":
        return TCNRegressor(channels, hidden)
    if name == "dlinear":
        return DLinearRegressor(channels, length, hidden)
    if name == "transformer":
        return TransformerRegressor(channels, hidden, length)
    if name == "informer":
        return TransformerRegressor(channels, hidden, length, sparse=True)
    if name == "nbeats":
        return NBeatsRegressor(channels, length, hidden)
    if name == "patchtst":
        return PatchTSTRegressor(channels, hidden, length)
    if name == "timesnet":
        return TimesNetRegressor(channels, hidden)
    if name == "pathformer":
        return PathformerRegressor(channels, hidden)
    if name == "sa_mcd":
        return SAMCDRegressor(channels, hidden, length)
    if name in {"dash_rul", "dash_rul_raw", "se_dash_direct"}:
        return DASHRUL(channels, hidden, bins)
    if name == "caps_rul":
        return CAPSRUL(channels, hidden, bins)
    if name == "cmrde_direct":
        return CMRDEDirect(channels, hidden)
    if name == "caps_no_cmrde":
        return CAPSNoCMRDE(channels, hidden, bins)
    if name == "sacm_direct":
        return SACMDirect(channels, hidden)
    if name == "molc_tcn":
        return MOSAICRUL(channels, hidden, use_sacm=False)
    if name == "mosaic_rul":
        return MOSAICRUL(channels, hidden, use_sacm=True)
    if name == "sctcn_direct":
        return SCTCNDirect(channels, hidden)
    if name == "scmolc_rul":
        return SCMOLCRUL(channels, hidden)
    if name == "eacal_tcn":
        return EACALMRUL(channels, hidden, use_modulation=False)
    if name == "eacal_bounds_narrow":
        return EACALMRUL(
            channels, hidden, use_modulation=False,
            base_low=0.05, base_high=0.35, outside_weight=0.90,
        )
    if name == "eacal_bounds_wide":
        return EACALMRUL(
            channels, hidden, use_modulation=False,
            base_low=0.15, base_high=0.65, outside_weight=0.98,
        )
    if name == "eacal_decay_slow":
        return EACALMRUL(channels, hidden, use_modulation=False, envelope_decay=4.0)
    if name == "eacal_decay_fast":
        return EACALMRUL(channels, hidden, use_modulation=False, envelope_decay=16.0)
    if name == "calm_rul_v2":
        return EACALMRUL(channels, hidden, use_modulation=True)
    raise KeyError(f"unknown model: {name}")
