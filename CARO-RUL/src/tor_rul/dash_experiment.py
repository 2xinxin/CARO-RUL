from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .neural_models import build_model
from .rul_dataset import RULSample, SequenceRULDataset, stack_tabular


@dataclass
class FitResult:
    name: str
    model: object
    best_validation: float
    epochs: int
    parameters: int
    domain_mean: np.ndarray | None = None
    domain_std: np.ndarray | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hazard_loss(output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    if "quantiles" in output:
        levels = output["quantile_levels"]
        error = target[:, None] - output["quantiles"]
        pinball = torch.maximum(levels * error, (levels - 1.0) * error).mean()
        point = F.smooth_l1_loss(output["median_prediction"], target)
        point = point + output["quantile_weight"] * pinball
    else:
        point = F.smooth_l1_loss(output["prediction"], target)
    if "aux_loss" in output:
        point = point + output["aux_loss"]
    if "hazard" not in output:
        return point
    bins = output["hazard"].shape[-1]
    event_bin = torch.clamp((target * (bins - 1)).round().long(), 0, bins - 1)
    hazard = output["hazard"]
    before = torch.arange(bins, device=target.device)[None, :] < event_bin[:, None]
    at = torch.arange(bins, device=target.device)[None, :] == event_bin[:, None]
    nll = -(
        torch.where(before, torch.log1p(-hazard), torch.zeros_like(hazard)).sum(-1)
        + torch.where(at, torch.log(hazard), torch.zeros_like(hazard)).sum(-1)
    ).mean()
    return point + 0.005 * nll


def fit_neural(
    name: str,
    train: list[RULSample],
    validation: list[RULSample],
    *,
    window: int,
    hidden: int,
    bins: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    seed: int,
    device: torch.device,
    target_mode: str | None = None,
) -> FitResult:
    seed_everything(seed)
    model = build_model(name, train[0].sequence.shape[0], window, hidden, bins).to(device)
    if hasattr(model, "set_prior_lifetime"):
        lifetimes = {sample.trace_id: sample.lifetime for sample in train}
        model.set_prior_lifetime(float(np.median(list(lifetimes.values()))))
    if hasattr(model, "set_population"):
        population = {}
        for sample in train:
            scale = float(np.expm1(sample.sequence[6, -1] * np.log1p(10000.0)))
            population[sample.trace_id] = (scale, sample.lifetime)
        model.set_population(
            [item[0] for item in population.values()],
            [item[1] for item in population.values()],
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    target_mode = target_mode or (
        "log_pulses"
        if name in {"se_dash_direct", "caps_rul", "cmrde_direct", "caps_no_cmrde",
                    "sacm_direct", "molc_tcn", "mosaic_rul"}
                    | {"sctcn_direct", "scmolc_rul"}
                    | {"eacal_tcn", "calm_rul_v2", "eacal_bounds_narrow",
                       "eacal_bounds_wide", "eacal_decay_slow", "eacal_decay_fast"}
        else "fraction"
    )
    loader_generator = torch.Generator().manual_seed(seed + 100003)
    train_loader = DataLoader(
        SequenceRULDataset(train, target_mode), batch_size=batch_size,
        shuffle=True, generator=loader_generator,
    )
    val_loader = DataLoader(SequenceRULDataset(validation, target_mode), batch_size=batch_size, shuffle=False)
    best = float("inf")
    best_state = None
    stale = 0
    completed = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for sequence, target in train_loader:
            sequence, target = sequence.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _hazard_loss(model(sequence), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        model.eval()
        validation_error_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for sequence, target in val_loader:
                sequence, target = sequence.to(device), target.to(device)
                if hasattr(model, "mc_validation_samples"):
                    for module in model.modules():
                        if isinstance(module, nn.Dropout):
                            module.train()
                    draws = [model(sequence)["prediction"] for _ in range(model.mc_validation_samples)]
                    prediction = torch.stack(draws).mean(0)
                    model.eval()
                else:
                    prediction = model(sequence)["prediction"]
                validation_error_sum += float(F.l1_loss(prediction, target, reduction="sum"))
                validation_count += int(target.numel())
        score = validation_error_sum / max(validation_count, 1)
        completed = epoch
        if score < best - 1e-5:
            best = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} produced no validation checkpoint")
    model.load_state_dict(best_state)
    domain_x, _ = stack_tabular(train)
    return FitResult(
        name, model, best, completed, sum(p.numel() for p in model.parameters()),
        domain_x.mean(axis=0), domain_x.std(axis=0) + 1e-6,
    )


def predict_neural(fit: FitResult, samples: list[RULSample], device: torch.device) -> tuple[np.ndarray, float]:
    loader = DataLoader(SequenceRULDataset(samples), batch_size=128, shuffle=False)
    values = []
    model = fit.model
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for sequence, _ in loader:
            sequence = sequence.to(device)
            if hasattr(model, "mc_samples"):
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                draws = [model(sequence)["prediction"] for _ in range(model.mc_samples)]
                prediction = torch.stack(draws).mean(0)
                model.eval()
            else:
                prediction = model(sequence)["prediction"]
            values.append(prediction.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / len(samples)
    prediction = np.concatenate(values)
    return prediction, elapsed_ms


def support_reliability(fit: FitResult, samples: list[RULSample]) -> np.ndarray:
    """Return a label-free support score for each target prefix."""
    if fit.domain_mean is None or fit.domain_std is None:
        return np.ones(len(samples), dtype=np.float64)
    target_x, _ = stack_tabular(samples)
    distance = np.mean(np.abs((target_x - fit.domain_mean) / fit.domain_std), axis=1)
    return np.exp(-3.0 * np.maximum(0.0, distance - 1.0))


def initial_state_scale(values: np.ndarray, width: int = 5, quantile: float = 0.5) -> float:
    """Robust causal scale using only the first few acquired measurements."""
    width = min(width, values.size)
    return max(float(np.quantile(np.abs(values[:width]), quantile)), np.finfo(np.float64).eps)


def estimated_lifetime_from_initial_scale(
    source_values: np.ndarray,
    target_values: np.ndarray,
    source_lifetime: int,
    *,
    width: int = 5,
    quantile: float = 0.5,
) -> float:
    """Scale-equivariant total-life estimate; no target EOL label is accessed."""
    ratio = initial_state_scale(target_values, width, quantile) / initial_state_scale(
        source_values, width, quantile
    )
    return float(source_lifetime) * ratio


def predict_clock_prior(samples: list[RULSample], estimated_lifetime: float) -> np.ndarray:
    """Convert an estimated total lifetime to RUL in acquisition-count units."""
    return np.asarray(
        [max(0.0, estimated_lifetime - float(sample.index)) for sample in samples],
        dtype=np.float64,
    )


def predict_scale_equivariant_dash(
    fit: FitResult,
    samples: list[RULSample],
    device: torch.device,
    estimated_lifetime: float,
) -> tuple[np.ndarray, float]:
    """Fuse the scale clock with a neural phase residual only on supported prefixes."""
    phase, latency = predict_neural(fit, samples, device)
    clock = predict_clock_prior(samples, estimated_lifetime)
    phase_rul = np.clip(phase, 0.0, 1.0) * estimated_lifetime
    reliability = support_reliability(fit, samples)
    prediction = clock + reliability * (phase_rul - clock)
    return np.maximum(prediction, 0.0), latency


def predict_direct_log_rul(
    fit: FitResult, samples: list[RULSample], device: torch.device,
) -> tuple[np.ndarray, float]:
    normalized_log, latency = predict_neural(fit, samples, device)
    pulses = np.expm1(np.clip(normalized_log, 0.0, 1.0) * np.log1p(1_000_000.0))
    return pulses, latency


def build_classical(name: str, seed: int):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.02, gamma="scale"))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.04, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
    raise KeyError(name)


def fit_classical(name: str, train: list[RULSample], seed: int) -> FitResult:
    x, y = stack_tabular(train)
    model = build_classical(name, seed)
    model.fit(x, y)
    return FitResult(name, model, float("nan"), 1, 0)


def fit_classical_log_rul(name: str, train: list[RULSample], seed: int) -> FitResult:
    x, _ = stack_tabular(train)
    target = np.asarray(
        [np.log1p(sample.target_pulses) / np.log1p(1_000_000.0) for sample in train],
        dtype=np.float64,
    )
    model = build_classical(name, seed)
    model.fit(x, target)
    return FitResult(name, model, float("nan"), 1, 0)


def predict_classical_log_rul(model: object, samples: list[RULSample]) -> tuple[np.ndarray, float]:
    normalized, elapsed_ms = predict_classical(model, samples)
    return np.expm1(normalized * np.log1p(1_000_000.0)), elapsed_ms


def predict_classical(model: object, samples: list[RULSample]) -> tuple[np.ndarray, float]:
    x, _ = stack_tabular(samples)
    start = time.perf_counter()
    result = np.clip(np.asarray(model.predict(x), dtype=np.float64), 0.0, 1.0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / len(samples)
    return result, elapsed_ms


def predict_wiener_like(train: list[RULSample], test: list[RULSample]) -> tuple[np.ndarray, float]:
    terminal_ratio = float(np.median([sample.sequence[0, -1] for sample in train[-max(8, len(train) // 10) :]]))
    predictions = []
    start = time.perf_counter()
    for sample in test:
        values = sample.sequence[0]
        width = min(32, values.size)
        x = np.arange(width, dtype=np.float64)
        slope = float(np.polyfit(x, values[-width:], 1)[0])
        if abs(slope) < 1e-8:
            remaining = 0.0
        else:
            remaining = max(0.0, (terminal_ratio - float(values[-1])) / slope)
        predictions.append(float(max(remaining, 0.0)))
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / len(test)
    return np.asarray(predictions), elapsed_ms


def metric_row(samples: list[RULSample], prediction_pulses: np.ndarray) -> dict[str, float]:
    """Score direct RUL predictions; target lifetime is used only as a denominator."""
    lifetimes = np.asarray([sample.lifetime for sample in samples], dtype=np.float64)
    truth = np.asarray([sample.target_pulses for sample in samples], dtype=np.float64)
    prediction = np.asarray(prediction_pulses, dtype=np.float64)
    error = prediction - truth
    safe = np.maximum(truth, 1.0)
    return {
        "normalized_mae": float(np.mean(np.abs(error) / lifetimes)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error) / safe) * 100.0),
        "bias": float(np.mean(error)),
    }
