from __future__ import annotations

"""Source-mechanism-faithful adapters for the TORD015 RUL comparison.

The THUML models are imported from the frozen official Time-Series-Library
snapshot under ``references/official_baselines`` and use its classification
interface with a one-dimensional normalized log-RUL target. N-BEATS uses the
frozen ServiceNow reference blocks. Pathformer and SA-MCD require task-specific
adapters because their official input/target contracts are unavailable for the
peak-voltage-only SiC-PCSS benchmark; those deviations are explicit in the
fidelity report and model names.
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


WORKSPACE = Path(__file__).resolve().parents[3]
TSLIB = WORKSPACE / "references" / "official_baselines" / "Time-Series-Library-main"
NBEATS_FILE = (
    WORKSPACE / "references" / "official_baselines" / "N-BEATS-master" / "models" / "nbeats.py"
)


def _tslib_modules():
    root = str(TSLIB)
    if root not in sys.path:
        sys.path.insert(0, root)
    from models import DLinear, PatchTST, TimesNet  # type: ignore
    from layers.Embed import DataEmbedding  # type: ignore
    from layers.SelfAttention_Family import AttentionLayer, ProbAttention  # type: ignore
    from layers.Transformer_EncDec import ConvLayer, Encoder, EncoderLayer  # type: ignore

    return DLinear, PatchTST, TimesNet, DataEmbedding, AttentionLayer, ProbAttention, ConvLayer, Encoder, EncoderLayer


def _config(channels: int, length: int, hidden: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="classification",
        seq_len=length,
        label_len=0,
        pred_len=0,
        enc_in=channels,
        dec_in=channels,
        c_out=channels,
        d_model=hidden,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=hidden * 2,
        moving_avg=25,
        factor=3,
        distil=True,
        dropout=0.1,
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        num_class=1,
        top_k=3,
        num_kernels=6,
    )


class TSLibClassificationAdapter(nn.Module):
    """Use an official TSLib architecture with its classification head."""

    def __init__(self, name: str, channels: int, length: int, hidden: int):
        super().__init__()
        DLinear, PatchTST, TimesNet, *_ = _tslib_modules()
        config = _config(channels, length, hidden)
        constructors = {
            "dlinear": DLinear.Model,
            "patchtst": PatchTST.Model,
            "timesnet": TimesNet.Model,
        }
        self.model = constructors[name](config)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence = x.transpose(1, 2)
        valid = torch.ones(sequence.shape[:2], device=x.device, dtype=x.dtype)
        raw = self.model(sequence, valid, None, None).squeeze(-1)
        return {"prediction": torch.sigmoid(raw)}


class InformerEncoderRUL(nn.Module):
    """Informer ProbSparse encoder with distillation and a scalar RUL head."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        (
            _, _, _, DataEmbedding, AttentionLayer, ProbAttention,
            ConvLayer, Encoder, EncoderLayer,
        ) = _tslib_modules()
        self.embedding = DataEmbedding(channels, hidden, "fixed", "h", 0.1)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor=3, attention_dropout=0.1, output_attention=False),
                        hidden,
                        4,
                    ),
                    hidden,
                    hidden * 2,
                    dropout=0.1,
                    activation="gelu",
                )
                for _ in range(2)
            ],
            [ConvLayer(hidden)],
            norm_layer=nn.LayerNorm(hidden),
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.embedding(x.transpose(1, 2), None)
        encoded, _ = self.encoder(embedded, attn_mask=None)
        return {"prediction": torch.sigmoid(self.head(encoded.mean(1)).squeeze(-1))}


def _load_nbeats_module():
    spec = importlib.util.spec_from_file_location("tor_official_nbeats", NBEATS_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load N-BEATS reference from {NBEATS_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NBeatsRUL(nn.Module):
    """Official generic residual-backcast N-BEATS core adapted to scalar RUL."""

    def __init__(self, channels: int, length: int, hidden: int):
        super().__init__()
        reference = _load_nbeats_module()
        self.pre = nn.Conv1d(channels, 1, 1)
        blocks = []
        layer_size = max(64, hidden * 2)
        for _ in range(4):
            blocks.append(
                reference.NBeatsBlock(
                    input_size=length,
                    theta_size=length + 1,
                    basis_function=reference.GenericBasis(length, 1),
                    layers=4,
                    layer_size=layer_size,
                )
            )
        self.model = reference.NBeats(nn.ModuleList(blocks))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        proxy = self.pre(x).squeeze(1)
        forecast = self.model(proxy, torch.ones_like(proxy)).squeeze(-1)
        return {"prediction": torch.sigmoid(forecast)}


class PathExpert(nn.Module):
    """Dual-attention expert at one temporal patch scale."""

    def __init__(self, channels: int, hidden: int, patch: int):
        super().__init__()
        self.patch = patch
        self.patch_embed = nn.Linear(patch, hidden)
        self.temporal_attention = nn.MultiheadAttention(hidden, 4, dropout=0.1, batch_first=True)
        self.channel_attention = nn.MultiheadAttention(hidden, 4, dropout=0.1, batch_first=True)
        self.norm_t = nn.LayerNorm(hidden)
        self.norm_c = nn.LayerNorm(hidden)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        usable = (length // self.patch) * self.patch
        patches = x[..., -usable:].reshape(batch, channels, usable // self.patch, self.patch)
        tokens = self.patch_embed(patches)
        temporal = tokens.reshape(batch * channels, tokens.shape[2], tokens.shape[3])
        attended, _ = self.temporal_attention(temporal, temporal, temporal, need_weights=False)
        temporal = self.norm_t(temporal + attended).mean(1).reshape(batch, channels, -1)
        mixed, _ = self.channel_attention(temporal, temporal, temporal, need_weights=False)
        return self.norm_c(temporal + mixed).mean(1)


class PathformerRULAdapter(nn.Module):
    """RUL adapter preserving multi-scale patches, dual attention, and adaptive routing."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.experts = nn.ModuleList([PathExpert(channels, hidden, p) for p in (4, 8, 16, 32)])
        self.router = nn.Sequential(
            nn.Linear(channels * 2, hidden), nn.GELU(), nn.Linear(hidden, len(self.experts))
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        descriptor = torch.cat([x.mean(-1), x.std(-1, unbiased=False)], dim=-1)
        logits = self.router(descriptor)
        top_values, top_indices = logits.topk(2, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)
        gates = torch.zeros_like(logits).scatter(1, top_indices, top_weights)
        paths = torch.stack([expert(x) for expert in self.experts], dim=1)
        hidden = (paths * gates.unsqueeze(-1)).sum(1)
        importance = gates.mean(0)
        balance = importance.var(unbiased=False) / (importance.mean().square() + 1e-8)
        return {
            "prediction": torch.sigmoid(self.head(hidden).squeeze(-1)),
            "aux_loss": 0.01 * balance,
        }


class SAMCDRULAdapter(nn.Module):
    """Peak-voltage-compatible self-attention RUL model with inference-time MCD."""

    mc_samples = 30
    mc_validation_samples = 10

    def __init__(self, channels: int, hidden: int, length: int):
        super().__init__()
        self.embed = nn.Linear(channels, hidden)
        self.position = nn.Parameter(torch.zeros(1, length, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 2, dropout=0.15, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.embed(x.transpose(1, 2)) + self.position[:, : x.shape[-1]]
        hidden = self.encoder(tokens).mean(1)
        return {"prediction": torch.sigmoid(self.head(self.dropout(hidden)).squeeze(-1))}


def build_faithful_model(name: str, channels: int, length: int, hidden: int) -> nn.Module:
    if name == "dlinear_faithful":
        return TSLibClassificationAdapter("dlinear", channels, length, hidden)
    if name == "patchtst_faithful":
        return TSLibClassificationAdapter("patchtst", channels, length, hidden)
    if name == "timesnet_faithful":
        return TSLibClassificationAdapter("timesnet", channels, length, hidden)
    if name == "informer_faithful":
        return InformerEncoderRUL(channels, hidden)
    if name == "nbeats_faithful":
        return NBeatsRUL(channels, length, hidden)
    if name == "pathformer_adapted":
        return PathformerRULAdapter(channels, hidden)
    if name == "sa_mcd_adapted":
        return SAMCDRULAdapter(channels, hidden, length)
    raise KeyError(name)
