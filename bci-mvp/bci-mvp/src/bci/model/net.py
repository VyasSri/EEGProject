"""Compact motor-imagery decoder.

An EEGNet-style convolutional frontend (cheap, proven on MI data) feeding a small
transformer encoder with a LoRA-adaptable classification head. For the MVP the
transformer is thin; the LoRA path exists so the "LoRA-adapted transformer"
fine-tuning story is real and per-subject adaptation is a config flag, not a
rewrite.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bci.contracts import N_CHANNELS, WINDOW_SAMPLES, CLASSES


class LoRALinear(nn.Module):
    """Linear layer with an optional low-rank adapter. When r=0 it's a plain
    frozen-friendly Linear; when r>0 the base weight can be frozen and only A,B
    trained."""

    def __init__(self, in_f: int, out_f: int, r: int = 0, alpha: int = 16):
        super().__init__()
        self.base = nn.Linear(in_f, out_f)
        self.r = r
        if r > 0:
            self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
            self.B = nn.Parameter(torch.zeros(out_f, r))
            self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            out = out + (x @ self.A.T @ self.B.T) * self.scale
        return out

    def freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad = False


class EEGNetFrontend(nn.Module):
    def __init__(self, n_ch: int = N_CHANNELS, f1: int = 8, d: int = 2,
                 dropout: float = 0.25):
        super().__init__()
        f2 = f1 * d
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise = nn.Conv2d(f1, f2, (n_ch, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.separable = nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()
        self.f2 = f2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_ch, win) -> (B, 1, n_ch, win)
        x = x.unsqueeze(1)
        x = self.bn1(self.temporal(x))
        x = self.act(self.bn2(self.depthwise(x)))
        x = self.drop(self.pool1(x))
        x = self.act(self.bn3(self.separable(x)))
        x = self.drop(self.pool2(x))
        # (B, f2, 1, T') -> (B, T', f2) token sequence for the transformer
        return x.squeeze(2).transpose(1, 2)


class Decoder(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), d_model: int = 16,
                 n_heads: int = 2, n_layers: int = 2, lora_r: int = 0,
                 dropout: float = 0.25):
        super().__init__()
        self.front = EEGNetFrontend(dropout=dropout)
        self.proj = nn.Linear(self.front.f2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = LoRALinear(d_model, n_classes, r=lora_r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.front(x))
        h = self.encoder(h)
        h = h.mean(dim=1)          # mean-pool tokens
        return self.head(h)

    def enable_lora_finetune(self) -> None:
        """Freeze everything except LoRA adapter params for per-subject tuning."""
        for p in self.parameters():
            p.requires_grad = False
        if self.head.r > 0:
            self.head.A.requires_grad = True
            self.head.B.requires_grad = True


def build_model(cfg: dict) -> Decoder:
    return Decoder(
        d_model=cfg.get("d_model", 16),
        n_heads=cfg.get("n_heads", 2),
        n_layers=cfg.get("n_layers", 2),
        lora_r=cfg.get("lora_r", 0),
        dropout=cfg.get("dropout", 0.25),
    )
