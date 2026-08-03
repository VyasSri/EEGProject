"""Frozen interfaces between subsystems.

These dataclasses are the contract. `ingest` produces `Epoch`; `model` trains on
batches of epochs; `serve` consumes the ONNX model and emits `Decode`; the
frontend consumes `Decode` over WebSocket. Do not change field names/shapes
without updating every consumer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal
import numpy as np

# ---- Signal geometry -------------------------------------------------------
# 22 EEG channels, matching BNCI2014001 (BCI IV-2a) used for training in
# model/train.py. OpenBCI Cyton is 8ch by default; real hardware in Phase 7
# needs a Cyton+Daisy (16ch) or a channel-subset/retrain to match this.
N_CHANNELS = 22
SFREQ = 250             # Hz, resampled target for all sources
WINDOW_MS = 500
STRIDE_MS = 100
WINDOW_SAMPLES = SFREQ * WINDOW_MS // 1000   # 125
STRIDE_SAMPLES = SFREQ * STRIDE_MS // 1000   # 25

# ---- Classes ---------------------------------------------------------------
# Keep this small for the MVP. left/right hand motor imagery + rest.
Label = Literal["left", "right", "rest"]
CLASSES: tuple[Label, ...] = ("left", "right", "rest")


@dataclass(frozen=True)
class Epoch:
    """One windowed segment ready for inference.

    data: float32 array, shape (N_CHANNELS, WINDOW_SAMPLES), already filtered
          and z-scored per channel.
    t_end: monotonic seconds timestamp of the last sample (for latency calc).
    """
    data: np.ndarray
    t_end: float

    def validate(self) -> None:
        assert self.data.dtype == np.float32, self.data.dtype
        assert self.data.shape == (N_CHANNELS, WINDOW_SAMPLES), self.data.shape


@dataclass(frozen=True)
class Decode:
    """Model output for one epoch, sent to the frontend."""
    label: Label
    confidence: float          # softmax max, 0..1
    probs: dict[str, float]    # full distribution
    t_end: float               # echoed from Epoch, for end-to-end latency
    inference_ms: float

    def to_json(self) -> dict:
        return asdict(self)
