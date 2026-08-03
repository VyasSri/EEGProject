"""LSL consumer -> ring buffer -> overlapping epochs.

Works identically for the synthetic stream, a replayed recording, or a real
OpenBCI stream: they all present as an LSL EEG inlet. Swap the source, keep the
rest.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Iterator, Optional
import numpy as np
from scipy import signal

from bci.contracts import (
    Epoch, N_CHANNELS, SFREQ, WINDOW_SAMPLES, STRIDE_SAMPLES,
)


class BandpassFilter:
    """Causal 8-30 Hz bandpass (motor imagery band) applied per channel with
    filter-state carry-over so streaming epochs stay continuous."""

    def __init__(self, low: float = 8.0, high: float = 30.0, order: int = 4):
        self.sos = signal.butter(order, [low, high], btype="band",
                                 fs=SFREQ, output="sos")
        self.zi = np.repeat(
            signal.sosfilt_zi(self.sos)[:, None, :],
            N_CHANNELS, axis=1,
        )  # (n_sections, n_channels, 2)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        # chunk: (n_channels, n_samples)
        out, self.zi = signal.sosfilt(self.sos, chunk, axis=1, zi=self.zi)
        return out


class EpochStreamer:
    """Pulls from an LSL inlet, maintains a ring buffer, yields fixed-shape
    z-scored Epochs at STRIDE cadence."""

    def __init__(self, inlet, filt: Optional[BandpassFilter] = None):
        self.inlet = inlet
        self.filt = filt or BandpassFilter()
        self.buf: deque[np.ndarray] = deque(maxlen=WINDOW_SAMPLES)
        self._since_emit = 0

    def _to_epoch(self) -> Epoch:
        arr = np.stack(self.buf, axis=1).astype(np.float32)  # (ch, win)
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True) + 1e-6
        arr = ((arr - mean) / std).astype(np.float32)
        ep = Epoch(data=arr, t_end=time.monotonic())
        ep.validate()
        return ep

    def epochs(self) -> Iterator[Epoch]:
        while True:
            sample, _ = self.inlet.pull_sample()
            filtered = self.filt(np.asarray(sample, np.float32)[:, None])[:, 0]
            self.buf.append(filtered)
            if len(self.buf) < WINDOW_SAMPLES:
                continue
            self._since_emit += 1
            if self._since_emit >= STRIDE_SAMPLES:
                self._since_emit = 0
                yield self._to_epoch()


def open_lsl_inlet(stream_type: str = "EEG", timeout: float = 10.0):
    """Resolve and open an LSL EEG inlet. Import kept local so non-streaming
    code paths (training) don't require pylsl."""
    from pylsl import resolve_byprop, StreamInlet
    streams = resolve_byprop("type", stream_type, timeout=timeout)
    if not streams:
        raise RuntimeError(f"No LSL stream of type {stream_type!r} found")
    return StreamInlet(streams[0], max_buflen=1)
