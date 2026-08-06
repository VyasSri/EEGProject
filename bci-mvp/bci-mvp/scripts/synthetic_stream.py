"""Emit a synthetic 8-channel EEG LSL stream so the whole pipeline runs with no
hardware. Motor imagery is faked as a mu-band (~10 Hz) power drop (ERD) on
contralateral channels, cycling left / right / rest every few seconds so you can
eyeball whether the decoder is learning anything real.

Run: python scripts/synthetic_stream.py
Requires: pylsl
"""
from __future__ import annotations

import math
import time
import numpy as np

try:
    from pylsl import StreamInfo, StreamOutlet
except ImportError:
    raise SystemExit("pip install pylsl  (and liblsl: see pylsl docs)")

N_CH = 22  # matches contracts.N_CHANNELS (BNCI2014001 training montage)
SFREQ = 250
CYCLE_S = 4.0  # seconds per imagined state

# channel indices we'll treat as "left motor cortex" / "right motor cortex",
# matching BNCI2014001's channel order (C5, C3, C1, Cz, C2, C4, C6 = idx 6-12).
LEFT_CH = [6, 7, 8]
RIGHT_CH = [10, 11, 12]


def main() -> None:
    info = StreamInfo("SyntheticEEG", "EEG", N_CH, SFREQ, "float32", "synth-001")
    outlet = StreamOutlet(info)
    print(f"Streaming synthetic EEG: {N_CH}ch @ {SFREQ}Hz. Ctrl-C to stop.")

    rng = np.random.default_rng(0)
    t = 0.0
    dt = 1.0 / SFREQ
    states = ["left", "right", "rest"]

    while True:
        state = states[int(t / CYCLE_S) % len(states)]
        sample = rng.normal(0, 1.0, N_CH).astype(np.float32)
        # baseline mu rhythm on all channels
        mu = 1.5 * math.sin(2 * math.pi * 10 * t)
        sample += mu
        # ERD: suppress mu on the contralateral hemisphere
        if state == "left":     # left hand -> right hemisphere ERD
            for c in RIGHT_CH:
                sample[c] -= mu * 0.8
        elif state == "right":  # right hand -> left hemisphere ERD
            for c in LEFT_CH:
                sample[c] -= mu * 0.8
        outlet.push_sample(sample)
        t += dt
        time.sleep(dt)


if __name__ == "__main__":
    main()
