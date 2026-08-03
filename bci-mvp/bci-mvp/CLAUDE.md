# Claude Code Build Guide

This repo is a **scaffold with frozen contracts**. Your job across the phases
below is to fill stubs and add the real dataset/hardware wiring — not to
redesign. `src/bci/contracts.py` is law: every subsystem conforms to `Epoch` and
`Decode`. `tests/test_pipeline.py` must stay green after every phase.

Work one phase at a time. Run `pytest -q` before moving on.

---

## Phase 0 — Environment (no hardware)
- Create the venv, `pip install -e ".[dev]"`.
- Get `pylsl` + liblsl installed (the native lib, not just the wheel).
- Verify: `python scripts/synthetic_stream.py` streams without error.

## Phase 1 — Real dataset loader (no hardware)
Replace the STUB in `src/bci/model/train.py::load_dataset` with MOABB:
- Use `BNCI2014001` (BCI Competition IV-2a) via `moabb.paradigms.MotorImagery`.
- Map its left-hand/right-hand/(feet or rest) classes onto `CLASSES`.
- Resample to `SFREQ` (250 Hz), bandpass 8–30 Hz, window to `WINDOW_SAMPLES`
  with `STRIDE_SAMPLES`, z-score per channel — matching `ingest/stream.py` so
  train and serve see identically-shaped data.
- **Acceptance**: shapes `(N, N_CHANNELS, WINDOW_SAMPLES)`; a quick train run
  gives macro-F1 clearly above chance (>0.4 on 3 classes).

## Phase 2 — Training (no hardware)
- `python -m bci.model.train` should log a real F1 to MLflow.
- Tune the frontend conv sizes if the dataset's channel count differs from 8
  (set `N_CHANNELS` in `contracts.py` once, centrally).
- **Acceptance**: `mlflow ui` shows the run with params, git SHA, dataset hash.

## Phase 3 — Sweep (no hardware)
- `python -m bci.repro.sweep --n-trials 30 --storage sqlite:///optuna.db`
- Confirm median pruning kills weak trials (look for PRUNED in logs).
- To "distribute": launch the same command in N processes pointing at the same
  `--storage`; Optuna coordinates. One process per GPU, set `CUDA_VISIBLE_DEVICES`.
- **Acceptance**: a study of ≥30 trials, best F1 beating the Phase-2 baseline.

## Phase 4 — Export (no hardware)
- Train best config, `python -m bci.model.train --export`.
- **Acceptance**: `artifacts/model.onnx` exists and parity test passes.

## Phase 5 — Serving (no hardware)
- Terminal A: `python scripts/synthetic_stream.py`
- Terminal B: `python -m bci.serve.app --model artifacts/model.onnx`
- Connect a WebSocket client to `ws://localhost:8080/ws`; confirm JSON Decodes
  arrive with `e2e_ms` populated.
- **Acceptance**: median `e2e_ms` < 200 on your dev machine.

## Phase 6 — Frontend (no hardware)
- `cd frontend && npm install && npm run dev`
- With synthetic stream + serve running, the cursor should drift left/right in
  sync with the synthetic ERD cycle.
- **Acceptance**: end-to-end demo on replayed/synthetic data, latency dot green.

## Phase 7 — Real OpenBCI (hardware)
- Start OpenBCI GUI or `brainflow` LSL bridge so a real EEG LSL stream exists.
- Point nothing new at it — `open_lsl_inlet()` resolves by type "EEG".
- Expect to spend time on: electrode contact/impedance, per-subject calibration
  (run `enable_lora_finetune()` on a short calibration recording), and
  thresholds (`CONF_GATE` in `App.jsx`).
- **Acceptance**: live cursor control from imagined movement.

---

## Guardrails
- Never change `contracts.py` field names/shapes without updating all consumers
  and tests.
- Keep `serve/` free of torch/mlflow imports — it only needs onnxruntime.
- If you add deps, add them to `pyproject.toml`.
- Every phase ends with green `pytest -q`.
