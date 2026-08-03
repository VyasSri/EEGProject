# BCI-MVP: Live Motor-Imagery Cursor Control

Real-time motor-imagery BCI: imagined left/right hand movement → cursor control,
with a reproducible training pipeline (Optuna + MLflow) and a low-latency
ONNX inference path.

## Subsystems

1. **Ingest** (`src/bci/ingest/`) — LSL consumer, ring buffer, epoch windowing.
   Runs on synthetic/replayed data with no hardware.
2. **Model** (`src/bci/model/`) — PyTorch decoder (EEGNet frontend + optional
   LoRA transformer head), training loop, ONNX export.
3. **Serve** (`src/bci/serve/`) — asyncio inference service + WebSocket, latency
   instrumentation.
4. **Repro** (`src/bci/repro/`) — MLflow logging wrapper, Optuna TPE sweep with
   median pruning.
5. **Frontend** (`frontend/`) — React + WebSocket, velocity-based cursor control.

## Build order

Each phase is independently testable. Do NOT jump ahead — each phase produces an
artifact the next consumes.

| Phase | Deliverable | Hardware |
|-------|-------------|----------|
| 0 | Repo, deps, config, synthetic LSL stream | No |
| 1 | Dataset loader + epoching, verified shapes | No |
| 2 | Model + training loop, a real F1 number | No |
| 3 | MLflow wrapper + Optuna sweep | No |
| 4 | ONNX export + parity check | No |
| 5 | asyncio inference service + WebSocket | No |
| 6 | React cursor frontend, end-to-end on replay | No |
| 7 | Real OpenBCI integration | Yes |

## Design decisions (do not relitigate mid-build)

- **Velocity control, not absolute position** — decoded class + confidence maps
  to cursor *velocity*. Far more robust to noisy decodes.
- **Windows**: 500 ms epochs, 100 ms stride (overlapping). Tunable in config.
- **Latency budget**: window stride (100ms) + inference (<20ms) + transport
  (<10ms) leaves headroom under 200ms. Instrument every hop.
- **Contract-first**: `ingest` emits fixed-shape float32 epochs; `serve` consumes
  the ONNX model + epoch stream. Interfaces are frozen in `contracts.py`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/synthetic_stream.py      # terminal 1: fake LSL stream
python -m bci.serve.app                  # terminal 2: inference service
cd frontend && npm install && npm run dev  # terminal 3: UI
```
