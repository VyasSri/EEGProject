"""Live inference service: consume the LSL epoch stream, run the ONNX model,
broadcast Decodes over WebSocket. asyncio so ingestion, inference, and the
socket never block each other. Every hop is timed for the sub-200ms budget.

Run: python -m bci.serve.app --model artifacts/model.onnx
"""
from __future__ import annotations

import argparse
import asyncio
import time
from typing import Set
import numpy as np

from bci.contracts import Decode, CLASSES


class OnnxDecoder:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, data: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        logits = self.sess.run(None, {self.inp: data[None].astype(np.float32)})[0][0]
        infer_ms = (time.perf_counter() - t0) * 1000
        e = np.exp(logits - logits.max())
        return e / e.sum(), infer_ms


async def ingest_loop(decoder: OnnxDecoder, clients: Set, stop: asyncio.Event):
    """Runs the (blocking) LSL pull + inference in a thread executor so the event
    loop stays responsive, then fans out Decodes to all WebSocket clients."""
    from bci.ingest.stream import EpochStreamer, open_lsl_inlet
    loop = asyncio.get_running_loop()
    inlet = await loop.run_in_executor(None, open_lsl_inlet)
    streamer = EpochStreamer(inlet)
    gen = streamer.epochs()

    while not stop.is_set():
        epoch = await loop.run_in_executor(None, next, gen)
        probs, infer_ms = decoder(epoch.data)
        idx = int(probs.argmax())
        decode = Decode(
            label=CLASSES[idx],
            confidence=float(probs[idx]),
            probs={c: float(p) for c, p in zip(CLASSES, probs)},
            t_end=epoch.t_end,
            inference_ms=infer_ms,
        )
        payload = decode.to_json()
        payload["e2e_ms"] = (time.monotonic() - epoch.t_end) * 1000
        dead = set()
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        clients -= dead


def build_app(model_path: str):
    from aiohttp import web

    clients: Set = set()
    stop = asyncio.Event()
    decoder = OnnxDecoder(model_path)

    async def ws_handler(request):
        from aiohttp import web
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            clients.discard(ws)
        return ws

    async def on_start(app):
        app["ingest"] = asyncio.create_task(ingest_loop(decoder, clients, stop))

    async def on_stop(app):
        stop.set()
        app["ingest"].cancel()

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    return app


def main() -> None:
    from aiohttp import web
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/model.onnx")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    web.run_app(build_app(args.model), port=args.port)


if __name__ == "__main__":
    main()
