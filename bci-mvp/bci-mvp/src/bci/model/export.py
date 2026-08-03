"""ONNX export + parity check, with no mlflow/optuna dependency so the serving
path and tests stay lightweight."""
from __future__ import annotations

import os
import numpy as np
import torch

from bci.contracts import N_CHANNELS, WINDOW_SAMPLES


def export_onnx(model, path: str = "artifacts/model.onnx") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.eval()
    dummy = torch.randn(1, N_CHANNELS, WINDOW_SAMPLES)
    torch.onnx.export(
        model, dummy, path,
        input_names=["epoch"], output_names=["logits"],
        dynamic_axes={"epoch": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = sess.run(None, {"epoch": dummy.numpy()})[0]
    assert np.allclose(torch_out, onnx_out, atol=1e-4), "ONNX parity failed"
    print(f"Exported + parity-checked: {path}")
    return path
