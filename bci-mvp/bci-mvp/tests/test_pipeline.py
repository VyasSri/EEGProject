"""Fast tests that run with no hardware and no trained model. They lock the
contract shapes and the model's forward pass so Claude Code can't silently drift
the interfaces during the build."""
import numpy as np
import torch

from bci.contracts import (
    Epoch, N_CHANNELS, WINDOW_SAMPLES, CLASSES,
)
from bci.model.net import build_model


def test_epoch_shape_contract():
    ep = Epoch(np.zeros((N_CHANNELS, WINDOW_SAMPLES), np.float32), t_end=0.0)
    ep.validate()  # should not raise


def test_model_forward_shape():
    cfg = {"d_model": 16, "n_layers": 2, "n_heads": 2, "lora_r": 0}
    model = build_model(cfg)
    x = torch.randn(4, N_CHANNELS, WINDOW_SAMPLES)
    out = model(x)
    assert out.shape == (4, len(CLASSES))


def test_lora_finetune_freezes_base():
    cfg = {"d_model": 16, "n_layers": 1, "n_heads": 2, "lora_r": 8}
    model = build_model(cfg)
    model.enable_lora_finetune()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("head.A" in n or "head.B" in n for n in trainable), trainable
    assert trainable, "LoRA params should be trainable"


def test_onnx_export_parity(tmp_path):
    from bci.model.export import export_onnx
    import os
    cfg = {"d_model": 16, "n_layers": 1, "n_heads": 2, "lora_r": 0}
    model = build_model(cfg)
    os.chdir(tmp_path)
    export_onnx(model, str(tmp_path / "m.onnx"))  # asserts parity internally
