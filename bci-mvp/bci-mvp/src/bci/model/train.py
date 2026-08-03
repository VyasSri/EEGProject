"""Training loop + ONNX export.

`train_one(cfg, trial)` is the single entry point both the standalone CLI and the
Optuna sweep call. It reports per-epoch validation F1 to the trial so median
pruning can kill weak runs early, logs metrics to MLflow, and returns best F1.

`load_dataset` pulls BNCI2014001 (BCI IV 2a) via MOABB.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import mlflow

from bci.contracts import N_CHANNELS, SFREQ, WINDOW_SAMPLES, STRIDE_SAMPLES, CLASSES
from bci.model.net import build_model

# BNCI2014001 (BCI Competition IV-2a) label -> our 3-class scheme. "feet" is
# the closest available non-hand class, standing in for "rest".
_MOABB_EVENTS = ["left_hand", "right_hand", "feet"]
_MOABB_TO_LABEL = {"left_hand": "left", "right_hand": "right", "feet": "rest"}


# ---- DATA -------------------------------------------------------------------
def load_dataset(cfg: dict):
    """Return (Xtr, ytr, Xval, yval) as float32/int64 tensors.

    Loads BNCI2014001 (BCI IV-2a) via MOABB's MotorImagery paradigm, which
    handles resampling to SFREQ and the 8-30Hz bandpass. Each MOABB trial
    (several seconds long) is then cut into overlapping WINDOW_SAMPLES epochs
    at STRIDE_SAMPLES, z-scored per channel exactly like ingest/stream.py, so
    train and serve see identically-shaped, identically-normalized data.
    Output shapes: X (N, N_CHANNELS, WINDOW_SAMPLES), y (N,) in 0..len(CLASSES).
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    dataset = BNCI2014_001()
    subjects = cfg.get("subjects", [1, 2, 3])
    paradigm = MotorImagery(
        events=_MOABB_EVENTS, n_classes=len(_MOABB_EVENTS),
        fmin=8, fmax=30, resample=SFREQ,
    )
    X, y, _meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    assert X.shape[1] == N_CHANNELS, (
        f"dataset has {X.shape[1]} channels; update N_CHANNELS in contracts.py"
    )

    label_idx = {label: i for i, label in enumerate(CLASSES)}
    y_idx = np.array([label_idx[_MOABB_TO_LABEL[e]] for e in y], dtype=np.int64)

    Xw, yw = _window_trials(X.astype(np.float32), y_idx)

    rng = np.random.default_rng(cfg.get("seed", 0))
    perm = rng.permutation(len(Xw))
    Xw, yw = Xw[perm], yw[perm]
    split = int(len(Xw) * 0.8)
    return (torch.from_numpy(Xw[:split]), torch.from_numpy(yw[:split]),
            torch.from_numpy(Xw[split:]), torch.from_numpy(yw[split:]))


def _window_trials(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut each (n_channels, n_times) trial into overlapping, per-channel
    z-scored WINDOW_SAMPLES epochs at STRIDE_SAMPLES, matching EpochStreamer."""
    windows, labels = [], []
    n_times = X.shape[2]
    starts = range(0, n_times - WINDOW_SAMPLES + 1, STRIDE_SAMPLES)
    for trial, label in zip(X, y):
        for start in starts:
            seg = trial[:, start:start + WINDOW_SAMPLES]
            mean = seg.mean(axis=1, keepdims=True)
            std = seg.std(axis=1, keepdims=True) + 1e-6
            windows.append(((seg - mean) / std).astype(np.float32))
            labels.append(label)
    return np.stack(windows), np.array(labels, dtype=np.int64)


def train_one(cfg: dict, trial=None) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr, Xval, yval = load_dataset(cfg)
    Xtr, ytr, Xval, yval = (t.to(device) for t in (Xtr, ytr, Xval, yval))

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 1e-5))
    lossf = nn.CrossEntropyLoss()
    best_f1 = 0.0
    bs = cfg.get("batch_size", 64)

    for epoch in range(cfg.get("epochs", 40)):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xval).argmax(1).cpu().numpy()
        f1 = f1_score(yval.cpu().numpy(), pred, average="macro")
        best_f1 = max(best_f1, f1)
        mlflow.log_metric("val_f1", f1, step=epoch)

        if trial is not None:
            trial.report(f1, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    mlflow.log_metric("best_val_f1", best_f1)
    return best_f1


from bci.model.export import export_onnx  # noqa: E402  (kept here for CLI use)
from bci.repro import tracking  # noqa: E402


def _dataset_files(subjects: list[int]) -> list[str]:
    """Resolve (downloading if needed) the on-disk files for these subjects,
    for provenance hashing. Called before tracking.run() opens so the hash
    reflects data that actually exists, not an empty/not-yet-downloaded cache."""
    from moabb.datasets import BNCI2014_001
    dataset = BNCI2014_001()
    paths: list[str] = []
    for s in subjects:
        paths.extend(dataset.data_path(s))
    return paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()
    cfg = {"lr": 1e-3, "dropout": 0.25, "d_model": 32, "n_layers": 2,
           "n_heads": 2, "lora_r": 0, "epochs": args.epochs, "weight_decay": 1e-5,
           "subjects": args.subjects}
    dataset_files = _dataset_files(cfg["subjects"])
    with tracking.run(cfg, dataset_path=dataset_files):
        f1 = train_one(cfg)
        print("best F1:", f1)
        if args.export:
            model = build_model(cfg)
            export_onnx(model)


if __name__ == "__main__":
    main()
