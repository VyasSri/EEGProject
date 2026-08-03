"""Thin MLflow wrapper that makes every run reproducible: it logs the resolved
config, the git SHA, the dataset hash/version, and metrics. This is the
"reproduce in minutes across 40+ checkpoints" layer.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional
import mlflow


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def dataset_hash(paths: str | Path | Iterable[str | Path]) -> str:
    """Stable hash of dataset file(s)/dir(s) for provenance. Accepts a single
    path or an iterable of paths (e.g. resolved per-subject file paths)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.rglob("*")) if p.is_dir() else [p])
    h = hashlib.sha256()
    for f in sorted(set(files)):
        if f.is_file():
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


@contextmanager
def run(cfg: dict, dataset_path: Optional[str | Path | Iterable[str | Path]] = None,
        experiment: str = "bci-mvp") -> Iterator[mlflow.ActiveRun]:
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as r:
        mlflow.log_params(_flatten(cfg))
        mlflow.set_tag("git_sha", _git_sha())
        if dataset_path:
            mlflow.set_tag("dataset_hash", dataset_hash(dataset_path))
        mlflow.log_text(json.dumps(cfg, indent=2), "resolved_config.json")
        yield r


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out
