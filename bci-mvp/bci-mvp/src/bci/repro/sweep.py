"""Optuna hyperparameter sweep: TPE sampler + median pruner, each trial logged
to MLflow. Mirrors the "200-trial distributed TPE sweep with median pruning"
story; point `--storage` at a shared DB to distribute across GPUs/processes.

Run: python -m bci.repro.sweep --n-trials 200 --storage sqlite:///optuna.db
"""
from __future__ import annotations

import argparse
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from bci.repro import tracking


def suggest_cfg(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "d_model": trial.suggest_categorical("d_model", [16, 32, 64]),
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "lora_r": trial.suggest_categorical("lora_r", [0, 4, 8]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "epochs": 40,
        "subjects": [1, 2, 3],
    }


def make_objective(train_fn):
    """train_fn(cfg, trial) -> best_val_f1. It should call trial.report(...) and
    check trial.should_prune() each epoch to enable median pruning."""

    def objective(trial: optuna.Trial) -> float:
        from bci.model.train import _dataset_files

        cfg = suggest_cfg(trial)
        dataset_files = _dataset_files(cfg["subjects"])
        with tracking.run(cfg, dataset_path=dataset_files):
            return train_fn(cfg, trial)

    return objective


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--storage", default="sqlite:///optuna.db")
    p.add_argument("--study", default="bci-mvp")
    args = p.parse_args()

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=0),
        pruner=MedianPruner(n_warmup_steps=5),
    )

    from bci.model.train import train_one  # imported here to avoid heavy deps
    study.optimize(make_objective(train_one), n_trials=args.n_trials)

    print("Best F1:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
