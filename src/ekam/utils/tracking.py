"""Thin MLflow helpers (local SQLite store, one experiment per stage)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from omegaconf import DictConfig


@contextmanager
def mlflow_run(cfg: DictConfig, experiment: str, run_name: str, params: dict | None = None):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=Path(cfg.mlflow.artifact_location).resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        yield mlflow, run
