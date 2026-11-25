"""Load the Hydra config outside of ``@hydra.main`` (for scripts, notebooks and tests)."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = REPO_ROOT / "conf"


def load_config(overrides: list[str] | None = None, config_name: str = "config") -> DictConfig:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides or []))
