"""`make data`: download -> privacy-filtered ingest -> synth sources -> entity split -> audit."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ekam.utils.config import CONF_DIR
from ekam.utils.logging import setup_logging

log = logging.getLogger("ekam.data")


def run(cfg: DictConfig) -> None:
    from ekam.data import audit, benchmarks, download_benchmarks, download_mca, ingest, split
    from ekam.synth import generator

    steps = list(cfg.pipeline.data_steps)
    if "download" in steps:
        download_mca.download_mca(cfg)
    if "benchmarks" in steps:
        download_benchmarks.download_benchmarks(cfg)
        summary = benchmarks.summarise(cfg)
        out = Path(cfg.paths.reports) / "benchmarks_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary.to_dicts(), indent=2), encoding="utf-8")
        log.info("benchmarks:\n%s", summary)
    if "ingest" in steps and (cfg.pipeline.force or not Path(cfg.data.clean_path).exists()):
        ingest.run(cfg)
    if "synth" in steps:
        generator.run(cfg)
    if "split" in steps:
        split.run(cfg)
    if "audit" in steps:
        audit.run(cfg)


@hydra.main(config_path=str(CONF_DIR), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    run(cfg)


if __name__ == "__main__":
    main()
