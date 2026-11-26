"""Download the MCA company master CSVs (one per state) listed in ``conf/data/mca.yaml``."""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from ekam.utils.download import download

log = logging.getLogger(__name__)


def download_mca(cfg: DictConfig, states: list[str] | None = None) -> list[Path]:
    raw_dir = Path(cfg.data.raw_dir)
    out = []
    for state in states or list(cfg.data.states):
        url = cfg.data.url_template.format(state=state)
        out.append(
            download(
                url,
                raw_dir / f"{state}.csv",
                timeout_s=cfg.data.timeout_s,
                retries=cfg.data.retries,
                expect_prefix=b'"CORPORATE_IDENTIFICATION_NUMBER"',
            )
        )
    return out


if __name__ == "__main__":
    import sys

    from ekam.utils.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_mca(load_config(sys.argv[1:]))
