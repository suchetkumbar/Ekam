"""One logging setup for every entrypoint (quiet third-party HTTP chatter)."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
