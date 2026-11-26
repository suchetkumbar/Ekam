"""Download and safely extract public entity-matching benchmarks (conf/benchmarks/*.yaml)."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from omegaconf import DictConfig

from ekam.utils.download import download

log = logging.getLogger(__name__)


def safe_extract(zip_path: Path, out_dir: Path) -> list[Path]:
    """Extract a zip, refusing absolute paths / ``..`` entries (zip-slip)."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX"):
                continue
            target = (out_dir / info.filename).resolve()
            if not target.is_relative_to(out_dir):
                raise ValueError(f"unsafe path in {zip_path.name}: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                while block := src.read(1 << 20):
                    dst.write(block)
            written.append(target)
    return written


def download_benchmarks(cfg: DictConfig, names: list[str] | None = None) -> dict[str, Path]:
    bcfg = cfg.benchmarks
    raw_dir, out_dir = Path(bcfg.raw_dir), Path(bcfg.out_dir)
    result = {}
    for name, spec in bcfg.datasets.items():
        if names and name not in names:
            continue
        zp = download(spec.url, raw_dir / f"{name}.zip", expect_prefix=b"PK")
        dest = out_dir / name
        if not dest.exists() or not any(dest.rglob("*")):
            files = safe_extract(zp, dest)
            log.info("extracted %s: %d files", name, len(files))
        result[name] = dest
    return result


if __name__ == "__main__":
    import sys

    from ekam.utils.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    download_benchmarks(load_config(sys.argv[1:]))
