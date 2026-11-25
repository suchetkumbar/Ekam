"""Resumable, checksummed HTTP downloads with a JSON manifest.

The manifest (``<dir>/manifest.json``) records url, bytes, sha256 and fetch time for every
file so ``docs/DATA.md`` can cite exactly what was used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tqdm import tqdm

log = logging.getLogger(__name__)

USER_AGENT = "ekam-research-downloader/0.1 (+open-data; non-commercial)"
_CHUNK = 1 << 16


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest(directory: Path) -> dict:
    mf = directory / "manifest.json"
    return json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}


def _save_manifest(directory: Path, manifest: dict) -> None:
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def download(
    url: str,
    dest: Path,
    *,
    timeout_s: float = 120,
    retries: int = 5,
    expect_prefix: bytes | None = None,
) -> Path:
    """Download ``url`` to ``dest`` (skipping if already complete), retrying with backoff.

    The server behind data.gov.in ignores Range requests, so a failed transfer restarts
    from zero; the partial file is written to ``<dest>.part`` and only renamed on success.
    ``expect_prefix`` guards against HTML error pages being saved as data.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(dest.parent)
    if dest.exists() and dest.name in manifest:
        log.info("skip (already downloaded): %s", dest.name)
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with httpx.stream(
                "GET",
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout_s,
                follow_redirects=True,
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                first = True
                with part.open("wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
                ) as bar:
                    for chunk in r.iter_bytes(_CHUNK):
                        if first and expect_prefix is not None:
                            if not chunk.lstrip(b"\xef\xbb\xbf").startswith(expect_prefix):
                                raise ValueError(
                                    f"unexpected content at {url!r}: {chunk[:60]!r}"
                                )
                            first = False
                        f.write(chunk)
                        bar.update(len(chunk))
            part.replace(dest)
            break
        except (httpx.HTTPError, ValueError) as e:
            log.warning("attempt %d/%d failed for %s: %s", attempt, retries, url, e)
            if isinstance(e, ValueError) or attempt == retries:
                part.unlink(missing_ok=True)
                raise
            time.sleep(min(60, 2**attempt))

    manifest[dest.name] = {
        "url": url,
        "bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _save_manifest(dest.parent, manifest)
    log.info("downloaded %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest
