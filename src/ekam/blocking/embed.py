"""Sentence embeddings for nearest-neighbour blocking, computed once and cached.

Texts are de-duplicated before encoding (many records share a normalised name+city), the
model runs in fp16 on the GPU when available, and vectors are L2-normalised float16 on
disk (``.npy``) next to the record ids they belong to.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def record_texts(df: pl.DataFrame, fields: list[str]) -> list[str]:
    return df.select(pl.concat_str([pl.col(f).fill_null("") for f in fields], separator=" ")
                     .str.strip_chars().alias("t"))["t"].to_list()


def _load_model(enc: DictConfig):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(enc.name, device=device)
    model.max_seq_length = int(enc.max_seq_length)
    if enc.fp16 and device == "cuda":
        model = model.half()
    return model, device


def encode(texts: list[str], enc: DictConfig, model=None) -> np.ndarray:
    uniq, inverse = np.unique(np.array(texts, dtype=object), return_inverse=True)
    model, device = (model, "given") if model is not None else _load_model(enc)
    t0 = time.perf_counter()
    vecs = model.encode([enc.prefix + t for t in uniq.tolist()], batch_size=int(enc.batch_size),
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False).astype(np.float16)
    log.info("encoded %d unique texts (%d records) on %s in %.1fs", len(uniq), len(texts),
             device, time.perf_counter() - t0)
    return vecs[inverse]


def cached_embeddings(cfg: DictConfig, level: str, split: str, df: pl.DataFrame,
                      model=None) -> np.ndarray:
    enc = cfg.encoder
    tag = hashlib.sha1(f"{enc.name}|{enc.prefix}|{list(enc.text_fields)}".encode()).hexdigest()[:8]
    path = Path(cfg.paths.processed) / level / f"emb_{split}_{tag}.npy"
    ids_path = path.with_suffix(".ids.parquet")
    if path.exists() and ids_path.exists():
        ids = pl.read_parquet(ids_path)["record_id"]
        if ids.equals(df["record_id"]):
            return np.load(path)
    vecs = encode(record_texts(df, list(enc.text_fields)), enc, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    df.select("record_id").write_parquet(ids_path)
    return vecs
