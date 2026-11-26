"""FAISS nearest-neighbour blocking on cosine similarity of record embeddings."""

from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


def build_index(vecs: np.ndarray, kind: str = "auto", flat_max: int = 150_000, hnsw_m: int = 32,
                ef_search: int = 128, n_threads: int | None = None):
    import faiss

    if n_threads:
        faiss.omp_set_num_threads(int(n_threads))
    x = np.ascontiguousarray(vecs, dtype=np.float32)
    d = x.shape[1]
    if kind == "flat" or (kind == "auto" and len(x) <= flat_max):
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 2 * int(hnsw_m)
        index.hnsw.efSearch = int(ef_search)
    t0 = time.perf_counter()
    index.add(x)
    log.info("faiss %s index on %d x %d built in %.1fs", type(index).__name__, *x.shape,
             time.perf_counter() - t0)
    return index


def knn_search(index, vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(vecs, dtype=np.float32)
    t0 = time.perf_counter()
    sims, nbrs = index.search(x, k + 1)  # +1: a record is its own nearest neighbour
    log.info("faiss search k=%d over %d queries in %.1fs", k, len(x), time.perf_counter() - t0)
    return sims, nbrs


def knn_pairs(sims: np.ndarray, nbrs: np.ndarray, k: int, min_similarity: float = 0.0
              ) -> np.ndarray:
    """Unique (l<r) pairs from the first ``k`` non-self neighbours of every record."""
    n = nbrs.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int64), nbrs.shape[1])
    cols = nbrs.reshape(-1).astype(np.int64)
    s = sims.reshape(-1)
    # rank of each neighbour after dropping self / invalid (-1) entries
    valid = (cols >= 0) & (cols != rows) & (s >= min_similarity)
    rank = np.cumsum(valid.reshape(n, -1), axis=1).reshape(-1)
    keep = valid & (rank <= k)
    p = np.stack([rows[keep], cols[keep]], axis=1)
    p = np.sort(p, axis=1)
    return np.unique(p, axis=0)
