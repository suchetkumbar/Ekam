"""Ditto-style cross-encoder matcher: a small pre-trained multilingual transformer reads both
serialised records together and outputs a match logit.

Training: binary cross-entropy, AdamW, linear warm-up/decay, fp16 autocast on CUDA, optional
Ditto augmentation, model selection on a fixed sub-sample of the valid split. Scoring sorts
pairs by length so batches carry little padding. Library code: plain arguments, no Hydra.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CEConfig:
    model: str = "intfloat/multilingual-e5-small"
    max_len: int = 128
    batch_size: int = 32
    eval_batch_size: int = 512
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup: float = 0.06
    epochs: int = 2
    fp16: bool = True
    augment_p: float = 0.0
    augment_ops: tuple[str, ...] = ()
    grad_clip: float = 1.0


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


class CrossEncoder:
    def __init__(self, cfg: CEConfig, seed: int = 0):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        set_seed(seed)
        self.cfg, self.seed = cfg, seed
        self.device = _device()
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        try:  # fused attention kernels: ~30% faster scoring on the RTX 3060
            self.model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model, num_labels=1, attn_implementation="sdpa")
        except (ValueError, ImportError):
            self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model,
                                                                            num_labels=1)
        self.model.to(self.device)
        self.history: list[dict] = []

    # ---- encoding ---------------------------------------------------------------------
    def _batch(self, left: list[str], right: list[str]):
        enc = self.tok(left, right, truncation="longest_first", max_length=self.cfg.max_len,
                       padding=True, return_tensors="pt")
        return {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

    def _autocast(self):
        import torch

        on = self.cfg.fp16 and self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=on)

    # ---- training -----------------------------------------------------------------------
    def fit(self, left: list[str], right: list[str], y: np.ndarray,
            val: tuple[list[str], list[str], np.ndarray] | None = None) -> CrossEncoder:
        import torch
        from transformers import get_linear_schedule_with_warmup

        from ekam.blocking.metrics import best_threshold, prf
        from ekam.match.augment import augment_pair

        c = self.cfg
        n = len(y)
        steps_per_epoch = (n + c.batch_size - 1) // c.batch_size
        total = steps_per_epoch * c.epochs
        no_decay = ("bias", "LayerNorm.weight", "layer_norm")
        groups = [
            {"params": [p for k, p in self.model.named_parameters()
                        if not any(nd in k for nd in no_decay)], "weight_decay": c.weight_decay},
            {"params": [p for k, p in self.model.named_parameters()
                        if any(nd in k for nd in no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=c.lr)
        sched = get_linear_schedule_with_warmup(opt, int(c.warmup * total), total)
        scaler = torch.amp.GradScaler(enabled=c.fp16 and self.device.type == "cuda")
        loss_fn = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.seed)
        arng = random.Random(self.seed)
        best, best_state = -1.0, None
        yt = torch.as_tensor(y, dtype=torch.float32)
        for ep in range(c.epochs):
            self.model.train()
            order = rng.permutation(n)
            t0, run_loss = time.perf_counter(), 0.0
            for s in range(steps_per_epoch):
                ix = order[s * c.batch_size:(s + 1) * c.batch_size]
                bl, br = [left[i] for i in ix], [right[i] for i in ix]
                if c.augment_p > 0 and c.augment_ops:
                    for j in range(len(ix)):
                        if arng.random() < c.augment_p:
                            bl[j], br[j] = augment_pair(bl[j], br[j], arng, tuple(c.augment_ops))
                batch = self._batch(bl, br)
                with self._autocast():
                    logits = self.model(**batch).logits.squeeze(-1)
                loss = loss_fn(logits.float(), yt[ix].to(self.device))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                run_loss += float(loss.detach())
            rec = {"epoch": ep + 1, "loss": run_loss / steps_per_epoch,
                   "seconds": time.perf_counter() - t0}
            if val is not None:
                sv = self.score(val[0], val[1])
                thr = best_threshold(val[2], sv)
                rec["val_f1"] = prf(val[2], (sv >= thr).astype(np.int8))["f1"]
                if rec["val_f1"] > best:
                    best = rec["val_f1"]
                    best_state = {k: v.detach().clone() for k, v in
                                  self.model.state_dict().items()}
            self.history.append(rec)
            log.info("epoch %d: loss %.4f, val F1 %s (%.0fs, %d pairs)", ep + 1, rec["loss"],
                     f"{rec.get('val_f1', float('nan')):.4f}", rec["seconds"], n)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # ---- scoring ------------------------------------------------------------------------
    def score(self, left: list[str], right: list[str]) -> np.ndarray:
        """Match probability per pair (aligned with the inputs)."""
        import torch

        self.model.eval()
        n = len(left)
        lens = np.fromiter((len(a) + len(b) for a, b in zip(left, right, strict=True)),
                           dtype=np.int64, count=n)
        order = np.argsort(-lens, kind="stable")
        out = np.empty(n, dtype=np.float32)
        bs = self.cfg.eval_batch_size
        with torch.inference_mode():
            for s in range(0, n, bs):
                ix = order[s:s + bs]
                batch = self._batch([left[i] for i in ix], [right[i] for i in ix])
                with self._autocast():
                    logits = self.model(**batch).logits.squeeze(-1)
                out[ix] = torch.sigmoid(logits.float()).cpu().numpy()
        return out

    def save(self, path) -> None:
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
