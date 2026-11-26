"""Ditto's data-augmentation operators on serialised records (generic, not our noise model).

Deliberately *not* the generator's own noise functions: augmenting with the exact noise
used to make the test data would teach the model the generator, not entity matching.
Operators (one is drawn per augmented example):

* ``span_del``  - delete a span of 1-2 tokens inside one value
* ``span_shuffle`` - shuffle the tokens of a span of up to 4 tokens inside one value
* ``attr_del``  - empty one attribute value (keeps its ``COL`` tag)
* ``attr_shuffle`` - shuffle the order of the attributes
* ``side_swap`` - swap left and right (the label is symmetric)
"""

from __future__ import annotations

import random
import re

OPS = ("span_del", "span_shuffle", "attr_del", "attr_shuffle", "side_swap")
_FIELD = re.compile(r"COL (\S+) VAL ?")


def parse(s: str) -> list[tuple[str, str]]:
    parts = _FIELD.split(s)
    # parts = ['', f1, v1, f2, v2, ...]
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def render(fields: list[tuple[str, str]]) -> str:
    return " ".join(f"COL {f} VAL {v}".rstrip() for f, v in fields)


def _value_op(fields, rng, fn):
    idx = [i for i, (_, v) in enumerate(fields) if len(v.split()) >= 2]
    if not idx:
        return fields
    i = rng.choice(idx)
    toks = fields[i][1].split()
    fields = list(fields)
    fields[i] = (fields[i][0], " ".join(fn(toks)))
    return fields


def augment_record(s: str, op: str, rng: random.Random) -> str:
    fields = parse(s)
    if not fields:
        return s
    if op == "span_del":
        def f(t):
            n = rng.randint(1, min(2, len(t) - 1))
            k = rng.randrange(0, len(t) - n + 1)
            return t[:k] + t[k + n:]
        fields = _value_op(fields, rng, f)
    elif op == "span_shuffle":
        def f(t):
            n = rng.randint(2, min(4, len(t)))
            k = rng.randrange(0, len(t) - n + 1)
            span = t[k:k + n]
            rng.shuffle(span)
            return t[:k] + span + t[k + n:]
        fields = _value_op(fields, rng, f)
    elif op == "attr_del":
        idx = [i for i, (_, v) in enumerate(fields) if v]
        if idx:
            i = rng.choice(idx)
            fields[i] = (fields[i][0], "")
    elif op == "attr_shuffle":
        rng.shuffle(fields)
    return render(fields)


def augment_pair(left: str, right: str, rng: random.Random,
                 ops: tuple[str, ...] = OPS) -> tuple[str, str]:
    op = rng.choice(ops)
    if op == "side_swap":
        return right, left
    if rng.random() < 0.5:
        return augment_record(left, op, rng), right
    return left, augment_record(right, op, rng)
