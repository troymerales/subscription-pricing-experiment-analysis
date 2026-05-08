"""Utilities: reproducibility, numerics, weighted sampling, export helpers."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Sequence

import numpy as np
from numpy.random import Generator


def set_global_seed(seed: int) -> None:
    """Set seeds for numpy Generator (callers own main rng) and legacy np.random."""
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except ImportError:
        pass


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic / sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def clip_prob(p: np.ndarray, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> np.ndarray:
    return np.clip(p.astype(np.float64), lo, hi)


def weighted_choice(
    rng: Generator,
    labels: Sequence[Any],
    probs: Sequence[float],
    size: int,
) -> np.ndarray:
    """Vectorized categorical draws from labels with given probabilities."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    idx = rng.choice(len(labels), size=size, p=p)
    return np.asarray(labels, dtype=object)[idx]


def apply_mcar_mask(
    rng: Generator,
    arr: np.ndarray,
    nan_prob: float,
) -> np.ndarray:
    """Replace random positions with np.nan (object array for strings)."""
    if nan_prob <= 0:
        return arr.copy()
    mask = rng.random(arr.shape) < nan_prob
    out = arr.astype(object, copy=True)
    out[mask] = np.nan
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def batch_iterator(n: int, chunk: int):
    """Yield (start, stop) ranges for [0, n)."""
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def months_from_signup(rng: Generator, max_days: int, size: int) -> np.ndarray:
    """Months between signup and experiment (0..floor(max_days/30))."""
    days = rng.integers(0, max_days + 1, size=size)
    return (days // 30).astype(np.int16)
