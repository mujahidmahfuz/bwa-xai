"""Run-to-run stability of an aggregator's own output (paper Section 5.6).

Given the consensus attributions from ``R`` independent repetitions of the
whole bootstrap-and-aggregate procedure on the same data, these metrics
measure how much the answer moves between repetitions. No ground truth is
needed.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from ._validation import as_replicates
from .metrics import jaccard_at_k, spearman_abs

STABILITY_METRICS = ("cosine", "norm_cv", "spearman", "jaccard")
DEFAULT_STABILITY_K = 10


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / denominator) if denominator > 0 else float("nan")


def _mean_pairwise(runs: np.ndarray, similarity: Callable[[np.ndarray, np.ndarray], float]) -> float:
    values = [similarity(runs[i], runs[j]) for i, j in combinations(range(len(runs)), 2)]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def norm_cv(runs: object) -> float:
    """Coefficient of variation (std / mean) of the L2 norm across runs."""
    norms = np.linalg.norm(as_replicates(runs), axis=1)
    mean = norms.mean()
    return float(norms.std() / mean) if mean > 0 else float("nan")


def stability_metrics(runs: object, k: int = DEFAULT_STABILITY_K) -> dict[str, float]:
    """Stability of ``runs`` (shape ``(R, d)``), one row per independent run.

    Returns the mean pairwise cosine similarity, the CV of the norm (lower is
    more stable), the mean pairwise Spearman correlation of ``|e|``, and the
    mean pairwise top-``k`` Jaccard overlap. Pairs with an undefined value
    (for example a zero vector) are skipped.
    """
    r = as_replicates(runs)
    k = min(k, r.shape[1])
    return {
        "cosine": _mean_pairwise(r, _cosine),
        "norm_cv": norm_cv(r),
        "spearman": _mean_pairwise(r, spearman_abs),
        "jaccard": _mean_pairwise(r, lambda u, v: jaccard_at_k(u, v, k)),
    }
