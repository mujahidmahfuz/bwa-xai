"""Attribution quality metrics against a ground-truth (or proxy) vector."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ._validation import as_pair, as_vector


def top_k(values: object, k: int) -> np.ndarray:
    """Indices of the ``k`` largest-magnitude entries (ties broken by index).

    A vector with fewer than ``k`` non-zero entries (BWA-v1 after abstention)
    is padded with its lowest-index zeros, which makes top-``k`` overlaps of
    such vectors partly an artefact of feature order.
    """
    vec = as_vector(values, "values")
    if not 1 <= k <= vec.shape[0]:
        raise ValueError(f"k must be in [1, {vec.shape[0]}]; got {k}")
    return np.argsort(-np.abs(vec), kind="stable")[:k]


def sign_accuracy(estimate: object, truth: object) -> float:
    """Fraction of truly non-zero features whose sign is recovered.

    An estimate of exactly zero on a non-zero feature (an abstention) counts
    as wrong, which is what penalises BWA-v1's hard sign test.
    """
    est, ref = as_pair(estimate, truth)
    active = ref != 0
    if not active.any():
        raise ValueError("truth has no non-zero features")
    return float(np.mean(np.sign(est[active]) == np.sign(ref[active])))


def jaccard_at_k(estimate: object, truth: object, k: int) -> float:
    """Jaccard index ``|A & B| / |A | B|`` of the two top-``k`` feature sets."""
    est, ref = as_pair(estimate, truth)
    selected = set(top_k(est, k).tolist())
    relevant = set(top_k(ref, k).tolist())
    return len(selected & relevant) / len(selected | relevant)


def spearman_abs(estimate: object, truth: object) -> float:
    """Spearman rank correlation of ``|estimate|`` and ``|truth|``.

    Returns NaN when either vector is constant in magnitude.
    """
    est, ref = as_pair(estimate, truth)
    abs_est, abs_ref = np.abs(est), np.abs(ref)
    if np.ptp(abs_est) == 0 or np.ptp(abs_ref) == 0:
        return float("nan")
    return float(spearmanr(abs_est, abs_ref).statistic)


def gini(values: object) -> float:
    """Gini coefficient of ``|values|``: 0 is uniform, near 1 is concentrated."""
    magnitudes = np.sort(np.abs(as_vector(values, "values")))
    total = magnitudes.sum()
    n = magnitudes.shape[0]
    if total == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2.0 * (ranks * magnitudes).sum()) / (n * total) - (n + 1.0) / n)


def l2_norm(values: object) -> float:
    """Euclidean norm of the attribution vector."""
    return float(np.linalg.norm(as_vector(values, "values")))


def evaluate(estimate: object, truth: object, k: int) -> dict[str, float]:
    """All quality metrics for one estimate."""
    return {
        "sign_accuracy": sign_accuracy(estimate, truth),
        "jaccard": jaccard_at_k(estimate, truth, k),
        "spearman": spearman_abs(estimate, truth),
        "gini": gini(estimate),
        "norm": l2_norm(estimate),
    }


QUALITY_METRICS = ("sign_accuracy", "jaccard", "spearman", "gini", "norm")
