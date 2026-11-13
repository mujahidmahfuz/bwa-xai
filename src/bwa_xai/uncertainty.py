"""Bootstrap-level uncertainty and soft-sign calibration (paper Sections 5.8-5.9)."""

from __future__ import annotations

import numpy as np

from ._validation import as_pair, as_replicates

DEFAULT_CALIBRATION_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)


def feature_std(replicates: object) -> np.ndarray:
    """Per-feature standard deviation across replicates (population, ddof=0)."""
    return as_replicates(replicates).std(axis=0)


def sigma_uq(replicates: object) -> float:
    """Scalar uncertainty ``mean_j sqrt(Var_b(e_j))`` (Algorithm 1, step 6).

    This depends only on the replicate set, not on the aggregator, so every
    method run on the same bootstrap reports the same value (paper Remark 5).
    """
    return float(feature_std(replicates).mean())


def soft_sign_calibration(
    soft_signs: object,
    true_signs: object,
    bin_edges: tuple[float, ...] = DEFAULT_CALIBRATION_BINS,
) -> list[dict[str, float]]:
    """Sign accuracy binned by confidence ``|s_j|`` (paper Table 6).

    Features with a zero true sign are ignored. Each row reports the bin, the
    number of features in it, and the fraction whose soft-sign direction
    matches the truth (NaN for an empty bin). Pool ``soft_signs`` over many
    trials to get a usable curve.

    A soft sign of exactly 0 has no direction and counts as incorrect.
    Features whose ``|s_j|`` lies outside ``bin_edges`` are left out.
    """
    signs, truth = as_pair(soft_signs, true_signs)
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing with at least two entries")
    magnitude = np.abs(signs)
    active = (truth != 0) & (magnitude >= edges[0]) & (magnitude <= edges[-1])
    confidence = magnitude[active]
    correct = np.sign(signs[active]) == np.sign(truth[active])
    # The last bin is closed on the right so that |s| == 1 is counted.
    bin_index = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        in_bin = bin_index == i
        count = int(in_bin.sum())
        accuracy = float(correct[in_bin].mean()) if count else float("nan")
        rows.append({"low": float(edges[i]), "high": float(edges[i + 1]), "n": count, "accuracy": accuracy})
    return rows
