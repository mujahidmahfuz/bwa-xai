"""Input validation shared by every public entry point."""

from __future__ import annotations

import numpy as np

MIN_REPLICATES = 2


def as_replicates(replicates: object) -> np.ndarray:
    """Return ``replicates`` as a finite float array of shape ``(B, d)``."""
    arr = np.asarray(replicates, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"replicates must be 2-D with shape (n_replicates, n_features); got ndim={arr.ndim}"
        )
    if arr.shape[0] < MIN_REPLICATES:
        raise ValueError(
            f"need at least {MIN_REPLICATES} bootstrap replicates; got {arr.shape[0]}"
        )
    if arr.shape[1] == 0:
        raise ValueError("replicates have zero features")
    if not np.all(np.isfinite(arr)):
        raise ValueError("replicates contain NaN or infinite values")
    return arr


def as_vector(values: object, name: str) -> np.ndarray:
    """Return ``values`` as a finite 1-D float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got ndim={arr.ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def as_pair(estimate: object, truth: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate an (estimate, ground truth) pair of equal-length vectors."""
    est = as_vector(estimate, "estimate")
    ref = as_vector(truth, "truth")
    if est.shape != ref.shape:
        raise ValueError(
            f"estimate and truth must have the same length; got {est.shape[0]} and {ref.shape[0]}"
        )
    return est, ref
