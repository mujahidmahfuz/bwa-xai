"""Simplex projection (paper Eq. 1) and the correlation cost matrix."""

from __future__ import annotations

import numpy as np

from ._validation import as_replicates

DEFAULT_SHRINKAGE = 0.1


def to_simplex(replicates: object) -> np.ndarray:
    """Project each replicate to a probability vector ``|e| / ||e||_1``.

    The projection discards sign and global scale. An all-zero replicate has
    no direction, so it maps to the uniform measure.
    """
    magnitudes = np.abs(as_replicates(replicates))
    totals = magnitudes.sum(axis=1, keepdims=True)
    uniform = np.full_like(magnitudes, 1.0 / magnitudes.shape[1])
    return np.divide(magnitudes, totals, out=uniform, where=totals > 0)


def _safe_correlation(features: np.ndarray) -> np.ndarray:
    """Pearson correlation with NaN (constant columns) replaced by 0."""
    centered = features - features.mean(axis=0)
    scale = np.sqrt((centered**2).sum(axis=0))
    normalized = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)
    return np.clip(normalized.T @ normalized, -1.0, 1.0)


def correlation_cost(features: object, shrinkage: float = DEFAULT_SHRINKAGE) -> np.ndarray:
    """Cost matrix ``C = 1 - |rho'|`` with ``rho' = (1 - s) * rho + s * I``.

    Strongly correlated features are cheap to move mass between. Constant
    columns have undefined correlation; they are treated as uncorrelated with
    everything (the Ionosphere fix described in the paper). The diagonal is
    set to exactly zero for every feature, constant ones included, so that
    leaving mass where it is never costs anything.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("features must be 2-D with at least two rows")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1]; got {shrinkage}")
    rho = (1.0 - shrinkage) * _safe_correlation(x)
    cost = 1.0 - np.abs(rho)
    np.fill_diagonal(cost, 0.0)
    return np.clip(cost, 0.0, 1.0)
