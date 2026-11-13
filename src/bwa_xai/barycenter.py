"""Entropic-regularised Wasserstein barycenter (paper Eqs. 2-3) via POT."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import ot

DEFAULT_REG = 0.05
DEFAULT_MAX_ITER = 1000
DEFAULT_TOL = 1e-5
SIMPLEX_ATOL = 1e-6


class BarycenterError(RuntimeError):
    """The Sinkhorn barycenter solver produced an unusable result."""


@dataclass(frozen=True)
class BarycenterResult:
    """A barycenter on the simplex and whether the solver reached ``tol``."""

    measure: np.ndarray
    converged: bool


def _validate(measures: np.ndarray, cost: np.ndarray, reg: float) -> None:
    if measures.ndim != 2:
        raise ValueError("measures must be 2-D with shape (n_measures, n_features)")
    n_features = measures.shape[1]
    if cost.shape != (n_features, n_features):
        raise ValueError(
            f"cost must have shape ({n_features}, {n_features}); got {cost.shape}"
        )
    if not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise ValueError("cost must be finite and non-negative")
    if np.any(measures < 0) or not np.allclose(measures.sum(axis=1), 1.0, atol=SIMPLEX_ATOL):
        raise ValueError("each measure must be non-negative and sum to 1")
    if reg <= 0:
        raise ValueError(f"reg must be positive; got {reg}")


def wasserstein_barycenter(
    measures: object,
    cost: object,
    reg: float = DEFAULT_REG,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    method: str = "sinkhorn",
) -> BarycenterResult:
    """Barycenter of ``measures`` (rows on the simplex) under ground cost ``cost``.

    ``reg`` below about 0.05 underflows the plain Sinkhorn kernel for costs in
    ``[0, 1]``; pass ``method="sinkhorn_log"`` to go lower. A run that stops at
    ``max_iter`` without reaching ``tol`` emits a ``RuntimeWarning`` and is
    flagged ``converged=False``; a non-finite result raises ``BarycenterError``.
    """
    a = np.asarray(measures, dtype=float)
    m = np.asarray(cost, dtype=float)
    _validate(a, m, reg)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        measure, log = ot.barycenter(
            a.T, m, reg, method=method, numItermax=max_iter, stopThr=tol, log=True, warn=False
        )
    measure = np.asarray(measure, dtype=float)
    if not np.all(np.isfinite(measure)) or measure.sum() <= 0:
        raise BarycenterError(
            f"Sinkhorn barycenter diverged at reg={reg}. Increase reg or use method='sinkhorn_log'."
        )
    errors = log.get("err", [])
    converged = bool(len(errors) > 0 and errors[-1] <= tol)
    if not converged:
        warnings.warn(
            f"Sinkhorn barycenter did not reach tol={tol} in {max_iter} iterations",
            RuntimeWarning,
            stacklevel=2,
        )
    return BarycenterResult(measure=measure / measure.sum(), converged=converged)
