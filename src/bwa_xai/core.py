"""High-level API: aggregate replicates and report per-feature reliability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import as_replicates
from .aggregators import bwa_v1_from_barycenter, bwa_v2
from .barycenter import DEFAULT_REG, wasserstein_barycenter
from .bootstrap import DEFAULT_N_BOOT, ModelFactory, bootstrap_attributions, default_model_factory
from .explainers import Explainer, coefficient_explainer
from .projection import DEFAULT_SHRINKAGE, correlation_cost, to_simplex
from .signs import DEFAULT_ALPHA, soft_sign
from .uncertainty import feature_std

METHODS = ("v1", "v2")
DEFAULT_RELIABILITY_THRESHOLD = 0.95


@dataclass(frozen=True)
class BWAResult:
    """Consensus attribution with its reliability signals.

    Attributes:
        attribution: consensus vector ``e*`` of shape ``(d,)``.
        soft_sign: ``s_j`` in ``[-1, 1]``; ``|s_j|`` says how stable feature
            ``j``'s direction is under refitting (not whether it is real).
        feature_std: per-feature standard deviation across replicates.
        sigma_uq: mean of ``feature_std``; a property of the bootstrap, the
            same for every aggregator.
        method: ``"v1"`` or ``"v2"``.
        barycenter: Wasserstein barycenter on the simplex, when computed.
        barycenter_converged: whether Sinkhorn reached tolerance.
    """

    attribution: np.ndarray
    soft_sign: np.ndarray
    feature_std: np.ndarray
    sigma_uq: float
    method: str
    barycenter: np.ndarray | None = None
    barycenter_converged: bool | None = None

    def reliable(self, threshold: float = DEFAULT_RELIABILITY_THRESHOLD) -> np.ndarray:
        """Boolean mask of features with ``|soft_sign| >= threshold``."""
        return np.abs(self.soft_sign) >= threshold

    def top_features(self, k: int) -> np.ndarray:
        """Indices of the ``k`` largest-magnitude attributions."""
        return np.argsort(-np.abs(self.attribution), kind="stable")[:k]


def _resolve_cost(
    n_features: int, X: object | None, cost: object | None, shrinkage: float
) -> np.ndarray:
    if cost is not None:
        return np.asarray(cost, dtype=float)
    if X is None:
        raise ValueError("the Wasserstein barycenter needs either a cost matrix or the features X")
    features = np.asarray(X, dtype=float)
    if features.ndim != 2 or features.shape[1] != n_features:
        raise ValueError(f"X must have {n_features} columns to match the replicates")
    return correlation_cost(features, shrinkage)


def aggregate(
    replicates: object,
    method: str = "v2",
    *,
    X: object | None = None,
    cost: object | None = None,
    reg: float = DEFAULT_REG,
    alpha: float = DEFAULT_ALPHA,
    shrinkage: float = DEFAULT_SHRINKAGE,
    diagnostic_barycenter: bool = False,
) -> BWAResult:
    """Aggregate bootstrap-replicate attributions of shape ``(B, d)``.

    ``method="v2"`` (recommended) needs only the replicates. ``method="v1"``
    needs a cost matrix, given directly as ``cost`` or built from the
    feature matrix ``X`` as ``1 - |shrunk correlation|``. With
    ``diagnostic_barycenter=True``, v2 also computes the barycenter; it does
    not change the v2 attribution.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    e = as_replicates(replicates)
    barycenter = None
    if method == "v1" or diagnostic_barycenter:
        ground_cost = _resolve_cost(e.shape[1], X, cost, shrinkage)
        barycenter = wasserstein_barycenter(to_simplex(e), ground_cost, reg=reg)
    if method == "v1":
        attribution = bwa_v1_from_barycenter(e, barycenter.measure, alpha)
    else:
        attribution = bwa_v2(e)
    spread = feature_std(e)
    return BWAResult(
        attribution=attribution,
        soft_sign=soft_sign(e),
        feature_std=spread,
        sigma_uq=float(spread.mean()),
        method=method,
        barycenter=None if barycenter is None else barycenter.measure,
        barycenter_converged=None if barycenter is None else barycenter.converged,
    )


def stable_attribution(
    X: object,
    y: object,
    method: str = "v2",
    *,
    x: object | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    model_factory: ModelFactory = default_model_factory,
    explainer: Explainer = coefficient_explainer,
    seed: np.random.Generator | int | None = None,
    **aggregate_kwargs: object,
) -> BWAResult:
    """Bootstrap, retrain, explain and aggregate in one call.

    Extra keyword arguments are forwarded to :func:`aggregate`.
    """
    replicates = bootstrap_attributions(
        X, y, n_boot=n_boot, model_factory=model_factory, explainer=explainer, x=x, rng=seed
    )
    return aggregate(replicates, method, X=X, **aggregate_kwargs)
