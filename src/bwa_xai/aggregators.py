"""Aggregators that turn ``B`` bootstrap-replicate attributions into one vector.

Every aggregator takes replicates of shape ``(B, d)`` and returns a vector of
shape ``(d,)``. ``AGGREGATORS`` exposes them under a uniform
``(replicates, features)`` signature for benchmarking.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from ._validation import as_replicates
from .barycenter import DEFAULT_REG, wasserstein_barycenter
from .projection import DEFAULT_SHRINKAGE, correlation_cost, to_simplex
from .signs import DEFAULT_ALPHA, binomial_sign, soft_sign

SIGN_KINDS = ("soft", "hard")
SCALE_KINDS = ("median", "mean")
DEFAULT_QUANTILE_ALPHA = 0.1


def vanilla_mean(replicates: object) -> np.ndarray:
    """Euclidean mean of the replicates."""
    return as_replicates(replicates).mean(axis=0)


def bootstrap_median(replicates: object) -> np.ndarray:
    """Component-wise median of the replicates."""
    return np.median(as_replicates(replicates), axis=0)


def bootstrapped_shap(replicates: object) -> np.ndarray:
    """Mean absolute value carrying the majority sign of the replicates."""
    e = as_replicates(replicates)
    majority = np.sign(np.sign(e).sum(axis=0))
    return majority * np.abs(e).mean(axis=0)


def quantile_midpoint(replicates: object, alpha: float = DEFAULT_QUANTILE_ALPHA) -> np.ndarray:
    """Midpoint of the per-feature ``[alpha/2, 1 - alpha/2]`` quantile interval."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    lower, upper = np.quantile(as_replicates(replicates), [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    return (lower + upper) / 2.0


def sign_scale_estimator(
    replicates: object, sign: str = "soft", scale: str = "median"
) -> np.ndarray:
    """The 2x2 ablation family of paper Table 5: ``sign_j * scale_j``.

    ``soft`` x ``median`` is BWA-v2; ``soft`` x ``mean`` equals the vanilla
    mean; ``hard`` discretises the soft sign to -1 / 0 / +1.
    """
    if sign not in SIGN_KINDS:
        raise ValueError(f"sign must be one of {SIGN_KINDS}; got {sign!r}")
    if scale not in SCALE_KINDS:
        raise ValueError(f"scale must be one of {SCALE_KINDS}; got {scale!r}")
    e = as_replicates(replicates)
    signs = soft_sign(e)
    if sign == "hard":
        signs = np.sign(signs)
    magnitudes = np.abs(e)
    scales = np.median(magnitudes, axis=0) if scale == "median" else magnitudes.mean(axis=0)
    return signs * scales


def bwa_v2(replicates: object) -> np.ndarray:
    """BWA-v2 (Eqs. 6-7): soft sign times the per-feature median magnitude.

    By Proposition 2 this is a non-negative per-feature rescaling of the
    vanilla mean, so it has the same signs and does not depend on any cost
    matrix.
    """
    return sign_scale_estimator(replicates, sign="soft", scale="median")


def bwa_v1_from_barycenter(
    replicates: object, barycenter: object, alpha: float = DEFAULT_ALPHA
) -> np.ndarray:
    """BWA-v1 consensus (Eq. 5) given an already-computed barycenter."""
    e = as_replicates(replicates)
    mass = np.asarray(barycenter, dtype=float)
    if mass.shape != (e.shape[1],):
        raise ValueError(f"barycenter must have shape ({e.shape[1]},); got {mass.shape}")
    robust_scale = np.median(np.linalg.norm(e, axis=1))
    return binomial_sign(e, alpha) * mass * robust_scale


def bwa_v1(
    replicates: object,
    cost: object,
    reg: float = DEFAULT_REG,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """BWA-v1 (Algorithm 1): Wasserstein barycenter with binomial sign abstention."""
    e = as_replicates(replicates)
    result = wasserstein_barycenter(to_simplex(e), cost, reg=reg)
    return bwa_v1_from_barycenter(e, result.measure, alpha)


def _bwa_v1_from_features(replicates: np.ndarray, features: np.ndarray) -> np.ndarray:
    return bwa_v1(replicates, correlation_cost(features, DEFAULT_SHRINKAGE))


Aggregator = Callable[[np.ndarray, np.ndarray], np.ndarray]

AGGREGATORS: Mapping[str, Aggregator] = {
    "vanilla_mean": lambda replicates, features: vanilla_mean(replicates),
    "bootstrap_median": lambda replicates, features: bootstrap_median(replicates),
    "bootstrapped_shap": lambda replicates, features: bootstrapped_shap(replicates),
    "quantile_midpoint": lambda replicates, features: quantile_midpoint(replicates),
    "bwa_v1": _bwa_v1_from_features,
    "bwa_v2": lambda replicates, features: bwa_v2(replicates),
}

DEFAULT_METHODS = ("vanilla_mean", "bootstrap_median", "bootstrapped_shap", "bwa_v1", "bwa_v2")
