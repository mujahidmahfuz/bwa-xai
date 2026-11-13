"""Bootstrap resampling: retrain the model and re-explain on each resample."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .explainers import Explainer, coefficient_explainer

DEFAULT_N_BOOT = 50
MAX_RESAMPLE_ATTEMPTS = 100

ModelFactory = Callable[[], Any]


def default_model_factory() -> LogisticRegression:
    """L2-regularised logistic regression, the model used in the paper."""
    return LogisticRegression(C=1.0, max_iter=1000)


def bootstrap_indices(
    y: np.ndarray, rng: np.random.Generator, ensure_all_classes: bool = True
) -> np.ndarray:
    """One bootstrap resample of ``range(len(y))``, drawn with replacement.

    With ``ensure_all_classes`` the draw is repeated until every class in
    ``y`` appears, so that a classifier can be fitted. Set it to ``False``
    for regression targets.
    """
    n_samples = len(y)
    n_classes = len(np.unique(y))
    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        indices = rng.integers(0, n_samples, size=n_samples)
        if not ensure_all_classes or len(np.unique(y[indices])) == n_classes:
            return indices
    raise RuntimeError(
        f"could not draw a bootstrap sample containing all {n_classes} classes "
        f"in {MAX_RESAMPLE_ATTEMPTS} attempts; some class is too rare"
    )


def _check_inputs(X: np.ndarray, y: np.ndarray, n_boot: int) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("y must be 1-D with one entry per row of X")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    if n_boot < 2:
        raise ValueError(f"n_boot must be at least 2; got {n_boot}")


def bootstrap_attributions(
    X: object,
    y: object,
    n_boot: int = DEFAULT_N_BOOT,
    model_factory: ModelFactory = default_model_factory,
    explainer: Explainer = coefficient_explainer,
    x: object | None = None,
    rng: np.random.Generator | int | None = None,
    ensure_all_classes: bool = True,
) -> np.ndarray:
    """Attribution replicates of shape ``(n_boot, d)``.

    For each of ``n_boot`` bootstrap resamples of ``(X, y)`` a fresh model
    from ``model_factory`` is fitted and passed to ``explainer`` together with
    the resampled features and the instance ``x`` (``None`` for global
    explainers).
    """
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y)
    _check_inputs(features, targets, n_boot)
    generator = np.random.default_rng(rng)
    instance = None if x is None else np.asarray(x, dtype=float)
    replicates = []
    for _ in range(n_boot):
        indices = bootstrap_indices(targets, generator, ensure_all_classes)
        model = model_factory().fit(features[indices], targets[indices])
        replicates.append(np.asarray(explainer(model, features[indices], instance), dtype=float))
    return np.vstack(replicates)
