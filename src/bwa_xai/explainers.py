"""Explanation functions ``E(model, X_train, x) -> attribution vector``.

Anything with this signature can be plugged into
:func:`bwa_xai.bootstrap.bootstrap_attributions`.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np


class Explainer(Protocol):
    """Maps a fitted model (and optionally an instance ``x``) to attributions."""

    def __call__(self, model: Any, X_train: np.ndarray, x: np.ndarray | None) -> np.ndarray: ...


def _linear_coefficients(model: Any) -> np.ndarray:
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        raise TypeError(
            f"{type(model).__name__} has no coef_ attribute; use a linear model or another explainer"
        )
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim == 2 and coefficients.shape[0] != 1:
        raise ValueError("multi-class coefficients are not supported; use a binary or regression model")
    return coefficients.ravel()


def coefficient_explainer(model: Any, X_train: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    """Global attribution: the coefficients of a fitted linear model."""
    return _linear_coefficients(model)


def linear_shap_explainer(model: Any, X_train: np.ndarray, x: np.ndarray | None) -> np.ndarray:
    """Exact SHAP values of a linear model under feature independence.

    ``phi_j = coef_j * (x_j - mean(X_train)_j)``, in log-odds units for a
    logistic model. Needs the instance ``x``.
    """
    if x is None:
        raise ValueError("linear_shap_explainer needs the instance x to explain")
    return _linear_coefficients(model) * (np.asarray(x, dtype=float) - X_train.mean(axis=0))


def make_shap_explainer(**explainer_kwargs: Any) -> Callable[..., np.ndarray]:
    """Wrap the ``shap`` library (optional dependency) as an explainer.

    Builds ``shap.Explainer(model, X_train, **explainer_kwargs)`` for every
    bootstrap model. For models whose SHAP output has one column per class,
    the last class is returned.
    """
    try:
        import shap
    except ImportError as error:
        raise ImportError("install the optional dependency: pip install 'bwa-xai[shap]'") from error

    def explain(model: Any, X_train: np.ndarray, x: np.ndarray | None) -> np.ndarray:
        if x is None:
            raise ValueError("the SHAP explainer needs the instance x to explain")
        explainer = shap.Explainer(model, X_train, **explainer_kwargs)
        values = np.asarray(explainer(np.asarray(x, dtype=float)[None, :]).values[0], dtype=float)
        return values[:, -1] if values.ndim == 2 else values

    return explain
