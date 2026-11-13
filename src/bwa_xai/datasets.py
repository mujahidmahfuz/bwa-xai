"""Benchmark datasets: a synthetic generator and standardised real datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_wine

SYNTHETIC_COEFFICIENTS = (-2.0, -1.0, 1.0, 2.0)
MIN_PER_CLASS = 2
MAX_LABEL_ATTEMPTS = 100
DIGITS_TARGET = 5


@dataclass(frozen=True)
class Dataset:
    """A binary-classification dataset with standardised features."""

    name: str
    X: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SyntheticData:
    """One draw of the synthetic benchmark with its ground-truth coefficients."""

    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray


def standardize(features: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance columns; constant columns become all zeros."""
    centered = features - features.mean(axis=0)
    scale = features.std(axis=0)
    return np.divide(centered, scale, out=np.zeros_like(centered, dtype=float), where=scale > 0)


def _block_correlated_features(
    n_samples: int, n_features: int, rho: float, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    n_blocks = -(-n_features // block_size)
    shared = np.repeat(rng.standard_normal((n_samples, n_blocks)), block_size, axis=1)
    noise = rng.standard_normal((n_samples, n_features))
    return np.sqrt(rho) * shared[:, :n_features] + np.sqrt(1.0 - rho) * noise


def make_synthetic(
    n_samples: int = 20,
    n_features: int = 100,
    n_active: int = 10,
    rho: float = 0.8,
    block_size: int = 10,
    rng: np.random.Generator | int | None = None,
) -> SyntheticData:
    """Sparse logistic data with block-correlated features (paper Section 5.1).

    ``n_active`` coefficients are drawn from {-2, -1, 1, 2}; the rest are
    zero. Labels are redrawn until each class has at least two samples.
    """
    if not 1 <= n_active <= n_features:
        raise ValueError("n_active must be between 1 and n_features")
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1); got {rho}")
    if n_samples < 2 * MIN_PER_CLASS:
        raise ValueError(f"n_samples must be at least {2 * MIN_PER_CLASS}")
    generator = np.random.default_rng(rng)
    beta = np.zeros(n_features)
    active = generator.choice(n_features, size=n_active, replace=False)
    beta[active] = generator.choice(SYNTHETIC_COEFFICIENTS, size=n_active)
    for _ in range(MAX_LABEL_ATTEMPTS):
        X = _block_correlated_features(n_samples, n_features, rho, block_size, generator)
        probabilities = 1.0 / (1.0 + np.exp(-X @ beta))
        y = (generator.random(n_samples) < probabilities).astype(int)
        if min(y.sum(), n_samples - y.sum()) >= MIN_PER_CLASS:
            return SyntheticData(X=X, y=y, beta=beta)
    raise RuntimeError("could not draw synthetic labels with both classes present")


def _from_sklearn(name: str, loader: Callable, positive: int) -> Dataset:
    bunch = loader()
    return Dataset(name, standardize(bunch.data.astype(float)), (bunch.target == positive).astype(int))


def _from_openml(name: str, openml_name: str, positive: str, top_variance: int | None = None) -> Dataset:
    bunch = fetch_openml(name=openml_name, version=1, as_frame=False, parser="auto")
    features = np.asarray(bunch.data, dtype=float)
    if top_variance is not None:
        keep = np.sort(np.argsort(-features.var(axis=0), kind="stable")[:top_variance])
        features = features[:, keep]
    return Dataset(name, standardize(features), (bunch.target == positive).astype(int))


LOADERS: Mapping[str, Callable[[], Dataset]] = {
    "breast_cancer": lambda: _from_sklearn("breast_cancer", load_breast_cancer, positive=1),
    "wine": lambda: _from_sklearn("wine", load_wine, positive=0),
    "digits": lambda: _from_sklearn("digits", load_digits, positive=DIGITS_TARGET),
    "ionosphere": lambda: _from_openml("ionosphere", "ionosphere", positive="g"),
    "sonar": lambda: _from_openml("sonar", "sonar", positive="Mine"),
    "leukemia_d500": lambda: _from_openml("leukemia_d500", "leukemia", "AML", top_variance=500),
    "leukemia_d2000": lambda: _from_openml("leukemia_d2000", "leukemia", "AML", top_variance=2000),
}

OFFLINE_DATASETS = ("breast_cancer", "wine", "digits")


def load_dataset(name: str) -> Dataset:
    """Load a named dataset with standardised features and 0/1 labels.

    ``breast_cancer``, ``wine`` (cultivar 1 vs. rest) and ``digits`` (5 vs.
    rest) ship with scikit-learn. ``ionosphere``, ``sonar`` and the Golub
    ``leukemia_d500`` / ``leukemia_d2000`` (top-variance probes) are
    downloaded from OpenML on first use and cached by scikit-learn.
    """
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name]()
