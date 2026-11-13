"""Per-feature sign estimators: the v1 binomial test and the v2 soft sign."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ._validation import as_replicates

DEFAULT_ALPHA = 0.05


def soft_sign(replicates: object) -> np.ndarray:
    """Magnitude-weighted soft sign ``s_j = 2 * p_j - 1`` in ``[-1, 1]`` (Eq. 6).

    ``p_j`` is the share of feature ``j``'s total absolute attribution that
    comes from positive replicates. ``|s_j|`` is the per-feature confidence
    signal: 1 means every replicate agrees on the direction, 0 means the
    positive and negative mass cancel. A feature that is zero in every
    replicate gets ``s_j = 0``.
    """
    e = as_replicates(replicates)
    magnitudes = np.abs(e)
    total = magnitudes.sum(axis=0)
    positive = np.where(e > 0, magnitudes, 0.0).sum(axis=0)
    p_hat = np.divide(positive, total, out=np.full(e.shape[1], 0.5), where=total > 0)
    return 2.0 * p_hat - 1.0


def binomial_sign(replicates: object, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Hard sign with abstention from a two-sided binomial test (Eq. 4).

    Returns +1 / -1 where the fraction of positive replicates differs
    significantly from 0.5, and 0 (abstain) otherwise.

    Replicates that are exactly zero carry no sign, so they are left out of
    the count (the usual sign-test treatment of ties); a feature that is zero
    in every replicate abstains. For attributions without exact zeros this
    is identical to Eq. 4.

    The test assumes independent replicate signs. Bootstrap replicates share
    about 63% of their support, so the test is anti-conservative in practice
    (paper Section 6.1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    e = as_replicates(replicates)
    n_signed = (e != 0).sum(axis=0)
    n_positive = (e > 0).sum(axis=0)
    p_hat = np.divide(n_positive, n_signed, out=np.full(e.shape[1], 0.5), where=n_signed > 0)
    deviation = p_hat - 0.5
    threshold = norm.ppf(1.0 - alpha / 2.0) * np.sqrt(0.25 / np.maximum(n_signed, 1))
    return np.where(np.abs(deviation) > threshold, np.sign(deviation), 0.0)
