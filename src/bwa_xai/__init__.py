"""Bootstrap Wasserstein Alignment: stable feature attribution in low-data regimes.

Reference: M. Islam and P. Dubey, "Bootstrap Wasserstein alignment for stable
feature attribution in low-data regimes", Knowledge-Based Systems 353 (2026)
117182. https://doi.org/10.1016/j.knosys.2026.117182
"""

from .aggregators import (
    AGGREGATORS,
    bootstrap_median,
    bootstrapped_shap,
    bwa_v1,
    bwa_v2,
    quantile_midpoint,
    sign_scale_estimator,
    vanilla_mean,
)
from .barycenter import BarycenterError, wasserstein_barycenter
from .bootstrap import bootstrap_attributions
from .core import BWAResult, aggregate, stable_attribution
from .explainers import coefficient_explainer, linear_shap_explainer, make_shap_explainer
from .metrics import evaluate, gini, jaccard_at_k, sign_accuracy, spearman_abs
from .projection import correlation_cost, to_simplex
from .signs import binomial_sign, soft_sign
from .stability import stability_metrics
from .uncertainty import sigma_uq, soft_sign_calibration

__version__ = "0.1.0"

__all__ = [
    "AGGREGATORS",
    "BWAResult",
    "BarycenterError",
    "aggregate",
    "binomial_sign",
    "bootstrap_attributions",
    "bootstrap_median",
    "bootstrapped_shap",
    "bwa_v1",
    "bwa_v2",
    "coefficient_explainer",
    "correlation_cost",
    "evaluate",
    "gini",
    "jaccard_at_k",
    "linear_shap_explainer",
    "make_shap_explainer",
    "quantile_midpoint",
    "sigma_uq",
    "sign_accuracy",
    "sign_scale_estimator",
    "soft_sign",
    "soft_sign_calibration",
    "spearman_abs",
    "stability_metrics",
    "stable_attribution",
    "to_simplex",
    "vanilla_mean",
    "wasserstein_barycenter",
]
