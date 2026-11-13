"""Quickstart: a stable attribution with per-feature reliability, on 20 samples.

Run:  python examples/quickstart.py
"""

import logging

import numpy as np
from sklearn.datasets import load_breast_cancer

from bwa_xai import aggregate, bootstrap_attributions, linear_shap_explainer, stable_attribution
from bwa_xai.datasets import standardize

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("quickstart")

N_SAMPLES = 20
TOP_K = 5


def main() -> None:
    bunch = load_breast_cancer()
    rng = np.random.default_rng(0)
    rows = rng.choice(len(bunch.target), size=N_SAMPLES, replace=False)
    X, y = standardize(bunch.data[rows]), bunch.target[rows]

    # 1. One call: bootstrap the data, refit a logistic regression on each
    #    resample, take its coefficients, and aggregate them with BWA-v2.
    result = stable_attribution(X, y, method="v2", n_boot=50, seed=0)

    LOG.info("Top %d features by |attribution| (N=%d training samples):", TOP_K, N_SAMPLES)
    for j in result.top_features(TOP_K):
        LOG.info(
            "  %-24s attribution=%+.3f  soft sign=%+.2f  std=%.3f",
            bunch.feature_names[j], result.attribution[j], result.soft_sign[j], result.feature_std[j],
        )

    # 2. |soft sign| says how stable each direction is under refitting.
    #    It does not say the feature is real: the bootstrap cannot see past
    #    the one small sample it resamples.
    reliable = result.reliable(threshold=0.95)
    LOG.info("%d of %d features have a stable direction (|s| >= 0.95)", reliable.sum(), len(reliable))
    LOG.info("sigma_UQ (mean bootstrap std per feature) = %.3f", result.sigma_uq)

    # 3. Bring your own explainer. Here: local SHAP values of one instance.
    #    Any callable (model, X_train, x) -> vector works, e.g.
    #    bwa_xai.make_shap_explainer() for the shap library.
    replicates = bootstrap_attributions(X, y, n_boot=50, explainer=linear_shap_explainer, x=X[0], rng=0)
    local = aggregate(replicates, method="v2")
    LOG.info("Local explanation of sample 0, strongest feature: %s", bunch.feature_names[local.top_features(1)[0]])

    # 4. BWA-v1 uses the Wasserstein barycenter and needs a cost matrix,
    #    built here from the feature correlations of X.
    v1 = aggregate(replicates, method="v1", X=X)
    LOG.info(
        "BWA-v1 abstained on %d features; barycenter converged: %s",
        int(np.sum(v1.attribution == 0)), v1.barycenter_converged,
    )


if __name__ == "__main__":
    main()
