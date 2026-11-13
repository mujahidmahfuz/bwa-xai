"""Run-to-run stability of each method, and calibration of the soft sign.

Run:  python examples/stability_and_calibration.py
"""

import logging

import numpy as np

from bwa_xai import bootstrap_attributions, bwa_v2, soft_sign, soft_sign_calibration, stability_metrics
from bwa_xai.benchmark import make_task, run_stability, summarize
from bwa_xai.datasets import make_synthetic
from bwa_xai.reporting import summary_table

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("stability")

N_CALIBRATION_TRIALS = 100


def stability_by_hand() -> None:
    """Repeat the whole bootstrap-and-aggregate procedure and compare the outputs."""
    data = make_synthetic(rng=0)
    runs = np.vstack([
        bwa_v2(bootstrap_attributions(data.X, data.y, n_boot=50, rng=seed)) for seed in range(8)
    ])
    scores = stability_metrics(runs, k=10)
    LOG.info("BWA-v2 on one synthetic dataset, 8 independent runs:")
    LOG.info(
        "  cosine=%.3f  norm CV=%.3f  Spearman=%.3f  Jaccard@10=%.3f",
        scores["cosine"], scores["norm_cv"], scores["spearman"], scores["jaccard"],
    )


def stability_benchmark() -> None:
    """The same measurement for every method, averaged over dataset draws."""
    records = run_stability(
        make_task("synthetic", 20), n_datasets=10, n_runs=8, n_boot=50, labels={"dataset": "synthetic"}
    )
    LOG.info("\n%s", summary_table(summarize(records)))


def calibration_curve() -> None:
    """Does a larger |soft sign| mean the direction is more often right?"""
    signs, truths = [], []
    for seed in range(N_CALIBRATION_TRIALS):
        data = make_synthetic(rng=seed)
        signs.append(soft_sign(bootstrap_attributions(data.X, data.y, n_boot=50, rng=seed)))
        truths.append(data.beta)
    rows = soft_sign_calibration(np.concatenate(signs), np.concatenate(truths))
    LOG.info("Soft-sign calibration on the %d truly active features:", sum(row["n"] for row in rows))
    for row in rows:
        LOG.info("  |s| in [%.2f, %.2f]  n=%4d  P(correct sign)=%.3f", row["low"], row["high"], row["n"], row["accuracy"])


if __name__ == "__main__":
    stability_by_hand()
    stability_benchmark()
    calibration_curve()
