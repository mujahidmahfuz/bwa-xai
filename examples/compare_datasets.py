"""Compare aggregation methods across datasets, with paired significance tests.

Run:  python examples/compare_datasets.py

The same experiment at full size, with CSV output, is one shell command:
    bwa-bench compare --datasets synthetic breast_cancer wine --n-trials 100
"""

import logging

from bwa_xai.benchmark import make_task, paired_wilcoxon, run_comparison, summarize
from bwa_xai.reporting import summary_table

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("compare")

DATASETS = ("synthetic", "breast_cancer", "wine")
N_SAMPLES = 20
N_TRIALS = 30
SIGNIFICANCE = 0.05


def main() -> None:
    records = []
    for name in DATASETS:
        LOG.info("running %s ...", name)
        # Each trial draws N_SAMPLES training points, bootstraps them, and
        # scores every method on the same replicates against a reference:
        # the true coefficients (synthetic) or the full-data fit (real data).
        records += run_comparison(
            make_task(name, N_SAMPLES), n_trials=N_TRIALS, n_boot=30, seed=0, labels={"dataset": name}
        )

    LOG.info("\n%s", summary_table(summarize(records)))

    LOG.info("Significant differences from the vanilla mean (Bonferroni-corrected):")
    for test in paired_wilcoxon(records, reference="vanilla_mean"):
        if test["p_bonferroni"] < SIGNIFICANCE:
            LOG.info(
                "  %-14s %-18s %-14s diff=%+.3f  p=%.2g",
                test["dataset"], test["method"], test["metric"], test["mean_diff"], test["p_bonferroni"],
            )


if __name__ == "__main__":
    main()
