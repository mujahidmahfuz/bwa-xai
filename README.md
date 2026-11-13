# bwa-xai

Bootstrap Wasserstein Alignment (BWA): stable feature attributions with per-feature reliability, for models trained on very little data, plus a toolkit for evaluating attribution aggregators across datasets.

This is the reference implementation of

> M. Islam and P. Dubey, "Bootstrap Wasserstein alignment for stable feature attribution in low-data regimes", *Knowledge-Based Systems* 353 (2026) 117182. https://doi.org/10.1016/j.knosys.2026.117182

When `N << d`, retraining a model on a bootstrap resample can flip the sign and change the size of its feature attributions. BWA retrains on `B` resamples, explains each model, and aggregates the `B` attribution vectors into one consensus together with a signal that says how far each feature's direction can be trusted.

## Install

```bash
pip install "git+https://github.com/mujahidmahfuz/bwa-xai.git"
```

Or, from a clone of the repository:

```bash
pip install -e .            # core: numpy, scipy, scikit-learn, POT
pip install -e ".[shap]"    # optional: wrap the shap library as an explainer
pip install -e ".[dev]"     # pytest and coverage
```

Requires Python 3.10 or newer.

## Quickstart

```python
from bwa_xai import stable_attribution

result = stable_attribution(X, y, method="v2", n_boot=50, seed=0)

result.attribution      # consensus attribution, shape (d,)
result.soft_sign        # s_j in [-1, 1]; |s_j| = how stable the direction is
result.reliable(0.95)   # mask of features with |s_j| >= 0.95
result.sigma_uq         # mean bootstrap standard deviation per feature
```

By default this fits a logistic regression on each resample and uses its coefficients. To use your own model and explainer, pass `model_factory` and any `explainer(model, X_train, x) -> vector`:

```python
from bwa_xai import aggregate, bootstrap_attributions, make_shap_explainer

replicates = bootstrap_attributions(
    X, y, n_boot=50, model_factory=my_model, explainer=make_shap_explainer(), x=X[0]
)
result = aggregate(replicates, method="v2")
```

If you already have replicate attributions from elsewhere, `aggregate(replicates)` is all you need; it takes any array of shape `(B, d)`.

## The two methods

| | BWA-v2 (recommended) | BWA-v1 |
|---|---|---|
| Sign | soft sign `s_j`: magnitude-weighted share of positive replicates, rescaled to [-1, 1] | hard binomial test; abstains (zero) when not significant |
| Magnitude | per-feature median of `\|e_j\|` | Wasserstein-barycenter mass times the median replicate norm |
| Needs a cost matrix | no | yes (`X=` or `cost=`) |
| Keeps natural attribution scale | yes | no (norm shrinks several-fold) |

BWA-v2 is an exact, non-negative per-feature rescaling of the plain bootstrap mean (paper Proposition 2), so it has the same signs as the mean and does not depend on the cost matrix. What it changes is the magnitude: features whose sign is unstable are shrunk toward zero.

## Evaluation toolkit

```bash
bwa-bench compare   --datasets synthetic breast_cancer wine --n-samples 20 50 --n-trials 100
bwa-bench stability --datasets synthetic sonar --n-samples 20 --n-datasets 20 --n-runs 8
```

Each command writes per-trial records, a mean ± SEM summary, paired Wilcoxon tests against the vanilla mean (Bonferroni-corrected over the methods compared), and a Markdown table to `results/`.

- **Methods:** `vanilla_mean`, `bootstrap_median`, `bootstrapped_shap`, `quantile_midpoint`, `bwa_v1`, `bwa_v2`.
- **Datasets:** `synthetic` (sparse logistic, block-correlated), `breast_cancer`, `wine`, `digits` (bundled with scikit-learn); `ionosphere`, `sonar`, `leukemia_d500`, `leukemia_d2000` (downloaded from OpenML on first use).
- **Quality metrics** (against a reference attribution): sign accuracy, Jaccard@k, Spearman of magnitudes, Gini sparsity, L2 norm.
- **Stability metrics** (no reference needed): mean pairwise cosine, Spearman and top-k Jaccard across independent reruns, and the coefficient of variation of the norm.
- **Calibration:** `soft_sign_calibration` bins features by `|s_j|` and reports how often the direction is right.

The same experiments are available from Python (`bwa_xai.benchmark`), and you can benchmark your own data by writing a task: a function that takes a random generator and returns a `Trial(X, y, truth, k)`.

## Results

The files in `results/` were produced with seed 0 and `B = 50` by:

```bash
bwa-bench compare --datasets synthetic breast_cancer wine ionosphere sonar digits leukemia_d500 --n-samples 20 50 --n-trials 100 --n-boot 50
bwa-bench stability --datasets synthetic breast_cancer sonar leukemia_d500 --n-samples 20 --n-datasets 20 --n-runs 8 --n-boot 50
```

### Comparison at N = 20 (mean ± SEM over 100 trials)

| Dataset | d | Sign acc.: mean | Sign acc.: BWA-v1 | Sign acc.: BWA-v2 | Gini: mean | Gini: BWA-v1 | Gini: BWA-v2 |
|---|---|---|---|---|---|---|---|
| synthetic | 100 | 0.806 ± 0.010 | 0.728 ± 0.012 | 0.806 ± 0.010 | 0.415 ± 0.003 | 0.376 ± 0.004 | 0.439 ± 0.003 |
| breast_cancer | 30 | 0.783 ± 0.006 | 0.737 ± 0.006 | 0.783 ± 0.006 | 0.336 ± 0.004 | 0.307 ± 0.004 | 0.351 ± 0.004 |
| wine | 13 | 0.788 ± 0.008 | 0.749 ± 0.007 | 0.788 ± 0.008 | 0.368 ± 0.006 | 0.352 ± 0.007 | 0.383 ± 0.007 |
| ionosphere | 34 | 0.646 ± 0.007 | 0.558 ± 0.007 | 0.646 ± 0.007 | 0.430 ± 0.004 | 0.399 ± 0.005 | 0.454 ± 0.004 |
| sonar | 60 | 0.643 ± 0.006 | 0.550 ± 0.006 | 0.643 ± 0.006 | 0.392 ± 0.004 | 0.355 ± 0.005 | 0.416 ± 0.004 |
| digits (5 vs. rest) | 64 | 0.656 ± 0.004 | 0.551 ± 0.005 | 0.656 ± 0.004 | 0.517 ± 0.003 | 0.497 ± 0.004 | 0.529 ± 0.003 |
| leukemia | 500 | 0.786 ± 0.003 | 0.713 ± 0.002 | 0.786 ± 0.003 | 0.415 ± 0.002 | 0.376 ± 0.002 | 0.436 ± 0.002 |

What the numbers show, at both N = 20 and N = 50 (full tables in `results/compare_summary.md`):

- **Sign accuracy:** BWA-v2 equals the vanilla mean on every trial, as Proposition 2 requires. BWA-v1 is 4 to 11 points lower on every dataset, the cost of abstaining.
- **Sparsity:** BWA-v2 has the highest Gini of the three on every dataset (+0.012 to +0.025 over the mean, Bonferroni-corrected p < 1e-16 in all 14 cells). The effect is real but small, about 3 to 6% relative.
- **Top-k recovery:** Jaccard@k does not separate BWA-v2 from the mean in any cell (corrected p > 0.6). BWA-v2 concentrates mass; it does not find the right features more often.
- **Scale:** BWA-v2's norm stays within 4% of the mean's. BWA-v1's norm is 3 to 17 times smaller.

### Stability at N = 20 (20 dataset draws, 8 independent reruns each)

| Dataset | Method | Cosine | Spearman | Jaccard@k | Norm CV |
|---|---|---|---|---|---|
| synthetic | vanilla mean | 0.991 ± 0.001 | 0.963 ± 0.003 | 0.795 ± 0.016 | 0.019 ± 0.001 |
| synthetic | bootstrap median | 0.985 ± 0.001 | 0.942 ± 0.004 | 0.727 ± 0.017 | 0.020 ± 0.001 |
| synthetic | BWA-v2 | 0.988 ± 0.001 | 0.956 ± 0.003 | 0.735 ± 0.016 | 0.022 ± 0.001 |
| sonar | vanilla mean | 0.991 ± 0.001 | 0.957 ± 0.004 | 0.841 ± 0.011 | 0.017 ± 0.001 |
| sonar | bootstrap median | 0.984 ± 0.001 | 0.930 ± 0.006 | 0.795 ± 0.015 | 0.018 ± 0.001 |
| sonar | BWA-v2 | 0.987 ± 0.001 | 0.947 ± 0.004 | 0.804 ± 0.015 | 0.019 ± 0.001 |
| leukemia (d = 500) | vanilla mean | 0.994 ± 0.000 | 0.978 ± 0.002 | 0.881 ± 0.006 | 0.018 ± 0.001 |
| leukemia (d = 500) | bootstrap median | 0.989 ± 0.001 | 0.963 ± 0.003 | 0.849 ± 0.007 | 0.020 ± 0.001 |
| leukemia (d = 500) | BWA-v2 | 0.991 ± 0.001 | 0.971 ± 0.002 | 0.856 ± 0.007 | 0.021 ± 0.001 |

The vanilla mean is slightly but consistently more stable than BWA-v2: cosine and Spearman differ at corrected p < 0.001 on all four datasets tested (breast cancer is in `results/stability_summary.md`). BWA-v2 in turn scores above the bootstrap median on cosine and Spearman in every row. If run-to-run stability is the goal rather than sparsity, use the mean.

### Soft-sign calibration

On the synthetic benchmark (100 trials, the 1000 truly active features), the accuracy of the sign rises with `|s_j|`: 0.52 for `|s| < 0.2`, 0.78 for `0.6 <= |s| < 0.8`, and 0.91 for `|s| >= 0.95` (run `examples/stability_and_calibration.py`).

## Examples

| Script | Shows |
|---|---|
| `examples/quickstart.py` | one-call attribution on 20 samples, reading the reliability signals, a custom local explainer, BWA-v1 |
| `examples/compare_datasets.py` | comparing methods on three datasets with significance tests |
| `examples/stability_and_calibration.py` | measuring run-to-run stability and the soft-sign calibration curve |

## Limitations

- **`|s_j|` measures stability, not truth.** The bootstrap resamples one small dataset; a direction can be perfectly stable and still wrong. Even at `|s| >= 0.95` about 9% of signs are wrong on the synthetic benchmark.
- **The gain is sparsity only.** BWA-v2 does not improve sign accuracy or top-k recovery over the mean, and is slightly less stable.
- **Real-data references are proxies.** On real datasets the "truth" is the full-data model's coefficients, so those results check self-consistency, not ground truth.
- **BWA-v1's sign test is anti-conservative,** because bootstrap replicates overlap and are not independent. Its top-k overlaps are also partly an artefact of feature order when it abstains on more than `d - k` features.
- **Sinkhorn convergence.** With the paper's settings (`reg = 0.05`, 1000 iterations, tolerance 1e-5) the barycenter did not always reach tolerance on low-dimensional data (7 of 10 spot-check runs converged on wine, 9 of 10 on breast cancer). The result is still returned, with a `RuntimeWarning` and `barycenter_converged = False`. Below `reg = 0.05` plain Sinkhorn fails; `method="sinkhorn_log"` is available in `wasserstein_barycenter`.
- **Cost:** the barycenter is O(d²) memory and time per iteration. BWA-v2 does not need it.

## Differences from the paper

- The cost matrix always has a zero diagonal, including for constant features.
- The binomial sign test ignores replicates that are exactly zero and abstains on a feature that is zero everywhere. For attributions without exact zeros this is identical to Eq. 4.
- The synthetic benchmark builds its cost matrix from the sample correlation, not the ground-truth correlation, and uses correlated blocks of 10 features (the paper does not state the block size).
- MNIST with Integrated Gradients, SmoothGrad, and the pathway-cost experiment are not included.

The numbers above come from this implementation and differ slightly from the paper's tables (for example synthetic sign accuracy 80.6% here against 82.3% in Table 1), as expected from different random draws and the points above.

## Tests

```bash
pytest --cov=bwa_xai
```

104 tests, 99% line coverage. They include the paper's Propositions 2 to 4 as executable checks: BWA-v2 is a non-negative rescaling of the mean, never disagrees with it in sign, and does not depend on the cost matrix.

## Guide and license

`GUIDELINES.md` covers day-to-day use, use cases, and how to publish and share the package. The code is released under the MIT License (see `LICENSE`).

## Citation

```bibtex
@article{islam2026bwa,
  title   = {Bootstrap Wasserstein alignment for stable feature attribution in low-data regimes},
  author  = {Islam, Mujahidul and Dubey, Priya},
  journal = {Knowledge-Based Systems},
  volume  = {353},
  pages   = {117182},
  year    = {2026},
  doi     = {10.1016/j.knosys.2026.117182}
}
```
