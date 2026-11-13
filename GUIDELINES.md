# Guidelines: using, publishing and sharing bwa-xai

This guide covers five things:

1. [Using the package yourself](#1-using-the-package-yourself)
2. [Use cases, and when not to use it](#2-use-cases)
3. [Putting it on GitHub](#3-putting-it-on-github)
4. [Sharing it with others](#4-sharing-it-with-others)
5. [How others use it](#5-how-others-use-it)

`README.md` is the reference (API summary, results, limitations). This file is the how-to.

---

## 1. Using the package yourself

### Install

From the project folder:

```bash
pip install -e .            # the package and its core dependencies
pip install -e ".[dev]"     # adds pytest and coverage
pip install -e ".[shap]"    # adds the shap library, only if you want SHAP explainers
```

`-e` installs in editable mode, so changes to the source take effect without reinstalling.

Check that it works:

```bash
pytest
python examples/quickstart.py
```

### The three ways to use it

**A. One call, from data to attribution.** Use this when a logistic regression is an acceptable model.

```python
from bwa_xai import stable_attribution

result = stable_attribution(X, y, method="v2", n_boot=50, seed=0)
```

`X` is a numeric array of shape `(N, d)`, ideally standardised; `y` holds binary labels.

**B. Your own model and explainer.** Give a function that builds a fresh model, and an explainer that turns a fitted model into an attribution vector.

```python
from sklearn.linear_model import LogisticRegression
from bwa_xai import aggregate, bootstrap_attributions, linear_shap_explainer

replicates = bootstrap_attributions(
    X, y,
    n_boot=50,
    model_factory=lambda: LogisticRegression(C=0.5, max_iter=1000),
    explainer=linear_shap_explainer,   # or make_shap_explainer(), or your own function
    x=X[0],                            # the instance to explain; omit for global explainers
    rng=0,
)
result = aggregate(replicates, method="v2")
```

An explainer is any function `explainer(model, X_train, x) -> array of shape (d,)`. For regression targets, pass `ensure_all_classes=False`.

**C. You already have the replicates.** If another pipeline produced `B` attribution vectors (for example from a deep network you retrained yourself), stack them into an array of shape `(B, d)` and aggregate:

```python
from bwa_xai import aggregate

result = aggregate(replicates)          # BWA-v2
```

### Reading the result

| Field | Meaning | How to use it |
|---|---|---|
| `result.attribution` | consensus attribution per feature | rank features, plot, report |
| `result.soft_sign` | `s_j` in [-1, 1] | `\|s_j\|` near 1: the direction is stable under refitting; near 0: it flips |
| `result.reliable(0.95)` | mask of features with `\|s_j\| >= 0.95` | report only these directions, or flag the rest |
| `result.feature_std` | bootstrap standard deviation per feature | error bars |
| `result.sigma_uq` | mean of `feature_std` | one-number summary of how noisy the explanation is |
| `result.top_features(k)` | indices of the `k` largest attributions | shortlist |

Read `|s_j|` as "how stable is this direction if I refit", never as "is this feature real". On the synthetic benchmark about 9% of signs are wrong even at `|s_j| >= 0.95`.

### Choosing settings

- **Method:** use `"v2"` unless you have a cost matrix that encodes real domain knowledge (for example gene pathways). `"v1"` zeroes features it is unsure about, loses 4 to 11 points of sign accuracy in our benchmarks, and shrinks the scale of the attributions.
- **`n_boot`:** 50 is a good default. Go to 75 or more when you need tighter reliability estimates. Cost grows linearly: each replicate is one model fit plus one explanation.
- **`seed`:** always set it, and record it, so results can be reproduced.
- **BWA-v1 only:** pass `X=` to build the cost matrix from feature correlations, or `cost=` for your own. Keep `reg` at 0.05 or above. Check `result.barycenter_converged`.

### Running the benchmarks

```bash
bwa-bench compare   --datasets synthetic breast_cancer wine --n-samples 20 50 --n-trials 100
bwa-bench stability --datasets synthetic sonar --n-samples 20 --n-datasets 20 --n-runs 8
bwa-bench compare --help      # all options
```

Results go to `results/` (change with `--out`): per-trial records, a summary CSV, Wilcoxon tests, and a Markdown table. The datasets `ionosphere`, `sonar`, `leukemia_d500` and `leukemia_d2000` are downloaded from OpenML on first use, so they need an internet connection once.

To benchmark your own dataset from Python:

```python
from bwa_xai.benchmark import run_comparison, subsample_task, summarize
from bwa_xai.datasets import Dataset, standardize
from bwa_xai.reporting import summary_table

dataset = Dataset("mine", standardize(X_full), y_full)
records = run_comparison(subsample_task(dataset, n_samples=20), n_trials=100, labels={"dataset": "mine"})
table = summary_table(summarize(records))
```

---

## 2. Use cases

### Where it fits

| Situation | What the package gives you |
|---|---|
| **Small clinical or biomedical cohorts** (tens of patients, hundreds or thousands of measurements, such as gene expression) | which biomarkers have a direction that survives refitting, and which do not |
| **Explaining a model before acting on it** (risk scoring, decision support) | an attribution with a per-feature reliability flag, instead of a bare point estimate |
| **Image or signal saliency from few training examples** | sparser maps than plain averaging, at the natural attribution scale |
| **Auditing an existing explanation** | `sigma_uq` and `soft_sign` show how much of a SHAP or coefficient explanation is sampling noise |
| **Research on attribution methods** | a ready benchmark: datasets, metrics, stability measurement, significance tests, and baselines to compare a new aggregator against |
| **Teaching** | a small, tested codebase showing why explanations are unstable when `N << d` |

To put your own aggregator through the benchmark, register a function `(replicates, features) -> vector` and name it in `methods`:

```python
import numpy as np
from bwa_xai.aggregators import AGGREGATORS
from bwa_xai.benchmark import make_task, run_comparison

AGGREGATORS["my_method"] = lambda replicates, features: np.median(replicates, axis=0)

records = run_comparison(make_task("synthetic", 20), methods=("vanilla_mean", "bwa_v2", "my_method"))
```

### When not to use it

- **Plenty of data.** If `N` is large compared with `d`, attributions are already stable and the bootstrap only costs time.
- **You need better feature recovery.** BWA-v2 does not find the right top-k features more often than the plain mean. Its gain is sparsity (about 3 to 6% higher Gini), not accuracy.
- **You need maximum run-to-run stability.** The plain bootstrap mean is slightly more stable. Use `vanilla_mean`.
- **Dense tabular data where every feature matters.** BWA-v2 only ties the mean there on accuracy; BWA-v1 does worse.
- **Models that are expensive to retrain.** The method needs `B` full retrains.
- **As proof that a feature is causal or real.** It measures stability of an explanation, nothing more.

---

## 3. Putting it on GitHub

### Before the first commit

1. **License.** The project is MIT-licensed; the `LICENSE` file is in the project root, with the copyright held by the paper's two authors. Change the holder line there if your institution requires a different one.
2. **Keep the article PDF out.** `.gitignore` already excludes `*.pdf`. The published PDF is the publisher's copyright; link to the DOI instead. Whether you may share an accepted manuscript depends on the journal's policy, so check it before uploading any version.
3. **Check for anything private.** The repository contains no credentials. The author names in `pyproject.toml` and `README.md` are already public in the paper.
4. **Decide whether to commit `results/`.** It is about 1 MB of CSVs and lets readers see the numbers without rerunning. Recommended: commit it.

### Create the repository and push

Install Git if needed (https://git-scm.com). Then, in the project folder:

```bash
git init
git add .
git status                      # confirm the PDF is NOT listed
git commit -m "feat: initial release of bwa-xai"
git branch -M main
```

Create an empty repository on GitHub: go to https://github.com/new, name it `bwa-xai`, choose Public or Private, and do **not** tick "Add a README" or "Add .gitignore" (the project already has them). Then connect and push:

```bash
git remote add origin https://github.com/mujahidmahfuz/bwa-xai.git
git push -u origin main
```

GitHub will ask you to sign in. If it asks for a password on the command line, use a personal access token (GitHub → Settings → Developer settings → Personal access tokens), not your account password.

If you install the GitHub CLI (https://cli.github.com), the last two steps become one:

```bash
gh auth login
gh repo create bwa-xai --public --source . --push
```

### After pushing

- **Add a description and topics** on the repository page (for example `explainable-ai`, `feature-attribution`, `optimal-transport`, `bootstrap`). This is how people find it.
- **Tag a release** so there is a fixed version to cite:

  ```bash
  git tag v0.1.0
  git push origin v0.1.0
  ```

  Then on GitHub: Releases → Draft a new release → choose the tag.
- **Later changes:** edit, then `git add -A`, `git commit -m "fix: ..."`, `git push`. Run `pytest` before each push.
- **Optional, automatic tests:** add a GitHub Actions workflow that runs `pip install -e ".[dev]"` and `pytest` on every push, so the repository shows a passing badge.

---

## 4. Sharing it with others

In order of effort:

**1. Share the GitHub link.** Anyone can install straight from it, with no further work from you:

```bash
pip install "git+https://github.com/mujahidmahfuz/bwa-xai.git"
```

This line is already in the README's Install section; it starts working once the repository exists and is public.

**2. Make it citable with a DOI.** Sign in to https://zenodo.org with your GitHub account, switch on the `bwa-xai` repository, then publish a GitHub release. Zenodo archives that release and gives it a DOI that you can put in the README and in papers. This is the standard way to make research software citable.

**3. Link it from the paper's side.** Add the repository URL to your institutional profile, ORCID, Google Scholar and ResearchGate entries for the paper, and to paperswithcode.com, which links code to papers.

**4. Publish on PyPI (optional).** This gives the shortest install command, `pip install bwa-xai`. The name was free when this guide was written; check again before publishing. You need a PyPI account and an API token.

```bash
pip install build twine
python -m build                 # creates dist/*.whl and dist/*.tar.gz
twine check dist/*
twine upload dist/*             # asks for your PyPI API token
```

A PyPI version number can never be reused, so bump `version` in `pyproject.toml` and `__version__` in `src/bwa_xai/__init__.py` for each upload. Do a trial run on https://test.pypi.org first (`twine upload --repository testpypi dist/*`).

**What to tell people when you share it.** Be accurate about what it does, since the README states it openly anyway:

- It attaches a reliability signal to feature attributions when data is scarce.
- BWA-v2 gives sparser attributions than plain averaging at the same sign accuracy.
- It does not improve top-k feature recovery, and the plain mean is slightly more stable.

---

## 5. How others use it

Send collaborators this section, or point them to the README.

### Install

```bash
pip install "git+https://github.com/mujahidmahfuz/bwa-xai.git"
```

or, to read and modify the code:

```bash
git clone https://github.com/mujahidmahfuz/bwa-xai.git
cd bwa-xai
pip install -e ".[dev]"
pytest
```

Requires Python 3.10 or newer.

### First steps

```bash
python examples/quickstart.py                    # a full attribution on 20 samples
python examples/compare_datasets.py              # methods compared on three datasets
python examples/stability_and_calibration.py     # stability and calibration
```

Then on their own data:

```python
from bwa_xai import stable_attribution

result = stable_attribution(X, y, method="v2", n_boot=50, seed=0)
```

### Reproducing the published numbers

The exact commands are in the README under "Results". With seed 0 they regenerate the files in `results/`. Small differences from the paper's tables are expected and are explained in the README under "Differences from the paper".

### Citing

Cite the paper (BibTeX is in the README). If you create a Zenodo DOI, ask users to cite the software version as well.

### Reporting problems and contributing

- **Bugs and questions:** open an issue on the GitHub repository, with the package version, a minimal example, and the full error message.
- **Changes:** fork the repository, create a branch, add a test for the change, make sure `pytest` passes, and open a pull request.
- **Code conventions:** PEP 8, type annotations on function signatures, no `print` in library code (use `logging`), functions under 50 lines, and a test for every new behaviour.

---

## Pre-publication checklist

- [x] `LICENSE` file added (MIT)
- [x] Repository URL filled in (`github.com/mujahidmahfuz/bwa-xai`)
- [x] Install-from-GitHub line added to the README
- [ ] `pytest` passes
- [ ] `git status` shows no PDF and nothing private
- [ ] Repository created on GitHub and first push done
- [ ] Release `v0.1.0` tagged
- [ ] (Optional) Zenodo DOI created and added to the README
- [ ] (Optional) PyPI upload done, after a trial on TestPyPI
