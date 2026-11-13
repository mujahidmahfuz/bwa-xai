"""Cross-dataset comparison and stability experiments (paper Sections 5.2-5.6)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from .aggregators import AGGREGATORS, DEFAULT_METHODS
from .bootstrap import ModelFactory, bootstrap_attributions, default_model_factory
from .datasets import Dataset, load_dataset, make_synthetic
from .metrics import QUALITY_METRICS, evaluate
from .stability import STABILITY_METRICS, stability_metrics

SYNTHETIC = "synthetic"
MIN_TOP_K = 3
NON_GROUP_KEYS = ("trial", "method")

Record = dict[str, object]


@dataclass(frozen=True)
class Trial:
    """One low-data problem: training data, reference attribution, top-k size."""

    X: np.ndarray
    y: np.ndarray
    truth: np.ndarray
    k: int


Task = Callable[[np.random.Generator], Trial]


def synthetic_task(n_samples: int = 20, **synthetic_kwargs: object) -> Task:
    """Each trial draws a fresh synthetic dataset; truth is the true ``beta``."""

    def draw(rng: np.random.Generator) -> Trial:
        data = make_synthetic(n_samples=n_samples, rng=rng, **synthetic_kwargs)
        return Trial(data.X, data.y, data.beta, k=int(np.count_nonzero(data.beta)))

    return draw


def subsample_task(
    dataset: Dataset,
    n_samples: int,
    k: int | None = None,
    model_factory: ModelFactory = default_model_factory,
) -> Task:
    """Each trial draws a class-stratified subset of ``dataset``.

    The reference attribution is the coefficient vector of the model fitted
    on the full dataset. It is a proxy, itself an estimate, so results are a
    self-consistency check rather than validation against a known truth.
    """
    if not 0 < n_samples < len(dataset.y):
        raise ValueError(f"n_samples must be in (0, {len(dataset.y)}); got {n_samples}")
    truth = np.asarray(model_factory().fit(dataset.X, dataset.y).coef_, dtype=float).ravel()
    top_k = k if k is not None else max(MIN_TOP_K, dataset.X.shape[1] // 3)

    def draw(rng: np.random.Generator) -> Trial:
        indices, _ = train_test_split(
            np.arange(len(dataset.y)),
            train_size=n_samples,
            stratify=dataset.y,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        return Trial(dataset.X[indices], dataset.y[indices], truth, top_k)

    return draw


def make_task(dataset_name: str, n_samples: int) -> Task:
    """Task for a dataset name: ``"synthetic"`` or any name in ``datasets.LOADERS``."""
    if dataset_name == SYNTHETIC:
        return synthetic_task(n_samples=n_samples)
    return subsample_task(load_dataset(dataset_name), n_samples)


def _check_methods(methods: Sequence[str]) -> None:
    unknown = [name for name in methods if name not in AGGREGATORS]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {sorted(AGGREGATORS)}")


def _trial_rngs(seed: int, count: int) -> list[np.random.Generator]:
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(count)]


def run_comparison(
    task: Task,
    n_trials: int = 100,
    n_boot: int = 50,
    methods: Sequence[str] = DEFAULT_METHODS,
    seed: int = 0,
    labels: dict[str, object] | None = None,
    model_factory: ModelFactory = default_model_factory,
) -> list[Record]:
    """Score every method on the same bootstrap replicates, trial by trial.

    Returns one record per (trial, method) holding the quality metrics of
    :func:`bwa_xai.metrics.evaluate` plus the entries of ``labels``. If you
    pass a custom ``model_factory``, build the task with the same one
    (``subsample_task(..., model_factory=...)``) so that the reference and
    the replicates come from the same model family.
    """
    _check_methods(methods)
    records: list[Record] = []
    for index, rng in enumerate(_trial_rngs(seed, n_trials)):
        trial = task(rng)
        replicates = bootstrap_attributions(
            trial.X, trial.y, n_boot=n_boot, model_factory=model_factory, rng=rng
        )
        for name in methods:
            estimate = AGGREGATORS[name](replicates, trial.X)
            scores = evaluate(estimate, trial.truth, trial.k)
            records.append({**(labels or {}), "trial": index, "method": name, **scores})
    return records


def run_stability(
    task: Task,
    n_datasets: int = 20,
    n_runs: int = 8,
    n_boot: int = 50,
    methods: Sequence[str] = DEFAULT_METHODS,
    seed: int = 0,
    labels: dict[str, object] | None = None,
    model_factory: ModelFactory = default_model_factory,
) -> list[Record]:
    """Repeat bootstrap-and-aggregate ``n_runs`` times per dataset draw.

    Returns one record per (dataset draw, method) holding the run-to-run
    stability metrics of :func:`bwa_xai.stability.stability_metrics`.
    """
    _check_methods(methods)
    if n_runs < 2:
        raise ValueError(f"n_runs must be at least 2; got {n_runs}")
    records: list[Record] = []
    for index, rng in enumerate(_trial_rngs(seed, n_datasets)):
        trial = task(rng)
        runs: dict[str, list[np.ndarray]] = {name: [] for name in methods}
        for _ in range(n_runs):
            replicates = bootstrap_attributions(
                trial.X, trial.y, n_boot=n_boot, model_factory=model_factory, rng=rng
            )
            for name in methods:
                runs[name].append(AGGREGATORS[name](replicates, trial.X))
        for name in methods:
            scores = stability_metrics(np.vstack(runs[name]), k=trial.k)
            records.append({**(labels or {}), "trial": index, "method": name, **scores})
    return records


def _metric_names(records: Sequence[Record]) -> tuple[str, ...]:
    # Some names (jaccard, spearman) belong to both families; list each once.
    known = dict.fromkeys(QUALITY_METRICS + STABILITY_METRICS)
    return tuple(name for name in known if records and name in records[0])


def _group_key(record: Record, metrics: Iterable[str], exclude: Iterable[str]) -> tuple:
    skipped = set(metrics) | set(exclude)
    return tuple((key, value) for key, value in record.items() if key not in skipped)


def summarize(records: Sequence[Record]) -> list[Record]:
    """Mean and standard error of every metric per (labels, method) group."""
    metrics = _metric_names(records)
    groups: dict[tuple, list[Record]] = defaultdict(list)
    for record in records:
        groups[_group_key(record, metrics, exclude=("trial",))].append(record)
    rows: list[Record] = []
    for key, members in groups.items():
        for metric in metrics:
            values = np.array([member[metric] for member in members], dtype=float)
            values = values[np.isfinite(values)]
            count = len(values)
            mean = float(values.mean()) if count else float("nan")
            sem = float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else float("nan")
            rows.append({**dict(key), "metric": metric, "mean": mean, "sem": sem, "n": count})
    return rows


def _paired_p_value(differences: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value; 1.0 when the two methods never differ."""
    if len(differences) == 0 or np.all(differences == 0):
        return 1.0
    return float(wilcoxon(differences).pvalue)


def _index_by_trial(
    records: Sequence[Record], metrics: Sequence[str]
) -> dict[tuple, dict[str, dict[object, Record]]]:
    """Nest records as ``{label group: {method: {trial: record}}}``."""
    groups: dict[tuple, dict[str, dict[object, Record]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        key = _group_key(record, metrics, exclude=NON_GROUP_KEYS)
        by_trial = groups[key][str(record["method"])]
        if record["trial"] in by_trial:
            raise ValueError(
                f"duplicate trial {record['trial']} for method {record['method']!r} in group "
                f"{dict(key)}; give result sets from separate runs different labels"
            )
        by_trial[record["trial"]] = record
    return groups


def paired_wilcoxon(records: Sequence[Record], reference: str = "vanilla_mean") -> list[Record]:
    """Paired Wilcoxon signed-rank tests of every method against ``reference``.

    Pairs are matched by trial within each label group. ``p_bonferroni``
    corrects for the number of methods compared with the reference on the
    same metric in the same label group, so it does not depend on which
    other datasets or metrics were run alongside.
    """
    metrics = _metric_names(records)
    groups = _index_by_trial(records, metrics)
    if not any(reference in by_method for by_method in groups.values()):
        raise ValueError(f"reference method {reference!r} does not appear in the records")
    rows: list[Record] = []
    for key, by_method in groups.items():
        baseline = by_method.get(reference, {})
        for method, by_trial in by_method.items():
            if method == reference:
                continue
            shared = [trial for trial in by_trial if trial in baseline]
            for metric in metrics:
                pairs = np.array(
                    [(by_trial[t][metric], baseline[t][metric]) for t in shared], dtype=float
                ).reshape(-1, 2)
                pairs = pairs[np.all(np.isfinite(pairs), axis=1)]
                differences = pairs[:, 0] - pairs[:, 1]
                rows.append({
                    **dict(key), "method": method, "reference": reference, "metric": metric,
                    "mean_diff": float(differences.mean()) if len(differences) else float("nan"),
                    "p_value": _paired_p_value(differences), "n": len(differences),
                })
    family_sizes: dict[tuple, int] = defaultdict(int)
    for row in rows:
        family_sizes[_family(row)] += int(row["n"] > 0)
    return [
        {**row, "p_bonferroni": min(1.0, float(row["p_value"]) * max(family_sizes[_family(row)], 1))}
        for row in rows
    ]


def _family(test_row: Record) -> tuple:
    """Bonferroni family of a test: its label group and metric."""
    own = ("method", "reference", "mean_diff", "p_value", "n", "p_bonferroni")
    return tuple((key, value) for key, value in test_row.items() if key not in own)
