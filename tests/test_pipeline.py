"""Tests for explainers, bootstrapping, datasets, benchmarks, reporting and the CLI."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from bwa_xai import bootstrap, datasets
from bwa_xai.benchmark import (
    make_task,
    paired_wilcoxon,
    run_comparison,
    run_stability,
    subsample_task,
    summarize,
    synthetic_task,
)
from bwa_xai.bootstrap import bootstrap_attributions, bootstrap_indices
from bwa_xai.cli import main
from bwa_xai.datasets import OFFLINE_DATASETS, load_dataset, make_synthetic, standardize
from bwa_xai.explainers import coefficient_explainer, linear_shap_explainer, make_shap_explainer
from bwa_xai.metrics import QUALITY_METRICS
from bwa_xai.reporting import summary_table, write_csv
from bwa_xai.stability import STABILITY_METRICS

SMALL_METHODS = ("vanilla_mean", "bwa_v1", "bwa_v2")


@pytest.fixture(scope="module")
def small_data():
    return make_synthetic(n_samples=40, n_features=12, n_active=3, rng=0)


@pytest.fixture(scope="module")
def fitted(small_data):
    return LogisticRegression().fit(small_data.X, small_data.y)


@pytest.fixture
def small_task():
    return synthetic_task(n_samples=20, n_features=20, n_active=4)


# --- explainers -------------------------------------------------------------


def test_coefficient_explainer_returns_flat_coefficients(fitted, small_data):
    attribution = coefficient_explainer(fitted, small_data.X)

    assert attribution.shape == (12,)
    assert np.array_equal(attribution, fitted.coef_[0])


def test_coefficient_explainer_rejects_unsupported_models(small_data):
    tree = DecisionTreeClassifier().fit(small_data.X, small_data.y)
    multiclass = LogisticRegression().fit(small_data.X, np.arange(40) % 3)

    with pytest.raises(TypeError, match="coef_"):
        coefficient_explainer(tree, small_data.X)
    with pytest.raises(ValueError, match="multi-class"):
        coefficient_explainer(multiclass, small_data.X)


def test_linear_shap_values_sum_to_the_prediction_gap(fitted, small_data):
    x = small_data.X[0]

    phi = linear_shap_explainer(fitted, small_data.X, x)

    gap = fitted.decision_function(x[None, :])[0] - fitted.decision_function(small_data.X).mean()
    assert phi.sum() == pytest.approx(gap)
    with pytest.raises(ValueError, match="instance"):
        linear_shap_explainer(fitted, small_data.X, None)


def test_shap_library_explainer_agrees_with_the_closed_form(fitted, small_data):
    pytest.importorskip("shap")
    explain = make_shap_explainer()
    x = small_data.X[0]

    phi = explain(fitted, small_data.X, x)

    assert np.allclose(phi, linear_shap_explainer(fitted, small_data.X, x), atol=1e-6)
    with pytest.raises(ValueError, match="instance"):
        explain(fitted, small_data.X, None)


# --- bootstrap --------------------------------------------------------------


def test_bootstrap_attributions_shape_and_reproducibility(small_data):
    first = bootstrap_attributions(small_data.X, small_data.y, n_boot=6, rng=1)
    again = bootstrap_attributions(small_data.X, small_data.y, n_boot=6, rng=1)
    other = bootstrap_attributions(small_data.X, small_data.y, n_boot=6, rng=2)

    assert first.shape == (6, 12)
    assert np.array_equal(first, again)
    assert not np.allclose(first, other)


def test_bootstrap_supports_local_explainers_and_regression(small_data):
    local = bootstrap_attributions(
        small_data.X, small_data.y, n_boot=4, explainer=linear_shap_explainer, x=small_data.X[0], rng=0
    )
    regression = bootstrap_attributions(
        small_data.X, small_data.X @ small_data.beta, n_boot=4,
        model_factory=Ridge, ensure_all_classes=False, rng=0,
    )

    assert local.shape == regression.shape == (4, 12)


def test_bootstrap_indices_always_contain_every_class():
    y = np.array([0] * 19 + [1])
    rng = np.random.default_rng(0)

    for _ in range(50):
        assert set(y[bootstrap_indices(y, rng)]) == {0, 1}


def test_bootstrap_indices_fail_loudly_when_a_class_cannot_be_drawn(monkeypatch):
    monkeypatch.setattr(bootstrap, "MAX_RESAMPLE_ATTEMPTS", 0)

    with pytest.raises(RuntimeError, match="too rare"):
        bootstrap_indices(np.array([0, 1]), np.random.default_rng(0))


def test_bootstrap_attributions_validate_inputs(small_data):
    with pytest.raises(ValueError, match="2-D"):
        bootstrap_attributions(small_data.X[0], small_data.y)
    with pytest.raises(ValueError, match="one entry per row"):
        bootstrap_attributions(small_data.X, small_data.y[:-1])
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_attributions(small_data.X, small_data.y, n_boot=1)
    with pytest.raises(ValueError, match="NaN"):
        bootstrap_attributions(np.full_like(small_data.X, np.nan), small_data.y)


# --- datasets ---------------------------------------------------------------


def test_make_synthetic_has_the_requested_shape_and_sparsity():
    data = make_synthetic(n_samples=20, n_features=100, n_active=10, rng=0)

    assert data.X.shape == (20, 100) and data.y.shape == (20,)
    assert np.count_nonzero(data.beta) == 10
    assert set(np.abs(data.beta[data.beta != 0])) <= {1.0, 2.0}
    assert min(data.y.sum(), 20 - data.y.sum()) >= 2
    assert np.array_equal(data.X, make_synthetic(rng=0).X)


def test_make_synthetic_features_are_block_correlated():
    data = make_synthetic(n_samples=4000, n_features=20, n_active=2, rho=0.8, block_size=10, rng=0)

    correlation = np.corrcoef(data.X, rowvar=False)

    assert correlation[0, 5] == pytest.approx(0.8, abs=0.05)
    assert correlation[0, 15] == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize(
    "kwargs", [{"n_active": 0}, {"n_active": 500}, {"rho": 1.0}, {"n_samples": 3}]
)
def test_make_synthetic_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        make_synthetic(**kwargs)


def test_standardize_leaves_constant_columns_at_zero():
    raw = np.column_stack([np.arange(5.0), np.full(5, 3.0)])

    scaled = standardize(raw)

    assert scaled[:, 0].mean() == pytest.approx(0.0) and scaled[:, 0].std() == pytest.approx(1.0)
    assert np.all(scaled[:, 1] == 0)


@pytest.mark.parametrize("name", OFFLINE_DATASETS)
def test_offline_datasets_are_standardised_and_binary(name):
    dataset = load_dataset(name)

    assert dataset.name == name
    assert np.all(np.isfinite(dataset.X))
    assert np.allclose(dataset.X.mean(axis=0), 0.0, atol=1e-8)
    assert set(np.unique(dataset.y)) == {0, 1}


def test_load_dataset_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("imagenet")


def test_openml_loader_keeps_the_highest_variance_features(monkeypatch):
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(30, 6)) * np.array([1.0, 9.0, 2.0, 8.0, 3.0, 7.0])
    target = np.array(["AML", "ALL"] * 15, dtype=object)
    monkeypatch.setattr(datasets, "fetch_openml", lambda **_: SimpleNamespace(data=raw, target=target))

    dataset = load_dataset("leukemia_d500")
    top_three = datasets._from_openml("toy", "leukemia", "AML", top_variance=3)

    assert dataset.X.shape == (30, 6)
    assert dataset.y.sum() == 15
    assert np.allclose(top_three.X, standardize(raw[:, [1, 3, 5]]))


# --- benchmark --------------------------------------------------------------


def test_run_comparison_scores_every_method_on_every_trial(small_task):
    kwargs = {"n_trials": 3, "n_boot": 8, "methods": SMALL_METHODS, "labels": {"dataset": "toy"}}

    records = run_comparison(small_task, **kwargs)
    again = run_comparison(small_task, **kwargs)

    assert len(records) == 3 * len(SMALL_METHODS)
    assert all(record["dataset"] == "toy" for record in records)
    assert all(metric in records[0] for metric in QUALITY_METRICS)
    assert [r["gini"] for r in records] == [r["gini"] for r in again]


def test_bwa_v2_matches_vanilla_mean_sign_accuracy_on_every_trial(small_task):
    records = run_comparison(small_task, n_trials=5, n_boot=8, methods=SMALL_METHODS)

    by_method = {
        name: [r["sign_accuracy"] for r in records if r["method"] == name] for name in SMALL_METHODS
    }
    assert by_method["bwa_v2"] == by_method["vanilla_mean"]


def test_benchmarks_reject_unknown_methods_and_too_few_runs(small_task):
    with pytest.raises(ValueError, match="unknown methods"):
        run_comparison(small_task, n_trials=1, methods=("magic",))
    with pytest.raises(ValueError, match="n_runs"):
        run_stability(small_task, n_runs=1)


def test_subsample_task_draws_stratified_subsets_with_a_full_data_reference():
    dataset = load_dataset("breast_cancer")
    task = subsample_task(dataset, n_samples=20)

    trial = task(np.random.default_rng(0))

    assert trial.X.shape == (20, 30)
    assert set(trial.y) == {0, 1}
    assert trial.truth.shape == (30,) and trial.k == 10
    with pytest.raises(ValueError, match="n_samples"):
        subsample_task(dataset, n_samples=10_000)


def test_make_task_dispatches_on_the_dataset_name():
    synthetic = make_task("synthetic", 20)(np.random.default_rng(0))
    real = make_task("wine", 20)(np.random.default_rng(0))

    assert synthetic.X.shape == (20, 100) and synthetic.k == 10
    assert real.X.shape == (20, 13)


def test_run_stability_reports_stability_metrics_per_dataset_draw(small_task):
    records = run_stability(small_task, n_datasets=2, n_runs=3, n_boot=8, methods=SMALL_METHODS)

    assert len(records) == 2 * len(SMALL_METHODS)
    assert all(metric in records[0] for metric in STABILITY_METRICS)
    assert all(0.0 < r["cosine"] <= 1.0 for r in records if r["method"] != "bwa_v1")


def _toy_records():
    rng = np.random.default_rng(0)
    base = rng.normal(size=10)
    records = []
    for trial in range(10):
        records.append({"dataset": "toy", "trial": trial, "method": "vanilla_mean", "gini": base[trial]})
        records.append({"dataset": "toy", "trial": trial, "method": "same", "gini": base[trial]})
        records.append(
            {"dataset": "toy", "trial": trial, "method": "better", "gini": base[trial] + 1.0 + 0.01 * trial}
        )
    return records


def test_summarize_reports_mean_and_standard_error_per_group():
    records = [
        {"dataset": "toy", "trial": 0, "method": "a", "gini": 1.0},
        {"dataset": "toy", "trial": 1, "method": "a", "gini": 3.0},
        {"dataset": "toy", "trial": 2, "method": "a", "gini": float("nan")},
    ]

    (row,) = summarize(records)

    assert (row["dataset"], row["method"], row["metric"], row["n"]) == ("toy", "a", "gini", 2)
    assert row["mean"] == pytest.approx(2.0)
    assert row["sem"] == pytest.approx(1.0)


def test_paired_wilcoxon_separates_real_differences_from_identical_methods():
    rows = {row["method"]: row for row in paired_wilcoxon(_toy_records())}

    assert rows["same"]["p_value"] == 1.0 and rows["same"]["mean_diff"] == 0.0
    assert rows["better"]["p_value"] < 0.01
    assert rows["better"]["mean_diff"] == pytest.approx(1.045)
    assert rows["better"]["p_bonferroni"] == pytest.approx(2 * rows["better"]["p_value"])
    assert rows["better"]["n"] == 10


def test_metrics_shared_by_both_families_are_reported_once(small_task):
    records = run_stability(small_task, n_datasets=3, n_runs=2, n_boot=8, methods=("vanilla_mean", "bwa_v2"))

    summary_metrics = [row["metric"] for row in summarize(records)]
    tests = paired_wilcoxon(records)

    assert sorted(summary_metrics) == sorted(2 * list(STABILITY_METRICS))
    assert sorted(row["metric"] for row in tests) == sorted(STABILITY_METRICS)
    assert all(row["p_bonferroni"] <= len(STABILITY_METRICS) * row["p_value"] + 1e-12 for row in tests)


def test_paired_wilcoxon_rejects_records_without_the_reference_method():
    records = [r for r in _toy_records() if r["method"] != "vanilla_mean"]

    with pytest.raises(ValueError, match="reference method"):
        paired_wilcoxon(records)


def test_paired_wilcoxon_rejects_duplicate_trials_in_one_group():
    with pytest.raises(ValueError, match="duplicate trial"):
        paired_wilcoxon(_toy_records() + _toy_records())


def test_bonferroni_family_is_unaffected_by_other_datasets():
    alone = {row["method"]: row for row in paired_wilcoxon(_toy_records())}
    other_dataset = [{**record, "dataset": "other"} for record in _toy_records()]

    together = [row for row in paired_wilcoxon(_toy_records() + other_dataset) if row["dataset"] == "toy"]

    assert len(together) == 2
    for row in together:
        assert row["p_bonferroni"] == alone[row["method"]]["p_bonferroni"]


def test_cli_omits_significance_tests_without_the_reference_method(tmp_path):
    argv = ["compare", "--datasets", "wine", "--n-samples", "20", "--n-trials", "2", "--n-boot", "6"]

    assert main(argv + ["--methods", "bwa_v2", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "compare_summary.csv").exists()
    assert not (tmp_path / "compare_wilcoxon.csv").exists()


# --- reporting and CLI ------------------------------------------------------


def test_write_csv_round_trips_records(tmp_path):
    rows = [{"method": "a", "gini": 0.5}, {"method": "b", "gini": 0.25, "extra": 1}]

    path = write_csv(rows, tmp_path / "nested" / "out.csv")

    with path.open(encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert [row["method"] for row in loaded] == ["a", "b"]
    assert loaded[1]["gini"] == "0.25" and loaded[0]["extra"] == ""
    with pytest.raises(ValueError, match="no rows"):
        write_csv([], tmp_path / "empty.csv")


def test_summary_table_renders_one_row_per_group():
    summary = [
        {"dataset": "toy", "method": "a", "metric": "gini", "mean": 0.5, "sem": 0.01, "n": 10},
        {"dataset": "toy", "method": "a", "metric": "norm", "mean": float("nan"), "sem": float("nan"), "n": 0},
        {"dataset": "toy", "method": "b", "metric": "gini", "mean": 0.25, "sem": float("nan"), "n": 1},
    ]

    lines = summary_table(summary).splitlines()

    assert lines[0] == "| dataset | method | gini | norm |"
    assert lines[2] == "| toy | a | 0.500 ± 0.010 | n/a |"
    assert lines[3] == "| toy | b | 0.250 | n/a |"
    with pytest.raises(ValueError, match="empty"):
        summary_table([])


@pytest.mark.parametrize(
    "command, extra",
    [("compare", ["--n-trials", "2"]), ("stability", ["--n-datasets", "2", "--n-runs", "2"])],
)
def test_cli_writes_records_summary_and_tests(tmp_path, capsys, command, extra):
    argv = [command, "--datasets", "wine", "--n-samples", "20", "--n-boot", "6", "--out", str(tmp_path)]

    exit_code = main(argv + extra)

    assert exit_code == 0
    for suffix in ("records.csv", "summary.csv", "wilcoxon.csv", "summary.md"):
        assert (tmp_path / f"{command}_{suffix}").exists()
    assert "| wine | 20 | bwa_v2 |" in capsys.readouterr().out


def test_cli_skips_infeasible_cells_and_reports_failure(tmp_path):
    argv = ["compare", "--datasets", "wine", "--n-samples", "5000", "--n-trials", "1", "--out", str(tmp_path)]

    assert main(argv) == 1
    assert not list(tmp_path.iterdir())
