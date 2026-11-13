"""Tests for the quality metrics and the run-to-run stability metrics."""

import numpy as np
import pytest

from bwa_xai import evaluate, gini, jaccard_at_k, sign_accuracy, spearman_abs, stability_metrics
from bwa_xai.metrics import QUALITY_METRICS, l2_norm, top_k
from bwa_xai.stability import STABILITY_METRICS, norm_cv


def test_sign_accuracy_counts_abstentions_on_active_features_as_wrong():
    estimate = np.array([1.0, -1.0, 0.0, 5.0])
    truth = np.array([2.0, 1.0, 3.0, 0.0])

    assert sign_accuracy(estimate, truth) == pytest.approx(1.0 / 3.0)


def test_sign_accuracy_requires_an_active_feature():
    with pytest.raises(ValueError, match="non-zero"):
        sign_accuracy([1.0, 2.0], [0.0, 0.0])


def test_metrics_reject_mismatched_or_malformed_vectors():
    with pytest.raises(ValueError, match="same length"):
        sign_accuracy([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="1-D"):
        gini(np.ones((2, 2)))
    with pytest.raises(ValueError, match="NaN"):
        l2_norm([1.0, np.nan])


def test_jaccard_is_intersection_over_union_of_top_k_sets():
    estimate = np.array([5.0, -4.0, 0.0, 0.0])
    truth = np.array([5.0, 0.0, 4.0, 0.0])

    assert jaccard_at_k(estimate, truth, k=2) == pytest.approx(1.0 / 3.0)
    assert jaccard_at_k(estimate, estimate, k=2) == 1.0


@pytest.mark.parametrize("k", [0, 5])
def test_top_k_rejects_out_of_range_k(k):
    with pytest.raises(ValueError, match="k must be"):
        top_k([1.0, 2.0, 3.0, 4.0], k)


def test_top_k_breaks_ties_by_index():
    assert top_k([0.0, 0.0, 3.0, 0.0], 3).tolist() == [2, 0, 1]


def test_spearman_ignores_sign_and_is_nan_for_constant_input():
    truth = np.array([1.0, 2.0, 3.0, 4.0])

    assert spearman_abs(-truth, truth) == pytest.approx(1.0)
    assert spearman_abs(truth[::-1], truth) == pytest.approx(-1.0)
    assert np.isnan(spearman_abs(np.ones(4), truth))


def test_gini_spans_uniform_to_concentrated():
    assert gini(np.ones(4)) == pytest.approx(0.0)
    assert gini([0.0, 0.0, 0.0, 9.0]) == pytest.approx(0.75)
    assert gini(np.zeros(4)) == 0.0


def test_gini_is_invariant_to_sign_and_scale():
    values = np.random.default_rng(0).normal(size=50)

    assert gini(-7.0 * values) == pytest.approx(gini(values))


def test_evaluate_returns_every_quality_metric():
    estimate = np.array([3.0, -4.0, 0.0])

    scores = evaluate(estimate, np.array([1.0, -1.0, 0.5]), k=2)

    assert tuple(scores) == QUALITY_METRICS
    assert scores["norm"] == pytest.approx(5.0)
    assert scores["sign_accuracy"] == pytest.approx(2.0 / 3.0)


# --- stability --------------------------------------------------------------


def test_identical_runs_are_perfectly_stable():
    runs = np.tile(np.array([3.0, -1.0, 0.5, 2.0]), (4, 1))

    scores = stability_metrics(runs, k=2)

    assert tuple(scores) == STABILITY_METRICS
    assert scores["cosine"] == pytest.approx(1.0)
    assert scores["norm_cv"] == pytest.approx(0.0)
    assert scores["spearman"] == pytest.approx(1.0)
    assert scores["jaccard"] == 1.0


def test_sign_flipped_runs_have_negative_cosine_but_stable_ranking():
    base = np.array([3.0, -1.0, 0.5, 2.0])

    scores = stability_metrics(np.vstack([base, -base]), k=2)

    assert scores["cosine"] == pytest.approx(-1.0)
    assert scores["spearman"] == pytest.approx(1.0)


def test_noisier_runs_score_as_less_stable():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=40)
    quiet = signal + 0.05 * rng.normal(size=(8, 40))
    noisy = signal + 1.0 * rng.normal(size=(8, 40))

    quiet_scores, noisy_scores = stability_metrics(quiet), stability_metrics(noisy)

    assert quiet_scores["cosine"] > noisy_scores["cosine"]
    assert quiet_scores["spearman"] > noisy_scores["spearman"]
    assert quiet_scores["jaccard"] > noisy_scores["jaccard"]
    assert quiet_scores["norm_cv"] < noisy_scores["norm_cv"]


def test_norm_cv_matches_hand_computed_value():
    runs = np.array([[3.0, 4.0], [6.0, 8.0]])  # norms 5 and 10

    assert norm_cv(runs) == pytest.approx(2.5 / 7.5)


def test_zero_runs_give_undefined_stability_instead_of_crashing():
    scores = stability_metrics(np.zeros((3, 4)), k=10)

    assert np.isnan(scores["cosine"])
    assert np.isnan(scores["norm_cv"])
    assert np.isnan(scores["spearman"])


def test_stability_needs_at_least_two_runs():
    with pytest.raises(ValueError, match="at least 2"):
        stability_metrics(np.ones((1, 4)))
