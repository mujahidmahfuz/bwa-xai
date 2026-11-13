"""Tests for the BWA algorithm: projection, signs, barycenter, aggregators, API."""

import dataclasses

import numpy as np
import pytest

from bwa_xai import (
    BarycenterError,
    aggregate,
    binomial_sign,
    bootstrap_median,
    bootstrapped_shap,
    bwa_v1,
    bwa_v2,
    correlation_cost,
    quantile_midpoint,
    sigma_uq,
    sign_scale_estimator,
    soft_sign,
    soft_sign_calibration,
    stable_attribution,
    to_simplex,
    vanilla_mean,
    wasserstein_barycenter,
)
from bwa_xai.aggregators import bwa_v1_from_barycenter
from bwa_xai.datasets import make_synthetic
from bwa_xai.uncertainty import feature_std


@pytest.fixture
def replicates():
    rng = np.random.default_rng(0)
    return rng.normal(size=(30, 12)) + np.linspace(-1.0, 1.0, 12)


@pytest.fixture
def features():
    return np.random.default_rng(1).normal(size=(25, 12))


# --- input validation -------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(5),
        np.ones((1, 5)),
        np.ones((3, 0)),
        np.array([[1.0, np.nan], [0.0, 1.0]]),
        np.array([[1.0, np.inf], [0.0, 1.0]]),
    ],
)
def test_aggregators_reject_malformed_replicates(bad):
    with pytest.raises(ValueError):
        bwa_v2(bad)


# --- projection -------------------------------------------------------------


def test_to_simplex_rows_are_probability_vectors(replicates):
    measures = to_simplex(replicates)

    assert np.all(measures >= 0)
    assert np.allclose(measures.sum(axis=1), 1.0)


def test_to_simplex_is_invariant_to_sign_and_scale(replicates):
    assert np.allclose(to_simplex(-3.5 * replicates), to_simplex(replicates))


def test_to_simplex_maps_zero_replicate_to_uniform():
    measures = to_simplex(np.array([[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 2.0]]))

    assert np.allclose(measures[0], 0.25)
    assert np.allclose(measures[1], [0.25, 0.25, 0.0, 0.5])


def test_correlation_cost_is_symmetric_bounded_with_zero_diagonal(features):
    cost = correlation_cost(features)

    assert np.allclose(cost, cost.T)
    assert np.all(np.diag(cost) == 0)
    assert cost.min() >= 0 and cost.max() <= 1


def test_correlation_cost_is_low_between_perfectly_correlated_features(features):
    duplicated = np.column_stack([features[:, 0], -2.0 * features[:, 0], features[:, 1]])

    cost = correlation_cost(duplicated, shrinkage=0.1)

    assert cost[0, 1] == pytest.approx(0.1)
    assert cost[0, 2] > cost[0, 1]


def test_correlation_cost_treats_constant_column_as_uncorrelated(features):
    with_constant = np.column_stack([features[:, :3], np.full(len(features), 7.0)])

    cost = correlation_cost(with_constant)

    assert np.all(np.isfinite(cost))
    assert np.allclose(cost[3, :3], 1.0)


@pytest.mark.parametrize("shrinkage", [-0.1, 1.5])
def test_correlation_cost_rejects_shrinkage_outside_unit_interval(features, shrinkage):
    with pytest.raises(ValueError, match="shrinkage"):
        correlation_cost(features, shrinkage=shrinkage)


def test_correlation_cost_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        correlation_cost(np.ones(4))


# --- signs ------------------------------------------------------------------


def test_soft_sign_matches_hand_computed_example():
    e = np.array([[1.0, -1.0, 0.0], [3.0, 1.0, 0.0]])

    assert np.allclose(soft_sign(e), [1.0, 0.0, 0.0])


def test_soft_sign_equals_sum_over_absolute_sum(replicates):
    expected = replicates.sum(axis=0) / np.abs(replicates).sum(axis=0)

    assert np.allclose(soft_sign(replicates), expected)
    assert np.all(np.abs(soft_sign(replicates)) <= 1.0)


def test_binomial_sign_commits_only_when_the_split_is_significant():
    n_replicates = 50
    e = np.ones((n_replicates, 4))
    e[:, 1] = -1.0
    e[: n_replicates // 2, 2] = -1.0  # 25 of 50 positive: no evidence
    e[:20, 3] = -1.0  # 30 of 50 positive: |0.1| is below the 0.139 threshold

    assert binomial_sign(e).tolist() == [1.0, -1.0, 0.0, 0.0]


def test_binomial_sign_abstains_less_at_a_looser_level():
    e = np.ones((50, 1))
    e[:20, 0] = -1.0

    assert binomial_sign(e, alpha=0.05)[0] == 0.0
    assert binomial_sign(e, alpha=0.5)[0] == 1.0


def test_binomial_sign_ignores_exact_zeros_and_abstains_when_nothing_is_signed():
    e = np.zeros((50, 3))
    e[:30, 1] = 1.0  # 30 signed replicates, all positive
    e[:3, 2] = -1.0  # only 3 signed replicates: too few to commit

    assert binomial_sign(e).tolist() == [0.0, 1.0, 0.0]


def test_bwa_v1_gives_an_all_zero_feature_zero_attribution(replicates, features):
    sparse = replicates.copy()
    sparse[:, 4] = 0.0

    assert bwa_v1(sparse, correlation_cost(features))[4] == 0.0
    assert bwa_v2(sparse)[4] == 0.0


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_binomial_sign_rejects_invalid_alpha(replicates, alpha):
    with pytest.raises(ValueError, match="alpha"):
        binomial_sign(replicates, alpha=alpha)


# --- baselines --------------------------------------------------------------


def test_baseline_aggregators_match_hand_computed_values():
    e = np.array([[1.0, -2.0], [3.0, -4.0], [-1.0, -6.0]])

    assert np.allclose(vanilla_mean(e), [1.0, -4.0])
    assert np.allclose(bootstrap_median(e), [1.0, -4.0])
    assert np.allclose(bootstrapped_shap(e), [5.0 / 3.0, -4.0])


def test_quantile_midpoint_of_symmetric_sample_is_its_centre():
    e = np.arange(11.0).reshape(-1, 1)

    assert quantile_midpoint(e)[0] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="alpha"):
        quantile_midpoint(e, alpha=0.0)


# --- BWA-v2 (Propositions 2-4) ----------------------------------------------


def test_bwa_v2_is_a_nonnegative_rescaling_of_the_mean(replicates):
    magnitudes = np.abs(replicates)
    factor = np.median(magnitudes, axis=0) / magnitudes.mean(axis=0)

    assert np.all(factor >= 0)
    assert np.allclose(bwa_v2(replicates), replicates.mean(axis=0) * factor)


def test_bwa_v2_never_disagrees_in_sign_with_the_mean():
    rng = np.random.default_rng(42)
    disagreements = 0
    for _ in range(100):
        e = rng.standard_t(df=2, size=(rng.integers(5, 60), 80))
        disagreements += int(np.sum(np.sign(bwa_v2(e)) != np.sign(e.mean(axis=0))))

    assert disagreements == 0


def test_soft_sign_with_mean_scale_equals_vanilla_mean(replicates):
    estimate = sign_scale_estimator(replicates, sign="soft", scale="mean")

    assert np.allclose(estimate, vanilla_mean(replicates))


def test_hard_sign_variant_keeps_the_full_median_scale(replicates):
    estimate = sign_scale_estimator(replicates, sign="hard", scale="median")

    assert np.allclose(np.abs(estimate), np.median(np.abs(replicates), axis=0))
    assert np.all(np.abs(estimate) >= np.abs(bwa_v2(replicates)) - 1e-12)


def test_sign_scale_estimator_rejects_unknown_options(replicates):
    with pytest.raises(ValueError, match="sign"):
        sign_scale_estimator(replicates, sign="bayes")
    with pytest.raises(ValueError, match="scale"):
        sign_scale_estimator(replicates, scale="max")


def test_all_zero_feature_gets_zero_attribution():
    e = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    assert bwa_v2(e).tolist() == [2.0, 0.0]


# --- barycenter and BWA-v1 --------------------------------------------------


def test_barycenter_lies_on_the_simplex_and_converges(replicates, features):
    result = wasserstein_barycenter(to_simplex(replicates), correlation_cost(features))

    assert result.converged
    assert np.all(result.measure >= 0)
    assert result.measure.sum() == pytest.approx(1.0)


def test_barycenter_of_identical_measures_recovers_the_measure():
    measure = np.array([0.5, 0.3, 0.15, 0.05])
    expensive_moves = 10.0 * (1.0 - np.eye(4))

    result = wasserstein_barycenter(np.tile(measure, (5, 1)), expensive_moves)

    assert np.allclose(result.measure, measure, atol=1e-3)


def test_barycenter_rejects_invalid_inputs(replicates, features):
    measures = to_simplex(replicates)
    cost = correlation_cost(features)

    with pytest.raises(ValueError, match="shape"):
        wasserstein_barycenter(measures, cost[:5, :5])
    with pytest.raises(ValueError, match="non-negative"):
        wasserstein_barycenter(measures, -cost)
    with pytest.raises(ValueError, match="sum to 1"):
        wasserstein_barycenter(2.0 * measures, cost)
    with pytest.raises(ValueError, match="reg"):
        wasserstein_barycenter(measures, cost, reg=0.0)
    with pytest.raises(ValueError, match="2-D"):
        wasserstein_barycenter(measures[0], cost)


def test_barycenter_raises_when_the_kernel_underflows():
    disjoint = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    with pytest.raises(BarycenterError, match="sinkhorn_log"):
        wasserstein_barycenter(disjoint, 1.0 - np.eye(3), reg=1e-4)


def test_barycenter_warns_when_it_stops_before_converging():
    data = make_synthetic(rng=0)
    measures = to_simplex(np.random.default_rng(0).normal(size=(20, 100)))

    with pytest.warns(RuntimeWarning, match="did not reach"):
        result = wasserstein_barycenter(measures, correlation_cost(data.X), reg=1e-3, max_iter=50)

    assert not result.converged


def test_bwa_v1_zeroes_features_that_fail_the_sign_test(replicates, features):
    estimate = bwa_v1(replicates, correlation_cost(features))
    abstained = binomial_sign(replicates) == 0

    assert abstained.any() and not abstained.all()
    assert np.all(estimate[abstained] == 0)
    assert np.all(np.sign(estimate[~abstained]) == binomial_sign(replicates)[~abstained])


def test_bwa_v1_norm_is_bounded_by_the_median_replicate_norm(replicates, features):
    estimate = bwa_v1(replicates, correlation_cost(features))

    assert np.linalg.norm(estimate) <= np.median(np.linalg.norm(replicates, axis=1))


def test_bwa_v1_from_barycenter_rejects_wrong_length(replicates):
    with pytest.raises(ValueError, match="barycenter"):
        bwa_v1_from_barycenter(replicates, np.ones(3) / 3)


# --- high-level API ---------------------------------------------------------


def test_aggregate_v2_needs_only_replicates(replicates):
    result = aggregate(replicates)

    assert result.method == "v2"
    assert np.allclose(result.attribution, bwa_v2(replicates))
    assert result.barycenter is None and result.barycenter_converged is None


def test_aggregate_v1_builds_the_cost_from_features(replicates, features):
    result = aggregate(replicates, "v1", X=features)

    assert np.allclose(result.attribution, bwa_v1(replicates, correlation_cost(features)))
    assert result.barycenter_converged is True


def test_aggregate_v2_output_is_independent_of_the_cost_matrix(replicates, features):
    first = aggregate(replicates, cost=correlation_cost(features), diagnostic_barycenter=True)
    second = aggregate(replicates, cost=1.0 - np.eye(12), diagnostic_barycenter=True)

    assert not np.allclose(first.barycenter, second.barycenter)
    assert np.array_equal(first.attribution, second.attribution)


def test_aggregate_rejects_bad_arguments(replicates, features):
    with pytest.raises(ValueError, match="method"):
        aggregate(replicates, "v3")
    with pytest.raises(ValueError, match="cost matrix"):
        aggregate(replicates, "v1")
    with pytest.raises(ValueError, match="columns"):
        aggregate(replicates, "v1", X=features[:, :5])


def test_uncertainty_is_a_property_of_the_replicates_not_the_method(replicates, features):
    v1 = aggregate(replicates, "v1", X=features)
    v2 = aggregate(replicates, "v2")

    assert v1.sigma_uq == v2.sigma_uq == sigma_uq(replicates)
    assert v2.sigma_uq == pytest.approx(feature_std(replicates).mean())


def test_result_exposes_reliable_mask_and_top_features():
    e = np.array([[5.0, 1.0, -0.2], [4.0, -1.0, -0.3], [6.0, 1.5, -0.1]])

    result = aggregate(e)

    assert result.reliable().tolist() == [True, False, True]
    assert result.top_features(2).tolist() == [0, 1]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.method = "v1"


def test_stable_attribution_is_reproducible_and_seed_dependent():
    data = make_synthetic(n_samples=30, n_features=15, n_active=3, rng=3)

    first = stable_attribution(data.X, data.y, n_boot=10, seed=7)
    again = stable_attribution(data.X, data.y, n_boot=10, seed=7)
    other = stable_attribution(data.X, data.y, "v1", n_boot=10, seed=8)

    assert np.array_equal(first.attribution, again.attribution)
    assert other.method == "v1" and other.barycenter is not None
    assert not np.allclose(first.soft_sign, other.soft_sign)


# --- soft-sign calibration --------------------------------------------------


def test_soft_sign_calibration_bins_by_confidence_and_ignores_null_features():
    soft = np.array([0.1, -0.5, 0.97, 1.0, 0.3])
    truth = np.array([1.0, 1.0, 1.0, -1.0, 0.0])

    rows = soft_sign_calibration(soft, truth)

    counts = [row["n"] for row in rows]
    assert counts == [1, 0, 1, 0, 0, 2]
    assert rows[0]["accuracy"] == 1.0
    assert rows[2]["accuracy"] == 0.0
    assert rows[5]["accuracy"] == 0.5
    assert np.isnan(rows[1]["accuracy"])


def test_soft_sign_calibration_leaves_out_values_outside_the_bins():
    soft = np.array([0.1, 0.5, 0.99, 1.0, -0.3])

    rows = soft_sign_calibration(soft, np.ones(5), bin_edges=(0.4, 0.8, 1.0))

    assert [row["n"] for row in rows] == [1, 2]
    assert all(row["accuracy"] == 1.0 for row in rows)


def test_soft_sign_calibration_rejects_bad_bins():
    with pytest.raises(ValueError, match="bin_edges"):
        soft_sign_calibration([0.5], [1.0], bin_edges=(0.5, 0.2))
    with pytest.raises(ValueError, match="same length"):
        soft_sign_calibration([0.5, 0.1], [1.0])
