import numpy as np
import pytest

from rubric_eval.agreement import (
    bootstrap_kappa_ci,
    quadratic_weighted_kappa,
    quadratic_weighted_kappa_closed_form,
)


def test_perfect_agreement_is_kappa_one():
    ratings = np.array([0, 1, 2, 3, 4, 2, 1, 0])
    assert quadratic_weighted_kappa(ratings, ratings) == pytest.approx(1.0)
    assert quadratic_weighted_kappa_closed_form(ratings, ratings) == pytest.approx(1.0)


def test_confusion_matrix_and_closed_form_agree_on_random_data():
    rng = np.random.default_rng(123)
    for _ in range(20):
        n = rng.integers(20, 200)
        a = rng.integers(0, 5, size=n)
        b = rng.integers(0, 5, size=n)
        k1 = quadratic_weighted_kappa(a, b)
        k2 = quadratic_weighted_kappa_closed_form(a, b)
        assert k1 == pytest.approx(k2, abs=1e-9), (k1, k2)


def test_systematic_offset_reduces_kappa_relative_to_matched_pair():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, size=500)
    b_matched = a.copy()
    b_offset = np.clip(a + 1, 0, 4)
    k_matched = quadratic_weighted_kappa(a, b_matched)
    k_offset = quadratic_weighted_kappa(a, b_offset)
    assert k_matched > k_offset


def test_near_random_ratings_give_low_kappa():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 5, size=5000)
    b = rng.integers(0, 5, size=5000)
    k = quadratic_weighted_kappa(a, b)
    assert abs(k) < 0.05  # independent draws: kappa should sit near zero


def test_constant_ratings_edge_case_is_defined():
    a = np.array([2, 2, 2, 2])
    b = np.array([2, 2, 2, 2])
    assert quadratic_weighted_kappa(a, b) == 1.0


def test_bootstrap_ci_contains_point_estimate_and_has_sane_width():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 5, size=300)
    b = np.clip(a + rng.integers(-1, 2, size=300), 0, 4)
    point, lower, upper = bootstrap_kappa_ci(a, b, n_boot=500, seed=0)
    assert lower <= point <= upper
    assert 0 < (upper - lower) < 1.0


def test_bootstrap_ci_narrows_with_more_items():
    rng = np.random.default_rng(4)
    a_small = rng.integers(0, 5, size=40)
    b_small = np.clip(a_small + rng.integers(-1, 2, size=40), 0, 4)
    a_large = rng.integers(0, 5, size=2000)
    b_large = np.clip(a_large + rng.integers(-1, 2, size=2000), 0, 4)

    _, lo_s, hi_s = bootstrap_kappa_ci(a_small, b_small, n_boot=800, seed=1)
    _, lo_l, hi_l = bootstrap_kappa_ci(a_large, b_large, n_boot=800, seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s)
