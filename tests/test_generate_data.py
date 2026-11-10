from rubric_eval.generate_data import NUM_PASSES, RATERS, generate
from rubric_eval.rubric import DIMENSION_NAMES, SCALE_MAX, SCALE_MIN


def test_shapes_and_bounds():
    scores, truth = generate(n_items=40, seed=7)

    expected_score_rows = 40 * len(DIMENSION_NAMES) * len(RATERS) * NUM_PASSES
    assert len(scores) == expected_score_rows
    assert len(truth) == 40 * len(DIMENSION_NAMES)

    assert scores["score"].between(SCALE_MIN, SCALE_MAX).all()
    assert truth["hidden_quality"].between(SCALE_MIN, SCALE_MAX).all()
    assert set(scores["dimension"].unique()) == set(DIMENSION_NAMES)
    assert set(scores["rater"].unique()) == set(RATERS)
    assert set(scores["pass_num"].unique()) == {1, 2, 3}


def test_deterministic_for_same_seed():
    scores_1, truth_1 = generate(n_items=25, seed=99)
    scores_2, truth_2 = generate(n_items=25, seed=99)
    assert scores_1.equals(scores_2)
    assert truth_1.equals(truth_2)


def test_different_seeds_differ():
    scores_a, _ = generate(n_items=25, seed=1)
    scores_b, _ = generate(n_items=25, seed=2)
    assert not scores_a["score"].equals(scores_b["score"])


def test_extending_item_count_preserves_earlier_items():
    """Generating more items must not reshuffle the earlier ones -- each item's
    data comes from an RNG seeded on its own index, independent of n_items."""
    scores_small, truth_small = generate(n_items=20, seed=5)
    scores_large, truth_large = generate(n_items=30, seed=5)

    overlap_ids = sorted(scores_small["item_id"].unique())
    small_slice = scores_large[scores_large["item_id"].isin(overlap_ids)].reset_index(drop=True)
    assert small_slice.equals(scores_small.reset_index(drop=True))

    truth_overlap_ids = sorted(truth_small["item_id"].unique())
    truth_slice = truth_large[truth_large["item_id"].isin(truth_overlap_ids)].reset_index(drop=True)
    assert truth_slice.equals(truth_small.reset_index(drop=True))


def test_ambiguous_share_roughly_respected():
    _, truth = generate(n_items=2000, seed=11, ambiguous_share=0.3)
    # one row per (item, dimension); is_ambiguous is set per item, so dedupe by item
    per_item = truth.drop_duplicates("item_id")
    share = per_item["is_ambiguous"].mean()
    assert 0.24 < share < 0.36
