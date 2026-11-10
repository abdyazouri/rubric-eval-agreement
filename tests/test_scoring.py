import pandas as pd

from rubric_eval.scoring import build_consensus_table, consensus_score, intra_rater_summary


def test_consensus_unique_mode():
    assert consensus_score([2, 2, 3]) == 2
    assert consensus_score([1, 1, 1]) == 1


def test_consensus_three_way_tie_breaks_toward_mean():
    # [1, 2, 3]: all three values occur once (three-way tie on count).
    # mean = 2.0, and 2 itself has distance 0 to the mean, so it wins outright.
    assert consensus_score([1, 2, 3]) == 2


def test_consensus_two_way_tie_with_distinct_counts():
    # [1, 1, 2, 4]: counts are {1: 2, 2: 1, 4: 1} -> mode is unique (1).
    assert consensus_score([1, 1, 2, 4]) == 1


def test_consensus_exact_tie_breaks_toward_smaller_value():
    # [0, 0, 4, 4]: counts are {0: 2, 4: 2}, a two-way tie.
    # mean = 2.0, and both 0 and 4 are equidistant (2.0) from the mean,
    # so the documented tie-break rule picks the smaller value: 0.
    assert consensus_score([0, 0, 4, 4]) == 0
    assert consensus_score([1, 1, 3, 3]) == 1  # same shape, mean = 2.0, smaller (1) wins


def test_build_consensus_table_shapes_and_flags():
    scores_long = pd.DataFrame(
        [
            {"item_id": "i1", "dimension": "accuracy", "rater": "human", "pass_num": 1, "score": 3},
            {"item_id": "i1", "dimension": "accuracy", "rater": "human", "pass_num": 2, "score": 3},
            {"item_id": "i1", "dimension": "accuracy", "rater": "human", "pass_num": 3, "score": 2},
            {"item_id": "i1", "dimension": "accuracy", "rater": "model", "pass_num": 1, "score": 1},
            {"item_id": "i1", "dimension": "accuracy", "rater": "model", "pass_num": 2, "score": 1},
            {"item_id": "i1", "dimension": "accuracy", "rater": "model", "pass_num": 3, "score": 1},
        ]
    )
    table = build_consensus_table(scores_long)
    assert len(table) == 2  # one row per (item, dimension, rater)

    human_row = table[table["rater"] == "human"].iloc[0]
    assert human_row["consensus_score"] == 3
    assert not human_row["passes_exact_match"]
    assert human_row["passes_max_spread"] == 1

    model_row = table[table["rater"] == "model"].iloc[0]
    assert model_row["consensus_score"] == 1
    assert model_row["passes_exact_match"]
    assert model_row["passes_max_spread"] == 0

    summary = intra_rater_summary(table)
    assert set(summary.columns) >= {"rater", "dimension", "exact_match_rate", "mean_pass_spread", "n_items"}
    assert len(summary) == 2
