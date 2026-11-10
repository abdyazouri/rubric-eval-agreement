import pandas as pd

from rubric_eval.report import disagreement_summary, flag_disagreements, validate_against_ground_truth


def _consensus_row(item_id, dimension, human, model):
    return [
        {"item_id": item_id, "dimension": dimension, "rater": "human", "consensus_score": human},
        {"item_id": item_id, "dimension": dimension, "rater": "model", "consensus_score": model},
    ]


def test_flag_disagreements_threshold():
    rows = (
        _consensus_row("i1", "accuracy", 3, 3)  # diff 0
        + _consensus_row("i2", "accuracy", 3, 1)  # diff 2 -> flagged
        + _consensus_row("i3", "accuracy", 4, 3)  # diff 1
    )
    consensus = pd.DataFrame(rows)
    flagged = flag_disagreements(consensus)
    assert flagged.set_index("item_id")["high_disagreement"].to_dict() == {"i1": False, "i2": True, "i3": False}


def test_disagreement_summary_counts():
    rows = (
        _consensus_row("i1", "accuracy", 3, 3)
        + _consensus_row("i2", "accuracy", 3, 1)
        + _consensus_row("i3", "clarity", 4, 4)
    )
    consensus = pd.DataFrame(rows)
    flagged = flag_disagreements(consensus)
    summary = disagreement_summary(flagged).set_index("dimension")
    assert summary.loc["accuracy", "high_disagreement_count"] == 1
    assert summary.loc["clarity", "high_disagreement_count"] == 0


def test_validation_shows_more_disagreement_on_ambiguous_items():
    # Construct a case where disagreement is concentrated on ambiguous items by design.
    rows = []
    truth_rows = []
    for i in range(10):
        rows += _consensus_row(f"amb_{i}", "accuracy", human=3, model=1)  # big diff, ambiguous
        truth_rows.append({"item_id": f"amb_{i}", "dimension": "accuracy", "is_ambiguous": True})
    for i in range(10):
        rows += _consensus_row(f"clr_{i}", "accuracy", human=3, model=3)  # no diff, clear
        truth_rows.append({"item_id": f"clr_{i}", "dimension": "accuracy", "is_ambiguous": False})

    consensus = pd.DataFrame(rows)
    ground_truth = pd.DataFrame(truth_rows)
    flagged = flag_disagreements(consensus)
    validation = validate_against_ground_truth(flagged, ground_truth).set_index("dimension")

    assert validation.loc["accuracy", "mean_abs_diff_ambiguous"] > validation.loc["accuracy", "mean_abs_diff_clear"]
    assert validation.loc["accuracy", "high_disagreement_share_ambiguous"] == 1.0
    assert validation.loc["accuracy", "high_disagreement_share_clear"] == 0.0
