"""Turn three independent passes per (item, dimension, rater) into one
consensus score, and measure how consistent the passes were with each other.

Consensus rule (deterministic, documented so it can be audited):
  1. Take the mode of the three passes.
  2. If the mode is not unique (e.g. [1, 2, 3], three distinct values),
     break the tie by choosing the value closest to the mean of the passes.
  3. If that is still tied (equally close on both sides), choose the smaller
     value. This never raises and never depends on dict/set ordering.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd


def consensus_score(passes: list[int]) -> int:
    counts = Counter(passes)
    max_count = max(counts.values())
    candidates = sorted(v for v, c in counts.items() if c == max_count)
    if len(candidates) == 1:
        return candidates[0]
    mean = sum(passes) / len(passes)
    candidates.sort(key=lambda v: (abs(v - mean), v))
    return candidates[0]


def build_consensus_table(scores_long: pd.DataFrame) -> pd.DataFrame:
    """scores_long: item_id, dimension, rater, pass_num, score
    -> one row per (item_id, dimension, rater) with the consensus score plus
       intra-rater consistency stats for that triplet of passes.
    """
    rows = []
    group_cols = ["item_id", "dimension", "rater"]
    for (item_id, dimension, rater), g in scores_long.groupby(group_cols, sort=False):
        passes = g.sort_values("pass_num")["score"].tolist()
        rows.append(
            {
                "item_id": item_id,
                "dimension": dimension,
                "rater": rater,
                "consensus_score": consensus_score(passes),
                "passes_exact_match": len(set(passes)) == 1,
                "passes_max_spread": max(passes) - min(passes),
            }
        )
    return pd.DataFrame(rows)


def intra_rater_summary(consensus_table: pd.DataFrame) -> pd.DataFrame:
    """Per rater x dimension: how often all 3 passes agreed exactly, and the
    average spread between the passes. This is reliability of a SINGLE rater
    with themselves, not agreement between raters."""
    return (
        consensus_table.groupby(["rater", "dimension"])
        .agg(
            exact_match_rate=("passes_exact_match", "mean"),
            mean_pass_spread=("passes_max_spread", "mean"),
            n_items=("item_id", "nunique"),
        )
        .reset_index()
        .sort_values(["dimension", "rater"])
    )
