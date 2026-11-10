"""Disagreement diagnostics and a human-readable markdown report.

The diagnostics below answer the question a calibration review actually
needs: not just "what is kappa", but *where* to spend review time next --
which dimension disagrees most, and which specific items drove it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HIGH_DISAGREEMENT_THRESHOLD = 2  # |human - model| at or above this is flagged


def flag_disagreements(consensus_table: pd.DataFrame) -> pd.DataFrame:
    wide = consensus_table.pivot(index=["item_id", "dimension"], columns="rater", values="consensus_score").reset_index()
    wide["abs_diff"] = (wide["human"] - wide["model"]).abs()
    wide["high_disagreement"] = wide["abs_diff"] >= HIGH_DISAGREEMENT_THRESHOLD
    return wide


def disagreement_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    return (
        flagged.groupby("dimension")
        .agg(
            n_items=("item_id", "nunique"),
            mean_abs_diff=("abs_diff", "mean"),
            high_disagreement_count=("high_disagreement", "sum"),
            high_disagreement_share=("high_disagreement", "mean"),
        )
        .reset_index()
        .sort_values("high_disagreement_share", ascending=False)
    )


def validate_against_ground_truth(flagged: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Sanity check only possible because this is synthetic data with known
    ground truth: do the items we *flagged* as high-disagreement actually
    correspond to the boundary (`is_ambiguous`) items we injected? If not,
    the disagreement metric itself would be suspect."""
    merged = flagged.merge(ground_truth, on=["item_id", "dimension"], how="left")
    return (
        merged.groupby("dimension")
        .apply(
            lambda g: pd.Series(
                {
                    "ambiguous_item_share": g["is_ambiguous"].mean(),
                    "mean_abs_diff_ambiguous": g.loc[g["is_ambiguous"], "abs_diff"].mean(),
                    "mean_abs_diff_clear": g.loc[~g["is_ambiguous"], "abs_diff"].mean(),
                    "high_disagreement_share_ambiguous": g.loc[g["is_ambiguous"], "high_disagreement"].mean(),
                    "high_disagreement_share_clear": g.loc[~g["is_ambiguous"], "high_disagreement"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def render_markdown(
    agreement_tbl: pd.DataFrame,
    intra_rater_tbl: pd.DataFrame,
    disagreement_tbl: pd.DataFrame,
    validation_tbl: pd.DataFrame,
    flagged: pd.DataFrame,
    n_items: int,
) -> str:
    lines: list[str] = []
    lines.append("# Rubric Agreement Report\n")
    lines.append(f"Synthetic dataset: {n_items} items, 4 rubric dimensions, 2 blind evaluators (`human`, `model`), "
                  "3 independent scoring passes per evaluator per item (\"precise mode\").\n")

    lines.append("## Inter-rater agreement (quadratic weighted kappa, 95% bootstrap CI)\n")
    lines.append("Sorted ascending, so the dimension needing calibration attention is listed first.\n")
    lines.append(agreement_tbl.round(3).to_markdown(index=False))
    lines.append("")

    lines.append("## Intra-rater consistency across the 3 passes (\"precise mode\")\n")
    lines.append(intra_rater_tbl.round(3).to_markdown(index=False))
    lines.append("")

    lines.append("## Disagreement summary by dimension\n")
    lines.append(disagreement_tbl.round(3).to_markdown(index=False))
    lines.append("")

    lines.append(f"## Items flagged for review (|human - model| >= {HIGH_DISAGREEMENT_THRESHOLD})\n")
    cols = ["item_id", "dimension", "human", "model", "abs_diff"]
    top_flagged = flagged[flagged["high_disagreement"]].sort_values("abs_diff", ascending=False)[cols].head(15)
    lines.append(top_flagged.to_markdown(index=False) if len(top_flagged) else "_None flagged._")
    lines.append("")

    lines.append("## Validation against injected ground truth (synthetic data only)\n")
    lines.append(
        "This section only exists because the dataset is synthetic and the true quality is known. "
        "It checks that disagreement concentrates on the boundary items we deliberately injected, "
        "rather than being noise spread evenly across the dataset -- i.e. that the disagreement "
        "metric is measuring something real.\n"
    )
    lines.append(validation_tbl.round(3).to_markdown(index=False))
    lines.append("")

    return "\n".join(lines)


def write_report(path: str | Path, markdown: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown)
