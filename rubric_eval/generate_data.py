"""Generate SYNTHETIC rubric-scoring data for two blind evaluators.

Nothing here is real evaluation data. We simulate:

  * a hidden, continuous "true quality" per item and per rubric dimension
    (never shown to the scoring/agreement code -- kept in a separate
    "ground truth" table used only by tests and by the report's validation
    section);
  * two independent evaluators -- "human" and "model" -- who never see each
    other's scores (blind double-scoring);
  * "precise mode": each evaluator scores each (item, dimension) THREE
    independent times (repeated passes), so we can also measure how
    consistent a single evaluator is with themselves;
  * a deliberate share of "boundary" items whose hidden quality sits exactly
    between two anchors. These are the items genuine disagreement should
    concentrate on, which the report's diagnostics are built to surface.

Determinism: every (item, dimension, rater, pass) draws from its own RNG
seeded with (seed, item_idx, dim_idx, rater_idx, pass_idx), so results are
fully reproducible and regenerating more items extends rather than reshuffles
the existing ones.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rubric_eval.rubric import DIMENSION_NAMES, SCALE_MAX, SCALE_MIN

RATERS = ("human", "model")
NUM_PASSES = 3


@dataclass(frozen=True)
class RaterProfile:
    """Per-rater, per-dimension scoring behaviour used only to SIMULATE data."""

    bias: float          # systematic leniency (+) or severity (-)
    noise_std: float      # how much a single pass wobbles around the rater's own mean
    anchor_slip_prob: float  # chance of misreading an adjacent anchor, independent of noise


# The "model" evaluator is tuned deliberately stricter on accuracy and noisier
# on empathy -- a plausible, simple story for why dimension-level agreement
# differs, which the diagnostics report is meant to surface.
RATER_PROFILES: dict[str, dict[str, RaterProfile]] = {
    "human": {
        "accuracy": RaterProfile(bias=0.0, noise_std=0.35, anchor_slip_prob=0.03),
        "clarity": RaterProfile(bias=0.0, noise_std=0.35, anchor_slip_prob=0.03),
        "empathy": RaterProfile(bias=0.1, noise_std=0.45, anchor_slip_prob=0.05),
        "actionability": RaterProfile(bias=0.0, noise_std=0.35, anchor_slip_prob=0.03),
    },
    "model": {
        "accuracy": RaterProfile(bias=-0.3, noise_std=0.30, anchor_slip_prob=0.03),
        "clarity": RaterProfile(bias=0.0, noise_std=0.30, anchor_slip_prob=0.03),
        "empathy": RaterProfile(bias=-0.2, noise_std=0.60, anchor_slip_prob=0.10),
        "actionability": RaterProfile(bias=0.1, noise_std=0.35, anchor_slip_prob=0.04),
    },
}


def _draw_hidden_quality(rng: np.random.Generator, ambiguous: bool) -> float:
    """Hidden, continuous ground-truth quality in [0, 4]."""
    if ambiguous:
        # Centre on a half-integer boundary between two anchors (0.5, 1.5, 2.5, 3.5).
        boundary = rng.choice([0.5, 1.5, 2.5, 3.5])
        value = boundary + rng.normal(0.0, 0.12)
    else:
        # Beta(2, 2) rescaled to [0, 4]: skewed toward the middle, occasionally extreme.
        value = rng.beta(2.0, 2.0) * SCALE_MAX
    return float(np.clip(value, SCALE_MIN, SCALE_MAX))


def _score_pass(rng: np.random.Generator, hidden_quality: float, profile: RaterProfile) -> int:
    """One evaluator's one pass: round the noisy, biased perception of hidden_quality,
    with a small independent chance of slipping to an adjacent anchor."""
    perceived = hidden_quality + profile.bias + rng.normal(0.0, profile.noise_std)
    score = int(np.clip(round(perceived), SCALE_MIN, SCALE_MAX))
    if rng.random() < profile.anchor_slip_prob:
        score = int(np.clip(score + rng.choice([-1, 1]), SCALE_MIN, SCALE_MAX))
    return score


def generate(n_items: int, seed: int, ambiguous_share: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (scores_long, ground_truth).

    scores_long columns : item_id, dimension, rater, pass_num, score
    ground_truth columns: item_id, dimension, hidden_quality, is_ambiguous
    """
    score_rows: list[dict] = []
    truth_rows: list[dict] = []

    for item_idx in range(n_items):
        item_id = f"item_{item_idx:05d}"
        item_rng = np.random.default_rng([seed, item_idx, 0, 0, 0])
        is_ambiguous_item = item_rng.random() < ambiguous_share

        for dim_idx, dimension in enumerate(DIMENSION_NAMES):
            truth_rng = np.random.default_rng([seed, item_idx, dim_idx, 0, 0])
            hidden_quality = _draw_hidden_quality(truth_rng, ambiguous=is_ambiguous_item)
            truth_rows.append(
                {
                    "item_id": item_id,
                    "dimension": dimension,
                    "hidden_quality": round(hidden_quality, 4),
                    "is_ambiguous": is_ambiguous_item,
                }
            )

            for rater_idx, rater in enumerate(RATERS):
                profile = RATER_PROFILES[rater][dimension]
                for pass_num in range(1, NUM_PASSES + 1):
                    pass_rng = np.random.default_rng([seed, item_idx, dim_idx, rater_idx, pass_num])
                    score = _score_pass(pass_rng, hidden_quality, profile)
                    score_rows.append(
                        {
                            "item_id": item_id,
                            "dimension": dimension,
                            "rater": rater,
                            "pass_num": pass_num,
                            "score": score,
                        }
                    )

    return pd.DataFrame(score_rows), pd.DataFrame(truth_rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ambiguous-share", type=float, default=0.25)
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    scores, truth = generate(args.items, args.seed, args.ambiguous_share)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "raw_scores.csv", index=False)
    truth.to_csv(out / "ground_truth.csv", index=False)
    print(f"wrote {len(scores):,} scored passes ({args.items} items) -> {out / 'raw_scores.csv'}")
    print(f"wrote {len(truth):,} ground-truth rows -> {out / 'ground_truth.csv'}")


if __name__ == "__main__":
    main()
