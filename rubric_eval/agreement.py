"""Inter-rater agreement: quadratic weighted kappa (QWK) and bootstrap CIs.

Two independent implementations of QWK are provided on purpose:

  * `quadratic_weighted_kappa`            -- the standard confusion-matrix
    formulation (Cohen's weighted kappa with quadratic weights). This is what
    gets used everywhere in the pipeline.

  * `quadratic_weighted_kappa_closed_form` -- the same quantity derived from a
    different identity: under independence, E[(A-B)^2] = Var(A) + Var(B) +
    (mean(A)-mean(B))^2, so
        kappa = 1 - mean((a-b)^2) / (Var(A) + Var(B) + (mean(A)-mean(B))^2)
    computed directly from the raw score arrays, with no confusion matrix at
    all. `tests/test_agreement.py` asserts the two agree to floating-point
    precision on random data -- a cross-check that the confusion-matrix
    implementation is correct, not just internally consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rubric_eval.rubric import NUM_CATEGORIES, SCALE_MIN


def confusion_matrix(ratings_a: np.ndarray, ratings_b: np.ndarray, num_categories: int = NUM_CATEGORIES) -> np.ndarray:
    conf = np.zeros((num_categories, num_categories), dtype=float)
    for a, b in zip(ratings_a - SCALE_MIN, ratings_b - SCALE_MIN):
        conf[int(a), int(b)] += 1
    return conf


def quadratic_weighted_kappa(ratings_a: np.ndarray, ratings_b: np.ndarray, num_categories: int = NUM_CATEGORIES) -> float:
    ratings_a = np.asarray(ratings_a)
    ratings_b = np.asarray(ratings_b)
    n = len(ratings_a)
    if n == 0:
        return float("nan")

    conf = confusion_matrix(ratings_a, ratings_b, num_categories)
    idx = np.arange(num_categories)
    weights = (idx[:, None] - idx[None, :]) ** 2

    row_marginal = conf.sum(axis=1)
    col_marginal = conf.sum(axis=0)
    expected = np.outer(row_marginal, col_marginal) / n

    observed_disagreement = float((weights * conf).sum())
    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement == 0:
        # No variance under independence (e.g. every item had the same score):
        # kappa is conventionally defined as 1.0 when there is also no observed
        # disagreement, and undefined otherwise.
        return 1.0 if observed_disagreement == 0 else float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def quadratic_weighted_kappa_closed_form(ratings_a: np.ndarray, ratings_b: np.ndarray) -> float:
    a = np.asarray(ratings_a, dtype=float)
    b = np.asarray(ratings_b, dtype=float)
    if len(a) == 0:
        return float("nan")
    mean_sq_diff = float(np.mean((a - b) ** 2))
    # population variance (ddof=0) -- matches the confusion-matrix derivation
    denom = float(a.var(ddof=0) + b.var(ddof=0) + (a.mean() - b.mean()) ** 2)
    if denom == 0:
        return 1.0 if mean_sq_diff == 0 else float("nan")
    return 1.0 - mean_sq_diff / denom


def bootstrap_kappa_ci(
    ratings_a: np.ndarray,
    ratings_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired (item-level) percentile bootstrap CI for QWK.

    Returns (point_estimate, lower, upper).
    """
    ratings_a = np.asarray(ratings_a)
    ratings_b = np.asarray(ratings_b)
    n = len(ratings_a)
    point = quadratic_weighted_kappa(ratings_a, ratings_b)

    rng = np.random.default_rng(seed)
    boot_values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_values[i] = quadratic_weighted_kappa(ratings_a[idx], ratings_b[idx])

    boot_values = boot_values[~np.isnan(boot_values)]
    lower = float(np.quantile(boot_values, alpha / 2))
    upper = float(np.quantile(boot_values, 1 - alpha / 2))
    return point, lower, upper


def agreement_table(consensus_table: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """consensus_table: item_id, dimension, rater, consensus_score (long format,
    one row per item x dimension x rater -- see scoring.build_consensus_table).

    Returns one row per dimension (plus one 'overall' row pooling every
    dimension together) with QWK and its bootstrap CI between human and model.
    """
    wide = consensus_table.pivot(index=["item_id", "dimension"], columns="rater", values="consensus_score").reset_index()

    rows = []
    for dimension, g in wide.groupby("dimension"):
        point, lower, upper = bootstrap_kappa_ci(g["human"].to_numpy(), g["model"].to_numpy(), n_boot=n_boot, seed=seed)
        rows.append({"dimension": dimension, "n_items": len(g), "kappa": point, "ci_low": lower, "ci_high": upper})

    point, lower, upper = bootstrap_kappa_ci(wide["human"].to_numpy(), wide["model"].to_numpy(), n_boot=n_boot, seed=seed)
    rows.append({"dimension": "overall (pooled)", "n_items": len(wide), "kappa": point, "ci_low": lower, "ci_high": upper})

    return pd.DataFrame(rows).sort_values("kappa").reset_index(drop=True)
