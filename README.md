# rubric-eval-agreement

A small, runnable demonstration of **inter-rater agreement methodology** for rubric-based text
scoring: two blind evaluators, each scoring the same items three times ("precise mode"), scored
against written behavioral anchors on a 0-4 scale, measured with quadratic weighted kappa and
bootstrap confidence intervals, with a disagreement-diagnostics report pointing at exactly which
dimension and which items need calibration attention.

> **The data is synthetic**, and the domain (customer-support reply quality) is a deliberately
> generic stand-in. `rubric_eval/generate_data.py` simulates a hidden "true quality" per item and
> two evaluators with configurable bias and noise -- nothing here is a real dataset or a real
> evaluation system. The point of the repo is the **statistical methodology**: how agreement is
> measured, validated, and used to direct calibration effort.

## What it demonstrates

| Area | Where |
|---|---|
| Rubric with written behavioral anchors, 0-4 scale, 4 independent dimensions | `rubric_eval/rubric.py` |
| Blind double-scoring (two evaluators, no access to each other's scores) | `rubric_eval/generate_data.py` |
| "Precise mode": 3 independent passes per evaluator per item, with a documented, deterministic consensus rule | `rubric_eval/scoring.py` |
| Intra-rater consistency (does a single evaluator agree with themselves across passes?) | `rubric_eval/scoring.py::intra_rater_summary` |
| Quadratic weighted kappa, implemented **two independent ways** and cross-checked in tests | `rubric_eval/agreement.py` |
| Bootstrap confidence intervals (paired, item-level resampling) | `rubric_eval/agreement.py::bootstrap_kappa_ci` |
| Disagreement diagnostics: which dimension, which items, ranked for calibration review | `rubric_eval/report.py` |
| Validating the disagreement metric against injected ground truth (only possible because the data is synthetic) | `rubric_eval/report.py::validate_against_ground_truth` |
| 20 tests, including statistical sanity checks (CI width shrinks with more data, near-random ratings give near-zero kappa, systematic offset lowers kappa) | `tests/` |

## Why two independent kappa implementations?

`quadratic_weighted_kappa` is the standard confusion-matrix formulation used everywhere in the
pipeline. `quadratic_weighted_kappa_closed_form` computes the same quantity from a different
identity entirely (under independence, `E[(A-B)^2] = Var(A) + Var(B) + (mean(A)-mean(B))^2`), with
no confusion matrix at all. `tests/test_agreement.py` asserts the two agree to floating-point
precision on random data on every run -- this is a genuine cross-check that the implementation is
correct, not just internally self-consistent.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10-3.13
make setup     # pip install -r requirements.txt
make test      # 20 tests, incl. the two-kappa-implementations cross-check
make run       # generate data -> score -> agreement -> reports/agreement_report.md
```

Everything runs locally and finishes in a few seconds; there is no database or service to stand up.

## Example output (synthetic data, 300 items, seed 7)

Kappa by dimension, sorted ascending so the dimension needing calibration attention is listed first:

| dimension        | kappa | 95% CI        |
|---|---|---|
| empathy          | 0.749 | [0.710, 0.784] |
| overall (pooled) | 0.825 | [0.809, 0.840] |
| accuracy         | 0.834 | [0.805, 0.861] |
| clarity          | 0.856 | [0.827, 0.880] |
| actionability     | 0.857 | [0.827, 0.884] |

`empathy` comes out lowest by design in this synthetic dataset (the simulated "model" evaluator is
deliberately noisier on that dimension) -- and the validation section confirms disagreement
concentrates on the boundary items injected for that purpose: mean `|human - model|` is higher on
injected-ambiguous items than on clear-cut ones, on every dimension, e.g. 0.46 vs 0.27 for
`actionability`. That is the check that matters: a disagreement metric is only useful if it is
pointing at real ambiguity rather than noise spread evenly across the dataset.

## Design decisions

**Why quadratic weights.** A one-point miss (3 vs 4) matters far less than a two-point miss (2 vs
4). Unweighted kappa treats every mismatch the same; quadratic weights penalize large
disagreements more, which matches how a calibration review actually treats these errors.

**Why bootstrap the CI instead of a closed-form one.** Kappa's sampling distribution is not
well-approximated by a normal distribution at the sample sizes typical of a calibration review (tens
to low hundreds of items). A paired, item-level percentile bootstrap makes no distributional
assumption and is simple to audit.

**Why a deterministic tie-break on the 3-pass consensus.** "Take the mode" is ambiguous when three
passes are all different, or when there's a two-way tie. The documented rule (closest to the mean
of the three passes, then the smaller value) is auditable and tested directly (`tests/test_scoring.py`),
rather than silently depending on dict or set ordering.

**Why validate against ground truth at all**, given that a real deployment never has it: this
section exists specifically to sanity-check the methodology itself on synthetic data before
trusting it on real data, and the repo says so explicitly in the report rather than implying the
validation is available in production.

## Limitations

- The synthetic rater-noise model (independent Gaussian noise plus a small anchor-slip
  probability) is a simplification; real rater disagreement often has structure (e.g. a rater
  consistently harsher on a specific sub-topic) that this generator does not model.
- Kappa is computed assuming a fixed 5-category scale throughout; a rubric with different
  per-dimension scales would need the category count passed through rather than assumed.
- With only 4 dimensions and a handful of rater profiles, the "which dimension needs calibration"
  ranking is illustrative rather than a general-purpose dimension-selection method.

## Layout

```
rubric_eval/
  rubric.py          rubric definition: dimensions + written behavioral anchors
  generate_data.py    synthetic items, hidden ground truth, two blind evaluators x 3 passes
  scoring.py          precise-mode consensus (documented tie-break) + intra-rater consistency
  agreement.py        quadratic weighted kappa (two independent implementations) + bootstrap CI
  report.py           disagreement diagnostics + ground-truth validation + markdown report
  cli.py              end-to-end pipeline entry point
tests/                20 tests across all of the above
```
