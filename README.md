# rubric-eval-agreement

### Abstract

A runnable demonstration of inter-rater agreement methodology for rubric-based text scoring. Two
blind evaluators score the same items against written behavioral anchors on a 0-4 scale, each item
scored three independent times ("precise mode"). Agreement is measured with quadratic weighted
kappa and bootstrap confidence intervals, and a disagreement-diagnostics step identifies exactly
which rubric dimension and which items need calibration attention. All data is synthetic, generated
by `rubric_eval/generate_data.py` with a hidden ground-truth quality per item; the domain (customer
support reply quality) is a deliberately generic stand-in. The point of the repo is the
statistical methodology, not the dataset.

### Method

1. **Rubric.** Four dimensions (`accuracy`, `clarity`, `empathy`, `actionability`), each with
   written behavioral anchors for scores 0 through 4 (`rubric_eval/rubric.py`).
2. **Blind double-scoring.** Two simulated evaluators, `human` and `model`, score every item with
   no access to each other's scores.
3. **Precise mode.** Each evaluator scores each (item, dimension) three independent times. A
   documented, deterministic rule resolves the three passes to one consensus score: take the mode;
   on a tie, take the value closest to the mean of the three; on a further tie, take the smaller
   value (`rubric_eval/scoring.py`).
4. **Agreement.** Quadratic weighted kappa is implemented two independent ways — the standard
   confusion-matrix formulation, and a closed-form identity derived from
   `E[(A-B)^2] = Var(A) + Var(B) + (mean(A)-mean(B))^2` under independence, with no confusion
   matrix at all. The test suite asserts the two agree to floating-point precision on random data
   every run, a genuine cross-check rather than a self-consistency check
   (`rubric_eval/agreement.py`).
5. **Validation.** Because the data is synthetic, a quarter of items are deliberately placed on a
   boundary between two anchors. The report checks that disagreement concentrates on those items
   rather than being noise spread evenly across the dataset — i.e. that the disagreement metric is
   measuring something real (`rubric_eval/report.py`).

### Reproducing these results

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10-3.13
make setup   # pip install -r requirements.txt
make test    # 20 tests, incl. the two-kappa cross-check
make run     # generate -> score -> agreement -> reports/agreement_report.md
```

### Results (300 items, seed 7)

| dimension        | kappa | 95% CI        |
|---|---|---|
| empathy          | 0.749 | [0.710, 0.784] |
| overall (pooled) | 0.825 | [0.809, 0.840] |
| accuracy         | 0.834 | [0.805, 0.861] |
| clarity          | 0.856 | [0.827, 0.880] |
| actionability    | 0.857 | [0.827, 0.884] |

`empathy` is lowest by construction: the simulated `model` evaluator is deliberately noisier on
that dimension. Mean `|human - model|` is higher on the injected boundary items than on clear-cut
ones for every single dimension (e.g. 0.46 vs 0.27 on `actionability`), confirming the
disagreement metric is pointing at genuine ambiguity.

### Threats to validity

- The rater-noise model (independent Gaussian noise plus a small anchor-slip probability) is a
  simplification; real disagreement often has structure this generator does not capture (e.g. a
  rater consistently harsher on one sub-topic).
- Kappa is computed on a fixed 5-category scale throughout; a rubric with per-dimension scales
  would need the category count threaded through rather than assumed.
- With 4 dimensions and a handful of rater profiles, "which dimension needs calibration" is
  illustrative here, not a general-purpose dimension-selection method.

### Repository map

`rubric_eval/{rubric,generate_data,scoring,agreement,report,cli}.py`, each with a matching file
under `tests/` (20 tests total).
