"""End-to-end pipeline: generate -> precise-mode consensus -> agreement -> report.

    python -m rubric_eval.cli --items 300 --passes-note "see generate_data.NUM_PASSES"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rubric_eval import agreement, generate_data, report, scoring


def run(items: int, seed: int, ambiguous_share: float, n_boot: int, data_dir: str, report_path: str) -> None:
    scores_long, ground_truth = generate_data.generate(items, seed, ambiguous_share)

    data_dir_p = Path(data_dir)
    data_dir_p.mkdir(parents=True, exist_ok=True)
    scores_long.to_csv(data_dir_p / "raw_scores.csv", index=False)
    ground_truth.to_csv(data_dir_p / "ground_truth.csv", index=False)

    consensus = scoring.build_consensus_table(scores_long)
    intra_rater = scoring.intra_rater_summary(consensus)

    agreement_tbl = agreement.agreement_table(consensus, n_boot=n_boot, seed=seed)

    flagged = report.flag_disagreements(consensus)
    disagreement_tbl = report.disagreement_summary(flagged)
    validation_tbl = report.validate_against_ground_truth(flagged, ground_truth)

    markdown = report.render_markdown(agreement_tbl, intra_rater, disagreement_tbl, validation_tbl, flagged, n_items=items)
    report.write_report(report_path, markdown)

    print(markdown)
    print(f"\nreport written to {report_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ambiguous-share", type=float, default=0.25)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--report-path", default="reports/agreement_report.md")
    args = p.parse_args()
    run(args.items, args.seed, args.ambiguous_share, args.n_boot, args.data_dir, args.report_path)


if __name__ == "__main__":
    main()
