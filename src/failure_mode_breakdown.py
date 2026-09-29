"""Failure-mode categorization for the 400-case provenance benchmark (replaces the stale S7 table
computed on the old, superseded 200-case sample).

Unlike that earlier categorization, this one was not a separate pass: `gap_type` and
`gap_type_note` were collected directly during the same Claude Opus 5 reference-annotation pass
that produced `verdict` (found / not_found), for all 400 cases -- not only the ones judged
"not_found". A `gap_type` therefore describes why literal string matching would miss the value
even for a `verdict == "found"` case (real evidence exists in the record, just not as a literal
substring), while for `verdict == "not_found"` cases it describes why Claude itself could not
locate any support.

Run as a script:
    python failure_mode_breakdown.py
"""

from __future__ import annotations

import pandas as pd

from paths import PROVENANCE_BENCHMARK_400_RESULTS_DIR

CLAUDE_REFERENCE_PARQUET = (
    PROVENANCE_BENCHMARK_400_RESULTS_DIR / "claude_reference.parquet"
)


def main() -> None:
    ref = pd.read_parquet(CLAUDE_REFERENCE_PARQUET)
    n_total = len(ref)

    print(f"Verdict counts (of {n_total}):")
    print(ref["verdict"].value_counts())
    print()

    counts = ref["gap_type"].value_counts()
    pct = (100 * counts / n_total).round(1)
    breakdown = pd.DataFrame({"n": counts, "pct_of_400": pct})
    print(f"Failure-mode / gap-type breakdown (all {n_total} cases):")
    print(breakdown.to_string())
    print()

    not_found = ref[ref["verdict"] != "found"]
    print(
        f"Of the {len(not_found)} cases Claude itself judged not_found, gap_type breakdown:"
    )
    print(not_found["gap_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
