"""Per-target-attribute breakdown of deterministic matches excluded purely for being negated.

A value in the deterministic residual (`strategy == "not found"`) can still carry `row_negated` and
a positive `n_matches`: this means a textual match existed but was discarded because every retained
match sat in a negated context (see `bs_entries.verify_extracted_against_raw_rows`). This script
reports how that population breaks down across the nine target attributes.

Run as a script:
    python negation_breakdown_by_attribute.py
"""

from __future__ import annotations

import pandas as pd

from paths import TRACE_BACK_FULL_PARQUET


def main() -> None:
    df = pd.read_parquet(
        TRACE_BACK_FULL_PARQUET,
        columns=["target_field", "strategy", "row_negated", "n_matches"],
    )
    residual = df[df["strategy"] == "not found"]
    negated_excluded = residual[residual["row_negated"] & (residual["n_matches"] > 0)]

    by_attr = negated_excluded.groupby("target_field").size()
    residual_by_attr = residual.groupby("target_field").size()
    report = (
        pd.DataFrame(
            {
                "negated_excluded": by_attr,
                "residual_total": residual_by_attr,
                "pct_of_residual": (100 * by_attr / residual_by_attr).round(2),
            }
        )
        .fillna(0)
        .sort_values("negated_excluded", ascending=False)
    )

    print(
        f"Overall: {len(negated_excluded):,} / {len(residual):,} ({100 * len(negated_excluded) / len(residual):.2f}%) "
        "of the deterministic residual are matches excluded purely for being negated."
    )
    print(report.to_string())


if __name__ == "__main__":
    main()
