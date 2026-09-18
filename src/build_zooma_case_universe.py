"""Builds the sample of trace-back cases to cross-check against EBI Zooma.

Scope: every row with a real assigned term, from any strategy -- including
"exact", kept in as a baseline/validity check (its evidence text is by
construction identical to the assigned value, so it's not expected to
surface new *disagreements* the way the non-exact strategies can, but it's
cheap to include and confirms the method itself isn't systematically
biased). "not found" rows are excluded since they have no assigned term to
check; that exclusion is redundant with the assigned_term_id filter below,
kept only for readability. Each row's raw_fields/raw_texts/raw_matched_phrase
are parallel arrays (a target can have multiple evidence citations), so
rows are exploded first (and the array columns renamed to their singular
form, matching field_diversity.explode_raw_fields' own convention). A
"case" is then deduplicated on
(raw_field, raw_matched_phrase, assigned_term_id): the same literal evidence
phrase, in the same kind of source field, judged to support the same term,
is the same test regardless of which accession it came from -- this is what
keeps the case universe (tens of thousands) so much smaller than the raw
row count (millions): Zooma's answer to identical text is deterministic, so
the same phrase repeated across many accessions is one case, not many.

Run once; re-run only if trace_back_full.parquet changes.
"""

from __future__ import annotations

import pandas as pd

from paths import DERIVED_DIR, TRACE_BACK_FULL_PARQUET, ZOOMA_CASE_UNIVERSE_PARQUET

STRATEGIES_EXCLUDED = {"not found"}


def build_case_universe() -> pd.DataFrame:
    df = pd.read_parquet(TRACE_BACK_FULL_PARQUET)

    # Restrict to rows with a real assigned term.
    df = df[~df["strategy"].isin(STRATEGIES_EXCLUDED)]
    df = df[df["assigned_term_id"].notna() & (df["assigned_term_id"] != "")]

    # Explode each row's parallel evidence-citation arrays into one row per citation.
    df = df.explode(["raw_fields", "raw_texts", "raw_matched_phrase", "negated"])
    df = df[df["negated"] != True]
    df = df.rename(columns={"raw_fields": "raw_field", "raw_texts": "raw_text"})

    # Dedup: same evidence phrase, same kind of source field, same assigned term = same case.
    df["dedup_key"] = list(
        zip(df["raw_field"], df["raw_matched_phrase"], df["assigned_term_id"])
    )
    df = df.drop_duplicates("dedup_key").reset_index(drop=True)
    df["case_id"] = df.index.astype(str)
    return df


def main() -> None:
    cases = build_case_universe()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(ZOOMA_CASE_UNIVERSE_PARQUET, index=False)
    print(
        f"{len(cases)} unique cases across {cases['target_field'].nunique()} target fields "
        f"-> {ZOOMA_CASE_UNIVERSE_PARQUET}"
    )
    print(cases["target_field"].value_counts().to_string())


if __name__ == "__main__":
    main()
