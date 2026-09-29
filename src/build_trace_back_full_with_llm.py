"""Extends `trace_back_full.parquet` (deterministic-cascade evidence only, 96.1% of extracted
values) with the raw field-name evidence the LLM stages additionally grounded, so the field-name
provenance analysis (which raw BioSample field names supplied evidence for each target attribute)
can cover all resolved values, not just the deterministic ones.

For every `(accession, target_field, target_value)` triple the deterministic cascade left
`strategy == "not found"`, if an LLM stage accepted it, that row's `strategy`/`raw_fields`/
`raw_texts`/`raw_matched_phrase`/`n_matches` are REPLACED with the LLM stage's values -- every
other column (accession metadata, ontology assignment, negation flags) is left untouched. This
mirrors the "deterministic -> Qwen3-8B -> Qwen3-32B-AWQ" cascade used throughout the paper:
Qwen3-8B's contribution is its attributes-pass resolutions only; Qwen3-32B-AWQ's is everything
else that ended up resolved (its own residual pass, its corroboration of Qwen3-8B's full-record
candidates, and its final long-context pass over what was still unresolved after that) -- see
`reconcile_final_pipeline_numbers.py`, which this script's accounting must match exactly.

Run as a script:
    python build_trace_back_full_with_llm.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paths import DERIVED_DIR, TRACE_BACK_FULL_PARQUET

QWEN3_8B = DERIVED_DIR / "llm_evidence_full_qwen3_8b_v2.jsonl"
MAIN_32B_SHARDS = [
    DERIVED_DIR / f"llm_evidence_second_pass_qwen3_32b_v2_shard{i}.jsonl"
    for i in (0, 1, 2)
]
EXTRA_PASS_32B_SHARDS = [
    DERIVED_DIR / f"qwen3_8b_long_context_extra_pass_32b_shard{i}.jsonl"
    for i in (0, 1, 2)
]
FINAL_EXPLAIN_32B_SHARDS = [
    DERIVED_DIR / f"final_explain_32b_shard{i}.jsonl" for i in (0, 1, 2)
]
OUT = DERIVED_DIR / "trace_back_full_with_llm.parquet"

_ROW_FIELDS = ["raw_fields", "raw_texts", "raw_matched_phrase", "n_matches"]
_Key = tuple[str, str, str]


def _accepted_rows(
    paths: list[Path], *, only_pass: str | None = None
) -> dict[_Key, dict]:
    """(accession, target_field, target_value) -> the LLM's raw-evidence fields, for every
    accepted (`strategy == "llm"`) row across `paths`. `only_pass` restricts to one `llm_pass`
    value (used to keep only Qwen3-8B's attributes-pass rows, not its superseded full-record ones).
    """
    resolved: dict[_Key, dict] = {}
    for path in paths:
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                accession = rec["accession"]
                for row in rec["rows"]:
                    if row["strategy"] != "llm":
                        continue
                    if only_pass is not None and row["llm_pass"] != only_pass:
                        continue
                    key = (accession, row["target_field"], row["target_value"])
                    resolved[key] = {field: row[field] for field in _ROW_FIELDS}
    return resolved


def main() -> None:
    det = pd.read_parquet(TRACE_BACK_FULL_PARQUET)
    total = len(det)
    key_cols = ["accession", "target_field", "target_value"]
    n_dup_keys = det.duplicated(key_cols, keep=False).sum()
    if n_dup_keys:
        # pre-existing data quirk (verbatim repeated extraction rows, always with matching
        # strategy across the duplicates), not introduced by this merge -- a left join against
        # the LLM tables' unique keys stays row-count-safe regardless, so this is informational.
        print(
            f"Note: {n_dup_keys:,} rows share a duplicated (accession, target_field, target_value) key in the input."
        )

    sources = [
        (
            "Qwen3-8B (attributes pass)",
            _accepted_rows([QWEN3_8B], only_pass="attributes"),
        ),
        ("Qwen3-32B-AWQ (main pass)", _accepted_rows(MAIN_32B_SHARDS)),
        (
            "Qwen3-32B-AWQ (corroboration of Qwen3-8B's full-record pass)",
            _accepted_rows(EXTRA_PASS_32B_SHARDS),
        ),
        (
            "Qwen3-32B-AWQ (final long-context pass)",
            _accepted_rows(FINAL_EXPLAIN_32B_SHARDS),
        ),
    ]

    merged: dict[_Key, dict] = {}
    for label, rows in sources:
        overlap = merged.keys() & rows.keys()
        if overlap:
            raise SystemExit(
                f"{label}: {len(overlap):,} triples were already resolved by an earlier stage -- investigate before merging"
            )
        print(f"{label}: {len(rows):,} accepted rows")
        merged.update(rows)
    print(f"Total LLM-resolved rows collected: {len(merged):,}")

    llm_df = pd.DataFrame(
        [
            {"accession": a, "target_field": tf, "target_value": tv, **row}
            for (a, tf, tv), row in merged.items()
        ]
    ).rename(columns={field: f"{field}_llm" for field in _ROW_FIELDS})

    det = det.merge(llm_df, on=key_cols, how="left")
    assert len(det) == total, (
        "merge changed row count -- key uniqueness assumption violated somewhere"
    )

    to_update = (det["strategy"] == "not found") & det["raw_fields_llm"].notna()
    n_updated = int(to_update.sum())
    n_updated_distinct_keys = det.loc[to_update, key_cols].drop_duplicates().shape[0]
    assert n_updated_distinct_keys == len(merged), (
        f"expected {len(merged):,} distinct resolved keys, actually updated {n_updated_distinct_keys:,} distinct keys "
        f"({n_updated:,} rows, reflecting duplicate input rows sharing a resolved key)"
    )

    det.loc[to_update, "strategy"] = "llm"
    for field in _ROW_FIELDS:
        det.loc[to_update, field] = det.loc[to_update, f"{field}_llm"]
    det = det.drop(columns=[f"{field}_llm" for field in _ROW_FIELDS])

    print(f"\nTotal rows: {len(det):,}")
    print(det["strategy"].value_counts().to_string())
    n_not_found = int((det["strategy"] == "not found").sum())
    print(f"\nFinal not-found: {n_not_found:,} ({100 * n_not_found / total:.2f}%)")

    det.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
