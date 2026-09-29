"""Merges the main Qwen3-32B-AWQ pass output (3 shards) with the extra pass output (3 shards, the
"8B short-only policy" comparison over Qwen3-8B's disputed long-context resolutions) into ONE
synthetic "prior pass" file covering every item still `strategy == "not found"` across BOTH --
i.e. the true final residual after every deterministic and LLM stage run so far.

Deliberately does NOT use `run_llm_evidence_batch.not_found_items_from_prior_pass` for this:
that function OVERWRITES `by_accession[accession]` per file it reads, so an accession appearing in
more than one input file (possible here -- the two passes are disjoint at the ITEM level by
construction, but not necessarily at the ACCESSION level, since one accession can carry items in
both buckets) would silently lose whichever file's items got overwritten. This script MERGES
(extends) per accession across all six input files instead.

Run as a script:
    python build_final_unresolved_residual.py
"""

from __future__ import annotations

import json

from paths import DERIVED_DIR

MAIN_PASS_SHARDS = [DERIVED_DIR / f"llm_evidence_second_pass_qwen3_32b_v2_shard{i}.jsonl" for i in (0, 1, 2)]
EXTRA_PASS_SHARDS = [DERIVED_DIR / f"qwen3_8b_long_context_extra_pass_32b_shard{i}.jsonl" for i in (0, 1, 2)]
OUT = DERIVED_DIR / "final_unresolved_residual.jsonl"


def main() -> None:
    by_accession: dict[str, list[dict]] = {}
    n_files_missing = 0
    for path in MAIN_PASS_SHARDS + EXTRA_PASS_SHARDS:
        if not path.exists():
            print(f"WARNING: {path} does not exist -- skipping (make sure all 6 source files have finished)")
            n_files_missing += 1
            continue
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                not_found_rows = [row for row in rec["rows"] if row["strategy"] == "not found"]
                if not_found_rows:
                    by_accession.setdefault(rec["accession"], []).extend(not_found_rows)

    if n_files_missing:
        raise SystemExit(f"{n_files_missing} source file(s) missing -- aborting rather than building a partial residual")

    n_items = sum(len(rows) for rows in by_accession.values())
    with OUT.open("w") as f:
        for accession, rows in by_accession.items():
            f.write(json.dumps({"accession": accession, "rows": rows}) + "\n")

    print(f"Wrote {len(by_accession):,} accessions ({n_items:,} items) to {OUT}")


if __name__ == "__main__":
    main()
