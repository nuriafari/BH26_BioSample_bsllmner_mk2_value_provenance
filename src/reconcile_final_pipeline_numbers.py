"""Computes the final, fully-reconciled headline coverage numbers for the paper.

The production LLM stages were run as several separate files rather than one clean pass (Qwen3-8B
short+long escalation, a main Qwen3-32B-AWQ pass over 8B's residual, an "extra pass" re-checking
8B's long-context catches, and a final long-context-only pass over whatever was still unresolved).
Each item is scored independently of batch/file membership, so these can be combined after the
fact into the same totals a single clean run would have produced -- this script does that combining
directly from the raw output files rather than restating numbers from memory.

Accounting used here (matches how the paper describes the cascade -- deterministic -> Qwen3-8B ->
Qwen3-32B-AWQ, with no short/long or cross-check detail):
- Qwen3-8B's number is its short-pass ("attributes") resolutions only -- the only ones it reached
  unconditionally, without any Qwen3-32B-AWQ involvement.
- Qwen3-32B-AWQ's number is everything else that ended up resolved: its main pass over 8B's
  residual, its corroboration of 8B's long-pass candidates (the "extra pass"), and its final
  long-context-only pass over whatever was still unresolved after that.

Run as a script:
    python reconcile_final_pipeline_numbers.py
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


def _count_resolved(paths: list[Path]) -> tuple[int, int]:
    """(n_items_seen, n_resolved) across one or more llm_evidence-style output files."""
    n_items, n_resolved = 0, 0
    for path in paths:
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                for row in rec["rows"]:
                    n_items += 1
                    if row["strategy"] == "llm":
                        n_resolved += 1
    return n_items, n_resolved


def main() -> None:
    det = pd.read_parquet(TRACE_BACK_FULL_PARQUET, columns=["strategy"])
    total = len(det)
    det_resolved = int((det["strategy"] != "not found").sum())
    residual = total - det_resolved

    n_8b_short = 0
    n_8b_long_total = 0
    with QWEN3_8B.open() as f:
        for line in f:
            rec = json.loads(line)
            for row in rec["rows"]:
                if row["strategy"] == "llm":
                    if row["llm_pass"] == "attributes":
                        n_8b_short += 1
                    else:
                        n_8b_long_total += 1

    _, main_32b_resolved = _count_resolved(MAIN_32B_SHARDS)
    n_extra_items, extra_32b_resolved = _count_resolved(EXTRA_PASS_32B_SHARDS)
    n_final_items, final_explain_resolved = _count_resolved(FINAL_EXPLAIN_32B_SHARDS)

    assert n_extra_items == n_8b_long_total, (
        f"extra-pass item count {n_extra_items} should equal 8B's long-pass claim count {n_8b_long_total}"
    )
    pre_final_pass_unresolved = (
        residual - n_8b_short - main_32b_resolved - extra_32b_resolved
    )
    assert n_final_items == pre_final_pass_unresolved, (
        f"final-explain-pass item count {n_final_items} should equal the residual entering it "
        f"{pre_final_pass_unresolved}"
    )

    qwen3_32b_total = main_32b_resolved + extra_32b_resolved + final_explain_resolved
    llm_total = n_8b_short + qwen3_32b_total
    final_unresolved = residual - llm_total

    report = {
        "total_extracted_values": total,
        "deterministic_resolved": det_resolved,
        "deterministic_resolved_pct": round(100 * det_resolved / total, 1),
        "deterministic_residual": residual,
        "deterministic_residual_pct": round(100 * residual / total, 1),
        "qwen3_8b_resolved": n_8b_short,
        "qwen3_8b_resolved_pct": round(100 * n_8b_short / total, 1),
        "qwen3_32b_awq_resolved": qwen3_32b_total,
        "qwen3_32b_awq_resolved_pct": round(100 * qwen3_32b_total / total, 1),
        "qwen3_32b_awq_breakdown": {
            "main_pass": main_32b_resolved,
            "corroboration_of_8b_long_pass": extra_32b_resolved,
            "final_long_context_pass": final_explain_resolved,
        },
        "final_unresolved": final_unresolved,
        "final_unresolved_pct": round(100 * final_unresolved / total, 2),
        "overall_coverage_pct": round(100 * (total - final_unresolved) / total, 2),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
