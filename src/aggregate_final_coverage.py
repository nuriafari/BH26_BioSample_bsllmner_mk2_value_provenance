"""Combines the deterministic cascade's output with both LLM-tier passes into one final coverage
report -- the headline numbers (total values, resolved per stage, final unresolved) for the fully
regenerated pipeline (deterministic fixes + two-pass attributes/full-record LLM escalation).

Run as a script:
    python aggregate_final_coverage.py --qwen3-8b path/to/qwen3_8b.jsonl --qwen3-32b path/to/qwen3_32b.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paths import TRACE_BACK_FULL_PARQUET


def _count_llm_jsonl(path: Path) -> tuple[int, int]:
    """(n_items, n_resolved) across every accession's rows in one LLM-tier output file."""
    n_items, n_resolved = 0, 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for row in rec["rows"]:
                n_items += 1
                if row["strategy"] == "llm":
                    n_resolved += 1
    return n_items, n_resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3-8b", type=Path, required=True)
    parser.add_argument("--qwen3-32b", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    det = pd.read_parquet(TRACE_BACK_FULL_PARQUET, columns=["strategy"])
    total = len(det)
    det_resolved = int((det["strategy"] != "not found").sum())
    det_not_found = total - det_resolved

    q8_items, q8_resolved = _count_llm_jsonl(args.qwen3_8b)
    q32_items, q32_resolved = _count_llm_jsonl(args.qwen3_32b)

    final_unresolved = det_not_found - q8_resolved - q32_resolved

    report = {
        "total_extracted_values": total,
        "deterministic_resolved": det_resolved,
        "deterministic_resolved_pct": round(100 * det_resolved / total, 2),
        "deterministic_residual": det_not_found,
        "qwen3_8b_items_seen": q8_items,
        "qwen3_8b_resolved": q8_resolved,
        "qwen3_8b_resolved_pct_of_all": round(100 * q8_resolved / total, 2),
        "qwen3_32b_awq_items_seen": q32_items,
        "qwen3_32b_awq_resolved": q32_resolved,
        "qwen3_32b_awq_resolved_pct_of_all": round(100 * q32_resolved / total, 2),
        "final_unresolved": final_unresolved,
        "final_unresolved_pct": round(100 * final_unresolved / total, 2),
        "overall_coverage_pct": round(100 * (total - final_unresolved) / total, 2),
    }

    print(json.dumps(report, indent=2))
    out_path = args.out or (args.qwen3_32b.parent / "final_coverage_report.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
