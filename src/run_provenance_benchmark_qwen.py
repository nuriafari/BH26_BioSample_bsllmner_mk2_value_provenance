"""Runs Benchmark A (`paths.PROVENANCE_BENCHMARK_400_PARQUET`) through one Qwen model, under BOTH
context conditions, as two independent, COMPLETE passes over all 400 cases -- not the production
escalation (`run_llm_evidence_batch.run_two_pass`), which only sends a model's own residual to the
wider rendering. This is a controlled comparison: `attributes` shows every case the same
attributes/title/comment text the production Pass A shows; `full_record` shows every case the
complete record, regardless of whether the narrower view already resolved it. Comparing the two
conditions on cases where the reference evidence is known to live in a submitted attribute answers
whether giving a model less context measurably increases hallucination or degrades recall, versus
giving it the complete record from the start.

Loads one model per process (same reasoning as `run_llm_evidence_batch.py`: vLLM/CUDA state isn't
reliably released by swapping models within one process) -- run once per model:
    CUDA_VISIBLE_DEVICES=0 python run_provenance_benchmark_qwen.py --model Qwen/Qwen2.5-3B-Instruct
    CUDA_VISIBLE_DEVICES=0 python run_provenance_benchmark_qwen.py --model Qwen/Qwen3-8B
    CUDA_VISIBLE_DEVICES=0 python run_provenance_benchmark_qwen.py --model Qwen/Qwen3-32B-AWQ
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from build_benchmark_annotation_chunks import _raw_records_by_run
from llm_evidence import (
    MAX_OUTPUT_TOKENS,
    NUM_CTX,
    default_sampling_params,
    load_engine,
    verify_not_found_with_llm_batch,
)
from paths import PROVENANCE_BENCHMARK_400_PARQUET, PROVENANCE_BENCHMARK_400_RESULTS_DIR

OUT_DIR = PROVENANCE_BENCHMARK_400_RESULTS_DIR


def run_one_condition(
    sample: pd.DataFrame,
    raw_by_accession: dict,
    llm,
    sampling_params,
    full_record: bool,
) -> list[dict]:
    """One batched `llm.chat()` call covering all 400 cases under a single context condition."""
    records = [
        (raw_by_accession[row.accession], [(row.target_field, row.target_value, None)])
        for row in sample.itertuples(index=False)
    ]
    results = verify_not_found_with_llm_batch(
        records, llm, sampling_params, full_record=full_record
    )

    rows = []
    for row, (item_rows, call_info) in zip(sample.itertuples(index=False), results):
        [item_row] = item_rows  # exactly one item per record here
        rows.append(
            {
                "case_index": row.case_index,
                "accession": row.accession,
                "target_field": row.target_field,
                "target_value": row.target_value,
                "condition": "full_record" if full_record else "attributes",
                "llm_verdict": item_row["llm_verdict"],
                "accepted": item_row["strategy"] == "llm",
                "raw_fields": item_row["raw_fields"],
                "raw_matched_phrase": item_row["raw_matched_phrase"],
                "llm_ungrounded_quotes": item_row["llm_ungrounded_quotes"],
                "n_matches": item_row["n_matches"],
                "call_error": call_info.get("error"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="lower than load_engine's own 0.85 default -- this benchmark's models (up to 32B-AWQ) don't need 85% "
        "of a 40GB GPU, and a lower request tolerates orphaned memory left behind by an earlier crashed run",
    )
    args = parser.parse_args()

    sample = pd.read_parquet(PROVENANCE_BENCHMARK_400_PARQUET)
    # row position, not a stored column -- matches `build_benchmark_annotation_chunks.py`'s
    # `case_index` (also just each row's position in this same freshly-read parquet), so Qwen's
    # rows and Claude's chunk verdicts line up on the same key
    sample["case_index"] = range(len(sample))
    print(
        f"Loaded {len(sample)} benchmark cases. Extracting raw records...", flush=True
    )
    raw_by_accession = _raw_records_by_run(sample)

    print(
        f"Loading vLLM engine for {args.model} (model load + CUDA graph capture -- a few minutes)...",
        flush=True,
    )
    llm = load_engine(args.model, gpu_memory_utilization=args.gpu_memory_utilization)
    sampling_params = default_sampling_params()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "__")
    out_path = OUT_DIR / f"{model_slug}.jsonl"
    manifest_path = OUT_DIR / f"{model_slug}.meta.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "temperature": sampling_params.temperature,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "max_model_len": NUM_CTX,
                "n_cases": len(sample),
                "conditions": ["attributes", "full_record"],
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )

    all_rows: list[dict] = []
    for full_record in (False, True):
        condition = "full_record" if full_record else "attributes"
        print(
            f"Running condition={condition} over all {len(sample)} cases (one batched call)...",
            flush=True,
        )
        rows = run_one_condition(
            sample, raw_by_accession, llm, sampling_params, full_record
        )
        all_rows.extend(rows)
        n_accepted = sum(r["accepted"] for r in rows)
        print(
            f"  condition={condition}: {n_accepted}/{len(rows)} accepted (found + all quotes grounded)",
            flush=True,
        )

    with out_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    print(
        f"Wrote {len(all_rows)} rows ({len(sample)} cases x 2 conditions) to {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
