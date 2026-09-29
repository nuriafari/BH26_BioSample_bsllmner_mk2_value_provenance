"""Prepares Benchmark A (`paths.PROVENANCE_BENCHMARK_400_PARQUET`) for independent annotation by
Claude Code subagents: pulls each sampled case's raw BioSample record and writes fixed-size chunk
JSON files an annotating subagent reads directly, rather than having each subagent re-scan the
crate's `inputs/*.jsonl` files itself.

Run as a script:
    python build_benchmark_annotation_chunks.py --out-dir /path/to/scratch/chunks --chunk-size 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bs_entries import get_accession, iter_bs_entries
from paths import CRATE_INPUTS_DIR, PROVENANCE_BENCHMARK_400_PARQUET, RUN_INDEX_TSV


def _raw_records_by_run(sample: pd.DataFrame) -> dict[str, dict]:
    """One scan per distinct `run_name` in the sample (not one scan per accession) -- collects
    just the accessions this benchmark actually needs from each run's input file.
    """
    run_index = pd.read_csv(RUN_INDEX_TSV, sep="\t").set_index("run_name")
    raw_by_accession: dict[str, dict] = {}
    for run_name, group in sample.groupby("run_name"):
        needed = set(group["accession"])
        run_row = run_index.loc[run_name]
        input_path = CRATE_INPUTS_DIR / run_row["dataset"] / run_row["input_file"]
        for raw in iter_bs_entries(input_path):
            accession = get_accession(raw)
            if accession in needed:
                raw_by_accession[accession] = raw
                needed.discard(accession)
                if not needed:
                    break
        if needed:
            raise ValueError(f"{len(needed)} accession(s) not found in {input_path}: {sorted(needed)[:5]}")
    return raw_by_accession


def build_chunks(out_dir: Path, chunk_size: int) -> list[Path]:
    sample = pd.read_parquet(PROVENANCE_BENCHMARK_400_PARQUET)
    raw_by_accession = _raw_records_by_run(sample)

    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    for start in range(0, len(sample), chunk_size):
        chunk = sample.iloc[start : start + chunk_size]
        cases = [
            {
                "case_index": start + i,
                "accession": row.accession,
                "target_field": row.target_field,
                "target_value": row.target_value,
                "raw_biosample_record": raw_by_accession[row.accession],
            }
            for i, row in enumerate(chunk.itertuples(index=False))
        ]
        chunk_path = out_dir / f"chunk_{start:04d}.json"
        chunk_path.write_text(json.dumps(cases, indent=2))
        chunk_paths.append(chunk_path)
    return chunk_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    args = parser.parse_args()

    chunk_paths = build_chunks(args.out_dir, args.chunk_size)
    print(f"Wrote {len(chunk_paths)} chunk files ({args.chunk_size} cases each) to {args.out_dir}")


if __name__ == "__main__":
    main()
