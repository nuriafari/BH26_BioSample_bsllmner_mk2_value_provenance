"""(Re)builds `data/derived/trace_back_full.parquet` -- the full-crate trace-back check, one row
per non-null extracted field value, across every completed run in the RO-Crate.

One worker process per run (`run_index.tsv`, 311 completed runs -- CPU-bound, embarrassingly
parallel across runs, no GPU involved), each writing its own chunk parquet under
`data/derived/trace_back_chunks/<run_name>.parquet` -- a run whose chunk already exists is skipped
on a re-run (resumable), and the final step concatenates every chunk into one file. The previous
`trace_back_full.parquet`, if any, is renamed to `.bak` rather than overwritten in place, so a
build that goes wrong doesn't destroy the only copy of the prior artifact.

Run as a script:
    python build_trace_back_full.py --workers 64
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass

import pandas as pd

from bs_entries import get_accession, iter_bs_entries, verify_extracted_against_raw_rows
from ontology import build_field_ontology_indexes
from paths import (
    CRATE_CONFIG_DIR,
    CRATE_INPUTS_DIR,
    CRATE_ONTOLOGY_DIR,
    CRATE_RESULTS_DIR,
    DERIVED_DIR,
    RUN_INDEX_TSV,
    TRACE_BACK_FULL_PARQUET,
)

CHUNKS_DIR = DERIVED_DIR / "trace_back_chunks"


@dataclass(frozen=True)
class Category:
    name: str
    organism: str
    dataset_prefix: str


# Mirrors the notebook's own CATEGORIES (biosample_input_output_pilot.ipynb) -- kept in sync
# manually since it's a small, stable, hand-written mapping already duplicated there as a
# notebook-local dataclass, not something worth centralizing into `paths.py` for one more caller.
CATEGORIES = [
    Category("DDBJ RNA-Seq", "Homo sapiens", "rnaseq_human"),
    Category("DDBJ RNA-Seq", "Mus musculus", "rnaseq_mouse"),
    Category("ChIP-Atlas", "Homo sapiens", "chipatlas_hg38"),
    Category("ChIP-Atlas", "Mus musculus", "chipatlas_mm10"),
]


def category_for_dataset(dataset: str) -> Category:
    for category in CATEGORIES:
        if dataset.startswith(category.dataset_prefix):
            return category
    raise ValueError(f"no category matches dataset {dataset!r}")


def process_run(run: dict) -> tuple[str, int]:
    """Runs entirely in one worker process. Returns `(run_name, n_rows)`. Loads its own
    select-config's ontology indexes locally (cheap -- a plain-text regex scan, `ontology.py`) and
    its own result file (a single `json.load`, per this crate's pilot file sizes) rather than
    sharing state across processes, since `multiprocessing.Pool` workers don't share memory anyway.
    """
    chunk_path = CHUNKS_DIR / f"{run['run_name']}.parquet"
    if chunk_path.exists():
        return run["run_name"], -1  # -1: already done, skipped (resumable)

    with (CRATE_CONFIG_DIR / run["select_config"]).open() as f:
        select_config = json.load(f)
    fields = list(select_config.get("fields", {}).keys())
    ontology_indexes = build_field_ontology_indexes(select_config, CRATE_ONTOLOGY_DIR, fields)

    with (CRATE_RESULTS_DIR / run["result_file"]).open("r", encoding="utf-8") as f:
        select_result = json.load(f)
    entries_by_accession = {e["extract"]["accession"]: e for e in select_result["entries"]}

    category = category_for_dataset(run["dataset"])
    input_path = CRATE_INPUTS_DIR / run["dataset"] / run["input_file"]

    rows: list[dict] = []
    for raw in iter_bs_entries(input_path):
        accession = get_accession(raw)
        entry = entries_by_accession.get(accession)
        if entry is None:
            continue
        extracted = entry["extract"]["extracted"]
        results = entry.get("results")
        record_rows = verify_extracted_against_raw_rows(raw, extracted, results, ontology_indexes)
        for row in record_rows:
            row["accession"] = accession
            row["run_name"] = run["run_name"]
            row["category"] = category.name
            row["organism"] = category.organism
            rows.append(row)

    df = pd.DataFrame(rows)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(chunk_path, index=False)
    return run["run_name"], len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(64, mp.cpu_count()))
    args = parser.parse_args()

    run_index = pd.read_csv(RUN_INDEX_TSV, sep="\t")
    runs = run_index[run_index["status"] == "completed"].to_dict("records")
    print(f"{len(runs)} completed runs, {args.workers} workers, chunks in {CHUNKS_DIR}", flush=True)

    with mp.Pool(args.workers) as pool:
        for i, (run_name, n_rows) in enumerate(pool.imap_unordered(process_run, runs), 1):
            status = "skipped (already done)" if n_rows == -1 else f"{n_rows:,} rows"
            print(f"[{i}/{len(runs)}] {run_name}: {status}", flush=True)

    chunk_files = sorted(CHUNKS_DIR.glob("*.parquet"))
    missing = {r["run_name"] for r in runs} - {p.stem for p in chunk_files}
    if missing:
        print(f"WARNING: {len(missing)} runs have no chunk file, excluded from the merge: {sorted(missing)[:10]}", flush=True)

    combined = pd.concat([pd.read_parquet(p) for p in chunk_files], ignore_index=True)
    if TRACE_BACK_FULL_PARQUET.exists():
        backup = TRACE_BACK_FULL_PARQUET.with_suffix(".parquet.bak")
        TRACE_BACK_FULL_PARQUET.rename(backup)
        print(f"Backed up previous artifact to {backup}", flush=True)
    combined.to_parquet(TRACE_BACK_FULL_PARQUET, index=False)
    print(f"Wrote {len(combined):,} rows across {len(chunk_files)} runs to {TRACE_BACK_FULL_PARQUET}", flush=True)


if __name__ == "__main__":
    main()
