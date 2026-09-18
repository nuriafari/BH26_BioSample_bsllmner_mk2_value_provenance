"""Runs the LLM evidence tier (`llm_evidence.verify_not_found_with_llm_batch`) over a random sample
of the crate's "not found" residual, unattended and time-boxed -- meant to run for hours on a GPU
this project already has, not interactively from a notebook cell.

Processes accessions in batches through vLLM (`--batch-size` records per `llm.chat()` call), not
one call at a time or via a thread pool: vLLM batches internally (continuous batching plus
PagedAttention), so handing it many prompts per call is what actually gets the throughput vLLM is
for -- measured directly, ~23 calls/sec on one A100 for this model with real prompts, versus
Ollama's ~0.2-0.5 calls/sec that motivated dropping it (see `llm_evidence` module docstring). GPU
selection is by `CUDA_VISIBLE_DEVICES` on the process itself (set by the shell launching it), one
process per GPU -- this script doesn't know anything about multi-GPU dispatch, same as before.

Writes one JSON line per BioSample accession to `data/derived/llm_evidence_sample.jsonl` as each
batch completes (flushed immediately), so a run that's killed partway still leaves a valid,
directly-usable partial result -- and re-running with the same `--out` path skips accessions
already in that file, so an interrupted run resumes rather than restarting.

Run as a script:
    CUDA_VISIBLE_DEVICES=0 python run_llm_evidence_batch.py --sample-size 8000 --max-hours 4 --seed 0

Omitting `--sample-size` processes the ENTIRE not-found residual instead of a sample. `--shard-count`/
`--shard-index` split that residual (by a stable hash of the accession, not a slice of the sampled
order, so shards stay fixed across runs) across independent invocations -- e.g. one per GPU, each
with its own `--out` file and `CUDA_VISIBLE_DEVICES`, run concurrently.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path
from typing import Any

import pandas as pd

from llm_evidence import (
    DEFAULT_MODEL,
    default_sampling_params,
    load_engine,
    verify_not_found_with_llm_batch,
)
from paths import CRATE_INPUTS_DIR, DERIVED_DIR, RUN_INDEX_TSV, TRACE_BACK_FULL_PARQUET

DEFAULT_OUT = DERIVED_DIR / "llm_evidence_sample.jsonl"


def _build_offset_index(path: Path) -> dict[str, int]:
    """Byte offset of each record's line in one `inputs/*.jsonl` file, built by a single scan.
    `bs_entries.find_bs_entry` scans the whole file per lookup (fine for its original notebook,
    per-record use -- see its own docstring), but at this scale a shard's batches revisit the same
    ~300 input files across 100K+ accessions, so re-scanning per accession dominates wall time
    (measured directly: throughput dropped from ~8/s to ~5/s over the run as batches spread across
    more files). Binary mode so offsets are exact regardless of multi-byte UTF-8 content.
    """
    index = {}
    with path.open("rb") as f:
        offset = f.tell()
        for line in f:
            stripped = line.strip()
            if stripped:
                accession = json.loads(stripped).get("accession")
                if isinstance(accession, str):
                    # setdefault, not assignment -- a handful of files have a duplicate accession
                    # (re-fetched at a later date); match `find_bs_entry`'s first-occurrence-wins
                    # behavior rather than silently switching to last-wins.
                    index.setdefault(accession, offset)
            offset = f.tell()
    return index


class RawRecordCache:
    """Caches each input file's accession -> byte-offset index (built once, on first touch), so
    looking up an accession already seen in this file costs one seek plus parsing its own line,
    not a rescan. Persists for the life of the process -- a long shard run touches the same files
    over and over, so the index-build cost amortizes across the whole 100K+-accession shard.
    """

    def __init__(self) -> None:
        self._indexes: dict[Path, dict[str, int]] = {}

    def get(self, path: Path, accession: str) -> dict[str, Any] | None:
        index = self._indexes.get(path)
        if index is None:
            index = _build_offset_index(path)
            self._indexes[path] = index
        offset = index.get(accession)
        if offset is None:
            return None
        with path.open("rb") as f:
            f.seek(offset)
            line = f.readline()
        return json.loads(line)


def _shard_of(accession: str, shard_count: int) -> int:
    """Stable hash of the accession itself (not its position in any sampled/shuffled order), so a
    shard's membership never shifts if the underlying parquet's row order changes between runs.
    """
    return zlib.crc32(accession.encode()) % shard_count


def sample_not_found_by_accession(
    sample_size: int | None, seed: int, shard_index: int = 0, shard_count: int = 1
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, str]]:
    """One entry per sampled accession: every `(target_field, target_value)` still `"not found"`
    for it -- the unit of work this tier actually batches on (see `llm_evidence.verify_not_found_with_llm`
    docstring), not one entry per row. `sample_size=None` takes every accession in this shard
    (shuffled by `seed` only for the progress log's sake, not to subset it).
    """
    trace_back = pd.read_parquet(
        TRACE_BACK_FULL_PARQUET, columns=["accession", "run_name", "target_field", "target_value", "strategy"]
    )
    not_found = trace_back[trace_back["strategy"] == "not found"]
    accessions = not_found["accession"].drop_duplicates()
    if shard_count > 1:
        accessions = accessions[accessions.map(lambda a: _shard_of(a, shard_count) == shard_index)]
    accessions = accessions.sample(frac=1, random_state=seed)
    if sample_size is not None:
        accessions = accessions.head(sample_size)
    sampled = not_found[not_found["accession"].isin(accessions)]

    by_accession: dict[str, list[tuple[str, str]]] = {}
    run_name_by_accession: dict[str, str] = {}
    for row in sampled.itertuples(index=False):
        by_accession.setdefault(row.accession, []).append((row.target_field, row.target_value))
        run_name_by_accession[row.accession] = row.run_name
    return by_accession, run_name_by_accession


def not_found_items_from_prior_pass(paths: list[Path]) -> dict[str, list[tuple[str, str]]]:
    """Items a prior pass of this same tier still couldn't ground (`strategy == "not found"` in
    its own output -- covers the LLM's own not-found verdicts, call failures, and hallucination-
    only founds that failed grounding), for reprocessing with a stronger model. `paths` are that
    prior pass's `--out` files (one per shard); an accession absent from every input `path` never
    had a gap and is skipped.
    """
    by_accession: dict[str, list[tuple[str, str]]] = {}
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                items = [(row["target_field"], row["target_value"]) for row in rec["rows"] if row["strategy"] == "not found"]
                if items:
                    by_accession[rec["accession"]] = items
    return by_accession


def run_name_lookup(accessions: set[str]) -> dict[str, str]:
    """`accession -> run_name` for a specific set of accessions, read from the same parquet
    `sample_not_found_by_accession` uses -- needed when `by_accession` instead comes from a prior
    pass's own output (see `not_found_items_from_prior_pass`), which doesn't carry `run_name`.
    Raises if the parquet was regenerated since that prior pass ran and no longer has one of these
    accessions -- better a clear failure here than a bare KeyError from `raw_path_for` mid-run.
    """
    trace_back = pd.read_parquet(TRACE_BACK_FULL_PARQUET, columns=["accession", "run_name"])
    trace_back = trace_back[trace_back["accession"].isin(accessions)].drop_duplicates("accession")
    run_name_by_accession = dict(zip(trace_back["accession"], trace_back["run_name"]))
    missing = accessions - run_name_by_accession.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} accession(s) from --source-jsonl are no longer in "
            f"{TRACE_BACK_FULL_PARQUET.name}: {sorted(missing)[:5]}"
        )
    return run_name_by_accession


def already_processed(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["accession"])
    return done


def _next_batch(
    todo: list[str], batch_size: int, by_accession: dict[str, list[tuple[str, str]]], raw_path_for, raw_cache: RawRecordCache
) -> tuple[list[str], list[tuple[dict[str, Any], list[tuple[str, str]]]], list[str]]:
    """Pulls up to `batch_size` accessions off the front of `todo` (mutated in place) and looks up
    each one's raw record via `raw_cache` (see `RawRecordCache`). Returns `(accessions, records,
    missing)` -- `records` (ready for `verify_not_found_with_llm_batch`) and `accessions` stay
    index-aligned; `missing` is accessions whose raw record couldn't be found (a lookup error, not
    a model call), reported separately.
    """
    batch_accessions = todo[:batch_size]
    del todo[:batch_size]

    accessions, records, missing = [], [], []
    for accession in batch_accessions:
        raw = raw_cache.get(raw_path_for(accession), accession)
        if raw is None:
            missing.append(accession)
        else:
            accessions.append(accession)
            records.append((raw, by_accession[accession]))
    return accessions, records, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None, help="distinct accessions to draw from the not-found residual (this shard); omit to process ALL of them")
    parser.add_argument(
        "--source-jsonl", type=Path, nargs="+", default=None,
        help="reprocess only items still 'not found' in these prior pass output files (e.g. a stronger-model second "
        "pass over the first pass's gaps), instead of drawing the full residual from the trace-back parquet",
    )
    parser.add_argument("--max-hours", type=float, default=4.0, help="stop after this much wall time, whatever's been done so far")
    parser.add_argument("--batch-size", type=int, default=500, help="accessions per vLLM batch call -- vLLM batches internally, so this trades checkpoint granularity for per-batch overhead, not raw throughput")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--shard-index", type=int, default=0, help="process only accessions whose hash mod --shard-count equals this")
    parser.add_argument("--shard-count", type=int, default=1, help="split the not-found residual into this many independent shards, e.g. one per GPU")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.source_jsonl:
        by_accession = not_found_items_from_prior_pass(args.source_jsonl)
        if args.shard_count > 1:
            by_accession = {a: items for a, items in by_accession.items() if _shard_of(a, args.shard_count) == args.shard_index}
        run_name_by_accession = run_name_lookup(set(by_accession))
    else:
        by_accession, run_name_by_accession = sample_not_found_by_accession(
            args.sample_size, args.seed, shard_index=args.shard_index, shard_count=args.shard_count
        )
    run_index = pd.read_csv(RUN_INDEX_TSV, sep="\t").set_index("run_name")

    done = already_processed(args.out)
    todo = [acc for acc in by_accession if acc not in done]
    n_todo = len(todo)
    print(
        f"Sampled {len(by_accession):,} accessions ({args.seed=}); {len(done):,} already in {args.out}, "
        f"{n_todo:,} left to process. Model: {args.model}. Batch size: {args.batch_size}. Time budget: {args.max_hours}h.",
        flush=True,
    )

    def raw_path_for(accession: str) -> Path:
        run_row = run_index.loc[run_name_by_accession[accession]]
        return CRATE_INPUTS_DIR / run_row["dataset"] / run_row["input_file"]

    print("Loading vLLM engine (model load + CUDA graph capture -- a few minutes)...", flush=True)
    llm = load_engine(args.model)
    sampling_params = default_sampling_params()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_cache = RawRecordCache()
    start = time.monotonic()
    deadline = start + args.max_hours * 3600
    n_done, n_errors = 0, 0

    with args.out.open("a") as f:
        while todo:
            if time.monotonic() >= deadline:
                print(f"Time budget ({args.max_hours}h) reached -- stopping before the next batch.", flush=True)
                break

            accessions, records, missing = _next_batch(todo, args.batch_size, by_accession, raw_path_for, raw_cache)
            n_errors += len(missing)

            results = verify_not_found_with_llm_batch(records, llm, sampling_params) if records else []
            for accession, (rows, call_info) in zip(accessions, results):
                f.write(json.dumps({"accession": accession, "rows": rows, "call_info": call_info}) + "\n")
            f.flush()
            n_done += len(results)

            elapsed = time.monotonic() - start
            rate = n_done / elapsed if elapsed else 0
            eta = (n_todo - n_done) / rate / 60 if rate else float("inf")
            print(
                f"{n_done:,}/{n_todo:,} done in {elapsed / 60:.1f}m ({rate:.2f} accessions/s, "
                f"~{eta:.0f}m left in this sample at this rate)",
                flush=True,
            )

    elapsed = time.monotonic() - start
    print(f"Finished: {n_done:,} accessions processed ({n_errors} lookup errors) in {elapsed / 60:.1f}m. Output: {args.out}", flush=True)


if __name__ == "__main__":
    main()
