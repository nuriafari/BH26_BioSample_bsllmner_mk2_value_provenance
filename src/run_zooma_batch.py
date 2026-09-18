"""Runs every distinct evidence phrase in the Zooma case universe through EBI's
Zooma text-annotation API and writes one JSON line per phrase to disk.

Deliberately does nothing but collect raw Zooma output -- no ontology-relation
scoring here. Scoring needs the is_a/xref graph built from the local OWL
files, which is comparatively slow to build and fast to apply; interleaving
it with the network calls would only slow the part that's actually at risk
from an overnight outage. Score the output separately, after this finishes
(or against however much of it exists, since output is flushed per phrase).

Batching: distinct phrases (not cases) are sent, since two cases can share
the same literal evidence phrase and Zooma's response to identical text is
deterministic -- sending it twice would waste a call and API-side result
matching by exact phrase text is used to pair phrase text 1:1 -> a
"[case_id] " prefix / offset bookkeeping to recover). A smoke test (see
project chat log; not re-run by this script) found:
  - Zooma's own response is a NDJSON stream with `segments` (its own
    tokenization) and `result` events (one per segment it could search
    for candidates, keyed by exact-match `mapping.textToMap` text) --
    segments with zero candidates produce no `result` event at all.
  - Latency scales gently (2500 phrases ~= 6s, ~2MB response) with no
    sign of throttling on rapid back-to-back calls up to that size.
  - A much larger single call (all ~16k distinct phrases, ~190K chars) was
    deliberately NOT tried -- the previous project's own Zooma client flagged
    a ~44K-char single text as already past anything confirmed safe, so
    BATCH_SIZE here is kept well under that, trading a handful of extra
    requests (all still sub-10s) for margin against an undocumented limit.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import requests

from paths import (
    ZOOMA_CASE_UNIVERSE_PARQUET,
    ZOOMA_FAILED_BATCHES_JSONL,
    ZOOMA_RESULTS_JSONL,
)

ZOOMA_URL = "https://www.ebi.ac.uk/spot/zooma/v3/api/services/annotate-text-stream"
BATCH_SIZE = 2000
MIN_SPLIT_SIZE = (
    50  # a batch that still fails at this size is logged as failed, not split further
)
BETWEEN_BATCH_SLEEP_S = 1.5
MAX_ATTEMPTS = 6


def already_processed_phrases() -> set[str]:
    if not ZOOMA_RESULTS_JSONL.exists():
        return set()
    seen = set()
    with ZOOMA_RESULTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seen.add(json.loads(line)["phrase"])
    return seen


def call_zooma(text: str, timeout: int = 120) -> list[dict]:
    payload = {
        "text": text,
        "targetOntologies": [],
        "includeOtherOntologies": True,  # open search across all of Zooma's ontologies, not just
        "filter": {
            "required": [],
            "preferred": [],
        },  # the pipeline's own assumed one -- see chat log
    }
    response = requests.post(
        ZOOMA_URL,
        json=payload,
        headers={"Accept": "application/x-ndjson"},
        timeout=timeout,
    )
    response.raise_for_status()
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def call_zooma_with_retry(text: str) -> list[dict] | None:
    """None means every attempt failed -- caller decides whether to split and retry."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call_zooma(text)
        except (requests.RequestException, TimeoutError) as e:
            wait = min(2**attempt, 60)
            print(
                f"  attempt {attempt + 1}/{MAX_ATTEMPTS} failed ({e!r}), retrying in {wait}s"
            )
            time.sleep(wait)
    return None


def process_batch(phrases: list[str], out_f, failed_f) -> int:
    """Calls Zooma for `phrases`, writes one result line each. Splits and recurses on
    failure until MIN_SPLIT_SIZE, so one bad phrase/timeout doesn't sink a whole batch.
    Returns the number of phrases successfully written."""
    text = "\n".join(phrases)
    events = call_zooma_with_retry(text)

    if events is None:
        if len(phrases) <= MIN_SPLIT_SIZE:
            failed_f.write(json.dumps({"phrases": phrases}) + "\n")
            failed_f.flush()
            return 0
        mid = len(phrases) // 2
        return process_batch(phrases[:mid], out_f, failed_f) + process_batch(
            phrases[mid:], out_f, failed_f
        )

    # `result` events are keyed by exact segment text; a phrase absent here got zero candidates.
    results_by_text = {
        e["mapping"]["textToMap"]: e["mapping"]
        for e in events
        if e.get("type") == "result"
    }
    for phrase in phrases:
        out_f.write(
            json.dumps({"phrase": phrase, "zooma_mapping": results_by_text.get(phrase)})
            + "\n"
        )
    out_f.flush()
    return len(phrases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap on distinct phrases, for testing"
    )
    args = parser.parse_args()

    cases = pd.read_parquet(ZOOMA_CASE_UNIVERSE_PARQUET)
    all_phrases = cases["raw_matched_phrase"].drop_duplicates().tolist()
    if args.limit:
        all_phrases = all_phrases[: args.limit]

    done = already_processed_phrases()
    todo = [p for p in all_phrases if p not in done]
    print(
        f"{len(all_phrases)} distinct phrases total, {len(done)} already done, {len(todo)} remaining"
    )

    start = time.time()
    n_written = 0
    with (
        ZOOMA_RESULTS_JSONL.open("a", encoding="utf-8") as out_f,
        ZOOMA_FAILED_BATCHES_JSONL.open("a", encoding="utf-8") as failed_f,
    ):
        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start : batch_start + args.batch_size]
            n_written += process_batch(batch, out_f, failed_f)

            elapsed = time.time() - start
            rate = n_written / elapsed if elapsed > 0 else 0
            remaining = len(todo) - (batch_start + len(batch))
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            print(
                f"[{n_written}/{len(todo)}] {rate:.1f} phrases/s, ETA {eta_min:.1f} min"
            )

            time.sleep(BETWEEN_BATCH_SLEEP_S)

    print(f"Done. {n_written} phrases written this run -> {ZOOMA_RESULTS_JSONL}")


if __name__ == "__main__":
    main()
