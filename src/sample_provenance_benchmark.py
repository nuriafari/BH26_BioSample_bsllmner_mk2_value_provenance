"""Draws Benchmark A: a reproducible sample of distinct `(target_field, target_value)` patterns
from the deterministic cascade's "not found" residual, before any LLM pass has run.

Unlike the earlier 200-pattern benchmark (`paths.QWEN_INDEPENDENT_REVIEW_200_PARQUET`), which has
no recoverable sampling script or seed and was sampled by accession -- so a common pattern like
`tissue=liver` could (and did) appear many times, e.g. `neuron` 14/200 -- this samples DISTINCT
patterns directly: the population is one row per distinct `(target_field, target_value)` pair, not
one row per unresolved (accession, field, value) occurrence, so every pattern in the sample is
guaranteed unique. Accessions are NOT required to be unique across the sample -- a pattern
unresolved on hundreds of accessions still gets exactly one (randomly chosen) representative
accession, and two different patterns coincidentally choosing the same accession is left alone
(see the discussion this design was agreed from: a specific accession being repeated is
inconsequential when the unit being evaluated is the pattern, not the record).

Run as a script:
    python sample_provenance_benchmark.py --n 400 --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from paths import PROVENANCE_BENCHMARK_400_PARQUET, TRACE_BACK_FULL_PARQUET


def sample_benchmark(n: int, seed: int) -> pd.DataFrame:
    """One row per sampled pattern: `target_field`, `target_value`, and one representative
    `accession`/`run_name`. A single `np.random.Generator` seeded once and advanced across every
    draw (the pattern sample, then each pattern's accession pick) -- not a fresh `random_state=seed`
    per call, which would reseed identically for every pattern and risk correlated picks.
    """
    rng = np.random.default_rng(seed)
    trace_back = pd.read_parquet(
        TRACE_BACK_FULL_PARQUET, columns=["accession", "run_name", "target_field", "target_value", "strategy"]
    )
    not_found = trace_back[trace_back["strategy"] == "not found"]

    pairs = not_found[["target_field", "target_value"]].drop_duplicates().reset_index(drop=True)
    if n > len(pairs):
        raise ValueError(f"requested n={n} exceeds {len(pairs)} distinct unresolved patterns available")
    sampled_pairs = pairs.sample(n=n, random_state=rng).reset_index(drop=True)

    grouped = not_found.groupby(["target_field", "target_value"], sort=False)
    rows = []
    for field, value in sampled_pairs.itertuples(index=False):
        occurrences = grouped.get_group((field, value))
        chosen = occurrences.sample(n=1, random_state=rng).iloc[0]
        rows.append(
            {
                "target_field": field,
                "target_value": value,
                "accession": chosen["accession"],
                "run_name": chosen["run_name"],
                "n_occurrences": len(occurrences),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = sample_benchmark(args.n, args.seed)
    PROVENANCE_BENCHMARK_400_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROVENANCE_BENCHMARK_400_PARQUET, index=False)
    print(f"Wrote {len(df):,} distinct patterns (n={args.n}, seed={args.seed}) to {PROVENANCE_BENCHMARK_400_PARQUET}")
    print(f"Distinct accessions in sample: {df['accession'].nunique()}/{len(df)}")


if __name__ == "__main__":
    main()
