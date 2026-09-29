"""Scores Benchmark A (`paths.PROVENANCE_BENCHMARK_400_PARQUET`) once every piece is ready:
the Claude Opus reference annotation (20 chunk `.verdicts.json` files, see
`build_benchmark_annotation_chunks.py`) and each Qwen model's two-condition run (see
`run_provenance_benchmark_qwen.py`, one `.jsonl` per model in `notebooks/fixtures/provenance_benchmark_400_results/`).

Three things this reports:
1. Precision/recall/accuracy per model, per context condition (`attributes` vs `full_record`),
   against the Claude reference -- the standard evidence-tier evaluation, done twice per model.
2. The context-scope comparison this benchmark was actually built for: restricted to cases where
   Claude's OWN grounded evidence sits in a submitted attribute (so the narrower `attributes`
   condition should in principle already be sufficient), does giving the model the complete record
   instead change its verdict? Agreement rate and every flip (either direction), so a real
   hallucination-by-omission example (attributes-only says found, full-record reveals it shouldn't)
   is directly inspectable rather than just counted.
3. Claude's own quotes, independently re-verified the same way Qwen's are (all quotes must ground,
   checked against the complete record it was shown) -- closes the asymmetry flagged during review.

Run as a script:
    python score_provenance_benchmark.py --chunks-dir /path/to/benchmark_chunks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bs_entries import as_list, unwrap_biosample
from build_benchmark_annotation_chunks import _raw_records_by_run
from llm_evidence import _ground_quotes
from paths import PROVENANCE_BENCHMARK_400_PARQUET
from run_provenance_benchmark_qwen import OUT_DIR as QWEN_OUT_DIR


def load_claude_annotations(chunks_dir: Path) -> pd.DataFrame:
    """Merges every `chunk_*.verdicts.json` in `chunks_dir` into one DataFrame, one row per case."""
    verdict_files = sorted(chunks_dir.glob("chunk_*.verdicts.json"))
    if not verdict_files:
        raise FileNotFoundError(f"no *.verdicts.json files in {chunks_dir}")
    rows = []
    for path in verdict_files:
        rows.extend(json.loads(path.read_text()))
    df = pd.DataFrame(rows)
    missing = set(range(400)) - set(df["case_index"])
    if missing:
        raise ValueError(
            f"{len(missing)} case(s) have no Claude verdict: {sorted(missing)[:10]}"
        )
    return df.sort_values("case_index").reset_index(drop=True)


def verify_claude_quotes(
    claude_df: pd.DataFrame, raw_by_accession: dict
) -> pd.DataFrame:
    """Independently re-grounds every Claude quote against the complete record (Claude was shown
    the complete record, so `full_record=True`), with the SAME all-quotes-must-ground rule used for
    Qwen -- closes the asymmetry flagged in the pre-existing methods review. Adds `claude_accepted`
    (bool) and `claude_evidence_fields` (the attribute/field names each grounded quote landed in).
    """
    accepted, evidence_fields = [], []
    for row in claude_df.itertuples(index=False):
        raw = raw_by_accession[row.accession]
        quotes = row.quotes or []
        matches, ungrounded = _ground_quotes(raw, quotes, full_record=True)
        accepted.append(row.verdict == "found" and bool(quotes) and not ungrounded)
        evidence_fields.append([name for name, _, _ in matches])
    claude_df = claude_df.copy()
    claude_df["claude_accepted"] = accepted
    claude_df["claude_evidence_fields"] = evidence_fields
    return claude_df


def evidence_in_attribute(
    accession: str, evidence_fields: list[str], raw_by_accession: dict
) -> bool:
    """Whether EVERY one of Claude's grounded evidence fields is a genuine submitted attribute name
    on this record (not `title`, `comment`, or any other secondary field) -- the filter this
    benchmark's context-scope comparison is restricted to, since only these cases are ones the
    narrower `attributes` condition could in principle resolve on its own.
    """
    if not evidence_fields:
        return False
    raw = raw_by_accession[accession]
    bs = unwrap_biosample(raw)
    attribute_names = {
        a.get("attribute_name")
        for a in as_list((bs.get("Attributes") or {}).get("Attribute"))
    }
    return all(field in attribute_names for field in evidence_fields)


def load_qwen_results() -> pd.DataFrame:
    frames = []
    for path in sorted(QWEN_OUT_DIR.glob("*.jsonl")):
        model = json.loads(path.with_suffix(".meta.json").read_text())["model"]
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        for row in rows:
            row["model"] = model
        frames.append(pd.DataFrame(rows))
    if not frames:
        raise FileNotFoundError(f"no *.jsonl results in {QWEN_OUT_DIR}")
    return pd.concat(frames, ignore_index=True)


def score(predicted: pd.Series, reference: pd.Series) -> dict:
    tp = int(((predicted) & (reference)).sum())
    fp = int(((predicted) & (~reference)).sum())
    fn = int(((~predicted) & (reference)).sum())
    tn = int(((~predicted) & (~reference)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    accuracy = (tp + tn) / len(predicted) if len(predicted) else float("nan")
    return {
        "n": len(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=Path, required=True)
    args = parser.parse_args()

    sample = pd.read_parquet(PROVENANCE_BENCHMARK_400_PARQUET)
    raw_by_accession = _raw_records_by_run(sample)

    claude_df = load_claude_annotations(args.chunks_dir)
    claude_df = verify_claude_quotes(claude_df, raw_by_accession)
    claude_df["evidence_in_attribute"] = [
        evidence_in_attribute(acc, fields, raw_by_accession)
        for acc, fields in zip(
            claude_df["accession"], claude_df["claude_evidence_fields"]
        )
    ]

    print(
        f"Claude reference: {claude_df['claude_accepted'].sum()}/{len(claude_df)} accepted as found"
    )
    n_hallucinated = (
        (claude_df["verdict"] == "found") & ~claude_df["claude_accepted"]
    ).sum()
    if n_hallucinated:
        print(
            f"  {n_hallucinated} Claude 'found' verdicts had a quote that failed independent grounding -- demoted to not-accepted"
        )
    print(
        f"  {claude_df['evidence_in_attribute'].sum()}/{claude_df['claude_accepted'].sum()} accepted cases have ALL evidence in a submitted attribute"
    )

    qwen_df = load_qwen_results()
    merged = qwen_df.merge(
        claude_df[["case_index", "claude_accepted", "evidence_in_attribute"]],
        on="case_index",
    )

    print(
        "\n=== Precision/recall/accuracy per model x condition, vs Claude reference ==="
    )
    summary_rows = []
    for (model, condition), group in merged.groupby(["model", "condition"]):
        s = score(group["accepted"], group["claude_accepted"])
        s["model"], s["condition"] = model, condition
        summary_rows.append(s)
        print(
            f"{model:30s} {condition:12s} n={s['n']:3d} precision={s['precision']:.3f} recall={s['recall']:.3f} accuracy={s['accuracy']:.3f}"
        )
    pd.DataFrame(summary_rows).to_parquet(
        QWEN_OUT_DIR / "summary_precision_recall.parquet", index=False
    )

    print("\n=== Context-scope comparison: evidence-in-attribute subset only ===")
    subset = merged[merged["evidence_in_attribute"]]
    for model, group in subset.groupby("model"):
        wide = group.pivot(index="case_index", columns="condition", values="accepted")
        agree = (wide["attributes"] == wide["full_record"]).mean()
        flips_to_found = ((~wide["attributes"]) & wide["full_record"]).sum()
        flips_to_not_found = (wide["attributes"] & (~wide["full_record"])).sum()
        print(
            f"{model:30s} n={len(wide):3d} agreement={agree:.3f} "
            f"attrs-only-missed-but-full-record-found={flips_to_found} "
            f"attrs-only-found-but-full-record-lost={flips_to_not_found}"
        )
        flipped = wide[wide["attributes"] != wide["full_record"]]
        if len(flipped):
            print(f"  flipped case_index values: {list(flipped.index)}")

    out_path = QWEN_OUT_DIR / "merged_benchmark_results.parquet"
    merged.to_parquet(out_path, index=False)
    claude_df.to_parquet(QWEN_OUT_DIR / "claude_reference.parquet", index=False)
    print(f"\nWrote merged results to {out_path}")


if __name__ == "__main__":
    main()
