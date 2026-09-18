"""Scores every case in the Zooma cross-check: does the pipeline's assigned
ontology term agree with what Zooma (an independent, more powerful lookup
service than the deterministic cascade's own non-exact strategies) proposes
for the same literal evidence text?

A case is one of:
  - "exact"            the assigned term is one of Zooma's candidates outright
  - "child"             the assigned term is a descendant (is_a chain) of a candidate
  - "xref"               the assigned term is a declared cross-ontology equivalent of a candidate
  - "different_term"    Zooma found candidates, none of which relate to the assigned term
  - "no_zooma_result"    Zooma found no candidates at all for this evidence phrase

"child"/"xref" need live OLS ancestor/cross-reference lookups (see
ols_client.py -- the crate's own local ontology files have no hierarchy data
to check this against). Only CL/MONDO/UBERON/CHEBI/EFO terms are
OLS-lookupable; NCBIGene and Cellosaurus (CVCL) assigned terms can only ever
land in "exact"/"different_term"/"no_zooma_result".
"""

from __future__ import annotations

import json
import re

import pandas as pd

from ols_client import OlsClient
from paths import ZOOMA_CASE_UNIVERSE_PARQUET, ZOOMA_RESULTS_JSONL

CONFIDENCE_THRESHOLD = 0.5


def canonical_term(raw_id: str | None) -> tuple[str, str] | None:
    """'UBERON:0002107' / 'UBERON_0002107' -> ('UBERON', '0002107'); None if unparseable."""
    if not raw_id:
        return None
    token = raw_id.replace(":", "_", 1) if ":" in raw_id else raw_id
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*)_(.+)$", token)
    return (m.group(1).upper(), m.group(2)) if m else None


def load_zooma_results() -> dict[str, dict | None]:
    """phrase -> Zooma's raw `mapping` dict (None if Zooma found no candidates)."""
    results = {}
    with ZOOMA_RESULTS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            results[rec["phrase"]] = rec["zooma_mapping"]
    return results


def confident_candidates(
    mapping: dict | None, min_confidence: float = CONFIDENCE_THRESHOLD
) -> list[dict]:
    if mapping is None:
        return []
    return [
        c
        for c in mapping.get("candidates", [])
        if (c.get("confidence") or 0.0) >= min_confidence
    ]


def classify_case(
    assigned_term_id: str, candidates: list[dict], ols: OlsClient
) -> dict:
    """One case's verdict against its confidence-filtered Zooma candidates."""
    assigned = canonical_term(assigned_term_id)
    top = candidates[0] if candidates else None
    base = {
        "top_candidate_term_id": top["termId"] if top else None,
        "top_candidate_label": top["label"] if top else None,
        "n_candidates": len(candidates),
    }

    if not candidates:
        return {**base, "verdict": "no_zooma_result", "matched_candidate_term_id": None}

    candidate_terms = [(canonical_term(c["termId"]), c) for c in candidates]

    for canon, c in candidate_terms:
        if canon == assigned:
            return {
                **base,
                "verdict": "exact",
                "matched_candidate_term_id": c["termId"],
            }

    if assigned is not None:
        info = ols.term_info(*assigned)
        ancestor_set = {tuple(a) for a in info["ancestors"]}
        xref_set = {tuple(x) for x in info["xrefs"]}
        for canon, c in candidate_terms:
            if canon in ancestor_set:
                return {
                    **base,
                    "verdict": "child",
                    "matched_candidate_term_id": c["termId"],
                }
        for canon, c in candidate_terms:
            if canon is not None and canon in xref_set:
                return {
                    **base,
                    "verdict": "xref",
                    "matched_candidate_term_id": c["termId"],
                }

    return {**base, "verdict": "different_term", "matched_candidate_term_id": None}


def score_all_cases() -> pd.DataFrame:
    cases = pd.read_parquet(ZOOMA_CASE_UNIVERSE_PARQUET)
    zooma_results = load_zooma_results()
    ols = OlsClient()

    verdicts = []
    for _, row in cases.iterrows():
        mapping = zooma_results.get(row["raw_matched_phrase"])
        candidates = confident_candidates(mapping)
        verdicts.append(classify_case(row["assigned_term_id"], candidates, ols))
    ols.close()

    return pd.concat([cases.reset_index(drop=True), pd.DataFrame(verdicts)], axis=1)
