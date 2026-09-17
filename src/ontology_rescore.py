"""Independent ontology re-scoring: a cheap, CPU-only, non-LLM tier for the trace-back check's
"not found" residual, using the SAME mechanism the crate's own extraction pipeline used to assign
ontology terms in the first place (`text2term`'s TF-IDF character-n-gram similarity, run against
the same local OWL files already in the crate).

The idea: a "not found" row usually still has an `assigned_term_id` (`results[field]`) -- the
pipeline's own ontology search already picked SOME term for the extracted value, even though our
deterministic search couldn't find that value's own text anywhere in the record. If one of the
record's own attributes independently re-scores well against that SAME assigned term (not
necessarily an exact string match), that's evidence the assignment is grounded in this record's
text after all -- just phrased differently than the extracted value's own wording (e.g. an
adjective/noun variant text2term's substring-based "ontology synonym" tier can't bridge either).

Categorized as an ontology-family tier (like `ontology synonym`), not folded into it: `ontology
synonym` checks literal substring containment of the assigned term's curated synonyms; this tier
checks continuous SIMILARITY SCORING against the term's label, which needs a different mechanism
(text2term) and has a different, measured failure mode (see below) -- kept distinct so the two are
independently measurable.

Two measured limitations shaped this design, neither hypothetical -- both found while validating
it against real crate data:

1. A strict "top-1 match must equal the assigned term" check is NOT reliable -- text2term's
   character-n-gram scoring has no morphological awareness, so an adjective/noun variant pair
   (e.g. "asthma"/"asthmatic") can score a different, unrelated term fractionally higher purely
   from string overlap (e.g. "status asthmaticus" edges out "asthma" for the query "Asthmatic":
   0.711 vs 0.714, effectively a coin flip). Fix: check whether the assigned term appears ANYWHERE
   in the top-N candidates (`TOP_N`), not just rank 1.
2. The raw score is also NOT a reliable pass/fail threshold on its own -- two genuine catches
   ("Prog" for progenitor cell, 0.372; "IgAN" for IgA nephropathy, 0.405) scored LOWER than two
   confirmed false positives ("male" for testis, 0.518; a stray "regionre" for Crohn's disease,
   0.456). No single cutoff keeps the former and excludes the latter. Fix: `best_supporting_
   candidate` additionally requires the candidate to share a real token with the term's own
   CANONICAL LABEL (see its docstring) -- every genuine catch found so far has an explainable
   textual relationship to the label; neither false positive shares anything with it.
"""

from __future__ import annotations

import re
from pathlib import Path

import text2term

TOP_N = 5  # see module docstring -- top-1-only is measured to be unreliable for word-form variants
MIN_SCORE = 0.3  # text2term's own default floor; below this a "match" is noise
_MIN_TOKEN_LENGTH = 3  # below this, a token is too short/generic to count as a meaningful shared identity anchor -- "IgA" itself is exactly 3 characters and must still pass
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_cached_ontologies: dict[Path, str] = {}


def cache_field_ontology(ontology_path: Path) -> str:
    """Caches one ontology file for text2term, once per path -- the expensive step (parsing the
    OWL file and building text2term's internal term index), reused across every candidate string
    subsequently scored against it. Returns the cache name to pass to `score_candidates`.
    """
    if ontology_path not in _cached_ontologies:
        name = ontology_path.stem
        text2term.cache_ontology(str(ontology_path), name)
        _cached_ontologies[ontology_path] = name
    return _cached_ontologies[ontology_path]


def score_candidates(candidates: list[str], ontology_name: str, top_n: int = TOP_N, min_score: float = MIN_SCORE) -> dict[str, list[tuple[str, float]]]:
    """Batch-scores every DISTINCT candidate string against the cached ontology in one text2term
    call (batching many candidates together is a speed optimization only -- text2term's term
    weighting is fit on the ontology's own vocabulary, not the query batch, so this doesn't change
    any individual candidate's own result). Returns `candidate -> [(term_curie, score), ...]`,
    top-`top_n` per candidate, empty list for a candidate with nothing above `min_score`.
    """
    distinct = list(dict.fromkeys(c for c in candidates if c and c.strip()))
    if not distinct:
        return {}
    df = text2term.map_terms(distinct, ontology_name, use_cache=True, max_mappings=top_n, min_score=min_score, incl_unmapped=True)
    result: dict[str, list[tuple[str, float]]] = {c: [] for c in distinct}
    for _, row in df.iterrows():
        curie = row["Mapped Term CURIE"]
        if isinstance(curie, str) and curie:
            result[row["Source Term"]].append((curie, row["Mapping Score"]))
    return result


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= _MIN_TOKEN_LENGTH]


def _shares_token_with_label(candidate: str, label: str) -> bool:
    """Whether any token in `candidate` and any token in `label` are related by substring
    containment in EITHER direction -- e.g. `"Prog"`/`"progenitor"` (the candidate is a prefix of
    the label's own word) or `"IgAN"`/`"IgA glomerulonephritis"` (the label's own `"IgA"` token is
    a prefix of the candidate). Checked against the term's CANONICAL LABEL ONLY, not its full
    synonym list -- measured directly (see module docstring): UBERON's `testis` has `"male
    gonad"` registered as an EXACT synonym, so checking the full synonym list would let a bare
    `"male"` candidate through (it's a genuine word inside a genuine synonym) even though "male"
    alone says nothing about testis specifically. A term's own label carries its core identity in
    a way a single modifier word from inside a multi-word synonym doesn't.
    """
    candidate_tokens = _tokens(candidate)
    label_tokens = _tokens(label)
    return any(a in b or b in a for a in candidate_tokens for b in label_tokens)


def best_supporting_candidate(
    candidate_scores: dict[str, list[tuple[str, float]]], target_term_id: str, target_label: str
) -> tuple[str, float] | None:
    """Among candidates whose top-N includes `target_term_id` AND that share a real token with
    the term's own label (`_shares_token_with_label` -- see its docstring for why label-only,
    not the full synonym list), the one with the highest score -- or `None` if nothing qualifies.

    The label check exists because the raw text2term score alone is NOT a reliable filter here:
    measured directly on a real sample, two of the most valuable genuine catches (`"Prog"` for
    `progenitor cell`, score 0.372; `"IgAN"` for `IgA nephropathy`, score 0.405) scored LOWER than
    two confirmed false positives (`"male"` for `testis`, 0.518; a stray `"regionre"` for `Crohn's
    disease`, 0.456) -- there is no score threshold that keeps the former and excludes the latter,
    since the false positives score HIGHER. The label-overlap check catches what the score can't:
    every genuine match found so far has a real, explainable textual relationship to the term's
    own label; both false positives share nothing with their term's label at all.
    """
    candidates_by_score = sorted(
        ((candidate, score) for candidate, scored_terms in candidate_scores.items() for term_id, score in scored_terms if term_id == target_term_id),
        key=lambda pair: -pair[1],
    )
    for candidate, score in candidates_by_score:
        if _shares_token_with_label(candidate, target_label):
            return candidate, score
    return None
