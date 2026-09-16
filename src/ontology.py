"""Reading term labels/synonyms out of the RO-Crate's own local ontology OWL
files (`data/2026-06_mistral-small3.1-24b/ontology/*.owl`).

Used to enrich a "not found" trace-back row with what the pipeline's own
ontology-search stage (`text2term`) actually knows about the term it assigned
-- often a *related* term, not a literal synonym of the raw text, so this is
diagnostic context for a human to judge plausibility, not a 3rd deterministic
match strategy.

`build_ontology_index` streams the file line by line rather than parsing it
as XML (no `rdflib` dependency, and a plain line scan is fast even on a
140MB+ file -- see the benchmark this was built from) -- term IDs map
predictably to `rdf:about="http://purl.obolibrary.org/obo/<PREFIX>_<NUM>"`
(Cellosaurus nests its own terms one level deeper, under
`.../obo/Cellosaurus#CVCL_<NUM>`, handled as a second URI shape below).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

_CLASS_RE = re.compile(r'rdf:about="http://purl\.obolibrary\.org/obo/(?:Cellosaurus#)?([A-Za-z]+_\d+)"')
_LABEL_RE = re.compile(r"<rdfs:label>(.*?)</rdfs:label>")
_EXACT_SYN_RE = re.compile(r"<oboInOwl:hasExactSynonym>(.*?)</oboInOwl:hasExactSynonym>")
_RELATED_SYN_RE = re.compile(r"<oboInOwl:hasRelatedSynonym>(.*?)</oboInOwl:hasRelatedSynonym>")


class OntologyTerm(TypedDict):
    label: str | None
    exact_synonyms: list[str]
    related_synonyms: list[str]


def build_ontology_index(path: Path) -> dict[str, OntologyTerm]:
    """One streaming pass over an OWL file, building `term_id -> OntologyTerm`
    for every term it defines. Call once per file and reuse the result --
    this is the only part of ontology lookup that costs real time (a fresh
    per-lookup scan would be O(file size) per call instead of O(file size)
    total).
    """
    index: dict[str, OntologyTerm] = {}
    current_id: str | None = None
    label: str | None = None
    exact: list[str] = []
    related: list[str] = []

    def flush() -> None:
        if current_id is not None:
            index[current_id] = {"label": label, "exact_synonyms": exact, "related_synonyms": related}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = _CLASS_RE.search(line)
            if match:
                flush()
                current_id = match.group(1).replace("_", ":", 1)
                label, exact, related = None, [], []
                continue
            if current_id is None:
                continue
            match = _LABEL_RE.search(line)
            if match:
                label = match.group(1)
            match = _EXACT_SYN_RE.search(line)
            if match:
                exact.append(match.group(1))
            match = _RELATED_SYN_RE.search(line)
            if match:
                related.append(match.group(1))
    flush()
    return index


def build_field_ontology_indexes(select_config: dict, ontology_dir: Path, fields: list[str]) -> dict[str, dict[str, OntologyTerm]]:
    """Builds one ontology index per requested field, keyed by field name --
    reads each field's `ontology_file` straight out of the select config
    (`config/select-config-*.json`, same file the pipeline itself used), and
    builds the underlying OWL index once per distinct file even when several
    fields share one (e.g. `disease` and any other MONDO-backed field).
    """
    built_by_path: dict[Path, dict[str, OntologyTerm]] = {}
    result: dict[str, dict[str, OntologyTerm]] = {}
    for field in fields:
        ontology_file = (select_config.get("fields", {}).get(field) or {}).get("ontology_file")
        if not ontology_file:
            continue
        path = ontology_dir / Path(ontology_file).name
        if not path.exists():
            continue
        if path not in built_by_path:
            built_by_path[path] = build_ontology_index(path)
        result[field] = built_by_path[path]
    return result
