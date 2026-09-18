"""Live lookups against EBI's Ontology Lookup Service (OLS4) -- the same backend
Zooma itself sits on. Used to check whether the pipeline's assigned term is a
descendant of (is_a chain up to) a Zooma candidate, or a declared cross-ontology
equivalent of one.

Why live lookups instead of a local is_a/xref graph: the RO-Crate's own local
ontology files under data/*/ontology/*.owl are flat label/synonym extracts --
checked directly, none of them contain rdfs:subClassOf or hasDbXref at all, so
there's no local hierarchy to walk. The number of distinct terms actually
needing a check is small enough (a few hundred, see
notebooks/zooma_cross_check.ipynb) that live, cached OLS calls are simpler and
more correct than downloading and parsing full-hierarchy ontology releases.

Only OBO PURL-style ontologies (CL, MONDO, UBERON, CHEBI, EFO) are supported --
NCBIGene and Cellosaurus (CVCL) terms aren't OLS-indexed ontology classes, so
callers get an empty result for those and fall back to exact-match only.
"""

from __future__ import annotations

import json
import time
import urllib.parse

import requests

from paths import DERIVED_DIR

CACHE_PATH = DERIVED_DIR / "ols_term_cache.json"
BETWEEN_CALL_SLEEP_S = 0.2

# OBO PURL is the base IRI for most OBO Foundry ontologies; EFO instead lives under EBI's own path.
_IRI_BASE = {"EFO": "http://www.ebi.ac.uk/efo/"}
_DEFAULT_IRI_BASE = "http://purl.obolibrary.org/obo/"
_LOOKUP_SUPPORTED_PREFIXES = {"CL", "MONDO", "UBERON", "CHEBI", "EFO"}


def _double_encode(iri: str) -> str:
    return urllib.parse.quote(urllib.parse.quote(iri, safe=""), safe="")


def _iri_for(prefix: str, local_id: str) -> str:
    return _IRI_BASE.get(prefix, _DEFAULT_IRI_BASE) + f"{prefix}_{local_id}"


class OlsClient:
    """One client per script run: loads the on-disk cache once, saves it back
    incrementally (every `_SAVE_EVERY` new lookups) so a long run surviving a
    crash doesn't lose already-fetched terms."""

    _SAVE_EVERY = 25

    def __init__(self):
        self.cache: dict[str, dict] = (
            json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
        )
        self._n_new = 0

    def _save(self) -> None:
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(self.cache))

    def term_info(self, prefix: str, local_id: str) -> dict:
        """{'ancestors': set[(prefix, local_id)], 'xrefs': set[(database, id)]} for one
        canonical (prefix, local_id) term -- both empty if the prefix isn't OLS-lookupable
        or the term/ontology isn't found."""
        prefix = prefix.upper()
        key = f"{prefix}_{local_id}"
        if key in self.cache:
            return self.cache[key]

        result = {"ancestors": [], "xrefs": []}
        if prefix in _LOOKUP_SUPPORTED_PREFIXES:
            ontology = prefix.lower()
            iri = _iri_for(prefix, local_id)
            enc = _double_encode(iri)
            result["ancestors"] = self._fetch_ancestors(ontology, enc)
            result["xrefs"] = self._fetch_xrefs(ontology, enc)
            time.sleep(BETWEEN_CALL_SLEEP_S)

        self.cache[key] = result
        self._n_new += 1
        if self._n_new % self._SAVE_EVERY == 0:
            self._save()
        return result

    def _fetch_ancestors(self, ontology: str, encoded_iri: str) -> list[list[str]]:
        url = f"https://www.ebi.ac.uk/ols4/api/ontologies/{ontology}/terms/{encoded_iri}/ancestors"
        data = self._get(url)
        if data is None:
            return []
        terms = (data.get("_embedded") or {}).get("terms") or []
        out = []
        for t in terms:
            obo_id = t.get("obo_id")
            if obo_id and ":" in obo_id:
                p, local = obo_id.split(":", 1)
                out.append([p.upper(), local])
        return out

    def _fetch_xrefs(self, ontology: str, encoded_iri: str) -> list[list[str]]:
        """Only xrefs whose own description asserts real equivalence (e.g. MONDO's
        "MONDO:equivalentTo") -- ontologies like MONDO also carry much looser xrefs
        (e.g. "MONDO:otherHierarchy", or a curator's bare ORCID as the description)
        that point at a related-but-not-equivalent term in another database; counting
        those as a match produced a real false positive during development (MONDO's
        "diabetes mellitus" via an "otherHierarchy" xref to HP's "abnormal glucose
        homeostasis" phenotype term -- related, not the same thing)."""
        url = (
            f"https://www.ebi.ac.uk/ols4/api/ontologies/{ontology}/terms/{encoded_iri}"
        )
        data = self._get(url)
        if data is None:
            return []
        return [
            [x["database"].upper(), x["id"]]
            for x in (data.get("obo_xref") or [])
            if x.get("database")
            and x.get("id")
            and "equivalentto" in (x.get("description") or "").lower()
        ]

    def _get(self, url: str) -> dict | None:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                return None
            return r.json()
        except requests.RequestException:
            return None

    def close(self) -> None:
        self._save()
