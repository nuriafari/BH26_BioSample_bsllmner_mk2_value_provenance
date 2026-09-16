"""Loading raw BioSample input records (the `inputs/*.jsonl` files in the RO-Crate).

The record-shape handling here (the "wrapped in a BioSample key, or not" and
the "Attribute is a dict when there's one, a list when there's more than one"
quirks) mirrors what bsllmner-viewer's own ETL does in
`etl/load_input.py:_normalize_entry` / `_as_list` -- ported and simplified here
(no Pydantic dependency) rather than imported, so this project stays
self-contained and does not reach into a sibling git checkout at runtime.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import spacy
from negspacy.negation import (
    Negex,  # noqa: F401 -- import registers spaCy's "negex" pipe factory
)

from notebook_utils import RawHTML

_ACCESSION_RE = re.compile(r'"accession":\s*"([^"]+)"')


def as_list(value: Any) -> list[Any]:
    """Normalize a field that is a bare dict when there's one value and a
    list when there's more than one (e.g. `Attributes.Attribute`) to always
    be a list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def unwrap_biosample(raw: dict[str, Any]) -> dict[str, Any]:
    """RNA-Seq input rows wrap the record under a `"BioSample"` key; ChIP-Atlas
    rows are flat. Return the inner record either way.
    """
    return raw.get("BioSample", raw)


def get_accession(raw: dict[str, Any]) -> str | None:
    """The accession is the join key back to the curated result entries
    (`entries[].extract.accession` in a `results/select_*.json` file).

    Read from the record's own top level, not from inside `BioSample` -- a
    handful of records are missing the nested `BioSample.accession` field
    entirely, but the top-level one is always present, in both the flat
    (ChIP-Atlas) and wrapped (RNA-Seq) record shapes.
    """
    accession = raw.get("accession")
    return accession if isinstance(accession, str) else None


def iter_bs_entries(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one raw (still-wrapped, un-normalized) BioSample record dict per
    line of an `inputs/*.jsonl` file.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_bs_entry(path: Path, accession: str) -> dict[str, Any] | None:
    """Scan an `inputs/*.jsonl` file for the raw record with this accession.

    Good enough for a pilot on a single small file (a few hundred KB-MB); at
    full scale you'd build an accession -> byte-offset index once instead of
    re-scanning per lookup.
    """
    for raw in iter_bs_entries(path):
        if get_accession(raw) == accession:
            return raw
    return None


def find_bs_entry_in_files(paths: list[Path], accession: str) -> dict[str, Any] | None:
    """Scan a list of `inputs/*.jsonl` files for the raw record with this
    accession, stopping at the first file that contains it. A cheap substring
    pre-check skips full JSON parsing on files that can't match.
    """
    needle = f'"{accession}"'
    for path in paths:
        if needle in path.read_text(encoding="utf-8"):
            record = find_bs_entry(path, accession)
            if record is not None:
                return record
    return None


def get_organism(raw: dict[str, Any]) -> str | None:
    """The organism name from `Description.Organism`. Some records key this
    `OrganismName`, others `taxonomy_name` -- both are checked, since which
    one a given record uses is inconsistent even within one input file.
    """
    organism = (unwrap_biosample(raw).get("Description") or {}).get("Organism") or {}
    return organism.get("OrganismName") or organism.get("taxonomy_name")


def _flatten_to_text(obj: Any, prefix: str = "") -> str:
    """Renders a small nested dict/list (e.g. `Owner`, `Package`, `Status`) as
    a compact "key: value; key: value" string, so every leaf value ends up as
    literal text somewhere in the display -- used for fields too small/simple
    to deserve their own nested sub-table.
    """
    if obj is None:
        return ""
    if isinstance(obj, dict):
        parts = [_flatten_to_text(v, prefix=f"{prefix}{k}.") for k, v in obj.items()]
        return "; ".join(p for p in parts if p)
    if isinstance(obj, list):
        return "; ".join(p for p in (_flatten_to_text(item, prefix=prefix) for item in obj) if p)
    return f"{prefix.rstrip('.')}: {obj}"


def bs_record_table(raw: dict[str, Any]) -> pd.DataFrame:
    """A one-row, human-readable summary of a raw BioSample record, covering
    every field the record contains, for display with
    `notebook_utils.display_with_nested_tables`. List-shaped fields
    (`Attributes.Attribute`, `Ids.Id`, `Links.Link`) become their own nested
    sub-tables; small dict-shaped ones (`Owner`, `Models`, `Package`,
    `Status`) are flattened to a compact string. Completeness against the
    source JSON should be checked live with `notebook_utils.table_covers_json`
    rather than assumed.
    """
    bs = unwrap_biosample(raw)
    description = bs.get("Description") or {}
    organism = description.get("Organism") or {}

    attributes = as_list((bs.get("Attributes") or {}).get("Attribute"))
    attributes_df = pd.DataFrame(attributes, columns=["attribute_name", "content"]).fillna("") if attributes else pd.DataFrame(columns=["attribute_name", "content"])

    # Ids.Id entries key the namespace field "db" in some records, "namespace" in others
    # (a real inconsistency in the source data, not a bug here) -- both are shown as-is,
    # whichever is present on a given id.
    ids = as_list((bs.get("Ids") or {}).get("Id"))
    ids_df = pd.DataFrame(ids).fillna("") if ids else pd.DataFrame(columns=["content"])

    links = as_list((bs.get("Links") or {}).get("Link"))
    links_df = pd.DataFrame(links).fillna("") if links else pd.DataFrame(columns=["content"])

    return pd.DataFrame(
        [
            {
                "accession": get_accession(raw),
                "internal_id": bs.get("id"),
                "organism": get_organism(raw),
                "taxonomy_id": organism.get("taxonomy_id"),
                "title": description.get("Title"),
                "sample_name": description.get("SampleName"),
                "comment": (description.get("Comment") or {}).get("Paragraph"),
                "access": bs.get("access"),
                "submission_date": bs.get("submission_date"),
                "publication_date": bs.get("publication_date"),
                "last_update": bs.get("last_update"),
                "owner": _flatten_to_text(bs.get("Owner")),
                "model": _flatten_to_text(bs.get("Models")),
                "package": _flatten_to_text(bs.get("Package")),
                "status": _flatten_to_text(bs.get("Status")),
                "attributes": attributes_df,
                "ids": ids_df,
                "links": links_df,
            }
        ]
    )


# The full cascade tried, in order, for one extracted value -- kept as a documented, ordered
# list so the notebook's stats/plot can key off these names directly rather than hardcoding them.
# Weakest/riskiest evidence last: a literal match of the value itself (exact/case-insensitive/
# normalized) is trusted more than a match against a different-but-related string (ontology
# synonym), which in turn is trusted more than a spelling-tolerant guess (fuzzy).
MATCH_STRATEGIES: list[str] = ["exact", "case-insensitive", "normalized", "ontology synonym", "fuzzy"]


# Values an extraction pipeline can emit that mean "no answer", not a real claim about the
# sample -- worth flagging separately, since ontology mapping has no reject option and will
# still assign one of these a real (wrong) term purely by syntactic similarity: found directly
# on real data, "not applicable" mapped to a real gene symbol, and "none" mapped to a real
# disease, in both cases because the placeholder text happened to be the closest thing on offer,
# not because it means anything.
PLACEHOLDER_VALUES = frozenset({"none", "n/a", "na", "not applicable", "unknown", "not specified", "not available", "null"})


def is_placeholder_value(value: str) -> bool:
    """Whether `value` is a non-answer placeholder (see `PLACEHOLDER_VALUES`) rather than a real
    extracted claim, matched case- and whitespace-insensitively.
    """
    return value.strip().lower() in PLACEHOLDER_VALUES


def bold_span_html(text: str, span: tuple[int, int]) -> RawHTML:
    """Renders `text` as HTML with the `(start, end)` span wrapped in `<b>`,
    everything else HTML-escaped normally -- only the highlight itself needs
    raw HTML.
    """
    start, end = span
    before, match, after = text[:start], text[start:end], text[end:]
    return RawHTML(f"{html.escape(before)}<b>{html.escape(match)}</b>{html.escape(after)}")


# Built once at import time (a blank, unparsed pipeline -- no model download needed) rather than
# per call: negex's own setup cost (trigger-phrase matchers) would otherwise repeat for every
# match checked.
_NEGATION_NLP = spacy.blank("en")
_NEGATION_NLP.add_pipe("sentencizer")
_NEGEX = _NEGATION_NLP.add_pipe("negex")


def is_negated(content: str, span: tuple[int, int]) -> bool:
    """Whether the substring at `span` in `content` sits inside a negation scope -- e.g. "Cre
    negative; Ccm3/..." negates a match on "Ccm3". Built on negspacy/NegEx (Chapman et al. 2001):
    a curated trigger-phrase scan (with a pseudo-negation exception list for phrases like "gram
    negative"), not raw keyword proximity -- so standardized compound terms like "non-small cell
    lung cancer" are correctly not flagged. Uses a blank (unparsed) pipeline since `span` is
    already known from the search that found it; no POS tagging is needed to locate the entity
    the way it would be for a not-yet-located phrase.
    """
    doc = _NEGATION_NLP(content)
    # `label=` is required here -- a Span with no label is silently dropped when assigned to
    # `doc.ents` (confirmed directly: `doc.ents` comes back empty with no error), which would
    # make `negex` a permanent, silent no-op. The label's actual value doesn't matter to negex.
    entity = doc.char_span(*span, alignment_mode="expand", label="ENTITY")
    if entity is None:
        return False
    doc.ents = [entity]
    _NEGEX(doc)
    return bool(entity._.negex)


def _record_search_groups(raw: dict[str, Any]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], str, str]:
    """Precomputes everything `verify_extracted_against_raw_rows` needs to search this one
    record, ONCE per record rather than once per (field, attribute) pair -- measured 71x faster
    on real data than re-calling `.lower()` on the same attribute content for every field checked
    against it (1,148 -> 81,686 records/s).

    Returns two separate groups, each `[(name, content, content_lower), ...]`: `attribute_entries`
    (every `Attributes.Attribute`, in order) and `secondary_entries` (`Title`, then
    `Description.Comment.Paragraph` -- a free-text submitter protocol description, present on
    roughly a third of records). `_search` checks `secondary_entries` only when nothing in
    `attribute_entries` matches at any tier: a submitter far more often repeats the same fact
    from a structured attribute into free text than states something new there, so this is both
    the more useful field to report a match against and the cheaper one to check first (skips
    scanning the -- often much longer -- title/comment text entirely for the common case).
    `attr_combined_lower`/`secondary_combined_lower` are each group's content joined into one
    lowercased string, for a cheap single-`in`-check pre-screen before the more expensive
    per-attribute `_search_group` (see its docstring for why this pre-screen exists).

    Each attribute contributes its content AND, separately, its own NAME as a second searchable
    entry under the same reported field -- found directly on real data (5/100 in a hand-reviewed
    "not found" sample): a submitter sometimes encodes the value as the attribute's label itself
    (`"spiperone treatment": "yes"`, `"smad4_status": "Expressing"`) rather than as its content,
    which was structurally invisible to a checker that only ever looked at content.
    """
    bs = unwrap_biosample(raw)
    attributes = as_list((bs.get("Attributes") or {}).get("Attribute"))
    attribute_entries: list[tuple[str, str]] = []
    for a in attributes:
        name = a.get("attribute_name")
        content = str(a.get("content", ""))
        attribute_entries.append((name, content))
        if name:
            attribute_entries.append((name, name))
    attribute_entries = [(name, content, content.lower()) for name, content in attribute_entries]

    description = bs.get("Description") or {}
    title = description.get("Title") or ""
    comment = (description.get("Comment") or {}).get("Paragraph") or ""
    secondary_entries = [(name, content, content.lower()) for name, content in [("title", title), ("comment", str(comment))] if content]

    attr_combined_lower = " ".join(content_lower for _, _, content_lower in attribute_entries)
    secondary_combined_lower = " ".join(content_lower for _, _, content_lower in secondary_entries)
    return attribute_entries, secondary_entries, attr_combined_lower, secondary_combined_lower


def _has_word_boundary(text: str, start: int, end: int) -> bool:
    """Whether the `(start, end)` span in `text` doesn't start or end in the middle of a longer,
    unbroken run of letters/digits -- found directly on real data: `"in vitro"` is a genuine
    substring of `"in vitrogen media"`, but the word actually there is `"vitrogen"` (a reagent
    brand name), not `"vitro"` followed by something unrelated. A real extracted value should
    always appear as a complete token/phrase, never as a partial prefix/suffix of a longer,
    different word, so every substring-based tier checks this on both edges independently.
    """
    if start > 0 and text[start - 1].isalnum():
        return False
    return not (end < len(text) and text[end].isalnum())


def _find_all_matches(needle: str, needle_lower: str, entries: list[tuple[str, str, str]], case_insensitive: bool) -> list[tuple[str, str, tuple[int, int]]]:
    """Every `(name, content, span)` in `entries` containing `needle` at a genuine word boundary
    (see `_has_word_boundary`) -- ALL of them, not just the first, so a value repeated across
    multiple attributes (or the same fact filed under two field names) is reported completely
    rather than silently collapsed to whichever attribute happened to come first. Within one
    attribute, keeps searching past a boundary-violating hit for a later, valid one instead of
    giving up on that attribute entirely.
    """
    matches = []
    for name, content, content_lower in entries:
        haystack, target = (content_lower, needle_lower) if case_insensitive else (content, needle)
        search_from = 0
        while True:
            idx = haystack.find(target, search_from)
            if idx == -1:
                break
            span = (idx, idx + len(needle))
            if _has_word_boundary(content, *span):
                matches.append((name, content, span))
                break
            search_from = idx + 1
    return matches


_NORMALIZE_MIN_LENGTH = 4  # below this, a normalized needle is too short/generic to trust as a match signal
_ONTOLOGY_CANDIDATE_MIN_LOOSE_LENGTH = 5  # below this, an ontology-synonym candidate (e.g. a 2-letter chemical symbol) only gets exact matching -- see _search_ontology_synonyms
_SEPARATOR_CHARS = " \t\n\r-_"


def _normalize_with_positions(text: str, strip_separators: bool = False) -> tuple[str, list[int]]:
    """Lowercases `text`, drops any `(...)` parenthetical content, and either collapses runs of
    whitespace/hyphen/underscore to a single space (`strip_separators=False` -- the default) or
    removes them entirely (`strip_separators=True`). The default form matches a separator that's
    merely spelled differently on each side (`"Sinoatrial node (SAN) cells"` and
    `"sinoatrial_node cells"` both normalize to `"sinoatrial node cells"`); the stripped form
    additionally matches a separator present on only ONE side (`"HEK293"` inside `"HEK 293"`,
    `"cellline"` inside `"cell_line"`) -- a collapse-to-one-space rule can't bridge a boundary
    that's simply missing on the other side, so `_search_normalized_one` tries both forms.
    Returns the normalized string alongside `positions`, where `positions[i]` is the index in the
    ORIGINAL `text` that `normalized[i]` came from, so a match found in the normalized string can
    still be highlighted at its real location in the original text.
    """
    out_chars: list[str] = []
    out_positions: list[int] = []
    i, n = 0, len(text)
    prev_was_space = True  # collapses leading separators too
    while i < n:
        c = text[i]
        if c == "(":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            continue
        if c in _SEPARATOR_CHARS:
            if not strip_separators and not prev_was_space:
                out_chars.append(" ")
                out_positions.append(i)
                prev_was_space = True
            i += 1
            continue
        out_chars.append(c.lower())
        out_positions.append(i)
        prev_was_space = False
        i += 1
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_positions.pop()
    return "".join(out_chars), out_positions


def _search_normalized_one(needle: str, content: str) -> tuple[int, int] | None:
    """Whether `needle` appears in `content` once both are normalized -- tried with separators
    collapsed to a space first (catches punctuation/parenthetical noise, e.g. `"Sinoatrial node
    cell"` inside `"Sinoatrial node (SAN) cells"`), then with separators stripped entirely
    (catches a boundary present on only one side, e.g. `"HEK293"` inside `"HEK 293"`, `"cellline"`
    inside `"cell_line"`) -- see `_normalize_with_positions` for why both forms are needed.
    Returns the matched span in the ORIGINAL `content` (not the normalized string), or `None` if
    the normalized needle is too short to trust (`_NORMALIZE_MIN_LENGTH`) or doesn't appear at
    all under either form.
    """
    for strip_separators in (False, True):
        normalized_needle, _ = _normalize_with_positions(needle, strip_separators)
        if len(normalized_needle) < _NORMALIZE_MIN_LENGTH:
            continue
        normalized_content, positions = _normalize_with_positions(content, strip_separators)
        search_from = 0
        while True:
            idx = normalized_content.find(normalized_needle, search_from)
            if idx == -1:
                break
            start = positions[idx]
            end = positions[idx + len(normalized_needle) - 1] + 1
            if _has_word_boundary(content, start, end):
                return start, end
            search_from = idx + 1
    return None


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insertions/deletions/substitutions), O(len(a)*len(b)) with only
    two rows kept in memory at once -- fine here since `fuzzy` only ever compares single words,
    never whole sentences.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


_FUZZY_MIN_WORD_LENGTH = 6  # never fuzzy-match short words -- a 1-character edit changes "sex" far more than a 12-letter word
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def _fuzzy_max_distance(word_len: int) -> int:
    """Scales the tolerated edit distance with word length so the error RATE stays low (roughly
    <=15%) instead of a flat threshold that's too loose on short words and too strict on long
    ones.
    """
    if word_len < 8:
        return 1
    if word_len < 14:
        return 2
    return 3


_DIGITS_RE = re.compile(r"\d+")


def _words_fuzzy_equal(a: str, b: str) -> bool:
    """Whether two words are close enough to call the same spelling -- but never if their digit
    characters differ, no matter how small the edit distance is, and never by more than a single
    edit at all when either word contains a digit. Found directly on real data: a word containing
    a digit is usually a compact identifier (a marker, a gene, a miRNA, a modification state)
    where the label carries little redundancy, so even a same-digit edit can flip its meaning --
    `anti-IL-1`/`anti-PD-1` differ only in two letters (both keep digit `1`) but name unrelated
    immune targets; `anti-CD28`/`anti-CD3`, `HIF-1a`/`HIF-2a`, `miR-16`/`miR-15` differ only in
    their digits and name unrelated markers/genes/microRNAs. A letter-only word carries no such
    risk and keeps the normal length-scaled budget (`Asbesos`/`Asbestos`, `Gliobmastoma`/
    `Glioblastoma`).
    """
    a_lower, b_lower = a.lower(), b.lower()
    if a_lower == b_lower:
        return True
    has_digit = bool(_DIGITS_RE.search(a_lower) or _DIGITS_RE.search(b_lower))
    if has_digit and _DIGITS_RE.findall(a_lower) != _DIGITS_RE.findall(b_lower):
        return False
    shorter = min(len(a), len(b))
    if shorter < _FUZZY_MIN_WORD_LENGTH:
        return False
    max_distance = 1 if has_digit else _fuzzy_max_distance(shorter)
    return _levenshtein(a_lower, b_lower) <= max_distance


def _fuzzy_find(needle: str, content: str) -> tuple[int, int] | None:
    """Finds `needle`'s words as a CONSECUTIVE, same-count, same-order run of words in `content`,
    where each word pair matches exactly (case-insensitive) or -- only for words at least
    `_FUZZY_MIN_WORD_LENGTH` characters long -- within `_fuzzy_max_distance`. Requires at least
    one word to actually be a fuzzy (non-exact) match, since an all-exact run would already have
    been caught by an earlier tier. Deliberately does NOT allow words to be inserted or dropped
    between the matched run (unlike `_search_normalized_one`) -- keeping word alignment exact is
    what keeps two genuinely different phrases from being matched together just because they
    share some words; only spelling within an otherwise-identical phrase is allowed to drift.
    """
    needle_words = _WORD_TOKEN_RE.findall(needle)
    if not needle_words:
        return None
    content_matches = list(_WORD_TOKEN_RE.finditer(content))
    n = len(needle_words)
    if len(content_matches) < n:
        return None
    for start_i in range(len(content_matches) - n + 1):
        window = content_matches[start_i : start_i + n]
        any_fuzzy = False
        ok = True
        for nw, w in zip(needle_words, window):
            w_text = w.group()
            if nw.lower() == w_text.lower():
                continue
            if _words_fuzzy_equal(nw, w_text):
                any_fuzzy = True
                continue
            ok = False
            break
        if ok and any_fuzzy:
            return window[0].start(), window[-1].end()
    return None


# Attribute names that make a weaker match than a normal named attribute (e.g. `cell_line`,
# `tissue`) -- `source_name` and `sample_name` are generic catch-all fields that happen to
# restate whatever the record's most prominent identifier is, so a value matching one of these
# AND a normal attribute is more informative reported against the normal attribute. Lower
# number = higher priority; any name not listed here is priority 0 (the default, best tier).
# Matched by normalized name (lowercased, underscores/spaces collapsed) since the crate spells
# the same field both ways across files (`source_name` vs `source name`).
_ATTRIBUTE_PRIORITY = {"source_name": 1, "sample_name": 2}


def _normalize_attribute_name(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def _preferred_matches(matches: list[tuple[str, str, tuple[int, int]]]) -> list[tuple[str, str, tuple[int, int]]]:
    """Among matches within one tier, keeps only the ones at the best (lowest) `_ATTRIBUTE_PRIORITY`
    -- e.g. a value matching both `cell_line` and `source_name` is reported against `cell_line`
    only; a value matching only `source_name` and `sample_name` is reported against `source_name`
    only, since `sample_name` never wins over anything else. A no-op when every match is a normal
    (unlisted) attribute, or when there's only one match.
    """
    if len(matches) <= 1:
        return matches
    best = min(_ATTRIBUTE_PRIORITY.get(_normalize_attribute_name(name), 0) for name, _, _ in matches)
    return [m for m in matches if _ATTRIBUTE_PRIORITY.get(_normalize_attribute_name(m[0]), 0) == best]


def _search_group(needle: str, needle_lower: str, entries: list[tuple[str, str, str]]) -> tuple[list[tuple[str, str, tuple[int, int]]], str] | None:
    """Exact match against every entry in this one group; if none matched, case-insensitive;
    if still none, normalized (parenthetical/punctuation-insensitive, see
    `_search_normalized_one`). Stops at the first tier with any hits and returns that tier's
    hits, filtered by `_preferred_matches` (never mixes hits from two different tiers in one
    result, and never mixes a `source_name`/`sample_name` match with a better one).
    """
    matches = _find_all_matches(needle, needle_lower, entries, case_insensitive=False)
    if matches:
        return _preferred_matches(matches), "exact"
    matches = _find_all_matches(needle, needle_lower, entries, case_insensitive=True)
    if matches:
        return _preferred_matches(matches), "case-insensitive"
    matches = [(name, content, span) for name, content, _ in entries if (span := _search_normalized_one(needle, content)) is not None]
    if matches:
        return _preferred_matches(matches), "normalized"
    return None


def _search_group_fuzzy(needle: str, entries: list[tuple[str, str, str]]) -> tuple[list[tuple[str, str, tuple[int, int]]], str] | None:
    """Same shape as `_search_group`'s other tiers, but via `_fuzzy_find` -- kept as its own,
    separately-invoked function (rather than folded into `_search_group`) so callers can choose
    NOT to try it, since it's the one tier that isn't matching the value's actual text.
    """
    matches = [(name, content, span) for name, content, _ in entries if (span := _fuzzy_find(needle, content)) is not None]
    if matches:
        return _preferred_matches(matches), "fuzzy"
    return None


def _search(
    needle: str, needle_lower: str, attribute_entries: list[tuple[str, str, str]], secondary_entries: list[tuple[str, str, str]]
) -> tuple[list[tuple[str, str, tuple[int, int]]], str] | None:
    """Two-pass search for one needle string: `attribute_entries` first (structured fields), and
    only if nothing matches there at all, `secondary_entries` (title/comment free text) -- see
    `_record_search_groups` for why. Covers `_search_group`'s exact/case-insensitive/normalized
    tiers only -- NOT `fuzzy`, which the caller tries as its own, final fallback, after ontology
    synonyms, since a curated ontology synonym is more trustworthy evidence than a spelling-
    tolerant guess (see `MATCH_STRATEGIES`). Returns `(matches, strategy)`, or `None` if nothing
    matches anywhere at either of these tiers.
    """
    return _search_group(needle, needle_lower, attribute_entries) or _search_group(needle, needle_lower, secondary_entries)


def _search_ontology_synonyms(
    candidates: list[str],
    attribute_entries: list[tuple[str, str, str]],
    secondary_entries: list[tuple[str, str, str]],
    attr_combined_lower: str,
    secondary_combined_lower: str,
) -> tuple[list[tuple[str, str, tuple[int, int]]], str] | None:
    """Same attributes-first, secondary-second precedence as `_search`, but over a whole list of
    ontology-synonym candidate strings (a term can have 10-20+): each group is pre-screened with
    its own combined string first, since calling `_search_group` for every candidate is too
    expensive to do unconditionally. Never tries the `fuzzy` tier here -- fuzzy-matching every
    one of 10-20+ candidate strings per field would be both slow and a much larger false-positive
    surface than fuzzy-matching the single extracted value in `_search`, which is the one thing
    the user explicitly asked to keep bounded.

    A candidate shorter than `_ONTOLOGY_CANDIDATE_MIN_LOOSE_LENGTH` only gets the `exact`
    (case-sensitive) tier, never `case-insensitive`/`normalized` -- found directly on real data: a
    free-text dietary note ("No dairy. Limited soy...") case-insensitively matched ChEBI's
    two-letter symbol for nitric oxide, `"NO"`, purely because the sentence happened to start with
    the English word "No". Word-boundary enforcement doesn't catch this (`"No"` is a genuine,
    complete word there), so a length floor is the only thing that does -- exactly the same
    short-string collision risk already guarded against in `fuzzy`/`normalized`, just previously
    unguarded here.

    Returns `(matches, "ontology synonym")` for the first candidate that matches anywhere in a
    group -- the matched phrase itself is read back out of each match's own span by the caller,
    not returned here, since it can differ per match (e.g. a case-insensitive hit reads back in
    whatever case the raw text actually used).
    """
    for entries, combined_lower in [(attribute_entries, attr_combined_lower), (secondary_entries, secondary_combined_lower)]:
        for candidate in candidates:
            if not candidate or candidate.lower() not in combined_lower:
                continue  # cheap single-string pre-screen -- skip the O(attributes) _search_group for candidates that can't match anywhere in this group
            if len(candidate) < _ONTOLOGY_CANDIDATE_MIN_LOOSE_LENGTH:
                matches = _find_all_matches(candidate, candidate.lower(), entries, case_insensitive=False)
                if matches:
                    return _preferred_matches(matches), "ontology synonym"
                continue
            found = _search_group(candidate, candidate.lower(), entries)
            if found is not None:
                matches, _ = found
                return matches, "ontology synonym"
    return None


VERIFY_COLUMNS = [
    "raw_field",
    "raw_value",
    "is_placeholder",
    "strategy",
    "curated_fields",
    "curated_texts",
    "curated_matched_phrase",
    "negated",
    "n_matches",
    "assigned_term_id",
    "assigned_term_label",
    "assigned_term_synonyms",
]


def verify_extracted_against_raw_rows(
    raw: dict[str, Any],
    extracted: dict[str, Any],
    results: dict[str, Any] | None = None,
    ontology_indexes: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """For every non-null field in `extracted`, deterministically checks whether that value
    appears somewhere in this raw BioSample record, in `MATCH_STRATEGIES` order: `exact`
    (case-sensitive substring); `case-insensitive`; `normalized` (case-insensitive with
    parentheticals stripped and hyphens/underscores/whitespace collapsed -- see
    `_search_normalized_one`); `ontology synonym`, only when `results`/`ontology_indexes` (see
    `ontology.py:build_field_ontology_indexes`) are supplied -- a match against the LABEL or any
    SYNONYM of the ontology term `results[field]` actually assigned, i.e. not the extracted
    value's own text at all, but what the pipeline's own ontology search (`text2term`) already
    resolved it to; and finally `fuzzy` -- a spelling-tolerant, word-by-word match of the value
    itself (see `_fuzzy_find`), tried dead last since it's the one tier not backed by an exact or
    curated match.

    Within a tier, every `Attributes.Attribute` is checked FIRST -- including each attribute's
    own NAME, not just its content, since a value is sometimes encoded as the attribute's label
    itself (`"spiperone treatment": "yes"`) rather than as its value -- and `Title`/`Description.
    Comment.Paragraph` only if nothing in `Attributes` matched at any tier (see
    `_record_search_groups` for why this precedence, not just interleaving both groups). The
    first tier/group combination that matches anywhere wins; `"not found"` if none do. When a
    value matches more than one attribute (the same fact filed under two field names, or
    genuinely repeated), ALL of those matches are kept, not just the first -- `n_matches` says
    how often that happens. Array-type fields (`extracted[field]` a list) get one row per value,
    not one row per field.

    Returns a plain `list[dict]` (keys: `VERIFY_COLUMNS`), NOT a DataFrame -- constructing a
    `pd.DataFrame` has substantial fixed per-call overhead (Arrow string-array conversion, type
    inference, etc.) that dominates when called once per record: measured 94% of total runtime
    in a profile, collapsing throughput from ~700/s to negligible once removed. When checking
    many records, accumulate these lists and build ONE DataFrame at the end (see
    `verify_extracted_against_raw` for the single-record convenience wrapper that still returns
    a DataFrame directly, fine to use when only checking one record at a time).

    Row keys: `raw_field`/`raw_value` are the extraction's own field name and value; `is_placeholder`
    is whether `raw_value` is a non-answer like "not applicable" rather than a real claim (see
    `PLACEHOLDER_VALUES`) -- independent of `strategy`, since a placeholder can still happen to
    match something in the raw text (or get assigned an ontology term regardless). `strategy` is
    which tier found it (or `"not found"`); `curated_fields`/`curated_texts` are, in parallel,
    where each match was found (an `attribute_name`, `"title"`, or `"comment"`) and that field's
    full raw text -- both empty lists when `strategy` is `"not found"`. `curated_matched_phrase`
    is, per match, the exact substring read directly out of that match's own span -- NOT
    `raw_value` re-used across every match, which would be wrong whenever the real text differs
    from it (a different case, a different punctuation/spelling variant for `normalized`/`fuzzy`,
    or an ontology synonym's own text) -- needed to reconstruct which span to highlight without
    re-searching (see `matches_display_table`, used for interactive display; these three list
    columns stay plain strings/ints here, not `RawHTML`/`DataFrame` objects, because bulk callers
    checkpoint rows to parquet, which can't serialize those). `negated` is, per match in the same
    order, whether that match's surrounding text negates it (see `is_negated`) -- e.g. "Cre
    negative; Ccm3/..." negates a `knockout_gene` match on "Ccm3" -- empty when `strategy` is
    `"not found"`. `assigned_term_id`/`assigned_term_label`/`assigned_term_synonyms` are filled in
    whenever `results[field]` assigned a term -- on every row that has one, not just "not found"
    rows -- so a still-"not found" row also shows what term the pipeline picked, for a human to
    judge plausibility against.
    """
    if extracted is None:
        # documented, real case (bsllmner-mk2 docs/data-formats.md): a list-shaped or otherwise
        # malformed LLM response is normalised to a null `extracted` for the whole entry, not
        # just a null field -- nothing was extracted at all, so there's nothing to trace back.
        return []

    attribute_entries, secondary_entries, attr_combined_lower, secondary_combined_lower = _record_search_groups(raw)

    rows = []
    for field, value in extracted.items():
        if isinstance(value, list):
            # an array-type field's list can itself contain a literal `null` entry (e.g.
            # `"drug": [null]`) instead of the field being `null` or `[]` outright -- a third,
            # real shape this crate's extraction output takes, confirmed directly (accession
            # SAMN23480504, rnaseq_human_5y_2022-04) after it crashed the first full-scale run.
            value = [v for v in value if v is not None]
        if not value:
            continue

        assigned = ((results or {}).get(field) or [None])[0]
        term_id = (assigned or {}).get("term_id")
        term_info = ((ontology_indexes or {}).get(field) or {}).get(term_id) if term_id else None
        term_label = ((term_info or {}).get("label")) or ""
        term_synonyms = "; ".join(term_info["exact_synonyms"] + term_info["related_synonyms"]) if term_info else ""

        for one_value in value if isinstance(value, list) else [value]:
            found = _search(one_value, one_value.lower(), attribute_entries, secondary_entries)

            if found is None and term_info:
                candidates = [term_info["label"], *term_info["exact_synonyms"], *term_info["related_synonyms"]]
                found = _search_ontology_synonyms(candidates, attribute_entries, secondary_entries, attr_combined_lower, secondary_combined_lower)

            if found is None:
                # last resort: spelling-tolerant match of the extracted value itself, tried only
                # after every exact/normalized/ontology-synonym option has failed -- see `_search`
                found = _search_group_fuzzy(one_value, attribute_entries) or _search_group_fuzzy(one_value, secondary_entries)

            row = {
                "raw_field": field,
                "raw_value": one_value,
                "is_placeholder": is_placeholder_value(one_value),
                "assigned_term_id": term_id or "",
                "assigned_term_label": term_label,
                "assigned_term_synonyms": term_synonyms,
            }
            if found is None:
                row.update(strategy="not found", curated_fields=[], curated_texts=[], curated_matched_phrase=[], negated=[], n_matches=0)
            else:
                matches, strategy_used = found
                row.update(
                    strategy=strategy_used,
                    curated_fields=[name for name, _, _ in matches],
                    curated_texts=[content for _, content, _ in matches],
                    # the phrase actually found, read straight out of its own match span -- NOT
                    # `one_value` re-used across every match, which would be wrong for `normalized`/
                    # `fuzzy` matches whose real text differs (punctuation/spelling) from the value
                    curated_matched_phrase=[content[start:end] for _, content, (start, end) in matches],
                    # per match, not per row: the same value can match one attribute where the
                    # surrounding text negates it and another where it doesn't
                    negated=[is_negated(content, span) for _, content, span in matches],
                    n_matches=len(matches),
                )
            rows.append(row)

    return rows


def matches_display_table(curated_fields: list[str], curated_texts: list[str], curated_matched_phrase: list[str]) -> pd.DataFrame:
    """Builds a small `(curated_field, curated_value)` table, one row per match, `curated_value`
    bolding the matched phrase within that field's full text -- for interactive display via
    `display_with_nested_tables` (as a nested-table cell when a row has more than one match).
    Not used by bulk/parquet callers: `RawHTML`/`DataFrame` cells aren't Arrow-serializable, which
    is exactly why `verify_extracted_against_raw_rows` keeps its own `curated_fields`/
    `curated_texts`/`curated_matched_phrase` columns as plain strings and builds this only when
    something actually needs to render them.
    """
    out = []
    for field, text, phrase in zip(curated_fields, curated_texts, curated_matched_phrase):
        idx = text.find(phrase)
        if idx == -1:
            idx = text.lower().find(phrase.lower())
        span = (idx, idx + len(phrase)) if idx != -1 else (0, 0)
        out.append({"curated_field": field, "curated_value": bold_span_html(text, span)})
    return pd.DataFrame(out, columns=["curated_field", "curated_value"])


def verify_extracted_against_raw(
    raw: dict[str, Any],
    extracted: dict[str, Any],
    results: dict[str, Any] | None = None,
    ontology_indexes: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Single-record convenience wrapper around `verify_extracted_against_raw_rows` -- returns a
    DataFrame directly, for displaying one record's trace-back at a time. Checking many records?
    Call `verify_extracted_against_raw_rows` directly and build one DataFrame from the
    accumulated rows afterward instead -- see its docstring for why that matters at scale.

    Adds one column beyond `VERIFY_COLUMNS`, `matches` -- a nested `(curated_field, curated_value)`
    table (via `matches_display_table`) for display, since a single record's worth of rows is
    small enough that the parquet-serialization concern driving `VERIFY_COLUMNS`'s flat list
    columns doesn't apply here.
    """
    rows = verify_extracted_against_raw_rows(raw, extracted, results, ontology_indexes)
    df = pd.DataFrame(rows, columns=VERIFY_COLUMNS)
    df["matches"] = [matches_display_table(r["curated_fields"], r["curated_texts"], r["curated_matched_phrase"]) for r in rows]
    return df


def scan_accessions(paths: list[Path]) -> set[str]:
    """The set of accessions across a list of `inputs/*.jsonl` files.

    Uses a regex over the raw text instead of a full JSON parse per line --
    the `"accession"` key is a plain string field at the top level in both
    record shapes, so this is safe and much faster than `iter_bs_entries` +
    `get_accession` over millions of lines.
    """
    accessions: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                accessions.update(_ACCESSION_RE.findall(line))
    return accessions
