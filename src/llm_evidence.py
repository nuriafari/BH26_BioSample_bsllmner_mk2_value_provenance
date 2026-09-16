"""LLM tier for the trace-back check, applied only to rows the deterministic cascade in
`bs_entries.py` already marked `"not found"` (exact/case-insensitive/normalized/ontology-synonym/
fuzzy all failed -- see `MATCH_STRATEGIES`).

Same output shape as the deterministic check (`curated_fields`/`curated_texts`/
`curated_matched_phrase`, rendered with the same `matches_display_table`), so the two tiers are a
drop-in continuation of each other, not a parallel system: the LLM is asked to find evidence
in the record's full text (which can bridge abbreviations, construct-name prefixes, or a value
split across two attributes -- gaps the deterministic tiers can't close), but every quote it
returns is independently re-checked with the SAME substring search the deterministic cascade
itself uses (`bs_entries._search`) before being trusted. This is the field-standard "LLM proposes,
deterministic check disposes" split (see the `grounding_verification_literature_design` memory,
modeled on `previous_work/scripts/common/evidence_grounding.py`'s Validator A/B pattern) -- an LLM
that hallucinates a quote is caught here regardless of how confident it sounds, because the check
never trusts the model's own claim that something is "verbatim."

Calls the SAME open-source model the crate's own extraction pipeline used --
`mistral-small3.1:24b`, served locally by Ollama (see `data/2026-06_mistral-small3.1-24b/README.md`:
"Runs used `bsllmner2_select` with `mistral-small3.1:24b` served by Ollama, `--num-ctx 4096`") --
rather than a hosted API: this keeps the validator in the same model family/weight class as the
extractor it's checking (a genuinely different model would also work, but there's no reason to pay
for a hosted frontier model when the project already has GPU capacity and the exact open-weight
model on hand), and it costs nothing per call beyond GPU time this environment already has.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from bs_entries import _record_search_groups, _search, as_list, unwrap_biosample

DEFAULT_MODEL = "mistral-small3.1:24b"
OLLAMA_HOST = "http://127.0.0.1:11434"
# Matches the extraction pipeline's own `--num-ctx 4096` (see module docstring) -- this tier's
# prompts (one record's text + a handful of field/value pairs) are well under that.
NUM_CTX = 4096

_SYSTEM_PROMPT = "You are a careful fact-checking assistant for biomedical metadata curation."

# Ollama's structured-output mode (`format` as a JSON Schema, constrained decoding -- supported
# regardless of whether the underlying model was trained for tool use) -- one verdict per
# requested item, matched back by a plain echoed integer `item_index` rather than by echoing the
# field/value strings themselves: measured directly against this model (unlike Claude in an
# earlier version of this tier), `mistral-small3.1:24b` sometimes echoes the whole labeled token
# (`'field="knockout_gene"'`) instead of just the value inside it when asked to echo a quoted
# string -- a plain integer has no such formatting ambiguity, and a call here only ever covers a
# handful of items (one record's own not-found fields), so trusting position via an explicit,
# model-stated index is both simpler and more robust than previous_work's string-echo approach
# (justified there by calls covering many more claims at once).
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_index": {"type": "integer", "description": "the number of this item from the numbered list above"},
                    "found": {"type": "boolean", "description": "true only if exact text above establishes this value"},
                    "quotes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "if found=true: one or more EXACT verbatim substrings copied from the text above",
                    },
                },
                "required": ["item_index", "found", "quotes"],
            },
        }
    },
    "required": ["verdicts"],
}


def _field_lines(raw: dict[str, Any]) -> list[str]:
    """One `"- {field}: {content}"` line per attribute/title/comment, for the prompt only -- a
    plain, deduplicated rendering (unlike `_record_search_groups`'s `attribute_entries`, which
    deliberately ALSO adds each attribute's own name as a second searchable entry for the
    deterministic cascade; that trick would just show as a confusing duplicate line here).
    """
    bs = unwrap_biosample(raw)
    attributes = as_list((bs.get("Attributes") or {}).get("Attribute"))
    lines = [f"- {a.get('attribute_name')}: {a.get('content')}" for a in attributes]

    description = bs.get("Description") or {}
    title = description.get("Title") or ""
    comment = (description.get("Comment") or {}).get("Paragraph") or ""
    if title:
        lines.append(f"- title: {title}")
    if comment:
        lines.append(f"- comment: {comment}")
    return lines


def build_prompt(raw: dict[str, Any], items: list[tuple[str, str]]) -> str:
    """`items`: `(raw_field, raw_value)` pairs already marked `"not found"` for this one record."""
    text_block = "\n".join(_field_lines(raw))
    items_block = "\n".join(f"{i}. {field} = {value}" for i, (field, value) in enumerate(items, 1))
    return f"""Below is a BioSample record's searchable text, one field per line:
{text_block}

A deterministic exact/case-insensitive/normalized substring search already failed to find the values below anywhere in this text. For each one, decide whether the text still provides genuine evidence for it -- either stated directly, or a specific, well-supported inference (e.g. an abbreviation, a construct name like "shBRD9" implying a knockdown of BRD9, or a value whose two halves are split across two different fields).

Values to check:
{items_block}

For each value: set found=true only if you can point to exact, verbatim text above that establishes it. If found=true, quotes must be one or more EXACT substrings copied character-for-character from the field text above (not paraphrased, not corrected, not re-punctuated) -- each quote copied from a single field's text; use more than one quote when the evidence is split across different fields. If you cannot find a genuine verbatim quote, set found=false and quotes=[] -- do NOT invent a quote just because you believe the value is probably true, and do NOT rely on outside world knowledge that goes beyond what this text itself supports.

Return one verdict per numbered item above, with `item_index` set to that item's number."""


def call_llm(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Runs one call against a local Ollama server's `/api/chat` with schema-validated structured
    output. Returns `(structured_output, call_info)` -- `structured_output` is `None` on any
    failure (connection error, timeout, or a response that isn't valid JSON), so callers have one
    place to check for "the call itself didn't work" separately from "the model said not found."
    `call_info` (wall time plus Ollama's own prompt/output token counts) is always returned, for
    the notebook's call log -- no cost field, since this is self-hosted GPU inference, not a
    billed API call.
    """
    body = {
        "model": model,
        "messages": [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "format": _VERDICT_SCHEMA,
        "stream": False,
        "options": {"num_ctx": NUM_CTX},
    }
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return None, {"error": str(e)}

    call_info = {
        "duration_ms": data.get("total_duration", 0) / 1e6,
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
    }
    content = (data.get("message") or {}).get("content")
    if not content:
        call_info["error"] = f"no message content in response: {data}"
        return None, call_info
    try:
        return json.loads(content), call_info
    except json.JSONDecodeError as e:
        call_info["error"] = f"model response wasn't valid JSON despite the schema: {e}"
        return None, call_info


def _ground_quotes(raw: dict[str, Any], quotes: list[str]) -> tuple[list[tuple[str, str, tuple[int, int]]], list[str]]:
    """Independently re-checks each of the LLM's `quotes` against the record's actual text, via
    the SAME `_search` cascade the deterministic checker uses (exact / case-insensitive /
    normalized -- never `fuzzy` or `ontology synonym`: stacking a second fallible/curated-lookup
    tier on top of an already-fallible LLM claim would compound risk rather than bound it, per
    the `grounding_verification_literature_design` memory). Returns `(matches, ungrounded)`:
    `matches` in the same `(field, content, span)` shape `verify_extracted_against_raw_rows`
    produces (one entry per quote that grounds, across however many different fields); `ungrounded`
    is the quotes that did NOT verify -- i.e. the LLM cited text that doesn't actually appear
    anywhere in the record, a caught hallucination, kept for transparency rather than silently
    dropped.
    """
    attribute_entries, secondary_entries, _, _ = _record_search_groups(raw)
    matches: list[tuple[str, str, tuple[int, int]]] = []
    ungrounded: list[str] = []
    for quote in quotes:
        found = _search(quote, quote.lower(), attribute_entries, secondary_entries)
        if found is None:
            ungrounded.append(quote)
        else:
            matches.extend(found[0])
    return matches, ungrounded


def verify_not_found_with_llm(
    raw: dict[str, Any], not_found_items: list[tuple[str, str]], model: str = DEFAULT_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One call to the local model covering every `(raw_field, raw_value)` in `not_found_items` for
    this one record (batched per record, not per field -- same reasoning as `previous_work`'s
    Validator B: one call amortizes the fixed per-request overhead -- prompt-processing the
    record's text, model load state -- over every field this record still needs checked). Returns
    `(rows, call_info)`; `rows` has the same shape as
    `bs_entries.verify_extracted_against_raw_rows` plus three columns:

    - `strategy`: `"llm"` if at least one of the LLM's quotes grounded (see `_ground_quotes`),
      else `"not found"` (unchanged from the deterministic result) -- covers both "the LLM itself
      said not found" and "the LLM said found but every quote it cited failed grounding."
    - `llm_verdict`: the LLM's own raw found/not-found call, before grounding -- lets you see
      the caught-hallucination case (`llm_verdict=True`, `strategy="not found"`) separately from
      simple agreement.
    - `llm_ungrounded_quotes`: quotes the LLM cited that did NOT verify against the record's own
      text -- empty in the normal case; non-empty is the hallucination signal this tier exists to
      catch.

    If the call itself fails (`call_llm` returns `None`), every item is returned with
    `strategy="not found"` and `llm_verdict=None` (a call failure, not a "not found" verdict --
    kept distinguishable via `llm_verdict is None` rather than silently treated as agreement).
    """
    prompt = build_prompt(raw, not_found_items)
    structured, call_info = call_llm(prompt, model=model)

    by_index: dict[int, dict[str, Any]] = {}
    if structured is not None:
        for v in structured.get("verdicts", []):
            by_index[v["item_index"]] = v

    rows = []
    for i, (field, value) in enumerate(not_found_items, 1):
        verdict = by_index.get(i)
        if verdict is None:
            rows.append(
                {
                    "raw_field": field, "raw_value": value, "strategy": "not found", "llm_verdict": None,
                    "curated_fields": [], "curated_texts": [], "curated_matched_phrase": [], "n_matches": 0,
                    "llm_ungrounded_quotes": [],
                }
            )
            continue

        if not verdict["found"]:
            rows.append(
                {
                    "raw_field": field, "raw_value": value, "strategy": "not found", "llm_verdict": False,
                    "curated_fields": [], "curated_texts": [], "curated_matched_phrase": [], "n_matches": 0,
                    "llm_ungrounded_quotes": [],
                }
            )
            continue

        matches, ungrounded = _ground_quotes(raw, verdict["quotes"])
        rows.append(
            {
                "raw_field": field, "raw_value": value, "llm_verdict": True,
                "strategy": "llm" if matches else "not found",
                "curated_fields": [name for name, _, _ in matches],
                "curated_texts": [content for _, content, _ in matches],
                "curated_matched_phrase": [content[start:end] for _, content, (start, end) in matches],
                "n_matches": len(matches),
                "llm_ungrounded_quotes": ungrounded,
            }
        )
    return rows, call_info
