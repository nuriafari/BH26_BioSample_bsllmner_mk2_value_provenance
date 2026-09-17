"""LLM tier for the trace-back check, applied only to rows the deterministic cascade in
`bs_entries.py` already marked `"not found"` (exact/case-insensitive/normalized/ontology-synonym/
fuzzy all failed -- see `MATCH_STRATEGIES`).

Same output shape as the deterministic check (`raw_fields`/`raw_texts`/`raw_matched_phrase`,
rendered with the same `matches_display_table`), so the two tiers are a
drop-in continuation of each other, not a parallel system: the LLM is asked to find evidence
in the record's full text (which can bridge abbreviations, construct-name prefixes, or a value
split across two attributes -- gaps the deterministic tiers can't close), but every quote it
returns is independently re-checked with the SAME substring search the deterministic cascade
itself uses (`bs_entries._search`) before being trusted. This is the field-standard "LLM proposes,
deterministic check disposes" split (see the `grounding_verification_literature_design` memory,
modeled on `previous_work/scripts/common/evidence_grounding.py`'s Validator A/B pattern) -- an LLM
that hallucinates a quote is caught here regardless of how confident it sounds, because the check
never trusts the model's own claim that something is "verbatim."

Calls a local open-source model served by vLLM, not a hosted API -- no per-call cost beyond GPU
time this environment already has. Deliberately a DIFFERENT model than the one that produced the
extraction being checked (`mistral-small3.1:24b`, see `EXTRACTOR_MODEL` below and
`data/2026-06_mistral-small3.1-24b/README.md`), not the same weights re-run: re-querying the exact
model that already produced a value can't be expected to independently catch that same model's own
blind spots (e.g. a multi-hop inference it wasn't inclined to make during extraction is unlikely to
suddenly appear when the same weights are asked about it again) -- the `grounding_verification_
literature_design` memory's own recommendation is explicit on this ("different model than the
extractor"), which an earlier version of this module got wrong by defaulting to `EXTRACTOR_MODEL`.
Still a small, self-hostable open model (not a frontier hosted one) -- one step up in scale from
the extractor, not a categorically bigger/more expensive tier -- so the design stays scalable to
the full crate's residual, per the cost/time discussion in the notebook.

vLLM (not Ollama): measured directly, Ollama's llama.cpp backend serves concurrent requests near-
serially under real load (throughput stayed flat at ~0.2-0.3 calls/sec per GPU from 4 to 16
concurrent requests, and JSON-schema-constrained decoding made it worse) -- vLLM's PagedAttention
lets it batch dynamically instead, measured at ~23 calls/sec on one GPU with 200 real prompts in
one batch. That means the calling convention here is batch-shaped, not one-call-per-record: the
`vllm.LLM` engine is loaded once per process (expensive -- model load plus CUDA graph capture) and
every function in this module that talks to it takes a *list* of records and returns a *list* of
results from one `llm.chat()` call, rather than being called once per accession. The vLLM import
is deferred into the functions that need it (not at module level) because this module is also
imported by notebooks running under the project's main conda env, which doesn't have vLLM
installed -- only code paths that actually call the model need it importable.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from bs_entries import _record_search_groups, _search, as_list, unwrap_biosample

if TYPE_CHECKING:
    import vllm

# The crate's own extraction model (see module docstring) -- named here for documentation/
# comparison purposes only; this module never calls it, on purpose (see DEFAULT_MODEL).
EXTRACTOR_MODEL = "mistral-small3.1:24b"

# A distinct, moderately larger open-weight model -- independent of EXTRACTOR_MODEL's own biases,
# still small enough to self-host at crate scale.
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# Passed as `max_model_len` when constructing the `vllm.LLM` engine. Unlike Ollama, vLLM doesn't
# reserve this much KV-cache per concurrent slot -- PagedAttention shares one dynamic pool across
# every in-flight sequence, so this just caps how long any single prompt+output can be (prompts
# here run ~350-950 tokens); it's not a lever for concurrency the way it was for Ollama.
NUM_CTX = 3072
# Plenty for a handful of `{item_index, found, quotes}` verdicts per record -- observed outputs
# top out under 120 tokens even for records with several not-found items.
MAX_OUTPUT_TOKENS = 256

_SYSTEM_PROMPT = "You are a careful fact-checking assistant for biomedical metadata curation."


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
    """`items`: `(target_field, target_value)` pairs already marked `"not found"` for this one record."""
    text_block = "\n".join(_field_lines(raw))
    items_block = "\n".join(f"{i}. {field} = {value}" for i, (field, value) in enumerate(items, 1))
    return f"""Below is a BioSample record's searchable text, one field per line:
{text_block}

A deterministic exact/case-insensitive/normalized substring search already failed to find the values below anywhere in this text. For each one, decide whether the text still provides genuine evidence for it -- either stated directly, or a specific, well-supported inference (e.g. an abbreviation, a construct name like "shBRD9" implying a knockdown of BRD9, or a value whose two halves are split across two different fields).

Values to check:
{items_block}

For each value: set found=true only if you can point to exact, verbatim text above that establishes it. If found=true, quotes must be one or more EXACT substrings copied character-for-character from the field text above (not paraphrased, not corrected, not re-punctuated) -- each quote copied from a single field's text; use more than one quote when the evidence is split across different fields. IMPORTANT: a value split across two different fields still counts as found=true -- e.g. if one field says "CD4" and a separate field says "TREG", that IS sufficient evidence for the value "CD4 TREG", even though no single field contains that whole phrase; quote "CD4" and quote "TREG" as two separate quotes. Only set found=false when the text genuinely provides no basis at all for the value, not merely because the exact phrase doesn't appear in one place. Each quote must be ONLY the raw text itself, copied exactly as it appears after the colon above -- never include the field name or a colon in the quote (e.g. quote `CD4`, never `cell_type: CD4`). If you cannot find a genuine verbatim quote, set found=false and quotes=[] -- do NOT invent a quote just because you believe the value is probably true, and do NOT rely on outside world knowledge that goes beyond what this text itself supports.

Return one verdict per numbered item above, with `item_index` set to that item's number -- a complete, separate verdict object for EVERY item listed above, written out in full. Never omit an item's verdict, and never write "..." (or similar) in place of a verdict to mean "the rest follow the same pattern" -- every item gets its own real object, no matter how many there are.

Respond with ONLY a JSON object, no other text. This example shows the shape for two items -- adjust the number of verdict objects to match the number of items above, writing every one out:
{{"verdicts": [{{"item_index": 1, "found": true, "quotes": ["exact quoted text"]}}, {{"item_index": 2, "found": false, "quotes": []}}]}}"""


def load_engine(model: str = DEFAULT_MODEL, gpu_memory_utilization: float = 0.85) -> vllm.LLM:
    """Constructs the `vllm.LLM` engine for one GPU (set `CUDA_VISIBLE_DEVICES` before calling this,
    one process per GPU -- vLLM doesn't need a second GPU to be handed data-parallel work the way a
    single Ollama instance did). Expensive (model load plus CUDA graph capture, a few minutes) --
    call this once per process and reuse the returned engine for every batch, never per-record.
    """
    from vllm import LLM

    return LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization, max_model_len=NUM_CTX)


def default_sampling_params() -> vllm.SamplingParams:
    """Greedy decoding (this is a fact-checking verdict, not creative generation -- there's no
    reason to sample) up to `MAX_OUTPUT_TOKENS`.
    """
    from vllm import SamplingParams

    return SamplingParams(temperature=0, max_tokens=MAX_OUTPUT_TOKENS)


_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _parse_verdicts_json(content: str) -> dict[str, Any] | None:
    """Unconstrained generation (see module docstring -- no `format`/grammar constraint, just a
    prompt instruction) sometimes wraps the JSON in a markdown fence, trails extra text after it, or
    leaves a trailing comma before a closing bracket (valid in Python/JS, not JSON) -- strip a fence
    if present and any trailing commas, then `raw_decode` (not `loads`) so trailing text after the
    JSON object doesn't break parsing -- only the leading object matters.
    """
    content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    content = _TRAILING_COMMA.sub(r"\1", content)
    try:
        parsed, _ = json.JSONDecoder().raw_decode(content)
        return parsed
    except json.JSONDecodeError:
        return None


def call_llm_batch(
    prompts: list[str], llm: vllm.LLM, sampling_params: vllm.SamplingParams
) -> list[tuple[dict[str, Any] | None, dict[str, Any]]]:
    """Runs every prompt in `prompts` through one `llm.chat()` call -- vLLM batches them internally
    (continuous batching + PagedAttention), so this is the unit that actually gets the throughput
    win described in the module docstring; calling this once per prompt would defeat the point.
    Returns one `(structured_output, call_info)` pair per prompt, in the same order --
    `structured_output` is `None` if the model's response wasn't valid JSON, so callers have one
    place to check for "the call itself didn't work" separately from "the model said not found."
    """
    messages_batch = [[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": p}] for p in prompts]

    # A handful of BioSample records have unusually many not-found items (one has 52) and produce
    # a prompt well past NUM_CTX -- vLLM raises for the WHOLE batch call if any single prompt's
    # token count leaves no room for the requested output under max_model_len, not just for that
    # one prompt (measured directly: took down a whole production run). Pre-check token counts and
    # route oversized ones to an error result instead of sending them to `llm.chat()` at all.
    tokenizer = llm.get_tokenizer()
    prompt_lengths = [len(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True)) for m in messages_batch]
    oversized = {i for i, length in enumerate(prompt_lengths) if length + sampling_params.max_tokens > NUM_CTX}

    fitting_messages = [m for i, m in enumerate(messages_batch) if i not in oversized]
    # Qwen3's chat template defaults to emitting a <think>...</think> block before the answer;
    # ignored harmlessly by chat templates (Qwen2.5's included) that don't reference this kwarg at
    # all. Disabled here because `_parse_verdicts_json` expects the JSON object to start the
    # response -- a thinking trace first would break `raw_decode`, not because thinking is bad.
    outputs = iter(
        llm.chat(fitting_messages, sampling_params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        if fitting_messages
        else []
    )

    results = []
    for i in range(len(prompts)):
        if i in oversized:
            results.append((None, {"error": f"prompt too long ({prompt_lengths[i]} tokens, max {NUM_CTX})"}))
            continue
        output = next(outputs)
        text = output.outputs[0].text
        call_info = {"output_tokens": len(output.outputs[0].token_ids), "prompt_tokens": len(output.prompt_token_ids)}
        structured = _parse_verdicts_json(text)
        if structured is None:
            call_info["error"] = f"model response wasn't valid JSON: {text!r}"
        results.append((structured, call_info))
    return results


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


def _rows_from_verdicts(
    structured: dict[str, Any] | None, not_found_items: list[tuple[str, str]], raw: dict[str, Any]
) -> list[dict[str, Any]]:
    """Shared by the single-record and batch entry points below: turns one call's parsed `structured`
    output (or `None`, on a call failure) into one row per `not_found_items` entry. Row shape matches
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

    If the call itself failed (`structured` is `None`), every item is returned with
    `strategy="not found"` and `llm_verdict=None` (a call failure, not a "not found" verdict --
    kept distinguishable via `llm_verdict is None` rather than silently treated as agreement).
    """
    by_index: dict[int, dict[str, Any]] = {}
    if structured is not None:
        for v in structured.get("verdicts", []):
            # Unconstrained generation (see module docstring) means a syntactically valid JSON
            # object can still be missing a required key -- treat that item as if the model hadn't
            # returned a verdict for it at all (falls into the `verdict is None` branch below),
            # rather than crashing the whole batch on one malformed item.
            if isinstance(v, dict) and isinstance(v.get("item_index"), int) and isinstance(v.get("found"), bool):
                by_index[v["item_index"]] = v

    rows = []
    for i, (field, value) in enumerate(not_found_items, 1):
        verdict = by_index.get(i)
        if verdict is None:
            rows.append(
                {
                    "target_field": field, "target_value": value, "strategy": "not found", "llm_verdict": None,
                    "raw_fields": [], "raw_texts": [], "raw_matched_phrase": [], "n_matches": 0,
                    "llm_ungrounded_quotes": [],
                }
            )
            continue

        if not verdict["found"]:
            rows.append(
                {
                    "target_field": field, "target_value": value, "strategy": "not found", "llm_verdict": False,
                    "raw_fields": [], "raw_texts": [], "raw_matched_phrase": [], "n_matches": 0,
                    "llm_ungrounded_quotes": [],
                }
            )
            continue

        quotes = [q for q in verdict.get("quotes") or [] if isinstance(q, str)]
        matches, ungrounded = _ground_quotes(raw, quotes)
        rows.append(
            {
                "target_field": field, "target_value": value, "llm_verdict": True,
                "strategy": "llm" if matches else "not found",
                "raw_fields": [name for name, _, _ in matches],
                "raw_texts": [content for _, content, _ in matches],
                "raw_matched_phrase": [content[start:end] for _, content, (start, end) in matches],
                "n_matches": len(matches),
                "llm_ungrounded_quotes": ungrounded,
            }
        )
    return rows


def verify_not_found_with_llm_batch(
    records: list[tuple[dict[str, Any], list[tuple[str, str]]]], llm: vllm.LLM, sampling_params: vllm.SamplingParams
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """The batch entry point real callers should use (see module docstring): one `(raw,
    not_found_items)` pair per BioSample record, all sent to vLLM in a single `llm.chat()` call via
    `call_llm_batch` (one call per record, batched per record not per field -- same reasoning as
    `previous_work`'s Validator B: one call amortizes the fixed per-request overhead over every
    field that record still needs checked). Returns one `(rows, call_info)` pair per input record,
    in the same order -- see `_rows_from_verdicts` for what `rows` contains.
    """
    prompts = [build_prompt(raw, items) for raw, items in records]
    call_results = call_llm_batch(prompts, llm, sampling_params)
    return [
        (_rows_from_verdicts(structured, items, raw), call_info)
        for (raw, items), (structured, call_info) in zip(records, call_results)
    ]


def verify_not_found_with_llm(
    raw: dict[str, Any], not_found_items: list[tuple[str, str]], llm: vllm.LLM, sampling_params: vllm.SamplingParams
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Single-record convenience wrapper over `verify_not_found_with_llm_batch`, for notebook use
    (checking one record at a time) -- real batch jobs should call the batch function directly
    rather than loop this over records one by one, since a length-1 batch wastes vLLM's whole
    reason for existing.
    """
    [(rows, call_info)] = verify_not_found_with_llm_batch([(raw, not_found_items)], llm, sampling_params)
    return rows, call_info
