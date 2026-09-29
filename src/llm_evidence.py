"""LLM tier for the trace-back check, applied only to rows the deterministic cascade in
`bs_entries.py` already marked `"not found"` (exact/case-insensitive/normalized/ontology-synonym/
fuzzy all failed -- see `MATCH_STRATEGIES`).

Same output shape as the deterministic check (`raw_fields`/`raw_texts`/`raw_matched_phrase`,
rendered with the same `matches_display_table`), so the two tiers are a
drop-in continuation of each other, not a parallel system: the LLM is asked to find evidence
in the record's text (which can bridge abbreviations, construct-name prefixes, or a value
split across two attributes -- gaps the deterministic tiers can't close), but every quote it
returns is independently re-checked with the SAME substring search the deterministic cascade
itself uses (`bs_entries._search`) before being trusted -- and ALL of an item's returned quotes
must ground, not just one, or the whole item is rejected (see `_rows_from_verdicts`). This is the
field-standard "LLM proposes, deterministic check disposes" split (see the `grounding_verification_
literature_design` memory, modeled on `previous_work/scripts/common/evidence_grounding.py`'s
Validator A/B pattern) -- an LLM that hallucinates a quote is caught here regardless of how
confident it sounds, because the check never trusts the model's own claim that something is
"verbatim."

Two-pass per model size, batched, never a per-record loop: Pass A shows only submitted
attributes/title/comment (`_attribute_lines`) for EVERY currently-unresolved item in one batched
`llm.chat()` call; whatever Pass A leaves unresolved is then re-sent, in a second batched call,
with the complete record (`_full_record_lines`) -- the same set of leaf fields the deterministic
cascade's own secondary search covers. This mirrors the escalation already used between model
sizes (Qwen3-8B's own residual feeds Qwen3-32B-AWQ) at a finer grain, within one model, and keeps
the call count at two per model tier regardless of how many records are being checked -- see
`run_llm_evidence_batch.run_two_pass`.

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
imported by notebooks that never call the model at all (they only read this tier's already-computed
JSONL output) -- importing vLLM costs real time (CUDA init, several seconds) and requires a GPU to
be visible, neither of which those notebooks need just to read a results file.
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


def _attribute_lines(raw: dict[str, Any]) -> list[str]:
    """One `"- {field}: {content}"` line per attribute/title/comment, for the prompt only -- a
    plain, deduplicated rendering (unlike `_record_search_groups`'s `attribute_entries`, which
    deliberately ALSO adds each attribute's own name as a second searchable entry for the
    deterministic cascade; that trick would just show as a confusing duplicate line here). This is
    the narrower, Pass-A rendering -- see `_full_record_lines` for the wider one.
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


def _full_record_lines(raw: dict[str, Any]) -> list[str]:
    """`_attribute_lines` plus every OTHER leaf field in the record -- `Ids.Id`, `Owner`,
    `Links.Link`, `Package`, `Status`, dates, and anything else the record happens to carry (the
    same set `bs_entries._record_search_groups`'s `secondary_entries` covers for the deterministic
    cascade's own secondary-field search). Used for Pass B: a value whose only textual basis is
    outside the submitted attributes/title/comment is invisible to `_attribute_lines` but visible
    here, so the LLM gets a real second chance at it instead of never seeing that text at all.
    """
    attribute_lines = _attribute_lines(raw)
    _, secondary_entries, _, _ = _record_search_groups(raw)
    # title/comment are already in attribute_lines -- skip them here to avoid a duplicate line
    secondary_lines = [f"- {name}: {content}" for name, content, _ in secondary_entries if name not in ("title", "comment")]
    return attribute_lines + secondary_lines


def _item_line(index: int, field: str, value: str, hint: str | None) -> str:
    """One numbered `"N. field = value"` line for the prompt's item list, with an optional bracketed
    note appended -- used to tell the model that a deterministic search already found this exact
    phrase but in a negated context (see `run_llm_evidence_batch.sample_not_found_by_accession`),
    so it can weigh that negation specifically rather than treating the item as a blank unknown.
    """
    line = f"{index}. {field} = {value}"
    return f"{line} [{hint}]" if hint else line


def build_prompt(
    raw: dict[str, Any], items: list[tuple[str, str, str | None]], full_record: bool = False, request_explanation: bool = False
) -> str:
    """`items`: `(target_field, target_value, hint)` triples still unresolved for this one record --
    `hint` is `None` for an ordinary unresolved item, or a note about prior negated evidence (see
    `_item_line`). `full_record=False` (Pass A) renders only submitted attributes/title/comment;
    `full_record=True` (Pass B) renders the complete record -- callers run Pass A first over every
    unresolved item, then Pass B only over whatever Pass A still couldn't ground (see
    `run_llm_evidence_batch.py`), never both at once for the same item.

    `request_explanation=True` additionally asks for a short `explanation` string on every
    `found=false` verdict -- why the model believes the record doesn't support the value (a
    genuinely different concept discussed nearby, a connection that would need outside domain
    knowledge, an apparent extraction error, etc.). Meant for a final pass over whatever is still
    unresolved after every other deterministic and LLM stage, specifically to produce material for
    a downstream error-clustering step, not for routine escalation passes (which don't need the
    extra output tokens or the risk of a model padding out a low-value field).
    """
    text_block = "\n".join(_full_record_lines(raw) if full_record else _attribute_lines(raw))
    items_block = "\n".join(_item_line(i, field, value, hint) for i, (field, value, hint) in enumerate(items, 1))
    explanation_instructions = (
        """ Additionally, when found=false, include a brief "explanation" field (one or two sentences) stating why you believe """
        """the record does not support this value -- e.g. the text discusses a related but different concept, establishing the """
        """value would require outside domain knowledge not present in the text, the value appears to be an extraction error """
        """with no textual basis at all, or any other specific reason grounded in what the text actually says. When found=true, """
        """set "explanation" to an empty string."""
        if request_explanation
        else ""
    )
    explanation_example = ', "explanation": "no mention of this concept or anything related to it appears anywhere in the record"' if request_explanation else ""
    explanation_example_true = ', "explanation": ""' if request_explanation else ""
    return f"""Below is a BioSample record's searchable text, one field per line:
{text_block}

A deterministic exact/case-insensitive/normalized substring search already failed to find the values below anywhere in this text (an item marked with a bracketed note found the phrase but only in what looks like a negated context -- weigh that note specifically). For each one, decide whether the text still provides genuine evidence for it -- either stated directly, or a specific, well-supported inference (e.g. an abbreviation, a construct name like "shBRD9" implying a knockdown of BRD9, or a value whose two halves are split across two different fields).

Values to check:
{items_block}

For each value: set found=true only if you can point to exact, verbatim text above that establishes it. If found=true, quotes must be one or more EXACT substrings copied character-for-character from the field text above (not paraphrased, not corrected, not re-punctuated) -- each quote copied from a single field's text; use more than one quote when the evidence is split across different fields. IMPORTANT: a value split across two different fields still counts as found=true -- e.g. if one field says "CD4" and a separate field says "TREG", that IS sufficient evidence for the value "CD4 TREG", even though no single field contains that whole phrase; quote "CD4" and quote "TREG" as two separate quotes. Only set found=false when the text genuinely provides no basis at all for the value, not merely because the exact phrase doesn't appear in one place. Each quote must be ONLY the raw text itself, copied exactly as it appears after the colon above -- never include the field name or a colon in the quote (e.g. quote `CD4`, never `cell_type: CD4`). If you cannot find a genuine verbatim quote, set found=false and quotes=[] -- do NOT invent a quote just because you believe the value is probably true, and do NOT rely on outside world knowledge that goes beyond what this text itself supports. Every quote you return will be independently checked against the text above -- ALL of your quotes for an item must be genuine, verbatim substrings, or the whole item is rejected, so never pad a real quote with a fabricated one.{explanation_instructions}

Return one verdict per numbered item above, with `item_index` set to that item's number -- a complete, separate verdict object for EVERY item listed above, written out in full. Never omit an item's verdict, and never write "..." (or similar) in place of a verdict to mean "the rest follow the same pattern" -- every item gets its own real object, no matter how many there are.

Respond with ONLY a JSON object, no other text. This example shows the shape for two items -- adjust the number of verdict objects to match the number of items above, writing every one out:
{{"verdicts": [{{"item_index": 1, "found": true, "quotes": ["exact quoted text"]{explanation_example_true}}}, {{"item_index": 2, "found": false, "quotes": []{explanation_example}}}]}}"""


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


def _ground_quotes(raw: dict[str, Any], quotes: list[str], full_record: bool) -> tuple[list[tuple[str, str, tuple[int, int]]], list[str]]:
    """Independently re-checks each of the LLM's `quotes` against the record's actual text, via
    the SAME `_search` cascade the deterministic checker uses (exact / case-insensitive /
    normalized -- never `fuzzy` or `ontology synonym`: stacking a second fallible/curated-lookup
    tier on top of an already-fallible LLM claim would compound risk rather than bound it, per
    the `grounding_verification_literature_design` memory).

    Scoped to exactly what the model was actually shown for THIS call: `full_record=False` (a
    Pass-A verdict, see `build_prompt`) restricts the check to attributes/title/comment, the same
    text `_attribute_lines` rendered; `full_record=True` (a Pass-B verdict) checks the complete
    record. A quote can never ground by coincidentally occurring somewhere the model never saw --
    provenance verification establishes that the model's claimed evidence actually came from its
    own input, not merely that it exists somewhere in the record.

    Returns `(matches, ungrounded)`: `matches` in the same `(field, content, span)` shape
    `verify_extracted_against_raw_rows` produces (one entry per quote that grounds, across however
    many different fields); `ungrounded` is the quotes that did NOT verify -- i.e. the LLM cited
    text that doesn't actually appear anywhere in what it was shown, a caught hallucination, kept
    for transparency rather than silently dropped.
    """
    attribute_entries, secondary_entries, _, _ = _record_search_groups(raw)
    if not full_record:
        secondary_entries = [entry for entry in secondary_entries if entry[0] in ("title", "comment")]
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
    structured: dict[str, Any] | None, not_found_items: list[tuple[str, str, str | None]], raw: dict[str, Any], full_record: bool
) -> list[dict[str, Any]]:
    """Shared by the single-record and batch entry points below: turns one call's parsed `structured`
    output (or `None`, on a call failure) into one row per `not_found_items` entry. Row shape matches
    `bs_entries.verify_extracted_against_raw_rows` plus four columns:

    - `strategy`: `"llm"` only when the model said `found=true` AND every quote it returned
      grounds (see `_ground_quotes`) -- at least one quote, none ungrounded; a single unverifiable
      quote rejects the WHOLE item, not just that quote, since a model willing to pad one genuine
      quote with a fabricated one can't be trusted on the genuine one either. Otherwise
      `"not found"` (unchanged from the deterministic result) -- covers "the LLM itself said not
      found," "the LLM said found but returned zero quotes," and "at least one cited quote failed
      grounding."
    - `llm_verdict`: the LLM's own raw found/not-found call, before grounding -- lets you see
      the caught-hallucination case (`llm_verdict=True`, `strategy="not found"`) separately from
      simple agreement.
    - `llm_ungrounded_quotes`: quotes the LLM cited that did NOT verify against what it was shown
      -- empty in the normal case; non-empty is the hallucination signal this tier exists to catch,
      preserved here even when it causes the whole item to be rejected, so a rejected call can
      still be inspected rather than silently discarded.
    - `llm_pass`: `"full_record"` if this verdict came from the Pass-B (complete record) prompt,
      `"attributes"` if from Pass A -- mirrors `full_record`, kept on the row since a merged
      per-record result (see `run_llm_evidence_batch.py`) mixes items from both passes.

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

    llm_pass = "full_record" if full_record else "attributes"
    rows = []
    for i, (field, value, hint) in enumerate(not_found_items, 1):
        verdict = by_index.get(i)
        if verdict is None:
            rows.append(
                {
                    "target_field": field, "target_value": value, "strategy": "not found", "llm_verdict": None,
                    "raw_fields": [], "raw_texts": [], "raw_matched_phrase": [], "n_matches": 0,
                    "llm_ungrounded_quotes": [], "llm_pass": llm_pass, "hint": hint, "explanation": "",
                }
            )
            continue

        if not verdict["found"]:
            explanation = verdict.get("explanation") if isinstance(verdict.get("explanation"), str) else ""
            rows.append(
                {
                    "target_field": field, "target_value": value, "strategy": "not found", "llm_verdict": False,
                    "raw_fields": [], "raw_texts": [], "raw_matched_phrase": [], "n_matches": 0,
                    "llm_ungrounded_quotes": [], "llm_pass": llm_pass, "hint": hint, "explanation": explanation,
                }
            )
            continue

        quotes = [q for q in verdict.get("quotes") or [] if isinstance(q, str)]
        matches, ungrounded = _ground_quotes(raw, quotes, full_record)
        # ALL returned quotes must ground, not just one -- and a bare found=true with zero quotes
        # is never (vacuously) accepted, since `not ungrounded` alone would be True for an empty list
        accepted = bool(quotes) and not ungrounded
        # a found=true call that got rejected on grounding still gets a chance to explain itself --
        # the model doesn't know its quote failed verification, so this is whatever it put in
        # "explanation" (often empty, since it believed found=true), not a post-hoc justification
        explanation = verdict.get("explanation") if isinstance(verdict.get("explanation"), str) else ""
        rows.append(
            {
                "target_field": field, "target_value": value, "llm_verdict": True,
                "strategy": "llm" if accepted else "not found",
                "raw_fields": [name for name, _, _ in matches],
                "raw_texts": [content for _, content, _ in matches],
                "raw_matched_phrase": [content[start:end] for _, content, (start, end) in matches],
                "n_matches": len(matches),
                "llm_ungrounded_quotes": ungrounded,
                "llm_pass": llm_pass,
                "explanation": explanation if not accepted else "",
                # kept for a possible next-model pass (see `not_found_items_from_prior_pass`) --
                # a `strategy="llm"` row carries `hint` too, harmlessly unused, so the shape stays
                # uniform across every row this function returns
                "hint": hint,
            }
        )
    return rows


def verify_not_found_with_llm_batch(
    records: list[tuple[dict[str, Any], list[tuple[str, str, str | None]]]],
    llm: vllm.LLM,
    sampling_params: vllm.SamplingParams,
    full_record: bool = False,
    request_explanation: bool = False,
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """The batch entry point real callers should use (see module docstring): one `(raw,
    not_found_items)` pair per BioSample record -- `not_found_items` a list of `(target_field,
    target_value, hint)` triples (see `build_prompt`) -- all sent to vLLM in a single `llm.chat()`
    call via `call_llm_batch` (one call per record, batched per record not per field -- same
    reasoning as `previous_work`'s Validator B: one call amortizes the fixed per-request overhead
    over every field that record still needs checked). `full_record` selects Pass A (`False`,
    attributes/title/comment only) or Pass B (`True`, the complete record) for EVERY record in this
    call -- callers run this once with `full_record=False` over every unresolved item, then again
    with `full_record=True` over only the items that call left unresolved (see
    `run_llm_evidence_batch.run_two_pass`); this function itself does not escalate between passes.
    `request_explanation` asks for a short "why not" explanation on every `found=false` verdict
    (see `build_prompt`) -- for a final pass over the true residual, not routine escalation.
    Returns one `(rows, call_info)` pair per input record, in the same order -- see
    `_rows_from_verdicts` for what `rows` contains.
    """
    prompts = [build_prompt(raw, items, full_record=full_record, request_explanation=request_explanation) for raw, items in records]
    call_results = call_llm_batch(prompts, llm, sampling_params)
    return [
        (_rows_from_verdicts(structured, items, raw, full_record), call_info)
        for (raw, items), (structured, call_info) in zip(records, call_results)
    ]


def verify_not_found_with_llm(
    raw: dict[str, Any], not_found_items: list[tuple[str, str, str | None]], llm: vllm.LLM, sampling_params: vllm.SamplingParams
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Single-record convenience wrapper over `verify_not_found_with_llm_batch`, for notebook use
    (checking one record at a time) -- real batch jobs should call the batch function directly
    rather than loop this over records one by one, since a length-1 batch wastes vLLM's whole
    reason for existing. Pass A only (`full_record=False`); notebook callers checking a single
    record interactively can call `verify_not_found_with_llm_batch` directly with
    `full_record=True` for the Pass-B behavior.
    """
    [(rows, call_info)] = verify_not_found_with_llm_batch([(raw, not_found_items)], llm, sampling_params)
    return rows, call_info
