"""Loading curated results (the `results/select_*.json` files in the RO-Crate).

One `select_*.json` is a `SelectResult` as documented in bsllmner-mk2's
`docs/data-formats.md`: `{entries, run_metadata, evaluation, performance,
errors}`, where each `entries[]` item is one BioSample's extraction +
ontology-mapping result.

Full-size result files run into the hundreds of MB to ~1GB (see the RO-Crate's
own README), which is why bsllmner-viewer streams them with `ijson`
(`etl/load_select.py:iter_select_entries`) instead of `json.load`-ing the
whole file. That streaming approach is documented there rather than ported
here: this project's pilot files are a few MB, so a plain `json.load` is
simpler and sufficient -- swap in `ijson.items(f, "entries.item")` first if
you point these helpers at a full-size file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from notebook_utils import md


def load_select_result(path: Path) -> dict[str, Any]:
    """Load one `results/select_*.json` file in full."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_entry(select_result: dict[str, Any], accession: str) -> dict[str, Any] | None:
    """Find the entry for one accession within an already-loaded SelectResult."""
    for entry in select_result["entries"]:
        if entry["extract"]["accession"] == accession:
            return entry
    return None


def check_extracted_matches_results(extracted: dict[str, Any], results: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Checks that `results[field][].value` exactly reproduces `extracted[field]`
    for every field -- per bsllmner-mk2's own documented schema, `ResolvedValue.value`
    is literally "the extracted value" (`docs/data-formats.md`), so if this always
    held, `extracted` would be redundant with `results` and safe to drop from display.

    It does NOT always hold: Stage 2/3 (ontology search + LLM selection) can fail to
    map an extracted value to any term at all, which leaves `results[field]` empty
    even though `extracted[field]` has a real value -- a genuine mapping failure, not
    a bug in this check. Measured directly across several result files in this crate:
    roughly 1 in 9 entries has at least one field where this happens. So this check
    must be run per entry, every time -- never assumed to pass.

    Returns `(all_match, mismatches)`: `mismatches` is a list of
    `{"field", "extracted_value", "results_values"}` dicts, one per field that
    disagreed, empty when `all_match` is True.
    """
    mismatches = []
    for field, extracted_value in extracted.items():
        result_values = [match["value"] for match in results.get(field, [])]
        expected = [] if extracted_value is None else extracted_value if isinstance(extracted_value, list) else [extracted_value]
        if result_values != expected:
            mismatches.append({"field": field, "extracted_value": extracted_value, "results_values": result_values})
    return len(mismatches) == 0, mismatches


def curated_entry_table(entry: dict[str, Any]) -> pd.DataFrame:
    """A one-row, human-readable summary of a curated `entries[]` item, for
    display with `notebook_utils.display_with_nested_tables`. `extracted` and
    `results` are both always shown in full -- `check_extracted_matches_results`
    confirms they are NOT always the same (see its docstring), so `extracted`
    is never dropped. `results` lists every field, including ones with zero
    picks (shown as a blank row) -- nothing is filtered out.

    Also runs that check here and prints a warning when it fails, so a
    mismatch (and which field it's in) is visible without having to manually
    compare the two tables.
    """
    extracted = entry["extract"]["extracted"]
    results = entry["results"]
    all_match, mismatches = check_extracted_matches_results(extracted, results)
    if not all_match:
        reasons = "; ".join(f"{m['field']}: extracted={m['extracted_value']!r}, results values={m['results_values']!r}" for m in mismatches)
        md(f"**extracted != results for accession `{entry['extract']['accession']}`:** " + reasons)

    # Sorted alphabetically by field -- extracted's and results' own field order differ (they
    # come from different stages of the pipeline), which made the two tables hard to visually
    # compare side by side; sorting both the same way lines up matching rows.
    extracted_df = pd.DataFrame(
        [{"field": field, "value": value} for field, value in extracted.items()], columns=["field", "value"]
    ).sort_values("field").reset_index(drop=True).fillna("")

    results_rows = []
    for field, field_matches in results.items():
        if field_matches:
            results_rows.extend({"field": field, **match} for match in field_matches)
        else:
            results_rows.append({"field": field, "value": None, "term_id": None, "term_uri": None, "label": None, "exact_match": None, "reasoning": None})
    results_df = pd.DataFrame(
        results_rows, columns=["field", "value", "term_id", "term_uri", "label", "exact_match", "reasoning"]
    ).sort_values("field").reset_index(drop=True).fillna("")

    return pd.DataFrame([{"extracted": extracted_df, "results": results_df}])
