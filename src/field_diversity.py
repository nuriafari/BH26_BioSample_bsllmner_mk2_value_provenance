"""Analyzing raw source-field naming diversity behind the curated `target_field`s.

`trace_back_full.parquet`'s own `raw_field`/`curated_fields` columns are confusingly named for
this purpose -- `raw_field` is actually the ~9-value LLM-curated TARGET field (`cell_line`,
`tissue`, ...), and `curated_fields` is the RAW metadata field name the evidence was found in
(thousands of values). `rename_for_analysis` renames both to the names used throughout this
notebook (`target_field`/`raw_field`) so "raw" means what it sounds like it means: as submitted,
not yet curated.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

RENAME_TO_ANALYSIS_NAMES = {"raw_field": "target_field", "curated_fields": "raw_field"}


def rename_for_analysis(trace_back: pd.DataFrame) -> pd.DataFrame:
    """Renames `trace_back_full.parquet`'s own columns to this notebook's clearer names -- see
    module docstring. Leaves every other column (`raw_value`, `strategy`, ...) untouched.
    """
    return trace_back.rename(columns=RENAME_TO_ANALYSIS_NAMES)


def explode_raw_fields(trace_back: pd.DataFrame) -> pd.DataFrame:
    """One row per (extracted value, evidence field) pair, dropping `strategy == "not found"`
    rows (their `raw_field` list is always empty -- nothing was found, so there's no source
    field to attribute the value to). A value matching more than one raw field (`n_matches > 1`)
    contributes one row per matching field, so it counts toward each -- e.g. a value filed under
    both `cell_line` and `source_name` counts as evidence for both when tallying raw-field usage.
    """
    found = trace_back[trace_back["strategy"] != "not found"]
    return found.explode("raw_field").rename(columns={"raw_field": "raw_field"})


_NORMALIZE_STRIP_RE = re.compile(r"[-/|_.\s]")


def normalize_field_name(name: str) -> str:
    """Lowercases `name` and drops separator characters (`-/|_.` and whitespace) entirely, so
    spelling variants of the same field name collapse together (`"cell type"`, `"cell_type"`,
    `"celltype"`, `"Cell-Type"` all normalize to `"celltype"`).
    """
    return _NORMALIZE_STRIP_RE.sub("", name.lower())


# Raw field names (matched after `normalize_field_name`) that are generic, submitter-authored
# free-text containers -- they don't name a specific biological property, so the same field name
# gets reused across wildly different studies for wildly different content. Everything NOT in
# this set is treated as "structured": named after a specific property (`cell_line`, `tissue`,
# `genotype`, ...) even when its own content is free-form text. Built from this crate's own
# top ~100 most common raw field names (see the notebook's own diversity tables) -- necessarily
# incomplete over the long tail of one-off field names, which the singleton-usage analysis
# quantifies separately.
_FREE_TEXT_FIELD_NAMES = {
    "title",
    "samplename",
    "sourcename",
    "comment",
    "submitterid",
    "studyname",
    "submittedsampleid",
    "biospecimenrepositorysampleid",
    "biospecimenrepository",
    "sample",
    "sampledescription",
    "submitterhandle",
    "description",
    "name",
    "sampleid",
    "group",
    "id",
    "identifier",
    "label",
    "accession",
}


def is_free_text_field(raw_field: str) -> bool:
    """Whether `raw_field` is a generic free-text container (`title`, `source_name`, ...) rather
    than a field named after a specific biological property -- see `_FREE_TEXT_FIELD_NAMES`.
    """
    return normalize_field_name(raw_field) in _FREE_TEXT_FIELD_NAMES


def rarefied_unique_counts(
    df: pd.DataFrame,
    group_col: str,
    unit_col: str,
    value_col: str,
    n_iter: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Rarefaction curve: for each group in `group_col` (e.g. a year), repeatedly subsamples
    down to `k` distinct `unit_col` values (`k` = the smallest group's own unit count) and counts
    `value_col.nunique()` among the rows for those sampled units, averaged over `n_iter` random
    subsamples -- the standard way (from ecological species-richness rarefaction) to compare a
    diversity count across groups of unequal size without the count simply tracking group size.

    Encodes `unit_col`/`value_col` to integer codes ONCE up front and does every subsample's
    membership test with `np.isin` over those integer arrays -- no per-group Python-level loop
    over units, which is what made a naive pandas-groupby version of this take minutes rather
    than seconds on this crate's real row counts (a single target_field/13-year call: >100s
    before, ~2s after).

    Returns one row per group: `group_col`, `n_units` (that group's real, un-subsampled unit
    count), `k` (the common subsample size used), `mean_unique`, `std_unique`.
    """
    rng = np.random.default_rng(seed)
    unit_codes_all, _ = pd.factorize(df[unit_col])
    value_codes_all, _ = pd.factorize(df[value_col])
    group_values = df[group_col].to_numpy()

    per_group: dict[object, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group in np.unique(group_values):
        mask = group_values == group
        unit_codes_g = unit_codes_all[mask]
        per_group[group] = (unit_codes_g, value_codes_all[mask], np.unique(unit_codes_g))
    k = min(len(units) for _, _, units in per_group.values())

    rows = []
    for group, (unit_codes_g, value_codes_g, units) in per_group.items():
        counts = []
        for _ in range(n_iter):
            sampled_units = rng.choice(units, size=k, replace=False)
            in_sample = np.isin(unit_codes_g, sampled_units)
            counts.append(len(np.unique(value_codes_g[in_sample])))
        rows.append({group_col: group, "n_units": len(units), "k": k, "mean_unique": float(np.mean(counts)), "std_unique": float(np.std(counts))})
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def concentration_curve(raw_field_counts: pd.Series) -> pd.DataFrame:
    """From a target_field's raw-field usage counts (raw_field -> count, any order), the
    cumulative-coverage curve: sorted by count descending, `n_fields` = how many top fields are
    included, `pct_covered` = what % of this target_field's total matched evidence those fields
    account for. Row 1 is always the single most-used raw field alone.
    """
    sorted_counts = raw_field_counts.sort_values(ascending=False)
    cumulative_pct = 100 * sorted_counts.cumsum() / sorted_counts.sum()
    return pd.DataFrame({"n_fields": range(1, len(sorted_counts) + 1), "pct_covered": cumulative_pct.to_numpy()})
