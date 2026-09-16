"""Building `data/derived/biosample_metadata.parquet`: per-accession submission/publication
dates, primary BioProject accession, and owning lab -- none of which live in
`trace_back_full.parquet`, so answering "how does this vary by year/bioproject/lab" needs its
own pass over the raw `inputs/*.jsonl` files (see `paths.py` for why this is a one-time,
stable-once-built artifact, unlike `trace_back_full.parquet`).

Run as a script (`python biosample_metadata.py`) to (re)build the parquet file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bs_entries import as_list, get_accession, iter_bs_entries, unwrap_biosample
from paths import BIOSAMPLE_METADATA_PARQUET, CRATE_INPUTS_DIR

# Field names (case/separator-insensitive) that carry a BioProject accession as a plain
# Attributes.Attribute -- how DDBJ-origin (SAMD) records encode it. NCBI-origin (SAMN) records
# instead link it under Links.Link (see `_primary_bioproject_from_links`).
_BIOPROJECT_ATTRIBUTE_NAMES = {"bioproject_id", "bioproject"}


def _primary_bioproject_from_attributes(bs: dict[str, Any]) -> str | None:
    attributes = as_list((bs.get("Attributes") or {}).get("Attribute"))
    for attribute in attributes:
        name = (attribute.get("attribute_name") or "").strip().lower().replace(" ", "_")
        if name in _BIOPROJECT_ATTRIBUTE_NAMES:
            content = attribute.get("content")
            if content:
                return str(content)
    return None


def _primary_bioproject_from_links(bs: dict[str, Any]) -> str | None:
    """The first (by record order) `Links.Link` entry targeting a BioProject -- a record can
    link to more than one (a multi-assay submission spanning two studies), but only the first
    is kept, per this project's own primary-link convention.
    """
    links = as_list((bs.get("Links") or {}).get("Link"))
    for link in links:
        if link.get("target") == "bioproject" and link.get("label"):
            return str(link["label"])
    return None


def get_primary_bioproject(bs: dict[str, Any]) -> str | None:
    """A record's BioProject accession (e.g. `PRJNA206510`/`PRJDB4662`), checking the
    DDBJ-style Attributes encoding first and the NCBI-style Links encoding second -- a given
    record only ever has one of the two shapes, so this order doesn't discard anything.
    """
    return _primary_bioproject_from_attributes(bs) or _primary_bioproject_from_links(bs)


def get_owner(bs: dict[str, Any]) -> str | None:
    """The submitting lab/institution's name, for grouping records by who submitted them --
    `Owner.Name.content` is present whenever `Owner` is, `abbreviation` as a fallback for the
    rare record that only has that. `Owner.Name` is a bare dict for one owner, a list for
    more than one (same "single item isn't wrapped" quirk as `Attributes.Attribute` -- see
    `bs_entries.as_list`) -- only the first is kept.
    """
    names = as_list((bs.get("Owner") or {}).get("Name"))
    if not names:
        return None
    return names[0].get("content") or names[0].get("abbreviation")


METADATA_COLUMNS = ["accession", "submission_date", "publication_date", "bioproject_id", "owner"]


def extract_metadata_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    """One `METADATA_COLUMNS` row for a raw BioSample record, or `None` if it has no accession
    (unusable as a join key back to `trace_back_full.parquet`).
    """
    accession = get_accession(raw)
    if accession is None:
        return None
    bs = unwrap_biosample(raw)
    return {
        "accession": accession,
        "submission_date": bs.get("submission_date"),
        "publication_date": bs.get("publication_date"),
        "bioproject_id": get_primary_bioproject(bs),
        "owner": get_owner(bs),
    }


def build_biosample_metadata(inputs_dir: Path = CRATE_INPUTS_DIR) -> pd.DataFrame:
    """Scans every `inputs/**/*.jsonl` file once and returns one row per accession. Rows
    accumulate as plain dicts and are turned into a DataFrame only once at the end -- same
    reasoning as `bs_entries.verify_extracted_against_raw_rows` (per-call DataFrame construction
    overhead dominates at millions-of-records scale).
    """
    rows = []
    for path in sorted(inputs_dir.glob("**/*.jsonl")):
        for raw in iter_bs_entries(path):
            row = extract_metadata_row(raw)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows, columns=METADATA_COLUMNS)


if __name__ == "__main__":
    metadata = build_biosample_metadata()
    BIOSAMPLE_METADATA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(BIOSAMPLE_METADATA_PARQUET, index=False)
    print(f"Wrote {len(metadata):,} rows to {BIOSAMPLE_METADATA_PARQUET}")
