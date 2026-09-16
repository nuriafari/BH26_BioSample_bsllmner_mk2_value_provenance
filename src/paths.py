"""Filesystem path constants for this project.

Every path is derived from this file's own location (not hardcoded), so the
project works no matter where the repo is checked out -- see
`previous_work/scripts/common/io_utils.py` for the convention this follows.
Notebooks should import from here instead of hand-typing "../data/..." strings.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

# The one RO-Crate release currently under data/. If/when a second release is
# added, give it its own CRATE_DIR constant rather than repointing this name.
CRATE_DIR = DATA_DIR / "2026-06_mistral-small3.1-24b"

CRATE_METADATA_JSON = CRATE_DIR / "ro-crate-metadata.json"
CRATE_README = CRATE_DIR / "README.md"

CRATE_INPUTS_DIR = CRATE_DIR / "inputs"
CRATE_RESULTS_DIR = CRATE_DIR / "results"
CRATE_CONFIG_DIR = CRATE_DIR / "config"
CRATE_ONTOLOGY_DIR = CRATE_DIR / "ontology"
CRATE_LOGS_DIR = CRATE_DIR / "logs"
CRATE_PROVENANCE_DIR = CRATE_DIR / "provenance"

RUN_INDEX_TSV = CRATE_PROVENANCE_DIR / "run_index.tsv"
RUN_SEGMENTS_TSV = CRATE_PROVENANCE_DIR / "run_segments.tsv"
CHECKSUMS_SHA256 = CRATE_PROVENANCE_DIR / "checksums.sha256"

# Artifacts this project computes from the crate (not part of the crate itself) -- e.g. the
# full-scale trace-back check, which takes too long to recompute on every notebook run.
DERIVED_DIR = DATA_DIR / "derived"
TRACE_BACK_FULL_PARQUET = DERIVED_DIR / "trace_back_full.parquet"
MANUAL_REVIEW_SAMPLE_PARQUET = DERIVED_DIR / "manual_review_sample_100.parquet"

# Per-BioSample metadata (submission/publication dates, primary BioProject, owner lab) that
# isn't in trace_back_full.parquet -- built once by `biosample_metadata.py` from a full scan of
# `inputs/`, and stable across trace-back re-runs (unlike trace_back_full.parquet, this doesn't
# change as the trace-back job progresses, so it never needs rebuilding once it exists).
BIOSAMPLE_METADATA_PARQUET = DERIVED_DIR / "biosample_metadata.parquet"
