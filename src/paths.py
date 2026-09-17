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
# A second, independently-drawn 100-sample against the CURRENT (post-five-tiers) "not found"
# residual, reviewed by Claude and cross-checked against the live Qwen3-8B evidence tier -- kept
# alongside MANUAL_REVIEW_SAMPLE_PARQUET (not overwriting it) so the original pre-fix snapshot
# stays available as a before/after reference.
MANUAL_REVIEW_SAMPLE_QWEN3_8B_PARQUET = DERIVED_DIR / "manual_review_sample_100_qwen3_8b.parquet"
QWEN3_8B_CLAUDE_REVIEW_JSON = DERIVED_DIR / "qwen3_8b_claude_review_100.json"

# Small, git-committed reference data the notebooks depend on: frozen historical samples,
# hand/LLM-review tables, and before/after baselines. Unlike DATA_DIR (gitignored, holds the large
# downloaded RO-Crate and its regenerable derived artifacts), these are tiny, one-of-a-kind, and
# can't be recomputed from the crate alone -- so they're checked into the repo instead.
NOTEBOOK_FIXTURES_DIR = PROJECT_ROOT / "notebooks" / "fixtures"
MANUAL_REVIEW_SAMPLE_PARQUET = NOTEBOOK_FIXTURES_DIR / "manual_review_sample_100.parquet"
MANUAL_REVIEW_100_JSON = NOTEBOOK_FIXTURES_DIR / "manual_review_100.json"
LLM_DEMO_ITEMS_JSON = NOTEBOOK_FIXTURES_DIR / "llm_demo_items.json"
TRACE_BACK_STRATEGY_COUNTS_BEFORE_FIVE_TIERS_JSON = (
    NOTEBOOK_FIXTURES_DIR / "trace_back_strategy_counts_before_five_tiers.json"
)
TRACE_BACK_STRATEGY_COUNTS_BEFORE_WHOLE_RECORD_SEARCH_JSON = (
    NOTEBOOK_FIXTURES_DIR / "trace_back_strategy_counts_before_whole_record_search.json"
)

# Per-BioSample metadata (submission/publication dates, primary BioProject, owner lab) that
# isn't in trace_back_full.parquet -- built once by `biosample_metadata.py` from a full scan of
# `inputs/`, and stable across trace-back re-runs (unlike trace_back_full.parquet, this doesn't
# change as the trace-back job progresses, so it never needs rebuilding once it exists).
BIOSAMPLE_METADATA_PARQUET = DERIVED_DIR / "biosample_metadata.parquet"
