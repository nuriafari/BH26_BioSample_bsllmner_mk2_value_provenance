# BH26 BioSample Curation

**Status: work in progress.** Parts of this pipeline (in particular the GPU/vLLM environment described below) are host-specific and may not run out of the box everywhere. Built with the assistance of [Claude Code](https://claude.com/claude-code).

This project was developed as part of [DBCLS BioHackathon 2026](https://2026.biohackathon.org/), held in Matsuyama, Ehime, Japan, from 13 to 19 September 2026

## What this is

This repository traces the values curated by [bsllmner-mk2](https://github.com/dbcls/bsllmner-mk2) for roughly 4.2 million BioSample records back to their source text through a sequence of deterministic matching, ontology-based rescoring, and LLM evidence steps, then evaluates the results across the full dataset (_LLM evidence and evaluation still under development_).

This repository builds upon the results of [bsllmner-mk2](https://github.com/dbcls/bsllmner-mk2). BioSample records (the metadata submitted alongside a sequencing experiment) are free text: the same concept ("cell line", "tissue") shows up under inconsistent field names and phrasing across labs and submitters. The upstream project runs an LLM (named-entity extraction) plus an ontology-search/mapping pipeline over ~4.2 million BioSample records from ChIP-Atlas and DDBJ/NCBI/EBI RNA-Seq, producing a curated value (and, where possible, an ontology term) for each of ~9 target fields. It packages its output as an [RO-Crate](https://www.researchobject.org/ro-crate/) — a structured, provenance-tracked data bundle — and that RO-Crate is the *only* external input this project depends on.

The trace-back check uses a deterministic match cascade, an ontology-synonym/rescoring tier, and an optional LLM-based tier for the residual the deterministic tiers cannot resolve, and analyzes what it finds. All code needed to run that analysis lives in this repository — the RO-Crate is downloaded as a versioned external dependency (see [Data](#data--the-ro-crate)).

See [Results](#results) for where the actual findings live.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | **Record loading:** raw and curated BioSample records (`bs_entries.py`, `select_results.py`).<br>**Trace-back matching:** deterministic matching, ontology rescoring, and optional LLM evidence (`bs_entries.py`, `ontology_rescore.py`, `llm_evidence.py`).<br>**Analysis support:** full-crate trace-back tables, per-accession metadata, notebook helpers, and path constants (`build_trace_back_full.py`, `biosample_metadata.py`, `notebook_utils.py`, `paths.py`). |
| `notebooks/` | **RO-Crate overview:** `biosample_rocrate_overview.ipynb`, covering categories, raw and curated inputs, source overlap, and paired examples.<br>**Trace-back analysis:** `biosample_trace_back_analysis.ipynb`, covering how the matching cascade and LLM tiers work, what each tier means, and raw-field diversity and trends by year, source, and BioProject.<br>**Trace-back evaluation:** `biosample_trace_back_evaluation.ipynb`, covering matching-method examples, LLM evidence-tier precision/recall, and pipeline issues. |
| `data/` | **Raw data:** the downloaded RO-Crate.<br>**Derived data:** this project's parquet tables under `data/derived/`.<br>Not tracked in Git. |
| `environment.yaml` | **Main environment:** everything except the optional LLM evidence tier. |

## Data — the RO-Crate

The dataset is not stored in this repo — `data/` is gitignored. It's a ~38GB RO-Crate published by the bsllmner-mk2 project.

```sh
curl -O https://biosampleplus.s3.ap-northeast-1.amazonaws.com/releases/2026-06_mistral-small3.1-24b.tar.gz
mkdir -p data
tar xzf 2026-06_mistral-small3.1-24b.tar.gz -C data
```

or simply `make data` (see [Reproducing](#reproducing)). Budget roughly 80GB of free disk: the tarball, the extracted crate, and room for the derived tables this project builds on top of it.

Crate layout (see `data/2026-06_mistral-small3.1-24b/README.md`, published inside the crate itself, for the authoritative description):

| Directory | Contents |
|---|---|
| `inputs/` | Raw BioSample records (JSONL), one file per run |
| `results/` | One curated, ontology-mapped result file per run |
| `ontology/` | The OWL ontology files the pipeline mapped terms against |
| `provenance/` | Run index, execution segments, checksums |
| `config/`, `logs/` | Select configs/prompts used, and per-worker execution logs |

The local `data/derived/` directory is a working cache, not a checked-in result bundle. It is ignored along with the rest of `data/`, and may contain checkpoints, backups, and exploratory LLM outputs in addition to the canonical tables. A clean checkout therefore cannot use those files without rebuilding them.

## Environments

This project genuinely needs two separate setups, not one:

1. **Main environment** (`environment.yaml`, Python 3.12) — everything needed to run both notebooks and the deterministic/ontology trace-back tiers: pandas, spaCy/negspacy for negation-aware text search, scikit-learn/rapidfuzz for fuzzy matching, matplotlib/streamlit/nbconvert for output. CPU-only is fine here.
2. **A separate GPU environment with [vLLM](https://github.com/vllm-project/vllm) installed** — needed only for the LLM evidence tier (`src/run_llm_evidence_batch.py`, `src/llm_evidence.py`), which asks a small open-weight model (`Qwen/Qwen2.5-3B-Instruct` by default) to find evidence for values the deterministic tiers marked "not found." This is deliberately not folded into `environment.yaml`: `llm_evidence.py` defers its `vllm` import so the main environment never needs it installed, and vLLM's CUDA/driver requirements are host-specific enough that pinning them here would be more likely to break than to help. Setting this environment up is left to whoever runs it.

If you only want the notebooks (`make notebooks`), you only need the main environment — the LLM evidence tier is optional and slow (GPU-bound, hours). The notebooks use any compatible local LLM evidence files already present under `data/derived/`, but those files are not included in this repository; a clean checkout must obtain them separately or omit that optional analysis.

## Reproducing

```sh
make env        # create the conda environment at .conda_env
make data       # download + extract the RO-Crate (~38GB)
make derived    # build this project's derived tables from the crate
make notebooks  # execute all three notebooks in place
```

`make help` lists every target. `make derived` runs a check across the full 4.2 million-record crate and can take a long time; `make llm-evidence` prints how to run the optional LLM evidence tier but requires the separate GPU/vLLM environment above, which `make` cannot set up for you.

## Results

See `notebooks/biosample_rocrate_overview.ipynb`, `notebooks/biosample_trace_back_analysis.ipynb`, and `notebooks/biosample_trace_back_evaluation.ipynb` for the data overview, trace-back analysis, and trace-back evaluation.

The generated results are not yet available as a published data release. The notebooks provide an overview of the data, the trace-back pipeline, and the current analysis.
