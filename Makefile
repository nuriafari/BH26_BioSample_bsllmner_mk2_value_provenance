# Reproduces the BH26 BioSample curation pipeline. See README.md for context on what this
# project does and why it needs two separate environments.
#
# Real files/directories are used as prerequisites throughout (not phony always-rerun targets),
# so `make notebooks` alone is self-sufficient from a bare clone -- it creates the environment,
# downloads the RO-Crate, and builds the derived tables, but skips any step whose output already
# exists. Re-running `make notebooks` after that only re-executes the notebooks.
#
# Work in progress: `derived` runs a check across the full 4.2M-record RO-Crate and can take a
# long time on a first build; `llm-evidence` needs a separate GPU/vLLM environment this Makefile
# cannot create.

CONDA_ENV := .conda_env
CRATE_URL := https://biosampleplus.s3.ap-northeast-1.amazonaws.com/releases/2026-06_mistral-small3.1-24b.tar.gz
CRATE_TARBALL := $(notdir $(CRATE_URL))
CRATE_DIR := data/2026-06_mistral-small3.1-24b
DERIVED_DIR := data/derived
BIOSAMPLE_METADATA := $(DERIVED_DIR)/biosample_metadata.parquet
TRACE_BACK_FULL := $(DERIVED_DIR)/trace_back_full.parquet
RUN := conda run --live-stream --prefix $(CONDA_ENV)

.PHONY: help env data derived notebooks lint llm-evidence clean

help:
	@echo "Targets:"
	@echo "  env          Create the conda environment ($(CONDA_ENV)) from environment.yaml"
	@echo "  data         Download and extract the RO-Crate into $(CRATE_DIR)"
	@echo "  derived      Build derived parquet tables from the RO-Crate"
	@echo "  notebooks    Execute all three notebooks in place (builds env/data/derived first if needed)"
	@echo "  lint         Run ruff over src/ and notebooks/"
	@echo "  llm-evidence Print how to run the optional GPU/vLLM evidence tier (see README.md)"
	@echo "  clean        Remove derived data and Python caches (keeps the downloaded RO-Crate)"

env: $(CONDA_ENV)

# Re-running with environment.yaml unchanged is a no-op; a changed environment.yaml re-triggers
# this and falls through to `conda env update` since the prefix already exists.
$(CONDA_ENV): environment.yaml
	conda env create -f environment.yaml --prefix $(CONDA_ENV) || \
		conda env update -f environment.yaml --prefix $(CONDA_ENV)

$(CRATE_DIR):
	mkdir -p data
	curl -O $(CRATE_URL)
	tar xzf $(CRATE_TARBALL) -C data
	rm -f $(CRATE_TARBALL)

data: $(CRATE_DIR)

# Order-only prerequisite on the conda env (`|`): the env's own mtime shouldn't force a rebuild
# of tables that are otherwise up to date.
$(BIOSAMPLE_METADATA): $(CRATE_DIR) | $(CONDA_ENV)
	$(RUN) python src/biosample_metadata.py

$(TRACE_BACK_FULL): $(CRATE_DIR) | $(CONDA_ENV)
	$(RUN) python src/build_trace_back_full.py --workers 64

derived: $(BIOSAMPLE_METADATA) $(TRACE_BACK_FULL)

notebooks: derived
	$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/biosample_rocrate_overview.ipynb
	$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/biosample_trace_back_pipeline.ipynb
	$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/biosample_trace_back_analysis.ipynb

lint: | $(CONDA_ENV)
	$(RUN) ruff check src notebooks
	$(RUN) ruff format --check src notebooks

llm-evidence:
	@echo "Requires a separate GPU environment with vLLM installed (see README.md, Environments)."
	@echo "Example (run from that environment):"
	@echo "  CUDA_VISIBLE_DEVICES=0 python src/run_llm_evidence_batch.py --sample-size 8000 --max-hours 4 --seed 0"

clean:
	rm -rf $(DERIVED_DIR)
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
