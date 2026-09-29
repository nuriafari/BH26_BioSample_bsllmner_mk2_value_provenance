#!/bin/bash
# One-command, end-to-end re-run of the production provenance pipeline with the corrected code:
# deterministic cascade (all runs) -> Qwen3-8B, attributes-only, over the ENTIRE unresolved
# residual -> Qwen3-32B-AWQ (attributes, then full-record escalation) over whatever Qwen3-8B
# couldn't ground -> final coverage report. Unattended and resumable: each LLM step appends to its
# own --out file and skips accessions already present in it, so an interrupted run picks back up
# rather than restarting.

set -u
SRC=/workspace/BH26/BH26_BioSample_curation/src
DERIVED=/workspace/BH26/BH26_BioSample_curation/data/derived
TMP=/workspace/BH26/BH26_BioSample_curation/.claude-config/jobs/11655e74/tmp
CONDA=/workspace/BH26/BH26_BioSample_curation/.conda_env/bin
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_DISABLE_XET=1
export CC=$CONDA/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA/x86_64-conda-linux-gnu-g++

cd "$SRC"
MARKER=$TMP/full_pipeline_status.log
echo "$(date -u +%FT%TZ) start" > "$MARKER"

# Step 1: deterministic cascade over all 311 runs -- already current as of this session's earlier
# regeneration, so this just confirms/resumes (chunks already on disk are skipped, not recomputed)
echo "$(date -u +%FT%TZ) deterministic cascade (confirm/resume)" >> "$MARKER"
"$CONDA/python3" build_trace_back_full.py --workers 64 >> "$TMP/full_pipeline_deterministic.log" 2>&1
echo "$(date -u +%FT%TZ) deterministic done (exit $?)" >> "$MARKER"

# Step 2: Qwen3-8B, attributes-only, over the ENTIRE deterministic residual (no --sample-size =
# full residual). --short-only keeps Qwen3-8B's contribution unconditional: everything it doesn't
# ground here is left for Qwen3-32B-AWQ's own attributes/full-record escalation in step 3, rather
# than giving Qwen3-8B a full-record pass whose positives would need a separate cross-check.
QWEN8B_OUT=$DERIVED/llm_evidence_full_qwen3_8b_v2.jsonl
echo "$(date -u +%FT%TZ) launching qwen3-8b (attributes-only) over full residual -> $QWEN8B_OUT" >> "$MARKER"
"$CONDA/python3" run_llm_evidence_batch.py --model Qwen/Qwen3-8B --out "$QWEN8B_OUT" --short-only --batch-size 200 --max-hours 999 \
    >> "$TMP/full_pipeline_qwen3_8b.log" 2>&1
echo "$(date -u +%FT%TZ) qwen3-8b done (exit $?)" >> "$MARKER"

# Step 3: Qwen3-32B-AWQ over whatever Qwen3-8B's own output still marks "not found"
QWEN32B_OUT=$DERIVED/llm_evidence_second_pass_qwen3_32b_v2.jsonl
echo "$(date -u +%FT%TZ) launching qwen3-32b-awq over qwen3-8b's residual -> $QWEN32B_OUT" >> "$MARKER"
"$CONDA/python3" run_llm_evidence_batch.py --model Qwen/Qwen3-32B-AWQ --source-jsonl "$QWEN8B_OUT" \
    --out "$QWEN32B_OUT" --batch-size 200 --max-hours 999 >> "$TMP/full_pipeline_qwen3_32b_awq.log" 2>&1
echo "$(date -u +%FT%TZ) qwen3-32b-awq done (exit $?)" >> "$MARKER"

# Step 4: final coverage report combining deterministic + both LLM passes
echo "$(date -u +%FT%TZ) aggregating final coverage" >> "$MARKER"
"$CONDA/python3" aggregate_final_coverage.py --qwen3-8b "$QWEN8B_OUT" --qwen3-32b "$QWEN32B_OUT" \
    --out "$TMP/final_coverage_report.json" >> "$TMP/full_pipeline_aggregate.log" 2>&1
echo "$(date -u +%FT%TZ) aggregation done (exit $?)" >> "$MARKER"

echo "$(date -u +%FT%TZ) ALL_DONE" >> "$MARKER"
