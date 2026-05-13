#!/bin/bash
#
# Compute pairwise gradient cosine similarity for a run's
# per-prompt gradients (produced by run_gradient.sh).
#
# Usage:
#   bash cal_similarity.sh <run_name> [start_idx] [end_idx]
#
# Configure via env var:
#   GRAD_ANALYSIS_OUTPUT_DIR  root containing per-run subdirs
#                             (default: ./grpo_gradient_analysis_output)

run_name="${1:-qwen_0.5b_demo}"
start_idx="${2:-0}"
end_idx="${3:-100000}"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
python -u "$SCRIPT_DIR/prompt_grad_similarity.py" \
    --run_name  "$run_name" \
    --start_idx "$start_idx" \
    --end_idx   "$end_idx"
